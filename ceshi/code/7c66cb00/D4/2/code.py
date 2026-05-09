# Variant of 7c66cb00
# Dimension: D4 — 目标对象：水平条结构 (dst)
# Change: 全宽水平条，底部堆叠2-3个，每个条4-8行高，由填充色（内部）和边框色（首尾两列）组成，条间间隔1行背景... → 空心矩形框架替代条

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D4 variant
# ----------------------------------
# spec_variant:
#   unchanged_core:
#     - bg is still the most common color.
#     - top area still contains 2-3 small sprites built from frame fill colors.
#     - sprite outer color still selects the destination object by color match.
#     - recolor rule is unchanged: outer -> destination border, inner -> destination fill.
#     - top sprites are erased in the output.
#   dst_change:
#     - full-width horizontal bars are replaced by 2-3 separated rectangular frames.
#     - each frame is a 2-color rectangle: border on all four sides, uniform fill inside.
#     - frames do not touch and are distributed in the lower part of the canvas instead of
#       being stacked into one bottom strip.
#     - placing into the destination is now 2D: the matched pattern is docked inside the
#       frame interior, bottom-aligned and horizontally centered.
#   concrete_behavior_trace:
#     - original:
#       sprite bbox (say 3x5, outer=green, inner=red) matches the green bar, then is pasted
#       at the same absolute x column and bar-bottom row. Green pixels become the bar border
#       color, red pixels become green.
#     - variant:
#       the same sprite matches the frame whose interior fill is green. The sprite is recolored
#       in the same way, but instead of using the old absolute x inside a full-width bar, it is
#       inserted into the frame interior, flush with the frame's bottom edge and centered across
#       the available interior width.
#     - difference:
#       the target is no longer a 1D stripe with implicit infinite horizontal room; it is a
#       bounded 2D container. The solver must detect rectangles, read both border/fill colors,
#       and compute an interior placement.
#   complexity_calibration:
#     - frames stay 2-3 in number, mirroring the original bar count.
#     - fill colors remain unique so color matching is visible and unambiguous.
#     - the original solver's per-color connected-component quirk is preserved: if a sprite has
#       an inner color block, that inner block is also detected as its own pattern and docks to
#       its own frame. The generator therefore constrains the sample so each frame receives at
#       most one detected pattern, avoiding hidden overlaps inside a frame.
#   implicit_assumption_audit:
#     - original row-scan bar detector assumed rows with >70% one fill color and borders only
#       on the first/last column. That fails here because frame rows are short local rectangles.
#     - original placement reused pat["c0"] because bars spanned the whole width. That is no
#       longer safe; the variant replaces it with a frame-local placement computed from frame
#       interior dimensions.
#     - original "clear everything above first bar" remains safe because the generator keeps all
#       sprites strictly above the highest frame.
#     - original pattern extraction by monochrome regions remains valid because sprites stay
#       isolated and use only frame fill colors.
#   legality_and_generator_guards:
#     - frames are axis-aligned rectangles with at least 1 bg cell gap from each other.
#     - every frame interior is large enough for the pattern assigned to it.
#     - top sprites are pairwise separated by at least 1 bg cell and remain entirely above the
#       highest frame.
#     - the generator validates each sample with main(input) == expected_output.
#   visual_observability:
#     - the destination objects are visibly not bars: they have finite width, full four-sided
#       borders, and occupy multiple row bands.
#     - frames are spread across both x and y so the lower scene reads as separate boxes rather
#       than a stripe stack.
#     - recoloring is still inferable because each frame offers the same fill/border cue as the
#       original task, now wrapped around a 2D container.


GRID_H = 30
GRID_W = 24

FRAME_COUNT_RANGE = (2, 3)
FRAME_H_RANGE = (7, 10)
FRAME_W_RANGE = (7, 11)
MIN_TOP_AREA = 8
MIN_FRAME_SPREAD_R = 3
MIN_FRAME_SPREAD_C = 4

SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 8)
DUAL_SPRITE_PROB = 0.55

GEN_RETRIES = 500
PLACEMENT_RETRIES = 250


def _rect_separated(
    rects: list[tuple[int, int, int, int]],
    r0: int,
    c0: int,
    h: int,
    w: int,
) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _frame_from_region(region: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    h, w = map(int, region.shape)
    if h < 5 or w < 5:
        return None

    border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
    border_counts = Counter(map(int, border.flatten()))
    border_color, border_count = border_counts.most_common(1)[0]
    if int(border_color) == int(bg) or int(border_count) != int(border.size):
        return None

    interior = region[1:-1, 1:-1]
    if interior.size == 0:
        return None
    fill_counts = Counter(map(int, interior.flatten()))
    fill_color, fill_count = fill_counts.most_common(1)[0]
    if (
        int(fill_color) == int(bg)
        or int(fill_color) == int(border_color)
        or int(fill_count) != int(interior.size)
    ):
        return None

    r0, c0, r1, c1 = map(int, bbox)
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "h": h,
        "w": w,
        "fill": int(fill_color),
        "border": int(border_color),
        "inner_h": int(h - 2),
        "inner_w": int(w - 2),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid != int(bg)
    lab, n = label_mask(mask, connectivity=4)
    frames: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        b = bbox_coords(coords)
        region = crop_bbox(grid, b)
        frame = _frame_from_region(region, b, bg)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda f: (int(f["r0"]), int(f["c0"])))
    return frames


def _find_patterns(grid: np.ndarray, bg: int, top_end: int) -> list[dict]:
    patterns: list[dict] = []
    top = grid[:top_end, :]
    for col in sorted(set(map(int, top.flatten())) - {int(bg)}):
        mask = top == int(col)
        lab, n = label_mask(mask, connectivity=4)
        for region_id in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(region_id))
            if coords.size == 0:
                continue
            b = bbox_coords(coords)
            r0, c0, r1, c1 = map(int, b)
            bbox = crop_bbox(grid, (r0, c0, r1, c1))
            colors_in_bbox = Counter(map(int, bbox.flatten()))
            colors_in_bbox.pop(int(bg), None)
            if not colors_in_bbox:
                continue
            outer_color = int(colors_in_bbox.most_common(1)[0][0])
            inner_colors = [int(c) for c in colors_in_bbox.keys() if int(c) != int(outer_color)]
            patterns.append(
                {
                    "outer_color": outer_color,
                    "inner_color": inner_colors[0] if inner_colors else None,
                    "r0": r0,
                    "c0": c0,
                    "h": int(bbox.shape[0]),
                    "w": int(bbox.shape[1]),
                    "content": bbox,
                }
            )
    return patterns


def _recolor_pattern(pattern: dict, frame: dict) -> np.ndarray:
    transformed = np.array(pattern["content"], dtype=int).copy()
    outer_color = int(pattern["outer_color"])
    inner_color = pattern["inner_color"]
    transformed[transformed == int(outer_color)] = int(frame["border"])
    if inner_color is not None:
        transformed[transformed == int(inner_color)] = int(frame["fill"])
    return transformed


def _frame_positions_for_matches(frame: dict, matches: list[dict]) -> list[tuple[dict, int, int]]:
    if not matches:
        return []

    inner_r0 = int(frame["r0"] + 1)
    inner_c0 = int(frame["c0"] + 1)
    inner_h = int(frame["inner_h"])
    inner_w = int(frame["inner_w"])
    ordered = sorted(matches, key=lambda p: (int(p["c0"]), int(p["r0"])))

    if len(ordered) == 1:
        pat = ordered[0]
        return [
            (
                pat,
                int(inner_r0 + inner_h - int(pat["h"])),
                int(inner_c0 + (inner_w - int(pat["w"])) // 2),
            )
        ]

    total_w = sum(int(p["w"]) for p in ordered) + len(ordered) - 1
    if total_w <= inner_w and max(int(p["h"]) for p in ordered) <= inner_h:
        c = int(inner_c0 + (inner_w - total_w) // 2)
        placed = []
        for pat in ordered:
            placed.append((pat, int(inner_r0 + inner_h - int(pat["h"])), int(c)))
            c += int(pat["w"]) + 1
        return placed

    total_h = sum(int(p["h"]) for p in ordered) + len(ordered) - 1
    if total_h <= inner_h and max(int(p["w"]) for p in ordered) <= inner_w:
        r = int(inner_r0 + inner_h - total_h)
        placed = []
        for pat in ordered:
            placed.append((pat, int(r), int(inner_c0 + (inner_w - int(pat["w"])) // 2)))
            r += int(pat["h"]) + 1
        return placed

    pat = ordered[0]
    return [
        (
            pat,
            int(inner_r0 + inner_h - int(pat["h"])),
            int(inner_c0 + (inner_w - int(pat["w"])) // 2),
        )
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock two-tone sprites into color-matched rectangular frames with a color swap.

    Variant rule:
    - Detect large 2-color rectangular frames (uniform border, uniform interior fill).
    - Detect sprite patterns in the top area using the original per-color component logic.
    - Match each detected pattern to the frame whose fill color equals the pattern outer color.
    - Recolor: outer -> frame border, inner -> frame fill.
    - Place the recolored pattern inside the frame interior, bottom-aligned and centered.
    - Erase the original sprites from the top area.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames = _find_frames(g, bg)
    assert frames, "No rectangular frames found"

    top_end = int(min(int(frame["r0"]) for frame in frames))
    patterns = _find_patterns(g, bg, top_end)

    out[:top_end, :] = np.where(out[:top_end, :] != int(bg), int(bg), out[:top_end, :])

    for frame in frames:
        matches = [p for p in patterns if int(p["outer_color"]) == int(frame["fill"])]
        for pat, place_r, place_c in _frame_positions_for_matches(frame, matches):
            transformed = _recolor_pattern(pat, frame)
            paste(out, transformed, int(place_r), int(place_c), transparent=int(bg))

    return out


def _build_base(bg: int, frames: list[dict]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        h = int(frame["h"])
        w = int(frame["w"])
        grid[r0 : r0 + h, c0 : c0 + w] = int(frame["border"])
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = int(frame["fill"])
    return grid


def _make_mono_visual(frame: dict, idx: int) -> dict:
    max_h = min(SPRITE_H_RANGE[1], int(frame["inner_h"]))
    max_w = min(SPRITE_W_RANGE[1], int(frame["inner_w"]))
    h = random.randint(SPRITE_H_RANGE[0], max_h)
    w = random.randint(SPRITE_W_RANGE[0], max_w)
    patch = np.full((h, w), int(frame["fill"]), dtype=int)
    return {
        "patch": patch,
        "h": h,
        "w": w,
        "jobs": [
            {
                "target_idx": idx,
                "outer_color": int(frame["fill"]),
                "inner_color": None,
                "content": patch.copy(),
            }
        ],
    }


def _make_dual_visual(outer_frame: dict, outer_idx: int, inner_frame: dict, inner_idx: int) -> dict | None:
    max_outer_h = min(SPRITE_H_RANGE[1], int(outer_frame["inner_h"]))
    max_outer_w = min(SPRITE_W_RANGE[1], int(outer_frame["inner_w"]))
    if max_outer_h < 3 or max_outer_w < 3:
        return None

    h = random.randint(3, max_outer_h)
    w = random.randint(3, max_outer_w)

    max_inner_h = min(int(inner_frame["inner_h"]), h - 2)
    max_inner_w = min(int(inner_frame["inner_w"]), w - 2)
    valid_inner = [
        (ih, iw)
        for ih in range(1, max_inner_h + 1)
        for iw in range(1, max_inner_w + 1)
        if ih * iw * 2 < h * w
    ]
    if not valid_inner:
        return None

    ih, iw = random.choice(valid_inner)
    ir0 = random.randint(1, h - ih - 1)
    ic0 = (w - iw) // 2

    patch = np.full((h, w), int(outer_frame["fill"]), dtype=int)
    patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_frame["fill"])

    inner_patch = np.full((ih, iw), int(inner_frame["fill"]), dtype=int)
    return {
        "patch": patch,
        "h": h,
        "w": w,
        "jobs": [
            {
                "target_idx": outer_idx,
                "outer_color": int(outer_frame["fill"]),
                "inner_color": int(inner_frame["fill"]),
                "content": patch.copy(),
            },
            {
                "target_idx": inner_idx,
                "outer_color": int(inner_frame["fill"]),
                "inner_color": None,
                "content": inner_patch,
            },
        ],
    }


def _place_frames(frame_dims: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    for _ in range(PLACEMENT_RETRIES):
        order = list(range(len(frame_dims)))
        random.shuffle(order)
        rects: list[tuple[int, int, int, int]] = []
        placed: dict[int, tuple[int, int]] = {}

        for idx in order:
            h, w = frame_dims[idx]
            candidates = []
            for r0 in range(MIN_TOP_AREA, GRID_H - h + 1):
                for c0 in range(0, GRID_W - w + 1):
                    if _rect_separated(rects, r0, c0, h, w):
                        candidates.append((r0, c0))
            if not candidates:
                break
            choice = random.choice(candidates)
            placed[idx] = choice
            rects.append((choice[0], choice[1], choice[0] + h - 1, choice[1] + w - 1))

        if len(placed) != len(frame_dims):
            continue

        positions = [placed[i] for i in range(len(frame_dims))]
        rows = [p[0] for p in positions]
        cols = [p[1] for p in positions]
        if max(rows) - min(rows) < MIN_FRAME_SPREAD_R:
            continue
        if max(cols) - min(cols) < MIN_FRAME_SPREAD_C:
            continue
        return positions
    return None


def _build_frames(bg: int, fills: list[int], borders: list[int]) -> list[dict] | None:
    frame_dims = [
        (
            random.randint(*FRAME_H_RANGE),
            random.randint(*FRAME_W_RANGE),
        )
        for _ in fills
    ]
    positions = _place_frames(frame_dims)
    if positions is None:
        return None

    frames = []
    for idx, (fill, border, dims, pos) in enumerate(zip(fills, borders, frame_dims, positions)):
        h, w = dims
        r0, c0 = pos
        frames.append(
            {
                "idx": idx,
                "fill": int(fill),
                "border": int(border),
                "h": int(h),
                "w": int(w),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + h - 1),
                "c1": int(c0 + w - 1),
                "inner_h": int(h - 2),
                "inner_w": int(w - 2),
            }
        )
    return frames


def _make_visuals(frames: list[dict]) -> list[dict] | None:
    n_frames = len(frames)
    if n_frames == 2:
        visuals = [_make_mono_visual(frame, int(frame["idx"])) for frame in frames]
        random.shuffle(visuals)
        return visuals

    if random.random() < DUAL_SPRITE_PROB:
        indices = list(range(n_frames))
        random.shuffle(indices)
        outer_idx, inner_idx, mono_idx = indices
        dual = _make_dual_visual(frames[outer_idx], outer_idx, frames[inner_idx], inner_idx)
        if dual is None:
            return None
        visuals = [dual, _make_mono_visual(frames[mono_idx], mono_idx)]
        random.shuffle(visuals)
        return visuals

    visuals = [_make_mono_visual(frame, int(frame["idx"])) for frame in frames]
    random.shuffle(visuals)
    return visuals


def _place_visuals(visuals: list[dict], top_h: int) -> list[dict] | None:
    rects: list[tuple[int, int, int, int]] = []
    placed: list[dict | None] = [None] * len(visuals)
    order = sorted(range(len(visuals)), key=lambda i: (-int(visuals[i]["w"]), -int(visuals[i]["h"])))

    for idx in order:
        visual = visuals[idx]
        h = int(visual["h"])
        w = int(visual["w"])
        if h > top_h:
            return None

        candidates = []
        for r0 in range(0, top_h - h + 1):
            for c0 in range(0, GRID_W - w + 1):
                if _rect_separated(rects, r0, c0, h, w):
                    candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        rects.append((r0, c0, r0 + h - 1, c0 + w - 1))
        placed[idx] = {**visual, "r0": int(r0), "c0": int(c0)}

    return [p for p in placed if p is not None]


def _render(base: np.ndarray, visuals: list[dict], frames: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()

    for visual in visuals:
        paste(inp, np.array(visual["patch"], dtype=int), int(visual["r0"]), int(visual["c0"]), transparent=int(bg))

    jobs_by_frame: dict[int, list[dict]] = {}
    for visual in visuals:
        for job in visual["jobs"]:
            jobs_by_frame.setdefault(int(job["target_idx"]), []).append(job)

    for frame in frames:
        idx = int(frame["idx"])
        jobs = jobs_by_frame.get(idx, [])
        patterns = []
        for job in jobs:
            content = np.array(job["content"], dtype=int).copy()
            patterns.append(
                {
                    "outer_color": int(job["outer_color"]),
                    "inner_color": job["inner_color"],
                    "h": int(content.shape[0]),
                    "w": int(content.shape[1]),
                    "c0": 0,
                    "r0": 0,
                    "content": content,
                }
            )

        for pat, place_r, place_c in _frame_positions_for_matches(frame, patterns):
            transformed = _recolor_pattern(pat, frame)
            paste(out, transformed, int(place_r), int(place_c), transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_frames]]
    borders = [int(v) for v in palette[1 + n_frames : 1 + 2 * n_frames]]

    frames = _build_frames(bg, fills, borders)
    if frames is None:
        return None

    top_h = min(int(frame["r0"]) for frame in frames)
    if top_h < MIN_TOP_AREA:
        return None

    visuals = _make_visuals(frames)
    if visuals is None:
        return None

    placed_visuals = _place_visuals(visuals, top_h)
    if placed_visuals is None:
        return None

    base = _build_base(bg, frames)
    inp, out = _render(base, placed_visuals, frames, bg)

    if int(get_bg(inp)) != int(bg):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
