# Combo variant of 7c66cb00 (index 12)
# combo_spec: D9/1+D12/2+D15/2
# Name: 大量精灵分配到多条并每条独立子网格输出

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Input remains a single global scene, but it is enlarged to 30x30 and now has 4-5
#   full-width horizontal bars stacked in the lower half, with many more sprites than
#   bars in the top strip.
# - Matching still uses the original color cue: a sprite/pattern belongs to the bar
#   whose fill color equals that pattern's outer color.
# - D9 changes the per-bar placement logic: once multiple patterns map to one bar, they
#   no longer keep their original x positions. They are ordered by visible left-to-right
#   position in the input and packed into the bar from the left, with one fill-colored
#   gap column between neighbors.
# - D15 changes the output canvas itself: instead of editing the original 30x30 scene in
#   place, each bar is rendered as its own cropped mini-strip containing only that bar
#   and the patterns docked into it. The strips are then concatenated horizontally with
#   one full-height background divider column.
# - Concrete trace:
#   - Suppose the second bar has fill=3, border=7, height=3, and three matching patterns
#     of widths 4, 2, and 3 found at input x positions 2, 11, and 18.
#   - Original task: each one would drop into the same global bar at columns 2, 11, 18.
#   - This combo: they become one local frame of width `2 + 4 + 1 + 2 + 1 + 3 = 13`,
#     are recolored, bottom-aligned inside that 3-row strip, and pasted at local columns
#     1, 6, and 9. The full original scene disappears from the output.
#
# b) Complexity calibration
# - The bar count is increased to 4-5 and the sprite count to 6-8, with every bar used
#   by at least one outer sprite and at least one bar receiving multiple patterns.
# - Output strips are intentionally cropped to their packed width so the frame-splitting
#   behavior is visually undeniable; leaving 30-column-wide bars would make D15 much less
#   apparent.
# - Sprite widths stay in 2..4 and bar heights in 3..4 so that crowded bars remain
#   readable instead of turning into a noisy ribbon of tiny fragments.
#
# c) Implicit assumptions audited against the original main()
# - Original bar detection assumed wide horizontal rows; that still holds, so the row
#   scan can stay with only minor threshold tuning for 30 columns.
# - Top objects should be treated as whole sprites, not as per-color fragments. The
#   solver therefore extracts connected non-background components first, then reads the
#   outer/inner colors from each sprite's bbox.
# - Original `place_c = pat["c0"]` is invalid after D9 and D15. Packed local columns are
#   recomputed from per-bar ordering, and global x no longer survives into the output.
# - Original "output size equals input size" is invalid. The solver now has to build a
#   new canvas from per-bar frames instead of mutating a copy of the input.
#
# d) Placement legality
# - Generator guarantees one empty background row between bars and at least one background
#   cell between top sprites, so component extraction is unambiguous.
# - All bars in one sample share the same height, which makes the concatenated output
#   strips naturally row-compatible without ad hoc stretching.
# - Every outer sprite height is at most its target bar height; inner regions also fit
#   because all bars in one sample use the same height.
# - The generator measures the exact patterns the solver will detect and rejects samples
#   whose packed strip for any bar would exceed a readability cap.
#
# e) Visual observability
# - A human can see two linked changes immediately:
#   1. many top sprites collapse into fewer destination groups by shared outer color
#   2. the output is no longer a full scene but a left-to-right gallery of bar strips
# - Ordering is based only on visible geometry (input left-to-right position), never on
#   numeric color ids or hidden generation metadata.
#
# f) Visual cleanliness
# - Each output frame contains exactly one bar strip, so there is no interference between
#   different bars after transformation.
# - The packed strips use only one-dimensional horizontal layout with a fixed 1-column
#   gap rule, which keeps crowded cases legible even when several full sprites share a
#   bar.
#
# g) Change inferability
# - Color semantics stay close to the original task:
#   - outer color identifies the target bar by matching its fill
#   - outer pixels recolor to that bar's border
#   - inner pixels recolor to that bar's fill
# - The new part is geometric, not symbolic: group by bar, pack locally, then emit one
#   cropped frame per bar.
#
# h) Anti-degeneracy target
# - A valid generated sample must show both core fused behaviors:
#   - at least one crowded bar (`>= 2` detected patterns in one strip)
#   - positive crop gain (`sum(frame_widths) < n_bars * 30`), proving the output is a
#     concatenation of cropped subgrids rather than the original full-width bars
#
# i) New solver assumptions and generator guarantees
# - Assumption: all movable patterns lie strictly above the first bar. `_build_bars()`
#   leaves a dedicated top strip and `_place_top_sprites()` uses only that region.
# - Assumption: each detected bar has a unique fill color. `_choose_colors()` enforces it.
# - Assumption: per-bar pattern order is recoverable from visible x positions. Top sprites
#   are separated, so the solver's `(c0, r0, w, h)` ordering is stable and unambiguous.


GRID_H = 30
GRID_W = 30
BAR_COUNT_RANGE = (4, 5)
SPRITE_COUNT_RANGE = (6, 8)
BAR_GAP = 1
FRAME_GAP = 1
SPRITE_H_RANGE = (2, 3)
SPRITE_W_RANGE = (2, 4)
INNER_COLOR_PROB = 0.35
MAX_FRAME_W = 18
GEN_RETRIES = 800


def _sample_bar_height(n_bars: int) -> int:
    if n_bars == 5:
        return 3
    return random.choice([3, 4])


def _find_bars(g: np.ndarray, bg: int) -> list[dict]:
    bars: list[dict] = []
    current: dict | None = None
    h, w = map(int, g.shape)

    for r in range(h):
        row = g[r, :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if not non_bg:
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        fill, count = Counter(non_bg).most_common(1)[0]
        if int(count) <= int(w * 0.75):
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
                current = None
            continue

        border = int(row[0]) if int(row[0]) != int(fill) else int(row[-1])
        if current is not None and int(current["fill"]) == int(fill):
            current["r_max"] = int(r)
        else:
            if current is not None:
                current["h"] = int(current["r_max"] - current["r_min"] + 1)
                bars.append(current)
            current = {
                "fill": int(fill),
                "border": int(border),
                "r_min": int(r),
                "r_max": int(r),
            }

    if current is not None:
        current["h"] = int(current["r_max"] - current["r_min"] + 1)
        bars.append(current)

    return bars


def _find_sprites(g: np.ndarray, bg: int, top_h: int) -> list[dict]:
    sprites: list[dict] = []
    top = g[:top_h, :]
    mask = top != int(bg)
    lab, n = label_mask(mask, connectivity=4)

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue

        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(g, (r0, c0, r1, c1))

        color_counts = Counter(int(v) for v in patch.flatten())
        color_counts.pop(int(bg), None)
        if not color_counts:
            continue

        ordered_colors = [int(color) for color, _ in color_counts.most_common()]
        outer_color = int(ordered_colors[0])
        inner_color = int(ordered_colors[1]) if len(ordered_colors) >= 2 else None
        sprites.append(
            {
                "outer_color": outer_color,
                "inner_color": inner_color,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "content": patch,
            }
        )

    return sprites


def _ordered_matching_sprites(sprites: list[dict], fill_color: int) -> list[dict]:
    return sorted(
        [sprite for sprite in sprites if int(sprite["outer_color"]) == int(fill_color)],
        key=lambda sprite: (int(sprite["c0"]), int(sprite["r0"]), int(sprite["w"]), int(sprite["h"])),
    )


def _frame_layout(matching: list[dict]) -> tuple[list[tuple[dict, int]], int]:
    if not matching:
        return [], 3

    placements: list[tuple[dict, int]] = []
    next_c = 1
    for pat in matching:
        placements.append((pat, int(next_c)))
        next_c += int(pat["w"]) + 1
    return placements, int(next_c)


def _recolor_sprite(sprite: dict, bar: dict) -> np.ndarray:
    src = np.array(sprite["content"], dtype=int)
    transformed = src.copy()
    transformed[src == int(sprite["outer_color"])] = int(bar["border"])
    if sprite["inner_color"] is not None:
        transformed[src == int(sprite["inner_color"])] = int(bar["fill"])
    return transformed


def _build_bar_frame(bar: dict, matching: list[dict], bg: int) -> np.ndarray:
    placements, frame_w = _frame_layout(matching)
    frame = np.full((int(bar["h"]), int(frame_w)), int(bar["fill"]), dtype=int)
    frame[:, 0] = int(bar["border"])
    frame[:, -1] = int(bar["border"])

    for sprite, place_c in placements:
        transformed = _recolor_sprite(sprite, bar)
        place_r = int(bar["h"] - sprite["h"])
        paste(frame, transformed, place_r, int(place_c), transparent=int(bg))

    return frame


def _assemble_frames(frames: list[np.ndarray], bg: int) -> np.ndarray:
    out_h = max(int(frame.shape[0]) for frame in frames)
    out_w = sum(int(frame.shape[1]) for frame in frames) + FRAME_GAP * (len(frames) - 1)
    out = np.full((out_h, out_w), int(bg), dtype=int)

    c = 0
    for i, frame in enumerate(frames):
        r = int(out_h - frame.shape[0])
        paste(out, frame, r, c)
        c += int(frame.shape[1])
        if i + 1 < len(frames):
            c += FRAME_GAP
    return out


def _solve_core(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    top_h = min(int(bar["r_min"]) for bar in bars)
    sprites = _find_sprites(g, bg, top_h)

    frames = []
    for bar in bars:
        matching = _ordered_matching_sprites(sprites, int(bar["fill"]))
        frames.append(_build_bar_frame(bar, matching, bg))
    return _assemble_frames(frames, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign many top sprites into color-matched bars, then emit one cropped strip per bar.

    Input structure:
    - The canvas is 30x30 with background = most common color.
    - The lower zone contains 4-5 full-width horizontal bars, each with a fill color and
      a border color on the left and right edges, separated by one background row.
    - The upper zone contains 6-8 detached sprites. Multiple sprites may share the same
      outer color, so several of them can target the same bar.

    Rule:
    - Detect each top sprite as one connected non-background component and read its outer
      and inner colors from the cropped bbox.
    - Match every detected sprite to the bar whose fill equals that sprite's outer color.
    - Recolor each matched sprite with the original swap rule (outer -> bar border,
      inner -> bar fill), bottom-align it inside the target bar, and pack all sprites for
      that bar from left to right with a 1-column fill gap.
    - Instead of writing back into the original 30x30 scene, crop each bar to the minimum
      strip width needed by its packed patterns and concatenate those strips horizontally
      with a 1-column background divider.

    Key insight: the important relation is no longer "where in the big scene does this
    sprite land?" but "which bar-group does it belong to?" The output is a bar-by-bar
    gallery of locally packed results, not an edited copy of the input canvas.
    """
    return _solve_core(np.asarray(grid, dtype=int))


def _choose_colors(n_bars: int) -> tuple[int, list[int], list[int]]:
    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    colors = [int(v) for v in palette[1:]]
    fills = colors[:n_bars]
    remaining = colors[n_bars:]

    borders: list[int] = []
    used_borders: set[int] = set()
    for i in range(n_bars):
        choices = [c for c in remaining if int(c) != int(fills[i]) and int(c) not in used_borders]
        if not choices:
            choices = [c for c in fills if int(c) != int(fills[i]) and int(c) not in used_borders]
        choice = int(random.choice(choices))
        borders.append(choice)
        used_borders.add(choice)
        if choice in remaining:
            remaining.remove(choice)

    return bg, fills, borders


def _build_bars(bg: int, fills: list[int], borders: list[int], bar_h: int) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    bars: list[dict] = [None] * len(fills)  # type: ignore[assignment]

    r = GRID_H
    for i in range(len(fills) - 1, -1, -1):
        r0 = int(r - bar_h)
        r1 = int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(bar_h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = int(r0 - BAR_GAP)

    return grid, bars


def _sample_outer_counts(n_bars: int, n_sprites: int) -> list[int]:
    while True:
        counts = [1] * n_bars
        for _ in range(n_sprites - n_bars):
            counts[random.randrange(n_bars)] += 1
        if max(counts) < 2:
            continue
        if n_sprites - n_bars >= 2 and sum(1 for count in counts if count >= 2) < 2:
            continue
        return counts


def _make_sprite(outer_idx: int, bars: list[dict], force_inner: bool = False) -> dict:
    outer_fill = int(bars[outer_idx]["fill"])
    min_h = 3 if force_inner else SPRITE_H_RANGE[0]
    min_w = 3 if force_inner else SPRITE_W_RANGE[0]
    h = random.randint(min_h, min(SPRITE_H_RANGE[1], int(bars[outer_idx]["h"])))
    w = random.randint(min_w, SPRITE_W_RANGE[1])
    patch = np.full((h, w), outer_fill, dtype=int)

    inner_idx = None
    inner_bbox = None
    if len(bars) >= 2 and h >= 3 and w >= 3 and (force_inner or random.random() < INNER_COLOR_PROB):
        inner_idx = random.choice([i for i in range(len(bars)) if i != outer_idx])
        valid_inner = [
            (ih, iw)
            for ih in range(1, h - 1)
            for iw in range(1, w - 1)
            if ih * iw * 2 < h * w
        ]
        if valid_inner:
            ih, iw = random.choice(valid_inner)
            ir0 = random.randint(1, h - ih - 1)
            ic0 = random.randint(1, w - iw - 1)
            patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(bars[inner_idx]["fill"])
            inner_bbox = (int(ir0), int(ic0), int(ih), int(iw))
        else:
            inner_idx = None

    return {
        "outer_idx": int(outer_idx),
        "inner_idx": inner_idx,
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "inner_bbox": inner_bbox,
    }


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_top_sprites(sprites: list[dict], top_h: int) -> list[dict] | None:
    rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    ordered = sorted(sprites, key=lambda sp: (-int(sp["h"] * sp["w"]), -int(sp["w"]), -int(sp["h"])))
    for sp in ordered:
        h = int(sp["h"])
        w = int(sp["w"])
        candidates: list[tuple[int, int]] = []
        rows = list(range(0, top_h - h + 1))
        cols = list(range(1, GRID_W - w))
        random.shuffle(rows)
        random.shuffle(cols)
        for r0 in rows:
            for c0 in cols:
                if _rect_separated(rects, r0, c0, h, w):
                    candidates.append((int(r0), int(c0)))
            if candidates:
                break

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        placed.append({**sp, "r0": int(r0), "c0": int(c0)})
        rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render_input(base: np.ndarray, placed: list[dict], bg: int) -> np.ndarray:
    inp = np.array(base, dtype=int).copy()
    for sp in placed:
        paste(inp, np.array(sp["patch"], dtype=int), int(sp["r0"]), int(sp["c0"]), transparent=int(bg))
    return inp


def _fusion_metrics(inp: np.ndarray) -> dict:
    bg = int(get_bg(inp))
    bars = _find_bars(inp, bg)
    top_h = min(int(bar["r_min"]) for bar in bars)
    sprites = _find_sprites(inp, bg, top_h)

    frame_widths: list[int] = []
    crowded_bars = 0
    shifted_sprites = 0
    bars_used = 0
    zero_fill_frames = 0

    for bar in bars:
        matching = _ordered_matching_sprites(sprites, int(bar["fill"]))
        placements, frame_w = _frame_layout(matching)
        frame_widths.append(int(frame_w))
        if matching:
            bars_used += 1
        if len(matching) >= 2:
            crowded_bars += 1
        for sprite, place_c in placements:
            if int(place_c) != int(sprite["c0"]):
                shifted_sprites += 1
        if matching:
            frame = _build_bar_frame(bar, matching, bg)
            if int((frame == int(bar["fill"])).sum()) == 0:
                zero_fill_frames += 1

    output = _solve_core(inp)
    total_crop_gain = len(bars) * GRID_W - sum(frame_widths)
    return {
        "bar_count": len(bars),
        "sprite_count": len(sprites),
        "two_tone_sprites": sum(1 for sprite in sprites if sprite["inner_color"] is not None),
        "bars_used": int(bars_used),
        "crowded_bars": int(crowded_bars),
        "shifted_sprites": int(shifted_sprites),
        "zero_fill_frames": int(zero_fill_frames),
        "frame_widths": tuple(int(v) for v in frame_widths),
        "max_frame_width": max(frame_widths),
        "total_crop_gain": int(total_crop_gain),
        "output_shape": tuple(int(v) for v in output.shape),
    }


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)
    bar_h = _sample_bar_height(n_bars)
    bg, fills, borders = _choose_colors(n_bars)
    base, bars = _build_bars(bg, fills, borders, bar_h)

    top_h = int(bars[0]["r_min"])
    if top_h < 8:
        return None

    outer_counts = _sample_outer_counts(n_bars, n_sprites)
    outer_indices = [idx for idx, count in enumerate(outer_counts) for _ in range(count)]
    random.shuffle(outer_indices)
    two_tone_target = min(n_sprites, random.randint(2, 3))
    forced_inner = set(random.sample(range(n_sprites), k=two_tone_target))
    sprites = [_make_sprite(idx, bars, force_inner=i in forced_inner) for i, idx in enumerate(outer_indices)]

    placed = _place_top_sprites(sprites, top_h)
    if placed is None:
        return None

    inp = _render_input(base, placed, bg)
    if int(get_bg(inp)) != int(bg):
        return None

    try:
        metrics = _fusion_metrics(inp)
    except Exception:
        return None

    if metrics["bars_used"] != n_bars:
        return None
    if metrics["crowded_bars"] < 1:
        return None
    if metrics["shifted_sprites"] < 2:
        return None
    if metrics["two_tone_sprites"] < 2:
        return None
    if metrics["zero_fill_frames"] != 0:
        return None
    if metrics["total_crop_gain"] <= n_bars * 10:
        return None
    if metrics["max_frame_width"] > MAX_FRAME_W:
        return None

    try:
        _ = _solve_core(inp)
    except Exception:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
