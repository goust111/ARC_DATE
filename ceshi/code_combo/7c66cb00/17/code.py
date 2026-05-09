# Combo variant of 7c66cb00 (index 17)
# combo_spec: D1/4+D4/2+D17/2
# Name: 连接器匹配图例映射空心框架

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#   - Original task:
#     small top sprites are matched to bottom bars by outer color, then docked into the
#     matched bar with a fixed local recolor rule.
#   - This combo:
#     the destination is no longer a bar but a separated hollow frame; a small legend in
#     the top-left says how source colors transform; and sprites no longer choose their
#     destination by color. Instead each sprite sits above one thin vertical connector line,
#     and the connector points to the frame it should enter.
#   - Concrete trace:
#     suppose the legend rows are [green -> red], [yellow -> blue], [pink -> grey].
#     A sprite with outer=green and inner=pink may be positioned above the yellow-filled
#     frame, not the green-filled frame. Its connector line points straight down to that
#     yellow frame. In the output the legend disappears, the connector disappears, the sprite
#     slides vertically into that frame interior, and its colors become red / grey by legend
#     lookup rather than frame-local swap.
#   - Difference:
#     color no longer determines target, target geometry is now a 2D container, and recolor
#     is explicit from a symbolic legend rather than implicit from the matched destination.
#
# b) complexity_calibration:
#   - There are only 2-3 frames, 2-3 legend rows, and exactly one sprite per frame.
#   - Sprite outer colors are a derangement of frame fill colors, so connector matching is
#     visibly necessary and the task does not collapse back to the original color-match rule.
#   - Legend outputs are a rotated permutation of frame border colors, so recoloring differs
#     from the old outer->own-border rule in every sample.
#   - At least one sprite gets an inner region when possible, so the legend acts on multiple
#     colors and not only on a single dominant color.
#
# c) implicit_assumption_audit:
#   - Original row-scan bar detection is invalid because the lower objects are separated local
#     rectangles, not full-width stripes.
#   - Original per-color component extraction is unsafe because it would split one sprite into
#     several movable pieces and would also treat the legend as movable content. The combo
#     therefore detects whole top-area objects after masking the legend, and it treats thin
#     1-column components as connectors rather than sprites.
#   - Original placement by absolute x inside a full-width bar only worked because the target
#     spanned the whole grid. Here placement is allowed only because the generator constrains
#     each sprite's x-range to already lie inside its target frame interior, so a pure vertical
#     slide along the connector is legal.
#   - Original recolor depended on the matched destination. That assumption is removed; every
#     sprite pixel is recolored by a legend lookup from the original patch.
#
# d) placement_legality:
#   - Frames share one top row and are separated horizontally, so every connector can be a
#     vertical non-overlapping line from sprite to frame.
#   - Each sprite width/height is chosen to fit its target frame interior, and its input x
#     location is restricted to that interior span.
#   - The legend occupies only the top-left 2 columns for 2-3 rows and sprite placement avoids
#     that reserved rectangle.
#
# e) visual_observability:
#   - The lower scene visibly consists of hollow boxes instead of horizontal bars.
#   - Sprites often sit above a frame whose fill color does not match the sprite, making the
#     connector visibly responsible for target selection.
#   - The legend is small and symbolic, and because it disappears in the output the examples
#     directly show that the top-left key controls the color change.
#
# f) visual_cleanliness:
#   - Connectors are detached one-column lines, so they do not merge with sprites or frames.
#   - Frames are aligned in one lower band with 1+ cell gaps, which keeps the scene readable.
#   - The top area contains only legend, sprites, and connectors; the output clears that whole
#     band, leaving a clean destination-only result.
#
# g) change_inferability:
#   - Legend inputs are exactly the frame fill colors, one row each.
#   - Legend outputs are exactly a non-identity permutation of frame border colors.
#   - Every frame receives one sprite and every fill color is used by a sprite, so all legend
#     rows and all connector matches are exercised in each sample.


GRID_H = 30
GRID_W = 24

FRAME_COUNT_RANGE = (2, 3)
FRAME_H_RANGE = (7, 9)
FRAME_W_RANGE = (5, 6)
FRAME_TOP_RANGE = (14, 18)
FRAME_GAP_RANGE = (1, 2)
MIN_LEFT_MARGIN = 3
MIN_RIGHT_MARGIN = 1

LEGEND_COLS = 2
CONNECTOR_MIN_LEN = 2

SPRITE_H_RANGE = (2, 5)
SPRITE_W_RANGE = (2, 4)
INNER_COLOR_PROB = 0.65

GEN_RETRIES = 600
PLACEMENT_RETRIES = 300


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

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    border = region[border_mask]
    border_color, _border_count = Counter(map(int, border)).most_common(1)[0]
    if int(border_color) == int(bg):
        return None

    marker_color = None
    marker_c = None
    border_outliers = np.argwhere(border_mask & (region != int(border_color)))
    if border_outliers.size:
        if int(border_outliers.shape[0]) != 1:
            return None
        marker_r, marker_local_c = map(int, border_outliers[0])
        if int(marker_r) != 0 or int(marker_local_c) in (0, int(w - 1)):
            return None
        marker_color = int(region[int(marker_r), int(marker_local_c)])
        if int(marker_color) == int(bg):
            return None

    inner = region[1:-1, 1:-1]
    if inner.size == 0:
        return None
    fill_color, fill_count = Counter(map(int, inner.flatten())).most_common(1)[0]
    if (
        int(fill_color) == int(bg)
        or int(fill_color) == int(border_color)
        or int(fill_count) != int(inner.size)
    ):
        return None

    r0, c0, r1, c1 = map(int, bbox)
    if marker_color is not None:
        if int(marker_color) == int(fill_color):
            return None
        marker_c = int(c0 + marker_local_c)

    return {
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "h": int(h),
        "w": int(w),
        "fill": int(fill_color),
        "border": int(border_color),
        "inner_h": int(h - 2),
        "inner_w": int(w - 2),
        "marker_c": marker_c,
        "marker_color": marker_color,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    mask = grid != int(bg)
    lab, n = label_mask(mask, connectivity=4)
    frames: list[dict] = []
    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        bbox = bbox_coords(coords)
        region = crop_bbox(grid, bbox)
        r0, c0, _r1, c1 = map(int, bbox)
        for trim in range(0, int(region.shape[0]) - 4):
            subregion = region[int(trim) :, :]
            sub_bbox = (int(r0 + trim), int(c0), int(r0 + region.shape[0] - 1), int(c1))
            frame = _frame_from_region(subregion, sub_bbox, bg)
            if frame is not None:
                frames.append(frame)
                break
    frames.sort(key=lambda f: int(f["c0"]))
    return frames


def _legend_mask(shape: tuple[int, int], n_rows: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[: int(n_rows), :LEGEND_COLS] = True
    return mask


def _read_legend(grid: np.ndarray, n_rows: int, bg: int) -> dict[int, int]:
    color_map: dict[int, int] = {}
    for rr in range(int(n_rows)):
        src = int(grid[int(rr), 0])
        dst = int(grid[int(rr), 1])
        if src != int(bg) and dst != int(bg):
            color_map[int(src)] = int(dst)
    return color_map


def _apply_color_map(patch: np.ndarray, color_map: dict[int, int], bg: int) -> np.ndarray:
    out = np.array(patch, dtype=int).copy()
    original = np.array(patch, dtype=int)
    for src, dst in color_map.items():
        out[original == int(src)] = int(dst)
    out[original == int(bg)] = int(bg)
    return out


def _extract_top_objects(
    grid: np.ndarray,
    bg: int,
    top_end: int,
    legend_mask: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    top = np.array(grid[:top_end, :], dtype=int).copy()
    top[legend_mask[:top_end, :]] = int(bg)

    lab, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []
    connectors: list[dict] = []

    for region_id in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(region_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        patch = crop_bbox(top, (r0, c0, r1, c1))
        colors = [int(v) for v in patch.flatten() if int(v) != int(bg)]
        if not colors:
            continue

        if int(c1) == int(c0) and len(set(colors)) == 1 and int(r1 - r0 + 1) >= CONNECTOR_MIN_LEN:
            connectors.append(
                {
                    "r0": int(r0),
                    "r1": int(r1),
                    "c": int(c0),
                    "color": int(colors[0]),
                }
            )
            continue

        color_counts = Counter(colors)
        dominant = int(color_counts.most_common(1)[0][0])
        sprites.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "h": int(patch.shape[0]),
                "w": int(patch.shape[1]),
                "dominant": int(dominant),
                "content": patch,
            }
        )

    sprites.sort(key=lambda s: (int(s["r0"]), int(s["c0"])))
    connectors.sort(key=lambda c: (int(c["c"]), int(c["r0"])))
    return sprites, connectors


def _match_sprite_to_connector(sprite: dict, connectors: list[dict]) -> dict:
    candidates = [
        conn
        for conn in connectors
        if int(sprite["c0"]) <= int(conn["c"]) <= int(sprite["c1"]) and int(conn["r0"]) > int(sprite["r1"])
    ]
    assert candidates, "Every sprite must have a connector directly below it"
    candidates.sort(key=lambda conn: int(conn["r0"]) - int(sprite["r1"]))
    return candidates[0]


def _match_connector_to_frame(connector: dict, frames: list[dict]) -> dict:
    candidates = [
        frame
        for frame in frames
        if frame["marker_c"] is not None
        and int(frame["marker_c"]) == int(connector["c"])
        and int(frame["marker_color"]) == int(connector["color"])
    ]
    assert len(candidates) == 1, "Connector must match exactly one frame marker by color and column"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide top sprites down connector lines into hollow frames, recoloring by a legend.

    Input structure:
    - The lower half contains 2-3 separated hollow rectangular frames with uniform border
      color and uniform interior fill color.
    - The top-left corner contains a tiny 2-column legend: each row is [source_color, output_color].
    - Above the frames sit 2-3 small sprites, each with a thin 1-column connector line hanging
      below it and pointing to one frame.

    Rule:
    - Read the legend as a global color map and then ignore it in the output.
    - Detect each sprite and the connector directly beneath it.
    - Use the connector's color and its aligned top-border marker to choose the frame.
    - Move the sprite straight down into that frame, bottom-aligned inside the frame interior.
    - Recolor every non-background sprite pixel by legend lookup while moving it.
    - Erase the legend, the original sprites, and the connectors from the top area; restore the
      frame border where the input used a colored connector marker.

    Key insight:
    - Matching is no longer color-based: sprite colors come from the frame-fill palette but often
      point to a different frame. The connector gives the destination, while the legend supplies
      the color transformation.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames = _find_frames(g, bg)
    assert frames, "No hollow frames found"
    top_end = min(int(frame["r0"]) for frame in frames)
    n_frames = len(frames)

    legend_mask = _legend_mask(g.shape, n_frames)
    color_map = _read_legend(g, n_frames, bg)
    sprites, connectors = _extract_top_objects(g, bg, top_end, legend_mask)

    out[:top_end, :] = int(bg)
    for frame in frames:
        if frame["marker_c"] is not None:
            out[int(frame["r0"]), int(frame["marker_c"])] = int(frame["border"])

    used_connector_cols: set[int] = set()
    for sprite in sprites:
        connector = _match_sprite_to_connector(sprite, connectors)
        used_connector_cols.add(int(connector["c"]))
        frame = _match_connector_to_frame(connector, frames)

        place_r = int(frame["r1"]) - int(sprite["h"])
        place_c = int(sprite["c0"])
        assert int(frame["c0"]) + 1 <= int(place_c)
        assert int(place_c + sprite["w"] - 1) <= int(frame["c1"]) - 1
        assert int(frame["r0"]) + 1 <= int(place_r)

        transformed = _apply_color_map(np.array(sprite["content"], dtype=int), color_map, bg)
        paste(out, transformed, int(place_r), int(place_c), transparent=int(bg))

    assert len(used_connector_cols) == len(sprites)
    return out


def _build_frames(bg: int, fills: list[int], borders: list[int]) -> list[dict] | None:
    n_frames = len(fills)
    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(*FRAME_TOP_RANGE)
        dims = [
            (
                random.randint(*FRAME_H_RANGE),
                random.randint(*FRAME_W_RANGE),
            )
            for _ in range(n_frames)
        ]
        if top + max(int(h) for h, _w in dims) > GRID_H:
            continue

        gaps = [random.randint(*FRAME_GAP_RANGE) for _ in range(max(0, n_frames - 1))]
        total_w = sum(int(w) for _h, w in dims) + sum(gaps)
        free_w = GRID_W - MIN_LEFT_MARGIN - MIN_RIGHT_MARGIN - total_w
        if free_w < 0:
            continue

        left = random.randint(MIN_LEFT_MARGIN, MIN_LEFT_MARGIN + free_w)
        frames = []
        c = left
        for idx, (fill, border, (h, w)) in enumerate(zip(fills, borders, dims)):
            frames.append(
                {
                    "idx": int(idx),
                    "fill": int(fill),
                    "border": int(border),
                    "r0": int(top),
                    "c0": int(c),
                    "r1": int(top + h - 1),
                    "c1": int(c + w - 1),
                    "h": int(h),
                    "w": int(w),
                    "inner_h": int(h - 2),
                    "inner_w": int(w - 2),
                }
            )
            if idx < n_frames - 1:
                c += int(w + gaps[idx])
        return frames
    return None


def _render_base(bg: int, frames: list[dict]) -> np.ndarray:
    grid = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(frame["border"])
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(frame["fill"])
    return grid


def _render_legend(grid: np.ndarray, fills: list[int], color_map: dict[int, int]) -> None:
    for rr, fill in enumerate(fills):
        grid[int(rr), 0] = int(fill)
        grid[int(rr), 1] = int(color_map[int(fill)])


def _make_sprite_patch(
    outer_color: int,
    target_frame: dict,
    fill_choices: list[int],
    force_inner: bool,
) -> dict | None:
    max_h = min(SPRITE_H_RANGE[1], int(target_frame["inner_h"]))
    max_w = min(SPRITE_W_RANGE[1], int(target_frame["inner_w"]))
    if max_h < SPRITE_H_RANGE[0] or max_w < SPRITE_W_RANGE[0]:
        return None

    can_have_inner = max_h >= 3 and max_w >= 3 and len(fill_choices) >= 2
    want_inner = can_have_inner and (force_inner or random.random() < INNER_COLOR_PROB)

    min_h = 3 if want_inner else SPRITE_H_RANGE[0]
    min_w = 3 if want_inner else SPRITE_W_RANGE[0]
    if max_h < min_h or max_w < min_w:
        if force_inner:
            return None
        want_inner = False
        min_h = SPRITE_H_RANGE[0]
        min_w = SPRITE_W_RANGE[0]

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    patch = np.full((h, w), int(outer_color), dtype=int)

    inner_color = None
    if want_inner:
        inner_choices = [int(c) for c in fill_choices if int(c) != int(outer_color)]
        if not inner_choices:
            if force_inner:
                return None
        else:
            valid_inner = [
                (ih, iw)
                for ih in range(1, h - 1)
                for iw in range(1, w - 1)
                if ih * iw * 2 < h * w
            ]
            if not valid_inner:
                if force_inner:
                    return None
            else:
                inner_color = int(random.choice(inner_choices))
                ih, iw = random.choice(valid_inner)
                ir0 = random.randint(1, h - ih - 1)
                ic0 = random.randint(1, w - iw - 1)
                patch[ir0 : ir0 + ih, ic0 : ic0 + iw] = int(inner_color)

    return {
        "patch": patch,
        "h": int(h),
        "w": int(w),
        "outer_color": int(outer_color),
        "inner_color": inner_color,
    }


def _make_derangement(n: int) -> list[int]:
    if n == 2:
        return [1, 0]
    return [1, 2, 0]


def _place_sprites(
    frames: list[dict],
    fills: list[int],
    connectors: list[int],
) -> list[dict] | None:
    top_end = min(int(frame["r0"]) for frame in frames)
    legend_rect = (0, 0, len(frames) - 1, LEGEND_COLS - 1)
    occ_rects: list[tuple[int, int, int, int]] = [legend_rect]

    target_to_outer = _make_derangement(len(frames))
    force_inner_idx = random.randint(0, len(frames) - 1)
    placed: list[dict] = []

    for target_idx, frame in enumerate(frames):
        outer_color = int(fills[target_to_outer[target_idx]])
        sprite = _make_sprite_patch(
            outer_color=outer_color,
            target_frame=frame,
            fill_choices=fills,
            force_inner=(target_idx == force_inner_idx),
        )
        if sprite is None:
            return None

        h = int(sprite["h"])
        w = int(sprite["w"])
        max_r0 = top_end - CONNECTOR_MIN_LEN - h - 2
        if max_r0 < 0:
            return None

        c_lo = int(frame["c0"]) + 1
        c_hi = int(frame["c1"]) - w
        if c_lo > c_hi:
            return None

        candidates = []
        for c0 in range(c_lo, c_hi + 1):
            connector_c = int(c0 + w // 2)
            line_start_offset = int(h + 1)
            line_end_row = int(top_end - 1)
            if not (int(frame["c0"]) + 1 <= connector_c <= int(frame["c1"]) - 1):
                continue
            for r0 in range(0, max_r0 + 1):
                line_r0 = int(r0 + line_start_offset)
                line_h = int(line_end_row - line_r0 + 1)
                if line_h < CONNECTOR_MIN_LEN:
                    continue
                sprite_rect = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))
                track_rect = (int(line_r0), int(connector_c), int(line_end_row), int(connector_c))
                if not _rect_separated(occ_rects, r0, c0, h, w):
                    continue
                if not _rect_separated(occ_rects, line_r0, connector_c, line_h, 1):
                    continue
                candidates.append((int(r0), int(c0), int(connector_c), sprite_rect, track_rect))

        if not candidates:
            return None

        r0, c0, connector_c, sprite_rect, track_rect = random.choice(candidates)
        occ_rects.append(sprite_rect)
        occ_rects.append(track_rect)
        placed.append(
            {
                "target_idx": int(target_idx),
                "patch": np.array(sprite["patch"], dtype=int),
                "h": int(h),
                "w": int(w),
                "r0": int(r0),
                "c0": int(c0),
                "connector_c": int(connector_c),
                "connector_color": int(connectors[target_idx]),
                "outer_color": int(sprite["outer_color"]),
                "inner_color": sprite["inner_color"],
            }
        )

    return placed


def _render(
    base: np.ndarray,
    placed: list[dict],
    frames: list[dict],
    color_map: dict[int, int],
    bg: int,
) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    top_end = min(int(frame["r0"]) for frame in frames)

    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        paste(inp, patch, r0, c0, transparent=int(bg))
        inp[int(r0 + sprite["h"] + 1) : int(top_end), int(sprite["connector_c"])] = int(
            sprite["connector_color"]
        )
        frame = frames[int(sprite["target_idx"])]
        inp[int(frame["r0"]), int(sprite["connector_c"])] = int(sprite["connector_color"])

    _render_legend(inp, [int(frame["fill"]) for frame in frames], color_map)
    out[:top_end, :] = int(bg)

    for sprite in placed:
        frame = frames[int(sprite["target_idx"])]
        transformed = _apply_color_map(np.array(sprite["patch"], dtype=int), color_map, bg)
        place_r = int(frame["r1"] - sprite["h"])
        place_c = int(sprite["c0"])
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_frames]]
    borders = [int(v) for v in palette[1 + n_frames : 1 + 2 * n_frames]]
    connector_colors = [int(v) for v in palette[1 + 2 * n_frames : 1 + 3 * n_frames]]

    frames = _build_frames(bg, fills, borders)
    if frames is None:
        return None

    color_map = {
        int(fill): int(dst)
        for fill, dst in zip(fills, borders[1:] + borders[:1])
    }
    placed = _place_sprites(frames, fills, connector_colors)
    if placed is None:
        return None

    base = _render_base(bg, frames)
    inp, out = _render(base, placed, frames, color_map, bg)

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
