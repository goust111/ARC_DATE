# Variant of 7c66cb00
# Dimension: D1 — 匹配条件：精灵与条的配对方式 (match_cond)
# Change: 精灵的外层颜色（多数色）等于目标条的填充色：s.outer_color == d.fill_color... → 边缘颜色匹配

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask, paste


# Design analysis for the D1 variant
# ----------------------------------
# spec_like:
#   unchanged_backbone:
#     grid: 30x24
#     bars:
#       count: 2-3
#       placement: bottom-up full-width horizontal bars with 1 bg row gap
#       geometry: border at col 0 and col W-1, fill at cols 1..W-2
#       heights: 4-8
#     movement:
#       erase: remove all top sprites from the input
#       place: preserve each sprite's x-position and bottom-align it in its matched bar
#   d1_change:
#     old_match: sprite outer/majority color == bar fill color
#     new_match: sprite outer contour first ring color == bar border color
#   sprite_design:
#     generator_shape: 2-color framed rectangles, size 3-5 by 3-9
#     edge_layer: 1-cell-thick contour colored with one bar's border color
#     interior: solid rectangle colored with another bar's fill color
#     reason:
#       - the decisive cue is now the perimeter color rather than the global dominant color
#       - fills and borders are sampled from disjoint palette slots, so border-match is visually necessary
#       - interior color is kept different from the target bar fill so output recoloring is visible
#   flow_trace:
#     original_local_behavior:
#       - solver scans per-color regions
#       - dominant color of the bbox decides the bar via fill-color match
#     variant_local_behavior:
#       - solver must first recover the whole sprite as one non-bg component
#       - solver flood-fills outside-background around the bbox and isolates only the outer contour ring
#       - that ring color selects the bar via border-color match
#       - edge pixels stay/become border color, all non-edge sprite pixels become bar fill
#     concrete_example:
#       - bar A: border=2 fill=7
#       - bar B: border=4 fill=6
#       - sprite: edge=2, interior=6
#       - old rule would be tempted by interior/fill color 6 or split the colors apart
#       - new rule must send the whole sprite to bar A because its contour ring is color 2
#   complexity_calibration:
#     - sprites stay 2-color and rectangular so the task remains readable in 2-4 examples
#     - every sample still has repeated bar/sprite structure; only the matching cue changes
#     - borders are unique per bar, avoiding ambiguous matches
#   hidden_assumptions_review:
#     old_solver_dependency_1:
#       code: iterate per color in top area and label components
#       issue: breaks a framed sprite into separate color objects, which is invalid once matching depends on the whole contour
#       fix: detect connected components on (top != bg) instead
#     old_solver_dependency_2:
#       code: outer_color = most_common(color counts in bbox)
#       issue: majority color no longer defines the target
#       fix: compute edge_mask and read its color
#     old_solver_dependency_3:
#       code: recolor only two named colors (outer/inner)
#       issue: new invariant is spatial layer, not color identity
#       fix: recolor by mask: edge ring -> border, remaining foreground -> fill
#   placement_legality:
#     - top sprites are separated by at least one bg cell in all directions
#     - sprites sharing a target bar are prevented from overlapping in x, so bottom-docking is collision-free
#     - each sprite height is clamped to its target bar height
#   observability:
#     - because fill colors and border colors are sampled from disjoint sets, a solver using fill-match cannot explain the examples
#     - the only stable repeated cue across examples is "the contour color equals the destination bar's side rails"
#   cleanliness:
#     - top area contains only a few separated framed rectangles
#     - bottom area contains only bars plus docked copies, with no extra sub-sprite placements
#   color_inference:
#     - output color change is structured: contour maps to bar border, interior maps to bar fill
#     - interior colors are borrowed from other bars' fills, keeping the palette small and interpretable
#   new_main_assumptions_and_generator_guarantees:
#     - each sprite has a uniform outer contour color
#     - each target bar border color is unique
#     - non-bg cells above the first bar belong only to sprites
#     - generator enforces all three conditions


def _find_bars(grid: np.ndarray, bg: int) -> list[dict]:
    """Detect full-width horizontal bars in the bottom portion."""
    h, w = grid.shape
    bars: list[dict] = []
    current: dict | None = None
    for r in range(h):
        row = grid[int(r), :]
        non_bg = [int(v) for v in row if int(v) != int(bg)]
        if non_bg:
            fill_color, count = Counter(non_bg).most_common(1)[0]
            if int(count) > int(w * 0.7):
                border = int(row[0]) if int(row[0]) != int(fill_color) else int(row[-1])
                if (
                    current is not None
                    and int(current["fill"]) == int(fill_color)
                    and int(current["border"]) == int(border)
                ):
                    current["r_max"] = int(r)
                else:
                    if current is not None:
                        bars.append(current)
                    current = {
                        "fill": int(fill_color),
                        "border": int(border),
                        "r_min": int(r),
                        "r_max": int(r),
                    }
            else:
                if current is not None:
                    bars.append(current)
                    current = None
        else:
            if current is not None:
                bars.append(current)
                current = None
    if current is not None:
        bars.append(current)
    return bars


def _outer_edge_mask(patch: np.ndarray, bg: int) -> np.ndarray:
    """
    Return the first outer contour ring of the foreground in `patch`.

    Background holes inside the sprite are ignored: only pixels touching the
    externally reachable background count as edge pixels.
    """
    fg = patch != int(bg)
    h, w = fg.shape
    padded_fg = np.pad(fg, 1, constant_values=False)
    external = np.zeros_like(padded_fg, dtype=bool)
    stack = [(0, 0)]
    external[0, 0] = True

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not external[nr, nc] and not padded_fg[nr, nc]:
                external[nr, nc] = True
                stack.append((nr, nc))

    edge = np.zeros_like(fg, dtype=bool)
    for r, c in np.argwhere(fg):
        pr = int(r) + 1
        pc = int(c) + 1
        if (
            external[pr - 1, pc]
            or external[pr + 1, pc]
            or external[pr, pc - 1]
            or external[pr, pc + 1]
        ):
            edge[int(r), int(c)] = True
    return edge


def _extract_sprites(grid: np.ndarray, bg: int, top_end: int) -> list[dict]:
    """Recover whole top sprites as non-bg connected components."""
    top = grid[:top_end, :]
    labels, n = label_mask(top != int(bg), connectivity=4)
    sprites: list[dict] = []
    for sprite_id in range(1, int(n) + 1):
        coords = np.argwhere(labels == int(sprite_id))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        content = crop_bbox(grid, (r0, c0, r1, c1))
        edge_mask = _outer_edge_mask(content, bg)
        edge_colors = Counter(map(int, content[edge_mask].flatten()))
        edge_colors.pop(int(bg), None)
        if not edge_colors:
            continue
        sprites.append(
            {
                "edge_color": int(edge_colors.most_common(1)[0][0]),
                "r0": int(r0),
                "c0": int(c0),
                "h": int(content.shape[0]),
                "w": int(content.shape[1]),
                "content": content,
                "edge_mask": edge_mask,
            }
        )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock top sprites into border-matched bars and recolor by spatial layer.

    The target bar is selected by contour color:
    sprite.outer_edge_color == bar.border_color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    bars = _find_bars(g, bg)
    assert bars, "No bars found"

    pattern_area_end = min(int(bar["r_min"]) for bar in bars)
    sprites = _extract_sprites(g, bg, pattern_area_end)

    top = out[:pattern_area_end, :]
    top[top != int(bg)] = int(bg)

    border_to_bar = {int(bar["border"]): bar for bar in bars}
    for sprite in sprites:
        bar = border_to_bar.get(int(sprite["edge_color"]))
        if bar is None:
            continue

        transformed = np.array(sprite["content"], dtype=int).copy()
        fg_mask = transformed != int(bg)
        transformed[fg_mask] = int(bar["fill"])
        transformed[sprite["edge_mask"]] = int(bar["border"])

        place_r = int(bar["r_max"] - sprite["h"] + 1)
        place_c = int(sprite["c0"])
        paste(out, transformed, place_r, place_c, transparent=int(bg))

    return out


# ---------------------------------------------------------------------------
# generator
# ---------------------------------------------------------------------------

GRID_H = 30
GRID_W = 24
BAR_COUNT_RANGE = (2, 3)
SPRITE_COUNT_RANGE = (2, 3)
BAR_HEIGHT_RANGE = (4, 8)
MAX_THREE_BAR_TOTAL = 20
MIN_TOP_AREA = 7
SPRITE_H_RANGE = (3, 5)
SPRITE_W_RANGE = (3, 9)
GEN_RETRIES = 300


def _sample_bar_heights(n_bars: int) -> list[int]:
    while True:
        heights = [random.randint(*BAR_HEIGHT_RANGE) for _ in range(n_bars)]
        if n_bars == 3 and sum(heights) > MAX_THREE_BAR_TOTAL:
            continue
        if GRID_H - sum(heights) - (n_bars - 1) < MIN_TOP_AREA:
            continue
        return heights


def _build_bars(bg: int, fills: list[int], borders: list[int], heights: list[int]) -> tuple[np.ndarray, list[dict]]:
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    bars: list[dict] = [None] * len(heights)  # type: ignore[list-item]
    r = GRID_H
    for i in range(len(heights) - 1, -1, -1):
        h = int(heights[i])
        r0 = int(r - h)
        r1 = int(r - 1)
        grid[r0 : r1 + 1, 1 : GRID_W - 1] = int(fills[i])
        grid[r0 : r1 + 1, 0] = int(borders[i])
        grid[r0 : r1 + 1, GRID_W - 1] = int(borders[i])
        bars[i] = {
            "fill": int(fills[i]),
            "border": int(borders[i]),
            "h": int(h),
            "r_min": int(r0),
            "r_max": int(r1),
        }
        r = int(r0 - 1)
    return grid, bars


def _make_sprite(target_idx: int, bars: list[dict]) -> dict:
    """Create a framed two-color sprite whose contour matches a bar border."""
    target_bar = bars[target_idx]
    h = random.randint(SPRITE_H_RANGE[0], min(SPRITE_H_RANGE[1], int(target_bar["h"])))
    w = random.randint(*SPRITE_W_RANGE)
    edge_color = int(target_bar["border"])
    other_fill_choices = [int(bar["fill"]) for i, bar in enumerate(bars) if int(i) != int(target_idx)]
    inner_color = int(random.choice(other_fill_choices))

    patch = np.full((h, w), edge_color, dtype=int)
    patch[1 : h - 1, 1 : w - 1] = inner_color

    return {
        "target_idx": int(target_idx),
        "h": int(h),
        "w": int(w),
        "patch": patch,
        "edge_mask": _outer_edge_mask(patch, bg=-1),
        "edge_color": int(edge_color),
        "inner_color": int(inner_color),
    }


def _interval_free(occ: np.ndarray, start: int, length: int) -> bool:
    return not occ[start : start + length].any()


def _rect_separated(rects: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int) -> bool:
    r1 = int(r0 + h - 1)
    c1 = int(c0 + w - 1)
    for rr0, cc0, rr1, cc1 in rects:
        if not (r1 < rr0 - 1 or rr1 < r0 - 1 or c1 < cc0 - 1 or cc1 < c0 - 1):
            return False
    return True


def _place_sprites(sprites: list[dict], bars: list[dict], top_h: int) -> list[dict] | None:
    bar_occ = [np.zeros(GRID_W, dtype=bool) for _ in bars]
    top_rects: list[tuple[int, int, int, int]] = []
    placed: list[dict] = []

    for sprite in sprites:
        h = int(sprite["h"])
        w = int(sprite["w"])
        target_idx = int(sprite["target_idx"])
        candidates: list[tuple[int, int]] = []

        for c0 in range(1, GRID_W - w):
            if not _interval_free(bar_occ[target_idx], c0, w):
                continue
            for r0 in range(0, top_h - h + 1):
                if _rect_separated(top_rects, r0, c0, h, w):
                    candidates.append((r0, c0))

        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        bar_occ[target_idx][c0 : c0 + w] = True
        placed.append({**sprite, "r0": int(r0), "c0": int(c0)})
        top_rects.append((int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)))

    return placed


def _render(base: np.ndarray, placed: list[dict], bars: list[dict], bg: int) -> tuple[np.ndarray, np.ndarray]:
    inp = base.copy()
    out = base.copy()
    for sprite in placed:
        patch = np.array(sprite["patch"], dtype=int)
        target_bar = bars[int(sprite["target_idx"])]
        h = int(sprite["h"])
        c0 = int(sprite["c0"])

        paste(inp, patch, int(sprite["r0"]), c0, transparent=int(bg))

        transformed = np.full_like(patch, int(bg))
        fg_mask = patch != int(bg)
        transformed[fg_mask] = int(target_bar["fill"])
        transformed[sprite["edge_mask"]] = int(target_bar["border"])
        paste(out, transformed, int(target_bar["r_max"] - h + 1), c0, transparent=int(bg))

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_bars = random.randint(*BAR_COUNT_RANGE)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    palette = list(range(10))
    random.shuffle(palette)
    bg = int(palette[0])
    fills = [int(v) for v in palette[1 : 1 + n_bars]]
    borders = [int(v) for v in palette[1 + n_bars : 1 + 2 * n_bars]]

    heights = _sample_bar_heights(n_bars)
    base, bars = _build_bars(bg, fills, borders, heights)
    top_h = int(bars[0]["r_min"])

    target_slots = [i for i in range(n_bars) for _ in range(2)]
    random.shuffle(target_slots)
    sprites = [_make_sprite(int(target_slots[i]), bars) for i in range(n_sprites)]
    sprites.sort(key=lambda sprite: (-int(sprite["w"]), -int(sprite["h"])))

    placed = _place_sprites(sprites, bars, top_h)
    if placed is None:
        return None

    inp, out = _render(base, placed, bars, bg)

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
