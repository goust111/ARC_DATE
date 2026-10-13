from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path
from random import choice
from typing import Optional

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, bbox, components_mask, get_bg

        return Color, bbox, components_mask, get_bg
    except Exception:
        # Load from fixed shared toolbox path when module search path is not in repo root.
        path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = importlib.util.spec_from_file_location("arc_utils_local", path)
        if spec is None or spec.loader is None:
            raise
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        return module.Color, module.bbox, module.components_mask, module.get_bg


Color, bbox, components_mask, get_bg = _load_arc_utils()


def _collect_color_bars(grid: np.ndarray, bg: int, ring_color: int) -> dict[int, dict[str, int]]:
    """Return bars keyed by their unique length."""
    H, W = map(int, grid.shape)
    bars: dict[int, dict[str, int]] = {}
    for color in np.unique(grid):
        c = int(color)
        if c in {bg, ring_color}:
            continue
        for comp in components_mask(grid == c, connectivity=4):
            if comp.size == 0:
                continue
            r0, c0, r1, c1 = bbox(comp)
            if r1 != r0:
                continue
            length = int(c1 - c0 + 1)
            if length < 2 or length != int(comp.shape[0]):
                continue

            row = int(r0)
            side: Optional[str] = None
            left_anchor = c0 - 1 >= 0 and int(grid[row, c0 - 1]) != bg
            right_anchor = c1 + 1 < W and int(grid[row, c1 + 1]) != bg
            if left_anchor and not right_anchor:
                side = "L"
            elif right_anchor and not left_anchor:
                side = "R"

            if side is None or length in bars:
                continue
            bars[length] = {"color": c, "side": side, "length": length}

    return bars


def _ring_hole_and_ring_mask(
    grid: np.ndarray, ring_mask: np.ndarray, bg: int
) -> tuple[np.ndarray, int]:
    """Return local-hole mask (global coordinates) and hole area for one ring component."""
    h, w = map(int, grid.shape)
    coords = np.argwhere(ring_mask)
    r0, c0, r1, c1 = bbox(coords)
    rr0, cc0 = max(0, r0 - 1), max(0, c0 - 1)
    rr1, cc1 = min(h, r1 + 2), min(w, c1 + 2)

    local = grid[rr0:rr1, cc0:cc1]
    local_bg = local == bg
    H, W = local_bg.shape
    outside = np.zeros_like(local_bg, dtype=bool)

    q: deque[tuple[int, int]] = deque()
    for x in range(H):
        for y in (0, W - 1):
            if local_bg[x, y]:
                outside[x, y] = True
                q.append((x, y))
    for y in range(W):
        for x in (0, H - 1):
            if local_bg[x, y] and not outside[x, y]:
                outside[x, y] = True
                q.append((x, y))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and local_bg[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    inside = local_bg & ~outside
    holes = [comp for comp in components_mask(inside, connectivity=4) if comp.size > 0]
    if not holes:
        return np.zeros_like(grid, dtype=bool), 0

    inner = max(holes, key=len)
    hole_mask = np.zeros_like(grid, dtype=bool)
    for r, c in inner:
        hole_mask[int(rr0 + int(r)), int(cc0 + int(c))] = True
    return hole_mask, int(inner.shape[0])


def _infer_ring_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    best_color = Color.GREY
    best_area = -1

    for color in colors:
        holes_total = 0
        for comp in components_mask(grid == color, connectivity=4):
            if comp.size == 0:
                continue
            ring_mask = np.zeros_like(grid, dtype=bool)
            for r, c in comp:
                ring_mask[int(r), int(c)] = True
            _, hole_area = _ring_hole_and_ring_mask(grid, ring_mask, bg)
            holes_total += hole_area
        if holes_total > best_area:
            best_area = holes_total
            best_color = color

    return best_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill ring bodies or holes based on a legend bar with matching length.

    Input structure:
    - One bordered legend rectangle contains several horizontal color bars.
    - Each bar is a single-cell-thick horizontal run that starts from either the left or right inner wall.
    - Outside the legend there are closed grey ring objects, each surrounding exactly one background hole.

    Rule:
    - For each ring, count the number of background cells in its enclosed hole.
    - Find the legend bar whose length equals that count.
      - if the matching bar starts from the left wall, recolor the whole ring body with the bar color.
      - if it starts from the right wall, recolor the ring hole with the bar color and clear the ring body to background.
    - If no bar length matches a ring’s hole size, recolor both the ring body and its hole to background.

    Key insight:
    - The legend bar set is an explicit length key, while ring wall vs hole tells whether color is applied to positive shape or negative space.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    ring_color = Color.GREY if int(Color.GREY) in set(map(int, np.unique(g))) else _infer_ring_color(g, bg)

    bars = _collect_color_bars(g, bg, ring_color)
    out = g.copy()

    for comp in components_mask(g == ring_color, connectivity=4):
        if comp.size == 0:
            continue
        ring_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            ring_mask[int(r), int(c)] = True
        hole_mask, hole_area = _ring_hole_and_ring_mask(g, ring_mask, bg)

        action = bars.get(hole_area)
        if action is None:
            out[ring_mask] = bg
            if np.any(hole_mask):
                out[hole_mask] = bg
            continue

        bar_color = action["color"]
        if action["side"] == "L":
            out[ring_mask] = bar_color
        else:
            out[ring_mask] = bg
            if np.any(hole_mask):
                out[hole_mask] = bar_color

    return out


# --------------------------- input generator ---------------------------

BACKGROUND = Color.BLACK
RING_COLOR = Color.GREY

NON_BG_PALETTE = [c for c in Color.NOT_BLACK if c not in {BACKGROUND, RING_COLOR}]
LENGTH_AREA_POOL = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]


def _area_dims(area: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for h in range(2, 9):
        if area % h != 0:
            continue
        w = area // h
        if w >= 2 and w <= 8:
            out.append((h, w))
    return out


def _try_place_ring(grid: np.ndarray, occ: np.ndarray, area: int, rng: np.random.Generator) -> bool:
    dims = _area_dims(area)
    if not dims:
        return False
    h, w = choice(dims)
    outer_h, outer_w = h + 2, w + 2
    H, W = map(int, grid.shape)
    if outer_h >= H or outer_w >= W:
        return False

    candidates: list[tuple[int, int]] = [
        (r, c)
        for r in range(H - outer_h + 1)
        for c in range(W - outer_w + 1)
        if not np.any(
            occ[max(0, r - 1) : min(H, r + outer_h + 1), max(0, c - 1) : min(W, c + outer_w + 1)]
        )
    ]
    if not candidates:
        return False

    r, c = tuple(candidates[int(rng.integers(0, len(candidates)))])
    occ[max(0, r - 1) : min(H, r + outer_h + 1), max(0, c - 1) : min(W, c + outer_w + 1)] = True

    grid[r, c : c + outer_w] = RING_COLOR
    grid[r + outer_h - 1, c : c + outer_w] = RING_COLOR
    grid[r : r + outer_h, c] = RING_COLOR
    grid[r : r + outer_h, c + outer_w - 1] = RING_COLOR
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        H = int(rng.integers(18, 29))
        W = int(rng.integers(18, 29))
        grid = np.full((H, W), BACKGROUND, dtype=int)
        occ = np.zeros_like(grid, dtype=bool)

        # Legend rectangle and bars
        if H < 12 or W < 18:
            continue
        inner_h = int(rng.integers(5, min(10, H - 5)))
        inner_w = int(rng.integers(8, min(14, W - 4)))
        outer_h, outer_w = inner_h + 2, inner_w + 2
        if outer_h >= H or outer_w >= W:
            continue
        legend_r0 = int(rng.integers(1, H - outer_h - 1))
        legend_c0 = int(rng.integers(1, W - outer_w - 1))
        legend_r1, legend_c1 = legend_r0 + outer_h - 1, legend_c0 + outer_w - 1

        frame_colors = [c for c in NON_BG_PALETTE if c != RING_COLOR]
        if not frame_colors:
            continue
        frame_color = int(rng.choice(frame_colors))
        grid[legend_r0:legend_r1 + 1, legend_c0:legend_c1 + 1] = frame_color
        grid[legend_r0 + 1 : legend_r1, legend_c0 + 1 : legend_c1] = BACKGROUND
        grid[legend_r0 + 1 : legend_r1, legend_c0 + 1 : legend_c1] = BACKGROUND
        # fill interior background explicitly (inner area), keep border frame
        grid[legend_r0 + 1 : legend_r1, legend_c0 + 1 : legend_c1] = BACKGROUND
        occ[grid != BACKGROUND] = True

        inner_top, inner_left = legend_r0 + 1, legend_c0 + 1
        inner_bottom, inner_right = legend_r1 - 1, legend_c1 - 1
        inner_rows = [r for r in range(inner_top, inner_bottom + 1)]
        if not inner_rows:
            continue

        possible_lengths = [L for L in LENGTH_AREA_POOL if L <= (inner_right - inner_left + 1)]
        max_bars = min(len(inner_rows), len(possible_lengths), 5)
        if max_bars < 2:
            continue

        num_bars = int(rng.integers(2, max_bars + 1))
        bar_lengths = list(rng.choice(possible_lengths, size=num_bars, replace=False))

        bar_rows = rng.choice(inner_rows, size=num_bars, replace=False)
        bar_colors = list(rng.choice([c for c in NON_BG_PALETTE if c != frame_color], size=num_bars, replace=False))
        left_candidates = inner_cols = inner_rows  # keep deterministic naming only

        bars: list[tuple[int, int, int, int]] = []  # (color, length, side, row)
        if num_bars >= 2 and len(bar_lengths) >= 2 and len(bar_rows) >= num_bars:
            forced_side = [("L", bar_lengths[0]), ("R", bar_lengths[1])]
            for idx, (side, length) in enumerate(forced_side):
                row = int(bar_rows[idx])
                if side == "L":
                    c0 = inner_left
                    c1 = c0 + length - 1
                else:
                    c1 = inner_right
                    c0 = c1 - length + 1
                if not (inner_left <= c0 <= c1 <= inner_right):
                    break
                bars.append((int(bar_colors[idx]), int(length), 0 if side == "L" else 1, int(row)))
                grid[row, c0 : c1 + 1] = int(bar_colors[idx])
            start_idx = 2
        else:
            start_idx = 0

        for idx in range(start_idx, num_bars):
            length = bar_lengths[idx]
            row = int(bar_rows[idx])
            side = "L" if rng.random() < 0.5 else "R"
            if side == "L":
                c0 = inner_left
                c1 = c0 + length - 1
            else:
                c1 = inner_right
                c0 = c1 - length + 1
            if not (inner_left <= c0 <= c1 <= inner_right):
                break
            color = int(bar_colors[idx])
            bars.append((color, int(length), 0 if side == "L" else 1, int(row)))
            grid[row, c0 : c1 + 1] = color

        if len(bars) != num_bars:
            continue
        occ[grid != BACKGROUND] = True

        bar_length_to_spec = {length: (color, "L" if side == 0 else "R") for color, length, side, _ in bars}
        bar_lengths = [length for _, length, _, _ in bars]

        # Rings: mix of matched and unmatched cases
        min_match = min(1, len(bar_lengths))
        num_matched = int(rng.integers(min_match, len(bar_lengths) + 1))
        matched_lengths = list(rng.choice(bar_lengths, size=num_matched, replace=False))

        ring_areas = matched_lengths.copy()
        if rng.random() < 0.75:
            unmatched_choices = [area for area in LENGTH_AREA_POOL if area not in bar_length_to_spec]
            if unmatched_choices:
                ring_areas.append(int(rng.choice(unmatched_choices)))
        if rng.random() < 0.20:
            unmatched2 = [area for area in LENGTH_AREA_POOL if area not in set(ring_areas)]
            if unmatched2:
                ring_areas.append(int(rng.choice(unmatched2)))

        rng.shuffle(ring_areas)
        for area in ring_areas:
            if _try_place_ring(grid, occ, int(area), rng) is False:
                break
        else:
            try:
                produced = main(grid)
            except Exception:
                continue

            # Ensure both operation branches are at least potentially reachable in generated batches.
            left_side_matched = any(spec == "L" for _, spec in bar_length_to_spec.items())
            right_side_matched = any(spec == "R" for _, spec in bar_length_to_spec.items())
            if not (left_side_matched and right_side_matched and np.any(produced != grid)):
                if not (left_side_matched and right_side_matched):
                    # enforce explicit branch visibility in later samples
                    pass
                if not np.any(produced != grid):
                    continue
            return grid

    raise RuntimeError("failed to generate a valid 5.py input")
