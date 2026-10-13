from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _gravity_drop(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    for c in range(w):
        vals = g[g[:, c] != BG, c]
        if len(vals):
            out[h - len(vals) :, c] = vals
    return out


def _legend_perimeter_cells(r0: int, c0: int, outer: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r0 + outer):
        cells.add((r, c0))
        cells.add((r, c0 + outer - 1))
    for c in range(c0, c0 + outer):
        cells.add((r0, c))
        cells.add((r0 + outer - 1, c))
    return cells


def _detect_legend(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for coords in _components_mask(g == GREY):
        r0, c0, r1, c1 = _bbox_coords(coords)
        comp_set = {tuple(map(int, rc)) for rc in coords}
        for outer in (5, 6):
            if r0 + outer - 1 > r1 or c0 + outer - 1 > c1:
                continue
            corner = (r0 + outer - 1, c0 + outer - 1)
            if corner not in comp_set:
                continue

            perimeter = _legend_perimeter_cells(r0, c0, outer)
            if not perimeter.issubset(comp_set):
                continue

            extra = comp_set - perimeter
            has_right = False
            has_down = False
            valid_extra = True
            for rr, cc in extra:
                if rr == corner[0] and cc > corner[1]:
                    has_right = True
                elif cc == corner[1] and rr > corner[0]:
                    has_down = True
                else:
                    valid_extra = False
                    break
            if not valid_extra or not has_right or not has_down:
                continue
            if comp_set != (
                perimeter
                | {(corner[0], cc) for cc in range(corner[1], c1 + 1)}
                | {(rr, corner[1]) for rr in range(corner[0], r1 + 1)}
            ):
                continue

            inside = g[r0 + 1 : r0 + outer - 1, c0 + 1 : c0 + outer - 1]
            target_cells = np.argwhere((inside != BG) & (inside != GREY))
            if target_cells.shape[0] != 2:
                continue
            t0 = tuple(map(int, target_cells[0]))
            t1 = tuple(map(int, target_cells[1]))
            if abs(t0[0] - t1[0]) + abs(t0[1] - t1[1]) <= 1:
                continue
            target_colors = sorted(set(map(int, inside[(inside != BG) & (inside != GREY)])))
            if len(target_colors) != 2:
                continue

            legend_mask = np.zeros_like(g, dtype=bool)
            legend_mask[r0 : r0 + outer, c0 : c0 + outer] = True
            for rr, cc in comp_set:
                legend_mask[rr, cc] = True
            candidates.append((r0 + c0, r0, c0, legend_mask, tuple(target_colors), (r0, c0, r1, c1)))

    assert candidates, "expected a top-left grey legend"
    candidates.sort(key=lambda item: item[:3])
    best = candidates[0]
    return best[3], best[4], best[5]


def _erase_matching_components(grid: np.ndarray, legend_mask: np.ndarray, target_colors: tuple[int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    g[legend_mask] = BG
    for coords in _components_mask(g != BG):
        colors = {int(g[r, c]) for r, c in coords}
        if set(target_colors).issubset(colors):
            for r, c in coords:
                g[r, c] = BG
    return g


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase every multicolor component that contains the legend's two target colors, then let the remaining pixels fall by column.

    Input structure:
    - A black background contains a top-left grey legend made from a square border with right/down boundary lines.
    - Inside the bordered square, exactly two non-grey colored pixels define the target color set.
    - Away from the legend, the foreground is split into several 4-connected multicolor components.

    Rule:
    - Read the two colors inside the legend border, ignoring the grey frame.
    - Among all non-legend 4-connected non-black components, erase every component whose set of colors contains both target colors.
    - Erase the whole legend region as well.
    - Finally process each column independently: collect that column's surviving non-black pixels from top to bottom and pack them against the bottom edge in the same top-to-bottom order.

    Key insight:
    - The legend does not name pixels directly; it filters whole connected components by color-set inclusion, and only the gaps created by that erasure determine the final gravity stack.
    """
    g = np.asarray(grid, dtype=int)
    legend_mask, target_colors, _legend_bbox = _detect_legend(g)
    erased = _erase_matching_components(g, legend_mask, target_colors)
    return _gravity_drop(erased)


def _make_shape(rng: np.random.Generator, area: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        if h * w < area:
            continue
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(0, len(frontier)))])
        if len(cells) != area:
            continue
        coords = np.array(sorted(cells), dtype=int)
        r0, c0, r1, c1 = _bbox_coords(coords)
        coords[:, 0] -= r0
        coords[:, 1] -= c0
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 2 or ww < 2:
            continue
        return coords
    return None


def _colorize_shape(
    rng: np.random.Generator,
    shape: np.ndarray,
    colors: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    cells = [tuple(map(int, rc)) for rc in shape]
    rng.shuffle(cells)
    order = list(colors)
    counts = [1] * len(order)
    for _ in range(len(cells) - len(order)):
        counts[int(rng.integers(0, len(order)))] += 1
    mapping: dict[tuple[int, int], int] = {}
    idx = 0
    for color, count in zip(order, counts):
        for _ in range(count):
            mapping[cells[idx]] = int(color)
            idx += 1
    return mapping


def _component_variant_colors(
    rng: np.random.Generator,
    target_colors: tuple[int, int],
    kind: str,
) -> tuple[int, ...]:
    t0, t1 = target_colors
    others = [c for c in PALETTE if c not in target_colors]
    if kind == "erase":
        extra = [int(rng.choice(others))] if rng.random() < 0.75 else []
        return (t0, t1, *extra)
    if kind == "partial":
        chosen_target = t0 if rng.random() < 0.5 else t1
        extras = list(rng.choice(others, size=int(rng.integers(1, 3)), replace=False))
        return (int(chosen_target), *map(int, extras))
    extras = list(rng.choice(others, size=int(rng.integers(2, 4)), replace=False))
    return tuple(map(int, extras))


def _place_component(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    legend_mask: np.ndarray,
    patch: dict[tuple[int, int], int],
    *,
    row_range: tuple[int, int] | None = None,
    col_range: tuple[int, int] | None = None,
    require_overlap_cols: set[int] | None = None,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    shape = np.array(list(patch.keys()), dtype=int)
    sr0, sc0, sr1, sc1 = _bbox_coords(shape)
    sh = sr1 - sr0 + 1
    sw = sc1 - sc0 + 1
    row_lo = 0 if row_range is None else max(0, row_range[0])
    row_hi = h - sh if row_range is None else min(h - sh, row_range[1])
    col_lo = 0 if col_range is None else max(0, col_range[0])
    col_hi = w - sw if col_range is None else min(w - sw, col_range[1])
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchors = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    rng.shuffle(anchors)
    legend_cells = set(map(tuple, np.argwhere(legend_mask)))
    for ar, ac in anchors:
        cells = [(ar + rr, ac + cc) for (rr, cc) in patch]
        if require_overlap_cols is not None and not (require_overlap_cols & {c for _, c in cells}):
            continue
        ok = True
        for r, c in cells:
            if not (0 <= r < h and 0 <= c < w):
                ok = False
                break
            if (r, c) in occupied or (r, c) in legend_cells:
                ok = False
                break
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) in occupied or (nr, nc) in legend_cells:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for (rr, cc), color in patch.items():
            grid[ar + rr, ac + cc] = int(color)
        placed = [(ar + rr, ac + cc) for rr, cc in patch]
        occupied.update(placed)
        return placed
    return None


def _remove_by_predicate(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    predicate,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int).copy()
    g[legend_mask] = BG
    for coords in _components_mask(g != BG):
        colors = {int(g[r, c]) for r, c in coords}
        if predicate(colors):
            for r, c in coords:
                g[r, c] = BG
    return g


def _oracle_output(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    target_colors: tuple[int, int],
) -> np.ndarray:
    erased = _remove_by_predicate(grid, legend_mask, lambda colors: set(target_colors).issubset(colors))
    return _gravity_drop(erased)


def _build_legend(
    grid: np.ndarray,
    rng: np.random.Generator,
    target_colors: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int], set[int]]:
    h, w = grid.shape
    inner = int(rng.choice([3, 4]))
    outer = inner + 2
    extra_right = int(rng.integers(2, 5))
    extra_down = int(rng.integers(2, 5))
    r0 = int(rng.integers(0, 2))
    c0 = int(rng.integers(0, 2))
    assert r0 + outer + extra_down <= h
    assert c0 + outer + extra_right <= w

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[r0 : r0 + outer, c0 : c0 + outer] = True
    corner_r = r0 + outer - 1
    corner_c = c0 + outer - 1
    legend_mask[corner_r, corner_c : corner_c + extra_right + 1] = True
    legend_mask[corner_r : corner_r + extra_down + 1, corner_c] = True

    for r, c in _legend_perimeter_cells(r0, c0, outer):
        grid[r, c] = GREY
    grid[corner_r, corner_c : corner_c + extra_right + 1] = GREY
    grid[corner_r : corner_r + extra_down + 1, corner_c] = GREY

    slots = [(r, c) for r in range(r0 + 1, r0 + outer - 1) for c in range(c0 + 1, c0 + outer - 1)]
    pairs = [(a, b) for i, a in enumerate(slots) for b in slots[i + 1 :] if abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1]
    p0, p1 = pairs[int(rng.integers(0, len(pairs)))]
    grid[p0] = int(target_colors[0])
    grid[p1] = int(target_colors[1])

    overlap_cols = set(range(c0 + 1, corner_c))
    return legend_mask, (r0, c0, corner_r + extra_down, corner_c + extra_right), overlap_cols


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(13, 19))
    w = int(rng.integers(13, 19))
    grid = np.zeros((h, w), dtype=int)

    target_colors = tuple(map(int, rng.choice(PALETTE, size=2, replace=False)))
    legend_mask, legend_bbox, overlap_cols = _build_legend(grid, rng, target_colors)
    occupied = set(map(tuple, np.argwhere(grid != BG)))

    comp_count = int(rng.integers(5, 8))
    kinds = ["erase", "erase", "partial", "none"]
    while len(kinds) < comp_count:
        kinds.append("erase" if rng.random() < 0.4 else ("partial" if rng.random() < 0.5 else "none"))
    rng.shuffle(kinds)

    placed_meta = []
    first_erase_lane_done = False
    first_survivor_lane_done = False
    legend_bottom = legend_bbox[2]

    for idx, kind in enumerate(kinds):
        area = int(rng.integers(4, 9))
        shape = _make_shape(rng, area)
        if shape is None:
            return None
        colors = _component_variant_colors(rng, target_colors, kind)
        patch = _colorize_shape(rng, shape, colors)

        row_range = None
        col_range = None
        require_cols = None
        if kind == "erase" and not first_erase_lane_done:
            row_range = (legend_bottom + 1, max(legend_bottom + 2, h // 2))
            col_range = (0, min(w - 2, max(overlap_cols)))
            require_cols = overlap_cols
        elif kind != "erase" and not first_survivor_lane_done:
            row_range = (max(legend_bottom + 3, h // 2), h - 2)
            col_range = (0, min(w - 2, max(overlap_cols)))
            require_cols = overlap_cols

        placed = _place_component(
            grid,
            occupied,
            legend_mask,
            patch,
            row_range=row_range,
            col_range=col_range,
            require_overlap_cols=require_cols,
            rng=rng,
        )
        if placed is None:
            placed = _place_component(grid, occupied, legend_mask, patch, rng=rng)
        if placed is None:
            return None

        color_set = {int(color) for color in patch.values()}
        placed_meta.append({"coords": placed, "colors": color_set, "kind": kind})
        if kind == "erase" and require_cols is not None:
            first_erase_lane_done = True
        if kind != "erase" and require_cols is not None:
            first_survivor_lane_done = True

    if not first_erase_lane_done or not first_survivor_lane_done:
        return None

    comp_mask = grid != BG
    comp_mask[legend_mask] = False
    comps = _components_mask(comp_mask)
    if len(comps) != comp_count:
        return None

    color_sets = [{int(grid[r, c]) for r, c in comp} for comp in comps]
    erasable = [colors for colors in color_sets if set(target_colors).issubset(colors)]
    survivors = [colors for colors in color_sets if not set(target_colors).issubset(colors)]
    if len(erasable) < 2 or len(survivors) < 2:
        return None
    if not any(len(set(colors) & set(target_colors)) == 1 for colors in survivors):
        return None
    if not any(len(set(colors) & set(target_colors)) == 0 for colors in survivors):
        return None
    if not any(colors == set(target_colors) for colors in erasable):
        return None
    if not any(len(colors) == 3 for colors in erasable):
        return None

    oracle = _oracle_output(grid, legend_mask, target_colors)
    if not np.array_equal(main(grid), oracle):
        return None

    erased_only = _remove_by_predicate(grid, legend_mask, lambda colors: set(target_colors).issubset(colors))
    if np.array_equal(erased_only, oracle):
        return None

    legend_only = _gravity_drop(_remove_by_predicate(grid, legend_mask, lambda colors: False))
    if np.array_equal(legend_only, oracle):
        return None

    any_target = _gravity_drop(_remove_by_predicate(grid, legend_mask, lambda colors: bool(set(colors) & set(target_colors))))
    if np.array_equal(any_target, oracle):
        return None

    exact_two = _gravity_drop(_remove_by_predicate(grid, legend_mask, lambda colors: set(colors) == set(target_colors)))
    if np.array_equal(exact_two, oracle):
        return None

    keep_legend = _gravity_drop(_remove_by_predicate(grid, np.zeros_like(legend_mask, dtype=bool), lambda colors: set(target_colors).issubset(colors)))
    if np.array_equal(keep_legend, oracle):
        return None

    if np.array_equal(grid, oracle):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
