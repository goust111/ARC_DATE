from __future__ import annotations

import random

import numpy as np


BG = 8
ANCHOR = 9
CONFLICT = 2
DOT_COLORS = (1, 3, 4, 5, 6, 7)
ORIENTATIONS = ("up", "down", "left", "right")
TRIANGLE_AREAS = (3, 6)
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _triangle_side(area: int) -> int:
    if area == 3:
        return 2
    if area == 6:
        return 3
    raise AssertionError(f"unexpected triangle area: {area}")


def _triangle_offsets(area: int, orientation: str) -> frozenset[tuple[int, int]]:
    side = _triangle_side(area)
    if orientation == "up":
        return frozenset((dr, dc) for dr in range(side) for dc in range(dr + 1))
    if orientation == "down":
        return frozenset((dr, dc) for dr in range(side) for dc in range(side - dr))
    if orientation == "left":
        return frozenset((dr, dc) for dc in range(side) for dr in range(dc + 1))
    if orientation == "right":
        return frozenset((dr, dc) for dc in range(side) for dr in range(side - dc))
    raise AssertionError(f"unknown orientation: {orientation}")


TRIANGLE_PATTERNS = {
    (area, orientation): _triangle_offsets(area, orientation)
    for area in TRIANGLE_AREAS
    for orientation in ORIENTATIONS
}


def _triangle_spec(component: list[tuple[int, int]]) -> dict[str, object]:
    area = len(component)
    assert area in TRIANGLE_AREAS, "triangle anchors must contain 3 or 6 pixels"

    pts = np.asarray(component, dtype=int)
    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    rel = frozenset((int(r) - r0, int(c) - c0) for r, c in component)

    orientation = None
    for candidate in ORIENTATIONS:
        if rel == TRIANGLE_PATTERNS[(area, candidate)]:
            orientation = candidate
            break
    assert orientation is not None, "anchor component must be one of the four staircase right triangles"

    if orientation == "up":
        axis_kind = "row"
        axis_index = r0
        source_sign = 1
    elif orientation == "down":
        axis_kind = "row"
        axis_index = r1
        source_sign = -1
    elif orientation == "left":
        axis_kind = "col"
        axis_index = c0
        source_sign = 1
    else:
        axis_kind = "col"
        axis_index = c1
        source_sign = -1

    return {
        "orientation": orientation,
        "band": area,
        "axis_kind": axis_kind,
        "axis_index": int(axis_index),
        "source_sign": int(source_sign),
        "cells": frozenset(component),
    }


def _classify(spec: dict[str, object], r: int, c: int) -> str | None:
    axis_kind = str(spec["axis_kind"])
    axis_index = int(spec["axis_index"])
    band = int(spec["band"])
    source_sign = int(spec["source_sign"])
    delta = r - axis_index if axis_kind == "row" else c - axis_index

    if abs(delta) > band:
        return None
    if delta == 0:
        return "axis"
    if delta * source_sign > 0:
        return "source"
    return "target"


def _reflect(spec: dict[str, object], r: int, c: int) -> tuple[int, int]:
    axis_index = int(spec["axis_index"])
    if str(spec["axis_kind"]) == "row":
        return 2 * axis_index - r, c
    return r, 2 * axis_index - c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the scattered dots toward each maroon triangle's indicated symmetry axis.

    Input structure:
    - The background is azure.
    - Maroon connected components are small solid right triangles of area 3 or 6,
      each pointing up, down, left, or right.
    - The remaining non-background cells are isolated colored dots.

    Rule:
    - Each triangle defines a fold axis through its pointed right-angle corner; the
      fold band extends `N` cells to each side of that axis, where `N` is the
      triangle's area.
    - Only the dots on the side the triangle points away from are folded across the
      axis; dots already on the destination side stay in place.
    - A moved dot landing on the same color merges unchanged, while a moved dot
      landing on a different color turns that destination cell red.
    - All triangle pixels, all source-side dots, and every dot lying directly on a
      fold axis are erased to the background.

    Key insight:
    - Each anchor triangle jointly encodes fold direction, axis position, and fold
      width, so the task is to decode the triangle geometry before performing a
      one-sided mirror overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = map(int, g.shape)

    bg = _background_color(g)
    triangles = [_triangle_spec(comp) for comp in _components4(g == ANCHOR)]
    assert triangles, "expected at least one maroon triangle"
    triangles.sort(key=lambda spec: (str(spec["axis_kind"]), int(spec["axis_index"]), str(spec["orientation"])))

    out = g.copy()
    for spec in triangles:
        for r, c in spec["cells"]:
            out[r, c] = bg
        if str(spec["axis_kind"]) == "row":
            out[int(spec["axis_index"]), :] = bg
        else:
            out[:, int(spec["axis_index"])] = bg

    moves: list[tuple[int, int, int]] = []
    dot_mask = (g != bg) & (g != ANCHOR)
    for r_raw, c_raw in np.argwhere(dot_mask):
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        matched = [spec for spec in triangles if _classify(spec, r, c) is not None]
        assert len(matched) <= 1, "a colored dot cannot belong to multiple fold bands"
        if not matched:
            continue
        role = _classify(matched[0], r, c)
        if role == "axis":
            out[r, c] = bg
        elif role == "source":
            out[r, c] = bg
            rr, cc = _reflect(matched[0], r, c)
            assert 0 <= rr < h and 0 <= cc < w, "reflected dot must remain in bounds"
            moves.append((rr, cc, color))

    for r, c, color in moves:
        cur = int(out[r, c])
        if cur == bg or cur == color:
            out[r, c] = color
        else:
            out[r, c] = CONFLICT

    return out


def _triangle_cells(top: int, left: int, area: int, orientation: str) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in sorted(_triangle_offsets(area, orientation))]


def _render_triangle(top: int, left: int, area: int, orientation: str) -> dict[str, object]:
    cells = _triangle_cells(top, left, area, orientation)
    return _triangle_spec(cells)


def _placeable_triangle_specs(h: int, w: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for area in TRIANGLE_AREAS:
        side = _triangle_side(area)
        for orientation in ORIENTATIONS:
            if orientation in ("up", "down"):
                for axis in range(area, h - area):
                    top = axis if orientation == "up" else axis - side + 1
                    for left in range(0, w - side + 1):
                        specs.append(_render_triangle(top, left, area, orientation))
            else:
                for axis in range(area, w - area):
                    left = axis if orientation == "left" else axis - side + 1
                    for top in range(0, h - side + 1):
                        specs.append(_render_triangle(top, left, area, orientation))
    return specs


def _triangle_halo(spec: dict[str, object], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    halo: set[tuple[int, int]] = set()
    for r, c in spec["cells"]:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                halo.add((rr, cc))
    return halo


def _band_mask(spec: dict[str, object], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    axis = int(spec["axis_index"])
    band = int(spec["band"])
    if str(spec["axis_kind"]) == "row":
        r0 = max(0, axis - band)
        r1 = min(h - 1, axis + band)
        mask[r0 : r1 + 1, :] = True
    else:
        c0 = max(0, axis - band)
        c1 = min(w - 1, axis + band)
        mask[:, c0 : c1 + 1] = True
    return mask


def _pair_candidates(
    spec: dict[str, object],
    membership: np.ndarray,
    triangle_forbidden: np.ndarray,
    triangle_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = membership.shape
    axis = int(spec["axis_index"])
    band = int(spec["band"])
    sign = int(spec["source_sign"])
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

    if str(spec["axis_kind"]) == "row":
        for dist in range(1, band + 1):
            src_r = axis + sign * dist
            dst_r = axis - sign * dist
            if not (0 <= src_r < h and 0 <= dst_r < h):
                continue
            for c in range(w):
                src = (src_r, c)
                dst = (dst_r, c)
                if src in triangle_cells or dst in triangle_cells:
                    continue
                if membership[src] != 1 or membership[dst] != 1:
                    continue
                if triangle_forbidden[src] or triangle_forbidden[dst]:
                    continue
                candidates.append((src, dst))
    else:
        for dist in range(1, band + 1):
            src_c = axis + sign * dist
            dst_c = axis - sign * dist
            if not (0 <= src_c < w and 0 <= dst_c < w):
                continue
            for r in range(h):
                src = (r, src_c)
                dst = (r, dst_c)
                if src in triangle_cells or dst in triangle_cells:
                    continue
                if membership[src] != 1 or membership[dst] != 1:
                    continue
                if triangle_forbidden[src] or triangle_forbidden[dst]:
                    continue
                candidates.append((src, dst))

    random.shuffle(candidates)
    return candidates


def _target_only_candidates(
    spec: dict[str, object],
    membership: np.ndarray,
    triangle_forbidden: np.ndarray,
    triangle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = membership.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in triangle_cells:
                continue
            if membership[r, c] != 1 or triangle_forbidden[r, c]:
                continue
            if _classify(spec, r, c) == "target":
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _axis_candidates(
    spec: dict[str, object],
    membership: np.ndarray,
    triangle_forbidden: np.ndarray,
    triangle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = membership.shape
    candidates: list[tuple[int, int]] = []
    axis = int(spec["axis_index"])
    if str(spec["axis_kind"]) == "row":
        for c in range(w):
            cell = (axis, c)
            if cell in triangle_cells:
                continue
            if membership[cell] == 1 and not triangle_forbidden[cell]:
                candidates.append(cell)
    else:
        for r in range(h):
            cell = (r, axis)
            if cell in triangle_cells:
                continue
            if membership[cell] == 1 and not triangle_forbidden[cell]:
                candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _outside_candidates(
    shape: tuple[int, int],
    membership: np.ndarray,
    triangle_forbidden: np.ndarray,
    triangle_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if membership[r, c] == 0 and not triangle_forbidden[r, c] and (r, c) not in triangle_cells
    ]
    random.shuffle(candidates)
    return candidates


def _pick_free_pair(
    candidates: list[tuple[tuple[int, int], tuple[int, int]]],
    occupied: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for src, dst in candidates:
        if src not in occupied and dst not in occupied:
            return src, dst
    return None


def _pick_free_cell(candidates: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> tuple[int, int] | None:
    for cell in candidates:
        if cell not in occupied:
            return cell
    return None


def _attempt_triangles(h: int, w: int, count: int) -> list[dict[str, object]] | None:
    candidates = _placeable_triangle_specs(h, w)
    random.shuffle(candidates)

    chosen: list[dict[str, object]] = []
    occupied_halo: set[tuple[int, int]] = set()
    for spec in candidates:
        cells = set(spec["cells"])
        if cells & occupied_halo:
            continue
        halo = _triangle_halo(spec, (h, w))
        chosen.append(spec)
        occupied_halo |= halo
        if len(chosen) == count:
            return chosen
    return None


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    triangle_count = random.randint(2, 3)
    triangles = _attempt_triangles(h, w, triangle_count)
    if triangles is None:
        return None

    triangle_cells = {cell for spec in triangles for cell in spec["cells"]}
    triangle_forbidden = np.zeros((h, w), dtype=bool)
    for spec in triangles:
        for cell in _triangle_halo(spec, (h, w)):
            triangle_forbidden[cell] = True

    band_masks = [_band_mask(spec, (h, w)) for spec in triangles]
    membership = sum(mask.astype(int) for mask in band_masks)

    pair_pools = [_pair_candidates(spec, membership, triangle_forbidden, triangle_cells) for spec in triangles]
    target_pools = [_target_only_candidates(spec, membership, triangle_forbidden, triangle_cells) for spec in triangles]
    axis_pools = [_axis_candidates(spec, membership, triangle_forbidden, triangle_cells) for spec in triangles]
    outside_pool = _outside_candidates((h, w), membership, triangle_forbidden, triangle_cells)

    if any(len(pool) < 2 for pool in pair_pools):
        return None
    if not outside_pool:
        return None
    if not any(axis_pools):
        return None

    pair_counts = [random.randint(2, 3) for _ in triangles]
    remaining = pair_counts[:]
    pair_plan: list[list[str]] = [[] for _ in triangles]
    mandatory_events = ["same", "diff", "empty"]
    for event in mandatory_events:
        choices = [idx for idx, slots in enumerate(remaining) if slots > 0]
        if not choices:
            return None
        idx = random.choice(choices)
        pair_plan[idx].append(event)
        remaining[idx] -= 1
    for idx in range(len(triangles)):
        while remaining[idx] > 0:
            pair_plan[idx].append(random.choice(["empty", "same", "diff"]))
            remaining[idx] -= 1

    grid = np.full((h, w), BG, dtype=int)
    for spec in triangles:
        for r, c in spec["cells"]:
            grid[r, c] = ANCHOR

    occupied: set[tuple[int, int]] = set()

    def paint(cell: tuple[int, int], color: int) -> None:
        grid[cell] = color
        occupied.add(cell)

    for idx, events in enumerate(pair_plan):
        for event in events:
            pair = _pick_free_pair(pair_pools[idx], occupied)
            if pair is None:
                return None
            src, dst = pair
            if event == "empty":
                paint(src, random.choice(DOT_COLORS))
            elif event == "same":
                color = random.choice(DOT_COLORS)
                paint(src, color)
                paint(dst, color)
            else:
                src_color, dst_color = random.sample(DOT_COLORS, 2)
                paint(src, src_color)
                paint(dst, dst_color)

        if target_pools[idx] and random.random() < 0.5:
            cell = _pick_free_cell(target_pools[idx], occupied)
            if cell is not None:
                paint(cell, random.choice(DOT_COLORS))

    axis_choices = [idx for idx, pool in enumerate(axis_pools) if pool]
    axis_idx = random.choice(axis_choices)
    axis_cell = _pick_free_cell(axis_pools[axis_idx], occupied)
    if axis_cell is None:
        return None
    paint(axis_cell, random.choice(DOT_COLORS))

    for idx, pool in enumerate(axis_pools):
        if idx == axis_idx or random.random() >= 0.35:
            continue
        cell = _pick_free_cell(pool, occupied)
        if cell is not None:
            paint(cell, random.choice(DOT_COLORS))

    outside_count = random.randint(1, 4)
    for _ in range(outside_count):
        cell = _pick_free_cell(outside_pool, occupied)
        if cell is None:
            break
        paint(cell, random.choice(DOT_COLORS))

    dot_colors = set(int(v) for v in np.unique(grid) if v not in {BG, ANCHOR})
    if len(dot_colors) < 2:
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
