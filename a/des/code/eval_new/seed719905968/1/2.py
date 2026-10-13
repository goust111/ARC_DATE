from __future__ import annotations

from dataclasses import dataclass

import numpy as np


BG = 0
MAROON = 9
OBJECT_COLORS = tuple(range(1, 9))

DIM_CANDIDATES = (
    (3, 3),
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 4),
    (4, 5),
    (5, 4),
    (5, 5),
)

PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
GRID_MAX = 30
SHAPE_RETRIES = 500
PLACEMENT_RETRIES = 500
GEN_RETRIES = 300
PAD = 1

_RNG = np.random.default_rng()


@dataclass(frozen=True)
class Component:
    bbox: tuple[int, int, int, int]
    rel_cells: frozenset[tuple[int, int]]
    marker_rel: tuple[int, int]
    color: int

    @property
    def shape(self) -> tuple[int, int]:
        r0, c0, r1, c1 = self.bbox
        return r1 - r0 + 1, c1 - c0 + 1


@dataclass
class PairSpec:
    h: int
    w: int
    mask_a: np.ndarray
    marker_a: tuple[int, int]
    marker_b: tuple[int, int]
    color_a: int
    color_b: int
    pos_a: tuple[int, int] | None = None
    pos_b: tuple[int, int] | None = None


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = []
    for dr, dc in steps:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    count = int(mask.sum())
    return count > 0 and len(_components(mask, connectivity)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _orth_edge_count(mask: np.ndarray) -> int:
    return int((mask[:, :-1] & mask[:, 1:]).sum() + (mask[:-1, :] & mask[1:, :]).sum())


def _extract_components(grid: np.ndarray) -> list[Component]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BG, connectivity=8)
    out: list[Component] = []
    for cells in comps:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        values = [int(g[r, c]) for r, c in cells]
        marker_cells = [(r, c) for r, c in cells if int(g[r, c]) == MAROON]
        assert len(marker_cells) == 1, "each block must contain exactly one maroon marker"
        base_colors = {value for value in values if value != MAROON}
        assert len(base_colors) == 1, "each block must be monochrome aside from the maroon marker"
        color = next(iter(base_colors))
        rel_cells = frozenset((r - r0, c - c0) for r, c in cells)
        mr, mc = marker_cells[0]
        out.append(
            Component(
                bbox=(r0, c0, r1, c1),
                rel_cells=rel_cells,
                marker_rel=(mr - r0, mc - c0),
                color=color,
            )
        )
    return out


def _are_complements(a: Component, b: Component) -> bool:
    if a.shape != b.shape:
        return False
    h, w = a.shape
    full = {(r, c) for r in range(h) for c in range(w)}
    return a.rel_cells.isdisjoint(b.rel_cells) and a.rel_cells | b.rel_cells == full


def _pair_components(comps: list[Component]) -> list[tuple[int, int]]:
    n = len(comps)
    assert n > 0 and n % 2 == 0, "expected an even number of blocks"
    can_pair = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            can_pair[i][j] = can_pair[j][i] = _are_complements(comps[i], comps[j])

    solutions: list[list[tuple[int, int]]] = []

    def backtrack(used: int, acc: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if used == (1 << n) - 1:
            solutions.append(acc.copy())
            return
        i = 0
        while used & (1 << i):
            i += 1
        for j in range(i + 1, n):
            if used & (1 << j):
                continue
            if not can_pair[i][j]:
                continue
            acc.append((i, j))
            backtrack(used | (1 << i) | (1 << j), acc)
            acc.pop()

    backtrack(0, [])
    assert len(solutions) == 1, "complementary pairing must be unique"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary partial rectangles, then place the completed rectangles at the midpoint between each pair and color them by the higher window marker.

    Input structure:
    - bg = black.
    - Several non-black blocks are scattered on the grid; each block occupies part of an axis-aligned rectangle but misses some cells.
    - Blocks come in complementary pairs: within a pair, the two blocks have different base colors, the same bbox size, and one maroon marker pixel each.

    Rule:
    - Detect each foreground block as one 8-connected component and normalize its occupied cells to its own bbox.
    - Pair blocks whose normalized occupied-cell sets are exact complements inside an equal-sized rectangle.
    - For each pair, compare the maroon marker positions inside the bbox; the marker with smaller row wins, and ties are broken by smaller column.
    - Place the completed solid rectangle at the midpoint of the two source bbox centers and fill it with the winning block's base color. Leave every other cell black.

    Key insight:
    - The puzzle requires two separate inferences: first recover pairs by complementary shape masks, then use the relative window-marker positions to choose the output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    comps = _extract_components(g)
    pairs = _pair_components(comps)

    out = np.zeros_like(g)
    placements: list[tuple[int, int, int, int, int]] = []
    for i, j in pairs:
        a = comps[i]
        b = comps[j]
        h, w = a.shape
        winner = a if a.marker_rel <= b.marker_rel else b
        r0 = (a.bbox[0] + b.bbox[0]) // 2
        c0 = (a.bbox[1] + b.bbox[1]) // 2
        placements.append((r0, c0, h, w, winner.color))

    for r0, c0, h, w, color in sorted(placements):
        assert 0 <= r0 <= g.shape[0] - h and 0 <= c0 <= g.shape[1] - w
        region = out[r0 : r0 + h, c0 : c0 + w]
        assert not np.any(region != BG), "output rectangles must not overlap"
        region[:, :] = color
    return out


def _row_col_candidates(mask: np.ndarray) -> tuple[list[tuple[str, tuple[int, int], tuple[int, int]]], list[tuple[str, tuple[int, int], tuple[int, int]]]]:
    cells_a = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    cells_b = [tuple(map(int, cell)) for cell in np.argwhere(~mask)]
    row_candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    col_candidates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    for a in cells_a:
        for b in cells_b:
            if a[0] < b[0]:
                row_candidates.append(("A", a, b))
            elif b[0] < a[0]:
                row_candidates.append(("B", a, b))
            if a[0] == b[0]:
                if a[1] < b[1]:
                    col_candidates.append(("A", a, b))
                elif b[1] < a[1]:
                    col_candidates.append(("B", a, b))
    return row_candidates, col_candidates


def _sample_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    area = h * w
    min_area = max(3, area // 3)
    max_area = min(area - 3, area - area // 3)
    for _ in range(SHAPE_RETRIES):
        p = rng.uniform(0.35, 0.65)
        mask = rng.random((h, w)) < p
        count = int(mask.sum())
        if not (min_area <= count <= max_area):
            continue
        comp = ~mask
        if not _touches_all_sides(mask) or not _touches_all_sides(comp):
            continue
        if not _is_connected(mask, 8) or not _is_connected(comp, 8):
            continue
        if _orth_edge_count(mask) < max(2, min(h, w) - 1):
            continue
        if _orth_edge_count(comp) < max(2, min(h, w) - 1):
            continue
        if h > 2 and w > 2:
            if not mask[1:-1, 1:-1].any():
                continue
            if not comp[1:-1, 1:-1].any():
                continue
        row_candidates, col_candidates = _row_col_candidates(mask)
        if row_candidates and col_candidates:
            return mask
    return None


def _sample_pair_spec(h: int, w: int, color_a: int, color_b: int, rng: np.random.Generator) -> PairSpec | None:
    mask = _sample_mask(h, w, rng)
    if mask is None:
        return None
    row_candidates, col_candidates = _row_col_candidates(mask)
    marker_mode = str(rng.choice(["row", "col"]))
    candidates = row_candidates if marker_mode == "row" else col_candidates
    winner_side, marker_a, marker_b = candidates[int(rng.integers(len(candidates)))]
    if winner_side == "B":
        marker_a, marker_b = marker_a, marker_b
    return PairSpec(
        h=h,
        w=w,
        mask_a=mask,
        marker_a=marker_a,
        marker_b=marker_b,
        color_a=color_a,
        color_b=color_b,
    )


def _box_clear(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    h_total, w_total = reserved.shape
    if r0 < 0 or c0 < 0 or r0 + h > h_total or c0 + w > w_total:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_total, r0 + h + pad)
    cc1 = min(w_total, c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve_box(reserved: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    h_total, w_total = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h_total, r0 + h + pad)
    cc1 = min(w_total, c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _place_specs(specs: list[PairSpec], grid_h: int, grid_w: int, rng: np.random.Generator) -> bool:
    input_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    output_reserved = np.zeros((grid_h, grid_w), dtype=bool)
    for spec in sorted(specs, key=lambda item: item.h * item.w, reverse=True):
        positions = [(r, c) for r in range(grid_h - spec.h + 1) for c in range(grid_w - spec.w + 1)]
        for _ in range(PLACEMENT_RETRIES):
            r1, c1 = positions[int(rng.integers(len(positions)))]
            if not _box_clear(input_reserved, r1, c1, spec.h, spec.w, PAD):
                continue
            r2, c2 = positions[int(rng.integers(len(positions)))]
            if (r1, c1) == (r2, c2):
                continue
            if not _box_clear(input_reserved, r2, c2, spec.h, spec.w, PAD):
                continue
            if max(abs(r1 - r2), abs(c1 - c2)) <= PAD:
                continue
            temp_reserved = input_reserved.copy()
            _reserve_box(temp_reserved, r1, c1, spec.h, spec.w, PAD)
            if not _box_clear(temp_reserved, r2, c2, spec.h, spec.w, PAD):
                continue
            out_r0 = (r1 + r2) // 2
            out_c0 = (c1 + c2) // 2
            if not _box_clear(output_reserved, out_r0, out_c0, spec.h, spec.w, PAD):
                continue

            _reserve_box(input_reserved, r1, c1, spec.h, spec.w, PAD)
            _reserve_box(input_reserved, r2, c2, spec.h, spec.w, PAD)
            _reserve_box(output_reserved, out_r0, out_c0, spec.h, spec.w, PAD)
            spec.pos_a = (r1, c1)
            spec.pos_b = (r2, c2)
            break
        else:
            return False
    return True


def _render_object(
    grid: np.ndarray,
    top_left: tuple[int, int],
    mask: np.ndarray,
    base_color: int,
    marker: tuple[int, int],
) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = base_color
    patch[marker] = MAROON


def _manual_output(specs: list[PairSpec], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        assert spec.pos_a is not None and spec.pos_b is not None
        winner_color = spec.color_a if spec.marker_a <= spec.marker_b else spec.color_b
        r0 = (spec.pos_a[0] + spec.pos_b[0]) // 2
        c0 = (spec.pos_a[1] + spec.pos_b[1]) // 2
        out[r0 : r0 + spec.h, c0 : c0 + spec.w] = winner_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.integers(PAIR_COUNT_MIN, PAIR_COUNT_MAX + 1))
    dim_indices = rng.choice(len(DIM_CANDIDATES), size=pair_count, replace=False)
    dims = [DIM_CANDIDATES[int(idx)] for idx in dim_indices]

    color_order = rng.permutation(np.array(OBJECT_COLORS, dtype=int))
    specs: list[PairSpec] = []
    for index, (h, w) in enumerate(dims):
        color_a = int(color_order[2 * index])
        color_b = int(color_order[2 * index + 1])
        spec = _sample_pair_spec(h, w, color_a, color_b, rng)
        if spec is None:
            return None
        specs.append(spec)

    min_side = max(16, 10 + 3 * pair_count)
    grid_h = int(rng.integers(min_side, GRID_MAX + 1))
    grid_w = int(rng.integers(min_side, GRID_MAX + 1))
    if not _place_specs(specs, grid_h, grid_w, rng):
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in specs:
        assert spec.pos_a is not None and spec.pos_b is not None
        _render_object(grid, spec.pos_a, spec.mask_a, spec.color_a, spec.marker_a)
        _render_object(grid, spec.pos_b, ~spec.mask_a, spec.color_b, spec.marker_b)

    expected = _manual_output(specs, grid.shape)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-window puzzle")
