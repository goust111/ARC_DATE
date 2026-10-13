from __future__ import annotations

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
OBJECT_COUNT_CHOICES = (1, 2, 2, 3, 3)
GAP_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_SIZE_MIN = 5
OBJECT_SIZE_MAX = 10
PLACEMENT_MARGIN = 1
OBJECT_PAD = 2
MAX_PLACEMENT_TRIES = 80
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _square_bounds(center_r: int, center_c: int, size: int) -> tuple[int, int, int, int]:
    if size % 2 == 1:
        top = center_r - size // 2
        left = center_c - size // 2
    else:
        # For even sizes, the square's geometric center is biased down-right of the marker.
        top = center_r - size // 2 + 1
        left = center_c - size // 2 + 1
    return top, left, top + size, left + size


def _square_extents(size: int) -> tuple[int, int]:
    if size % 2 == 1:
        return size // 2, size // 2
    return size // 2 - 1, size // 2


def _analyze_object(g: np.ndarray, cells: list[tuple[int, int]]) -> tuple[int, tuple[int, int], list[tuple[int, int]]]:
    values = [int(g[r, c]) for r, c in cells]
    colors, counts = np.unique(values, return_counts=True)
    color_to_count = {int(color): int(count) for color, count in zip(colors, counts)}
    marker_colors = [color for color, count in color_to_count.items() if count == 1]
    assert len(marker_colors) == 1, "each object must contain one singleton marker color"
    marker_color = int(marker_colors[0])

    marker_cells = [(r, c) for r, c in cells if int(g[r, c]) == marker_color]
    assert len(marker_cells) == 1, "marker color must occur exactly once"
    marker_pos = marker_cells[0]

    r0, c0, r1, c1 = _bbox(cells)
    gaps = [(r, c) for r, c in _boundary_cells(r0, c0, r1, c1) if int(g[r, c]) == BG]
    assert gaps, "each object must have at least one boundary gap"
    return marker_color, marker_pos, gaps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's bounding-box edge gaps with its marker color, then draw a
    marker-colored square whose side length equals the number of filled gaps.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()
    h, w = g.shape

    for cells in _components(g != BG):
        marker_color, (mr, mc), gaps = _analyze_object(g, cells)
        for gr, gc in gaps:
            out[gr, gc] = marker_color

        size = len(gaps)
        top, left, bottom, right = _square_bounds(mr, mc, size)
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)
        out[top:bottom, left:right] = marker_color

    return out


def _render_input(height: int, width: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        obj_h = int(spec["h"])
        obj_w = int(spec["w"])
        body_color = int(spec["body"])
        marker_color = int(spec["marker"])
        mr, mc = spec["marker_pos"]  # type: ignore[misc]
        gaps = spec["gaps"]  # type: ignore[assignment]

        grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = body_color
        for gr, gc in gaps:
            grid[gr, gc] = BG
        grid[int(mr), int(mc)] = marker_color
    return grid


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for spec in specs:
        marker_color = int(spec["marker"])
        mr, mc = spec["marker_pos"]  # type: ignore[misc]
        gaps = spec["gaps"]  # type: ignore[assignment]
        for gr, gc in gaps:
            out[gr, gc] = marker_color
        size = len(gaps)
        top, left, bottom, right = _square_bounds(int(mr), int(mc), size)
        out[max(0, top) : min(h, bottom), max(0, left) : min(w, right)] = marker_color
    return out


def _fits_with_padding(
    r0: int,
    c0: int,
    obj_h: int,
    obj_w: int,
    placed: list[tuple[int, int, int, int]],
    pad: int,
) -> bool:
    r1 = r0 + obj_h - 1
    c1 = c0 + obj_w - 1
    for pr0, pc0, ph, pw in placed:
        pr1 = pr0 + ph - 1
        pc1 = pc0 + pw - 1
        separated = r1 + pad < pr0 or pr1 + pad < r0 or c1 + pad < pc0 or pc1 + pad < c0
        if not separated:
            return False
    return True


def _gap_candidates(obj_h: int, obj_w: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for c in range(1, obj_w - 1):
        candidates.append((0, c))
        candidates.append((obj_h - 1, c))
    for r in range(1, obj_h - 1):
        candidates.append((r, 0))
        candidates.append((r, obj_w - 1))
    return candidates


def _local_object_connected(obj_h: int, obj_w: int, local_gaps: list[tuple[int, int]]) -> bool:
    mask = np.ones((obj_h, obj_w), dtype=bool)
    for gr, gc in local_gaps:
        mask[gr, gc] = False
    return len(_components(mask)) == 1


def _choose_marker_local(
    rng: np.random.Generator, obj_h: int, obj_w: int, gap_count: int
) -> tuple[int, int] | None:
    up, down = _square_extents(gap_count)
    row_min = max(1, up)
    row_max = min(obj_h - 2, obj_h - 1 - down)
    col_min = max(1, up)
    col_max = min(obj_w - 2, obj_w - 1 - down)
    if row_min > row_max or col_min > col_max:
        return None
    local_r = int(rng.integers(row_min, row_max + 1))
    local_c = int(rng.integers(col_min, col_max + 1))
    return local_r, local_c


def _build_specs(
    rng: np.random.Generator, height: int, width: int, object_count: int
) -> list[dict[str, object]] | None:
    if 2 * object_count > 9:
        return None

    palette = rng.choice(np.arange(1, 10), size=2 * object_count, replace=False).tolist()
    specs: list[dict[str, object]] = []
    placed: list[tuple[int, int, int, int]] = []

    for idx in range(object_count):
        body_color = int(palette[2 * idx])
        marker_color = int(palette[2 * idx + 1])
        placed_this_object = False

        for _ in range(MAX_PLACEMENT_TRIES):
            gap_count = int(rng.choice(GAP_COUNT_CHOICES))
            min_side = max(OBJECT_SIZE_MIN, gap_count + 1)
            max_h = min(OBJECT_SIZE_MAX, height - 2 * PLACEMENT_MARGIN)
            max_w = min(OBJECT_SIZE_MAX, width - 2 * PLACEMENT_MARGIN)
            if min_side > max_h or min_side > max_w:
                return None

            obj_h = int(rng.integers(min_side, max_h + 1))
            obj_w = int(rng.integers(min_side, max_w + 1))
            marker_local = _choose_marker_local(rng, obj_h, obj_w, gap_count)
            if marker_local is None:
                continue

            r0 = int(rng.integers(PLACEMENT_MARGIN, height - PLACEMENT_MARGIN - obj_h + 1))
            c0 = int(rng.integers(PLACEMENT_MARGIN, width - PLACEMENT_MARGIN - obj_w + 1))
            if not _fits_with_padding(r0, c0, obj_h, obj_w, placed, OBJECT_PAD):
                continue

            candidates = _gap_candidates(obj_h, obj_w)
            if gap_count > len(candidates):
                continue
            chosen = rng.choice(len(candidates), size=gap_count, replace=False)
            local_gaps = [candidates[int(i)] for i in chosen]
            if not _local_object_connected(obj_h, obj_w, local_gaps):
                continue
            gaps = [(r0 + gr, c0 + gc) for gr, gc in local_gaps]

            local_r, local_c = marker_local
            spec = {
                "r0": r0,
                "c0": c0,
                "h": obj_h,
                "w": obj_w,
                "body": body_color,
                "marker": marker_color,
                "marker_pos": (r0 + local_r, c0 + local_c),
                "gaps": gaps,
            }
            specs.append(spec)
            placed.append((r0, c0, obj_h, obj_w))
            placed_this_object = True
            break

        if not placed_this_object:
            return None

    return specs


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    width = int(rng.integers(GRID_SIDE_MIN, GRID_SIDE_MAX + 1))
    object_count = int(rng.choice(OBJECT_COUNT_CHOICES))

    specs = _build_specs(rng, height, width, object_count)
    if specs is None:
        return None

    grid = _render_input(height, width, specs)
    expected = _expected_output(grid, specs)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
