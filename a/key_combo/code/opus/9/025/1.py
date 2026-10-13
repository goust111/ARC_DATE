from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SOURCE_COLORS = (BLUE, RED, GREEN)
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
QUADRANTS = ("TL", "TR", "BL", "BR")
CORNER_CELLS = {
    "TL": lambda n: (0, 0),
    "TR": lambda n: (0, n - 1),
    "BL": lambda n: (n - 1, 0),
    "BR": lambda n: (n - 1, n - 1),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _assert_valid_sources(sources: list[tuple[int, int, int]]) -> None:
    rows = set()
    cols = set()
    main_diags = set()
    anti_diags = set()
    for r, c, color in sources:
        assert color in SOURCE_COLORS
        assert r not in rows
        assert c not in cols
        assert (r - c) not in main_diags
        assert (r + c) not in anti_diags
        rows.add(r)
        cols.add(c)
        main_diags.add(r - c)
        anti_diags.add(r + c)


def _build_owner_map(
    shape: tuple[int, int], sources: list[tuple[int, int, int]]
) -> dict[tuple[int, int], set[int]]:
    owners: dict[tuple[int, int], set[int]] = defaultdict(set)
    for idx, (r, c, _color) in enumerate(sources):
        for dr, dc in DIAGONALS:
            rr, cc = r, c
            while _in_bounds(shape, rr, cc):
                owners[(rr, cc)].add(idx)
                rr += dr
                cc += dc
    return owners


def _quadrant_of(r: int, c: int, mid: int) -> str | None:
    if r == mid or c == mid:
        return None
    if r < mid and c < mid:
        return "TL"
    if r < mid and c > mid:
        return "TR"
    if r > mid and c < mid:
        return "BL"
    return "BR"


def _analyze_sources(
    shape: tuple[int, int], sources: list[tuple[int, int, int]]
) -> dict[str, object]:
    owners = _build_owner_map(shape, sources)
    h, w = shape
    assert h == w and h % 2 == 1
    mid = h // 2

    intersections = {cell for cell, srcs in owners.items() if len(srcs) >= 2}
    counts = {name: 0 for name in QUADRANTS}
    counted_intersections: dict[str, set[tuple[int, int]]] = {name: set() for name in QUADRANTS}
    centerline_intersections = set()
    source_any_hits = [0] * len(sources)
    source_counted_hits = [0] * len(sources)

    for cell in intersections:
        srcs = owners[cell]
        for idx in srcs:
            source_any_hits[idx] += 1
        quadrant = _quadrant_of(cell[0], cell[1], mid)
        if quadrant is None:
            centerline_intersections.add(cell)
            continue
        counts[quadrant] += 1
        counted_intersections[quadrant].add(cell)
        for idx in srcs:
            source_counted_hits[idx] += 1

    return {
        "owners": owners,
        "intersections": intersections,
        "counted_intersections": counted_intersections,
        "centerline_intersections": centerline_intersections,
        "counts": counts,
        "source_any_hits": source_any_hits,
        "source_counted_hits": source_counted_hits,
    }


def _corner_distances(n: int, corner: str, r: int, c: int) -> tuple[int, int]:
    if corner == "TL":
        return r, c
    if corner == "TR":
        return r, n - 1 - c
    if corner == "BL":
        return n - 1 - r, c
    return n - 1 - r, n - 1 - c


def _paint_corner_triangle(out: np.ndarray, corner: str, side: int) -> None:
    if side <= 0:
        return
    n = out.shape[0]
    for r in range(n):
        for c in range(n):
            dr, dc = _corner_distances(n, corner, r, c)
            if dr + dc < side:
                out[r, c] = ORANGE


def _triangle_mask(n: int, corner: str, side: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    if side <= 0:
        return mask
    for r in range(n):
        for c in range(n):
            dr, dc = _corner_distances(n, corner, r, c)
            if dr + dc < side:
                mask[r, c] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw multi-source diagonal rays, count their quadrant-wise intersections, and replace the corner markers with orange corner triangles sized by those counts.

    Input structure:
    - A square odd-sized black grid contains exactly four grey corner markers, one in each corner.
    - The remaining foreground consists of 3 to 5 isolated source pixels colored blue, red, or green.
    - Every source pixel is at least two cells away from the border, and no two sources share a row, column, or diagonal.

    Rule:
    - From each source pixel, extend rays in all four diagonal directions to the grid boundary using the source color.
    - A cell is an intersection if rays from at least two different source pixels pass through it; repaint every such cell yellow.
    - Split the grid by the unique center row and center column, count yellow intersections separately in the four quadrants, and ignore intersections lying on the center row or center column.
    - Erase the four grey corner markers back to black, then draw an orange solid right triangle in each corner whose side length equals that corner quadrant's intersection count.

    Key insight:
    - The important objects are not the input pixels themselves but the emergent overlap cells created only after all diagonal rays from all sources are traced and compared by source identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    for corner in QUADRANTS:
        assert int(g[CORNER_CELLS[corner](h)]) == GREY

    sources = []
    for r, c in np.argwhere(g != BLACK):
        rr, cc = int(r), int(c)
        if (rr, cc) in {CORNER_CELLS[name](h) for name in QUADRANTS}:
            continue
        color = int(g[rr, cc])
        assert color in SOURCE_COLORS
        assert 2 <= rr <= h - 3 and 2 <= cc <= w - 3
        sources.append((rr, cc, color))

    assert 3 <= len(sources) <= 5
    _assert_valid_sources(sources)
    analysis = _analyze_sources(g.shape, sources)
    counts = analysis["counts"]
    owners = analysis["owners"]
    intersections = analysis["intersections"]

    out = np.full_like(g, BLACK)
    for cell, srcs in owners.items():
        if len(srcs) == 1:
            src = sources[next(iter(srcs))]
            out[cell] = int(src[2])
    for cell in intersections:
        out[cell] = YELLOW

    for corner in QUADRANTS:
        out[CORNER_CELLS[corner](h)] = BLACK
        _paint_corner_triangle(out, corner, int(counts[corner]))
    return out


GEN_SIZE_CHOICES = (13, 15, 17, 19)
GEN_SOURCE_COUNT_CHOICES = (3, 4, 5)
GEN_SOURCE_COUNT_WEIGHTS = (4, 3, 1)
GEN_MODE_CHOICES = ("four_quadrants", "three_quadrants", "centerline")
GEN_MODE_WEIGHTS = (4, 3, 2)
GEN_MAX_ATTEMPTS = 4000
GEN_MAX_TRIANGLE_SIDE = 6
GEN_MAX_TOTAL_INTERSECTIONS = 9


def _sample_colors(k: int) -> list[int]:
    palette_size = random.choices([1, 2, 3], weights=[1, 4, 3], k=1)[0]
    palette = random.sample(list(SOURCE_COLORS), k=palette_size)
    return [random.choice(palette) for _ in range(k)]


def _sample_sources(n: int, k: int) -> list[tuple[int, int, int]] | None:
    positions: list[tuple[int, int]] = []
    used_rows = set()
    used_cols = set()
    used_main = set()
    used_anti = set()

    candidates = [
        (r, c)
        for r in range(2, n - 2)
        for c in range(2, n - 2)
        if (r, c) != (n // 2, n // 2)
    ]
    random.shuffle(candidates)

    for r, c in candidates:
        if r in used_rows or c in used_cols or (r - c) in used_main or (r + c) in used_anti:
            continue
        if any(max(abs(r - rr), abs(c - cc)) < 2 for rr, cc in positions):
            continue
        positions.append((r, c))
        used_rows.add(r)
        used_cols.add(c)
        used_main.add(r - c)
        used_anti.add(r + c)
        if len(positions) == k:
            break

    if len(positions) != k:
        return None

    colors = _sample_colors(k)
    return [(r, c, color) for (r, c), color in zip(positions, colors)]


def _passes_semantic_checks(
    n: int,
    sources: list[tuple[int, int, int]],
    mode: str,
) -> bool:
    analysis = _analyze_sources((n, n), sources)
    counts = analysis["counts"]
    nonzero_quadrants = sum(int(v > 0) for v in counts.values())
    total_counted = sum(counts.values())
    centerline_intersections = analysis["centerline_intersections"]
    source_any_hits = analysis["source_any_hits"]
    source_counted_hits = analysis["source_counted_hits"]

    if any(v == 0 for v in source_any_hits):
        return False
    if max(counts.values()) > min(GEN_MAX_TRIANGLE_SIDE, n // 2):
        return False
    if total_counted < 3 or total_counted > GEN_MAX_TOTAL_INTERSECTIONS:
        return False
    if nonzero_quadrants < 3:
        return False
    if sum(int(v > 0) for v in source_counted_hits) < max(2, len(sources) - 1):
        return False
    if len(set(counts.values())) == 1 and next(iter(counts.values())) == 0:
        return False

    if mode == "four_quadrants" and nonzero_quadrants != 4:
        return False
    if mode == "three_quadrants" and nonzero_quadrants != 3:
        return False
    if mode == "centerline" and (len(centerline_intersections) == 0 or nonzero_quadrants < 3):
        return False

    out = main(_render_input_grid(n, sources))
    for corner in QUADRANTS:
        mask = _triangle_mask(n, corner, int(counts[corner]))
        if counts[corner] == 0:
            if int(out[CORNER_CELLS[corner](n)]) == ORANGE:
                return False
            continue
        if not np.all(out[mask] == ORANGE):
            return False
    if np.all(out == BLACK):
        return False
    return True


def _render_input_grid(n: int, sources: list[tuple[int, int, int]]) -> np.ndarray:
    grid = np.zeros((n, n), dtype=int)
    for corner in QUADRANTS:
        grid[CORNER_CELLS[corner](n)] = GREY
    for r, c, color in sources:
        grid[r, c] = int(color)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        n = random.choice(GEN_SIZE_CHOICES)
        k = random.choices(GEN_SOURCE_COUNT_CHOICES, weights=GEN_SOURCE_COUNT_WEIGHTS, k=1)[0]
        mode = random.choices(GEN_MODE_CHOICES, weights=GEN_MODE_WEIGHTS, k=1)[0]
        sources = _sample_sources(n, k)
        if sources is None:
            continue
        if not _passes_semantic_checks(n, sources, mode):
            continue
        grid = _render_input_grid(n, sources)
        analysis = _analyze_sources((n, n), sources)
        counts = analysis["counts"]
        if max(counts.values()) == 0:
            continue
        assert np.array_equal(main(grid), main(grid.copy()))
        return grid
    raise RuntimeError("failed to generate a valid multi-source diagonal-ray ARC input")
