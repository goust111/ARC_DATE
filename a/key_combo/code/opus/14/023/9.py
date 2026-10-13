from __future__ import annotations

import numpy as np


BG = 0
DIVIDER = 5
GRID_PARTS = 3
AREA_VALUES = tuple(range(2, 11))
CELL_COLORS = tuple(range(1, 10))
CELL_SIZE_CHOICES = (7, 8, 9)
MAX_GENERATION_ATTEMPTS = 500


def _pattern_to_coords(pattern: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    coords: list[tuple[int, int]] = []
    for r, row in enumerate(pattern):
        for c, ch in enumerate(row):
            if ch == "#":
                coords.append((r, c))
    return tuple(coords)


TEMPLATE_LIBRARY: dict[int, tuple[tuple[str, tuple[tuple[int, int], ...]], ...]] = {
    2: (("path", _pattern_to_coords(("##",))),),
    3: (
        ("path", _pattern_to_coords(("###",))),
        ("branch", _pattern_to_coords(("##", "#.",))),
    ),
    4: (
        ("compact", _pattern_to_coords(("##", "##"))),
        ("path", _pattern_to_coords(("####",))),
        ("branch", _pattern_to_coords((".#.", "###"))),
    ),
    5: (
        ("compact", _pattern_to_coords(("##.", "###"))),
        ("path", _pattern_to_coords(("#####",))),
        ("branch", _pattern_to_coords((".#.", "###", ".#."))),
    ),
    6: (
        ("compact", _pattern_to_coords(("###", "###"))),
        ("path", _pattern_to_coords(("####", "##.."))),
        ("branch", _pattern_to_coords((".#.", "###", ".##"))),
    ),
    7: (
        ("compact", _pattern_to_coords(("###", "###", "#.."))),
        ("path", _pattern_to_coords(("###", "..#", "###"))),
        ("branch", _pattern_to_coords(("###", ".#.", "###"))),
    ),
    8: (
        ("compact", _pattern_to_coords(("###", "###", "##."))),
        ("path", _pattern_to_coords(("####", "...#", "###."))),
        ("branch", _pattern_to_coords(("###", ".#.", "###", ".#."))),
    ),
    9: (
        ("compact", _pattern_to_coords(("###", "###", "###"))),
        ("path", _pattern_to_coords(("####", "...#", "####"))),
        ("branch", _pattern_to_coords(("###", ".#.", "###", ".#.", "#.."))),
    ),
    10: (
        ("compact", _pattern_to_coords(("#####", "#####"))),
        ("path", _pattern_to_coords(("#####", "...##", "###.."))),
        ("branch", _pattern_to_coords(("####", "###.", "###."))),
    ),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _transform(coords: tuple[tuple[int, int], ...], rotation: int, reflect: bool) -> tuple[tuple[int, int], ...]:
    transformed = list(coords)
    if reflect:
        transformed = [(r, -c) for r, c in transformed]
    for _ in range(rotation % 4):
        transformed = [(c, -r) for r, c in transformed]
    return _normalize(tuple(transformed))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in coords) + 1, max(c for _, c in coords) + 1


def _connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if points.size == 0:
        return False
    start = tuple(map(int, points[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _extract_cells(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    divider_rows = [r for r in range(g.shape[0]) if bool(np.all(g[r, :] == DIVIDER))]
    divider_cols = [c for c in range(g.shape[1]) if bool(np.all(g[:, c] == DIVIDER))]
    assert len(divider_rows) == GRID_PARTS - 1
    assert len(divider_cols) == GRID_PARTS - 1

    row_edges = [-1] + divider_rows + [g.shape[0]]
    col_edges = [-1] + divider_cols + [g.shape[1]]
    row_spans = [(row_edges[i] + 1, row_edges[i + 1]) for i in range(GRID_PARTS)]
    col_spans = [(col_edges[i] + 1, col_edges[i + 1]) for i in range(GRID_PARTS)]
    row_sizes = {r1 - r0 for r0, r1 in row_spans}
    col_sizes = {c1 - c0 for c0, c1 in col_spans}
    assert len(row_sizes) == 1 and len(col_sizes) == 1

    stats: list[dict[str, int]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            block = g[r0:r1, c0:c1]
            fg_mask = block != BG
            assert bool(fg_mask.any()), "each cell must contain one foreground shape"
            colors = np.unique(block[fg_mask])
            assert colors.size == 1, "each cell must contain a single-color shape"
            assert _connected(fg_mask), "each cell shape must be 4-connected"

            coords = np.argwhere(fg_mask)
            rr0, cc0 = coords.min(axis=0)
            rr1, cc1 = coords.max(axis=0)
            stats.append(
                {
                    "color": int(colors[0]),
                    "area": int(coords.shape[0]),
                    "bbox_area": int((rr1 - rr0 + 1) * (cc1 - cc0 + 1)),
                }
            )
    assert len(stats) == GRID_PARTS * GRID_PARTS
    return stats


def _ranks(values: list[int]) -> list[int]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    assert len({values[idx] for idx in order}) == len(values)
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _stable_ranks(values: list[int]) -> list[int]:
    order = sorted(range(len(values)), key=lambda idx: (values[idx], idx))
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _expected_output(colors: list[int], areas: list[int]) -> np.ndarray:
    ranks = _ranks(areas)
    out = np.empty((GRID_PARTS, GRID_PARTS), dtype=int)
    for idx, rank in enumerate(ranks):
        out[idx // GRID_PARTS, idx % GRID_PARTS] = int(colors[rank - 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each partition cell by permuting the read-order color list with area ranks.

    Input structure:
    - A black background is partitioned by full gray divider lines into a 3x3 grid of equal-sized cells.
    - Each cell contains exactly one single-color 4-connected shape on black.
    - The nine cell shapes have pairwise distinct areas, while their colors provide a row-major palette.

    Rule:
    - Read the nine cell colors in row-major order to form a color list.
    - Measure the area of each cell's shape and rank the cells from smallest area to largest area.
    - Output a plain 3x3 grid with no dividers, where each cell receives the color whose row-major index equals that cell's area rank.

    Key insight:
    - Each cell contributes two independent signals: its visible color enters the palette, but only its shape area determines which palette entry is reassigned back to that position.
    """
    stats = _extract_cells(np.asarray(grid, dtype=int))
    colors = [item["color"] for item in stats]
    areas = [item["area"] for item in stats]
    return _expected_output(colors, areas)


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _choose_offset(slack: int) -> int:
    if slack >= 2:
        return _randint(1, slack - 1)
    return _randint(0, slack)


def _paint_shape(
    grid: np.ndarray,
    cell_row: int,
    cell_col: int,
    cell_size: int,
    coords: tuple[tuple[int, int], ...],
    color: int,
) -> None:
    height, width = _shape_dims(coords)
    r_off = _choose_offset(cell_size - height)
    c_off = _choose_offset(cell_size - width)
    base_r = cell_row * (cell_size + 1)
    base_c = cell_col * (cell_size + 1)
    for r, c in coords:
        grid[base_r + r_off + r, base_c + c_off + c] = int(color)


def _bbox_rank_shortcut_matches(stats: list[dict[str, int]]) -> bool:
    colors = [item["color"] for item in stats]
    bbox_ranks = _stable_ranks([item["bbox_area"] for item in stats])
    bbox_out = np.empty((GRID_PARTS, GRID_PARTS), dtype=int)
    for idx, rank in enumerate(bbox_ranks):
        bbox_out[idx // GRID_PARTS, idx % GRID_PARTS] = int(colors[rank - 1])
    return bool(np.array_equal(bbox_out, _expected_output(colors, [item["area"] for item in stats])))


def _has_bbox_area_inversion(stats: list[dict[str, int]]) -> bool:
    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            if stats[i]["area"] < stats[j]["area"] and stats[i]["bbox_area"] > stats[j]["bbox_area"]:
                return True
            if stats[j]["area"] < stats[i]["area"] and stats[j]["bbox_area"] > stats[i]["bbox_area"]:
                return True
    return False


def _semantic_ok(grid: np.ndarray, chosen_modes: set[str]) -> bool:
    try:
        stats = _extract_cells(grid)
    except AssertionError:
        return False

    colors = [item["color"] for item in stats]
    areas = [item["area"] for item in stats]
    ranks = _ranks(areas)
    perm = [rank - 1 for rank in ranks]
    expected = _expected_output(colors, areas)

    if set(colors) != set(CELL_COLORS):
        return False
    if sorted(areas) != list(AREA_VALUES):
        return False
    if chosen_modes != {"compact", "path", "branch"}:
        return False
    if sum(perm[idx] != idx for idx in range(len(perm))) < 6:
        return False
    if all(perm[perm[idx]] == idx for idx in range(len(perm))):
        return False
    if np.array_equal(expected, np.array(colors, dtype=int).reshape(GRID_PARTS, GRID_PARTS)):
        return False

    sorted_by_area = np.array([colors[idx] for idx in np.argsort(np.array(areas))], dtype=int).reshape(GRID_PARTS, GRID_PARTS)
    if np.array_equal(expected, sorted_by_area):
        return False
    if not _has_bbox_area_inversion(stats):
        return False
    if _bbox_rank_shortcut_matches(stats):
        return False

    try:
        pred = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(pred, expected))


def _gen_attempt() -> np.ndarray | None:
    cell_size = int(np.random.choice(np.array(CELL_SIZE_CHOICES, dtype=int)))
    full_size = GRID_PARTS * cell_size + (GRID_PARTS - 1)
    grid = np.full((full_size, full_size), BG, dtype=int)

    for divider_index in range(1, GRID_PARTS):
        pos = divider_index * (cell_size + 1) - 1
        grid[pos, :] = DIVIDER
        grid[:, pos] = DIVIDER

    colors = list(map(int, np.random.permutation(np.array(CELL_COLORS, dtype=int))))
    areas = list(map(int, np.random.permutation(np.array(AREA_VALUES, dtype=int))))

    chosen_modes: set[str] = set()
    for idx, area in enumerate(areas):
        template_mode, template_coords = TEMPLATE_LIBRARY[area][_randint(0, len(TEMPLATE_LIBRARY[area]) - 1)]
        transformed = _transform(template_coords, rotation=_randint(0, 3), reflect=bool(_randint(0, 1)))
        chosen_modes.add(template_mode)
        _paint_shape(
            grid,
            cell_row=idx // GRID_PARTS,
            cell_col=idx % GRID_PARTS,
            cell_size=cell_size,
            coords=transformed,
            color=colors[idx],
        )

    if _semantic_ok(grid, chosen_modes):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
