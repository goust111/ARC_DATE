from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
RETRY_LIMIT = 400
SIZE_CHOICES = (11, 13, 15)
CORNER_NAMES = ("TL", "TR", "BL", "BR")
CORNER_COORDS = {
    "TL": lambda h, w: (0, 0),
    "TR": lambda h, w: (0, w - 1),
    "BL": lambda h, w: (h - 1, 0),
    "BR": lambda h, w: (h - 1, w - 1),
}


def _all_corners(h: int, w: int) -> list[tuple[int, int]]:
    return [fn(h, w) for fn in CORNER_COORDS.values()]


def _corner_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in _all_corners(h, w):
        mask[r, c] = True
    return mask


def _parse_grid(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], tuple[int, int], np.ndarray, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    corners = _all_corners(h, w)
    corner_set = set(corners)

    markers: list[tuple[int, int, int]] = []
    for r, c in corners:
        color = int(g[r, c])
        if color != BG:
            markers.append((r, c, color))

    assert 2 <= len(markers) <= 3, "expected 2-3 corner markers"
    assert len({color for _, _, color in markers}) == len(markers), "marker colors must be distinct"

    nonzero = np.argwhere(g != BG)
    inner = [(int(r), int(c)) for r, c in nonzero if (int(r), int(c)) not in corner_set]
    assert len(inner) == 4, "expected exactly one 2x2 legend block outside corners"

    rows = [r for r, _ in inner]
    cols = [c for _, c in inner]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 == r0 + 1 and c1 == c0 + 1, "legend must be a full 2x2 block"
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            assert int(g[rr, cc]) != BG, "legend must fill its entire 2x2 box"

    legend = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    pair_map: dict[int, int] = {}
    for rr in range(2):
        left = int(legend[rr, 0])
        right = int(legend[rr, 1])
        assert left != BG and right != BG and left != right, "each legend row must contain a color pair"
        assert left not in pair_map and right not in pair_map, "legend colors must be unique"
        pair_map[left] = right
        pair_map[right] = left

    marker_colors = {color for _, _, color in markers}
    assert marker_colors.issubset(pair_map), "every marker color must appear in the legend"
    return markers, (r0, c0), legend, pair_map


def _compute_fields(
    shape: tuple[int, int], markers: list[tuple[int, int, int]], pair_map: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    owners = np.full((h, w), -1, dtype=int)
    parity = np.full((h, w), -1, dtype=int)
    fill = np.zeros((h, w), dtype=int)

    for r in range(h):
        for c in range(w):
            dists = [(r - mr) * (r - mr) + (c - mc) * (c - mc) for mr, mc, _ in markers]
            best = min(dists)
            if dists.count(best) != 1:
                continue
            owner = dists.index(best)
            mr, mc, color = markers[owner]
            ring = max(abs(r - mr), abs(c - mc))
            owners[r, c] = owner
            parity[r, c] = ring & 1
            fill[r, c] = color if parity[r, c] == 0 else pair_map[color]
    return owners, parity, fill


def _solve(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    markers, (r0, c0), legend, pair_map = _parse_grid(g)
    owners, parity, fill = _compute_fields(g.shape, markers, pair_map)
    out = fill.copy()
    out[r0 : r0 + 2, c0 : c0 + 2] = legend
    return out, owners, parity, fill


def main(grid: np.ndarray) -> np.ndarray:
    """Fill nearest-corner square rings, alternating between the marker color and its legend row-mate.

    Input structure:
    - A black square grid contains 2-3 single-pixel corner markers with distinct colors.
    - A separate 2x2 legend block sits away from the corners; each legend row is a color pair.
    - Every corner-marker color appears in exactly one legend row.

    Rule:
    - For each non-legend, non-marker cell, assign it to the uniquely nearest corner marker by squared Euclidean distance; ties stay black.
    - Let d be that cell's Chebyshev distance to its assigned marker.
    - If d is even, paint the cell with the marker's original color.
    - If d is odd, find the legend row containing the marker color and use the other color from that row.
    - Keep the legend block unchanged, overlaying the ring pattern underneath it.

    Key insight:
    - The corners define clipped Voronoi ownership, while the legend converts odd Chebyshev rings into paired replacement colors instead of leaving them blank.
    """
    out, _owners, _parity, _fill = _solve(grid)
    return out


def _build_uniform_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    legend_pos: tuple[int, int],
    legend: np.ndarray,
    pair_map: dict[int, int],
    mode: str,
) -> np.ndarray:
    owners, parity, _fill = _compute_fields(grid.shape, markers, pair_map)
    out = np.zeros_like(grid)
    for idx, (_, _, color) in enumerate(markers):
        owned = owners == idx
        if mode == "original":
            out[owned] = color
        elif mode == "swapped":
            out[owned] = pair_map[color]
        elif mode == "even_black":
            out[owned & (parity == 0)] = color
        else:
            raise ValueError(mode)
    r0, c0 = legend_pos
    out[r0 : r0 + 2, c0 : c0 + 2] = legend
    return out


def _has_shared_boundary(owners: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = owners.shape
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            nbr_owners = set()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and owners[nr, nc] >= 0:
                    nbr_owners.add(int(owners[nr, nc]))
            if owners[r, c] >= 0:
                nbr_owners.add(int(owners[r, c]))
            if len(nbr_owners) >= 2:
                return True
    return False


def _is_valid_generated(grid: np.ndarray) -> bool:
    try:
        markers, legend_pos, legend, pair_map = _parse_grid(grid)
        out, owners, parity, fill = _solve(grid)
    except Exception:
        return False

    h, w = grid.shape
    r0, c0 = legend_pos
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[r0 : r0 + 2, c0 : c0 + 2] = True
    corner_mask = _corner_mask(grid.shape)
    work_mask = ~(legend_mask | corner_mask)

    if np.array_equal(out, grid):
        return False

    if not np.any((owners < 0) & work_mask):
        return False

    if not _has_shared_boundary(owners, legend_mask):
        return False

    for idx in range(len(markers)):
        owned = (owners == idx) & work_mask
        if not np.any(owned & (parity == 0)):
            return False
        if not np.any(owned & (parity == 1)):
            return False

    if np.all(fill[legend_mask] == grid[legend_mask]):
        return False

    odd_colors = set(map(int, fill[(parity == 1) & work_mask]))
    odd_colors.discard(BG)
    if len(odd_colors) < 2:
        return False

    nonzero_out = set(map(int, np.unique(out))) - {BG}
    if nonzero_out != set(pair_map):
        return False

    original_only = _build_uniform_output(grid, markers, legend_pos, legend, pair_map, "original")
    swapped_only = _build_uniform_output(grid, markers, legend_pos, legend, pair_map, "swapped")
    even_black = _build_uniform_output(grid, markers, legend_pos, legend, pair_map, "even_black")
    if np.array_equal(out, original_only):
        return False
    if np.array_equal(out, swapped_only):
        return False
    if np.array_equal(out, even_black):
        return False

    try:
        if not np.array_equal(main(grid), out):
            return False
    except Exception:
        return False
    return True


def _sample_corners(marker_count: int) -> list[str]:
    return random.sample(CORNER_NAMES, marker_count)


def _build_legend_rows(marker_colors: list[int], other_colors: list[int]) -> np.ndarray:
    if len(marker_colors) == 2:
        rows = [
            [marker_colors[0], other_colors[0]],
            [marker_colors[1], other_colors[1]],
        ]
    else:
        shared = random.sample(marker_colors, 2)
        remaining = [color for color in marker_colors if color not in shared][0]
        rows = [
            shared,
            [remaining, other_colors[0]],
        ]

    random.shuffle(rows)
    for row in rows:
        if random.random() < 0.5:
            row[0], row[1] = row[1], row[0]
    return np.array(rows, dtype=int)


def _sample_legend_pos(n: int) -> tuple[int, int]:
    mid = n // 2
    row_candidates = [r for r in range(max(2, mid - 3), min(n - 2, mid + 2))]
    col_candidates = [c for c in range(max(2, mid - 3), min(n - 2, mid + 2))]
    return random.choice(row_candidates), random.choice(col_candidates)


def _generate_attempt() -> np.ndarray | None:
    n = random.choice(SIZE_CHOICES)
    marker_count = random.choice((2, 3))
    corner_names = _sample_corners(marker_count)
    colors = random.sample(PALETTE, 4)
    marker_colors = colors[:marker_count]
    other_colors = colors[marker_count:]
    legend = _build_legend_rows(marker_colors, other_colors)
    legend_r0, legend_c0 = _sample_legend_pos(n)

    grid = np.zeros((n, n), dtype=int)
    for name, color in zip(corner_names, random.sample(marker_colors, len(marker_colors))):
        r, c = CORNER_COORDS[name](n, n)
        grid[r, c] = color
    grid[legend_r0 : legend_r0 + 2, legend_c0 : legend_c0 + 2] = legend

    if not _is_valid_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
