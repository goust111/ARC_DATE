from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
ORTHOGONALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("tl", "tr", "bl", "br")
MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _corner_coord(shape: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = shape
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, w - 1
    if name == "bl":
        return h - 1, 0
    if name == "br":
        return h - 1, w - 1
    raise ValueError(f"unknown corner name: {name}")


def _legend_info(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    hits = []
    for name in CORNERS:
        r, c = _corner_coord(grid.shape, name)
        value = int(grid[r, c])
        if value != BACKGROUND:
            hits.append(((r, c), value))
    assert len(hits) == 1, f"expected exactly one non-background corner legend, got {hits}"
    return hits[0]


def _expand_singletons(grid: np.ndarray) -> np.ndarray:
    out = np.full_like(grid, BACKGROUND)
    for r, c in np.argwhere(grid != BACKGROUND):
        color = int(grid[r, c])
        for rr, cc in ((r - 1, c - 1), (r - 1, c), (r, c - 1), (r, c)):
            assert _in_bounds(grid.shape, int(rr), int(cc)), "expanded block must stay in bounds"
            out[int(rr), int(cc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand interior singleton pixels into 2x2 blocks, add diagonal-only halo cells, then draw a matching canvas border.

    Input structure:
    - The input is a black background grid with one non-black legend pixel in exactly one corner.
    - Away from the border, there are 3 to 5 isolated non-background singleton pixels sharing one object role.
    - The corner legend color is distinct from the singleton object color and only supplies the halo/border color.

    Rule:
    - Erase the corner legend back to black.
    - Expand every remaining non-background singleton into a 2x2 block anchored up-left from the original pixel.
    - On the expanded grid, recolor each black cell to the legend color iff it is diagonally adjacent to at least one block cell and orthogonally adjacent to none.
    - Finally overwrite the whole outer canvas border with the same legend color.

    Key insight: The corner pixel is a color key, while the true geometry comes from applying diagonal-only adjacency to the expanded 2x2 blocks rather than to the original singletons.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_pos, halo_color = _legend_info(g)
    work = g.copy()
    work[legend_pos] = BACKGROUND

    expanded = _expand_singletons(work)
    out = expanded.copy()
    h, w = out.shape

    for r in range(h):
        for c in range(w):
            if int(expanded[r, c]) != BACKGROUND:
                continue
            has_diagonal = any(
                _in_bounds(expanded.shape, r + dr, c + dc) and int(expanded[r + dr, c + dc]) != BACKGROUND
                for dr, dc in DIAGONALS
            )
            if not has_diagonal:
                continue
            has_orthogonal = any(
                _in_bounds(expanded.shape, r + dr, c + dc) and int(expanded[r + dr, c + dc]) != BACKGROUND
                for dr, dc in ORTHOGONALS
            )
            if not has_orthogonal:
                out[r, c] = halo_color

    out[0, :] = halo_color
    out[-1, :] = halo_color
    out[:, 0] = halo_color
    out[:, -1] = halo_color
    return out


def _bbox_footprint(pixel: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = pixel
    return r - 2, c - 2, r + 1, c + 1


def _footprints_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0, ar1, ac1 = _bbox_footprint(a)
    br0, bc0, br1, bc1 = _bbox_footprint(b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _valid_singleton_positions(points: list[tuple[int, int]]) -> bool:
    for i, (r0, c0) in enumerate(points):
        for r1, c1 in points[i + 1 :]:
            if abs(r0 - r1) + abs(c0 - c1) < 5:
                return False
            if _footprints_overlap((r0, c0), (r1, c1)):
                return False
    return True


def _sample_positions(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(3, h - 2) for c in range(3, w - 2)]
    if len(candidates) < count:
        return None

    for _ in range(200):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        for cell in candidates:
            if all(
                abs(cell[0] - other[0]) + abs(cell[1] - other[1]) >= 5 and not _footprints_overlap(cell, other)
                for other in chosen
            ):
                chosen.append(cell)
                if len(chosen) == count:
                    return chosen
    return None


def _check_input(grid: np.ndarray, object_color: int, halo_color: int, count: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        legend_pos, legend_value = _legend_info(grid)
    except AssertionError:
        return False
    if int(legend_value) != int(halo_color):
        return False

    pixels = [(int(r), int(c)) for r, c in np.argwhere(grid == object_color)]
    if len(pixels) != count:
        return False
    if int(np.count_nonzero((grid != BACKGROUND) & (grid != object_color) & (grid != halo_color))) != 0:
        return False
    if int(np.count_nonzero(grid == halo_color)) != 1:
        return False
    if legend_pos not in {_corner_coord(grid.shape, name) for name in CORNERS}:
        return False
    for r, c in pixels:
        if r < 3 or r > grid.shape[0] - 3 or c < 3 or c > grid.shape[1] - 3:
            return False
    if not _valid_singleton_positions(pixels):
        return False

    out = main(grid)
    border_cells = 2 * grid.shape[0] + 2 * grid.shape[1] - 4
    if int(np.count_nonzero(out == object_color)) != 4 * count:
        return False
    if int(np.count_nonzero(out == halo_color)) != border_cells + 4 * count:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(10, 18)
        w = random.randint(10, 18)
        count = random.randint(3, 5)
        palette = list(range(1, 10))
        object_color, halo_color = random.sample(palette, 2)
        corner_name = random.choice(CORNERS)

        positions = _sample_positions(h, w, count)
        if positions is None:
            continue

        grid = np.full((h, w), BACKGROUND, dtype=int)
        for r, c in positions:
            grid[r, c] = object_color
        lr, lc = _corner_coord((h, w), corner_name)
        grid[lr, lc] = halo_color

        if _check_input(grid, object_color, halo_color, count):
            return grid

    raise RuntimeError("generate_input: exhausted attempts")
