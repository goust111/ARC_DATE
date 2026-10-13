from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
MARKER_COLOR = 3
CODE_COLORS = (1, 2, 4, 7)
MAX_GENERATION_ATTEMPTS = 400
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _grid_to_lattice_coord(coord: int) -> int:
    return (int(coord) - 1) // 2


def _lattice_to_grid_coord(idx: int) -> int:
    return 1 + 2 * int(idx)


def _corner_lattice_coord(i: int, j: int, corner: int, lattice_n: int) -> tuple[int, int]:
    if corner in (1, 3):
        j = lattice_n - 1 - int(j)
    if corner in (2, 3):
        i = lattice_n - 1 - int(i)
    return _lattice_to_grid_coord(i), _lattice_to_grid_coord(j)


def _decode_corner_code(
    grid: np.ndarray,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int, int]], dict[int, int], dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    n, m = g.shape
    assert n == m and n % 2 == 1 and n >= 9

    bg = _get_bg(g)
    marker_coords = np.argwhere(g == MARKER_COLOR)
    assert marker_coords.shape[0] == 1
    center = (int(marker_coords[0, 0]), int(marker_coords[0, 1]))
    assert center == (n // 2, n // 2)

    code_pixels = []
    for r, c in np.argwhere((g != bg) & (g != MARKER_COLOR)):
        rr = int(r)
        cc = int(c)
        assert rr % 2 == 1 and cc % 2 == 1
        code_pixels.append((_grid_to_lattice_coord(rr), _grid_to_lattice_coord(cc), int(g[rr, cc]), rr, cc))

    assert 3 <= len(code_pixels) <= 5
    lattice_n = (n - 1) // 2

    orientations = []
    for flip_r in (False, True):
        for flip_c in (False, True):
            local_pixels = []
            for lr, lc, color, gr, gc in code_pixels:
                i = lattice_n - 1 - lr if flip_r else lr
                j = lattice_n - 1 - lc if flip_c else lc
                local_pixels.append((int(i), int(j), int(color), int(gr), int(gc)))
            score = (
                max(max(i, j) for i, j, _color, _gr, _gc in local_pixels),
                sum(i + j for i, j, _color, _gr, _gc in local_pixels),
            )
            orientations.append((score, local_pixels))

    orientations.sort(key=lambda item: item[0])
    assert orientations[0][0] < orientations[1][0]
    local_pixels = orientations[0][1]

    corner_colors: dict[int, int] = {}
    horizontal_edge_colors: dict[int, int] = {}
    vertical_edge_colors: dict[int, int] = {}

    for i, j, color, _gr, _gc in local_pixels:
        assert 0 <= i < lattice_n and 0 <= j < lattice_n
        if i == j:
            assert i not in corner_colors or corner_colors[i] == color
            corner_colors[i] = color
        elif i < j:
            depth = int(i)
            assert depth not in horizontal_edge_colors or horizontal_edge_colors[depth] == color
            horizontal_edge_colors[depth] = color
        else:
            depth = int(j)
            assert depth not in vertical_edge_colors or vertical_edge_colors[depth] == color
            vertical_edge_colors[depth] = color

    assert len(corner_colors) >= 2
    for depth in set(horizontal_edge_colors) | set(vertical_edge_colors):
        assert depth in corner_colors

    return center, local_pixels, corner_colors, horizontal_edge_colors, vertical_edge_colors


def _render_unfolded_frames(
    shape: tuple[int, int],
    corner_colors: dict[int, int],
    horizontal_edge_colors: dict[int, int],
    vertical_edge_colors: dict[int, int],
) -> np.ndarray:
    n, m = map(int, shape)
    assert n == m
    out = np.full((n, m), BACKGROUND, dtype=int)

    for depth, color in corner_colors.items():
        lo = int(depth)
        hi = n - 1 - lo
        out[lo, lo] = int(color)
        out[lo, hi] = int(color)
        out[hi, lo] = int(color)
        out[hi, hi] = int(color)

    for depth, color in horizontal_edge_colors.items():
        lo = int(depth)
        hi = n - 1 - lo
        if hi - lo >= 2:
            out[lo, lo + 1 : hi] = int(color)
            out[hi, lo + 1 : hi] = int(color)

    for depth, color in vertical_edge_colors.items():
        lo = int(depth)
        hi = n - 1 - lo
        if hi - lo >= 2:
            out[lo + 1 : hi, lo] = int(color)
            out[lo + 1 : hi, hi] = int(color)

    return out


def _apply_parity_filter(pattern: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    rr, cc = np.indices(pattern.shape)
    keep = (rr % 2 == center[0] % 2) & (cc % 2 == center[1] % 2)
    out = np.full_like(pattern, BACKGROUND)
    out[keep] = pattern[keep]
    out[center] = BACKGROUND
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unfold a corner code into concentric square frames, then erase every frame pixel off the center marker's parity lattice.

    Input structure:
    - A black odd-sized square grid contains one green marker at the geometric center.
    - A compact 3-5 pixel code sits near exactly one corner, using only odd-row/odd-column lattice cells.
    - Diagonal code pixels identify ring depths and corner colors; off-diagonal pixels at the same local row or column identify horizontal or vertical edge colors for those depths.

    Rule:
    - Detect which corner the compact code belongs to by choosing the corner-local orientation that makes all code pixels closest to one corner.
    - Decode diagonal local pixels as corner colors for concentric square frames, local pixels above the diagonal as horizontal-edge colors, and local pixels below the diagonal as vertical-edge colors.
    - Expand that code into full same-size concentric square frames, overlay the original corner-code pixels, then keep only pixels whose row parity and column parity both match the center marker.
    - Erase the center marker itself to black.

    Key insight:
    - The sparse corner pattern is a symbolic specification for nested frames, while the center marker does not contribute color; it selects which checkerboard parity of the unfolded structure remains visible.
    """
    g = np.asarray(grid, dtype=int)
    center, local_pixels, corner_colors, horizontal_edge_colors, vertical_edge_colors = _decode_corner_code(g)

    pattern = _render_unfolded_frames(g.shape, corner_colors, horizontal_edge_colors, vertical_edge_colors)
    for _i, _j, color, gr, gc in local_pixels:
        pattern[int(gr), int(gc)] = int(color)
    return _apply_parity_filter(pattern, center)


def _sample_size(num_rings: int) -> int:
    choices = [n for n in (11, 13, 15, 17, 19, 21) if n >= 2 * num_rings + 7]
    return random.choice(choices)


def _sample_edge_specs(num_rings: int) -> list[tuple[int, str]]:
    max_extra = min(2, 5 - num_rings)
    edge_count = random.randint(1, max_extra)
    if edge_count == 1:
        return [(random.randrange(num_rings), random.choice(("h", "v")))]

    depths = random.sample(range(num_rings), 2)
    return [(depths[0], "h"), (depths[1], "v")]


def _build_input(
    n: int,
    corner: int,
    corner_colors: dict[int, int],
    horizontal_edge_colors: dict[int, int],
    vertical_edge_colors: dict[int, int],
) -> np.ndarray:
    lattice_n = (n - 1) // 2
    grid = np.full((n, n), BACKGROUND, dtype=int)
    grid[n // 2, n // 2] = MARKER_COLOR

    for depth, color in corner_colors.items():
        r, c = _corner_lattice_coord(depth, depth, corner, lattice_n)
        grid[r, c] = int(color)

    for depth, color in horizontal_edge_colors.items():
        r, c = _corner_lattice_coord(depth, depth + 1, corner, lattice_n)
        grid[r, c] = int(color)

    for depth, color in vertical_edge_colors.items():
        r, c = _corner_lattice_coord(depth + 1, depth, corner, lattice_n)
        grid[r, c] = int(color)

    return grid


def _count_surviving_frame_pixels(
    n: int,
    center: tuple[int, int],
    corner_colors: dict[int, int],
    horizontal_edge_colors: dict[int, int],
    vertical_edge_colors: dict[int, int],
) -> tuple[int, int]:
    frames = _render_unfolded_frames((n, n), corner_colors, horizontal_edge_colors, vertical_edge_colors)
    rr, cc = np.indices((n, n))
    frame_mask = frames != BACKGROUND
    keep = (rr % 2 == center[0] % 2) & (cc % 2 == center[1] % 2)
    kept = int(np.count_nonzero(frame_mask & keep))
    erased = int(np.count_nonzero(frame_mask & (~keep)))
    return kept, erased


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=np.int8)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _gen_attempt() -> np.ndarray | None:
    num_rings = random.randint(2, 4)
    n = _sample_size(num_rings)
    corner = random.randrange(4)

    depth_order = list(range(num_rings))
    corner_colors = {depth: color for depth, color in zip(depth_order, random.sample(CODE_COLORS, num_rings))}

    horizontal_edge_colors: dict[int, int] = {}
    vertical_edge_colors: dict[int, int] = {}
    for depth, kind in _sample_edge_specs(num_rings):
        choices = [color for color in CODE_COLORS if color != corner_colors[depth]]
        color = random.choice(choices)
        if kind == "h":
            horizontal_edge_colors[depth] = color
        else:
            vertical_edge_colors[depth] = color

    inp = _build_input(n, corner, corner_colors, horizontal_edge_colors, vertical_edge_colors)
    center = (n // 2, n // 2)
    kept, erased = _count_surviving_frame_pixels(n, center, corner_colors, horizontal_edge_colors, vertical_edge_colors)
    if kept == 0 or erased == 0:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if out.shape != inp.shape:
        return None
    if int(np.count_nonzero(out)) == 0:
        return None
    sig = _output_signature(out)
    if sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(sig)
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-5 input")
