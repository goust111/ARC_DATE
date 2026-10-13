import numpy as np

from arc_utils import get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a corner-coded set of concentric square frames by reflecting it to all four sides.
    Input structure
    - The grid is an odd-sized square with black/empty background and foreground pixels only on the odd-coordinate lattice.
    - The foreground pixels form a compact code near one corner of the lattice.
    Rule
    - Choose the unique corner closest to all foreground pixels and read pixels as local coordinates from that corner.
    - A diagonal local pixel gives the color of the four corners of that ring.
    - A local pixel on the horizontal or vertical side of a ring gives the color for both opposite sides of that ring.
    Key insight
    - The sparse input is one corner's description of nested square frames, not the frames themselves; the missing cells are filled by bilateral symmetry and side expansion.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    assert h == w and h % 2 == 1 and h >= 3

    lattice_n = (h - 1) // 2
    foreground = np.argwhere(g != bg)
    assert len(foreground) > 0
    assert all(int(r) % 2 == 1 and int(c) % 2 == 1 for r, c in foreground)

    lattice_pixels = [
        ((int(r) - 1) // 2, (int(c) - 1) // 2, int(g[r, c]))
        for r, c in foreground
    ]

    orientations = []
    for flip_r in (False, True):
        for flip_c in (False, True):
            local_pixels = []
            for lr, lc, color in lattice_pixels:
                i = lattice_n - 1 - lr if flip_r else lr
                j = lattice_n - 1 - lc if flip_c else lc
                local_pixels.append((i, j, color))
            score = (
                max(max(i, j) for i, j, _color in local_pixels),
                sum(i + j for i, j, _color in local_pixels),
            )
            orientations.append((score, local_pixels))

    orientations.sort(key=lambda item: item[0])
    assert orientations[0][0] < orientations[1][0]
    local_pixels = orientations[0][1]

    corner_colors: dict[int, int] = {}
    horizontal_edge_colors: dict[int, int] = {}
    vertical_edge_colors: dict[int, int] = {}

    for i, j, color in local_pixels:
        assert 0 <= i < lattice_n and 0 <= j < lattice_n
        if i == j:
            assert i not in corner_colors or corner_colors[i] == color
            corner_colors[i] = color
        elif i < j:
            depth = i
            assert depth < j < lattice_n - depth
            assert depth not in horizontal_edge_colors or horizontal_edge_colors[depth] == color
            horizontal_edge_colors[depth] = color
        else:
            depth = j
            assert depth < i < lattice_n - depth
            assert depth not in vertical_edge_colors or vertical_edge_colors[depth] == color
            vertical_edge_colors[depth] = color

    for depth in set(horizontal_edge_colors) | set(vertical_edge_colors):
        assert depth in corner_colors

    out = np.full_like(g, bg)

    def grid_coord(lattice_index: int) -> int:
        return 1 + 2 * lattice_index

    for depth, color in corner_colors.items():
        lo = depth
        hi = lattice_n - 1 - depth
        for ri in {lo, hi}:
            for ci in {lo, hi}:
                out[grid_coord(ri), grid_coord(ci)] = color

    for depth, color in horizontal_edge_colors.items():
        lo = depth
        hi = lattice_n - 1 - depth
        assert lo < hi
        for ci in range(lo + 1, hi):
            out[grid_coord(lo), grid_coord(ci)] = color
            out[grid_coord(hi), grid_coord(ci)] = color

    for depth, color in vertical_edge_colors.items():
        lo = depth
        hi = lattice_n - 1 - depth
        assert lo < hi
        for ri in range(lo + 1, hi):
            out[grid_coord(ri), grid_coord(lo)] = color
            out[grid_coord(ri), grid_coord(hi)] = color

    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice geometry.
GRID_SIZE, LATTICE_SIZE, BACKGROUND_COLOR = 19, 9, 0
# Code limits: rings use one pixel/color; edge rings use two pixels/one color.
MIN_RINGS, MAX_RINGS = 2, 4
MIN_EDGE_RINGS = 1
MAX_COLOR_SLOTS, MAX_CODE_PIXELS = 5, 7
# Palette, orientation, and retry controls.
COLOR_RANGE = range(1, 10)
NUM_CORNERS, MAX_GENERATION_ATTEMPTS = 4, 300
EDGE_LOCAL_OFFSETS = ((0, 1), (1, 0))


def _gen_lattice_to_grid(idx: int) -> int:
    return 1 + 2 * idx


def _gen_corner_coord(i: int, j: int, corner: int) -> tuple[int, int]:
    if corner in (1, 3):
        j = LATTICE_SIZE - 1 - j
    if corner in (2, 3):
        i = LATTICE_SIZE - 1 - i
    return _gen_lattice_to_grid(i), _gen_lattice_to_grid(j)


def _gen_sample_code() -> tuple[dict[int, int], dict[int, int]] | None:
    """Sample ring depths and globally distinct colors under the original caps."""
    num_rings = random.randint(MIN_RINGS, MAX_RINGS)
    max_edges = min(num_rings, MAX_COLOR_SLOTS - num_rings, (MAX_CODE_PIXELS - num_rings) // 2)
    if max_edges < MIN_EDGE_RINGS:
        return None
    num_edges = random.randint(MIN_EDGE_RINGS, max_edges)
    edge_depths = set(random.sample(range(num_rings), num_edges))
    colors = random.sample(COLOR_RANGE, num_rings + num_edges)
    corner_colors = {depth: colors[depth] for depth in range(num_rings)}
    edge_colors = {
        depth: colors[num_rings + k]
        for k, depth in enumerate(sorted(edge_depths))
    }
    return corner_colors, edge_colors


def _gen_build_output(corner_colors: dict[int, int], edge_colors: dict[int, int]) -> np.ndarray:
    """Render the completed concentric frame grid for the sampled code."""
    out = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    for depth, color in corner_colors.items():
        lo, hi = depth, LATTICE_SIZE - 1 - depth
        for ri in (lo, hi):
            for ci in (lo, hi):
                out[_gen_lattice_to_grid(ri), _gen_lattice_to_grid(ci)] = color
    for depth, color in edge_colors.items():
        lo, hi = depth, LATTICE_SIZE - 1 - depth
        line = [_gen_lattice_to_grid(k) for k in range(lo + 1, hi)]
        grid_lo = _gen_lattice_to_grid(lo)
        grid_hi = _gen_lattice_to_grid(hi)
        out[grid_lo, line] = color
        out[grid_hi, line] = color
        out[line, grid_lo] = color
        out[line, grid_hi] = color
    return out


def _gen_build_input(corner_colors: dict[int, int], edge_colors: dict[int, int], corner: int) -> np.ndarray:
    """Render the compact corner-local code for one chosen orientation."""
    inp = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    for depth, color in corner_colors.items():
        r, c = _gen_corner_coord(depth, depth, corner)
        inp[r, c] = color
    for depth, color in edge_colors.items():
        for di, dj in EDGE_LOCAL_OFFSETS:
            r, c = _gen_corner_coord(depth + di, depth + dj, corner)
            inp[r, c] = color
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input through structure, palette, render, and oracle."""
    code = _gen_sample_code()
    if code is None:
        return None
    corner_colors, edge_colors = code
    out = _gen_build_output(corner_colors, edge_colors)
    corner = random.randrange(NUM_CORNERS)
    inp = _gen_build_input(corner_colors, edge_colors, corner)
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9d9215db input")
