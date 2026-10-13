import numpy as np
import random


def _clipped_spiral_edges(center: tuple[int, int], size: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    def inside(r: int, c: int) -> bool:
        return 0 <= r < size and 0 <= c < size

    r, c = center
    assert inside(r, c)
    visited = {(r, c)}
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    step_len = 1
    direction_index = 0

    while len(visited) < size * size:
        assert step_len <= size * 2
        for _ in range(2):
            dr, dc = directions[direction_index % len(directions)]
            for _ in range(step_len):
                nr, nc = r + dr, c + dc
                if inside(r, c) and inside(nr, nc):
                    edges.append(((r, c), (nr, nc)))
                r, c = nr, nc
                if inside(r, c):
                    visited.add((r, c))
            direction_index += 1
        step_len += 1

    return edges


def main(grid: np.ndarray) -> np.ndarray:
    """Expand the single colored pixel into a clipped square-spiral graph on a 9x9 lattice.

    Input structure
    - The input is a 3x3 grid with one foreground pixel and eight background pixels.

    Rule
    - Map the foreground pixel at input position (r, c) to node (2r, 2c) of a 5x5 output lattice.
    - Paint every lattice node, then trace an outward square spiral from that node with step lengths
      1, 1, 2, 2, ... in up, right, down, left order.
    - Paint only spiral edge midpoints whose two endpoint nodes both lie inside the 5x5 lattice.

    Key insight
    - The 9x9 grid is a doubled 5x5 node lattice: even-even cells are nodes, and the colored
      cells between adjacent nodes are the clipped edges of one spiral centered on the marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (3, 3)

    colors, counts = np.unique(g, return_counts=True)
    assert len(colors) == 2
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) == 8
    fg_candidates = [int(color) for color in colors if int(color) != bg]
    assert len(fg_candidates) == 1
    fg = fg_candidates[0]

    coords = np.argwhere(g == fg)
    assert coords.shape == (1, 2)
    source_r, source_c = map(int, coords[0])

    lattice_size = 5
    out = np.full((9, 9), bg, dtype=int)
    out[::2, ::2] = fg

    center = (source_r * 2, source_c * 2)
    for (r1, c1), (r2, c2) in _clipped_spiral_edges(center, lattice_size):
        out[r1 + r2, c1 + c2] = fg

    return out

# --------------- generator (o5x) ---------------

# Canvas and lattice dimensions.
INPUT_SIZE = 3
OUTPUT_SIZE = 9
LATTICE_SIZE = 5

# Input sampling.
INPUT_CELL_COUNT = INPUT_SIZE * INPUT_SIZE
FOREGROUND_MIN = 1
FOREGROUND_MAX = 9

# Rejection guard.
GENERATION_ATTEMPTS = 300


def _sample_input() -> np.ndarray:
    """Return a 3x3 grid with one uniformly sampled foreground marker."""
    inp = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=int)
    pos = random.randrange(INPUT_CELL_COUNT)
    inp[pos // INPUT_SIZE, pos % INPUT_SIZE] = random.randint(FOREGROUND_MIN, FOREGROUND_MAX)
    return inp


def _extract_marker(inp: np.ndarray) -> tuple[int, int, int]:
    """Return the single nonzero marker as (row, col, color)."""
    fg_coords = np.argwhere(inp != 0)
    assert fg_coords.shape == (1, 2)
    source_r, source_c = map(int, fg_coords[0])
    return source_r, source_c, int(inp[source_r, source_c])


def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    """Render the deterministic 9x9 output for a sampled input."""
    source_r, source_c, fg = _extract_marker(inp)
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=int)
    out[::2, ::2] = fg

    center = (source_r * 2, source_c * 2)
    for (r1, c1), (r2, c2) in _clipped_spiral_edges(center, LATTICE_SIZE):
        out[r1 + r2, c1 + c2] = fg

    return out


def _gen_attempt() -> np.ndarray | None:
    inp = _sample_input()
    expected = _gen_expected_output(inp)
    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 8b28cd80 input")
