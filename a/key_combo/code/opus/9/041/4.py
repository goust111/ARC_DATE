from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
SIZE_CHOICES = (11, 13, 15)
STEP_PATTERNS = ((2, 5), (3, 6), (4, 7))
CORNER_FLIPS = (
    (False, False),  # top-left
    (False, True),   # top-right
    (True, False),   # bottom-left
    (True, True),    # bottom-right
)
MAX_GENERATION_ATTEMPTS = 400


def _lattice_n(size: int) -> int:
    return (size - 1) // 2


def _corner_coord(local_r: int, local_c: int, flip_r: bool, flip_c: bool, lattice_n: int) -> tuple[int, int]:
    rr = lattice_n - 1 - local_r if flip_r else local_r
    cc = lattice_n - 1 - local_c if flip_c else local_c
    return 1 + 2 * rr, 1 + 2 * cc


def _orientation_candidates(grid: np.ndarray) -> list[tuple[tuple[int, int, int], list[tuple[int, int, int]]]]:
    size = int(grid.shape[0])
    lattice_n = _lattice_n(size)
    foreground = np.argwhere(grid != BACKGROUND)
    lattice_pixels = [
        ((int(r) - 1) // 2, (int(c) - 1) // 2, int(grid[r, c]))
        for r, c in foreground
    ]

    candidates = []
    for flip_r, flip_c in CORNER_FLIPS:
        local_pixels = []
        for lr, lc, color in lattice_pixels:
            i = lattice_n - 1 - lr if flip_r else lr
            j = lattice_n - 1 - lc if flip_c else lc
            local_pixels.append((i, j, color))
        score = (
            max(max(i, j) for i, j, _ in local_pixels),
            sum(i + j for i, j, _ in local_pixels),
            sum(abs(i - j) for i, j, _ in local_pixels),
        )
        candidates.append((score, local_pixels))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _decode_corner_code(grid: np.ndarray) -> tuple[list[int], dict[int, int], dict[int, int], dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    assert h == w and h % 2 == 1 and h >= 11

    foreground = np.argwhere(g != BACKGROUND)
    assert len(foreground) >= 9
    assert all(int(r) % 2 == 1 and int(c) % 2 == 1 for r, c in foreground)

    candidates = _orientation_candidates(g)
    assert candidates[0][0] < candidates[1][0], "corner code must have a unique nearest corner"
    local_pixels = candidates[0][1]

    corner_colors: dict[int, int] = {}
    horizontal_colors: dict[int, int] = {}
    vertical_colors: dict[int, int] = {}

    for i, j, color in local_pixels:
        assert 0 <= i < _lattice_n(h) and 0 <= j < _lattice_n(h)
        if i == j:
            assert i not in corner_colors or corner_colors[i] == color
            corner_colors[i] = color
        elif j == i + 1:
            assert i not in horizontal_colors or horizontal_colors[i] == color
            horizontal_colors[i] = color
        elif i == j + 1:
            assert j not in vertical_colors or vertical_colors[j] == color
            vertical_colors[j] = color
        else:
            raise AssertionError("off-diagonal code pixels must sit immediately beside their ring diagonal")

    depths = sorted(corner_colors)
    assert len(depths) >= 3
    assert depths == list(range(len(depths))), "rings must be consecutive from the outside inward"
    for depth in depths:
        assert depth in horizontal_colors and depth in vertical_colors
    assert len(foreground) == 3 * len(depths)
    assert len(depths) <= (_lattice_n(h) - 1), "grid is too small for the encoded ring count"
    return depths, corner_colors, horizontal_colors, vertical_colors


def _erase_side(out: np.ndarray, coords: list[tuple[int, int]], erase_parity: int) -> None:
    for offset, (r, c) in enumerate(coords):
        if offset % 2 == erase_parity:
            out[r, c] = BACKGROUND


def _render_output(
    size: int,
    depths: list[int],
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
    phase_mode: str = "alternating",
) -> np.ndarray:
    out = np.full((size, size), BACKGROUND, dtype=int)

    for ring_index, depth in enumerate(depths):
        lo = depth
        hi = size - 1 - depth
        assert hi - lo >= 4, "generator only uses rings with visible side interiors"

        corner_color = int(corner_colors[depth])
        horizontal_color = int(horizontal_colors[depth])
        vertical_color = int(vertical_colors[depth])

        out[lo, lo] = corner_color
        out[lo, hi] = corner_color
        out[hi, lo] = corner_color
        out[hi, hi] = corner_color
        out[lo, lo + 1 : hi] = horizontal_color
        out[hi, lo + 1 : hi] = horizontal_color
        out[lo + 1 : hi, lo] = vertical_color
        out[lo + 1 : hi, hi] = vertical_color

        if phase_mode == "alternating":
            erase_parity = 1 if ring_index % 2 == 0 else 0
        elif phase_mode == "all_even":
            erase_parity = 1
        elif phase_mode == "all_odd":
            erase_parity = 0
        elif phase_mode == "no_holes":
            continue
        else:
            raise ValueError(f"unknown phase mode: {phase_mode}")

        top = [(lo, c) for c in range(lo + 1, hi)]
        right = [(r, hi) for r in range(lo + 1, hi)]
        bottom = [(hi, c) for c in range(hi - 1, lo, -1)]
        left = [(r, lo) for r in range(hi - 1, lo, -1)]
        _erase_side(out, top, erase_parity)
        _erase_side(out, right, erase_parity)
        _erase_side(out, bottom, erase_parity)
        _erase_side(out, left, erase_parity)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner-coded nested square-ring description, then punch alternating side holes by ring order.

    Input structure:
    - The grid is an odd-sized square on a black background, with all foreground pixels on odd row/column coordinates.
    - Those sparse pixels form a compact code near exactly one corner.
    - For each ring depth, the diagonal code pixel gives the four corner color, while the two adjacent off-diagonal pixels give the horizontal and vertical side colors.

    Rule:
    - Identify the unique nearest corner and reinterpret every foreground pixel in that corner-local coordinate system.
    - Decode consecutive outer-to-inner ring depths, then unfold them into full square rings on the full grid: diagonal colors paint the four corners, horizontal-code colors paint top/bottom sides, and vertical-code colors paint left/right sides.
    - Number the rings from the outside inward as 0, 1, 2, ... and erase alternating non-corner side pixels clockwise on every side: even-indexed rings erase odd offsets, odd-indexed rings erase even offsets.
    - Corner pixels are never erased, and erased cells become black.

    Key insight:
    - The sparse corner code only determines the ring colors; the final hole phase is a second-stage rule that depends on each ring's nesting order after the symmetric unfolding is complete.
    """
    depths, corner_colors, horizontal_colors, vertical_colors = _decode_corner_code(grid)
    return _render_output(
        int(grid.shape[0]),
        depths,
        corner_colors,
        horizontal_colors,
        vertical_colors,
        phase_mode="alternating",
    )


def _sample_ring_spec(ring_count: int) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    palette = random.sample(PALETTE, len(PALETTE))
    step_h, step_v = random.choice(STEP_PATTERNS)

    corner_colors: dict[int, int] = {}
    horizontal_colors: dict[int, int] = {}
    vertical_colors: dict[int, int] = {}
    triples = []
    for depth in range(ring_count):
        triple = (
            palette[depth % len(palette)],
            palette[(depth + step_h) % len(palette)],
            palette[(depth + step_v) % len(palette)],
        )
        assert len(set(triple)) == 3
        triples.append(triple)
        corner_colors[depth], horizontal_colors[depth], vertical_colors[depth] = triple

    if len(set(triples)) != ring_count:
        raise ValueError("duplicate ring triple")
    return corner_colors, horizontal_colors, vertical_colors


def _build_input(
    size: int,
    ring_count: int,
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
    corner_index: int,
) -> np.ndarray:
    grid = np.full((size, size), BACKGROUND, dtype=int)
    flip_r, flip_c = CORNER_FLIPS[corner_index]
    lattice_n = _lattice_n(size)

    for depth in range(ring_count):
        for local_r, local_c, color in (
            (depth, depth, corner_colors[depth]),
            (depth, depth + 1, horizontal_colors[depth]),
            (depth + 1, depth, vertical_colors[depth]),
        ):
            r, c = _corner_coord(local_r, local_c, flip_r, flip_c, lattice_n)
            grid[r, c] = int(color)
    return grid


def _valid_generated_pair(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        decoded = _decode_corner_code(inp)
    except Exception:
        return False

    depths, corner_colors, horizontal_colors, vertical_colors = decoded
    size = int(inp.shape[0])

    if not np.array_equal(
        expected,
        _render_output(size, depths, corner_colors, horizontal_colors, vertical_colors, phase_mode="alternating"),
    ):
        return False
    if np.array_equal(expected, _render_output(size, depths, corner_colors, horizontal_colors, vertical_colors, phase_mode="all_even")):
        return False
    if np.array_equal(expected, _render_output(size, depths, corner_colors, horizontal_colors, vertical_colors, phase_mode="all_odd")):
        return False
    if np.array_equal(expected, _render_output(size, depths, corner_colors, horizontal_colors, vertical_colors, phase_mode="no_holes")):
        return False
    if not np.array_equal(main(inp), expected):
        return False

    for ring_index, depth in enumerate(depths):
        lo = depth
        hi = size - 1 - depth
        side_cells = hi - lo - 1
        if side_cells < 3:
            return False
        erased = 0
        parity = 1 if ring_index % 2 == 0 else 0
        for offset in range(side_cells):
            if offset % 2 == parity:
                erased += 1
        if erased == 0:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        size = random.choice(SIZE_CHOICES)
        max_rings = min(5, (size - 3) // 2)
        ring_count = random.randint(3, max_rings)

        try:
            corner_colors, horizontal_colors, vertical_colors = _sample_ring_spec(ring_count)
        except ValueError:
            continue

        corner_index = random.randrange(4)
        inp = _build_input(size, ring_count, corner_colors, horizontal_colors, vertical_colors, corner_index)
        expected = _render_output(
            size,
            list(range(ring_count)),
            corner_colors,
            horizontal_colors,
            vertical_colors,
            phase_mode="alternating",
        )

        if _valid_generated_pair(inp, expected):
            return inp

    raise RuntimeError("failed to generate a valid corner-code ring puzzle")
