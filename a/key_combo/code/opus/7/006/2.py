import random

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4

MAX_RETRIES = 500


def _find_diagonal_segments(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return all maximal occupied segments running along (+1,+1)."""
    h, w = mask.shape
    segments: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r > 0 and c > 0 and bool(mask[r - 1, c - 1]):
            continue
        seg: list[tuple[int, int]] = []
        rr, cc = r, c
        while rr < h and cc < w and bool(mask[rr, cc]):
            seg.append((rr, cc))
            rr += 1
            cc += 1
        segments.append(seg)
    segments.sort(key=lambda seg: (seg[0][0], seg[0][1], len(seg)))
    return segments


def _alternate_segments(source: np.ndarray) -> np.ndarray:
    """Color each maximal diagonal segment blue-yellow-blue... from its top-left end."""
    mask = source != BG
    out = np.zeros_like(source)
    for seg in _find_diagonal_segments(mask):
        for idx, (r, c) in enumerate(seg):
            out[r, c] = BLUE if idx % 2 == 0 else YELLOW
    return out


def _alternate_segments_reverse(source: np.ndarray) -> np.ndarray:
    """Wrong-direction variant used by generator validation."""
    mask = source != BG
    out = np.zeros_like(source)
    for seg in _find_diagonal_segments(mask):
        for idx, (r, c) in enumerate(reversed(seg)):
            out[r, c] = BLUE if idx % 2 == 0 else YELLOW
    return out


def _reflect_fourfold(colored_source: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    out = np.zeros((2 * center_r + 1, 2 * center_c + 1), dtype=int)
    for r, c in np.argwhere(colored_source != BG):
        r = int(r)
        c = int(c)
        color = int(colored_source[r, c])
        out[r, c] = color
        out[r, 2 * center_c - c] = color
        out[2 * center_r - r, c] = color
        out[2 * center_r - r, 2 * center_c - c] = color
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor the upper-left diagonal segments with alternating blue/yellow and complete them by fourfold reflection.

    Input structure:
    - The grid is an odd-sized black canvas with foreground pixels only in the upper-left quadrant.
    - Foreground pixels form 2-4 disjoint maximal segments running in the (+1,+1) direction.
    - The lower/right halves are empty, and one blank center row plus one blank center column separate the source quadrant from its reflections.

    Rule:
    - Find every maximal upper-left to lower-right diagonal segment in the source quadrant.
    - For each segment, keep index 0, 2, 4, ... blue and recolor index 1, 3, 5, ... yellow, counting from the segment's top-left endpoint.
    - Let the blank symmetry axes sit immediately below and to the right of the source extent, then mirror the recolored source across the center column, center row, and both axes together.
    - Leave the center row and center column black.

    Key insight:
    - The decisive information is the per-segment order along each diagonal; reflecting only after building that local blue-yellow rhythm creates the final four-quadrant windmill pattern.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    fg = np.argwhere(grid != BG)
    if fg.size == 0:
        return grid.copy()

    # The occupied source lives strictly above/left of the blank symmetry axes.
    center_r = int(fg[:, 0].max()) + 1
    center_c = int(fg[:, 1].max()) + 1
    source = (grid[:center_r, :center_c] != BG).astype(int) * BLUE

    colored_source = _alternate_segments(source)
    return _reflect_fourfold(colored_source, center_r, center_c)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _segment_cells(r0: int, c0: int, length: int) -> list[tuple[int, int]]:
    return [(r0 + k, c0 + k) for k in range(length)]


def _can_place(
    occupied: set[tuple[int, int]],
    segment: list[tuple[int, int]],
    used_diagonals: set[int],
) -> bool:
    diag = segment[0][0] - segment[0][1]
    if diag in used_diagonals:
        return False
    for r, c in segment:
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_segment(
    occupied: set[tuple[int, int]],
    used_diagonals: set[int],
    segment: list[tuple[int, int]],
) -> None:
    used_diagonals.add(segment[0][0] - segment[0][1])
    for cell in segment:
        occupied.add(cell)


def _uniform_reflection(source: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    return _reflect_fourfold((source != BG).astype(int) * BLUE, center_r, center_c)


def _top_left_only_canvas(colored_source: np.ndarray, center_r: int, center_c: int) -> np.ndarray:
    out = np.zeros((2 * center_r + 1, 2 * center_c + 1), dtype=int)
    out[:center_r, :center_c] = colored_source
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        h, w = g.shape
        assert g.ndim == 2
        assert h % 2 == 1 and w % 2 == 1
        assert max(h, w) <= 30

        center_r = h // 2
        center_c = w // 2
        assert not np.any(g[center_r, :] != BG)
        assert not np.any(g[:, center_c] != BG)
        assert not np.any(g[center_r + 1 :, :] != BG)
        assert not np.any(g[:, center_c + 1 :] != BG)

        source = g[:center_r, :center_c]
        mask = source != BG
        assert np.any(mask)
        assert int(np.argwhere(mask)[:, 0].max()) == center_r - 1
        assert int(np.argwhere(mask)[:, 1].max()) == center_c - 1

        segments = _find_diagonal_segments(mask)
        assert 2 <= len(segments) <= 4
        lengths = [len(seg) for seg in segments]
        assert all(3 <= length <= 6 for length in lengths)
        assert any(length % 2 == 0 for length in lengths)
        assert any(length % 2 == 1 for length in lengths)

        coords = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
        for r, c in coords:
            for nr, nc in _neighbors4(r, c):
                assert (nr, nc) not in coords

        colored_source = _alternate_segments(source)
        out = main(g)
        assert out.shape == g.shape
        assert np.array_equal(out, _reflect_fourfold(colored_source, center_r, center_c))
        assert not np.array_equal(out, _uniform_reflection(source, center_r, center_c))
        assert not np.array_equal(out, _reflect_fourfold(_alternate_segments_reverse(source), center_r, center_c))
        assert not np.array_equal(out, _top_left_only_canvas(colored_source, center_r, center_c))
        assert BLUE in out and YELLOW in out
        return True
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    center_r = random.randint(5, 14)
    center_c = random.randint(5, 14)
    n_segments = random.randint(2, 4)

    max_anchor_len = min(6, center_r, center_c)
    if max_anchor_len < 3:
        return None

    occupied: set[tuple[int, int]] = set()
    used_diagonals: set[int] = set()
    segments: list[list[tuple[int, int]]] = []

    anchor_len = random.randint(3, max_anchor_len)
    anchor = _segment_cells(center_r - anchor_len, center_c - anchor_len, anchor_len)
    if not _can_place(occupied, anchor, used_diagonals):
        return None
    _place_segment(occupied, used_diagonals, anchor)
    segments.append(anchor)

    parity_needed = {0, 1} - {anchor_len % 2}
    for _ in range(n_segments - 1):
        options = [3, 4, 5, 6]
        options = [length for length in options if length <= center_r and length <= center_c]
        if not options:
            return None
        if parity_needed:
            wanted = parity_needed.pop()
            parity_options = [length for length in options if length % 2 == wanted]
            if parity_options:
                options = parity_options

        placed = False
        for _ in range(80):
            length = random.choice(options)
            r0 = random.randint(0, center_r - length)
            c0 = random.randint(0, center_c - length)
            seg = _segment_cells(r0, c0, length)
            if not _can_place(occupied, seg, used_diagonals):
                continue
            _place_segment(occupied, used_diagonals, seg)
            segments.append(seg)
            placed = True
            break
        if not placed:
            return None

    start_rows = {seg[0][0] for seg in segments}
    start_cols = {seg[0][1] for seg in segments}
    if len(start_rows) == 1 and len(start_cols) == 1:
        return None

    grid = np.zeros((2 * center_r + 1, 2 * center_c + 1), dtype=int)
    for r, c in occupied:
        grid[r, c] = BLUE

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
