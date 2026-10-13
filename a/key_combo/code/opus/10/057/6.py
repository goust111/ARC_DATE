from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5

_GEN_HEIGHT_RANGE = (5, 8)
_GEN_WIDTH_RANGE = (8, 13)
_GEN_BAND_WIDTH_RANGE = (3, 6)
_GEN_MAX_ATTEMPTS = 400


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return 8-connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            components.append(comp)
    return components


def _stack_intermediate(grid: np.ndarray) -> np.ndarray:
    """Stack W, flipud(W), W, flipud(W) with one-row overlaps and upper-copy priority."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((4 * h - 3, w), BLACK, dtype=int)
    copies = [g, np.flipud(g), g, np.flipud(g)]

    for idx, patch in enumerate(copies):
        r0 = idx * (h - 1)
        target = out[r0 : r0 + h, :]
        write_mask = (patch != BLACK) & (target == BLACK)
        target[write_mask] = patch[write_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stack an alternately flipped wave and recolor every stacked row touched by grey seam markers.

    Input structure:
    - The grid has black background and one connected wave-shaped foreground band.
    - The band is blue except for 1-2 grey seam-marker pixels.
    - Every grey marker lies on the wave's first row or last row, and both boundary rows contain foreground.

    Rule:
    - Treat the input as wave W and form a vertically flipped copy W'.
    - Stack W, W', W, W' from top to bottom with a one-row overlap between consecutive copies.
    - On each overlap row, if both copies contribute a non-black pixel at the same position, keep the upper copy's color.
    - After stacking, recolor every blue pixel on each row that contains grey to green, and recolor the grey pixels themselves to green.

    Key insight:
    - The grey boundary markers are not used in place; they propagate through the alternating stack, and only their final stacked row positions tell you which full rows must be recolored.
    """
    stacked = _stack_intermediate(np.asarray(grid, dtype=int))
    marked_rows = np.any(stacked == GREY, axis=1)
    out = stacked.copy()

    for r in np.flatnonzero(marked_rows):
        row = out[r]
        row[row == BLUE] = GREEN
        row[row == GREY] = GREEN
    return out


def _sample_wave_shape() -> tuple[int, int, list[int], list[int]] | None:
    """Generate row intervals for a connected, gently meandering wave band."""
    h = random.randint(*_GEN_HEIGHT_RANGE)
    w = random.randint(*_GEN_WIDTH_RANGE)

    max_band_w = min(_GEN_BAND_WIDTH_RANGE[1], w - 1)
    if max_band_w < _GEN_BAND_WIDTH_RANGE[0]:
        return None

    widths = [random.randint(_GEN_BAND_WIDTH_RANGE[0], max_band_w)]
    lefts = [random.randint(0, w - widths[0])]
    nonzero_step = False
    width_change = False

    for _ in range(1, h):
        prev_left = lefts[-1]
        prev_width = widths[-1]
        next_width = max(_GEN_BAND_WIDTH_RANGE[0], min(max_band_w, prev_width + random.choice((-1, 0, 1))))
        if next_width != prev_width:
            width_change = True

        prev_right = prev_left + prev_width - 1
        valid_lefts = []
        for cand_left in range(0, w - next_width + 1):
            cand_right = cand_left + next_width - 1
            # 8-connectivity between consecutive row runs.
            if cand_left <= prev_right + 1 and cand_right >= prev_left - 1:
                valid_lefts.append(cand_left)
        if not valid_lefts:
            return None

        preferred = prev_left + random.choice((-1, 0, 1))
        cand_left = min(valid_lefts, key=lambda val: (abs(val - preferred), random.random()))
        if cand_left != prev_left:
            nonzero_step = True
        widths.append(next_width)
        lefts.append(cand_left)

    top_cols = set(range(lefts[0], lefts[0] + widths[0]))
    bottom_cols = set(range(lefts[-1], lefts[-1] + widths[-1]))
    if not (top_cols & bottom_cols):
        return None

    if len(set(lefts)) == 1 and len(set(widths)) == 1:
        return None
    if not (nonzero_step or width_change):
        return None

    return h, w, lefts, widths


def _render_wave(h: int, w: int, lefts: list[int], widths: list[int]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for r, (left, width) in enumerate(zip(lefts, widths)):
        grid[r, left : left + width] = BLUE
    return grid


def _choose_marker_columns(
    boundary_cols: list[int], shared_cols: set[int], want_two: bool
) -> list[int] | None:
    """Pick one or two marker columns, preferring a shared/non-shared pair when available."""
    if want_two and len(boundary_cols) < 2:
        return None

    exclusive_cols = [c for c in boundary_cols if c not in shared_cols]
    picks: list[int]
    if want_two:
        if shared_cols and exclusive_cols:
            picks = [random.choice(sorted(shared_cols)), random.choice(exclusive_cols)]
        else:
            picks = random.sample(boundary_cols, 2)
    else:
        pool = list(boundary_cols)
        if shared_cols and exclusive_cols and random.random() < 0.5:
            pool = [random.choice(sorted(shared_cols)), random.choice(exclusive_cols)]
        picks = [random.choice(pool)]
    return picks


def _add_markers(grid: np.ndarray) -> tuple[np.ndarray, str] | None:
    """Place 1-2 grey markers on the first/last row with branch coverage over row choices."""
    h, _w = grid.shape
    top_cols = list(np.flatnonzero(grid[0] == BLUE))
    bottom_cols = list(np.flatnonzero(grid[h - 1] == BLUE))
    shared_cols = set(top_cols) & set(bottom_cols)
    mode = random.choice(("top_only", "bottom_only", "both_rows"))
    marked = grid.copy()

    if mode == "both_rows":
        top_pick = _choose_marker_columns(top_cols, shared_cols, want_two=False)
        bottom_pick = _choose_marker_columns(bottom_cols, shared_cols, want_two=False)
        if top_pick is None or bottom_pick is None:
            return None
        marked[0, top_pick[0]] = GREY
        marked[h - 1, bottom_pick[0]] = GREY
        return marked, mode

    row = 0 if mode == "top_only" else h - 1
    boundary_cols = top_cols if row == 0 else bottom_cols
    want_two = random.random() < 0.5
    picks = _choose_marker_columns(boundary_cols, shared_cols, want_two=want_two)
    if picks is None:
        return None
    for c in picks:
        marked[row, c] = GREY
    return marked, mode


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "input must obey ARC size limits"
    assert 4 * h - 3 <= 30, "stacked output height must stay within ARC limits"

    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, BLUE, GREY}, "input should only contain black, blue, and grey"
    grey_cells = np.argwhere(g == GREY)
    assert 1 <= len(grey_cells) <= 2, "expected 1-2 grey markers"
    assert set(map(int, grey_cells[:, 0])) <= {0, h - 1}, "grey markers must stay on the first or last row"

    nonblack = g != BLACK
    assert len(_connected_components(nonblack)) == 1, "wave must be a single connected component"
    assert np.all(nonblack.sum(axis=1) > 0), "wave must occupy every row from top to bottom"

    top_cols = set(np.flatnonzero(nonblack[0]))
    bottom_cols = set(np.flatnonzero(nonblack[h - 1]))
    assert top_cols and bottom_cols and (top_cols & bottom_cols), "top and bottom rows must overlap in at least one column"


def _validate_output_semantics(grid: np.ndarray, out: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    expected_pre = _stack_intermediate(g)
    expected_rows = np.flatnonzero(np.any(expected_pre == GREY, axis=1))

    assert out.shape == expected_pre.shape, "output shape mismatch"
    assert not np.any(out == GREY), "grey must be fully recolored in the output"

    green_rows = np.flatnonzero(np.any(out == GREEN, axis=1))
    assert np.array_equal(green_rows, expected_rows), "only grey-bearing stacked rows should be recolored"
    expected_row_set = set(map(int, expected_rows))

    for r in range(out.shape[0]):
        if r in expected_row_set:
            assert np.array_equal(out[r], np.where(expected_pre[r] != BLACK, GREEN, BLACK))
        else:
            assert np.array_equal(out[r], expected_pre[r])

    h = g.shape[0]
    expected_positions: list[int] = []
    if np.any(g[0] == GREY):
        expected_positions.extend([0, 2 * h - 2, 4 * h - 4])
    if np.any(g[h - 1] == GREY):
        expected_positions.extend([h - 1, 3 * h - 3])
    expected_positions.sort()
    assert expected_rows.tolist() == expected_positions, "marker propagation rows are incorrect"

    copies = [g, np.flipud(g), g, np.flipud(g)]
    for k in range(3):
        overlap_row = (k + 1) * (h - 1)
        upper = copies[k][-1]
        lower = copies[k + 1][0]
        resolved = np.where(upper != BLACK, upper, lower)
        assert np.array_equal(expected_pre[overlap_row], resolved), "overlap row must keep upper-copy colors"


def _generate_candidate() -> np.ndarray | None:
    sampled = _sample_wave_shape()
    if sampled is None:
        return None
    h, w, lefts, widths = sampled
    grid = _render_wave(h, w, lefts, widths)
    with_markers = _add_markers(grid)
    if with_markers is None:
        return None
    marked, _mode = with_markers
    try:
        _validate_input(marked)
        out = main(marked)
        _validate_output_semantics(marked, out)
    except AssertionError:
        return None
    return marked


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid wave-stacking input")
