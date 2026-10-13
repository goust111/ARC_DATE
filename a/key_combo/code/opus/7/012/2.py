from __future__ import annotations

from collections import deque
import random

import numpy as np


GREY = 5
BLACK = 0
RED = 2
GREEN = 3
MARKER_COLORS = (1, 4, 6, 7, 8, 9)

# A hand-built 11x11 line object whose enclosed topology changes under
# many cyclic shifts, not just by plain translation.
BASE_PAYLOAD = np.array(
    [
        [2, 2, 2, 2, 0, 2, 0, 0, 2, 2, 2],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0],
        [2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0],
        [2, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2],
        [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0],
        [0, 2, 2, 0, 0, 2, 0, 0, 2, 2, 0],
    ],
    dtype=int,
)

_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CANDIDATE_CACHE: dict[bytes, list[tuple[int, int]]] = {}


def _boundary_reachable_black(mask_grid: np.ndarray) -> np.ndarray:
    """Return the 4-connected black cells reachable from the outer border."""
    h, w = mask_grid.shape
    reachable = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        if mask_grid[r, 0] == BLACK and not reachable[r, 0]:
            reachable[r, 0] = True
            queue.append((r, 0))
        if mask_grid[r, w - 1] == BLACK and not reachable[r, w - 1]:
            reachable[r, w - 1] = True
            queue.append((r, w - 1))
    for c in range(w):
        if mask_grid[0, c] == BLACK and not reachable[0, c]:
            reachable[0, c] = True
            queue.append((0, c))
        if mask_grid[h - 1, c] == BLACK and not reachable[h - 1, c]:
            reachable[h - 1, c] = True
            queue.append((h - 1, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask_grid[nr, nc] == BLACK and not reachable[nr, nc]:
                reachable[nr, nc] = True
                queue.append((nr, nc))

    return reachable


def _enclosed_black_mask(payload: np.ndarray) -> np.ndarray:
    reachable = _boundary_reachable_black(payload)
    return (payload == BLACK) & (~reachable)


def _red_component_count(payload: np.ndarray) -> int:
    h, w = payload.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for rr, cc in np.argwhere(payload == RED):
        r = int(rr)
        c = int(cc)
        if seen[r, c]:
            continue
        count += 1
        stack = [(r, c)]
        seen[r, c] = True
        while stack:
            cr, cc2 = stack.pop()
            for dr, dc in _DIR4:
                nr = cr + dr
                nc = cc2 + dc
                if 0 <= nr < h and 0 <= nc < w and payload[nr, nc] == RED and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _roll_payload(payload: np.ndarray, dx: int, dy: int) -> np.ndarray:
    return np.roll(np.roll(payload, int(dx), axis=1), int(dy), axis=0)


def _solve_payload(payload: np.ndarray, fill_color: int, dx: int, dy: int) -> np.ndarray:
    shifted = _roll_payload(payload, dx, dy)
    reachable = _boundary_reachable_black(shifted)

    out = shifted.copy()
    out[(shifted == BLACK) & reachable] = GREEN
    out[(shifted == BLACK) & (~reachable)] = int(fill_color)
    return out


def _dihedral_transform(grid: np.ndarray, rot_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, int(rot_k))
    if flip_lr:
        out = np.fliplr(out)
    return out


def _candidate_shifts(payload: np.ndarray) -> list[tuple[int, int]]:
    key = payload.tobytes()
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]

    h, w = payload.shape
    candidates: list[tuple[int, int]] = []

    if _red_component_count(payload) != 1:
        _CANDIDATE_CACHE[key] = candidates
        return candidates

    pre_holes = _enclosed_black_mask(payload)
    if not pre_holes.any():
        _CANDIDATE_CACHE[key] = candidates
        return candidates

    for dy in range(1, h):
        for dx in range(1, w):
            shifted = _roll_payload(payload, dx, dy)
            if _red_component_count(shifted) != 1:
                continue

            post_holes = _enclosed_black_mask(shifted)
            if not post_holes.any():
                continue

            # Reject plain translated-hole cases; the seam cut must alter topology.
            if np.array_equal(post_holes, _roll_payload(pre_holes.astype(int), dx, dy).astype(bool)):
                continue

            correct = _solve_payload(payload, 7, dx, dy)
            only_x = _solve_payload(payload, 7, dx, 0)
            only_y = _solve_payload(payload, 7, 0, dy)
            no_shift = _solve_payload(payload, 7, 0, 0)

            if np.array_equal(correct, only_x) or np.array_equal(correct, only_y) or np.array_equal(correct, no_shift):
                continue

            if not np.any(correct == GREEN) or not np.any(correct == 7):
                continue

            candidates.append((dx, dy))

    _CANDIDATE_CACHE[key] = candidates
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift the payload line object, then flood-fill exterior and enclosed regions with different colors.

    Input structure:
    - The top row is a grey control row with exactly one non-grey marker pixel.
    - The left column is a grey control column with exactly one same-colored marker pixel.
    - Removing that row and column leaves a black payload containing one 4-connected red line object.

    Rule:
    - The control-row marker color is the enclosed-region fill color, and its column index is the rightward cyclic shift.
    - The control-column marker row index is the downward cyclic shift.
    - Remove the control row and control column, cyclically roll the payload by those offsets, then treat red as walls.
    - Flood-fill all boundary-reachable black cells and recolor them green.
    - Recolor the remaining black cells, which are enclosed by the shifted topology, with the marker color.

    Key insight:
    - The same marker simultaneously encodes the fill color and the cyclic offset, and that offset changes which black regions stay connected to the border after the seam cut moves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 3 and g.shape[1] >= 3

    row_markers = [(0, c) for c in range(1, g.shape[1]) if g[0, c] not in (GREY, BLACK)]
    col_markers = [(r, 0) for r in range(1, g.shape[0]) if g[r, 0] not in (GREY, BLACK)]

    assert len(row_markers) == 1
    assert len(col_markers) == 1

    _, dx = row_markers[0]
    dy, _ = col_markers[0]
    fill_color = int(g[row_markers[0]])
    assert int(g[col_markers[0]]) == fill_color

    payload = g[1:, 1:].copy()
    return _solve_payload(payload, fill_color, dx, dy)


def generate_input() -> np.ndarray:
    for _ in range(200):
        payload = _dihedral_transform(
            BASE_PAYLOAD,
            rot_k=random.randrange(4),
            flip_lr=bool(random.randrange(2)),
        )
        payload = _roll_payload(payload, random.randrange(payload.shape[1]), random.randrange(payload.shape[0]))

        shifts = _candidate_shifts(payload)
        if not shifts:
            continue

        dx, dy = random.choice(shifts)
        color = random.choice(MARKER_COLORS)

        grid = np.full((payload.shape[0] + 1, payload.shape[1] + 1), GREY, dtype=int)
        grid[1:, 1:] = payload
        grid[0, dx] = color
        grid[dy, 0] = color

        out = main(grid)
        only_x = _solve_payload(payload, color, dx, 0)
        only_y = _solve_payload(payload, color, 0, dy)
        no_shift = _solve_payload(payload, color, 0, 0)

        assert _red_component_count(payload) == 1
        assert _enclosed_black_mask(payload).any()
        assert np.any(out == color)
        assert np.any(out == GREEN)
        assert not np.array_equal(out, only_x)
        assert not np.array_equal(out, only_y)
        assert not np.array_equal(out, no_shift)
        return grid

    raise RuntimeError("failed to generate a valid cyclic-topology ARC input")
