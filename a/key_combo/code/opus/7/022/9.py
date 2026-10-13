from __future__ import annotations

from collections import deque
import random

import numpy as np


WALL = 0
OLD_CLOSED = 1
GATE = 2
NEW_CLOSED = 4
OPEN = 5

_MAX_GENERATION_ATTEMPTS = 800
_HEIGHT_WALL_RANGE = (4, 6)
_WIDTH_RANGE = (14, 18)
_GATE_COUNT_RANGE = (2, 4)
_HOLES_PER_PIPE_RANGE = (1, 3)
_MIN_GATE_NEIGHBOR_GREY = 3


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _red_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = grid == GATE
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)
    return components


def _find_horizontal_gates(grid: np.ndarray) -> list[tuple[int, int]]:
    gates: list[tuple[int, int]] = []
    for coords in _red_components(grid):
        if len(coords) != 3:
            continue
        rows = {r for r, _c in coords}
        cols = sorted(c for _r, c in coords)
        if len(rows) != 1:
            continue
        if cols != list(range(cols[0], cols[0] + 3)):
            continue
        gates.append((next(iter(rows)), cols[1]))
    return sorted(gates)


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    return (grid == OPEN) | (grid == OLD_CLOSED)


def _edge_reachable(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _closed_mask(passable: np.ndarray) -> np.ndarray:
    return passable & ~_edge_reachable(passable)


def _rotate_selected_gates(
    grid: np.ndarray,
    gates: list[tuple[int, int]],
    active_ids: set[int] | None = None,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    if active_ids is None:
        active_ids = set(range(len(gates)))

    for gate_id, (r, c) in enumerate(gates):
        if gate_id not in active_ids:
            continue
        out[r, c - 1 : c + 2] = OPEN
        out[r - 1 : r + 2, c] = GATE
    return out


def _solve_with_gate_subset(
    grid: np.ndarray,
    active_ids: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    original = np.asarray(grid, dtype=int)
    gates = _find_horizontal_gates(original)
    rotated = _rotate_selected_gates(original, gates, active_ids)
    new_passable = _passable_mask(rotated)
    new_closed = _closed_mask(new_passable)

    out = rotated.copy()
    out[(original == OLD_CLOSED) & ~new_closed] = OPEN
    out[new_closed & new_passable & (original != OLD_CLOSED)] = NEW_CLOSED
    return out, rotated, new_closed, gates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate every horizontal gate so only newly enclosed pipe regions stay highlighted.

    Input structure:
    - Black cells are pipe walls, grey cells are open pipe space, and red 1x3 horizontal bars are gates embedded in wall rows.
    - Some pipe cells are already blue because they form enclosed regions before any gate rotates.
    - Each gate keeps its center fixed and has grey pipe cells immediately above and below its span.

    Rule:
    - Rotate every red horizontal gate into a red vertical 3x1 gate: its former horizontal cells become grey pipe, while the center column becomes red in the row above, center, and below.
    - On this rotated topology, flood-fill from the grid edge through every passable pipe cell (grey or blue) to mark the exterior.
    - Any passable cell not reached by that flood is enclosed after rotation.
    - Keep old blue cells blue only if they are still enclosed, clear opened blue cells back to grey, and color every other newly enclosed passable cell yellow.

    Key insight:
    - The gate rotation matters because it both opens side passages and blocks a center column, so the answer depends on comparing enclosure status before and after the topology change.
    """
    out, _rotated, _new_closed, _gates = _solve_with_gate_subset(grid)
    return out


def _sample_gate_layout(height: int, width: int) -> list[tuple[int, int]] | None:
    wall_rows = list(range(1, height, 2))
    gate_count = random.randint(_GATE_COUNT_RANGE[0], min(_GATE_COUNT_RANGE[1], len(wall_rows)))
    chosen_rows = sorted(random.sample(wall_rows, gate_count))
    gates: list[tuple[int, int]] = []

    for r in chosen_rows:
        candidates = list(range(2, width - 2))
        random.shuffle(candidates)
        center = None
        for c in candidates:
            if all(not (abs(r - rr) == 2 and abs(c - cc) < 2) for rr, cc in gates):
                center = c
                break
        if center is None:
            return None
        gates.append((r, center))
    return gates


def _place_holes(base: np.ndarray, gates: list[tuple[int, int]]) -> None:
    blocked = {
        (rr, cc)
        for r, c in gates
        for rr in (r - 1, r + 1)
        for cc in (c - 1, c, c + 1)
    }

    for r in range(0, base.shape[0], 2):
        candidates = list(range(1, base.shape[1] - 1))
        random.shuffle(candidates)
        target = random.randint(*_HOLES_PER_PIPE_RANGE)
        chosen: list[int] = []
        for c in candidates:
            if (r, c) in blocked:
                continue
            if any(abs(c - other) < 2 for other in chosen):
                continue
            chosen.append(c)
            if len(chosen) >= target:
                break
        for c in chosen:
            base[r, c] = WALL


def _gate_neighborhood_grey_counts(grid: np.ndarray, gates: list[tuple[int, int]]) -> list[int]:
    counts: list[int] = []
    for r, c in gates:
        grey_count = 0
        for rr in (r - 1, r + 1):
            for cc in range(c - 1, c + 2):
                if grid[rr, cc] == OPEN:
                    grey_count += 1
        counts.append(grey_count)
    return counts


def _gate_roles(
    inp: np.ndarray,
    full_closed: np.ndarray,
    full_passable: np.ndarray,
    gates: list[tuple[int, int]],
) -> tuple[bool, bool]:
    opener_gate = False
    closer_gate = False

    for gate_id in range(len(gates)):
        active = set(range(len(gates)))
        active.remove(gate_id)
        _partial_out, _partial_rotated, partial_closed, _ = _solve_with_gate_subset(inp, active)

        if np.any((inp == OLD_CLOSED) & partial_closed & ~full_closed):
            opener_gate = True
        if np.any((inp != OLD_CLOSED) & full_passable & full_closed & ~partial_closed):
            closer_gate = True

    return opener_gate, closer_gate


def _attempt_generate() -> np.ndarray | None:
    n_walls = random.randint(*_HEIGHT_WALL_RANGE)
    height = 2 * n_walls + 1
    width = random.randint(*_WIDTH_RANGE)

    gates = _sample_gate_layout(height, width)
    if gates is None:
        return None

    base = np.zeros((height, width), dtype=int)
    base[::2, :] = OPEN
    for r, c in gates:
        base[r, c - 1 : c + 2] = GATE

    _place_holes(base, gates)
    old_closed = _closed_mask(base == OPEN)
    if int(old_closed.sum()) == 0:
        return None

    inp = base.copy()
    inp[old_closed] = OLD_CLOSED

    if min(_gate_neighborhood_grey_counts(inp, gates)) < _MIN_GATE_NEIGHBOR_GREY:
        return None

    out, rotated, new_closed, detected_gates = _solve_with_gate_subset(inp)
    if detected_gates != gates:
        return None

    opened_count = int(np.count_nonzero((inp == OLD_CLOSED) & ~new_closed))
    kept_count = int(np.count_nonzero((inp == OLD_CLOSED) & new_closed))
    yellow_count = int(np.count_nonzero((inp != OLD_CLOSED) & _passable_mask(rotated) & new_closed))

    if opened_count < 2 or kept_count < 2 or yellow_count < 2:
        return None

    opener_gate, closer_gate = _gate_roles(inp, new_closed, _passable_mask(rotated), gates)
    if not (opener_gate and closer_gate):
        return None

    if not np.array_equal(main(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-topology ARC input")
