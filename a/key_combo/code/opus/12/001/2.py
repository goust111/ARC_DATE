from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

GRID_SIZE = 19
CELL_SIZE = 3
CELL_STRIDE = 4
CELL_STARTS = (0, 4, 8, 12, 16)
SEPARATORS = (3, 7, 11, 15)
NETWORK_LINES = (7, 11, 15)
INSTRUCTION_ORIGIN = (1, 1)
INSTRUCTION_SIZE = 6
INSTRUCTION_CELL_STARTS = {(0, 0), (0, 4), (4, 0), (4, 4)}
MAX_GENERATION_ATTEMPTS = 400


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    dst[r0 : r0 + src.shape[0], c0 : c0 + src.shape[1]] = src


def _instruction_mask(grid: np.ndarray) -> np.ndarray:
    r0, c0 = INSTRUCTION_ORIGIN
    block = grid[r0 : r0 + INSTRUCTION_SIZE, c0 : c0 + INSTRUCTION_SIZE]
    mask = np.zeros((3, 3), dtype=bool)
    for mr in range(3):
        for mc in range(3):
            patch = block[2 * mr : 2 * mr + 2, 2 * mc : 2 * mc + 2]
            if np.all(patch == GREEN):
                mask[mr, mc] = True
            else:
                assert np.all(patch == BLACK), "instruction bits must be solid 2x2 green blocks"
    assert np.any(mask), "instruction mask must contain at least one marked bit"
    return mask


def _is_instruction_cell(r0: int, c0: int) -> bool:
    return (r0, c0) in INSTRUCTION_CELL_STARTS


def _trace_rays(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    starts = np.argwhere(g == RED)
    assert len(starts) == 1, "expected exactly one red start pixel"
    start_r, start_c = map(int, starts[0])
    assert start_c == 0 and start_r in NETWORK_LINES, "start must sit on the left edge of a network line"

    line_mask = (g == YELLOW) | (g == ORANGE) | (g == RED)
    marker_mask = g == ORANGE
    out = g.copy()
    pending: list[tuple[int, int, str]] = [(start_r, start_c, "h")]
    seen: set[tuple[int, int, str]] = set()
    opened: set[tuple[int, int]] = set()
    hit_counts: dict[tuple[int, int], int] = {}
    h, w = g.shape

    while pending:
        r, c, axis = pending.pop()
        key = (int(r), int(c), axis)
        if key in seen:
            continue
        seen.add(key)

        if not marker_mask[r, c]:
            out[r, c] = RED

        if axis == "h":
            directions = ((0, -1), (0, 1))
            next_axis = "v"
        else:
            directions = ((-1, 0), (1, 0))
            next_axis = "h"

        new_segments: list[tuple[int, int, str]] = []
        for dr, dc in directions:
            rr, cc = r + dr, c + dc
            while 0 <= rr < h and 0 <= cc < w and line_mask[rr, cc]:
                if marker_mask[rr, cc]:
                    marker = (int(rr), int(cc))
                    hit_counts[marker] = hit_counts.get(marker, 0) + 1
                    if marker not in opened:
                        opened.add(marker)
                        new_segments.append((marker[0], marker[1], next_axis))
                    break
                out[rr, cc] = RED
                rr += dr
                cc += dc
        pending.extend(new_segments)

    return out, opened, hit_counts


def _active_cells(grid_with_rays: np.ndarray) -> set[tuple[int, int]]:
    active: set[tuple[int, int]] = set()
    h, w = grid_with_rays.shape
    for r0 in CELL_STARTS:
        for c0 in CELL_STARTS:
            if _is_instruction_cell(r0, c0):
                continue
            top = r0 - 1 if r0 > 0 else 0
            bottom = r0 + CELL_SIZE if r0 + CELL_SIZE < h else h - 1
            left = c0 - 1 if c0 > 0 else 0
            right = c0 + CELL_SIZE if c0 + CELL_SIZE < w else w - 1
            if (
                np.any(grid_with_rays[top, c0 : c0 + CELL_SIZE] == RED)
                or np.any(grid_with_rays[bottom, c0 : c0 + CELL_SIZE] == RED)
                or np.any(grid_with_rays[r0 : r0 + CELL_SIZE, left] == RED)
                or np.any(grid_with_rays[r0 : r0 + CELL_SIZE, right] == RED)
            ):
                active.add((r0, c0))
    return active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a ray-selected lattice region to recolor only mask-matching blue pixels.

    Input structure
    - A 19x19 lattice is divided by yellow separator lines into 3x3 cells, with a special 6x6 instruction block in the top-left 2x2-cell corner.
    - The instruction block contains only black and solid 2x2 green blocks, which encode a 3x3 binary mask.
    - Outside the instruction block, cells contain black and blue pixels; a red start pixel lies on the left edge, and orange marker pixels sit on yellow gridline intersections.

    Rule
    - Decode the top-left 6x6 instruction block into a 3x3 mask, where each solid 2x2 green block marks one mask position.
    - Starting from the left-edge red pixel, extend a red ray horizontally along yellow gridlines; when a ray first reaches an orange marker, keep the marker orange, stop that branch there, and emit the perpendicular gridline ray from the crossing.
    - After the full ray network is traced, mark a non-instruction cell active iff at least one of its four boundary lines contains a red ray pixel.
    - In every active non-instruction cell whose blue pixels cover all marked mask positions, recolor exactly those mask-position blue pixels to green.

    Key insight
    - The ray network does not paint the target patterns directly; it only selects which cells are eligible for the instruction-mask recoloring step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 19x19 grid"

    mask = _instruction_mask(g)
    traced, _opened, _hit_counts = _trace_rays(g)
    active = _active_cells(traced)

    out = traced.copy()
    for r0 in CELL_STARTS:
        for c0 in CELL_STARTS:
            if (r0, c0) not in active or _is_instruction_cell(r0, c0):
                continue
            cell = g[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
            if np.all(cell[mask] == BLUE):
                out_cell = out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
                out_cell[mask] = GREEN

    return out


def _rand_mask() -> np.ndarray:
    mask = np.zeros(9, dtype=bool)
    count = int(np.random.choice([1, 2, 3, 4, 5], p=[0.08, 0.24, 0.30, 0.24, 0.14]))
    idx = np.random.choice(9, size=count, replace=False)
    mask[idx] = True
    return mask.reshape(3, 3)


def _instruction_block(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(int), np.full((2, 2), GREEN, dtype=int))


def _network_points() -> list[tuple[int, int]]:
    return [(r, c) for r in NETWORK_LINES for c in NETWORK_LINES]


def _content_cells() -> list[tuple[int, int]]:
    return [
        (r0, c0)
        for r0 in CELL_STARTS
        for c0 in CELL_STARTS
        if not _is_instruction_cell(r0, c0)
    ]


def _base_grid(mask: np.ndarray, start_row: int, markers: set[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for sep in SEPARATORS:
        grid[sep, :] = YELLOW
        grid[:, sep] = YELLOW
    _paste(grid, _instruction_block(mask), *INSTRUCTION_ORIGIN)
    for r, c in markers:
        grid[r, c] = ORANGE
    grid[start_row, 0] = RED
    return grid


def _make_matching_cell(mask: np.ndarray) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    flat = cell.reshape(-1)
    mask_pos = np.flatnonzero(mask.reshape(-1))
    flat[mask_pos] = BLUE
    extras = np.flatnonzero(~mask.reshape(-1))
    extra_count = int(np.random.randint(0, min(5 - len(mask_pos), len(extras)) + 1))
    if extra_count:
        chosen = np.random.choice(extras, size=extra_count, replace=False)
        flat[chosen] = BLUE
    return cell


def _make_nonmatching_cell(mask: np.ndarray, *, near: bool) -> np.ndarray:
    cell = np.zeros((CELL_SIZE, CELL_SIZE), dtype=int)
    flat = cell.reshape(-1)
    mask_pos = np.flatnonzero(mask.reshape(-1))
    blocked = int(np.random.choice(mask_pos))
    allowed = np.array([i for i in range(9) if i != blocked], dtype=int)
    if near and len(mask_pos) > 1:
        lo = max(1, len(mask_pos) - 1)
        hi = 5
    else:
        lo = 0
        hi = 5
    count = int(np.random.randint(lo, hi + 1))
    count = min(count, len(allowed))
    if count:
        chosen = np.random.choice(allowed, size=count, replace=False)
        flat[chosen] = BLUE
    assert not np.all(cell[mask] == BLUE)
    return cell


def _expected_output(
    grid: np.ndarray, mask: np.ndarray, active_match: set[tuple[int, int]]
) -> np.ndarray:
    traced, _opened, _hits = _trace_rays(grid)
    out = traced.copy()
    for r0, c0 in active_match:
        out[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE][mask] = GREEN
    return out


def _attempt() -> np.ndarray | None:
    mask = _rand_mask()
    marker_pool = _network_points()
    start_row = int(np.random.choice(NETWORK_LINES))
    marker_count = int(np.random.randint(2, len(marker_pool) + 1))
    markers = {marker_pool[i] for i in np.random.permutation(len(marker_pool))[:marker_count]}
    grid = _base_grid(mask, start_row, markers)

    traced, opened, hit_counts = _trace_rays(grid)
    if any(count > 1 for count in hit_counts.values()):
        return None
    if not (2 <= len(opened) <= 6):
        return None

    active = sorted(_active_cells(traced))
    inactive = sorted(set(_content_cells()) - set(active))
    if len(active) < 3 or len(inactive) < 2:
        return None

    mask_count = int(mask.sum())
    if mask_count > 5:
        return None

    active_match_count = int(np.random.randint(1, min(3, len(active) - 1) + 1))
    inactive_match_count = int(np.random.randint(1, min(2, len(inactive)) + 1))
    active_match = set(tuple(x) for x in np.array(active)[np.random.permutation(len(active))[:active_match_count]])
    remaining_active = [cell for cell in active if cell not in active_match]
    probe_active_nonmatch = remaining_active[int(np.random.randint(len(remaining_active)))]
    inactive_match = set(
        tuple(x) for x in np.array(inactive)[np.random.permutation(len(inactive))[:inactive_match_count]]
    )

    for r0, c0 in _content_cells():
        if (r0, c0) in active_match or (r0, c0) in inactive_match:
            cell = _make_matching_cell(mask)
        elif (r0, c0) == probe_active_nonmatch:
            cell = _make_nonmatching_cell(mask, near=True)
        else:
            cell = _make_nonmatching_cell(mask, near=bool(np.random.randint(2)))
        _paste(grid, cell, r0, c0)

    expected = _expected_output(grid, mask, active_match)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    changed = actual != grid
    if np.count_nonzero(changed) < mask_count:
        return None

    for r0, c0 in active_match:
        out_cell = actual[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        in_cell = grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        if not np.all(in_cell[mask] == BLUE) or not np.all(out_cell[mask] == GREEN):
            return None

    for r0, c0 in inactive_match:
        out_cell = actual[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE]
        if not np.all(out_cell[mask] == BLUE):
            return None

    probe_out = actual[
        probe_active_nonmatch[0] : probe_active_nonmatch[0] + CELL_SIZE,
        probe_active_nonmatch[1] : probe_active_nonmatch[1] + CELL_SIZE,
    ]
    if np.any(probe_out[mask] == GREEN):
        return None

    if len(opened) == len(markers) and np.random.rand() < 0.45:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 2")
