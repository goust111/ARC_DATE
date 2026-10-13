from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8

TRACK_COLORS = {RED, GREEN, GREY, TEAL}
YELLOW_PARTIAL_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
)
GENERATION_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_full_yellow_blocks(grid: np.ndarray) -> int:
    count = 0
    for comp in _components_of_color(grid, YELLOW):
        if len(comp) != 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2:
            count += 1
    return count


def _line_candidate_from_positions(
    positions: list[int], colors: list[int], axis: int, fixed: int
) -> dict | None:
    if len(positions) < 4:
        return None
    if colors.count(TEAL) != 1 or colors.count(RED) != 1:
        return None

    diffs = [b - a for a, b in zip(positions, positions[1:])]
    if not diffs or any(d <= 0 for d in diffs):
        return None
    step = diffs[0]
    if any(d != step for d in diffs[1:]):
        return None

    teal_idx = colors.index(TEAL)
    if teal_idx not in (0, len(colors) - 1):
        return None

    coords = []
    for pos in positions:
        if axis == 0:
            coords.append((fixed, pos))
        else:
            coords.append((pos, fixed))

    if teal_idx == 0:
        path = coords
    else:
        path = list(reversed(coords))

    return {"axis": axis, "fixed": fixed, "step": step, "path": path}


def _find_track(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []

    for r in range(h):
        positions = [c for c in range(w) if int(g[r, c]) in TRACK_COLORS]
        colors = [int(g[r, c]) for c in positions]
        cand = _line_candidate_from_positions(positions, colors, axis=0, fixed=r)
        if cand is not None:
            candidates.append(cand)

    for c in range(w):
        positions = [r for r in range(h) if int(g[r, c]) in TRACK_COLORS]
        colors = [int(g[r, c]) for r in positions]
        cand = _line_candidate_from_positions(positions, colors, axis=1, fixed=c)
        if cand is not None:
            candidates.append(cand)

    assert len(candidates) == 1, "expected a unique track"
    return candidates[0]


def _simulate_slots(
    slots: list[int], steps: int, *, return_stats: bool = False
) -> list[int] | tuple[list[int], int, int] | None:
    occ = list(slots)
    red_idx = occ.index(RED)
    pushes = 0
    max_chain = 0

    for _ in range(int(steps)):
        if red_idx == 0:
            return None
        target = red_idx - 1
        if occ[target] == GREY:
            start = target
            while start > 0 and occ[start - 1] == GREY:
                start -= 1
            if start == 0 or occ[start - 1] != BLACK:
                return None
            chain_len = red_idx - start
            occ[start - 1 : red_idx - 1] = [GREY] * chain_len
            occ[target] = BLACK
            pushes += 1
            max_chain = max(max_chain, chain_len)

        if occ[target] != BLACK:
            return None
        occ[red_idx] = BLACK
        red_idx = target
        occ[red_idx] = RED

    if return_stats:
        return occ, pushes, max_chain
    return occ


def _render_track(grid: np.ndarray, path: list[tuple[int, int]], slots: list[int]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for idx, ((r, c), occ) in enumerate(zip(path, slots)):
        if occ == RED or occ == GREY:
            out[r, c] = occ
        else:
            out[r, c] = TEAL if idx == 0 else GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the red block along the slot track toward the teal end by the number of full yellow 2x2 blocks, pushing adjacent grey cargo chains forward.

    Input structure:
    - A horizontal or vertical straight track is marked by evenly spaced green slot pixels, with one endpoint slot colored teal.
    - Exactly one red block occupies one slot, and zero or more grey cargo blocks occupy other slots on the same track.
    - Away from the track, yellow objects include some solid 2x2 squares plus incomplete 2- or 3-cell yellow fragments, and there may be isolated red noise pixels.

    Rule:
    - Count the yellow 4-connected components whose bounding box is exactly 2x2 and whose area is four; call this count N.
    - Move the red block one slot at a time toward the teal endpoint for N steps.
    - Whenever the destination slot is occupied by grey cargo, push that cargo together with the contiguous grey chain in front of it forward by one slot; redraw every vacated non-end slot as green.

    Key insight:
    - The movement distance is not marked on the track; it must be inferred by filtering the yellow objects down to only complete 2x2 blocks, while the grey cells behave as movable cargo rather than fixed obstacles.
    """
    g = np.asarray(grid, dtype=int)
    track = _find_track(g)
    path = track["path"]
    slots = []
    for r, c in path:
        color = int(g[r, c])
        slots.append(color if color in (RED, GREY) else BLACK)

    steps = _count_full_yellow_blocks(g)
    final_slots = _simulate_slots(slots, steps)
    assert final_slots is not None, "input must allow the required movement"
    return _render_track(g, path, final_slots)


def _rotate_shape(shape: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    pts = np.array(shape, dtype=int)
    for _ in range(turns % 4):
        pts = np.column_stack((-pts[:, 1], pts[:, 0]))
    pts -= pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _can_place_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    *,
    forbid_rows: set[int],
    forbid_cols: set[int],
    same_color: int | None = None,
) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if grid[r, c] != BLACK:
            return False
        if r in forbid_rows or c in forbid_cols:
            return False

    if same_color is None:
        return True

    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == same_color and (nr, nc) not in cells:
                return False
    return True


def _place_yellow_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    *,
    forbid_rows: set[int],
    forbid_cols: set[int],
) -> bool:
    h, w = grid.shape
    sh = max(r for r, _ in shape) + 1
    sw = max(c for _, c in shape) + 1
    anchors = [
        (r0, c0)
        for r0 in range(h - sh + 1)
        for c0 in range(w - sw + 1)
    ]
    random.shuffle(anchors)

    for r0, c0 in anchors:
        cells = [(r0 + r, c0 + c) for r, c in shape]
        if not _can_place_cells(
            grid,
            cells,
            forbid_rows=forbid_rows,
            forbid_cols=forbid_cols,
            same_color=YELLOW,
        ):
            continue
        for r, c in cells:
            grid[r, c] = YELLOW
        return True
    return False


def _place_red_noise(
    grid: np.ndarray, count: int, *, forbid_rows: set[int], forbid_cols: set[int]
) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(cells)
    placed = 0
    for r, c in cells:
        if grid[r, c] != BLACK or r in forbid_rows or c in forbid_cols:
            continue
        if any(
            0 <= nr < h and 0 <= nc < w and grid[nr, nc] == RED
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
        ):
            continue
        grid[r, c] = RED
        placed += 1
        if placed == count:
            return True
    return False


def _expected_output_for_generated(
    grid: np.ndarray, path: list[tuple[int, int]], final_slots: list[int]
) -> np.ndarray:
    return _render_track(grid, path, final_slots)


def _build_track(
    h: int, w: int, steps: int
) -> tuple[list[tuple[int, int]], list[int], list[int], int] | None:
    orientation = random.choice(("horizontal", "vertical"))
    dim = w if orientation == "horizontal" else h
    step = random.choice((2, 3))
    max_slots = min(10, 1 + (dim - 3) // step)
    min_slots = max(6, steps + 3)
    if max_slots < min_slots:
        return None
    slot_count = random.randint(min_slots, max_slots)
    span = 1 + (slot_count - 1) * step

    if orientation == "horizontal":
        row = random.randint(2, h - 3)
        left = random.randint(1, w - span - 1)
        right = left + span - 1
        if random.choice((True, False)):
            teal = (row, left)
            delta = (0, step)
        else:
            teal = (row, right)
            delta = (0, -step)
    else:
        col = random.randint(2, w - 3)
        top = random.randint(1, h - span - 1)
        bottom = top + span - 1
        if random.choice((True, False)):
            teal = (top, col)
            delta = (step, 0)
        else:
            teal = (bottom, col)
            delta = (-step, 0)

    path = []
    r, c = teal
    dr, dc = delta
    for _ in range(slot_count):
        path.append((r, c))
        r += dr
        c += dc

    branch = random.choices(
        ("no_push", "push", "chain"),
        weights=(1, 2, 2),
        k=1,
    )[0]
    all_positions = list(range(1, slot_count))

    for _ in range(250):
        red_idx = random.randint(steps, slot_count - 1)
        slots = [BLACK] * slot_count
        slots[red_idx] = RED

        max_cargo = min(5, slot_count - 2)
        cargo_count = random.randint(1, max_cargo) if max_cargo > 0 else 0
        choices = [p for p in all_positions if p != red_idx]
        cargos = set(random.sample(choices, k=min(cargo_count, len(choices))))

        if branch in {"push", "chain"} and red_idx > 0:
            cargos.add(red_idx - 1)
        if branch == "chain" and red_idx > 2:
            chain_len = random.randint(2, min(3, red_idx))
            for pos in range(red_idx - chain_len, red_idx):
                if pos > 0:
                    cargos.add(pos)

        for pos in cargos:
            slots[pos] = GREY

        sim = _simulate_slots(slots, steps, return_stats=True)
        if sim is None:
            continue
        final_slots, pushes, max_chain = sim

        if branch == "no_push" and pushes != 0:
            continue
        if branch == "push" and pushes < 1:
            continue
        if branch == "chain" and max_chain < 2:
            continue
        return path, slots, final_slots, step

    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        h = random.randint(14, 24)
        w = random.randint(14, 24)
        grid = np.full((h, w), BLACK, dtype=int)

        full_block_count = random.randint(1, 4)
        track_spec = _build_track(h, w, full_block_count)
        if track_spec is None:
            continue
        path, slots, final_slots, _ = track_spec

        forbid_rows: set[int] = set()
        forbid_cols: set[int] = set()
        horizontal = path[0][0] == path[1][0]
        if horizontal:
            row = path[0][0]
            for rr in (row - 1, row, row + 1):
                if 0 <= rr < h:
                    forbid_rows.add(rr)
        else:
            col = path[0][1]
            for cc in (col - 1, col, col + 1):
                if 0 <= cc < w:
                    forbid_cols.add(cc)

        for idx, (r, c) in enumerate(path):
            grid[r, c] = TEAL if idx == 0 else GREEN
        for occ, (r, c) in zip(slots, path):
            if occ in (RED, GREY):
                grid[r, c] = occ

        ok = True
        for _ in range(full_block_count):
            if not _place_yellow_shape(
                grid,
                ((0, 0), (0, 1), (1, 0), (1, 1)),
                forbid_rows=forbid_rows,
                forbid_cols=forbid_cols,
            ):
                ok = False
                break
        if not ok:
            continue

        partial_count = random.randint(2, 5)
        for _ in range(partial_count):
            shape = random.choice(YELLOW_PARTIAL_SHAPES)
            shape = _rotate_shape(shape, random.randint(0, 3))
            if not _place_yellow_shape(
                grid,
                shape,
                forbid_rows=forbid_rows,
                forbid_cols=forbid_cols,
            ):
                ok = False
                break
        if not ok:
            continue

        if not _place_red_noise(
            grid,
            random.randint(1, 3),
            forbid_rows=forbid_rows,
            forbid_cols=forbid_cols,
        ):
            continue

        expected = _expected_output_for_generated(grid, path, final_slots)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
