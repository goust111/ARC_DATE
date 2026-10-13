from __future__ import annotations

from collections import Counter
from typing import NamedTuple
import random

import numpy as np


GRAY = 5
ARC_COLORS = list(range(10))
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Pointer(NamedTuple):
    tail: tuple[int, int]
    head: tuple[int, int]
    direction: tuple[int, int]
    color: int


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(shape: tuple[int, int], r: int, c: int):
    for dr, dc in DIRS:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            yield rr, cc


def _ray_contains(pointer: Pointer, cell: tuple[int, int]) -> bool:
    hr, hc = pointer.head
    r, c = cell
    dr, dc = pointer.direction
    if dr:
        return c == hc and (r - hr) * dr > 0
    return r == hr and (c - hc) * dc > 0


def _segment_to_hub(pointer: Pointer, hub: tuple[int, int]) -> list[tuple[int, int]]:
    assert _ray_contains(pointer, hub), "pointer ray must pass through hub"
    r, c = pointer.head
    dr, dc = pointer.direction
    cells: list[tuple[int, int]] = []
    while (r, c) != hub:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _parse_pointers(grid: np.ndarray, bg: int) -> list[Pointer]:
    shape = grid.shape
    tails = [tuple(map(int, rc)) for rc in np.argwhere(grid == GRAY)]
    head_cells = {
        tuple(map(int, rc))
        for rc in np.argwhere((grid != bg) & (grid != GRAY))
    }
    assert len(tails) >= 3, "expected several grey-tailed pointers"
    assert head_cells, "expected colored pointer heads"

    seen_heads: set[tuple[int, int]] = set()
    pointers: list[Pointer] = []
    for tail in sorted(tails):
        tr, tc = tail
        adjacent_heads = [
            (rr, cc)
            for rr, cc in _neighbors4(shape, tr, tc)
            if int(grid[rr, cc]) not in (bg, GRAY)
        ]
        assert len(adjacent_heads) == 1, "each grey tail must touch exactly one colored head"
        head = adjacent_heads[0]
        assert head not in seen_heads, "a head cannot belong to two tails"

        grey_neighbors = [
            (rr, cc)
            for rr, cc in _neighbors4(shape, head[0], head[1])
            if int(grid[rr, cc]) == GRAY
        ]
        assert grey_neighbors == [tail], "each colored head must touch only its own grey tail"

        dr, dc = head[0] - tr, head[1] - tc
        assert abs(dr) + abs(dc) == 1, "pointer must be an orthogonal two-cell object"
        pointers.append(Pointer(tail, head, (dr, dc), int(grid[head])))
        seen_heads.add(head)

    assert seen_heads == head_cells, "all non-background non-grey cells must be pointer heads"
    return pointers


def _solve_details(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids are at most 30 by 30"

    bg = _get_bg(g)
    assert bg != GRAY, "background must differ from the shared grey tail color"
    pointers = _parse_pointers(g, bg)

    bg_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == bg)]
    scored: list[tuple[int, tuple[int, int]]] = []
    for cell in bg_cells:
        scored.append((sum(_ray_contains(pointer, cell) for pointer in pointers), cell))
    assert scored, "expected at least one hidden background hub candidate"

    best = max(score for score, _ in scored)
    winners = [cell for score, cell in scored if score == best]
    assert best == len(pointers) - 1, "exactly one pointer must fail to vote for the hub"
    assert len(winners) == 1, "hub must be the unique maximum-consensus ray intersection"

    hub = winners[0]
    truthful = [pointer for pointer in pointers if _ray_contains(pointer, hub)]
    liars = [pointer for pointer in pointers if not _ray_contains(pointer, hub)]
    assert len(truthful) == len(pointers) - 1
    assert len(liars) == 1
    return bg, hub, truthful, liars[0], pointers


def _render_solution(
    grid: np.ndarray,
    bg: int,
    hub: tuple[int, int],
    truthful: list[Pointer],
    liar: Pointer,
) -> np.ndarray:
    out = grid.copy()
    liar_cells = {liar.tail, liar.head}
    for r, c in liar_cells:
        out[r, c] = bg

    painted: set[tuple[int, int]] = set()
    for pointer in truthful:
        for cell in _segment_to_hub(pointer, hub):
            assert cell not in liar_cells, "liar cells must not lie on a truthful connector"
            r, c = cell
            if cell == pointer.head:
                assert int(out[r, c]) == pointer.color
            else:
                assert int(out[r, c]) == bg, "connector segments must cross only background"
                assert cell not in painted, "connector segments must not overlap before the hub"
            out[r, c] = pointer.color
            painted.add(cell)

    assert int(out[hub]) == bg, "hub must be invisible in the input"
    out[hub] = GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each truthful two-tone pointer to the hidden consensus hub and erase the single liar pointer.

    Input structure:
    - The grid has a dominant background and several isolated two-cell pointers.
    - Each pointer consists of one shared grey tail cell and one colored head cell adjacent orthogonally.
    - The tail-to-head direction defines a horizontal or vertical ray; all but one ray pass through one invisible background hub cell.

    Rule:
    - Vote over background cells by counting how many directed pointer rays pass through each cell.
    - Use the unique cell with all-but-one ray votes as the hub.
    - For every pointer whose directed ray reaches the hub, draw the head-to-hub segment in the head color, excluding the hub; color the hub grey.
    - Erase both cells of the unique pointer whose directed ray does not reach the hub.

    Key insight:
    - The common target is not drawn in the input; it is recovered only by directed majority-ray consensus, so aligned but backward pointers must be rejected.
    """
    g = np.asarray(grid, dtype=int)
    bg, hub, truthful, liar, _ = _solve_details(g)
    return _render_solution(g, bg, hub, truthful, liar)


# ---------------- generator ----------------

_GEN_MAX_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _truthful_pointer(
    side: str,
    hub: tuple[int, int],
    dist: int,
    color: int,
) -> Pointer:
    hr, hc = hub
    if side == "top":
        head = (hr - dist, hc)
        tail = (hr - dist - 1, hc)
        direction = (1, 0)
    elif side == "bottom":
        head = (hr + dist, hc)
        tail = (hr + dist + 1, hc)
        direction = (-1, 0)
    elif side == "left":
        head = (hr, hc - dist)
        tail = (hr, hc - dist - 1)
        direction = (0, 1)
    elif side == "right":
        head = (hr, hc + dist)
        tail = (hr, hc + dist + 1)
        direction = (0, -1)
    else:
        raise ValueError(f"unknown side {side!r}")
    return Pointer(tail, head, direction, color)


def _max_truth_distance(side: str, shape: tuple[int, int], hub: tuple[int, int]) -> int:
    h, w = shape
    hr, hc = hub
    if side == "top":
        return hr - 1
    if side == "bottom":
        return h - hr - 2
    if side == "left":
        return hc - 1
    if side == "right":
        return w - hc - 2
    raise ValueError(f"unknown side {side!r}")


def _cells_too_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def _candidate_clear(
    tail: tuple[int, int],
    head: tuple[int, int],
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> bool:
    if tail in occupied or head in occupied or tail in protected or head in protected:
        return False
    for cell in (tail, head):
        for old in occupied:
            if _cells_too_close(cell, old):
                return False
    return True


def _liar_mode(pointer: Pointer, hub: tuple[int, int]) -> str | None:
    hr, hc = pointer.head
    dr, dc = pointer.direction
    if _ray_contains(pointer, hub):
        return None
    if dr == 0 and hr == hub[0] and (hub[1] - hc) * dc < 0:
        return "backward_aligned"
    if dc == 0 and hc == hub[1] and (hub[0] - hr) * dr < 0:
        return "backward_aligned"
    if (dr == 0 and hr != hub[0]) or (dc == 0 and hc != hub[1]):
        return "off_axis"
    return None


def _make_input(
    shape: tuple[int, int],
    bg: int,
    pointers: list[Pointer],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for pointer in pointers:
        tr, tc = pointer.tail
        hr, hc = pointer.head
        assert int(grid[tr, tc]) == bg and int(grid[hr, hc]) == bg
        grid[tr, tc] = GRAY
        grid[hr, hc] = pointer.color
    return grid


def _known_expected(
    shape: tuple[int, int],
    bg: int,
    truthful: list[Pointer],
    liar: Pointer,
    hub: tuple[int, int],
) -> np.ndarray:
    inp = _make_input(shape, bg, truthful + [liar])
    return _render_solution(inp, bg, hub, truthful, liar)


def _directionless_reaches(pointer: Pointer, cell: tuple[int, int]) -> bool:
    hr, hc = pointer.head
    r, c = cell
    dr, dc = pointer.direction
    return (dr == 0 and r == hr) or (dc == 0 and c == hc)


def _enumerate_liars(
    shape: tuple[int, int],
    hub: tuple[int, int],
    color: int,
    wanted_mode: str,
    occupied: set[tuple[int, int]],
    protected: set[tuple[int, int]],
) -> list[Pointer]:
    h, w = shape
    options: list[Pointer] = []
    for tr in range(h):
        for tc in range(w):
            for dr, dc in DIRS:
                hr, hc = tr + dr, tc + dc
                if not _in_bounds(shape, hr, hc):
                    continue
                pointer = Pointer((tr, tc), (hr, hc), (dr, dc), color)
                if _liar_mode(pointer, hub) != wanted_mode:
                    continue
                if not _candidate_clear(pointer.tail, pointer.head, occupied, protected):
                    continue
                options.append(pointer)
    random.shuffle(options)
    return options


def _sample_truthful(
    shape: tuple[int, int],
    hub: tuple[int, int],
    colors: list[int],
) -> list[Pointer] | None:
    sides = ["top", "bottom", "left", "right"]
    n_truthful = random.choice([3, 4])
    chosen = random.sample(sides, n_truthful)
    truthful: list[Pointer] = []
    for side, color in zip(chosen, colors):
        max_dist = min(_max_truth_distance(side, shape, hub), 6)
        if max_dist < 2:
            return None
        dist = random.randint(2, max_dist)
        truthful.append(_truthful_pointer(side, hub, dist, color))
    return truthful


def _semantic_accepts(
    inp: np.ndarray,
    expected: np.ndarray,
    hub: tuple[int, int],
    truthful: list[Pointer],
    liar: Pointer,
    wanted_mode: str,
) -> bool:
    try:
        bg, solved_hub, solved_truthful, solved_liar, solved_pointers = _solve_details(inp)
        actual = main(inp)
    except AssertionError:
        return False
    if solved_hub != hub or solved_liar != liar:
        return False
    if set(solved_truthful) != set(truthful):
        return False
    if _liar_mode(liar, hub) != wanted_mode:
        return False
    if not np.array_equal(actual, expected):
        return False

    changed_to_color = int(np.sum((actual != inp) & (actual != bg)))
    erased = int(np.sum((inp != bg) & (actual == bg)))
    if changed_to_color < len(truthful) + 1 or erased != 2:
        return False

    if wanted_mode == "backward_aligned":
        undirected_votes = sum(_directionless_reaches(pointer, hub) for pointer in solved_pointers)
        if undirected_votes != len(solved_pointers):
            return False
    return True


def _gen_attempt(wanted_mode: str) -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    shape = (h, w)
    hub = (random.randint(4, h - 5), random.randint(4, w - 5))
    bg = random.choice([c for c in ARC_COLORS if c != GRAY])
    palette = [c for c in ARC_COLORS if c not in (bg, GRAY)]
    random.shuffle(palette)

    truthful = _sample_truthful(shape, hub, palette)
    if truthful is None:
        return None
    liar_color = palette[len(truthful)]

    occupied = {cell for pointer in truthful for cell in (pointer.tail, pointer.head)}
    protected = {hub}
    for pointer in truthful:
        protected.update(_segment_to_hub(pointer, hub))

    for liar in _enumerate_liars(shape, hub, liar_color, wanted_mode, occupied, protected):
        inp = _make_input(shape, bg, truthful + [liar])
        expected = _known_expected(shape, bg, truthful, liar, hub)
        if _semantic_accepts(inp, expected, hub, truthful, liar, wanted_mode):
            return inp
    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    modes = ["backward_aligned", "off_axis", "backward_aligned", "off_axis"]
    start = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    preferred = modes[start % len(modes)]
    mode_order = [preferred, "off_axis" if preferred == "backward_aligned" else "backward_aligned"]

    for wanted_mode in mode_order:
        for _ in range(_GEN_MAX_ATTEMPTS):
            grid = _gen_attempt(wanted_mode)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid invisible-hub pointer input")
