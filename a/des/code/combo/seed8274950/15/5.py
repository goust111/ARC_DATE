from __future__ import annotations

from collections import deque

import numpy as np


BLUE = 1
RED = 2
DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
INPUT_COLORS = [0, 3, 4, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_chain(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    coord_set = set(coords)
    assert len(coord_set) >= 2, "chain must contain at least two pixels"

    degree: dict[tuple[int, int], int] = {}
    for r, c in coord_set:
        degree[(r, c)] = sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))

    assert all(d in (1, 2) for d in degree.values()), "chain may not branch or thicken"
    endpoints = sorted(cell for cell, d in degree.items() if d == 1)
    assert len(endpoints) == 2, "chain must have exactly two endpoints"

    ordered = [endpoints[0]]
    prev: tuple[int, int] | None = None
    curr = endpoints[0]
    while True:
        nxts = [
            nbr
            for nbr in _neighbors4(curr[0], curr[1])
            if nbr in coord_set and nbr != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "chain traversal must be unambiguous"
        prev, curr = curr, nxts[0]
        ordered.append(curr)

    assert len(ordered) == len(coord_set), "chain must be a single simple path"
    assert ordered[-1] == endpoints[1], "traversal must end at the other endpoint"
    return ordered


def _turn_marker(prev: tuple[int, int], curr: tuple[int, int], nxt: tuple[int, int]) -> int | None:
    dr1 = curr[0] - prev[0]
    dc1 = curr[1] - prev[1]
    dr2 = nxt[0] - curr[0]
    dc2 = nxt[1] - curr[1]
    cross = dr1 * dc2 - dc1 * dr2
    if cross > 0:
        return BLUE
    if cross < 0:
        return RED
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark each bend of every thin 4-connected chain, using blue for left turns and
    red for right turns when traversed from the chain's upper-left endpoint.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)

    out = g.copy()
    for comp in _connected_components(g != bg):
        ordered = _ordered_chain(comp)
        for i in range(1, len(ordered) - 1):
            marker = _turn_marker(ordered[i - 1], ordered[i], ordered[i + 1])
            if marker is not None:
                r, c = ordered[i]
                out[r, c] = marker
    return out


def _valid_next_cell(cell: tuple[int, int], prev: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    if cell in occupied:
        return False
    for nbr in _neighbors4(cell[0], cell[1]):
        if nbr in occupied and nbr != prev:
            return False
    return True


def _build_relative_chain(rng: np.random.Generator, require_both_turns: bool) -> list[tuple[int, int]] | None:
    min_turns = 2 if require_both_turns else 1
    for _ in range(200):
        n_turns = int(rng.integers(min_turns, 6))
        turns = rng.choice([-1, 1], size=n_turns)
        if require_both_turns and set(map(int, turns)) != {-1, 1}:
            continue

        lengths = rng.integers(1, 5, size=n_turns + 1)
        dir_idx = int(rng.integers(0, 4))
        current = (0, 0)
        cells = [current]
        occupied = {current}
        ok = True

        for seg_idx, seg_len in enumerate(lengths):
            dr, dc = DIR4[dir_idx]
            for _step in range(int(seg_len)):
                nxt = (current[0] + dr, current[1] + dc)
                if not _valid_next_cell(nxt, current, occupied):
                    ok = False
                    break
                occupied.add(nxt)
                cells.append(nxt)
                current = nxt
            if not ok:
                break
            if seg_idx < n_turns:
                dir_idx = (dir_idx + int(turns[seg_idx])) % 4

        if not ok:
            continue

        if len(cells) < 6:
            continue

        try:
            ordered = _ordered_chain(cells)
        except AssertionError:
            continue

        markers = [
            _turn_marker(ordered[i - 1], ordered[i], ordered[i + 1])
            for i in range(1, len(ordered) - 1)
        ]
        if require_both_turns and not ({BLUE, RED} <= set(m for m in markers if m is not None)):
            continue
        return cells
    return None


def _place_relative_chain(
    rng: np.random.Generator,
    shape: list[tuple[int, int]],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    if max_r - min_r + 1 > h or max_c - min_c + 1 > w:
        return None

    placements: list[list[tuple[int, int]]] = []
    interior: list[list[tuple[int, int]]] = []
    for roff in range(-min_r, h - max_r):
        for coff in range(-min_c, w - max_c):
            placed = [(r + roff, c + coff) for r, c in shape]
            placed_set = set(placed)
            if placed_set & occupied:
                continue
            if any(nbr in occupied for r, c in placed for nbr in _neighbors4(r, c)):
                continue
            placements.append(placed)
            if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in placed):
                interior.append(placed)

    pool = interior if interior else placements
    if not pool:
        return None
    return pool[int(rng.integers(0, len(pool)))]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_chains = int(rng.integers(1, 4))
    h = int(rng.integers(8 + 2 * n_chains, 19))
    w = int(rng.integers(8 + 2 * n_chains, 19))

    bg = int(rng.choice(INPUT_COLORS))
    fg_choices = [color for color in INPUT_COLORS if color != bg]
    use_shared_color = bool(rng.integers(0, 2))
    if use_shared_color:
        colors = [int(rng.choice(fg_choices))] * n_chains
    else:
        colors = [int(c) for c in rng.choice(fg_choices, size=n_chains, replace=False)]

    grid = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for chain_idx in range(n_chains):
        placed = None
        for _ in range(120):
            shape = _build_relative_chain(rng, require_both_turns=chain_idx == 0)
            if shape is None:
                continue
            placed = _place_relative_chain(rng, shape, h, w, occupied)
            if placed is not None:
                break
        if placed is None:
            return None
        for r, c in placed:
            grid[r, c] = colors[chain_idx]
        occupied.update(placed)

    if np.count_nonzero(grid != bg) * 2 >= grid.size:
        return None

    out = main(grid)
    if not bool((out == BLUE).any()) or not bool((out == RED).any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {MAX_RETRIES} attempts")
