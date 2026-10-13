from __future__ import annotations

import random

import numpy as np


WALL = 0
GATE = 5
PIPE = 7
PALETTE = (1, 2, 3, 4, 6, 8, 9)


def _find_gate_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    h, w = grid.shape
    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) != GATE:
                c += 1
                continue
            start = c
            while c + 1 < w and int(grid[r, c + 1]) == GATE:
                c += 1
            end = c
            if end - start + 1 == 3:
                centers.append((int(r), int(start + 1)))
            c += 1
    return centers


def _open_gates(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, _w = g.shape
    for r, c in _find_gate_centers(g):
        g[r, c - 1 : c + 2] = PIPE
        if r - 1 >= 0:
            g[r - 1, c] = WALL
        if r + 1 < h:
            g[r + 1, c] = WALL
    return g


def _segment_bounds(base: np.ndarray, occupied: np.ndarray, r: int, c: int) -> tuple[int, int]:
    left = int(c)
    while left - 1 >= 0 and int(base[r, left - 1]) == PIPE and not bool(occupied[r, left - 1]):
        left -= 1
    right = int(c)
    w = base.shape[1]
    while right + 1 < w and int(base[r, right + 1]) == PIPE and not bool(occupied[r, right + 1]):
        right += 1
    return int(left), int(right)


def _entry_side(left: int, right: int, c: int) -> str:
    if c <= left:
        return "left"
    if c >= right:
        return "right"
    return "left" if c - left <= right - c else "right"


def _drop_token(base: np.ndarray, occupied: np.ndarray, start: tuple[int, int]) -> tuple[int, int]:
    h, _w = base.shape
    r, c = map(int, start)
    while True:
        while r + 1 < h and int(base[r + 1, c]) == PIPE and not bool(occupied[r + 1, c]):
            r += 1
        if r + 1 >= h:
            return int(r), int(c)
        if bool(occupied[r + 1, c]):
            return int(r), int(c)

        left, right = _segment_bounds(base, occupied, r, c)
        candidates = [
            cc
            for cc in range(left, right + 1)
            if r + 1 < h and int(base[r + 1, cc]) == PIPE and not bool(occupied[r + 1, cc])
        ]
        if not candidates:
            return int(r), int(c)

        side = _entry_side(left, right, c)
        c = int(candidates[0] if side == "left" else candidates[-1])


def _simulate(base: np.ndarray, tokens: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    occupied = np.zeros(base.shape, dtype=bool)
    settled: list[tuple[int, int, int]] = []
    for r, c, color in sorted(tokens, key=lambda t: (t[1], t[0])):
        rr, cc = _drop_token(base, occupied, (int(r), int(c)))
        occupied[rr, cc] = True
        settled.append((int(rr), int(cc), int(color)))
    return settled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the pipe gates, drop the tokens through the routed network, and recolor them from the exit legend.

    Input structure:
    - Black (0) walls surround orange (7) pipe interior cells.
    - Grey (5) horizontal 1x3 bars are closed gates inside the pipe network.
    - Non-wall/non-pipe colored pixels on the top row are tokens; colored pixels on the bottom row are legend entries.

    Rule:
    - Rotate every 1x3 gate into an open state: the three gate cells become orange, while the gate center's cell above and below become black walls.
    - Process top-row tokens from left to right. Each token falls through orange cells; when a horizontal run offers multiple downward continuations, the token chooses the extreme continuation on the side it entered from (left end -> leftmost, right end -> rightmost). A settled token blocks later tokens, and collision with a settled token stops the fall immediately.
    - After a token stops, recolor it with the color found in the same column on the bottom-row legend; if that column has no legend entry, keep the token's original color.
    - Remove all legend pixels from the bottom row by turning them back to orange.

    Key insight:
    - Gate rotation changes the pipe topology, the changed topology determines the token's final column, and that final column alone determines the token's final color via the bottom legend.
    """
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    tokens = [(0, int(c), int(g[0, c])) for c in range(w) if int(g[0, c]) not in (WALL, GATE, PIPE)]
    legends = {int(c): int(g[h - 1, c]) for c in range(w) if int(g[h - 1, c]) not in (WALL, GATE, PIPE)}

    for _r, c, _color in tokens:
        g[0, c] = PIPE
    opened = _open_gates(g)

    settled = _simulate(opened, tokens)
    out = np.array(opened, copy=True)
    for r, c, color in settled:
        out[r, c] = int(legends.get(int(c), int(color)))
    for c in legends:
        out[h - 1, c] = PIPE
    return out


def _build_opened_layout(kind: str) -> tuple[np.ndarray, list[tuple[int, int]], list[int]]:
    if kind == "three_gate":
        w = random.randint(14, 17)
        h = random.randint(8, 10)
        c1 = random.randint(4, w - 10)
        c2 = random.randint(c1 + 5, w - 5)
        c3 = random.randint(c1 + 2, c2 - 2)
        opened = np.full((h, w), WALL, dtype=int)
        opened[0, 1 : w - 1] = PIPE
        opened[0, c1] = WALL
        opened[0, c2] = WALL
        opened[1, c1 - 1 : c1 + 2] = PIPE
        opened[1, c2 - 1 : c2 + 2] = PIPE
        opened[2, 1 : w - 1] = PIPE
        opened[2, c1] = WALL
        opened[2, c2] = WALL
        opened[3, [c1 - 1, c2 + 1]] = PIPE
        opened[3, c3 - 1 : c3 + 2] = PIPE
        exit_cols = [c1 - 1, c3 - 1, c3 + 1, c2 + 1]
        for r in range(4, h - 1):
            for c in exit_cols:
                opened[r, c] = PIPE
        gates = [(1, c1), (1, c2), (3, c3)]
        return opened, gates, exit_cols

    w = random.randint(12, 17)
    h = random.randint(7, 10)
    c1 = random.randint(4, w - 8)
    c2 = random.randint(c1 + 4, w - 4)
    opened = np.full((h, w), WALL, dtype=int)
    opened[0, 1 : w - 1] = PIPE
    opened[0, c1] = WALL
    opened[0, c2] = WALL
    opened[1, c1 - 1 : c1 + 2] = PIPE
    opened[1, c2 - 1 : c2 + 2] = PIPE
    opened[2, 1 : w - 1] = PIPE
    opened[2, c1] = WALL
    opened[2, c2] = WALL
    exit_cols = [c1 - 1, c1 + 1, c2 - 1, c2 + 1]
    for r in range(3, h - 1):
        for c in exit_cols:
            opened[r, c] = PIPE
    gates = [(1, c1), (1, c2)]
    return opened, gates, exit_cols


def _opened_to_input(opened: np.ndarray, gates: list[tuple[int, int]]) -> np.ndarray:
    inp = np.array(opened, dtype=int, copy=True)
    h, _w = inp.shape
    for r, c in gates:
        inp[r, c - 1 : c + 2] = GATE
        if r - 1 >= 0:
            inp[r - 1, c] = PIPE
        if r + 1 < h:
            inp[r + 1, c] = PIPE
    return inp


def _choose_token_columns(kind: str, opened: np.ndarray, gates: list[tuple[int, int]]) -> list[int]:
    _h, w = opened.shape
    row1_centers = sorted(c for r, c in gates if r == 1)
    c1, c2 = row1_centers[0], row1_centers[1]

    left_dup = [1, c1 - 1]
    right_dup = [c2 + 1, w - 2]
    middle_split = [c1 + 1, c2 - 1]
    distinct_three = [random.choice((1, c1 - 1)), c1 + 1, random.choice((c2 + 1, w - 2))]

    if kind == "three_gate":
        profiles = [
            middle_split,
            middle_split + [random.choice((1, w - 2))],
            left_dup + [c2 - 1],
            [c1 + 1] + right_dup,
            distinct_three,
        ]
    else:
        profiles = [
            middle_split,
            middle_split + [random.choice((1, w - 2))],
            left_dup + [c2 - 1],
            [c1 + 1] + right_dup,
            distinct_three,
        ]
    return [int(c) for c in random.choice(profiles)]


def _route_columns(inp: np.ndarray) -> list[int]:
    opened = np.array(inp, copy=True)
    tokens = [(0, int(c), 1) for c in range(inp.shape[1]) if int(inp[0, c]) not in (WALL, GATE, PIPE)]
    for _r, c, _color in tokens:
        opened[0, c] = PIPE
    opened = _open_gates(opened)
    return [int(c) for _r, c, _color in _simulate(opened, tokens)]


def generate_input() -> np.ndarray:
    for _ in range(300):
        kind = "three_gate" if random.random() < 0.55 else "two_gate"
        opened, gates, exit_cols = _build_opened_layout(kind)
        inp = _opened_to_input(opened, gates)

        token_cols = _choose_token_columns(kind, opened, gates)
        token_count = len(token_cols)
        if not (2 <= token_count <= 3):
            continue

        palette = list(PALETTE)
        random.shuffle(palette)
        token_colors = palette[:token_count]
        legend_colors = palette[token_count : token_count + len(exit_cols)]
        if len(set(token_colors)) != token_count or len(set(legend_colors)) != len(exit_cols):
            continue

        for c, color in zip(token_cols, token_colors):
            inp[0, int(c)] = int(color)
        inp[-1, :] = PIPE
        for c, color in zip(exit_cols, legend_colors):
            inp[-1, int(c)] = int(color)

        routed_cols = _route_columns(inp)
        if len(routed_cols) != token_count:
            continue
        if len(set(routed_cols)) < 2:
            continue
        if middle_split := sorted(set(token_cols) & {sorted(c for r, c in gates if r == 1)[0] + 1, sorted(c for r, c in gates if r == 1)[1] - 1}):
            if len(middle_split) == 2 and len(set(routed_cols)) < 2:
                continue

        out = main(inp)
        if out.shape != inp.shape:
            continue
        if any(int(out[-1, c]) != PIPE for c in exit_cols):
            continue
        if not np.any(out != inp):
            continue
        return inp

    raise RuntimeError("failed to generate input for task 8")
