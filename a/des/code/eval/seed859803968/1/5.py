import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
LIQUID_COLORS = [1, 2, 3, 4, 6, 7, 8]


def _is_liquid(value: int) -> bool:
    return int(value) not in (BLACK, GREY, MAROON)


def _find_containers(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    containers: list[dict] = []
    seen: set[tuple[int, int, int]] = set()

    for bottom in range(3, h):
        c = 0
        while c < w:
            color = int(grid[bottom, c])
            if color not in (GREY, MAROON):
                c += 1
                continue
            c0 = c
            while c < w and int(grid[bottom, c]) == color:
                c += 1
            c1 = c - 1
            if c1 - c0 + 1 < 5 or (bottom, c0, c1) in seen:
                continue

            top = bottom
            rr = bottom - 1
            while rr >= 0 and int(grid[rr, c0]) == GREY and int(grid[rr, c1]) == GREY:
                top = rr
                rr -= 1
            if bottom - top < 4 or c1 - c0 < 4:
                continue

            seen.add((bottom, c0, c1))
            containers.append(
                {
                    "top": top,
                    "bottom": bottom,
                    "c0": c0,
                    "c1": c1,
                    "terminal": color == MAROON,
                }
            )

    containers.sort(key=lambda item: (item["bottom"], item["c0"]))
    return containers


def _stack_for(grid: np.ndarray, container: dict) -> list[int]:
    stack: list[int] = []
    c0, c1 = int(container["c0"]), int(container["c1"])
    for r in range(int(container["bottom"]) - 1, int(container["top"]) - 1, -1):
        row = grid[r, c0 + 1 : c1]
        liquids = [int(x) for x in row if _is_liquid(int(x))]
        if not liquids:
            break
        counts: dict[int, int] = defaultdict(int)
        for color in liquids:
            counts[color] += 1
        stack.append(max(counts, key=counts.get))
    return stack


def _pipe_connections(grid: np.ndarray, containers: list[dict]) -> dict[int, list[tuple[int, int]]]:
    conns: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, a in enumerate(containers):
        for j, b in enumerate(containers):
            if i == j:
                continue
            top = max(int(a["top"]), int(b["top"]))
            bottom = min(int(a["bottom"]), int(b["bottom"])) - 1
            if bottom < top:
                continue
            for r in range(top, bottom + 1):
                if int(a["c1"]) < int(b["c0"]):
                    segment = grid[r, int(a["c1"]) : int(b["c0"]) + 1]
                elif int(b["c1"]) < int(a["c0"]):
                    segment = grid[r, int(b["c1"]) : int(a["c0"]) + 1]
                else:
                    continue
                if len(segment) >= 3 and np.all(segment == GREY):
                    conns[i].append((j, r))
                    break
    return conns


def _clear_liquids(out: np.ndarray, containers: list[dict]) -> None:
    for c in containers:
        for r in range(int(c["top"]), int(c["bottom"])):
            for col in range(int(c["c0"]) + 1, int(c["c1"])):
                if _is_liquid(int(out[r, col])):
                    out[r, col] = BLACK


def _paint_stack(out: np.ndarray, container: dict, stack: list[int]) -> None:
    c0, c1 = int(container["c0"]), int(container["c1"])
    capacity = int(container["bottom"]) - int(container["top"])
    for offset, color in enumerate(stack[:capacity]):
        r = int(container["bottom"]) - 1 - offset
        out[r, c0 + 1 : c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    containers = _find_containers(g)
    if not containers:
        return g.copy()

    terminal_indices = [i for i, c in enumerate(containers) if c["terminal"]]
    terminal_idx = terminal_indices[0] if terminal_indices else len(containers) - 1
    stacks = [_stack_for(g, c) for c in containers]
    conns = _pipe_connections(g, containers)

    transfers: list[tuple[int, int, int]] = []
    for i, c in enumerate(containers):
        if i == terminal_idx or not stacks[i]:
            continue
        possible = [(j, r) for j, r in conns.get(i, []) if j == terminal_idx]
        if not possible:
            continue
        target, pipe_r = max(possible, key=lambda item: item[1])
        surface_r = int(c["bottom"]) - len(stacks[i])
        if pipe_r < surface_r:
            continue
        amount = min(len(stacks[i]), pipe_r - surface_r + 1)
        removed = stacks[i][-amount:]
        stacks[i] = stacks[i][:-amount]
        center_col = (int(c["c0"]) + int(c["c1"])) // 2
        for color in removed:
            transfers.append((center_col, target, color))

    transfers.sort(key=lambda item: item[0])
    for _center, target, color in transfers:
        stacks[target].append(color)

    out = g.copy()
    _clear_liquids(out, containers)
    for container, stack in zip(containers, stacks):
        _paint_stack(out, container, stack)
    return out


def _draw_container(
    grid: np.ndarray,
    top: int,
    bottom: int,
    c0: int,
    c1: int,
    terminal: bool,
) -> dict:
    grid[top:bottom, c0] = GREY
    grid[top:bottom, c1] = GREY
    grid[bottom, c0 : c1 + 1] = MAROON if terminal else GREY
    return {"top": top, "bottom": bottom, "c0": c0, "c1": c1, "terminal": terminal}


def _fill_liquid(grid: np.ndarray, container: dict, color: int, rows: int) -> None:
    bottom = int(container["bottom"])
    c0, c1 = int(container["c0"]), int(container["c1"])
    for r in range(bottom - rows, bottom):
        grid[r, c0 + 1 : c1] = int(color)


def _draw_pipe(grid: np.ndarray, row: int, a_col: int, b_col: int) -> None:
    lo, hi = sorted((a_col, b_col))
    grid[row, lo : hi + 1] = GREY


def generate_input() -> np.ndarray:
    for _ in range(300):
        top = random.randint(4, 6)
        height = random.randint(8, 10)
        bottom = top + height
        left_w = random.randint(5, 7)
        term_w = random.randint(8, 10)
        right_w = random.randint(5, 7)
        gap1 = random.randint(3, 5)
        gap2 = random.randint(3, 5)

        left_c0 = 2
        left_c1 = left_c0 + left_w
        term_c0 = left_c1 + gap1
        term_c1 = term_c0 + term_w
        right_c0 = term_c1 + gap2
        right_c1 = right_c0 + right_w
        h = bottom + random.randint(4, 7)
        w = right_c1 + 3
        grid = np.zeros((h, w), dtype=int)

        left = _draw_container(grid, top, bottom, left_c0, left_c1, False)
        terminal = _draw_container(grid, top + random.randint(0, 1), bottom + 1, term_c0, term_c1, True)
        right = _draw_container(grid, top, bottom, right_c0, right_c1, False)

        colors = random.sample(LIQUID_COLORS, 3)
        left_fill = random.randint(3, height - 2)
        right_fill = random.randint(3, height - 2)
        term_fill = random.randint(1, 3)
        _fill_liquid(grid, left, colors[0], left_fill)
        _fill_liquid(grid, right, colors[1], right_fill)
        _fill_liquid(grid, terminal, colors[2], term_fill)

        left_surface = int(left["bottom"]) - left_fill
        right_surface = int(right["bottom"]) - right_fill
        left_pipe = random.randint(left_surface, int(left["bottom"]) - 2)
        if random.random() < 0.25 and left_surface > int(left["top"]):
            left_pipe = random.randint(int(left["top"]), left_surface - 1)
        right_pipe = random.randint(right_surface, int(right["bottom"]) - 2)
        _draw_pipe(grid, left_pipe, int(left["c1"]), int(terminal["c0"]))
        _draw_pipe(grid, right_pipe, int(terminal["c1"]), int(right["c0"]))

        out = main(grid)
        if out.shape == grid.shape and np.any(out != grid):
            return grid
    raise RuntimeError("failed to generate valid input")
