from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
YELLOW = 4
BALL_COLORS = (1, 2, 3, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
GEN_GRID_H = (10, 18)
GEN_GRID_W = (14, 28)
GEN_CONTAINER_COUNT = (2, 4)
GEN_TOTAL_BALLS = (4, 7)
GEN_BALLS_PER_CONTAINER_MAX = 4
GEN_EXTRA_CAPACITY = (1, 3)
GEN_BOTTOM_ROW_GAP = 2
GEN_CONTAINER_MIN_SEP = 5
GEN_BAR_EDGE_MARGIN = 0


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_containers(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    containers: list[tuple[int, int, int, int]] = []
    for comp in _components_of_color(grid, GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if width != 3 or height < 2:
            continue
        expected = {(r, c0) for r in range(r0, r1 + 1)}
        expected |= {(r, c1) for r in range(r0, r1 + 1)}
        expected.add((r1, c0 + 1))
        if set(comp) != expected:
            continue
        containers.append((int(r0), int(c0), int(r1), int(c1)))
    containers.sort(key=lambda box: (box[1], box[0]))
    return containers


def _instruction_order(grid: np.ndarray, bg: int) -> list[int]:
    row = grid[-1]
    cols = [c for c, value in enumerate(row) if int(value) not in {int(bg), GREY, YELLOW}]
    assert cols, "expected a bottom-row instruction bar"
    colors = [int(row[c]) for c in cols]
    assert len(colors) == len(set(colors)), "instruction colors must be unique"
    return colors


def _container_stack(grid: np.ndarray, box: tuple[int, int, int, int], bg: int) -> list[dict[str, int]]:
    r0, c0, r1, _c1 = box
    inner_c = c0 + 1
    items: list[dict[str, int]] = []
    seen_gap = False
    for idx, r in enumerate(range(r1 - 1, r0 - 1, -1)):
        value = int(grid[r, inner_c])
        if value == int(bg):
            seen_gap = True
            continue
        assert not seen_gap, "balls must be stacked contiguously from the bottom"
        assert value not in {GREY, YELLOW}, "container interior must hold ball colors only"
        items.append({"orig_index": int(idx), "old_r": int(r), "color": int(value)})
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder each container's stacked balls using the bottom instruction bar and mark moved balls.

    Input structure:
    - Black background.
    - Several grey U-shaped containers of outer width 3 and varying heights.
    - Each container holds a vertical, gap-free stack of colored 1x1 balls.
    - The bottom row contains a horizontal instruction bar; leftmost color has the highest priority.

    Rule:
    - Sort each container's balls from bottom to top by the global priority defined by the instruction bar.
    - A ball that ends lower than before gets a yellow marker on the left outside of its final row.
    - A ball that ends higher than before gets a yellow marker on the right outside of its final row.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    order = _instruction_order(g, bg)
    priority = {color: idx for idx, color in enumerate(order)}
    containers = _find_containers(g)
    assert containers, "expected at least one container"

    out = g.copy()
    for r0, c0, r1, c1 in containers:
        inner_c = c0 + 1
        items = _container_stack(g, (r0, c0, r1, c1), bg)
        out[r0:r1, inner_c] = bg

        sorted_items = sorted(items, key=lambda item: priority.get(int(item["color"]), len(priority)))
        for new_index, item in enumerate(sorted_items):
            new_r = int(r1 - 1 - new_index)
            out[new_r, inner_c] = int(item["color"])
            old_r = int(item["old_r"])
            if new_r > old_r and c0 - 1 >= 0:
                out[new_r, c0 - 1] = YELLOW
            elif new_r < old_r and c1 + 1 < g.shape[1]:
                out[new_r, c1 + 1] = YELLOW

    return out


def _partition_total(total: int, parts: int, max_each: int) -> list[int] | None:
    for _ in range(200):
        counts = [1] * parts
        remaining = total - parts
        while remaining > 0:
            choices = [i for i, count in enumerate(counts) if count < max_each]
            if not choices:
                break
            idx = random.choice(choices)
            counts[idx] += 1
            remaining -= 1
        if sum(counts) == total and all(1 <= count <= max_each for count in counts):
            random.shuffle(counts)
            return counts
    return None


def _sample_container_starts(width: int, count: int) -> list[int] | None:
    positions = list(range(1, width - 3))
    for _ in range(200):
        starts = sorted(random.sample(positions, count))
        if all(b - a >= GEN_CONTAINER_MIN_SEP for a, b in zip(starts, starts[1:])):
            return starts
    return None


def _movement_counts(stacks: list[list[int]], priority: dict[int, int]) -> tuple[int, int, int]:
    down = up = stationary = 0
    for stack in stacks:
        items = [{"orig_index": i, "color": color} for i, color in enumerate(stack)]
        sorted_items = sorted(items, key=lambda item: priority[int(item["color"])])
        for new_index, item in enumerate(sorted_items):
            if new_index < int(item["orig_index"]):
                down += 1
            elif new_index > int(item["orig_index"]):
                up += 1
            else:
                stationary += 1
    return down, up, stationary


def _draw_container(grid: np.ndarray, r0: int, c0: int, r1: int) -> None:
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c0 + 2] = GREY
    grid[r1, c0 : c0 + 3] = GREY


def _build_candidate(
    height: int,
    width: int,
    starts: list[int],
    bottoms: list[int],
    capacities: list[int],
    stacks: list[list[int]],
    instruction: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for c_start, bottom, capacity, stack in zip(starts, bottoms, capacities, stacks):
        top = int(bottom - capacity)
        _draw_container(grid, top, c_start, bottom)
        inner_c = c_start + 1
        for idx, color in enumerate(stack):
            grid[bottom - 1 - idx, inner_c] = int(color)

    bar_start = random.randint(GEN_BAR_EDGE_MARGIN, width - len(instruction) - GEN_BAR_EDGE_MARGIN)
    grid[height - 1, bar_start : bar_start + len(instruction)] = np.asarray(instruction, dtype=int)
    return grid


def _gen_attempt() -> np.ndarray | None:
    container_count = random.randint(*GEN_CONTAINER_COUNT)
    total_balls = random.randint(max(container_count, GEN_TOTAL_BALLS[0]), min(len(BALL_COLORS), GEN_TOTAL_BALLS[1]))
    counts = _partition_total(total_balls, container_count, GEN_BALLS_PER_CONTAINER_MAX)
    if counts is None:
        return None

    capacities = [count + random.randint(*GEN_EXTRA_CAPACITY) for count in counts]
    min_height = max(GEN_GRID_H[0], max(capacities) + GEN_BOTTOM_ROW_GAP + 2)
    max_height = GEN_GRID_H[1]
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    min_width = max(GEN_GRID_W[0], len(instruction := random.sample(random.sample(BALL_COLORS, total_balls), total_balls)), 5 * container_count + 1)
    max_width = GEN_GRID_W[1]
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)
    starts = _sample_container_starts(width, container_count)
    if starts is None:
        return None

    bottoms: list[int] = []
    for capacity in capacities:
        low = capacity + 1
        high = height - 1 - GEN_BOTTOM_ROW_GAP
        if low > high:
            return None
        bottoms.append(random.randint(low, high))

    assigned = random.sample(instruction, len(instruction))
    stacks: list[list[int]] = []
    offset = 0
    for count in counts:
        colors = assigned[offset : offset + count]
        offset += count
        random.shuffle(colors)
        stacks.append(colors)

    priority = {color: idx for idx, color in enumerate(instruction)}
    down, up, stationary = _movement_counts(stacks, priority)
    if down == 0 or up == 0:
        return None
    if random.random() < 0.5 and stationary == 0:
        return None

    grid = _build_candidate(height, width, starts, bottoms, capacities, stacks, instruction)

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
