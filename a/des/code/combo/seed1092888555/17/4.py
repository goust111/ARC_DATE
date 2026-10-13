import random
from collections import defaultdict

import numpy as np


BG = 0
GRAY = 5
SOURCE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_gray_color(grid: np.ndarray) -> int:
    candidates = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != BG):
        count = 0
        for size in (2, 3):
            for r in range(grid.shape[0] - size + 1):
                for c in range(grid.shape[1] - size + 1):
                    if np.all(grid[r : r + size, c : c + size] == color):
                        count += 1
        if count:
            candidates.append((count, color))
    assert candidates, "expected at least one gray square receiver candidate"
    return max(candidates)[1]


def _detect_receivers(grid: np.ndarray, gray: int) -> list[dict]:
    candidates = []
    for size in (3, 2):
        for r in range(grid.shape[0] - size + 1):
            for c in range(grid.shape[1] - size + 1):
                if np.all(grid[r : r + size, c : c + size] == gray):
                    candidates.append((size, r, c))

    candidates.sort(reverse=True)
    receivers = []
    covered = np.zeros(grid.shape, dtype=bool)
    for size, r, c in candidates:
        region = covered[r : r + size, c : c + size]
        if np.any(region):
            continue
        receivers.append(
            {
                "size": size,
                "bbox": (r, c, r + size - 1, c + size - 1),
                "mask": {(rr, cc) for rr in range(r, r + size) for cc in range(c, c + size)},
            }
        )
        covered[r : r + size, c : c + size] = True

    receiver_ids = np.full(grid.shape, -1, dtype=int)
    for idx, receiver in enumerate(receivers):
        r0, c0, r1, c1 = receiver["bbox"]
        receiver_ids[r0 : r1 + 1, c0 : c1 + 1] = idx
    for receiver in receivers:
        r0, c0, r1, c1 = receiver["bbox"]
        has_wire_contact = False
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == BG:
                        has_wire_contact = True
        assert has_wire_contact, "receiver should touch background channel space"
    return receivers


def _is_wire_cell(grid: np.ndarray, r: int, c: int, gray: int, receiver_ids: np.ndarray) -> bool:
    if int(grid[r, c]) != BG or int(receiver_ids[r, c]) != -1:
        return False
    up = r - 1 >= 0 and int(grid[r - 1, c]) == gray
    down = r + 1 < grid.shape[0] and int(grid[r + 1, c]) == gray
    left = c - 1 >= 0 and int(grid[r, c - 1]) == gray
    right = c + 1 < grid.shape[1] and int(grid[r, c + 1]) == gray
    return (up and down) or (left and right)


def _collect_component(mask: np.ndarray, start: tuple[int, int]) -> set[tuple[int, int]]:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    gray = _find_gray_color(g)
    receivers = _detect_receivers(g, gray)
    assert receivers, "expected at least one receiver"

    receiver_ids = np.full(g.shape, -1, dtype=int)
    for idx, receiver in enumerate(receivers):
        r0, c0, r1, c1 = receiver["bbox"]
        receiver_ids[r0 : r1 + 1, c0 : c1 + 1] = idx

    wire_mask = np.zeros(g.shape, dtype=bool)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            wire_mask[r, c] = _is_wire_cell(g, r, c, gray, receiver_ids)

    source_cells = [
        (r, c)
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if int(g[r, c]) not in (BG, gray)
    ]
    assert source_cells, "expected at least one signal source"

    receiver_signals: dict[int, list[tuple[tuple[int, int], int]]] = defaultdict(list)
    used_wire_cells: set[tuple[int, int]] = set()

    for sr, sc in source_cells:
        adjacent_wires = []
        for dr, dc in DIRS4:
            nr, nc = sr + dr, sc + dc
            if _in_bounds(g.shape, nr, nc) and bool(wire_mask[nr, nc]):
                adjacent_wires.append((nr, nc))
        assert len(adjacent_wires) == 1, "each source should attach to exactly one wire"

        start = adjacent_wires[0]
        if start in used_wire_cells:
            continue
        component = _collect_component(wire_mask, start)
        used_wire_cells |= component

        touched = []
        for wr, wc in component:
            for dr, dc in DIRS4:
                nr, nc = wr + dr, wc + dc
                if _in_bounds(g.shape, nr, nc) and int(receiver_ids[nr, nc]) >= 0:
                    touched.append((int(receiver_ids[nr, nc]), (nr, nc)))
        assert touched, "wire should connect to a receiver"
        touched.sort(key=lambda x: (x[1][0], x[1][1], x[0]))
        receiver_id, contact = touched[0]
        assert all(t[0] == receiver_id for t in touched), "wire should touch one receiver only"
        receiver_signals[receiver_id].append((contact, int(g[sr, sc])))

    out = g.copy()
    for rid, receiver in enumerate(receivers):
        assert rid in receiver_signals, "every receiver should have at least one source"
        signals = sorted(receiver_signals[rid], key=lambda item: (item[0][0], item[0][1]))
        ordered_colors = []
        for _, color in signals:
            if color not in ordered_colors:
                ordered_colors.append(color)

        r0, c0, r1, c1 = receiver["bbox"]
        size = receiver["size"]
        if len(ordered_colors) == 1:
            out[r0 : r1 + 1, c0 : c1 + 1] = ordered_colors[0]
            continue

        top_color, bottom_color = ordered_colors[:2]
        split = size // 2
        out[r0 : r0 + split, c0 : c1 + 1] = top_color
        out[r0 + split : r1 + 1, c0 : c1 + 1] = bottom_color

    return out


def _expanded_box_free(used: np.ndarray, r0: int, c0: int, r1: int, c1: int, margin: int) -> bool:
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(used.shape[0] - 1, r1 + margin)
    cc1 = min(used.shape[1] - 1, c1 + margin)
    return not np.any(used[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _reserve(used: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        used[r, c] = True


def _wire_plan_for_side(
    bbox: tuple[int, int, int, int], side: str, slot: int, length: int
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    if side == "left":
        source = (slot, c0 - length - 1)
        corridor = [(slot, c) for c in range(c0 - length, c0)]
        walls = [(slot - 1, c) for c in range(c0 - length, c0)] + [(slot + 1, c) for c in range(c0 - length, c0)]
        return source, corridor, walls
    if side == "right":
        source = (slot, c1 + length + 1)
        corridor = [(slot, c) for c in range(c1 + 1, c1 + length + 1)]
        walls = [(slot - 1, c) for c in range(c1 + 1, c1 + length + 1)] + [(slot + 1, c) for c in range(c1 + 1, c1 + length + 1)]
        return source, corridor, walls
    if side == "up":
        source = (r0 - length - 1, slot)
        corridor = [(r, slot) for r in range(r0 - length, r0)]
        walls = [(r, slot - 1) for r in range(r0 - length, r0)] + [(r, slot + 1) for r in range(r0 - length, r0)]
        return source, corridor, walls
    source = (r1 + length + 1, slot)
    corridor = [(r, slot) for r in range(r1 + 1, r1 + length + 1)]
    walls = [(r, slot - 1) for r in range(r1 + 1, r1 + length + 1)] + [(r, slot + 1) for r in range(r1 + 1, r1 + length + 1)]
    return source, corridor, walls


def _place_receiver(grid: np.ndarray, used: np.ndarray, size: int) -> tuple[int, int, int, int] | None:
    for _ in range(100):
        r0 = random.randint(3, grid.shape[0] - size - 4)
        c0 = random.randint(3, grid.shape[1] - size - 4)
        r1, c1 = r0 + size - 1, c0 + size - 1
        if _expanded_box_free(used, r0, c0, r1, c1, 3):
            grid[r0 : r1 + 1, c0 : c1 + 1] = GRAY
            used[r0 : r1 + 1, c0 : c1 + 1] = True
            return (r0, c0, r1, c1)
    return None


def _try_attach_wire(
    grid: np.ndarray,
    used: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    color: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    if side in ("left", "right"):
        slots = list(range(r0, r1 + 1))
    else:
        slots = list(range(c0, c1 + 1))
    random.shuffle(slots)
    lengths = list(range(2, 6))
    random.shuffle(lengths)

    for slot in slots:
        for length in lengths:
            source, corridor, walls = _wire_plan_for_side(bbox, side, slot, length)
            cells = [source] + corridor + walls
            if not all(_in_bounds(grid.shape, r, c) for r, c in cells):
                continue
            if any(used[r, c] for r, c in cells):
                continue
            if any(int(grid[r, c]) != BG for r, c in [source] + walls):
                continue
            _reserve(used, cells)
            sr, sc = source
            grid[sr, sc] = color
            for wr, wc in walls:
                grid[wr, wc] = GRAY
            return source, corridor[-1]
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 20)
    width = random.randint(12, 20)
    grid = np.full((height, width), BG, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    receiver_count = random.randint(1, 3)
    receivers = []
    sizes = [random.choice([2, 3]) for _ in range(receiver_count)]
    for size in sizes:
        bbox = _place_receiver(grid, used, size)
        if bbox is None:
            return None
        receivers.append(bbox)

    any_two = False
    for bbox in receivers:
        wire_count = random.choice([1, 2])
        any_two |= wire_count == 2
        sides = ["left", "right", "up", "down"]
        random.shuffle(sides)
        chosen_sides = sides[:wire_count]
        colors = random.sample(SOURCE_COLORS, k=wire_count)
        connections = []
        for side, color in zip(chosen_sides, colors):
            placed = _try_attach_wire(grid, used, bbox, side, color)
            if placed is None:
                return None
            connections.append((side, placed))
        if len(connections) == 2:
            r0, c0, r1, c1 = bbox
            points = []
            for side, (_, end_cell) in connections:
                if side == "left":
                    points.append((end_cell[0], c0))
                elif side == "right":
                    points.append((end_cell[0], c1))
                elif side == "up":
                    points.append((r0, end_cell[1]))
                else:
                    points.append((r1, end_cell[1]))
            if points[0] == points[1]:
                return None

    if not any_two and receivers:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
