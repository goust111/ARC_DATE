from __future__ import annotations

import random

import numpy as np


BG = 0
OBSTACLE = 5
ARROW_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS: dict[str, tuple[int, int]] = {
    "right": (0, 1),
    "left": (0, -1),
    "up": (-1, 0),
    "down": (1, 0),
}
GEN_ATTEMPTS = 400
GEN_MIN_SHIFT = 2


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors8(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)

    return comps


def _arrow_template(direction: str, length: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    assert length >= 2
    if direction == "right":
        patch = np.zeros((3, length + 2), dtype=bool)
        patch[1, :length] = True
        patch[0, length] = True
        patch[2, length] = True
        patch[1, length + 1] = True
        line = [(1, c) for c in range(length)]
        return patch, line
    if direction == "left":
        patch = np.zeros((3, length + 2), dtype=bool)
        patch[1, 2:] = True
        patch[0, 1] = True
        patch[2, 1] = True
        patch[1, 0] = True
        line = [(1, c) for c in range(2, length + 2)]
        return patch, line
    if direction == "up":
        patch = np.zeros((length + 2, 3), dtype=bool)
        patch[0, 1] = True
        patch[1, 0] = True
        patch[1, 2] = True
        patch[2:, 1] = True
        line = [(r, 1) for r in range(2, length + 2)]
        return patch, line

    assert direction == "down"
    patch = np.zeros((length + 2, 3), dtype=bool)
    patch[:length, 1] = True
    patch[length, 0] = True
    patch[length, 2] = True
    patch[length + 1, 1] = True
    line = [(r, 1) for r in range(length)]
    return patch, line


def _parse_arrow_component(coords: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = True

    candidates: list[tuple[str, list[tuple[int, int]]]] = []
    if h == 3 and w >= 4:
        length = int(w - 2)
        for direction in ("right", "left"):
            tmpl, line_rel = _arrow_template(direction, length)
            if patch.shape == tmpl.shape and bool(np.array_equal(patch, tmpl)):
                candidates.append((direction, line_rel))
    if w == 3 and h >= 4:
        length = int(h - 2)
        for direction in ("up", "down"):
            tmpl, line_rel = _arrow_template(direction, length)
            if patch.shape == tmpl.shape and bool(np.array_equal(patch, tmpl)):
                candidates.append((direction, line_rel))

    assert len(candidates) == 1, "each foreground component must be one readable arrow-line object"
    direction, line_rel = candidates[0]
    line_abs = [(int(r0 + rr), int(c0 + cc)) for rr, cc in line_rel]
    return direction, line_abs


def _shift_cells(cells: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _front_cell(cells: list[tuple[int, int]], direction: str) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return max(cells, key=lambda rc: rc[0] * dr + rc[1] * dc)


def _project_line(
    line_cells: list[tuple[int, int]],
    direction: str,
    obstacle_mask: np.ndarray,
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], int, str, set[tuple[int, int]]]:
    h, w = shape
    dr, dc = DIRS[direction]
    current = [(int(r), int(c)) for r, c in line_cells]
    corridor = set(current)
    steps = 0

    while True:
        nxt = _shift_cells(current, dr, dc)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in nxt):
            return current, int(steps), "boundary", corridor
        if any(bool(obstacle_mask[r, c]) for r, c in nxt):
            return current, int(steps), "obstacle", corridor
        current = nxt
        corridor.update(current)
        steps += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each colored tail segment along its arrow direction until it stops against a grey block or the grid edge.

    Input structure:
    - The background is black.
    - Grey rectangular obstacle blocks stay fixed in place.
    - Every other foreground object is a single-color arrow-line component: a 3-pixel arrow head points in one cardinal direction, and a straight same-color tail segment sits immediately behind that head on the same axis.

    Rule:
    - Detect each arrow head orientation from its 3-pixel shape.
    - Remove the arrow head and erase the tail from its original location.
    - Translate only the tail segment in the arrow direction until the next step would hit a grey obstacle or leave the grid.
    - Draw the translated tail segment in its stopping position and leave all grey obstacles unchanged.

    Key insight:
    - The 3-pixel chevron gives the direction, and only after decoding that direction can the attached segment be projected forward to its collision stop.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    obstacle_mask = g == OBSTACLE
    out = g.copy()
    out[(out != bg) & (out != OBSTACLE)] = bg

    colors = [int(color) for color in np.unique(g) if int(color) not in {int(bg), OBSTACLE}]
    for color in colors:
        for comp in _components_of_color(g, int(color)):
            direction, line_cells = _parse_arrow_component(comp)
            final_line, _steps, _stop, _corridor = _project_line(line_cells, direction, obstacle_mask, g.shape)
            for r, c in final_line:
                out[int(r), int(c)] = int(color)

    return out


def _pad_cells(cells: set[tuple[int, int]] | list[tuple[int, int]], shape: tuple[int, int], pad: int) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, int(r - pad)), min(h - 1, int(r + pad)) + 1):
            for cc in range(max(0, int(c - pad)), min(w - 1, int(c + pad)) + 1):
                out.add((int(rr), int(cc)))
    return out


def _mark(mask: np.ndarray, cells: set[tuple[int, int]] | list[tuple[int, int]], pad: int = 0) -> None:
    for r, c in _pad_cells(cells, mask.shape, pad):
        mask[int(r), int(c)] = True


def _build_arrow(base_r: int, base_c: int, direction: str, length: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if direction == "right":
        line = [(int(base_r), int(base_c - k)) for k in range(1, length + 1)]
        head = [(int(base_r - 1), int(base_c)), (int(base_r + 1), int(base_c)), (int(base_r), int(base_c + 1))]
        return line, head
    if direction == "left":
        line = [(int(base_r), int(base_c + k)) for k in range(1, length + 1)]
        head = [(int(base_r - 1), int(base_c)), (int(base_r + 1), int(base_c)), (int(base_r), int(base_c - 1))]
        return line, head
    if direction == "up":
        line = [(int(base_r + k), int(base_c)) for k in range(1, length + 1)]
        head = [(int(base_r), int(base_c - 1)), (int(base_r), int(base_c + 1)), (int(base_r - 1), int(base_c))]
        return line, head

    assert direction == "down"
    line = [(int(base_r - k), int(base_c)) for k in range(1, length + 1)]
    head = [(int(base_r), int(base_c - 1)), (int(base_r), int(base_c + 1)), (int(base_r + 1), int(base_c))]
    return line, head


def _place_obstacles(grid: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for _ in range(count):
        placed = False
        for _attempt in range(160):
            oh = random.randint(2, min(5, h - 2))
            ow = random.randint(2, min(5, w - 2))
            r0 = random.randint(1, h - oh - 1)
            c0 = random.randint(1, w - ow - 1)
            if bool(blocked[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1].any()):
                continue
            grid[r0 : r0 + oh, c0 : c0 + ow] = OBSTACLE
            blocked[r0 - 1 : r0 + oh + 1, c0 - 1 : c0 + ow + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _candidate_bases(shape: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]]:
    h, w = shape
    bases: list[tuple[int, int]] = []
    if direction == "right":
        for r in range(1, h - 1):
            for c in range(length, w - 1):
                bases.append((int(r), int(c)))
    elif direction == "left":
        for r in range(1, h - 1):
            for c in range(1, w - length):
                bases.append((int(r), int(c)))
    elif direction == "up":
        for r in range(1, h - length):
            for c in range(1, w - 1):
                bases.append((int(r), int(c)))
    else:
        assert direction == "down"
        for r in range(length, h - 1):
            for c in range(1, w - 1):
                bases.append((int(r), int(c)))
    random.shuffle(bases)
    return bases


def _enumerate_arrow_candidates(
    grid: np.ndarray,
    footprint_block: np.ndarray,
    path_block: np.ndarray,
    output_pad: np.ndarray,
    direction: str,
    length: int,
    desired_stop: str,
) -> list[dict]:
    obstacle_mask = grid == OBSTACLE
    candidates: list[dict] = []

    for base_r, base_c in _candidate_bases(grid.shape, direction, length):
        line, head = _build_arrow(base_r, base_c, direction, length)
        footprint = line + head
        if any(int(grid[r, c]) != BG for r, c in footprint):
            continue
        if any(bool(footprint_block[r, c]) for r, c in footprint):
            continue

        final_line, steps, stop_type, corridor = _project_line(line, direction, obstacle_mask, grid.shape)
        if int(steps) < GEN_MIN_SHIFT or stop_type != desired_stop:
            continue
        if any(bool(path_block[r, c]) for r, c in corridor):
            continue
        if any(bool(output_pad[r, c]) for r, c in final_line):
            continue

        candidates.append(
            {
                "direction": direction,
                "length": int(length),
                "line": line,
                "head": head,
                "final_line": final_line,
                "corridor": corridor,
                "steps": int(steps),
            }
        )

    return candidates


def _balanced_stop_types(count: int) -> list[str]:
    stop_types = ["boundary"] * (count // 2) + ["obstacle"] * (count // 2)
    if count % 2 == 1:
        stop_types.append(random.choice(["boundary", "obstacle"]))
    random.shuffle(stop_types)
    if count >= 2 and "boundary" not in stop_types:
        stop_types[0] = "boundary"
    if count >= 2 and "obstacle" not in stop_types:
        stop_types[-1] = "obstacle"
    random.shuffle(stop_types)
    return stop_types


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 24)
    w = random.randint(16, 24)
    grid = np.full((h, w), BG, dtype=int)

    obstacle_count = random.choice([1, 2, 2, 3])
    if not _place_obstacles(grid, obstacle_count):
        return None

    expected = np.where(grid == OBSTACLE, OBSTACLE, BG)
    footprint_block = np.zeros((h, w), dtype=bool)
    path_block = np.zeros((h, w), dtype=bool)
    obstacle_cells = list(zip(*np.where(grid == OBSTACLE)))
    _mark(footprint_block, obstacle_cells, pad=1)
    _mark(path_block, obstacle_cells, pad=0)
    output_pad = np.zeros((h, w), dtype=bool)

    arrow_count = random.choices([2, 3, 4], weights=[1, 2, 2])[0]
    colors = random.sample(ARROW_COLORS, arrow_count)
    directions = random.sample(list(DIRS.keys()), arrow_count)
    stop_types = _balanced_stop_types(arrow_count)

    for color, direction, desired_stop in zip(colors, directions, stop_types):
        # Length 1 is omitted because the 3-pixel head plus a one-pixel tail becomes direction-ambiguous.
        viable_by_length: dict[int, list[dict]] = {}
        lengths = [2, 3, 4, 5]
        random.shuffle(lengths)
        for length in lengths:
            cands = _enumerate_arrow_candidates(grid, footprint_block, path_block, output_pad, direction, length, desired_stop)
            if cands:
                viable_by_length[int(length)] = cands
        if not viable_by_length:
            return None

        length = random.choice(sorted(viable_by_length))
        cand = random.choice(viable_by_length[int(length)])
        for r, c in cand["line"] + cand["head"]:
            grid[int(r), int(c)] = int(color)
        for r, c in cand["final_line"]:
            expected[int(r), int(c)] = int(color)

        _mark(footprint_block, cand["line"] + cand["head"], pad=1)
        _mark(footprint_block, cand["final_line"], pad=1)
        _mark(path_block, cand["line"] + cand["head"], pad=0)
        _mark(path_block, cand["corridor"], pad=0)
        _mark(output_pad, cand["final_line"], pad=1)

    if not bool(np.array_equal(main(grid), expected)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _attempt in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid arrow-projection puzzle")
