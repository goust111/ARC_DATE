from __future__ import annotations

import numpy as np


RED = 2
ALL_COLORS = tuple(range(10))
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))
MIN_GRID = 12
MAX_GRID = 20
MIN_LENGTH = 2
MAX_LENGTH = 5
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ordered_line(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = sorted(coords, key=lambda rc: rc[1])
        line = [c for _, c in ordered]
        if line != list(range(line[0], line[0] + len(line))):
            return None
        return ordered, (0, 1)
    if len(cols) == 1:
        ordered = sorted(coords, key=lambda rc: rc[0])
        line = [r for r, _ in ordered]
        if line != list(range(line[0], line[0] + len(line))):
            return None
        return ordered, (1, 0)
    return None


def _extract_scene(grid: np.ndarray, bg: int) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    pointers: list[dict[str, object]] = []
    singletons: dict[tuple[int, int], int] = {}
    for comp in _components(grid != bg):
        if len(comp) == 1:
            pos = comp[0]
            singletons[pos] = int(grid[pos])
            continue

        info = _ordered_line(comp)
        if info is None:
            continue
        ordered, axis_step = info
        values = [int(grid[r, c]) for r, c in ordered]
        arrow_indices = [i for i, value in enumerate(values) if value == RED]
        if len(arrow_indices) != 1:
            continue
        arrow_idx = arrow_indices[0]
        if arrow_idx not in {0, len(ordered) - 1}:
            continue
        tail_colors = {value for i, value in enumerate(values) if i != arrow_idx}
        if len(tail_colors) != 1 or RED in tail_colors:
            continue

        dr0, dc0 = axis_step
        if arrow_idx == len(ordered) - 1:
            step = (dr0, dc0)
        else:
            step = (-dr0, -dc0)

        pointers.append(
            {
                "tail": [coord for i, coord in enumerate(ordered) if i != arrow_idx],
                "arrow": ordered[arrow_idx],
                "step": step,
                "length": len(ordered),
            }
        )
    return pointers, singletons


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    pointers, singletons = _extract_scene(g, bg)

    out = g.copy()
    for pointer in pointers:
        ar, ac = pointer["arrow"]
        dr, dc = pointer["step"]
        length = int(pointer["length"])
        target = (int(ar) + int(dr) * length, int(ac) + int(dc) * length)
        assert _in_bounds(g.shape, target[0], target[1]), "pointer target must stay in bounds"
        assert target in singletons, "pointer must land on a singleton target"
        fill_color = int(singletons[target])
        for r, c in pointer["tail"]:
            out[int(r), int(c)] = fill_color
    return out


def _pointer_geometry(
    red_pos: tuple[int, int], step: tuple[int, int], length: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    rr, cc = red_pos
    dr, dc = step
    segment = [(rr - dr * k, cc - dc * k) for k in range(length - 1, -1, -1)]
    tail = segment[:-1]
    target = (rr + dr * length, cc + dc * length)
    gap = [(rr + dr * k, cc + dc * k) for k in range(1, length)]
    corridor = segment + gap + [target]
    return segment, tail, target, corridor


def _expanded_cells(
    cells: list[tuple[int, int]], shape: tuple[int, int], radius: int = 1
) -> list[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if _in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return list(out)


def _red_bounds(size: int, step: int, length: int) -> tuple[int, int] | None:
    if step == 1:
        lo, hi = length - 1, size - 1 - length
    elif step == -1:
        lo, hi = length, size - length
    else:
        lo, hi = 0, size - 1
    if hi < lo:
        return None
    return lo, hi


def _place_pointer(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    length: int,
    step: tuple[int, int],
    tail_color: int,
    target_color: int,
) -> dict[str, object] | None:
    h, w = grid.shape
    dr, dc = step
    row_bounds = _red_bounds(h, dr, length)
    col_bounds = _red_bounds(w, dc, length)
    if row_bounds is None or col_bounds is None:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        rr = int(rng.integers(row_bounds[0], row_bounds[1] + 1))
        cc = int(rng.integers(col_bounds[0], col_bounds[1] + 1))
        segment, tail, target, corridor = _pointer_geometry((rr, cc), step, length)
        if not all(_in_bounds(grid.shape, r, c) for r, c in corridor):
            continue
        expanded = _expanded_cells(corridor, grid.shape, radius=1)
        if any(blocked[r, c] for r, c in expanded):
            continue

        for r, c in tail:
            grid[r, c] = tail_color
        grid[rr, cc] = RED
        grid[target] = target_color
        for r, c in expanded:
            blocked[r, c] = True
        return {"tail": tail, "target_color": int(target_color)}
    return None


def _place_singleton(
    rng: np.random.Generator, grid: np.ndarray, blocked: np.ndarray, color: int
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        r = int(rng.integers(0, h))
        c = int(rng.integers(0, w))
        expanded = _expanded_cells([(r, c)], grid.shape, radius=1)
        if any(blocked[rr, cc] for rr, cc in expanded):
            continue
        grid[r, c] = color
        for rr, cc in expanded:
            blocked[rr, cc] = True
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    w = int(rng.integers(MIN_GRID, MAX_GRID + 1))
    bg_choices = [color for color in ALL_COLORS if color != RED]
    bg = int(rng.choice(bg_choices))
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    area = h * w
    max_pointers = 2 + int(area >= 220) + int(area >= 300)
    n_pointers = int(rng.integers(1, max_pointers + 1))

    palette = [color for color in ALL_COLORS if color not in {bg, RED}]
    rng.shuffle(palette)
    tail_colors = palette[:n_pointers]
    target_pool = palette[n_pointers:]
    if not target_pool:
        return None

    dir_pool = list(DIRECTIONS)
    while len(dir_pool) < n_pointers:
        dir_pool.extend(DIRECTIONS)
    rng.shuffle(dir_pool)

    specs: list[dict[str, object]] = []
    for i in range(n_pointers):
        step = tuple(int(v) for v in dir_pool[i])
        axis_size = w if step[0] == 0 else h
        max_len = min(MAX_LENGTH, (axis_size - 1) // 2)
        if max_len < MIN_LENGTH:
            return None
        length = int(rng.integers(MIN_LENGTH, max_len + 1))
        target_color = int(rng.choice(target_pool))
        specs.append(
            {
                "length": length,
                "step": step,
                "tail_color": int(tail_colors[i]),
                "target_color": target_color,
            }
        )
    specs.sort(key=lambda spec: int(spec["length"]), reverse=True)

    placements: list[dict[str, object]] = []
    for spec in specs:
        placed = _place_pointer(
            rng,
            grid,
            blocked,
            int(spec["length"]),
            spec["step"],
            int(spec["tail_color"]),
            int(spec["target_color"]),
        )
        if placed is None:
            return None
        placements.append(placed)

    extra_singletons = int(rng.integers(0, 3))
    extra_colors = [color for color in ALL_COLORS if color not in {bg, RED}]
    for _ in range(extra_singletons):
        color = int(rng.choice(extra_colors))
        _place_singleton(rng, grid, blocked, color)

    expected = grid.copy()
    for placed in placements:
        fill_color = int(placed["target_color"])
        for r, c in placed["tail"]:
            expected[int(r), int(c)] = fill_color

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
