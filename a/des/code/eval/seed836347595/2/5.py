import random

import numpy as np


_BG = 0
_CARDINAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_EDGE_TO_DIR = {
    (0, 1): (-1, 0),
    (1, 2): (0, 1),
    (2, 1): (1, 0),
    (1, 0): (0, -1),
}
_DIR_TO_EDGE = {direction: edge for edge, direction in _EDGE_TO_DIR.items()}
_LEGEND_TILES = ((0, 0), (0, 3), (3, 0), (3, 3))

_GRID_H_RANGE = (16, 24)
_GRID_W_RANGE = (18, 26)
_RULE_ROW_RANGE = (6, 8)
_RULE_LEN_RANGE = (4, 10)
_GEN_MAX_ATTEMPTS = 300


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in _CARDINAL_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), max(rows), min(cols), max(cols)


def _detect_legend(grid: np.ndarray) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    row_tries = max(1, min(3, h - 5))
    col_tries = max(1, min(3, w - 5))

    for r0 in range(row_tries):
        for c0 in range(col_tries):
            patch = grid[r0 : r0 + 6, c0 : c0 + 6]
            if patch.shape != (6, 6) or int(np.count_nonzero(patch)) != 4:
                continue

            mapping: dict[int, tuple[int, int]] = {}
            seen_dirs: set[tuple[int, int]] = set()
            ok = True
            for tr, tc in _LEGEND_TILES:
                tile = patch[tr : tr + 3, tc : tc + 3]
                if int(tile[1, 1]) != _BG:
                    ok = False
                    break
                nz = [(rr, cc) for rr in range(3) for cc in range(3) if int(tile[rr, cc]) != _BG]
                if len(nz) != 1 or nz[0] not in _EDGE_TO_DIR:
                    ok = False
                    break
                edge = nz[0]
                color = int(tile[edge])
                direction = _EDGE_TO_DIR[edge]
                if color in mapping or direction in seen_dirs:
                    ok = False
                    break
                mapping[color] = direction
                seen_dirs.add(direction)

            if ok and len(mapping) == 4:
                legend_mask = np.zeros_like(grid, dtype=bool)
                legend_mask[r0 : r0 + 6, c0 : c0 + 6] = True
                return mapping, legend_mask

    raise AssertionError("legend not found")


def _detect_rule_bar(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    mapping: dict[int, tuple[int, int]],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    mask = (grid != _BG) & ~legend_mask
    candidates: list[tuple[int, int, list[int]]] = []
    for comp in _components(mask):
        r0, r1, c0, c1 = _bbox(comp)
        if r0 != r1 or len(comp) < 4 or len(comp) != c1 - c0 + 1:
            continue
        colors = [int(grid[r0, c]) for c in range(c0, c1 + 1)]
        if any(color not in mapping for color in colors):
            continue
        candidates.append((r0, c0, colors))

    if not candidates:
        raise AssertionError("rule bar not found")

    row, c0, colors = min(candidates, key=lambda item: (item[0], -len(item[2]), item[1]))
    rule_mask = np.zeros_like(grid, dtype=bool)
    rule_mask[row, c0 : c0 + len(colors)] = True
    return [mapping[color] for color in colors], rule_mask


def _extract_seed(grid: np.ndarray, reserved: np.ndarray) -> list[tuple[int, int]]:
    comps = _components((grid != _BG) & ~reserved)
    if not comps:
        raise AssertionError("seed not found")
    return max(comps, key=lambda comp: (len(comp), sum(r for r, _ in comp)))


def _grow_pattern(
    cells_to_color: dict[tuple[int, int], int],
    directions: list[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[dict[tuple[int, int], int], bool]:
    h, w = shape
    current = dict(cells_to_color)
    clipped = False
    for dr, dc in directions:
        grown = dict(current)
        for (r, c), color in current.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                grown.setdefault((nr, nc), color)
            else:
                clipped = True
        current = grown
    return current, clipped


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _detect_legend(g)
    directions, rule_mask = _detect_rule_bar(g, legend_mask, mapping)
    reserved = legend_mask | rule_mask
    seed = _extract_seed(g, reserved)

    start = {(r, c): int(g[r, c]) for r, c in seed}
    grown, _ = _grow_pattern(start, directions, g.shape)

    out = g.copy()
    for (r, c), color in grown.items():
        if not reserved[r, c]:
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _normalize(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_seed_shape() -> tuple[tuple[int, int], ...]:
    target = random.randint(2, 4)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in _CARDINAL_DIRS:
                frontier.add((r + dr, c + dc))
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))
    return _normalize(cells)


def _random_sequence(clip_mode: bool) -> tuple[list[tuple[int, int]], str | None]:
    length = random.randint(*_RULE_LEN_RANGE)
    if clip_mode:
        target = random.choice([(0, 1), (1, 0)])
        heavy = random.randint((length + 1) // 2, length - 1)
        seq = [target] * heavy + [random.choice(_CARDINAL_DIRS) for _ in range(length - heavy)]
        random.shuffle(seq)
        if len(set(seq)) == 1:
            seq[random.randrange(length)] = (-1, 0) if target != (-1, 0) else (0, -1)
        return seq, "right" if target == (0, 1) else "bottom"

    if random.random() < 0.5:
        seq = random.sample(_CARDINAL_DIRS, 4)
        seq.extend(random.choice(_CARDINAL_DIRS) for _ in range(length - 4))
        random.shuffle(seq)
    else:
        seq = [random.choice(_CARDINAL_DIRS) for _ in range(length)]
        if len(set(seq)) == 1:
            seq[random.randrange(length)] = random.choice([d for d in _CARDINAL_DIRS if d != seq[0]])
    return seq, None


def _grow_relative(shape: tuple[tuple[int, int], ...], directions: list[tuple[int, int]]) -> set[tuple[int, int]]:
    current = set(shape)
    for dr, dc in directions:
        current |= {(r + dr, c + dc) for r, c in current}
    return current


def _place_legend(grid: np.ndarray, dir_to_color: dict[tuple[int, int], int]) -> np.ndarray:
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[:6, :6] = True
    tile_dirs = random.sample(_CARDINAL_DIRS, len(_CARDINAL_DIRS))
    for (tr, tc), direction in zip(_LEGEND_TILES, tile_dirs):
        er, ec = _DIR_TO_EDGE[direction]
        grid[tr + er, tc + ec] = dir_to_color[direction]
    return legend_mask


def _choose_origin(
    h: int,
    w: int,
    seed_shape: tuple[tuple[int, int], ...],
    final_shape: set[tuple[int, int]],
    reserved: np.ndarray,
    rule_row: int,
    clip_mode: bool,
    clip_side: str | None,
) -> tuple[int, int] | None:
    candidates: list[tuple[float, int, int]] = []
    target_row = 0.68 * (h - 1)
    target_col = 0.50 * (w - 1)

    for r0 in range(h):
        for c0 in range(w):
            seed_cells = [(r0 + r, c0 + c) for r, c in seed_shape]
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in seed_cells):
                continue
            if any(reserved[rr, cc] for rr, cc in seed_cells):
                continue

            any_out = False
            visible_cells: list[tuple[int, int]] = []
            blocked = False
            for r, c in final_shape:
                rr, cc = r0 + r, c0 + c
                if 0 <= rr < h and 0 <= cc < w:
                    if reserved[rr, cc] or rr <= rule_row + 1:
                        blocked = True
                        break
                    visible_cells.append((rr, cc))
                else:
                    any_out = True
            if blocked or not visible_cells or any_out != clip_mode:
                continue

            if clip_side == "right" and not any(c0 + c >= w for _, c in final_shape):
                continue
            if clip_side == "bottom" and not any(r0 + r >= h for r, _ in final_shape):
                continue

            seed_center_r = sum(rr for rr, _ in seed_cells) / len(seed_cells)
            seed_center_c = sum(cc for _, cc in seed_cells) / len(seed_cells)
            if seed_center_r < 0.45 * (h - 1):
                continue

            score = abs(seed_center_r - target_row) + abs(seed_center_c - target_col)
            if clip_side == "right":
                score += 0.25 * abs(seed_center_c - 0.68 * (w - 1))
            candidates.append((score + random.random() * 0.15, r0, c0))

    if not candidates:
        return None

    candidates.sort()
    top = candidates[: min(12, len(candidates))]
    _, r0, c0 = random.choice(top)
    return r0, c0


def _expected_output(
    grid: np.ndarray,
    final_shape: set[tuple[int, int]],
    origin: tuple[int, int],
    seed_color: int,
    reserved: np.ndarray,
) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    r0, c0 = origin
    for r, c in final_shape:
        rr, cc = r0 + r, c0 + c
        if 0 <= rr < h and 0 <= cc < w and not reserved[rr, cc]:
            out[rr, cc] = seed_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    clip_mode = random.random() < 0.5
    directions, clip_side = _random_sequence(clip_mode)
    if w - len(directions) - 1 < 7:
        return None

    colors = random.sample(range(1, 10), 5)
    random.shuffle(colors)
    dir_to_color = {direction: color for direction, color in zip(random.sample(_CARDINAL_DIRS, 4), colors[:4])}
    seed_color = colors[4]

    grid = np.zeros((h, w), dtype=int)
    reserved = _place_legend(grid, dir_to_color)

    max_rule_row = min(_RULE_ROW_RANGE[1], h - 8)
    if max_rule_row < _RULE_ROW_RANGE[0]:
        return None
    rule_row = random.randint(_RULE_ROW_RANGE[0], max_rule_row)
    rule_start = random.randint(7, w - len(directions) - 1)
    for idx, direction in enumerate(directions):
        grid[rule_row, rule_start + idx] = dir_to_color[direction]
    reserved = reserved.copy()
    reserved[rule_row, rule_start : rule_start + len(directions)] = True

    seed_shape = _random_seed_shape()
    final_shape = _grow_relative(seed_shape, directions)
    origin = _choose_origin(h, w, seed_shape, final_shape, reserved, rule_row, clip_mode, clip_side)
    if origin is None:
        return None

    r0, c0 = origin
    for r, c in seed_shape:
        grid[r0 + r, c0 + c] = seed_color

    expected = _expected_output(grid, final_shape, origin, seed_color, reserved)
    if int(np.count_nonzero(expected != grid)) <= len(seed_shape):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
