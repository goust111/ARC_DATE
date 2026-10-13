import random

import numpy as np


BG = 0
MAX_ATTEMPTS = 300
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def main(grid: np.ndarray) -> np.ndarray:
    """Apply the left legend's ordered color substitutions to the right canvas."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[1] >= 4, "expected two legend columns, one separator, and a canvas"
    assert np.all(g[:, 2] == BG), "expected a black separator column"

    rules: list[tuple[int, int]] = []
    for src, dst in g[:, :2]:
        if src == BG and dst == BG:
            continue
        assert src != BG and dst != BG, "legend rows must be complete color pairs"
        rules.append((int(src), int(dst)))
    assert rules, "expected at least one replacement rule"

    out = g[:, 3:].copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            color = int(out[r, c])
            if color == BG:
                continue
            for src, dst in rules:
                if color == src:
                    color = dst
            out[r, c] = color
    return out


def _paint_shape(canvas: np.ndarray, occupied: np.ndarray, color: int, cells: list[tuple[int, int]]) -> bool:
    h, w = canvas.shape
    expanded = set()
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    if any(occupied[r, c] for r, c in expanded):
        return False

    for r, c in cells:
        canvas[r, c] = color
    for r, c in expanded:
        occupied[r, c] = True
    return True


def _rectangle_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r in range(height) for c in range(width)]


def _line_cells(top: int, left: int, length: int, horizontal: bool) -> list[tuple[int, int]]:
    if horizontal:
        return [(top, left + i) for i in range(length)]
    return [(top + i, left) for i in range(length)]


def _l_cells(top: int, left: int, height: int, width: int, arm_from_left: bool, arm_from_top: bool) -> list[tuple[int, int]]:
    cells = {(top + r, left if arm_from_left else left + width - 1) for r in range(height)}
    row = top if arm_from_top else top + height - 1
    cells |= {(row, left + c) for c in range(width)}
    return sorted(cells)


def _plus_cells(center_r: int, center_c: int, arm: int) -> list[tuple[int, int]]:
    cells = {(center_r, center_c)}
    for step in range(1, arm + 1):
        for dr, dc in NEIGHBORS4:
            cells.add((center_r + dr * step, center_c + dc * step))
    return sorted(cells)


def _random_shape(rng: random.Random, h: int, w: int) -> list[tuple[int, int]]:
    kind = rng.choice(["rect", "line", "l", "plus"])
    if kind == "rect":
        sh = rng.randint(1, min(3, h))
        sw = rng.randint(1, min(4, w))
        top = rng.randint(0, h - sh)
        left = rng.randint(0, w - sw)
        return _rectangle_cells(top, left, sh, sw)
    if kind == "line":
        horizontal = rng.choice([False, True])
        length = rng.randint(2, min(5, w if horizontal else h))
        top = rng.randint(0, h - (1 if horizontal else length))
        left = rng.randint(0, w - (length if horizontal else 1))
        return _line_cells(top, left, length, horizontal)
    if kind == "l":
        sh = rng.randint(2, min(4, h))
        sw = rng.randint(2, min(4, w))
        top = rng.randint(0, h - sh)
        left = rng.randint(0, w - sw)
        return _l_cells(top, left, sh, sw, rng.choice([False, True]), rng.choice([False, True]))

    arm = rng.randint(1, min(2, (min(h, w) - 1) // 2))
    center_r = rng.randint(arm, h - arm - 1)
    center_c = rng.randint(arm, w - arm - 1)
    return _plus_cells(center_r, center_c, arm)


def _apply_rules(canvas: np.ndarray, rules: list[tuple[int, int]]) -> np.ndarray:
    out = canvas.copy()
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            color = int(out[r, c])
            if color == BG:
                continue
            for src, dst in rules:
                if color == src:
                    color = dst
            out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    canvas_h = rng.randint(7, 13)
    canvas_w = rng.randint(7, 13)

    usable_colors = rng.sample(range(1, 10), rng.randint(5, 8))
    chain_len = rng.choice([2, 3])
    chain = usable_colors[: chain_len + 1]
    extra_palette = usable_colors[chain_len + 1 :]
    start_color = chain[0]
    mid_color = chain[1]
    final_color = chain[-1]

    rules = [(chain[i], chain[i + 1]) for i in range(chain_len)]
    if extra_palette:
        if rng.choice([False, True]):
            extra_src, extra_dst = rng.sample(extra_palette, 2) if len(extra_palette) >= 2 else (extra_palette[0], final_color)
            if extra_src != extra_dst:
                insert_at = rng.randint(0, len(rules))
                rules.insert(insert_at, (extra_src, extra_dst))
        else:
            absent_src = extra_palette[0]
            absent_dst = rng.choice([c for c in usable_colors if c != absent_src])
            insert_at = rng.randint(0, len(rules))
            rules.insert(insert_at, (absent_src, absent_dst))

    if rng.choice([False, True]):
        bridge_src = mid_color
        bridge_dst = rng.choice([c for c in usable_colors if c != bridge_src])
        insert_at = rng.randint(0, len(rules))
        rules.insert(insert_at, (bridge_src, bridge_dst))

    rule_count = len(rules)
    grid_h = max(canvas_h, rule_count + rng.randint(0, 2))

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    mandatory_colors = [start_color, mid_color]
    if chain_len == 3:
        mandatory_colors.append(chain[2])
    unaffected_choices = [c for c in usable_colors if c not in {src for src, _ in rules}]
    if unaffected_choices:
        mandatory_colors.append(rng.choice(unaffected_choices))

    shape_colors = mandatory_colors[:]
    extra_shape_colors = [c for c in usable_colors if c not in shape_colors]
    rng.shuffle(extra_shape_colors)
    shape_colors.extend(extra_shape_colors[: rng.randint(1, min(3, len(extra_shape_colors)))])
    rng.shuffle(shape_colors)

    for color in shape_colors:
        painted = False
        for _ in range(80):
            cells = _random_shape(rng, canvas_h, canvas_w)
            if _paint_shape(canvas, occupied, color, cells):
                painted = True
                break
        if not painted:
            return None

    source_colors = {src for src, _ in rules}
    if not np.any(canvas == start_color) or not np.any(canvas == mid_color):
        return None
    if chain_len == 3 and not np.any(canvas == chain[2]):
        return None
    if not any(np.any(canvas == color) for color in set(np.unique(canvas)) - {BG} - source_colors):
        return None

    transformed = _apply_rules(canvas, rules)
    if np.array_equal(canvas, transformed):
        return None
    if not np.any(transformed == final_color):
        return None
    if np.any(canvas == start_color) and np.any(transformed == start_color):
        return None

    grid = np.zeros((grid_h, canvas_w + 3), dtype=int)
    grid[:canvas_h, 3:] = canvas
    for i, (src, dst) in enumerate(rules):
        grid[i, 0] = src
        grid[i, 1] = dst

    # Validate the generated input against the solver's parsing assumptions.
    if any((grid[i, 0] == BG) ^ (grid[i, 1] == BG) for i in range(grid_h)):
        return None
    if not np.array_equal(main(grid), transformed):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
