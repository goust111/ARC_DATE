import random

import numpy as np


AZURE = 8
RED = 2
MAROON = 9
DRAW_COLORS = [1, 3, 4, 5, 6, 7]
MAX_GENERATION_ATTEMPTS = 300


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the top template against the bottom defective copy and mark mismatches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r] == MAROON)]
    assert len(sep_rows) == 1, "expected exactly one full-width maroon separator row"

    sep = sep_rows[0]
    assert sep * 2 + 1 == h, "top and bottom halves must have equal height"

    top = g[:sep]
    bottom = g[sep + 1 :]
    assert top.shape == bottom.shape and top.shape[1] == w

    out = np.full(top.shape, AZURE, dtype=int)
    out[top != bottom] = RED
    return out


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _add_object(canvas: np.ndarray, occupied: np.ndarray, cells, color: int) -> bool:
    h, w = canvas.shape
    coords = list(dict.fromkeys((int(r), int(c)) for r, c in cells))
    if not coords:
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in coords):
        return False
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    for r, c in coords:
        canvas[r, c] = color
        occupied[r, c] = True
    return True


def _rect_cells(r0: int, c0: int, hh: int, ww: int):
    return [(r, c) for r in range(r0, r0 + hh) for c in range(c0, c0 + ww)]


def _frame_cells(r0: int, c0: int, hh: int, ww: int):
    cells = []
    for r in range(r0, r0 + hh):
        for c in range(c0, c0 + ww):
            if r in (r0, r0 + hh - 1) or c in (c0, c0 + ww - 1):
                cells.append((r, c))
    return cells


def _line_cells(r0: int, c0: int, length: int, horizontal: bool):
    if horizontal:
        return [(r0, c) for c in range(c0, c0 + length)]
    return [(r, c0) for r in range(r0, r0 + length)]


def _l_cells(r0: int, c0: int, hh: int, ww: int):
    cells = [(r, c0) for r in range(r0, r0 + hh)]
    cells.extend((r0 + hh - 1, c) for c in range(c0, c0 + ww))
    return cells


def _plus_cells(cr: int, cc: int, arm: int):
    cells = [(cr, cc)]
    for d in range(1, arm + 1):
        cells.extend([(cr - d, cc), (cr + d, cc), (cr, cc - d), (cr, cc + d)])
    return cells


def _zigzag_cells(r0: int, c0: int, length: int, height: int):
    cells = []
    for i in range(length):
        r = r0 + (i % height)
        c = c0 + i
        cells.append((r, c))
        if i + 1 < length:
            rn = r0 + ((i + 1) % height)
            step = 1 if rn > r else -1
            for rr in range(r, rn, step):
                cells.append((rr, c))
    return cells


def _diamond_cells(cr: int, cc: int, radius: int):
    cells = []
    for dr in range(-radius, radius + 1):
        span = radius - abs(dr)
        for dc in range(-span, span + 1):
            cells.append((cr + dr, cc + dc))
    return cells


def _patch_cells(h: int, w: int, rng: random.Random):
    seed_r = rng.randrange(h)
    seed_c = rng.randrange(w)
    target = rng.randint(4, min(9, h * w))
    cells = {(seed_r, seed_c)}
    frontier = [(seed_r, seed_c)]
    while frontier and len(cells) < target:
        r, c = frontier[rng.randrange(len(frontier))]
        options = [(rr, cc) for rr, cc in _neighbors4(r, c, h, w) if (rr, cc) not in cells]
        rng.shuffle(options)
        if not options:
            frontier.remove((r, c))
            continue
        rr, cc = options[0]
        cells.add((rr, cc))
        frontier.append((rr, cc))
    return list(cells)


def _random_shape_cells(h: int, w: int, rng: random.Random):
    kinds = ["rect", "patch"]
    if h >= 3 and w >= 3:
        kinds.extend(["frame", "l", "plus", "diamond"])
    if w >= 3:
        kinds.append("line_h")
    if h >= 3:
        kinds.append("line_v")
    if h >= 2 and w >= 4:
        kinds.append("zigzag")
    kind = rng.choice(kinds)
    if kind == "rect":
        hh = rng.randint(2, min(4, h))
        ww = rng.randint(2, min(4, w))
        return _rect_cells(rng.randint(0, h - hh), rng.randint(0, w - ww), hh, ww)
    if kind == "frame":
        hh = rng.randint(3, min(5, h))
        ww = rng.randint(3, min(5, w))
        return _frame_cells(rng.randint(0, h - hh), rng.randint(0, w - ww), hh, ww)
    if kind == "line_h":
        length = rng.randint(3, min(6, w))
        return _line_cells(rng.randint(0, h - 1), rng.randint(0, w - length), length, True)
    if kind == "line_v":
        length = rng.randint(3, min(6, h))
        return _line_cells(rng.randint(0, h - length), rng.randint(0, w - 1), length, False)
    if kind == "l":
        hh = rng.randint(3, min(5, h))
        ww = rng.randint(3, min(5, w))
        return _l_cells(rng.randint(0, h - hh), rng.randint(0, w - ww), hh, ww)
    if kind == "plus":
        arm = rng.randint(1, min((h - 1) // 2, (w - 1) // 2, 2))
        cr = rng.randint(arm, h - arm - 1)
        cc = rng.randint(arm, w - arm - 1)
        return _plus_cells(cr, cc, arm)
    if kind == "zigzag":
        height = rng.randint(2, min(3, h))
        length = rng.randint(4, min(7, w))
        return _zigzag_cells(rng.randint(0, h - height), rng.randint(0, w - length), length, height)
    if kind == "diamond":
        radius = rng.randint(1, min((h - 1) // 2, (w - 1) // 2, 2))
        cr = rng.randint(radius, h - radius - 1)
        cc = rng.randint(radius, w - radius - 1)
        return _diamond_cells(cr, cc, radius)
    return _patch_cells(h, w, rng)


def _build_template(half_h: int, w: int, rng: random.Random) -> np.ndarray | None:
    canvas = np.full((half_h, w), AZURE, dtype=int)
    occupied = np.zeros((half_h, w), dtype=bool)
    object_target = rng.randint(2, min(6, max(2, (half_h * w) // 18)))
    colors = rng.sample(DRAW_COLORS, k=min(len(DRAW_COLORS), max(2, object_target)))

    placed = 0
    attempts = 0
    while placed < object_target and attempts < 80:
        attempts += 1
        local_h = rng.randint(2, min(half_h, 6))
        local_w = rng.randint(2, min(w, 6))
        cells = _random_shape_cells(local_h, local_w, rng)
        offset_r = rng.randint(0, half_h - local_h)
        offset_c = rng.randint(0, w - local_w)
        shifted = [(offset_r + r, offset_c + c) for r, c in cells]
        color = colors[placed % len(colors)]
        if _add_object(canvas, occupied, shifted, color):
            placed += 1

    if placed < 2:
        return None
    fg = int(np.count_nonzero(canvas != AZURE))
    if fg < max(8, (half_h * w) // 10):
        return None
    if fg > (half_h * w * 3) // 5:
        return None
    return canvas


def _mutate_copy(template: np.ndarray, rng: random.Random) -> tuple[np.ndarray, int]:
    copy = template.copy()
    h, w = copy.shape
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    fg_cells = [(r, c) for r, c in all_cells if copy[r, c] != AZURE]
    bg_cells = [(r, c) for r, c in all_cells if copy[r, c] == AZURE]
    if len(fg_cells) < 4 or len(bg_cells) < 4:
        raise ValueError("template too degenerate for mutation")

    defect_count = rng.randint(1, min(5, h * w))
    used = set()

    for _ in range(defect_count):
        remaining_fg = [pos for pos in fg_cells if pos not in used]
        remaining_bg = [pos for pos in bg_cells if pos not in used]
        actions = []
        if remaining_fg:
            actions.extend(["erase", "recolor"])
        if remaining_bg:
            actions.append("add")
        action = rng.choice(actions)
        if action == "erase":
            r, c = rng.choice(remaining_fg)
            copy[r, c] = AZURE
            used.add((r, c))
        elif action == "recolor":
            r, c = rng.choice(remaining_fg)
            old = int(copy[r, c])
            new_color = rng.choice([c2 for c2 in DRAW_COLORS if c2 != old])
            copy[r, c] = new_color
            used.add((r, c))
        else:
            r, c = rng.choice(remaining_bg)
            neighbor_colors = [int(template[rr, cc]) for rr, cc in _neighbors4(r, c, h, w) if template[rr, cc] != AZURE]
            if neighbor_colors:
                new_color = rng.choice(neighbor_colors)
            else:
                new_color = rng.choice(DRAW_COLORS)
            copy[r, c] = new_color
            used.add((r, c))

    mismatches = int(np.count_nonzero(copy != template))
    return copy, mismatches


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    half_h = rng.randint(6, 15)
    w = rng.randint(7, 16)
    template = _build_template(half_h, w, rng)
    if template is None:
        return None

    defective, mismatches = _mutate_copy(template, rng)
    if not (1 <= mismatches <= 5):
        return None

    grid = np.vstack(
        [
            template,
            np.full((1, w), MAROON, dtype=int),
            defective,
        ]
    )
    out = main(grid)
    if int(np.count_nonzero(out == RED)) != mismatches:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
