import random
from collections import deque

import numpy as np


BG = 0
INSTR_H = 3
PALETTE = (1, 2, 3, 4)
DIRS = ("U", "R", "D", "L")
DIR_TO_AREA = {"U": 3, "R": 4, "D": 5, "L": 6}
AREA_TO_DIR = {v: k for k, v in DIR_TO_AREA.items()}
DIR_STEP = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
RECIPES = (("D", "R"), ("D", "L"), ("R", "D"), ("L", "D"))
MAX_ATTEMPTS = 400


GLYPH_POOL = {
    3: (
        np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int),
    ),
    4: (
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=int),
    ),
    5: (
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    ),
    6: (
        np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]], dtype=int),
        np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
    ),
}


OBJECT_POOL = (
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (1, 0), (2, 0), (3, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)},
    {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)},
)


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_dims(coords):
    r0, c0, r1, c1 = _bbox(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _normalize(coords):
    r0, c0, _, _ = _bbox(coords)
    return {(r - r0, c - c0) for r, c in coords}


def _translated(coords, dr, dc):
    return {(r + dr, c + dc) for r, c in coords}


def _paint(grid, coords, color):
    for r, c in coords:
        grid[r, c] = int(color)


def _components_from_mask(mask, connectivity):
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _parse_instructions(grid):
    top = np.asarray(grid[:INSTR_H], dtype=int)
    glyphs = []
    for color in sorted(int(x) for x in np.unique(top) if int(x) != BG):
        comps = _components_from_mask(top == color, connectivity=8)
        assert len(comps) == 1, f"glyph color {color} must form one component"
        comp = comps[0]
        area = len(comp)
        assert area in AREA_TO_DIR, f"unsupported glyph area {area}"
        _r0, c0, _r1, _c1 = _bbox(comp)
        glyphs.append({"color": int(color), "area": int(area), "dir": AREA_TO_DIR[area], "c0": int(c0)})
    glyphs.sort(key=lambda g: int(g["c0"]))
    assert 3 <= len(glyphs) <= 4, "expected 3 or 4 glyphs"
    assert len({g["color"] for g in glyphs}) == len(glyphs), "glyph colors must be distinct"
    assert len({g["area"] for g in glyphs}) == len(glyphs), "glyph areas must be distinct"
    return glyphs


def _parse_objects(grid, glyphs):
    g = np.asarray(grid, dtype=int)
    objects = {}
    for glyph in glyphs:
        color = int(glyph["color"])
        mask = (g == color).copy()
        mask[:INSTR_H, :] = False
        comps = _components_from_mask(mask, connectivity=4)
        assert len(comps) == 1, f"color {color} must match exactly one lower object"
        coords = {tuple(map(int, p)) for p in comps[0]}
        objects[color] = coords
    return objects


def _slide(coords, occupied_by_color, step, shape):
    h, w = shape
    dr, dc = step
    current = set(coords)
    distance = 0
    while True:
        nxt = {(r + dr, c + dc) for r, c in current}
        hit_boundary = False
        blockers = set()
        for nr, nc in nxt:
            if nr < INSTR_H or nr >= h or nc < 0 or nc >= w:
                hit_boundary = True
            elif (nr, nc) in occupied_by_color:
                blockers.add(int(occupied_by_color[(nr, nc)]))
        if hit_boundary or blockers:
            if hit_boundary and blockers:
                cause = "mixed"
            elif blockers:
                cause = "object"
            else:
                cause = "edge"
            return current, distance, cause, blockers
        current = nxt
        distance += 1


def _simulate(grid, order=None):
    g = np.asarray(grid, dtype=int)
    glyphs = _parse_instructions(g)
    objects = _parse_objects(g, glyphs)
    run_order = glyphs if order is None else order

    moved_colors = set()
    records = []
    for instr in run_order:
        color = int(instr["color"])
        occupied = {}
        for other_color, coords in objects.items():
            if other_color == color:
                continue
            for pos in coords:
                occupied[pos] = int(other_color)
        final_coords, distance, cause, blockers = _slide(objects[color], occupied, DIR_STEP[instr["dir"]], g.shape)
        objects[color] = final_coords
        prev_blockers = set(blockers) & moved_colors
        future_blockers = set(blockers) - moved_colors
        records.append(
            {
                "color": color,
                "dir": instr["dir"],
                "distance": int(distance),
                "cause": cause,
                "blockers": set(blockers),
                "prev_blockers": prev_blockers,
                "future_blockers": future_blockers,
            }
        )
        moved_colors.add(color)

    out = np.zeros_like(g)
    for color, coords in objects.items():
        _paint(out, coords, color)
    return out, glyphs, objects, records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode top glyph instructions, match them to same-color objects, then slide the objects in order while earlier moves become later obstacles.

    Input structure:
    - The top 3 rows form an instruction strip on a black background with 3 or 4 separated 3x3 monochrome glyphs.
    - Each glyph uses one of the colors blue, red, green, or yellow, and each glyph area is one of 3, 4, 5, or 6 without repetition.
    - Below the instruction strip, there is exactly one 4-connected object in the matching color for each glyph.

    Rule:
    - Read glyphs from left to right and decode direction from glyph area: 3->up, 4->right, 5->down, 6->left.
    - Match each glyph to the unique lower object of the same color.
    - Process the objects in that glyph order, sliding each rigid object until the next step would hit the grid edge, the cleared instruction-zone boundary above row 3, or another object; each move updates the obstacle field immediately.
    - After all moves, clear the entire top 3-row instruction strip to black.

    Key insight:
    - Glyph pixel count gives direction, glyph color selects the target object, and execution order matters because moved objects become dynamic blockers for later ones.
    """
    out, _glyphs, _objects, _records = _simulate(grid)
    return out


def _random_extra_shape(direction):
    candidates = []
    for shape in OBJECT_POOL:
        h, w = _shape_dims(shape)
        if direction in ("L", "R"):
            if w >= h:
                candidates.append(shape)
        else:
            if h >= w:
                candidates.append(shape)
    pool = candidates if candidates and random.random() < 0.75 else list(OBJECT_POOL)
    return set(random.choice(pool))


def _can_place(coords, occupied, shape, gap=1):
    h, w = shape
    for r, c in coords:
        if r < INSTR_H or r >= h or c < 0 or c >= w:
            return False
        for dr in range(-gap, gap + 1):
            for dc in range(-gap, gap + 1):
                if (r + dr, c + dc) in occupied:
                    return False
    return True


def _one_step_open(coords, direction, occupied, shape):
    dr, dc = DIR_STEP[direction]
    h, w = shape
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if nr < INSTR_H or nr >= h or nc < 0 or nc >= w or (nr, nc) in occupied:
            return False
    return True


def _place_pair(recipe, h, w):
    a_dir, b_dir = recipe
    if recipe[0] == "D":
        len_a = random.randint(3, 5)
        len_b = random.randint(3, 5)
        if w < len_b + 5 or h < len_a + 6:
            return None
        if b_dir == "R":
            c_a = random.randint(len_b + 2, w - 2)
            r_a = random.randint(INSTR_H + 1, h - len_a - 3)
            a = {(r_a + i, c_a) for i in range(len_a)}
            r_b = random.randint(h - len_a, h - 1)
            max_c0 = c_a - len_b - 1
            if max_c0 < 0:
                return None
            c_b = random.randint(0, max_c0)
            b = {(r_b, c_b + i) for i in range(len_b)}
        else:
            c_a = random.randint(1, w - len_b - 3)
            r_a = random.randint(INSTR_H + 1, h - len_a - 3)
            a = {(r_a + i, c_a) for i in range(len_a)}
            r_b = random.randint(h - len_a, h - 1)
            min_c0 = c_a + 2
            max_c0 = w - len_b
            if min_c0 > max_c0:
                return None
            c_b = random.randint(min_c0, max_c0)
            b = {(r_b, c_b + i) for i in range(len_b)}
    elif recipe[0] == "R":
        len_a = random.randint(3, 5)
        len_b = random.randint(3, 5)
        if h < len_b + 6 or w < len_a + 6:
            return None
        r_a = random.randint(INSTR_H + len_b + 1, h - 2)
        c_a = random.randint(1, w - len_a - 3)
        a = {(r_a, c_a + i) for i in range(len_a)}
        c_b = random.randint(w - len_a, w - 1)
        max_r0 = r_a - len_b - 1
        if max_r0 < INSTR_H:
            return None
        r_b = random.randint(INSTR_H, max_r0)
        b = {(r_b + i, c_b) for i in range(len_b)}
    else:
        len_a = random.randint(3, 5)
        len_b = random.randint(3, 5)
        if h < len_b + 6 or w < len_a + 6:
            return None
        r_a = random.randint(INSTR_H + len_b + 1, h - 2)
        c_a = random.randint(2, w - len_a - 1)
        a = {(r_a, c_a + i) for i in range(len_a)}
        c_b = random.randint(0, len_a - 1)
        max_r0 = r_a - len_b - 1
        if max_r0 < INSTR_H:
            return None
        r_b = random.randint(INSTR_H, max_r0)
        b = {(r_b + i, c_b) for i in range(len_b)}

    if _bbox(a)[0] < INSTR_H or _bbox(b)[0] < INSTR_H:
        return None
    return {a_dir: a, b_dir: b}


def _place_extra(direction, h, w, occupied, pair_info):
    step = DIR_STEP[direction]
    for _ in range(150):
        shape = _random_extra_shape(direction)
        sh_h, sh_w = _shape_dims(shape)
        if direction == "U":
            r_lo = max(INSTR_H + 2, h // 2)
            r_hi = h - sh_h
            c_lo, c_hi = 0, w - sh_w
        elif direction == "D":
            r_lo = INSTR_H
            r_hi = max(INSTR_H, h // 3)
            c_lo, c_hi = 0, w - sh_w
        elif direction == "L":
            r_lo, r_hi = INSTR_H, h - sh_h
            c_lo = max(0, w // 2 - 1)
            c_hi = w - sh_w
        else:
            r_lo, r_hi = INSTR_H, h - sh_h
            c_lo = 0
            c_hi = max(0, w // 2)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        coords = _translated(shape, r0, c0)
        if not _can_place(coords, occupied, (h, w), gap=1):
            continue
        if not _one_step_open(coords, direction, occupied, (h, w)):
            continue
        pr = pair_info["primary_row"]
        pc = pair_info["primary_col"]
        rows = {r for r, _ in coords}
        cols = {c for _, c in coords}
        if direction in ("L", "R") and pc in cols and random.random() < 0.8:
            continue
        if direction in ("U", "D") and pr in rows and random.random() < 0.8:
            continue
        return coords
    return None


def _render_input(h, w, glyph_order, dir_to_color, objects):
    grid = np.zeros((h, w), dtype=int)
    left_margin = random.randint(0, w - (4 * len(glyph_order) - 1))
    for i, direction in enumerate(glyph_order):
        area = DIR_TO_AREA[direction]
        color = dir_to_color[direction]
        glyph = random.choice(GLYPH_POOL[area])
        c0 = left_margin + 4 * i
        grid[:INSTR_H, c0:c0 + 3] = np.where(glyph == 1, color, grid[:INSTR_H, c0:c0 + 3])
    for direction, coords in objects.items():
        _paint(grid, coords, dir_to_color[direction])
    return grid


def _semantic_ok(inp):
    try:
        out, glyphs, _objects, records = _simulate(inp)
    except Exception:
        return False

    if np.any(out[:INSTR_H] != BG):
        return False
    if any(rec["distance"] <= 0 for rec in records):
        return False
    if not any(
        rec["cause"] == "object"
        and rec["prev_blockers"]
        and not rec["future_blockers"]
        for rec in records
    ):
        return False

    reversed_order = list(reversed(glyphs))
    try:
        reverse_out, _glyphs2, _objects2, _records2 = _simulate(inp, order=reversed_order)
    except Exception:
        return False
    if np.array_equal(out, reverse_out):
        return False

    lower_in = inp[INSTR_H:]
    lower_out = out[INSTR_H:]
    if np.array_equal(lower_in, lower_out):
        return False

    colors = {int(x) for x in np.unique(inp)} - {BG}
    if not (3 <= len(colors) <= 4):
        return False
    return True


def _gen_attempt():
    n = random.choice((3, 4))
    recipe = random.choice(RECIPES)
    selected = list(recipe)
    remaining = [d for d in DIRS if d not in selected]
    selected.extend(random.sample(remaining, n - 2))

    glyph_order = selected[:]
    random.shuffle(glyph_order)
    while glyph_order.index(recipe[0]) > glyph_order.index(recipe[1]):
        random.shuffle(glyph_order)

    h = random.randint(14, 18)
    w = random.randint(max(15, 4 * n - 1), 18)
    pair = _place_pair(recipe, h, w)
    if pair is None:
        return None

    occupied = set()
    for coords in pair.values():
        occupied |= coords

    pair_cells = set().union(*pair.values())
    r0, c0, r1, c1 = _bbox(pair_cells)
    pair_info = {"primary_row": (r0 + r1) // 2, "primary_col": (c0 + c1) // 2}

    objects = dict(pair)
    for direction in selected:
        if direction in objects:
            continue
        coords = _place_extra(direction, h, w, occupied, pair_info)
        if coords is None:
            return None
        objects[direction] = coords
        occupied |= coords

    colors = random.sample(PALETTE, n)
    dir_to_color = {direction: color for direction, color in zip(selected, colors)}
    inp = _render_input(h, w, glyph_order, dir_to_color, objects)
    if not _semantic_ok(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
