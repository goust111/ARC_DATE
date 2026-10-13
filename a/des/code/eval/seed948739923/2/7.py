import random
import numpy as np


BG = 8
GREY = 5
RULER = 9
POINTER_COLORS = [0, 1, 2, 3, 4, 6, 7]

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ROOT_OFFSETS = {
    (-1, 0): "up",
    (1, 0): "down",
    (0, -1): "left",
    (0, 1): "right",
}

SIDE_OPTIONS = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}

SIDE_SIGN = {
    "left": -1,
    "up": -1,
    "right": 1,
    "down": 1,
}

GEN_RETRIES = 600
MIN_FILL_STEPS = 2


def _root_pos(spec: dict) -> tuple[int, int]:
    dr, dc = DIR_VECS[spec["direction"]]
    return spec["center_r"] + dr, spec["center_c"] + dc


def _tip_pos(spec: dict) -> tuple[int, int]:
    rr, cc = _root_pos(spec)
    dr, dc = DIR_VECS[spec["direction"]]
    k = spec["pointer_len"]
    return rr + dr * k, cc + dc * k


def _width_bounds(spec: dict) -> tuple[int, int]:
    width = spec["ruler_len"]
    bias = SIDE_SIGN[spec["ruler_side"]]
    center = spec["center_c"] if spec["direction"] in ("up", "down") else spec["center_r"]
    half = width // 2
    if width % 2 == 1:
        return center - half, center + half
    if bias < 0:
        return center - half, center + half - 1
    return center - half + 1, center + half


def _object_cells(spec: dict) -> set[tuple[int, int]]:
    r, c = spec["center_r"], spec["center_c"]
    cells = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}

    root_r, root_c = _root_pos(spec)
    dr, dc = DIR_VECS[spec["direction"]]
    for step in range(1, spec["pointer_len"] + 1):
        cells.add((root_r + dr * step, root_c + dc * step))

    side = spec["ruler_side"]
    sign = SIDE_SIGN[side]
    if side in ("left", "right"):
        for offset in range(spec["ruler_len"]):
            cells.add((r, c + sign * (offset + 2)))
    else:
        for offset in range(spec["ruler_len"]):
            cells.add((r + sign * (offset + 2), c))
    return cells


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _spec_fits(spec: dict, h: int, w: int) -> bool:
    cells = _object_cells(spec)
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
        return False
    lo, hi = _width_bounds(spec)
    if spec["direction"] in ("up", "down"):
        return 0 <= lo <= hi < w
    return 0 <= lo <= hi < h


def _fill_sections_from_grid(grid: np.ndarray, spec: dict) -> tuple[list[tuple[int, int]], str, int]:
    h, w = grid.shape
    tip_r, tip_c = _tip_pos(spec)
    dr, dc = DIR_VECS[spec["direction"]]
    lo, hi = _width_bounds(spec)

    fill: list[tuple[int, int]] = []
    step = 1
    while True:
        rr = tip_r + dr * step
        cc = tip_c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            return fill, "boundary", step - 1
        if spec["direction"] in ("up", "down"):
            section = [(rr, x) for x in range(lo, hi + 1)]
        else:
            section = [(x, cc) for x in range(lo, hi + 1)]
        if any(grid[r, c] != BG for r, c in section):
            return fill, "obstacle", step - 1
        fill.extend(section)
        step += 1


def _find_dials(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dials: list[dict] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            if int(np.count_nonzero(patch == GREY)) != 8:
                continue
            non_grey = [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2) if int(g[rr, cc]) != GREY]
            if len(non_grey) != 1:
                continue
            root_r, root_c = non_grey[0]
            offset = (root_r - r, root_c - c)
            if offset not in ROOT_OFFSETS:
                continue
            color = int(g[root_r, root_c])
            if color in (BG, GREY, RULER):
                continue

            direction = ROOT_OFFSETS[offset]
            dr, dc = DIR_VECS[direction]
            tip_r, tip_c = root_r, root_c
            while 0 <= tip_r + dr < h and 0 <= tip_c + dc < w and int(g[tip_r + dr, tip_c + dc]) == color:
                tip_r += dr
                tip_c += dc

            candidates: list[tuple[str, int]] = []
            if direction in ("up", "down"):
                for side in ("left", "right"):
                    sign = SIDE_SIGN[side]
                    length = 0
                    cc = c + sign * 2
                    while 0 <= cc < w and int(g[r, cc]) == RULER:
                        length += 1
                        cc += sign
                    if length > 0:
                        candidates.append((side, length))
            else:
                for side in ("up", "down"):
                    sign = SIDE_SIGN[side]
                    length = 0
                    rr = r + sign * 2
                    while 0 <= rr < h and int(g[rr, c]) == RULER:
                        length += 1
                        rr += sign
                    if length > 0:
                        candidates.append((side, length))

            assert len(candidates) == 1
            ruler_side, ruler_len = candidates[0]
            dials.append(
                {
                    "center_r": r,
                    "center_c": c,
                    "color": color,
                    "direction": direction,
                    "pointer_len": abs(tip_r - root_r) + abs(tip_c - root_c),
                    "ruler_side": ruler_side,
                    "ruler_len": ruler_len,
                }
            )
    return dials


def main(grid: np.ndarray) -> np.ndarray:
    """Fill forward rectangles using the pointer direction and adjacent ruler length."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for spec in _find_dials(g):
        fill, _mode, _steps = _fill_sections_from_grid(g, spec)
        for r, c in fill:
            out[r, c] = spec["color"]
    return out


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        r, c = spec["center_r"], spec["center_c"]
        grid[r - 1 : r + 2, c - 1 : c + 2] = GREY

        root_r, root_c = _root_pos(spec)
        grid[root_r, root_c] = spec["color"]

        dr, dc = DIR_VECS[spec["direction"]]
        for step in range(1, spec["pointer_len"] + 1):
            grid[root_r + dr * step, root_c + dc * step] = spec["color"]

        sign = SIDE_SIGN[spec["ruler_side"]]
        if spec["ruler_side"] in ("left", "right"):
            for offset in range(spec["ruler_len"]):
                grid[r, c + sign * (offset + 2)] = RULER
        else:
            for offset in range(spec["ruler_len"]):
                grid[r + sign * (offset + 2), c] = RULER
    return grid


def _sample_single_spec(h: int, w: int, color: int, blocked: set[tuple[int, int]]) -> dict | None:
    direction = random.choice(tuple(DIR_VECS))
    ruler_side = random.choice(SIDE_OPTIONS[direction])
    pointer_len = random.randint(1, 3)
    ruler_len = random.randint(1, 4)

    centers = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            spec = {
                "center_r": r,
                "center_c": c,
                "color": color,
                "direction": direction,
                "pointer_len": pointer_len,
                "ruler_side": ruler_side,
                "ruler_len": ruler_len,
            }
            if not _spec_fits(spec, h, w):
                continue
            if _object_cells(spec) & blocked:
                continue
            centers.append((r, c))

    if not centers:
        return None
    center_r, center_c = random.choice(centers)
    return {
        "center_r": center_r,
        "center_c": center_c,
        "color": color,
        "direction": direction,
        "pointer_len": pointer_len,
        "ruler_side": ruler_side,
        "ruler_len": ruler_len,
    }


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    n_dials = random.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
    want_obstacle = n_dials >= 2 and random.random() < 0.55

    colors = random.sample(POINTER_COLORS, n_dials)
    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    object_sets: list[set[tuple[int, int]]] = []

    for color in colors:
        placed = None
        for _ in range(120):
            candidate = _sample_single_spec(h, w, color, blocked)
            if candidate is None:
                break
            placed = candidate
            break
        if placed is None:
            return None
        specs.append(placed)
        cells = _object_cells(placed)
        object_sets.append(cells)
        blocked |= _expanded_cells(cells, h, w)

    grid = _render_input(h, w, specs)
    found = _find_dials(grid)
    if len(found) != n_dials:
        return None

    all_fill_cells: set[tuple[int, int]] = set()
    modes = []
    for spec, cells in zip(specs, object_sets):
        fill, mode, steps = _fill_sections_from_grid(grid, spec)
        if steps < MIN_FILL_STEPS:
            return None
        fill_set = set(fill)
        if fill_set & all_fill_cells:
            return None
        if fill_set & cells:
            return None
        all_fill_cells |= fill_set
        modes.append(mode)

    if want_obstacle and "obstacle" not in modes:
        return None
    if "boundary" not in modes:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid clock-pointer input")
