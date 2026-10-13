from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components


BG = 0
BRACKET = 2
OBSTACLE = 5
CLUSTER_COLORS = (1, 3, 4, 6, 7, 8, 9)
STABLE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)
MAX_GENERATE_ATTEMPTS = 400


def _is_bracket_component(component: dict) -> bool:
    if int(component["color"]) != BRACKET:
        return False
    r0, c0, r1, c1 = map(int, component["bbox"])
    if (r1 - r0, c1 - c0, int(component["area"])) != (2, 2, 5):
        return False
    coords = {tuple(map(int, rc)) for rc in component["coords"]}
    target = {(r, c1) for r in range(r0, r1 + 1)} | {(r1, c) for c in range(c0, c1 + 1)}
    return coords == target


def _locate_bracket_and_marker(grid: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    bracket_components = [
        comp
        for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[BRACKET])
        if _is_bracket_component(comp)
    ]
    assert len(bracket_components) == 1, "expected exactly one bottom-right corner bracket"
    bracket = bracket_components[0]
    bracket_cells = {tuple(map(int, rc)) for rc in bracket["coords"]}
    r0, c0, r1, c1 = map(int, bracket["bbox"])
    markers = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if (r, c) not in bracket_cells and int(grid[r, c]) not in (BG, BRACKET)
    ]
    assert len(markers) == 1, "expected exactly one non-background marker inside the bracket"
    return bracket_cells, markers[0]


def _static_mask(grid: np.ndarray, bracket_cells: set[tuple[int, int]], marker: tuple[int, int]) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in bracket_cells:
        mask[r, c] = True
    mask[grid == OBSTACLE] = True
    mask[marker] = True
    return mask


def _settle_pixels(grid: np.ndarray, static_mask: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    movers = sorted(
        [(int(r), int(c)) for r, c in np.argwhere((out != BG) & (~static_mask))],
        key=lambda rc: (rc[0], rc[1]),
        reverse=True,
    )
    for r, c in movers:
        if int(out[r, c]) == BG or bool(static_mask[r, c]):
            continue
        color = int(out[r, c])
        out[r, c] = BG
        nr = r
        while nr + 1 < out.shape[0] and int(out[nr + 1, c]) == BG:
            nr += 1
        out[nr, c] = color
    return out


def _transform_with_color(grid: np.ndarray, erase_color: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bracket_cells, marker = _locate_bracket_and_marker(g)
    out = g.copy()
    out[out == int(erase_color)] = BG
    out[marker] = int(g[marker])
    return _settle_pixels(out, _static_mask(out, bracket_cells, marker))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the bracket-selected support color and let the remaining pixels settle by vertical gravity.

    Input structure:
    - The background is black. A red 3x3 corner bracket sits in the lower-right region, with one non-red marker pixel inside its bounding box.
    - Outside the bracket there are small 4-connected color clusters, plus 2-4 gray obstacle pixels that never move.
    - The marker color matches one cluster color whose pixels act as removable support for other colors stacked above them.

    Rule:
    - Find the red corner bracket and read the unique marker color inside it.
    - Erase every pixel of that marker color except the marker itself.
    - Then, processing movable pixels from lower rows toward upper rows, let each non-background, non-obstacle, non-bracket, non-marker pixel fall straight downward until the next cell would be occupied or it reaches the bottom row.
    - Keep the bracket, the marker, and all gray obstacle pixels fixed in place.

    Key insight:
    - The bracket does not name the final color to keep or remove for its own sake; it selects which support structure disappears, and that selected erasure changes the later gravity outcome.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _bracket_cells, marker = _locate_bracket_and_marker(g)
    marker_color = int(g[marker])
    assert marker_color not in (BG, BRACKET, OBSTACLE)
    return _transform_with_color(g, marker_color)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _shift_cells(cells: list[tuple[int, int]], dr: int) -> list[tuple[int, int]]:
    return [(r - dr, c) for r, c in cells]


def _interval_layout(width: int, include_stable: bool) -> dict[str, tuple[int, int]]:
    specs = [("tower", 3), ("drop", 1)]
    if include_stable:
        specs.append(("stable", 2))
    random.shuffle(specs)
    usable_width = width - 4
    total_needed = sum(w for _name, w in specs) + (len(specs) - 1)
    extra = usable_width - total_needed
    if extra < 0:
        raise AssertionError("grid too narrow for the reserved layout")
    pads = [0] * (len(specs) + 1)
    for _ in range(extra):
        pads[random.randrange(len(pads))] += 1
    layout: dict[str, tuple[int, int]] = {}
    col = pads[0]
    for i, (name, span) in enumerate(specs):
        layout[name] = (col, col + span)
        col += span
        if i < len(specs) - 1:
            col += 1 + pads[i + 1]
    return layout


def _build_case() -> tuple[np.ndarray, np.ndarray, dict]:
    h = random.randint(12, 14)
    w = random.randint(12, 14)
    include_stable = bool(random.getrandbits(1))
    layout = _interval_layout(w, include_stable)

    support_color = random.choice(CLUSTER_COLORS)
    mover_palette = [c for c in CLUSTER_COLORS if c != support_color]
    base_color = random.choice(mover_palette)
    extra_color = random.choice([c for c in mover_palette if c != base_color])
    use_extra_color = bool(random.getrandbits(1))

    tower_a_color = base_color
    tower_b_color = extra_color if use_extra_color else base_color
    drop_color = random.choice((base_color, tower_b_color))
    stable_color = random.choice((base_color, tower_b_color, extra_color))

    marker_candidates = [(h - 3, w - 3), (h - 3, w - 2), (h - 2, w - 3), (h - 2, w - 2)]
    marker = random.choice(marker_candidates)
    bracket_cells = {(r, w - 1) for r in range(h - 3, h)} | {(h - 1, c) for c in range(w - 3, w)}

    tower_c0, tower_c1 = layout["tower"]
    tower_cols = list(range(tower_c0, tower_c1))
    drop_col = layout["drop"][0]

    drop_len = random.randint(3, 4)
    tower_drop = random.randint(2, 3)
    tower_shelf_r = random.choice((h - 5, h - 4))
    drop_shift = random.randint(3, 4)
    if tower_shelf_r - 2 - tower_drop < 1 or h - drop_len - drop_shift < 0:
        raise AssertionError("invalid sampled heights")

    out = np.full((h, w), BG, dtype=int)
    _paint_cells(out, sorted(bracket_cells), BRACKET)
    out[marker] = support_color

    tower_shelf = [(tower_shelf_r, c) for c in tower_cols]
    tower_a = [(tower_shelf_r - 1, c) for c in tower_cols]
    tower_b = [(tower_shelf_r - 2, c) for c in tower_cols]
    drop_final = [(r, drop_col) for r in range(h - drop_len, h)]

    _paint_cells(out, tower_shelf, OBSTACLE)
    _paint_cells(out, tower_a, tower_a_color)
    _paint_cells(out, tower_b, tower_b_color)
    _paint_cells(out, drop_final, drop_color)

    stable_cells: list[tuple[int, int]] = []
    if include_stable:
        stable_c0, _stable_c1 = layout["stable"]
        stable_shape = random.choice(STABLE_SHAPES)
        stable_r0 = random.randint(1, max(1, h - 8))
        stable_cells = [(stable_r0 + dr, stable_c0 + dc) for dr, dc in stable_shape]
        _paint_cells(out, stable_cells, stable_color)

    tower_a_input = _shift_cells(tower_a, tower_drop)
    tower_b_input = _shift_cells(tower_b, tower_drop)
    tower_support = [(tower_a_input[0][0] + 1, c) for c in tower_cols]
    drop_input = _shift_cells(drop_final, drop_shift)
    drop_support = [(drop_input[-1][0] + 1 + k, drop_col) for k in range(3)]

    inp = out.copy()
    for r, c in tower_a + tower_b + drop_final:
        inp[r, c] = BG
    _paint_cells(inp, tower_a_input, tower_a_color)
    _paint_cells(inp, tower_b_input, tower_b_color)
    _paint_cells(inp, tower_support, support_color)
    _paint_cells(inp, drop_input, drop_color)
    _paint_cells(inp, drop_support, support_color)

    movable_cols = set(tower_cols) | {drop_col}
    spare_obstacles = 0
    candidates = [
        (r, c)
        for r in range(0, h - 3)
        for c in range(0, w - 4)
        if c not in movable_cols
        and int(inp[r, c]) == BG
        and int(out[r, c]) == BG
    ]
    if candidates and bool(random.getrandbits(1)):
        obstacle_cell = random.choice(candidates)
        inp[obstacle_cell] = OBSTACLE
        out[obstacle_cell] = OBSTACLE
        spare_obstacles = 1

    meta = {
        "support_color": support_color,
        "marker": marker,
        "expected": out,
        "tower_a": tower_a,
        "tower_b": tower_b,
        "drop_final": drop_final,
        "spare_obstacles": spare_obstacles,
    }
    return inp, out, meta


def _validate_case(inp: np.ndarray, out: np.ndarray, meta: dict) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    try:
        bracket_cells, marker = _locate_bracket_and_marker(inp)
    except AssertionError:
        return False
    if marker != meta["marker"] or int(inp[marker]) != int(meta["support_color"]):
        return False
    if not np.array_equal(main(inp), out):
        return False
    if np.array_equal(inp, out):
        return False

    present = set(map(int, np.unique(inp))) - {BG, BRACKET, OBSTACLE}
    if meta["support_color"] not in present or not (2 <= len(present) <= 3):
        return False

    stripped = inp.copy()
    stripped[marker] = BG
    for comp in find_components(stripped, bg=BG, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        if color in (BRACKET, OBSTACLE):
            continue
        if not (3 <= int(comp["area"]) <= 6):
            return False

    if not any(
        r > 0 and int(inp[r, c]) == meta["support_color"] and int(inp[r - 1, c]) not in (BG, BRACKET, OBSTACLE, meta["support_color"])
        for r, c in np.argwhere(inp == meta["support_color"])
        if (int(r), int(c)) != marker
    ):
        return False

    erase_only = inp.copy()
    erase_only[erase_only == meta["support_color"]] = BG
    erase_only[marker] = meta["support_color"]
    if np.array_equal(erase_only, out):
        return False

    no_erase_gravity = _settle_pixels(inp, _static_mask(inp, bracket_cells, marker))
    if np.array_equal(no_erase_gravity, out):
        return False

    for other_color in present - {meta["support_color"]}:
        if np.array_equal(_transform_with_color(inp, int(other_color)), out):
            return False

    obstacle_count = int(np.sum(inp == OBSTACLE))
    if obstacle_count not in (3, 4):
        return False

    if not any(int(out[r + 1, c]) == OBSTACLE for r, c in meta["tower_a"]):
        return False
    if not any(r == out.shape[0] - 1 for r, _c in meta["drop_final"]):
        return False
    tower_a_set = set(meta["tower_a"])
    if not all((r + 1, c) in tower_a_set for r, c in meta["tower_b"]):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        inp, out, meta = _build_case()
        if _validate_case(inp, out, meta):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 3")
