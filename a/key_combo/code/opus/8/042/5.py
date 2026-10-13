from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4)
SIDES = ("left", "right", "above", "below")
ORTHOGONAL_SIDES = {
    "left": ("above", "below"),
    "right": ("above", "below"),
    "above": ("left", "right"),
    "below": ("left", "right"),
}

GRID_MIN = 15
GRID_MAX = 18
SQUARE_COUNT_CHOICES = (3, 4)
SQUARE_SIZE = 3
SQUARE_GAP = 3
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_grey_squares(grid: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for coords in _components_of_mask(grid == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != 9 or r1 - r0 != 2 or c1 - c0 != 2:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            squares.append(
                {
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                }
            )
    squares.sort(key=lambda sq: (sq["r0"], sq["c0"]))
    return squares


def _find_color_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in PALETTE:
        for coords in _components_of_mask(grid == color):
            coords.sort()
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"]))
    return comps


def _touch_side(square: dict, r: int, c: int) -> str | None:
    if square["r0"] <= r <= square["r1"]:
        if c == square["c0"] - 1:
            return "left"
        if c == square["c1"] + 1:
            return "right"
    if square["c0"] <= c <= square["c1"]:
        if r == square["r0"] - 1:
            return "above"
        if r == square["r1"] + 1:
            return "below"
    return None


def _attach_components(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    squares = _find_grey_squares(grid)
    comps = _find_color_components(grid)
    attached: list[dict] = []
    for comp in comps:
        hits: list[tuple[int, str]] = []
        for sq_idx, square in enumerate(squares):
            sides = {
                side
                for r, c in comp["coords"]
                for side in [_touch_side(square, r, c)]
                if side is not None
            }
            if sides:
                if len(sides) != 1:
                    raise AssertionError("component touches multiple sides")
                hits.append((sq_idx, next(iter(sides))))
        if len(hits) != 1:
            raise AssertionError("component must touch exactly one square")
        sq_idx, side = hits[0]
        depth = 1 if comp["area"] == 1 else 3
        attached.append(
            {
                **comp,
                "square_index": sq_idx,
                "side": side,
                "depth": depth,
            }
        )
    return squares, attached


def _project_cells(square: dict, side: str, coords: list[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if side == "left":
        for r, _c in coords:
            for d in range(depth):
                cells.append((r, square["c0"] + d))
    elif side == "right":
        for r, _c in coords:
            for d in range(depth):
                cells.append((r, square["c1"] - d))
    elif side == "above":
        for _r, c in coords:
            for d in range(depth):
                cells.append((square["r0"] + d, c))
    else:
        for _r, c in coords:
            for d in range(depth):
                cells.append((square["r1"] - d, c))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """Project side-touching colored components into grey 3x3 squares, with depth determined by component size.

    Input structure:
    - The background is black.
    - The main objects are solid grey 3x3 squares.
    - Colored 4-connected components in blue/red/green/yellow touch square sides from outside.

    Rule:
    - For each colored component touching a grey square, the touched side determines the projection direction into that square.
    - Single-pixel components project only one layer into the square, while components of size 2 or 3 project through the full depth of the 3x3 square.
    - After projecting, erase every grey square to black and keep the original colored source components in place.
    - If projections overlap, the deeper projection overrides the shallower one.

    Key insight:
    - Two independent cues must be combined: spatial contact with the square determines direction, and component size determines depth.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == GREY] = BG

    squares, attached = _attach_components(g)
    depth_map = np.full(g.shape, -1, dtype=int)
    for comp in attached:
        square = squares[comp["square_index"]]
        for r, c in _project_cells(square, comp["side"], comp["coords"], comp["depth"]):
            if comp["depth"] > depth_map[r, c]:
                depth_map[r, c] = comp["depth"]
                out[r, c] = comp["color"]
            elif comp["depth"] == depth_map[r, c] and out[r, c] == comp["color"]:
                continue
    return out


def _side_slots(length: int, force_include: int | None = None) -> tuple[int, ...]:
    options = []
    for start in range(0, SQUARE_SIZE - length + 1):
        slots = tuple(range(start, start + length))
        if force_include is None or force_include in slots:
            options.append(slots)
    return random.choice(options)


def _source_coords(square: tuple[int, int], side: str, slots: tuple[int, ...]) -> list[tuple[int, int]]:
    r0, c0 = square
    if side == "left":
        return [(r0 + idx, c0 - 1) for idx in slots]
    if side == "right":
        return [(r0 + idx, c0 + SQUARE_SIZE) for idx in slots]
    if side == "above":
        return [(r0 - 1, c0 + idx) for idx in slots]
    return [(r0 + SQUARE_SIZE, c0 + idx) for idx in slots]


def _build_component(square: tuple[int, int], side: str, color: int, kind: str, *, force_include: int | None = None) -> dict:
    if kind == "single":
        slots = (random.randrange(SQUARE_SIZE),)
    else:
        length = random.choice((2, 3))
        slots = _side_slots(length, force_include=force_include)
    return {
        "side": side,
        "color": color,
        "kind": kind,
        "coords": _source_coords(square, side, slots),
    }


def _square_as_dict(square: tuple[int, int]) -> dict:
    r0, c0 = square
    return {"r0": r0, "c0": c0, "r1": r0 + 2, "c1": c0 + 2}


def _projection_profile(square: tuple[int, int], components: list[dict]) -> tuple[bool, bool, bool]:
    square_dict = _square_as_dict(square)
    winners: dict[tuple[int, int], tuple[int, int]] = {}
    deep_shallow_overlap = False
    for comp in components:
        depth = 1 if comp["kind"] == "single" else 3
        for cell in _project_cells(square_dict, comp["side"], comp["coords"], depth):
            if cell in winners:
                cur_depth, cur_color = winners[cell]
                if cur_depth == depth and cur_color != comp["color"]:
                    return False, False, False
                if cur_depth != depth:
                    deep_shallow_overlap = True
            prev = winners.get(cell)
            if prev is None or depth > prev[0]:
                winners[cell] = (depth, comp["color"])
    visible_shallow = any(depth == 1 for depth, _color in winners.values())
    visible_deep = any(depth == 3 for depth, _color in winners.values())
    return True, deep_shallow_overlap, visible_shallow or visible_deep


def _single_only_square(square: tuple[int, int]) -> list[dict]:
    for _ in range(40):
        side_count = random.choice((1, 2, 2))
        sides = random.sample(SIDES, side_count)
        colors = random.sample(PALETTE, side_count)
        comps = [_build_component(square, side, color, "single") for side, color in zip(sides, colors)]
        ok, _overlap, has_visible = _projection_profile(square, comps)
        if ok and has_visible:
            return comps
    raise RuntimeError("failed to design single-only square")


def _random_square_components(square: tuple[int, int]) -> list[dict]:
    for _ in range(80):
        side_count = random.choice((1, 2, 2, 3))
        sides = random.sample(SIDES, side_count)
        colors = random.sample(PALETTE, side_count)
        multi_side = None
        if random.random() < 0.65:
            multi_side = random.choice(sides)
        comps = []
        for side, color in zip(sides, colors):
            kind = "multi" if side == multi_side else "single"
            comps.append(_build_component(square, side, color, kind))
        ok, _overlap, _has_visible = _projection_profile(square, comps)
        if ok:
            return comps
    raise RuntimeError("failed to design random square")


def _overlap_square(square: tuple[int, int]) -> list[dict]:
    deep_side = random.choice(SIDES)
    shallow_side = random.choice(ORTHOGONAL_SIDES[deep_side])
    deep_color, shallow_color = random.sample(PALETTE, 2)
    edge_slot = 0 if shallow_side in {"above", "left"} else 2
    deep = _build_component(square, deep_side, deep_color, "multi", force_include=edge_slot)
    shallow = _build_component(square, shallow_side, shallow_color, "single")
    comps = [deep, shallow]
    ok, overlap, _has_visible = _projection_profile(square, comps)
    if not (ok and overlap):
        raise RuntimeError("failed to design overlap square")
    return comps


def _squares_clear(r0: int, c0: int, placed: list[tuple[int, int]]) -> bool:
    r1 = r0 + SQUARE_SIZE - 1
    c1 = c0 + SQUARE_SIZE - 1
    for pr0, pc0 in placed:
        pr1 = pr0 + SQUARE_SIZE - 1
        pc1 = pc0 + SQUARE_SIZE - 1
        separated = (
            r1 + SQUARE_GAP < pr0
            or pr1 + SQUARE_GAP < r0
            or c1 + SQUARE_GAP < pc0
            or pc1 + SQUARE_GAP < c0
        )
        if not separated:
            return False
    return True


def _place_squares(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(1, height - SQUARE_SIZE) for c in range(1, width - SQUARE_SIZE)]
    random.shuffle(candidates)
    placed: list[tuple[int, int]] = []
    for r0, c0 in candidates:
        if _squares_clear(r0, c0, placed):
            placed.append((r0, c0))
            if len(placed) == count:
                return placed
    return None


def _render_instance(height: int, width: int, squares: list[tuple[int, int]], square_components: list[list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)

    for r0, c0 in squares:
        grid[r0 : r0 + SQUARE_SIZE, c0 : c0 + SQUARE_SIZE] = GREY

    for comps in square_components:
        for comp in comps:
            for r, c in comp["coords"]:
                grid[r, c] = comp["color"]

    expected[:, :] = grid
    expected[expected == GREY] = BG
    depth_map = np.full((height, width), -1, dtype=int)
    for square, comps in zip(squares, square_components):
        square_dict = _square_as_dict(square)
        for comp in comps:
            depth = 1 if comp["kind"] == "single" else 3
            for r, c in _project_cells(square_dict, comp["side"], comp["coords"], depth):
                if depth > depth_map[r, c]:
                    depth_map[r, c] = depth
                    expected[r, c] = comp["color"]
    return grid, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        squares, attached = _attach_components(inp)
    except AssertionError:
        return False
    if len(squares) not in SQUARE_COUNT_CHOICES:
        return False
    if not any(comp["area"] == 1 for comp in attached):
        return False
    if not any(comp["area"] >= 2 for comp in attached):
        return False

    side_colors: dict[int, dict[str, int]] = defaultdict(dict)
    per_square_count = defaultdict(int)
    winners: dict[tuple[int, int], tuple[int, int]] = {}
    has_overlap = False
    visible_shallow = False
    for comp in attached:
        sq_idx = comp["square_index"]
        per_square_count[sq_idx] += 1
        if comp["side"] in side_colors[sq_idx]:
            return False
        side_colors[sq_idx][comp["side"]] = comp["color"]
        for cell in _project_cells(squares[sq_idx], comp["side"], comp["coords"], comp["depth"]):
            prev = winners.get(cell)
            if prev is not None:
                if prev[0] == comp["depth"] and prev[1] != comp["color"]:
                    return False
                if prev[0] != comp["depth"]:
                    has_overlap = True
            if prev is None or comp["depth"] > prev[0]:
                winners[cell] = (comp["depth"], comp["color"])

    if any(count == 0 for count in per_square_count.values()):
        return False
    if len(per_square_count) != len(squares):
        return False
    if not any(count >= 2 for count in per_square_count.values()):
        return False
    if not has_overlap:
        return False
    if not any(depth == 1 for depth, _color in winners.values()):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    square_count = random.choice(SQUARE_COUNT_CHOICES)
    squares = _place_squares(height, width, square_count)
    if squares is None:
        return None

    square_components: list[list[dict]] = []
    for idx, square in enumerate(squares):
        if idx == 0:
            comps = _overlap_square(square)
        elif idx == 1:
            comps = _single_only_square(square)
        else:
            comps = _random_square_components(square)
        square_components.append(comps)

    inp, expected = _render_instance(height, width, squares, square_components)
    if not _semantic_checks(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input")
