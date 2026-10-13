from __future__ import annotations

import random

import numpy as np


BG = 0
SQUARE = 5
MARKER = 9
SQUARE_SIZE = 5
ERASE_THRESHOLD = 2

GRID_MIN = 16
GRID_MAX = 24
POSITION_DELTA = 8
GEN_ATTEMPTS = 400

SIDES = ("left", "right", "above", "below")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_dicts(mask: np.ndarray) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r1, c1 = arr.min(axis=0)
            r2, c2 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _square_objects(grid: np.ndarray) -> list[dict]:
    squares = []
    for idx, comp in enumerate(_component_dicts(grid == SQUARE)):
        r0, c0, r1, c1 = comp["bbox"]
        assert comp["area"] == SQUARE_SIZE * SQUARE_SIZE, "grey object must be a 5x5 square"
        assert r1 - r0 + 1 == SQUARE_SIZE and c1 - c0 + 1 == SQUARE_SIZE, "grey object bbox must be 5x5"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == SQUARE), "grey square must be solid"
        squares.append(
            {
                "id": idx,
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert 3 <= len(squares) <= 4, "expected 3-4 grey squares"
    squares.sort(key=lambda item: item["bbox"])
    for idx, square in enumerate(squares):
        square["id"] = idx
    return squares


def _touch_side(r: int, c: int, square: dict) -> str | None:
    r0, c0, r1, c1 = square["bbox"]
    if r0 <= r <= r1:
        if c == c0 - 1:
            return "left"
        if c == c1 + 1:
            return "right"
    if c0 <= c <= c1:
        if r == r0 - 1:
            return "above"
        if r == r1 + 1:
            return "below"
    return None


def _projection_step(side: str) -> tuple[int, int]:
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    if side == "above":
        return 1, 0
    assert side == "below"
    return -1, 0


def _project_targets(coords: np.ndarray, side: str, area: int, square: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = square["bbox"]
    dr, dc = _projection_step(side)
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        tr = int(r + dr * area)
        tc = int(c + dc * area)
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            out.add((tr, tc))
    return out


def _analyze(grid: np.ndarray) -> tuple[list[dict], dict[int, dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BG, SQUARE, MARKER}, "unexpected colors present"

    squares = _square_objects(g)
    marker_components = _component_dicts(g == MARKER)
    assert len(marker_components) == len(squares), "expected exactly one marker component per square"

    by_square: dict[int, dict] = {}
    for comp in marker_components:
        assert 1 <= comp["area"] <= 3, "marker component area must be 1-3"
        matches: list[tuple[dict, str]] = []
        for square in squares:
            side_hits = {
                side
                for r, c in comp["coords"]
                for side in [_touch_side(int(r), int(c), square)]
                if side is not None
            }
            if side_hits:
                assert len(side_hits) == 1, "marker component must touch only one side of its square"
                matches.append((square, next(iter(side_hits))))
        assert len(matches) == 1, "marker component must belong to exactly one square"
        square, side = matches[0]
        assert square["id"] not in by_square, "each square must have exactly one marker component"
        by_square[square["id"]] = {
            "square": square,
            "side": side,
            "area": comp["area"],
            "coords": comp["coords"],
        }

    assert len(by_square) == len(squares), "every square must have a marker component"
    return squares, by_square


def _build_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    squares, by_square = _analyze(g)
    out = g.copy()

    for info in by_square.values():
        for r, c in _project_targets(info["coords"], info["side"], info["area"], info["square"]):
            out[r, c] = MARKER

    for square in squares:
        info = by_square[square["id"]]
        if info["area"] < ERASE_THRESHOLD:
            r0, c0, r1, c1 = square["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Project each maroon marker component into its adjacent grey square, then erase squares whose marker area is too small.

    Input structure:
    - The grid has a black background, 3-4 disjoint solid grey 5x5 squares, and one maroon 4-connected marker component attached to exactly one side of each square.
    - Every marker component touches a unique square on exactly one side, and its 4-connected area is 1-3 pixels.
    - The marker pixels stay outside the square; the square itself is otherwise a solid 5x5 block.

    Rule:
    - For each square, find its adjacent marker component and the side it touches.
    - Let A be that component's 4-connected area. Project every marker pixel A cells inward along the normal of the touched side and paint any in-square landing cells maroon.
    - After all projections are painted, erase the entire square to black if A < 2; the external marker component itself stays unchanged.

    Key insight:
    - The same connected-component area controls two coupled effects: how deep the projection enters the square and whether the square survives at all.
    """
    return _build_expected(np.asarray(grid, dtype=int))


def _marker_slots(r0: int, c0: int, side: str) -> list[tuple[int, int]]:
    if side == "left":
        return [(r, c0 - 1) for r in range(r0 + 1, r0 + SQUARE_SIZE - 1)]
    if side == "right":
        return [(r, c0 + SQUARE_SIZE) for r in range(r0 + 1, r0 + SQUARE_SIZE - 1)]
    if side == "above":
        return [(r0 - 1, c) for c in range(c0 + 1, c0 + SQUARE_SIZE - 1)]
    assert side == "below"
    return [(r0 + SQUARE_SIZE, c) for c in range(c0 + 1, c0 + SQUARE_SIZE - 1)]


def _choose_component_slots(r0: int, c0: int, side: str, area: int) -> list[tuple[int, int]]:
    slots = _marker_slots(r0, c0, side)
    start = random.randint(0, len(slots) - area)
    return slots[start : start + area]


def _choose_positions(h: int, w: int, count: int) -> list[tuple[int, int]]:
    max_top = h - (SQUARE_SIZE + 1) - POSITION_DELTA
    max_left = w - (SQUARE_SIZE + 1) - POSITION_DELTA
    assert max_top >= 1 and max_left >= 1

    top_r = random.randint(1, max_top)
    left_c = random.randint(1, max_left)
    bottom_r = random.randint(top_r + POSITION_DELTA, h - SQUARE_SIZE - 1)
    right_c = random.randint(left_c + POSITION_DELTA, w - SQUARE_SIZE - 1)

    quads = [(top_r, left_c), (top_r, right_c), (bottom_r, left_c), (bottom_r, right_c)]
    chosen = random.sample(quads, count)
    random.shuffle(chosen)
    return chosen


def _choose_specs(count: int) -> list[tuple[str, int]]:
    same_side = random.choice(SIDES)
    other_side = random.choice([side for side in SIDES if side != same_side])
    if count == 3:
        middle_area = random.choice([2, 3])
        areas = [1, middle_area, 1]
        sides = [same_side, same_side, other_side]
    else:
        duplicate = random.choice([1, 2, 3])
        if duplicate == 1:
            areas = [1, 2, 1, 3]
        elif duplicate == 2:
            areas = [1, 2, 3, 2]
        else:
            areas = [1, 3, 2, 3]
        remaining = [side for side in SIDES if side != same_side]
        other_a, other_b = random.sample(remaining, 2)
        sides = [same_side, same_side, other_a, other_b]
    specs = list(zip(sides, areas))
    random.shuffle(specs)
    return specs


def _render_instance(h: int, w: int, positions: list[tuple[int, int]], specs: list[tuple[str, int]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for (r0, c0), (side, area) in zip(positions, specs):
        grid[r0 : r0 + SQUARE_SIZE, c0 : c0 + SQUARE_SIZE] = SQUARE
        for r, c in _choose_component_slots(r0, c0, side, area):
            grid[r, c] = MARKER
    return grid


def _has_same_side_different_area(infos: list[dict]) -> bool:
    for i, info_a in enumerate(infos):
        for info_b in infos[i + 1 :]:
            if info_a["side"] == info_b["side"] and info_a["area"] != info_b["area"]:
                return True
    return False


def _has_same_area_different_side(infos: list[dict]) -> bool:
    for i, info_a in enumerate(infos):
        for info_b in infos[i + 1 :]:
            if info_a["area"] == info_b["area"] and info_a["side"] != info_b["side"]:
                return True
    return False


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BG, SQUARE, MARKER}

    squares, by_square = _analyze(g)
    infos = [by_square[square["id"]] for square in squares]
    areas = [int(info["area"]) for info in infos]
    sides = [str(info["side"]) for info in infos]

    assert min(areas) == 1, "must include an erased-square example"
    assert any(area >= 2 for area in areas), "must include surviving-square examples"
    if len(squares) == 4:
        assert set(areas) == {1, 2, 3}, "4-square cases must cover all three areas"
    else:
        assert set(areas) in ({1, 2}, {1, 3}), "3-square cases must cover erase + one survivor depth family"
    assert len(set(sides)) >= 2, "must vary touched sides"
    assert _has_same_side_different_area(infos), "must include same-side/different-area counterexample"
    assert _has_same_area_different_side(infos), "must include same-area/different-side counterexample"

    for info in infos:
        square = info["square"]
        r0, c0, r1, c1 = square["bbox"]
        touching = [
            (int(r), int(c))
            for r, c in info["coords"]
            if _touch_side(int(r), int(c), square) == info["side"]
        ]
        assert touching, "marker component must directly touch its square"
        for r, c in touching:
            if info["side"] in {"left", "right"}:
                assert r0 < r < r1, "marker may not touch square corners"
            else:
                assert c0 < c < c1, "marker may not touch square corners"

    expected = _build_expected(g)
    actual = main(g)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[g == MARKER], g[g == MARKER]), "external markers must stay unchanged"

    for info in infos:
        square = info["square"]
        r0, c0, r1, c1 = square["bbox"]
        patch = actual[r0 : r1 + 1, c0 : c1 + 1]
        if info["area"] == 1:
            assert np.all(patch == BG), "area-1 square must be fully erased"
            continue
        expected_patch = np.full((SQUARE_SIZE, SQUARE_SIZE), SQUARE, dtype=int)
        for tr, tc in _project_targets(info["coords"], info["side"], info["area"], square):
            expected_patch[tr - r0, tc - c0] = MARKER
        assert np.array_equal(patch, expected_patch), "surviving square must contain exactly its depth projection"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.choices([3, 4], weights=[2, 3], k=1)[0]

    try:
        positions = _choose_positions(h, w, count)
    except ValueError:
        return None

    specs = _choose_specs(count)
    grid = _render_instance(h, w, positions, specs)
    try:
        _validate_generated_grid(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
