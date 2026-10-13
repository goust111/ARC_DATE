from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TOP = 1
RIGHT = 2
BOTTOM = 3
LEFT = 4
GREY = 5
ORANGE = 7

SIDE_COLORS = {
    "top": TOP,
    "right": RIGHT,
    "bottom": BOTTOM,
    "left": LEFT,
}
DIRS = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
CROSS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _parse_frame(grid: np.ndarray) -> dict[str, object]:
    grey_coords = np.argwhere(grid == GREY)
    assert len(grey_coords) == 4, "expected exactly four grey frame corners in the input"

    rows = sorted({int(r) for r, _ in grey_coords})
    cols = sorted({int(c) for _, c in grey_coords})
    assert len(rows) == 2 and len(cols) == 2, "grey corners must define one rectangle"
    r0, r1 = rows
    c0, c1 = cols
    assert {(r0, c0), (r0, c1), (r1, c0), (r1, c1)} == {
        (int(r), int(c)) for r, c in grey_coords
    }
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must enclose a non-trivial interior"

    top_color = int(grid[r0, c0 + 1])
    right_color = int(grid[r0 + 1, c1])
    bottom_color = int(grid[r1, c0 + 1])
    left_color = int(grid[r0 + 1, c0])
    assert np.all(grid[r0, c0 + 1 : c1] == top_color)
    assert np.all(grid[r1, c0 + 1 : c1] == bottom_color)
    assert np.all(grid[r0 + 1 : r1, c0] == left_color)
    assert np.all(grid[r0 + 1 : r1, c1] == right_color)

    side_colors = {
        "top": top_color,
        "right": right_color,
        "bottom": bottom_color,
        "left": left_color,
    }
    assert len(set(side_colors.values())) == 4, "frame sides must use four distinct colors"
    return {
        "bbox": (r0, c0, r1, c1),
        "interior": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "side_colors": side_colors,
        "corners": {(r0, c0), (r0, c1), (r1, c0), (r1, c1)},
    }


def _inside_box(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _notch_pattern(r: int, c: int, direction: str) -> dict[str, tuple[tuple[int, int], ...] | str]:
    dr, dc = DIRS[direction]
    pr, pc = -dc, dr
    cross = tuple((r + rr, c + cc) for rr, cc in CROSS)
    support = (
        (r - 2 * dr, c - 2 * dc),
        (r - dr + pr, c - dc + pc),
        (r - dr - pr, c - dc - pc),
        (r + 2 * pr, c + 2 * pc),
        (r - 2 * pr, c - 2 * pc),
    )
    open_side = (
        (r + 2 * dr, c + 2 * dc),
        (r + dr + pr, c + dc + pc),
        (r + dr - pr, c + dc - pc),
    )
    return {"direction": direction, "cross": cross, "support": support, "open_side": open_side}


def _notch_directions(grid: np.ndarray, frame: dict[str, object], r: int, c: int) -> list[str]:
    interior = frame["interior"]
    cross = tuple((r + rr, c + cc) for rr, cc in CROSS)
    if not all(_inside_box(interior, rr, cc) for rr, cc in cross):
        return []
    if not all(int(grid[rr, cc]) == BG for rr, cc in cross):
        return []

    directions = []
    for direction in DIRS:
        pattern = _notch_pattern(r, c, direction)
        support = pattern["support"]
        open_side = pattern["open_side"]
        if not all(_inside_box(interior, rr, cc) for rr, cc in support + open_side):
            continue
        if not all(int(grid[rr, cc]) == ORANGE for rr, cc in support):
            continue
        if not all(int(grid[rr, cc]) == BG for rr, cc in open_side):
            continue
        directions.append(direction)
    return directions


def _find_notches(grid: np.ndarray, frame: dict[str, object]) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame["interior"]
    found: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            directions = _notch_directions(grid, frame, r, c)
            if len(directions) != 1:
                continue
            pattern = _notch_pattern(r, c, directions[0])
            cross = tuple(pattern["cross"])
            assert not any(cell in used for cell in cross), "plus notches must not overlap"
            used.update(cross)
            found.append(
                {
                    "center": (r, c),
                    "direction": directions[0],
                    "cross": cross,
                    "color": int(frame["side_colors"][directions[0]]),
                }
            )
    assert found, "expected at least one plus-shaped notch"
    return found


def _expand_territories(out: np.ndarray, frame: dict[str, object], notches: list[dict[str, object]]) -> np.ndarray:
    interior = frame["interior"]
    frontiers = [set(notch["cross"]) for notch in notches]

    while True:
        claims: dict[tuple[int, int], set[int]] = {}
        for idx, frontier in enumerate(frontiers):
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if not _inside_box(interior, nr, nc):
                        continue
                    if int(out[nr, nc]) != BG:
                        continue
                    claims.setdefault((nr, nc), set()).add(idx)

        if not claims:
            break

        next_frontiers = [set() for _ in notches]
        for cell, owners in claims.items():
            if len(owners) == 1:
                owner = next(iter(owners))
                out[cell] = int(notches[owner]["color"])
                next_frontiers[owner].add(cell)
            else:
                out[cell] = GREY
        frontiers = next_frontiers

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each cross notch with its facing frame-side color, then grow all seeds through the interior background until collisions turn into grey borders.

    Input structure:
    - The grid has black background, a one-cell-thick rectangular frame whose corners are grey and whose four sides have distinct colors.
    - Inside the frame sits an orange foreground object with several five-cell black plus notches carved into its boundary.
    - Each notch opens toward exactly one frame side, while the other three surrounding directions are supported by orange object cells.

    Rule:
    - Detect every black plus notch and determine which side of the frame it faces from the local open direction.
    - Recolor all five cells of that notch with the facing side's frame color; these filled pluses become the initial occupied territories.
    - Repeatedly expand every territory one 4-neighbor layer at a time into black cells strictly inside the frame, while orange object pixels and the frame stay fixed.
    - If multiple territories reach the same black cell in the same round, paint that cell grey and block further expansion through it.

    Key insight:
    - The frame-side colors label the notch seeds, and the actual output comes from a synchronous multi-source expansion in the black moat around the orange object, with same-round conflicts freezing into grey Voronoi-style boundaries.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame = _parse_frame(g)
    notches = _find_notches(g, frame)

    out = g.copy()
    for notch in notches:
        for rr, cc in notch["cross"]:
            out[rr, cc] = int(notch["color"])

    return _expand_territories(out, frame, notches)


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0] = GREY
    grid[r0, c1] = GREY
    grid[r1, c0] = GREY
    grid[r1, c1] = GREY
    grid[r0, c0 + 1 : c1] = TOP
    grid[r1, c0 + 1 : c1] = BOTTOM
    grid[r0 + 1 : r1, c0] = LEFT
    grid[r0 + 1 : r1, c1] = RIGHT


def _choose_layout(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int, int, int, int, int, int] | None:
    frame_h = int(rng.integers(15, 23))
    frame_w = int(rng.integers(15, 23))
    pad_top = int(rng.integers(0, 4))
    pad_bottom = int(rng.integers(0, 4))
    pad_left = int(rng.integers(0, 4))
    pad_right = int(rng.integers(0, 4))
    h = frame_h + pad_top + pad_bottom
    w = frame_w + pad_left + pad_right
    if h > 30 or w > 30:
        return None

    moat_top = int(rng.integers(2, 6))
    moat_bottom = int(rng.integers(2, 6))
    moat_left = int(rng.integers(2, 6))
    moat_right = int(rng.integers(2, 6))
    obj_h = frame_h - 2 - moat_top - moat_bottom
    obj_w = frame_w - 2 - moat_left - moat_right
    if obj_h < 7 or obj_w < 7:
        return None

    r0 = pad_top
    c0 = pad_left
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    or0 = r0 + 1 + moat_top
    oc0 = c0 + 1 + moat_left
    or1 = or0 + obj_h - 1
    oc1 = oc0 + obj_w - 1
    return h, w, r0, c0, r1, c1, or0, oc0, or1, oc1


def _notch_center_choices(
    direction: str, or0: int, oc0: int, or1: int, oc1: int
) -> list[tuple[int, int]]:
    if direction == "top":
        return [(or0, c) for c in range(oc0 + 2, oc1 - 1)]
    if direction == "bottom":
        return [(or1, c) for c in range(oc0 + 2, oc1 - 1)]
    if direction == "left":
        return [(r, oc0) for r in range(or0 + 2, or1 - 1)]
    return [(r, oc1) for r in range(or0 + 2, or1 - 1)]


def _paint_notch_input(grid: np.ndarray, center: tuple[int, int], direction: str) -> None:
    pattern = _notch_pattern(center[0], center[1], direction)
    for rr, cc in pattern["cross"]:
        grid[rr, cc] = BG


def _valid_generated_example(
    inp: np.ndarray, expected_directions: set[str], notch_count: int
) -> bool:
    try:
        frame = _parse_frame(inp)
        notches = _find_notches(inp, frame)
        out = main(inp)
    except Exception:
        return False

    if len(notches) != notch_count:
        return False
    if {str(notch["direction"]) for notch in notches} != expected_directions:
        return False

    if np.count_nonzero(out != inp) <= 5 * notch_count:
        return False
    if np.count_nonzero(out == GREY) <= 4:
        return False
    for direction in expected_directions:
        if np.count_nonzero(out == SIDE_COLORS[direction]) <= 5:
            return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layout = _choose_layout(rng)
    if layout is None:
        return None
    h, w, r0, c0, r1, c1, or0, oc0, or1, oc1 = layout

    grid = np.zeros((h, w), dtype=int)
    _draw_frame(grid, r0, c0, r1, c1)
    grid[or0 : or1 + 1, oc0 : oc1 + 1] = ORANGE

    notch_count = 4 if float(rng.random()) < 0.6 else 3
    directions = set(random.sample(tuple(DIRS), notch_count))
    for direction in directions:
        choices = _notch_center_choices(direction, or0, oc0, or1, oc1)
        if not choices:
            return None
        center = choices[int(rng.integers(0, len(choices)))]
        _paint_notch_input(grid, center, direction)

    if not _valid_generated_example(grid, directions, notch_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
