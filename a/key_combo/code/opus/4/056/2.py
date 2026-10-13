from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, paste, random_free_location


TARGET_COLORS = (Color.BLUE, Color.RED, Color.GREEN)
BRACKET_COLOR = Color.GREY
BRACKET_ORIENTATIONS = ("tl", "tr", "bl", "br")
BRACKET_SIZES = ((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4))
GRID_SIDE_RANGE = (14, 19)
RECT_COUNT_RANGE = (5, 7)
RECT_SIDE_CHOICES = (3, 4, 5)
MAX_GEN_ATTEMPTS = 500
_GEN_CALL_COUNT = 0


def _bracket_cells(height: int, width: int, orientation: str) -> set[tuple[int, int]]:
    if orientation == "tl":
        return {(0, c) for c in range(width)} | {(r, 0) for r in range(height)}
    if orientation == "tr":
        return {(0, c) for c in range(width)} | {(r, width - 1) for r in range(height)}
    if orientation == "bl":
        return {(height - 1, c) for c in range(width)} | {(r, 0) for r in range(height)}
    if orientation == "br":
        return {(height - 1, c) for c in range(width)} | {(r, width - 1) for r in range(height)}
    raise ValueError(f"unknown bracket orientation: {orientation}")


def _is_corner_bracket(component: dict) -> bool:
    if int(component["color"]) != int(BRACKET_COLOR):
        return False
    r0, c0, r1, c1 = component["bbox"]
    height = bbox_h(component["bbox"])
    width = bbox_w(component["bbox"])
    if height < 3 or width < 3:
        return False
    if int(component["area"]) != height + width - 1:
        return False
    if not (5 <= int(component["area"]) <= 7):
        return False

    coords = {(int(r), int(c)) for r, c in component["coords"]}
    for orientation in BRACKET_ORIENTATIONS:
        expected = {(r0 + rr, c0 + cc) for rr, cc in _bracket_cells(height, width, orientation)}
        if coords == expected:
            return True
    return False


def _find_bracket_and_marker(grid: np.ndarray, bg: int) -> tuple[dict, tuple[int, int]]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        min_area=1,
    )
    matches: list[tuple[dict, tuple[int, int]]] = []
    for component in components:
        if not _is_corner_bracket(component):
            continue
        r0, c0, r1, c1 = component["bbox"]
        bracket_coords = {(int(r), int(c)) for r, c in component["coords"]}
        markers = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if (r, c) not in bracket_coords and int(grid[r, c]) != int(bg)
        ]
        if len(markers) == 1:
            matches.append((component, markers[0]))
    assert len(matches) == 1, "expected exactly one grey corner bracket with one enclosed marker"
    return matches[0]


def _rectangle_components(
    grid: np.ndarray,
    bg: int,
    marker: tuple[int, int],
) -> list[dict]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=TARGET_COLORS,
        min_area=1,
    )

    marker_hits = 0
    rectangles: list[dict] = []
    for component in components:
        if int(component["area"]) == 1:
            cell = tuple(map(int, component["coords"][0]))
            assert cell == marker, "the only non-rectangle singleton should be the bracket marker"
            marker_hits += 1
            continue

        r0, c0, r1, c1 = component["bbox"]
        height = bbox_h(component["bbox"])
        width = bbox_w(component["bbox"])
        color = int(component["color"])
        assert height >= 3 and width >= 3, "every colored object outside the bracket must be at least 3x3"
        assert int(component["area"]) == height * width, "expected solid filled rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle bbox must be uniformly filled"
        rectangles.append(component)

    assert marker_hits == 1, "expected exactly one enclosed singleton marker"
    assert RECT_COUNT_RANGE[0] <= len(rectangles) <= RECT_COUNT_RANGE[1], "unexpected rectangle count"
    rect_colors = {int(rect["color"]) for rect in rectangles}
    assert rect_colors == set(map(int, TARGET_COLORS)), "expected blue, red, and green rectangles"
    return rectangles


def _apply_rule(
    grid: np.ndarray,
    rectangles: list[dict],
    target_color: int,
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        if int(rect["color"]) == int(target_color):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        else:
            out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)
    return out


def _erase_only(
    grid: np.ndarray,
    rectangles: list[dict],
    target_color: int,
    bg: int,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for rect in rectangles:
        if int(rect["color"]) != int(target_color):
            continue
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Erase all solid rectangles whose color matches the bracket marker, then hollow the surviving rectangles.

    Input structure:
    - The background is black, with one grey 4-connected corner bracket and exactly
      one colored marker strictly inside the bracket's bounding box.
    - Outside the bracket there are 5 to 7 non-touching solid axis-aligned
      rectangles, colored only blue, red, or green, with each color present.
    - Every rectangle is a single-color filled block of size at least 3x3.

    Rule:
    - Read the marker color inside the grey bracket; that color is the target color.
    - Erase every solid rectangle of the target color to the black background.
    - For every rectangle of the other two colors, keep its 1-pixel outer border
      and recolor its strict interior to black.
    - Leave the grey bracket and its enclosed marker unchanged.

    Key insight:
    - The bracket is a color selector: it decides which rectangles disappear
      completely, while every rectangle of the other colors survives only as a hollow frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    bracket, marker = _find_bracket_and_marker(g, bg)
    marker_color = int(g[marker])
    assert marker_color in TARGET_COLORS, "marker color must match one of the rectangle colors"

    rectangles = _rectangle_components(g, bg, marker)
    return _apply_rule(g, rectangles, marker_color, bg)


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _make_bracket_patch(target_color: int, orientation: str) -> np.ndarray:
    height, width = random.choice(BRACKET_SIZES)
    patch = np.zeros((height, width), dtype=int)
    for r, c in _bracket_cells(height, width, orientation):
        patch[r, c] = int(BRACKET_COLOR)

    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if patch[r, c] == 0
    ]
    marker_r, marker_c = random.choice(interior)
    patch[marker_r, marker_c] = int(target_color)
    return patch


def _place_rectangles(grid: np.ndarray, rect_specs: list[tuple[int, int, int]]) -> bool:
    for color, height, width in rect_specs:
        patch = np.full((height, width), int(color), dtype=int)
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=Color.BLACK,
                border=0,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return False
        paste(grid, patch, r0, c0, transparent=Color.BLACK)
    return True


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        bg = get_bg(grid)
        bracket, marker = _find_bracket_and_marker(grid, bg)
        rectangles = _rectangle_components(grid, bg, marker)
    except AssertionError:
        return False

    del bracket
    target_color = int(grid[marker])
    output = _apply_rule(grid, rectangles, target_color, bg)
    if np.array_equal(output, grid):
        return False
    if np.array_equal(output, _erase_only(grid, rectangles, target_color, bg)):
        return False

    # The marker choice must matter: changing it to another legal color must change the output.
    for color in TARGET_COLORS:
        if int(color) == target_color:
            continue
        alternate = _apply_rule(grid, rectangles, int(color), bg)
        if np.array_equal(output, alternate):
            return False

    # The bracket and marker must stay untouched.
    changed = output != grid
    for r, c in np.argwhere(grid == BRACKET_COLOR):
        if changed[int(r), int(c)]:
            return False
    if changed[marker]:
        return False

    # Non-target rectangles must really become hollow.
    survivors = [rect for rect in rectangles if int(rect["color"]) != target_color]
    if not survivors:
        return False
    for rect in survivors:
        r0, c0, r1, c1 = rect["bbox"]
        if not np.all(output[r0, c0 : c1 + 1] == int(rect["color"])):
            return False
        if not np.all(output[r1, c0 : c1 + 1] == int(rect["color"])):
            return False
        if not np.all(output[r0 : r1 + 1, c0] == int(rect["color"])):
            return False
        if not np.all(output[r0 : r1 + 1, c1] == int(rect["color"])):
            return False
        if not np.all(output[r0 + 1 : r1, c0 + 1 : c1] == int(bg)):
            return False

    return True


def _gen_attempt(call_index: int) -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((height, width), dtype=int)

    rect_count = random.randint(*RECT_COUNT_RANGE)
    counts = _random_composition(rect_count, len(TARGET_COLORS))
    color_counts = dict(zip(TARGET_COLORS, counts))
    target_color = int(TARGET_COLORS[call_index % len(TARGET_COLORS)])
    orientation = BRACKET_ORIENTATIONS[call_index % len(BRACKET_ORIENTATIONS)]

    rect_specs: list[tuple[int, int, int]] = []
    for color in TARGET_COLORS:
        for _ in range(color_counts[int(color)]):
            rect_specs.append(
                (
                    int(color),
                    random.choice(RECT_SIDE_CHOICES),
                    random.choice(RECT_SIDE_CHOICES),
                )
            )
    random.shuffle(rect_specs)
    if not _place_rectangles(grid, rect_specs):
        return None

    bracket_patch = _make_bracket_patch(target_color, orientation)
    try:
        r0, c0 = random_free_location(
            grid,
            bracket_patch,
            transparent=Color.BLACK,
            border=0,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return None
    paste(grid, bracket_patch, r0, c0, transparent=Color.BLACK)

    if not _semantic_ok(grid):
        return None
    if not np.array_equal(main(grid), _apply_rule(grid, _rectangle_components(grid, get_bg(grid), _find_bracket_and_marker(grid, get_bg(grid))[1]), target_color, get_bg(grid))):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid bracket-select erase-and-hollow instance")
