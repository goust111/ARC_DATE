from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_TOOLBOX = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX) not in sys.path:
    sys.path.append(str(_TOOLBOX))

from arc_utils import Color, bbox_coords, components_mask, find_components


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GRID_HEIGHT_RANGE = (15, 19)
GRID_WIDTH_RANGE = (15, 19)
PAIR_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
PATCH_GAP = 1
PAIR_DIMS = ((3, 3), (3, 4), (4, 3))


def _bbox_center(coords: np.ndarray) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox_coords(coords)
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _shift_coords(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    moved = coords.copy()
    moved[:, 0] += int(dr)
    moved[:, 1] += int(dc)
    return moved


def _coords_to_set(coords: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in coords}


def _set_to_coords(cells: set[Coord]) -> np.ndarray:
    return np.array(sorted(cells), dtype=int)


def _direction_and_order(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    ra, ca = _bbox_center(comp_a)
    rb, cb = _bbox_center(comp_b)
    dr = rb - ra
    dc = cb - ca
    if abs(dc) >= abs(dr):
        if ca <= cb:
            return "h", comp_a, comp_b
        return "h", comp_b, comp_a
    if ra <= rb:
        return "v", comp_a, comp_b
    return "v", comp_b, comp_a


def _moved_pair(comp_a: np.ndarray, comp_b: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    orientation, first, second = _direction_and_order(comp_a, comp_b)
    if orientation == "h":
        return orientation, _shift_coords(first, 0, 1), _shift_coords(second, 0, -1)
    return orientation, _shift_coords(first, 1, 0), _shift_coords(second, -1, 0)


def _filled_rectangle_bbox(cells: set[Coord]) -> BBox | None:
    if not cells:
        return None
    coords = _set_to_coords(cells)
    r0, c0, r1, c1 = bbox_coords(coords)
    if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
        return None
    return r0, c0, r1, c1


def _paint_classified_rectangle(out: np.ndarray, bbox: BBox, base_color: int) -> None:
    r0, c0, r1, c1 = bbox
    corner_color = Color.PINK if int(base_color) == Color.BLUE else Color.BLUE
    interior_color = Color.ORANGE if int(base_color) == Color.RED else Color.RED
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_top_bottom = r in (r0, r1)
            on_left_right = c in (c0, c1)
            if on_top_bottom and on_left_right:
                out[r, c] = int(corner_color)
            elif on_top_bottom or on_left_right:
                out[r, c] = int(Color.YELLOW)
            else:
                out[r, c] = int(interior_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-colored fragment pair one step toward each other only when the shifted union becomes a solid rectangle.

    Input structure:
    - The grid has black background and 2 or 3 foreground colors.
    - Each foreground color appears in exactly two 4-connected fragments, with the two fragments roughly aligned horizontally or vertically and separated by a one-cell gap along that axis.
    - Different colors represent independent fragment pairs.

    Rule:
    - For each color, identify its two fragments and determine whether their relative direction is horizontal or vertical from the fragment bbox centers.
    - Shift the two fragments one cell toward each other along that axis, then test the shifted union.
    - If the shifted union is exactly one filled axis-aligned rectangle, erase the original fragments and recolor that rectangle: corners blue unless the base color is blue, in which case corners become magenta; non-corner border cells yellow; strict interior red unless the base color is red, in which case the interior becomes orange.
    - If the shifted union is not a filled rectangle, leave that color's original fragments unchanged.

    Key insight:
    - The translation is only a conditional probe: it determines which fragment pairs qualify for the rectangle recoloring, so direction, movement, and rectangle validation all matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = Color.BLACK
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    by_color: dict[int, list[np.ndarray]] = {}
    for comp in comps:
        color = int(comp["color"])
        by_color.setdefault(color, []).append(comp["coords"])

    assert 2 <= len(by_color) <= 3
    out = g.copy()
    for color, pair in sorted(by_color.items()):
        assert len(pair) == 2
        orientation, moved_a, moved_b = _moved_pair(pair[0], pair[1])
        del orientation
        union = _coords_to_set(moved_a) | _coords_to_set(moved_b)
        rect_bbox = _filled_rectangle_bbox(union)
        if rect_bbox is None:
            continue
        for coords in pair:
            out[coords[:, 0], coords[:, 1]] = bg
        _paint_classified_rectangle(out, rect_bbox, color)
    return out


def _connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    mask = np.zeros((max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4)) == 1


def _build_rect_parts(
    orientation: str,
    h: int,
    w: int,
    seam: int,
    states: tuple[int, ...],
) -> tuple[set[Coord], set[Coord]]:
    first: set[Coord] = set()
    second: set[Coord] = set()
    if orientation == "h":
        for r in range(h):
            for c in range(seam):
                first.add((r, c))
            for c in range(seam + 1, w):
                second.add((r, c))
            if states[r] in (0, 2):
                first.add((r, seam))
            if states[r] in (1, 2):
                second.add((r, seam))
    else:
        for c in range(w):
            for r in range(seam):
                first.add((r, c))
            for r in range(seam + 1, h):
                second.add((r, c))
            if states[c] in (0, 2):
                first.add((seam, c))
            if states[c] in (1, 2):
                second.add((seam, c))
    return first, second


def _candidate_patterns(success: bool, orientation: str) -> list[dict]:
    patterns: list[dict] = []
    state_space = (0, 1, 2) if success else (0, 1, 2, 3)
    for h, w in PAIR_DIMS:
        seam_range = range(1, w - 1) if orientation == "h" else range(1, h - 1)
        line_len = h if orientation == "h" else w
        for seam in seam_range:
            for states in itertools.product(state_space, repeat=line_len):
                if not any(s in (0, 2) for s in states):
                    continue
                if not any(s in (1, 2) for s in states):
                    continue
                if success:
                    if any(s == 3 for s in states):
                        continue
                else:
                    if 3 not in states:
                        continue
                first, second = _build_rect_parts(orientation, h, w, seam, states)
                if not (3 <= len(first) <= 8 and 3 <= len(second) <= 8):
                    continue
                if not (_connected(first) and _connected(second)):
                    continue
                full = len(first | second) == h * w
                if full != success:
                    continue
                missing = h * w - len(first | second)
                if not success and missing > 2:
                    continue
                patterns.append(
                    {
                        "orientation": orientation,
                        "rect_h": h,
                        "rect_w": w,
                        "first_target": first,
                        "second_target": second,
                    }
                )
    return patterns


SUCCESS_PATTERNS = {
    "h": _candidate_patterns(True, "h"),
    "v": _candidate_patterns(True, "v"),
}
FAIL_PATTERNS = {
    "h": _candidate_patterns(False, "h"),
    "v": _candidate_patterns(False, "v"),
}


def _localize_pair(pattern: dict, color: int, success: bool) -> dict:
    h = int(pattern["rect_h"])
    w = int(pattern["rect_w"])
    first_target = set(pattern["first_target"])
    second_target = set(pattern["second_target"])
    orientation = pattern["orientation"]
    if orientation == "h":
        patch = np.zeros((h, w + 2), dtype=int)
        first_orig = {(r, c) for r, c in first_target}
        second_orig = {(r, c + 2) for r, c in second_target}
        rect_bbox = (0, 1, h - 1, w)
    else:
        patch = np.zeros((h + 2, w), dtype=int)
        first_orig = {(r, c) for r, c in first_target}
        second_orig = {(r + 2, c) for r, c in second_target}
        rect_bbox = (1, 0, h, w - 1)
    for r, c in first_orig | second_orig:
        patch[r, c] = int(color)
    return {
        "patch": patch,
        "orientation": orientation,
        "success": bool(success),
        "color": int(color),
        "local_components": [first_orig, second_orig],
        "local_rect_bbox": rect_bbox,
    }


def _choose_colors(pair_count: int) -> list[int]:
    if pair_count == 3:
        third = random.choice((Color.GREEN, Color.TEAL, Color.MAROON))
        colors = [Color.BLUE, Color.RED, third]
        random.shuffle(colors)
        return colors
    palettes = (
        (Color.BLUE, Color.GREEN),
        (Color.RED, Color.TEAL),
        (Color.BLUE, Color.RED),
        (Color.BLUE, Color.MAROON),
        (Color.RED, Color.GREEN),
        (Color.GREEN, Color.TEAL),
    )
    return list(random.choice(palettes))


def _placeable_positions(shape: tuple[int, int], patch_shape: tuple[int, int], occupied: np.ndarray) -> list[Coord]:
    h, w = shape
    ph, pw = patch_shape
    positions: list[Coord] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            rr0 = max(0, r0 - PATCH_GAP)
            cc0 = max(0, c0 - PATCH_GAP)
            rr1 = min(h, r0 + ph + PATCH_GAP)
            cc1 = min(w, c0 + pw + PATCH_GAP)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            positions.append((r0, c0))
    return positions


def _translate_cells(cells: set[Coord], dr: int, dc: int) -> set[Coord]:
    return {(r + dr, c + dc) for r, c in cells}


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        if not spec["success"]:
            continue
        for comp_cells in spec["components"]:
            for r, c in comp_cells:
                out[r, c] = Color.BLACK
        _paint_classified_rectangle(out, spec["rect_bbox"], spec["color"])
    return out


def _bbox_gap(spec: dict) -> int:
    a = _set_to_coords(spec["components"][0])
    b = _set_to_coords(spec["components"][1])
    orientation, first, second = _direction_and_order(a, b)
    if orientation == "h":
        fa = bbox_coords(first)
        fb = bbox_coords(second)
        return int(fb[1] - fa[3] - 1)
    fa = bbox_coords(first)
    fb = bbox_coords(second)
    return int(fb[0] - fa[2] - 1)


def _validate_semantics(inp: np.ndarray, specs: list[dict]) -> bool:
    if not (inp.ndim == 2 and max(inp.shape) <= 30):
        return False
    if int(np.count_nonzero(inp == Color.BLACK)) <= inp.size // 2:
        return False
    if sum(not spec["success"] for spec in specs) != 1:
        return False

    expected = _render_expected(inp, specs)
    solved = main(inp)
    if not np.array_equal(expected, solved):
        return False

    comps = find_components(inp, bg=Color.BLACK, connectivity=4, monochromatic=True)
    by_color: dict[int, list[np.ndarray]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp["coords"])

    if len(by_color) != len(specs):
        return False

    for spec in specs:
        color = int(spec["color"])
        pair = by_color.get(color)
        if pair is None or len(pair) != 2:
            return False
        actual_components = [_coords_to_set(coords) for coords in pair]
        expected_components = [set(comp) for comp in spec["components"]]
        if {frozenset(c) for c in actual_components} != {frozenset(c) for c in expected_components}:
            return False
        if any(not (3 <= len(comp) <= 8) for comp in spec["components"]):
            return False
        if _bbox_gap(spec) != 1:
            return False
        a = _set_to_coords(spec["components"][0])
        b = _set_to_coords(spec["components"][1])
        orientation, moved_a, moved_b = _moved_pair(a, b)
        if orientation != spec["orientation"]:
            return False
        rect_bbox = _filled_rectangle_bbox(_coords_to_set(moved_a) | _coords_to_set(moved_b))
        if spec["success"] != (rect_bbox is not None):
            return False
        if spec["success"] and rect_bbox != spec["rect_bbox"]:
            return False
        if not spec["success"]:
            for cells in spec["components"]:
                for r, c in cells:
                    if solved[r, c] != color:
                        return False
    return True


def _gen_orientations(pair_count: int) -> list[str]:
    choices = [random.choice(("h", "v")) for _ in range(pair_count)]
    if pair_count >= 2 and len(set(choices)) == 1 and random.random() < 0.75:
        choices[random.randrange(pair_count)] = "v" if choices[0] == "h" else "h"
    return choices


def _build_specs(shape: tuple[int, int], pairs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    placed_specs: list[dict] = []
    for pair in pairs:
        patch = pair["patch"]
        positions = _placeable_positions(shape, patch.shape, occupied)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        ph, pw = patch.shape
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        sub[patch != 0] = patch[patch != 0]
        rr0 = max(0, r0 - PATCH_GAP)
        cc0 = max(0, c0 - PATCH_GAP)
        rr1 = min(h, r0 + ph + PATCH_GAP)
        cc1 = min(w, c0 + pw + PATCH_GAP)
        occupied[rr0:rr1, cc0:cc1] = True

        global_components = [_translate_cells(cells, r0, c0) for cells in pair["local_components"]]
        rb = pair["local_rect_bbox"]
        rect_bbox = (rb[0] + r0, rb[1] + c0, rb[2] + r0, rb[3] + c0)
        placed_specs.append(
            {
                "color": pair["color"],
                "success": pair["success"],
                "orientation": pair["orientation"],
                "components": global_components,
                "rect_bbox": rect_bbox,
            }
        )
    return grid, placed_specs


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    colors = _choose_colors(pair_count)
    success_flags = [True] * (pair_count - 1) + [False]
    random.shuffle(success_flags)
    orientations = _gen_orientations(pair_count)

    pairs: list[dict] = []
    for color, success, orientation in zip(colors, success_flags, orientations):
        source = SUCCESS_PATTERNS if success else FAIL_PATTERNS
        pattern = random.choice(source[orientation])
        pairs.append(_localize_pair(pattern, color, success))

    for _ in range(40):
        height = random.randint(*GRID_HEIGHT_RANGE)
        width = random.randint(*GRID_WIDTH_RANGE)
        built = _build_specs((height, width), pairs)
        if built is None:
            continue
        inp, specs = built
        if _validate_semantics(inp, specs):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
