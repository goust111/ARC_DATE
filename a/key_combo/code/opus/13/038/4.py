from __future__ import annotations

import random
from collections import deque

import numpy as np


TOP_BG = 0
PANEL_BG = 5
CROSS_COLOR = 2
TOP_H = 2
SQUARE_COLORS = (1, 3, 4)
ODD_SIDES = (3, 5)
EVEN_SIDES = (2, 4)
TOTAL_SQUARE_RANGE = (6, 10)
PANEL_H_RANGE = (12, 18)
PANEL_W_RANGE = (12, 18)
MAX_GEN_ATTEMPTS = 600

PLUS_SHAPE = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})

_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            pixels: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                pixels.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pixels, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _is_plus(comp: dict) -> bool:
    if int(comp["color"]) != CROSS_COLOR or int(comp["area"]) != 5:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return False
    rel = {(int(r - r0), int(c - c0)) for r, c in comp["coords"]}
    return rel == PLUS_SHAPE


def _is_solid_square(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    return side_h == side_w and int(comp["area"]) == side_h * side_w


def _extract_structure(grid: np.ndarray) -> tuple[list[tuple[int, int]], dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > TOP_H and g.shape[1] >= 2, "expected a 2D ARC grid with top strip and panel"

    top = g[:TOP_H]
    markers = [(row, int(top[row, 0])) for row in range(TOP_H) if int(top[row, 0]) != TOP_BG]
    assert 1 <= len(markers) <= 2, "expected 1-2 marker pixels in the first column of the top strip"
    assert len({color for _row, color in markers}) == len(markers), "marker colors must be distinct"
    assert np.all(top[:, 1:] == TOP_BG), "top strip must only use column 0 in the input"

    panel = g[TOP_H:]
    components = _find_components(panel, bg=PANEL_BG)
    pluses = [comp for comp in components if _is_plus(comp)]
    assert len(pluses) == 1, "expected exactly one red plus marker in the panel"
    plus = pluses[0]

    squares: list[dict] = []
    for comp in components:
        if comp is plus:
            continue
        assert _is_solid_square(comp), "every non-plus component in the panel must be a solid square"
        squares.append(comp)

    assert TOTAL_SQUARE_RANGE[0] <= len(squares) <= TOTAL_SQUARE_RANGE[1], "unexpected square count"
    return markers, plus, squares


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter the marked-color squares, keep only their odd ones as center-marked counts, and erase the marked even ones.

    Input structure:
    - The top 2 rows are a black instruction strip whose first column contains 1-2 single-pixel marker colors.
    - The lower panel uses gray background and contains one unique red 3x3 plus marker plus several separated solid single-color squares.
    - Marker-colored squares and non-marker-colored squares both appear, and marked colors can have both odd and even side lengths.

    Rule:
    - Read the marker color set from the top-strip first column.
    - In the panel, treat every non-plus 4-connected monochromatic component as a solid square; non-marker colors remain unchanged.
    - For marker-colored squares only, recolor the center of each odd-sided square to the plus color red, and erase each even-sided square to gray background.
    - Replace each top marker pixel by a horizontal bar from column 0 whose length equals that marker color's surviving odd-square count; erase it if the count is zero.
    - Keep the red plus only when its center lies inside the bounding box of all newly marked centers; otherwise erase the plus to gray.

    Key insight:
    - The transformation is a nested filter: the top strip first selects square colors, then odd/even side length decides whether each selected square contributes a red center or is fully erased.
    """
    g = np.asarray(grid, dtype=int)
    markers, plus, squares = _extract_structure(g)
    marker_colors = {color for _row, color in markers}

    out = g.copy()
    out[:TOP_H] = TOP_BG

    counts = {color: 0 for color in marker_colors}
    marked_centers: list[tuple[int, int]] = []

    for square in squares:
        color = int(square["color"])
        r0, c0, r1, c1 = square["bbox"]
        side = int(r1 - r0 + 1)
        global_r0 = TOP_H + r0
        global_r1 = TOP_H + r1

        if color not in marker_colors:
            continue
        if side % 2 == 1:
            center = (TOP_H + r0 + side // 2, c0 + side // 2)
            out[center] = CROSS_COLOR
            marked_centers.append((r0 + side // 2, c0 + side // 2))
            counts[color] += 1
        else:
            out[global_r0 : global_r1 + 1, c0 : c1 + 1] = PANEL_BG

    plus_center = (int(plus["bbox"][0] + 1), int(plus["bbox"][1] + 1))
    keep_plus = False
    if marked_centers:
        rows = [r for r, _c in marked_centers]
        cols = [c for _r, c in marked_centers]
        keep_plus = min(rows) <= plus_center[0] <= max(rows) and min(cols) <= plus_center[1] <= max(cols)
    if not keep_plus:
        for r, c in plus["coords"]:
            out[TOP_H + int(r), int(c)] = PANEL_BG

    for row, color in markers:
        count = counts[color]
        if count > 0:
            out[row, :count] = color

    return out


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _can_place(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], panel_h: int, panel_w: int) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= panel_h or c1 >= panel_w:
        return False
    return all(_separated(box, other, gap=1) for other in placed)


def _choice(seq):
    return random.choice(tuple(seq))


def _build_grid(
    panel_h: int,
    panel_w: int,
    markers: list[tuple[int, int]],
    squares: list[dict],
    plus_center: tuple[int, int],
) -> np.ndarray:
    grid = np.full((TOP_H + panel_h, panel_w), PANEL_BG, dtype=int)
    grid[:TOP_H] = TOP_BG

    for row, color in markers:
        grid[row, 0] = int(color)

    for square in squares:
        r0 = int(square["r0"])
        c0 = int(square["c0"])
        side = int(square["side"])
        grid[TOP_H + r0 : TOP_H + r0 + side, c0 : c0 + side] = int(square["color"])

    pr, pc = plus_center
    plus_pixels = [(pr - 1, pc), (pr, pc - 1), (pr, pc), (pr, pc + 1), (pr + 1, pc)]
    for rr, cc in plus_pixels:
        grid[TOP_H + rr, cc] = CROSS_COLOR

    return grid


def _expected_from_spec(
    panel_h: int,
    panel_w: int,
    markers: list[tuple[int, int]],
    squares: list[dict],
    plus_center: tuple[int, int],
) -> np.ndarray:
    grid = _build_grid(panel_h, panel_w, markers, squares, plus_center)
    out = grid.copy()
    out[:TOP_H] = TOP_BG

    marker_colors = {color for _row, color in markers}
    counts = {color: 0 for color in marker_colors}
    odd_centers: list[tuple[int, int]] = []

    for square in squares:
        r0 = int(square["r0"])
        c0 = int(square["c0"])
        side = int(square["side"])
        color = int(square["color"])
        if color not in marker_colors:
            continue
        if side % 2 == 1:
            rr = r0 + side // 2
            cc = c0 + side // 2
            out[TOP_H + rr, cc] = CROSS_COLOR
            odd_centers.append((rr, cc))
            counts[color] += 1
        else:
            out[TOP_H + r0 : TOP_H + r0 + side, c0 : c0 + side] = PANEL_BG

    if odd_centers:
        rows = [r for r, _c in odd_centers]
        cols = [c for _r, c in odd_centers]
        keep_plus = min(rows) <= plus_center[0] <= max(rows) and min(cols) <= plus_center[1] <= max(cols)
    else:
        keep_plus = False

    if not keep_plus:
        pr, pc = plus_center
        for rr, cc in ((pr - 1, pc), (pr, pc - 1), (pr, pc), (pr, pc + 1), (pr + 1, pc)):
            out[TOP_H + rr, cc] = PANEL_BG

    for row, color in markers:
        count = counts[color]
        if count > 0:
            out[row, :count] = color

    return out


def _choose_sizes(marker_colors: list[int], nonmarker_colors: list[int], target_total: int) -> list[dict]:
    specs: list[dict] = []

    for color in marker_colors:
        specs.append({"color": color, "side": _choice(ODD_SIDES)})
        specs.append({"color": color, "side": _choice(EVEN_SIDES)})

    if len(marker_colors) == 1:
        specs.append({"color": marker_colors[0], "side": _choice(ODD_SIDES)})

    for color in nonmarker_colors:
        pool = ODD_SIDES + EVEN_SIDES
        specs.append({"color": color, "side": _choice(pool)})

    while len(specs) < target_total:
        color = _choice(SQUARE_COLORS)
        if color in marker_colors:
            side = _choice(ODD_SIDES + EVEN_SIDES)
        else:
            side = _choice(ODD_SIDES + EVEN_SIDES)
        specs.append({"color": color, "side": side})

    random.shuffle(specs)
    return specs


def _place_squares(panel_h: int, panel_w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[tuple[int, int, int, int]] = []
    squares: list[dict] = []

    for spec in sorted(specs, key=lambda item: int(item["side"]), reverse=True):
        side = int(spec["side"])
        candidates = [
            (r0, c0)
            for r0 in range(panel_h - side + 1)
            for c0 in range(panel_w - side + 1)
            if _can_place((r0, c0, r0 + side - 1, c0 + side - 1), placed, panel_h, panel_w)
        ]
        if not candidates:
            return None
        r0, c0 = _choice(candidates)
        box = (r0, c0, r0 + side - 1, c0 + side - 1)
        placed.append(box)
        squares.append({"color": int(spec["color"]), "side": side, "r0": r0, "c0": c0})

    return squares


def _choose_plus_center(
    panel_h: int,
    panel_w: int,
    squares: list[dict],
    marker_colors: set[int],
    keep_inside_bbox: bool,
) -> tuple[int, int] | None:
    odd_centers = [
        (int(square["r0"] + square["side"] // 2), int(square["c0"] + square["side"] // 2))
        for square in squares
        if int(square["color"]) in marker_colors and int(square["side"]) % 2 == 1
    ]
    if not odd_centers:
        return None

    rows = [r for r, _c in odd_centers]
    cols = [c for _r, c in odd_centers]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    placed = [
        (int(square["r0"]), int(square["c0"]), int(square["r0"] + square["side"] - 1), int(square["c0"] + square["side"] - 1))
        for square in squares
    ]

    candidates: list[tuple[int, int]] = []
    for pr in range(1, panel_h - 1):
        for pc in range(1, panel_w - 1):
            inside = bbox[0] <= pr <= bbox[2] and bbox[1] <= pc <= bbox[3]
            if inside != keep_inside_bbox:
                continue
            plus_box = (pr - 1, pc - 1, pr + 1, pc + 1)
            if _can_place(plus_box, placed, panel_h, panel_w):
                candidates.append((pr, pc))

    if not candidates:
        return None
    return _choice(candidates)


def _validate_generated_case(
    grid: np.ndarray,
    panel_h: int,
    panel_w: int,
    markers: list[tuple[int, int]],
    squares: list[dict],
    plus_center: tuple[int, int],
    expect_plus_kept: bool,
) -> bool:
    try:
        parsed_markers, plus, parsed_squares = _extract_structure(grid)
        if parsed_markers != markers:
            return False
        if len(parsed_squares) != len(squares):
            return False

        marker_colors = {color for _row, color in markers}
        nonmarker_colors = {int(square["color"]) for square in squares if int(square["color"]) not in marker_colors}
        if not nonmarker_colors:
            return False

        for color in marker_colors:
            sides = [int(square["side"]) for square in squares if int(square["color"]) == color]
            if not any(side % 2 == 1 for side in sides):
                return False
            if not any(side % 2 == 0 for side in sides):
                return False

        expected = _expected_from_spec(panel_h, panel_w, markers, squares, plus_center)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False

        plus_center_from_parse = (int(plus["bbox"][0] + 1), int(plus["bbox"][1] + 1))
        odd_centers = [
            (int(square["r0"] + square["side"] // 2), int(square["c0"] + square["side"] // 2))
            for square in squares
            if int(square["color"]) in marker_colors and int(square["side"]) % 2 == 1
        ]
        rows = [r for r, _c in odd_centers]
        cols = [c for _r, c in odd_centers]
        observed_keep = min(rows) <= plus_center_from_parse[0] <= max(rows) and min(cols) <= plus_center_from_parse[1] <= max(cols)
        if observed_keep != expect_plus_kept:
            return False

        for row, color in markers:
            expected_len = sum(
                1 for square in squares if int(square["color"]) == color and int(square["side"]) % 2 == 1
            )
            if not np.all(actual[row, :expected_len] == color):
                return False
            if not np.all(actual[row, expected_len:] == TOP_BG):
                return False

        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    marker_count = _choice((1, 2))
    marker_colors = random.sample(list(SQUARE_COLORS), marker_count)
    nonmarker_colors = [color for color in SQUARE_COLORS if color not in marker_colors]
    target_total = random.randint(*TOTAL_SQUARE_RANGE)

    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    specs = _choose_sizes(marker_colors, nonmarker_colors, target_total)
    squares = _place_squares(panel_h, panel_w, specs)
    if squares is None:
        return None

    odd_marker_centers = [
        (int(square["r0"] + square["side"] // 2), int(square["c0"] + square["side"] // 2))
        for square in squares
        if int(square["color"]) in set(marker_colors) and int(square["side"]) % 2 == 1
    ]
    if len(odd_marker_centers) < 2:
        return None

    keep_plus = bool(random.getrandbits(1))
    plus_center = _choose_plus_center(panel_h, panel_w, squares, set(marker_colors), keep_plus)
    if plus_center is None:
        return None

    marker_rows = [0, 1] if marker_count == 2 else [_choice((0, 1))]
    markers = sorted(zip(marker_rows, marker_colors))
    grid = _build_grid(panel_h, panel_w, markers, squares, plus_center)

    if not _validate_generated_case(grid, panel_h, panel_w, markers, squares, plus_center, keep_plus):
        return None

    out = main(grid)
    in_sig = (tuple(map(int, grid.shape)), grid.tobytes())
    out_sig = (tuple(map(int, out.shape)), out.tobytes())
    if in_sig in _SEEN_INPUTS or out_sig in _SEEN_OUTPUTS:
        return None

    _SEEN_INPUTS.add(in_sig)
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
