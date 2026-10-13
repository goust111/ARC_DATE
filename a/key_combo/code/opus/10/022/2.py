from __future__ import annotations

import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

Coord = tuple[int, int]


def _grid_signature(grid: np.ndarray) -> str:
    g = np.asarray(grid, dtype=int)
    shape = f"{g.shape[0]}x{g.shape[1]}".encode()
    return hashlib.sha1(g.tobytes() + shape).hexdigest()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_color_components(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            out.append(coords)
    return out


def _bbox_orientation(coords: list[Coord]) -> str:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    assert width != height, "red voter bbox must have a unique long axis"
    return "horizontal" if width > height else "vertical"


def _ordered_pair(coords: np.ndarray) -> tuple[Coord, Coord]:
    pts = sorted((int(r), int(c)) for r, c in coords.tolist())
    assert len(pts) == 2, "expected exactly two same-color markers"
    start, end = pts
    assert start != end
    assert start[0] != end[0] and start[1] != end[1], "same-color markers must require a true L route"
    return start, end


def _find_pair(grid: np.ndarray, color: int) -> tuple[Coord, Coord]:
    return _ordered_pair(np.argwhere(grid == int(color)))


def _l_path(start: Coord, end: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    assert r0 != r1 and c0 != c1
    assert first_axis in {"horizontal", "vertical"}
    path = [start]
    if first_axis == "horizontal":
        c_step = 1 if c1 > c0 else -1
        for c in range(c0 + c_step, c1 + c_step, c_step):
            path.append((r0, c))
        r_step = 1 if r1 > r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            path.append((r, c1))
    else:
        r_step = 1 if r1 > r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            path.append((r, c0))
        c_step = 1 if c1 > c0 else -1
        for c in range(c0 + c_step, c1 + c_step, c_step):
            path.append((r1, c))
    return path


def _assert_route_clear(grid: np.ndarray, path: list[Coord], color: int) -> None:
    assert int(grid[path[0]]) == int(color)
    assert int(grid[path[-1]]) == int(color)
    for r, c in path[1:-1]:
        assert int(grid[r, c]) == BLACK, "routes may only pass through background between endpoints"


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREEN}

    green_pair = _find_pair(g, GREEN)
    blue_pair = _find_pair(g, BLUE)
    markers = set(green_pair + blue_pair)
    assert len(markers) == 4, "all four markers must be distinct"

    red_components = _find_color_components(g, RED)
    assert 3 <= len(red_components) <= 5, "expected 3-5 red voter objects"

    marker_rows = {r for r, _ in markers}
    marker_cols = {c for _, c in markers}
    horizontal_count = 0
    vertical_count = 0
    for comp in red_components:
        assert 3 <= len(comp) <= 6, "each red voter must have 3-6 pixels"
        for r, c in comp:
            assert r not in marker_rows and c not in marker_cols, "red voters must not share rows or columns with markers"
        orientation = _bbox_orientation(comp)
        if orientation == "horizontal":
            horizontal_count += 1
        else:
            vertical_count += 1

    assert horizontal_count != vertical_count, "red voter majority must be unique"
    return {
        "grid": g,
        "green_pair": green_pair,
        "blue_pair": blue_pair,
        "horizontal_count": horizontal_count,
        "vertical_count": vertical_count,
    }


def _render_from_scene(scene: dict, majority_horizontal: bool) -> tuple[np.ndarray, list[Coord], list[Coord]]:
    g = scene["grid"]
    green_order = "horizontal" if majority_horizontal else "vertical"
    blue_order = "vertical" if majority_horizontal else "horizontal"

    green_path = _l_path(scene["green_pair"][0], scene["green_pair"][1], green_order)
    blue_path = _l_path(scene["blue_pair"][0], scene["blue_pair"][1], blue_order)
    _assert_route_clear(g, green_path, GREEN)
    _assert_route_clear(g, blue_path, BLUE)

    out = g.copy()
    for r, c in green_path:
        out[r, c] = GREEN
    for r, c in blue_path:
        out[r, c] = BLUE
    out[out == RED] = BLACK
    return out, green_path, blue_path


def main(grid: np.ndarray) -> np.ndarray:
    """Use red-object orientation counts to choose opposite L-routing orders for the green and blue marker pairs.

    Input structure:
    - The grid has a black background, 3-5 red 4-connected voter objects, and exactly two green plus two blue single-pixel markers.
    - Each red object has 3-6 pixels and a non-square bbox whose longer side defines its orientation.
    - Within each same-color marker pair, the two markers are in different rows and different columns, so either L turn order is geometrically possible.

    Rule:
    - Count how many red objects have horizontal bboxes and how many have vertical bboxes.
    - If horizontal voters are the majority, connect the green pair with the lexicographically earlier marker as the start using a horizontal-then-vertical L path, and connect the blue pair using vertical-then-horizontal; if vertical voters are the majority, swap those two routing orders.
    - Each route includes both endpoints, uses only background cells between endpoints, green is drawn before blue so blue wins at crossings, and all red voter pixels are erased afterward.

    Key insight:
    - The red objects never appear in the output; they only vote for which of the two valid L routes each colored pair must take.
    """
    scene = _analyze_grid(grid)
    out, _green_path, _blue_path = _render_from_scene(
        scene,
        majority_horizontal=scene["horizontal_count"] > scene["vertical_count"],
    )
    return out


_HORIZONTAL_BASE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 2)),
)
_PAIRINGS = (
    ((0, 1), (2, 3)),
    ((0, 2), (1, 3)),
    ((0, 3), (1, 2)),
)
_GEN_OUTPUT_SIGS: set[str] = set()
_GEN_INPUT_SIGS: set[str] = set()
_GEN_MAX_ATTEMPTS = 500


def _normalize_shape(cells: list[Coord]) -> tuple[Coord, ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape_variants(base: tuple[Coord, ...]) -> list[tuple[Coord, ...]]:
    normalized = _normalize_shape(list(base))
    height = max(r for r, _ in normalized) + 1
    width = max(c for _, c in normalized) + 1
    variants = set()
    for flip_r in (False, True):
        for flip_c in (False, True):
            cells = []
            for r, c in normalized:
                rr = height - 1 - r if flip_r else r
                cc = width - 1 - c if flip_c else c
                cells.append((rr, cc))
            variants.add(_normalize_shape(cells))
    return sorted(variants)


def _random_shape(orientation: str) -> tuple[Coord, ...]:
    base = list(random.choice(_HORIZONTAL_BASE_SHAPES))
    if orientation == "vertical":
        base = [(c, r) for r, c in base]
    return random.choice(_shape_variants(tuple(base)))


def _chebyshev_gap(cells_a: set[Coord], cells_b: set[Coord]) -> int:
    return min(max(abs(r1 - r2), abs(c1 - c2)) for r1, c1 in cells_a for r2, c2 in cells_b)


def _marker_path_clear(path: list[Coord], markers: set[Coord], endpoints: tuple[Coord, Coord]) -> bool:
    endpoint_set = set(endpoints)
    for cell in path[1:-1]:
        if cell in markers:
            return False
    return path[0] in endpoint_set and path[-1] in endpoint_set


def _render_marker_only_scene(
    shape: tuple[int, int],
    green_pair: tuple[Coord, Coord],
    blue_pair: tuple[Coord, Coord],
    majority_horizontal: bool,
) -> tuple[np.ndarray, list[Coord], list[Coord]]:
    grid = np.zeros(shape, dtype=int)
    for r, c in green_pair:
        grid[r, c] = GREEN
    for r, c in blue_pair:
        grid[r, c] = BLUE
    scene = {
        "grid": grid,
        "green_pair": green_pair,
        "blue_pair": blue_pair,
    }
    return _render_from_scene(scene, majority_horizontal=majority_horizontal)


def _choose_marker_layout(
    height: int,
    width: int,
    majority_horizontal: bool,
    want_crossing: bool,
) -> tuple[tuple[Coord, Coord], tuple[Coord, Coord]] | None:
    candidate_cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(1200):
        pts = random.sample(candidate_cells, 4)
        if len({r for r, _ in pts}) != 4 or len({c for _, c in pts}) != 4:
            continue
        if min(_chebyshev_gap({pts[i]}, {pts[j]}) for i in range(4) for j in range(i + 1, 4)) <= 2:
            continue
        for pairing in _PAIRINGS:
            for swap in (False, True):
                first_pair = (pts[pairing[0][0]], pts[pairing[0][1]])
                second_pair = (pts[pairing[1][0]], pts[pairing[1][1]])
                green_pair = tuple(sorted(second_pair if swap else first_pair))
                blue_pair = tuple(sorted(first_pair if swap else second_pair))
                if green_pair[0][0] == green_pair[1][0] or green_pair[0][1] == green_pair[1][1]:
                    continue
                if blue_pair[0][0] == blue_pair[1][0] or blue_pair[0][1] == blue_pair[1][1]:
                    continue
                if min(abs(green_pair[0][0] - green_pair[1][0]), abs(green_pair[0][1] - green_pair[1][1])) < 2:
                    continue
                if min(abs(blue_pair[0][0] - blue_pair[1][0]), abs(blue_pair[0][1] - blue_pair[1][1])) < 2:
                    continue

                markers = set(green_pair + blue_pair)
                green_h = _l_path(green_pair[0], green_pair[1], "horizontal")
                green_v = _l_path(green_pair[0], green_pair[1], "vertical")
                blue_h = _l_path(blue_pair[0], blue_pair[1], "horizontal")
                blue_v = _l_path(blue_pair[0], blue_pair[1], "vertical")
                if set(green_h) == set(green_v) or set(blue_h) == set(blue_v):
                    continue
                if not _marker_path_clear(green_h, markers, green_pair):
                    continue
                if not _marker_path_clear(green_v, markers, green_pair):
                    continue
                if not _marker_path_clear(blue_h, markers, blue_pair):
                    continue
                if not _marker_path_clear(blue_v, markers, blue_pair):
                    continue

                actual_out, actual_green, actual_blue = _render_marker_only_scene(
                    (height, width), green_pair, blue_pair, majority_horizontal
                )
                alt_out, _alt_green, _alt_blue = _render_marker_only_scene(
                    (height, width), green_pair, blue_pair, not majority_horizontal
                )
                if np.array_equal(actual_out, alt_out):
                    continue

                intersection = set(actual_green) & set(actual_blue)
                if len(intersection) > 1:
                    continue
                if want_crossing and len(intersection) != 1:
                    continue
                if (not want_crossing) and intersection:
                    continue
                return green_pair, blue_pair
    return None


def _choose_orientation_counts(majority_horizontal: bool) -> list[str]:
    horizontal_count, vertical_count = random.choice(
        [(2, 1), (3, 1), (3, 2), (4, 1)]
    )
    labels = ["horizontal"] * horizontal_count + ["vertical"] * vertical_count
    if not majority_horizontal:
        labels = ["vertical" if x == "horizontal" else "horizontal" for x in labels]
    random.shuffle(labels)
    return labels


def _place_red_objects(
    shape: tuple[int, int],
    green_pair: tuple[Coord, Coord],
    blue_pair: tuple[Coord, Coord],
    orientations: list[str],
) -> list[set[Coord]] | None:
    height, width = shape
    markers = set(green_pair + blue_pair)
    marker_rows = {r for r, _ in markers}
    marker_cols = {c for _, c in markers}
    occupied_groups = [{cell} for cell in markers]
    placed: list[set[Coord]] = []

    for orientation in orientations:
        chosen: set[Coord] | None = None
        for _ in range(120):
            patch = _random_shape(orientation)
            patch_h = max(r for r, _ in patch) + 1
            patch_w = max(c for _, c in patch) + 1
            candidates: list[set[Coord]] = []
            for r0 in range(height - patch_h + 1):
                for c0 in range(width - patch_w + 1):
                    cells = {(r0 + r, c0 + c) for r, c in patch}
                    if any(r in marker_rows or c in marker_cols for r, c in cells):
                        continue
                    if any(_chebyshev_gap(cells, other) <= 2 for other in occupied_groups):
                        continue
                    candidates.append(cells)
            if candidates:
                chosen = random.choice(candidates)
                break
        if chosen is None:
            return None
        placed.append(chosen)
        occupied_groups.append(chosen)
    return placed


def _build_grid(
    shape: tuple[int, int],
    green_pair: tuple[Coord, Coord],
    blue_pair: tuple[Coord, Coord],
    red_objects: list[set[Coord]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in red_objects:
        for r, c in obj:
            grid[r, c] = RED
    for r, c in green_pair:
        grid[r, c] = GREEN
    for r, c in blue_pair:
        grid[r, c] = BLUE
    return grid


def _semantic_validate(grid: np.ndarray) -> np.ndarray:
    scene = _analyze_grid(grid)
    actual = main(grid)
    expected, green_path, blue_path = _render_from_scene(
        scene,
        majority_horizontal=scene["horizontal_count"] > scene["vertical_count"],
    )
    assert np.array_equal(actual, expected)

    green_h = _l_path(scene["green_pair"][0], scene["green_pair"][1], "horizontal")
    green_v = _l_path(scene["green_pair"][0], scene["green_pair"][1], "vertical")
    blue_h = _l_path(scene["blue_pair"][0], scene["blue_pair"][1], "horizontal")
    blue_v = _l_path(scene["blue_pair"][0], scene["blue_pair"][1], "vertical")
    assert set(green_h) != set(green_v)
    assert set(blue_h) != set(blue_v)
    _assert_route_clear(grid, green_h, GREEN)
    _assert_route_clear(grid, green_v, GREEN)
    _assert_route_clear(grid, blue_h, BLUE)
    _assert_route_clear(grid, blue_v, BLUE)

    counterfactual, _g2, _b2 = _render_from_scene(
        scene,
        majority_horizontal=not (scene["horizontal_count"] > scene["vertical_count"]),
    )
    assert not np.array_equal(actual, counterfactual), "majority vote must change the output"
    assert not np.any(actual == RED), "all red voters must be erased"

    overlap = set(green_path) & set(blue_path)
    for r, c in overlap:
        assert int(actual[r, c]) == BLUE, "blue must overpaint green at crossings"
    return actual


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 16)
    width = random.randint(12, 16)
    majority_horizontal = random.random() < 0.5
    want_crossing = random.random() < 0.6

    pairs = _choose_marker_layout(height, width, majority_horizontal, want_crossing)
    if pairs is None:
        return None
    green_pair, blue_pair = pairs

    orientations = _choose_orientation_counts(majority_horizontal)
    red_objects = _place_red_objects((height, width), green_pair, blue_pair, orientations)
    if red_objects is None:
        return None

    grid = _build_grid((height, width), green_pair, blue_pair, red_objects)
    output = _semantic_validate(grid)

    input_sig = _grid_signature(grid)
    output_sig = _grid_signature(output)
    if input_sig in _GEN_INPUT_SIGS or output_sig in _GEN_OUTPUT_SIGS:
        return None
    _GEN_INPUT_SIGS.add(input_sig)
    _GEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orientation-routing ARC input")
