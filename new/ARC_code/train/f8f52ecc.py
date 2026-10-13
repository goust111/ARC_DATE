from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg
import random

Coord = tuple[int, int]


def _path_cells(start: Coord, stop: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    pts: list[Coord] = []

    if first_axis == "h":
        c_step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + c_step, c_step):
            pts.append((r0, c))
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            pts.append((r, c1))
        return pts

    assert first_axis == "v", "path axis must be horizontal or vertical"
    r_step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + r_step, r_step):
        pts.append((r, c0))
    c_step = 1 if c1 >= c0 else -1
    for c in range(c0 + c_step, c1 + c_step, c_step):
        pts.append((r1, c))
    return pts


def _last_axis(start: Coord, stop: Coord, first_axis: str) -> str:
    dr = abs(start[0] - stop[0])
    dc = abs(start[1] - stop[1])
    if first_axis == "h":
        return "v" if dr else "h"
    assert first_axis == "v", "path axis must be horizontal or vertical"
    return "h" if dc else "v"


def _axis_priority(start: Coord, stop: Coord, previous_axis: str | None) -> list[str]:
    dr = abs(start[0] - stop[0])
    dc = abs(start[1] - stop[1])
    assert dr + dc > 0, "components must be distinct"

    orders: list[str] = []
    if previous_axis == "h" and dc:
        orders.append("h")
    if previous_axis == "v" and dr:
        orders.append("v")

    preferred = "h" if dc >= dr else "v"
    if preferred not in orders:
        orders.append(preferred)
    other = "v" if preferred == "h" else "h"
    if other not in orders:
        orders.append(other)
    return orders


def _clear(path: list[Coord], canvas: np.ndarray, bg: int, color: int) -> bool:
    return all(int(canvas[r, c]) in (bg, color) for r, c in path)


def _connector_path(
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    canvas: np.ndarray,
    bg: int,
    color: int,
    previous_axis: str | None,
) -> tuple[list[Coord], str]:
    pairs: list[tuple[int, int, int, int, int]] = []
    for a_raw in comp_a:
        ar, ac = map(int, a_raw)
        for b_raw in comp_b:
            br, bc = map(int, b_raw)
            pairs.append((abs(ar - br) + abs(ac - bc), ar, ac, br, bc))

    pairs.sort()
    for _dist, ar, ac, br, bc in pairs:
        start = (ar, ac)
        stop = (br, bc)
        for first_axis in _axis_priority(start, stop, previous_axis):
            path = _path_cells(start, stop, first_axis)
            if _clear(path, canvas, bg, color):
                return path, _last_axis(start, stop, first_axis)

    raise AssertionError(f"expected an unobstructed one-bend connector for color {color}")


def _source_colors(g: np.ndarray, bg: int, wall: int) -> list[int]:
    colors: list[int] = []
    seen = {int(bg), int(wall)}
    for raw in g.ravel():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-colored foreground components with one-cell-wide orthogonal paths around teal wall segments.

    Input structure
    - The dominant color is the background.
    - Teal cells are fixed wall segments and are never source connectors.
    - Other foreground colors appear as one or more 4-connected components; colors with a single component stay unchanged.

    Rule
    - For each non-background, non-teal color, read its 4-connected components in row-major order.
    - Connect each consecutive pair by the shortest clear L-shaped path between the two components.
    - Clear paths may pass only through background or the same color, and they preserve the previous travel axis when both bends are possible.

    Key insight
    - The output is not a global recolor: each color traces a routed chain through its own components while teal cells and other colors act as blockers.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    wall = Color.TEAL
    assert np.any(g == wall), "expected teal wall cells"

    out = g.copy()
    connected_color_count = 0
    for color in _source_colors(g, bg, wall):
        comps = [
            comp["coords"]
            for comp in find_components(
                g,
                bg=bg,
                connectivity=4,
                monochromatic=True,
                colors=[color],
            )
        ]
        comps.sort(key=lambda coords: tuple(map(int, coords.min(axis=0))))
        if len(comps) < 2:
            continue
        if len(comps) == 2 and len(comps[0]) == 1 and len(comps[1]) == 1:
            a = tuple(map(int, comps[0][0]))
            b = tuple(map(int, comps[1][0]))
            if a[0] != b[0] and a[1] != b[1]:
                h_clear = _clear(_path_cells(a, b, "h"), g, bg, color)
                v_clear = _clear(_path_cells(a, b, "v"), g, bg, color)
                assert not (h_clear and v_clear), "two diagonal singleton components need a blocking obstacle"

        connected_color_count += 1
        previous_axis: str | None = None
        for comp_a, comp_b in zip(comps, comps[1:]):
            path, previous_axis = _connector_path(comp_a, comp_b, out, bg, color, previous_axis)
            for r, c in path:
                assert int(out[r, c]) in (bg, color), "connector collision with another color"
                out[r, c] = color

    assert connected_color_count > 0, "expected at least one color with multiple components"
    return out

# --------------- generator (c) ---------------

BG = Color.BLUE
WALL = Color.TEAL

# Palette.
N_COLORS = 10
ALL_COLORS = [c for c in range(N_COLORS) if c not in (BG, WALL)]

# Canvas sizing.
MIN_ROWS = 8
MAX_ROWS = 12
MIN_COLS = 10
MAX_COLS = 12
LARGE_GRID_CONNECTABLE_THRESHOLD = 3
LARGE_GRID_MIN_ROWS = 10
LARGE_GRID_MIN_COLS = 11

# Foreground object counts and component sizes.
MIN_CONNECTABLE_COLORS = 1
MAX_CONNECTABLE_COLORS = 4
MIN_DISTRACTOR_COLORS = 0
MAX_DISTRACTOR_COLORS = 4
MIN_CHAIN_COMPONENTS = 2
MAX_CHAIN_COMPONENTS = 4
BLOB_SIZE_CHOICES = [1, 2, 3]
BLOB_SIZE_WEIGHTS = [6, 3, 1]

# Wall geometry.
MIN_WALL_SEGMENTS = 1
MAX_WALL_SEGMENTS = 2
WALL_ROW_BORDER = 1
WALL_ROW_GAP = 1
MIN_WALL_LENGTH = 1
MAX_WALL_LENGTH = 8

# Validation and retry limits.
BLOB_PLACEMENT_ATTEMPTS = 80
DISTRACTOR_PLACEMENT_ATTEMPTS = 50
GENERATION_ATTEMPTS = 300
MAX_PATH_CELLS = 25
MIN_BG_PCT = 79
MAX_BG_PCT = 90

AXES = ("h", "v")
CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ConnectorOption = tuple[str, list[Coord], Coord, Coord]


def _gen_source_colors(g: np.ndarray, bg: int, wall: int) -> list[int]:
    colors: list[int] = []
    seen = {int(bg), int(wall)}
    for raw in g.ravel():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _gen_path_has_marker(
    output: np.ndarray,
    color: int,
    path: list[Coord],
    start: Coord,
    stop: Coord,
) -> bool:
    endpoints = {start, stop}
    return any((r, c) not in endpoints and int(output[r, c]) == color for r, c in path)


def _gen_path_cells(start: Coord, stop: Coord, first_axis: str) -> list[Coord]:
    r0, c0 = start
    r1, c1 = stop
    pts: list[Coord] = []

    if first_axis == "h":
        c_step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + c_step, c_step):
            pts.append((r0, c))
        r_step = 1 if r1 >= r0 else -1
        for r in range(r0 + r_step, r1 + r_step, r_step):
            pts.append((r, c1))
        return pts

    r_step = 1 if r1 >= r0 else -1
    for r in range(r0, r1 + r_step, r_step):
        pts.append((r, c0))
    c_step = 1 if c1 >= c0 else -1
    for c in range(c0 + c_step, c1 + c_step, c_step):
        pts.append((r1, c))
    return pts


def _gen_last_axis(start: Coord, stop: Coord, first_axis: str) -> str:
    dr = abs(start[0] - stop[0])
    dc = abs(start[1] - stop[1])
    if first_axis == "h":
        return "v" if dr else "h"
    return "h" if dc else "v"


def _gen_axis_priority(
    start: Coord, stop: Coord, previous_axis: str | None
) -> list[str]:
    dr = abs(start[0] - stop[0])
    dc = abs(start[1] - stop[1])

    orders: list[str] = []
    if previous_axis == "h" and dc:
        orders.append("h")
    if previous_axis == "v" and dr:
        orders.append("v")

    preferred = "h" if dc >= dr else "v"
    if preferred not in orders:
        orders.append(preferred)
    other = "v" if preferred == "h" else "h"
    if other not in orders:
        orders.append(other)
    return orders


def _gen_path_is_clear(canvas: np.ndarray, bg: int, color: int, path: list[Coord]) -> bool:
    return all(int(canvas[r, c]) in (bg, color) for r, c in path)


def _gen_sorted_cell_pairs(comp_a, comp_b) -> list[tuple[int, int, int, int, int]]:
    pairs: list[tuple[int, int, int, int, int]] = []
    for a_cell in comp_a:
        ar, ac = int(a_cell[0]), int(a_cell[1])
        for b_cell in comp_b:
            br, bc = int(b_cell[0]), int(b_cell[1])
            pairs.append((abs(ar - br) + abs(ac - bc), ar, ac, br, bc))
    pairs.sort()
    return pairs


def _gen_clear_connector_options(
    canvas: np.ndarray,
    bg: int,
    color: int,
    comp_a,
    comp_b,
) -> list[ConnectorOption]:
    best_dist: int | None = None
    options: list[ConnectorOption] = []
    seen_paths: set[tuple[Coord, ...]] = set()

    for a_cell in comp_a:
        ar, ac = int(a_cell[0]), int(a_cell[1])
        for b_cell in comp_b:
            br, bc = int(b_cell[0]), int(b_cell[1])
            if ar == br and ac == bc:
                continue
            dist = abs(ar - br) + abs(ac - bc)
            for first_axis in AXES:
                if (first_axis == "h" and ac == bc) or (first_axis == "v" and ar == br):
                    continue
                path = _gen_path_cells((ar, ac), (br, bc), first_axis)
                if not _gen_path_is_clear(canvas, bg, color, path):
                    continue
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    options = []
                    seen_paths = set()
                if dist == best_dist:
                    key = tuple(path)
                    if key not in seen_paths:
                        seen_paths.add(key)
                        options.append((first_axis, path, (ar, ac), (br, bc)))
    return options


def _gen_ambiguous_component_step(
    canvas: np.ndarray,
    bg: int,
    color: int,
    comp_a,
    comp_b,
    previous_axis: str | None,
) -> bool:
    options = _gen_clear_connector_options(canvas, bg, color, comp_a, comp_b)
    if len(options) <= 1:
        return False

    if previous_axis is not None:
        previous_options = [option for option in options if option[0] == previous_axis]
        if len(previous_options) == 1:
            return False

    marked_options = [
        option
        for option in options
        if _gen_path_has_marker(canvas, color, option[1], option[2], option[3])
    ]
    return len(marked_options) != 1


def _gen_input_has_ambiguous_connector(inp: np.ndarray) -> bool:
    """Replay generator routing and reject inputs where the solver route is not unique."""
    bg = get_bg(inp)
    canvas = inp.copy()
    for color in _gen_source_colors(inp, bg, WALL):
        components = [
            comp["coords"]
            for comp in find_components(
                inp,
                bg=bg,
                connectivity=4,
                monochromatic=True,
                colors=[color],
            )
        ]
        components.sort(key=lambda coords: tuple(map(int, coords.min(axis=0))))
        if len(components) < 2:
            continue

        previous_axis: str | None = None
        for comp_a, comp_b in zip(components, components[1:]):
            found = False
            for _d, ar, ac, br, bc in _gen_sorted_cell_pairs(comp_a, comp_b):
                if ar == br and ac == bc:
                    continue
                start = (ar, ac)
                stop = (br, bc)
                if _gen_ambiguous_component_step(canvas, bg, color, comp_a, comp_b, previous_axis):
                    return True
                for first_axis in _gen_axis_priority(start, stop, previous_axis):
                    path = _gen_path_cells(start, stop, first_axis)
                    if _gen_path_is_clear(canvas, bg, color, path):
                        for pr, pc in path:
                            canvas[pr, pc] = color
                        previous_axis = _gen_last_axis(start, stop, first_axis)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    return False


def _try_connect_chain(
    output: np.ndarray, color: int, components: list[list[Coord]]
) -> list[Coord] | None:
    """Try to connect sorted components with L-paths. Returns new path cells or None."""
    path_cells: list[Coord] = []
    previous_axis: str | None = None
    component_cells = {cell for comp in components for cell in comp}
    used_route_cells: set[Coord] = set()

    for i in range(len(components) - 1):
        comp_a = np.array(components[i])
        comp_b = np.array(components[i + 1])

        found = False
        for _d, ar, ac, br, bc in _gen_sorted_cell_pairs(comp_a, comp_b):
            if ar == br and ac == bc:
                continue
            if _gen_ambiguous_component_step(output, BG, color, comp_a, comp_b, previous_axis):
                continue
            for first_axis in _gen_axis_priority((ar, ac), (br, bc), previous_axis):
                path = _gen_path_cells((ar, ac), (br, bc), first_axis)
                if any(cell in used_route_cells for cell in path):
                    continue
                if any(int(output[pr, pc]) == color and (pr, pc) not in component_cells for pr, pc in path):
                    continue
                if _gen_path_is_clear(output, BG, color, path):
                    for pr, pc in path:
                        if output[pr, pc] == BG:
                            output[pr, pc] = color
                            path_cells.append((pr, pc))
                    used_route_cells.update(cell for cell in path if cell not in component_cells)
                    previous_axis = _gen_last_axis((ar, ac), (br, bc), first_axis)
                    found = True
                    break
            if found:
                break
        if not found:
            for pr, pc in path_cells:
                output[pr, pc] = BG
            return None
    return path_cells


def _place_blob(output: np.ndarray, color: int, size: int) -> list[Coord] | None:
    """Place a connected blob of given size on bg cells, not touching same color."""
    H, W = output.shape
    for _ in range(BLOB_PLACEMENT_ATTEMPTS):
        r = random.randint(0, H - 1)
        c = random.randint(0, W - 1)
        if output[r, c] != BG:
            continue
        blob = [(r, c)]
        for _ in range(size - 1):
            base = random.choice(blob)
            for dr, dc in random.sample(CARDINAL_DIRS, len(CARDINAL_DIRS)):
                nr, nc = base[0] + dr, base[1] + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in blob and output[nr, nc] == BG:
                    blob.append((nr, nc))
                    break
        if len(blob) != size:
            continue
        blob_set = set(blob)
        touches_same = any(
            0 <= br + dr < H and 0 <= bc + dc < W
            and (br + dr, bc + dc) not in blob_set
            and output[br + dr, bc + dc] == color
            for br, bc in blob for dr, dc in CARDINAL_DIRS
        )
        if not touches_same:
            for br, bc in blob:
                output[br, bc] = color
            return blob
    return None


def _choose_grid_shape(n_connectable: int) -> tuple[int, int]:
    if n_connectable >= LARGE_GRID_CONNECTABLE_THRESHOLD:
        return (
            random.randint(LARGE_GRID_MIN_ROWS, MAX_ROWS),
            random.randint(LARGE_GRID_MIN_COLS, MAX_COLS),
        )
    return random.randint(MIN_ROWS, MAX_ROWS), random.randint(MIN_COLS, MAX_COLS)


def _place_walls(output: np.ndarray) -> None:
    """Place horizontal teal segments with at least one clear row between chosen rows."""
    H, W = output.shape
    n_walls = random.randint(MIN_WALL_SEGMENTS, MAX_WALL_SEGMENTS)
    rows_pool = list(range(WALL_ROW_BORDER, H - WALL_ROW_BORDER))
    random.shuffle(rows_pool)
    for _ in range(n_walls):
        if not rows_pool:
            break
        row = rows_pool.pop()
        rows_pool = [r for r in rows_pool if abs(r - row) > WALL_ROW_GAP]
        length = random.randint(MIN_WALL_LENGTH, min(MAX_WALL_LENGTH, W - 2))
        c_start = random.randint(0, W - length)
        output[row, c_start : c_start + length] = WALL


def _choose_palette(n_connectable: int) -> tuple[list[int], list[int]]:
    n_distractor = random.randint(
        MIN_DISTRACTOR_COLORS,
        min(MAX_DISTRACTOR_COLORS, len(ALL_COLORS) - n_connectable),
    )
    palette = random.sample(ALL_COLORS, n_connectable + n_distractor)
    return palette[:n_connectable], palette[n_connectable:]


def _gen_component_sort_key(cells: list[Coord]) -> tuple[int, int]:
    return min(r for r, _ in cells), min(c for _, c in cells)


def _place_connectable_chains(output: np.ndarray, colors: list[int]) -> list[Coord] | None:
    """Place source components and draw their hidden output paths."""
    all_path_cells: list[Coord] = []
    for color in colors:
        n_comps = random.randint(MIN_CHAIN_COMPONENTS, MAX_CHAIN_COMPONENTS)
        components: list[list[Coord]] = []
        for _ in range(n_comps):
            blob_size = random.choices(BLOB_SIZE_CHOICES, weights=BLOB_SIZE_WEIGHTS)[0]
            blob = _place_blob(output, color, blob_size)
            if blob is None:
                return None
            components.append(blob)

        components.sort(key=_gen_component_sort_key)
        path_cells = _try_connect_chain(output, color, components)
        if path_cells is None:
            return None
        all_path_cells.extend(path_cells)
    return all_path_cells


def _place_distractors(output: np.ndarray, colors: list[int]) -> bool:
    H, W = output.shape
    for color in colors:
        for _ in range(DISTRACTOR_PLACEMENT_ATTEMPTS):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if output[r, c] == BG:
                output[r, c] = color
                break
        else:
            return False
    return True


def _erase_paths(output: np.ndarray, path_cells: list[Coord]) -> np.ndarray:
    inp = output.copy()
    if path_cells:
        rows, cols = zip(*path_cells)
        inp[rows, cols] = BG
    return inp


def _input_bg_pct(inp: np.ndarray) -> float:
    return float(np.sum(inp == BG) / inp.size * 100)


def _solver_matches_output(inp: np.ndarray, output: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(main(inp), output))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid."""
    n_connectable = random.randint(MIN_CONNECTABLE_COLORS, MAX_CONNECTABLE_COLORS)
    H, W = _choose_grid_shape(n_connectable)
    output = np.full((H, W), BG, dtype=int)

    _place_walls(output)
    connectable_colors, distractor_colors = _choose_palette(n_connectable)

    all_path_cells = _place_connectable_chains(output, connectable_colors)
    if all_path_cells is None or len(all_path_cells) > MAX_PATH_CELLS:
        return None

    if not _place_distractors(output, distractor_colors):
        return None

    inp = _erase_paths(output, all_path_cells)
    if not (MIN_BG_PCT <= _input_bg_pct(inp) <= MAX_BG_PCT):
        return None
    if _gen_input_has_ambiguous_connector(inp):
        return None
    if _solver_matches_output(inp, output):
        return inp
    return None


def generate_input() -> np.ndarray:
    """Generate a valid input grid for task f8f52ecc."""
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
