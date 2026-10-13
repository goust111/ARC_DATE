from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_opus_6_013_2",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg
in_bounds = _ARC.in_bounds
neighbors4 = _ARC.neighbors4
neighbors8 = _ARC.neighbors8


BG = Color.BLACK
SOURCE = Color.RED
PATH = Color.GREY
CYCLE = (Color.RED, Color.TEAL, Color.GREY)

HEIGHT_RANGE = range(15, 19)
WIDTH_RANGE = range(15, 19)
SOURCE_COUNT_RANGE = (3, 5)
LENGTH_CHOICES = tuple(range(2, 9))
MAX_ATTEMPTS = 600
SHAPE_TRIES = 200
PLACEMENT_TRIES = 250


Coord = tuple[int, int]


def _touches_any_8(cell: Coord, occupied: set[Coord]) -> bool:
    if cell in occupied:
        return True
    return any(neighbor in occupied for neighbor in neighbors8(*cell))


def _path_dirs(path: list[Coord], source: Coord) -> list[Coord]:
    points = [source] + path
    return [
        (points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1])
        for i in range(len(points) - 1)
    ]


def _is_turned(path: list[Coord], source: Coord) -> bool:
    return len(set(_path_dirs(path, source))) >= 2


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _relative_path(length: int, must_turn: bool) -> list[Coord] | None:
    for _ in range(SHAPE_TRIES):
        if must_turn:
            segments = random.randint(2, min(4, length))
        else:
            segments = random.choice([1, 1, 2, 2, 3])
            segments = min(segments, length)
        if must_turn and segments < 2:
            continue

        segment_lengths = _random_composition(length, segments)
        first_axis = random.choice(("v", "h"))
        sign_v = random.choice((-1, 1))
        sign_h = random.choice((-1, 1))

        r = 0
        c = 0
        path: list[Coord] = []
        for i, seg_len in enumerate(segment_lengths):
            axis = first_axis if i % 2 == 0 else ("h" if first_axis == "v" else "v")
            dr, dc = (sign_v, 0) if axis == "v" else (0, sign_h)
            for _step in range(seg_len):
                r += dr
                c += dc
                path.append((r, c))

        if len(set(path)) != length:
            continue

        distances = [abs(rr) + abs(cc) for rr, cc in path]
        if distances[-1] != max(distances) or distances.count(distances[-1]) != 1:
            continue
        if must_turn and not _is_turned(path, (0, 0)):
            continue
        return path
    return None


def _can_place_object(shape: tuple[int, int], source: Coord, path: list[Coord], occupied: set[Coord]) -> bool:
    object_cells = {source}
    translated_path = []
    for dr, dc in path:
        cell = (source[0] + dr, source[1] + dc)
        if not in_bounds(shape, *cell):
            return False
        translated_path.append(cell)
        object_cells.add(cell)

    if any(_touches_any_8(cell, occupied) for cell in object_cells):
        return False

    path_set = set(translated_path)
    if len(path_set) != len(translated_path):
        return False

    for i, cell in enumerate(translated_path):
        source_neighbors = [neighbor for neighbor in neighbors4(*source) if neighbor == cell]
        path_neighbors = [neighbor for neighbor in neighbors4(*cell) if neighbor in path_set]
        degree = len(path_neighbors) + len(source_neighbors)
        if i == 0:
            if degree != 2 if len(translated_path) > 1 else degree != 1:
                return False
        elif i == len(translated_path) - 1:
            if degree != 1:
                return False
        else:
            if degree != 2:
                return False

    source_path_neighbors = [neighbor for neighbor in neighbors4(*source) if neighbor in path_set]
    return len(source_path_neighbors) == 1


def _place_object(shape: tuple[int, int], path: list[Coord], occupied: set[Coord]) -> tuple[Coord, list[Coord]] | None:
    rows = [0] + [dr for dr, _dc in path]
    cols = [0] + [dc for _dr, dc in path]
    min_r = min(rows)
    max_r = max(rows)
    min_c = min(cols)
    max_c = max(cols)

    source_rows = list(range(-min_r, shape[0] - max_r))
    source_cols = list(range(-min_c, shape[1] - max_c))
    if not source_rows or not source_cols:
        return None

    candidates = [(r, c) for r in source_rows for c in source_cols]
    random.shuffle(candidates)
    for source in candidates[:PLACEMENT_TRIES]:
        if _can_place_object(shape, source, path, occupied):
            translated = [(source[0] + dr, source[1] + dc) for dr, dc in path]
            return source, translated
    return None


def _touching_sources_for_component(sources: list[Coord], cells: set[Coord]) -> list[Coord]:
    out = []
    for source in sources:
        if any(neighbor in cells for neighbor in neighbors4(*source)):
            out.append(source)
    return out


def _trace_path_from_source(source: Coord, cells: set[Coord]) -> list[Coord]:
    adjacent = [neighbor for neighbor in neighbors4(*source) if neighbor in cells]
    assert len(adjacent) == 1, "each source must touch exactly one grey path cell"

    degrees = {
        cell: sum(neighbor in cells for neighbor in neighbors4(*cell))
        for cell in cells
    }
    assert all(degree in (1, 2) for degree in degrees.values()), "path must be unbranched"
    assert sum(degree == 1 for degree in degrees.values()) == 2, "path must have exactly two ends"
    assert degrees[adjacent[0]] == 1, "source must attach at a path endpoint"

    prev = source
    cur = adjacent[0]
    ordered: list[Coord] = []
    seen: set[Coord] = set()
    while True:
        assert cur not in seen, "path cannot loop"
        ordered.append(cur)
        seen.add(cur)
        nxt = [neighbor for neighbor in neighbors4(*cur) if neighbor in cells and neighbor != prev]
        assert len(nxt) <= 1, "path traversal must be unique"
        if not nxt:
            break
        prev, cur = cur, nxt[0]

    assert seen == cells, "path traversal must visit every grey cell"
    distances = [abs(cell[0] - source[0]) + abs(cell[1] - source[1]) for cell in ordered]
    assert distances[-1] == max(distances) and distances.count(distances[-1]) == 1, (
        "the endpoint must be the unique grey cell farthest from the source"
    )
    return ordered


def _analyze(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"
    assert set(map(int, np.unique(g))) <= {BG, SOURCE, PATH}, "unexpected colors present"

    sources = [tuple(map(int, coord)) for coord in np.argwhere(g == SOURCE)]
    assert 3 <= len(sources) <= 5, "expected 3-5 red sources"
    assert len(set(sources)) == len(sources)
    for source in sources:
        red_neighbors = [neighbor for neighbor in neighbors4(*source) if in_bounds(g.shape, *neighbor) and int(g[neighbor]) == SOURCE]
        assert not red_neighbors, "source pixels must be isolated"

    path_components = find_components(
        g,
        bg=bg,
        colors=[PATH],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert len(path_components) == len(sources), "each source must have exactly one grey path"

    comp_cells = [{tuple(map(int, coord)) for coord in comp["coords"]} for comp in path_components]
    for i in range(len(comp_cells)):
        for j in range(i + 1, len(comp_cells)):
            assert not any(_touches_any_8(cell, comp_cells[j]) for cell in comp_cells[i]), (
                "different paths must not touch, even diagonally"
            )

    infos = []
    used_sources: set[Coord] = set()
    all_path_cells = set().union(*comp_cells)
    for cells in comp_cells:
        touching_sources = _touching_sources_for_component(sources, cells)
        assert len(touching_sources) == 1, "each grey path must belong to exactly one source"
        source = touching_sources[0]
        assert source not in used_sources, "a source cannot own multiple paths"
        ordered_path = _trace_path_from_source(source, cells)
        used_sources.add(source)
        infos.append(
            {
                "source": source,
                "path": ordered_path,
                "endpoint": ordered_path[-1],
                "length": len(ordered_path),
            }
        )

    assert used_sources == set(sources), "every source must own one path"
    for source in sources:
        path_neighbors = [neighbor for neighbor in neighbors4(*source) if neighbor in all_path_cells]
        assert len(path_neighbors) == 1

    lengths = [info["length"] for info in infos]
    assert all(2 <= length <= 8 for length in lengths)
    assert len(set(lengths)) == len(lengths), "path lengths must be distinct"
    return infos


def _render_output(shape: tuple[int, int], ordered_infos: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for i, info in enumerate(ordered_infos):
        out[info["endpoint"]] = CYCLE[i % len(CYCLE)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red source to its trail endpoint, then recolor the endpoints by increasing trail length.

    Input structure:
    - The grid is black background with 3-5 red source pixels and one grey 4-connected path attached to each source.
    - Every grey path is one-pixel wide, unbranched, loop-free, and disjoint from every other grey path even diagonally.
    - Each source touches exactly one endpoint of its own grey path, and the grey cell farthest from that source is the path's other endpoint.

    Rule:
    - For each red source, follow its attached grey path from the adjacent grey cell to the other endpoint and count all grey pixels on that path.
    - Sort the sources by that path length in ascending order.
    - Erase every red source and every grey path, then place one pixel at each path endpoint using the sorted color cycle red(2), teal(8), grey(5), repeating every three endpoints.

    Key insight:
    - The output color at an endpoint is determined by the hidden path-length ranking of the source that reaches it, not by the endpoint's spatial position.
    """
    infos = sorted(_analyze(np.asarray(grid, dtype=int)), key=lambda info: info["length"])
    return _render_output(np.asarray(grid).shape, infos)


def _anti_shortcut_checks(grid: np.ndarray, infos: list[dict]) -> bool:
    true_order = sorted(infos, key=lambda info: info["length"])
    true_out = _render_output(grid.shape, true_order)

    alternative_orders = [
        sorted(infos, key=lambda info: info["endpoint"]),
        sorted(infos, key=lambda info: (info["endpoint"][1], info["endpoint"][0])),
        sorted(infos, key=lambda info: info["source"]),
        sorted(infos, key=lambda info: (info["source"][1], info["source"][0])),
    ]
    if any(np.array_equal(true_out, _render_output(grid.shape, order)) for order in alternative_orders):
        return False

    close_pairs = 0
    for i, info_a in enumerate(true_order):
        for j, info_b in enumerate(true_order):
            if i >= j:
                continue
            if CYCLE[i % 3] == CYCLE[j % 3]:
                continue
            dist = abs(info_a["endpoint"][0] - info_b["endpoint"][0]) + abs(info_a["endpoint"][1] - info_b["endpoint"][1])
            if dist <= 4:
                close_pairs += 1
    return close_pairs >= 1


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(HEIGHT_RANGE), random.choice(WIDTH_RANGE))
    source_count = random.randint(*SOURCE_COUNT_RANGE)
    lengths = random.sample(LENGTH_CHOICES, source_count)
    must_turn_count = random.randint(1, min(3, source_count))
    must_turn_lengths = set(random.sample(lengths, must_turn_count))

    occupied: set[Coord] = set()
    objects: list[dict] = []
    for length in sorted(lengths, reverse=True):
        rel_path = _relative_path(length, must_turn=length in must_turn_lengths)
        if rel_path is None:
            return None
        placement = _place_object(shape, rel_path, occupied)
        if placement is None:
            return None
        source, path = placement
        occupied.add(source)
        occupied.update(path)
        objects.append({"source": source, "path": path, "length": length})

    grid = np.full(shape, BG, dtype=int)
    for obj in objects:
        grid[obj["source"]] = SOURCE
        for cell in obj["path"]:
            grid[cell] = PATH

    infos = _analyze(grid)
    if not any(_is_turned(info["path"], info["source"]) for info in infos):
        return None
    if not any(not _is_turned(info["path"], info["source"]) for info in infos):
        return None
    if not _anti_shortcut_checks(grid, infos):
        return None
    if not np.array_equal(main(grid), _render_output(shape, sorted(infos, key=lambda info: info["length"]))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid trajectory-length sorting instance")
