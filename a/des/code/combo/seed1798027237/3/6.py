from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
START_COLOR = 4
CONNECTOR_COLOR = 5
END_COLOR = 7
SEGMENT_COLOR_POOL = [1, 2, 3, 6, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 2000
PLACE_RETRIES = 250
WALK_RETRIES = 120

Coord = tuple[int, int]


def _add(a: Coord, b: Coord) -> Coord:
    return (int(a[0] + b[0]), int(a[1] + b[1]))


def _sub(a: Coord, b: Coord) -> Coord:
    return (int(a[0] - b[0]), int(a[1] - b[1]))


def _neg(a: Coord) -> Coord:
    return (-int(a[0]), -int(a[1]))


def _mul(a: Coord, k: int) -> Coord:
    return (int(a[0] * k), int(a[1] * k))


def _in_bounds(cell: Coord, shape: tuple[int, int]) -> bool:
    return 0 <= int(cell[0]) < int(shape[0]) and 0 <= int(cell[1]) < int(shape[1])


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _discover_segment_colors(grid: np.ndarray) -> list[int]:
    seen: set[int] = set()
    colors: list[int] = []
    h, w = grid.shape
    for r in range(int(h)):
        for c in range(int(w)):
            color = int(grid[r, c])
            if color in {BG, START_COLOR, END_COLOR} or color in seen:
                continue
            seen.add(color)
            colors.append(color)
    return colors


def _single_marker(grid: np.ndarray, color: int) -> Coord:
    coords = np.argwhere(grid == int(color))
    assert coords.shape == (1, 2), f"expected exactly one marker of color {color}"
    return tuple(map(int, coords[0]))


def _ordered_path(coords: np.ndarray) -> list[Coord]:
    cells = [tuple(map(int, rc)) for rc in coords]
    cell_set = set(cells)
    assert len(cell_set) == len(cells) >= 2

    adjacency: dict[Coord, list[Coord]] = {}
    for cell in cells:
        nbrs = [nbr for nbr in _neighbors4(cell) if nbr in cell_set]
        adjacency[cell] = nbrs

    endpoints = [cell for cell, nbrs in adjacency.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "segment must be a simple path with two endpoints"
    for cell, nbrs in adjacency.items():
        assert len(nbrs) in {1, 2}, "segment path cannot branch"

    ordered = [endpoints[0]]
    prev: Coord | None = None
    cur = endpoints[0]
    while len(ordered) < len(cells):
        nxts = [nbr for nbr in adjacency[cur] if nbr != prev]
        assert len(nxts) == 1
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert ordered[-1] == endpoints[1]
    return ordered


def _segment_orientations(color: int, coords: np.ndarray) -> list[dict]:
    ordered = _ordered_path(coords)
    variants = []
    for path in (ordered, list(reversed(ordered))):
        entry = path[0]
        exit_ = path[-1]
        entry_out = _sub(path[0], path[1])
        exit_out = _sub(path[-1], path[-2])
        rel = [_sub(cell, entry) for cell in path]
        variants.append(
            {
                "color": int(color),
                "path": path,
                "rel": rel,
                "entry_out": entry_out,
                "exit_out": exit_out,
                "delta_end": rel[-1],
            }
        )
    return variants


def _expected_end_delta(oriented_segments: list[dict]) -> Coord:
    total = _neg(oriented_segments[0]["entry_out"])
    for i, seg in enumerate(oriented_segments):
        total = _add(total, seg["delta_end"])
        if i + 1 < len(oriented_segments):
            total = _add(total, _mul(seg["exit_out"], 2))
        else:
            total = _add(total, seg["exit_out"])
    return total


def _render_candidate(
    shape: tuple[int, int],
    start: Coord,
    end: Coord,
    oriented_segments: list[dict],
) -> np.ndarray | None:
    out = np.full(shape, BG, dtype=int)
    out[start] = START_COLOR
    out[end] = END_COLOR
    occupied = {start, end}

    prev_exit: Coord | None = None
    prev_dir: Coord | None = None

    for i, seg in enumerate(oriented_segments):
        if i == 0:
            entry_abs = _sub(start, seg["entry_out"])
        else:
            assert prev_exit is not None and prev_dir is not None
            if seg["entry_out"] != _neg(prev_dir):
                return None
            connector = _add(prev_exit, prev_dir)
            if not _in_bounds(connector, shape) or connector in occupied:
                return None
            out[connector] = CONNECTOR_COLOR
            occupied.add(connector)
            entry_abs = _add(connector, prev_dir)

        abs_cells: list[Coord] = []
        for rel in seg["rel"]:
            cell = _add(entry_abs, rel)
            if not _in_bounds(cell, shape) or cell in occupied:
                return None
            abs_cells.append(cell)

        for cell in abs_cells:
            out[cell] = int(seg["color"])
            occupied.add(cell)

        prev_exit = abs_cells[-1]
        prev_dir = seg["exit_out"]

    assert prev_exit is not None and prev_dir is not None
    if _add(prev_exit, prev_dir) != end:
        return None
    return out


def _unique_solution(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert START_COLOR in g and END_COLOR in g
    assert CONNECTOR_COLOR not in g, "input must not already contain connectors"

    start = _single_marker(g, START_COLOR)
    end = _single_marker(g, END_COLOR)
    segment_colors = _discover_segment_colors(g)
    assert 2 <= len(segment_colors) <= len(SEGMENT_COLOR_POOL)

    segment_variants: list[list[dict]] = []
    for color in segment_colors:
        coords = np.argwhere(g == int(color))
        ordered = _ordered_path(coords)
        assert len(ordered) <= 5, "segment length must stay within the puzzle family"
        segment_variants.append(_segment_orientations(color, coords))

    target_delta = _sub(end, start)
    solutions: dict[bytes, np.ndarray] = {}
    n = len(segment_variants)

    for perm in itertools.permutations(range(n)):
        for flips in itertools.product((0, 1), repeat=n):
            oriented = [segment_variants[perm[i]][flips[i]] for i in range(n)]
            ok = True
            for i in range(n - 1):
                if oriented[i]["exit_out"] != _neg(oriented[i + 1]["entry_out"]):
                    ok = False
                    break
            if not ok:
                continue
            if _expected_end_delta(oriented) != target_delta:
                continue

            candidate = _render_candidate(g.shape, start, end, oriented)
            if candidate is None:
                continue
            solutions[candidate.tobytes()] = candidate

    assert len(solutions) == 1, f"expected a unique reconstruction, found {len(solutions)}"
    return next(iter(solutions.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered colored path fragments into a single start-to-end polyline.

    Input structure:
    - Black background.
    - One yellow start marker and one orange end marker.
    - Several uniquely colored 4-connected path fragments, each a simple path.

    Rule:
    - Determine fragment order from endpoint extension directions.
    - Translate fragments without rotation so the first touches the start marker,
      consecutive fragments are separated by exactly one cell, and the last touches
      the end marker.
    - Fill each inter-fragment gap with one gray connector pixel.
    """
    return _unique_solution(np.asarray(grid, dtype=int))


def _turn_count(path: list[Coord]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        a = _sub(path[i], path[i - 1])
        b = _sub(path[i + 1], path[i])
        turns += int(a != b)
    return turns


def _is_simple_sequence(path: list[Coord]) -> bool:
    cells = set(path)
    if len(cells) != len(path):
        return False
    for i, cell in enumerate(path):
        intended: set[Coord] = set()
        if i > 0:
            intended.add(path[i - 1])
        if i + 1 < len(path):
            intended.add(path[i + 1])
        actual = {nbr for nbr in _neighbors4(cell) if nbr in cells}
        if actual != intended:
            return False
    return True


_SEGMENT_PATH_CACHE: dict[tuple[int, Coord], list[list[Coord]]] = {}


def _relative_segment_paths(length: int, in_dir: Coord) -> list[list[Coord]]:
    key = (int(length), in_dir)
    cached = _SEGMENT_PATH_CACHE.get(key)
    if cached is not None:
        return cached

    base = [(0, 0), in_dir]
    paths: list[list[Coord]] = []
    used = set(base)

    def dfs(path: list[Coord]) -> None:
        if len(path) == length:
            if _is_simple_sequence(path):
                paths.append(path.copy())
            return
        for step in DIRS:
            nxt = _add(path[-1], step)
            if nxt in used:
                continue
            used.add(nxt)
            path.append(nxt)
            dfs(path)
            path.pop()
            used.remove(nxt)

    dfs(base.copy())
    _SEGMENT_PATH_CACHE[key] = paths
    return paths


def _extension_valid(
    prefix_end: Coord,
    occupied: set[Coord],
    new_positions: list[Coord],
    shape: tuple[int, int],
) -> bool:
    new_set = set(new_positions)
    if len(new_set) != len(new_positions):
        return False
    for cell in new_positions:
        if not _in_bounds(cell, shape) or cell in occupied:
            return False

    universe = set(occupied) | new_set
    for i, cell in enumerate(new_positions):
        intended: set[Coord] = set()
        intended.add(prefix_end if i == 0 else new_positions[i - 1])
        if i + 1 < len(new_positions):
            intended.add(new_positions[i + 1])
        actual = {nbr for nbr in _neighbors4(cell) if nbr in universe}
        if actual != intended:
            return False
    return True


def _chain_dfs(
    shape: tuple[int, int],
    lengths: list[int],
    index: int,
    entry: Coord,
    in_dir: Coord,
    occupied: set[Coord],
    prefix_end: Coord,
    segments: list[list[Coord]],
    connectors: list[Coord],
    turns: int,
) -> tuple[Coord, list[list[Coord]], list[Coord], int] | None:
    candidates = _relative_segment_paths(lengths[index], in_dir).copy()
    random.shuffle(candidates)

    for rel_path in candidates:
        abs_cells = [_add(entry, rel) for rel in rel_path]
        out_dir = _sub(rel_path[-1], rel_path[-2])

        if index + 1 == len(lengths):
            end = _add(abs_cells[-1], out_dir)
            extension = abs_cells + [end]
            if not _extension_valid(prefix_end, occupied, extension, shape):
                continue
            return end, segments + [abs_cells], connectors, turns + _turn_count(abs_cells)

        connector = _add(abs_cells[-1], out_dir)
        extension = abs_cells + [connector]
        if not _extension_valid(prefix_end, occupied, extension, shape):
            continue

        next_entry = _add(connector, out_dir)
        if not _in_bounds(next_entry, shape) or next_entry in occupied or next_entry in extension:
            continue

        result = _chain_dfs(
            shape=shape,
            lengths=lengths,
            index=index + 1,
            entry=next_entry,
            in_dir=out_dir,
            occupied=occupied | set(extension),
            prefix_end=connector,
            segments=segments + [abs_cells],
            connectors=connectors + [connector],
            turns=turns + _turn_count(abs_cells),
        )
        if result is not None:
            return result
    return None


def _generate_output_chain(
    shape: tuple[int, int],
    lengths: list[int],
) -> tuple[Coord, Coord, list[list[Coord]], list[Coord], int] | None:
    h, w = shape
    starts = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(starts)
    directions = DIRS.copy()

    for start in starts[: min(50, len(starts))]:
        random.shuffle(directions)
        for in_dir in directions:
            entry = _add(start, in_dir)
            result = _chain_dfs(
                shape=shape,
                lengths=lengths,
                index=0,
                entry=entry,
                in_dir=in_dir,
                occupied={start},
                prefix_end=start,
                segments=[],
                connectors=[],
                turns=0,
            )
            if result is None:
                continue
            end, segments, connectors, turns = result
            return start, end, segments, connectors, turns
    return None


def _shape_spec(coords: list[Coord], color: int) -> dict:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    return {
        "color": int(color),
        "rel": [(r - r0, c - c0) for r, c in coords],
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "original": set(coords),
        "bbox_area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
    }


def _mark_blocked(blocked: np.ndarray, cells: list[Coord]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_scattered_segments(
    shape: tuple[int, int],
    start: Coord,
    end: Coord,
    segments: list[list[Coord]],
    colors: list[int],
) -> np.ndarray | None:
    inp = np.full(shape, BG, dtype=int)
    inp[start] = START_COLOR
    inp[end] = END_COLOR

    blocked = np.zeros(shape, dtype=bool)
    _mark_blocked(blocked, [start, end])

    specs = [_shape_spec(seg, color) for seg, color in zip(segments, colors)]
    random.shuffle(specs)
    specs.sort(key=lambda spec: (-spec["bbox_area"], -len(spec["rel"])))

    h, w = shape
    for spec in specs:
        placed = False
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(0, h - spec["h"])
            c0 = random.randint(0, w - spec["w"])
            cells = [(r0 + dr, c0 + dc) for dr, dc in spec["rel"]]
            if set(cells) == spec["original"]:
                continue
            if any(blocked[r, c] for r, c in cells):
                continue
            for r, c in cells:
                inp[r, c] = int(spec["color"])
            _mark_blocked(blocked, cells)
            placed = True
            break
        if not placed:
            return None
    return inp


def _build_output(
    shape: tuple[int, int],
    start: Coord,
    end: Coord,
    segments: list[list[Coord]],
    connectors: list[Coord],
    colors: list[int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    out[start] = START_COLOR
    out[end] = END_COLOR
    for cells, color in zip(segments, colors):
        for r, c in cells:
            out[r, c] = int(color)
    for r, c in connectors:
        out[r, c] = CONNECTOR_COLOR
    return out


def _generate_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_segments = random.randint(2, 5)
    lengths = [random.randint(2, 5) for _ in range(n_segments)]

    h = random.randint(12, 18)
    w = random.randint(12, 18)

    need_turn = random.random() < 0.7
    for _ in range(WALK_RETRIES):
        built = _generate_output_chain((h, w), lengths)
        if built is None:
            continue
        start, end, segments, connectors, turns = built
        if need_turn and turns == 0:
            continue
        if turns < random.randint(1, 3):
            continue
        colors = random.sample(SEGMENT_COLOR_POOL, n_segments)
        out = _build_output((h, w), start, end, segments, connectors, colors)
        inp = _place_scattered_segments((h, w), start, end, segments, colors)
        if inp is None or np.array_equal(inp, out):
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is None:
            continue
        inp, out = case
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, out):
            return inp
    raise RuntimeError("failed to generate a valid puzzle input")
