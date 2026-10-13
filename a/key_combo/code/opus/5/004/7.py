from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import find_components, get_bg, in_bounds, neighbors4
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(values[counts.argmax()])


    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= int(r) < h and 0 <= int(c) < w


    def neighbors4(r: int, c: int):
        yield int(r) - 1, int(c)
        yield int(r) + 1, int(c)
        yield int(r), int(c) - 1
        yield int(r), int(c) + 1


    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = False,
        return_format: str = "dict",
        **_: object,
    ) -> list[dict]:
        assert connectivity == 4
        assert not monochromatic
        assert return_format == "dict"
        g = np.asarray(grid, dtype=int)
        if bg is None:
            bg = get_bg(g)
        seen: set[tuple[int, int]] = set()
        out: list[dict] = []
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if int(g[r, c]) == int(bg) or (r, c) in seen:
                    continue
                queue = deque([(r, c)])
                seen.add((r, c))
                coords: list[tuple[int, int]] = []
                while queue:
                    cr, cc = queue.popleft()
                    coords.append((cr, cc))
                    for nr, nc in neighbors4(cr, cc):
                        if not in_bounds(g.shape, nr, nc):
                            continue
                        nxt = (int(nr), int(nc))
                        if nxt in seen or int(g[nxt]) == int(bg):
                            continue
                        seen.add(nxt)
                        queue.append(nxt)
                arr = np.asarray(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": None,
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(arr.shape[0]),
                    }
                )
        return out


BG = 0
PATH = 3
MARKER = 5
SQUARE = 6
GATE_END = 9

Coord = tuple[int, int]

PROJ_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _add(a: Coord, b: Coord, k: int = 1) -> Coord:
    return int(a[0] + b[0] * k), int(a[1] + b[1] * k)


def _neg(v: Coord) -> Coord:
    return -int(v[0]), -int(v[1])


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield int(r) + dr, int(c) + dc


def _axis_for_projection(proj: Coord) -> Coord:
    if proj[0] != 0:
        return 0, 1
    return 1, 0


def _bbox(coords: set[Coord]) -> tuple[int, int, int, int]:
    arr = np.asarray(sorted(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        p = stack.pop()
        for q in neighbors4(*p):
            q = (int(q[0]), int(q[1]))
            if q in coords and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(coords)


def _project_target(square_bbox: tuple[int, int, int, int], proj: Coord) -> Coord:
    r0, c0, r1, c1 = square_bbox
    if proj[0] != 0:
        return (r0 if proj[0] > 0 else r1), c0 + 1
    return r0 + 1, (c0 if proj[1] > 0 else c1)


def _ordered_path(path_cells: set[Coord], start: Coord) -> list[Coord]:
    order = [start]
    prev: Coord | None = None
    cur = start
    while True:
        nxts = [q for q in neighbors4(*cur) if (int(q[0]), int(q[1])) in path_cells]
        nxts = [(int(r), int(c)) for r, c in nxts if (int(r), int(c)) != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path traversal must stay unambiguous"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(path_cells), "ordered traversal must cover the full path"
    return order


def _inspect_component(g: np.ndarray, coords: np.ndarray) -> dict[str, object]:
    coord_set = {tuple(map(int, p)) for p in coords}
    greens = {p for p in coord_set if int(g[p]) == PATH}
    maroons = {p for p in coord_set if int(g[p]) == GATE_END}
    magentas = {p for p in coord_set if int(g[p]) == SQUARE}
    markers = [p for p in coord_set if int(g[p]) == MARKER]

    assert len(markers) == 1, "each component must contain exactly one gray marker"
    assert len(maroons) == 2, "each component must contain exactly two gate-end pixels"
    assert len(magentas) == 9, "each component must contain one 3x3 magenta square"

    square_bbox = _bbox(magentas)
    r0, c0, r1, c1 = square_bbox
    assert (r1 - r0, c1 - c0) == (2, 2), "square must be exactly 3x3"
    assert magentas == {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}

    gate_specs: list[tuple[Coord, Coord, Coord, Coord]] = []
    for gate in greens:
        for axis in ((1, 0), (0, 1)):
            a = _add(gate, axis, -1)
            b = _add(gate, axis, 1)
            if a not in maroons or b not in maroons:
                continue
            perp0 = (-axis[1], axis[0])
            perp1 = (axis[1], -axis[0])
            proj_dirs = [d for d in (perp0, perp1) if _add(gate, d) in magentas]
            path_dirs = [d for d in (perp0, perp1) if _add(gate, d) in greens]
            if len(proj_dirs) == 1 and len(path_dirs) == 1 and path_dirs[0] == _neg(proj_dirs[0]):
                gate_specs.append((gate, axis, proj_dirs[0], _add(gate, path_dirs[0])))

    assert len(gate_specs) == 1, "each component must contain exactly one oriented gate"
    gate, axis, proj, parking = gate_specs[0]

    path_cells = set(greens) - {gate}
    assert parking in path_cells, "parking cell must be the path cell beside the gate"
    assert _connected(path_cells), "path cells must be 4-connected"

    degrees = {
        p: sum((_add(p, d) in path_cells) for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        for p in path_cells
    }
    assert all(d <= 2 for d in degrees.values()), "path must be branch-free"
    endpoints = {p for p, d in degrees.items() if d <= 1}
    marker = markers[0]
    marker_touch = {p for p in path_cells if abs(p[0] - marker[0]) + abs(p[1] - marker[1]) == 1}
    assert len(marker_touch) == 1, "marker must attach to exactly one path endpoint"
    marker_endpoint = next(iter(marker_touch))
    if len(path_cells) == 1:
        assert parking == marker_endpoint
        bends = 0
    else:
        assert endpoints == {parking, marker_endpoint}, "path must run from marker end to gate end"
        ordered = _ordered_path(path_cells, parking)
        dirs = [
            (ordered[i + 1][0] - ordered[i][0], ordered[i + 1][1] - ordered[i][1])
            for i in range(len(ordered) - 1)
        ]
        bends = sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))

    return {
        "coords": coord_set,
        "gate": gate,
        "axis": axis,
        "proj": proj,
        "parking": parking,
        "marker": marker,
        "path_cells": path_cells,
        "square_bbox": square_bbox,
        "square_cells": magentas,
        "marker_endpoint": marker_endpoint,
        "bends": bends,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray marker to its gate, then project the same mark into the adjacent magenta square using the gate's orientation.

    Input structure:
    - The background is black.
    - Each active component contains one branch-free green path, one gray marker attached to a path endpoint, one maroon-green-maroon gate motif, and one solid 3x3 magenta square touching the gate center on its back side.
    - The gate center is green, the two maroon pixels are collinear, and the path approaches the gate from the opposite side of the magenta square.

    Rule:
    - For every component, move the gray marker along the green path until it stops on the unique green path cell adjacent to the gate center on the path side.
    - Erase the original marker and every traversed green path pixel, leaving only the green gate center and the two maroon gate-end pixels unchanged.
    - Read the gate orientation from the maroon pair: its axis is the maroon-maroon line, and the projection direction is the perpendicular direction from the gate center toward the magenta square.
    - Inside the 3x3 magenta square, mark the center-line contact cell on the gate-facing side with gray, preserving the rest of the square.

    Key insight:
    - The three-pixel gate plays two roles at once: it is the stopping structure for the sliding marker, and its maroon alignment determines which axis is used for the projection into the neighboring square.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        rec = _inspect_component(g, comp["coords"])
        for p in rec["path_cells"]:
            out[p] = bg
        out[rec["marker"]] = bg
        out[rec["parking"]] = MARKER
        out[_project_target(rec["square_bbox"], rec["proj"])] = MARKER
        out[rec["gate"]] = PATH
        for p in rec["square_cells"]:
            if p != _project_target(rec["square_bbox"], rec["proj"]):
                out[p] = SQUARE
        for p in rec["coords"] - rec["path_cells"] - {rec["marker"]} - rec["square_cells"]:
            out[p] = int(g[p])

    return out


def _path_shape(mode: str, outward: Coord, axis: Coord) -> tuple[list[Coord], Coord, int]:
    turn = axis if random.random() < 0.5 else _neg(axis)
    cells = [outward]
    current = cells[0]
    bends = 0

    if mode == "straight":
        for _ in range(random.randint(2, 5)):
            current = _add(current, outward)
            cells.append(current)
        last_dir = outward
    elif mode == "turn":
        for _ in range(random.randint(1, 3)):
            current = _add(current, outward)
            cells.append(current)
        for _ in range(random.randint(2, 5)):
            current = _add(current, turn)
            cells.append(current)
        last_dir = turn
        bends = 1
    else:
        for _ in range(random.randint(1, 3)):
            current = _add(current, outward)
            cells.append(current)
        for _ in range(random.randint(1, 3)):
            current = _add(current, turn)
            cells.append(current)
        for _ in range(random.randint(1, 3)):
            current = _add(current, outward)
            cells.append(current)
        last_dir = outward
        bends = 2

    assert len(set(cells)) == len(cells)
    return cells, _add(current, last_dir), bends


def _make_component(proj: Coord, mode: str) -> dict[str, object]:
    axis = _axis_for_projection(proj)
    outward = _neg(proj)
    path_cells, marker, bends = _path_shape(mode, outward, axis)

    square = {_add(_add((0, 0), proj, depth), axis, side) for depth in (1, 2, 3) for side in (-1, 0, 1)}
    gate = (0, 0)
    gate_ends = {_add(gate, axis, -1), _add(gate, axis, 1)}

    colored: dict[Coord, int] = {p: PATH for p in path_cells}
    colored[gate] = PATH
    for p in gate_ends:
        colored[p] = GATE_END
    for p in square:
        colored[p] = SQUARE
    colored[marker] = MARKER

    rec = {
        "colored": colored,
        "proj": proj,
        "axis": axis,
        "gate": gate,
        "parking": path_cells[0],
        "path_cells": set(path_cells),
        "marker": marker,
        "square_cells": square,
        "square_bbox": _bbox(square),
        "bends": bends,
    }

    path_set = set(path_cells)
    assert _connected(path_set)
    assert sum(abs(marker[0] - p[0]) + abs(marker[1] - p[1]) == 1 for p in path_set) == 1
    return rec


def _translate_component(rec: dict[str, object], offset: Coord) -> dict[str, object]:
    def shift(p: Coord) -> Coord:
        return p[0] + offset[0], p[1] + offset[1]

    colored = {shift(p): color for p, color in rec["colored"].items()}
    square_cells = {shift(p) for p in rec["square_cells"]}
    path_cells = {shift(p) for p in rec["path_cells"]}
    return {
        "colored": colored,
        "proj": rec["proj"],
        "axis": rec["axis"],
        "gate": shift(rec["gate"]),
        "parking": shift(rec["parking"]),
        "path_cells": path_cells,
        "marker": shift(rec["marker"]),
        "square_cells": square_cells,
        "square_bbox": _bbox(square_cells),
        "bends": rec["bends"],
    }


def _separated(cells: set[Coord], occupied: set[Coord]) -> bool:
    for p in cells:
        if p in occupied:
            return False
        for q in _neighbors8(*p):
            if q in occupied:
                return False
    return True


def _placement_offsets(shape: tuple[int, int], rec: dict[str, object]) -> list[Coord]:
    h, w = shape
    coords = set(rec["colored"])
    min_r, min_c, max_r, max_c = _bbox(coords)
    sq_r0, sq_c0, sq_r1, sq_c1 = rec["square_bbox"]

    offsets: list[Coord] = []
    for dr in range(-min_r, h - max_r):
        for dc in range(-min_c, w - max_c):
            if dr + sq_r0 <= 0 or dr + sq_r1 >= h - 1:
                continue
            if dc + sq_c0 <= 0 or dc + sq_c1 >= w - 1:
                continue
            offsets.append((dr, dc))
    random.shuffle(offsets)
    return offsets


def _render_grid(shape: tuple[int, int], records: list[dict[str, object]]) -> np.ndarray:
    g = np.full(shape, BG, dtype=int)
    for rec in records:
        for p, color in rec["colored"].items():
            g[p] = color
    return g


def _expected_output(shape: tuple[int, int], records: list[dict[str, object]]) -> np.ndarray:
    out = _render_grid(shape, records)
    for rec in records:
        for p in rec["path_cells"]:
            out[p] = BG
        out[rec["marker"]] = BG
        out[rec["parking"]] = MARKER
        out[_project_target(rec["square_bbox"], rec["proj"])] = MARKER
        out[rec["gate"]] = PATH
    return out


def _validate_instance(inp: np.ndarray, records: list[dict[str, object]], out: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if int(get_bg(inp)) != BG:
        return False
    if len(records) < 3 or len(records) > 4:
        return False
    if len({rec["proj"] for rec in records}) < 2:
        return False
    comps = find_components(inp, bg=BG, connectivity=4, monochromatic=False)
    if len(comps) != len(records):
        return False

    gate_count = 0
    any_bent = False
    for comp in comps:
        try:
            rec = _inspect_component(inp, comp["coords"])
        except AssertionError:
            return False
        gate_count += 1
        any_bent |= int(rec["bends"]) >= 1
        square_mark = _project_target(rec["square_bbox"], rec["proj"])
        if out[rec["gate"]] != PATH:
            return False
        if out[rec["parking"]] != MARKER:
            return False
        if out[square_mark] != MARKER:
            return False
        if any(out[p] != BG for p in rec["path_cells"] - {rec["parking"]}):
            return False
        if out[rec["marker"]] != BG:
            return False
    if gate_count != len(records):
        return False
    if not any_bent:
        return False
    return bool(np.array_equal(main(inp), out))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(16, 24)
    n = random.randint(3, 4)

    proj_names = list(PROJ_DIRS)
    random.shuffle(proj_names)
    selected = proj_names[:n]

    base_modes = ["straight", "turn", "double"]
    if n == 4:
        base_modes.append(random.choice(base_modes))
    random.shuffle(base_modes)

    local_records = [_make_component(PROJ_DIRS[name], base_modes[i]) for i, name in enumerate(selected)]
    placed: list[dict[str, object]] = []
    occupied: set[Coord] = set()

    order = list(range(n))
    random.shuffle(order)
    for idx in order:
        base = local_records[idx]
        offset = None
        for cand in _placement_offsets((h, w), base):
            translated = _translate_component(base, cand)
            cells = set(translated["colored"])
            if _separated(cells, occupied):
                offset = cand
                placed.append(translated)
                occupied |= cells
                break
        if offset is None:
            return None

    inp = _render_grid((h, w), placed)
    out = _expected_output((h, w), placed)
    if not _validate_instance(inp, placed, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-projection puzzle instance")
