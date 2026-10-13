from __future__ import annotations

import numpy as np

from arc_utils import colors_present, find_components, get_bg, in_bounds, neighbors4
import random


Coord = tuple[int, int]


def _role_colors(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    gate_specs: list[tuple[int, int]] = []
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            center = int(g[r, c])
            if center == bg:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                r0, c0 = r - dr, c - dc
                r1, c1 = r + dr, c + dc
                if not (in_bounds(g.shape, r0, c0) and in_bounds(g.shape, r1, c1)):
                    continue
                end0 = int(g[r0, c0])
                end1 = int(g[r1, c1])
                if end0 == end1 and end0 != bg and end0 != center:
                    gate_specs.append((center, end0))

    assert gate_specs
    path_colors = {path for path, _end in gate_specs}
    end_colors = {end for _path, end in gate_specs}
    assert len(path_colors) == 1
    assert len(end_colors) == 1
    path_color = path_colors.pop()
    end_color = end_colors.pop()

    non_bg = set(colors_present(g, exclude={bg}))
    marker_colors = non_bg - {path_color, end_color}
    assert len(marker_colors) == 1
    marker_color = marker_colors.pop()
    return path_color, end_color, marker_color


def _component_gates(
    g: np.ndarray, coords: np.ndarray, path_color: int, end_color: int
) -> list[tuple[int, int, int, int]]:
    coords_set = {tuple(map(int, coord)) for coord in coords}
    gates: list[tuple[int, int, int, int]] = []
    for r, c in coords_set:
        if int(g[r, c]) != path_color:
            continue
        for dr, dc in ((1, 0), (0, 1)):
            a = (r - dr, c - dc)
            b = (r + dr, c + dc)
            if a in coords_set and b in coords_set:
                if int(g[a]) == end_color and int(g[b]) == end_color:
                    gates.append((r, c, dr, dc))
    return gates


def _path_reaches_marker(
    g: np.ndarray,
    start: Coord,
    marker: Coord,
    path_color: int,
    blocked: set[Coord],
) -> bool:
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        if abs(r - marker[0]) + abs(c - marker[1]) == 1:
            return True
        for nr, nc in neighbors4(r, c):
            nxt = (int(nr), int(nc))
            if nxt in seen or nxt in blocked:
                continue
            if not in_bounds(g.shape, nr, nc):
                continue
            if int(g[nr, nc]) != path_color:
                continue
            seen.add(nxt)
            stack.append(nxt)
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gray marker along its green path until it reaches the cell next to a black-green-black gate, erasing the path behind it.

    Input structure:
    - The background is the dominant color.
    - Foreground components are made from a path color, a paired gate-end color, and single marker pixels.
    - A gate is a straight end-center-end motif whose center has the path color and whose two ends share the gate-end color.

    Rule:
    - In every 4-connected foreground component with exactly one marker and one gate, erase the marker and all path-color pixels except the gate center.
    - Place the marker on the unique path-color pixel adjacent to the gate center but not between the two gate ends.
    - Components without both a marker and a gate stay unchanged.

    Key insight: The black-green-black motif marks the stopping gate; the gray marker consumes the connected green path until it rests on the path cell immediately beside that gate.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    path_color, end_color, marker_color = _role_colors(g, bg)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False):
        coords = comp["coords"]
        marker_coords = [
            tuple(map(int, coord)) for coord in coords if int(g[tuple(coord)]) == marker_color
        ]
        gates = _component_gates(g, coords, path_color, end_color)
        if not marker_coords or not gates:
            continue

        assert len(marker_coords) == 1
        assert len(gates) == 1
        gate_r, gate_c, dr, dc = gates[0]
        gate = (gate_r, gate_c)
        gate_ends = {(gate_r - dr, gate_c - dc), (gate_r + dr, gate_c + dc)}
        coords_set = {tuple(map(int, coord)) for coord in coords}

        parking = []
        for nr, nc in neighbors4(gate_r, gate_c):
            candidate = (int(nr), int(nc))
            if candidate in gate_ends:
                continue
            if candidate in coords_set and int(g[candidate]) == path_color:
                parking.append(candidate)
        assert len(parking) == 1

        marker = marker_coords[0]
        assert _path_reaches_marker(g, parking[0], marker, path_color, {gate})

        for r, c in coords:
            if int(g[r, c]) in (path_color, marker_color):
                out[r, c] = bg
        out[gate] = path_color
        out[parking[0]] = marker_color

    return out

# --------------- generator (o5x) ---------------

# -- Colors --
BG = 7
PATH = 3
END = 0
MARKER = 5

# -- Grid dimensions --
GRID_MIN, GRID_MAX = 9, 14  # height/width range (inclusive)

# -- Component counts --
N_ACTIVE_MIN, N_ACTIVE_MAX = 1, 2      # active components per grid
N_STATIC_GATE_MIN, N_STATIC_GATE_MAX = 0, 2  # static-gate (and static-marker) count

# -- Path geometry --
ARM1_EXTRA_MAX = 1   # extra perpendicular arm cells (0 or 1)
ARM2_MAX = 10        # max parallel arm length

# -- Retry budget --
MAX_RETRIES = 300


def _add(a: tuple[int, int], b: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return a[0] + b[0] * k, a[1] + b[1] * k


def _on_boundary(shape: tuple[int, int], r: int, c: int) -> bool:
    H, W = shape
    return r == 0 or c == 0 or r == H - 1 or c == W - 1


def _gate_frames(H: int, W: int):
    """Yield all valid gate placements: (center, gate_ends, inward_dir, parallel_dirs)."""
    for c in range(1, W - 1):
        yield (0, c), {(0, c - 1), (0, c + 1)}, (1, 0), ((0, -1), (0, 1))
        yield (H - 1, c), {(H - 1, c - 1), (H - 1, c + 1)}, (-1, 0), ((0, -1), (0, 1))
    for r in range(1, H - 1):
        yield (r, 0), {(r - 1, 0), (r + 1, 0)}, (0, 1), ((-1, 0), (1, 0))
        yield (r, W - 1), {(r - 1, W - 1), (r + 1, W - 1)}, (0, -1), ((-1, 0), (1, 0))


def _fits(component: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    """Check that no cell in component overlaps or is 4-adjacent to occupied cells."""
    for p in component:
        if p in occupied:
            return False
        if any(q in occupied for q in neighbors4(*p)):
            return False
    return True


def _active_markers(
    shape: tuple[int, int],
    far_end: tuple[int, int],
    path_cells: set[tuple[int, int]],
    gate_ends: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Return valid marker positions: on boundary, adjacent to far_end, not adjacent to other path/gate cells."""
    markers = []
    for nr, nc in neighbors4(*far_end):
        m = (nr, nc)
        if not in_bounds(shape, nr, nc) or not _on_boundary(shape, nr, nc):
            continue
        if m in path_cells or m in gate_ends:
            continue
        if any(q != far_end and abs(nr - q[0]) + abs(nc - q[1]) == 1 for q in path_cells):
            continue
        if any(abs(nr - q[0]) + abs(nc - q[1]) == 1 for q in gate_ends):
            continue
        markers.append(m)
    return markers


def _record_candidates(
    H: int, W: int, kind: str, occupied: set[tuple[int, int]]
) -> list[dict[str, object]]:
    """Enumerate all valid component placements of the given kind that don't conflict with occupied."""
    shape = (H, W)
    records: list[dict[str, object]] = []
    for center, gate_ends, inward, parallel_dirs in _gate_frames(H, W):
        parking = _add(center, inward)
        if not in_bounds(shape, *parking):
            continue
        for extra in range(ARM1_EXTRA_MAX + 1):
            turn = parking if extra == 0 else _add(parking, inward)
            if not in_bounds(shape, *turn):
                continue
            for parallel in parallel_dirs:
                for arm2 in range(1, ARM2_MAX + 1 - extra):
                    arm_cells = [_add(turn, parallel, k) for k in range(1, arm2 + 1)]
                    if any(not in_bounds(shape, *p) for p in arm_cells):
                        continue
                    path_tail = {parking, *arm_cells}
                    if extra:
                        path_tail.add(turn)
                    path_cells = {center, *path_tail}
                    far_end = arm_cells[-1]
                    base_component = set(gate_ends) | path_cells

                    if kind == "static_gate":
                        if _fits(base_component, occupied):
                            records.append({"kind": kind, "component": base_component,
                                            "out_path": path_cells, "gate_ends": gate_ends})
                        continue

                    for marker in _active_markers(shape, far_end, path_cells, gate_ends):
                        component = base_component | {marker}
                        if not _fits(component, occupied):
                            continue
                        records.append({"kind": kind, "component": component,
                                        "center": center, "parking": parking,
                                        "path_tail": path_tail, "marker": marker,
                                        "gate_ends": gate_ends})
    return records


def _static_marker_candidates(
    H: int, W: int, occupied: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return boundary cells that are isolated (not adjacent) from all occupied cells."""
    shape = (H, W)
    return [
        (r, c) for r in range(H) for c in range(W)
        if _on_boundary(shape, r, c) and _fits({(r, c)}, occupied)
    ]


def _render_grids(
    H: int, W: int, records: list[dict[str, object]], static_markers: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Build the output and input grids from placed component records."""
    out = np.full((H, W), BG, dtype=int)

    for rec in records:
        gate_ends = rec["gate_ends"]
        for p in gate_ends:
            out[p] = END
        if rec["kind"] == "active":
            out[rec["center"]] = PATH
            out[rec["parking"]] = MARKER
        else:
            for p in rec["out_path"]:
                out[p] = PATH

    for p in static_markers:
        out[p] = MARKER

    inp = out.copy()
    for rec in records:
        if rec["kind"] != "active":
            continue
        for p in rec["path_tail"]:
            inp[p] = PATH
        inp[rec["marker"]] = MARKER

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_active = random.randint(N_ACTIVE_MIN, N_ACTIVE_MAX)
    n_static_gate = random.randint(N_STATIC_GATE_MIN, N_STATIC_GATE_MAX)
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.choice([w for w in (H - 1, H, H + 1) if GRID_MIN <= w <= GRID_MAX])

    kinds = ["active"] * n_active + ["static_gate"] * n_static_gate
    random.shuffle(kinds)

    # Place components one by one, each must fit without overlap/adjacency
    records: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    for kind in kinds:
        candidates = _record_candidates(H, W, kind, occupied)
        if not candidates:
            return None
        rec = random.choice(candidates)
        records.append(rec)
        occupied.update(rec["component"])

    # Place isolated static markers (one per static gate)
    static_markers: list[tuple[int, int]] = []
    for _ in range(n_static_gate):
        candidates = _static_marker_candidates(H, W, occupied)
        if not candidates:
            return None
        marker = random.choice(candidates)
        static_markers.append(marker)
        occupied.add(marker)

    inp, out = _render_grids(H, W, records, static_markers)

    # Validate: BG is dominant, exactly 4 colors, solver reproduces output
    counts = np.bincount(inp.ravel(), minlength=10)
    if int(np.argmax(counts)) != BG:
        return None
    if set(np.unique(inp)) != {END, PATH, MARKER, BG}:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 182e5d0f input")
