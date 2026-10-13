from __future__ import annotations

import random
from typing import Iterable

import numpy as np


Coord = tuple[int, int]
Dir = tuple[int, int]

BLACK = 0
NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]

STEP = 4
MIN_SIDE = 18
MAX_SIDE = 28
MIN_REFLECTIONS = 3
MAX_REFLECTIONS = 5
MAX_GENERATE_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors8(r: int, c: int) -> Iterable[Coord]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: list[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _c in coords]
    cs = [c for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _components_mask(mask: np.ndarray, connectivity: int = 8) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    if connectivity == 4:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        nbrs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in nbrs:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _component_orientation(coords: list[Coord]) -> str:
    assert len(coords) >= 2, "mirror segments must have length >= 2"
    pts = {(int(r), int(c)) for r, c in coords}

    deltas = {r - c for r, c in pts}
    if len(deltas) == 1:
        delta = next(iter(deltas))
        r0, _c0, r1, _c1 = _bbox_coords(list(pts))
        expected = {(r, r - delta) for r in range(r0, r1 + 1)}
        if expected == pts:
            return "\\"

    sums = {r + c for r, c in pts}
    if len(sums) == 1:
        sigma = next(iter(sums))
        r0, _c0, r1, _c1 = _bbox_coords(list(pts))
        expected = {(r, sigma - r) for r in range(r0, r1 + 1)}
        if expected == pts:
            return "/"

    raise AssertionError("every mirror component must be one contiguous diagonal segment")


def _marker_and_mirrors(grid: np.ndarray) -> tuple[Coord, dict[Coord, tuple[int, str]]]:
    h, w = grid.shape
    boundary: list[Coord] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BLACK:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                boundary.append((r, c))
    assert len(boundary) == 1, "expected exactly one non-black boundary marker"
    marker = boundary[0]

    mirrors: dict[Coord, tuple[int, str]] = {}
    for color in sorted(set(int(v) for v in grid.flatten()) - {BLACK}):
        mask = grid == color
        if color == int(grid[marker]):
            mask = mask.copy()
            mask[marker] = False
        for comp in _components_mask(mask, connectivity=8):
            orient = _component_orientation(comp)
            for cell in comp:
                assert cell not in mirrors
                mirrors[cell] = (color, orient)
    return marker, mirrors


def _marker_direction(marker: Coord, shape: tuple[int, int]) -> Dir:
    r, c = marker
    h, w = shape
    dirs: list[Dir] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "marker must touch exactly one border side"
    return dirs[0]


def _reflect(direction: Dir, orient: str) -> Dir:
    dr, dc = direction
    if orient == "\\":
        return dc, dr
    if orient == "/":
        return -dc, -dr
    raise ValueError("invalid mirror orientation")


def _turn_to_orient(in_dir: Dir, out_dir: Dir) -> str:
    if out_dir == _reflect(in_dir, "\\"):
        return "\\"
    if out_dir == _reflect(in_dir, "/"):
        return "/"
    raise AssertionError("outgoing direction must be the reflection of incoming direction")


def _trace_ray(
    grid: np.ndarray,
    *,
    same_color_transparent: bool = True,
    recolor_on_reflect: bool = True,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker, mirrors = _marker_and_mirrors(g)
    direction = _marker_direction(marker, g.shape)
    current_color = int(g[marker])
    out = g.copy()

    r, c = marker
    visited: set[tuple[int, int, int, int, int]] = set()
    events: list[dict] = []
    painted = 0
    reflections = 0
    pass_throughs = 0

    while True:
        state = (r, c, direction[0], direction[1], current_color)
        assert state not in visited, "ray entered a loop"
        visited.add(state)

        nr, nc = r + direction[0], c + direction[1]
        if not _in_bounds((h, w), nr, nc):
            break

        if (nr, nc) in mirrors:
            mirror_color, orient = mirrors[(nr, nc)]
            event = {
                "coord": (nr, nc),
                "orient": orient,
                "mirror_color": mirror_color,
                "color_before": current_color,
                "reflections_before": reflections,
            }
            if same_color_transparent and mirror_color == current_color:
                pass_throughs += 1
                event["kind"] = "pass"
            else:
                direction = _reflect(direction, orient)
                if recolor_on_reflect:
                    current_color = mirror_color
                reflections += 1
                event["kind"] = "reflect"
                event["color_after"] = current_color
            events.append(event)
            r, c = nr, nc
            continue

        if (nr, nc) != marker:
            assert g[nr, nc] == BLACK, f"unexpected non-mirror foreground at {(nr, nc)}"
            if out[nr, nc] == BLACK:
                out[nr, nc] = current_color
                painted += 1
        r, c = nr, nc

    return out, {
        "events": events,
        "painted": painted,
        "reflections": reflections,
        "pass_throughs": pass_throughs,
        "marker": marker,
        "mirrors": mirrors,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the boundary-emitted light ray through diagonal mirrors, recoloring it after each true reflection and letting same-color mirrors become transparent.

    Input structure:
    - The background is black.
    - A single non-black boundary marker pixel is the ray source; its border side determines the inward starting direction.
    - Interior foreground objects are monochrome one-cell-thick diagonal mirror segments, each oriented as "/" or "\\".

    Rule:
    - Move the ray one grid cell at a time and paint only background cells it traverses with the current ray color.
    - When the ray enters a mirror cell whose color differs from the current ray color, reflect it by 90 degrees according to that mirror and change the ray color to the mirror color.
    - When the ray enters a mirror cell of the same color, pass straight through without turning or recoloring.
    - Stop once the ray steps outside the grid; keep the marker and all mirror pixels unchanged.

    Key insight:
    - Reflection depends on both geometry and color history: once the ray has adopted a mirror's color, every mirror of that color stops affecting its direction.
    """
    return _trace_ray(np.asarray(grid, dtype=int))[0]


def _line_candidates(size: int) -> list[int]:
    vals = list(range(2, size - 2, STEP))
    return [v for v in vals if 1 <= v <= size - 2]


def _perpendiculars(direction: Dir) -> list[Dir]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _segment_cells(start: Coord, end: Coord) -> list[Coord]:
    r0, c0 = start
    r1, c1 = end
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        return [(r0, c) for c in range(c0 + step, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 > r0 else -1
    return [(r, c0) for r in range(r0 + step, r1 + step, step)]


def _exit_cells(start: Coord, direction: Dir, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    r, c = start
    dr, dc = direction
    out: list[Coord] = []
    while True:
        r += dr
        c += dc
        if not _in_bounds((h, w), r, c):
            return out
        out.append((r, c))


def _nodes_ahead(pos: Coord, direction: Dir, rows: list[int], cols: list[int]) -> list[Coord]:
    r, c = pos
    if direction == (1, 0):
        return [(rr, c) for rr in rows if rr > r]
    if direction == (-1, 0):
        return [(rr, c) for rr in reversed(rows) if rr < r]
    if direction == (0, 1):
        return [(r, cc) for cc in cols if cc > c]
    assert direction == (0, -1)
    return [(r, cc) for cc in reversed(cols) if cc < c]


def _search_path(
    pos: Coord,
    direction: Dir,
    turns_left: int,
    rows: list[int],
    cols: list[int],
    shape: tuple[int, int],
    occupied: set[Coord],
) -> list[tuple[Coord, Dir]] | None:
    if turns_left == 0:
        exit_cells = _exit_cells(pos, direction, shape)
        if len(exit_cells) < 2 or any(cell in occupied for cell in exit_cells):
            return None
        return []

    candidates = _nodes_ahead(pos, direction, rows, cols)
    random.shuffle(candidates)
    for node in candidates:
        seg = _segment_cells(pos, node)
        if len(seg) < 2 or any(cell in occupied for cell in seg):
            continue
        occupied.update(seg)
        out_dirs = _perpendiculars(direction)
        random.shuffle(out_dirs)
        for out_dir in out_dirs:
            tail = _search_path(node, out_dir, turns_left - 1, rows, cols, shape, occupied)
            if tail is not None:
                return [(node, out_dir)] + tail
        for cell in seg:
            occupied.remove(cell)
    return None


def _build_plan() -> dict | None:
    h = random.randint(MIN_SIDE, MAX_SIDE)
    w = random.randint(MIN_SIDE, MAX_SIDE)
    rows = _line_candidates(h)
    cols = _line_candidates(w)
    if len(rows) < 4 or len(cols) < 4:
        return None

    edge = random.choice(["top", "bottom", "left", "right"])
    if edge == "top":
        marker = (0, random.choice(cols))
        start_dir = (1, 0)
    elif edge == "bottom":
        marker = (h - 1, random.choice(cols))
        start_dir = (-1, 0)
    elif edge == "left":
        marker = (random.choice(rows), 0)
        start_dir = (0, 1)
    else:
        marker = (random.choice(rows), w - 1)
        start_dir = (0, -1)

    max_turns = min(MAX_REFLECTIONS, len(rows) + len(cols) - 3)
    if max_turns < MIN_REFLECTIONS:
        return None
    turns = random.randint(MIN_REFLECTIONS, max_turns)
    reflections = _search_path(marker, start_dir, turns, rows, cols, (h, w), {marker})
    if reflections is None:
        return None

    return {
        "shape": (h, w),
        "marker": marker,
        "start_dir": start_dir,
        "reflections": reflections,
    }


def _segment_midpoint_cell(start: Coord, end: Coord) -> Coord | None:
    cells = _segment_cells(start, end)
    bg_cells = cells[:-1]
    if len(bg_cells) < 3:
        return None
    return bg_cells[len(bg_cells) // 2]


def _diag_steps(orient: str) -> list[Dir]:
    if orient == "\\":
        return [(-1, -1), (1, 1)]
    return [(-1, 1), (1, -1)]


def _place_segments(
    shape: tuple[int, int],
    path_cells: set[Coord],
    specs: list[dict],
) -> dict[Coord, tuple[int, str]] | None:
    h, w = shape
    mirror_map: dict[Coord, tuple[int, str]] = {}
    occupied: set[Coord] = set()

    for spec in specs:
        pos = spec["pos"]
        orient = spec["orient"]
        options: list[list[Coord]] = []
        lengths = [2, 3]
        random.shuffle(lengths)
        dirs = _diag_steps(orient)
        random.shuffle(dirs)
        for length in lengths:
            for dr, dc in dirs:
                cells = [(pos[0] + k * dr, pos[1] + k * dc) for k in range(length)]
                if not all(0 < r < h - 1 and 0 < c < w - 1 for r, c in cells):
                    continue
                if any(cell in occupied for cell in cells):
                    continue
                if any(cell in path_cells and cell != pos for cell in cells):
                    continue
                bad_contact = False
                cell_set = set(cells)
                for r, c in cells:
                    for nr, nc in _neighbors8(r, c):
                        if (nr, nc) in occupied and (nr, nc) not in cell_set:
                            bad_contact = True
                            break
                    if bad_contact:
                        break
                if not bad_contact:
                    options.append(cells)
        if not options:
            return None
        cells = random.choice(options)
        for cell in cells:
            mirror_map[cell] = (int(spec["color"]), orient)
        occupied.update(cells)

    return mirror_map


def _render_candidate(plan: dict) -> np.ndarray | None:
    h, w = plan["shape"]
    marker = plan["marker"]
    start_dir = plan["start_dir"]
    reflections = plan["reflections"]
    n_ref = len(reflections)
    if n_ref < MIN_REFLECTIONS:
        return None

    palette = random.sample(NOT_BLACK, random.randint(4, 6))
    marker_color = random.choice(palette)
    reflect_colors: list[int] = []
    current = marker_color
    for _ in range(n_ref):
        choices = [color for color in palette if color != current]
        if not choices:
            return None
        nxt = random.choice(choices)
        reflect_colors.append(nxt)
        current = nxt
    if len(set([marker_color] + reflect_colors)) < 3:
        return None

    specs: list[dict] = []
    path_cells = {marker}

    incoming = start_dir
    prev = marker
    reflective_orients: list[str] = []
    between_nodes: list[tuple[int, Coord, Coord]] = []
    for idx, (node, out_dir) in enumerate(reflections):
        path_cells.update(_segment_cells(prev, node))
        orient = _turn_to_orient(incoming, out_dir)
        reflective_orients.append(orient)
        specs.append(
            {
                "kind": "reflect",
                "pos": node,
                "orient": orient,
                "color": reflect_colors[idx],
            }
        )
        if idx + 1 < n_ref:
            between_nodes.append((idx, node, reflections[idx + 1][0]))
        prev = node
        incoming = out_dir
    path_cells.update(_exit_cells(reflections[-1][0], reflections[-1][1], (h, w)))

    if not between_nodes:
        return None

    pass_count = random.randint(1, min(2, len(between_nodes)))
    pass_segments = random.sample(between_nodes, pass_count)
    needed_orients = {"/", "\\"} - set(reflective_orients)
    pass_specs: list[dict] = []
    for seg_idx, start, end in pass_segments:
        pos = _segment_midpoint_cell(start, end)
        if pos is None:
            return None
        orient = random.choice(["/", "\\"])
        if needed_orients:
            orient = needed_orients.pop()
        pass_specs.append(
            {
                "kind": "pass",
                "pos": pos,
                "orient": orient,
                "color": reflect_colors[seg_idx],
            }
        )

    if needed_orients:
        return None
    specs.extend(pass_specs)
    random.shuffle(specs)

    mirror_map = _place_segments((h, w), path_cells, specs)
    if mirror_map is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[marker] = marker_color
    for (r, c), (color, _orient) in mirror_map.items():
        grid[r, c] = color
    return grid


def _wrong_variant_differs(grid: np.ndarray, *, same_color_transparent: bool, recolor_on_reflect: bool) -> bool:
    try:
        wrong, _stats = _trace_ray(
            grid,
            same_color_transparent=same_color_transparent,
            recolor_on_reflect=recolor_on_reflect,
        )
    except Exception:
        return True
    correct = main(grid)
    return not np.array_equal(correct, wrong)


def _valid_generated(grid: np.ndarray) -> bool:
    try:
        out, stats = _trace_ray(grid)
    except Exception:
        return False

    if not np.array_equal(out, main(grid)):
        return False
    if stats["reflections"] < MIN_REFLECTIONS:
        return False
    if stats["pass_throughs"] < 1:
        return False
    if stats["painted"] < 6:
        return False
    if not any(event["kind"] == "pass" and event["reflections_before"] > 0 for event in stats["events"]):
        return False
    if {event["orient"] for event in stats["events"]} != {"/", "\\"}:
        return False
    if not _wrong_variant_differs(grid, same_color_transparent=False, recolor_on_reflect=True):
        return False
    if not _wrong_variant_differs(grid, same_color_transparent=True, recolor_on_reflect=False):
        return False

    marker, mirrors = _marker_and_mirrors(grid)
    if any(cell == marker for cell in mirrors):
        return False
    for (r, c), (_color, _orient) in mirrors.items():
        if out[r, c] != grid[r, c]:
            return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        plan = _build_plan()
        if plan is None:
            continue
        grid = _render_candidate(plan)
        if grid is None:
            continue
        if _valid_generated(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
