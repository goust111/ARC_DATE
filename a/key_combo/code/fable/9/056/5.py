from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CANVAS_SIZE = 30
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(cell: tuple[int, int], step: tuple[int, int], k: int = 1) -> tuple[int, int]:
    return cell[0] + step[0] * k, cell[1] + step[1] * k


def _diag_neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [_add(cell, step) for step in DIAGONALS]


def _cardinal_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _box_cells(r0: int, c0: int, h: int, w: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    border: set[tuple[int, int]] = set()
    interior: set[tuple[int, int]] = set()
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            if r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1):
                border.add((r, c))
            else:
                interior.add((r, c))
    return border, interior


def _box_from_endpoint(endpoint: tuple[int, int], last_dir: tuple[int, int], size: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    dr, dc = last_dir
    if (dr, dc) == (1, 1):
        r0, c0 = endpoint[0] + 1, endpoint[1] + 1
    elif (dr, dc) == (1, -1):
        r0, c0 = endpoint[0] + 1, endpoint[1] - size
    elif (dr, dc) == (-1, 1):
        r0, c0 = endpoint[0] - size, endpoint[1] + 1
    else:
        assert (dr, dc) == (-1, -1)
        r0, c0 = endpoint[0] - size, endpoint[1] - size
    return _box_cells(r0, c0, size, size)


def _build_path(start: tuple[int, int], segments: list[tuple[tuple[int, int], int]]) -> list[tuple[int, int]]:
    path = [start]
    cur = start
    for step, length in segments:
        for _ in range(length):
            cur = _add(cur, step)
            path.append(cur)
    return path


def _has_bend(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return any(d0 != d1 for d0, d1 in zip(dirs, dirs[1:]))


def _nonconsecutive_diag_touch(path: list[tuple[int, int]]) -> bool:
    for i, a in enumerate(path):
        for j in range(i + 1, len(path)):
            if j == i + 1:
                continue
            b = path[j]
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 and abs(a[0] - b[0]) == abs(a[1] - b[1]):
                return True
    return False


def _crossing_cells(wire_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in wire_cells:
        if (
            (r - 1, c - 1) in wire_cells
            and (r + 1, c + 1) in wire_cells
            and (r - 1, c + 1) in wire_cells
            and (r + 1, c - 1) in wire_cells
        ):
            out.add((r, c))
    return out


def _detect_boxes(grid: np.ndarray) -> list[dict]:
    grey_mask = grid == GREY
    boxes: list[dict] = []
    for comp in _cardinal_components(grey_mask):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border, interior = _box_cells(r0, c0, h, w)
        if set(comp) != border:
            continue
        if not interior:
            continue
        if any(int(grid[r, c]) != BLACK for r, c in interior):
            continue
        boxes.append(
            {
                "border": border,
                "interior": interior,
                "bbox": (r0, c0, r1, c1),
            }
        )
    return boxes


def _trace_wire(
    source: tuple[int, int],
    wire_cells: set[tuple[int, int]],
    border_to_box: dict[tuple[int, int], int],
) -> tuple[bool, list[tuple[int, int]], int | None]:
    start_candidates = [cell for cell in _diag_neighbors(source) if cell in wire_cells]
    if len(start_candidates) != 1:
        return False, [], None

    prev = source
    current = start_candidates[0]
    direction = (current[0] - source[0], current[1] - source[1])
    path = [current]
    seen = {(current, direction)}

    while True:
        next_candidates = [cell for cell in _diag_neighbors(current) if cell in wire_cells and cell != prev]
        straight = _add(current, direction)
        if straight in next_candidates:
            nxt = straight
            next_direction = direction
        elif len(next_candidates) == 1:
            nxt = next_candidates[0]
            next_direction = (nxt[0] - current[0], nxt[1] - current[1])
        elif len(next_candidates) == 0:
            box_hits = {border_to_box[cell] for cell in _diag_neighbors(current) if cell in border_to_box}
            if len(box_hits) == 1:
                return True, path, next(iter(box_hits))
            return False, path, None
        else:
            return False, path, None

        state = (nxt, next_direction)
        if state in seen:
            return False, path, None
        seen.add(state)
        prev, current, direction = current, nxt, next_direction
        path.append(current)


def main(grid: np.ndarray) -> np.ndarray:
    """Send each source color through grey diagonal wire cells and light only the reachable hollow box.

    Input structure:
    - A black background contains several grey one-cell-wide diagonal chains, each using only diagonal adjacency.
    - Every chain starts next to a single colored source pixel and ends next to a grey hollow box with black interior.
    - Different chains may share one crossing cell, and hollow-box borders always stay grey.

    Rule:
    - For each source pixel, trace its wire cell by cell using only diagonal neighbors.
    - At a shared crossing cell, continue in the incoming diagonal direction instead of turning onto the other wire.
    - If the trace reaches a box endpoint, recolor that wire to the source color except for shared crossing cells, and fill the box interior with the same color.
    - If the wire is broken before the box, leave that wire grey and keep the box interior black.

    Key insight:
    - The grey cells form diagonal wiring rather than ordinary 4-neighbor components, so boxes light only after a direction-aware trace that goes straight through crossings and fails on hidden gaps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    boxes = _detect_boxes(g)
    border_to_box: dict[tuple[int, int], int] = {}
    for idx, box in enumerate(boxes):
        for cell in box["border"]:
            border_to_box[cell] = idx

    wire_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == GREY)} - set(border_to_box)
    crossings = _crossing_cells(wire_cells)
    sources = [tuple(map(int, cell)) for cell in np.argwhere((g != BLACK) & (g != GREY))]

    for source in sources:
        success, path, box_idx = _trace_wire(source, wire_cells, border_to_box)
        if not success or box_idx is None:
            continue
        color = int(g[source])
        for cell in path:
            if cell not in crossings:
                out[cell] = color
        for r, c in boxes[box_idx]["interior"]:
            out[r, c] = color

    return out


def _sample_crossing_wire(
    rng: np.random.Generator,
    cross: tuple[int, int],
    core_dir: tuple[int, int],
    *,
    force_bend: bool,
) -> list[tuple[int, int]] | None:
    before = int(rng.integers(2, 5))
    after = int(rng.integers(2, 5))
    turn_dirs = [step for step in DIAGONALS if step != core_dir and step != (-core_dir[0], -core_dir[1])]

    use_pre = force_bend or bool(rng.integers(0, 2))
    use_post = bool(rng.integers(0, 2))
    if force_bend and not (use_pre or use_post):
        use_pre = True

    pre_len = int(rng.integers(2, 5)) if use_pre else 0
    post_len = int(rng.integers(2, 5)) if use_post else 0
    pre_dir = turn_dirs[int(rng.integers(len(turn_dirs)))] if use_pre else None
    post_dir = turn_dirs[int(rng.integers(len(turn_dirs)))] if use_post else None

    core_start = _add(cross, core_dir, -before)
    start = _add(core_start, pre_dir, -pre_len) if pre_dir is not None else core_start
    segments: list[tuple[tuple[int, int], int]] = []
    if pre_dir is not None:
        segments.append((pre_dir, pre_len))
    segments.append((core_dir, before + after))
    if post_dir is not None:
        segments.append((post_dir, post_len))

    path = _build_path(start, segments)
    if cross not in path or len(set(path)) != len(path):
        return None
    if _nonconsecutive_diag_touch(path):
        return None
    return path


def _sample_free_wire(
    rng: np.random.Generator,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    *,
    must_break: bool,
) -> dict | None:
    for _ in range(120):
        nseg = 2 if bool(rng.integers(0, 2)) else 3
        dirs = [DIAGONALS[int(rng.integers(len(DIAGONALS)))]]
        while len(dirs) < nseg:
            prev = dirs[-1]
            options = [step for step in DIAGONALS if step != prev and step != (-prev[0], -prev[1])]
            dirs.append(options[int(rng.integers(len(options)))])
        lengths = [int(rng.integers(2, 5)) for _ in range(nseg)]
        start = (int(rng.integers(4, 26)), int(rng.integers(4, 26)))
        path = _build_path(start, list(zip(dirs, lengths)))
        if len(set(path)) != len(path) or _nonconsecutive_diag_touch(path):
            continue
        first_dir = dirs[0]
        last_dir = dirs[-1]
        source = _add(path[0], first_dir, -1)
        if not _in_bounds((CANVAS_SIZE, CANVAS_SIZE), source):
            continue
        size = int(rng.integers(3, 6))
        border, interior = _box_from_endpoint(path[-1], last_dir, size)
        if not all(_in_bounds((CANVAS_SIZE, CANVAS_SIZE), cell) for cell in border | interior | {source} | set(path)):
            continue
        blocked = set(path) | border | interior | {source}
        if any(cell in occupied or cell in reserved for cell in blocked):
            continue
        break_idx = None
        if must_break:
            if len(path) < 7:
                continue
            break_idx = int(rng.integers(2, len(path) - 2))
        return {
            "path": path,
            "source": source,
            "border": border,
            "interior": interior,
            "broken": must_break,
            "break_idx": break_idx,
        }
    return None


def _owner_map(scene: list[dict]) -> dict[tuple[int, int], set[int]]:
    owners: dict[tuple[int, int], set[int]] = {}
    for idx, wire in enumerate(scene):
        for cell in wire["path"]:
            owners.setdefault(cell, set()).add(idx)
    return owners


def _wire_cells_from_scene(scene: list[dict]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for wire in scene:
        for i, cell in enumerate(wire["path"]):
            if wire["broken"] and i == wire["break_idx"]:
                continue
            cells.add(cell)
    return cells


def _expected_output(inp: np.ndarray, scene: list[dict]) -> np.ndarray:
    out = inp.copy()
    crossings = _crossing_cells(_wire_cells_from_scene(scene))
    for wire in scene:
        if wire["broken"]:
            continue
        color = wire["color"]
        for cell in wire["path"]:
            if cell not in crossings:
                out[cell] = color
        for r, c in wire["interior"]:
            out[r, c] = color
    return out


def _crop_visible(
    inp: np.ndarray,
    out: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    coords = np.argwhere(inp != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    rr0 = max(0, int(r0) - top)
    cc0 = max(0, int(c0) - left)
    rr1 = min(inp.shape[0] - 1, int(r1) + bottom)
    cc1 = min(inp.shape[1] - 1, int(c1) + right)
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1], rr0, cc0


def _shift_scene(scene: list[dict], row_off: int, col_off: int) -> list[dict]:
    shifted: list[dict] = []
    for wire in scene:
        shifted.append(
            {
                **wire,
                "path": [(r - row_off, c - col_off) for r, c in wire["path"]],
                "source": (wire["source"][0] - row_off, wire["source"][1] - col_off),
                "border": {(r - row_off, c - col_off) for r, c in wire["border"]},
                "interior": {(r - row_off, c - col_off) for r, c in wire["interior"]},
            }
        )
    return shifted


def _validate_scene(inp: np.ndarray, out: np.ndarray, scene: list[dict]) -> bool:
    if np.array_equal(inp, out):
        return False
    if inp.ndim != 2 or out.ndim != 2 or max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    broken_count = sum(int(wire["broken"]) for wire in scene)
    success_count = len(scene) - broken_count
    if broken_count < 1 or success_count < 2:
        return False
    if not any(_has_bend(wire["path"]) for wire in scene if not wire["broken"]):
        return False

    owners = _owner_map(scene)
    crossings = _crossing_cells(_wire_cells_from_scene(scene))
    multi_owner = {cell for cell, who in owners.items() if len(who) > 1}
    if not crossings or multi_owner != crossings:
        return False

    wire_cells = _wire_cells_from_scene(scene)
    for cell in wire_cells:
        deg = sum(int(_add(cell, step) in wire_cells) for step in DIAGONALS)
        if deg not in (1, 2, 4):
            return False
        if deg == 4 and cell not in crossings:
            return False

    boxes = _detect_boxes(inp)
    if len(boxes) != len(scene):
        return False

    source_cells = [tuple(map(int, cell)) for cell in np.argwhere((inp != BLACK) & (inp != GREY))]
    if len(source_cells) != len(scene):
        return False
    for source in source_cells:
        if sum(int(nei in wire_cells) for nei in _diag_neighbors(source)) != 1:
            return False

    for box in boxes:
        adjacent_wire_endpoints = sum(int(nei in wire_cells) for cell in box["border"] for nei in _diag_neighbors(cell))
        if adjacent_wire_endpoints < 1:
            return False

    border_to_box = {}
    for idx, box in enumerate(boxes):
        for cell in box["border"]:
            border_to_box[cell] = idx

    painted_boxes = 0
    for source in source_cells:
        success, path, box_idx = _trace_wire(source, wire_cells, border_to_box)
        if success:
            painted_boxes += 1
            color = int(inp[source])
            if box_idx is None:
                return False
            if not all(int(out[r, c]) == color for r, c in boxes[box_idx]["interior"]):
                return False
        else:
            color = int(inp[source])
            if any(int(out[cell]) == color for cell in path if cell not in crossings):
                return False
    if painted_boxes < 2:
        return False

    crossing_colors = {
        int(inp[wire["source"]])
        for wire in scene
        if not wire["broken"] and any(cell in crossings for cell in wire["path"])
    }
    if len(crossing_colors) < 2:
        return False

    return True


def _build_input(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(colors)

    cross = (int(rng.integers(10, 20)), int(rng.integers(10, 20)))
    core_a = DIAGONALS[int(rng.integers(len(DIAGONALS)))]
    core_b_options = [step for step in DIAGONALS if step[0] * step[1] == -core_a[0] * core_a[1]]
    core_b = core_b_options[int(rng.integers(len(core_b_options)))]

    path_a = _sample_crossing_wire(rng, cross, core_a, force_bend=True)
    path_b = _sample_crossing_wire(rng, cross, core_b, force_bend=False)
    if path_a is None or path_b is None:
        return None
    if set(path_a) & set(path_b) != {cross}:
        return None

    size_a = int(rng.integers(3, 6))
    size_b = int(rng.integers(3, 6))
    source_a = _add(path_a[0], (path_a[1][0] - path_a[0][0], path_a[1][1] - path_a[0][1]), -1)
    source_b = _add(path_b[0], (path_b[1][0] - path_b[0][0], path_b[1][1] - path_b[0][1]), -1)
    border_a, interior_a = _box_from_endpoint(path_a[-1], (path_a[-1][0] - path_a[-2][0], path_a[-1][1] - path_a[-2][1]), size_a)
    border_b, interior_b = _box_from_endpoint(path_b[-1], (path_b[-1][0] - path_b[-2][0], path_b[-1][1] - path_b[-2][1]), size_b)

    scene = [
        {
            "path": path_a,
            "source": source_a,
            "border": border_a,
            "interior": interior_a,
            "broken": False,
            "break_idx": None,
            "color": colors[0],
        },
        {
            "path": path_b,
            "source": source_b,
            "border": border_b,
            "interior": interior_b,
            "broken": False,
            "break_idx": None,
            "color": colors[1],
        },
    ]

    all_reserved = set()
    for wire in scene:
        all_reserved |= set(wire["path"]) | wire["border"] | wire["interior"] | {wire["source"]}
    if not all(_in_bounds((CANVAS_SIZE, CANVAS_SIZE), cell) for cell in all_reserved):
        return None

    extra_broken = _sample_free_wire(rng, occupied=_wire_cells_from_scene(scene), reserved=all_reserved, must_break=True)
    if extra_broken is None:
        return None
    extra_broken["color"] = colors[2]
    scene.append(extra_broken)

    all_reserved |= set(extra_broken["path"]) | extra_broken["border"] | extra_broken["interior"] | {extra_broken["source"]}
    occupied = _wire_cells_from_scene(scene)
    if bool(rng.integers(0, 2)):
        extra_success = _sample_free_wire(rng, occupied=occupied, reserved=all_reserved, must_break=False)
        if extra_success is None:
            return None
        extra_success["color"] = colors[3]
        scene.append(extra_success)

    owners = _owner_map(scene)
    if {cell for cell, who in owners.items() if len(who) > 1} != {cross}:
        return None

    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for wire in scene:
        for r, c in wire["border"]:
            inp[r, c] = GREY
    for wire in scene:
        for i, (r, c) in enumerate(wire["path"]):
            if wire["broken"] and i == wire["break_idx"]:
                continue
            inp[r, c] = GREY
    for wire in scene:
        sr, sc = wire["source"]
        inp[sr, sc] = wire["color"]

    out = _expected_output(inp, scene)
    if not _validate_scene(inp, out, scene):
        return None
    inp, out, row_off, col_off = _crop_visible(inp, out, rng)
    shifted_scene = _shift_scene(scene, row_off, col_off)
    if not np.array_equal(main(inp), out):
        return None
    if not _validate_scene(inp, out, shifted_scene):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        built = _build_input(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid diagonal-wire lighting input")
