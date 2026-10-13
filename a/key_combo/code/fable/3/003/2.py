from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass

import numpy as np


BG = 0
TRACK = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        comp = [(sr, sc)]
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
                    comp.append((rr, cc))
        comps.append(comp)
    return comps


def _track_info(g: np.ndarray) -> tuple[str, int, int, int, list[tuple[int, int]]]:
    pts = [(int(r), int(c)) for r, c in np.argwhere(g == TRACK)]
    assert len(pts) >= 3, "expected at least three visible gray track dots"
    rows = {r for r, _c in pts}
    cols = {c for _r, c in pts}
    if len(rows) == 1:
        orient = "h"
        line = rows.pop()
        vals = sorted(c for _r, c in pts)
    elif len(cols) == 1:
        orient = "v"
        line = cols.pop()
        vals = sorted(r for r, _c in pts)
    else:
        raise AssertionError("gray dots must lie on one straight row or column")

    diffs = [b - a for a, b in zip(vals, vals[1:]) if b > a]
    assert diffs, "track needs more than one visible slot"
    spacing = int(diffs[0])
    for d in diffs[1:]:
        spacing = math.gcd(spacing, int(d))
    assert spacing >= 2, "track slot spacing must be visible"
    phase = vals[0] % spacing
    assert all(v % spacing == phase for v in vals), "visible dots must share one track phase"
    return orient, int(line), int(spacing), int(phase), pts


def _track_points(
    shape: tuple[int, int], orient: str, line: int, spacing: int, phase: int
) -> list[tuple[int, int]]:
    h, w = shape
    pts = []
    if orient == "h":
        assert 0 <= line < h
        for c in range(int(phase), w, int(spacing)):
            pts.append((int(line), int(c)))
    else:
        assert 0 <= line < w
        for r in range(int(phase), h, int(spacing)):
            pts.append((int(r), int(line)))
    return pts


def _frame_components(g: np.ndarray) -> list[tuple[int, list[tuple[int, int]], tuple[int, int, int, int]]]:
    mask = (g != BG) & (g != TRACK)
    frames = []
    for comp in _components_mask(mask):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each frame component must be monochrome"
        color = colors.pop()
        frames.append((int(color), comp, _bbox(comp)))
    frames.sort(key=lambda item: (item[2][0], item[2][1], item[0]))
    assert frames, "expected at least one colored frame"
    return frames


def _door_direction(
    g: np.ndarray, color: int, box: tuple[int, int, int, int], orient: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if orient == "h":
        left_gap = [(r, c0) for r in range(r0 + 1, r1) if int(g[r, c0]) != color]
        right_gap = [(r, c1) for r in range(r0 + 1, r1) if int(g[r, c1]) != color]
        if len(left_gap) == 1 and len(right_gap) == 0:
            return 0, -1
        if len(right_gap) == 1 and len(left_gap) == 0:
            return 0, 1
    else:
        top_gap = [(r0, c) for c in range(c0 + 1, c1) if int(g[r0, c]) != color]
        bottom_gap = [(r1, c) for c in range(c0 + 1, c1) if int(g[r1, c]) != color]
        if len(top_gap) == 1 and len(bottom_gap) == 0:
            return -1, 0
        if len(bottom_gap) == 1 and len(top_gap) == 0:
            return 1, 0
    raise AssertionError("expected exactly one outer door gap on a track-facing side")


def _count_closed_rooms(g: np.ndarray, color: int, box: tuple[int, int, int, int]) -> int:
    h, w = g.shape
    r0, c0, r1, c1 = box
    er0, ec0 = max(0, r0 - 1), max(0, c0 - 1)
    er1, ec1 = min(h - 1, r1 + 1), min(w - 1, c1 + 1)
    walls = (g[er0 : er1 + 1, ec0 : ec1 + 1] == int(color))
    open_mask = ~walls
    ch, cw = open_mask.shape

    count = 0
    for comp in _components_mask(open_mask):
        touches_edge = any(r == 0 or c == 0 or r == ch - 1 or c == cw - 1 for r, c in comp)
        if not touches_edge:
            count += 1
    return int(count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each door-marked partitioned frame along the gray-dot track by its sealed room count.

    Input structure:
    - Black background with a single straight row or column of equally spaced gray track dots.
    - One or more monochrome rectangular partition frames sit on that track; their wall pixels may cover some dots.
    - Internal same-color walls divide each frame into cells, with one-cell inner door gaps on some walls.
    - Each frame has exactly one one-cell outer door gap on the left/right side for a horizontal track, or top/bottom side for a vertical track.

    Rule:
    - Reconstruct the full dotted track from the visible gray dots.
    - For each colored frame, treat only its own colored pixels as walls and count background regions enclosed by them.
    - Regions connected through the outer door gap to the outside are not counted; inner door gaps merge adjacent cells before counting.
    - Move the frame wall pixels k track slots toward the outer door side, where k is that sealed-room count, then redraw the restored track underneath.

    Key insight:
    - The outer door is both the direction arrow and an air leak, so the movement distance is the number of still-sealed rooms after all inner and outer gaps are respected.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    orient, line, spacing, phase, _visible = _track_info(g)
    frames = _frame_components(g)

    moved: list[tuple[int, list[tuple[int, int]]]] = []
    for color, comp, box in frames:
        dr, dc = _door_direction(g, color, box, orient)
        rooms = _count_closed_rooms(g, color, box)
        shifted = []
        for r, c in comp:
            rr = int(r + dr * spacing * rooms)
            cc = int(c + dc * spacing * rooms)
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "shifted frame must remain in bounds"
            shifted.append((rr, cc))
        moved.append((int(color), shifted))

    out = np.full_like(g, BG)
    for r, c in _track_points(g.shape, orient, line, spacing, phase):
        out[r, c] = TRACK

    occupied: set[tuple[int, int]] = set()
    for color, cells in moved:
        for r, c in cells:
            assert (r, c) not in occupied, "shifted frames must not overlap"
            occupied.add((r, c))
            out[r, c] = int(color)
    return out


@dataclass(frozen=True)
class FrameSpec:
    color: int
    walls: np.ndarray
    side: str
    direction: tuple[int, int]
    anchor: tuple[int, int]
    room_count: int
    total_cells: int
    leak_size: int
    largest_room_size: int
    door_count: int


def _line_positions(lengths: list[int]) -> list[int]:
    out = [0]
    for length in lengths:
        out.append(out[-1] + int(length) + 1)
    return out


def _cell_id(row: int, col: int, n_cols: int) -> int:
    return int(row) * int(n_cols) + int(col)


def _uf_find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return int(x)


def _uf_union(parent: list[int], a: int, b: int) -> bool:
    ra = _uf_find(parent, int(a))
    rb = _uf_find(parent, int(b))
    if ra == rb:
        return False
    parent[ra] = rb
    return True


def _count_rooms_from_walls(walls: np.ndarray) -> int:
    padded = np.pad(walls.astype(bool), 1, constant_values=False)
    open_mask = ~padded
    h, w = open_mask.shape
    total = 0
    for comp in _components_mask(open_mask):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            total += 1
    return int(total)


def _build_frame(orient: str, side: str, color: int) -> FrameSpec | None:
    if orient == "h":
        n_rows = random.choice([2, 3])
        n_cols = 2
        row_heights = [random.choice([1, 1, 2]) for _ in range(n_rows)]
        col_widths = [1, 1]
    else:
        n_rows = 2
        n_cols = random.choice([2, 3])
        row_heights = [1, 1]
        col_widths = [random.choice([1, 1, 2]) for _ in range(n_cols)]

    row_lines = _line_positions(row_heights)
    col_lines = _line_positions(col_widths)
    height, width = row_lines[-1] + 1, col_lines[-1] + 1
    walls = np.zeros((height, width), dtype=bool)
    walls[row_lines, :] = True
    walls[:, col_lines] = True

    if side in ("left", "right"):
        leak_row = random.randrange(n_rows)
        leak_col = 0 if side == "left" else n_cols - 1
        door_r = random.randrange(row_lines[leak_row] + 1, row_lines[leak_row + 1])
        door_c = 0 if side == "left" else width - 1
        direction = (0, -1) if side == "left" else (0, 1)
        anchor = (door_r, width - 1 if side == "left" else 0)
    else:
        leak_row = 0 if side == "top" else n_rows - 1
        leak_col = random.randrange(n_cols)
        door_r = 0 if side == "top" else height - 1
        door_c = random.randrange(col_lines[leak_col] + 1, col_lines[leak_col + 1])
        direction = (-1, 0) if side == "top" else (1, 0)
        anchor = (height - 1 if side == "top" else 0, door_c)
    walls[door_r, door_c] = False
    if not walls[anchor]:
        return None

    parent = list(range(n_rows * n_cols))
    candidates = []
    for r in range(1, n_rows):
        wall_r = row_lines[r]
        for c in range(n_cols):
            a = _cell_id(r - 1, c, n_cols)
            b = _cell_id(r, c, n_cols)
            door_c2 = random.randrange(col_lines[c] + 1, col_lines[c + 1])
            candidates.append((wall_r, door_c2, a, b))
    for c in range(1, n_cols):
        wall_c = col_lines[c]
        for r in range(n_rows):
            a = _cell_id(r, c - 1, n_cols)
            b = _cell_id(r, c, n_cols)
            door_r2 = random.randrange(row_lines[r] + 1, row_lines[r + 1])
            candidates.append((door_r2, wall_c, a, b))
    random.shuffle(candidates)

    target_doors = random.randint(1, min(3, len(candidates)))
    chosen = 0
    for rr, cc, a, b in candidates:
        if chosen >= target_doors:
            break
        if _uf_union(parent, a, b):
            walls[rr, cc] = False
            chosen += 1
    if chosen == 0:
        return None

    roots: dict[int, list[int]] = {}
    for idx in range(n_rows * n_cols):
        roots.setdefault(_uf_find(parent, idx), []).append(idx)
    leak = _cell_id(leak_row, leak_col, n_cols)
    leak_root = _uf_find(parent, leak)
    room_count = len([root for root in roots if root != leak_root])
    leak_size = len(roots[leak_root])
    largest_room_size = max([len(v) for root, v in roots.items() if root != leak_root] or [0])

    total_cells = n_rows * n_cols
    if not (1 <= room_count <= 3):
        return None
    if room_count > total_cells - 2:
        return None
    if _count_rooms_from_walls(walls) != room_count:
        return None

    return FrameSpec(
        color=int(color),
        walls=walls,
        side=side,
        direction=direction,
        anchor=anchor,
        room_count=int(room_count),
        total_cells=int(total_cells),
        leak_size=int(leak_size),
        largest_room_size=int(largest_room_size),
        door_count=int(chosen),
    )


def _frame_cells(spec: FrameSpec, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(spec.walls)}


def _fits(shape: tuple[int, int], spec: FrameSpec, r0: int, c0: int) -> bool:
    h, w = spec.walls.shape
    return 0 <= r0 and 0 <= c0 and r0 + h <= shape[0] and c0 + w <= shape[1]


def _place_specs(
    shape: tuple[int, int],
    orient: str,
    line: int,
    spacing: int,
    slots: list[int],
    specs: list[FrameSpec],
) -> list[tuple[int, int, int, int]] | None:
    candidates: list[list[tuple[int, int, int, int, set[tuple[int, int]], set[tuple[int, int]]]]] = []
    for spec in specs:
        item = []
        for slot_idx, slot in enumerate(slots):
            dest_idx = slot_idx + (spec.direction[1] if orient == "h" else spec.direction[0]) * spec.room_count
            if not (0 <= dest_idx < len(slots)):
                continue
            if orient == "h":
                r0 = line - spec.anchor[0]
                c0 = int(slot) - spec.anchor[1]
                dr, dc = 0, spec.direction[1] * spacing * spec.room_count
            else:
                r0 = int(slot) - spec.anchor[0]
                c0 = line - spec.anchor[1]
                dr, dc = spec.direction[0] * spacing * spec.room_count, 0
            r1, c1 = int(r0 + dr), int(c0 + dc)
            if not (_fits(shape, spec, r0, c0) and _fits(shape, spec, r1, c1)):
                continue
            start_cells = _frame_cells(spec, r0, c0)
            final_cells = _frame_cells(spec, r1, c1)
            item.append((slot_idx, int(r0), int(c0), int(dest_idx), start_cells, final_cells))
        random.shuffle(item)
        if not item:
            return None
        candidates.append(item)

    order = sorted(range(len(specs)), key=lambda i: len(candidates[i]))
    chosen: list[tuple[int, int, int, int] | None] = [None] * len(specs)
    used_start: set[tuple[int, int]] = set()
    used_final: set[tuple[int, int]] = set()

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for slot_idx, r0, c0, dest_idx, start_cells, final_cells in candidates[idx]:
            if start_cells & used_start:
                continue
            if final_cells & used_final:
                continue
            used_start.update(start_cells)
            used_final.update(final_cells)
            chosen[idx] = (slot_idx, r0, c0, dest_idx)
            if rec(pos + 1):
                return True
            for cell in start_cells:
                used_start.remove(cell)
            for cell in final_cells:
                used_final.remove(cell)
            chosen[idx] = None
        return False

    if not rec(0):
        return None
    return [item for item in chosen if item is not None]


def _render(
    shape: tuple[int, int],
    orient: str,
    line: int,
    spacing: int,
    phase: int,
    specs: list[FrameSpec],
    placements: list[tuple[int, int, int, int]],
    *,
    final: bool,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for r, c in _track_points(shape, orient, line, spacing, phase):
        out[r, c] = TRACK
    for spec, (_slot_idx, r0, c0, _dest_idx) in zip(specs, placements):
        rr0 = r0 + (0 if not final else spec.direction[0] * spacing * spec.room_count)
        cc0 = c0 + (0 if not final else spec.direction[1] * spacing * spec.room_count)
        rows, cols = np.where(spec.walls)
        out[rows + rr0, cols + cc0] = spec.color
    return out


def _valid_generation(
    inp: np.ndarray,
    expected: np.ndarray,
    orient: str,
    line: int,
    spacing: int,
    phase: int,
    specs: list[FrameSpec],
) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        got_orient, got_line, got_spacing, got_phase, visible = _track_info(inp)
    except Exception:
        return False
    if (got_orient, got_line, got_spacing, got_phase) != (orient, line, spacing, phase):
        return False
    if len(visible) < 3:
        return False
    if len({spec.room_count for spec in specs}) < min(2, len(specs)):
        return False
    if max(spec.room_count for spec in specs) < 2:
        return False
    if not any(spec.leak_size > 1 for spec in specs):
        return False
    if not any(spec.largest_room_size > 1 for spec in specs):
        return False
    signs = {spec.direction[1] if orient == "h" else spec.direction[0] for spec in specs}
    if signs != {-1, 1}:
        return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    orient = random.choice(["h", "v"])
    spacing = 6
    n_frames = random.choice([2, 2, 3])

    if orient == "h":
        rows = random.randint(18, 30)
        cols = 30
        phase = random.choice([2, 3])
        slots = list(range(phase, cols, spacing))
        if len(slots) < 5:
            return None
        line = random.randint(4, rows - 5)
        base_sides = ["left", "right"]
    else:
        rows = 30
        cols = random.randint(18, 30)
        phase = random.choice([2, 3])
        slots = list(range(phase, rows, spacing))
        if len(slots) < 5:
            return None
        line = random.randint(4, cols - 5)
        base_sides = ["top", "bottom"]

    if n_frames == 3 and len(slots) < 5:
        n_frames = 2
    sides = base_sides[:]
    while len(sides) < n_frames:
        sides.append(random.choice(base_sides))
    random.shuffle(sides)

    colors = random.sample(FRAME_COLORS, n_frames)
    specs = []
    for side, color in zip(sides, colors):
        spec = None
        for _ in range(60):
            spec = _build_frame(orient, side, color)
            if spec is not None:
                break
        if spec is None:
            return None
        specs.append(spec)

    placements = _place_specs((rows, cols), orient, line, spacing, slots, specs)
    if placements is None:
        return None

    inp = _render((rows, cols), orient, line, spacing, phase, specs, placements, final=False)
    expected = _render((rows, cols), orient, line, spacing, phase, specs, placements, final=True)
    if not _valid_generation(inp, expected, orient, line, spacing, phase, specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for outer-door room-count slide")
