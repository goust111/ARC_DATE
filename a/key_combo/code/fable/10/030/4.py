from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _border_cells(box: BBox) -> list[Coord]:
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _expanded_box_cells(shape: tuple[int, int], box: BBox, pad: int = 1) -> list[Coord]:
    h, w = shape
    r0, c0, r1, c1 = box
    cells: list[Coord] = []
    for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1):
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1):
            cells.append((r, c))
    return cells


def _gate_side(box: BBox, gate: Coord) -> str:
    r0, c0, r1, c1 = box
    r, c = gate
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gate must be a non-corner border cell")


def _gate_vector(side: str) -> Coord:
    if side == "top":
        return (-1, 0)
    if side == "bottom":
        return (1, 0)
    if side == "left":
        return (0, -1)
    if side == "right":
        return (0, 1)
    raise AssertionError("unknown gate side")


def _outward_cells(shape: tuple[int, int], gate: Coord, side: str) -> list[Coord]:
    h, w = shape
    dr, dc = _gate_vector(side)
    r, c = gate
    cells: list[Coord] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        border = set(_border_cells(box))
        color_cells = {tuple(map(int, cell)) for cell in coords}
        if not color_cells.issubset(border):
            continue
        missing = sorted(border - color_cells)
        if len(missing) != 1:
            continue
        gate = tuple(map(int, missing[0]))
        try:
            side = _gate_side(box, gate)
        except AssertionError:
            continue
        if len(color_cells) != len(border) - 1:
            continue
        frames.append(
            {
                "color": color,
                "bbox": box,
                "gate": gate,
                "side": side,
                "corridor": _outward_cells(g.shape, gate, side),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    assert frames, "expected at least one gate frame"
    return frames


def _infer_marker_color(grid: np.ndarray, bg: int, frames: list[dict]) -> int:
    frame_colors = {int(frame["color"]) for frame in frames}
    candidates: list[int] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in frame_colors | {bg}):
        comps = _components(grid == color)
        if comps and all(len(comp) == 1 for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, "expected exactly one singleton marker color"
    return candidates[0]


def _gate_admission_count(grid: np.ndarray, bg: int, marker_color: int, frame: dict) -> int:
    for r, c in frame["corridor"]:
        value = int(grid[r, c])
        if value == bg:
            continue
        return int(value == marker_color)
    return 0


def _render_from_counts(grid: np.ndarray, bg: int, marker_color: int, frames: list[dict], counts: dict[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[out == marker_color] = bg
    for frame in frames:
        count = int(counts[frame["color"]])
        r0, c0, r1, c1 = frame["bbox"]
        fillable = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(out[r, c]) == bg
        ]
        assert 0 <= count <= len(fillable), "frame tally must fit inside its interior"
        for r, c in fillable[:count]:
            out[r, c] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each gate frame's interior markers plus the outside marker that can enter through its gap, then render that tally back inside the frame.

    Input structure:
    - The background contains several one-cell-thick hollow rectangular frames, each in its own color.
    - Every frame border is complete except for one missing non-corner border cell that acts as a gate.
    - One singleton marker color appears both inside frames and on the outside background, while other non-frame colors form blocker fragments.

    Rule:
    - For each frame, count the marker cells already inside its rectangle.
    - Then scan straight outward from that frame's gate: only the first non-background cell on that gate line can matter, and it contributes exactly when it is a marker.
    - Keep frames and blockers unchanged, erase all original marker cells, and refill the first N interior background cells of each frame in row-major order with the marker color.

    Key insight:
    - The gate is the only admission channel, so a frame's outside contribution is not "all aligned markers" but only the nearest visible object on the gate ray after accounting for blockers and nearer markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    frames = _detect_frames(g, bg)
    marker_color = _infer_marker_color(g, bg, frames)

    counts: dict[int, int] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside = int(np.count_nonzero(g[r0 + 1 : r1, c0 + 1 : c1] == marker_color))
        counts[frame["color"]] = inside + _gate_admission_count(g, bg, marker_color, frame)

    return _render_from_counts(g, bg, marker_color, frames, counts)


# --------------- generator ---------------

_BG = 0
_PALETTE = tuple(range(1, 10))
_FRAME_COUNT_RANGE = (2, 4)
_GRID_MIN = 18
_GRID_MAX = 29
_FRAME_SIZE_RANGE = range(5, 9)
_INTERIOR_MARKER_MAX = 6
_EXTRA_MARKERS_RANGE = (2, 6)
_EXTRA_BLOCKER_RANGE = (1, 4)
_ATTEMPTS = 500


def _randint_inclusive(lo: int, hi: int) -> int:
    return random.randint(lo, hi)


def _scenario_list(k: int) -> list[str]:
    base = ["admit", random.choice(["marker_block", "blocker_then_marker"])]
    if k >= 3:
        base.append("blocker_then_marker" if "blocker_then_marker" not in base else "marker_block")
    extras = ["admit", "none", "marker_block", "blocker_then_marker"]
    while len(base) < k:
        base.append(random.choice(extras))
    random.shuffle(base)
    return base


def _needed_margin(scenario: str) -> int:
    if scenario == "admit":
        return 2
    if scenario in {"marker_block", "blocker_then_marker"}:
        return 4
    return 1


def _candidate_gate_positions(box: BBox, side: str) -> list[Coord]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _box_ranges(h: int, w: int, fh: int, fw: int, side: str, need: int) -> tuple[range, range] | None:
    min_r0, max_r0 = 1, h - fh - 1
    min_c0, max_c0 = 1, w - fw - 1
    if side == "top":
        min_r0 = max(min_r0, need)
    if side == "bottom":
        max_r0 = min(max_r0, h - fh - need)
    if side == "left":
        min_c0 = max(min_c0, need)
    if side == "right":
        max_c0 = min(max_c0, w - fw - need)
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None
    return range(min_r0, max_r0 + 1), range(min_c0, max_c0 + 1)


def _place_frame_bbox(
    shape: tuple[int, int],
    reserved: np.ndarray,
    scenario: str,
) -> dict | None:
    h, w = shape
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        need = _needed_margin(scenario)
        dims = [(fh, fw) for fh in _FRAME_SIZE_RANGE for fw in _FRAME_SIZE_RANGE]
        random.shuffle(dims)
        for fh, fw in dims:
            ranges = _box_ranges(h, w, fh, fw, side, need)
            if ranges is None:
                continue
            r_range, c_range = ranges
            positions = [(r0, c0) for r0 in r_range for c0 in c_range]
            random.shuffle(positions)
            for r0, c0 in positions:
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                moat = _expanded_box_cells(shape, box, pad=1)
                if any(reserved[r, c] for r, c in moat):
                    continue
                gates = _candidate_gate_positions(box, side)
                random.shuffle(gates)
                for gate in gates:
                    corridor = _outward_cells(shape, gate, side)
                    if len(corridor) < need:
                        continue
                    if any(reserved[r, c] for r, c in corridor):
                        continue
                    return {
                        "bbox": box,
                        "gate": gate,
                        "side": side,
                        "corridor": corridor,
                        "moat": moat,
                        "scenario": scenario,
                    }
    return None


def _mark_reserved(reserved: np.ndarray, cells: list[Coord]) -> None:
    for r, c in cells:
        reserved[r, c] = True


def _can_place_marker(grid: np.ndarray, marker_cells: set[Coord], cell: Coord) -> bool:
    r, c = cell
    if int(grid[r, c]) != _BG:
        return False
    return all((nr, nc) not in marker_cells for nr, nc in _neighbors4(r, c))


def _place_marker(grid: np.ndarray, marker_cells: set[Coord], cell: Coord, marker_color: int) -> bool:
    if not _can_place_marker(grid, marker_cells, cell):
        return False
    grid[cell] = marker_color
    marker_cells.add(cell)
    return True


def _segment_candidates(shape: tuple[int, int], intercept: Coord, side: str) -> list[list[Coord]]:
    h, w = shape
    r, c = intercept
    choices: list[list[Coord]] = []
    if side in {"top", "bottom"}:
        for offsets in ((0, 1), (-1, 0), (-1, 0, 1)):
            cells = [(r, c + dc) for dc in offsets]
            if all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                choices.append(cells)
    else:
        for offsets in ((0, 1), (-1, 0), (-1, 0, 1)):
            cells = [(r + dr, c) for dr in offsets]
            if all(0 <= rr < h and 0 <= cc < w for rr, cc in cells):
                choices.append(cells)
    random.shuffle(choices)
    return choices


def _place_blocker_segment(
    grid: np.ndarray,
    reserved: np.ndarray,
    intercept: Coord,
    side: str,
    blocker_color: int,
) -> bool:
    for cells in _segment_candidates(grid.shape, intercept, side):
        ok = True
        for cell in cells:
            if int(grid[cell]) != _BG:
                ok = False
                break
            if cell != intercept and reserved[cell]:
                ok = False
                break
        if not ok:
            continue
        for cell in cells:
            grid[cell] = blocker_color
        return True
    return False


def _place_interior_markers(
    grid: np.ndarray,
    marker_cells: set[Coord],
    marker_color: int,
    box: BBox,
    target: int,
) -> int:
    r0, c0, r1, c1 = box
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    random.shuffle(cells)
    placed = 0
    for cell in cells:
        if _place_marker(grid, marker_cells, cell, marker_color):
            placed += 1
            if placed == target:
                return placed
    return placed


def _place_gate_scenario(
    grid: np.ndarray,
    reserved: np.ndarray,
    marker_cells: set[Coord],
    marker_color: int,
    blocker_colors: list[int],
    frame: dict,
) -> int | None:
    corridor = frame["corridor"]
    scenario = frame["scenario"]
    if scenario == "none":
        return 0

    if scenario == "admit":
        dist = _randint_inclusive(1, min(4, len(corridor)))
        return 1 if _place_marker(grid, marker_cells, corridor[dist - 1], marker_color) else None

    if scenario == "marker_block":
        max_near = min(3, len(corridor) - 1)
        if max_near < 1:
            return None
        near_dist = _randint_inclusive(1, max_near)
        far_dist = _randint_inclusive(near_dist + 1, min(len(corridor), near_dist + 3))
        if not _place_marker(grid, marker_cells, corridor[near_dist - 1], marker_color):
            return None
        if not _place_marker(grid, marker_cells, corridor[far_dist - 1], marker_color):
            return None
        return 1

    if scenario == "blocker_then_marker":
        max_block = min(3, len(corridor) - 1)
        if max_block < 1:
            return None
        block_dist = _randint_inclusive(1, max_block)
        far_dist = _randint_inclusive(block_dist + 1, min(len(corridor), block_dist + 3))
        blocker_color = random.choice(blocker_colors)
        if not _place_blocker_segment(grid, reserved, corridor[block_dist - 1], frame["side"], blocker_color):
            return None
        if not _place_marker(grid, marker_cells, corridor[far_dist - 1], marker_color):
            return None
        return 0

    raise AssertionError("unknown scenario")


def _random_free_marker(
    grid: np.ndarray,
    marker_cells: set[Coord],
    forbidden: set[Coord],
    marker_color: int,
) -> bool:
    cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    random.shuffle(cells)
    for cell in cells:
        if cell in forbidden:
            continue
        if _place_marker(grid, marker_cells, cell, marker_color):
            return True
    return False


def _random_blocker_segment(
    grid: np.ndarray,
    reserved: np.ndarray,
    blocker_colors: list[int],
    forbidden: set[Coord],
) -> bool:
    cells = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    random.shuffle(cells)
    for cell in cells:
        if cell in forbidden or int(grid[cell]) != _BG or reserved[cell]:
            continue
        side = random.choice(["top", "bottom", "left", "right"])
        if _place_blocker_segment(grid, reserved, cell, side, random.choice(blocker_colors)):
            return True
    return False


def _count_on_corridor(grid: np.ndarray, marker_color: int, corridor: list[Coord]) -> int:
    return sum(int(grid[r, c] == marker_color) for r, c in corridor)


def _build_shortcut_output(grid: np.ndarray, frames: list[dict], marker_color: int, bg: int, mode: str) -> np.ndarray:
    counts: dict[int, int] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == marker_color))
        if mode == "inside_only":
            counts[frame["color"]] = inside
        elif mode == "all_aligned":
            counts[frame["color"]] = inside + _count_on_corridor(grid, marker_color, frame["corridor"])
        else:
            raise AssertionError("bad shortcut mode")
    return _render_from_counts(grid, bg, marker_color, frames, counts)


def _validate_case(grid: np.ndarray, frame_specs: list[dict], marker_color: int) -> bool:
    try:
        bg = _most_common_color(grid)
        frames = _detect_frames(grid, bg)
        detected = {int(frame["color"]): frame for frame in frames}
        if len(detected) != len(frame_specs):
            return False
        singleton_candidates = []
        for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg and int(v) not in detected):
            comps = _components(grid == color)
            if comps and all(len(comp) == 1 for comp in comps):
                singleton_candidates.append(color)
        if singleton_candidates != [marker_color]:
            return False

        counts: dict[int, int] = {}
        has_admit = False
        has_blocked_alignment = False
        gate_ray_cells: set[Coord] = set()
        for spec in frame_specs:
            frame = detected.get(int(spec["color"]))
            if frame is None:
                return False
            if frame["bbox"] != spec["bbox"] or frame["gate"] != spec["gate"] or frame["side"] != spec["side"]:
                return False
            gate_ray_cells.update(frame["corridor"])
            r0, c0, r1, c1 = frame["bbox"]
            inside = int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == marker_color))
            if inside != spec["inside_count"]:
                return False
            admit = _gate_admission_count(grid, bg, marker_color, frame)
            if admit != spec["outside_count"]:
                return False
            has_admit |= admit == 1
            has_blocked_alignment |= _count_on_corridor(grid, marker_color, frame["corridor"]) > admit
            counts[frame["color"]] = inside + admit

        if not has_admit or not has_blocked_alignment:
            return False

        marker_positions = {tuple(map(int, p)) for p in np.argwhere(grid == marker_color)}
        if not any(cell not in gate_ray_cells for cell in marker_positions):
            return False

        expected = _render_from_counts(grid, bg, marker_color, frames, counts)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(expected, _build_shortcut_output(grid, frames, marker_color, bg, "inside_only")):
            return False
        if np.array_equal(expected, _build_shortcut_output(grid, frames, marker_color, bg, "all_aligned")):
            return False
        return not np.array_equal(expected, grid)
    except AssertionError:
        return False


def _attempt() -> np.ndarray | None:
    frame_count = random.randint(*_FRAME_COUNT_RANGE)
    size_min = max(_GRID_MIN, 18 + 2 * frame_count)
    h = random.randint(size_min, _GRID_MAX)
    w = random.randint(size_min, _GRID_MAX)
    grid = np.full((h, w), _BG, dtype=int)

    usable_colors = list(_PALETTE)
    marker_color = random.choice(usable_colors)
    usable_colors.remove(marker_color)
    frame_colors = random.sample(usable_colors, frame_count)
    remaining = [color for color in usable_colors if color not in frame_colors]
    blocker_color_count = random.randint(1, min(3, len(remaining)))
    blocker_colors = random.sample(remaining, blocker_color_count)

    reserved = np.zeros((h, w), dtype=bool)
    marker_cells: set[Coord] = set()
    frame_specs: list[dict] = []

    for color, scenario in zip(frame_colors, _scenario_list(frame_count)):
        placement = _place_frame_bbox((h, w), reserved, scenario)
        if placement is None:
            return None
        box = placement["bbox"]
        gate = placement["gate"]
        side = placement["side"]

        _mark_reserved(reserved, placement["moat"])
        _mark_reserved(reserved, placement["corridor"])

        r0, c0, r1, c1 = box
        for r, c in _border_cells(box):
            if (r, c) != gate:
                grid[r, c] = color

        interior_area = (r1 - r0 - 1) * (c1 - c0 - 1)
        max_inside = min(_INTERIOR_MARKER_MAX, interior_area - 1)
        inside_target = random.randint(0, max_inside)
        inside_count = _place_interior_markers(grid, marker_cells, marker_color, box, inside_target)
        if inside_count != inside_target:
            return None

        outside_count = _place_gate_scenario(grid, reserved, marker_cells, marker_color, blocker_colors, placement)
        if outside_count is None:
            return None

        frame_specs.append(
            {
                "color": color,
                "bbox": box,
                "gate": gate,
                "side": side,
                "inside_count": inside_count,
                "outside_count": outside_count,
                "scenario": scenario,
            }
        )

    forbidden = {tuple(map(int, p)) for frame in frame_specs for p in _outward_cells(grid.shape, frame["gate"], frame["side"])}
    extra_markers = random.randint(*_EXTRA_MARKERS_RANGE)
    for _ in range(extra_markers):
        if not _random_free_marker(grid, marker_cells, forbidden, marker_color):
            return None

    extra_blockers = random.randint(*_EXTRA_BLOCKER_RANGE)
    for _ in range(extra_blockers):
        _random_blocker_segment(grid, reserved, blocker_colors, forbidden)

    return grid if _validate_case(grid, frame_specs, marker_color) else None


def generate_input() -> np.ndarray:
    for _ in range(_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-admission counting input")
