from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS) not in sys.path:
    sys.path.append(str(_ARC_UTILS))

from arc_utils import Color, bbox_coords, colors_present, components_mask, get_bg


DIRS: dict[str, tuple[int, int]] = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
LEFT_TURN = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT_TURN = {"N": "E", "E": "S", "S": "W", "W": "N"}
REFLECT_SLASH = {"N": "E", "E": "N", "S": "W", "W": "S"}
REFLECT_BACKSLASH = {"N": "W", "W": "N", "S": "E", "E": "S"}

START_PATTERNS: dict[str, dict[str, tuple[tuple[int, int], ...]]] = {
    "E": {"cap": ((-1, -1), (0, -1)), "empty": ((1, -1), (1, 0))},
    "S": {"cap": ((-1, -1), (-1, 0)), "empty": ((0, 1), (-1, 1))},
    "W": {"cap": ((-1, 1), (0, 1)), "empty": ((-1, 0), (1, 1))},
    "N": {"cap": ((1, 0), (1, 1)), "empty": ((0, 1), (1, -1))},
}

COLOR_POOL = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
FRAME_COUNT_OPTIONS = (1, 2, 3)
MIRROR_COUNT_OPTIONS = (0, 1, 2)
MIRROR_COUNT_WEIGHTS = (0.25, 0.45, 0.30)
GRID_PAD_RANGE = (1, 4)
FINAL_GAP_RANGE = (4, 8)
MID_GAP_RANGE = (2, 5)
FRAME_INTERIOR_RANGE = (2, 5)
FRAME_MINOR_RANGE = (3, 5)
GEN_RETRIES = 400
BG = Color.BLACK


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _mul(d: tuple[int, int], k: int) -> tuple[int, int]:
    return d[0] * k, d[1] * k


def _reflect(direction: str, orient: str) -> str:
    if orient == "/":
        return REFLECT_SLASH[direction]
    if orient == "\\":
        return REFLECT_BACKSLASH[direction]
    raise ValueError(f"unknown mirror orientation: {orient}")


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _start_and_cap(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], str, int, int]:
    H, W = grid.shape
    candidates: list[tuple[tuple[int, int], str, int, int]] = []
    for r in range(H):
        for c in range(W):
            start_color = int(grid[r, c])
            if start_color in (bg, Color.GREY):
                continue
            if int(np.count_nonzero(grid == start_color)) != 1:
                continue
            for direction, spec in START_PATTERNS.items():
                cap_cells = []
                ok = True
                for dr, dc in spec["cap"]:
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        ok = False
                        break
                    cap_cells.append((rr, cc))
                if not ok:
                    continue
                cap_color = int(grid[cap_cells[0]])
                if cap_color in (bg, Color.GREY, start_color):
                    continue
                if any(int(grid[rr, cc]) != cap_color for rr, cc in cap_cells):
                    continue
                if int(np.count_nonzero(grid == cap_color)) != 2:
                    continue
                for dr, dc in spec["empty"]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) != bg:
                        ok = False
                        break
                dr, dc = DIRS[direction]
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W) or int(grid[rr, cc]) != bg:
                    ok = False
                if ok:
                    candidates.append(((r, c), direction, start_color, cap_color))
    assert len(candidates) == 1, "expected exactly one valid start marker"
    return candidates[0]


def _parse_frames(
    grid: np.ndarray, bg: int, start_color: int, cap_color: int
) -> tuple[list[dict[str, object]], dict[tuple[int, int], dict[str, object]]]:
    frames: list[dict[str, object]] = []
    wall_lookup: dict[tuple[int, int], dict[str, object]] = {}
    skip = {bg, Color.GREY, start_color, cap_color}
    for color in colors_present(grid, exclude=skip):
        comps = components_mask(grid == color, connectivity=4)
        assert len(comps) == 1, "each frame color should form a single frame"
        coords = comps[0]
        r0, c0, r1, c1 = bbox_coords(coords)
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have non-empty interior"
        bbox = (r0, c0, r1, c1)
        expected = _perimeter_cells(bbox)
        actual = {tuple(map(int, cell)) for cell in coords.tolist()}
        assert actual == expected, "frame must be a 1-cell-thick hollow rectangle"
        frame = {"color": int(color), "bbox": bbox}
        frames.append(frame)
        for cell in actual:
            assert cell not in wall_lookup, "frame walls must not overlap"
            wall_lookup[cell] = frame
    assert frames, "expected at least one frame"
    return frames, wall_lookup


def _parse_mirrors(grid: np.ndarray) -> dict[tuple[int, int], str]:
    mirror_lookup: dict[tuple[int, int], str] = {}
    for coords in components_mask(grid == Color.GREY, connectivity=8):
        points = [tuple(map(int, cell)) for cell in coords.tolist()]
        assert len(points) == 2, "each mirror must contain exactly two pixels"
        (r0, c0), (r1, c1) = points
        dr, dc = r1 - r0, c1 - c0
        assert abs(dr) == 1 and abs(dc) == 1, "mirror pixels must be diagonal neighbors"
        orient = "\\" if dr == dc else "/"
        for cell in points:
            mirror_lookup[cell] = orient
    return mirror_lookup


def _trace(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    start, direction, current_color, cap_color = _start_and_cap(g, bg)
    _frames, wall_lookup = _parse_frames(g, bg, current_color, cap_color)
    mirror_lookup = _parse_mirrors(g)
    out = g.copy()
    H, W = g.shape

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < H and 0 <= cell[1] < W

    current = start
    for _ in range(H * W * 8):
        step = _add(current, DIRS[direction])
        if not in_bounds(step):
            return out
        if step in mirror_lookup:
            direction = _reflect(direction, mirror_lookup[step])
            next_cell = _add(step, DIRS[direction])
            assert in_bounds(next_cell), "mirror must reflect back into the grid"
            assert int(g[next_cell]) == bg, "reflected ray must land on background"
            current = next_cell
            out[current] = current_color
            continue
        if step in wall_lookup:
            frame = wall_lookup[step]
            next_cell = _add(step, DIRS[direction])
            assert in_bounds(next_cell), "frame crossing must stay inside the grid"
            assert int(g[next_cell]) == bg, "frame wall must lead to background on the far side"
            r0, c0, r1, c1 = frame["bbox"]
            inside = r0 < next_cell[0] < r1 and c0 < next_cell[1] < c1
            outside = not (r0 <= next_cell[0] <= r1 and c0 <= next_cell[1] <= c1)
            assert inside or outside, "ray must cross straight through a frame wall"
            if inside:
                current_color = int(frame["color"])
            current = next_cell
            out[current] = current_color
            continue
        assert int(g[step]) == bg, "ray may only travel through background"
        current = step
        out[current] = current_color
    raise AssertionError("ray tracing did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch the beam from the capped start, recolor it whenever it passes through a hollow frame, and reflect it at diagonal mirrors.

    Input structure:
    - The background is the dominant empty color.
    - There is one single-pixel emitter, and a neighboring two-pixel monochrome cap marks its initial launch direction.
    - The scene also contains one or more 1-cell-thick hollow rectangular frames of distinct colors and zero or more gray diagonal mirrors made of two diagonally touching pixels.
    - Frames, mirrors, start, and cap do not overlap, and the beam only ever meets frame walls perpendicularly.

    Rule:
    - Start a ray from the emitter in the cap-indicated direction, with the emitter's color as the initial beam color.
    - Draw the ray only on background cells.
    - When the ray crosses into a frame interior through one wall, switch the beam to that frame's color and keep that color until a later frame changes it again.
    - When the ray hits a gray diagonal mirror, turn it by 90 degrees according to the mirror's slash direction, keep the current beam color, and continue until the next step would leave the grid.

    Key insight:
    - The beam's visible color is the path state "most recent frame entered"; mirrors only redirect the route, while hollow frames act as serial color filters rather than obstacles.
    """
    return _trace(grid)


def _build_scene(
    rng: np.random.Generator,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], tuple[int, int], str]:
    start_color, cap_color = rng.choice(COLOR_POOL, size=2, replace=False).tolist()
    frame_count = int(rng.choice(FRAME_COUNT_OPTIONS))
    mirror_count = int(rng.choice(MIRROR_COUNT_OPTIONS, p=MIRROR_COUNT_WEIGHTS))
    needed = 2 + frame_count
    palette = rng.choice([c for c in COLOR_POOL if c not in {start_color, cap_color}], size=frame_count, replace=False).tolist()
    frame_colors = [int(c) for c in palette]

    segment_count = mirror_count + 1
    frame_bins = rng.multinomial(frame_count, np.full(segment_count, 1.0 / segment_count))
    turns = [LEFT_TURN if t == 0 else RIGHT_TURN for t in rng.integers(0, 2, size=mirror_count)]
    direction = rng.choice(list(DIRS.keys())).item()
    current = (0, 0)
    current_color = int(start_color)
    occupied: dict[tuple[int, int], int] = {current: int(start_color)}
    painted: dict[tuple[int, int], int] = {}
    forbidden_future = {current}

    for dr, dc in START_PATTERNS[direction]["cap"]:
        occupied[(current[0] + dr, current[1] + dc)] = int(cap_color)
        forbidden_future.add((current[0] + dr, current[1] + dc))

    def can_use_background(cell: tuple[int, int]) -> bool:
        return cell not in occupied and cell not in painted and cell not in forbidden_future

    def move_gap(steps: int) -> bool:
        nonlocal current
        dr, dc = DIRS[direction]
        for _ in range(steps):
            nxt = (current[0] + dr, current[1] + dc)
            if not can_use_background(nxt):
                return False
            painted[nxt] = current_color
            current = nxt
        return True

    frame_color_idx = 0
    for seg_idx in range(segment_count):
        if not move_gap(int(rng.integers(MID_GAP_RANGE[0], MID_GAP_RANGE[1] + 1))):
            raise ValueError("failed initial gap")
        frames_on_segment = int(frame_bins[seg_idx])
        for local_idx in range(frames_on_segment):
            dr, dc = DIRS[direction]
            interior_len = int(rng.integers(FRAME_INTERIOR_RANGE[0], FRAME_INTERIOR_RANGE[1] + 1))
            minor = int(rng.integers(FRAME_MINOR_RANGE[0], FRAME_MINOR_RANGE[1] + 1))
            offset = int(rng.integers(1, minor - 1))
            wall_in = (current[0] + dr, current[1] + dc)
            interior = [(current[0] + dr * k, current[1] + dc * k) for k in range(2, interior_len + 2)]
            wall_out = (current[0] + dr * (interior_len + 2), current[1] + dc * (interior_len + 2))
            outside1 = (current[0] + dr * (interior_len + 3), current[1] + dc * (interior_len + 3))
            if direction in ("E", "W"):
                row = interior[0][0]
                cols = [cell[1] for cell in interior] + [wall_in[1], wall_out[1]]
                left, right = min(cols), max(cols)
                top = row - offset
                bottom = top + minor - 1
            else:
                col = interior[0][1]
                rows = [cell[0] for cell in interior] + [wall_in[0], wall_out[0]]
                top, bottom = min(rows), max(rows)
                left = col - offset
                right = left + minor - 1
            bbox = (top, left, bottom, right)
            walls = _perimeter_cells(bbox)
            box_cells = _bbox_cells(bbox)
            if any(cell in occupied or cell in painted for cell in walls):
                raise ValueError("frame wall overlap")
            if any(cell in occupied or cell in painted or cell in forbidden_future for cell in interior + [outside1]):
                raise ValueError("frame path overlap")
            if any(cell in occupied for cell in box_cells - set(interior)):
                raise ValueError("frame bbox overlap")
            frame_color = frame_colors[frame_color_idx]
            frame_color_idx += 1
            for cell in walls:
                occupied[cell] = frame_color
            current_color = frame_color
            for cell in interior:
                painted[cell] = current_color
            painted[outside1] = current_color
            current = outside1
            forbidden_future |= box_cells
            if local_idx != frames_on_segment - 1:
                if not move_gap(int(rng.integers(MID_GAP_RANGE[0], MID_GAP_RANGE[1] + 1))):
                    raise ValueError("failed mid gap")
        if seg_idx == segment_count - 1:
            if not move_gap(int(rng.integers(FINAL_GAP_RANGE[0], FINAL_GAP_RANGE[1] + 1))):
                raise ValueError("failed final gap")
            break
        if not move_gap(int(rng.integers(MID_GAP_RANGE[0], MID_GAP_RANGE[1] + 1))):
            raise ValueError("failed pre-mirror gap")
        dr, dc = DIRS[direction]
        turn_map = turns[seg_idx]
        new_direction = turn_map[direction]
        dr2, dc2 = DIRS[new_direction]
        hit = (current[0] + dr, current[1] + dc)
        new_pos = (hit[0] + dr2, hit[1] + dc2)
        partner = (hit[0] + dr + dr2, hit[1] + dc + dc2)
        if any(cell in occupied or cell in painted or cell in forbidden_future for cell in (hit, partner, new_pos)):
            raise ValueError("mirror overlap")
        occupied[hit] = Color.GREY
        occupied[partner] = Color.GREY
        painted[new_pos] = current_color
        mirror_bbox = (min(hit[0], partner[0]), min(hit[1], partner[1]), max(hit[0], partner[0]), max(hit[1], partner[1]))
        forbidden_future |= _bbox_cells(mirror_bbox)
        current = new_pos
        direction = new_direction

    assert frame_color_idx == frame_count
    return occupied, painted, current, direction


def _embed_scene(
    occupied: dict[tuple[int, int], int],
    painted: dict[tuple[int, int], int],
    end_cell: tuple[int, int],
    end_direction: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    used = set(occupied) | set(painted)
    rows = [r for r, _ in used]
    cols = [c for _, c in used]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    if end_direction == "N":
        if min_r != end_cell[0]:
            raise ValueError("north-bound endpoint is not topmost")
        shift_r = -end_cell[0]
        H = max_r + shift_r + int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) + 1
    else:
        shift_r = int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) - min_r
        if end_direction == "S":
            if max_r != end_cell[0]:
                raise ValueError("south-bound endpoint is not bottommost")
            H = end_cell[0] + shift_r + 1
        else:
            H = max_r + shift_r + int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) + 1

    if end_direction == "W":
        if min_c != end_cell[1]:
            raise ValueError("west-bound endpoint is not leftmost")
        shift_c = -end_cell[1]
        W = max_c + shift_c + int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) + 1
    else:
        shift_c = int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) - min_c
        if end_direction == "E":
            if max_c != end_cell[1]:
                raise ValueError("east-bound endpoint is not rightmost")
            W = end_cell[1] + shift_c + 1
        else:
            W = max_c + shift_c + int(rng.integers(GRID_PAD_RANGE[0], GRID_PAD_RANGE[1] + 1)) + 1

    if max(H, W) > 30:
        raise ValueError("grid exceeds ARC size limit")

    inp = np.full((H, W), BG, dtype=int)
    exp = inp.copy()
    for (r, c), color in occupied.items():
        rr, cc = r + shift_r, c + shift_c
        inp[rr, cc] = color
        exp[rr, cc] = color
    for (r, c), color in painted.items():
        rr, cc = r + shift_r, c + shift_c
        exp[rr, cc] = color
    return inp, exp


def _validate_scene(inp: np.ndarray, exp: np.ndarray) -> bool:
    try:
        out = main(inp)
    except Exception:
        return False
    if not np.array_equal(out, exp):
        return False
    bg = int(get_bg(inp))
    if int(np.count_nonzero(out != inp)) < 6:
        return False
    if len(colors_present(inp, exclude={bg, Color.GREY})) < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        try:
            occupied, painted, end_cell, end_direction = _build_scene(rng)
            inp, exp = _embed_scene(occupied, painted, end_cell, end_direction, rng)
        except Exception:
            continue
        if _validate_scene(inp, exp):
            return inp
    raise RuntimeError("failed to generate a valid filter-light-path input")
