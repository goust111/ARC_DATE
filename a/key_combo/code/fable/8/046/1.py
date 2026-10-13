import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np


Coord = tuple[int, int]


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    gap: Coord
    side: str


SIDES = ("top", "bottom", "left", "right")
SIDE_TO_OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if (
                        _in_bounds(g.shape, nr, nc)
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(cells),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _frame_from_component(grid: np.ndarray, bg: int, comp: dict) -> Frame | None:
    coords = comp["coords"]
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    coord_set = {tuple(map(int, coord)) for coord in coords}
    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    missing: list[Coord] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if (r, c) not in coord_set:
                    missing.append((r, c))

    if len(missing) != 1:
        return None

    gap = missing[0]
    gr, gc = gap
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None
    if len(coord_set) != 2 * h + 2 * w - 5:
        return None

    if gr == r0:
        side = "top"
    elif gr == r1:
        side = "bottom"
    elif gc == c0:
        side = "left"
    else:
        side = "right"
    return Frame(color=color, bbox=(r0, c0, r1, c1), gap=gap, side=side)


def _detect_frames_and_markers(grid: np.ndarray) -> tuple[list[Frame], dict[Coord, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[Frame] = []
    markers: dict[Coord, int] = {}

    for comp in _find_components(g, bg):
        if comp["area"] == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(comp["color"])
            continue
        frame = _frame_from_component(g, bg, comp)
        assert frame is not None, "expected every non-singleton component to be a gapped hollow rectangle"
        frames.append(frame)

    assert frames, "expected at least one frame"
    return frames, markers


def _inside_path(frame: Frame) -> list[Coord]:
    r0, c0, r1, c1 = frame.bbox
    gr, gc = frame.gap
    dr, dc = (-SIDE_TO_OUTWARD[frame.side][0], -SIDE_TO_OUTWARD[frame.side][1])
    r, c = gr + dr, gc + dc
    cells: list[Coord] = []
    while r0 < r < r1 and c0 < c < c1:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _outward_ray_cells(shape: tuple[int, int], frame: Frame) -> list[Coord]:
    gr, gc = frame.gap
    dr, dc = SIDE_TO_OUTWARD[frame.side]
    r, c = gr + dr, gc + dc
    cells: list[Coord] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _raw_queue_for_frame(grid: np.ndarray, frame: Frame, markers: dict[Coord, int]) -> list[Coord]:
    g = np.asarray(grid, dtype=int)
    queue: list[Coord] = []
    for cell in _outward_ray_cells(g.shape, frame):
        color = int(g[cell])
        if color == _get_bg(g):
            continue
        if cell in markers and color == frame.color:
            queue.append(cell)
            continue
        break
    return queue


def _solve_moves(grid: np.ndarray) -> list[tuple[Coord, Coord, int]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames, markers = _detect_frames_and_markers(g)

    raw_queues = [_raw_queue_for_frame(g, frame, markers) for frame in frames]
    assignments: dict[Coord, list[int]] = {}
    for frame_i, queue in enumerate(raw_queues):
        for cell in queue:
            assignments.setdefault(cell, []).append(frame_i)

    moves: list[tuple[Coord, Coord, int]] = []
    for frame_i, frame in enumerate(frames):
        queue = raw_queues[frame_i]
        usable: list[Coord] = []
        for cell in queue:
            if len(assignments[cell]) != 1:
                break
            usable.append(cell)

        deep_cells = list(reversed(_inside_path(frame)))
        for src, dst in zip(usable, deep_cells, strict=False):
            moves.append((src, dst, frame.color))

    return moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull same-colored singleton markers through each hollow frame's unique border gap and stack them inside along the entry direction.

    Input structure:
    - The grid has a dominant background color, several monochrome hollow rectangular frames, and isolated singleton pixels.
    - Each frame is a 1-cell-thick rectangle whose border misses exactly one non-corner pixel, creating a unique gap.
    - Some singleton pixels share a frame's color and lie on the outward row or column of that gap; other singletons are distractors.

    Rule:
    - For each frame, look outward from its gap along the perpendicular sightline.
    - Same-colored singleton pixels on that ray can enter in nearest-to-farthest order until a different foreground pixel blocks the ray.
    - After passing through the gap, entering pixels continue in the same direction and occupy the deepest available interior cells first, stacking toward the gap; all other pixels stay put.

    Key insight:
    - The missing border cell is a doorway, not a flaw: geometry identifies the only valid entry ray, and the queue order outside the frame determines the stack order inside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()
    moves = _solve_moves(g)
    for src, _dst, _color in moves:
        out[src] = bg
    for _src, dst, color in moves:
        out[dst] = color
    return out


GEN_BG = 0
GEN_GRID_MIN = 16
GEN_GRID_MAX = 28
GEN_FRAME_COUNT = (2, 4)
GEN_FRAME_H = (4, 8)
GEN_FRAME_W = (4, 8)
GEN_EDGE_MARGIN = 1
GEN_MIN_BBOX_GAP = 2
GEN_MIN_RAY_LEN = 3
GEN_MAX_MOVERS_PER_FRAME = 3
GEN_EXTRA_NOISE = (1, 4)
GEN_ATTEMPTS = 500


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_sep = ar1 + gap < br0 or br1 + gap < ar0
    col_sep = ac1 + gap < bc0 or bc1 + gap < ac0
    return row_sep or col_sep


def _sample_frame_candidate(h: int, w: int, color: int) -> Frame | None:
    fh = random.randint(*GEN_FRAME_H)
    fw = random.randint(*GEN_FRAME_W)
    side = random.choice(SIDES)

    r_lo = GEN_EDGE_MARGIN
    r_hi = h - fh - GEN_EDGE_MARGIN
    c_lo = GEN_EDGE_MARGIN
    c_hi = w - fw - GEN_EDGE_MARGIN
    if side == "top":
        r_lo = max(r_lo, GEN_MIN_RAY_LEN)
    elif side == "bottom":
        r_hi = min(r_hi, h - fh - GEN_MIN_RAY_LEN)
    elif side == "left":
        c_lo = max(c_lo, GEN_MIN_RAY_LEN)
    else:
        c_hi = min(c_hi, w - fw - GEN_MIN_RAY_LEN)

    if r_lo > r_hi or c_lo > c_hi:
        return None

    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    r1 = r0 + fh - 1
    c1 = c0 + fw - 1
    if side in ("top", "bottom"):
        gap = (r0 if side == "top" else r1, random.randint(c0 + 1, c1 - 1))
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)
    return Frame(color=color, bbox=(r0, c0, r1, c1), gap=gap, side=side)


def _place_frames(h: int, w: int, colors: list[int]) -> list[Frame] | None:
    frames: list[Frame] = []
    reserved_corridors: set[Coord] = set()
    for color in colors:
        placed = False
        for _ in range(200):
            frame = _sample_frame_candidate(h, w, color)
            if frame is None:
                continue
            corridor = set(_outward_ray_cells((h, w), frame))
            if len(corridor) < GEN_MIN_RAY_LEN:
                continue
            bbox = frame.bbox
            bbox_cells = {
                (r, c)
                for r in range(bbox[0], bbox[2] + 1)
                for c in range(bbox[1], bbox[3] + 1)
            }
            if corridor & reserved_corridors:
                continue
            if bbox_cells & reserved_corridors:
                continue
            if any(not _bbox_gap_ok(frame.bbox, other.bbox, GEN_MIN_BBOX_GAP) for other in frames):
                continue
            if any(corridor & {
                (r, c)
                for r in range(other.bbox[0], other.bbox[2] + 1)
                for c in range(other.bbox[1], other.bbox[3] + 1)
            } for other in frames):
                continue
            frames.append(frame)
            reserved_corridors |= corridor
            placed = True
            break
        if not placed:
            return None
    return frames


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    r0, c0, r1, c1 = frame.bbox
    color = frame.color
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    grid[frame.gap] = GEN_BG


def _marker_clear(grid: np.ndarray, cell: Coord) -> bool:
    r, c = cell
    if int(grid[r, c]) != GEN_BG:
        return False
    return all(
        not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == GEN_BG
        for nr, nc in _neighbors4(r, c)
    )


def _ray_distance_to_cell(frame: Frame, dist: int) -> Coord:
    gr, gc = frame.gap
    dr, dc = SIDE_TO_OUTWARD[frame.side]
    return gr + dr * dist, gc + dc * dist


def _slot_positions(ray_len: int, needed: int) -> list[int] | None:
    choices = []
    for start in (1, 2):
        slots = list(range(start, ray_len + 1, 2))
        if len(slots) >= needed:
            choices.append(slots)
    if not choices:
        return None
    slots = random.choice(choices)
    return sorted(random.sample(slots, needed))


def _free_noise_cell(grid: np.ndarray, forbidden: set[Coord], predicate) -> Coord | None:
    cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r, c) not in forbidden and _marker_clear(grid, (r, c)) and predicate(r, c)
    ]
    return random.choice(cells) if cells else None


def _generate_expected(inp: np.ndarray, moves: list[tuple[Coord, Coord, int]]) -> np.ndarray:
    out = inp.copy()
    for src, _dst, _color in moves:
        out[src] = GEN_BG
    for _src, dst, color in moves:
        out[dst] = color
    return out


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    frames: list[Frame],
    planned_moves: list[tuple[Coord, Coord, int]],
    fixed_markers: list[tuple[Coord, int]],
    queue_expected: bool,
    blocked_expected: bool,
) -> bool:
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    if len(_solve_moves(inp)) != len(planned_moves):
        return False

    plan_set = {(src, dst, color) for src, dst, color in planned_moves}
    actual_set = {(src, dst, color) for src, dst, color in _solve_moves(inp)}
    if plan_set != actual_set:
        return False

    if queue_expected:
        if not any(sum(1 for src, _dst, color in planned_moves if color == frame.color) >= 2 for frame in frames):
            return False
    if blocked_expected:
        if not any(int(inp[pos]) == color and int(expected[pos]) == color for pos, color in fixed_markers):
            return False

    for src, dst, color in planned_moves:
        if int(inp[src]) != color or int(expected[src]) != GEN_BG or int(expected[dst]) != color:
            return False
    for pos, color in fixed_markers:
        if int(inp[pos]) != color or int(expected[pos]) != color:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    w = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    frame_count = random.randint(*GEN_FRAME_COUNT)
    colors = random.sample(range(1, 10), frame_count)
    frames = _place_frames(h, w, colors)
    if frames is None:
        return None

    inp = np.full((h, w), GEN_BG, dtype=int)
    for frame in frames:
        _draw_frame(inp, frame)

    ray_lengths = {frame.color: len(_outward_ray_cells((h, w), frame)) for frame in frames}
    capacities = {frame.color: len(_inside_path(frame)) for frame in frames}
    slot_caps = {color: max(0, (ray_lengths[color] + 1) // 2) for color in colors}
    max_movers = {
        color: min(GEN_MAX_MOVERS_PER_FRAME, capacities[color], slot_caps[color]) for color in colors
    }

    queue_expected = random.random() < 0.7
    blocked_expected = random.random() < 0.7
    queue_color = None
    if queue_expected:
        queue_choices = [color for color in colors if max_movers[color] >= 2]
        if not queue_choices:
            return None
        queue_color = random.choice(queue_choices)

    move_counts = {color: 0 for color in colors}
    for color in colors:
        m = max_movers[color]
        if m <= 0:
            continue
        if color == queue_color:
            move_counts[color] = random.randint(2, m)
        else:
            move_counts[color] = random.choices(
                list(range(m + 1)),
                weights=[3] + [4] + [2] * max(0, m - 1),
                k=1,
            )[0]

    if sum(move_counts.values()) == 0:
        pickable = [color for color in colors if max_movers[color] >= 1]
        if not pickable:
            return None
        move_counts[random.choice(pickable)] = 1

    blocked_color = None
    if blocked_expected:
        blocked_choices = [
            color for color in colors if slot_caps[color] >= move_counts[color] + 2
        ]
        if not blocked_choices:
            return None
        blocked_color = random.choice(blocked_choices)

    free_noise_colors = [color for color in range(1, 10) if color not in colors]
    if not free_noise_colors:
        return None

    noise_blocked = set()
    for frame in frames:
        r0, c0, r1, c1 = frame.bbox
        noise_blocked |= {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
        }
        noise_blocked |= set(_outward_ray_cells((h, w), frame))

    planned_moves: list[tuple[Coord, Coord, int]] = []
    fixed_markers: list[tuple[Coord, int]] = []
    forbidden_noise: set[Coord] = set()

    for frame in frames:
        color = frame.color
        moved_count = move_counts[color]
        need_blocked = color == blocked_color
        total_slots = moved_count + (2 if need_blocked else 0)
        slots = _slot_positions(ray_lengths[color], total_slots)
        if total_slots > 0 and slots is None:
            return None

        inside_deep = list(reversed(_inside_path(frame)))
        if len(inside_deep) < moved_count:
            return None

        if slots is None:
            slots = []
        moved_slots = slots[:moved_count]
        for src_dist, dst in zip(sorted(moved_slots), inside_deep[:moved_count], strict=True):
            src = _ray_distance_to_cell(frame, src_dist)
            if not _marker_clear(inp, src):
                return None
            inp[src] = color
            planned_moves.append((src, dst, color))
            forbidden_noise.add(src)

        if need_blocked:
            blocker_dist = slots[moved_count]
            blocked_dist = slots[moved_count + 1]
            blocker = _ray_distance_to_cell(frame, blocker_dist)
            blocked = _ray_distance_to_cell(frame, blocked_dist)
            blocker_color = random.choice(free_noise_colors)
            if not _marker_clear(inp, blocker) or not _marker_clear(inp, blocked):
                return None
            inp[blocker] = blocker_color
            inp[blocked] = color
            fixed_markers.append((blocker, blocker_color))
            fixed_markers.append((blocked, color))
            forbidden_noise.add(blocker)
            forbidden_noise.add(blocked)

    extra_same_color = random.randint(1, max(1, len(frames) // 2))
    for frame in random.sample(frames, extra_same_color):
        cell = _free_noise_cell(
            inp,
            forbidden_noise | noise_blocked,
            lambda r, c, gap=frame.gap: r != gap[0] and c != gap[1],
        )
        if cell is None:
            return None
        inp[cell] = frame.color
        fixed_markers.append((cell, frame.color))
        forbidden_noise.add(cell)

    extra_noise_count = random.randint(*GEN_EXTRA_NOISE)
    for _ in range(extra_noise_count):
        cell = _free_noise_cell(inp, forbidden_noise | noise_blocked, lambda _r, _c: True)
        if cell is None:
            return None
        color = random.choice(free_noise_colors)
        inp[cell] = color
        fixed_markers.append((cell, color))
        forbidden_noise.add(cell)

    expected = _generate_expected(inp, planned_moves)
    if not _semantic_checks(
        inp,
        expected,
        frames,
        planned_moves,
        fixed_markers,
        queue_expected,
        blocked_expected,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
