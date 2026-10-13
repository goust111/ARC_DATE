from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


try:
    _ARC_UTILS = SourceFileLoader(
        "arc_utils_shared", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _ARC_UTILS.Color
    components_mask = _ARC_UTILS.components_mask
    get_bg = _ARC_UTILS.get_bg
    in_bounds = _ARC_UTILS.in_bounds
except Exception:  # pragma: no cover - fallback for standalone loading
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9

    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
        return int(vals[np.argmax(counts)])

    def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
        h, w = shape
        return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)

    def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
        mask = np.asarray(mask, dtype=bool)
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=bool)
        if int(connectivity) == 4:
            steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            steps = [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ]

        out: list[np.ndarray] = []
        for r in range(h):
            for c in range(w):
                if not mask[r, c] or seen[r, c]:
                    continue
                q = deque([(r, c)])
                seen[r, c] = True
                coords: list[tuple[int, int]] = []
                while q:
                    rr, cc = q.popleft()
                    coords.append((rr, cc))
                    for dr, dc in steps:
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and mask[nr, nc]
                            and not seen[nr, nc]
                        ):
                            seen[nr, nc] = True
                            q.append((nr, nc))
                out.append(np.array(coords, dtype=int))
        return out


Coord = tuple[int, int]
Direction = tuple[int, int]

UP: Direction = (-1, 0)
DOWN: Direction = (1, 0)
LEFT: Direction = (0, -1)
RIGHT: Direction = (0, 1)
ALL_DIRS: tuple[Direction, ...] = (UP, DOWN, LEFT, RIGHT)

SLASH_REFLECT = {
    RIGHT: UP,
    UP: RIGHT,
    LEFT: DOWN,
    DOWN: LEFT,
}
BACKSLASH_REFLECT = {
    RIGHT: DOWN,
    DOWN: RIGHT,
    LEFT: UP,
    UP: LEFT,
}

BG = Color.BLACK
MIRROR = Color.GREY
EMITTER_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

PAIR_COUNT_RANGE = (2, 3)
SINGLE_COUNT_RANGE = (1, 2)
GRID_HEIGHT_RANGE = (15, 20)
GRID_WIDTH_RANGE = (15, 20)
INTERSECTION_MARGIN = 4
TARGET_MARGIN = 4
MIN_SEG_LEN = 2
MIN_TAIL_LEN = 2
MAX_GENERATION_ATTEMPTS = 400
PAIR_BUILD_ATTEMPTS = 70
SINGLE_BUILD_ATTEMPTS = 50
CANDIDATE_SAMPLE_ATTEMPTS = 50


@dataclass(frozen=True)
class Emitter:
    color: int
    cells: tuple[Coord, Coord]
    start: Coord
    direction: Direction
    side: str


@dataclass(frozen=True)
class Trace:
    color: int
    emitter: Emitter
    path: tuple[Coord, ...]
    mirror_hits: tuple[Coord, ...]

    @property
    def path_set(self) -> frozenset[Coord]:
        return frozenset(self.path)


@dataclass(frozen=True)
class RayCandidate:
    emitter_cells: tuple[Coord, Coord]
    side: str
    lane: int
    start: Coord
    initial_dir: Direction
    final_dir: Direction
    path: tuple[Coord, ...]
    turns: tuple[tuple[Coord, Direction, Direction], ...]
    mirror_components: tuple[tuple[Coord, Coord], ...]
    grey_cells: frozenset[Coord]

    @property
    def path_set(self) -> frozenset[Coord]:
        return frozenset(self.path)

    @property
    def all_used(self) -> frozenset[Coord]:
        return frozenset(self.path).union(self.emitter_cells).union(self.grey_cells)

    @property
    def turn_count(self) -> int:
        return len(self.turns)


def _axis(direction: Direction) -> str:
    return "v" if int(direction[0]) != 0 else "h"


def _add(a: Coord, b: Direction) -> Coord:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br > ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment must be axis-aligned")


def _reflect(direction: Direction, mirror_kind: str) -> Direction:
    if mirror_kind == "/":
        return SLASH_REFLECT[direction]
    if mirror_kind == "\\":
        return BACKSLASH_REFLECT[direction]
    raise ValueError("unknown mirror")


def _orientation_for_turn(in_dir: Direction, out_dir: Direction) -> str:
    if SLASH_REFLECT[in_dir] == out_dir:
        return "/"
    if BACKSLASH_REFLECT[in_dir] == out_dir:
        return "\\"
    raise ValueError("turn is not a mirror reflection")


def _interior(coord: Coord, shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = coord
    return 0 < int(r) < int(h) - 1 and 0 < int(c) < int(w) - 1


def _sorted_colors(grid: np.ndarray, exclude: set[int]) -> list[int]:
    colors = sorted(set(map(int, np.asarray(grid, dtype=int).ravel())) - set(exclude))
    return colors


def _parse_emitter_component(coords: np.ndarray, color: int, shape: tuple[int, int]) -> Emitter:
    h, w = shape
    pts = sorted((int(r), int(c)) for r, c in coords)
    if len(pts) != 2:
        raise AssertionError("emitter must be size 2")
    (r0, c0), (r1, c1) = pts
    if r0 == r1:
        if [c0, c1] == [0, 1]:
            start = (r0, 2)
            assert in_bounds(shape, *start)
            return Emitter(int(color), ((r0, c0), (r1, c1)), start, RIGHT, "left")
        if [c0, c1] == [w - 2, w - 1]:
            start = (r0, w - 3)
            assert in_bounds(shape, *start)
            return Emitter(int(color), ((r0, c0), (r1, c1)), start, LEFT, "right")
    if c0 == c1:
        if [r0, r1] == [0, 1]:
            start = (2, c0)
            assert in_bounds(shape, *start)
            return Emitter(int(color), ((r0, c0), (r1, c1)), start, DOWN, "top")
        if [r0, r1] == [h - 2, h - 1]:
            start = (h - 3, c0)
            assert in_bounds(shape, *start)
            return Emitter(int(color), ((r0, c0), (r1, c1)), start, UP, "bottom")
    raise AssertionError("invalid emitter placement")


def _parse_emitters(grid: np.ndarray, bg: int) -> dict[int, list[Emitter]]:
    g = np.asarray(grid, dtype=int)
    emitters: dict[int, list[Emitter]] = defaultdict(list)
    for color in _sorted_colors(g, exclude={int(bg), int(MIRROR)}):
        comps = components_mask(g == int(color), connectivity=4)
        for comp in comps:
            emitter = _parse_emitter_component(comp, int(color), g.shape)
            emitters[int(color)].append(emitter)
    for color in emitters:
        emitters[color].sort(key=lambda e: (e.cells[0], e.cells[1]))
    return dict(emitters)


def _parse_mirrors(grid: np.ndarray) -> dict[Coord, str]:
    g = np.asarray(grid, dtype=int)
    mirrors: dict[Coord, str] = {}
    for comp in components_mask(g == int(MIRROR), connectivity=8):
        pts = sorted((int(r), int(c)) for r, c in comp)
        if len(pts) != 2:
            raise AssertionError("mirror must be a 2-pixel diagonal")
        (r0, c0), (r1, c1) = pts
        if r1 - r0 != 1 or abs(c1 - c0) != 1:
            raise AssertionError("mirror must be diagonal")
        mirror_kind = "\\" if c1 == c0 + 1 else "/"
        for pt in pts:
            if pt in mirrors and mirrors[pt] != mirror_kind:
                raise AssertionError("conflicting mirror orientation")
            mirrors[pt] = mirror_kind
    return mirrors


def _trace_path(
    shape: tuple[int, int], start: Coord, direction: Direction, mirrors: dict[Coord, str]
) -> Trace:
    r, c = start
    d = direction
    path: list[Coord] = []
    mirror_hits: list[Coord] = []
    seen_states: set[tuple[int, int, Direction]] = set()
    while in_bounds(shape, r, c):
        state = (int(r), int(c), d)
        if state in seen_states:
            raise AssertionError("ray entered a loop")
        seen_states.add(state)
        if (int(r), int(c)) in mirrors:
            d = _reflect(d, mirrors[(int(r), int(c))])
            mirror_hits.append((int(r), int(c)))
        path.append((int(r), int(c)))
        r += d[0]
        c += d[1]
    return tuple(path), tuple(mirror_hits)


def _analyze_grid(grid: np.ndarray) -> tuple[int, dict[int, list[Emitter]], dict[int, list[Trace]], dict[int, Coord]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    mirrors = _parse_mirrors(g)
    emitters = _parse_emitters(g, bg)
    traces: dict[int, list[Trace]] = defaultdict(list)
    intersections: dict[int, Coord] = {}

    for color, color_emitters in emitters.items():
        for emitter in color_emitters:
            path, hits = _trace_path(g.shape, emitter.start, emitter.direction, mirrors)
            traces[int(color)].append(
                Trace(int(color), emitter, tuple(path), tuple(hits))
            )
        traces[int(color)].sort(key=lambda t: (t.emitter.cells[0], t.emitter.cells[1]))

    for color, color_traces in traces.items():
        if len(color_traces) == 1:
            continue
        if len(color_traces) != 2:
            raise AssertionError("each color must have one or two emitters")
        common = color_traces[0].path_set & color_traces[1].path_set
        if len(common) != 1:
            raise AssertionError("paired emitters must have exactly one common cell")
        intersections[int(color)] = next(iter(common))

    return bg, emitters, dict(traces), intersections


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the unique crossing cell of each same-colored reflected light pair.

    Input structure:
    - The background is uniform; non-background objects are border emitters and grey diagonal mirrors.
    - Each emitter is a same-colored 2-pixel bar on one border, perpendicular to that border, and it shoots inward.
    - Each mirror is a 2-pixel grey diagonal with orientation "/" or "\".
    - Some colors appear on exactly two emitters, while some colors appear only once as decoys.

    Rule:
    - Trace one invisible ray from each emitter, starting just inside the 2-pixel bar and moving inward.
    - Whenever the ray enters a grey mirror cell, turn it 90 degrees according to that mirror's slash orientation, then continue until it leaves the grid.
    - For every color that has exactly two emitters, find the unique grid cell visited by both rays and paint only that cell with the emitter color.
    - Remove everything else: emitters, mirrors, and the rest of every light path all become background.

    Key insight:
    - The answer is determined by hidden trajectories: the visible border bars only set launch directions, grey diagonals only bend the rays, and singleton colors are pure distractors.
    """
    g = np.asarray(grid, dtype=int)
    bg, _emitters, _traces, intersections = _analyze_grid(g)
    out = np.full(g.shape, int(bg), dtype=int)
    for color, (r, c) in intersections.items():
        out[int(r), int(c)] = int(color)
    return out


def _emitter_template(shape: tuple[int, int], side: str, lane: int) -> tuple[tuple[Coord, Coord], Coord, Direction]:
    h, w = shape
    if side == "top":
        return ((0, lane), (1, lane)), (2, lane), DOWN
    if side == "bottom":
        return ((h - 2, lane), (h - 1, lane)), (h - 3, lane), UP
    if side == "left":
        return ((lane, 0), (lane, 1)), (lane, 2), RIGHT
    if side == "right":
        return ((lane, w - 2), (lane, w - 1)), (lane, w - 3), LEFT
    raise ValueError("unknown side")


def _lane_values(shape: tuple[int, int], side: str) -> list[int]:
    h, w = shape
    if side in {"top", "bottom"}:
        return list(range(1, w - 1))
    return list(range(1, h - 1))


def _compatible_sides(final_dir: Direction, turn_count: int) -> tuple[str, ...]:
    final_axis = _axis(final_dir)
    if turn_count % 2 == 0:
        return ("top", "bottom") if final_axis == "v" else ("left", "right")
    return ("left", "right") if final_axis == "v" else ("top", "bottom")


def _build_polylines(
    shape: tuple[int, int],
    start: Coord,
    initial_dir: Direction,
    target: Coord,
    final_dir: Direction,
    turn_count: int,
) -> list[tuple[list[Coord], list[tuple[Coord, Direction, Direction]]]]:
    out: list[tuple[list[Coord], list[tuple[Coord, Direction, Direction]]]] = []
    init_axis = _axis(initial_dir)
    final_axis = _axis(final_dir)

    def straight_ok(a: Coord, b: Coord, d: Direction) -> bool:
        if a[0] == b[0]:
            return (b[1] - a[1]) * d[1] > 0 and abs(b[1] - a[1]) >= MIN_SEG_LEN
        if a[1] == b[1]:
            return (b[0] - a[0]) * d[0] > 0 and abs(b[0] - a[0]) >= MIN_SEG_LEN
        return False

    if turn_count == 0:
        if initial_dir == final_dir and straight_ok(start, target, initial_dir):
            out.append(([start, target], []))
        return out

    if turn_count == 1:
        if init_axis == final_axis:
            return out
        if init_axis == "v":
            turn = (target[0], start[1])
        else:
            turn = (start[0], target[1])
        if (
            _interior(turn, shape)
            and straight_ok(start, turn, initial_dir)
            and straight_ok(turn, target, final_dir)
        ):
            out.append(([start, turn, target], [(turn, initial_dir, final_dir)]))
        return out

    if turn_count != 2 or init_axis != final_axis:
        return out

    h, w = shape
    if init_axis == "v":
        if start[1] == target[1]:
            return out
        middle_dir = RIGHT if target[1] > start[1] else LEFT
        for row in range(1, h - 1):
            t1 = (row, start[1])
            t2 = (row, target[1])
            if not (_interior(t1, shape) and _interior(t2, shape)):
                continue
            if not (
                straight_ok(start, t1, initial_dir)
                and abs(t2[1] - t1[1]) >= MIN_SEG_LEN
                and straight_ok(t2, target, final_dir)
            ):
                continue
            turns = [(t1, initial_dir, middle_dir), (t2, middle_dir, final_dir)]
            out.append(([start, t1, t2, target], turns))
        return out

    if start[0] == target[0]:
        return out
    middle_dir = DOWN if target[0] > start[0] else UP
    for col in range(1, w - 1):
        t1 = (start[0], col)
        t2 = (target[0], col)
        if not (_interior(t1, shape) and _interior(t2, shape)):
            continue
        if not (
            straight_ok(start, t1, initial_dir)
            and abs(t2[0] - t1[0]) >= MIN_SEG_LEN
            and straight_ok(t2, target, final_dir)
        ):
            continue
        turns = [(t1, initial_dir, middle_dir), (t2, middle_dir, final_dir)]
        out.append(([start, t1, t2, target], turns))
    return out


def _path_from_polyline(
    shape: tuple[int, int], points: list[Coord], final_dir: Direction
) -> tuple[tuple[Coord, ...], bool]:
    path: list[Coord] = []
    for i in range(len(points) - 1):
        segment = _segment_cells(points[i], points[i + 1])
        if i > 0:
            segment = segment[1:]
        path.extend(segment)
    tail: list[Coord] = []
    cur = _add(points[-1], final_dir)
    while in_bounds(shape, *cur):
        tail.append(cur)
        cur = _add(cur, final_dir)
    if len(tail) < MIN_TAIL_LEN:
        return tuple(), False
    path.extend(tail)
    if len(path) != len(set(path)):
        return tuple(), False
    return tuple(path), True


def _chebyshev_close(a: Coord, b: Coord) -> bool:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1


def _buddy_options(turn: Coord, mirror_kind: str) -> tuple[Coord, Coord]:
    r, c = turn
    if mirror_kind == "/":
        return (r - 1, c + 1), (r + 1, c - 1)
    return (r - 1, c - 1), (r + 1, c + 1)


def _assign_mirrors(
    shape: tuple[int, int],
    turns: tuple[tuple[Coord, Direction, Direction], ...],
    path_set: frozenset[Coord],
    occupied: frozenset[Coord],
    existing_grey: frozenset[Coord],
) -> tuple[tuple[tuple[Coord, Coord], ...], frozenset[Coord]] | None:
    chosen: list[tuple[Coord, Coord]] = []
    grey_so_far = set(existing_grey)

    def recurse(i: int) -> bool:
        if i == len(turns):
            return True
        turn, in_dir, out_dir = turns[i]
        mirror_kind = _orientation_for_turn(in_dir, out_dir)
        for buddy in _buddy_options(turn, mirror_kind):
            if not _interior(buddy, shape):
                continue
            if buddy in path_set or buddy in occupied:
                continue
            component = {turn, buddy}
            bad = False
            for cell in component:
                for other in grey_so_far:
                    if cell == other:
                        continue
                    if _chebyshev_close(cell, other):
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            chosen.append((turn, buddy))
            grey_so_far.update(component)
            if recurse(i + 1):
                return True
            chosen.pop()
            grey_so_far.difference_update(component)
        return False

    if recurse(0):
        grey_cells: set[Coord] = set()
        for a, b in chosen:
            grey_cells.add(a)
            grey_cells.add(b)
        return tuple(chosen), frozenset(grey_cells)
    return None


def _enumerate_candidates(
    shape: tuple[int, int],
    target: Coord,
    final_dir: Direction,
    occupied: frozenset[Coord],
    existing_grey: frozenset[Coord],
) -> list[RayCandidate]:
    candidates: list[RayCandidate] = []
    for turn_count in (0, 1, 2):
        for side in _compatible_sides(final_dir, turn_count):
            lanes = _lane_values(shape, side)
            random.shuffle(lanes)
            for lane in lanes:
                emitter_cells, start, initial_dir = _emitter_template(shape, side, lane)
                emitter_set = frozenset(emitter_cells)
                if emitter_set & occupied:
                    continue
                for points, turns_list in _build_polylines(
                    shape, start, initial_dir, target, final_dir, turn_count
                ):
                    path, ok = _path_from_polyline(shape, points, final_dir)
                    if not ok:
                        continue
                    path_set = frozenset(path)
                    if path_set & occupied:
                        continue
                    turns = tuple(turns_list)
                    assigned = _assign_mirrors(
                        shape, turns, path_set, occupied.union(emitter_set), existing_grey
                    )
                    if assigned is None:
                        continue
                    mirror_components, grey_cells = assigned
                    candidate = RayCandidate(
                        emitter_cells=tuple(emitter_cells),
                        side=side,
                        lane=int(lane),
                        start=start,
                        initial_dir=initial_dir,
                        final_dir=final_dir,
                        path=tuple(path),
                        turns=turns,
                        mirror_components=mirror_components,
                        grey_cells=grey_cells,
                    )
                    candidates.append(candidate)
    random.shuffle(candidates)
    return candidates


def _candidate_from_spec(
    shape: tuple[int, int],
    emitter_cells: tuple[Coord, Coord],
    side: str,
    lane: int,
    start: Coord,
    initial_dir: Direction,
    final_dir: Direction,
    points: list[Coord],
    turns: tuple[tuple[Coord, Direction, Direction], ...],
    occupied: frozenset[Coord],
    existing_grey: frozenset[Coord],
) -> RayCandidate | None:
    emitter_set = frozenset(emitter_cells)
    if emitter_set & occupied:
        return None
    path, ok = _path_from_polyline(shape, points, final_dir)
    if not ok:
        return None
    path_set = frozenset(path)
    if path_set & occupied:
        return None
    assigned = _assign_mirrors(
        shape, turns, path_set, occupied.union(emitter_set), existing_grey
    )
    if assigned is None:
        return None
    mirror_components, grey_cells = assigned
    return RayCandidate(
        emitter_cells=tuple(emitter_cells),
        side=side,
        lane=int(lane),
        start=start,
        initial_dir=initial_dir,
        final_dir=final_dir,
        path=tuple(path),
        turns=turns,
        mirror_components=mirror_components,
        grey_cells=grey_cells,
    )


def _sample_candidate(
    shape: tuple[int, int],
    target: Coord,
    final_dir: Direction,
    occupied: frozenset[Coord],
    existing_grey: frozenset[Coord],
    *,
    min_turns: int = 0,
) -> RayCandidate | None:
    turn_choices = [turn_count for turn_count in (0, 1, 2) if turn_count >= min_turns]
    weights = {0: 1, 1: 3, 2: 4}
    for _ in range(CANDIDATE_SAMPLE_ATTEMPTS):
        turn_count = random.choices(
            turn_choices, weights=[weights[t] for t in turn_choices], k=1
        )[0]
        side = random.choice(_compatible_sides(final_dir, turn_count))
        lane = random.choice(_lane_values(shape, side))
        emitter_cells, start, initial_dir = _emitter_template(shape, side, lane)
        polylines = _build_polylines(
            shape, start, initial_dir, target, final_dir, turn_count
        )
        if not polylines:
            continue
        random.shuffle(polylines)
        for points, turns_list in polylines[: min(4, len(polylines))]:
            candidate = _candidate_from_spec(
                shape,
                emitter_cells,
                side,
                lane,
                start,
                initial_dir,
                final_dir,
                points,
                tuple(turns_list),
                occupied,
                existing_grey,
            )
            if candidate is not None:
                return candidate
    return None


def _pair_candidates_compatible(a: RayCandidate, b: RayCandidate, target: Coord) -> bool:
    if a.turn_count + b.turn_count <= 0:
        return False
    if a.all_used & b.all_used != {target}:
        return False
    if a.path_set & b.path_set != {target}:
        return False
    for cell_a in a.grey_cells:
        for cell_b in b.grey_cells:
            if _chebyshev_close(cell_a, cell_b):
                return False
    return True


def _random_interior_target(shape: tuple[int, int], margin: int) -> Coord:
    h, w = shape
    return (
        random.randint(margin, h - margin - 1),
        random.randint(margin, w - margin - 1),
    )


def _render_grid(
    shape: tuple[int, int],
    rays_by_color: dict[int, list[RayCandidate]],
) -> np.ndarray:
    grid = np.full(shape, int(BG), dtype=int)
    for color, rays in rays_by_color.items():
        for ray in rays:
            for r, c in ray.emitter_cells:
                assert grid[r, c] == int(BG)
                grid[r, c] = int(color)
    for rays in rays_by_color.values():
        for ray in rays:
            for a, b in ray.mirror_components:
                for r, c in (a, b):
                    assert grid[r, c] in (int(BG), int(MIRROR))
                    grid[r, c] = int(MIRROR)
    return grid


def _expected_output(shape: tuple[int, int], intersections: dict[int, Coord]) -> np.ndarray:
    out = np.full(shape, int(BG), dtype=int)
    for color, (r, c) in intersections.items():
        out[int(r), int(c)] = int(color)
    return out


def _path_signatures(rays: list[RayCandidate | Trace]) -> list[frozenset[Coord]]:
    return sorted((frozenset(ray.path) for ray in rays), key=lambda s: sorted(s))


def _validate_semantics(
    grid: np.ndarray,
    rays_by_color: dict[int, list[RayCandidate]],
    intersections: dict[int, Coord],
    pair_colors: set[int],
    singleton_colors: set[int],
) -> bool:
    try:
        bg, emitters, traces, parsed_intersections = _analyze_grid(grid)
    except AssertionError:
        return False

    if int(bg) != int(BG):
        return False
    if set(emitters) != set(rays_by_color):
        return False
    if parsed_intersections != intersections:
        return False

    expected = _expected_output(grid.shape, intersections)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    any_reflection = False
    all_traces = [trace for color_traces in traces.values() for trace in color_traces]
    for color, expected_rays in rays_by_color.items():
        parsed_traces = traces.get(color, [])
        if len(parsed_traces) != len(expected_rays):
            return False
        if len(parsed_traces) != (2 if color in pair_colors else 1):
            return False
        if _path_signatures(parsed_traces) != _path_signatures(expected_rays):
            return False
        any_reflection = any_reflection or any(len(trace.mirror_hits) > 0 for trace in parsed_traces)

    if not any_reflection:
        return False
    if not singleton_colors:
        return False

    colors = sorted(traces)
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            for trace_a in traces[color_a]:
                for trace_b in traces[color_b]:
                    if trace_a.path_set & trace_b.path_set:
                        return False
    return True


def _choose_pair(
    shape: tuple[int, int],
    occupied: frozenset[Coord],
    grey_cells: frozenset[Coord],
) -> tuple[Coord, RayCandidate, RayCandidate] | None:
    for _ in range(PAIR_BUILD_ATTEMPTS):
        target = _random_interior_target(shape, INTERSECTION_MARGIN)
        if target in occupied:
            continue
        vertical_dir = random.choice((UP, DOWN))
        horizontal_dir = random.choice((LEFT, RIGHT))
        ray_a = _sample_candidate(shape, target, vertical_dir, occupied, grey_cells)
        if ray_a is None:
            continue
        occupied_b = occupied.union(ray_a.all_used - {target})
        grey_b = grey_cells.union(ray_a.grey_cells)
        min_turns_b = 0 if ray_a.turn_count > 0 else 1
        ray_b = _sample_candidate(
            shape,
            target,
            horizontal_dir,
            occupied_b,
            grey_b,
            min_turns=min_turns_b,
        )
        if ray_b is not None and _pair_candidates_compatible(ray_a, ray_b, target):
            return target, ray_a, ray_b
    return None


def _choose_singleton(
    shape: tuple[int, int],
    occupied: frozenset[Coord],
    grey_cells: frozenset[Coord],
) -> RayCandidate | None:
    for _ in range(SINGLE_BUILD_ATTEMPTS):
        target = _random_interior_target(shape, TARGET_MARGIN)
        if target in occupied:
            continue
        final_dir = random.choice(ALL_DIRS)
        candidate = _sample_candidate(
            shape, target, final_dir, occupied, grey_cells, min_turns=1
        )
        if candidate is not None:
            return candidate
    return None


def _gen_attempt() -> np.ndarray | None:
    shape = (
        random.randint(*GRID_HEIGHT_RANGE),
        random.randint(*GRID_WIDTH_RANGE),
    )
    pair_count = 2 + int(random.random() < 0.25)
    single_count = 1 + int(random.random() < 0.35)
    color_count = pair_count + single_count
    if color_count > len(EMITTER_COLORS):
        return None
    colors = random.sample(EMITTER_COLORS, color_count)
    pair_colors = colors[:pair_count]
    singleton_colors = colors[pair_count:]

    occupied: frozenset[Coord] = frozenset()
    grey_cells: frozenset[Coord] = frozenset()
    rays_by_color: dict[int, list[RayCandidate]] = defaultdict(list)
    intersections: dict[int, Coord] = {}

    for color in pair_colors:
        chosen = _choose_pair(shape, occupied, grey_cells)
        if chosen is None:
            return None
        target, ray_a, ray_b = chosen
        intersections[int(color)] = target
        rays_by_color[int(color)].extend([ray_a, ray_b])
        occupied = occupied.union(ray_a.all_used).union(ray_b.all_used)
        grey_cells = grey_cells.union(ray_a.grey_cells).union(ray_b.grey_cells)

    for color in singleton_colors:
        chosen = _choose_singleton(shape, occupied, grey_cells)
        if chosen is None:
            return None
        rays_by_color[int(color)].append(chosen)
        occupied = occupied.union(chosen.all_used)
        grey_cells = grey_cells.union(chosen.grey_cells)

    grid = _render_grid(shape, dict(rays_by_color))
    if not _validate_semantics(
        grid,
        dict(rays_by_color),
        intersections,
        set(map(int, pair_colors)),
        set(map(int, singleton_colors)),
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-light-path instance")
