"""ARC task: complete a broken diagonal ray and restore its erased mirrors."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


BG = 0
BLUE = 1
RED = 2
DISTRACTOR_COLORS = (3, 4, 5, 6, 7, 8, 9)
MIN_LINE_SUPPORT = 3
MAX_GENERATION_ATTEMPTS = 800

Coord = tuple[int, int]
Line = tuple[str, int]  # ("back", r-c) or ("slash", r+c)


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _segment(a: Coord, b: Coord) -> list[Coord]:
    """Inclusive 45-degree segment from a to b."""
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    assert abs(dr) == abs(dc) and dr != 0, "ray segments must be nonempty diagonals"
    sr, sc = _sign(dr), _sign(dc)
    return [(a[0] + k * sr, a[1] + k * sc) for k in range(abs(dr) + 1)]


def _line_of(a: Coord, b: Coord) -> Line:
    assert abs(b[0] - a[0]) == abs(b[1] - a[1])
    if _sign(b[0] - a[0]) == _sign(b[1] - a[1]):
        return ("back", a[0] - a[1])
    return ("slash", a[0] + a[1])


def _line_contains(line: Line, cell: Coord) -> bool:
    kind, value = line
    r, c = cell
    return (r - c == value) if kind == "back" else (r + c == value)


def _intersection(back_value: int, slash_value: int) -> Coord | None:
    if (back_value + slash_value) % 2:
        return None
    return ((back_value + slash_value) // 2, (slash_value - back_value) // 2)


def _mirror_cells(center: Coord, orient: str) -> tuple[Coord, Coord, Coord]:
    r, c = center
    if orient == "h":
        return ((r, c - 1), (r, c), (r, c + 1))
    assert orient == "v"
    return ((r - 1, c), (r, c), (r + 1, c))


@dataclass(frozen=True)
class Mirror:
    center: Coord
    orient: str

    @property
    def cells(self) -> tuple[Coord, Coord, Coord]:
        return _mirror_cells(self.center, self.orient)


@dataclass
class DecodedRay:
    path: set[Coord]
    mirrors: list[Mirror]
    lines: list[Line]
    endpoints: list[Coord]


def _red_mirrors(grid: np.ndarray) -> list[Mirror]:
    """Read isolated, complete red length-three mirror segments."""
    h, w = grid.shape
    unseen = set(map(tuple, np.argwhere(grid == RED)))
    result: list[Mirror] = []
    while unseen:
        seed = unseen.pop()
        comp = {seed}
        queue = [seed]
        while queue:
            r, c = queue.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    comp.add(q)
                    queue.append(q)
        assert len(comp) == 3, "each retained red mirror must have length three"
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1 and max(cols) - min(cols) == 2:
            result.append(Mirror((next(iter(rows)), (min(cols) + max(cols)) // 2), "h"))
        elif len(cols) == 1 and max(rows) - min(rows) == 2:
            result.append(Mirror(((min(rows) + max(rows)) // 2, next(iter(cols))), "v"))
        else:
            raise AssertionError("red mirror must be a straight horizontal or vertical segment")
    assert result, "at least one retained red mirror is needed to determine ray direction"
    return result


def _active_lines(grid: np.ndarray) -> tuple[list[Line], dict[Line, list[Coord]]]:
    blue = [tuple(map(int, p)) for p in np.argwhere(grid == BLUE)]
    assert blue, "the input must contain blue ray fragments"
    back: dict[int, list[Coord]] = defaultdict(list)
    slash: dict[int, list[Coord]] = defaultdict(list)
    for r, c in blue:
        back[r - c].append((r, c))
        slash[r + c].append((r, c))
    support: dict[Line, list[Coord]] = {}
    for value, cells in back.items():
        if len(cells) >= MIN_LINE_SUPPORT:
            support[("back", value)] = cells
    for value, cells in slash.items():
        if len(cells) >= MIN_LINE_SUPPORT:
            support[("slash", value)] = cells
    lines = list(support)
    assert len(lines) >= 3, "a ray must contain at least two turns"
    return lines, support


def _ordered_line_chain(
    grid: np.ndarray, lines: list[Line], support: dict[Line, list[Coord]]
) -> tuple[list[Line], dict[frozenset[Line], Coord]]:
    h, w = grid.shape
    backs = [line for line in lines if line[0] == "back"]
    slashes = [line for line in lines if line[0] == "slash"]
    adjacency: dict[Line, list[Line]] = {line: [] for line in lines}
    corners: dict[frozenset[Line], Coord] = {}
    for back in backs:
        for slash in slashes:
            p = _intersection(back[1], slash[1])
            if p is None:
                continue
            r, c = p
            if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BLUE:
                adjacency[back].append(slash)
                adjacency[slash].append(back)
                corners[frozenset((back, slash))] = p

    assert all(1 <= len(adjacency[line]) <= 2 for line in lines), "turn graph must be a chain"
    ends = [line for line in lines if len(adjacency[line]) == 1]
    assert len(ends) == 2 and len(corners) == len(lines) - 1, "turn graph must have two ends"

    ordered: list[Line] = []
    prev: Line | None = None
    cur = ends[0]
    while True:
        ordered.append(cur)
        nxt = [line for line in adjacency[cur] if line != prev]
        if not nxt:
            break
        assert len(nxt) == 1 and nxt[0] not in ordered
        prev, cur = cur, nxt[0]
    assert len(ordered) == len(lines)

    # Only the first and last physical segments may expose a blue boundary cell.
    for i, line in enumerate(ordered):
        boundary = [
            p for p in support[line]
            if p[0] in (0, h - 1) or p[1] in (0, w - 1)
        ]
        if i in (0, len(ordered) - 1):
            assert len(boundary) == 1, "each ray end must have one visible boundary endpoint"
        else:
            assert not boundary, "interior ray segments cannot touch the boundary"
    return ordered, corners


def _candidate_decode(
    ordered: list[Line],
    corners: dict[frozenset[Line], Coord],
    support: dict[Line, list[Coord]],
    shape: tuple[int, int],
) -> DecodedRay:
    h, w = shape
    start_candidates = [
        p for p in support[ordered[0]]
        if p[0] in (0, h - 1) or p[1] in (0, w - 1)
    ]
    end_candidates = [
        p for p in support[ordered[-1]]
        if p[0] in (0, h - 1) or p[1] in (0, w - 1)
    ]
    assert len(start_candidates) == len(end_candidates) == 1
    points = [start_candidates[0]]
    points.extend(corners[frozenset((a, b))] for a, b in zip(ordered, ordered[1:]))
    points.append(end_candidates[0])

    path: set[Coord] = set()
    for a, b in zip(points, points[1:]):
        path.update(_segment(a, b))

    mirrors: list[Mirror] = []
    for i in range(1, len(points) - 1):
        before, corner, after = points[i - 1], points[i], points[i + 1]
        incoming = (_sign(corner[0] - before[0]), _sign(corner[1] - before[1]))
        outgoing = (_sign(after[0] - corner[0]), _sign(after[1] - corner[1]))
        row_flip = incoming[0] == -outgoing[0] and incoming[1] == outgoing[1]
        col_flip = incoming[0] == outgoing[0] and incoming[1] == -outgoing[1]
        assert row_flip ^ col_flip, "each corner must flip exactly one coordinate direction"
        center = (corner[0] + incoming[0], corner[1] + incoming[1])
        mirror = Mirror(center, "h" if row_flip else "v")
        assert all(0 <= r < h and 0 <= c < w for r, c in mirror.cells)
        mirrors.append(mirror)
    return DecodedRay(path, mirrors, ordered, points)


def _decode(grid: np.ndarray) -> DecodedRay:
    lines, support = _active_lines(grid)
    ordered, corners = _ordered_line_chain(grid, lines, support)
    retained = set(_red_mirrors(grid))
    candidates: list[DecodedRay] = []
    for order in (ordered, list(reversed(ordered))):
        decoded = _candidate_decode(order, corners, support, grid.shape)
        if retained.issubset(set(decoded.mirrors)):
            candidates.append(decoded)
    assert len(candidates) == 1, "retained mirror(s) must determine one travel direction"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the broken diagonal ray trace and restore every erased red mirror.

    Input structure:
    - A black grid contains blue fragments of one boundary-to-boundary diagonal ray with at least two turns.
    - One or more complete red length-three mirror segments remain; colored horizontal and vertical bars are distractors.
    - Every true diagonal segment has at least three visible blue cells, and only the two end segments touch the boundary.

    Rule:
    - Join the supported diagonal segments through their shared blue corners and use a retained red mirror to fix travel direction.
    - Fill black gaps between each segment's two physical endpoints with blue.
    - At every corner, flip exactly one direction component; place a horizontal red mirror for a row flip or a vertical one for a column flip, centered one incoming diagonal step beyond the corner.
    - Preserve retained mirrors and all colored distractor-bar cells.

    Key insight:
    - The corner joins identify the ray geometry, while the offset and orientation of a surviving red mirror disambiguate forward from reverse traversal; non-red bars crossed by blue are not reflectors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    decoded = _decode(g)
    out = g.copy()

    # Gaps are black by definition; colored bars, retained mirrors, and visible
    # blue crossing holes therefore remain untouched.
    for r, c in decoded.path:
        if int(out[r, c]) == BG:
            out[r, c] = BLUE

    retained = set(_red_mirrors(g))
    for mirror in decoded.mirrors:
        if mirror in retained:
            continue
        for r, c in mirror.cells:
            assert int(out[r, c]) == BG, "an erased mirror location must be empty"
            out[r, c] = RED
    return out


def _distance_to_boundary(p: Coord, direction: Coord, shape: tuple[int, int]) -> int:
    h, w = shape
    r, c = p
    dr, dc = direction
    row_room = h - 1 - r if dr > 0 else r
    col_room = w - 1 - c if dc > 0 else c
    return min(row_room, col_room)


def _new_direction(direction: Coord, axis: str) -> Coord:
    dr, dc = direction
    return (-dr, dc) if axis == "row" else (dr, -dc)


def _make_ray(h: int, w: int, turn_count: int):
    """Sample a non-self-intersecting reflected ray with both flip branches."""
    sides = ("top", "bottom", "left", "right")
    side = random.choice(sides)
    if side == "top":
        start = (0, random.randint(3, w - 4))
        direction = (1, random.choice((-1, 1)))
    elif side == "bottom":
        start = (h - 1, random.randint(3, w - 4))
        direction = (-1, random.choice((-1, 1)))
    elif side == "left":
        start = (random.randint(3, h - 4), 0)
        direction = (random.choice((-1, 1)), 1)
    else:
        start = (random.randint(3, h - 4), w - 1)
        direction = (random.choice((-1, 1)), -1)

    axes = ["row", "col"] + [random.choice(("row", "col")) for _ in range(turn_count - 2)]
    random.shuffle(axes)
    points = [start]
    mirror_specs: list[Mirror] = []
    used_path: set[Coord] = {start}
    used_mirrors: set[Coord] = set()
    current = start

    for axis in axes:
        room = _distance_to_boundary(current, direction, (h, w))
        choices = list(range(5, room))
        random.shuffle(choices)
        accepted = None
        for steps in choices:
            corner = (current[0] + steps * direction[0], current[1] + steps * direction[1])
            outgoing = _new_direction(direction, axis)
            if _distance_to_boundary(corner, outgoing, (h, w)) < 5:
                continue
            center = (corner[0] + direction[0], corner[1] + direction[1])
            mirror = Mirror(center, "h" if axis == "row" else "v")
            if not all(0 <= r < h and 0 <= c < w for r, c in mirror.cells):
                continue
            seg = set(_segment(current, corner))
            if (seg - {current}) & used_path:
                continue
            if set(mirror.cells) & (used_path | seg | used_mirrors):
                continue
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for r, c in mirror.cells for rr, cc in used_mirrors):
                continue
            accepted = (corner, outgoing, mirror, seg)
            break
        if accepted is None:
            return None
        corner, outgoing, mirror, seg = accepted
        used_path.update(seg)
        used_mirrors.update(mirror.cells)
        points.append(corner)
        mirror_specs.append(mirror)
        current, direction = corner, outgoing

    final_steps = _distance_to_boundary(current, direction, (h, w))
    if final_steps < 5:
        return None
    end = (current[0] + final_steps * direction[0], current[1] + final_steps * direction[1])
    last_seg = set(_segment(current, end))
    if (last_seg - {current}) & used_path or last_seg & used_mirrors:
        return None
    points.append(end)
    used_path.update(last_seg)

    lines = [_line_of(a, b) for a, b in zip(points, points[1:])]
    if len(lines) != len(set(lines)):
        return None
    if sum(len(_segment(a, b)) for a, b in zip(points, points[1:])) != len(used_path) + turn_count:
        return None
    return points, used_path, mirror_specs, axes


def _far_from(cells: set[Coord], occupied: set[Coord], distance: int = 1) -> bool:
    return all(
        max(abs(r - rr), abs(c - cc)) > distance
        for r, c in cells
        for rr, cc in occupied
    )


def _bar_candidates_through(anchor: Coord, orient: str, h: int, w: int):
    lengths = list(range(7, 12))
    random.shuffle(lengths)
    r, c = anchor
    for length in lengths:
        if orient == "h" and length <= w - 2:
            starts = list(range(max(1, c - length + 1), min(c, w - length - 1) + 1))
            random.shuffle(starts)
            for start in starts:
                yield {(r, x) for x in range(start, start + length)}
        elif orient == "v" and length <= h - 2:
            starts = list(range(max(1, r - length + 1), min(r, h - length - 1) + 1))
            random.shuffle(starts)
            for start in starts:
                yield {(x, c) for x in range(start, start + length)}


def _place_bars(
    h: int,
    w: int,
    path: set[Coord],
    mirrors: list[Mirror],
    forbidden_path: set[Coord],
):
    base = np.zeros((h, w), dtype=int)
    mirror_cells = {p for mirror in mirrors for p in mirror.cells}
    occupied: set[Coord] = set()
    all_bars: list[tuple[set[Coord], int, bool]] = []
    colors = random.sample(DISTRACTOR_COLORS, 3)

    # Force both visual bar orientations to be crossed. Their blue holes are
    # explicit counterexamples to treating every straight colored bar as a mirror.
    for orient, color in zip(("h", "v"), colors[:2]):
        anchors = list(path - forbidden_path - mirror_cells)
        random.shuffle(anchors)
        chosen = None
        for anchor in anchors:
            for cells in _bar_candidates_through(anchor, orient, h, w):
                if cells & path != {anchor} or cells & mirror_cells or cells & occupied:
                    continue
                if occupied and not _far_from(cells, occupied, 1):
                    continue
                chosen = cells
                break
            if chosen is not None:
                break
        if chosen is None:
            return None
        occupied.update(chosen)
        all_bars.append((chosen, color, True))

    # Add an un-crossed bar so intersection alone, rather than bar identity, is
    # visibly irrelevant to the reflection process.
    orient = random.choice(("h", "v"))
    chosen = None
    for _ in range(250):
        length = random.randint(7, 11)
        if orient == "h":
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - length - 1)
            cells = {(r, c) for c in range(c0, c0 + length)}
        else:
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - length - 1)
            cells = {(r, c) for r in range(r0, r0 + length)}
        if cells & (path | mirror_cells | occupied):
            continue
        if not _far_from(cells, occupied, 1):
            continue
        chosen = cells
        break
    if chosen is None:
        return None
    all_bars.append((chosen, colors[2], False))

    for cells, color, _crossed in all_bars:
        for r, c in cells:
            base[r, c] = color
    return base, all_bars


def _choose_gaps(points: list[Coord], base: np.ndarray, protected: set[Coord]):
    segments = [_segment(a, b) for a, b in zip(points, points[1:])]
    eligible = []
    options_by_segment: dict[int, list[tuple[Coord, ...]]] = {}
    for i, cells in enumerate(segments):
        options: list[tuple[Coord, ...]] = []
        for length in (1, 2, 3):
            for start in range(1, len(cells) - length):
                chunk = tuple(cells[start : start + length])
                if any(p in protected or int(base[p]) != BG for p in chunk):
                    continue
                if len(cells) - length < 4:
                    continue
                options.append(chunk)
        if options:
            eligible.append(i)
            options_by_segment[i] = options
    if len(eligible) < 2:
        return None

    count = random.randint(2, min(4, len(eligible)))
    chosen_segments = random.sample(eligible, count)
    gaps: list[tuple[Coord, ...]] = []
    for j, i in enumerate(chosen_segments):
        options = options_by_segment[i]
        if j == 0:
            longer = [chunk for chunk in options if len(chunk) >= 2]
            if not longer:
                return None
            gaps.append(random.choice(longer))
        else:
            gaps.append(random.choice(options))
    return gaps, chosen_segments


def _build_expected(base: np.ndarray, path: set[Coord], mirrors: list[Mirror]) -> np.ndarray:
    out = base.copy()
    for p in path:
        out[p] = BLUE
    for mirror in mirrors:
        for p in mirror.cells:
            assert int(out[p]) == BG
            out[p] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(20, 30)
    w = random.randint(20, 30)
    turn_count = random.randint(2, 4)
    ray = _make_ray(h, w, turn_count)
    if ray is None:
        return None
    points, path, mirrors, axes = ray
    assert "row" in axes and "col" in axes

    turns = set(points[1:-1])
    endpoints = {points[0], points[-1]}
    protected = turns | endpoints | {p for mirror in mirrors for p in mirror.cells}
    bars_result = _place_bars(h, w, path, mirrors, protected)
    if bars_result is None:
        return None
    base, bars = bars_result

    gap_result = _choose_gaps(points, base, protected)
    if gap_result is None:
        return None
    gaps, gap_segments = gap_result
    gap_cells = {p for chunk in gaps for p in chunk}
    if len(gap_segments) < 2 or max(map(len, gaps)) < 2:
        return None

    expected = _build_expected(base, path, mirrors)
    inp = base.copy()
    for p in path:
        inp[p] = BLUE
    retained = random.choice(mirrors)
    for p in retained.cells:
        inp[p] = RED
    for p in gap_cells:
        assert int(base[p]) == BG
        inp[p] = BG

    # Semantic validation: recover the intended geometry, direction, every
    # mirror branch, and every erased cell—not merely main(generate_input()).
    try:
        decoded = _decode(inp)
        if decoded.path != path or decoded.mirrors != mirrors:
            return None
        if set(_red_mirrors(inp)) != {retained}:
            return None
        if {m.orient for m in decoded.mirrors} != {"h", "v"}:
            return None
        active, _support = _active_lines(inp)
        intended_lines = {_line_of(a, b) for a, b in zip(points, points[1:])}
        if set(active) != intended_lines:
            return None
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return None
    if not np.array_equal(actual, expected):
        return None

    missing_mirror_cells = {p for m in mirrors if m != retained for p in m.cells}
    changed = set(map(tuple, np.argwhere(actual != inp)))
    if changed != gap_cells | missing_mirror_cells:
        return None
    if not gap_cells or not missing_mirror_cells:
        return None
    for cells, _color, _crossed in bars:
        for p in cells - path:
            if int(actual[p]) != int(inp[p]):
                return None
    crossed = [cells & path for cells, _color, flag in bars if flag]
    if len(crossed) != 2 or any(len(cells) != 1 for cells in crossed):
        return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely decodable broken-ray instance."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-mirror ray input")
