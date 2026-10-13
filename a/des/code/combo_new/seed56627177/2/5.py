import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

OUTPUT_COLORS = [BLUE, YELLOW, ORANGE, RED]

GEN_ATTEMPTS = 400
GRID_MIN = 10
GRID_MAX = 28
AXIS_START = 3
AXIS_STEP = 5
TAIL_MIN = 1
TAIL_MAX = 3


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _row_runs(mask: np.ndarray) -> list[dict[str, int | str]]:
    h, w = mask.shape
    runs: list[dict[str, int | str]] = []
    for r in range(h):
        c = 0
        while c < w:
            if not mask[r, c]:
                c += 1
                continue
            c0 = c
            while c + 1 < w and mask[r, c + 1]:
                c += 1
            c1 = c
            if c1 - c0 + 1 >= 2:
                runs.append({"ori": "H", "r": r, "c0": c0, "c1": c1})
            c += 1
    return runs


def _col_runs(mask: np.ndarray) -> list[dict[str, int | str]]:
    h, w = mask.shape
    runs: list[dict[str, int | str]] = []
    for c in range(w):
        r = 0
        while r < h:
            if not mask[r, c]:
                r += 1
                continue
            r0 = r
            while r + 1 < h and mask[r + 1, c]:
                r += 1
            r1 = r
            if r1 - r0 + 1 >= 2:
                runs.append({"ori": "V", "c": c, "r0": r0, "r1": r1})
            r += 1
    return runs


def _segment_cells(seg: dict[str, int | str]) -> list[tuple[int, int]]:
    if seg["ori"] == "H":
        r = int(seg["r"])
        return [(r, c) for c in range(int(seg["c0"]), int(seg["c1"]) + 1)]
    c = int(seg["c"])
    return [(r, c) for r in range(int(seg["r0"]), int(seg["r1"]) + 1)]


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, int | str]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) == 2, f"expected exactly two foreground colors, got {colors}"

    color_counts = Counter(map(int, g.reshape(-1)))
    line_color, mark_color = sorted(colors, key=lambda color: color_counts[color], reverse=True)
    occupied = (g == line_color) | (g == mark_color)

    segments = _row_runs(occupied) + _col_runs(occupied)
    assert 2 <= len(segments) <= 4, f"expected 2-4 segments, got {len(segments)}"

    for seg_id, seg in enumerate(segments):
        seg["id"] = seg_id

    memberships: dict[tuple[int, int], list[int]] = defaultdict(list)
    for seg in segments:
        for cell in _segment_cells(seg):
            memberships[cell].append(int(seg["id"]))

    occupied_cells = set(map(tuple, np.argwhere(occupied)))
    assert occupied_cells == set(memberships), "every occupied cell must belong to at least one segment and vice versa"

    for (r, c), ids in memberships.items():
        if g[r, c] == mark_color:
            assert len(ids) == 1, "marker cells must belong to exactly one segment"
        else:
            assert 1 <= len(ids) <= 2, "line cells must belong to one segment or one crossing pair"

    return bg, line_color, mark_color, segments


def _crosses(hseg: dict[str, int | str], vseg: dict[str, int | str]) -> bool:
    return (
        int(hseg["c0"]) <= int(vseg["c"]) <= int(hseg["c1"])
        and int(vseg["r0"]) <= int(hseg["r"]) <= int(vseg["r1"])
    )


def _unique_topological_order(segment_ids: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {sid: 0 for sid in segment_ids}
    for src in segment_ids:
        for dst in edges[src]:
            indegree[dst] += 1

    frontier = deque(sorted([sid for sid in segment_ids if indegree[sid] == 0]))
    order: list[int] = []

    while frontier:
        assert len(frontier) == 1, "stacking order must be a unique total order"
        cur = frontier.popleft()
        order.append(cur)
        for nxt in sorted(edges[cur]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)

    assert len(order) == len(segment_ids), "stacking graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor intersecting straight paths by recovering their global over-under stacking order.

    Input structure:
    - The background is black.
    - Several straight horizontal or vertical path segments are drawn in one line color.
    - Whenever a horizontal segment crosses a vertical segment, exactly one red pixel sits
      in one of the four neighboring cells around the crossing on the segment that passes over.

    Rule:
    - Treat each maximal horizontal run and maximal vertical run of line-or-marker pixels as one segment.
    - At every crossing, a left/right red neighbor means the horizontal segment is above the vertical one;
      an up/down red neighbor means the vertical segment is above the horizontal one.
    - Combine all local above/below relations into the unique global bottom-to-top order.
    - Recolor segments from bottom upward with blue, yellow, orange, then red, using only the needed prefix.
      The crossing pixel takes the upper segment color, and all red marker pixels are removed to background.

    Key insight:
    - The red pixels encode only local pairwise precedence, so the task is to rebuild the hidden
      total ordering of the full segments before repainting the shared crossings.
    """
    g = np.asarray(grid, dtype=int)
    bg, line_color, mark_color, segments = _extract_scene(g)
    h, w = g.shape

    horizontal = [seg for seg in segments if seg["ori"] == "H"]
    vertical = [seg for seg in segments if seg["ori"] == "V"]

    edges: dict[int, set[int]] = {int(seg["id"]): set() for seg in segments}
    used_markers: set[tuple[int, int]] = set()

    for hseg in horizontal:
        for vseg in vertical:
            if not _crosses(hseg, vseg):
                continue

            r = int(hseg["r"])
            c = int(vseg["c"])
            assert g[r, c] == line_color, "crossing pixel must stay in the base line color"

            directions: list[str] = []
            if c - 1 >= 0 and g[r, c - 1] == mark_color:
                directions.append("L")
                used_markers.add((r, c - 1))
            if c + 1 < w and g[r, c + 1] == mark_color:
                directions.append("R")
                used_markers.add((r, c + 1))
            if r - 1 >= 0 and g[r - 1, c] == mark_color:
                directions.append("U")
                used_markers.add((r - 1, c))
            if r + 1 < h and g[r + 1, c] == mark_color:
                directions.append("D")
                used_markers.add((r + 1, c))

            assert len(directions) == 1, "each crossing must have exactly one adjacent red marker"
            upper = int(hseg["id"]) if directions[0] in {"L", "R"} else int(vseg["id"])
            lower = int(vseg["id"]) if upper == int(hseg["id"]) else int(hseg["id"])
            edges[lower].add(upper)

    marker_cells = set(map(tuple, np.argwhere(g == mark_color)))
    assert marker_cells == used_markers, "every red marker must belong to one crossing"

    order = _unique_topological_order([int(seg["id"]) for seg in segments], edges)
    assert len(order) <= len(OUTPUT_COLORS), "output palette supports at most four stacked segments"
    segment_colors = {sid: OUTPUT_COLORS[idx] for idx, sid in enumerate(order)}

    out = np.full_like(g, bg)
    for sid in order:
        seg = segments[sid]
        color = int(segment_colors[sid])
        for r, c in _segment_cells(seg):
            if g[r, c] == mark_color:
                continue
            out[r, c] = color

    return out


def _sample_canvas(required_slots: int) -> int:
    for _ in range(80):
        size = random.randint(GRID_MIN, GRID_MAX)
        slots = list(range(AXIS_START, size - AXIS_START + 1, AXIS_STEP))
        if len(slots) >= required_slots:
            return size
    raise RuntimeError("failed to sample a large enough canvas")


def _pick_positions(count: int, size: int) -> list[int]:
    slots = list(range(AXIS_START, size - AXIS_START + 1, AXIS_STEP))
    assert len(slots) >= count
    return random.sample(slots, count)


def _cross_plan(segment_ids: list[int], orientations: dict[int, str], dense: bool) -> dict[int, set[int]]:
    plan: dict[int, set[int]] = {sid: set() for sid in segment_ids}
    if dense:
        horiz = [sid for sid in segment_ids if orientations[sid] == "H"]
        vert = [sid for sid in segment_ids if orientations[sid] == "V"]
        for hid in horiz:
            for vid in vert:
                plan[hid].add(vid)
                plan[vid].add(hid)
        return plan

    for sid in segment_ids:
        if sid - 1 >= 0 and orientations[sid - 1] != orientations[sid]:
            plan[sid].add(sid - 1)
        if sid + 1 < len(segment_ids) and orientations[sid + 1] != orientations[sid]:
            plan[sid].add(sid + 1)
    return plan


def _gen_attempt() -> np.ndarray | None:
    segment_count = random.choice([2, 3, 4])
    start_ori = random.choice(["H", "V"])
    orientations = {
        sid: (start_ori if sid % 2 == 0 else ("V" if start_ori == "H" else "H"))
        for sid in range(segment_count)
    }
    dense = segment_count == 4 and random.random() < 0.5

    horiz_ids = [sid for sid, ori in orientations.items() if ori == "H"]
    vert_ids = [sid for sid, ori in orientations.items() if ori == "V"]

    h = _sample_canvas(len(horiz_ids))
    w = _sample_canvas(len(vert_ids))
    row_values = _pick_positions(len(horiz_ids), h)
    col_values = _pick_positions(len(vert_ids), w)
    random.shuffle(row_values)
    random.shuffle(col_values)

    row_by_id = {sid: row_values[idx] for idx, sid in enumerate(horiz_ids)}
    col_by_id = {sid: col_values[idx] for idx, sid in enumerate(vert_ids)}
    crosses = _cross_plan(list(range(segment_count)), orientations, dense)

    grid = np.full((h, w), BLACK, dtype=int)
    segments: dict[int, dict[str, int | str]] = {}

    for sid in horiz_ids:
        cols = sorted(col_by_id[other] for other in crosses[sid])
        c0 = min(cols) - random.randint(TAIL_MIN, TAIL_MAX)
        c1 = max(cols) + random.randint(TAIL_MIN, TAIL_MAX)
        if not (0 <= c0 < c1 < w):
            return None
        r = row_by_id[sid]
        grid[r, c0 : c1 + 1] = GREEN
        segments[sid] = {"id": sid, "ori": "H", "r": r, "c0": c0, "c1": c1}

    for sid in vert_ids:
        rows = sorted(row_by_id[other] for other in crosses[sid])
        r0 = min(rows) - random.randint(TAIL_MIN, TAIL_MAX)
        r1 = max(rows) + random.randint(TAIL_MIN, TAIL_MAX)
        if not (0 <= r0 < r1 < h):
            return None
        c = col_by_id[sid]
        grid[r0 : r1 + 1, c] = GREEN
        segments[sid] = {"id": sid, "ori": "V", "c": c, "r0": r0, "r1": r1}

    marker_dir: dict[int, int] = {}
    for sid in range(segment_count):
        marker_dir[sid] = random.choice([-1, 1])

    intersections = {
        (int(segments[hid]["r"]), int(segments[vid]["c"]))
        for hid in horiz_ids
        for vid in vert_ids
        if hid in crosses[vid]
    }
    used_markers: set[tuple[int, int]] = set()

    for hid in horiz_ids:
        for vid in vert_ids:
            if vid not in crosses[hid]:
                continue
            r = int(segments[hid]["r"])
            c = int(segments[vid]["c"])
            upper = max(hid, vid)
            upper_seg = segments[upper]

            if upper_seg["ori"] == "H":
                dc = marker_dir[upper]
                cell = (r, c + dc)
                if not (int(upper_seg["c0"]) <= cell[1] <= int(upper_seg["c1"])):
                    return None
            else:
                dr = marker_dir[upper]
                cell = (r + dr, c)
                if not (int(upper_seg["r0"]) <= cell[0] <= int(upper_seg["r1"])):
                    return None

            if cell in intersections or cell in used_markers:
                return None
            if not (0 <= cell[0] < h and 0 <= cell[1] < w):
                return None
            if grid[cell] != GREEN:
                return None

            grid[cell] = RED
            used_markers.add(cell)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stacking-order ARC input")
