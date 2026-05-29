# Combo variant of 581f7754 (index 15)
# combo_spec: D5/1+D6/1+D8/0
# Name: 碰撞停止+滑动留痕+按距离排序

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Keep the original object vocabulary: isolated border anchors and interior multi-color sprites whose
#      least-frequent color is the marker.
#    - The fused rule changes the execution model itself:
#      1. match each sprite to the anchor of the same marker color;
#      2. sort sprites by marker-to-anchor distance from near to far;
#      3. process them one at a time, removing only the current sprite from the board;
#      4. slide step by step along the anchor-implied axis until the next step would hit any non-bg cell
#         or leave the grid;
#      5. paste the sprite at that last reachable position and paint the traversed marker path in the
#         marker color.
#    - Small horizontal example: anchor is on the top edge at column 9; two sprites with the same marker
#      color sit on row 6, marker columns 7 and 3.
#      Original task: both end with marker column 9, with no order effect and no trail.
#      Fused task: the column-7 sprite moves first, reaches column 9, and paints a trail on columns 7..8.
#      Then the column-3 sprite moves; when its next step would enter column 7 it collides with that trail,
#      so it stops earlier instead of reaching column 9.
# b) Complexity calibration
#    - The trail color is exactly the marker color; this keeps the extra visual language small and legible.
#    - Generator forces at least one shared lane with 2-3 sprites so order and interference are visible.
#    - The lane order is purely geometric: distance to the matched anchor, never color id or hidden metadata.
# c) Hidden assumptions to revisit
#    - The base solver's "clear all movers first, then paste all at once" no longer works because it erases
#      the very occupancy information that later sprites must react to.
#    - The base solver's destination-only collision logic also fails: in the combo task, path cells matter,
#      because both settled sprites and earlier trails can stop motion before target alignment.
#    - Trail painting must happen after each sprite is pasted so that later sprites see the trail as an
#      obstacle, while the current sprite is not blocked by its own trail.
# d) Placement legality
#    - Generator keeps all input sprites 8-neighbor separated and fully inside the grid.
#    - Shared-lane sprites start on one side of the same anchor with strictly increasing distances, so the
#      near-to-far sort is visible and stable.
#    - Every candidate grid is validated by the trace solver and rejected unless it exhibits the required
#      sequential collision behavior.
# e) Visual observability
#    - Humans can see three consistent cues together: anchor color chooses the destination axis, nearer same-
#      color sprites move first, and marker-colored lines are exactly the swept path that can block later
#      sprites in the same lane.
# f) Visual cleanliness
#    - The generator prefers one primary lane and only a small number of extra sprites, which keeps the
#      output readable instead of filling the board with intersecting trails.
#    - Trail width is always one cell.
# g) New solver assumptions and how the generator satisfies them
#    - Assumption: a movable component has a unique anchor-colored marker. Generator uses two-color sprites
#      with one marker pixel.
#    - Assumption: later sprites can be blocked by cells already written by earlier steps. Generator filters
#      for at least one sample where the blocking category is specifically a trail, not only a final body.
#    - Assumption: the distance order is meaningful. Generator explicitly creates a shared lane whose sprite
#      distances are strictly increasing.


# Shared constants.
_COLORS = tuple(range(10))
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Canvas and placement.
_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1
_SAFE_LANE_MARGIN = 2

# Counts and retries.
_GEN_RETRY_LIMIT = 600
_SPRITE_BUILD_RETRIES = 80
_LANE_PLACEMENT_RETRIES = 100
_ANCHOR_COUNT_CHOICES = (1, 2)
_SPRITE_COUNT_RANGE = (3, 5)
_PRIMARY_LANE_SIZE_RANGE = (2, 3)
_MIN_MOVE_STEPS = 2

# Shape mix.
_BODY_AREA_RANGE = (4, 13)
_BODY_HEIGHT_RANGE = (2, 5)
_BODY_WIDTH_RANGE = (3, 7)
_FRAME_HEIGHT_RANGE = (3, 5)
_FRAME_WIDTH_RANGE = (3, 7)
_FRAME_MIN_AREA = 8
_FRAME_MAX_GAP = 3
_FRAME_BODY_THRESHOLD = 0.28
_WIDE_BODY_THRESHOLD = 0.55
_SYMMETRIC_BODY_THRESHOLD = 0.82
_SYMMETRIC_TRANSPOSE_MIN_HEIGHT = 3
_SYMMETRIC_TRANSPOSE_PROB = 0.45
_FLIP_PROB = 0.5


def _norm(cells):
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _bbox(cells):
    return (
        min(r for r, _c in cells),
        min(c for _r, c in cells),
        max(r for r, _c in cells),
        max(c for _r, c in cells),
    )


def _flip(cells):
    cells = set(cells)
    if random.random() < _FLIP_PROB:
        mc = max(c for _r, c in cells)
        cells = {(r, mc - c) for r, c in cells}
    if random.random() < _FLIP_PROB:
        mr = max(r for r, _c in cells)
        cells = {(mr - r, c) for r, c in cells}
    return _norm(cells)


def _inclusive_range(bounds):
    return range(int(bounds[0]), int(bounds[1]) + 1)


def _frontier_cells(cells, height, width):
    frontier = []
    for r, c in cells:
        for dr, dc in _NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                frontier.append((rr, cc))
    return frontier


_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


def _grow_body(area):
    choices = [
        (h, w)
        for h in _inclusive_range(_BODY_HEIGHT_RANGE)
        for w in _inclusive_range(_BODY_WIDTH_RANGE)
        if h * w >= area
    ]
    h, w = random.choice(choices)
    cells = set(random.choice(_GROW_BODY_SEEDS))
    while len(cells) < area:
        frontier = _frontier_cells(cells, h, w)
        if not frontier:
            return _grow_body(area)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _symmetric_body(area):
    choices = []
    for height in _inclusive_range(_BODY_HEIGHT_RANGE):
        extra = area - height
        if extra >= 2 and extra % 2 == 0 and extra // 2 <= height:
            choices.append((height, extra // 2))
    if not choices:
        return _grow_body(area)

    height, pair_count = random.choice(choices)
    cells = {(r, 1) for r in range(height)}
    rows = list(range(height))
    random.shuffle(rows)
    for r in rows[:pair_count]:
        cells.add((r, 0))
        cells.add((r, 2))

    if height >= _SYMMETRIC_TRANSPOSE_MIN_HEIGHT and random.random() < _SYMMETRIC_TRANSPOSE_PROB:
        cells = {(c, r) for r, c in cells}
    return _flip(cells)


def _wide_body(area):
    width = random.randint(_BODY_WIDTH_RANGE[0], min(_BODY_WIDTH_RANGE[1], area - 1))
    height = random.choice([h for h in _inclusive_range(_BODY_HEIGHT_RANGE) if h * width >= area])
    cells = {(0, c) for c in range(width)}
    cells.add((1, random.randrange(width)))
    while len(cells) < area:
        frontier = _frontier_cells(cells, height, width)
        cells.add(random.choice(frontier))
    return _flip(cells)


def _frame_body(area):
    choices = []
    for h in _inclusive_range(_FRAME_HEIGHT_RANGE):
        for w in _inclusive_range(_FRAME_WIDTH_RANGE):
            perim = 2 * h + 2 * w - 4
            removable = 2 * max(0, h - 2) + 2 * max(0, w - 2)
            if perim >= area and perim - area <= min(_FRAME_MAX_GAP, removable):
                choices.append((h, w))
    if not choices:
        return _grow_body(area)

    h, w = random.choice(choices)
    cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    gap = len(cells) - area
    if gap:
        removable = [
            (r, c)
            for r, c in cells
            if not (r in (0, h - 1) and c in (0, w - 1))
        ]
        for cell in random.sample(removable, gap):
            cells.remove(cell)
    return _flip(cells)


def _make_body(area):
    x = random.random()
    if area >= _FRAME_MIN_AREA and x < _FRAME_BODY_THRESHOLD:
        return _frame_body(area)
    if x < _WIDE_BODY_THRESHOLD:
        return _wide_body(area)
    if x < _SYMMETRIC_BODY_THRESHOLD:
        return _symmetric_body(area)
    return _grow_body(area)


def _directed_marker_cell(body, axis: str, step: int):
    r0, c0, r1, c1 = _bbox(body)
    preferred = []
    fallback = []
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            fallback.append(p)
            if axis == "h":
                if step > 0 and p[1] > c1:
                    preferred.append(p)
                if step < 0 and p[1] < c0:
                    preferred.append(p)
            else:
                if step > 0 and p[0] > r1:
                    preferred.append(p)
                if step < 0 and p[0] < r0:
                    preferred.append(p)
    pool = list(set(preferred or fallback))
    return random.choice(pool) if pool else None


def _make_directed_sprite(area, body_color, marker_color, bg, axis: str, step: int):
    for _ in range(_SPRITE_BUILD_RETRIES):
        body = _make_body(area)
        marker = _directed_marker_cell(body, axis, step)
        if marker is None:
            continue
        all_cells = set(body) | {marker}
        r0, c0, r1, c1 = _bbox(all_cells)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
        for r, c in body:
            patch[r - r0, c - c0] = int(body_color)
        patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
        return {"patch": patch, "offset": (int(marker[0] - r0), int(marker[1] - c0))}
    return None


def _top_left(sprite, marker_r, marker_c):
    off_r, off_c = sprite["offset"]
    return int(marker_r) - int(off_r), int(marker_c) - int(off_c)


def _touches_foreground(canvas, mask, top, left, bg):
    h, w = mask.shape
    radius = _ISOLATION_RADIUS
    occupied = canvas[top - radius : top + h + radius, left - radius : left + w + radius] != int(bg)
    touch_zone = np.zeros((h + 2 * radius, w + 2 * radius), dtype=bool)
    span = 2 * radius + 1
    for dr in range(span):
        for dc in range(span):
            touch_zone[dr : dr + h, dc : dc + w] |= mask
    return bool((occupied & touch_zone).any())


def _can_place(canvas, sprite, marker_r, marker_c, bg):
    patch = sprite["patch"]
    top, left = _top_left(sprite, marker_r, marker_c)
    h, w = patch.shape
    H, W = canvas.shape
    if top < _INNER_BORDER_GAP or left < _INNER_BORDER_GAP:
        return False
    if top + h > H - _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
        return False
    mask = patch != int(bg)
    view = canvas[top : top + h, left : left + w]
    if bool((view[mask] != int(bg)).any()):
        return False
    return not _touches_foreground(canvas, mask, top, left, bg)


def _place(canvas, sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    paste(canvas, sprite["patch"], top, left, transparent=int(bg))


def _is_border(r: int, c: int, H: int, W: int) -> bool:
    return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not _is_border(int(r), int(c), int(H), int(W)):
        return False
    if r in (0, H - 1) and c in (0, W - 1):
        return False
    return _isolated_bg_cell(canvas, r, c, bg)


def _anchor_candidates(canvas, bg):
    H, W = canvas.shape
    rows = list(range(_ANCHOR_EDGE_MARGIN, H - _ANCHOR_EDGE_MARGIN))
    rows = rows or list(range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP))
    cols = list(range(_ANCHOR_EDGE_MARGIN, W - _ANCHOR_EDGE_MARGIN))
    cols = cols or list(range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP))
    out = []
    for c in cols:
        out.append(("h", 0, c))
        out.append(("h", H - 1, c))
    for r in rows:
        out.append(("v", r, 0))
        out.append(("v", r, W - 1))
    random.shuffle(out)
    return [(axis, r, c) for axis, r, c in out if _anchor_ok(canvas, r, c, bg)]


def _choose_axis_plan(total_sprites: int) -> list[str]:
    modes = ["h", "h", "h", "both", "both", "v"]
    mode = random.choice(modes)
    if mode == "both" and int(total_sprites) >= 3:
        axes = ["h", "v"]
        random.shuffle(axes)
        return axes
    return [mode]


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _component_marker_color(g: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(g[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(g: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    S = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in S and int(g[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _detect_anchors(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    H, W = g.shape
    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not _is_border(int(r), int(c), int(H), int(W)):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and v not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))
    return anchors_by_color


def _extract_moves(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    anchors_by_color = _detect_anchors(g, int(bg))
    assert anchors_by_color
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    moves = []
    for comp in comps:
        coords = comp["coords"]
        if coords.size == 0:
            continue

        marker_col = _component_marker_color(g, coords)
        if marker_col is None:
            if int(coords.shape[0]) != 1:
                continue
            r, c = map(int, coords[0])
            v = int(g[r, c])
            if v == int(bg) or v not in anchors_by_color or (r, c) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = bool(int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1)))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])
        distance = abs(int(dc) if horizontal else int(dr))
        step_r = 0 if int(dr) == 0 else (1 if int(dr) > 0 else -1)
        step_c = 0 if int(dc) == 0 else (1 if int(dc) > 0 else -1)
        lane_key = (
            int(ar),
            int(ac),
            "h" if horizontal else "v",
            int(mp[0]) if horizontal else int(mp[1]),
            int(step_c if horizontal else step_r),
        )
        moves.append(
            {
                "coords": coords.astype(int),
                "marker_color": int(marker_col),
                "marker_pixel": (int(mp[0]), int(mp[1])),
                "anchor": (int(ar), int(ac)),
                "horizontal": bool(horizontal),
                "dr": int(dr),
                "dc": int(dc),
                "distance": int(distance),
                "step_r": int(step_r),
                "step_c": int(step_c),
                "lane_key": lane_key,
            }
        )

    moves.sort(key=lambda m: (int(m["distance"]), int(m["marker_pixel"][0]), int(m["marker_pixel"][1])))
    return g, bg, anchors_by_color, moves


def _marker_trail_cells(src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, src)
    dr = int(dst[0] - sr)
    dc = int(dst[1] - sc)
    steps = int(max(abs(dr), abs(dc)))
    if steps <= 0:
        return []
    step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
    step_c = 0 if dc == 0 else (1 if dc > 0 else -1)
    return [(int(sr + k * step_r), int(sc + k * step_c)) for k in range(steps)]


def _paste_translated(board: np.ndarray, src: np.ndarray, coords: np.ndarray, off_r: int, off_c: int):
    for r, c in coords:
        board[int(r) + int(off_r), int(c) + int(off_c)] = int(src[int(r), int(c)])


def _run_original_simultaneous(g: np.ndarray, bg: int, moves: list[dict]) -> np.ndarray:
    board = np.array(g, copy=True)
    H, W = board.shape
    for move in moves:
        coords = move["coords"]
        board[coords[:, 0], coords[:, 1]] = int(bg)
    for move in moves:
        coords = move["coords"]
        for r, c in coords:
            rr = int(r) + int(move["dr"])
            cc = int(c) + int(move["dc"])
            if 0 <= rr < H and 0 <= cc < W and int(board[rr, cc]) == int(bg):
                board[rr, cc] = int(g[int(r), int(c)])
    return board


def _run_sequential(g: np.ndarray, bg: int, anchors_by_color: dict[int, tuple[int, int]], moves: list[dict], *, paint_trails: bool, return_trace: bool = False):
    board = np.array(g, copy=True)
    labels = np.zeros(board.shape, dtype=np.int8)
    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
    for r, c in anchor_cells:
        labels[int(r), int(c)] = 1
    for r, c in np.argwhere(board != int(bg)):
        if labels[int(r), int(c)] == 0:
            labels[int(r), int(c)] = 3

    trace = []
    H, W = board.shape
    for move in moves:
        coords = move["coords"]
        board[coords[:, 0], coords[:, 1]] = int(bg)
        labels[coords[:, 0], coords[:, 1]] = 0

        max_steps = int(move["distance"])
        step_r = int(move["step_r"])
        step_c = int(move["step_c"])
        actual_steps = 0
        blocker_codes: set[int] = set()

        for step in range(1, max_steps + 1):
            blocked = False
            step_blockers: set[int] = set()
            for r, c in coords:
                rr = int(r) + step_r * step
                cc = int(c) + step_c * step
                if not (0 <= rr < H and 0 <= cc < W):
                    blocked = True
                    step_blockers.add(4)
                    break
                if int(board[rr, cc]) != int(bg):
                    blocked = True
                    step_blockers.add(int(labels[rr, cc]))
            if blocked:
                blocker_codes = step_blockers
                break
            actual_steps = int(step)

        off_r = step_r * actual_steps
        off_c = step_c * actual_steps
        _paste_translated(board, g, coords, off_r, off_c)
        for r, c in coords:
            labels[int(r) + off_r, int(c) + off_c] = 3

        src_marker = move["marker_pixel"]
        dst_marker = (int(src_marker[0] + off_r), int(src_marker[1] + off_c))
        if paint_trails:
            for r, c in _marker_trail_cells(src_marker, dst_marker):
                if int(board[r, c]) == int(bg):
                    board[r, c] = int(move["marker_color"])
                    labels[r, c] = 2

        if return_trace:
            names = tuple(
                sorted(
                    {
                        {1: "anchor", 2: "trail", 3: "sprite", 4: "wall"}.get(int(code), "sprite")
                        for code in blocker_codes
                    }
                )
            )
            trace.append(
                {
                    **move,
                    "actual_steps": int(actual_steps),
                    "blocked": bool(actual_steps < max_steps),
                    "blockers": names,
                    "dst_marker": dst_marker,
                }
            )

    if return_trace:
        return board, trace
    return board


def _variant_metrics(grid: np.ndarray):
    g, bg, anchors_by_color, moves = _extract_moves(grid)
    out_combo, trace = _run_sequential(g, bg, anchors_by_color, moves, paint_trails=True, return_trace=True)
    out_no_trail = _run_sequential(g, bg, anchors_by_color, moves, paint_trails=False, return_trace=False)
    out_original = _run_original_simultaneous(g, bg, moves)
    lane_sizes = Counter(move["lane_key"] for move in moves)
    blocker_hist = Counter(
        "/".join(item["blockers"]) if item["blockers"] else "none"
        for item in trace
        if item["blocked"]
    )
    return {
        "move_count": int(len(moves)),
        "moved_count": int(sum(1 for item in trace if int(item["actual_steps"]) > 0)),
        "reached_target_count": int(sum(1 for item in trace if int(item["actual_steps"]) == int(item["distance"]))),
        "blocked_count": int(sum(1 for item in trace if item["blocked"])),
        "trail_blocked_count": int(sum(1 for item in trace if item["blocked"] and "trail" in item["blockers"])),
        "shared_lane_max": int(max(lane_sizes.values(), default=0)),
        "changed_vs_original": bool(not np.array_equal(out_combo, out_original)),
        "changed_vs_no_trail": bool(not np.array_equal(out_combo, out_no_trail)),
        "blocker_hist": dict(blocker_hist),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward matching border anchors, resolving the moves sequentially with collision stops and marker trails.

    Input structure:
    - The background is the most common color.
    - Interior sprites are 8-connected foreground components with a least-frequent marker pixel whose color matches a border anchor.
    - Border anchors are isolated non-background singletons, and several sprites can share one anchor color and one slide lane.

    Rule:
    - Match each sprite to the anchor of the same marker color.
    - A top/bottom anchor requests horizontal motion; a left/right anchor requests vertical motion.
    - Sort sprites by their marker-to-anchor distance along that axis from near to far, then move them one by one.
    - On its turn, remove only that sprite, step toward the target line until the next step would leave the grid or hit any non-background cell, paste the sprite there, and paint the marker's traversed cells as a one-cell-wide trail in the marker color.

    Key insight:
    - The anchor still determines the slide axis, but the visible result now depends on execution order: earlier sprites occupy space and leave trails, and those newly written cells can stop later sprites before they reach the anchor line.
    """
    g, bg, anchors_by_color, moves = _extract_moves(np.array(grid, dtype=int))
    return _run_sequential(g, bg, anchors_by_color, moves, paint_trails=True, return_trace=False)


def _body_color(bg, marker, used):
    pool = [c for c in _COLORS if c not in {int(bg), int(marker)} and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in {int(bg), int(marker)}]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _lane_sides(axis: str):
    return ["left", "right"] if axis == "h" else ["top", "bottom"]


def _lane_direction(axis: str, side: str) -> int:
    if axis == "h":
        return 1 if side == "left" else -1
    return 1 if side == "top" else -1


def _lane_lines(shape: tuple[int, int], axis: str):
    H, W = shape
    if axis == "h":
        return list(range(_SAFE_LANE_MARGIN, H - _SAFE_LANE_MARGIN))
    return list(range(_SAFE_LANE_MARGIN, W - _SAFE_LANE_MARGIN))


def _lane_positions(canvas, sprite, anchor, side: str, line: int, bg):
    H, W = canvas.shape
    axis = anchor["axis"]
    positions = []
    if axis == "h":
        ac = int(anchor["c"])
        cols = range(ac - 1, _INNER_BORDER_GAP - 1, -1) if side == "left" else range(ac + 1, W - _INNER_BORDER_GAP)
        for c in cols:
            dist = abs(int(c) - ac)
            if dist < _MIN_MOVE_STEPS:
                continue
            if _can_place(canvas, sprite, int(line), int(c), bg):
                positions.append(((int(line), int(c)), int(dist)))
    else:
        ar = int(anchor["r"])
        rows = range(ar - 1, _INNER_BORDER_GAP - 1, -1) if side == "top" else range(ar + 1, H - _INNER_BORDER_GAP)
        for r in rows:
            dist = abs(int(r) - ar)
            if dist < _MIN_MOVE_STEPS:
                continue
            if _can_place(canvas, sprite, int(r), int(line), bg):
                positions.append(((int(r), int(line)), int(dist)))
    return positions


def _place_lane(canvas, anchor, count, bg, used_body):
    axis = anchor["axis"]
    lines = _lane_lines(canvas.shape, axis)
    random.shuffle(lines)
    sides = _lane_sides(axis)
    random.shuffle(sides)

    for line in lines[:_LANE_PLACEMENT_RETRIES]:
        for side in sides:
            temp = canvas.copy()
            temp_used = set(used_body)
            placements = []
            last_distance = 0
            step = _lane_direction(axis, side)
            ok = True
            for _ in range(count):
                placed = False
                for _try in range(_SPRITE_BUILD_RETRIES):
                    marker = int(anchor["color"])
                    body = _body_color(bg, marker, temp_used)
                    sprite = _make_directed_sprite(
                        random.randint(*_BODY_AREA_RANGE),
                        body,
                        marker,
                        bg,
                        axis,
                        step,
                    )
                    if sprite is None:
                        continue
                    candidates = [
                        (pos, dist)
                        for pos, dist in _lane_positions(temp, sprite, anchor, side, line, bg)
                        if int(dist) > int(last_distance)
                    ]
                    if not candidates:
                        continue
                    candidates.sort(key=lambda item: item[1])
                    take = candidates[: min(5, len(candidates))]
                    pos, dist = random.choice(take)
                    _place(temp, sprite, pos[0], pos[1], bg)
                    temp_used.add(int(body))
                    placements.append(
                        {
                            "sprite": sprite,
                            "marker_pos": (int(pos[0]), int(pos[1])),
                            "distance": int(dist),
                            "side": side,
                            "line": int(line),
                            "anchor": anchor,
                        }
                    )
                    last_distance = int(dist)
                    placed = True
                    break
                if not placed:
                    ok = False
                    break
            if ok:
                canvas[:, :] = temp
                used_body |= temp_used
                return placements
    return None


def _new_canvas():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    return np.full((H, W), int(bg), dtype=int), int(bg)


def _place_anchors(canvas, bg, axis_plan: list[str]):
    colors = random.sample([c for c in _COLORS if c != int(bg)], len(axis_plan))
    anchors = []
    for color, desired_axis in zip(colors, axis_plan):
        candidates = _anchor_candidates(canvas, bg)
        preferred = [item for item in candidates if item[0] == desired_axis]
        if not preferred:
            return None
        axis, r, c = random.choice(preferred)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        _put_anchor(canvas, r, c, color)
        anchors.append(anchor)
    return anchors


def _unique_move_distances(moves: list[dict]) -> bool:
    distances = [int(move["distance"]) for move in moves]
    return len(distances) == len(set(distances))


def _generated_grid_is_valid(grid: np.ndarray, bg: int):
    if not _bg_is_mode(grid, bg):
        return False
    metrics = _variant_metrics(grid)
    _, _, anchors_by_color, moves = _extract_moves(grid)
    if int(metrics["move_count"]) < 3:
        return False
    if not _unique_move_distances(moves):
        return False
    if int(metrics["shared_lane_max"]) < 2:
        return False
    if int(metrics["blocked_count"]) < 1:
        return False
    if int(metrics["trail_blocked_count"]) < 1:
        return False
    if int(metrics["reached_target_count"]) < 1:
        return False
    if not bool(metrics["changed_vs_original"]):
        return False
    if not bool(metrics["changed_vs_no_trail"]):
        return False
    if len(anchors_by_color) >= 2:
        axes = {
            "h" if int(r) in (0, int(grid.shape[0] - 1)) and int(c) not in (0, int(grid.shape[1] - 1)) else "v"
            for r, c in anchors_by_color.values()
        }
        if len(axes) < 2:
            return False
    _ = main(grid)
    return True


def _gen_attempt():
    canvas, bg = _new_canvas()
    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    axis_plan = _choose_axis_plan(total_sprites)
    if len(axis_plan) == 2:
        primary_max = min(_PRIMARY_LANE_SIZE_RANGE[1], total_sprites - 1)
    else:
        primary_max = min(_PRIMARY_LANE_SIZE_RANGE[1], total_sprites)
    primary_count = random.randint(_PRIMARY_LANE_SIZE_RANGE[0], primary_max)
    remaining = total_sprites - primary_count
    anchors = _place_anchors(canvas, bg, axis_plan)
    if anchors is None:
        return None

    used_body = set()
    primary_anchor = random.choice(anchors)
    if _place_lane(canvas, primary_anchor, primary_count, bg, used_body) is None:
        return None

    extra_anchors = [anchor for anchor in anchors if anchor is not primary_anchor]
    for anchor in extra_anchors:
        if remaining <= 0:
            return None
        if _place_lane(canvas, anchor, 1, bg, used_body) is None:
            return None
        remaining -= 1

    for _ in range(remaining):
        anchor = random.choice(anchors)
        if _place_lane(canvas, anchor, 1, bg, used_body) is None:
            return None

    return canvas if _generated_grid_is_valid(canvas, bg) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
