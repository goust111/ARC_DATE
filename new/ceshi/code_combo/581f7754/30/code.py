# Combo variant of 581f7754 (index 30)
# combo_spec: D8/1+D16/0
# Name: 水平先垂直后+颜色混合

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_THIS_FILE = Path(__file__).resolve()
for _parent in _THIS_FILE.parents:
    _solver_dir = _parent / "solvers"
    if _solver_dir.is_dir():
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break

from arc_utils import find_components, get_bg, neighbors4, paste


# Design analysis for combo D8/1 + D16/0
# ---------------------------------------
# a) Concrete behavior and traced example:
#    - Base task logic is preserved: each interior sprite has a rare marker color that matches one isolated
#      border anchor, and the anchor edge still chooses the slide axis.
#    - The fused change is operational rather than purely set-theoretic:
#        round 1: move all horizontal sprites;
#        round 2: move all vertical sprites on top of round 1's bitmap.
#      Within a round, moved source cells are cleared together, then translated pixels are pasted.
#    - Pasting now uses per-cell color max instead of "only if destination is bg".
#    - Small trace:
#        H wants to slide right; V wants to slide downward.
#        Some H landing cells overlap V's current source cells. In round 1 those cells become max(H,V).
#        In round 2 V's original coordinates are cleared from that intermediate bitmap, so any round-1 paint
#        sitting on V's old footprint can disappear. Later, when V lands on H's settled cells, the visible
#        color at those overlap cells becomes max(existing_H, incoming_V).
#      This is the intended fusion: one interaction exposes the D8 ordering dependency, while another exposes
#      the D16 max-mixing rule.
# b) Complexity calibration:
#    - Every generated grid contains at least one horizontal mover and one vertical mover.
#    - A forced H/V pair is built so that:
#        1. combo output differs from sequential-skip D8-only output;
#        2. combo output differs from simultaneous-max D16-only output.
#      This avoids degenerating into either single variant.
#    - Extra sprites are added without further overlaps so the board stays sparse and the forced interaction
#      remains the dominant clue.
# c) Implicit assumption audit for main():
#    - Anchor detection still assumes border singletons with 4-neighbor bg; generator preserves that.
#    - Marker detection still assumes each movable component is either multi-color with a least-frequent color
#      or an anchor-colored singleton; generator keeps input sprites isolated so component parsing is stable.
#    - The original all-at-once clear pass is no longer valid, because round-2 clearing must read round-1 state.
#    - The original collision rule "skip if non-bg" is invalid under D16; it is replaced by cellwise max.
#    - The code now assumes each sprite moves exactly once from its original input coordinates; generator
#      guarantees that no sprite is already on the border and no object splits between rounds.
# d) Placement legality:
#    - Input placement still enforces interior placement, no overlaps, and an 8-neighbor bg moat between
#      distinct input sprites and anchors.
#    - Solved placements for ordinary extra sprites are kept disjoint. Only the forced pair is allowed to
#      interact, and even there the overlap sizes are bounded and both sprites keep visible non-overlap parts.
# e) Visual observability:
#    - The generator forces two observable signatures:
#        1. cells where the combo keeps a max-color overlap that D8-only would have left unchanged;
#        2. cells where combo differs from simultaneous max because round-2 clearing erases paint that had
#           temporarily landed on a vertical sprite's old footprint.
#    - To make the max rule legible, the vertical member of the forced pair is biased toward a high body color,
#      typically 9, so the winning overlap color is visually stable instead of arbitrary-looking.
# f) Visual cleanliness:
#    - Only one interaction pair is forced. Other sprites are placed away from that pair in both input and
#      solved space, which prevents the final output from turning into a dense overlap cloud.
# g) Change inferability:
#    - The color change is not free-floating recoloring; it appears only at geometric intersections created by
#      the motion rule, and the winning color is already present on one of the interacting sprites.
#    - The generator keeps the overlap counts small and repeats the same interaction pattern every sample, so
#      the human clue is "overlap inherits the stronger participant's color under a fixed two-round order".
# post-change solver assumptions checked against generator:
#    - main() assumes at least one anchor exists and at least one matching movable component uses each forced
#      axis present in the sample; generator constructs both axes explicitly.
#    - main() clears round-2 source cells from the intermediate bitmap; generator therefore forces a pair where
#      this matters, and validates against a simultaneous-max baseline to ensure the assumption is exercised.


def _component_marker_color(grid: np.ndarray, coords: np.ndarray) -> int | None:
    vals = [int(grid[int(r), int(c)]) for r, c in coords]
    cnt = Counter(vals)
    if len(cnt) <= 1:
        return None
    return int(min(cnt.items(), key=lambda kv: (int(kv[1]), int(kv[0])))[0])


def _marker_pixel(grid: np.ndarray, coords: np.ndarray, marker: int) -> tuple[int, int] | None:
    pts = [(int(r), int(c)) for r, c in coords if int(grid[int(r), int(c)]) == int(marker)]
    if not pts:
        return None
    if len(pts) == 1:
        return pts[0]
    cell_set = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if (int(rr), int(cc)) in cell_set and int(grid[int(rr), int(cc)]) != int(marker):
                return (int(r), int(c))
    return pts[0]


def _detect_moves(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_border(r: int, c: int) -> bool:
        return int(r) in (0, int(H - 1)) or int(c) in (0, int(W - 1))

    anchors_by_color: dict[int, tuple[int, int]] = {}
    for r in range(H):
        for c in range(W):
            if not is_border(r, c):
                continue
            v = int(g[r, c])
            if v == int(bg):
                continue
            ok = True
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) != int(bg):
                    ok = False
                    break
            if ok and int(v) not in anchors_by_color:
                anchors_by_color[int(v)] = (int(r), int(c))

    assert anchors_by_color

    anchor_cells = {(int(r), int(c)) for r, c in anchors_by_color.values()}
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    moves: list[dict] = []
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
            if v == int(bg) or int(v) not in anchors_by_color or (int(r), int(c)) in anchor_cells:
                continue
            marker_col = int(v)

        if int(marker_col) not in anchors_by_color:
            continue

        mp = _marker_pixel(g, coords, int(marker_col))
        if mp is None:
            continue

        ar, ac = anchors_by_color[int(marker_col)]
        horizontal = int(ar) in (0, int(H - 1)) and int(ac) not in (0, int(W - 1))
        dr = 0
        dc = 0
        if horizontal:
            dc = int(ac - mp[1])
        else:
            dr = int(ar - mp[0])

        moves.append(
            {
                "coords": coords,
                "dr": int(dr),
                "dc": int(dc),
                "axis": "h" if horizontal else "v",
                "marker_color": int(marker_col),
            }
        )

    return g, bg, moves


def _apply_move_batch(state: np.ndarray, source: np.ndarray, moves: list[dict], bg: int, *, mode: str) -> None:
    H, W = state.shape
    for move in moves:
        coords = move["coords"]
        state[coords[:, 0], coords[:, 1]] = int(bg)

    for move in moves:
        dr = int(move["dr"])
        dc = int(move["dc"])
        for r, c in move["coords"]:
            rr = int(r) + dr
            cc = int(c) + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            color = int(source[int(r), int(c)])
            if int(state[rr, cc]) == int(bg):
                state[rr, cc] = int(color)
                continue
            if mode == "skip":
                continue
            if mode == "max":
                state[rr, cc] = max(int(state[rr, cc]), int(color))
                continue
            raise ValueError(f"unsupported mode: {mode}")


def _sequential_skip_main(grid: np.ndarray) -> np.ndarray:
    g, bg, moves = _detect_moves(grid)
    out = g.copy()
    horizontal = [move for move in moves if move["axis"] == "h"]
    vertical = [move for move in moves if move["axis"] == "v"]
    _apply_move_batch(out, g, horizontal, bg, mode="skip")
    _apply_move_batch(out, g, vertical, bg, mode="skip")
    return out


def _simultaneous_max_main(grid: np.ndarray) -> np.ndarray:
    g, bg, moves = _detect_moves(grid)
    out = g.copy()
    _apply_move_batch(out, g, moves, bg, mode="max")
    return out


def _combo_diffs(grid: np.ndarray) -> tuple[int, int]:
    combo = main(grid)
    seq_skip = _sequential_skip_main(grid)
    sim_max = _simultaneous_max_main(grid)
    return int(np.count_nonzero(combo != seq_skip)), int(np.count_nonzero(combo != sim_max))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide interior sprites toward their matching border anchors in two ordered rounds with max-color overlaps.

    Input structure:
    - The background is the most frequent color.
    - Interior connected sprites are mostly two-color objects with a minority-color marker pixel; optional
      anchor-colored singletons are still treated as tiny movable sprites.
    - Isolated non-background border pixels act as anchors, and their edge placement still determines slide axis.

    Rule:
    - Match each movable sprite to the anchor sharing its marker color.
    - Top/bottom anchors trigger horizontal sliding, while left/right anchors trigger vertical sliding.
    - Execute all horizontal moves first and all vertical moves second. In each round, clear that round's source
      sprites from the current canvas, then paste their translated pixels; if a translated pixel hits non-background,
      keep the larger of the two colors.

    Key insight: The final image depends on both axis choice and execution order. Max-color overlap is applied on the
    live intermediate canvas, so vertical sprites can both overwrite horizontal landings by color and erase earlier
    paint again when their own old positions are cleared in round 2.
    """
    g, bg, moves = _detect_moves(grid)
    out = g.copy()
    horizontal = [move for move in moves if move["axis"] == "h"]
    vertical = [move for move in moves if move["axis"] == "v"]
    _apply_move_batch(out, g, horizontal, bg, mode="max")
    _apply_move_batch(out, g, vertical, bg, mode="max")
    return out


# --------------- generator ---------------

_COLORS = tuple(range(10))
_ANCHOR_COLOR_POOL = tuple(range(1, 9))
_LOW_BODY_PREFERENCE = (1, 2, 3, 4, 5, 6)
_HIGH_BODY_PREFERENCE = (9, 8, 7)

_BG_COLOR_RANGE = (1, 8)
_GRID_HEIGHT_RANGE = (10, 22)
_GRID_WIDTH_RANGE = (8, 22)

_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_NEIGHBORS8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
_INNER_BORDER_GAP = 1
_ANCHOR_EDGE_MARGIN = 2
_ISOLATION_RADIUS = 1

_ANCHOR_COUNT_RANGE = (2, 3)
_SPRITE_COUNT_RANGE = (3, 5)
_BODY_AREA_RANGE = (4, 13)
_GEN_RETRY_LIMIT = 500

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
_MARKER_INTERIOR_PROB = 0.65
_FLIP_PROB = 0.5

_PAIR_BUILD_ATTEMPTS = 120
_SOURCE_OVERLAP_MIN = 2
_SOURCE_OVERLAP_MAX = 5
_SOURCE_ONLY_MIN = 2
_DEST_OVERLAP_MIN = 2
_DEST_OVERLAP_MAX = 4
_VISIBLE_REMAINDER_MIN = 2
_DEST_WIN_MIN = 2
_MIN_DIFF_VS_SKIP = 2
_MIN_DIFF_VS_SIM_MAX = 2

_GROW_BODY_SEEDS = [
    {(0, 0), (0, 1), (0, 2), (1, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 2), (1, 0), (1, 1), (1, 2)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
]


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


def _marker_cell(body):
    r0, c0, r1, c1 = _bbox(body)
    interior, exterior = set(), set()
    for r, c in body:
        for dr, dc in _NEIGHBORS4:
            p = (r + dr, c + dc)
            if p in body:
                continue
            if r0 <= p[0] <= r1 and c0 <= p[1] <= c1:
                interior.add(p)
            else:
                exterior.add(p)
    pool = list(interior if interior and random.random() < _MARKER_INTERIOR_PROB else (exterior | interior))
    return random.choice(pool)


def _make_sprite(area, body_color, marker_color, bg):
    body = _make_body(area)
    marker = _marker_cell(body)
    all_cells = set(body) | {marker}
    r0, c0, r1, c1 = _bbox(all_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in body:
        patch[r - r0, c - c0] = int(body_color)
    patch[marker[0] - r0, marker[1] - c0] = int(marker_color)
    return {"patch": patch, "offset": (marker[0] - r0, marker[1] - c0)}


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


def _isolated_bg_cell(canvas, r, c, bg):
    H, W = canvas.shape
    radius = _ISOLATION_RADIUS
    r0, r1 = max(0, r - radius), min(H, r + radius + 1)
    c0, c1 = max(0, c - radius), min(W, c + radius + 1)
    return not bool((canvas[r0:r1, c0:c1] != int(bg)).any())


def _anchor_ok(canvas, r, c, bg):
    H, W = canvas.shape
    if not (r in (0, H - 1) or c in (0, W - 1)):
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


def _put_anchor(canvas, r, c, color):
    paste(canvas, np.array([[int(color)]], dtype=int), int(r), int(c))


def _aligned_positions_free(shape, sprite, anchor):
    H, W = shape
    h, w = sprite["patch"].shape
    off_r, off_c = sprite["offset"]
    positions = []
    if anchor["axis"] == "h":
        col = int(anchor["c"])
        left = col - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            return []
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP):
            top = int(r) - int(off_r)
            if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
                continue
            positions.append((int(r), int(col)))
        return positions

    row = int(anchor["r"])
    top = row - int(off_r)
    if top < _INNER_BORDER_GAP or top + h > H - _INNER_BORDER_GAP:
        return []
    for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP):
        left = int(c) - int(off_c)
        if left < _INNER_BORDER_GAP or left + w > W - _INNER_BORDER_GAP:
            continue
        positions.append((int(row), int(c)))
    return positions


def _scatter_positions(canvas, sprite, anchor, marker, bg):
    H, W = canvas.shape
    mr, mc = marker
    if anchor["axis"] == "h":
        return [
            (mr, c)
            for c in range(_INNER_BORDER_GAP, W - _INNER_BORDER_GAP)
            if c != mc and _can_place(canvas, sprite, mr, c, bg)
        ]
    return [
        (r, mc)
        for r in range(_INNER_BORDER_GAP, H - _INNER_BORDER_GAP)
        if r != mr and _can_place(canvas, sprite, r, mc, bg)
    ]


def _sprite_cell_map(sprite, marker_r, marker_c, bg):
    top, left = _top_left(sprite, marker_r, marker_c)
    patch = sprite["patch"]
    h, w = patch.shape
    cells = {}
    for r in range(h):
        for c in range(w):
            v = int(patch[r, c])
            if v != int(bg):
                cells[(top + r, left + c)] = int(v)
    return cells


def _expand_cells(cells, shape):
    H, W = shape
    out = set()
    for r, c in cells:
        for dr, dc in _NEIGHBORS8:
            rr = int(r) + int(dr)
            cc = int(c) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out.add((int(rr), int(cc)))
    return out


def _anchor_keepout(anchors, shape):
    return _expand_cells({(int(anchor["r"]), int(anchor["c"])) for anchor in anchors}, shape)


def _sample_positions(seq, cap):
    seq = list(seq)
    random.shuffle(seq)
    if len(seq) <= cap:
        return seq
    return seq[:cap]


def _far_positions(anchor, solved, positions):
    if anchor["axis"] == "h":
        strong = [pos for pos in positions if abs(int(pos[1]) - int(solved[1])) >= 2]
    else:
        strong = [pos for pos in positions if abs(int(pos[0]) - int(solved[0])) >= 2]
    return strong or positions


def _body_color(bg, marker, used, *, prefer: str = "any"):
    blocked = {int(bg), int(marker)}
    if prefer == "low":
        pref = [c for c in _LOW_BODY_PREFERENCE if c not in blocked and c not in used]
        if pref:
            return random.choice(pref)
    if prefer == "high":
        pref = [c for c in _HIGH_BODY_PREFERENCE if c not in blocked and c not in used]
        if pref:
            return random.choice(pref)
    pool = [c for c in _COLORS if c not in blocked and c not in used]
    if not pool:
        pool = [c for c in _COLORS if c not in blocked]
    return random.choice(pool)


def _bg_is_mode(grid, bg):
    counts = Counter(map(int, grid.ravel()))
    return counts[int(bg)] > max((n for c, n in counts.items() if c != int(bg)), default=0)


def _new_canvas():
    H = random.randint(*_GRID_HEIGHT_RANGE)
    W = random.randint(*_GRID_WIDTH_RANGE)
    bg = random.randint(*_BG_COLOR_RANGE)
    canvas = np.full((H, W), int(bg), dtype=int)
    return bg, canvas


def _place_anchors(inp, bg):
    n_anchors = random.randint(*_ANCHOR_COUNT_RANGE)
    anchor_colors = random.sample([c for c in _ANCHOR_COLOR_POOL if c != int(bg)], n_anchors)
    axes = ["h", "v"]
    if n_anchors == 3:
        axes.append(random.choice(["h", "v"]))
    random.shuffle(axes)

    anchors = []
    for color, needed_axis in zip(anchor_colors, axes):
        candidates = [cand for cand in _anchor_candidates(inp, bg) if cand[0] == needed_axis]
        if not candidates:
            return None
        axis, r, c = random.choice(candidates)
        anchor = {"color": int(color), "r": int(r), "c": int(c), "axis": axis}
        anchors.append(anchor)
        _put_anchor(inp, r, c, color)
    return anchors


def _noncolliding_solved_positions(shape, sprite, anchor, occupied, anchor_keepout, bg):
    positions = []
    for marker_pos in _aligned_positions_free(shape, sprite, anchor):
        cell_map = _sprite_cell_map(sprite, marker_pos[0], marker_pos[1], bg)
        cells = set(cell_map)
        if cells & anchor_keepout:
            continue
        if _expand_cells(cells, shape) & occupied:
            continue
        positions.append((int(marker_pos[0]), int(marker_pos[1])))
    return positions


def _record(sprite, anchor, solved, inp_pos):
    return {
        "sprite": sprite,
        "anchor": anchor,
        "solved": (int(solved[0]), int(solved[1])),
        "input": (int(inp_pos[0]), int(inp_pos[1])),
    }


def _place_interacting_pair(inp, anchors, bg, used_body):
    shape = inp.shape
    anchor_keepout = _anchor_keepout(anchors, shape)
    h_anchors = [a for a in anchors if a["axis"] == "h"]
    v_anchors = [a for a in anchors if a["axis"] == "v"]
    if not h_anchors or not v_anchors:
        return None

    pair_choices = [(h_anchor, v_anchor) for h_anchor in h_anchors for v_anchor in v_anchors]
    random.shuffle(pair_choices)

    for h_anchor, v_anchor in pair_choices:
        for _ in range(_PAIR_BUILD_ATTEMPTS):
            h_body = _body_color(bg, h_anchor["color"], used_body, prefer="low")
            v_body = _body_color(bg, v_anchor["color"], used_body | {h_body}, prefer="high")
            h_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), h_body, h_anchor["color"], bg)
            v_sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), v_body, v_anchor["color"], bg)

            v_solved_positions = _sample_positions(
                _noncolliding_solved_positions(shape, v_sprite, v_anchor, set(), anchor_keepout, bg),
                10,
            )
            if not v_solved_positions:
                continue

            for v_solved in v_solved_positions:
                v_solved_map = _sprite_cell_map(v_sprite, v_solved[0], v_solved[1], bg)
                v_solved_cells = set(v_solved_map)

                h_solved_positions = _sample_positions(_aligned_positions_free(shape, h_sprite, h_anchor), 24)
                for h_solved in h_solved_positions:
                    h_solved_map = _sprite_cell_map(h_sprite, h_solved[0], h_solved[1], bg)
                    h_solved_cells = set(h_solved_map)
                    if h_solved_cells & anchor_keepout:
                        continue

                    dest_overlap = h_solved_cells & v_solved_cells
                    if not (_DEST_OVERLAP_MIN <= len(dest_overlap) <= _DEST_OVERLAP_MAX):
                        continue
                    if tuple(h_solved) in dest_overlap or tuple(v_solved) in dest_overlap:
                        continue
                    if len(h_solved_cells - dest_overlap) < _VISIBLE_REMAINDER_MIN:
                        continue
                    if len(v_solved_cells - dest_overlap) < _VISIBLE_REMAINDER_MIN:
                        continue
                    winning_dest = sum(int(v_solved_map[cell]) > int(h_solved_map[cell]) for cell in dest_overlap)
                    if winning_dest < _DEST_WIN_MIN:
                        continue

                    v_input_positions = _sample_positions(
                        _far_positions(v_anchor, v_solved, _scatter_positions(inp, v_sprite, v_anchor, v_solved, bg)),
                        24,
                    )
                    for v_input in v_input_positions:
                        v_input_map = _sprite_cell_map(v_sprite, v_input[0], v_input[1], bg)
                        v_input_cells = set(v_input_map)
                        source_overlap = h_solved_cells & v_input_cells
                        source_only = source_overlap - v_solved_cells
                        if not (_SOURCE_OVERLAP_MIN <= len(source_overlap) <= _SOURCE_OVERLAP_MAX):
                            continue
                        if len(source_only) < _SOURCE_ONLY_MIN:
                            continue
                        if tuple(h_solved) in source_overlap or tuple(v_input) in source_overlap:
                            continue

                        temp_in = inp.copy()
                        _place(temp_in, v_sprite, v_input[0], v_input[1], bg)
                        h_input_positions = _sample_positions(
                            _far_positions(h_anchor, h_solved, _scatter_positions(temp_in, h_sprite, h_anchor, h_solved, bg)),
                            24,
                        )
                        for h_input in h_input_positions:
                            trial = inp.copy()
                            _place(trial, v_sprite, v_input[0], v_input[1], bg)
                            _place(trial, h_sprite, h_input[0], h_input[1], bg)
                            diff_vs_skip, diff_vs_sim_max = _combo_diffs(trial)
                            if diff_vs_skip < _MIN_DIFF_VS_SKIP or diff_vs_sim_max < _MIN_DIFF_VS_SIM_MAX:
                                continue

                            _place(inp, v_sprite, v_input[0], v_input[1], bg)
                            _place(inp, h_sprite, h_input[0], h_input[1], bg)
                            used_body.update({int(h_body), int(v_body)})
                            return [
                                _record(h_sprite, h_anchor, h_solved, h_input),
                                _record(v_sprite, v_anchor, v_solved, v_input),
                            ]
    return None


def _place_extra_sprites(inp, anchors, bg, used_body, records, total_sprites):
    shape = inp.shape
    anchor_keepout = _anchor_keepout(anchors, shape)
    occupied_solved = set()
    for rec in records:
        occupied_solved |= set(_sprite_cell_map(rec["sprite"], rec["solved"][0], rec["solved"][1], bg))

    by_color = {anchor["color"]: anchor for anchor in anchors}
    for _ in range(max(0, total_sprites - len(records))):
        marker = random.choice(list(by_color))
        anchor = by_color[int(marker)]
        placed = False
        for _attempt in range(90):
            body = _body_color(bg, marker, used_body, prefer="any")
            sprite = _make_sprite(random.randint(*_BODY_AREA_RANGE), body, marker, bg)
            solved_positions = _sample_positions(
                _noncolliding_solved_positions(shape, sprite, anchor, occupied_solved, anchor_keepout, bg),
                16,
            )
            for solved in solved_positions:
                input_positions = _far_positions(anchor, solved, _scatter_positions(inp, sprite, anchor, solved, bg))
                if not input_positions:
                    continue
                inp_pos = random.choice(input_positions)
                _place(inp, sprite, inp_pos[0], inp_pos[1], bg)
                records.append(_record(sprite, anchor, solved, inp_pos))
                occupied_solved |= set(_sprite_cell_map(sprite, solved[0], solved[1], bg))
                used_body.add(int(body))
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _move_axis_counts(inp):
    _g, _bg, moves = _detect_moves(inp)
    horizontal = sum(1 for move in moves if move["axis"] == "h")
    vertical = sum(1 for move in moves if move["axis"] == "v")
    return horizontal, vertical


def _generated_grid_is_valid(inp, bg):
    if not _bg_is_mode(inp, bg):
        return False
    try:
        horizontal, vertical = _move_axis_counts(inp)
        if horizontal == 0 or vertical == 0:
            return False
        diff_vs_skip, diff_vs_sim_max = _combo_diffs(inp)
        if diff_vs_skip < _MIN_DIFF_VS_SKIP or diff_vs_sim_max < _MIN_DIFF_VS_SIM_MAX:
            return False
        main(inp)
    except Exception:
        return False
    return True


def _gen_attempt():
    bg, inp = _new_canvas()
    anchors = _place_anchors(inp, bg)
    if anchors is None:
        return None

    used_body = set()
    records = _place_interacting_pair(inp, anchors, bg, used_body)
    if records is None:
        return None

    total_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if not _place_extra_sprites(inp, anchors, bg, used_body, records, total_sprites):
        return None

    if not _generated_grid_is_valid(inp, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 581f7754 combo input")
