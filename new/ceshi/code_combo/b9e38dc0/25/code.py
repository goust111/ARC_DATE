# Combo variant of b9e38dc0 (index 25)
# combo_spec: D2/0+D6/1
# Name: 对角线开口+对角线孤立像素移动
#
# Design analysis for the fused variant
# =====================================
# a) Concrete behavior trace:
#    - Base family structure: a wall sprite encloses a seed, stage 1 flood-fills the interior and then
#      radiates outward through the opening; stage 2 moves isolated pixels and clears their trails.
#    - Fused geometry here: the wall is a thin rectangle missing one corner, so its two degree-1 endpoints
#      lie on adjacent sides. Their outward directions sum to a diagonal radiate vector, e.g. left + up -> NW.
#    - Small example: if the missing corner is top-left, the top endpoint points left and the left endpoint
#      points up. Stage 1 therefore floods the rectangle interior and also expands into the NW wedge bounded
#      by those two rays. A mover to the SE of that sprite then walks NW one diagonal step at a time until it
#      either reaches the border or its next step would enter the filled sprite, in which case it stops on the
#      last free cell. Compared with the original cardinal version, both the opening geometry and the mover
#      displacement now visibly change both row and column together.
# b) Complexity calibration:
#    - The wall is restricted to a 1-pixel rectangular frame with exactly one missing corner. This keeps the
#      diagonal opening obvious across 2-4 examples instead of hiding it inside irregular noise.
#    - Movers remain singleton pixels and all use the single task-wide diagonal radiate direction. That keeps
#      the rule learnable while still making the fused change unmistakable.
# c) Implicit assumptions review:
#    - The old generator assumed the opening was encoded by two endpoints on the same row/col. That fails here;
#      endpoints must instead lie on adjacent sides so that `rad` becomes diagonal.
#    - The original 4-neighbor flood assumption also fails for a pure corner opening: the interior reaches the
#      outside only through a diagonal mouth. Stage 1 therefore has to use corner-connected reachability so the
#      radiate wedge is physically realizable on the pixel grid.
#    - Any solver branch that expects stage-2 pixels to be static obstacles is invalid for this family. The new
#      solver treats all remaining non-bg/non-wall/non-fill pixels as movers after stage 1.
#    - Fill-color selection still relies on "closest non-wall color to the wall". The generator therefore keeps
#      movers at taxicab distance >= 3 from the wall so the interior seed remains uniquely closest.
# d) Placement legality:
#    - The rectangle frame is placed with open-corner margin toward its radiate border and closed-corner margin
#      on the opposite side, so both the diagonal wedge and the interior seed fit without overlap or clipping.
#    - Movers are placed only on background cells outside the final stage-1 sprite. Candidate footprints are
#      checked so starts, traversed cells, and endpoints do not overlap each other.
# e) Visual observability:
#    - Every accepted sample has a diagonal `rad` and at least one mover whose row and column both change.
#      This prevents silent regression back to a cardinal-looking instance.
# f) Visual cleanliness:
#    - The output contains only three layers: static wall, static fill sprite, and a few moved singleton pixels.
#      No extra path coloring is added, so the diagonal wedge remains easy to read.
# g) Color-rule clarity:
#    - Colors retain simple roles: background, wall, fill, mover(s). No color-id ordering or hidden palette
#      semantics are used.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes the active wall component has exactly two degree-1 endpoints whose outward directions are
#      axis-aligned and sum to one diagonal vector. The generator constructs only one-corner-missing frames, so
#      this is guaranteed.
#    - `main()` also assumes mover paths are independent. The generator enforces disjoint mover footprints.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_cdt

ROOT = Path("/home/lixinyu/snap/new")
for _p in (ROOT / "solvers", ROOT / "arc2_solver_select"):
    s = str(_p)
    if s not in sys.path:
        sys.path.append(s)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


GRID_H = (12, 18)
GRID_W = (12, 18)
BOX_H = (7, 9)
BOX_W = (7, 9)
OPEN_SEG = (2, 4)
OPEN_END_MARGIN = 3
CLOSED_END_MARGIN = 2

WALL_PIXELS_MIN = 17
WALL_PIXELS_MAX = 25

SEED_SIZE_WEIGHTS = [1, 1, 1, 2, 2, 3, 4, 5, 6]
MAX_MOVERS = 2
MOVER_MIN_TRAVEL = 3
MOVER_MIN_WALL_DIST = 3
PREFER_BLOCKED_MOVER = 0.75

MIN_NEW_FILL = 18
PALETTE_SIZE = (4, 6)
GEN_RETRIES = 500


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _degree1_pixels(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r, c in points
        if sum((rr, cc) in points for rr, cc in neighbors8(r, c)) == 1
    ]


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _grow_seed_blob(seed: tuple[int, int], interior_set: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_WEIGHTS)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _build_corner_wall(
    rad: tuple[int, int],
    box_h: int,
    box_w: int,
    gap_a: int,
    gap_b: int,
) -> set[tuple[int, int]] | None:
    pts = set()
    for c in range(box_w):
        pts.add((0, c))
        pts.add((box_h - 1, c))
    for r in range(box_h):
        pts.add((r, 0))
        pts.add((r, box_w - 1))

    if rad == (-1, -1):
        for c in range(gap_a):
            pts.discard((0, c))
        for r in range(gap_b):
            pts.discard((r, 0))
    elif rad == (-1, 1):
        for c in range(box_w - gap_a, box_w):
            pts.discard((0, c))
        for r in range(gap_b):
            pts.discard((r, box_w - 1))
    elif rad == (1, -1):
        for c in range(gap_a):
            pts.discard((box_h - 1, c))
        for r in range(box_h - gap_b, box_h):
            pts.discard((r, 0))
    else:
        for c in range(box_w - gap_a, box_w):
            pts.discard((box_h - 1, c))
        for r in range(box_h - gap_b, box_h):
            pts.discard((r, box_w - 1))

    wall_pixels = len(pts)
    if not (WALL_PIXELS_MIN <= wall_pixels <= WALL_PIXELS_MAX):
        return None
    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts


def _place_anchor(rad: tuple[int, int], h: int, w: int, box_h: int, box_w: int) -> tuple[int, int]:
    if rad == (-1, -1):
        r0 = random.randint(OPEN_END_MARGIN, h - box_h - CLOSED_END_MARGIN)
        c0 = random.randint(OPEN_END_MARGIN, w - box_w - CLOSED_END_MARGIN)
    elif rad == (-1, 1):
        r0 = random.randint(OPEN_END_MARGIN, h - box_h - CLOSED_END_MARGIN)
        c0 = random.randint(CLOSED_END_MARGIN, w - box_w - OPEN_END_MARGIN)
    elif rad == (1, -1):
        r0 = random.randint(CLOSED_END_MARGIN, h - box_h - OPEN_END_MARGIN)
        c0 = random.randint(OPEN_END_MARGIN, w - box_w - CLOSED_END_MARGIN)
    else:
        r0 = random.randint(CLOSED_END_MARGIN, h - box_h - OPEN_END_MARGIN)
        c0 = random.randint(CLOSED_END_MARGIN, w - box_w - OPEN_END_MARGIN)
    return r0, c0


def _seed_pool(rad: tuple[int, int], box_h: int, box_w: int) -> list[tuple[int, int]]:
    interior = {(r, c) for r in range(1, box_h - 1) for c in range(1, box_w - 1)}
    if rad == (-1, -1):
        candidates = [(r, c) for r, c in interior if r >= box_h - 3 and c >= box_w - 3]
    elif rad == (-1, 1):
        candidates = [(r, c) for r, c in interior if r >= box_h - 3 and c <= 2]
    elif rad == (1, -1):
        candidates = [(r, c) for r, c in interior if r <= 2 and c >= box_w - 3]
    else:
        candidates = [(r, c) for r, c in interior if r <= 2 and c <= 2]
    return candidates


def _compute_stage1(grid: np.ndarray) -> dict:
    g = np.array(grid, copy=True)
    h, w = g.shape
    bg = int(get_bg(g))

    vals = np.unique(g)
    non_bg = [int(v) for v in vals if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    wall = max(non_bg, key=lambda c: int((g == c).sum()))
    wall_mask = g == wall
    comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(wall_mask, connectivity=8)
    ]
    assert comps, "expected at least one wall component"

    other_colors = [c for c in non_bg if c != wall]
    assert other_colors, "expected at least one non-wall color"

    wall_points = [p for comp in comps for p in comp]

    def min_dist_to_wall(color: int) -> tuple[int, int]:
        pts = [(int(r), int(c)) for r, c in zip(*np.where(g == color))]
        best_d = 10**9
        for r, c in pts:
            for wr, wc in wall_points:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0, len(pts)
        return best_d, len(pts)

    fill = min(other_colors, key=min_dist_to_wall)
    fill_pos = [(int(r), int(c)) for r, c in zip(*np.where(g == fill))]
    assert fill_pos, "expected at least one fill seed"

    def comp_min_dist(comp: set[tuple[int, int]]) -> int:
        best_d = 10**9
        for r, c in fill_pos:
            for wr, wc in comp:
                d = abs(r - wr) + abs(c - wc)
                if d < best_d:
                    best_d = d
                    if best_d == 0:
                        return 0
        return best_d

    wall_comp = min(comps, key=lambda comp: (comp_min_dist(comp), -len(comp)))
    comp_mask = np.zeros_like(wall_mask, dtype=bool)
    for r, c in wall_comp:
        comp_mask[r, c] = True

    def comp_neighbors(r: int, c: int) -> list[tuple[int, int]]:
        return [
            (rr, cc)
            for rr, cc in neighbors8(r, c)
            if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]
        ]

    endpoints = [(r, c) for r, c in wall_comp if len(comp_neighbors(r, c)) == 1]
    assert len(endpoints) == 2, "expected exactly two wall endpoints"

    out_dirs = []
    for ep in endpoints:
        n1 = comp_neighbors(*ep)[0]
        inward = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        out_dirs.append((-inward[0], -inward[1]))

    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad[0] != 0 and rad[1] != 0, "expected diagonal radiate direction"

    rows = [r for r, _ in wall_comp]
    cols = [c for _, c in wall_comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    interior_passable = np.zeros_like(g, dtype=bool)
    interior_passable[r0 + 1 : r1, c0 + 1 : c1] = True
    interior_passable &= (g == bg) | (g == fill)

    q = [p for p in fill_pos if interior_passable[p[0], p[1]]]
    seen = np.zeros_like(interior_passable, dtype=bool)
    for r, c in q:
        seen[r, c] = True

    while q:
        r, c = q.pop()
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and interior_passable[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    out = g.copy()
    filled_mask = seen & (g == bg)
    out[filled_mask] = fill

    mouth_cells: set[tuple[int, int]] = set()
    if rad == (-1, -1):
        top_ep = min((ep for ep in endpoints if ep[0] == r0), key=lambda p: p[1])
        left_ep = min((ep for ep in endpoints if ep[1] == c0), key=lambda p: p[0])
        mouth_cells |= {(r0, c) for c in range(c0, top_ep[1])}
        mouth_cells |= {(r, c0) for r in range(r0, left_ep[0])}
    elif rad == (-1, 1):
        top_ep = max((ep for ep in endpoints if ep[0] == r0), key=lambda p: p[1])
        right_ep = min((ep for ep in endpoints if ep[1] == c1), key=lambda p: p[0])
        mouth_cells |= {(r0, c) for c in range(top_ep[1] + 1, c1 + 1)}
        mouth_cells |= {(r, c1) for r in range(r0, right_ep[0])}
    elif rad == (1, -1):
        bottom_ep = min((ep for ep in endpoints if ep[0] == r1), key=lambda p: p[1])
        left_ep = max((ep for ep in endpoints if ep[1] == c0), key=lambda p: p[0])
        mouth_cells |= {(r1, c) for c in range(c0, bottom_ep[1])}
        mouth_cells |= {(r, c0) for r in range(left_ep[0] + 1, r1 + 1)}
    else:
        bottom_ep = max((ep for ep in endpoints if ep[0] == r1), key=lambda p: p[1])
        right_ep = max((ep for ep in endpoints if ep[1] == c1), key=lambda p: p[0])
        mouth_cells |= {(r1, c) for c in range(bottom_ep[1] + 1, c1 + 1)}
        mouth_cells |= {(r, c1) for r in range(right_ep[0] + 1, r1 + 1)}

    for sr, sc in mouth_cells:
        rr, cc = sr, sc
        while 0 <= rr < h and 0 <= cc < w:
            if out[rr, cc] == bg:
                out[rr, cc] = fill
                filled_mask[rr, cc] = True
            elif out[rr, cc] != fill:
                break
            rr += rad[0]
            cc += rad[1]

    sprite_mask = comp_mask | (out == fill)

    return {
        "out": out,
        "filled_mask": filled_mask,
        "sprite_mask": sprite_mask,
        "bg": bg,
        "wall": wall,
        "fill": fill,
        "rad": rad,
    }


def _trace_mover(start: tuple[int, int], d: tuple[int, int], sprite_mask: np.ndarray) -> dict:
    h, w = sprite_mask.shape
    cur = (int(start[0]), int(start[1]))
    dr, dc = int(d[0]), int(d[1])
    path: list[tuple[int, int]] = []

    while True:
        nr, nc = cur[0] + dr, cur[1] + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return {"end": cur, "path": path, "blocked": False}
        if sprite_mask[nr, nc]:
            return {"end": cur, "path": path, "blocked": True}
        cur = (int(nr), int(nc))
        path.append(cur)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill a corner-open enclosure and move isolated pixels along the same diagonal opening direction.

    Input structure:
    - The background is the most common color. The most frequent non-background color is a wall that forms a thin enclosure with one missing corner, so its opening points diagonally.
    - The non-wall color nearest that enclosure is the fill seed inside it. Any remaining non-background pixels are isolated movers placed outside the finished filled sprite.

    Rule:
    - First fill all background cells reachable from the seed while the wall and the two endpoint rays define a diagonally opening wedge, using corner-connected reachability so the fill can exit through that corner mouth.
    - Then move each isolated pixel one diagonal step at a time in that same radiate direction, erase its starting cell to background, and stop when the next step would leave the grid or enter the wall+fill sprite.

    Key insight: The wall endpoints no longer indicate a single up/down/left/right exit; instead they sit on adjacent sides of a missing corner, and their outward vectors combine into the diagonal that governs both the fill wedge and the mover displacement.
    """
    info = _compute_stage1(np.array(grid, copy=True))
    out = np.array(info["out"], copy=True)
    bg = int(info["bg"])
    wall = int(info["wall"])
    fill = int(info["fill"])
    rad = tuple(info["rad"])
    sprite_mask = np.array(info["sprite_mask"], copy=True)

    mover_cells = [
        (int(r), int(c), int(out[r, c]))
        for r, c in np.argwhere((np.array(grid) != bg) & (np.array(grid) != wall) & (np.array(grid) != fill))
    ]
    for r, c, _col in mover_cells:
        out[r, c] = bg

    for r, c, col in sorted(mover_cells):
        traced = _trace_mover((r, c), rad, sprite_mask)
        end_r, end_c = traced["end"]
        out[end_r, end_c] = col

    return out


def _find_mover_candidates(grid: np.ndarray, stage1_info: dict) -> tuple[list[dict], list[dict]]:
    bg = int(stage1_info["bg"])
    wall = int(stage1_info["wall"])
    rad = tuple(stage1_info["rad"])
    sprite_mask = np.array(stage1_info["sprite_mask"], copy=True)

    wall_mask = grid == wall
    dist = distance_transform_cdt(~wall_mask, metric="taxicab")
    eligible = (grid == bg) & (~sprite_mask) & (dist >= MOVER_MIN_WALL_DIST)

    blocked: list[dict] = []
    free: list[dict] = []
    for r, c in zip(*np.where(eligible)):
        traced = _trace_mover((int(r), int(c)), rad, sprite_mask)
        if len(traced["path"]) < MOVER_MIN_TRAVEL:
            continue
        end = traced["end"]
        if traced["blocked"]:
            if end[0] in (0, grid.shape[0] - 1) or end[1] in (0, grid.shape[1] - 1):
                continue
        else:
            if end[0] not in (0, grid.shape[0] - 1) and end[1] not in (0, grid.shape[1] - 1):
                continue
        dr = end[0] - int(r)
        dc = end[1] - int(c)
        if dr == 0 or dc == 0:
            continue
        footprint = {(int(r), int(c)), *traced["path"]}
        item = {
            "start": (int(r), int(c)),
            "end": (int(end[0]), int(end[1])),
            "path": list(traced["path"]),
            "footprint": footprint,
            "blocked": bool(traced["blocked"]),
        }
        if traced["blocked"]:
            blocked.append(item)
        else:
            free.append(item)

    random.shuffle(blocked)
    random.shuffle(free)
    blocked.sort(key=lambda x: len(x["path"]), reverse=True)
    free.sort(key=lambda x: len(x["path"]), reverse=True)
    return blocked, free


def _choose_movers(blocked: list[dict], free: list[dict]) -> list[dict] | None:
    if not free:
        return None

    chosen: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    def try_add(item: dict) -> bool:
        nonlocal occupied
        if item["footprint"] & occupied:
            return False
        chosen.append(item)
        occupied |= item["footprint"]
        return True

    try_add(free[0])
    if blocked and random.random() < PREFER_BLOCKED_MOVER:
        for item in blocked:
            if try_add(item):
                break
    if len(chosen) < MAX_MOVERS:
        for item in free[1:]:
            if try_add(item):
                break
    return chosen if chosen else None


def _gen_attempt() -> np.ndarray | None:
    rad = random.choice([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    box_h = random.randint(*BOX_H)
    box_w = random.randint(*BOX_W)
    gap_a = random.randint(*OPEN_SEG)
    gap_b = random.randint(*OPEN_SEG)

    local_wall = _build_corner_wall(rad, box_h, box_w, gap_a, gap_b)
    if local_wall is None:
        return None

    try:
        anchor_r, anchor_c = _place_anchor(rad, h, w, box_h, box_w)
    except ValueError:
        return None

    palette = random.sample(range(10), random.randint(*PALETTE_SIZE))
    bg, wall_color, fill_color = palette[:3]
    mover_palette = [c for c in palette[3:] if c not in (bg, wall_color, fill_color)]
    available = [c for c in range(10) if c not in (bg, wall_color, fill_color)]
    while len(mover_palette) < MAX_MOVERS:
        c = random.choice(available)
        if c not in mover_palette:
            mover_palette.append(c)

    grid = np.full((h, w), bg, dtype=int)
    wall_global = set()
    for lr, lc in local_wall:
        gr = anchor_r + lr
        gc = anchor_c + lc
        if not (0 <= gr < h and 0 <= gc < w):
            return None
        wall_global.add((gr, gc))
        grid[gr, gc] = wall_color

    seed_candidates = _seed_pool(rad, box_h, box_w)
    interior_set = {(r, c) for r in range(1, box_h - 1) for c in range(1, box_w - 1)}
    adjacent = []
    for p in seed_candidates:
        if any((p[0] + dr, p[1] + dc) in local_wall for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
            adjacent.append(p)
    if not adjacent:
        return None

    blob = _grow_seed_blob(random.choice(adjacent), interior_set)
    for lr, lc in blob:
        gr = anchor_r + lr
        gc = anchor_c + lc
        if grid[gr, gc] != bg:
            return None
        grid[gr, gc] = fill_color

    try:
        stage1_info = _compute_stage1(grid)
    except Exception:
        return None

    if tuple(stage1_info["rad"]) != rad:
        return None

    nominal_out = np.array(stage1_info["out"], copy=True)
    blocked, free = _find_mover_candidates(grid, stage1_info)
    chosen = _choose_movers(blocked, free)
    if not chosen:
        return None

    for i, item in enumerate(chosen):
        r, c = item["start"]
        grid[r, c] = mover_palette[i]

    try:
        out = main(grid)
    except Exception:
        return None

    new_fill = (out == fill_color) & (grid == bg)
    if int(new_fill.sum()) < MIN_NEW_FILL or not _touches_border(new_fill):
        return None

    if not np.array_equal((nominal_out == fill_color), (out == fill_color)):
        return None

    moved_diagonal = 0
    for i, item in enumerate(chosen):
        sr, sc = item["start"]
        er, ec = item["end"]
        if out[sr, sc] != bg:
            return None
        if out[er, ec] != mover_palette[i]:
            return None
        if er != sr and ec != sc:
            moved_diagonal += 1

    if moved_diagonal == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid b9e38dc0 combo D2/0+D6/1 input")
