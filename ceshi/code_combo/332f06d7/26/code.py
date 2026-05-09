# Combo variant of 332f06d7 (index 26)
# combo_spec: D15/2+D18/2
# Name: 图例控制转向+内部编码距离

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import STRUCT4, bbox_coords


# Design analysis
# ---------------
# a) Concrete fused behavior
#    - Keep the original visual vocabulary: green background, blue river, black mover, red
#      destination. The mover is no longer monochrome: yellow pixels inside the mover encode
#      the movement cap.
#    - Add a dedicated legend strip on the top edge with grey background. Each turning-wall
#      color appears twice there:
#          row 0 anchor + row 1 shifted left  -> CCW
#          row 0 anchor + row 1 shifted right -> CW
#    - In the playfield, the blue network is not a single forced bend-only corridor. At each
#      turning site the forward direction is blocked by a thin colored wall, while both side
#      branches are open. The legend chooses which side branch the mover actually takes.
#    - Example trace on the fixed coarse template:
#        start -> step 1 straight -> step 2 straight -> colored wall A, legend says CW,
#        so step 3 turns downward -> colored wall B, legend says CCW, so step 4 turns right
#        -> colored wall B again, so step 5 turns upward.
#      If the mover contains 5 yellow pixels, it stops there even though the legend-driven
#      route has one more legal step. If the legend were swapped, the same 5-step budget would
#      send it into a different dead-end branch.
#    - Difference from the original task:
#        original solver can ignore direction state and just run BFS on all feasible placements.
#        This fusion cannot: geometry alone leaves multiple branch choices open, and the mover
#        also has a hard step budget read from its own interior.
#
# b) Complexity calibration
#    - Exactly two wall colors are used per sample, one for CW and one for CCW. Both are
#      repeated across turns, so the mapping has repeated evidence.
#    - The mover size is fixed to 3x3. That gives 1..8 visible symbol pixels, enough room to
#      count, without making the patch visually noisy.
#    - Generated samples use 3, 5, or 7 symbol pixels, so the mover is forced to traverse one,
#      two, or three legend-controlled turns before the distance budget binds.
#
# c) Implicit-assumption audit versus the original main()
#    - "Mover and destination are monochrome solid squares" is broken by the internal symbol.
#      The fused solver must detect square objects by connected-component bbox, then classify
#      mover by "contains symbol color" instead of by color ordering.
#    - "Geometry alone determines the final location" is broken by the legend-controlled turns.
#      The fused solver must simulate directed motion with heading state.
#    - "Output repaint is a solid mover color" is broken by the symbol payload. The exact mover
#      patch must be carried to the new location.
#    - "Traversable colors are just river/mover/dest" is too weak now. The solver treats the
#      mover and destination bboxes as traversable windows while still ignoring wall colors.
#    - New generator guarantees:
#        * the top 2 rows are reserved for the legend only;
#        * the mover start has exactly one outgoing feasible direction;
#        * each turn wall exposes exactly one control color ahead;
#        * the river remains the largest non-background color in the cropped playfield.
#
# d) Placement legality
#    - The generator now builds the route directly in placement space, so one symbol pixel means
#      one actual top-left shift of the mover, matching the solver's notion of distance exactly.
#    - Decoy branches are short one-step spurs off the placement path, and every rendered sample
#      is revalidated by the solver plus two anti-degeneracy checks.
#
# e) Visual observability
#    - The same control colors appear in the legend and at the blocked faces inside the river.
#    - The mover visibly carries 3..5 yellow pixels, and output displacement varies with that
#      count across samples.
#    - Because both side branches are open at turns, a human must explain which branch was taken;
#      the legend is the only stable cue for that.
#
# f) Cleanliness
#    - The legend is isolated from the playfield.
#    - Extra colors are limited to grey legend background, two thin wall colors, and one symbol
#      color inside the mover; nothing else is overlaid on the river.
#
# g) Inferability of the changes
#    - No rule depends on numeric color ids. CW/CCW is explicitly encoded by legend geometry.
#    - The symbol always uses a fixed internal layout, so counting yellow cells is sufficient to
#      recover the step cap from a few examples.


LEGEND_H = 2
LEGEND_BG = 5

_BG, _RIVER, _MOVER, _DEST, _SYMBOL = 3, 1, 0, 2, 8
_TURN_COLORS = (4, 6)
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

_TOTAL_H = 20
_TOTAL_W = 20
_CORE_H = _TOTAL_H - LEGEND_H
_CORE_W = _TOTAL_W
_N = 3
_H2 = _CORE_H - _N + 1
_W2 = _CORE_W - _N + 1

_BASE_MAIN_PATH = [
    (8, 2),
    (8, 3),
    (8, 4),
    (9, 4),
    (10, 4),
    (11, 4),
    (11, 5),
    (11, 6),
    (11, 7),
    (10, 7),
    (9, 7),
]
_BASE_DECOY_EDGES = [
    ((11, 4), (11, 3)),
    ((11, 7), (12, 7)),
]
_BASE_RED_BRANCH = [(7, 4), (6, 4), (5, 4), (4, 4)]


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _pick_farthest_within_steps(dist: np.ndarray, dr: int, dc: int, max_steps: int) -> tuple[int, int]:
    if 0 <= dr < dist.shape[0] and 0 <= dc < dist.shape[1]:
        d = int(dist[dr, dc])
        if 0 <= d <= max_steps:
            return int(dr), int(dc)

    best = -1
    tr, tc = 0, 0
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            d = int(dist[r, c])
            if 0 <= d <= max_steps and d > best:
                best = d
                tr, tc = int(r), int(c)
    assert best >= 0
    return tr, tc


def _parse_legend(grid: np.ndarray) -> dict[int, str]:
    legend = np.array(grid[:LEGEND_H, :], dtype=int)
    turn_by_color: dict[int, str] = {}
    for col in sorted(set(map(int, legend.flatten()))):
        if int(col) == LEGEND_BG:
            continue
        pts = np.argwhere(legend == int(col))
        if len(pts) != 2:
            continue
        top = [p for p in pts if int(p[0]) == 0]
        bot = [p for p in pts if int(p[0]) == 1]
        if len(top) != 1 or len(bot) != 1:
            continue
        turn_by_color[int(col)] = "ccw" if int(bot[0][1]) < int(top[0][1]) else "cw"
    return turn_by_color


def _build_legend(width: int, turn_by_color: dict[int, str], order: list[int]) -> np.ndarray:
    legend = np.full((LEGEND_H, width), LEGEND_BG, dtype=int)
    cursor = 1
    for col in order:
        if cursor + 2 >= width:
            break
        legend[0, cursor + 1] = int(col)
        if turn_by_color[int(col)] == "ccw":
            legend[1, cursor] = int(col)
        else:
            legend[1, cursor + 2] = int(col)
        cursor += 4
    return legend


def _infer_river(core: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, core.flatten()))
    return int(max((col for col in counts if int(col) != int(bg)), key=lambda col: int(counts[int(col)])))


def _detect_square_objects(core: np.ndarray, bg: int, river: int) -> tuple[dict, dict]:
    # Only the mover patch and the destination are object candidates.
    # Control-wall colors can touch the destination through 4-connectivity,
    # so including every non-bg / non-river cell merges them into one component.
    mask = np.isin(core, [int(_MOVER), int(_SYMBOL), int(_DEST)])
    labeled, num = scipy_label(mask.astype(int), structure=STRUCT4)

    squares: list[dict] = []
    for k in range(1, int(num) + 1):
        coords = np.argwhere(labeled == int(k))
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        if int(r1 - r0) != int(c1 - c0):
            continue
        patch = core[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        comp_mask = labeled[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(k)
        if not bool(comp_mask.all()):
            continue
        colors = {int(x) for x in np.unique(patch)}
        squares.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "n": int(r1 - r0 + 1),
                "patch": patch.copy(),
                "colors": colors,
            }
        )

    assert len(squares) >= 2
    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    movers = [sq for sq in cand if int(_SYMBOL) in sq["colors"]]
    dests = [sq for sq in cand if sq["colors"] == {int(_DEST)}]
    assert len(movers) == 1
    assert len(dests) == 1
    return movers[0], dests[0]


def _feasible_mask(core: np.ndarray, river: int, mover: dict, dest: dict) -> np.ndarray:
    n = int(mover["n"])
    h2 = int(core.shape[0] - n + 1)
    w2 = int(core.shape[1] - n + 1)
    assert h2 > 0 and w2 > 0

    trav = core == int(river)
    trav[int(mover["r0"]) : int(mover["r0"] + n), int(mover["c0"]) : int(mover["c0"] + n)] = True
    trav[int(dest["r0"]) : int(dest["r0"] + n), int(dest["c0"]) : int(dest["c0"] + n)] = True

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _start_direction(feasible: np.ndarray, sr: int, sc: int) -> int:
    dirs = []
    for d, (dr, dc) in enumerate(_DIRS):
        rr, cc = int(sr + dr), int(sc + dc)
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]):
            dirs.append(d)
    assert len(dirs) == 1
    return int(dirs[0])


def _obstacle_color_ahead(
    core: np.ndarray,
    r: int,
    c: int,
    d: int,
    n: int,
    ignore: set[int],
) -> int | None:
    h, w = core.shape
    if d == 0:
        if r - 1 < 0:
            return None
        strip = core[int(r - 1), int(c) : int(c + n)]
    elif d == 1:
        if c + n >= w:
            return None
        strip = core[int(r) : int(r + n), int(c + n)]
    elif d == 2:
        if r + n >= h:
            return None
        strip = core[int(r + n), int(c) : int(c + n)]
    else:
        if c - 1 < 0:
            return None
        strip = core[int(r) : int(r + n), int(c - 1)]
    cols = {int(x) for x in np.array(strip).flatten() if int(x) not in ignore}
    if len(cols) != 1:
        return None
    return int(next(iter(cols)))


def _simulate(core: np.ndarray, turn_by_color: dict[int, str], mover: dict, dest: dict, max_steps: int) -> tuple[int, int]:
    bg = int(Counter([int(core[0, 0]), int(core[0, -1]), int(core[-1, 0]), int(core[-1, -1])]).most_common(1)[0][0])
    river = _infer_river(core, bg=bg)
    feasible = _feasible_mask(core, river=river, mover=mover, dest=dest)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    assert bool(feasible[sr, sc])

    d = _start_direction(feasible, sr=sr, sc=sc)
    r, c = int(sr), int(sc)
    steps = 0
    ignore = {int(bg), int(river), int(_MOVER), int(_DEST), int(_SYMBOL)}
    seen: set[tuple[int, int, int, int]] = set()

    while steps < max_steps:
        if (int(r), int(c)) == (int(dr), int(dc)):
            break
        state = (int(r), int(c), int(d), int(steps))
        if state in seen:
            break
        seen.add(state)

        dr0, dc0 = _DIRS[int(d)]
        rr, cc = int(r + dr0), int(c + dc0)
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]):
            r, c = int(rr), int(cc)
            steps += 1
            continue

        wall_col = _obstacle_color_ahead(core, r=r, c=c, d=d, n=n, ignore=ignore)
        if wall_col is None:
            break
        turn = turn_by_color.get(int(wall_col))
        if turn is None:
            break

        nd = _rot_cw(d) if turn == "cw" else _rot_ccw(d)
        dr1, dc1 = _DIRS[int(nd)]
        rr, cc = int(r + dr1), int(c + dc1)
        if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and bool(feasible[rr, cc]):
            r, c, d = int(rr), int(cc), int(nd)
            steps += 1
            continue
        break

    return int(r), int(c)


def _sample_metrics(grid: np.ndarray) -> dict[str, int]:
    turn_by_color = _parse_legend(grid)
    core = np.array(grid[LEGEND_H:, :], dtype=int)
    bg = int(Counter([int(core[0, 0]), int(core[0, -1]), int(core[-1, 0]), int(core[-1, -1])]).most_common(1)[0][0])
    river = _infer_river(core, bg=bg)
    mover, dest = _detect_square_objects(core, bg=bg, river=river)
    max_steps = int(np.count_nonzero(np.array(mover["patch"]) == int(_SYMBOL)))
    actual = _simulate(core, turn_by_color=turn_by_color, mover=mover, dest=dest, max_steps=max_steps)

    feasible = _feasible_mask(core, river=river, mover=mover, dest=dest)
    dist = _bfs_placement(feasible, int(mover["r0"]), int(mover["c0"]))
    geom = _pick_farthest_within_steps(dist, int(dest["r0"]), int(dest["c0"]), max_steps=max_steps)

    swapped = {int(col): ("ccw" if rule == "cw" else "cw") for col, rule in turn_by_color.items()}
    swapped_pos = _simulate(core, turn_by_color=swapped, mover=mover, dest=dest, max_steps=max_steps)

    return {
        "max_steps": int(max_steps),
        "actual_r": int(actual[0]),
        "actual_c": int(actual[1]),
        "geom_r": int(geom[0]),
        "geom_c": int(geom[1]),
        "geom_diff": int(actual != geom),
        "swap_diff": int(actual != swapped_pos),
        "actual_dist": int(dist[int(actual[0]), int(actual[1])]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Read the top legend to map wall colors to CW / CCW turns.
    - Detect the mover as the only square object containing the symbol color.
    - Count symbol pixels inside the mover; that count is the maximum number of placement steps.
    - Simulate directed motion through the river: move forward when possible, otherwise read the
      colored wall ahead, turn as the legend instructs, and move one step in that new heading.
    - Stop on reaching the step cap, the red square, or a blockage that cannot be turned through.
    - Replace the mover's old position with river and paste the full mover patch at the stop.
    """
    g = np.array(grid, dtype=int)
    assert g.shape[0] > LEGEND_H

    turn_by_color = _parse_legend(g)
    core = g[LEGEND_H:, :].copy()

    bg = int(Counter([int(core[0, 0]), int(core[0, -1]), int(core[-1, 0]), int(core[-1, -1])]).most_common(1)[0][0])
    river = _infer_river(core, bg=bg)
    mover, dest = _detect_square_objects(core, bg=bg, river=river)

    mover_patch = np.array(mover["patch"], dtype=int)
    max_steps = int(np.count_nonzero(mover_patch == int(_SYMBOL)))
    assert max_steps >= 1

    tr, tc = _simulate(core, turn_by_color=turn_by_color, mover=mover, dest=dest, max_steps=max_steps)

    sr, sc, n = int(mover["r0"]), int(mover["c0"]), int(mover["n"])
    out_core = core.copy()
    out_core[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out_core[int(tr) : int(tr + n), int(tc) : int(tc + n)] = mover_patch

    out = g.copy()
    out[LEGEND_H:, :] = out_core
    return out


def _make_mover_patch(n: int, steps: int) -> np.ndarray:
    assert 1 <= steps <= n * n - 1
    patch = np.full((n, n), _MOVER, dtype=int)
    cells = [(r, c) for r in range(n) for c in range(n) if not (r == 0 and c == 0)]
    for r, c in cells[:steps]:
        patch[r, c] = _SYMBOL
    return patch


def _pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _paint_edge(grid: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int) -> None:
    pr1, pc1 = p1
    pr2, pc2 = p2
    if pr1 == pr2:
        c0, c1 = min(pc1, pc2), max(pc1, pc2) + n
        grid[int(pr1) : int(pr1 + n), int(c0) : int(c1)] = _RIVER
    else:
        r0, r1 = min(pr1, pr2), max(pr1, pr2) + n
        grid[int(r0) : int(r1), int(pc1) : int(pc1 + n)] = _RIVER


def _render_network(
    grid: np.ndarray,
    nodes: set[tuple[int, int]],
    edges: set[tuple[tuple[int, int], tuple[int, int]]],
    n: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    pixel = {cell: (int(cell[0]), int(cell[1])) for cell in nodes}
    for pr, pc in pixel.values():
        grid[int(pr) : int(pr + n), int(pc) : int(pc + n)] = _RIVER
    for a, b in edges:
        _paint_edge(grid, pixel[a], pixel[b], n=n)
    return pixel


def _add_turn_wall(grid: np.ndarray, pr: int, pc: int, n: int, incoming_dir: int, color: int) -> bool:
    h, w = grid.shape
    if incoming_dir == 0:
        if pr - 1 < 0:
            return False
        grid[int(pr - 1), int(pc) : int(pc + n)] = int(color)
    elif incoming_dir == 1:
        if pc + n >= w:
            return False
        grid[int(pr) : int(pr + n), int(pc + n)] = int(color)
    elif incoming_dir == 2:
        if pr + n >= h:
            return False
        grid[int(pr + n), int(pc) : int(pc + n)] = int(color)
    else:
        if pc - 1 < 0:
            return False
        grid[int(pr) : int(pr + n), int(pc - 1)] = int(color)
    return True


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    for d, (rr, cc) in enumerate(_DIRS):
        if (dr, dc) == (rr, cc):
            return int(d)
    raise AssertionError("non-adjacent coarse step")


def _transform_cell(cell: tuple[int, int], flip_r: bool, flip_c: bool) -> tuple[int, int]:
    r, c = cell
    if flip_r:
        r = (_H2 - 1) - r
    if flip_c:
        c = (_W2 - 1) - c
    return int(r), int(c)


def _build_template_instance() -> tuple[list[tuple[int, int]], list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]]]:
    flip_r = random.random() < 0.5
    flip_c = random.random() < 0.5
    main_path = [_transform_cell(cell, flip_r=flip_r, flip_c=flip_c) for cell in _BASE_MAIN_PATH]
    decoy_edges = [
        (
            _transform_cell(a, flip_r=flip_r, flip_c=flip_c),
            _transform_cell(b, flip_r=flip_r, flip_c=flip_c),
        )
        for a, b in _BASE_DECOY_EDGES
    ]
    red_branch = [_transform_cell(cell, flip_r=flip_r, flip_c=flip_c) for cell in _BASE_RED_BRANCH]
    return main_path, decoy_edges, red_branch


def _gen_attempt() -> np.ndarray | None:
    n = _N
    max_steps = random.choice([3, 5, 7])

    turn_by_color = {
        int(_TURN_COLORS[0]): random.choice(["cw", "ccw"]),
        int(_TURN_COLORS[1]): None,
    }
    turn_by_color[int(_TURN_COLORS[1])] = "ccw" if turn_by_color[int(_TURN_COLORS[0])] == "cw" else "cw"
    color_for_turn = {rule: col for col, rule in turn_by_color.items()}

    main_path, decoy_edges, red_branch = _build_template_instance()
    turn_events: list[dict] = []
    for i in range(1, len(main_path) - 1):
        d_in = _dir_between(main_path[i - 1], main_path[i])
        d_out = _dir_between(main_path[i], main_path[i + 1])
        if d_in == d_out:
            continue
        turn = "cw" if _rot_cw(d_in) == d_out else "ccw"
        turn_events.append({"node": tuple(main_path[i]), "incoming_dir": int(d_in), "turn": turn})
    if len(turn_events) != 3:
        return None

    core = np.full((_CORE_H, _CORE_W), _BG, dtype=int)
    main_edges = {_pair(a, b) for a, b in zip(main_path, main_path[1:])}
    branch_edges = list(decoy_edges)
    branch_edges.append((tuple(main_path[2]), tuple(red_branch[0])))
    for a, b in zip(red_branch, red_branch[1:], strict=False):
        branch_edges.append((tuple(a), tuple(b)))

    nodes = set(main_path)
    for a, b in branch_edges:
        nodes.add(a)
        nodes.add(b)
    all_edges = set(main_edges) | {_pair(a, b) for a, b in branch_edges}
    pixel = _render_network(core, nodes=nodes, edges=all_edges, n=n)

    for ev in turn_events:
        pr, pc = pixel[tuple(ev["node"])]
        wall_col = int(color_for_turn[str(ev["turn"])])
        if not _add_turn_wall(core, pr=pr, pc=pc, n=n, incoming_dir=int(ev["incoming_dir"]), color=wall_col):
            return None

    mover_cell = tuple(main_path[0])
    expected_cell = tuple(main_path[max_steps])
    dest_cell = tuple(red_branch[-1])
    mover_r, mover_c = pixel[mover_cell]
    dest_r, dest_c = pixel[dest_cell]

    mover_patch = _make_mover_patch(n, max_steps)
    core[int(mover_r) : int(mover_r + n), int(mover_c) : int(mover_c + n)] = mover_patch
    core[int(dest_r) : int(dest_r + n), int(dest_c) : int(dest_c + n)] = _DEST

    legend_order = list(_TURN_COLORS)
    random.shuffle(legend_order)
    legend = _build_legend(_CORE_W, turn_by_color=turn_by_color, order=legend_order)
    full = np.vstack([legend, core])

    turn_map = _parse_legend(full)
    if set(turn_map) != set(_TURN_COLORS):
        return None
    if {turn_map[c] for c in _TURN_COLORS} != {"cw", "ccw"}:
        return None

    core_only = np.array(full[LEGEND_H:, :], dtype=int)
    bg = int(Counter([int(core_only[0, 0]), int(core_only[0, -1]), int(core_only[-1, 0]), int(core_only[-1, -1])]).most_common(1)[0][0])
    river = _infer_river(core_only, bg=bg)
    mover, dest = _detect_square_objects(core_only, bg=bg, river=river)
    if int(mover["n"]) != n:
        return None

    try:
        variant_tr, variant_tc = _simulate(core_only, turn_by_color=turn_map, mover=mover, dest=dest, max_steps=max_steps)
    except AssertionError:
        return None

    exp_r, exp_c = pixel[expected_cell]
    if (variant_tr, variant_tc) != (int(exp_r), int(exp_c)):
        return None
    if (variant_tr, variant_tc) == (int(dest["r0"]), int(dest["c0"])):
        return None

    metrics = _sample_metrics(full)
    if int(metrics["geom_diff"]) != 1:
        return None
    if int(metrics["swap_diff"]) != 1:
        return None
    if int(metrics["actual_dist"]) != max_steps:
        return None

    out = main(full)
    expected = full.copy()
    expected[LEGEND_H + int(mover_r) : LEGEND_H + int(mover_r + n), int(mover_c) : int(mover_c + n)] = _RIVER
    expected[LEGEND_H + int(exp_r) : LEGEND_H + int(exp_r + n), int(exp_c) : int(exp_c + n)] = mover_patch
    if not np.array_equal(out, expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate valid input after 400 attempts")
