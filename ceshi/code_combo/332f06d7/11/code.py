# Combo variant of 332f06d7 (index 11)
# combo_spec: D5/0+D12/0
# Name: 多目标选择+分支通道

"""
Fusion design analysis
======================

a) Concrete behavior and trace
   - The source task has one black n x n mover, one red n x n destination, and one winding blue
     corridor. The solver picks the only destination and BFS runs on legal n x n placements.
   - This combo keeps the same slide mechanics, but the blue corridor now visibly branches. There
     are 2-3 colored n x n target squares placed at branch termini: one at the end of the main
     continuation branch and the others at dead-end side branches.
   - A small isolated bar in the green background repeats exactly one target color. That visible
     color echo selects the true target. The dead-end targets are reachable distractors; they are
     never the selected target.
   - Example trace:
       original: black + red -> BFS to red if reachable.
       combo: black + yellow/orange/grey targets + isolated yellow bar -> BFS still explores every
       legal placement on the branched river, but it aims for the yellow target because yellow is
       the only target color repeated by the selector bar. If a chokepoint blocks the main branch,
       the yellow target becomes unreachable while orange/grey dead-end targets may still be
       reachable; the mover then stops at the farthest legal placement before the gate.

b) Complexity calibration
   - Branch count is limited to 1-2 dead-end branches, and target count is therefore 2-3. This is
     enough to make the branch choice visually obvious without turning the river into a maze.
   - Only one selector bar is used, and target colors are all distinct. That keeps the selection
     rule crisp and avoids ambiguous matches.
   - In reachable samples, at least one distractor target is also reachable, so the puzzle does not
     collapse to "go to the only reachable target". In stuck samples, at least one distractor target
     stays reachable while the selected target is blocked, so the stop behavior stays informative.

c) Implicit assumptions audit against the source main()
   - Source assumption: there are exactly two same-sized solid squares and color order picks mover
     and destination. This is broken by the combo, so the new solver identifies the mover by its
     fixed black color and gathers all same-sized target squares separately.
   - Source assumption: traversable colors are river ∪ mover ∪ one destination color. This is no
     longer valid because all target squares lie on the river and must count as traversable, while
     the selector bar must not. The new solver therefore paints traversability from the river plus
     the detected target square bounding boxes, not from all pixels of matching colors.
   - Source assumption: the "second square by color" is the true destination. That hidden ordering
     is not ARC-legible here. The new solver instead chooses the target color whose component set
     includes a non-square bar, with a component-count fallback.
   - Branching itself does not break BFS. The only new generator-side risk is tie-breaking in stuck
     cases: a dead-end branch must not be longer than the blocked main continuation. The generator
     validates that the farthest reachable placement still lies on the main trunk before accepting.

d) Placement legality
   - The river is generated in placement-coordinate space, then rendered as n x n blue windows, so
     every intended corridor position is guaranteed to be a legal placement before optional choking.
   - Targets occupy distinct branch endpoints and all have the same size as the mover, so they do
     not overlap and stay valid solid squares.
   - The selector bar is placed only on a background patch with a one-cell background moat, so it
     cannot merge with the river or with a target.
   - The chokepoint is carved only on the final main-branch segment after the side branches split
     off, ensuring dead-end targets stay reachable in blocked cases.

e) Visual observability
   - The rule is human-visible: branch topology is seen directly in the blue river, and the target
     choice is indicated by exact color equality between one target and the isolated bar.
   - No hidden numeric ordering such as "smallest color id wins" is used.
   - Across examples, a solver can infer both parts: side-branch targets are distractors, and the
     repeated-color target is the only one the black square tries to cover.

f) Visual cleanliness
   - Colors are limited to the original background/river/mover colors plus 2-3 target colors.
   - The selector bar lives off-river, so the river geometry remains readable.
   - Junctions are sparse and separated; targets sit at branch tips instead of cluttering the
     interior channel.

g) Change inferability
   - There is no color remapping rule beyond equality. One target color is repeated by a small bar,
     and the mover heads to that matching target when possible.
   - The branch structure provides the second visible cue: some targets are placed at obvious
     dead-end termini, and the selected target is always on the main continuation branch.

New solver assumptions enforced by generate_input()
   - Exactly one solid black square exists and defines n.
   - Every target is a solid n x n square with a distinct non-river color.
   - Exactly one target color also appears in a separate non-square bar.
   - The river is the most common remaining non-background color after excluding mover/target
     colors.
   - At least one dead-end target is reachable from the start, and in blocked cases the selected
     target is unreachable while the farthest reachable placement stays on the main trunk.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


_BG, _RIVER, _MOVER = 3, 1, 0
_TARGET_COLORS = [2, 4, 5, 6, 7, 8, 9]


def _is_full_square(grid: np.ndarray, comp: dict) -> bool:
    b = comp["bbox"]
    if bbox_h(b) != bbox_w(b):
        return False
    col = int(comp["color"])
    patch = grid[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
    return bool((patch == int(col)).all())


def _collect_square_components(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[dict]], Counter]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    squares: list[dict] = []
    by_color: dict[int, list[dict]] = defaultdict(list)
    counts: Counter = Counter()
    for comp in comps:
        col = int(comp["color"])
        counts[col] += 1
        by_color[col].append(comp)
        if not _is_full_square(grid, comp):
            continue
        b = comp["bbox"]
        squares.append({"col": col, "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})
    return squares, by_color, counts


def _detect_river_color(grid: np.ndarray, bg: int, excluded_colors: set[int]) -> int:
    counts = Counter(map(int, grid.flatten()))
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in excluded_colors:
            continue
        return int(col)
    raise AssertionError("river color not found")


def _selected_target_color(
    grid: np.ndarray,
    by_color: dict[int, list[dict]],
    component_counts: Counter,
    target_colors: set[int],
) -> int:
    selected = []
    for col in sorted(target_colors):
        if any(not _is_full_square(grid, comp) for comp in by_color[col]):
            selected.append(int(col))
    if len(selected) == 1:
        return int(selected[0])
    selected = [int(col) for col in sorted(target_colors) if int(component_counts[col]) >= 2]
    assert len(selected) == 1
    return int(selected[0])


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _find_farthest(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares, by_color, component_counts = _collect_square_components(g, bg)
    movers = [sq for sq in squares if int(sq["col"]) == _MOVER]
    assert len(movers) == 1
    mover = movers[0]
    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])

    targets = [sq for sq in squares if int(sq["col"]) != _MOVER and int(sq["n"]) == n]
    assert len(targets) >= 2
    target_colors = {int(t["col"]) for t in targets}
    assert len(target_colors) == len(targets)

    selected_color = _selected_target_color(g, by_color, component_counts, target_colors)
    selected_targets = [t for t in targets if int(t["col"]) == selected_color]
    assert len(selected_targets) == 1
    dest = selected_targets[0]
    dr, dc = int(dest["r0"]), int(dest["c0"])

    river = _detect_river_color(g, bg, excluded_colors={_MOVER, *target_colors})
    trav = g == river
    trav[sr : sr + n, sc : sc + n] = True
    for sq in targets:
        rr, cc = int(sq["r0"]), int(sq["c0"])
        trav[rr : rr + n, cc : cc + n] = True

    feasible = _compute_feasible(trav, n)
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2 and bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
    else:
        tr, tc = _find_farthest(dist, sr, sc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _polyline(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    assert r1 == r2 or c1 == c2
    out = []
    if r1 == r2:
        step = 1 if c2 >= c1 else -1
        for c in range(c1, c2 + step, step):
            out.append((r1, c))
    else:
        step = 1 if r2 >= r1 else -1
        for r in range(r1, r2 + step, step):
            out.append((r, c1))
    return out


def _concat_segments(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _polyline(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        out.extend(seg)
    return out


def _render_from_placements(grid: np.ndarray, placements: set[tuple[int, int]], n: int, color: int) -> None:
    for r, c in placements:
        grid[r : r + n, c : c + n] = color


def _choices_with_gap(lo: int, hi: int, gap: int) -> list[tuple[int, int, int]]:
    vals = list(range(lo, hi + 1))
    out = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            for k in range(j + 1, len(vals)):
                a, b, c = vals[i], vals[j], vals[k]
                if b - a >= gap and c - b >= gap:
                    out.append((a, b, c))
    return out


def _branch_metrics(feasible: np.ndarray, sr: int, sc: int) -> dict[str, object]:
    h2, w2 = feasible.shape
    dist = _bfs_placement(feasible, sr, sc)
    reachable = [(r, c) for r in range(h2) for c in range(w2) if int(dist[r, c]) >= 0]
    degrees: dict[tuple[int, int], int] = {}
    for r, c in reachable:
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and int(dist[rr, cc]) >= 0:
                deg += 1
        degrees[(r, c)] = deg
    leaves = [p for p, deg in degrees.items() if deg == 1]
    branches = [p for p, deg in degrees.items() if deg >= 3]
    return {"dist": dist, "reachable": reachable, "degrees": degrees, "leaves": leaves, "branches": branches}


def _carve_horizontal_chokepoint(grid: np.ndarray, row: int, c_lo: int, c_hi: int, n: int) -> bool:
    span_start = c_lo + 1
    span_end = c_hi + n - 2
    if span_end < span_start:
        return False
    x = random.randint(span_start, span_end)
    remove = random.randint(1, n - 1)
    if random.random() < 0.5:
        grid[row : row + remove, x] = _BG
    else:
        grid[row + n - remove : row + n, x] = _BG
    return True


def _size_choices(n: int) -> list[int]:
    if n == 1:
        return list(range(10, 15))
    if n == 2:
        return list(range(12, 17))
    if n == 3:
        return list(range(15, 21))
    return [18, 19, 20]


def _find_selector_slot(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    H, W = grid.shape
    shapes = [(1, 2), (2, 1)]
    random.shuffle(shapes)
    positions = [(r, c) for r in range(1, H - 1) for c in range(1, W - 1)]
    positions.sort(
        key=lambda rc: min(
            rc[0] + rc[1],
            rc[0] + (W - 1 - rc[1]),
            (H - 1 - rc[0]) + rc[1],
            (H - 1 - rc[0]) + (W - 1 - rc[1]),
        )
    )
    for h, w in shapes:
        for r, c in positions:
            if r + h >= H or c + w >= W:
                continue
            if not bool((grid[r : r + h, c : c + w] == _BG).all()):
                continue
            rr0, rr1 = r - 1, r + h + 1
            cc0, cc1 = c - 1, c + w + 1
            if rr0 < 0 or cc0 < 0 or rr1 > H or cc1 > W:
                continue
            if bool((grid[rr0:rr1, cc0:cc1] == _BG).all()):
                return int(r), int(c), int(h), int(w)
    return None


def _target_specs(
    main_end: tuple[int, int],
    branch_ends: list[tuple[int, int]],
) -> list[dict[str, int]]:
    colors = random.sample(_TARGET_COLORS, 1 + len(branch_ends))
    specs = [{"r": int(main_end[0]), "c": int(main_end[1]), "col": int(colors[0]), "selected": 1}]
    for i, pos in enumerate(branch_ends, start=1):
        specs.append({"r": int(pos[0]), "c": int(pos[1]), "col": int(colors[i]), "selected": 0})
    return specs


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.6
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])

    G = random.choice(_size_choices(n))
    H = W = G
    h2 = H - n + 1
    w2 = W - n + 1
    if h2 < 9 or w2 < 9:
        return None

    triples = _choices_with_gap(2, h2 - 3, 2)
    if not triples:
        return None
    a, b, c = random.choice(triples)
    if random.random() < 0.5:
        r0, r1, r2 = b, a, c
        branch_dir_1 = -1
        branch_dir_2 = 1
    else:
        r0, r1, r2 = b, c, a
        branch_dir_1 = 1
        branch_dir_2 = -1

    c1_lo = 4
    c1_hi = min(max(4, w2 // 3), w2 - 6)
    if c1_hi < c1_lo:
        return None
    c1 = random.randint(c1_lo, c1_hi)
    c2_lo = c1 + 3
    c2_hi = w2 - 4
    if c2_hi < c2_lo:
        return None
    c2 = random.randint(c2_lo, c2_hi)

    start = (r0, 1)
    p1 = (r0, c1)
    p2 = (r1, c1)
    p3 = (r1, c2)
    p4 = (r2, c2)
    main_end = (r2, w2 - 1)
    trunk = _concat_segments([start, p1, p2, p3, p4, main_end])
    if len(trunk) != len(set(trunk)):
        return None
    placements = set(trunk)

    bp1_col = random.randint(c1 + 1, c2 - 1)
    bp1 = (r1, bp1_col)
    end1 = (0, bp1_col) if branch_dir_1 < 0 else (h2 - 1, bp1_col)
    branch1 = _polyline(bp1, end1)[1:]
    if len(branch1) < 2:
        return None
    placements.update(branch1)
    branch_ends = [end1]

    if c1 >= 4 and random.random() < 0.65:
        bp2_col = random.randint(2, c1 - 1)
        bp2 = (r0, bp2_col)
        end2 = (0, bp2_col) if branch_dir_2 < 0 else (h2 - 1, bp2_col)
        branch2 = _polyline(bp2, end2)[1:]
        if len(branch2) >= 2 and not placements.intersection(branch2):
            placements.update(branch2)
            branch_ends.append(end2)

    grid = np.full((H, W), _BG, dtype=int)
    _render_from_placements(grid, placements, n, _RIVER)

    river_ratio = float(np.count_nonzero(grid == _RIVER)) / float(H * W)
    if not (0.20 <= river_ratio <= 0.62):
        return None

    if not reached:
        if not _carve_horizontal_chokepoint(grid, r2, c2, main_end[1], n):
            return None

    target_specs = _target_specs(main_end, branch_ends)
    selected_target = next(spec for spec in target_specs if int(spec["selected"]) == 1)
    distractors = [spec for spec in target_specs if int(spec["selected"]) == 0]

    for spec in target_specs:
        rr, cc = int(spec["r"]), int(spec["c"])
        grid[rr : rr + n, cc : cc + n] = int(spec["col"])

    marker = _find_selector_slot(grid)
    if marker is None:
        return None
    mr, mc, mh, mw = marker
    grid[mr : mr + mh, mc : mc + mw] = int(selected_target["col"])

    sr, sc = start
    grid[sr : sr + n, sc : sc + n] = _MOVER

    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    if Counter(corners).most_common(1)[0][0] != _BG:
        return None

    trav = grid == _RIVER
    trav[sr : sr + n, sc : sc + n] = True
    for spec in target_specs:
        rr, cc = int(spec["r"]), int(spec["c"])
        trav[rr : rr + n, cc : cc + n] = True
    feasible = _compute_feasible(trav, n)
    if not bool(feasible[sr, sc]):
        return None

    metrics = _branch_metrics(feasible, sr, sc)
    dist = metrics["dist"]
    if len(metrics["branches"]) < 1:
        return None
    if len(metrics["leaves"]) < 3:
        return None

    dr, dc = int(selected_target["r"]), int(selected_target["c"])
    dest_ok = bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if reached != dest_ok:
        return None

    reachable_distractors = 0
    for spec in distractors:
        rr, cc = int(spec["r"]), int(spec["c"])
        if bool(feasible[rr, cc]) and int(dist[rr, cc]) >= 0:
            reachable_distractors += 1
    if reachable_distractors < 1:
        return None

    if dest_ok:
        tr, tc = dr, dc
    else:
        tr, tc = _find_farthest(dist, sr, sc)
        if (int(tr), int(tc)) not in set(trunk):
            return None
        if int(dist[tr, tc]) <= max(int(dist[int(spec["r"]), int(spec["c"])] ) for spec in distractors):
            return None

    output = grid.copy()
    output[sr : sr + n, sc : sc + n] = _RIVER
    output[tr : tr + n, tc : tc + n] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
