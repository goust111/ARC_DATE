# Combo variant of 332f06d7 (index 20)
# combo_spec: D9/2+D14/1
# Name: 非正方形网格+密集宽通道

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

"""
Design analysis for combo D9/2 + D14/1
--------------------------------------

a) Concrete behavior and trace
   - Original task:
     the grid is square, green background dominates the corners, and blue forms
     a relatively sparse winding river. A black n x n square moves through blue
     if and only if every intermediate n x n placement stays entirely on blue.
   - This combo changes two meta layers together:
     1. D9/2: the canvas becomes rectangular instead of square, often clearly
        wide or tall, so one axis visually supports long straight travel while
        the other axis is the constrained crossing axis.
     2. D14/1: blue now occupies most of the board, while green becomes a small
        set of obstacle islands / barrier strips inside a large water region.
   - Small concrete trace for n = 2 on a 12 x 20 grid:
     start with mostly blue water. Insert three thin vertical green barriers.
     Each barrier spans almost the full height but leaves one opening. Openings
     alternate high / low / high, so the black 2 x 2 block must slide right,
     then down, then right, then up, then right to approach the red target.
     In a reachable sample, every opening has height >= 2, so the 2 x 2 block
     can pass through all barriers and land on the red square.
     In a stuck sample, the middle opening is only 1 pixel tall. The blue water
     still visually connects through that slit, but a 2 x 2 window cannot fit,
     so BFS on feasible placements disconnects there and the block stops at the
     farthest reachable location just before that choke.
   - Difference from original:
     visually, the "channel" is no longer a thin river drawn inside a dominant
     background. Instead, the water is the dominant field and green obstacles
     carve a dense wide corridor system out of it. The motion rule itself stays
     the same, but the object-identification and generation assumptions change.

b) Complexity calibration
   - The combo is intentionally not simplified into a nearly empty rectangle.
     Every sample has:
       * a clearly non-square aspect ratio,
       * traversable area ratio in the 70-90% range,
       * multiple barrier crossings with alternating gap heights,
       * long straight travel on the long axis and constrained crossings on the
         short axis.
   - Obstacles are sparse and low-variety: a few long barriers plus at most two
     tiny non-square islands. That keeps the scene readable while making the
     D14/1 inversion obvious to the eye.

c) Hidden-assumption audit of the original solver
   - Original line:
       bg = corner-majority(...)
     Hidden assumption: background dominates corners. Broken here because dense
     water may reach 2-4 corners. Fix: do not infer bg from corners at all.
   - Original component scan:
       find_components(..., bg=bg)
     Hidden assumption: we already know bg. Broken if corner-majority fails.
     Fix: detect solid same-color square components directly per color without
     excluding any presumed background.
   - Original river inference:
       most common remaining color excluding bg and the two token colors
     Hidden assumption: bg is already known. Unnecessary here. Since D14/1
     makes blue dominant, we can infer river as the most common non-token color.
   - Original feasible-placement BFS:
       shape (H - n + 1, W - n + 1), 4-neighbor movement, row-major tie break
     Hidden assumption: none about squareness. This still holds unchanged on a
     rectangular board, so the actual motion engine is preserved.

d) Placement legality
   - Generation works on a rectangular canvas with long axis > short axis.
   - Barrier strips are carved before tokens are placed, so mover/destination
     are always pasted onto verified feasible n x n water placements.
   - Stuck samples only use n >= 2; the critical opening height is chosen from
     1..n-1, so the water remains pixel-connected but placement-connectedness is
     broken exactly by width.
   - Additional islands are only non-square 1x2 / 1x3 / 2x3 / 2x4 rectangles,
     preventing accidental extra n x n token candidates.

e) Visual observability
   - Humans can see three stable cues across 2-4 examples:
       * the canvas is obviously rectangular,
       * blue occupies most cells and green is sparse obstacle matter,
       * progress along the long axis happens in long runs, while crossing the
         short axis happens through offset gates.
   - The rule is still inferable from I/O because the output only moves the
     black square and restores its old cells to blue, exactly as before; the
     visible novelty is the obstacle-dense rectangular environment.

f) Visual cleanliness
   - Obstacles are all green, connected in simple strips / tiny blobs, and do
     not overlap the black/red squares.
   - The sparse islands are deliberately kept non-square and few in number, so
     the only salient solid square objects remain the mover and destination.

g) Quantitative behavior target for this combo
   - For generated inputs, the intended fused behavior is:
       * aspect ratio >= 1.25,
       * traversable ratio (blue + black + red) between 0.70 and 0.90,
       * old corner-background solver should often disagree or fail because its
         corner-majority assumption is no longer reliable.

h) New solver assumptions and generator guarantees
   - Assumption: the only repeated solid n x n components are the mover and
     destination. Guaranteed by using only strip/blob obstacles that are not
     solid squares.
   - Assumption: among non-token colors, the dominant color is the river.
     Guaranteed by enforcing dense-water occupancy.
   - Assumption: stuck cases have a true sub-n bottleneck. Guaranteed by one
     barrier opening of height 1..n-1 while all other required openings are >= n.
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)


def _solid_square_components(g: np.ndarray) -> list[dict]:
    squares: list[dict] = []
    for col in sorted(map(int, np.unique(g))):
        mask = g == int(col)
        labeled, num = scipy_label(mask.astype(int), structure=_STRUCT4)
        for idx in range(1, int(num) + 1):
            coords = np.argwhere(labeled == idx)
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h != w:
                continue
            patch = g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
            if bool((patch == int(col)).all()):
                squares.append({"col": int(col), "r0": int(r0), "c0": int(c0), "n": int(h)})
    return squares


def _locate_tokens(g: np.ndarray) -> tuple[dict, dict, int]:
    squares = _solid_square_components(g)
    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)

    valid = [item for item in by_n.items() if len(item[1]) >= 2]
    assert valid
    nside, cand = max(valid, key=lambda kv: (len(kv[1]) == 2, len(kv[1]), kv[0]))
    cand = sorted(cand, key=lambda s: int(s["col"]))
    mover = cand[0]
    dest = cand[1]
    return mover, dest, int(nside)


def _infer_river(g: np.ndarray, token_colors: tuple[int, int]) -> int:
    counts = Counter(map(int, g.flatten()))
    for col in token_colors:
        counts.pop(int(col), None)
    assert counts
    return int(max(counts.items(), key=lambda kv: kv[1])[0])


def _compute_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = map(int, trav.shape)
    h2 = int(H - n + 1)
    w2 = int(W - n + 1)
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[int(r) : int(r + n), int(c) : int(c + n)].all()):
                feasible[int(r), int(c)] = True
    return feasible


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = map(int, feasible.shape)
    dist = -np.ones((h2, w2), dtype=int)
    dist[int(sr), int(sc)] = 0
    q = deque([(int(sr), int(sc))])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = int(r + dr), int(c + dc)
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[int(r), int(c)] + 1)
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the solid square through dense water exactly as in the base task.

    The movement law is unchanged from 332f06d7:
    - infer the mover and destination as the two same-sized solid squares,
    - infer the river color as the dominant non-token color,
    - BFS over all n x n placements fully covered by traversable cells,
    - reachable -> land on destination, otherwise stop at the farthest
      reachable placement in row-major tie order,
    - restore the old mover footprint to river color.

    What changes in this combo is not the motion engine but the scene geometry:
    rectangular boards and dense water invalidate the original corner-based
    background heuristic, so token / river detection is done without it.
    """
    g = np.array(grid, dtype=int)
    mover, dest, n = _locate_tokens(g)

    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    river = _infer_river(g, (int(mover["col"]), int(dest["col"])))

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _compute_feasible(trav, int(n))
    h2, w2 = map(int, feasible.shape)
    assert 0 <= sr < h2 and 0 <= sc < w2 and bool(feasible[int(sr), int(sc)])

    dist = _bfs_placement(feasible, int(sr), int(sc))
    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[int(dr), int(dc)]) and int(dist[int(dr), int(dc)]) >= 0:
        tr, tc = int(dr), int(dc)
    else:
        best = -1
        tr, tc = int(sr), int(sc)
        for r in range(h2):
            for c in range(w2):
                if int(dist[int(r), int(c)]) > best:
                    best = int(dist[int(r), int(c)])
                    tr, tc = int(r), int(c)

    out = g.copy()
    out[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(river)
    out[int(tr) : int(tr + n), int(tc) : int(tc + n)] = int(mover["col"])
    return out


def _choose_dims(n: int) -> tuple[int, int, bool]:
    short_ranges = {
        1: (10, 14),
        2: (12, 16),
        3: (14, 18),
        4: (16, 20),
    }
    short_lo, short_hi = short_ranges[int(n)]
    short_dim = random.randint(int(short_lo), int(short_hi))
    long_lo = max(int(short_dim + 4), int(short_lo + 5))
    long_hi = min(28, int(short_dim + 10))
    if long_lo > long_hi:
        return _choose_dims(int(n))
    long_dim = random.randint(int(long_lo), int(long_hi))
    make_tall = random.random() < 0.5
    return int(short_dim), int(long_dim), bool(make_tall)


def _pick_gap_start(H: int, gap_h: int, idx: int) -> int | None:
    limit = int(H - 1 - gap_h)
    if limit < 1:
        return None
    if idx % 2 == 0:
        lo, hi = 1, max(1, limit // 3)
    else:
        lo, hi = min(limit, max(1, (2 * limit) // 3)), limit
    if lo > hi:
        lo = hi = max(1, min(limit, lo))
    return int(random.randint(int(lo), int(hi)))


def _add_barriers(grid: np.ndarray, n: int, reached: bool) -> dict | None:
    H, W = map(int, grid.shape)
    stripe_count = random.choice([2, 3] if W < 24 else [3, 4])
    thick = random.choice([1, 2] if n <= 2 else [2, 3])
    centers = np.linspace(W / (stripe_count + 1), W * stripe_count / (stripe_count + 1), stripe_count)
    starts = []
    for center in centers:
        c0 = int(round(float(center) - thick / 2.0))
        c0 = max(2, min(W - thick - 2, c0))
        starts.append(int(c0))
    starts = sorted(starts)
    if len(starts) != stripe_count:
        return None
    for i in range(len(starts) - 1):
        if starts[i + 1] - (starts[i] + thick) < max(3, n + 1):
            return None

    critical_idx = None if reached or n == 1 else random.randrange(stripe_count)
    gap_heights: list[int] = []
    for idx, c0 in enumerate(starts):
        grid[1 : H - 1, int(c0) : int(c0 + thick)] = int(_BG)
        if critical_idx is not None and idx == critical_idx:
            gap_h = random.randint(1, n - 1)
        else:
            wide_lo = max(n + 1, min(H - 4, n + 2))
            wide_hi = max(wide_lo, min(H - 3, n + 5))
            gap_h = random.randint(int(wide_lo), int(wide_hi))
        gap_r = _pick_gap_start(H, int(gap_h), idx)
        if gap_r is None:
            return None
        grid[int(gap_r) : int(gap_r + gap_h), int(c0) : int(c0 + thick)] = int(_RIVER)
        gap_heights.append(int(gap_h))

    return {"stripe_count": int(stripe_count), "thick": int(thick), "critical_idx": critical_idx, "gap_heights": gap_heights}


def _add_small_islands(grid: np.ndarray) -> None:
    H, W = map(int, grid.shape)
    shapes = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 2)]
    for _ in range(random.randint(0, 2)):
        for _attempt in range(20):
            bh, bw = random.choice(shapes)
            if bh >= H - 4 or bw >= W - 4:
                continue
            r = random.randint(2, H - bh - 2)
            c = random.randint(2, W - bw - 2)
            if not bool((grid[int(r) : int(r + bh), int(c) : int(c + bw)] == int(_RIVER)).all()):
                continue
            grid[int(r) : int(r + bh), int(c) : int(c + bw)] = int(_BG)
            break


def _pick_terminals(feasible: np.ndarray, reached: bool) -> tuple[tuple[int, int], tuple[int, int], np.ndarray] | None:
    h2, w2 = map(int, feasible.shape)
    band = max(1, min(3, w2 // 6))
    left = [(r, c) for r in range(h2) for c in range(min(w2, band + 1)) if bool(feasible[r, c])]
    right = [(r, c) for r in range(h2) for c in range(max(0, w2 - band - 1), w2) if bool(feasible[r, c])]
    if not left or not right:
        return None

    random.shuffle(left)
    left.sort(key=lambda rc: (rc[1], abs(rc[0] - h2 // 2)))
    right_pref = sorted(right, key=lambda rc: (-rc[1], abs(rc[0] - h2 // 2)))

    for sr, sc in left:
        dist = _bfs_placement(feasible, int(sr), int(sc))
        if reached:
            candidates = [(int(dist[r, c]), r, c) for r, c in right_pref if int(dist[r, c]) >= max(3, w2 // 3)]
            if candidates:
                candidates.sort(key=lambda item: (item[0], item[2], -abs(item[1] - h2 // 2)), reverse=True)
                _, dr, dc = candidates[0]
                return (int(sr), int(sc)), (int(dr), int(dc)), dist
        else:
            if int(dist.max()) < max(3, w2 // 4):
                continue
            candidates = [(r, c) for r, c in right_pref if int(dist[r, c]) == -1]
            if candidates:
                dr, dc = candidates[0]
                return (int(sr), int(sc)), (int(dr), int(dc)), dist
    return None


def _trav_ratio(grid: np.ndarray) -> float:
    return float(np.isin(grid, (int(_RIVER), int(_MOVER), int(_DEST))).mean())


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([1, 2, 3, 4])
    reached = True if n == 1 else bool(random.random() < 0.55)
    Hc, Wc, make_tall = _choose_dims(int(n))
    grid = np.full((int(Hc), int(Wc)), int(_RIVER), dtype=int)

    barrier_info = _add_barriers(grid, int(n), bool(reached))
    if barrier_info is None:
        return None
    _add_small_islands(grid)

    river_ratio = float((grid == int(_RIVER)).mean())
    if not (0.70 <= river_ratio <= 0.90):
        return None

    feasible = _compute_feasible(grid == int(_RIVER), int(n))
    picks = _pick_terminals(feasible, bool(reached))
    if picks is None:
        return None
    (sr, sc), (dr, dc), dist = picks

    grid[int(sr) : int(sr + n), int(sc) : int(sc + n)] = int(_MOVER)
    grid[int(dr) : int(dr + n), int(dc) : int(dc + n)] = int(_DEST)

    if make_tall:
        grid = grid.T

    H, W = map(int, grid.shape)
    if H == W or max(H, W) / min(H, W) < 1.25:
        return None
    if not (0.70 <= _trav_ratio(grid) <= 0.90):
        return None

    out = main(grid)
    mover, dest, n2 = _locate_tokens(grid)
    if int(n2) != int(n):
        return None
    river = _infer_river(grid, (int(mover["col"]), int(dest["col"])))
    trav = (grid == int(river)) | (grid == int(_MOVER)) | (grid == int(_DEST))
    feasible2 = _compute_feasible(trav, int(n2))
    dist2 = _bfs_placement(feasible2, int(mover["r0"]), int(mover["c0"]))
    dest_ok = int(dist2[int(dest["r0"]), int(dest["c0"])]) >= 0 if bool(feasible2[int(dest["r0"]), int(dest["c0"])]) else False
    if bool(reached) != bool(dest_ok):
        return None
    if not np.array_equal(main(grid), out):
        return None
    if not reached and barrier_info["critical_idx"] is None:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
