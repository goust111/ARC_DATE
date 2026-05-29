# Combo variant of 332f06d7 (index 30)
# combo_spec: D3/3+D6/3+D19/1
# Name: 反向擦除+碰撞消除+始终卡住

from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label as scipy_label

_ARC_HELPER_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(_ARC_HELPER_DIR) not in sys.path:
    sys.path.append(str(_ARC_HELPER_DIR))

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo D3/3 + D6/3 + D19/1
==============================================

a) Concrete behavior and trace
- Source task: a black n x n square slides through a blue river toward a same-sized red square.
  If the n x n placement graph reaches the red top-left placement, black covers red; otherwise
  black stops at the farthest reachable placement and the start cell turns blue.
- This combo changes three stages at once.
  1. D19/1 removes reachable training instances entirely, so every generated sample contains a
     real chokepoint narrower than n and the mover always halts upstream.
  2. D3/3 inverts the trail rendering: every departed placement is repainted to background green,
     not river blue, so the output shows a carved-away travel range.
  3. D6/3 would erase mover and target together on contact, but D19 suppresses that branch in all
     generated examples. The solver still supports it for robustness, yet the dataset's visible rule
     is uniform: move until blocked, erase the traversed river behind the block.
- Small trace with n=2:
  input: black 2x2 at one end, red 2x2 downstream, a blue river with one 1-pixel neck.
  source output: start becomes blue, black moves to the last wide-enough 2x2 placement, red stays.
  combo output: start and every intermediate 2x2 placement become green; only the final blocked
  black square remains, and red still stays because the destination is unreachable.

b) Complexity calibration
- "Always stuck" is only physically possible for n >= 2, so the generator forbids n=1.
- The reverse-erasure effect is easy to miss if the mover advances by only one trivial step or if
  only the original black square changes. The generator therefore requires genuine motion and also
  checks that at least n former river pixels become background in the output.
- A single chokepoint keeps the scene legible. Multiple gates would make the erased range harder to
  interpret and introduce unnecessary ambiguity about which bottleneck matters.

c) Implicit assumption audit
- Solid-square detection still assumes the two salient non-background objects are monochromatic and
  axis-aligned; the generator preserves that.
- The solver still identifies mover/destination by the original palette convention after grouping by
  size. This is acceptable because the generator keeps exactly one black and one red square of the
  chosen size, so no new invisible tie-breaker is introduced.
- The source solver only needed the final target. D3 breaks that assumption because output painting
  now depends on the full path, so this combo reconstructs a predecessor chain from BFS.
- D19 lets main() simplify the intended branch to "farthest reachable placement", but D6 adds a
  reachable-contact behavior. The combined solver keeps both branches coherent: erase departed
  placements first, then either leave the final blocked mover or erase the destination on contact.

d) Placement legality
- The generator builds the river first, then carves exactly one chokepoint while preserving pixel
  connectivity of the blue river. This keeps the scene visually continuous but splits the n x n
  placement graph into reachable and unreachable regions.
- It verifies: mover start placement feasible, destination placement feasible, destination
  unreachable, final blocked placement distinct from start, and main(input) equals the intended
  combo output.

e) Visual observability
- Across a few examples, the red target is always still present, the black square always stops at a
  visible neck, and a green scar marks exactly where the mover could travel. That is enough for a
  human to infer "the river gets erased behind the sliding block until the first too-narrow gate."
- No rule depends on hidden ordering among multiple symmetric objects; there is just one mover, one
  target, and one connected river.

f) Visual cleanliness
- Only three output-side effects remain: untouched blue downstream river, green erased upstream
  range, and the final blocked black square. Because the target is always unreachable in generated
  samples, there is no overlap between the erased trail and the red square.

g) New solver assumptions enforced by the generator
- The reconstructed path assumes the chosen target placement is reachable from the start in the
  feasible-placement graph; the generator checks that before accepting a sample.
- The farthest reachable placement is intended to represent the blocker-adjacent halt point; the
  generator keeps a single winding corridor and a single valid chokepoint so that interpretation is
  stable and visible.
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2 = H - n + 1
    w2 = W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r : r + n, c : c + n].all())
    return feasible


def _detect_scene(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(
        Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0]
    )

    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        bbox = comp["bbox"]
        if bbox_h(bbox) != bbox_w(bbox):
            continue
        patch = g[int(bbox[0]) : int(bbox[2]) + 1, int(bbox[1]) : int(bbox[3]) + 1]
        if not bool((patch == col).all()):
            continue
        squares.append({"col": col, "r0": int(bbox[0]), "c0": int(bbox[1]), "n": int(bbox_h(bbox))})
    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for square in squares:
        by_n[int(square["n"])].append(square)
    nside = max(by_n.items(), key=lambda kv: (len(kv[1]), kv[0]))[0]
    cand = by_n[int(nside)]
    assert len(cand) >= 2

    cand.sort(key=lambda s: int(s["col"]))
    mover = cand[0]
    dest = cand[1]

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) == int(bg):
            continue
        if int(col) in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _build_feasible(trav, n)
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2
    assert bool(feasible[sr, sc])

    return {
        "grid": g,
        "H": H,
        "W": W,
        "bg": bg,
        "river": int(river),
        "mover_col": int(mover["col"]),
        "dest_col": int(dest["col"]),
        "n": n,
        "sr": sr,
        "sc": sc,
        "dr": dr,
        "dc": dc,
        "feasible": feasible,
    }


def _bfs_with_parent(feasible: np.ndarray, sr: int, sc: int) -> tuple[np.ndarray, np.ndarray]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    prev = -np.ones((h2, w2, 2), dtype=int)
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                prev[rr, cc] = (r, c)
                q.append((rr, cc))
    return dist, prev


def _farthest_reachable(dist: np.ndarray, sr: int, sc: int) -> tuple[int, int]:
    best = -1
    tr, tc = sr, sc
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc


def _trace_path(prev: np.ndarray, sr: int, sc: int, tr: int, tc: int) -> list[tuple[int, int]]:
    path = [(int(tr), int(tc))]
    while path[-1] != (int(sr), int(sc)):
        pr, pc = prev[path[-1][0], path[-1][1]]
        assert pr >= 0 and pc >= 0
        path.append((int(pr), int(pc)))
    path.reverse()
    return path


def _plan_motion(grid: np.ndarray) -> dict:
    scene = _detect_scene(grid)
    dist, prev = _bfs_with_parent(scene["feasible"], scene["sr"], scene["sc"])
    h2, w2 = scene["feasible"].shape
    dr, dc = scene["dr"], scene["dc"]

    dest_reached = (
        0 <= dr < h2
        and 0 <= dc < w2
        and bool(scene["feasible"][dr, dc])
        and int(dist[dr, dc]) >= 0
    )
    if dest_reached:
        tr, tc = dr, dc
    else:
        tr, tc = _farthest_reachable(dist, scene["sr"], scene["sc"])
    assert int(dist[tr, tc]) >= 0

    scene["dist"] = dist
    scene["prev"] = prev
    scene["dest_reached"] = bool(dest_reached)
    scene["tr"] = int(tr)
    scene["tc"] = int(tc)
    scene["path"] = _trace_path(prev, scene["sr"], scene["sc"], int(tr), int(tc))
    return scene


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Erase every departed n x n placement to background (D3/3).
    - If the destination is somehow reachable, mover and target annihilate to river (D6/3).
    - Generated samples always choose the blocked branch, so the mover normally stops at the
      farthest reachable placement before the chokepoint (D19/1).
    """
    scene = _plan_motion(grid)
    out = scene["grid"].copy()
    n = int(scene["n"])

    for r, c in scene["path"][:-1]:
        out[r : r + n, c : c + n] = int(scene["bg"])

    if bool(scene["dest_reached"]):
        out[scene["dr"] : scene["dr"] + n, scene["dc"] : scene["dc"] + n] = int(scene["river"])
    else:
        out[scene["tr"] : scene["tr"] + n, scene["tc"] : scene["tc"] + n] = int(scene["mover_col"])
    return out


def _legacy_stuck_output(grid: np.ndarray, tr: int, tc: int, n: int, mover_r: int, mover_c: int) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    out[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    dist = -np.ones(feasible.shape, dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < feasible.shape[0] and 0 <= cc < feasible.shape[1] and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    total = rows * cols
    best_path = None
    best_len = 0
    for _attempt in range(80):
        visited = {start}
        path = [start]
        cur = start
        while True:
            if cur == end and len(path) >= total * 0.7:
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
                break
            neighbors = []
            for dr, dc in _DIR4:
                nr, nc = cur[0] + dr, cur[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited:
                    neighbors.append((nr, nc))
            if not neighbors:
                break
            valid = []
            for nb in neighbors:
                if nb == end:
                    valid.append(nb)
                    continue
                test_visited = visited | {nb}
                seen = {end}
                q = deque([end])
                while q:
                    cr, cc = q.popleft()
                    for dr2, dc2 in _DIR4:
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in _DIR4:
                    ar, ac = nb[0] + dr2, nb[1] + dc2
                    if (
                        0 <= ar < rows
                        and 0 <= ac < cols
                        and (ar, ac) not in test_visited
                        and (ar, ac) in seen
                    ):
                        valid.append(nb)
                        break
            if not valid:
                break
            random.shuffle(valid)
            nxt = valid[0]
            visited.add(nxt)
            path.append(nxt)
            cur = nxt
        if best_path and best_len >= total * 0.85:
            break
    return best_path


def _render_channel(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    n: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr : pr + n, pc : pc + n] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
            grid[pr1 : pr1 + n, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
            grid[r_lo:r_hi, pc1 : pc1 + n] = _RIVER
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    n: int,
    direction: str,
) -> None:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j
    narrow_w = random.randint(1, n - 1)
    remove = n - narrow_w
    side = random.randint(0, 1)

    if direction == "h":
        c_lo = min(pc1, pc2) + n
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + n
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if c < 0 or c >= W:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
    else:
        r_lo = min(pr1, pr2) + n
        r_hi = max(pr1, pr2)
        if r_hi <= r_lo:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + n
        hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
        total_len = 2 * remove + max(1, hold_len)
        r_mid = (r_lo + r_hi) // 2
        r_start = max(r_lo, r_mid - total_len // 2)
        for step in range(total_len):
            r = r_start + step
            if r < 0 or r >= H:
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < W:
                    grid[r, cc] = _BG


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = n + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, n, coarse_rows)
    offsets_c = _compute_offsets(G, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = set()
    for c in range(coarse_cols):
        edge_cells.add((0, c))
        edge_cells.add((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.add((r, 0))
        edge_cells.add((r, coarse_cols - 1))
    edge_cells = list(edge_cells)
    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 5:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    def _extend_to_edge(pr: int, pc: int) -> tuple[int, int] | None:
        candidates = []
        for direction in ("top", "bottom", "left", "right"):
            if direction == "top":
                dist = pr
                if dist == 0:
                    candidates.append((0, pc, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n) if c < W):
                    candidates.append((0, pc, dist))
            elif direction == "bottom":
                dist = H - (pr + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(
                    grid[r, c] in (_BG, _RIVER)
                    for r in range(pr + n, H)
                    for c in range(pc, pc + n)
                    if c < W
                ):
                    candidates.append((H - n, pc, dist))
            elif direction == "left":
                dist = pc
                if dist == 0:
                    candidates.append((pr, 0, 0))
                    continue
                if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc) if r < H):
                    candidates.append((pr, 0, dist))
            else:
                dist = W - (pc + n)
                if dist == 0:
                    candidates.append((pr, pc, 0))
                    continue
                if all(
                    grid[r, c] in (_BG, _RIVER)
                    for r in range(pr, pr + n)
                    for c in range(pc + n, W)
                    if r < H
                ):
                    candidates.append((pr, W - n, dist))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[2])
        return candidates[0][:2]

    result = _extend_to_edge(dest_r, dest_c)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not (dest_r == 0 or dest_r + n == H or dest_c == 0 or dest_c + n == W):
        return None

    touching = mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W
    if touching:
        if mover_r == 0:
            grid[0, mover_c : mover_c + n] = _RIVER
            mover_r = 1
        elif mover_r + n == H:
            grid[H - 1, mover_c : mover_c + n] = _RIVER
            mover_r = H - n - 1
        elif mover_c == 0:
            grid[mover_r : mover_r + n, 0] = _RIVER
            mover_c = 1
        else:
            grid[mover_r : mover_r + n, W - 1] = _RIVER
            mover_c = W - n - 1
    if mover_r == 0 or mover_r + n == H or mover_c == 0 or mover_c + n == W:
        return None
    if min(mover_r, H - (mover_r + n), mover_c, W - (mover_c + n)) > 2:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _RIVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _RIVER

    seg_lo = max(1, len(walk) // 3)
    seg_hi = len(walk) - 2
    if seg_lo >= seg_hi:
        return None
    candidates = list(range(seg_lo, seg_hi))
    random.shuffle(candidates)
    choked = False
    for seg_idx in candidates:
        pr1, pc1 = pixel_pos[seg_idx]
        pr2, pc2 = pixel_pos[seg_idx + 1]
        direction = "h" if pr1 == pr2 else "v"
        trial = grid.copy()
        _add_chokepoint(trial, (pr1, pc1), (pr2, pc2), n, direction)
        if _river_connected(trial):
            grid = trial
            choked = True
            break
    if not choked:
        return None

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _build_feasible(trav, n)
    if not bool(feasible[mover_r, mover_c]):
        return None
    if not bool(feasible[dest_r, dest_c]):
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    if int(dist[dest_r, dest_c]) >= 0:
        return None

    tr, tc = _farthest_reachable(dist, mover_r, mover_c)
    if (tr, tc) == (mover_r, mover_c):
        return None

    variant_output = main(grid)
    legacy_output = _legacy_stuck_output(grid, tr, tc, n, mover_r, mover_c)
    if np.array_equal(variant_output, legacy_output):
        return None

    erased_river = int(np.count_nonzero((grid == _RIVER) & (variant_output == _BG)))
    if erased_river < max(1, n):
        return None
    if not bool((variant_output[dest_r : dest_r + n, dest_c : dest_c + n] == _DEST).all()):
        return None

    scene = _plan_motion(grid)
    if bool(scene["dest_reached"]):
        return None
    if len(scene["path"]) <= 1:
        return None
    if (scene["tr"], scene["tc"]) != (tr, tc):
        return None
    if not np.array_equal(variant_output, main(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
