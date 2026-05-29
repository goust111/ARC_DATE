# Combo variant of 332f06d7 (index 9)
# combo_spec: D4/2+D6/1
# Name: 多色方块+合并融合

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components

"""
Design analysis for combo D4/2 + D6/1
=====================================

This fusion keeps the original "slide an n×n footprint through a river" core,
but changes both the mover representation and the reachable end-state.

a) Concrete behavior and trace
- Original seed:
  - mover = solid n×n square
  - destination = solid n×n square on the border
  - reachable => mover replaces destination with one color
  - stuck => mover stops at farthest feasible placement
- Fused variant:
  - mover = multicolor n×n pattern, visually asymmetric
  - destination = solid n×n square touching exactly one border
  - movement legality is still defined on the whole n×n footprint
  - reachable => destination keeps its own n×n patch on the border side, and
    the mover patch is attached on the inward side, forming a 2n×n or n×2n
    composite rectangle
  - stuck => mover patch is translated intact to the farthest reachable slot
- Small example for n=2, destination on the top edge:
  - input mover:
      [[0, 4],
       [5, 6]]
  - input destination:
      [[2, 2],
       [2, 2]]
  - original reachable output would place only the mover in the destination
    2×2 slot
  - fused reachable output is:
      [[2, 2],
       [2, 2],
       [0, 4],
       [5, 6]]
    attached to the same top border

b) Complexity calibration
- n is restricted to 2..4 so the multicolor structure is always visible.
- The mover pattern comes from a tiny template family, using at most 4 colors,
  which keeps the rule obvious without creating too many one-off cases.
- Reachable behavior has one extra branch only: fusion orientation is determined
  by the visible border side of the destination.

c) Implicit-assumption audit against the original main()
- Original assumption: the mover is a monochrome connected component.
  Broken by D4/2. The solver must now locate a full multicolor n×n window.
- Original assumption: sorting square colors identifies mover vs destination.
  Broken by the fusion. Roles are now inferred geometrically:
  the destination is the unique same-sized solid square touching exactly one
  border, and the mover is the unique interior multicolor n×n patch.
- Original assumption: reachable output is written by a single scalar fill over
  the destination slot. Broken by D6/1. Reachable output now paints two
  adjacent n×n patches and preserves both input patterns.
- Original assumption: destination may touch a corner. That would make fusion
  direction ambiguous, so the generator must forbid it.

d) Placement legality
- The generator builds the river first, then extends the destination to exactly
  one edge, never a corner.
- The inward half of the fused rectangle is always inside the grid because the
  destination touches exactly one edge.
- The mover remains near an edge but never touching it, so the two object roles
  are spatially distinct and non-overlapping in the input.

e) Visual observability
- Across 2-4 examples, a human can see the same multicolor motif reappear at
  the stop position in stuck cases and attached to the target in reachable
  cases.
- The fusion direction is readable from the destination's border contact, not
  from invisible metadata like color ids.

f) Visual cleanliness
- The output composite is a single compact doubled rectangle, not scattered
  markers.
- The mover never contains background / river / destination colors, so the
  object boundaries stay crisp.

g) Color-change inferability
- No recoloring rule is hidden. The mover keeps its internal colors exactly, and
  the destination keeps its own solid patch. The only new inference is spatial:
  keep the destination on the border side, attach the mover on the inner side.

h) New solver assumptions introduced here and enforced by the generator
- Exactly one solid border-touching n×n square exists for the chosen n.
- Exactly one interior multicolor n×n patch of that same size exists.
- The mover patch contains at least 2 colors and excludes background, river,
  and destination colors.
- The river is the most common non-background color.
- The destination touches exactly one side, so the fusion direction is unique.

i) Review of the modified main() logic
- `_find_destination_square()` assumes the target is monochrome, square,
  border-touching, and size 2..4. The generator constructs exactly that.
- `_find_mover_patch()` assumes the mover can be found as an interior n×n
  window with at least 2 colors and without bg/river/dest colors. The template
  generator enforces this and keeps the mover unique.
- `_build_feasible()` assumes the mover footprint can traverse any all-non-bg
  n×n window. This remains true because river, mover, and destination are all
  traversable material in the input.
- `_paint_fusion()` assumes the destination side uniquely determines the inward
  attachment direction. Corner targets are therefore disallowed upstream.
"""

_BG = 3
_RIVER = 1
_DEST = 2
_MOVER_COLORS = [0, 4, 5, 6, 7, 8, 9]
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

_PATTERN_TEMPLATES = {
    2: np.array([[0, 1], [2, 3]], dtype=int),
    3: np.array([[0, 1, 2], [3, 0, 1], [2, 3, 0]], dtype=int),
    4: np.array(
        [
            [0, 1, 2, 3],
            [2, 0, 3, 1],
            [3, 1, 0, 2],
            [1, 3, 2, 0],
        ],
        dtype=int,
    ),
}


def _touching_sides(r0: int, c0: int, n: int, H: int, W: int) -> list[str]:
    sides: list[str] = []
    if r0 == 0:
        sides.append("top")
    if r0 + n == H:
        sides.append("bottom")
    if c0 == 0:
        sides.append("left")
    if c0 + n == W:
        sides.append("right")
    return sides


def _bfs_placement(feasible: np.ndarray, sr: int, sc: int) -> np.ndarray:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    dist[sr, sc] = 0
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and feasible[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = dist[r, c] + 1
                q.append((rr, cc))
    return dist


def _infer_river(g: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, g.flatten()))
    for col, _ct in counts.most_common():
        if int(col) != int(bg):
            return int(col)
    raise AssertionError("river color not found")


def _find_destination_square(g: np.ndarray, bg: int, river: int) -> dict:
    H, W = g.shape
    candidates: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col == river:
            continue
        b = comp["bbox"]
        n = int(bbox_h(b))
        if n != int(bbox_w(b)) or n < 2 or n > 4:
            continue
        r0, c0, r1, c1 = map(int, b)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if not bool((patch == col).all()):
            continue
        sides = _touching_sides(r0, c0, n, H, W)
        if len(sides) != 1:
            continue
        candidates.append({"col": col, "r0": r0, "c0": c0, "n": n, "side": sides[0], "patch": patch.copy()})
    assert candidates
    candidates.sort(key=lambda s: (-int(s["n"]), int(s["r0"]), int(s["c0"])))
    return candidates[0]


def _find_mover_patch(g: np.ndarray, bg: int, river: int, dest: dict) -> dict:
    H, W = g.shape
    n = int(dest["n"])
    dest_col = int(dest["col"])
    candidates: list[dict] = []
    for r in range(H - n + 1):
        for c in range(W - n + 1):
            patch = g[r : r + n, c : c + n]
            if bool((patch == bg).any()) or bool((patch == river).any()) or bool((patch == dest_col).any()):
                continue
            colors = {int(x) for x in patch.flatten()}
            if len(colors) < 2:
                continue
            edge_dist = min(r, c, H - (r + n), W - (c + n))
            if edge_dist <= 0:
                continue
            candidates.append({"r0": r, "c0": c, "patch": patch.copy(), "edge_dist": edge_dist})
    assert candidates
    candidates.sort(key=lambda s: (int(s["edge_dist"]), int(s["r0"]), int(s["c0"])))
    return candidates[0]


def _build_feasible(trav: np.ndarray, n: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - n + 1, W - n + 1
    assert h2 > 0 and w2 > 0
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    return feasible


def _choose_terminal(
    dist: np.ndarray, feasible: np.ndarray, sr: int, sc: int, dr: int, dc: int
) -> tuple[int, int, bool]:
    h2, w2 = feasible.shape
    dest_ok = 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0
    if dest_ok:
        return dr, dc, True

    best = -1
    tr, tc = sr, sc
    for r in range(h2):
        for c in range(w2):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = r, c
    return tr, tc, False


def _paint_fusion(out: np.ndarray, dest: dict, mover_patch: np.ndarray) -> None:
    n = int(dest["n"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    dest_patch = np.array(dest["patch"], copy=False)
    side = str(dest["side"])

    if side == "top":
        out[dr : dr + n, dc : dc + n] = dest_patch
        out[dr + n : dr + 2 * n, dc : dc + n] = mover_patch
    elif side == "bottom":
        out[dr - n : dr, dc : dc + n] = mover_patch
        out[dr : dr + n, dc : dc + n] = dest_patch
    elif side == "left":
        out[dr : dr + n, dc : dc + n] = dest_patch
        out[dr : dr + n, dc + n : dc + 2 * n] = mover_patch
    elif side == "right":
        out[dr : dr + n, dc - n : dc] = mover_patch
        out[dr : dr + n, dc : dc + n] = dest_patch
    else:
        raise AssertionError(f"unexpected side: {side}")


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])
    river = _infer_river(g, bg)
    dest = _find_destination_square(g, bg, river)
    mover = _find_mover_patch(g, bg, river, dest)

    n = int(dest["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])
    mover_patch = np.array(mover["patch"], copy=True)

    trav = g != bg
    feasible = _build_feasible(trav, n)
    assert bool(feasible[sr, sc])

    dist = _bfs_placement(feasible, sr, sc)
    tr, tc, reached = _choose_terminal(dist, feasible, sr, sc, dr, dc)

    out = g.copy()
    out[sr : sr + n, sc : sc + n] = river
    if reached:
        _paint_fusion(out, dest, mover_patch)
    else:
        out[tr : tr + n, tc : tc + n] = mover_patch

    meta = {
        "n": n,
        "bg": bg,
        "river": river,
        "mover": mover,
        "dest": dest,
        "reached": reached,
        "terminal": (tr, tc),
        "mover_patch": mover_patch,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a multicolor n×n patch through the river.

    Stuck branch: identical to the D4-style transport rule, preserving the
    mover patch.
    Reachable branch: destination and mover fuse into a doubled rectangle,
    keeping the destination on the border side and attaching the mover on the
    inward side.
    """
    out, _meta = _solve(np.array(grid, dtype=int))
    return out


def _hamiltonian_walk(rows: int, cols: int, start: tuple[int, int], end: tuple[int, int]):
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
                    if 0 <= ar < rows and 0 <= ac < cols and (ar, ac) not in test_visited and (ar, ac) in seen:
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


def _render_channel(grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]):
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


def _add_chokepoint(grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str):
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
        total_len = 2 * remove + hold_len
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
        total_len = 2 * remove + hold_len
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
    mask = grid != _BG
    if not bool(mask.any()):
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(G: int, n: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_single_edge(grid: np.ndarray, pr: int, pc: int, n: int):
    H, W = grid.shape
    candidates = []

    if 0 < pc < W - n:
        clear_top = all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n))
        if clear_top:
            candidates.append((pr, random.random(), 0, pc, "top"))
        clear_bottom = all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, H) for c in range(pc, pc + n))
        if clear_bottom:
            candidates.append((H - (pr + n), random.random(), H - n, pc, "bottom"))

    if 0 < pr < H - n:
        clear_left = all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc))
        if clear_left:
            candidates.append((pc, random.random(), pr, 0, "left"))
        clear_right = all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, W))
        if clear_right:
            candidates.append((W - (pc + n), random.random(), pr, W - n, "right"))

    if not candidates:
        return None

    dist, _jitter, r0, c0, side = min(candidates, key=lambda item: (item[0], item[1]))
    if side in ("top", "bottom"):
        r_lo, r_hi = min(pr, r0), max(pr, r0) + n
        grid[r_lo:r_hi, pc : pc + n] = _RIVER
    else:
        c_lo, c_hi = min(pc, c0), max(pc, c0) + n
        grid[pr : pr + n, c_lo:c_hi] = _RIVER

    sides = _touching_sides(int(r0), int(c0), n, H, W)
    if len(sides) != 1 or sides[0] != side:
        return None
    return int(r0), int(c0), side


def _make_mover_patch(n: int) -> np.ndarray:
    palette = random.sample(_MOVER_COLORS, 4)
    return np.array([[palette[idx] for idx in row] for row in _PATTERN_TEMPLATES[n]], dtype=int)


def _transport_only_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.array(grid, copy=True)
    n = int(meta["n"])
    sr, sc = int(meta["mover"]["r0"]), int(meta["mover"]["c0"])
    tr, tc = map(int, meta["terminal"])
    out[sr : sr + n, sc : sc + n] = int(meta["river"])
    out[tr : tr + n, tc : tc + n] = np.array(meta["mover_patch"], copy=False)
    return out


def _gen_attempt():
    reached = random.random() < 0.5
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
    if walk is None or len(walk) < 4:
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

    result = _extend_to_single_edge(grid, dest_r, dest_c, n)
    if result is None:
        return None
    dest_r, dest_c, dest_side = result

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

    if not reached:
        num_choke = random.randint(1, min(3, len(walk) - 2))
        min_seg = max(1, len(walk) // 4)
        available = list(range(min_seg, len(walk) - 1))
        random.shuffle(available)
        for seg_idx in available[:num_choke]:
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[min(seg_idx + 1, len(pixel_pos) - 1)]
            direction = "h" if pr1 == pr2 else "v"
            _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)
        if not _river_connected(grid):
            return None

    mover_patch = _make_mover_patch(n)
    grid[mover_r : mover_r + n, mover_c : mover_c + n] = mover_patch
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    sides = _touching_sides(dest_r, dest_c, n, H, W)
    if sides != [dest_side]:
        return None

    out, meta = _solve(grid)
    if bool(meta["reached"]) != bool(reached):
        return None

    if not np.array_equal(main(grid), out):
        return None

    if reached:
        side = str(meta["dest"]["side"])
        if side in ("top", "bottom"):
            if out.shape[0] < 2 * n:
                return None
        else:
            if out.shape[1] < 2 * n:
                return None
        if np.array_equal(out, _transport_only_output(grid, meta)):
            return None
        dest_pixels = np.count_nonzero(out == _DEST)
        mover_pixels = sum(np.count_nonzero(out == col) for col in np.unique(mover_patch))
        if dest_pixels < n * n or mover_pixels < n * n:
            return None

    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
