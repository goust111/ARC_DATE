# Combo variant of 332f06d7 (index 17)
# combo_spec: D7/2+D15/0
# Name: 网格缩放+固定转折模式

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for combo variant D7/2 + D15/0
===============================================

a) Concrete behavior trace
--------------------------
- Kept from the source task:
  - Input is still a square 10-20 grid.
  - Background is green, river is blue, mover is a solid black n×n square,
    destination is a solid red n×n square of the same size.
  - Motion is still computed on the feasible top-left placement grid:
    a placement is legal iff the full n×n window lies in river∪mover∪dest.
  - Reachable case: mover reaches the red square.
  - Stuck case: a later corridor segment is narrowed below n, so BFS stops at
    the farthest reachable placement before the gate.

- D15/0 change in this combo:
  - The river is no longer a random winding walk. It is rendered from a fixed
    clockwise spiral path in placement space. Traversing from the outer start
    toward the inner destination therefore yields the same visible turning
    pattern sample after sample: right, right, right, ...
  - This does not add a second hidden movement rule; it changes the geometry of
    the single traversable channel that BFS runs on.

- D7/2 change in this combo:
  - Output is no longer the original-sized grid with a moved n×n block.
  - Instead, solve the sliding problem first, reconstruct the actual shortest
    placement path from mover start to final stop, keep only start/turns/end,
    compress row and column coordinates independently, and draw that compressed
    orthogonal skeleton.
  - Each turn vertex maps to one output pixel. The final stop pixel is black.

- Small trace:
  - Suppose the actual feasible top-left path prefix is
      [(1,1),(1,2),(1,3),(2,3),(3,3),(3,2),(3,1),(4,1)].
  - Original task output would repaint the full n×n mover at (4,1).
  - Combo output keeps turning vertices
      [(1,1),(1,3),(3,3),(3,1),(4,1)],
    compresses rows {1,3,4}-> {0,1,2} and cols {1,3}->{0,1},
    and renders a small polyline ending in a black pixel.
  - Because the underlying path is a clockwise spiral, these compressed turns
    form a regular nested pattern rather than an arbitrary zig-zag.

b) Complexity calibration
-------------------------
- If we only changed the output size, this would collapse into plain D7/2.
  The generator therefore requires:
  - at least 4 turning vertices in the realized motion path,
  - all turns to have the same orientation sign,
  - output shape strictly smaller than input.
- If we only changed the path generator to a spiral but kept full-size output,
  the D7 effect would be visually weak. The compressed skeleton makes the fixed
  turning rule obvious because the spiral becomes a compact regular glyph.
- The freedom remains small and ARC-readable:
  - one fixed path family (clockwise spiral),
  - two reachability branches (reach / choke),
  - no extra legend colors or per-sample symbolic remapping.

c) Implicit assumptions review against main()
---------------------------------------------
- Square-role detection:
  - `main()` still assumes there are at least two same-sized solid squares.
  - Generator preserves exactly that structure.
- River inference:
  - `main()` chooses the most common non-bg, non-square color as river.
  - Generator uses only one non-role color in the playfield, so this remains
    safe.
- Feasible-window BFS:
  - The original code assumes the movement problem is fully captured by the
    feasible-placement graph. That still holds here because D15 only changes
    corridor geometry, not movement semantics.
- Parent reconstruction:
  - New output logic assumes the final stop is connected to the start by a
    unique BFS parent chain. This is true for any BFS tree; the generator also
    keeps the spiral path simple so the shortest path is visually unambiguous.
- Turn compression:
  - `_compress_turn_coords()` assumes the motion path is orthogonal and uses a
    modest number of distinct turn rows/cols. Spiral paths satisfy this.
- New combo-specific assumption:
  - Samples should expose the fixed-turn property. The solver itself does not
    numerically depend on "spiral"; instead the generator enforces it and the
    compressed output makes it visible.

d) Placement legality
---------------------
- The generator works in placement-coordinate space first:
  - choose square size n,
  - choose a spiral box,
  - build a spiral path of legal top-left placements inset from the border,
  - paint each placement as an n×n river window.
- This guarantees no placement goes out of bounds.
- Mover sits at the first spiral placement, destination at the last.
- In stuck cases, the gate is carved on the entering face of a later spiral
  step, preserving the upstream cell while making the downstream cell infeasible.
- Because the path is simple and one-cell-spaced in placement space, river arms
  do not accidentally merge into ambiguous branches.

e) Visual observability
-----------------------
- From a few examples, a human can see:
  - the input river is always a rectangular inward spiral,
  - the output is always a much smaller blue path skeleton,
  - the skeleton preserves only the turn structure and black final stop.
- No invisible rule uses color ids, ordering by numeric value, or hidden state.
  The only numeric color ordering left is internal mover/dest role selection,
  which is supported by their stable visual roles across all examples.

f) Visual cleanliness
---------------------
- Input contains one dominant coherent blue structure, not a noisy maze.
- Output contains only background, a 1-pixel blue polyline, and one black
  endpoint pixel.
- There is no overlap between thick source objects and thin target marks because
  the output is a fresh scaled-down grid.

g) Color-change inference
-------------------------
- No new colors are introduced.
- Colors keep their source-task meanings, so the fusion challenge is geometric:
  infer "spiral channel" from the input and "compressed turn skeleton" from the
  output.

Post-change audit of new assumptions
------------------------------------
- `_simplify_turns()` assumes the reconstructed path is at least length 2 for a
  meaningful output; generator enforces many-step paths.
- `_render_compressed_path()` assumes compressed consecutive vertices never
  collapse into the same pixel; generator enforces actual turns and nonzero path
  length.
- `_all_turns_same_orientation()` is only a generator-side validation metric,
  not a solver dependency; this avoids making `main()` brittle while still
  verifying that D15 behavior is present in produced samples.
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_DIR4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _detect_roles(g: np.ndarray) -> tuple[int, dict, dict, int]:
    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append({"col": int(col), "r0": int(b[0]), "c0": int(b[1]), "n": int(bbox_h(b))})

    assert len(squares) >= 2

    by_n: dict[int, list[dict]] = defaultdict(list)
    for sq in squares:
        by_n[int(sq["n"])].append(sq)
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
    return bg, mover, dest, river


def _bfs_with_parent(
    feasible: np.ndarray, sr: int, sc: int
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int] | None]]:
    h2, w2 = feasible.shape
    dist = -np.ones((h2, w2), dtype=int)
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    q = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in _DIR4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h2 and 0 <= cc < w2 and bool(feasible[rr, cc]) and int(dist[rr, cc]) == -1:
                dist[rr, cc] = int(dist[r, c] + 1)
                parent[(rr, cc)] = (r, c)
                q.append((rr, cc))
    return dist, parent


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int] | None], end: tuple[int, int]
) -> list[tuple[int, int]]:
    path = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _simplify_turns(path: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return list(path)

    out = [path[0]]
    for i in range(1, len(path) - 1):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        nr, nc = path[i + 1]
        d1 = (cr - pr, cc - pc)
        d2 = (nr - cr, nc - cc)
        if d1 != d2:
            out.append((cr, cc))
    out.append(path[-1])
    return out


def _compress_turn_coords(turns: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = sorted({r for r, _c in turns})
    cols = sorted({c for _r, c in turns})
    r_map = {r: i for i, r in enumerate(rows)}
    c_map = {c: i for i, c in enumerate(cols)}
    return [(r_map[r], c_map[c]) for r, c in turns]


def _render_compressed_path(bg: int, river: int, mover_col: int, turns: list[tuple[int, int]]) -> np.ndarray:
    pts = _compress_turn_coords(turns)
    h = max(r for r, _c in pts) + 1
    w = max(c for _r, c in pts) + 1
    out = np.full((h, w), bg, dtype=int)
    for (r1, c1), (r2, c2) in zip(pts, pts[1:]):
        if r1 == r2:
            c_lo, c_hi = sorted((c1, c2))
            out[r1, c_lo : c_hi + 1] = river
        else:
            r_lo, r_hi = sorted((r1, r2))
            out[r_lo : r_hi + 1, c1] = river
    if len(pts) == 1:
        out[pts[0]] = river
    out[pts[-1]] = mover_col
    return out


def _solve_core(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, mover, dest, river = _detect_roles(g)

    n = int(mover["n"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    trav = (g == int(river)) | (g == int(mover["col"])) | (g == int(dest["col"]))
    h2 = int(h - n + 1)
    w2 = int(w - n + 1)
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    dist, parent = _bfs_with_parent(feasible, sr, sc)

    if 0 <= dr < h2 and 0 <= dc < w2 and bool(feasible[dr, dc]) and int(dist[dr, dc]) >= 0:
        tr, tc = dr, dc
        reached = True
    else:
        reached = False
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if int(dist[r, c]) > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    path = _reconstruct_path(parent, (tr, tc))
    turns = _simplify_turns(path)
    return {
        "bg": bg,
        "river": river,
        "mover_col": int(mover["col"]),
        "dest_col": int(dest["col"]),
        "n": n,
        "start": (sr, sc),
        "dest": (dr, dc),
        "target": (tr, tc),
        "reached": reached,
        "feasible": feasible,
        "dist": dist,
        "path": path,
        "turns": turns,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    1. Solve the original sliding-square task on feasible n×n placements.
    2. Reconstruct the mover's actual path to its final stop.
    3. Keep only start/end/turn points.
    4. Compress those turn coordinates into a smaller orthogonal diagram.
    5. Draw the compressed path in river color and overwrite the endpoint in black.

    The generator supplies fixed-turn spiral inputs, so the compressed output is a
    regular nested skeleton rather than an arbitrary winding sketch.
    """
    info = _solve_core(np.array(grid, dtype=int))
    return _render_compressed_path(info["bg"], info["river"], info["mover_col"], info["turns"])


def _spiral_path(rows: int, cols: int, top: int = 0, left: int = 0) -> list[tuple[int, int]]:
    used = np.zeros((rows, cols), dtype=bool)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    path: list[tuple[int, int]] = []
    d = 0
    r = 0
    c = 0
    while True:
        path.append((top + r, left + c))
        used[r, c] = True
        moved = False
        for turn in range(4):
            nd = (d + turn) % 4
            dr, dc = dirs[nd]
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or bool(used[nr, nc]):
                continue
            blocked = False
            for adr, adc in dirs:
                ar, ac = nr + adr, nc + adc
                if 0 <= ar < rows and 0 <= ac < cols and bool(used[ar, ac]) and (ar, ac) != (r, c):
                    blocked = True
                    break
            if blocked:
                continue
            d = nd
            r, c = nr, nc
            moved = True
            break
        if not moved:
            break
    return path


def _compute_offsets(gsize: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = gsize - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


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
    grid: np.ndarray, pixel_pos_i: tuple[int, int], pixel_pos_j: tuple[int, int], n: int, direction: str
) -> None:
    h, w = grid.shape
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
            if not (0 <= c < w):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + n - 1 - k
                if 0 <= rr < h:
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
            if not (0 <= r < h):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                cc = pc1 + k if side == 0 else pc1 + n - 1 - k
                if 0 <= cc < w:
                    grid[r, cc] = _BG


def _turn_signs(path: list[tuple[int, int]]) -> list[int]:
    signs: list[int] = []
    for i in range(1, len(path) - 1):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        nr, nc = path[i + 1]
        d1 = (cr - pr, cc - pc)
        d2 = (nr - cr, nc - cc)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if cross > 0:
            signs.append(1)
        elif cross < 0:
            signs.append(-1)
    return signs


def _all_turns_same_orientation(path: list[tuple[int, int]]) -> bool:
    signs = _turn_signs(path)
    return len(signs) > 0 and len(set(signs)) == 1


def _turn_count(path: list[tuple[int, int]]) -> int:
    return max(0, len(_simplify_turns(path)) - 2)


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([1, 2, 3, 4]) if reached else random.choice([2, 3, 4])
    min_size = max(10, 4 * n + 4)
    if min_size > 20:
        return None
    gsize = random.randint(min_size, 20)
    h = w = gsize

    min_spacing = n + 1
    max_coarse = max(3, (gsize - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))

    offsets_r = _compute_offsets(gsize, n, coarse_dim)
    offsets_c = _compute_offsets(gsize, n, coarse_dim)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < min_spacing:
            return None
        if offsets_c[i + 1] - offsets_c[i] < min_spacing:
            return None

    walk = _spiral_path(coarse_dim, coarse_dim, top=0, left=0)
    if len(walk) < 8:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    if not reached:
        min_gate = max(2, len(walk) // 2)
        max_gate = len(walk) - 2
        if min_gate > max_gate:
            return None
        gate_idx = random.randint(min_gate, max_gate)
        pr1, pc1 = pixel_pos[gate_idx]
        pr2, pc2 = pixel_pos[gate_idx + 1]
        direction = "h" if pr1 == pr2 else "v"
        _add_chokepoint(grid, (pr1, pc1), (pr2, pc2), n, direction)

    grid[mover_r : mover_r + n, mover_c : mover_c + n] = _MOVER
    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST

    try:
        info = _solve_core(grid)
        out = main(grid)
    except Exception:
        return None

    if reached != bool(info["reached"]):
        return None
    if reached and info["target"] != (dest_r, dest_c):
        return None
    if not reached and info["target"] == (dest_r, dest_c):
        return None
    if _turn_count(info["path"]) < 2:
        return None
    if not _all_turns_same_orientation(info["path"]):
        return None
    signs = _turn_signs(info["path"])
    if not signs or any(sign != signs[0] for sign in signs):
        return None
    if signs[0] != -1:
        return None
    if out.shape[0] >= h or out.shape[1] >= w:
        return None
    if min(out.shape) < 2:
        return None
    if len(info["turns"]) < 4:
        return None
    if not np.any(out == _MOVER):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 500 attempts")
