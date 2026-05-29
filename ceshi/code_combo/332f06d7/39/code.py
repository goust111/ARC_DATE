# Combo variant of 332f06d7 (index 39)
# combo_spec: D18/0+D20/2
# Name: 边缘控制像素+起点标记指示

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task: input explicitly shows a black n x n mover on the blue river
#      plus a red n x n destination. The solver starts BFS directly from the black
#      square and can leave in any feasible direction.
#    - D20 side of the combo: the black mover disappears from the input. Instead,
#      one yellow border pixel marks the hidden source pocket:
#        top    (0, c)   -> hidden start at (1, c)
#        bottom (H-1, c) -> hidden start at (H-n-1, c)
#        left   (r, 0)   -> hidden start at (r, 1)
#        right  (r, W-1) -> hidden start at (r, W-n-1)
#    - D18 side of the combo: a red border singleton separately tells the solver
#      which first step the hidden mover must take:
#        top border red pixel    -> first step down
#        bottom border red pixel -> first step up
#        left border red pixel   -> first step right
#        right border red pixel  -> first step left
#    - Combo behavior on a small example:
#      1. Yellow pixel on the top edge says "the mover actually starts at row 1".
#      2. Around that hidden pocket, the river visibly has two exits: a main route
#         and a short dead-end stub.
#      3. A red singleton on the right edge says "the first move must be left".
#      4. The solver takes exactly that first move, removes the hidden start from
#         the feasible placement graph, and therefore cannot come back to switch
#         onto the other branch later.
#      5. Output draws the black square only at its final resting place; the yellow
#         marker and red control pixel remain as input annotations.
#
# b) Complexity calibration
#    - The combo adds exactly two one-pixel annotations:
#      one yellow source marker and one red direction marker.
#    - The destination remains the only n x n red square; the mover never appears
#      in the input. This keeps roles visually distinct without introducing a full
#      legend table.
#    - n is restricted to 2..4. This prevents the red destination from collapsing
#      into a 1 x 1 red pixel that would be ambiguous with the red direction marker.
#
# c) Implicit assumption audit of the original solver
#    - "There are at least two visible same-size squares" breaks: the mover is now
#      hidden, so the solver cannot sort visible squares by color anymore.
#    - "Red uniquely means destination" breaks: red now appears as both an n x n
#      target square and a border singleton. The solver must distinguish them by
#      size and border placement, not by raw color alone.
#    - "BFS may leave the start through any feasible neighbor" breaks: the combo
#      requires a decoded first step and forbids returning through the hidden start.
#    - "The input already contains mover-colored traversable cells" breaks: the
#      hidden start pocket is blue river, so traversability should use river+dest.
#
# d) Placement legality requirements enforced by the generator
#    - The yellow source marker is always on a non-corner border cell and decodes
#      to a legal hidden n x n source pocket one cell inside the grid.
#    - The red direction marker is also on a non-corner border cell, never merges
#      4-neighbor-wise with the red destination square, and never overlaps the
#      yellow source marker.
#    - The hidden source pocket, river, destination, and optional chokepoints are
#      all built before overlaying markers so that the decoded state is legal.
#
# e) Visual observability
#    - Across samples, humans can see that:
#      1. the black square is absent from every input,
#      2. a yellow edge pixel consistently marks where the black square appears from,
#      3. a red edge singleton separately determines which visible branch it takes.
#    - The dead-end stub near the hidden start makes the red direction control
#      visibly consequential instead of decorative.
#
# f) Visual cleanliness
#    - Both annotations live on the border, outside the busy channel interior.
#    - Only one extra dead-end stub is added near the hidden start; the rest of the
#      river keeps the original task's low-entropy winding appearance.
#
# g) Inferability of changes
#    - Color usage stays low-entropy:
#      green background, blue river, red target + direction control, yellow source
#      marker, black mover only in the output.
#    - The two border annotations have different structural roles:
#      yellow decodes location, red decodes direction. Because the output changes
#      when the chosen branch changes, that two-step reading process is inferable.
#
# h) Anti-degeneration target
#    - Hard generator invariants are:
#      1. zero black pixels in the input,
#      2. exactly one yellow border source marker,
#      3. exactly one red border singleton separate from the red n x n target.
#    - Separately, we audit a corpus-level metric after implementation:
#      compare combo outputs against "hidden source only, no forced direction".
#      The direction control should change the result on a substantial fraction of
#      samples, otherwise the fusion would be visually decorative.
#
# i) New solver assumptions and matching generator guarantees
#    - Solver assumes the largest solid square is the destination; generator makes
#      the destination the only solid square with size >= 2.
#    - Solver assumes there is exactly one non-red border singleton (yellow source
#      marker) and exactly one red border singleton (direction control); generator
#      constructs exactly that.
#    - Solver assumes the decoded hidden source pocket is feasible and that the
#      direction marker points to a branch that the hidden mover can actually take;
#      generator validates this by simulating the exact combo rule before returning.


_BG, _RIVER, _MOVER, _DEST, _SRC = 3, 1, 0, 2, 4
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


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


def _branch_dist(feasible: np.ndarray, sr: int, sc: int, step: tuple[int, int]) -> tuple[np.ndarray | None, tuple[int, int]]:
    dr, dc = step
    h2, w2 = feasible.shape
    fr, fc = sr + dr, sc + dc
    if not (0 <= fr < h2 and 0 <= fc < w2):
        return None, (fr, fc)
    if not feasible[fr, fc]:
        return None, (fr, fc)
    allowed = feasible.copy()
    allowed[sr, sc] = False
    if not allowed[fr, fc]:
        return None, (fr, fc)
    return _bfs_placement(allowed, fr, fc), (fr, fc)


def _pick_farthest(dist: np.ndarray, default: tuple[int, int]) -> tuple[int, int]:
    best = -1
    tr, tc = default
    for r in range(dist.shape[0]):
        for c in range(dist.shape[1]):
            if int(dist[r, c]) > best:
                best = int(dist[r, c])
                tr, tc = int(r), int(c)
    return tr, tc


def _simulate_direction(
    feasible: np.ndarray, sr: int, sc: int, dest_r: int, dest_c: int, step: tuple[int, int]
) -> tuple[int, int, bool, np.ndarray | None]:
    dist, start_pos = _branch_dist(feasible, sr, sc, step)
    if dist is None:
        return int(sr), int(sc), False, None
    if 0 <= dest_r < dist.shape[0] and 0 <= dest_c < dist.shape[1] and int(dist[dest_r, dest_c]) >= 0:
        return int(dest_r), int(dest_c), True, dist
    tr, tc = _pick_farthest(dist, start_pos)
    return int(tr), int(tc), False, dist


def _solid_square_components(g: np.ndarray, bg: int) -> list[dict]:
    squares: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        if bbox_h(b) != bbox_w(b):
            continue
        patch = g[int(b[0]) : int(b[2]) + 1, int(b[1]) : int(b[3]) + 1]
        if not bool((patch == int(col)).all()):
            continue
        squares.append(
            {
                "col": int(col),
                "r0": int(b[0]),
                "c0": int(b[1]),
                "n": int(bbox_h(b)),
                "area": int(comp["area"]),
            }
        )
    return squares


def _is_border_square(square: dict, h: int, w: int) -> bool:
    r0, c0, n = int(square["r0"]), int(square["c0"]), int(square["n"])
    return r0 == 0 or c0 == 0 or r0 + n == h or c0 + n == w


def _decode_start_from_src_marker(src_pt: tuple[int, int], n: int, h: int, w: int) -> tuple[int, int]:
    r, c = src_pt
    if r == 0:
        return 1, c
    if r == h - 1:
        return h - n - 1, c
    if c == 0:
        return r, 1
    if c == w - 1:
        return r, w - n - 1
    raise AssertionError("Source marker must lie on the border")


def _control_to_step(h: int, w: int, ctrl_r: int, ctrl_c: int) -> tuple[int, int]:
    hits: list[tuple[int, int]] = []
    if ctrl_r == 0:
        hits.append((1, 0))
    if ctrl_r == h - 1:
        hits.append((-1, 0))
    if ctrl_c == 0:
        hits.append((0, 1))
    if ctrl_c == w - 1:
        hits.append((0, -1))
    assert len(hits) == 1
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - The visible red n x n square is the destination.
    - The black mover is hidden from the input.
    - A yellow border singleton decodes the hidden source pocket.
    - A red border singleton decodes the mandatory first move direction.
    - After taking that forced first step, the hidden start may not be revisited.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    squares = _solid_square_components(g, bg)
    assert squares

    dest_candidates = [s for s in squares if int(s["n"]) >= 2]
    assert dest_candidates
    dest = max(dest_candidates, key=lambda s: (int(s["n"]), int(s["area"])))
    n = int(dest["n"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    border_singletons = [s for s in squares if int(s["n"]) == 1 and _is_border_square(s, h, w)]
    src_candidates = [s for s in border_singletons if int(s["col"]) != int(dest["col"])]
    dir_candidates = [s for s in border_singletons if int(s["col"]) == int(dest["col"])]
    assert len(src_candidates) == 1
    assert len(dir_candidates) == 1

    src_col = int(src_candidates[0]["col"])
    src_pt = (int(src_candidates[0]["r0"]), int(src_candidates[0]["c0"]))
    dir_pt = (int(dir_candidates[0]["r0"]), int(dir_candidates[0]["c0"]))
    sr, sc = _decode_start_from_src_marker(src_pt, n, h, w)
    step = _control_to_step(h, w, dir_pt[0], dir_pt[1])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if int(col) in (int(bg), int(dest["col"]), int(src_col)):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == int(river)) | (g == int(dest["col"]))
    h2 = h - n + 1
    w2 = w - n + 1
    assert h2 > 0 and w2 > 0
    assert 0 <= sr < h2 and 0 <= sc < w2

    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool(trav[r : r + n, c : c + n].all()):
                feasible[r, c] = True
    assert bool(feasible[sr, sc])

    tr, tc, _dest_ok, _dist = _simulate_direction(feasible, sr, sc, dr, dc, step)

    out = g.copy()
    out[tr : tr + n, tc : tc + n] = _MOVER
    return out


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


def _render_channel(
    grid: np.ndarray, path: list[tuple[int, int]], n: int, offsets_r: list[int], offsets_c: list[int]
) -> list[tuple[int, int]]:
    pixel_pos: list[tuple[int, int]] = []
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
        total_len = 2 * remove + max(1, hold_len)
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
        total_len = 2 * remove + max(1, hold_len)
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


def _river_connected(grid: np.ndarray) -> bool:
    mask = grid == _RIVER
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return int(num) == 1


def _compute_offsets(gsize: int, n: int, num_coarse: int) -> list[int] | None:
    if num_coarse == 1:
        return [1]
    first = 1
    last = gsize - n - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _paint_stub(grid: np.ndarray, sr: int, sc: int, n: int, step: tuple[int, int], length: int) -> None:
    dr, dc = step
    if dr == 1:
        grid[sr : sr + n + length, sc : sc + n] = _RIVER
    elif dr == -1:
        grid[sr - length : sr + n, sc : sc + n] = _RIVER
    elif dc == 1:
        grid[sr : sr + n, sc : sc + n + length] = _RIVER
    else:
        grid[sr : sr + n, sc - length : sc + n] = _RIVER


def _max_bg_stub_length(grid: np.ndarray, sr: int, sc: int, n: int, step: tuple[int, int]) -> int:
    h, w = grid.shape
    dr, dc = step
    if dr == 1:
        max_len = h - (sr + n)
        for length in range(max_len, 0, -1):
            if bool((grid[sr + n : sr + n + length, sc : sc + n] == _BG).all()):
                return int(length)
    elif dr == -1:
        max_len = sr
        for length in range(max_len, 0, -1):
            if bool((grid[sr - length : sr, sc : sc + n] == _BG).all()):
                return int(length)
    elif dc == 1:
        max_len = w - (sc + n)
        for length in range(max_len, 0, -1):
            if bool((grid[sr : sr + n, sc + n : sc + n + length] == _BG).all()):
                return int(length)
    else:
        max_len = sc
        for length in range(max_len, 0, -1):
            if bool((grid[sr : sr + n, sc - length : sc] == _BG).all()):
                return int(length)
    return 0


def _feasible_from_river(grid: np.ndarray, n: int) -> np.ndarray:
    h, w = grid.shape
    h2, w2 = h - n + 1, w - n + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            if bool((grid[r : r + n, c : c + n] == _RIVER).all()):
                feasible[r, c] = True
    return feasible


def _branch_summary(
    feasible: np.ndarray, sr: int, sc: int, dest_r: int, dest_c: int, step: tuple[int, int]
) -> tuple[int, int, bool] | None:
    dist, _start = _branch_dist(feasible, sr, sc, step)
    if dist is None:
        return None
    reachable = int((dist >= 0).sum())
    farthest = int(dist.max())
    dest_ok = 0 <= dest_r < dist.shape[0] and 0 <= dest_c < dist.shape[1] and int(dist[dest_r, dest_c]) >= 0
    return reachable, farthest, bool(dest_ok)


def _pick_dir_control_cell(
    grid: np.ndarray, step: tuple[int, int], sr: int, sc: int, n: int, forbidden: set[tuple[int, int]]
) -> tuple[int, int] | None:
    h, w = grid.shape
    pref_r = sr + n // 2
    pref_c = sc + n // 2

    if step == (1, 0):
        row = 0
        cols = sorted(range(1, w - 1), key=lambda c: abs(c - pref_c))
        for c in cols:
            if (row, c) in forbidden or grid[row, c] != _BG:
                continue
            if row + 1 < h and grid[row + 1, c] == _DEST:
                continue
            return row, c
    elif step == (-1, 0):
        row = h - 1
        cols = sorted(range(1, w - 1), key=lambda c: abs(c - pref_c))
        for c in cols:
            if (row, c) in forbidden or grid[row, c] != _BG:
                continue
            if row - 1 >= 0 and grid[row - 1, c] == _DEST:
                continue
            return row, c
    elif step == (0, 1):
        col = 0
        rows = sorted(range(1, h - 1), key=lambda r: abs(r - pref_r))
        for r in rows:
            if (r, col) in forbidden or grid[r, col] != _BG:
                continue
            if col + 1 < w and grid[r, col + 1] == _DEST:
                continue
            return r, col
    else:
        col = w - 1
        rows = sorted(range(1, h - 1), key=lambda r: abs(r - pref_r))
        for r in rows:
            if (r, col) in forbidden or grid[r, col] != _BG:
                continue
            if col - 1 >= 0 and grid[r, col - 1] == _DEST:
                continue
            return r, col
    return None


def _start_marker_from_side(
    start_coarse: tuple[int, int], sr: int, sc: int, coarse_rows: int, coarse_cols: int, h: int, w: int
) -> tuple[int, int]:
    cr, cc = start_coarse
    if cr == 0:
        return 0, sc
    if cr == coarse_rows - 1:
        return h - 1, sc
    if cc == 0:
        return sr, 0
    if cc == coarse_cols - 1:
        return sr, w - 1
    raise AssertionError("Start coarse cell must be on the border")


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, n: int) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates = []
    for direction in ("top", "bottom", "left", "right"):
        if direction == "top":
            dist = pr
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + n)):
                candidates.append((0, pc, dist))
        elif direction == "bottom":
            dist = h - (pr + n)
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + n, h) for c in range(pc, pc + n)):
                candidates.append((h - n, pc, dist))
        elif direction == "left":
            dist = pc
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc)):
                candidates.append((pr, 0, dist))
        else:
            dist = w - (pc + n)
            if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + n) for c in range(pc + n, w)):
                candidates.append((pr, w - n, dist))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return int(candidates[0][0]), int(candidates[0][1])


def _gen_attempt() -> np.ndarray | None:
    reached = random.random() < 0.5
    n = random.choice([2, 3, 4])

    min_size = max(10, 5 * n)
    if min_size > 20:
        return None
    gsize = random.randint(min_size, 20)
    h = w = gsize

    min_spacing = n + 1
    max_coarse = max(3, (gsize - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(gsize, n, coarse_rows)
    offsets_c = _compute_offsets(gsize, n, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < n or offsets_c[i + 1] - offsets_c[i] < n:
            return None

    edge_cells = []
    for c in range(1, coarse_cols - 1):
        edge_cells.append((0, c))
        edge_cells.append((coarse_rows - 1, c))
    for r in range(1, coarse_rows - 1):
        edge_cells.append((r, 0))
        edge_cells.append((r, coarse_cols - 1))
    if len(edge_cells) < 2:
        return None

    random.shuffle(edge_cells)
    start_coarse = edge_cells[0]
    far = sorted(edge_cells, key=lambda e: -(abs(e[0] - start_coarse[0]) + abs(e[1] - start_coarse[1])))
    end_coarse = far[0] if far[0] != start_coarse else far[1]

    walk = _hamiltonian_walk(coarse_rows, coarse_cols, start_coarse, end_coarse)
    if walk is None or len(walk) < 4:
        return None
    if len(set(cr for cr, _ in walk)) < coarse_rows or len(set(cc for _, cc in walk)) < coarse_cols:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    pixel_pos = _render_channel(grid, walk, n, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / h < 0.70 or np.any(mask, axis=0).sum() / w < 0.70:
        return None

    hidden_sr, hidden_sc = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    edge_dest = _extend_to_edge(grid, dest_r, dest_c, n)
    if edge_dest is None:
        return None
    new_dest_r, new_dest_c = edge_dest
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + n
        grid[r_lo:r_hi, dest_c : dest_c + n] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + n
        grid[dest_r : dest_r + n, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c

    if not (dest_r == 0 or dest_r + n == h or dest_c == 0 or dest_c + n == w):
        return None
    if (dest_r in (0, h - n)) and (dest_c in (0, w - n)):
        return None

    if not (0 < hidden_sr < h - n and 0 < hidden_sc < w - n):
        return None

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

    feasible_base = _feasible_from_river(grid, n)
    if not bool(feasible_base[hidden_sr, hidden_sc]):
        return None

    base_options = []
    for step in _DIR4:
        summary = _branch_summary(feasible_base, hidden_sr, hidden_sc, dest_r, dest_c, step)
        if summary is None:
            continue
        reachable_cells, farthest, dest_ok = summary
        base_options.append((int(dest_ok), reachable_cells, farthest, step))
    if not base_options:
        return None
    base_options.sort(reverse=True)
    main_step = tuple(base_options[0][3])

    decoy_trials: list[tuple[tuple[int, int], int]] = []
    for step in _DIR4:
        if step == main_step:
            continue
        max_len = _max_bg_stub_length(grid, hidden_sr, hidden_sc, n, step)
        if max_len <= 0:
            continue
        lengths = list(range(1, max_len + 1))
        random.shuffle(lengths)
        for length in lengths[: min(6, len(lengths))]:
            decoy_trials.append((step, int(length)))
    random.shuffle(decoy_trials)

    feasible = None
    decoy_step = None
    decoy_target = None
    main_target = None
    for step, length in decoy_trials:
        trial = grid.copy()
        _paint_stub(trial, hidden_sr, hidden_sc, n, step, length)
        feasible_trial = _feasible_from_river(trial, n)
        if not bool(feasible_trial[hidden_sr, hidden_sc]):
            continue
        mtr, mtc, main_hits_dest, _mdist = _simulate_direction(feasible_trial, hidden_sr, hidden_sc, dest_r, dest_c, main_step)
        dtr, dtc, decoy_hits_dest, _ddist = _simulate_direction(feasible_trial, hidden_sr, hidden_sc, dest_r, dest_c, step)
        if (mtr, mtc) == (hidden_sr, hidden_sc):
            continue
        if decoy_hits_dest or (dtr, dtc) == (hidden_sr, hidden_sc):
            continue
        if (dtr, dtc) == (mtr, mtc):
            continue
        if reached and not main_hits_dest:
            continue
        if (not reached) and main_hits_dest:
            continue
        grid = trial
        feasible = feasible_trial
        decoy_step = step
        decoy_target = (dtr, dtc)
        main_target = (mtr, mtc)
        break

    if feasible is None or decoy_step is None or decoy_target is None or main_target is None:
        return None

    use_main = random.random() < 0.5
    chosen_step = main_step if use_main else decoy_step
    tr, tc, chosen_hits_dest, _dist = _simulate_direction(feasible, hidden_sr, hidden_sc, dest_r, dest_c, chosen_step)
    if (tr, tc) == (hidden_sr, hidden_sc):
        return None
    if use_main:
        if reached and (tr, tc) != (dest_r, dest_c):
            return None
        if (not reached) and chosen_hits_dest:
            return None
    else:
        if (tr, tc) != decoy_target or chosen_hits_dest:
            return None

    src_marker = _start_marker_from_side(start_coarse, hidden_sr, hidden_sc, coarse_rows, coarse_cols, h, w)
    if src_marker[0] in (0, h - 1) and src_marker[1] in (0, w - 1):
        return None

    grid[dest_r : dest_r + n, dest_c : dest_c + n] = _DEST
    grid[src_marker[0], src_marker[1]] = _SRC

    dir_marker = _pick_dir_control_cell(grid, chosen_step, hidden_sr, hidden_sc, n, {src_marker})
    if dir_marker is None:
        return None
    grid[dir_marker[0], dir_marker[1]] = _DEST

    if np.count_nonzero(grid == _MOVER) != 0:
        return None
    if np.count_nonzero(grid == _SRC) != 1:
        return None

    red_components = [
        comp
        for comp in find_components(grid, bg=_BG, connectivity=4, monochromatic=True, return_format="dict")
        if int(comp["color"]) == _DEST
    ]
    red_singletons = [comp for comp in red_components if int(comp["area"]) == 1]
    red_big = [comp for comp in red_components if int(comp["area"]) == n * n]
    if len(red_singletons) != 1 or len(red_big) != 1:
        return None

    output = grid.copy()
    output[tr : tr + n, tc : tc + n] = _MOVER

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            result = _gen_attempt()
        except Exception:
            result = None
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
