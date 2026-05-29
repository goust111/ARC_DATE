# Variant of 332f06d7
# Dimension: D4 — 源对象形态 (src)
# Change: 黑色实心n*n正方形，尺寸n=1~4，位于通道内部但不接触网格边缘... → 矩形移动体

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label as scipy_label

from arc_utils import bbox_h, bbox_w, find_components


"""
Design analysis for the D4 variant

a) Concrete behavior and trace
   - Visual objects:
     * background stays green (3), river stays blue (1).
     * mover stays a solid black rectangle of size h×w.
     * destination stays a solid red rectangle of the same h×w size.
     * h and w are axis-aligned and fixed; no rotation occurs during motion.
   - The important visible change from the source task is that the mover is no longer
     a square. Example: with h=2, w=3, horizontal river runs only need vertical
     thickness 2, while vertical river runs need horizontal thickness 3.
   - Small trace:
     * Original square logic with n=3 used a single feasibility test: every candidate
       placement needed a full 3×3 traversable window, so chokepoints narrower than 3
       in any local cross-section blocked equally.
     * Variant rectangle logic with h=2, w=3 uses a 2×3 placement window. A
       horizontal corridor only has to be 2 pixels tall, but a vertical corridor must
       be 3 pixels wide. If the generator narrows a vertical run to width 2, the mover
       stops before that gate even though width 2 would have been enough for its height.
     * The solver change therefore happens exactly at "window feasibility": `n×n`
       becomes `h×w`, and the stop point is now controlled by direction-specific river
       cross-sections.

b) Complexity calibration
   - To make the change obvious, the generator only uses non-square rectangles
     (`h != w`). This avoids visually degenerate samples that still look like the
     original task.
   - Dimensions are restricted to 1..4 in each axis, matching the original scale and
     keeping the task readable. A small fixed palette of aspect ratios gives repeated
     evidence across samples without exploding the search space.
   - The river remains a single winding component with both horizontal and vertical
     stretches, so the difference between height and width matters in the samples.

c) Hidden-assumption audit against the original `main()`
   - Original rectangle detection line `if bbox_h(b) != bbox_w(b): continue`
     assumed the mover/destination were squares. This is invalid now and must be
     removed.
   - Original grouping by a single side length `n` assumed a unique scalar size.
     This must become grouping by `(h, w)`.
   - Original mover/destination selection sorted by color id, which is not visually
     inferable. The variant instead relies on the already visible geometry:
     destination touches the grid border, mover does not.
   - Original feasible-placement loops used `trav[r:r+n, c:c+n]`. These slices must
     become `trav[r:r+h, c:c+w]`.
   - Original chokepoint logic used one width threshold `n`. The variant must narrow
     horizontal runs by reducing height, and vertical runs by reducing width.

d) Placement legality guarantees
   - Generator creates exactly two solid non-background rectangles of one shared shape.
   - Destination is extended to a border-aligned position while mover remains strictly
     interior, so solver can distinguish them without color ordering.
   - Coarse placement offsets reserve at least one cell of outer margin before border
     extension and at least `h`/`w` separation between adjacent coarse anchors.
   - Output-first verification remains: a sample is accepted only if `main(input)`
     exactly matches the constructed output.

e) Visual observability
   - Non-square rectangles make the D4 change directly visible in every sample.
   - Border-touching destination vs interior mover is human-detectable and replaces the
     invisible "sort colors numerically" dependency.
   - Stuck cases are produced by a gate that narrows only the relevant cross-section,
     so the reasoning genuinely depends on rectangle orientation.

f) Visual cleanliness
   - Objects use the original four-color scheme and do not overlap.
   - River is still a single clean corridor; no extra markers or labels are added.
   - Chokepoints are carved out of blue river pixels only, so the scene remains legible.

g) Color inferability
   - No color remapping rule is introduced.
   - Colors remain semantic and stable: black = mover, red = destination, blue = river,
     green = background.

Variant spec
  grid: 10-20 rows × 10-20 cols, kept square for continuity with the source task
  mover/dest shape: same solid rectangle h×w, each axis in 1..4, with h != w
  mover location: on the river, strictly inside the grid
  destination location: on the river, touches exactly one border side after extension
  traversal rule: feasible placement requires the full h×w window to lie in river/mover/dest
  stuck rule: at least one corridor segment is narrowed below the needed cross-section
              for that movement axis (height for horizontal travel, width for vertical)
"""


_BG, _RIVER, _MOVER, _DEST = 3, 1, 0, 2
_RECT_DIMS = [
    (1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1),
    (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3),
]


def _touches_border(r0: int, c0: int, h: int, w: int, H: int, W: int) -> bool:
    return r0 == 0 or c0 == 0 or r0 + h == H or c0 + w == W


def _feasible_mask(trav: np.ndarray, h: int, w: int) -> np.ndarray:
    H, W = trav.shape
    h2, w2 = H - h + 1, W - w + 1
    feasible = np.zeros((h2, w2), dtype=bool)
    for r in range(h2):
        for c in range(w2):
            feasible[r, c] = bool(trav[r:r + h, c:c + w].all())
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide a solid rectangle through a winding channel to reach a same-shaped border target.

    Input structure:
    - The corner-majority background surrounds one blue winding corridor.
    - Two solid same-sized rectangles lie on the corridor: an interior black mover and a
      border-touching red destination.

    Rule:
    - The black rectangle keeps its orientation and slides along the corridor toward the red one.
    - A placement is legal only when the full h×w mover footprint fits inside river pixels
      together with the mover/destination cells.
    - If the destination placement is reachable, the mover replaces it; otherwise the mover
      stops at the farthest reachable legal placement, and its start position becomes river.

    Key insight:
    - Horizontal motion is limited by corridor height, while vertical motion is limited by
      corridor width, so non-square rectangles are blocked by direction-specific chokepoints.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    bg = int(Counter([int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]).most_common(1)[0][0])

    rects: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        b = comp["bbox"]
        r0, c0 = int(b[0]), int(b[1])
        h, w = int(bbox_h(b)), int(bbox_w(b))
        patch = g[r0:int(b[2]) + 1, c0:int(b[3]) + 1]
        if not bool((patch == col).all()):
            continue
        rects.append(
            {
                "col": col,
                "r0": r0,
                "c0": c0,
                "h": h,
                "w": w,
                "touch": _touches_border(r0, c0, h, w, H, W),
            }
        )

    assert len(rects) >= 2

    by_shape: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for rect in rects:
        by_shape[(int(rect["h"]), int(rect["w"]))].append(rect)

    candidates: list[tuple[int, int, dict, dict]] = []
    for (h, w), items in by_shape.items():
        touching = [item for item in items if bool(item["touch"])]
        interior = [item for item in items if not bool(item["touch"])]
        if touching and interior:
            candidates.append((len(items), h * w, interior[0], touching[0]))
    assert candidates

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    mover = candidates[0][2]
    dest = candidates[0][3]

    h, w = int(mover["h"]), int(mover["w"])
    sr, sc = int(mover["r0"]), int(mover["c0"])
    dr, dc = int(dest["r0"]), int(dest["c0"])

    counts = Counter(map(int, g.flatten()))
    river = None
    for col, _ct in counts.most_common():
        if col == bg:
            continue
        if col in (int(mover["col"]), int(dest["col"])):
            continue
        river = int(col)
        break
    assert river is not None

    trav = (g == river) | (g == int(mover["col"])) | (g == int(dest["col"]))
    feasible = _feasible_mask(trav, h, w)
    h2, w2 = feasible.shape
    assert 0 <= sr < h2 and 0 <= sc < w2 and feasible[sr, sc]

    dist = _bfs_placement(feasible, sr, sc)

    if 0 <= dr < h2 and 0 <= dc < w2 and feasible[dr, dc] and dist[dr, dc] >= 0:
        tr, tc = dr, dc
    else:
        best = -1
        tr, tc = sr, sc
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] > best:
                    best = int(dist[r, c])
                    tr, tc = r, c

    out = g.copy()
    out[sr:sr + h, sc:sc + w] = river
    out[tr:tr + h, tc:tc + w] = int(mover["col"])
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
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
                    for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr2, nc2 = cr + dr2, cc + dc2
                        if (
                            0 <= nr2 < rows
                            and 0 <= nc2 < cols
                            and (nr2, nc2) not in test_visited
                            and (nr2, nc2) not in seen
                        ):
                            seen.add((nr2, nc2))
                            q.append((nr2, nc2))
                for dr2, dc2 in ((1, 0), (-1, 0), (0, 1), (0, -1)):
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
    grid: np.ndarray,
    path: list[tuple[int, int]],
    h: int,
    w: int,
    offsets_r: list[int],
    offsets_c: list[int],
) -> list[tuple[int, int]]:
    pixel_pos = []
    for cr, cc in path:
        pr, pc = offsets_r[cr], offsets_c[cc]
        pixel_pos.append((pr, pc))
        grid[pr:pr + h, pc:pc + w] = _RIVER
    for i in range(len(path) - 1):
        pr1, pc1 = pixel_pos[i]
        pr2, pc2 = pixel_pos[i + 1]
        if pr1 == pr2:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + w
            grid[pr1:pr1 + h, c_lo:c_hi] = _RIVER
        else:
            r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + h
            grid[r_lo:r_hi, pc1:pc1 + w] = _RIVER
    return pixel_pos


def _add_chokepoint(
    grid: np.ndarray,
    pixel_pos_i: tuple[int, int],
    pixel_pos_j: tuple[int, int],
    h: int,
    w: int,
    direction: str,
) -> bool:
    H, W = grid.shape
    pr1, pc1 = pixel_pos_i
    pr2, pc2 = pixel_pos_j

    if direction == "h":
        if h <= 1:
            return False
        narrow = random.randint(1, h - 1)
        remove = h - narrow
        side = random.randint(0, 1)
        c_lo = min(pc1, pc2) + w
        c_hi = max(pc1, pc2)
        if c_hi <= c_lo:
            c_lo, c_hi = min(pc1, pc2), max(pc1, pc2) + w
        hold_len = random.randint(1, min(6, max(1, c_hi - c_lo - 2 * remove)))
        total_len = 2 * remove + hold_len
        c_mid = (c_lo + c_hi) // 2
        c_start = max(c_lo, c_mid - total_len // 2)
        for step in range(total_len):
            c = c_start + step
            if not (0 <= c < W):
                continue
            if step < remove:
                depth = step + 1
            elif step >= total_len - remove:
                depth = total_len - step
            else:
                depth = remove
            for k in range(depth):
                rr = pr1 + k if side == 0 else pr1 + h - 1 - k
                if 0 <= rr < H:
                    grid[rr, c] = _BG
        return True

    if w <= 1:
        return False
    narrow = random.randint(1, w - 1)
    remove = w - narrow
    side = random.randint(0, 1)
    r_lo = min(pr1, pr2) + h
    r_hi = max(pr1, pr2)
    if r_hi <= r_lo:
        r_lo, r_hi = min(pr1, pr2), max(pr1, pr2) + h
    hold_len = random.randint(1, min(6, max(1, r_hi - r_lo - 2 * remove)))
    total_len = 2 * remove + hold_len
    r_mid = (r_lo + r_hi) // 2
    r_start = max(r_lo, r_mid - total_len // 2)
    for step in range(total_len):
        r = r_start + step
        if not (0 <= r < H):
            continue
        if step < remove:
            depth = step + 1
        elif step >= total_len - remove:
            depth = total_len - step
        else:
            depth = remove
        for k in range(depth):
            cc = pc1 + k if side == 0 else pc1 + w - 1 - k
            if 0 <= cc < W:
                grid[r, cc] = _BG
    return True


def _river_connected(grid: np.ndarray) -> bool:
    mask = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    if not mask.any():
        return False
    _labeled, num = scipy_label(mask.astype(int))
    return num == 1


def _compute_offsets(G: int, span: int, num_coarse: int):
    if num_coarse == 1:
        return [1]
    first = 1
    last = G - span - 1
    if last < first:
        return None
    step = (last - first) / (num_coarse - 1)
    return [int(round(first + i * step)) for i in range(num_coarse)]


def _extend_to_edge(grid: np.ndarray, pr: int, pc: int, h: int, w: int):
    H, W = grid.shape
    candidates = []

    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr) for c in range(pc, pc + w)):
        candidates.append((0, pc, pr))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr + h, H) for c in range(pc, pc + w)):
        candidates.append((H - h, pc, H - (pr + h)))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + h) for c in range(pc)):
        candidates.append((pr, 0, pc))
    if all(grid[r, c] in (_BG, _RIVER) for r in range(pr, pr + h) for c in range(pc + w, W)):
        candidates.append((pr, W - w, W - (pc + w)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[2])
    return candidates[0][:2]


def _pick_dims(reached: bool) -> tuple[int, int]:
    if reached:
        return random.choice(_RECT_DIMS)
    # Stuck cases need at least one axis > 1, which is already guaranteed, but bias
    # away from ultra-thin easy cases so directionality is more visible.
    weighted = [dims for dims in _RECT_DIMS if max(dims) >= 3 or min(dims) >= 2]
    return random.choice(weighted)


def _gen_attempt():
    reached = random.random() < 0.5
    h, w = _pick_dims(reached)
    span = max(h, w)

    min_size = max(10, 5 * span)
    if min_size > 20:
        return None
    G = random.randint(min_size, 20)
    H = W = G

    min_spacing = span + 1
    max_coarse = max(3, (G - 2) // min_spacing)
    coarse_dim = random.randint(3, min(6, max_coarse))
    coarse_rows = coarse_cols = coarse_dim

    offsets_r = _compute_offsets(G, h, coarse_rows)
    offsets_c = _compute_offsets(G, w, coarse_cols)
    if offsets_r is None or offsets_c is None:
        return None
    for i in range(len(offsets_r) - 1):
        if offsets_r[i + 1] - offsets_r[i] < h:
            return None
    for i in range(len(offsets_c) - 1):
        if offsets_c[i + 1] - offsets_c[i] < w:
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
    pixel_pos = _render_channel(grid, walk, h, w, offsets_r, offsets_c)

    mask = grid != _BG
    if np.any(mask, axis=1).sum() / H < 0.70 or np.any(mask, axis=0).sum() / W < 0.70:
        return None

    mover_r, mover_c = pixel_pos[0]
    dest_r, dest_c = pixel_pos[-1]

    result = _extend_to_edge(grid, dest_r, dest_c, h, w)
    if result is None:
        return None
    new_dest_r, new_dest_c = result
    if new_dest_r != dest_r:
        r_lo, r_hi = min(new_dest_r, dest_r), max(new_dest_r, dest_r) + h
        grid[r_lo:r_hi, dest_c:dest_c + w] = _RIVER
    if new_dest_c != dest_c:
        c_lo, c_hi = min(new_dest_c, dest_c), max(new_dest_c, dest_c) + w
        grid[dest_r:dest_r + h, c_lo:c_hi] = _RIVER
    dest_r, dest_c = new_dest_r, new_dest_c
    if not _touches_border(dest_r, dest_c, h, w, H, W):
        return None

    if _touches_border(mover_r, mover_c, h, w, H, W):
        return None
    if min(mover_r, H - (mover_r + h), mover_c, W - (mover_c + w)) > 2:
        return None

    grid[mover_r:mover_r + h, mover_c:mover_c + w] = _RIVER
    grid[dest_r:dest_r + h, dest_c:dest_c + w] = _RIVER

    if not reached:
        min_seg = max(1, len(walk) // 4)
        available = []
        for seg_idx in range(min_seg, len(walk) - 1):
            pr1, pc1 = pixel_pos[seg_idx]
            pr2, pc2 = pixel_pos[seg_idx + 1]
            direction = "h" if pr1 == pr2 else "v"
            if (direction == "h" and h > 1) or (direction == "v" and w > 1):
                available.append((seg_idx, direction))
        if not available:
            return None
        random.shuffle(available)
        applied = _add_chokepoint(grid, pixel_pos[available[0][0]], pixel_pos[available[0][0] + 1], h, w, available[0][1])
        if not applied or not _river_connected(grid):
            return None

    grid[mover_r:mover_r + h, mover_c:mover_c + w] = _MOVER
    grid[dest_r:dest_r + h, dest_c:dest_c + w] = _DEST

    trav = (grid == _RIVER) | (grid == _MOVER) | (grid == _DEST)
    feasible = _feasible_mask(trav, h, w)
    if not feasible[mover_r, mover_c]:
        return None

    dist = _bfs_placement(feasible, mover_r, mover_c)
    h2, w2 = feasible.shape
    dest_ok = 0 <= dest_r < h2 and 0 <= dest_c < w2 and feasible[dest_r, dest_c] and dist[dest_r, dest_c] >= 0
    if reached and not dest_ok:
        return None
    if not reached and dest_ok:
        return None

    if dest_ok:
        tr, tc = dest_r, dest_c
    else:
        best_d = -1
        tr, tc = mover_r, mover_c
        for r in range(h2):
            for c in range(w2):
                if dist[r, c] > best_d:
                    best_d = int(dist[r, c])
                    tr, tc = r, c

    output = grid.copy()
    output[mover_r:mover_r + h, mover_c:mover_c + w] = _RIVER
    output[tr:tr + h, tc:tc + w] = _MOVER

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input():
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
