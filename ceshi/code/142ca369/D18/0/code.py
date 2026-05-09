# Variant of 142ca369
# Dimension: D18 — 箭头方向编码方式 (src)
# Change: L形缺角方向编码——2x2中缺失的角决定射线的对角线发射方向（远离缺角方向）... → 箭头朝向由像素排列隐含

from __future__ import annotations

import random as _rng
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import colors_present, components_mask, get_bg


# Design analysis for the D18 variant
# ----------------------------------
# a) Concrete object design and behavior trace
#    - Original arrow: a 3-cell 2x2 L. The missing corner directly encodes one of
#      the four diagonal ray directions; the ray starts from the degree-2 corner.
#    - Variant arrow: still 3 monochrome cells, but now it is a 3-cell staircase
#      living in a 3x2 or 2x3 bbox and interpreted with 8-neighbor connectivity.
#      Its internal graph must contain exactly one orthogonal edge and one diagonal
#      edge:
#          tail --orth--> middle --diag--> head
#      The ray direction is the diagonal sign of (head - tail), and the ray starts
#      at the head pixel, continuing the endpoint-to-endpoint extension direction.
#    - Small trace:
#      Original SE arrow cells {(0,0),(0,1),(1,1)}:
#        missing corner is (1,0), so fire direction is SE, start is (0,1).
#      Variant SE arrow cells {(4,4),(5,4),(6,5)}:
#        (4,4)->(5,4) is orth, (5,4)->(6,5) is diag, so tail=(4,4), head=(6,5),
#        direction=SE, start=(6,5). Reflection and color inheritance are unchanged.
#    - Difference: direction is no longer read from a missing corner in a 2x2 box;
#      it is inferred from the asymmetric 3-pixel path geometry itself.
#
# b) Complexity calibration
#    - I keep only two staircase families per diagonal direction:
#        "row-first" and "col-first".
#      This gives visible variety without exploding the pattern vocabulary.
#    - I intentionally do NOT use 3-pixel straight lines here. Axis-aligned lines
#      would visually collide with wall segments, and diagonal 3-pixel lines would
#      leave a head/tail sign ambiguity that is hard to infer from 2-4 ARC examples.
#
# c) Hidden-assumption audit against the original main()
#    - Original `is_arrow_pts()` assumes every arrow fits a 2x2 bbox. This breaks.
#      Replaced by `_arrow_dir_and_start()` over 8-neighbor graph structure.
#    - Original `arrow_dir_and_start()` assumes the missing corner exists and that
#      the degree-2 corner is the launch point. This breaks. Replaced by:
#        tail = endpoint attached to the middle by the orthogonal edge
#        head = endpoint attached to the middle by the diagonal edge
#        start = head, dir = sign(head - tail)
#    - Original arrow detection uses 4-connectivity components. This breaks because
#      the new arrow may contain a diagonal adjacency. Replaced with 8-connectivity.
#    - Reflection, wall detection, color inheritance, and stop conditions do not
#      depend on the old encoding and remain valid.
#
# d) Placement legality guarantees required from the generator
#    - Arrow cells must stay in bounds for the chosen direction and staircase family.
#    - Rays, arrows, and walls still may not overlap.
#    - Walls keep the original placement semantics: one cell off the diagonal path
#      on the axis that should flip.
#    - Arrow colors are unique, and wall colors are sampled from the remaining
#      palette, so a wall never merges into an arrow under 8-connectivity.
#
# e) Visual observability
#    - Humans can now see that the emitting object is elongated as a slanted
#      staircase rather than a compact 2x2 corner. The output still reveals which
#      diagonal continuation was intended, making the new cue learnable.
#    - Both staircase families appear across samples, so the rule is not tied to a
#      single template.
#
# f) Visual cleanliness
#    - The arrow vocabulary remains size-3 and monochrome, preserving the original
#      task's sparse look.
#    - Preventing wall/arrow color sharing avoids accidental merged blobs that would
#      obscure the new source shape.
#
# g) Color-change inferability
#    - Color behavior is unchanged from the source task: rays inherit colors from
#      touched walls. The variant changes only source-direction encoding, not the
#      bounce color logic, so no extra color mapping burden is introduced.


def _sgn(x: int) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Recognize a 3-cell implicit-direction staircase arrow.

    Valid arrow graph in 8-neighbor space:
      endpoint(tail) -- orth -- middle -- diag -- endpoint(head)
    The ray starts at head and continues along sign(head - tail).
    """
    norm = [(int(r), int(c)) for r, c in pts]
    if len(norm) != 3 or len(set(norm)) != 3:
        return None

    deg = {p: 0 for p in norm}
    edge_kind: dict[frozenset[tuple[int, int]], str] = {}

    for i in range(3):
        r1, c1 = norm[i]
        for j in range(i + 1, 3):
            r2, c2 = norm[j]
            adr = abs(r1 - r2)
            adc = abs(c1 - c2)
            kind = None
            if adr + adc == 1:
                kind = "orth"
            elif adr == 1 and adc == 1:
                kind = "diag"
            if kind is None:
                continue
            key = frozenset((norm[i], norm[j]))
            edge_kind[key] = kind
            deg[norm[i]] += 1
            deg[norm[j]] += 1

    if sorted(deg.values()) != [1, 1, 2]:
        return None
    if len(edge_kind) != 2:
        return None

    middle = next(p for p, d in deg.items() if d == 2)
    ends = [p for p, d in deg.items() if d == 1]
    tail = None
    head = None
    for p in ends:
        kind = edge_kind[frozenset((p, middle))]
        if kind == "orth":
            tail = p
        elif kind == "diag":
            head = p

    if tail is None or head is None:
        return None

    dr = _sgn(head[0] - tail[0])
    dc = _sgn(head[1] - tail[1])
    if dr == 0 or dc == 0:
        return None
    return (int(dr), int(dc)), (int(head[0]), int(head[1]))


def _build_arrow(head_r: int, head_c: int, dr: int, dc: int, family: str) -> list[tuple[int, int]]:
    """Create one of the two staircase families for the requested diagonal direction."""
    mid = (head_r - dr, head_c - dc)
    if family == "row":
        tail = (head_r - 2 * dr, head_c - dc)
    elif family == "col":
        tail = (head_r - dr, head_c - 2 * dc)
    else:
        raise ValueError(f"unknown arrow family: {family}")
    return [tail, mid, (head_r, head_c)]


def _head_bounds(H: int, W: int, dr: int, dc: int, family: str) -> tuple[int, int, int, int]:
    pts = _build_arrow(0, 0, dr, dc, family)
    r_offsets = [r for r, _ in pts]
    c_offsets = [c for _, c in pts]
    r_lo = max(-r for r in r_offsets)
    r_hi = min(H - 1 - r for r in r_offsets)
    c_lo = max(-c for c in c_offsets)
    c_hi = min(W - 1 - c for c in c_offsets)
    return int(r_lo), int(r_hi), int(c_lo), int(c_hi)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from implicit-direction staircase arrows, reflecting off colored walls.

    Input structure:
    - bg grid with monochrome 3-pixel arrows encoded as 8-connected staircases.
    - Scattered colored wall pixels / wall segments remain unchanged from the source task.

    Rule:
    - Each arrow's 3-pixel geometry implies a tail->head order:
        tail--orth-->middle--diag-->head
      The ray starts at head and continues along sign(head - tail).
    - Reflection and color inheritance are unchanged from the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            pts = [(int(r), int(c)) for r, c in coords]
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            (dr, dc), start = info
            arrows.append((int(col), (int(dr), int(dc)), start))
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, (dr0, dc0), start in arrows:
        dr, dc = int(dr0), int(dc0)
        cur = (int(start[0]), int(start[1]))
        cur_col = int(start_col)
        seen = set()

        while True:
            state = (cur, int(dr), int(dc), int(cur_col))
            if state in seen:
                break
            seen.add(state)

            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if (nr, nc) in wall_cells:
                wcol = int(g[nr, nc])
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(wcol)
                cur_col = int(wcol)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(wcol)
                cur_col = int(wcol)
                dc = -int(dc)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt for the D18 variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = palette[n_arrows:] or palette[:]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for color in arrow_colors:
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        family = _rng.choice(["row", "col"])
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(100):
            r_lo, r_hi, c_lo, c_hi = _head_bounds(H, W, dr, dc, family)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr = _rng.randint(r_lo, r_hi)
            sc = _rng.randint(c_lo, c_hi)
            arrow_pts = _build_arrow(sr, sc, dr, dc, family)
            arrow_set = set(arrow_pts)

            if any((r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = color
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            ok = True
            bounces_left = n_bounces

            for seg_idx in range(n_bounces + 1):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max(3, max_steps - 1))
                else:
                    seg_len = max_steps

                for _step in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, cur_col))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(wr, wc)], "dr"))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(wr2, wc2)], "dc"))

                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda o: len(o[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = _rng.choice(wall_palette)
                walls.append((wpxs, wcol))

                if ray_cells:
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = color
            for r, c, col in ray_cells:
                out[r, c] = col
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = wcol
            occupied |= new_cells
            placed = True
            break

        if not placed:
            return None

    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the D18 variant of task 142ca369."""
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
