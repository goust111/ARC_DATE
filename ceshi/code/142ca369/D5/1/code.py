# Variant of 142ca369
# Dimension: D5 — 射线源对象形状 (src)
# Change: L形三角箭头——2x2外接矩形中缺少一个角的3像素L形，缺失角的方向决定射线发射的对角线方向
#         → 3x3 的 5 像素 Y-like 箭头，靠唯一角尾尖端的朝向编码对角发射方向
#
# Design analysis
# a) Concrete shape and behavior:
#    - Old source: 3-cell L in a 2x2 box. The missing corner encoded one of four diagonal ray directions,
#      and the ray started from the right-angle vertex inside the L.
#    - New source: a 5-cell monochrome source in a 3x3 box. It has a full 3-pixel middle bar plus a
#      short fork on one side and a single corner tail-tip on the opposite side:
#          NW: #..   NE: ..#   SW: .#.   SE: .#.
#              ###       ###       ###       ###
#              .#.       .#.       #..       ..#
#      The unique occupied corner is the tail tip; its corner position directly encodes the diagonal
#      direction. The ray now starts from that tip and continues outward along the same diagonal.
#    - Small trace difference:
#      old SE arrow: start at the L's right-angle vertex, then first painted cell is one step SE.
#      new SE arrow: start at the Y-like tail tip at the SE corner of its 3x3 bbox, then first painted
#      cell is one step SE outside the bbox. Reflection/color-inheritance after launch is unchanged.
# b) Complexity calibration:
#    - The source grows from 3 to 5 pixels and from 2x2 to 3x3, so it is visually distinct from the
#      original task and easier to notice in few-shot ARC examples.
#    - Only four rotations are allowed, so direction decoding stays low-entropy and learnable.
# c) Implicit assumptions from old main() that must change:
#    - `is_arrow_pts` assumed exactly 3 pixels in a 2x2 bbox. This is no longer valid.
#    - `arrow_dir_and_start` assumed "missing corner" semantics and that the launch point is the degree-2
#      right-angle vertex. The new source has no missing-corner rule; direction must come from shape match,
#      and the launch point is the unique occupied bbox corner (tail tip).
#    - All reflection code depends only on `(dr, dc)`, `start`, `arrow_cells`, and `wall_cells`, so that
#      downstream logic remains valid once detection/placement are updated.
# d) Placement legality:
#    - Generator places each arrow by its launch tip, then materializes the 3x3 source behind that tip.
#    - Exact-cell overlap between arrows, rays, and walls is forbidden.
#    - Additional guard: no newly placed arrow/wall cell may be 4-neighbor-adjacent to an existing
#      component of the same color in the input, otherwise same-color components could merge and break
#      solver-side source detection.
# e) Visual observability:
#    - The bigger Y-like source is immediately distinguishable from the old 3-pixel L.
#    - Because the ray emerges from the unique corner tail, two examples with different rotations make
#      the direction cue visible even before any wall interaction happens.
# f) Visual cleanliness:
#    - The generator keeps the original sparse output-first layout. Rays, walls, and sources never overlap,
#      and same-color component-merging is actively avoided.
# g) Color-rule clarity:
#    - Colors behave exactly as in the original task: sources start in their own color, walls recolor the
#      bounce point and become the new ray color. No new color mapping is introduced.
# h) New solver assumptions introduced here:
#    - A valid source is exactly one 5-cell 4-connected component matching one of the four 3x3 templates.
#    - The generator enforces those templates exactly and prevents same-color 4-connected mergers.

from __future__ import annotations

import random as _rng
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import colors_present, components_mask, get_bg


ARROW_PATTERNS: dict[tuple[int, int], frozenset[tuple[int, int]]] = {
    (-1, -1): frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)}),
    (-1, 1): frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 1)}),
    (1, -1): frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 0)}),
    (1, 1): frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
ARROW_CORNERS = {(0, 0), (0, 2), (2, 0), (2, 2)}
DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _arrow_info(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(pts) != 5:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0 != 2) or (c1 - c0 != 2):
        return None

    norm = frozenset((r - r0, c - c0) for r, c in pts)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm != pattern:
            continue
        tip_rel = next(iter(norm & ARROW_CORNERS))
        start = (r0 + tip_rel[0], c0 + tip_rel[1])
        return direction, start
    return None


def _make_arrow_pts(sr: int, sc: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    pattern = ARROW_PATTERNS[direction]
    tip_rel = next(iter(pattern & ARROW_CORNERS))
    dr0 = sr - tip_rel[0]
    dc0 = sc - tip_rel[1]
    return [(dr0 + r, dc0 + c) for r, c in pattern]


def _same_color_adjacent(cells: list[tuple[int, int]], color: int, grid: np.ndarray) -> bool:
    H, W = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cell_set and int(grid[nr, nc]) == int(color):
                return True
    return False


def _forbidden_neighbor_colors(cells: list[tuple[int, int]], color_map: dict[tuple[int, int], int]) -> set[int]:
    forbidden: set[int] = set()
    cell_set = set(cells)
    for r, c in cells:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set:
                continue
            if (nr, nc) in color_map:
                forbidden.add(int(color_map[(nr, nc)]))
    return forbidden


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from Y-like tail arrows, reflecting off colored wall pixels.

    Input structure:
    - bg grid with monochrome 5-pixel Y-like arrows (3x3 bbox, one unique occupied corner tip)
      and scattered colored wall pixels.

    Rule:
    - Each arrow fires a ray diagonally outward from its unique corner tail tip.
    - A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
    - On any bounce the ray adopts the wall's color and recolors the contact cell.
    - Ray stops when it exits the grid or reaches another arrow.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            info = _arrow_info(pts)
            if info is None:
                continue
            direction, start = info
            arrows.append((int(col), direction, start))
            arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, (dr, dc), cur in arrows:
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
    """
    Output-first generator for the D5 variant.

    For each Y-like arrow:
    - choose one of four diagonal tail-tip directions
    - place the 3x3 source so the tip is the ray launch point
    - plan 0-4 side-touch bounces
    - place the corresponding wall pixels and verify with `main`
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        direction = _rng.choice(DIRS)
        dr, dc = direction
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        tip_rel = next(iter(ARROW_PATTERNS[direction] & ARROW_CORNERS))
        r_lo, r_hi = tip_rel[0], H - (3 - tip_rel[0])
        c_lo, c_hi = tip_rel[1], W - (3 - tip_rel[1])
        if r_lo > r_hi or c_lo > c_hi:
            return None

        placed = False
        for _ in range(80):
            sr = _rng.randint(r_lo, r_hi)
            sc = _rng.randint(c_lo, c_hi)
            arrow_pts = _make_arrow_pts(sr, sc, direction)
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue
            if _same_color_adjacent(arrow_pts, color, grid):
                continue

            max_first_leg = min((H - 1 - sr) if dr == 1 else sr, (W - 1 - sc) if dc == 1 else sc)
            if max_first_leg < 3:
                continue

            cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, color
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            planned_colors: dict[tuple[int, int], int] = {p: color for p in arrow_pts}
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
                options.sort(key=lambda item: len(item[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    forbidden = _forbidden_neighbor_colors(wpxs, planned_colors)
                    forbidden |= _forbidden_neighbor_colors(wpxs, {(r, c): int(grid[r, c]) for r, c in np.argwhere(grid != 0)})
                    candidates = [col for col in palette if col not in forbidden]
                    if not candidates:
                        continue
                    chosen_wall = (wpxs, flip, int(_rng.choice(candidates)))
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip, wcol = chosen_wall
                walls.append((wpxs, wcol))
                for cell in wpxs:
                    planned_colors[cell] = wcol

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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
