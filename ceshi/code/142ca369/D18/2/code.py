# Variant of 142ca369
# Dimension: D18 — 箭头方向编码方式 (src)
# Change: L形缺角方向编码——2x2中缺失的角决定射线的对角线发射方向（远离缺角方向）... → 网格边缘位置编码方向

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

"""
design_analysis:
  concrete_rule:
    preserved:
      - arrows are still monochrome 3-pixel L-shapes in a 2x2 bbox
      - walls are still 1-pixel or 3-pixel axis-aligned monochrome segments
      - rays still move diagonally, side-bounce on adjacent walls, inherit wall color, and stop on exit/another arrow
    changed:
      - an arrow is valid only if its 2x2 bbox touches exactly one grid edge
      - the ray still starts at the degree-2 right-angle vertex, but direction no longer depends on the missing corner
      - direction now depends on which edge the bbox touches, plus whether the right-angle sits on the upper/lower or left/right end of that edge span
    direction_table:
      - left edge:  dc = +1; top-left elbow -> dr = +1, bottom-left elbow -> dr = -1
      - right edge: dc = -1; top-right elbow -> dr = +1, bottom-right elbow -> dr = -1
      - top edge:   dr = +1; top-left elbow -> dc = +1, top-right elbow -> dc = -1
      - bottom edge: dr = -1; bottom-left elbow -> dc = +1, bottom-right elbow -> dc = -1
  behavior_trace:
    example:
      input_shape:
        - place cells {(4,0),(4,1),(5,0)} on the left edge
        - the elbow is (4,0), i.e. the upper endpoint of the left-edge span
      original_logic:
        - missing corner is (5,1), so old code fires away from it: (-1,-1)
        - the ray immediately points out of the grid, so the example is effectively dead
      variant_logic:
        - left edge forces inward horizontal motion dc=+1
        - elbow at the upper endpoint forces dr=+1
        - the same 3 pixels now fire down-right from (4,0), producing a visible ray and later side-bounces
      difference:
        - the identical local L-shape now means something different because border placement, not the absent corner, is the source of direction
  complexity_calibration:
    - There are still only eight diagonal launch states, so the task stays learnable.
    - Restricting arrows to borders makes the new rule visually salient in 2-4 examples.
    - The generator mixes all four edges and both endpoint polarities so the learner sees repeated evidence for the edge-based encoding.
  implicit_assumption_audit:
    solver_lines:
      - is_arrow_pts remains valid because the arrow geometry itself is unchanged.
      - old arrow_dir_and_start assumed every 2x2 L could infer direction from its missing corner; this is false in the variant and must be replaced.
      - the reflection loop assumes the start cell is part of the arrow and that the ray begins from that cell; keeping the right-angle start avoids unrelated downstream changes.
      - arrow detection is still per-color connected-component based; to keep that safe, the generator forbids walls from using arrow colors, so no same-color wall can merge into an arrow component.
      - the new solver assumes a valid arrow bbox touches exactly one edge; the generator avoids corners so this assumption is always satisfied.
  placement_legality:
    - left/right-edge arrows use row anchors in [1, H-3]; top/bottom-edge arrows use col anchors in [1, W-3], so no arrow touches two edges at once
    - occupied tracks arrows, planned rays, and walls, preventing overlap between objects
    - each planned bounce chooses a side wall exactly one cell off the ray in the travel direction, matching the solver's side-touch checks
    - every candidate input is oracle-checked by main(input) == planned_output before being accepted
  visual_observability:
    - because every arrow lives on the border, the "source of direction" is visible at a glance
    - two arrows with identical missing-corner patterns but on different edges behave differently, making the D18 change observable rather than cosmetic
  visual_cleanliness:
    - arrows are kept off corners and rays must contain at least 3 painted cells, so the border markers remain readable
    - bounce walls remain sparse and non-overlapping, which keeps the grid from turning into dense noise
  color_reasoning:
    - colors keep their original meaning: arrow color = initial ray color, wall color = inherited bounce color
    - no extra color mapping is introduced, so the only conceptual novelty is the new direction encoding
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from border-anchored L-arrows, reflecting off colored walls.

    Variant rule:
    - Every arrow is a 3-pixel L in a 2x2 bbox that touches exactly one border.
    - The ray starts at the right-angle vertex on the border.
    - The touched border determines the inward axis of travel; the elbow's upper/lower
      or left/right position inside the 2x2 bbox determines the other diagonal sign.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
        if len(pts) != 3:
            return False
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)

    def arrow_dir_and_start(pts: list[tuple[int, int]]):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)

        pts_set = set(pts)
        deg = {p: 0 for p in pts}
        for r, c in pts:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (rr, cc) in pts_set:
                    deg[(r, c)] += 1
        starts = [p for p, d in deg.items() if d == 2]
        if len(starts) != 1:
            return None
        start = starts[0]

        edges: list[str] = []
        if c0 == 0:
            edges.append("left")
        if c1 == W - 1:
            edges.append("right")
        if r0 == 0:
            edges.append("top")
        if r1 == H - 1:
            edges.append("bottom")
        if len(edges) != 1:
            return None

        edge = edges[0]
        sr, sc = start
        if edge == "left":
            if sc != c0:
                return None
            dr, dc = (1 if sr == r0 else -1), 1
        elif edge == "right":
            if sc != c1:
                return None
            dr, dc = (1 if sr == r0 else -1), -1
        elif edge == "top":
            if sr != r0:
                return None
            dr, dc = 1, (1 if sc == c0 else -1)
        else:
            if sr != r1:
                return None
            dr, dc = -1, (1 if sc == c0 else -1)
        return (int(dr), int(dc)), (int(sr), int(sc))

    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if is_arrow_pts(pts):
                info = arrow_dir_and_start(pts)
                if info is not None:
                    arrows.append((int(col), pts))
                    arrow_cells |= set(pts)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != int(bg)) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for start_col, pts in arrows:
        info = arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
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


def _max_steps(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _build_arrow(edge: str, anchor: int, polarity: int, H: int, W: int):
    """
    polarity chooses which endpoint of the border span hosts the elbow:
    - for left/right: +1 = upper endpoint, -1 = lower endpoint
    - for top/bottom: +1 = left endpoint, -1 = right endpoint
    """
    if edge == "left":
        r0 = int(anchor)
        if polarity == 1:
            pts = [(r0, 0), (r0, 1), (r0 + 1, 0)]
            start = (r0, 0)
            direction = (1, 1)
        else:
            pts = [(r0, 0), (r0 + 1, 0), (r0 + 1, 1)]
            start = (r0 + 1, 0)
            direction = (-1, 1)
    elif edge == "right":
        r0 = int(anchor)
        c1 = W - 1
        if polarity == 1:
            pts = [(r0, c1), (r0, c1 - 1), (r0 + 1, c1)]
            start = (r0, c1)
            direction = (1, -1)
        else:
            pts = [(r0, c1), (r0 + 1, c1), (r0 + 1, c1 - 1)]
            start = (r0 + 1, c1)
            direction = (-1, -1)
    elif edge == "top":
        c0 = int(anchor)
        if polarity == 1:
            pts = [(0, c0), (1, c0), (0, c0 + 1)]
            start = (0, c0)
            direction = (1, 1)
        else:
            pts = [(0, c0), (0, c0 + 1), (1, c0 + 1)]
            start = (0, c0 + 1)
            direction = (1, -1)
    else:
        c0 = int(anchor)
        r1 = H - 1
        if polarity == 1:
            pts = [(r1, c0), (r1 - 1, c0), (r1, c0 + 1)]
            start = (r1, c0)
            direction = (-1, 1)
        else:
            pts = [(r1, c0), (r1, c0 + 1), (r1 - 1, c0 + 1)]
            start = (r1, c0 + 1)
            direction = (-1, -1)
    return [(int(r), int(c)) for r, c in pts], (int(start[0]), int(start[1])), direction


def _gen_attempt() -> np.ndarray | None:
    """Single output-first attempt for the D18 border-direction variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(1, 7)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:n_arrows]
    wall_palette = [c for c in palette if c not in arrow_colors]

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    edge_modes = [
        ("left", 1),
        ("left", -1),
        ("right", 1),
        ("right", -1),
        ("top", 1),
        ("top", -1),
        ("bottom", 1),
        ("bottom", -1),
    ]

    for color in arrow_colors:
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]
        placed = False

        for _ in range(90):
            edge, polarity = _rng.choice(edge_modes)
            if edge in ("left", "right"):
                anchor = _rng.randint(1, H - 3)
            else:
                anchor = _rng.randint(1, W - 3)

            arrow_pts, start, direction = _build_arrow(edge, anchor, polarity, H, W)
            dr, dc = direction
            if any((r, c) in occupied for r, c in arrow_pts):
                continue
            if _max_steps(start[0], start[1], dr, dc, H, W) < 3:
                continue

            cur_r, cur_c = start
            cur_dr, cur_dc = dr, dc
            cur_col = int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            bounces_left = n_bounces

            for seg_idx in range(n_bounces + 1):
                max_steps = _max_steps(cur_r, cur_c, cur_dr, cur_dc, H, W)
                if seg_idx < n_bounces and bounces_left > 0:
                    if max_steps < 4:
                        bounces_left = 0
                        seg_len = max_steps
                    else:
                        seg_len = _rng.randint(3, max_steps - 1)
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
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = int(_rng.choice(wall_palette))
                walls.append((wpxs, wcol))

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
    """Generate a random valid input for the D18 border-direction variant."""
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
