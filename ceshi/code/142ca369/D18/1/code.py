# Variant of 142ca369
# Dimension: D18 — 箭头方向编码方式 (src)
# Change: L形缺角方向编码——2x2中缺失的角决定射线的对角线发射方向（远离缺角方向）... → 颜色编码方向
#
# Design analysis for the D18 variant
# ==================================
# variant_spec:
#   arrows:
#     old:
#       shape: monochrome 3-cell L inside a 2x2 box
#       direction_code: the missing corner; the ray starts at the right-angle corner
#     new:
#       shape: isolated 3x3 full square
#       colors: exactly two colors with multiplicities 8 + 1
#       body: the 8-cell majority color; the ray starts in the corner opposite the singleton
#       marker: the singleton color; its corner position encodes the direction
#       direction_rule: fire diagonally away from the marker corner
#       example:
#         marker at top-left -> start at bottom-right -> direction (1, 1)
#         marker at bottom-right -> start at top-left -> direction (-1, -1)
#   unchanged_core:
#     walls: 1-cell or 3-cell axis-aligned mono-color segments
#     ray: diagonal trace, side-touch reflection, color inheritance, stop at exit/other arrow
#
# a) Behavior trace
#   Original local reasoning:
#     A 2x2 box missing its top-left corner means the L points away from top-left.
#     The ray starts at bottom-right and moves down-right.
#   Variant local reasoning:
#     A full 2x2 square with a singleton marker in top-left means the marker stands in for the
#     old "missing corner". The opposite corner is still the launch point, and the ray still moves
#     down-right. The output changes only because the input cue is now color asymmetry instead of shape asymmetry.
#   Concrete mini-example:
#     Input arrow block:
#       8 3 3
#       3 3 3
#       3 3 3
#     Marker color 8 at top-left, body color 3 elsewhere -> direction (1,1), ray color starts as 3.
#     Under the old logic, the body color is an 8-cell component, so there is no hidden 3-cell L-source anymore.
#
# b) Complexity calibration
#   The new cue uses only four marker positions, matching the old four diagonal directions.
#   This keeps the freedom small enough for human inference while making the change visually obvious:
#   the arrow is no longer an L, so the solver must use the odd color, not the silhouette.
#   Each generated grid uses at least two arrows and enforces at least two distinct directions to make
#   the encoding observable from a few examples.
#
# c) Original hidden assumptions that break under D18
#   - Arrow detection by per-color 4-connected components only works for monochrome arrows.
#   - "len == 3 and bbox == 2x2" assumes one corner is absent; a 2x2 square would be misclassified as walls.
#   - start_col == arrow color assumed every arrow pixel shared one color; now only the body color should seed the ray.
#   - wall_cells = non-bg minus arrow_cells is still valid, but only after arrow detection is upgraded.
#
# d) Placement legality
#   Generator guarantees:
#   - each arrow is an isolated 2x2 non-bg component with no orthogonal non-bg neighbors outside the block;
#   - no overlap between arrows, walls, and planned ray cells;
#   - walls stay out of the arrow's 4-neighborhood so arrow recognition is stable;
#   - the planned output is still verified by main(input) == output.
#
# e) Visual observability
#   The key visible difference is that the source object is always a filled 3x3 block.
#   Direction can only be recovered from the singleton-color corner, and the ray leaves from the opposite corner.
#   Because multiple arrows with different marker corners appear together, a human can line up marker positions
#   with ray directions across examples.
#
# f) Visual cleanliness
#   Arrow blocks are isolated from other input objects in 4-neighborhood terms, so no wall touches an arrow edge.
#   This keeps each 2x2 block visually readable as a single source object rather than as cluttered wall fragments.
#
# g) Color inferability
#   The marker color is not a global color-to-direction mapping; only its position matters.
#   The singleton may reuse colors seen elsewhere, preventing overfitting to a special marker palette.
#   The body color remains the initial ray color, so the role split between "odd pixel encodes direction" and
#   "majority color paints the ray" is visible in the outputs.
#
# h) Variant-specific robustness check to avoid degeneration
#   After generation we re-run the arrow detector and require the detected 3x3 8+1 blocks to match the intended arrows.
#   This rejects accidental wall layouts that would also look like arrows.
#
# i) New solver assumptions audited
#   - Every intended arrow is an isolated 3x3 full block with exactly one singleton color placed on a corner.
#   - The singleton is unique within the block, so the opposite corner is well defined.
#   - No wall-only structure accidentally matches the same 3x3 8+1 isolated pattern.
#   - Rays may pass near arrows in the output, but only input non-bg cells matter for arrow detection.

from __future__ import annotations

from collections import Counter
import random as _rng
from typing import NamedTuple

import numpy as np

from arc_utils import get_bg


class Arrow(NamedTuple):
    body_color: int
    marker_color: int
    cells: tuple[tuple[int, int], ...]
    marker_pos: tuple[int, int]
    direction: tuple[int, int]
    start: tuple[int, int]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_arrows(g: np.ndarray) -> tuple[int, list[Arrow], set[tuple[int, int]]]:
    bg = int(get_bg(g))
    H, W = g.shape
    arrows: list[Arrow] = []
    arrow_cells: set[tuple[int, int]] = set()

    for r0 in range(H - 2):
        for c0 in range(W - 2):
            block = tuple((r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3))
            vals = [int(g[r, c]) for r, c in block]
            if any(v == bg for v in vals):
                continue
            counts = Counter(vals)
            if sorted(counts.values()) != [1, 8]:
                continue

            block_set = set(block)
            isolated = True
            for r, c in block:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in block_set and int(g[nr, nc]) != bg:
                        isolated = False
                        break
                if not isolated:
                    break
            if not isolated:
                continue

            body_color = next(int(col) for col, cnt in counts.items() if cnt == 8)
            marker_color = next(int(col) for col, cnt in counts.items() if cnt == 1)
            marker_pos = next((r, c) for r, c in block if int(g[r, c]) == marker_color)
            r2, c2 = r0 + 2, c0 + 2
            if marker_pos not in ((r0, c0), (r0, c2), (r2, c0), (r2, c2)):
                continue
            dr = 1 if marker_pos[0] == r0 else -1
            dc = 1 if marker_pos[1] == c0 else -1
            start = (r2 if marker_pos[0] == r0 else r0, c2 if marker_pos[1] == c0 else c0)

            arrow = Arrow(
                body_color=body_color,
                marker_color=marker_color,
                cells=tuple(sorted(block)),
                marker_pos=(int(marker_pos[0]), int(marker_pos[1])),
                direction=(int(dr), int(dc)),
                start=(int(start[0]), int(start[1])),
            )
            arrows.append(arrow)
            arrow_cells.update(block_set)

    return bg, arrows, arrow_cells


def _arrow_signature(arrow: Arrow) -> tuple[int, int, tuple[tuple[int, int], ...], tuple[int, int]]:
    return (arrow.body_color, arrow.marker_color, arrow.cells, arrow.marker_pos)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace diagonal rays from 3x3 square arrows whose singleton-color corner encodes direction.

    Input structure:
    - bg grid with isolated 3x3 source blocks using an 8+1 color split
    - scattered colored wall pixels / 3-pixel wall segments

    Rule:
    - The singleton-colored corner acts like the old missing corner.
    - The ray launches from the opposite corner in the majority color and travels diagonally away.
    - Reflection and color inheritance are unchanged from the original task.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, arrows, arrow_cells = _find_arrows(g)

    wall_cells: set[tuple[int, int]] = {
        (int(r), int(c)) for r, c in np.argwhere(g != bg) if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()

    for arrow in arrows:
        dr, dc = arrow.direction
        cur = arrow.start
        cur_col = int(arrow.body_color)
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
                if int(g[int(cur[0]), int(cur[1])]) == bg:
                    out[int(cur[0]), int(cur[1])] = wcol
                cur_col = wcol
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == bg:
                out[nr, nc] = int(cur_col)

            if int(dr) == 1 and (nr + 1, nc) in wall_cells:
                wcol = int(g[nr + 1, nc])
                if int(g[nr, nc]) == bg:
                    out[nr, nc] = wcol
                cur_col = wcol
                dr = -int(dr)
            elif int(dr) == -1 and (nr - 1, nc) in wall_cells:
                wcol = int(g[nr - 1, nc])
                if int(g[nr, nc]) == bg:
                    out[nr, nc] = wcol
                cur_col = wcol
                dr = -int(dr)

            if int(dc) == 1 and (nr, nc + 1) in wall_cells:
                wcol = int(g[nr, nc + 1])
                if int(g[nr, nc]) == bg:
                    out[nr, nc] = wcol
                cur_col = wcol
                dc = -int(dc)
            elif int(dc) == -1 and (nr, nc - 1) in wall_cells:
                wcol = int(g[nr, nc - 1])
                if int(g[nr, nc]) == bg:
                    out[nr, nc] = wcol
                cur_col = wcol
                dc = -int(dc)

    return out


def _arrow_block_for_start(start: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    sr, sc = start
    dr, dc = direction
    rows = (sr, sr - dr, sr - 2 * dr)
    cols = (sc, sc - dc, sc - 2 * dc)
    return tuple(sorted((r, c) for r in rows for c in cols))


def _marker_for_start(start: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    sr, sc = start
    dr, dc = direction
    return (int(sr - 2 * dr), int(sc - 2 * dc))


def _is_arrow_isolated(block: set[tuple[int, int]], other_input_cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    H, W = shape
    for r, c in block:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in block and (nr, nc) in other_input_cells:
                return False
    return True


def _build_direction_list(n_arrows: int) -> list[tuple[int, int]]:
    dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    _rng.shuffle(dirs)
    chosen = dirs[: min(2, n_arrows)]
    while len(chosen) < n_arrows:
        chosen.append(dirs[_rng.randrange(4)])
    _rng.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt for the D18 direction-encoding variant."""
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    n_arrows = _rng.randint(2, 5)
    palette = list(range(1, 10))
    _rng.shuffle(palette)
    body_colors = palette[:n_arrows]
    directions = _build_direction_list(n_arrows)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    input_occupied: set[tuple[int, int]] = set()
    planned_arrows: list[Arrow] = []

    for ai in range(n_arrows):
        body_color = int(body_colors[ai])
        marker_choices = [col for col in palette if col != body_color]
        marker_color = int(_rng.choice(marker_choices))
        dr, dc = directions[ai]
        n_bounces = _rng.choices([0, 1, 2, 3, 4], weights=[2, 3, 3, 2, 1])[0]

        placed = False
        for _ in range(80):
            r_lo, r_hi = (2 if dr == 1 else 0), (H - 1 if dr == 1 else H - 3)
            c_lo, c_hi = (2 if dc == 1 else 0), (W - 1 if dc == 1 else W - 3)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
            start = (int(sr), int(sc))
            arrow_pts = _arrow_block_for_start(start, (dr, dc))
            arrow_set = set(arrow_pts)
            marker_pos = _marker_for_start(start, (dr, dc))

            if any((r, c) in occupied for r, c in arrow_pts):
                continue
            if not _is_arrow_isolated(arrow_set, input_occupied, (H, W)):
                continue

            cur_r, cur_c, cur_dr, cur_dc, cur_col = sr, sc, dr, dc, body_color
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            wall_cells: set[tuple[int, int]] = set()
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
                    ray_cells.append((int(nr), int(nc), int(cur_col)))
                    ray_set.add((int(nr), int(nc)))
                    cur_r, cur_c = int(nr), int(nc)
                if not ok:
                    break

                if seg_idx >= n_bounces or bounces_left <= 0:
                    continue
                bounces_left -= 1

                options: list[tuple[list[tuple[int, int]], str]] = []
                wr, wc = cur_r + cur_dr, cur_c
                if 0 <= wr < H:
                    w3 = [(int(wr), int(wc + d)) for d in (-1, 0, 1)]
                    if all(0 <= c < W for _, c in w3):
                        options.append((w3, "dr"))
                    options.append(([(int(wr), int(wc))], "dr"))

                wr2, wc2 = cur_r, cur_c + cur_dc
                if 0 <= wc2 < W:
                    w3v = [(int(wr2 + d), int(wc2)) for d in (-1, 0, 1)]
                    if all(0 <= r < H for r, _ in w3v):
                        options.append((w3v, "dc"))
                    options.append(([(int(wr2), int(wc2))], "dc"))
                if not options:
                    ok = False
                    break

                _rng.shuffle(options)
                prefer_3 = _rng.random() < 0.6
                options.sort(key=lambda item: len(item[0]), reverse=prefer_3)
                chosen_wall = None
                for wpxs, flip in options:
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set or (r, c) in wall_cells for r, c in wpxs):
                        continue
                    if not _is_arrow_isolated(arrow_set, wall_cells | set(wpxs), (H, W)):
                        continue
                    chosen_wall = (wpxs, flip)
                    break
                if chosen_wall is None:
                    ok = False
                    break

                wpxs, flip = chosen_wall
                wcol = int(_rng.choice(palette))
                walls.append((wpxs, wcol))
                wall_cells.update(wpxs)

                if ray_cells:
                    lr, lc, _old = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, wcol)
                cur_col = wcol
                if flip == "dr":
                    cur_dr = -cur_dr
                else:
                    cur_dc = -cur_dc

            if not ok or len(ray_cells) < 3:
                continue

            input_cells = arrow_set | wall_cells
            all_cells = input_cells | ray_set
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(wpxs) for wpxs, _wcol in walls)
            if len(all_cells) != expected_count or all_cells & occupied:
                continue
            if not _is_arrow_isolated(arrow_set, wall_cells | input_occupied, (H, W)):
                continue

            arrow = Arrow(
                body_color=body_color,
                marker_color=marker_color,
                cells=tuple(sorted(arrow_pts)),
                marker_pos=marker_pos,
                direction=(int(dr), int(dc)),
                start=start,
            )

            for r, c in arrow_pts:
                grid[r, c] = body_color
                out[r, c] = body_color
            grid[marker_pos] = marker_color
            out[marker_pos] = marker_color

            for r, c, col in ray_cells:
                out[r, c] = col
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = wcol
                    out[r, c] = wcol

            occupied |= all_cells
            input_occupied |= input_cells
            planned_arrows.append(arrow)
            placed = True
            break

        if not placed:
            return None

    _bg, found_arrows, _arrow_cells = _find_arrows(grid)
    if sorted(_arrow_signature(a) for a in found_arrows) != sorted(_arrow_signature(a) for a in planned_arrows):
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
