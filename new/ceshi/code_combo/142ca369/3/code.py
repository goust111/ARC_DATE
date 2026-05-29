# Combo variant of 142ca369 (index 3)
# combo_spec: D4/0+D16/1
# Name: 颜色混合+颜色对决定反射

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# This file fuses two changes on top of the base 142ca369 task:
# - D16/1: reflection is controlled by the ray/wall color pair
# - D4/0: when reflection happens, the new ray color is a mixed color instead of
#   direct wall-color inheritance
#
# a) Concrete behavior and trace
# - Geometry stays the same as the base task:
#   - arrows are 3-cell monochrome L-shapes in a 2x2 box
#   - rays move diagonally from the arrow's right-angle corner
#   - walls are 1-cell or 3-cell axis-aligned monochrome segments
# - The fused rule is:
#   - if current ray color == touched wall color: no reflection; the ray passes
#     through that interaction and keeps both direction and color
#   - if current ray color != wall color: reflect exactly as in the base task
#     (corner hit flips both axes, side-touch flips one axis), but the new ray
#     color is not the wall color. Instead, it becomes the mixed color
# - To keep the mixed color inferable from 2-4 ARC examples, the generator uses
#   exactly 3 active non-background colors in every input. The mixing map is:
#   - for any different pair of active colors, the mixed color is the third
#     active color not present in the pair
# - Example with active colors {2, 4, 7}:
#   - ray 2 side-touches wall 2 -> no bend, next cells stay 2
#   - ray 2 side-touches wall 4 -> reflect, and the reflected segment becomes 7
#   - ray 7 later side-touches wall 4 -> reflect, and the reflected segment
#     becomes 2
# - Difference from the base task:
#   - same-color walls can now be ignored
#   - different-color reflections no longer inherit the wall color; they switch
#     to a third color
#
# b) Complexity calibration
# - A free-form color-mixing table across up to 9 colors would not be inferable.
# - Restricting each sample to exactly 3 active colors makes the mixing rule both
#   structured and visible: every different-color reflection turns into "the
#   remaining active color".
# - The generator forces repeated support:
#   - at least one same-color pass-through
#   - at least three different-color reflections
#   - all three unordered different-color pairs appear in reflected events
# - This is enough for a human to see that equality controls reflection, while
#   different pairs all map to the remaining third color.
#
# c) Hidden assumptions in the original solver that must change
# - Original corner-hit logic assumed every wall blocks and reflects. That fails
#   here because same-color corner hits must pass through.
# - Original side-touch logic flipped dr/dc whenever a wall existed on that side.
#   That also fails because same-color side touches must not flip.
# - Original bounce recoloring wrote the wall color into the contact cell. That is
#   invalid here: on different-color reflections, the contact cell must be
#   recolored to the mixed color, not the wall color.
# - Original solver implicitly tolerated arbitrary numbers of active colors because
#   inheritance used the observed wall color directly. The mixed-color rule creates
#   a new assumption: the intended task family uses exactly 3 active input colors
#   so the "remaining third color" is unique. The solver keeps a deterministic
#   fallback for non-ideal inputs, and the generator guarantees the intended case.
# - The state cache still needs the current ray color. That remains necessary,
#   because the same geometry with a different current color can produce a
#   different next reflection/pass decision.
#
# d) Placement legality
# - The generator stays output-first and only places side-touch events, which are
#   the cleanest way to show the fused rule.
# - Arrows, walls, and traced ray cells are kept disjoint in the input.
# - Every planned wall is placed one cell off the diagonal at the intended bounce
#   point, matching the base task's physical layout.
# - Final outputs are validated by the solver itself.
#
# e) Visual observability
# - Same-color walls are visually "ignored": the ray keeps going straight through
#   a wall of its own color.
# - Different-color walls bend the ray, but the bent segment is neither the old
#   ray color nor the wall color. Because only 3 active colors exist, the new
#   color is visibly the remaining third one.
# - Repeating all three different-color pairs in a single sample prevents the
#   mixing rule from looking accidental.
#
# f) Visual cleanliness
# - The generator uses only 3 active colors and 3-4 arrows, keeping the scene
#   sparse enough to read.
# - It avoids overlap between arrows, walls, and ray paths, so each object type
#   remains visually distinct.
#
# g) Change inferability
# - The key structural hint is closure on a 3-color set:
#   - same-color pair -> no bend
#   - different-color pair -> bend, new segment becomes the third color
# - Because the same 3 colors repeat across arrows and walls in each grid, a human
#   can infer the pair rule without needing color-id arithmetic.
#
# h) New assumptions review
# - Solver assumption: mixed color is uniquely determined by the active palette.
#   Generator guarantee: every input uses exactly 3 active non-bg colors.
# - Solver assumption: arrows are detected per monochrome 4-connected component.
#   Generator guarantee: arrow shapes remain isolated 3-cell monochrome L-shapes.
# - Solver assumption: generated interactions are sparse enough that a ray will not
#   immediately collide with unrelated arrows/walls. Generator guarantee: occupied
#   cells are tracked globally and tentative placements are rejected if they would
#   cause overlap or path reuse.


def _is_arrow_pts(pts: list[tuple[int, int]]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None
    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _detect_arrows_and_walls(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, list[tuple[int, int]]]], set[tuple[int, int]], dict[tuple[int, int], int]]:
    arrow_cells: set[tuple[int, int]] = set()
    arrows: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if _is_arrow_pts(pts):
                arrows.append((int(col), pts))
                arrow_cells |= set(pts)

    wall_colors = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }
    return arrows, arrow_cells, wall_colors


def _mix_color(ray_col: int, wall_col: int, palette: list[int]) -> int:
    if int(ray_col) == int(wall_col):
        return int(ray_col)
    candidates = [int(col) for col in palette if int(col) not in {int(ray_col), int(wall_col)}]
    if candidates:
        return int(candidates[0])
    return int(wall_col)


def _trace_mode(grid: np.ndarray, mode: str = "combo", collect_stats: bool = False):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    palette = colors_present(g, exclude={bg})
    arrows, arrow_cells, wall_colors = _detect_arrows_and_walls(g, bg)
    wall_cells = set(wall_colors)

    out = g.copy()
    diff_pairs_seen: set[tuple[int, int]] = set()
    stats = {
        "same_color_passes": 0,
        "diff_color_reflections": 0,
        "mixed_reflections": 0,
        "same_corner_passes": 0,
        "same_side_passes": 0,
        "diff_corner_reflections": 0,
        "diff_side_reflections": 0,
        "ambiguous_mix_events": 0,
    }

    def reflect_color(ray_col: int, wall_col: int) -> int:
        if mode in {"base", "pair_only"}:
            return int(wall_col)
        mixed = _mix_color(ray_col, wall_col, palette)
        if int(ray_col) != int(wall_col):
            candidates = [int(col) for col in palette if int(col) not in {int(ray_col), int(wall_col)}]
            if len(candidates) != 1:
                stats["ambiguous_mix_events"] += 1
        return int(mixed)

    for start_col, pts in arrows:
        info = _arrow_dir_and_start(pts)
        if info is None:
            continue
        (dr, dc), cur = info
        cur_col = int(start_col)
        seen: set[tuple[tuple[int, int], int, int, int]] = set()

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
                wcol = int(wall_colors[(nr, nc)])
                if mode != "base" and int(cur_col) == int(wcol):
                    cur = (int(nr), int(nc))
                    stats["same_color_passes"] += 1
                    stats["same_corner_passes"] += 1
                    continue
                prev_col = int(cur_col)
                new_col = reflect_color(prev_col, wcol)
                if mode != "base":
                    diff_pairs_seen.add(tuple(sorted((int(prev_col), int(wcol)))))
                    stats["diff_color_reflections"] += 1
                    stats["diff_corner_reflections"] += 1
                    if mode == "combo" and int(new_col) != int(wcol):
                        stats["mixed_reflections"] += 1
                if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                    out[int(cur[0]), int(cur[1])] = int(new_col)
                cur_col = int(new_col)
                dr, dc = -int(dr), -int(dc)
                continue

            if (nr, nc) in arrow_cells:
                break

            cur = (int(nr), int(nc))
            if int(out[nr, nc]) == int(bg):
                out[nr, nc] = int(cur_col)

            def handle_side_touch(axis: str, wpos: tuple[int, int]) -> None:
                nonlocal dr, dc, cur_col
                if wpos not in wall_cells:
                    return
                wcol = int(wall_colors[wpos])
                if mode != "base" and int(cur_col) == int(wcol):
                    stats["same_color_passes"] += 1
                    stats["same_side_passes"] += 1
                    return
                prev_col = int(cur_col)
                new_col = reflect_color(prev_col, wcol)
                if mode != "base":
                    diff_pairs_seen.add(tuple(sorted((int(prev_col), int(wcol)))))
                    stats["diff_color_reflections"] += 1
                    stats["diff_side_reflections"] += 1
                    if mode == "combo" and int(new_col) != int(wcol):
                        stats["mixed_reflections"] += 1
                if int(g[nr, nc]) == int(bg):
                    out[nr, nc] = int(new_col)
                cur_col = int(new_col)
                if axis == "dr":
                    dr = -int(dr)
                else:
                    dc = -int(dc)

            if int(dr) == 1:
                handle_side_touch("dr", (nr + 1, nc))
            else:
                handle_side_touch("dr", (nr - 1, nc))

            if int(dc) == 1:
                handle_side_touch("dc", (nr, nc + 1))
            else:
                handle_side_touch("dc", (nr, nc - 1))

    if collect_stats:
        stats["pair_coverage"] = len(diff_pairs_seen)
        stats["pair_labels"] = tuple(sorted(f"{a}-{b}" for a, b in diff_pairs_seen))
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - same ray/wall color => no reflection
    - different ray/wall color => normal reflection
    - reflected ray color => the third active non-bg color in the input palette
    """
    return _trace_mode(grid, mode="combo", collect_stats=False)


def _wall_options(
    cur_r: int,
    cur_c: int,
    cur_dr: int,
    cur_dc: int,
    axis: str,
    H: int,
    W: int,
) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    if axis == "dr":
        wr, wc = cur_r + cur_dr, cur_c
        if 0 <= wr < H:
            w3 = [(wr, wc + d) for d in (-1, 0, 1)]
            if all(0 <= c < W for _, c in w3):
                options.append(w3)
            options.append([(wr, wc)])
    else:
        wr, wc = cur_r, cur_c + cur_dc
        if 0 <= wc < W:
            w3 = [(wr + d, wc) for d in (-1, 0, 1)]
            if all(0 <= r < H for r, _ in w3):
                options.append(w3)
            options.append([(wr, wc)])
    _rng.shuffle(options)
    prefer_long = _rng.random() < 0.6
    options.sort(key=len, reverse=prefer_long)
    return options


def _gen_attempt() -> np.ndarray | None:
    """
    Output-first generator for the combo variant.

    Every input uses exactly 3 active non-bg colors. Different-color reflections
    recolor the bounce point to the remaining third color; same-color interactions
    pass through unchanged.
    """
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    palette = _rng.sample(list(range(1, 10)), k=3)
    n_arrows = _rng.randint(3, 4)
    arrow_colors = palette[:] + [_rng.choice(palette) for _ in range(n_arrows - 3)]
    _rng.shuffle(arrow_colors)

    n_events_per_arrow = [_rng.choices([1, 2, 3], weights=[1, 3, 2])[0] for _ in range(n_arrows)]
    total_events = sum(n_events_per_arrow)
    if total_events < 4:
        return None

    kind_tokens = ["same", "diff", "diff", "diff"]
    while len(kind_tokens) < total_events:
        kind_tokens.append(_rng.choices(["same", "diff"], weights=[2, 5])[0])
    _rng.shuffle(kind_tokens)

    grid = np.zeros((H, W), dtype=int)
    out = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()

    uncovered_pairs: set[tuple[int, int]] = {
        tuple(sorted((int(palette[0]), int(palette[1])))),
        tuple(sorted((int(palette[0]), int(palette[2])))),
        tuple(sorted((int(palette[1]), int(palette[2])))),
    }
    saw_same = False

    token_idx = 0
    for ai in range(n_arrows):
        color = int(arrow_colors[ai])
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        event_kinds = [str(kind_tokens[token_idx + j]) for j in range(n_events_per_arrow[ai])]
        token_idx += n_events_per_arrow[ai]

        placed = False
        for _ in range(140):
            r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
            c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
            if r_lo > r_hi or c_lo > c_hi:
                continue
            sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)

            mr, mc = sr - dr, sc - dc
            arrow_pts = [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]
            if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                continue

            cur_r, cur_c = sr, sc
            cur_dr, cur_dc = dr, dc
            cur_col = int(color)
            ray_cells: list[tuple[int, int, int]] = []
            walls: list[tuple[list[tuple[int, int]], int]] = []
            ray_set: set[tuple[int, int]] = set()
            arrow_set = set(arrow_pts)
            ok = True
            local_uncovered = set(uncovered_pairs)
            local_saw_same = bool(saw_same)

            for event_i, kind in enumerate(event_kinds):
                max_steps = min(
                    (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                    (W - 1 - cur_c) if cur_dc == 1 else cur_c,
                )
                remaining = len(event_kinds) - event_i - 1
                min_tail = 2 * remaining
                if max_steps < 2 or max_steps - 1 < min_tail:
                    ok = False
                    break
                seg_len_hi = max(2, max_steps - 1 - min_tail)
                seg_len = _rng.randint(2, seg_len_hi)

                for _ in range(seg_len):
                    nr, nc = cur_r + cur_dr, cur_c + cur_dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        ok = False
                        break
                    if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                        ok = False
                        break
                    ray_cells.append((nr, nc, int(cur_col)))
                    ray_set.add((nr, nc))
                    cur_r, cur_c = nr, nc
                if not ok:
                    break

                axis = _rng.choice(["dr", "dc"])
                chosen_wall = None
                for wpxs in _wall_options(cur_r, cur_c, cur_dr, cur_dc, axis, H, W):
                    if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wpxs):
                        continue
                    chosen_wall = wpxs
                    break
                if chosen_wall is None:
                    ok = False
                    break

                if kind == "same":
                    wcol = int(cur_col)
                    local_saw_same = True
                else:
                    diff_options = [int(col) for col in palette if int(col) != int(cur_col)]
                    preferred = [
                        int(col)
                        for col in diff_options
                        if tuple(sorted((int(cur_col), int(col)))) in local_uncovered
                    ]
                    wcol = int(_rng.choice(preferred or diff_options))
                    local_uncovered.discard(tuple(sorted((int(cur_col), int(wcol)))))
                    mixed = _mix_color(int(cur_col), int(wcol), palette)
                    lr, lc, _ = ray_cells[-1]
                    ray_cells[-1] = (lr, lc, int(mixed))
                    cur_col = int(mixed)
                    if axis == "dr":
                        cur_dr = -int(cur_dr)
                    else:
                        cur_dc = -int(cur_dc)

                walls.append((chosen_wall, int(wcol)))

            if not ok:
                continue

            max_steps = min(
                (H - 1 - cur_r) if cur_dr == 1 else cur_r,
                (W - 1 - cur_c) if cur_dc == 1 else cur_c,
            )
            if max_steps < 2:
                continue
            for _ in range(max_steps):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    ok = False
                    break
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_cells.append((nr, nc, int(cur_col)))
                ray_set.add((nr, nc))
                cur_r, cur_c = nr, nc
            if not ok or len(ray_cells) < 4:
                continue

            new_cells = set(arrow_pts) | ray_set
            for wpxs, _ in walls:
                new_cells.update(wpxs)
            expected_count = len(arrow_pts) + len(ray_cells) + sum(len(w) for w, _ in walls)
            if len(new_cells) != expected_count or new_cells & occupied:
                continue

            for r, c in arrow_pts:
                grid[r, c] = out[r, c] = int(color)
            for r, c, col in ray_cells:
                out[r, c] = int(col)
            for wpxs, wcol in walls:
                for r, c in wpxs:
                    grid[r, c] = out[r, c] = int(wcol)

            occupied |= new_cells
            uncovered_pairs = local_uncovered
            saw_same = local_saw_same
            placed = True
            break

        if not placed:
            return None

    if uncovered_pairs or not saw_same:
        return None

    solved_out, stats = _trace_mode(grid, mode="combo", collect_stats=True)
    if not np.array_equal(solved_out, out):
        return None
    if stats["same_color_passes"] < 1:
        return None
    if stats["diff_color_reflections"] < 3:
        return None
    if stats["mixed_reflections"] < 3:
        return None
    if stats["pair_coverage"] < 3:
        return None
    if stats["ambiguous_mix_events"] != 0:
        return None

    pair_only_out = _trace_mode(grid, mode="pair_only", collect_stats=False)
    base_out = _trace_mode(grid, mode="base", collect_stats=False)
    if int(np.count_nonzero(solved_out != pair_only_out)) == 0:
        return None
    if int(np.count_nonzero(solved_out != base_out)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random valid input for the combo variant of 142ca369."""
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")


def _variant_stats(grid: np.ndarray) -> dict[str, int | tuple[str, ...]]:
    """Helper for validation scripts."""
    combo_out, stats = _trace_mode(grid, mode="combo", collect_stats=True)
    pair_only_out = _trace_mode(grid, mode="pair_only", collect_stats=False)
    base_out = _trace_mode(grid, mode="base", collect_stats=False)
    stats["cells_changed_vs_pair_only"] = int(np.count_nonzero(combo_out != pair_only_out))
    stats["cells_changed_vs_base"] = int(np.count_nonzero(combo_out != base_out))
    return stats
