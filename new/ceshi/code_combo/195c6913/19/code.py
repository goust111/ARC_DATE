# Combo variant of 195c6913 (index 19)
# combo_spec: D6/3+D15/0
# Name: 碰撞打洞+分轮模拟

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Input invariant: the visible path color stays 4-connected."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Each left-edge marker keeps path-colored vertical neighbors."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color


# Design analysis for D6/3 + D15/0
# a) Concrete trace:
#    If a marker at (mr,0) enters a rightward lane, round 1 paints only that
#    straight lane until the cell ahead is not path. At that corner cell C we no
#    longer place an end-colored mark; instead C is erased to bg, creating a gap.
#    Round 2 then starts by taking the scheduled turn from C into the side branch
#    and continues on the already-damaged path, so the output shows a colored
#    route split by bg holes exactly at collision corners.
# b) Complexity calibration:
#    The generator uses one marker and a staircase path with 3/5/7 forced
#    collision-holes. That keeps the new behavior obvious without introducing
#    ambiguous branches: repeated "colored segment -> bg hole -> turned segment"
#    motifs appear in every sample.
# c) Hidden assumptions broken from the original:
#    The original solver read traversability from the immutable input grid and
#    used end_color as a pure output marker. That is no longer valid. Here
#    traversability comes from a mutable live_path mask, collision cells can
#    disappear mid-simulation, and post-collision continuation must consult the
#    modified mask rather than the original grid.
# d) Placement legality:
#    The generator reserves the top-left template strip, keeps the zigzag inside
#    bounds, preserves a single path component after marker overwrite via a small
#    bridge around the marker, and places the 2x2 end block in an isolated
#    bottom-right bg area.
# e) Visual observability:
#    Every accepted sample enforces at least two holes and a non-empty second
#    phase, so the fused behavior is visible directly in the output rather than
#    hidden in execution order alone.
# f) Visual cleanliness:
#    Only one trajectory is active, the path is sparse, and template/end objects
#    stay far from the traversed zigzag. That avoids noisy overlaps between holes,
#    color cycling, and structural objects.
# g) Inferability of changes:
#    The color-cycle rule stays simple and repeated along the route. The only new
#    color/topology event is that collision corners revert to bg, giving a stable
#    structural cue across examples.


def _parse_pattern(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]]]:
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    H, W = g.shape
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                pattern.append(v)
                for rr, cc in ((r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)):
                    seen.add((rr, cc))
                    pat_cells.append((rr, cc))
    return pattern, pat_cells


def _parse_end_block(
    g: np.ndarray, template_bg: int, path_color: int, pattern: list[int]
) -> tuple[int | None, tuple[int, int] | None]:
    H, W = g.shape
    pat_set = set(pattern)
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if (
                int(g[r + 1, c]) == v
                and int(g[r, c + 1]) == v
                and int(g[r + 1, c + 1]) == v
            ):
                return v, (r, c)
    return None, None


def _parse_markers(
    g: np.ndarray, pattern: list[int], path_color: int
) -> list[tuple[int, int, int]]:
    H, W = g.shape
    pat_set = set(pattern)
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((r, c, v))
    return markers


def _left_turn(dr: int, dc: int) -> tuple[int, int]:
    return -dc, dr


def _right_turn(dr: int, dc: int) -> tuple[int, int]:
    return dc, -dr


def _advance_until_block(
    live_path: np.ndarray,
    out: np.ndarray,
    pattern: list[int],
    r: int,
    c: int,
    dr: int,
    dc: int,
    idx: int,
    my_cells: set[tuple[int, int]],
) -> tuple[str, int, int, int, int, int, int]:
    H, W = live_path.shape
    painted = 0
    for _ in range(H * W * 4):
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            return "edge", r, c, dr, dc, idx, painted
        if not bool(live_path[nr, nc]):
            return "collision", r, c, dr, dc, idx, painted
        idx = (idx + 1) % len(pattern)
        out[nr, nc] = int(pattern[idx])
        my_cells.add((nr, nc))
        painted += 1
        r, c = nr, nc
    return "limit", r, c, dr, dc, idx, painted


def _dig_corner(
    live_path: np.ndarray,
    out: np.ndarray,
    bg: int,
    r: int,
    c: int,
    my_cells: set[tuple[int, int]],
) -> int:
    if not bool(live_path[r, c]):
        return 0
    live_path[r, c] = False
    out[r, c] = int(bg)
    my_cells.discard((r, c))
    return 1


def _solve(
    grid: np.ndarray, collect_stats: bool = False
) -> tuple[np.ndarray, list[dict[str, int]]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max((cnt, int(col)) for col, cnt in counts.items() if int(col) != template_bg)[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells = _parse_pattern(g, template_bg, path_color)
    end_color, end_pos = _parse_end_block(g, template_bg, path_color, pattern)
    markers = _parse_markers(g, pattern, path_color)
    _assert_marker_embedded_in_path(g, markers, path_color)

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = template_bg

    if not pattern:
        return out, []

    live_path = g == path_color
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    per_marker_cells: list[set[tuple[int, int]]] = []
    stats: list[dict[str, int]] = []

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = sr + dr
            cc = sc + dc
            if 0 <= rr < H and 0 <= cc < W and bool(live_path[rr, cc]):
                init = (dr, dc)
                break

        my_cells: set[tuple[int, int]] = set()
        marker_stats = {
            "first_leg": 0,
            "second_leg": 0,
            "holes": 0,
            "collisions": 0,
            "end_color_found": 1 if end_color is not None else 0,
        }
        if init is None:
            per_marker_cells.append(my_cells)
            stats.append(marker_stats)
            continue

        r, c = sr, sc
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0

        phase, r, c, dr, dc, idx, painted = _advance_until_block(
            live_path, out, pattern, r, c, dr, dc, idx, my_cells
        )
        marker_stats["first_leg"] += painted
        hits = 0
        post_round = False

        while phase == "collision":
            marker_stats["holes"] += _dig_corner(live_path, out, template_bg, r, c, my_cells)
            turn_dr, turn_dc = _left_turn(dr, dc) if (hits % 2 == 0) else _right_turn(dr, dc)
            rr = r + turn_dr
            cc = c + turn_dc
            marker_stats["collisions"] += 1
            if not (0 <= rr < H and 0 <= cc < W and bool(live_path[rr, cc])):
                break

            hits += 1
            idx = (idx + 1) % len(pattern)
            out[rr, cc] = int(pattern[idx])
            my_cells.add((rr, cc))
            marker_stats["second_leg"] += 1
            post_round = True

            phase, r, c, dr, dc, idx, painted = _advance_until_block(
                live_path, out, pattern, rr, cc, turn_dr, turn_dc, idx, my_cells
            )
            marker_stats["second_leg"] += painted

        if not post_round:
            marker_stats["second_leg"] = 0

        per_marker_cells.append(my_cells)
        stats.append(marker_stats)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"

    return out, stats if collect_stats else []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate a cycle-colored particle in two rounds, with collisions punching holes in the path.

    Input structure:
    - Top-left: a row of 2x2 monochrome blocks gives the repeating color cycle.
    - Left edge: isolated single-pixel markers use one cycle color and sit inside the path band.
    - Elsewhere: one distinct 2x2 block marks the terminal color/object to erase.

    Rule:
    - Round 1: each marker moves only straight along path cells, advancing through the cycle colors.
    - When straight motion first collides with a non-path cell, the corner path cell is erased to bg.
    - Round 2 starts from that collision corner by taking the scheduled alternating turn
      (CCW, then CW, then CCW, ...), and all later movement uses the already-damaged path.
    - Template blocks and the terminal 2x2 block are erased; collision corners stay as bg holes.

    Key insight:
    - The crucial change is that a collision mutates the traversable structure before the turn phase.
      The particle is not just leaving a mark; it is deleting the corner it just used, so the continued
      walk must read a new topology rather than the original path.
    """
    out, _ = _solve(grid, collect_stats=False)
    return out


def _add_segment(cells: set[tuple[int, int]], r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        for c in range(lo, hi + 1):
            cells.add((r0, c))
    elif c0 == c1:
        lo, hi = sorted((r0, r1))
        for r in range(lo, hi + 1):
            cells.add((r, c0))
    else:
        raise ValueError("segments must be axis-aligned")


def _build_stair_path(H: int, W: int, marker_row: int, n_horiz: int) -> np.ndarray | None:
    h_lengths = [random.randint(4, 7) for _ in range(n_horiz)]
    x_ends: list[int] = []
    cur_x = 1
    for length in h_lengths:
        cur_x += length
        x_ends.append(cur_x)
    if x_ends[-1] > W - 7 or x_ends[0] < 6:
        return None

    available_up = marker_row - 4
    if available_up < 3 * n_horiz:
        return None

    v_lengths: list[int] = []
    remaining = available_up
    for i in range(n_horiz - 1):
        slots_left = (n_horiz - 1) - i
        max_len = min(6, remaining - 3 * slots_left)
        if max_len < 3:
            return None
        step = random.randint(3, max_len)
        v_lengths.append(step)
        remaining -= step

    cells: set[tuple[int, int]] = set()
    for r in range(marker_row + 1, H):
        cells.add((r, 0))
    cells.add((marker_row - 1, 0))
    for r in (marker_row - 1, marker_row, marker_row + 1):
        cells.add((r, 1))

    cur_r = marker_row
    cur_c = 1
    for i, x_end in enumerate(x_ends):
        _add_segment(cells, cur_r, cur_c, cur_r, x_end)
        cur_c = x_end
        target_r = cur_r - v_lengths[i] if i < len(v_lengths) else 0
        _add_segment(cells, cur_r, cur_c, target_r, cur_c)
        cur_r = target_r

    _add_segment(cells, 0, cur_c, 0, W - 1)

    mask = np.zeros((H, W), dtype=bool)
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return None
        mask[r, c] = True

    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    return mask


def _gen_attempt() -> np.ndarray | None:
    branch_roll = random.random()
    if branch_roll < 0.15:
        n_horiz = 2
    elif branch_roll < 0.45:
        n_horiz = 3
    else:
        n_horiz = 4
    side_ranges = {
        2: (20, 26),
        3: (22, 28),
        4: (25, 30),
    }
    marker_mins = {
        2: 10,
        3: 13,
        4: 16,
    }

    lo, hi = side_ranges[n_horiz]
    H = W = random.randint(lo, hi)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]
    n_blocks = random.randint(2, 4)
    cycle_colors = palette[3 : 3 + n_blocks]

    marker_low = max(marker_mins[n_horiz], H // 2)
    marker_high = H - 5
    if marker_low > marker_high:
        return None
    marker_row = random.randint(marker_low, marker_high)
    mask = _build_stair_path(H, W, marker_row, n_horiz)
    if mask is None:
        return None

    if mask[0, 0]:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    if not (mask[marker_row - 1, 0] and mask[marker_row + 1, 0] and mask[marker_row, 1]):
        return None

    template_row = random.choice([1, 2])
    block_cols = [1, 4, 7, 10][:n_blocks]
    for bc in block_cols:
        for dr in range(2):
            for dc in range(2):
                rr = template_row + dr
                cc = bc + dc
                if rr >= H or cc >= W or bool(mask[rr, cc]):
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for i, bc in enumerate(block_cols):
        g[template_row : template_row + 2, bc : bc + 2] = int(cycle_colors[i])

    end_candidates: list[tuple[int, int]] = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1 = max(0, r - 1), min(H, r + 3)
            c0, c1 = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1, c0:c1] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    g[marker_row, 0] = int(cycle_colors[0])

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    counts = Counter(map(int, g.flatten()))
    non_bg = sorted((cnt, col) for col, cnt in counts.items() if col != bg)
    if not non_bg or int(non_bg[-1][1]) != path_color:
        return None

    try:
        _, stats = _solve(g, collect_stats=True)
    except Exception:
        return None
    if len(stats) != 1:
        return None
    st = stats[0]
    if st["holes"] < 2:
        return None
    if st["second_leg"] < 4:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
