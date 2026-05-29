# Combo variant of 2b83f449 (index 10)
# combo_spec: D3/2+D9/1
# Name: 惯性移动+逐个占位下落
#
# Design analysis
# 1. Concrete fused behavior on the ARC grid:
#    - The gate geometry stays close to the seed task. Every horizontal 1x3 gate
#      opens into the usual 3-tall vertical post: the wall-row flanks become bg,
#      while the center column becomes color 6 on rows r-1,r,r+1.
#    - The D3/2 change is purely about motion. A token starts with direction "down".
#      If the cell two rows below is blocked, it does not instantly choose the
#      unique exit of its current segment; instead it turns left or right and keeps
#      that horizontal direction until it either finds a drop column or is blocked.
#    - The D9/1 change removes the old global "remove all tokens, simulate, then
#      pack buckets" phase. Gates still open first, but tokens are then processed in
#      input order: top-to-bottom, left-to-right. As soon as one token settles, it is
#      painted into the opened grid and becomes a new blocker for later tokens.
#
# 2. Small traced example used by the generator:
#    - Width 17, gate center 8 on every wall row, lower pipe rows split by holes.
#      Bottom row has holes at 5 and 11, so the only bottom basins are 6-7 and 9-10.
#    - First left token (row 0, col 0): it cannot move down through col 0, so it
#      slides right to col 7, drops repeatedly, then settles at (10,6).
#    - Second left token (row 2, col 0): same inertial slide, same shaft, then it
#      settles at (10,7).
#    - Third left token (row 4, col 0): it reaches row 8 at col 7, but now the cell
#      (10,7) is occupied by the second token. Downward motion is therefore blocked
#      by occupancy, so the token turns left and settles earlier at (8,4).
#    - This is exactly the fused rule: inertial sideways search is still the motion
#      primitive, and sequential occupancy changes the path of later tokens.
#
# 3. Why this is not an over-simplified or visually invisible tweak:
#    - The original task relied on <=1 exit per segment. This combo intentionally
#      breaks that assumption: the top pipe rows expose both gate flanks as possible
#      exits, so the old solver would assert on the same input.
#    - The sequential effect is also visible in the output, not just in hidden state:
#      extra tokens get stranded one pipe row above the bottom because earlier tokens
#      already filled the tiny bottom basins.
#    - Only one gate center is used per wall row and only two hole offsets are used
#      (`center±5` for lower-row isolation, `center±3` for bottom capacity). This
#      keeps the family readable from 2-3 examples instead of introducing many cases.
#
# 4. Original implicit assumptions that break:
#    - `seg_exit()` / `entry_from_above()` from the seed solver assume at most one
#      usable column below / above each segment. That is false here by construction.
#    - The old "Phase 2 bucket packing" assumed token interaction happens only after
#      all paths are known. That is false here because occupancy changes later paths.
#    - The old code could safely erase every token before simulation. Here we still
#      erase future tokens, but we must immediately redraw each settled token so later
#      motion tests treat it as a blocker.
#
# 5. Generator legality guarantees:
#    - Gate centers stay far enough from both borders so gate bars and both hole pairs
#      (`center±5`, `center±3`) stay in-bounds and off the outermost columns.
#    - Lower pipe rows get two holes at `center±5`. This isolates the late edge tokens
#      from the central shaft while still allowing earlier central tokens to descend.
#    - The bottom row gets holes at `center±3`, leaving exactly two 2-cell basins
#      around the central post. With three upper tokens per side, occupancy is forced.
#
# 6. Visual cleanliness / human inferability:
#    - Colors stay the same as the source task: wall=0, token=3, opened post=6,
#      gate=7, bg=8. The new rule is legible through geometry and token placement.
#    - Lower isolated edge tokens remain near the sides, while upper tokens form a
#      central "backed-up" queue. This makes the sequential interaction easy to spot.
#
# 7. New solver assumptions introduced by the combo and enforced by the generator:
#    - Gates are still isolated horizontal 1x3 components of one color.
#    - Horizontal inertial search never needs to move upward. The generator uses a
#      monotone downward shaft, so down/left/right motion is sufficient.
#    - No later token spawns on a cell already occupied by an earlier one. The lower
#      isolation holes prevent the backed-up central queue from touching edge spawns.

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import label_mask

WALL = 0
TOKEN = 3
OPEN = 6
GATE = 7
BG = 8


def _infer_gate_token_bg(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.array(grid, dtype=int)
    colors = set(map(int, g.flatten()))

    gate = None
    for col in sorted(colors):
        if int(col) == WALL:
            continue
        lab, n = label_mask(g == int(col), connectivity=4)
        found = False
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(k))
            if int(pts.shape[0]) != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                found = True
                break
        if found:
            gate = int(col)
            break
    if gate is None:
        raise ValueError("failed to infer gate color")

    cnt = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if int(c) not in (WALL, int(gate))]
    if not token_candidates:
        raise ValueError("failed to infer token color")
    token = int(min(token_candidates, key=lambda c: (int(cnt[int(c)]), int(c))))

    bg_candidates = [c for c in colors if int(c) not in (WALL, int(gate), int(token))]
    if not bg_candidates:
        raise ValueError("failed to infer bg color")
    bg = int(max(bg_candidates, key=lambda c: (int(cnt[int(c)]), -int(c))))

    open_col = OPEN if int(gate) == GATE and OPEN not in colors else int(gate - 1) if int(gate) > 0 else OPEN
    return int(gate), int(token), int(bg), int(open_col)


def _gate_centers(grid: np.ndarray, gate: int) -> list[tuple[int, int]]:
    lab, n = label_mask(grid == int(gate), connectivity=4)
    centers: list[tuple[int, int]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        if int(pts.shape[0]) != 3:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if int(r1 - r0 + 1) != 1 or int(c1 - c0 + 1) != 3:
            continue
        centers.append((int(r0), int(c0 + 1)))
    centers.sort()
    return centers


def _open_gates(grid: np.ndarray, gate: int, bg: int, open_col: int) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    h, _w = out.shape
    for r, c in _gate_centers(out, gate):
        out[int(r), int(c - 1) : int(c + 2)] = int(bg)
        for rr in (int(r - 1), int(r), int(r + 1)):
            if 0 <= int(rr) < h:
                out[int(rr), int(c)] = int(open_col)
    return out


def _can_drop(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    bg: int,
    *,
    use_occupancy: bool,
) -> bool:
    h = grid.shape[0]
    if int(r + 2) >= h:
        return False
    if int(grid[r + 1, c]) != int(bg) or int(grid[r + 2, c]) != int(bg):
        return False
    return (r + 2, c) not in occupied if use_occupancy else True


def _can_side(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dc: int,
    bg: int,
    *,
    use_occupancy: bool,
) -> bool:
    nc = int(c + dc)
    if not (0 <= nc < grid.shape[1]):
        return False
    if int(grid[r, nc]) != int(bg):
        return False
    return (r, nc) not in occupied if use_occupancy else True


def _first_drop_distance(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    dc: int,
    bg: int,
    *,
    use_occupancy: bool,
) -> int | None:
    cur_c = int(c)
    steps = 0
    while _can_side(grid, occupied, r, cur_c, dc, bg, use_occupancy=use_occupancy):
        cur_c += int(dc)
        steps += 1
        if _can_drop(grid, occupied, r, cur_c, bg, use_occupancy=use_occupancy):
            return int(steps)
    return None


def _pick_turn(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    r: int,
    c: int,
    bg: int,
    preferred_dc: int,
    *,
    use_occupancy: bool,
) -> int | None:
    choices: list[tuple[tuple[int, int, int], int]] = []
    for dc in (-1, 1):
        if not _can_side(grid, occupied, r, c, dc, bg, use_occupancy=use_occupancy):
            continue
        dist = _first_drop_distance(grid, occupied, r, c, dc, bg, use_occupancy=use_occupancy)
        score = (
            0 if dist is not None else 1,
            int(dist if dist is not None else 10**6),
            0 if int(dc) == int(preferred_dc) else 1,
        )
        choices.append((score, int(dc)))
    if not choices:
        return None
    return int(min(choices)[1])


def _simulate_one_token(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    start_r: int,
    start_c: int,
    bg: int,
    *,
    use_occupancy: bool,
) -> tuple[int, int, int, int, int]:
    if int(grid[start_r, start_c]) != int(bg):
        return int(start_r), int(start_c), 0, 0, 0
    if use_occupancy and (int(start_r), int(start_c)) in occupied:
        return int(start_r), int(start_c), 0, 0, 0

    r = int(start_r)
    c = int(start_c)
    direction = 0
    preferred_dc = 1 if int(start_c) <= grid.shape[1] // 2 else -1
    horizontal_steps = 0
    drop_steps = 0
    occupied_turns = 0
    seen: set[tuple[int, int, int]] = set()

    while True:
        state = (int(r), int(c), int(direction))
        if state in seen:
            break
        seen.add(state)

        if int(direction) == 0:
            if _can_drop(grid, occupied, r, c, bg, use_occupancy=use_occupancy):
                r += 2
                drop_steps += 1
                continue
            if (
                use_occupancy
                and int(r + 2) < grid.shape[0]
                and int(grid[r + 1, c]) == int(bg)
                and int(grid[r + 2, c]) == int(bg)
                and (int(r + 2), int(c)) in occupied
            ):
                occupied_turns += 1

            turn = _pick_turn(
                grid,
                occupied,
                r,
                c,
                bg,
                preferred_dc,
                use_occupancy=use_occupancy,
            )
            if turn is None:
                break
            direction = int(turn)
            c += int(direction)
            horizontal_steps += 1
            continue

        if _can_drop(grid, occupied, r, c, bg, use_occupancy=use_occupancy):
            r += 2
            drop_steps += 1
            direction = 0
            continue

        if _can_side(grid, occupied, r, c, direction, bg, use_occupancy=use_occupancy):
            c += int(direction)
            horizontal_steps += 1
            continue

        other = int(-direction)
        if _can_side(grid, occupied, r, c, other, bg, use_occupancy=use_occupancy):
            dist = _first_drop_distance(
                grid,
                occupied,
                r,
                c,
                other,
                bg,
                use_occupancy=use_occupancy,
            )
            if dist is not None:
                direction = int(other)
                c += int(direction)
                horizontal_steps += 1
                continue
        break

    return int(r), int(c), int(horizontal_steps), int(drop_steps), int(occupied_turns)


def _simulate_tokens(
    opened: np.ndarray,
    token: int,
    bg: int,
    *,
    use_occupancy: bool,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, int]]:
    token_pts = sorted((int(r), int(c)) for r, c in np.argwhere(opened == int(token)))
    path_grid = np.array(opened, dtype=int).copy()
    path_grid[path_grid == int(token)] = int(bg)
    out = np.array(path_grid, dtype=int).copy()

    occupied: set[tuple[int, int]] = set()
    finals: list[tuple[int, int]] = []
    metrics = {
        "horizontal_steps": 0,
        "drop_steps": 0,
        "occupied_turns": 0,
        "spawn_blocked": 0,
    }

    for start_r, start_c in token_pts:
        if use_occupancy and (int(start_r), int(start_c)) in occupied:
            metrics["spawn_blocked"] += 1
            continue

        end_r, end_c, hsteps, dsteps, occ_turns = _simulate_one_token(
            path_grid,
            occupied,
            start_r,
            start_c,
            bg,
            use_occupancy=use_occupancy,
        )
        metrics["horizontal_steps"] += int(hsteps)
        metrics["drop_steps"] += int(dsteps)
        metrics["occupied_turns"] += int(occ_turns)
        finals.append((int(end_r), int(end_c)))

        if use_occupancy:
            occupied.add((int(end_r), int(end_c)))

    for end_r, end_c in finals:
        out[int(end_r), int(end_c)] = int(token)

    return out, finals, metrics


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg, open_col = _infer_gate_token_bg(g)
    opened = _open_gates(g, gate, bg, open_col)
    out, _finals, _metrics = _simulate_tokens(opened, token, bg, use_occupancy=True)
    return out


def _build_input(n_walls: int, width: int, center: int) -> np.ndarray:
    height = 2 * int(n_walls) + 1
    grid = np.full((height, width), WALL, dtype=int)

    for pipe_idx in range(int(n_walls) + 1):
        r = 2 * int(pipe_idx)
        grid[r, :] = BG

    for wall_idx in range(int(n_walls)):
        r = 2 * int(wall_idx) + 1
        grid[r, int(center - 1) : int(center + 2)] = GATE

    funnel_rows = 3
    for pipe_idx in range(funnel_rows, int(n_walls)):
        r = 2 * int(pipe_idx)
        grid[r, int(center - 5)] = WALL
        grid[r, int(center + 5)] = WALL

    bottom_r = 2 * int(n_walls)
    grid[bottom_r, int(center - 3)] = WALL
    grid[bottom_r, int(center + 3)] = WALL

    for pipe_idx in range(int(n_walls)):
        r = 2 * int(pipe_idx)
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN

    return grid


def _candidate_centers(width: int) -> list[int]:
    centers = [c for c in range(6, width - 6) if 1 <= c - 5 and c + 5 <= width - 2]
    if not centers:
        raise ValueError("no legal gate centers for requested width")
    mid = width // 2
    centers.sort(key=lambda c: (abs(c - mid), c))
    return centers[:3]


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    gate, token, bg, open_col = _infer_gate_token_bg(inp)
    opened = _open_gates(inp, gate, bg, open_col)

    seq_out, seq_finals, seq_stats = _simulate_tokens(opened, token, bg, use_occupancy=True)
    _free_out, free_finals, free_stats = _simulate_tokens(opened, token, bg, use_occupancy=False)

    diverted = sum(
        1
        for idx in range(min(len(seq_finals), len(free_finals)))
        if tuple(seq_finals[idx]) != tuple(free_finals[idx])
    )
    above_bottom = sum(1 for r, _c in seq_finals if int(r) < inp.shape[0] - 1)

    return {
        "horizontal_steps": int(seq_stats["horizontal_steps"]),
        "drop_steps": int(seq_stats["drop_steps"]),
        "occupied_turns": int(seq_stats["occupied_turns"]),
        "spawn_blocked": int(seq_stats["spawn_blocked"]),
        "diverted_tokens": int(diverted),
        "above_bottom_tokens": int(above_bottom),
        "output_token_count": int(np.count_nonzero(seq_out == int(token))),
        "free_horizontal_steps": int(free_stats["horizontal_steps"]),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_walls = int(rng.integers(5, 8))
    width = int(rng.integers(17, 20))
    centers = _candidate_centers(width)
    center = int(centers[int(rng.integers(0, len(centers)))])

    inp = _build_input(n_walls, width, center)
    out = main(inp)
    metrics = _fusion_metrics(inp)

    if metrics["spawn_blocked"] != 0:
        return None
    if metrics["occupied_turns"] < 1:
        return None
    if metrics["diverted_tokens"] < 2:
        return None
    if metrics["horizontal_steps"] < 20:
        return None
    if metrics["above_bottom_tokens"] < 4:
        return None
    if int(np.count_nonzero(out == TOKEN)) != 2 * int(n_walls):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(50):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/10")
