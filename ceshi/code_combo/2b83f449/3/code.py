# Combo variant of 2b83f449 (index 3)
# combo_spec: D1/1+D3/2
# Name: 十字阻隔+惯性移动
#
# Design analysis
# 1. Concrete fusion behavior:
#    - Seed task: each 1x3 gate becomes a 3-tall center post, leaving two side gaps
#      in the wall row. Tokens reason segment-by-segment and immediately choose the
#      unique exit below.
#    - Fusion: each gate becomes a visible plus blocker centered on the old gate:
#      blocker cells at center/up/down/left/right, plus two carved wall-row side
#      doorways at center-2 and center+2 so the wall is still traversable.
#      Tokens no longer "teleport to the unique exit". They keep moving in their
#      current direction until blocked; when downward motion is blocked by a cross,
#      they slide horizontally inside the current pipe segment until they reach the
#      nearest reachable doorway, then continue falling.
#    - Small example on width 17 with consecutive centers 8, 9, 8:
#      left token on row 0 starts at col 0, cannot drop straight through center 8,
#      slides right to doorway 6, drops, then on the next pipe row slides to doorway
#      7 for center 9, drops again. The output therefore shows both a wider blocker
#      geometry and a repeated sideways reroute that the seed task never needed.
#
# 2. Complexity / observability calibration:
#    - Only one gate per wall row is generated. This keeps the picture readable while
#      still making the cross silhouette obvious.
#    - Adjacent gate centers alternate between two neighboring columns, so the same
#      rerouting pattern repeats several times in one example. Humans can see both
#      the 5-cell blocker and the inertial left/right slides from only a few samples.
#    - We avoid many holes or extra colors; structure, not color arithmetic, carries
#      the rule.
#
# 3. Original implicit assumptions broken by the fusion:
#    - "gate cells become bg except for a center post" is false now: the old gate bar
#      mostly stays blocked and even gains a vertical arm.
#    - "each segment has at most one exit below / one entry above" is false. The
#      solver cannot rely on static segment exit counting.
#    - "final pack end can be inferred from the row's unique entry above" is too weak:
#      with inertial sideways motion we track each token's last vertical entry column.
#
# 4. Generator legality guarantees:
#    - Gate centers stay in [3, W-4] so cross arms and side doorways fit in-bounds.
#    - Consecutive centers differ by at most 1. This guarantees that both left-edge
#      and right-edge tokens can keep reaching the next doorway without needing to
#      cross an existing pipe-row blocker.
#    - Final left/right basin capacities are at least the number of tokens stacked
#      there, so packing never overflows the available pipe cells.
#
# 5. Visual cleanliness / inferability:
#    - The only salient output color change is still gate->blocker (7->6). The new
#      information is geometric: 3-cell bars become plus blockers, and token endpoints
#      shift because paths now include repeated sideways detours around those pluses.
#    - No hidden use of color ids, parity-only tricks, or ambiguous overlapping marks.
#
# 6. New solver assumptions and how the generator enforces them:
#    - `main()` assumes every true gate is an isolated horizontal 1x3 component.
#      `generate_input()` creates exactly that.
#    - `main()` assumes sideways search does not need upward motion or indefinite
#      bouncing. The generator uses a monotone downward spine, so the intended path is
#      always found by down/left/right motion only.
#    - `main()` assumes tokens remain on bg cells and blocker color 6 is never passable.
#      The generator only creates legal doorway columns with bg in the wall row.

from __future__ import annotations

from collections import Counter, defaultdict

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
        if col == WALL:
            continue
        lab, n = label_mask(g == col, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = np.argwhere(lab == k)
            if pts.shape[0] != 3:
                continue
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            if int(r1 - r0 + 1) == 1 and int(c1 - c0 + 1) == 3:
                gate = int(col)
                break
        if gate is not None:
            break
    if gate is None:
        raise ValueError("failed to infer gate color")

    cnt = Counter(map(int, g.flatten()))
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    if not token_candidates:
        raise ValueError("failed to infer token color")
    token = int(min(token_candidates, key=lambda c: (cnt[int(c)], int(c))))

    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    if not bg_candidates:
        raise ValueError("failed to infer bg color")
    bg = int(max(bg_candidates, key=lambda c: (cnt[int(c)], -int(c))))

    open_col = OPEN if gate == GATE and OPEN not in colors else max(1, gate - 1)
    return gate, token, bg, open_col


def _gate_centers(grid: np.ndarray, gate: int) -> list[tuple[int, int]]:
    g = np.array(grid, dtype=int)
    lab, n = label_mask(g == gate, connectivity=4)
    centers: list[tuple[int, int]] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == k)
        if pts.shape[0] != 3:
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
    h, w = out.shape
    for r, c in _gate_centers(out, gate):
        # The visible blocker is a 5-cell plus centered on the old gate, but the
        # wall row also keeps two farther side doorways so the network remains
        # traversable. Without those c±2 bg cells, the cross would seal the wall.
        out[r, c - 1 : c + 2] = int(open_col)
        if 0 <= c - 2 < w:
            out[r, c - 2] = int(bg)
        if 0 <= c + 2 < w:
            out[r, c + 2] = int(bg)
        if 0 <= r - 1 < h:
            out[r - 1, c] = int(open_col)
        if 0 <= r + 1 < h:
            out[r + 1, c] = int(open_col)
    return out


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    w = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < w:
        if int(grid[r, c]) != int(bg):
            c += 1
            continue
        start = c
        while c + 1 < w and int(grid[r, c + 1]) == int(bg):
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int] | None:
    for seg in _segments_in_row(grid, r, bg):
        if seg[0] <= c <= seg[1]:
            return seg
    return None


def _can_drop(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    h = grid.shape[0]
    return (
        r + 2 < h
        and int(grid[r + 1, c]) == int(bg)
        and int(grid[r + 2, c]) == int(bg)
    )


def _can_side(grid: np.ndarray, r: int, c: int, dc: int, bg: int) -> bool:
    nc = c + dc
    return 0 <= nc < grid.shape[1] and int(grid[r, nc]) == int(bg)


def _nearest_drop_distance(grid: np.ndarray, r: int, c: int, dc: int, bg: int) -> int | None:
    cur = c
    dist = 0
    while True:
        if _can_drop(grid, r, cur, bg):
            return dist
        if not _can_side(grid, r, cur, dc, bg):
            return None
        cur += dc
        dist += 1


def _simulate_token(
    grid: np.ndarray, bg: int, start_r: int, start_c: int
) -> tuple[int, int, int | None, int]:
    if int(grid[start_r, start_c]) != int(bg):
        return start_r, start_c, None, 0

    r = int(start_r)
    c = int(start_c)
    horizontal_steps = 0
    last_drop_col: int | None = None
    dir_dc = 0  # 0 = down, -1 = left, +1 = right
    pref = 1 if start_c <= grid.shape[1] // 2 else -1
    seen: set[tuple[int, int, int]] = set()

    while True:
        state = (r, c, dir_dc)
        if state in seen:
            break
        seen.add(state)

        if dir_dc == 0:
            if _can_drop(grid, r, c, bg):
                last_drop_col = c
                r += 2
                continue

            choices: list[tuple[tuple[int, int, int], int]] = []
            for dc in (-1, 1):
                if not _can_side(grid, r, c, dc, bg):
                    continue
                dist = _nearest_drop_distance(grid, r, c, dc, bg)
                score = (
                    0 if dist is not None else 1,
                    int(dist if dist is not None else 10**6),
                    0 if dc == pref else 1,
                )
                choices.append((score, dc))
            if not choices:
                break
            dir_dc = min(choices)[1]
            c += dir_dc
            horizontal_steps += 1
            continue

        if _can_drop(grid, r, c, bg):
            # Sideways motion only exists to route around a cross; once a doorway
            # is found, the token resumes its primary downward motion immediately.
            last_drop_col = c
            r += 2
            dir_dc = 0
            continue

        if _can_side(grid, r, c, dir_dc, bg):
            c += dir_dc
            horizontal_steps += 1
            continue

        alt = -dir_dc
        if _can_side(grid, r, c, alt, bg) and _nearest_drop_distance(grid, r, c, alt, bg) is not None:
            dir_dc = alt
            c += dir_dc
            horizontal_steps += 1
            continue
        break

    return r, c, last_drop_col, horizontal_steps


def _simulate_all_tokens(
    opened: np.ndarray, token: int, bg: int
) -> tuple[list[tuple[int, int, int | None, int]], int]:
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(opened == int(token))]
    opened[opened == int(token)] = int(bg)

    finals: list[tuple[int, int, int | None, int]] = []
    total_horizontal_steps = 0
    for r, c in token_pts:
        fr, fc, last_drop_col, hsteps = _simulate_token(opened, bg, r, c)
        finals.append((fr, fc, last_drop_col, c))
        total_horizontal_steps += hsteps
    return finals, total_horizontal_steps


def _pack_tokens(
    opened: np.ndarray,
    finals: list[tuple[int, int, int | None, int]],
    token: int,
    bg: int,
) -> np.ndarray:
    out = opened.copy()
    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)

    for r, c, last_drop_col, start_c in finals:
        seg = _find_segment(out, r, c, bg)
        if seg is None:
            continue
        a, b = seg
        if last_drop_col is not None and a <= last_drop_col <= b:
            pack_end = b if (last_drop_col - a) <= (b - last_drop_col) else a
        else:
            pack_end = a if start_c <= out.shape[1] // 2 else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = int(token)
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = int(token)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate, token, bg, open_col = _infer_gate_token_bg(g)
    opened = _open_gates(g, gate, bg, open_col)
    finals, _ = _simulate_all_tokens(opened, token, bg)
    return _pack_tokens(opened, finals, token, bg)


def _build_opened_grid(centers: list[int], width: int) -> np.ndarray:
    n_walls = len(centers)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        grid[2 * pipe_idx, :] = BG
    for wall_idx, c in enumerate(centers):
        r = 2 * wall_idx + 1
        grid[r, c - 2] = BG
        grid[r, c + 2] = BG
        grid[r - 1, c] = OPEN
        grid[r, c - 1 : c + 2] = OPEN
        grid[r + 1, c] = OPEN
    return grid


def _build_input(centers: list[int], width: int) -> np.ndarray:
    n_walls = len(centers)
    height = 2 * n_walls + 1
    grid = np.full((height, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        if pipe_idx < n_walls:
            grid[r, 0] = TOKEN
            grid[r, width - 1] = TOKEN
    for wall_idx, c in enumerate(centers):
        r = 2 * wall_idx + 1
        grid[r, c - 1 : c + 2] = GATE
    return grid


def _choose_centers(n_walls: int, width: int, rng: np.random.Generator) -> list[int]:
    base_min = max(3, n_walls)
    base_max = min(width - 5, width - n_walls - 2)
    if base_min > base_max:
        raise ValueError("no legal base center for requested shape")
    base = int(rng.integers(base_min, base_max + 1))
    phase = int(rng.integers(2))
    centers = [base + ((i + phase) % 2) for i in range(n_walls)]
    return centers


def _fusion_metrics(inp: np.ndarray) -> dict[str, int]:
    gate, token, bg, open_col = _infer_gate_token_bg(inp)
    opened = _open_gates(inp, gate, bg, open_col)
    finals, total_horizontal_steps = _simulate_all_tokens(opened, token, bg)
    output = _pack_tokens(opened, finals, token, bg)
    return {
        "horizontal_steps": int(total_horizontal_steps),
        "output_token_count": int(np.count_nonzero(output == token)),
        "gate_count": len(_gate_centers(inp, gate)),
        "proxy_sideways_score": int(total_horizontal_steps),
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_walls = int(rng.integers(6, 8))
    width = int(rng.integers(max(17, 2 * n_walls + 2), 20))
    centers = _choose_centers(n_walls, width, rng)

    inp = _build_input(centers, width)
    out = main(inp)

    metrics = _fusion_metrics(inp)
    if metrics["horizontal_steps"] < n_walls:
        return None
    if int(np.count_nonzero(out == TOKEN)) != 2 * n_walls:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(100):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for combo variant 2b83f449/3")
