# Combo variant of 2b83f449 (index 1)
# combo_spec: D1/0+D3/0
# Name: 门消失+令牌上升
#
# Design analysis for the fusion:
# a) Concrete behavior:
#    Original: a 1x3 gate rotates into a 3x1 post, so pipe segments are cut by
#    those posts and tokens fall downward through unique exits.
#    Fusion: the 1x3 gate disappears and leaves only its center cell as a 1-cell
#    wall hole; no 3-tall post is created. Tokens start in lower pipe rows and
#    move upward through unique holes. They finally pack at the end of the
#    destination segment that is farthest from the entry coming from below.
#    Tiny example: if a token starts on row 8 at the left edge, rises through a
#    hole at column 9 into row 6, and row 6 contains one long segment 0..14,
#    then the token enters row 6 from below at c=9 and settles toward the
#    farther end of that segment rather than staying under the hole.
# b) Complexity calibration:
#    Removing posts merges segments aggressively, so the generator uses one gate
#    per wall row plus a few black pipe holes. That keeps the rule visually clear
#    while still creating repeated "continue upward" vs "stop here" cases.
#    Tokens remain on every non-top pipe row, which is the clean upward mirror of
#    the base task's "every non-bottom pipe row" placement.
# c) Broken hidden assumptions from the base solver:
#    - There is no output color-6 vertical post anymore.
#    - Segment boundaries come only from black holes in pipe rows, not from posts.
#    - Motion is upward, so "exit below / entry above" becomes "entry above /
#      entry below for packing".
#    - Packing must look at the arrival direction from below, not above.
# d) Placement legality:
#    - Gates stay interior so `c-1:c+1` fits.
#    - Pipe holes never use edge columns, so source tokens at col 0 / W-1 stay valid.
#    - Each row has at most two holes, no overlaps, and all edits stay in-bounds.
# e) Visual observability:
#    The output visibly deletes every orange gate bar and replaces it with a thin
#    wall hole; tokens also move upward from lower rows rather than downward.
# f) Visual cleanliness:
#    The generator limits clutter by using one gate per wall row and sparse holes.
# g) New solver assumptions and generator guarantees:
#    The solver asserts that each reachable pipe segment has at most one upward
#    link and at most one downward link. The generator enforces that by using one
#    gate per wall row and splitting rows with black holes when paths should stop.
# h) Non-degeneracy check used during development:
#    On generated samples, combo outputs should contain no color 6 at all, at
#    least one token should rise >= 3 wall levels, and the combo output should
#    differ from the original solver's output on the same input. A quick sample
#    check gave 0/80 matches with the original solver and max-rise values 3..6.

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

WALL = 0
TOKEN = 3
GATE = 7
BG = 8


def _infer_gate_color(grid: np.ndarray) -> int:
    for color in sorted(set(map(int, grid.flatten()))):
        if color == WALL:
            continue
        for r in range(1, grid.shape[0], 2):
            for c in range(1, grid.shape[1] - 1):
                if (
                    int(grid[r, c - 1]) == color
                    and int(grid[r, c]) == color
                    and int(grid[r, c + 1]) == color
                ):
                    return color
    raise AssertionError("gate color not found")


def _infer_token_bg(grid: np.ndarray, gate: int) -> tuple[int, int]:
    counts = Counter(map(int, grid.flatten()))
    colors = {int(x) for x in grid.flatten()}
    token_candidates = [c for c in colors if c not in (WALL, gate)]
    assert token_candidates
    token = min(token_candidates, key=lambda c: (counts[c], c))
    bg_candidates = [c for c in colors if c not in (WALL, gate, token)]
    assert bg_candidates
    bg = max(bg_candidates, key=lambda c: (counts[c], -c))
    return int(token), int(bg)


def _erase_gates_to_holes(grid: np.ndarray, gate: int, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int)
    height, width = out.shape
    for r in range(1, height, 2):
        c = 1
        while c < width - 1:
            if (
                int(out[r, c - 1]) == gate
                and int(out[r, c]) == gate
                and int(out[r, c + 1]) == gate
            ):
                out[r, c - 1] = WALL
                out[r, c] = bg
                out[r, c + 1] = WALL
                c += 2
            c += 1
    return out


def _segments_in_row(grid: np.ndarray, r: int, bg: int) -> list[tuple[int, int]]:
    width = grid.shape[1]
    segs: list[tuple[int, int]] = []
    c = 0
    while c < width:
        if int(grid[r, c]) != bg:
            c += 1
            continue
        start = c
        while c + 1 < width and int(grid[r, c + 1]) == bg:
            c += 1
        segs.append((start, c))
        c += 1
    return segs


def _find_segment(segs: list[tuple[int, int]], c: int) -> tuple[int, int] | None:
    for a, b in segs:
        if a <= c <= b:
            return (a, b)
    return None


def _up_links(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r < 2:
        return []
    a, b = seg
    return [
        c
        for c in range(a, b + 1)
        if int(grid[r - 1, c]) == bg and int(grid[r - 2, c]) == bg
    ]


def _down_links(grid: np.ndarray, r: int, seg: tuple[int, int], bg: int) -> list[int]:
    if r + 2 >= grid.shape[0]:
        return []
    a, b = seg
    return [
        c
        for c in range(a, b + 1)
        if int(grid[r + 1, c]) == bg and int(grid[r + 2, c]) == bg
    ]


def _solve_upward(opened_with_tokens: np.ndarray, token: int, bg: int) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    out = np.array(opened_with_tokens, dtype=int)
    token_pts = [(int(r), int(c)) for r, c in np.argwhere(out == token)]
    out[out == token] = bg

    finals: list[tuple[int, int]] = []
    rise_counts: list[int] = []

    for r, c in token_pts:
        cur_r, cur_c = int(r), int(c)
        rises = 0
        if int(out[cur_r, cur_c]) != bg:
            finals.append((cur_r, cur_c))
            rise_counts.append(rises)
            continue
        while True:
            seg = _find_segment(_segments_in_row(out, cur_r, bg), cur_c)
            if seg is None:
                break
            links = _up_links(out, cur_r, seg, bg)
            assert len(links) <= 1, f"segment {seg} at row {cur_r} has upward choices {links}"
            if not links:
                break
            cur_r -= 2
            cur_c = int(links[0])
            rises += 1
        finals.append((cur_r, cur_c))
        rise_counts.append(rises)

    buckets: dict[tuple[int, tuple[int, int], int], int] = defaultdict(int)
    for r, c in finals:
        seg = _find_segment(_segments_in_row(out, r, bg), c)
        if seg is None:
            continue
        links = _down_links(out, r, seg, bg)
        assert len(links) <= 1, f"segment {seg} at row {r} has downward choices {links}"
        a, b = seg
        if links:
            entry = int(links[0])
            pack_end = b if (entry - a) <= (b - entry) else a
        else:
            pack_end = a if c <= a else b
        buckets[(r, seg, pack_end)] += 1

    for (r, (a, b), pack_end), count in buckets.items():
        if pack_end == a:
            for i in range(count):
                if a + i <= b:
                    out[r, a + i] = token
        else:
            for i in range(count):
                if b - i >= a:
                    out[r, b - i] = token

    return out, rise_counts, finals


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    gate = _infer_gate_color(g)
    token, bg = _infer_token_bg(g, gate)
    opened = _erase_gates_to_holes(g, gate, bg)
    out, _rise_counts, _finals = _solve_upward(opened, token, bg)
    return out


def _build_erased_grid(gates: list[int], holes: list[set[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, center in enumerate(gates):
        grid[2 * wall_idx + 1, center] = BG
    return grid


def _build_input(gates: list[int], holes: list[set[int]], width: int) -> np.ndarray:
    n_walls = len(gates)
    grid = np.full((2 * n_walls + 1, width), WALL, dtype=int)
    for pipe_idx in range(n_walls + 1):
        r = 2 * pipe_idx
        grid[r, :] = BG
        for c in holes[pipe_idx]:
            grid[r, c] = WALL
    for wall_idx, center in enumerate(gates):
        r = 2 * wall_idx + 1
        grid[r, center - 1 : center + 2] = GATE
    for pipe_idx in range(1, n_walls + 1):
        r = 2 * pipe_idx
        grid[r, 0] = TOKEN
        grid[r, width - 1] = TOKEN
    return grid


def _simulate_output(gates: list[int], holes: list[set[int]], width: int) -> tuple[np.ndarray, list[int], list[tuple[int, int]]]:
    opened = _build_erased_grid(gates, holes, width)
    seeded = opened.copy()
    n_walls = len(gates)
    for pipe_idx in range(1, n_walls + 1):
        seeded[2 * pipe_idx, 0] = TOKEN
        seeded[2 * pipe_idx, width - 1] = TOKEN
    return _solve_upward(seeded, TOKEN, BG)


def _pick_gate_centers(rng: np.random.Generator, n_walls: int, width: int) -> list[int] | None:
    centers: list[int] = []
    prev: int | None = None
    go_left = bool(rng.integers(2))
    for _ in range(n_walls):
        if prev is None:
            options = list(range(3, width - 3))
        elif go_left:
            options = list(range(2, prev - 1))
        else:
            options = list(range(prev + 2, width - 2))
        if not options:
            return None
        center = int(rng.choice(options))
        centers.append(center)
        prev = center
        go_left = not go_left
    return centers


def _side_hole_candidates(center: int, width: int, side: str) -> list[int]:
    if side == "left":
        return list(range(1, center))
    return list(range(center + 1, width - 1))


def _add_random_side_hole(
    holes: list[set[int]],
    row_idx: int,
    center: int,
    width: int,
    rng: np.random.Generator,
    side: str | None = None,
) -> None:
    if len(holes[row_idx]) >= 2:
        return
    sides = [side] if side is not None else ["left", "right"]
    candidates: list[int] = []
    for cur_side in sides:
        candidates.extend(_side_hole_candidates(center, width, cur_side))
    candidates = [c for c in candidates if c not in holes[row_idx]]
    if not candidates:
        return
    holes[row_idx].add(int(rng.choice(candidates)))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_walls = int(rng.integers(6, 10))
    width = int(rng.integers(16, 20))

    gates = _pick_gate_centers(rng, n_walls, width)
    if gates is None:
        return None

    holes: list[set[int]] = [set() for _ in range(n_walls + 1)]

    max_rise_target = int(rng.integers(3, min(6, n_walls) + 1))
    protected_rows = set(range(n_walls - max_rise_target + 1, n_walls))

    for pipe_idx in range(1, n_walls):
        upper = gates[pipe_idx - 1]
        lower = gates[pipe_idx]
        if pipe_idx not in protected_rows:
            lo, hi = sorted((upper, lower))
            if hi - lo <= 1:
                return None
            holes[pipe_idx].add((lo + hi) // 2)
        elif rng.random() < 0.2:
            prefer = "left" if max(upper, lower) < width // 2 else "right"
            _add_random_side_hole(holes, pipe_idx, upper, width, rng, side=prefer)

    if rng.random() < 0.55:
        side = "left" if bool(rng.integers(2)) else "right"
        _add_random_side_hole(holes, n_walls, gates[-1], width, rng, side=side)
    if rng.random() < 0.45:
        side = "left" if bool(rng.integers(2)) else "right"
        _add_random_side_hole(holes, 0, gates[0], width, rng, side=side)

    output, rise_counts, finals = _simulate_output(gates, holes, width)
    if int((output == TOKEN).sum()) != 2 * n_walls:
        return None
    if max(rise_counts, default=0) < 3:
        return None
    if len(set(rise_counts)) < 2:
        return None
    if len({r for r, _c in finals}) < 2:
        return None

    inp = _build_input(gates, holes, width)
    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate combo input for task 2b83f449")
