from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 600


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the partial rotational path inside its odd square, then reverse the completed path's color sequence.

    Input structure:
    - The background is the dominant color.
    - All non-background pixels form one one-cell-wide 4-connected path inside one hidden odd square region.
    - The visible path is only a rotationally consistent subset of the completed path, so closing the square adds new path pixels.

    Rule:
    - Find the unique smallest odd square containing the foreground whose rotation-closure yields one simple 4-connected path.
    - Fill every missing rotation mate inside that square with the same color, then trace the completed path from the lexicographically smaller endpoint to the other endpoint.
    - Rewrite those same path cells with the traced color sequence in reverse order, leaving the geometry and the background unchanged.

    Key insight:
    - The colors must be reversed on the completed path, not on the visible prefix: the symmetry completion changes the path length and therefore changes where every color ends up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    candidate = _choose_candidate(g, bg)
    return candidate["output"]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _rot90_local(side: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    return c, side - 1 - r


def _rot180_local(side: int, pos: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    return side - 1 - r, side - 1 - c


def _rotate_patch_point(side: int, pos: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = pos
    for _ in range(k % 4):
        r, c = _rot90_local(side, (r, c))
    return r, c


def _all_foreground_coords(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0, "expected foreground"
    return coords


def _ordered_simple_path(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(grid != bg)}
    assert pts, "expected foreground"

    start = min(pts)
    seen = set()
    q = deque([start])
    seen.add(start)
    while q:
        cur = q.popleft()
        for nxt in _neighbors4(*cur):
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    assert seen == pts, "foreground must be 4-connected"

    degrees = {}
    for p in pts:
        degree = sum(nb in pts for nb in _neighbors4(*p))
        assert degree in (1, 2), "path must be non-branching"
        degrees[p] = degree
    endpoints = sorted([p for p, degree in degrees.items() if degree == 1])
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    path = [endpoints[0]]
    prev = None
    cur = endpoints[0]
    while cur != endpoints[1]:
        nxts = [nb for nb in _neighbors4(*cur) if nb in pts and nb != prev]
        assert len(nxts) == 1, "path traversal must be unique"
        prev, cur = cur, nxts[0]
        path.append(cur)
        assert len(path) <= len(pts), "path traversal looped"
    assert len(path) == len(pts), "path traversal missed pixels"
    return path


def _reverse_colors_on_path(grid: np.ndarray, bg: int) -> np.ndarray:
    path = _ordered_simple_path(grid, bg)
    values = [int(grid[r, c]) for r, c in path]
    out = grid.copy()
    for (r, c), value in zip(path, reversed(values)):
        out[r, c] = value
    return out


def _close_square(square: np.ndarray, bg: int, turns: tuple[int, ...]) -> np.ndarray | None:
    side = square.shape[0]
    assert square.shape[0] == square.shape[1]
    painted = square.copy()
    for r, c in np.argwhere(square != bg):
        color = int(square[r, c])
        for k in turns:
            rr, cc = _rotate_patch_point(side, (int(r), int(c)), k)
            current = int(painted[rr, cc])
            if current not in (bg, color):
                return None
            painted[rr, cc] = color
    return painted


def _collect_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    H, W = grid.shape
    fg = _all_foreground_coords(grid, bg)
    r_min, c_min = fg.min(axis=0)
    r_max, c_max = fg.max(axis=0)
    min_side = int(max(r_max - r_min + 1, c_max - c_min + 1))
    if min_side % 2 == 0:
        min_side += 1

    candidates: list[dict] = []
    rotation_kinds = [
        ("quarter_turn", (0, 1, 2, 3)),
        ("half_turn", (0, 2)),
    ]

    for side in range(min_side, min(H, W) + 1, 2):
        r0_lo = max(0, int(r_max) - side + 1)
        r0_hi = min(int(r_min), H - side)
        c0_lo = max(0, int(c_max) - side + 1)
        c0_hi = min(int(c_min), W - side)
        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        side_candidates = []
        for r0 in range(r0_lo, r0_hi + 1):
            for c0 in range(c0_lo, c0_hi + 1):
                square = grid[r0 : r0 + side, c0 : c0 + side]
                input_fg = int(np.count_nonzero(square != bg))
                for kind, turns in rotation_kinds:
                    closed = _close_square(square, bg, turns)
                    if closed is None:
                        continue
                    output_fg = int(np.count_nonzero(closed != bg))
                    if output_fg <= input_fg:
                        continue

                    completed = grid.copy()
                    completed[r0 : r0 + side, c0 : c0 + side] = closed
                    try:
                        output = _reverse_colors_on_path(completed, bg)
                    except AssertionError:
                        continue

                    side_candidates.append(
                        {
                            "side": side,
                            "r0": r0,
                            "c0": c0,
                            "kind": kind,
                            "turns": turns,
                            "added": output_fg - input_fg,
                            "completed": completed,
                            "output": output,
                        }
                    )
        candidates.extend(side_candidates)
    return candidates


def _choose_candidate(grid: np.ndarray, bg: int) -> dict:
    candidates = _collect_candidates(grid, bg)
    assert candidates, "no valid rotation-closure square found"
    for cand in candidates:
        cand["score"] = (
            cand["added"],
            int(np.count_nonzero(cand["completed"] != bg)),
            len(cand["turns"]),
            -cand["side"],
            -cand["r0"],
            -cand["c0"],
        )

    candidates.sort(key=lambda cand: cand["score"], reverse=True)
    if len(candidates) >= 2 and candidates[0]["score"] == candidates[1]["score"]:
        if not np.array_equal(candidates[0]["output"], candidates[1]["output"]):
            raise AssertionError("ambiguous best rotation-closure square")
    return candidates[0]


def _segment_lengths(total: int, n_segments: int) -> list[int]:
    remaining = total
    lengths = []
    for i in range(n_segments):
        slots = n_segments - i - 1
        lo = max(2, remaining - 4 * slots)
        hi = min(4, remaining - 2 * slots)
        lengths.append(random.randint(lo, hi))
        remaining -= lengths[-1]
    random.shuffle(lengths)
    return lengths


def _color_sequence(n: int) -> list[int]:
    palette = random.sample(PALETTE, random.randint(3, 4))
    valid_counts = [k for k in range(len(palette), n // 2 + 1) if 2 * k <= n <= 4 * k]
    n_segments = random.choice(valid_counts)
    lengths = _segment_lengths(n, n_segments)

    colors = random.sample(palette, len(palette))
    while len(colors) < n_segments:
        choices = [c for c in palette if c != colors[-1]]
        colors.append(random.choice(choices))

    seq = []
    for color, length in zip(colors, lengths):
        seq.extend([int(color)] * int(length))

    if seq == seq[::-1]:
        for i in range(1, len(seq)):
            if seq[i] != seq[0]:
                seq[0], seq[i] = seq[i], seq[0]
                break
    return seq


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    for a, b in zip(dirs, dirs[1:]):
        if a != b:
            turns += 1
    return turns


def _search_center_to_edge(side: int) -> list[tuple[int, int]] | None:
    center = (side // 2, side // 2)
    min_len = side // 2 + 3
    max_len = min(side * side // 2, min_len + side + 1)
    desired_len = random.randint(min_len, max_len)

    def search(path: list[tuple[int, int]], used: set[tuple[int, int]], turns: int):
        cur = path[-1]
        if cur[0] == 0 and len(path) >= desired_len and turns >= 1:
            return list(path)
        if len(path) >= max_len:
            return None

        options = []
        for nr, nc in _neighbors4(*cur):
            nxt = (nr, nc)
            if not _in_bounds((side, side), nr, nc):
                continue
            if nxt in used:
                continue
            rot = _rot180_local(side, nxt)
            if rot in used and nxt != center:
                continue
            min_edge_dist = nr
            budget_left = max_len - len(path)
            if min_edge_dist > budget_left:
                continue
            if nxt == center:
                continue
            score = (
                random.random(),
                -min_edge_dist if len(path) < desired_len - 1 else min_edge_dist,
            )
            options.append((score, nxt))

        options.sort()
        for _score, nxt in options:
            prev_dir = None if len(path) < 2 else (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
            new_dir = (nxt[0] - path[-1][0], nxt[1] - path[-1][1])
            found = search(
                path + [nxt],
                used | {nxt},
                turns + int(prev_dir is not None and new_dir != prev_dir),
            )
            if found is not None:
                return found
        return None

    return search([center], {center}, 0)


def _build_full_local_path(side: int) -> list[tuple[int, int]] | None:
    center_to_edge = _search_center_to_edge(side)
    if center_to_edge is None:
        return None
    edge_to_center = list(reversed(center_to_edge))
    tail = [_rot180_local(side, p) for p in edge_to_center[-2::-1]]
    full_path = edge_to_center + tail

    patch = np.zeros((side, side), dtype=int)
    for r, c in full_path:
        patch[r, c] = 1
    try:
        if len(_ordered_simple_path(patch, 0)) != len(full_path):
            return None
    except AssertionError:
        return None
    if _path_turns(full_path) < 2:
        return None
    return full_path


def _paint_path(path: list[tuple[int, int]], colors: list[int], shape: tuple[int, int]) -> np.ndarray:
    patch = np.zeros(shape, dtype=int)
    for (r, c), color in zip(path, colors):
        patch[r, c] = int(color)
    return patch


def _rotate_coords(side: int, coords: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    return [_rotate_patch_point(side, pos, k) for pos in coords]


def _naive_reverse_input_only(grid: np.ndarray, bg: int) -> np.ndarray:
    return _reverse_colors_on_path(grid, bg)


def _gen_attempt() -> np.ndarray | None:
    side = random.choice([7, 9])
    full_path = _build_full_local_path(side)
    if full_path is None:
        return None

    full_len = len(full_path)
    if full_len < 13:
        return None
    min_keep = full_len // 2 + 1
    max_keep = int(np.floor(full_len * 0.68))
    if min_keep > max_keep:
        return None
    input_path = full_path[: max_keep]
    colors = _color_sequence(full_len)

    full_patch = _paint_path(full_path, colors, (side, side))
    input_patch = np.zeros((side, side), dtype=int)
    for idx, pos in enumerate(input_path):
        input_patch[pos] = colors[idx]

    k = random.randrange(4)
    full_patch = np.rot90(full_patch, k)
    input_patch = np.rot90(input_patch, k)

    H = random.randint(max(11, side + 2), min(16, side + 6))
    W = random.randint(max(11, side + 2), min(16, side + 6))
    grid = np.zeros((H, W), dtype=int)
    r0 = random.randint(1, H - side - 1)
    c0 = random.randint(1, W - side - 1)
    grid[r0 : r0 + side, c0 : c0 + side] = input_patch

    try:
        chosen = _choose_candidate(grid, 0)
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, chosen["output"]):
        return None

    chosen_completed = chosen["completed"]
    added = int(np.count_nonzero(chosen_completed) - np.count_nonzero(grid))
    if added * 10 < 3 * int(np.count_nonzero(chosen_completed)):
        return None

    if chosen["kind"] != "half_turn":
        return None

    if np.array_equal(_naive_reverse_input_only(grid, 0), chosen["output"]):
        return None

    if chosen["side"] < 7:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 8")
