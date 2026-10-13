from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500
TURN_OPTIONS = (0, 1, 2, 3, 4)

Coord = tuple[int, int]


def _foreground_components(grid: np.ndarray) -> list[list[Coord]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == BLACK:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != BLACK:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(coords)

    return components


def _order_path(coords: list[Coord]) -> list[Coord]:
    coord_set = set(coords)
    neighbors: dict[Coord, list[Coord]] = {}
    endpoints: list[Coord] = []

    for r, c in coords:
        ns = [
            (nr, nc)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
            if (nr, nc) in coord_set
        ]
        neighbors[(r, c)] = ns
        degree = len(ns)
        assert degree in (1, 2), "expected a non-branching path"
        if degree == 1:
            endpoints.append((r, c))

    assert len(coords) >= 3, "path must include at least one non-endpoint cell"
    assert len(endpoints) == 2, "expected exactly two path endpoints"

    ordered = [endpoints[0]]
    prev: Coord | None = None
    curr = endpoints[0]
    while True:
        next_steps = [nxt for nxt in neighbors[curr] if nxt != prev]
        if not next_steps:
            break
        assert len(next_steps) == 1, "path traversal must be unambiguous"
        prev, curr = curr, next_steps[0]
        ordered.append(curr)

    assert len(ordered) == len(coords), "path traversal missed cells"
    assert ordered[-1] == endpoints[1], "path must terminate at the second endpoint"
    return ordered


def _count_turns(path: list[Coord]) -> int:
    axes: list[int] = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr = int(r1 - r0)
        dc = int(c1 - c0)
        assert abs(dr) + abs(dc) == 1, "path steps must be orthogonal"
        axes.append(0 if dr != 0 else 1)
    return sum(a0 != a1 for a0, a1 in zip(axes, axes[1:]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each orthogonal path body to the endpoint-marker color shared by its turn-count group.

    Input structure:
    - The background is black.
    - Each foreground object is a single-pixel-wide 4-connected orthogonal path with no branches or crossings.
    - Every path has one body color on all non-endpoint cells, while its two endpoint cells are overwritten by a shared marker color.

    Rule:
    - Trace every foreground component as an ordered path from one degree-1 endpoint to the other.
    - Count how many times the step direction switches between horizontal and vertical; paths with the same turn count share the same endpoint-marker color.
    - Recolor every non-endpoint cell of each path to that marker color and keep the two endpoint marker cells unchanged.

    Key insight:
    - The visible endpoint markers encode a hidden structural property of the path: the number of right-angle bends along the traced route.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    turn_to_marker: dict[int, int] = {}
    components = _foreground_components(g)
    assert components, "expected at least one path"

    for coords in components:
        path = _order_path(coords)
        marker = int(g[path[0]])
        assert int(g[path[-1]]) == marker, "path endpoints must share a marker color"

        body_colors = {int(g[r, c]) for r, c in path[1:-1]}
        assert len(body_colors) == 1, "each path body must be monochrome"
        body_color = next(iter(body_colors))
        assert body_color != marker, "body and endpoint marker colors must differ"

        turns = _count_turns(path)
        if turns in turn_to_marker:
            assert turn_to_marker[turns] == marker, "equal-turn paths must share a marker color"
        else:
            turn_to_marker[turns] = marker

        for r, c in path[1:-1]:
            out[r, c] = marker

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    pieces = [cuts[0]]
    for a, b in zip(cuts, cuts[1:]):
        pieces.append(b - a)
    pieces.append(total - cuts[-1])
    return pieces


def _build_path(turns: int) -> list[Coord]:
    if turns == 0:
        steps = random.randint(2, 7)
        if random.choice((False, True)):
            path = [(0, c) for c in range(steps + 1)]
        else:
            path = [(r, 0) for r in range(steps + 1)]
    else:
        run_count = turns + 1
        start_horizontal = random.choice((False, True))
        horiz_runs = (run_count + (1 if start_horizontal else 0)) // 2
        vert_runs = run_count - horiz_runs
        horiz_total = random.randint(horiz_runs, horiz_runs + 4)
        vert_total = random.randint(vert_runs, vert_runs + 4)
        horiz_lengths = _positive_composition(horiz_total, horiz_runs)
        vert_lengths = _positive_composition(vert_total, vert_runs)

        path = [(0, 0)]
        r = 0
        c = 0
        hi = 0
        vi = 0
        for run_idx in range(run_count):
            horizontal = (run_idx % 2 == 0) == start_horizontal
            if horizontal:
                length = horiz_lengths[hi]
                hi += 1
                for _ in range(length):
                    c += 1
                    path.append((r, c))
            else:
                length = vert_lengths[vi]
                vi += 1
                for _ in range(length):
                    r += 1
                    path.append((r, c))

    h = max(r for r, _ in path) + 1
    w = max(c for _, c in path) + 1
    if random.choice((False, True)):
        path = [(h - 1 - r, c) for r, c in path]
    if random.choice((False, True)):
        path = [(r, w - 1 - c) for r, c in path]
    if random.choice((False, True)):
        path = list(reversed(path))
    return path


def _render_patch(path: list[Coord], marker_color: int, body_color: int) -> tuple[np.ndarray, np.ndarray]:
    h = max(r for r, _ in path) + 1
    w = max(c for _, c in path) + 1
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)

    for r, c in path:
        out[r, c] = marker_color
    for r, c in path[1:-1]:
        inp[r, c] = body_color

    r0, c0 = path[0]
    r1, c1 = path[-1]
    inp[r0, c0] = marker_color
    inp[r1, c1] = marker_color
    return inp, out


def _pack_patches(patches: list[dict[str, object]]) -> tuple[list[tuple[dict[str, object], int, int]], int, int]:
    ordered = patches[:]
    random.shuffle(ordered)

    min_limit = max(int(patch["width"]) for patch in ordered)
    row_limit = random.randint(max(14, min_limit), 26)
    placements: list[tuple[dict[str, object], int, int]] = []

    row_y = 0
    row_x = 0
    row_h = 0
    packed_w = 0
    for patch in ordered:
        h = int(patch["height"])
        w = int(patch["width"])
        gap = 0 if row_x == 0 else random.randint(2, 4)
        if row_x > 0 and row_x + gap + w > row_limit:
            row_y += row_h + random.randint(2, 4)
            row_x = 0
            row_h = 0
            gap = 0
        x = row_x + gap
        placements.append((patch, row_y, x))
        row_x = x + w
        row_h = max(row_h, h)
        packed_w = max(packed_w, row_x)

    packed_h = row_y + row_h
    return placements, packed_h, packed_w


def _sample_group_counts(path_count: int, group_count: int) -> list[int]:
    counts = [1] * group_count
    remaining = path_count - group_count
    while remaining > 0:
        counts[random.randrange(group_count)] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(4, 5)
    group_count = random.randint(2, min(4, path_count - 1))
    turn_counts = random.sample(TURN_OPTIONS, group_count)
    group_sizes = _sample_group_counts(path_count, group_count)

    color_pool = PALETTE[:]
    random.shuffle(color_pool)
    marker_colors = color_pool[:group_count]
    body_colors = color_pool[group_count : group_count + path_count]
    if len(body_colors) != path_count:
        return None

    patches: list[dict[str, object]] = []
    body_idx = 0
    for turns, marker_color, copies in zip(turn_counts, marker_colors, group_sizes):
        for _ in range(copies):
            path = _build_path(int(turns))
            inp_patch, out_patch = _render_patch(path, int(marker_color), int(body_colors[body_idx]))
            body_idx += 1
            h, w = map(int, inp_patch.shape)
            patches.append(
                {
                    "input": inp_patch,
                    "output": out_patch,
                    "height": h,
                    "width": w,
                    "turns": int(turns),
                    "marker": int(marker_color),
                }
            )

    placements, packed_h, packed_w = _pack_patches(patches)
    if packed_h > 30 or packed_w > 30:
        return None

    final_h = random.randint(packed_h, min(30, packed_h + 4))
    final_w = random.randint(packed_w, min(30, packed_w + 4))
    top = random.randint(0, final_h - packed_h)
    left = random.randint(0, final_w - packed_w)

    inp = np.full((final_h, final_w), BLACK, dtype=int)
    out = np.full((final_h, final_w), BLACK, dtype=int)
    for patch, y, x in placements:
        patch_in = patch["input"]
        patch_out = patch["output"]
        h, w = patch_in.shape
        rr = top + y
        cc = left + x
        inp_dst = inp[rr : rr + h, cc : cc + w]
        out_dst = out[rr : rr + h, cc : cc + w]
        if np.any((inp_dst != BLACK) & (patch_in != BLACK)):
            return None
        if np.any((out_dst != BLACK) & (patch_out != BLACK)):
            return None
        inp_dst[patch_in != BLACK] = patch_in[patch_in != BLACK]
        out_dst[patch_out != BLACK] = patch_out[patch_out != BLACK]

    repeated_turns = {}
    for patch in patches:
        turns = int(patch["turns"])
        repeated_turns[turns] = repeated_turns.get(turns, 0) + 1
    if max(repeated_turns.values()) < 2:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
