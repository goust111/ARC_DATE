from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste
import random


GREY = 5
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _in_bbox(p: tuple[int, int], b: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return r0 <= r <= r1 and c0 <= c <= c1


def _outside_both_frame_axes(p: tuple[int, int], b: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return (r < r0 - 1 or r > r1 + 1) and (c < c0 - 1 or c > c1 + 1)


def _frame_clamp(p: tuple[int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = p
    r0, c0, r1, c1 = b
    return min(max(r, r0 - 1), r1 + 1), min(max(c, c0 - 1), c1 + 1)


def _ray_target(
    g: np.ndarray, p: tuple[int, int], b: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    if not _outside_both_frame_axes(p, b):
        return None

    h, w = g.shape
    hits: list[tuple[int, tuple[int, int]]] = []
    for dr, dc in DIRS8:
        r, c = p
        prev = p
        steps = 0
        while True:
            r += dr
            c += dc
            steps += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(g[r, c]) == GREY:
                if _in_bbox((r, c), b):
                    hits.append((steps, prev))
                break
            prev = (r, c)

    if not hits:
        return None
    min_steps = min(steps for steps, _target in hits)
    nearest = [target for steps, target in hits if steps == min_steps]
    assert len(set(nearest)) == 1, "nearest exact sightline target must be unique"
    return nearest[0]


def _grey_blocks(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    comps = find_components(g, bg=bg, colors=[GREY], connectivity=4)
    assert comps, "expected grey 2x2 anchor blocks"

    blocks: list[tuple[int, int, int, int]] = []
    for comp in comps:
        b = comp["bbox"]
        assert comp["area"] == 4 and bbox_h(b) == 2 and bbox_w(b) == 2, "grey anchors must be solid 2x2 blocks"
        blocks.append(b)
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each isolated colored pixel into the one-cell frame around a grey 2x2 block.

    Input structure:
    - Background is the dominant color.
    - Grey foreground objects are solid 2x2 anchor blocks.
    - Every other non-background, non-grey cell is a colored singleton source.

    Rule:
    - Keep the grey anchors fixed and clear all colored singleton source pixels.
    - For each singleton and each anchor, the normal destination is the cell reached by
      clamping the pixel into the anchor's one-cell outer frame.
    - If a diagonally outside singleton has an exact 8-direction sightline into an anchor,
      also allow the cell just before the first grey pixel on that ray.
    - Choose distinct destinations that minimize total king-move distance; at equal
      distance prefer exact sightlines, then the shorter Manhattan movement.

    Key insight: The grey 2x2 blocks define magnetic frames; conflicts are resolved
    structurally by nearest-frame assignment rather than by color identity.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    blocks = _grey_blocks(g, bg)
    dots = [tuple(map(int, p)) for p in np.argwhere((g != bg) & (g != GREY))]
    assert dots, "expected colored singleton pixels"

    candidates: list[tuple[int, tuple[int, int], int]] = []
    for i, p in enumerate(dots):
        for b in blocks:
            target = _frame_clamp(p, b)
            candidates.append((i, target, 1))
            ray = _ray_target(g, p, b)
            if ray is not None and ray != target:
                candidates.append((i, ray, 0))

    targets = sorted({target for _i, target, _kind in candidates})
    assert len(targets) >= len(dots), "expected enough destination frame cells"
    target_index = {target: i for i, target in enumerate(targets)}

    h, w = g.shape
    kind_weight = 100
    cheb_weight = 10_000
    assert 2 * max(h, w) < kind_weight < cheb_weight, "cost weights must preserve lexicographic geometry"

    costs = np.full((len(dots), len(targets)), 10**9, dtype=int)
    for i, target, kind_penalty in candidates:
        r, c = target
        assert 0 <= r < h and 0 <= c < w, "destination must stay inside the grid"
        assert int(g[r, c]) != GREY, "destination cannot be inside a grey anchor"
        dr = abs(r - dots[i][0])
        dc = abs(c - dots[i][1])
        cheb = max(dr, dc)
        manhattan = dr + dc
        cost = cheb * cheb_weight + kind_penalty * kind_weight + manhattan
        j = target_index[target]
        costs[i, j] = min(costs[i, j], cost)

    rows, cols = linear_sum_assignment(costs)
    assert len(rows) == len(dots), "every colored pixel must receive a destination"
    assert all(costs[r, c] < 10**9 for r, c in zip(rows, cols)), "all assignments must be valid"

    out = g.copy()
    for r, c in dots:
        out[r, c] = bg
    for r, c in zip(rows, cols):
        out[targets[c]] = int(g[dots[r]])
    return out

# --------------- generator (o5x) ---------------

# Color palette and geometry primitives.
BACKGROUND = 0
GREY = 5
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

# Grid and block layout.
SMALL_GRID_SIZE = 10
LARGE_GRID_SIZE = 15
LARGE_GRID_BLOCK_THRESHOLD = 5
LARGE_GRID_PROB = 0.5
MIN_BLOCKS = 3
MAX_BLOCKS = 6
BLOCK_SIZE = 2
MIN_BLOCK_CHEB_GAP = 3
EDGE_BLOCK_PROB = 0.25
EDGE_BLOCK_QUOTA = 1

# Singleton allocation and inverse placement.
FRAME_RADIUS = 1
MIN_SINGLETONS = 5
MAX_SINGLETONS = 12
MAX_SINGLETONS_FOR_MIN_BLOCKS = 9
MAX_SINGLETONS_PER_BLOCK = 5
SOURCE_DISTANCE_OPTIONS = (1, 2, 3, 4)
MIN_DISTINCT_COLORS = 4
MAX_DISTINCT_COLORS = 7
MAX_COLOR_REPEATS = 4

# Assignment scoring and retry limits.
CHEB_COST_WEIGHT = 10_000
KIND_COST_WEIGHT = 100
UNREACHABLE_COST = 10**9
GENERATION_RETRIES = 300


def _touches_edge(b: tuple[int, int, int, int], h: int, w: int) -> bool:
    r0, c0, r1, c1 = b
    return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1


def _axis_gap(a0: int, a1: int, b0: int, b1: int) -> int:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0


def _bbox_cheb_dist(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return max(_axis_gap(a[0], a[2], b[0], b[2]), _axis_gap(a[1], a[3], b[1], b[3]))


def _place_blocks(h: int, w: int, n: int) -> list[tuple[int, int, int, int]] | None:
    edge_quota = EDGE_BLOCK_QUOTA if random.random() < EDGE_BLOCK_PROB else 0
    anchors = [(r, c) for r in range(h - BLOCK_SIZE + 1) for c in range(w - BLOCK_SIZE + 1)]
    random.shuffle(anchors)
    candidates = [(r, c, r + BLOCK_SIZE - 1, c + BLOCK_SIZE - 1) for r, c in anchors]

    def search(
        start: int,
        chosen: list[tuple[int, int, int, int]],
        edge_count: int,
    ) -> list[tuple[int, int, int, int]] | None:
        if len(chosen) == n:
            return chosen
        if len(candidates) - start < n - len(chosen):
            return None

        for j in range(start, len(candidates)):
            b = candidates[j]
            next_edges = edge_count + int(_touches_edge(b, h, w))
            if next_edges > edge_quota:
                continue
            if all(_bbox_cheb_dist(b, old) >= MIN_BLOCK_CHEB_GAP for old in chosen):
                out = search(j + 1, chosen + [b], next_edges)
                if out is not None:
                    return out
        return None

    return search(0, [], 0)


def _frame_cells(b: tuple[int, int, int, int], h: int, w: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = b
    cells = []
    for r in range(r0 - FRAME_RADIUS, r1 + FRAME_RADIUS + 1):
        for c in range(c0 - FRAME_RADIUS, c1 + FRAME_RADIUS + 1):
            if 0 <= r < h and 0 <= c < w and not (r0 <= r <= r1 and c0 <= c <= c1):
                cells.append((r, c))
    return cells


def _outward_dir(t: tuple[int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = t
    r0, c0, r1, c1 = b
    dr = -1 if r < r0 else 1 if r > r1 else 0
    dc = -1 if c < c0 else 1 if c > c1 else 0
    return dr, dc


def _in_bbox(p: tuple[int, int], b: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return r0 <= r <= r1 and c0 <= c <= c1


def _outside_both_frame_axes(p: tuple[int, int], b: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = b
    return (r < r0 - FRAME_RADIUS or r > r1 + FRAME_RADIUS) and (
        c < c0 - FRAME_RADIUS or c > c1 + FRAME_RADIUS
    )


def _frame_clamp(p: tuple[int, int], b: tuple[int, int, int, int]) -> tuple[int, int]:
    r, c = p
    r0, c0, r1, c1 = b
    return (
        min(max(r, r0 - FRAME_RADIUS), r1 + FRAME_RADIUS),
        min(max(c, c0 - FRAME_RADIUS), c1 + FRAME_RADIUS),
    )


def _ray_target(
    grey_grid: np.ndarray,
    p: tuple[int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    if not _outside_both_frame_axes(p, b):
        return None

    h, w = grey_grid.shape
    hits = []
    for dr, dc in DIRS8:
        r, c = p
        prev = p
        steps = 0
        while True:
            r += dr
            c += dc
            steps += 1
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(grey_grid[r, c]) == GREY:
                if _in_bbox((r, c), b):
                    hits.append((steps, prev))
                break
            prev = (r, c)

    if not hits:
        return None
    min_steps = min(steps for steps, _target in hits)
    nearest = [target for steps, target in hits if steps == min_steps]
    if len(set(nearest)) != 1:
        return None
    return nearest[0]


def _candidate_costs(
    grey_grid: np.ndarray,
    p: tuple[int, int],
    blocks: list[tuple[int, int, int, int]],
) -> dict[tuple[int, int], int]:
    costs: dict[tuple[int, int], int] = {}
    for b in blocks:
        entries = [(_frame_clamp(p, b), 1)]
        ray = _ray_target(grey_grid, p, b)
        if ray is not None and ray != entries[0][0]:
            entries.append((ray, 0))
        for target, kind_penalty in entries:
            dr = abs(target[0] - p[0])
            dc = abs(target[1] - p[1])
            cost = max(dr, dc) * CHEB_COST_WEIGHT + kind_penalty * KIND_COST_WEIGHT + dr + dc
            costs[target] = min(costs.get(target, UNREACHABLE_COST), cost)
    return costs


def _unique_best_target(
    grey_grid: np.ndarray,
    p: tuple[int, int],
    target: tuple[int, int],
    blocks: list[tuple[int, int, int, int]],
) -> bool:
    costs = _candidate_costs(grey_grid, p, blocks)
    if target not in costs:
        return False
    target_cost = costs[target]
    return all(other == target or target_cost < cost for other, cost in costs.items())


def _source_options(
    grey_grid: np.ndarray,
    target: tuple[int, int],
    block: tuple[int, int, int, int],
    blocks: list[tuple[int, int, int, int]],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grey_grid.shape
    dr, dc = _outward_dir(target, block)
    distances = list(SOURCE_DISTANCE_OPTIONS)
    random.shuffle(distances)
    opts = []
    for dist in distances:
        p = (target[0] + dr * dist, target[1] + dc * dist)
        if not (0 <= p[0] < h and 0 <= p[1] < w):
            continue
        if p in forbidden or int(grey_grid[p]) != BACKGROUND:
            continue
        if _unique_best_target(grey_grid, p, target, blocks):
            opts.append((dist, p))
    random.shuffle(opts)
    opts.sort(reverse=True)
    return [p for _dist, p in opts]


def _assign_sources(
    grey_grid: np.ndarray,
    targets: list[tuple[tuple[int, int], tuple[int, int, int, int], int]],
    blocks: list[tuple[int, int, int, int]],
    all_frames: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
    random.shuffle(targets)
    base_forbidden = set(all_frames)

    def search(
        i: int,
        used: set[tuple[int, int]],
        chosen: list[tuple[tuple[int, int], tuple[int, int], int]],
    ) -> list[tuple[tuple[int, int], tuple[int, int], int]] | None:
        if i == len(targets):
            return chosen
        target, block, color = targets[i]
        opts = _source_options(grey_grid, target, block, blocks, base_forbidden | used)
        for src in opts:
            out = search(i + 1, used | {src}, chosen + [(src, target, color)])
            if out is not None:
                return out
        return None

    return search(0, set(), [])


def _colors(n: int) -> list[int]:
    distinct = random.randint(MIN_DISTINCT_COLORS, min(MAX_DISTINCT_COLORS, n, len(COLORS)))
    palette = random.sample(COLORS, distinct)
    counts = {c: 1 for c in palette}
    out = palette[:]
    while len(out) < n:
        choices = [c for c in palette if counts[c] < MAX_COLOR_REPEATS]
        c = random.choice(choices)
        counts[c] += 1
        out.append(c)
    random.shuffle(out)
    return out


def _alloc_counts(caps: list[int], total: int) -> list[int] | None:
    if sum(caps) < total:
        return None
    counts = [0] * len(caps)
    for _ in range(total):
        choices = [i for i, cap in enumerate(caps) if counts[i] < cap]
        if not choices:
            return None
        counts[random.choice(choices)] += 1
    return counts


def _draw_blocks(size: int, blocks: list[tuple[int, int, int, int]]) -> np.ndarray:
    grey_grid = np.zeros((size, size), dtype=int)
    block_patch = np.full((BLOCK_SIZE, BLOCK_SIZE), GREY, dtype=int)
    for r0, c0, _r1, _c1 in blocks:
        paste(grey_grid, block_patch, r0, c0)
    return grey_grid


def _prepare_viable_targets(
    grey_grid: np.ndarray,
    blocks: list[tuple[int, int, int, int]],
    size: int,
) -> tuple[list[list[tuple[int, int]]], set[tuple[int, int]]]:
    """Return invertible frame targets plus the full forbidden frame set."""
    frames_by_block = []
    all_frames = set()
    for b in blocks:
        cells = _frame_cells(b, size, size)
        frames_by_block.append(cells)
        all_frames.update(cells)

    viable_by_block = []
    for b, cells in zip(blocks, frames_by_block):
        viable = [
            t
            for t in cells
            if _source_options(grey_grid, t, b, blocks, all_frames)
        ]
        random.shuffle(viable)
        viable_by_block.append(viable)
    return viable_by_block, all_frames


def _choose_singleton_count(n_blocks: int) -> int:
    max_singletons = min(MAX_SINGLETONS, MAX_SINGLETONS_PER_BLOCK * n_blocks)
    if n_blocks == MIN_BLOCKS:
        max_singletons = min(max_singletons, MAX_SINGLETONS_FOR_MIN_BLOCKS)
    return random.randint(MIN_SINGLETONS, max_singletons)


def _build_target_specs(
    grey_grid: np.ndarray,
    blocks: list[tuple[int, int, int, int]],
    viable_by_block: list[list[tuple[int, int]]],
    counts: list[int],
    colors: list[int],
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int, int, int], int]]]:
    output = grey_grid.copy()
    target_specs = []
    k = 0
    for b, viable, count in zip(blocks, viable_by_block, counts):
        for target in random.sample(viable, count):
            color = colors[k]
            k += 1
            output[target] = color
            target_specs.append((target, b, color))
    return output, target_specs


def _gen_attempt() -> np.ndarray | None:
    n_blocks = random.randint(MIN_BLOCKS, MAX_BLOCKS)
    size = LARGE_GRID_SIZE if n_blocks >= LARGE_GRID_BLOCK_THRESHOLD or random.random() < LARGE_GRID_PROB else SMALL_GRID_SIZE
    blocks = _place_blocks(size, size, n_blocks)
    if blocks is None:
        return None

    grey_grid = _draw_blocks(size, blocks)
    viable_by_block, all_frames = _prepare_viable_targets(grey_grid, blocks, size)

    n_singletons = _choose_singleton_count(n_blocks)
    caps = [min(MAX_SINGLETONS_PER_BLOCK, len(v)) for v in viable_by_block]
    counts = _alloc_counts(caps, n_singletons)
    if counts is None:
        return None

    colors = _colors(n_singletons)
    output, target_specs = _build_target_specs(grey_grid, blocks, viable_by_block, counts, colors)
    assigned = _assign_sources(grey_grid, target_specs, blocks, all_frames)
    if assigned is None:
        return None

    inp = grey_grid.copy()
    for src, _target, color in assigned:
        inp[src] = color

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        out = _gen_attempt()
        if out is not None:
            return out
    raise RuntimeError("failed to generate valid 1b8318e3 input")
