from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

GRID_MIN = 12
GRID_MAX = 30
MAX_EXTRA_PAD = 6

MIN_TOTAL_FRAMES = 2
MAX_TOTAL_FRAMES = 7
MAX_ROOTS = 3
MAX_DEPTH = 4
MAX_CHILDREN = 2

LEAF_H_CHOICES = (3, 3, 4, 4, 5, 5, 6)
LEAF_W_CHOICES = (3, 3, 4, 4, 5, 5, 6)
CHILD_GAP_RANGE = (1, 3)
ROOT_GAP_RANGE = (2, 5)
LAYOUT_EXTRA_SPAN = 2

GEN_RETRIES = 400


class _GenNode:
    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.color = BG
        self.children: list[_GenNode] = []
        self.h = 0
        self.w = 0
        self.rel = (0, 0)
        self.pos = (0, 0)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _strictly_contains(outer: dict[str, int], inner: dict[str, int]) -> bool:
    return (
        int(outer["r0"]) < int(inner["r0"])
        and int(outer["c0"]) < int(inner["c0"])
        and int(outer["r1"]) > int(inner["r1"])
        and int(outer["c1"]) > int(inner["c1"])
    )


def _find_frames(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames: list[dict[str, int]] = []
    for coords in _components(g != BG):
        colors = {int(g[r, c]) for r, c in coords.tolist()}
        assert len(colors) == 1, "each frame component must be monochrome"
        color = next(iter(colors))

        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "each component must span a hollow rectangle"

        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.array_equal(patch, border), "each component must be a one-pixel-thick rectangular frame"

        frames.append(
            {
                "color": int(color),
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "depth": 0,
            }
        )

    assert frames, "expected at least one frame"
    return frames


def _assign_depths(frames: list[dict[str, int]]) -> None:
    memo: dict[int, int] = {}

    def depth_of(idx: int) -> int:
        if idx in memo:
            return memo[idx]
        frame = frames[idx]
        containers = [j for j, other in enumerate(frames) if _strictly_contains(other, frame)]
        if not containers:
            memo[idx] = 1
        else:
            parent = min(
                containers,
                key=lambda j: (frames[j]["r1"] - frames[j]["r0"] + 1) * (frames[j]["c1"] - frames[j]["c0"] + 1),
            )
            memo[idx] = depth_of(parent) + 1
        return memo[idx]

    for i, frame in enumerate(frames):
        frame["depth"] = int(depth_of(i))


def _paint_patterns(base: np.ndarray, frames: list[dict[str, int]]) -> np.ndarray:
    out = base.copy()
    for frame in sorted(frames, key=lambda item: int(item["depth"])):
        r0, c0, r1, c1 = (int(frame[k]) for k in ("r0", "c0", "r1", "c1"))
        step = int(frame["depth"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue

        mask = np.zeros(inner.shape, dtype=bool)
        mask[::step, ::step] = True

        blocked = np.zeros(inner.shape, dtype=bool)
        for other in frames:
            if other is frame or not _strictly_contains(frame, other):
                continue
            rr0 = max(r0 + 1, int(other["r0"])) - (r0 + 1)
            rr1 = min(r1 - 1, int(other["r1"])) - (r0 + 1)
            cc0 = max(c0 + 1, int(other["c0"])) - (c0 + 1)
            cc1 = min(c1 - 1, int(other["c1"])) - (c0 + 1)
            if rr0 <= rr1 and cc0 <= cc1:
                blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True

        inner[mask & ~blocked] = int(frame["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame interior with a depth-controlled lattice of its own color.

    Input structure:
    - The background is black.
    - The foreground consists of one-pixel-thick hollow rectangular frames in various colors.
    - Frames may be nested or disjoint, and every nested child is separated from its parent by at least one black pixel.
    - Each frame interior is black in the input, except where deeper frames occupy part of that interior.

    Rule:
    - Compute the nesting depth D of every frame: outermost frames have depth 1, frames inside them have depth 2, and so on.
    - For each frame, look at its interior region excluding the border pixels.
    - Starting from the interior's top-left cell, paint the frame's own color at positions whose row and column offsets are both multiples of D, so the sampling step is exactly D in both directions.
    - If a candidate cell lies inside any deeper frame's full bbox, including that deeper frame's border, skip it so the deeper frame keeps priority over the outer fill.

    Key insight:
    - The rectangles form a containment tree, and the depth in that tree directly determines how sparse the interior lattice should be.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    _assign_depths(frames)
    return _paint_patterns(g, frames)


def _weighted_choice(rng: random.Random, items: list[_GenNode], weights: list[int]) -> _GenNode:
    total = sum(weights)
    pick = rng.randrange(total)
    running = 0
    for item, weight in zip(items, weights):
        running += weight
        if pick < running:
            return item
    return items[-1]


def _build_topology(rng: random.Random) -> tuple[list[_GenNode], list[_GenNode]] | None:
    total_frames = rng.randint(MIN_TOTAL_FRAMES, MAX_TOTAL_FRAMES)
    max_roots = min(MAX_ROOTS, total_frames - 1)
    root_count = rng.randint(1, max_roots)
    roots = [_GenNode(depth=1) for _ in range(root_count)]
    all_nodes = roots[:]

    chain_root = rng.choice(roots)
    max_chain = min(MAX_DEPTH, total_frames - root_count + 1)
    chain_len = rng.randint(2, max_chain)
    current = chain_root
    for _ in range(chain_len - 1):
        child = _GenNode(depth=current.depth + 1)
        current.children.append(child)
        all_nodes.append(child)
        current = child

    used = root_count + chain_len - 1
    for _ in range(total_frames - used):
        candidates = [node for node in all_nodes if node.depth < MAX_DEPTH and len(node.children) < MAX_CHILDREN]
        if not candidates:
            return None
        weights = [3 if node.children else 1 for node in candidates]
        parent = _weighted_choice(rng, candidates, weights)
        child = _GenNode(depth=parent.depth + 1)
        parent.children.append(child)
        all_nodes.append(child)

    colors = rng.sample(COLOR_POOL, len(all_nodes))
    for node, color in zip(all_nodes, colors):
        node.color = int(color)
    return roots, all_nodes


def _arrange_boxes(
    dims: list[tuple[int, int]],
    rng: random.Random,
    gap_range: tuple[int, int],
) -> tuple[list[tuple[int, int]], int, int]:
    if len(dims) == 1:
        h, w = dims[0]
        return [(0, 0)], h, w

    mode = rng.choice(("horizontal", "vertical", "diagonal"))
    gaps = [rng.randint(*gap_range) for _ in range(len(dims) - 1)]
    placements: list[tuple[int, int]] = []

    if mode == "horizontal":
        group_h = max(h for h, _ in dims) + rng.randint(0, LAYOUT_EXTRA_SPAN)
        group_w = sum(w for _, w in dims) + sum(gaps)
        c = 0
        for (h, w), gap in zip(dims, gaps + [0]):
            placements.append((rng.randint(0, group_h - h), c))
            c += w + gap
        return placements, group_h, group_w

    if mode == "vertical":
        group_h = sum(h for h, _ in dims) + sum(gaps)
        group_w = max(w for _, w in dims) + rng.randint(0, LAYOUT_EXTRA_SPAN)
        r = 0
        for (h, w), gap in zip(dims, gaps + [0]):
            placements.append((r, rng.randint(0, group_w - w)))
            r += h + gap
        return placements, group_h, group_w

    group_h = sum(h for h, _ in dims) + sum(gaps)
    group_w = sum(w for _, w in dims) + sum(gaps)
    r = 0
    c = 0
    for (h, w), gap in zip(dims, gaps + [0]):
        placements.append((r, c))
        r += h + gap
        c += w + gap
    return placements, group_h, group_w


def _realize_subtree(node: _GenNode, rng: random.Random) -> bool:
    for child in node.children:
        if not _realize_subtree(child, rng):
            return False

    if not node.children:
        node.h = rng.choice(LEAF_H_CHOICES)
        node.w = rng.choice(LEAF_W_CHOICES)
        return True

    ordered = node.children[:]
    rng.shuffle(ordered)
    dims = [(child.h, child.w) for child in ordered]
    placements, group_h, group_w = _arrange_boxes(dims, rng, CHILD_GAP_RANGE)

    min_h = group_h + 4
    min_w = group_w + 4
    max_h = min(GRID_MAX - 2, min_h + 4)
    max_w = min(GRID_MAX - 2, min_w + 4)
    if min_h > max_h or min_w > max_w:
        return False

    node.h = rng.randint(min_h, max_h)
    node.w = rng.randint(min_w, max_w)
    top = rng.randint(2, node.h - group_h - 2)
    left = rng.randint(2, node.w - group_w - 2)
    for child, (r, c) in zip(ordered, placements):
        child.rel = (top + r, left + c)
    return True


def _flatten(node: _GenNode, base_r: int, base_c: int) -> list[_GenNode]:
    node.pos = (base_r, base_c)
    items = [node]
    for child in node.children:
        cr, cc = child.rel
        items.extend(_flatten(child, base_r + cr, base_c + cc))
    return items


def _draw_frame(grid: np.ndarray, frame: dict[str, int]) -> None:
    r0, c0, r1, c1 = (int(frame[k]) for k in ("r0", "c0", "r1", "c1"))
    color = int(frame["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frames_from_nodes(nodes: list[_GenNode]) -> list[dict[str, int]]:
    frames = []
    for node in nodes:
        r0, c0 = node.pos
        frames.append(
            {
                "color": int(node.color),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r0 + node.h - 1),
                "c1": int(c0 + node.w - 1),
                "depth": int(node.depth),
            }
        )
    return frames


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    topology = _build_topology(rng)
    if topology is None:
        return None
    roots, _all_nodes = topology

    for root in roots:
        if not _realize_subtree(root, rng):
            return None

    ordered_roots = roots[:]
    rng.shuffle(ordered_roots)
    root_dims = [(root.h, root.w) for root in ordered_roots]
    placements, group_h, group_w = _arrange_boxes(root_dims, rng, ROOT_GAP_RANGE)
    if group_h > GRID_MAX or group_w > GRID_MAX:
        return None

    extra_h_lo = max(0, GRID_MIN - group_h)
    extra_w_lo = max(0, GRID_MIN - group_w)
    extra_h_hi = min(MAX_EXTRA_PAD, GRID_MAX - group_h)
    extra_w_hi = min(MAX_EXTRA_PAD, GRID_MAX - group_w)
    if extra_h_lo > extra_h_hi or extra_w_lo > extra_w_hi:
        return None

    extra_h = rng.randint(extra_h_lo, extra_h_hi)
    extra_w = rng.randint(extra_w_lo, extra_w_hi)
    top_pad = rng.randint(0, extra_h)
    left_pad = rng.randint(0, extra_w)

    h = group_h + extra_h
    w = group_w + extra_w
    grid = np.full((h, w), BG, dtype=int)

    flat_nodes: list[_GenNode] = []
    for root, (r, c) in zip(ordered_roots, placements):
        flat_nodes.extend(_flatten(root, top_pad + r, left_pad + c))

    frames = _frames_from_nodes(flat_nodes)
    for frame in frames:
        _draw_frame(grid, frame)

    expected = _paint_patterns(grid, frames)
    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
