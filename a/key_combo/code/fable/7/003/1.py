from __future__ import annotations

from itertools import combinations
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, find_components, get_bg, neighbors4


BG = 0
PALETTE = list(range(1, 10))
MIN_FRAMES = 3
MAX_FRAMES = 5
MIN_GRID = 20
MAX_GRID = 30
MIN_FRAME_DIM = 8
MAX_FRAME_DIM = 16
GEN_ATTEMPTS = 500


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _random_bbox(rng: random.Random, height: int, width: int) -> tuple[int, int, int, int]:
    frame_h = rng.randint(MIN_FRAME_DIM, min(MAX_FRAME_DIM, height - 2))
    frame_w = rng.randint(MIN_FRAME_DIM, min(MAX_FRAME_DIM, width - 2))
    r0 = rng.randint(1, height - frame_h - 1)
    c0 = rng.randint(1, width - frame_w - 1)
    return (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)


def _paint_frames(
    shape: tuple[int, int], order: list[int], bboxes: dict[int, tuple[int, int, int, int]]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    grid = np.full(shape, BG, dtype=int)
    masks = {color: _frame_mask(shape, bboxes[color]) for color in order}
    for color in order:
        grid[masks[color]] = color
    return grid, masks


def _visible_bbox_matches(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> bool:
    visible = np.argwhere(grid == color)
    if visible.size == 0:
        return False
    if bbox_coords(visible) != bbox:
        return False
    r0, c0, r1, c1 = bbox
    return (
        np.any(grid[r0, c0 : c1 + 1] == color)
        and np.any(grid[r1, c0 : c1 + 1] == color)
        and np.any(grid[r0 : r1 + 1, c0] == color)
        and np.any(grid[r0 : r1 + 1, c1] == color)
    )


def _frame_union(masks: dict[int, np.ndarray]) -> np.ndarray:
    union = np.zeros_like(next(iter(masks.values())), dtype=bool)
    for mask in masks.values():
        union |= mask
    return union


def _expand4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _infer_frames(
    grid: np.ndarray,
) -> tuple[int, list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    bg = int(get_bg(grid))
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    assert len(colors) >= 3, "expected at least three frame colors"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each foreground color must be visible"
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame bbox must span a rectangle"
        mask = _frame_mask(grid.shape, bbox)
        assert np.any((grid == color) & mask), "each color must remain visible on its inferred frame"
        bboxes[color] = bbox
        masks[color] = mask
    return bg, colors, bboxes, masks


def _crossing_edges(
    grid: np.ndarray, colors: list[int], masks: dict[int, np.ndarray]
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(colors, 2):
        crossing = masks[a] & masks[b]
        if not crossing.any():
            continue
        visible = {int(v) for v in grid[crossing].tolist() if int(v) != BG}
        assert visible, "every crossing must show a visible frame color"
        assert visible <= {a, b}, "crossings may only show one of the two participating frame colors"
        assert len(visible) == 1, "pairwise crossings must encode a consistent occlusion direction"
        top = next(iter(visible))
        edges.add((b, a) if top == a else (a, b))
    return edges


def _unique_chain(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    succ = {color: set() for color in colors}
    indeg = {color: 0 for color in colors}
    for src, dst in edges:
        if dst not in succ[src]:
            succ[src].add(dst)
            indeg[dst] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zeros = sorted(color for color in remaining if indeg[color] == 0)
        assert len(zeros) == 1, "crossings must determine a unique full color order"
        current = zeros[0]
        order.append(current)
        remaining.remove(current)
        for nxt in succ[current]:
            indeg[nxt] -= 1
    return order


def _find_groups(grid: np.ndarray, bg: int, union: np.ndarray) -> list[dict]:
    off_frame = np.full_like(grid, bg)
    off_frame[~union] = grid[~union]
    groups = find_components(off_frame, bg=bg, connectivity=4, monochromatic=True)
    assert groups, "expected at least one off-frame pixel group"
    return groups


def _infer_order_info(grid: np.ndarray) -> dict:
    bg, colors, bboxes, masks = _infer_frames(grid)
    edges = _crossing_edges(grid, colors, masks)
    order = _unique_chain(colors, edges)
    union = _frame_union(masks)
    return {
        "bg": bg,
        "colors": colors,
        "bboxes": bboxes,
        "masks": masks,
        "edges": edges,
        "order": order,
        "union": union,
    }


def _analyze(grid: np.ndarray) -> dict:
    info = _infer_order_info(grid)
    groups = _find_groups(grid, info["bg"], info["union"])
    return {
        **info,
        "groups": groups,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the overlap order of rectangular frames as a color ladder, then advance each off-frame group along that ladder by its size.

    Input structure:
    - A black background contains several one-pixel-thick axis-aligned monochrome rectangular frames that overlap pairwise.
    - At each frame crossing, the visible color records which frame lies on top; each frame color still reveals its full rectangular bbox.
    - Away from all frame perimeters, there are several 4-connected monochrome pixel groups whose colors are frame colors.

    Rule:
    - Reconstruct each frame from the bbox of its visible color, and at every crossing add a directed relation from the hidden frame color to the visible frame color.
    - Recover the unique full frame order from back to front, then for each off-frame group of size k move k steps forward along that order, wrapping from the front color back to the back color.
    - Keep all rectangular frames unchanged and recolor each whole off-frame group to the destination color reached by its size-based walk.

    Key insight:
    - The overlapping frames are not targets to redraw; their crossings encode a unique occlusion order, and each detached group uses only its area as a cyclic step count along that order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    info = _analyze(g)
    order = info["order"]
    index = {color: i for i, color in enumerate(order)}
    count = len(order)

    out = g.copy()
    for group in info["groups"]:
        color = int(group["color"])
        steps = int(group["area"])
        dst = order[(index[color] + steps) % count]
        coords = group["coords"]
        out[coords[:, 0], coords[:, 1]] = int(dst)
    return out


def _sample_bboxes(
    rng: random.Random, height: int, width: int, order: list[int]
) -> tuple[dict[int, tuple[int, int, int, int]], dict[int, np.ndarray], np.ndarray] | None:
    for _ in range(120):
        bboxes: dict[int, tuple[int, int, int, int]] = {}
        for i, color in enumerate(order):
            placed = False
            for _ in range(120):
                bbox = _random_bbox(rng, height, width)
                candidate_mask = _frame_mask((height, width), bbox)
                if i > 0:
                    prev_mask = _frame_mask((height, width), bboxes[order[i - 1]])
                    if not np.any(candidate_mask & prev_mask):
                        continue
                bboxes[color] = bbox
                placed = True
                break
            if not placed:
                break
        if len(bboxes) != len(order):
            continue

        frame_only, masks = _paint_frames((height, width), order, bboxes)
        stack = np.zeros((height, width), dtype=int)
        for color in order:
            stack += masks[color].astype(int)
        if np.max(stack) > 2:
            continue
        if not all(_visible_bbox_matches(frame_only, color, bboxes[color]) for color in order):
            continue
        try:
            inferred = _infer_order_info(frame_only)
        except AssertionError:
            continue
        if inferred["order"] != order:
            continue
        return bboxes, masks, frame_only
    return None


def _safe_cells_for_color(
    bbox: tuple[int, int, int, int], forbidden: np.ndarray
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    safe: set[tuple[int, int]] = set()
    for r in range(r0 + 2, r1 - 1):
        for c in range(c0 + 2, c1 - 1):
            if not forbidden[r, c]:
                safe.add((r, c))
    return safe


def _grow_group(
    rng: random.Random,
    available: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    size: int,
) -> list[tuple[int, int]] | None:
    starts = [cell for cell in available if cell not in reserved]
    rng.shuffle(starts)
    for start in starts:
        group = [start]
        used = {start}
        frontier = [cell for cell in neighbors4(*start) if cell in available and cell not in reserved]
        while len(group) < size and frontier:
            cell = frontier.pop(rng.randrange(len(frontier)))
            if cell in used or cell in reserved or cell not in available:
                continue
            used.add(cell)
            group.append(cell)
            for nbr in neighbors4(*cell):
                if nbr in available and nbr not in reserved and nbr not in used and nbr not in frontier:
                    frontier.append(nbr)
        if len(group) == size:
            return group
    return None


def _reserve_with_halo(reserved: set[tuple[int, int]], cells: list[tuple[int, int]], shape: tuple[int, int]) -> None:
    height, width = shape
    for r, c in cells:
        reserved.add((r, c))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    reserved.add((rr, cc))


def _group_specs(order: list[int], rng: random.Random) -> list[tuple[int, int]]:
    count = len(order)
    specs = [
        (order[-1], 2),
        (order[0], 2),
        (order[min(1, count - 1)], 1),
    ]

    for color in order:
        if rng.random() < 0.75:
            size = rng.randint(1, count + 2)
            if size % count == 0:
                size += 1
            specs.append((color, min(size, count + 2)))

    extra = rng.randint(1, count)
    for _ in range(extra):
        color = rng.choice(order)
        size = rng.randint(1, count + 2)
        if size % count == 0:
            size += 1
        specs.append((color, min(size, count + 2)))

    rng.shuffle(specs)
    return specs


def _place_groups(
    rng: random.Random,
    frame_grid: np.ndarray,
    order: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    masks: dict[int, np.ndarray],
) -> np.ndarray | None:
    grid = frame_grid.copy()
    union = _frame_union(masks)
    forbidden = _expand4(union)
    safe_by_color = {color: _safe_cells_for_color(bboxes[color], forbidden) for color in order}
    if any(len(cells) < 3 for cells in safe_by_color.values()):
        return None

    reserved: set[tuple[int, int]] = set()
    specs = _group_specs(order, rng)
    placed_groups = 0
    wrap_groups = 0
    changed_groups = 0
    index = {color: i for i, color in enumerate(order)}

    for color, size in specs:
        group = _grow_group(rng, safe_by_color[color], reserved, size)
        if group is None:
            continue
        for r, c in group:
            grid[r, c] = color
        _reserve_with_halo(reserved, group, grid.shape)
        placed_groups += 1
        if index[color] + size >= len(order):
            wrap_groups += 1
        if order[(index[color] + size) % len(order)] != color:
            changed_groups += 1

    if placed_groups < max(4, len(order)):
        return None
    if wrap_groups == 0 or changed_groups < 3:
        return None
    return grid


def _validate_generated(inp: np.ndarray, out: np.ndarray) -> None:
    assert isinstance(inp, np.ndarray) and isinstance(out, np.ndarray)
    assert inp.ndim == 2 and out.ndim == 2
    assert max(inp.shape) <= 30 and max(out.shape) <= 30

    info = _analyze(inp)
    order = info["order"]
    union = info["union"]
    groups = info["groups"]
    assert len(order) >= 3
    assert np.array_equal(inp[union], out[union]), "frames must stay unchanged"
    assert len(groups) >= max(4, len(order)), "need several detached groups"

    index = {color: i for i, color in enumerate(order)}
    wrap_seen = False
    multi_step_seen = False
    changed_seen = False
    for group in groups:
        color = int(group["color"])
        size = int(group["area"])
        dst = order[(index[color] + size) % len(order)]
        coords = group["coords"]
        assert np.all(out[coords[:, 0], coords[:, 1]] == dst), "group recolor must follow the ladder"
        if index[color] + size >= len(order):
            wrap_seen = True
        if size > 1:
            multi_step_seen = True
        if dst != color:
            changed_seen = True
    assert wrap_seen, "generator must include a wrap-around group"
    assert multi_step_seen, "generator must include a multi-step group"
    assert changed_seen, "generator must produce a non-trivial recolor"


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        frame_count = rng.randint(MIN_FRAMES, MAX_FRAMES)
        order = rng.sample(PALETTE, frame_count)
        min_side = max(MIN_GRID, frame_count * 4 + 8)
        height = rng.randint(min_side, MAX_GRID)
        width = rng.randint(min_side, MAX_GRID)

        sampled = _sample_bboxes(rng, height, width, order)
        if sampled is None:
            continue
        bboxes, masks, frame_grid = sampled

        grid = _place_groups(rng, frame_grid, order, bboxes, masks)
        if grid is None:
            continue

        try:
            out = main(grid)
            _validate_generated(grid, out)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid occlusion-ladder instance")
