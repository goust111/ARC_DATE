from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
MARKER = 5
FRAME_PALETTE = (1, 2, 3, 4, 7)
MAX_GENERATION_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) `(r,c)` coordinate array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    """Return the one-pixel-thick rectangular perimeter mask for an inclusive bbox."""
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def interior_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    """Return strict interior cells of an inclusive bbox."""
    r0, c0, r1, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def infer_frames(grid: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    """Infer one full rectangular frame bbox per non-bg, non-marker color."""
    colors = sorted(set(map(int, np.unique(grid))) - {BG, MARKER})
    assert 3 <= len(colors) <= 5, "expected 3 to 5 frame colors"

    frames: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "each frame color must remain visible"
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 >= 3 and c1 - c0 >= 3, "frames need non-empty interiors"
        mask = frame_mask(grid.shape, box)
        assert np.all(mask[coords[:, 0], coords[:, 1]]), "visible pixels must lie on the inferred perimeter"
        frames[color] = box
    return frames


def crossing_edges(
    grid: np.ndarray, frames: dict[int, tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    """
    Build visible-priority edges `top -> bottom` from pairwise frame crossings.

    At each shared perimeter pixel, exactly one of the two frame colors must be visible.
    """
    edges: set[tuple[int, int]] = set()
    for a, b in combinations(sorted(frames), 2):
        overlap = frame_mask(grid.shape, frames[a]) & frame_mask(grid.shape, frames[b])
        if not overlap.any():
            continue
        visible = set(map(int, np.unique(grid[overlap])))
        visible.discard(BG)
        visible.discard(MARKER)
        assert visible <= {a, b} and len(visible) == 1, "each crossing pair must expose one of its own colors"
        top = next(iter(visible))
        bottom = b if top == a else a
        edges.add((top, bottom))
    assert edges, "expected at least one visible occlusion relation"
    return edges


def unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    """Return the unique topological order, asserting that uniqueness holds at every step."""
    remaining = set(colors)
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for u, v in edges:
        if v not in outgoing[u]:
            outgoing[u].add(v)
            indegree[v] += 1

    order: list[int] = []
    while remaining:
        zeros = sorted(color for color in remaining if indegree[color] == 0)
        assert len(zeros) == 1, "occlusion graph must have a unique topological order"
        cur = zeros[0]
        order.append(cur)
        remaining.remove(cur)
        for nxt in outgoing[cur]:
            indegree[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize overlapping rectangular frames into priority-ordered count rows.

    Input structure:
    - Black background with 3 to 5 one-pixel-thick colored rectangular frames, one color per frame.
    - Frames overlap at some perimeter crossings, and the visible color at each crossing records which frame is on top.
    - Gray marker pixels lie strictly inside frame bboxes, never on frame perimeters.

    Rule:
    - Reconstruct each frame from the bbox of its visible pixels.
    - For each pair of overlapping perimeters, read the visible crossing color and add a priority edge from the visible frame to the hidden frame.
    - The unique topological order of that graph gives row order from highest priority to lowest; for each frame, count gray markers inside its bbox interior and fill that many leading cells in its row with the frame color.

    Key insight:
    - Row order comes from spatial occlusion, while row length comes from interior marker counts; neither source alone determines the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    frames = infer_frames(g)
    edges = crossing_edges(g, frames)
    order = unique_topological_order(sorted(frames), edges)

    counts: dict[int, int] = {}
    for color, (r0, c0, r1, c1) in frames.items():
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        count = int(np.count_nonzero(interior == MARKER))
        assert count >= 1, "each frame must contain at least one marker"
        counts[color] = count

    assert len(set(counts.values())) == len(counts), "marker counts must be distinct"
    width = max(counts.values())
    out = np.zeros((len(order), width), dtype=int)
    for row, color in enumerate(order):
        out[row, : counts[color]] = color
    return out


def _sorted_or_reversed(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _sample_order_and_counts(n_frames: int) -> tuple[list[int], dict[int, int]]:
    """Choose frame colors and marker counts while avoiding easy numeric shortcuts."""
    while True:
        order_high_to_low = random.sample(FRAME_PALETTE, n_frames)
        count_values = random.sample([1, 2, 3, 4, 5], n_frames)
        if _sorted_or_reversed(order_high_to_low):
            continue
        if _sorted_or_reversed(count_values):
            continue
        counts = {color: count for color, count in zip(order_high_to_low, count_values, strict=True)}
        return order_high_to_low, counts


def _sample_layout(n_frames: int) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int, int, int]] | None:
    """
    Sample a staircase layout where only consecutive frames overlap.

    All frames share one size. Consecutive boxes are shifted down-right by `(sr, sc)`.
    """
    for _ in range(200):
        box_h = random.randint(7, 10)
        box_w = random.randint(7, 10)
        min_sr = box_h // 2 + 1
        min_sc = box_w // 2 + 1
        if min_sr > box_h - 2 or min_sc > box_w - 2:
            continue
        sr = random.randint(min_sr, box_h - 2)
        sc = random.randint(min_sc, box_w - 2)

        top = random.randint(1, 3)
        left = random.randint(1, 3)
        bottom = random.randint(1, 3)
        right = random.randint(1, 3)
        height = top + box_h + (n_frames - 1) * sr + bottom
        width = left + box_w + (n_frames - 1) * sc + right
        if 16 <= height <= 30 and 16 <= width <= 30:
            return (box_h, box_w), (sr, sc), (top, left, bottom, right)
    return None


def _boxes_for_layout(
    n_frames: int,
    box_size: tuple[int, int],
    shift: tuple[int, int],
    margins: tuple[int, int, int, int],
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    box_h, box_w = box_size
    sr, sc = shift
    top, left, bottom, right = margins
    height = top + box_h + (n_frames - 1) * sr + bottom
    width = left + box_w + (n_frames - 1) * sc + right
    boxes = []
    for idx in range(n_frames):
        r0 = top + idx * sr
        c0 = left + idx * sc
        boxes.append((r0, c0, r0 + box_h - 1, c0 + box_w - 1))
    return (height, width), boxes


def _strictly_inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 < r < r1 and c0 < c < c1


def _marker_candidates(
    box: tuple[int, int, int, int],
    all_boxes: list[tuple[int, int, int, int]],
    all_masks: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Prefer interior cells that belong only to this frame's interior."""
    own_index = all_boxes.index(box)
    candidates = []
    for r, c in interior_cells(box):
        if any(mask[r, c] for j, mask in enumerate(all_masks) if j != own_index):
            continue
        if any(_strictly_inside(other, r, c) for j, other in enumerate(all_boxes) if j != own_index):
            continue
        candidates.append((r, c))
    return candidates


def _render_grid(
    shape: tuple[int, int],
    boxes_low_to_high: list[tuple[int, int, int, int]],
    colors_low_to_high: list[int],
    counts_by_color: dict[int, int],
) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    masks = [frame_mask(shape, box) for box in boxes_low_to_high]

    for color, mask in zip(colors_low_to_high, masks, strict=True):
        grid[mask] = color

    for color, box in zip(colors_low_to_high, boxes_low_to_high, strict=True):
        need = counts_by_color[color]
        candidates = _marker_candidates(box, boxes_low_to_high, masks)
        if len(candidates) < need:
            return None
        for r, c in random.sample(candidates, need):
            grid[r, c] = MARKER
    return grid


def _validate_chain_geometry(shape: tuple[int, int], boxes_low_to_high: list[tuple[int, int, int, int]]) -> bool:
    masks = [frame_mask(shape, box) for box in boxes_low_to_high]
    n = len(boxes_low_to_high)
    for i in range(n):
        for j in range(i + 1, n):
            overlap = masks[i] & masks[j]
            if j == i + 1:
                if int(overlap.sum()) != 2:
                    return False
            elif overlap.any():
                return False
    return True


def _expected_output(order_high_to_low: list[int], counts_by_color: dict[int, int]) -> np.ndarray:
    width = max(counts_by_color.values())
    out = np.zeros((len(order_high_to_low), width), dtype=int)
    for row, color in enumerate(order_high_to_low):
        out[row, : counts_by_color[color]] = color
    return out


def _validate_instance(
    grid: np.ndarray,
    order_high_to_low: list[int],
    counts_by_color: dict[int, int],
    boxes_low_to_high: list[tuple[int, int, int, int]],
) -> bool:
    expected = _expected_output(order_high_to_low, counts_by_color)
    if not np.array_equal(main(grid), expected):
        return False

    inferred = infer_frames(grid)
    expected_boxes = {
        color: box for color, box in zip(reversed(order_high_to_low), boxes_low_to_high, strict=True)
    }
    if inferred != expected_boxes:
        return False

    edges = crossing_edges(grid, inferred)
    order = unique_topological_order(sorted(inferred), edges)
    if order != order_high_to_low:
        return False

    for color, box in inferred.items():
        r0, c0, r1, c1 = box
        if int(np.count_nonzero(grid[r0 + 1 : r1, c0 + 1 : c1] == MARKER)) != counts_by_color[color]:
            return False

    if _sorted_or_reversed(order_high_to_low):
        return False
    if _sorted_or_reversed([counts_by_color[color] for color in order_high_to_low]):
        return False
    if not _validate_chain_geometry(grid.shape, boxes_low_to_high):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(3, 5)
    order_high_to_low, counts_by_color = _sample_order_and_counts(n_frames)
    layout = _sample_layout(n_frames)
    if layout is None:
        return None
    box_size, shift, margins = layout
    shape, boxes_low_to_high = _boxes_for_layout(n_frames, box_size, shift, margins)
    grid = _render_grid(shape, boxes_low_to_high, list(reversed(order_high_to_low)), counts_by_color)
    if grid is None:
        return None
    if not _validate_instance(grid, order_high_to_low, counts_by_color, boxes_low_to_high):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occlusion/count summary task")
