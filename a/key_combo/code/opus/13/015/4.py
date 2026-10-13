from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


BBox = tuple[int, int, int, int]

BG = 0
FRAME_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 400
LAYOUT_RETRIES = 120


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def visible_frames(grid: np.ndarray) -> tuple[list[int], dict[int, np.ndarray]]:
    upper = np.asarray(grid[:-1], dtype=int)
    bg = get_bg(grid)
    colors = sorted(int(x) for x in np.unique(upper) if int(x) != bg)
    assert 3 <= len(colors) <= 4, "expected 3-4 frame colors in the upper region"

    frames: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(upper == color)
        assert coords.size > 0, "every frame color must remain visible"
        box = bbox_coords(coords)
        r0, c0, r1, c1 = box
        assert r1 - r0 + 1 >= 4 and c1 - c0 + 1 >= 4, "frames must be at least 4x4"
        mask = frame_mask(grid.shape, box)
        assert np.all(mask[:-1][upper == color]), "visible pixels must lie on the inferred frame"
        frames[color] = mask
    return colors, frames


def crossing_edges(
    grid: np.ndarray, colors: list[int], frames: dict[int, np.ndarray]
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    upper = grid[:-1]
    for a, b in combinations(colors, 2):
        seen: set[tuple[int, int]] = set()
        overlap = frames[a][:-1] & frames[b][:-1]
        if not np.any(overlap):
            continue
        for r_raw, c_raw in np.argwhere(overlap):
            r = int(r_raw)
            c = int(c_raw)
            visible = int(upper[r, c])
            if visible == a:
                seen.add((b, a))
            elif visible == b:
                seen.add((a, b))
        assert len(seen) == 1, "every overlapping pair must imply one unambiguous direction"
        edges |= seen
    assert edges, "expected at least one visible frame overlap"
    return edges


def unique_total_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    succ = {color: set() for color in colors}
    pred = {color: set() for color in colors}
    for a, b in edges:
        succ[a].add(b)
        pred[b].add(a)

    order: list[int] = []
    available = [color for color in colors if not pred[color]]
    while available:
        assert len(available) == 1, "occlusion graph must determine a unique total order"
        color = available.pop()
        order.append(color)
        newly_available: list[int] = []
        for nxt in list(succ[color]):
            pred[nxt].remove(color)
            if not pred[nxt]:
                newly_available.append(nxt)
        available = sorted(newly_available)
    assert len(order) == len(colors), "occlusion graph must be acyclic and span all colors"
    return order


def apply_column_propagation(
    grid: np.ndarray, frames: dict[int, np.ndarray], order: list[int]
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    H, W = g.shape
    rank = {color: i for i, color in enumerate(order)}

    marked_cols = [c for c in range(W) if int(g[H - 1, c]) != bg]
    assert 2 <= len(marked_cols) <= 4, "expected 2-4 marked columns"

    for c in marked_cols:
        current = int(g[H - 1, c])
        assert current in rank, "bottom markers must reuse frame colors"
        for r in range(H - 2, -1, -1):
            contenders = [color for color, mask in frames.items() if bool(mask[r, c])]
            if contenders:
                current = max(contenders, key=lambda color: rank[color])
            out[r, c] = current
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a global occlusion order from overlapping rectangular frames, then use that order to arbitrate upward color propagation in marked columns.

    Input structure:
    - A black background contains 3-4 one-cell-thick axis-aligned rectangular frames in the upper region, each with a distinct color.
    - Frame perimeters overlap, and the visible color at each shared perimeter cell records which frame is in front.
    - The bottom row contains 2-4 colored marker pixels; each marker selects one propagation column and uses a color taken from one of the frames.

    Rule:
    - Reconstruct each frame as the perimeter of the minimal bounding box of its visible upper-region pixels.
    - For every overlapping pair of reconstructed perimeters, read the visible color on their shared cells to get a directed occlusion edge from hidden frame to visible frame.
    - Topologically sort those edges to obtain the unique global priority order from back to front.
    - In every marked column, scan upward from the bottom marker: when a cell lies on one or more reconstructed frame perimeters, paint it with the highest-priority frame color among them; otherwise copy the nearest painted color from below. Leave unmarked columns unchanged.

    Key insight:
    - The frame overlaps do not directly determine the output picture; they first reveal a global front-to-back ranking, and that ranking decides which frame color wins whenever a marked column encounters competing frame edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 4

    colors, frames = visible_frames(g)
    edges = crossing_edges(g, colors, frames)
    order = unique_total_order(colors, edges)
    return apply_column_propagation(g, frames, order)


def _render_input(
    shape: tuple[int, int],
    role_boxes: dict[str, BBox],
    role_colors: dict[str, int],
    order_roles: list[str],
    marked_cols: list[int],
    marker_colors: dict[int, int],
) -> tuple[np.ndarray, dict[int, np.ndarray], list[int]]:
    grid = np.full(shape, BG, dtype=int)
    frames_by_color: dict[int, np.ndarray] = {}
    for role in order_roles:
        color = role_colors[role]
        mask = frame_mask(shape, role_boxes[role])
        grid[mask] = color
        frames_by_color[color] = mask
    for c in marked_cols:
        grid[shape[0] - 1, c] = marker_colors[c]
    order_colors = [role_colors[role] for role in order_roles]
    return grid, frames_by_color, order_colors


def _column_frame_count(frames_by_color: dict[int, np.ndarray], c: int) -> int:
    return sum(int(np.any(mask[:-1, c])) for mask in frames_by_color.values())


def _visible_bbox_matches(grid: np.ndarray, color: int, box: BBox) -> bool:
    coords = np.argwhere(grid[:-1] == color)
    return coords.size > 0 and bbox_coords(coords) == box


def _choose_layout(rng: np.random.Generator, frame_count: int) -> tuple[tuple[int, int], dict[str, BBox]] | None:
    if frame_count == 3:
        H = int(rng.integers(16, 23))
        W = int(rng.integers(18, 25))
    else:
        H = int(rng.integers(19, 25))
        W = int(rng.integers(20, 26))

    for _ in range(LAYOUT_RETRIES):
        w0 = int(rng.integers(4, 7))
        w2 = int(rng.integers(4, 7))

        c0 = int(rng.integers(1, max(2, W // 4 + 1)))
        c2_min = c0 + w0 + 3
        c2_max = W - w2 - 2
        if c2_min > c2_max:
            continue
        c2 = int(rng.integers(max(c2_min, W // 2 - 1), c2_max + 1))

        t0 = int(rng.integers(1, 4))
        t2 = int(rng.integers(1, 4))

        if frame_count == 3:
            b0 = int(rng.integers(H // 2, H - 5))
            b2 = int(rng.integers(H // 2, H - 4))
        else:
            b0 = int(rng.integers(H // 2, H - 8))
            b2 = int(rng.integers(b0 + 4, H - 4))
        if min(b0 - t0 + 1, b2 - t2 + 1) < 7:
            continue

        h1 = int(rng.integers(4, 6))
        h1_top_min = max(t0, t2) + 1
        h1_top_max = min(b0, b2) - h1
        if frame_count == 4:
            h1_top_max = min(h1_top_max, b0 - h1 - 1)
        if h1_top_min > h1_top_max:
            continue
        th1 = int(rng.integers(h1_top_min, h1_top_max + 1))
        lh1 = max(0, c0 - int(rng.integers(0, 2)))
        rh1 = min(W - 2, c2 + w2 - 1 + int(rng.integers(0, 2)))
        if rh1 - lh1 + 1 < 6:
            continue

        boxes: dict[str, BBox] = {
            "v0": (t0, c0, b0, c0 + w0 - 1),
            "h1": (th1, lh1, th1 + h1 - 1, rh1),
            "v2": (t2, c2, b2, c2 + w2 - 1),
        }

        if frame_count == 4:
            h3 = int(rng.integers(4, 6))
            h3_top_min = max(th1 + h1 + 1, b0 + 1)
            h3_top_max = b2 - h3
            if h3_top_min > h3_top_max:
                continue
            th3 = int(rng.integers(h3_top_min, h3_top_max + 1))
            lh3 = max(0, c2 - int(rng.integers(0, 2)))
            rh3 = min(W - 2, c2 + w2 - 1 + int(rng.integers(1, 4)))
            if rh3 - lh3 + 1 < 4:
                continue
            boxes["h3"] = (th3, lh3, th3 + h3 - 1, rh3)

        masks = [frame_mask((H, W), box) for box in boxes.values()]
        overlap_count = np.sum(np.stack(masks, axis=0), axis=0)
        if int(overlap_count[:-1].max()) > 2:
            continue

        return (H, W), boxes

    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.choice([3, 4]))
    layout = _choose_layout(rng, frame_count)
    if layout is None:
        return None
    shape, role_boxes = layout

    roles = ["v0", "h1", "v2"] + (["h3"] if frame_count == 4 else [])
    role_colors = dict(zip(roles, rng.permutation(FRAME_COLORS)[:frame_count].tolist()))
    order_roles = roles if bool(rng.integers(0, 2)) else roles[::-1]

    preview_grid, frames_by_color, order_colors = _render_input(
        shape,
        role_boxes,
        role_colors,
        order_roles,
        marked_cols=[],
        marker_colors={},
    )

    for color, box in ((role_colors[role], role_boxes[role]) for role in roles):
        if not _visible_bbox_matches(preview_grid, color, box):
            return None

    try:
        inferred_colors, inferred_frames = visible_frames(preview_grid)
        inferred_edges = crossing_edges(preview_grid, inferred_colors, inferred_frames)
        inferred_order = unique_total_order(inferred_colors, inferred_edges)
    except AssertionError:
        return None
    if inferred_order != order_colors:
        return None

    candidate_cols: list[int] = []
    for role in roles:
        if not role.startswith("v"):
            continue
        _, c0, _, c1 = role_boxes[role]
        for c in (c0, c1):
            if c not in candidate_cols and _column_frame_count(frames_by_color, c) >= 2:
                candidate_cols.append(c)
    if len(candidate_cols) < 2:
        return None

    mark_count = int(rng.integers(2, min(4, len(candidate_cols)) + 1))
    marked_cols = sorted(rng.choice(candidate_cols, size=mark_count, replace=False).tolist())
    marker_palette = [role_colors[role] for role in roles]
    marker_colors = {c: int(rng.choice(marker_palette)) for c in marked_cols}

    grid, frames_by_color, order_colors = _render_input(
        shape,
        role_boxes,
        role_colors,
        order_roles,
        marked_cols,
        marker_colors,
    )
    out = apply_column_propagation(grid, frames_by_color, order_colors)

    if not np.any(out != grid):
        return None
    if not np.array_equal(main(grid), out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
