from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 500
MAX_ARC_SIDE = 30


def _find_components(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    region_id = np.full((height, width), -1, dtype=int)
    components: list[dict[str, object]] = []

    next_id = 0
    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == BG or region_id[r, c] != -1:
                continue

            queue: deque[tuple[int, int]] = deque([(r, c)])
            region_id[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if region_id[nr, nc] != -1 or int(g[nr, nc]) != color:
                        continue
                    region_id[nr, nc] = next_id
                    queue.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            components.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                }
            )
            next_id += 1

    return components, region_id


def _adjacency_sets(grid: np.ndarray, region_id: np.ndarray, n_regions: int) -> list[set[int]]:
    g = np.asarray(grid, dtype=int)
    adj = [set() for _ in range(n_regions)]
    height, width = g.shape

    for r in range(height):
        for c in range(width):
            if int(g[r, c]) == BG:
                continue
            src = int(region_id[r, c])
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if int(g[nr, nc]) == BG:
                    continue
                dst = int(region_id[nr, nc])
                if dst != src:
                    adj[src].add(dst)

    return adj


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _erode_once(mask: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if not mask[nr, nc] and not support_mask[nr, nc]:
                out[rr, cc] = False
                break
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink each colored region by an amount determined by how many other regions it touches.

    Input structure:
    - The grid is a black background with several non-overlapping 4-connected colored regions.
    - Each region uses a distinct color and may touch other colored regions along shared edges.
    - Regions are compact rectangles or near-rectangles, with enough structure to have a clear bounding box center.

    Rule:
    - Build the 4-neighbor adjacency graph of the colored regions, counting for each region how many distinct other regions touch it.
    - If a region touches 0 others, keep it unchanged.
    - If it touches 1 other, erode away every pixel that is 4-adjacent to black; if nothing remains, keep only the region bbox center pixel.
    - If it touches 2 others, apply that same black-adjacency erosion twice; if nothing remains, keep only the region bbox center pixel.
    - If it touches 3 or more others, erase the whole region except for the single bbox center pixel.

    Key insight:
    - The task is two-stage: first infer the hidden adjacency graph between colored regions, then map each region's adjacency count to a different amount of self-shrinking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components, region_id = _find_components(g)
    adjacency = _adjacency_sets(g, region_id, len(components))
    out = np.zeros_like(g)

    for comp in components:
        rid = int(comp["id"])
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = comp["bbox"]
        degree = len(adjacency[rid])

        mask = np.zeros_like(g, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        support_mask = (g != BG) & ~mask

        if degree == 0:
            kept = mask
            center = None
        elif degree >= 3:
            kept = np.zeros_like(mask)
            center = _bbox_center(bbox)
        else:
            kept = mask.copy()
            for _ in range(degree):
                kept = _erode_once(kept, support_mask)
                if not np.any(kept):
                    break
            center = None if np.any(kept) else _bbox_center(bbox)

        out[kept] = color
        if center is not None:
            out[center] = color

    return out


def _render_rects(rects: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    min_r = min(r0 for _label, r0, _c0, _h, _w in rects)
    min_c = min(c0 for _label, _r0, c0, _h, _w in rects)
    shifted = [
        (label, r0 - min_r, c0 - min_c, h, w)
        for label, r0, c0, h, w in rects
    ]
    height = max(r0 + h for _label, r0, _c0, h, _w in shifted)
    width = max(c0 + w for _label, _r0, c0, _h, w in shifted)
    grid = np.zeros((height, width), dtype=int)
    for label, r0, c0, h, w in shifted:
        grid[r0 : r0 + h, c0 : c0 + w] = label
    return grid


def _make_isolated() -> np.ndarray:
    height = random.randint(3, 8)
    width = random.randint(3, 8)
    grid = np.ones((height, width), dtype=int)
    if height >= 4 and width >= 4 and random.random() < 0.4:
        rr = 0 if random.random() < 0.5 else height - 1
        cc = 0 if random.random() < 0.5 else width - 1
        grid[rr, cc] = 0
    return grid


def _offset_choices(anchor_len: int, other_len: int, min_overlap: int) -> list[int]:
    low = -(other_len - min_overlap)
    high = anchor_len - min_overlap
    return list(range(low, high + 1))


def _make_chain() -> np.ndarray:
    orientation = random.choice(["H", "V"])
    dims = [(random.randint(2, 7), random.randint(2, 7)) for _ in range(3)]

    if orientation == "H":
        (h1, w1), (h2, w2), (h3, w3) = dims
        overlap12 = random.randint(1, min(3, h1, h2))
        overlap23 = random.randint(1, min(3, h2, h3))
        r1 = random.choice(_offset_choices(h2, h1, overlap12))
        r2 = 0
        r3 = random.choice(_offset_choices(h2, h3, overlap23))
        rects = [
            (1, r1, 0, h1, w1),
            (2, r2, w1, h2, w2),
            (3, r3, w1 + w2, h3, w3),
        ]
    else:
        (h1, w1), (h2, w2), (h3, w3) = dims
        overlap12 = random.randint(1, min(3, w1, w2))
        overlap23 = random.randint(1, min(3, w2, w3))
        c1 = random.choice(_offset_choices(w2, w1, overlap12))
        c2 = 0
        c3 = random.choice(_offset_choices(w2, w3, overlap23))
        rects = [
            (1, 0, c1, h1, w1),
            (2, h1, c2, h2, w2),
            (3, h1 + h2, c3, h3, w3),
        ]

    return _render_rects(rects)


def _make_star() -> np.ndarray:
    degree = random.choice([3, 4])
    sides = random.sample(["top", "bottom", "left", "right"], degree)
    center_h = random.randint(3, 7)
    center_w = random.randint(3, 7)

    leaves: dict[str, tuple[int, int, int]] = {}
    for side in sides:
        if side in {"top", "bottom"}:
            leaf_h = random.randint(2, 5)
            leaf_w = random.randint(2, center_w)
            offset = random.randint(0, center_w - leaf_w)
            leaves[side] = (leaf_h, leaf_w, offset)
        else:
            leaf_h = random.randint(2, center_h)
            leaf_w = random.randint(2, 5)
            offset = random.randint(0, center_h - leaf_h)
            leaves[side] = (leaf_h, leaf_w, offset)

    north = leaves["top"][0] if "top" in leaves else 0
    south = leaves["bottom"][0] if "bottom" in leaves else 0
    west = leaves["left"][1] if "left" in leaves else 0
    east = leaves["right"][1] if "right" in leaves else 0

    rects: list[tuple[int, int, int, int, int]] = [(1, north, west, center_h, center_w)]
    next_label = 2

    if "top" in leaves:
        leaf_h, leaf_w, offset = leaves["top"]
        rects.append((next_label, 0, west + offset, leaf_h, leaf_w))
        next_label += 1
    if "bottom" in leaves:
        leaf_h, leaf_w, offset = leaves["bottom"]
        rects.append((next_label, north + center_h, west + offset, leaf_h, leaf_w))
        next_label += 1
    if "left" in leaves:
        leaf_h, leaf_w, offset = leaves["left"]
        rects.append((next_label, north + offset, 0, leaf_h, leaf_w))
        next_label += 1
    if "right" in leaves:
        leaf_h, leaf_w, offset = leaves["right"]
        rects.append((next_label, north + offset, west + center_w, leaf_h, leaf_w))

    return _render_rects(rects)


def _recolor_motif(motif: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.zeros_like(motif)
    labels = sorted(int(v) for v in np.unique(motif) if v != 0)
    assert len(labels) == len(colors)
    for label, color in zip(labels, colors):
        out[motif == label] = int(color)
    return out


def _compose_motifs(motifs: list[np.ndarray]) -> np.ndarray | None:
    slots = random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], len(motifs))
    row_heights = [0, 0]
    col_widths = [0, 0]
    for motif, (slot_r, slot_c) in zip(motifs, slots):
        h, w = motif.shape
        row_heights[slot_r] = max(row_heights[slot_r], h)
        col_widths[slot_c] = max(col_widths[slot_c], w)

    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    row_gap = random.randint(2, 4)
    col_gap = random.randint(2, 4)

    height = top + row_heights[0] + row_gap + row_heights[1] + bottom
    width = left + col_widths[0] + col_gap + col_widths[1] + right
    if height > MAX_ARC_SIDE or width > MAX_ARC_SIDE:
        return None

    grid = np.zeros((height, width), dtype=int)
    for motif, (slot_r, slot_c) in zip(motifs, slots):
        cell_top = top + (row_heights[0] + row_gap if slot_r == 1 else 0)
        cell_left = left + (col_widths[0] + col_gap if slot_c == 1 else 0)
        slack_r = row_heights[slot_r] - motif.shape[0]
        slack_c = col_widths[slot_c] - motif.shape[1]
        r0 = cell_top + random.randint(0, slack_r)
        c0 = cell_left + random.randint(0, slack_c)
        patch = grid[r0 : r0 + motif.shape[0], c0 : c0 + motif.shape[1]]
        patch[motif != 0] = motif[motif != 0]

    return grid


def _component_degrees(grid: np.ndarray) -> list[int]:
    components, region_id = _find_components(grid)
    adjacency = _adjacency_sets(grid, region_id, len(components))
    return [len(neighbors) for neighbors in adjacency]


def _covers_all_branches(degrees: list[int]) -> bool:
    return (
        any(degree == 0 for degree in degrees)
        and any(degree == 1 for degree in degrees)
        and any(degree == 2 for degree in degrees)
        and any(degree >= 3 for degree in degrees)
    )


def _gen_attempt() -> np.ndarray | None:
    motif_specs = [_make_isolated(), _make_chain(), _make_star()]
    region_count = int(sum(np.max(motif) for motif in motif_specs))
    colors = random.sample(PALETTE, region_count)

    recolored: list[np.ndarray] = []
    cursor = 0
    for motif in motif_specs:
        n_regions = int(np.max(motif))
        recolored.append(_recolor_motif(motif, colors[cursor : cursor + n_regions]))
        cursor += n_regions

    random.shuffle(recolored)
    grid = _compose_motifs(recolored)
    if grid is None:
        return None
    if max(grid.shape) > MAX_ARC_SIDE:
        return None

    nonzero_colors = np.unique(grid[grid != BG])
    if len(nonzero_colors) != region_count:
        return None

    degrees = _component_degrees(grid)
    if not _covers_all_branches(degrees):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
