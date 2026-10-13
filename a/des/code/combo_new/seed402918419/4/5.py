from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (16, 28)
GRID_W_MAX = 30
HOLE_COUNT_RANGE = (2, 5)
SHAPE_AREA_RANGE = (2, 6)
SHAPE_BBOX_MAX = 4
TEMPLATE_MARGIN_RANGE = (1, 2)
CHAIN_TOP_OFFSET_RANGE = (0, 2)
CHAIN_GAP_RANGE = (2, 4)
TEMPLATE_LEFT_MARGIN_RANGE = (1, 3)
TEMPLATE_TOP_MARGIN_RANGE = (1, 3)
RIGHT_GAP_RANGE = (2, 4)
PIECE_PADDING = 1
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert len(rows) > 0 and len(cols) > 0
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _normalize_cells(cells: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cells = tuple((int(r), int(c)) for r, c in cells)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _cells_to_mask(cells: Iterable[tuple[int, int]]) -> np.ndarray:
    norm = _normalize_cells(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for k in range(4):
        rot = _trim(np.rot90(mask, k))
        variants.append((tuple(map(int, rot.shape)), rot.astype(np.uint8).tobytes()))
    return min(variants)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    cells = list(cells)
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return (min(rs), min(cs), max(rs), max(cs))


def _component_mask_from_cells(
    cells: Iterable[tuple[int, int]],
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cells = list(cells)
    r0, c0, r1, c1 = _bbox(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _find_holes_in_bbox(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    zero_comps = _components(sub == BG)
    holes: list[dict] = []
    for comp in zero_comps:
        if any(rr in {0, sub.shape[0] - 1} or cc in {0, sub.shape[1] - 1} for rr, cc in comp):
            continue
        global_cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in comp]
        mask, hbbox = _component_mask_from_cells(global_cells)
        holes.append(
            {
                "cells": global_cells,
                "cell_set": set(global_cells),
                "bbox": hbbox,
                "mask": mask,
                "key": _shape_key(mask),
            }
        )
    holes.sort(key=lambda hole: hole["bbox"])
    return holes


def _template_record(grid: np.ndarray) -> dict:
    candidates: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _components(grid == int(color)):
            bbox = _bbox(comp)
            holes = _find_holes_in_bbox(grid, bbox)
            if not holes:
                continue
            candidates.append(
                {
                    "color": int(color),
                    "cells": comp,
                    "cell_set": set(comp),
                    "bbox": bbox,
                    "holes": holes,
                }
            )
    assert candidates, "expected a template component with internal holes"
    candidates.sort(
        key=lambda item: (
            len(item["holes"]),
            len(item["cells"]),
            -item["bbox"][1],
        ),
        reverse=True,
    )
    return candidates[0]


def _piece_records(grid: np.ndarray, template_cells: set[tuple[int, int]]) -> list[dict]:
    mask = (grid != BG).copy()
    for r, c in template_cells:
        mask[r, c] = False
    pieces: list[dict] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each movable fragment must be monochrome"
        color = next(iter(colors))
        pmask, pbbox = _component_mask_from_cells(comp)
        pieces.append(
            {
                "cells": comp,
                "bbox": pbbox,
                "mask": pmask,
                "key": _shape_key(pmask),
                "color": int(color),
            }
        )
    return pieces


def _vertical_parent(
    holes: list[dict],
    template_color: int,
    grid: np.ndarray,
) -> dict[int, int]:
    parents: dict[int, tuple[int, int, int]] = {}
    for upper_idx, upper in enumerate(holes):
        for lower_idx, lower in enumerate(holes):
            if upper_idx == lower_idx:
                continue
            overlap = 0
            for r, c in upper["cells"]:
                if (r + 2, c) in lower["cell_set"] and int(grid[r + 1, c]) == int(template_color):
                    overlap += 1
            if overlap == 0:
                continue
            candidate = (overlap, upper["bbox"][2], upper_idx)
            prev = parents.get(lower_idx)
            if prev is None or candidate > prev:
                parents[lower_idx] = candidate
    return {lower_idx: upper_idx for lower_idx, (_ov, _row, upper_idx) in parents.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert the scattered pieces into matching template holes, then let colors flow downward through vertically adjacent holes.

    Input structure:
    - The background is black.
    - One large gray connected template on the left contains several enclosed black holes of different 4-connected shapes.
    - On the right, the same number of monochrome colored fragments are scattered; each fragment matches exactly one hole up to 0/90/180/270-degree rotation.

    Rule:
    - Detect every enclosed hole inside the gray template and every colored fragment outside it.
    - Match holes to fragments by shape up to rotation and assign each hole the matched fragment's original color.
    - If a hole has another hole directly above it with exactly one gray row between them in at least one shared column, recolor the lower hole with the upper hole's original fragment color; otherwise keep its own original color.
    - Remove all right-side fragments and leave only the filled template.

    Key insight:
    - Shape matching solves which fragment belongs in each hole, but the final colors come from a second relation: vertical hole adjacency makes the upper hole's original color cascade one step downward.
    """
    g = np.asarray(grid, dtype=int)
    template = _template_record(g)
    holes = template["holes"]
    pieces = _piece_records(g, template["cell_set"])
    assert len(holes) == len(pieces), "hole count must match piece count"

    piece_buckets: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for piece in pieces:
        piece_buckets.setdefault(piece["key"], []).append(piece)

    original_colors: dict[int, int] = {}
    for idx, hole in enumerate(holes):
        bucket = piece_buckets.get(hole["key"], [])
        assert bucket, "every hole must have a matching fragment"
        original_colors[idx] = int(bucket.pop()["color"])
    assert all(not bucket for bucket in piece_buckets.values()), "all fragments must be consumed"

    parents = _vertical_parent(holes, template["color"], g)
    final_colors: dict[int, int] = {}
    for idx in range(len(holes)):
        if idx in parents:
            final_colors[idx] = int(original_colors[parents[idx]])
        else:
            final_colors[idx] = int(original_colors[idx])

    out = np.full_like(g, BG)
    for r, c in template["cells"]:
        out[r, c] = int(template["color"])
    for idx, hole in enumerate(holes):
        for r, c in hole["cells"]:
            out[r, c] = int(final_colors[idx])
    return out


def _random_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(SHAPE_AREA_RANGE[0], SHAPE_AREA_RANGE[1])
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            candidates = []
            for cell in frontier:
                new_cells = set(cells)
                new_cells.add(cell)
                norm = _normalize_cells(new_cells)
                h = max(r for r, _ in norm) + 1
                w = max(c for _, c in norm) + 1
                if h <= SHAPE_BBOX_MAX and w <= SHAPE_BBOX_MAX:
                    candidates.append(set(norm))
            if not candidates:
                break
            cells = random.choice(candidates)
        if len(cells) != area:
            continue
        mask = _cells_to_mask(cells)
        if mask.size <= 1:
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _sample_unique_shapes(count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    while len(shapes) < count:
        shape = _random_shape()
        key = _shape_key(shape)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(shape)
    return shapes


def _random_partition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    last = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - last)
        last = cut
    return out


def _build_chain(masks: list[np.ndarray]) -> tuple[list[dict], tuple[int, int]]:
    placements: list[tuple[int, int]] = [(0, 0)]
    min_x = 0
    max_x = masks[0].shape[1]
    max_y = masks[0].shape[0]
    for prev_mask, curr_mask in zip(masks, masks[1:], strict=False):
        prev_x, prev_y = placements[-1]
        prev_cols = [int(c) for c in np.where(prev_mask[-1])[0]]
        curr_cols = [int(c) for c in np.where(curr_mask[0])[0]]
        curr_x = prev_x + random.choice(prev_cols) - random.choice(curr_cols)
        curr_y = prev_y + prev_mask.shape[0] + 1
        placements.append((int(curr_x), int(curr_y)))
        min_x = min(min_x, curr_x)
        max_x = max(max_x, curr_x + curr_mask.shape[1])
        max_y = max(max_y, curr_y + curr_mask.shape[0])

    shift_x = -min_x
    records = []
    for mask, (x, y) in zip(masks, placements, strict=False):
        records.append({"mask": mask, "x": int(x + shift_x), "y": int(y)})
    return records, (int(max_y), int(max_x - min_x))


def _can_place_piece(
    grid: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    min_col: int,
) -> bool:
    h, w = mask.shape
    if left < min_col:
        return False
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False

    r0 = max(0, top - PIECE_PADDING)
    c0 = max(0, left - PIECE_PADDING)
    r1 = min(grid.shape[0], top + h + PIECE_PADDING)
    c1 = min(grid.shape[1], left + w + PIECE_PADDING)
    if np.any(grid[r0:r1, c0:c1] != BG):
        return False
    return True


def _place_piece(grid: np.ndarray, mask: np.ndarray, color: int, min_col: int) -> bool:
    h, w = mask.shape
    positions = [
        (top, left)
        for top in range(grid.shape[0] - h + 1)
        for left in range(min_col, grid.shape[1] - w + 1)
    ]
    random.shuffle(positions)
    for top, left in positions:
        if not _can_place_piece(grid, mask, top, left, min_col):
            continue
        grid[top : top + h, left : left + w][mask] = int(color)
        return True
    return False


def _expected_output(
    shape_to_color: dict[tuple[tuple[int, int], bytes], int],
    hole_specs: list[dict],
    template_top: int,
    template_left: int,
    template_patch: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    h, w = template_patch.shape
    out[template_top : template_top + h, template_left : template_left + w] = template_patch
    for hole in hole_specs:
        key = hole["key"]
        if hole["parent"] is None:
            color = int(shape_to_color[key])
        else:
            color = int(shape_to_color[hole_specs[hole["parent"]]["key"]])
        for rr, cc in hole["cells"]:
            out[template_top + rr, template_left + cc] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    hole_count = random.randint(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1])
    chain_count = random.randint(1, hole_count - 1) if hole_count > 1 else 1
    chain_lengths = _random_partition(hole_count, chain_count)
    shapes = _sample_unique_shapes(hole_count)
    colors = random.sample(PIECE_COLORS, hole_count)

    hole_orientations = [np.rot90(shape, random.randrange(4)) for shape in shapes]
    piece_orientations = [np.rot90(shape, random.randrange(4)) for shape in shapes]

    chains: list[list[int]] = []
    start = 0
    for length in chain_lengths:
        chains.append(list(range(start, start + length)))
        start += length
    random.shuffle(chains)

    chain_records = []
    hole_specs: list[dict | None] = [None] * hole_count
    for chain in chains:
        masks = [hole_orientations[idx] for idx in chain]
        built, size = _build_chain(masks)
        chain_records.append({"indices": chain, "holes": built, "size": size})
        for local_idx, global_idx in enumerate(chain):
            hole_specs[global_idx] = {
                "key": _shape_key(shapes[global_idx]),
                "mask": built[local_idx]["mask"],
                "cells": [],
                "parent": chain[local_idx - 1] if local_idx > 0 else None,
            }
    assert all(spec is not None for spec in hole_specs)

    top_margin = random.randint(TEMPLATE_MARGIN_RANGE[0], TEMPLATE_MARGIN_RANGE[1])
    bottom_margin = random.randint(TEMPLATE_MARGIN_RANGE[0], TEMPLATE_MARGIN_RANGE[1])
    left_margin = random.randint(TEMPLATE_MARGIN_RANGE[0], TEMPLATE_MARGIN_RANGE[1])
    right_margin = random.randint(TEMPLATE_MARGIN_RANGE[0], TEMPLATE_MARGIN_RANGE[1])
    top_offsets = [
        random.randint(CHAIN_TOP_OFFSET_RANGE[0], CHAIN_TOP_OFFSET_RANGE[1]) for _ in chain_records
    ]
    gaps = [
        random.randint(CHAIN_GAP_RANGE[0], CHAIN_GAP_RANGE[1]) for _ in range(max(0, len(chain_records) - 1))
    ]

    inner_h = 0
    inner_w = 0
    for idx, chain in enumerate(chain_records):
        inner_h = max(inner_h, top_offsets[idx] + chain["size"][0])
        inner_w += chain["size"][1]
        if idx < len(gaps):
            inner_w += gaps[idx]
    template_h = top_margin + inner_h + bottom_margin
    template_w = left_margin + inner_w + right_margin
    if template_h > 24 or template_w > 18:
        return None

    template_patch = np.full((template_h, template_w), TEMPLATE_COLOR, dtype=int)
    cursor_x = left_margin
    for chain_idx, chain in enumerate(chain_records):
        base_y = top_margin + top_offsets[chain_idx]
        for local_idx, global_idx in enumerate(chain["indices"]):
            hole = chain["holes"][local_idx]
            mask = hole["mask"]
            y0 = base_y + hole["y"]
            x0 = cursor_x + hole["x"]
            template_patch[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]][mask] = BG
            cells = [(int(y0 + rr), int(x0 + cc)) for rr, cc in np.argwhere(mask)]
            hole_specs[global_idx]["cells"] = cells
        cursor_x += chain["size"][1]
        if chain_idx < len(gaps):
            cursor_x += gaps[chain_idx]

    h_min = max(GRID_H_RANGE[0], template_h + 2)
    h_max = min(GRID_H_RANGE[1], 30)
    if h_min > h_max:
        return None
    min_right_width = max(8, max(piece.shape[1] for piece in piece_orientations) + 3)
    min_total_w = template_w + RIGHT_GAP_RANGE[0] + min_right_width + 2
    if min_total_w > GRID_W_MAX:
        return None
    grid_h = random.randint(h_min, h_max)
    grid_w = random.randint(min_total_w, GRID_W_MAX)

    template_top = random.randint(
        TEMPLATE_TOP_MARGIN_RANGE[0],
        grid_h - template_h - TEMPLATE_TOP_MARGIN_RANGE[0],
    )
    max_left_margin = min(TEMPLATE_LEFT_MARGIN_RANGE[1], grid_w - template_w - min_right_width - 1)
    min_left_margin = TEMPLATE_LEFT_MARGIN_RANGE[0]
    if min_left_margin > max_left_margin:
        return None
    template_left = random.randint(min_left_margin, max_left_margin)
    right_gap = random.randint(RIGHT_GAP_RANGE[0], RIGHT_GAP_RANGE[1])
    min_piece_col = template_left + template_w + right_gap
    if min_piece_col >= grid_w:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[template_top : template_top + template_h, template_left : template_left + template_w] = template_patch

    piece_order = list(range(hole_count))
    random.shuffle(piece_order)
    shape_to_color = {_shape_key(shape): int(color) for shape, color in zip(shapes, colors, strict=False)}
    for idx in piece_order:
        if not _place_piece(grid, piece_orientations[idx], int(colors[idx]), min_piece_col):
            return None

    expected = _expected_output(
        shape_to_color=shape_to_color,
        hole_specs=hole_specs,  # type: ignore[arg-type]
        template_top=template_top,
        template_left=template_left,
        template_patch=template_patch,
        grid_shape=grid.shape,
    )
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
