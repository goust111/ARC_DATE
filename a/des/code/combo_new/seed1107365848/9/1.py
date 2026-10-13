import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
OBJECT_PLACE_TRIES = 240


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    best: tuple[tuple[int, int, int], tuple[int, int, int, int]] | None = None
    for c in range(w - 1):
        pair_rows = (grid[:, c] == GREY) & (grid[:, c + 1] != BG) & (grid[:, c + 1] != GREY)
        r = 0
        while r < h:
            if not bool(pair_rows[r]):
                r += 1
                continue
            r0 = r
            while r + 1 < h and bool(pair_rows[r + 1]):
                r += 1
            r1 = r
            score = (c, r1, r1 - r0 + 1)
            box = (r0, c, r1, c + 1)
            if best is None or score > best[0]:
                best = (score, box)
            r += 1
    if best is None:
        raise AssertionError("expected a vertical grey-to-color legend")

    r0, c0, r1, c1 = best[1]
    mapping = {idx + 1: int(grid[r0 + idx, c1]) for idx in range(r1 - r0 + 1)}
    mask = np.zeros_like(grid, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mapping, mask


def _parse_objects(grid: np.ndarray) -> tuple[dict[int, int], list[dict]]:
    mapping, _legend_mask = _find_legend(grid)
    black_components = _components(grid == BG)

    regions: list[dict] = []
    h, w = grid.shape
    for coords in black_components:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        regions.append({"coords": coords, "bbox": _bbox(coords)})

    parent: list[int | None] = [None] * len(regions)
    for child in range(len(regions)):
        containers = [
            parent_idx
            for parent_idx in range(len(regions))
            if parent_idx != child and _strictly_contains(regions[parent_idx]["bbox"], regions[child]["bbox"])
        ]
        if containers:
            parent[child] = min(containers, key=lambda idx: _bbox_area(regions[idx]["bbox"]))

    def root_of(idx: int) -> int:
        root = idx
        while parent[root] is not None:
            root = int(parent[root])
        return root

    objects: list[dict] = []
    for idx, p in enumerate(parent):
        if p is not None:
            continue
        hole_count = sum(1 for parent_idx in parent if parent_idx == idx)
        fill_indices = [region_idx for region_idx in range(len(regions)) if root_of(region_idx) == idx]
        objects.append({"hole_count": hole_count, "fill_indices": fill_indices, "regions": regions})
    return mapping, objects


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mapping, objects = _parse_objects(g)
    out = g.copy()
    for obj in objects:
        hole_count = int(obj["hole_count"])
        if hole_count not in mapping:
            continue
        fill_color = int(mapping[hole_count])
        regions = obj["regions"]
        for region_idx in obj["fill_indices"]:
            for r, c in regions[region_idx]["coords"]:
                out[r, c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded interior using the legend color indexed by its hole count.

    Input structure:
    - The background is black.
    - Several grey outline objects enclose black interior regions, and each interior contains one or more smaller grey-bounded black holes.
    - A vertical legend near the bottom-right consists of rows of two adjacent cells: grey on the left and a color on the right.

    Rule:
    - Read the legend from top to bottom so row 1 gives the fill color for objects with 1 hole, row 2 for objects with 2 holes, and so on.
    - Find every enclosed top-level black interior region, count how many smaller enclosed black hole regions lie inside it, and use that count as the legend index.
    - Recolor every non-grey pixel inside that outer contour, including both the main interior background and its holes, with the selected legend color.

    Key insight:
    - The small holes are the counting key, while the legend row number converts that topological count into the final fill color.
    """
    return _solve(grid)


def _draw_rect_border(block: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    block[top, left : left + width] = color
    block[top + height - 1, left : left + width] = color
    block[top : top + height, left] = color
    block[top : top + height, left + width - 1] = color


def _layout_for_count(hole_count: int, rng: random.Random) -> list[tuple[int, int]]:
    layouts = {
        1: [[(0, 0)]],
        2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)]],
        3: [
            [(0, 0), (0, 1), (1, 0)],
            [(0, 0), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (0, 2)],
            [(0, 0), (1, 0), (2, 0)],
        ],
        4: [
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [(0, 0), (0, 1), (1, 1), (1, 2)],
            [(0, 1), (1, 0), (1, 1), (2, 1)],
            [(0, 0), (0, 1), (0, 2), (1, 1)],
        ],
    }
    return list(rng.choice(layouts[hole_count]))


def _make_object(hole_count: int, rng: random.Random) -> np.ndarray:
    layout = _layout_for_count(hole_count, rng)
    hole_sizes = [rng.choice((1, 2)) for _ in range(hole_count)]

    n_rows = 1 + max(r for r, _ in layout)
    n_cols = 1 + max(c for _, c in layout)
    row_heights = [0] * n_rows
    col_widths = [0] * n_cols
    slot_sizes = [size + 4 for size in hole_sizes]

    for idx, (rr, cc) in enumerate(layout):
        row_heights[rr] = max(row_heights[rr], slot_sizes[idx])
        col_widths[cc] = max(col_widths[cc], slot_sizes[idx])

    row_gaps = [rng.randint(0, 1) for _ in range(n_rows - 1)]
    col_gaps = [rng.randint(0, 1) for _ in range(n_cols - 1)]
    top_slack = rng.randint(1, 2)
    bottom_slack = rng.randint(1, 2)
    left_slack = rng.randint(1, 2)
    right_slack = rng.randint(1, 2)

    inner_h = top_slack + bottom_slack + sum(row_heights) + sum(row_gaps)
    inner_w = left_slack + right_slack + sum(col_widths) + sum(col_gaps)
    block = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    _draw_rect_border(block, 0, 0, inner_h + 2, inner_w + 2, GREY)

    row_offsets = [0] * n_rows
    col_offsets = [0] * n_cols
    row_cursor = 1 + top_slack
    for rr in range(n_rows):
        row_offsets[rr] = row_cursor
        row_cursor += row_heights[rr]
        if rr < n_rows - 1:
            row_cursor += row_gaps[rr]
    col_cursor = 1 + left_slack
    for cc in range(n_cols):
        col_offsets[cc] = col_cursor
        col_cursor += col_widths[cc]
        if cc < n_cols - 1:
            col_cursor += col_gaps[cc]

    for idx, (rr, cc) in enumerate(layout):
        slot = slot_sizes[idx]
        cell_h = row_heights[rr]
        cell_w = col_widths[cc]
        jitter_r = rng.randint(0, cell_h - slot)
        jitter_c = rng.randint(0, cell_w - slot)
        ring_size = hole_sizes[idx] + 2
        ring_top = row_offsets[rr] + jitter_r + 1
        ring_left = col_offsets[cc] + jitter_c + 1
        _draw_rect_border(block, ring_top, ring_left, ring_size, ring_size, GREY)

    return np.rot90(block, rng.randrange(4)).copy()


def _stamp_nonzero(grid: np.ndarray, block: np.ndarray, top: int, left: int) -> None:
    h, w = block.shape
    patch = grid[top : top + h, left : left + w]
    mask = block != BG
    patch[mask] = block[mask]


def _mark_blocked(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _place_objects(
    grid: np.ndarray,
    blocked: np.ndarray,
    objects: list[tuple[int, np.ndarray]],
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    objects_sorted = sorted(objects, key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)
    for _hole_count, block in objects_sorted:
        bh, bw = block.shape
        if bh + 2 > h or bw + 2 > w:
            return False
        placed = False
        for _ in range(OBJECT_PLACE_TRIES):
            top = rng.randint(1, h - bh - 1)
            left = rng.randint(1, w - bw - 1)
            if bool(blocked[top : top + bh, left : left + bw].any()):
                continue
            _stamp_nonzero(grid, block, top, left)
            _mark_blocked(blocked, top, left, bh, bw, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _legend_block(colors: list[int]) -> np.ndarray:
    legend = np.zeros((len(colors), 2), dtype=int)
    legend[:, 0] = GREY
    legend[:, 1] = np.array(colors, dtype=int)
    return legend


def _expected_fill_counts(grid: np.ndarray) -> list[int]:
    _, objects = _parse_objects(grid)
    return sorted(int(obj["hole_count"]) for obj in objects)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_len = rng.choice((2, 3, 4))
    legend_colors = rng.sample(list(PALETTE_COLORS), legend_len)
    object_count = 2 if legend_len == 2 else rng.choice((2, 3))
    used_counts = sorted(rng.sample(list(range(1, legend_len + 1)), object_count))
    object_blocks = [(count, _make_object(count, rng)) for count in used_counts]

    max_obj_h = max(block.shape[0] for _, block in object_blocks)
    max_obj_w = max(block.shape[1] for _, block in object_blocks)
    reserve_area = sum((block.shape[0] + 2) * (block.shape[1] + 2) for _, block in object_blocks) + (legend_len + 2) * 6
    side_hint = int(np.ceil(np.sqrt(reserve_area)))
    min_h = max(16, max_obj_h + 2, side_hint - 1, legend_len + 4)
    min_w = max(16, max_obj_w + 5, side_hint + 1)
    if min_h > 30 or min_w > 30:
        return None

    h = rng.randint(min_h, 30)
    w = rng.randint(min_w, 30)
    if rng.random() < 0.35:
        h, w = w, h
    if h > 30 or w > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    legend = _legend_block(legend_colors)
    leg_h, leg_w = legend.shape
    bottom_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    leg_top = h - leg_h - bottom_margin
    leg_left = w - leg_w - right_margin
    if leg_top < 1 or leg_left < 1:
        return None
    _stamp_nonzero(grid, legend, leg_top, leg_left)
    _mark_blocked(blocked, leg_top, leg_left, leg_h, leg_w, pad=1)

    if not _place_objects(grid, blocked, object_blocks, rng):
        return None

    try:
        mapping, _objects = _parse_objects(grid)
    except AssertionError:
        return None
    if mapping != {idx + 1: legend_colors[idx] for idx in range(legend_len)}:
        return None
    if _expected_fill_counts(grid) != used_counts:
        return None

    out = _solve(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
