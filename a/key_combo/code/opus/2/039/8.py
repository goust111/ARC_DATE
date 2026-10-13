from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SEP = 5
TEMPLATE_COLOR = 8
TARGET_COLORS = (1, 2, 4)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

TOP_H_RANGE = (7, 12)
BOTTOM_H_RANGE = (6, 12)
LEFT_W_RANGE = (7, 12)
RIGHT_W_RANGE = (6, 12)
GEN_RETRY_LIMIT = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _anchor_from_bbox(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 1 and w % 2 == 1, "template bbox center must land on a grid cell"
    return r0 + h // 2, c0 + w // 2


def _connected_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "component must be non-empty"
    start = tuple(map(int, coords[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    q.append((nr, nc))
    assert len(seen) == int(mask.sum()), "template must be exactly one 4-connected component"
    return np.array(sorted(seen), dtype=int)


def _find_separators(grid: np.ndarray) -> tuple[int, int]:
    row_candidates = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    col_candidates = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(row_candidates) == 1 and len(col_candidates) == 1, "expected one full grey row and one full grey column"
    return int(row_candidates[0]), int(col_candidates[0])


def _region_bounds(shape: tuple[int, int], sep_r: int, sep_c: int) -> dict[str, tuple[int, int, int, int]]:
    h, w = map(int, shape)
    return {
        "tl": (0, sep_r, 0, sep_c),
        "tr": (0, sep_r, sep_c + 1, w),
        "bl": (sep_r + 1, h, 0, sep_c),
        "br": (sep_r + 1, h, sep_c + 1, w),
    }


def _extract_template_offsets(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    sub = grid[r0:r1, c0:c1]
    assert np.all((sub == BG) | (sub == TEMPLATE_COLOR)), "top-left region may only contain background and the teal template"
    coords = _connected_component(sub == TEMPLATE_COLOR)
    box = _bbox_coords(coords)
    ar, ac = _anchor_from_bbox(box)
    assert any(int(r) == ar and int(c) == ac for r, c in coords), "template anchor cell must be occupied"
    return sorted((int(r) - ar, int(c) - ac) for r, c in coords)


def _extract_target(grid: np.ndarray, bounds: tuple[int, int, int, int]) -> tuple[int, int, int]:
    r0, r1, c0, c1 = bounds
    sub = grid[r0:r1, c0:c1]
    coords = np.argwhere((sub != BG) & (sub != SEP))
    assert coords.shape[0] == 1, "each non-template region must contain exactly one target pixel"
    rr, cc = map(int, coords[0])
    return r0 + rr, c0 + cc, int(sub[rr, cc])


def _stamp_positions(
    anchor: tuple[int, int],
    offsets: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    ar, ac = anchor
    r0, r1, c0, c1 = bounds
    out = []
    for dr, dc in offsets:
        rr = ar + dr
        cc = ac + dc
        if r0 <= rr < r1 and c0 <= cc < c1:
            out.append((rr, cc))
    return out


def _solve(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_r, sep_c = _find_separators(g)
    regions = _region_bounds(g.shape, sep_r, sep_c)
    offsets = _extract_template_offsets(g, regions["tl"])

    out = g.copy()
    out[:sep_r, :sep_c][g[:sep_r, :sep_c] == TEMPLATE_COLOR] = BG

    stamp_sizes: dict[str, int] = {}
    for name in ("tr", "bl", "br"):
        tr, tc, color = _extract_target(g, regions[name])
        positions = _stamp_positions((tr, tc), offsets, regions[name])
        assert positions, "each target region must retain at least the anchor cell"
        for rr, cc in positions:
            out[rr, cc] = color
        stamp_sizes[name] = len(positions)

    if return_meta:
        return out, {"offsets": offsets, "stamp_sizes": stamp_sizes, "sep": (sep_r, sep_c)}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the top-left teal template into the other partition cells while clipping it against the grey cross boundaries.

    Input structure:
    - A black background contains one full grey row and one full grey column, splitting the canvas into four rectangular regions.
    - The top-left region contains exactly one 4-connected teal template shape.
    - Each of the other three regions contains exactly one isolated colored target pixel.

    Rule:
    - Read the teal template by its occupied cells relative to the center of its bounding box.
    - For each non-top-left region, place the same offsets around that region's lone target pixel and recolor every kept template cell to the target color.
    - Clip away every stamped cell that would cross a grey separator or leave the target region, erase the original teal template to black, and keep the grey separator lines unchanged.

    Key insight:
    - The four partitions are not independent pictures: the top-left region stores one reusable template, and the grey cross acts as a hard clipping mask when that template is re-anchored on the three target pixels.
    """
    return _solve(grid)


def _is_connected_cells(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    q = deque([next(iter(cells))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nbr = (r + dr, c + dc)
            if nbr in cells and nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return seen == cells


def _build_template_catalog() -> list[list[tuple[int, int]]]:
    templates: set[tuple[tuple[int, int], ...]] = set()
    center_bit = 1 << 4
    for bits in range(1 << 9):
        if not (bits & center_bit):
            continue
        area = bits.bit_count()
        if not 5 <= area <= 7:
            continue
        cells = {
            (idx // 3 - 1, idx % 3 - 1)
            for idx in range(9)
            if (bits >> idx) & 1
        }
        if not _is_connected_cells(cells):
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != -1 or max(rows) != 1 or min(cols) != -1 or max(cols) != 1:
            continue
        templates.add(tuple(sorted(cells)))

    templates.add(tuple(sorted({(-1, 0), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (1, 0)})))
    templates.add(tuple(sorted({(-2, 0), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (2, 0)})))
    return [list(template) for template in sorted(templates)]


TEMPLATE_CATALOG = _build_template_catalog()


def _place_template(
    grid: np.ndarray,
    offsets: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> bool:
    r0, r1, c0, c1 = bounds
    rows = [dr for dr, _ in offsets]
    cols = [dc for _, dc in offsets]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    anchor_rows = list(range(r0 + 1 - min_r, r1 - 1 - max_r))
    anchor_cols = list(range(c0 + 1 - min_c, c1 - 1 - max_c))
    if not anchor_rows or not anchor_cols:
        return False

    ar = random.choice(anchor_rows)
    ac = random.choice(anchor_cols)
    for dr, dc in offsets:
        grid[ar + dr, ac + dc] = TEMPLATE_COLOR
    return True


def _candidate_targets(
    bounds: tuple[int, int, int, int],
    offsets: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    r0, r1, c0, c1 = bounds
    area = len(offsets)
    preferred_min_visible = max(2, (area + 1) // 2)
    full: list[tuple[int, int]] = []
    cropped_preferred: list[tuple[int, int]] = []
    cropped_any: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            visible = len(_stamp_positions((r, c), offsets, bounds))
            if visible == area:
                full.append((r, c))
            elif 2 <= visible < area:
                cropped_any.append((r, c))
                if visible >= preferred_min_visible:
                    cropped_preferred.append((r, c))
    return full, cropped_preferred, cropped_any


def _choose_target_layout(
    regions: dict[str, tuple[int, int, int, int]],
    offsets: list[tuple[int, int]],
) -> dict[str, tuple[int, int]] | None:
    candidates = {
        name: _candidate_targets(bounds, offsets)
        for name, bounds in regions.items()
        if name != "tl"
    }
    full_regions = [name for name, (full, _, _) in candidates.items() if full]
    if not full_regions:
        return None

    full_region = random.choice(full_regions)
    layout: dict[str, tuple[int, int]] = {}
    for name, (full, cropped_preferred, cropped_any) in candidates.items():
        if name == full_region:
            layout[name] = random.choice(full)
        else:
            pool = cropped_preferred if cropped_preferred else cropped_any
            if not pool:
                return None
            layout[name] = random.choice(pool)
    return layout


def _generate_attempt() -> np.ndarray | None:
    top_h = random.randint(*TOP_H_RANGE)
    bottom_h = random.randint(*BOTTOM_H_RANGE)
    left_w = random.randint(*LEFT_W_RANGE)
    right_w = random.randint(*RIGHT_W_RANGE)
    height = top_h + 1 + bottom_h
    width = left_w + 1 + right_w

    grid = np.full((height, width), BG, dtype=int)
    grid[top_h, :] = SEP
    grid[:, left_w] = SEP

    regions = _region_bounds(grid.shape, top_h, left_w)
    offsets = random.choice(TEMPLATE_CATALOG)
    if not _place_template(grid, offsets, regions["tl"]):
        return None

    layout = _choose_target_layout(regions, offsets)
    if layout is None:
        return None

    region_names = ["tr", "bl", "br"]
    colors = list(TARGET_COLORS)
    random.shuffle(colors)
    for name, color in zip(region_names, colors, strict=True):
        r, c = layout[name]
        grid[r, c] = color

    out, meta = _solve(grid, return_meta=True)
    stamp_sizes = meta["stamp_sizes"]
    if sorted(color for color in colors) != sorted(TARGET_COLORS):
        return None
    if not any(size < len(offsets) for size in stamp_sizes.values()):
        return None
    if not any(size == len(offsets) for size in stamp_sizes.values()):
        return None
    if min(stamp_sizes.values()) < 2:
        return None
    if np.any(out[top_h, :] != SEP) or np.any(out[:, left_w] != SEP):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid clipped-template stamping puzzle")
