from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
FRAME = 9
ORANGE = 7
GREEN = 3
YELLOW = 4

TEMPLATE_COLORS = [1, 2, 5, 6]
TEMPLATE_COUNT_CHOICES = [2, 3]
TEMPLATE_DIMS = [(2, 3), (3, 2), (3, 3)]
SCALE_CHOICES = [3, 4, 5]
TOP_MARGIN_CHOICES = [1, 2]
TEMPLATE_GAP_CHOICES = [2, 3]
FRAME_SIDE_MARGIN_CHOICES = [1, 2, 3]
FRAME_BOTTOM_MARGIN_CHOICES = [1, 2, 3]
GEN_ATTEMPTS = 400


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_component_coords(mask)) == 1


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(bool), np.ones((factor, factor), dtype=bool))


def _enum_masks(h: int, w: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    area_lo = 3
    area_hi = min(5, h * w - 1)
    for bits in range(1 << (h * w)):
        n = int(bits.bit_count())
        if n < area_lo or n > area_hi:
            continue
        flat = np.array([(bits >> k) & 1 for k in range(h * w)], dtype=bool)
        mask = flat.reshape(h, w)
        if not _is_connected(mask):
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        masks.append(mask)
    return masks


MASK_LIBRARY = {dims: _enum_masks(*dims) for dims in TEMPLATE_DIMS}


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "expected a maroon frame"
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.shape[0] >= 5 and patch.shape[1] >= 5
    assert np.all(patch[:, 0] == FRAME)
    assert np.all(patch[:, -1] == FRAME)
    assert np.all(patch[-1, :] == FRAME)
    assert np.all(np.isin(patch[0, :], [FRAME, GREEN]))
    assert np.count_nonzero(patch[0, :] == GREEN) >= 1
    return r0, c0, r1, c1


def _extract_templates(grid: np.ndarray, frame_r0: int) -> list[np.ndarray]:
    top = grid[:frame_r0, :]
    comps: list[np.ndarray] = []
    for color in sorted(set(map(int, np.unique(top))) - {BG}):
        if color == FRAME:
            continue
        for coords in _component_coords(top == color):
            r0, c0, r1, c1 = _bbox_coords(coords)
            patch = top[r0 : r1 + 1, c0 : c1 + 1] == color
            comps.append(patch)
    assert 2 <= len(comps) <= 3, "expected 2-3 upper templates"
    return comps


def _matched_scaled_mask(templates: list[np.ndarray], seed_mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ih, iw = map(int, shape)
    matches: list[np.ndarray] = []
    for template in templates:
        th, tw = template.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        sh = ih // th
        sw = iw // tw
        if sh != sw:
            continue
        scaled = _scale_mask(template, sh)
        if np.all(~seed_mask | scaled):
            matches.append(scaled)
    assert len(matches) == 1, "seed pixels must select exactly one template"
    return matches[0]


def _seed_row_active(obstacle_mask: np.ndarray, marker_col: int) -> np.ndarray:
    ih, iw = obstacle_mask.shape
    assert 0 <= marker_col < iw
    active = np.zeros(iw, dtype=bool)
    if obstacle_mask[0, marker_col]:
        return active

    left = marker_col
    while left - 1 >= 0 and not obstacle_mask[0, left - 1]:
        left -= 1
    right = marker_col
    while right + 1 < iw and not obstacle_mask[0, right + 1]:
        right += 1

    for col in range(left, right + 1):
        if not obstacle_mask[0, col] and (col - marker_col) % 2 == 0:
            active[col] = True
    return active


def _grow_single_tree(obstacle_mask: np.ndarray, marker_col: int) -> list[tuple[int, np.ndarray, int]]:
    ih, iw = obstacle_mask.shape
    active = _seed_row_active(obstacle_mask, marker_col)
    if not np.any(active):
        return []

    layers: list[tuple[int, np.ndarray, int]] = [(0, active.copy(), GREEN)]
    row = 0
    color = YELLOW
    while row + 2 < ih:
        children = np.zeros(iw, dtype=bool)
        if iw >= 3:
            children[1:-1] = active[:-2] & (~obstacle_mask[row, 1:-1]) & active[2:]
        children &= ~obstacle_mask[row + 2]
        if not np.any(children):
            break
        layers.append((row + 2, children.copy(), color))
        active = children
        row += 2
        color = GREEN if color == YELLOW else YELLOW
    return layers


def _merge_trees(obstacle_mask: np.ndarray, marker_cols: list[int]) -> np.ndarray:
    out = np.full(obstacle_mask.shape, BG, dtype=int)
    out[obstacle_mask] = ORANGE
    trees = [(int(col), _grow_single_tree(obstacle_mask, int(col))) for col in marker_cols]
    for _col, layers in sorted(trees, key=lambda item: item[0]):
        for row, mask, color in layers:
            place = mask & (out[row, :] == BG)
            out[row, place] = color
    return out


def _vertical_depth(obstacle_mask: np.ndarray, col: int) -> int:
    depth = 0
    for r in range(obstacle_mask.shape[0]):
        if obstacle_mask[r, col]:
            break
        depth += 1
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the selected upper template inside the lower frame, then grow alternating rows through its background gaps.

    Input structure:
    - Teal is the background, with 2-3 small monochrome template masks above one large maroon rectangular frame.
    - The frame interior contains a few orange seed pixels and the top border contains 1-2 green entry markers.
    - The frame interior size is an integer block-scale of exactly one upper template's bounding box.

    Rule:
    - Find the unique upper template whose integer-scaled foreground covers every orange seed inside the frame.
    - Reset the frame interior to teal and paint the full scaled template in orange; these orange cells are walls.
    - For each green marker, seed the first interior row by taking every other open cell in the top background interval aligned with that marker's column.
    - Independently grow that seed downward: two rows below the current active row, place children between active-background-active triplets, alternating child-row colors yellow, green, yellow, ...; children may only land on teal cells.
    - Merge the independently grown trees from left to right, so earlier markers keep any colliding cells.

    Key insight:
    - The orange completion is not just a fill task: its scaled holes define the only corridors where the interval-row growth can reproduce.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _find_frame(g)
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = interior == ORANGE
    assert np.any(seed_mask), "expected orange seeds inside the frame"

    templates = _extract_templates(g, r0)
    scaled_mask = _matched_scaled_mask(templates, seed_mask, interior.shape)

    marker_cols = [int(c - (c0 + 1)) for c in np.where(g[r0, c0 + 1 : c1] == GREEN)[0] + (c0 + 1)]
    assert 1 <= len(marker_cols) <= 2, "expected 1-2 green markers on the top border"

    filled_interior = _merge_trees(scaled_mask, marker_cols)
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = filled_interior
    return out


def _unique_seed_subset(
    scaled_masks: list[np.ndarray], target_index: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    target = scaled_masks[target_index]
    candidates = list(range(len(scaled_masks)))
    seeds: list[tuple[int, int]] = []
    cells = [tuple(map(int, rc)) for rc in np.argwhere(target)]
    rng.shuffle(cells)

    while len(candidates) > 1:
        best_cell = None
        best_elim = 0
        for cell in cells:
            if cell in seeds:
                continue
            elim = sum(1 for idx in candidates if idx != target_index and not scaled_masks[idx][cell])
            if elim > best_elim:
                best_elim = elim
                best_cell = cell
        if best_cell is None:
            return None
        seeds.append(best_cell)
        candidates = [idx for idx in candidates if scaled_masks[idx][best_cell]]

    remaining = [cell for cell in cells if cell not in seeds]
    rng.shuffle(remaining)
    max_extra = min(len(remaining), max(0, min(2, target.sum() - len(seeds) - 1)))
    for cell in remaining[:max_extra]:
        seeds.append(cell)
    return seeds


def _pick_templates(
    dims: tuple[int, int], count: int, scale: int, rng: random.Random
) -> tuple[list[np.ndarray], int, list[tuple[int, int]]] | None:
    library = MASK_LIBRARY[dims]
    order = list(range(len(library)))
    rng.shuffle(order)
    for target_idx in order:
        target = library[target_idx]
        other_idx = [idx for idx in order if idx != target_idx]
        rng.shuffle(other_idx)
        for chosen_other in other_idx:
            chosen = [target_idx, chosen_other]
            if count == 3:
                for third in other_idx:
                    if third in chosen:
                        continue
                    chosen_now = chosen + [third]
                    scaled_masks = [_scale_mask(library[idx], scale) for idx in chosen_now]
                    seeds = _unique_seed_subset(scaled_masks, 0, rng)
                    if seeds is not None:
                        return [library[idx] for idx in chosen_now], 0, seeds
            else:
                scaled_masks = [_scale_mask(library[idx], scale) for idx in chosen]
                seeds = _unique_seed_subset(scaled_masks, 0, rng)
                if seeds is not None:
                    return [library[idx] for idx in chosen], 0, seeds
    return None


def _tree_stats(layers: list[tuple[int, np.ndarray, int]]) -> tuple[int, int, int]:
    if not layers:
        return 0, 0, 0
    total = sum(int(mask.sum()) for _row, mask, _color in layers)
    child_rows = max(0, len(layers) - 1)
    deepest = max(int(row) for row, _mask, _color in layers)
    return total, child_rows, deepest


def _marker_candidates(obstacle_mask: np.ndarray) -> list[dict]:
    iw = obstacle_mask.shape[1]
    options: list[dict] = []
    for col in range(iw):
        if obstacle_mask[0, col]:
            continue
        if _vertical_depth(obstacle_mask, col) < 3:
            continue
        layers = _grow_single_tree(obstacle_mask, col)
        total, child_rows, deepest = _tree_stats(layers)
        if child_rows < 1 or total < 4 or deepest < 2:
            continue
        cells = {(int(row), int(c)) for row, mask, _color in layers for c in np.where(mask)[0]}
        options.append(
            {
                "col": int(col),
                "layers": layers,
                "cells": cells,
                "child_rows": int(child_rows),
                "total": int(total),
                "deepest": int(deepest),
            }
        )
    return options


def _pair_options(obstacle_mask: np.ndarray, singles: list[dict]) -> list[dict]:
    pairs: list[dict] = []
    for i in range(len(singles)):
        for j in range(i + 1, len(singles)):
            a = singles[i]
            b = singles[j]
            merged = _merge_trees(obstacle_mask, [a["col"], b["col"]])
            growth = int(np.count_nonzero((merged == GREEN) | (merged == YELLOW)))
            overlap = len(a["cells"] & b["cells"])
            pairs.append(
                {
                    "cols": [a["col"], b["col"]],
                    "growth": growth,
                    "overlap": overlap,
                    "deepest": max(a["deepest"], b["deepest"]),
                }
            )
    return pairs


def _render_template(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _oracle_output(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    scaled_mask: np.ndarray,
    marker_cols: list[int],
) -> np.ndarray:
    r0, c0, r1, c1 = frame_bbox
    out = grid.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = _merge_trees(scaled_mask, marker_cols)
    return out


def _attempt() -> np.ndarray | None:
    rng = random.Random()
    count = rng.choice(TEMPLATE_COUNT_CHOICES)
    dims = rng.choice(TEMPLATE_DIMS)
    scale = rng.choice(SCALE_CHOICES)
    ih = dims[0] * scale
    iw = dims[1] * scale
    if ih < 8:
        return None

    picked = _pick_templates(dims, count, scale, rng)
    if picked is None:
        return None
    masks, target_index, seeds = picked
    tagged_masks = [("target" if idx == target_index else "other", mask) for idx, mask in enumerate(masks)]
    rng.shuffle(tagged_masks)
    masks = [mask for _tag, mask in tagged_masks]
    target = next(mask for tag, mask in tagged_masks if tag == "target")
    target_scaled = _scale_mask(target, scale)

    single_markers = _marker_candidates(target_scaled)
    if not single_markers:
        return None
    pair_markers = _pair_options(target_scaled, single_markers)

    marker_cols: list[int] | None = None
    if pair_markers and rng.random() < 0.65:
        collision_pairs = [p for p in pair_markers if p["overlap"] > 0]
        deep_pairs = [p for p in pair_markers if p["deepest"] >= 4 and p["growth"] >= 8]
        candidates = collision_pairs or deep_pairs or pair_markers
        chosen = rng.choice(candidates)
        marker_cols = list(chosen["cols"])
    else:
        good_singles = [m for m in single_markers if m["deepest"] >= 4 and m["total"] >= 6]
        chosen_single = rng.choice(good_singles or single_markers)
        marker_cols = [int(chosen_single["col"])]

    oracle_interior = _merge_trees(target_scaled, marker_cols)
    growth_pixels = int(np.count_nonzero((oracle_interior == GREEN) | (oracle_interior == YELLOW)))
    if growth_pixels < 4:
        return None

    alt_outputs = []
    for alt in masks[1:]:
        alt_scaled = _scale_mask(alt, scale)
        alt_outputs.append(_merge_trees(alt_scaled, marker_cols))
    if not alt_outputs or not any(not np.array_equal(out, oracle_interior) for out in alt_outputs):
        return None

    template_gap = rng.choice(TEMPLATE_GAP_CHOICES)
    top_margin = rng.choice(TOP_MARGIN_CHOICES)
    frame_side_margin = rng.choice(FRAME_SIDE_MARGIN_CHOICES)
    frame_bottom_margin = rng.choice(FRAME_BOTTOM_MARGIN_CHOICES)

    template_h, template_w = dims
    template_span = count * template_w + (count - 1) * template_gap
    frame_w = iw + 2
    frame_h = ih + 2
    grid_w = max(template_span + 2 * frame_side_margin, frame_w + 2 * frame_side_margin)
    grid_w += rng.randint(0, 2)
    grid_h = top_margin + template_h + 2 + frame_h + frame_bottom_margin
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    colors = TEMPLATE_COLORS.copy()
    rng.shuffle(colors)
    template_colors = colors[:count]

    template_c0 = rng.randint(1, max(1, grid_w - template_span - 1))
    template_row = top_margin
    cursor = template_c0
    for mask, color in zip(masks, template_colors):
        patch = _render_template(mask, color)
        grid[template_row : template_row + template_h, cursor : cursor + template_w] = patch
        cursor += template_w + template_gap

    frame_row = template_row + template_h + 2
    frame_col = rng.randint(1, max(1, grid_w - frame_w - 1))
    r0, c0 = frame_row, frame_col
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    for rr, cc in seeds:
        grid[r0 + 1 + rr, c0 + 1 + cc] = ORANGE
    for col in marker_cols:
        if col == 0 or col == iw - 1:
            return None
        grid[r0, c0 + 1 + col] = GREEN

    frame_bbox = (r0, c0, r1, c1)
    expected = _oracle_output(grid, frame_bbox, target_scaled, marker_cols)

    # Semantic checks.
    if np.array_equal(grid, expected):
        return None
    if int(np.count_nonzero(expected == ORANGE)) <= int(np.count_nonzero(grid == ORANGE)):
        return None
    if not np.any(expected[r0 + 1 : r1, c0 + 1 : c1] == YELLOW):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    templates = _extract_templates(grid, r0)
    matched = []
    seed_mask = grid[r0 + 1 : r1, c0 + 1 : c1] == ORANGE
    for tpl in templates:
        th, tw = tpl.shape
        if ih % th != 0 or iw % tw != 0:
            continue
        factor_h = ih // th
        factor_w = iw // tw
        if factor_h != factor_w:
            continue
        scaled = _scale_mask(tpl, factor_h)
        if np.all(~seed_mask | scaled):
            matched.append(scaled)
    if len(matched) != 1:
        return None
    if not any(not np.all(~seed_mask | _scale_mask(tpl, scale)) for tpl in templates if tpl.shape == dims and not np.array_equal(tpl, target)):
        return None
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
