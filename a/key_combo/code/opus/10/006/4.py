from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6

STRUCT4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
STRUCT8 = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

GEN_HEIGHT_RANGE = (14, 17)
GEN_WIDTH_RANGE = (18, 24)
GEN_SEPARATOR_ROW = 5
GEN_TEMPLATE_H_RANGE = (3, 4)
GEN_TEMPLATE_W_RANGE = (4, 5)
GEN_TEMPLATE_AREA_RANGE = (8, 12)
GEN_COPY_COUNT_CHOICES = (2, 3)
GEN_MAX_ATTEMPTS = 400


def _coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(mask: np.ndarray) -> np.ndarray:
    box = _bbox_from_mask(mask)
    if box is None:
        return np.zeros((0, 0), dtype=bool)
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    deltas = STRUCT4 if int(connectivity) == 4 else STRUCT8
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[np.ndarray] = []
    H, W = mask.shape
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in deltas:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(sorted(cells), dtype=int))
    parts.sort(key=lambda arr: tuple(map(int, arr.min(axis=0))))
    return parts


def _is_connected(mask: np.ndarray, connectivity: int = 8) -> bool:
    cells = int(np.count_nonzero(mask))
    return cells > 0 and len(_components(mask, connectivity=connectivity)) == 1


def _all_rows_cols_used(mask: np.ndarray) -> bool:
    return bool(mask.shape[0]) and bool(mask.shape[1]) and np.all(mask.any(axis=1)) and np.all(mask.any(axis=0))


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for mirrored in (False, True):
            variant = np.fliplr(rotated) if mirrored else rotated
            cropped = _crop_to_bbox(variant)
            key = cropped.astype(np.uint8).tobytes() + repr(cropped.shape).encode()
            if key not in seen:
                seen.add(key)
                variants.append(cropped)
    return variants


def _dihedral_triplets(red_mask: np.ndarray, blue_mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    seen: set[bytes] = set()
    for k in range(4):
        red_rot = np.rot90(red_mask, k)
        blue_rot = np.rot90(blue_mask, k)
        for mirrored in (False, True):
            red_var = np.fliplr(red_rot) if mirrored else red_rot
            blue_var = np.fliplr(blue_rot) if mirrored else blue_rot
            open_var = red_var | blue_var
            box = _bbox_from_mask(open_var)
            assert box is not None
            r0, c0, r1, c1 = box
            open_crop = open_var[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            red_crop = red_var[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            blue_crop = blue_var[r0 : r1 + 1, c0 : c1 + 1].astype(bool)
            key = open_crop.astype(np.uint8).tobytes() + repr(open_crop.shape).encode()
            if key not in seen:
                seen.add(key)
                out.append((red_crop, blue_crop, open_crop))
    return out


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return int(rows[0])


def _extract_template_parts(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red_box = _bbox_from_mask(grid == RED)
    blue_box = _bbox_from_mask(grid == BLUE)
    assert red_box is not None and blue_box is not None, "expected one red part and one blue part"
    red_h = red_box[2] - red_box[0] + 1
    red_w = red_box[3] - red_box[1] + 1
    blue_h = blue_box[2] - blue_box[0] + 1
    blue_w = blue_box[3] - blue_box[1] + 1
    assert (red_h, red_w) == (blue_h, blue_w), "red and blue masks must align by bbox"

    red_crop = (grid[red_box[0] : red_box[2] + 1, red_box[1] : red_box[3] + 1] == RED)
    blue_crop = (grid[blue_box[0] : blue_box[2] + 1, blue_box[1] : blue_box[3] + 1] == BLUE)
    assert red_crop.shape == blue_crop.shape
    assert np.any(red_crop) and np.any(blue_crop), "both template fragments must be non-empty"
    assert not np.any(red_crop & blue_crop), "template fragments must be complementary"
    template = red_crop | blue_crop
    assert _all_rows_cols_used(template), "template bbox must be tight"
    return red_crop, blue_crop, template


def _match_components(
    grid: np.ndarray,
    template_mask: np.ndarray,
) -> list[dict[str, object]]:
    sep = _find_separator_row(grid)
    lower_top = sep + 1
    magenta_mask = np.zeros_like(grid, dtype=bool)
    magenta_mask[lower_top:, :] = grid[lower_top:, :] == PINK
    components = _components(magenta_mask, connectivity=8)
    assert components, "expected at least one magenta copy"

    variants = _dihedral_variants(template_mask)
    H, W = grid.shape
    matches: list[dict[str, object]] = []

    for comp in components:
        comp_set = {tuple(map(int, rc)) for rc in comp}
        candidates: list[dict[str, object]] = []
        for variant in variants:
            open_cells = _coords(variant)
            vh, vw = variant.shape
            for top in range(lower_top, H - vh + 1):
                for left in range(0, W - vw + 1):
                    placed = {(top + r, left + c) for r, c in open_cells}
                    if not comp_set.issubset(placed):
                        continue
                    values = [int(grid[r, c]) for r, c in placed]
                    if any(v not in (BLACK, PINK) for v in values):
                        continue
                    placed_magenta = {(r, c) for r, c in placed if int(grid[r, c]) == PINK}
                    if placed_magenta != comp_set:
                        continue
                    fill_cells = sorted(placed - comp_set)
                    if not fill_cells:
                        continue
                    candidates.append(
                        {
                            "component": comp,
                            "top": top,
                            "left": left,
                            "variant": variant,
                            "fill_cells": fill_cells,
                        }
                    )

        assert len(candidates) == 1, "each magenta cluster must determine a unique oriented placement"
        matches.append(candidates[0])
    return matches


def _apply_template(grid: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for match in _match_components(out, template_mask):
        for r, c in match["fill_cells"]:
            out[int(r), int(c)] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge the separated red and blue fragments into one hidden template, then complete every magenta rotated/reflected copy with green.

    Input structure:
    - A black grid contains one full-width grey separator row that splits an upper red fragment from a lower blue fragment.
    - The red and blue pixels occupy aligned bounding boxes and never overlap within that aligned local frame.
    - Elsewhere in the lower region, 2-3 isolated magenta clusters show partial rotated or reflected copies of the same hidden template.

    Rule:
    - Extract the red and blue local masks from their tight aligned bounding boxes and take their cellwise union to recover the template's open cells.
    - For each magenta cluster, test every rotation and reflection of that template over the lower region and keep the unique placement whose open cells contain exactly that cluster's magenta subset.
    - Paint every missing open cell of the matched placement green, while leaving the original magenta cells, the separator, and the reference fragments unchanged.

    Key insight:
    - The template never appears directly: it only emerges after unioning the complementary red and blue masks, and that synthesized template is what disambiguates the incomplete transformed copies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _red_mask, _blue_mask, template_mask = _extract_template_parts(g)
    return _apply_template(g, template_mask)


def _random_connected_template() -> np.ndarray | None:
    h = random.randint(*GEN_TEMPLATE_H_RANGE)
    w = random.randint(*GEN_TEMPLATE_W_RANGE)
    min_area = max(GEN_TEMPLATE_AREA_RANGE[0], (h * w + 2) // 3)
    max_area = min(GEN_TEMPLATE_AREA_RANGE[1], h * w - 1)
    if min_area > max_area:
        return None
    target = random.randint(min_area, max_area)

    mask = np.zeros((h, w), dtype=bool)
    seed = (random.randrange(h), random.randrange(w))
    mask[seed] = True
    while int(np.count_nonzero(mask)) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in _coords(mask):
            for dr, dc in STRUCT8:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True

    if not _is_connected(mask, connectivity=8):
        return None
    if not _all_rows_cols_used(mask):
        return None
    if not np.all(mask.sum(axis=1) >= 2):
        return None
    if not np.all(mask.sum(axis=0) >= 2):
        return None
    if len(_dihedral_variants(mask)) != 8:
        return None
    return mask


def _random_split(template_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(200):
        red_mask = np.zeros_like(template_mask, dtype=bool)
        blue_mask = np.zeros_like(template_mask, dtype=bool)
        for r, c in _coords(template_mask):
            if random.random() < 0.5:
                red_mask[r, c] = True
            else:
                blue_mask[r, c] = True
        if np.any(red_mask) and np.any(blue_mask) and _all_rows_cols_used(red_mask) and _all_rows_cols_used(blue_mask):
            return red_mask, blue_mask
    return None


def _rects_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_connected_subset(mask: np.ndarray, target_size: int) -> np.ndarray | None:
    open_set = set(_coords(mask))
    seed = random.choice(tuple(open_set))
    subset = {seed}
    frontier = set()
    while len(subset) < target_size:
        frontier.clear()
        for r, c in subset:
            for dr, dc in STRUCT8:
                nb = (r + dr, c + dc)
                if nb in open_set and nb not in subset:
                    frontier.add(nb)
        if not frontier:
            return None
        subset.add(random.choice(tuple(frontier)))
    out = np.zeros_like(mask, dtype=bool)
    for r, c in subset:
        out[r, c] = True
    return out if _is_connected(out, connectivity=8) else None


def _component_cells_local(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _expected_output(
    grid: np.ndarray,
    copies: Iterable[dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for copy in copies:
        top = int(copy["top"])
        left = int(copy["left"])
        variant = np.asarray(copy["open_mask"], dtype=bool)
        visible = np.asarray(copy["visible_mask"], dtype=bool)
        for r, c in _coords(variant & ~visible):
            out[top + r, left + c] = GREEN
    return out


def _safe_apply_template(grid: np.ndarray, template_mask: np.ndarray) -> np.ndarray | None:
    try:
        return _apply_template(grid, template_mask)
    except AssertionError:
        return None


def _generate_attempt() -> np.ndarray | None:
    H = random.randint(*GEN_HEIGHT_RANGE)
    W = random.randint(*GEN_WIDTH_RANGE)
    if GEN_SEPARATOR_ROW + 2 >= H:
        return None

    template_mask = _random_connected_template()
    if template_mask is None:
        return None
    split = _random_split(template_mask)
    if split is None:
        return None
    red_mask, blue_mask = split
    triplets = _dihedral_triplets(red_mask, blue_mask)
    if len(triplets) != 8:
        return None

    th, tw = template_mask.shape
    upper_top_max = GEN_SEPARATOR_ROW - th
    if upper_top_max < 0:
        return None

    ref_top = random.randint(0, upper_top_max)
    ref_left_min = 2
    ref_left_max = max(ref_left_min, W - tw - 7)
    if ref_left_min > ref_left_max:
        return None
    ref_left = random.randint(ref_left_min, ref_left_max)
    blue_top = GEN_SEPARATOR_ROW + 1
    if blue_top + th > H:
        return None

    grid = np.zeros((H, W), dtype=int)
    grid[GEN_SEPARATOR_ROW, :] = GREY
    for r, c in _coords(red_mask):
        grid[ref_top + r, ref_left + c] = RED
    for r, c in _coords(blue_mask):
        grid[blue_top + r, ref_left + c] = BLUE

    ref_blue_rect = (blue_top, ref_left, blue_top + th - 1, ref_left + tw - 1)
    copy_count = random.choice(GEN_COPY_COUNT_CHOICES)
    variant_order = list(range(len(triplets)))
    random.shuffle(variant_order)

    placed_rects = [ref_blue_rect]
    copies: list[dict[str, object]] = []
    saw_red_completion = False
    saw_blue_completion = False

    for idx in range(copy_count):
        red_var, blue_var, open_var = triplets[variant_order[idx]]
        vh, vw = open_var.shape
        if GEN_SEPARATOR_ROW + 1 > H - vh:
            return None

        preferred_positions: list[tuple[int, int]] = []
        fallback_positions: list[tuple[int, int]] = []
        for top in range(GEN_SEPARATOR_ROW + 1, H - vh + 1):
            for left in range(0, W - vw + 1):
                rect = (top, left, top + vh - 1, left + vw - 1)
                if any(not _rects_separated(rect, other, pad=1) for other in placed_rects):
                    continue
                if top == blue_top and left == ref_left:
                    continue
                is_right = left > ref_left + tw + 1
                is_below = top > blue_top + th
                fallback_positions.append((top, left))
                if idx == 0 and is_right:
                    preferred_positions.append((top, left))
                elif idx > 0 and (is_right or is_below):
                    preferred_positions.append((top, left))

        positions = preferred_positions or fallback_positions
        if not positions:
            return None
        random.shuffle(positions)

        placed = False
        open_count = int(np.count_nonzero(open_var))
        min_visible = (open_count + 1) // 2
        for top, left in positions:
            for _ in range(40):
                visible_size = random.randint(min_visible, open_count - 1)
                visible_mask = _random_connected_subset(open_var, visible_size)
                if visible_mask is None:
                    continue
                if np.array_equal(visible_mask, open_var):
                    continue
                visible_cells = _component_cells_local(visible_mask)
                missing_red = any(red_var[r, c] and (r, c) not in visible_cells for r, c in _coords(red_var))
                missing_blue = any(blue_var[r, c] and (r, c) not in visible_cells for r, c in _coords(blue_var))
                local_fill = open_var & ~visible_mask
                if np.count_nonzero(local_fill) == 0:
                    continue
                copy_info = {
                    "top": top,
                    "left": left,
                    "open_mask": open_var.copy(),
                    "visible_mask": visible_mask.copy(),
                }
                copies.append(copy_info)
                rect = (top, left, top + vh - 1, left + vw - 1)
                placed_rects.append(rect)
                saw_red_completion = saw_red_completion or missing_red
                saw_blue_completion = saw_blue_completion or missing_blue
                for r, c in _coords(visible_mask):
                    grid[top + r, left + c] = PINK
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    if not (saw_red_completion and saw_blue_completion):
        return None

    expected = _expected_output(grid, copies)
    solved = _safe_apply_template(grid, template_mask)
    if solved is None or not np.array_equal(solved, expected):
        return None

    if _safe_apply_template(grid, red_mask) is not None and np.array_equal(_safe_apply_template(grid, red_mask), expected):
        return None
    if _safe_apply_template(grid, blue_mask) is not None and np.array_equal(_safe_apply_template(grid, blue_mask), expected):
        return None

    if np.array_equal(grid, expected):
        return None
    if len(set(np.unique(grid)) - {BLACK, BLUE, RED, GREY, PINK}) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
