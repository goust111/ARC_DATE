from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CONTOUR = 5
CENTER = 4
ARM = 7

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHO_DIRS:
        yield r + dr, c + dc


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comps: dict[int, list[tuple[int, int]]] = {}
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q = deque([(r, c)])
            labels[r, c] = next_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))
            comps[next_id] = coords
    return labels, comps


def _largest_two_ids(comps: dict[int, list[tuple[int, int]]]) -> set[int]:
    ranked = sorted(comps.items(), key=lambda item: len(item[1]), reverse=True)
    return {comp_id for comp_id, _coords in ranked[:2]}


def _find_cross_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != CENTER:
                continue
            if any(grid[r + dr, c + dc] != ARM for dr, dc in ORTHO_DIRS):
                continue
            if any(grid[r + dr, c + dc] != BG for dr, dc in DIAG_DIRS):
                continue
            centers.append((r, c))
    return centers


def _cross_region_id(bg_labels: np.ndarray, center: tuple[int, int]) -> int:
    r, c = center
    for dr, dc in DIAG_DIRS:
        nr, nc = r + dr, c + dc
        comp_id = int(bg_labels[nr, nc])
        if comp_id != 0:
            return comp_id
    raise AssertionError("cross must expose a background corner to identify its region")


def _expand_crosses(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    enclosed_centers: set[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in centers:
        if (r, c) in enclosed_centers:
            ortho_color = CENTER
            diag_color = ARM
        else:
            ortho_color = ARM
            diag_color = CENTER

        out[r, c] = CENTER
        for step in (1, 2):
            out[r - step, c] = ortho_color
            out[r + step, c] = ortho_color
            out[r, c - step] = ortho_color
            out[r, c + step] = ortho_color
            out[r - step, c - step] = diag_color
            out[r - step, c + step] = diag_color
            out[r + step, c - step] = diag_color
            out[r + step, c + step] = diag_color
    return out


def _fill_enclosed_background(grid: np.ndarray) -> np.ndarray:
    bg_labels, comps = _label_mask(grid == BG)
    exterior_ids = _largest_two_ids(comps)
    out = grid.copy()
    for comp_id in comps:
        if comp_id not in exterior_ids:
            out[bg_labels == comp_id] = CONTOUR
    return out


def _analyze_input(grid: np.ndarray) -> tuple[list[tuple[int, int]], set[tuple[int, int]], dict[int, int]]:
    bg_labels, comps = _label_mask(grid == BG)
    exterior_ids = _largest_two_ids(comps)
    centers = _find_cross_centers(grid)
    enclosed: set[tuple[int, int]] = set()
    component_sizes = {comp_id: len(coords) for comp_id, coords in comps.items()}
    for center in centers:
        comp_id = _cross_region_id(bg_labels, center)
        if comp_id not in exterior_ids:
            enclosed.add(center)
    return centers, enclosed, component_sizes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the orthogonal/diagonal growth roles for enclosed crosses, then fill the remaining enclosed background.

    Input structure:
    - Black background with gray barrier pixels: several closed gray contour loops and one long gray separator that splits
      the outside background into two large components.
    - Yellow/orange 3x3 crosses are scattered both inside contour interiors and in the outside field.
    - Each closed contour contains exactly one cross, and every cross has a clear 5x5 expansion footprint.

    Rule:
    - Treat every non-black pixel as a barrier and split the black background into 4-connected components; the two largest
      components are the exterior background.
    - For a cross whose surrounding background belongs to a non-exterior component, reverse the growth roles: yellow grows
      on the orthogonal arms while orange grows on the diagonals.
    - For a cross in one of the two exterior components, keep the standard roles: orange grows orthogonally and yellow grows
      diagonally.
    - After all crosses expand to their 5x5 motifs, recompute the background components and recolor every non-exterior
      background cell gray.

    Key insight:
    - The same 3x3 cross must first be classified by enclosure status before its two colors can be assigned to orthogonal
      versus diagonal growth, and the final gray fill is a second component-level operation on the post-growth background.
    """
    g = np.asarray(grid, dtype=int)
    centers, enclosed_centers, _sizes = _analyze_input(g)
    expanded = _expand_crosses(g, centers, enclosed_centers)
    return _fill_enclosed_background(expanded)


def _draw_rect_loop(grid: np.ndarray, top: int, left: int, inner_h: int, inner_w: int) -> tuple[int, int, int, int]:
    bottom = top + inner_h + 1
    right = left + inner_w + 1
    grid[top, left : right + 1] = CONTOUR
    grid[bottom, left : right + 1] = CONTOUR
    grid[top : bottom + 1, left] = CONTOUR
    grid[top : bottom + 1, right] = CONTOUR
    return top, left, bottom, right


def _paint_cross(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    grid[r, c] = CENTER
    for dr, dc in ORTHO_DIRS:
        grid[r + dr, c + dc] = ARM


def _footprint_is_clear(grid: np.ndarray, center: tuple[int, int]) -> bool:
    r, c = center
    h, w = grid.shape
    if not (2 <= r < h - 2 and 2 <= c < w - 2):
        return False
    patch = grid[r - 2 : r + 3, c - 2 : c + 3]
    return np.all(patch == BG)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return not (ab + margin < bt or bb + margin < at or ar + margin < bl or br + margin < al)


def _sample_external_center(
    grid: np.ndarray,
    side: str,
    divider_index: int,
    orientation: str,
    blocked_boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if orientation == "vertical":
                if side == "a" and c >= divider_index:
                    continue
                if side == "b" and c <= divider_index:
                    continue
            else:
                if side == "a" and r >= divider_index:
                    continue
                if side == "b" and r <= divider_index:
                    continue
            if not _footprint_is_clear(grid, (r, c)):
                continue
            box = (r - 2, c - 2, r + 2, c + 2)
            if any(_bbox_overlap(box, other, margin=1) for other in blocked_boxes):
                continue
            candidates.append((r, c))
    return random.choice(candidates) if candidates else None


def _region_counts(grid: np.ndarray) -> list[int]:
    _labels, comps = _label_mask(grid == BG)
    return sorted((len(coords) for coords in comps.values()), reverse=True)


def _all_standard(grid: np.ndarray) -> np.ndarray:
    centers = _find_cross_centers(grid)
    expanded = _expand_crosses(grid, centers, set())
    return _fill_enclosed_background(expanded)


def _all_reversed(grid: np.ndarray) -> np.ndarray:
    centers = _find_cross_centers(grid)
    expanded = _expand_crosses(grid, centers, set(centers))
    return _fill_enclosed_background(expanded)


def _validate_case(grid: np.ndarray, contour_boxes: list[tuple[int, int, int, int]]) -> bool:
    centers, enclosed_centers, component_sizes = _analyze_input(grid)
    if len(centers) < 4:
        return False
    if len(enclosed_centers) != len(contour_boxes):
        return False
    if len(centers) == len(enclosed_centers):
        return False

    size_values = sorted(component_sizes.values(), reverse=True)
    if len(size_values) < len(contour_boxes) + 2:
        return False
    if size_values[1] <= size_values[2]:
        return False

    for top, left, bottom, right in contour_boxes:
        inside = [
            center
            for center in enclosed_centers
            if top < center[0] < bottom and left < center[1] < right
        ]
        if len(inside) != 1:
            return False

    expanded = _expand_crosses(grid, centers, enclosed_centers)
    out = _fill_enclosed_background(expanded)
    if np.array_equal(expanded, out):
        return False
    if np.array_equal(out, _all_standard(grid)):
        return False
    if np.array_equal(out, _all_reversed(grid)):
        return False

    post_sizes = _region_counts(expanded)
    if len(post_sizes) < len(contour_boxes) + 2:
        return False
    if post_sizes[1] <= post_sizes[2]:
        return False

    newly_filled = int(np.count_nonzero((out == CONTOUR) & (expanded == BG)))
    if newly_filled == 0:
        return False
    return True


def _place_contour(
    grid: np.ndarray,
    side: str,
    divider_index: int,
    orientation: str,
    contour_boxes: list[tuple[int, int, int, int]],
    blocked_boxes: list[tuple[int, int, int, int]],
) -> bool:
    h, w = grid.shape
    inner_h = random.randint(6, 8)
    inner_w = random.randint(6, 8)
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    if orientation == "vertical":
        row_lo, row_hi = 1, h - outer_h - 1
        if side == "a":
            col_lo, col_hi = 1, divider_index - outer_w - 2
        else:
            col_lo, col_hi = divider_index + 2, w - outer_w - 1
    else:
        col_lo, col_hi = 1, w - outer_w - 1
        if side == "a":
            row_lo, row_hi = 1, divider_index - outer_h - 2
        else:
            row_lo, row_hi = divider_index + 2, h - outer_h - 1

    if row_lo > row_hi or col_lo > col_hi:
        return False

    positions = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(positions)
    for top, left in positions:
        box = (top, left, top + outer_h - 1, left + outer_w - 1)
        if any(_bbox_overlap(box, other, margin=2) for other in blocked_boxes):
            continue

        _draw_rect_loop(grid, top, left, inner_h, inner_w)
        center_r = random.randint(top + 3, top + inner_h - 2)
        center_c = random.randint(left + 3, left + inner_w - 2)
        _paint_cross(grid, (center_r, center_c))
        contour_boxes.append(box)
        blocked_boxes.append(box)
        blocked_boxes.append((center_r - 2, center_c - 2, center_r + 2, center_c + 2))
        return True
    return False


def _generate_case() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = random.randint(20, 28)
    w = random.randint(20, 28)
    grid = np.full((h, w), BG, dtype=int)

    orientation = random.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        divider_index = random.randint(max(7, w // 3), min(w - 8, (2 * w) // 3))
        grid[:, divider_index] = CONTOUR
    else:
        divider_index = random.randint(max(7, h // 3), min(h - 8, (2 * h) // 3))
        grid[divider_index, :] = CONTOUR

    contour_count = random.randint(2, 3)
    if contour_count == 2:
        side_order = ["a", "b"]
    else:
        side_order = ["a", "b", random.choice(["a", "b"])]
    random.shuffle(side_order)

    contour_boxes: list[tuple[int, int, int, int]] = []
    blocked_boxes: list[tuple[int, int, int, int]] = []

    if orientation == "vertical":
        blocked_boxes.append((0, divider_index, h - 1, divider_index))
    else:
        blocked_boxes.append((divider_index, 0, divider_index, w - 1))

    for side in side_order:
        if not _place_contour(grid, side, divider_index, orientation, contour_boxes, blocked_boxes):
            raise ValueError("failed to place contour")

    external_count = random.randint(2, 4)
    ext_sides = ["a", "b"]
    while len(ext_sides) < external_count:
        ext_sides.append(random.choice(["a", "b"]))
    random.shuffle(ext_sides)

    for side in ext_sides:
        center = _sample_external_center(grid, side, divider_index, orientation, blocked_boxes)
        if center is None:
            raise ValueError("failed to place external cross")
        _paint_cross(grid, center)
        r, c = center
        blocked_boxes.append((r - 2, c - 2, r + 2, c + 2))

    return grid, contour_boxes


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, contour_boxes = _generate_case()
        except ValueError:
            continue
        if not _validate_case(grid, contour_boxes):
            continue
        if not np.array_equal(main(grid), _fill_enclosed_background(_expand_crosses(grid, _find_cross_centers(grid), _analyze_input(grid)[1]))):
            continue
        return grid
    raise RuntimeError("failed to generate a valid contour-enclosure role-reversal puzzle")
