from __future__ import annotations

from collections import deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_BG = 0
_PALETTE = tuple(range(1, 10))
_GRID_SIZE_RANGE = (12, 18)
_MAX_OUTER_ATTEMPTS = 800
_MAX_PLACEMENT_ATTEMPTS = 80
_SHAPE_SIZE_OPTIONS = (
    (3, 4),
    (3, 5),
    (3, 6),
    (4, 4),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 5),
    (5, 6),
)


def _bbox_from_coords(coords: set[Coord] | list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(list(coords), dtype=int)
    if arr.size == 0:
        raise ValueError("bbox requires at least one coordinate")
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _coords_from_mask(mask: np.ndarray, *, offset: Coord = (0, 0)) -> set[Coord]:
    off_r, off_c = map(int, offset)
    return {
        (int(r) + off_r, int(c) + off_c)
        for r, c in np.argwhere(np.asarray(mask, dtype=bool))
    }


def _mask_from_coords(coords: set[Coord] | list[Coord], bbox: BBox | None = None) -> np.ndarray:
    if bbox is None:
        bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[int(r) - r0, int(c) - c0] = True
    return out


def _coords_are_lr_symmetric(coords: set[Coord] | list[Coord], bbox: BBox | None = None) -> bool:
    if not coords:
        return False
    mask = _mask_from_coords(coords, bbox=bbox)
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _is_4_connected(coords: set[Coord] | list[Coord]) -> bool:
    cells = set(coords)
    if not cells:
        return False
    queue = deque([next(iter(cells))])
    seen = {queue[0]}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _centered_row_bounds(full_width: int, row_width: int) -> tuple[int, int]:
    full_width = int(full_width)
    row_width = int(row_width)
    if full_width % 2 == 1:
        mid = full_width // 2
        half = row_width // 2
        return int(mid - half), int(mid + half)
    half = row_width // 2
    left = full_width // 2 - half
    right = full_width // 2 + half - 1
    return int(left), int(right)


def _build_symmetric_mask(row_widths: list[int], full_width: int) -> np.ndarray:
    h = len(row_widths)
    mask = np.zeros((h, int(full_width)), dtype=bool)
    for r, row_width in enumerate(row_widths):
        c0, c1 = _centered_row_bounds(int(full_width), int(row_width))
        mask[r, c0 : c1 + 1] = True
    return mask


def _sample_symmetric_mask() -> np.ndarray:
    for _ in range(300):
        height, width = random.choice(_SHAPE_SIZE_OPTIONS)
        low = 1 if width % 2 else 2
        widths = list(range(low, width + 1, 2))
        row_widths = [random.choice(widths) for _ in range(height)]
        row_widths[random.randrange(height)] = width
        area = sum(row_widths)
        if not (8 <= area <= 15):
            continue
        if len(set(row_widths)) == 1:
            continue
        if min(row_widths) == width:
            continue
        return _build_symmetric_mask(row_widths, width)
    raise RuntimeError("failed to sample a symmetric shape")


def _fits_with_margin(canvas_shape: tuple[int, int], patch_shape: tuple[int, int], top_left: Coord, margin: int = 1) -> bool:
    h, w = map(int, patch_shape)
    H, W = map(int, canvas_shape)
    r0, c0 = map(int, top_left)
    return margin <= r0 and margin <= c0 and r0 + h <= H - margin and c0 + w <= W - margin


def _random_top_left(canvas_shape: tuple[int, int], patch_shape: tuple[int, int], margin: int = 1) -> Coord:
    H, W = map(int, canvas_shape)
    h, w = map(int, patch_shape)
    r = random.randint(margin, H - h - margin)
    c = random.randint(margin, W - w - margin)
    return int(r), int(c)


def _render_input(
    canvas_shape: tuple[int, int],
    back_coords: set[Coord],
    back_color: int,
    front_coords: set[Coord],
    front_color: int,
) -> np.ndarray:
    grid = np.zeros(canvas_shape, dtype=int)
    for r, c in back_coords:
        grid[r, c] = int(back_color)
    for r, c in front_coords:
        grid[r, c] = int(front_color)
    return grid


def _expected_output(canvas_shape: tuple[int, int], back_coords: set[Coord], back_color: int) -> np.ndarray:
    grid = np.zeros(canvas_shape, dtype=int)
    for r, c in back_coords:
        grid[r, c] = int(back_color)
    return _crop_bbox(grid, _bbox_from_coords(back_coords))


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the fully hidden rear symmetric object from a two-layer partial overlap and crop it.

    Input structure:
    - The input is a black-background grid containing exactly two non-black colors.
    - Each color forms one visible 4-connected object, and each underlying full object is left-right symmetric
      about the vertical axis through its own bounding box.
    - One object is fully visible in front, while the other loses some pixels where the front object overlaps it.

    Rule:
    - For each non-black color, crop its visible pixels to their bounding box and test exact left-right symmetry.
    - The symmetric visible object is the front layer; the asymmetric visible object is the rear layer.
    - Reflect every visible rear-layer pixel across the rear object's bounding-box axis; whenever the mirrored
      position currently shows the front color, restore that position as a hidden rear-layer pixel.
    - Draw the recovered rear object alone on black and crop to its bounding box.

    Key insight:
    - Layer order is inferred from visible symmetry itself: the intact symmetric object must be in front, and that
      lets the front-colored overlap cells serve as evidence for which mirrored rear pixels were hidden.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = sorted(int(c) for c in np.unique(g) if int(c) != _BG)
    assert len(colors) == 2, "expected exactly two non-background colors"

    symmetry_by_color: dict[int, bool] = {}
    coords_by_color: dict[int, set[Coord]] = {}
    bbox_by_color: dict[int, BBox] = {}
    for color in colors:
        coords = _coords_from_mask(g == color)
        assert coords, "foreground color must be visible"
        coords_by_color[color] = coords
        bbox_by_color[color] = _bbox_from_coords(coords)
        symmetry_by_color[color] = _coords_are_lr_symmetric(coords, bbox=bbox_by_color[color])

    front_candidates = [color for color, symmetric in symmetry_by_color.items() if symmetric]
    back_candidates = [color for color, symmetric in symmetry_by_color.items() if not symmetric]
    assert len(front_candidates) == 1 and len(back_candidates) == 1, "expected one symmetric front and one asymmetric back"
    front_color = front_candidates[0]
    back_color = back_candidates[0]

    visible_back = coords_by_color[back_color]
    back_bbox = bbox_by_color[back_color]
    _, c0, _, c1 = back_bbox
    csum = c0 + c1

    recovered = set(visible_back)
    for r, c in tuple(visible_back):
        mirrored = (int(r), int(csum - c))
        if g[mirrored] == front_color:
            recovered.add(mirrored)

    assert _coords_are_lr_symmetric(recovered, bbox=back_bbox), "recovered rear object must be symmetric"

    out = np.zeros_like(g)
    for r, c in recovered:
        out[r, c] = int(back_color)
    return _crop_bbox(out, back_bbox)


def _generate_case(*, with_meta: bool = False):
    for _ in range(_MAX_OUTER_ATTEMPTS):
        canvas_shape = (
            random.randint(*_GRID_SIZE_RANGE),
            random.randint(*_GRID_SIZE_RANGE),
        )
        back_mask = _sample_symmetric_mask()
        front_mask = _sample_symmetric_mask()
        back_h, back_w = map(int, back_mask.shape)
        front_h, front_w = map(int, front_mask.shape)
        if canvas_shape[0] < max(back_h, front_h) + 2 or canvas_shape[1] < max(back_w, front_w) + 2:
            continue

        back_pos = _random_top_left(canvas_shape, back_mask.shape, margin=1)
        hide_left = random.random() < 0.5
        dr_lo = -front_h + 2
        dr_hi = back_h - 2
        dc_choices = (
            list(range(-front_w + 2, 0))
            if hide_left
            else list(range(1, back_w - 1))
        )
        if dr_lo > dr_hi or not dc_choices:
            continue

        front_color, back_color = random.sample(_PALETTE, 2)
        back_coords = _coords_from_mask(back_mask, offset=back_pos)

        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            dr = random.randint(dr_lo, dr_hi)
            dc = random.choice(dc_choices)
            front_pos = (back_pos[0] + dr, back_pos[1] + dc)
            if not _fits_with_margin(canvas_shape, front_mask.shape, front_pos, margin=1):
                continue

            front_coords = _coords_from_mask(front_mask, offset=front_pos)
            overlap = front_coords & back_coords
            if not (2 <= len(overlap) <= 5):
                continue
            if not _is_4_connected(overlap):
                continue

            visible_back = back_coords - overlap
            if not _is_4_connected(visible_back):
                continue

            back_bbox = _bbox_from_coords(back_coords)
            front_bbox = _bbox_from_coords(front_coords)
            visible_back_bbox = _bbox_from_coords(visible_back)
            if visible_back_bbox != back_bbox:
                continue
            if front_bbox[1] + front_bbox[3] == back_bbox[1] + back_bbox[3]:
                continue

            if not _coords_are_lr_symmetric(front_coords, bbox=front_bbox):
                continue
            if _coords_are_lr_symmetric(visible_back, bbox=back_bbox):
                continue

            sides = set()
            csum = back_bbox[1] + back_bbox[3]
            bad_overlap = False
            for r, c in overlap:
                if c in (back_bbox[1], back_bbox[3]):
                    bad_overlap = True
                    break
                mirrored = (int(r), int(csum - c))
                if mirrored == (r, c) or mirrored not in visible_back:
                    bad_overlap = True
                    break
                sides.add(int(np.sign(mirrored[1] - c)))
            if bad_overlap or len(sides) != 1:
                continue

            grid = _render_input(canvas_shape, back_coords, back_color, front_coords, front_color)
            visible_front = _coords_from_mask(grid == front_color)
            if visible_front != front_coords:
                continue
            if _coords_from_mask(grid == back_color) != visible_back:
                continue

            inferred = set(visible_back)
            for r, c in tuple(visible_back):
                mirrored = (int(r), int(csum - c))
                if grid[mirrored] == front_color:
                    inferred.add(mirrored)
            if inferred != back_coords:
                continue

            expected = _expected_output(canvas_shape, back_coords, back_color)
            try:
                actual = main(grid)
            except AssertionError:
                continue
            if not np.array_equal(actual, expected):
                continue

            case = {
                "input": grid,
                "output": expected,
                "front_color": int(front_color),
                "back_color": int(back_color),
                "front_bbox": front_bbox,
                "back_bbox": back_bbox,
                "overlap": overlap,
                "hide_left": bool(hide_left),
            }
            return case if with_meta else grid.copy()
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    return _generate_case(with_meta=False)
