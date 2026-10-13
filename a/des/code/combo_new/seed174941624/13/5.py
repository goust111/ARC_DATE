from __future__ import annotations

import random
from typing import Optional

import numpy as np


BG = 0
ARC_MAX = 30

# Generator bounds.
RECT_COUNT_MIN = 2
RECT_COUNT_MAX = 5
SIDE_MIN = 2
SIDE_MAX = 8
MAX_GENERATION_ATTEMPTS = 500

# Two protected edges remain fully visible after later rectangles overlap them.
ADJACENT_PROTECTED_OPTIONS = (
    ("top", "left"),
    ("top", "right"),
    ("bottom", "left"),
    ("bottom", "right"),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Separate overlapped colored rectangles, reconstruct their full sizes, and repack them from smallest area to largest.

    Input structure:
    - The background is black.
    - Each non-black color belongs to one solid monochrome rectangle, but upper rectangles may cover parts of lower ones.
    - A color may remain visible only as corner pieces or border strips, yet its full bounding box is still exposed.

    Rule:
    - For each non-black color, collect all visible pixels and take their minimal bounding box as the rectangle's original full extent.
    - Compute the reconstructed area of each rectangle as height times width.
    - Build a black output grid whose height is the maximum rectangle height and whose width is the sum of rectangle widths plus one black separator column between neighbors.
    - Paint the reconstructed solid rectangles from left to right in ascending area order, top-aligned.

    Key insight:
    - Occlusion changes which pixels are visible, but not the full row span and column span of each color, so the task is bbox recovery followed by area-based packing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    rects: list[tuple[int, int, int, int, int, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each rectangle color must remain visible"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rects.append((h * w, int(r0), int(c0), h, w, color))

    assert rects, "expected at least one non-background rectangle"
    rects.sort(key=lambda item: (item[0], item[1], item[2]))

    out_h = max(item[3] for item in rects)
    out_w = sum(item[4] for item in rects) + len(rects) - 1
    assert max(out_h, out_w) <= ARC_MAX, "output must stay within ARC size bounds"

    out = np.full((out_h, out_w), bg, dtype=int)
    c = 0
    for _area, _r0, _c0, h, w, color in rects:
        out[:h, c : c + w] = color
        c += w + 1

    return out


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _overlap(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> Optional[tuple[int, int, int, int]]:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 <= r1 and c0 <= c1:
        return int(r0), int(c0), int(r1), int(c1)
    return None


def _sample_protected_edges(rng: random.Random, h: int, w: int) -> tuple[str, str]:
    opposite: list[tuple[str, str]] = []
    if h >= 3:
        opposite.append(("top", "bottom"))
    if w >= 3:
        opposite.append(("left", "right"))

    if opposite and rng.random() < 0.5:
        return rng.choice(opposite)
    return rng.choice(ADJACENT_PROTECTED_OPTIONS)


def _respects_protected_edges(rect: dict, upper_bbox: tuple[int, int, int, int]) -> bool:
    inter = _overlap(rect["bbox"], upper_bbox)
    if inter is None:
        return True

    ir0, ic0, ir1, ic1 = inter
    r0, c0, r1, c1 = rect["bbox"]
    protected = set(rect["protect"])

    if "top" in protected and ir0 <= r0:
        return False
    if "bottom" in protected and ir1 >= r1:
        return False
    if "left" in protected and ic0 <= c0:
        return False
    if "right" in protected and ic1 >= c1:
        return False
    return True


def _paint_scene(rects: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = rect["color"]
    return grid


def _protected_edges_intact(grid: np.ndarray, rect: dict) -> bool:
    r0, c0, r1, c1 = rect["bbox"]
    color = rect["color"]

    for edge in rect["protect"]:
        if edge == "top" and not np.all(grid[r0, c0 : c1 + 1] == color):
            return False
        if edge == "bottom" and not np.all(grid[r1, c0 : c1 + 1] == color):
            return False
        if edge == "left" and not np.all(grid[r0 : r1 + 1, c0] == color):
            return False
        if edge == "right" and not np.all(grid[r0 : r1 + 1, c1] == color):
            return False
    return True


def _choose_dimensions(rng: random.Random, count: int) -> Optional[list[tuple[int, int]]]:
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        while len(dims) < count:
            h = rng.randint(SIDE_MIN, SIDE_MAX)
            w = rng.randint(SIDE_MIN, SIDE_MAX)
            area = h * w
            if area in used_areas:
                continue
            dims.append((h, w))
            used_areas.add(area)
        if sum(w for _h, w in dims) + count - 1 <= ARC_MAX:
            return dims
    return None


def _place_rectangles(
    rng: random.Random, dims: list[tuple[int, int]], colors: list[int], shape: tuple[int, int]
) -> Optional[list[dict]]:
    grid_h, grid_w = shape
    layer_order = list(range(len(dims)))
    rng.shuffle(layer_order)

    placed: list[dict] = []
    overlapped_layers = 0

    for depth, rect_idx in enumerate(layer_order):
        h, w = dims[rect_idx]
        protect = _sample_protected_edges(rng, h, w)
        all_candidates: list[dict] = []
        overlap_candidates: list[dict] = []

        for r0 in range(grid_h - h + 1):
            for c0 in range(grid_w - w + 1):
                bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
                ok = True
                overlaps_lower = False
                for lower in placed:
                    if _overlap(lower["bbox"], bbox) is not None:
                        overlaps_lower = True
                        if not _respects_protected_edges(lower, bbox):
                            ok = False
                            break
                if not ok:
                    continue

                rect = {
                    "bbox": bbox,
                    "color": colors[rect_idx],
                    "h": h,
                    "w": w,
                    "protect": protect,
                }
                all_candidates.append(rect)
                if overlaps_lower:
                    overlap_candidates.append(rect)

        if not all_candidates:
            return None

        must_overlap = depth > 0 and overlapped_layers == 0
        want_overlap = depth > 0 and (must_overlap or rng.random() < 0.8)
        if must_overlap and not overlap_candidates:
            return None

        pool = overlap_candidates if want_overlap and overlap_candidates else all_candidates
        rect = rng.choice(pool)
        if any(_overlap(lower["bbox"], rect["bbox"]) is not None for lower in placed):
            overlapped_layers += 1
        placed.append(rect)

    return placed if overlapped_layers > 0 else None


def _validate_scene(grid: np.ndarray, rects: list[dict]) -> bool:
    if len(rects) < RECT_COUNT_MIN or len(rects) > RECT_COUNT_MAX:
        return False

    areas = [rect["h"] * rect["w"] for rect in rects]
    if len(areas) != len(set(areas)):
        return False

    visible_colors = {int(color) for color in np.unique(grid) if int(color) != BG}
    if visible_colors != {rect["color"] for rect in rects}:
        return False

    hidden_count = 0
    for rect in rects:
        mask = grid == rect["color"]
        if not mask.any():
            return False
        if _bbox_from_mask(mask) != rect["bbox"]:
            return False
        if int(mask.sum()) < rect["h"] * rect["w"]:
            hidden_count += 1
        if not _protected_edges_intact(grid, rect):
            return False

    if hidden_count == 0:
        return False

    filled_ratio = float(np.count_nonzero(grid)) / float(grid.size)
    if not (0.08 <= filled_ratio <= 0.75):
        return False

    out = main(grid)
    return bool(max(grid.shape) <= ARC_MAX and max(out.shape) <= ARC_MAX)


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    count = rng.randint(RECT_COUNT_MIN, RECT_COUNT_MAX)
    dims = _choose_dimensions(rng, count)
    if dims is None:
        return None

    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)
    grid_h = rng.randint(max(max_h + 2, 10), min(24, max_h + 10))
    grid_w = rng.randint(max(max_w + 2, 10), min(24, max_w + 10))
    colors = rng.sample(list(range(1, 10)), count)

    rects = _place_rectangles(rng, dims, colors, (grid_h, grid_w))
    if rects is None:
        return None

    grid = _paint_scene(rects, (grid_h, grid_w))
    if not _validate_scene(grid, rects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(
        f"failed to generate a valid occluded-rectangle scene after {MAX_GENERATION_ATTEMPTS} attempts"
    )
