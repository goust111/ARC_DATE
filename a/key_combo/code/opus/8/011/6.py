from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PATTERN_COLORS = (BLUE, RED, GREEN, YELLOW)
DIAG_COLORS = (RED, GREEN)
GEN_ATTEMPTS = 500
SLOT = 4


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == int(color)
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _foreground_component_count(grid: np.ndarray, connectivity: int) -> int:
    mask = grid != BLACK
    visited = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    count = 0
    nbrs = neighbors4 if int(connectivity) == 4 else neighbors8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in nbrs(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _crop_pattern(top: np.ndarray) -> np.ndarray:
    coords = np.argwhere(top != BLACK)
    assert coords.size > 0, "top region must contain a non-black pattern"
    return crop_bbox(top, bbox_coords(coords))


def _is_full_2x2(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    return int(comp["area"]) == 4 and (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2


def _full_green_blocks(diag: np.ndarray) -> list[dict]:
    return [comp for comp in _components_of_color(diag, GREEN) if _is_full_2x2(comp)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the complete green 2x2 diagnosis blocks and use that count as the zoom factor for the upper pattern.

    Input structure:
    - The background is black.
    - One full grey row separates the grid into an upper pattern region and a lower diagnosis region.
    - The upper region contains one small multicolor pattern in blue, red, green, and yellow on black.
    - The lower region contains scattered red and green objects, including some complete green 2x2 squares and distractors.

    Rule:
    - Find the unique full grey separator row and split the grid above and below it.
    - Crop the upper region to the bounding box of its non-black pattern pixels.
    - In the lower region, inspect only green 4-connected components and keep only those whose bounding box is exactly 2x2 and whose area is exactly 4.
    - Let N be the number of such complete green squares, then replace each cropped pattern pixel with an N x N solid block of the same color.
    - Output only the enlarged cropped pattern.

    Key insight:
    - The upper and lower regions play different roles: the lower region encodes the scale factor by counting only complete green 2x2 squares, while the upper region provides the content to enlarge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep = _find_separator_row(g)
    assert 0 < sep < g.shape[0] - 1, "separator must have non-empty regions on both sides"

    pattern = _crop_pattern(g[:sep, :])
    assert pattern.shape[0] <= 4 and pattern.shape[1] <= 4, "cropped pattern should stay small"

    diag = g[sep + 1 :, :]
    n = len(_full_green_blocks(diag))
    assert 2 <= n <= 4, "expected 2 to 4 complete green blocks"
    return scale_sprite(pattern, n)


def _random_sprite(height: int, width: int, scale: int) -> np.ndarray:
    distinct = random.choice([k for k in (2, 3, 4) if k != scale])
    palette = random.sample(PATTERN_COLORS, distinct)
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)

    sprite = np.empty((height, width), dtype=int)
    for idx, color in enumerate(palette):
        r, c = cells[idx]
        sprite[r, c] = int(color)
    for r, c in cells[len(palette) :]:
        sprite[r, c] = int(random.choice(palette))

    assert len(set(map(int, sprite.flatten()))) == distinct
    return sprite


def _shape_variants() -> list[np.ndarray]:
    base_l = np.array([[GREEN, GREEN], [GREEN, BLACK]], dtype=int)
    variants = []
    seen = set()
    for k in range(4):
        arr = np.rot90(base_l, k)
        key = tuple(map(tuple, arr))
        if key not in seen:
            seen.add(key)
            variants.append(arr.copy())
    variants.extend(
        [
            np.array([[GREEN]], dtype=int),
            np.array([[GREEN, GREEN]], dtype=int),
            np.array([[GREEN], [GREEN]], dtype=int),
            np.array([[GREEN, GREEN, GREEN]], dtype=int),
            np.array([[GREEN], [GREEN], [GREEN]], dtype=int),
        ]
    )
    return variants


GREEN_NOISE_SHAPES = _shape_variants()


def _safe_offsets(patch: np.ndarray) -> tuple[int, int]:
    h, w = patch.shape
    max_r = max(0, 3 - h)
    max_c = max(0, 3 - w)
    return random.randint(0, max_r), random.randint(0, max_c)


def _stamp(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _slot_anchors(slot_rows: int, slot_cols: int) -> list[tuple[int, int]]:
    anchors = []
    for sr in range(slot_rows):
        for sc in range(slot_cols):
            anchors.append((sr * SLOT, sc * SLOT))
    random.shuffle(anchors)
    return anchors


def _break_one_full_square(diag: np.ndarray) -> np.ndarray:
    broken = diag.copy()
    full = _full_green_blocks(diag)
    assert full, "expected at least one full green square to break"
    rr, cc = map(int, full[0]["coords"][0])
    broken[rr, cc] = BLACK
    return broken


def _validate_scene(grid: np.ndarray, expected_scale: int, expected_sprite: np.ndarray) -> None:
    sep = _find_separator_row(grid)
    top = grid[:sep, :]
    diag = grid[sep + 1 :, :]

    sprite = _crop_pattern(top)
    assert np.array_equal(sprite, expected_sprite)

    green_comps = _components_of_color(diag, GREEN)
    red_comps = _components_of_color(diag, RED)
    full_green = [comp for comp in green_comps if _is_full_2x2(comp)]
    full_red = [comp for comp in red_comps if _is_full_2x2(comp)]
    incomplete_green = [comp for comp in green_comps if not _is_full_2x2(comp)]

    assert len(full_green) == expected_scale
    assert len(full_red) >= 1
    assert len(incomplete_green) >= 1
    assert len(green_comps) > len(full_green), "green distractors must exist"
    assert len(set(map(int, sprite.flatten()))) != expected_scale, "sprite color count must not encode scale"
    assert _foreground_component_count(diag, 8) == len(green_comps) + len(red_comps), "diagnosis objects must be 8-separated"

    broken = _break_one_full_square(diag)
    assert len(_full_green_blocks(broken)) == expected_scale - 1

    out = main(grid)
    expected = scale_sprite(expected_sprite, expected_scale)
    assert np.array_equal(out, expected)
    assert max(grid.shape) <= 30 and max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    scale = random.randint(2, 4)
    sprite_h = random.randint(3, 4)
    sprite_w = random.randint(3, 4)
    sprite = _random_sprite(sprite_h, sprite_w, scale)

    red_blocks = random.randint(1, 3)
    green_noise = random.randint(2, 4)
    total_objects = scale + red_blocks + green_noise

    feasible_rows = [rows for rows in (2, 3) if math.ceil(total_objects / rows) <= 6]
    if not feasible_rows:
        return None
    slot_rows = random.choice(feasible_rows)
    min_cols = math.ceil(total_objects / slot_rows)
    slot_cols = random.randint(min_cols, 6)

    width = slot_cols * SLOT
    top_h = sprite_h + random.randint(1, 3)
    height = top_h + 1 + slot_rows * SLOT
    if max(height, width, sprite_h * scale, sprite_w * scale) > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    pr = random.randint(0, top_h - sprite_h)
    pc = random.randint(0, width - sprite_w)
    _stamp(grid, sprite, pr, pc)
    grid[top_h, :] = GREY

    objects = (
        [np.full((2, 2), GREEN, dtype=int)] * scale
        + [np.full((2, 2), RED, dtype=int)] * red_blocks
        + [random.choice(GREEN_NOISE_SHAPES).copy() for _ in range(green_noise)]
    )
    random.shuffle(objects)

    diag = grid[top_h + 1 :, :]
    anchors = _slot_anchors(slot_rows, slot_cols)
    if len(anchors) < len(objects):
        return None

    for patch, (ar, ac) in zip(objects, anchors):
        dr, dc = _safe_offsets(patch)
        _stamp(diag, patch, ar + dr, ac + dc)

    try:
        _validate_scene(grid, scale, sprite)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
