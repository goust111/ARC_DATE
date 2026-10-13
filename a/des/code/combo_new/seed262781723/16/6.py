from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
TILE_REPLACE_COLOR = 1
TILE_KEEP_COLOR = 8
PAIR_COLOR_POOL = (2, 3, 4, 6, 7, 9)
TILE_SIZE_CHOICES = ((2, 2), (2, 3), (3, 2), (3, 3))
GENERATION_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    shifted = coords - np.array([r0, c0], dtype=int)
    order = np.lexsort((shifted[:, 1], shifted[:, 0]))
    return tuple((int(shifted[i, 0]), int(shifted[i, 1])) for i in order)


def _rect_frame_candidate(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 4 or w < 4:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    if h > 2 and w > 2 and np.any(mask[1:-1, 1:-1]):
        return False
    return True


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], np.ndarray]:
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _component_coords(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
            if not _rect_frame_candidate(patch):
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior == bg):
                continue
            colors = np.unique(interior)
            if len(colors) != 2:
                continue
            candidates.append((color, (r0, c0, r1, c1), interior.copy()))
    assert len(candidates) == 1, "expected one framed template region"
    return candidates[0]


def _tile_roles(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], tile: np.ndarray
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    colors = [int(color) for color in np.unique(tile)]
    counts = {color: int(np.count_nonzero(grid[outside] == color)) for color in colors}
    inside_counts = {color: int(np.count_nonzero(tile == color)) for color in colors}
    ordered = sorted(colors, key=lambda color: (counts[color], -inside_counts[color]))
    replace_color, keep_color = ordered[0], ordered[1]
    assert replace_color != keep_color
    return replace_color, keep_color


def _find_pairs(
    grid: np.ndarray, bg: int, banned_colors: set[int]
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg or color in banned_colors:
            continue
        components = _component_coords(grid == color)
        if len(components) != 2:
            continue
        if len(components[0]) < 2 or len(components[1]) < 2:
            continue
        if _normalize(components[0]) != _normalize(components[1]):
            continue
        bbox_a = _bbox(components[0])
        bbox_b = _bbox(components[1])
        a_r0, a_c0, a_r1, a_c1 = bbox_a
        b_r0, b_c0, b_r1, b_c1 = bbox_b
        orientation: str | None = None
        if a_r0 == b_r0 and a_r1 == b_r1:
            left, right = sorted((bbox_a, bbox_b), key=lambda box: box[1])
            if left[3] + 1 <= right[1] - 1:
                orientation = "h"
                ordered_boxes = (left, right)
        elif a_c0 == b_c0 and a_c1 == b_c1:
            top, bottom = sorted((bbox_a, bbox_b), key=lambda box: box[0])
            if top[2] + 1 <= bottom[0] - 1:
                orientation = "v"
                ordered_boxes = (top, bottom)
        if orientation is None:
            continue
        pairs.append(
            {
                "color": color,
                "orientation": orientation,
                "boxes": ordered_boxes,
            }
        )
    assert pairs, "expected at least one valid sprite pair"
    return pairs


def _paint_corridor(
    out: np.ndarray,
    original: np.ndarray,
    pair: dict[str, object],
    tile: np.ndarray,
    bg: int,
) -> None:
    box_a, box_b = pair["boxes"]  # type: ignore[assignment]
    orientation = pair["orientation"]  # type: ignore[assignment]
    th, tw = tile.shape
    if orientation == "h":
        r0, _, r1, c1_left = box_a
        _, c0_right, _, _ = box_b
        c_start = c1_left + 1
        for r in range(r0, r1 + 1):
            for c in range(c_start, c0_right):
                if original[r, c] != bg:
                    continue
                out[r, c] = int(tile[(r - r0) % th, (c - c_start) % tw])
    else:
        r0_top, c0, r1_top, c1 = box_a
        r0_bottom, _, _, _ = box_b
        r_start = r1_top + 1
        for r in range(r_start, r0_bottom):
            for c in range(c0, c1 + 1):
                if original[r, c] != bg:
                    continue
                out[r, c] = int(tile[(r - r_start) % th, (c - c0) % tw])


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each corridor between twin same-color sprites with a recolored copy of the framed template tile.

    Input structure:
    - The background is the dominant color. Away from one framed template area,
      several colors each form exactly two identical monochrome sprites.
    - Each twin-sprite pair is aligned either horizontally or vertically, leaving
      a rectangular corridor of background cells between their bounding boxes.
    - A unique one-pixel-thick rectangular frame encloses a two-color alternating
      tile sample; one tile color also appears outside the frame as blocker pixels.

    Rule:
    - Locate the unique framed sample and read its interior as the tile to repeat.
    - Treat the tile color that stays confined to the sample as the replaceable
      placeholder; keep the other tile color unchanged.
    - For every same-color twin-sprite pair, determine whether its corridor runs
      horizontally or vertically, replace the placeholder tile color by that pair's
      sprite color, and tile the corridor.
    - Only recolor background pixels inside the corridor; blockers, the original
      sprites, and the framed sample itself remain unchanged.

    Key insight:
    - The framed region is not another object to connect; it is a reusable tiling
      legend whose placeholder color must be swapped with each pair color before
      painting the gap between the paired sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frame_color, frame_bbox, tile = _find_template(g, bg)
    replace_color, _keep_color = _tile_roles(g, frame_bbox, tile)
    banned_colors = {frame_color}
    banned_colors.update(int(color) for color in np.unique(tile))
    pairs = _find_pairs(g, bg, banned_colors)

    out = g.copy()
    for pair in pairs:
        pair_color = int(pair["color"])
        paint_tile = np.where(tile == replace_color, pair_color, tile)
        _paint_corridor(out, g, pair, paint_tile, bg)
    return out


def _connected(mask: np.ndarray) -> bool:
    comps = _component_coords(mask)
    return len(comps) == 1


def _valid_sprite_mask(mask: np.ndarray, orientation: str) -> bool:
    if int(mask.sum()) < 3:
        return False
    if not _connected(mask):
        return False
    if orientation == "h":
        if not np.all(mask.any(axis=1)):
            return False
        if not mask[:, 0].any() or not mask[:, -1].any():
            return False
    else:
        if not np.all(mask.any(axis=0)):
            return False
        if not mask[0, :].any() or not mask[-1, :].any():
            return False
    return True


def _make_sprite_mask(orientation: str, cross_span: int, long_span: int, rng: np.random.Generator) -> np.ndarray:
    if orientation == "h":
        h, w = cross_span, long_span
    else:
        h, w = long_span, cross_span
    mask = np.ones((h, w), dtype=bool)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    max_removals = max(0, min(len(cells) - 3, (h * w) // 3))
    target_removals = int(rng.integers(0, max_removals + 1))
    removed = 0
    for r, c in cells:
        if removed >= target_removals:
            break
        trial = mask.copy()
        trial[r, c] = False
        if _valid_sprite_mask(trial, orientation):
            mask = trial
            removed += 1
    assert _valid_sprite_mask(mask, orientation)
    return mask


def _reserve(rect_mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = rect_mask.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    rect_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _checkerboard(tile_h: int, tile_w: int) -> np.ndarray:
    tile = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            tile[r, c] = TILE_REPLACE_COLOR if (r + c) % 2 == 0 else TILE_KEEP_COLOR
    return tile


def _place_template(grid: np.ndarray, reserved: np.ndarray, tile: np.ndarray, rng: np.random.Generator) -> None:
    h, w = grid.shape
    th, tw = tile.shape
    frame_h = th + 2
    frame_w = tw + 2
    corners = [
        (1, 1),
        (1, w - frame_w - 1),
        (h - frame_h - 1, 1),
        (h - frame_h - 1, w - frame_w - 1),
    ]
    r0, c0 = corners[int(rng.integers(0, len(corners)))]
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR
    grid[r0 + 1 : r1, c0 + 1 : c1] = tile
    _reserve(reserved, (r0, c0, r1, c1), pad=1)


def _orientation_plan(num_pairs: int, rng: np.random.Generator) -> list[str]:
    if num_pairs == 2:
        plan = ["h", "v"]
    elif num_pairs == 3:
        plan = ["h", "v", "h" if rng.random() < 0.5 else "v"]
    else:
        plan = ["h", "h", "v", "v"]
    rng.shuffle(plan)
    return plan


def _try_place_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    orientation: str,
    tile_shape: tuple[int, int],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    tile_h, tile_w = tile_shape
    if orientation == "h":
        sprite_h = tile_h
        sprite_w = int(rng.integers(2, 6))
        gap = int(rng.integers(2, 9))
        if sprite_h + 2 > h or sprite_w * 2 + gap + 2 > w:
            return False
        mask = _make_sprite_mask("h", sprite_h, sprite_w, rng)
        for _ in range(120):
            r0 = int(rng.integers(0, h - sprite_h + 1))
            c0 = int(rng.integers(0, w - (sprite_w * 2 + gap) + 1))
            region = (r0, c0, r0 + sprite_h - 1, c0 + sprite_w * 2 + gap - 1)
            r0r, c0r, r1r, c1r = region
            if reserved[r0r : r1r + 1, c0r : c1r + 1].any():
                continue
            left = grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
            right_c = c0 + sprite_w + gap
            right = grid[r0 : r0 + sprite_h, right_c : right_c + sprite_w]
            left[mask] = color
            right[mask] = color
            _reserve(reserved, region, pad=1)
            return True
    else:
        sprite_w = tile_w
        sprite_h = int(rng.integers(2, 6))
        gap = int(rng.integers(2, 9))
        if sprite_w + 2 > w or sprite_h * 2 + gap + 2 > h:
            return False
        mask = _make_sprite_mask("v", sprite_w, sprite_h, rng)
        for _ in range(120):
            r0 = int(rng.integers(0, h - (sprite_h * 2 + gap) + 1))
            c0 = int(rng.integers(0, w - sprite_w + 1))
            region = (r0, c0, r0 + sprite_h * 2 + gap - 1, c0 + sprite_w - 1)
            r0r, c0r, r1r, c1r = region
            if reserved[r0r : r1r + 1, c0r : c1r + 1].any():
                continue
            top = grid[r0 : r0 + sprite_h, c0 : c0 + sprite_w]
            bottom_r = r0 + sprite_h + gap
            bottom = grid[bottom_r : bottom_r + sprite_h, c0 : c0 + sprite_w]
            top[mask] = color
            bottom[mask] = color
            _reserve(reserved, region, pad=1)
            return True
    return False


def _pair_corridor_background(
    grid: np.ndarray, color: int, orientation: str
) -> list[tuple[int, int]]:
    comps = _component_coords(grid == color)
    boxes = sorted((_bbox(comp) for comp in comps), key=lambda box: (box[0], box[1]))
    cells: list[tuple[int, int]] = []
    if orientation == "h":
        left, right = sorted(boxes, key=lambda box: box[1])
        for r in range(left[0], left[2] + 1):
            for c in range(left[3] + 1, right[1]):
                if grid[r, c] == BG:
                    cells.append((r, c))
    else:
        top, bottom = sorted(boxes, key=lambda box: box[0])
        for r in range(top[2] + 1, bottom[0]):
            for c in range(top[1], top[3] + 1):
                if grid[r, c] == BG:
                    cells.append((r, c))
    return cells


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(18, 31))
    width = int(rng.integers(18, 31))
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    tile_h, tile_w = TILE_SIZE_CHOICES[int(rng.integers(0, len(TILE_SIZE_CHOICES)))]
    tile = _checkerboard(tile_h, tile_w)
    _place_template(grid, reserved, tile, rng)

    num_pairs = int(rng.integers(2, 5))
    orientations = _orientation_plan(num_pairs, rng)
    pair_colors = list(rng.choice(np.array(PAIR_COLOR_POOL, dtype=int), size=num_pairs, replace=False))
    placed: list[tuple[int, str]] = []
    for color, orientation in zip(pair_colors, orientations):
        if not _try_place_pair(grid, reserved, int(color), orientation, (tile_h, tile_w), rng):
            return None
        placed.append((int(color), orientation))

    placed_blocker = False
    for index, (color, orientation) in enumerate(placed):
        corridor = _pair_corridor_background(grid, color, orientation)
        if not corridor:
            return None
        want_blockers = rng.random() < 0.7 or (index == len(placed) - 1 and not placed_blocker)
        if not want_blockers:
            continue
        max_blockers = min(3, len(corridor) - 1)
        if max_blockers <= 0:
            continue
        blocker_count = int(rng.integers(1, max_blockers + 1))
        chosen = rng.choice(len(corridor), size=blocker_count, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = corridor[int(idx)]
            grid[r, c] = TILE_KEEP_COLOR
            placed_blocker = True

    if not placed_blocker:
        return None
    if _most_common_color(grid) != BG:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
