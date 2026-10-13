from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
SELECTOR_COLOR = 5
MARKER_COLORS = (1, 2, 3, 4)
SELECTOR_SIZE = 7
SPRITE_SIZE = 3
MARKER_POSITIONS = ((1, 1), (1, 5), (5, 1), (5, 5))
QUADRANT_ANCHORS = {(1, 1): (0, 0), (1, 5): (0, 4), (5, 1): (4, 0), (5, 5): (4, 4)}
RANK_TO_ROT_K = {1: 0, 2: -1, 3: 2, 4: 1}

GRID_ROWS_RANGE = (16, 19)
GRID_COLS_RANGE = (16, 19)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: tuple[int, int, int, int]) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: tuple[int, int, int, int]) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]


def neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def components_mask(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_selector(grid: np.ndarray) -> tuple[int, int, int, int]:
    selector_mask = grid == SELECTOR_COLOR
    candidates = []
    for coords in components_mask(selector_mask, connectivity=4):
        box = bbox_coords(coords)
        if int(coords.shape[0]) != 45:
            continue
        if bbox_h(box) != SELECTOR_SIZE or bbox_w(box) != SELECTOR_SIZE:
            continue
        patch = crop_bbox(grid, box)
        marker_colors = [int(patch[r, c]) for r, c in MARKER_POSITIONS]
        if sorted(marker_colors) != list(MARKER_COLORS):
            continue
        if np.count_nonzero(patch == SELECTOR_COLOR) != 45:
            continue
        outside_markers = patch.copy()
        for r, c in MARKER_POSITIONS:
            outside_markers[r, c] = SELECTOR_COLOR
        if not np.all(outside_markers == SELECTOR_COLOR):
            continue
        candidates.append(box)

    assert len(candidates) == 1, "expected exactly one selector block"
    return candidates[0]


def _sprite_for_color(grid: np.ndarray, color: int, selector_box: tuple[int, int, int, int]) -> np.ndarray:
    outside = np.ones(grid.shape, dtype=bool)
    r0, c0, r1, c1 = selector_box
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = components_mask((grid == int(color)) & outside, connectivity=8)
    assert len(comps) == 1, f"expected exactly one external sprite for color {color}"
    coords = comps[0]
    box = bbox_coords(coords)
    assert bbox_h(box) == SPRITE_SIZE and bbox_w(box) == SPRITE_SIZE
    sprite = crop_bbox(grid, box)
    assert set(map(int, np.unique(sprite))) <= {BG, int(color)}
    return sprite


def main(grid: np.ndarray) -> np.ndarray:
    """Place each legend-matched sprite into its selector quadrant after rotating it by area rank.

    Input structure:
    - The grid has black background, one grey 7x7 selector block, and exactly four external monochrome sprites.
    - The selector contains one marker pixel at the center of each 3x3 quadrant, using the four colors blue, red, green, and yellow.
    - Outside the selector, each marker color appears in exactly one 8-connected sprite whose bounding box is 3x3.

    Rule:
    - Pair each external sprite with the selector marker of the same color.
    - Count each sprite's non-black pixels, rank the four counts from smallest to largest, and map ranks 1..4 to clockwise rotations 0, 90, 180, and 270 degrees.
    - Rotate each cropped 3x3 sprite by its rank-derived angle, then place it into the output quadrant indicated by its same-color marker; keep the middle row and column black.

    Key insight:
    - Placement is local color matching, but rotation is a global comparison: each sprite's angle depends on how its area ranks against all other sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    selector_box = _find_selector(g)
    selector = crop_bbox(g, selector_box)

    sprites_by_color: dict[int, np.ndarray] = {}
    areas: dict[int, int] = {}
    for color in MARKER_COLORS:
        sprite = _sprite_for_color(g, color, selector_box)
        sprites_by_color[int(color)] = sprite
        areas[int(color)] = int(np.count_nonzero(sprite != BG))

    ranked = sorted(areas.items(), key=lambda item: item[1])
    assert len({area for _, area in ranked}) == 4, "sprite areas must be unique"
    rank_by_color = {color: idx + 1 for idx, (color, _area) in enumerate(ranked)}

    out = np.zeros((SELECTOR_SIZE, SELECTOR_SIZE), dtype=int)
    for marker_pos in MARKER_POSITIONS:
        color = int(selector[marker_pos])
        sprite = sprites_by_color[color]
        rank = rank_by_color[color]
        rotated = np.rot90(sprite, RANK_TO_ROT_K[rank])
        q_r, q_c = QUADRANT_ANCHORS[marker_pos]
        paste(out, rotated, q_r, q_c, transparent=BG)

    return out


def _build_mask_catalog() -> dict[int, list[np.ndarray]]:
    masks_by_area: dict[int, list[np.ndarray]] = {area: [] for area in range(3, 9)}
    for bits in range(1, 1 << (SPRITE_SIZE * SPRITE_SIZE)):
        mask = np.array([(bits >> idx) & 1 for idx in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=bool).reshape(
            SPRITE_SIZE, SPRITE_SIZE
        )
        area = int(mask.sum())
        if area not in masks_by_area:
            continue
        coords = np.argwhere(mask)
        box = bbox_coords(coords)
        if bbox_h(box) != SPRITE_SIZE or bbox_w(box) != SPRITE_SIZE:
            continue
        if len(components_mask(mask, connectivity=8)) != 1:
            continue
        masks_by_area[area].append(mask)
    assert all(masks_by_area.values())
    return masks_by_area


MASKS_BY_AREA = _build_mask_catalog()


def _rotation_signature(mask: np.ndarray, k: int) -> bytes:
    return np.rot90(mask.astype(np.uint8), k).tobytes()


def _is_rotation_visible(mask: np.ndarray, k: int) -> bool:
    return _rotation_signature(mask, 0) != _rotation_signature(mask, k)


def _choose_mask(area: int, rank: int) -> np.ndarray:
    candidates = MASKS_BY_AREA[int(area)]
    if rank in (2, 4):
        visible = [mask for mask in candidates if _is_rotation_visible(mask, -1)]
        assert visible
        return random.choice(visible).copy()
    if rank == 3:
        visible = [mask for mask in candidates if _is_rotation_visible(mask, 2)]
        assert visible
        return random.choice(visible).copy()
    return random.choice(candidates).copy()


def _mask_to_sprite(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    sprite[mask] = int(color)
    return sprite


def _box_with_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    g = int(gap)
    return ar1 + g < br0 or br1 + g < ar0 or ac1 + g < bc0 or bc1 + g < ac0


def _box_at(r: int, c: int, size: int = SPRITE_SIZE) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r) + size - 1, int(c) + size - 1


def _selector_box(rows: int, cols: int) -> tuple[int, int, int, int]:
    max_r = rows - SELECTOR_SIZE
    max_c = cols - SELECTOR_SIZE
    corner = random.choice(((0, 0), (0, max_c), (max_r, 0), (max_r, max_c)))
    r0, c0 = corner
    return int(r0), int(c0), int(r0) + SELECTOR_SIZE - 1, int(c0) + SELECTOR_SIZE - 1


def _place_sprite_boxes(rows: int, cols: int, selector_box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    candidates = []
    for r in range(rows - SPRITE_SIZE + 1):
        for c in range(cols - SPRITE_SIZE + 1):
            box = _box_at(r, c)
            if _box_with_gap(box, selector_box):
                candidates.append((int(r), int(c)))
    random.shuffle(candidates)

    need = len(MARKER_COLORS)

    def search(chosen: list[tuple[int, int]], chosen_boxes: list[tuple[int, int, int, int]], start: int) -> list[tuple[int, int]] | None:
        if len(chosen) == need:
            return chosen
        remaining = need - len(chosen)
        if len(candidates) - start < remaining:
            return None
        for idx in range(start, len(candidates)):
            r, c = candidates[idx]
            box = _box_at(r, c)
            if not all(_box_with_gap(box, prev) for prev in chosen_boxes):
                continue
            found = search(chosen + [(r, c)], chosen_boxes + [box], idx + 1)
            if found is not None:
                return found
        return None

    return search([], [], 0)


def _make_selector(marker_assignment: dict[tuple[int, int], int]) -> np.ndarray:
    selector = np.full((SELECTOR_SIZE, SELECTOR_SIZE), SELECTOR_COLOR, dtype=int)
    for pos, color in marker_assignment.items():
        selector[pos] = int(color)
    return selector


def _oracle_output(marker_assignment: dict[tuple[int, int], int], sprites_by_color: dict[int, np.ndarray]) -> np.ndarray:
    areas = {color: int(np.count_nonzero(sprite != BG)) for color, sprite in sprites_by_color.items()}
    ranked = sorted(areas.items(), key=lambda item: item[1])
    rank_by_color = {color: idx + 1 for idx, (color, _area) in enumerate(ranked)}

    out = np.zeros((SELECTOR_SIZE, SELECTOR_SIZE), dtype=int)
    for pos, color in marker_assignment.items():
        rotated = np.rot90(sprites_by_color[int(color)], RANK_TO_ROT_K[rank_by_color[int(color)]])
        q_r, q_c = QUADRANT_ANCHORS[pos]
        paste(out, rotated, q_r, q_c, transparent=BG)
    return out


def _attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    selector_box = _selector_box(rows, cols)
    sprite_anchors = _place_sprite_boxes(rows, cols, selector_box)
    if sprite_anchors is None:
        return None

    chosen_areas = random.sample(range(3, 9), 4)
    sorted_areas = sorted(chosen_areas)

    area_by_color: dict[int, int] = {}
    sprites_by_color: dict[int, np.ndarray] = {}
    colors_in_rank_order = list(MARKER_COLORS)
    random.shuffle(colors_in_rank_order)
    for rank, area in enumerate(sorted_areas, start=1):
        color = int(colors_in_rank_order[rank - 1])
        area_by_color[color] = int(area)
        mask = _choose_mask(area, rank)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        mask = np.rot90(mask, random.randrange(4))
        if rank in (2, 4) and not _is_rotation_visible(mask, -1):
            return None
        if rank == 3 and not _is_rotation_visible(mask, 2):
            return None
        sprites_by_color[color] = _mask_to_sprite(mask, color)

    marker_positions = list(MARKER_POSITIONS)
    random.shuffle(marker_positions)
    marker_assignment = {pos: color for pos, color in zip(marker_positions, random.sample(MARKER_COLORS, 4))}
    selector = _make_selector(marker_assignment)
    expected = _oracle_output(marker_assignment, sprites_by_color)

    # Reject a fully symmetric summary to keep the rotation signal visible.
    if np.array_equal(expected, np.rot90(expected)):
        return None

    grid = np.zeros((rows, cols), dtype=int)
    sr0, sc0, _sr1, _sc1 = selector_box
    paste(grid, selector, sr0, sc0)

    sprite_items = list(sprites_by_color.items())
    random.shuffle(sprite_items)
    for (color, sprite), (r, c) in zip(sprite_items, sprite_anchors):
        paste(grid, sprite, r, c, transparent=BG)

    if not np.array_equal(main(grid), expected):
        return None

    counts = Counter(int(np.count_nonzero(sprite != BG)) for sprite in sprites_by_color.values())
    if sorted(counts) != sorted_areas:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
