from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BACKGROUND = 0
SPRITE_COLOR = 5
RECT_MAIN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
RECT_COUNT_OPTIONS = (3, 4)
GRID_HEIGHT_RANGE = (15, 19)
GRID_WIDTH_RANGE = (18, 24)
RECT_HEIGHT_RANGE = (3, 5)
RECT_WIDTH_RANGE = (4, 6)
WINNER_MARKER_RANGE = (1, 5)
SPRITE_BLACK_RANGE = (3, 6)
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(crop: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(crop, return_counts=True)
    max_count = int(counts.max())
    assert int(np.count_nonzero(counts == max_count)) == 1, "expected a unique dominant fill color"
    idx = int(np.argmax(counts))
    return int(colors[idx]), max_count


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rectangles: list[dict] = []
    for coords in _find_components(grid != BACKGROUND):
        r0, c0, r1, c1 = _bbox(coords)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(coords.shape[0]) != int(crop.size):
            continue
        if np.unique(crop).size < 2:
            continue
        fill_color, fill_count = _dominant_color(crop)
        marker_count = int(crop.size) - fill_count
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "crop": crop,
                "fill_color": fill_color,
                "marker_count": marker_count,
            }
        )
    return rectangles


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    sprite_coords = np.argwhere(grid == SPRITE_COLOR)
    assert sprite_coords.size > 0, "expected a gray sprite"
    r0, c0, r1, c1 = _bbox(sprite_coords)
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert sprite.shape == (3, 3), "expected the sprite bbox to be 3x3"
    assert set(map(int, np.unique(sprite))) <= {BACKGROUND, SPRITE_COLOR}
    return sprite


def _render_output(sprite: np.ndarray, copies: int) -> np.ndarray:
    black_slots = int(np.count_nonzero(sprite == BACKGROUND))
    assert black_slots > 0, "sprite must contain black pixels"
    assert copies <= black_slots * black_slots, "copy count must fit in the slot grid"

    out = np.zeros((black_slots * 3, black_slots * 3), dtype=int)
    for index in range(copies):
        block_r, block_c = divmod(index, black_slots)
        r0 = block_r * 3
        c0 = block_c * 3
        out[r0 : r0 + 3, c0 : c0 + 3] = sprite
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the sparsest marked rectangle and use its marker count as the sprite tiling amount.

    Input structure:
    - A black background contains 3-4 separated solid colored rectangles in the upper area.
    - Each rectangle is filled mostly by one dominant color plus a few non-black marker pixels of other colors.
    - A separate 3x3 gray sprite in the lower-right area contains both gray(5) and black(0) cells.

    Rule:
    - Find every solid non-black rectangle and count, inside each rectangle, the pixels that differ from that rectangle's dominant fill color.
    - Select the rectangle with the fewest such marker pixels; break ties by smaller top row, then smaller left column, and call that marker count M.
    - Take the 3x3 gray sprite, count its black pixels B, and build a black output canvas of size (3B) x (3B) interpreted as a B x B grid of 3x3 slots.
    - Copy the full sprite into the first M slots in row-major order and leave all remaining slots black.

    Key insight:
    - The rectangles are only a competition that encodes one number, and that number drives how many times the unrelated sprite is tiled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _find_rectangles(g)
    assert rectangles, "expected at least one solid rectangle"
    winner = min(rectangles, key=lambda item: (item["marker_count"], item["bbox"][0], item["bbox"][1]))

    sprite = _extract_sprite(g)
    return _render_output(sprite, int(winner["marker_count"]))


def _sample_sprite(min_black: int) -> np.ndarray:
    black_count = random.randint(max(SPRITE_BLACK_RANGE[0], min_black), SPRITE_BLACK_RANGE[1])
    gray_count = 9 - black_count
    cells = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        gray_cells = random.sample(cells, gray_count)
        rows = {r for r, _ in gray_cells}
        cols = {c for _, c in gray_cells}
        if min(rows) != 0 or max(rows) != 2 or min(cols) != 0 or max(cols) != 2:
            continue
        sprite = np.zeros((3, 3), dtype=int)
        for r, c in gray_cells:
            sprite[r, c] = SPRITE_COLOR
        return sprite
    raise RuntimeError("failed to sample a 3x3 sprite with full bbox")


def _sample_marker_counts(rect_count: int) -> list[int]:
    winner_markers = random.randint(*WINNER_MARKER_RANGE)
    if winner_markers == WINNER_MARKER_RANGE[1]:
        tie_count = rect_count
    elif random.random() < 0.6:
        tie_count = random.randint(2, rect_count)
    else:
        tie_count = 1

    counts = [winner_markers] * tie_count
    while len(counts) < rect_count:
        counts.append(random.randint(winner_markers + 1, WINNER_MARKER_RANGE[1]))
    random.shuffle(counts)
    return counts


def _make_rectangle_patch(height: int, width: int, main_color: int, marker_count: int) -> np.ndarray:
    patch = np.full((height, width), main_color, dtype=int)
    marker_cells = random.sample([(r, c) for r in range(height) for c in range(width)], marker_count)
    other_colors = [color for color in RECT_MAIN_COLORS if color != main_color]
    if marker_count >= 2 and random.random() < 0.5:
        marker_palette = random.sample(other_colors, 2)
    else:
        marker_palette = [random.choice(other_colors)]
    for index, (r, c) in enumerate(marker_cells):
        if len(marker_palette) == 2 and index < 2:
            patch[r, c] = marker_palette[index]
        else:
            patch[r, c] = random.choice(marker_palette)
    return patch


def _separated(candidate: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], padding: int = 1) -> bool:
    r0, c0, r1, c1 = candidate
    for or0, oc0, or1, oc1 in others:
        if not (r1 + padding < or0 or or1 + padding < r0 or c1 + padding < oc0 or oc1 + padding < c0):
            return False
    return True


def _place_rectangles(
    grid_shape: tuple[int, int],
    sprite_bbox: tuple[int, int, int, int],
    patches: list[np.ndarray],
) -> list[dict] | None:
    grid_h, grid_w = grid_shape
    max_rect_h = max(int(patch.shape[0]) for patch in patches)
    upper_band = min(grid_h // 2 + 1, sprite_bbox[0] - 2)
    if upper_band < max_rect_h:
        return None

    indices = sorted(range(len(patches)), key=lambda idx: int(patches[idx].size), reverse=True)
    occupied = [sprite_bbox]
    placed: list[dict] = [None] * len(patches)  # type: ignore[list-item]

    for idx in indices:
        patch = patches[idx]
        h, w = map(int, patch.shape)
        positions = [
            (r0, c0)
            for r0 in range(0, upper_band - h + 1)
            for c0 in range(0, grid_w - w + 1)
        ]
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _separated(bbox, occupied, padding=1):
                chosen = bbox
                break
        if chosen is None:
            return None
        occupied.append(chosen)
        placed[idx] = {"bbox": chosen, "patch": patch}
    return placed


def _manual_expected(sprite: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    winner = min(rectangles, key=lambda item: (item["marker_count"], item["bbox"][0], item["bbox"][1]))
    return _render_output(sprite, int(winner["marker_count"]))


def _generate_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_HEIGHT_RANGE)
    grid_w = random.randint(*GRID_WIDTH_RANGE)
    rect_count = random.choice(RECT_COUNT_OPTIONS)
    marker_counts = _sample_marker_counts(rect_count)
    min_black = math.ceil(math.sqrt(min(marker_counts)))
    sprite = _sample_sprite(min_black)

    sprite_r0 = random.choice([grid_h - 4, grid_h - 3])
    sprite_c0 = random.choice([grid_w - 4, grid_w - 3])
    sprite_bbox = (sprite_r0, sprite_c0, sprite_r0 + 2, sprite_c0 + 2)

    main_colors = random.sample(list(RECT_MAIN_COLORS), rect_count)
    patches: list[np.ndarray] = []
    for main_color, marker_count in zip(main_colors, marker_counts):
        height = random.randint(*RECT_HEIGHT_RANGE)
        width = random.randint(*RECT_WIDTH_RANGE)
        patch = _make_rectangle_patch(height, width, main_color, marker_count)
        patches.append(patch)

    placements = _place_rectangles((grid_h, grid_w), sprite_bbox, patches)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    grid[sprite_r0 : sprite_r0 + 3, sprite_c0 : sprite_c0 + 3] = sprite

    rectangle_meta: list[dict] = []
    for placement in placements:
        r0, c0, r1, c1 = placement["bbox"]
        patch = placement["patch"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = patch
        fill_color, fill_count = _dominant_color(patch)
        rectangle_meta.append(
            {
                "bbox": placement["bbox"],
                "marker_count": int(patch.size) - fill_count,
                "fill_color": fill_color,
            }
        )

    expected = _manual_expected(sprite, rectangle_meta)
    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None

    rectangles = _find_rectangles(grid)
    if len(rectangles) != rect_count:
        return None

    gray_coords = np.argwhere(grid == SPRITE_COLOR)
    if gray_coords.size == 0 or _bbox(gray_coords) != sprite_bbox:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
