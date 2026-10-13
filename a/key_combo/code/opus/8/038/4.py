from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, components_mask, crop_bbox, find_components, get_bg


BG = 0
RED = 2
TEAL = 8
MAROON = 9
RECT_COLORS = (1, 3, 4, 5, 6, 7)
GRID_RANGE = (18, 26)
RECT_COUNT_CHOICES = (3, 4)
SPRITE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400


def _border_values(patch: np.ndarray) -> np.ndarray:
    return np.concatenate([patch[0], patch[-1], patch[1:-1, 0], patch[1:-1, -1]])


def _rectangle_components(grid: np.ndarray, bg: int) -> list[dict]:
    rectangles: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        box = comp["bbox"]
        h = bbox_h(box)
        w = bbox_w(box)
        if h < 5 or w < 5 or int(comp["area"]) != h * w:
            continue

        patch = crop_bbox(grid, box)
        border = _border_values(patch)
        border_color = int(border[0])
        if border_color in (bg, RED):
            continue
        if not np.all(border == border_color):
            continue

        interior = patch[1:-1, 1:-1]
        if interior.size == 0:
            continue
        if not np.isin(interior, [bg, RED, border_color]).all():
            continue

        rectangles.append(
            {
                "bbox": box,
                "border_color": border_color,
                "interior": interior.copy(),
                "marker_count": int(np.count_nonzero(interior == RED)),
            }
        )
    return rectangles


def _sprite_components(grid: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    teal_components = find_components(
        grid,
        bg=bg,
        colors=[TEAL],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    for comp in teal_components:
        area = int(comp["area"])
        if not 3 <= area <= 5:
            continue
        box = comp["bbox"]
        patch = crop_bbox(grid, box)
        mask = patch == TEAL
        if int(mask.sum()) != area:
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        sprites.append({"bbox": box, "mask": mask.copy(), "area": area})

    assert sprites, "expected teal sprite components"
    ref_mask = sprites[0]["mask"]
    for sprite in sprites[1:]:
        assert sprite["mask"].shape == ref_mask.shape and np.array_equal(sprite["mask"], ref_mask), (
            "all teal sprites must share one mask"
        )
    return sprites


def _template_from_rectangle(rect: dict) -> np.ndarray:
    interior = rect["interior"]
    out = interior.copy()
    out[out == rect["border_color"]] = MAROON
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the interior of the most-marked rectangle as a stamp template and replace every teal sprite with it.

    Input structure:
    - A black background contains several separated filled rectangles of distinct colors.
    - Each rectangle has a one-cell border in its own color, and its interior may contain red marker cells.
    - The grid also contains several identical 4-connected teal sprites whose bounding boxes match one rectangle interior.

    Rule:
    - For each rectangle, count the red cells only inside its one-pixel-inset interior.
    - Select the unique rectangle with the largest interior red-marker count.
    - Extract that rectangle's interior as a template, recoloring its original rectangle color to maroon while keeping red and black unchanged.
    - Replace the full bounding box of every teal sprite with that template, aligned by top-left corner.
    - Leave all rectangles, including the source rectangle, unchanged.

    Key insight:
    - The red markers do not indicate where to paint; they choose which framed rectangle contributes the reusable interior pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    assert bg == BG, "expected a black background"

    rectangles = _rectangle_components(g, bg)
    sprites = _sprite_components(g, bg)
    assert len(rectangles) >= 3, "expected multiple framed rectangles"
    assert 2 <= len(sprites) <= 4, "expected 2-4 teal sprites"

    best_count = max(rect["marker_count"] for rect in rectangles)
    winners = [rect for rect in rectangles if rect["marker_count"] == best_count]
    assert len(winners) == 1, "expected a unique most-marked rectangle"
    winner = winners[0]

    template = _template_from_rectangle(winner)
    template_shape = template.shape
    assert all(crop_bbox(g, sprite["bbox"]).shape == template_shape for sprite in sprites), (
        "sprite bounding boxes must match the chosen rectangle interior"
    )

    out = g.copy()
    h, w = map(int, template.shape)
    for sprite in sprites:
        r0, c0, _r1, _c1 = map(int, sprite["bbox"])
        out[r0 : r0 + h, c0 : c0 + w] = template
    return out


def _enumerate_sprite_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << 9):
        area = bits.bit_count()
        if not 3 <= area <= 5:
            continue
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        coords = np.argwhere(mask)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if (int(r0), int(c0), int(r1), int(c1)) != (0, 0, 2, 2):
            continue
        if len(components_mask(mask, connectivity=4)) != 1:
            continue
        masks.append(mask)
    return masks


SPRITE_MASKS = _enumerate_sprite_masks()


def _place_boxes(grid_shape: tuple[int, int], sizes: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    h_grid, w_grid = map(int, grid_shape)
    blocked = np.zeros((h_grid, w_grid), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    for idx in order:
        h, w = sizes[idx]
        choices = []
        for r0 in range(h_grid - h + 1):
            for c0 in range(w_grid - w + 1):
                if blocked[r0 : r0 + h, c0 : c0 + w].any():
                    continue
                choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placements[idx] = (r0, c0)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h_grid, r0 + h + 1)
        cc1 = min(w_grid, c0 + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    return [pos for pos in placements if pos is not None]


def _render_rectangle(
    grid: np.ndarray,
    top_left: tuple[int, int],
    size: tuple[int, int],
    color: int,
    marker_count: int,
) -> np.ndarray:
    r0, c0 = top_left
    h, w = size
    grid[r0 : r0 + h, c0 : c0 + w] = color
    interior = grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1]
    if marker_count:
        cells = [(r, c) for r in range(interior.shape[0]) for c in range(interior.shape[1])]
        for rr, cc in random.sample(cells, marker_count):
            interior[rr, cc] = RED
    return interior.copy()


def _render_sprite(grid: np.ndarray, top_left: tuple[int, int], mask: np.ndarray) -> None:
    r0, c0 = top_left
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    patch[mask] = TEAL


def _build_expected(grid: np.ndarray, sprite_positions: list[tuple[int, int]], template: np.ndarray) -> np.ndarray:
    out = grid.copy()
    h, w = map(int, template.shape)
    for r0, c0 in sprite_positions:
        out[r0 : r0 + h, c0 : c0 + w] = template
    return out


def _marker_counts(n_rectangles: int, winner_count: int) -> list[int]:
    losers = random.sample(list(range(winner_count)), n_rectangles - 1)
    counts = losers + [winner_count]
    random.shuffle(counts)
    return counts


def _rectangle_sizes(n_rectangles: int) -> list[tuple[int, int]]:
    sizes = [(5, 5), (5, 5)]
    larger_choices = [(5, 6), (6, 5), (6, 6), (5, 7), (7, 5), (6, 7), (7, 6), (7, 7)]
    while len(sizes) < n_rectangles:
        if random.random() < 0.45:
            sizes.append((5, 5))
        else:
            sizes.append(random.choice(larger_choices))
    random.shuffle(sizes)
    return sizes


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_RANGE)
    cols = random.randint(*GRID_RANGE)
    n_rectangles = random.choice(RECT_COUNT_CHOICES)
    n_sprites = random.randint(*SPRITE_COUNT_RANGE)

    rect_sizes = _rectangle_sizes(n_rectangles)
    winner_candidates = [idx for idx, size in enumerate(rect_sizes) if size == (5, 5)]
    if len(winner_candidates) < 2:
        return None
    winner_index = random.choice(winner_candidates)

    winner_count = random.randint(max(3, n_rectangles - 1), 5)
    counts = _marker_counts(n_rectangles, winner_count)
    counts[winner_index] = winner_count
    remaining = [idx for idx in range(n_rectangles) if idx != winner_index]
    loser_values = random.sample(list(range(winner_count)), len(remaining))
    random.shuffle(loser_values)
    for idx, value in zip(remaining, loser_values):
        counts[idx] = value

    colors = random.sample(list(RECT_COLORS), n_rectangles)
    sprite_mask = random.choice(SPRITE_MASKS)

    object_sizes = rect_sizes + [(3, 3)] * n_sprites
    placements = _place_boxes((rows, cols), object_sizes)
    if placements is None:
        return None

    rect_positions = placements[:n_rectangles]
    sprite_positions = placements[n_rectangles:]

    grid = np.full((rows, cols), BG, dtype=int)
    interiors: list[np.ndarray] = []
    for pos, size, color, marker_count in zip(rect_positions, rect_sizes, colors, counts):
        interiors.append(_render_rectangle(grid, pos, size, color, marker_count))
    for pos in sprite_positions:
        _render_sprite(grid, pos, sprite_mask)

    winner_template = interiors[winner_index].copy()
    winner_template[winner_template == colors[winner_index]] = MAROON
    expected = _build_expected(grid, sprite_positions, winner_template)

    if winner_template.shape != (3, 3):
        return None
    if not any(size == rect_sizes[winner_index] for i, size in enumerate(rect_sizes) if i != winner_index):
        return None
    if np.array_equal(grid, expected):
        return None
    if len(np.unique(counts)) != len(counts):
        return None
    if np.count_nonzero(winner_template == RED) != winner_count:
        return None
    if np.count_nonzero(winner_template == MAROON) == 0:
        return None
    if np.count_nonzero(grid == TEAL) < 3 * n_sprites:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
