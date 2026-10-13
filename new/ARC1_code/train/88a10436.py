from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, bbox_h, bbox_w, components_mask, get_bg, is_contiguous, paste
import itertools
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the single colored sprite so its bounding-box center lands on the gray marker.

    Input structure:
    - A background grid containing one non-gray foreground sprite.
    - Exactly one gray marker pixel sits in an otherwise empty region.
    - The sprite's bounding box has odd height and odd width, so it has a center cell.

    Rule:
    - Keep the original sprite unchanged.
    - Extract the full bounding box of the non-gray sprite, treating background cells inside the box as transparent.
    - Paste that sprite box centered on the gray marker, replacing the marker.

    Key insight:
    The gray pixel is not a color to preserve; it marks the center of a second copy of the existing sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg != Color.GREY

    marker_coords = np.argwhere(g == Color.GREY)
    assert len(marker_coords) == 1
    marker_r, marker_c = map(int, marker_coords[0])

    sprite_mask = (g != bg) & (g != Color.GREY)
    sprite_components = components_mask(sprite_mask, connectivity=4)
    assert len(sprite_components) == 1

    sprite_bbox = bbox(sprite_mask)
    assert sprite_bbox is not None
    r0, c0, r1, c1 = sprite_bbox
    h = bbox_h(sprite_bbox)
    w = bbox_w(sprite_bbox)
    assert h % 2 == 1 and w % 2 == 1

    patch = g[r0 : r1 + 1, c0 : c1 + 1]
    assert int(patch[h // 2, w // 2]) != bg

    paste_r = marker_r - h // 2
    paste_c = marker_c - w // 2
    assert 0 <= paste_r and 0 <= paste_c
    assert paste_r + h <= g.shape[0] and paste_c + w <= g.shape[1]

    target = g[paste_r : paste_r + h, paste_c : paste_c + w]
    assert bool(np.all((target == bg) | (target == Color.GREY)))

    out = g.copy()
    out[marker_r, marker_c] = bg
    paste(out, patch, paste_r, paste_c, transparent=bg)
    return out

# --------------- generator (o5x) ---------------

# Canvas, background, and retry knobs.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (8, 10), (7, 11)
BACKGROUND, GEN_ATTEMPTS = 0, 300

# Sprite masks/colors: 3x3, connected, edge-touching, asymmetric, 6-7 cells full.
SPRITE_SIZE = 3
SPRITE_CENTER = SPRITE_SIZE // 2
SPRITE_FILLED_COUNTS = (6, 7)
SPRITE_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SPRITE_COLOR_COUNT = 3

def _valid_sprite_masks88a10436() -> list[np.ndarray]:
    masks = []
    cells = [(r, c) for r in range(SPRITE_SIZE) for c in range(SPRITE_SIZE) if (r, c) != (SPRITE_CENTER, SPRITE_CENTER)]
    for filled_count in SPRITE_FILLED_COUNTS:
        for holes in itertools.combinations(cells, SPRITE_SIZE * SPRITE_SIZE - filled_count):
            mask = np.ones((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
            for r, c in holes:
                mask[r, c] = False
            touches_edges = mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
            symmetric = np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask))
            if touches_edges and not symmetric and is_contiguous(
                mask.astype(int), background=BACKGROUND, connectivity=4
            ):
                masks.append(mask)
    return masks


SPRITE_MASKS = _valid_sprite_masks88a10436()


def _gen_sprite88a10436() -> np.ndarray:
    mask = random.choice(SPRITE_MASKS)
    palette = random.sample(SPRITE_PALETTE, SPRITE_COLOR_COUNT)
    coords = list(map(tuple, np.argwhere(mask)))
    random.shuffle(coords)

    sprite = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=int)
    for (r, c), color in zip(coords[:SPRITE_COLOR_COUNT], palette):
        sprite[r, c] = color
    for r, c in coords[SPRITE_COLOR_COUNT:]:
        sprite[r, c] = random.choice(palette)
    return sprite


def _gen_positions88a10436(height: int, width: int, sh: int, sw: int) -> tuple[int, int, int, int] | None:
    positions = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
    pairs = []
    for r1, c1 in positions:
        for r2, c2 in positions:
            row_separated = r1 + sh <= r2 or r2 + sh <= r1
            col_separated = c1 + sw <= c2 or c2 + sw <= c1
            if row_separated or col_separated:
                pairs.append((r1, c1, r2, c2))
    return random.choice(pairs) if pairs else None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    sprite = _gen_sprite88a10436()
    sh, sw = sprite.shape

    positions = _gen_positions88a10436(height, width, sh, sw)
    if positions is None:
        return None
    r1, c1, r2, c2 = positions

    out = np.zeros((height, width), dtype=int)
    paste(out, sprite, r1, c1, transparent=BACKGROUND)
    paste(out, sprite, r2, c2, transparent=BACKGROUND)

    inp = out.copy()
    patch_mask = sprite != BACKGROUND
    inp[r2 : r2 + sh, c2 : c2 + sw][patch_mask] = BACKGROUND
    inp[r2 + sh // 2, c2 + sw // 2] = Color.GREY

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
