from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste


BG = Color.BLACK
FRAME_COLOR = Color.GREY
SPRITE_INPUT_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.ORANGE]
DEPTH_TO_COLOR = {
    1: Color.BLUE,
    2: Color.RED,
    3: Color.GREEN,
}

GEN_RETRIES = 300
MANDATORY_DEPTHS = (1, 2, 3)
EXTRA_DEPTH_CHOICES = (1, 1, 2)
EXTRA_COUNT_CHOICES = (0, 1, 1, 2)
SLACK_CHOICES = (0, 0, 1)

SPRITE_TEMPLATES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _random_sprite_mask(rng: random.Random) -> np.ndarray:
    mask = SPRITE_TEMPLATES[rng.randrange(len(SPRITE_TEMPLATES))].copy()
    mask = np.rot90(mask, k=rng.randrange(4))
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    mask = _trim_mask(mask.astype(bool))
    area = int(mask.sum())
    assert 2 <= area <= 5
    comps = components_mask(mask, connectivity=4)
    assert len(comps) == 1
    return mask


def _frame_patch(height: int, width: int) -> np.ndarray:
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR
    return patch


def _build_chain(depth: int, sprite_color: int, rng: random.Random) -> np.ndarray:
    sprite_mask = _random_sprite_mask(rng)
    sprite_h, sprite_w = map(int, sprite_mask.shape)

    inner_h = sprite_h + 4 + rng.choice(SLACK_CHOICES)
    inner_w = sprite_w + 4 + rng.choice(SLACK_CHOICES)
    patch = _frame_patch(inner_h, inner_w)

    sprite_r = rng.randint(2, inner_h - sprite_h - 2)
    sprite_c = rng.randint(2, inner_w - sprite_w - 2)
    sprite = np.full(sprite_mask.shape, BG, dtype=int)
    sprite[sprite_mask] = int(sprite_color)
    paste(patch, sprite, sprite_r, sprite_c, transparent=BG)

    for _ in range(depth - 1):
        child = patch
        child_h, child_w = map(int, child.shape)
        outer_h = child_h + 4 + rng.choice(SLACK_CHOICES)
        outer_w = child_w + 4 + rng.choice(SLACK_CHOICES)
        patch = _frame_patch(outer_h, outer_w)
        child_r = rng.randint(2, outer_h - child_h - 2)
        child_c = rng.randint(2, outer_w - child_w - 2)
        paste(patch, child, child_r, child_c, transparent=BG)

    return patch


def _frame_bbox(component: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox_coords(component)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    local = np.zeros((height, width), dtype=bool)
    local[component[:, 0] - r0, component[:, 1] - c0] = True

    border = np.zeros_like(local)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(local, border):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _sprite_depth(component: np.ndarray, frames: list[tuple[int, int, int, int]]) -> int:
    r0, c0, r1, c1 = bbox_coords(component)
    return sum(fr0 < r0 and r1 < fr1 and fc0 < c0 and c1 < fc1 for fr0, fc0, fr1, fc1 in frames)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each enclosed sprite according to how many nested grey frames surround it.

    Input structure:
    - Black background with several grey hollow rectangular frames, sometimes nested into containment chains.
    - Each innermost cavity contains exactly one small 4-connected monochrome sprite in a non-black, non-grey color.
    - Sibling frame chains may appear side by side, and frame geometry is always axis-aligned with one-pixel-thick borders.

    Rule:
    - Detect every grey connected component that forms a hollow rectangular frame.
    - For each non-grey sprite, count how many frame interiors strictly contain it; that count is its nesting depth.
    - Recolor the whole sprite to blue at depth 1, red at depth 2, and green at depth 3; keep the frames and background unchanged.

    Key insight:
    - The sprite's original color is irrelevant noise: only the number of enclosing rectangular frame layers determines its output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    frame_components = components_mask(g == FRAME_COLOR, connectivity=4)
    frames = []
    for component in frame_components:
        frame = _frame_bbox(component)
        if frame is not None:
            frames.append(frame)
    assert frames, "expected at least one grey rectangular frame"

    sprite_mask = (g != BG) & (g != FRAME_COLOR)
    sprite_components = components_mask(sprite_mask, connectivity=4)
    assert sprite_components, "expected at least one sprite"

    out = g.copy()
    for component in sprite_components:
        depth = _sprite_depth(component, frames)
        assert depth in DEPTH_TO_COLOR, f"unsupported nesting depth: {depth}"
        out[component[:, 0], component[:, 1]] = DEPTH_TO_COLOR[depth]
    return out


def _depths_present(grid: np.ndarray) -> set[int]:
    frame_components = components_mask(grid == FRAME_COLOR, connectivity=4)
    frames = []
    for component in frame_components:
        frame = _frame_bbox(component)
        if frame is not None:
            frames.append(frame)
    sprite_components = components_mask((grid != BG) & (grid != FRAME_COLOR), connectivity=4)
    return {_sprite_depth(component, frames) for component in sprite_components}


def _place_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    reserved_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)
    base = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(reserved_area * 1.55))))
    if base > 29:
        return None

    height = min(30, max(max_h + 2, base + rng.randint(-1, 2)))
    width = min(30, max(max_w + 2, base + rng.randint(-1, 2)))
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    order = sorted(
        patches,
        key=lambda patch: (int(patch.shape[0] * patch.shape[1]), int(patch.shape[0]), int(patch.shape[1])),
        reverse=True,
    )

    for patch in order:
        patch_h, patch_w = map(int, patch.shape)
        positions = [(r, c) for r in range(height - patch_h + 1) for c in range(width - patch_w + 1)]
        rng.shuffle(positions)

        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + patch_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + patch_w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            paste(grid, patch, r0, c0, transparent=BG)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    patches = []
    for depth in MANDATORY_DEPTHS:
        target_color = DEPTH_TO_COLOR[depth]
        input_choices = [color for color in SPRITE_INPUT_COLORS if color != target_color]
        sprite_color = rng.choice(input_choices)
        patches.append(_build_chain(depth, sprite_color, rng))

    for _ in range(rng.choice(EXTRA_COUNT_CHOICES)):
        depth = rng.choice(EXTRA_DEPTH_CHOICES)
        sprite_color = rng.choice(SPRITE_INPUT_COLORS)
        patches.append(_build_chain(depth, sprite_color, rng))

    grid = _place_patches(patches, rng)
    if grid is None:
        return None

    depths = _depths_present(grid)
    if depths != {1, 2, 3} and not {1, 2, 3}.issubset(depths):
        return None

    transformed = main(grid)
    for depth, target_color in DEPTH_TO_COLOR.items():
        if target_color not in transformed:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid nested-frame sample")
