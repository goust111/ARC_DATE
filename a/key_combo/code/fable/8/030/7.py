from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


AXES = ("lr", "ud")
GRID_SIDE_RANGE = (18, 30)
SPRITE_COUNT_RANGE = (4, 7)
PATCH_SIDE_RANGE = (3, 7)
PATCH_DENSITY_RANGE = (0.42, 0.72)
PATCH_COLORS_RANGE = (2, 4)
PLACEMENT_PADDING = 1
GENERATION_RETRIES = 600
FOUR_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _mirror_patch(patch: np.ndarray, axis: str) -> np.ndarray:
    if axis == "lr":
        return patch[:, ::-1].copy()
    if axis == "ud":
        return patch[::-1, :].copy()
    raise ValueError(f"unknown axis: {axis}")


def _patch_key(patch: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, patch.shape)), tuple(int(x) for x in patch.ravel())


def _matching_axes(a: np.ndarray, b: np.ndarray) -> list[str]:
    if a.shape != b.shape:
        return []
    matches = []
    for axis in AXES:
        if np.array_equal(_mirror_patch(a, axis), b):
            matches.append(axis)
    return matches


def _extract_sprites(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = int(get_bg(grid))
    sprites = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    ):
        bbox = tuple(map(int, comp["bbox"]))
        sprites.append(
            {
                "bbox": bbox,
                "patch": crop_bbox(grid, bbox),
            }
        )
    sprites.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))
    return bg, sprites


def _find_demo_pair(sprites: list[dict[str, object]]) -> tuple[int, int, str]:
    matches: list[tuple[int, int, str]] = []
    for i in range(len(sprites)):
        for j in range(i + 1, len(sprites)):
            axes = _matching_axes(sprites[i]["patch"], sprites[j]["patch"])
            for axis in axes:
                matches.append((i, j, axis))
    assert len(matches) == 1, "expected exactly one mirrored sprite pair"
    return matches[0]


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Find the only mirrored sprite pair, infer its mirror axis, and mirror every other sprite in place.

    Input structure:
    - The most common color is the background.
    - The non-background cells form several separated 4-connected multicolor sprites.
    - Exactly one pair of sprites has tight colored crops that are mutual left-right
      or up-down mirrors, and no sprite is symmetric under those mirrors.

    Rule:
    - Crop every sprite to its tight bounding box, preserving background holes inside the crop.
    - Identify the unique pair whose full crops are mirror matches and read whether
      they demonstrate a left-right or up-down mirror.
    - Keep that demonstration pair unchanged, and apply the same mirror inside the
      original bounding box of every other sprite.

    Key insight:
    - The repeated-looking pair is not the answer object; it is an in-scene instruction
      showing which mirror transform must be applied to all remaining sprites.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    _bg, sprites = _extract_sprites(grid)
    assert len(sprites) >= 3, "expected at least one non-demo sprite"
    demo_i, demo_j, axis = _find_demo_pair(sprites)

    out = grid.copy()
    for idx, sprite in enumerate(sprites):
        if idx in (demo_i, demo_j):
            continue
        r0, c0, r1, c1 = sprite["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _mirror_patch(sprite["patch"], axis)
    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_connected_mask() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*PATCH_SIDE_RANGE)
        w = random.randint(*PATCH_SIDE_RANGE)
        target = random.randint(
            max(5, int(np.ceil(h * w * PATCH_DENSITY_RANGE[0]))),
            max(5, int(np.floor(h * w * PATCH_DENSITY_RANGE[1]))),
        )
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        cells = {start}

        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            cells.add(cell)
            mask[cell] = True

        cropped = _crop_mask(mask)
        if min(cropped.shape) < 3:
            continue
        if int(cropped.sum()) < 5:
            continue
        return cropped
    raise RuntimeError("failed to build a connected mask")


def _random_patch(bg: int) -> np.ndarray:
    palette = [c for c in range(10) if c != bg]
    for _ in range(300):
        mask = _random_connected_mask()
        coords = [tuple(rc) for rc in np.argwhere(mask)]
        color_count = random.randint(PATCH_COLORS_RANGE[0], min(PATCH_COLORS_RANGE[1], len(coords), len(palette)))
        colors = random.sample(palette, color_count)
        assignments = colors[:] + [random.choice(colors) for _ in range(len(coords) - color_count)]
        random.shuffle(assignments)

        patch = np.full(mask.shape, bg, dtype=int)
        for (r, c), color in zip(coords, assignments):
            patch[r, c] = int(color)

        lr = _mirror_patch(patch, "lr")
        ud = _mirror_patch(patch, "ud")
        if np.array_equal(patch, lr) or np.array_equal(patch, ud):
            continue
        if np.array_equal(lr, ud):
            continue
        return patch
    raise RuntimeError("failed to build a non-symmetric patch")


def _is_compatible_candidate(candidate: np.ndarray, accepted: list[np.ndarray]) -> bool:
    cand_key = _patch_key(candidate)
    cand_lr = _mirror_patch(candidate, "lr")
    cand_ud = _mirror_patch(candidate, "ud")
    mirror_keys = {_patch_key(cand_lr), _patch_key(cand_ud)}

    for other in accepted:
        other_key = _patch_key(other)
        if cand_key == other_key:
            return False
        if other_key in mirror_keys:
            return False
        if _patch_key(_mirror_patch(other, "lr")) == cand_key:
            return False
        if _patch_key(_mirror_patch(other, "ud")) == cand_key:
            return False
    return True


def _place_patches(shape: tuple[int, int], bg: int, patches: list[np.ndarray]) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full(shape, bg, dtype=int)
    bbox_canvas = np.zeros(shape, dtype=int)
    placed: list[dict[str, object]] = []

    ordered = sorted(
        enumerate(patches),
        key=lambda item: item[1].shape[0] * item[1].shape[1],
        reverse=True,
    )
    for original_idx, patch in ordered:
        block = np.ones(patch.shape, dtype=int)
        try:
            r0, c0 = random_free_location(
                bbox_canvas,
                block,
                transparent=0,
                padding=PLACEMENT_PADDING,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(bbox_canvas, block, r0, c0, transparent=0)
        paste(grid, patch, r0, c0, transparent=bg)
        h, w = map(int, patch.shape)
        placed.append(
            {
                "source_index": original_idx,
                "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                "patch": patch,
            }
        )
    placed.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))
    return grid, placed


def _expected_from_metadata(grid: np.ndarray, placed: list[dict[str, object]], demo_sources: set[int], axis: str) -> np.ndarray:
    out = grid.copy()
    for item in placed:
        if int(item["source_index"]) in demo_sources:
            continue
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _mirror_patch(item["patch"], axis)
    return out


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], tuple[int, int, str]]:
    _bg, sprites = _extract_sprites(grid)
    pair = _find_demo_pair(sprites)
    return sprites, pair


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    axis = random.choice(AXES)
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)

    demo_base = _random_patch(bg)
    demo_other = _mirror_patch(demo_base, axis)
    accepted = [demo_base, demo_other]
    while len(accepted) < sprite_count:
        candidate = _random_patch(bg)
        if _is_compatible_candidate(candidate, accepted):
            accepted.append(candidate)

    random.shuffle(accepted)
    shape = (
        random.randint(*GRID_SIDE_RANGE),
        random.randint(*GRID_SIDE_RANGE),
    )
    placed_scene = _place_patches(shape, bg, accepted)
    if placed_scene is None:
        return None
    grid, placed = placed_scene

    demo_sources = {idx for idx, patch in enumerate(accepted) if _patch_key(patch) in {_patch_key(demo_base), _patch_key(demo_other)}}
    if len(demo_sources) != 2:
        return None

    expected = _expected_from_metadata(grid, placed, demo_sources, axis)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    sprites, (i, j, found_axis) = _analyze_scene(grid)
    if found_axis != axis:
        return None

    demo_bboxes = {tuple(int(x) for x in sprites[i]["bbox"]), tuple(int(x) for x in sprites[j]["bbox"])}
    changed = 0
    for sprite in sprites:
        bbox = tuple(int(x) for x in sprite["bbox"])
        if bbox in demo_bboxes:
            if not np.array_equal(crop_bbox(actual, bbox), sprite["patch"]):
                return None
        else:
            mirrored = _mirror_patch(sprite["patch"], axis)
            if np.array_equal(mirrored, sprite["patch"]):
                return None
            if not np.array_equal(crop_bbox(actual, bbox), mirrored):
                return None
            changed += 1
    if changed == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
