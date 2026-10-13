from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste, random_free_location


BG = Color.BLACK
SPRITE_COLOR = Color.GREEN
SEGMENT_COLORS = [Color.BLUE, Color.RED, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL]
SPRITE_DIMS = [(3, 2), (3, 3), (4, 2), (4, 3)]
GRID_SIDE_RANGE = (13, 17)
SEGMENT_LENGTHS = [2, 3, 4, 5, 6]
GEN_ATTEMPTS = 500


def _segment_info(component: dict) -> dict:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(component["area"])
    assert h == 1 or w == 1, "segment must be horizontal or vertical"
    length = max(h, w)
    assert area == length, "segment must be a solid straight line"
    return {
        "color": color,
        "length": length,
        "bbox": (r0, c0, r1, c1),
        "orientation": "v" if h > 1 else "h",
    }


def _extract_parts(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG, "background must be black"

    full_components = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert len(full_components) == 5, "expected one sprite and four separated segments"

    green_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[SPRITE_COLOR],
        return_format="dict",
    )
    assert len(green_components) == 1, "expected exactly one green sprite"
    sprite_component = green_components[0]
    sprite = crop_bbox(g, sprite_component["bbox"])

    sprite_mask = sprite == SPRITE_COLOR
    assert sprite_mask.sum() == int(sprite_component["area"])
    assert 5 <= int(sprite_mask.sum()) <= 8, "sprite area must stay in range"
    assert np.all(sprite[~sprite_mask] == bg), "sprite bbox may only contain bg holes"

    mono_components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    segments: list[dict] = []
    for component in mono_components:
        color = int(component["color"])
        if color == SPRITE_COLOR:
            continue
        assert color in SEGMENT_COLORS, "unexpected segment color"
        segments.append(_segment_info(component))

    assert len(segments) == 4, "expected exactly four line segments"
    lengths = [segment["length"] for segment in segments]
    colors = [segment["color"] for segment in segments]
    assert len(set(lengths)) == 4, "segment lengths must all differ"
    assert len(set(colors)) == 4, "segment colors must all differ"
    return sprite, sorted(segments, key=lambda item: item["length"], reverse=True)


def _recolor_sprite(sprite: np.ndarray, color: int) -> np.ndarray:
    patch = sprite.copy()
    patch[patch == SPRITE_COLOR] = int(color)
    return patch


def _ring_layout(sprite: np.ndarray, ring_colors: list[int]) -> np.ndarray:
    assert len(ring_colors) == 4
    h, w = map(int, sprite.shape)
    out = np.full((h + 2 * w, h + 2 * w), BG, dtype=int)

    top = _recolor_sprite(np.rot90(sprite, -1), ring_colors[0])
    right = _recolor_sprite(np.fliplr(sprite), ring_colors[1])
    bottom = _recolor_sprite(np.rot90(sprite, 1), ring_colors[2])
    left = _recolor_sprite(sprite, ring_colors[3])

    paste(out, top, 0, w, transparent=BG)
    paste(out, left, w, 0, transparent=BG)
    paste(out, right, w, w + h, transparent=BG)
    paste(out, bottom, w + h, w, transparent=BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the green sprite, build a four-way ring, and recolor each side by descending segment length.

    Input structure:
    - Black is the background.
    - One irregular green 4-connected sprite occupies a small bbox and may contain background holes inside that bbox.
    - Four other foreground components are monochromatic straight line segments, each with a distinct color and a distinct length.

    Rule:
    - Crop the green sprite to its minimum bbox.
    - Sort the four line segments by length from longest to shortest.
    - Create the 2697da3f four-way ring layout: clockwise rotation on top, original on the left,
      horizontal mirror on the right, and counterclockwise rotation on the bottom, around a blank center.
    - Recolor the top, right, bottom, and left copies respectively with the sorted segment colors.

    Key insight:
    - The line segments do not contribute shape or position; only their relative lengths decide which color is assigned to each ring direction.
    """
    sprite, segments = _extract_parts(grid)
    ordered_colors = [segment["color"] for segment in segments]
    return _ring_layout(sprite, ordered_colors)


def _random_connected_mask(h: int, w: int, target_area: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    mask[start] = True

    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return np.zeros((h, w), dtype=bool)
        choice = random.choice(tuple(frontier))
        cells.add(choice)
        mask[choice] = True
    return mask


def _good_sprite_mask(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.shape[0] < 5 or coords.shape[0] > 8:
        return False
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if (int(r1 - r0 + 1), int(c1 - c0 + 1)) != mask.shape:
        return False

    if len(find_components(mask.astype(int), bg=0, connectivity=4, monochromatic=True)) != 1:
        return False

    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if mask.shape == np.rot90(mask, 2).shape and np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if mask.shape == np.rot90(mask, 1).shape and np.array_equal(mask, np.rot90(mask, 1)):
        return False
    return True


def _make_sprite_patch() -> np.ndarray:
    for _ in range(300):
        h, w = random.choice(SPRITE_DIMS)
        target_area = random.randint(5, min(8, h * w))
        mask = _random_connected_mask(h, w, target_area)
        if not mask.any():
            continue
        if not _good_sprite_mask(mask):
            continue
        patch = np.full((h, w), BG, dtype=int)
        patch[mask] = SPRITE_COLOR
        return patch
    raise RuntimeError("failed to build a non-degenerate sprite")


def _make_segment_patch(length: int, color: int, vertical: bool) -> np.ndarray:
    if vertical:
        patch = np.full((length, 1), BG, dtype=int)
        patch[:, 0] = int(color)
    else:
        patch = np.full((1, length), BG, dtype=int)
        patch[0, :] = int(color)
    return patch


def _place_objects(canvas: np.ndarray, patches: list[np.ndarray]) -> np.ndarray | None:
    grid = canvas.copy()
    for patch in patches:
        try:
            r0, c0 = random_free_location(
                grid,
                patch,
                transparent=BG,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(grid, patch, r0, c0, transparent=BG)
    return grid


def _expected_from_parts(sprite: np.ndarray, segments: list[dict]) -> np.ndarray:
    return _ring_layout(sprite, [segment["color"] for segment in segments])


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        sprite, segments = _extract_parts(grid)
    except AssertionError:
        return False

    h, w = map(int, sprite.shape)
    if (h + 2 * w) > 20:
        return False

    rel_rows = []
    rel_cols = []
    sprite_coords = np.argwhere(grid == SPRITE_COLOR)
    sr0, sc0 = sprite_coords.min(axis=0)
    sr1, sc1 = sprite_coords.max(axis=0)
    for segment in segments:
        r0, c0, r1, c1 = segment["bbox"]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0
        rel_rows.append(center_r < sr0 or center_r > sr1)
        rel_cols.append(center_c < sc0 or center_c > sc1)
    if not any(rel_rows) or not any(rel_cols):
        return False

    expected = _expected_from_parts(sprite, segments)
    if expected.shape != (h + 2 * w, h + 2 * w):
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    quadrant_colors = []
    for patch in (
        actual[:w, w : w + h],
        actual[w : w + h, w + h : w + h + w],
        actual[w + h : w + h + w, w : w + h],
        actual[w : w + h, :w],
    ):
        vals = sorted(set(map(int, patch.flatten())) - {BG})
        if len(vals) != 1:
            return False
        quadrant_colors.append(vals[0])
    if quadrant_colors != [segment["color"] for segment in segments]:
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    grid = np.full((rows, cols), BG, dtype=int)

    sprite = _make_sprite_patch()
    lengths = sorted(random.sample(SEGMENT_LENGTHS, 4), reverse=True)
    colors = random.sample(SEGMENT_COLORS, 4)

    orientations = [bool(random.getrandbits(1)) for _ in range(4)]
    if len(set(orientations)) == 1 and random.random() < 0.75:
        orientations[random.randrange(4)] = not orientations[0]

    segment_patches = [
        _make_segment_patch(length, color, vertical)
        for length, color, vertical in zip(lengths, colors, orientations)
    ]

    patches = [sprite] + segment_patches
    patches.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
    placed = _place_objects(grid, patches)
    if placed is None:
        return None
    if not _validate_generated(placed):
        return None
    return placed


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
