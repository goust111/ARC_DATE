from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import bbox, colors_present, components_mask, crop_bbox, get_bg, paste, random_free_location


BG = 0
PANEL_H_RANGE = (7, 10)
PANEL_W_RANGE = (10, 14)
SPRITE_SIDE_OPTIONS = (3, 4)
SPRITE_COUNT_OPTIONS = (3, 4)
MAX_GENERATION_ATTEMPTS = 400
PATTERN_MODES = (
    "vertical_split",
    "horizontal_split",
    "checker",
    "diagonal",
    "center_block",
    "corner_block",
)


def _pattern_key(sprite: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in np.asarray(sprite, dtype=int))


def _swap_two_colors(sprite: np.ndarray, bg: int) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    colors = colors_present(out, exclude={bg})
    assert len(colors) == 2, "expected exactly two non-background colors"
    a, b = map(int, colors)
    out[sprite == a] = b
    out[sprite == b] = a
    return out


def _make_sprite(height: int, width: int, colors: tuple[int, int], mode: str | None = None) -> np.ndarray:
    a, b = map(int, colors)
    if mode is None:
        mode = str(np.random.choice(PATTERN_MODES))

    sprite = np.full((height, width), a, dtype=int)

    if mode == "vertical_split":
        split = np.random.randint(1, width)
        sprite[:, split:] = b
    elif mode == "horizontal_split":
        split = np.random.randint(1, height)
        sprite[split:, :] = b
    elif mode == "checker":
        rows, cols = np.indices((height, width))
        sprite[(rows + cols) % 2 == 1] = b
    elif mode == "diagonal":
        rows, cols = np.indices((height, width))
        threshold = np.random.randint(1, height + width - 1)
        sprite[rows + cols >= threshold] = b
    elif mode == "center_block":
        block_h = 1 if height == 3 else np.random.randint(1, height - 1)
        block_w = 1 if width == 3 else np.random.randint(1, width - 1)
        r0 = (height - block_h) // 2
        c0 = (width - block_w) // 2
        sprite[r0 : r0 + block_h, c0 : c0 + block_w] = b
    elif mode == "corner_block":
        block_h = np.random.randint(1, height)
        block_w = np.random.randint(1, width)
        corner = int(np.random.randint(4))
        r0 = 0 if corner < 2 else height - block_h
        c0 = 0 if corner % 2 == 0 else width - block_w
        sprite[r0 : r0 + block_h, c0 : c0 + block_w] = b
    else:
        raise ValueError(f"unknown pattern mode: {mode}")

    if np.all(sprite == a) or np.all(sprite == b):
        return _make_sprite(height, width, colors, mode=None)

    if np.random.rand() < 0.5:
        sprite = np.fliplr(sprite)
    if np.random.rand() < 0.5:
        sprite = np.flipud(sprite)
    return sprite


def _composite_panels(top: np.ndarray, middle: np.ndarray, bottom: np.ndarray, bg: int) -> np.ndarray:
    out = middle.copy()
    top_mask = top != bg
    bottom_mask = bottom != bg
    out[top_mask] = top[top_mask]
    out[bottom_mask] = bottom[bottom_mask]
    return out


def _slot_bboxes(top: np.ndarray, middle: np.ndarray, bottom: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    union_mask = (top != bg) | (middle != bg) | (bottom != bg)
    comps = components_mask(union_mask, connectivity=4)
    out = []
    for comp in comps:
        box = (int(comp[:, 0].min()), int(comp[:, 1].min()), int(comp[:, 0].max()), int(comp[:, 1].max()))
        r0, c0, r1, c1 = box
        slot_mask = union_mask[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(slot_mask), "each sprite position should be a solid rectangle"
        out.append(box)
    out.sort()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare aligned sprites across three stacked panels, isolate the single odd layer, then crop and swap its two colors.

    Input structure:
    - The grid is three equal-height panels stacked vertically on a black background.
    - Each panel contains the same 3-4 separated sprite slots at matching coordinates.
    - Every sprite slot is a fully occupied small rectangle using exactly two non-background colors.

    Rule:
    - Split the input into top, middle, and bottom panels.
    - For each shared sprite slot, extract the three layer-local sprite patterns and compare them directly.
    - Exactly one slot has a 2-versus-1 pattern split across the layers; take the unique sprite from that slot's odd layer.
    - Crop that sprite to its minimal bounding box and swap its two non-background colors.

    Key insight:
    - The layer overlay is a distraction: the hidden odd sprite is found by slotwise comparison of the original panels, not by reading the composite image.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert g.shape[0] % 3 == 0, "expected three equal-height stacked panels"

    bg = int(get_bg(g))
    panel_h = g.shape[0] // 3
    top = g[:panel_h]
    middle = g[panel_h : 2 * panel_h]
    bottom = g[2 * panel_h :]

    boxes = _slot_bboxes(top, middle, bottom, bg)
    assert 3 <= len(boxes) <= 4, "expected 3 or 4 sprite positions"

    odd_sprites: list[np.ndarray] = []
    for r0, c0, r1, c1 in boxes:
        layer_sprites = [panel[r0 : r1 + 1, c0 : c1 + 1].copy() for panel in (top, middle, bottom)]
        keys = [_pattern_key(sprite) for sprite in layer_sprites]
        counts = Counter(keys)
        if len(counts) == 1:
            continue
        assert len(counts) == 2 and sorted(counts.values()) == [1, 2], "expected exactly one odd sprite slot"
        odd_key = next(key for key, count in counts.items() if count == 1)
        odd_sprites.append(layer_sprites[keys.index(odd_key)])

    assert len(odd_sprites) == 1, "expected exactly one anomalous sprite"
    sprite = odd_sprites[0]
    sprite_box = bbox(sprite != bg)
    assert sprite_box is not None, "anomalous sprite must be non-empty"
    cropped = crop_bbox(sprite, sprite_box)
    return _swap_two_colors(cropped, bg)


def _attempt_layout(panel_h: int, panel_w: int, sizes: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    canvas = np.zeros((panel_h, panel_w), dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    for height, width in sorted(sizes, key=lambda shape: shape[0] * shape[1], reverse=True):
        patch = np.ones((height, width), dtype=int)
        try:
            r0, c0 = random_free_location(canvas, patch, transparent=0, border=1, padding=1, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, r0, c0)
        boxes.append((int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)))

    fill_ratio = float(np.count_nonzero(canvas)) / float(canvas.size)
    if not (0.22 <= fill_ratio <= 0.58):
        return None
    comps = components_mask(canvas.astype(bool), connectivity=4)
    if len(comps) != len(sizes):
        return None
    return sorted(boxes)


def _build_panels(
    panel_h: int,
    panel_w: int,
    boxes: list[tuple[int, int, int, int]],
    normal_pairs: list[tuple[int, int]],
    anomaly_pair: tuple[int, int],
    anomaly_slot: int,
    anomaly_layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    panels = [np.zeros((panel_h, panel_w), dtype=int) for _ in range(3)]
    used_normal_keys: set[tuple[tuple[int, ...], ...]] = set()
    repeated_at_anomaly: np.ndarray | None = None
    anomaly_sprite: np.ndarray | None = None

    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        repeated = None
        for _ in range(30):
            candidate = _make_sprite(height, width, normal_pairs[idx])
            key = _pattern_key(candidate)
            if key not in used_normal_keys:
                repeated = candidate
                used_normal_keys.add(key)
                break
        if repeated is None:
            return None

        for layer in range(3):
            paste(panels[layer], repeated, r0, c0, transparent=BG)

        if idx == anomaly_slot:
            repeated_at_anomaly = repeated
            for _ in range(40):
                candidate = _make_sprite(height, width, anomaly_pair)
                if _pattern_key(candidate) != _pattern_key(repeated):
                    anomaly_sprite = candidate
                    break
            if anomaly_sprite is None:
                return None
            paste(panels[anomaly_layer], anomaly_sprite, r0, c0, transparent=BG)

    assert repeated_at_anomaly is not None and anomaly_sprite is not None
    composite = _composite_panels(panels[0], panels[1], panels[2], BG)
    r0, c0, r1, c1 = boxes[anomaly_slot]
    if np.array_equal(composite[r0 : r1 + 1, c0 : c1 + 1], anomaly_sprite):
        return None
    return panels[0], panels[1], panels[2], anomaly_sprite


def _gen_attempt() -> np.ndarray | None:
    panel_h = int(np.random.randint(PANEL_H_RANGE[0], PANEL_H_RANGE[1] + 1))
    panel_w = int(np.random.randint(PANEL_W_RANGE[0], PANEL_W_RANGE[1] + 1))
    sprite_count = int(np.random.choice(SPRITE_COUNT_OPTIONS))
    sizes = [
        (
            int(np.random.choice(SPRITE_SIDE_OPTIONS)),
            int(np.random.choice(SPRITE_SIDE_OPTIONS)),
        )
        for _ in range(sprite_count)
    ]

    boxes = _attempt_layout(panel_h, panel_w, sizes)
    if boxes is None:
        return None

    all_colors = list(range(1, 10))
    anomaly_pair = tuple(int(c) for c in np.random.choice(all_colors, size=2, replace=False))
    remaining = [c for c in all_colors if c not in anomaly_pair]
    normal_pairs = []
    for _ in range(sprite_count):
        pair = tuple(int(c) for c in np.random.choice(remaining, size=2, replace=False))
        normal_pairs.append(pair)

    anomaly_slot = int(np.random.randint(sprite_count))
    anomaly_layer = int(np.random.randint(2))
    built = _build_panels(panel_h, panel_w, boxes, normal_pairs, anomaly_pair, anomaly_slot, anomaly_layer)
    if built is None:
        return None
    top, middle, bottom, anomaly_sprite = built

    grid = np.vstack([top, middle, bottom])
    expected = _swap_two_colors(anomaly_sprite, BG)

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
    raise RuntimeError("failed to generate a valid input for puzzle 9")
