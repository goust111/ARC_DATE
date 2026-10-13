from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
SPRITE_COLORS = (1, 2, 3)
SLOT_COLOR = 4
MARKER_COLOR = 7

GEN_RETRY_LIMIT = 400
GEN_BRANCH_CYCLE = 8

_GEN_COUNTER = 0
_MASK_CACHE: dict[int, list[np.ndarray]] = {}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=int))


def _analyze_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprite_mask = np.isin(g, SPRITE_COLORS)
    sprite_components = _components(sprite_mask)
    assert len(sprite_components) == 1, "expected exactly one sprite component"
    sprite_coords = sprite_components[0]
    sprite_bbox = _bbox_from_coords(sprite_coords)
    sprite = _crop(g, sprite_bbox)
    sh, sw = sprite.shape
    assert (sh, sw) in {(2, 2), (3, 3)}, "sprite bbox must be 2x2 or 3x3"
    sprite_palette = sorted(set(int(v) for v in sprite[sprite != BACKGROUND]))
    assert 2 <= len(sprite_palette) <= 3, "sprite must use 2-3 colors"

    marker_components = _components(g == MARKER_COLOR)
    assert len(marker_components) in {2, 3}, "marker count must encode scale 2 or 3"
    assert all(len(comp) == 1 for comp in marker_components), "markers must be single pixels"
    scale = len(marker_components)

    slot_mask = g == SLOT_COLOR
    slot_bbox = _bbox(slot_mask)
    assert slot_bbox is not None, "expected one yellow target slot"
    slot = _crop(g, slot_bbox)
    assert np.all(slot == SLOT_COLOR), "target slot must be a solid rectangle"
    assert slot.shape == (sh * scale, sw * scale), "slot size must match scaled sprite size"

    return {
        "sprite_bbox": sprite_bbox,
        "sprite": sprite,
        "scale": scale,
        "slot_bbox": slot_bbox,
        "marker_coords": [tuple(map(int, comp[0])) for comp in marker_components],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Count orange markers to choose a pixel-upscale factor, then stamp the enlarged sprite into the yellow slot.

    Input structure:
    - Black background contains one small multi-color sprite made only from blue, red, and green cells.
    - The sprite's tight bounding box is either 2x2 or 3x3, and orange singleton markers elsewhere encode a scale of 2 or 3.
    - A single solid yellow rectangle in the lower-right region is sized to fit the scaled sprite exactly.

    Rule:
    - Crop the sprite from its tight bounding box, keeping any black holes inside that box.
    - Count the orange marker pixels to get scale N, enlarge the cropped sprite by replacing each pixel with an NxN block, and paste the result over the yellow slot with top-left alignment.
    - Erase all orange markers to black while leaving the original sprite untouched.

    Key insight:
    - The yellow rectangle is only a destination slot; the actual enlargement factor is encoded implicitly by the number of orange markers scattered in the input.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    sprite = np.asarray(info["sprite"], dtype=int)
    scale = int(info["scale"])
    r0, c0, _r1, _c1 = info["slot_bbox"]

    out = np.asarray(grid, dtype=int).copy()
    out[out == MARKER_COLOR] = BACKGROUND
    scaled = _scale_sprite(sprite, scale)
    h, w = scaled.shape
    out[r0 : r0 + h, c0 : c0 + w] = scaled
    return out


def _valid_sprite_masks(side: int) -> list[np.ndarray]:
    side = int(side)
    if side in _MASK_CACHE:
        return _MASK_CACHE[side]

    masks: list[np.ndarray] = []
    cells = side * side
    for bits in range(1, 1 << cells):
        mask = np.zeros((side, side), dtype=bool)
        for idx in range(cells):
            if bits & (1 << idx):
                mask[idx // side, idx % side] = True
        area = int(mask.sum())
        if side == 2 and area not in {3, 4}:
            continue
        if side == 3 and not (4 <= area <= 8):
            continue
        if _bbox(mask) != (0, 0, side - 1, side - 1):
            continue
        if len(_components(mask)) != 1:
            continue
        masks.append(mask)

    assert masks, f"no valid masks for side {side}"
    _MASK_CACHE[side] = masks
    return masks


def _choose_sprite(side: int, color_count: int, branch: int) -> np.ndarray:
    masks = _valid_sprite_masks(side)
    prefer_hole = side == 3 and branch % 4 == 0
    prefer_sparse = branch % 4 == 1

    candidates = masks
    if prefer_hole:
        holey = [mask for mask in masks if not mask[1, 1]]
        if holey:
            candidates = holey
    elif prefer_sparse:
        sparse = [mask for mask in masks if int(mask.sum()) < side * side]
        if sparse:
            candidates = sparse

    mask = random.choice(candidates)
    palette = random.sample(list(SPRITE_COLORS), color_count)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    values = palette[:]
    while len(values) < len(coords):
        values.append(random.choice(palette))
    random.shuffle(values)

    sprite = np.zeros((side, side), dtype=int)
    for (r, c), value in zip(coords, values):
        sprite[r, c] = int(value)
    return sprite


def _expanded_box(box: tuple[int, int, int, int], shape: tuple[int, int], pad: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _cell_in_box(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _zone_id(r: int, c: int, shape: tuple[int, int]) -> int:
    h, w = shape
    if r < h // 2 and c >= w // 2:
        return 0
    if r >= h // 2 and c < w // 2:
        return 1
    return 2


def _place_markers(
    grid: np.ndarray,
    count: int,
    sprite_box: tuple[int, int, int, int],
    slot_box: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    forbidden_a = _expanded_box(sprite_box, grid.shape, 1)
    forbidden_b = _expanded_box(slot_box, grid.shape, 1)
    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if grid[r, c] == BACKGROUND
        and not _cell_in_box(r, c, forbidden_a)
        and not _cell_in_box(r, c, forbidden_b)
    ]
    if len(candidates) < count:
        return None

    desired_zone_count = 2 if count == 2 else 3
    for _ in range(80):
        random.shuffle(candidates)
        chosen: list[tuple[int, int]] = []
        used_zones: set[int] = set()
        for r, c in candidates:
            if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in chosen):
                continue
            zone = _zone_id(r, c, grid.shape)
            if len(used_zones) < desired_zone_count and zone in used_zones:
                continue
            chosen.append((r, c))
            used_zones.add(zone)
            if len(chosen) == count:
                return chosen

    for _ in range(80):
        chosen = random.sample(candidates, count)
        if all(max(abs(r0 - r1), abs(c0 - c1)) > 1 for i, (r0, c0) in enumerate(chosen) for r1, c1 in chosen[i + 1 :]):
            return chosen
    return None


def _render_expected(grid: np.ndarray, sprite: np.ndarray, scale: int, slot_pos: tuple[int, int]) -> np.ndarray:
    out = grid.copy()
    out[out == MARKER_COLOR] = BACKGROUND
    scaled = _scale_sprite(sprite, scale)
    r0, c0 = slot_pos
    h, w = scaled.shape
    out[r0 : r0 + h, c0 : c0 + w] = scaled
    return out


def _build_instance(branch: int) -> dict[str, object] | None:
    side = 2 if branch % 2 == 0 else 3
    scale = 2 if (branch // 2) % 2 == 0 else 3
    color_count = 2 if (branch // 4) % 2 == 0 else 3
    sprite = _choose_sprite(side, color_count, branch)
    slot_side = side * scale

    top_pad = random.randint(0, 2)
    left_pad = random.randint(0, 2)
    gap_r = random.randint(2, 5)
    gap_c = random.randint(2, 5)
    bottom_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 2)

    height = top_pad + side + gap_r + slot_side + bottom_pad + extra_h
    width = left_pad + side + gap_c + slot_side + right_pad + extra_w
    if max(height, width) > 30:
        return None

    sprite_r = top_pad
    sprite_c = left_pad
    slot_r = height - bottom_pad - slot_side
    slot_c = width - right_pad - slot_side
    if slot_r - (sprite_r + side) < 2 or slot_c - (sprite_c + side) < 2:
        return None

    grid = np.zeros((height, width), dtype=int)
    sprite_box = (sprite_r, sprite_c, sprite_r + side - 1, sprite_c + side - 1)
    grid[sprite_r : sprite_r + side, sprite_c : sprite_c + side] = sprite
    grid[slot_r : slot_r + slot_side, slot_c : slot_c + slot_side] = SLOT_COLOR
    slot_box = (slot_r, slot_c, slot_r + slot_side - 1, slot_c + slot_side - 1)

    markers = _place_markers(grid, scale, sprite_box, slot_box)
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = MARKER_COLOR

    expected = _render_expected(grid, sprite, scale, (slot_r, slot_c))
    return {
        "input": grid,
        "output": expected,
        "sprite": sprite,
        "scale": scale,
        "sprite_box": sprite_box,
        "slot_box": slot_box,
        "markers": markers,
        "color_count": color_count,
    }


def _validate_instance(spec: dict[str, object]) -> bool:
    inp = np.asarray(spec["input"], dtype=int)
    expected = np.asarray(spec["output"], dtype=int)
    sprite = np.asarray(spec["sprite"], dtype=int)
    scale = int(spec["scale"])
    slot_box = spec["slot_box"]
    sprite_box = spec["sprite_box"]
    markers = list(spec["markers"])
    color_count = int(spec["color_count"])

    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(inp, expected):
        return False
    if len(markers) != scale or scale not in {2, 3}:
        return False
    if len(set(int(v) for v in sprite[sprite != BACKGROUND])) != color_count:
        return False
    if not np.all(inp[sprite_box[0] : sprite_box[2] + 1, sprite_box[1] : sprite_box[3] + 1] == sprite):
        return False
    sr0, sc0, sr1, sc1 = sprite_box
    tr0, tc0, tr1, tc1 = slot_box
    if (sr1 - sr0 + 1, sc1 - sc0 + 1) not in {(2, 2), (3, 3)}:
        return False
    if not np.all(inp[tr0 : tr1 + 1, tc0 : tc1 + 1] == SLOT_COLOR):
        return False

    analyzed = _analyze_input(inp)
    if analyzed["scale"] != scale:
        return False
    if not np.array_equal(np.asarray(analyzed["sprite"]), sprite):
        return False
    if analyzed["slot_bbox"] != slot_box:
        return False

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False

    erase_only = inp.copy()
    erase_only[erase_only == MARKER_COLOR] = BACKGROUND
    if np.array_equal(erase_only, expected):
        return False

    wrong_scale = 5 - scale
    if (sprite.shape[0] * wrong_scale, sprite.shape[1] * wrong_scale) == (tr1 - tr0 + 1, tc1 - tc0 + 1):
        return False

    if any(inp[r, c] != MARKER_COLOR for r, c in markers):
        return False
    if any(expected[r, c] != BACKGROUND for r, c in markers):
        return False
    return True


def generate_input() -> np.ndarray:
    global _GEN_COUNTER

    branch = _GEN_COUNTER % GEN_BRANCH_CYCLE
    _GEN_COUNTER += 1

    for _ in range(GEN_RETRY_LIMIT):
        spec = _build_instance(branch)
        if spec is not None and _validate_instance(spec):
            return np.asarray(spec["input"], dtype=int)
    raise RuntimeError("failed to generate a valid input")
