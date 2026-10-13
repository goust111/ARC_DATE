from __future__ import annotations

import random
from collections import deque

import numpy as np


GRID_SIZE = 12
BACKGROUND = 0
MARKER_COLORS = (3, 4, 5, 7)
SPRITE_COLORS = (1, 2, 6, 8, 9)
CORNER_ORDER = ("top_left", "top_right", "bottom_right", "bottom_left")
CORNER_CELLS = {
    "top_left": (0, 0),
    "top_right": (0, GRID_SIZE - 1),
    "bottom_right": (GRID_SIZE - 1, GRID_SIZE - 1),
    "bottom_left": (GRID_SIZE - 1, 0),
}
# np.rot90 turns counter-clockwise; the rule is specified in clockwise angles.
ROTATION_TURNS = {
    "top_left": 0,
    "top_right": 3,
    "bottom_right": 2,
    "bottom_left": 1,
}
MAX_GENERATION_ATTEMPTS = 400


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
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste_nonzero(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    nz = patch != BACKGROUND
    view[nz] = patch[nz]


def _stamp_top_left(corner: str, patch_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, patch_shape)
    if corner == "top_left":
        return 1, 1
    if corner == "top_right":
        return 1, 10 - (w - 1)
    if corner == "bottom_right":
        return 10 - (h - 1), 10 - (w - 1)
    if corner == "bottom_left":
        return 10 - (h - 1), 1
    raise ValueError(f"unknown corner {corner}")


def _stamp_bbox(corner: str, size: int) -> tuple[int, int, int, int]:
    r0, c0 = _stamp_top_left(corner, (size, size))
    return r0, c0, r0 + size - 1, c0 + size - 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _rotation_family_size(mask: np.ndarray) -> int:
    keys = {tuple(np.rot90(mask, k).astype(int).ravel()) for k in range(4)}
    return len(keys)


def _extract_template_and_markers(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape == (GRID_SIZE, GRID_SIZE), "expected a 12x12 grid"

    markers: dict[str, int] = {}
    for corner, (r, c) in CORNER_CELLS.items():
        color = int(g[r, c])
        if color != BACKGROUND:
            markers[corner] = color
    assert 1 <= len(markers) <= 3, "expected one to three corner markers"

    core_mask = g != BACKGROUND
    for r, c in CORNER_CELLS.values():
        core_mask[r, c] = False

    comps = _components(core_mask)
    assert len(comps) == 1, "expected exactly one non-corner foreground component"
    coords = comps[0]
    sprite_mask = np.zeros_like(core_mask, dtype=bool)
    sprite_mask[coords[:, 0], coords[:, 1]] = True

    box = _bbox(sprite_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    size_h = r1 - r0 + 1
    size_w = c1 - c0 + 1
    assert size_h == size_w and size_h in (3, 4), "expected a 3x3 or 4x4 sprite bbox"
    assert 4 <= int(coords.shape[0]) <= 8, "expected a sprite with 4-8 cells"
    assert _rotation_family_size(_crop_bbox(sprite_mask.astype(int), box).astype(bool)) == 4, (
        "sprite occupancy must have four distinct rotations"
    )

    template = _crop_bbox(g, box)
    return template, markers, box


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the central sprite and stamp recolored rotated copies near the marked corners.

    Input structure:
    - The grid is a 12x12 black canvas with one multicolor 4-connected sprite near the center.
    - The sprite occupies a 3x3 or 4x4 bounding box and is the only non-corner foreground object.
    - One to three corners contain single-pixel markers whose colors are distinct from the sprite colors.

    Rule:
    - Crop the full bounding box of the central sprite, preserving any black cells inside that box.
    - Use the marked corner position to choose a rotation: top-left=0, top-right=clockwise 90, bottom-right=180, bottom-left=clockwise 270.
    - Recolor every non-black pixel of the rotated crop to the marker color and stamp it one cell inward from that corner.
    - Erase the original corner markers to black and leave the center sprite unchanged.

    Key insight:
    - Every corner uses the same cropped sprite as its template; only the corner location determines the rotation, while the marker color supplies the paint.
    """
    g = np.asarray(grid, dtype=int)
    template, markers, _box = _extract_template_and_markers(g)

    out = g.copy()
    for r, c in CORNER_CELLS.values():
        out[r, c] = BACKGROUND

    for corner, color in markers.items():
        rotated = np.rot90(template, ROTATION_TURNS[corner])
        stamped = np.where(rotated != BACKGROUND, int(color), BACKGROUND).astype(int)
        r0, c0 = _stamp_top_left(corner, stamped.shape)
        _paste_nonzero(out, stamped, r0, c0)

    return out


def _enumerate_valid_masks(size: int) -> tuple[np.ndarray, ...]:
    valid: list[np.ndarray] = []
    total = size * size
    for bits in range(1 << total):
        area = bits.bit_count()
        if not 4 <= area <= 8:
            continue
        mask = np.zeros((size, size), dtype=bool)
        for idx in range(total):
            if bits & (1 << idx):
                mask[idx // size, idx % size] = True
        if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(_components(mask)) != 1:
            continue
        if _rotation_family_size(mask) != 4:
            continue
        valid.append(mask)
    return tuple(valid)


VALID_MASKS = {
    3: _enumerate_valid_masks(3),
    4: _enumerate_valid_masks(4),
}


def _colorize_mask(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    fg = list(map(tuple, np.argwhere(mask)))
    random.shuffle(fg)
    split = random.randint(1, len(fg) - 1)
    first = set(fg[:split])
    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in fg:
        sprite[r, c] = colors[0] if (r, c) in first else colors[1]
    return sprite


def _corner_subset(size: int) -> tuple[str, ...]:
    if size == 4:
        return (random.choice(CORNER_ORDER),)
    roll = random.random()
    count = 1 if roll < 0.15 else 2 if roll < 0.60 else 3
    return tuple(random.sample(CORNER_ORDER, count))


def _center_positions(size: int, active_corners: tuple[str, ...]) -> list[tuple[int, int]]:
    candidates = []
    sprite_box_candidates = range(3, 6) if size == 4 else range(4, 6)
    forbidden = [_stamp_bbox(corner, size) for corner in active_corners]
    for r0 in sprite_box_candidates:
        for c0 in sprite_box_candidates:
            center_box = (r0, c0, r0 + size - 1, c0 + size - 1)
            if any(_boxes_overlap(center_box, box) for box in forbidden):
                continue
            candidates.append((r0, c0))
    return candidates


def _render_expected_from_parts(
    inp: np.ndarray,
    sprite_box: tuple[int, int, int, int],
    markers: dict[str, int],
) -> np.ndarray:
    out = inp.copy()
    for r, c in CORNER_CELLS.values():
        out[r, c] = BACKGROUND
    template = _crop_bbox(inp, sprite_box)
    for corner, color in markers.items():
        rotated = np.rot90(template, ROTATION_TURNS[corner])
        patch = np.where(rotated != BACKGROUND, int(color), BACKGROUND).astype(int)
        r0, c0 = _stamp_top_left(corner, patch.shape)
        _paste_nonzero(out, patch, r0, c0)
    return out


def _validate_constructed_instance(
    inp: np.ndarray,
    sprite_mask: np.ndarray,
    sprite_box: tuple[int, int, int, int],
    markers: dict[str, int],
) -> bool:
    if inp.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if not (1 <= len(markers) <= 3):
        return False
    if len(set(markers.values())) != len(markers):
        return False

    size = sprite_mask.shape[0]
    if size not in (3, 4) or sprite_mask.shape[1] != size:
        return False
    if _rotation_family_size(sprite_mask) != 4:
        return False

    sprite_colors = set(int(x) for x in np.unique(inp) if x not in (BACKGROUND, *markers.values()))
    if len(sprite_colors) != 2:
        return False
    if not sprite_colors.isdisjoint(MARKER_COLORS):
        return False

    for corner in markers:
        if any(_boxes_overlap(_stamp_bbox(corner, size), _stamp_bbox(other, size)) for other in markers if other != corner):
            return False
        if _boxes_overlap(_stamp_bbox(corner, size), sprite_box):
            return False

    expected = _render_expected_from_parts(inp, sprite_box, markers)
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    out = actual
    if any(int(out[r, c]) != BACKGROUND for r, c in CORNER_CELLS.values()):
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    size = 4 if random.random() < 0.22 else 3
    active_corners = _corner_subset(size)
    center_positions = _center_positions(size, active_corners)
    if not center_positions:
        return None

    sprite_mask = random.choice(VALID_MASKS[size])
    sprite_colors = tuple(random.sample(SPRITE_COLORS, 2))
    sprite = _colorize_mask(sprite_mask, sprite_colors)

    marker_colors = random.sample(MARKER_COLORS, len(active_corners))
    markers = {corner: color for corner, color in zip(active_corners, marker_colors)}

    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for corner, color in markers.items():
        r, c = CORNER_CELLS[corner]
        grid[r, c] = color

    r0, c0 = random.choice(center_positions)
    grid[r0 : r0 + size, c0 : c0 + size] = sprite
    sprite_box = (r0, c0, r0 + size - 1, c0 + size - 1)

    if not _validate_constructed_instance(grid, sprite_mask, sprite_box, markers):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
