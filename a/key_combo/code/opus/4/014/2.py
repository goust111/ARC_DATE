from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_THIS_FILE = Path(__file__).resolve()
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _candidate = _parent / "new" / "solvers"
    if (_candidate / "arc_utils.py").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
else:
    raise ImportError("Could not locate new/solvers/arc_utils.py")

from arc_utils import Color, components_mask, paste


SPRITE_SIZE = 3
SPRITE_BOX = (0, 0, 2, 2)
HALF_MIN = 6
HALF_MAX = 12
MARKER_MIN = 4
MARKER_MAX = 8
GEN_ATTEMPTS = 400
QUADRANTS = ("tl", "tr", "bl", "br")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _marker_box(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 1, c - 1, r + 1, c + 1


def _divider_indices(grid: np.ndarray) -> tuple[int, int]:
    row_hits = np.flatnonzero(np.all(grid == Color.GREY, axis=1))
    col_hits = np.flatnonzero(np.all(grid == Color.GREY, axis=0))
    assert len(row_hits) == 1 and len(col_hits) == 1, "expected one full grey row and one full grey column"
    return int(row_hits[0]), int(col_hits[0])


def _sprite_variants(sprite: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return sprite, np.fliplr(sprite), np.flipud(sprite), np.rot90(sprite, 2)


def _variant_for_marker(sprite: np.ndarray, r: int, c: int, div_r: int, div_c: int) -> np.ndarray:
    if r < div_r and c < div_c:
        return sprite
    if r < div_r and c > div_c:
        return np.fliplr(sprite)
    if r > div_r and c < div_c:
        return np.flipud(sprite)
    return np.rot90(sprite, 2)


def _quadrant_name(r: int, c: int, div_r: int, div_c: int) -> str:
    if r < div_r and c < div_c:
        return "tl"
    if r < div_r and c > div_c:
        return "tr"
    if r > div_r and c < div_c:
        return "bl"
    return "br"


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    sprite = np.where(grid[:SPRITE_SIZE, :SPRITE_SIZE] == Color.BLUE, Color.BLUE, Color.BLACK)
    return sprite.astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp reflected copies of the corner blue sprite onto red markers according to each marker's quadrant.

    Input structure:
    - The grid is a black background split by one full grey horizontal line and one full grey vertical line into four quadrants.
    - A blue asymmetric 4-connected sprite occupies the fixed top-left 3x3 corner box.
    - Each quadrant contains one or more isolated red marker pixels whose 3x3 neighborhoods do not hit the divider lines or each other.

    Rule:
    - Read the blue sprite from the top-left 3x3 box.
    - For every red marker, choose the sprite orientation from its quadrant: original in the top-left, horizontal flip in the top-right, vertical flip in the bottom-left, and 180-degree rotation in the bottom-right.
    - Stamp that 3x3 oriented sprite with its center aligned to the marker, painting only blue cells.
    - Remove the red markers, while keeping the divider lines and the original corner sprite unchanged.

    Key insight:
    - The marker carries no explicit direction code; its position relative to the two divider lines implicitly selects which reflected sprite variant must be stamped there.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    div_r, div_c = _divider_indices(g)
    sprite = _extract_sprite(g)
    marker_coords = [tuple(map(int, coord)) for coord in np.argwhere(g == Color.RED)]

    out = g.copy()
    out[g == Color.RED] = Color.BLACK
    for r, c in marker_coords:
        oriented = _variant_for_marker(sprite, r, c, div_r, div_c)
        paste(out, oriented, r - 1, c - 1, transparent=Color.BLACK, clip=False)

    out[div_r, :] = Color.GREY
    out[:, div_c] = Color.GREY
    out[:SPRITE_SIZE, :SPRITE_SIZE] = g[:SPRITE_SIZE, :SPRITE_SIZE]
    return out


def _enumerate_valid_sprites() -> tuple[list[np.ndarray], list[np.ndarray]]:
    center_on: list[np.ndarray] = []
    center_off: list[np.ndarray] = []
    for bits in range(1 << (SPRITE_SIZE * SPRITE_SIZE)):
        area = bits.bit_count()
        if area < 3 or area > 5:
            continue
        flat = np.array([(bits >> idx) & 1 for idx in range(SPRITE_SIZE * SPRITE_SIZE)], dtype=int)
        mask = flat.reshape(SPRITE_SIZE, SPRITE_SIZE)
        if len(components_mask(mask.astype(bool), connectivity=4)) != 1:
            continue
        variants = _sprite_variants(mask)
        if len({variant.tobytes() for variant in variants}) != 4:
            continue
        colored = (mask * Color.BLUE).astype(int)
        if mask[1, 1]:
            center_on.append(colored)
        else:
            center_off.append(colored)
    assert center_on and center_off, "expected both center-on and center-off asymmetric sprites"
    return center_on, center_off


_CENTER_ON_SPRITES, _CENTER_OFF_SPRITES = _enumerate_valid_sprites()


def _candidate_centers(half_h: int, half_w: int) -> dict[str, list[tuple[int, int]]]:
    height = 2 * half_h + 1
    width = 2 * half_w + 1
    top_rows = range(1, half_h - 1)
    bottom_rows = range(half_h + 2, height - 1)
    left_cols = range(1, half_w - 1)
    right_cols = range(half_w + 2, width - 1)

    candidates = {
        "tl": [(r, c) for r in top_rows for c in left_cols],
        "tr": [(r, c) for r in top_rows for c in right_cols],
        "bl": [(r, c) for r in bottom_rows for c in left_cols],
        "br": [(r, c) for r in bottom_rows for c in right_cols],
    }
    candidates["tl"] = [pt for pt in candidates["tl"] if not _boxes_overlap(_marker_box(pt), SPRITE_BOX)]
    return candidates


def _place_markers(half_h: int, half_w: int, total_markers: int) -> list[tuple[int, int]] | None:
    candidates = _candidate_centers(half_h, half_w)
    placed: list[tuple[int, int]] = []
    placed_boxes: list[tuple[int, int, int, int]] = []

    for quadrant in QUADRANTS:
        options = candidates[quadrant][:]
        random.shuffle(options)
        pick = None
        for center in options:
            box = _marker_box(center)
            if all(not _boxes_overlap(box, other) for other in placed_boxes):
                pick = center
                break
        if pick is None:
            return None
        placed.append(pick)
        placed_boxes.append(_marker_box(pick))

    while len(placed) < total_markers:
        available: list[tuple[int, int]] = []
        for quadrant in QUADRANTS:
            for center in candidates[quadrant]:
                if center in placed:
                    continue
                box = _marker_box(center)
                if all(not _boxes_overlap(box, other) for other in placed_boxes):
                    available.append(center)
        if not available:
            return None
        pick = random.choice(available)
        placed.append(pick)
        placed_boxes.append(_marker_box(pick))

    random.shuffle(placed)
    return placed


def _expected_output_from_spec(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    div_r, div_c = _divider_indices(g)
    sprite = _extract_sprite(g)
    out = g.copy()
    markers = [tuple(map(int, coord)) for coord in np.argwhere(g == Color.RED)]
    out[g == Color.RED] = Color.BLACK

    for r, c in markers:
        oriented = _variant_for_marker(sprite, r, c, div_r, div_c)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if oriented[dr + 1, dc + 1] == Color.BLUE:
                    out[r + dr, c + dc] = Color.BLUE

    out[div_r, :] = Color.GREY
    out[:, div_c] = Color.GREY
    out[:SPRITE_SIZE, :SPRITE_SIZE] = g[:SPRITE_SIZE, :SPRITE_SIZE]
    return out


def _validate_input(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.BLUE, Color.RED, Color.GREY}

    div_r, div_c = _divider_indices(g)
    line_mask = np.zeros_like(g, dtype=bool)
    line_mask[div_r, :] = True
    line_mask[:, div_c] = True
    assert not np.any((g == Color.GREY) & ~line_mask), "grey should appear only on divider lines"

    sprite = _extract_sprite(g)
    sprite_mask = sprite == Color.BLUE
    assert 3 <= int(np.sum(sprite_mask)) <= 5
    assert len(components_mask(sprite_mask, connectivity=4)) == 1
    assert len({variant.tobytes() for variant in _sprite_variants(sprite_mask.astype(int))}) == 4

    blue_coords = np.argwhere(g == Color.BLUE)
    assert blue_coords.size > 0
    assert all(int(r) < SPRITE_SIZE and int(c) < SPRITE_SIZE for r, c in blue_coords)

    markers = [tuple(map(int, coord)) for coord in np.argwhere(g == Color.RED)]
    assert MARKER_MIN <= len(markers) <= MARKER_MAX
    counts = {name: 0 for name in QUADRANTS}
    boxes: list[tuple[int, int, int, int]] = []
    for r, c in markers:
        quadrant = _quadrant_name(r, c, div_r, div_c)
        counts[quadrant] += 1
        box = _marker_box((r, c))
        assert 0 <= box[0] and 0 <= box[1] and box[2] < g.shape[0] and box[3] < g.shape[1]
        assert not (box[0] <= div_r <= box[2]), "marker neighborhood touches divider row"
        assert not (box[1] <= div_c <= box[3]), "marker neighborhood touches divider column"
        assert not _boxes_overlap(box, SPRITE_BOX), "marker neighborhood touches the source sprite"
        assert all(not _boxes_overlap(box, other) for other in boxes), "marker neighborhoods overlap"
        boxes.append(box)
    assert all(counts[name] >= 1 for name in QUADRANTS)


def _sample_sprite() -> np.ndarray:
    pool = _CENTER_ON_SPRITES if random.random() < 0.5 else _CENTER_OFF_SPRITES
    return random.choice(pool).copy()


def _gen_attempt() -> np.ndarray | None:
    half_h = random.randint(HALF_MIN, HALF_MAX)
    half_w = random.randint(HALF_MIN, HALF_MAX)
    height = 2 * half_h + 1
    width = 2 * half_w + 1

    grid = np.zeros((height, width), dtype=int)
    grid[half_h, :] = Color.GREY
    grid[:, half_w] = Color.GREY
    grid[:SPRITE_SIZE, :SPRITE_SIZE] = _sample_sprite()

    total_markers = random.randint(MARKER_MIN, MARKER_MAX)
    markers = _place_markers(half_h, half_w, total_markers)
    if markers is None:
        return None
    for r, c in markers:
        grid[r, c] = Color.RED

    try:
        _validate_input(grid)
    except AssertionError:
        return None

    expected = _expected_output_from_spec(grid)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-reflection stamping input")
