from __future__ import annotations

import itertools
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
MARKER = 6
SPRITE_COLORS = (1, 2, 3, 4, 7, 8, 9)
QUADRANT_ORDER = ("UL", "UR", "LL", "LR")
MARKER_ANCHORS = {
    "UL": lambda q: (0, 0),
    "UR": lambda q: (0, q - 1),
    "LL": lambda q: (q - 1, 0),
    "LR": lambda q: (q - 1, q - 1),
}
# np.rot90 k is counter-clockwise; these values encode the required clockwise turns.
ROTATION_K_BY_MARKER_QUADRANT = {
    "UL": 0,
    "UR": 3,
    "LL": 1,
    "LR": 2,
}
QUADRANT_SIZE_RANGE = (5, 8)
GENERATION_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        count += 1
        stack = [(int(r), int(c))]
        seen[r, c] = True
        while stack:
            rr, cc = stack.pop()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
    return count


def _valid_shape_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    cells = [(r, c) for r in range(3) for c in range(3)]
    for area in range(3, 6):
        for chosen in itertools.combinations(cells, area):
            mask = np.zeros((3, 3), dtype=bool)
            for r, c in chosen:
                mask[r, c] = True
            box = _bbox(mask)
            if box != (0, 0, 2, 2):
                continue
            if _component_count(mask) != 1:
                continue
            if any(np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3)):
                continue
            masks.append(mask)
    return masks


VALID_SHAPES = _valid_shape_masks()


def _extract_crop(region: np.ndarray, quadrant_name: str) -> tuple[np.ndarray, int, bool]:
    q = region.shape[0]
    assert region.shape == (q, q), "quadrants must be square and equally sized"

    marker_cells = np.argwhere(region == MARKER)
    has_marker = len(marker_cells) == 1
    assert len(marker_cells) <= 1, "expected at most one marker per quadrant"

    work = region.copy()
    if has_marker:
        mr, mc = map(int, marker_cells[0])
        assert (mr, mc) == MARKER_ANCHORS[quadrant_name](q), "marker must sit at the quadrant's outer corner"
        work[mr, mc] = BACKGROUND

    colors = {int(v) for v in np.unique(work)} - {BACKGROUND}
    assert len(colors) == 1, "each quadrant must contain exactly one sprite color after removing the marker"
    color = next(iter(colors))
    assert color not in {SEPARATOR, MARKER}, "sprite colors must differ from separator and marker"

    mask = work == color
    assert 3 <= int(mask.sum()) <= 5, "sprite area must be 3 to 5 cells"
    assert _component_count(mask) == 1, "sprite must be 4-connected"
    box = _bbox(mask)
    assert box is not None and (box[2] - box[0] + 1, box[3] - box[1] + 1) == (3, 3), "sprite bbox must be 3x3"
    r0, c0, r1, c1 = box
    crop = work[r0 : r1 + 1, c0 : c1 + 1].copy()
    return crop, color, has_marker


def _rotation_k_from_marker_quadrant(marker_quadrant: str) -> int:
    return ROTATION_K_BY_MARKER_QUADRANT[marker_quadrant]


def _assemble(crops: list[np.ndarray], rotation_k: int) -> np.ndarray:
    rotated = [np.rot90(crop, k=rotation_k) for crop in crops]
    top = np.hstack(rotated[:2])
    bottom = np.hstack(rotated[2:])
    return np.vstack([top, bottom]).astype(int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the four cross-separated sprites, rotate them by the marker quadrant's global code, and pack them into a 2x2 montage.

    Input structure:
    - A black background contains one full grey row and one full grey column that form a separator cross through the canvas center.
    - The cross splits the grid into four equal square quadrants, each containing exactly one single-color 4-connected sprite with a tight 3x3 bounding box.
    - The four sprites share the same shape but use four distinct colors, and exactly one quadrant also contains a magenta singleton marker at that quadrant's outer corner.

    Rule:
    - Remove the grey separator cross and ignore the magenta marker.
    - Crop each quadrant's sprite to its tight 3x3 bounding box.
    - Read the marker quadrant as a global rotation code: upper-left means no rotation, upper-right means clockwise 90 degrees, lower-right means 180 degrees, and lower-left means clockwise 270 degrees.
    - Rotate every cropped sprite by that same amount and concatenate the four 3x3 crops in upper-left, upper-right, lower-left, lower-right order.

    Key insight:
    - The marker does not point to a corner pair; its quadrant alone encodes one global quarter-turn that must be applied uniformly to all four cropped sprites.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = map(int, g.shape)
    assert h % 2 == 1 and w % 2 == 1, "separator cross must split the grid into equal halves"

    separator_rows = np.flatnonzero(np.all(g == SEPARATOR, axis=1))
    separator_cols = np.flatnonzero(np.all(g == SEPARATOR, axis=0))
    assert len(separator_rows) == 1 and len(separator_cols) == 1, "expected one solid separator cross"
    cross_r = int(separator_rows[0])
    cross_c = int(separator_cols[0])
    assert cross_r == h // 2 and cross_c == w // 2, "separator cross must be centered"

    qh, qw = cross_r, cross_c
    assert qh == h - cross_r - 1 and qw == w - cross_c - 1 and qh == qw, "quadrants must be equal squares"
    q = qh

    regions = {
        "UL": g[:cross_r, :cross_c],
        "UR": g[:cross_r, cross_c + 1 :],
        "LL": g[cross_r + 1 :, :cross_c],
        "LR": g[cross_r + 1 :, cross_c + 1 :],
    }

    crops: list[np.ndarray] = []
    colors: list[int] = []
    shape_masks: list[np.ndarray] = []
    marker_quadrants: list[str] = []
    for quadrant_name in QUADRANT_ORDER:
        crop, color, has_marker = _extract_crop(regions[quadrant_name], quadrant_name)
        crops.append(crop)
        colors.append(color)
        shape_masks.append(crop != BACKGROUND)
        if has_marker:
            marker_quadrants.append(quadrant_name)

    assert len(set(colors)) == 4, "the four sprites must use distinct colors"
    assert len(marker_quadrants) == 1, "expected exactly one marker quadrant"
    assert all(np.array_equal(shape_masks[0], mask) for mask in shape_masks[1:]), "all quadrants must contain the same sprite shape"

    rotation_k = _rotation_k_from_marker_quadrant(marker_quadrants[0])
    return _assemble(crops, rotation_k)


def _paint_sprite(patch: np.ndarray, mask: np.ndarray, color: int) -> None:
    patch[mask] = color


def _random_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    _paint_sprite(patch, mask, color)
    return patch


def _quadrant_bounds(quadrant_name: str, q: int) -> tuple[int, int]:
    if quadrant_name == "UL":
        return 0, 0
    if quadrant_name == "UR":
        return 0, q + 1
    if quadrant_name == "LL":
        return q + 1, 0
    return q + 1, q + 1


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _local_marker_anchor(quadrant_name: str, q: int) -> tuple[int, int]:
    return MARKER_ANCHORS[quadrant_name](q)


def _sample_positions(q: int, marker_quadrant: str) -> dict[str, tuple[int, int]] | None:
    choices: dict[str, list[tuple[int, int]]] = {}
    for quadrant_name in QUADRANT_ORDER:
        anchor_r, anchor_c = _local_marker_anchor(quadrant_name, q)
        valid_positions: list[tuple[int, int]] = []
        for r0 in range(q - 2):
            for c0 in range(q - 2):
                if quadrant_name == marker_quadrant and r0 <= anchor_r <= r0 + 2 and c0 <= anchor_c <= c0 + 2:
                    continue
                valid_positions.append((r0, c0))
        if not valid_positions:
            return None
        choices[quadrant_name] = valid_positions

    positions = {name: random.choice(choices[name]) for name in QUADRANT_ORDER}
    if len(set(positions.values())) == 1:
        return None
    return positions


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    q = random.randint(*QUADRANT_SIZE_RANGE)
    marker_quadrant = random.choice(QUADRANT_ORDER)
    mask = random.choice(VALID_SHAPES)
    colors = random.sample(SPRITE_COLORS, 4)
    positions = _sample_positions(q, marker_quadrant)
    if positions is None:
        return None

    grid = np.zeros((2 * q + 1, 2 * q + 1), dtype=int)
    grid[q, :] = SEPARATOR
    grid[:, q] = SEPARATOR

    crops: list[np.ndarray] = []
    for quadrant_name, color in zip(QUADRANT_ORDER, colors):
        patch = _random_patch(mask, color)
        crops.append(patch)
        qr, qc = _quadrant_bounds(quadrant_name, q)
        pr, pc = positions[quadrant_name]
        _place_patch(grid, patch, qr + pr, qc + pc)

    mr, mc = _local_marker_anchor(marker_quadrant, q)
    qtop, qleft = _quadrant_bounds(marker_quadrant, q)
    if grid[qtop + mr, qleft + mc] != BACKGROUND:
        return None
    grid[qtop + mr, qleft + mc] = MARKER

    expected = _assemble(crops, _rotation_k_from_marker_quadrant(marker_quadrant))
    return grid, expected


def _validate_input(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(out, expected):
        return False
    if out.shape != (6, 6):
        return False

    g = np.asarray(grid, dtype=int)
    q = g.shape[0] // 2
    regions = {
        "UL": g[:q, :q],
        "UR": g[:q, q + 1 :],
        "LL": g[q + 1 :, :q],
        "LR": g[q + 1 :, q + 1 :],
    }

    marker_quadrants = []
    crops = []
    placements = []
    for quadrant_name in QUADRANT_ORDER:
        crop, _color, has_marker = _extract_crop(regions[quadrant_name], quadrant_name)
        crops.append(crop)
        box = _bbox((regions[quadrant_name] != BACKGROUND) & (regions[quadrant_name] != MARKER))
        assert box is not None
        placements.append((box[0], box[1]))
        if has_marker:
            marker_quadrants.append(quadrant_name)

    if len(marker_quadrants) != 1:
        return False
    if len(set(placements)) == 1:
        return False

    unrotated = _assemble(crops, 0)
    all_outputs = {
        _assemble(crops, _rotation_k_from_marker_quadrant(name)).tobytes()
        for name in QUADRANT_ORDER
    }
    if len(all_outputs) != 4:
        return False
    if marker_quadrants[0] != "UL" and np.array_equal(out, unrotated):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, expected = candidate
        if _validate_input(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid marker-quadrant rotation example")
