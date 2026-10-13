from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLUE = 1
RED = 2
BLACK = 0


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


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    *,
    transparent: int | None = None,
    clip: bool = False,
) -> np.ndarray:
    h, w = patch.shape
    H, W = dst.shape
    if not clip:
        if top < 0 or left < 0 or top + h > H or left + w > W:
            raise ValueError("patch is out of bounds")
        rr0, cc0, rr1, cc1 = top, left, top + h, left + w
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(H, top + h)
        cc1 = min(W, left + w)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - top
        pc0 = cc0 - left
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    view = patch[pr0:pr1, pc0:pc1]
    if transparent is None:
        dst[rr0:rr1, cc0:cc1] = view
    else:
        mask = view != int(transparent)
        dst_view = dst[rr0:rr1, cc0:cc1]
        dst_view[mask] = view[mask]
    return dst


def _components(mask: np.ndarray) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _rotate_clockwise(sprite: np.ndarray, quarter_turns: int) -> np.ndarray:
    turns = int(quarter_turns) % 4
    return np.rot90(sprite, -turns)


def _shape_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, arr.shape)), arr.astype(np.uint8).tobytes()


def _all_rotations_unique(sprite: np.ndarray) -> bool:
    sigs = {_shape_signature(_rotate_clockwise(sprite, turns)) for turns in range(4)}
    return len(sigs) == 4


def _stamp_top_left(marker: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = marker
    h, w = shape
    return int(r - (h - 1) // 2), int(c - (w - 1) // 2)


def _centered_box(marker: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    top, left = _stamp_top_left(marker, shape)
    h, w = shape
    return top, left, top + h - 1, left + w - 1


def _boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    *,
    pad: int = 0,
) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _box_in_bounds(box: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    H, W = shape
    return 0 <= box[0] <= box[2] < H and 0 <= box[1] <= box[3] < W


def _parse_input(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}), "unexpected color present"

    blue_components = _components(g == BLUE)
    assert len(blue_components) == 1, "expected exactly one blue sprite"
    sprite_coords = blue_components[0]
    sprite_box = _bbox(g == BLUE)
    assert sprite_box is not None
    sprite = (_crop_bbox(g, sprite_box) == BLUE).astype(int)

    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    for (r1, c1), (r2, c2) in combinations(markers, 2):
        assert max(abs(r1 - r2), abs(c1 - c2)) >= 2, "markers must be 8-neighborhood separated"

    return {
        "grid": g,
        "sprite": sprite,
        "sprite_box": sprite_box,
        "sprite_coords": [tuple(map(int, rc)) for rc in sprite_coords],
        "markers": markers,
    }


def _transform(grid: np.ndarray, quarter_turns: int | None = None) -> np.ndarray:
    parsed = _parse_input(grid)
    g = parsed["grid"]
    sprite = parsed["sprite"]
    markers = parsed["markers"]
    turns = len(markers) % 4 if quarter_turns is None else int(quarter_turns) % 4
    rotated = _rotate_clockwise(sprite, turns)

    out = g.copy()
    for marker in markers:
        top, left = _stamp_top_left(marker, rotated.shape)
        _paste(out, rotated, top, left, transparent=BLACK, clip=True)

    for r, c in parsed["sprite_coords"]:
        out[r, c] = BLACK
    for r, c in markers:
        out[r, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the cropped blue sprite by the red-marker count and stamp it at every red marker.

    Input structure:
    - The grid uses black background, one compact blue 4-connected sprite, and several isolated red single-pixel markers.
    - The blue sprite is the only multi-pixel blue object and is separated from the red markers.
    - Marker positions are interpreted in the grid's native (row, col) coordinate system.

    Rule:
    - Crop the blue sprite to its minimal bounding box and count the red markers as K.
    - Rotate the cropped sprite clockwise by (K mod 4) * 90 degrees.
    - For each red marker, align that marker to the rotated sprite bbox center, using the upper-left cell of the central 2x2 block when a bbox dimension is even, and stamp the blue foreground there.
    - Erase the original blue sprite cells to black and keep all red markers unchanged.

    Key insight:
    - The red markers have dual meaning: their count selects the rotation state, while their positions select where the rotated sprite is stamped.
    """
    return _transform(grid)


SPRITE_LIBRARY = [
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 1]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
]

for _template in SPRITE_LIBRARY:
    assert 3 <= int(_template.sum()) <= 6
    assert len(_components(_template.astype(bool))) == 1
    assert _all_rotations_unique(_template)


GRID_MIN = 11
GRID_MAX = 18
MARKER_COUNT_BY_RESIDUE = {
    0: [4],
    1: [1, 5],
    2: [2, 6],
    3: [3, 7],
}
GENERATION_ATTEMPTS = 500


def _random_marker_count() -> int:
    residue = random.randrange(4)
    return random.choice(MARKER_COUNT_BY_RESIDUE[residue])


def _choose_grid_shape(marker_count: int) -> tuple[int, int]:
    min_side = GRID_MIN
    if marker_count >= 6:
        min_side = 15
    elif marker_count >= 4:
        min_side = 13
    return (
        random.randint(min_side, GRID_MAX),
        random.randint(min_side, GRID_MAX),
    )


def _choose_markers(
    grid_shape: tuple[int, int],
    stamp_shape: tuple[int, int],
    count: int,
) -> list[tuple[int, int]] | None:
    H, W = grid_shape
    candidates = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if _box_in_bounds(_centered_box((r, c), stamp_shape), grid_shape)
    ]
    random.shuffle(candidates)
    markers: list[tuple[int, int]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for marker in candidates:
        box = _centered_box(marker, stamp_shape)
        if any(_boxes_overlap(box, other, pad=1) for other in boxes):
            continue
        if any(max(abs(marker[0] - r), abs(marker[1] - c)) < 2 for r, c in markers):
            continue
        markers.append(marker)
        boxes.append(box)
        if len(markers) == count:
            return markers
    return None


def _choose_sprite_box(
    grid_shape: tuple[int, int],
    sprite_shape: tuple[int, int],
    forbidden: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    H, W = grid_shape
    h, w = sprite_shape
    candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        box = (top, left, top + h - 1, left + w - 1)
        if any(_boxes_overlap(box, other, pad=1) for other in forbidden):
            continue
        return top, left
    return None


def _build_input_example() -> np.ndarray | None:
    marker_count = _random_marker_count()
    base_sprite = random.choice(SPRITE_LIBRARY)
    sprite = _rotate_clockwise(base_sprite, random.randrange(4))
    rotated = _rotate_clockwise(sprite, marker_count % 4)

    for _ in range(60):
        grid_shape = _choose_grid_shape(marker_count)
        markers = _choose_markers(grid_shape, rotated.shape, marker_count)
        if markers is None:
            continue
        stamp_boxes = [_centered_box(marker, rotated.shape) for marker in markers]
        sprite_top_left = _choose_sprite_box(grid_shape, sprite.shape, stamp_boxes)
        if sprite_top_left is None:
            continue

        H, W = grid_shape
        grid = np.zeros((H, W), dtype=int)
        top, left = sprite_top_left
        _paste(grid, sprite, top, left, transparent=BLACK)
        for r, c in markers:
            grid[r, c] = RED
        if _validate_generated_example(grid):
            return grid
    return None


def _validate_generated_example(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_input(grid)
        sprite = parsed["sprite"]
        sprite_box = parsed["sprite_box"]
        markers = parsed["markers"]
        assert 1 <= len(markers) <= 7
        assert 3 <= int(sprite.sum()) <= 6
        assert sprite.shape in {(2, 3), (3, 2), (3, 3)}
        assert len(_components(sprite.astype(bool))) == 1
        assert _all_rotations_unique(sprite)

        for marker in markers:
            assert not (
                sprite_box[0] <= marker[0] <= sprite_box[2]
                and sprite_box[1] <= marker[1] <= sprite_box[3]
            )

        rotated = _rotate_clockwise(sprite, len(markers) % 4)
        stamp_boxes = [_centered_box(marker, rotated.shape) for marker in markers]
        for box in stamp_boxes:
            assert _box_in_bounds(box, grid.shape)
            assert not _boxes_overlap(box, sprite_box)
        for a, b in combinations(stamp_boxes, 2):
            assert not _boxes_overlap(a, b)

        out = main(grid)
        assert out.shape == grid.shape
        assert not np.array_equal(out, grid)

        expected = _transform(grid)
        assert np.array_equal(out, expected)

        for turns in range(4):
            if turns == len(markers) % 4:
                continue
            assert not np.array_equal(out, _transform(grid, quarter_turns=turns))

        for marker in markers:
            modified = grid.copy()
            modified[marker] = BLACK
            assert not np.array_equal(out, main(modified))

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_input_example()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
