from __future__ import annotations

import random

import numpy as np


FRAME_SIZE = 5
INNER_SIZE = 3
BG = 0
DIRECTIONS = ("up", "right", "down", "left")
ROTATIONS_CCW = {"up": 0, "right": 3, "down": 2, "left": 1}
COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400

BASE_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _tight_mask(coords: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return (r0, c0, r1, c1), mask


def _frame_mask(direction: str) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    if direction != "up":
        mask[0, :] = True
    if direction != "down":
        mask[-1, :] = True
    if direction != "left":
        mask[:, 0] = True
    if direction != "right":
        mask[:, -1] = True
    return mask


def _detect_frame_direction(mask: np.ndarray) -> str | None:
    if mask.shape != (FRAME_SIZE, FRAME_SIZE):
        return None
    if bool(mask[1:-1, 1:-1].any()):
        return None
    area = int(mask.sum())
    if area < 11 or area > 13:
        return None

    counts = [
        int(mask[0, :].sum()),
        int(mask[:, -1].sum()),
        int(mask[-1, :].sum()),
        int(mask[:, 0].sum()),
    ]
    min_count = min(counts)
    if min_count > 2 or counts.count(min_count) != 1:
        return None
    for idx, count in enumerate(counts):
        if idx != counts.index(min_count) and count < 4:
            return None
    return DIRECTIONS[counts.index(min_count)]


def _rotated_mask(mask: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(mask, ROTATIONS_CCW[direction])


def _target_top_left(frame_bbox: tuple[int, int, int, int], piece_shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    ph, pw = piece_shape
    return r0 + 1 + (INNER_SIZE - ph) // 2, c0 + 1 + (INNER_SIZE - pw) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each monochrome fragment into the 5x5 same-color open frame.

    The frame opening encodes the rotation:
    up -> 0 degrees, right -> 90 degrees clockwise,
    down -> 180 degrees, left -> 270 degrees clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    out = g.copy()

    for color in sorted(int(x) for x in np.unique(g) if int(x) != bg):
        comps = _components(g == color)
        assert len(comps) == 2, "each color must have one frame and one fragment"

        frame_bbox = None
        frame_dir = None
        piece_coords = None
        piece_mask = None

        for coords in comps:
            bbox, mask = _tight_mask(coords)
            direction = _detect_frame_direction(mask)
            if direction is not None:
                assert frame_bbox is None, "frame color must be unique"
                frame_bbox = bbox
                frame_dir = direction
            else:
                assert piece_coords is None, "fragment color must be unique"
                piece_coords = coords
                piece_mask = mask

        assert frame_bbox is not None and frame_dir is not None
        assert piece_coords is not None and piece_mask is not None

        out[piece_coords[:, 0], piece_coords[:, 1]] = bg
        rotated = _rotated_mask(piece_mask, frame_dir)
        rh, rw = rotated.shape
        assert rh <= INNER_SIZE and rw <= INNER_SIZE
        tr, tc = _target_top_left(frame_bbox, (rh, rw))
        patch = out[tr : tr + rh, tc : tc + rw]
        patch[rotated] = color

    return out


def _box_is_free(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_box(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _draw_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    h, w = mask.shape
    grid[top : top + h, left : left + w][mask] = color


def _sample_position(
    occupied: np.ndarray, height: int, width: int, border: int = 1, tries: int = 200
) -> tuple[int, int] | None:
    h, w = occupied.shape
    if h < height + 2 * border or w < width + 2 * border:
        return None
    for _ in range(tries):
        top = random.randint(border, h - height - border)
        left = random.randint(border, w - width - border)
        if _box_is_free(occupied, top, left, height, width):
            return top, left
    return None


def _render_example(
    grid_shape: tuple[int, int],
    colors: list[int],
    directions: list[str],
    shape_ids: list[int],
    frame_positions: list[tuple[int, int]],
    piece_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(grid_shape, BG, dtype=int)
    expected = np.full(grid_shape, BG, dtype=int)

    for idx, color in enumerate(colors):
        direction = directions[idx]
        frame_top, frame_left = frame_positions[idx]
        piece_top, piece_left = piece_positions[idx]
        base_shape = BASE_SHAPES[shape_ids[idx]]
        rotated_shape = _rotated_mask(base_shape, direction)

        frame = _frame_mask(direction)
        _draw_mask(inp, frame_top, frame_left, frame, color)
        _draw_mask(expected, frame_top, frame_left, frame, color)
        _draw_mask(inp, piece_top, piece_left, base_shape, color)

        target_top, target_left = _target_top_left(
            (frame_top, frame_left, frame_top + FRAME_SIZE - 1, frame_left + FRAME_SIZE - 1),
            rotated_shape.shape,
        )
        _draw_mask(expected, target_top, target_left, rotated_shape, color)

    return inp, expected


def _generate_once() -> np.ndarray | None:
    count = random.randint(2, 4)
    height = random.randint(16 + count, 24 + count)
    width = random.randint(16 + count, 24 + count)
    occupied = np.zeros((height, width), dtype=bool)

    colors = random.sample(COLORS, count)
    directions = random.sample(list(DIRECTIONS), count)
    if count <= len(BASE_SHAPES):
        shape_ids = random.sample(list(range(len(BASE_SHAPES))), count)
    else:
        shape_ids = [random.randrange(len(BASE_SHAPES)) for _ in range(count)]

    frame_positions: list[tuple[int, int]] = []
    piece_positions: list[tuple[int, int]] = []

    for _ in range(count):
        pos = _sample_position(occupied, FRAME_SIZE, FRAME_SIZE)
        if pos is None:
            return None
        frame_positions.append(pos)
        _mark_box(occupied, pos[0], pos[1], FRAME_SIZE, FRAME_SIZE)

    for shape_id in shape_ids:
        shape = BASE_SHAPES[shape_id]
        pos = _sample_position(occupied, int(shape.shape[0]), int(shape.shape[1]))
        if pos is None:
            return None
        piece_positions.append(pos)
        _mark_box(occupied, pos[0], pos[1], int(shape.shape[0]), int(shape.shape[1]))

    inp, expected = _render_example(
        (height, width),
        colors,
        directions,
        shape_ids,
        frame_positions,
        piece_positions,
    )

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        example = _generate_once()
        if example is not None:
            return example
    raise RuntimeError("failed to generate a valid ARC example")
