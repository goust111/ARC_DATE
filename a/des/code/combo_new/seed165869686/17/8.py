import random
from collections import deque

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 400
WORK_SIZE = 30
MAX_INNER_SPAN = 11
MIN_OBJECTS = 1
MAX_OBJECTS = 3
PLACEMENT_PADDING = 1

PATTERN_LIBRARY = {
    (2, 2): [
        ((1, 1), (1, 0)),
        ((1, 0), (1, 1)),
    ],
    (2, 3): [
        ((1, 1, 0), (1, 0, 1)),
        ((1, 0, 1), (1, 1, 0)),
        ((1, 1, 1), (0, 1, 0)),
    ],
    (3, 2): [
        ((1, 1), (0, 1), (1, 0)),
        ((1, 0), (1, 1), (0, 1)),
        ((1, 1), (1, 0), (1, 1)),
    ],
    (3, 3): [
        ((1, 1, 0), (1, 0, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 1, 0), (0, 1, 1)),
        ((1, 1, 1), (1, 0, 0), (1, 1, 0)),
    ],
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_of_color(
    grid: np.ndarray, start_r: int, start_c: int, visited: np.ndarray
) -> np.ndarray:
    color = int(grid[start_r, start_c])
    q = deque([(start_r, start_c)])
    visited[start_r, start_c] = True
    coords = []
    height, width = grid.shape
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if visited[nr, nc] or int(grid[nr, nc]) != color:
                continue
            visited[nr, nc] = True
            q.append((nr, nc))
    return np.asarray(coords, dtype=int)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False
    rows = coords[:, 0]
    cols = coords[:, 1]
    on_border = (rows == r0) | (rows == r1) | (cols == c0) | (cols == c1)
    return bool(np.all(on_border))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    visited = np.zeros(grid.shape, dtype=bool)
    frames = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            coords = _component_of_color(grid, r, c, visited)
            if _is_rectangular_frame(coords):
                frames.append(_bbox(coords))
    frames.sort()
    return frames


def _extract_stamp_patch(interior: np.ndarray, bg: int) -> np.ndarray:
    mask = interior != bg
    rows, cols = np.where(mask)
    assert rows.size > 0, "each frame must contain a stamp"
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())
    assert (r0, c0) == (0, 0), "stamp bbox must start at the inner top-left corner"
    patch = interior[r0 : r1 + 1, c0 : c1 + 1].copy()
    colors = np.unique(patch[patch != bg])
    assert colors.size == 1, "stamp must use exactly one color"
    return patch


def _paste_transparent_clipped(
    canvas: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int
) -> None:
    ph, pw = patch.shape
    height, width = canvas.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(height, top + ph)
    c1 = min(width, left + pw)
    if r0 >= r1 or c0 >= c1:
        return
    pr0 = r0 - top
    pc0 = c0 - left
    window = patch[pr0 : pr0 + (r1 - r0), pc0 : pc0 + (c1 - c0)]
    mask = window != transparent
    target = canvas[r0:r1, c0:c1]
    target[mask] = window[mask]


def _roll_stamp_along_frame(
    canvas: np.ndarray, frame_bbox: tuple[int, int, int, int], patch: np.ndarray, bg: int
) -> None:
    r0, c0, r1, c1 = frame_bbox
    inner_top = r0 + 1
    inner_left = c0 + 1
    inner_height = r1 - r0 - 1
    inner_width = c1 - c0 - 1

    top_patch = patch
    right_patch = np.rot90(patch, -1)
    bottom_patch = np.rot90(patch, 2)
    left_patch = np.rot90(patch, 1)

    for offset in range(0, inner_width, top_patch.shape[1]):
        _paste_transparent_clipped(
            canvas, top_patch, inner_top, inner_left + offset, bg
        )

    right_left = inner_left + inner_width - right_patch.shape[1]
    for offset in range(0, inner_height, right_patch.shape[0]):
        _paste_transparent_clipped(
            canvas, right_patch, inner_top + offset, right_left, bg
        )

    bottom_top = inner_top + inner_height - bottom_patch.shape[0]
    for offset in range(
        inner_width - bottom_patch.shape[1], -bottom_patch.shape[1], -bottom_patch.shape[1]
    ):
        _paste_transparent_clipped(
            canvas, bottom_patch, bottom_top, inner_left + offset, bg
        )

    for offset in range(
        inner_height - left_patch.shape[0], -left_patch.shape[0], -left_patch.shape[0]
    ):
        _paste_transparent_clipped(
            canvas, left_patch, inner_top + offset, inner_left, bg
        )


def main(grid: np.ndarray) -> np.ndarray:
    """Replicate each inner stamp clockwise along the inner walls of its rectangular frame.

    Input structure:
    - The background is black.
    - Several disjoint 1-pixel-thick monochrome rectangular outlines appear on the canvas.
    - Each outline encloses one monochrome stamp patch whose bbox is aligned to the frame's inner top-left corner.

    Rule:
    - Detect every rectangular outline and treat the non-background patch in its inner top-left corner as the seed stamp.
    - Keep the outline unchanged, then repeat the stamp left-to-right along the top inner wall.
    - Continue clockwise along the right, bottom, and left inner walls, rotating the stamp by 90, 180, and 270 degrees clockwise and clipping any copy that overruns the remaining edge segment.

    Key insight:
    - The interior mark is not an isolated decoration: it is a reusable stamp whose bbox is anchored at the inner top-left corner and then rolled clockwise around the rectangle with quarter-turn rotations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for frame_bbox in frames:
        r0, c0, r1, c1 = frame_bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        patch = _extract_stamp_patch(interior, bg)
        _roll_stamp_along_frame(out, frame_bbox, patch, bg)
    return out


def _sample_inner_span(
    rng: random.Random, step: int, minimum: int, truncated: bool
) -> int:
    assert step >= 2
    residues = [0] if not truncated else list(range(1, step))
    candidates = []
    for residue in residues:
        min_repeat = max(1, (minimum - residue + step - 1) // step)
        for repeat in range(min_repeat, min_repeat + 3):
            span = repeat * step + residue
            if minimum <= span <= MAX_INNER_SPAN:
                candidates.append(span)
    assert candidates, "no legal inner span candidates"
    return int(rng.choice(candidates))


def _choose_pattern(rng: random.Random) -> np.ndarray:
    size = rng.choice(sorted(PATTERN_LIBRARY))
    raw = rng.choice(PATTERN_LIBRARY[size])
    return np.asarray(raw, dtype=int)


def _build_object(
    rng: random.Random, frame_color: int, stamp_color: int
) -> tuple[np.ndarray, np.ndarray]:
    pattern = _choose_pattern(rng)
    ph, pw = pattern.shape
    min_span = max(ph, pw)
    width_truncated = bool(rng.randrange(2))
    height_truncated = bool(rng.randrange(2))
    inner_width = _sample_inner_span(rng, pw, min_span, width_truncated)
    inner_height = _sample_inner_span(rng, pw, min_span, height_truncated)

    obj = np.full((inner_height + 2, inner_width + 2), BACKGROUND, dtype=int)
    obj[0, :] = frame_color
    obj[-1, :] = frame_color
    obj[:, 0] = frame_color
    obj[:, -1] = frame_color

    stamp_window = obj[1 : 1 + ph, 1 : 1 + pw]
    stamp_window[pattern.astype(bool)] = stamp_color

    expected = obj.copy()
    _roll_stamp_along_frame(
        expected, (0, 0, expected.shape[0] - 1, expected.shape[1] - 1), stamp_window.copy(), BACKGROUND
    )
    return obj, expected


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(occupied.shape[0], top + height + PLACEMENT_PADDING)
    c1 = min(occupied.shape[1], left + width + PLACEMENT_PADDING)
    return not bool(occupied[r0:r1, c0:c1].any())


def _place_objects(
    rng: random.Random, objects: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    input_canvas = np.full((WORK_SIZE, WORK_SIZE), BACKGROUND, dtype=int)
    output_canvas = np.full((WORK_SIZE, WORK_SIZE), BACKGROUND, dtype=int)
    occupied = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)

    order = sorted(
        range(len(objects)),
        key=lambda idx: objects[idx][0].shape[0] * objects[idx][0].shape[1],
        reverse=True,
    )
    for idx in order:
        inp_obj, out_obj = objects[idx]
        height, width = inp_obj.shape
        candidates = []
        for top in range(WORK_SIZE - height + 1):
            for left in range(WORK_SIZE - width + 1):
                if _can_place(occupied, top, left, height, width):
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = rng.choice(candidates)
        input_canvas[top : top + height, left : left + width] = inp_obj
        output_canvas[top : top + height, left : left + width] = out_obj
        occupied[top : top + height, left : left + width] = True

    used = input_canvas != BACKGROUND
    rows, cols = np.where(used)
    if rows.size == 0:
        return None
    r0, c0 = int(rows.min()), int(cols.min())
    r1, c1 = int(rows.max()), int(cols.max())

    top_margin = rng.randint(1, min(3, r0)) if r0 > 0 else 0
    left_margin = rng.randint(1, min(3, c0)) if c0 > 0 else 0
    bottom_margin = (
        rng.randint(1, min(3, WORK_SIZE - 1 - r1)) if r1 < WORK_SIZE - 1 else 0
    )
    right_margin = (
        rng.randint(1, min(3, WORK_SIZE - 1 - c1)) if c1 < WORK_SIZE - 1 else 0
    )

    crop_r0 = r0 - top_margin
    crop_c0 = c0 - left_margin
    crop_r1 = r1 + 1 + bottom_margin
    crop_c1 = c1 + 1 + right_margin
    return (
        input_canvas[crop_r0:crop_r1, crop_c0:crop_c1].copy(),
        output_canvas[crop_r0:crop_r1, crop_c0:crop_c1].copy(),
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(MIN_OBJECTS, MAX_OBJECTS)
    palette = rng.sample(list(range(1, 10)), 2 * n_objects)

    objects = []
    for i in range(n_objects):
        frame_color = palette[2 * i]
        stamp_color = palette[2 * i + 1]
        inp_obj, out_obj = _build_object(rng, frame_color, stamp_color)
        objects.append((inp_obj, out_obj))

    placed = _place_objects(rng, objects)
    if placed is None:
        return None
    inp, expected = placed
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return None
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
