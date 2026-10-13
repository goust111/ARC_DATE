import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 30
FRAME_COUNT_CHOICES = (2, 3, 4, 5)
MIN_GRID_BY_COUNT = {2: 16, 3: 18, 4: 22, 5: 24}

MIN_FRAME_SIDE = 5
MAX_FRAME_SIDE_BY_COUNT = {2: 10, 3: 9, 4: 8, 5: 7}
FRAME_GAP = 2
FRAME_PLACE_TRIES = 120
MAX_GENERATION_ATTEMPTS = 300

FRAGMENT_PATTERNS = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
)


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))

    return components


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return not (a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1])


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False

    patch = np.zeros((h, w), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(patch, expected))


def _frame_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: list[dict] = []
    covered = np.zeros_like(g, dtype=bool)
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        for coords in _find_components(g == color):
            if not _is_rect_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            inner_colors = [int(v) for v in np.unique(interior) if int(v) != BG]
            assert len(inner_colors) == 1, "each frame interior must contain exactly one fragment color"
            inner_color = inner_colors[0]
            assert inner_color != color, "fragment color must differ from the frame color"
            fragment_mask = interior == inner_color
            assert bool(fragment_mask.any()), "frame interior must contain at least one fragment pixel"
            frame = {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "border_color": color,
                "inner_color": inner_color,
            }
            frames.append(frame)
            covered[coords[:, 0], coords[:, 1]] = True
            covered[r0 + 1 : r1, c0 + 1 : c1] |= fragment_mask

    assert len(frames) >= 2, "expected at least two frames"

    border_colors = [int(frame["border_color"]) for frame in frames]
    inner_colors = [int(frame["inner_color"]) for frame in frames]
    assert len(set(border_colors)) == len(frames), "frame colors must be unique"
    assert len(set(inner_colors)) == len(frames), "fragment colors must be unique"
    assert set(inner_colors) == set(border_colors), "fragment colors must come from the frame palette"

    frames.sort(key=lambda frame: frame["bbox"])
    for i, frame in enumerate(frames):
        for other in frames[i + 1 :]:
            assert not _boxes_overlap(frame["bbox"], other["bbox"]), "frames must be non-overlapping"

    assert not np.any((g != BG) & ~covered), "all non-background cells must be frames or fragments"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each rectangular frame border with the unique fragment color scattered inside it.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping one-pixel-thick rectangular
      frames, each in a distinct color.
    - Inside every frame there are one or more fragment pixels of exactly one
      other frame color, mixed with black background.

    Rule:
    - For each frame, read the unique non-black color that appears inside it.
    - Recolor the entire border of that frame to the interior fragment color.
    - Recolor every interior fragment pixel back to the frame's original border
      color, while leaving black interior cells unchanged.

    Key insight:
    The fragment color inside a frame names which other frame color should move
    onto the border, so the border and the interior fragments simply exchange
    their colors.
    """
    g = np.asarray(grid, dtype=int)
    frames = _frame_records(g)

    out = g.copy()
    for frame in frames:
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = int(frame["inner_color"])

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        fragment_mask = g[r0 + 1 : r1, c0 + 1 : c1] == int(frame["inner_color"])
        out[r0 + 1 : r1, c0 + 1 : c1][fragment_mask] = int(frame["border_color"])

    return out


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_derangement(colors: list[int]) -> dict[int, int]:
    for _ in range(100):
        shuffled = colors[:]
        random.shuffle(shuffled)
        if all(src != dst for src, dst in zip(colors, shuffled)):
            return {src: dst for src, dst in zip(colors, shuffled)}

    shift = random.randint(1, len(colors) - 1)
    return {colors[i]: colors[(i + shift) % len(colors)] for i in range(len(colors))}


def _sample_frame_dims(frame_count: int) -> tuple[int, int]:
    max_side = MAX_FRAME_SIDE_BY_COUNT[frame_count]
    h = random.randint(MIN_FRAME_SIDE, max_side)
    w = random.randint(MIN_FRAME_SIDE, max_side)
    return h, w


def _place_boxes(h: int, w: int, frame_count: int) -> list[tuple[int, int, int, int]] | None:
    dims = [_sample_frame_dims(frame_count) for _ in range(frame_count)]
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    for box_h, box_w in dims:
        placed = False
        for _ in range(FRAME_PLACE_TRIES):
            r0 = random.randint(0, h - box_h)
            c0 = random.randint(0, w - box_w)
            bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if any(_boxes_too_close(bbox, other, FRAME_GAP) for other in boxes):
                continue
            boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _choose_point_fragments(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    random.shuffle(candidates)
    target = random.randint(1, min(5, max(1, (inner_h * inner_w + 2) // 3)))

    cells: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in cells):
            continue
        cells.append((r, c))
        if len(cells) == target:
            break

    return cells or None


def _choose_shape_fragments(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    candidates = [pattern for pattern in FRAGMENT_PATTERNS if pattern.shape[0] <= inner_h and pattern.shape[1] <= inner_w]
    if not candidates:
        return None

    pattern = random.choice(candidates)
    ph, pw = pattern.shape
    off_r = random.randint(0, inner_h - ph)
    off_c = random.randint(0, inner_w - pw)
    cells = [(off_r + int(r), off_c + int(c)) for r, c in np.argwhere(pattern)]

    extras = [
        (r, c)
        for r in range(inner_h)
        for c in range(inner_w)
        if (r, c) not in cells and all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in cells)
    ]
    if extras and random.random() < 0.5:
        cells.append(random.choice(extras))

    return cells


def _choose_fragment_cells(inner_h: int, inner_w: int) -> list[tuple[int, int]] | None:
    if random.random() < 0.5:
        cells = _choose_point_fragments(inner_h, inner_w)
        if cells is not None:
            return cells

    cells = _choose_shape_fragments(inner_h, inner_w)
    if cells is not None:
        return cells
    return _choose_point_fragments(inner_h, inner_w)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    min_dim = MIN_GRID_BY_COUNT[frame_count]
    h = random.randint(min_dim, MAX_GRID_SIZE)
    w = random.randint(min_dim, MAX_GRID_SIZE)

    boxes = _place_boxes(h, w, frame_count)
    if boxes is None:
        return None

    border_colors = random.sample(PALETTE, frame_count)
    swap_map = _random_derangement(border_colors)
    grid = np.zeros((h, w), dtype=int)

    for bbox, color in zip(boxes, border_colors):
        _paint_frame(grid, bbox, color)

    for bbox, border_color in zip(boxes, border_colors):
        r0, c0, r1, c1 = bbox
        fragment_color = swap_map[border_color]
        fragment_cells = _choose_fragment_cells(r1 - r0 - 1, c1 - c0 - 1)
        if not fragment_cells:
            return None
        for rr, cc in fragment_cells:
            grid[r0 + 1 + rr, c0 + 1 + cc] = fragment_color

    try:
        frames = _frame_records(grid)
        if len(frames) != frame_count:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
