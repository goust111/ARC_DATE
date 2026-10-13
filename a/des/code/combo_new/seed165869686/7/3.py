from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
PATTERN_COLORS = [1, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 120

DIR_TO_ROT_K = {
    "up": 0,
    "right": 3,  # 90 degrees clockwise
    "down": 2,
    "left": 1,   # 90 degrees counterclockwise
}

# The 3-pixel arrow is a 2x2 corner block with one missing cell.
MISSING_TO_DIR = {
    (0, 0): "up",
    (0, 1): "right",
    (1, 1): "down",
    (1, 0): "left",
}
DIR_TO_MISSING = {direction: missing for missing, direction in MISSING_TO_DIR.items()}
CORNER_NAMES = ("tl", "tr", "bl", "br")


def _most_common(arr: np.ndarray) -> int:
    counts = Counter(int(v) for v in arr.ravel())
    assert counts
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    assert indices
    runs: list[tuple[int, int]] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev + 1))
        start = idx
        prev = idx
    runs.append((start, prev + 1))
    return runs


def _frame_spans(grid: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert sep_cols, "expected full-height grey separator columns"
    sep_runs = _runs(sep_cols)
    assert 2 <= len(sep_runs) <= 3, "expected 3 or 4 frames"

    spans: list[tuple[int, int]] = []
    left = 0
    for start, end in sep_runs:
        spans.append((left, start))
        left = end
    spans.append((left, grid.shape[1]))

    widths = {end - start for start, end in spans}
    assert len(widths) == 1, "all frames must have the same width"
    frame_size = next(iter(widths))
    assert frame_size > 0
    return spans, int(frame_size)


def _corner_top_left(size: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, size - 2
    if corner == "bl":
        return size - 2, 0
    if corner == "br":
        return size - 2, size - 2
    raise AssertionError(f"unknown corner {corner}")


def _find_arrow_direction(frame: np.ndarray) -> str:
    assert frame.shape[0] == frame.shape[1], "each frame must be square"
    red_coords = np.argwhere(frame == RED)
    assert red_coords.shape[0] == 3, "each frame must contain exactly one 3-pixel red arrow"

    candidates: list[str] = []
    size = frame.shape[0]
    all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    for corner in CORNER_NAMES:
        r0, c0 = _corner_top_left(size, corner)
        box = frame[r0 : r0 + 2, c0 : c0 + 2]
        if box.shape != (2, 2):
            continue
        occupied = {(int(r), int(c)) for r, c in np.argwhere(box == RED)}
        if len(occupied) != 3:
            continue
        missing = all_cells - occupied
        if len(missing) != 1:
            continue
        candidates.append(MISSING_TO_DIR[next(iter(missing))])

    assert len(candidates) == 1, "arrow must occupy exactly one frame corner"
    return candidates[0]


def _pattern_layer(frame: np.ndarray, bg: int) -> np.ndarray:
    layer = frame.copy()
    layer[layer == RED] = bg
    assert not np.any(layer == GREY), "grey is reserved for separators"
    assert np.any(layer != bg), "each frame must contain a non-arrow pattern"
    return layer


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _random_connected_layer(size: int, color: int) -> np.ndarray | None:
    usable = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    core = [
        (r, c)
        for r, c in usable
        if abs(r - (size - 1) / 2) <= 1.25 and abs(c - (size - 1) / 2) <= 1.25
    ]
    if len(usable) < 4 or not core:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(4, min(8, len(usable)))
        mask = np.zeros((size, size), dtype=bool)
        start = random.choice(core)
        mask[start] = True
        frontier = {
            cell
            for cell in _neighbors4(start[0], start[1], size, size)
            if 1 <= cell[0] < size - 1 and 1 <= cell[1] < size - 1
        }

        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nxt in _neighbors4(cell[0], cell[1], size, size):
                if 1 <= nxt[0] < size - 1 and 1 <= nxt[1] < size - 1 and not mask[nxt]:
                    frontier.add(nxt)

        if int(mask.sum()) != target_area:
            continue

        box_h, box_w = _bbox(mask)
        if box_h < 2 or box_w < 2:
            continue

        layer = np.full((size, size), BLACK, dtype=int)
        layer[mask] = color
        if any(np.array_equal(layer, np.rot90(layer, k)) for k in (1, 2, 3)):
            continue
        return layer

    return None


def _inverse_rotate(layer: np.ndarray, direction: str) -> np.ndarray:
    return np.rot90(layer, (-DIR_TO_ROT_K[direction]) % 4)


def _corner_box_is_empty(layer: np.ndarray, corner: str) -> bool:
    r0, c0 = _corner_top_left(layer.shape[0], corner)
    return bool(np.all(layer[r0 : r0 + 2, c0 : c0 + 2] == BLACK))


def _paint_arrow(frame: np.ndarray, corner: str, direction: str) -> None:
    r0, c0 = _corner_top_left(frame.shape[0], corner)
    missing = DIR_TO_MISSING[direction]
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) != missing:
                frame[r0 + dr, c0 + dc] = RED


def _compose_layers(layers: list[np.ndarray]) -> tuple[np.ndarray, int, list[int]]:
    out = np.full_like(layers[0], BLACK)
    occupied = np.zeros(layers[0].shape, dtype=bool)
    overlap = 0
    for layer in layers:
        mask = layer != BLACK
        overlap += int(np.count_nonzero(occupied & mask))
        out[mask] = layer[mask]
        occupied |= mask

    visible: list[int] = []
    for layer in layers:
        mask = layer != BLACK
        visible.append(int(np.count_nonzero(mask & (out == layer))))
    return out, overlap, visible


def _assemble_frames(frames: list[np.ndarray]) -> np.ndarray:
    size = frames[0].shape[0]
    width = len(frames) * size + (len(frames) - 1)
    grid = np.full((size, width), BLACK, dtype=int)
    for idx, frame in enumerate(frames):
        c0 = idx * (size + 1)
        grid[:, c0 : c0 + size] = frame
        if idx + 1 < len(frames):
            grid[:, c0 + size] = GREY
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each frame-sized pattern by its corner arrow and overlay the rotated results onto one canvas.

    Input structure:
    - The background is black, and full-height grey separator columns split the grid
      into 3 or 4 equal square frames arranged left to right.
    - Each frame contains exactly one small colored pattern on black background.
    - Each frame also contains one red 3-pixel L marker occupying a 2x2 corner box.
    - The marker is a 2x2 block with one missing cell; that missing corner encodes
      the opening direction in the order up, right, down, left.

    Rule:
    - Split the input into equal frames between the grey separator columns.
    - In each frame, read the red corner marker, then erase that marker from the frame.
    - Rotate the remaining frame-sized pattern layer by the marker instruction:
      up = unchanged, right = 90 degrees clockwise, down = 180 degrees,
      left = 90 degrees counterclockwise.
    - Create a single blank canvas the size of one frame and overlay the rotated
      layers from left to right, letting later frames overwrite earlier overlaps.

    Key insight:
    - The red corner glyph is not part of the picture; it is a per-frame rotation
      code, and the final output is the ordered overlay of the rotated frame layers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common(g)
    spans, frame_size = _frame_spans(g)
    assert 3 <= len(spans) <= 4
    assert g.shape[0] == frame_size, "frames must be square so 90-degree rotations stay in-bounds"

    out = np.full((frame_size, frame_size), bg, dtype=int)
    for start, end in spans:
        frame = g[:, start:end]
        assert frame.shape == (frame_size, frame_size)
        direction = _find_arrow_direction(frame)
        layer = _pattern_layer(frame, bg)
        rotated = np.rot90(layer, DIR_TO_ROT_K[direction])
        mask = rotated != bg
        out[mask] = rotated[mask]
    return out


def _sample_directions(frame_count: int) -> list[str]:
    directions = list(DIR_TO_ROT_K)
    if frame_count == 4:
        random.shuffle(directions)
        return directions
    chosen = random.sample(directions, frame_count)
    random.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((3, 4))
    frame_size = random.randint(7, 9) if frame_count == 3 else 6
    directions = _sample_directions(frame_count)
    colors = random.sample(PATTERN_COLORS, frame_count)
    corners = [random.choice(CORNER_NAMES) for _ in range(frame_count)]

    output_layers: list[np.ndarray] = []
    frames: list[np.ndarray] = []
    for color, direction, corner in zip(colors, directions, corners):
        built = False
        for _ in range(SHAPE_ATTEMPTS):
            layer = _random_connected_layer(frame_size, color)
            if layer is None:
                continue
            input_layer = _inverse_rotate(layer, direction)
            if not _corner_box_is_empty(input_layer, corner):
                continue
            frame = input_layer.copy()
            _paint_arrow(frame, corner, direction)
            output_layers.append(layer)
            frames.append(frame)
            built = True
            break
        if not built:
            return None

    expected, overlap, visible = _compose_layers(output_layers)
    if overlap == 0:
        return None
    if min(visible) < 2:
        return None

    inp = _assemble_frames(frames)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
