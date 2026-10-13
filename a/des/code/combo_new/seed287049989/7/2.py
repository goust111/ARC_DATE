from __future__ import annotations

import random
from collections import Counter

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            sprite = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                sprite[rr - r0, cc - c0] = True
            parts.append(sprite)

    return parts


def _mask_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    return set(map(tuple, np.argwhere(mask)))


def _rectangle_dims_from_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    cells_a = _mask_cells(mask_a)
    cells_b = tuple(map(tuple, np.argwhere(mask_b)))
    h1, w1 = map(int, mask_a.shape)
    h2, w2 = map(int, mask_b.shape)
    candidates: set[tuple[int, int]] = set()

    for dr in range(-h2, h1 + 1):
        for dc in range(-w2, w1 + 1):
            shifted_b = {(r + dr, c + dc) for r, c in cells_b}
            if cells_a & shifted_b:
                continue

            union = cells_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            rect_h = max(rows) - min(rows) + 1
            rect_w = max(cols) - min(cols) + 1
            if rect_h * rect_w == len(union):
                candidates.add((rect_h, rect_w))

    assert len(candidates) == 1, f"expected one assembled rectangle, got {sorted(candidates)}"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored scattered L-shaped fragments, reassemble each pair into a solid rectangle, and pack the rectangles by increasing area.

    Input structure:
    - The grid has an azure-like background and several foreground colors.
    - Each foreground color appears in exactly two disconnected L-shaped components.
    - Every such pair can be translated without rotation or overlap so that their union becomes one solid axis-aligned rectangle.

    Rule:
    - Group components by color and pair the two L-shapes of each color.
    - For each pair, search the relative translations that make the two shapes disjoint and whose union is a filled rectangle; record that rectangle's height, width, and area.
    - Sort the reconstructed rectangles by area from small to large.
    - Output a new background grid containing one solid rectangle per color, laid out left-to-right with one background column between neighbors and all rectangle bottoms aligned.

    Key insight:
    - The visible L-shapes are incomplete fragments of larger rectangles; the task is to infer each hidden rectangle by testing how the two same-colored pieces can interlock under translation, then ignore their original positions and repack the completed rectangles by area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    bg = _background_color(g)
    counts = Counter(map(int, g.reshape(-1)))
    assert counts[bg] == max(counts.values()), "background must be most common"

    rectangles: list[tuple[int, int, int, int]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        pieces = _components_of_color(g, color)
        assert len(pieces) == 2, f"color {color} must have exactly two components"
        rect_h, rect_w = _rectangle_dims_from_pair(pieces[0], pieces[1])
        rectangles.append((rect_h * rect_w, rect_h, rect_w, color))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    out_h = max(rect_h for _, rect_h, _, _ in rectangles)
    out_w = sum(rect_w for _, _, rect_w, _ in rectangles) + len(rectangles) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c0 = 0
    for _, rect_h, rect_w, color in rectangles:
        out[out_h - rect_h :, c0 : c0 + rect_w] = color
        c0 += rect_w + 1

    return out


# --------------- generator ---------------

BG = 8
ARC_MAX = 30
PAIR_COUNT_RANGE = (2, 4)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (4, 8)
INPUT_H_RANGE = (18, 30)
INPUT_W_RANGE = (18, 30)
GEN_RETRIES = 400
PAIR_RETRIES = 120
PLACEMENT_RETRIES = 80


def _apply_symmetry(mask: np.ndarray, op: int) -> np.ndarray:
    transformed = np.rot90(mask, op % 4)
    if op >= 4:
        transformed = np.fliplr(transformed)
    return transformed.copy()


def _make_pair_masks(rect_h: int, rect_w: int) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        split = random.randint(1, rect_h - 1)
        top_w, bottom_w = random.sample(range(1, rect_w), 2)
        part_a = np.zeros((rect_h, rect_w), dtype=bool)
        part_a[:split, :top_w] = True
        part_a[split:, :bottom_w] = True
    else:
        split = random.randint(1, rect_w - 1)
        left_h, right_h = random.sample(range(1, rect_h), 2)
        part_a = np.zeros((rect_h, rect_w), dtype=bool)
        part_a[:left_h, :split] = True
        part_a[:right_h, split:] = True

    part_b = ~part_a
    op = random.randrange(8)
    return _crop_mask(_apply_symmetry(part_a, op)), _crop_mask(_apply_symmetry(part_b, op))


def _sample_pair_specs() -> list[tuple[int, int, int, np.ndarray, np.ndarray]] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample([c for c in range(10) if c != BG], pair_count)
    specs: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    used_areas: set[int] = set()
    packed_w = 0
    packed_h = 0

    for color in colors:
        placed = False
        for _ in range(PAIR_RETRIES):
            rect_h = random.randint(*RECT_H_RANGE)
            rect_w = random.randint(*RECT_W_RANGE)
            piece_a, piece_b = _make_pair_masks(rect_h, rect_w)
            try:
                out_h, out_w = _rectangle_dims_from_pair(piece_a, piece_b)
            except AssertionError:
                continue

            area = out_h * out_w
            next_w = packed_w + out_w + (1 if specs else 0)
            next_h = max(packed_h, out_h)
            if area in used_areas or next_w > ARC_MAX or next_h > ARC_MAX:
                continue

            specs.append((color, out_h, out_w, piece_a, piece_b))
            used_areas.add(area)
            packed_w = next_w
            packed_h = next_h
            placed = True
            break
        if not placed:
            return None

    return specs


def _can_place_bbox(occupied: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = occupied.shape
    sh, sw = shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + sh + 1)
    c1 = min(w, left + sw + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_piece(
    grid: np.ndarray,
    occupied: np.ndarray,
    piece: np.ndarray,
    color: int,
) -> bool:
    h, w = grid.shape
    ph, pw = piece.shape
    choices: list[tuple[int, int]] = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            if _can_place_bbox(occupied, r0, c0, (ph, pw)):
                choices.append((r0, c0))
    if not choices:
        return False

    r0, c0 = random.choice(choices)
    grid[r0 : r0 + ph, c0 : c0 + pw][piece] = color
    occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_pair_specs()
    if specs is None:
        return None

    input_h = random.randint(*INPUT_H_RANGE)
    input_w = random.randint(*INPUT_W_RANGE)
    grid = np.full((input_h, input_w), BG, dtype=int)
    occupied = np.zeros((input_h, input_w), dtype=bool)

    pieces: list[tuple[int, np.ndarray]] = []
    for color, _out_h, _out_w, piece_a, piece_b in specs:
        pieces.append((color, piece_a))
        pieces.append((color, piece_b))
    random.shuffle(pieces)
    pieces.sort(key=lambda item: item[1].shape[0] * item[1].shape[1], reverse=True)

    for color, piece in pieces:
        if not _place_piece(grid, occupied, piece, color):
            return None

    bg_area = int(np.sum(grid == BG))
    fg_area = grid.size - bg_area
    if bg_area <= fg_area:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > ARC_MAX:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
