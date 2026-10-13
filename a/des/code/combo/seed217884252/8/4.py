from __future__ import annotations

import random
from collections import deque

import numpy as np


ORANGE = 7
GREY = 5
PALETTE = tuple(color for color in range(10) if color not in (ORANGE, GREY))

GRID_MIN = 18
GRID_MAX = 30
RECT_COUNT_RANGE = (2, 3)
OUTER_MARGIN = 1
OBJECT_PADDING = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_RECTANGLE_ATTEMPTS = 120
MAX_PLACEMENT_ATTEMPTS = 160

GRID_TYPES = ("horizontal", "vertical", "grid")
GRID_SHAPES = ((2, 2), (2, 3), (3, 2))
SEGMENT_VALUES = tuple(range(2, 7))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    pieces: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            pieces.append(np.array(coords, dtype=int))
    return pieces


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _split_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        assert 0 < cut < length - 1, "separator must be internal to the rectangle"
        assert start < cut, "separators may not touch"
        spans.append((start, cut))
        start = cut + 1
    assert start < length, "final span must be nonempty"
    spans.append((start, length))
    return spans


def _pack_pieces(pieces: list[tuple[int, int]], box_h: int, box_w: int) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int]] = []
    row = 0
    col = 0
    row_height = 0

    for piece_h, piece_w in pieces:
        if piece_w > box_w or piece_h > box_h:
            return None
        if col + piece_w > box_w:
            row += row_height
            col = 0
            row_height = 0
        if row + piece_h > box_h:
            return None
        placements.append((row, col))
        col += piece_w
        row_height = max(row_height, piece_h)

    return placements


def _render_patch(color: int, row_segments: list[int], col_segments: list[int]) -> np.ndarray:
    height = sum(row_segments) + len(row_segments) - 1
    width = sum(col_segments) + len(col_segments) - 1
    patch = np.full((height, width), int(color), dtype=int)

    pos = 0
    for size in row_segments[:-1]:
        pos += size
        patch[pos, :] = GREY
        pos += 1

    pos = 0
    for size in col_segments[:-1]:
        pos += size
        patch[:, pos] = GREY
        pos += 1

    return patch


def _transform_patch(patch: np.ndarray) -> np.ndarray:
    colors = set(map(int, np.unique(patch)))
    colors.discard(GREY)
    assert len(colors) == 1, "each rectangle must use one parent color plus grey separators"
    color = next(iter(colors))

    height, width = map(int, patch.shape)
    grey_rows = [r for r in range(height) if np.all(patch[r, :] == GREY)]
    grey_cols = [c for c in range(width) if np.all(patch[:, c] == GREY)]
    assert grey_rows or grey_cols, "expected at least one separator line"

    row_spans = _split_spans(height, grey_rows)
    col_spans = _split_spans(width, grey_cols)

    pieces: list[tuple[int, int, int, int]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            block = patch[r0:r1, c0:c1]
            assert np.all(block == color), "subregions must be solid rectangles"
            pieces.append((r1 - r0, c1 - c0, r0, c0))

    pieces.sort(key=lambda item: (-item[0] * item[1], item[2], item[3]))
    placements = _pack_pieces([(h, w) for h, w, _r, _c in pieces], height, width)
    assert placements is not None, "sorted pieces must fit back into the source bbox"

    out = np.full((height, width), ORANGE, dtype=int)
    for (piece_h, piece_w, _r, _c), (dst_r, dst_c) in zip(pieces, placements):
        out[dst_r : dst_r + piece_h, dst_c : dst_c + piece_w] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut each rectangle along its grey separators and repack the subrectangles by descending area.

    Input structure:
    - The background is orange.
    - Each object is a solid colored rectangle whose bbox is fully filled by its
      color and one or more grey full-span separator lines.
    - Different rectangles are separated by orange background.

    Rule:
    - For every rectangle, split it at each internal grey row/column.
    - Remove the original rectangle and separators.
    - Sort the resulting solid subrectangles from largest area to smallest.
    - Repack them inside the same bbox in reading order, placing pieces left to
      right and wrapping to the next row when needed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    rectangles = _components(g != ORANGE)
    rectangles.sort(key=lambda coords: _bbox(coords))

    for coords in rectangles:
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != ORANGE), "rectangle bbox must be fully occupied"
        assert int(coords.shape[0]) == int(patch.size), "each object must be a filled rectangle with separators"

        transformed = _transform_patch(patch)
        out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    return out


def _sample_segments(count: int) -> list[int]:
    return random.sample(SEGMENT_VALUES, count)


def _sample_rectangle_spec(kind: str) -> tuple[list[int], list[int]] | None:
    if kind == "horizontal":
        row_segments = _sample_segments(random.randint(2, 4))
        col_segments = [random.randint(3, 7)]
        return row_segments, col_segments

    if kind == "vertical":
        row_segments = [random.randint(3, 7)]
        col_segments = _sample_segments(random.randint(2, 4))
        return row_segments, col_segments

    for _ in range(MAX_RECTANGLE_ATTEMPTS):
        n_rows, n_cols = random.choice(GRID_SHAPES)
        row_segments = _sample_segments(n_rows)
        col_segments = _sample_segments(n_cols)
        areas = [rh * cw for rh in row_segments for cw in col_segments]
        if len(set(areas)) != len(areas):
            continue
        height = sum(row_segments) + n_rows - 1
        width = sum(col_segments) + n_cols - 1
        pieces = sorted(
            [(rh, cw, rr, cc) for rr, rh in enumerate(row_segments) for cc, cw in enumerate(col_segments)],
            key=lambda item: (-item[0] * item[1], item[2], item[3]),
        )
        if _pack_pieces([(h, w) for h, w, _r, _c in pieces], height, width) is not None:
            return row_segments, col_segments
    return None


def _choose_types(count: int) -> list[str]:
    if count >= len(GRID_TYPES):
        types = list(GRID_TYPES)
        random.shuffle(types)
        return types[:count]
    return random.sample(GRID_TYPES, count)


def _can_place(used: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = map(int, used.shape)
    r0 = max(0, top - OBJECT_PADDING)
    c0 = max(0, left - OBJECT_PADDING)
    r1 = min(h, top + height + OBJECT_PADDING)
    c1 = min(w, left + width + OBJECT_PADDING)
    return not used[r0:r1, c0:c1].any()


def _mark_used(used: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    used[top : top + height, left : left + width] = True


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_room = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in patches)
    base = max(max_h + 2 * OUTER_MARGIN, max_w + 2 * OUTER_MARGIN, int(np.ceil(np.sqrt(total_room))) + 2)

    low = max(GRID_MIN, base)
    if low > GRID_MAX:
        return None

    height = random.randint(low, GRID_MAX)
    width = random.randint(low, GRID_MAX)
    grid = np.full((height, width), ORANGE, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    order = list(range(len(patches)))
    random.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}

    for index in order:
        patch = patches[index]
        patch_h, patch_w = map(int, patch.shape)
        if patch_h + 2 * OUTER_MARGIN > height or patch_w + 2 * OUTER_MARGIN > width:
            return None

        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            top = random.randint(OUTER_MARGIN, height - patch_h - OUTER_MARGIN)
            left = random.randint(OUTER_MARGIN, width - patch_w - OUTER_MARGIN)
            if not _can_place(used, top, left, patch_h, patch_w):
                continue
            placements[index] = (top, left)
            _mark_used(used, top, left, patch_h, patch_w)
            placed = True
            break
        if not placed:
            return None

    for index, patch in enumerate(patches):
        top, left = placements[index]
        patch_h, patch_w = map(int, patch.shape)
        grid[top : top + patch_h, left : left + patch_w] = patch

    return grid


def _gen_attempt() -> np.ndarray | None:
    rect_count = random.randint(*RECT_COUNT_RANGE)
    types = _choose_types(rect_count)
    colors = random.sample(PALETTE, rect_count)

    patches: list[np.ndarray] = []
    changed = False
    for kind, color in zip(types, colors):
        spec = _sample_rectangle_spec(kind)
        if spec is None:
            return None
        row_segments, col_segments = spec
        patch = _render_patch(color, row_segments, col_segments)
        packed = _transform_patch(patch)
        if not np.array_equal(patch, packed):
            changed = True
        patches.append(patch)

    if not changed:
        return None

    grid = _place_patches(patches)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
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
