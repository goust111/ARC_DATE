import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
PANEL_ROW_RANGE = (3, 5)
PANEL_COL_RANGE = (3, 5)
PANEL_HEIGHT_RANGE = (3, 7)
PANEL_WIDTH_RANGE = (3, 7)
SEPARATOR_THICKNESS_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = end = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == end + 1:
            end = value
        else:
            runs.append((start, end))
            start = end = value
    runs.append((start, end))
    return runs


def _spans(total: int, runs: list[tuple[int, int]]) -> list[slice]:
    spans: list[slice] = []
    cursor = 0
    for start, end in runs:
        if cursor < start:
            spans.append(slice(cursor, start))
        cursor = end + 1
    if cursor < total:
        spans.append(slice(cursor, total))
    return [span for span in spans if span.stop > span.start]


def _find_divider_color(grid: np.ndarray) -> int:
    candidates: list[tuple[int, int]] = []
    for color in map(int, np.unique(grid)):
        full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
        full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]
        if full_rows and full_cols:
            candidates.append((len(full_rows) + len(full_cols), color))
    assert candidates, "expected a separator color spanning full rows and columns"
    return max(candidates)[1]


def _panel_slices(grid: np.ndarray, divider: int) -> tuple[list[slice], list[slice]]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == divider)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == divider)]
    row_slices = _spans(grid.shape[0], _contiguous_runs(full_rows))
    col_slices = _spans(grid.shape[1], _contiguous_runs(full_cols))
    assert len(row_slices) >= 2 and len(col_slices) >= 2, "need a panel grid with headers"
    heights = {row_slice.stop - row_slice.start for row_slice in row_slices}
    widths = {col_slice.stop - col_slice.start for col_slice in col_slices}
    assert len(heights) == 1 and len(widths) == 1, "all panels must have equal size"
    return row_slices, col_slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each interior panel by the pixelwise intersection of its column header
    mask and row header mask.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    divider = _find_divider_color(g)
    row_slices, col_slices = _panel_slices(g, divider)

    counts = Counter(map(int, g.reshape(-1)))
    bg_candidates = [color for color in counts if color != divider]
    assert bg_candidates, "expected a background color distinct from the divider"
    background = max(bg_candidates, key=lambda color: counts[color])

    top_masks = [g[row_slices[0], col_slice] == divider for col_slice in col_slices]
    left_masks = [g[row_slice, col_slices[0]] == divider for row_slice in row_slices]
    panel_shape = top_masks[0].shape
    assert all(mask.shape == panel_shape for mask in top_masks + left_masks), "header masks must align"

    out = g.copy()
    for row_index in range(1, len(row_slices)):
        for col_index in range(1, len(col_slices)):
            panel = out[row_slices[row_index], col_slices[col_index]]
            panel[:] = background
            panel[left_masks[row_index] & top_masks[col_index]] = divider
    return out


def _mask_is_valid(mask: np.ndarray) -> bool:
    if mask.dtype != bool:
        mask = mask.astype(bool)
    area = int(mask.size)
    count = int(mask.sum())
    if count < 2 or count >= area:
        return False
    if np.any(mask.all(axis=0)) or np.any(mask.all(axis=1)):
        return False
    return True


def _grow_connected_mask(height: int, width: int, target: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    seed = (rng.randrange(height), rng.randrange(width))
    mask[seed] = True
    cells = {seed}
    while len(cells) < target:
        frontier: set[tuple[int, int]] = set()
        for row, col in cells:
            for dr, dc in DIR4:
                nr, nc = row + dr, col + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        cells.add(cell)
        mask[cell] = True
    return mask


def _mask_from_segments(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    piece_count = rng.randint(2, 4)
    for _ in range(piece_count):
        kind = rng.choice(("hseg", "vseg", "rect"))
        if kind == "hseg":
            row = rng.randrange(height)
            seg_width = rng.randint(1, width - 1)
            col0 = rng.randint(0, width - seg_width)
            mask[row, col0 : col0 + seg_width] = True
        elif kind == "vseg":
            col = rng.randrange(width)
            seg_height = rng.randint(1, height - 1)
            row0 = rng.randint(0, height - seg_height)
            mask[row0 : row0 + seg_height, col] = True
        else:
            rect_height = rng.randint(1, height - 1)
            rect_width = rng.randint(1, width - 1)
            row0 = rng.randint(0, height - rect_height)
            col0 = rng.randint(0, width - rect_width)
            mask[row0 : row0 + rect_height, col0 : col0 + rect_width] = True
    return mask


def _mask_from_diagonals(height: int, width: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mode = rng.choice(("diag", "anti", "x", "l"))
    box_height = rng.randint(2, height)
    box_width = rng.randint(2, width)
    row0 = rng.randint(0, height - box_height)
    col0 = rng.randint(0, width - box_width)
    size = min(box_height, box_width)

    if mode in ("diag", "x"):
        for offset in range(size):
            mask[row0 + offset, col0 + offset] = True
    if mode in ("anti", "x"):
        for offset in range(size):
            mask[row0 + offset, col0 + box_width - 1 - offset] = True
    if mode == "l":
        mask[row0 : row0 + box_height, col0] = True
        mask[row0 + box_height - 1, col0 : col0 + box_width] = True
    return mask


def _random_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    for _ in range(120):
        family = rng.randrange(3)
        if family == 0:
            max_target = max(2, min(height * width - 1, (height * width * 2) // 3))
            target = rng.randint(2, max_target)
            mask = _grow_connected_mask(height, width, target, rng)
            if mask is None:
                continue
        elif family == 1:
            mask = _mask_from_segments(height, width, rng)
        else:
            mask = _mask_from_diagonals(height, width, rng)
        if _mask_is_valid(mask):
            return mask
    raise RuntimeError("failed to build a valid panel mask")


def _sample_unique_masks(count: int, height: int, width: int, rng: random.Random, first: np.ndarray) -> list[np.ndarray] | None:
    masks = [first.copy()]
    attempts = 0
    while len(masks) < count:
        attempts += 1
        if attempts > 200:
            return None
        candidate = _random_mask(height, width, rng)
        if any(np.array_equal(candidate, existing) for existing in masks):
            continue
        masks.append(candidate)
    return masks


def _sample_headers(panel_rows: int, panel_cols: int, height: int, width: int, rng: random.Random) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    for _ in range(200):
        corner = _random_mask(height, width, rng)
        top_masks = _sample_unique_masks(panel_cols, height, width, rng, corner)
        left_masks = _sample_unique_masks(panel_rows, height, width, rng, corner)
        if top_masks is None or left_masks is None:
            continue

        intersections = [
            left_masks[row_index] & top_masks[col_index]
            for row_index in range(1, panel_rows)
            for col_index in range(1, panel_cols)
        ]
        nonempty = [int(mask.sum()) > 0 for mask in intersections]
        if not any(nonempty) or all(nonempty):
            continue
        if max(int(mask.sum()) for mask in intersections) < 2:
            continue
        if len({mask.tobytes() for mask in intersections}) < 3:
            continue
        return top_masks, left_masks
    return None


def _perturb_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = mask.shape
    cells = [(row, col) for row in range(height) for col in range(width)]
    for _ in range(80):
        candidate = mask.copy()
        flip_count = rng.randint(1, max(1, (height * width) // 3))
        for row, col in rng.sample(cells, flip_count):
            candidate[row, col] = ~candidate[row, col]
        if _mask_is_valid(candidate) and not np.array_equal(candidate, mask):
            return candidate
    while True:
        candidate = _random_mask(height, width, rng)
        if not np.array_equal(candidate, mask):
            return candidate


def _interior_input_mask(expected: np.ndarray, rng: random.Random) -> np.ndarray:
    mode = rng.randrange(4)
    if mode == 0:
        return np.zeros_like(expected, dtype=bool)
    if mode == 1:
        return expected.copy()
    if mode == 2:
        candidate = _random_mask(*expected.shape, rng)
        if not np.array_equal(candidate, expected):
            return candidate
    return _perturb_mask(expected, rng)


def _panel_offsets(count: int, size: int, separator: int) -> list[slice]:
    spans: list[slice] = []
    cursor = 0
    for index in range(count):
        spans.append(slice(cursor, cursor + size))
        cursor += size
        if index != count - 1:
            cursor += separator
    return spans


def _paint_panel(grid: np.ndarray, row_slice: slice, col_slice: slice, mask: np.ndarray) -> None:
    panel = grid[row_slice, col_slice]
    panel[:] = BLACK
    panel[mask] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_rows = rng.randint(*PANEL_ROW_RANGE)
    panel_cols = rng.randint(*PANEL_COL_RANGE)
    panel_height = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_width = rng.randint(*PANEL_WIDTH_RANGE)
    separator = rng.choice(SEPARATOR_THICKNESS_CHOICES)

    headers = _sample_headers(panel_rows, panel_cols, panel_height, panel_width, rng)
    if headers is None:
        return None
    top_masks, left_masks = headers

    total_height = panel_rows * panel_height + (panel_rows - 1) * separator
    total_width = panel_cols * panel_width + (panel_cols - 1) * separator
    grid = np.full((total_height, total_width), BLACK, dtype=int)

    row_slices = _panel_offsets(panel_rows, panel_height, separator)
    col_slices = _panel_offsets(panel_cols, panel_width, separator)
    for row_slice in row_slices[:-1]:
        grid[row_slice.stop : row_slice.stop + separator, :] = GREY
    for col_slice in col_slices[:-1]:
        grid[:, col_slice.stop : col_slice.stop + separator] = GREY

    _paint_panel(grid, row_slices[0], col_slices[0], top_masks[0])
    for col_index in range(1, panel_cols):
        _paint_panel(grid, row_slices[0], col_slices[col_index], top_masks[col_index])
    for row_index in range(1, panel_rows):
        _paint_panel(grid, row_slices[row_index], col_slices[0], left_masks[row_index])

    changed_panel = False
    blank_outputs = 0
    nonblank_outputs = 0
    for row_index in range(1, panel_rows):
        for col_index in range(1, panel_cols):
            expected = left_masks[row_index] & top_masks[col_index]
            if int(expected.sum()) == 0:
                blank_outputs += 1
            else:
                nonblank_outputs += 1
            candidate = _interior_input_mask(expected, rng)
            if not np.array_equal(candidate, expected):
                changed_panel = True
            _paint_panel(grid, row_slices[row_index], col_slices[col_index], candidate)

    if not changed_panel or blank_outputs == 0 or nonblank_outputs == 0:
        return None

    solved = main(grid)
    if np.array_equal(solved, grid):
        return None

    solved_row_slices, solved_col_slices = _panel_slices(solved, GREY)
    for row_index in range(1, len(solved_row_slices)):
        for col_index in range(1, len(solved_col_slices)):
            expected = left_masks[row_index] & top_masks[col_index]
            panel = solved[solved_row_slices[row_index], solved_col_slices[col_index]]
            if not np.array_equal(panel == GREY, expected):
                return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-intersection puzzle")
