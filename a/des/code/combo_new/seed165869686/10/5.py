import random

import numpy as np


BG_COLOR = 0
MARKER_COLOR = 2
GRID_COLOR = 7
PATTERN_COLOR_POOL = (1, 3, 4, 5, 6, 8, 9)

REGION_COUNT_RANGE = (3, 5)
CELL_SIZE_RANGE = (3, 5)
MIN_DICT_LEN = 2
MAX_GENERATION_ATTEMPTS = 300


def _interior_spans(length: int, separators: list[int]) -> list[slice]:
    bounds = [-1] + sorted({int(x) for x in separators if 0 <= int(x) < length}) + [int(length)]
    spans: list[slice] = []
    for lo, hi in zip(bounds, bounds[1:], strict=False):
        if hi - lo > 1:
            spans.append(slice(int(lo + 1), int(hi)))
    return spans


def _find_layout(grid: np.ndarray) -> tuple[int, list[slice], list[slice]]:
    height, width = map(int, grid.shape)
    row_lines: dict[int, list[int]] = {}
    col_lines: dict[int, list[int]] = {}

    for r in range(height):
        color = int(grid[r, 0])
        if np.all(grid[r, :] == color):
            row_lines.setdefault(color, []).append(int(r))

    for c in range(width):
        color = int(grid[0, c])
        if np.all(grid[:, c] == color):
            col_lines.setdefault(color, []).append(int(c))

    best: tuple[tuple[int, int, int], int, list[slice], list[slice]] | None = None
    for color in set(row_lines) & set(col_lines):
        row_spans = _interior_spans(height, row_lines[color])
        col_spans = _interior_spans(width, col_lines[color])
        if len(row_spans) < 1 or len(col_spans) < 2:
            continue
        if len({sl.stop - sl.start for sl in row_spans}) != 1:
            continue
        if len({sl.stop - sl.start for sl in col_spans}) != 1:
            continue
        score = (
            int(len(row_spans) == len(col_spans)),
            int(len(row_spans) * len(col_spans)),
            int(len(row_lines[color]) + len(col_lines[color])),
        )
        candidate = (score, int(color), row_spans, col_spans)
        if best is None or candidate[0] > best[0]:
            best = candidate

    assert best is not None, "could not detect the region lattice"
    _, line_color, row_spans, col_spans = best
    return int(line_color), row_spans, col_spans


def _is_dictionary_cell(cell: np.ndarray) -> bool:
    colors = set(map(int, np.unique(cell)))
    colors.discard(BG_COLOR)
    return any(color != MARKER_COLOR for color in colors)


def _diagonal_mask(height: int, width: int, reverse: bool) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if height == 1:
        mask[0, 0 if not reverse else width - 1] = True
        return mask
    for r in range(height):
        t = r / float(height - 1)
        if reverse:
            t = 1.0 - t
        c = int(round(t * (width - 1)))
        mask[r, c] = True
    return mask


def _pattern_masks(height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []

    plus = np.zeros((height, width), dtype=bool)
    plus[height // 2, :] = True
    plus[:, width // 2] = True
    masks.append(_diagonal_mask(height, width, reverse=False))
    masks.append(_diagonal_mask(height, width, reverse=True))
    masks.append(plus)

    xmask = _diagonal_mask(height, width, reverse=False) | _diagonal_mask(height, width, reverse=True)
    masks.append(xmask)

    frame = np.zeros((height, width), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    masks.append(frame)

    masks.append(np.ones((height, width), dtype=bool))

    vstripe = np.zeros((height, width), dtype=bool)
    vstripe[:, ::2] = True
    masks.append(vstripe)

    hstripe = np.zeros((height, width), dtype=bool)
    hstripe[::2, :] = True
    masks.append(hstripe)

    corners = np.zeros((height, width), dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True
    masks.append(corners)

    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for mask in masks:
        key = mask.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(mask)
    return unique


def _paint_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    patch[mask] = int(color)
    return patch


def _region_slices(region_count: int, cell_h: int, cell_w: int) -> tuple[list[slice], list[slice]]:
    row_slices = [
        slice(1 + r * (cell_h + 1), 1 + r * (cell_h + 1) + cell_h)
        for r in range(region_count)
    ]
    col_slices = [
        slice(1 + c * (cell_w + 1), 1 + c * (cell_w + 1) + cell_w)
        for c in range(region_count)
    ]
    return row_slices, col_slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each region with the left-edge pattern whose index matches that region's red-marker count.

    Input structure:
    - A single-color line lattice splits the grid into equal rectangular regions arranged as a square board.
    - A top-to-bottom prefix of the leftmost region column is a pattern dictionary: each of those cells contains a colored pattern on black.
    - Every non-dictionary region contains only black and possibly some scattered red marker pixels.

    Rule:
    - Detect the regular region layout from the full-length separator rows and columns.
    - Read the dictionary as the consecutive patterned cells in the leftmost column, from top to bottom, giving pattern indices 1, 2, 3, ...
    - For every non-dictionary region, count its red pixels K; if K is zero, clear the region to black, otherwise replace the region with the K-th dictionary pattern.
    - Keep the dictionary regions and the separator lattice unchanged.

    Key insight:
    - The red pixels are not content to preserve; they are a count-based lookup key into the ordered pattern dictionary embedded on the grid edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _line_color, row_spans, col_spans = _find_layout(g)
    assert len(row_spans) == len(col_spans), "expected a square region array"

    dictionary: list[np.ndarray] = []
    first_col = col_spans[0]
    for row_span in row_spans:
        cell = g[row_span, first_col]
        if _is_dictionary_cell(cell):
            dictionary.append(cell.copy())
        else:
            break

    assert dictionary, "expected at least one dictionary pattern"

    out = g.copy()
    for r_idx, row_span in enumerate(row_spans):
        for c_idx, col_span in enumerate(col_spans):
            if c_idx == 0 and r_idx < len(dictionary):
                continue
            cell = g[row_span, col_span]
            marker_count = int(np.count_nonzero(cell == MARKER_COLOR))
            if marker_count == 0:
                out[row_span, col_span] = BG_COLOR
                continue
            assert marker_count <= len(dictionary), "marker count exceeds dictionary size"
            out[row_span, col_span] = dictionary[marker_count - 1]

    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    region_count = random.randint(*REGION_COUNT_RANGE)
    size_options = [
        (cell_h, cell_w)
        for cell_h in range(CELL_SIZE_RANGE[0], CELL_SIZE_RANGE[1] + 1)
        for cell_w in range(CELL_SIZE_RANGE[0], CELL_SIZE_RANGE[1] + 1)
        if region_count * cell_h + region_count + 1 <= 30
        and region_count * cell_w + region_count + 1 <= 30
    ]
    if not size_options:
        return None

    cell_h, cell_w = random.choice(size_options)
    dict_len = random.randint(MIN_DICT_LEN, region_count)
    masks = _pattern_masks(cell_h, cell_w)
    if len(masks) < dict_len:
        return None

    height = region_count * cell_h + region_count + 1
    width = region_count * cell_w + region_count + 1
    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = inp.copy()

    line_rows = [idx * (cell_h + 1) for idx in range(region_count + 1)]
    line_cols = [idx * (cell_w + 1) for idx in range(region_count + 1)]
    inp[line_rows, :] = GRID_COLOR
    inp[:, line_cols] = GRID_COLOR
    out[:, :] = inp

    row_slices, col_slices = _region_slices(region_count, cell_h, cell_w)

    chosen_masks = random.sample(masks, dict_len)
    chosen_colors = random.sample(PATTERN_COLOR_POOL, dict_len)
    dictionary = [
        _paint_pattern(mask, color)
        for mask, color in zip(chosen_masks, chosen_colors, strict=False)
    ]

    for r_idx, patch in enumerate(dictionary):
        inp[row_slices[r_idx], col_slices[0]] = patch
        out[row_slices[r_idx], col_slices[0]] = patch

    non_dict_positions = [
        (r_idx, c_idx)
        for r_idx in range(region_count)
        for c_idx in range(region_count)
        if not (c_idx == 0 and r_idx < dict_len)
    ]
    required_counts = list(range(dict_len + 1))
    extra_count = len(non_dict_positions) - len(required_counts)
    assert extra_count >= 0, "grid must have enough non-dictionary cells"
    counts = required_counts + [random.randint(0, dict_len) for _ in range(extra_count)]
    random.shuffle(counts)

    cell_area = cell_h * cell_w
    assert dict_len <= cell_area, "dictionary index must fit inside a region"
    for (r_idx, c_idx), marker_count in zip(non_dict_positions, counts, strict=False):
        patch = np.full((cell_h, cell_w), BG_COLOR, dtype=int)
        if marker_count > 0:
            for flat_idx in random.sample(range(cell_area), marker_count):
                patch[flat_idx // cell_w, flat_idx % cell_w] = MARKER_COLOR
            out_patch = dictionary[marker_count - 1]
        else:
            out_patch = np.full((cell_h, cell_w), BG_COLOR, dtype=int)
        inp[row_slices[r_idx], col_slices[c_idx]] = patch
        out[row_slices[r_idx], col_slices[c_idx]] = out_patch

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
