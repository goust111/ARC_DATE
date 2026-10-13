import numpy as np


BG = 0
SEPARATOR = 1
OBJECT = 5
LEGEND_PALETTE = (2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _left_panel_is_valid(panel: np.ndarray) -> bool:
    return panel.size > 0 and bool(np.all(panel == panel[:, :1]))


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(1, width - 1):
        column = grid[:, c]
        if int(column[0]) == bg or not np.all(column == column[0]):
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if not _left_panel_is_valid(left):
            continue
        if not np.any(right != bg):
            continue
        row_colors = left[:, 0]
        candidates.append(
            (
                len({int(color) for color in row_colors}),
                int(np.count_nonzero(right != bg)),
                c,
            )
        )
    assert candidates, "expected a unique separator column"
    return max(candidates)[2]


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each gray right-panel pixel by the color of its matching legend row.

    Input structure:
    - The background is black, and one full-height blue separator column splits
      the grid into a left legend panel and a right object panel.
    - In the left panel, every row is uniform: it is either all background or
      fully filled with one color, forming a row-by-row legend.
    - The right panel contains gray connected objects on the background.

    Rule:
    - Treat the left panel as a mapping from row index to color.
    - For every non-background pixel in the right panel, look up the color of
      the same row in the left panel.
    - If that legend row is colored, replace the gray pixel by that color; if
      the legend row is background, leave the gray pixel unchanged.
    - Keep the left panel and the separator column unchanged.

    Key insight:
    - The left stripes are a row-wise lookup table, so one right-panel object
      can split into multiple colors when it crosses multiple legend rows.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _dominant_color(g)
    sep = _find_separator_col(g, bg)
    row_colors = g[:, 0]

    out = g.copy()
    for r, color in enumerate(row_colors):
        if int(color) == bg:
            continue
        row = out[r, sep + 1 :]
        row[row != bg] = int(color)
    return out


def _random_partition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [total]
    cuts = np.sort(rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int).tolist()


def _build_row_legend(height: int, rng: np.random.Generator) -> np.ndarray:
    n_runs = int(rng.integers(4, min(8, height) + 1))
    run_lengths = _random_partition(height, n_runs, rng)
    palette_size = int(rng.integers(2, min(6, len(LEGEND_PALETTE)) + 1))
    palette = rng.choice(LEGEND_PALETTE, size=palette_size, replace=False).astype(int).tolist()
    choices = [BG] + palette
    run_colors = [BG, palette[0], palette[1]]
    run_colors.extend(rng.choice(choices, size=n_runs - 3, replace=True).astype(int).tolist())
    rng.shuffle(run_colors)

    for i in range(1, len(run_colors)):
        if run_colors[i] == run_colors[i - 1]:
            alternatives = [color for color in choices if color != run_colors[i - 1]]
            run_colors[i] = int(rng.choice(alternatives))
    for i in range(len(run_colors) - 2, -1, -1):
        if run_colors[i] == run_colors[i + 1]:
            alternatives = [color for color in choices if color != run_colors[i + 1]]
            run_colors[i] = int(rng.choice(alternatives))

    row_colors = np.empty(height, dtype=int)
    start = 0
    for length, color in zip(run_lengths, run_colors):
        row_colors[start : start + length] = int(color)
        start += length

    assert np.any(row_colors == BG)
    assert len({int(color) for color in row_colors if int(color) != BG}) >= 2
    return row_colors


def _mixed_spans(row_colors: np.ndarray) -> list[tuple[int, int, int, int]]:
    spans: list[tuple[int, int, int, int]] = []
    height = len(row_colors)
    for r0 in range(height):
        for r1 in range(r0 + 1, min(height, r0 + 6)):
            segment = row_colors[r0 : r1 + 1]
            if not np.any(segment == BG) or not np.any(segment != BG):
                continue
            spans.append(
                (
                    len({int(color) for color in segment}),
                    r1 - r0 + 1,
                    r0,
                    r1,
                )
            )
    return spans


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == len(coords)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _random_connected_mask(
    height: int,
    width: int,
    rng: np.random.Generator,
    require_vertical_span: bool,
) -> np.ndarray:
    if height == 2 and width == 2:
        return np.array([[1, 1], [1, 0]], dtype=bool)

    mask = np.ones((height, width), dtype=bool)
    boundary = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
    ]
    rng.shuffle(boundary)
    rng.shuffle(interior)
    removal_budget = int(rng.integers(0, max(1, height * width // 3) + 1))

    for r, c in boundary + interior:
        if removal_budget == 0 or int(mask.sum()) <= 3:
            break
        trial = mask.copy()
        trial[r, c] = False
        if require_vertical_span and (not trial[0].any() or not trial[-1].any()):
            continue
        if not _is_connected(trial):
            continue
        mask = trial
        removal_budget -= 1

    mask = _trim_mask(mask)
    if require_vertical_span:
        assert mask[0].any() and mask[-1].any()
    return mask


def _can_place(occupied: np.ndarray, r0: int, c0: int, height: int, width: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + height + 1)
    cc1 = min(occupied.shape[1], c0 + width + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _stamp_mask(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    right_start: int,
) -> None:
    height, width = mask.shape
    window = grid[top : top + height, right_start + left : right_start + left + width]
    window[mask] = OBJECT
    rr0 = max(0, top - 1)
    cc0 = max(0, left - 1)
    rr1 = min(occupied.shape[0], top + height + 1)
    cc1 = min(occupied.shape[1], left + width + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _expected_output(inp: np.ndarray, row_colors: np.ndarray, right_start: int) -> np.ndarray:
    out = inp.copy()
    for r, color in enumerate(row_colors):
        if int(color) == BG:
            continue
        row = out[r, right_start:]
        row[row == OBJECT] = int(color)
    return out


def _place_mixed_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    row_colors: np.ndarray,
    right_width: int,
    right_start: int,
    rng: np.random.Generator,
) -> bool:
    spans = _mixed_spans(row_colors)
    assert spans
    spans.sort()
    best_distinct, best_height = spans[-1][0], spans[-1][1]
    good_spans = [span for span in spans if span[0] == best_distinct and span[1] >= min(3, best_height)]
    if not good_spans:
        good_spans = [span for span in spans if span[0] == best_distinct]
    _, _span_h, top, bottom = good_spans[int(rng.integers(len(good_spans)))]
    bbox_height = bottom - top + 1

    for _ in range(80):
        width = int(rng.integers(2, min(7, right_width) + 1))
        mask = _random_connected_mask(bbox_height, width, rng, require_vertical_span=True)
        height, width = mask.shape
        if width > right_width:
            continue
        left = int(rng.integers(0, right_width - width + 1))
        if not _can_place(occupied, top, left, height, width):
            continue
        _stamp_mask(grid, occupied, mask, top, left, right_start)
        return True
    return False


def _place_extra_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    height: int,
    right_width: int,
    right_start: int,
    rng: np.random.Generator,
) -> bool:
    for _ in range(120):
        bbox_height = int(rng.integers(2, min(7, height) + 1))
        bbox_width = int(rng.integers(2, min(7, right_width) + 1))
        mask = _random_connected_mask(
            bbox_height,
            bbox_width,
            rng,
            require_vertical_span=bool(rng.integers(0, 2)),
        )
        bbox_height, bbox_width = mask.shape
        top = int(rng.integers(0, height - bbox_height + 1))
        left = int(rng.integers(0, right_width - bbox_width + 1))
        if not _can_place(occupied, top, left, bbox_height, bbox_width):
            continue
        _stamp_mask(grid, occupied, mask, top, left, right_start)
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(10, 25))
    left_width = int(rng.integers(2, 8))
    right_width = int(rng.integers(8, 30 - left_width))
    total_width = left_width + 1 + right_width
    if total_width > 30:
        return None

    row_colors = _build_row_legend(height, rng)
    if not _mixed_spans(row_colors):
        return None

    grid = np.full((height, total_width), BG, dtype=int)
    grid[:, :left_width] = row_colors[:, None]
    grid[:, left_width] = SEPARATOR

    occupied = np.zeros((height, right_width), dtype=bool)
    right_start = left_width + 1

    if not _place_mixed_object(grid, occupied, row_colors, right_width, right_start, rng):
        return None

    target_objects = int(rng.integers(2, 6))
    placed = 1
    while placed < target_objects:
        if not _place_extra_object(grid, occupied, height, right_width, right_start, rng):
            break
        placed += 1
    if placed < 2:
        return None

    expected = _expected_output(grid, row_colors, right_start)
    if _dominant_color(grid) != BG:
        return None
    if _find_separator_col(grid, BG) != left_width:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
