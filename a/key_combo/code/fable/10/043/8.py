import random

import numpy as np


def _render_tile(tile: np.ndarray, height: int, width: int) -> np.ndarray:
    rows = np.arange(height) % tile.shape[0]
    cols = np.arange(width) % tile.shape[1]
    return tile[rows[:, None], cols[None, :]]


def _exact_row_period(grid: np.ndarray) -> int:
    height = grid.shape[0]
    for period in range(1, height + 1):
        if np.array_equal(grid, grid[np.arange(height) % period]):
            return period
    raise AssertionError("row period not found")


def _exact_col_period(grid: np.ndarray) -> int:
    width = grid.shape[1]
    for period in range(1, width + 1):
        if np.array_equal(grid, grid[:, np.arange(width) % period]):
            return period
    raise AssertionError("column period not found")


def _majority_prediction(grid: np.ndarray, row_period: int, col_period: int) -> tuple[np.ndarray, int] | None:
    height, width = grid.shape
    tile = np.zeros((row_period, col_period), dtype=int)
    score = 0

    for rr in range(row_period):
        for cc in range(col_period):
            values = grid[rr:height:row_period, cc:width:col_period].ravel()
            counts = np.bincount(values, minlength=10)
            best = int(counts.max())
            winners = np.flatnonzero(counts == best)
            if winners.size != 1:
                return None
            tile[rr, cc] = int(winners[0])
            score += best

    return _render_tile(tile, height, width), score


def _candidate_scores(grid: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for row_period in range(1, height // 2 + 1):
        for col_period in range(1, width // 2 + 1):
            result = _majority_prediction(grid, row_period, col_period)
            if result is None:
                continue
            _predicted, score = result
            candidates.append((score, row_period, col_period))
    return candidates


def _best_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    best_key = None
    best_period = None
    best_predicted = None

    height, width = grid.shape
    for row_period in range(1, height // 2 + 1):
        for col_period in range(1, width // 2 + 1):
            result = _majority_prediction(grid, row_period, col_period)
            if result is None:
                continue
            predicted, score = result
            key = (score, -(row_period * col_period), -row_period, -col_period)
            if best_key is None or key > best_key:
                best_key = key
                best_period = (row_period, col_period)
                best_predicted = predicted

    assert best_period is not None and best_predicted is not None, "expected a visible 2D wallpaper period"
    return best_period[0], best_period[1], best_predicted


def _compress_violations(grid: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != predicted)
    assert coords.size > 0, "expected at least one period violation"

    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1, "violations must share one visible color"
    color = next(iter(colors))
    assert color != 0, "violation color must remain visible after compression"

    row_parity = {int(r % 2) for r, _c in coords}
    col_parity = {int(c % 2) for _r, c in coords}
    assert len(row_parity) == 1 and len(col_parity) == 1, "violations must lie on one two-step lattice"

    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    logical_rows = ((coords[:, 0] - r0) // 2).astype(int)
    logical_cols = ((coords[:, 1] - c0) // 2).astype(int)
    assert np.all(coords[:, 0] == r0 + 2 * logical_rows)
    assert np.all(coords[:, 1] == c0 + 2 * logical_cols)

    out = np.zeros((int(logical_rows.max()) + 1, int(logical_cols.max()) + 1), dtype=int)
    out[logical_rows, logical_cols] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the sparse two-step sprite by subtracting the shortest wallpaper period and compressing the violations.

    Input structure:
    - The full grid is a 2D wallpaper made from a shortest repeating tile in both rows and columns.
    - A set of cells has been recolored away from that wallpaper prediction.
    - All recolored cells share one color, but that color also appears normally inside the wallpaper.

    Rule:
    - Search the visible proper row/column periods and choose the shortest pair whose repeated tile explains the most cells.
    - Reconstruct the wallpaper by majority vote within each period residue class, then mark every cell that disagrees with that prediction.
    - Take the mismatch set, translate it so its top-left mismatch becomes the origin, divide both coordinates by 2, and output the tight bounding box with mismatch cells in the violation color and everything else black.

    Key insight:
    - The object is never directly contiguous in the input: only the wallpaper period reveals which same-colored cells are genuine two-step-lattice violations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2

    row_period, col_period, predicted = _best_period(g)
    assert row_period <= g.shape[0] // 2 and col_period <= g.shape[1] // 2
    return _compress_violations(g, predicted)


GEN_ATTEMPTS = 400
PERIOD_RANGE = (2, 5)
PALETTE_SIZE_RANGE = (3, 5)
LOGICAL_DIM_RANGE = (1, 5)
LOGICAL_AREA_RANGE = (2, 10)
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _random_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    if area > height * width:
        return None

    for _ in range(120):
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True
        cells = [start]

        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True
            cells.append((nr, nc))

        if mask.sum() != area:
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows[0] != 0 or rows[-1] != height - 1 or cols[0] != 0 or cols[-1] != width - 1:
            continue
        return mask
    return None


def _sample_shape(max_height: int, max_width: int) -> np.ndarray | None:
    for _ in range(150):
        height = random.randint(1, max(1, min(LOGICAL_DIM_RANGE[1], max_height)))
        width = random.randint(1, max(1, min(LOGICAL_DIM_RANGE[1], max_width)))
        max_area = min(LOGICAL_AREA_RANGE[1], height * width)
        if max_area < LOGICAL_AREA_RANGE[0]:
            continue
        area = random.randint(LOGICAL_AREA_RANGE[0], max_area)
        mask = _random_connected_mask(height, width, area)
        if mask is not None:
            return mask
    return None


def _sample_clean_wallpaper() -> tuple[np.ndarray, np.ndarray, int, int] | None:
    for _ in range(120):
        row_period = random.randint(*PERIOD_RANGE)
        col_period = random.randint(*PERIOD_RANGE)

        max_height = min(30, 3 * row_period + 1)
        max_width = min(30, 3 * col_period + 1)
        height = random.randint(2 * row_period, max_height)
        width = random.randint(2 * col_period, max_width)

        palette_size = random.randint(*PALETTE_SIZE_RANGE)
        palette = random.sample(range(1, 10), palette_size)
        tile = np.array(random.choices(palette, k=row_period * col_period), dtype=int).reshape(row_period, col_period)
        clean = _render_tile(tile, height, width)

        if len(set(map(int, np.unique(clean)))) < 3:
            continue
        if _exact_row_period(clean) != row_period or _exact_col_period(clean) != col_period:
            continue
        return clean, tile, row_period, col_period
    return None


def _embed_violations(clean: np.ndarray, shape: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = clean.shape
    logical_h, logical_w = shape.shape
    row_span = 2 * logical_h - 1
    col_span = 2 * logical_w - 1
    if row_span > height or col_span > width:
        return None

    coords = np.argwhere(shape)
    palette = list(map(int, np.unique(clean)))
    random.shuffle(palette)

    for color in palette:
        for _ in range(120):
            r0 = random.randint(0, height - row_span)
            c0 = random.randint(0, width - col_span)
            absolute = np.array([(r0 + 2 * int(r), c0 + 2 * int(c)) for r, c in coords], dtype=int)
            if np.any(clean[absolute[:, 0], absolute[:, 1]] == color):
                continue
            if not np.any(clean == color):
                continue

            grid = clean.copy()
            grid[absolute[:, 0], absolute[:, 1]] = color
            expected = np.zeros(shape.shape, dtype=int)
            expected[shape] = color
            return grid, expected
    return None


def _validate_instance(
    grid: np.ndarray,
    expected: np.ndarray,
    clean: np.ndarray,
    row_period: int,
    col_period: int,
) -> bool:
    result = _majority_prediction(grid, row_period, col_period)
    if result is None:
        return False
    predicted, true_score = result
    if not np.array_equal(predicted != grid, clean != grid):
        return False

    scores = _candidate_scores(grid)
    if not scores:
        return False
    scores.sort(reverse=True)
    best_score = scores[0][0]
    best_pairs = {(rp, cp) for score, rp, cp in scores if score == best_score}
    if best_pairs != {(row_period, col_period)}:
        return False

    chosen_row_period, chosen_col_period, _predicted = _best_period(grid)
    if (chosen_row_period, chosen_col_period) != (row_period, col_period):
        return False

    violations = np.argwhere(grid != predicted)
    if violations.shape[0] != int(np.count_nonzero(expected)):
        return False
    if len({int(grid[r, c]) for r, c in violations}) != 1:
        return False
    if len({int(r % 2) for r, _c in violations}) != 1:
        return False
    if len({int(c % 2) for _r, c in violations}) != 1:
        return False
    if np.array_equal(expected, np.zeros_like(expected)):
        return False

    if true_score != grid.size - violations.shape[0]:
        return False

    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        wallpaper = _sample_clean_wallpaper()
        if wallpaper is None:
            continue
        clean, _tile, row_period, col_period = wallpaper

        max_shape_h = (clean.shape[0] + 1) // 2
        max_shape_w = (clean.shape[1] + 1) // 2
        shape = _sample_shape(max_shape_h, max_shape_w)
        if shape is None:
            continue

        embedded = _embed_violations(clean, shape)
        if embedded is None:
            continue
        grid, expected = embedded

        if not _validate_instance(grid, expected, clean, row_period, col_period):
            continue

        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return grid

    raise RuntimeError("generate_input: failed to construct a nondegenerate wallpaper-break sample")
