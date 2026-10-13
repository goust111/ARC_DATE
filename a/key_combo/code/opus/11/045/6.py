from __future__ import annotations

import numpy as np


BLACK = 0
PINK = 6
TEAL = 8

CARRIER_COLORS = (1, 2, 3, 4)
CONTENT_COLORS = (0, 1, 2, 3, 4, 5, 7, 9)

HEIGHT_RANGE = (10, 14)
WIDTH_RANGE = (14, 22)
ACTIVE_COUNT_RANGE = (3, 6)
SEGMENT_WIDTH_RANGE = (2, 8)
MAX_GENERATION_ATTEMPTS = 400


def _stack_rows(rows: list[np.ndarray]) -> np.ndarray:
    assert rows, "expected at least one extracted row"
    width = max(int(row.size) for row in rows)
    out = np.zeros((len(rows), width), dtype=int)
    for r, row in enumerate(rows):
        out[r, : row.size] = row
    return out


def _marker_pair(row: np.ndarray) -> tuple[int, int] | None:
    pink_cols = np.flatnonzero(row == PINK)
    teal_cols = np.flatnonzero(row == TEAL)
    if len(pink_cols) != 1 or len(teal_cols) != 1:
        return None
    left = int(pink_cols[0])
    right = int(teal_cols[0])
    if left >= right or right - left < 3:
        return None
    inner = row[left + 1 : right]
    if np.isin(inner, [PINK, TEAL]).any():
        return None
    return left, right


def _active_row_info(row: np.ndarray) -> tuple[int, int, int] | None:
    pair = _marker_pair(row)
    if pair is None:
        return None
    left, right = pair

    outer = np.concatenate((row[:left], row[right + 1 :]))
    if outer.size == 0:
        return None

    carrier = int(outer[0])
    if carrier == BLACK:
        return None
    if not np.all(outer == carrier):
        return None
    return left, right, carrier


def main(grid: np.ndarray) -> np.ndarray:
    """Select carrier-uniform bracket rows, extract the bracketed spans, recolor them by row color, and stack them.

    Input structure:
    - The grid is mostly black-backed and contains several horizontal candidate rows.
    - A valid source row has exactly one magenta pixel and one teal pixel, with the magenta to the left.
    - In each valid source row, every cell outside that marker pair is the same non-black carrier color; the cells between the markers may contain black and other non-marker colors.

    Rule:
    - Scan rows from top to bottom and keep only rows whose outer carrier pixels are uniform and non-black.
    - For each kept row, copy the cells strictly between the magenta and teal markers.
    - Recolor every non-black copied cell to that row's carrier color, keep black cells black, then stack the resulting row segments in source-row order.
    - Pad shorter stacked rows on the right with black so the output width matches the longest extracted segment.

    Key insight:
    - The marker pair alone is insufficient because decoy rows can also contain brackets; the real sources are identified by their uniform carrier color outside the brackets, and that same carrier color determines the recoloring of the extracted content.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    rows: list[np.ndarray] = []
    for r in range(g.shape[0]):
        info = _active_row_info(g[r])
        if info is None:
            continue
        left, right, carrier = info
        segment = g[r, left + 1 : right].copy()
        segment[segment != BLACK] = carrier
        rows.append(segment)

    return _stack_rows(rows)


def _random_segment(
    rng: np.random.Generator,
    length: int,
    carrier: int,
    *,
    require_black: bool,
    require_foreign: bool,
) -> np.ndarray:
    foreign_pool = [c for c in CONTENT_COLORS if c not in (BLACK, carrier)]
    palette = [BLACK, carrier] + rng.choice(foreign_pool, size=int(rng.integers(2, 5)), replace=False).tolist()
    weights = np.array([0.28 if c == BLACK else 0.30 if c == carrier else 0.42 / (len(palette) - 2) for c in palette], dtype=float)
    weights /= weights.sum()

    segment = rng.choice(np.array(palette, dtype=int), size=length, p=weights).astype(int)
    touched: set[int] = set()

    if require_black:
        idx = int(rng.integers(length))
        segment[idx] = BLACK
        touched.add(idx)

    if require_foreign:
        choices = [i for i in range(length) if i not in touched]
        idx = int(rng.choice(choices if choices else np.arange(length)))
        segment[idx] = int(rng.choice(foreign_pool))
        touched.add(idx)

    if not np.any(segment != BLACK):
        segment[int(rng.integers(length))] = carrier

    return segment


def _build_active_row(
    rng: np.random.Generator,
    width: int,
    carrier: int,
    segment_width: int,
    *,
    require_black: bool,
    require_foreign: bool,
) -> np.ndarray:
    max_left = width - segment_width - 3
    left = int(rng.integers(1, max_left + 1))
    right = left + segment_width + 1

    row = np.full(width, carrier, dtype=int)
    row[left] = PINK
    row[right] = TEAL
    row[left + 1 : right] = _random_segment(
        rng,
        segment_width,
        carrier,
        require_black=require_black,
        require_foreign=require_foreign,
    )
    assert _active_row_info(row) is not None
    return row


def _build_paired_decoy_row(rng: np.random.Generator, width: int) -> np.ndarray:
    segment_width = int(rng.integers(SEGMENT_WIDTH_RANGE[0], min(SEGMENT_WIDTH_RANGE[1], width - 4) + 1))
    max_left = width - segment_width - 3
    left = int(rng.integers(1, max_left + 1))
    right = left + segment_width + 1

    row = np.zeros(width, dtype=int)
    mode = int(rng.integers(3))
    if mode == 0:
        c1, c2 = rng.choice(np.array(CARRIER_COLORS, dtype=int), size=2, replace=False)
        row[:left] = int(c1)
        row[right + 1 :] = int(c2)
    elif mode == 1:
        carrier = int(rng.choice(CARRIER_COLORS))
        row[:left] = carrier
        row[right + 1 :] = BLACK
    else:
        carrier = int(rng.choice(CARRIER_COLORS))
        row[:left] = carrier
        row[right + 1 :] = carrier
        if left >= 2:
            row[int(rng.integers(0, left))] = int(rng.choice([c for c in CARRIER_COLORS if c != carrier]))
        else:
            row[right + 1 + int(rng.integers(max(1, width - right - 1)))] = int(rng.choice([c for c in CARRIER_COLORS if c != carrier]))

    row[left] = PINK
    row[right] = TEAL
    inner = _random_segment(
        rng,
        segment_width,
        int(rng.choice(CARRIER_COLORS)),
        require_black=bool(rng.integers(0, 2)),
        require_foreign=True,
    )
    row[left + 1 : right] = inner
    assert _marker_pair(row) is not None and _active_row_info(row) is None
    return row


def _build_single_marker_decoy_row(rng: np.random.Generator, width: int) -> np.ndarray:
    row = np.full(width, int(rng.choice(CARRIER_COLORS)), dtype=int)
    if rng.random() < 0.55:
        alt = int(rng.choice([c for c in CONTENT_COLORS if c not in (PINK, TEAL, row[0])]))
        count = int(rng.integers(2, max(3, width // 3 + 1)))
        cols = rng.choice(width, size=count, replace=False)
        row[cols] = alt
    marker_col = int(rng.integers(width))
    row[marker_col] = int(rng.choice([PINK, TEAL]))
    assert _marker_pair(row) is None
    return row


def _build_filler_row(rng: np.random.Generator, width: int) -> np.ndarray:
    palette = rng.choice(np.array(CONTENT_COLORS, dtype=int), size=int(rng.integers(2, 5)), replace=False)
    row = rng.choice(palette, size=width).astype(int)
    row[row == PINK] = 5
    row[row == TEAL] = 7
    if np.all(row == row[0]):
        row[int(rng.integers(width))] = int((row[0] + 2) % 10)
        if row[int(np.argmax(row != row[0]))] in (PINK, TEAL):
            row[int(np.argmax(row != row[0]))] = 5
    assert _marker_pair(row) is None
    return row


def _naive_extract_all_paired(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        pair = _marker_pair(row)
        if pair is None:
            continue
        left, right = pair
        rows.append(row[left + 1 : right].copy())
    return _stack_rows(rows)


def _naive_extract_active_without_recolor(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        info = _active_row_info(row)
        if info is None:
            continue
        left, right, _carrier = info
        rows.append(row[left + 1 : right].copy())
    return _stack_rows(rows)


def _naive_extract_active_fill_solid(grid: np.ndarray) -> np.ndarray:
    rows = []
    for row in np.asarray(grid, dtype=int):
        info = _active_row_info(row)
        if info is None:
            continue
        left, right, carrier = info
        rows.append(np.full(right - left - 1, carrier, dtype=int))
    return _stack_rows(rows)


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    active_rows: list[tuple[int, int, int, np.ndarray]] = []
    paired_decoys = 0
    single_marker_decoys = 0

    for row in grid:
        info = _active_row_info(row)
        pair = _marker_pair(row)
        pink_count = int(np.count_nonzero(row == PINK))
        teal_count = int(np.count_nonzero(row == TEAL))

        if info is not None:
            left, right, carrier = info
            active_rows.append((left, right, carrier, row[left + 1 : right].copy()))
        elif pair is not None:
            paired_decoys += 1
        elif pink_count + teal_count == 1:
            single_marker_decoys += 1

    if not (ACTIVE_COUNT_RANGE[0] <= len(active_rows) <= ACTIVE_COUNT_RANGE[1]):
        return False
    if paired_decoys < 1 or single_marker_decoys < 1:
        return False

    carriers = {carrier for _left, _right, carrier, _segment in active_rows}
    if len(carriers) < 2:
        return False

    segment_widths = {int(segment.size) for _left, _right, _carrier, segment in active_rows}
    if len(segment_widths) < 2:
        return False

    if not any(np.any(segment == BLACK) for _left, _right, _carrier, segment in active_rows):
        return False
    if not any(np.any((segment != BLACK) & (segment != carrier)) for _left, _right, carrier, segment in active_rows):
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if out.shape[0] != len(active_rows) or out.shape[1] != max(segment_widths):
        return False

    if np.array_equal(out, _naive_extract_all_paired(grid)):
        return False
    if np.array_equal(out, _naive_extract_active_without_recolor(grid)):
        return False
    if np.array_equal(out, _naive_extract_active_fill_solid(grid)):
        return False

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    width = int(rng.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    max_segment = min(SEGMENT_WIDTH_RANGE[1], width - 4)
    if max_segment < SEGMENT_WIDTH_RANGE[0]:
        return None

    active_count = int(rng.integers(ACTIVE_COUNT_RANGE[0], ACTIVE_COUNT_RANGE[1] + 1))
    paired_decoy_count = int(rng.integers(1, 3))
    single_decoy_count = int(rng.integers(1, 3))
    filler_count = int(rng.integers(1, 4))
    height = active_count + paired_decoy_count + single_decoy_count + filler_count
    if not (HEIGHT_RANGE[0] <= height <= HEIGHT_RANGE[1]):
        return None

    segment_widths = rng.integers(SEGMENT_WIDTH_RANGE[0], max_segment + 1, size=active_count).astype(int)
    if len(set(map(int, segment_widths))) == 1:
        idx = int(rng.integers(active_count))
        segment_widths[idx] = SEGMENT_WIDTH_RANGE[0] if int(segment_widths[idx]) != SEGMENT_WIDTH_RANGE[0] else max_segment

    carriers = rng.choice(np.array(CARRIER_COLORS, dtype=int), size=active_count, replace=True).astype(int)
    if len(set(map(int, carriers))) == 1:
        carriers[0] = CARRIER_COLORS[0]
        carriers[1] = CARRIER_COLORS[1]

    rows: list[np.ndarray] = []
    for i in range(active_count):
        rows.append(
            _build_active_row(
                rng,
                width,
                int(carriers[i]),
                int(segment_widths[i]),
                require_black=(i == 0) or (i > 1 and bool(rng.integers(0, 2))),
                require_foreign=(i == 1) or (i > 1 and bool(rng.integers(0, 2))),
            )
        )

    for _ in range(paired_decoy_count):
        rows.append(_build_paired_decoy_row(rng, width))
    for _ in range(single_decoy_count):
        rows.append(_build_single_marker_decoy_row(rng, width))
    for _ in range(filler_count):
        rows.append(_build_filler_row(rng, width))

    order = rng.permutation(len(rows))
    grid = np.vstack([rows[int(i)] for i in order]).astype(int)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
