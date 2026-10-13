from __future__ import annotations

from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
DIVIDER_COLOR = GREY
ROW_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
MIN_PANEL_WIDTH = 7
MAX_PANEL_WIDTH = 14
MIN_HEIGHT = 5
MAX_HEIGHT = 10
MAX_GENERATION_ATTEMPTS = 400


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    height, total_width = map(int, g.shape)
    assert height > 0

    divider_candidates = [
        c for c in range(total_width) if (2 * c + 1 == total_width) and np.all(g[:, c] == DIVIDER_COLOR)
    ]
    assert len(divider_candidates) == 1, "expected one full-height central gray divider"

    mid = divider_candidates[0]
    left = g[:, :mid]
    right = g[:, mid + 1 :]
    assert left.shape == right.shape and left.shape[1] > 0
    return left, right


def _marker_in_row(row: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(row != BLACK)
    assert len(cols) == 1, "each right-panel row must have exactly one colored marker"

    col = int(cols[0])
    color = int(row[col])
    assert color != DIVIDER_COLOR, "row markers must be distinct from the gray divider"
    return col, color


def _unique_color_col(row: np.ndarray, color: int) -> int:
    cols = np.flatnonzero(row == int(color))
    assert len(cols) == 1, "marker color must occur exactly once in the corresponding left row"
    return int(cols[0])


def _singleton_count(row: np.ndarray) -> int:
    counts = Counter(map(int, row.tolist()))
    return sum(count == 1 for count in counts.values())


def _analyze(grid: np.ndarray) -> dict[str, object]:
    left, right = _split_panels(grid)
    height, width = left.shape

    marker_cols: list[int] = []
    marker_colors: list[int] = []
    source_cols: list[int] = []
    shifts: list[int] = []
    singleton_counts: list[int] = []

    for r in range(height):
        marker_col, marker_color = _marker_in_row(right[r])
        source_col = _unique_color_col(left[r], marker_color)
        marker_cols.append(marker_col)
        marker_colors.append(marker_color)
        source_cols.append(source_col)
        shifts.append(marker_col - source_col)
        singleton_counts.append(_singleton_count(left[r]))

    return {
        "left": left,
        "right": right,
        "height": int(height),
        "width": int(width),
        "marker_cols": marker_cols,
        "marker_colors": marker_colors,
        "source_cols": source_cols,
        "shifts": shifts,
        "singleton_counts": singleton_counts,
    }


def _spec_output(grid: np.ndarray) -> np.ndarray:
    info = _analyze(grid)
    left = np.asarray(info["left"], dtype=int)
    shifts = info["shifts"]

    out = np.empty_like(left)
    for r, shift in enumerate(shifts):
        out[r] = np.roll(left[r], int(shift))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cyclically shift each left-panel row so its marker-colored pixel docks at the marker column.

    Input structure:
    - A full-height gray divider splits the grid into equal-width left and right panels.
    - The left panel is a multicolor row pattern.
    - The right panel is black except for exactly one colored marker in each row.

    Rule:
    - For each row, read the marker color and marker column from the right panel.
    - In the corresponding left-panel row, find the unique pixel of that same color.
    - Cyclically shift the entire left row until that pixel lands in the marker's column position.

    Key insight:
    - The marker color identifies which left-row pixel to track, while the marker position gives the
      docking column; the actual shift amount must be inferred from both together.
    """
    return _spec_output(grid)


def _build_row(
    rng: np.random.Generator,
    width: int,
    target_color: int,
    require_extra_singleton: bool,
) -> np.ndarray:
    for _ in range(200):
        palette_size = int(rng.integers(3, min(6, width - 1) + 1))
        other_colors = [int(c) for c in ROW_COLORS if int(c) != int(target_color)]
        chosen_others = [int(c) for c in rng.choice(other_colors, size=palette_size - 1, replace=False)]
        palette = [int(target_color), *chosen_others]

        min_singletons = 2 if require_extra_singleton else 1
        max_singletons = min(3, palette_size - 1)
        if min_singletons > max_singletons:
            continue

        singleton_total = int(rng.integers(min_singletons, max_singletons + 1))
        extra_singletons = singleton_total - 1
        singleton_colors = {int(target_color)}
        if extra_singletons > 0:
            singleton_colors.update(
                int(c) for c in rng.choice(chosen_others, size=extra_singletons, replace=False)
            )

        counts = {color: (1 if color in singleton_colors else 2) for color in palette}
        duplicate_colors = [color for color in palette if color not in singleton_colors]
        if not duplicate_colors:
            continue

        base_total = sum(counts.values())
        if base_total > width:
            continue

        leftover = width - base_total
        for _ in range(leftover):
            chosen = int(rng.choice(duplicate_colors))
            counts[chosen] += 1

        ordered_palette = [palette[i] for i in rng.permutation(len(palette))]
        row = np.array(
            [color for color in ordered_palette for _ in range(counts[color])],
            dtype=int,
        )
        if row.shape != (width,):
            continue
        if np.count_nonzero(row == int(target_color)) != 1:
            continue
        if len(set(map(int, row.tolist()))) < 3:
            continue
        if require_extra_singleton and _singleton_count(row) < 2:
            continue
        return row

    raise RuntimeError("failed to build a valid row")


def _choose_target_colors(rng: np.random.Generator, height: int) -> list[int]:
    targets = [int(rng.choice(ROW_COLORS)) for _ in range(height)]

    shared_color = int(rng.choice(ROW_COLORS))
    shared_count = int(rng.integers(2, min(4, height) + 1))
    shared_rows = rng.choice(height, size=shared_count, replace=False)
    for r in shared_rows:
        targets[int(r)] = shared_color

    if len(set(targets)) == 1:
        targets[0] = int(shared_color)
        targets[-1] = int(next(color for color in ROW_COLORS if int(color) != shared_color))

    return targets


def _pick_marker_cols(
    rng: np.random.Generator,
    source_cols: list[int],
    width: int,
    include_zero_shift: bool,
) -> list[int]:
    height = len(source_cols)

    pos_candidates = [r for r, source in enumerate(source_cols) if source < width - 1]
    neg_candidates = [r for r, source in enumerate(source_cols) if source > 0]
    assert pos_candidates and neg_candidates

    pos_big = [r for r, source in enumerate(source_cols) if source <= width - 3]
    neg_big = [r for r, source in enumerate(source_cols) if source >= 2]
    pos_row = int(rng.choice(pos_big or pos_candidates))

    neg_pool = [r for r in (neg_big or neg_candidates) if r != pos_row]
    if not neg_pool:
        neg_pool = [r for r in neg_candidates if r != pos_row]
    if not neg_pool:
        neg_pool = neg_candidates
    neg_row = int(rng.choice(neg_pool))

    zero_row = None
    if include_zero_shift:
        zero_choices = [r for r in range(height) if r not in {pos_row, neg_row}]
        if zero_choices:
            zero_row = int(rng.choice(zero_choices))

    marker_cols = [0] * height
    for r, source in enumerate(source_cols):
        if r == pos_row:
            low = source + 2 if source + 2 < width else source + 1
            marker_cols[r] = int(rng.integers(low, width))
        elif r == neg_row:
            high = source - 1 if source >= 2 else source
            marker_cols[r] = int(rng.integers(0, high))
        elif zero_row is not None and r == zero_row:
            marker_cols[r] = int(source)
        else:
            choices = np.arange(width, dtype=int)
            if rng.random() < 0.8 and width > 1:
                choices = choices[choices != source]
            marker_cols[r] = int(rng.choice(choices))

    return marker_cols


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        left = np.asarray(info["left"], dtype=int)
        right = np.asarray(info["right"], dtype=int)
        marker_cols = list(info["marker_cols"])
        marker_colors = list(info["marker_colors"])
        source_cols = list(info["source_cols"])
        shifts = list(info["shifts"])
        singleton_counts = list(info["singleton_counts"])

        assert left.shape == right.shape
        height, width = map(int, left.shape)
        assert 1 <= height <= 30 and 1 <= width <= 14
        assert np.all(left != BLACK), "left panel should be visually dense"
        assert np.all(left != DIVIDER_COLOR), "divider color must stay unique"
        assert np.all((right == BLACK) | np.isin(right, ROW_COLORS))

        for r in range(height):
            assert len(set(map(int, left[r].tolist()))) >= 3
            assert singleton_counts[r] >= 1
            assert np.count_nonzero(right[r] != BLACK) == 1

        assert any(count >= 2 for count in singleton_counts), "need a row where color is necessary"
        assert any(shift > 0 for shift in shifts), "need rightward shifts"
        assert any(shift < 0 for shift in shifts), "need leftward shifts"
        assert any(abs(shift) >= 2 for shift in shifts), "need visible wrap-around shifts"
        assert sum(shift != 0 for shift in shifts) >= 2
        assert len(set(marker_cols)) >= 2
        assert len(set(source_cols)) >= 2
        assert np.any(_spec_output(grid) != left)

        by_color: dict[int, list[int]] = {}
        for r, color in enumerate(marker_colors):
            by_color.setdefault(int(color), []).append(r)

        shared_rows = [rows for rows in by_color.values() if len(rows) >= 2]
        assert shared_rows, "need at least one repeated marker color across rows"
        assert any(len({shifts[r] for r in rows}) > 1 for rows in shared_rows), (
            "repeated marker colors must not imply a fixed shift"
        )

        out = _spec_output(grid)
        for r in range(height):
            assert np.array_equal(np.sort(out[r]), np.sort(left[r]))
            assert out[r, marker_cols[r]] == marker_colors[r]
            assert np.array_equal(out[r], np.roll(left[r], shifts[r]))

        return True
    except AssertionError:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(MIN_HEIGHT, MAX_HEIGHT + 1))
    width = int(rng.integers(MIN_PANEL_WIDTH, MAX_PANEL_WIDTH + 1))

    target_colors = _choose_target_colors(rng, height)
    extra_rows = {
        int(r)
        for r in rng.choice(height, size=int(rng.integers(1, min(3, height) + 1)), replace=False)
    }

    rows: list[np.ndarray] = []
    source_cols: list[int] = []
    for r in range(height):
        row = _build_row(rng, width, target_colors[r], r in extra_rows)
        rows.append(row)
        source_cols.append(_unique_color_col(row, target_colors[r]))

    include_zero_shift = bool(rng.random() < 0.4)
    marker_cols = _pick_marker_cols(rng, source_cols, width, include_zero_shift)

    grid = np.zeros((height, 2 * width + 1), dtype=int)
    grid[:, :width] = np.vstack(rows)
    grid[:, width] = DIVIDER_COLOR
    for r, (marker_col, marker_color) in enumerate(zip(marker_cols, target_colors)):
        grid[r, width + 1 + marker_col] = int(marker_color)

    if not _validate_instance(grid):
        return None
    if not np.array_equal(main(grid), _spec_output(grid)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid homing-panel input")
