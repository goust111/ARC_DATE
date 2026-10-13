from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

GEN_ATTEMPTS = 400
RNG = np.random.default_rng()


def _extract_row_spec(row: np.ndarray) -> dict[str, int]:
    row = np.asarray(row, dtype=int)
    assert row.ndim == 1
    assert set(map(int, np.unique(row))).issubset({BLACK, BLUE, GREEN, YELLOW})

    colored = np.flatnonzero(row != BLACK)
    assert colored.size > 0, "each row must contain one colored strip"
    band_start = int(colored[0])
    band_end = int(colored[-1])
    assert np.all(row[:band_start] == BLACK)
    assert np.all(row[band_end + 1 :] == BLACK)
    assert np.all(row[band_start : band_end + 1] != BLACK), "strip must be contiguous"

    green_cols = np.flatnonzero(row == GREEN)
    assert green_cols.size > 0, "each row must contain one green run"
    green_start = int(green_cols[0])
    green_end = int(green_cols[-1])
    assert green_cols.size == green_end - green_start + 1, "green cells must form one run"

    blue_len = green_start - band_start
    yellow_len = band_end - green_end
    assert blue_len > 0 and yellow_len > 0, "green run must be bracketed by blue and yellow"
    assert np.all(row[band_start:green_start] == BLUE)
    assert np.all(row[green_end + 1 : band_end + 1] == YELLOW)

    return {
        "band_start": band_start,
        "band_end": band_end,
        "blue_len": blue_len,
        "green_start": green_start,
        "green_end": green_end,
        "green_width": green_end - green_start + 1,
        "yellow_len": yellow_len,
    }


def _align_rows(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, int]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    specs = [_extract_row_spec(g[r]) for r in range(h)]
    green_widths = {spec["green_width"] for spec in specs}
    assert len(green_widths) == 1, "all green runs must share the same width"
    green_width = next(iter(green_widths))
    anchor_end = min(spec["green_end"] for spec in specs)

    out = np.full_like(g, BLACK)
    for r, spec in enumerate(specs):
        band_start = anchor_end - green_width + 1 - spec["blue_len"]
        band_end = anchor_end + spec["yellow_len"]
        assert 0 <= band_start < band_end < w, "aligned strip must stay within bounds"
        out[r, band_start : band_start + spec["blue_len"]] = BLUE
        out[r, band_start + spec["blue_len"] : anchor_end + 1] = GREEN
        out[r, anchor_end + 1 : band_end + 1] = YELLOW

    return out, specs, anchor_end


def _paint_from_new_blank_columns(original: np.ndarray, aligned: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blank_before = np.all(original == BLACK, axis=0)
    blank_after = np.all(aligned == BLACK, axis=0)
    seed_cols = blank_after & ~blank_before
    assert np.any(seed_cols), "alignment must expose at least one new blank column"

    paint_cols = np.zeros_like(seed_cols, dtype=bool)
    for c in np.flatnonzero(seed_cols):
        left = int(c)
        while left >= 0 and bool(blank_after[left]):
            paint_cols[left] = True
            left -= 1
        right = int(c) + 1
        while right < aligned.shape[1] and bool(blank_after[right]):
            paint_cols[right] = True
            right += 1

    out = aligned.copy()
    out[:, paint_cols] = RED
    return out, blank_before, blank_after, seed_cols, paint_cols


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    aligned, specs, anchor_end = _align_rows(grid)
    out, blank_before, blank_after, seed_cols, paint_cols = _paint_from_new_blank_columns(
        np.asarray(grid, dtype=int), aligned
    )
    return {
        "aligned": aligned,
        "output": out,
        "specs": specs,
        "anchor_end": anchor_end,
        "blank_before": blank_before,
        "blank_after": blank_after,
        "seed_cols": seed_cols,
        "paint_cols": paint_cols,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align each row's green run to the leftmost right edge, then paint the newly exposed blank column band red.

    Input structure:
    - The grid uses black background with one contiguous colored strip per row.
    - Each strip is `[blue...][green...][yellow...]` in that order, with one green run per row.
    - All rows share the same green-run width, but their horizontal offsets differ.

    Rule:
    - Find the green run in every row and compute the minimum green right-edge column.
    - Slide each row's whole colored strip horizontally so the green right edge lands on that minimum column, keeping the row's blue length, green width, and yellow length unchanged.
    - Compare the aligned grid to the original grid, mark every column that becomes fully black only after alignment as a red seed, then extend red left and right through adjacent fully black columns in the aligned grid.
    - Keep all blue, green, and yellow cells from the aligned state unchanged.

    Key insight:
    - The informative columns are not the original blank columns but the blank band exposed by alignment; those new empty columns seed a red fill that grows across the connected empty corridor they reveal.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _build_row(width: int, band_start: int, blue_len: int, green_width: int, yellow_len: int) -> np.ndarray:
    row = np.full(width, BLACK, dtype=int)
    green_start = band_start + blue_len
    green_end = green_start + green_width
    yellow_end = green_end + yellow_len
    row[band_start:green_start] = BLUE
    row[green_start:green_end] = GREEN
    row[green_end:yellow_end] = YELLOW
    return row


def _make_candidate() -> np.ndarray | None:
    h = int(RNG.integers(8, 12))
    w = int(RNG.integers(13, 18))
    green_width = int(RNG.integers(3, min(6, w - 6)))

    blue_lens = RNG.integers(1, 4, size=h)
    yellow_lens = RNG.integers(1, 4, size=h)
    if np.unique(blue_lens).size < 2 or np.unique(yellow_lens).size < 2:
        return None

    want_left_blank_branch = bool(RNG.integers(0, 10) < 6)
    lower = int(green_width + blue_lens.max() - 1 + (1 if want_left_blank_branch else 0))
    upper = int(w - 3 - yellow_lens.max())
    if lower > upper:
        return None
    anchor_end = int(RNG.integers(lower, upper + 1))

    output_band_starts = anchor_end - green_width + 1 - blue_lens
    if np.any(output_band_starts < 0):
        return None

    caps = (w - 2) - (anchor_end + yellow_lens)
    if int(caps.max()) < 1:
        return None

    deltas = np.zeros(h, dtype=int)
    anchor_count = int(RNG.integers(1, min(4, h)))
    anchor_rows = RNG.choice(h, size=anchor_count, replace=False)
    non_anchor_rows = [r for r in range(h) if r not in set(map(int, anchor_rows))]
    positive_rows = [r for r in range(h) if caps[r] >= 1]
    if not positive_rows:
        return None

    for r in non_anchor_rows:
        cap = int(caps[r])
        deltas[r] = int(RNG.integers(0, cap + 1))

    if not np.any(deltas > 0):
        chosen = int(RNG.choice(positive_rows))
        deltas[chosen] = int(RNG.integers(1, int(caps[chosen]) + 1))

    distinct_deltas = np.unique(deltas)
    if distinct_deltas.size < 2:
        return None

    for _ in range(10):
        if np.unique(deltas).size >= 3:
            break
        adjustable = [r for r in range(h) if caps[r] >= 1]
        if not adjustable:
            break
        r = int(RNG.choice(adjustable))
        deltas[r] = int(RNG.integers(0, int(caps[r]) + 1))
    if np.unique(deltas).size < 2:
        return None

    input_band_starts = output_band_starts + deltas
    input_ends = input_band_starts + blue_lens + green_width + yellow_lens - 1
    if np.any(input_ends >= w):
        return None

    grid = np.zeros((h, w), dtype=int)
    for r in range(h):
        grid[r] = _build_row(
            width=w,
            band_start=int(input_band_starts[r]),
            blue_len=int(blue_lens[r]),
            green_width=green_width,
            yellow_len=int(yellow_lens[r]),
        )

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    try:
        details = _solve_details(g)
    except AssertionError:
        return False

    specs = details["specs"]
    aligned = details["aligned"]
    output = details["output"]
    blank_before = details["blank_before"]
    blank_after = details["blank_after"]
    seed_cols = details["seed_cols"]
    paint_cols = details["paint_cols"]
    anchor_end = int(details["anchor_end"])

    if not np.array_equal(output, main(g)):
        return False

    green_starts = [int(spec["green_start"]) for spec in specs]
    green_ends = [int(spec["green_end"]) for spec in specs]
    blue_lens = [int(spec["blue_len"]) for spec in specs]
    yellow_lens = [int(spec["yellow_len"]) for spec in specs]

    if len(set(green_starts)) < 2 or len(set(green_ends)) < 2:
        return False
    if len(set(blue_lens)) < 2 or len(set(yellow_lens)) < 2:
        return False

    aligned_specs = [_extract_row_spec(aligned[r]) for r in range(aligned.shape[0])]
    if any(int(spec["green_end"]) != anchor_end for spec in aligned_specs):
        return False

    for in_spec, out_spec in zip(specs, aligned_specs):
        if in_spec["blue_len"] != out_spec["blue_len"]:
            return False
        if in_spec["green_width"] != out_spec["green_width"]:
            return False
        if in_spec["yellow_len"] != out_spec["yellow_len"]:
            return False

    if not np.any(seed_cols):
        return False
    if not np.any(paint_cols):
        return False
    if np.array_equal(seed_cols, paint_cols):
        return False
    if not np.any(paint_cols & blank_before):
        return False
    if not np.any(seed_cols & ~blank_before):
        return False
    if np.array_equal(blank_before, paint_cols):
        return False

    if np.any(blank_after & paint_cols):
        pass
    else:
        return False

    repainted = aligned.copy()
    repainted[:, paint_cols] = RED
    if not np.array_equal(repainted, output):
        return False
    if np.any(output[:, ~paint_cols] == RED):
        return False

    if np.any(blank_after & ~paint_cols):
        left_blank = np.flatnonzero(blank_after & ~paint_cols)
        if left_blank.size:
            if not np.all(output[:, left_blank] == BLACK):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid puzzle-10 input")
