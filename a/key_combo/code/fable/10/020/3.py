from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int]] = []
    for c in range(w):
        col = grid[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and np.all(vals == vals[0]):
            candidates.append((int(vals.size), c, int(vals[0])))
    if not candidates:
        return None, None
    candidates.sort(reverse=True)
    _mass, sep_col, sep_color = candidates[0]
    return int(sep_col), int(sep_color)


def _choose_echo_side(grid: np.ndarray, sep_col: int, bg: int) -> str:
    left_mass = int(np.count_nonzero(grid[:, :sep_col] != bg))
    right_mass = int(np.count_nonzero(grid[:, sep_col + 1 :] != bg))
    return "right" if right_mass >= left_mass else "left"


def _distance_from_separator(sep_col: int, c: int) -> int:
    return abs(int(c) - int(sep_col))


def _collect_echo_data(
    grid: np.ndarray, sep_col: int, sep_color: int, echo_side: str, bg: int
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, list[int]]]:
    h, w = grid.shape
    color_to_distances: dict[int, list[int]] = defaultdict(list)
    row_to_color: dict[int, int] = {}
    row_to_distances: dict[int, list[int]] = {}

    if echo_side == "right":
        c_start, c_stop = sep_col + 1, w
    else:
        c_start, c_stop = 0, sep_col

    for r in range(h):
        entries: list[tuple[int, int]] = []
        for c in range(c_start, c_stop):
            color = int(grid[r, c])
            if color in (bg, sep_color):
                continue
            dist = _distance_from_separator(sep_col, c)
            entries.append((c, dist))
            color_to_distances[color].append(dist)

        if not entries:
            continue

        colors = [int(grid[r, c]) for c, _dist in entries]
        row_color = Counter(colors).most_common(1)[0][0]
        row_to_color[r] = int(row_color)
        row_to_distances[r] = sorted(
            dist for c, dist in entries if int(grid[r, c]) == int(row_color)
        )

    return color_to_distances, row_to_color, row_to_distances


def _periods_from_distances(color_to_distances: dict[int, list[int]]) -> dict[int, int]:
    periods: dict[int, int] = {}
    for color, distances in color_to_distances.items():
        period = 0
        for dist in distances:
            period = math.gcd(period, int(dist))
        if period > 0:
            periods[int(color)] = int(period)
    return periods


def _paint_full_echoes(
    out: np.ndarray, row: int, color: int, period: int, sep_col: int, echo_side: str
) -> None:
    h, w = out.shape
    del h
    if echo_side == "right":
        for dist in range(period, w - sep_col, period):
            out[row, sep_col + dist] = color
    else:
        for dist in range(period, sep_col + 1, period):
            out[row, sep_col - dist] = color


def _paint_bar(
    out: np.ndarray, row: int, color: int, period: int, sep_col: int, echo_side: str
) -> None:
    if echo_side == "right":
        out[row, sep_col - period : sep_col] = color
    else:
        out[row, sep_col + 1 : sep_col + 1 + period] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each bar length from sparse periodic echoes, then reconstruct the missing bar and the full echo train.

    Input structure:
    - A near-solid vertical separator column splits the canvas into a blank bar side and an echo side.
    - The bar side has been erased to background; the echo side keeps sparse colored pixels on some rows.
    - All echo pixels of one color lie at distances k×L from the separator for that color's hidden bar length L.

    Rule:
    - Find the separator column and treat the non-empty side as the echo side.
    - For each color, gather every echo distance to the separator and take their greatest common divisor as L.
    - For every row that contains echoes, rebuild a same-color bar of length L flush against the separator on the blank side.
    - Also fill that row's echo side at every multiple of L until the canvas edge.

    Key insight:
    - A row may show only a late beat or even a single beat, so the period is recoverable only by pooling same-color distances across rows and taking their gcd.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_col, sep_color = _find_separator(g, bg)
    if sep_col is None:
        return g.copy()

    echo_side = _choose_echo_side(g, sep_col, bg)
    color_to_distances, row_to_color, _row_to_distances = _collect_echo_data(
        g, sep_col, int(sep_color), echo_side, bg
    )
    periods = _periods_from_distances(color_to_distances)

    out = g.copy()
    for row, color in row_to_color.items():
        period = periods.get(int(color))
        if period is None or period <= 0:
            continue
        _paint_bar(out, int(row), int(color), int(period), int(sep_col), echo_side)
        _paint_full_echoes(out, int(row), int(color), int(period), int(sep_col), echo_side)
    return out


def _sample_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_anchor_beats(rng: np.random.Generator, max_k: int) -> list[int]:
    candidates = [
        [2, 3],
        [1, 2],
        [1, 3],
        [1, 2, 3],
        [2, 3, 4],
        [2, 4, 5],
        [1, 4],
        [3, 4],
    ]
    valid = [beats for beats in candidates if max(beats) <= max_k and math.gcd(*beats) == 1]
    choice = valid[int(rng.integers(0, len(valid)))]
    return list(choice)


def _sample_other_beats(
    rng: np.random.Generator, max_k: int, *, singleton_bias: bool
) -> list[int]:
    if singleton_bias or rng.random() < 0.5:
        return [int(rng.integers(1, max_k + 1))]

    width = int(rng.integers(2, min(4, max_k) + 1))
    beats = sorted(map(int, rng.choice(np.arange(1, max_k + 1), size=width, replace=False)))
    if len(beats) == max_k:
        drop = int(rng.integers(0, len(beats)))
        beats.pop(drop)
    return beats


def _build_output_from_specs(
    h: int,
    w: int,
    sep_col: int,
    sep_color: int,
    echo_side: str,
    gap_row: int | None,
    row_specs: list[dict[str, int | list[int]]],
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color
    if gap_row is not None:
        out[gap_row, sep_col] = 0

    for spec in row_specs:
        row = int(spec["row"])
        color = int(spec["color"])
        period = int(spec["period"])
        _paint_bar(out, row, color, period, sep_col, echo_side)
        _paint_full_echoes(out, row, color, period, sep_col, echo_side)
    return out


def _build_input_from_specs(
    h: int,
    w: int,
    sep_col: int,
    sep_color: int,
    echo_side: str,
    gap_row: int | None,
    row_specs: list[dict[str, int | list[int]]],
) -> np.ndarray:
    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color
    if gap_row is not None:
        inp[gap_row, sep_col] = 0

    for spec in row_specs:
        row = int(spec["row"])
        color = int(spec["color"])
        period = int(spec["period"])
        beats = list(map(int, spec["beats"]))
        for k in beats:
            dist = k * period
            c = sep_col + dist if echo_side == "right" else sep_col - dist
            inp[row, c] = color
    return inp


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    latent_periods: dict[int, int],
    row_specs: list[dict[str, int | list[int]]],
    sep_col: int,
    sep_color: int,
    echo_side: str,
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    bg = _get_bg(inp)
    found_sep_col, found_sep_color = _find_separator(inp, bg)
    if found_sep_col != sep_col or found_sep_color != sep_color:
        return False
    if _choose_echo_side(inp, sep_col, bg) != echo_side:
        return False

    color_to_distances, row_to_color, row_to_distances = _collect_echo_data(
        inp, sep_col, sep_color, echo_side, bg
    )
    inferred_periods = _periods_from_distances(color_to_distances)
    if inferred_periods != latent_periods:
        return False

    has_singleton_row = False
    has_missing_first = False
    for spec in row_specs:
        row = int(spec["row"])
        color = int(spec["color"])
        beats = sorted(map(int, spec["beats"]))
        dists = sorted(map(int, row_to_distances.get(row, [])))
        want = [int(color) == int(row_to_color.get(row, -1)), dists == [b * latent_periods[color] for b in beats]]
        if not all(want):
            return False
        if len(beats) == 1:
            has_singleton_row = True
        if beats[0] > 1:
            has_missing_first = True

    if not has_singleton_row or not has_missing_first:
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(8, 16))
    max_active = (h + 1) // 2
    if max_active < 4:
        return None

    bar_width = int(rng.integers(3, 8))
    echo_width = int(rng.integers(9, 18))
    w = bar_width + 1 + echo_width
    if w > 30:
        return None

    period_pool = list(range(2, min(bar_width, echo_width // 3) + 1))
    if len(period_pool) < 2:
        return None

    active_count = int(rng.integers(4, min(7, max_active) + 1))
    n_colors = int(rng.integers(2, min(4, active_count - 1, len(period_pool)) + 1))

    echo_side = "right" if rng.random() < 0.5 else "left"
    sep_col = bar_width if echo_side == "right" else w - 1 - bar_width

    palette = list(range(1, 10))
    sep_color = int(rng.choice(palette))
    palette.remove(sep_color)
    color_ids = list(map(int, rng.choice(palette, size=n_colors, replace=False)))
    periods = list(map(int, rng.choice(period_pool, size=n_colors, replace=False)))

    row_counts = [1] * n_colors
    extra_rows = active_count - n_colors
    featured_color = int(rng.integers(0, n_colors))
    row_counts[featured_color] += 1
    extra_rows -= 1
    for _ in range(extra_rows):
        row_counts[int(rng.integers(0, n_colors))] += 1

    active_rows = _sample_active_rows(rng, h, active_count)
    row_cursor = 0
    row_specs: list[dict[str, int | list[int]]] = []
    latent_periods: dict[int, int] = {}

    has_missing_first = False
    has_singleton = False
    for idx, color in enumerate(color_ids):
        period = int(periods[idx])
        latent_periods[color] = period
        max_k = echo_width // period
        if max_k < 3:
            return None

        anchor_beats = _sample_anchor_beats(rng, max_k)
        if anchor_beats[0] > 1:
            has_missing_first = True
        row_specs.append(
            {
                "row": active_rows[row_cursor],
                "color": color,
                "period": period,
                "beats": anchor_beats,
            }
        )
        row_cursor += 1

        remaining = row_counts[idx] - 1
        if remaining <= 0:
            continue

        singleton_slot = int(rng.integers(0, remaining))
        for rem_idx in range(remaining):
            beats = _sample_other_beats(rng, max_k, singleton_bias=rem_idx == singleton_slot)
            if beats[0] > 1:
                has_missing_first = True
            if len(beats) == 1:
                has_singleton = True
            row_specs.append(
                {
                    "row": active_rows[row_cursor],
                    "color": color,
                    "period": period,
                    "beats": beats,
                }
            )
            row_cursor += 1

    if not has_singleton or not has_missing_first:
        return None

    gap_row = None
    inactive_rows = [r for r in range(h) if r not in {int(spec["row"]) for spec in row_specs}]
    if inactive_rows and rng.random() < 0.25:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]

    expected = _build_output_from_specs(
        h, w, sep_col, sep_color, echo_side, gap_row, row_specs
    )
    inp = _build_input_from_specs(
        h, w, sep_col, sep_color, echo_side, gap_row, row_specs
    )

    if not _validate_generated_case(
        inp, expected, latent_periods, row_specs, sep_col, sep_color, echo_side
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
