from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BG = 0
BAR_COLORS = (1, 2, 3)
YELLOW = 4
GREY = 5

GEN_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_col(grid: np.ndarray) -> int:
    matches = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(matches) == 1, "expected exactly one full-height grey separator column"
    return matches[0]


def _iter_left_bars(grid: np.ndarray, sep_col: int) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    for r in range(grid.shape[0]):
        c = 0
        while c < sep_col:
            color = int(grid[r, c])
            if color not in BAR_COLORS:
                c += 1
                continue
            c0 = c
            while c < sep_col and int(grid[r, c]) == color:
                c += 1
            c1 = c - 1
            length = c1 - c0 + 1
            dist = sep_col - c1 - 1
            bars.append((dist, r, c0, c1, length, color))
    bars.sort(key=lambda item: (item[0], item[1], item[2]))
    return bars


def _simulate_candidate_layer(grid: np.ndarray, sep_col: int) -> np.ndarray:
    candidate = np.zeros_like(grid)
    for _dist, r, _c0, _c1, length, color in _iter_left_bars(grid, sep_col):
        for c in range(sep_col + 1, grid.shape[1], length):
            if int(grid[r, c]) == BG and int(candidate[r, c]) == BG:
                candidate[r, c] = color
    return candidate


def _template_candidates(grid: np.ndarray, sep_col: int) -> list[tuple[int, int, np.ndarray]]:
    candidates: list[tuple[int, int, np.ndarray]] = []
    for r in range(grid.shape[0] - 2):
        for c in range(sep_col + 1, grid.shape[1] - 2):
            window = grid[r : r + 3, c : c + 3]
            if np.all(window != BG) and np.any(window == YELLOW) and np.all(np.isin(window, BAR_COLORS + (YELLOW,))):
                candidates.append((r, c, window.copy()))
    return candidates


def _find_template(grid: np.ndarray, sep_col: int) -> tuple[tuple[int, int], np.ndarray]:
    candidates = _template_candidates(grid, sep_col)
    assert len(candidates) == 1, "expected exactly one complete 3x3 template"
    r, c, template = candidates[0]
    return (r, c), template


def _find_copy_origins(
    grid: np.ndarray, sep_col: int, template: np.ndarray, template_origin: tuple[int, int]
) -> list[tuple[int, int]]:
    yellow_mask = template == YELLOW
    missing_mask = ~yellow_mask
    copies: list[tuple[int, int]] = []
    for r in range(grid.shape[0] - 2):
        for c in range(sep_col + 1, grid.shape[1] - 2):
            if (r, c) == template_origin:
                continue
            window = grid[r : r + 3, c : c + 3]
            if np.array_equal(window[yellow_mask], template[yellow_mask]) and np.all(window[missing_mask] == BG):
                copies.append((r, c))
    return copies


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partial 3x3 copy only where periodic candidate ink agrees with the template color.

    Input structure:
    - A black background is split by one full-height grey separator column; the left panel contains 2-3 horizontal bars in blue/red/green.
    - Each bar is a monochrome contiguous run on its own row, and its length is the period for rightward extension.
    - The right panel contains one complete 3x3 template with yellow anchor cells plus 2-3 lower partial copies that keep only those yellow cells.

    Rule:
    - Read every left-panel bar as (row, color, length, distance-to-separator) and simulate its periodic extension on a separate candidate layer from the first cell right of the separator, nearest bars first, without covering non-background cells.
    - Locate the unique complete 3x3 template on the right and record its full color layout.
    - Find every 3x3 partial copy whose yellow cells match the template exactly and whose other template positions are black.
    - For each missing copy cell, paint the template color only if the candidate layer covers that exact cell with the same color; otherwise leave it black.

    Key insight:
    - Periodic extension supplies possible ink, but template matching validates both location and color, so a missing cell changes only when geometry and template semantics agree.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_col = _find_separator_col(g)
    candidate = _simulate_candidate_layer(g, sep_col)
    template_origin, template = _find_template(g, sep_col)
    copies = _find_copy_origins(g, sep_col, template, template_origin)
    assert 2 <= len(copies) <= 3, "expected 2-3 incomplete copies"

    out = g.copy()
    for r0, c0 in copies:
        for dr in range(3):
            for dc in range(3):
                expected = int(template[dr, dc])
                if expected in BAR_COLORS and int(out[r0 + dr, c0 + dc]) == BG:
                    if int(candidate[r0 + dr, c0 + dc]) == expected:
                        out[r0 + dr, c0 + dc] = expected
    return out


def _sample_template(
    active_rows: tuple[int, ...], row_bar_colors: dict[int, int]
) -> np.ndarray | None:
    template = np.zeros((3, 3), dtype=int)
    yellow_cols = list(range(3))
    random.shuffle(yellow_cols)
    for r in range(3):
        y = yellow_cols[r]
        template[r, y] = YELLOW
        others = [c for c in range(3) if c != y]
        if r in active_rows:
            match_color = row_bar_colors[r]
            other_color = random.choice([color for color in BAR_COLORS if color != match_color])
            row_colors = [match_color, other_color]
        else:
            row_colors = random.sample(list(BAR_COLORS), 2)
        random.shuffle(row_colors)
        template[r, others[0]] = row_colors[0]
        template[r, others[1]] = row_colors[1]
    non_yellow = template[template != YELLOW]
    if len(set(map(int, non_yellow))) < 3:
        return None
    return template


def _copy_origin_tuples(sep_col: int, width: int, count: int) -> list[tuple[int, ...]]:
    origins = list(range(sep_col + 1, width - 2))
    return [
        combo
        for combo in combinations(origins, count)
        if all(combo[i + 1] - combo[i] >= 5 for i in range(len(combo) - 1))
    ]


def _coverage_status(
    sep_col: int,
    template: np.ndarray,
    copy_cols: tuple[int, ...],
    active_rows: tuple[int, ...],
    row_lengths: dict[int, int],
    row_bar_colors: dict[int, int],
) -> tuple[list[str], dict[int, list[str]], list[tuple[int, ...]]]:
    statuses: list[str] = []
    by_row = {r: [] for r in range(3)}
    signatures: list[tuple[int, ...]] = []
    for c0 in copy_cols:
        signature: list[int] = []
        for r in range(3):
            for dc in range(3):
                color = int(template[r, dc])
                if color == YELLOW:
                    continue
                status = "none"
                if r in active_rows:
                    length = row_lengths[r]
                    covered = (c0 + dc - (sep_col + 1)) % length == 0
                    if covered:
                        status = "correct" if row_bar_colors[r] == color else "wrong"
                statuses.append(status)
                by_row[r].append(status)
                signature.append(1 if status == "correct" else 0)
        signatures.append(tuple(signature))
    return statuses, by_row, signatures


def _semantic_layout_ok(
    sep_col: int,
    template: np.ndarray,
    copy_cols: tuple[int, ...],
    active_rows: tuple[int, ...],
    row_lengths: dict[int, int],
    row_bar_colors: dict[int, int],
) -> bool:
    statuses, by_row, signatures = _coverage_status(
        sep_col, template, copy_cols, active_rows, row_lengths, row_bar_colors
    )
    if statuses.count("correct") < max(2, len(active_rows)):
        return False
    if "wrong" not in statuses or "none" not in statuses:
        return False
    if len(set(signatures)) < 2:
        return False
    for r in active_rows:
        row_stats = by_row[r]
        if "correct" not in row_stats:
            return False
        if not any(status != "correct" for status in row_stats):
            return False
    for r in range(3):
        if r not in active_rows and any(status != "none" for status in by_row[r]):
            return False
    return True


def _choose_copy_layout(
    sep_col: int,
    width: int,
    template: np.ndarray,
    active_rows: tuple[int, ...],
    row_lengths: dict[int, int],
    row_bar_colors: dict[int, int],
) -> tuple[int, tuple[int, ...]] | None:
    copy_count_options = [2, 3]
    random.shuffle(copy_count_options)
    for copy_count in copy_count_options:
        candidates = _copy_origin_tuples(sep_col, width, copy_count)
        random.shuffle(candidates)
        for copy_cols in candidates:
            if _semantic_layout_ok(sep_col, template, copy_cols, active_rows, row_lengths, row_bar_colors):
                return copy_count, copy_cols
    return None


def _count_template_and_copies(
    grid: np.ndarray,
) -> tuple[int, int]:
    sep_col = _find_separator_col(grid)
    template_candidates = _template_candidates(grid, sep_col)
    if len(template_candidates) != 1:
        return len(template_candidates), -1
    template_origin = (template_candidates[0][0], template_candidates[0][1])
    template = template_candidates[0][2]
    copies = _find_copy_origins(grid, sep_col, template, template_origin)
    return 1, len(copies)


def _build_grid(
    height: int,
    width: int,
    sep_col: int,
    template_row: int,
    template_col: int,
    copy_row: int,
    copy_cols: tuple[int, ...],
    template: np.ndarray,
    active_rows: tuple[int, ...],
    row_lengths: dict[int, int],
    row_bar_colors: dict[int, int],
    row_distances: dict[int, int],
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[:, sep_col] = GREY

    for rel_r in active_rows:
        length = row_lengths[rel_r]
        color = row_bar_colors[rel_r]
        dist = row_distances[rel_r]
        abs_r = copy_row + rel_r
        c1 = sep_col - dist - 1
        c0 = c1 - length + 1
        grid[abs_r, c0 : c1 + 1] = color

    grid[template_row : template_row + 3, template_col : template_col + 3] = template
    yellow_mask = template == YELLOW
    for c0 in copy_cols:
        window = np.zeros((3, 3), dtype=int)
        window[yellow_mask] = YELLOW
        grid[copy_row : copy_row + 3, c0 : c0 + 3] = window
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    sep_col: int,
    template: np.ndarray,
    template_origin: tuple[int, int],
    copy_origins: tuple[tuple[int, int], ...],
    active_rows: tuple[int, ...],
    row_lengths: dict[int, int],
    row_bar_colors: dict[int, int],
) -> bool:
    if max(grid.shape) > 30:
        return False

    found_sep = _find_separator_col(grid)
    if found_sep != sep_col:
        return False

    bars = _iter_left_bars(grid, sep_col)
    if len(bars) != len(active_rows):
        return False
    if len({bar[1] for bar in bars}) != len(bars):
        return False

    t_candidates, copy_count = _count_template_and_copies(grid)
    if t_candidates != 1 or copy_count != len(copy_origins):
        return False

    found_origin, found_template = _find_template(grid, sep_col)
    if found_origin != template_origin or not np.array_equal(found_template, template):
        return False

    found_copies = tuple(sorted(_find_copy_origins(grid, sep_col, template, template_origin)))
    if found_copies != tuple(sorted(copy_origins)):
        return False

    statuses, by_row, _signatures = _coverage_status(
        sep_col,
        template,
        tuple(c for _r, c in sorted(copy_origins)),
        active_rows,
        row_lengths,
        row_bar_colors,
    )
    if "correct" not in statuses or "wrong" not in statuses or "none" not in statuses:
        return False

    out = main(grid)
    expected = grid.copy()
    for r0, c0 in copy_origins:
        for dr in range(3):
            for dc in range(3):
                color = int(template[dr, dc])
                if color == YELLOW:
                    continue
                status = "none"
                if dr in active_rows:
                    covered = (c0 + dc - (sep_col + 1)) % row_lengths[dr] == 0
                    if covered:
                        status = "correct" if row_bar_colors[dr] == color else "wrong"
                if status == "correct":
                    expected[r0 + dr, c0 + dc] = color
    if not np.array_equal(out, expected):
        return False

    if np.array_equal(out, grid):
        return False

    changed = np.argwhere(out != grid)
    if changed.size == 0:
        return False
    for r, c in changed:
        if not any(r0 <= r < r0 + 3 and c0 <= c < c0 + 3 for r0, c0 in copy_origins):
            return False

    for r in active_rows:
        if "correct" not in by_row[r]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 15)
    width = random.randint(18, 23)
    sep_col = random.randint(6, 7)

    template_row = random.randint(1, 2)
    min_copy_row = template_row + 5
    if min_copy_row > height - 3:
        return None
    copy_row = random.randint(min_copy_row, height - 3)

    bar_count = 3 if random.random() < 0.6 else 2
    active_rows = tuple(sorted(random.sample(range(3), bar_count)))

    active_palette = random.sample(list(BAR_COLORS), len(active_rows))
    row_bar_colors = {r: color for r, color in zip(active_rows, active_palette)}

    row_distances = {}
    for r, dist in zip(active_rows, random.sample([1, 2, 3], len(active_rows))):
        row_distances[r] = dist

    row_lengths: dict[int, int] = {}
    for r in active_rows:
        max_length = min(4, sep_col - row_distances[r])
        if max_length < 2:
            return None
        row_lengths[r] = random.randint(2, max_length)
    if len(set(row_lengths.values())) == 1 and len(row_lengths) > 1 and random.random() < 0.5:
        return None

    template = _sample_template(active_rows, row_bar_colors)
    if template is None:
        return None

    layout = _choose_copy_layout(sep_col, width, template, active_rows, row_lengths, row_bar_colors)
    if layout is None:
        return None
    copy_count, copy_cols = layout

    template_col_options = list(range(sep_col + 1, width - 2))
    random.shuffle(template_col_options)
    for template_col in template_col_options:
        grid = _build_grid(
            height,
            width,
            sep_col,
            template_row,
            template_col,
            copy_row,
            copy_cols,
            template,
            active_rows,
            row_lengths,
            row_bar_colors,
            row_distances,
        )
        template_origin = (template_row, template_col)
        copy_origins = tuple((copy_row, c) for c in copy_cols)
        if _validate_generated_case(
            grid,
            sep_col,
            template,
            template_origin,
            copy_origins,
            active_rows,
            row_lengths,
            row_bar_colors,
        ):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-template ARC input")
