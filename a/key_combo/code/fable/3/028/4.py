from __future__ import annotations

from collections import Counter
import random

import numpy as np


GEN_RETRIES = 500
COLOR_VALUES = tuple(range(10))
PANEL_COUNT_CHOICES = (3, 4, 5)
ROW_RANGE = (8, 13)
MAX_CONTENT_ROWS_PER_PANEL = 3


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _separator_columns(grid: np.ndarray, bg: int) -> tuple[int, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int]]] = []

    for color in _colors_present(grid, exclude={bg}):
        mask = grid == int(color)
        full_cols = [c for c in range(w) if bool(np.all(mask[:, c]))]
        if not full_cols:
            continue
        if int(mask.sum()) != h * len(full_cols):
            continue

        panel_w = int(full_cols[0])
        if panel_w <= 0:
            continue
        expected = [panel_w + i * (panel_w + 1) for i in range(len(full_cols))]
        if full_cols != expected:
            continue
        if w != (len(full_cols) + 1) * panel_w + len(full_cols):
            continue
        candidates.append((int(color), full_cols))

    assert len(candidates) == 1, "expected one divider color forming equal-width panels"
    return candidates[0]


def _split_panels(grid: np.ndarray, seps: list[int]) -> list[np.ndarray]:
    starts = [0, *[c + 1 for c in seps]]
    stops = [*seps, grid.shape[1]]
    panels = [grid[:, c0:c1] for c0, c1 in zip(starts, stops)]
    widths = {int(panel.shape[1]) for panel in panels}
    assert len(widths) == 1, "panels must have equal width"
    return panels


def _half_for_marker_rows(rows: np.ndarray, height: int) -> str:
    if bool(np.all(rows < height // 2)):
        return "upper"
    if bool(np.all(rows >= (height + 1) // 2)):
        return "lower"
    raise AssertionError("markers must lie entirely in one strict half")


def _marker_info(panels: list[np.ndarray], bg: int) -> tuple[int, list[int], list[str]]:
    height = int(panels[0].shape[0])
    panel_count = len(panels)
    target_counts = list(range(1, panel_count + 1))
    colors = sorted(
        set().union(*(set(map(int, np.unique(panel))) for panel in panels)) - {int(bg)}
    )
    candidates: list[tuple[int, list[int], list[str]]] = []

    for color in colors:
        counts: list[int] = []
        halves: list[str] = []
        ok = True
        for panel in panels:
            coords = np.argwhere(panel == int(color))
            counts.append(int(coords.shape[0]))
            if coords.shape[0] == 0:
                ok = False
                break
            try:
                halves.append(_half_for_marker_rows(coords[:, 0], height))
            except AssertionError:
                ok = False
                break
        if ok and sorted(counts) == target_counts:
            candidates.append((int(color), counts, halves))

    assert len(candidates) == 1, "expected one marker color with per-panel counts 1..n"
    return candidates[0]


def _read_rows(
    panels: list[np.ndarray],
    bg: int,
    marker: int,
    marker_counts: list[int],
    halves: list[str],
    *,
    order: list[int] | None = None,
    strip_markers: bool = True,
    use_half_directions: bool = True,
) -> list[np.ndarray]:
    height = int(panels[0].shape[0])
    if order is None:
        order = sorted(range(len(panels)), key=lambda i: marker_counts[i])

    rows: list[np.ndarray] = []
    for panel_index in order:
        panel = panels[panel_index].copy()
        if strip_markers:
            panel[panel == int(marker)] = int(bg)

        if use_half_directions and halves[panel_index] == "lower":
            row_order = range(height - 1, -1, -1)
        else:
            row_order = range(height)

        for r in row_order:
            row = panel[r, :]
            if bool(np.any(row != int(bg))):
                rows.append(row.copy())
    return rows


def _render_rows(rows: list[np.ndarray], height: int, width: int, bg: int) -> np.ndarray:
    assert len(rows) <= height, "packed rows must fit in one panel"
    out = np.full((height, width), int(bg), dtype=int)
    if rows:
        out[: len(rows), :] = np.asarray(rows, dtype=int)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pack panel rows by marker count order and marker-half row direction.

    Input structure:
    - A rectangular grid is divided by full-height divider columns into several equal-width panels.
    - The divider color appears only in those divider columns, and the background is the dominant color.
    - Each panel contains content pixels plus a dedicated marker color; the panel read as kth has exactly k markers.
    - All markers in a panel lie strictly in either its upper half or its lower half.

    Rule:
    - Find the equal-width panels, then remove every marker pixel before testing whether panel rows are occupied.
    - Read panels in increasing order of their marker counts.
    - Read occupied rows top-to-bottom for panels whose markers are in the upper half, and bottom-to-top for panels whose markers are in the lower half.
    - Place the read rows from the top of a single panel-shaped output and fill all remaining rows with background.

    Key insight:
    - The same marker pixels encode both rank and scan direction, while marker-only rows are decoys that disappear before row occupancy is judged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _divider, seps = _separator_columns(g, bg)
    panels = _split_panels(g, seps)
    height, panel_w = panels[0].shape

    marker, marker_counts, halves = _marker_info(panels, bg)
    rows = _read_rows(panels, bg, marker, marker_counts, halves)
    assert rows, "expected at least one content row after removing markers"
    return _render_rows(rows, int(height), int(panel_w), bg)


def _choose_colors() -> tuple[int, int, int, list[int]]:
    bg = random.choice(COLOR_VALUES)
    remaining = [c for c in COLOR_VALUES if c != bg]
    divider, marker = random.sample(remaining, 2)
    content_pool = [c for c in remaining if c not in (divider, marker)]
    return bg, divider, marker, random.sample(content_pool, random.randint(3, min(5, len(content_pool))))


def _content_counts(panel_count: int, height: int, halves: list[str]) -> list[int]:
    max_total = min(height - 1, panel_count * MAX_CONTENT_ROWS_PER_PANEL)
    min_total = panel_count + 1
    if min_total > max_total:
        raise ValueError("not enough panel height for nondegenerate content")

    total = random.randint(min_total, max_total)
    counts = [1] * panel_count
    lower_indices = [i for i, half in enumerate(halves) if half == "lower"]
    forced = random.choice(lower_indices)
    counts[forced] += 1
    remaining = total - panel_count - 1

    while remaining > 0:
        choices = [i for i, value in enumerate(counts) if value < MAX_CONTENT_ROWS_PER_PANEL]
        if not choices:
            break
        idx = random.choice(choices)
        counts[idx] += 1
        remaining -= 1
    return counts


def _row_pattern(width: int, bg: int, colors: list[int], used: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        row = np.full(width, int(bg), dtype=int)
        segment_count = random.choice((1, 1, 2))
        occupied: set[int] = set()
        for _seg in range(segment_count):
            seg_len = random.randint(1, max(1, min(3, width - 1)))
            start = random.randint(0, width - seg_len)
            color = random.choice(colors)
            for c in range(start, start + seg_len):
                row[c] = color
                occupied.add(c)
        if not occupied:
            continue
        key = tuple(map(int, row.tolist()))
        if key not in used:
            used.add(key)
            return row

    for color in colors:
        for c in range(width):
            row = np.full(width, int(bg), dtype=int)
            row[c] = int(color)
            key = tuple(map(int, row.tolist()))
            if key not in used:
                used.add(key)
                return row
    raise RuntimeError("failed to make a distinct row pattern")


def _build_panel(
    height: int,
    width: int,
    bg: int,
    marker: int,
    marker_count: int,
    half: str,
    content_count: int,
    content_colors: list[int],
    used_rows: set[tuple[int, ...]],
) -> np.ndarray:
    panel = np.full((height, width), int(bg), dtype=int)
    half_rows = list(range(0, height // 2)) if half == "upper" else list(range((height + 1) // 2, height))
    trap_row = random.choice(half_rows)
    for c in random.sample(range(width), marker_count):
        panel[trap_row, c] = int(marker)

    content_candidates = [r for r in range(height) if r != trap_row]
    content_rows = random.sample(content_candidates, content_count)
    for r in content_rows:
        panel[r, :] = _row_pattern(width, bg, content_colors, used_rows)
    return panel


def _expected_from_metadata(
    panels: list[np.ndarray],
    bg: int,
    marker: int,
    marker_counts: list[int],
    halves: list[str],
) -> np.ndarray:
    height, width = panels[0].shape
    rows = _read_rows(panels, bg, marker, marker_counts, halves)
    return _render_rows(rows, int(height), int(width), bg)


def _semantic_accepts(
    grid: np.ndarray,
    expected: np.ndarray,
    marker: int,
    marker_counts: list[int],
    halves: list[str],
) -> bool:
    try:
        bg = _get_bg(grid)
        _divider, seps = _separator_columns(grid, bg)
        panels = _split_panels(grid, seps)
        found_marker, found_counts, found_halves = _marker_info(panels, bg)
        actual = main(grid)
    except Exception:
        return False

    if found_marker != int(marker) or found_counts != marker_counts or found_halves != halves:
        return False
    if not np.array_equal(actual, expected):
        return False

    height, width = panels[0].shape
    correct_rows = _read_rows(panels, bg, marker, marker_counts, halves)
    if not (0 < len(correct_rows) < height):
        return False
    if len({tuple(map(int, row.tolist())) for row in correct_rows}) != len(correct_rows):
        return False

    correct_order = sorted(range(len(panels)), key=lambda i: marker_counts[i])
    if correct_order == list(range(len(panels))):
        return False
    natural = _read_rows(
        panels,
        bg,
        marker,
        marker_counts,
        halves,
        order=list(range(len(panels))),
    )
    if np.array_equal(_render_rows(natural, int(height), int(width), bg), expected):
        return False

    if "upper" not in halves or "lower" not in halves:
        return False
    all_top = _read_rows(
        panels,
        bg,
        marker,
        marker_counts,
        halves,
        strip_markers=True,
        use_half_directions=False,
    )
    if np.array_equal(_render_rows(all_top, int(height), int(width), bg), expected):
        return False

    no_strip = _read_rows(
        panels,
        bg,
        marker,
        marker_counts,
        halves,
        strip_markers=False,
        use_half_directions=True,
    )
    if len(no_strip) <= len(correct_rows):
        return False
    marker_only_rows = [
        row
        for panel in panels
        for row in panel
        if bool(np.any(row == int(marker))) and not bool(np.any((row != int(bg)) & (row != int(marker))))
    ]
    if not marker_only_rows:
        return False

    lower_multi = [
        i
        for i, half in enumerate(halves)
        if half == "lower"
        and sum(bool(np.any((panels[i][r, :] != bg) & (panels[i][r, :] != marker))) for r in range(height)) >= 2
    ]
    return bool(lower_multi)


def _gen_attempt() -> np.ndarray | None:
    panel_count = random.choice(PANEL_COUNT_CHOICES)
    height = random.randint(*ROW_RANGE)
    max_width = (30 - (panel_count - 1)) // panel_count
    min_width = max(4, panel_count)
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    bg, divider, marker, content_colors = _choose_colors()
    marker_counts = random.sample(range(1, panel_count + 1), panel_count)
    if marker_counts == list(range(1, panel_count + 1)):
        return None

    halves = [random.choice(("upper", "lower")) for _ in range(panel_count)]
    if "upper" not in halves:
        halves[random.randrange(panel_count)] = "upper"
    if "lower" not in halves:
        halves[random.randrange(panel_count)] = "lower"

    try:
        content_counts = _content_counts(panel_count, height, halves)
    except ValueError:
        return None

    used_rows: set[tuple[int, ...]] = set()
    panels = [
        _build_panel(
            height,
            width,
            bg,
            marker,
            marker_counts[i],
            halves[i],
            content_counts[i],
            content_colors,
            used_rows,
        )
        for i in range(panel_count)
    ]

    sep_col = np.full((height, 1), int(divider), dtype=int)
    pieces: list[np.ndarray] = []
    for i, panel in enumerate(panels):
        if i:
            pieces.append(sep_col)
        pieces.append(panel)
    grid = np.hstack(pieces)

    expected = _expected_from_metadata(panels, bg, marker, marker_counts, halves)
    if not _semantic_accepts(grid, expected, marker, marker_counts, halves):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-rank packing input")
