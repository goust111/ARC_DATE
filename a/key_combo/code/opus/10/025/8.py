from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


MARKER_COLOR = 7
SEPARATOR_COLOR = 5
PAIR_COUNT_RANGE = (3, 4)
MARKER_LEN_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 500
PAIR_GAP_RANGE = (1, 2)
PAIR_HEIGHT_RANGE = (2, 4)
PAIR_WIDTH_RANGE = (4, 7)
SOURCE_AREA_RANGE = (3, 8)
PAIR_COLORS = tuple(c for c in range(10) if c not in {MARKER_COLOR, SEPARATOR_COLOR})
BACKGROUND_CHOICES = (0, 1, 2, 3, 4, 6, 8, 9)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int, row_start: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = sorted(set(map(int, np.unique(g[row_start:, :]))) - {int(bg)})
    comps: list[dict] = []
    for color in colors:
        submask = g[row_start:, :] == int(color)
        for coords in _components_from_mask(submask):
            full = coords.copy()
            full[:, 0] += int(row_start)
            rows = full[:, 0]
            cols = full[:, 1]
            comps.append(
                {
                    "color": int(color),
                    "coords": full,
                    "bbox": (
                        int(rows.min()),
                        int(cols.min()),
                        int(rows.max()),
                        int(cols.max()),
                    ),
                    "area": int(len(full)),
                }
            )
    return comps


def _coord_set(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _parse_upper_structure(grid: np.ndarray, bg: int) -> tuple[int, int, int, Counter[int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)

    separator_rows = [
        r
        for r in range(h)
        if int(g[r, 0]) != int(bg) and np.all(g[r, :] == int(g[r, 0]))
    ]
    assert len(separator_rows) == 1, "expected exactly one solid non-background separator row"
    separator_row = separator_rows[0]
    assert 1 <= separator_row < h - 2, "separator row must leave room above and below"

    upper_mask = g[:separator_row, :] != int(bg)
    marker_rows = np.flatnonzero(upper_mask.any(axis=1))
    assert len(marker_rows) == 1, "expected exactly one marker row above the separator"
    marker_row = int(marker_rows[0])
    marker_cols = np.flatnonzero(upper_mask[marker_row])
    assert len(marker_cols) >= 2, "marker run must contain at least two cells"
    assert np.all(np.diff(marker_cols) == 1), "marker run must be contiguous"
    assert int(upper_mask.sum()) == len(marker_cols), "expected no other foreground above the separator"
    marker_colors = {int(g[marker_row, c]) for c in marker_cols}
    assert marker_colors == {MARKER_COLOR}, "marker run must use the marker color"
    marker_count = len(marker_cols)

    code_row = separator_row + 1
    code_values = [int(v) for v in g[code_row, :] if int(v) != int(bg)]
    assert code_values, "expected a non-empty code row directly below the separator"
    code_counts = Counter(code_values)
    return separator_row, marker_row, code_row, code_counts


def _completion_offsets(
    grid: np.ndarray,
    source: frozenset[tuple[int, int]],
    target: frozenset[tuple[int, int]],
    bg: int,
    lower_start: int,
) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    offsets: list[tuple[int, int]] = []
    for dr in range(1 - h, h):
        for dc in range(1 - w, w):
            moved = {(r + dr, c + dc) for r, c in source}
            if any(r < lower_start or r >= h or c < 0 or c >= w for r, c in moved):
                continue
            if moved & target:
                continue
            if any(int(g[r, c]) != int(bg) for r, c in moved):
                continue
            union = moved | target
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            if len(union) != (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1):
                continue
            offsets.append((int(dr), int(dc)))
    return offsets


def _collect_pairs(grid: np.ndarray, bg: int, lower_start: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    components = _find_components(g, bg=bg, row_start=lower_start)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(comp)

    pairs: list[dict] = []
    for color, comps in sorted(by_color.items()):
        if len(comps) != 2:
            continue
        left = [c for c in comps if int(c["bbox"][1]) == 0]
        anchor = [c for c in comps if int(c["bbox"][1]) > 0]
        if len(left) != 1 or len(anchor) != 1:
            continue
        source = left[0]
        target = anchor[0]
        target_rows = target["coords"][:, 0]
        target_cols = target["coords"][:, 1]
        assert int(target_rows.min()) > lower_start - 1, "anchor must stay in the lower region"
        assert int(target_rows.max()) < h - 1, "anchor must not touch the bottom border"
        assert 0 < int(target_cols.min()) and int(target_cols.max()) < w - 1, "anchor must stay away from side borders"

        source_rows = source["coords"][:, 0]
        source_cols = source["coords"][:, 1]
        assert int(source_cols.min()) == 0, "source fragment must touch the left border"
        assert int(source_rows.min()) > lower_start - 1, "source fragment must stay in the lower region"
        assert int(source_rows.max()) < h - 1, "source fragment must not touch the bottom border"

        offsets = _completion_offsets(
            g,
            _coord_set(source["coords"]),
            _coord_set(target["coords"]),
            bg,
            lower_start,
        )
        assert len(offsets) == 1, "each fragment pair must have a unique rectangle-completion translation"
        pairs.append(
            {
                "color": int(color),
                "source": source,
                "target": target,
                "offset": offsets[0],
            }
        )

    assert len(pairs) >= 2, "expected at least two fragment pairs in the lower region"
    return pairs


def _apply_moves(grid: np.ndarray, pairs: list[dict], active_colors: set[int], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for pair in pairs:
        if int(pair["color"]) not in active_colors:
            continue
        coords = pair["source"]["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    for pair in pairs:
        if int(pair["color"]) not in active_colors:
            continue
        dr, dc = pair["offset"]
        for r, c in pair["source"]["coords"]:
            rr, cc = int(r + dr), int(c + dc)
            assert int(out[rr, cc]) == int(bg), "translated fragment must land on background"
            out[rr, cc] = int(pair["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the marker-run count to gate which same-color fragment pairs complete into rectangles.
    
    Input structure:
    - The grid has a dominant background, one full-width gray separator row, one orange marker run above it, and one code row directly below it.
    - The lower region contains several colors, each split into exactly two 4-connected fragments: one touches the left border and the other is an interior anchor.
    - For every lower-region color, translating the left-edge fragment by one unique offset would complete a solid axis-aligned rectangle with its anchor.
    
    Rule:
    - Count the orange marker cells above the separator to obtain N.
    - In the code row, collect every non-background color whose frequency is exactly N; this set is the activation gate.
    - For each lower-region pair whose color is activated, erase the original left-edge fragment and translate it by the unique rectangle-completing offset.
    - Leave every non-activated pair unchanged, and preserve the entire upper structure.
    
    Key insight:
    - The count/frequency subsystem does not draw anything by itself; it only selects which otherwise-valid rectangle completions are allowed to happen in the separate fragment subsystem.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    separator_row, _marker_row, code_row, code_counts = _parse_upper_structure(g, bg)
    lower_start = code_row + 1
    marker_count = int(np.count_nonzero(g[:separator_row, :] == MARKER_COLOR))
    active_colors = {int(color) for color, count in code_counts.items() if int(count) == marker_count}
    assert active_colors, "expected at least one activated color"

    pairs = _collect_pairs(g, bg, lower_start)
    return _apply_moves(g, pairs, active_colors, bg)


def _sample_source_widths(rect_h: int, rect_w: int) -> list[int] | None:
    for _ in range(200):
        widths = [random.randint(1, rect_w - 1) for _ in range(rect_h)]
        area = sum(widths)
        if not (SOURCE_AREA_RANGE[0] <= area <= SOURCE_AREA_RANGE[1]):
            continue
        if len(set(widths)) == 1:
            continue
        if max(widths) >= rect_w:
            continue
        return widths
    return None


def _sample_counts(
    pair_colors: list[int],
    active_colors: set[int],
    n: int,
    bg: int,
) -> dict[int, int] | None:
    counts: dict[int, int] = {}
    non_n = [v for v in range(1, 7) if v != n]
    for color in pair_colors:
        counts[color] = int(n if color in active_colors else random.choice(non_n))

    filler_pool = [c for c in PAIR_COLORS if c not in pair_colors and c != int(bg)]
    random.shuffle(filler_pool)
    filler_count = random.randint(1, 2)
    for color in filler_pool[:filler_count]:
        counts[color] = int(random.choice(non_n))
    return counts


def _render_code_row(width: int, counts: dict[int, int], bg: int) -> np.ndarray | None:
    total = sum(int(v) for v in counts.values())
    if total > width - 1:
        return None
    row = np.full(width, int(bg), dtype=int)
    cols = random.sample(range(width), total)
    values = [color for color, count in counts.items() for _ in range(int(count))]
    random.shuffle(values)
    for col, color in zip(cols, values):
        row[col] = int(color)
    return row


def _build_pair_shapes(
    top: int,
    width: int,
    color: int,
) -> dict | None:
    rect_h = random.randint(*PAIR_HEIGHT_RANGE)
    rect_w = random.randint(*PAIR_WIDTH_RANGE)
    widths = _sample_source_widths(rect_h, rect_w)
    if widths is None:
        return None
    min_c0 = max(widths) + 3
    max_c0 = width - rect_w - 2
    if min_c0 > max_c0:
        return None
    rect_c0 = random.randint(min_c0, max_c0)

    source_cells: list[tuple[int, int]] = []
    target_cells: list[tuple[int, int]] = []
    for dr, left_width in enumerate(widths):
        r = top + dr
        for dc in range(left_width):
            source_cells.append((r, dc))
        for c in range(rect_c0 + left_width, rect_c0 + rect_w):
            target_cells.append((r, c))

    return {
        "color": int(color),
        "rect_h": int(rect_h),
        "rect_w": int(rect_w),
        "rect_c0": int(rect_c0),
        "source": np.asarray(source_cells, dtype=int),
        "target": np.asarray(target_cells, dtype=int),
        "offset": (0, int(rect_c0)),
    }


def _paint_cells(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    pair_specs: list[dict],
    active_colors: set[int],
    bg: int,
    marker_len: int,
) -> bool:
    try:
        separator_row, _marker_row, code_row, code_counts = _parse_upper_structure(grid, bg)
    except AssertionError:
        return False
    if separator_row + 1 != code_row:
        return False
    recovered_active = {int(color) for color, count in code_counts.items() if int(count) == marker_len}
    if recovered_active != active_colors:
        return False

    try:
        pairs = _collect_pairs(grid, bg, code_row + 1)
    except AssertionError:
        return False
    if len(pairs) != len(pair_specs):
        return False

    intended_offsets = {spec["color"]: tuple(spec["offset"]) for spec in pair_specs}
    for pair in pairs:
        if tuple(pair["offset"]) != intended_offsets[int(pair["color"])]:
            return False

    if not np.array_equal(main(grid), expected):
        return False

    move_all = _apply_moves(grid, pairs, {int(pair["color"]) for pair in pairs}, bg)
    if np.array_equal(move_all, expected):
        return False

    move_none = _apply_moves(grid, pairs, set(), bg)
    if np.array_equal(move_none, expected):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BACKGROUND_CHOICES))
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    marker_len = random.randint(*MARKER_LEN_RANGE)
    pair_colors = random.sample([c for c in PAIR_COLORS if c != bg], pair_count)
    active_count = random.randint(1, pair_count - 1)
    active_colors = set(random.sample(pair_colors, active_count))
    counts = _sample_counts(pair_colors, active_colors, marker_len, bg)

    min_width = max(18, sum(counts.values()) + 2)
    if min_width > 26:
        return None
    width = random.randint(min_width, 26)
    code_row = _render_code_row(width, counts, bg)
    if code_row is None:
        return None

    separator_row = random.randint(4, 6)
    marker_row = random.randint(0, separator_row - 2)
    fragment_start = separator_row + 3

    pair_specs: list[dict] = []
    pair_tops: list[int] = []
    current_top = fragment_start + random.randint(0, 1)
    for idx, color in enumerate(pair_colors):
        spec = _build_pair_shapes(current_top, width, int(color))
        if spec is None:
            return None
        pair_specs.append(spec)
        pair_tops.append(current_top)
        gap = random.randint(*PAIR_GAP_RANGE) if idx < pair_count - 1 else random.randint(1, 2)
        current_top += int(spec["rect_h"]) + gap

    height = current_top + 1
    if not (16 <= height <= 30):
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    marker_start = random.randint(0, width - marker_len)
    grid[marker_row, marker_start : marker_start + marker_len] = MARKER_COLOR
    grid[separator_row, :] = SEPARATOR_COLOR
    grid[separator_row + 1, :] = code_row

    expected = grid.copy()
    for spec in pair_specs:
        _paint_cells(grid, spec["source"], spec["color"])
        _paint_cells(grid, spec["target"], spec["color"])
        if spec["color"] in active_colors:
            _paint_cells(expected, spec["target"], spec["color"])
            moved = spec["source"] + np.array(spec["offset"], dtype=int)
            _paint_cells(expected, moved, spec["color"])
        else:
            _paint_cells(expected, spec["source"], spec["color"])
            _paint_cells(expected, spec["target"], spec["color"])

    if not _semantic_checks(grid, expected, pair_specs, active_colors, bg, marker_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
