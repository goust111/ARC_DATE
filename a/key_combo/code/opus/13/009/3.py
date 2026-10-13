from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 7)
GEN_ATTEMPTS = 500
SIZE_RANGE = (12, 17)  # high-exclusive for randrange
LINE_COUNT_OPTIONS = ((2, 2), (2, 3), (3, 2))
MIN_BAND = 2


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = (grid == color)
    seen = np.zeros_like(mask, dtype=bool)
    h, w = grid.shape
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _classify_objects(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], int]]:
    horizontal: list[tuple[int, int]] = []
    vertical: list[tuple[int, int]] = []
    markers: dict[tuple[int, int], int] = {}

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BG):
        for comp in _color_components(grid, color):
            rows = {r for r, _ in comp}
            cols = {c for _, c in comp}
            if len(comp) == 1:
                markers[comp[0]] = color
                continue
            if len(rows) == 1 and 2 <= len(comp) <= 4:
                ordered = sorted(c for _, c in comp)
                assert ordered == list(range(ordered[0], ordered[-1] + 1))
                horizontal.append((next(iter(rows)), color))
                continue
            if len(cols) == 1 and 2 <= len(comp) <= 4:
                ordered = sorted(r for r, _ in comp)
                assert ordered == list(range(ordered[0], ordered[-1] + 1))
                vertical.append((next(iter(cols)), color))
                continue
            raise AssertionError("expected isolated markers or straight 1-cell-thick segments")

    assert 2 <= len(horizontal) <= 3
    assert 2 <= len(vertical) <= 3
    assert len(markers) >= 1
    return sorted(horizontal), sorted(vertical), markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend short horizontal and vertical segments into full lines, then fill each chamber by its internal marker majority.

    Input structure:
    - The grid has a black background with several 1-cell-thick short horizontal and vertical colored segments.
    - Additional isolated single-pixel colored markers lie away from those segments.
    - The segment rows and columns define future full-width/full-height divider lines whose pairwise crossings are initially black.

    Rule:
    - Detect each short same-color straight segment; extend every horizontal one across its whole row and every vertical one across its whole column.
    - Treat the extended lines, plus their yellow crossing cells, as chamber boundaries.
    - For each resulting 4-connected non-line chamber, count the original marker colors inside it and fill the whole chamber with the unique majority color.
    - Draw chamber fills first, then redraw the extended colored lines, and finally recolor every horizontal/vertical crossing yellow.

    Key insight:
    - The short segments specify a full partition of the canvas, and only after extending them can the marker pixels be grouped into the chambers whose majority colors control the fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    horizontal, vertical, markers = _classify_objects(g)
    h, w = g.shape

    line_mask = np.zeros((h, w), dtype=bool)
    for row, _color in horizontal:
        line_mask[row, :] = True
    for col, _color in vertical:
        line_mask[:, col] = True

    out = np.zeros_like(g)
    chambers = _mask_components(~line_mask)
    assert chambers

    for chamber in chambers:
        counts = Counter(markers[pos] for pos in chamber if pos in markers)
        assert counts, "every chamber must contain at least one marker"
        ranked = counts.most_common(2)
        if len(ranked) > 1:
            assert ranked[0][1] > ranked[1][1], "marker majority must be unique"
        fill = ranked[0][0]
        for r, c in chamber:
            out[r, c] = fill

    for row, color in horizontal:
        out[row, :] = color
    for col, color in vertical:
        out[:, col] = color
    for row, _h_color in horizontal:
        for col, _v_color in vertical:
            out[row, col] = YELLOW
    return out


def _sample_band_sizes(total: int, count: int, min_size: int = MIN_BAND) -> list[int]:
    sizes = [min_size] * count
    for _ in range(total - min_size * count):
        sizes[random.randrange(count)] += 1
    random.shuffle(sizes)
    return sizes


def _lines_from_bands(bands: list[int]) -> list[int]:
    lines: list[int] = []
    cursor = bands[0]
    for size in bands[1:]:
        lines.append(cursor)
        cursor += 1 + size
    return lines


def _spans(length: int, lines: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in sorted(lines):
        spans.append((start, line))
        start = line + 1
    spans.append((start, length))
    return spans


def _segment_color_sets(n_h: int, n_v: int) -> tuple[list[int], list[int]]:
    if n_h + n_v <= len(PALETTE) and random.random() < 0.7:
        chosen = random.sample(PALETTE, n_h + n_v)
        return chosen[:n_h], chosen[n_h:]
    return random.sample(PALETTE, n_h), random.sample(PALETTE, n_v)


def _place_horizontal_segments(
    grid: np.ndarray, h_rows: list[int], col_spans: list[tuple[int, int]], colors: list[int]
) -> None:
    usable = [span for span in col_spans if span[1] - span[0] >= 2]
    for row, color in zip(h_rows, colors):
        c0, c1 = random.choice(usable)
        length = random.randint(2, min(4, c1 - c0))
        start = random.randint(c0, c1 - length)
        grid[row, start : start + length] = color


def _place_vertical_segments(
    grid: np.ndarray, v_cols: list[int], row_spans: list[tuple[int, int]], colors: list[int]
) -> None:
    usable = [span for span in row_spans if span[1] - span[0] >= 2]
    for col, color in zip(v_cols, colors):
        r0, r1 = random.choice(usable)
        length = random.randint(2, min(4, r1 - r0))
        start = random.randint(r0, r1 - length)
        grid[start : start + length, col] = color


def _chamber_slots(span_r: tuple[int, int], span_c: tuple[int, int]) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(*span_r) for c in range(*span_c)]
    parity_groups = [
        [cell for cell in cells if (cell[0] + cell[1]) % 2 == parity]
        for parity in (0, 1)
    ]
    sizes = [len(group) for group in parity_groups]
    best = max(sizes)
    candidates = [group for group in parity_groups if len(group) == best]
    return random.choice(candidates)


def _target_colors(n: int) -> list[int]:
    colors: list[int] = []
    while len(colors) < n:
        batch = list(PALETTE)
        random.shuffle(batch)
        colors.extend(batch)
    return colors[:n]


def _counts_have_unique_majority(colors: list[int]) -> bool:
    ranked = Counter(colors).most_common(2)
    if not ranked:
        return False
    return len(ranked) == 1 or ranked[0][1] > ranked[1][1]


def _gen_attempt() -> np.ndarray | None:
    n_h, n_v = random.choice(LINE_COUNT_OPTIONS)
    height = random.randrange(*SIZE_RANGE)
    width = random.randrange(*SIZE_RANGE)

    if height - n_h < MIN_BAND * (n_h + 1) or width - n_v < MIN_BAND * (n_v + 1):
        return None

    row_bands = _sample_band_sizes(height - n_h, n_h + 1)
    col_bands = _sample_band_sizes(width - n_v, n_v + 1)
    h_rows = _lines_from_bands(row_bands)
    v_cols = _lines_from_bands(col_bands)
    row_spans = _spans(height, h_rows)
    col_spans = _spans(width, v_cols)
    chamber_count = len(row_spans) * len(col_spans)

    grid = np.zeros((height, width), dtype=int)
    h_colors, v_colors = _segment_color_sets(n_h, n_v)
    _place_horizontal_segments(grid, h_rows, col_spans, h_colors)
    _place_vertical_segments(grid, v_cols, row_spans, v_colors)

    chambers: list[dict] = []
    targets = _target_colors(chamber_count)
    idx = 0
    for span_r in row_spans:
        for span_c in col_spans:
            slots = _chamber_slots(span_r, span_c)
            if not slots:
                return None
            chambers.append(
                {
                    "slots": slots,
                    "markers": [targets[idx]],
                    "target": targets[idx],
                }
            )
            idx += 1

    min_total = chamber_count if random.random() < 0.15 else chamber_count + 1
    target_total = random.randint(min_total, min(15, chamber_count + 4))
    extras = target_total - chamber_count

    competitive = [i for i, chamber in enumerate(chambers) if len(chamber["slots"]) >= 3]
    random.shuffle(competitive)
    for i in competitive:
        if extras < 2:
            break
        if random.random() < 0.7:
            target = chamber_target = chambers[i]["target"]
            alt_choices = [color for color in PALETTE if color != chamber_target]
            chambers[i]["markers"].extend([target, random.choice(alt_choices)])
            extras -= 2

    while extras > 0:
        candidates = [
            i
            for i, chamber in enumerate(chambers)
            if len(chamber["markers"]) < len(chamber["slots"])
        ]
        if not candidates:
            return None
        i = random.choice(candidates)
        chambers[i]["markers"].append(chambers[i]["target"])
        extras -= 1

    marker_total = 0
    for chamber in chambers:
        if len(chamber["markers"]) > len(chamber["slots"]):
            return None
        if not _counts_have_unique_majority(chamber["markers"]):
            return None
        marker_total += len(chamber["markers"])

    if not (8 <= marker_total <= 15):
        return None

    if len({chamber["target"] for chamber in chambers}) < 2:
        return None

    for chamber in chambers:
        for (r, c), color in zip(chamber["slots"], chamber["markers"]):
            if grid[r, c] != BG:
                return None
            grid[r, c] = color

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected_yellow = n_h * n_v
    if int((out == YELLOW).sum()) != expected_yellow:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
