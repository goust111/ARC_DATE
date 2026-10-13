from __future__ import annotations

from collections import Counter

import numpy as np
import random


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(values).reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected at least one value"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected a unique mode"
    return int(ranked[0][0])


def _non_separator_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    sep_set = set(map(int, separators))
    spans: list[tuple[int, int]] = []
    start = None
    for idx in range(length):
        if idx in sep_set:
            if start is not None:
                spans.append((start, idx - 1))
                start = None
        elif start is None:
            start = idx
    if start is not None:
        spans.append((start, length - 1))
    assert spans, "separator lines must leave at least one panel"
    return spans


def _separator_structure(
    g: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    height, width = map(int, g.shape)
    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]]]] = []

    for color_raw in np.unique(g):
        color = int(color_raw)
        sep_rows = [r for r in range(height) if np.all(g[r, :] == color)]
        sep_cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if not sep_rows or not sep_cols:
            continue

        row_spans = _non_separator_spans(height, sep_rows)
        col_spans = _non_separator_spans(width, sep_cols)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue

        panel_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
        panel_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(panel_heights) == 1 and len(panel_widths) == 1:
            candidates.append((color, row_spans, col_spans))

    assert len(candidates) == 1, "expected one separator color with equal-sized panels"
    return candidates[0]


def _dominant_foreground(tile: np.ndarray, bg: int) -> int:
    fg = tile[tile != bg]
    assert fg.size > 0, "source panels must contain foreground pixels"
    return _unique_mode(fg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marker-only question panel with an exact copy of the panel whose
    dominant foreground color matches the marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    separator, row_spans, col_spans = _separator_structure(g)
    bg = _unique_mode(g[g != separator])
    assert bg != separator

    panel_h = row_spans[0][1] - row_spans[0][0] + 1
    panel_w = col_spans[0][1] - col_spans[0][0] + 1

    sources: dict[int, np.ndarray] = {}
    questions: list[tuple[int, int, int]] = []

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (panel_h, panel_w)
            assert not np.any(tile == separator), "separator color must stay on separator lines"

            fg_coords = np.argwhere(tile != bg)
            assert len(fg_coords) > 0, "every panel must contain a marker or foreground pattern"
            if len(fg_coords) == 1:
                rr, cc = map(int, fg_coords[0])
                marker = int(tile[rr, cc])
                assert marker not in {bg, separator}
                questions.append((r0, c0, marker))
                continue

            dominant = _dominant_foreground(tile, bg)
            assert dominant not in {bg, separator}
            assert dominant not in sources, "dominant foreground colors must identify panels uniquely"
            sources[dominant] = tile.copy()

    assert questions, "expected at least one question panel"
    assert sources, "expected at least one source panel"

    out = g.copy()
    for r0, c0, marker in questions:
        assert marker in sources, "every marker must match a source panel"
        out[r0 : r0 + panel_h, c0 : c0 + panel_w] = sources[marker]
    return out


# --------------- generator ---------------

COLOR_POOL = list(range(10))
PANEL_ROWS_CHOICES = (2, 3)
PANEL_COLS_CHOICES = (2, 3)
PANEL_H_RANGE = (4, 7)
PANEL_W_RANGE = (4, 7)
SEPARATOR_WIDTH_CHOICES = (1, 2)
MAX_QUESTIONS = 3
ACCENT_COUNT_CHOICES = (1, 2)
DOMINANT_SIZE_MIN = 4
COMPONENT_RETRIES = 40
TILE_RETRIES = 80
GENERATION_RETRIES = 300
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(h: int, w: int, r: int, c: int):
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _grow_component(h: int, w: int, size: int, blocked: np.ndarray) -> np.ndarray | None:
    free = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
    if len(free) < size:
        return None

    for _ in range(COMPONENT_RETRIES):
        seed = random.choice(free)
        selected = {seed}
        frontier = {
            cell for cell in _neighbors4(h, w, seed[0], seed[1]) if not blocked[cell]
        }
        while len(selected) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if cell in selected or blocked[cell]:
                continue
            selected.add(cell)
            for nxt in _neighbors4(h, w, cell[0], cell[1]):
                if not blocked[nxt] and nxt not in selected:
                    frontier.add(nxt)

        if len(selected) == size:
            return np.array(sorted(selected), dtype=int)
    return None


def _make_source_tile(h: int, w: int, bg: int, dominant: int, palette: list[int]) -> np.ndarray | None:
    area = h * w
    other_colors = [color for color in palette if color != dominant]
    assert other_colors

    dom_min = max(DOMINANT_SIZE_MIN, area // 6)
    dom_max = min(max(dom_min, area // 3), max(dom_min, area - 3))

    for _ in range(TILE_RETRIES):
        tile = np.full((h, w), bg, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)

        dominant_size = random.randint(dom_min, dom_max)
        dominant_cells = _grow_component(h, w, dominant_size, occupied)
        if dominant_cells is None:
            continue
        tile[dominant_cells[:, 0], dominant_cells[:, 1]] = dominant
        occupied[dominant_cells[:, 0], dominant_cells[:, 1]] = True

        accent_count = random.choice(ACCENT_COUNT_CHOICES)
        accent_colors = random.sample(other_colors, accent_count)
        remaining_free = area - dominant_size
        if remaining_free < accent_count:
            continue

        success = True
        for idx, color in enumerate(accent_colors):
            min_left = accent_count - idx - 1
            max_size = min(4, dominant_size - 1, remaining_free - min_left)
            if max_size < 1:
                success = False
                break
            size = random.randint(1, max_size)
            accent_cells = _grow_component(h, w, size, occupied)
            if accent_cells is None:
                success = False
                break
            tile[accent_cells[:, 0], accent_cells[:, 1]] = color
            occupied[accent_cells[:, 0], accent_cells[:, 1]] = True
            remaining_free -= size

        if not success:
            continue

        fg = tile[tile != bg]
        counts = Counter(map(int, fg))
        if len(counts) < 2:
            continue
        if counts[dominant] <= max(v for k, v in counts.items() if k != dominant):
            continue
        return tile

    return None


def _panel_origin(panel_r: int, panel_c: int, panel_h: int, panel_w: int, sep_w: int) -> tuple[int, int]:
    return panel_r * (panel_h + sep_w), panel_c * (panel_w + sep_w)


def _build_base_grid(
    panel_rows: int,
    panel_cols: int,
    panel_h: int,
    panel_w: int,
    sep_w: int,
    bg: int,
    separator: int,
) -> np.ndarray:
    height = panel_rows * panel_h + (panel_rows - 1) * sep_w
    width = panel_cols * panel_w + (panel_cols - 1) * sep_w
    grid = np.full((height, width), bg, dtype=int)

    for block in range(1, panel_rows):
        r0 = block * panel_h + (block - 1) * sep_w
        grid[r0 : r0 + sep_w, :] = separator
    for block in range(1, panel_cols):
        c0 = block * panel_w + (block - 1) * sep_w
        grid[:, c0 : c0 + sep_w] = separator
    return grid


def _marker_tile(panel_h: int, panel_w: int, bg: int, color: int) -> np.ndarray:
    tile = np.full((panel_h, panel_w), bg, dtype=int)
    r = random.randrange(panel_h)
    c = random.randrange(panel_w)
    tile[r, c] = color
    return tile


def _choose_question_colors(source_colors: list[int], question_count: int) -> list[int]:
    if question_count == 1:
        return [random.choice(source_colors)]

    if random.choice((False, True)):
        repeated = random.choice(source_colors)
        refs = [repeated, repeated]
        while len(refs) < question_count:
            refs.append(random.choice(source_colors))
        random.shuffle(refs)
        return refs

    return random.sample(source_colors, question_count)


def _gen_attempt() -> np.ndarray | None:
    panel_rows = random.choice(PANEL_ROWS_CHOICES)
    panel_cols = random.choice(PANEL_COLS_CHOICES)
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)
    sep_w = random.choice(SEPARATOR_WIDTH_CHOICES)
    total_panels = panel_rows * panel_cols

    question_count = random.randint(1, min(MAX_QUESTIONS, total_panels // 2))
    source_count = total_panels - question_count

    separator, bg = random.sample(COLOR_POOL, 2)
    palette = [color for color in COLOR_POOL if color not in {bg, separator}]
    source_colors = random.sample(palette, source_count)

    source_tiles: dict[int, np.ndarray] = {}
    for color in source_colors:
        tile = _make_source_tile(panel_h, panel_w, bg, color, palette)
        if tile is None:
            return None
        source_tiles[color] = tile

    references = _choose_question_colors(source_colors, question_count)
    positions = [(r, c) for r in range(panel_rows) for c in range(panel_cols)]
    random.shuffle(positions)
    question_positions = positions[:question_count]
    source_positions = positions[question_count:]

    inp = _build_base_grid(panel_rows, panel_cols, panel_h, panel_w, sep_w, bg, separator)
    expected = inp.copy()

    for pos, color in zip(source_positions, source_colors):
        r0, c0 = _panel_origin(pos[0], pos[1], panel_h, panel_w, sep_w)
        tile = source_tiles[color]
        inp[r0 : r0 + panel_h, c0 : c0 + panel_w] = tile
        expected[r0 : r0 + panel_h, c0 : c0 + panel_w] = tile

    for pos, ref_color in zip(question_positions, references):
        r0, c0 = _panel_origin(pos[0], pos[1], panel_h, panel_w, sep_w)
        inp[r0 : r0 + panel_h, c0 : c0 + panel_w] = _marker_tile(panel_h, panel_w, bg, ref_color)
        expected[r0 : r0 + panel_h, c0 : c0 + panel_w] = source_tiles[ref_color]

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(inp, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
