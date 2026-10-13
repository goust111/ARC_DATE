from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
PATTERN_COLORS = (1, 3, 4, 6, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _read_legend(grid: np.ndarray) -> tuple[list[np.ndarray], tuple[int, int]]:
    grey_comps = _components(grid == GREY)
    assert grey_comps, "missing grey legend"
    legend = max(grey_comps, key=len)
    r0, c0, r1, c1 = _bbox(legend)

    assert np.all(grid[r0, c0 : c1 + 1] == GREY)
    assert np.all(grid[r1, c0 : c1 + 1] == GREY)

    divider_cols = [c for c in range(c0, c1 + 1) if np.all(grid[r0 : r1 + 1, c] == GREY)]
    assert len(divider_cols) >= 4
    cell_h = r1 - r0 - 1
    cell_w = divider_cols[1] - divider_cols[0] - 1
    assert 1 <= cell_h <= 3
    assert 1 <= cell_w <= 3

    step = cell_w + 1
    assert all(divider_cols[i + 1] - divider_cols[i] == step for i in range(len(divider_cols) - 1))
    patterns: list[np.ndarray] = []
    for left, right in zip(divider_cols, divider_cols[1:]):
        pattern = grid[r0 + 1 : r1, left + 1 : right].copy()
        assert pattern.shape == (cell_h, cell_w)
        patterns.append(pattern)

    assert 3 <= len(patterns) <= 5
    return patterns, (cell_h, cell_w)


def _is_rectangular_border(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    return len(coords) == len(border) and set(coords) == border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each orange frame with the legend template indexed by its red marker count.

    Input structure:
    - Black background.
    - Near the top, a grey one-pixel legend bar is split by grey vertical dividers into 3-5 equal cells.
    - Each legend cell contains a small colored template on a black interior.
    - Below the legend, several orange one-pixel rectangular frames enclose black interiors with red marker dots.

    Rule:
    - Read the legend templates from left to right as indices 1, 2, 3, ...
    - For each orange frame, count the red dots inside its interior.
    - Remove those red dots by overwriting the whole interior with the template having the matching 1-based index.
    - Keep the grey legend and every orange frame border unchanged.

    Key insight:
    - The red markers encode which left-to-right legend template to paste; they are selectors rather than output pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    patterns, cell_shape = _read_legend(g)
    out = g.copy()

    frame_comps = _components(g == ORANGE)
    assert frame_comps, "missing orange frames"
    for comp in frame_comps:
        assert _is_rectangular_border(comp), "orange component must be a single rectangular border"
        r0, c0, r1, c1 = _bbox(comp)
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.shape == cell_shape
        count = int(np.count_nonzero(interior == RED))
        assert 1 <= count <= len(patterns)
        out[r0 + 1 : r1, c0 + 1 : c1] = patterns[count - 1]

    return out


def _positive_composition(total: int, parts: int) -> list[int]:
    assert 1 <= parts <= total
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _choose_cell_shape(count: int) -> tuple[int, int]:
    options = []
    for h in (2, 3):
        for w in (2, 3):
            if h * w >= count:
                options.append((h, w))
    return random.choice(options)


def _random_pattern(cell_h: int, cell_w: int, base: int, accent_choices: tuple[int, ...]) -> np.ndarray:
    pattern = np.zeros((cell_h, cell_w), dtype=int)
    cells = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    painted = random.sample(cells, random.randint(1, len(cells)))
    for r, c in painted:
        pattern[r, c] = base

    if len(painted) >= 2 and accent_choices and random.random() < 0.45:
        accent = random.choice(accent_choices)
        accent_count = random.randint(1, len(painted) - 1)
        for r, c in random.sample(painted, accent_count):
            pattern[r, c] = accent

    return pattern


def _make_patterns(count: int, cell_h: int, cell_w: int) -> list[np.ndarray]:
    bases = random.sample(PATTERN_COLORS, count)
    patterns: list[np.ndarray] = []
    seen: set[bytes] = set()
    for idx, base in enumerate(bases):
        accent_pool = tuple(color for color in PATTERN_COLORS if color != base)
        for _ in range(50):
            pattern = _random_pattern(cell_h, cell_w, base, accent_pool)
            sig = pattern.tobytes()
            if sig not in seen:
                seen.add(sig)
                patterns.append(pattern)
                break
        else:
            raise RuntimeError(f"failed to make distinct pattern {idx}")
    return patterns


def _render_legend(patterns: list[np.ndarray]) -> np.ndarray:
    cell_h, cell_w = patterns[0].shape
    count = len(patterns)
    legend = np.full((cell_h + 2, count * cell_w + count + 1), BLACK, dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    for idx in range(count + 1):
        legend[:, idx * (cell_w + 1)] = GREY
    for idx, pattern in enumerate(patterns):
        c0 = idx * (cell_w + 1) + 1
        legend[1 : 1 + cell_h, c0 : c0 + cell_w] = pattern
    return legend


def _render_frame(cell_h: int, cell_w: int, count: int) -> np.ndarray:
    frame = np.zeros((cell_h + 2, cell_w + 2), dtype=int)
    frame[0, :] = ORANGE
    frame[-1, :] = ORANGE
    frame[:, 0] = ORANGE
    frame[:, -1] = ORANGE
    cells = [(r, c) for r in range(cell_h) for c in range(cell_w)]
    for r, c in random.sample(cells, count):
        frame[r + 1, c + 1] = RED
    return frame


def _layout(frame_count: int, legend_shape: tuple[int, int], frame_shape: tuple[int, int]) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    legend_h, legend_w = legend_shape
    frame_h, frame_w = frame_shape

    for _ in range(200):
        row_count = random.randint(1, min(3, frame_count))
        rows = _positive_composition(frame_count, row_count)
        row_gaps = [random.randint(1, 3) for _ in range(max(0, row_count - 1))]
        frame_gaps = [[random.randint(1, 4) for _ in range(max(0, width - 1))] for width in rows]
        row_widths = [width * frame_w + sum(gaps) for width, gaps in zip(rows, frame_gaps)]
        gap_below_legend = random.randint(1, 4)

        min_h = legend_h + gap_below_legend + row_count * frame_h + sum(row_gaps)
        min_w = max([legend_w] + row_widths)
        if min_h > 30 or min_w > 30:
            continue

        height = min_h + random.randint(0, 30 - min_h)
        width = min_w + random.randint(0, 30 - min_w)
        top_pad = random.randint(0, min(3, height - min_h))
        legend_r = top_pad
        legend_c = random.randint(0, width - legend_w)

        positions: list[tuple[int, int]] = []
        r = legend_r + legend_h + gap_below_legend
        for row_idx, (row_size, gaps, row_width) in enumerate(zip(rows, frame_gaps, row_widths)):
            c = random.randint(0, width - row_width)
            for idx in range(row_size):
                positions.append((r, c))
                c += frame_w
                if idx < len(gaps):
                    c += gaps[idx]
            r += frame_h
            if row_idx < len(row_gaps):
                r += row_gaps[row_idx]

        if len(positions) != frame_count:
            continue
        return height, width, legend_r, legend_c, positions

    raise RuntimeError("failed to lay out grid")


def _gen_attempt() -> np.ndarray:
    legend_count = random.randint(3, 5)
    cell_h, cell_w = _choose_cell_shape(legend_count)
    patterns = _make_patterns(legend_count, cell_h, cell_w)
    legend = _render_legend(patterns)

    frame_count = random.randint(legend_count, legend_count + 2)
    counts = list(range(1, legend_count + 1))
    counts.extend(random.randint(1, legend_count) for _ in range(frame_count - legend_count))
    random.shuffle(counts)

    height, width, legend_r, legend_c, positions = _layout(frame_count, legend.shape, (cell_h + 2, cell_w + 2))
    grid = np.zeros((height, width), dtype=int)
    grid[legend_r : legend_r + legend.shape[0], legend_c : legend_c + legend.shape[1]] = legend

    for (r, c), count in zip(positions, counts):
        frame = _render_frame(cell_h, cell_w, count)
        grid[r : r + frame.shape[0], c : c + frame.shape[1]] = frame

    assert max(grid.shape) <= 30
    assert set(counts) == set(range(1, legend_count + 1))
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            return _gen_attempt()
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
