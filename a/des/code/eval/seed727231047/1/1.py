from __future__ import annotations

import random

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))
GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_and_palette(
    grid: np.ndarray, bg: int
) -> tuple[int, int, int, list[int]]:
    h, _w = grid.shape
    for sep_row in range(h - 2, -1, -1):
        row = grid[sep_row]
        if not np.all(row == row[0]):
            continue
        gray = int(row[0])
        if gray == bg:
            continue
        palette_row = sep_row + 1
        palette_mask = (grid[palette_row] != bg) & (grid[palette_row] != gray)
        if int(palette_mask.sum()) < 2:
            continue
        cols = np.where(palette_mask)[0]
        if int(cols[-1] - cols[0] + 1) != int(len(cols)):
            continue
        palette = [int(v) for v in grid[palette_row, cols[0] : cols[-1] + 1]]
        if len(set(palette)) != len(palette):
            continue
        if sep_row > 0 and not np.any(grid[:sep_row] == gray):
            continue
        return gray, sep_row, palette_row, palette
    raise AssertionError("failed to find palette separator")


def _is_rect_frame(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray rectangular border according to how many colored pixels it
    contains, using the bottom palette row as a 1-based lookup table.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    gray, sep_row, _palette_row, palette = _find_separator_and_palette(g, bg)

    scene = g[:sep_row, :]
    gray_mask = scene == gray
    comps = _components(gray_mask)
    assert comps, "expected at least one gray frame above the separator"

    out = g.copy()
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        bbox = (min(rows), min(cols), max(rows), max(cols))
        assert _is_rect_frame(gray_mask, bbox), "gray components must be hollow rectangles"
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        assert not np.any(inner == gray), "frame interior must not contain gray"
        count = int(np.count_nonzero((inner != bg) & (inner != gray)))
        assert 1 <= count <= len(palette), "interior count must index into the palette"
        new_color = int(palette[count - 1])
        out[r0, c0 : c1 + 1] = new_color
        out[r1, c0 : c1 + 1] = new_color
        out[r0 : r1 + 1, c0] = new_color
        out[r0 : r1 + 1, c1] = new_color

    return out


def _choose_palette_length(width: int) -> int:
    hi = min(7, width - 2, len(ARC_COLORS) - 2)
    return random.randint(3, hi)


def _choose_rect_count(width: int, height: int) -> int:
    max_rects = 4
    if width <= 15 or height <= 13:
        max_rects = 3
    if width <= 12 or height <= 10:
        max_rects = 2
    return random.randint(1, max_rects)


def _choose_counts(palette_len: int, rect_count: int) -> list[int]:
    options = list(range(1, palette_len + 1))
    mode = random.choice(("sample", "repeat", "extremes"))
    if mode == "sample" and rect_count <= len(options):
        return random.sample(options, rect_count)

    counts: list[int] = []
    if mode == "extremes":
        counts.append(random.choice((1, palette_len)))
        if rect_count >= 2:
            second = palette_len if counts[0] != palette_len else 1
            counts.append(second)
    while len(counts) < rect_count:
        counts.append(random.choice(options))
    random.shuffle(counts)
    return counts[:rect_count]


def _size_options(count: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    options: list[tuple[int, int]] = []
    for h in range(3, max_h + 1):
        for w in range(3, max_w + 1):
            if (h - 2) * (w - 2) >= count:
                options.append((h, w))
    return options


def _place_rectangles(grid: np.ndarray, gray: int, counts: list[int]) -> bool:
    h, w = grid.shape
    content_bottom = h - 3
    if content_bottom < 3:
        return False

    blocked = np.zeros((content_bottom, w), dtype=bool)
    border_margin_top = 1
    border_margin_side = 1
    max_h = min(8, content_bottom - border_margin_top)
    max_w = min(9, w - 2 * border_margin_side)
    if max_h < 3 or max_w < 3:
        return False

    order = sorted(
        [
            {
                "count": int(count),
                "marker": random.choice([c for c in ARC_COLORS if c not in {BLACK, gray}]),
            }
            for count in counts
        ],
        key=lambda item: item["count"],
        reverse=True,
    )

    placed = 0
    for spec in order:
        count = int(spec["count"])
        options = _size_options(count, max_h, max_w)
        random.shuffle(options)
        success = False
        for rect_h, rect_w in options[:80]:
            row_lo = border_margin_top
            row_hi = content_bottom - rect_h
            col_lo = border_margin_side
            col_hi = w - rect_w - border_margin_side
            if row_hi < row_lo or col_hi < col_lo:
                continue
            for _ in range(120):
                r0 = random.randint(row_lo, row_hi)
                c0 = random.randint(col_lo, col_hi)
                r1 = r0 + rect_h - 1
                c1 = c0 + rect_w - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(content_bottom - 1, r1 + 1)
                cc1 = min(w - 1, c1 + 1)
                if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
                    continue

                grid[r0, c0 : c1 + 1] = gray
                grid[r1, c0 : c1 + 1] = gray
                grid[r0 : r1 + 1, c0] = gray
                grid[r0 : r1 + 1, c1] = gray

                inner_cells = [
                    (rr, cc)
                    for rr in range(r0 + 1, r1)
                    for cc in range(c0 + 1, c1)
                ]
                picks = random.sample(inner_cells, count)
                for rr, cc in picks:
                    grid[rr, cc] = int(spec["marker"])
                blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
                placed += 1
                success = True
                break
            if success:
                break
        if not success:
            return False

    return placed == len(order)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 22)
    width = random.randint(14, 24)
    palette_len = _choose_palette_length(width)

    color_pool = [c for c in ARC_COLORS if c != BLACK]
    gray = random.choice(color_pool)
    palette_colors = random.sample([c for c in color_pool if c != gray], palette_len)

    grid = np.full((height, width), BLACK, dtype=int)
    sep_row = height - 2
    palette_row = height - 1
    grid[sep_row, :] = gray

    start = random.randint(0, width - palette_len)
    grid[palette_row, start : start + palette_len] = np.array(palette_colors, dtype=int)

    rect_count = _choose_rect_count(width, height)
    counts = _choose_counts(palette_len, rect_count)
    if not _place_rectangles(grid, gray, counts):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    changed = (out != grid)
    if int(changed.sum()) == 0:
        return None
    # Ensure only frame colors change.
    if np.any(changed & (grid != gray)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
