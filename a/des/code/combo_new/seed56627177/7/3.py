from __future__ import annotations

import random

import numpy as np


BG = 0
RECT_COLOR = 5
MAX_GENERATION_ATTEMPTS = 300

PALETTE_LEN_RANGE = (3, 7)
RECT_COUNT_RANGE = (2, 5)
RECT_H_RANGE = (3, 8)
RECT_W_RANGE = (3, 8)
TOP_MARGIN_RANGE = (1, 4)
BOTTOM_GAP_RANGE = (1, 4)
ROW_GAP_RANGE = (1, 3)
COL_GAP_RANGE = (1, 3)
MAX_LAYOUT_ROWS = 3


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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _palette_from_bottom(row: np.ndarray) -> tuple[int, list[int]]:
    cols = np.flatnonzero(row != BG)
    assert cols.size > 0, "expected a non-black palette strip on the bottom row"
    start = int(cols[0])
    end = int(cols[-1])

    assert np.all(row[start : end + 1] != BG), "palette strip must be contiguous"
    assert np.all(row[:start] == BG) and np.all(row[end + 1 :] == BG), "palette must be isolated on the bottom row"

    palette = [int(v) for v in row[start : end + 1]]
    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    return start, palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangle's holes with the palette color indexed by its hole count.

    Input structure:
    - Black is the background color.
    - The bottom row contains a one-pixel-high contiguous color strip made of distinct non-black cells.
    - Above the strip are several same-colored filled rectangles; each rectangle contains one or more black interior hole pixels.

    Rule:
    - Read the bottom strip from left to right as positions 1..N.
    - For each rectangle, count its black interior hole pixels.
    - Use that count as a 1-based index into the bottom strip, then fill every hole in that rectangle with the indexed color.
    - Keep the rectangle outline color and the bottom strip unchanged.

    Key insight:
    - The holes are not merely empty cells: their count is an index into the bottom color legend, so the solver must detect a count-to-color lookup.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with at least two rows"

    out = g.copy()
    _palette_start, palette = _palette_from_bottom(g[-1])
    body = g[:-1]

    non_bg = body[body != BG]
    assert non_bg.size > 0, "expected at least one rectangle above the palette"
    colors, counts = np.unique(non_bg, return_counts=True)
    rect_color = int(colors[counts.argmax()])
    assert set(np.unique(body)) <= {BG, rect_color}, "body should contain only background and rectangle color"

    found_rectangles = 0
    for comp in _components(body == rect_color):
        r0, c0, r1, c1 = _bbox(comp)
        window = body[r0 : r1 + 1, c0 : c1 + 1]

        if not np.all((window == rect_color) | (window == BG)):
            continue
        if not np.all(window[0, :] == rect_color):
            continue
        if not np.all(window[-1, :] == rect_color):
            continue
        if not np.all(window[:, 0] == rect_color):
            continue
        if not np.all(window[:, -1] == rect_color):
            continue

        holes = np.argwhere(window == BG)
        if holes.size == 0:
            continue

        assert np.all((holes[:, 0] > 0) & (holes[:, 0] < window.shape[0] - 1)), "holes must stay off the rectangle edge"
        assert np.all((holes[:, 1] > 0) & (holes[:, 1] < window.shape[1] - 1)), "holes must stay off the rectangle edge"

        hole_count = int(len(holes))
        assert 1 <= hole_count <= len(palette), "hole count must index the palette"
        fill_color = int(palette[hole_count - 1])

        for dr, dc in holes:
            out[r0 + int(dr), c0 + int(dc)] = fill_color
        found_rectangles += 1

    assert found_rectangles > 0, "expected at least one valid holey rectangle"
    return out


def _sample_holes(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    # Sampling from one parity class keeps holes visually separated.
    groups: list[list[tuple[int, int]]] = []
    for row_parity in (0, 1):
        for col_parity in (0, 1):
            cells = [
                (r, c)
                for r in range(1, height - 1)
                for c in range(1, width - 1)
                if (r - 1) % 2 == row_parity and (c - 1) % 2 == col_parity
            ]
            if len(cells) >= count:
                groups.append(cells)
    if not groups:
        return None
    return random.sample(random.choice(groups), count)


def _sample_rect_spec(hole_count: int) -> dict[str, object] | None:
    candidates: list[tuple[int, int]] = []
    for height in range(RECT_H_RANGE[0], RECT_H_RANGE[1] + 1):
        for width in range(RECT_W_RANGE[0], RECT_W_RANGE[1] + 1):
            inner_h = height - 2
            inner_w = width - 2
            capacity = ((inner_h + 1) // 2) * ((inner_w + 1) // 2)
            if capacity >= hole_count:
                candidates.append((height, width))

    if not candidates:
        return None

    for _ in range(50):
        height, width = random.choice(candidates)
        holes = _sample_holes(height, width, hole_count)
        if holes is not None:
            return {
                "h": height,
                "w": width,
                "hole_count": hole_count,
                "holes": holes,
            }
    return None


def _partition_sizes(n_items: int, n_parts: int) -> list[int]:
    if n_parts == 1:
        return [n_items]

    cuts = sorted(random.sample(range(1, n_items), n_parts - 1))
    sizes: list[int] = []
    prev = 0
    for cut in cuts + [n_items]:
        sizes.append(cut - prev)
        prev = cut
    return sizes


def _render_case(
    width: int,
    body_height: int,
    palette: list[int],
    rects: list[dict[str, object]],
    placements: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((body_height + 1, width), BG, dtype=int)
    out = inp.copy()

    for spec, (r0, c0) in zip(rects, placements, strict=True):
        height = int(spec["h"])
        rect_width = int(spec["w"])
        hole_count = int(spec["hole_count"])
        fill_color = int(palette[hole_count - 1])

        inp[r0 : r0 + height, c0 : c0 + rect_width] = RECT_COLOR
        out[r0 : r0 + height, c0 : c0 + rect_width] = RECT_COLOR

        for dr, dc in spec["holes"]:
            inp[r0 + dr, c0 + dc] = BG
            out[r0 + dr, c0 + dc] = fill_color

    bar_start = random.randint(0, width - len(palette))
    inp[-1, bar_start : bar_start + len(palette)] = palette
    out[-1, bar_start : bar_start + len(palette)] = palette
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(*PALETTE_LEN_RANGE)
    rect_count = random.randint(RECT_COUNT_RANGE[0], min(RECT_COUNT_RANGE[1], palette_len))
    hole_counts = random.sample(range(1, palette_len + 1), rect_count)

    rects: list[dict[str, object]] = []
    for hole_count in hole_counts:
        spec = _sample_rect_spec(hole_count)
        if spec is None:
            return None
        rects.append(spec)
    random.shuffle(rects)

    row_count = random.randint(1, min(MAX_LAYOUT_ROWS, rect_count))
    group_sizes = _partition_sizes(rect_count, row_count)

    groups: list[list[dict[str, object]]] = []
    index = 0
    for size in group_sizes:
        groups.append(rects[index : index + size])
        index += size

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_gap = random.randint(*BOTTOM_GAP_RANGE)
    row_gaps = [random.randint(*ROW_GAP_RANGE) for _ in range(row_count - 1)]
    row_heights = [max(int(spec["h"]) for spec in group) for group in groups]
    body_height = top_margin + sum(row_heights) + sum(row_gaps) + bottom_gap
    if body_height + 1 > 30:
        return None

    col_gaps_by_row: list[list[int]] = []
    row_widths: list[int] = []
    for group in groups:
        gaps = [random.randint(*COL_GAP_RANGE) for _ in range(len(group) - 1)]
        col_gaps_by_row.append(gaps)
        row_widths.append(sum(int(spec["w"]) for spec in group) + sum(gaps))

    min_width = max(max(row_widths), palette_len)
    if min_width > 30:
        return None
    width = random.randint(min_width, min(30, min_width + 8))

    placements: list[tuple[int, int]] = []
    y = top_margin
    for row_index, (group, row_height, gaps) in enumerate(zip(groups, row_heights, col_gaps_by_row, strict=True)):
        row_width = sum(int(spec["w"]) for spec in group) + sum(gaps)
        x = random.randint(0, width - row_width)
        for item_index, spec in enumerate(group):
            r0 = y + random.randint(0, row_height - int(spec["h"]))
            placements.append((r0, x))
            x += int(spec["w"])
            if item_index < len(gaps):
                x += gaps[item_index]
        y += row_height
        if row_index < len(row_gaps):
            y += row_gaps[row_index]

    palette = random.sample([color for color in range(1, 10) if color != RECT_COLOR], palette_len)
    inp, expected = _render_case(width, body_height, palette, rects, placements)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
