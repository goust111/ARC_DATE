import random
from collections import Counter, deque

import numpy as np


MAX_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _cyclic_from_gap(colors: list[int], gap_idx: int) -> list[int]:
    return colors[gap_idx:] + colors[:gap_idx]


def _build_output(
    shape: tuple[int, int],
    bg: int,
    span_left: int,
    completed_palette: list[int],
    gap_idx: int,
    width_by_color: dict[int, int],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), bg, dtype=int)
    span_right = span_left + len(completed_palette) - 1
    gap_col = span_left + gap_idx
    out[h - 1, span_left : span_right + 1] = np.array(completed_palette, dtype=int)
    for level, color in enumerate(_cyclic_from_gap(completed_palette, gap_idx)):
        row = h - 2 - level
        width = int(width_by_color[color])
        half = width // 2
        out[row, gap_col - half : gap_col + half + 1] = color
    return out


def _build_output_from_order(
    shape: tuple[int, int],
    bg: int,
    span_left: int,
    completed_palette: list[int],
    gap_idx: int,
    width_by_color: dict[int, int],
    order: list[int],
) -> np.ndarray:
    h, w = map(int, shape)
    out = np.full((h, w), bg, dtype=int)
    span_right = span_left + len(completed_palette) - 1
    gap_col = span_left + gap_idx
    out[h - 1, span_left : span_right + 1] = np.array(completed_palette, dtype=int)
    for level, color in enumerate(order):
        row = h - 2 - level
        width = int(width_by_color[color])
        half = width // 2
        out[row, gap_col - half : gap_col + half + 1] = color
    return out


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    bg = _get_bg(g)

    bottom = g[h - 1]
    visible_cols = np.flatnonzero(bottom != bg)
    assert visible_cols.size >= 3, "expected a bottom palette with a hidden interior gap"
    span_left = int(visible_cols.min())
    span_right = int(visible_cols.max())
    assert np.all(bottom[:span_left] == bg) and np.all(bottom[span_right + 1 :] == bg)

    span = bottom[span_left : span_right + 1]
    gap_offsets = np.flatnonzero(span == bg)
    assert gap_offsets.size == 1, "expected exactly one erased palette cell"
    gap_idx = int(gap_offsets[0])
    assert 0 < gap_idx < len(span) - 1, "gap must be internal to force cyclic wrap"

    visible_palette = [int(x) for x in span if int(x) != bg]
    assert len(visible_palette) == len(set(visible_palette)), "visible palette colors must be unique"

    upper = g[: h - 1, :]
    comps = _find_components(upper, bg)
    assert len(comps) == len(span), "bar count must match completed palette length"

    width_by_color: dict[int, int] = {}
    bars = []
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        width = int(comp["area"])
        assert r0 == r1, "each foreground object must be a one-cell-high bar"
        assert width == c1 - c0 + 1, "bars must be solid horizontal runs"
        assert width % 2 == 1, "bar widths must be odd"
        assert color not in width_by_color, "each bar color must be unique"
        width_by_color[color] = width
        bars.append((r0, c0, color, width))

    all_widths = list(width_by_color.values())
    assert len(all_widths) == len(set(all_widths)), "bar widths must all be distinct"

    bar_colors = set(width_by_color)
    visible_set = set(visible_palette)
    hidden_colors = sorted(bar_colors - visible_set)
    assert len(hidden_colors) == 1, "exactly one bar color must be absent from the visible palette"
    hidden_color = int(hidden_colors[0])
    assert visible_set | {hidden_color} == bar_colors, "other bars must match visible palette colors"

    completed_palette = [hidden_color if i == gap_idx else int(span[i]) for i in range(len(span))]
    order = _cyclic_from_gap(completed_palette, gap_idx)
    gap_col = span_left + gap_idx

    for color, width in width_by_color.items():
        half = width // 2
        assert half <= gap_col and gap_col + half < w, "centered stack must fit inside the canvas"

    return {
        "grid": g,
        "shape": (h, w),
        "bg": bg,
        "span_left": span_left,
        "span_right": span_right,
        "gap_idx": gap_idx,
        "gap_col": gap_col,
        "visible_palette": visible_palette,
        "completed_palette": completed_palette,
        "order": order,
        "width_by_color": width_by_color,
        "bars": bars,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bar whose color is missing from the legend to refill the gap, then restack all bars centered on that gap column in cyclic palette order.

    Input structure:
    - The bottom row contains one contiguous palette segment with unique colors and exactly one erased background gap inside it.
    - Above the palette lie several separated monochrome horizontal bars, each one cell tall and of a distinct odd width.
    - Exactly one bar color does not appear in the visible palette; every other bar color matches one visible palette color.

    Rule:
    - Recover the missing palette cell using the unique bar color absent from the visible legend.
    - Starting from that restored gap position, read the completed palette cyclically to the right with wraparound; this color order becomes the bottom-to-top stack order.
    - Erase every original bar, then redraw the bars densely above the palette, one row per bar, with each bar centered on the gap column; keep the completed palette row.

    Key insight:
    - The gap plays three roles at once: it identifies the hidden color, it is the cyclic reading start, and it is the alignment axis for the final stack, so neither width order nor left-to-right palette order is sufficient.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    return _build_output(
        info["shape"],
        info["bg"],
        info["span_left"],
        info["completed_palette"],
        info["gap_idx"],
        info["width_by_color"],
    )


def _scan_order(bars: list[tuple[int, int, int, int]]) -> list[int]:
    return [color for _r, _c, color, _w in sorted(bars)]


def _width_order(width_by_color: dict[int, int], reverse: bool = False) -> list[int]:
    return [color for color, _width in sorted(width_by_color.items(), key=lambda kv: kv[1], reverse=reverse)]


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(PALETTE)
    colors = [c for c in PALETTE if c != bg]

    palette_len = random.randint(4, 7)
    completed_palette = random.sample(colors, palette_len)
    gap_idx = random.randint(1, palette_len - 2)

    odd_pool = list(range(1, 2 * (palette_len + 4), 2))
    widths = random.sample(odd_pool, palette_len)
    max_width = max(widths)
    max_half = max_width // 2

    min_w = max(max_width + 2, palette_len + 4)
    max_w = min(30, min_w + 8)
    if min_w > max_w:
        return None
    width = random.randint(min_w, max_w)

    candidate_lefts = [
        left
        for left in range(width - palette_len + 1)
        if max_half <= left + gap_idx <= width - 1 - max_half
    ]
    if not candidate_lefts:
        return None
    span_left = random.choice(candidate_lefts)

    min_h = palette_len + 1
    max_h = min(30, palette_len + 7)
    height = random.randint(min_h, max_h)
    upper_rows = list(range(height - 1))
    if len(upper_rows) < palette_len:
        return None
    bar_rows = sorted(random.sample(upper_rows, palette_len))

    width_mode = random.choice(("asc", "desc", "random"))
    if width_mode == "asc":
        widths_by_row = sorted(widths)
    elif width_mode == "desc":
        widths_by_row = sorted(widths, reverse=True)
    else:
        widths_by_row = widths[:]
        random.shuffle(widths_by_row)

    cyclic_order = _cyclic_from_gap(completed_palette, gap_idx)
    left_to_right = completed_palette[:]

    for _ in range(80):
        row_colors = random.sample(completed_palette, palette_len)
        width_by_color = {color: widths_by_row[i] for i, color in enumerate(row_colors)}

        if cyclic_order == left_to_right:
            return None
        if _width_order(width_by_color, reverse=False) == cyclic_order:
            continue
        if _width_order(width_by_color, reverse=True) == cyclic_order:
            continue

        grid = np.full((height, width), bg, dtype=int)
        span = completed_palette[:]
        span[gap_idx] = bg
        grid[height - 1, span_left : span_left + palette_len] = np.array(span, dtype=int)

        bars = []
        ok = True
        for row, color in zip(bar_rows, row_colors):
            bar_w = width_by_color[color]
            start = random.randint(0, width - bar_w)
            grid[row, start : start + bar_w] = color
            bars.append((row, start, color, bar_w))

        if not ok:
            continue

        scan_order = _scan_order(bars)
        if scan_order == cyclic_order or scan_order == left_to_right:
            continue
        if scan_order == cyclic_order[::-1]:
            continue

        gap_col = span_left + gap_idx
        if all(start == gap_col - width_by_color[color] // 2 for _row, start, color, _w in bars):
            continue

        try:
            info = _analyze_input(grid)
        except AssertionError:
            continue

        if info["completed_palette"] != completed_palette:
            continue
        if info["gap_idx"] != gap_idx or info["span_left"] != span_left:
            continue
        if info["order"] != cyclic_order:
            continue

        expected = _build_output(grid.shape, bg, span_left, completed_palette, gap_idx, width_by_color)
        if not np.array_equal(main(grid), expected):
            continue

        wrong_left = _build_output(grid.shape, bg, span_left, completed_palette, 0, width_by_color)
        if np.array_equal(expected, wrong_left):
            continue

        wrong_width = _build_output_from_order(
            grid.shape,
            bg,
            span_left,
            completed_palette,
            gap_idx,
            width_by_color,
            _width_order(width_by_color, reverse=False),
        )
        if np.array_equal(expected, wrong_width):
            continue

        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            out = main(grid)
            assert isinstance(out, np.ndarray)
            assert out.ndim == 2 and max(out.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid断层转轮 input")
