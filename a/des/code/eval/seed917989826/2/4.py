import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_component_sizes(grid: np.ndarray, bg: int) -> np.ndarray:
    h, w = grid.shape
    sizes = np.zeros((h, w), dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            coords = []
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            size = len(coords)
            for cr, cc in coords:
                sizes[cr, cc] = size
    return sizes


def _find_priority_tower(grid: np.ndarray, bg: int) -> tuple[int, list[int], set[tuple[int, int]]]:
    h, w = grid.shape
    comp_sizes = _same_color_component_sizes(grid, bg)
    candidates = []
    for c in range(w):
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size < 2:
            continue
        colors = []
        ok = True
        for r in rows:
            if comp_sizes[r, c] != 1:
                ok = False
                break
            if c > 0 and int(grid[r, c - 1]) != bg:
                ok = False
                break
            if c + 1 < w and int(grid[r, c + 1]) != bg:
                ok = False
                break
            colors.append(int(grid[r, c]))
        if ok and len(set(colors)) == len(colors):
            candidates.append((int(rows.size), -c, c, rows.tolist(), colors))
    assert candidates, "priority tower not found"
    _count, _neg_c, tower_col, rows, colors = max(candidates)
    tower_coords = {(int(r), int(tower_col)) for r in rows}
    return int(tower_col), [int(color) for color in colors], tower_coords


def _longest_run(mask_1d: np.ndarray) -> int:
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return 0
    best = 1
    cur = 1
    prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            cur += 1
        else:
            best = max(best, cur)
            cur = 1
        prev = value
    return max(best, cur)


def _group_consecutive(values: list[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    groups = []
    start = prev = int(values[0])
    for value in values[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _detect_horizontal_strips(mask: np.ndarray) -> list[dict]:
    h, _w = mask.shape
    candidate_rows = [r for r in range(h) if _longest_run(mask[r]) >= 3]
    strips = []
    for r0, r1 in _group_consecutive(candidate_rows):
        assert 1 <= r1 - r0 + 1 <= 2, "horizontal strip width must be 1 or 2"
        cols = np.flatnonzero(np.any(mask[r0 : r1 + 1], axis=0))
        assert cols.size >= 3, "horizontal strip must be visibly long"
        c0, c1 = int(cols[0]), int(cols[-1])
        block = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(block.all()), "horizontal strip occupancy must form a solid rectangle"
        strips.append(
            {
                "orientation": "H",
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
            }
        )
    return strips


def _detect_vertical_strips(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    candidate_cols = [c for c in range(w) if _longest_run(mask[:, c]) >= 3]
    strips = []
    for c0, c1 in _group_consecutive(candidate_cols):
        assert 1 <= c1 - c0 + 1 <= 2, "vertical strip width must be 1 or 2"
        rows = np.flatnonzero(np.any(mask[:, c0 : c1 + 1], axis=1))
        assert rows.size >= 3, "vertical strip must be visibly long"
        r0, r1 = int(rows[0]), int(rows[-1])
        block = mask[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(block.all()), "vertical strip occupancy must form a solid rectangle"
        strips.append(
            {
                "orientation": "V",
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
            }
        )
    return strips


def _overlap(a: dict, b: dict) -> tuple[int, int, int, int] | None:
    r0 = max(int(a["r0"]), int(b["r0"]))
    r1 = min(int(a["r1"]), int(b["r1"]))
    c0 = max(int(a["c0"]), int(b["c0"]))
    c1 = min(int(a["c1"]), int(b["c1"]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _infer_strip_color(grid: np.ndarray, bg: int, strip: dict, orthogonal: list[dict]) -> int:
    r0, r1 = int(strip["r0"]), int(strip["r1"])
    c0, c1 = int(strip["c0"]), int(strip["c1"])
    support = np.ones((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for other in orthogonal:
        overlap = _overlap(strip, other)
        if overlap is None:
            continue
        or0, oc0, or1, oc1 = overlap
        support[or0 - r0 : or1 - r0 + 1, oc0 - c0 : oc1 - c0 + 1] = False
    values = grid[r0 : r1 + 1, c0 : c1 + 1][support]
    values = values[values != bg]
    if values.size == 0:
        values = grid[r0 : r1 + 1, c0 : c1 + 1]
        values = values[values != bg]
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "strip color support vanished"
    ranked = counts.most_common()
    if len(ranked) >= 2:
        assert ranked[0][1] > ranked[1][1], "strip color inference is ambiguous"
    return int(ranked[0][0])


def _paint_if_higher(best_rank: np.ndarray, paint: np.ndarray, rows: slice, cols: slice, rank: int, color: int) -> None:
    current = best_rank[rows, cols]
    better = rank < current
    if not np.any(better):
        return
    best_rank[rows, cols] = np.where(better, rank, current)
    paint[rows, cols] = np.where(better, color, paint[rows, cols])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve strip crossings using the left priority tower.

    Input structure:
    - The most common color is background.
    - On the left, one column contains isolated colored pixels whose top-to-bottom
      order defines the priority from high to low.
    - On the right, straight horizontal and vertical strips of width 1-2 cross.

    Rule:
    - Read the priority tower.
    - Reconstruct the horizontal and vertical strip rectangles from occupied rows
      and columns.
    - At each crossing, repaint the whole overlap with the higher-priority color.
    - Then repaint one extra cell beyond each side of the overlap along the
      lower-priority strip's axis.
    - Leave the tower unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    _tower_col, priority_colors, tower_coords = _find_priority_tower(g, bg)
    priority = {int(color): idx for idx, color in enumerate(priority_colors)}

    mask = g != bg
    for r, c in tower_coords:
        mask[r, c] = False

    horizontal = _detect_horizontal_strips(mask)
    vertical = _detect_vertical_strips(mask)
    assert horizontal and vertical, "expected both horizontal and vertical strips"

    used_colors = set(priority_colors)
    for strip in horizontal:
        strip["color"] = _infer_strip_color(g, bg, strip, vertical)
        assert int(strip["color"]) in used_colors, "strip color must exist in the tower"
    for strip in vertical:
        strip["color"] = _infer_strip_color(g, bg, strip, horizontal)
        assert int(strip["color"]) in used_colors, "strip color must exist in the tower"

    out = g.copy()
    best_rank = np.full(g.shape, fill_value=len(priority_colors) + 1, dtype=int)
    paint = out.copy()

    for h_strip in horizontal:
        for v_strip in vertical:
            overlap = _overlap(h_strip, v_strip)
            if overlap is None:
                continue
            hrank = priority[int(h_strip["color"])]
            vrank = priority[int(v_strip["color"])]
            if hrank < vrank:
                high_color = int(h_strip["color"])
                high_rank = int(hrank)
                low = v_strip
            else:
                high_color = int(v_strip["color"])
                high_rank = int(vrank)
                low = h_strip

            r0, c0, r1, c1 = overlap
            _paint_if_higher(best_rank, paint, slice(r0, r1 + 1), slice(c0, c1 + 1), high_rank, high_color)

            if low["orientation"] == "V":
                if r0 - 1 >= int(low["r0"]):
                    _paint_if_higher(
                        best_rank,
                        paint,
                        slice(r0 - 1, r0),
                        slice(int(low["c0"]), int(low["c1"]) + 1),
                        high_rank,
                        high_color,
                    )
                if r1 + 1 <= int(low["r1"]):
                    _paint_if_higher(
                        best_rank,
                        paint,
                        slice(r1 + 1, r1 + 2),
                        slice(int(low["c0"]), int(low["c1"]) + 1),
                        high_rank,
                        high_color,
                    )
            else:
                if c0 - 1 >= int(low["c0"]):
                    _paint_if_higher(
                        best_rank,
                        paint,
                        slice(int(low["r0"]), int(low["r1"]) + 1),
                        slice(c0 - 1, c0),
                        high_rank,
                        high_color,
                    )
                if c1 + 1 <= int(low["c1"]):
                    _paint_if_higher(
                        best_rank,
                        paint,
                        slice(int(low["r0"]), int(low["r1"]) + 1),
                        slice(c1 + 1, c1 + 2),
                        high_rank,
                        high_color,
                    )

    changed = best_rank <= len(priority_colors)
    out[changed] = paint[changed]
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_COLOR_POOL = list(range(10))
_COUNT_OPTIONS = (2, 3)
_WIDTH_OPTIONS = (1, 2)
_BETWEEN_GAP_RANGE = (2, 4)
_END_MARGIN_RANGE = (2, 4)
_TOWER_GAP_RANGE = (1, 2)
_TOWER_LEFT_MARGIN_RANGE = (1, 2)
_TOWER_TOP_RANGE = (1, 3)
_MAIN_TOP_RANGE = (1, 3)
_TOWER_TO_MAIN_GAP_RANGE = (3, 5)
_RIGHT_MARGIN_RANGE = (1, 3)
_BOTTOM_MARGIN_RANGE = (1, 3)
_GENERATION_RETRIES = 400


def _sample_axis_layout(count: int, widths: list[int]) -> tuple[list[int], int]:
    leading = int(_RNG.integers(_END_MARGIN_RANGE[0], _END_MARGIN_RANGE[1] + 1))
    trailing = int(_RNG.integers(_END_MARGIN_RANGE[0], _END_MARGIN_RANGE[1] + 1))
    gaps = [
        int(_RNG.integers(_BETWEEN_GAP_RANGE[0], _BETWEEN_GAP_RANGE[1] + 1))
        for _ in range(max(0, count - 1))
    ]
    starts = []
    pos = leading
    for idx, width in enumerate(widths):
        starts.append(int(pos))
        pos += int(width)
        if idx < len(gaps):
            pos += int(gaps[idx])
    total = int(pos + trailing)
    return starts, total


def _sample_tower_rows(count: int) -> tuple[list[int], int]:
    rows = [0]
    pos = 0
    for _ in range(count - 1):
        pos += 1 + int(_RNG.integers(_TOWER_GAP_RANGE[0], _TOWER_GAP_RANGE[1] + 1))
        rows.append(int(pos))
    return rows, int(rows[-1] + 1)


def _paint_strip(grid: np.ndarray, strip: dict) -> None:
    grid[int(strip["r0"]) : int(strip["r1"]) + 1, int(strip["c0"]) : int(strip["c1"]) + 1] = int(strip["color"])


def _gen_attempt() -> np.ndarray | None:
    bg = int(_RNG.choice(_COLOR_POOL))

    n_h = int(random.choice(_COUNT_OPTIONS))
    n_v = int(random.choice(_COUNT_OPTIONS))
    n_colors = n_h + n_v
    palette = [color for color in _COLOR_POOL if color != bg]
    colors = _RNG.choice(palette, size=n_colors, replace=False).astype(int).tolist()

    priority_colors = colors[:]
    random.shuffle(priority_colors)

    strip_colors = colors[:]
    random.shuffle(strip_colors)
    h_colors = strip_colors[:n_h]
    v_colors = strip_colors[n_h:]

    h_widths = [int(random.choice(_WIDTH_OPTIONS)) for _ in range(n_h)]
    v_widths = [int(random.choice(_WIDTH_OPTIONS)) for _ in range(n_v)]

    h_starts_local, main_height = _sample_axis_layout(n_h, h_widths)
    v_starts_local, main_width = _sample_axis_layout(n_v, v_widths)

    tower_rows_local, tower_height = _sample_tower_rows(n_colors)
    tower_top = int(_RNG.integers(_TOWER_TOP_RANGE[0], _TOWER_TOP_RANGE[1] + 1))
    main_top = int(_RNG.integers(_MAIN_TOP_RANGE[0], _MAIN_TOP_RANGE[1] + 1))
    tower_left = int(_RNG.integers(_TOWER_LEFT_MARGIN_RANGE[0], _TOWER_LEFT_MARGIN_RANGE[1] + 1))
    main_left = tower_left + 1 + int(_RNG.integers(_TOWER_TO_MAIN_GAP_RANGE[0], _TOWER_TO_MAIN_GAP_RANGE[1] + 1))

    height = max(tower_top + tower_height, main_top + main_height) + int(
        _RNG.integers(_BOTTOM_MARGIN_RANGE[0], _BOTTOM_MARGIN_RANGE[1] + 1)
    )
    width = main_left + main_width + int(_RNG.integers(_RIGHT_MARGIN_RANGE[0], _RIGHT_MARGIN_RANGE[1] + 1))
    grid = np.full((height, width), bg, dtype=int)

    for row_local, color in zip(tower_rows_local, priority_colors):
        grid[tower_top + int(row_local), tower_left] = int(color)

    horizontals = []
    for row_local, band_width, color in zip(h_starts_local, h_widths, h_colors):
        horizontals.append(
            {
                "orientation": "H",
                "r0": main_top + int(row_local),
                "r1": main_top + int(row_local) + int(band_width) - 1,
                "c0": main_left,
                "c1": main_left + main_width - 1,
                "color": int(color),
            }
        )

    verticals = []
    for col_local, band_width, color in zip(v_starts_local, v_widths, v_colors):
        verticals.append(
            {
                "orientation": "V",
                "r0": main_top,
                "r1": main_top + main_height - 1,
                "c0": main_left + int(col_local),
                "c1": main_left + int(col_local) + int(band_width) - 1,
                "color": int(color),
            }
        )

    for strip in horizontals + verticals:
        _paint_strip(grid, strip)

    saw_low_visible = False
    for h_strip in horizontals:
        for v_strip in verticals:
            overlap = _overlap(h_strip, v_strip)
            assert overlap is not None
            hrank = priority_colors.index(int(h_strip["color"]))
            vrank = priority_colors.index(int(v_strip["color"]))
            if random.random() < 0.5:
                visible_color = int(h_strip["color"])
            else:
                visible_color = int(v_strip["color"])
            if (hrank < vrank and visible_color == int(v_strip["color"])) or (
                vrank < hrank and visible_color == int(h_strip["color"])
            ):
                saw_low_visible = True
            r0, c0, r1, c1 = overlap
            grid[r0 : r1 + 1, c0 : c1 + 1] = visible_color

    if not saw_low_visible and horizontals and verticals:
        h_strip = horizontals[0]
        v_strip = verticals[0]
        overlap = _overlap(h_strip, v_strip)
        assert overlap is not None
        hrank = priority_colors.index(int(h_strip["color"]))
        vrank = priority_colors.index(int(v_strip["color"]))
        visible_color = int(v_strip["color"] if hrank < vrank else h_strip["color"])
        r0, c0, r1, c1 = overlap
        grid[r0 : r1 + 1, c0 : c1 + 1] = visible_color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid priority-strip puzzle")
