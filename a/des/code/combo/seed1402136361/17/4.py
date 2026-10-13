import random

import numpy as np


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _solid_square_size(grid: np.ndarray, cells: list[tuple[int, int]]) -> tuple[int, int] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or len(cells) != h * w:
        return None
    color = int(grid[cells[0]])
    if any(int(grid[r, c]) != color for r, c in cells):
        return None
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
        return None
    return color, h


def _square_window(center_r: int, center_c: int, size: int, h: int, w: int) -> tuple[int, int, int, int, bool]:
    if size % 2 == 1:
        raw_r0 = center_r - size // 2
        raw_c0 = center_c - size // 2
    else:
        raw_r0 = center_r - size // 2 + 1
        raw_c0 = center_c - size // 2 + 1
    raw_r1 = raw_r0 + size
    raw_c1 = raw_c0 + size
    clipped = raw_r0 < 0 or raw_c0 < 0 or raw_r1 > h or raw_c1 > w
    r0 = max(0, raw_r0)
    c0 = max(0, raw_c0)
    r1 = min(h, raw_r1)
    c1 = min(w, raw_c1)
    return r0, r1, c0, c1, clipped


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, dict[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, dict[int, int]]] = []

    for r0 in range(h):
        for c1 in range(1, w + 1):
            legend = g[r0:, :c1]
            if not np.any(legend != bg):
                continue
            if r0 > 0 and np.any(g[r0 - 1, :c1] != bg):
                continue
            if c1 < w and np.any(g[r0:, c1] != bg):
                continue
            if not np.any(legend[0, :] != bg):
                continue
            if not np.any(legend[:, -1] != bg):
                continue

            mapping: dict[int, int] = {}
            ok = True
            for cells in _components(legend != bg):
                info = _solid_square_size(legend, cells)
                if info is None:
                    ok = False
                    break
                color, size = info
                if color in mapping:
                    ok = False
                    break
                mapping[color] = size
            if not ok or len(mapping) < 2:
                continue

            outside_mask = (g != bg).copy()
            outside_mask[r0:, :c1] = False
            outside = _components(outside_mask)
            if not outside:
                continue
            for cells in outside:
                if len(cells) != 1:
                    ok = False
                    break
                r, c = cells[0]
                if int(g[r, c]) not in mapping:
                    ok = False
                    break
            if ok:
                candidates.append((r0, c1, mapping))

    assert len(candidates) == 1, "legend region must be uniquely identifiable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the bottom-left legend as a color->size lookup, then expand each main-area marker
    into a same-colored solid square centered on that marker. Even sizes use an upper-left
    center bias, and off-grid parts are clipped.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    values, counts = np.unique(g, return_counts=True)
    bg = int(values[counts.argmax()])

    legend_r0, legend_c1, size_by_color = _find_legend(g, bg)
    out = g.copy()

    marker_mask = (g != bg).copy()
    marker_mask[legend_r0:, :legend_c1] = False
    for cells in _components(marker_mask):
        assert len(cells) == 1, "main-area objects must be single-pixel markers"
        r, c = cells[0]
        color = int(g[r, c])
        assert color in size_by_color, "marker color must appear in the legend"
        size = size_by_color[color]
        r0, r1, c0, c1, _clipped = _square_window(r, c, size, h, w)
        out[r0:r1, c0:c1] = color

    return out


# --------------- generator ---------------

_GEN_ATTEMPTS = 400
_SIZE_POOL = (1, 2, 3, 4, 5)
_ODD_SIZES = (1, 3, 5)
_EVEN_SIZES = (2, 4)
_LEGEND_GAP = 1
_GRID_H_BASE = (14, 24)
_GRID_W_BASE = (18, 30)
_EXTRA_MARKER_CAP = 1


def _reserve_bbox(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    mask[r0:r1, c0:c1] = True


def _build_output(
    grid: np.ndarray,
    markers: list[tuple[int, int, int]],
    size_by_color: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    h, w = grid.shape
    for r, c, color in markers:
        r0, r1, c0, c1, _ = _square_window(r, c, size_by_color[color], h, w)
        out[r0:r1, c0:c1] = color
    return out


def _choose_sizes(color_count: int) -> list[int]:
    sizes = [random.choice(_ODD_SIZES), random.choice(_EVEN_SIZES)]
    remaining = [size for size in _SIZE_POOL if size not in sizes]
    sizes.extend(random.sample(remaining, color_count - 2))
    random.shuffle(sizes)
    return sizes


def _marker_specs(size_by_color: dict[int, int]) -> list[tuple[int, bool]]:
    colors = list(size_by_color)
    extra = random.randint(0, min(_EXTRA_MARKER_CAP, 5 - len(colors)))
    specs = [(color, False) for color in colors]
    for color in random.choices(colors, k=extra):
        specs.append((color, False))

    multi_idx = [i for i, (color, _clip) in enumerate(specs) if size_by_color[color] > 1]
    assert multi_idx
    clip_count = random.randint(1, len(multi_idx))
    for idx in random.sample(multi_idx, clip_count):
        specs[idx] = (specs[idx][0], True)
    random.shuffle(specs)
    return specs


def _place_markers(
    shape: tuple[int, int],
    legend_r0: int,
    legend_w: int,
    marker_specs: list[tuple[int, bool]],
    size_by_color: dict[int, int],
) -> list[tuple[int, int, int]] | None:
    h, w = shape
    main_c0 = legend_w + 1
    reserved = np.zeros((h, w), dtype=bool)
    _reserve_bbox(reserved, legend_r0, h, 0, legend_w)
    markers: list[tuple[int, int, int]] = []
    marker_cells = set()

    for color, want_clipped in marker_specs:
        size = size_by_color[color]
        candidates: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(main_c0, w):
                if (r, c) in marker_cells:
                    continue
                r0, r1, c0, c1, clipped = _square_window(r, c, size, h, w)
                if clipped != want_clipped:
                    continue
                if reserved[r0:r1, c0:c1].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None

        r, c = random.choice(candidates)
        markers.append((r, c, color))
        marker_cells.add((r, c))
        r0, r1, c0, c1, _ = _square_window(r, c, size, h, w)
        _reserve_bbox(reserved, r0, r1, c0, c1)

    return markers


def _gen_attempt() -> np.ndarray | None:
    bg = random.randrange(10)
    color_count = random.randint(2, 4)
    colors = random.sample([c for c in range(10) if c != bg], color_count)
    sizes = _choose_sizes(color_count)
    entries = list(zip(colors, sizes, strict=False))
    random.shuffle(entries)
    size_by_color = {color: size for color, size in entries}

    legend_w = sum(size for _color, size in entries) + _LEGEND_GAP * (len(entries) - 1)
    legend_h = max(size for _color, size in entries)
    max_size = max(size_by_color.values())

    min_h = max(_GRID_H_BASE[0], legend_h + max_size + 5)
    max_h = max(min_h, _GRID_H_BASE[1])
    min_w = max(_GRID_W_BASE[0], legend_w + max_size + 6)
    max_w = max(min_w, _GRID_W_BASE[1])
    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)

    grid = np.full((h, w), bg, dtype=int)
    legend_r0 = h - legend_h

    c = 0
    for idx, (color, size) in enumerate(entries):
        r0 = h - size
        grid[r0:h, c : c + size] = color
        c += size
        if idx != len(entries) - 1:
            c += _LEGEND_GAP

    marker_specs = _marker_specs(size_by_color)
    markers = _place_markers((h, w), legend_r0, legend_w, marker_specs, size_by_color)
    if markers is None:
        return None

    for r, c, color in markers:
        grid[r, c] = color

    expected = _build_output(grid, markers, size_by_color)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
