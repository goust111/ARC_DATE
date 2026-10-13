from __future__ import annotations

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(m[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(m[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _is_rectangular_ring(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3 or int(comp["area"]) != 2 * (h + w) - 4:
        return False

    patch = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _solid_square_size(comp: dict) -> int | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or int(comp["area"]) != h * w:
        return None
    return h


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    frame_candidates = [comp for comp in _find_components(grid, bg) if _is_rectangular_ring(comp)]
    assert frame_candidates, "expected at least one rectangular frame"

    by_color: dict[int, list[dict]] = {}
    for comp in frame_candidates:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    frame_color, frames = max(by_color.items(), key=lambda item: (len(item[1]), -item[0]))
    return int(frame_color), frames


def _legend_entries(comps: list[dict], h: int, w: int, frame_color: int, bg: int) -> list[int]:
    groups: list[tuple[int, str, list[dict]]] = []

    for side in ("left", "bottom"):
        by_size: dict[int, list[dict]] = {}
        for comp in comps:
            color = int(comp["color"])
            if color in (bg, frame_color):
                continue
            size = _solid_square_size(comp)
            if size is None:
                continue

            r0, c0, r1, c1 = comp["bbox"]
            if side == "left" and c0 == 0:
                by_size.setdefault(size, []).append(comp)
            if side == "bottom" and r1 == h - 1:
                by_size.setdefault(size, []).append(comp)

        for size, group in by_size.items():
            if len(group) < 2:
                continue
            ordered = sorted(group, key=lambda comp: comp["bbox"][0] if side == "left" else comp["bbox"][1])
            groups.append((len(ordered), side, ordered))

    assert groups, "expected an edge legend made of at least two solid squares"
    _count, side, ordered = max(groups, key=lambda item: (item[0], 1 if item[1] == "left" else 0))
    _ = side
    return [int(comp["color"]) for comp in ordered]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with the legend color indexed by its internal marker count.

    Input structure:
    - The background is black.
    - A legend of distinct-colored square entries sits on the left edge or bottom edge, ordered top-to-bottom or left-to-right.
    - The main area contains several same-colored hollow rectangular frames, and each frame encloses some isolated same-colored marker pixels on black background.

    Rule:
    - Read the legend as a 1-based lookup table: the first entry means count 1, the second means count 2, and so on.
    - For each hollow rectangular frame, count how many isolated marker components of the frame color lie strictly inside it.
    - Replace the entire interior of that frame, including the marker pixels, with the legend color at that count index.
    - Keep the frame outlines and the legend itself unchanged.

    Key insight:
    The tiny interior same-color markers are counters, not decoration: their quantity selects a fill color from the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h, w = g.shape

    comps = _find_components(g, bg)
    frame_color, frames = _find_frames(g, bg)
    legend = _legend_entries(comps, h, w, frame_color, bg)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_count = len(_mask_components(inner == frame_color))
        assert 1 <= marker_count <= len(legend), "marker count must index the legend"
        out[r0 + 1 : r1, c0 + 1 : c1] = int(legend[marker_count - 1])

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_ATTEMPTS = 400
GEN_H_RANGE = (16, 31)
GEN_W_RANGE = (16, 31)
GEN_LEGEND_LEN_RANGE = (2, 6)
GEN_SQUARE_SIZES = (1, 2)
GEN_GAPS = (1, 2)
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_COLORS = tuple(range(1, 10))
GEN_FRAME_RETRIES = 80


def _mark_rect(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_start(limit: int, span: int, rng: np.random.Generator) -> int:
    slack = limit - span
    if slack <= 0:
        return 0
    if slack == 1:
        return 0
    return int(rng.integers(0, slack))


def _place_legend(
    grid: np.ndarray,
    occ: np.ndarray,
    colors: list[int],
    side: str,
    size: int,
    gap: int,
    rng: np.random.Generator,
) -> None:
    h, w = grid.shape
    n = len(colors)
    span = n * size + (n - 1) * gap

    if side == "left":
        start = _legend_start(h, span, rng)
        for i, color in enumerate(colors):
            r0 = start + i * (size + gap)
            grid[r0 : r0 + size, 0:size] = int(color)
        _mark_rect(occ, start, 0, start + span - 1, size - 1, pad=1)
    else:
        start = 1 if w - span > 1 else 0
        if w - span > 2:
            start = int(rng.integers(1, w - span))
        for i, color in enumerate(colors):
            c0 = start + i * (size + gap)
            grid[h - size : h, c0 : c0 + size] = int(color)
        _mark_rect(occ, h - size, start, h - 1, start + span - 1, pad=1)


def _marker_positions(ih: int, iw: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    if ih < 3 or iw < 3:
        return None
    parity_cells = [
        [(r, c) for r in range(1, ih - 1) for c in range(1, iw - 1) if (r + c) % 2 == parity]
        for parity in (0, 1)
    ]
    valid = [cells for cells in parity_cells if len(cells) >= count]
    if not valid:
        return None
    cells = valid[int(rng.integers(len(valid)))]
    pick = rng.choice(len(cells), size=count, replace=False)
    return [cells[int(i)] for i in pick]


def _frame_positions(occ: np.ndarray, oh: int, ow: int) -> list[tuple[int, int]]:
    h, w = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - oh):
        for c0 in range(1, w - ow):
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            if not occ[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                positions.append((r0, c0))
    return positions


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    ih: int,
    iw: int,
    frame_color: int,
    markers: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    r1 = r0 + ih + 1
    c1 = c0 + iw + 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    for mr, mc in markers:
        grid[r0 + 1 + mr, c0 + 1 + mc] = frame_color
    return r0, c0, r1, c1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GEN_H_RANGE))
    w = int(rng.integers(*GEN_W_RANGE))
    grid = np.full((h, w), GEN_BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    side = "left" if bool(rng.integers(2)) else "bottom"
    legend_len = int(rng.integers(*GEN_LEGEND_LEN_RANGE))
    size = int(GEN_SQUARE_SIZES[int(rng.integers(len(GEN_SQUARE_SIZES)))])
    gap = int(GEN_GAPS[int(rng.integers(len(GEN_GAPS)))])
    span = legend_len * size + (legend_len - 1) * gap
    if (side == "left" and span > h - 2) or (side == "bottom" and span > w - 2):
        return None

    color_order = list(rng.choice(np.array(GEN_COLORS, dtype=int), size=legend_len + 1, replace=False))
    frame_color = int(color_order[0])
    legend_colors = [int(c) for c in color_order[1:]]
    _place_legend(grid, occ, legend_colors, side, size, gap, rng)

    frame_count = int(rng.integers(*GEN_FRAME_COUNT_RANGE))
    base_counts = list(range(1, legend_len + 1))
    rng.shuffle(base_counts)
    counts = base_counts[: min(frame_count, legend_len)]
    while len(counts) < frame_count:
        counts.append(int(rng.integers(1, legend_len + 1)))
    if len(set(counts)) < 2:
        counts[-1] = 1 if counts[0] != 1 else legend_len
    rng.shuffle(counts)

    frames: list[tuple[int, int, int, int, int]] = []
    for count in counts:
        placed = False
        for _ in range(GEN_FRAME_RETRIES):
            ih = int(rng.integers(5, 9))
            iw = int(rng.integers(5, 9))
            if ((ih - 2) * (iw - 2) + 1) // 2 < count:
                continue
            markers = _marker_positions(ih, iw, count, rng)
            if markers is None:
                continue
            oh = ih + 2
            ow = iw + 2
            positions = _frame_positions(occ, oh, ow)
            if not positions:
                continue
            r0, c0 = positions[int(rng.integers(len(positions)))]
            bbox = _draw_frame(grid, r0, c0, ih, iw, frame_color, markers)
            _mark_rect(occ, *bbox, pad=1)
            frames.append((*bbox, count))
            placed = True
            break
        if not placed:
            return None

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[int(cnts.argmax())]) != GEN_BG:
        return None

    expected = grid.copy()
    for r0, c0, r1, c1, count in frames:
        expected[r0 + 1 : r1, c0 + 1 : c1] = legend_colors[count - 1]

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
