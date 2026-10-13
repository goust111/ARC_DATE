from __future__ import annotations

import numpy as np


SOURCE_COLOR = 4
BACKGROUND = 0
BAND_COLOR_POOL = np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype=int)

MIN_TOP_ROWS = 1
MAX_TOP_ROWS = 3
MIN_BANDS = 3
MAX_BANDS = 5
MIN_BAND_HEIGHT = 3
MAX_BAND_HEIGHT = 5
MIN_WIDTH = 11
MAX_WIDTH = 18
MIN_SOURCES = 2
MAX_SOURCES = 5
GEN_ATTEMPTS = 600

_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _sample_without_replacement(items: list[int], count: int) -> list[int]:
    if count <= 0:
        return []
    picks = np.random.choice(np.array(items, dtype=int), size=count, replace=False)
    return [int(x) for x in np.atleast_1d(picks)]


def _inspect_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    top_rows = 0
    while top_rows < h:
        row_colors = set(map(int, np.unique(g[top_rows])))
        if row_colors <= {BACKGROUND, SOURCE_COLOR}:
            top_rows += 1
        else:
            break
    assert 1 <= top_rows < h, "expected 1-3 source rows above the colored bands"
    assert not np.any(g[top_rows:] == SOURCE_COLOR), "yellow should appear only in the top source rows"

    sources = [(int(r), int(c)) for r, c in np.argwhere(g[:top_rows] == SOURCE_COLOR)]
    assert len(sources) >= 1, "expected at least one yellow source pixel"
    assert len({c for _, c in sources}) == len(sources), "expected at most one source per column"

    bands: list[dict] = []
    r = top_rows
    prev_color = None
    while r < h:
        row = g[r]
        row_colors = sorted(set(map(int, row)) - {BACKGROUND})
        assert len(row_colors) == 1, "each band row must contain exactly one non-black color"
        color = int(row_colors[0])
        assert color != SOURCE_COLOR, "band colors must not use the source color"

        r0 = r
        while r < h:
            row_colors = sorted(set(map(int, g[r])) - {BACKGROUND})
            if len(row_colors) != 1 or int(row_colors[0]) != color:
                break
            r += 1
        r1 = r - 1

        height = r1 - r0 + 1
        assert height >= MIN_BAND_HEIGHT, "each colored band must be at least 3 rows tall"
        crop = g[r0 : r1 + 1]
        non_black = crop[crop != BACKGROUND]
        assert non_black.size > 0 and np.all(non_black == color), "band must be monochrome apart from markers"

        marker_coords = np.argwhere(crop == BACKGROUND)
        assert marker_coords.shape[0] <= 1, "each band may contain at most one marker"
        marker_row = marker_col = None
        if marker_coords.shape[0] == 1:
            rr, cc = marker_coords[0]
            rr = int(rr)
            cc = int(cc)
            assert 0 < rr < height - 1, "marker row must be interior to its band"
            marker_row = int(r0 + rr)
            marker_col = cc

        if prev_color is not None:
            assert color != prev_color, "adjacent bands must have different colors"
        prev_color = color
        bands.append(
            {
                "color": color,
                "r0": int(r0),
                "r1": int(r1),
                "marker_row": marker_row,
                "marker_col": marker_col,
            }
        )

    return {"grid": g, "top_rows": int(top_rows), "sources": sources, "bands": bands}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand band-internal marker rows into black dividers, then stop yellow downward rays at the first divider.

    Input structure:
    - The top 1-3 rows are black except for isolated yellow source pixels, with at most one source per column.
    - Below them, the grid is tightly stacked full-width monochrome horizontal bands of distinct colors.
    - A band may contain one interior black marker pixel; otherwise it is solid.

    Rule:
    - For every band containing a black marker pixel, recolor that entire marker row to black across the full band width.
    - Then, for every yellow source, paint yellow straight downward from the next row in the same column.
    - Each ray stops just above the nearest generated black divider row; if there is no divider below the source area, it reaches the bottom.

    Key insight:
    - The black obstacle is not given directly as a full line: it must first be inferred from a marker inside a specific colored band, and only then can the yellow ray length be determined.
    """
    info = _inspect_grid(grid)
    g = info["grid"]
    h, _w = map(int, g.shape)

    out = g.copy()
    divider_rows = sorted(
        int(band["marker_row"]) for band in info["bands"] if band["marker_row"] is not None
    )
    for row in divider_rows:
        out[row, :] = BACKGROUND

    stop_row = h - 1 if not divider_rows else divider_rows[0] - 1
    for sr, sc in info["sources"]:
        start = sr + 1
        if start <= stop_row:
            out[start : stop_row + 1, sc] = SOURCE_COLOR

    return out


def _branch_marker_count(mode: int, band_count: int) -> int:
    max_markers = min(3, band_count - 1)
    if mode == 0:
        return 0
    if mode == 1:
        return 1
    if mode == 2:
        return min(2, max_markers)
    return int(np.random.randint(1, max_markers + 1))


def _make_candidate() -> np.ndarray:
    global _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1

    top_rows = int(np.random.randint(MIN_TOP_ROWS, MAX_TOP_ROWS + 1))
    band_count = int(np.random.randint(MIN_BANDS, MAX_BANDS + 1))
    band_heights = [int(x) for x in np.random.randint(MIN_BAND_HEIGHT, MAX_BAND_HEIGHT + 1, size=band_count)]
    height = int(top_rows + sum(band_heights))
    width = int(np.random.randint(MIN_WIDTH, MAX_WIDTH + 1))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    band_colors = _sample_without_replacement(list(map(int, BAND_COLOR_POOL)), band_count)

    mode = _GEN_CALL_COUNT % 4
    marker_count = _branch_marker_count(mode, band_count)
    marked_indices = sorted(_sample_without_replacement(list(range(band_count)), marker_count))
    marker_cols = _sample_without_replacement(list(range(width)), marker_count)

    r = top_rows
    for idx, (band_h, color) in enumerate(zip(band_heights, band_colors)):
        grid[r : r + band_h, :] = int(color)
        if idx in marked_indices:
            rel_idx = marked_indices.index(idx)
            marker_row = int(r + np.random.randint(1, band_h - 1))
            marker_col = int(marker_cols[rel_idx])
            grid[marker_row, marker_col] = BACKGROUND
        r += band_h

    source_count = int(np.random.randint(MIN_SOURCES, min(MAX_SOURCES, width) + 1))
    source_cols: list[int] = []
    if marker_cols and len(marker_cols) < width:
        outside = [c for c in range(width) if c not in set(marker_cols)]
        source_cols.append(int(outside[int(np.random.randint(len(outside)))]))
    remaining_cols = [c for c in range(width) if c not in set(source_cols)]
    source_cols.extend(_sample_without_replacement(remaining_cols, source_count - len(source_cols)))
    source_cols = sorted(source_cols)

    source_rows = [int(np.random.randint(0, top_rows)) for _ in range(source_count)]
    if top_rows > 1 and len(set(source_rows)) == 1:
        source_rows[0] = (source_rows[0] + 1) % top_rows

    for sr, sc in zip(source_rows, source_cols):
        grid[int(sr), int(sc)] = SOURCE_COLOR

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    info = _inspect_grid(grid)
    out = main(grid)
    h, w = map(int, grid.shape)
    source_cols = {c for _, c in info["sources"]}
    divider_rows = sorted(
        int(band["marker_row"]) for band in info["bands"] if band["marker_row"] is not None
    )
    marker_cols = {
        int(band["marker_col"]) for band in info["bands"] if band["marker_col"] is not None
    }
    stop_row = h - 1 if not divider_rows else divider_rows[0] - 1

    assert not np.array_equal(out, grid), "output must not equal input"
    assert len(info["bands"]) >= MIN_BANDS
    assert 2 <= len(info["sources"]) <= MAX_SOURCES

    for row in divider_rows:
        assert np.all(out[row, :] == BACKGROUND), "generated divider rows must be fully black"

    for sr, sc in info["sources"]:
        assert out[sr, sc] == SOURCE_COLOR
        if sr + 1 <= stop_row:
            assert np.all(out[sr + 1 : stop_row + 1, sc] == SOURCE_COLOR)
        for r in range(stop_row + 1, h):
            expected = BACKGROUND if r in divider_rows else int(grid[r, sc])
            assert out[r, sc] == expected

    for c in range(w):
        if c in source_cols:
            continue
        for r in range(h):
            expected = BACKGROUND if r in divider_rows else int(grid[r, c])
            assert out[r, c] == expected

    if divider_rows:
        assert any(c not in marker_cols for _, c in info["sources"]), (
            "need at least one source column that is not a marker column"
        )
        cleaned = grid.copy()
        for band in info["bands"]:
            if band["marker_row"] is not None:
                cleaned[int(band["marker_row"]), int(band["marker_col"])] = int(band["color"])
        assert not np.array_equal(main(cleaned), out), "markers must materially affect the output"
        top_divider = divider_rows[0]
        non_marker_sources = [c for _, c in info["sources"] if c not in marker_cols]
        test_col = int(non_marker_sources[0])
        assert out[top_divider, test_col] == BACKGROUND
        assert out[top_divider - 1, test_col] == SOURCE_COLOR
    else:
        for _, c in info["sources"]:
            assert out[h - 1, c] == SOURCE_COLOR, "without dividers, rays must reach the bottom"

    sig = (tuple(map(int, out.shape)), out.tobytes())
    if sig in _SEEN_OUTPUT_SIGS:
        return False
    _SEEN_OUTPUT_SIGS.add(sig)
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_candidate()
        if _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid divider-ray input")
