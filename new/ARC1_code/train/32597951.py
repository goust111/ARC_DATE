from collections import Counter

import numpy as np

from arc_utils import Color, bbox, colors_present


def _foreground_roles(g: np.ndarray) -> tuple[int, int]:
    nonzero = colors_present(g, exclude={Color.BLACK})
    assert len(nonzero) == 2, "expected exactly two non-black colors"

    counts = Counter(map(int, g.reshape(-1)))
    marker_candidates = [color for color in nonzero if counts[color] == min(counts[c] for c in nonzero)]
    assert len(marker_candidates) == 1, "expected a unique sparse marker color"
    marker = int(marker_candidates[0])
    pattern = int(next(color for color in nonzero if color != marker))
    assert counts[marker] < counts[pattern], "marker must be less frequent than the pattern color"
    return marker, pattern


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the non-black pattern pixels inside the sparse marker mask's bounding rectangle to green.

    Input structure
    - The grid contains black empty pixels and exactly two non-black colors.
    - One non-black color is a sparse marker mask, uniquely less frequent than the recurring pattern color.
    - Green is absent from the input and is used as the output highlight color.

    Rule
    - Find the bounding rectangle of the sparse marker mask.
    - Within that rectangle, recolor every pixel of the other non-black color to green.
    - Preserve black pixels, marker pixels, and all pixels outside the rectangle.

    Key insight
    - The marker pixels identify a rectangular window; the transformation highlights only the underlying pattern pixels in that window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == Color.BLACK), "expected black empty pixels"
    assert not np.any(g == Color.GREEN), "green must not already appear in the input"

    marker, pattern = _foreground_roles(g)
    marker_bbox = bbox(g == marker)
    assert marker_bbox is not None, "expected marker pixels"
    r0, c0, r1, c1 = marker_bbox

    out = g.copy()
    window = out[r0 : r1 + 1, c0 : c1 + 1]
    pattern_mask = window == pattern
    assert np.any(pattern_mask), "marker bbox must contain pattern pixels"
    window[pattern_mask] = Color.GREEN
    return out

# --------------- generator (o5x) ---------------
# Canvas and fixed colors.
_GEN_GRID_SIZE = 17
_GEN_BLACK, _GEN_BLUE, _GEN_GREEN, _GEN_MARKER = Color.BLACK, Color.BLUE, Color.GREEN, Color.TEAL

# Full-grid pattern controls.
_GEN_RANDOM_PATTERN_PROB = 0.5
_GEN_MIN_BLUE, _GEN_MAX_BLUE = 96, 144
_GEN_MIN_PERIOD_ROWS, _GEN_MAX_PERIOD_ROWS = 2, 4
_GEN_MIN_PERIOD_COLS, _GEN_MAX_PERIOD_COLS = 3, 4
_GEN_TILE_MIN_DENSITY_DIVISOR, _GEN_TILE_MAX_DENSITY_DIVISOR = 3, 2

# Marker rectangle controls.
_GEN_BBOX_MARGIN = 1
_GEN_MIN_BBOX_HEIGHT, _GEN_MAX_BBOX_HEIGHT = 3, 5
_GEN_MIN_BBOX_WIDTH, _GEN_MAX_BBOX_WIDTH = 5, 8
_GEN_MIN_MARKERS, _GEN_MAX_MARKERS = 12, 17
_GEN_DIM_FIRST_BBOX_PROB = 0.5
_GEN_RETRY_LIMIT = 300
_GenBox = tuple[int, int, int, int]
_GenBuckets = dict[tuple[int, int], dict[int, list[_GenBox]]]
_GEN_PERIODIC_CACHE: dict[tuple[int, int], list[np.ndarray]] | None = None


def _gen_choice(items):
    return items[int(np.random.randint(len(items)))]


def _gen_repeat_tile(tile: np.ndarray, period_r: int, period_c: int) -> np.ndarray:
    rr = np.arange(_GEN_GRID_SIZE) % period_r
    cc = np.arange(_GEN_GRID_SIZE) % period_c
    return tile[rr[:, None], cc[None, :]].astype(int)


def _gen_exact_period(tile: np.ndarray, period_r: int, period_c: int) -> bool:
    for dr in range(1, period_r):
        if np.array_equal(tile, np.roll(tile, -dr, axis=0)):
            return False
    for dc in range(1, period_c):
        if np.array_equal(tile, np.roll(tile, -dc, axis=1)):
            return False
    return True


def _gen_periodic_options() -> dict[tuple[int, int], list[np.ndarray]]:
    global _GEN_PERIODIC_CACHE
    if _GEN_PERIODIC_CACHE is not None:
        return _GEN_PERIODIC_CACHE

    cache: dict[tuple[int, int], list[np.ndarray]] = {}
    for period_r in range(_GEN_MIN_PERIOD_ROWS, _GEN_MAX_PERIOD_ROWS + 1):
        for period_c in range(_GEN_MIN_PERIOD_COLS, _GEN_MAX_PERIOD_COLS + 1):
            area = period_r * period_c
            min_on = (area + _GEN_TILE_MIN_DENSITY_DIVISOR - 1) // _GEN_TILE_MIN_DENSITY_DIVISOR
            max_on = area // _GEN_TILE_MAX_DENSITY_DIVISOR
            tiles = []
            for bits in range(1 << area):
                if not min_on <= bits.bit_count() <= max_on:
                    continue
                tile = np.zeros((period_r, period_c), dtype=int)
                for idx in range(area):
                    if bits & (1 << idx):
                        tile[idx // period_c, idx % period_c] = _GEN_BLUE
                if not _gen_exact_period(tile, period_r, period_c):
                    continue
                blue_count = int(np.count_nonzero(_gen_repeat_tile(tile, period_r, period_c)))
                if _GEN_MIN_BLUE <= blue_count <= _GEN_MAX_BLUE:
                    tiles.append(tile)
            if tiles:
                cache[(period_r, period_c)] = tiles

    _GEN_PERIODIC_CACHE = cache
    return cache


def _gen_random_pattern() -> np.ndarray:
    blue_count = int(np.random.randint(_GEN_MIN_BLUE, _GEN_MAX_BLUE + 1))
    flat = np.zeros(_GEN_GRID_SIZE * _GEN_GRID_SIZE, dtype=int)
    flat[np.random.choice(flat.size, size=blue_count, replace=False)] = _GEN_BLUE
    return flat.reshape((_GEN_GRID_SIZE, _GEN_GRID_SIZE))


def _gen_periodic_pattern() -> np.ndarray | None:
    options = _gen_periodic_options()
    if not options:
        return None
    periods = sorted(options)
    period_r, period_c = _gen_choice(periods)
    tiles = options[(period_r, period_c)]
    tile = _gen_choice(tiles)
    return _gen_repeat_tile(tile, period_r, period_c)


def _gen_pattern() -> np.ndarray | None:
    if np.random.random() < _GEN_RANDOM_PATTERN_PROB:
        return _gen_random_pattern()
    return _gen_periodic_pattern()


def _gen_bbox_buckets(pattern: np.ndarray) -> _GenBuckets:
    buckets: _GenBuckets = {}
    for height in range(_GEN_MIN_BBOX_HEIGHT, _GEN_MAX_BBOX_HEIGHT + 1):
        for width in range(_GEN_MIN_BBOX_WIDTH, _GEN_MAX_BBOX_WIDTH + 1):
            row_stop = _GEN_GRID_SIZE - height - _GEN_BBOX_MARGIN + 1
            col_stop = _GEN_GRID_SIZE - width - _GEN_BBOX_MARGIN + 1
            for r0 in range(_GEN_BBOX_MARGIN, row_stop):
                for c0 in range(_GEN_BBOX_MARGIN, col_stop):
                    window = pattern[r0 : r0 + height, c0 : c0 + width]
                    marker_mask = window == _GEN_BLACK
                    marker_count = int(np.count_nonzero(marker_mask))
                    if not _GEN_MIN_MARKERS <= marker_count <= _GEN_MAX_MARKERS:
                        continue
                    if marker_count == height * width:
                        continue
                    if not (
                        np.any(marker_mask[0])
                        and np.any(marker_mask[-1])
                        and np.any(marker_mask[:, 0])
                        and np.any(marker_mask[:, -1])
                    ):
                        continue
                    by_count = buckets.setdefault((height, width), {})
                    by_count.setdefault(marker_count, []).append(
                        (r0, c0, r0 + height - 1, c0 + width - 1)
                    )
    return buckets


def _gen_pick_bbox_dim_first(by_dim: _GenBuckets) -> _GenBox:
    dim = _gen_choice(sorted(by_dim))
    by_count = by_dim[dim]
    marker_count = _gen_choice(sorted(by_count))
    return _gen_choice(by_count[marker_count])


def _gen_pick_bbox_count_first(by_dim: _GenBuckets) -> _GenBox:
    by_count_all: dict[int, dict[tuple[int, int], list[_GenBox]]] = {}
    for dim, counts in by_dim.items():
        for marker_count, boxes in counts.items():
            by_count_all.setdefault(marker_count, {})[dim] = boxes
    marker_count = _gen_choice(sorted(by_count_all))
    dim = _gen_choice(sorted(by_count_all[marker_count]))
    return _gen_choice(by_dim[dim][marker_count])


def _gen_pick_bbox(pattern: np.ndarray) -> _GenBox | None:
    by_dim = _gen_bbox_buckets(pattern)
    if not by_dim:
        return None

    if np.random.random() < _GEN_DIM_FIRST_BBOX_PROB:
        return _gen_pick_bbox_dim_first(by_dim)
    return _gen_pick_bbox_count_first(by_dim)


def _gen_render_pair(pattern: np.ndarray, box: _GenBox) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, r1, c1 = box

    out = pattern.copy()
    out_window = out[r0 : r1 + 1, c0 : c1 + 1]
    marker_mask = out_window == _GEN_BLACK
    out_window[marker_mask] = _GEN_MARKER
    out_window[~marker_mask] = _GEN_GREEN

    inp = out.copy()
    inp_window = inp[r0 : r1 + 1, c0 : c1 + 1]
    inp_window[inp_window == _GEN_GREEN] = _GEN_BLUE

    return inp, out


def _gen_passes_oracle(inp: np.ndarray, out: np.ndarray) -> bool:
    if int(np.count_nonzero(inp == _GEN_MARKER)) >= int(np.count_nonzero(inp == _GEN_BLUE)):
        return False
    return np.array_equal(main(inp), out)


def _gen_attempt() -> np.ndarray | None:
    pattern = _gen_pattern()
    if pattern is None:
        return None

    box = _gen_pick_bbox(pattern)
    if box is None:
        return None

    inp, out = _gen_render_pair(pattern, box)
    if not _gen_passes_oracle(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid 32597951 input")
