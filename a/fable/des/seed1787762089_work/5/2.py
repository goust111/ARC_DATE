import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
RECOLOR_COLORS = (2, 4, 6)
SOURCE_COLORS = (1, 3, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

LAYOUT_CHOICES = (
    (2, 3),
    (3, 2),
    (2, 4),
    (4, 2),
    (3, 3),
    (3, 4),
    (4, 3),
    (4, 4),
)
TILE_HEIGHT_RANGE = (4, 7)
TILE_WIDTH_RANGE = (4, 7)

CORNER_TO_POS = {
    "tl": lambda h, w: (0, 0),
    "tr": lambda h, w: (0, w - 1),
    "bl": lambda h, w: (h - 1, 0),
    "br": lambda h, w: (h - 1, w - 1),
}


def _transform_mask(mask: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return mask.copy()
    if corner == "tr":
        return np.fliplr(mask)
    if corner == "bl":
        return np.flipud(mask)
    if corner == "br":
        return np.flipud(np.fliplr(mask))
    raise AssertionError(f"unknown corner: {corner}")


def _uniform_line_color(values: list[int], other_values: list[int]) -> int:
    candidates = set(values) & set(other_values)
    assert len(candidates) == 1, "expected exactly one separator color"
    return int(next(iter(candidates)))


def _non_separator_runs(sep_mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, is_sep in enumerate(sep_mask.tolist(), start=0):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(sep_mask)))
    return runs


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    full_rows = [int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])]
    full_cols = [int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])]
    grey = _uniform_line_color(full_rows, full_cols)

    sep_rows = np.array([bool(np.all(g[r, :] == grey)) for r in range(h)], dtype=bool)
    sep_cols = np.array([bool(np.all(g[:, c] == grey)) for c in range(w)], dtype=bool)
    row_runs = _non_separator_runs(sep_rows)
    col_runs = _non_separator_runs(sep_cols)
    assert len(row_runs) >= 2 and len(col_runs) >= 2, "expected multiple tiles"

    row_heights = {r1 - r0 for r0, r1 in row_runs}
    col_widths = {c1 - c0 for c0, c1 in col_runs}
    assert len(row_heights) == 1 and len(col_widths) == 1, "tiles must be equal-sized"
    tile_h = row_runs[0][1] - row_runs[0][0]
    tile_w = col_runs[0][1] - col_runs[0][0]
    assert tile_h >= 4 and tile_w >= 4

    interior = g[np.ix_(~sep_rows, ~sep_cols)]
    bg = int(Counter(interior.ravel().tolist()).most_common(1)[0][0])
    assert bg != grey

    pos_to_corner = {
        (0, 0): "tl",
        (0, tile_w - 1): "tr",
        (tile_h - 1, 0): "bl",
        (tile_h - 1, tile_w - 1): "br",
    }

    source = None
    markers: list[dict] = []
    for ri, (r0, r1) in enumerate(row_runs):
        for ci, (c0, c1) in enumerate(col_runs):
            patch = g[r0:r1, c0:c1]
            coords = np.argwhere(patch != bg)
            if coords.shape[0] == 0:
                continue
            if coords.shape[0] == 1:
                rr, cc = map(int, coords[0])
                corner = pos_to_corner.get((rr, cc))
                assert corner is not None, "marker must sit on an interior tile corner"
                markers.append(
                    {
                        "tile": (ri, ci),
                        "bounds": (r0, r1, c0, c1),
                        "corner": corner,
                        "color": int(patch[rr, cc]),
                    }
                )
                continue

            colors = {int(v) for v in patch[patch != bg]}
            assert len(colors) == 1, "source tile must be monochrome"
            source_color = int(next(iter(colors)))
            assert source_color != grey, "source color must differ from separator color"
            assert source is None, "expected exactly one source tile"
            source = {
                "tile": (ri, ci),
                "bounds": (r0, r1, c0, c1),
                "mask": patch == source_color,
                "color": source_color,
            }

    assert source is not None, "missing source tile"
    assert markers, "missing marked tiles"

    return {
        "grid": g,
        "grey": grey,
        "bg": bg,
        "row_runs": row_runs,
        "col_runs": col_runs,
        "source": source,
        "markers": markers,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the unique source pattern into each marked tile.

    The marker corner chooses the orientation:
    - top-left: identity
    - top-right: horizontal reflection
    - bottom-left: vertical reflection
    - bottom-right: 180-degree rotation

    Grey markers keep the source color; non-grey markers recolor the full stamp.
    Source tile, blank tiles, and separator lines stay unchanged.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    bg = info["bg"]
    grey = info["grey"]
    source_mask = info["source"]["mask"]
    source_color = info["source"]["color"]

    out = g.copy()
    for marker in info["markers"]:
        r0, r1, c0, c1 = marker["bounds"]
        patch = out[r0:r1, c0:c1]
        patch[:] = bg
        transformed = _transform_mask(source_mask, marker["corner"])
        stamp_color = source_color if marker["color"] == grey else marker["color"]
        patch[transformed] = stamp_color
    return out


def _balanced_cycle(items: tuple[int, ...] | tuple[str, ...], length: int) -> list:
    out = []
    while len(out) < length:
        block = list(items)
        random.shuffle(block)
        out.extend(block)
    return out[:length]


def _keep_flags(length: int) -> list[bool]:
    keep_count = length // 2
    if length % 2 == 1:
        keep_count += random.randint(0, 1)
    flags = [True] * keep_count + [False] * (length - keep_count)
    random.shuffle(flags)
    return flags


def _neighbors(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    result = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            result.append((rr, cc))
    return result


def _random_source_mask(tile_h: int, tile_w: int) -> np.ndarray | None:
    max_cells = min(tile_h * tile_w // 2, tile_h + tile_w + 2, 10)
    for _ in range(200):
        target_cells = random.randint(3, max(3, max_cells))
        cells = {(0, 0)}
        frontier = []
        for pos in _neighbors(0, 0, tile_h, tile_w):
            frontier.append(pos)

        while len(cells) < target_cells and frontier:
            pick = random.choice(frontier)
            frontier.remove(pick)
            if pick in cells:
                continue
            cells.add(pick)
            for nxt in _neighbors(pick[0], pick[1], tile_h, tile_w):
                if nxt not in cells and nxt not in frontier:
                    frontier.append(nxt)

        if len(cells) != target_cells:
            continue

        mask = np.zeros((tile_h, tile_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.where(mask)
        if rows.size < 3:
            continue
        if rows.max() == 0 or cols.max() == 0:
            continue
        if mask[0, tile_w - 1] or mask[tile_h - 1, 0] or mask[tile_h - 1, tile_w - 1]:
            continue

        variants = {
            _transform_mask(mask, corner).tobytes()
            for corner in ("tl", "tr", "bl", "br")
        }
        if len(variants) != 4:
            continue

        return mask
    return None


def _build_bounds(count: int, tile_len: int) -> tuple[list[tuple[int, int]], list[int], int]:
    bounds: list[tuple[int, int]] = []
    separators: list[int] = []
    pos = 0
    for idx in range(count):
        bounds.append((pos, pos + tile_len))
        pos += tile_len
        if idx + 1 < count:
            separators.append(pos)
            pos += 1
    return bounds, separators, pos


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols = random.choice(LAYOUT_CHOICES)
    tile_h = random.randint(*TILE_HEIGHT_RANGE)
    tile_w = random.randint(*TILE_WIDTH_RANGE)

    source_mask = _random_source_mask(tile_h, tile_w)
    if source_mask is None:
        return None

    row_bounds, sep_rows, height = _build_bounds(tile_rows, tile_h)
    col_bounds, sep_cols, width = _build_bounds(tile_cols, tile_w)

    grid = np.full((height, width), BLACK, dtype=int)
    if sep_rows:
        grid[sep_rows, :] = GREY
    if sep_cols:
        grid[:, sep_cols] = GREY

    tile_positions = [(ri, ci) for ri in range(tile_rows) for ci in range(tile_cols)]
    source_tile = random.choice(tile_positions)
    source_color = random.choice(SOURCE_COLORS)
    sr0, sr1 = row_bounds[source_tile[0]]
    sc0, sc1 = col_bounds[source_tile[1]]
    source_patch = grid[sr0:sr1, sc0:sc1]
    source_patch[source_mask] = source_color

    remaining = [pos for pos in tile_positions if pos != source_tile]
    if len(remaining) < 4:
        return None

    max_targets = len(remaining)
    min_targets = 4
    if max_targets > 4 and random.random() < 0.6:
        target_count = random.randint(min_targets, max_targets - 1)
    else:
        target_count = max_targets

    target_tiles = random.sample(remaining, target_count)
    orientations = _balanced_cycle(("tl", "tr", "bl", "br"), target_count)
    keep_flags = _keep_flags(target_count)
    recolor_pool = _balanced_cycle(RECOLOR_COLORS, max(1, keep_flags.count(False)))
    recolor_idx = 0

    for tile, corner, keep_source in zip(target_tiles, orientations, keep_flags, strict=False):
        r0, r1 = row_bounds[tile[0]]
        c0, c1 = col_bounds[tile[1]]
        mr, mc = CORNER_TO_POS[corner](tile_h, tile_w)
        marker_color = GREY
        if not keep_source:
            marker_color = recolor_pool[recolor_idx]
            recolor_idx += 1
        grid[r0 + mr, c0 + mc] = marker_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
