from __future__ import annotations

from collections import Counter, defaultdict
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_fable_2_027_7",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste

BBox = tuple[int, int, int, int]
TileIndex = tuple[int, int]


def _unique_most_common(counts: Counter[int], what: str) -> int:
    ranked = counts.most_common()
    assert ranked, f"expected at least one {what}"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], f"expected unique {what}"
    return int(ranked[0][0])


def _interior_regions(g: np.ndarray, bg: int, frame: int) -> list[tuple[int, BBox]]:
    regions: list[tuple[int, BBox]] = []
    for coords in components_mask(g != bg, connectivity=4):
        values = set(map(int, g[coords[:, 0], coords[:, 1]]))
        if frame not in values:
            continue

        interior_colors = [color for color in values if color != frame]
        assert len(interior_colors) == 1, "each framed tile must contain one interior color"

        color = int(interior_colors[0])
        interior_coords = coords[g[coords[:, 0], coords[:, 1]] == color]
        r0, c0, r1, c1 = bbox_coords(interior_coords)
        assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == color), "tile interior must be a solid block"
        regions.append((color, (r0, c0, r1, c1)))

    assert regions, "expected at least one framed tile"
    return regions


def _find_target(
    regions: list[tuple[int, BBox]],
    default_color: int,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], BBox]:
    boxes_by_color: dict[int, list[BBox]] = defaultdict(list)
    for color, box in regions:
        boxes_by_color[int(color)].append(box)

    candidates: list[tuple[int, list[tuple[int, int]], list[tuple[int, int]], BBox]] = []
    for color, boxes in boxes_by_color.items():
        if color == int(default_color) or len(boxes) != 3:
            continue

        row_intervals = sorted({(box[0], box[2]) for box in boxes})
        col_intervals = sorted({(box[1], box[3]) for box in boxes})
        if len(row_intervals) != 2 or len(col_intervals) != 2:
            continue

        expected = {(rows, cols) for rows in row_intervals for cols in col_intervals}
        actual = {((box[0], box[2]), (box[1], box[3])) for box in boxes}
        if len(actual) != 3 or not actual.issubset(expected):
            continue

        missing = list(expected - actual)
        if len(missing) != 1:
            continue

        row_band, col_band = missing[0]
        matches = [
            box
            for _other_color, box in regions
            if (box[0], box[2]) == row_band and (box[1], box[3]) == col_band
        ]
        if len(matches) != 1:
            continue

        candidates.append((int(color), row_intervals, col_intervals, matches[0]))

    assert len(candidates) == 1, "expected one marker color forming three rectangle corners"
    return candidates[0]


def _connector_cells(
    g: np.ndarray,
    bg: int,
    row_intervals: list[tuple[int, int]],
    col_intervals: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    top_rows, bottom_rows = row_intervals
    left_cols, right_cols = col_intervals
    cells: set[tuple[int, int]] = set()

    for r0, r1 in row_intervals:
        for r in range(r0, r1 + 1):
            for c in range(left_cols[1] + 1, right_cols[0]):
                if int(g[r, c]) == int(bg):
                    cells.add((r, c))

    for c0, c1 in col_intervals:
        for c in range(c0, c1 + 1):
            for r in range(top_rows[1] + 1, bottom_rows[0]):
                if int(g[r, c]) == int(bg):
                    cells.add((r, c))

    return tuple(sorted(cells))


def _plan_task(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    non_bg = [int(v) for v in g.ravel() if int(v) != int(bg)]
    frame = _unique_most_common(Counter(non_bg), "frame color")

    regions = _interior_regions(g, bg, frame)
    default_color = _unique_most_common(Counter(color for color, _box in regions), "default interior color")
    marker_color, row_intervals, col_intervals, missing_box = _find_target(regions, default_color)

    connectors = _connector_cells(g, bg, row_intervals, col_intervals)
    return {
        "bg": int(bg),
        "frame": int(frame),
        "default": int(default_color),
        "marker": int(marker_color),
        "row_intervals": row_intervals,
        "col_intervals": col_intervals,
        "missing_box": missing_box,
        "connectors": connectors,
        "regions": regions,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing corner tile of a marker rectangle, recolor it, and draw the rectangle edges through background channels.

    Input structure:
    - A background contains separated framed tiles; the frame color is the unique most frequent non-background color.
    - Each tile has one solid interior block, and the unique most common interior color is the default tile color.
    - One special interior color appears in exactly three tiles, and those three tiles occupy three corners of a larger axis-aligned rectangle.

    Rule:
    - Infer the fourth corner tile from the two marker row bands and two marker column bands.
    - Recolor that missing corner tile's solid interior block to the marker color.
    - Paint the same marker color along the four sides of the implied rectangle, but only on background-channel cells; leave tile frames and tile interiors unchanged.

    Key insight:
    - The three marked interiors determine one complete 2x2 corner pattern in tile-band coordinates, so the missing corner recolor and the channel connectors are two consequences of the same rectangle inference.
    """
    g = np.asarray(grid, dtype=int)
    plan = _plan_task(g)

    out = g.copy()
    r0, c0, r1, c1 = plan["missing_box"]
    out[r0 : r1 + 1, c0 : c1 + 1] = int(plan["marker"])

    for r, c in plan["connectors"]:
        out[r, c] = int(plan["marker"])
    return out


GEN_BG = Color.BLACK
GEN_ATTEMPTS = 400
GEN_MARGIN_CHOICES = (1, 2, 3)
GEN_GAP_CHOICES = (1, 2, 3)
GEN_INNER_SIZES = (1, 2, 3, 4)
GEN_TILE_ROWS = (4, 5, 6)
GEN_TILE_COLS = (4, 5, 6)
GEN_RECT_DISTRACTOR_PROB = 0.6


def _layout_options() -> list[tuple[int, int, int, int, int, int, int, int, int, int]]:
    options: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for inner_h in GEN_INNER_SIZES:
        for inner_w in GEN_INNER_SIZES:
            tile_h = inner_h + 2
            tile_w = inner_w + 2
            for gap_r in GEN_GAP_CHOICES:
                for gap_c in GEN_GAP_CHOICES:
                    for tile_rows in GEN_TILE_ROWS:
                        for tile_cols in GEN_TILE_COLS:
                            body_h = tile_rows * tile_h + (tile_rows - 1) * gap_r
                            body_w = tile_cols * tile_w + (tile_cols - 1) * gap_c
                            for top in GEN_MARGIN_CHOICES:
                                for bottom in GEN_MARGIN_CHOICES:
                                    if top + body_h + bottom > 30:
                                        continue
                                    for left in GEN_MARGIN_CHOICES:
                                        for right in GEN_MARGIN_CHOICES:
                                            if left + body_w + right <= 30:
                                                options.append(
                                                    (
                                                        inner_h,
                                                        inner_w,
                                                        gap_r,
                                                        gap_c,
                                                        tile_rows,
                                                        tile_cols,
                                                        top,
                                                        bottom,
                                                        left,
                                                        right,
                                                    )
                                                )
    assert options, "expected at least one valid layout"
    return options


_LAYOUTS = _layout_options()


def _sample_pair(count: int) -> tuple[int, int]:
    assert count >= 2
    span = random.randint(1, count - 1)
    start = random.randint(0, count - span - 1)
    return start, start + span


def _tile_patch(inner_h: int, inner_w: int, frame_color: int, interior_color: int) -> np.ndarray:
    tile = np.full((inner_h + 2, inner_w + 2), int(frame_color), dtype=int)
    tile[1:-1, 1:-1] = int(interior_color)
    return tile


def _tile_origin(layout: tuple[int, int, int, int, int, int, int, int, int, int], tile: TileIndex) -> tuple[int, int]:
    inner_h, inner_w, gap_r, gap_c, _tile_rows, _tile_cols, top, _bottom, left, _right = layout
    period_r = inner_h + 2 + gap_r
    period_c = inner_w + 2 + gap_c
    tr, tc = tile
    return int(top + tr * period_r), int(left + tc * period_c)


def _interior_box(layout: tuple[int, int, int, int, int, int, int, int, int, int], tile: TileIndex) -> BBox:
    inner_h, inner_w, _gap_r, _gap_c, _tile_rows, _tile_cols, _top, _bottom, _left, _right = layout
    r0, c0 = _tile_origin(layout, tile)
    return (r0 + 1, c0 + 1, r0 + inner_h, c0 + inner_w)


def _render_scene(
    layout: tuple[int, int, int, int, int, int, int, int, int, int],
    frame_color: int,
    tile_colors: dict[TileIndex, int],
) -> np.ndarray:
    inner_h, inner_w, gap_r, gap_c, tile_rows, tile_cols, top, bottom, left, right = layout
    tile_h = inner_h + 2
    tile_w = inner_w + 2
    height = top + bottom + tile_rows * tile_h + (tile_rows - 1) * gap_r
    width = left + right + tile_cols * tile_w + (tile_cols - 1) * gap_c

    grid = np.full((height, width), int(GEN_BG), dtype=int)
    cache: dict[int, np.ndarray] = {}
    for tr in range(tile_rows):
        for tc in range(tile_cols):
            color = int(tile_colors[(tr, tc)])
            patch = cache.get(color)
            if patch is None:
                patch = _tile_patch(inner_h, inner_w, frame_color, color)
                cache[color] = patch
            r0, c0 = _tile_origin(layout, (tr, tc))
            paste(grid, patch, r0, c0, transparent=None)
    return grid


def _rectangle_corners(row_pair: tuple[int, int], col_pair: tuple[int, int]) -> set[TileIndex]:
    return {(r, c) for r in row_pair for c in col_pair}


def _available_rectangle(
    tile_rows: int,
    tile_cols: int,
    forbidden: set[TileIndex],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    options: list[tuple[tuple[int, int], tuple[int, int]]] = []
    row_pairs = [(a, b) for a in range(tile_rows) for b in range(a + 1, tile_rows)]
    col_pairs = [(a, b) for a in range(tile_cols) for b in range(a + 1, tile_cols)]
    for row_pair in row_pairs:
        for col_pair in col_pairs:
            corners = _rectangle_corners(row_pair, col_pair)
            if corners.isdisjoint(forbidden):
                options.append((row_pair, col_pair))
    if not options:
        return None
    return random.choice(options)


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(_LAYOUTS)
    inner_h, inner_w, _gap_r, _gap_c, tile_rows, tile_cols, _top, _bottom, _left, _right = layout
    total_tiles = tile_rows * tile_cols

    colors = random.sample(list(range(1, 10)), 5)
    frame_color, default_color, marker_color, distractor_color, rectangle_color = colors

    row_pair = _sample_pair(tile_rows)
    col_pair = _sample_pair(tile_cols)
    target_corners = sorted(_rectangle_corners(row_pair, col_pair))
    missing_corner = random.choice(target_corners)
    marked_corners = [tile for tile in target_corners if tile != missing_corner]

    tile_colors: dict[TileIndex, int] = {
        (tr, tc): int(default_color)
        for tr in range(tile_rows)
        for tc in range(tile_cols)
    }
    for tile in marked_corners:
        tile_colors[tile] = int(marker_color)

    free_tiles = [tile for tile in tile_colors if tile not in set(target_corners)]
    random.shuffle(free_tiles)

    isolated_count = random.choice((1, 2))
    if len(free_tiles) < isolated_count:
        return None
    isolated_tiles = free_tiles[:isolated_count]
    free_tiles = free_tiles[isolated_count:]
    for tile in isolated_tiles:
        tile_colors[tile] = int(distractor_color)

    if random.random() < GEN_RECT_DISTRACTOR_PROB:
        distractor_rect = _available_rectangle(tile_rows, tile_cols, set(target_corners) | set(isolated_tiles))
        if distractor_rect is not None:
            for tile in _rectangle_corners(*distractor_rect):
                tile_colors[tile] = int(rectangle_color)

    default_count = sum(color == int(default_color) for color in tile_colors.values())
    assert default_count > 4 and total_tiles - default_count >= 4

    solved_tile_colors = dict(tile_colors)
    solved_tile_colors[missing_corner] = int(marker_color)

    solved = _render_scene(layout, frame_color, solved_tile_colors)
    boxes = [_interior_box(layout, tile) for tile in target_corners]
    row_intervals = sorted({(box[0], box[2]) for box in boxes})
    col_intervals = sorted({(box[1], box[3]) for box in boxes})
    connectors = _connector_cells(solved, GEN_BG, row_intervals, col_intervals)

    if not connectors:
        return None
    if not any(row_intervals[0][0] <= r <= row_intervals[0][1] for r, _c in connectors):
        return None
    if not any(row_intervals[1][0] <= r <= row_intervals[1][1] for r, _c in connectors):
        return None
    if not any(col_intervals[0][0] <= c <= col_intervals[0][1] for _r, c in connectors):
        return None
    if not any(col_intervals[1][0] <= c <= col_intervals[1][1] for _r, c in connectors):
        return None

    expected = solved.copy()
    for r, c in connectors:
        expected[r, c] = int(marker_color)

    inp = expected.copy()
    mr0, mc0, mr1, mc1 = _interior_box(layout, missing_corner)
    inp[mr0 : mr1 + 1, mc0 : mc1 + 1] = int(default_color)
    for r, c in connectors:
        inp[r, c] = int(GEN_BG)

    try:
        plan = _plan_task(inp)
        out = main(inp)
    except Exception:
        return None

    if int(plan["marker"]) != int(marker_color):
        return None
    if tuple(plan["missing_box"]) != (mr0, mc0, mr1, mc1):
        return None
    if set(plan["connectors"]) != set(connectors):
        return None
    if not np.array_equal(out, expected):
        return None

    missing_pixels = {
        (r, c)
        for r in range(mr0, mr1 + 1)
        for c in range(mc0, mc1 + 1)
    }
    changed = {tuple(map(int, coord)) for coord in np.argwhere(inp != expected)}
    if changed != missing_pixels | set(connectors):
        return None
    if any(int(inp[r, c]) != int(GEN_BG) for r, c in connectors):
        return None
    if any(int(expected[r, c]) != int(marker_color) for r, c in connectors):
        return None
    if any(int(inp[r, c]) == int(marker_color) for r, c in missing_pixels):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-with-missing-corner input")
