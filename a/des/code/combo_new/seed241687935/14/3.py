from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, "expected a unique dominant background color"
    return int(winners[0])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _tiled_patch(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    rr = np.arange(h)[:, None] % th
    cc = np.arange(w)[None, :] % tw
    return tile[rr, cc]


def _separator_row_and_line_color(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])
    ]
    assert rows, "expected a full-width separator row"
    sep = int(min(rows))
    return sep, int(grid[sep, 0])


def _extract_tiles(header: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    tiles: dict[int, np.ndarray] = {}
    for comp in _components(header != bg):
        r0, c0, r1, c1 = _bbox(comp)
        patch = header[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.size > 0 and np.all(patch != bg), "legend blocks must be solid rectangles without gaps"
        base = _dominant_color(patch)
        assert base not in tiles, "legend base colors must be unique"
        tiles[base] = patch.copy()
    assert tiles, "expected at least one legend block"
    return tiles


def _extract_regions(panel: np.ndarray, bg: int, line_color: int) -> list[tuple[int, int, int, int, int]]:
    regions: list[tuple[int, int, int, int, int]] = []
    ph, pw = panel.shape
    for comp in _components(panel != line_color):
        assert not _touches_border(comp, ph, pw), "regions must be enclosed by the line network"
        r0, c0, r1, c1 = _bbox(comp)
        patch = panel[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != line_color)
        non_bg = np.argwhere(patch != bg)
        assert non_bg.shape == (1, 2), "each region must contain exactly one marker pixel"
        mr, mc = map(int, non_bg[0])
        marker = int(patch[mr, mc])
        assert np.count_nonzero(patch == marker) == 1, "marker color must appear only once in its region"
        regions.append((r0, c0, r1, c1, marker))
    assert regions, "expected at least one enclosed region"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with the tiled legend pattern selected by its marker color.

    Input structure:
    - The background is the unique dominant color.
    - A single foreground line color forms a full-width separator row and a connected
      rectilinear wall network below it, creating closed rectangular regions.
    - Above the separator, the header contains several rectangular color blocks on the
      background; each block has one dominant base color plus a smaller overlaid motif
      in a second color.
    - Each closed region below contains exactly one single-pixel marker, and every
      marker color matches the dominant base color of exactly one header block.

    Rule:
    - Find the header block whose dominant base color matches each region's marker color.
    - Treat that entire header block as a tile, including both its base color and its
      overlaid motif.
    - Replace the full interior of the matching region with repeated copies of that
      tile, aligned from the region's own top-left corner and clipped at the edges.
    - Keep the wall network and the header unchanged; the marker is overwritten.

    Key insight:
    - The top strip is a color-keyed legend of reusable tiles: solve the puzzle by
      first isolating the wall-bounded regions, then matching each marker color to its
      header block, and finally tiling that block across the whole region interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    sep_row, line_color = _separator_row_and_line_color(g, bg)
    assert 0 < sep_row < g.shape[0] - 1, "separator row must leave room for header and regions"

    header = g[:sep_row, :]
    tiles = _extract_tiles(header, bg)

    panel = g[sep_row:, :]
    assert np.all(panel[0, :] == line_color), "panel must start with the separator row"
    assert np.all(panel[-1, :] == line_color), "panel must end with a bottom border"
    assert np.all(panel[:, 0] == line_color) and np.all(panel[:, -1] == line_color), "panel must have side borders"

    regions = _extract_regions(panel, bg, line_color)
    assert len(regions) == len(tiles), "legend block count and region count must match"

    out = g.copy()
    seen_markers: set[int] = set()
    for r0, c0, r1, c1, marker in regions:
        assert marker in tiles, "every marker color must map to a legend block"
        assert marker not in seen_markers, "marker colors must be unique across regions"
        seen_markers.add(marker)
        out[sep_row + r0 : sep_row + r1 + 1, c0 : c1 + 1] = _tiled_patch(
            tiles[marker], (r1 - r0 + 1, c1 - c0 + 1)
        )

    return out


# --------------- generator ---------------

_ARC_COLORS = tuple(range(10))
_LAYOUTS = ((1, 2), (1, 3), (1, 4), (2, 1), (2, 2))
_LEGEND_HEIGHTS = (2, 3, 4)
_REGION_MIN, _REGION_MAX = 4, 8
_BLOCK_MIN_W, _BLOCK_MAX_W = 2, 4
_GEN_ATTEMPTS = 500


def _pattern_candidates(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    area = h * w
    out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    def add(coords: list[tuple[int, int]]) -> None:
        cleaned = tuple(sorted({(int(r), int(c)) for r, c in coords if 0 <= r < h and 0 <= c < w}))
        if not cleaned or len(cleaned) * 2 >= area:
            return
        if cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)

    add([(0, 0)])
    add([(0, w - 1)])
    add([(h - 1, 0)])
    add([(h - 1, w - 1)])
    add([(h // 2, w // 2)])
    add([(0, w // 2)])
    add([(h - 1, w // 2)])
    add([(h // 2, 0)])
    add([(h // 2, w - 1)])
    add([(0, 0), (h - 1, w - 1)])
    add([(0, w - 1), (h - 1, 0)])
    add([(0, w // 2), (h - 1, w // 2)])
    add([(h // 2, 0), (h // 2, w - 1)])
    add([(0, 0), (0, w - 1), (h - 1, 0)])
    add([(0, 0), (0, w - 1), (h - 1, w - 1)])
    add([(0, 0), (h - 1, 0), (h - 1, w - 1)])
    add([(0, w - 1), (h - 1, 0), (h - 1, w - 1)])
    add([(0, 0), (1, min(1, w - 1)), (min(2, h - 1), min(2, w - 1))])
    add([(0, w - 1), (1, max(0, w - 2)), (min(2, h - 1), max(0, w - 3))])
    assert len(out) >= 4, "need enough pattern variants for legend blocks"
    return out


def _sample_block_widths(count: int, total_width: int) -> list[int] | None:
    min_span = count * _BLOCK_MIN_W + (count - 1)
    if total_width < min_span:
        return None

    widths = [_BLOCK_MIN_W] * count
    budget = min(total_width - (count - 1) - sum(widths), count * (_BLOCK_MAX_W - _BLOCK_MIN_W))
    indices = list(range(count))
    random.shuffle(indices)
    while budget > 0 and indices:
        i = random.choice(indices)
        if widths[i] == _BLOCK_MAX_W:
            indices = [j for j in indices if widths[j] < _BLOCK_MAX_W]
            continue
        widths[i] += 1
        budget -= 1
    return widths


def _tile_from_mask(h: int, w: int, base: int, accent: int, mask: tuple[tuple[int, int], ...]) -> np.ndarray:
    tile = np.full((h, w), base, dtype=int)
    for r, c in mask:
        tile[r, c] = accent
    return tile


def _region_boxes(legend_h: int, heights: list[int], widths: list[int]) -> list[tuple[int, int, int, int]]:
    row_edges = [legend_h]
    for h in heights:
        row_edges.append(row_edges[-1] + h + 1)

    col_edges = [0]
    for w in widths:
        col_edges.append(col_edges[-1] + w + 1)

    boxes: list[tuple[int, int, int, int]] = []
    for r in range(len(heights)):
        top = row_edges[r] + 1
        bottom = top + heights[r] - 1
        for c in range(len(widths)):
            left = col_edges[c] + 1
            right = left + widths[c] - 1
            boxes.append((top, left, bottom, right))
    return boxes


def _render_expected(inp: np.ndarray, region_map: dict[int, tuple[np.ndarray, tuple[int, int, int, int]]]) -> np.ndarray:
    out = inp.copy()
    for marker, (tile, (r0, c0, r1, c1)) in region_map.items():
        out[r0 : r1 + 1, c0 : c1 + 1] = _tiled_patch(tile, (r1 - r0 + 1, c1 - c0 + 1))
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rows, n_cols = random.choice(_LAYOUTS)
    n_regions = n_rows * n_cols
    legend_h = random.choice(_LEGEND_HEIGHTS)

    heights = [random.randint(_REGION_MIN, _REGION_MAX) for _ in range(n_rows)]
    widths = [random.randint(_REGION_MIN, _REGION_MAX) for _ in range(n_cols)]
    grid_h = legend_h + sum(heights) + n_rows + 1
    grid_w = sum(widths) + n_cols + 1
    if grid_h > 30 or grid_w > 30:
        return None

    block_widths = _sample_block_widths(n_regions, grid_w)
    if block_widths is None:
        return None

    needed_colors = 2 * n_regions + 2
    if needed_colors > len(_ARC_COLORS):
        return None
    colors = random.sample(_ARC_COLORS, needed_colors)
    bg = colors[0]
    line = colors[1]
    markers = colors[2 : 2 + n_regions]
    accents = colors[2 + n_regions :]

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    inp[legend_h, :] = line
    inp[-1, :] = line

    col = 0
    for width in widths:
        inp[legend_h:, col] = line
        col += width + 1
    inp[legend_h:, -1] = line

    row = legend_h
    for height in heights[:-1]:
        row += height + 1
        inp[row, :] = line

    region_boxes = _region_boxes(legend_h, heights, widths)
    marker_order = list(markers)
    random.shuffle(marker_order)

    used_masks: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    tile_by_marker: dict[int, np.ndarray] = {}
    for marker, accent, bw in zip(marker_order, accents, block_widths):
        candidates = _pattern_candidates(legend_h, bw)
        random.shuffle(candidates)
        chosen = None
        for mask in candidates:
            key = (legend_h, bw, mask)
            if key not in used_masks:
                used_masks.add(key)
                chosen = mask
                break
        if chosen is None:
            return None
        tile_by_marker[marker] = _tile_from_mask(legend_h, bw, marker, accent, chosen)

    legend_items = [(marker, tile_by_marker[marker]) for marker in marker_order]
    random.shuffle(legend_items)
    span = sum(tile.shape[1] for _, tile in legend_items) + (n_regions - 1)
    left_margin = random.randint(0, grid_w - span)
    c = left_margin
    for marker, tile in legend_items:
        th, tw = tile.shape
        inp[:th, c : c + tw] = tile
        c += tw + 1

    region_map: dict[int, tuple[np.ndarray, tuple[int, int, int, int]]] = {}
    boxes = list(region_boxes)
    random.shuffle(boxes)
    for marker, box in zip(marker_order, boxes):
        r0, c0, r1, c1 = box
        mr = random.randint(r0, r1)
        mc = random.randint(c0, c1)
        inp[mr, mc] = marker
        region_map[marker] = (tile_by_marker[marker], box)

    try:
        if _dominant_color(inp) != bg:
            return None
    except AssertionError:
        return None

    expected = _render_expected(inp, region_map)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tiled-region ARC input")
