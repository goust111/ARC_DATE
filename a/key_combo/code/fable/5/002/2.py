from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


def _mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    assert counts, "expected nonempty values"
    return int(counts.most_common(1)[0][0])


def _border_values(g: np.ndarray) -> np.ndarray:
    top = g[0, :]
    bottom = g[-1, :]
    left = g[1:-1, 0]
    right = g[1:-1, -1]
    return np.concatenate([top, bottom, left, right])


def _background_from_border(g: np.ndarray) -> int:
    bg = _mode(_border_values(g))
    border = _border_values(g)
    assert int(np.count_nonzero(border == bg)) > int(border.size // 2), "outer ring must be mostly background"
    assert int(g[0, 0]) == bg and int(g[0, -1]) == bg
    assert int(g[-1, 0]) == bg and int(g[-1, -1]) == bg
    return int(bg)


def _separator_lines(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    inner = g[1:-1, 1:-1]
    assert inner.ndim == 2 and min(inner.shape) >= 3, "expected a nontrivial inner tiled area"
    candidates: list[tuple[int, list[int], list[int]]] = []
    for raw in np.unique(inner):
        color = int(raw)
        if color == int(bg):
            continue
        rows = [r for r in range(inner.shape[0]) if np.all(inner[r, :] == color)]
        cols = [c for c in range(inner.shape[1]) if np.all(inner[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color with full inner rows and columns"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in sorted(map(int, separators)):
        assert start < sep, "separator lines must leave nonempty tile bands"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final tile band"
    spans.append((start, length - 1))
    return spans


def _span_index(spans: list[tuple[int, int]], coord: int) -> int:
    coord = int(coord)
    hits = [i for i, (lo, hi) in enumerate(spans) if lo <= coord <= hi]
    assert len(hits) == 1, "anchor projection must fall inside exactly one tile band"
    return int(hits[0])


def _tile_slice(row_span: tuple[int, int], col_span: tuple[int, int]) -> tuple[slice, slice]:
    r0, r1 = row_span
    c0, c1 = col_span
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def _locate_template(
    g: np.ndarray,
    bg: int,
    separator: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
) -> tuple[tuple[int, int], np.ndarray]:
    templates: list[tuple[tuple[int, int], np.ndarray]] = []
    for tr, rs in enumerate(row_spans):
        for tc, cs in enumerate(col_spans):
            tile = g[_tile_slice(rs, cs)]
            colors = {int(c) for c in np.unique(tile)}
            assert int(separator) not in colors, "separator color should only occur on separator lines"
            non_bg = colors - {int(bg)}
            if not non_bg:
                assert np.all(tile == int(bg)), "plain tiles must be pure background"
                continue
            assert len(non_bg) >= 2, "template tile must contain a multicolor pattern"
            templates.append(((tr, tc), tile.copy()))
    assert len(templates) == 1, "expected exactly one multicolor template tile"
    return templates[0]


def _border_anchors(g: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    H, W = g.shape
    anchors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for c in range(W):
        if int(g[0, c]) != int(bg):
            anchors[int(g[0, c])].append((0, c))
        if int(g[H - 1, c]) != int(bg):
            anchors[int(g[H - 1, c])].append((H - 1, c))
    for r in range(1, H - 1):
        if int(g[r, 0]) != int(bg):
            anchors[int(g[r, 0])].append((r, 0))
        if int(g[r, W - 1]) != int(bg):
            anchors[int(g[r, W - 1])].append((r, W - 1))
    assert anchors, "expected at least one same-color border anchor pair"
    return anchors


def _transform_template(template: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    tile = template.copy()
    if bool(flip_ud):
        tile = np.flipud(tile)
    if bool(flip_lr):
        tile = np.fliplr(tile)
    return tile.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp a unique template tile into border-anchor intersections, mirroring it according to the anchor edges.

    Input structure:
    - The outer ring is mostly background, with non-background border anchors on
      top/bottom and left/right edges.
    - Inside that ring, full one-cell separator rows and columns split the area
      into equal rectangular tiles.
    - Exactly one tile contains a multicolor template pattern; every other tile
      is pure background in the input.

    Rule:
    - Pair border anchors by color; each pair has one anchor on a horizontal
      edge and one anchor on a vertical edge.
    - The horizontal anchor's column selects a tile column band, and the
      vertical anchor's row selects a tile row band; their intersection is the
      target tile.
    - Copy the template into each target tile, first flipping vertically when
      the horizontal anchor is on the bottom edge, then flipping horizontally
      when the vertical anchor is on the right edge.
    - Keep the original template tile, separator lines, and border anchors
      unchanged.

    Key insight:
    - The target location is not marked inside the tile: it is the row/column
      projection intersection of a same-color border pair, and the two touched
      edges encode the two independent mirror choices.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 8, "expected a 2D ARC grid with an outer ring"
    H, W = map(int, g.shape)

    bg = _background_from_border(g)
    separator, sep_rows_inner, sep_cols_inner = _separator_lines(g, bg)
    row_spans_inner = _spans(H - 2, sep_rows_inner)
    col_spans_inner = _spans(W - 2, sep_cols_inner)
    row_spans = [(lo + 1, hi + 1) for lo, hi in row_spans_inner]
    col_spans = [(lo + 1, hi + 1) for lo, hi in col_spans_inner]

    tile_heights = {hi - lo + 1 for lo, hi in row_spans}
    tile_widths = {hi - lo + 1 for lo, hi in col_spans}
    assert len(tile_heights) == 1 and len(tile_widths) == 1, "tile bands must be equal sized"
    tile_h = int(next(iter(tile_heights)))
    tile_w = int(next(iter(tile_widths)))

    source_pos, template = _locate_template(g, bg, separator, row_spans, col_spans)
    assert template.shape == (tile_h, tile_w)

    out = g.copy()
    used_targets: set[tuple[int, int]] = set()
    for color, cells in _border_anchors(g, bg).items():
        assert len(cells) == 2, "each anchor color must appear exactly twice on the border"
        horizontal = [(r, c) for r, c in cells if r in (0, H - 1) and c not in (0, W - 1)]
        vertical = [(r, c) for r, c in cells if c in (0, W - 1) and r not in (0, H - 1)]
        assert len(horizontal) == 1 and len(vertical) == 1, "each anchor pair needs one horizontal and one vertical edge anchor"

        hr, hc = horizontal[0]
        vr, vc = vertical[0]
        tile_r = _span_index(row_spans, vr)
        tile_c = _span_index(col_spans, hc)
        target = (tile_r, tile_c)
        assert target != source_pos, "anchor pairs must point to background target tiles"
        assert target not in used_targets, "target tiles must be uniquely selected"
        used_targets.add(target)

        patch = _transform_template(template, flip_ud=(hr == H - 1), flip_lr=(vc == W - 1))
        rs, cs = _tile_slice(row_spans[tile_r], col_spans[tile_c])
        assert np.all(g[rs, cs] == int(bg)), "target tiles must be background in the input"
        out[rs, cs] = patch

    return out


# ---------------- generator ----------------

COLORS = tuple(range(10))
TILE_ROWS_CHOICES = (3, 4)
TILE_COLS_CHOICES = (3, 4)
TILE_H_RANGE = (4, 6)
TILE_W_RANGE = (4, 6)
PAIR_COUNT_RANGE = (4, 5)
GEN_ATTEMPTS = 500

ORIENTATIONS = (
    ("top_left", False, False),
    ("bottom_left", True, False),
    ("top_right", False, True),
    ("bottom_right", True, True),
)


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return 1 + int(tile_r) * (int(tile_h) + 1), 1 + int(tile_c) * (int(tile_w) + 1)


def _base_grid(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int, bg: int, separator: int) -> np.ndarray:
    H = 2 + int(tile_rows) * int(tile_h) + (int(tile_rows) - 1)
    W = 2 + int(tile_cols) * int(tile_w) + (int(tile_cols) - 1)
    grid = np.full((H, W), int(bg), dtype=int)
    for tr in range(1, int(tile_rows)):
        r = 1 + tr * int(tile_h) + (tr - 1)
        grid[r, 1 : W - 1] = int(separator)
    for tc in range(1, int(tile_cols)):
        c = 1 + tc * int(tile_w) + (tc - 1)
        grid[1 : H - 1, c] = int(separator)
    return grid


def _variants(template: np.ndarray) -> list[np.ndarray]:
    return [
        template.copy(),
        np.flipud(template).copy(),
        np.fliplr(template).copy(),
        np.fliplr(np.flipud(template)).copy(),
    ]


def _all_variants_distinct(template: np.ndarray) -> bool:
    variants = _variants(template)
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            if np.array_equal(variants[i], variants[j]):
                return False
    return True


def _connected_cells(h: int, w: int, count: int) -> set[tuple[int, int]]:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    while len(cells) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    return cells


def _make_template(tile_h: int, tile_w: int, bg: int, colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        tile = np.full((tile_h, tile_w), int(bg), dtype=int)

        # Fixed asymmetric signal pixels make every mirror branch visibly distinct.
        anchors = [
            (0, 1, colors[0]),
            (1, tile_w - 1, colors[1]),
            (tile_h - 1, max(0, tile_w - 3), colors[0]),
            (max(1, tile_h - 2), 0, colors[-1]),
        ]
        for r, c, color in anchors:
            tile[int(r), int(c)] = int(color)

        blob_color = random.choice(colors)
        blob_size = random.randint(3, max(3, min(tile_h * tile_w // 3, 7)))
        for r, c in _connected_cells(tile_h, tile_w, blob_size):
            if random.random() < 0.75:
                tile[r, c] = int(blob_color)

        for _dot in range(random.randint(2, 5)):
            r = random.randrange(tile_h)
            c = random.randrange(tile_w)
            tile[r, c] = int(random.choice(colors))

        if len({int(c) for c in np.unique(tile)} - {int(bg)}) < 2:
            continue
        if not _all_variants_distinct(tile):
            continue
        # Keep at least one background cell so stamping reads as a pattern, not a solid tile.
        if not np.any(tile == int(bg)):
            continue
        return tile
    return None


def _orientation_edges(flip_ud: bool, flip_lr: bool, H: int, W: int) -> tuple[int, int]:
    h_edge_r = H - 1 if bool(flip_ud) else 0
    v_edge_c = W - 1 if bool(flip_lr) else 0
    return int(h_edge_r), int(v_edge_c)


def _place_anchor_pair(
    grid: np.ndarray,
    color: int,
    target: tuple[int, int],
    flip_ud: bool,
    flip_lr: bool,
    tile_h: int,
    tile_w: int,
    used: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    tr, tc = target
    r0, c0 = _tile_origin(tr, tc, tile_h, tile_w)
    h_edge_r, v_edge_c = _orientation_edges(flip_ud, flip_lr, H, W)

    col_choices = list(range(c0, c0 + tile_w))
    row_choices = list(range(r0, r0 + tile_h))
    random.shuffle(col_choices)
    random.shuffle(row_choices)
    for hc in col_choices:
        h_cell = (h_edge_r, hc)
        if h_cell in used:
            continue
        for vr in row_choices:
            v_cell = (vr, v_edge_c)
            if v_cell in used:
                continue
            grid[h_cell] = int(color)
            grid[v_cell] = int(color)
            used.add(h_cell)
            used.add(v_cell)
            return True
    return False


def _stamp_tile(grid: np.ndarray, pos: tuple[int, int], tile: np.ndarray, tile_h: int, tile_w: int) -> None:
    r0, c0 = _tile_origin(pos[0], pos[1], tile_h, tile_w)
    grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile


def _manual_expected(
    inp: np.ndarray,
    source_pos: tuple[int, int],
    template: np.ndarray,
    placements: list[tuple[tuple[int, int], bool, bool]],
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    out = inp.copy()
    for target, flip_ud, flip_lr in placements:
        if target == source_pos:
            continue
        _stamp_tile(out, target, _transform_template(template, flip_ud, flip_lr), tile_h, tile_w)
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, placements: list[tuple[tuple[int, int], bool, bool]]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(inp, expected):
        return False
    flips = {(bool(ud), bool(lr)) for _target, ud, lr in placements}
    if flips != {(False, False), (True, False), (False, True), (True, True)}:
        return False
    targets = [target for target, _ud, _lr in placements]
    return len(targets) == len(set(targets))


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.choice(TILE_ROWS_CHOICES)
    tile_cols = random.choice(TILE_COLS_CHOICES)
    tile_h = random.randint(*TILE_H_RANGE)
    tile_w = random.randint(*TILE_W_RANGE)

    bg, separator = random.sample(COLORS, 2)
    template_color_count = random.choice([2, 3])
    template_colors = random.sample([c for c in COLORS if c not in {bg, separator}], template_color_count)
    remaining_colors = [c for c in COLORS if c not in {bg, separator, *template_colors}]

    pair_count = random.randint(*PAIR_COUNT_RANGE)
    pair_count = min(pair_count, tile_rows * tile_cols - 1, len(remaining_colors))
    if pair_count < 4:
        return None

    template = _make_template(tile_h, tile_w, bg, template_colors)
    if template is None:
        return None

    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    source_pos = random.choice(positions)
    targets = random.sample([p for p in positions if p != source_pos], pair_count)

    orientation_specs = list(ORIENTATIONS)
    while len(orientation_specs) < pair_count:
        orientation_specs.append(random.choice(ORIENTATIONS))
    random.shuffle(orientation_specs)

    grid = _base_grid(tile_rows, tile_cols, tile_h, tile_w, bg, separator)
    _stamp_tile(grid, source_pos, template, tile_h, tile_w)

    used_anchor_cells: set[tuple[int, int]] = set()
    anchor_colors = random.sample(remaining_colors, pair_count)
    placements: list[tuple[tuple[int, int], bool, bool]] = []
    for target, color, (_name, flip_ud, flip_lr) in zip(targets, anchor_colors, orientation_specs):
        ok = _place_anchor_pair(grid, color, target, flip_ud, flip_lr, tile_h, tile_w, used_anchor_cells)
        if not ok:
            return None
        placements.append((target, flip_ud, flip_lr))

    expected = _manual_expected(grid, source_pos, template, placements, tile_h, tile_w)
    if not _semantic_checks(grid, expected, placements):
        return None
    assert max(grid.shape) <= 30
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid border-anchor mirror-stamp input after {GEN_ATTEMPTS} attempts")
