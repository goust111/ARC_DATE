from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
REGION_COLORS = (1, 2, 3, 4, 6, 7, 9)
SYMBOLS = ("hflip", "vflip", "rot90", "identity")
FACTOR_CHOICES = {
    4: ((2, 2), (1, 4), (4, 1)),
    6: ((2, 3), (3, 2), (1, 6), (6, 1)),
}
MAX_GENERATION_ATTEMPTS = 300


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(comp)
    return out


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _classify_symbol(mask_black: np.ndarray) -> str:
    coords = np.argwhere(mask_black)
    assert coords.size > 0, "each region must contain a black instruction symbol"
    if len(coords) == 1:
        return "identity"
    if len(coords) == 3:
        rows = sorted({int(r) for r, _ in coords})
        cols = sorted({int(c) for _, c in coords})
        if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
            return "hflip"
        if len(cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
            return "vflip"
    if len(coords) == 4:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if int(r1 - r0 + 1) == 2 and int(c1 - c0 + 1) == 2 and mask_black[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].all():
            return "rot90"
    raise AssertionError("unrecognized instruction symbol")


def _apply_transform(mask: np.ndarray, symbol: str) -> np.ndarray:
    if symbol == "identity":
        return mask.copy()
    if symbol == "hflip":
        return np.fliplr(mask)
    if symbol == "vflip":
        return np.flipud(mask)
    if symbol == "rot90":
        return np.rot90(mask, -1)
    raise AssertionError(symbol)


def _find_map_and_thumbnail(g: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected the map rectangle and the thumbnail"
    map_bbox = None
    thumb_bbox = None
    for comp in comps:
        b = _bbox(comp)
        patch = g[b[0] : b[2] + 1, b[1] : b[3] + 1]
        colors = set(map(int, np.unique(patch)))
        if GREY in colors:
            assert thumb_bbox is None, "expected a unique grey thumbnail"
            thumb_bbox = b
        else:
            assert map_bbox is None, "expected a unique colored map"
            map_bbox = b
    assert map_bbox is not None and thumb_bbox is not None
    return map_bbox, thumb_bbox


def _thumbnail_layout(g: np.ndarray, thumb_bbox: tuple[int, int, int, int], bg: int) -> dict[int, tuple[int, int]]:
    tr0, tc0, tr1, tc1 = thumb_bbox
    patch = g[tr0 : tr1 + 1, tc0 : tc1 + 1]
    positions: dict[int, tuple[int, int]] = {}
    for r, c in np.argwhere((patch != bg) & (patch != GREY)):
        color = int(patch[r, c])
        assert color not in positions, "each region color must appear once in the thumbnail"
        positions[color] = (int(r), int(c))
    assert positions, "thumbnail must contain colored guide pixels"
    rows = sorted({r for r, _ in positions.values()})
    cols = sorted({c for _, c in positions.values()})
    assert len(rows) * len(cols) == len(positions), "thumbnail must define a complete layout grid"
    row_index = {r: i for i, r in enumerate(rows)}
    col_index = {c: i for i, c in enumerate(cols)}
    return {color: (row_index[r], col_index[c]) for color, (r, c) in positions.items()}


def _extract_regions(
    g: np.ndarray,
    map_bbox: tuple[int, int, int, int],
    layout: dict[int, tuple[int, int]],
    bg: int,
) -> dict[int, np.ndarray]:
    mr0, mc0, mr1, mc1 = map_bbox
    map_patch = g[mr0 : mr1 + 1, mc0 : mc1 + 1]
    out: dict[int, np.ndarray] = {}
    for color in layout:
        comps = _components(g == color)
        assert len(comps) >= 2, "expected a map region and a thumbnail pixel for each color"
        region_comp = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(region_comp)
        assert mr0 <= r0 <= r1 <= mr1 and mc0 <= c0 <= c1 <= mc1, "largest same-color component must be inside the map"
        patch = map_patch[r0 - mr0 : r1 - mr0 + 1, c0 - mc0 : c1 - mc0 + 1]
        allowed = {color, BLACK}
        assert set(map(int, np.unique(patch))).issubset(allowed), "region bbox may only contain its own color and black"
        color_mask = patch == color
        black_mask = patch == BLACK
        assert np.all(color_mask | black_mask), "regions must be rectangular with only black cut-outs"
        symbol = _classify_symbol(black_mask)
        transformed = _apply_transform(color_mask, symbol)
        out[int(color)] = transformed
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each colored region according to its embedded black symbol, then
    reassemble the transformed regions using the grey thumbnail layout.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    map_bbox, thumb_bbox = _find_map_and_thumbnail(g, bg)
    layout = _thumbnail_layout(g, thumb_bbox, bg)
    transformed = _extract_regions(g, map_bbox, layout, bg)

    row_heights: dict[int, int] = {}
    col_widths: dict[int, int] = {}
    for color, (rr, cc) in layout.items():
        patch = transformed[color]
        h, w = map(int, patch.shape)
        if rr in row_heights:
            assert row_heights[rr] == h, "pieces in the same thumbnail row must share a height"
        else:
            row_heights[rr] = h
        if cc in col_widths:
            assert col_widths[cc] == w, "pieces in the same thumbnail column must share a width"
        else:
            col_widths[cc] = w

    n_rows = max(row_heights) + 1
    n_cols = max(col_widths) + 1
    assert set(row_heights) == set(range(n_rows))
    assert set(col_widths) == set(range(n_cols))

    row_offsets = [0]
    for r in range(n_rows):
        row_offsets.append(row_offsets[-1] + row_heights[r])
    col_offsets = [0]
    for c in range(n_cols):
        col_offsets.append(col_offsets[-1] + col_widths[c])

    out = np.full((row_offsets[-1], col_offsets[-1]), bg, dtype=int)
    for color, (rr, cc) in layout.items():
        patch = transformed[color]
        r0 = row_offsets[rr]
        c0 = col_offsets[cc]
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w][patch] = int(color)
    return out


def _balanced_symbols(n_regions: int) -> list[str]:
    symbols = list(SYMBOLS)
    extras = list(SYMBOLS)
    random.shuffle(extras)
    while len(symbols) < n_regions:
        symbols.append(extras.pop())
    random.shuffle(symbols)
    return symbols


def _symbol_mask(size: int, symbol: str, top: int, left: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    if symbol == "identity":
        mask[top, left] = True
    elif symbol == "hflip":
        mask[top, left : left + 3] = True
    elif symbol == "vflip":
        mask[top : top + 3, left] = True
    elif symbol == "rot90":
        mask[top : top + 2, left : left + 2] = True
    else:
        raise AssertionError(symbol)
    return mask


def _symbol_shape(symbol: str) -> tuple[int, int]:
    if symbol == "identity":
        return 1, 1
    if symbol == "hflip":
        return 1, 3
    if symbol == "vflip":
        return 3, 1
    if symbol == "rot90":
        return 2, 2
    raise AssertionError(symbol)


def _make_piece(size: int, color: int, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    color_mask = np.ones((size, size), dtype=bool)
    sh, sw = _symbol_shape(symbol)
    choices: list[tuple[int, int]] = []
    for top in range(1, size - sh):
        for left in range(1, size - sw):
            black_mask = _symbol_mask(size, symbol, top, left)
            if symbol != "identity" and np.array_equal(
                _apply_transform(color_mask & ~black_mask, symbol),
                color_mask & ~black_mask,
            ):
                continue
            choices.append((top, left))
    assert choices, "piece size must leave at least one non-degenerate symbol location"
    top, left = random.choice(choices)
    black_mask = _symbol_mask(size, symbol, top, left)

    patch = np.full((size, size), color, dtype=int)
    patch[black_mask] = BLACK
    transformed = _apply_transform(patch == color, symbol)
    return patch, transformed


def _layout_choices(n_regions: int) -> tuple[int, int]:
    options = list(FACTOR_CHOICES[n_regions])
    weights = []
    for r, c in options:
        weights.append(3 if r > 1 and c > 1 else 1)
    return random.choices(options, weights=weights, k=1)[0]


def _arrangement(shape: tuple[int, int], colors: list[int]) -> list[list[int]]:
    rows, cols = shape
    shuffled = colors[:]
    random.shuffle(shuffled)
    return [shuffled[i * cols : (i + 1) * cols] for i in range(rows)]


def _render_map(layout: list[list[int]], pieces: dict[int, np.ndarray], bg: int) -> np.ndarray:
    rows = len(layout)
    cols = len(layout[0])
    cell_h, cell_w = next(iter(pieces.values())).shape
    out = np.full((rows * cell_h, cols * cell_w), bg, dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = layout[r][c]
            patch = pieces[color]
            r0 = r * cell_h
            c0 = c * cell_w
            out[r0 : r0 + cell_h, c0 : c0 + cell_w] = patch
    return out


def _render_thumbnail(layout: list[list[int]]) -> np.ndarray:
    rows = len(layout)
    cols = len(layout[0])
    thumb = np.full((2 * rows + 1, 2 * cols + 1), GREY, dtype=int)
    for r in range(rows):
        for c in range(cols):
            thumb[1 + 2 * r, 1 + 2 * c] = int(layout[r][c])
    return thumb


def _render_expected(layout: list[list[int]], transformed: dict[int, np.ndarray], bg: int) -> np.ndarray:
    rows = len(layout)
    cols = len(layout[0])
    cell_h, cell_w = next(iter(transformed.values())).shape
    out = np.full((rows * cell_h, cols * cell_w), bg, dtype=int)
    for r in range(rows):
        for c in range(cols):
            color = layout[r][c]
            patch = transformed[color]
            r0 = r * cell_h
            c0 = c * cell_w
            out[r0 : r0 + cell_h, c0 : c0 + cell_w][patch] = int(color)
    return out


def _rectangles_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_on_canvas(map_patch: np.ndarray, thumb_patch: np.ndarray) -> np.ndarray | None:
    map_h, map_w = map_patch.shape
    thumb_h, thumb_w = thumb_patch.shape
    for _ in range(80):
        canvas_h = max(map_h, thumb_h) + random.randint(4, 10)
        canvas_w = max(map_w, thumb_w) + random.randint(4, 12)
        if canvas_h < map_h + 2 or canvas_h < thumb_h + 2 or canvas_w < map_w + 2 or canvas_w < thumb_w + 2:
            continue

        mr = random.randint(1, canvas_h - map_h - 1)
        mc = random.randint(1, canvas_w - map_w - 1)
        tr = random.randint(1, canvas_h - thumb_h - 1)
        tc = random.randint(1, canvas_w - thumb_w - 1)
        map_rect = (mr, mc, mr + map_h - 1, mc + map_w - 1)
        thumb_rect = (tr, tc, tr + thumb_h - 1, tc + thumb_w - 1)
        if not _rectangles_separated(map_rect, thumb_rect, gap=1):
            continue

        canvas = np.full((canvas_h, canvas_w), AZURE, dtype=int)
        canvas[mr : mr + map_h, mc : mc + map_w] = map_patch
        canvas[tr : tr + thumb_h, tc : tc + thumb_w] = thumb_patch
        return canvas
    return None


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choice((4, 6))
    colors = random.sample(REGION_COLORS, n_regions)
    symbols = _balanced_symbols(n_regions)
    size = random.randint(6, 8)

    pieces: dict[int, np.ndarray] = {}
    transformed: dict[int, np.ndarray] = {}
    changed = 0
    for color, symbol in zip(colors, symbols):
        patch, transformed_mask = _make_piece(size, color, symbol)
        pieces[color] = patch
        transformed[color] = transformed_mask
        changed += int(not np.array_equal(patch == color, transformed_mask))
    if changed == 0:
        return None

    in_shape = _layout_choices(n_regions)
    out_shape = _layout_choices(n_regions)
    in_layout = _arrangement(in_shape, colors)
    out_layout = _arrangement(out_shape, colors)
    if in_shape == out_shape and in_layout == out_layout:
        return None

    map_patch = _render_map(in_layout, pieces, AZURE)
    thumb_patch = _render_thumbnail(out_layout)
    expected = _render_expected(out_layout, transformed, AZURE)
    canvas = _place_on_canvas(map_patch, thumb_patch)
    if canvas is None:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    counts = Counter(symbols)
    if any(counts[symbol] == 0 for symbol in SYMBOLS):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
