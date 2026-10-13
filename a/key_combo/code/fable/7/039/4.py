"""ARC task: decode a stamp's D4 transform from an anchor-image marker."""

from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _d4(tile: np.ndarray) -> tuple[np.ndarray, ...]:
    """The eight square symmetries, in a fixed but semantically irrelevant order."""
    reflected = np.fliplr(tile)
    return tuple(
        [np.rot90(tile, k).copy() for k in range(4)]
        + [np.rot90(reflected, k).copy() for k in range(4)]
    )


def _separator_structure(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    candidates = []
    height, width = map(int, g.shape)
    for value in np.unique(g):
        color = int(value)
        rows = [r for r in range(height) if np.all(g[r, :] == color)]
        cols = [c for c in range(width) if np.all(g[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color forming full rows and columns"
    return candidates[0]


def _tile_spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for line in separators:
        assert line == start or line > start, "separator lines must be ordered"
        assert line > start, "tiles and separator lines must alternate"
        spans.append((start, line))
        start = line + 1
    assert start < length, "a final tile must follow the last separator"
    spans.append((start, length))
    return spans


def _unique_mode(values: np.ndarray) -> int:
    ranked = Counter(map(int, values.ravel())).most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), \
        "background must be uniquely most frequent"
    return int(ranked[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp a uniquely transformed template into each tile whose anchor marker encodes that transform.

    Input structure:
    - Full rows and columns of one separator color divide the grid into equal square tiles.
    - One tile contains a multicolor asymmetric template with one uniquely colored anchor pixel.
    - Some otherwise blank tiles contain one anchor-colored marker, while the remaining tiles are blank.

    Rule:
    - In each marked tile, select the unique one of the square's eight symmetries that maps the
      template anchor to the marker's local coordinate, then replace that tile by the transformed template.
    - Preserve the source tile, blank tiles, and separator lines.

    Key insight:
    - A marker is the image of the off-axis anchor, so its position identifies the entire rotation or
      reflection to apply to the template rather than merely indicating where an unchanged copy belongs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size, "expected a nonempty 2D grid"

    separator, sep_rows, sep_cols = _separator_structure(g)
    row_spans = _tile_spans(g.shape[0], sep_rows)
    col_spans = _tile_spans(g.shape[1], sep_cols)
    heights = {b - a for a, b in row_spans}
    widths = {b - a for a, b in col_spans}
    assert len(heights) == len(widths) == 1
    tile_size = heights.pop()
    assert tile_size == widths.pop() and tile_size >= 2, "expected equal square tiles"

    interior = np.concatenate(
        [g[r0:r1, c0:c1].ravel() for r0, r1 in row_spans for c0, c1 in col_spans]
    )
    bg = _unique_mode(interior)
    assert bg != separator

    tiles = []
    source_candidates = []
    singleton_tiles = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0:r1, c0:c1]
            assert separator not in set(map(int, np.unique(tile))), \
                "separator color may occur only on separator lines"
            non_bg = np.argwhere(tile != bg)
            colors = set(map(int, np.unique(tile[tile != bg])))
            record = (r0, c0, tile.copy(), non_bg, colors)
            tiles.append(record)
            if len(colors) >= 2:
                source_candidates.append(record)
            elif len(non_bg) == 1:
                singleton_tiles.append(record)
            else:
                assert len(non_bg) == 0, "non-source tiles must be blank or singleton-marked"

    assert len(source_candidates) == 1, "expected one multicolor template tile"
    assert singleton_tiles, "expected at least one marked target tile"
    source = source_candidates[0]
    marker_colors = {next(iter(record[4])) for record in singleton_tiles}
    assert len(marker_colors) == 1, "all singleton markers must share one color"
    anchor_color = marker_colors.pop()
    assert anchor_color in source[4], "marker color must also occur in the template"
    anchor_coords = np.argwhere(source[2] == anchor_color)
    assert len(anchor_coords) == 1, "template must contain one anchor pixel"

    template = source[2]
    variants = _d4(template)
    anchor_images = []
    for variant in variants:
        coords = np.argwhere(variant == anchor_color)
        assert len(coords) == 1
        anchor_images.append(tuple(map(int, coords[0])))
    assert len(set(anchor_images)) == 8, "anchor must have eight distinct symmetry images"

    by_anchor = {coord: variant for coord, variant in zip(anchor_images, variants)}
    out = g.copy()
    marked_origins = {(record[0], record[1]) for record in singleton_tiles}
    for r0, c0, _tile, non_bg, _colors in tiles:
        if (r0, c0) not in marked_origins:
            continue
        marker = tuple(map(int, non_bg[0]))
        assert marker in by_anchor, "marker must lie in the template anchor's D4 orbit"
        out[r0:r0 + tile_size, c0:c0 + tile_size] = by_anchor[marker]
    return out


# Generator constants deliberately keep every grid at or below 30x30.
_COLORS = tuple(range(10))
_MAX_ATTEMPTS = 300


def _orbit(n: int, r: int, c: int) -> set[tuple[int, int]]:
    marker = np.zeros((n, n), dtype=int)
    marker[r, c] = 1
    return {
        tuple(map(int, np.argwhere(variant == 1)[0]))
        for variant in _d4(marker)
    }


def _make_template(n: int, bg: int, anchor_color: int, sprite_colors: tuple[int, int]) \
        -> tuple[np.ndarray, tuple[int, int]]:
    anchor_choices = [
        (r, c) for r in range(n) for c in range(n)
        if len(_orbit(n, r, c)) == 8
    ]
    anchor = random.choice(anchor_choices)
    target_area = random.randint(max(7, n + 2), min(n * n // 2, 14))

    occupied = {anchor}
    while len(occupied) < target_area:
        frontier = {
            (r + dr, c + dc)
            for r, c in occupied
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < n and 0 <= c + dc < n and (r + dr, c + dc) not in occupied
        }
        if not frontier:
            break
        occupied.add(random.choice(tuple(frontier)))

    template = np.full((n, n), bg, dtype=int)
    template[anchor] = anchor_color
    others = list(occupied - {anchor})
    random.shuffle(others)
    # Both ordinary sprite colors are visibly present; their ids encode nothing.
    for i, (r, c) in enumerate(others):
        template[r, c] = sprite_colors[i % 2]
    return template, anchor


def _render(tile_rows: int, tile_cols: int, n: int, separator: int, bg: int,
            template: np.ndarray, anchor_color: int, source_index: int,
            target_specs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    height = tile_rows * n + tile_rows - 1
    width = tile_cols * n + tile_cols - 1
    inp = np.full((height, width), bg, dtype=int)
    inp[n::n + 1, :] = separator
    inp[:, n::n + 1] = separator
    expected = inp.copy()
    variants = _d4(template)

    def origin(index: int) -> tuple[int, int]:
        tr, tc = divmod(index, tile_cols)
        return tr * (n + 1), tc * (n + 1)

    sr, sc = origin(source_index)
    inp[sr:sr + n, sc:sc + n] = template
    expected[sr:sr + n, sc:sc + n] = template
    for index, transform_index in target_specs:
        r0, c0 = origin(index)
        transformed = variants[transform_index]
        marker = np.argwhere(transformed == anchor_color)
        assert len(marker) == 1
        mr, mc = map(int, marker[0])
        inp[r0 + mr, c0 + mc] = anchor_color
        expected[r0:r0 + n, c0:c0 + n] = transformed
    return inp, expected


def _generate_attempt() -> np.ndarray | None:
    n = random.choice((5, 6))
    # At least twelve slots let every instance display all eight transforms,
    # one source, and multiple blank counterexample tiles.
    tile_rows, tile_cols = random.choice(((3, 4), (4, 3), (4, 4)))
    total_tiles = tile_rows * tile_cols

    separator, bg, anchor_color, color_a, color_b = random.sample(_COLORS, 5)
    template, _anchor = _make_template(n, bg, anchor_color, (color_a, color_b))
    variants = _d4(template)
    if len({variant.tobytes() for variant in variants}) != 8:
        return None

    source_index = random.randrange(total_tiles)
    available = [i for i in range(total_tiles) if i != source_index]
    # Leave at least two truly empty tiles. Every symmetry appears, while a
    # varying number of repeated target transforms varies the object count.
    marker_count = random.randint(8, min(10, len(available) - 2))
    target_indices = random.sample(available, marker_count)
    transform_indices = list(range(8))
    transform_indices.extend(random.randrange(8) for _ in range(marker_count - 8))
    random.shuffle(transform_indices)

    specs = list(zip(target_indices, transform_indices))
    inp, expected = _render(
        tile_rows, tile_cols, n, separator, bg, template, anchor_color,
        source_index, specs,
    )
    try:
        actual = main(inp)
    except (AssertionError, ValueError, IndexError):
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, inp):
        return None

    # Semantic anti-degeneracy checks: markers select distinct results, an
    # identity-only stamp is wrong, and every chosen transform moves its anchor
    # to exactly the visible marker coordinate.
    chosen_tiles = [variants[k] for k in transform_indices]
    if len({tile.tobytes() for tile in chosen_tiles}) != 8:
        return None
    if all(np.array_equal(tile, template) for tile in chosen_tiles):
        return None
    for k in transform_indices:
        if len(np.argwhere(variants[k] == anchor_color)) != 1:
            return None
    return inp


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate anchor-image stamping instance."""
    for _ in range(_MAX_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid anchor-image stamping grid")
