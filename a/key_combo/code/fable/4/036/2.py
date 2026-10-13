from __future__ import annotations

from collections import Counter

import numpy as np


ARC_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400
LAYOUTS = ((2, 2), (2, 3), (3, 2), (3, 3))
TILE_SIDE_RANGE = (4, 8)  # upper bound is exclusive
PALETTE_SIZE_RANGE = (3, 5)  # upper bound is exclusive

# Avoid returning two distinct generated inputs with the same answer during one
# loaded-module session.  The space of possible answers is large; this set also
# makes the non-constant-output requirement explicit and mechanically enforced.
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _interior_segments(length: int, separator_indices: np.ndarray) -> list[slice] | None:
    """Return non-separator spans, treating adjacent indices as one thick line."""
    indices = [int(x) for x in separator_indices]
    if not indices or indices[0] == 0 or indices[-1] == length - 1:
        return None

    runs: list[tuple[int, int]] = []
    start = end = indices[0]
    for index in indices[1:]:
        if index == end + 1:
            end = index
        else:
            runs.append((start, end))
            start = end = index
    runs.append((start, end))

    spans: list[slice] = []
    cursor = 0
    for start, end in runs:
        if cursor >= start:
            return None
        spans.append(slice(cursor, start))
        cursor = end + 1
    if cursor >= length:
        return None
    spans.append(slice(cursor, length))

    sizes = [span.stop - span.start for span in spans]
    if len(spans) < 2 or len(set(sizes)) != 1:
        return None
    return spans


def _split_tiles(grid: np.ndarray) -> tuple[int, list[list[np.ndarray]]]:
    """Find the unique same-color full-row/full-column separator lattice."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a two-dimensional grid"

    uniform_rows = np.all(g == g[:, :1], axis=1)
    uniform_cols = np.all(g == g[:1, :], axis=0)
    row_colors = {int(g[r, 0]) for r in np.flatnonzero(uniform_rows)}
    col_colors = {int(g[0, c]) for c in np.flatnonzero(uniform_cols)}

    candidates: list[tuple[int, list[slice], list[slice]]] = []
    for color in row_colors & col_colors:
        separator_rows = np.flatnonzero(uniform_rows & (g[:, 0] == color))
        separator_cols = np.flatnonzero(uniform_cols & (g[0, :] == color))
        row_spans = _interior_segments(g.shape[0], separator_rows)
        col_spans = _interior_segments(g.shape[1], separator_cols)
        if row_spans is not None and col_spans is not None:
            candidates.append((color, row_spans, col_spans))

    assert len(candidates) == 1, "expected one unambiguous separator lattice"
    separator, row_spans, col_spans = candidates[0]
    tiles = [[g[rs, cs].copy() for cs in col_spans] for rs in row_spans]
    shapes = {tile.shape for row in tiles for tile in row}
    assert len(shapes) == 1, "all tiles must have the same shape"
    return separator, tiles


def _consensus_and_errors(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int, tuple[int, int]]:
    """Recover the majority tile and each tile's sole overwritten cell."""
    separator, tile_rows = _split_tiles(grid)
    layout = (len(tile_rows), len(tile_rows[0]))
    flat_tiles = [tile for row in tile_rows for tile in row]
    assert len(flat_tiles) >= 3, "majority voting needs at least three tiles"

    stack = np.stack(flat_tiles, axis=0)
    tile_h, tile_w = stack.shape[1:]
    consensus = np.empty((tile_h, tile_w), dtype=int)

    for r in range(tile_h):
        for c in range(tile_w):
            counts = Counter(int(value) for value in stack[:, r, c])
            best_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == best_count]
            assert len(winners) == 1 and best_count > len(flat_tiles) // 2, (
                "every tile coordinate must have a unique strict majority"
            )
            consensus[r, c] = winners[0]

    palette = set(int(x) for x in np.unique(consensus))
    errors: list[tuple[int, int, int, int]] = []
    positions: set[tuple[int, int]] = set()
    for tile in flat_tiles:
        differences = np.argwhere(tile != consensus)
        assert len(differences) == 1, "each tile must contain exactly one rewrite"
        r, c = map(int, differences[0])
        visible_color = int(tile[r, c])
        erased_color = int(consensus[r, c])
        assert visible_color in palette and visible_color != erased_color, (
            "a rewrite must use another color from the consensus palette"
        )
        assert (r, c) not in positions, "rewrite coordinates must be pairwise distinct"
        positions.add((r, c))
        errors.append((r, c, erased_color, visible_color))

    return consensus, errors, separator, layout


def main(grid: np.ndarray) -> np.ndarray:
    """Reveal all overwritten cells by collecting their erased consensus colors on black.

    Input structure:
    - Equal-sized copies of one multicolor tile form a rectangular lattice, with
      complete monochromatic rows and columns separating neighboring tiles.
    - Every copy has exactly one recolored cell; its replacement color is another
      color from the common tile palette, and the damaged coordinates are distinct.

    Rule:
    - At every within-tile coordinate, take the strict majority across all copies
      to reconstruct the consensus tile.  Compare each copy with that consensus,
      then place the consensus color erased at its unique differing coordinate on
      an otherwise black canvas having the dimensions of one tile.

    Key insight:
    - The visible error color is a decoy: the error positions come from differences
      among copies, while the output colors are the majority colors they replaced.
    """
    consensus, errors, _separator, _layout = _consensus_and_errors(grid)
    output = np.zeros(consensus.shape, dtype=int)
    for r, c, erased_color, _visible_color in errors:
        output[r, c] = erased_color
    return output


def _make_consensus_tile(
    rng: np.random.Generator, height: int, width: int, palette: np.ndarray
) -> np.ndarray | None:
    """Make a clear, region-like tile using nearest randomly placed color seeds."""
    count = len(palette)
    all_coords = np.array([(r, c) for r in range(height) for c in range(width)], dtype=int)
    seed_indices = rng.choice(len(all_coords), size=count, replace=False)
    seeds = all_coords[seed_indices]

    rr, cc = np.indices((height, width))
    distances = np.stack(
        [np.abs(rr - int(sr)) + np.abs(cc - int(sc)) for sr, sc in seeds], axis=0
    )
    # A random priority resolves geometric ties without giving color ids meaning.
    priority = rng.permutation(count)
    tie_bias = np.empty(count, dtype=float)
    tie_bias[priority] = np.arange(count, dtype=float) / (count + 1)
    owner = np.argmin(distances + tie_bias[:, None, None], axis=0)
    tile = palette[owner].astype(int)

    color_counts = Counter(int(x) for x in tile.flat)
    if any(color_counts[int(color)] < 2 for color in palette):
        return None
    return tile


def _choose_rewrites(
    rng: np.random.Generator, consensus: np.ndarray, palette: np.ndarray, count: int
) -> tuple[list[tuple[int, int]], list[int]] | None:
    """Choose distinct positions with varied erased and replacement colors."""
    by_color = {
        int(color): [tuple(map(int, rc)) for rc in np.argwhere(consensus == int(color))]
        for color in palette
    }
    usable_colors = [color for color, coords in by_color.items() if coords]
    if len(usable_colors) < 2 or consensus.size < count:
        return None

    chosen_positions: list[tuple[int, int]] = []
    # Force at least two erased colors, ruling out a single-color output shortcut.
    first_colors = list(rng.choice(usable_colors, size=2, replace=False))
    for color in first_colors:
        coords = by_color[int(color)]
        chosen_positions.append(coords[int(rng.integers(0, len(coords)))])

    remaining = [
        tuple(map(int, rc))
        for rc in np.argwhere(np.ones(consensus.shape, dtype=bool))
        if tuple(map(int, rc)) not in set(chosen_positions)
    ]
    if len(remaining) < count - 2:
        return None
    if count > 2:
        picked = rng.choice(len(remaining), size=count - 2, replace=False)
        chosen_positions.extend(remaining[int(i)] for i in picked)
    rng.shuffle(chosen_positions)

    replacement_colors: list[int] = []
    for r, c in chosen_positions:
        original = int(consensus[r, c])
        choices = [int(color) for color in palette if int(color) != original]
        replacement_colors.append(int(rng.choice(choices)))

    if len(set(replacement_colors)) < 2:
        # Deterministically change one replacement to a second legal palette color.
        for i, (r, c) in enumerate(chosen_positions):
            original = int(consensus[r, c])
            alternatives = [
                int(color)
                for color in palette
                if int(color) != original and int(color) != replacement_colors[0]
            ]
            if alternatives:
                replacement_colors[i] = int(rng.choice(alternatives))
                break
    if len(set(replacement_colors)) < 2:
        return None
    return chosen_positions, replacement_colors


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    consensus: np.ndarray,
    positions: list[tuple[int, int]],
    replacements: list[int],
    layout: tuple[int, int],
    separator: int,
) -> bool:
    """Check puzzle semantics independently of merely calling main(generate_input())."""
    if inp.ndim != 2 or max(inp.shape) > 30 or expected.shape != consensus.shape:
        return False
    if len(set(positions)) != len(positions) or len(positions) != layout[0] * layout[1]:
        return False
    if separator in set(int(x) for x in np.unique(consensus)):
        return False

    # Known construction must have one legal, palette-sourced change per tile.
    palette = set(int(x) for x in np.unique(consensus))
    if any(int(new) not in palette or int(new) == int(consensus[pos])
           for pos, new in zip(positions, replacements)):
        return False
    if len({int(consensus[pos]) for pos in positions}) < 2:
        return False

    try:
        parsed_consensus, errors, parsed_separator, parsed_layout = _consensus_and_errors(inp)
        solved = main(inp)
    except (AssertionError, ValueError):
        return False
    if parsed_separator != separator or parsed_layout != layout:
        return False
    if not np.array_equal(parsed_consensus, consensus) or not np.array_equal(solved, expected):
        return False

    recovered_positions = [(r, c) for r, c, _old, _new in errors]
    if set(recovered_positions) != set(positions):
        return False
    if np.count_nonzero(expected) != len(positions):
        return False

    # Explicit counterexample to the tempting shortcut of copying visible errors.
    visible_copy = np.zeros(consensus.shape, dtype=int)
    for position, replacement in zip(positions, replacements):
        visible_copy[position] = replacement
    if np.array_equal(visible_copy, expected):
        return False
    return True


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    tile_rows, tile_cols = LAYOUTS[int(rng.integers(0, len(LAYOUTS)))]
    tile_count = tile_rows * tile_cols
    tile_h = int(rng.integers(*TILE_SIDE_RANGE))
    tile_w = int(rng.integers(*TILE_SIDE_RANGE))

    separator = int(rng.choice(ARC_COLORS))
    available = [color for color in ARC_COLORS if color not in (0, separator)]
    palette_size = int(rng.integers(*PALETTE_SIZE_RANGE))
    if len(available) < palette_size:
        return None
    palette = rng.choice(available, size=palette_size, replace=False).astype(int)

    consensus = _make_consensus_tile(rng, tile_h, tile_w, palette)
    if consensus is None:
        return None
    rewrites = _choose_rewrites(rng, consensus, palette, tile_count)
    if rewrites is None:
        return None
    positions, replacements = rewrites

    tiles: list[np.ndarray] = []
    for position, replacement in zip(positions, replacements):
        tile = consensus.copy()
        tile[position] = replacement
        tiles.append(tile)

    input_h = tile_rows * tile_h + tile_rows - 1
    input_w = tile_cols * tile_w + tile_cols - 1
    inp = np.full((input_h, input_w), separator, dtype=int)
    for index, tile in enumerate(tiles):
        block_r, block_c = divmod(index, tile_cols)
        r0 = block_r * (tile_h + 1)
        c0 = block_c * (tile_w + 1)
        inp[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile

    expected = np.zeros(consensus.shape, dtype=int)
    for position in positions:
        expected[position] = int(consensus[position])

    if not _validate_case(
        inp, expected, consensus, positions, replacements,
        (tile_rows, tile_cols), separator,
    ):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    """Generate a diverse, semantically checked separator-grid corruption case."""
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is None:
            continue
        inp, expected = case
        signature = (expected.shape, expected.tobytes())
        if signature in _SEEN_OUTPUTS:
            continue
        _SEEN_OUTPUTS.add(signature)
        return inp
    raise RuntimeError("failed to generate a fresh valid error-revelation input")
