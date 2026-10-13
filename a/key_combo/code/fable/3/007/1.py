from __future__ import annotations

import numpy as np


BLACK = 0
NON_BLACK_COLORS = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 500


def _rect_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_hole_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    black = g == BLACK
    r0, c0, r1, c1 = _rect_bbox(black)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(black.sum()) == area, "black cells must form exactly one rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK)
    return r0, c0, r1, c1


def _mirror_bbox(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0, width - 1 - c1, r1, width - 1 - c0


def _swap_colors(values: np.ndarray, a: int, b: int) -> np.ndarray:
    out = values.copy()
    a_mask = values == int(a)
    b_mask = values == int(b)
    out[a_mask] = int(b)
    out[b_mask] = int(a)
    return out


def _infer_swap_pair(g: np.ndarray, hole_box: tuple[int, int, int, int]) -> tuple[int, int]:
    h, w = g.shape
    _hole_r0, hole_c0, _hole_r1, hole_c1 = hole_box
    mirror_box = _mirror_bbox(hole_box, w)
    mr0, mc0, mr1, mc1 = mirror_box
    assert 0 <= mc0 <= mc1 < w
    assert not (hole_c0 <= mc1 and mc0 <= hole_c1), "hole must be off the mirror axis"
    assert np.all(g[mr0 : mr1 + 1, mc0 : mc1 + 1] != BLACK), "mirror source must be intact"

    differing_pairs: set[frozenset[int]] = set()
    directed_differences: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w // 2):
            mc = w - 1 - c
            left = int(g[r, c])
            right = int(g[r, mc])
            if left == BLACK or right == BLACK:
                continue
            if left == right:
                continue
            pair = frozenset((left, right))
            assert len(pair) == 2
            differing_pairs.add(pair)
            directed_differences.add((left, right))

    assert len(differing_pairs) == 1, "exactly one two-color swap must be visible"
    a, b = tuple(next(iter(differing_pairs)))
    assert (a, b) in directed_differences and (b, a) in directed_differences, (
        "both swap directions must be inferable outside the hole"
    )

    if w % 2:
        center = g[:, w // 2]
        assert not np.any((center == a) | (center == b)), "axis cells cannot carry swapped colors"
    return int(a), int(b)


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a black rectangular hole by mirrored copy followed by an inferred two-color swap.

    Input structure:
    - The grid is a dense non-black multicolor pattern except for one black rectangular hole.
    - The pattern is symmetric across the vertical center only after swapping one pair of colors.
    - The hole is off the center axis, and its mirrored rectangle is intact.

    Rule:
    - Compare intact vertically mirrored cell pairs outside the hole to infer the unique
      pair of colors that exchange with each other; all other colors mirror to themselves.
    - Copy the intact mirror rectangle into the hole after a left-right flip.
    - Swap the inferred pair inside that copied patch and leave all other cells unchanged.

    Key insight:
    - The missing pixels are not a plain mirror image: the true symmetry is mirror followed
      by a two-color exchange, and the exchanged pair must be recovered from the visible area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    hole_box = _find_hole_bbox(g)
    a, b = _infer_swap_pair(g, hole_box)
    r0, c0, r1, c1 = hole_box
    _mr0, mc0, _mr1, mc1 = _mirror_bbox(hole_box, g.shape[1])

    source = g[r0 : r1 + 1, mc0 : mc1 + 1]
    assert source.shape == (r1 - r0 + 1, c1 - c0 + 1)
    patch = _swap_colors(source[:, ::-1], a, b)

    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = patch
    return out


# --------------- generator ---------------


def _swap_color_value(color: int, pair: tuple[int, int]) -> int:
    a, b = pair
    if int(color) == int(a):
        return int(b)
    if int(color) == int(b):
        return int(a)
    return int(color)


def _sample_palette(rng: np.random.Generator) -> tuple[list[int], tuple[int, int]]:
    n_colors = int(rng.integers(4, 8))
    palette = [int(c) for c in rng.choice(NON_BLACK_COLORS, size=n_colors, replace=False)]
    pair_positions = rng.choice(np.arange(n_colors), size=2, replace=False)
    pair = (palette[int(pair_positions[0])], palette[int(pair_positions[1])])
    return palette, pair


def _weighted_left_color(
    rng: np.random.Generator,
    palette: list[int],
    pair: tuple[int, int],
    row_bias: int,
    col_bias: int,
) -> int:
    roll = float(rng.random())
    if roll < 0.32:
        return int(row_bias)
    if roll < 0.56:
        return int(col_bias)
    if roll < 0.78:
        return int(pair[int(rng.integers(2))])
    return int(palette[int(rng.integers(len(palette)))])


def _make_complete_pattern(
    rng: np.random.Generator, height: int, width: int, palette: list[int], pair: tuple[int, int]
) -> np.ndarray:
    left_width = width // 2
    fixed_colors = [c for c in palette if c not in pair]
    assert fixed_colors

    row_biases = rng.choice(np.array(palette, dtype=int), size=height, replace=True)
    col_biases = rng.choice(np.array(palette, dtype=int), size=left_width, replace=True)
    g = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(left_width):
            color = _weighted_left_color(rng, palette, pair, int(row_biases[r]), int(col_biases[c]))
            g[r, c] = color
            g[r, width - 1 - c] = _swap_color_value(color, pair)

    if width % 2:
        g[:, width // 2] = rng.choice(np.array(fixed_colors, dtype=int), size=height, replace=True)
    return g


def _choose_hole(
    rng: np.random.Generator, height: int, width: int
) -> tuple[int, int, int, int] | None:
    side_width = width // 2
    if side_width < 2:
        return None
    hole_h = int(rng.integers(2, min(6, height) + 1))
    hole_w = int(rng.integers(2, min(6, side_width) + 1))
    r0 = int(rng.integers(0, height - hole_h + 1))

    place_left = bool(rng.integers(2))
    if place_left:
        c0 = int(rng.integers(0, side_width - hole_w + 1))
    else:
        right_start = width - side_width
        c0 = right_start + int(rng.integers(0, side_width - hole_w + 1))
    return r0, c0, r0 + hole_h - 1, c0 + hole_w - 1


def _source_patch(g: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, _c0, r1, _c1 = box
    _mr0, mc0, _mr1, mc1 = _mirror_bbox(box, g.shape[1])
    return g[r0 : r1 + 1, mc0 : mc1 + 1]


def _semantic_check(inp: np.ndarray, expected: np.ndarray, pair: tuple[int, int]) -> bool:
    try:
        box = _find_hole_bbox(inp)
        inferred = _infer_swap_pair(inp, box)
        actual = main(inp)
    except AssertionError:
        return False

    if set(inferred) != set(pair):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == BLACK):
        return False

    r0, c0, r1, c1 = box
    outside = np.ones(inp.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    if not np.array_equal(actual[outside], inp[outside]):
        return False

    source = _source_patch(inp, box)
    source_colors = set(map(int, source.ravel()))
    if not set(pair).issubset(source_colors):
        return False
    if not any(color not in set(pair) for color in source_colors):
        return False

    direct_patch = source[:, ::-1]
    corrected_patch = _swap_colors(direct_patch, inferred[0], inferred[1])
    if np.count_nonzero(direct_patch != corrected_patch) < 2:
        return False

    hole_expected = expected[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(hole_expected, corrected_patch)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(8, 23))
    width = int(rng.integers(10, 31))
    palette, pair = _sample_palette(rng)
    expected = _make_complete_pattern(rng, height, width, palette, pair)

    hole_box = _choose_hole(rng, height, width)
    if hole_box is None:
        return None

    r0, c0, r1, c1 = hole_box
    inp = expected.copy()
    inp[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    return inp if _semantic_check(inp, expected, pair) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-swap mirror-completion input")
