"""ARC task: assemble color-matched notched rings with every notch facing inward."""

from __future__ import annotations

import random
from collections import Counter

import numpy as np


_MARKER_CENTERS = ((1, 1), (1, 5), (5, 1), (5, 5))
_OUT_ANCHORS = ((0, 0), (0, 4), (4, 0), (4, 4))
# The corner of each 3x3 output quadrant closest to the 7x7 canvas center.
_INWARD_CORNERS = ((2, 2), (2, 0), (0, 2), (0, 0))
_CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_selector(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """Find the unique 7x7 block whose only optional exceptions are marker centers."""
    h, w = grid.shape
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for r0 in range(h - 6):
        for c0 in range(w - 6):
            block = grid[r0 : r0 + 7, c0 : c0 + 7]
            body_positions = np.ones((7, 7), dtype=bool)
            for pos in _MARKER_CENTERS:
                body_positions[pos] = False
            body_values = block[body_positions]
            color = int(body_values[0])
            if color == bg or not np.all(body_values == color):
                continue
            if any(int(block[pos]) == bg for pos in _MARKER_CENTERS):
                continue
            candidates.append((color, (r0, c0, r0 + 6, c0 + 6)))
    assert len(candidates) == 1, "expected one unambiguous 7x7 selector"
    return candidates[0]


def _notch_corner(mask: np.ndarray) -> tuple[int, int]:
    assert mask.shape == (3, 3)
    assert not bool(mask[1, 1]), "ring center must be empty"
    assert int(mask.sum()) == 7, "a notched ring must contain seven pixels"
    missing = {(r, c) for r in range(3) for c in range(3) if (r != 1 or c != 1) and not mask[r, c]}
    assert len(missing) == 1
    corner = next(iter(missing))
    assert corner in _CORNERS, "the sole perimeter gap must be a corner"
    return corner


def _external_sprite(
    grid: np.ndarray, color: int, selector_box: tuple[int, int, int, int]
) -> np.ndarray:
    r0, c0, r1, c1 = selector_box
    coords = np.argwhere(grid == color)
    outside = coords[
        ~(
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
    ]
    assert outside.shape == (7, 2), "each marker color needs one seven-pixel external ring"
    sr0, sc0 = outside.min(axis=0)
    sr1, sc1 = outside.max(axis=0)
    assert (int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)) == (3, 3)
    sprite = np.full((3, 3), _background(grid), dtype=int)
    sprite[outside[:, 0] - sr0, outside[:, 1] - sc0] = color
    _notch_corner(sprite == color)
    return sprite


def _rotate_notch_to(sprite: np.ndarray, color: int, target: tuple[int, int]) -> np.ndarray:
    rotated = sprite.copy()
    for _ in range(4):
        if _notch_corner(rotated == color) == target:
            return rotated
        rotated = np.rot90(rotated)
    raise AssertionError("corner notch cannot be rotated to target")


def main(grid: np.ndarray) -> np.ndarray:
    """Place color-matched notched rings in selector quadrants with every notch facing the center.

    Input structure:
    A dominant background surrounds one 7x7 selector block.  The block has one
    body color and optional differently colored markers at the centers of its
    four 3x3 quadrants.  Each marker color also forms exactly one external 3x3
    seven-pixel square ring whose sole missing perimeter pixel is a corner.

    Rule:
    Make a background-filled 7x7 output split by its empty middle row and
    column.  A marker position chooses an output quadrant and its color chooses
    the matching external ring.  Rotate that ring by a multiple of 90 degrees
    until its missing corner is the quadrant corner nearest the output center,
    then copy it into the quadrant.  Leave unmarked quadrants empty.

    Key insight:
    The selector simultaneously encodes position and color matching, while the
    common center—not each sprite's input orientation—determines every rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)
    selector_color, box = _find_selector(g, bg)
    r0, c0, _r1, _c1 = box

    out = np.full((7, 7), bg, dtype=int)
    seen_colors: set[int] = set()
    marker_count = 0
    for index, (mr, mc) in enumerate(_MARKER_CENTERS):
        color = int(g[r0 + mr, c0 + mc])
        if color == selector_color:
            continue
        assert color != bg and color not in seen_colors, "marker colors must be distinct"
        seen_colors.add(color)
        marker_count += 1
        sprite = _external_sprite(g, color, box)
        sprite = _rotate_notch_to(sprite, color, _INWARD_CORNERS[index])
        ar, ac = _OUT_ANCHORS[index]
        mask = sprite != bg
        out[ar : ar + 3, ac : ac + 3][mask] = sprite[mask]

    assert 1 <= marker_count <= 4
    return out


def _ring(color: int, bg: int, missing: tuple[int, int]) -> np.ndarray:
    sprite = np.full((3, 3), color, dtype=int)
    sprite[1, 1] = bg
    sprite[missing] = bg
    return sprite


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True when two inclusive rectangles have at least one blank row/column between them."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + 1 < br0 or br1 + 1 < ar0 or ac1 + 1 < bc0 or bc1 + 1 < ac0


def _place_regions(h: int, w: int, sizes: list[int]) -> list[tuple[int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    anchors: list[tuple[int, int]] = []
    for side in sizes:
        choices = [(r, c) for r in range(1, h - side) for c in range(1, w - side)]
        random.shuffle(choices)
        for r, c in choices:
            box = (r, c, r + side - 1, c + side - 1)
            if all(_separated(box, old) for old in boxes):
                boxes.append(box)
                anchors.append((r, c))
                break
        else:
            return None
    return anchors


def _source_directions(indices: list[int]) -> list[tuple[int, int]] | None:
    """Choose distinct input notches, none already equal to its inward target."""
    directions = list(_CORNERS)
    random.shuffle(directions)

    def search(k: int, used: set[tuple[int, int]], result: list[tuple[int, int]]):
        if k == len(indices):
            return result.copy()
        options = [d for d in directions if d not in used and d != _INWARD_CORNERS[indices[k]]]
        random.shuffle(options)
        for direction in options:
            used.add(direction)
            result.append(direction)
            found = search(k + 1, used, result)
            if found is not None:
                return found
            result.pop()
            used.remove(direction)
        return None

    return search(0, set(), [])


def _oracle(bg: int, indices: list[int], colors: list[int]) -> np.ndarray:
    expected = np.full((7, 7), bg, dtype=int)
    for index, color in zip(indices, colors):
        ar, ac = _OUT_ANCHORS[index]
        expected[ar : ar + 3, ac : ac + 3] = _ring(color, bg, _INWARD_CORNERS[index])
    return expected


_COUNT_CYCLE = random.sample([2, 3, 4], 3)
_generation_count = 0


def generate_input() -> np.ndarray:
    """Generate varied, nondegenerate selector/ring assemblies."""
    global _generation_count
    marker_count = _COUNT_CYCLE[_generation_count % len(_COUNT_CYCLE)]

    for _ in range(500):
        h = random.randint(16, 26)
        w = random.randint(16, 26)
        palette = random.sample(range(10), marker_count + 2)
        bg, selector_color = palette[:2]
        marker_colors = palette[2:]
        indices = sorted(random.sample(range(4), marker_count))
        source_notches = _source_directions(indices)
        assert source_notches is not None

        anchors = _place_regions(h, w, [7] + [3] * marker_count)
        if anchors is None:
            continue
        grid = np.full((h, w), bg, dtype=int)
        sr, sc = anchors[0]
        grid[sr : sr + 7, sc : sc + 7] = selector_color

        naive = np.full((7, 7), bg, dtype=int)
        for index, color, missing, (r, c) in zip(
            indices, marker_colors, source_notches, anchors[1:]
        ):
            mr, mc = _MARKER_CENTERS[index]
            grid[sr + mr, sc + mc] = color
            sprite = _ring(color, bg, missing)
            grid[r : r + 3, c : c + 3] = sprite
            ar, ac = _OUT_ANCHORS[index]
            naive[ar : ar + 3, ac : ac + 3] = sprite

        expected = _oracle(bg, indices, marker_colors)
        # Semantic guards: source orientation must matter, directions must be
        # uniquely readable, and the independent oracle must match the solver.
        assert len(set(source_notches)) == marker_count
        assert all(src != _INWARD_CORNERS[i] for i, src in zip(indices, source_notches))
        assert not np.array_equal(naive, expected)
        try:
            actual = main(grid)
        except (AssertionError, ValueError, IndexError):
            continue
        if np.array_equal(actual, expected):
            _generation_count += 1
            return grid

    raise RuntimeError("failed to generate a valid notched-ring assembly")
