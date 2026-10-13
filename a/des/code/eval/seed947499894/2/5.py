from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
PIECE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


@dataclass
class Sprite:
    color: int
    mask: np.ndarray
    lefts: np.ndarray
    rights: np.ndarray


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows), min(cols), max(cols)


def _row_intervals(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = mask.shape[0]
    lefts = np.zeros(h, dtype=int)
    rights = np.zeros(h, dtype=int)
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        assert cols.size > 0
        assert int(cols[-1] - cols[0] + 1) == int(cols.size)
        lefts[r] = int(cols[0])
        rights[r] = int(cols[-1] + 1)
    return lefts, rights


def _extract_palette(grid: np.ndarray) -> tuple[list[int], tuple[int, int, int, int]]:
    grey_components = _components(grid == GREY)
    for comp in grey_components:
        r0, r1, c0, c1 = _bbox(comp)
        block = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = block.shape
        if h != 3 or w < 5 or w % 2 == 0:
            continue
        if not np.all(block[0] == GREY) or not np.all(block[2] == GREY):
            continue
        if not np.all(block[1, ::2] == GREY):
            continue
        colors = [int(x) for x in block[1, 1::2]]
        if any(color in (BG, GREY) for color in colors):
            continue
        if len(set(colors)) != len(colors):
            continue
        return colors, (r0, r1, c0, c1)
    raise AssertionError("palette not found")


def _extract_sprites(grid: np.ndarray, palette_bbox: tuple[int, int, int, int], bg: int) -> dict[int, Sprite]:
    masked = np.array(grid, copy=True)
    r0, r1, c0, c1 = palette_bbox
    masked[r0 : r1 + 1, c0 : c1 + 1] = bg

    pieces: dict[int, Sprite] = {}
    for comp in _components(masked != bg):
        colors = {int(masked[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = colors.pop()
        assert color != GREY
        br0, br1, bc0, bc1 = _bbox(comp)
        shape = masked[br0 : br1 + 1, bc0 : bc1 + 1] == color
        lefts, rights = _row_intervals(shape)
        pieces[color] = Sprite(color=color, mask=shape, lefts=lefts, rights=rights)
    return pieces


def _assemble_labels(colors: list[int], sprites: dict[int, Sprite]) -> np.ndarray:
    ordered = [sprites[color] for color in colors]
    heights = {sprite.mask.shape[0] for sprite in ordered}
    assert len(heights) == 1
    h = heights.pop()

    x_positions = [0]
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        deltas = x_positions[-1] + prev.rights - cur.lefts
        assert np.all(deltas == deltas[0])
        x_positions.append(int(deltas[0]))

    width = max(x + sprite.mask.shape[1] for x, sprite in zip(x_positions, ordered))
    labels = np.full((h, width), -1, dtype=int)
    for idx, (x0, sprite) in enumerate(zip(x_positions, ordered)):
        rr, cc = np.nonzero(sprite.mask)
        for r, c in zip(rr, cc):
            ac = x0 + int(c)
            assert labels[int(r), ac] == -1
            labels[int(r), ac] = idx

    used = np.flatnonzero(np.any(labels != -1, axis=0))
    assert used.size > 0
    labels = labels[:, int(used[0]) : int(used[-1]) + 1]
    assert np.all(labels >= 0)
    return labels


def _render_output(labels: np.ndarray, colors: list[int]) -> np.ndarray:
    out = np.full(labels.shape, GREY, dtype=int)
    h, w = labels.shape
    for r in range(h):
        for c in range(w):
            idx = int(labels[r, c])
            if idx < 0 or idx >= len(colors) - 1:
                continue
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) == idx + 1:
                    out[r, c] = colors[idx]
                    break
    return out


def _labels_from_boundaries(boundaries: list[np.ndarray]) -> np.ndarray:
    h = len(boundaries[0])
    width = int(boundaries[-1][0])
    labels = np.full((h, width), -1, dtype=int)
    for idx in range(len(boundaries) - 1):
        left = boundaries[idx]
        right = boundaries[idx + 1]
        for r in range(h):
            labels[r, int(left[r]) : int(right[r])] = idx
    assert np.all(labels >= 0)
    return labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered monochrome puzzle fragments into a rectangle, then
    redraw only the internal seams.

    Input structure:
    - Background is the most common color.
    - Several single-color fragments are scattered around the grid.
    - A small grey palette rectangle lists the fragment colors from left to right.

    Rule:
    - Read the palette order.
    - Reconstruct the full rectangle by matching the shared jagged boundaries of
      consecutive fragments in that order.
    - Paint the assembled rectangle grey, then recolor each seam pixel with the
      original color of the fragment on its left.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    colors, palette_bbox = _extract_palette(g)
    sprites = _extract_sprites(g, palette_bbox, bg)
    assert set(colors) == set(sprites)
    labels = _assemble_labels(colors, sprites)
    return _render_output(labels, colors)


def _make_palette(colors: list[int]) -> np.ndarray:
    palette = np.full((3, 2 * len(colors) + 1), GREY, dtype=int)
    palette[1, 1::2] = np.array(colors, dtype=int)
    return palette


def _random_walk(height: int, rng: random.Random) -> np.ndarray:
    values = [rng.choice([-1, 0, 1])]
    for _ in range(1, height):
        step = rng.choice([-1, 0, 1])
        values.append(max(-1, min(1, values[-1] + step)))
    return np.array(values, dtype=int)


def _valid_piece_widths(boundaries: list[np.ndarray]) -> bool:
    for idx in range(len(boundaries) - 1):
        widths = boundaries[idx + 1] - boundaries[idx]
        if np.min(widths) < 2:
            return False
    return True


def _piece_masks_from_labels(labels: np.ndarray, colors: list[int]) -> list[np.ndarray]:
    pieces: list[np.ndarray] = []
    for idx, color in enumerate(colors):
        mask = labels == idx
        rows = np.flatnonzero(np.any(mask, axis=1))
        cols = np.flatnonzero(np.any(mask, axis=0))
        sprite = np.full((rows.size, cols.size), BG, dtype=int)
        sprite[mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]] = color
        pieces.append(sprite)
    return pieces


def _can_place(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + height + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + width + 1)
    return not np.any(reserved[r0:r1, c0:c1])


def _paint_object(grid: np.ndarray, obj: np.ndarray, top: int, left: int) -> None:
    h, w = obj.shape
    target = grid[top : top + h, left : left + w]
    mask = obj != BG
    target[mask] = obj[mask]


def _place_objects(
    grid_shape: tuple[int, int], objects: list[tuple[str, np.ndarray]], rng: random.Random
) -> np.ndarray | None:
    h, w = grid_shape
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    order = list(objects)
    rng.shuffle(order)
    for _name, obj in order:
        oh, ow = obj.shape
        placed = False
        for _ in range(300):
            top = rng.randint(0, h - oh)
            left = rng.randint(0, w - ow)
            if not _can_place(reserved, top, left, oh, ow):
                continue
            reserved[top : top + oh, left : left + ow] = True
            _paint_object(grid, obj, top, left)
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_boundaries(piece_count: int, height: int, rng: random.Random) -> list[np.ndarray] | None:
    base_widths = [rng.randint(3, 5) for _ in range(piece_count)]
    internal: list[np.ndarray] = []
    for _ in range(piece_count - 1):
        walk = _random_walk(height, rng)
        if len(set(int(x) for x in walk)) < 2:
            return None
        internal.append(walk)

    cumulative = 0
    boundaries = [np.zeros(height, dtype=int)]
    for idx, width in enumerate(base_widths[:-1]):
        cumulative += width
        boundaries.append(np.full(height, cumulative, dtype=int) + internal[idx])
    boundaries.append(np.full(height, sum(base_widths), dtype=int))

    if not _valid_piece_widths(boundaries):
        return None
    return boundaries


def _expected_output_from_labels(labels: np.ndarray, colors: list[int]) -> np.ndarray:
    return _render_output(labels, colors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    piece_count = rng.randint(2, 4)
    height = rng.randint(4, 8)
    boundaries = _generate_boundaries(piece_count, height, rng)
    if boundaries is None:
        return None

    labels = _labels_from_boundaries(boundaries)
    width = labels.shape[1]
    if width < 7:
        return None

    colors = rng.sample(PIECE_COLORS, piece_count)
    palette = _make_palette(colors)
    pieces = _piece_masks_from_labels(labels, colors)
    expected = _expected_output_from_labels(labels, colors)

    grid_h = height + rng.randint(7, 12)
    grid_w = max(width, palette.shape[1]) + rng.randint(10, 16)
    objects = [("palette", palette)] + [(f"piece_{i}", piece) for i, piece in enumerate(pieces)]
    grid = _place_objects((grid_h, grid_w), objects, rng)
    if grid is None:
        return None

    if Counter(grid.ravel()).most_common(1)[0][0] != BG:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle")
