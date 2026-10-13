from __future__ import annotations

from collections import Counter
import random

import numpy as np


BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(box: BBox) -> int:
    return int(box[2] - box[0] + 1)


def bbox_w(box: BBox) -> int:
    return int(box[3] - box[1] + 1)


def crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def find_components(grid: np.ndarray, *, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    fg = g != int(bg)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r, c in np.argwhere(fg):
        rr0 = int(r)
        cc0 = int(c)
        if seen[rr0, cc0]:
            continue

        stack = [(rr0, cc0)]
        seen[rr0, cc0] = True
        coords: list[tuple[int, int]] = []

        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))

        coords_arr = np.asarray(coords, dtype=int)
        components.append(
            {
                "coords": coords_arr,
                "bbox": bbox_coords(coords_arr),
                "area": int(coords_arr.shape[0]),
            }
        )

    return components


def _marker_color(grid: np.ndarray, bg: int) -> int:
    fg_values = np.asarray(grid, dtype=int)[np.asarray(grid, dtype=int) != int(bg)]
    counts = Counter(map(int, fg_values.tolist()))
    assert counts, "expected at least one foreground color"
    least = min(counts.values())
    markers = [color for color, count in counts.items() if count == least]
    assert len(markers) == 1, "expected a unique least frequent foreground color"
    return int(markers[0])


def _select_template(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    marker = _marker_color(g, bg)
    components = find_components(g, bg=bg)
    assert 3 <= len(components) <= 4, "expected 3 to 4 foreground sprites"

    scored: list[tuple[int, np.ndarray]] = []
    for component in components:
        box = component["bbox"]
        assert (bbox_h(box), bbox_w(box)) == (3, 3), "each sprite must have a 3x3 bbox"
        assert int(component["area"]) == 9, "each sprite must fill its 3x3 bbox"

        sprite = crop_bbox(g, box)
        assert np.all(sprite != bg), "sprite bbox must be fully foreground"
        assert len(colors_present(sprite, exclude={bg})) == 2, "each sprite must use exactly two colors"

        marker_count = int(np.count_nonzero(sprite == marker))
        assert marker_count >= 1, "every sprite must contain the marker color"
        scored.append((marker_count, sprite))

    best = max(score for score, _sprite in scored)
    winners = [sprite for score, sprite in scored if score == best]
    assert len(winners) == 1, "expected a unique marker-densest sprite"
    return int(bg), int(marker), winners[0]


def _recursive_stamp(template: np.ndarray, marker: int, bg: int) -> np.ndarray:
    out = np.full((9, 9), int(bg), dtype=int)
    for r, c in np.argwhere(np.asarray(template, dtype=int) == int(marker)):
        rr = int(r) * 3
        cc = int(c) * 3
        out[rr : rr + 3, cc : cc + 3] = template
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the marker-densest 3x3 sprite, then recursively stamp it into the marker blocks of a 9x9 canvas.

    Input structure:
    - A black background contains 3 to 4 separated 4-connected solid 3x3 sprites.
    - Every sprite is fully filled and uses exactly two non-background colors.
    - One foreground color is globally least frequent; it serves as the marker color across all sprites.

    Rule:
    - Infer the marker as the unique least frequent non-background color.
    - Crop each 4-connected foreground component to its 3x3 sprite and count marker pixels inside it.
    - Select the unique sprite with the largest marker count as the template.
    - Build a 9x9 output split into a 3x3 lattice of 3x3 blocks, and copy the whole template into block (r, c) exactly when template[r, c] is the marker color; all other blocks stay black.

    Key insight:
    - The same rare color is used twice: globally to choose which sprite becomes the template, and locally inside that template to choose the macro-level stamp positions.
    """
    bg, marker, template = _select_template(grid)
    return _recursive_stamp(template, marker, bg)


BACKGROUND_COLOR = 0
SPRITE_SIZE = 3
GRID_HEIGHT_RANGE = (10, 14)
GRID_WIDTH_RANGE = (10, 14)
SPRITE_COUNT_CHOICES = (3, 4)
COUNT_SETS_BY_SPRITE_COUNT = {
    3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4)],
    4: [(1, 2, 3, 4)],
}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _make_sprite(base_color: int, marker_color: int, marker_count: int) -> np.ndarray:
    sprite = np.full((SPRITE_SIZE, SPRITE_SIZE), int(base_color), dtype=int)
    for idx in random.sample(range(SPRITE_SIZE * SPRITE_SIZE), int(marker_count)):
        sprite[idx // SPRITE_SIZE, idx % SPRITE_SIZE] = int(marker_color)
    return sprite


def _choose_slots(height: int, width: int) -> list[tuple[int, int]]:
    r_top = random.randint(0, height - 7)
    r_bottom = random.randint(r_top + 4, height - SPRITE_SIZE)
    c_left = random.randint(0, width - 7)
    c_right = random.randint(c_left + 4, width - SPRITE_SIZE)
    slots = [
        (r_top, c_left),
        (r_top, c_right),
        (r_bottom, c_left),
        (r_bottom, c_right),
    ]
    random.shuffle(slots)
    return slots


def _expected_output(template: np.ndarray, marker_color: int) -> np.ndarray:
    return _recursive_stamp(template, marker_color, BACKGROUND_COLOR)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    sprite_count = random.choice(SPRITE_COUNT_CHOICES)
    marker_color, base_color = random.sample(PALETTE, 2)
    marker_counts = list(random.choice(COUNT_SETS_BY_SPRITE_COUNT[sprite_count]))
    random.shuffle(marker_counts)

    grid = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    slots = _choose_slots(height, width)

    sprites: list[tuple[int, np.ndarray]] = []
    for marker_count in marker_counts:
        sprites.append((int(marker_count), _make_sprite(base_color, marker_color, int(marker_count))))
    random.shuffle(sprites)

    for (_count, sprite), (r0, c0) in zip(sprites, slots[:sprite_count]):
        _paste(grid, sprite, int(r0), int(c0))

    template = max(sprites, key=lambda item: item[0])[1]
    expected = _expected_output(template, marker_color)

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
