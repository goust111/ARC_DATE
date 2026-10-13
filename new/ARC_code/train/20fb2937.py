from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg, paste, random_free_location
import random


def _solid_separator_row(g: np.ndarray, bg: int) -> int:
    rows = [
        r
        for r in range(g.shape[0])
        if np.all(g[r] == int(g[r, 0])) and int(g[r, 0]) != int(bg)
    ]
    assert len(rows) == 1, "expected one solid non-background separator row"
    return int(rows[0])


def _legend_mapping(g: np.ndarray, sep: int, bg: int) -> tuple[dict[int, int], int]:
    top = g[:sep]
    comps = find_components(top, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    sprites = []
    markers = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(comp["area"])
        if area == 1:
            markers.append(comp)
        else:
            assert h == w and area == h * w, "legend sprites must be filled squares"
            sprites.append(comp)

    assert len(sprites) == len(markers) and len(sprites) > 0, "legend must pair each sprite with one marker"

    sizes = {int(s["bbox"][2] - s["bbox"][0] + 1) for s in sprites}
    assert len(sizes) == 1, "all legend sprites must have the same size"
    size = sizes.pop()
    assert size % 2 == 1, "legend sprite size must have a center pixel"

    marker_rows = {int(m["coords"][0][0]) for m in markers}
    assert len(marker_rows) == 1, "legend markers must lie on one row"
    marker_colors = [int(m["color"]) for m in markers]
    assert len(set(marker_colors)) == len(marker_colors), "legend marker colors must be unique"

    sprites.sort(key=lambda s: (int(s["bbox"][1] + s["bbox"][3]), int(s["bbox"][0])))
    markers.sort(key=lambda m: (int(m["coords"][0][1]), int(m["coords"][0][0])))

    mapping: dict[int, int] = {}
    for sprite, marker in zip(sprites, markers):
        _sr0, sc0, _sr1, sc1 = sprite["bbox"]
        mr, mc = map(int, marker["coords"][0])
        assert mc == (int(sc0) + int(sc1)) // 2, "marker must align with its sprite center column"
        assert mr > int(sprite["bbox"][2]), "marker must sit below its sprite"
        mapping[int(marker["color"])] = int(sprite["color"])

    return mapping, int(size)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each marker below the separator with the mapped top legend square centered on that marker.

    Input structure:
    - The background surrounds a top legend, one solid non-background separator row, and a lower marker field.
    - The legend contains equal-size filled square sprites, with a singleton marker below each sprite and aligned to its center column.
    - The lower field contains only background and marker colors from the legend.

    Rule:
    - Pair legend markers to legend sprites by left-to-right position.
    - Discard the legend and separator; keep the grid width and the rows below the separator.
    - In the lower field, replace every marker pixel by a filled square of the paired sprite color, centered at the marker location.

    Key insight: The top area is a positional color legend, and the actual output is the post-separator marker field expanded by that legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep = _solid_separator_row(g, bg)
    mapping, size = _legend_mapping(g, sep, bg)

    lower = g[sep + 1 :]
    out = np.full_like(lower, bg)
    radius = size // 2

    for r, c in np.argwhere(lower != bg):
        key = int(lower[r, c])
        assert key in mapping, "lower field marker must appear in the legend"
        r0 = int(r) - radius
        c0 = int(c) - radius
        r1 = r0 + size
        c1 = c0 + size
        assert 0 <= r0 and 0 <= c0 and r1 <= out.shape[0] and c1 <= out.shape[1], "expanded square must fit in output"
        assert np.all(out[r0:r1, c0:c1] == bg), "expanded squares must not overlap"
        out[r0:r1, c0:c1] = mapping[key]

    return out

# --------------- generator (o5x) ---------------

# Canvas and legend geometry.
BACKGROUND_COLOR, SEPARATOR_COLOR = 7, 6
FIELD_SHAPE = (13, 11)
FIELD_WIDTH = FIELD_SHAPE[1]
SQUARE_SIZE = 3
SQUARE_RADIUS = SQUARE_SIZE // 2
LEGEND_HEIGHT, LEGEND_MARKER_ROW, LEGEND_COLUMN_STRIDE = 6, 4, 4

# Palette and object counts.
PALETTE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
NUM_COLOR_PAIRS, SQUARE_COUNT_CHOICES = 3, [8, 9]

EMPTY, MAX_GENERATION_ATTEMPTS = 0, 300


def _color_pairs() -> tuple[list[int], list[int]]:
    """Return disjoint sprite and marker palettes in legend order."""
    picked = random.sample(PALETTE_COLORS, NUM_COLOR_PAIRS * 2)
    return picked[:NUM_COLOR_PAIRS], picked[NUM_COLOR_PAIRS:]


def _place_centers(n: int) -> list[tuple[int, int]] | None:
    """Place non-overlapping 3x3 squares and return their centers."""
    occ = np.full(FIELD_SHAPE, EMPTY, dtype=int)
    patch = np.ones((SQUARE_SIZE, SQUARE_SIZE), dtype=int)
    centers: list[tuple[int, int]] = []
    for _ in range(n):
        try:
            r0, c0 = random_free_location(occ, patch, transparent=EMPTY)
        except ValueError:
            return None
        paste(occ, patch, r0, c0)
        centers.append((r0 + SQUARE_RADIUS, c0 + SQUARE_RADIUS))
    return centers


def _touch_graph(centers: list[tuple[int, int]]) -> list[set[int]]:
    """Connect square indices whose rendered 3x3 blocks share an edge."""
    neighbors = [set() for _ in centers]
    for i, (r1, c1) in enumerate(centers):
        for j, (r2, c2) in enumerate(centers[i + 1 :], start=i + 1):
            dr = abs(r1 - r2)
            dc = abs(c1 - c2)
            if (dr == SQUARE_SIZE and dc < SQUARE_SIZE) or (dc == SQUARE_SIZE and dr < SQUARE_SIZE):
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _assign_square_ids(centers: list[tuple[int, int]]) -> list[int] | None:
    """Assign all color ids while preventing same-color touching squares."""
    neighbors = _touch_graph(centers)
    order = list(range(len(centers)))
    random.shuffle(order)
    order.sort(key=lambda i: len(neighbors[i]), reverse=True)
    ids: list[int | None] = [None] * len(centers)
    required = set(range(NUM_COLOR_PAIRS))

    def backtrack(pos: int, used: set[int]) -> bool:
        if pos == len(order):
            return used == required
        if len(required - used) > len(order) - pos:
            return False
        i = order[pos]
        choices = list(range(NUM_COLOR_PAIRS))
        random.shuffle(choices)
        for color_id in choices:
            if any(ids[j] == color_id for j in neighbors[i]):
                continue
            ids[i] = color_id
            if backtrack(pos + 1, used | {color_id}):
                return True
            ids[i] = None
        return False

    if not backtrack(0, set()):
        return None
    return [int(x) for x in ids]


def _build_grids(
    centers: list[tuple[int, int]],
    square_ids: list[int],
    sprite_colors: list[int],
    marker_colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return an assembled input grid and the matching oracle output."""
    out = np.full(FIELD_SHAPE, BACKGROUND_COLOR, dtype=int)
    lower = np.full(FIELD_SHAPE, BACKGROUND_COLOR, dtype=int)
    for (r, c), idx in zip(centers, square_ids):
        patch = np.full((SQUARE_SIZE, SQUARE_SIZE), sprite_colors[idx], dtype=int)
        paste(out, patch, r - SQUARE_RADIUS, c - SQUARE_RADIUS)
        lower[r, c] = marker_colors[idx]

    legend = np.full((LEGEND_HEIGHT, FIELD_WIDTH), BACKGROUND_COLOR, dtype=int)
    for i, (sprite, marker) in enumerate(zip(sprite_colors, marker_colors)):
        c0 = i * LEGEND_COLUMN_STRIDE
        legend[:SQUARE_SIZE, c0 : c0 + SQUARE_SIZE] = sprite
        legend[LEGEND_MARKER_ROW, c0 + SQUARE_RADIUS] = marker

    sep = np.full((1, FIELD_WIDTH), SEPARATOR_COLOR, dtype=int)
    return np.vstack([legend, sep, lower]), out


def _gen_attempt() -> np.ndarray | None:
    sprite_colors, marker_colors = _color_pairs()
    n_squares = random.choice(SQUARE_COUNT_CHOICES)
    centers = _place_centers(n_squares)
    if centers is None:
        return None
    square_ids = _assign_square_ids(centers)
    if square_ids is None:
        return None

    inp, out = _build_grids(centers, square_ids, sprite_colors, marker_colors)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid 20fb2937 input")
