import random
from collections import deque

import numpy as np


BG = 0
SPRITE_COLORS = (1, 2, 3, 4)
GRAY = 5
QUADRANT_MARKERS = {
    "tl": 6,
    "tr": 7,
    "bl": 8,
    "br": 9,
}
MAX_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _foreground_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _quadrant_fill_map(inp: np.ndarray) -> dict[str, int]:
    return {
        "tl": int(inp[0, 0]),
        "tr": int(inp[0, -1]),
        "bl": int(inp[-1, 0]),
        "br": int(inp[-1, -1]),
    }


def _normalize_sprite(sprite: np.ndarray) -> dict[str, object]:
    vals = sorted(set(int(v) for v in np.unique(sprite)) - {BG, GRAY})
    assert len(vals) == 1, "each sprite must have one base color plus seam markers"
    color = vals[0]
    left_rows = [r for r in range(sprite.shape[0]) if int(sprite[r, 0]) == GRAY]
    right_rows = [r for r in range(sprite.shape[0]) if int(sprite[r, -1]) == GRAY]
    assert left_rows and right_rows, "each sprite must expose left and right seam markers"
    recolored = sprite.copy()
    recolored[recolored == GRAY] = color
    return {
        "input_grid": sprite.copy(),
        "grid": recolored,
        "color": color,
        "left_rows": left_rows,
        "right_rows": right_rows,
        "shape": recolored.shape,
    }


def _aligned_top(prev_top: int, prev_right_rows: list[int], next_left_rows: list[int]) -> int:
    assert len(prev_right_rows) == len(next_left_rows), "seam marker counts must match"
    top = prev_top + prev_right_rows[0] - next_left_rows[0]
    for a, b in zip(prev_right_rows, next_left_rows):
        assert prev_top + a == top + b, "all seam rows must align together"
    return top


def _place_sprites(
    sprites: list[dict[str, object]],
    *,
    align: bool,
) -> np.ndarray:
    tops = [0]
    lefts = [0]
    for i in range(1, len(sprites)):
        prev = sprites[i - 1]
        cur = sprites[i]
        if align:
            top = _aligned_top(
                tops[-1],
                prev["right_rows"],  # type: ignore[arg-type]
                cur["left_rows"],  # type: ignore[arg-type]
            )
        else:
            top = 0
        tops.append(top)
        lefts.append(lefts[-1] + prev["grid"].shape[1])  # type: ignore[index]

    min_top = min(tops)
    norm_tops = [t - min_top for t in tops]
    height = max(t + spr["grid"].shape[0] for t, spr in zip(norm_tops, sprites))  # type: ignore[index]
    width = sum(spr["grid"].shape[1] for spr in sprites)  # type: ignore[index]
    out = np.zeros((height, width), dtype=int)
    for top, left, spr in zip(norm_tops, lefts, sprites):
        patch = spr["grid"]  # type: ignore[assignment]
        h, w = patch.shape
        out[top : top + h, left : left + w][patch != BG] = patch[patch != BG]
    return out


def _enclosed_zero_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            touches_boundary = False
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                if cr in (0, h - 1) or cc in (0, w - 1):
                    touches_boundary = True
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == BG:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_boundary:
                comps.append(comp)
    return comps


def _hole_quadrant(coords: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    h, w = shape
    n = len(coords)
    sum_r = sum(r for r, _ in coords)
    sum_c = sum(c for _, c in coords)
    top = 2 * sum_r < n * (h - 1)
    left = 2 * sum_c < n * (w - 1)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _fill_holes(grid: np.ndarray, fill_map: dict[str, int]) -> np.ndarray:
    out = grid.copy()
    for comp in _enclosed_zero_components(out):
        quad = _hole_quadrant(comp, out.shape)
        color = fill_map[quad]
        for r, c in comp:
            out[r, c] = color
    return out


def _sprites_from_input(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[str, int]]:
    g = np.asarray(grid, dtype=int).copy()
    fill_map = _quadrant_fill_map(g)
    g[0, 0] = BG
    g[0, -1] = BG
    g[-1, 0] = BG
    g[-1, -1] = BG
    comps = _foreground_components(g)
    assert len(comps) == 4, "input must contain exactly four sprites"
    comps.sort(key=lambda comp: _bbox(comp)[0])
    sprites = [_normalize_sprite(_crop_component(g, comp)) for comp in comps]
    return sprites, fill_map


def _assemble_solution(
    sprites: list[dict[str, object]],
    fill_map: dict[str, int],
    *,
    reverse_order: bool,
    align: bool,
    fill: bool,
) -> np.ndarray:
    ordered = list(reversed(sprites)) if reverse_order else list(sprites)
    placed = _place_sprites(ordered, align=align)
    return _fill_holes(placed, fill_map) if fill else placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the top-to-bottom sprite order, seam-align the new neighbors, then fill the enclosed cavities by quadrant color.

    Input structure:
    - The background is black, with four colored sprites separated vertically.
    - Each sprite is a single-color 4-connected shape with gray seam markers on its left and right bbox edges.
    - The four corners contain single-pixel quadrant-color markers for top-left, top-right, bottom-left, and bottom-right.

    Rule:
    - Sort the four sprites by their bbox top row and reverse that order.
    - Crop each sprite to its bbox, recolor every gray seam marker to the sprite's own color, and concatenate the reversed sprites left-to-right.
    - For each adjacent pair, shift the right sprite vertically so the previous sprite's right-edge seam rows align with the next sprite's left-edge seam rows.
    - Find every 4-connected black region enclosed by the concatenated sprites and fill it with the color indicated by the quadrant containing that region's centroid.

    Key insight:
    - Reversing the vertical order changes which sprites become neighbors, so the seam-alignment offsets change first, and the filled cavities only emerge after that new aligned concatenation.
    """
    sprites, fill_map = _sprites_from_input(np.asarray(grid, dtype=int))
    return _assemble_solution(sprites, fill_map, reverse_order=True, align=True, fill=True)


def _all_row_sets(h: int) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    for r in range(1, h - 1):
        rows.append((r,))
    for r in range(1, h - 2):
        rows.append((r, r + 1))
    return rows


def _choose_middle_rows(rng: random.Random, h_left: int, h_right: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left_opts = _all_row_sets(h_left)
    right_opts = _all_row_sets(h_right)
    pairs = [
        (a, b)
        for a in left_opts
        for b in right_opts
        if len(a) == len(b) and a[0] != b[0]
    ]
    assert pairs
    return rng.choice(pairs)


def _make_sprite(
    kind: str,
    h: int,
    w: int,
    left_rows: tuple[int, ...],
    right_rows: tuple[int, ...],
    color: int,
) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    g[0, :] = color
    g[-1, :] = color
    if kind == "open_right":
        g[:, 0] = color
        for r in right_rows:
            g[r, :] = color
    else:
        assert kind == "open_left"
        g[:, -1] = color
        for r in left_rows:
            g[r, :] = color
    for r in left_rows:
        g[r, 0] = GRAY
    for r in right_rows:
        g[r, -1] = GRAY
    return g


def _build_output_sprites(rng: random.Random) -> list[dict[str, object]]:
    colors = list(SPRITE_COLORS)
    rng.shuffle(colors)
    h_a = rng.randint(4, 5)
    h_b = rng.randint(4, 5)
    w0 = rng.randint(4, 6)
    w1 = rng.randint(4, 6)
    w2 = rng.randint(4, 6)
    w3 = rng.randint(4, 6)

    pair1_rows = rng.choice(_all_row_sets(h_a))
    pair2_rows = rng.choice(_all_row_sets(h_b))
    mid_right_rows, mid_left_rows = _choose_middle_rows(rng, h_a, h_b)

    far_left = rng.choice(_all_row_sets(h_a))
    far_right = rng.choice(_all_row_sets(h_b))

    raw = [
        _make_sprite("open_right", h_a, w0, far_left, pair1_rows, colors[0]),
        _make_sprite("open_left", h_a, w1, pair1_rows, mid_right_rows, colors[1]),
        _make_sprite("open_right", h_b, w2, mid_left_rows, pair2_rows, colors[2]),
        _make_sprite("open_left", h_b, w3, pair2_rows, far_right, colors[3]),
    ]
    return [_normalize_sprite(spr) for spr in raw]


def _derive_input(rng: random.Random, output_sprites: list[dict[str, object]]) -> np.ndarray:
    input_sprites = list(reversed(output_sprites))
    heights = [spr["grid"].shape[0] for spr in input_sprites]  # type: ignore[index]
    widths = [spr["grid"].shape[1] for spr in input_sprites]  # type: ignore[index]
    gaps = [rng.randint(1, 2) for _ in range(3)]
    h = 2 + sum(heights) + sum(gaps)
    w = max(widths) + rng.randint(3, 5)
    grid = np.zeros((h, w), dtype=int)

    top = 1
    for i, spr in enumerate(input_sprites):
        patch = spr["input_grid"]  # type: ignore[assignment]
        ph, pw = patch.shape
        left = rng.randint(1, w - pw - 1)
        grid[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
        if i < len(gaps):
            top += ph + gaps[i]

    grid[0, 0] = QUADRANT_MARKERS["tl"]
    grid[0, -1] = QUADRANT_MARKERS["tr"]
    grid[-1, 0] = QUADRANT_MARKERS["bl"]
    grid[-1, -1] = QUADRANT_MARKERS["br"]
    return grid


def _valid_instance(inp: np.ndarray, output_sprites: list[dict[str, object]]) -> bool:
    fill_map = QUADRANT_MARKERS.copy()
    expected = _assemble_solution(output_sprites, fill_map, reverse_order=False, align=True, fill=True)

    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    holes = _enclosed_zero_components(
        _assemble_solution(output_sprites, fill_map, reverse_order=False, align=True, fill=False)
    )
    if len(holes) < 2:
        return False
    quadrants = {_hole_quadrant(comp, expected.shape) for comp in holes}
    if len(quadrants) < 2:
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    try:
        no_reverse = _assemble_solution(output_sprites, fill_map, reverse_order=True, align=True, fill=True)
    except AssertionError:
        no_reverse = None
    if no_reverse is not None and np.array_equal(no_reverse, expected):
        return False

    no_align = _assemble_solution(output_sprites, fill_map, reverse_order=False, align=False, fill=True)
    if np.array_equal(no_align, expected):
        return False

    no_fill = _assemble_solution(output_sprites, fill_map, reverse_order=False, align=True, fill=False)
    if np.array_equal(no_fill, expected):
        return False

    g = inp.copy()
    g[0, 0] = g[0, -1] = g[-1, 0] = g[-1, -1] = BG
    comps = _foreground_components(g)
    if len(comps) != 4:
        return False
    tops = sorted(_bbox(comp)[0] for comp in comps)
    if len(set(tops)) != 4:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    output_sprites = _build_output_sprites(rng)
    inp = _derive_input(rng, output_sprites)
    if not _valid_instance(inp, output_sprites):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
