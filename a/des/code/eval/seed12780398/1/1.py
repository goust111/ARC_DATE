from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        (coords[:, 0] == 0).any()
        or (coords[:, 0] == h - 1).any()
        or (coords[:, 1] == 0).any()
        or (coords[:, 1] == w - 1).any()
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _enclosed_rectangles(g: np.ndarray) -> list[dict]:
    rects = []
    for coords in _components(g != GREY):
        if _touches_boundary(coords, g.shape):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == GREY):
            continue
        rects.append({"bbox": (r0, c0, r1, c1), "patch": patch.copy()})
    return rects


def _parse_entry(patch: np.ndarray) -> tuple[int, np.ndarray] | None:
    if patch.shape != (3, 4):
        return None
    if np.any(patch[:, 1:] == GREY):
        return None
    marker_rows = np.flatnonzero(patch[:, 0] != BLACK)
    if marker_rows.size != 1:
        return None
    marker_color = int(patch[int(marker_rows[0]), 0])
    if marker_color == GREY:
        return None
    texture = patch[:, 1:]
    if len(set(map(int, texture.reshape(-1)))) != 2:
        return None
    return marker_color, texture.copy()


def _room_seed(patch: np.ndarray) -> int | None:
    seeds = np.argwhere(patch != BLACK)
    if seeds.shape[0] != 1:
        return None
    r, c = map(int, seeds[0])
    color = int(patch[r, c])
    if color == GREY:
        return None
    return color


def _tile(texture: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None] % 3
    cols = np.arange(w)[None, :] % 3
    return texture[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room by looking up its seed color in the bottom dictionary.

    - Grey walls define enclosed rectangular regions.
    - A room interior contains exactly one non-black seed pixel.
    - A dictionary entry is a 3x4 enclosed rectangle: one seed pixel on the left
      and a 3x3 texture on the right.
    - The room interior is overwritten by the matched texture, tiled from the
      room's inner top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    rects = _enclosed_rectangles(g)

    palette: dict[int, np.ndarray] = {}
    rooms: list[tuple[tuple[int, int, int, int], int]] = []
    for rect in rects:
        patch = rect["patch"]
        entry = _parse_entry(patch)
        if entry is not None:
            seed_color, texture = entry
            assert seed_color not in palette, "seed colors must map to unique dictionary entries"
            palette[seed_color] = texture
            continue

        seed_color = _room_seed(patch)
        if seed_color is not None:
            rooms.append((rect["bbox"], seed_color))

    assert palette, "expected at least one dictionary entry"
    assert rooms, "expected at least one room"

    out = g.copy()
    for (r0, c0, r1, c1), seed_color in rooms:
        assert seed_color in palette, "every room seed must appear in the dictionary"
        out[r0 : r1 + 1, c0 : c1 + 1] = _tile(palette[seed_color], (r1 - r0 + 1, c1 - c0 + 1))
    return out


# --------------- generator ---------------

GRID_OPTIONS = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 2))
ROOM_SIZE_RANGE = (3, 6)
EXTRA_WIDTH_MAX = 4
EXTRA_HEIGHT_MAX = 3
MAX_WIDTH = 36
MAX_HEIGHT = 32
ENTRY_STRIDE = 5
DICT_HEIGHT = 5
GEN_RETRIES = 400

BASE_MASKS = (
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
)


def _vary_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    out = np.rot90(out, random.randrange(4))
    return out.copy()


def _sample_texture(seed_color: int, used: set[bytes]) -> np.ndarray | None:
    choices = [color for color in NON_SPECIAL_COLORS if color != seed_color]
    for _ in range(80):
        a, b = random.sample(choices, 2)
        mask = _vary_mask(random.choice(BASE_MASKS))
        texture = np.where(mask == 1, a, b).astype(int)
        sig = texture.tobytes()
        if sig in used:
            continue
        used.add(sig)
        return texture
    return None


def _draw_room_grid(
    grid: np.ndarray,
    top: int,
    left: int,
    heights: list[int],
    widths: list[int],
    seeds: list[int],
) -> list[dict]:
    row_edges = [top]
    for height in heights:
        row_edges.append(row_edges[-1] + height + 1)
    col_edges = [left]
    for width in widths:
        col_edges.append(col_edges[-1] + width + 1)

    bottom = row_edges[-1]
    right = col_edges[-1]
    for r in row_edges:
        grid[r, left:right + 1] = GREY
    for c in col_edges:
        grid[top:bottom + 1, c] = GREY

    rooms = []
    room_index = 0
    for rr in range(len(heights)):
        for cc in range(len(widths)):
            r0 = row_edges[rr] + 1
            r1 = row_edges[rr + 1] - 1
            c0 = col_edges[cc] + 1
            c1 = col_edges[cc + 1] - 1
            seed_color = seeds[room_index]
            seed_r = random.randint(r0, r1)
            seed_c = random.randint(c0, c1)
            grid[seed_r, seed_c] = seed_color
            rooms.append({"bbox": (r0, c0, r1, c1), "seed": seed_color})
            room_index += 1
    return rooms


def _draw_dictionary(
    grid: np.ndarray,
    top: int,
    left: int,
    order: list[int],
    textures: dict[int, np.ndarray],
) -> None:
    width = ENTRY_STRIDE * len(order) + 1
    grid[top, left : left + width] = GREY
    grid[top + DICT_HEIGHT - 1, left : left + width] = GREY
    for idx in range(len(order) + 1):
        col = left + ENTRY_STRIDE * idx
        grid[top : top + DICT_HEIGHT, col] = GREY

    for idx, seed_color in enumerate(order):
        wall_col = left + ENTRY_STRIDE * idx
        grid[top + 1 + random.randrange(3), wall_col + 1] = seed_color
        grid[top + 1 : top + 4, wall_col + 2 : wall_col + 5] = textures[seed_color]


def _expected_output(inp: np.ndarray, rooms: list[dict], textures: dict[int, np.ndarray]) -> np.ndarray:
    out = inp.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = _tile(textures[room["seed"]], (r1 - r0 + 1, c1 - c0 + 1))
    return out


def _sample_canvas() -> tuple[np.ndarray, list[dict], dict[int, np.ndarray]] | None:
    n_rows, n_cols = random.choice(GRID_OPTIONS)
    room_count = n_rows * n_cols
    if room_count > len(NON_SPECIAL_COLORS):
        return None

    heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_rows)]
    widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(n_cols)]

    room_zone_h = sum(heights) + n_rows + 1
    room_zone_w = sum(widths) + n_cols + 1
    dict_w = ENTRY_STRIDE * room_count + 1

    min_h = room_zone_h + DICT_HEIGHT + 2
    min_w = max(room_zone_w + 2, dict_w + 2)
    if min_h > MAX_HEIGHT or min_w > MAX_WIDTH:
        return None

    height = random.randint(min_h, min(MAX_HEIGHT, min_h + EXTRA_HEIGHT_MAX))
    width = random.randint(min_w, min(MAX_WIDTH, min_w + EXTRA_WIDTH_MAX))

    dict_top = height - DICT_HEIGHT
    max_room_top = dict_top - room_zone_h - 1
    if max_room_top < 1:
        return None
    room_top = random.randint(1, max_room_top)
    room_left = random.randint(1, width - room_zone_w - 1)
    dict_left = random.randint(1, width - dict_w - 1)

    grid = np.full((height, width), BLACK, dtype=int)
    seeds = random.sample(list(NON_SPECIAL_COLORS), room_count)
    rooms = _draw_room_grid(grid, room_top, room_left, heights, widths, seeds)

    textures: dict[int, np.ndarray] = {}
    used_signatures: set[bytes] = set()
    for seed_color in seeds:
        texture = _sample_texture(seed_color, used_signatures)
        if texture is None:
            return None
        textures[seed_color] = texture

    order = seeds[:]
    random.shuffle(order)
    _draw_dictionary(grid, dict_top, dict_left, order, textures)
    return grid, rooms, textures


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_canvas()
    if sampled is None:
        return None
    inp, rooms, textures = sampled
    expected = _expected_output(inp, rooms, textures)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 1")
