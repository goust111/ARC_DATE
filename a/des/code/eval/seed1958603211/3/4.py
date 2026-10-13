import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIR_TO_ROT = {"up": 0, "right": 1, "down": 2, "left": 3}

GENERATE_RETRIES = 300
PLACEMENT_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
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


def _rotation_key(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return arr.shape, arr.tobytes()


def _all_rotations_distinct(arr: np.ndarray) -> bool:
    keys = {_rotation_key(np.rot90(arr, -k)) for k in range(4)}
    return len(keys) == 4


def _tile_texture(texture: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, None] % texture.shape[0]
    cols = np.arange(shape[1])[None, :] % texture.shape[1]
    return texture[rows, cols]


def _extract_legend(grid: np.ndarray) -> dict[int, np.ndarray]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey legend panel"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]

    full_rows = np.flatnonzero(np.all(panel == GREY, axis=1))
    full_cols = np.flatnonzero(np.all(panel == GREY, axis=0))
    assert full_rows.size >= 2 and full_cols.size >= 2, "legend needs outer separators"
    assert full_rows[0] == 0 and full_rows[-1] == panel.shape[0] - 1
    assert full_cols[0] == 0 and full_cols[-1] == panel.shape[1] - 1

    row_bands = [(a + 1, b) for a, b in zip(full_rows, full_rows[1:]) if b - a > 1]
    col_bands = [(a + 1, b) for a, b in zip(full_cols, full_cols[1:]) if b - a > 1]
    assert row_bands and col_bands, "legend must contain at least one block"

    mapping: dict[int, np.ndarray] = {}
    block_shapes = set()
    for rr0, rr1 in row_bands:
        for cc0, cc1 in col_bands:
            block = panel[rr0:rr1, cc0:cc1]
            if np.all(block == BLACK):
                continue
            label = int(block[0, 0])
            assert label not in (BLACK, GREY, MAROON), "invalid label color"
            texture = block[1:, 1:]
            assert texture.size > 0, "texture block must have a non-empty interior"
            assert np.all(texture != BLACK), "texture should be visually explicit"
            assert label not in mapping, "label colors must be unique"
            mapping[label] = texture.copy()
            block_shapes.add(block.shape)

    assert mapping, "legend mapping cannot be empty"
    assert len(block_shapes) == 1, "legend blocks must be uniform"
    return mapping


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    rooms: list[dict[str, object]] = []
    for coords in _components(grid == MAROON):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "room bbox must leave interior space"

        patch = np.zeros((h, w), dtype=bool)
        patch[rows - r0, cols - c0] = True
        assert not np.any(patch[1:-1, 1:-1]), "room border must be 1-cell thick"

        gaps = {
            "up": int(w - patch[0, :].sum()),
            "right": int(h - patch[:, -1].sum()),
            "down": int(w - patch[-1, :].sum()),
            "left": int(h - patch[:, 0].sum()),
        }
        gap_sides = [name for name, missing in gaps.items() if missing == 1]
        assert len(gap_sides) == 1, "room must have exactly one 1-cell exit"
        assert sum(gaps.values()) == 1, "room border may only contain one gap"
        direction = gap_sides[0]

        if direction in ("up", "down"):
            gap_idx = np.flatnonzero(~patch[0 if direction == "up" else -1, :])
            assert gap_idx.size == 1 and 0 < gap_idx[0] < w - 1
        else:
            gap_idx = np.flatnonzero(~patch[:, -1 if direction == "right" else 0])
            assert gap_idx.size == 1 and 0 < gap_idx[0] < h - 1

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = np.unique(interior)
        colors = colors[colors != BLACK]
        assert colors.size == 1, "each room interior must contain exactly one seed color"
        seed_color = int(colors[0])
        assert np.count_nonzero(interior == seed_color) == 1, "seed must be a single pixel"

        rooms.append({"bbox": (r0, c0, r1, c1), "direction": direction, "seed": seed_color})

    assert rooms, "expected at least one room"
    rooms.sort(key=lambda room: room["bbox"])
    return rooms


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    mapping = _extract_legend(g)
    rooms = _find_rooms(g)

    out = g.copy()
    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        direction = room["direction"]
        seed_color = room["seed"]
        assert seed_color in mapping, "every room seed must have a legend entry"
        texture = np.rot90(mapping[seed_color], -DIR_TO_ROT[direction])
        fill = _tile_texture(texture, (r1 - r0 - 1, c1 - c0 - 1))
        out[r0 + 1 : r1, c0 + 1 : c1] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each room interior with the legend texture selected by its seed color,
    rotated according to the side containing the room's border gap.
    """
    return _transform(grid)


def _make_texture(rng: random.Random, height: int, width: int, label: int) -> np.ndarray:
    palette = [color for color in COLOR_POOL if color != label]
    coeffs = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
    accent_cells = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1), (height // 2, width // 2)]
    for _ in range(200):
        color_count = rng.randint(2, 4)
        colors = rng.sample(palette, color_count)
        rr, cc = np.indices((height, width))
        a, b = rng.choice(coeffs)
        base = (a * rr + b * cc + rng.randrange(color_count)) % color_count
        texture = np.array(colors, dtype=int)[base]
        rng.shuffle(accent_cells)
        for ar, ac in accent_cells[: rng.randint(1, min(3, len(accent_cells)))]:
            texture[ar, ac] = colors[rng.randrange(color_count)]
        for idx, color in enumerate(colors):
            texture[idx % height, (idx * 2) % width] = color
        if len(np.unique(texture)) < 2:
            continue
        if _all_rotations_distinct(texture):
            return texture
    raise AssertionError("failed to synthesize a directional texture")


def _build_legend(
    labels: list[int],
    textures: dict[int, np.ndarray],
    orientation: str,
) -> np.ndarray:
    tex_h, tex_w = next(iter(textures.values())).shape
    block_h = tex_h + 1
    block_w = tex_w + 1
    count = len(labels)

    if orientation == "horizontal":
        panel = np.full((block_h + 2, count * block_w + count + 1), GREY, dtype=int)
        for idx, label in enumerate(labels):
            r0 = 1
            c0 = 1 + idx * (block_w + 1)
            panel[r0 : r0 + block_h, c0 : c0 + block_w] = BLACK
            panel[r0, c0] = label
            panel[r0 + 1 : r0 + block_h, c0 + 1 : c0 + block_w] = textures[label]
        return panel

    panel = np.full((count * block_h + count + 1, block_w + 2), GREY, dtype=int)
    for idx, label in enumerate(labels):
        r0 = 1 + idx * (block_h + 1)
        c0 = 1
        panel[r0 : r0 + block_h, c0 : c0 + block_w] = BLACK
        panel[r0, c0] = label
        panel[r0 + 1 : r0 + block_h, c0 + 1 : c0 + block_w] = textures[label]
    return panel


def _place_panel(grid: np.ndarray, panel: np.ndarray, corner: str) -> tuple[int, int]:
    h, w = grid.shape
    ph, pw = panel.shape
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, w - pw
    elif corner == "bl":
        r0, c0 = h - ph, 0
    else:
        r0, c0 = h - ph, w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = panel
    return r0, c0


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _draw_room(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    seed_color: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = MAROON
    grid[r1, c0 : c1 + 1] = MAROON
    grid[r0 : r1 + 1, c0] = MAROON
    grid[r0 : r1 + 1, c1] = MAROON

    if direction == "up":
        grid[r0, rng.randint(c0 + 1, c1 - 1)] = BLACK
    elif direction == "right":
        grid[rng.randint(r0 + 1, r1 - 1), c1] = BLACK
    elif direction == "down":
        grid[r1, rng.randint(c0 + 1, c1 - 1)] = BLACK
    else:
        grid[rng.randint(r0 + 1, r1 - 1), c0] = BLACK

    sr = rng.randint(r0 + 1, r1 - 1)
    sc = rng.randint(c0 + 1, c1 - 1)
    grid[sr, sc] = seed_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choice([4, 5])
    grid_h = rng.randint(28, 36)
    grid_w = rng.randint(28, 36)
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)

    seed_colors = rng.sample(list(COLOR_POOL), room_count)
    tex_h = rng.randint(2, 4)
    tex_w = rng.randint(2, 4)
    textures: dict[int, np.ndarray] = {}
    seen_textures = set()
    for color in seed_colors:
        for _ in range(80):
            texture = _make_texture(rng, tex_h, tex_w, color)
            key = _rotation_key(texture)
            if key not in seen_textures:
                textures[color] = texture
                seen_textures.add(key)
                break
        else:
            return None

    orientation = rng.choice(["horizontal", "vertical"])
    panel = _build_legend(rng.sample(seed_colors, len(seed_colors)), textures, orientation)
    ph, pw = panel.shape
    if ph > grid_h or pw > grid_w:
        return None

    corner = rng.choice(["tl", "tr", "bl", "br"])
    pr0, pc0 = _place_panel(grid, panel, corner)
    occupied = np.zeros_like(grid, dtype=bool)
    _reserve(occupied, pr0, pc0, pr0 + ph - 1, pc0 + pw - 1, pad=1)

    directions = ["up", "right", "down", "left"]
    if room_count == 5:
        directions.append(rng.choice(directions))
    rng.shuffle(directions)
    room_pairs = list(zip(seed_colors, directions))
    rng.shuffle(room_pairs)

    for seed_color, direction in room_pairs:
        placed = False
        base_h_min = max(3, tex_h)
        base_w_min = max(3, tex_w)
        base_h_max = base_h_min + (3 if room_count == 4 else 2)
        base_w_max = base_w_min + (4 if room_count == 4 else 3)
        for _ in range(PLACEMENT_RETRIES):
            inner_h = rng.randint(base_h_min, base_h_max)
            inner_w = rng.randint(base_w_min, base_w_max)
            room_h = inner_h + 2
            room_w = inner_w + 2
            if room_h >= grid_h - 1 or room_w >= grid_w - 1:
                continue
            r0 = rng.randint(1, grid_h - room_h - 1)
            c0 = rng.randint(1, grid_w - room_w - 1)
            r1 = r0 + room_h - 1
            c1 = c0 + room_w - 1
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            _draw_room(grid, (r0, c0, r1, c1), direction, seed_color, rng)
            _reserve(occupied, r0, c0, r1, c1, pad=1)
            placed = True
            break
        if not placed:
            return None

    try:
        out = _transform(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
