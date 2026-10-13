from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 5
SEP = 0
FILL = 7
TILE_SIZE = 4
INNER_SIZE = 2
GRID_BANDS = 3
STRIDE = TILE_SIZE + 1
LATTICE_SPAN = GRID_BANDS * TILE_SIZE + (GRID_BANDS - 1)
PALETTE = (1, 2, 3, 4, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def find_monochrome_components(grid: np.ndarray, *, exclude: set[int]) -> list[dict]:
    out: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in exclude):
        for coords in components_mask(grid == color):
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _frame_mask(size: int = TILE_SIZE) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


FRAME_MASK = _frame_mask()


def _is_frame_component(grid: np.ndarray, comp: dict) -> bool:
    if comp["area"] != 12:
        return False
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (TILE_SIZE, TILE_SIZE):
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    color = int(comp["color"])
    return bool(np.array_equal(patch == color, FRAME_MASK))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the scattered same-color sprites and use each color's sprite-count parity to decide which tiles invert their 2x2 occupancy.

    Input structure:
    - The background is grey, with a 3x3 layout of 4x4 tiles separated by single black rows and columns.
    - Each tile is a hollow monochrome border frame whose inner 2x2 area contains orange occupancy cells on grey.
    - Small monochrome 4-connected sprites of tile-border colors are scattered outside the tile lattice.

    Rule:
    - Remove every scattered sprite by turning its cells back to the grey background.
    - For each border color C, count how many removed sprites have color C.
    - Every tile with border color C inverts only its inner 2x2 area when that count is odd; orange becomes grey and grey becomes orange.
    - Tile borders and black separator lines never change.

    Key insight:
    - The outside sprites are not copied into the tiles; their count only matters through odd-versus-even parity, and all tiles sharing a border color use the same parity result.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)

    comps = find_monochrome_components(g, exclude={bg, SEP, FILL})
    frames = [comp for comp in comps if _is_frame_component(g, comp)]
    sprites = [comp for comp in comps if not _is_frame_component(g, comp)]

    color_counts = Counter(int(comp["color"]) for comp in sprites)

    for comp in sprites:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for frame in frames:
        color = int(frame["color"])
        if color_counts[color] % 2 == 0:
            continue
        r0, c0, _r1, _c1 = frame["bbox"]
        inner = out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
        inner[:, :] = np.where(inner == FILL, bg, FILL)

    return out


def _tile_origin(top: int, left: int, band_r: int, band_c: int) -> tuple[int, int]:
    return top + band_r * STRIDE, left + band_c * STRIDE


def _draw_tile(grid: np.ndarray, r0: int, c0: int, color: int, inner_mask: np.ndarray) -> None:
    grid[r0, c0 : c0 + TILE_SIZE] = color
    grid[r0 + TILE_SIZE - 1, c0 : c0 + TILE_SIZE] = color
    grid[r0 : r0 + TILE_SIZE, c0] = color
    grid[r0 : r0 + TILE_SIZE, c0 + TILE_SIZE - 1] = color
    grid[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3] = np.where(inner_mask, FILL, BG)


def _random_inner_mask(rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((INNER_SIZE, INNER_SIZE), dtype=bool)
    count = int(rng.integers(1, 4))
    picks = rng.choice(INNER_SIZE * INNER_SIZE, size=count, replace=False)
    mask.flat[picks] = True
    return mask


def _tight_patch_from_cells(cells: set[tuple[int, int]], color: int) -> np.ndarray:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    return patch


def _make_sprite(rng: np.random.Generator, color: int) -> np.ndarray:
    area = int(rng.integers(2, 5))
    dims = [(h, w) for h in range(1, 4) for w in range(1, 4) if h * w >= area]
    h, w = dims[int(rng.integers(len(dims)))]
    seed = (int(rng.integers(h)), int(rng.integers(w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _tight_patch_from_cells(cells, color)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _random_location_for_patch(
    allowed: np.ndarray, occupied: np.ndarray, patch_mask: np.ndarray, rng: np.random.Generator
) -> tuple[int, int] | None:
    h, w = allowed.shape
    ph, pw = patch_mask.shape
    blocked = _dilate8(occupied)
    candidates = []
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            allow_view = allowed[r0 : r0 + ph, c0 : c0 + pw]
            block_view = blocked[r0 : r0 + ph, c0 : c0 + pw]
            if np.all(allow_view[patch_mask]) and not np.any(block_view & patch_mask):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _assign_tile_colors(rng: np.random.Generator, colors: list[int]) -> list[int]:
    assignments = list(colors)
    while len(assignments) < GRID_BANDS * GRID_BANDS:
        assignments.append(int(rng.choice(colors)))
    rng.shuffle(assignments)
    return [int(v) for v in assignments]


def _sample_color_parities(rng: np.random.Generator, colors: list[int]) -> dict[int, int]:
    while True:
        odd_size = int(rng.integers(1, len(colors)))
        odd_colors = set(int(v) for v in rng.choice(colors, size=odd_size, replace=False))
        if 0 < len(odd_colors) < len(colors):
            break
    counts = {}
    for color in colors:
        if color in odd_colors:
            counts[color] = int(rng.choice([1, 3]))
        else:
            counts[color] = int(rng.choice([2, 4]))
    return counts


def _build_expected(
    grid: np.ndarray,
    sprites: list[tuple[np.ndarray, int, int]],
    tile_records: list[dict],
    sprite_counts: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for patch, r0, c0 in sprites:
        mask = patch != BG
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = BG
    for rec in tile_records:
        if sprite_counts[int(rec["color"])] % 2 == 0:
            continue
        r0, c0 = rec["origin"]
        inner = out[r0 + 1 : r0 + 3, c0 + 1 : c0 + 3]
        inner[:, :] = np.where(inner == FILL, BG, FILL)
    return out


def _generation_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    top = int(rng.integers(2, 5))
    bottom = int(rng.integers(2, 5))
    left = int(rng.integers(2, 5))
    right = int(rng.integers(2, 5))
    h = top + LATTICE_SPAN + bottom
    w = left + LATTICE_SPAN + right

    grid = np.full((h, w), BG, dtype=int)

    for r in (top + TILE_SIZE, top + TILE_SIZE + STRIDE):
        grid[r, :] = SEP
    for c in (left + TILE_SIZE, left + TILE_SIZE + STRIDE):
        grid[:, c] = SEP

    color_count = int(rng.integers(3, 5))
    colors = [int(v) for v in rng.choice(np.array(PALETTE), size=color_count, replace=False)]
    tile_colors = _assign_tile_colors(rng, colors)

    tile_records = []
    freq = Counter(tile_colors)
    repeated_colors = {color for color, count in freq.items() if count >= 2}
    if len(repeated_colors) < 2:
        return None

    for band_r in range(GRID_BANDS):
        for band_c in range(GRID_BANDS):
            idx = band_r * GRID_BANDS + band_c
            color = int(tile_colors[idx])
            r0, c0 = _tile_origin(top, left, band_r, band_c)
            inner_mask = _random_inner_mask(rng)
            _draw_tile(grid, r0, c0, color, inner_mask)
            tile_records.append({"origin": (r0, c0), "color": color, "inner_mask": inner_mask})

    sprite_counts = _sample_color_parities(rng, colors)
    odd_repeated = any(sprite_counts[c] % 2 == 1 for c in repeated_colors)
    even_repeated = any(sprite_counts[c] % 2 == 0 for c in repeated_colors)
    if not (odd_repeated and even_repeated):
        return None

    row_ids = np.arange(h)[:, None]
    col_ids = np.arange(w)[None, :]
    outside_lattice = (
        (row_ids < top)
        | (row_ids >= top + LATTICE_SPAN)
        | (col_ids < left)
        | (col_ids >= left + LATTICE_SPAN)
    )
    allowed = outside_lattice & (grid == BG)
    occupied = grid != BG

    placed_sprites: list[tuple[np.ndarray, int, int]] = []
    sprite_total = []
    for color in colors:
        for _ in range(sprite_counts[color]):
            sprite_total.append(color)
    rng.shuffle(sprite_total)

    for color in sprite_total:
        patch = _make_sprite(rng, int(color))
        location = _random_location_for_patch(allowed, occupied, patch != BG, rng)
        if location is None:
            return None
        r0, c0 = location
        patch_mask = patch != BG
        grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch_mask] = patch[patch_mask]
        occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][patch_mask] = True
        placed_sprites.append((patch, r0, c0))

    expected = _build_expected(grid, placed_sprites, tile_records, sprite_counts)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
