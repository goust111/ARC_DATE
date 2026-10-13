from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEPARATOR = 8
WALL = 9
PALETTE_COLORS = tuple(range(1, 8))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == SEPARATOR, axis=1))[0]
    assert rows.size > 0, "expected a full-width azure separator row"
    return int(rows[0])


def _try_parse_palette(palette: np.ndarray, tile_size: int) -> dict[int, np.ndarray] | None:
    ph, pw = palette.shape
    if ph not in (tile_size, tile_size * 2):
        return None

    band_count = ph // tile_size
    mapping: dict[int, np.ndarray] = {}
    used = np.zeros_like(palette, dtype=bool)

    for band in range(band_count):
        r0 = band * tile_size
        block = palette[r0 : r0 + tile_size, :]
        found = False
        c = 0
        while c < pw:
            if np.all(block[:, c] == BG):
                c += 1
                continue
            if c + 1 + tile_size > pw:
                return None
            if int(block[0, c]) == BG or np.any(block[1:, c] != BG):
                return None

            chip = int(block[0, c])
            pattern = block[:, c + 1 : c + 1 + tile_size]
            colors = {int(v) for v in np.unique(pattern) if int(v) != BG}
            if len(colors) != 1:
                return None
            if chip in (SEPARATOR, WALL) or chip in mapping:
                return None

            pattern_color = next(iter(colors))
            if pattern_color in (SEPARATOR, WALL) or pattern_color == chip:
                return None

            mapping[chip] = pattern.copy()
            used[r0, c] = True
            used[r0 : r0 + tile_size, c + 1 : c + 1 + tile_size] = True
            found = True
            c += 1 + tile_size
        if not found:
            return None

    if len(mapping) < 2:
        return None
    if np.any((palette != BG) & ~used):
        return None
    return mapping


def _parse_palette(grid: np.ndarray) -> tuple[dict[int, np.ndarray], int, int]:
    sep = _find_separator_row(grid)
    palette = grid[:sep, :]
    for tile_size in (2, 3):
        mapping = _try_parse_palette(palette, tile_size)
        if mapping is not None:
            return mapping, tile_size, sep
    raise AssertionError("failed to parse palette legend")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled chamber by tiling the palette pattern keyed by its seed color.

    Input structure:
    - The background is black.
    - Above a full-width azure separator row sits a palette made of one or two horizontal bands.
    - In each palette entry, a 1x1 color chip is immediately followed by a fixed-size monochrome pattern block.
    - Below the separator, maroon 1-cell walls enclose several chambers; each chamber contains exactly one non-black, non-maroon seed pixel.

    Rule:
    - Read every palette entry as a mapping from chip color to its adjacent pattern block.
    - In the lower field, find each connected non-wall chamber and locate its seed color.
    - Look up the seed color in the palette, then tile that pattern across the entire chamber.
    - Use the seed position as the tiling anchor, so the seed cell becomes the pattern's top-left phase.
    - Leave the palette, separator, and maroon walls unchanged.

    Key insight:
    - The top palette provides a visible color-to-pattern legend, while each chamber's seed supplies both the lookup key and the alignment origin for periodic filling.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mapping, tile_size, sep = _parse_palette(g)

    lower = g[sep + 1 :, :]
    room_mask = lower != WALL
    for comp in _components(room_mask):
        cells = [(r + sep + 1, c) for r, c in comp]
        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BG, WALL)]
        if len(seeds) != 1:
            continue

        seed_r, seed_c = seeds[0]
        seed_color = int(g[seed_r, seed_c])
        pattern = mapping[seed_color]
        for r, c in cells:
            out[r, c] = int(pattern[(r - seed_r) % tile_size, (c - seed_c) % tile_size])

    return out


def _random_pattern(tile_size: int, color: int, forbidden: set[bytes]) -> np.ndarray | None:
    for _ in range(80):
        mask = np.random.randint(0, 2, size=(tile_size, tile_size), dtype=int)
        if int(mask.sum()) in (0, tile_size * tile_size):
            continue
        pattern = np.where(mask == 1, color, BG).astype(int)
        sig = pattern.tobytes()
        if sig in forbidden:
            continue
        forbidden.add(sig)
        return pattern
    return None


def _band_entry_counts(n_entries: int, n_bands: int) -> list[int]:
    if n_bands == 1:
        return [n_entries]
    first = random.randint(1, n_entries - 1)
    return [first, n_entries - first]


def _band_width(entry_count: int, tile_size: int, gaps: list[int]) -> int:
    return entry_count * (tile_size + 1) + sum(gaps)


def _coarse_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    out: list[tuple[int, int]] = []
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _partition_cells(rows: int, cols: int, n_regions: int) -> dict[tuple[int, int], int]:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    seeds = random.sample(cells, n_regions)
    region_of: dict[tuple[int, int], int] = {cell: idx for idx, cell in enumerate(seeds)}
    while len(region_of) < len(cells):
        options: list[tuple[int, tuple[int, int]]] = []
        for cell, rid in list(region_of.items()):
            for nb in _coarse_neighbors(cell, rows, cols):
                if nb not in region_of:
                    options.append((rid, nb))
        assert options
        rid, nb = random.choice(options)
        region_of[nb] = rid
    return region_of


def _render_main_area(
    coarse_rows: int, coarse_cols: int, cell_size: int, region_of: dict[tuple[int, int], int]
) -> tuple[np.ndarray, dict[int, set[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    h = coarse_rows * (cell_size + 1) + 1
    w = coarse_cols * (cell_size + 1) + 1
    area = np.full((h, w), WALL, dtype=int)
    region_pixels: dict[int, set[tuple[int, int]]] = {}
    seed_candidates: dict[int, list[tuple[int, int]]] = {}

    for rr in range(coarse_rows):
        for cc in range(coarse_cols):
            rid = region_of[(rr, cc)]
            region_pixels.setdefault(rid, set())
            seed_candidates.setdefault(rid, [])

            r0 = 1 + rr * (cell_size + 1)
            c0 = 1 + cc * (cell_size + 1)
            area[r0 : r0 + cell_size, c0 : c0 + cell_size] = BG
            for r in range(r0, r0 + cell_size):
                for c in range(c0, c0 + cell_size):
                    region_pixels[rid].add((r, c))
                    seed_candidates[rid].append((r, c))

            if cc + 1 < coarse_cols and region_of[(rr, cc + 1)] == rid:
                area[r0 : r0 + cell_size, c0 + cell_size] = BG
                for r in range(r0, r0 + cell_size):
                    region_pixels[rid].add((r, c0 + cell_size))

            if rr + 1 < coarse_rows and region_of[(rr + 1, cc)] == rid:
                area[r0 + cell_size, c0 : c0 + cell_size] = BG
                for c in range(c0, c0 + cell_size):
                    region_pixels[rid].add((r0 + cell_size, c))

    return area, region_pixels, seed_candidates


def _paint_expected(
    grid: np.ndarray,
    mapping: dict[int, np.ndarray],
    tile_size: int,
    region_pixels: dict[int, set[tuple[int, int]]],
    seed_pos: dict[int, tuple[int, int]],
    seed_color: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for rid, pixels in region_pixels.items():
        sr, sc = seed_pos[rid]
        pattern = mapping[seed_color[rid]]
        for r, c in pixels:
            out[r, c] = int(pattern[(r - sr) % tile_size, (c - sc) % tile_size])
    return out


def _gen_attempt() -> np.ndarray | None:
    tile_size = random.choice((2, 3))
    palette_bands = random.choice((1, 2))
    cell_size = random.choice((2, 3))
    coarse_rows = random.randint(3, 5)
    coarse_cols = random.randint(3, 6)
    max_regions = min(5, coarse_rows * coarse_cols)
    n_regions = random.randint(2, max_regions)
    n_entries = random.randint(2, min(4, n_regions))

    region_of = _partition_cells(coarse_rows, coarse_cols, n_regions)
    main_area, local_region_pixels, seed_candidates = _render_main_area(
        coarse_rows, coarse_cols, cell_size, region_of
    )
    main_h, main_w = main_area.shape

    band_counts = _band_entry_counts(n_entries, palette_bands)
    band_gaps = [[random.randint(0, 1) for _ in range(max(0, count - 1))] for count in band_counts]
    required_palette_width = max(
        _band_width(count, tile_size, gaps) for count, gaps in zip(band_counts, band_gaps, strict=True)
    )

    min_width = max(required_palette_width, main_w)
    if min_width > 30:
        return None

    palette_height = palette_bands * tile_size
    if palette_height + 1 + main_h > 30:
        return None

    width = random.randint(min_width, 30)
    max_bottom_margin = min(2, 30 - (palette_height + 1 + main_h))
    bottom_margin = random.randint(0, max_bottom_margin)
    height = palette_height + 1 + main_h + bottom_margin

    grid = np.full((height, width), BG, dtype=int)
    grid[palette_height, :] = SEPARATOR

    chip_colors = random.sample(PALETTE_COLORS, n_entries)
    forbidden_patterns: set[bytes] = set()
    mapping: dict[int, np.ndarray] = {}

    remaining_pattern_colors = list(PALETTE_COLORS)
    random.shuffle(remaining_pattern_colors)
    for chip in chip_colors:
        choices = [c for c in remaining_pattern_colors if c != chip]
        if not choices:
            choices = [c for c in PALETTE_COLORS if c != chip]
        pattern_color = random.choice(choices)
        pattern = _random_pattern(tile_size, pattern_color, forbidden_patterns)
        if pattern is None:
            return None
        mapping[chip] = pattern

    chip_idx = 0
    for band, count in enumerate(band_counts):
        band_width = _band_width(count, tile_size, band_gaps[band])
        left = random.randint(0, width - band_width)
        c = left
        r0 = band * tile_size
        for entry_idx in range(count):
            chip = chip_colors[chip_idx]
            chip_idx += 1
            grid[r0, c] = chip
            grid[r0 : r0 + tile_size, c + 1 : c + 1 + tile_size] = mapping[chip]
            c += tile_size + 1
            if entry_idx + 1 < count:
                c += band_gaps[band][entry_idx]

    main_left = random.randint(0, width - main_w)
    main_top = palette_height + 1
    grid[main_top : main_top + main_h, main_left : main_left + main_w] = main_area

    region_ids = sorted(local_region_pixels)
    assigned_colors = chip_colors.copy()
    while len(assigned_colors) < len(region_ids):
        assigned_colors.append(random.choice(chip_colors))
    random.shuffle(assigned_colors)

    global_region_pixels: dict[int, set[tuple[int, int]]] = {}
    seed_pos: dict[int, tuple[int, int]] = {}
    seed_color: dict[int, int] = {}
    for rid, chip in zip(region_ids, assigned_colors, strict=True):
        pixels = {
            (main_top + r, main_left + c)
            for r, c in local_region_pixels[rid]
        }
        candidates = [
            (main_top + r, main_left + c)
            for r, c in seed_candidates[rid]
        ]
        seed = random.choice(candidates)
        grid[seed] = chip
        global_region_pixels[rid] = pixels
        seed_pos[rid] = seed
        seed_color[rid] = chip

    expected = _paint_expected(grid, mapping, tile_size, global_region_pixels, seed_pos, seed_color)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_ATTEMPTS} attempts")
