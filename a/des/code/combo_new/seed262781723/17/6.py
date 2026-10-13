import random

import numpy as np


BG = 0
GRAY = 5
PALETTE = tuple(color for color in range(10) if color not in {BG, GRAY})
SYMMETRY_ORDER = ("vertical", "horizontal", "both", "none")
SYMBOL_MASKS = {
    "vertical": np.array([[1], [1], [1]], dtype=bool),
    "horizontal": np.array([[1, 1, 1]], dtype=bool),
    "both": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "none": np.array([[1]], dtype=bool),
}

GRID_MIN = 18
GRID_MAX = 30
INTERIOR_MIN = 4
INTERIOR_MAX = 8
EXTRA_FRAME_PROB = 0.55
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _bbox(mask: np.ndarray):
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_to_bbox(arr: np.ndarray, mask: np.ndarray):
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _classify_symmetry(mask: np.ndarray) -> str:
    vertical = np.array_equal(mask, mask[:, ::-1])
    horizontal = np.array_equal(mask, mask[::-1, :])
    if vertical and horizontal:
        return "both"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "none"


def _match_symbol(symbol_mask: np.ndarray):
    for name, template in SYMBOL_MASKS.items():
        if symbol_mask.shape == template.shape and np.array_equal(symbol_mask, template):
            return name
    return None


def _tile_pattern(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    th, tw = tile.shape
    reps = ((h + th - 1) // th, (w + tw - 1) // tw)
    return np.tile(tile, reps)[:h, :w]


def _find_frames(grid: np.ndarray, bg: int):
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, GRAY}:
            continue
        for comp in _components(grid == color):
            rows = comp[:, 0]
            cols = comp[:, 1]
            r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh < 3 or ww < 3:
                continue
            expected_area = 2 * hh + 2 * ww - 4
            if comp.shape[0] != expected_area:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(sub[0, :] == color):
                continue
            if not np.all(sub[-1, :] == color):
                continue
            if not np.all(sub[:, 0] == color):
                continue
            if not np.all(sub[:, -1] == color):
                continue
            if hh > 2 and ww > 2 and np.any(sub[1:-1, 1:-1] == color):
                continue
            frames.append((r0, c0, r1, c1, color))
    frames.sort()
    return frames


def _legend_mapping(grid: np.ndarray, bg: int):
    h, w = grid.shape
    mapping = {}
    for comp in _components(grid != bg):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, c0, r1, c1 = int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if max(hh, ww) > 8:
            continue
        if r1 != h - 1 and c1 != w - 1:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        gray_mask = sub == GRAY
        color_mask = (sub != bg) & (sub != GRAY)
        if not gray_mask.any() or not color_mask.any():
            continue
        symbol_crop = _crop_to_bbox(gray_mask, gray_mask)
        kind = _match_symbol(symbol_crop)
        if kind is None:
            continue
        tile = _crop_to_bbox(sub, color_mask)
        mapping[kind] = tile.copy()
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed gray pattern by the legend tile that matches its symmetry class.

    Input structure:
    - The background is black.
    - Several colored 1-cell-thick rectangular frames contain gray pixel masks inside a black interior.
    - Four small legend items sit along the bottom edge or the right edge; each item combines a gray symbol and an adjacent colored tile.

    Rule:
    - Read the legend symbols as symmetry labels: vertical line = vertical symmetry, horizontal line = horizontal symmetry, plus = both axes, dot = no symmetry.
    - For each colored frame, inspect the gray mask inside the border and classify it by vertical, horizontal, both-axis, or no symmetry.
    - Replace the entire interior of that frame by tiling the colored legend tile associated with the matching symmetry class, cropped to the interior size.

    Key insight:
    - The gray interior pattern is only a clue for symmetry type; the actual output fill comes from a separate edge legend that maps visual symmetry symbols to replacement tiles.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping = _legend_mapping(g, bg)
    assert mapping, "expected legend items on the grid edge"

    out = g.copy()
    for r0, c0, r1, c1, _color in _find_frames(g, bg):
        if r1 - r0 <= 1 or c1 - c0 <= 1:
            continue
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        gray_mask = inner == GRAY
        assert gray_mask.any(), "each frame must contain a gray clue pattern"
        kind = _classify_symmetry(gray_mask)
        assert kind in mapping, f"missing legend tile for symmetry type {kind}"
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile_pattern(mapping[kind], inner.shape)
    return out


def _vertical_reflect(seed: np.ndarray, total_w: int) -> np.ndarray:
    if total_w % 2:
        half = total_w // 2
        left = seed[:, :half]
        center = seed[:, half : half + 1]
        return np.concatenate([left, center, left[:, ::-1]], axis=1)
    return np.concatenate([seed, seed[:, ::-1]], axis=1)


def _horizontal_reflect(seed: np.ndarray, total_h: int) -> np.ndarray:
    if total_h % 2:
        half = total_h // 2
        top = seed[:half, :]
        center = seed[half : half + 1, :]
        return np.concatenate([top, center, top[::-1, :]], axis=0)
    return np.concatenate([seed, seed[::-1, :]], axis=0)


def _random_bool_grid(rng: random.Random, h: int, w: int, p: float) -> np.ndarray:
    return np.array([[rng.random() < p for _ in range(w)] for _ in range(h)], dtype=bool)


def _make_gray_pattern(kind: str, h: int, w: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        density = rng.uniform(0.25, 0.7)
        if kind == "vertical":
            seed = _random_bool_grid(rng, h, (w + 1) // 2, density)
            mask = _vertical_reflect(seed, w)
        elif kind == "horizontal":
            seed = _random_bool_grid(rng, (h + 1) // 2, w, density)
            mask = _horizontal_reflect(seed, h)
        elif kind == "both":
            seed = _random_bool_grid(rng, (h + 1) // 2, (w + 1) // 2, density)
            mask = _horizontal_reflect(_vertical_reflect(seed, w), h)
        else:
            mask = _random_bool_grid(rng, h, w, density)
        if mask.sum() <= 1 or mask.sum() >= h * w:
            continue
        if _classify_symmetry(mask) == kind:
            return mask
    raise RuntimeError(f"failed to build {kind} gray pattern")


def _make_tile(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 3)
    w = rng.randint(2, 3)
    palette_size = 2 if rng.random() < 0.8 else 1
    colors = rng.sample(PALETTE, palette_size)
    tile = np.array([[rng.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
    if len(colors) > 1 and np.unique(tile).size == 1:
        rr = rng.randrange(h)
        cc = rng.randrange(w)
        tile[rr, cc] = colors[1]
    return tile


def _build_legend_item(kind: str, tile: np.ndarray, orientation: str) -> np.ndarray:
    symbol = SYMBOL_MASKS[kind]
    sh, sw = symbol.shape
    th, tw = tile.shape
    if orientation == "bottom":
        patch = np.full((max(sh, th), sw + tw), BG, dtype=int)
        patch[:sh, :sw][symbol] = GRAY
        patch[:th, sw : sw + tw] = tile
    else:
        patch = np.full((sh + th, max(sw, tw)), BG, dtype=int)
        patch[:sh, :sw][symbol] = GRAY
        patch[sh : sh + th, :tw] = tile
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, h: int, w: int):
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_frames(rng: random.Random, play_h: int, play_w: int):
    extra = 1 if rng.random() < EXTRA_FRAME_PROB else 0
    kinds = list(SYMMETRY_ORDER) + [rng.choice(SYMMETRY_ORDER)] * extra
    rng.shuffle(kinds)
    border_colors = rng.sample(PALETTE, len(kinds))
    occupied = np.zeros((play_h, play_w), dtype=bool)
    frames = []

    for kind, border_color in zip(kinds, border_colors):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            inner_h = rng.randint(INTERIOR_MIN, INTERIOR_MAX)
            inner_w = rng.randint(INTERIOR_MIN, INTERIOR_MAX)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > play_h or outer_w + 2 > play_w:
                continue
            r0 = rng.randint(1, play_h - outer_h - 1)
            c0 = rng.randint(1, play_w - outer_w - 1)
            if not _can_place(occupied, r0, c0, outer_h, outer_w):
                continue
            pattern = _make_gray_pattern(kind, inner_h, inner_w, rng)
            frames.append(
                {
                    "kind": kind,
                    "border_color": border_color,
                    "outer": (r0, c0, outer_h, outer_w),
                    "pattern": pattern,
                }
            )
            _mark_occupied(occupied, r0, c0, outer_h, outer_w)
            placed = True
            break
        if not placed:
            return None
    return frames


def _build_grid(rng: random.Random):
    orientation = rng.choice(("bottom", "right"))
    legend_order = list(SYMMETRY_ORDER)
    rng.shuffle(legend_order)
    tiles = {kind: _make_tile(rng) for kind in SYMMETRY_ORDER}
    items = [(kind, _build_legend_item(kind, tiles[kind], orientation)) for kind in legend_order]

    gap = 1
    if orientation == "bottom":
        band_h = max(item.shape[0] for _kind, item in items)
        legend_w = sum(item.shape[1] for _kind, item in items) + gap * (len(items) - 1)
        grid_h = rng.randint(max(GRID_MIN, band_h + 12), GRID_MAX)
        grid_w = rng.randint(max(GRID_MIN, legend_w + 1), GRID_MAX)
        play_h = grid_h - band_h - 1
        play_w = grid_w
    else:
        band_w = max(item.shape[1] for _kind, item in items)
        legend_h = sum(item.shape[0] for _kind, item in items) + gap * (len(items) - 1)
        grid_h = rng.randint(max(GRID_MIN, legend_h + 1), GRID_MAX)
        grid_w = rng.randint(max(GRID_MIN, band_w + 12), GRID_MAX)
        play_h = grid_h
        play_w = grid_w - band_w - 1

    if play_h < 12 or play_w < 12:
        return None

    frames = _place_frames(rng, play_h, play_w)
    if frames is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for frame in frames:
        r0, c0, outer_h, outer_w = frame["outer"]
        border_color = frame["border_color"]
        pattern = frame["pattern"]
        grid[r0, c0 : c0 + outer_w] = border_color
        grid[r0 + outer_h - 1, c0 : c0 + outer_w] = border_color
        grid[r0 : r0 + outer_h, c0] = border_color
        grid[r0 : r0 + outer_h, c0 + outer_w - 1] = border_color
        inner = grid[r0 + 1 : r0 + outer_h - 1, c0 + 1 : c0 + outer_w - 1]
        inner[pattern] = GRAY

    if orientation == "bottom":
        total_w = sum(item.shape[1] for _kind, item in items) + gap * (len(items) - 1)
        start_c = rng.randint(0, grid_w - total_w)
        cursor = start_c
        for _kind, item in items:
            ih, iw = item.shape
            r0 = grid_h - ih
            item_mask = item != BG
            target = grid[r0:, cursor : cursor + iw]
            target[item_mask] = item[item_mask]
            cursor += iw + gap
    else:
        total_h = sum(item.shape[0] for _kind, item in items) + gap * (len(items) - 1)
        start_r = rng.randint(0, grid_h - total_h)
        cursor = start_r
        for _kind, item in items:
            ih, iw = item.shape
            c0 = grid_w - iw
            item_mask = item != BG
            target = grid[cursor : cursor + ih, c0:]
            target[item_mask] = item[item_mask]
            cursor += ih + gap

    mapping = _legend_mapping(grid, BG)
    if set(mapping) != set(SYMMETRY_ORDER):
        return None
    frames_found = _find_frames(grid, BG)
    if len(frames_found) != len(frames):
        return None
    found_types = {_classify_symmetry(grid[r0 + 1 : r1, c0 + 1 : c1] == GRAY) for r0, c0, r1, c1, _ in frames_found}
    if found_types != set(SYMMETRY_ORDER):
        return None
    out = main(grid)
    if np.array_equal(out, grid):
        return None
    for r0, c0, r1, c1, _ in frames_found:
        if np.any(out[r0 + 1 : r1, c0 + 1 : c1] == GRAY):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
