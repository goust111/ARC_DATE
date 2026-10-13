import random

import numpy as np


BG = 7
FRAME = 1
MARKER_COLORS = [2, 3, 4, 5, 6, 8, 9]
SYMMETRY_TYPES = ("horizontal", "vertical", "both")
GEN_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _is_rect_frame(component: list[tuple[int, int]]) -> bool:
    rows = [r for r, _c in component]
    cols = [c for _r, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in component:
        local[r - r0, c - c0] = True
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return bool(np.array_equal(local, ring))


def _find_frames(grid: np.ndarray, bg: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for component in _components(grid == color):
            if not _is_rect_frame(component):
                continue
            rows = [r for r, _c in component]
            cols = [c for _r, c in component]
            frames.append((min(rows), max(rows), min(cols), max(cols)))
    frames.sort()
    return frames


def _occupied_bbox(inner: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    pts = np.argwhere(inner != bg)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(r1), int(c0), int(c1)


def _repeat_rows(strip: np.ndarray, out_h: int, anchor_r: int) -> np.ndarray:
    row_idx = (np.arange(out_h) - int(anchor_r)) % strip.shape[0]
    return strip[row_idx, :]


def _repeat_cols(strip: np.ndarray, out_w: int, anchor_c: int) -> np.ndarray:
    col_idx = (np.arange(out_w) - int(anchor_c)) % strip.shape[1]
    return strip[:, col_idx]


def _repeat_tile(tile: np.ndarray, out_shape: tuple[int, int], anchor_r: int, anchor_c: int) -> np.ndarray:
    out_h, out_w = out_shape
    row_idx = (np.arange(out_h) - int(anchor_r)) % tile.shape[0]
    col_idx = (np.arange(out_w) - int(anchor_c)) % tile.shape[1]
    return tile[np.ix_(row_idx, col_idx)]


def _expand_inner(inner: np.ndarray, bg: int) -> np.ndarray:
    horizontal = bool(np.array_equal(inner, inner[::-1, :]))
    vertical = bool(np.array_equal(inner, inner[:, ::-1]))
    assert horizontal or vertical
    r0, r1, c0, c1 = _occupied_bbox(inner, bg)
    if horizontal and vertical:
        tile = inner[r0 : r1 + 1, c0 : c1 + 1]
        return _repeat_tile(tile, inner.shape, r0, c0)
    if horizontal:
        strip = inner[r0 : r1 + 1, :]
        return _repeat_rows(strip, inner.shape[0], r0)
    strip = inner[:, c0 : c1 + 1]
    return _repeat_cols(strip, inner.shape[1], c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the markers inside each rectangular frame according to their axis symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames = _find_frames(g, bg)
    assert frames
    out = g.copy()
    for r0, r1, c0, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _expand_inner(inner, bg)
    return out


def _palette(rng: random.Random, n: int) -> list[int]:
    return rng.sample(MARKER_COLORS, n)


def _ensure_not_all_bg(arr: np.ndarray, rng: random.Random, palette: list[int]) -> None:
    if np.any(arr != BG):
        return
    arr[rng.randrange(arr.shape[0]), rng.randrange(arr.shape[1])] = rng.choice(palette)


def _ensure_row_non_bg(arr: np.ndarray, row: int, rng: random.Random, palette: list[int]) -> None:
    if np.any(arr[row, :] != BG):
        return
    arr[row, rng.randrange(arr.shape[1])] = rng.choice(palette)


def _ensure_col_non_bg(arr: np.ndarray, col: int, rng: random.Random, palette: list[int]) -> None:
    if np.any(arr[:, col] != BG):
        return
    arr[rng.randrange(arr.shape[0]), col] = rng.choice(palette)


def _build_horizontal_strip(rng: random.Random, height: int, width: int) -> np.ndarray:
    for _ in range(200):
        palette = _palette(rng, rng.randint(2, 4))
        strip = np.full((height, width), BG, dtype=int)
        top = (height + 1) // 2
        density = rng.uniform(0.25, 0.45)
        for r in range(top):
            for c in range(width):
                if rng.random() < density:
                    strip[r, c] = rng.choice(palette)
        _ensure_not_all_bg(strip[:top, :], rng, palette)
        _ensure_row_non_bg(strip, 0, rng, palette)
        for r in range(height // 2):
            strip[height - 1 - r, :] = strip[r, :]
        if not np.any(strip[0, :] != BG):
            continue
        if np.array_equal(strip, strip[:, ::-1]):
            continue
        return strip
    raise RuntimeError("failed to build horizontal-only strip")


def _build_vertical_strip(rng: random.Random, height: int, width: int) -> np.ndarray:
    for _ in range(200):
        palette = _palette(rng, rng.randint(2, 4))
        strip = np.full((height, width), BG, dtype=int)
        left = (width + 1) // 2
        density = rng.uniform(0.25, 0.45)
        for r in range(height):
            for c in range(left):
                if rng.random() < density:
                    strip[r, c] = rng.choice(palette)
        _ensure_not_all_bg(strip[:, :left], rng, palette)
        _ensure_col_non_bg(strip, 0, rng, palette)
        for c in range(width // 2):
            strip[:, width - 1 - c] = strip[:, c]
        if not np.any(strip[:, 0] != BG):
            continue
        if np.array_equal(strip, strip[::-1, :]):
            continue
        return strip
    raise RuntimeError("failed to build vertical-only strip")


def _build_biaxial_tile(rng: random.Random, height: int, width: int) -> np.ndarray:
    for _ in range(200):
        palette = _palette(rng, rng.randint(2, 4))
        tile = np.full((height, width), BG, dtype=int)
        density = rng.uniform(0.25, 0.45)
        for r in range((height + 1) // 2):
            for c in range((width + 1) // 2):
                color = BG
                if rng.random() < density:
                    color = rng.choice(palette)
                for rr, cc in {
                    (r, c),
                    (height - 1 - r, c),
                    (r, width - 1 - c),
                    (height - 1 - r, width - 1 - c),
                }:
                    tile[rr, cc] = color
        _ensure_not_all_bg(tile, rng, palette)
        _ensure_row_non_bg(tile, 0, rng, palette)
        _ensure_row_non_bg(tile, height - 1, rng, palette)
        _ensure_col_non_bg(tile, 0, rng, palette)
        _ensure_col_non_bg(tile, width - 1, rng, palette)
        if np.array_equal(tile, tile[::-1, :]) and np.array_equal(tile, tile[:, ::-1]):
            return tile
    raise RuntimeError("failed to build biaxial tile")


def _same_parity_sizes(total: int, min_size: int, max_size: int) -> list[int]:
    return [size for size in range(min_size, max_size + 1) if (total - size) % 2 == 0]


def _make_inner(rng: random.Random, kind: str, height: int, width: int) -> np.ndarray:
    inner = np.full((height, width), BG, dtype=int)
    if kind == "horizontal":
        band_choices = _same_parity_sizes(height, 2, height - 2)
        band_h = rng.choice(band_choices)
        band = _build_horizontal_strip(rng, band_h, width)
        r0 = (height - band_h) // 2
        inner[r0 : r0 + band_h, :] = band
        return inner
    if kind == "vertical":
        band_choices = _same_parity_sizes(width, 2, width - 2)
        band_w = rng.choice(band_choices)
        band = _build_vertical_strip(rng, height, band_w)
        c0 = (width - band_w) // 2
        inner[:, c0 : c0 + band_w] = band
        return inner
    tile_h = rng.choice(_same_parity_sizes(height, 2, height - 2))
    tile_w = rng.choice(_same_parity_sizes(width, 2, width - 2))
    tile = _build_biaxial_tile(rng, tile_h, tile_w)
    r0 = (height - tile_h) // 2
    c0 = (width - tile_w) // 2
    inner[r0 : r0 + tile_h, c0 : c0 + tile_w] = tile
    return inner


def _symmetry_kind(inner: np.ndarray) -> str:
    horizontal = bool(np.array_equal(inner, inner[::-1, :]))
    vertical = bool(np.array_equal(inner, inner[:, ::-1]))
    if horizontal and vertical:
        return "both"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return "none"


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    return not bool(np.any(occupied[rr0:rr1, cc0:cc1]))


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner: np.ndarray) -> None:
    h, w = inner.shape
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0 + 1 : r1, c0 + 1 : c1] = inner


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(20, 31)
    width = rng.randint(20, 31)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    max_frames = 4 if height * width >= 500 else 3
    frame_count = rng.randint(1, max_frames)
    kinds_used: list[str] = []

    for _ in range(frame_count):
        kind = rng.choice(SYMMETRY_TYPES)
        placed = False
        for _ in range(120):
            inner_h = rng.randint(6, 12)
            inner_w = rng.randint(6, 12)
            if inner_h + 2 > height or inner_w + 2 > width:
                continue
            inner = _make_inner(rng, kind, inner_h, inner_w)
            if _symmetry_kind(inner) != kind:
                continue
            frame_h = inner_h + 2
            frame_w = inner_w + 2
            r0 = rng.randint(0, height - frame_h)
            c0 = rng.randint(0, width - frame_w)
            if not _can_place(occupied, r0, c0, frame_h, frame_w, pad=1):
                continue
            _paint_frame(grid, r0, c0, inner)
            occupied[r0 : r0 + frame_h, c0 : c0 + frame_w] = True
            kinds_used.append(kind)
            placed = True
            break
        if not placed:
            return None

    frames = _find_frames(grid, BG)
    if len(frames) != frame_count:
        return None

    found_kinds = [_symmetry_kind(grid[r0 + 1 : r1, c0 + 1 : c1]) for r0, r1, c0, c1 in frames]
    if sorted(found_kinds) != sorted(kinds_used):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
