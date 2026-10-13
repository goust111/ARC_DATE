import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_filled_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[dict[str, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if len(coords) < 4:
                continue
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            area = hh * ww
            if hh > 1 and ww > 1 and len(coords) == area:
                rects.append(
                    {
                        "r0": r0,
                        "c0": c0,
                        "r1": r1,
                        "c1": c1,
                        "h": hh,
                        "w": ww,
                        "area": area,
                        "color": color,
                    }
                )

    rects.sort(key=lambda item: (-item["area"], item["r0"], item["c0"], item["color"]))
    return rects


def _rect_mask(shape: tuple[int, int], rects: list[dict[str, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rects:
        mask[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = True
    return mask


def _infer_minimal_tile(grid: np.ndarray, visible_mask: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    visible = np.asarray(visible_mask, dtype=bool)
    h, w = g.shape
    points = np.argwhere(visible)
    assert len(points) > 0, "expected at least one visible cell"

    best_key = None
    best_tile = None
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            key = (ph * pw, ph, pw)
            if best_key is not None and key >= best_key:
                continue

            tile = np.full((ph, pw), -1, dtype=int)
            seen = np.zeros((ph, pw), dtype=bool)
            ok = True
            for r, c in points:
                rr = int(r % ph)
                cc = int(c % pw)
                color = int(g[r, c])
                if seen[rr, cc]:
                    if int(tile[rr, cc]) != color:
                        ok = False
                        break
                else:
                    tile[rr, cc] = color
                    seen[rr, cc] = True

            if ok and np.all(seen):
                best_key = key
                best_tile = tile

    assert best_tile is not None, "failed to infer a fully observed periodic tile"
    return best_tile


def _tile_grid(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    th, tw = tile.shape
    h, w = shape
    reps_r = (h + th - 1) // th
    reps_c = (w + tw - 1) // tw
    return np.tile(tile, (reps_r, reps_c))[:h, :w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the background pattern under the largest solid rectangular occluder.

    Input structure:
    - A small multicolor tile repeats across the whole grid.
    - Several non-overlapping monochromatic solid rectangles cover parts of that
      periodic background.

    Rule:
    - Detect every 4-connected monochromatic object whose bounding box is a
      fully filled rectangle with both dimensions greater than one.
    - Treat those rectangles as occlusion masks.
    - Infer the smallest periodic tile consistent with the remaining visible
      pixels.
    - Repaint only the largest rectangle with the recovered background pattern.
    """
    g = np.asarray(grid, dtype=int)
    rects = _find_filled_rectangles(g)
    if not rects:
        return g.copy()

    hidden = _rect_mask(g.shape, rects)
    tile = _infer_minimal_tile(g, ~hidden)

    target = rects[0]
    out = g.copy()
    th, tw = tile.shape
    for r in range(target["r0"], target["r1"] + 1):
        for c in range(target["c0"], target["c1"] + 1):
            out[r, c] = int(tile[r % th, c % tw])
    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 24
TILE_DIM_MIN = 2
TILE_DIM_MAX = 5
TILE_COLORS_MIN = 3
TILE_COLORS_MAX = 4
RECT_COUNTS = (2, 3, 4)
RECT_GAP = 1
MIN_VISIBLE_PER_RESIDUE = 2
MAX_TILE_ATTEMPTS = 100
MAX_RECT_ATTEMPTS = 120
MAX_GENERATE_ATTEMPTS = 300


def _random_tile(rng: np.random.Generator, h: int, w: int, colors: np.ndarray) -> np.ndarray:
    mode = int(rng.integers(0, 3))
    k = len(colors)
    if mode == 0:
        tile = colors[rng.integers(0, k, size=(h, w))]
    elif mode == 1:
        base = rng.integers(0, k, size=w)
        shifts = rng.integers(0, w, size=h)
        rows = [np.roll(base, int(shift)) for shift in shifts]
        tile = colors[np.asarray(rows, dtype=int)]
    else:
        a = int(rng.integers(1, k + 1))
        b = int(rng.integers(1, k + 1))
        offset = int(rng.integers(0, k))
        tile = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                idx = (a * r + b * c + offset + (r * c % max(2, k))) % k
                tile[r, c] = int(colors[idx])

    if int(rng.integers(0, 2)) == 1:
        tile = np.roll(tile, int(rng.integers(0, h)), axis=0)
    if int(rng.integers(0, 2)) == 1:
        tile = np.roll(tile, int(rng.integers(0, w)), axis=1)
    return tile


def _choose_tile(rng: np.random.Generator, grid_shape: tuple[int, int]):
    h, w = grid_shape
    max_th = min(TILE_DIM_MAX, h // 3)
    max_tw = min(TILE_DIM_MAX, w // 3)
    if max_th < TILE_DIM_MIN or max_tw < TILE_DIM_MIN:
        return None

    for _ in range(MAX_TILE_ATTEMPTS):
        th = int(rng.integers(TILE_DIM_MIN, max_th + 1))
        tw = int(rng.integers(TILE_DIM_MIN, max_tw + 1))
        color_count = int(rng.integers(TILE_COLORS_MIN, TILE_COLORS_MAX + 1))
        colors = rng.choice(np.arange(10), size=color_count, replace=False)
        tile = _random_tile(rng, th, tw, colors)

        if len(np.unique(tile)) < 2:
            continue
        if len(np.unique(tile)) < color_count:
            continue
        if _infer_minimal_tile(tile, np.ones(tile.shape, dtype=bool)).shape != tile.shape:
            continue

        background = _tile_grid(tile, grid_shape)
        if _find_filled_rectangles(background):
            continue
        return tile, background, colors
    return None


def _rect_signature(rects: list[dict[str, int]]) -> list[tuple[int, int, int, int, int]]:
    return sorted((r["r0"], r["c0"], r["r1"], r["c1"], r["color"]) for r in rects)


def _can_place_rect(
    occupied: np.ndarray, r0: int, c0: int, rh: int, rw: int, gap: int
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r0 + rh + gap)
    cc1 = min(w, c0 + rw + gap)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _sample_rect_size(
    rng: np.random.Generator, grid_shape: tuple[int, int], is_largest: bool
) -> tuple[int, int]:
    h, w = grid_shape
    mode = int(rng.integers(0, 3))
    if is_largest:
        if mode == 0:
            rh = int(rng.integers(max(3, h // 3), min(h - 1, h // 2 + 2)))
            rw = int(rng.integers(2, max(3, w // 4 + 2)))
        elif mode == 1:
            rh = int(rng.integers(2, max(3, h // 4 + 2)))
            rw = int(rng.integers(max(3, w // 3), min(w - 1, w // 2 + 2)))
        else:
            rh = int(rng.integers(2, max(3, h // 3 + 2)))
            rw = int(rng.integers(2, max(3, w // 3 + 2)))
    else:
        if mode == 0:
            rh = int(rng.integers(2, max(3, h // 4 + 2)))
            rw = int(rng.integers(2, max(3, w // 5 + 2)))
        elif mode == 1:
            rh = int(rng.integers(2, max(3, h // 5 + 2)))
            rw = int(rng.integers(2, max(3, w // 4 + 2)))
        else:
            rh = int(rng.integers(2, max(3, h // 4 + 2)))
            rw = int(rng.integers(2, max(3, w // 4 + 2)))
    rh = min(rh, h - 1)
    rw = min(rw, w - 1)
    return rh, rw


def _place_rectangles(
    rng: np.random.Generator, background: np.ndarray, tile_colors: np.ndarray
):
    h, w = background.shape
    rect_count = int(rng.choice(RECT_COUNTS))
    remaining_colors = np.array([c for c in range(10) if c not in set(map(int, tile_colors))], dtype=int)
    rect_colors = rng.choice(remaining_colors, size=rect_count, replace=False)
    occupied = np.zeros((h, w), dtype=bool)
    rects: list[dict[str, int]] = []
    largest_area = None

    for idx in range(rect_count):
        is_largest = idx == 0
        for _ in range(MAX_RECT_ATTEMPTS):
            rh, rw = _sample_rect_size(rng, background.shape, is_largest=is_largest)
            area = rh * rw
            if is_largest:
                if area < max(10, (h * w) // 12) or area > (h * w) // 3:
                    continue
            else:
                assert largest_area is not None
                if area >= largest_area or area < 4:
                    continue

            r0 = int(rng.integers(0, h - rh + 1))
            c0 = int(rng.integers(0, w - rw + 1))
            if not _can_place_rect(occupied, r0, c0, rh, rw, RECT_GAP):
                continue

            rect = {
                "r0": r0,
                "c0": c0,
                "r1": r0 + rh - 1,
                "c1": c0 + rw - 1,
                "h": rh,
                "w": rw,
                "area": area,
                "color": int(rect_colors[idx]),
            }
            rects.append(rect)
            occupied[r0 : r0 + rh, c0 : c0 + rw] = True
            if is_largest:
                largest_area = area
            break
        else:
            return None

    areas = [rect["area"] for rect in rects]
    if areas.count(max(areas)) != 1:
        return None
    return rects


def _visible_residue_counts(shape: tuple[int, int], rects: list[dict[str, int]], tile_shape: tuple[int, int]) -> np.ndarray:
    hidden = _rect_mask(shape, rects)
    th, tw = tile_shape
    counts = np.zeros((th, tw), dtype=int)
    for r, c in np.argwhere(~hidden):
        counts[int(r % th), int(c % tw)] += 1
    return counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))

    tile_result = _choose_tile(rng, (h, w))
    if tile_result is None:
        return None
    tile, background, tile_colors = tile_result

    rects = _place_rectangles(rng, background, tile_colors)
    if rects is None:
        return None

    residue_counts = _visible_residue_counts(background.shape, rects, tile.shape)
    if int(residue_counts.min()) < MIN_VISIBLE_PER_RESIDUE:
        return None

    grid = background.copy()
    for rect in rects:
        grid[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]

    detected = _find_filled_rectangles(grid)
    if _rect_signature(detected) != _rect_signature(rects):
        return None

    expected = grid.copy()
    largest = max(rects, key=lambda item: item["area"])
    expected[largest["r0"] : largest["r1"] + 1, largest["c0"] : largest["c1"] + 1] = background[
        largest["r0"] : largest["r1"] + 1, largest["c0"] : largest["c1"] + 1
    ]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {MAX_GENERATE_ATTEMPTS} attempts")
