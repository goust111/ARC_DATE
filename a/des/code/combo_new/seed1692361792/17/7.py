import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_GREY = 5
_THEME_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_RETRIES = 400
_DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

_PATTERN_BANK = {
    3: (
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=int),
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int),
        np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int),
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
    ),
    4: (
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=int),
        np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=int),
        np.array([[1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]], dtype=int),
    ),
}

_AREA_BUCKETS = (
    (2, 3, 4),
    (5, 6, 8),
    (9, 10, 12),
    (15, 16, 18),
)

_AREA_FACTORS = {
    2: ((1, 2),),
    3: ((1, 3),),
    4: ((1, 4), (2, 2)),
    5: ((1, 5),),
    6: ((1, 6), (2, 3)),
    8: ((2, 4),),
    9: ((3, 3),),
    10: ((2, 5),),
    12: ((2, 6), (3, 4)),
    15: ((3, 5),),
    16: ((4, 4),),
    18: ((3, 6),),
}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each central rectangle by the corner sample pattern assigned to its area rank.

    Input structure:
    - Black background surrounds four square corner samples and several central foreground objects.
    - Each corner sample is a solid small square made only of grey and one unique theme color.
    - Central objects are separated monochrome 4-connected filled rectangles colored with the theme colors.

    Rule:
    - Detect the four corner sample squares and infer grey as the only non-background color shared by all four.
    - Sort the corner samples by their count of non-grey pixels from sparse to dense.
    - Find every central rectangular object and sort the distinct object areas from small to large into four ranks.
    - Replace each object by tiling the equally ranked corner sample across its bounding rectangle, starting at the object's top-left cell.
    - Leave the corner sample regions unchanged.

    Key insight:
    - The corner squares form a density-ordered legend, and object colors are irrelevant once area rank is aligned with that legend rank.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    corners, grey = _extract_corner_templates(g, bg)

    ordered_corners = sorted(corners, key=lambda item: item["theme_count"])
    assert len({item["theme_count"] for item in ordered_corners}) == 4

    work = g.copy()
    theme_colors = {item["theme"] for item in corners}
    for item in corners:
        r0, r1, c0, c1 = item["bbox"]
        work[r0:r1, c0:c1] = bg

    objects = _find_rectangles(work, bg)
    assert objects
    for obj in objects:
        assert obj["color"] in theme_colors

    area_values = sorted({obj["area"] for obj in objects})
    assert len(area_values) == 4
    area_to_rank = {area: rank for rank, area in enumerate(area_values)}

    out = g.copy()
    for obj in objects:
        rank = area_to_rank[obj["area"]]
        pattern = ordered_corners[rank]["pattern"]
        tiled = _tile_pattern(pattern, obj["h"], obj["w"])
        out[obj["r0"] : obj["r1"], obj["c0"] : obj["c1"]] = tiled

    # `grey` is computed for validation; keep a cheap assertion so corner legend detection is not dead code.
    assert grey in out
    return out


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel())
    return int(np.argmax(counts))


def _edge_run(values: np.ndarray, bg: int, reverse: bool = False) -> int:
    line = values[::-1] if reverse else values
    run = 0
    for value in line:
        if int(value) == bg:
            break
        run += 1
    return run


def _extract_corner_templates(grid: np.ndarray, bg: int):
    h, w = grid.shape
    corner_specs = []
    corner_defs = (
        ("tl", min(_edge_run(grid[0], bg), _edge_run(grid[:, 0], bg)), (0, 0)),
        ("tr", min(_edge_run(grid[0], bg, True), _edge_run(grid[:, -1], bg)), (0, w)),
        ("bl", min(_edge_run(grid[-1], bg), _edge_run(grid[:, 0], bg, True)), (h, 0)),
        ("br", min(_edge_run(grid[-1], bg, True), _edge_run(grid[:, -1], bg, True)), (h, w)),
    )

    for name, size, anchor in corner_defs:
        assert size >= 2
        if name == "tl":
            r0, r1, c0, c1 = 0, size, 0, size
        elif name == "tr":
            r0, r1, c0, c1 = 0, size, w - size, w
        elif name == "bl":
            r0, r1, c0, c1 = h - size, h, 0, size
        else:
            r0, r1, c0, c1 = h - size, h, w - size, w
        patch = grid[r0:r1, c0:c1]
        assert patch.shape == (size, size)
        assert np.all(patch != bg)
        colors = set(map(int, np.unique(patch)))
        assert len(colors) == 2
        corner_specs.append(
            {
                "name": name,
                "anchor": anchor,
                "bbox": (r0, r1, c0, c1),
                "pattern": patch.copy(),
                "colors": colors,
            }
        )

    color_sets = [item["colors"] for item in corner_specs]
    grey_candidates = set.intersection(*color_sets)
    assert len(grey_candidates) == 1
    grey = next(iter(grey_candidates))

    for item in corner_specs:
        theme_colors = item["colors"] - {grey}
        assert len(theme_colors) == 1
        item["theme"] = next(iter(theme_colors))
        item["theme_count"] = int(np.count_nonzero(item["pattern"] != grey))

    return corner_specs, grey


def _find_rectangles(grid: np.ndarray, bg: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows) + 1
            c0, c1 = min(cols), max(cols) + 1
            rect = grid[r0:r1, c0:c1]
            assert np.all(rect == color)
            area = len(coords)
            assert area == rect.size
            objects.append(
                {
                    "color": color,
                    "r0": r0,
                    "r1": r1,
                    "c0": c0,
                    "c1": c1,
                    "h": r1 - r0,
                    "w": c1 - c0,
                    "area": area,
                }
            )

    return objects


def _tile_pattern(pattern: np.ndarray, height: int, width: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps = ((height + ph - 1) // ph, (width + pw - 1) // pw)
    tiled = np.tile(pattern, reps)
    return tiled[:height, :width]


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, int(_RNG.integers(0, 4)))
    if float(_RNG.random()) < 0.5:
        out = np.fliplr(out)
    if float(_RNG.random()) < 0.5:
        out = np.flipud(out)
    return out.copy()


def _corner_bbox(name: str, size: int, height: int, width: int):
    if name == "tl":
        return 0, size, 0, size
    if name == "tr":
        return 0, size, width - size, width
    if name == "bl":
        return height - size, height, 0, size
    return height - size, height, width - size, width


def _mark_reserved(mask: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c1 + 1)
    mask[rr0:rr1, cc0:cc1] = True


def _build_pattern(mask: np.ndarray, theme: int) -> np.ndarray:
    return np.where(mask.astype(bool), theme, _GREY).astype(int)


def _sample_object_specs():
    areas = [int(_RNG.choice(bucket)) for bucket in _AREA_BUCKETS]
    counts = [int(_RNG.integers(1, 3)) for _ in range(4)]
    if float(_RNG.random()) < 0.4:
        counts[int(_RNG.integers(0, 4))] += 1

    specs = []
    for area, count in zip(areas, counts):
        for _ in range(count):
            h, w = _AREA_FACTORS[area][int(_RNG.integers(0, len(_AREA_FACTORS[area])))]
            if float(_RNG.random()) < 0.5:
                h, w = w, h
            specs.append({"area": area, "h": h, "w": w})
    specs.sort(key=lambda item: (-item["area"], -max(item["h"], item["w"])))
    return areas, specs


def _place_objects(grid: np.ndarray, occupied: np.ndarray, size: int, theme_colors: list[int], specs) -> bool:
    h, w = grid.shape
    row_lo = size + 1
    row_hi = h - size - 1
    col_lo = size + 1
    col_hi = w - size - 1
    if row_hi <= row_lo or col_hi <= col_lo:
        return False

    for spec in specs:
        obj_h, obj_w = spec["h"], spec["w"]
        if row_hi - row_lo < obj_h or col_hi - col_lo < obj_w:
            return False

        placed = False
        for _ in range(200):
            r0 = int(_RNG.integers(row_lo, row_hi - obj_h + 1))
            c0 = int(_RNG.integers(col_lo, col_hi - obj_w + 1))
            if occupied[r0:r0 + obj_h, c0:c0 + obj_w].any():
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(h, r0 + obj_h + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(w, c0 + obj_w + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            color = int(_RNG.choice(theme_colors))
            grid[r0:r0 + obj_h, c0:c0 + obj_w] = color
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return False

    return True


def _gen_attempt():
    size = int(_RNG.choice((3, 4)))
    height = int(_RNG.integers(20, 31))
    width = int(_RNG.integers(20, 31))
    grid = np.full((height, width), _BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    theme_colors = list(map(int, _RNG.choice(_THEME_POOL, size=4, replace=False)))
    corner_names = ["tl", "tr", "bl", "br"]
    pattern_defs = []
    for theme, mask in zip(theme_colors, _PATTERN_BANK[size]):
        pattern_defs.append(_build_pattern(_random_transform(mask), theme))
    _RNG.shuffle(pattern_defs)

    for name, pattern in zip(corner_names, pattern_defs):
        r0, r1, c0, c1 = _corner_bbox(name, size, height, width)
        grid[r0:r1, c0:c1] = pattern
        _mark_reserved(occupied, r0, r1, c0, c1)

    _areas, specs = _sample_object_specs()
    if not _place_objects(grid, occupied, size, theme_colors, specs):
        return None

    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != _BG or int(counts[_BG]) <= int(np.max(counts[1:])):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {_MAX_RETRIES} attempts")
