from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _count_regions(mask: np.ndarray) -> int:
    return len(_components(mask))


def _read_legend(grid: np.ndarray, bg: int) -> tuple[int, list[np.ndarray]]:
    rows = np.flatnonzero((grid != bg).any(axis=1))
    assert rows.size >= 2, "expected a 2-row legend band near the top"
    top = int(rows[0])
    assert top + 1 < grid.shape[0]
    band = np.asarray(grid[top : top + 2], dtype=int)

    patterns: list[np.ndarray] = []
    c = 0
    while c < band.shape[1]:
        if np.all(band[:, c] == bg):
            c += 1
            continue
        assert c + 1 < band.shape[1], "legend blocks must be 2 columns wide"
        block = np.asarray(band[:, c : c + 2], dtype=int)
        assert block.shape == (2, 2)
        assert np.all(block != bg), "legend pattern cells must be non-background"
        patterns.append(block.copy())
        c += 2
        while c < band.shape[1] and np.all(band[:, c] == bg):
            c += 1

    assert patterns, "expected at least one legend block"
    return top, patterns


def _find_objects(grid: np.ndarray, bg: int, min_row: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[np.ndarray] = []
    for r in range(min_row, h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if nr < min_row:
                        continue
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            objects.append(np.asarray(coords, dtype=int))
    return objects


def _indent_count(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> int:
    r0, c0, r1, c1 = bbox
    crop = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    return _count_regions(crop == bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each object's bbox background-region count as a legend index, then tile
    the matching 2x2 pattern into that bbox while preserving the object itself.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend_top, patterns = _read_legend(g, bg)
    assert len(patterns) >= 5, "expected five legend patterns for counts 0..4"

    out = g.copy()
    for coords in _find_objects(g, bg, legend_top + 2):
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        holes = _indent_count(g, (r0, c0, r1, c1), bg)
        assert 0 <= holes < len(patterns)
        pattern = patterns[holes]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if int(g[rr, cc]) == bg:
                    out[rr, cc] = int(pattern[(rr - r0) % 2, (cc - c0) % 2])
    return out


# --------------- generator ---------------

GRID_MIN = 20
GRID_MAX = 30
LEGEND_TOP_MARGIN_MAX = 1
LEGEND_SIDE_MARGIN_MAX = 2
OBJECT_GAP_ROWS = (2, 4)
N_OBJECTS = (5, 7)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 400
SHAPE_RETRIES = 300

TEMPLATES = (
    np.array([[0, 1], [1, 0]], dtype=int),
    np.array([[0, 1], [0, 1]], dtype=int),
    np.array([[0, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [2, 0]], dtype=int),
    np.array([[0, 1], [2, 3]], dtype=int),
)

HOLE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
)


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(shape, dtype=int)
    if random.random() < 0.5:
        arr = arr.copy()
        arr[:, 1] *= -1
    for _ in range(random.randint(0, 3)):
        arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    pts = sorted(set((int(r), int(c)) for r, c in arr))
    return tuple(pts)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _random_hole_shape() -> tuple[tuple[int, int], ...]:
    return _transform_shape(random.choice(HOLE_SHAPES))


def _sample_pattern_block(accent_colors: list[int], template: np.ndarray) -> np.ndarray:
    needed = int(template.max()) + 1
    chosen = random.sample(accent_colors, needed)
    block = np.zeros((2, 2), dtype=int)
    for idx, color in enumerate(chosen):
        block[template == idx] = int(color)
    return block


def _sample_patterns(bg: int) -> list[np.ndarray]:
    colors = [c for c in range(10) if c != bg]
    accent_colors = random.sample(colors, 4)
    patterns: list[np.ndarray] = []
    used: set[bytes] = set()
    for template in TEMPLATES:
        for _ in range(50):
            block = _sample_pattern_block(accent_colors, template)
            key = block.tobytes()
            if key not in used:
                used.add(key)
                patterns.append(block)
                break
        else:
            raise RuntimeError("failed to build distinct legend patterns")
    return patterns


def _build_legend(patterns: list[np.ndarray], bg: int) -> np.ndarray:
    width = 2 * len(patterns) + (len(patterns) - 1)
    legend = np.full((2, width), bg, dtype=int)
    c = 0
    for i, pattern in enumerate(patterns):
        legend[:, c : c + 2] = pattern
        c += 2
        if i + 1 < len(patterns):
            c += 1
    return legend


def _sample_object_mask(k: int) -> np.ndarray:
    if k == 0:
        h = random.randint(3, 7)
        w = random.randint(3, 7)
        return np.ones((h, w), dtype=bool)

    for _ in range(SHAPE_RETRIES):
        low = 4 + (1 if k >= 3 else 0)
        h = random.randint(low, 8)
        w = random.randint(low, 8)
        hole_cells: set[tuple[int, int]] = set()
        require_boundary = random.random() < 0.7
        ok = True
        for idx in range(k):
            want_boundary = (idx == 0 and require_boundary) or (random.random() < 0.35)
            placed = False
            for _ in range(120):
                shape = _random_hole_shape()
                sh, sw = _shape_dims(shape)
                if sh > h or sw > w:
                    continue
                r0 = random.randint(0, h - sh)
                c0 = random.randint(0, w - sw)
                cells = {(r0 + dr, c0 + dc) for dr, dc in shape}
                if want_boundary and not any(
                    r in {0, h - 1} or c in {0, w - 1} for r, c in cells
                ):
                    continue
                if cells & hole_cells:
                    continue
                touches_other = False
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c, h, w):
                        if (nr, nc) in hole_cells:
                            touches_other = True
                            break
                    if touches_other:
                        break
                if touches_other:
                    continue
                hole_cells |= cells
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        mask = np.ones((h, w), dtype=bool)
        for r, c in hole_cells:
            mask[r, c] = False
        if not mask.any():
            continue
        cropped = _crop_mask(mask)
        if _count_regions(cropped) != 1:
            continue
        if _count_regions(~cropped) != k:
            continue
        if int(cropped.sum()) < k + 5:
            continue
        if cropped.shape[0] < 3 or cropped.shape[1] < 3:
            continue
        if random.random() < 0.5:
            cropped = np.fliplr(cropped)
        if random.random() < 0.5:
            cropped = np.flipud(cropped)
        return cropped
    raise RuntimeError(f"failed to generate an object with {k} indent regions")


def _render_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, int(color), int(bg))


def _place_objects(
    grid: np.ndarray,
    expected: np.ndarray,
    sprites: list[tuple[np.ndarray, np.ndarray, int]],
    patterns: list[np.ndarray],
    bg: int,
    min_row: int,
) -> bool:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    for sprite, mask, k in sprites:
        sh, sw = sprite.shape
        placed = False
        row_lo = max(min_row, 1)
        row_hi = h - sh - 1
        col_lo = 1
        col_hi = w - sw - 1
        if row_lo > row_hi or col_lo > col_hi:
            return False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(row_lo, row_hi)
            c0 = random.randint(col_lo, col_hi)
            if blocked[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            grid[r0 : r0 + sh, c0 : c0 + sw][mask] = sprite[mask]
            blocked[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True

            pattern = patterns[k]
            bbox_view = expected[r0 : r0 + sh, c0 : c0 + sw]
            bbox_view[mask] = sprite[mask]
            for rr in range(sh):
                for cc in range(sw):
                    if not mask[rr, cc]:
                        bbox_view[rr, cc] = int(pattern[rr % 2, cc % 2])
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    patterns = _sample_patterns(bg)
    legend = _build_legend(patterns, bg)

    available = [c for c in range(10) if c != bg]
    object_palette = random.sample(available, random.randint(2, 4))

    n_objects = random.randint(*N_OBJECTS)
    ks = list(range(5))
    while len(ks) < n_objects:
        ks.append(random.randint(0, 4))
    random.shuffle(ks)

    sprite_specs: list[tuple[np.ndarray, np.ndarray, int]] = []
    max_h = 0
    max_w = 0
    for k in ks:
        mask = _sample_object_mask(k)
        color = random.choice(object_palette)
        sprite = _render_sprite(mask, color, bg)
        sprite_specs.append((sprite, mask, k))
        max_h = max(max_h, int(sprite.shape[0]))
        max_w = max(max_w, int(sprite.shape[1]))

    gap_rows = random.randint(*OBJECT_GAP_ROWS)
    legend_top = random.randint(0, LEGEND_TOP_MARGIN_MAX)
    min_row = legend_top + 2 + gap_rows
    min_h = min_row + max_h + 2
    min_w = max(
        legend.shape[1] + 2 * LEGEND_SIDE_MARGIN_MAX,
        max_w + 2,
    )

    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    grid_h = random.randint(max(GRID_MIN, min_h), GRID_MAX)
    grid_w = random.randint(max(GRID_MIN, min_w), GRID_MAX)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    expected = grid.copy()

    legend_left = random.randint(0, grid_w - legend.shape[1])
    grid[legend_top : legend_top + 2, legend_left : legend_left + legend.shape[1]] = legend
    expected[legend_top : legend_top + 2, legend_left : legend_left + legend.shape[1]] = legend

    sprite_specs.sort(key=lambda item: int(item[0].shape[0] * item[0].shape[1]), reverse=True)
    if not _place_objects(grid, expected, sprite_specs, patterns, bg, min_row):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
