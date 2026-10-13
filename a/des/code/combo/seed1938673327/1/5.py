from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
RECOLOR_SEQUENCE = (2, 3, 4, 6, 7)  # red, green, yellow, magenta, orange
GEN_RETRIES = 400
TILE_HEIGHT = len(RECOLOR_SEQUENCE)
TILE_WIDTH_RANGE = (4, 7)
TILE_ROW_RANGE = (3, 5)
TILE_COL_RANGE = (3, 6)
TILE_COLORS = (1, 8, 9)
OBJECT_INPUT_COLORS = (1, 5, 8, 9)
PATTERN_FILL_RANGE = (0.30, 0.46)
PATTERN_RETRIES = 300
OBJECT_SIZE_RANGE = (1, 3)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    return len(_components(mask.astype(bool))) == 1


def _fundamental_period(mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    for ph in _divisors(h):
        for pw in _divisors(w):
            prototype = mask[:ph, :pw]
            repeated = np.tile(prototype, (h // ph, w // pw))
            if np.array_equal(repeated, mask):
                return int(ph), int(pw)
    return h, w


def _infer_period(grid: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(c) for c in np.unique(g) if int(c) != bg]
    assert colors, "expected non-background content"

    tile_color = max(colors, key=lambda color: int(np.count_nonzero(g == color)))
    tile_mask = g == tile_color
    h, w = g.shape

    best: tuple[tuple[int, int, int, int], int, int, np.ndarray] | None = None
    for ph in _divisors(h):
        if h // ph < 2:
            continue
        for pw in _divisors(w):
            if w // pw < 2:
                continue

            base = np.zeros((ph, pw), dtype=bool)
            hole_cost = 0
            for r in range(ph):
                for c in range(pw):
                    values = tile_mask[r::ph, c::pw]
                    true_count = int(values.sum())
                    if true_count == 0:
                        continue
                    base[r, c] = True
                    hole_cost += int(values.size) - true_count

            if not np.any(base) or np.all(base):
                continue

            valid = True
            for r in range(h):
                for c in range(w):
                    if base[r % ph, c % pw] and g[r, c] not in (bg, tile_color):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue

            score = (ph * pw + 4 * hole_cost, ph * pw, hole_cost, ph + pw)
            candidate = (score, ph, pw, base)
            if best is None or candidate[0] < best[0]:
                best = candidate

    assert best is not None, "failed to infer a repeating tile"
    _, tile_h, tile_w, base = best
    return int(tile_color), int(tile_h), int(tile_w), base.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect the hidden periodic tiling, read hole rows inside modified tiles,
    and recolor the non-tiling object inside each holed tile accordingly.
    """
    g = np.asarray(grid, dtype=int)
    tile_color, tile_h, tile_w, base = _infer_period(g)
    bg = _most_common_color(g)
    h, w = g.shape
    assert h % tile_h == 0 and w % tile_w == 0

    object_mask = (g != bg) & (g != tile_color)
    objects_by_tile: dict[tuple[int, int], list[np.ndarray]] = {}
    for coords in _components(object_mask):
        r0, c0, r1, c1 = _bbox(coords)
        tile_r = r0 // tile_h
        tile_c = c0 // tile_w
        assert r1 // tile_h == tile_r and c1 // tile_w == tile_c, "objects must stay inside one tile"
        objects_by_tile.setdefault((tile_r, tile_c), []).append(coords)

    out = g.copy()
    for tile_r in range(h // tile_h):
        for tile_c in range(w // tile_w):
            r0 = tile_r * tile_h
            c0 = tile_c * tile_w
            hole_rows = set()
            for lr, lc in np.argwhere(base):
                value = int(g[r0 + int(lr), c0 + int(lc)])
                if value == bg:
                    hole_rows.add(int(lr))
                else:
                    assert value == tile_color, "template cells may only disappear into background"

            if not hole_rows:
                continue

            assert len(hole_rows) == 1, "hole row must be unambiguous"
            hole_row = next(iter(hole_rows))
            assert 0 <= hole_row < len(RECOLOR_SEQUENCE)
            comps = objects_by_tile.get((tile_r, tile_c), [])
            assert len(comps) == 1, "each modified tile must contain exactly one object"
            coords = comps[0]
            out[coords[:, 0], coords[:, 1]] = RECOLOR_SEQUENCE[hole_row]

    return out


def _cells_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    remaining = set(cells)
    start = next(iter(remaining))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in remaining and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(remaining)


def _object_candidates(base: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = base.shape
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if not base[r, c]]
    candidates: list[tuple[tuple[int, int], ...]] = []
    for size in range(OBJECT_SIZE_RANGE[0], OBJECT_SIZE_RANGE[1] + 1):
        for subset in itertools.combinations(cells, size):
            if _cells_connected(subset):
                candidates.append(tuple(sorted(subset)))
    return candidates


def _sample_tile_pattern(tile_w: int) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]], dict[int, list[int]]] | None:
    min_fill = int(np.ceil(PATTERN_FILL_RANGE[0] * TILE_HEIGHT * tile_w))
    max_fill = int(np.floor(PATTERN_FILL_RANGE[1] * TILE_HEIGHT * tile_w))

    for _ in range(PATTERN_RETRIES):
        base = np.zeros((TILE_HEIGHT, tile_w), dtype=bool)
        spine = random.randint(1, tile_w - 2)
        for r in range(TILE_HEIGHT):
            left = random.randint(0, spine)
            right = random.randint(0, tile_w - 1 - spine)
            if left == spine and right == tile_w - 1 - spine:
                if left > 0 and (right == 0 or random.random() < 0.5):
                    left -= 1
                elif right > 0:
                    right -= 1
            base[r, spine - left : spine + right + 1] = True

        area = int(base.sum())
        if area < min_fill or area > max_fill:
            continue
        if not _is_connected(base):
            continue
        if _fundamental_period(base) != base.shape:
            continue
        if np.all(base == base[0]):
            continue

        candidates = _object_candidates(base)
        if len(candidates) < 8:
            continue

        hole_positions = {r: [int(c) for c in np.flatnonzero(base[r])] for r in range(TILE_HEIGHT)}
        if any(len(cols) == 0 for cols in hole_positions.values()):
            continue
        return base, candidates, hole_positions

    return None


def _sample_hole_rows(count: int) -> list[int]:
    rows = list(range(TILE_HEIGHT))
    picked: list[int] = []
    while len(picked) < count:
        random.shuffle(rows)
        picked.extend(rows)
    return picked[:count]


def _feasible_tile_cols(tile_w: int) -> list[int]:
    return [n for n in range(TILE_COL_RANGE[0], TILE_COL_RANGE[1] + 1) if tile_w * n <= 30]


def _paint_object(grid: np.ndarray, origin_r: int, origin_c: int, cells: tuple[tuple[int, int], ...], color: int) -> None:
    for dr, dc in cells:
        grid[origin_r + dr, origin_c + dc] = color


def _gen_attempt() -> np.ndarray | None:
    tile_color = random.choice(TILE_COLORS)
    object_colors = [color for color in OBJECT_INPUT_COLORS if color != tile_color]
    tile_w = random.randint(*TILE_WIDTH_RANGE)
    col_options = _feasible_tile_cols(tile_w)
    if not col_options:
        return None

    sampled = _sample_tile_pattern(tile_w)
    if sampled is None:
        return None
    base, object_shapes, hole_positions = sampled

    tile_rows = random.randint(*TILE_ROW_RANGE)
    tile_cols = random.choice(col_options)
    h = TILE_HEIGHT * tile_rows
    w = tile_w * tile_cols

    grid = np.full((h, w), BG, dtype=int)
    for tile_r in range(tile_rows):
        for tile_c in range(tile_cols):
            r0 = tile_r * TILE_HEIGHT
            c0 = tile_c * tile_w
            patch = grid[r0 : r0 + TILE_HEIGHT, c0 : c0 + tile_w]
            patch[base] = tile_color

    all_tiles = [(tr, tc) for tr in range(tile_rows) for tc in range(tile_cols)]
    max_objects = min(len(all_tiles) - 2, 8)
    if max_objects < 4:
        return None
    n_objects = random.randint(4, max_objects)
    object_tiles = random.sample(all_tiles, n_objects)

    n_modified = random.randint(2, min(n_objects - 1, 6))
    modified_tiles = set(random.sample(object_tiles, n_modified))
    hole_rows = _sample_hole_rows(n_modified)

    objects: dict[tuple[int, int], tuple[tuple[tuple[int, int], ...], int]] = {}
    for tile in object_tiles:
        shape = random.choice(object_shapes)
        color = random.choice(object_colors)
        objects[tile] = (shape, color)
        r0 = tile[0] * TILE_HEIGHT
        c0 = tile[1] * tile_w
        _paint_object(grid, r0, c0, shape, color)

    expected = grid.copy()
    for tile, hole_row in zip(sorted(modified_tiles), hole_rows, strict=False):
        hole_cols = hole_positions[hole_row]
        hole_count = 1 if len(hole_cols) == 1 else random.randint(1, min(2, len(hole_cols)))
        for hole_col in random.sample(hole_cols, hole_count):
            rr = tile[0] * TILE_HEIGHT + hole_row
            cc = tile[1] * tile_w + hole_col
            grid[rr, cc] = BG
            expected[rr, cc] = BG

        shape, _color = objects[tile]
        rr = tile[0] * TILE_HEIGHT
        cc = tile[1] * tile_w
        for dr, dc in shape:
            expected[rr + dr, cc + dc] = RECOLOR_SEQUENCE[hole_row]

    try:
        inferred_color, inferred_h, inferred_w, inferred_base = _infer_period(grid)
    except AssertionError:
        return None
    if inferred_color != tile_color or inferred_h != TILE_HEIGHT or inferred_w != tile_w:
        return None
    if not np.array_equal(inferred_base, base):
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid periodic-hole ARC input")
