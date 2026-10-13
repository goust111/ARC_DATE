from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
GREY = 5
SEED_COLORS = (2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for sep in separators:
        assert start < sep, "separator lines must leave nonempty regions"
        spans.append((start, sep - 1))
        start = sep + 1
    assert start < length, "separator lines must leave a final nonempty region"
    spans.append((start, length - 1))
    return spans


def _separator_spans(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = g.shape
    sep_rows = [r for r in range(h) if np.all(g[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == GREY)]
    assert sep_rows and sep_cols, "expected full grey horizontal and vertical separator lines"

    row_spans = _spans(h, sep_rows)
    col_spans = _spans(w, sep_cols)
    assert len({r1 - r0 + 1 for r0, r1 in row_spans}) == 1, "row regions must have equal height"
    assert len({c1 - c0 + 1 for c0, c1 in col_spans}) == 1, "column regions must have equal width"
    return row_spans, col_spans


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                q.append((rr, cc))
    return len(seen) == int(np.count_nonzero(mask))


def _depth_map(mask: np.ndarray) -> np.ndarray:
    """Return each template pixel's inward erosion layer; non-template cells are -1."""
    h, w = mask.shape
    depth = np.full((h, w), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        boundary = False
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or not mask[rr, cc]:
                boundary = True
                break
        if boundary:
            depth[r, c] = 0
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and depth[rr, cc] < 0:
                depth[rr, cc] = depth[r, c] + 1
                q.append((rr, cc))

    assert np.all(depth[mask] >= 0), "every template pixel must receive a layer"
    return depth


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored concentric rings inward inside each blue template tile.

    Input structure:
    - The background is black and full grey separator rows and columns split the grid into equal rectangular regions.
    - A region is either empty black or contains one connected blue template shape with zero or more same-colored seed pixels on template cells.
    - Seed colors are any colors other than black, grey, and blue; their positions inside the template do not encode direction or placement.

    Rule:
    - For each region with a template and seeds, count the seed pixels N and read their shared color.
    - Treat the template boundary as layer 0, where boundary means template pixels 4-adjacent to black space, a grey separator, or the region edge.
    - Moving inward by repeated 4-neighbor erosion gives layers 1, 2, and so on; recolor only layers 1 through N with the seed color.
    - Layer 0 and any layers deeper than N are blue, so the original seed pixels are overwritten by their final layer color.

    Key insight:
    - The number of seed pixels controls the inward coloring depth; seed locations are irrelevant once their count and color are known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    row_spans, col_spans = _separator_spans(g)
    out = g.copy()

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            seed_mask = (tile != BLACK) & (tile != GREY) & (tile != BLUE)
            template = (tile == BLUE) | seed_mask
            if not np.any(template):
                assert np.all(tile == BLACK), "empty regions must be black"
                continue
            assert np.any(tile == BLUE), "nonempty regions must include a blue template"

            region = out[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(seed_mask):
                region[template] = BLUE
                continue

            seed_values = np.unique(tile[seed_mask])
            assert seed_values.size == 1, "all seeds in one region must share one color"
            seed_color = int(seed_values[0])
            seed_count = int(np.count_nonzero(seed_mask))

            depth = _depth_map(template)
            region[template] = BLUE
            fill = template & (depth >= 1) & (depth <= seed_count)
            region[fill] = seed_color

    return out


# --------------- generator ---------------

TILE_ROWS_RANGE = (2, 4)
TILE_COLS_RANGE = (2, 4)
TILE_SIDE_RANGE = (5, 8)
ACTIVE_REGION_MIN = 2
ACTIVE_REGION_MAX = 8
MAX_SEEDS = 5
GEN_RETRIES = 400


def _valid_tile_sizes(count: int) -> list[int]:
    return [size for size in range(TILE_SIDE_RANGE[0], TILE_SIDE_RANGE[1] + 1) if count * size + count - 1 <= 30]


def _base_grid(tile_rows: int, tile_cols: int, tile_h: int, tile_w: int) -> np.ndarray:
    h = tile_rows * tile_h + tile_rows - 1
    w = tile_cols * tile_w + tile_cols - 1
    grid = np.full((h, w), BLACK, dtype=int)
    for r in range(tile_h, h, tile_h + 1):
        grid[r, :] = GREY
    for c in range(tile_w, w, tile_w + 1):
        grid[:, c] = GREY
    return grid


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int) -> tuple[int, int]:
    return tile_r * (tile_h + 1), tile_c * (tile_w + 1)


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in NEIGHBORS4
        ):
            cells.append((r, c))
    return cells


def _random_notched_box(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    removals = random.randint(1, max(1, (height + width) // 2))
    for _ in range(removals):
        candidates = _boundary_cells(mask)
        random.shuffle(candidates)
        for r, c in candidates:
            trial = mask.copy()
            trial[r, c] = False
            if int(np.count_nonzero(trial)) < 6:
                continue
            if not _is_connected(trial):
                continue
            if int(_depth_map(trial).max()) < 1:
                continue
            mask = trial
            break
    return mask


def _diamond(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    mask = np.zeros((size, size), dtype=bool)
    for r in range(size):
        for c in range(size):
            if abs(r - radius) + abs(c - radius) <= radius:
                mask[r, c] = True
    return mask


def _thick_cross(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cr = height // 2
    cc = width // 2
    mask[max(0, cr - 1) : min(height, cr + 2), :] = True
    mask[:, max(0, cc - 1) : min(width, cc + 2)] = True
    return mask


def _make_template(tile_h: int, tile_w: int, need_depth: bool) -> np.ndarray | None:
    for _ in range(80):
        kind = random.choice(["box", "notched", "diamond", "cross"])
        if kind == "diamond":
            max_radius = min((tile_h - 1) // 2, (tile_w - 1) // 2)
            if max_radius < 2:
                continue
            mask = _diamond(random.randint(2, max_radius))
        else:
            min_h = 5 if kind == "cross" else 4
            min_w = 5 if kind == "cross" else 4
            if tile_h < min_h or tile_w < min_w:
                continue
            height = random.randint(min_h, tile_h)
            width = random.randint(min_w, tile_w)
            if kind == "box":
                mask = np.ones((height, width), dtype=bool)
            elif kind == "notched":
                mask = _random_notched_box(height, width)
            else:
                mask = _thick_cross(height, width)

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if need_depth and int(_depth_map(mask).max()) < 1:
            continue
        return mask.copy()
    return None


def _choose_seed_count(mask: np.ndarray) -> int:
    max_depth = int(_depth_map(mask).max())
    max_count = min(MAX_SEEDS, int(np.count_nonzero(mask)))
    if max_depth >= 2 and random.random() < 0.5:
        return random.randint(1, min(max_depth - 1, max_count))
    low = max(1, max_depth)
    high = min(max_count, max_depth + 2)
    if low > high:
        low = 1
    return random.randint(low, high)


def _paste_template(
    grid: np.ndarray,
    tile_r: int,
    tile_c: int,
    tile_h: int,
    tile_w: int,
    mask: np.ndarray,
    seed_count: int,
    seed_color: int | None,
) -> None:
    r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w)
    mh, mw = mask.shape
    rr = r0 + random.randint(0, tile_h - mh)
    cc = c0 + random.randint(0, tile_w - mw)

    tile = grid[rr : rr + mh, cc : cc + mw]
    tile[mask] = BLUE
    if seed_count:
        assert seed_color is not None
        coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
        for sr, sc in random.sample(coords, seed_count):
            tile[sr, sc] = seed_color


def _region_plan(tile_rows: int, tile_cols: int) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    positions = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    random.shuffle(positions)
    max_active = min(len(positions) - 1, ACTIVE_REGION_MAX)
    active_count = random.randint(ACTIVE_REGION_MIN, max(ACTIVE_REGION_MIN, max_active))
    active = positions[:active_count]

    seedless_count = 0
    if active_count > 1 and random.random() < 0.45:
        seedless_count = random.randint(1, min(2, active_count - 1))
    seedless = set(random.sample(active, seedless_count))
    return active, seedless


def _render_expected(inp: np.ndarray) -> np.ndarray | None:
    try:
        return main(inp)
    except Exception:
        return None


def _gen_attempt() -> np.ndarray | None:
    tile_rows = random.randint(*TILE_ROWS_RANGE)
    tile_cols = random.randint(*TILE_COLS_RANGE)
    row_sizes = _valid_tile_sizes(tile_rows)
    col_sizes = _valid_tile_sizes(tile_cols)
    if not row_sizes or not col_sizes:
        return None
    tile_h = random.choice(row_sizes)
    tile_w = random.choice(col_sizes)

    grid = _base_grid(tile_rows, tile_cols, tile_h, tile_w)
    active, seedless = _region_plan(tile_rows, tile_cols)

    used_seeded = 0
    for pos in active:
        has_seeds = pos not in seedless
        mask = _make_template(tile_h, tile_w, need_depth=has_seeds)
        if mask is None:
            return None
        seed_count = 0
        seed_color = None
        if has_seeds:
            seed_count = _choose_seed_count(mask)
            seed_color = random.choice(SEED_COLORS)
            used_seeded += 1
        _paste_template(grid, pos[0], pos[1], tile_h, tile_w, mask, seed_count, seed_color)

    if used_seeded == 0:
        return None
    if max(grid.shape) > 30:
        return None

    expected = _render_expected(grid)
    if expected is None or expected.shape != grid.shape or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1030756833/7/3 input")
