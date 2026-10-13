from __future__ import annotations

import numpy as np

BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

START_OFFSETS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_H_RANGE = (14, 21)
GRID_W_RANGE = (14, 21)
RECT_H_RANGE = (6, 10)
RECT_W_RANGE = (6, 10)
RECT_EDGE_GAP = 2
SEED_TOP_BAND = 3
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_rectangle_and_markers(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    rect_mask = (grid == BLUE) | (grid == RED)
    assert np.any(rect_mask), "expected a blue rectangle with red markers"

    r0, c0, r1, c1 = _bbox_from_mask(rect_mask)
    bbox_mask = np.zeros_like(rect_mask, dtype=bool)
    bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert np.array_equal(rect_mask, bbox_mask), "blue/red region must form one filled rectangle"
    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((rect == BLUE) | (rect == RED)), "rectangle may only contain blue and red"

    marker_coords = np.argwhere(rect == RED)
    assert marker_coords.size > 0, "expected at least one red marker"
    marker_rows = sorted({r0 + int(r) for r, _c in marker_coords})
    marker_cols = sorted({c0 + int(c) for _r, c in marker_coords})
    assert len(marker_rows) == len(marker_coords), "marker rows must be unique"
    assert len(marker_cols) == len(marker_coords), "marker cols must be unique"
    return (r0, c0, r1, c1), marker_rows, marker_cols


def _inside_rect(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _seed_order(grid: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    return sorted(coords, key=lambda rc: (rc[1], rc[0]))


def _draw_trails(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    marker_rows: set[int],
    marker_cols: set[int],
    seeds: list[tuple[int, int]],
) -> None:
    h, w = out.shape
    for seed_r, seed_c in seeds:
        for dr, dc in START_OFFSETS:
            r = seed_r + dr
            c = seed_c + dc
            if not _in_bounds((h, w), r, c):
                continue

            if out[r, c] != RED and out[r, c] != GREEN:
                out[r, c] = GREEN

            while True:
                nr = r + 2
                if not _in_bounds((h, w), nr, c):
                    break
                if _inside_rect(rect, nr, c) and (nr in marker_rows or c in marker_cols):
                    break
                if out[nr, c] != RED and out[nr, c] != GREEN:
                    out[nr, c] = GREEN
                r = nr


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the red markers into a blocking cross grid, then grow downward green trails that stop on those blockers.

    Input structure:
    - The grid has a black background plus one filled blue rectangle.
    - Inside the rectangle, sparse red marker pixels occupy distinct rows and distinct columns.
    - Several isolated green seed pixels sit in the top band outside the rectangle.

    Rule:
    - Keep the full input canvas.
    - Inside the blue rectangle, expand every red marker into its full red row and full red column, clipped to the rectangle bounds.
    - From each green seed, start five vertical period-2 trails at the seed and its four diagonal neighbors; each trail continues downward in the same column.
    - When a trail's next landing cell is inside the rectangle and already belongs to the red cross grid, stop that trail before drawing the blocked cell; otherwise paint the landing cell green.

    Key insight:
    - The red markers define a row-and-column obstacle lattice inside the rectangle, and the green falling trails are determined by how their period-2 columns intersect that lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rect, marker_rows, marker_cols = _find_rectangle_and_markers(g)
    out = g.copy()
    r0, c0, r1, c1 = rect
    for r in marker_rows:
        out[r, c0 : c1 + 1] = RED
    for c in marker_cols:
        out[r0 : r1 + 1, c] = RED

    seeds = _seed_order(g)
    _draw_trails(out, rect, set(marker_rows), set(marker_cols), seeds)
    return out


def _sample_spaced_values(
    rng: np.random.Generator,
    values: list[int],
    n: int,
    min_gap: int,
) -> list[int] | None:
    order = values[:]
    rng.shuffle(order)
    chosen: list[int] = []
    for value in order:
        if all(abs(value - old) >= min_gap for old in chosen):
            chosen.append(int(value))
            if len(chosen) == n:
                return sorted(chosen)
    return None


def _sample_markers(
    rng: np.random.Generator,
    rect: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = rect
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    n_markers = 3 if rng.random() < 0.4 else 2

    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    if len(rows) < n_markers or len(cols) < n_markers:
        return None

    min_gap = 2 if min(rect_h, rect_w) >= 7 else 1
    marker_rows = _sample_spaced_values(rng, rows, n_markers, min_gap)
    marker_cols = _sample_spaced_values(rng, cols, n_markers, min_gap)
    if marker_rows is None or marker_cols is None:
        marker_rows = _sample_spaced_values(rng, rows, n_markers, 1)
        marker_cols = _sample_spaced_values(rng, cols, n_markers, 1)
    if marker_rows is None or marker_cols is None:
        return None

    permuted_cols = marker_cols[:]
    rng.shuffle(permuted_cols)
    return list(zip(marker_rows, permuted_cols, strict=True))


def _seed_stats(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    marker_rows: set[int],
    marker_cols: set[int],
    seed: tuple[int, int],
) -> dict[str, int | bool | tuple[int, int]]:
    enters = False
    blocked = 0
    paints_inside = 0

    for dr, dc in START_OFFSETS:
        r = seed[0] + dr
        c = seed[1] + dc
        if not _in_bounds(shape, r, c):
            continue
        while True:
            nr = r + 2
            if not _in_bounds(shape, nr, c):
                break
            if _inside_rect(rect, nr, c):
                enters = True
                if nr in marker_rows or c in marker_cols:
                    blocked += 1
                    break
                paints_inside += 1
            r = nr

    return {
        "seed": seed,
        "enters": enters,
        "blocked": blocked > 0,
        "paints": paints_inside > 0,
        "blocked_count": blocked,
        "inside_count": paints_inside,
    }


def _seed_spacing_ok(seed: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(seed[0] - r), abs(seed[1] - c)) >= 2 for r, c in chosen)


def _pick_seed(
    rng: np.random.Generator,
    pool: list[dict[str, int | bool | tuple[int, int]]],
    chosen: list[tuple[int, int]],
) -> tuple[int, int] | None:
    ranked = pool[:]
    rng.shuffle(ranked)
    ranked.sort(
        key=lambda item: (int(item["blocked_count"]), int(item["inside_count"])),
        reverse=True,
    )
    for candidate in ranked:
        seed = candidate["seed"]
        assert isinstance(seed, tuple)
        if _seed_spacing_ok(seed, chosen):
            return seed
    return None


def _choose_seeds(
    rng: np.random.Generator,
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    marker_rows: set[int],
    marker_cols: set[int],
    n_seeds: int,
) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates: list[dict[str, int | bool | tuple[int, int]]] = []
    for r in range(min(SEED_TOP_BAND, h)):
        for c in range(w):
            if _inside_rect(rect, r, c):
                continue
            stats = _seed_stats(shape, rect, marker_rows, marker_cols, (r, c))
            if bool(stats["enters"]):
                candidates.append(stats)

    blocked_pool = [item for item in candidates if bool(item["blocked"])]
    paint_pool = [item for item in candidates if bool(item["paints"])]
    if not blocked_pool or not paint_pool:
        return None

    chosen: list[tuple[int, int]] = []
    first = _pick_seed(rng, blocked_pool, chosen)
    if first is None:
        return None
    chosen.append(first)

    second = _pick_seed(
        rng,
        [item for item in paint_pool if item["seed"] != first],
        chosen,
    )
    if second is None:
        return None
    chosen.append(second)

    while len(chosen) < n_seeds:
        extra = _pick_seed(
            rng,
            [item for item in candidates if item["seed"] not in chosen],
            chosen,
        )
        if extra is None:
            return None
        chosen.append(extra)

    return sorted(chosen)


def _rectangle_params(rng: np.random.Generator) -> tuple[int, int, int, int, int, int]:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    rect_h = int(rng.integers(*RECT_H_RANGE))
    rect_w = int(rng.integers(*RECT_W_RANGE))
    rect_h = min(rect_h, h - 2 * RECT_EDGE_GAP)
    rect_w = min(rect_w, w - 2 * RECT_EDGE_GAP)
    if rect_h < 6 or rect_w < 6:
        raise ValueError("grid too small for the required rectangle")

    lower_band_start = max(4, h // 2 - 1)
    lower_band_end = h - rect_h - RECT_EDGE_GAP
    if lower_band_start > lower_band_end:
        lower_band_start = RECT_EDGE_GAP + SEED_TOP_BAND + 1
        lower_band_end = h - rect_h - RECT_EDGE_GAP
    r0 = int(rng.integers(lower_band_start, lower_band_end + 1))
    c0 = int(rng.integers(RECT_EDGE_GAP, w - rect_w - RECT_EDGE_GAP + 1))
    return h, w, r0, c0, rect_h, rect_w


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    try:
        h, w, r0, c0, rect_h, rect_w = _rectangle_params(rng)
    except ValueError:
        return None

    rect = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
    markers = _sample_markers(rng, rect)
    if markers is None:
        return None

    marker_rows = {r for r, _c in markers}
    marker_cols = {c for _r, c in markers}
    n_seeds = 3 if rng.random() < 0.45 else 2
    seeds = _choose_seeds(rng, (h, w), rect, marker_rows, marker_cols, n_seeds)
    if seeds is None:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[r0 : r0 + rect_h, c0 : c0 + rect_w] = BLUE
    for r, c in markers:
        grid[r, c] = RED
    for r, c in seeds:
        grid[r, c] = GREEN

    out = main(grid)
    inside = out[r0 : r0 + rect_h, c0 : c0 + rect_w]
    green_inside = int(np.count_nonzero(inside == GREEN))
    if green_inside < 1:
        return None
    if green_inside > max(8, rect_h * rect_w // 4):
        return None

    if not np.any((inside == RED) & (grid[r0 : r0 + rect_h, c0 : c0 + rect_w] == BLUE)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
