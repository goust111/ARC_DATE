import numpy as np


BACKGROUND = 0
SEPARATOR_COLOR = 5
HEADER_ROWS = 3
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_HEIGHT_RANGE = (14, 19)
GEN_WIDTH_OPTIONS = (14, 16, 18, 20)
GEN_PALETTE_LEN_RANGE = (3, 6)
GEN_SEED_COUNT_OPTIONS = (3, 4, 5)
GEN_MAX_ATTEMPTS = 400
GEN_EXTRA_PLACEMENT_ATTEMPTS = 120


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _read_palette_row(row: np.ndarray, bg: int) -> np.ndarray:
    row = np.asarray(row, dtype=int)
    end = 0
    while end < row.size and int(row[end]) != bg:
        end += 1
    assert 3 <= end <= 5, "palette rows must have a contiguous 3-5 cell prefix"
    assert np.all(row[end:] == bg), "palette rows must be left-aligned contiguous prefixes"
    palette = row[:end].astype(int)
    assert len(set(map(int, palette))) == len(palette), "palette colors must be distinct within a row"
    return palette


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h > HEADER_ROWS and w % 2 == 0, "expected a lower region and an even width"

    bg = _background_color(g)
    assert bg == BACKGROUND, "expected black background"

    sep_values = np.unique(g[1])
    assert len(sep_values) == 1 and int(sep_values[0]) == SEPARATOR_COLOR, "row 1 must be the grey separator"

    palette1 = _read_palette_row(g[0], bg)
    palette2 = _read_palette_row(g[2], bg)
    seed_color = int(palette1[0])
    assert int(palette2[-1]) == seed_color, "isolated seeds must match the reversed second palette at distance 0"

    lower = g[HEADER_ROWS:]
    seed_coords = np.argwhere(lower != bg)
    assert 3 <= len(seed_coords) <= 5, "expected 3-5 seed pixels"
    seed_coords[:, 0] += HEADER_ROWS
    seeds = sorted((int(r), int(c)) for r, c in seed_coords)
    seed_set = set(seeds)

    for r, c in seeds:
        assert int(g[r, c]) == seed_color, "all seeds must use the first color of the top palette"
        assert 2 <= r <= h - 3 and 2 <= c <= w - 3, "seeds must stay at least two cells from the border"
        for dr, dc in NEIGHBORS4:
            assert (r + dr, c + dc) not in seed_set, "seeds must not be 4-connected"

    seed_infos = _seed_infos(seeds, w)
    assert any(paired for _r, _c, paired in seed_infos), "expected at least one mirrored seed pair"
    assert any(not paired for _r, _c, paired in seed_infos), "expected at least one isolated seed"

    return palette1, palette2, seeds, g


def _seed_infos(seeds: list[tuple[int, int]], width: int) -> list[tuple[int, int, bool]]:
    seed_set = set(seeds)
    return [(r, c, (r, width - 1 - c) in seed_set) for r, c in seeds]


def _winner_maps(
    shape: tuple[int, int],
    seed_infos: list[tuple[int, int, bool]],
    palette1: np.ndarray,
    palette2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map(int, shape)
    rows, cols = np.indices((h, w))
    best_dist = np.full((h, w), 1_000_000, dtype=int)
    best_row = np.full((h, w), h + 1, dtype=int)
    best_col = np.full((h, w), w + 1, dtype=int)
    winner_idx = np.full((h, w), -1, dtype=int)
    winner_color = np.full((h, w), BACKGROUND, dtype=int)
    reversed_palette2 = palette2[::-1]

    for idx, (sr, sc, is_paired) in enumerate(seed_infos):
        palette = palette1 if is_paired else reversed_palette2
        dist = np.maximum(np.abs(rows - sr), np.abs(cols - sc))
        valid = (rows >= HEADER_ROWS) & (dist < len(palette))
        better = valid & (
            (dist < best_dist)
            | ((dist == best_dist) & ((sr < best_row) | ((sr == best_row) & (sc < best_col))))
        )
        if not np.any(better):
            continue
        winner_idx[better] = idx
        best_dist[better] = dist[better]
        best_row[better] = sr
        best_col[better] = sc
        winner_color[better] = palette[dist[better]]

    return winner_idx, best_dist, winner_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand mirrored seed pairs and isolated seeds with different palette directions, then keep the nearest ring color.

    Input structure:
    - The grid has an even width and a black background.
    - Row 0 is a left-aligned palette prefix for mirrored seed pairs, row 1 is a full grey separator, and row 2 is a left-aligned palette prefix for isolated seeds.
    - Rows below the separator contain 3-5 singleton seed pixels, all in the first color of the top palette; at least one same-row horizontal mirror pair exists, and at least one seed is unpaired.

    Rule:
    - A seed is paired iff the same row also contains a seed at the horizontally mirrored column `W-1-c`.
    - Paired seeds use the top palette in forward order; isolated seeds use the lower palette in reverse order.
    - For each lower-region cell, consider every seed whose Chebyshev radius reaches that cell and keep the color from the seed with smallest distance; distance ties break by smaller seed row, then smaller seed column.
    - The palette rows, separator row, and seed pixels themselves remain unchanged.

    Key insight:
    - Horizontal mirror pairing decides which palette direction each seed reads, while overlaps are resolved purely by nearest-seed Chebyshev distance with a fixed lexicographic tie-break.
    """
    palette1, palette2, seeds, g = _parse_grid(grid)
    seed_infos = _seed_infos(seeds, g.shape[1])
    winner_idx, _winner_dist, winner_color = _winner_maps(g.shape, seed_infos, palette1, palette2)

    out = np.full_like(g, BACKGROUND)
    covered = winner_idx >= 0
    out[covered] = winner_color[covered]
    out[:HEADER_ROWS] = g[:HEADER_ROWS]
    for r, c in seeds:
        out[r, c] = int(g[r, c])
    return out


def _sample_palettes(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    len1 = int(rng.integers(*GEN_PALETTE_LEN_RANGE))
    len2 = int(rng.integers(*GEN_PALETTE_LEN_RANGE))
    seed_color = int(rng.choice(COLOR_POOL))
    remaining = [int(c) for c in COLOR_POOL if int(c) != seed_color]

    palette1 = np.array([seed_color, *rng.choice(remaining, size=len1 - 1, replace=False)], dtype=int)
    while True:
        prefix = rng.choice(remaining, size=len2 - 1, replace=False)
        palette2 = np.array([*prefix, seed_color], dtype=int)
        if int(palette1[1]) != int(palette2[-2]) and int(palette1[1]) != int(palette2[-3]):
            return palette1, palette2


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _reserve_zone(cells: list[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    reserved.add((nr, nc))
    return reserved


def _can_place_seed(
    seeds: set[tuple[int, int]],
    r: int,
    c: int,
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
) -> bool:
    if not (2 <= r <= h - 3 and 2 <= c <= w - 3):
        return False
    if (r, c) in seeds or (r, c) in reserved:
        return False
    for sr, sc in seeds:
        if abs(sr - r) + abs(sc - c) == 1:
            return False
    return True


def _place_extra_isolated(
    rng: np.random.Generator,
    seeds: set[tuple[int, int]],
    h: int,
    w: int,
    row_min: int,
    reserved: set[tuple[int, int]],
) -> bool:
    for _ in range(GEN_EXTRA_PLACEMENT_ATTEMPTS):
        r = int(rng.integers(row_min, h - 2))
        c = int(rng.integers(2, w - 2))
        if (r, w - 1 - c) in seeds:
            continue
        if _can_place_seed(seeds, r, c, h, w, reserved):
            seeds.add((r, c))
            return True
    return False


def _place_extra_pair(
    rng: np.random.Generator,
    seeds: set[tuple[int, int]],
    h: int,
    w: int,
    row_min: int,
    reserved: set[tuple[int, int]],
) -> bool:
    left_choices = list(range(2, (w // 2) - 1))
    for _ in range(GEN_EXTRA_PLACEMENT_ATTEMPTS):
        r = int(rng.integers(row_min, h - 2))
        c = int(rng.choice(left_choices))
        mc = w - 1 - c
        if not _can_place_seed(seeds, r, c, h, w, reserved):
            continue
        if not _can_place_seed(seeds, r, mc, h, w, reserved):
            continue
        seeds.add((r, c))
        seeds.add((r, mc))
        return True
    return False


def _best_candidate_by_branch(
    candidates: list[tuple[int, int, int, int, bool]],
    want_paired: bool,
) -> tuple[int, int, int, int, bool] | None:
    filtered = [cand for cand in candidates if cand[4] == want_paired]
    if not filtered:
        return None
    return min(filtered, key=lambda item: (item[0], item[1], item[2]))


def _validate_candidate(grid: np.ndarray) -> bool:
    try:
        palette1, palette2, seeds, g = _parse_grid(grid)
        h, w = map(int, g.shape)
        seed_infos = _seed_infos(seeds, w)
        reversed_palette2 = palette2[::-1]
        seed_set = set(seeds)

        paired_seeds = [(r, c) for r, c, paired in seed_infos if paired]
        isolated_seeds = [(r, c) for r, c, paired in seed_infos if not paired]
        if not paired_seeds or not isolated_seeds:
            return False

        if not any(any(orow != r and ocol == w - 1 - c for orow, ocol in seeds) for r, c in isolated_seeds):
            return False

        winner_idx, winner_dist, winner_color = _winner_maps(g.shape, seed_infos, palette1, palette2)
        if np.any(winner_idx[:HEADER_ROWS] != -1):
            return False

        paired_ring1_visible = False
        isolated_ring1_visible = False
        tie_between_branches = False
        overlap_between_branches = False

        for r in range(HEADER_ROWS, h):
            for c in range(w):
                if (r, c) in seed_set:
                    continue

                candidates: list[tuple[int, int, int, int, bool]] = []
                for sr, sc, is_paired in seed_infos:
                    palette = palette1 if is_paired else reversed_palette2
                    dist = _chebyshev((r, c), (sr, sc))
                    if dist < len(palette):
                        candidates.append((dist, sr, sc, int(palette[dist]), is_paired))

                if not candidates:
                    continue

                if winner_idx[r, c] < 0:
                    return False

                winner = min(candidates, key=lambda item: (item[0], item[1], item[2]))
                if winner_dist[r, c] != winner[0] or winner_color[r, c] != winner[3]:
                    return False

                if winner[0] == 1 and winner[4]:
                    paired_ring1_visible = True
                if winner[0] == 1 and not winner[4]:
                    isolated_ring1_visible = True

                best_pair = _best_candidate_by_branch(candidates, True)
                best_isolated = _best_candidate_by_branch(candidates, False)
                if best_pair is not None and best_isolated is not None and best_pair[3] != best_isolated[3]:
                    if best_pair[0] == best_isolated[0]:
                        tie_between_branches = True
                    else:
                        overlap_between_branches = True

        return paired_ring1_visible and isolated_ring1_visible and tie_between_branches and overlap_between_branches
    except AssertionError:
        return False


def _build_candidate(rng: np.random.Generator) -> np.ndarray | None:
    palette1, palette2 = _sample_palettes(rng)
    radius = max(len(palette1), len(palette2)) - 1

    height_min = max(GEN_HEIGHT_RANGE[0], HEADER_ROWS + radius + 6)
    if height_min >= GEN_HEIGHT_RANGE[1]:
        return None
    h = int(rng.integers(height_min, GEN_HEIGHT_RANGE[1]))
    w = int(rng.choice(GEN_WIDTH_OPTIONS))

    row_min = HEADER_ROWS + radius + 1
    if h - 5 <= row_min:
        return None

    pair_r = int(rng.integers(row_min, h - 4))
    pair_left = int(rng.integers(2, (w // 2) - 1))
    pair_right = w - 1 - pair_left
    shared_c = pair_left if bool(rng.integers(0, 2)) else pair_right
    iso_r = pair_r + 2

    seeds: set[tuple[int, int]] = {
        (pair_r, pair_left),
        (pair_r, pair_right),
        (iso_r, shared_c),
    }

    focus_cells = [
        (pair_r + 1, shared_c),
        (pair_r - 1, shared_c),
        (iso_r + 1, shared_c),
    ]
    reserved = _reserve_zone([*seeds, *focus_cells], h, w)

    target_count = int(rng.choice(GEN_SEED_COUNT_OPTIONS))
    additions: list[str]
    if target_count == 3:
        additions = []
    elif target_count == 4:
        additions = ["isolated"]
    else:
        additions = ["pair"] if bool(rng.integers(0, 2)) else ["isolated", "isolated"]

    for kind in additions:
        if kind == "pair":
            if len(seeds) > 3:
                return None
            if not _place_extra_pair(rng, seeds, h, w, row_min, reserved):
                return None
        else:
            if not _place_extra_isolated(rng, seeds, h, w, row_min, reserved):
                return None

    if len(seeds) != target_count:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[0, : len(palette1)] = palette1
    grid[1, :] = SEPARATOR_COLOR
    grid[2, : len(palette2)] = palette2
    for r, c in seeds:
        grid[r, c] = int(palette1[0])
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _build_candidate(rng)
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirror-palette multi-seed input")
