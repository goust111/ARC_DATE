import random

import numpy as np


MISSING = 0
RED = 2
WALLPAPER_COLORS = (1, 3, 4)
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRY_LIMIT = 400


def _polluted_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    polluted = np.zeros((h, w), dtype=bool)
    for sr, sc in np.argwhere(g == RED):
        sr = int(sr)
        sc = int(sc)
        for dr, dc in DIRS4:
            r, c = sr + dr, sc + dc
            while 0 <= r < h and 0 <= c < w:
                if g[r, c] == MISSING:
                    polluted[r, c] = True
                    r += dr
                    c += dc
                    continue
                break
    return polluted


def _valid_period(g: np.ndarray, tile_h: int, tile_w: int) -> bool:
    phase_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere((g != MISSING) & (g != RED)):
        r = int(r)
        c = int(c)
        key = (r % tile_h, c % tile_w)
        color = int(g[r, c])
        if key in phase_colors and phase_colors[key] != color:
            return False
        phase_colors[key] = color
    return True


def _infer_minimal_tile(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    candidates = sorted(
        ((tile_h, tile_w) for tile_h in range(1, h + 1) for tile_w in range(1, w + 1)),
        key=lambda x: (x[0] * x[1], x[0], x[1]),
    )
    for tile_h, tile_w in candidates:
        if not _valid_period(g, tile_h, tile_w):
            continue
        tile = np.full((tile_h, tile_w), -1, dtype=int)
        for r, c in np.argwhere((g != MISSING) & (g != RED)):
            r = int(r)
            c = int(c)
            tile[r % tile_h, c % tile_w] = int(g[r, c])
        if np.any(tile < 0):
            continue
        return tile
    raise AssertionError("failed to infer a complete minimal tile")


def _tile_wallpaper(tile: np.ndarray, h: int, w: int, off_r: int = 0, off_c: int = 0) -> np.ndarray:
    rows = (np.arange(h)[:, None] + off_r) % tile.shape[0]
    cols = (np.arange(w)[None, :] + off_c) % tile.shape[1]
    return tile[rows, cols]


def _min_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = tile.shape
    for tile_h in range(1, h + 1):
        for tile_w in range(1, w + 1):
            if tile_h == h and tile_w == w:
                return h, w
            probe = _tile_wallpaper(tile[:tile_h, :tile_w], h, w)
            if np.array_equal(probe, tile):
                return tile_h, tile_w
    return h, w


def _sample_tile() -> np.ndarray | None:
    tile_h = random.choice([3, 4])
    tile_w = random.choice([3, 4])
    n_colors = random.choice([2, 3])
    colors = random.sample(list(WALLPAPER_COLORS), n_colors)
    tile = np.array(
        [[random.choice(colors) for _ in range(tile_w)] for _ in range(tile_h)],
        dtype=int,
    )
    if len(set(map(int, tile.flat))) < 2:
        return None
    if _min_period(tile) != (tile_h, tile_w):
        return None
    return tile


def _phase_visible(masked: set[tuple[int, int]], h: int, w: int, tile_h: int, tile_w: int) -> bool:
    counts = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(h):
        for c in range(w):
            if (r, c) not in masked:
                counts[r % tile_h, c % tile_w] += 1
    return bool(np.all(counts > 0))


def _source_candidates(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]


def _choose_sources(h: int, w: int, n_sources: int) -> list[tuple[int, int]] | None:
    candidates = _source_candidates(h, w)
    random.shuffle(candidates)
    sources: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(r == rr or c == cc or abs(r - rr) + abs(c - cc) < 5 for rr, cc in sources):
            continue
        sources.append((r, c))
        if len(sources) == n_sources:
            return sources
    return None


def _dir_capacity(h: int, w: int, sr: int, sc: int, dr: int, dc: int) -> int:
    steps = 0
    r, c = sr + dr, sc + dc
    while 0 <= r < h and 0 <= c < w:
        steps += 1
        r += dr
        c += dc
    return steps


def _run_cells(sr: int, sc: int, dr: int, dc: int, length: int) -> list[tuple[int, int]]:
    return [(sr + dr * step, sc + dc * step) for step in range(1, length + 1)]


def _choose_polluted_runs(
    h: int, w: int, sources: list[tuple[int, int]]
) -> set[tuple[int, int]] | None:
    source_set = set(sources)
    polluted: set[tuple[int, int]] = set()
    for sr, sc in sources:
        options = []
        for dr, dc in DIRS4:
            cap = _dir_capacity(h, w, sr, sc, dr, dc)
            for length in range(2, min(cap, 5) + 1):
                cells = _run_cells(sr, sc, dr, dc, length)
                if any(cell in source_set or cell in polluted for cell in cells):
                    continue
                options.append((dr, dc, length, cells))
        if not options:
            return None

        target_dirs = min(len({(dr, dc) for dr, dc, _, _ in options}), random.choice([1, 1, 2, 2, 3]))
        random.shuffle(options)
        chosen_dirs = []
        used_dirs: set[tuple[int, int]] = set()
        for dr, dc, length, cells in options:
            if (dr, dc) in used_dirs:
                continue
            if any(cell in polluted for cell in cells):
                continue
            chosen_dirs.append((dr, dc, length, cells))
            used_dirs.add((dr, dc))
            if len(chosen_dirs) == target_dirs:
                break
        if not chosen_dirs:
            return None
        for _, _, _, cells in chosen_dirs:
            polluted.update(cells)
    return polluted


def _eligible_extra_holes(
    h: int, w: int, sources: list[tuple[int, int]], polluted: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    blocked_rows = {r for r, _ in sources}
    blocked_cols = {c for _, c in sources}
    source_set = set(sources)
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in source_set
        and (r, c) not in polluted
        and r not in blocked_rows
        and c not in blocked_cols
    ]


def _expected_output(
    wallpaper: np.ndarray,
    sources: list[tuple[int, int]],
    polluted: set[tuple[int, int]],
) -> np.ndarray:
    out = wallpaper.copy()
    for r, c in polluted:
        out[r, c] = RED
    for r, c in sources:
        out[r, c] = RED
    return out


def _gen_attempt() -> np.ndarray | None:
    tile = _sample_tile()
    if tile is None:
        return None

    tile_h, tile_w = tile.shape
    h = random.randint(max(12, tile_h * 3), min(16, 30))
    w = random.randint(max(12, tile_w * 3), min(16, 30))
    off_r = random.randrange(tile_h)
    off_c = random.randrange(tile_w)
    wallpaper = _tile_wallpaper(tile, h, w, off_r=off_r, off_c=off_c)

    n_sources = 2 if random.random() < 0.45 else 1
    sources = _choose_sources(h, w, n_sources)
    if sources is None:
        return None

    polluted = _choose_polluted_runs(h, w, sources)
    if polluted is None:
        return None

    target_holes = random.randint(int(round(h * w * 0.16)), int(round(h * w * 0.22)))
    if target_holes <= len(polluted) + 2:
        target_holes = len(polluted) + 3

    eligible = _eligible_extra_holes(h, w, sources, polluted)
    extra_count = target_holes - len(polluted)
    if extra_count < 3 or len(eligible) < extra_count:
        return None

    extra_holes = set(random.sample(eligible, extra_count))
    all_holes = polluted | extra_holes
    masked = all_holes | set(sources)
    if not _phase_visible(masked, h, w, tile_h, tile_w):
        return None

    inp = wallpaper.copy()
    for r, c in all_holes:
        inp[r, c] = MISSING
    for r, c in sources:
        inp[r, c] = RED

    inferred = _infer_minimal_tile(inp)
    if inferred.shape != tile.shape:
        return None
    if not np.array_equal(_tile_wallpaper(inferred, h, w), wallpaper):
        return None

    expected = _expected_output(wallpaper, sources, polluted)
    if np.count_nonzero(expected == RED) <= len(sources):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore a periodic wallpaper while recoloring ray-shadowed holes red.

    Input structure:
    - The grid is a repeated wallpaper made from a small 2D tile using non-red non-black colors.
    - Some wallpaper cells are replaced by black holes, and 1-2 red source pixels overwrite wallpaper cells.
    - Each red source may have contiguous black holes extending from it in one or more cardinal directions.

    Rule:
    - From every red source, trace rays up, right, down, and left.
    - Along each ray, mark consecutive black cells as polluted and stop as soon as a non-black cell or the grid edge is reached.
    - Infer the smallest repeating wallpaper tile from all cells that are neither black nor red.
    - Fill unpolluted black holes with their wallpaper tile color, fill polluted black holes with red, and keep red source cells unchanged.

    Key insight:
    - Black holes split into two classes: ordinary missing wallpaper pixels and ray-shadowed holes claimed by a red source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    polluted = _polluted_mask(g)
    tile = _infer_minimal_tile(g)
    wallpaper = _tile_wallpaper(tile, g.shape[0], g.shape[1])

    out = g.copy()
    missing = g == MISSING
    out[missing & ~polluted] = wallpaper[missing & ~polluted]
    out[missing & polluted] = RED
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
