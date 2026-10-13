import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).reshape(-1))).most_common(1)[0][0])


def _full_rows(grid: np.ndarray, color: int) -> list[int]:
    return [int(r) for r in range(grid.shape[0]) if bool((grid[r] == int(color)).all())]


def _full_cols(grid: np.ndarray, color: int) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if bool((grid[:, c] == int(color)).all())]


def _find_separator_color(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    h, w = map(int, grid.shape)
    best: tuple[int, int, list[int], list[int]] | None = None
    for color in map(int, np.unique(grid)):
        rows = _full_rows(grid, color)
        cols = _full_cols(grid, color)
        if len(rows) < 5 or len(cols) < 4:
            continue
        row_gaps = [int(rows[i + 1] - rows[i]) for i in range(len(rows) - 1)]
        col_gaps = [int(cols[i + 1] - cols[i]) for i in range(len(cols) - 1)]
        if not row_gaps or not col_gaps:
            continue
        if min(row_gaps) < 2 or min(col_gaps) < 2:
            continue
        score = int(len(rows) + len(cols))
        cand = (score, int(color), rows, cols)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None, "could not detect separator color"
    return int(best[1]), best[2], best[3]


def _crop_non_bg(tile: np.ndarray, bg: int, sep: int) -> np.ndarray:
    mask = (tile != int(bg)) & (tile != int(sep))
    assert bool(mask.any()), "sample strip must contain a visible pattern"
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(tile[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


def _stamp_center(dst: np.ndarray, tile: np.ndarray, bg: int, sep: int) -> None:
    th, tw = map(int, tile.shape)
    dh, dw = map(int, dst.shape)
    assert th <= dh and tw <= dw, "pattern must fit in the destination cell"
    top = int((dh - th) // 2)
    left = int((dw - tw) // 2)
    mask = (tile != int(bg)) & (tile != int(sep))
    rr, cc = np.nonzero(mask)
    dst[rr + top, cc + left] = tile[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count seed pixels inside each separated cell and use that count as a 1..4 index
    into the right-side sample strips. Remove the sample column in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    sep, row_lines, col_lines = _find_separator_color(g)

    assert len(row_lines) == 5, "expected four sample strips / four cell rows"
    row_heights = [int(row_lines[i + 1] - row_lines[i] - 1) for i in range(4)]
    assert len(set(row_heights)) == 1 and row_heights[0] > 0, "row cells must be equal-sized"
    cell_h = int(row_heights[0])

    region_widths = [int(col_lines[i + 1] - col_lines[i] - 1) for i in range(len(col_lines) - 1)]
    assert len(region_widths) >= 2, "need at least one normal column plus one sample column"
    cell_widths = region_widths[:-1]
    assert len(set(cell_widths)) == 1 and cell_widths[0] > 0, "normal cell columns must be equal-sized"
    cell_w = int(cell_widths[0])

    sample_c0 = int(col_lines[-2])
    sample_c1 = int(col_lines[-1])
    sample_patterns: list[np.ndarray] = []
    for i in range(4):
        r0 = int(row_lines[i])
        r1 = int(row_lines[i + 1])
        strip = g[r0 + 1 : r1, sample_c0 + 1 : sample_c1]
        assert strip.shape[0] == cell_h
        sample_patterns.append(_crop_non_bg(strip, bg=bg, sep=sep))

    normal_area = g[:, : sample_c0 + 1]
    sample_area = g[:, sample_c0 + 1 : sample_c1]
    normal_colors = {int(v) for v in normal_area.reshape(-1) if int(v) not in {bg, sep}}
    sample_colors = {int(v) for v in sample_area.reshape(-1) if int(v) not in {bg, sep}}
    seed_colors = normal_colors - sample_colors
    assert len(seed_colors) == 1, "seed color must be uniquely identifiable"
    seed = int(next(iter(seed_colors)))

    out = np.array(g[:, : sample_c0 + 1], dtype=int, copy=True)

    for ri in range(4):
        r0 = int(row_lines[ri])
        r1 = int(row_lines[ri + 1])
        for ci in range(len(col_lines) - 2):
            c0 = int(col_lines[ci])
            c1 = int(col_lines[ci + 1])
            cell = out[r0 + 1 : r1, c0 + 1 : c1]
            seed_count = int(np.count_nonzero(cell == int(seed)))
            cell[:, :] = int(bg)
            if 1 <= seed_count <= 4:
                _stamp_center(cell, sample_patterns[seed_count - 1], bg=bg, sep=sep)
            else:
                assert seed_count == 0, "seed count must be in 0..4"

    return out


# --------------- generator ---------------

NUM_ROWS = 4
CELL_H_RANGE = (5, 7)
CELL_W_RANGE = (5, 6)
SAMPLE_W_RANGE = (5, 7)
NUM_COLS_RANGE = (2, 3)
MAX_TOTAL_WIDTH = 30
MAX_TOTAL_HEIGHT = 34
RETRY_LIMIT = 300
PATTERN_BUILD_TRIES = 40

PATTERN_KINDS = (
    "diag_main",
    "diag_anti",
    "plus",
    "frame",
    "block",
    "x",
    "t_up",
    "l_corner",
    "diamond",
)


def _mask_signature(mask: np.ndarray) -> bytes:
    return np.packbits(mask.astype(np.uint8), axis=None).tobytes() + str(mask.shape).encode()


def _build_pattern_mask(kind: str, max_h: int, max_w: int, rng: random.Random) -> np.ndarray:
    max_h = max(3, int(max_h))
    max_w = max(3, int(max_w))

    if kind in {"diag_main", "diag_anti", "x", "diamond"}:
        side = rng.randint(3, min(max_h, max_w))
        if kind == "diamond" and side % 2 == 0:
            side = max(3, side - 1)
        arr = np.zeros((side, side), dtype=bool)
        if kind == "diag_main":
            arr[np.arange(side), np.arange(side)] = True
        elif kind == "diag_anti":
            arr[np.arange(side), np.arange(side - 1, -1, -1)] = True
        elif kind == "x":
            arr[np.arange(side), np.arange(side)] = True
            arr[np.arange(side), np.arange(side - 1, -1, -1)] = True
        else:
            mid = side // 2
            for r in range(side):
                reach = mid - abs(r - mid)
                arr[r, mid - reach : mid + reach + 1] = True
        return arr

    if kind == "plus":
        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        arr = np.zeros((h, w), dtype=bool)
        arr[h // 2, :] = True
        arr[:, w // 2] = True
        return arr

    if kind == "frame":
        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        arr = np.zeros((h, w), dtype=bool)
        arr[0, :] = True
        arr[-1, :] = True
        arr[:, 0] = True
        arr[:, -1] = True
        return arr

    if kind == "block":
        h = rng.randint(2, max_h)
        w = rng.randint(2, max_w)
        return np.ones((h, w), dtype=bool)

    if kind == "t_up":
        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        arr = np.zeros((h, w), dtype=bool)
        arr[0, :] = True
        arr[:, w // 2] = True
        return arr

    if kind == "l_corner":
        h = rng.randint(3, max_h)
        w = rng.randint(3, max_w)
        arr = np.zeros((h, w), dtype=bool)
        arr[0, :] = True
        arr[:, 0] = True
        return arr

    raise ValueError(f"unknown pattern kind: {kind}")


def _make_unique_patterns(max_h: int, max_w: int, rng: random.Random) -> list[np.ndarray] | None:
    chosen_kinds = rng.sample(PATTERN_KINDS, 4)
    patterns: list[np.ndarray] = []
    sigs: set[bytes] = set()
    for kind in chosen_kinds:
        placed = None
        for _ in range(PATTERN_BUILD_TRIES):
            mask = _build_pattern_mask(kind, max_h=max_h, max_w=max_w, rng=rng)
            sig = _mask_signature(mask)
            if sig not in sigs:
                sigs.add(sig)
                placed = mask
                break
        if placed is None:
            return None
        patterns.append(placed)
    return patterns


def _unique_dominant_color(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    if not counts:
        return False
    most_common = counts.most_common(2)
    if len(most_common) == 1:
        return True
    return int(most_common[0][1]) > int(most_common[1][1])


def _expected_output(
    grid: np.ndarray,
    row_lines: list[int],
    col_lines: list[int],
    bg: int,
    pattern_tiles: list[np.ndarray],
    pattern_colors: list[int],
    counts: list[list[int]],
) -> np.ndarray:
    sample_c0 = int(col_lines[-2])
    out = np.array(grid[:, : sample_c0 + 1], dtype=int, copy=True)
    for ri in range(NUM_ROWS):
        r0 = int(row_lines[ri])
        r1 = int(row_lines[ri + 1])
        for ci in range(len(counts[ri])):
            c0 = int(col_lines[ci])
            c1 = int(col_lines[ci + 1])
            cell = out[r0 + 1 : r1, c0 + 1 : c1]
            cell[:, :] = int(bg)
            n = int(counts[ri][ci])
            if 1 <= n <= 4:
                tile = np.where(pattern_tiles[n - 1], int(pattern_colors[n - 1]), int(bg))
                _stamp_center(cell, tile, bg=bg, sep=-1)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    cell_h = rng.randint(*CELL_H_RANGE)
    cell_w = rng.randint(*CELL_W_RANGE)
    sample_w = rng.randint(*SAMPLE_W_RANGE)
    num_cols = rng.randint(*NUM_COLS_RANGE)

    total_h = NUM_ROWS * cell_h + (NUM_ROWS + 1)
    total_w = num_cols * cell_w + sample_w + (num_cols + 2)
    if total_h > MAX_TOTAL_HEIGHT or total_w > MAX_TOTAL_WIDTH:
        return None

    max_pat_h = cell_h - 2
    max_pat_w = min(cell_w, sample_w) - 2
    if max_pat_h < 3 or max_pat_w < 3:
        return None

    pattern_tiles = _make_unique_patterns(max_pat_h, max_pat_w, rng)
    if pattern_tiles is None:
        return None

    palette = rng.sample(list(range(10)), 7)
    bg, sep, seed = map(int, palette[:3])
    pattern_colors = [int(c) for c in palette[3:]]

    row_lines = [int(i * (cell_h + 1)) for i in range(NUM_ROWS + 1)]
    col_lines = [0]
    cur = 0
    for _ in range(num_cols):
        cur += cell_w + 1
        col_lines.append(int(cur))
    cur += sample_w + 1
    col_lines.append(int(cur))

    grid = np.full((total_h, total_w), int(bg), dtype=int)
    grid[row_lines, :] = int(sep)
    grid[:, col_lines] = int(sep)

    total_cells = NUM_ROWS * num_cols
    forced = [0, 1, 2, 3, 4]
    counts_flat = forced + [rng.randint(0, 4) for _ in range(total_cells - len(forced))]
    rng.shuffle(counts_flat)
    counts = [counts_flat[i * num_cols : (i + 1) * num_cols] for i in range(NUM_ROWS)]

    for ri in range(NUM_ROWS):
        r0 = int(row_lines[ri])
        r1 = int(row_lines[ri + 1])
        for ci in range(num_cols):
            c0 = int(col_lines[ci])
            c1 = int(col_lines[ci + 1])
            n = int(counts[ri][ci])
            if n == 0:
                continue
            positions = [(rr, cc) for rr in range(cell_h) for cc in range(cell_w)]
            for rr, cc in rng.sample(positions, n):
                grid[r0 + 1 + rr, c0 + 1 + cc] = int(seed)

    sample_c0 = int(col_lines[-2])
    sample_c1 = int(col_lines[-1])
    for idx in range(4):
        r0 = int(row_lines[idx])
        r1 = int(row_lines[idx + 1])
        strip = grid[r0 + 1 : r1, sample_c0 + 1 : sample_c1]
        strip[:, :] = int(bg)
        tile = np.where(pattern_tiles[idx], int(pattern_colors[idx]), int(bg))
        _stamp_center(strip, tile, bg=bg, sep=-1)

    if not _unique_dominant_color(grid):
        return None

    try:
        expected = _expected_output(
            grid=grid,
            row_lines=row_lines,
            col_lines=col_lines,
            bg=bg,
            pattern_tiles=pattern_tiles,
            pattern_colors=pattern_colors,
            counts=counts,
        )
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
