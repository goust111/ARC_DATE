import random
import sys
from collections import Counter

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import Color, bbox, scale_sprite


BLACK = Color.BLACK
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
TILE_H_RANGE = (2, 5)
TILE_W_RANGE = (2, 5)
SCALE_RANGE = (2, 4)
REPEAT_EXTRA_RANGE = (2, 4)
HOLE_BLOB_COUNT_RANGE = (1, 3)
HOLE_BLOB_SIZE_RANGE = (1, 6)
WRONG_BLOCK_LIMIT = 4


def _min_period(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    for th in range(1, h + 1):
        for tw in range(1, w + 1):
            if h % th != 0 or w % tw != 0:
                continue
            if np.array_equal(tile, tile[rows % th, cols % tw]):
                return th, tw
    return h, w


def _phase_counts(shape: tuple[int, int], th: int, tw: int) -> np.ndarray:
    h, w = map(int, shape)
    counts = np.zeros((th, tw), dtype=int)
    np.add.at(counts, (np.arange(h)[:, None] % th, np.arange(w)[None, :] % tw), 1)
    return counts


def _fill_from_tile(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    th, tw = map(int, tile.shape)
    rows = np.arange(shape[0])[:, None] % th
    cols = np.arange(shape[1])[None, :] % tw
    return tile[rows, cols]


def _infer_tile_exact(
    grid: np.ndarray, th: int, tw: int, *, exclude_bbox: tuple[int, int, int, int] | None = None
) -> np.ndarray | None:
    known = grid != BLACK
    if exclude_bbox is not None:
        r0, c0, r1, c1 = exclude_bbox
        known[r0 : r1 + 1, c0 : c1 + 1] = False

    tile = np.full((th, tw), -1, dtype=int)
    seen = np.zeros((th, tw), dtype=bool)
    for r, c in np.argwhere(known):
        pr, pc = int(r) % th, int(c) % tw
        value = int(grid[r, c])
        if tile[pr, pc] == -1:
            tile[pr, pc] = value
            seen[pr, pc] = True
        elif tile[pr, pc] != value:
            return None
    if not np.all(seen):
        return None
    return tile


def _majority_tile(grid: np.ndarray, th: int, tw: int) -> np.ndarray | None:
    buckets: list[list[int]] = [[[] for _ in range(tw)] for _ in range(th)]
    for r, c in np.argwhere(grid != BLACK):
        buckets[int(r) % th][int(c) % tw].append(int(grid[r, c]))

    tile = np.full((th, tw), -1, dtype=int)
    for pr in range(th):
        for pc in range(tw):
            if not buckets[pr][pc]:
                return None
            counts = Counter(buckets[pr][pc]).most_common(2)
            if len(counts) > 1 and counts[0][1] == counts[1][1]:
                return None
            tile[pr, pc] = int(counts[0][0])
    return tile


def _compress_uniform_blocks(window: np.ndarray, k: int) -> np.ndarray | None:
    h, w = map(int, window.shape)
    if h % k != 0 or w % k != 0:
        return None
    out = np.full((h // k, w // k), -1, dtype=int)
    for br in range(h // k):
        for bc in range(w // k):
            block = window[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            if np.unique(block).size != 1:
                return None
            out[br, bc] = int(block[0, 0])
    return out


def _candidate_models(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    known = g != BLACK
    models: list[dict] = []

    for th in range(1, h // 2 + 1):
        for tw in range(1, w // 2 + 1):
            majority_tile = _majority_tile(g, th, tw)
            if majority_tile is None:
                continue

            wallpaper_guess = _fill_from_tile(g.shape, majority_tile)
            mismatch = known & (g != wallpaper_guess)
            win_bbox = bbox(mismatch)
            if win_bbox is None:
                continue

            r0, c0, r1, c1 = map(int, win_bbox)
            win_h = r1 - r0 + 1
            win_w = c1 - c0 + 1
            if win_h % th != 0 or win_w % tw != 0:
                continue
            k_r = win_h // th
            k_c = win_w // tw
            if k_r != k_c or k_r < 2:
                continue
            if np.any(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
                continue

            tile = _infer_tile_exact(g, th, tw, exclude_bbox=win_bbox)
            if tile is None:
                continue

            wallpaper = _fill_from_tile(g.shape, tile)
            mismatch2 = known & (g != wallpaper)
            if bbox(mismatch2) != win_bbox:
                continue

            block_grid = _compress_uniform_blocks(g[r0 : r1 + 1, c0 : c1 + 1], k_r)
            if block_grid is None or block_grid.shape != tile.shape:
                continue

            wrong_blocks = int(np.count_nonzero(block_grid != tile))
            expected_window = scale_sprite(tile, k_r)
            out = wallpaper.copy()
            out[r0 : r1 + 1, c0 : c1 + 1] = expected_window

            models.append(
                {
                    "tile": tile,
                    "bbox": win_bbox,
                    "k": k_r,
                    "wrong_blocks": wrong_blocks,
                    "out": out,
                }
            )
    return models


def _select_model(grid: np.ndarray) -> dict:
    models = _candidate_models(grid)
    assert models, "expected at least one valid wallpaper/window model"

    best_area = min(int(m["tile"].shape[0] * m["tile"].shape[1]) for m in models)
    best = [m for m in models if int(m["tile"].shape[0] * m["tile"].shape[1]) == best_area]
    best_wrong = min(int(m["wrong_blocks"]) for m in best)
    best = [m for m in best if int(m["wrong_blocks"]) == best_wrong]
    assert len(best) == 1, "expected a unique minimal valid model"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the minimal wallpaper tile and use it to heal both black holes and a single enlarged corrupted window.

    Input structure:
    - The grid is mostly a periodic wallpaper built from one unknown minimal tile, with a few black holes outside a special region.
    - Exactly one rectangular window contains one integer-upscaled copy of that minimal tile instead of the wallpaper continuation.
    - The window has no black cells, and some of its `k x k` color blocks may be painted to the wrong solid color.

    Rule:
    - Ignore the black holes and the special window while inferring the unique minimal wallpaper tile from the remaining cells.
    - Fill every black hole outside the window with the wallpaper color at the matching tile phase.
    - Detect the window's scale factor from its size relative to the tile, then replace the whole window by the correctly upscaled tile.

    Key insight:
    - The enlarged window cannot be repaired by extending the local wallpaper phase; the only reliable reference is the minimal tile inferred from the untouched wallpaper outside the window.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    model = _select_model(g)
    return model["out"]


def _random_tile(th: int, tw: int) -> np.ndarray | None:
    for _ in range(200):
        color_count = random.randint(2, min(5, th * tw, len(PALETTE)))
        colors = random.sample(PALETTE, color_count)
        tile = np.random.choice(colors, size=(th, tw)).astype(int)
        if len(np.unique(tile)) < 2:
            continue
        if _min_period(tile) != (th, tw):
            continue
        return tile
    return None


def _grow_blob(
    available: np.ndarray, phase_budget: np.ndarray, th: int, tw: int, max_size: int
) -> list[tuple[int, int]] | None:
    starts = [(int(r), int(c)) for r, c in np.argwhere(available)]
    random.shuffle(starts)
    for start in starts:
        if phase_budget[start[0] % th, start[1] % tw] <= 0:
            continue
        blob = [start]
        used = {start}
        phase_budget[start[0] % th, start[1] % tw] -= 1
        target = random.randint(1, max_size)
        while len(blob) < target:
            frontier = []
            for r, c in blob:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < available.shape[0] and 0 <= cc < available.shape[1]):
                        continue
                    if (rr, cc) in used or not available[rr, cc]:
                        continue
                    if phase_budget[rr % th, cc % tw] <= 0:
                        continue
                    frontier.append((rr, cc))
            if not frontier:
                break
            cell = random.choice(frontier)
            blob.append(cell)
            used.add(cell)
            phase_budget[cell[0] % th, cell[1] % tw] -= 1
        return blob
    return None


def _make_holes(
    shape: tuple[int, int], window_bbox: tuple[int, int, int, int], th: int, tw: int, k: int
) -> np.ndarray | None:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = window_bbox
    outside = np.ones((h, w), dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    total_counts = _phase_counts(shape, th, tw)
    outside_counts = total_counts - (k * k)
    phase_budget = outside_counts - (k * k + 1)
    if np.any(phase_budget < 0):
        return None

    blob_count = random.randint(*HOLE_BLOB_COUNT_RANGE)
    for _ in range(blob_count):
        available = outside & ~mask
        blob = _grow_blob(available, phase_budget, th, tw, HOLE_BLOB_SIZE_RANGE[1])
        if not blob:
            break
        if len(blob) < HOLE_BLOB_SIZE_RANGE[0]:
            return None
        for r, c in blob:
            mask[r, c] = True

    return mask if mask.any() else None


def _needed_side_blocks(tile: np.ndarray, wallpaper_slice: np.ndarray, k: int) -> set[tuple[int, int]]:
    mismatch = scale_sprite(tile, k) != wallpaper_slice
    h, w = mismatch.shape
    out: set[tuple[int, int]] = set()
    if not mismatch[0].any():
        out.add((0, random.randrange(tile.shape[1])))
    if not mismatch[h - 1].any():
        out.add((tile.shape[0] - 1, random.randrange(tile.shape[1])))
    if not mismatch[:, 0].any():
        out.add((random.randrange(tile.shape[0]), 0))
    if not mismatch[:, w - 1].any():
        out.add((random.randrange(tile.shape[0]), tile.shape[1] - 1))
    return out


def _paint_wrong_blocks(window: np.ndarray, tile: np.ndarray, k: int, wrong_cells: list[tuple[int, int]]) -> None:
    pool = list(PALETTE)
    for tr, tc in wrong_cells:
        choices = [c for c in pool if c != int(tile[tr, tc])]
        color = random.choice(choices)
        window[tr * k : (tr + 1) * k, tc * k : (tc + 1) * k] = color


def _true_model_checks(
    inp: np.ndarray,
    out: np.ndarray,
    *,
    tile: np.ndarray,
    window_bbox: tuple[int, int, int, int],
    k: int,
    holes: np.ndarray,
    wrong_cells: list[tuple[int, int]],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    models = _candidate_models(inp)
    tile_area = int(tile.shape[0] * tile.shape[1])
    best = [m for m in models if int(m["tile"].shape[0] * m["tile"].shape[1]) == tile_area]
    if len(best) != 1:
        return False
    if tuple(best[0]["bbox"]) != tuple(window_bbox) or int(best[0]["k"]) != int(k):
        return False

    wallpaper = _fill_from_tile(inp.shape, tile)
    wallpaper_only = wallpaper.copy()
    if np.array_equal(out, wallpaper_only):
        return False
    if not holes.any() or not wrong_cells:
        return False
    if np.array_equal(inp, out):
        return False

    filled_only = inp.copy()
    filled_only[holes] = wallpaper[holes]
    if np.array_equal(filled_only, out):
        return False

    repaired_window_only = wallpaper.copy()
    r0, c0, r1, c1 = window_bbox
    repaired_window_only[r0 : r1 + 1, c0 : c1 + 1] = scale_sprite(tile, k)
    repaired_window_only[holes] = BLACK
    if np.array_equal(repaired_window_only, out):
        return False

    inferred = _infer_tile_exact(inp, tile.shape[0], tile.shape[1], exclude_bbox=window_bbox)
    if inferred is None or not np.array_equal(inferred, tile):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    th = random.randint(*TILE_H_RANGE)
    tw = random.randint(*TILE_W_RANGE)
    tile = _random_tile(th, tw)
    if tile is None:
        return None

    feasible_k = [
        k
        for k in range(SCALE_RANGE[0], SCALE_RANGE[1] + 1)
        if th * (k + REPEAT_EXTRA_RANGE[0]) <= 30 and tw * (k + REPEAT_EXTRA_RANGE[0]) <= 30
    ]
    if not feasible_k:
        return None
    k = random.choice(feasible_k)

    rep_r = random.randint(k + REPEAT_EXTRA_RANGE[0], min(k + REPEAT_EXTRA_RANGE[1], 30 // th))
    rep_c = random.randint(k + REPEAT_EXTRA_RANGE[0], min(k + REPEAT_EXTRA_RANGE[1], 30 // tw))
    h = th * rep_r
    w = tw * rep_c

    wallpaper = _fill_from_tile((h, w), tile)
    window_h = th * k
    window_w = tw * k
    r0 = random.randint(0, h - window_h)
    c0 = random.randint(0, w - window_w)
    r1 = r0 + window_h - 1
    c1 = c0 + window_w - 1
    window_bbox = (r0, c0, r1, c1)

    holes = _make_holes((h, w), window_bbox, th, tw, k)
    if holes is None:
        return None

    inp = wallpaper.copy()
    inp[holes] = BLACK

    clean_window = scale_sprite(tile, k)
    wallpaper_slice = wallpaper[r0 : r1 + 1, c0 : c1 + 1]
    wrong_needed = _needed_side_blocks(tile, wallpaper_slice, k)
    all_cells = [(tr, tc) for tr in range(th) for tc in range(tw)]
    random.shuffle(all_cells)
    max_wrong = min(WRONG_BLOCK_LIMIT, th * tw)
    wrong_count = random.randint(max(1, len(wrong_needed)), max(1, max_wrong))
    wrong_set = set(wrong_needed)
    for cell in all_cells:
        if len(wrong_set) >= wrong_count:
            break
        wrong_set.add(cell)
    wrong_cells = list(wrong_set)

    window = clean_window.copy()
    _paint_wrong_blocks(window, tile, k, wrong_cells)
    inp[r0 : r1 + 1, c0 : c1 + 1] = window

    mismatch = inp != wallpaper
    mismatch[holes] = False
    if bbox(mismatch) != window_bbox:
        return None

    block_grid = _compress_uniform_blocks(window, k)
    if block_grid is None:
        return None
    if int(np.count_nonzero(block_grid != tile)) != len(wrong_cells):
        return None

    out = wallpaper.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = clean_window

    if not _true_model_checks(
        inp,
        out,
        tile=tile,
        window_bbox=window_bbox,
        k=k,
        holes=holes,
        wrong_cells=wrong_cells,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for fable/1/006/2")
