import random
from collections import Counter

import numpy as np


BG = 0
NOISE = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ALLOWED_TILE_SHAPES = ((2, 2), (2, 3), (3, 2))
GEN_ATTEMPTS = 400


def _tile_fill(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    h, w = map(int, shape)
    th, tw = tile.shape
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            out[r, c] = int(tile[r % th, c % tw])
    return out


def _shear_down_left(rect: np.ndarray) -> np.ndarray:
    h, w = rect.shape
    out = np.zeros((h, w + h - 1), dtype=int)
    for r in range(h):
        start = h - 1 - r
        out[r, start : start + w] = rect[r]
    return out


def _infer_shear(grid: np.ndarray) -> tuple[list[int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    starts: list[int] = []
    widths: list[int] = []

    for r in range(h):
        cols = np.flatnonzero(g[r] != BG)
        assert cols.size > 0, "every row must contain sheared foreground"
        c0 = int(cols[0])
        c1 = int(cols[-1])
        assert np.all(g[r, c0 : c1 + 1] != BG), "row content must be contiguous"
        starts.append(c0)
        widths.append(c1 - c0 + 1)

    content_w = widths[0]
    assert all(wi == content_w for wi in widths), "sheared rows must preserve width"
    expected_starts = [h - 1 - r for r in range(h)]
    assert starts == expected_starts, "expected down-left shear offsets"
    assert w == content_w + h - 1, "canvas width must match shear geometry"
    return starts, content_w


def _unshear(grid: np.ndarray, starts: list[int], content_w: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h = g.shape[0]
    rect = np.zeros((h, content_w), dtype=int)
    for r, start in enumerate(starts):
        rect[r] = g[r, start : start + content_w]
    return rect


def _majority_color(values: list[int]) -> int | None:
    counts = Counter(values)
    best = counts.most_common()
    if len(best) > 1 and best[0][1] == best[1][1]:
        return None
    return int(best[0][0])


def _evaluate_candidate(rect: np.ndarray, th: int, tw: int) -> tuple[np.ndarray, int] | None:
    h, w = rect.shape
    if h // th < 2 or w // tw < 2:
        return None

    tile = np.zeros((th, tw), dtype=int)
    mismatch = 0
    for rr in range(th):
        for cc in range(tw):
            values = [int(rect[r, c]) for r in range(rr, h, th) for c in range(cc, w, tw)]
            maj = _majority_color(values)
            if maj is None:
                return None
            tile[rr, cc] = maj
            mismatch += sum(v != maj for v in values)
    return tile, int(mismatch)


def _choose_tile(rect: np.ndarray, noise_budget: int) -> np.ndarray:
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for th, tw in ALLOWED_TILE_SHAPES:
        result = _evaluate_candidate(rect, th, tw)
        if result is None:
            continue
        tile, mismatch = result
        if mismatch <= noise_budget:
            candidates.append((th * tw, th, tw, tile))

    assert candidates, "no tile shape satisfies the noise bound"
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best = candidates[0]
    same_size = [item for item in candidates if item[:3] == best[:3]]
    assert len(same_size) == 1
    return best[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse a noisy down-left shear, recover the periodic tile, and shear the repaired pattern back.

    Input structure:
    - Black background surrounds one multicolor foreground parallelogram whose rows are contiguous.
    - Row `r` starts at column `H-1-r`, so the left edge forms a descending diagonal.
    - Inside the sheared shape, a 2x2, 2x3, or 3x2 periodic rectangle has been corrupted by 3..5 grey noise pixels.

    Rule:
    - Infer the common row width and down-left shear offsets from the non-background row segments.
    - Undo the shear to obtain the hidden rectangle, then test the allowed tile sizes with residue-class majority voting.
    - Pick the smallest valid tile whose disagreement count does not exceed the grey-pixel count, rebuild the clean rectangle, and apply the same down-left shear again.

    Key insight:
    The periodic structure is aligned only in the unsheared rectangle; tile inference must happen after reversing the shear and before reapplying it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    starts, content_w = _infer_shear(g)
    rect = _unshear(g, starts, content_w)
    noise_budget = int(np.count_nonzero(rect == NOISE))
    tile = _choose_tile(rect, noise_budget)
    repaired = _tile_fill(rect.shape, tile)
    return _shear_down_left(repaired)


def _row_and_col_structure_ok(tile: np.ndarray) -> bool:
    row_patterns = {tuple(map(int, row)) for row in tile}
    col_patterns = {tuple(map(int, tile[:, c])) for c in range(tile.shape[1])}
    return len(row_patterns) > 1 and len(col_patterns) > 1


def _pick_base_tile(shape: tuple[int, int]) -> np.ndarray | None:
    th, tw = shape
    for _ in range(200):
        n_colors = random.choice((2, 3))
        colors = random.sample(PALETTE, n_colors)
        tile = np.array(random.choices(colors, k=th * tw), dtype=int).reshape(th, tw)
        if len(set(map(int, tile.ravel()))) != n_colors:
            continue
        if not _row_and_col_structure_ok(tile):
            continue
        ok = True
        for ph in range(1, th + 1):
            for pw in range(1, tw + 1):
                if ph == th and pw == tw:
                    continue
                if th % ph != 0 or tw % pw != 0:
                    continue
                if np.array_equal(tile, _tile_fill((th, tw), tile[:ph, :pw])):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return tile
    return None


def _allowed_repeats(shape: tuple[int, int]) -> list[tuple[int, int]]:
    th, tw = shape
    repeats: list[tuple[int, int]] = []
    for rh in range(2, 6):
        for rw in range(2, 6):
            h = th * rh
            w = tw * rw
            if h <= 30 and w + h - 1 <= 30:
                repeats.append((rh, rw))
    random.shuffle(repeats)
    return repeats


def _add_noise(rect: np.ndarray, tile_shape: tuple[int, int]) -> np.ndarray | None:
    th, tw = tile_shape
    noisy = rect.copy()
    residues = [(rr, cc) for rr in range(th) for cc in range(tw)]
    random.shuffle(residues)

    max_noise = min(5, len(residues))
    noise_count = random.randint(3, max_noise)
    chosen_residues = residues[:noise_count]

    for rr, cc in chosen_residues:
        positions = [(r, c) for r in range(rr, rect.shape[0], th) for c in range(cc, rect.shape[1], tw)]
        r, c = random.choice(positions)
        noisy[r, c] = NOISE
    return noisy


def _validate_instance(base_tile: np.ndarray, clean_rect: np.ndarray, noisy_rect: np.ndarray) -> bool:
    noise_budget = int(np.count_nonzero(noisy_rect == NOISE))
    expected = _tile_fill(clean_rect.shape, base_tile)
    if not np.array_equal(clean_rect, expected):
        return False

    valid_shapes: list[tuple[int, int]] = []
    for th, tw in ALLOWED_TILE_SHAPES:
        result = _evaluate_candidate(noisy_rect, th, tw)
        if result is None:
            continue
        tile, mismatch = result
        if mismatch <= noise_budget:
            rebuilt = _tile_fill(noisy_rect.shape, tile)
            if np.array_equal(rebuilt, clean_rect):
                valid_shapes.append((th, tw))
    return valid_shapes == [tuple(map(int, base_tile.shape))]


def _gen_attempt() -> np.ndarray | None:
    tile_shape = random.choice(ALLOWED_TILE_SHAPES)
    base_tile = _pick_base_tile(tile_shape)
    if base_tile is None:
        return None

    for rh, rw in _allowed_repeats(tile_shape):
        clean_rect = _tile_fill((tile_shape[0] * rh, tile_shape[1] * rw), base_tile)
        noisy_rect = _add_noise(clean_rect, tile_shape)
        if noisy_rect is None:
            continue
        if not _validate_instance(base_tile, clean_rect, noisy_rect):
            continue

        sheared = _shear_down_left(noisy_rect)
        solved = main(sheared)
        target = _shear_down_left(clean_rect)
        if np.array_equal(solved, target):
            return sheared
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid shear-tile-noise input")
