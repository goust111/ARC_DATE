import numpy as np


MAROON = 9


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _key(a):
    return np.ascontiguousarray(a).tobytes()


def _locate_misprint(grid):
    h, w = grid.shape
    best = None
    for th in _divisors(h):
        for tw in _divisors(w):
            tile_count = (h // th) * (w // tw)
            if tile_count < 3 or (th == h and tw == w):
                continue

            counts = {}
            exemplars = {}
            positions = {}
            for r in range(0, h, th):
                for c in range(0, w, tw):
                    tile = grid[r:r + th, c:c + tw]
                    k = _key(tile)
                    counts[k] = counts.get(k, 0) + 1
                    if k not in exemplars:
                        exemplars[k] = tile.copy()
                        positions[k] = []
                    positions[k].append((r, c))

            max_count = max(counts.values())
            for k, tile in exemplars.items():
                flipped = tile[:, ::-1]
                if np.array_equal(tile, flipped):
                    continue
                fk = _key(flipped)
                if counts.get(fk, 0) != 1 or counts[k] < 2:
                    continue
                if counts[k] != max_count:
                    continue

                r, c = positions[fk][0]
                score = (th * tw, counts[k], tile_count)
                candidate = (score, th, tw, r, c)
                if best is None or score > best[0]:
                    best = candidate

    if best is None:
        return None
    _, th, tw, r, c = best
    return th, tw, r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Correct the one horizontally mirrored tile and mark its border.

    Input structure:
    The grid is a rectangular mosaic split perfectly into same-size tiles.
    Each tile contains a small non-symmetric color pattern.  Most tiles repeat
    one correct pattern, sometimes with a few other repeated correct patterns,
    and exactly one tile is the left-right mirror image of the most frequent
    correct pattern.

    Rule:
    Try all uniform tile sizes that divide the grid.  For each tiling, count
    exact tile patterns and look for a most frequent pattern whose horizontal
    mirror appears exactly once.  The unique mirrored tile is flipped back to
    the correct orientation.  Then the outer one-cell border of that tile is
    recolored maroon; if the tile is too small, this naturally covers the whole
    tile.  All other cells remain unchanged.

    Key insight:
    The repeated tile lattice must be discovered first; within that lattice,
    the anomaly is not a new color but a hidden horizontal reflection of the
    dominant tile pattern.
    """
    grid = np.asarray(grid)
    out = grid.copy()
    found = _locate_misprint(grid)
    if found is None:
        return out

    th, tw, r, c = found
    corrected = grid[r:r + th, c:c + tw][:, ::-1].copy()
    out[r:r + th, c:c + tw] = corrected
    out[r, c:c + tw] = MAROON
    out[r + th - 1, c:c + tw] = MAROON
    out[r:r + th, c] = MAROON
    out[r:r + th, c + tw - 1] = MAROON
    return out


def _random_pattern(rng, th, tw, colors, existing):
    for _ in range(500):
        arr = rng.choice(colors, size=(th, tw)).astype(int)
        if len(set(arr.ravel())) < 2:
            continue
        if np.array_equal(arr, arr[:, ::-1]):
            continue
        k = _key(arr)
        fk = _key(arr[:, ::-1])
        if k in existing or fk in existing:
            continue
        return arr
    arr = np.full((th, tw), colors[0], dtype=int)
    split = max(1, tw // 2)
    arr[:, :split] = colors[0]
    arr[:, split:] = colors[1]
    arr[0, 0] = colors[-1]
    if np.array_equal(arr, arr[:, ::-1]):
        arr[0, 0] = colors[0]
        arr[0, -1] = colors[-1]
    return arr


def _build_candidate(rng):
    th = int(rng.choice([2, 3, 4, 5, 6]))
    tw = int(rng.choice([2, 3, 4, 5, 6]))
    max_rows = max(2, min(7, 30 // th))
    max_cols = max(2, min(7, 30 // tw))
    tr = int(rng.integers(2, max_rows + 1))
    tc = int(rng.integers(2, max_cols + 1))
    if tr * tc < 6:
        if tr < max_rows:
            tr += 1
        elif tc < max_cols:
            tc += 1

    total = tr * tc
    max_patterns = min(3, max(1, total - 3))
    pattern_count = int(rng.integers(1, max_patterns + 1))

    palette = rng.choice(np.arange(0, 9), size=int(rng.integers(2, 4)), replace=False)
    existing = set()
    patterns = []
    for _ in range(pattern_count):
        pat = _random_pattern(rng, th, tw, palette, existing)
        patterns.append(pat)
        existing.add(_key(pat))
        existing.add(_key(pat[:, ::-1]))

    slots = total - 1
    labels = None
    weights = np.array([0.64, 0.23, 0.13][:pattern_count], dtype=float)
    weights /= weights.sum()
    for _ in range(300):
        trial = rng.choice(np.arange(pattern_count), size=slots, p=weights)
        counts = np.bincount(trial, minlength=pattern_count)
        if counts[0] >= 2 and counts[0] == counts.max() and np.sum(counts == counts[0]) == 1:
            if pattern_count == 1 or np.all(counts[1:] > 0):
                labels = list(map(int, trial))
                break
    if labels is None:
        labels = [0] * slots
        for i in range(1, pattern_count):
            labels[i - 1] = i
        rng.shuffle(labels)

    wrong_index = int(rng.integers(0, total))
    labels.insert(wrong_index, -1)

    grid = np.empty((tr * th, tc * tw), dtype=int)
    for idx, label in enumerate(labels):
        rr = (idx // tc) * th
        cc = (idx % tc) * tw
        tile = patterns[0][:, ::-1] if label == -1 else patterns[label]
        grid[rr:rr + th, cc:cc + tw] = tile
    return grid, (th, tw, (wrong_index // tc) * th, (wrong_index % tc) * tw)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(200):
        grid, target = _build_candidate(rng)
        found = _locate_misprint(grid)
        if found == target:
            return grid
    return grid
