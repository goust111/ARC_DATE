from __future__ import annotations

import numpy as np


def _mode_preserve_order(values: np.ndarray) -> int:
    flat = [int(v) for v in np.asarray(values).ravel()]
    counts: dict[int, int] = {}
    order: list[int] = []
    for value in flat:
        if value not in counts:
            counts[value] = 1
            order.append(value)
        else:
            counts[value] += 1
    best = max(counts.values())
    for value in order:
        if counts[value] == best:
            return value
    raise AssertionError("mode computation failed")


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _repeat_divisors(n: int) -> list[int]:
    divs = [d for d in _divisors(int(n)) if int(n) // int(d) >= 2]
    return divs or [int(n)]


def _majority_tile(grid: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    tile = np.zeros((int(tile_h), int(tile_w)), dtype=int)
    for r in range(int(tile_h)):
        for c in range(int(tile_w)):
            tile[r, c] = _mode_preserve_order(grid[r::tile_h, c::tile_w])
    return tile


def _infer_tile(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    h, w = map(int, grid.shape)
    best_key: tuple[int, int, int, int] | None = None
    best_tile: np.ndarray | None = None
    best_shape: tuple[int, int] | None = None

    for tile_h in _repeat_divisors(h):
        for tile_w in _repeat_divisors(w):
            tile = _majority_tile(grid, int(tile_h), int(tile_w))
            rebuilt = np.tile(tile, (h // int(tile_h), w // int(tile_w)))
            mismatches = int(np.count_nonzero(rebuilt != grid))
            key = (mismatches, int(tile_h) * int(tile_w), int(tile_h), int(tile_w))
            if best_key is None or key < best_key:
                best_key = key
                best_tile = tile
                best_shape = (int(tile_h), int(tile_w))

    assert best_tile is not None and best_shape is not None
    return best_shape[0], best_shape[1], best_tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find the smallest repeated tile, compare every copy against the clean template,
    and compress the contamination labels into one cell per tile copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_preserve_order(g)
    tile_h, tile_w, tile = _infer_tile(g)

    n_rows = g.shape[0] // tile_h
    n_cols = g.shape[1] // tile_w
    out = np.full((int(n_rows), int(n_cols)), int(bg), dtype=int)

    for tr in range(int(n_rows)):
        for tc in range(int(n_cols)):
            block = g[
                tr * tile_h : (tr + 1) * tile_h,
                tc * tile_w : (tc + 1) * tile_w,
            ]
            diff = block != tile
            if bool(diff.any()):
                out[tr, tc] = _mode_preserve_order(block[diff])

    return out


def _minimal_subtile_shape(tile: np.ndarray) -> tuple[int, int]:
    h, w = map(int, tile.shape)
    best = (h, w)
    best_key = (h * w, h, w)
    for sub_h in _divisors(h):
        for sub_w in _divisors(w):
            base = tile[:sub_h, :sub_w]
            rebuilt = np.tile(base, (h // int(sub_h), w // int(sub_w)))
            if np.array_equal(rebuilt, tile):
                key = (int(sub_h) * int(sub_w), int(sub_h), int(sub_w))
                if key < best_key:
                    best = (int(sub_h), int(sub_w))
                    best_key = key
    return best


def _choose_color(rng: np.random.Generator, exclude: set[int]) -> int:
    choices = [c for c in range(10) if int(c) not in exclude]
    assert choices
    return int(rng.choice(choices))


def _random_tile(
    rng: np.random.Generator,
    tile_h: int,
    tile_w: int,
    bg: int,
    tile_colors: list[int],
) -> np.ndarray | None:
    for _ in range(200):
        tile = np.full((int(tile_h), int(tile_w)), int(bg), dtype=int)
        n_ops = int(rng.integers(2, 6))

        for _ in range(n_ops):
            color = int(rng.choice(tile_colors))
            kind = int(rng.integers(0, 5))

            if kind == 0:
                r = int(rng.integers(0, tile_h))
                c = int(rng.integers(0, tile_w))
                tile[r, c] = color
            elif kind == 1:
                r = int(rng.integers(0, tile_h))
                c0 = int(rng.integers(0, tile_w))
                c1 = int(rng.integers(c0, tile_w))
                tile[r, c0 : c1 + 1] = color
            elif kind == 2:
                c = int(rng.integers(0, tile_w))
                r0 = int(rng.integers(0, tile_h))
                r1 = int(rng.integers(r0, tile_h))
                tile[r0 : r1 + 1, c] = color
            elif kind == 3:
                rect_h = int(rng.integers(1, min(3, tile_h) + 1))
                rect_w = int(rng.integers(1, min(3, tile_w) + 1))
                r0 = int(rng.integers(0, tile_h - rect_h + 1))
                c0 = int(rng.integers(0, tile_w - rect_w + 1))
                tile[r0 : r0 + rect_h, c0 : c0 + rect_w] = color
            else:
                r = int(rng.integers(0, tile_h))
                c = int(rng.integers(0, tile_w))
                r1 = int(rng.integers(r, tile_h))
                c1 = int(rng.integers(c, tile_w))
                tile[r : r1 + 1, c] = color
                tile[r, c : c1 + 1] = color

        if bool(rng.random() < 0.35):
            cut_h = int(rng.integers(1, min(2, tile_h) + 1))
            cut_w = int(rng.integers(1, min(2, tile_w) + 1))
            r0 = int(rng.integers(0, tile_h - cut_h + 1))
            c0 = int(rng.integers(0, tile_w - cut_w + 1))
            tile[r0 : r0 + cut_h, c0 : c0 + cut_w] = bg

        fg_mask = tile != int(bg)
        fg_count = int(np.count_nonzero(fg_mask))
        if fg_count < 2:
            continue
        if fg_count > tile.size // 2:
            continue
        if tile_h > 1 and int(np.count_nonzero(fg_mask.any(axis=1))) < 2:
            continue
        if tile_w > 1 and int(np.count_nonzero(fg_mask.any(axis=0))) < 2:
            continue
        if _minimal_subtile_shape(tile) != (int(tile_h), int(tile_w)):
            continue

        bg_count = int(np.count_nonzero(tile == int(bg)))
        other_counts = [
            int(np.count_nonzero(tile == color))
            for color in np.unique(tile)
            if int(color) != int(bg)
        ]
        if not other_counts or bg_count <= max(other_counts):
            continue
        return tile

    return None


def _strict_most_common(arr: np.ndarray, color: int) -> bool:
    values, counts = np.unique(arr, return_counts=True)
    lookup = {int(v): int(c) for v, c in zip(values, counts)}
    target = lookup.get(int(color), 0)
    others = [count for value, count in lookup.items() if int(value) != int(color)]
    return bool(others) and target > max(others)


def _sample_noise_plan(
    rng: np.random.Generator,
    tile: np.ndarray,
    bg: int,
) -> tuple[list[tuple[int, int]], int, int | None]:
    h, w = map(int, tile.shape)
    all_positions = [(r, c) for r in range(h) for c in range(w)]

    mode = int(rng.integers(0, 3))
    if mode == 0:
        change_count = 1
        dominant_count = 1
    elif mode == 1:
        change_count = int(rng.integers(2, min(4, h * w) + 1))
        dominant_count = change_count
    else:
        if h * w < 3:
            change_count = int(rng.integers(2, min(4, h * w) + 1))
            dominant_count = change_count
        else:
            change_count = int(rng.integers(3, min(5, h * w) + 1))
            dominant_count = change_count - 1

    chosen_idx = rng.choice(len(all_positions), size=int(change_count), replace=False)
    positions = [all_positions[int(i)] for i in chosen_idx]
    dominant_positions = positions[:dominant_count]
    excluded = {int(bg)}
    excluded.update(int(tile[r, c]) for r, c in dominant_positions)
    label = _choose_color(rng, excluded)

    minority = None
    if dominant_count < change_count:
        mr, mc = positions[-1]
        minority = _choose_color(rng, {int(bg), int(label), int(tile[mr, mc])})

    return positions, int(label), minority


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.integers(0, 10))
    tile_h = int(rng.integers(2, 6))
    tile_w = int(rng.integers(2, 6))
    rep_h = int(rng.integers(2, 6))
    rep_w = int(rng.integers(2, 6))

    if tile_h * rep_h > 30 or tile_w * rep_w > 30:
        return None

    pool = [color for color in range(10) if int(color) != int(bg)]
    rng.shuffle(pool)
    tile_color_count = int(rng.integers(1, min(4, len(pool)) + 1))
    tile_colors = [int(color) for color in pool[:tile_color_count]]
    tile = _random_tile(rng, int(tile_h), int(tile_w), int(bg), tile_colors)
    if tile is None:
        return None

    full = np.tile(tile, (int(rep_h), int(rep_w)))
    expected = np.full((int(rep_h), int(rep_w)), int(bg), dtype=int)

    total_tiles = int(rep_h) * int(rep_w)
    dirty_upper = min(
        total_tiles - 1,
        (total_tiles - 1) // 2,
        max(1, total_tiles // 3 + 1),
    )
    dirty_count = int(rng.integers(1, dirty_upper + 1))
    dirty_linear = rng.choice(total_tiles, size=int(dirty_count), replace=False)

    for linear in dirty_linear:
        tr = int(linear) // int(rep_w)
        tc = int(linear) % int(rep_w)
        block = full[
            tr * tile_h : (tr + 1) * tile_h,
            tc * tile_w : (tc + 1) * tile_w,
        ]
        positions, label, minority = _sample_noise_plan(rng, tile, int(bg))
        dominant_count = len(positions) if minority is None else len(positions) - 1

        for index, (r, c) in enumerate(positions):
            if index < dominant_count:
                block[r, c] = int(label)
            else:
                assert minority is not None
                block[r, c] = int(minority)
        expected[tr, tc] = int(label)

    if not _strict_most_common(full, int(bg)):
        return None

    try:
        solved = main(full)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    sig_shape = _infer_tile(full)[:2]
    if sig_shape != (int(tile_h), int(tile_w)):
        return None
    return full


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-tile anomaly input")
