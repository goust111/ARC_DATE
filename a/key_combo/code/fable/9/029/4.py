from __future__ import annotations

from collections import Counter
from typing import Iterator

import numpy as np


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, int(n) + 1) if int(n) % d == 0]


def _iter_blocks(grid: np.ndarray, tile_h: int, tile_w: int) -> Iterator[tuple[tuple[int, int], np.ndarray]]:
    h, w = map(int, grid.shape)
    for br in range(h // tile_h):
        for bc in range(w // tile_w):
            r0 = br * tile_h
            c0 = bc * tile_w
            yield (br, bc), grid[r0 : r0 + tile_h, c0 : c0 + tile_w]


def _infer_mapping(standard: np.ndarray, deviant: np.ndarray) -> dict[int, int] | None:
    mapping: dict[int, int] = {}
    for src, dst in zip(standard.ravel(), deviant.ravel()):
        src = int(src)
        dst = int(dst)
        if src in mapping and mapping[src] != dst:
            return None
        mapping[src] = dst
    return mapping


def _candidate_periods(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    candidates: list[dict] = []

    for tile_h in _divisors(h):
        for tile_w in _divisors(w):
            n_blocks = (h // tile_h) * (w // tile_w)
            if n_blocks < 3 or (tile_h == h and tile_w == w):
                continue

            blocks = []
            counts: Counter[bytes] = Counter()
            examples: dict[bytes, np.ndarray] = {}
            for pos, block in _iter_blocks(grid, tile_h, tile_w):
                sig = block.tobytes()
                blocks.append((pos, block.copy(), sig))
                counts[sig] += 1
                examples.setdefault(sig, block.copy())

            ranked = counts.most_common()
            if len(ranked) != 2:
                continue
            if ranked[0][1] != n_blocks - 1 or ranked[1][1] != 1:
                continue

            standard_sig = ranked[0][0]
            deviant_blocks = [(pos, block) for pos, block, sig in blocks if sig != standard_sig]
            if len(deviant_blocks) != 1:
                continue

            deviant_pos, deviant = deviant_blocks[0]
            standard = examples[standard_sig]
            mapping = _infer_mapping(standard, deviant)
            if mapping is None or all(src == dst for src, dst in mapping.items()):
                continue

            candidates.append(
                {
                    "tile_h": int(tile_h),
                    "tile_w": int(tile_w),
                    "standard": standard,
                    "deviant": deviant,
                    "deviant_pos": tuple(map(int, deviant_pos)),
                    "mapping": mapping,
                }
            )

    candidates.sort(key=lambda item: (item["tile_h"] * item["tile_w"], item["tile_h"], item["tile_w"]))
    return candidates


def _solve_instance(grid: np.ndarray) -> dict:
    candidates = _candidate_periods(grid)
    assert candidates, "expected a periodic tiling with exactly one deviant tile"

    best_area = candidates[0]["tile_h"] * candidates[0]["tile_w"]
    best = [cand for cand in candidates if cand["tile_h"] * cand["tile_w"] == best_area]
    assert len(best) == 1, "expected a unique smallest tiling period"
    return best[0]


def _apply_mapping(grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    src_grid = np.asarray(grid, dtype=int)
    for src, dst in mapping.items():
        out[src_grid == int(src)] = int(dst)
    return out


def _tile_from_period(tile: np.ndarray, h: int, w: int) -> np.ndarray:
    tile_h, tile_w = map(int, tile.shape)
    return np.tile(tile, (h // tile_h, w // tile_w))


def _has_exact_period(tile: np.ndarray) -> bool:
    tile_h, tile_w = map(int, tile.shape)
    for sub_h in _divisors(tile_h):
        for sub_w in _divisors(tile_w):
            if sub_h == tile_h and sub_w == tile_w:
                continue
            if np.array_equal(tile, _tile_from_period(tile[:sub_h, :sub_w], tile_h, tile_w)):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color-replacement legend from the single deviant wallpaper tile and apply it to the whole wallpaper.

    Input structure:
    - The full grid is tiled periodically in both rows and columns by one multicolor tile.
    - Exactly one tile instance is deviant: it keeps the same per-cell structure as the standard tile,
      but each standard-tile color is replaced consistently by its legend color in that one instance.
    - No other tile differs from the standard tile.

    Rule:
    - Recover the tile period and locate the unique deviant tile instance.
    - Compare the deviant tile with any standard tile cell-by-cell to decode the color replacement table.
    - Apply that replacement table to every pixel in the grid and output the recolored wallpaper.

    Key insight:
    - The odd-looking tile is not damage to repair back to the majority pattern; it is the legend that tells
      how every standard color should be recolored, so the whole wallpaper must change to match the deviant tile.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    solved = _solve_instance(g)
    return _apply_mapping(g, solved["mapping"])


# --------------- generator (o5x) ---------------

_RNG = np.random.default_rng()
_PALETTE = tuple(range(10))
_TILE_DIM_OPTIONS = [(h, w) for h in range(2, 6) for w in range(2, 6) if h * w >= 6]
_REPEAT_OPTIONS = (2, 3, 4, 5)
_MAX_ATTEMPTS = 500
_MODE_MERGE = "merge"
_MODE_NEW = "new"
_MODE_MIXED = "mixed"


def _random_counts(total: int, parts: int) -> list[int]:
    cuts = sorted(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(int(cut - prev))
        prev = int(cut)
    return out


def _make_standard_tile(tile_h: int, tile_w: int) -> np.ndarray | None:
    area = tile_h * tile_w
    max_colors = min(5, area - 1)
    color_count = int(_RNG.integers(3, max_colors + 1))

    if area >= 2 * color_count:
        base_counts = [2] * color_count
        leftovers = area - 2 * color_count
        counts = base_counts.copy()
        for idx, extra in enumerate(_random_counts(leftovers + color_count, color_count)):
            counts[idx] += extra - 1
    else:
        counts = _random_counts(area, color_count)

    colors = _RNG.choice(_PALETTE, size=color_count, replace=False).tolist()
    flat = np.array([color for color, count in zip(colors, counts) for _ in range(int(count))], dtype=int)
    _RNG.shuffle(flat)
    tile = flat.reshape(tile_h, tile_w)

    if len(np.unique(tile)) != color_count:
        return None
    if not _has_exact_period(tile):
        return None
    if sum(len(np.unique(tile[r, :])) >= 2 for r in range(tile_h)) < 2:
        return None
    if sum(len(np.unique(tile[:, c])) >= 2 for c in range(tile_w)) < 2:
        return None
    return tile


def _choose_targets(source_colors: list[int], mode: str) -> dict[int, int] | None:
    source_set = set(source_colors)
    spare = [color for color in _PALETTE if color not in source_set]
    color_count = len(source_colors)

    if mode == _MODE_MERGE:
        rep_count = int(_RNG.integers(2, color_count))
        reps = _RNG.choice(source_colors, size=rep_count, replace=False).tolist()
        new_targets: list[int] = []
    elif mode == _MODE_NEW:
        new_count = int(_RNG.integers(1, min(3, len(spare)) + 1))
        rep_count = int(_RNG.integers(0, min(color_count - 1, 2) + 1))
        reps = [] if rep_count == 0 else _RNG.choice(source_colors, size=rep_count, replace=False).tolist()
        new_targets = _RNG.choice(spare, size=new_count, replace=False).tolist()
    else:
        if not spare:
            return None
        rep_count = int(_RNG.integers(1, color_count))
        reps = _RNG.choice(source_colors, size=rep_count, replace=False).tolist()
        max_new = min(2, len(spare), color_count - rep_count)
        if max_new == 0:
            return None
        new_count = int(_RNG.integers(1, max_new + 1))
        new_targets = _RNG.choice(spare, size=new_count, replace=False).tolist()

    targets = reps + new_targets
    if len(targets) < 2:
        return None

    mapping = {int(rep): int(rep) for rep in reps}
    remaining = [int(color) for color in source_colors if int(color) not in mapping]
    _RNG.shuffle(remaining)

    forced_targets = [int(color) for color in new_targets]
    for target in forced_targets:
        if not remaining:
            return None
        mapping[int(remaining.pop())] = int(target)

    for color in remaining:
        mapping[int(color)] = int(_RNG.choice(targets))

    if all(src == dst for src, dst in mapping.items()):
        return None

    if mode == _MODE_MERGE and all(dst not in source_set or src == dst for src, dst in mapping.items()):
        return None

    return mapping


def _make_mapping(tile: np.ndarray) -> dict[int, int] | None:
    source_colors = sorted(map(int, np.unique(tile)))
    modes = [_MODE_MERGE, _MODE_NEW, _MODE_MIXED]
    _RNG.shuffle(modes)
    for mode in modes:
        mapping = _choose_targets(source_colors, mode)
        if mapping is not None and len(set(mapping.values())) >= 2:
            return mapping
    return None


def _expected_output(tile: np.ndarray, repeats_r: int, repeats_c: int, mapping: dict[int, int]) -> np.ndarray:
    full = np.tile(tile, (repeats_r, repeats_c))
    return _apply_mapping(full, mapping)


def _designed_instance() -> tuple[np.ndarray, np.ndarray, dict] | None:
    tile_h, tile_w = _TILE_DIM_OPTIONS[int(_RNG.integers(len(_TILE_DIM_OPTIONS)))]
    repeat_r = int(_RNG.choice([n for n in _REPEAT_OPTIONS if tile_h * n <= 30]))
    repeat_c = int(_RNG.choice([n for n in _REPEAT_OPTIONS if tile_w * n <= 30]))
    if repeat_r * repeat_c < 4:
        return None

    tile = _make_standard_tile(tile_h, tile_w)
    if tile is None:
        return None

    mapping = _make_mapping(tile)
    if mapping is None:
        return None

    deviant = _apply_mapping(tile, mapping)
    if np.array_equal(deviant, tile):
        return None

    dev_r = int(_RNG.integers(repeat_r))
    dev_c = int(_RNG.integers(repeat_c))

    inp = np.tile(tile, (repeat_r, repeat_c))
    r0 = dev_r * tile_h
    c0 = dev_c * tile_w
    inp[r0 : r0 + tile_h, c0 : c0 + tile_w] = deviant

    out = _expected_output(tile, repeat_r, repeat_c, mapping)
    meta = {
        "tile_h": tile_h,
        "tile_w": tile_w,
        "deviant_pos": (dev_r, dev_c),
        "mapping": mapping,
        "standard": tile,
        "deviant": deviant,
    }
    return inp, out, meta


def _semantics_ok(inp: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    solved = _solve_instance(inp)
    if solved["tile_h"] != meta["tile_h"] or solved["tile_w"] != meta["tile_w"]:
        return False
    if solved["deviant_pos"] != meta["deviant_pos"]:
        return False
    if solved["mapping"] != meta["mapping"]:
        return False
    if not np.array_equal(main(inp), expected):
        return False

    tile_h = int(meta["tile_h"])
    tile_w = int(meta["tile_w"])
    dev_r, dev_c = map(int, meta["deviant_pos"])
    repaired = inp.copy()
    r0 = dev_r * tile_h
    c0 = dev_c * tile_w
    repaired[r0 : r0 + tile_h, c0 : c0 + tile_w] = meta["standard"]
    if np.array_equal(expected, inp) or np.array_equal(expected, repaired):
        return False

    changed = expected != inp
    changed[r0 : r0 + tile_h, c0 : c0 + tile_w] = False
    if int(changed.sum()) == 0:
        return False

    candidates = _candidate_periods(inp)
    if not candidates:
        return False
    best_area = candidates[0]["tile_h"] * candidates[0]["tile_w"]
    if sum(c["tile_h"] * c["tile_w"] == best_area for c in candidates) != 1:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        built = _designed_instance()
        if built is None:
            continue
        inp, expected, meta = built
        if _semantics_ok(inp, expected, meta):
            return inp
    raise RuntimeError(f"failed to generate a valid deviant-tile wallpaper after {_MAX_ATTEMPTS} attempts")
