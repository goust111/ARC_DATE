from __future__ import annotations

from collections import defaultdict
from itertools import combinations
import random

import numpy as np


CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIAGONAL_DIRS = ((-1, -1), (1, 1), (-1, 1), (1, -1))
DIRS = CARDINAL_DIRS + DIAGONAL_DIRS
OPPOSITE = {d: (-d[0], -d[1]) for d in DIRS}
PAIR_GROUPS = (
    ((0, 1), (0, -1)),
    ((1, 0), (-1, 0)),
    ((1, 1), (-1, -1)),
    ((1, -1), (-1, 1)),
)
CARDINAL_PAIRS = PAIR_GROUPS[:2]
DIAGONAL_PAIRS = PAIR_GROUPS[2:]
GEN_COLORS = (1, 2, 3, 4)
GEN_ATTEMPTS = 500


def _sign(x: int) -> int:
    return 0 if x == 0 else 1 if x > 0 else -1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _anchor_cells(anchor: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = anchor
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _find_anchor(mask: np.ndarray) -> tuple[int, int, int, int]:
    h, w = mask.shape
    anchors = []
    for r in range(h - 2):
        for c in range(w - 2):
            if mask[r : r + 3, c : c + 3].all():
                anchors.append((r, c, r + 2, c + 2))
    assert len(anchors) == 1
    return anchors[0]


def _direction_and_distance(
    coord: tuple[int, int], center: tuple[int, int]
) -> tuple[tuple[int, int], int]:
    dr = int(coord[0]) - int(center[0])
    dc = int(coord[1]) - int(center[1])
    assert not (dr == 0 and dc == 0)
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc)
    step = (_sign(dr), _sign(dc))
    dist = max(abs(dr), abs(dc))
    assert dist >= 2
    return step, dist


def _window(grid: np.ndarray, center: tuple[int, int], radius: int = 3) -> np.ndarray:
    size = 2 * radius + 1
    out = np.zeros((size, size), dtype=int)
    cr, cc = center
    for wr in range(size):
        for wc in range(size):
            gr = cr + wr - radius
            gc = cc + wc - radius
            if _in_bounds(grid.shape, gr, gc):
                out[wr, wc] = int(grid[gr, gc])
    return out


def _is_rot180_symmetric(window: np.ndarray) -> bool:
    return bool(np.array_equal(window, np.rot90(window, 2)))


def _compress_objects(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    out = np.zeros_like(g)
    infos: list[dict[str, object]] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) != 0):
        mask = g == color
        anchor = _find_anchor(mask)
        center = (anchor[0] + 1, anchor[1] + 1)
        counts: dict[tuple[int, int], int] = {d: 0 for d in DIRS}
        cells: list[tuple[int, int]] = []

        for r, c in _anchor_cells(anchor):
            assert out[r, c] == 0
            out[r, c] = color
            cells.append((r, c))

        rays: dict[tuple[int, int], list[int]] = defaultdict(list)
        for raw_r, raw_c in np.argwhere(mask):
            r = int(raw_r)
            c = int(raw_c)
            if anchor[0] <= r <= anchor[2] and anchor[1] <= c <= anchor[3]:
                continue
            step, dist = _direction_and_distance((r, c), center)
            rays[step].append(dist)

        for step, dists in rays.items():
            dists = sorted(dists)
            assert len(dists) == len(set(dists))
            counts[step] = len(dists)
            for i, _ in enumerate(dists):
                rr = center[0] + step[0] * (i + 2)
                cc = center[1] + step[1] * (i + 2)
                assert _in_bounds(g.shape, rr, cc)
                assert out[rr, cc] == 0
                out[rr, cc] = color
                cells.append((rr, cc))

        infos.append(
            {
                "color": color,
                "anchor": anchor,
                "center": center,
                "counts": counts,
                "cells": cells,
            }
        )

    return out, infos


def main(grid: np.ndarray) -> np.ndarray:
    """Compress each anchor's scattered ray pixels, then keep only anchors whose compressed 7x7 neighborhood is 180-degree symmetric.

    Input structure:
    - A black background contains 3-4 foreground colors.
    - Each foreground color has exactly one solid 3x3 anchor block plus several isolated same-colored pixels.
    - Every loose pixel lies on a horizontal, vertical, or 45-degree line through its anchor center.

    Rule:
    - For each color, slide its loose pixels inward along their own ray directions, packing them against the anchor or against already packed same-colored pixels on that ray.
    - After all colors are compressed, inspect the 7x7 window centered on each anchor center, treating out-of-bounds cells as background.
    - Keep colors whose 7x7 compressed pattern is invariant under 180-degree rotation; erase every pixel of colors that fail.

    Key insight:
    - The scattered pixels matter only after being packed back into a compact local arrangement; that compressed arrangement is what determines the symmetry test and therefore whether the whole color survives.
    """
    compressed, infos = _compress_objects(np.asarray(grid, dtype=int))
    out = compressed.copy()

    for info in infos:
        color = int(info["color"])
        center = info["center"]
        keep = _is_rot180_symmetric(_window(compressed, center))
        if not keep:
            out[out == color] = 0

    return out


def _zero_counts() -> dict[tuple[int, int], int]:
    return {d: 0 for d in DIRS}


def _total_count(counts: dict[tuple[int, int], int]) -> int:
    return sum(int(v) for v in counts.values())


def _nonzero_dirs(counts: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    return [d for d, n in counts.items() if n]


def _counts_are_symmetric(counts: dict[tuple[int, int], int]) -> bool:
    return all(counts[a] == counts[b] for a, b in PAIR_GROUPS)


def _make_anti_shortcut_counts(
    prefer_diagonal: bool,
) -> tuple[dict[tuple[int, int], int], tuple[tuple[int, int], tuple[int, int]]]:
    counts = _zero_counts()
    pair = random.choice(DIAGONAL_PAIRS if prefer_diagonal else CARDINAL_PAIRS)
    counts[pair[0]] = 1
    counts[pair[1]] = 1
    return counts, pair


def _make_stacked_asymmetric_counts(
    prefer_diagonal: bool,
) -> dict[tuple[int, int], int]:
    primary_pool = DIAGONAL_DIRS if prefer_diagonal else CARDINAL_DIRS
    secondary_pool = CARDINAL_DIRS if prefer_diagonal else DIAGONAL_DIRS

    for _ in range(200):
        counts = _zero_counts()
        counts[random.choice(primary_pool)] = 2
        counts[random.choice(secondary_pool)] += 1
        while _total_count(counts) < random.randint(3, 5):
            d = random.choice(DIRS)
            if counts[d] < 2:
                counts[d] += 1
        if _counts_are_symmetric(counts):
            continue
        if len(_nonzero_dirs(counts)) < 2:
            continue
        return counts
    raise RuntimeError("failed to build stacked asymmetric counts")


def _make_symmetric_counts() -> dict[tuple[int, int], int]:
    counts = _zero_counts()
    mode = random.choice(("pair1", "pair2", "two_pairs"))
    if mode == "pair1":
        pair = random.choice(PAIR_GROUPS)
        counts[pair[0]] = 1
        counts[pair[1]] = 1
    elif mode == "pair2":
        pair = random.choice(PAIR_GROUPS)
        counts[pair[0]] = 2
        counts[pair[1]] = 2
    else:
        pair_a, pair_b = random.sample(PAIR_GROUPS, 2)
        counts[pair_a[0]] = counts[pair_a[1]] = 1
        counts[pair_b[0]] = counts[pair_b[1]] = 1
    return counts


def _make_asymmetric_counts() -> dict[tuple[int, int], int]:
    for _ in range(300):
        counts = _zero_counts()
        target = random.randint(2, 5)
        while _total_count(counts) < target:
            d = random.choice(DIRS)
            if counts[d] < 2:
                counts[d] += 1
        if _counts_are_symmetric(counts):
            continue
        if len(_nonzero_dirs(counts)) < 2:
            continue
        return counts
    raise RuntimeError("failed to build asymmetric counts")


def _max_raw_distance(
    center: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]
) -> int:
    cr, cc = center
    t = 0
    while True:
        nr = cr + (t + 1) * step[0]
        nc = cc + (t + 1) * step[1]
        if not _in_bounds(shape, nr, nc):
            return t
        t += 1


def _distance_combos(
    center: tuple[int, int],
    step: tuple[int, int],
    count: int,
    shape: tuple[int, int],
) -> list[tuple[int, ...]]:
    max_t = min(8, _max_raw_distance(center, step, shape))
    candidates = [t for t in range(3, max_t + 1)]
    if count == 1:
        return [(t,) for t in candidates]
    combos = []
    for combo in combinations(candidates, count):
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1)):
            combos.append(combo)
    return combos


def _place_distances(
    center: tuple[int, int],
    counts: dict[tuple[int, int], int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    forced_pair: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> dict[tuple[int, int], tuple[int, ...]] | None:
    assigned: dict[tuple[int, int], tuple[int, ...]] = {}

    if forced_pair is not None:
        a, b = forced_pair
        forced_options = [((3,), (5,)), ((5,), (3,))]
        random.shuffle(forced_options)
        ok = False
        for da, db in forced_options:
            cand = {a: da, b: db}
            trial_cells = []
            for step, dist_tuple in cand.items():
                for t in dist_tuple:
                    rr = center[0] + step[0] * t
                    cc = center[1] + step[1] * t
                    if not _in_bounds(shape, rr, cc) or (rr, cc) in occupied:
                        trial_cells = []
                        break
                    trial_cells.append((rr, cc))
                if not trial_cells:
                    break
            if not trial_cells:
                continue
            assigned.update(cand)
            occupied.update(trial_cells)
            ok = True
            break
        if not ok:
            return None

    remaining = [d for d in _nonzero_dirs(counts) if d not in assigned]
    random.shuffle(remaining)
    for step in remaining:
        combos = _distance_combos(center, step, counts[step], shape)
        random.shuffle(combos)
        placed = False
        for combo in combos:
            cells = [
                (center[0] + step[0] * t, center[1] + step[1] * t)
                for t in combo
            ]
            if any(cell in occupied for cell in cells):
                continue
            assigned[step] = combo
            occupied.update(cells)
            placed = True
            break
        if not placed:
            return None

    return assigned


def _build_input_from_configs(
    shape: tuple[int, int],
    colors: list[int],
    centers: list[tuple[int, int]],
    configs: list[dict[str, object]],
) -> np.ndarray | None:
    grid = np.zeros(shape, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for color, center, cfg in zip(colors, centers, configs):
        cr, cc = center
        anchor = (cr - 1, cc - 1, cr + 1, cc + 1)
        for cell in _anchor_cells(anchor):
            if cell in occupied:
                return None
            occupied.add(cell)
            grid[cell] = color

        counts = cfg["counts"]
        forced_pair = cfg.get("forced_pair")
        distances = _place_distances(center, counts, shape, occupied, forced_pair)
        if distances is None:
            return None
        cfg["distances"] = distances

        for step, combo in distances.items():
            for t in combo:
                rr = cr + step[0] * t
                rc = cc + step[1] * t
                grid[rr, rc] = color

    return grid


def _gen_centers(h: int, w: int) -> list[tuple[int, int]]:
    return [(5, 5), (5, w - 6), (h - 6, 5), (h - 6, w - 6)]


def _build_configs(k: int) -> list[dict[str, object]]:
    anti_is_diag = bool(random.getrandbits(1))
    configs: list[dict[str, object]] = []

    anti_counts, anti_pair = _make_anti_shortcut_counts(anti_is_diag)
    configs.append(
        {
            "counts": anti_counts,
            "keep": True,
            "forced_pair": anti_pair,
        }
    )

    configs.append(
        {
            "counts": _make_stacked_asymmetric_counts(not anti_is_diag),
            "keep": False,
        }
    )

    while len(configs) < k:
        keep = bool(random.getrandbits(1))
        configs.append(
            {
                "counts": _make_symmetric_counts() if keep else _make_asymmetric_counts(),
                "keep": keep,
            }
        )

    random.shuffle(configs)
    return configs


def _raw_symmetry_flags(grid: np.ndarray, infos: list[dict[str, object]]) -> dict[int, bool]:
    return {
        int(info["color"]): _is_rot180_symmetric(_window(grid, info["center"]))
        for info in infos
    }


def _compressed_symmetry_flags(
    compressed: np.ndarray, infos: list[dict[str, object]]
) -> dict[int, bool]:
    return {
        int(info["color"]): _is_rot180_symmetric(_window(compressed, info["center"]))
        for info in infos
    }


def _validate_generated(grid: np.ndarray) -> bool:
    compressed, infos = _compress_objects(grid)
    out = main(grid)
    raw_flags = _raw_symmetry_flags(grid, infos)
    compressed_flags = _compressed_symmetry_flags(compressed, infos)

    if len(infos) not in (3, 4):
        return False

    keep_colors = {color for color, keep in compressed_flags.items() if keep}
    erase_colors = {color for color, keep in compressed_flags.items() if not keep}
    if not keep_colors or not erase_colors:
        return False

    if not any(raw_flags[color] != compressed_flags[color] for color in compressed_flags):
        return False

    saw_cardinal = False
    saw_diagonal = False
    saw_stack = False
    for info in infos:
        counts = info["counts"]
        if any(counts[d] for d in CARDINAL_DIRS):
            saw_cardinal = True
        if any(counts[d] for d in DIAGONAL_DIRS):
            saw_diagonal = True
        if any(counts[d] == 2 for d in DIRS):
            saw_stack = True

        total = _total_count(counts)
        if not (2 <= total <= 5):
            return False

    if not (saw_cardinal and saw_diagonal and saw_stack):
        return False

    for info in infos:
        color = int(info["color"])
        expected = compressed == color if color in keep_colors else np.zeros_like(grid, dtype=bool)
        got = out == color
        if not np.array_equal(got, expected):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 26)
    w = random.randint(22, 26)
    k = random.randint(3, 4)

    centers = _gen_centers(h, w)
    chosen_idx = random.sample(range(4), k)
    chosen_centers = [centers[i] for i in chosen_idx]
    random.shuffle(chosen_centers)
    colors = random.sample(list(GEN_COLORS), k)
    configs = _build_configs(k)

    grid = _build_input_from_configs((h, w), colors, chosen_centers, configs)
    if grid is None:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-rotation-filter sample")
