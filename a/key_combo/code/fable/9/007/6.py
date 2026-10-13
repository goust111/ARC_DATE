from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


MAX_PERIOD = 6
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _majority_color(values: list[int]) -> int | None:
    counts = Counter(values)
    color, count = counts.most_common(1)[0]
    if count * 2 <= len(values):
        return None
    return int(color)


def _class_coords(shape: tuple[int, int], period_h: int, period_w: int, rr: int, cc: int) -> list[tuple[int, int]]:
    height, width = shape
    return [
        (r, c)
        for r in range(rr, height, period_h)
        for c in range(cc, width, period_w)
    ]


def _evaluate_candidate(grid: np.ndarray, period_h: int, period_w: int) -> tuple[np.ndarray, dict[int, int], np.ndarray] | None:
    height, width = map(int, grid.shape)
    motif = np.zeros((period_h, period_w), dtype=int)
    deviations: dict[int, set[int]] = defaultdict(set)

    for rr in range(period_h):
        for cc in range(period_w):
            values = [int(grid[r, c]) for r, c in _class_coords((height, width), period_h, period_w, rr, cc)]
            majority = _majority_color(values)
            if majority is None:
                return None
            motif[rr, cc] = majority
            for value in values:
                if value != majority:
                    deviations[majority].add(int(value))

    colors = set(map(int, np.unique(motif)))
    if 0 in colors or len(colors) < 2:
        return None
    if set(deviations) != colors:
        return None

    permutation: dict[int, int] = {}
    used_targets: set[int] = set()
    for color in colors:
        targets = deviations[color]
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        if target == color or target not in colors or target in used_targets:
            return None
        permutation[color] = target
        used_targets.add(target)

    if used_targets != colors:
        return None

    rows, cols = np.indices(grid.shape)
    expected = motif[rows % period_h, cols % period_w]
    actual = grid
    ok = (actual == expected)
    for src, dst in permutation.items():
        ok |= (expected == src) & (actual == dst)
    if not np.all(ok):
        return None
    if np.array_equal(actual, expected):
        return None

    return motif, permutation, expected


def _infer_solution(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must be at most 30x30"
    assert int(np.min(g)) >= 0 and int(np.max(g)) <= 9, "expected ARC palette ids"
    assert not np.any(g == 0), "the tiled motif must not contain black"

    candidates: list[tuple[int, int, np.ndarray, dict[int, int], np.ndarray]] = []
    max_h = min(MAX_PERIOD, g.shape[0])
    max_w = min(MAX_PERIOD, g.shape[1])
    for period_h in range(1, max_h + 1):
        for period_w in range(1, max_w + 1):
            result = _evaluate_candidate(g, period_h, period_w)
            if result is not None:
                motif, permutation, expected = result
                candidates.append((period_h * period_w, period_h, period_w, motif, permutation, expected))

    assert candidates, "failed to infer any valid periodic explanation"
    min_area = min(candidate[0] for candidate in candidates)
    minimal = [candidate for candidate in candidates if candidate[0] == min_area]
    assert len(minimal) == 1, "expected a unique smallest valid motif"
    _area, period_h, period_w, motif, permutation, expected = minimal[0]
    assert motif.shape == (period_h, period_w)
    return motif, permutation, expected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the broken cells of a periodic tiling as entries of one global recoloring law, then apply that law to the whole intended tiling.

    Input structure:
    - The grid is a seamless periodic tiling of one small non-black motif, except for a few deviant cells.
    - In every residue class of the true tiling period, the rule-following color is the absolute majority.
    - Every deviant cell shows one entry of the same color permutation: expected motif color -> actually shown color.

    Rule:
    - Infer the unique smallest motif by taking the majority color in each periodic residue class.
    - Collect all deviant cells to recover the complete permutation on the motif colors.
    - Output a grid of the same size where every cell becomes permutation(expected_color_at_that_position).

    Key insight:
    - The anomalies are not noise to repair: each one is already a sample of the global recoloring law, so the output keeps those cells unchanged and recolors the previously correct ones.
    """
    motif, permutation, expected = _infer_solution(np.asarray(grid, dtype=int))
    out = expected.copy()
    for src, dst in permutation.items():
        out[expected == src] = dst
    return out


def _random_derangement(colors: list[int]) -> dict[int, int]:
    shuffled = colors[:]
    for _ in range(200):
        random.shuffle(shuffled)
        if all(src != dst for src, dst in zip(colors, shuffled)):
            return {src: dst for src, dst in zip(colors, shuffled)}
    raise AssertionError("failed to sample a derangement")


def _is_primitive_motif(motif: np.ndarray) -> bool:
    height, width = map(int, motif.shape)
    for sub_h in range(1, height + 1):
        if height % sub_h != 0:
            continue
        for sub_w in range(1, width + 1):
            if width % sub_w != 0:
                continue
            if sub_h == height and sub_w == width:
                continue
            base = motif[:sub_h, :sub_w]
            tiled = np.tile(base, (height // sub_h, width // sub_w))
            if np.array_equal(tiled, motif):
                return False
    return True


def _tile_motif(motif: np.ndarray, height: int, width: int) -> np.ndarray:
    rows, cols = np.indices((height, width))
    return motif[rows % motif.shape[0], cols % motif.shape[1]]


def _sample_motif() -> np.ndarray:
    period_h = random.randint(2, 5)
    period_w = random.randint(2, 5)
    color_count = random.randint(3, min(5, period_h * period_w))
    colors = random.sample(PALETTE, color_count)
    for _ in range(300):
        motif = np.array(
            [[random.choice(colors) for _ in range(period_w)] for _ in range(period_h)],
            dtype=int,
        )
        if len(set(map(int, np.unique(motif)))) != color_count:
            continue
        if not _is_primitive_motif(motif):
            continue
        return motif
    raise AssertionError("failed to sample a primitive motif")


def _occurrence_counts(shape: tuple[int, int], period_h: int, period_w: int) -> np.ndarray:
    counts = np.zeros((period_h, period_w), dtype=int)
    for rr in range(period_h):
        for cc in range(period_w):
            counts[rr, cc] = len(_class_coords(shape, period_h, period_w, rr, cc))
    return counts


def _sample_anomalies(expected: np.ndarray, motif: np.ndarray, permutation: dict[int, int]) -> list[tuple[int, int]]:
    period_h, period_w = map(int, motif.shape)
    counts = _occurrence_counts(expected.shape, period_h, period_w)
    color_to_residues: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for rr in range(period_h):
        for cc in range(period_w):
            color_to_residues[int(motif[rr, cc])].append((rr, cc))

    capacities = {
        (rr, cc): max(1, (int(counts[rr, cc]) - 1) // 2)
        for rr in range(period_h)
        for cc in range(period_w)
    }

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for color, residues in color_to_residues.items():
        feasible = [residue for residue in residues if capacities[residue] > 0]
        if not feasible:
            raise AssertionError("every motif color must have anomaly capacity")
        rr, cc = random.choice(feasible)
        coords = _class_coords(expected.shape, period_h, period_w, rr, cc)
        coord = random.choice(coords)
        chosen.append(coord)
        used.add(coord)
        capacities[(rr, cc)] -= 1

    for rr in range(period_h):
        for cc in range(period_w):
            if capacities[(rr, cc)] <= 0:
                continue
            if random.random() < 0.45:
                coords = [coord for coord in _class_coords(expected.shape, period_h, period_w, rr, cc) if coord not in used]
                if not coords:
                    continue
                extra_count = random.randint(1, min(capacities[(rr, cc)], len(coords)))
                for coord in random.sample(coords, extra_count):
                    chosen.append(coord)
                    used.add(coord)

    random.shuffle(chosen)
    return chosen


def _expected_output_from_parts(expected: np.ndarray, permutation: dict[int, int]) -> np.ndarray:
    out = expected.copy()
    for src, dst in permutation.items():
        out[expected == src] = dst
    return out


def _apply_anomalies(expected: np.ndarray, anomalies: list[tuple[int, int]], permutation: dict[int, int]) -> np.ndarray:
    inp = expected.copy()
    for r, c in anomalies:
        inp[r, c] = permutation[int(expected[r, c])]
    return inp


def _has_direct_input_permutation(inp: np.ndarray, out: np.ndarray) -> bool:
    mapping: dict[int, int] = {}
    for src in map(int, np.unique(inp)):
        targets = set(map(int, np.unique(out[inp == src])))
        if len(targets) != 1:
            return False
        mapping[src] = next(iter(targets))
    remapped = inp.copy()
    for src, dst in mapping.items():
        remapped[inp == src] = dst
    return np.array_equal(remapped, out)


def _validate_instance(inp: np.ndarray, motif: np.ndarray, permutation: dict[int, int]) -> bool:
    expected = _tile_motif(motif, *inp.shape)
    out = _expected_output_from_parts(expected, permutation)
    if np.array_equal(inp, expected) or np.array_equal(inp, out):
        return False
    if _has_direct_input_permutation(inp, out):
        return False

    deviations = inp != expected
    if not np.all(inp[deviations] == np.vectorize(permutation.get)(expected[deviations])):
        return False

    counts = _occurrence_counts(inp.shape, motif.shape[0], motif.shape[1])
    for rr in range(motif.shape[0]):
        for cc in range(motif.shape[1]):
            bad = int(np.count_nonzero(
                deviations[
                    np.arange(rr, inp.shape[0], motif.shape[0])[:, None],
                    np.arange(cc, inp.shape[1], motif.shape[1])[None, :],
                ]
            ))
            if bad * 2 >= counts[rr, cc]:
                return False

    colors = set(map(int, np.unique(motif)))
    seen_sources = {int(expected[r, c]) for r, c in np.argwhere(deviations)}
    if seen_sources != colors:
        return False

    inferred_motif, inferred_perm, inferred_expected = _infer_solution(inp)
    if inferred_motif.shape != motif.shape or not np.array_equal(inferred_motif, motif):
        return False
    if inferred_perm != permutation:
        return False
    if not np.array_equal(inferred_expected, expected):
        return False
    if not np.array_equal(main(inp), out):
        return False
    return True


def _attempt() -> np.ndarray | None:
    motif = _sample_motif()
    period_h, period_w = map(int, motif.shape)

    height = random.randint(max(period_h * 2 + 1, 9), min(30, period_h * 5 + 7))
    width = random.randint(max(period_w * 2 + 1, 9), min(30, period_w * 5 + 7))
    expected = _tile_motif(motif, height, width)

    colors = sorted(set(map(int, np.unique(motif))))
    permutation = _random_derangement(colors)
    anomalies = _sample_anomalies(expected, motif, permutation)
    inp = _apply_anomalies(expected, anomalies, permutation)

    if _validate_instance(inp, motif, permutation):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid periodic-permutation ARC input")
