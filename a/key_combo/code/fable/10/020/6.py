from __future__ import annotations

import random
from typing import Callable

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GEN_CONFIGS = (
    {"m": 4, "residues": (0, 2, 3)},
    {"m": 4, "residues": (1, 2, 3)},
    {"m": 6, "residues": (0, 2, 3)},
    {"m": 6, "residues": (0, 2, 3, 5)},
    {"m": 6, "residues": (1, 3, 4)},
    {"m": 8, "residues": (1, 5)},
    {"m": 8, "residues": (0, 2, 4)},
    {"m": 8, "residues": (0, 2, 4, 6)},
    {"m": 9, "residues": (1, 4)},
    {"m": 9, "residues": (0, 3, 6)},
    {"m": 10, "residues": (0, 2, 5)},
    {"m": 12, "residues": (1, 7, 9)},
    {"m": 12, "residues": (0, 4, 6, 10)},
)

GEN_MIN_SIDE = 10
GEN_MAX_SIDE = 20
GEN_RETRY_LIMIT = 400


def _non_black_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(grid) if int(c) != BLACK)


def _sum_diagonal_index(r: int, c: int) -> int:
    return int(r + c)


def _diff_diagonal_index(r: int, c: int) -> int:
    return int(r - c)


def _color_indices(grid: np.ndarray, indexer: Callable[[int, int], int]) -> dict[int, list[int]]:
    mapping: dict[int, list[int]] = {}
    for color in _non_black_colors(grid):
        coords = np.argwhere(grid == int(color))
        mapping[int(color)] = sorted({int(indexer(int(r), int(c))) for r, c in coords})
    return mapping


def _infer_minimal_solution(
    grid: np.ndarray,
    indexer: Callable[[int, int], int],
) -> tuple[int, dict[int, int], dict[int, list[int]]] | None:
    g = np.asarray(grid, dtype=int)
    by_color = _color_indices(g, indexer)
    if not by_color:
        return None

    h, w = g.shape
    max_m = h + w - 1
    for modulus in range(2, max_m + 1):
        residues: dict[int, int] = {}
        ok = True
        for color, values in by_color.items():
            rems = {int(v % modulus) for v in values}
            if len(rems) != 1:
                ok = False
                break
            residues[int(color)] = next(iter(rems))
        if ok and len(set(residues.values())) == len(residues):
            return int(modulus), residues, by_color
    return None


def _individual_min_modulus(values: list[int], max_m: int) -> int | None:
    for modulus in range(2, max_m + 1):
        if len({int(v % modulus) for v in values}) == 1:
            return int(modulus)
    return None


def _proper_divisors(n: int) -> list[int]:
    return [d for d in range(2, n) if n % d == 0]


def _render_sum_output(shape: tuple[int, int], modulus: int, residues: dict[int, int]) -> np.ndarray:
    h, w = shape
    residue_to_color = {int(rem): int(color) for color, rem in residues.items()}
    out = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            color = residue_to_color.get((r + c) % modulus)
            if color is not None:
                out[r, c] = int(color)
    return out


def _sum_diagonal_coords(h: int, w: int, diagonal: int) -> list[tuple[int, int]]:
    r0 = max(0, int(diagonal) - (w - 1))
    r1 = min(h - 1, int(diagonal))
    return [(int(r), int(diagonal - r)) for r in range(r0, r1 + 1)]


def _candidate_points_for_diagonal(
    h: int,
    w: int,
    diagonal: int,
    residue: int,
    modulus: int,
) -> list[tuple[int, int]]:
    coords = _sum_diagonal_coords(h, w, diagonal)
    n = len(coords)
    if n == 1:
        return coords

    pivot = (diagonal // modulus + residue + h + w) % n
    preferred = [
        pivot,
        0,
        n - 1,
        n // 2,
        n // 4,
        (3 * (n - 1)) // 4,
        1,
        n - 2,
    ]

    ordered: list[tuple[int, int]] = []
    seen = set()
    for idx in preferred:
        idx = int(idx) % n
        point = coords[idx]
        if point not in seen:
            ordered.append(point)
            seen.add(point)
    for point in coords:
        if point not in seen:
            ordered.append(point)
            seen.add(point)
    return ordered


def _chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _select_scattered_points(
    requests: list[tuple[int, int, int]],
    h: int,
    w: int,
    modulus: int,
) -> dict[tuple[int, int], tuple[int, int]] | None:
    candidates = {
        (int(residue), int(diagonal)): _candidate_points_for_diagonal(h, w, int(diagonal), int(residue), modulus)
        for residue, diagonal, _color in requests
    }
    ordered_requests = sorted(
        requests,
        key=lambda item: (len(candidates[(int(item[0]), int(item[1]))]), int(item[1]), int(item[0])),
    )

    def search(
        idx: int,
        chosen: dict[tuple[int, int], tuple[int, int]],
        used: list[tuple[int, int]],
    ) -> dict[tuple[int, int], tuple[int, int]] | None:
        if idx == len(ordered_requests):
            return chosen.copy()

        residue, diagonal, _color = ordered_requests[idx]
        key = (int(residue), int(diagonal))
        points = candidates[key]

        scored = []
        for order, point in enumerate(points):
            min_cheb = min((_chebyshev_distance(point, other) for other in used), default=99)
            min_manhattan = min((_manhattan_distance(point, other) for other in used), default=99)
            scored.append(((min_cheb <= 1, -min_cheb, -min_manhattan, order), point))

        scored.sort(key=lambda item: item[0])
        for _score, point in scored:
            if any(point == other for other in used):
                continue
            if any(_chebyshev_distance(point, other) <= 1 for other in used):
                continue
            chosen[key] = point
            used.append(point)
            result = search(idx + 1, chosen, used)
            if result is not None:
                return result
            used.pop()
            del chosen[key]
        return None

    return search(0, {}, [])


def _observed_sum_diagonals(full_diagonals: list[int], residue: int, h: int, w: int) -> list[int] | None:
    phase = (int(residue) + h + w) % 2
    observed = full_diagonals[phase::2]
    alternate = full_diagonals[1 - phase :: 2]
    if len(observed) < 2:
        observed = alternate
    if len(observed) < 2:
        return None

    if len(observed) <= 4:
        return [int(d) for d in observed]

    mid = len(observed) // 2
    pick = [observed[0], observed[mid], observed[-1]]
    if len(observed) >= 6 and ((residue + h) % 2 == 0):
        pick = [observed[0], observed[2], observed[-1]]
    return [int(d) for d in pick]


def _latent_expected_output(
    h: int,
    w: int,
    modulus: int,
    color_residues: dict[int, int],
) -> np.ndarray:
    return _render_sum_output((h, w), modulus, color_residues)


def _generate_from_latent(
    h: int,
    w: int,
    modulus: int,
    color_residues: dict[int, int],
) -> np.ndarray | None:
    requests: list[tuple[int, int, int]] = []
    max_diagonal = h + w - 2
    for residue in sorted(set(int(r) for r in color_residues.values())):
        full_diagonals = list(range(int(residue), max_diagonal + 1, int(modulus)))
        observed = _observed_sum_diagonals(full_diagonals, residue, h, w)
        if observed is None:
            return None
        color = next(int(c) for c, rem in color_residues.items() if int(rem) == int(residue))
        for diagonal in observed:
            requests.append((int(residue), int(diagonal), int(color)))

    placements = _select_scattered_points(requests, h, w, modulus)
    if placements is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for residue, diagonal, color in requests:
        r, c = placements[(int(residue), int(diagonal))]
        grid[int(r), int(c)] = int(color)
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    max_m = h + w - 1

    sum_solution = _infer_minimal_solution(g, _sum_diagonal_index)
    if sum_solution is None:
        return False
    modulus, residues, by_color = sum_solution

    if modulus <= 2:
        return False
    divisors = _proper_divisors(modulus)
    if not divisors:
        return False

    # The intended diagonal family should be the only workable one.
    if _infer_minimal_solution(g, _diff_diagonal_index) is not None:
        return False

    # Each color alone should admit a smaller decoy modulus.
    for values in by_color.values():
        if len(values) < 2:
            return False
        individual = _individual_min_modulus(values, max_m)
        if individual is None or individual >= modulus:
            return False

    # Every proper divisor of the true modulus must fail globally.
    for divisor in divisors:
        divisor_residues = []
        for values in by_color.values():
            rems = {int(v % divisor) for v in values}
            if len(rems) != 1:
                return False
            divisor_residues.append(next(iter(rems)))
        if len(set(divisor_residues)) == len(divisor_residues):
            return False

    out = main(g)
    expected = _render_sum_output(g.shape, modulus, residues)
    if not np.array_equal(out, expected):
        return False

    if not np.any(out == BLACK):
        return False
    if len(_non_black_colors(g)) > 4:
        return False

    points = [tuple(map(int, coord)) for coord in np.argwhere(g != BLACK)]
    for i, p in enumerate(points):
        for q in points[i + 1 :]:
            if _chebyshev_distance(p, q) <= 1:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer one shared diagonal modulus and fill every matching congruence class across the grid.

    Input structure:
    - A black grid contains up to four non-black colors, each shown only as scattered witness pixels.
    - All witness pixels belong to the same "/"-slope diagonal family, indexed by `row + col`.
    - For each color, every witness lies on diagonals whose indices are congruent modulo one unknown shared modulus.

    Rule:
    - Find the smallest modulus `m >= 2` such that each color's witness diagonals share one residue mod `m`,
      and different colors use different residues.
    - For each color, paint every cell whose `row + col` has that residue modulo `m`.
    - Leave all unused residues black.

    Key insight:
    - A single color is compatible with smaller decoy moduli, so the true answer only appears after jointly
      reconciling all colors and rejecting residue collisions between them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    solution = _infer_minimal_solution(g, _sum_diagonal_index)
    assert solution is not None, "expected at least one globally consistent modulus"
    modulus, residues, _by_color = solution
    return _render_sum_output(g.shape, modulus, residues)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        config = random.choice(GEN_CONFIGS)
        modulus = int(config["m"])
        residues = tuple(int(r) for r in config["residues"])

        min_diagonal = max(residues) + 2 * modulus
        shapes = [
            (h, w)
            for h in range(GEN_MIN_SIDE, GEN_MAX_SIDE + 1)
            for w in range(GEN_MIN_SIDE, GEN_MAX_SIDE + 1)
            if h + w - 2 >= min_diagonal
        ]
        if not shapes:
            continue
        h, w = random.choice(shapes)

        colors = random.sample(PALETTE, len(residues))
        color_residues = {int(color): int(residue) for color, residue in zip(colors, residues)}

        grid = _generate_from_latent(h, w, modulus, color_residues)
        if grid is None:
            continue

        expected = _latent_expected_output(h, w, modulus, color_residues)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        if not _semantic_checks(grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid modulus-inference ARC input")
