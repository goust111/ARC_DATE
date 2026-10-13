import random
from dataclasses import dataclass

import numpy as np


PATTERN_SIZE = 3
LEGEND_ROWS = 3
LEGEND_COLS = 7
BORDER_ROW = 3
BORDER_COL = 7

MAIN_MIN = 10
MAIN_MAX = 16
STAMP_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 500

STEP_CHOICES = [
    (-2, 0),
    (-1, 0),
    (1, 0),
    (2, 0),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
    (-2, -1),
    (-2, 1),
    (2, -1),
    (2, 1),
    (-1, -2),
    (1, -2),
    (-1, 2),
    (1, 2),
]


@dataclass(frozen=True)
class Candidate:
    kind: int
    r0: int
    c0: int
    bits: int
    mismatch: int
    center_color: int


def _cell_bit(r: int, c: int, width: int) -> int:
    return 1 << (r * width + c)


def _footprint_bits(r0: int, c0: int, width: int) -> int:
    bits = 0
    for dr in range(PATTERN_SIZE):
        for dc in range(PATTERN_SIZE):
            bits |= _cell_bit(r0 + dr, c0 + dc, width)
    return bits


def _bit_count(bits: int) -> int:
    return int(bits.bit_count())


def _is_asymmetric(pattern: np.ndarray) -> bool:
    transforms = [
        np.flipud(pattern),
        np.fliplr(pattern),
        np.rot90(pattern, 1),
        np.rot90(pattern, 2),
        np.rot90(pattern, 3),
        pattern.T,
        np.fliplr(np.rot90(pattern, 1)),
    ]
    return all(not np.array_equal(pattern, transformed) for transformed in transforms)


def _find_border(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    for color in map(int, np.unique(grid)):
        coords = np.argwhere(grid == color)
        if coords.shape[0] != h + w - 1:
            continue
        row_counts = np.bincount(coords[:, 0], minlength=h)
        col_counts = np.bincount(coords[:, 1], minlength=w)
        full_rows = np.where(row_counts == w)[0]
        full_cols = np.where(col_counts == h)[0]
        if len(full_rows) == 1 and len(full_cols) == 1:
            return color, int(full_rows[0]), int(full_cols[0])
    raise AssertionError("failed to locate the legend border")


def _extract_parts(grid: np.ndarray) -> tuple[int, tuple[np.ndarray, np.ndarray], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _border_color, border_r, border_c = _find_border(g)
    assert border_r == BORDER_ROW and border_c == BORDER_COL, "unexpected legend layout"

    bg = int(g[0, 3])
    pattern_a = g[0:3, 0:3].copy()
    pattern_b = g[0:3, 4:7].copy()
    assert pattern_a.shape == (3, 3) and pattern_b.shape == (3, 3)
    assert np.all(pattern_a != bg) and np.all(pattern_b != bg), "stamps must be fully colored"

    main = g[border_r + 1 :, border_c + 1 :].copy()
    assert main.shape[0] >= 3 and main.shape[1] >= 3, "main area must fit a stamp"
    return bg, (pattern_a, pattern_b), main


def _build_candidates(
    main: np.ndarray, patterns: tuple[np.ndarray, np.ndarray], bg: int
) -> tuple[int, list[Candidate], list[list[int]], dict[tuple[int, int, int], int]]:
    h, w = main.shape
    target_bits = 0
    for r, c in np.argwhere(main != bg):
        target_bits |= _cell_bit(int(r), int(c), w)

    candidates: list[Candidate] = []
    cell_to_candidates: list[list[int]] = [[] for _ in range(h * w)]
    by_key: dict[tuple[int, int, int], int] = {}

    for kind, pattern in enumerate(patterns):
        for r0 in range(h - PATTERN_SIZE + 1):
            for c0 in range(w - PATTERN_SIZE + 1):
                window = main[r0 : r0 + PATTERN_SIZE, c0 : c0 + PATTERN_SIZE]
                if np.any(window == bg):
                    continue

                bits = 0
                mismatch = 0
                for dr in range(PATTERN_SIZE):
                    for dc in range(PATTERN_SIZE):
                        rr = r0 + dr
                        cc = c0 + dc
                        bit = _cell_bit(rr, cc, w)
                        bits |= bit
                        if int(window[dr, dc]) != int(pattern[dr, dc]):
                            mismatch |= bit

                cand = Candidate(
                    kind=kind,
                    r0=r0,
                    c0=c0,
                    bits=bits,
                    mismatch=mismatch,
                    center_color=int(pattern[1, 1]),
                )
                cid = len(candidates)
                candidates.append(cand)
                by_key[(kind, r0, c0)] = cid
                for dr in range(PATTERN_SIZE):
                    for dc in range(PATTERN_SIZE):
                        cell_to_candidates[(r0 + dr) * w + (c0 + dc)].append(cid)

    return target_bits, candidates, cell_to_candidates, by_key


def _solve_candidate_sets(
    main: np.ndarray,
    patterns: tuple[np.ndarray, np.ndarray],
    bg: int,
    limit: int = 2,
) -> tuple[set[frozenset[int]], list[Candidate], dict[tuple[int, int, int], int]]:
    target_bits, candidates, cell_to_candidates, by_key = _build_candidates(main, patterns, bg)
    assert target_bits != 0, "main area must contain at least one stamp trace"

    memo: dict[int, set[frozenset[int]]] = {}

    def dfs(covered: int) -> set[frozenset[int]]:
        if covered == target_bits:
            return {frozenset()}
        if covered in memo:
            return memo[covered]

        remaining = target_bits & ~covered
        pivot_bit = remaining & -remaining
        pivot_idx = pivot_bit.bit_length() - 1

        solutions: set[frozenset[int]] = set()
        for cid in cell_to_candidates[pivot_idx]:
            cand = candidates[cid]
            new_visible = cand.bits & ~covered
            if new_visible == 0 or (new_visible & pivot_bit) == 0:
                continue
            if cand.mismatch & new_visible:
                continue

            for sub_solution in dfs(covered | cand.bits):
                solutions.add(sub_solution | {cid})
                if len(solutions) >= limit:
                    memo[covered] = solutions
                    return solutions

        memo[covered] = solutions
        return solutions

    return dfs(0), candidates, by_key


def _recover_placements(
    main: np.ndarray, patterns: tuple[np.ndarray, np.ndarray], bg: int
) -> list[Candidate]:
    solution_sets, candidates, _by_key = _solve_candidate_sets(main, patterns, bg, limit=2)
    assert solution_sets, "no valid stamp decomposition found"
    assert len(solution_sets) == 1, "stamp decomposition is ambiguous"
    placement_ids = sorted(next(iter(solution_sets)))
    return [candidates[cid] for cid in placement_ids]


def _compose_main(
    shape: tuple[int, int],
    patterns: tuple[np.ndarray, np.ndarray],
    placements: list[tuple[int, int, int]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for kind, r0, c0 in placements:
        out[r0 : r0 + PATTERN_SIZE, c0 : c0 + PATTERN_SIZE] = patterns[kind]
    return out


def _visible_bits_for_sequence(
    placements: list[tuple[int, int, int]], width: int
) -> list[int]:
    later_bits = 0
    visible_reversed: list[int] = []
    for kind, r0, c0 in reversed(placements):
        del kind
        bits = _footprint_bits(r0, c0, width)
        visible_reversed.append(bits & ~later_bits)
        later_bits |= bits
    return list(reversed(visible_reversed))


def _sample_pattern_pair(rng: random.Random, palette: list[int]) -> tuple[np.ndarray, np.ndarray]:
    center_a, center_b = rng.sample(palette, 2)
    shared_palette = palette[:]

    while True:
        pattern_a = np.array(
            [[rng.choice(shared_palette) for _ in range(3)] for _ in range(3)],
            dtype=int,
        )
        pattern_b = np.array(
            [[rng.choice(shared_palette) for _ in range(3)] for _ in range(3)],
            dtype=int,
        )
        pattern_a[1, 1] = center_a
        pattern_b[1, 1] = center_b

        if np.array_equal(pattern_a, pattern_b):
            continue
        if not _is_asymmetric(pattern_a) or not _is_asymmetric(pattern_b):
            continue
        if len(set(map(int, pattern_a.ravel()))) < 3:
            continue
        if len(set(map(int, pattern_b.ravel()))) < 3:
            continue
        if int(np.sum(pattern_a != pattern_b)) < 4:
            continue
        return pattern_a, pattern_b


def _sample_chain_positions(
    rng: random.Random, stamp_count: int
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    for _ in range(200):
        relative = [(0, 0)]
        used = {(0, 0)}
        for _i in range(1, stamp_count):
            step = rng.choice(STEP_CHOICES)
            prev_r, prev_c = relative[-1]
            cur = (prev_r + step[0], prev_c + step[1])
            if cur in used:
                break
            relative.append(cur)
            used.add(cur)
        if len(relative) != stamp_count:
            continue

        rows = [r for r, _c in relative]
        cols = [c for _r, c in relative]
        span_h = max(rows) - min(rows) + PATTERN_SIZE
        span_w = max(cols) - min(cols) + PATTERN_SIZE
        if span_h > MAIN_MAX - 1 or span_w > MAIN_MAX - 1:
            continue

        main_h = rng.randint(max(MAIN_MIN, span_h + 1), MAIN_MAX)
        main_w = rng.randint(max(MAIN_MIN, span_w + 1), MAIN_MAX)
        offset_r = rng.randint(0, main_h - span_h)
        offset_c = rng.randint(0, main_w - span_w)
        shift_r = offset_r - min(rows)
        shift_c = offset_c - min(cols)
        absolute = [(r + shift_r, c + shift_c) for r, c in relative]
        return (main_h, main_w), absolute

    raise RuntimeError("failed to sample stamp positions")


def _validate_instance(
    patterns: tuple[np.ndarray, np.ndarray],
    placements: list[tuple[int, int, int]],
    main: np.ndarray,
    bg: int,
) -> bool:
    if {kind for kind, _r0, _c0 in placements} != {0, 1}:
        return False

    visible_bits = _visible_bits_for_sequence(placements, main.shape[1])
    visible_counts = [_bit_count(bits) for bits in visible_bits]
    if visible_counts[-1] != 9:
        return False
    if any(count <= 1 for count in visible_counts[:-1]):
        return False
    if any(count == 9 for count in visible_counts[:-1]):
        return False
    if min(visible_counts[:-1], default=9) > 6:
        return False

    coverage = int(np.sum(main != bg)) / float(main.size)
    if not (0.15 <= coverage <= 0.55):
        return False

    solution_sets, candidates, by_key = _solve_candidate_sets(main, patterns, bg, limit=2)
    if len(solution_sets) != 1:
        return False

    actual_ids = set()
    for kind, r0, c0 in placements:
        key = (kind, r0, c0)
        if key not in by_key:
            return False
        actual_ids.add(by_key[key])
    if next(iter(solution_sets)) != frozenset(actual_ids):
        return False

    for (kind, r0, c0), visible in zip(placements, visible_bits):
        cid = by_key[(kind, r0, c0)]
        for other_id, cand in enumerate(candidates):
            if other_id == cid:
                continue
            if visible & ~cand.bits:
                continue
            if cand.mismatch & visible:
                continue
            return False

    expected = np.full(main.shape, bg, dtype=int)
    for kind, r0, c0 in placements:
        expected[r0 + 1, c0 + 1] = int(patterns[kind][1, 1])
    try:
        solved = _render_solution(main, patterns, bg)
    except AssertionError:
        return False
    return np.array_equal(solved, expected)


def _build_input_grid(
    patterns: tuple[np.ndarray, np.ndarray], main: np.ndarray, bg: int, border: int
) -> np.ndarray:
    h, w = main.shape
    grid = np.full((BORDER_ROW + 1 + h, BORDER_COL + 1 + w), bg, dtype=int)
    grid[0:3, 0:3] = patterns[0]
    grid[0:3, 4:7] = patterns[1]
    grid[BORDER_ROW, :] = border
    grid[:, BORDER_COL] = border
    grid[BORDER_ROW + 1 :, BORDER_COL + 1 :] = main
    return grid


def _render_solution(
    main: np.ndarray, patterns: tuple[np.ndarray, np.ndarray], bg: int
) -> np.ndarray:
    placements = _recover_placements(main, patterns, bg)
    out = np.full(main.shape, bg, dtype=int)
    for cand in placements:
        out[cand.r0 + 1, cand.c0 + 1] = cand.center_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover every overlaid 3x3 stamp center from a legend of two stamp patterns.

    Input structure:
    - The upper-left corner is a legend area cut off by one full horizontal border
      line and one full vertical border line.
    - Inside that legend sit two different asymmetric 3x3 stamp patterns.
    - The lower-right main area is the visible result of stamping those patterns
      multiple times, with later stamps covering earlier overlaps.

    Rule:
    - Read the two legend patterns and ignore the border row and border column.
    - In the main area, a stamp placement is valid when every currently visible cell
      of its 3x3 footprint matches the legend pattern exactly, and every missing cell
      can only be hidden under later stamp footprints.
    - Reconstruct the unique full set of stamp placements, then return a blank grid
      the size of the main area with only each stamp center marked by that pattern's
      own center color.

    Key insight:
    - The composite trace must be explained causally from top layers downward: a
      partially visible 3x3 patch is accepted only if its absent cells are accounted
      for by other stamp footprints, so solving means inverting occlusion, not just
      spotting isolated motifs.
    """
    bg, patterns, main_area = _extract_parts(np.asarray(grid, dtype=int))
    return _render_solution(main_area, patterns, bg)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        bg, border = rng.sample(range(10), 2)
        palette = [color for color in range(10) if color not in {bg, border}]
        rng.shuffle(palette)
        palette = palette[:4]

        patterns = _sample_pattern_pair(rng, palette)
        stamp_count = rng.randint(*STAMP_COUNT_RANGE)
        (main_h, main_w), positions = _sample_chain_positions(rng, stamp_count)

        kinds = [rng.randrange(2) for _ in range(stamp_count)]
        if 0 not in kinds:
            kinds[rng.randrange(stamp_count)] = 0
        if 1 not in kinds:
            kinds[rng.randrange(stamp_count)] = 1

        placements = [(kind, r0, c0) for kind, (r0, c0) in zip(kinds, positions)]
        main_grid = _compose_main((main_h, main_w), patterns, placements, bg)
        if not _validate_instance(patterns, placements, main_grid, bg):
            continue

        return _build_input_grid(patterns, main_grid, bg, border)

    raise RuntimeError("failed to generate a valid stamp-inversion input")
