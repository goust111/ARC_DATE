from __future__ import annotations

import itertools
import random
from collections import Counter

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Mirror the shared ARC helper: background = most common color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _mirror_col(width: int, c: int) -> int:
    return int(width) - 1 - int(c)


def _paired_mask(grid: np.ndarray) -> np.ndarray:
    fg = np.asarray(grid) != 0
    return fg & np.fliplr(fg)


def _isolated_mask(grid: np.ndarray) -> np.ndarray:
    fg = np.asarray(grid) != 0
    return fg & ~_paired_mask(grid)


def _paired_color_counts(grid: np.ndarray) -> Counter[int]:
    paired = _paired_mask(grid)
    return Counter(int(v) for v in np.asarray(grid)[paired])


def _solve_with_counts(grid: np.ndarray, counts: Counter[int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = map(int, g.shape)
    out = np.zeros_like(g)

    for r in range(height):
        for c in range(width // 2):
            mc = _mirror_col(width, c)
            left = int(g[r, c])
            right = int(g[r, mc])
            if left == 0 or right == 0:
                continue
            if left == right:
                winner = left
            else:
                left_count = int(counts[left])
                right_count = int(counts[right])
                assert left_count != right_count, "paired-color frequencies must be distinct"
                winner = left if left_count > right_count else right
            out[r, c] = winner
            out[r, mc] = winner
    return out


def _solve_preserving_isolates(grid: np.ndarray, counts: Counter[int]) -> np.ndarray:
    out = _solve_with_counts(grid, counts)
    isolated = _isolated_mask(grid)
    out[isolated] = np.asarray(grid)[isolated]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase unpaired pixels and resolve each horizontal mirror pair by the higher paired-set color frequency.

    Input structure:
    - The grid has a black background on an even-width canvas.
    - Foreground consists of scattered single pixels in 3-4 nonblack colors.
    - Some foreground pixels have a foreground horizontal mirror partner across the vertical midline, while others mirror to black.

    Rule:
    - For each cell (r, c), compare it with its horizontal mirror (r, W-1-c).
    - If exactly one side is foreground, that foreground pixel is isolated and is erased to black in the output.
    - If both sides are foreground, the two cells form one mirror pair; count colors only over this paired-pixel set.
    - Same-color pairs stay unchanged; different-color pairs are recolored on both sides to the color with higher frequency inside the paired-pixel set.
    - Background cells stay black.

    Key insight:
    - Mirror membership determines survival, and frequency ranking is computed only on surviving paired pixels rather than on all foreground pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _, width = map(int, g.shape)
    assert width % 2 == 0, "expected an even-width grid"
    assert 0 in g, "expected black background"

    paired_counts = _paired_color_counts(g)
    assert paired_counts, "expected at least one mirror pair"
    freqs = list(paired_counts.values())
    assert len(freqs) == len(set(freqs)), "paired-set color frequencies must be unique"

    return _solve_with_counts(g, paired_counts)


# --------------- generator ---------------

HEIGHT_RANGE = (10, 15)
EVEN_WIDTHS = (12, 14, 16)
PAIR_COLOR_COUNT_CHOICES = (3, 4)
PAIR_COUNT_PATTERNS = {
    3: ((7, 5, 4), (6, 5, 3), (8, 5, 3), (5, 4, 3)),
    4: ((6, 5, 4, 3), (5, 4, 3, 2), (7, 4, 3, 2), (6, 4, 3, 1)),
}
INPUT_FG_RANGE = (20, 30)
ISOLATED_PER_COLOR_MAX = 5
MAX_PAIRING_ATTEMPTS = 300
MAX_PLACEMENT_ATTEMPTS = 60
MAX_GENERATION_ATTEMPTS = 400


def _output_colors_for_pairs(
    pairs: list[tuple[int, int]],
    pair_counts: dict[int, int],
) -> set[int]:
    colors = set()
    for left, right in pairs:
        if left == right:
            colors.add(int(left))
        else:
            colors.add(int(left if pair_counts[left] > pair_counts[right] else right))
    return colors


def _build_pair_list(pair_counts: dict[int, int], rng: random.Random) -> list[tuple[int, int]]:
    endpoints = [color for color, count in pair_counts.items() for _ in range(int(count))]
    assert len(endpoints) % 2 == 0

    for _ in range(MAX_PAIRING_ATTEMPTS):
        rng.shuffle(endpoints)
        pairs = list(zip(endpoints[::2], endpoints[1::2]))
        diff_pairs = [(a, b) for a, b in pairs if a != b]
        if not any(a == b for a, b in pairs):
            continue
        if len(diff_pairs) < 2:
            continue
        if len({color for a, b in diff_pairs for color in (a, b)}) < min(3, len(pair_counts)):
            continue
        if len(_output_colors_for_pairs(pairs, pair_counts)) < 2:
            continue
        return pairs

    raise RuntimeError("failed to construct a varied mirror-pair multiset")


def _count_conflicting_pairs(
    pairs: list[tuple[int, int]],
    pair_counts: dict[int, int],
    total_counts: dict[int, int],
) -> int:
    conflicts = 0
    for left, right in pairs:
        if left == right:
            continue
        pair_winner = left if pair_counts[left] > pair_counts[right] else right
        total_winner = left if total_counts[left] > total_counts[right] else right
        if pair_winner != total_winner:
            conflicts += 1
    return conflicts


def _choose_isolated_counts(
    pair_counts: dict[int, int],
    pairs: list[tuple[int, int]],
    rng: random.Random,
) -> dict[int, int]:
    colors = list(pair_counts)
    pair_total = sum(pair_counts.values())
    min_isolated = max(4, INPUT_FG_RANGE[0] - pair_total)
    max_isolated = min(10, INPUT_FG_RANGE[1] - pair_total)
    assert min_isolated <= max_isolated

    valid: list[tuple[int, dict[int, int]]] = []
    for values in itertools.product(range(ISOLATED_PER_COLOR_MAX + 1), repeat=len(colors)):
        isolated_total = sum(values)
        if not (min_isolated <= isolated_total <= max_isolated):
            continue

        isolated_counts = {color: int(value) for color, value in zip(colors, values)}
        total_counts = {color: int(pair_counts[color] + isolated_counts[color]) for color in colors}
        if len(set(total_counts.values())) != len(total_counts):
            continue

        conflicts = _count_conflicting_pairs(pairs, pair_counts, total_counts)
        if conflicts == 0:
            continue

        pair_top = max(colors, key=lambda color: pair_counts[color])
        global_top = max(colors, key=lambda color: total_counts[color])
        score = 10 * conflicts
        if pair_top != global_top:
            score += 5
        score += sum(value > 0 for value in values)
        valid.append((score, isolated_counts))

    assert valid, "failed to find isolated counts that force paired-set frequency arbitration"
    best_score = max(score for score, _ in valid)
    finalists = [counts for score, counts in valid if score >= best_score - 2]
    return dict(rng.choice(finalists))


def _too_close(pos: tuple[int, int], occupied: set[tuple[int, int]], gap: int) -> bool:
    if gap <= 0:
        return False
    r, c = pos
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= gap:
            return True
    return False


def _sample_positions(
    height: int,
    width: int,
    n_pairs: int,
    n_isolated: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for gap in (1, 0):
        occupied: set[tuple[int, int]] = set()
        pair_left_positions: list[tuple[int, int]] = []
        row_pair_counts: Counter[int] = Counter()
        left_cells = [(r, c) for r in range(height) for c in range(width // 2)]
        rng.shuffle(left_cells)

        for r, c in left_cells:
            mirror = (r, _mirror_col(width, c))
            if row_pair_counts[r] >= 2:
                continue
            if (r, c) in occupied or mirror in occupied:
                continue
            if _too_close((r, c), occupied, gap) or _too_close(mirror, occupied, gap):
                continue
            pair_left_positions.append((r, c))
            occupied.add((r, c))
            occupied.add(mirror)
            row_pair_counts[r] += 1
            if len(pair_left_positions) == n_pairs:
                break

        if len(pair_left_positions) != n_pairs:
            continue

        isolated_positions: list[tuple[int, int]] = []
        row_isolated_counts: Counter[int] = Counter()
        all_cells = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(all_cells)

        for pos in all_cells:
            r, c = pos
            mirror = (r, _mirror_col(width, c))
            if row_isolated_counts[r] >= 2:
                continue
            if pos in occupied or mirror in occupied:
                continue
            if _too_close(pos, occupied, gap):
                continue
            isolated_positions.append(pos)
            occupied.add(pos)
            row_isolated_counts[r] += 1
            if len(isolated_positions) == n_isolated:
                return pair_left_positions, isolated_positions

    return None


def _build_case(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = int(rng.choice(EVEN_WIDTHS))
    n_pair_colors = int(rng.choice(PAIR_COLOR_COUNT_CHOICES))
    active_colors = rng.sample(list(range(1, 10)), n_pair_colors)

    pair_pattern = list(rng.choice(PAIR_COUNT_PATTERNS[n_pair_colors]))
    rng.shuffle(pair_pattern)
    pair_counts = {color: int(count) for color, count in zip(active_colors, pair_pattern)}
    pairs = _build_pair_list(pair_counts, rng)
    isolated_counts = _choose_isolated_counts(pair_counts, pairs, rng)

    placement = _sample_positions(
        height=height,
        width=width,
        n_pairs=len(pairs),
        n_isolated=sum(isolated_counts.values()),
        rng=rng,
    )
    if placement is None:
        return None
    pair_left_positions, isolated_positions = placement

    grid = np.zeros((height, width), dtype=int)
    rng.shuffle(pairs)
    for (r, c), (a, b) in zip(pair_left_positions, pairs):
        left, right = (a, b) if rng.random() < 0.5 else (b, a)
        mc = _mirror_col(width, c)
        grid[r, c] = int(left)
        grid[r, mc] = int(right)

    isolated_colors = [color for color, count in isolated_counts.items() for _ in range(int(count))]
    rng.shuffle(isolated_positions)
    rng.shuffle(isolated_colors)
    for pos, color in zip(isolated_positions, isolated_colors):
        grid[pos] = int(color)

    return grid


def _validate_generated_case(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30
    assert width % 2 == 0
    assert get_bg(g) == 0

    fg_colors = colors_present(g, exclude={0})
    assert 3 <= len(fg_colors) <= 4
    fg_count = int(np.count_nonzero(g))
    assert INPUT_FG_RANGE[0] <= fg_count <= INPUT_FG_RANGE[1]

    pair_counts = _paired_color_counts(g)
    global_counts = Counter(int(v) for v in g[g != 0])
    same_pairs = 0
    diff_pairs = 0
    isolated_pixels = 0
    conflicting_pairs = 0

    for r in range(height):
        for c in range(width // 2):
            mc = _mirror_col(width, c)
            left = int(g[r, c])
            right = int(g[r, mc])
            if left != 0 and right != 0:
                if left == right:
                    same_pairs += 1
                else:
                    diff_pairs += 1
                    pair_winner = left if pair_counts[left] > pair_counts[right] else right
                    total_winner = left if global_counts[left] > global_counts[right] else right
                    if pair_winner != total_winner:
                        conflicting_pairs += 1
            elif (left != 0) ^ (right != 0):
                isolated_pixels += 1

    assert len(pair_counts) == len(fg_colors)
    assert len(set(pair_counts.values())) == len(pair_counts)
    assert len(set(global_counts.values())) == len(global_counts)
    assert same_pairs >= 1
    assert diff_pairs >= 1
    assert isolated_pixels >= 1
    assert conflicting_pairs >= 1

    out = main(g)
    wrong_global = _solve_with_counts(g, global_counts)
    wrong_keep = _solve_preserving_isolates(g, pair_counts)

    assert np.any(out != g), "the puzzle must transform the input"
    assert not np.array_equal(out, wrong_global), "generator must defeat global-frequency shortcuts"
    assert not np.array_equal(out, wrong_keep), "generator must require isolated-pixel erasure"
    assert not np.any(out[_isolated_mask(g)]), "isolated pixels must be erased"
    assert 2 <= len(colors_present(out, exclude={0}))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        try:
            grid = _build_case(rng)
        except (AssertionError, RuntimeError):
            continue
        if grid is None:
            continue
        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid mirror-frequency arbitration input")
