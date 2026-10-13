from __future__ import annotations

from collections import Counter

import numpy as np


ALL_COLORS = tuple(range(10))
RETRY_LIMIT = 400


def _strict_majority(values: np.ndarray) -> int | None:
    counts = Counter(map(int, values.tolist()))
    ranked = counts.most_common()
    if not ranked:
        return None
    if ranked[0][1] * 2 <= len(values):
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return int(ranked[0][0])


def _period_consensus(grid: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = grid.shape
    if height % period != 0:
        return None
    copies = height // period
    if copies < 3:
        return None

    blocks = grid.reshape(copies, period, width)
    motif = np.empty((period, width), dtype=int)
    local_damage = np.zeros((period, width), dtype=int)

    for r in range(period):
        for c in range(width):
            color = _strict_majority(blocks[:, r, c])
            if color is None:
                return None
            motif[r, c] = color
            local_damage[r, c] = int(np.count_nonzero(blocks[:, r, c] != color))
            if local_damage[r, c] * 2 >= copies:
                return None

    diff = blocks != motif[None, :, :]
    per_copy_damage = diff.reshape(copies, period * width).sum(axis=1)
    if np.any(per_copy_damage == 0):
        return None
    return motif, diff


def _valid_periods(grid: np.ndarray) -> list[tuple[int, np.ndarray, np.ndarray]]:
    height = int(grid.shape[0])
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for period in range(1, height + 1):
        result = _period_consensus(grid, period)
        if result is None:
            continue
        motif, diff = result
        out.append((period, motif, diff))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Identify the smallest vertical repeat and compress the repaired majority motif to one period.

    Input structure:
    - The full grid consists of one unknown row motif of height `p` repeated an integer
      number of times vertically, with at least three repeats.
    - Some scattered cells in the repeated copies have been overwritten with other colors.
    - At every motif coordinate, fewer than half of the copies are damaged there, and every
      copy contains at least one damaged cell.

    Rule:
    - Find the smallest valid vertical period `p`.
    - For each local coordinate inside that `p x W` motif, take the strict majority color
      across all repeated copies.
    - Output only the repaired consensus motif of height `p`, keeping the original width.

    Key insight:
    - No single copy is trustworthy, so the motif must be recovered by per-coordinate voting
      across vertically aligned copies, and the answer is the recovered single period rather
      than the full repeated canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    candidates = _valid_periods(g)
    assert candidates, "expected at least one valid periodic consensus"
    period, motif, _diff = min(candidates, key=lambda item: item[0])
    assert period <= g.shape[0] // 3, "a valid input must contain at least three copies"
    return motif


def _exact_row_period(motif: np.ndarray) -> int:
    period = int(motif.shape[0])
    for candidate in range(1, period):
        if period % candidate != 0:
            continue
        if all(np.array_equal(motif[r], motif[r % candidate]) for r in range(period)):
            return candidate
    return period


def _build_segment_motif(rng: np.random.Generator, period: int, width: int, palette: list[int]) -> np.ndarray:
    bg = int(palette[0])
    colors = list(map(int, palette[1:]))
    motif = np.full((period, width), bg, dtype=int)

    for r in range(period):
        segment_count = int(rng.integers(1, min(3, width) + 1))
        starts = rng.choice(width, size=segment_count, replace=False)
        for start in starts:
            max_len = min(width - int(start), max(2, width // 2))
            length = int(rng.integers(1, max_len + 1))
            motif[r, int(start) : int(start) + length] = int(rng.choice(colors))

    for _ in range(int(rng.integers(1, 3))):
        col = int(rng.integers(0, width))
        band_w = int(rng.integers(1, min(3, width - col) + 1))
        top = int(rng.integers(0, period))
        band_h = int(rng.integers(1, period - top + 1))
        motif[top : top + band_h, col : col + band_w] = int(rng.choice(colors))

    return motif


def _build_patch_motif(rng: np.random.Generator, period: int, width: int, palette: list[int]) -> np.ndarray:
    bg = int(palette[0])
    colors = list(map(int, palette[1:]))
    motif = np.full((period, width), bg, dtype=int)

    patch_count = int(rng.integers(period + 1, period + 5))
    for _ in range(patch_count):
        h = int(rng.integers(1, min(3, period) + 1))
        w = int(rng.integers(1, min(max(2, width // 2), width) + 1))
        r0 = int(rng.integers(0, period - h + 1))
        c0 = int(rng.integers(0, width - w + 1))
        motif[r0 : r0 + h, c0 : c0 + w] = int(rng.choice(colors))

    for r in range(period):
        if np.all(motif[r] == bg):
            start = int(rng.integers(0, width))
            length = int(rng.integers(1, min(4, width - start) + 1))
            motif[r, start : start + length] = int(rng.choice(colors))

    return motif


def _refine_motif(rng: np.random.Generator, motif: np.ndarray, palette: list[int]) -> np.ndarray:
    period, width = motif.shape
    bg = int(palette[0])
    colors = list(map(int, palette))

    for _ in range(30):
        if _exact_row_period(motif) == period:
            break
        r = int(rng.integers(0, period))
        c = int(rng.integers(0, width))
        choices = [color for color in colors if color != int(motif[r, c])]
        motif[r, c] = int(rng.choice(choices))

    non_bg = int(np.count_nonzero(motif != bg))
    area = int(period * width)
    used = set(map(int, np.unique(motif)))
    if len(used) < 3:
        return motif[:0]
    if non_bg < max(4, area // 5):
        return motif[:0]
    if non_bg > (4 * area) // 5:
        return motif[:0]
    if len({motif[r].tobytes() for r in range(period)}) < 2:
        return motif[:0]
    if _exact_row_period(motif) != period:
        return motif[:0]
    return motif


def _make_motif(rng: np.random.Generator, period: int, width: int) -> np.ndarray | None:
    palette_size = int(rng.integers(3, 6))
    palette = list(map(int, rng.choice(ALL_COLORS, size=palette_size, replace=False)))
    if rng.random() < 0.5:
        motif = _build_segment_motif(rng, period, width, palette)
    else:
        motif = _build_patch_motif(rng, period, width, palette)
    motif = _refine_motif(rng, motif, palette)
    if motif.size == 0:
        return None
    return motif


def _assign_damage(
    rng: np.random.Generator,
    blocks: np.ndarray,
    copy_idx: int,
    r: int,
    c: int,
) -> None:
    original = int(blocks[copy_idx, r, c])
    choices = [color for color in ALL_COLORS if color != original]
    blocks[copy_idx, r, c] = int(rng.choice(choices))


def _damage_repeated_grid(
    rng: np.random.Generator,
    motif: np.ndarray,
    copies: int,
) -> tuple[np.ndarray, np.ndarray]:
    period, width = motif.shape
    max_local_damage = (copies - 1) // 2
    blocks = np.repeat(motif[None, :, :], copies, axis=0)
    damaged = np.zeros((copies, period, width), dtype=bool)
    local_damage = np.zeros((period, width), dtype=int)
    copy_damage = np.zeros(copies, dtype=int)
    coords = [(r, c) for r in range(period) for c in range(width)]
    rng.shuffle(coords)

    for copy_idx in rng.permutation(copies):
        options = [(r, c) for r, c in coords if local_damage[r, c] < max_local_damage]
        r, c = options[int(rng.integers(0, len(options)))]
        _assign_damage(rng, blocks, int(copy_idx), r, c)
        damaged[int(copy_idx), r, c] = True
        local_damage[r, c] += 1
        copy_damage[int(copy_idx)] += 1

    capacity = int(period * width * max_local_damage)
    min_total = max(copies + 1, 4)
    max_total = min(capacity, copies + max(3, (period * width) // 3))
    target_total = int(rng.integers(min_total, max_total + 1))

    while int(damaged.sum()) < target_total:
        existing = [(r, c) for r, c in coords if 0 < local_damage[r, c] < max_local_damage]
        fresh = [(r, c) for r, c in coords if local_damage[r, c] == 0]
        if existing and rng.random() < 0.45:
            r, c = existing[int(rng.integers(0, len(existing)))]
        elif fresh:
            r, c = fresh[int(rng.integers(0, len(fresh)))]
        elif existing:
            r, c = existing[int(rng.integers(0, len(existing)))]
        else:
            break

        copy_options = [idx for idx in range(copies) if not damaged[idx, r, c]]
        if not copy_options:
            local_damage[r, c] = max_local_damage
            continue
        min_damage = min(int(copy_damage[idx]) for idx in copy_options)
        balanced = [idx for idx in copy_options if int(copy_damage[idx]) == min_damage]
        copy_idx = int(balanced[int(rng.integers(0, len(balanced)))])
        _assign_damage(rng, blocks, copy_idx, r, c)
        damaged[copy_idx, r, c] = True
        local_damage[r, c] += 1
        copy_damage[copy_idx] += 1

    return blocks.reshape(period * copies, width), damaged


def _semantic_check(grid: np.ndarray, motif: np.ndarray, period: int) -> bool:
    solved = main(grid)
    if not np.array_equal(solved, motif):
        return False

    candidates = _valid_periods(grid)
    if not candidates or min(period0 for period0, _, _ in candidates) != period:
        return False

    copies = grid.shape[0] // period
    blocks = grid.reshape(copies, period, grid.shape[1])
    diff = blocks != motif[None, :, :]
    if np.any(diff.reshape(copies, -1).sum(axis=1) == 0):
        return False
    if len({diff[idx].tobytes() for idx in range(copies)}) < 2:
        return False

    local_damage = diff.sum(axis=0)
    if np.any(local_damage * 2 >= copies):
        return False
    if int(np.count_nonzero(local_damage)) < 2:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    period = int(rng.integers(2, 9))
    max_copies = min(6, 30 // period)
    copies = int(rng.integers(3, max_copies + 1))
    width = int(rng.integers(5, 13))

    motif = _make_motif(rng, period, width)
    if motif is None:
        return None

    grid, _damaged = _damage_repeated_grid(rng, motif, copies)
    if _semantic_check(grid, motif, period):
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-consensus input")
