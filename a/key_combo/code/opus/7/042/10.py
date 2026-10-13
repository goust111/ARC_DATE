import numpy as np


BLACK = 0
RED = 2


GEN_HEIGHT_MIN = 10
GEN_HEIGHT_MAX = 16
GEN_PERIODS = (2, 3, 4)
GEN_REPEATS = (3, 4)
GEN_PSEUDO_MIN = 2
GEN_PSEUDO_MAX = 3
GEN_NOISE_MIN = 2
GEN_NOISE_MAX = 4
GEN_ATTEMPTS = 500
GEN_TEMPLATE_ATTEMPTS = 200


def _paired_mask(grid: np.ndarray) -> np.ndarray:
    fg = np.asarray(grid, dtype=int) == RED
    return fg & np.fliplr(fg)


def _period_keep_and_score(left: np.ndarray, period: int) -> tuple[int, int, np.ndarray]:
    height, half = left.shape
    keep = np.zeros_like(left, dtype=bool)
    ambiguous = 0
    disagreements = 0

    for r in range(height):
        row = left[r]
        for residue in range(period):
            cols = np.arange(residue, half, period)
            present = int(row[cols].sum())
            total = int(cols.size)
            absent = total - present
            if present * 2 == total:
                ambiguous += 1
            if present > absent:
                keep[r, cols] = row[cols]
                disagreements += absent
            else:
                disagreements += present

    return ambiguous, disagreements, keep


def _period_table(left: np.ndarray) -> list[tuple[int, int, int]]:
    half = int(left.shape[1])
    upper = max(1, half // 2)
    stats = []
    for period in range(1, upper + 1):
        ambiguous, disagreements, _ = _period_keep_and_score(left, period)
        stats.append((period, ambiguous, disagreements))
    return stats


def _infer_period(left: np.ndarray) -> tuple[int, np.ndarray, tuple[int, int]]:
    best_signature = None
    best_period = 1
    best_keep = np.zeros_like(left, dtype=bool)

    for period, ambiguous, disagreements in _period_table(left):
        signature = (ambiguous, disagreements, period)
        if best_signature is None or signature < best_signature:
            best_signature = signature
            best_period = period
            _, _, best_keep = _period_keep_and_score(left, period)

    assert best_signature is not None
    return best_period, best_keep, best_signature[:2]


def main(grid: np.ndarray) -> np.ndarray:
    """Keep only the red mirror pairs that also agree with the smallest repeated horizontal period.

    Input structure:
    - The grid has a black background, even width, and uses red as the only foreground color.
    - Some red pixels appear in left-right mirror pairs across the vertical midline.
    - Additional noise pixels may be unpaired, and some paired pixels may be periodic outliers.

    Rule:
    - First keep only red pixels whose same-row mirror partner is also red.
    - On the left half of that paired mask, infer the smallest horizontal period that repeats at least twice by grouping columns with the same residue mod P.
    - Within each row/residue class, keep the class only when foreground is the strict majority; erase minority paired outliers and mirror the decision back to the right half.

    Key insight:
    - Mirror symmetry removes asymmetric noise first, then a residue-class majority vote over repeated columns identifies which symmetric pairs belong to the true periodic structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    assert width % 2 == 0 and width >= 4

    paired = _paired_mask(g)
    half = width // 2
    left = paired[:, :half]
    _, keep_left, _ = _infer_period(left)

    keep = np.zeros_like(paired, dtype=bool)
    keep[:, :half] = keep_left
    keep[:, half:] = np.fliplr(keep_left)

    out = np.zeros_like(g)
    out[keep] = RED
    return out


def _sample_row_templates(rng: np.random.Generator, height: int, period: int) -> np.ndarray | None:
    max_active = max(4, (2 * height) // 3)
    min_active = max(4, height // 3)

    for _ in range(GEN_TEMPLATE_ATTEMPTS):
        templates = np.zeros((height, period), dtype=bool)
        active_rows = int(rng.integers(min_active, max_active + 1))
        rows = rng.choice(height, size=active_rows, replace=False)
        rows.sort()

        previous = None
        for r in rows:
            if previous is not None and rng.random() < 0.35:
                templates[r] = previous
                continue

            max_residues = min(period - 1, 2)
            active_residues = int(rng.integers(1, max_residues + 1))
            residues = rng.choice(period, size=active_residues, replace=False)
            templates[r, residues] = True
            previous = templates[r].copy()

        if templates.any(axis=1).sum() < 4:
            continue
        if not (~templates).any():
            continue
        if period >= 3 and not np.any(templates.sum(axis=1) >= 2):
            continue

        return templates

    return None


def _repeat_templates(templates: np.ndarray, repeats: int) -> np.ndarray:
    return np.tile(templates, (1, repeats))


def _true_pattern_grid(left_true: np.ndarray) -> np.ndarray:
    return np.hstack([left_true, np.fliplr(left_true)]).astype(int) * RED


def _noise_slots(grid: np.ndarray) -> list[tuple[int, int]]:
    height, width = grid.shape
    half = width // 2
    slots = []
    for r in range(height):
        for c in range(half):
            mirror = width - 1 - c
            if grid[r, c] == BLACK and grid[r, mirror] == BLACK:
                slots.append((r, c))
    return slots


def _add_unpaired_noise(rng: np.random.Generator, grid: np.ndarray, count: int) -> bool:
    slots = _noise_slots(grid)
    if len(slots) < count:
        return False

    choices = rng.choice(len(slots), size=count, replace=False)
    for idx in choices:
        r, c = slots[int(idx)]
        if rng.random() < 0.5:
            grid[r, c] = RED
        else:
            grid[r, grid.shape[1] - 1 - c] = RED
    return True


def _inject_pseudo_pairs(
    rng: np.random.Generator,
    left_true: np.ndarray,
    period: int,
    pseudo_count: int,
) -> np.ndarray | None:
    left = left_true.copy()
    height, half = left.shape
    candidates = []

    for r in range(height):
        for residue in range(period):
            cols = np.arange(residue, half, period)
            if cols.size < 3:
                continue
            if np.any(left[r, cols]):
                continue
            candidates.append((r, cols))

    if len(candidates) < pseudo_count:
        return None

    picks = rng.choice(len(candidates), size=pseudo_count, replace=False)
    for idx in picks:
        r, cols = candidates[int(idx)]
        left[r, int(rng.choice(cols))] = True
    return left


def _is_unique_best_period(left: np.ndarray, period: int) -> bool:
    stats = _period_table(left)
    best_pair = min((ambiguous, disagreements) for _, ambiguous, disagreements in stats)
    winners = [p for p, ambiguous, disagreements in stats if (ambiguous, disagreements) == best_pair]
    return winners == [period]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    period = int(rng.choice(GEN_PERIODS))
    repeats = int(rng.choice(GEN_REPEATS))
    half = period * repeats
    width = half * 2
    height = int(rng.integers(GEN_HEIGHT_MIN, GEN_HEIGHT_MAX + 1))

    templates = _sample_row_templates(rng, height, period)
    if templates is None:
        return None

    left_true = _repeat_templates(templates, repeats)
    true_fg = int(left_true.sum())
    if true_fg < height * 2 or true_fg > (height * half) // 2:
        return None

    inferred_true, _, true_score = _infer_period(left_true)
    if inferred_true != period or true_score != (0, 0):
        return None

    pseudo_count = int(rng.integers(GEN_PSEUDO_MIN, GEN_PSEUDO_MAX + 1))
    left_with_pseudo = _inject_pseudo_pairs(rng, left_true, period, pseudo_count)
    if left_with_pseudo is None:
        return None

    inferred_pseudo, keep_left, pseudo_score = _infer_period(left_with_pseudo)
    if inferred_pseudo != period:
        return None
    if pseudo_score[0] != 0 or pseudo_score[1] != pseudo_count:
        return None
    if not _is_unique_best_period(left_with_pseudo, period):
        return None
    if not np.array_equal(keep_left, left_true):
        return None

    grid = _true_pattern_grid(left_with_pseudo)
    noise_count = int(rng.integers(GEN_NOISE_MIN, GEN_NOISE_MAX + 1))
    if not _add_unpaired_noise(rng, grid, noise_count):
        return None

    if not np.array_equal(main(grid), _true_pattern_grid(left_true)):
        return None
    if width > 30 or height > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for task 10")
