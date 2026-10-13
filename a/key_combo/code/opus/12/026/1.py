import random
from collections import Counter

import numpy as np


_BLACK = 0
_FG_COLORS = (1, 2)

_QUADRANT_HEIGHT_RANGE = (5, 10)
_QUADRANT_WIDTH_RANGE = (5, 10)
_BASE_DENSITY_RANGE = (0.24, 0.52)
_MAX_GENERATION_ATTEMPTS = 400
_MAX_BASE_ATTEMPTS = 200


def _rot180(grid: np.ndarray) -> np.ndarray:
    return np.flipud(np.fliplr(grid))


def _find_separator(lines: np.ndarray) -> int:
    n = int(lines.shape[0])
    candidates = [idx for idx, is_sep in enumerate(lines.tolist()) if is_sep and idx == n - 1 - idx]
    assert len(candidates) == 1, "expected one centered separator line"
    return int(candidates[0])


def _split_quadrants(grid: np.ndarray) -> tuple[list[np.ndarray], int, int]:
    row_sep = _find_separator(np.all(grid == _BLACK, axis=1))
    col_sep = _find_separator(np.all(grid == _BLACK, axis=0))
    qh, qw = int(row_sep), int(col_sep)
    assert grid.shape == (2 * qh + 1, 2 * qw + 1), "quadrants must have equal size"
    quadrants = [
        grid[:qh, :qw].copy(),
        grid[:qh, col_sep + 1 :].copy(),
        grid[row_sep + 1 :, :qw].copy(),
        grid[row_sep + 1 :, col_sep + 1 :].copy(),
    ]
    return quadrants, row_sep, col_sep


def _to_base(quadrant: np.ndarray, index: int) -> np.ndarray:
    if index == 0:
        return _rot180(quadrant)
    if index == 1:
        return np.flipud(quadrant)
    if index == 2:
        return np.fliplr(quadrant)
    if index == 3:
        return quadrant.copy()
    raise ValueError("invalid quadrant index")


def _from_base(base: np.ndarray, index: int) -> np.ndarray:
    if index == 0:
        return _rot180(base)
    if index == 1:
        return np.flipud(base)
    if index == 2:
        return np.fliplr(base)
    if index == 3:
        return base.copy()
    raise ValueError("invalid quadrant index")


def _difference_output(actual: np.ndarray, correct: np.ndarray) -> np.ndarray:
    out = np.zeros_like(actual)
    diff = actual != correct
    out[diff] = actual[diff]
    return out


def _array_key(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, grid.shape)), grid.tobytes()


def main(grid: np.ndarray) -> np.ndarray:
    """Locate the corrupted mirror quadrant and extract only its wrong pixels.

    Input structure:
    - A black grid is split by one full black row and one full black column into four equal quadrants.
    - The four quadrants form a mirror mosaic: lower-right is the original, lower-left is its left-right mirror,
      upper-right is its up-down mirror, and upper-left is its 180-degree rotation.
    - Exactly one quadrant is corrupted by changing 1 to 3 pixels among black, blue, and red.

    Rule:
    - Find the separator row and column and cut out the four quadrants.
    - Convert every quadrant back into the lower-right orientation, then identify the one outlier against the
      three matching canonical quadrants.
    - Rebuild the corrupted quadrant's correct version from the shared canonical pattern.
    - Compare the actual corrupted quadrant with its rebuilt correct version and output a same-sized black grid
      that keeps only the actual colors at the differing pixels.

    Key insight:
    - The wrong quadrant is not defined by color frequency; it is the only block whose pixels break the required
      mirror consistency after all four quadrants are mapped into one common orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    quadrants, _row_sep, _col_sep = _split_quadrants(g)
    canonical = [_to_base(quadrant, idx) for idx, quadrant in enumerate(quadrants)]

    groups: dict[tuple[tuple[int, int], bytes], list[int]] = {}
    for idx, patch in enumerate(canonical):
        groups.setdefault(_array_key(patch), []).append(idx)

    assert len(groups) == 2, "expected three matching quadrants and one outlier"
    majority_key, majority_members = max(groups.items(), key=lambda item: len(item[1]))
    assert len(majority_members) == 3, "expected exactly one corrupted quadrant"

    tampered_idx = next(idx for idx in range(4) if idx not in majority_members)
    correct_base = canonical[majority_members[0]]
    assert _array_key(correct_base) == majority_key

    correct_quadrant = _from_base(correct_base, tampered_idx)
    actual_quadrant = quadrants[tampered_idx]
    return _difference_output(actual_quadrant, correct_quadrant)


def _has_required_color_mix(patch: np.ndarray) -> bool:
    return int(np.count_nonzero(patch == 1)) >= 2 and int(np.count_nonzero(patch == 2)) >= 2


def _is_base_patch_valid(patch: np.ndarray) -> bool:
    if patch.ndim != 2 or patch.size == 0:
        return False
    if np.any(np.all(patch == _BLACK, axis=1)) or np.any(np.all(patch == _BLACK, axis=0)):
        return False
    if int(np.count_nonzero(patch)) < 6:
        return False
    if not _has_required_color_mix(patch):
        return False
    if np.array_equal(patch, np.fliplr(patch)):
        return False
    if np.array_equal(patch, np.flipud(patch)):
        return False
    if np.array_equal(patch, _rot180(patch)):
        return False
    return True


def _generate_base_patch() -> np.ndarray:
    for _ in range(_MAX_BASE_ATTEMPTS):
        height = random.randint(*_QUADRANT_HEIGHT_RANGE)
        width = random.randint(*_QUADRANT_WIDTH_RANGE)
        patch = np.zeros((height, width), dtype=int)

        for r in range(height):
            count = random.randint(1, min(3, width))
            for c in random.sample(range(width), count):
                patch[r, c] = random.choice(_FG_COLORS)

        for c in range(width):
            if np.any(patch[:, c] != _BLACK):
                continue
            patch[random.randrange(height), c] = random.choice(_FG_COLORS)

        area = height * width
        lower = max(max(height, width) + 2, int(np.ceil(_BASE_DENSITY_RANGE[0] * area)))
        upper = max(lower, int(np.floor(_BASE_DENSITY_RANGE[1] * area)))
        target_fg = random.randint(lower, upper)

        empty_cells = [(r, c) for r in range(height) for c in range(width) if patch[r, c] == _BLACK]
        random.shuffle(empty_cells)
        need = max(0, target_fg - int(np.count_nonzero(patch)))
        for r, c in empty_cells[:need]:
            patch[r, c] = random.choice(_FG_COLORS)

        fg_coords = [(r, c) for r in range(height) for c in range(width) if patch[r, c] != _BLACK]
        if len(fg_coords) < 6:
            continue
        random.shuffle(fg_coords)
        patch[fg_coords[0]] = 1
        patch[fg_coords[1]] = 1
        patch[fg_coords[2]] = 2
        patch[fg_coords[3]] = 2

        if _is_base_patch_valid(patch):
            return patch

    raise RuntimeError("failed to generate a non-degenerate base patch")


def _apply_mutations(correct_quadrant: np.ndarray) -> np.ndarray:
    actual = correct_quadrant.copy()
    used: set[tuple[int, int]] = set()
    mutation_count = random.randint(1, 3)
    height, width = actual.shape

    for _ in range(mutation_count):
        unused = [(r, c) for r in range(height) for c in range(width) if (r, c) not in used]
        zero_cells = [(r, c) for r, c in unused if actual[r, c] == _BLACK]
        fg_cells = [(r, c) for r, c in unused if actual[r, c] in _FG_COLORS]

        categories = []
        if zero_cells:
            categories.append("add")
        if fg_cells:
            categories.append("erase")
            categories.append("swap")

        category = random.choice(categories)
        if category == "add":
            r, c = random.choice(zero_cells)
            actual[r, c] = random.choice(_FG_COLORS)
        elif category == "erase":
            r, c = random.choice(fg_cells)
            actual[r, c] = _BLACK
        else:
            r, c = random.choice(fg_cells)
            actual[r, c] = 2 if actual[r, c] == 1 else 1
        used.add((r, c))

    return actual


def _assemble_grid(base: np.ndarray, tampered_idx: int, tampered_quadrant: np.ndarray) -> np.ndarray:
    quadrants = [_from_base(base, idx) for idx in range(4)]
    quadrants[tampered_idx] = tampered_quadrant
    qh, qw = base.shape
    grid = np.zeros((2 * qh + 1, 2 * qw + 1), dtype=int)
    grid[:qh, :qw] = quadrants[0]
    grid[:qh, qw + 1 :] = quadrants[1]
    grid[qh + 1 :, :qw] = quadrants[2]
    grid[qh + 1 :, qw + 1 :] = quadrants[3]
    return grid


def _validate_instance(
    base: np.ndarray,
    grid: np.ndarray,
    tampered_idx: int,
    expected_output: np.ndarray,
) -> bool:
    if not _is_base_patch_valid(base):
        return False
    if int(np.sum(np.all(grid == _BLACK, axis=1))) != 1:
        return False
    if int(np.sum(np.all(grid == _BLACK, axis=0))) != 1:
        return False

    quadrants, _row_sep, _col_sep = _split_quadrants(grid)
    correct_quadrants = [_from_base(base, idx) for idx in range(4)]
    for idx, quadrant in enumerate(quadrants):
        if idx == tampered_idx:
            continue
        if not np.array_equal(quadrant, correct_quadrants[idx]):
            return False

    tampered = quadrants[tampered_idx]
    correct = correct_quadrants[tampered_idx]
    diff_count = int(np.count_nonzero(tampered != correct))
    if diff_count < 1 or diff_count > 3:
        return False

    canonical = [_to_base(quadrant, idx) for idx, quadrant in enumerate(quadrants)]
    counts = Counter(_array_key(patch) for patch in canonical)
    if sorted(counts.values()) != [1, 3]:
        return False

    if not np.array_equal(expected_output, _difference_output(tampered, correct)):
        return False
    if not np.array_equal(main(grid), expected_output):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        base = _generate_base_patch()
        tampered_idx = random.randrange(4)
        correct_quadrant = _from_base(base, tampered_idx)
        tampered_quadrant = _apply_mutations(correct_quadrant)
        expected_output = _difference_output(tampered_quadrant, correct_quadrant)
        grid = _assemble_grid(base, tampered_idx, tampered_quadrant)

        if _validate_instance(base, grid, tampered_idx, expected_output):
            return grid

    raise RuntimeError("failed to generate a valid mirror-defect ARC input")
