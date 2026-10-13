from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_for_fable_10_029_2", arc_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _ARC_UTILS = _load_arc_utils()
    Color = _ARC_UTILS.Color
except Exception:  # pragma: no cover - toolbox fallback
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9


BG = int(Color.BLACK)
STAMP = int(Color.GREY)
PANEL_PALETTE = np.array(
    [int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.YELLOW), int(Color.PINK), int(Color.ORANGE), int(Color.TEAL), int(Color.MAROON)],
    dtype=int,
)
MAX_GENERATION_ATTEMPTS = 800
_RNG = np.random.default_rng()
_SEEN_INPUT_SIGS: set[str] = set()
_SEEN_OUTPUT_SIGS: set[str] = set()


def _signature(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=int)
    return hashlib.md5(arr.tobytes() + str(tuple(arr.shape)).encode()).hexdigest()


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    n, width = map(int, g.shape)
    assert 3 <= n <= 5, "panel size n must satisfy 3 <= n <= 5"
    assert width == 2 * n + 1, "expected two n x n panels with a single divider column"

    divider = g[:, n]
    assert np.all(divider == STAMP), "expected a grey divider column"

    left = g[:, :n]
    right = g[:, n + 1 :]
    assert left.shape == (n, n) and right.shape == (n, n)

    left_colors = sorted(set(map(int, np.unique(left))) - {BG})
    right_colors = sorted(set(map(int, np.unique(right))) - {BG})
    assert len(left_colors) == 1, "expected exactly one left-panel foreground color"
    assert len(right_colors) == 1, "expected exactly one right-panel foreground color"
    assert left_colors[0] != right_colors[0], "expected distinct panel foreground colors"
    assert left_colors[0] != STAMP and right_colors[0] != STAMP, "grey is reserved for the divider"
    assert set(map(int, np.unique(left))).issubset({BG, left_colors[0]})
    assert set(map(int, np.unique(right))).issubset({BG, right_colors[0]})
    return left, right


def _hidden_mask(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left == BG) & (right == BG)


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    n = int(mask.shape[0])
    out = np.full((n * n, n * n), BG, dtype=int)
    stamp = np.where(mask, STAMP, BG).astype(int)
    for block_r, block_c in np.argwhere(mask):
        rs = int(block_r) * n
        cs = int(block_c) * n
        out[rs : rs + n, cs : cs + n] = stamp
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the double-empty complement mask from two aligned panels, then self-stamp that mask into a larger square.

    Input structure
    - The input is an n x (2n+1) grid with two n x n side panels separated by one grey divider column.
    - Each side panel uses black as background and exactly one distinct non-black foreground color.
    - The side panels are aligned cell-by-cell in the same row/column coordinate system.

    Rule
    - Build a binary mask M on the shared n x n panel coordinates: M[r, c] is true iff the left and right cells at (r, c) are both black.
    - Create an n^2 x n^2 black output partitioned into n x n blocks of size n x n.
    - For every true cell (r, c) in M, paste the whole mask M into output block (r, c), drawing true mask cells in grey and leaving false cells black.

    Key insight
    - The hidden template is neither visible panel alone nor their union/intersection; it is the complement of combined occupancy, and that same template serves as both the placement mask and the stamp.
    """
    left, right = _parse_input(grid)
    return _expand_mask(_hidden_mask(left, right))


def _mask_structure_ok(mask: np.ndarray) -> bool:
    n = int(mask.shape[0])
    true_count = int(mask.sum())
    false_count = int(mask.size - true_count)
    if true_count < 3 or false_count < 3:
        return False
    if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    if np.unique(mask.astype(int), axis=0).shape[0] < 2:
        return False
    if np.unique(mask.T.astype(int), axis=0).shape[0] < 2:
        return False
    if n >= 4:
        inner = mask[1:-1, 1:-1]
        if inner.size == 0 or not inner.any() or bool(inner.all()):
            return False
    return True


def _sample_mask(n: int) -> np.ndarray | None:
    total = n * n
    min_true = max(3, int(np.ceil(total * 0.25)))
    max_true = min(total - 3, int(np.floor(total * 0.6)))
    if min_true > max_true:
        return None

    for _ in range(200):
        true_count = int(_RNG.integers(min_true, max_true + 1))
        flat = np.zeros(total, dtype=bool)
        chosen = _RNG.choice(total, size=true_count, replace=False)
        flat[chosen] = True
        mask = flat.reshape(n, n)
        if _mask_structure_ok(mask):
            return mask
    return None


def _sample_panel_occupancy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(mask.shape[0])
    false_coords = np.argwhere(~mask)
    n_false = int(false_coords.shape[0])
    if n_false < 3:
        return None

    for _ in range(200):
        perm = _RNG.permutation(n_false)
        false_coords = false_coords[perm]
        cut_points = np.sort(_RNG.choice(np.arange(1, n_false), size=2, replace=False))
        left_only_end = int(cut_points[0])
        right_only_end = int(cut_points[1])

        left = np.zeros((n, n), dtype=bool)
        right = np.zeros((n, n), dtype=bool)

        for r, c in false_coords[:left_only_end]:
            left[int(r), int(c)] = True
        for r, c in false_coords[left_only_end:right_only_end]:
            right[int(r), int(c)] = True
        for r, c in false_coords[right_only_end:]:
            left[int(r), int(c)] = True
            right[int(r), int(c)] = True

        if left.sum() < 2 or right.sum() < 2:
            continue
        if np.array_equal(left, right):
            continue
        return left, right
    return None


def _render_input(left_occ: np.ndarray, right_occ: np.ndarray) -> np.ndarray:
    n = int(left_occ.shape[0])
    left_color, right_color = map(int, _RNG.choice(PANEL_PALETTE, size=2, replace=False))
    grid = np.full((n, 2 * n + 1), BG, dtype=int)
    grid[:, n] = STAMP
    grid[:, :n][left_occ] = left_color
    grid[:, n + 1 :][right_occ] = right_color
    return grid


def _semantic_example_ok(grid: np.ndarray) -> bool:
    try:
        left, right = _parse_input(grid)
    except AssertionError:
        return False

    mask = _hidden_mask(left, right)
    if not _mask_structure_ok(mask):
        return False

    left_occ = left != BG
    right_occ = right != BG
    left_only = left_occ & ~right_occ
    right_only = right_occ & ~left_occ
    both_occ = left_occ & right_occ

    if min(int(left_only.sum()), int(right_only.sum()), int(both_occ.sum())) < 1:
        return False

    correct = _expand_mask(mask)
    if not np.array_equal(main(grid), correct):
        return False

    shortcut_masks = [
        left == BG,
        right == BG,
        left_occ | right_occ,
        left_occ & right_occ,
        left_occ,
        right_occ,
    ]
    for alt_mask in shortcut_masks:
        if np.array_equal(mask, alt_mask):
            return False
        if np.array_equal(correct, _expand_mask(alt_mask)):
            return False

    if max(correct.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.integers(3, 6))
    mask = _sample_mask(n)
    if mask is None:
        return None

    occupancy = _sample_panel_occupancy(mask)
    if occupancy is None:
        return None
    left_occ, right_occ = occupancy
    grid = _render_input(left_occ, right_occ)

    if not _semantic_example_ok(grid):
        return None

    input_sig = _signature(grid)
    output_sig = _signature(main(grid))
    if input_sig in _SEEN_INPUT_SIGS or output_sig in _SEEN_OUTPUT_SIGS:
        return None

    _SEEN_INPUT_SIGS.add(input_sig)
    _SEEN_OUTPUT_SIGS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complement-stamp ARC input")
