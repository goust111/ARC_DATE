from __future__ import annotations

import sys
import numpy as np

try:
    from arc_utils import Color
except Exception:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color


def _find_separator_block(g: np.ndarray) -> tuple[int, int, int]:
    h, w = g.shape
    bg = Color.BLACK
    columns = []
    c = 0
    while c < w:
        color = int(g[0, c])
        end = c
        while end + 1 < w and np.all(g[:, end + 1] == color):
            end += 1
        if color != bg:
            columns.append((c, end, color))
        c = end + 1

    # Prefer the gray divider; fall back only if needed.
    gray_cols = [c for c in columns if c[2] == Color.GRAY and c[0] > 0 and c[1] < w - 1]
    if len(gray_cols) >= 1:
        gray_cols.sort(
            key=lambda item: (item[1] - item[0], abs(((item[0] + item[1]) / 2) - (w - 1) / 2))
        )
        return gray_cols[0]

    for c0, c1, color in columns:
        if c0 > 0 and c1 < w - 1:
            return c0, c1, color

    raise ValueError("no separator block found")


def _split_sides(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    sep_start, sep_end, _ = _find_separator_block(grid)
    left = grid[:, :sep_start]
    right = grid[:, sep_end + 1 :]
    if left.shape[1] != right.shape[1] or left.size == 0 or right.size == 0:
        raise ValueError("left/right masks must exist and be same width")
    return left, right, sep_start, sep_end


def _side_color(side: np.ndarray) -> int:
    bg = Color.BLACK
    colors = np.setdiff1d(np.unique(side), [bg])
    if colors.size != 1:
        raise ValueError("each side must use exactly one non-black color")
    return int(colors[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the row-wise minority-exclusive territory (or overlap when tied) between two side-by-side masks.

    Input structure:
    - Input is a black background split by a gray separator column into left and right masks.
    - Left mask is monochrome blue on black; right mask is monochrome red on black.
    - The two masks are equal in height and width.

    Rule:
    - For each row, count active cells on both sides (excluding black background).
    - If left_count < right_count, keep cells active only on the left and inactive on the right at that row.
    - If right_count < left_count, keep cells active only on the right and inactive on the left at that row.
    - If counts tie, keep only cells where both sides are active (intersection).
    - Output is a black canvas with the same shape as one side mask; kept cells are green.

    Key insight:
    - Each row independently dispatches to a different set operation, so the operation is controlled by row-local majority/minority comparison instead of global cues.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")

    left, right, *_ = _split_sides(g)
    left_color = _side_color(left)
    right_color = _side_color(right)
    if left.shape[1] != right.shape[1]:
        raise ValueError("left and right widths must match")

    out = np.full((g.shape[0], left.shape[1]), Color.BLACK, dtype=int)
    for r in range(g.shape[0]):
        l = left[r] == left_color
        rr = right[r] == right_color
        left_count = int(np.count_nonzero(l))
        right_count = int(np.count_nonzero(rr))
        if left_count < right_count:
            keep = l & ~rr
        elif right_count < left_count:
            keep = rr & ~l
        else:
            keep = l & rr
        out[r, keep] = Color.GREEN
    return out


def _sample_counts(rng: np.random.Generator, w: int, relation: str) -> tuple[int, int]:
    if relation == "blue_weak":
        left = int(rng.integers(1, w))
        right = int(rng.integers(left + 1, w + 1))
        return left, right
    if relation == "red_weak":
        right = int(rng.integers(1, w))
        left = int(rng.integers(right + 1, w + 1))
        return left, right
    if relation == "tie":
        if rng.random() < 0.2:
            k = 0
        else:
            k = int(rng.integers(1, w + 1))
        return k, k
    raise ValueError(f"invalid relation: {relation}")


def _row_masks(rng: np.random.Generator, w: int, relation: str) -> tuple[np.ndarray, np.ndarray]:
    left_count, right_count = _sample_counts(rng, w, relation)
    left_row = np.zeros(w, dtype=bool)
    right_row = np.zeros(w, dtype=bool)

    overlap_max = min(left_count, right_count)
    overlap = int(rng.integers(0, overlap_max + 1))
    left_only = left_count - overlap
    right_only = right_count - overlap

    perm = rng.permutation(w)
    overlap_cols = perm[:overlap]
    left_only_cols = perm[overlap : overlap + left_only]
    right_only_cols = perm[overlap + left_only : overlap + left_only + right_only]

    left_row[overlap_cols] = True
    right_row[overlap_cols] = True
    left_row[left_only_cols] = True
    right_row[right_only_cols] = True

    # Add a light shape variation while preserving counts.
    if rng.random() < 0.35:
        if left_row.sum() >= 3 and (rng.random() < 0.4):
            left_idx = np.flatnonzero(left_row)
            lo = int(left_idx.min())
            hi = int(left_idx.max())
            if hi - lo + 1 <= w and (rng.random() < 0.3):
                left_row[lo:hi + 1] = left_row[lo:hi + 1] | (rng.random(hi - lo + 1) > 0.45)
        if right_row.sum() >= 3 and (rng.random() < 0.4):
            right_idx = np.flatnonzero(right_row)
            lo = int(right_idx.min())
            hi = int(right_idx.max())
            if hi - lo + 1 <= w and (rng.random() < 0.3):
                right_row[lo:hi + 1] = right_row[lo:hi + 1] | (rng.random(hi - lo + 1) > 0.45)

    return left_row, right_row


def _build_sides(rng: np.random.Generator, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    modes = {
        "blue_preferred": np.array([0.68, 0.18, 0.14], dtype=float),  # blue weaker
        "red_preferred": np.array([0.18, 0.68, 0.14]),               # red weaker
        "tie_preferred": np.array([0.22, 0.20, 0.58]),               # ties more frequent
    }
    mode = rng.choice(list(modes.keys()))
    probs = modes[mode]
    relations = rng.choice(["blue_weak", "red_weak", "tie"], size=h, p=probs)
    if h >= 3:
        present = set(relations.tolist())
        if len(present) == 1:
            i = int(rng.integers(h))
            target = "blue_weak" if relations[0] != "blue_weak" else "red_weak"
            relations[i] = target
        if len(set(relations)) < 3 and rng.random() < 0.35 and h >= 4:
            missing = {"blue_weak", "red_weak", "tie"} - set(relations.tolist())
            if missing:
                i = int(rng.integers(h))
                relations[i] = next(iter(missing))

    left = np.zeros((h, w), dtype=bool)
    right = np.zeros((h, w), dtype=bool)
    for r in range(h):
        left[r], right[r] = _row_masks(rng, w, str(relations[r]))

    return left, right


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    attempts = 300
    for _ in range(attempts):
        h = int(rng.integers(7, 20))
        w = int(rng.integers(5, 13))
        left_mask, right_mask = _build_sides(rng, h, w)
        left_color = Color.BLUE
        right_color = Color.RED

        input_grid = np.full((h, 2 * w + 1), Color.BLACK, dtype=int)
        input_grid[:, :w] = np.where(left_mask, left_color, Color.BLACK)
        input_grid[:, w + 1 :] = np.where(right_mask, right_color, Color.BLACK)
        input_grid[:, w] = Color.GRAY

        out = main(input_grid)
        if out.shape != (h, w):
            continue
        if np.all(out == Color.BLACK):
            continue

        # Ensure solver-relevant variation: both minority/majority behavior and ties appear often.
        left_counts = np.count_nonzero(input_grid[:, :w] == left_color, axis=1)
        right_counts = np.count_nonzero(input_grid[:, w + 1 :] == right_color, axis=1)
        relation_codes = np.array([0 if l < r else 1 if l > r else 2 for l, r in zip(left_counts, right_counts)])
        if len(np.unique(relation_codes)) < 2:
            continue

        return input_grid

    raise RuntimeError("failed to generate valid weak-side puzzle input")
