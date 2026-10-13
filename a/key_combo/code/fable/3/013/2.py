from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import random
import sys
from typing import Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Make arc_utils available when loaded directly via SourceFileLoader.
# ---------------------------------------------------------------------------
try:
    from arc_utils import Color
except Exception:  # pragma: no cover - exercised in dynamic import fallback path
    for parent in Path(__file__).resolve().parents:
        _utils_root = parent / "new" / "solvers" / "arc_utils.py"
        if _utils_root.exists():
            _sys_path = str(_utils_root.parent)
            if _sys_path not in sys.path:
                sys.path.insert(0, _sys_path)
            from arc_utils import Color  # type: ignore
            break
    else:

        class Color:
            BLACK = 0
            RED = 2
            GREEN = 3
            BLUE = 1
            YELLOW = 4
            PINK = 6
            NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# ------------------------------
# Core helpers
# ------------------------------

def _segment_cost(lines: np.ndarray, start: int, end: int) -> int:
    """Cost of grouping contiguous lines [start..end] into one block.

    A block is valid if each column is mostly constant; every non-majority pixel
    contributes one rewrite when reconstructing a clean latent row/column stripe.
    """
    segment = np.asarray(lines[start : end + 1], dtype=int)
    h, _w = segment.shape
    if h <= 1:
        return 0

    cost = 0
    for c in range(_w):
        vals, counts = np.unique(segment[:, c], return_counts=True)
        cost += int(h - counts.max())
    return cost


def _infer_axis_segments(lines: np.ndarray, penalty: int) -> list[tuple[int, int]]:
    """Infer contiguous run segments along one axis by 1D DP.

    Returns [(r0, r1), ...] with inclusive boundaries.
    """
    axis = np.asarray(lines, dtype=int)
    n = int(axis.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [(0, 0)]

    cost = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i, n):
            cost[i, j] = _segment_cost(axis, i, j)

    inf = 10**9
    dp = [inf] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = 0

    for end in range(1, n + 1):
        best = inf
        best_start = 0
        for start in range(end):
            # One new segment [start, end-1] adds `penalty` plus disagreement cost.
            value = dp[start] + int(penalty) + int(cost[start, end - 1])
            if value < best:
                best = value
                best_start = start
        dp[end] = best
        parent[end] = best_start

    if parent[n] < 0:
        raise AssertionError("failed to infer axis partitions")

    parts: list[tuple[int, int]] = []
    cursor = n
    while cursor > 0:
        s = parent[cursor]
        parts.append((s, cursor - 1))
        cursor = s
    parts.reverse()
    return parts


def _candidate_penalties(n: int) -> list[int]:
    n = int(n)
    if n <= 1:
        return [1]
    c = {
        1,
        2,
        3,
        max(1, n // 4),
        max(1, n // 3),
        max(1, n // 2 - 1),
    }
    return sorted({p for p in c if p >= 1})


def _unique_axis_partitions(candidates: Iterable[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    seen = set()
    out: list[list[tuple[int, int]]] = []
    for cand in candidates:
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def _majority_with_anomalies(block: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    """Return dominant color, anomaly count, and non-dominant coordinates."""
    vals, counts = np.unique(block, return_counts=True)
    idx = int(np.argmax(counts))
    dom = int(vals[idx])
    dom_count = int(counts[idx])
    anom = int(block.size - dom_count)

    if anom <= 0:
        return dom, 0, []

    rr, cc = np.where(block != dom)
    pos = list(zip(rr.astype(int).tolist(), cc.astype(int).tolist()))
    return dom, anom, pos


def _analyze_candidate(
    grid: np.ndarray,
    row_parts: list[tuple[int, int]],
    col_parts: list[tuple[int, int]],
) -> tuple[float, np.ndarray, list[tuple[int, int, int, int, int]], int]:
    """Score one `(rows, cols)` partition hypothesis.

    Returns: (score, compressed_grid, defects, penalty).
    """
    out_h, out_w = len(row_parts), len(col_parts)
    compressed = np.zeros((out_h, out_w), dtype=int)

    defects: list[tuple[int, int, int, int, int]] = []
    bad = 0.0
    anomaly_total = 0

    for bi, (r0, r1) in enumerate(row_parts):
        for bj, (c0, c1) in enumerate(col_parts):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            dom, anomaly_count, non_dom = _majority_with_anomalies(block)
            compressed[bi, bj] = dom

            anomaly_total += anomaly_count
            if anomaly_count == 0:
                continue

            if anomaly_count != 1:
                bad += 30000.0 + 5000.0 * (anomaly_count - 1)
                continue

            if len(non_dom) != 1:
                bad += 42000.0
                continue

            dr, dc = non_dom[0]
            abs_r = r0 + dr
            abs_c = c0 + dc

            # Invalid if touching a block corner or not on boundary.
            on_top = abs_r == r0
            on_bottom = abs_r == r1
            on_left = abs_c == c0
            on_right = abs_c == c1

            if (on_top and on_left) or (on_top and on_right) or (on_bottom and on_left) or (on_bottom and on_right):
                bad += 42000.0
                continue

            if on_top and not on_left and not on_right:
                if r0 == r1:
                    bad += 42000.0
                    continue
                tr, tc = bi - 1, bj
            elif on_bottom and not on_left and not on_right:
                if r0 == r1:
                    bad += 42000.0
                    continue
                tr, tc = bi + 1, bj
            elif on_left and not on_top and not on_bottom:
                if c0 == c1:
                    bad += 42000.0
                    continue
                tr, tc = bi, bj - 1
            elif on_right and not on_top and not on_bottom:
                if c0 == c1:
                    bad += 42000.0
                    continue
                tr, tc = bi, bj + 1
            else:
                bad += 42000.0
                continue

            if not (0 <= tr < out_h and 0 <= tc < out_w):
                bad += 42000.0
                continue

            defects.append((int(grid[abs_r, abs_c]), bi, bj, tr, tc))

    # Each output cell can be rewritten at most once.
    target_counts = Counter((tr, tc) for _col, _sr, _sc, tr, tc in defects)
    for key, cnt in target_counts.items():
        if cnt > 1:
            bad += 70000.0 + 20000.0 * (cnt - 1)

    # Prefer compact partitions to avoid over-splitting.
    compact_penalty = out_h + out_w
    score = float(bad + anomaly_total + compact_penalty)
    return score, compressed, defects, int(compact_penalty)


def _choose_partitions(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    row_candidates = _unique_axis_partitions(_infer_axis_segments(row, p) for row in () )
    # Build row/column candidate sets.
    rp_candidates: list[list[tuple[int, int]]] = []
    for p in _candidate_penalties(g.shape[0]):
        rp_candidates.append(_infer_axis_segments(g, p))

    cp_candidates: list[list[tuple[int, int]]] = []
    for p in _candidate_penalties(g.shape[1]):
        cp_candidates.append(_infer_axis_segments(g.T, p))

    rp_candidates = _unique_axis_partitions(rp_candidates)
    cp_candidates = _unique_axis_partitions(cp_candidates)

    if not rp_candidates or not cp_candidates:
        raise AssertionError("failed to infer axis partitions")

    best_score = math.inf
    best_comp = None
    best_defects: list[tuple[int, int, int, int, int]] = []

    for rparts in rp_candidates:
        for cparts in cp_candidates:
            score, comp, defects, _pen = _analyze_candidate(g, rparts, cparts)
            if score < best_score:
                best_score = score
                best_comp = comp
                best_defects = defects

    if best_comp is None:
        raise AssertionError("failed to evaluate any axis partition")

    # Keep one canonical solution by rebuilding from the best compact candidate.
    return best_comp, best_defects


# ------------------------------
# Public interface
# ------------------------------

def main(grid: np.ndarray) -> np.ndarray:
    """Compress a stretched block matrix and deliver each defect-pixel color to its neighbor.

    Input structure:
    - A small base color matrix has been expanded by repeating each original row and column by an
      integer factor, producing large single-color rectangular blocks.
    - A few blocks contain exactly one isolated non-block-color pixel on a non-corner boundary pixel.
    - These defect pixels do not survive compression and only define which adjacent block should be recolored.

    Rule:
    - Ignore defect pixels and infer the original rows and columns by grouping expanded lines into
      stripe-like segments with minimal disagreement.
    - Build the compressed grid from the dominant color of each inferred block.
    - For each defect pixel, determine the edge of its source block it touches and recolor that
      neighboring compressed cell with the defect color.

    Key insight:
    - The expanded defects are not noise to keep; they are directional notes. After collapsing each
      repeated row/column stripe, only the source block and the edge it touches remain as the true
      operation signal for one-step recoloring.
    """
    g = np.asarray(grid, dtype=int)
    compressed, defects = _choose_partitions(g)

    out = compressed.copy()
    rewritten: dict[tuple[int, int], int] = {}

    for color, sr, sc, tr, tc in defects:
        key = (tr, tc)
        if key in rewritten:
            if rewritten[key] != color:
                raise AssertionError("conflicting defect rewrite at the same output cell")
            continue
        rewritten[key] = color
        out[tr, tc] = color

    return out


def _prefix_starts(lengths: list[int]) -> list[int]:
    out = [0]
    cur = 0
    for v in lengths[:-1]:
        cur += int(v)
        out.append(cur)
    return out


def _expand_matrix(base: np.ndarray, row_reps: list[int], col_reps: list[int]) -> np.ndarray:
    h, w = map(int, base.shape)
    hs = _prefix_starts(row_reps)
    ws = _prefix_starts(col_reps)

    out = np.zeros((sum(row_reps), sum(col_reps)), dtype=int)
    for i in range(h):
        for j in range(w):
            r0 = hs[i]
            c0 = ws[j]
            out[r0 : r0 + row_reps[i], c0 : c0 + col_reps[j]] = int(base[i, j])
    return out


def _inject_random_defects(
    canvas: np.ndarray,
    base_h: int,
    base_w: int,
    row_reps: list[int],
    col_reps: list[int],
    branch: str,
) -> np.ndarray:
    if branch == "no_defects":
        return canvas

    canvas = canvas.copy()
    r_starts = _prefix_starts(row_reps) + [sum(row_reps)]
    c_starts = _prefix_starts(col_reps) + [sum(col_reps)]

    block_used_targets: set[tuple[int, int]] = set()
    # Precompute which blocks can carry a valid non-corner boundary defect.
    block_options: list[tuple[int, int, list[tuple[int, int, tuple[int, int]]]]] = []
    for br in range(base_h):
        for bc in range(base_w):
            r0, r1 = r_starts[br], r_starts[br + 1] - 1
            c0, c1 = c_starts[bc], c_starts[bc + 1] - 1
            h = r1 - r0 + 1
            w = c1 - c0 + 1

            options: list[tuple[int, int, tuple[int, int]]] = []
            # up / down need width > 1 to avoid corner-only placement.
            if br > 0 and w >= 2:
                options.append((-1, 0, (1, 1)))
            if br < base_h - 1 and w >= 2:
                options.append((1, 0, (1, 1)))
            # left / right need height > 1 to avoid corner-only placement.
            if bc > 0 and h >= 2:
                options.append((0, -1, (1, 1)))
            if bc < base_w - 1 and h >= 2:
                options.append((0, 1, (1, 1)))
            if not options:
                continue
            block_options.append((br, bc, options))

    if not block_options:
        return canvas

    random.shuffle(block_options)

    # Branches:
    #  - mixed: random number of defects (sparse/denser)
    #  - horizontal / vertical: force mostly that orientation.
    branch_weights = {
        "horizontal": ("left", "right"),
        "vertical": ("up", "down"),
        "mixed": (),
    }

    max_defects = max(0, len(block_options))
    if branch == "mixed":
        # Favors sparse and medium densities.
        target_defects = int(max(1, np.random.randint(1, min(4, max_defects + 1))))
        if np.random.random() < 0.5:
            target_defects = min(max_defects, target_defects + 1)
    elif branch in branch_weights:
        target_defects = max(1, np.random.randint(1, min(3, max_defects + 1)))
    else:
        target_defects = 1

    actual = 0
    for br, bc, options in block_options:
        if actual >= target_defects:
            break

        r0, r1 = r_starts[br], r_starts[br + 1] - 1
        c0, c1 = c_starts[bc], c_starts[bc + 1] - 1
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        # Build weighted direction candidates by branch.
        direction_map = {
            (-1, 0): "up",
            (1, 0): "down",
            (0, -1): "left",
            (0, 1): "right",
        }
        preferred = set(branch_weights.get(branch, ()))

        opts = options[:]
        random.shuffle(opts)
        if preferred:
            # bias toward requested orientation, with fallback to any valid if empty.
            preferred_opts = [opt for opt in opts if direction_map[(opt[0], opt[1])] in preferred]
            if preferred_opts:
                opts = preferred_opts + [opt for opt in opts if opt not in preferred_opts]

        placed = False
        for dr, dc, _span in opts:
            # Choose a boundary pixel that is not a corner.
            if dr == 0:  # left/right, choose row interior
                if h < 2:
                    continue
                rr = r0 + random.randint(1, h - 1) if h > 1 else r0
                cc = c0 if dc == -1 else c1
            else:  # up/down, choose col interior
                if w < 2:
                    continue
                cc = c0 + random.randint(1, w - 1)
                rr = r0 if dr == -1 else r1

            # Ensure unique output destination in compressed space.
            tr = br + dr
            tc = bc + dc
            if not (0 <= tr < base_h and 0 <= tc < base_w):
                continue
            if (tr, tc) in block_used_targets:
                continue

            palette = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            palette = [c for c in palette if c != int(canvas[rr, cc])]
            if not palette:
                break
            defect_color = int(random.choice(palette))

            canvas[rr, cc] = defect_color
            block_used_targets.add((tr, tc))
            actual += 1
            placed = True
            break

        if not placed:
            continue

    return canvas


def generate_input() -> np.ndarray:
    """Generate random inputs that satisfy the stretch + non-corner boundary defect rule."""
    for _attempt in range(300):
        base_h = int(np.random.randint(2, 7))
        base_w = int(np.random.randint(2, 7))

        palette = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], k=random.randint(3, min(6, base_h * base_w)))
        base = np.random.choice(palette, size=(base_h, base_w), replace=True)

        # Repetition factors are allowed to be 1, but ensure at least one dimension usually stretches.
        row_reps = [int(np.random.randint(1, 5)) for _ in range(base_h)]
        col_reps = [int(np.random.randint(1, 5)) for _ in range(base_w)]

        # Keep sizes safely within ARC bounds.
        if sum(row_reps) > 28 or sum(col_reps) > 28:
            continue
        if all(v == 1 for v in row_reps):
            row_reps[random.randrange(base_h)] = int(np.random.randint(2, 5))
        if sum(row_reps) > 28:
            continue
        if all(v == 1 for v in col_reps):
            col_reps[random.randrange(base_w)] = int(np.random.randint(2, 5))
        if sum(col_reps) > 28:
            continue

        expanded = _expand_matrix(base, row_reps, col_reps)
        if np.random.random() < 0.18:
            branch = "no_defects"
        else:
            branch = random.choice(["horizontal", "vertical", "mixed"])

        with_defects = _inject_random_defects(expanded, base_h, base_w, row_reps, col_reps, branch)

        # Sanity-check self-consistency: our generator can call the solver itself.
        try:
            _ = main(with_defects)
        except Exception:
            continue

        if with_defects.ndim == 2 and with_defects.shape[0] <= 30 and with_defects.shape[1] <= 30:
            return with_defects

    raise RuntimeError("failed to generate a valid stretch+defect input")

