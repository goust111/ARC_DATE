# Variant of 1818057f
# Dimension: D14 — '+'字图案的数量 (meta)
# Change: 1-9个'+'字图案，数量可变... → '+' 的数量决定变换行为

from __future__ import annotations

import numpy as np
from arc_utils import get_bg

# Design analysis
# ----------------
# spec:
#   preserved_scene:
#     - Input still uses one most-common background color and one foreground color.
#     - Foreground pixels form noisy lines / blobs with embedded thin '+' motifs.
#     - A '+' motif is still defined locally: center + 4 orthogonal neighbors are fg.
#   variant_core:
#     - The local shape detector does not change.
#     - The output color is now chosen globally from the total number of detected '+' motifs.
#     - Odd number of pluses  -> recolor every detected footprint to teal (8).
#     - Even number of pluses -> recolor every detected footprint to magenta (6).
#   concrete_trace:
#     - If the grid contains 3 valid centers, the solver must first count all 3, then recolor
#       the 15 involved cells to 8. This matches the old task only because 3 is odd.
#     - If the same style of scene contains 4 valid centers, the old solver would still paint 8,
#       but the variant must paint all four '+' footprints to 6 instead.
#     - Therefore the changed logic sits between "find all local centers" and "paint outputs":
#       counting is now a mandatory global decision step.
#   complexity_calibration:
#     - Only one extra binary branch is added, so the task is harder than the original but still
#       inferable from a few examples.
#     - We deliberately keep the same crowded red/yellow visual language so the new reasoning load
#       comes from counting, not from introducing new object types.
#   implicit_assumption_review:
#     - `bg = get_bg(g)` is still valid because the generator keeps the background as the majority color.
#     - `fg = only non-bg color` is still valid for the input; the solver reads the input grid, not
#       an already transformed output.
#     - The original loop could recolor immediately because the target color was fixed at 8.
#       That is no longer safe conceptually: the solver must collect all centers first, then choose
#       the shared highlight color from the global count, then paint.
#     - Neighbor checks must read from the original grid `g`, never from `out`, so earlier repainting
#       cannot alter later detections.
#   placement_legality:
#     - Centers are placed with at least 2 cells of margin from the border and pairwise L1 distance >= 4.
#     - Each intended plus receives at least one extra extension cell so the motif is embedded, not isolated.
#     - Extension cells are accepted only if the total number of plus centers stays exactly at the
#       intended count; distractor cells are accepted only if they create no new plus.
#   visual_observability:
#     - The only per-sample branch is teal vs magenta, so the count parity is visually obvious once
#       both branches appear.
#     - To make that happen reliably in a tiny ARC training set, successive generator calls alternate
#       the requested parity instead of leaving branch coverage to chance.
#   cleanliness:
#     - Plus motifs do not overlap, and generator checks prevent accidental extra centers caused by
#       clutter growth, keeping the scenes busy but still readable.

PLUS_DELTAS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
TEAL = 8
MAGENTA = 6
_NEXT_PARITY = 1  # 1 = odd, 0 = even


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    """Return all interior cells that are fg and have fg in the four orthogonal neighbors."""
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in PLUS_DELTAS[1:]):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' motifs in one foreground color and recolor them using the total motif count.

    Input structure:
    - The background is the most frequent color in the grid.
    - A single non-background color draws scattered shapes, and some locations form 5-cell '+' motifs.

    Rule:
    - First find every foreground cell whose 4 orthogonal neighbors are also foreground.
    - If the number of such centers is odd, recolor every matched 5-cell '+' footprint to teal (8).
    - If the number is even, recolor every matched 5-cell '+' footprint to magenta (6).
    - Leave all other pixels unchanged.

    Key insight: The local detector is unchanged, but the repaint color is a global property of the
    entire grid, so the solver must count all qualifying '+' centers before painting any of them.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    if not centers:
        return out

    highlight = TEAL if (len(centers) % 2 == 1) else MAGENTA
    for r, c in centers:
        for dr, dc in PLUS_DELTAS:
            out[r + dr, c + dc] = highlight
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether changing one bg cell to fg would create a plus at that cell or a 4-neighbor."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_DELTAS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _choose_count(rng: np.random.Generator, target_parity: int) -> int:
    options = [k for k in range(1, 10) if k % 2 == target_parity]
    return int(options[int(rng.integers(0, len(options)))])


def _sample_shape_for_count(rng: np.random.Generator, count: int) -> tuple[int, int]:
    """Pick dimensions large enough that a simple spaced layout for `count` centers exists."""
    n_rows = int(np.ceil(np.sqrt(count)))
    n_cols = int(np.ceil(count / n_rows))
    H = int(rng.integers(max(10, 4 * n_rows + 1), 23))
    W = int(rng.integers(max(10, 4 * n_cols + 1), 23))
    if rng.random() < 0.5:
        H, W = W, H
    return H, W


def _try_add_extension_cell(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    intended_centers: set[tuple[int, int]],
) -> bool:
    """Accept an extension only if it preserves the intended plus-center set size."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    if int(grid[r, c]) == fg:
        return False

    old = int(grid[r, c])
    grid[r, c] = fg
    centers = set(_find_plus_centers(grid, fg))
    ok = (len(centers) == len(intended_centers)) and intended_centers.issubset(centers)
    if not ok:
        grid[r, c] = old
    return ok


def _gen_attempt(target_parity: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    bg, fg = 2, 4
    K = _choose_count(rng, target_parity)
    H, W = _sample_shape_for_count(rng, K)

    grid = np.full((H, W), bg, dtype=int)

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
            if len(centers) == K:
                break
    if len(centers) != K:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_DELTAS:
            grid[cr + dr, cc + dc] = fg

    intended_centers = set(centers)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for cr, cc in centers:
        extended = False
        order = list(rng.permutation(4))
        n_ext = int(rng.integers(1, 3))
        added_dirs = 0
        for di in order:
            dr, dc = dirs[int(di)]
            ext_len = int(rng.integers(1, 4))
            dir_added = False
            for step in range(1, ext_len + 1):
                nr = cr + dr * (step + 1)
                nc = cc + dc * (step + 1)
                if not _try_add_extension_cell(grid, nr, nc, fg, intended_centers):
                    break
                dir_added = True
                extended = True
            if dir_added:
                added_dirs += 1
                if added_dirs >= n_ext:
                    break
        if not extended:
            return None

    if len(_find_plus_centers(grid, fg)) != K:
        return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(np.ceil(total * target_density))
    current_fg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, int(r), int(c), fg):
            grid[int(r), int(c)] = fg
            current_fg += 1

    density = current_fg / total
    if density < 0.26 or density > 0.41:
        return None
    if current_fg >= total // 2:
        return None

    actual_centers = _find_plus_centers(grid, fg)
    if len(actual_centers) != K:
        return None

    out = main(grid)
    expected_color = TEAL if (K % 2 == 1) else MAGENTA
    forbidden_color = MAGENTA if expected_color == TEAL else TEAL
    if not np.any(out == expected_color):
        return None
    if np.any(out == forbidden_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_PARITY
    target_parity = _NEXT_PARITY
    _NEXT_PARITY = 1 - _NEXT_PARITY

    for _ in range(400):
        result = _gen_attempt(target_parity)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
