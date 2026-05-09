# Variant of 1818057f
# Dimension: D2 — 变换的目标颜色 (transform)
# Change: 所有'+'字图案统一着色为青色(8/teal)，单一固定目标颜色... → 颜色交换：前景色与新颜色互换

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# ===============
# concretization:
#   input keeps the original visual language: red background, one true foreground color
#   (yellow here) forming lines/blobs with embedded thin '+' motifs. The only new visual
#   element is a small set of pre-existing teal pixels placed as distractors on background
#   cells. In the output, every detected '+' footprint becomes teal, while those input-teal
#   distractors become yellow. So the same two colors exchange roles in opposite directions.
#   Small trace:
#     - original logic on an input with one teal singleton and one yellow plus:
#       plus -> teal, singleton teal would stay teal forever.
#     - variant logic:
#       detect the plus using only yellow in the input;
#       swap all original teal pixels to yellow;
#       repaint the plus footprint to teal.
#     - difference:
#       teal is now an observable input-side signal, but it never participates in '+' detection.
# complexity_calibration:
#   keep only two active non-background colors: one real foreground color and teal. This is
#   enough to make the bidirectional rule visible without introducing extra categories. The
#   generator injects 2..5 teal distractors so the swap is usually repeated several times in
#   each sample instead of appearing as a one-off coincidence.
# implicit_assumption_audit:
#   original `fg = colors[0]` relied on there being exactly one non-bg color; that breaks once
#   teal may already be present. The new solver therefore chooses the non-bg, non-teal color as
#   the structural foreground.
#   original `out[...] = 8` was safe because no input cell started as teal. Under color-swap,
#   a naive later pass such as `out[out == 8] = fg` would erase freshly painted pluses. The
#   safe ordering is: convert only the original teal cells first, then paint plus footprints.
#   '+' detection must read from the untouched input grid, not from `out`, otherwise swapped
#   teal cells could accidentally join the foreground and create bogus motifs.
# placement_legality:
#   plus centers still keep a 1-cell margin from borders and L1 distance >= 4 from one another,
#   so their 5-cell footprints do not overlap. Teal distractors are only placed on background
#   cells and spaced apart, which prevents overlap with the structural foreground.
# observability_and_cleanliness:
#   isolated teal distractors are visually legible and do not clutter the yellow components.
#   Across 2-3 samples a human can see that old teal marks disappear into yellow exactly while
#   newly highlighted '+' motifs appear in teal.
# color_inference:
#   the mapping is fully structural, not color-id driven: cells already teal in the input swap
#   back to the unique non-teal foreground color, while '+' footprints swap the other way.
# main_assumption_audit:
#   `main()` assumes exactly one structural foreground color distinct from bg and teal; the
#   generator enforces that by using only bg=2, fg=4, and optional teal=8. It also assumes teal
#   distractors are not required for '+' detection; the generator guarantees this by placing them
#   only on background cells.

TEAL = 8
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS = [(0, 0), *ORTHO]


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find '+' motifs in the main foreground color, while swapping any pre-existing teal back to that foreground.

    Input structure:
    - bg is the most common color.
    - One non-bg, non-teal color forms the real pattern; a few teal pixels may already exist as decoys.

    Rule:
    - A '+' center is any pixel of the real foreground color whose 4 orthogonal neighbors are also that same foreground color in the input.
    - In the output, swap the two active colors directionally: every input teal pixel becomes the foreground color, and every detected 5-cell '+' footprint becomes teal.

    Key insight: Teal is not a second shape language to analyze. It is only a swap source, so '+' detection still runs on the untouched foreground layout before the output colors are exchanged.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out

    fg_candidates = [color for color in colors if color != TEAL]
    if not fg_candidates:
        return out
    fg = int(fg_candidates[0])

    out[g == TEAL] = fg
    for r, c in _find_plus_centers(g, fg):
        for dr, dc in PLUS:
            out[r + dr, c + dc] = TEAL
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check if setting (r,c) to fg would create any new plus center at (r,c) or its neighbors."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= H or nc < 0 or nc >= W:
                all_fg = False
                break
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _teal_candidate_score(grid: np.ndarray, r: int, c: int, bg: int) -> tuple[int, int]:
    """Prefer isolated background pockets so teal distractors stay visually clean."""
    H, W = grid.shape
    non_bg_neighbors = 0
    bg_neighbors = 0
    for dr, dc in ORTHO:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            if int(grid[nr, nc]) == bg:
                bg_neighbors += 1
            else:
                non_bg_neighbors += 1
    return non_bg_neighbors, -bg_neighbors


def _place_teal_distractors(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    bg: int,
    teal_count: int,
    rng: np.random.Generator,
) -> bool:
    H, W = grid.shape
    protected = set()
    for cr, cc in centers:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    protected.add((nr, nc))

    candidates = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != bg or (r, c) in protected:
                continue
            candidates.append((r, c))

    rng.shuffle(candidates)
    candidates.sort(key=lambda pos: _teal_candidate_score(grid, pos[0], pos[1], bg))

    placed: list[tuple[int, int]] = []
    for r, c in candidates:
        if any(abs(r - pr) + abs(c - pc) <= 1 for pr, pc in placed):
            continue
        non_bg_neighbors, _ = _teal_candidate_score(grid, r, c, bg)
        if placed and non_bg_neighbors > 1:
            continue
        if not placed and non_bg_neighbors > 2:
            continue
        grid[r, c] = TEAL
        placed.append((r, c))
        if len(placed) >= teal_count:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))  # 10..22
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    K = int(rng.integers(1, 10))  # 1..9
    centers: list[tuple[int, int]] = []
    for _ in range(K * 30):
        if len(centers) >= K:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
    if not centers:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS:
            grid[cr + dr, cc + dc] = fg

    dirs = ORTHO
    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))  # 1 or 2 directions
        chosen = rng.choice(len(dirs), size=n_ext, replace=False)
        for di in chosen:
            dr, dc = dirs[int(di)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))  # 1..3 cells
            for step in range(1, ext_len + 1):
                nr, nc = tip_r + dr * step, tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                grid[nr, nc] = fg

    total = H * W
    teal_count = int(rng.integers(2, 6))  # 2..5 teal distractors to make the swap visible
    max_fg_allowed = (total - teal_count - 1) // 2
    if max_fg_allowed <= 0:
        return None

    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = min(int(total * target_density), max_fg_allowed)
    current_fg = int(np.sum(grid == fg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, int(r), int(c), fg):
            grid[int(r), int(c)] = fg
            current_fg += 1

    if current_fg < 5 or current_fg > max_fg_allowed:
        return None

    if not _place_teal_distractors(grid, centers, bg, teal_count, rng):
        return None

    bg_count = int(np.sum(grid == bg))
    fg_count = int(np.sum(grid == fg))
    teal_pixels = int(np.sum(grid == TEAL))
    if not (bg_count > fg_count and bg_count > teal_pixels):
        return None

    inp = grid.copy()
    out = main(inp)
    plus_centers = _find_plus_centers(inp, fg)
    if not (1 <= len(plus_centers) <= 9):
        return None
    if teal_pixels < 2:
        return None
    if not np.any((inp == TEAL) & (out == fg)):
        return None
    if not np.any(out == TEAL):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
