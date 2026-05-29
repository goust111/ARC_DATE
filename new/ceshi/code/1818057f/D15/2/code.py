# Variant of 1818057f
# Dimension: D15 — 图例/符号指令 (symbol_instructions)
# Change: 无图例或符号指令，变换规则完全隐含在输入输出的差异中
#   → 用单像素标记指定哪些'+'需要变换

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# a) Concrete behavior:
#    The original task recolored every local '+' motif. In this variant the plus test
#    stays the same, but activation is explicit: a sparse control color appears as a
#    single pixel in one diagonal corner of a selected plus center. Example: if a fg
#    plus is centered at (r,c), a blue pixel at (r-1,c+1) means "this plus only".
#    The input therefore has three colors: bg, many fg shape pixels, and a few control
#    pixels. The output preserves the control pixels and recolors only the marked plus
#    footprints to teal.
# b) Complexity calibration:
#    Markers are single-pixel and always adjacent in the same way (diagonal to the
#    center), so the symbolic instruction is visible but not noisy. Each grid contains
#    both marked and unmarked pluses, preventing collapse back to the original "all
#    pluses change" rule.
# c) Hidden assumptions from the old solver:
#    The original `main()` assumed exactly one non-bg color and recolored every detected
#    plus unconditionally. That breaks here because the smallest non-bg color could be
#    the control color, and unguarded recoloring would overwrite unmarked pluses. The
#    new solver instead infers fg as the dominant non-bg color and requires a diagonal
#    control neighbor before recoloring.
# d) Placement legality:
#    Generator centers are spaced so a control pixel cannot be diagonal to two different
#    plus centers. Markers are placed only on in-bounds diagonal cells. Distractor fg
#    pixels are added only when they do not create any new plus center.
# e) Visual observability:
#    Across 2-4 samples a human can compare marked and unmarked embedded pluses of the
#    same yellow shape color and see that only those with a nearby blue dot change.
# f) Visual cleanliness:
#    Only one control pixel is used per marked plus, kept off the plus footprint itself,
#    which avoids overlap between the symbol and the object it controls.
# g) Color inferability:
#    Colors stay simple: bg red, fg yellow, control blue, output teal. The extra color
#    has a clear structural role rather than being used in any hidden ordering scheme.
# h) New solver assumptions:
#    The solver assumes the control color is much rarer than the shape color and that
#    marking uses diagonal adjacency to the plus center. The generator guarantees both.


PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTH_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    if r < 1 or r >= H - 1 or c < 1 or c >= W - 1:
        return False
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTH_OFFSETS)


def _plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the '+' motifs that are explicitly tagged by single-pixel markers.

    Input structure:
    - bg is the most common color.
    - A dominant foreground color forms mixed shapes containing embedded '+' motifs.
    - A sparse second non-bg color appears as isolated control pixels placed diagonally beside selected '+' centers.

    Rule:
    - A '+' center is a foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Recolor that 5-pixel '+' to teal (8) only when at least one diagonal neighbor of the center is the control color.
    - Unmarked '+' motifs and all control pixels stay unchanged.

    Key insight:
    - Plus detection is still local, but the transform is no longer global: the nearby control pixel acts like an instruction symbol that selects which otherwise-identical '+' motifs should react.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if len(colors) < 2:
        return out

    counts = {color: int(np.sum(g == int(color))) for color in colors}
    fg = max(colors, key=lambda color: (counts[color], -int(color)))
    ctrl = min(colors, key=lambda color: (counts[color], int(color)))

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if not _is_plus_center(g, r, c, fg):
                continue
            if not any(int(g[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG_OFFSETS):
                continue
            for dr, dc in PLUS_OFFSETS:
                out[r + dr, c + dc] = 8
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check if setting (r,c) to fg would create any new plus center nearby."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in PLUS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != int(fg):
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _expected_output(inp: np.ndarray, fg: int, ctrl: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for r, c in _plus_centers(inp, fg):
        if any(int(inp[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG_OFFSETS):
            for rr, cc in ((r + dr, c + dc) for dr, dc in PLUS_OFFSETS):
                out[rr, cc] = 8
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg, ctrl = 2, 4, 1

    grid = np.full((H, W), bg, dtype=int)

    candidate_centers = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidate_centers)

    desired = int(rng.integers(2, 9))
    centers: list[tuple[int, int]] = []
    for cr, cc in candidate_centers:
        if len(centers) >= desired:
            break
        if all(
            abs(cr - er) + abs(cc - ec) >= 4 and max(abs(cr - er), abs(cc - ec)) >= 3
            for er, ec in centers
        ):
            centers.append((cr, cc))

    if len(centers) < 2:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_OFFSETS:
            grid[cr + dr, cc + dc] = fg

    for cr, cc in centers:
        n_ext = int(rng.integers(1, 3))
        chosen = rng.choice(len(ORTH_OFFSETS), size=n_ext, replace=False)
        for idx in chosen:
            dr, dc = ORTH_OFFSETS[int(idx)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, 4))
            for step in range(1, ext_len + 1):
                nr = tip_r + dr * step
                nc = tip_c + dc * step
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    break
                if int(grid[nr, nc]) != int(bg):
                    break
                grid[nr, nc] = fg

    marked_count = int(rng.integers(1, len(centers)))
    marked_indices = set(map(int, rng.choice(len(centers), size=marked_count, replace=False)))

    for idx, (cr, cc) in enumerate(centers):
        if idx not in marked_indices:
            continue
        marker_options = [(cr + dr, cc + dc) for dr, dc in DIAG_OFFSETS if int(grid[cr + dr, cc + dc]) == int(bg)]
        if not marker_options:
            return None
        mr, mc = marker_options[int(rng.integers(len(marker_options)))]
        grid[mr, mc] = ctrl

    total = H * W
    target_density = float(rng.uniform(0.26, 0.39))
    target_nonbg = int(total * target_density)
    current_nonbg = int(np.sum(grid != bg))

    bg_positions = list(zip(*np.where(grid == bg)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_nonbg >= target_nonbg:
            break
        if not _would_create_plus(grid, int(r), int(c), fg):
            grid[int(r), int(c)] = fg
            current_nonbg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    centers_now = _plus_centers(grid, fg)
    marked_now = [
        (r, c)
        for r, c in centers_now
        if any(int(grid[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG_OFFSETS)
    ]
    if not marked_now or len(marked_now) >= len(centers_now):
        return None

    inp = grid.copy()
    expected = _expected_output(inp, fg, ctrl)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if not np.any(actual == 8):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
