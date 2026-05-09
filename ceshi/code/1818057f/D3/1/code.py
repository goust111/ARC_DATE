# Variant of 1818057f
# Dimension: D3 — 识别目标：'+'字图案的定义 (src)
# Change: 黄色'+'字精灵：任何前景像素(r,c)若其4个正交邻居均为前景色，则(r,c)及其4邻居构成一个'+'... → L型图案识别

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the D3 variant
# a) Concrete behavior:
#    - The old task marked 5-cell pluses. The new task marks 3-cell elbows:
#      a foreground center plus exactly two orthogonal neighbors that meet at a right angle.
#    - Example trace on a local 3x3 patch:
#        old:  .Y.
#              YYY   -> center qualifies because all 4 orthogonal neighbors are yellow
#              .Y.
#        new:  ...
#              .YY   -> right/down corner qualifies only if the center has exactly
#              .Y.      those two neighbors and the other two orthogonal sides are bg.
#      The processing stage affected is the local motif test inside main(); generation
#      must therefore switch from planting cross centers to planting elbow centers.
# b) Complexity calibration:
#    - Using "exactly two orthogonal neighbors" keeps the rule visible and avoids the
#      ambiguity of coloring T-junctions or pluses just because they contain an L subset.
#    - Four rotations are sampled repeatedly so examples show the full rule family.
# c) Implicit assumptions in the old main():
#    - Old code assumed a valid target has four fg neighbors. That is no longer true.
#    - Old unconditional recoloring of four arms would overpaint unrelated pixels here;
#      we must recolor only the two participating neighbors discovered at each center.
#    - Old larger blobs were harmless because the predicate was monotone in fg-neighbors.
#      Here extra neighbors invalidate the motif, so generation must protect each elbow
#      from accidental side attachments.
# d) Placement legality:
#    - Centers stay away from the border so their orthogonal neighborhood is fully defined.
#    - Each planted elbow occupies distinct cells when created, and every tentative
#      addition is validated by rescanning the grid: intended elbows must still be the
#      complete detected elbow set.
# e) Visual observability:
#    - Output recolors only bent corners, while straight runs remain yellow. With several
#      orientations per sample, a human can infer "corners only" from 2-4 examples.
# f) Visual neatness:
#    - Distractors are mostly straight segments or isolated pixels, which raise density
#      without layering many competing local motifs in the same area.
# g) Color inference:
#    - Colors stay in the original scheme: red background, yellow foreground, teal output.
#      No hidden dependence on color ids or color ordering is introduced.
# h) Non-degeneration check target:
#    - Samples should contain multiple elbow centers and usually zero plus centers, so the
#      new behavior is measurably different from the original cross-highlighting task.
# i) Generator assumptions required by main():
#    - main() assumes a target center is interior and uniquely characterized by exactly two
#      orthogonal fg neighbors; generate_input() enforces this by rejecting any placement
#      that creates extra elbow centers or changes an intended elbow's local degree.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ELBOW_DIRS = [
    ((-1, 0), (0, -1)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((1, 0), (0, 1)),
]
ELBOW_DIR_SET = {tuple(sorted(dirs)) for dirs in ELBOW_DIRS}


def _norm_dirs(dirs):
    return tuple(sorted((int(dr), int(dc)) for dr, dc in dirs))


def _find_elbows(grid: np.ndarray, fg: int) -> dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int]]]:
    hits = {}
    H, W = grid.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            nbrs = [(dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == fg]
            if len(nbrs) != 2:
                continue
            key = _norm_dirs(nbrs)
            if key in ELBOW_DIR_SET:
                hits[(r, c)] = key
    return hits


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    out = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS):
                out.append((r, c))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin L-turns in a single foreground color and recolor each elbow triplet to teal.

    Input structure:
    - bg is the most common color.
    - One foreground color (e.g. yellow) forms sparse lines, bends, and small blobs on the bg; some local 3-cell corners are L motifs.

    Rule:
    - An L-center is any foreground pixel with exactly two foreground orthogonal neighbors, with one vertical and one horizontal.
    - Recolor that center and those two participating neighbors to teal (8).
    - Straight runs, isolated pixels, and higher-degree junctions remain unchanged.

    Key insight: The decision is purely local degree-and-orientation matching. A bend counts only when the center has exactly the two corner-forming neighbors, so extra side attachments disqualify what would otherwise look like an L subset.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            nbrs = [(dr, dc) for dr, dc in ORTHO_DIRS if int(g[r + dr, c + dc]) == fg]
            if len(nbrs) != 2:
                continue
            if _norm_dirs(nbrs) not in ELBOW_DIR_SET:
                continue
            out[r, c] = 8
            for dr, dc in nbrs:
                out[r + dr, c + dc] = 8
    return out


def _orientation_schedule(rng: np.random.Generator, count: int):
    base = list(ELBOW_DIRS)
    rng.shuffle(base)
    out = []
    while len(out) < count:
        rng.shuffle(base)
        take = min(len(base), count - len(out))
        out.extend(base[:take])
    return out


def _component_cells(center: tuple[int, int], dirs, extra_lengths):
    r, c = center
    cells = {(r, c)}
    for (dr, dc), extra in zip(dirs, extra_lengths):
        for step in range(1, 2 + int(extra)):
            cells.add((r + dr * step, c + dc * step))
    return cells


def _try_paint(grid: np.ndarray, cells, fg: int, intended):
    H, W = grid.shape
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return None
        if int(grid[r, c]) == fg:
            return None

    trial = grid.copy()
    for r, c in cells:
        trial[r, c] = fg

    hits = _find_elbows(trial, fg)
    if hits != intended:
        return None
    if _find_plus_centers(trial, fg):
        return None
    return trial


def _gen_attempt():
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    max_k = min(9, max(3, (H * W) // 40 + 2))
    K = int(rng.integers(2, max_k + 1))
    intended = {}

    for dirs in _orientation_schedule(rng, K):
        placed = False
        wanted = _norm_dirs(dirs)
        for _ in range(180):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            extra = [int(rng.integers(0, 4)), int(rng.integers(0, 4))]
            if extra[0] == 0 and extra[1] == 0:
                extra[int(rng.integers(0, 2))] = int(rng.integers(1, 4))
            cells = _component_cells(center, dirs, extra)
            intended_next = dict(intended)
            intended_next[center] = wanted
            trial = _try_paint(grid, cells, fg, intended_next)
            if trial is None or _find_elbows(trial, fg).get(center) != wanted:
                continue
            grid = trial
            intended = intended_next
            placed = True
            break
        if not placed:
            return None

    total = H * W
    target_density = float(rng.uniform(0.26, 0.38))
    target_fg = int(total * target_density)

    attempts = 0
    while int(np.sum(grid != bg)) < target_fg and attempts < 2200:
        attempts += 1
        length = int(rng.integers(1, 6))
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            r = int(rng.integers(0, H))
            c0 = int(rng.integers(0, W - length + 1))
            cells = {(r, c0 + dc) for dc in range(length)}
        else:
            r0 = int(rng.integers(0, H - length + 1))
            c = int(rng.integers(0, W))
            cells = {(r0 + dr, c) for dr in range(length)}

        trial = _try_paint(grid, cells, fg, intended)
        if trial is not None:
            grid = trial

    if int(np.sum(grid != bg)) < max(target_fg - 2, len(intended) * 4):
        return None
    if int(np.sum(grid != bg)) >= total // 2:
        return None

    hits = _find_elbows(grid, fg)
    if hits != intended:
        return None

    inp = grid.copy()
    out = main(inp)
    expected = inp.copy()
    for (r, c), dirs in hits.items():
        expected[r, c] = 8
        for dr, dc in dirs:
            expected[r + dr, c + dc] = 8
    if not np.array_equal(out, expected):
        return None
    if len(hits) < 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
