# Combo variant of 1818057f (index 16)
# combo_spec: D1/4+D15/2
# Name: 标记选择性膨胀

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for combo D1/4 + D15/2
# ---------------------------------------
# a) Concrete behavior and trace:
#    - Original task: every local '+' (center + 4 orthogonal fg neighbors) turns only its
#      own 5 cells teal.
#    - This combo keeps the same '+' detector, but adds a control color and enlarges the
#      output footprint. A single blue control pixel in one diagonal corner of a plus means
#      "this plus only". If marked, the whole 3x3 neighborhood around the center becomes teal.
#      If unmarked, the yellow plus stays unchanged.
#    - Minimal example around one center:
#          input:
#            2 1 2        2 2 2
#            4 4 4   or   4 4 4
#            2 4 2        2 4 2
#          output:
#            8 8 8        2 2 2
#            8 8 8   or   4 4 4
#            8 8 8        2 4 2
#      The left case shows the marker being consumed by the 3x3 fill; the right case shows
#      an unmarked plus of the same geometry staying untouched.
# b) Complexity calibration:
#    - Every sample contains both marked and unmarked pluses, preventing collapse to either
#      "expand all pluses" or "do nothing".
#    - Marking uses exactly one adjacent diagonal pixel, so the symbolic rule is repeated and
#      easy to compare across samples without introducing many marker types or orientations
#      that would be hard to infer from 2-4 examples.
#    - Each true plus keeps a mostly empty 3x3 neighborhood in the input; otherwise the visual
#      difference between 5-cell recolor, 3x3 dilation, and no-op would blur together.
# c) Hidden assumptions review for main():
#    - `bg = get_bg(g)`: still valid only if background is most common; generator keeps total
#      non-bg cells below half.
#    - The old `colors[0]` foreground assumption breaks because the input now has fg and ctrl.
#      New solver infers fg as the dominant non-bg color and ctrl as the rarer one.
#    - Iterating only interior cells still works: plus detection needs 1-cell margin, and the
#      3x3 fill also writes only within that same margin.
#    - The transform cannot be unconditional anymore. A plus must also have at least one
#      diagonal control pixel; otherwise unmarked pluses would be changed incorrectly.
#    - Because the fill footprint is 3x3, generator must keep intended centers well separated
#      so teal blocks never overlap and one marker cannot appear to belong to two pluses.
# d) Placement legality:
#    - Centers are placed at least 2 cells from borders and with Chebyshev distance >= 4.
#      That keeps each 3x3 neighborhood in-bounds and disjoint.
#    - A reserved 3x3 zone is maintained around every intended center. Distractors cannot enter
#      it, so diagonal marker slots remain readable and no accidental near-overlaps occur.
#    - Extra fg growth for each plus starts outside the reserved zone, ensuring every plus is
#      embedded in a larger component while preserving the clean local cross cue.
# e) Visual observability:
#    - Humans can compare two otherwise similar yellow pluses: one has a nearby blue dot and
#      becomes a teal square, another lacks the dot and remains yellow. The control semantics
#      and the dilation semantics are both visible in a single sample.
# f) Visual cleanliness:
#    - Only three input colors are used: red bg, yellow structure, blue marker. The marker is
#      a lone diagonal pixel rather than a thick legend region, so the grid stays uncluttered.
#    - Reserved zones prevent distractor pixels from touching the corners of true pluses, which
#      would make it ambiguous whether a diagonal cell is part of the object or a control mark.
# g) Change inferability:
#    - The only color that disappears in output is the blue control pixel, and it disappears
#      exactly when the adjacent plus expands into a teal 3x3 block. This provides a structural
#      cue for both selection and output color without relying on numeric color ids.
# h) New solver assumptions guaranteed by generator:
#    - Exactly two non-bg colors appear in the input.
#    - fg is much more frequent than ctrl.
#    - Every marked plus has exactly one diagonal control pixel.
#    - There is at least one marked plus and at least one unmarked plus in every grid.


CROSS5 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTH4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    if r < 1 or r >= H - 1 or c < 1 or c >= W - 1:
        return False
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTH4)


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(grid, r, c, fg):
                centers.append((int(r), int(c)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect marked '+' motifs and expand only those into teal 3x3 blocks.

    Input structure:
    - bg is the most common color.
    - One dominant foreground color forms noisy shapes that include embedded thin '+' motifs.
    - A sparse second non-bg color appears as isolated diagonal control pixels beside only some plus centers.

    Rule:
    - A '+' center is a foreground pixel whose 4 orthogonal neighbors are also foreground.
    - If at least one diagonal neighbor of that center is the control color, recolor the full 3x3 neighborhood around the center to teal (8).
    - Unmarked '+' motifs, other foreground pixels, and unrelated background cells remain unchanged.

    Key insight:
    - The local cross still defines the candidate object, but the nearby control pixel decides whether that object reacts, and the reaction is a dilation from a 5-cell cross into a solid 3x3 square.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if len(colors) < 2:
        return out

    counts = {color: int(np.sum(g == int(color))) for color in colors}
    fg = int(max(colors, key=lambda color: (counts[color], -int(color))))
    ctrl = int(min((color for color in colors if int(color) != fg), key=lambda color: (counts[color], int(color))))

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if not _is_plus_center(g, r, c, fg):
                continue
            if not any(int(g[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG4):
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    out[r + dr, c + dc] = 8
    return out


def _marked_centers(grid: np.ndarray, fg: int, ctrl: int) -> list[tuple[int, int]]:
    return [
        (int(r), int(c))
        for r, c in _find_plus_centers(grid, fg)
        if any(int(grid[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG4)
    ]


def _can_paint_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    reserved: np.ndarray,
    allowed_centers: set[tuple[int, int]],
    bg: int,
) -> bool:
    H, W = grid.shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if bool(reserved[int(r), int(c)]):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False

    candidate = grid.copy()
    for r, c in cells:
        candidate[int(r), int(c)] = int(fg)
    return set(_find_plus_centers(candidate, fg)) == allowed_centers


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _expected_output(inp: np.ndarray, fg: int, ctrl: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for r, c in _find_plus_centers(inp, fg):
        if not any(int(inp[r + dr, c + dc]) == int(ctrl) for dr, dc in DIAG4):
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out[r + dr, c + dc] = 8
    return out


def _expected_all_dilate(inp: np.ndarray, fg: int) -> np.ndarray:
    out = np.array(inp, copy=True)
    for r, c in _find_plus_centers(inp, fg):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out[r + dr, c + dc] = 8
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg, ctrl = 2, 4, 1

    grid = np.full((H, W), bg, dtype=int)

    target_k = int(rng.integers(2, 7))
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(max(abs(int(cr) - er), abs(int(cc) - ec)) >= 4 for er, ec in centers):
            centers.append((int(cr), int(cc)))
            if len(centers) >= target_k:
                break
    if len(centers) < 2:
        return None

    reserved = np.zeros((H, W), dtype=bool)
    for cr, cc in centers:
        reserved[cr - 1 : cr + 2, cc - 1 : cc + 2] = True

    for cr, cc in centers:
        for dr, dc in CROSS5:
            grid[cr + dr, cc + dc] = fg

    intended_centers = set((int(r), int(c)) for r, c in centers)
    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    for cr, cc in centers:
        corridors: list[list[tuple[int, int]]] = []
        for dr, dc in ORTH4:
            cells: list[tuple[int, int]] = []
            step = 2
            while True:
                nr = int(cr + dr * step)
                nc = int(cc + dc * step)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if bool(reserved[nr, nc]):
                    break
                cells.append((nr, nc))
                step += 1
            if cells:
                corridors.append(cells)
        if not corridors:
            return None

        n_ext = int(rng.integers(1, min(2, len(corridors)) + 1))
        chosen_ids = rng.choice(len(corridors), size=n_ext, replace=False)
        extended = False
        for idx in np.atleast_1d(chosen_ids):
            corridor = corridors[int(idx)]
            max_len = min(3, len(corridor))
            ext_len = int(rng.integers(1, max_len + 1))
            cells = corridor[:ext_len]
            if not _can_paint_cells(grid, cells, fg, reserved, intended_centers, bg):
                return None
            _paint_cells(grid, cells, fg)
            extended = True
        if not extended:
            return None

    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    marked_count = int(rng.integers(1, len(centers)))
    marked_indices = set(map(int, rng.choice(len(centers), size=marked_count, replace=False)))
    intended_marked = {centers[idx] for idx in marked_indices}
    if not intended_marked or len(intended_marked) >= len(intended_centers):
        return None

    for idx, (cr, cc) in enumerate(centers):
        if idx not in marked_indices:
            continue
        marker_options = [(cr + dr, cc + dc) for dr, dc in DIAG4 if int(grid[cr + dr, cc + dc]) == int(bg)]
        if len(marker_options) != 4:
            return None
        mr, mc = marker_options[int(rng.integers(len(marker_options)))]
        grid[mr, mc] = ctrl

    shapes = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ]

    total = H * W
    target_density = float(rng.uniform(0.26, 0.39))
    target_nonbg = int(total * target_density)
    current_nonbg = int(np.sum(grid != bg))

    for _ in range(total * 30):
        if current_nonbg >= target_nonbg:
            break
        remaining = target_nonbg - current_nonbg
        usable_shapes = [shape for shape in shapes if len(shape) <= max(1, remaining)]
        shape = usable_shapes[int(rng.integers(len(usable_shapes)))]
        anchor_r = int(rng.integers(0, H))
        anchor_c = int(rng.integers(0, W))
        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if not _can_paint_cells(grid, cells, fg, reserved, intended_centers, bg):
            continue
        _paint_cells(grid, cells, fg)
        current_nonbg += len(cells)

    current_nonbg = int(np.sum(grid != bg))
    if current_nonbg < int(total * 0.24):
        return None
    if current_nonbg >= total // 2:
        return None

    centers_now = set(_find_plus_centers(grid, fg))
    if centers_now != intended_centers:
        return None

    marked_now = set(_marked_centers(grid, fg, ctrl))
    if marked_now != set((int(r), int(c)) for r, c in intended_marked):
        return None

    inp = grid.copy()
    expected = _expected_output(inp, fg, ctrl)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, _expected_all_dilate(inp, fg)):
        return None

    if int(np.sum(actual == 8)) != 9 * len(marked_now):
        return None

    ctrl_to_teal = int(np.sum((inp == ctrl) & (actual == 8)))
    if ctrl_to_teal != len(marked_now):
        return None

    bg_to_teal = int(np.sum((inp == bg) & (actual == 8)))
    if bg_to_teal != 3 * len(marked_now):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
