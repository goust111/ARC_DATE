# Variant of 1818057f
# Dimension: D12 — 对称性检测与利用 (meta)
# Change: '+'字图案本身是关于中心点四重旋转对称的，但规则中不显式利用对称性... → 修复'+'的对称性缺陷

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

# Design analysis:
# variant:
#   concrete_form:
#     - Keep the original palette and global look: red background, one yellow foreground color,
#       mixed with line/blob distractors.
#     - Replace "complete plus gets recolored" with "broken plus gets repaired": the informative
#       local object is now a thin 3-of-4 cross centered on a foreground pixel.
#     - A valid broken plus has foreground at the center and at exactly three orthogonal neighbors,
#       background at the fourth neighbor, and background on all four diagonals. Output fills only
#       the missing orthogonal cell using the same foreground color.
#   behavior_trace:
#     - Original: a full 5-cell plus already exists in the input, and the solver marks those five
#       cells teal.
#     - Variant: the output-first generator still starts from a full 5-cell plus, but derives the
#       input by deleting one arm cell. The solver therefore searches for 4-fold symmetry defects
#       and restores the missing arm instead of recoloring an existing full plus.
#     - Example: if center (r,c) has fg at up/down/left but bg at right, original logic does
#       nothing because the right arm is absent; variant logic infers right by rotational symmetry
#       and writes fg at (r,c+1).
#   complexity_calibration:
#     - "Exactly three orthogonal neighbors" alone would also match arbitrary T-junctions inside
#       thick clutter, which is too ambiguous. Requiring all four diagonals to stay bg keeps the
#       motif visibly thin and makes the symmetry defect legible to a human.
#     - Extensions are allowed only beyond existing arms, starting two cells away from the center,
#       so the motif remains visually close to the original embedded-plus task without creating
#       extra candidate centers near the core.
#   implicit_assumption_audit:
#     - Original `main()` assumes every target center already has 4 orthogonal fg neighbors and
#       that overwriting those cells with teal cannot hide needed information.
#     - That assumption no longer holds: now a target center must have exactly one missing arm, and
#       unconditional recoloring would destroy the intended "same-color completion" rule.
#     - The new solver reads only from the original grid copy `g` and writes into `out`, so filling
#       one gap cannot create chain reactions during the same pass.
#   placement_legality:
#     - Generator uses output-first construction: stamp complete pluses in `out`, remove one arm in
#       `inp`, and validate `main(inp) == out`.
#     - Motif centers stay off the border, diagonals around each center are protected, and extra
#       distractor shapes are accepted only if that equality still holds.
#   observability_and_cleanliness:
#     - The visual change is a single missing yellow arm becoming present in the output; this is
#       obvious in 2-4 examples and does not rely on color-id arithmetic.
#     - Restricting to one missing arm type per motif and preserving thin diagonals avoids messy
#       overlap, making the repaired symmetry easy to see.

_N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
]


def _repair_site(grid: np.ndarray, r: int, c: int, fg: int, bg: int) -> tuple[int, int] | None:
    if int(grid[r, c]) != fg:
        return None

    present = []
    missing = []
    for dr, dc in _N4:
        if int(grid[r + dr, c + dc]) == fg:
            present.append((dr, dc))
        else:
            missing.append((dr, dc))

    if len(present) != 3 or len(missing) != 1:
        return None

    if any(int(grid[r + dr, c + dc]) != bg for dr, dc in _DIAGS):
        return None

    return missing[0]


def _find_repair_sites(grid: np.ndarray, fg: int, bg: int) -> list[tuple[int, int, tuple[int, int]]]:
    h, w = grid.shape
    found: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            miss = _repair_site(grid, r, c, fg, bg)
            if miss is not None:
                found.append((r, c, miss))
    return found


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair thin almost-'+' shapes by restoring their missing arm.

    Input structure:
    - bg = most common color.
    - One foreground color forms mixed thin patterns on the bg; some local motifs look like a '+' with exactly one orthogonal arm missing.

    Rule:
    - A repairable center is any foreground pixel with foreground on exactly 3 of its 4 orthogonal neighbors, background on the remaining side, and background on all 4 diagonals.
    - Fill the single missing orthogonal neighbor with the same foreground color.
    - All existing pixels otherwise stay unchanged.

    Key insight: The local object is recognized by broken 4-fold rotational symmetry, not by a finished cross. The output restores the unique missing arm that makes the thin motif a complete '+'.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            miss = _repair_site(g, r, c, fg, bg)
            if miss is None:
                continue
            dr, dc = miss
            out[r + dr, c + dc] = fg
    return out


def _place_motif(
    out: np.ndarray,
    center: tuple[int, int],
    missing_dir: tuple[int, int],
    rng: np.random.Generator,
    fg: int,
) -> tuple[bool, set[tuple[int, int]]]:
    h, w = out.shape
    r, c = center

    core = {(r, c)}
    for dr, dc in _N4:
        core.add((r + dr, c + dc))
    diags = {(r + dr, c + dc) for dr, dc in _DIAGS}

    if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in core | diags):
        return False, set()
    if any(int(out[rr, cc]) != 2 for rr, cc in core | diags):
        return False, set()

    ext_dirs = [d for d in _N4 if d != missing_dir]
    rng.shuffle(ext_dirs)
    n_ext = int(rng.integers(1, min(3, len(ext_dirs)) + 1))
    cells = set(core)

    for dr, dc in ext_dirs[:n_ext]:
        extra_len = int(rng.integers(0, 3))
        for step in range(2, 2 + extra_len):
            rr, cc = r + dr * step, c + dc * step
            if not (0 <= rr < h and 0 <= cc < w):
                break
            cells.add((rr, cc))

    if any(int(out[rr, cc]) != 2 for rr, cc in cells):
        return False, set()

    for rr, cc in cells:
        out[rr, cc] = fg
    return True, cells | diags


def _carve_input(output_grid: np.ndarray, removed: list[tuple[int, int]]) -> np.ndarray:
    inp = output_grid.copy()
    for r, c in removed:
        inp[r, c] = 2
    return inp


def _try_add_shape(
    inp: np.ndarray,
    out: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    blocked: set[tuple[int, int]],
    expected_sites: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    if not cells:
        return False
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if any((r, c) in blocked for r, c in cells):
        return False
    if any(int(inp[r, c]) != 2 or int(out[r, c]) != 2 for r, c in cells):
        return False

    inp2 = inp.copy()
    out2 = out.copy()
    for r, c in cells:
        inp2[r, c] = fg
        out2[r, c] = fg

    solved = main(inp2)
    if not np.array_equal(solved, out2):
        return False

    sites = {(r, c) for r, c, _ in _find_repair_sites(inp2, fg, 2)}
    if sites != expected_sites:
        return False

    inp[:, :] = inp2
    out[:, :] = out2
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg, fg = 2, 4

    out = np.full((h, w), bg, dtype=int)

    max_k = min(7, max(1, (h * w) // 35))
    min_k = 1 if max_k == 1 else 2
    k = int(rng.integers(min_k, max_k + 1))

    centers: list[tuple[int, int]] = []
    removed_cells: list[tuple[int, int]] = []
    expected_sites: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    attempts = 0
    while len(centers) < k and attempts < k * 80:
        attempts += 1
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(2, w - 2))
        center = (r, c)
        if any(max(abs(r - rr), abs(c - cc)) < 4 for rr, cc in centers):
            continue

        missing_dir = _N4[int(rng.integers(0, 4))]
        draft = out.copy()
        ok, protect = _place_motif(draft, center, missing_dir, rng, fg)
        if not ok:
            continue

        cand_inp = _carve_input(draft, removed_cells + [(r + missing_dir[0], c + missing_dir[1])])
        sites = _find_repair_sites(cand_inp, fg, bg)
        site_centers = {(rr, cc) for rr, cc, _ in sites}
        if site_centers != expected_sites | {center}:
            continue

        out = draft
        centers.append(center)
        removed_cells.append((r + missing_dir[0], c + missing_dir[1]))
        expected_sites.add(center)
        blocked.update(protect)

    if len(centers) != k:
        return None

    inp = _carve_input(out, removed_cells)
    if {(r, c) for r, c, _ in _find_repair_sites(inp, fg, bg)} != expected_sites:
        return None
    if not np.array_equal(main(inp), out):
        return None

    total = h * w
    target_density = float(rng.uniform(0.24, 0.38))
    target_fg = int(total * target_density)
    cur_fg = int(np.sum(out != bg))

    for _ in range(total * 8):
        if cur_fg >= target_fg:
            break
        shape = _SHAPES[int(rng.integers(0, len(_SHAPES)))]
        base_r = int(rng.integers(0, h))
        base_c = int(rng.integers(0, w))
        cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
        if _try_add_shape(inp, out, cells, fg, blocked, expected_sites):
            cur_fg = int(np.sum(out != bg))

    if int(np.sum(out != bg)) >= total // 2:
        return None
    if {(r, c) for r, c, _ in _find_repair_sites(inp, fg, bg)} != expected_sites:
        return None
    if not np.array_equal(main(inp), out):
        return None
    if len(removed_cells) < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
