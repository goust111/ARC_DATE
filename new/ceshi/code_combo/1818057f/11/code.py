# Combo variant of 1818057f (index 11)
# combo_spec: D7/0+D14/2
# Name: 裁剪区域+奇偶着色

from __future__ import annotations

import numpy as np
from arc_utils import bbox, crop_bbox, get_bg


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   preserved_scene:
#     - The input still has one dominant background color and one foreground color.
#     - Foreground pixels form noisy lines / blobs, and some interior locations satisfy
#       the local thin-plus test: the center and its 4 orthogonal neighbors are foreground.
#     - Intended plus footprints remain non-overlapping, with centers at least 2 cells from
#       the border and pairwise Manhattan distance >= 4, matching the original placement style.
#   combo_core:
#     - D14/2 changes the paint decision from fixed-color to count-conditioned:
#       first collect every valid plus center, then choose one shared highlight color.
#       Odd plus count -> teal (8); even plus count -> magenta (6).
#     - D7/0 changes the output frame:
#       after repainting those plus footprints, crop the output to the minimal bbox of the
#       union of all repainted plus cells.
#     - The two changes interact: the crop is computed from the same globally counted set
#       of plus footprints whose count decides the color.
#   concrete_trace:
#     - Suppose the input contains 4 detected centers at (4,5), (4,10), (8,5), (8,10).
#       Original task: repaint all four 5-cell crosses teal on the full canvas.
#       This combo: detect the same 4 centers, count = even, repaint every plus cell magenta,
#       then crop to rows 3..9 and cols 4..11, discarding unrelated foreground outside.
#     - If one center is removed so only 3 remain, the crop region may also shrink, but the
#       key change is the color branch flipping from magenta to teal before cropping.
#   complexity_calibration:
#     - Only two latent decisions matter to the solver: the global parity bit and the tight
#       bbox of the union of all plus footprints. Both are visually recoverable from a few
#       examples because the output is smaller than the input and the highlight color has only
#       two repeated possibilities.
#     - The generator alternates requested parity across calls so both color branches appear
#       often in a small training set instead of relying on chance.
#   implicit_assumption_audit:
#     - `bg = get_bg(g)` remains valid because the generator keeps background as the majority.
#     - `fg = first non-bg color` remains valid because inputs still use exactly one foreground color.
#     - Detection must read from the original grid `g`, not from the recolored output, because
#       both the parity decision and the crop bbox depend on the full center set.
#     - The old immediate repaint loop is no longer conceptually safe: repaint color depends on
#       the final plus count, so centers must be collected before painting.
#     - Returning the full canvas is no longer valid: after painting, solver must crop using the
#       bbox of the detected plus-footprint union, not the bbox of all foreground.
#   placement_legality:
#     - Generator samples dimensions large enough to host the requested plus count with spacing.
#     - Extension cells are only accepted if the detected plus-center set stays exactly equal to
#       the intended one; distractor cells are accepted only if they create no new plus.
#     - At least one foreground distractor is forced outside the plus-footprint bbox so the crop
#       rule is observable and the output is strictly smaller than the input.
#   observability_and_tidiness:
#     - Every accepted sample has at least one row/column removed by the crop on all sides of the
#       plus bbox, so the framing change is immediately visible.
#     - Output highlight color is uniform across all pluses in a sample, making the parity branch
#       legible without introducing many colors.
#     - We avoid layered annotations: only original fg survives, plus footprints switch to one
#       highlight color, and cropping removes outside clutter cleanly.
#   new_solver_assumptions_checked_against_generator:
#     - `main()` assumes there is at least one detected plus for nontrivial samples; generator
#       guarantees 1..9 pluses and validates that no accidental extra centers appear.
#     - `main()` assumes the plus-footprint bbox is non-empty before cropping; generator enforces
#       intended centers and solver guards the no-center fallback anyway.
#     - `main()` assumes parity is meaningful because different counts can occur; generator
#       alternates odd/even targets to guarantee both branches across sampled instances.


PLUS_DELTAS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ORTHO_DELTAS = PLUS_DELTAS[1:]
TEAL = 8
MAGENTA = 6
_NEXT_PARITY = 1  # 1 = odd, 0 = even


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DELTAS):
                centers.append((r, c))
    return centers


def _plus_mask_from_centers(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in centers:
        for dr, dc in PLUS_DELTAS:
            mask[r + dr, c + dc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs, choose their paint color from the total motif count, and crop to their span.

    Input structure:
    - The most frequent color is the background.
    - Exactly one other color forms mixed foreground clutter, and some interior locations make 5-cell '+' motifs.

    Rule:
    - First find every foreground cell whose 4 orthogonal neighbors are also foreground.
    - If the number of detected '+' centers is odd, recolor every matched 5-cell footprint to teal (8); if even, recolor them all to magenta (6).
    - After recoloring, crop the grid to the smallest bounding rectangle covering the union of all recolored '+' cells.

    Key insight: The local shape test is unchanged, but the solver cannot paint or crop greedily anymore — it must collect the full set of '+' motifs first, because that same global set determines both the shared output color and the final crop rectangle.
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

    plus_mask = _plus_mask_from_centers(g.shape, centers)
    highlight = TEAL if (len(centers) % 2 == 1) else MAGENTA
    out[plus_mask] = highlight

    plus_box = bbox(plus_mask)
    assert plus_box is not None
    return crop_bbox(out, plus_box)


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether changing one bg cell to fg would create a plus at that cell or an orthogonal neighbor."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        good = True
        for dr, dc in PLUS_DELTAS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                good = False
                break
        if good:
            return True
    return False


def _choose_count(rng: np.random.Generator, target_parity: int) -> int:
    options = [k for k in range(1, 10) if k % 2 == target_parity]
    return int(options[int(rng.integers(0, len(options)))])


def _sample_shape_for_count(rng: np.random.Generator, count: int) -> tuple[int, int]:
    """Pick dimensions that can comfortably host `count` spaced centers with crop margins."""
    n_rows = int(np.ceil(np.sqrt(count)))
    n_cols = int(np.ceil(count / n_rows))
    H = int(rng.integers(max(10, 4 * n_rows + 1), 23))
    W = int(rng.integers(max(12, 4 * n_cols + 1), 23))
    # Preserve the original task's width lower bound when swapping orientation.
    if H >= 12 and rng.random() < 0.5:
        H, W = W, H
    return H, W


def _try_add_extension_cell(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    intended_centers: set[tuple[int, int]],
) -> bool:
    """Accept one extension cell only if the detected center set stays exactly the intended set."""
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    if int(grid[r, c]) == fg:
        return False

    old = int(grid[r, c])
    grid[r, c] = fg
    centers = set(_find_plus_centers(grid, fg))
    ok = centers == intended_centers
    if not ok:
        grid[r, c] = old
    return ok


def _add_arm_extensions(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
) -> bool:
    """Embed each plus into a larger fg structure while preserving the center set."""
    intended = set(centers)
    for cr, cc in centers:
        extended = False
        target_dirs = int(rng.integers(1, 3))
        added_dirs = 0
        for idx in rng.permutation(len(ORTHO_DELTAS)):
            dr, dc = ORTHO_DELTAS[int(idx)]
            ext_len = int(rng.integers(1, 4))
            grew_this_dir = False
            for step in range(1, ext_len + 1):
                nr = cr + dr * (step + 1)
                nc = cc + dc * (step + 1)
                if not _try_add_extension_cell(grid, nr, nc, fg, intended):
                    break
                extended = True
                grew_this_dir = True
            if grew_this_dir:
                added_dirs += 1
                if added_dirs >= target_dirs:
                    break
        if not extended:
            return False
    return True


def _try_place_cluster(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    fg: int,
) -> bool:
    placed: list[tuple[int, int]] = []
    for r, c in cells:
        if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
            for rr, cc in placed:
                grid[rr, cc] = bg
            return False
        if int(grid[r, c]) == fg:
            continue
        if _would_create_plus(grid, r, c, fg):
            for rr, cc in placed:
                grid[rr, cc] = bg
            return False
        grid[r, c] = fg
        placed.append((r, c))
    return bool(placed)


def _place_outside_cluster(
    grid: np.ndarray,
    plus_box: tuple[int, int, int, int],
    bg: int,
    fg: int,
    rng: np.random.Generator,
) -> bool:
    """Force at least one foreground distractor strictly outside the final crop box."""
    r0, c0, r1, c1 = plus_box
    H, W = grid.shape
    outside = [(r, c) for r in range(H) for c in range(W) if not (r0 <= r <= r1 and c0 <= c <= c1)]
    rng.shuffle(outside)

    line_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for ar, ac in outside:
        shapes = [
            [(ar, ac)],
            [(ar, ac), (ar, ac + 1)],
            [(ar, ac), (ar + 1, ac)],
        ]
        dr, dc = line_dirs[int(rng.integers(len(line_dirs)))]
        shapes.append([(ar + step * dr, ac + step * dc) for step in range(3)])
        rng.shuffle(shapes)
        for cells in shapes:
            if _try_place_cluster(grid, cells, bg, fg):
                return True
    return False


def _sprinkle_distractors(
    grid: np.ndarray,
    bg: int,
    fg: int,
    rng: np.random.Generator,
    target_density: float,
) -> None:
    total = grid.shape[0] * grid.shape[1]
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


def _gen_attempt(target_parity: int) -> np.ndarray | None:
    rng = np.random.default_rng()
    bg, fg = 2, 4
    count = _choose_count(rng, target_parity)
    H, W = _sample_shape_for_count(rng, count)

    grid = np.full((H, W), bg, dtype=int)

    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)
    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((cr, cc))
            if len(centers) == count:
                break
    if len(centers) != count:
        return None

    for cr, cc in centers:
        for dr, dc in PLUS_DELTAS:
            grid[cr + dr, cc + dc] = fg

    plus_mask = _plus_mask_from_centers(grid.shape, centers)
    plus_box = bbox(plus_mask)
    if plus_box is None:
        return None
    r0, c0, r1, c1 = plus_box
    if not (r0 >= 1 and c0 >= 1 and r1 <= H - 2 and c1 <= W - 2):
        return None

    if not _add_arm_extensions(grid, centers, fg, rng):
        return None

    if not _place_outside_cluster(grid, plus_box, bg, fg, rng):
        return None

    _sprinkle_distractors(grid, bg, fg, rng, target_density=float(rng.uniform(0.26, 0.41)))

    current_fg = int(np.sum(grid != bg))
    density = current_fg / (H * W)
    if density < 0.26 or density > 0.41:
        return None
    if current_fg >= (H * W) // 2:
        return None

    actual_centers = _find_plus_centers(grid, fg)
    if sorted(actual_centers) != sorted(centers):
        return None

    actual_mask = _plus_mask_from_centers(grid.shape, actual_centers)
    actual_box = bbox(actual_mask)
    if actual_box is None or actual_box != plus_box:
        return None

    outside_crop_fg = int(np.sum(grid[:r0, :] != bg))
    outside_crop_fg += int(np.sum(grid[r1 + 1 :, :] != bg))
    outside_crop_fg += int(np.sum(grid[r0 : r1 + 1, :c0] != bg))
    outside_crop_fg += int(np.sum(grid[r0 : r1 + 1, c1 + 1 :] != bg))
    if outside_crop_fg <= 0:
        return None

    out = main(grid)
    expected_color = TEAL if (count % 2 == 1) else MAGENTA
    wrong_color = MAGENTA if expected_color == TEAL else TEAL
    if out.shape == grid.shape:
        return None
    if out.shape != (r1 - r0 + 1, c1 - c0 + 1):
        return None
    if not np.any(out == expected_color):
        return None
    if np.any(out == wrong_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_PARITY
    target_parity = _NEXT_PARITY
    _NEXT_PARITY = 1 - _NEXT_PARITY

    for _ in range(500):
        result = _gen_attempt(target_parity)
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
