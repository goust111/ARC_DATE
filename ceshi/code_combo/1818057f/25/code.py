# Combo variant of 1818057f (index 25)
# combo_spec: D3/0+D8/0+D10/1
# Name: X型与T型混合高迷惑检测

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

# Design analysis
# ---
# concretization_and_behavior_trace:
#   - The source task recolored a 5-cell orthogonal '+' when a foreground center
#     had all 4 orthogonal neighbors occupied.
#   - This combo splits the old notion of "almost plus" into a true target and a
#     distractor, while also adding a diagonal target. The two true targets are:
#       1. T motif: center + exactly 3 orthogonal neighbors -> recolor those
#          4 cells to teal (8).
#       2. X motif: center + all 4 diagonal neighbors -> recolor those 5 cells
#          to magenta/pink (6).
#   - Small trace:
#       2 4 2        old source: nothing
#       4 4 4   ->   combo: recolor these 4 cells teal because degree-3 orthogonal
#       2 2 2        support is now a target.
#     and
#       4 2 4        old source: nothing
#       2 4 2   ->   combo: recolor these 5 cells magenta because the match is
#       4 2 4        purely diagonal.
#     A full '+' remains unchanged in the combo because it is degree 4, not an
#     exact T, and it does not satisfy the diagonal X test.
# complexity_calibration:
#   - T detection uses "exactly 3" orthogonal neighbors. Using "at least 3"
#     would collapse T and '+' together and erase the D3 change.
#   - X detection keeps the diagonal-only criterion from D10, but the generator
#     mixes it with T targets in the same board so the two output colors must be
#     inferred jointly from shape rather than from separate datasets.
#   - High-confusion distractors are deliberate: the generator adds pseudo-T
#     centers with only 2 orthogonal neighbors and pseudo-X centers with only 2
#     or 3 diagonal neighbors. These near-misses are repeated across samples so
#     the human can learn that "close" is not enough.
# implicit_assumptions_review:
#   - `get_bg()` as most-common-color is still valid; the generator keeps color 2
#     as the majority and uses a single input foreground color 4.
#   - `main()` still reads detections from the original input grid, collects T and
#     X hits first, then recolors. This avoids order dependence if two detected
#     motifs are nearby.
#   - The old single predicate `all(orthogonals)` is no longer sufficient. The
#     solver must maintain two local tests with different footprints and colors.
#   - The new solver silently assumes a detected T footprint never overlaps a
#     detected X footprint; otherwise one recolor would overwrite the other. The
#     generator enforces disjoint occupied cells for all planted motifs, so this
#     assumption holds on produced samples.
# placement_legality:
#   - Every planned center is at least 2 cells from the border, so T and X cores
#     fit and each motif can be extended by at least one extra foreground cell.
#   - Candidate motifs and distractors are committed only if, after the addition,
#     the exact set of T centers equals the planned T set and the exact set of X
#     centers equals the planned X set. This rejects accidental targets.
#   - X centers reserve their 4 orthogonal neighbors as background to keep the
#     diagonal evidence clean. T centers implicitly reserve their missing side,
#     because filling it would destroy the exact-degree-3 invariant.
# visual_observability:
#   - The same input color (yellow) can lead to either teal or magenta output, but
#     the distinction is tied to visible local geometry: orthogonal T versus
#     diagonal X. No invisible ordering or color-id rule is involved.
#   - Each generated board contains at least one true T, at least one true X, at
#     least one pseudo-T, and at least one pseudo-X, so the fusion behavior is
#     inferable from a few examples rather than from a single rare motif.
# cleanliness_and_inferability:
#   - Motifs never overlap and are spaced apart with a coarse center-distance
#     filter, which keeps the board readable even at the original density range.
#   - Output colors are limited to the original foreground replacement style:
#     teal marks all T footprints, magenta marks all X footprints, and ordinary
#     distractors stay yellow. The small color vocabulary keeps the mapping easy
#     to infer.

ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
X_FOOTPRINT = [(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
NOISE_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
]

BG = 2
FG = 4
T_MARK = 8
X_MARK = 6


def _cycled_choices(
    rng: np.random.Generator,
    options: list[tuple[int, int]] | list[int],
    count: int,
) -> list[tuple[int, int]] | list[int]:
    order = list(map(int, rng.permutation(len(options))))
    return [options[order[i % len(options)]] for i in range(count)]


def _t_arm_offsets(grid: np.ndarray, r: int, c: int, fg: int) -> list[tuple[int, int]] | None:
    if int(grid[r, c]) != fg:
        return None
    arms = [(dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == fg]
    return arms if len(arms) == 3 else None


def _is_x_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return int(grid[r, c]) == fg and all(int(grid[r + dr, c + dc]) == fg for dr, dc in DIAG_DIRS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect exact orthogonal T motifs and diagonal X motifs in one foreground color and recolor them.

    Input structure:
    - The background is the most frequent color in the grid.
    - A single non-background color forms mixed strands, branches, and compact near-miss distractors.
    - Some local motifs are exact T shapes with 3 orthogonal arms; others are exact X shapes with 4 diagonal arms.

    Rule:
    - If a foreground center has exactly 3 orthogonal foreground neighbors, recolor the center and those
      3 arm cells to teal (8).
    - If a foreground center has all 4 diagonal foreground neighbors, recolor the center and those
      4 diagonal cells to magenta/pink (6).
    - Full '+' crosses, partial T-like branches, and partial X-like diagonals are left unchanged.

    Key insight:
    - The same yellow foreground splits by local geometry into two output colors: orthogonal degree-3
      junctions become teal, while diagonal 5-cell X footprints become magenta. The solver must inspect
      both neighborhood systems on the original grid, not reason by connected components.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    t_hits: list[tuple[int, int, list[tuple[int, int]]]] = []
    x_hits: list[tuple[int, int]] = []

    for r in range(1, H - 1):
        for c in range(1, W - 1):
            arms = _t_arm_offsets(g, r, c, fg)
            if arms is not None:
                t_hits.append((r, c, arms))
            if _is_x_center(g, r, c, fg):
                x_hits.append((r, c))

    for r, c, arms in t_hits:
        out[r, c] = T_MARK
        for dr, dc in arms:
            out[r + dr, c + dc] = T_MARK

    for r, c in x_hits:
        for dr, dc in X_FOOTPRINT:
            out[r + dr, c + dc] = X_MARK

    return out


def _input_view(output_grid: np.ndarray) -> np.ndarray:
    return np.where(output_grid == BG, BG, FG)


def _find_t_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _t_arm_offsets(grid, r, c, fg) is not None:
                centers.append((r, c))
    return centers


def _find_x_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_x_center(grid, r, c, fg):
                centers.append((r, c))
    return centers


def _count_pseudo_t_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            deg = sum(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_DIRS)
            if deg == 2:
                count += 1
    return count


def _count_pseudo_x_centers(grid: np.ndarray, fg: int) -> int:
    H, W = grid.shape
    count = 0
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            diag = sum(int(grid[r + dr, c + dc]) == fg for dr, dc in DIAG_DIRS)
            if diag in (2, 3):
                count += 1
    return count


def _center_far_enough(center: tuple[int, int], others: list[tuple[int, int]], min_cheb: int = 4) -> bool:
    r, c = center
    return all(max(abs(r - orow), abs(c - ocol)) >= min_cheb for orow, ocol in others)


def _cells_in_bounds(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    H, W = shape
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _planned_t_centers(t_specs: list[tuple[int, int, tuple[int, int]]]) -> set[tuple[int, int]]:
    return {(r, c) for r, c, _ in t_specs}


def _planned_x_centers(x_centers: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(x_centers)


def _matches_plan(
    output_grid: np.ndarray,
    planned_t: list[tuple[int, int, tuple[int, int]]],
    planned_x: list[tuple[int, int]],
) -> bool:
    inp = _input_view(output_grid)
    found_t = set(_find_t_centers(inp, FG))
    found_x = set(_find_x_centers(inp, FG))
    return found_t == _planned_t_centers(planned_t) and found_x == _planned_x_centers(planned_x)


def _try_commit(
    output_grid: np.ndarray,
    planned_t: list[tuple[int, int, tuple[int, int]]],
    planned_x: list[tuple[int, int]],
    color_cells: list[tuple[int, int, int]],
) -> bool:
    trial = output_grid.copy()
    H, W = trial.shape

    seen: set[tuple[int, int]] = set()
    for r, c, color in color_cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if (r, c) in seen:
            continue
        seen.add((r, c))
        if int(trial[r, c]) != BG:
            return False
        trial[r, c] = int(color)

    if not _matches_plan(trial, planned_t, planned_x):
        return False

    output_grid[:, :] = trial
    return True


def _t_cells(
    center: tuple[int, int],
    missing: tuple[int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int]]:
    r, c = center
    arms = [d for d in ORTHO_DIRS if d != missing]
    cells = [(r, c, T_MARK)]
    cells.extend((r + dr, c + dc, T_MARK) for dr, dc in arms)

    n_ext = int(rng.integers(1, 3))
    chosen = rng.choice(len(arms), size=n_ext, replace=False)
    for idx in chosen:
        dr, dc = arms[int(idx)]
        ext_len = int(rng.integers(1, 4))
        for step in range(2, ext_len + 2):
            cells.append((r + dr * step, c + dc * step, FG))
    return cells


def _x_cells(center: tuple[int, int], rng: np.random.Generator) -> list[tuple[int, int, int]]:
    r, c = center
    cells = [(r, c, X_MARK)]
    cells.extend((r + dr, c + dc, X_MARK) for dr, dc in DIAG_DIRS)

    n_ext = int(rng.integers(1, 4))
    chosen = rng.choice(len(DIAG_DIRS), size=n_ext, replace=False)
    for idx in chosen:
        dr, dc = DIAG_DIRS[int(idx)]
        ext_len = int(rng.integers(1, 4))
        for step in range(2, ext_len + 2):
            cells.append((r + dr * step, c + dc * step, FG))
    return cells


def _pseudo_t_cells(
    center: tuple[int, int],
    rng: np.random.Generator,
    blocked: tuple[int, int] | None = None,
) -> list[tuple[int, int, int]]:
    r, c = center
    source = [d for d in ORTHO_DIRS if d != blocked] if blocked is not None else list(ORTHO_DIRS)
    chosen = rng.choice(len(source), size=2, replace=False)
    arms = [source[int(idx)] for idx in chosen]
    cells = [(r, c, FG)]
    cells.extend((r + dr, c + dc, FG) for dr, dc in arms)
    if float(rng.random()) < 0.75:
        dr, dc = arms[int(rng.integers(0, len(arms)))]
        ext_len = int(rng.integers(1, 3))
        for step in range(2, ext_len + 2):
            cells.append((r + dr * step, c + dc * step, FG))
    return cells


def _pseudo_x_cells(
    center: tuple[int, int],
    rng: np.random.Generator,
    diag_n: int | None = None,
) -> list[tuple[int, int, int]]:
    r, c = center
    diag_n = 2 if diag_n is None else diag_n
    chosen = rng.choice(len(DIAG_DIRS), size=diag_n, replace=False)
    arms = [DIAG_DIRS[int(idx)] for idx in chosen]
    cells = [(r, c, FG)]
    cells.extend((r + dr, c + dc, FG) for dr, dc in arms)
    if float(rng.random()) < 0.75:
        dr, dc = arms[int(rng.integers(0, len(arms)))]
        ext_len = int(rng.integers(1, 3))
        for step in range(2, ext_len + 2):
            cells.append((r + dr * step, c + dc * step, FG))
    return cells


def _reserved_bg(
    t_specs: list[tuple[int, int, tuple[int, int]]],
    x_centers: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for r, c, missing in t_specs:
        reserved.add((r + missing[0], c + missing[1]))
    for r, c in x_centers:
        for dr, dc in ORTHO_DIRS:
            reserved.add((r + dr, c + dc))
    return reserved


def _noise_cells(anchor: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    r, c = anchor
    return [(r + dr, c + dc, FG) for dr, dc in offsets]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    total = H * W
    output = np.full((H, W), BG, dtype=int)

    max_targets = min(8, max(2, total // 28))
    target_total = int(rng.integers(2, max_targets + 1))
    t_count = int(rng.integers(1, target_total))
    x_count = target_total - t_count
    if x_count < 1:
        return None

    pseudo_t_goal = int(rng.integers(1, min(4, t_count + 2) + 1))
    pseudo_x_goal = int(rng.integers(1, min(4, x_count + 2) + 1))
    t_missing_plan = _cycled_choices(rng, ORTHO_DIRS, t_count)
    pseudo_t_blocked = _cycled_choices(rng, ORTHO_DIRS, pseudo_t_goal)
    pseudo_x_sizes = [2 + (i % 2) for i in range(pseudo_x_goal)]
    rng.shuffle(pseudo_x_sizes)

    t_specs: list[tuple[int, int, tuple[int, int]]] = []
    x_centers: list[tuple[int, int]] = []
    all_centers: list[tuple[int, int]] = []

    for i in range(t_count):
        placed = False
        missing = t_missing_plan[i]
        for _ in range(180):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            if not _center_far_enough(center, all_centers):
                continue
            trial_specs = t_specs + [(center[0], center[1], missing)]
            if _try_commit(output, trial_specs, x_centers, _t_cells(center, missing, rng)):
                t_specs = trial_specs
                all_centers.append(center)
                placed = True
                break
        if not placed:
            return None

    for _ in range(x_count):
        placed = False
        for _ in range(220):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            if not _center_far_enough(center, all_centers):
                continue
            trial_x = x_centers + [center]
            if _try_commit(output, t_specs, trial_x, _x_cells(center, rng)):
                x_centers = trial_x
                all_centers.append(center)
                placed = True
                break
        if not placed:
            return None

    reserved_bg = _reserved_bg(t_specs, x_centers)

    pseudo_t_done = 0
    for blocked in pseudo_t_blocked:
        placed = False
        for _ in range(90):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            if center in reserved_bg or not _center_far_enough(center, all_centers, min_cheb=3):
                continue
            if _try_commit(output, t_specs, x_centers, _pseudo_t_cells(center, rng, blocked)):
                pseudo_t_done += 1
                all_centers.append(center)
                placed = True
                break
        if not placed:
            continue
    if pseudo_t_done < 1:
        return None

    pseudo_x_done = 0
    for diag_n in pseudo_x_sizes:
        placed = False
        for _ in range(90):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            if center in reserved_bg or not _center_far_enough(center, all_centers, min_cheb=3):
                continue
            if _try_commit(output, t_specs, x_centers, _pseudo_x_cells(center, rng, diag_n)):
                pseudo_x_done += 1
                all_centers.append(center)
                placed = True
                break
        if not placed:
            continue
    if pseudo_x_done < 1:
        return None

    target_density = float(rng.uniform(0.22, 0.3))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(output != BG))
    noise_anchors: list[tuple[int, int]] = []

    for _ in range(total * 20):
        if current_fg >= target_fg:
            break
        anchor = (int(rng.integers(0, H)), int(rng.integers(0, W)))
        if not _center_far_enough(anchor, noise_anchors, min_cheb=2):
            continue
        offsets = NOISE_SHAPES[int(rng.integers(0, len(NOISE_SHAPES)))]
        cells = _noise_cells(anchor, offsets)
        if any((r, c) in reserved_bg for r, c, _ in cells):
            continue
        if _try_commit(output, t_specs, x_centers, cells):
            current_fg += len(cells)
            noise_anchors.append(anchor)

    input_grid = _input_view(output)
    pseudo_t_count = _count_pseudo_t_centers(input_grid, FG)
    pseudo_x_count = _count_pseudo_x_centers(input_grid, FG)
    pseudo_cap = max(12, total // 10)

    if int(np.sum(input_grid != BG)) >= total // 2:
        return None
    if set(_find_t_centers(input_grid, FG)) != _planned_t_centers(t_specs):
        return None
    if set(_find_x_centers(input_grid, FG)) != _planned_x_centers(x_centers):
        return None
    if pseudo_t_count < 1 or pseudo_t_count > pseudo_cap:
        return None
    if pseudo_x_count < 1 or pseudo_x_count > pseudo_cap:
        return None
    if not np.any(output == T_MARK) or not np.any(output == X_MARK):
        return None
    if not np.array_equal(main(input_grid), output):
        return None

    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
