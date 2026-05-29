# Combo variant of 1818057f (index 8)
# combo_spec: D3/0+D15/1
# Name: 样板匹配T型图案

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

# Design analysis
# ---------------
# a) Concretization and behavior trace
#    - The source task colors any local '+' center: one yellow cell with all 4
#      orthogonal yellow neighbors. This fusion changes both the cue and the
#      target.
#    - The top-left 3x3 corner now contains a teal T sample, for example:
#          8 8 8
#          2 8 2
#          2 2 2
#      The sample is an instruction, not a candidate in the main field.
#    - Elsewhere, yellow components contain local branch points. A target is a
#      yellow center with exactly 3 orthogonal yellow neighbors; its 4 cells are
#      recolored to teal. A yellow '+' with 4 neighbors is now a distractor.
#    - The legend may point upward, downward, left, or right, but the solver
#      must recolor all rotated T variants. This is visible because each grid
#      contains target Ts in at least two orientations, including one that does
#      not match the legend's exact rotation.
#    - Concrete difference:
#        legend = upward T, field contains both:
#          2 4 2        2 4 2
#          4 4 4   and  4 4 4
#          2 2 2        2 4 2
#      Fusion rule paints the left motif (a T) and leaves the right motif (a +)
#      yellow. The source task did the opposite.
#
# b) Complexity calibration
#    - The legend uses only one symbol and one output color, so the clue is
#      explicit instead of requiring color-id guesswork.
#    - The field uses one searchable color, preserving the original task's
#      simplicity while still forcing the solver to separate legend color from
#      search color.
#    - The generator enforces at least one plus-shaped decoy and at least one
#      target T whose orientation differs from the legend. This prevents
#      degenerate "legend is decorative" samples and makes both modified
#      dimensions observable from a few examples.
#
# c) Implicit assumptions review
#    - The source `colors[0]` assumption breaks because the input now contains
#      both the yellow field color and the teal legend color. `main()` must
#      extract the legend color first, then search the remaining non-bg colors.
#    - The source `all(four neighbors)` predicate no longer matches the target.
#      The new predicate is "exactly 3 orthogonal neighbors" and is tied back to
#      the corner sample by reconstructing a T arm-set from the legend.
#    - Recolor order still must not affect detection, so `main()` reads only
#      from the original grid and writes to a copy.
#
# d) Placement legality
#    - A 5x5 corner reserve keeps the legend isolated.
#    - Main-field components are placed with a 4-neighbor moat, so targets,
#      plus decoys, and small distractors never merge accidentally.
#    - Target T components are built from an exact 4-cell core plus optional
#      straight extensions along existing arms; this keeps the designated center
#      at degree 3 while avoiding accidental extra T centers.
#
# e) Visual observability
#    - Humans can align the teal corner T with the yellow branch points and see
#      that only 3-armed junctions change.
#    - Because every sample also contains at least one yellow '+', the rule
#      cannot be mistaken for the original cross-highlighting task.
#    - Because some recolored Ts disagree with the legend's exact facing, the
#      task reads as "same shape up to rotation" rather than "copy this exact
#      pose only".
#
# f) Visual cleanliness
#    - The legend has its own moat, and main-field components are separated.
#      This keeps the grid readable even with multiple distractor families.
#    - Extra noise is limited to small lines/L-shapes/blocks whose local degree
#      never reaches 3, so they clutter the scene without obscuring the rule.
#
# g) Color-change inferability
#    - Only three colors appear: red background, yellow search objects, teal
#      legend/output. The legend anchors teal as the recolor target.
#
# i) New assumptions after modification
#    - `main()` assumes the top-left 3x3 window contains exactly one connected
#      teal T template of size 4. The generator paints exactly that.
#    - `main()` assumes searchable objects do not use the legend color in the
#      input. The generator keeps teal exclusive to the legend before solving.
#    - `main()` assumes all target motifs are decided by orthogonal adjacency
#      alone. The generator avoids accidental degree-3 junctions outside the
#      planned target set.

ORTHO_DIRS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
RESERVED = 5
BG = 2
FG = 4
TARGET = 8

DISTRACTOR_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _rotate_offsets(offsets: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((-dc, dr) for dr, dc in offsets))


def _arm_variants(arms: tuple[tuple[int, int], ...]) -> tuple[frozenset[tuple[int, int]], ...]:
    variants: list[frozenset[tuple[int, int]]] = []
    cur = tuple(sorted(arms))
    for _ in range(4):
        frozen = frozenset(cur)
        if frozen not in variants:
            variants.append(frozen)
        cur = _rotate_offsets(cur)
    return tuple(variants)


def _extract_corner_template(g: np.ndarray, bg: int) -> tuple[int, tuple[tuple[int, int], ...]] | None:
    if g.shape[0] < 3 or g.shape[1] < 3:
        return None
    zone = g[:3, :3]
    colors = sorted(set(map(int, zone.flatten())) - {int(bg)})
    if len(colors) != 1:
        return None
    color = int(colors[0])
    cells = [tuple(map(int, coord)) for coord in np.argwhere(zone == color)]
    if len(cells) != 4:
        return None

    cell_set = set(cells)
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
        degrees[(r, c)] = int(deg)

    centers = [cell for cell, deg in degrees.items() if deg == 3]
    leaves = [cell for cell, deg in degrees.items() if deg == 1]
    if len(centers) != 1 or len(leaves) != 3:
        return None

    cr, cc = centers[0]
    arms = tuple(sorted((int(r - cr), int(c - cc)) for r, c in leaves))
    if any(offset not in ORTHO_DIRS for offset in arms):
        return None
    return color, arms


def _matching_t_arms(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    allowed_variants: tuple[frozenset[tuple[int, int]], ...],
) -> tuple[tuple[int, int], ...] | None:
    if int(grid[r, c]) != int(fg):
        return None
    arms = tuple(sorted((dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == int(fg)))
    if len(arms) != 3:
        return None
    return arms if frozenset(arms) in allowed_variants else None


def _missing_dir_from_arms(arms: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arm_set = set(arms)
    for direction in ORTHO_DIRS:
        if direction not in arm_set:
            return direction
    raise ValueError("arms do not define a T")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner legend to identify T junctions and recolor them to teal.

    Input structure:
    - The background is the most common color.
    - The top-left 3x3 corner contains a small teal T sample that shows both the target shape and the output color.
    - Elsewhere, one other foreground color forms isolated or extended yellow components; some contain local T junctions and some are '+' or path-like distractors.

    Rule:
    - Read the corner sample, recover its 3 orthogonal arms, and treat every rotation of that T as a valid target.
    - For each non-legend foreground color, find every pixel whose orthogonal neighborhood matches one of those rotated T arm-sets exactly.
    - Recolor the center and its 3 occupied arm neighbors to the legend color.
    - Leave the legend, '+' centers, and every non-T cell unchanged.

    Key insight:
    - The corner object is an instruction and only defines the motif family; matching is still local and rotation-invariant, so a differently oriented T in the field must be recolored even when it does not copy the legend's exact pose.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    template = _extract_corner_template(g, bg)
    if template is None:
        return out
    target_color, arms = template
    allowed_variants = _arm_variants(arms)

    colors = sorted(set(map(int, g.flatten())) - {bg, target_color})
    if not colors:
        return out

    H, W = g.shape
    for fg in colors:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                matched = _matching_t_arms(g, r, c, int(fg), allowed_variants)
                if matched is None:
                    continue
                out[r, c] = int(target_color)
                for dr, dc in matched:
                    out[r + dr, c + dc] = int(target_color)
    return out


def _legacy_plus_main(grid: np.ndarray) -> np.ndarray:
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
            if int(g[r, c]) != int(fg):
                continue
            if all(int(g[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO_DIRS):
                for dr, dc in ((0, 0),) + ORTHO_DIRS:
                    out[r + dr, c + dc] = TARGET
    return out


def _legend_locked_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    template = _extract_corner_template(g, bg)
    if template is None:
        return out
    target_color, arms = template

    colors = sorted(set(map(int, g.flatten())) - {bg, target_color})
    H, W = g.shape
    for fg in colors:
        for r in range(1, H - 1):
            for c in range(1, W - 1):
                matched = _matching_t_arms(g, r, c, int(fg), (frozenset(arms),))
                if matched is None:
                    continue
                out[r, c] = int(target_color)
                for dr, dc in matched:
                    out[r + dr, c + dc] = int(target_color)
    return out


def _make_reserved_mask(H: int, W: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[: min(RESERVED, H), : min(RESERVED, W)] = True
    return mask


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(color)


def _component_ok(grid: np.ndarray, cells: set[tuple[int, int]], reserved: np.ndarray, bg: int) -> bool:
    H, W = grid.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if reserved[r, c] or int(grid[r, c]) != int(bg):
            return False
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in cells and int(grid[nr, nc]) != int(bg):
                return False
    return True


def _try_place_component(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    color: int,
    reserved: np.ndarray,
    bg: int,
) -> bool:
    if not _component_ok(grid, cells, reserved, bg):
        return False
    _paint_cells(grid, cells, color)
    return True


def _t_core(center: tuple[int, int], missing: tuple[int, int]) -> set[tuple[int, int]]:
    cr, cc = center
    cells = {(int(cr), int(cc))}
    for dr, dc in ORTHO_DIRS:
        if (dr, dc) != missing:
            cells.add((int(cr + dr), int(cc + dc)))
    return cells


def _build_target_component(
    rng: np.random.Generator,
    center: tuple[int, int],
    missing: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    core = _t_core(center, missing)
    cells = set(core)
    present = [direction for direction in ORTHO_DIRS if direction != missing]
    n_ext = int(rng.integers(1, min(3, len(present) + 1)))
    chosen_ids = rng.choice(len(present), size=n_ext, replace=False)
    for idx in chosen_ids:
        dr, dc = present[int(idx)]
        ext_len = int(rng.integers(1, 4))
        for step in range(2, ext_len + 2):
            cells.add((int(center[0] + dr * step), int(center[1] + dc * step)))
    return cells, core


def _build_plus_component(
    rng: np.random.Generator,
    center: tuple[int, int],
) -> set[tuple[int, int]]:
    cr, cc = center
    cells = {(int(cr), int(cc))}
    for dr, dc in ORTHO_DIRS:
        cells.add((int(cr + dr), int(cc + dc)))
    if float(rng.random()) < 0.75:
        dr, dc = ORTHO_DIRS[int(rng.integers(0, 4))]
        ext_len = int(rng.integers(1, 3))
        for step in range(2, ext_len + 2):
            cells.add((int(cr + dr * step), int(cc + dc * step)))
    return cells


def _paint_corner_legend(grid: np.ndarray, bg: int, color: int, missing: tuple[int, int]) -> None:
    grid[: min(RESERVED, grid.shape[0]), : min(RESERVED, grid.shape[1])] = int(bg)
    legend_cells = _t_core((1, 1), missing)
    _paint_cells(grid, legend_cells, color)


def _place_random_shape(
    rng: np.random.Generator,
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    color: int,
    reserved: np.ndarray,
    bg: int,
    attempts: int = 120,
) -> bool:
    H, W = grid.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    if H <= shape_h + 1 or W <= shape_w + 1:
        return False
    for _ in range(attempts):
        top = int(rng.integers(1, H - shape_h))
        left = int(rng.integers(1, W - shape_w))
        cells = {(int(top + r), int(left + c)) for r, c in shape}
        if _try_place_component(grid, cells, color, reserved, bg):
            return True
    return False


def _find_t_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int, tuple[int, int]]]:
    H, W = grid.shape
    found: list[tuple[int, int, tuple[int, int]]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            arms = tuple(sorted((dr, dc) for dr, dc in ORTHO_DIRS if int(grid[r + dr, c + dc]) == int(fg)))
            if len(arms) != 3:
                continue
            found.append((int(r), int(c), _missing_dir_from_arms(arms)))
    return found


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    found: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO_DIRS):
                found.append((int(r), int(c)))
    return found


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))
    grid = np.full((H, W), BG, dtype=int)
    reserved = _make_reserved_mask(H, W)

    legend_missing = ORTHO_DIRS[int(rng.integers(0, 4))]
    _paint_corner_legend(grid, BG, TARGET, legend_missing)

    area = H * W
    max_targets = min(7, max(2, area // 42))
    n_targets = int(rng.integers(2, max_targets + 1))

    target_orientations = [direction for direction in ORTHO_DIRS if direction != legend_missing]
    rng.shuffle(target_orientations)
    target_orientations.append(legend_missing)
    while len(target_orientations) < n_targets:
        target_orientations.append(ORTHO_DIRS[int(rng.integers(0, 4))])
    rng.shuffle(target_orientations)
    target_orientations = target_orientations[:n_targets]
    if all(direction == legend_missing for direction in target_orientations):
        target_orientations[0] = [direction for direction in ORTHO_DIRS if direction != legend_missing][0]

    planned_targets: list[tuple[int, int, tuple[int, int]]] = []
    target_cores: list[set[tuple[int, int]]] = []
    for missing in target_orientations:
        placed = False
        for _ in range(220):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            cells, core = _build_target_component(rng, center, missing)
            if _try_place_component(grid, cells, FG, reserved, BG):
                planned_targets.append((int(center[0]), int(center[1]), missing))
                target_cores.append(core)
                placed = True
                break
        if not placed:
            return None

    n_plus = int(rng.integers(1, min(4, max(2, area // 85)) + 1))
    plus_centers: list[tuple[int, int]] = []
    for _ in range(n_plus):
        placed = False
        for _ in range(180):
            center = (int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2)))
            cells = _build_plus_component(rng, center)
            if _try_place_component(grid, cells, FG, reserved, BG):
                plus_centers.append(center)
                placed = True
                break
        if not placed and not plus_centers:
            return None

    target_density = float(rng.uniform(0.18, 0.30))
    target_fg = int(area * target_density)
    current_fg = int(np.count_nonzero(grid == FG))
    distractor_placed = 0
    for _ in range(area * 8):
        if current_fg >= target_fg:
            break
        shape = DISTRACTOR_LIBRARY[int(rng.integers(0, len(DISTRACTOR_LIBRARY)))]
        if _place_random_shape(rng, grid, shape, FG, reserved, BG, attempts=20):
            current_fg = int(np.count_nonzero(grid == FG))
            distractor_placed += 1

    if distractor_placed == 0:
        return None
    if int(np.count_nonzero(grid == BG)) <= area // 2:
        return None

    actual_t = sorted(_find_t_centers(grid, FG))
    expected_t = sorted(planned_targets)
    if actual_t != expected_t:
        return None

    actual_plus = _find_plus_centers(grid, FG)
    if len(actual_plus) < len(plus_centers) or not actual_plus:
        return None

    if len({missing for _, _, missing in planned_targets}) < 2:
        return None
    if not any(missing != legend_missing for _, _, missing in planned_targets):
        return None

    expected = grid.copy()
    for core in target_cores:
        _paint_cells(expected, core, TARGET)

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    if np.array_equal(_legacy_plus_main(grid), expected):
        return None
    if np.array_equal(_legend_locked_main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
