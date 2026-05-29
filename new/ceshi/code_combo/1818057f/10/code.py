# Combo variant of 1818057f (index 10)
# combo_spec: D2/2+D4/2
# Name: 多背景分区+密度感知着色

from __future__ import annotations

from collections import Counter

import numpy as np

# Design analysis
# a) Concrete behavior and trace:
#    - The detected object stays the original one: any foreground cell whose
#      four orthogonal neighbors are also foreground is a '+' center.
#    - The D4 part changes the canvas: instead of one global background color,
#      the grid is split into 2 or 3 wide vertical bands with different colors.
#      The shared foreground stays yellow.
#    - The D2 part changes recoloring: after a valid '+' is found, inspect the
#      four diagonal cells in its 3x3 neighborhood. Those four cells are exactly
#      the 8-neighborhood remainder once the 5 cells of the plus are removed.
#      If more than half of them are foreground (3 of 4 in generated samples),
#      recolor the whole plus teal; otherwise recolor it magenta.
#    - Example trace:
#        seed task: yellow plus on red -> always teal.
#        fused task: yellow plus inside a blue band with 3 occupied diagonals
#        -> teal; another yellow plus inside that same blue band with 1 occupied
#        diagonal -> magenta. The background partition is visible, but the color
#        branch is still driven only by local foreground density.
# b) Complexity calibration:
#    - The partition vocabulary is intentionally small: only 2 or 3 repeated
#      vertical bands from a fixed palette. This makes the spatial structure
#      obvious without introducing a separate region->color lookup puzzle.
#    - The density branch has only two outcomes, teal vs magenta, and every
#      sample contains both. At least one region also contains both colors, so
#      the task cannot visually collapse into "each region maps to one color".
# c) Implicit-assumption review of the original main():
#    - `get_bg()` is no longer valid because there is no single global background.
#    - "take the only non-background color as foreground" is also invalid because
#      several non-foreground colors are now legitimate background bands.
#    - Writing a fixed teal output for every plus is invalid because the output
#      color now depends on local diagonal density.
#    - Reading density from `out` would be wrong because earlier recolors could
#      contaminate later counts. The new solver reads density only from `g`.
# d) Placement legality:
#    - Every planned plus center is at least two cells from the outer border and
#      at least two columns from its band's edges, so its full 5x5 protected zone
#      stays inside both the grid and a single background band.
#    - Centers are spaced by Chebyshev distance >= 5, preventing overlaps between
#      protected zones and avoiding accidental interference between local cues.
#    - Each plus gets exactly one extra arm cell so it remains embedded in a
#      larger component, matching the seed task's "not isolated" behavior.
# e) Visual observability:
#    - A human can first notice the large background bands, then notice that
#      identical embedded pluses inside the same band can still split into two
#      output colors. That points away from region identity and toward a local
#      neighborhood test.
#    - The decisive cue is visible: count the four corner cells around each plus.
#      Dense corners -> teal; sparse corners -> magenta.
# f) Visual cleanliness:
#    - Noise is added only outside each plus's 5x5 protected zone, so the local
#      diagonal-density signal remains readable.
#    - Foreground density stays below half the grid, preserving the background
#      bands as visible organizational structure instead of turning the grid into
#      an undifferentiated foreground mass.
# g) New main() assumptions and generator guarantees:
#    - main() reconstructs the foreground by asking which candidate color, when
#      ignored, makes columns collapse back into 2 or 3 clean vertical bands.
#    - generate_input() guarantees that this signal is stable by using only wide
#      vertical bands and by keeping foreground density below half the grid.
#    - The scan over interior cells assumes plus centers are not on the border;
#      generate_input() places them with a two-cell margin.

FG = 4
BG_COLORS = (1, 2, 5)
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PLUS = [(0, 0), *ORTHO]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
EXT_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _band_profile_if_fg(grid: np.ndarray, fg: int) -> tuple[float, np.ndarray] | None:
    H, W = grid.shape
    col_bg = np.full(W, -1, dtype=int)
    total_major = 0
    total_support = 0

    for c in range(W):
        vals = [int(grid[r, c]) for r in range(H) if int(grid[r, c]) != fg]
        if not vals:
            return None
        bg, major = Counter(vals).most_common(1)[0]
        col_bg[c] = int(bg)
        total_major += int(major)
        total_support += len(vals)

    runs = 1
    min_run = W
    run_len = 1
    for c in range(1, W):
        if int(col_bg[c]) == int(col_bg[c - 1]):
            run_len += 1
        else:
            runs += 1
            min_run = min(min_run, run_len)
            run_len = 1
    min_run = min(min_run, run_len)

    if runs not in (2, 3) or min_run < 5:
        return None
    return total_major / total_support, col_bg


def _infer_foreground(grid: np.ndarray) -> int | None:
    colors = sorted(set(map(int, grid.flatten())))
    best_fg = None
    best_score = -1.0

    for color in colors:
        profile = _band_profile_if_fg(grid, int(color))
        if profile is None:
            continue
        score, _col_bg = profile
        if score > best_score or (score == best_score and int(color) == FG):
            best_score = score
            best_fg = int(color)

    return best_fg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs in a shared foreground color laid over several background bands, and recolor them by local foreground density.

    Input structure:
    - The grid background is partitioned into 2 or 3 wide vertical color bands instead of a single global bg.
    - One shared foreground color forms mixed shapes on top of those bands, including embedded 5-cell '+' motifs.
    - Around each '+' center, the four diagonal corner cells may be sparse or crowded with foreground, independent of the background partition.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - For each such center, count how many of the 4 diagonal cells in its 3x3 neighborhood are foreground in the input.
    - If that diagonal density is greater than 50% (3 or 4 of the 4 corners), recolor the 5 plus cells teal (8); otherwise recolor them magenta (6).
    - Leave all non-plus pixels unchanged, and do not let the background bands affect the density count.

    Key insight:
    - The background partition matters only for recovering which color is foreground and for giving the eye spatial structure; the actual recolor branch is determined solely by the local corner density around each detected plus.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()

    fg = _infer_foreground(g)
    if fg is None:
        return out

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                diag_fg = sum(int(g[r + dr, c + dc]) == fg for dr, dc in DIAGS)
                target = 8 if diag_fg > 2 else 6
                for dr, dc in PLUS:
                    out[r + dr, c + dc] = target
    return out


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


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    for cr, cc in [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        ok = True
        for dr, dc in PLUS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                ok = False
                break
        if ok:
            return True
    return False


def _diag_subsets_for_extension(dir_idx: int) -> dict[int, list[tuple[int, ...]]]:
    blocked_pairs = {
        0: {0, 1},  # north extension would create a new plus if NW and NE are both set
        1: {2, 3},  # south extension blocks SW+SE together
        2: {0, 2},  # west extension blocks NW+SW together
        3: {1, 3},  # east extension blocks NE+SE together
    }
    blocked = blocked_pairs[dir_idx]
    by_count: dict[int, list[tuple[int, ...]]] = {0: [tuple()], 1: [], 2: [], 3: []}
    for mask in range(1, 1 << 4):
        subset = tuple(i for i in range(4) if mask & (1 << i))
        if len(blocked & set(subset)) == 2:
            continue
        if len(subset) <= 3:
            by_count[len(subset)].append(subset)
    return by_count


def _paint_plus_region(
    grid: np.ndarray,
    center: tuple[int, int],
    dir_idx: int,
    diag_subset: tuple[int, ...],
    fg: int,
) -> None:
    r, c = center
    for dr, dc in PLUS:
        grid[r + dr, c + dc] = fg

    er, ec = EXT_DIRS[dir_idx]
    grid[r + 2 * er, c + 2 * ec] = fg

    for idx in diag_subset:
        dr, dc = DIAGS[idx]
        grid[r + dr, c + dc] = fg


def _add_protected_zone(protected: set[tuple[int, int]], shape: tuple[int, int], center: tuple[int, int]) -> None:
    H, W = shape
    r, c = center
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                protected.add((nr, nc))


def _far_enough(centers: list[tuple[int, int]], candidate: tuple[int, int]) -> bool:
    r, c = candidate
    return all(max(abs(r - rr), abs(c - cc)) >= 5 for rr, cc in centers)


def _build_regions(rng: np.random.Generator, width: int, n_regions: int) -> list[tuple[int, int, int]]:
    widths = [5] * n_regions
    for _ in range(width - 5 * n_regions):
        widths[int(rng.integers(0, n_regions))] += 1

    colors = list(map(int, rng.choice(BG_COLORS, size=n_regions, replace=False)))
    regions = []
    c0 = 0
    for band_w, bg in zip(widths, colors):
        c1 = c0 + int(band_w) - 1
        regions.append((c0, c1, bg))
        c0 = c1 + 1
    return regions


def _region_candidates(H: int, W: int, c0: int, c1: int) -> list[tuple[int, int]]:
    low_c = max(c0 + 2, 2)
    high_c = min(c1 - 2, W - 3)
    if low_c > high_c:
        return []
    return [(r, c) for r in range(2, H - 2) for c in range(low_c, high_c + 1)]


def _pick_center(
    candidates: list[tuple[int, int]],
    centers: list[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    order = candidates.copy()
    rng.shuffle(order)
    for candidate in order:
        if _far_enough(centers, candidate):
            return candidate
    return None


def _find_region(regions: list[tuple[int, int, int]], c: int) -> tuple[int, int, int]:
    for region in regions:
        c0, c1, _ = region
        if c0 <= c <= c1:
            return region
    raise ValueError(f"column {c} is outside every region")


def _try_add_line_in_region(
    grid: np.ndarray,
    fg: int,
    protected: set[tuple[int, int]],
    rng: np.random.Generator,
    regions: list[tuple[int, int, int]],
) -> int:
    H, W = grid.shape
    c0, c1, _ = regions[int(rng.integers(0, len(regions)))]
    r = int(rng.integers(0, H))
    c = int(rng.integers(c0, c1 + 1))
    dr, dc = EXT_DIRS[int(rng.integers(0, 4))]
    length = int(rng.integers(1, 5))

    added = 0
    for step in range(length):
        nr, nc = r + dr * step, c + dc * step
        if not (0 <= nr < H and 0 <= nc < W):
            break
        if nc < c0 or nc > c1:
            break
        if (nr, nc) in protected or int(grid[nr, nc]) == fg:
            continue
        if _would_create_plus(grid, nr, nc, fg):
            continue
        grid[nr, nc] = fg
        added += 1
    return added


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    n_regions = 2 if rng.random() < 0.5 else 3
    min_width = 10 if n_regions == 2 else 15
    W = int(rng.integers(min_width, 23))
    total = H * W

    regions = _build_regions(rng, W, n_regions)
    region_candidates = {idx: _region_candidates(H, W, c0, c1) for idx, (c0, c1, _) in enumerate(regions)}
    if any(not candidates for candidates in region_candidates.values()):
        return None

    anchor_choices = [idx for idx, candidates in region_candidates.items() if len(candidates) >= 2]
    if not anchor_choices:
        return None
    anchor_idx = int(rng.choice(anchor_choices))

    grid = np.zeros((H, W), dtype=int)
    for c0, c1, bg in regions:
        grid[:, c0 : c1 + 1] = bg

    centers: list[tuple[int, int]] = []
    planned_colors: dict[tuple[int, int], int] = {}

    anchor_first = _pick_center(region_candidates[anchor_idx], centers, rng)
    if anchor_first is None:
        return None
    centers.append(anchor_first)
    planned_colors[anchor_first] = 8

    anchor_second = _pick_center(region_candidates[anchor_idx], centers, rng)
    if anchor_second is None:
        return None
    centers.append(anchor_second)
    planned_colors[anchor_second] = 6

    for idx, _region in enumerate(regions):
        if idx == anchor_idx:
            continue
        center = _pick_center(region_candidates[idx], centers, rng)
        if center is None:
            return None
        centers.append(center)
        planned_colors[center] = int(rng.choice([6, 8]))

    all_candidates: list[tuple[int, int]] = []
    for candidates in region_candidates.values():
        all_candidates.extend(candidates)
    rng.shuffle(all_candidates)

    max_k = min(9, len(all_candidates), max(len(centers), total // 38))
    target_k = int(rng.integers(len(centers), max_k + 1))
    for candidate in all_candidates:
        if len(centers) >= target_k:
            break
        if candidate in planned_colors:
            continue
        if not _far_enough(centers, candidate):
            continue
        centers.append(candidate)
        planned_colors[candidate] = int(rng.choice([6, 8]))

    protected: set[tuple[int, int]] = set()
    for center in centers:
        plus_color = planned_colors[center]
        dir_idx = int(rng.integers(0, 4))
        subsets = _diag_subsets_for_extension(dir_idx)
        if plus_color == 8:
            diag_subset = subsets[3][int(rng.integers(0, len(subsets[3])))]
        else:
            sparse_count = int(rng.integers(0, 3))
            diag_subset = subsets[sparse_count][int(rng.integers(0, len(subsets[sparse_count])))]
        _paint_plus_region(grid, center, dir_idx, diag_subset, FG)
        _add_protected_zone(protected, grid.shape, center)

    actual_centers = set(_find_plus_centers(grid, FG))
    if actual_centers != set(centers):
        return None

    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid == FG))

    attempts = 0
    max_attempts = total * 25
    while current_fg < target_fg and attempts < max_attempts:
        current_fg += _try_add_line_in_region(grid, FG, protected, rng, regions)
        attempts += 1

    current_fg = int(np.sum(grid == FG))
    if current_fg >= total // 2:
        return None

    actual_centers = set(_find_plus_centers(grid, FG))
    if actual_centers != set(centers):
        return None

    expected = grid.copy()
    region_to_targets: dict[int, set[int]] = {idx: set() for idx in range(len(regions))}
    for r, c in centers:
        target = planned_colors[(r, c)]
        region_idx = next(idx for idx, (c0, c1, _) in enumerate(regions) if c0 <= c <= c1)
        region_to_targets[region_idx].add(target)
        for dr, dc in PLUS:
            expected[r + dr, c + dc] = target

    if 6 not in planned_colors.values() or 8 not in planned_colors.values():
        return None
    if not any(targets == {6, 8} for targets in region_to_targets.values()):
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
