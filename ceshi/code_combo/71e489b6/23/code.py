# Combo variant of 71e489b6 (index 23)
# combo_spec: D2/2+D3/0+D9/1
# Name: 泛化异常+毛刺品红+多色干扰

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_ROOT = ROOT / "solvers"
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from arc_utils import components_mask


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
MAGENTA = 6
ORANGE = 7
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ANOMALY_COLORS = (BLACK, RED, GRAY)
ANOMALY_COLOR_WEIGHTS = (5, 2, 2)
NOISE_COLORS = (RED, GRAY)

# Design analysis for combo D2/2 + D3/0 + D9/1
# a) Concrete behavior trace:
#    - Original pipeline:
#      1. blue spur pixels inside black are absorbed back to black,
#      2. tiny black components inside blue and leaf pixels of large black components
#         get orange halos,
#      3. input is binary so there is no extra color branch at all.
#    - Fused pipeline:
#      1. any non-blue component that is fully enclosed by blue is now an orange target,
#         so black / red / gray anomalies inside blue all behave the same structurally;
#      2. blue spur pixels inside black are no longer absorbed in the rendered output,
#         but are still treated as black in a hidden analysis grid so they do not break
#         black-component leaf detection;
#      3. non-blue, non-black pixels inside black are pure interference and are erased to
#         black before analysis and in the final output.
#    - Small trace:
#          input                         output
#          1 1 1 1 1                     1 7 7 7 1
#          1 2 1 0 0                     7 2 7 6 0
#          1 1 1 1 0                     1 7 7 0 0
#      The red cell inside blue gets an orange halo, the blue spur in black turns magenta,
#      and any red/gray cell placed in black would simply disappear to black.
# b) Complexity calibration:
#    - The palette stays compact: only two extra input colors (red, gray) and two output
#      markers (orange, magenta). The same extra colors are reused in both blue-region
#      anomalies and black-region noise so the human-visible rule must come from context,
#      not from a color lookup table.
#    - Generator always includes at least one colored anomaly inside blue, at least two
#      blue spurs inside black, and at least two black-region noise pixels using both
#      extra colors. That prevents silent collapse to the source task or to a one-branch
#      variant.
# c) Hidden-assumption audit against the original main():
#    - Original STEP 0 assumed every qualifying blue pixel should be rewritten to black.
#      Broken by D3/0: those pixels must survive as visible magenta output marks.
#    - Original defect collection assumed all relevant targets are black components.
#      Broken by D2/2: red/gray anomalies inside blue must also be haloed.
#    - Original binary-color assumption is broken by D9/1. Extra colors cannot simply be
#      treated as foreground or background globally; they need contextual routing:
#      blue-enclosed extra colors are preserved and haloed, black-enclosed extra colors
#      are deleted.
#    - New solver assumption: colored anomalies inside blue appear as small enclosed
#      components, while colored interference inside black appears as isolated black-region
#      pixels. The generator guarantees exactly that construction.
# d) Placement legality:
#    - Blue anomalies are placed only on strict blue interiors, with Chebyshev clearance,
#      so they never touch each other or merge with large black regions.
#    - Leaf pixels are added only on blue cells with exactly one black 4-neighbor from a
#      large black component.
#    - Blue spur candidates are sampled only from black cells satisfying the same local
#      predicate that the solver later uses for magenta marking.
#    - Colored black-region noise is placed only on interior black cells, away from halo
#      zones and away from spur placements.
# e) Visual observability:
#    - Across examples, the user repeatedly sees the same two extra colors behave in two
#      opposite ways depending on region context: inside blue they persist and get orange;
#      inside black they vanish. Blue-in-black becomes magenta instead of vanishing.
#    - This makes each branch independently visible from 2-4 samples without relying on
#      color ids or scan order.
# f) Visual cleanliness:
#    - The generator keeps a safety buffer around orange targets when placing spurs/noise,
#      so magenta dots and deleted noise do not sit directly inside orange rings.
#    - Leaves remain sparse and black-region noise is singleton-only, preventing the output
#      from becoming a dense multicolor carpet.
# g) Change inferability:
#    - The rule is human-readable as three local grammars:
#      non-blue in blue => orange halo, blue in black => magenta point, other colors in
#      black => erased. The same red/gray colors appear on both sides of the boundary, so
#      the solver cannot be inferred as "red means halo" or "gray means delete" alone.
# h) Quantitative non-degeneracy metric:
#    - For each generated sample we track:
#      `colored_blue_targets` = number of orange-target cells whose center is red/gray,
#      `spur_count` = number of magenta cells,
#      `noise_color_count` = how many distinct extra colors appear only as black-region
#      noise, and `baseline_diff` = pixel difference from an original-like binary solver.
#      Accepted samples require all four signals to be nonzero / active.
# i) New solver assumptions and generator guarantees:
#    - `_collect_blue_enclosed_targets()` assumes meaningful blue-region anomalies are
#      monochrome components individually enclosed by blue. The generator places them that
#      way.
#    - `_find_spur_marks()` assumes black-region noise has already been normalized to black.
#      The solver does that before spur detection, and the generator only puts extra colors
#      in black where that normalization is safe.
#    - `_collect_black_leaf_targets()` assumes spur pixels should count as black for shape
#      analysis even though they render magenta. The solver uses a hidden analysis grid,
#      and the generator avoids placing spurs near leaves so this interpretation stays clean.


GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
PARTITION_STYLE_WEIGHTS = [6, 1, 3]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
INNER_INSET_MIN_GAP = 6
INNER_INSET_PROB = 0.5
ANOMALY_COUNT_RANGE = (3, 8)
ANOMALY_SIZE_WEIGHTS = [6, 2, 1]
ANOMALY_CLEARANCE = 2
ANOMALY_PLACE_TRIES = 60
LEAF_COUNT_RANGE = (0, 4)
SPUR_COUNT_RANGE = (2, 6)
NOISE_COUNT_RANGE = (2, 5)
MIN_ANOMALIES = 3
MIN_SPURS = 2
MIN_NOISE = 2
SAFETY_MARGIN = 2
MAX_GEN_ATTEMPTS = 450


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(
    pixels: set[tuple[int, int]] | list[tuple[int, int]],
    dist: int,
) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _effective_black_neighbors(grid: np.ndarray, r: int, c: int) -> int:
    H, W = grid.shape
    black_n = 0
    oob = 0
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if not (0 <= rr < H and 0 <= cc < W):
            oob += 1
            continue
        if int(grid[rr, cc]) == BLACK:
            black_n += 1
    if oob >= 2:
        black_n += oob
    return black_n


def _collect_blue_enclosed_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Cells of monochrome non-blue components fully surrounded by blue."""
    H, W = grid.shape
    targets: set[tuple[int, int]] = set()
    colors = sorted(set(map(int, grid.flatten())) - {BLUE})
    for color in colors:
        for comp in components_mask(grid == color, connectivity=4):
            pts = [(int(r), int(c)) for r, c in comp]
            comp_set = set(pts)
            enclosed = True
            for r, c in pts:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if not (0 <= rr < H and 0 <= cc < W):
                        enclosed = False
                        break
                    if (rr, cc) in comp_set:
                        continue
                    if int(grid[rr, cc]) != BLUE:
                        enclosed = False
                        break
                if not enclosed:
                    break
            if enclosed:
                targets.update(comp_set)
    return targets


def _normalize_black_region_noise(
    grid: np.ndarray,
    blue_targets: set[tuple[int, int]],
) -> np.ndarray:
    """Erase extra colors in black regions, but keep colored blue-region anomalies."""
    work = np.array(grid, dtype=int)
    H, W = work.shape
    for r in range(H):
        for c in range(W):
            color = int(work[r, c])
            if color in (BLACK, BLUE):
                continue
            if (r, c) not in blue_targets:
                work[r, c] = BLACK
    return work


def _find_spur_marks(grid: np.ndarray) -> set[tuple[int, int]]:
    """Blue cells in black territory that should render as magenta."""
    H, W = grid.shape
    marks: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLUE:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                marks.add((r, c))
    return marks


def _analysis_grid(
    grid: np.ndarray,
    spur_marks: set[tuple[int, int]],
) -> np.ndarray:
    """Hidden structural grid: black-region spurs count as black for leaf analysis."""
    analysis = np.array(grid, dtype=int)
    for r, c in spur_marks:
        analysis[r, c] = BLACK
    return analysis


def _collect_black_leaf_targets(grid: np.ndarray) -> set[tuple[int, int]]:
    """Leaf pixels of large black components, matching the source task logic."""
    H, W = grid.shape
    leaves: set[tuple[int, int]] = set()
    for comp in components_mask(grid == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            continue
        comp_set = set(pts)
        for r, c in pts:
            degree = 0
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    degree += 1
            if degree <= 1:
                leaves.add((r, c))
    return leaves


def _analyze_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Return the render base, magenta spur marks, and orange-halo target centers.
    """
    g = np.array(grid, dtype=int)
    blue_targets = _collect_blue_enclosed_targets(g)
    work = _normalize_black_region_noise(g, blue_targets)
    spur_marks = _find_spur_marks(work)
    analysis = _analysis_grid(work, spur_marks)
    orange_targets = set(blue_targets)
    orange_targets |= _collect_black_leaf_targets(analysis)
    return work, spur_marks, orange_targets


def _render_output(
    base: np.ndarray,
    orange_targets: set[tuple[int, int]],
    spur_marks: set[tuple[int, int]],
) -> np.ndarray:
    """Paint orange halos and magenta spur centers on the cleaned render base."""
    out = np.array(base, dtype=int)
    H, W = out.shape
    for r, c in orange_targets:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in orange_targets:
                    out[rr, cc] = ORANGE
    for r, c in orange_targets:
        out[r, c] = int(base[r, c])
    for r, c in spur_marks:
        out[r, c] = MAGENTA
    return out


def _absorb_blue_spurs_binary(grid: np.ndarray) -> np.ndarray:
    """Original-like absorption used only for degeneration checks."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            if _effective_black_neighbors(base, r, c) >= 3:
                base[r, c] = BLACK
    return base


def _baseline_originalish(grid: np.ndarray) -> np.ndarray:
    """Binaryized source-task behavior, for non-degeneracy measurement."""
    binary = np.array(grid, dtype=int)
    binary[(binary != BLACK) & (binary != BLUE)] = BLACK
    base = _absorb_blue_spurs_binary(binary)
    preserve: set[tuple[int, int]] = set()
    H, W = base.shape
    for comp in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in comp]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        comp_set = set(pts)
        for r, c in pts:
            degree = 0
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) in comp_set:
                    degree += 1
            if degree <= 1:
                preserve.add((r, c))
    out = np.array(base, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Quantify whether all fused behaviors are active on a generated sample.
    """
    inp = np.array(grid, dtype=int)
    work, spur_marks, orange_targets = _analyze_scene(inp)
    colored_blue_targets = sum(
        1 for r, c in orange_targets if int(work[r, c]) not in (BLACK, BLUE)
    )
    noise_pixels = [
        (r, c)
        for r in range(inp.shape[0])
        for c in range(inp.shape[1])
        if int(inp[r, c]) not in (BLACK, BLUE) and (r, c) not in orange_targets
    ]
    noise_color_count = len({int(inp[r, c]) for r, c in noise_pixels})
    combo_out = _render_output(work, orange_targets, spur_marks)
    baseline_diff = int(np.count_nonzero(combo_out != _baseline_originalish(inp)))
    return {
        "colored_blue_targets": int(colored_blue_targets),
        "spur_count": int(len(spur_marks)),
        "noise_count": int(len(noise_pixels)),
        "noise_color_count": int(noise_color_count),
        "baseline_diff": int(baseline_diff),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark every blue-region anomaly with orange, preserve blue spurs in black as magenta, and erase other black-region color noise.

    Input structure:
    - Large blue (1) and black (0) regions partition the board, as in the source task.
    - Blue regions may contain tiny enclosed anomalies of several colors: black, red (2), or gray (5); large black regions may also contribute 1-cell leaf tips into blue.
    - Black regions may contain two nuisance types: blue spur pixels and extra red/gray interference pixels.

    Rule:
    - First identify every monochrome non-blue component that is completely enclosed by blue; those cells are orange-halo targets and keep their own center colors.
    - Then erase every extra red/gray pixel that is not such a blue-enclosed anomaly, so black-region color noise collapses to black. Blue pixels in black with at least three effective black 4-neighbors are recorded as spur marks and treated as black only in a hidden analysis grid.
    - Finally add black leaf pixels from large black components to the orange-target set, paint orange (7) in the Chebyshev-1 halo of every target, preserve target centers, and render each recorded spur center as magenta (6).

    Key insight: The transformation is context-driven rather than color-driven. The same extra colors survive with orange only when trapped inside blue, but disappear when trapped inside black; meanwhile the old removable blue spur is no longer absorbed and becomes a magenta marker.
    """
    base, spur_marks, orange_targets = _analyze_scene(grid)
    return _render_output(base, orange_targets, spur_marks)


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a blue/black partition using the same visual family as the source task."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choices(PARTITION_STYLES, weights=PARTITION_STYLE_WEIGHTS, k=1)[0]

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_count = min(n_cuts, H - 2 * CUT_MARGIN)
        cuts = sorted(random.sample(range(CUT_MARGIN, H - CUT_MARGIN), cut_count))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for i in range(len(boundaries) - 1):
            grid[boundaries[i] : boundaries[i + 1], :] = (first_color + i) % 2

    elif style == "rect_inset":
        # Keep the outer field black for this combo so the base partition does not
        # accidentally create a giant black component fully enclosed by blue, which
        # the fused D2 rule would correctly treat as an anomaly.
        bg = BLACK
        grid[:, :] = bg
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = 1 - bg

    else:
        bg = random.randint(0, 1)
        grid[:, :] = bg
        vc = random.randint(W // 3, 2 * W // 3)
        if random.random() < 0.5:
            grid[:, :vc] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[:hc, vc:] = 1 - bg
        else:
            grid[:, vc:] = 1 - bg
            hc = random.randint(H // 3, 2 * H // 3)
            grid[hc:, :vc] = 1 - bg

    return grid


def _find_interior_blue(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue cells whose in-bounds 4-neighbors are all blue."""
    blue = grid == BLUE
    eroded = blue.copy()
    eroded[:-1, :] &= blue[1:, :]
    eroded[1:, :] &= blue[:-1, :]
    eroded[:, :-1] &= blue[:, 1:]
    eroded[:, 1:] &= blue[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _place_blue_anomalies(
    grid: np.ndarray,
    interior_blue: list[tuple[int, int]],
) -> tuple[set[tuple[int, int]], int, int]:
    """Place 3-8 small monochrome anomalies inside blue regions."""
    H, W = grid.shape
    n_anomalies = random.randint(*ANOMALY_COUNT_RANGE)
    planned_colors = [random.choice(NOISE_COLORS)]
    for _ in range(max(0, n_anomalies - 1)):
        planned_colors.append(random.choices(ANOMALY_COLORS, weights=ANOMALY_COLOR_WEIGHTS)[0])
    random.shuffle(planned_colors)

    anomaly_pixels: set[tuple[int, int]] = set()
    interior_set = set(interior_blue)
    placed = 0
    colored_cells = 0

    for color in planned_colors:
        if not interior_blue:
            break
        size = random.choices([1, 2, 3], weights=ANOMALY_SIZE_WEIGHTS)[0]
        for _ in range(ANOMALY_PLACE_TRIES):
            if not interior_blue:
                break
            seed_r, seed_c = interior_blue[random.randint(0, len(interior_blue) - 1)]
            if _nearby(seed_r, seed_c, ANOMALY_CLEARANCE, anomaly_pixels):
                continue

            comp = [(seed_r, seed_c)]
            if size > 1:
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in anomaly_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in DIR4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if int(grid[rr, cc]) != BLUE or (rr, cc) in anomaly_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, ANOMALY_CLEARANCE, anomaly_pixels):
                        comp.append((nr, nc))
                        if len(comp) >= size:
                            break

            comp_set = set(comp)
            valid = True
            for r, c in comp:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in comp_set:
                        continue
                    if 0 <= rr < H and 0 <= cc < W:
                        if int(grid[rr, cc]) != BLUE or (rr, cc) in anomaly_pixels:
                            valid = False
                            break
                if not valid:
                    break

            if not valid:
                continue

            for r, c in comp:
                grid[r, c] = color
            anomaly_pixels.update(comp_set)
            if color != BLACK:
                colored_cells += len(comp_set)
            placed += 1
            remove = _exclusion_zone(comp, ANOMALY_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            break

    return anomaly_pixels, placed, colored_cells


def _place_leaves(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Place a few black leaf tips from large black components into blue."""
    H, W = grid.shape
    n_leaves = random.randint(*LEAF_COUNT_RANGE)
    leaf_pixels: set[tuple[int, int]] = set()

    black_comps = components_mask(grid == BLACK, connectivity=4)
    large_black = [comp for comp in black_comps if len(comp) > 3]

    for _ in range(n_leaves):
        if not large_black:
            break
        comp_set = {(int(r), int(c)) for r, c in random.choice(large_black)}
        candidates: list[tuple[int, int]] = []
        for r, c in comp_set:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W) or int(grid[rr, cc]) != BLUE:
                    continue
                if (rr, cc) in blocked or (rr, cc) in leaf_pixels:
                    continue
                black_n = 0
                for dr2, dc2 in DIR4:
                    nr = rr + dr2
                    nc = cc + dc2
                    if 0 <= nr < H and 0 <= nc < W:
                        if int(grid[nr, nc]) == BLACK or (nr, nc) in leaf_pixels:
                            black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))

        if not candidates:
            continue
        lr, lc = random.choice(candidates)
        if _nearby(lr, lc, ANOMALY_CLEARANCE, blocked):
            continue
        leaf_pixels.add((lr, lc))
        grid[lr, lc] = BLACK

    return leaf_pixels


def _find_spur_candidates(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Black cells that can become blue spur pixels, including corner cases."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if _effective_black_neighbors(grid, r, c) >= 3:
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_spurs(
    clean_grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Inject input-only blue spur pixels inside black regions."""
    work = np.array(clean_grid, dtype=int)
    target = random.randint(*SPUR_COUNT_RANGE)
    spurs: set[tuple[int, int]] = set()
    blocked_zone = set(blocked)
    H, W = work.shape

    def is_boundary(cell: tuple[int, int]) -> bool:
        r, c = cell
        return r in (0, H - 1) or c in (0, W - 1)

    candidates = _find_spur_candidates(work, blocked_zone)
    boundary_candidates = [cell for cell in candidates if is_boundary(cell)]
    if boundary_candidates and random.random() < 0.5:
        r, c = boundary_candidates[0]
        work[r, c] = BLUE
        spurs.add((r, c))
        blocked_zone |= _exclusion_zone({(r, c)}, SAFETY_MARGIN)

    while len(spurs) < target:
        candidates = _find_spur_candidates(work, blocked_zone)
        if not candidates:
            break
        r, c = candidates[0]
        work[r, c] = BLUE
        spurs.add((r, c))
        blocked_zone |= _exclusion_zone({(r, c)}, SAFETY_MARGIN)

    return spurs


def _find_noise_candidates(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Interior black cells suitable for extra color interference pixels."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != BLACK or (r, c) in blocked:
                continue
            if all(int(grid[r + dr, c + dc]) == BLACK for dr, dc in DIR4):
                candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _place_colored_noise(
    clean_grid: np.ndarray,
    blocked: set[tuple[int, int]],
) -> dict[tuple[int, int], int]:
    """Inject black-region interference using both extra colors."""
    target = random.randint(*NOISE_COUNT_RANGE)
    candidates = _find_noise_candidates(clean_grid, blocked)
    if len(candidates) < max(MIN_NOISE, 2):
        return {}
    take = min(target, len(candidates))
    picks = candidates[:take]
    palette = [RED, GRAY]
    while len(palette) < len(picks):
        palette.append(random.choice(NOISE_COLORS))
    random.shuffle(palette)
    return {pos: palette[i] for i, pos in enumerate(picks)}


def _validate_regions(grid: np.ndarray) -> bool:
    blue_comps = list(components_mask(grid == BLUE, connectivity=4))
    black_comps = list(components_mask(grid == BLACK, connectivity=4))
    if not blue_comps or not black_comps:
        return False
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return False
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)
    if not _validate_regions(grid):
        return None

    interior_blue = _find_interior_blue(grid)
    anomaly_pixels, anomalies_placed, colored_cells = _place_blue_anomalies(grid, interior_blue)
    if anomalies_placed < MIN_ANOMALIES or colored_cells <= 0:
        return None

    leaf_pixels = _place_leaves(grid, anomaly_pixels)
    orange_targets = set(anomaly_pixels) | set(leaf_pixels)
    clean_grid = np.array(grid, dtype=int)

    blocked = _exclusion_zone(orange_targets, SAFETY_MARGIN)
    spurs = _place_spurs(clean_grid, blocked)
    if len(spurs) < MIN_SPURS:
        return None

    noise = _place_colored_noise(clean_grid, blocked | _exclusion_zone(spurs, SAFETY_MARGIN))
    if len(noise) < MIN_NOISE or len(set(noise.values())) < len(NOISE_COLORS):
        return None

    inp = np.array(clean_grid, dtype=int)
    for r, c in spurs:
        inp[r, c] = BLUE
    for (r, c), color in noise.items():
        inp[r, c] = color

    expected = _render_output(clean_grid, orange_targets, spurs)
    try:
        solver_out = main(inp)
    except Exception:
        return None
    if not np.array_equal(solver_out, expected):
        return None

    metrics = behavior_metrics(inp)
    if metrics["colored_blue_targets"] <= 0:
        return None
    if metrics["spur_count"] < MIN_SPURS:
        return None
    if metrics["noise_color_count"] < len(NOISE_COLORS):
        return None
    if metrics["baseline_diff"] <= 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
