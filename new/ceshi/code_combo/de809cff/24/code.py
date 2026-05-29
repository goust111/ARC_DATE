# Combo variant of de809cff (index 24)
# combo_spec: D11/0+D16/2
# Name: 输出裁剪为最小外接矩形+补丁镜像复制
#
# Design analysis for the fused variant:
# a) Behavior trace:
#    Original rule turns each cleaned black dent into one 3x3 opposite-color patch
#    with a teal center. The fused rule keeps that step, then point-reflects the
#    whole patch around the full-grid center, so one input dent now yields two
#    teal-marked 3x3 patches. Finally the output is cropped to the tight bbox of
#    all non-black pixels. Example: a dent at (r, c) in an HxW grid used to stamp
#    only around (r, c); now it also stamps around (H-1-r, W-1-c), and the black
#    outer frame disappears in the output.
# b) Complexity calibration:
#    The generator keeps only one new symmetry mode (global point reflection) and
#    only one crop mode (tight non-black bbox). To make the difference visible but
#    not chaotic, it uses a point-symmetric clean layout, 8-14 dent groups, and a
#    larger spacing budget across both original and mirrored centers.
# c) Hidden assumptions audited:
#    `_main_colors` still assumes exactly two meaningful foreground colors in the
#    input; noise never introduces a third stable fg color. `_clean_foreground`
#    still works because the corruption model is unchanged. `_dent_centers` still
#    reads only original black holes, so mirrored centers must not already be black;
#    the generator forbids selecting both a cell and its reflection. Mirrored patch
#    conflicts would make order matter, so the generator also enforces spacing for
#    reflected centers, not just original ones.
# d) Placement legality:
#    Clean regions are built fully inside a 2-4 cell black frame, dents are chosen
#    only where both the original and reflected 3x3 neighborhoods are valid host
#    neighborhoods, and reflected centers are checked for non-overlap. Pair dents
#    are rare and may overlap only within their own 2-cell group, as intended.
# e) Visual observability:
#    Across samples, humans can see two stable clues: output teal centers appear in
#    centrally symmetric pairs, and the output frame is tighter than the input frame.
#    No invisible ordering by numeric color IDs is used.
# f) Visual cleanliness:
#    The reflected layout is kept legible by a symmetric clean scaffold, a padded
#    black border, and Chebyshev spacing across all transformed centers. This avoids
#    ambiguous multi-patch pileups.
# g) Inferability of changes:
#    Color behavior stays close to the base task: dents still create opposite-color
#    3x3 patches with teal centers. The only extra inference is that every such
#    patch is duplicated by point reflection, then the finished scene is cropped.

from __future__ import annotations

from collections import Counter as _Counter
from pathlib import Path
import random
import sys

import numpy as np

try:
    from arc_utils import Color
except ImportError:  # pragma: no cover - fallback for direct file loading
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    Two sweeps are enough for this corruption model: the first clears isolated
    mistakes, and the second lets neighboring cells stabilize around the fixed
    local majority.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                sub = cur[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == bg))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = bg
                elif own <= 2:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    return centers


def _reflect_point(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    return int(h - 1 - r), int(w - 1 - c)


def _crop_foreground(grid: np.ndarray, bg: int) -> np.ndarray:
    fg = np.argwhere(grid != int(bg))
    if len(fg) == 0:
        return grid[0:0, 0:0]
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _apply_rule(clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int], bg: int) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    teal_points: set[tuple[int, int]] = set()

    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for rr, cc in ((int(r), int(c)), _reflect_point(h, w, int(r), int(c))):
            out[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)] = int(other)
            teal_points.add((int(rr), int(cc)))

    for r, c in teal_points:
        out[int(r), int(c)] = Color.TEAL

    return _crop_foreground(out, bg)


def _apply_base_rule(clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int], bg: int) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for r, c, _host in centers:
        out[int(r), int(c)] = Color.TEAL
    return _crop_foreground(out, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn cleaned black dents into paired opposite-color patches, then crop tightly.

    Input structure:
    - Black background with exactly two large foreground colors.
    - The foreground contains true black dents plus one-cell noise specks and
      boundary protrusions that should be cleaned before dent detection.

    Rule:
    - First restore the two foreground regions locally: isolated stray cells
      disappear into black, and one-cell cross-color protrusions revert to the
      surrounding region color.
    - Each original black dent whose cleaned 3x3 neighborhood is dominated by one
      foreground color creates a 3x3 patch of the opposite color at that dent.
    - Copy that whole patch by point reflection about the full-grid center, mark
      both patch centers teal (8), then crop the final result to the minimal
      rectangle containing every non-black cell.

    Key insight: The original dents are still the only triggers, but each trigger
    now produces a symmetric pair of teal-centered opposite-color patches, and the
    black outer frame is discarded at the end.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _apply_rule(clean, centers, colors, bg)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
FRAME_MARGIN_MIN, FRAME_MARGIN_MAX = 2, 4
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.42, 0.60
MIN_REGION_CELLS = 110
COVERAGE_MIN, COVERAGE_MAX = 0.52, 0.80

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 8, 14
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 4
DENT_PAIR_PROB = 0.12

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.80
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """
    Build a point-symmetric two-color scaffold inside a black frame.

    The symmetry is deliberate: it guarantees that reflected dent centers land
    inside the same host color, so mirrored patches are visually nontrivial.
    """
    margin_r = rng.randint(FRAME_MARGIN_MIN, FRAME_MARGIN_MAX)
    margin_c = rng.randint(FRAME_MARGIN_MIN, FRAME_MARGIN_MAX)
    min_h = max(8, int(round(N * RECT_SIZE_FRAC_MIN)))
    max_h = min(N - margin_r - 1, int(round(N * RECT_SIZE_FRAC_MAX)))
    min_w = max(8, int(round(N * RECT_SIZE_FRAC_MIN)))
    max_w = min(N - margin_c - 1, int(round(N * RECT_SIZE_FRAC_MAX)))
    if min_h > max_h or min_w > max_w:
        return None

    h_a = rng.randint(min_h, max_h)
    w_a = rng.randint(min_w, max_w)
    h_b = rng.randint(min_h, max_h)
    w_b = rng.randint(min_w, max_w)

    top = margin_r
    left = margin_c
    a_right = N - margin_c - w_a
    a_bottom = N - margin_r - h_a
    b_right = N - margin_c - w_b
    b_bottom = N - margin_r - h_b

    # Need overlap on both axes so each color gets visible occlusion boundaries.
    if left + w_a <= b_right or left + w_b <= a_right:
        return None
    if top + h_a <= b_bottom or top + h_b <= a_bottom:
        return None

    grid = np.zeros((N, N), dtype=int)
    grid[top : top + h_a, left : left + w_a] = int(color_a)
    grid[a_bottom : a_bottom + h_a, a_right : a_right + w_a] = int(color_a)
    grid[top : top + h_b, b_right : b_right + w_b] = int(color_b)
    grid[b_bottom : b_bottom + h_b, left : left + w_b] = int(color_b)

    if not np.array_equal(grid, np.flipud(np.fliplr(grid))):
        return None

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None

    coverage = (cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None

    fg = np.argwhere(grid != 0)
    if len(fg) == 0:
        return None
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0)
    if r0 < 2 or c0 < 2 or r1 > N - 3 or c1 > N - 3:
        return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if grid[r, c] != host:
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == host)) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == other))


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = color
                if int(np.sum(block == color)) == 4:
                    return True
    return False


def _pairwise_far(points: list[tuple[int, int]], r: int, c: int, min_sep: int) -> bool:
    return all(max(abs(int(r) - rr), abs(int(c) - cc)) >= int(min_sep) for rr, cc in points)


def _canonical_candidates(clean: np.ndarray, colors: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    h, w = clean.shape
    color_a, color_b = map(int, colors)
    valid: dict[int, list[tuple[int, int]]] = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host not in valid:
                continue
            rr, cc = _reflect_point(h, w, r, c)
            if (r > rr) or (r == rr and c >= cc):
                continue
            if (rr, cc) == (r, c):
                continue
            if int(clean[rr, cc]) != host:
                continue
            other = color_b if host == color_a else color_a
            if not _is_safe_dent_center(clean, r, c, host, other):
                continue
            if not _is_safe_dent_center(clean, rr, cc, host, other):
                continue
            valid[host].append((int(r), int(c)))
    return valid


def _pick_dent_centers(
    candidates: list[tuple[int, int]], n: int, shape: tuple[int, int], rng: random.Random
) -> list[tuple[int, int]] | None:
    h, w = map(int, shape)
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    occupied: list[tuple[int, int]] = []
    for r, c in pool:
        rr, cc = _reflect_point(h, w, r, c)
        pair = [(int(r), int(c)), (int(rr), int(cc))]
        if all(_pairwise_far(occupied, pr, pc, DENT_PATCH_SEPARATION) for pr, pc in pair):
            chosen.append((int(r), int(c)))
            occupied.extend(pair)
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_dents: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    occupied_other: list[tuple[int, int]] = []
    for r, c in all_dents:
        rr, cc = _reflect_point(h, w, r, c)
        occupied_other.extend([(int(r), int(c)), (int(rr), int(cc))])

    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(int(r), int(c))]
        if rng.random() < DENT_PAIR_PROB:
            others = [pt for pt in occupied_other if pt not in {(int(r), int(c)), _reflect_point(h, w, int(r), int(c))}]
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if int(clean[nr, nc]) != int(host):
                    continue
                rr, cc = _reflect_point(h, w, int(nr), int(nc))
                if int(clean[rr, cc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, int(nr), int(nc), int(host), int(other)):
                    continue
                if not _is_safe_dent_center(clean, int(rr), int(cc), int(host), int(other)):
                    continue
                if not all(_pairwise_far(others, pr, pc, DENT_PATCH_SEPARATION) for pr, pc in [(nr, nc), (rr, cc)]):
                    continue
                cells.append((int(nr), int(nc)))
                break
        result.append(cells)
    return result


def _groups_to_centers(groups_a: list[list[tuple[int, int]]], groups_b: list[list[tuple[int, int]]], color_a: int, color_b: int):
    centers: list[tuple[int, int, int]] = []
    for cells in groups_a:
        for r, c in cells:
            centers.append((int(r), int(c), int(color_a)))
    for cells in groups_b:
        for r, c in cells:
            centers.append((int(r), int(c), int(color_b)))
    return centers


def _blacken_dents(clean: np.ndarray, groups_a: list[list[tuple[int, int]]], groups_b: list[list[tuple[int, int]]]) -> np.ndarray:
    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[int(r), int(c)] = 0
    return inp


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    h, w = inp.shape
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = int(speck_color)
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random) -> None:
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((int(r), int(c), int(v), int(other)))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == int(host)))
        other_count = int(np.sum(sub_now == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, int(other)):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, int(host), int(other)):
            continue
        inp[r, c] = int(other)
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    valid = _canonical_candidates(clean, colors)
    if len(valid[color_a]) < 5 or len(valid[color_b]) < 5:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a
    if len(valid[color_a]) < n_a or len(valid[color_b]) < n_b:
        return None

    dents_a = _pick_dent_centers(valid[color_a], n_a, clean.shape, rng)
    dents_b = _pick_dent_centers(valid[color_b], n_b, clean.shape, rng)
    if dents_a is None or dents_b is None:
        return None

    all_base = dents_a + dents_b
    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, all_base, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, all_base, rng)
    centers = _groups_to_centers(groups_a, groups_b, color_a, color_b)

    actual_centers = [(r, c) for r, c, _host in centers]
    reflected_centers = [_reflect_point(N, N, r, c) for r, c in actual_centers]
    if len(set(actual_centers) | set(reflected_centers)) != 2 * len(actual_centers):
        return None

    out = _apply_rule(clean, centers, colors, Color.BLACK)
    base_out = _apply_base_rule(clean, centers, colors, Color.BLACK)
    if np.array_equal(out, base_out):
        return None
    if out.shape[0] >= N or out.shape[1] >= N:
        return None
    if int(np.sum(out == int(Color.TEAL))) != 2 * len(actual_centers):
        return None

    inp = _blacken_dents(clean, groups_a, groups_b)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
