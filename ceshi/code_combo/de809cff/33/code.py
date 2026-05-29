# Combo variant of de809cff (index 33)
# combo_spec: D8/1+D11/0
# Name: 背景填充为第三种颜色+输出裁剪

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the fused variant
# a) Behavior trace:
#    Original: clean one-cell noise -> detect black dents in the two main regions ->
#    stamp opposite-color 3x3 patches -> keep the full black-framed canvas.
#    Fused: keep the same local dent logic, but after stamping, every remaining black
#    cell becomes a third color (grey=5), then crop to the bbox of non-grey cells.
#    A dent on the outer silhouette can push the final bbox outward by one cell,
#    so crop is determined by the transformed foreground, not by the original region bbox.
# b) Complexity calibration:
#    Only one extra global color is introduced, and it is fixed across samples.
#    Foreground still uses exactly two colors, so the recolor rule stays inferable.
#    Generator forces at least two silhouette dents so the crop change is visually obvious.
# c) Implicit assumptions review:
#    - The original cleaner assumes black is the only non-foreground color in input.
#      This still holds because the new grey only appears in output.
#    - `_main_colors` must recover exactly two meaningful colors, so generator forbids
#      grey and teal as input foreground colors.
#    - Cropping cannot use black anymore; it must use the new output background color.
# d) Placement legality:
#    Rectangles are inset from the canvas, dents are chosen from valid host cells, and
#    global Chebyshev spacing avoids tangled 3x3 patch overlaps.
# e) Visual observability:
#    Humans can see three stable signals: black exists only in input, grey fills all
#    leftover background in output, and the output rectangle shrinks around colored content.
# f) Visual cleanliness:
#    The canvas keeps two large blocky regions, few one-cell noise points, and spaced dents;
#    this avoids clutter after patch expansion and crop.
# g) Inferrability of color change:
#    Grey is not sampled as a foreground color, so it is consistently the "new background"
#    color rather than an ambiguous reused region color.


OUTPUT_BG = Color.GREY


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


def _crop_to_foreground_bbox(g: np.ndarray, bg_fill: int) -> np.ndarray:
    rs, cs = np.where(g != int(bg_fill))
    assert len(rs) > 0, "expected non-background output"
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _solve_full(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    centers = _dent_centers(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)

    for r, c, _host in centers:
        out[r, c] = Color.TEAL

    out[out == int(bg)] = int(OUTPUT_BG)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert black dents into marked opposite-color patches, then recolor and crop the canvas.

    Input structure:
    - A black background surrounds two large foreground regions in two non-black colors.
    - The regions contain true black dents plus a few one-cell specks or boundary intrusions.

    Rule:
    - First clean the foreground locally so isolated specks disappear and one-cell boundary
      contamination snaps back to the dominant neighboring region color.
    - Each original black dent whose cleaned 3x3 neighborhood is mostly one region color
      becomes the center of an opposite-color 3x3 patch, with the center recolored teal (8).
    - After all dent patches are applied, recolor every remaining black cell to grey (5),
      then crop away any outer rows or columns that are entirely grey.

    Key insight: black is only an input-time cue for dents and removable noise; once dents are
    resolved, the output background is a uniform third color, and the final frame is determined
    by the transformed colored content rather than by the original canvas size.
    """
    full = _solve_full(grid)
    return _crop_to_foreground_bbox(full, OUTPUT_BG)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.52, 0.74
RECT_MARGIN_MIN, RECT_MARGIN_MAX = 1, 4
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.82

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 10, 18
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.0
REQUIRED_EXPANSION_DENTS = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.78, 0.92
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _bbox_of_color(grid: np.ndarray, bg: int = 0) -> tuple[int, int, int, int]:
    rs, cs = np.where(grid != int(bg))
    assert len(rs) > 0
    return int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    chosen_corners = rng.choice([("TL", "BR"), ("TR", "BL")])
    rect_colors = [color_a, color_b]
    if rng.random() < 0.5:
        rect_colors.reverse()

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(RECT_MARGIN_MIN, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None

    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None

    r0, r1, c0, c1 = _bbox_of_color(grid)
    if r0 < 1 or c0 < 1 or r1 > N - 2 or c1 > N - 2:
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


def _far_enough(cell: tuple[int, int], others: list[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in others)


def _choose_required_side_dents(
    valid: dict[int, list[tuple[int, int]]],
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    r0, r1, c0, c1 = bbox
    side_candidates: dict[str, list[tuple[int, int, int]]] = {"top": [], "bottom": [], "left": [], "right": []}
    for color, cells in valid.items():
        for r, c in cells:
            if r == r0:
                side_candidates["top"].append((r, c, color))
            if r == r1:
                side_candidates["bottom"].append((r, c, color))
            if c == c0:
                side_candidates["left"].append((r, c, color))
            if c == c1:
                side_candidates["right"].append((r, c, color))

    sides = [side for side, cells in side_candidates.items() if cells]
    if len(sides) < REQUIRED_EXPANSION_DENTS:
        return None

    rng.shuffle(sides)
    chosen: list[tuple[int, int, int]] = []
    used_cells: list[tuple[int, int]] = []
    for side in sides:
        candidates = side_candidates[side][:]
        rng.shuffle(candidates)
        candidates.sort(key=lambda item: (abs(item[0] - (r0 + r1) / 2) + abs(item[1] - (c0 + c1) / 2)))
        for r, c, color in candidates:
            if (r, c) in used_cells:
                continue
            if _far_enough((r, c), used_cells):
                chosen.append((r, c, color))
                used_cells.append((r, c))
                break
        if len(chosen) >= REQUIRED_EXPANSION_DENTS:
            return chosen
    return None


def _pick_color_dents(
    valid: dict[int, list[tuple[int, int]]],
    counts: dict[int, int],
    reserved: list[tuple[int, int, int]],
    rng: random.Random,
) -> dict[int, list[tuple[int, int]]] | None:
    selected = {color: [] for color in valid}
    occupied: list[tuple[int, int]] = []
    for r, c, color in reserved:
        selected[color].append((r, c))
        occupied.append((r, c))
    if any(len(selected[color]) > counts[color] for color in valid):
        return None

    order = sorted(valid.keys(), key=lambda color: counts[color] - len(selected[color]))
    for color in order:
        candidates = valid[color][:]
        rng.shuffle(candidates)
        for r, c in candidates:
            if len(selected[color]) >= counts[color]:
                break
            if (r, c) in occupied:
                continue
            if not _far_enough((r, c), occupied):
                continue
            selected[color].append((r, c))
            occupied.append((r, c))
        if len(selected[color]) != counts[color]:
            return None
    return selected


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_base_dents: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(r, c)]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if clean[nr, nc] != host or not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in all_base_dents
                    if (r2, c2) != (r, c)
                ):
                    cells.append((nr, nc))
                    break
        result.append(cells)
    return result


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, stamp_color in [(dent_groups_a, color_b), (dent_groups_b, color_a)]:
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = stamp_color
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _finalize_variant_output(full_output: np.ndarray) -> np.ndarray:
    out = full_output.copy()
    out[out == int(Color.BLACK)] = int(OUTPUT_BG)
    return _crop_to_foreground_bbox(out, OUTPUT_BG)


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
        inp[r, c] = speck_color
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or inp[nr, nc] != host:
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = other
            own_n = int(np.sum(nsub == host))
            other_n = int(np.sum(nsub == other))
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
            if v not in (color_a, color_b):
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if inp[r, c] != host:
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == host))
        other_count = int(np.sum(sub_now == other))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c not in (Color.TEAL, OUTPUT_BG)]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a
    counts = {color_a: n_a, color_b: n_b}

    h, w = clean.shape
    valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((r, c))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((r, c))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    bbox = _bbox_of_color(clean)
    reserved = _choose_required_side_dents(valid, bbox, rng)
    if reserved is None:
        return None

    dents_by_color = _pick_color_dents(valid, counts, reserved, rng)
    if dents_by_color is None:
        return None

    all_base_dents = dents_by_color[color_a] + dents_by_color[color_b]
    groups_a = _expand_to_pairs(dents_by_color[color_a], color_a, color_b, clean, all_base_dents, rng)
    groups_b = _expand_to_pairs(dents_by_color[color_b], color_b, color_a, clean, all_base_dents, rng)

    full_output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b)
    output = _finalize_variant_output(full_output)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None
    if produced.shape == inp.shape:
        return None

    full_from_solver = _solve_full(inp)
    rs, cs = np.where(full_from_solver != int(OUTPUT_BG))
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())
    clean_r0, clean_r1, clean_c0, clean_c1 = _bbox_of_color(clean)
    if (r0, r1, c0, c1) == (clean_r0, clean_r1, clean_c0, clean_c1):
        return None
    if np.any(produced == int(Color.BLACK)):
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
