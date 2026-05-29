# Combo variant of de809cff (index 29)
# combo_spec: D1/0+D2/3+D7/2
# Name: 十字形补丁+交替中心色+凹痕沿直线排列

from __future__ import annotations

# Design analysis for the combo variant
# -------------------------------------
# fused_rule:
#   - Keep the original task's two-stage reasoning:
#       1. clean one-cell foreground noise and boundary protrusions
#       2. detect dents as original black cells whose cleaned 3x3 neighborhood has a strict
#          foreground majority with at least 5 host cells
#   - Replace the original filled 3x3 opposite-color patch with a sparse orthogonal cross.
#   - After all dents are found, sort them in normal scan order (top-to-bottom, left-to-right).
#     Odd-index dents get a teal center; even-index dents get a maroon center.
#   - The generator no longer scatters dents arbitrarily: dents are arranged as visible straight
#     horizontal lines, so the alternating center colors form an obvious two-color rhythm.
# concrete_behavior:
#   - Original local event:
#       a black hole inside region A becomes a 3x3 square of region B with a teal center.
#   - Combo local event:
#       the same hole becomes a 5-cell cross:
#         up/down/left/right -> region B
#         center -> teal for dent #1, maroon for dent #2, teal for dent #3, ...
#   - Example:
#       if a row contains dent centers at columns 5, 8, 11, 14 in scan order, the output row
#       becomes a regular sequence of cross centers colored 8, 9, 8, 9 while the arms keep the
#       opposite foreground color. Relative to the original task, the diagonal cells of each 3x3
#       neighborhood are no longer recolored.
# complexity_calibration:
#   - The line arrangement is intentionally restrictive rather than fully free-form. This keeps the
#     alternation legible from only a few ARC examples: human solvers can see both "these holes lie
#     on rows" and "the center markers alternate".
#   - Using only horizontal lines avoids scan-order ambiguity. If vertical and horizontal lines were
#     mixed freely, the global alternating rule would be much harder to infer from few examples.
#   - The generator still uses two foreground regions and several dents, so the variant is visibly
#     distinct from both the original square-patch task and a trivial one-line toy construction.
# implicit_assumption_audit:
#   - `_clean_foreground` still assumes all corruption is one-cell noise/protrusions; the generator
#     preserves that contract.
#   - `_dent_centers` still assumes dents are encoded as original black cells and that host color is
#     recoverable from cleaned local majority; unchanged and still valid.
#   - The original stamping line `out[r-1:r+2, c-1:c+2] = other` depended on "patch = full 3x3".
#     That assumption is invalid now and is replaced by explicit writes only to the cross cells.
#   - The original code assumed every marked center had the same color. That is no longer true, so
#     center recoloring must happen in a second ordered pass after dents are globally sorted.
#   - `_main_colors` must not treat maroon as a possible region color in generated inputs, otherwise
#     output markers and region colors would become ambiguous. The generator therefore excludes both
#     teal and maroon from the foreground palette.
# placement_legality:
#   - Every chosen dent center is required to have its full cross footprint inside its host region,
#     so cross arms stay clean and do not clip into background.
#   - Centers on the same line are spaced by 3 columns, which keeps crosses disjoint while still
#     reading as a straight rhythmic chain.
#   - Different rows of dents are separated by at least 3 rows, preventing cross overlap between
#     lines. Oracle validation still checks `main(input) == intended_output`.
# visual_observability:
#   - Two quantitative cues are enforced:
#       1. cross-vs-square difference must be positive, so the D1 shape change is real
#       2. at least one even-index dent exists, so maroon appears and the D2 alternation is visible
#   - All dents belong to horizontal arithmetic progressions, making the D7 straight-line structure
#     visually obvious instead of statistically incidental.
# main_new_assumptions:
#   - Scan-order alternation only makes sense if dent order is stable, so `main()` explicitly sorts
#     the detected centers by `(row, col)`.
#   - The generator guarantees enough full crosses and row-wise grouping so humans can infer the
#     alternation from the outputs rather than needing hidden numeric color-id reasoning.

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    seen: list[int] = []
    present: set[int] = set()
    for value in g.flat:
        color = int(value)
        if color == int(bg) or color in present:
            continue
        present.add(color)
        seen.append(color)
    counts = {color: int(np.sum(g == color)) for color in seen}
    seen.sort(key=lambda color: counts[color], reverse=True)
    assert len(seen) >= 2, "expected two foreground colors"
    return int(seen[0]), int(seen[1])


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

    Two sweeps are enough for the corruption model used by the generator.
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
    centers.sort(key=lambda item: (int(item[0]), int(item[1])))
    return centers


def _cross_cells(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    cells: list[tuple[int, int]] = []
    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < h and 0 <= cc < w:
            cells.append((rr, cc))
    return cells


def _marker_color(index_1based: int) -> int:
    return int(Color.TEAL if index_1based % 2 == 1 else Color.MAROON)


def _stamp_combo(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for rr, cc in _cross_cells(clean.shape, r, c):
            out[rr, cc] = int(other)
    for idx, (r, c, _host) in enumerate(centers, start=1):
        out[r, c] = _marker_color(idx)
    return out


def _stamp_square_baseline(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for idx, (r, c, _host) in enumerate(centers, start=1):
        out[r, c] = _marker_color(idx)
    return out


def _stamp_all_teal_cross(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    out = clean.copy()
    for r, c, host in centers:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        for rr, cc in _cross_cells(clean.shape, r, c):
            out[rr, cc] = int(other)
        out[r, c] = int(Color.TEAL)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into opposite-color crosses with alternating centers.

    Input structure:
    - Black background with two large foreground regions in different non-marker colors.
    - The regions contain black dents plus one-cell foreground noise in the background or at the
      boundary between the two foreground regions.

    Rule:
    - First locally clean the foreground so isolated specks revert to black and one-cell boundary
      contamination flips back to its surrounding region color.
    - For each original black dent whose cleaned 3x3 neighborhood is dominated by one foreground
      color, paint only the orthogonal cross at that dent in the opposite foreground color.
    - Order all such dents by normal top-left scanning; recolor odd centers to teal (8) and even
      centers to maroon (9).

    Key insight: The dents are still identified from cleaned local majority, but the output mark is
    now a sparse opposite-color plus sign, and the center color alternates by scan order instead of
    being a constant marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _stamp_combo(clean, centers, colors)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent-line placement ---
LINE_STEP = 3
MANDATORY_LINE_LEN_MIN, MANDATORY_LINE_LEN_MAX = 5, 8
EXTRA_LINE_LEN_MIN, EXTRA_LINE_LEN_MAX = 4, 6
LINE_ROW_GAP = 3
TOTAL_DENTS_MIN, TOTAL_DENTS_MAX = 12, 22
MIN_MAROON_CENTERS = 3

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 22
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen_corners = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        mid = color_b if first == color_a else color_a
        rect_colors = [first, mid, first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = int(color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == int(rect_colors[0]))
        if int(n_components) < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= 5 and host_count > int(np.sum(sub == int(other)))


def _is_full_cross_host(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    for rr, cc in _cross_cells(grid.shape, r, c):
        if int(grid[rr, cc]) != int(host):
            return False
    return True


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _find_horizontal_runs(
    clean: np.ndarray, host: int, other: int
) -> list[tuple[int, list[int]]]:
    h, w = clean.shape
    runs: list[tuple[int, list[int]]] = []
    for r in range(1, h - 1):
        valid_cols: list[int] = []
        for c in range(1, w - 1):
            if not _is_safe_dent_center(clean, r, c, host, other):
                continue
            if not _is_full_cross_host(clean, r, c, host):
                continue
            valid_cols.append(int(c))

        for residue in range(LINE_STEP):
            cols = [c for c in valid_cols if c % LINE_STEP == residue]
            if not cols:
                continue
            current = [cols[0]]
            for c in cols[1:]:
                if int(c) == int(current[-1]) + LINE_STEP:
                    current.append(int(c))
                else:
                    if len(current) >= EXTRA_LINE_LEN_MIN:
                        runs.append((int(r), current.copy()))
                    current = [int(c)]
            if len(current) >= EXTRA_LINE_LEN_MIN:
                runs.append((int(r), current.copy()))
    return runs


def _row_is_free(row: int, reserved_rows: list[int]) -> bool:
    return all(abs(int(row) - int(prev)) >= LINE_ROW_GAP for prev in reserved_rows)


def _choose_line(
    runs: list[tuple[int, list[int]]],
    reserved_rows: list[int],
    rng: random.Random,
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]] | None:
    candidates = runs.copy()
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: len(item[1]), reverse=True)
    for row, cols in candidates:
        if not _row_is_free(int(row), reserved_rows):
            continue
        usable_max = min(int(max_len), len(cols))
        if usable_max < int(min_len):
            continue
        length = rng.randint(int(min_len), int(usable_max))
        start = rng.randint(0, len(cols) - length)
        return [(int(row), int(cols[start + i])) for i in range(length)]
    return None


def _select_lines(
    clean: np.ndarray, color_a: int, color_b: int, rng: random.Random
) -> list[tuple[int, int, int]] | None:
    runs = {
        int(color_a): _find_horizontal_runs(clean, int(color_a), int(color_b)),
        int(color_b): _find_horizontal_runs(clean, int(color_b), int(color_a)),
    }
    if not runs[int(color_a)] or not runs[int(color_b)]:
        return None

    reserved_rows: list[int] = []
    selected: list[tuple[int, list[tuple[int, int]]]] = []
    counts = {int(color_a): 0, int(color_b): 0}

    order = [int(color_a), int(color_b)]
    order.sort(key=lambda color: len(runs[color]))
    for color in order:
        other = int(color_b if color == int(color_a) else color_a)
        _ = other
        line = _choose_line(
            runs[color], reserved_rows, rng, MANDATORY_LINE_LEN_MIN, MANDATORY_LINE_LEN_MAX
        )
        if line is None:
            return None
        selected.append((color, line))
        reserved_rows.append(int(line[0][0]))
        counts[color] += len(line)

    target_total = rng.randint(TOTAL_DENTS_MIN, TOTAL_DENTS_MAX)
    while sum(counts.values()) < target_total:
        remaining_budget = TOTAL_DENTS_MAX - sum(counts.values())
        if remaining_budget < EXTRA_LINE_LEN_MIN:
            break
        extra_order = [int(color_a), int(color_b)]
        extra_order.sort(key=lambda color: (counts[color], len(runs[color])))
        placed = False
        for color in extra_order:
            line = _choose_line(
                runs[color],
                reserved_rows,
                rng,
                EXTRA_LINE_LEN_MIN,
                min(EXTRA_LINE_LEN_MAX, remaining_budget),
            )
            if line is None:
                continue
            selected.append((color, line))
            reserved_rows.append(int(line[0][0]))
            counts[color] += len(line)
            placed = True
            break
        if not placed:
            break

    if sum(counts.values()) < TOTAL_DENTS_MIN:
        return None

    centers: list[tuple[int, int, int]] = []
    for color, line in selected:
        for r, c in line:
            centers.append((int(r), int(c), int(color)))
    centers.sort(key=lambda item: (int(item[0]), int(item[1])))

    maroon_count = len(centers) // 2
    if maroon_count < MIN_MAROON_CENTERS:
        return None
    return centers


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
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
        speck_color = (
            _Counter(nearby_fg).most_common(1)[0][0]
            if nearby_fg
            else rng.choice([int(color_a), int(color_b)])
        )
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


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b if v == int(color_a) else color_a)
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((int(r), int(c), v, other))
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


def _line_pattern_score(centers: list[tuple[int, int, int]]) -> int:
    by_row: dict[int, list[int]] = {}
    for r, c, _host in centers:
        by_row.setdefault(int(r), []).append(int(c))
    score = 0
    for cols in by_row.values():
        cols.sort()
        if len(cols) < EXTRA_LINE_LEN_MIN:
            continue
        if all(cols[i + 1] - cols[i] == LINE_STEP for i in range(len(cols) - 1)):
            score += len(cols)
    return score


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c not in (Color.TEAL, Color.MAROON)]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, int(color_a), int(color_b), rng)
    if clean is None:
        return None

    centers = _select_lines(clean, int(color_a), int(color_b), rng)
    if centers is None:
        return None

    colors = (int(color_a), int(color_b))
    output = _stamp_combo(clean, centers, colors)

    if _line_pattern_score(centers) != len(centers):
        return None
    if int(np.sum(output == int(Color.MAROON))) < MIN_MAROON_CENTERS:
        return None

    square_baseline = _stamp_square_baseline(clean, centers, colors)
    if int(np.sum(output != square_baseline)) <= 0:
        return None

    teal_baseline = _stamp_all_teal_cross(clean, centers, colors)
    if int(np.sum(output != teal_baseline)) < MIN_MAROON_CENTERS:
        return None

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, int(color_a), int(color_b), rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, int(color_a), int(color_b), rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
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
