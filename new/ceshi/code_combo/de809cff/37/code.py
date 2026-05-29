# Combo variant of de809cff (index 37)
# combo_spec: D13/2+D15/1
# Name: 色斑指示激活条件+单向映射

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# -------------------------------------
# a) concretization_and_trace:
#    - The scene still contains two large foreground regions on black background,
#      with black dents punched into both colors and a small amount of boundary
#      contamination noise that must be cleaned away before dent reasoning.
#    - The original task transformed dents in both colors symmetrically:
#      host A dent -> 3x3 patch of B with teal center, and host B dent -> 3x3
#      patch of A with teal center.
#    - D13/2 removes that symmetry: only one host color is active, so dents in
#      the inactive color stay black in the output.
#    - D15/1 supplies the missing "which host is active?" signal. Background
#      isolated specks are no longer irrelevant dirt; their color is the control
#      channel. If the isolated background specks are red, then red-region dents
#      are the ones that transform, while blue-region dents remain black.
#    - Concrete trace: imagine a blue rectangle and a yellow rectangle. Input
#      also contains several isolated yellow pixels in otherwise black space.
#      After cleanup those yellow specks disappear, but they have already told
#      the solver that yellow is active. Therefore yellow dents become 3x3 blue
#      patches with teal centers, while blue dents remain black holes.
# b) complexity_calibration:
#    - The control variable is binary and repeated 2-4 times per grid via
#      several same-color background specks, so it is visible rather than being
#      a single brittle pixel.
#    - Both regions still carry multiple dents. Each sample therefore shows
#      transformed dents and untouched dents side by side, making the one-way
#      mapping visually obvious.
#    - The generator does not encode activation via color id ordering or a fixed
#      size rule. Active color is chosen independently, so the control specks
#      carry real information rather than merely confirming another cue.
# c) implicit_assumption_review:
#    - `_main_colors(...)` may still recover the two foreground colors, because
#      the control specks reuse one of those colors instead of introducing a
#      third color.
#    - `_clean_foreground(...)` still removes isolated specks, but the solver
#      can no longer ignore them. Activation must be inferred from the original
#      grid before those specks vanish in the cleaned grid.
#    - `_dent_centers(...)` still detects dents in both hosts. The changed
#      assumption is in `main(...)`: only the host named by the control specks
#      gets stamped, and inactive dents must be restored to black because
#      `clean.copy()` would otherwise heal them.
#    - Boundary contamination remains safe because the generator places it where
#      cleanup returns it to a foreground host, not to black; this keeps control
#      speck detection distinct from region noise.
# d) placement_legality:
#    - Dent centers across both colors obey a global Chebyshev separation, so an
#      active 3x3 patch cannot overwrite an inactive dent center.
#    - Control specks are placed only on background cells whose 3x3
#      neighborhoods are mostly black, and those specks are also mutually
#      separated so they stay visibly isolated.
#    - Boundary noise is injected only where the cleanup heuristic will revert
#      it to the surrounding host color, preserving solver assumptions.
# e) visual_observability:
#    - A human can see three repeated facts in each training pair: one color of
#      isolated background specks exists, dents in that same color transform,
#      and dents in the other color remain black.
#    - Region size is intentionally not the deciding cue. The active color may
#      be either the larger or smaller region across different examples, forcing
#      attention onto the background speck color.
# f) visual_cleanliness:
#    - Only one semantic use of background specks remains: control. There are no
#      opposite-color background specks to create contradictory signals.
#    - Other noise stays confined to one-cell boundary contamination, so the
#      grid is still readable after the control specks are noticed.
# g) color_mapping_inference:
#    - Only two foreground colors participate. The mapping remains structured:
#      active-color dents map to the other foreground color, inactive-color
#      dents do not map at all.
#    - Because the control pixels share a color with one region, the solver can
#      infer a direct color-to-region correspondence from visible repetition
#      rather than from hidden numeric ordering.


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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
    Remove isolated specks and one-cell protrusions before dent detection.

    The control specks are intentionally removed here; `main(...)` reads their
    color from the original grid before relying on the cleaned foreground.
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


def _control_speck_counts(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> dict[int, int]:
    counts = {int(color): 0 for color in colors}
    h, w = original.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            v = int(original[r, c])
            if v not in counts or int(clean[r, c]) != int(bg):
                continue
            sub = _sub3x3(original, r, c)
            if int(np.sum(sub == v)) > 2:
                continue
            if int(np.sum(sub == int(bg))) < 6:
                continue
            counts[v] += 1
    return counts


def _active_host_color(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> int:
    counts = _control_speck_counts(original, clean, colors, bg)
    active = max(colors, key=lambda color: counts[int(color)])
    assert counts[int(active)] > 0, "expected at least one control speck"
    return int(active)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform dents only for the foreground color named by isolated background specks.

    Input structure:
    - Black background with two large foreground colors.
    - Both colored regions may contain black dents, and there can be a few
      one-cell boundary contaminants between the two regions.
    - The background also contains several isolated one-cell specks, all in a
      single foreground color; these specks are a control signal, not output content.

    Rule:
    - First locally clean the foreground in the same style as the base task:
      isolated specks disappear to black and wrong-color boundary protrusions
      revert to their surrounding host region.
    - Read the control signal from the original input: whichever foreground
      color appears as isolated background specks becomes the active host color.
    - Only dents attributed to that active host produce a 3x3 patch in the
      opposite foreground color, with the dent center recolored teal (8).
      Dents in the inactive host stay black.

    Key insight: The background specks do not survive into the output; they only
    indicate which colored region is allowed to perform the usual opposite-color
    dent expansion, turning the original symmetric rule into a speck-controlled
    one-way mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    active = _active_host_color(g, clean, colors, bg)
    inactive = int(colors[1] if active == int(colors[0]) else colors[0])

    out = clean.copy()
    active_centers: list[tuple[int, int]] = []
    inactive_centers: list[tuple[int, int]] = []
    for r, c, host in _dent_centers(g, clean, colors, bg):
        if int(host) == int(active):
            active_centers.append((r, c))
        else:
            inactive_centers.append((r, c))

    h, w = g.shape
    for r, c in active_centers:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = inactive

    for r, c in active_centers:
        out[r, c] = Color.TEAL

    for r, c in inactive_centers:
        out[r, c] = bg

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.82
AREA_GAP_MAX = 110

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
MIN_ACTIVE_DENTS = 6
MIN_INACTIVE_DENTS = 4
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: noise / control ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 3, 10
BOUNDARY_MIN_HOST_3x3 = 6
CONTROL_SPECK_COUNT_MIN, CONTROL_SPECK_COUNT_MAX = 2, 4
CONTROL_BG_MIN_3X3 = 7
CONTROL_PICK_ATTEMPTS = 12

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 300
CENTER_PICK_ATTEMPTS = 12


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles (simple occlusion, painter's algorithm)."""
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
        grid[r0:r1, c0:c1] = color

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    if abs(cnt_a - cnt_b) > AREA_GAP_MAX:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == rect_colors[0])
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    """After setting (r,c) to black, the solver must detect it as a dent in host region."""
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


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


def _far_enough(r: int, c: int, others: list[tuple[int, int]]) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in others)


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    blocked = [] if blocked is None else list(blocked)
    for r, c in pool:
        if _far_enough(r, c, chosen) and _far_enough(r, c, blocked):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    blocked_centers: list[tuple[int, int]],
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
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any((nr, nc) == center for center in blocked_centers if center != (r, c)):
                    continue
                if not _far_enough(nr, nc, [center for center in blocked_centers if center != (r, c)]):
                    continue
                cells.append((nr, nc))
                break
        result.append(cells)
    return result


def _build_output(
    clean: np.ndarray,
    active_groups: list[list[tuple[int, int]]],
    inactive_groups: list[list[tuple[int, int]]],
    inactive_color: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for cells in active_groups:
        for r, c in cells:
            output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = inactive_color
    for cells in active_groups:
        for r, c in cells:
            output[r, c] = Color.TEAL
    for cells in inactive_groups:
        for r, c in cells:
            output[r, c] = Color.BLACK
    return output


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
            nsub[r - nr0, c - nc0] = other
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, color_a: int, color_b: int, rng: random.Random
) -> None:
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b if v == int(color_a) else color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((r, c, v, other))
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
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = other
        placed += 1


def _pick_control_specks(
    clean: np.ndarray, n_specks: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(clean[r, c]) != Color.BLACK:
                continue
            if int(np.sum(_sub3x3(clean, r, c) == Color.BLACK)) < CONTROL_BG_MIN_3X3:
                continue
            candidates.append((r, c))

    for _ in range(CONTROL_PICK_ATTEMPTS):
        pool = list(candidates)
        rng.shuffle(pool)
        chosen: list[tuple[int, int]] = []
        for r, c in pool:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) >= n_specks:
                    return chosen
    return None


def _add_control_specks(
    inp: np.ndarray, clean: np.ndarray, active_color: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    n_specks = rng.randint(CONTROL_SPECK_COUNT_MIN, CONTROL_SPECK_COUNT_MAX)
    positions = _pick_control_specks(clean, n_specks, rng)
    if positions is None:
        return None
    for r, c in positions:
        inp[r, c] = int(active_color)
    return positions


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    active_color = int(rng.choice([color_a, color_b]))
    inactive_color = int(color_b if active_color == int(color_a) else color_a)

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_active_min = max(MIN_ACTIVE_DENTS, n_total // 2)
    n_active_max = n_total - MIN_INACTIVE_DENTS
    if n_active_min > n_active_max:
        return None
    n_active = rng.randint(n_active_min, n_active_max)
    n_inactive = n_total - n_active

    h, w = clean.shape
    valid = {int(color_a): [], int(color_b): []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == int(color_a) and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[int(color_a)].append((r, c))
            elif v == int(color_b) and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[int(color_b)].append((r, c))

    if len(valid[active_color]) < n_active + 2 or len(valid[inactive_color]) < n_inactive + 2:
        return None

    dents_active = None
    dents_inactive = None
    for _ in range(CENTER_PICK_ATTEMPTS):
        dents_active = _pick_dent_centers(valid[active_color], n_active, rng)
        if dents_active is None:
            continue
        dents_inactive = _pick_dent_centers(valid[inactive_color], n_inactive, rng, blocked=dents_active)
        if dents_inactive is not None:
            break
    if dents_active is None or dents_inactive is None:
        return None

    blocked_centers = dents_active + dents_inactive
    active_groups = _expand_to_pairs(dents_active, active_color, inactive_color, clean, blocked_centers, rng)
    inactive_groups = _expand_to_pairs(dents_inactive, inactive_color, active_color, clean, blocked_centers, rng)

    output = _build_output(clean, active_groups, inactive_groups, inactive_color)

    inp = clean.copy()
    for cells in active_groups + inactive_groups:
        for r, c in cells:
            inp[r, c] = Color.BLACK

    control_positions = _add_control_specks(inp, clean, active_color, rng)
    if control_positions is None:
        return None

    n_boundary_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    _add_boundary_noise(inp, clean, n_boundary_noise, color_a, color_b, rng)

    solver_clean = _clean_foreground(inp, (int(color_a), int(color_b)), Color.BLACK)
    control_counts = _control_speck_counts(inp, solver_clean, (int(color_a), int(color_b)), Color.BLACK)
    if control_counts[int(active_color)] < CONTROL_SPECK_COUNT_MIN:
        return None
    if control_counts[int(inactive_color)] != 0:
        return None

    resolved_active = _active_host_color(inp, solver_clean, (int(color_a), int(color_b)), Color.BLACK)
    if int(resolved_active) != int(active_color):
        return None

    solver_centers = _dent_centers(inp, solver_clean, (int(color_a), int(color_b)), Color.BLACK)
    resolved_active_count = sum(int(host == int(active_color)) for _r, _c, host in solver_centers)
    resolved_inactive_count = sum(int(host == int(inactive_color)) for _r, _c, host in solver_centers)
    if resolved_active_count < MIN_ACTIVE_DENTS or resolved_inactive_count < MIN_INACTIVE_DENTS:
        return None

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
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
