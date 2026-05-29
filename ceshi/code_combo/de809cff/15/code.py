# Combo variant of de809cff (index 15)
# combo_spec: D4/0+D11/2
# Name: 三色区域分别输出为独立网格

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for the combo variant
# -------------------------------------
# concrete_behavior:
#   clean scene:
#     - black background plus exactly three large foreground regions with stable
#       spatial roles: left, right, bottom
#     - each region contains black dents; one-cell wrong-color noise may also
#       appear in the background or near region boundaries
#   output scene:
#     - each host region is processed independently: its dents stamp 3x3 patches
#       in the cyclic target color left -> right -> bottom -> left, with teal
#       centers
#     - instead of returning one shared canvas, the three per-host results are
#       cropped to tight bounding boxes and concatenated horizontally with solid
#       separator columns
#   small execution trace:
#     - source task: a black dent inside one of two regions becomes a 3x3 patch
#       of the only other foreground color on the original canvas
#     - D4 change: with three regions, a left-region dent no longer has a unique
#       "other"; it uses the right-region color, while right uses bottom and
#       bottom uses left
#     - D11 change: after those dents are resolved, the global canvas is not the
#       answer; instead, isolate each host region's transformed result, crop it,
#       and place the three crops side by side
# complexity_calibration:
#   - the color rule has only one repeated structure: a visible 3-cycle on the
#     stable roles left/right/bottom
#   - the packing rule also has only one repeated structure: exactly three
#     horizontally ordered panels with a fixed separator color
#   - each sample forces dents in all three roles, so neither the color cycle nor
#     the panel split can disappear from the training signal
# implicit_assumption_audit:
#   - `_main_colors` can no longer assume two colors; it must recover three
#   - cleanup can no longer compare against only one rival color, so it requires
#     a unique dominant foreground color in the local 3x3 neighborhood
#   - dent stamping can no longer be "host -> unique other"; it depends on role
#     inference from cleaned region centroids
#   - output assembly can no longer assume same-shape output; it now assumes each
#     role has a non-empty region that yields a non-empty cropped panel
# placement_legality:
#   - the generator paints three large rectangles in stable left/right/bottom
#     roles so centroid-based role inference is reliable
#   - dent centers are selected only where the host color remains the unique 3x3
#     majority after blacking the center, and centers are kept Chebyshev-spaced
#   - noise is inserted only when the cleanup logic restores the intended clean
#     scene; every sample is then verified by the full solver
# visual_observability:
#   - role identity comes from position, not from hidden properties such as color
#     ids or area sorting
#   - the output panels are ordered by the same visible roles left, right, bottom
#   - the fixed separator color makes the repacking operation obvious to a human
# neatness:
#   - each panel contains only one base region plus its own stamped patches and
#     teal centers, so different hosts do not visually overwrite each other
#   - cropping removes irrelevant empty canvas, and separators keep the three
#     transformed sub-results distinct


PANEL_SEPARATOR = Color.PINK


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 3, "expected three foreground colors"
    return int(colors[0]), int(colors[1]), int(colors[2])


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


def _dominant_color(sub: np.ndarray, colors: tuple[int, int, int], threshold: int = 5) -> int | None:
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    best = max(counts.values())
    winners = [int(color) for color, count in counts.items() if count == best]
    if best < threshold or len(winners) != 1:
        return None
    return int(winners[0])


def _clean_cell(cur: np.ndarray, r: int, c: int, colors: tuple[int, int, int], bg: int) -> int:
    v = int(cur[r, c])
    if v not in colors:
        return int(bg)

    if _has_same_2x2(cur, r, c, v):
        return v

    sub = _sub3x3(cur, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    own = counts[int(v)]
    bg_count = int(np.sum(sub == int(bg)))
    dominant = _dominant_color(sub, colors, threshold=5)

    if dominant is not None and int(dominant) != int(v) and counts[int(dominant)] > own:
        return int(dominant)
    if bg_count >= 5 and own <= 4:
        return int(bg)
    if own <= 2:
        return int(bg)
    return int(v)


def _clean_foreground(g: np.ndarray, colors: tuple[int, int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before detecting black dents.

    The original two-pass local cleanup still works here as long as recoloring
    toward a rival foreground requires a unique dominant color among the three.
    """
    h, w = g.shape
    cur = g.copy()
    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                nxt[r, c] = _clean_cell(cur, r, c, colors, bg)
        cur = nxt
    return cur


def _role_colors(clean: np.ndarray, colors: tuple[int, int, int]) -> dict[str, int]:
    stats: dict[int, tuple[float, float]] = {}
    for color in colors:
        coords = np.argwhere(clean == int(color))
        assert len(coords) > 0, "missing region color"
        stats[int(color)] = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

    bottom = max(colors, key=lambda color: stats[int(color)][0])
    top_two = [int(color) for color in colors if int(color) != int(bottom)]
    left = min(top_two, key=lambda color: stats[int(color)][1])
    right = max(top_two, key=lambda color: stats[int(color)][1])
    return {"left": int(left), "right": int(right), "bottom": int(bottom)}


def _stamp_map(clean: np.ndarray, colors: tuple[int, int, int]) -> dict[int, int]:
    roles = _role_colors(clean, colors)
    return {
        int(roles["left"]): int(roles["right"]),
        int(roles["right"]): int(roles["bottom"]),
        int(roles["bottom"]): int(roles["left"]),
    }


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int, int], bg: int
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(original == int(bg)):
        rr = int(r)
        cc = int(c)
        dominant = _dominant_color(_sub3x3(clean, rr, cc), colors, threshold=5)
        if dominant is not None:
            centers.append((rr, cc, int(dominant)))
    return centers


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != int(bg))
    assert len(coords) > 0, "expected non-empty panel"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _ordered_hosts(clean: np.ndarray, colors: tuple[int, int, int]) -> list[int]:
    roles = _role_colors(clean, colors)
    return [int(roles["left"]), int(roles["right"]), int(roles["bottom"])]


def _build_host_panel(
    clean: np.ndarray,
    centers: list[tuple[int, int, int]],
    host: int,
    stamp_color: int,
    bg: int,
) -> np.ndarray:
    h, w = clean.shape
    panel = np.full((h, w), int(bg), dtype=int)
    panel[clean == int(host)] = int(host)

    for r, c, center_host in centers:
        if int(center_host) != int(host):
            continue
        panel[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for r, c, center_host in centers:
        if int(center_host) == int(host):
            panel[r, c] = int(Color.TEAL)

    return _crop_non_bg(panel, bg)


def _join_panels_horizontally(panels: list[np.ndarray], bg: int, separator: int) -> np.ndarray:
    assert panels, "expected at least one panel"
    max_h = max(int(panel.shape[0]) for panel in panels)
    total_w = sum(int(panel.shape[1]) for panel in panels) + (len(panels) - 1)
    out = np.full((max_h, total_w), int(bg), dtype=int)

    c0 = 0
    for i, panel in enumerate(panels):
        h, w = panel.shape
        out[:h, c0 : c0 + w] = panel
        c0 += w
        if i + 1 < len(panels):
            out[:, c0] = int(separator)
            c0 += 1
    return out


def _build_output(clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int, int], bg: int) -> np.ndarray:
    stamp_map = _stamp_map(clean, colors)
    panels = [
        _build_host_panel(clean, centers, host, int(stamp_map[int(host)]), bg)
        for host in _ordered_hosts(clean, colors)
    ]
    return _join_panels_horizontally(panels, bg, PANEL_SEPARATOR)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in three colored regions into separate cropped result panels.

    Input structure:
    - Black background with three large foreground regions, arranged visually as
      left, right, and bottom masses.
    - The regions contain black dents, plus a few one-cell foreground specks in
      the background or along region boundaries.

    Rule:
    - First clean one-cell corruption locally: isolated specks revert to black,
      and a lone boundary intrusion flips back to the unique neighboring
      foreground majority.
    - Detect each original black dent whose cleaned 3x3 neighborhood is
      dominated by one region color, then stamp a 3x3 patch in the cyclic target
      color left -> right -> bottom -> left and recolor the dent center teal.
    - Do not return the full transformed canvas. Instead, isolate each host
      region's transformed result, crop it to a tight bounding box, and place
      the three crops side by side with solid separator columns.

    Key insight: The solver must combine two inferences: first infer the
    left/right/bottom role cycle for dent recoloring, then realize the answer is
    a repacked triptych of per-region transformed panels rather than the original
    full-scene layout.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    return _build_output(clean, centers, colors, bg)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
TOP_RECT_H_FRAC = (0.58, 0.80)
TOP_RECT_W_FRAC = (0.46, 0.62)
BOTTOM_RECT_H_FRAC = (0.36, 0.52)
BOTTOM_RECT_W_FRAC = (0.58, 0.82)
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 85
COVERAGE_MIN, COVERAGE_MAX = 0.60, 0.84
LEFT_RIGHT_OVERLAP_MIN = 2
BOTTOM_OVERLAP_MIN = 10

# --- CFG: dent placement ---
DENTS_PER_ROLE_MIN, DENTS_PER_ROLE_MAX = 4, 6
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.14

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 8, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.80
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 450


def _rand_span(N: int, frac_range: tuple[float, float], rng: random.Random) -> int:
    lo = max(1, int(round(N * frac_range[0])))
    hi = max(lo, int(round(N * frac_range[1])))
    return rng.randint(lo, hi)


def _build_regions(N: int, role_colors: dict[str, int], rng: random.Random) -> np.ndarray | None:
    left = int(role_colors["left"])
    right = int(role_colors["right"])
    bottom = int(role_colors["bottom"])

    for _ in range(60):
        l_h = _rand_span(N, TOP_RECT_H_FRAC, rng)
        l_w = _rand_span(N, TOP_RECT_W_FRAC, rng)
        r_h = _rand_span(N, TOP_RECT_H_FRAC, rng)
        r_w = _rand_span(N, TOP_RECT_W_FRAC, rng)
        b_h = _rand_span(N, BOTTOM_RECT_H_FRAC, rng)
        b_w = _rand_span(N, BOTTOM_RECT_W_FRAC, rng)

        l_r0 = rng.randint(0, RECT_MARGIN_MAX)
        l_c0 = rng.randint(0, RECT_MARGIN_MAX)
        r_r0 = rng.randint(0, RECT_MARGIN_MAX)
        r_c0 = max(0, N - r_w - rng.randint(0, RECT_MARGIN_MAX))
        b_r0 = max(0, N - b_h - rng.randint(0, RECT_MARGIN_MAX))
        b_center_jitter = rng.randint(-max(1, N // 10), max(1, N // 10))
        b_c0 = (N - b_w) // 2 + b_center_jitter
        b_c0 = min(max(0, b_c0), max(0, N - b_w))

        l_r1, l_c1 = min(N, l_r0 + l_h), min(N, l_c0 + l_w)
        r_r1, r_c1 = min(N, r_r0 + r_h), min(N, r_c0 + r_w)
        b_r1, b_c1 = min(N, b_r0 + b_h), min(N, b_c0 + b_w)

        lr_overlap_h = max(0, min(l_r1, r_r1) - max(l_r0, r_r0))
        lr_overlap_w = max(0, min(l_c1, r_c1) - max(l_c0, r_c0))
        left_bottom_overlap = max(0, min(l_r1, b_r1) - max(l_r0, b_r0)) * max(
            0, min(l_c1, b_c1) - max(l_c0, b_c0)
        )
        right_bottom_overlap = max(0, min(r_r1, b_r1) - max(r_r0, b_r0)) * max(
            0, min(r_c1, b_c1) - max(r_c0, b_c0)
        )
        if lr_overlap_h <= 0 or lr_overlap_w < LEFT_RIGHT_OVERLAP_MIN:
            continue
        if left_bottom_overlap < BOTTOM_OVERLAP_MIN or right_bottom_overlap < BOTTOM_OVERLAP_MIN:
            continue

        grid = np.zeros((N, N), dtype=int)
        grid[l_r0:l_r1, l_c0:l_c1] = int(left)
        grid[r_r0:r_r1, r_c0:r_c1] = int(right)
        grid[b_r0:b_r1, b_c0:b_c1] = int(bottom)

        counts = {int(color): int(np.sum(grid == int(color))) for color in role_colors.values()}
        if min(counts.values()) < MIN_REGION_CELLS:
            continue
        coverage = sum(counts.values()) / float(N * N)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        if _role_colors(grid, _main_colors(grid, Color.BLACK)) != role_colors:
            continue
        return grid
    return None


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, colors: tuple[int, int, int]) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    counts[int(host)] -= 1
    host_count = counts[int(host)]
    other_max = max(counts[int(color)] for color in colors if int(color) != int(host))
    return host_count >= DENT_MIN_HOST_3x3 and host_count > other_max


def _pick_dent_centers(
    valid: dict[int, list[tuple[int, int]]], wanted: dict[int, int], rng: random.Random
) -> dict[int, list[tuple[int, int]]] | None:
    chosen: dict[int, list[tuple[int, int]]] = {int(color): [] for color in valid}
    occupied: list[tuple[int, int]] = []
    order = sorted(valid, key=lambda color: len(valid[int(color)]))

    for color in order:
        candidates = valid[int(color)].copy()
        rng.shuffle(candidates)
        for r, c in candidates:
            if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in occupied):
                chosen[int(color)].append((int(r), int(c)))
                occupied.append((int(r), int(c)))
                if len(chosen[int(color)]) >= int(wanted[int(color)]):
                    break
        if len(chosen[int(color)]) < int(wanted[int(color)]):
            return None
    return chosen


def _expand_to_pairs(
    dents: dict[int, list[tuple[int, int]]],
    clean: np.ndarray,
    colors: tuple[int, int, int],
    rng: random.Random,
) -> dict[int, list[list[tuple[int, int]]]]:
    h, w = clean.shape
    occupied = {(r, c) for centers in dents.values() for r, c in centers}
    groups: dict[int, list[list[tuple[int, int]]]] = {int(color): [] for color in dents}

    for color, centers in dents.items():
        for r, c in centers:
            cells = [(int(r), int(c))]
            if rng.random() < DENT_PAIR_PROB:
                neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
                rng.shuffle(neighbors)
                for nr, nc in neighbors:
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in occupied:
                        continue
                    if int(clean[nr, nc]) != int(color):
                        continue
                    if not _is_safe_dent_center(clean, nr, nc, int(color), colors):
                        continue
                    if any(
                        max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION
                        for rr, cc in occupied
                        if (rr, cc) != (r, c)
                    ):
                        continue
                    cells.append((int(nr), int(nc)))
                    occupied.add((int(nr), int(nc)))
                    break
            groups[int(color)].append(cells)
    return groups


def _flatten_groups(dent_groups: dict[int, list[list[tuple[int, int]]]]) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for host, groups in dent_groups.items():
        for cells in groups:
            for r, c in cells:
                centers.append((int(r), int(c), int(host)))
    return centers


def _nearby_speck_color(
    clean: np.ndarray, r: int, c: int, colors: tuple[int, int, int], rng: random.Random
) -> int:
    nearby = [int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors]
    if nearby:
        return int(_Counter(nearby).most_common(1)[0][0])
    return int(rng.choice(list(colors)))


def _add_bg_noise(
    inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int, int], rng: random.Random
) -> int:
    bg_candidates = list(np.argwhere(clean == int(Color.BLACK)))
    rng.shuffle(bg_candidates)

    placed = 0
    for r, c in bg_candidates:
        if placed >= n_in_bg:
            break
        rr = int(r)
        cc = int(c)
        if int(inp[rr, cc]) != int(Color.BLACK):
            continue
        if int(np.sum(_sub3x3(inp, rr, cc) == int(Color.BLACK))) < BG_SPECK_MIN_BG_3x3:
            continue
        inp[rr, cc] = _nearby_speck_color(clean, rr, cc, colors, rng)
        placed += 1
    return placed


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if bool(np.all(block == int(color))):
                    return True
    return False


def _boundary_noise_safe(
    inp: np.ndarray,
    r: int,
    c: int,
    host: int,
    other: int,
    colors: tuple[int, int, int],
    bg: int,
) -> bool:
    if int(inp[r, c]) != int(host):
        return False

    sub_now = _sub3x3(inp, r, c)
    host_count = int(np.sum(sub_now == int(host)))
    other_count = int(np.sum(sub_now == int(other)))
    if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
        return False
    if _would_form_2x2(inp, r, c, other):
        return False

    test = inp.copy()
    test[r, c] = int(other)
    if _clean_cell(test, r, c, colors, bg) != int(host):
        return False

    r0 = max(0, r - 1)
    r1 = min(inp.shape[0], r + 2)
    c0 = max(0, c - 1)
    c1 = min(inp.shape[1], c + 2)
    for rr in range(r0, r1):
        for cc in range(c0, c1):
            before = _clean_cell(inp, rr, cc, colors, bg)
            after = _clean_cell(test, rr, cc, colors, bg)
            if (rr, cc) == (r, c):
                if after != int(host):
                    return False
            elif after != before:
                return False
    return True


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_at_boundary: int,
    colors: tuple[int, int, int],
    rng: random.Random,
) -> int:
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            host = int(clean[r, c])
            if host not in colors:
                continue
            nearby = sorted({int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors and int(x) != host})
            for other in nearby:
                candidates.append((int(r), int(c), int(host), int(other)))
    rng.shuffle(candidates)

    placed = 0
    for r, c, host, other in candidates:
        if placed >= n_at_boundary:
            break
        if _boundary_noise_safe(inp, r, c, host, other, colors, Color.BLACK):
            inp[r, c] = int(other)
            placed += 1
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [
        int(c)
        for c in Color.NOT_BLACK
        if int(c) not in (int(Color.TEAL), int(PANEL_SEPARATOR))
    ]
    left_color, right_color, bottom_color = rng.sample(fg_pool, 3)
    role_colors = {"left": left_color, "right": right_color, "bottom": bottom_color}

    clean = _build_regions(N, role_colors, rng)
    if clean is None:
        return None

    colors = _main_colors(clean, Color.BLACK)
    valid: dict[int, list[tuple[int, int]]] = {int(color): [] for color in colors}
    for r in range(N):
        for c in range(N):
            host = int(clean[r, c])
            if host in valid and _is_safe_dent_center(clean, r, c, host, colors):
                valid[host].append((int(r), int(c)))

    wanted: dict[int, int] = {}
    for color in colors:
        wanted[int(color)] = rng.randint(DENTS_PER_ROLE_MIN, DENTS_PER_ROLE_MAX)
        if len(valid[int(color)]) < wanted[int(color)] + 2:
            return None

    dents = _pick_dent_centers(valid, wanted, rng)
    if dents is None:
        return None

    dent_groups = _expand_to_pairs(dents, clean, colors, rng)
    centers = _flatten_groups(dent_groups)
    output = _build_output(clean, centers, colors, Color.BLACK)

    inp = clean.copy()
    for r, c, _host in centers:
        inp[r, c] = int(Color.BLACK)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    placed_bg = _add_bg_noise(inp, clean, n_bg, colors, rng)
    placed_bd = _add_boundary_noise(inp, clean, n_noise - placed_bg, colors, rng)
    if placed_bg + placed_bd < max(6, n_noise - 3):
        return None

    produced = main(inp)
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
