# Combo variant of de809cff (index 20)
# combo_spec: D8/2+D15/1
# Name: 背景散落像素提供映射线索+色斑指示激活条件

from __future__ import annotations

# Design analysis for the D8/2 + D15/1 combo
# -------------------------------------------
# a) Concrete behavior and trace:
#    The base task always removes isolated foreground pixels in the black background
#    before processing every black dent. This combo flips the meaning of those
#    isolated background pixels: they are now persistent control symbols, and their
#    colors encode which dent family is active.
#    We use a single fused rule: a background speck's color equals the 3x3 patch
#    color that may appear in the output. Since the base transformation stamps the
#    opposite foreground color, a speck of color `b` activates dents hosted by the
#    `a` region, while a speck of color `a` activates dents hosted by the `b`
#    region.
#    Small trace:
#      input fragment with only a `b` control speck:
#        0 0 0 0        a a a a
#        0 b 0 0   and  a 0 a a
#        0 0 0 0        a a a a
#      old task:
#        - the `b` speck is deleted as noise
#        - every valid black dent in both regions is patched
#      new task:
#        - the `b` speck stays visible as a legend pixel
#        - only dents in the `a` host region are patched to color `b`
#        - dents in the `b` host region remain black holes
#    The color clue and the activation condition are therefore inseparable: the same
#    background symbol both says "use patch color `b`" and "only the opposite host
#    may transform".
# b) Complexity calibration:
#    To keep the variant legible from 2-4 ARC examples, there are still only two
#    foreground colors plus teal. The new freedom is limited to three control states:
#    only color-a specks present, only color-b specks present, or both present.
#    Each sample repeats every active control color with 2-4 isolated specks, so the
#    legend is supported by multiple copies rather than a single accidental pixel.
#    Dents are still present in both regions even when only one side is active, which
#    makes the activation effect visually obvious.
# c) Hidden assumptions broken from the base solver:
#    - `_clean_foreground` used to assume every isolated foreground pixel in the
#      background was disposable noise. That is false now, so the solver must detect
#      control specks first and reinsert them after cleanup.
#    - The base `_dent_centers` implicitly assumed all valid dents should fire. That
#      fails under D15/1; now a dent also requires its host color to be activated by
#      an opposite-color background legend.
#    - The base output started from the cleaned grid only. That would silently erase
#      the new symbolic controls, so the final writeback must preserve them.
# d) Placement legality:
#    The generator keeps all dent centers at least Chebyshev distance 3 apart so
#    3x3 patches and teal centers remain separable. Control specks are placed only on
#    interior background cells with black-dominant 3x3 neighborhoods and are kept
#    away from every dent and every other control speck. Boundary contamination is
#    added only on cells that the cleanup pass deterministically restores.
# e) Visual observability:
#    A human can infer the rule because some samples visibly leave one region's black
#    holes untouched while the opposite region transforms, and the preserved
#    background specks always match the patch color that does appear. When both speck
#    colors are present, both dent families fire, which reinforces the color-based
#    activation reading.
# f) Visual cleanliness:
#    There is no disposable foreground noise in the background anymore; every
#    background singleton is meaningful. The legend pixels are repeated but sparse,
#    and they never overlap any 3x3 dent patch, so the scene does not devolve into a
#    cluttered mixture of unrelated singletons and patches.
# g) Color-change inferability:
#    No logic depends on numeric color ids. The visible invariant is: "a background
#    legend pixel shows the patch color, so it activates dents in the other large
#    region." With only two foreground colors, the opposite-host mapping is easy to
#    infer across a few examples.
# 9) New solver assumptions audited against the generator:
#    - Control specks are isolated interior background pixels whose clipped 3x3 is
#      still strongly black-majority. The generator enforces that, so the solver can
#      distinguish them from boundary contamination.
#    - True dents are still original black cells whose cleaned 3x3 has a unique
#      foreground majority of at least 5 cells. The generator uses the same legality
#      test when placing dents.
#    - Control specks never fall inside any dent's 3x3 patch footprint. The generator
#      reserves those cells, so keeping controls in the output cannot conflict with a
#      later stamped patch or teal center.

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(color: int, colors: tuple[int, int]) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


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

    The cleanup itself is unchanged from the base task: symbolic background specks
    are deliberately erased here and reinserted later after they have served as a
    control legend.
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
                    nxt[r, c] = int(bg)
                    continue

                sub = _sub3x3(cur, r, c)
                own = int(np.sum(sub == v))
                other = b if v == a else a
                other_count = int(np.sum(sub == other))
                bg_count = int(np.sum(sub == int(bg)))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = v
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = other
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = int(bg)
                elif own <= 2:
                    nxt[r, c] = int(bg)
                else:
                    nxt[r, c] = v
        cur = nxt

    return cur


def _is_control_speck(g: np.ndarray, r: int, c: int, colors: tuple[int, int], bg: int) -> bool:
    if not (1 <= r < g.shape[0] - 1 and 1 <= c < g.shape[1] - 1):
        return False
    v = int(g[r, c])
    if v not in colors:
        return False
    if _has_same_2x2(g, r, c, v):
        return False

    sub = _sub3x3(g, r, c)
    own = int(np.sum(sub == v))
    other = _other_color(v, colors)
    other_count = int(np.sum(sub == int(other)))
    bg_count = int(np.sum(sub == int(bg)))
    if own > 2:
        return False
    if other_count > 1:
        return False
    return bg_count >= 6


def _find_control_specks(g: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    controls: list[tuple[int, int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if _is_control_speck(g, r, c, colors, bg):
                controls.append((int(r), int(c), int(g[r, c])))
    return controls


def _active_hosts_from_controls(
    colors: tuple[int, int], controls: list[tuple[int, int, int]]
) -> set[int]:
    active_patch_colors = {int(color) for _r, _c, color in controls}
    return {int(_other_color(color, colors)) for color in active_patch_colors}


def _dent_centers(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    active_hosts: set[int],
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = _other_color(int(host), colors)
        if (
            int(host) in active_hosts
            and counts[int(host)] >= 5
            and counts[int(host)] > counts[int(other)]
        ):
            centers.append((rr, cc, int(host)))
    return centers


def _render_output(
    original: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    force_all_active: bool = False,
) -> np.ndarray:
    controls = _find_control_specks(original, colors, bg)
    clean = _clean_foreground(original, colors, bg)
    active_hosts = set(map(int, colors)) if force_all_active else _active_hosts_from_controls(colors, controls)

    out = clean.copy()
    centers = _dent_centers(original, clean, colors, bg, active_hosts)
    h, w = original.shape
    for r, c, host in centers:
        stamp_color = _other_color(host, colors)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)

    for r, c, _host in centers:
        out[r, c] = int(Color.TEAL)

    for r, c, color in controls:
        out[r, c] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use background legend specks to decide which dents should expand.

    Input structure:
    - Black background with two large foreground colors and some black dents inside
      both regions.
    - The background may also contain a few isolated one-cell foreground specks; in
      this variant they are not noise but repeated control markers.
    - A small amount of one-cell boundary contamination can still appear where the
      two foreground regions meet.

    Rule:
    - Read every isolated foreground speck in the background as a control symbol,
      then locally clean the two foreground regions exactly as in the base task:
      boundary contamination is repaired and isolated foreground singletons collapse
      back to black.
    - A control speck's color specifies the patch color that is allowed to appear.
      Because dents still stamp the opposite foreground color, a speck of color A
      activates only dents whose host region is color B, and vice versa.
    - For each original black dent whose cleaned 3x3 neighborhood has a unique
      active foreground majority of at least 5 cells, paint its 3x3 neighborhood in
      the opposite color and recolor the dent center teal (8). Dents in inactive
      regions stay black. Finally, restore the control specks to their original
      background positions.

    Key insight:
    - The isolated background pixels are a legend, not corruption. Their colors match
      the patch colors that do appear, so they simultaneously reveal the mapping and
      gate which host region is allowed to transform.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    return _render_output(g, colors, bg)


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3x3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.13

# --- CFG: symbolic controls & boundary noise ---
SINGLE_ACTIVE_PROB = 0.7
CONTROL_SPECKS_PER_COLOR_MIN, CONTROL_SPECKS_PER_COLOR_MAX = 2, 4
CONTROL_MIN_BG_3x3 = 6
CONTROL_MIN_NEARBY_FG_5x5 = 2
BOUNDARY_NOISE_MIN, BOUNDARY_NOISE_MAX = 2, 8
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


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
        grid[r0:r1, c0:c1] = int(color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        from scipy.ndimage import label

        _, n_components = label(grid == int(rect_colors[0]))
        if n_components < 2:
            return None
    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    return host_count >= DENT_MIN_HOST_3x3 and host_count > int(np.sum(sub == int(other)))


def _pick_dent_centers(
    candidates: list[tuple[int, int]],
    n: int,
    rng: random.Random,
    blocked: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    rng.shuffle(candidates)
    chosen = [] if blocked is None else list(blocked)
    result: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_PATCH_SEPARATION for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            result.append((int(r), int(c)))
            if len(result) >= n:
                return result
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_reserved: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    reserved = list(all_reserved)
    result: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(int(r), int(c))]
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
                if all(
                    max(abs(nr - r2), abs(nc - c2)) >= DENT_PATCH_SEPARATION
                    for r2, c2 in reserved
                    if (r2, c2) != (r, c)
                ):
                    cells.append((int(nr), int(nc)))
                    reserved.append((int(nr), int(nc)))
                    break
        result.append(cells)
    return result


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


def _stamp_active_dents(
    clean: np.ndarray,
    dent_groups_by_host: dict[int, list[list[tuple[int, int]]]],
    colors: tuple[int, int],
    active_hosts: set[int],
) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    for host, dent_groups in dent_groups_by_host.items():
        if int(host) not in active_hosts:
            continue
        stamp_color = _other_color(int(host), colors)
        for cells in dent_groups:
            for r, c in cells:
                out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
        for cells in dent_groups:
            for r, c in cells:
                out[r, c] = int(Color.TEAL)
    return out


def _reserved_patch_cells(groups: list[list[tuple[int, int]]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    reserved: set[tuple[int, int]] = set()
    for cells in groups:
        for r, c in cells:
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    reserved.add((int(rr), int(cc)))
    return reserved


def _expand_reserved(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                expanded.add((int(rr), int(cc)))
    return expanded


def _choose_activation_patch_colors(colors: tuple[int, int], rng: random.Random) -> set[int]:
    if rng.random() < SINGLE_ACTIVE_PROB:
        return {int(rng.choice(list(colors)))}
    return {int(colors[0]), int(colors[1])}


def _control_candidate_score(clean: np.ndarray, r: int, c: int) -> int:
    r0, r1 = max(0, r - 2), min(clean.shape[0], r + 3)
    c0, c1 = max(0, c - 2), min(clean.shape[1], c + 3)
    window = clean[r0:r1, c0:c1]
    return int(np.sum(window != 0))


def _place_control_specks(
    inp: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    active_patch_colors: set[int],
    reserved_cells: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int]] | None:
    h, w = inp.shape
    controls: list[tuple[int, int, int]] = []
    occupied = set(reserved_cells)

    for patch_color in active_patch_colors:
        candidates: list[tuple[int, int, int]] = []
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if int(clean[r, c]) != 0 or (r, c) in occupied:
                    continue
                sub = _sub3x3(clean, r, c)
                if int(np.sum(sub == 0)) < CONTROL_MIN_BG_3x3:
                    continue
                if _control_candidate_score(clean, r, c) < CONTROL_MIN_NEARBY_FG_5x5:
                    continue
                probe = inp.copy()
                probe[r, c] = int(patch_color)
                if not _is_control_speck(probe, r, c, colors, 0):
                    continue
                candidates.append((int(_control_candidate_score(clean, r, c)), int(r), int(c)))

        candidates.sort(reverse=True)
        need = rng.randint(CONTROL_SPECKS_PER_COLOR_MIN, CONTROL_SPECKS_PER_COLOR_MAX)
        placed = 0
        for _score, r, c in candidates:
            if placed >= need:
                break
            if any(max(abs(r - rr), abs(c - cc)) <= 2 for rr, cc in occupied):
                continue
            inp[r, c] = int(patch_color)
            controls.append((int(r), int(c), int(patch_color)))
            occupied.add((int(r), int(c)))
            placed += 1
        if placed < need:
            return None

    return controls


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    n_noise: int,
    protected: set[tuple[int, int]],
    rng: random.Random,
) -> int:
    color_a, color_b = map(int, colors)
    h, w = clean.shape
    boundary_candidates: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in colors or (r, c) in protected:
                continue
            other = color_b if v == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((int(r), int(c), int(v), int(other)))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_noise:
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
        inp[r, c] = int(other)
        placed += 1
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    color_a, color_b = map(int, rng.sample(fg_pool, 2))
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a, n_b = n_total // 2, n_total - n_total // 2
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    valid = {color_a: [], color_b: []}
    for r in range(clean.shape[0]):
        for c in range(clean.shape[1]):
            v = int(clean[r, c])
            if v == color_a and _is_safe_dent_center(clean, r, c, color_a, color_b):
                valid[color_a].append((int(r), int(c)))
            elif v == color_b and _is_safe_dent_center(clean, r, c, color_b, color_a):
                valid[color_b].append((int(r), int(c)))

    if len(valid[color_a]) < n_a + 2 or len(valid[color_b]) < n_b + 2:
        return None

    dents_a = _pick_dent_centers(valid[color_a], n_a, rng)
    if dents_a is None:
        return None
    dents_b = _pick_dent_centers(valid[color_b], n_b, rng, blocked=dents_a)
    if dents_b is None:
        return None

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, dents_a + dents_b, rng)
    all_reserved = dents_a + dents_b + [cell for group in groups_a for cell in group[1:]]
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, all_reserved, rng)
    dent_groups_by_host = {int(color_a): groups_a, int(color_b): groups_b}
    all_groups = groups_a + groups_b

    inp = clean.copy()
    for cells in all_groups:
        for r, c in cells:
            inp[r, c] = 0

    patch_reserved = _reserved_patch_cells(all_groups, clean.shape)
    active_patch_colors = _choose_activation_patch_colors(colors, rng)
    controls = _place_control_specks(inp, clean, colors, active_patch_colors, patch_reserved, rng)
    if controls is None:
        return None

    control_cells = {(int(r), int(c)) for r, c, _color in controls}
    protected = set(patch_reserved) | _expand_reserved(control_cells, clean.shape, radius=1)
    n_noise = rng.randint(BOUNDARY_NOISE_MIN, BOUNDARY_NOISE_MAX)
    _add_boundary_noise(inp, clean, colors, n_noise, protected, rng)

    produced = main(inp)
    detected_controls = _find_control_specks(inp, colors, 0)
    detected_patch_colors = {int(color) for _r, _c, color in detected_controls}
    if detected_patch_colors != set(active_patch_colors):
        return None

    active_hosts = _active_hosts_from_controls(colors, detected_controls)
    clean_after_noise = _clean_foreground(inp, colors, 0)
    detected_centers = _dent_centers(inp, clean_after_noise, colors, 0, active_hosts)
    if len(detected_centers) < 4:
        return None
    host_counter = _Counter(int(host) for _r, _c, host in detected_centers)
    if any(host_counter.get(int(host), 0) < 2 for host in active_hosts):
        return None

    # Reject degenerate cases where the control legend color set fails to matter.
    counterfactual = _render_output(inp, colors, 0, force_all_active=True)
    if len(active_hosts) == 1 and np.array_equal(counterfactual, produced):
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
