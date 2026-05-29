# Combo variant of de809cff (index 17)
# combo_spec: D1/4+D10/0
# Name: 嵌套框架补丁+补丁大小随距离变化

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import Color

# -----------------------------------------------------------------------------
# Design analysis for the combo variant D1/4 + D10/0
#
# concretization:
#   - The original scene grammar stays intact: two large foreground regions sit
#     on black background, and the input still contains black dents plus
#     one-cell foreground noise that must be cleaned first.
#   - D1/4 replaces the old solid opposite-color 3x3 dent patch with a nested
#     patch: the outer square frame is the opposite foreground color, the inner
#     square is restored to the dent's host color, and the exact dent center is
#     teal. For the smallest legal case this is:
#         O O O O O
#         O H H H O
#         O H 8 H O
#         O H H H O
#         O O O O O
#     where H is the host region color and O is the opposite region color.
#   - D10/0 now controls the overall side length of that nested patch. To keep
#     the visual tiers few and readable, the solver maps host support depth to
#     three odd sizes only:
#       full host square radius 2 -> 5x5 nested patch
#       full host square radius 3 -> 7x7 nested patch
#       full host square radius 4+ -> 9x9 nested patch
#   - "Distance to boundary" is therefore operationalized as the largest clean
#     same-color square centered on the dent. This is the same observable cue as
#     "deeper inside the region means larger patch", but now shifted upward so
#     the smallest valid fused patch is 5x5 instead of 3x3.
#
# behavior trace:
#   - Original task:
#       detect dent host from cleaned 3x3 majority
#       -> always stamp a solid opposite-color 3x3 block
#       -> recolor center teal
#   - Fused task:
#       detect dent host from cleaned 3x3 majority
#       -> measure the clean centered host square radius around the dent
#       -> choose radius 2/3/4 (capped) for a 5x5/7x7/9x9 patch
#       -> paint the outer square in the opposite color
#       -> repaint the inner square in the host color
#       -> recolor the center teal
#   - Concrete example: if a dent sits in a host area that supports a clean 7x7
#     centered square but not a 9x9 one, the original code would still emit one
#     3x3 opposite-color block, while the fused code emits a 7x7 nested patch:
#     an opposite-color border, a 5x5 host-colored interior, and teal at center.
#
# complexity calibration:
#   - If the generator allowed radius-1 dents, the output could collapse toward
#     the original visual pattern. Those dents are disallowed; every kept dent
#     must support at least radius 2, so every fused sample visibly contains
#     5x5-or-larger structures.
#   - If the generator allowed arbitrary large radii, the task would add noisy
#     degrees of freedom. The size rule is therefore quantized to three tiers
#     only: 5x5, 7x7, 9x9.
#   - If large patches overlapped, the nested rings would become unreadable, so
#     dent centers are spaced by strictly more than the sum of their radii.
#
# hidden assumptions reviewed:
#   - The old solver assumed "every detected dent produces exactly one solid 3x3
#     opposite-color square". That is false now; the fused solver computes one
#     radius per dent and performs a two-stage stamp (outer opposite, inner host)
#     before marking teal.
#   - The old detector only needed a 3x3 local majority. That is still enough to
#     infer the host color, but it is no longer enough to determine patch size,
#     so the solver additionally probes larger centered host squares on the
#     cleaned grid.
#   - The original generator allowed 2-cell dent pairs because 3x3 solid patches
#     could tolerate some adjacency. Under nested 5x5-9x9 patches, pairs would
#     fuse into ambiguous composite marks, so this combo generator uses only
#     single-cell dents.
#   - The nested patch is only visually meaningful when its inner host square is
#     fully present. The generator therefore guarantees full host support for the
#     chosen tier instead of relying on clipped stamps at true region edges.
#
# placement legality:
#   - Dent candidates are chosen on the clean grid, and each candidate's maximum
#     supported square radius is measured before blackening the center.
#   - The generator only uses candidates whose capped patch radius is 2, 3, or
#     4; patch squares are fully inside the grid and their host region by
#     construction.
#   - A protected mask excludes noise near every chosen patch, so cleanup does
#     not alter the support radius that the solver later infers.
#
# observability / cleanliness / inferability:
#   - Every accepted sample contains repeated 5x5 and 7x7 examples across both
#     colors, plus at least one 9x9 case. That gives the human solver multiple
#     demonstrations of the distance-to-size rule instead of a single outlier.
#   - The color mapping is structurally hinted, not arbitrary: the outer border
#     flips to the opposite region color, the inner area returns to the local
#     host color, and teal always marks the original defect cell.
#   - Using only two foreground colors and three size tiers keeps the output
#     legible even with many dents; the nested patches read as crisp framed
#     markers rather than noisy multi-color blobs.
# -----------------------------------------------------------------------------


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
    Remove one-cell specks/protrusions before reading dent geometry.

    Two sweeps are enough for this corruption model: one to wipe isolated
    outliers, one to let their neighbors settle to the corrected local majority.
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


def _full_square_radius(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """Largest centered square radius whose cells are all `host`, including center."""
    if int(clean[r, c]) != int(host):
        return 0
    h, w = clean.shape
    radius = 0
    probe = 0
    while True:
        r0 = int(r) - probe
        c0 = int(c) - probe
        r1 = int(r) + probe + 1
        c1 = int(c) + probe + 1
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            break
        if not bool(np.all(clean[r0:r1, c0:c1] == int(host))):
            break
        radius = probe
        probe += 1
    return int(radius)


def _support_radius_ignoring_center(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """
    Largest centered square radius whose non-center cells are all `host`.

    This is the measurement the solver needs after the dent center has turned
    black in the input.
    """
    h, w = clean.shape
    radius = 0
    probe = 1
    while True:
        r0 = int(r) - probe
        c0 = int(c) - probe
        r1 = int(r) + probe + 1
        c1 = int(c) + probe + 1
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            break
        patch = clean[r0:r1, c0:c1]
        center_rr = int(r) - r0
        center_cc = int(c) - c0
        ok = True
        for rr in range(patch.shape[0]):
            for cc in range(patch.shape[1]):
                if rr == center_rr and cc == center_cc:
                    continue
                if int(patch[rr, cc]) != int(host):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
        radius = probe
        probe += 1
    return int(radius)


def _patch_radius(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """
    Convert clean host support depth into one of the fused patch radii.

    Radius 2 -> 5x5, radius 3 -> 7x7, radius 4 -> 9x9.
    Larger support is capped at 9x9 to keep the task visually discrete.
    """
    support = _support_radius_ignoring_center(clean, r, c, host)
    if support < 2:
        return 0
    return int(min(support, 4))


def _dent_specs(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int, int]]:
    h, w = original.shape
    specs: list[tuple[int, int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = clean[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)]
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] < 5 or counts[int(host)] <= counts[int(other)]:
            continue
        radius = _patch_radius(clean, rr, cc, int(host))
        if radius >= 2:
            specs.append((rr, cc, int(host), int(radius)))
    return specs


def _stamp_nested_patch(out: np.ndarray, r: int, c: int, host: int, other: int, radius: int) -> None:
    """Paint one nested square patch: outer opposite-color frame, inner host square, teal center."""
    h, w = out.shape
    out[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)] = int(other)
    inner = int(radius - 1)
    out[max(0, r - inner) : min(h, r + inner + 1), max(0, c - inner) : min(w, c + inner + 1)] = int(host)
    out[r, c] = Color.TEAL


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into size-scaled nested frame patches.

    Input structure:
    - Black background with two large foreground regions in distinct colors.
    - The regions contain black dents plus one-cell foreground specks and
      one-cell cross-boundary contamination that should be cleaned first.

    Rule:
    - First denoise the two foreground regions exactly as in the original task:
      isolated specks fall back to black, and one-cell protrusions revert to the
      locally dominant neighboring region.
    - For each original black dent, infer the host foreground color from the
      cleaned 3x3 majority, then measure the largest centered same-color square
      that the cleaned host region supports around that dent.
    - Convert that depth into a 5x5, 7x7, or 9x9 nested patch: the outer frame
      is the opposite foreground color, the next inner square is the host color,
      and the dent center itself becomes teal (8).

    Key insight: The output is determined by two coupled cues at once: color
    inversion happens only on the outer frame, while patch size increases with
    how deeply the dent sits inside its cleaned host region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    for r, c, host, radius in _dent_specs(g, clean, colors, bg):
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        _stamp_nested_patch(out, int(r), int(c), int(host), int(other), int(radius))

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.48, 0.78
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.86

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 7, 10
TARGET_RADII = (2, 3, 4)
REQUIRED_RADIUS4_COUNT = 1

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles with painter's-algorithm occlusion."""
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
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        _, n_components = label(grid == rect_colors[0])
        if int(n_components) < 2:
            return None
    return grid


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


def _extract_radius_buckets(clean: np.ndarray, color_a: int, color_b: int) -> dict[int, dict[int, list[tuple[int, int]]]]:
    buckets: dict[int, dict[int, list[tuple[int, int]]]] = {
        int(color_a): {radius: [] for radius in TARGET_RADII},
        int(color_b): {radius: [] for radius in TARGET_RADII},
    }
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host not in buckets:
                continue
            radius = _patch_radius(clean, int(r), int(c), int(host))
            if radius in TARGET_RADII:
                buckets[host][int(radius)].append((int(r), int(c)))
    return buckets


def _far_enough(selected: list[tuple[int, int, int, int]], r: int, c: int, radius: int) -> bool:
    for rr, cc, _host, old_radius in selected:
        if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) <= int(radius) + int(old_radius):
            return False
    return True


def _pick_from_bucket(
    bucket: list[tuple[int, int]],
    radius: int,
    host: int,
    selected: list[tuple[int, int, int, int]],
    used: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    choices = bucket[:]
    rng.shuffle(choices)
    for r, c in choices:
        rc = (int(r), int(c))
        if rc in used:
            continue
        if _far_enough(selected, int(r), int(c), int(radius)):
            used.add(rc)
            return int(r), int(c), int(host), int(radius)
    return None


def _select_dents(
    buckets: dict[int, dict[int, list[tuple[int, int]]]],
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    selected: list[tuple[int, int, int, int]] = []
    used: set[tuple[int, int]] = set()

    required = [
        (int(color_a), 2),
        (int(color_a), 3),
        (int(color_b), 2),
        (int(color_b), 3),
    ]
    for host, radius in required:
        spec = _pick_from_bucket(buckets[host][radius], radius, host, selected, used, rng)
        if spec is None:
            return None
        selected.append(spec)

    radius4_pool = [
        (int(host), int(r), int(c))
        for host in (int(color_a), int(color_b))
        for r, c in buckets[host][4]
        if (int(r), int(c)) not in used
    ]
    rng.shuffle(radius4_pool)
    radius4_added = 0
    for host, r, c in radius4_pool:
        if radius4_added >= REQUIRED_RADIUS4_COUNT:
            break
        if not _far_enough(selected, int(r), int(c), 4):
            continue
        used.add((int(r), int(c)))
        selected.append((int(r), int(c), int(host), 4))
        radius4_added += 1
    if radius4_added < REQUIRED_RADIUS4_COUNT:
        return None

    target_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    extras = [
        (int(radius), int(host), int(r), int(c))
        for host in (int(color_a), int(color_b))
        for radius in TARGET_RADII
        for r, c in buckets[host][radius]
        if (int(r), int(c)) not in used
    ]
    rng.shuffle(extras)
    for radius, host, r, c in extras:
        if len(selected) >= target_total:
            break
        if not _far_enough(selected, int(r), int(c), int(radius)):
            continue
        used.add((int(r), int(c)))
        selected.append((int(r), int(c), int(host), int(radius)))

    if len(selected) < 5:
        return None
    return selected


def _stamp_dents(clean: np.ndarray, dent_specs: list[tuple[int, int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    output = clean.copy()
    for r, c, host, radius in dent_specs:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        _stamp_nested_patch(output, int(r), int(c), int(host), int(other), int(radius))
    return output


def _build_protected_mask(shape: tuple[int, int], dent_specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    protected = np.zeros((h, w), dtype=bool)
    for r, c, _host, radius in dent_specs:
        pad = int(radius) + 1
        protected[max(0, r - pad) : min(h, r + pad + 1), max(0, c - pad) : min(w, c + pad + 1)] = True
    return protected


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    bg_candidates = list(zip(*np.where((clean == 0) & (~protected)), strict=False))
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


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            if protected[r, c]:
                continue
            host = int(clean[r, c])
            if host not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if host == int(color_a) else int(color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
                boundary_candidates.append((int(r), int(c), host, other))
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
        inp[r, c] = int(other)
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    buckets = _extract_radius_buckets(clean, color_a, color_b)
    if any(not buckets[int(color_a)][radius] for radius in (2, 3)):
        return None
    if any(not buckets[int(color_b)][radius] for radius in (2, 3)):
        return None
    if len(buckets[int(color_a)][4]) + len(buckets[int(color_b)][4]) < REQUIRED_RADIUS4_COUNT:
        return None

    dent_specs = _select_dents(buckets, color_a, color_b, rng)
    if dent_specs is None:
        return None

    colors = (int(color_a), int(color_b))
    output = _stamp_dents(clean, dent_specs, colors)

    inp = clean.copy()
    for r, c, _host, _radius in dent_specs:
        inp[r, c] = 0

    protected = _build_protected_mask(clean.shape, dent_specs)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, protected, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, protected, n_noise - n_in_bg, color_a, color_b, rng)

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
