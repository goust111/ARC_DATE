# Variant of de809cff
# Dimension: D10 — 补丁大小 (transform)
# Change: 固定为3x3的补丁（以凹痕为中心的9像素方块）... → 补丁大小由凹痕到最近区域边界的距离决定

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import Color

# -----------------------------------------------------------------------------
# Design analysis for the D10 variant
#
# concretization:
#   - The two large foreground regions and the black background remain unchanged.
#   - The input still contains isolated wrong-color specks and cross-boundary
#     contamination, so the cleanup stage remains necessary.
#   - The changed object is the opposite-color patch produced around each dent.
#     In the original task every valid dent produced a fixed 3x3 square.
#     Here, each dent produces the largest odd square supported by the host
#     region around that dent: if the dent has one full same-color ring around
#     it, output 3x3; two rings, output 5x5; three rings, output 7x7.
#   - Concretely, after cleanup, a dent at (r,c) hosted by color A has radius d
#     iff every cell in the square [r-d:r+d, c-d:c+d] except the center is A,
#     and radius d+1 fails. The output stamps a (2d+1)x(2d+1) square in color B
#     and then recolors the center teal.
#
# behavior trace:
#   - Original logic at a dent: detect host from cleaned 3x3 majority -> always
#     stamp radius 1.
#   - New logic at a dent: detect host from cleaned 3x3 majority -> expand the
#     checked square while all non-center cells stay host-colored -> stamp the
#     resulting radius d.
#   - Example: a dent with host rings of widths 1, 2, 3 behaves as 3x3, 5x5,
#     7x7 respectively. This is the only semantic change in the transform.
#
# complexity calibration:
#   - If the generator allowed only radius-1 dents, the variant would visually
#     collapse back to the original task. To keep D10 observable, each sample is
#     forced to include several dents with radius > 1 and at least one 7x7 case.
#   - To avoid introducing unrelated ambiguity, the generator only uses three
#     repeated tiers (radii 1/2/3), not a wide unconstrained range.
#
# hidden assumptions reviewed:
#   - main() originally assumed every detected dent should stamp exactly one
#     clipped 3x3 square. That assumption fails under D10, so the solver now
#     computes a per-dent radius before stamping.
#   - The old dent detector accepted boundary dents with only 5 host cells in
#     the 3x3 neighborhood. Under D10 that would create ambiguous "distance"
#     values. The new solver still uses the 3x3 majority to infer the host, but
#     only accepts dents whose host support expands to at least one full ring.
#   - Because patch sizes now vary, overlapping large patches would hide the
#     scaling cue. The generator therefore spaces dents so their stamped squares
#     do not overlap.
#
# placement legality:
#   - Candidate dent cells are chosen from the clean regions before blackening.
#     Their exact support radius is measured on the clean grid.
#   - Dents are selected with Chebyshev spacing greater than the sum of their
#     radii, so opposite-color squares stay disjoint and fully inside the host
#     region that justified the radius.
#   - Noise placement skips protected neighborhoods around dents so cleanup
#     remains stable and the solver's inferred radii match the intended output.
#
# observability / cleanliness:
#   - Both foreground colors still participate, and every generated sample
#     contains repeated radius tiers. This gives the human solver multiple
#     examples of "deeper dent -> larger odd square" instead of a one-off.
#   - Patches remain monochrome with a teal center, so the output stays visually
#     tidy even though patch sizes now vary.
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


def _square_support_radius(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """Largest radius whose centered square is all host-colored except the dent center."""
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
        radius = _square_support_radius(clean, rr, cc, int(host))
        if radius >= 1:
            specs.append((rr, cc, int(host), int(radius)))
    return specs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents in two colored regions into size-scaled opposite-color patches.

    Input structure:
    - Black background with two large foreground regions in distinct colors.
    - The regions contain black dents plus one-cell foreground specks and
      one-cell cross-boundary contamination that should be cleaned first.

    Rule:
    - First denoise the two foreground regions exactly as in the original task:
      isolated specks fall back to black, and one-cell protrusions revert to the
      locally dominant neighboring region.
    - For each original black dent, infer which foreground color hosts it from
      the cleaned local majority, then measure how many full same-color square
      rings surround the dent center.
    - Stamp the opposite foreground color on the centered odd square whose side
      length is 2*d+1, where d is that measured ring depth, and recolor the
      dent center itself teal (8).

    Key insight: The output patch is no longer fixed-size; it is the largest
    centered odd square that the cleaned host region can support around the
    original defect, with teal preserving the defect location.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)

    out = clean.copy()
    dent_specs = _dent_specs(g, clean, colors, bg)
    h, w = g.shape
    for r, c, host, radius in dent_specs:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        out[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)] = int(
            other
        )

    for r, c, _host, _radius in dent_specs:
        out[r, c] = Color.TEAL

    return out


# --------------- generator (cx) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 8, 12
TARGET_RADII = (1, 2, 3)
REQUIRED_RADIUS3_COUNT = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400


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


def _full_square_radius(clean: np.ndarray, r: int, c: int, host: int) -> int:
    """Largest radius whose centered square is entirely host-colored, including the center."""
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
            v = int(clean[r, c])
            if v not in buckets:
                continue
            radius = _full_square_radius(clean, r, c, v)
            if radius in TARGET_RADII:
                buckets[v][int(radius)].append((int(r), int(c)))
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
        if (int(r), int(c)) in used:
            continue
        if _far_enough(selected, int(r), int(c), int(radius)):
            used.add((int(r), int(c)))
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
        (int(color_a), 1),
        (int(color_a), 2),
        (int(color_b), 1),
        (int(color_b), 2),
    ]
    for host, radius in required:
        spec = _pick_from_bucket(buckets[host][radius], radius, host, selected, used, rng)
        if spec is None:
            return None
        selected.append(spec)

    radius3_pool = [
        (int(host), int(r), int(c))
        for host in (int(color_a), int(color_b))
        for r, c in buckets[host][3]
        if (int(r), int(c)) not in used
    ]
    rng.shuffle(radius3_pool)
    radius3_added = 0
    for host, r, c in radius3_pool:
        if radius3_added >= REQUIRED_RADIUS3_COUNT:
            break
        if not _far_enough(selected, int(r), int(c), 3):
            continue
        used.add((int(r), int(c)))
        selected.append((int(r), int(c), int(host), 3))
        radius3_added += 1
    if radius3_added < REQUIRED_RADIUS3_COUNT:
        return None

    target_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    extras = [
        (radius, int(host), int(r), int(c))
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

    if len(selected) < max(6, DENT_COUNT_MIN):
        return None
    return selected


def _stamp_dents(clean: np.ndarray, dent_specs: list[tuple[int, int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c, host, radius in dent_specs:
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        output[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)] = int(other)
    for r, c, _host, _radius in dent_specs:
        output[r, c] = Color.TEAL
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
    h, w = inp.shape
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
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = int(color_b) if v == int(color_a) else int(color_a)
            if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
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
    if any(not buckets[int(color_a)][radius] for radius in (1, 2)):
        return None
    if any(not buckets[int(color_b)][radius] for radius in (1, 2)):
        return None
    if len(buckets[int(color_a)][3]) + len(buckets[int(color_b)][3]) < REQUIRED_RADIUS3_COUNT:
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
