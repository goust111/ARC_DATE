# Combo variant of de809cff (index 25)
# combo_spec: D4/3+D7/0
# Name: 大小不对称区域+凹痕仅在边界附近
#
# Design analysis
# 1. Concrete behavior:
#    Use one dominant rectangle-like foreground region plus one much smaller
#    rectangle embedded inside it. Example: on a 26x26 grid, the outer region
#    may occupy ~82% of all cells while the inner region occupies ~10%.
#    Compared with the original task, black dents are no longer allowed deep
#    inside either color. They are chosen only from host cells whose true clean
#    location is within Chebyshev distance 1-2 from a real region boundary.
#    After blackening those cells in the input, the output still stamps a 3x3
#    opposite-color patch and marks each dent center teal.
# 2. Complexity calibration:
#    The asymmetry is intentionally strong but uses only two foreground colors.
#    This keeps the color rule inferable: dense rare-color patches appear along
#    the long boundary band of the dominant region, while only a few common-color
#    patches appear near the short boundary band of the minor region.
# 3. Hidden assumptions audit:
#    - `_main_colors` assumes exactly two meaningful foreground colors; the
#      generator keeps the minor region far larger than any added noise.
#    - `_clean_foreground` relies on 2x2 support to preserve true regions; both
#      regions are rectangles, so valid cells retain such support even near dents.
#    - Dent detection must not accept deep black cells. The new `_is_boundary_dent`
#      therefore requires both local host majority in 3x3 and some additional
#      non-host evidence within radius 2, excluding the center itself.
# 4. Placement legality:
#    The minor rectangle stays fully inside the major rectangle with a gap of at
#    least 2 cells, so the major region remains connected. Dent centers are only
#    selected from host cells satisfying the same host-majority check used by the
#    solver and are spaced to avoid ambiguous overlaps. The generator is allowed
#    to omit optional noise so the visible variation budget stays on the fused
#    asymmetry + boundary-band behavior instead of being spent on cleanup artifacts.
# 5. Visual observability / non-degeneracy:
#    Humans can see one dominant color mass, one much smaller inserted region,
#    many dents near the long dominant boundary, and only a few near the minor
#    boundary. The generator is later validated with metrics: dominant visible
#    area stays high, minor visible area stays around one tenth, and all oracle
#    dents lie in the intended boundary band rather than the deep interior.

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub(grid: np.ndarray, r: int, c: int, radius: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)]


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

    for _pass in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = bg
                    continue

                sub = _sub(cur, r, c, 1)
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


def _is_boundary_dent(original: np.ndarray, clean: np.ndarray, r: int, c: int, host: int, other: int, bg: int) -> bool:
    if int(original[r, c]) != int(bg):
        return False

    sub3 = _sub(clean, r, c, 1)
    host_count = int(np.sum(sub3 == int(host)))
    other_count = int(np.sum(sub3 == int(other)))
    if host_count < 5 or host_count <= other_count:
        return False

    band = _sub(clean, r, c, 2)
    r0 = max(0, r - 2)
    c0 = max(0, c - 2)
    for rr in range(r0, min(clean.shape[0], r + 3)):
        for cc in range(c0, min(clean.shape[1], c + 3)):
            if rr == r and cc == c:
                continue
            if int(clean[rr, cc]) != int(host):
                return True
    return False


def _dent_centers(original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub3 = _sub(clean, rr, cc, 1)
        counts = {int(color): int(np.sum(sub3 == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if _is_boundary_dent(original, clean, rr, cc, int(host), int(other), bg):
            centers.append((rr, cc, int(host)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert boundary-band black dents into opposite-color 3x3 stamps with teal centers.

    Input structure:
    - Black background with exactly two foreground colors, but now their areas are
      strongly unbalanced: one dominant region covers most of the canvas and one
      noticeably smaller region occupies only a short embedded band.
    - Black dents appear only near genuine region boundaries. Inputs may also
      include a few isolated foreground specks or one-cell boundary contaminations
      that should be cleaned first.

    Rule:
    - First denoise the foreground locally: isolated specks revert to black and
      one-cell boundary protrusions revert to the surrounding region color.
    - Then inspect each original black cell. It counts as a dent only if its
      cleaned 3x3 neighborhood is still mostly one foreground color and there is
      extra non-host evidence within radius 2, showing that the missing cell came
      from the region's 1-2-pixel boundary band rather than deep interior.
    - For every such dent, paint the clipped 3x3 neighborhood with the opposite
      foreground color and recolor the dent center itself teal (8).

    Key insight: The transformation is still "opposite-color patch + teal center",
    but only boundary-adjacent black holes qualify, so the dominant region picks up
    many rare-color patches along its long boundary while the minor region gets only a few.
    """
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

    return out


# --------------- generator (combo) ---------------

# Canvas / asymmetry
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
TOTAL_COVERAGE_MIN, TOTAL_COVERAGE_MAX = 0.88, 0.94
MAJOR_VISIBLE_MIN = 0.80
MINOR_VISIBLE_MIN, MINOR_VISIBLE_MAX = 0.07, 0.14
INNER_GAP_MIN, INNER_GAP_MAX = 2, 4

# Dent placement
MAJOR_DENTS_RANGE = (8, 12)
MINOR_DENTS_RANGE = (2, 3)
DENT_MIN_HOST_3X3 = 5
DENT_SEPARATION = 3
MAJOR_INTERFACE_FRAC_MIN, MAJOR_INTERFACE_FRAC_MAX = 0.6, 0.8

# Noise
BG_NOISE_RANGE = (0, 0)
BOUNDARY_NOISE_RANGE = (0, 0)
NOISE_CLEARANCE = 2

# Attempts
MAX_GENERATION_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _boundary_band_cell(clean: np.ndarray, r: int, c: int, host: int) -> bool:
    h, w = clean.shape
    if int(clean[r, c]) != int(host):
        return False
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if not _in_bounds((h, w), rr, cc):
                return True
            if int(clean[rr, cc]) != int(host):
                return True
    return False


def _safe_dent_candidate(clean: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(clean[r, c]) != int(host):
        return False
    sub3 = _sub(clean, r, c, 1)
    host_count = int(np.sum(sub3 == int(host))) - 1
    other_count = int(np.sum(sub3 == int(other)))
    return host_count >= DENT_MIN_HOST_3X3 and host_count > other_count and _boundary_band_cell(clean, r, c, host)


def _classify_candidate(clean: np.ndarray, r: int, c: int, host: int, other: int, bg: int) -> str:
    seen_other = False
    seen_bg = False
    h, w = clean.shape
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if not _in_bounds((h, w), rr, cc):
                seen_bg = True
                continue
            val = int(clean[rr, cc])
            if val == int(other):
                seen_other = True
            elif val == int(bg):
                seen_bg = True
    if seen_other:
        return "interface"
    if seen_bg:
        return "outer"
    return "none"


def _pick_spaced(candidates: list[tuple[int, int]], n: int, banned: set[tuple[int, int]], rng: random.Random) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if any(max(abs(r - br), abs(c - bc)) < DENT_SEPARATION for br, bc in banned):
            continue
        if any(max(abs(r - rr), abs(c - cc)) < DENT_SEPARATION for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) >= n:
            return chosen
    return None


def _build_regions(n: int, major: int, minor: int, rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((n, n), dtype=int)

    for _try in range(80):
        top = rng.randint(0, 1)
        left = rng.randint(0, 1)
        bottom = rng.randint(0, 1)
        right = rng.randint(0, 1)
        r0, c0 = top, left
        r1, c1 = n - bottom, n - right
        total_area = (r1 - r0) * (c1 - c0)
        total_ratio = total_area / (n * n)
        if not (TOTAL_COVERAGE_MIN <= total_ratio <= TOTAL_COVERAGE_MAX):
            continue

        gap = rng.randint(INNER_GAP_MIN, INNER_GAP_MAX)
        minor_h = rng.randint(max(6, n // 4), max(7, n // 3))
        minor_w = rng.randint(max(6, n // 4), max(7, n // 3))
        minor_area = minor_h * minor_w
        minor_ratio = minor_area / (n * n)
        major_visible_ratio = (total_area - minor_area) / (n * n)
        if not (MINOR_VISIBLE_MIN <= minor_ratio <= MINOR_VISIBLE_MAX):
            continue
        if major_visible_ratio < MAJOR_VISIBLE_MIN:
            continue
        if minor_h + 2 * gap >= (r1 - r0) or minor_w + 2 * gap >= (c1 - c0):
            continue

        anchor = rng.choice(["tl", "tr", "bl", "br"])
        if anchor == "tl":
            sr0 = r0 + gap
            sc0 = c0 + gap
        elif anchor == "tr":
            sr0 = r0 + gap
            sc0 = c1 - gap - minor_w
        elif anchor == "bl":
            sr0 = r1 - gap - minor_h
            sc0 = c0 + gap
        else:
            sr0 = r1 - gap - minor_h
            sc0 = c1 - gap - minor_w
        sr1 = sr0 + minor_h
        sc1 = sc0 + minor_w

        grid[:, :] = 0
        grid[r0:r1, c0:c1] = int(major)
        grid[sr0:sr1, sc0:sc1] = int(minor)
        return grid

    return None


def _candidate_sets(clean: np.ndarray, major: int, minor: int, bg: int) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    valid = {int(major): [], int(minor): []}
    kinds = {int(major): [], int(minor): []}
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host not in (int(major), int(minor)):
                continue
            other = int(minor) if host == int(major) else int(major)
            if not _safe_dent_candidate(clean, r, c, host, other):
                continue
            valid[host].append((r, c))
            kinds[host].append(_classify_candidate(clean, r, c, host, other, bg))
    return valid, kinds


def _sample_dents(clean: np.ndarray, major: int, minor: int, rng: random.Random) -> list[tuple[int, int, int]] | None:
    valid, kinds = _candidate_sets(clean, major, minor, Color.BLACK)

    major_interface = [pt for pt, kind in zip(valid[int(major)], kinds[int(major)], strict=False) if kind == "interface"]
    minor_all = valid[int(minor)]

    n_major = rng.randint(*MAJOR_DENTS_RANGE)
    n_minor = rng.randint(*MINOR_DENTS_RANGE)
    if len(major_interface) < n_major or len(minor_all) < n_minor:
        return None

    banned: set[tuple[int, int]] = set()
    picked_major_interface = _pick_spaced(major_interface, n_major, banned, rng)
    if picked_major_interface is None:
        return None
    banned.update(picked_major_interface)

    picked_minor = _pick_spaced(minor_all, n_minor, banned, rng)
    if picked_minor is None:
        return None

    dents = [(r, c, int(major)) for r, c in picked_major_interface]
    dents.extend((r, c, int(minor)) for r, c in picked_minor)
    return dents


def _stamp_output(clean: np.ndarray, dents: list[tuple[int, int, int]], colors: tuple[int, int]) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    major, minor = map(int, colors)
    for r, c, host in dents:
        other = minor if host == major else major
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for r, c, _host in dents:
        out[r, c] = int(Color.TEAL)
    return out


def _near_any(points: set[tuple[int, int]], r: int, c: int, dist: int) -> bool:
    return any(max(abs(r - rr), abs(c - cc)) <= dist for rr, cc in points)


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    dent_cells: set[tuple[int, int]],
    output: np.ndarray,
    rng: random.Random,
) -> None:
    n_noise = rng.randint(*BG_NOISE_RANGE)
    if n_noise <= 0:
        return
    h, w = inp.shape
    placed = 0
    candidates: list[tuple[int, int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(Color.BLACK):
                continue
            if _near_any(dent_cells, r, c, NOISE_CLEARANCE):
                continue
            sub = _sub(clean, r, c, 1)
            if int(np.sum(sub != int(Color.BLACK))) == 0:
                continue
            counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
            pick = max(colors, key=lambda color: counts[int(color)])
            candidates.append((r, c, int(pick)))
    rng.shuffle(candidates)
    for r, c, color in candidates:
        if placed >= n_noise:
            break
        if int(inp[r, c]) != int(Color.BLACK):
            continue
        trial = inp.copy()
        trial[r, c] = int(color)
        if np.array_equal(main(trial), output):
            inp[r, c] = int(color)
            placed += 1


def _boundary_noise_candidates(clean: np.ndarray, host: int, other: int, dent_cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = clean.shape
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(host):
                continue
            if _near_any(dent_cells, r, c, NOISE_CLEARANCE):
                continue
            if _has_same_2x2(clean, r, c, host) and _classify_candidate(clean, r, c, host, other, Color.BLACK) == "interface":
                out.append((r, c))
    return out


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    major: int,
    minor: int,
    dent_cells: set[tuple[int, int]],
    output: np.ndarray,
    rng: random.Random,
) -> None:
    target = rng.randint(*BOUNDARY_NOISE_RANGE)
    if target <= 0:
        return
    candidates = []
    for host, other in ((int(major), int(minor)), (int(minor), int(major))):
        for r, c in _boundary_noise_candidates(clean, host, other, dent_cells):
            candidates.append((r, c, host, other))
    rng.shuffle(candidates)

    placed = 0
    for r, c, host, other in candidates:
        if placed >= target:
            break
        if int(inp[r, c]) != int(host):
            continue
        trial = inp.copy()
        trial[r, c] = int(other)
        if np.array_equal(main(trial), output):
            inp[r, c] = int(other)
            placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    major, minor = rng.sample(fg_pool, 2)

    clean = _build_regions(n, major, minor, rng)
    if clean is None:
        return None

    dents = _sample_dents(clean, major, minor, rng)
    if dents is None:
        return None

    output = _stamp_output(clean, dents, (major, minor))

    inp = clean.copy()
    dent_cells = {(r, c) for r, c, _host in dents}
    for r, c in dent_cells:
        inp[r, c] = int(Color.BLACK)

    _add_bg_noise(inp, clean, (major, minor), dent_cells, output, rng)
    _add_boundary_noise(inp, clean, major, minor, dent_cells, output, rng)

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _try in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
