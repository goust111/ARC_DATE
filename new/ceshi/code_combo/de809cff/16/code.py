# Combo variant of de809cff (index 16)
# combo_spec: D7/1+D9/2
# Name: 补丁溢出连锁反应+凹痕密集成簇
#
# Design analysis:
# - Concrete behavior trace:
#   In the base task, three black dents packed near an A/B boundary would still
#   behave independently: clean the regions once, stamp three overlapping 3x3
#   opposite-color patches once, then stop. In this combo, the same dense cluster
#   creates a first-wave union of overlapping 3x3 patches; every spilled patch
#   cell that lands inside the opposite cleaned region and still has >=5 local
#   support cells for that region becomes a new center for the next wave. The
#   process repeats on the static cleaned regions until no unseen supported
#   overflow cells remain. The visible difference is a short cascade of extra
#   teal centers and alternating color spill on both sides of the interface.
# - Complexity calibration:
#   If every touched cell propagated without a support test, the cascade would
#   flood the board and look noisy. If dents stayed far apart, the result would
#   collapse back to the original task. The generator therefore uses one compact
#   3-cell cluster per sample, positioned near the region-region boundary, while
#   the solver only propagates through locally supported cells in the opposite
#   region.
# - Implicit assumptions audited against the original solver:
#   `_main_colors` still assumes exactly two dominant foreground colors. That is
#   preserved. `_clean_foreground` still treats all non-foreground cells as black;
#   the generator keeps noise to one-cell specks so two cleanup sweeps remain
#   valid. The original "one black cell -> one 3x3 patch" assumption is no
#   longer valid, so the new solver separates initial black-dent detection from
#   multi-wave propagation over the cleaned static regions.
# - Placement legality:
#   Clusters are chosen from fixed small polyominoes, clipped to real host cells,
#   kept several cells apart from other clusters, and validated on the final
#   dented grid so every black cell is still detectable as an initial dent.
# - Visual observability:
#   The only extra marker is teal, reused as "this location fired in some wave".
#   Humans can compare one-wave and chained examples because the dense black
#   clusters are visible in the input, while the output shows extra teal cells
#   appearing across the interface, not just at the original black holes.
# - Cleanliness / inferability:
#   The generator rejects samples unless the chained output differs from the
#   single-wave baseline by a noticeable number of cells and the propagation
#   reaches at least wave 1 beyond the original dents. This prevents degeneration
#   back to the source puzzle and avoids ambiguous same-round color conflicts by
#   spacing clusters apart.

from __future__ import annotations

from collections import Counter as _Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1] if int(color) == int(colors[0]) else colors[0])


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

                sub = _sub3x3(cur, r, c)
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


def _host_supported(clean: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    sub = _sub3x3(clean, r, c)
    host_count = int(np.sum(sub == int(host)))
    other_count = int(np.sum(sub == int(other)))
    return bool(host_count >= 5 and host_count > other_count)


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    h, w = original.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        counts = {int(color): int(np.sum(_sub3x3(clean, rr, cc) == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = _other_color(colors, int(host))
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)]:
            centers.append((rr, cc, int(host)))
    centers.sort()
    return centers


def _single_pass_output(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    out = clean.copy()
    h, w = clean.shape
    for r, c, host in centers:
        other = _other_color(colors, host)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
    for r, c, _host in centers:
        out[r, c] = Color.TEAL
    return out


def _chain_output(
    clean: np.ndarray, centers: list[tuple[int, int, int]], colors: tuple[int, int]
) -> tuple[np.ndarray, dict[str, object]]:
    h, w = clean.shape
    out = clean.copy()
    active = [(int(r), int(c), int(host), 0) for r, c, host in centers]
    seen_chain = set()
    all_centers = [(int(r), int(c), int(host), 0) for r, c, host in centers]
    wave_sizes = _Counter({0: len(active)})

    while active:
        next_map: dict[tuple[int, int], tuple[int, int, int]] = {}
        for r, c, host, wave in active:
            other = _other_color(colors, host)
            out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(other)
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if int(clean[rr, cc]) != int(other):
                        continue
                    if (rr, cc) in seen_chain:
                        continue
                    if not _host_supported(clean, rr, cc, other, host):
                        continue
                    prev = next_map.get((rr, cc))
                    cand = (rr, cc, int(other))
                    if prev is None or cand < prev:
                        next_map[(rr, cc)] = cand

        if not next_map:
            break

        next_wave = int(active[0][3] + 1)
        active = []
        for rr, cc in sorted(next_map):
            seen_chain.add((rr, cc))
            host = int(next_map[(rr, cc)][2])
            active.append((rr, cc, host, next_wave))
            all_centers.append((rr, cc, host, next_wave))
        wave_sizes[next_wave] = len(active)

    for r, c, _host, _wave in all_centers:
        out[int(r), int(c)] = Color.TEAL

    info: dict[str, object] = {
        "all_centers": all_centers,
        "wave_sizes": dict(wave_sizes),
        "max_wave": max((wave for _r, _c, _h, wave in all_centers), default=0),
        "secondary_centers": max(0, len(all_centers) - len(centers)),
    }
    return out, info


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    centers = _dent_centers(g, clean, colors, bg)
    single = _single_pass_output(clean, centers, colors)
    chained, info = _chain_output(clean, centers, colors)
    info.update(
        {
            "grid": g,
            "clean": clean,
            "colors": colors,
            "initial_centers": centers,
            "single_pass": single,
            "output": chained,
            "delta_vs_single": int(np.sum(chained != single)),
        }
    )
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform clustered black dents into overlapping opposite-color patches with
    spill-driven chain propagation across the interface of the two regions.

    Input structure:
    - Black background with two large foreground colors that still need the same
      local cleanup as the base puzzle.
    - Black dents now appear in tight 2-4-cell clusters instead of only spaced
      singles, so their initial 3x3 opposite-color patches overlap heavily.
    - The densest clusters sit near the boundary between the two foreground
      regions, making some first-wave patch cells spill directly into the other
      cleaned region.

    Rule:
    - First clean isolated specks and one-cell boundary contamination exactly as
      in the source task, yielding the static host regions used for all later
      decisions.
    - Detect every original black dent whose cleaned 3x3 neighborhood is still
      dominated by one foreground color; each such dent starts wave 0 and stamps
      a 3x3 patch in the opposite foreground color.
    - Whenever a stamped patch cell lands inside the opposite cleaned region and
      that landing cell is itself well-supported by that region locally, treat it
      as a new center for the next wave. Continue wave by wave until no unseen
      supported spill cells remain, then mark every fired center teal (8).

    Key insight: region identity comes from the cleaned pre-transform grid, so
    the cascade is a bounded propagation on top of the original geometry rather
    than an uncontrolled repaint of already modified pixels.
    """
    return np.asarray(_analyze(np.asarray(grid, dtype=int))["output"], dtype=int)


# --------------- generator ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.55, 0.78
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84
MIN_INTERFACE_CONTACT = 12

# --- CFG: clustered dents ---
CLUSTER_COUNT_MIN, CLUSTER_COUNT_MAX = 1, 1
CLUSTER_SIZE_CHOICES = [3]
CLUSTER_GAP = 2
MIN_CHAIN_DELTA = 6
MIN_SECONDARY_CENTERS = 3

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 6, 14
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.60, 0.82
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400

def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    first_color, second_color = (color_a, color_b) if rng.random() < 0.5 else (color_b, color_a)

    h1 = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    w1 = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    h2 = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
    w2 = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))

    r0a = rng.randint(0, RECT_MARGIN_MAX)
    c0a = rng.randint(0, RECT_MARGIN_MAX)
    r0b = max(0, N - h2 - rng.randint(0, RECT_MARGIN_MAX))
    c0b = max(0, N - w2 - rng.randint(0, RECT_MARGIN_MAX))
    r1a, c1a = min(N, r0a + h1), min(N, c0a + w1)
    r1b, c1b = min(N, r0b + h2), min(N, c0b + w2)

    overlap_h = min(r1a, r1b) - max(r0a, r0b)
    overlap_w = min(c1a, c1b) - max(c0a, c0b)
    if overlap_h < 4 or overlap_w < 4:
        return None
    if r0b - r0a < 3 or c0b - c0a < 3:
        return None
    if r1b - r1a < 3 or c1b - c1a < 3:
        return None

    grid = np.zeros((N, N), dtype=int)
    grid[r0a:r1a, c0a:c1a] = int(first_color)
    grid[r0b:r1b, c0b:c1b] = int(second_color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if int(np.sum(grid == int(first_color))) == h1 * w1:
        return None

    interface = 0
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            v = int(grid[r, c])
            if v == 0:
                continue
            other = color_b if v == color_a else color_a
            if r + 1 < H and int(grid[r + 1, c]) == int(other):
                interface += 1
            if c + 1 < W and int(grid[r, c + 1]) == int(other):
                interface += 1
    if interface < MIN_INTERFACE_CONTACT:
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
            own_n = int(np.sum(nsub == host))
            other_n = int(np.sum(nsub == other))
            bg_n = int(np.sum(nsub == 0))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, color_a: int, color_b: int, rng: random.Random) -> None:
    bg_candidates = list(zip(*np.where(clean == 0), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        if int(np.sum(_sub3x3(inp, r, c) == 0)) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != 0]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = int(speck_color)
        placed += 1


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
        if int(inp[r, c]) != int(host):
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
        inp[r, c] = int(other)
        placed += 1


def _cluster_far_enough(cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in occupied:
            if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) < CLUSTER_GAP:
                return False
    return True


def _cluster_supported(
    clean: np.ndarray, dent_cells: set[tuple[int, int]], cells: list[tuple[int, int]], host: int, other: int
) -> bool:
    tmp = clean.copy()
    for r, c in dent_cells | set(cells):
        tmp[int(r), int(c)] = 0

    touches_other = False
    for r, c in cells:
        if int(clean[r, c]) != int(host):
            return False
        if int(np.sum(_sub3x3(clean, r, c) == other)) >= 1:
            touches_other = True
        if not _host_supported(tmp, r, c, host, other):
            return False

    if not touches_other:
        return False

    spawned = 0
    for r, c in cells:
        for rr in range(max(0, r - 1), min(clean.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(clean.shape[1], c + 2)):
                if int(clean[rr, cc]) == int(other) and _host_supported(clean, rr, cc, other, host):
                    spawned += 1
    return spawned >= 1


def _boundary_candidates(clean: np.ndarray, host: int, other: int) -> list[tuple[int, int]]:
    h, w = clean.shape
    out = []
    for r in range(h):
        for c in range(w):
            if int(clean[r, c]) != int(host):
                continue
            if not _host_supported(clean, r, c, host, other):
                continue
            sub5 = clean[max(0, r - 2) : min(h, r + 3), max(0, c - 2) : min(w, c + 3)]
            if int(np.sum(sub5 == other)) == 0:
                continue
            out.append((r, c))
    return out


def _try_place_cluster(
    clean: np.ndarray,
    dent_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    host: int,
    other: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidates = _boundary_candidates(clean, host, other)
    rng.shuffle(candidates)

    def _compact(cells: list[tuple[int, int]]) -> bool:
        rows = [r for r, _c in cells]
        cols = [c for _r, c in cells]
        return max(rows) - min(rows) <= 2 and max(cols) - min(cols) <= 2

    def _neighbors(cluster: list[tuple[int, int]]) -> list[tuple[int, int]]:
        opts = set()
        cluster_set = set(cluster)
        for r, c in cluster:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = int(r + dr), int(c + dc)
                    if not (0 <= rr < clean.shape[0] and 0 <= cc < clean.shape[1]):
                        continue
                    if int(clean[rr, cc]) != int(host) or (rr, cc) in cluster_set:
                        continue
                    if not _cluster_far_enough([(rr, cc)], occupied):
                        continue
                    cand = cluster + [(rr, cc)]
                    if not _compact(cand):
                        continue
                    opts.add((rr, cc))
        out = list(opts)
        rng.shuffle(out)
        out.sort(key=lambda cell: int(np.sum(_sub3x3(clean, cell[0], cell[1]) == other) == 0))
        return out

    def _search(cluster: list[tuple[int, int]], target_size: int) -> list[tuple[int, int]] | None:
        if len(cluster) == target_size:
            return sorted(cluster) if _cluster_supported(clean, dent_cells, cluster, host, other) else None
        for cell in _neighbors(cluster):
            found = _search(cluster + [cell], target_size)
            if found is not None:
                return found
        return None

    sizes = CLUSTER_SIZE_CHOICES[:]
    rng.shuffle(sizes)
    for size in sizes:
        for ar, ac in candidates:
            if not _cluster_far_enough([(ar, ac)], occupied):
                continue
            found = _search([(int(ar), int(ac))], int(size))
            if found is not None:
                return found
    return None


def _place_clusters(clean: np.ndarray, color_a: int, color_b: int, rng: random.Random) -> list[tuple[int, int]]:
    n_clusters = rng.randint(CLUSTER_COUNT_MIN, CLUSTER_COUNT_MAX)
    dent_cells: set[tuple[int, int]] = set()
    occupied: set[tuple[int, int]] = set()

    for _ in range(n_clusters):
        host_order = [color_a, color_b]
        rng.shuffle(host_order)
        found = None
        for host in host_order:
            other = color_b if host == color_a else color_a
            cluster = _try_place_cluster(clean, dent_cells, occupied, host, other, rng)
            if cluster is not None:
                found = cluster
                break
        if found is None:
            return []
        for cell in found:
            dent_cells.add(cell)
            occupied.add(cell)
    return sorted(dent_cells)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    dents = _place_clusters(clean, color_a, color_b, rng)
    if not dents:
        return None

    inp = clean.copy()
    for r, c in dents:
        inp[r, c] = 0

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        info = _analyze(inp)
    except Exception:
        return None

    if len(info["initial_centers"]) != len(dents):
        return None
    if int(info["secondary_centers"]) < MIN_SECONDARY_CENTERS:
        return None
    if int(info["max_wave"]) < 1:
        return None
    if int(info["delta_vs_single"]) < MIN_CHAIN_DELTA:
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
