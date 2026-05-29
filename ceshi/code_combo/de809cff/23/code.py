# Combo variant of de809cff (index 23)
# combo_spec: D6/3+D9/2
# Name: 清洁与变换交替迭代+补丁溢出连锁
#
# Design analysis for the fused variant
# a) Concrete behavior:
#    In the base task, all cleanup happens first and every black dent is transformed once.
#    Here the state evolves by rounds:
#    clean-to-stable -> transform once -> clean-to-stable -> transform once -> ...
#    Example: a black dent in color A near the A/B boundary creates a 3x3 color-B patch.
#    The part of that patch that lands inside region B is not a stable B block there; after the
#    next cleanup it is shaved into small black gaps/protrusions, and those new black cells can
#    themselves qualify as dents inside B, creating a second-round color-A patch. This is the
#    visible difference from the original one-shot rule.
# b) Complexity calibration:
#    The chain is intentionally limited. The generator plants only a few boundary-adjacent
#    "primary" dents and accepts samples only when they cause 1-3 extra rounds, so the effect
#    is visible but not chaotic.
# c) Assumption audit:
#    The original solver assumed that only the input black cells mattered and that one cleanup
#    pass was enough. That is no longer true. The new solver therefore:
#    - re-detects dents after each cleanup phase from the current grid,
#    - keeps transformed centers locked so earlier transforms do not evaporate entirely,
#    - rejects outside background cells by requiring a strong 3x3 majority and at least two
#      axial host neighbors, which separates true dents from the external background.
# d) Placement legality:
#    Dents are sampled from cells whose 3x3 neighborhoods remain host-dominant after blackening.
#    Centers are kept several cells apart, patches clip to grid bounds, and noise is only added
#    where the cleanup rules can legally undo it.
# e) Visual observability:
#    The generator requires secondary-round dents and a measurable difference from a single-pass
#    baseline, so training pairs actually exhibit the iterative chain instead of degenerating back
#    to the original task.
# f) Visual cleanliness:
#    Coverage, dent counts, and accepted chain depth are capped. This keeps the two main regions
#    recognizable and avoids grids where too many overlapping patches obscure the rule.
# g) Inferability:
#    Colors still play the same structural roles as the base task: two large foreground colors,
#    black dents/noise, teal markers. The only new degree of freedom is time/iteration, which is
#    exposed spatially by secondary patches appearing near earlier boundary overflows.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import Color


GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.84

PRIMARY_DENT_MIN, PRIMARY_DENT_MAX = 2, 4
EXTRA_DENT_MIN, EXTRA_DENT_MAX = 4, 8
DENT_SEPARATION = 4

NOISE_COUNT_MIN, NOISE_COUNT_MAX = 5, 11
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.55, 0.8
BG_SPECK_MIN_BG_3X3 = 6
BOUNDARY_MIN_HOST_3X3 = 6

MAX_CLEAN_SWEEPS = 4
MAX_ITER_ROUNDS = 12
MAX_GENERATION_ATTEMPTS = 500


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) not in (int(bg), int(Color.TEAL))]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
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


def _clean_sweep(cur: np.ndarray, colors: tuple[int, int], bg: int, locked: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    h, w = cur.shape
    a, b = map(int, colors)
    nxt = cur.copy()
    stats = {"to_bg": 0, "to_other": 0}

    for r in range(h):
        for c in range(w):
            if bool(locked[r, c]):
                nxt[r, c] = cur[r, c]
                continue

            v = int(cur[r, c])
            if v not in colors:
                nxt[r, c] = bg
                continue

            other = b if v == a else a
            sub = _sub3x3(cur, r, c)
            own = int(np.sum(sub == v))
            other_count = int(np.sum(sub == other))
            bg_count = int(np.sum(sub == bg))
            same2 = _has_same_2x2(cur, r, c, v)

            if other_count >= 5 and other_count >= own + 2 and not same2:
                nxt[r, c] = other
                stats["to_other"] += 1
            elif own <= 2:
                nxt[r, c] = bg
                stats["to_bg"] += 1
            elif bg_count >= 5 and own <= 4 and not same2:
                nxt[r, c] = bg
                stats["to_bg"] += 1
            elif other_count >= 3 and own <= 3 and not same2:
                # This extra shave step is what turns spillover strips into new black dents.
                nxt[r, c] = bg
                stats["to_bg"] += 1
            else:
                nxt[r, c] = v

    return nxt, stats


def _clean_phase(cur: np.ndarray, colors: tuple[int, int], bg: int, locked: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    acc = {"to_bg": 0, "to_other": 0, "sweeps": 0}
    out = cur.copy()
    for _ in range(MAX_CLEAN_SWEEPS):
        nxt, stats = _clean_sweep(out, colors, bg, locked)
        acc["to_bg"] += stats["to_bg"]
        acc["to_other"] += stats["to_other"]
        acc["sweeps"] += 1
        if np.array_equal(nxt, out):
            return out, acc
        out = nxt
    return out, acc


def _host_at(clean: np.ndarray, r: int, c: int, colors: tuple[int, int], bg: int) -> tuple[int, int] | None:
    if int(clean[r, c]) != int(bg):
        return None

    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = _other_color(colors, host)
    if counts[int(host)] < 5 or counts[int(host)] <= counts[int(other)]:
        return None

    axial_host = 0
    h, w = clean.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w and int(clean[rr, cc]) == int(host):
            axial_host += 1
    if axial_host < 2:
        return None

    return int(host), int(other)


def _dent_centers(clean: np.ndarray, colors: tuple[int, int], bg: int, seen: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(clean == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        if (rr, cc) in seen:
            continue
        host_info = _host_at(clean, rr, cc, colors, bg)
        if host_info is None:
            continue
        host, _other = host_info
        centers.append((rr, cc, int(host)))
    return centers


def _apply_patches(clean: np.ndarray, colors: tuple[int, int], centers: list[tuple[int, int, int]]) -> tuple[np.ndarray, int]:
    out = clean.copy()
    h, w = clean.shape
    overflow_pixels = 0
    for r, c, host in centers:
        other = _other_color(colors, host)
        r0, r1 = max(0, r - 1), min(h, r + 2)
        c0, c1 = max(0, c - 1), min(w, c + 2)
        patch_before = clean[r0:r1, c0:c1]
        overflow_pixels += int(np.sum(patch_before == int(other)))
        out[r0:r1, c0:c1] = int(other)
    return out, overflow_pixels


def _simulate(grid: np.ndarray, *, collect_trace: bool = False) -> tuple[np.ndarray, dict[str, object]] | np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)

    cur = g.copy()
    locked = np.zeros(cur.shape, dtype=bool)
    centers_all: list[tuple[int, int]] = []
    seen = set()
    trace: dict[str, object] = {
        "rounds": 0,
        "centers_per_round": [],
        "cleanup_to_bg": [],
        "cleanup_to_other": [],
        "cleanup_sweeps": [],
        "overflow_pixels": 0,
    }

    for _round in range(MAX_ITER_ROUNDS):
        clean, clean_stats = _clean_phase(cur, colors, bg, locked)
        centers = _dent_centers(clean, colors, bg, seen)

        trace["cleanup_to_bg"].append(int(clean_stats["to_bg"]))
        trace["cleanup_to_other"].append(int(clean_stats["to_other"]))
        trace["cleanup_sweeps"].append(int(clean_stats["sweeps"]))
        trace["centers_per_round"].append([(int(r), int(c), int(host)) for r, c, host in centers])

        if not centers:
            if np.array_equal(clean, cur):
                cur = clean
                break
            cur = clean
            continue

        nxt, overflow = _apply_patches(clean, colors, centers)
        trace["overflow_pixels"] = int(trace["overflow_pixels"]) + int(overflow)
        for r, c, _host in centers:
            seen.add((int(r), int(c)))
            centers_all.append((int(r), int(c)))
            locked[int(r), int(c)] = True
        cur = nxt
        trace["rounds"] = int(trace["rounds"]) + 1

    out = cur.copy()
    for r, c in centers_all:
        out[int(r), int(c)] = int(Color.TEAL)

    secondary = sum(len(round_centers) for round_centers in trace["centers_per_round"][1:])
    trace["secondary_centers"] = int(secondary)
    trace["total_centers"] = int(len(centers_all))
    trace["locked_centers"] = int(np.sum(locked))

    if collect_trace:
        return out, trace
    return out


def _single_pass_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    locked = np.zeros(g.shape, dtype=bool)
    clean, _stats = _clean_phase(g, colors, bg, locked)
    centers = _dent_centers(clean, colors, bg, set())
    out, _overflow = _apply_patches(clean, colors, centers)
    for r, c, _host in centers:
        out[int(r), int(c)] = int(Color.TEAL)
    return out


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    out, trace = _simulate(grid, collect_trace=True)
    baseline = _single_pass_baseline(grid)
    return {
        "rounds": int(trace["rounds"]),
        "secondary_centers": int(trace["secondary_centers"]),
        "total_centers": int(trace["total_centers"]),
        "overflow_pixels": int(trace["overflow_pixels"]),
        "diff_vs_single_pass": int(np.sum(out != baseline)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly clean two large color regions and let boundary-adjacent dents trigger chained opposite-color patches.

    Input structure:
    - Black background with two dominant foreground colors occupying large rectangular/L-shaped regions.
    - Black dents sit inside those regions, and a small number of one-cell specks or boundary contaminations are mixed in.
    - Some dents are intentionally close to the interface between the two foreground regions, so a later 3x3 patch can cross that interface.

    Rule:
    - Work in rounds. In each round, first clean the current foreground arrangement until local one-cell noise/protrusions are stable.
    - Then find every current black cell whose cleaned 3x3 neighborhood is strongly dominated by one foreground color; stamp a 3x3 patch of the opposite foreground color there.
    - Patch centers are remembered and become teal only at the end. Because boundary dents can create spillover strips inside the other region, later cleanup rounds may shave those strips into fresh black dents, which trigger more opposite-color patches.
    - Stop when a cleanup round makes no further changes and no new dents are found.

    Key insight: The output is not a one-shot response to the original black cells. It is the fixed point of an alternating clean/transform process where overflow from one patch seeds the next transform near the opposite region.
    """
    return _simulate(grid, collect_trace=False)


def _build_regions(n: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        rect_colors = [first, _other_color((color_a, color_b), first), first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        hh = rng.randint(int(n * RECT_SIZE_FRAC_MIN), int(n * RECT_SIZE_FRAC_MAX))
        ww = rng.randint(int(n * RECT_SIZE_FRAC_MIN), int(n * RECT_SIZE_FRAC_MAX))
        margin = rng.randint(0, RECT_MARGIN_MAX)
        r0 = margin if corner[0] == "T" else max(0, n - hh - margin)
        c0 = margin if corner[1] == "L" else max(0, n - ww - margin)
        return r0, c0, min(n, r0 + hh), min(n, c0 + ww)

    grid = np.zeros((n, n), dtype=int)
    for corner, color in zip(chosen, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = int(color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None

    coverage = (cnt_a + cnt_b) / float(n * n)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None

    if n_rects == 3:
        first_color = rect_colors[0]
        _labels, n_components = label(grid == int(first_color))
        if int(n_components) < 2:
            return None

    return grid


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    if host_count < 5 or host_count <= other_count:
        return False

    trial = grid.copy()
    trial[int(r), int(c)] = int(Color.BLACK)
    host_info = _host_at(trial, int(r), int(c), (int(host), int(other)), int(Color.BLACK))
    return host_info is not None and int(host_info[0]) == int(host)


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[int(r - rr), int(c - cc)] = int(color)
                if bool(np.all(block == int(color))):
                    return True
    return False


def _pick_spaced(candidates: list[tuple[int, int]], n: int, rng: random.Random) -> list[tuple[int, int]] | None:
    order = candidates[:]
    rng.shuffle(order)
    chosen: list[tuple[int, int]] = []
    for r, c in order:
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_SEPARATION for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) >= n:
                return chosen
    return None


def _candidate_sets(clean: np.ndarray, colors: tuple[int, int]) -> dict[str, list[tuple[int, int, int]]]:
    color_a, color_b = map(int, colors)
    spill: list[tuple[int, int, int]] = []
    interior: list[tuple[int, int, int]] = []
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in colors:
                continue
            other = color_b if v == color_a else color_a
            if not _is_safe_dent_center(clean, r, c, v, other):
                continue
            sub = _sub3x3(clean, r, c)
            other_count = int(np.sum(sub == int(other)))
            if other_count >= 1:
                spill.append((int(r), int(c), int(v)))
            else:
                interior.append((int(r), int(c), int(v)))
    return {"spill": spill, "interior": interior}


def _add_bg_noise(inp: np.ndarray, clean: np.ndarray, n_in_bg: int, colors: tuple[int, int], rng: random.Random) -> None:
    bg_candidates = list(zip(*np.where(clean == int(Color.BLACK)), strict=False))
    rng.shuffle(bg_candidates)
    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r = int(r_)
        c = int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == int(Color.BLACK))) < BG_SPECK_MIN_BG_3X3:
            continue
        nearby = [int(x) for x in _sub3x3(clean, r, c).flat if int(x) in colors]
        color = int(Counter(nearby).most_common(1)[0][0]) if nearby else int(rng.choice(list(colors)))
        inp[r, c] = int(color)
        placed += 1


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0 = max(0, nr - 1)
            nc0 = max(0, nc - 1)
            nsub[int(r - nr0), int(c - nc0)] = int(other)
            own = int(np.sum(nsub == int(host)))
            other_count = int(np.sum(nsub == int(other)))
            bg_count = int(np.sum(nsub == int(Color.BLACK)))
            if other_count >= 5 and other_count >= own + 2:
                return False
            if own <= 2:
                return False
            if bg_count >= 5 and own <= 4:
                return False
    return True


def _add_boundary_noise(inp: np.ndarray, clean: np.ndarray, n_at_boundary: int, colors: tuple[int, int], rng: random.Random) -> None:
    color_a, color_b = map(int, colors)
    boundary_candidates: list[tuple[int, int, int, int]] = []
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            host = int(clean[r, c])
            if host not in colors:
                continue
            other = color_b if host == color_a else color_a
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((int(r), int(c), int(host), int(other)))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub == int(host)))
        other_count = int(np.sum(sub == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3X3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = int(other)
        placed += 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [int(c) for c in Color.NOT_BLACK if int(c) != int(Color.TEAL)]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(n, color_a, color_b, rng)
    if clean is None:
        return None

    candidates = _candidate_sets(clean, colors)
    spill_by_color = {
        int(color_a): [(r, c) for r, c, host in candidates["spill"] if int(host) == int(color_a)],
        int(color_b): [(r, c) for r, c, host in candidates["spill"] if int(host) == int(color_b)],
    }
    interior_by_color = {
        int(color_a): [(r, c) for r, c, host in candidates["interior"] if int(host) == int(color_a)],
        int(color_b): [(r, c) for r, c, host in candidates["interior"] if int(host) == int(color_b)],
    }

    if min(len(spill_by_color[color_a]), len(spill_by_color[color_b])) < 1:
        return None

    primary_target = rng.randint(PRIMARY_DENT_MIN, PRIMARY_DENT_MAX)
    primary_need_a = max(1, primary_target // 2)
    primary_need_b = max(1, primary_target - primary_need_a)
    chosen_a = _pick_spaced(spill_by_color[color_a], primary_need_a, rng)
    chosen_b = _pick_spaced(spill_by_color[color_b], primary_need_b, rng)
    if chosen_a is None or chosen_b is None:
        return None

    primary = chosen_a + chosen_b
    if len(primary) < primary_target:
        return None

    extra_target = rng.randint(EXTRA_DENT_MIN, EXTRA_DENT_MAX)
    interior_pool = interior_by_color[color_a] + interior_by_color[color_b]
    interior_pool = [
        (r, c)
        for r, c in interior_pool
        if all(max(abs(r - rr), abs(c - cc)) >= DENT_SEPARATION for rr, cc in primary)
    ]
    extras = _pick_spaced(interior_pool, extra_target, rng)
    if extras is None:
        return None

    dents = primary + extras
    inp = clean.copy()
    for r, c in dents:
        inp[int(r), int(c)] = int(Color.BLACK)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, n_in_bg, colors, rng)
    _add_boundary_noise(inp, clean, n_noise - n_in_bg, colors, rng)

    try:
        out, trace = _simulate(inp, collect_trace=True)
    except Exception:
        return None

    baseline = _single_pass_baseline(inp)
    diff_vs_single = int(np.sum(out != baseline))
    rounds = int(trace["rounds"])
    secondary = int(trace["secondary_centers"])
    total_centers = int(trace["total_centers"])
    overflow = int(trace["overflow_pixels"])

    if rounds < 2 or rounds > 4:
        return None
    if secondary < 2:
        return None
    if overflow < 4:
        return None
    if diff_vs_single < 6:
        return None
    if total_centers > 18:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
