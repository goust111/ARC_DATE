# Combo variant of de809cff (index 34)
# combo_spec: D3/1+D6/2
# Name: 扩大清洁阈值+严格全一邻域判定

from __future__ import annotations

import random
from collections import Counter as _Counter

import numpy as np

from arc_utils import Color

# Design analysis for combo D3/1 + D6/2:
# - Concrete behavior:
#   The original task had two permissive stages: cleanup only removed one-cell
#   artifacts, and a black dent transformed whenever one foreground color merely
#   dominated the cleaned 3x3. The combo tightens both.
#   Small trace:
#     old clean kept a detached 6-cell island `aaa / a0a / 00a`, then the black
#     center transformed because the local majority was color `a`.
#     combo clean first deletes that whole <8-cell component into background, so
#     the same black center is no longer inside a region and nothing transforms.
#     Separately, for a boundary hole `aaa / a0b / abb`, the old rule used
#     host=a by 5>3, while the combo rejects it because the 3x3 contains both
#     foreground colors.
# - Complexity calibration:
#   The generator therefore creates three visibly different black-hole cases:
#   1) deep dents inside large regions that still transform;
#   2) mixed-neighborhood dents on the color-color interface that stay black;
#   3) small detached island dents that would qualify structurally, but whose
#      whole host component is erased by the aggressive size threshold.
#   This keeps the combo from collapsing into a pure D3 variant.
# - Implicit assumptions audit:
#   `_main_colors` still assumes exactly two dominant foreground colors, so the
#   generator keeps every added artifact small compared with the two main areas.
#   The old local cleanup no longer applies: with D6/2, cleanup is component-
#   based and must absorb a small component into the dominant surrounding color
#   instead of always painting it black; otherwise a tiny foreign blot inside a
#   region would become a new black hole and distort the task.
#   `_dent_centers` also can no longer rely on a majority host; it now needs
#   "all non-black neighbors belong to one foreground color" plus the original
#   minimum support of 5 cells.
# - Placement legality:
#   Deep transforming dents are sampled only from full 3x3 interiors of a large
#   region. Mixed dents are sampled only where both foreground colors appear in
#   the clean 3x3 and no background is present. Small islands are placed in 5x5
#   background windows so they remain detached and get removed as one component.
#   Embedded foreign blots are placed in uniform 5x5 host interiors so cleanup
#   deterministically absorbs them back into the host region.
# - Visual observability:
#   From a few examples, a human can compare three black holes that look similar
#   in the input but behave differently in the output: deep holes get the teal-
#   centered opposite-color 3x3 patch; boundary holes stay black; tiny isolated
#   hole-bearing islands vanish completely. The rule depends on local geometry,
#   not on numeric color ids.
# - Visual cleanliness:
#   All object centers are spaced so 3x3 stamps do not overwrite untouched holes
#   or cleanup artifacts. Only two foreground colors plus teal are used, and the
#   extra small components are few enough to read as deliberate exceptions.
# - Inference of the change:
#   The output reduction is explainable from visible structure: small connected
#   pieces disappear, and among the surviving large regions only dents whose 3x3
#   support is purely one color get transformed.

AGGRESSIVE_COMPONENT_MIN_SIZE = 8
MIN_UNIFORM_HOST_3X3 = 5

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.60, 0.78

# --- CFG: dent placement ---
STRICT_DENT_COUNT_MIN, STRICT_DENT_COUNT_MAX = 6, 10
MIXED_DENT_COUNT_MIN, MIXED_DENT_COUNT_MAX = 3, 5
ISLAND_DENT_COUNT_MIN, ISLAND_DENT_COUNT_MAX = 2, 4
DENT_PATCH_SEPARATION = 3
SMALL_OBJECT_SEPARATION = 5
DENT_PAIR_PROB = 0.15

# --- CFG: extra aggressive-clean examples ---
SMALL_BLOT_COUNT_MIN, SMALL_BLOT_COUNT_MAX = 2, 4

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 800

ISLAND_PATTERNS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1)),
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)),
    ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)),
)

BLOT_PATTERNS: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0),),
    ((-1, 0), (0, 0)),
    ((0, -1), (0, 0), (1, 0)),
    ((-1, 0), (0, -1), (0, 0), (0, 1)),
)


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_replacement(
    grid: np.ndarray, component: list[tuple[int, int]], color: int, colors: tuple[int, int], bg: int
) -> int:
    counts: _Counter[int] = _Counter()
    comp_set = set(component)
    h, w = grid.shape
    valid_colors = {int(bg), int(colors[0]), int(colors[1])}
    for r, c in component:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in comp_set:
                    continue
                v = int(grid[nr, nc])
                counts[v if v in valid_colors else int(bg)] += 1
    if not counts:
        return int(bg)
    max_count = max(counts.values())
    best = [color_id for color_id, count in counts.items() if count == max_count]
    if len(best) == 1:
        return int(best[0])
    if int(bg) in best:
        return int(bg)
    return int(min(best))


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Aggressively absorb every small foreground component into its surroundings.

    Components smaller than `AGGRESSIVE_COMPONENT_MIN_SIZE` are rewritten to the
    dominant color around their boundary: detached islands fall back to black,
    while tiny foreign-color blots inside the other region get merged into that
    host region.
    """
    cur = np.where(np.isin(g, colors), g, int(bg)).astype(int, copy=False)
    nxt = cur.copy()
    for color in colors:
        for comp in _components_of_color(cur, int(color)):
            if len(comp) >= AGGRESSIVE_COMPONENT_MIN_SIZE:
                continue
            replacement = _component_replacement(cur, comp, int(color), colors, bg)
            for r, c in comp:
                nxt[r, c] = int(replacement)
    return nxt


def _uniform_host(clean: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> int | None:
    sub = _sub3x3(clean, r, c)
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    present = [int(color) for color in colors if counts[int(color)] > 0]
    if len(present) != 1:
        return None
    host = int(present[0])
    if counts[host] < MIN_UNIFORM_HOST_3X3:
        return None
    return host


def _dent_centers(
    original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        host = _uniform_host(clean, rr, cc, colors)
        if host is not None:
            centers.append((rr, cc, host))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform only deeply enclosed dents after aggressively removing tiny pieces.

    Input structure:
    - A black canvas contains two large foreground-color regions.
    - The grid may also contain black dents, mixed-neighborhood boundary holes,
      and several tiny foreground components in the background or embedded as
      small foreign-color blots inside the large regions.

    Rule:
    - First clean the foreground aggressively: any foreground connected component
      smaller than the threshold is absorbed into the locally dominant surround,
      so detached little islands disappear to black and tiny foreign blots are
      repainted into the larger nearby region.
    - Then inspect each originally black cell on the cleaned grid. It is a true
      dent only if every non-black pixel in its clipped 3x3 neighborhood belongs
      to one foreground color, and that color occupies at least 5 cells there.
    - For each such dent, paint its clipped 3x3 neighborhood with the opposite
      foreground color and recolor the dent center itself to teal (8).

    Key insight: the combo keeps only dents that survive both filters: they must
    lie inside a large surviving region, and their cleaned 3x3 support must be
    color-pure rather than merely majority-dominated. Small island dents vanish,
    and mixed boundary dents remain black.
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


# --------------- generator (cx) ---------------


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles to produce two large dominant regions."""
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

    cnt_a = int(np.sum(grid == color_a))
    cnt_b = int(np.sum(grid == color_b))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / float(N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3 and len(_components_of_color(grid, int(rect_colors[0]))) < 2:
        return None
    return grid


def _is_deep_strict_center(grid: np.ndarray, r: int, c: int, host: int) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != int(host):
        return False
    return bool(np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == int(host)))


def _is_mixed_majority_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = grid.shape
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if int(grid[r, c]) != int(host):
        return False
    sub = grid[r - 1 : r + 2, c - 1 : c + 2]
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    bg_count = int(np.sum(sub == Color.BLACK))
    return host_count >= MIN_UNIFORM_HOST_3X3 and host_count > other_count and other_count >= 1 and bg_count == 0


def _far_enough(r: int, c: int, blocked: list[tuple[int, int]], dist: int) -> bool:
    return all(max(abs(r - rr), abs(c - cc)) >= dist for rr, cc in blocked)


def _pick_with_blockers(
    candidates: list[tuple[int, int]],
    n: int,
    blocked: list[tuple[int, int]],
    dist: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    pool = list(candidates)
    rng.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for r, c in pool:
        if _far_enough(r, c, blocked + chosen, dist):
            chosen.append((r, c))
            if len(chosen) >= n:
                return chosen
    return None


def _expand_strict_pairs(
    dents: list[tuple[int, int]],
    host: int,
    clean: np.ndarray,
    occupied: list[tuple[int, int]],
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
                if not _is_deep_strict_center(clean, nr, nc, host):
                    continue
                if any((nr, nc) == (rr, cc) for rr, cc in occupied):
                    continue
                if _far_enough(nr, nc, [pos for pos in occupied if pos != (r, c)], DENT_PATCH_SEPARATION):
                    cells.append((nr, nc))
                    occupied.append((nr, nc))
                    break
        result.append(cells)
    return result


def _split_two(total: int, rng: random.Random) -> tuple[int, int]:
    a, b = total // 2, total - total // 2
    if rng.random() < 0.5:
        a, b = b, a
    return a, b


def _stamp_output(
    clean: np.ndarray,
    strict_groups_a: list[list[tuple[int, int]]],
    strict_groups_b: list[list[tuple[int, int]]],
    mixed_a: list[tuple[int, int]],
    mixed_b: list[tuple[int, int]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for r, c in mixed_a + mixed_b:
        output[r, c] = Color.BLACK
    for dent_groups, stamp_color in ((strict_groups_a, color_b), (strict_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(stamp_color)
    for dent_groups in (strict_groups_a, strict_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = Color.TEAL
    return output


def _pick_bg_centers(
    clean: np.ndarray, blocked: list[tuple[int, int]], n: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(clean[r, c]) != Color.BLACK:
                continue
            if not bool(np.all(clean[r - 2 : r + 3, c - 2 : c + 3] == Color.BLACK)):
                continue
            candidates.append((r, c))
    return _pick_with_blockers(candidates, n, blocked, SMALL_OBJECT_SEPARATION, rng)


def _pick_host_centers(
    clean: np.ndarray, host: int, blocked: list[tuple[int, int]], n: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    h, w = clean.shape
    candidates: list[tuple[int, int]] = []
    for r in range(2, h - 2):
        for c in range(2, w - 2):
            if int(clean[r, c]) != int(host):
                continue
            if not bool(np.all(clean[r - 2 : r + 3, c - 2 : c + 3] == int(host))):
                continue
            candidates.append((r, c))
    return _pick_with_blockers(candidates, n, blocked, SMALL_OBJECT_SEPARATION, rng)


def _paint_pattern(grid: np.ndarray, center: tuple[int, int], pattern: tuple[tuple[int, int], ...], color: int) -> None:
    r0, c0 = center
    for dr, dc in pattern:
        grid[r0 + dr, c0 + dc] = int(color)


def _apply_island_dents(
    inp: np.ndarray,
    centers: list[tuple[int, int]],
    colors: tuple[int, int],
    rng: random.Random,
) -> None:
    for center in centers:
        pattern = rng.choice(ISLAND_PATTERNS)
        color = rng.choice(colors)
        _paint_pattern(inp, center, pattern, int(color))
        inp[center] = Color.BLACK


def _apply_small_blots(
    inp: np.ndarray,
    centers_a: list[tuple[int, int]],
    centers_b: list[tuple[int, int]],
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    for center in centers_a:
        _paint_pattern(inp, center, rng.choice(BLOT_PATTERNS), int(color_b))
    for center in centers_b:
        _paint_pattern(inp, center, rng.choice(BLOT_PATTERNS), int(color_a))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    h, w = clean.shape
    strict_valid = {color_a: [], color_b: []}
    mixed_valid = {color_a: [], color_b: []}
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v == color_a:
                if _is_deep_strict_center(clean, r, c, color_a):
                    strict_valid[color_a].append((r, c))
                elif _is_mixed_majority_center(clean, r, c, color_a, color_b):
                    mixed_valid[color_a].append((r, c))
            elif v == color_b:
                if _is_deep_strict_center(clean, r, c, color_b):
                    strict_valid[color_b].append((r, c))
                elif _is_mixed_majority_center(clean, r, c, color_b, color_a):
                    mixed_valid[color_b].append((r, c))

    if len(strict_valid[color_a]) < STRICT_DENT_COUNT_MIN // 2 or len(strict_valid[color_b]) < STRICT_DENT_COUNT_MIN // 2:
        return None
    if len(mixed_valid[color_a]) + len(mixed_valid[color_b]) < MIXED_DENT_COUNT_MIN:
        return None

    n_strict = rng.randint(STRICT_DENT_COUNT_MIN, STRICT_DENT_COUNT_MAX)
    n_mixed = rng.randint(MIXED_DENT_COUNT_MIN, MIXED_DENT_COUNT_MAX)
    n_strict_a, n_strict_b = _split_two(n_strict, rng)
    n_mixed_a, n_mixed_b = _split_two(n_mixed, rng)

    mixed_a = _pick_with_blockers(mixed_valid[color_a], n_mixed_a, [], DENT_PATCH_SEPARATION, rng)
    mixed_b = _pick_with_blockers(
        mixed_valid[color_b], n_mixed_b, mixed_a or [], DENT_PATCH_SEPARATION, rng
    )
    if mixed_a is None or mixed_b is None:
        return None

    blocked = mixed_a + mixed_b
    strict_a = _pick_with_blockers(
        strict_valid[color_a], n_strict_a, blocked, DENT_PATCH_SEPARATION, rng
    )
    strict_b = _pick_with_blockers(
        strict_valid[color_b], n_strict_b, blocked + (strict_a or []), DENT_PATCH_SEPARATION, rng
    )
    if strict_a is None or strict_b is None:
        return None

    occupied = mixed_a + mixed_b + strict_a + strict_b
    strict_groups_a = _expand_strict_pairs(strict_a, color_a, clean, occupied, rng)
    strict_groups_b = _expand_strict_pairs(strict_b, color_b, clean, occupied, rng)

    n_island = rng.randint(ISLAND_DENT_COUNT_MIN, ISLAND_DENT_COUNT_MAX)
    island_centers = _pick_bg_centers(clean, occupied, n_island, rng)
    if island_centers is None:
        return None
    occupied += island_centers

    n_blots = rng.randint(SMALL_BLOT_COUNT_MIN, SMALL_BLOT_COUNT_MAX)
    n_blot_a, n_blot_b = _split_two(n_blots, rng)
    blot_centers_a = _pick_host_centers(clean, color_a, occupied, n_blot_a, rng)
    blot_centers_b = _pick_host_centers(clean, color_b, occupied + (blot_centers_a or []), n_blot_b, rng)
    if blot_centers_a is None or blot_centers_b is None:
        return None

    output = _stamp_output(clean, strict_groups_a, strict_groups_b, mixed_a, mixed_b, color_a, color_b)

    inp = clean.copy()
    for r, c in mixed_a + mixed_b:
        inp[r, c] = Color.BLACK
    for cells in strict_groups_a + strict_groups_b:
        for r, c in cells:
            inp[r, c] = Color.BLACK

    _apply_island_dents(inp, island_centers, (color_a, color_b), rng)
    _apply_small_blots(inp, blot_centers_a, blot_centers_b, color_a, color_b, rng)

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
