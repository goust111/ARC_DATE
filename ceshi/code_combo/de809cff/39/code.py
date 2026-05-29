# Combo variant of de809cff (index 39)
# combo_spec: D4/2+D5/0
# Name: 视觉相近两色+矩形区域

"""
Design analysis for combo D4/2 + D5/0
=====================================

a) Concrete behavior and trace
- Old task: the two foreground regions were large irregular shapes formed by 2-3
  occluding rectangles. A black cell was recognized as a dent because, after
  local cleanup, one foreground color still dominated its 3x3 neighborhood.
- New combo: the visible regions themselves must be two clean axis-aligned
  rectangles whose colors come from a curated "similar-looking" pair such as
  `(yellow, orange)` or `(red, maroon)`.
- Small trace:
  1. Suppose the input is a 24x24 grid with one tall left rectangle and one tall
     right rectangle, colored yellow and orange, touching on a straight vertical
     split. A few orange contamination cells sit just inside the yellow side.
  2. Under the old reasoning chain, the solver only had local majority cleanup.
     That still works sometimes, but it ignores the new geometric cue that each
     region should be a solid rectangle.
  3. Under the fused rule, the solver first recovers each rectangle from the
     dense row/column projections of its color, rebuilding two full rectangles.
     Then every original black cell that lies inside exactly one recovered
     rectangle and still has host-majority support in the cleaned 3x3 window is
     treated as a dent. Its 3x3 patch becomes the opposite rectangle color and
     its center becomes teal.
- Difference: D4/2 changes the palette choice, while D5/0 changes the shape cue.
  The solver therefore uses the rectangle geometry to disambiguate subtle
  colors, rather than treating the combo as a pure cosmetic recolor.

b) Complexity calibration
- If the generator produced arbitrary rectangles with gaps between them, the task
  would become too easy and would lose the original "region boundary" feel.
  Therefore the generator always places two large rectangles that touch along
  one straight shared edge, preserving a meaningful interface for dents and
  contamination.
- If the colors were unrestricted, D4/2 would often be invisible. Therefore the
- generator only samples from a small repeated family of close-looking pairs.
- If every dent were deep interior, rectangle recovery would be visually
  unnecessary. Therefore a controlled fraction of dents is forced onto rectangle
  edges where the straight border cue is observable.

c) Hidden assumptions audited
- `_clean_foreground(...)` in the original code assumed the true region shape can
  be recovered by two local 3x3 sweeps. That assumption is no longer the best
  fit: with D5/0 the stronger invariant is "each color occupies one dense
  rectangle".
- `_dent_centers(...)` originally inferred the host only from local majority.
  In this combo that is too weak as the primary cue, because the human-facing
  change is exactly that region ownership is now also readable from rectangle
  geometry. The new solver first checks rectangle membership, then keeps the old
  `>= 5 host cells` local guard as a sanity filter.
- The original generator allowed visible L-shapes via occlusion. That breaks the
  new solver assumption that each main color has one dominant rectangular
  projection, so the generator must now guarantee exactly two large visible
  rectangles.
- Similar colors create a new risk that the combo becomes visually different but
  behaviorally identical to the old task. To avoid that, the generator enforces
  straight shared borders and edge dents so the rectangle cue actually appears
  in every sample family.

d) Placement legality
- The generator constructs two half-open rectangles inside the canvas, checks
  coverage and minimum area, and never allows them to overlap.
- Dent centers are chosen from valid host cells only, with global Chebyshev
  separation so distinct 3x3 patches do not collapse into unreadable tangles.
- Optional 2-cell dents are only added orthogonally, only inside the same host
  rectangle, and only when they remain far from every other dent center.
- Background specks are placed on true background cells; boundary contamination
  is placed only on cells adjacent to the other rectangle along the shared edge.

e) Visual observability
- A human sees two big regular blocks with nearly related colors and a straight
  boundary between them. That geometric regularity is the visible clue for
  deciding which region a black hole belongs to.
- The teal center still marks the original black hole, and the surrounding 3x3
  patch still flips to the opposite region color, so the output relation remains
  traceable from a few examples.
- No hidden numeric ordering of color ids is used. The only color structure is a
  small family of repeated similar-looking pairs plus the fixed teal marker.

f) Visual cleanliness
- Restricting the source to two large rectangles keeps the scene less cluttered
  than the original irregular occlusion version.
- Boundary contamination is capped and spaced out so it reads as a few mistaken
  cells near the split, not as a ragged second boundary.
- Dent centers are separated enough that their 3x3 opposite-color patches remain
  legible.

g) Change inferability
- The palette change has a repeated structural cue: the two region colors are
  close, but they occupy different rectangles on opposite sides of a straight
  partition.
- The geometric change has repeated evidence: every generated instance contains
  the same "two rectangles" motif, so the solver can infer that black holes are
  classified by rectangle ownership rather than by arbitrary local color flips.
- New solver assumption: each main color has one dominant dense row span and one
  dominant dense column span. The generator guarantees this by producing exactly
  one large rectangle per color and keeping all stray cells sparse.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color


SIMILAR_COLOR_PAIRS: list[tuple[int, int]] = [
    (Color.RED, Color.MAROON),
    (Color.YELLOW, Color.ORANGE),
    (Color.GREEN, Color.YELLOW),
    (Color.BLUE, Color.GREY),
    (Color.PINK, Color.MAROON),
]

# --- CFG: canvas & rectangle layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
OUTER_FRAC_MIN, OUTER_FRAC_MAX = 0.62, 0.82
OUTER_SIZE_MIN = 15
MIN_REGION_SIDE = 8
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.58, 0.82

# --- CFG: dent placement ---
DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 22
DENT_MIN_HOST_3X3 = 5
DENT_PATCH_SEPARATION = 3
DENT_PAIR_PROB = 0.14
DENT_BOUNDARY_FRAC_MIN, DENT_BOUNDARY_FRAC_MAX = 0.18, 0.35

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3X3 = 6
BOUNDARY_NOISE_MAX = 5

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 400

_LAST_GEN_INFO: dict | None = None


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) == 2, "expected exactly two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _longest_consecutive_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    best_start = best_end = int(indices[0])
    cur_start = cur_end = int(indices[0])
    for raw in indices[1:]:
        idx = int(raw)
        if idx == cur_end + 1:
            cur_end = idx
        else:
            if cur_end - cur_start > best_end - best_start:
                best_start, best_end = cur_start, cur_end
            cur_start = cur_end = idx
    if cur_end - cur_start > best_end - best_start:
        best_start, best_end = cur_start, cur_end
    return best_start, best_end


def _dominant_span(counts: np.ndarray) -> tuple[int, int]:
    peak = int(np.max(counts))
    assert peak > 0
    for slack in (4, 5, 6):
        threshold = max(3, peak - slack)
        idx = np.where(counts >= threshold)[0]
        if idx.size > 0:
            return _longest_consecutive_span(idx)
    return _longest_consecutive_span(np.where(counts > 0)[0])


def _infer_rectangle(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = g == int(color)
    row_counts = np.sum(mask, axis=1)
    col_counts = np.sum(mask, axis=0)
    r0, r1 = _dominant_span(row_counts)
    c0, c1 = _dominant_span(col_counts)
    return int(r0), int(c0), int(r1) + 1, int(c1) + 1


def _inside(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return int(r0) <= int(r) < int(r1) and int(c0) <= int(c) < int(c1)


def _fill_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0:r1, c0:c1] = int(color)


def _reconstruct_rectangles(
    g: np.ndarray, colors: tuple[int, int], bg: int
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    rects = {int(color): _infer_rectangle(g, int(color)) for color in colors}
    clean = np.full_like(g, int(bg))
    used = np.zeros_like(g, dtype=bool)
    for color in colors:
        r0, c0, r1, c1 = rects[int(color)]
        if bool(np.any(used[r0:r1, c0:c1])):
            raise AssertionError("inferred rectangles overlap")
        _fill_rect(clean, rects[int(color)], int(color))
        used[r0:r1, c0:c1] = True
    return clean, rects


def _dent_centers(
    original: np.ndarray,
    clean: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    colors: tuple[int, int],
    bg: int,
) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        owners = [int(color) for color in colors if _inside(rects[int(color)], rr, cc)]
        if len(owners) != 1:
            continue
        host = int(owners[0])
        other = int(colors[1] if host == int(colors[0]) else colors[0])
        sub = _sub3x3(clean, rr, cc)
        host_count = int(np.sum(sub == host))
        other_count = int(np.sum(sub == other))
        if host_count >= DENT_MIN_HOST_3X3 and host_count > other_count:
            centers.append((rr, cc, host))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild two similar-colored rectangles, then flip each black dent into an opposite-color patch.

    Input structure:
    - Black background with exactly two large foreground colors chosen from a visually close pair.
    - Each foreground color mainly occupies one solid axis-aligned rectangle, and the two rectangles
      touch along one straight boundary.
    - Some cells are corrupted: a few foreground specks appear in the background, a few cross-color
      contamination cells appear near the shared border, and several black dents punch into the rectangles.

    Rule:
    - Recover the two main rectangles from the dense row/column spans of the two foreground colors,
      discarding sparse stray cells outside those rectangles.
    - For each original black cell that lies inside exactly one recovered rectangle and still has a
      host-majority 3x3 neighborhood in the cleaned reconstruction, stamp a 3x3 patch of the opposite
      rectangle color around it.
    - After stamping all such patches, recolor every dent center itself to teal (8).

    Key insight: The new combo makes the region geometry more reliable than noisy local pixels: even if the
    two colors look similar and a few border cells are wrong, each main color still forms one dominant
    rectangle, and the black dents are the holes inside those recovered rectangles.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Color.BLACK)
    colors = _main_colors(g, bg)
    clean, rects = _reconstruct_rectangles(g, colors, bg)
    centers = _dent_centers(g, clean, rects, colors, bg)

    out = clean.copy()
    h, w = out.shape
    for r, c, host in centers:
        other = int(colors[1] if host == int(colors[0]) else colors[0])
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = other
    for r, c, _host in centers:
        out[r, c] = int(Color.TEAL)
    return out


# --------------- generator (combo D4/2 + D5/0) ---------------


def _choose_similar_colors(rng: random.Random) -> tuple[int, int]:
    a, b = rng.choice(SIMILAR_COLOR_PAIRS)
    if rng.random() < 0.5:
        return int(a), int(b)
    return int(b), int(a)


def _build_regions(
    n: int, color_a: int, color_b: int, rng: random.Random
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]], dict] | None:
    min_outer = max(OUTER_SIZE_MIN, int(np.ceil(n * OUTER_FRAC_MIN)))
    max_outer = max(min_outer, int(np.floor(n * OUTER_FRAC_MAX)))
    if min_outer >= n:
        return None

    for _ in range(80):
        outer_h = rng.randint(min_outer, max_outer)
        outer_w = rng.randint(min_outer, max_outer)
        coverage = (outer_h * outer_w) / float(n * n)
        if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
            continue
        r0 = rng.randint(1, n - outer_h - 1)
        c0 = rng.randint(1, n - outer_w - 1)
        orientation = rng.choice(["V", "H"])

        if orientation == "V":
            min_split = max(MIN_REGION_SIDE, int(np.ceil(MIN_REGION_CELLS / outer_h)))
            max_split = min(outer_w - MIN_REGION_SIDE, outer_w - int(np.ceil(MIN_REGION_CELLS / outer_h)))
            if min_split > max_split:
                continue
            split = rng.randint(min_split, max_split)
            rect_a = (r0, c0, r0 + outer_h, c0 + split)
            rect_b = (r0, c0 + split, r0 + outer_h, c0 + outer_w)
        else:
            min_split = max(MIN_REGION_SIDE, int(np.ceil(MIN_REGION_CELLS / outer_w)))
            max_split = min(outer_h - MIN_REGION_SIDE, outer_h - int(np.ceil(MIN_REGION_CELLS / outer_w)))
            if min_split > max_split:
                continue
            split = rng.randint(min_split, max_split)
            rect_a = (r0, c0, r0 + split, c0 + outer_w)
            rect_b = (r0 + split, c0, r0 + outer_h, c0 + outer_w)

        area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
        area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
        if area_a < MIN_REGION_CELLS or area_b < MIN_REGION_CELLS:
            continue

        clean = np.zeros((n, n), dtype=int)
        _fill_rect(clean, rect_a, color_a)
        _fill_rect(clean, rect_b, color_b)
        rects = {int(color_a): rect_a, int(color_b): rect_b}
        meta = {
            "orientation": orientation,
            "outer_rect": (r0, c0, r0 + outer_h, c0 + outer_w),
            "coverage": coverage,
        }
        return clean, rects, meta
    return None


def _is_safe_dent_center(grid: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    if int(grid[r, c]) != int(host):
        return False
    sub = _sub3x3(grid, r, c)
    host_count = int(np.sum(sub == int(host))) - 1
    other_count = int(np.sum(sub == int(other)))
    return host_count >= DENT_MIN_HOST_3X3 and host_count > other_count


def _is_boundary_cell(rect: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = rect
    return int(r) in (r0, r1 - 1) or int(c) in (c0, c1 - 1)


def _pick_spaced(
    candidates: list[tuple[int, int]],
    n: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    items = candidates[:]
    rng.shuffle(items)
    chosen: list[tuple[int, int]] = []
    for r, c in items:
        if any(max(abs(r - rr), abs(c - cc)) < DENT_PATCH_SEPARATION for rr, cc in blocked):
            continue
        if any(max(abs(r - rr), abs(c - cc)) < DENT_PATCH_SEPARATION for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        if len(chosen) == n:
            return chosen
    return None


def _pick_color_dents(
    clean: np.ndarray,
    rect: tuple[int, int, int, int],
    host: int,
    other: int,
    n: int,
    blocked: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = rect
    valid: list[tuple[int, int]] = []
    boundary: list[tuple[int, int]] = []
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not _is_safe_dent_center(clean, r, c, host, other):
                continue
            pos = (int(r), int(c))
            valid.append(pos)
            if _is_boundary_cell(rect, r, c):
                boundary.append(pos)
    if len(valid) < n:
        return None

    desired_boundary = min(
        len(boundary),
        int(round(n * rng.uniform(DENT_BOUNDARY_FRAC_MIN, DENT_BOUNDARY_FRAC_MAX))),
    )
    if boundary:
        desired_boundary = max(1, desired_boundary)

    boundary_set = set(boundary)
    for take_boundary in range(desired_boundary, -1, -1):
        picked_boundary = _pick_spaced(boundary, take_boundary, blocked, rng)
        if picked_boundary is None:
            continue
        remaining = [pos for pos in valid if pos not in boundary_set or pos in picked_boundary]
        picked_rest = _pick_spaced(remaining, n - take_boundary, blocked + picked_boundary, rng)
        if picked_rest is None:
            continue
        return picked_boundary + picked_rest
    return None


def _expand_to_pairs(
    dents: list[tuple[int, int]],
    host: int,
    other: int,
    clean: np.ndarray,
    all_centers: list[tuple[int, int]],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    h, w = clean.shape
    reserved = set(all_centers)
    groups: list[list[tuple[int, int]]] = []
    for r, c in dents:
        cells = [(int(r), int(c))]
        if rng.random() < DENT_PAIR_PROB:
            neighbors = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if (nr, nc) in reserved:
                    continue
                if int(clean[nr, nc]) != int(host):
                    continue
                if not _is_safe_dent_center(clean, nr, nc, host, other):
                    continue
                if any(
                    max(abs(nr - rr), abs(nc - cc)) < DENT_PATCH_SEPARATION
                    for rr, cc in all_centers
                    if (rr, cc) != (r, c)
                ):
                    continue
                cells.append((int(nr), int(nc)))
                reserved.add((int(nr), int(nc)))
                break
        groups.append(cells)
    return groups


def _stamp_dents(
    clean: np.ndarray,
    dent_groups_a: list[list[tuple[int, int]]],
    dent_groups_b: list[list[tuple[int, int]]],
    color_a: int,
    color_b: int,
) -> np.ndarray:
    h, w = clean.shape
    output = clean.copy()
    for dent_groups, patch_color in ((dent_groups_a, color_b), (dent_groups_b, color_a)):
        for cells in dent_groups:
            for r, c in cells:
                output[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(patch_color)
    for dent_groups in (dent_groups_a, dent_groups_b):
        for cells in dent_groups:
            for r, c in cells:
                output[r, c] = int(Color.TEAL)
    return output


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    n_in_bg: int,
    colors: tuple[int, int],
    rng: random.Random,
) -> int:
    bg_candidates = list(zip(*np.where(clean == int(Color.BLACK)), strict=False))
    rng.shuffle(bg_candidates)
    placed = 0
    used: list[tuple[int, int]] = []
    for rr, cc in bg_candidates:
        if placed >= n_in_bg:
            break
        r = int(rr)
        c = int(cc)
        if any(max(abs(r - ur), abs(c - uc)) < 2 for ur, uc in used):
            continue
        if int(np.sum(_sub3x3(inp, r, c) == int(Color.BLACK))) < BG_SPECK_MIN_BG_3X3:
            continue
        nearby = _sub3x3(clean, r, c)
        counts = [int(np.sum(nearby == int(color))) for color in colors]
        if counts[0] == counts[1] == 0:
            speck_color = int(rng.choice(list(colors)))
        else:
            speck_color = int(colors[0] if counts[0] >= counts[1] else colors[1])
        inp[r, c] = speck_color
        used.append((r, c))
        placed += 1
    return placed


def _boundary_candidates(
    clean: np.ndarray, rects: dict[int, tuple[int, int, int, int]], colors: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    a, b = map(int, colors)
    candidates: list[tuple[int, int, int, int]] = []
    for host, other in ((a, b), (b, a)):
        r0, c0, r1, c1 = rects[host]
        for r in range(r0, r1):
            for c in range(c0, c1):
                if int(clean[r, c]) != host:
                    continue
                if any(
                    0 <= nr < clean.shape[0]
                    and 0 <= nc < clean.shape[1]
                    and int(clean[nr, nc]) == other
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                ):
                    candidates.append((int(r), int(c), host, other))
    return candidates


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    rects: dict[int, tuple[int, int, int, int]],
    colors: tuple[int, int],
    n_at_boundary: int,
    rng: random.Random,
) -> int:
    candidates = _boundary_candidates(clean, rects, colors)
    rng.shuffle(candidates)
    placed = 0
    used: list[tuple[int, int]] = []
    for r, c, host, other in candidates:
        if placed >= min(n_at_boundary, BOUNDARY_NOISE_MAX):
            break
        if int(inp[r, c]) != int(host):
            continue
        if any(max(abs(r - ur), abs(c - uc)) < 2 for ur, uc in used):
            continue
        inp[r, c] = int(other)
        used.append((r, c))
        placed += 1
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_a, color_b = _choose_similar_colors(rng)
    built = _build_regions(n, color_a, color_b, rng)
    if built is None:
        return None
    clean, rects, meta = built

    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a = n_total // 2
    n_b = n_total - n_a
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a

    order = [(color_a, color_b, n_a), (color_b, color_a, n_b)]
    if rng.random() < 0.5:
        order.reverse()

    chosen: dict[int, list[tuple[int, int]]] = {int(color_a): [], int(color_b): []}
    blocked: list[tuple[int, int]] = []
    for host, other, need in order:
        picked = _pick_color_dents(clean, rects[int(host)], int(host), int(other), need, blocked, rng)
        if picked is None:
            return None
        chosen[int(host)] = picked
        blocked.extend(picked)

    dents_a = chosen[int(color_a)]
    dents_b = chosen[int(color_b)]
    all_centers = dents_a + dents_b

    groups_a = _expand_to_pairs(dents_a, color_a, color_b, clean, all_centers, rng)
    groups_b = _expand_to_pairs(dents_b, color_b, color_a, clean, all_centers, rng)
    output = _stamp_dents(clean, groups_a, groups_b, color_a, color_b)

    inp = clean.copy()
    for cells in groups_a + groups_b:
        for r, c in cells:
            inp[r, c] = int(Color.BLACK)

    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    placed_bg = _add_bg_noise(inp, clean, n_in_bg, (color_a, color_b), rng)
    _add_boundary_noise(inp, clean, rects, (color_a, color_b), n_noise - placed_bg, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    global _LAST_GEN_INFO
    rect_a = rects[int(color_a)]
    rect_b = rects[int(color_b)]
    _LAST_GEN_INFO = {
        "colors": (int(color_a), int(color_b)),
        "orientation": meta["orientation"],
        "coverage": float(meta["coverage"]),
        "rectangles": {int(color_a): rect_a, int(color_b): rect_b},
        "areas": {
            int(color_a): int((rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])),
            int(color_b): int((rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])),
        },
        "dent_count": int(sum(len(cells) for cells in groups_a + groups_b)),
        "boundary_dent_count": int(
            sum(_is_boundary_cell(rects[int(color_a)], r, c) for cells in groups_a for r, c in cells)
            + sum(_is_boundary_cell(rects[int(color_b)], r, c) for cells in groups_b for r, c in cells)
        ),
    }
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
