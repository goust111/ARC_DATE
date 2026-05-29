# Combo variant of de809cff (index 13)
# combo_spec: D5/1+D7/0
# Name: 棋盘格交错区域+凹痕仅在边界附近

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label

_THIS_FILE = Path(__file__).resolve()
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _solver_dir = _parent / "solvers"
    if _solver_dir.exists():
        if str(_solver_dir) not in sys.path:
            sys.path.append(str(_solver_dir))
        break

from arc_utils import Color

# -----------------------------------------------------------------------------
# Design analysis for combo D5/1 + D7/0
#
# concretization:
#   - The original task assumes two large mostly solid foreground regions with
#     black dents somewhere inside them. This combo replaces those solids with a
#     single thick foreground silhouette whose interior is colored by coarse
#     checkerboard tiles of the two foreground colors.
#   - Dents are no longer allowed deep inside a tile. A valid dent is a
#     single-cell black hole whose four orthogonal neighbors are still inside
#     the foreground silhouette, but whose cleaned 3x3 neighborhood already
#     mixes the two colors because the host tile sits next to a checker seam.
#   - Concrete trace on a 4x4-tile checkerboard: if a red cell sits one pixel
#     away from a vertical red/green seam, blackening that cell yields a 3x3
#     neighborhood with 5 red and 3 green cells. Original task logic would read
#     mostly-one-color local support in a solid blob; the combo keeps that same
#     3x3 majority test but forces it to happen inside a visibly mixed setting.
#   - A black cell just outside the silhouette is not a dent even if it touches
#     many foreground cells, because its orthogonal neighborhood is not fully
#     foreground. This separates interior defects from outer background.
#
# complexity calibration:
#   - A 1x1 alternating checkerboard would be unreadable, while huge tiles would
#     make the scene collapse back toward the base task. The generator therefore
#     uses tile sizes 3-4 so the checker structure is obvious and mixed 3x3
#     neighborhoods are common near seams.
#   - Dents are restricted to single cells. Pair dents near checker seams create
#     cluttered overlapping 3x3 stamps and obscure the intended "count 5 vs 3"
#     reasoning, so the fusion keeps one-cell dents and instead varies many
#     repeated placements of the same mixed-boundary case.
#   - Although the spec allows boundary distance 1-2, the generator biases to
#     the most visible subset: candidates whose 3x3 neighborhood has exactly
#     three opposite-color cells after blackening. They are still within the
#     allowed 1-2-pixel band, but stay maximally ambiguous and easy to notice.
#
# hidden assumptions reviewed:
#   - The original cleanup relied on same-color 2x2 support and on one color
#     dominating most of a region. Checkerboard interiors break both
#     assumptions: many legitimate seam-adjacent cells belong to thin local
#     color strips even though the foreground silhouette itself is thick.
#   - The new cleanup therefore reasons about foreground occupancy, not about
#     same-color majority. A colored cell is removed only when it looks like an
#     isolated background speck or a one-cell outer protrusion of the entire
#     silhouette.
#   - Dent detection now assumes true dents are internal holes: the center is
#     black, all four orthogonal neighbors remain foreground after cleanup, and
#     the cleaned 3x3 shows a strict 5:3 or 6:2 color majority. The generator
#     guarantees this and keeps noise away from dent neighborhoods.
#   - main() still assumes exactly two non-black colors are the real foreground
#     palette. The generator uses only those two colors plus black and teal.
#
# placement legality:
#   - The clean silhouette is built from overlapping thick rectangles so it is
#     connected, bounded away from the grid edge, and large enough for many
#     checker tiles of each color.
#   - Dent centers are sampled only from legal host cells that remain inside the
#     silhouette after blackening, are within two pixels of an opposite-color
#     tile, and are separated by Chebyshev distance >= 3 so their 3x3 output
#     patches do not overlap.
#   - Noise is placed only in background cells, and protrusions are limited to
#     one-cell outward bumps with a single orthogonal anchor. Both forms are
#     chosen outside protected halos around dents, so cleanup and dent counting
#     stay unambiguous.
#
# observability / cleanliness / inferability:
#   - The human-visible novelty is structural rather than numeric: the input is
#     a coarse checkerboard mosaic, and the black dents hug checker seams rather
#     than appearing in big uniform color pools.
#   - The output still paints opposite-color 3x3 squares with teal centers, so
#     the learner can compare input and output and infer that black seam-adjacent
#     holes trigger the transformation after removing trivial one-cell noise.
#   - Keeping only two foreground colors plus teal avoids color-mapping
#     ambiguity. Repetition of the same seam-adjacent 5:3 case across both
#     colors gives enough evidence from a few examples without making the board
#     visually chaotic.
# -----------------------------------------------------------------------------


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _is_fg(value: int, colors: tuple[int, int]) -> bool:
    return int(value) == int(colors[0]) or int(value) == int(colors[1])


def _foreground_neighbor_stats(grid: np.ndarray, r: int, c: int, colors: tuple[int, int]) -> tuple[int, int]:
    orth_fg = 0
    total_fg = 0
    h, w = grid.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if _is_fg(int(grid[rr, cc]), colors):
                total_fg += 1
                if dr == 0 or dc == 0:
                    orth_fg += 1
    return int(orth_fg), int(total_fg)


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove isolated colored specks and one-cell outer protrusions.

    In the checkerboard combo, same-color support is not reliable at internal
    seams, so cleanup only asks whether the entire foreground silhouette around
    a colored cell is too thin to be genuine structure.
    """
    h, w = g.shape
    cur = g.copy()
    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if not _is_fg(v, colors):
                    nxt[r, c] = bg
                    continue
                orth_fg, total_fg = _foreground_neighbor_stats(cur, r, c, colors)
                if total_fg <= 1:
                    nxt[r, c] = bg
                elif orth_fg <= 1 and total_fg <= 3:
                    nxt[r, c] = bg
                else:
                    nxt[r, c] = v
        cur = nxt
    return cur


def _dent_centers(original: np.ndarray, clean: np.ndarray, colors: tuple[int, int], bg: int) -> list[tuple[int, int, int]]:
    h, w = clean.shape
    centers: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        if rr <= 0 or rr >= h - 1 or cc <= 0 or cc >= w - 1:
            continue
        orth = [
            int(clean[rr - 1, cc]),
            int(clean[rr + 1, cc]),
            int(clean[rr, cc - 1]),
            int(clean[rr, cc + 1]),
        ]
        if any(not _is_fg(v, colors) for v in orth):
            continue
        sub = _sub3x3(clean, rr, cc)
        counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
        host = max(colors, key=lambda color: counts[int(color)])
        other = colors[1] if int(host) == int(colors[0]) else colors[0]
        if counts[int(host)] >= 5 and counts[int(host)] > counts[int(other)] and counts[int(other)] >= 2:
            centers.append((rr, cc, int(host)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn seam-adjacent black dents in a checkerboard foreground into opposite-color patches.

    Input structure:
    - Black background surrounds one large connected foreground silhouette.
    - Inside that silhouette, the two foreground colors alternate as coarse
      checkerboard tiles rather than forming two mostly solid blobs.
    - Black dents appear only near checker seams, and there may also be a few
      isolated colored specks or one-cell outer protrusions in the background.

    Rule:
    - First clean the foreground silhouette by deleting colored cells that are
      too weakly supported to belong to the real shape: isolated specks and
      one-cell outer bumps become black.
    - For each original black cell that sits inside the cleaned silhouette
      (all four orthogonal neighbors are foreground), count the two colors in
      its cleaned 3x3 neighborhood.
    - If one foreground color still has the strict local majority, stamp a 3x3
      patch of the opposite foreground color around that dent and then recolor
      the dent center teal (8).

    Key insight: The decisive signal is no longer "which big region is this
    hole inside?" but "which color wins the mixed 3x3 count at this checker
    seam after trivial one-cell noise is removed?"
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

GRID_SIZE_MIN, GRID_SIZE_MAX = 20, 30
MASK_COVERAGE_MIN, MASK_COVERAGE_MAX = 0.58, 0.82
MIN_REGION_CELLS = 100
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.42, 0.72
TILE_SIZES = (3, 4)

DENT_COUNT_MIN, DENT_COUNT_MAX = 12, 20
DENT_SEPARATION = 3
DENT_OTHER_COUNT_TARGET = 3

NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 16
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.55, 0.8
MAX_GENERATION_ATTEMPTS = 400


def _connected(mask: np.ndarray) -> bool:
    _, n = label(mask.astype(int))
    return int(n) == 1


def _build_mask(N: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(80):
        mask = np.zeros((N, N), dtype=bool)
        n_rects = rng.choice([2, 3])
        for _rect in range(n_rects):
            h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
            w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
            r0 = rng.randint(1, max(1, N - h - 1))
            c0 = rng.randint(1, max(1, N - w - 1))
            mask[r0 : r0 + h, c0 : c0 + w] = True
        coverage = float(np.mean(mask))
        if not (MASK_COVERAGE_MIN <= coverage <= MASK_COVERAGE_MAX):
            continue
        if not _connected(mask):
            continue
        rows, cols = np.where(mask)
        if rows.size == 0:
            continue
        if rows.min() < 1 or cols.min() < 1 or rows.max() > N - 2 or cols.max() > N - 2:
            continue
        return mask
    return None


def _paint_checkerboard(mask: np.ndarray, color_a: int, color_b: int, block: int, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    rr, cc = np.indices((h, w))
    shift_r = rng.randint(0, block - 1)
    shift_c = rng.randint(0, block - 1)
    parity = ((rr + shift_r) // block + (cc + shift_c) // block) % 2
    grid = np.zeros((h, w), dtype=int)
    grid[mask & (parity == 0)] = int(color_a)
    grid[mask & (parity == 1)] = int(color_b)
    return grid


def _mixed_boundary_distance(clean: np.ndarray, r: int, c: int, other: int) -> int | None:
    best: int | None = None
    h, w = clean.shape
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            rr = int(r + dr)
            cc = int(c + dc)
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if int(clean[rr, cc]) != int(other):
                continue
            dist = max(abs(dr), abs(dc))
            if best is None or dist < best:
                best = int(dist)
    return best


def _valid_dent_candidates(clean: np.ndarray, host: int, other: int) -> list[tuple[int, int]]:
    h, w = clean.shape
    valid: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(clean[r, c]) != int(host):
                continue
            orth = [int(clean[r - 1, c]), int(clean[r + 1, c]), int(clean[r, c - 1]), int(clean[r, c + 1])]
            if any(v == int(Color.BLACK) for v in orth):
                continue
            sub = _sub3x3(clean, r, c)
            host_count = int(np.sum(sub == int(host))) - 1
            other_count = int(np.sum(sub == int(other)))
            if host_count < 5 or host_count <= other_count:
                continue
            if other_count != DENT_OTHER_COUNT_TARGET:
                continue
            dist = _mixed_boundary_distance(clean, r, c, other)
            if dist is None or dist > 2:
                continue
            valid.append((int(r), int(c)))
    return valid


def _pick_dents(
    candidates_a: list[tuple[int, int]],
    candidates_b: list[tuple[int, int]],
    n_a: int,
    n_b: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(50):
        order = [(0, candidates_a[:], n_a), (1, candidates_b[:], n_b)]
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        per_color: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
        ok = True
        for key, candidates, need in order:
            rng.shuffle(candidates)
            for r, c in candidates:
                if all(max(abs(r - rr), abs(c - cc)) >= DENT_SEPARATION for rr, cc in chosen):
                    per_color[key].append((r, c))
                    chosen.append((r, c))
                    if len(per_color[key]) >= need:
                        break
            if len(per_color[key]) < need:
                ok = False
                break
        if ok:
            return per_color[0], per_color[1]
    return None


def _stamp_output(clean: np.ndarray, dents_a: list[tuple[int, int]], dents_b: list[tuple[int, int]], color_a: int, color_b: int) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    for r, c in dents_a:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(color_b)
    for r, c in dents_b:
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = int(color_a)
    for r, c in dents_a + dents_b:
        out[r, c] = Color.TEAL
    return out


def _protect_zone(points: list[tuple[int, int]], radius: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    protected: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    protected.add((rr, cc))
    return protected


def _add_bg_specks(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: set[tuple[int, int]],
    count: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = inp.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in protected or int(clean[r, c]) != int(Color.BLACK):
                continue
            sub = _sub3x3(clean, r, c)
            if int(np.sum(sub != int(Color.BLACK))) <= 1:
                candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[:count]:
        inp[r, c] = rng.choice([int(color_a), int(color_b)])


def _outer_protrusion_candidates(clean: np.ndarray, colors: tuple[int, int], protected: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    h, w = clean.shape
    result: list[tuple[int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (r, c) in protected or int(clean[r, c]) != int(Color.BLACK):
                continue
            orth_vals = [
                int(clean[r - 1, c]),
                int(clean[r + 1, c]),
                int(clean[r, c - 1]),
                int(clean[r, c + 1]),
            ]
            anchors = [v for v in orth_vals if _is_fg(v, colors)]
            if len(anchors) != 1:
                continue
            total_fg = int(np.sum(_sub3x3(clean, r, c) != int(Color.BLACK)))
            if not (1 <= total_fg <= 3):
                continue
            result.append((r, c, int(anchors[0])))
    return result


def _add_outer_protrusions(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: set[tuple[int, int]],
    count: int,
    colors: tuple[int, int],
    rng: random.Random,
) -> None:
    blocked = set(protected)
    candidates = _outer_protrusion_candidates(clean, colors, protected)
    rng.shuffle(candidates)
    placed = 0
    for r, c, color in candidates:
        if placed >= count:
            break
        if (r, c) in blocked or int(inp[r, c]) != int(Color.BLACK):
            continue
        inp[r, c] = int(color)
        placed += 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    block = rng.choice(TILE_SIZES)

    mask = _build_mask(N, rng)
    if mask is None:
        return None
    clean = _paint_checkerboard(mask, color_a, color_b, block, rng)

    cnt_a = int(np.sum(clean == int(color_a)))
    cnt_b = int(np.sum(clean == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None

    candidates_a = _valid_dent_candidates(clean, int(color_a), int(color_b))
    candidates_b = _valid_dent_candidates(clean, int(color_b), int(color_a))
    n_total = rng.randint(DENT_COUNT_MIN, DENT_COUNT_MAX)
    n_a = n_total // 2
    n_b = n_total - n_a
    if rng.random() < 0.5:
        n_a, n_b = n_b, n_a
    if len(candidates_a) < n_a or len(candidates_b) < n_b:
        return None
    picked = _pick_dents(candidates_a, candidates_b, n_a, n_b, rng)
    if picked is None:
        return None
    dents_a, dents_b = picked

    out = _stamp_output(clean, dents_a, dents_b, int(color_a), int(color_b))
    inp = clean.copy()
    for r, c in dents_a + dents_b:
        inp[r, c] = Color.BLACK

    protected = _protect_zone(dents_a + dents_b, 2, inp.shape)
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_specks(inp, clean, protected, n_bg, int(color_a), int(color_b), rng)
    _add_outer_protrusions(inp, clean, protected, n_noise - n_bg, (int(color_a), int(color_b)), rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
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
