# Combo variant of b99e7126 (index 37)
# combo_spec: D1/1+D7/1
# Name: 颜色标记识别 + 多个相同异常

# Design analysis
# a) Concrete behavior and trace
#    - The outer geometry stays the same as the base task: a 29x29 board, a 7x7 grid of
#      3x3 cell interiors, and 1-pixel divider lines.
#    - The original task had one globally special cell pattern. Its interior mask was scaled
#      once to the whole cell grid, and the few shown minority cells were only partial clues
#      for that single placement.
#    - This fused variant instead shows 2-3 identical marked template cells. Each template
#      sits at the center of its own local 3x3 coarse block. Around each center there are
#      already 2-3 ordinary minority cells using the same shape but a different active color.
#    - Small trace: suppose the active mask is
#          1 1 1
#          0 1 1
#          1 0 1
#      and two marked centers appear at coarse cells (2,2) and (5,5). The input shows:
#        * the center cell of each block in marker color,
#        * several other active locations in ordinary minority color,
#        * everything else as the majority pattern.
#      The output completes both local 3x3 blocks: every active location of each block becomes
#      the ordinary minority pattern, and the marked centers are recolored to ordinary too.
#    - The difference from the original is structural in two places at once:
#        * D1/1: source cells are identified by marker color, not by being the lone rare pattern.
#        * D7/1: there are several same-content sources, so the same mask must be overlaid at
#          multiple anchors instead of solved as one global placement.
# b) Complexity calibration
#    - We keep only 5 colors: separator, majority-exclusive, shared, ordinary-active, marker.
#      That is enough to expose the fusion without creating unreadable color bookkeeping.
#    - Every sample has repeated support for both new ideas:
#        * 2-3 marked centers prove the marker color is a class-level cue, not a singleton.
#        * 4-9 ordinary minority cells prove the final propagated pattern/color.
#    - Ordinary minority cells are forced to outnumber marked centers, so a solver cannot
#      simply treat the marker pattern itself as the final fill pattern and ignore recoloring.
# c) Implicit assumptions from the original solver that break here
#    - `minority_pat = min(counts.items(), ...)` is no longer meaningful. The rare pattern is
#      now the marked template, but it is only a clue pattern, not the output pattern.
#    - The original unique `(top, left)` placement search assumed all observed minority cells
#      belong to one global 3x3 coarse block. That is false once several local blocks coexist.
#    - Equality-to-one-pattern no longer recovers all active evidence, because active evidence
#      is split between the marked template pattern and the ordinary minority pattern.
#    - The output write can no longer use the rarest pattern directly; marked cells must be
#      rewritten into the ordinary minority pattern.
# d) Placement legality
#    - Every marked center is sampled from coarse coordinates 1..5 so its centered 3x3 block
#      stays in bounds.
#    - Different centers are chosen so their 3x3 coarse blocks are pairwise disjoint. That
#      prevents conflicting writes and keeps the visual scene legible.
#    - Input generation is output-first and every candidate is re-validated with `main()`.
# e) Visual observability
#    - Across examples, one color always appears in exactly 2-3 cells, always at the center of
#      a partially shown local cluster, and always disappears in the output. That makes the
#      marker role visible to a human from a few examples.
#    - The ordinary minority pattern also appears repeatedly in the input, so the recoloring
#      target is supported rather than invented from nowhere.
# f) Visual cleanliness
#    - Blocks are disjoint and always 3x3 at coarse scale, so the output reads as several clean
#      local completions rather than a cluttered union of overlapping overlays.
#    - Each non-majority cell still uses the same inactive/shared color, so the relationship
#      between marked and ordinary cells is visually obvious.
# g) Color-change inferability
#    - The marker color has exactly one role: "this active center cell is a template anchor".
#    - The ordinary-active color has exactly one role: "this is the completed active pattern".
#    - Because ordinary minority cells are already present in every input, the output's
#      marker-to-ordinary recoloring is inferable from visible support.
# h) Anti-degeneration metric to inspect after implementation
#    - We will measure how often the original solver either errors or produces a different
#      output on fused samples. A high disagreement rate shows the generator is not collapsing
#      back to the original single-global-placement regime.
# i) New solver assumptions and generator guarantees
#    - Solver assumes exactly one majority pattern, one repeated marked-template pattern, and
#      one repeated ordinary minority pattern. Generator constructs exactly those 3 patterns.
#    - Solver assumes there are exactly 2 non-majority-exclusive colors, with marker appearing
#      in fewer cells than ordinary-active. Generator enforces at least 2 ordinary cells per
#      marked center, so the ordering is stable and visible.
#    - Solver assumes the marker mask has an active center pixel, because each marked cell is
#      the center anchor of its own local 3x3 coarse block. Generator filters masks to ensure
#      that property.

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

for _parent in Path(__file__).resolve().parents:
    if (_parent / "arc_utils.py").exists():
        if str(_parent) not in sys.path:
            sys.path.append(str(_parent))
        break

from arc_utils import paste


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete several local fractal overlays using repeated color-marked center templates.

    Input structure:
    - The board is partitioned into equal cells by full divider rows and columns of one color.
    - One interior pattern is the majority pattern and fills most cells.
    - There are 2-3 identical marked template cells whose active pixels use one marker color.
    - There are also several ordinary minority cells with the same active/inactive shape but a
      different active color; these preview what the completed active cells should look like.

    Rule:
    - Find the marker color by looking for the less frequent non-majority color, and collect all
      cells containing it. Those cells are the centers of the local overlays.
    - Read the marked template's marker pixels as a 3x3 boolean mask. Its center pixel is active,
      so each marked cell anchors a centered 3x3 coarse block.
    - For every marked center, stamp the ordinary minority pattern into all mask-active coarse
      positions of that centered block, leaving mask-inactive positions as the majority pattern.
    - Replace the marked center cells too: the marker is only an input cue and disappears in the
      completed output.

    Key insight: The special color no longer identifies a single global outlier. It names a
    repeated class of anchor cells. One shared 3x3 mask is reused locally at every anchor, while
    the already-shown ordinary minority cells reveal the final pattern that must replace the marker.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert inner == 3, "this fused variant expects 3x3 cell interiors"
    radius = inner // 2

    patterns, cell_pat = _extract_cells(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 3, "need majority, ordinary minority, and marked template patterns"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    majority_colors = {int(v) for v in np.unique(majority_arr)}

    color_cells: dict[int, set[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            arr = patterns[cell_pat[i][j]]
            for color in np.unique(arr):
                color_cells.setdefault(int(color), set()).add((i, j))

    special_colors = [
        (color, cells)
        for color, cells in color_cells.items()
        if color not in majority_colors
    ]
    assert len(special_colors) == 2, "need marker and ordinary active colors outside the majority palette"

    special_colors.sort(key=lambda item: (len(item[1]), item[0]))
    marker_color, marker_cells = special_colors[0]
    ordinary_color, ordinary_cells = special_colors[1]
    assert 2 <= len(marker_cells) <= 3, "must have 2-3 marked centers"
    assert len(ordinary_cells) > len(marker_cells), "ordinary minority support must outnumber marked centers"

    source_pats = {cell_pat[i][j] for i, j in marker_cells}
    assert len(source_pats) == 1, "all marked centers must share one template pattern"
    source_pat = next(iter(source_pats))
    source_arr = patterns[source_pat]
    mask = source_arr == int(marker_color)
    assert bool(mask[radius, radius]), "marked template must have an active center pixel"

    ordinary_pats = {cell_pat[i][j] for i, j in ordinary_cells}
    assert len(ordinary_pats) == 1, "all ordinary active cells must share one pattern"
    ordinary_pat = next(iter(ordinary_pats))
    ordinary_arr = patterns[ordinary_pat]
    assert ordinary_pat != source_pat and ordinary_pat != majority_pat
    ordinary_mask = ordinary_arr == int(ordinary_color)
    assert np.array_equal(mask, ordinary_mask), "marked and ordinary patterns must share one active/inactive shape"

    out = g.copy()
    occupied = np.zeros((n, n), dtype=bool)
    allowed_ordinary = np.zeros((n, n), dtype=bool)
    for ci, cj in sorted(marker_cells):
        assert radius <= ci < n - radius and radius <= cj < n - radius, "marked center block must fit"
        block = np.zeros((n, n), dtype=bool)
        block[ci - radius : ci + radius + 1, cj - radius : cj + radius + 1] = True
        assert not np.any(occupied & block), "local coarse blocks must be disjoint"
        occupied |= block

        for di in range(inner):
            for dj in range(inner):
                if mask[di, dj]:
                    allowed_ordinary[ci + di - radius, cj + dj - radius] = True
                target_pat = ordinary_arr if mask[di, dj] else majority_arr
                r0 = (ci + di - radius) * step + 1
                c0 = (cj + dj - radius) * step + 1
                out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    for oi, oj in ordinary_cells:
        assert allowed_ordinary[oi, oj], "ordinary preview cells must lie on an active position of some local block"
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
SEED_COUNTS = (2, 3)
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_ORDINARY_PER_SEED = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 400

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

_VALID_CENTERS = [(r, c) for r in range(1, CELL_GRID_N - 1) for c in range(1, CELL_GRID_N - 1)]


def _precompute_masks() -> list[np.ndarray]:
    """Enumerate readable LR-symmetric 3x3 masks with an active center pixel."""
    weights = [len(group) for group in _LR_GROUPS]
    masks: list[np.ndarray] = []
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = CELL_INNER * CELL_INNER - active_count
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
                    continue
                mask = np.ones((CELL_INNER, CELL_INNER), dtype=bool)
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if not bool(mask[1, 1]):
                    continue
                if not mask[0, :].any() or not mask[2, :].any() or not mask[:, 0].any() or not mask[:, 2].any():
                    continue
                masks.append(mask)
    return masks


def _precompute_center_sets() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """All center choices whose centered 3x3 coarse blocks are pairwise disjoint."""
    by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for count in SEED_COUNTS:
        valid: list[tuple[tuple[int, int], ...]] = []
        for combo in combinations(_VALID_CENTERS, count):
            ok = True
            for (r1, c1), (r2, c2) in combinations(combo, 2):
                if abs(r1 - r2) < CELL_INNER and abs(c1 - c2) < CELL_INNER:
                    ok = False
                    break
            if ok:
                valid.append(combo)
        by_count[count] = valid
    return by_count


_MASKS = _precompute_masks()
_CENTER_SETS = _precompute_center_sets()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((CELL_INNER, CELL_INNER), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(choice == choices[0] for choice in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, overrides: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Construct a full board using majority as the default cell interior."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for (i, j), pat in overrides.items():
        paste(grid, pat, *_cell_origin(i, j))
    return grid


def _active_cells_for_center(center: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    ci, cj = center
    radius = CELL_INNER // 2
    cells: list[tuple[int, int]] = []
    for di, dj in np.argwhere(mask):
        cells.append((int(ci + di - radius), int(cj + dj - radius)))
    return cells


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, ordinary_active, marker = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = random.choice(_MASKS).copy()
    ordinary = np.where(mask, ordinary_active, shared).astype(int)
    source = np.where(mask, marker, shared).astype(int)

    seed_count = random.choice(SEED_COUNTS)
    centers = random.choice(_CENTER_SETS[seed_count])

    out_overrides: dict[tuple[int, int], np.ndarray] = {}
    inp_overrides: dict[tuple[int, int], np.ndarray] = {}
    noncenter_offsets = [
        (int(di), int(dj))
        for di, dj in np.argwhere(mask)
        if not (int(di) == CELL_INNER // 2 and int(dj) == CELL_INNER // 2)
    ]
    min_ordinary = min(GIVEN_ORDINARY_PER_SEED)
    assert len(noncenter_offsets) >= min_ordinary

    for center in centers:
        active_cells = _active_cells_for_center(center, mask)
        for cell in active_cells:
            out_overrides[cell] = ordinary

        ordinary_count = random.choice([count for count in GIVEN_ORDINARY_PER_SEED if count <= len(noncenter_offsets)])
        shown_offsets = random.sample(noncenter_offsets, ordinary_count)
        ci, cj = center
        inp_overrides[(int(ci), int(cj))] = source
        for di, dj in shown_offsets:
            cell = (int(ci + di - CELL_INNER // 2), int(cj + dj - CELL_INNER // 2))
            inp_overrides[cell] = ordinary

    inp = _build_grid(separator, majority, inp_overrides)
    out = _build_grid(separator, majority, out_overrides)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo D1/1+D7/1 input")
