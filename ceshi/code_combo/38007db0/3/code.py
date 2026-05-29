# Combo variant of 38007db0 (index 3)
# combo_spec: D2/3+D4/2
# Name: 对称性破坏差异提取

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and execution trace
---------------------------------------
- The lattice framing from the seed task stays unchanged: a regular 1-pixel grid
  partitions the image into square cells.
- The source-selection rule changes first. In one lattice row, the non-target
  cells are all left-right symmetric sprites, but they are not all identical.
  A single cell is derived from one symmetric sprite by adding a few one-sided
  foreground pixels, so it is the only cell that breaks the row's symmetry rule.
- The copy rule changes second. The output no longer copies that whole odd cell.
  Instead it keeps only the pixels that make the cell asymmetric. Concretely, if
  `odd` is the broken sprite, we compare it to its mirror image and write only
  the visible side of the disagreement:
      diff[p] = odd[p] if odd[p] != mirror(odd)[p] else bg
- Small example on a 3x3 inner sprite:
      symmetric cell:  .#.
                       #.#
                       .#.
      odd cell:        .##
                       #.#
                       .#.
      mirrored odd:    ##.
                       #.#
                       .#.
      output diff:     ..#
                       ...
                       ...
  The seed task would have output the whole odd cell. This combo outputs only
  the rightmost added pixel that destroyed mirror symmetry.

b) Complexity calibration
-------------------------
- I keep exactly one symmetry family for the whole task: left-right reflection.
  That is enough to realize D2/3 while avoiding a jump in rule complexity.
- Symmetric distractors vary within a row, so the task does not collapse back to
  "pick the singleton picture". This is important because otherwise the fusion
  would silently degenerate to the original task plus a post-diff step.
- The asymmetry is limited to 2-3 added pixels, all inside mirrored pairs that
  were background on both sides. This makes the output sparse and legible while
  keeping the defect visually obvious.

c) Implicit-assumption audit against the seed solver
----------------------------------------------------
- Seed assumption: "the desired cell is the one whose bytes occur once".
  Broken. In the combo rows, several symmetric distractors are byte-unique.
- Seed assumption: "the whole odd cell should be copied to output".
  Broken. Only its asymmetry-causing pixels should survive.
- Seed assumption: "a repeated majority template is recoverable by frequency".
  Also broken. There need not be any repeated full cell pattern at all.
- Still valid: lattice rows/cols are uniform, the lattice color is the most
  common uniform-line color, cell slices are the gaps between lattice lines, and
  background can be recovered as the most common non-lattice color.

d) Placement legality
---------------------
- All edits happen strictly inside the `(cell_size-2) x (cell_size-2)` inner
  sprite area. The 1-pixel background padding and the lattice lines are never
  overwritten.
- `_build_lattice` places cells on a regular grid, so no overlap or out-of-
  bounds placement can occur.
- Broken variants are formed only by turning mirrored bg/bg pairs into sp/bg on
  one chosen side, so the result always remains inside the anchor cell.

e) Visual observability
-----------------------
- A human can inspect one row and see that all but one cell are mirror-symmetric.
  The odd cell contains a few extra one-sided pixels.
- In the output, only those extra pixels remain. Across examples this reveals a
  coherent rule: find the non-symmetric sprite, then strip away the symmetric
  part and keep only the symmetry-breaking additions.

f) Visual cleanliness
---------------------
- Only three colors are used: lattice, background, foreground sprite.
- The output keeps the same clean single-column lattice as the seed task.
- Difference pixels are sparse, so the result reads as a highlight rather than a
  second dense object competing with the input structure.

g) Color-rule audit
-------------------
- There is no new color mapping. The visible difference pixels keep the sprite
  color; all other positions become background.
- To avoid invisible deletions, the generator only uses bg->sp additions when
  breaking symmetry. The output therefore always contains visible foreground.

h) Post-implementation anti-degeneration metric
-----------------------------------------------
- Metric 1: count how many byte-singletons exist per row. The combo generator is
  constructed so this count is usually greater than 1, proving the seed
  singleton-frequency rule no longer isolates the target.
- Metric 2: compare the combo output with a hypothetical "copy the whole broken
  cell" output. They should differ on essentially every sample because the combo
  keeps only the asymmetric delta.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: exactly one cell per row has positive left-right mismatch score.
  Guarantee: every distractor is symmetric, and the odd cell is created by at
  least two one-sided additions.
- Assumption: visible asymmetry pixels are the odd cell's foreground pixels at
  mismatching locations.
  Guarantee: broken variants add only foreground on one side of bg/bg mirrored
  pairs, so `cell != mirror(cell)` identifies exactly the intended delta.
- Assumption: enough symmetric variants exist to fill each row with distinct
  distractors.
  Guarantee: symmetric variants come from toggling mirrored orbits of a shared
  base sprite, and failed attempts are rejected before returning a sample.
"""


def _lattice_slices(
    g: np.ndarray,
) -> tuple[int, int, int, list[int], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    lattice = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted(r for r in row_lines if int(g[r, 0]) == lattice)
    col_lines = sorted(c for c in col_lines if int(g[0, c]) == lattice)

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    return H, W, lattice, row_lines, r_slices, c_slices


def _lr_mismatch_score(cell: np.ndarray) -> int:
    return int(np.count_nonzero(cell != np.fliplr(cell)))


def _visible_symmetry_diff(cell: np.ndarray, bg: int) -> np.ndarray:
    mirror = np.fliplr(cell)
    diff = np.full(cell.shape, int(bg), dtype=int)
    mask = cell != mirror
    diff[mask] = cell[mask]
    return diff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Per lattice row, find the only left-right non-symmetric cell and output only
    its visible asymmetry-causing pixels in a single-column lattice.
    """
    g = np.array(grid, dtype=int)
    H, _, lattice, row_lines, r_slices, c_slices = _lattice_slices(g)
    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(lattice)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_diffs: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        scores = [_lr_mismatch_score(cell) for cell in cells]
        odd_idxs = [idx for idx, score in enumerate(scores) if int(score) > 0]
        assert len(odd_idxs) == 1
        chosen_diffs.append(_visible_symmetry_diff(cells[int(odd_idxs[0])], bg))

    out = np.full((H, int(cell_w + 2)), int(bg), dtype=int)
    out[:, 0] = int(lattice)
    out[:, -1] = int(lattice)
    for r in row_lines:
        out[int(r), :] = int(lattice)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_diffs[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    size = int(inner.shape[0] + 2)
    cell = np.full((size, size), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = int(cells[0][0].shape[0])
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _mirror_orbits(
    inner_size: int,
) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    all_orbits: list[tuple[tuple[int, int], ...]] = []
    pair_orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(inner_size):
        for c in range((inner_size + 1) // 2):
            mc = inner_size - 1 - c
            if c == mc:
                all_orbits.append(((r, c),))
            else:
                orbit = ((r, c), (r, mc))
                all_orbits.append(orbit)
                pair_orbits.append(orbit)
    return all_orbits, pair_orbits


def _apply_orbit_toggles(
    base_inner: np.ndarray,
    orbits: list[tuple[tuple[int, int], ...]],
    toggle_ids: tuple[int, ...],
    bg: int,
    sp: int,
) -> np.ndarray:
    inner = base_inner.copy()
    for orb_id in toggle_ids:
        for r, c in orbits[int(orb_id)]:
            inner[int(r), int(c)] = int(sp) if int(inner[int(r), int(c)]) == int(bg) else int(bg)
    return inner


def _rand_subset_ids(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    max_k = min(3, count)
    k = int(rng.integers(0, max_k + 1))
    if k == 0:
        return ()
    return tuple(sorted(int(x) for x in rng.choice(count, size=k, replace=False)))


def _gen_symmetric_base(
    rng: np.random.Generator,
    inner_size: int,
    bg: int,
    sp: int,
    orbits: list[tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    inner_area = int(inner_size * inner_size)
    min_fill = math.ceil(0.3 * inner_area)
    max_fill = math.floor(0.65 * inner_area)
    for _ in range(200):
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        for orbit in orbits:
            if rng.random() < 0.45:
                for r, c in orbit:
                    inner[int(r), int(c)] = int(sp)
        fill = int(np.count_nonzero(inner == int(sp)))
        if min_fill <= fill <= max_fill and 0 < fill < inner_area:
            empty_pairs = 0
            for left, right in (orb for orb in orbits if len(orb) == 2):
                if int(inner[left]) == int(bg) and int(inner[right]) == int(bg):
                    empty_pairs += 1
            if empty_pairs >= 3:
                return inner
    return None


def _collect_symmetric_variants(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    orbits: list[tuple[tuple[int, int], ...]],
    bg: int,
    sp: int,
    count: int,
) -> list[np.ndarray] | None:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    base_cell = _make_cell(base_inner, bg)
    variants.append(base_cell)
    seen.add(base_cell.tobytes())

    for _ in range(500):
        if len(variants) >= count:
            return variants[:count]
        toggle_ids = _rand_subset_ids(rng, len(orbits))
        inner = _apply_orbit_toggles(base_inner, orbits, toggle_ids, bg, sp)
        cell = _make_cell(inner, bg)
        sig = cell.tobytes()
        if sig in seen or _lr_mismatch_score(cell) != 0:
            continue
        seen.add(sig)
        variants.append(cell)
    return None


def _candidate_break_sets(
    anchor_inner: np.ndarray,
    pair_orbits: list[tuple[tuple[int, int], tuple[int, int]]],
    bg: int,
) -> list[tuple[tuple[int, int], ...]]:
    candidates: list[tuple[tuple[int, int], ...]] = []
    for count in range(2, min(3, len(pair_orbits)) + 1):
        for orb_ids in itertools.combinations(range(len(pair_orbits)), count):
            side_options: list[list[tuple[int, int]]] = []
            valid = True
            for orb_idx in orb_ids:
                left, right = pair_orbits[int(orb_idx)]
                local: list[tuple[int, int]] = []
                if int(anchor_inner[left]) == int(bg) and int(anchor_inner[right]) == int(bg):
                    local.append((int(orb_idx), 0))
                    local.append((int(orb_idx), 1))
                if not local:
                    valid = False
                    break
                side_options.append(local)
            if not valid:
                continue
            for choice in itertools.product(*side_options):
                coords = [pair_orbits[int(orb_idx)][int(side)] for orb_idx, side in choice]
                rs = [int(r) for r, _ in coords]
                cs = [int(c) for _, c in coords]
                bbox_area = (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)
                if bbox_area <= count + 3:
                    candidates.append(tuple((int(orb_idx), int(side)) for orb_idx, side in choice))
    return candidates


def _break_symmetry_with_additions(
    rng: np.random.Generator,
    sym_cell: np.ndarray,
    pair_orbits: list[tuple[tuple[int, int], tuple[int, int]]],
    bg: int,
    sp: int,
    used_diffs: set[bytes],
    used_odds: set[bytes],
) -> tuple[np.ndarray, np.ndarray] | None:
    anchor_inner = sym_cell[1:-1, 1:-1]
    candidates = _candidate_break_sets(anchor_inner, pair_orbits, bg)
    if not candidates:
        return None

    order = list(rng.permutation(len(candidates)))
    for idx in order:
        additions = candidates[int(idx)]
        odd_inner = anchor_inner.copy()
        diff_inner = np.full(anchor_inner.shape, int(bg), dtype=int)
        for orb_idx, side in additions:
            r, c = pair_orbits[int(orb_idx)][int(side)]
            odd_inner[int(r), int(c)] = int(sp)
            diff_inner[int(r), int(c)] = int(sp)

        odd_cell = _make_cell(odd_inner, bg)
        diff_cell = _make_cell(diff_inner, bg)
        odd_sig = odd_cell.tobytes()
        diff_sig = diff_cell.tobytes()
        if odd_sig in used_odds or diff_sig in used_diffs:
            continue
        if _lr_mismatch_score(odd_cell) <= 0:
            continue
        if not np.array_equal(_visible_symmetry_diff(odd_cell, bg), diff_cell):
            continue
        return odd_cell, diff_cell
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(4, 6))

    inner_size = int(cell_size - 2)
    orbits, pair_orbits = _mirror_orbits(inner_size)
    if len(pair_orbits) < 2:
        return None

    base_inner = _gen_symmetric_base(rng, inner_size, bg, sp, orbits)
    if base_inner is None:
        return None

    output_cells: list[np.ndarray] = []
    input_rows: list[list[np.ndarray]] = []
    used_diffs: set[bytes] = set()
    used_odds: set[bytes] = set()

    for _ in range(n_rows):
        sym_cells = _collect_symmetric_variants(rng, base_inner, orbits, bg, sp, n_cols - 1)
        if sym_cells is None:
            return None

        anchor = sym_cells[int(rng.integers(0, len(sym_cells)))]
        broken = _break_symmetry_with_additions(rng, anchor, pair_orbits, bg, sp, used_diffs, used_odds)
        if broken is None:
            return None
        odd_cell, diff_cell = broken
        used_odds.add(odd_cell.tobytes())
        used_diffs.add(diff_cell.tobytes())
        output_cells.append(diff_cell.copy())

        row_cells = [cell.copy() for cell in sym_cells]
        odd_col = int(rng.integers(1, n_cols))
        row_cells.insert(odd_col, odd_cell.copy())
        input_rows.append(row_cells)

    inp = _build_lattice(input_rows, lattice, bg)
    out = _build_lattice([[cell] for cell in output_cells], lattice, bg)
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
