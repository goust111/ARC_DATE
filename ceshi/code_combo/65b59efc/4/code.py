# Combo variant of 65b59efc (index 4)
# combo_spec: D2/1+D12/0
# Name: 转置定位旋转印章
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVERS = ROOT / "solvers"
if str(SOLVERS) not in sys.path:
    sys.path.insert(0, str(SOLVERS))

from arc_utils import paste, random_sprite


# Design analysis for the fused rule:
# - Concrete trace: with N=4, a marker at local (lr, lc) = (1, 3) used to stamp output block
#   (1, 3). In this combo it first transposes to block (3, 1), then row 3 implies a 270-degree
#   clockwise stamp rotation. The marker column therefore drives both destination row and rotation.
# - The two dimensions are coupled, not independent: after transposition, the new block row is lc,
#   so the rotation chain is computed from lc rather than from the original lr.
# - Hidden-assumption audit:
#   * Original slot uniqueness was checked on (lr, lc); here the meaningful uniqueness check is on
#     the transposed destination (lc, lr), although the map remains bijective.
#   * Fixed-orientation templates are no longer safe. The generator must avoid rotationally
#     symmetric masks, otherwise the row-conditioned rotation becomes invisible and the task
#     degenerates toward the simpler transpose-only variant.
#   * Mostly diagonal marker layouts would weaken the transpose cue because (lr, lc) == (lc, lr);
#     the generator therefore seeds many off-diagonal slots and guarantees every local column
#     appears multiple times.
# - Legality and cleanliness:
#   * Input markers stay single-pixel and non-overlapping because each template still owns one
#     middle-band tile column and every local slot is used at most once.
#   * Output stamps remain non-overlapping because each occupied slot maps to exactly one N-by-N
#     destination block in the tiled output canvas.
#   * Every template is forced to appear in multiple transposed rows so the same shape is seen under
#     different rotations, making the row-dependent transform human-observable in 2-3 examples.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a transposed stamp legend: recover template masks, marker assignments, and palette colors,
    then render the enlarged stamp mosaic.

    Input structure:
    - The input is split into K tile columns of width N, separated by 1-cell dashed dividers.
    - Rows 0..N-1 hold the K template masks, one template per tile column.
    - Rows N+1..2N hold sparse marker pixels; each tile column still uses one marker color only.
    - Rows below the second dashed separator contain one palette swatch per tile column.

    Rule:
    - Marker color selects which template/palette pair to use.
    - A marker at local coordinates (lr, lc) maps to output block (lc, lr), so row and column are
      swapped before placement.
    - The chosen template is recolored with its palette color and rotated by 90-degree clockwise
      steps according to the transposed output row: row 0 -> 0 deg, row 1 -> 90 deg, row 2 -> 180
      deg, row 3 -> 270 deg, then repeating modulo 4 for larger N.

    Key insight: The marker's local column now has a double role: after transposition it becomes the
    output block row, and that very row index also decides the stamp rotation.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = int(sep_cols[0])
    starts = [0] + [int(c + 1) for c in sep_cols]
    k = len(starts)

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = int(vals.pop())
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = int(sep_rows[-1] + 1)
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(int(vals.pop()))

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = int(tcol_to_j[v])
                dr, dc = int(lc), int(lr)
                assert (dr, dc) not in slot_owner or slot_owner[(dr, dc)] == j, (
                    "different templates must occupy different transposed slots"
                )
                slot_owner[(dr, dc)] = j
                rot_k = dr % 4
                mask = _rotate_mask(tmasks[j], rot_k)
                tile = np.where(mask, pcols[j], bg)
                out[dr * n : (dr + 1) * n, dc * n : (dc + 1) * n] = tile

    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (4, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.42, 0.78)
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.65
_SWATCH_DOUBLE_PROB = 0.35
_MAX_TEMPLATE_TRIES = 80
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _rotate_mask(mask: np.ndarray, turns_cw: int) -> np.ndarray:
    k = turns_cw % 4
    if k == 0:
        return mask.copy()
    if k == 1:
        return np.rot90(mask, 3)
    if k == 2:
        return np.rot90(mask, 2)
    return np.rot90(mask, 1)


def _is_rotation_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, _rotate_mask(mask, k)) for k in (1, 2, 3))


def _sample_template(n: int, color: int) -> np.ndarray:
    for _ in range(_MAX_TEMPLATE_TRIES):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry="not_symmetric",
            color_palette=[color],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == color
        if _is_rotation_distinct(mask):
            return mask
    raise RuntimeError("failed to sample a rotation-distinct template")


def _render_stamp(mask: np.ndarray, color: int, out_row: int) -> np.ndarray:
    rot = _rotate_mask(mask, out_row % 4)
    return np.where(rot, color, _BG)


def _build_expected(
    n: int, templates: list[np.ndarray], pcols: list[int], placements: list[tuple[int, int, int]]
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, lr, lc in placements:
        dr, dc = int(lc), int(lr)
        tile = _render_stamp(templates[j], pcols[j], dr)
        paste(expected, tile, dr * n, dc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _seed_slots(n: int) -> list[tuple[int, int]]:
    seeded: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for lc in range(n):
        for lr in ((lc + 1) % n, (lc + 2) % n):
            slot = (int(lr), int(lc))
            if slot not in seen:
                seeded.append(slot)
                seen.add(slot)
    return seeded


def _choose_slots(n: int) -> list[tuple[int, int]]:
    seeded = _seed_slots(n)
    all_slots = [(lr, lc) for lr in range(n) for lc in range(n)]
    remaining = [slot for slot in all_slots if slot not in set(seeded)]
    random.shuffle(remaining)
    min_marks = max(len(seeded), int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)
    return seeded + remaining[: mark_count - len(seeded)]


def _assign_templates(k: int, n: int, used_slots: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    slots_by_col: dict[int, list[tuple[int, int]]] = {lc: [] for lc in range(n)}
    for slot in used_slots:
        slots_by_col[slot[1]].append(slot)
    for lc in range(n):
        random.shuffle(slots_by_col[lc])

    placements: list[tuple[int, int, int]] = []
    claimed: set[tuple[int, int]] = set()

    # Every template appears in two distinct transposed rows so the same shape shows multiple rotations.
    for j in range(k):
        for lc in (j % n, (j + 2) % n):
            while slots_by_col[lc]:
                slot = slots_by_col[lc].pop()
                if slot not in claimed:
                    claimed.add(slot)
                    placements.append((j, slot[0], slot[1]))
                    break
            else:
                raise RuntimeError("seeded slot coverage invariant broken")

    leftovers = [slot for slot in used_slots if slot not in claimed]
    random.shuffle(leftovers)
    assign_pool = (list(range(k)) * ((len(leftovers) + k - 1) // k + 1))[: len(leftovers)]
    random.shuffle(assign_pool)
    for slot, j in zip(leftovers, assign_pool):
        placements.append((j, slot[0], slot[1]))

    random.shuffle(placements)
    return placements


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    try:
        templates = [_sample_template(n, col) for col in tcols]
    except RuntimeError:
        return None

    used_slots = _choose_slots(n)
    placements = _assign_templates(k, n, used_slots)
    expected = _build_expected(n, templates, pcols, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, lr, lc in placements:
        grid[n + 1 + lr, _raw_col(tile_assign[j], lc, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused 65b59efc input")
