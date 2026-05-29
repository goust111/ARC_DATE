# Combo variant of 65b59efc (index 3)
# combo_spec: D2/0+D12/3
# Name: 反向万花筒铺贴
#
# Design analysis for the fused rule:
# a) Concrete behavior trace:
#    With N=4, a marker at local (0,1) used to stamp output block (0,1) directly.
#    In this combo it first maps to the reversed block index (3,2). Because row 3 is
#    odd and col 2 is even, the chosen template is vertically flipped before stamping.
#    A marker at local (2,2) maps to (1,1); both indices are odd, so that block gets
#    both flips, i.e. a 180-degree rotation. The solver therefore must infer two linked
#    changes: destination reversal and parity-based mirroring at that destination.
# b) Complexity calibration:
#    If templates were horizontally/vertically symmetric, odd-row/odd-col flips would
#    become invisible. The generator rejects such masks and also forces markers into odd
#    rows and odd cols so the kaleidoscope effect appears repeatedly inside each sample.
# c) Hidden assumptions audit:
#    - Top templates still use one distinct template color each.
#    - Bottom palette swatches still encode one recolor per template column.
#    - Middle tile columns still have uniform marker color and remain one-to-one with
#      templates, so color-to-template decoding is unchanged.
#    - The original "slot_owner" assumption must move from source local coords to the
#      reversed destination coords, because those are now the semantic block addresses.
#    - Mirroring parity must be computed after the coordinate reversal, not before.
# d) Placement legality:
#    Generation is output-first: distinct destination blocks are sampled, then mapped
#    back through the same involution (r,c) <-> (N-1-r,N-1-c) to place markers. This
#    guarantees in-bounds, non-overlapping markers and legal output blocks.
# e) Visual observability:
#    Each accepted sample must differ from the original rule, from D2-only, and from
#    D12-only on the same input. This prevents accidental regression to a simpler task.
# f) Visual cleanliness:
#    The output still consists of clean N-by-N blocks; no extra overlays are added.
#    Separation between legend bands remains the original dashed structure.
# g) Inferrability:
#    Colors keep their original structural roles (template colors vs palette colors),
#    so the only new burden on the solver is noticing reversed placement plus parity-
#    controlled mirroring, both of which are repeated across many blocks per sample.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import paste, random_sprite  # noqa: E402


# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- CFG: grid dimensions --
_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

# -- CFG: template generation --
_DENSITY_RANGE = (0.45, 0.82)
_TEMPLATE_CONNECTIVITY = 8
_TEMPLATE_ATTEMPTS = 80

# -- CFG: marker placement --
_MIN_SLOT_COVERAGE = 0.70

# -- CFG: palette swatch --
_SWATCH_DOUBLE_PROB = 0.35

# -- CFG: retry --
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _reverse_slot(n: int, rr: int, cc: int) -> tuple[int, int]:
    return n - 1 - rr, n - 1 - cc


def _kaleidoscope(mask: np.ndarray, block_r: int, block_c: int) -> np.ndarray:
    cur = mask
    if block_r % 2 == 1:
        cur = np.flipud(cur)
    if block_c % 2 == 1:
        cur = np.fliplr(cur)
    return cur


def _render_stamp(mask: np.ndarray, color: int, bg: int, block_r: int, block_c: int) -> np.ndarray:
    warped = _kaleidoscope(mask, block_r, block_c)
    return np.where(warped, color, bg)


def _sample_template(n: int, color: int) -> np.ndarray:
    for _ in range(_TEMPLATE_ATTEMPTS):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry=random.choice([None, "not_symmetric"]),
            color_palette=[color],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == color
        if (
            np.any(mask)
            and not np.array_equal(mask, np.fliplr(mask))
            and not np.array_equal(mask, np.flipud(mask))
            and not np.array_equal(mask, np.rot90(mask, 2))
        ):
            return mask
    raise RuntimeError("failed to sample an oriented template")


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


def _count_diff_blocks(a: np.ndarray, b: np.ndarray, n: int) -> int:
    diff = 0
    for rr in range(n):
        for cc in range(n):
            sa = a[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
            sb = b[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
            if not np.array_equal(sa, sb):
                diff += 1
    return diff


def _render_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    src_records: list[tuple[int, int, int]],
    *,
    reverse_map: bool,
    kaleidoscope: bool,
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, src_r, src_c in src_records:
        dst_r, dst_c = _reverse_slot(n, src_r, src_c) if reverse_map else (src_r, src_c)
        mask = templates[j]
        if kaleidoscope:
            tile = _render_stamp(mask, pcols[j], _BG, dst_r, dst_c)
        else:
            tile = np.where(mask, pcols[j], _BG)
        paste(expected, tile, dst_r * n, dst_c * n)
    return expected


def _fusion_metrics(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    src_records: list[tuple[int, int, int]],
) -> dict[str, int]:
    combo = _render_expected(n, templates, pcols, src_records, reverse_map=True, kaleidoscope=True)
    original = _render_expected(n, templates, pcols, src_records, reverse_map=False, kaleidoscope=False)
    d2_only = _render_expected(n, templates, pcols, src_records, reverse_map=True, kaleidoscope=False)
    d12_only = _render_expected(n, templates, pcols, src_records, reverse_map=False, kaleidoscope=True)

    moved_slots = sum(
        1 for _, src_r, src_c in src_records if (src_r, src_c) != _reverse_slot(n, src_r, src_c)
    )
    transformed_slots = 0
    for j, src_r, src_c in src_records:
        dst_r, dst_c = _reverse_slot(n, src_r, src_c)
        plain = np.where(templates[j], pcols[j], _BG)
        warped = _render_stamp(templates[j], pcols[j], _BG, dst_r, dst_c)
        if not np.array_equal(plain, warped):
            transformed_slots += 1

    return {
        "moved_slots": moved_slots,
        "transformed_slots": transformed_slots,
        "diff_vs_original": _count_diff_blocks(combo, original, n),
        "diff_vs_d2_only": _count_diff_blocks(combo, d2_only, n),
        "diff_vs_d12_only": _count_diff_blocks(combo, d12_only, n),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a mirrored stamp legend: read templates, markers, and palette, then build the tiled output.

    Input structure:
    - The grid is split into K tile columns of width N, separated by 1-column dashed dividers.
    - Rows 0..N-1 contain the K template masks, one per tile column.
    - Rows N+1..2N contain sparse marker pixels; each tile column uses one marker color.
    - Rows 2N+2..end contain one palette swatch per tile column, and rows N / 2N+1 are dashed separators.

    Rule:
    - A marker color still selects which template/palette pair to use.
    - Its local marker coordinate (lr, lc) is mapped to output block (N-1-lr, N-1-lc), not (lr, lc).
    - After choosing that output block, odd block columns flip the template left-right and odd block rows
      flip it top-bottom; when both parities are odd the stamp is flipped on both axes.

    Key insight:
    - The legend itself is unchanged in layout, but the marker position must be interpreted backwards,
      and the final stamp orientation depends on the parity of the reversed destination block.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if (g[:, c] == sep).sum() >= h // 2]
    sep_rows = [r for r in range(h) if (g[r, :] == sep).sum() >= w // 2]
    assert sep_cols and len(sep_rows) >= 2

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    assert k >= 1

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    mid_r0 = n + 1

    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                block_r, block_c = _reverse_slot(n, lr, lc)
                assert (
                    (block_r, block_c) not in slot_owner or slot_owner[(block_r, block_c)] == j
                ), "different templates must occupy different destination blocks"
                slot_owner[(block_r, block_c)] = j
                tile = _render_stamp(tmasks[j], pcols[j], bg, block_r, block_c)
                out[block_r * n : (block_r + 1) * n, block_c * n : (block_c + 1) * n] = tile

    return out


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

    templates = [_sample_template(n, col) for col in tcols]

    all_dst_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_dst_slots)

    # Force every kaleidoscope branch to appear at least once: no flip, row-only,
    # col-only, and both flips. This keeps the fused rule observable in each sample.
    parity_slots = {
        (0, 0): [(rr, cc) for rr, cc in all_dst_slots if rr % 2 == 0 and cc % 2 == 0],
        (1, 0): [(rr, cc) for rr, cc in all_dst_slots if rr % 2 == 1 and cc % 2 == 0],
        (0, 1): [(rr, cc) for rr, cc in all_dst_slots if rr % 2 == 0 and cc % 2 == 1],
        (1, 1): [(rr, cc) for rr, cc in all_dst_slots if rr % 2 == 1 and cc % 2 == 1],
    }
    used_dst_slots = [random.choice(parity_slots[key]) for key in ((0, 0), (1, 0), (0, 1), (1, 1))]

    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)
    for slot in all_dst_slots:
        if len(used_dst_slots) >= mark_count:
            break
        if slot not in used_dst_slots:
            used_dst_slots.append(slot)
    random.shuffle(used_dst_slots)

    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    dst_placements = [(j, rr, cc) for (rr, cc), j in zip(used_dst_slots, assign_pool)]
    src_records = [(j, *_reverse_slot(n, rr, cc)) for j, rr, cc in dst_placements]

    metrics = _fusion_metrics(n, templates, pcols, src_records)
    if (
        metrics["moved_slots"] < 2
        or metrics["transformed_slots"] < 2
        or metrics["diff_vs_original"] < 2
        or metrics["diff_vs_d2_only"] < 1
        or metrics["diff_vs_d12_only"] < 1
    ):
        return None

    expected = _render_expected(n, templates, pcols, src_records, reverse_map=True, kaleidoscope=True)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, src_r, src_c in src_records:
        grid[n + 1 + src_r, _raw_col(tile_assign[j], src_c, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    color_counts = Counter(map(int, grid.flatten()))
    if any(color_counts[_SEP] <= color_counts[col] for col in _AVAILABLE_COLORS if col in color_counts):
        return None

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
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
