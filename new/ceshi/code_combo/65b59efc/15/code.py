# Combo variant of 65b59efc (index 15)
# combo_spec: D1/0+D2/2+D12/0
# Name: 全旋转推理链

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import paste

# Design analysis for the fused variant
# a) Concrete behavior:
#    - Keep the original three-band layout: templates on top, markers in the middle, palette at the bottom.
#    - The top band still contains K monochrome NxN templates.
#    - The middle band now contains one connected marker shape per tile column. Its color is non-informative:
#      every marker column uses the same marker color. The solver must identify the template by shape, allowing
#      0/90/180/270 rotations.
#    - Each occupied marker cell at local coordinate (lr, lc) no longer maps to output block (lr, lc). Instead it
#      maps to (lc, N-1-lr), i.e. one clockwise coordinate rotation.
#    - The template stamped into that output block is then rotated again by the output block row rr: row 0 keeps the
#      template, row 1 rotates it 90 degrees clockwise, row 2 rotates it 180 degrees, row 3 rotates it 270 degrees.
#    - Example with N=4: if a matched marker pixel sits at (1, 3), it lands in output block (3, 2); because rr=3,
#      the recolored template is rotated 270 degrees clockwise before being pasted there.
# b) Complexity calibration:
#    - If marker colors still identified templates, D1/0 would be visually fake, so the generator uses one shared
#      marker color that is different from all template colors.
#    - If templates were rotation-symmetric, both shape matching and row-based stamp rotation would collapse, so the
#      generator only emits rotation-asymmetric templates.
#    - If two templates were the same up to rotation, shape matching would become ambiguous, so all templates are
#      distinct modulo the full rotation group.
# c) Broken original assumptions:
#    - Original `tcol_to_j[v]` color lookup is invalid because middle-band colors are no longer template IDs.
#    - Original direct slot write at `(lr, lc)` is invalid because slot coordinates must be rotated first.
#    - Original `tile = template` is invalid because the pasted stamp orientation now depends on the destination row.
# d) Placement legality:
#    - The generator picks three templates and searches marker rotations whose occupied local cells are pairwise
#      disjoint. This preserves the original "one template per output slot" assumption after the slot-coordinate
#      rotation, because that rotation is bijective on the NxN local grid.
#    - All objects remain inside their own NxN tile columns; separators and palette swatches still avoid overlap.
# e) Visual observability:
#    - Shape matching is visible because the same marker color appears in every middle column while the silhouettes
#      match different top templates only after rotation.
#    - Coordinate rotation is visible because occupied output blocks line up by marker columns, not by marker rows.
#    - Row-based stamp rotation is visible because every occupied block row in the output shows a consistent stamp
#      orientation class.
# f) Visual cleanliness:
#    - K is fixed to 3 and N to 4. That leaves enough space for three distinct asymmetric shapes without crowding the
#      4x4 slot grid.
#    - Template colors, palette colors, and the shared marker color are all distinct, so the three bands remain easy
#      to parse by eye.
# g) Change inferability:
#    - No hidden ordering by numeric color ID is used. The only control variables are visible geometry, visible local
#      position, and visible output-row grouping.

# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a rotated stamp map by matching marker silhouettes to templates, then render rotated stamps.

    Input structure:
    - The grid is split into K tile columns of width N, separated by dashed separator columns.
    - Top band (rows 0..N-1): K monochrome template masks, one per tile column.
    - Middle band (rows N+1..2N): one connected marker shape per tile column; all marker columns use the same color.
    - Bottom band (rows 2N+2..): K palette swatches aligned with the top templates.
    - Rows N and 2N+1 are dashed separator rows.

    Rule:
    - For each middle-band marker shape, find which top template it matches under a 0/90/180/270 rotation.
    - Every occupied local marker cell (lr, lc) maps to output block (lc, N-1-lr), not to (lr, lc).
    - The matched template is recolored with its palette color and then rotated clockwise by the destination block row.

    Key insight: The middle band no longer encodes template choice by color; it encodes it by rotated shape, and the
    local marker coordinates feed a second rotation chain before the stamp is finally row-rotated in the output.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if int((g[:, c] == sep).sum()) >= H // 2]
    sep_rows = [r for r in range(H) if int((g[r, :] == sep).sum()) >= W // 2]
    assert sep_cols and sep_rows

    n = int(sep_cols[0])
    starts = [0] + [int(c + 1) for c in sep_cols]
    k = len(starts)

    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        tcol = vals.pop()
        tmasks.append(sub == tcol)

    pal_r0 = int(sep_rows[-1] + 1)
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:H, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(int(vals.pop()))

    mid_r0 = int(sep_rows[0] + 1)
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    stamp_cache: dict[tuple[int, int], np.ndarray] = {}
    matched_templates: set[int] = set()

    for st in starts:
        sub = g[mid_r0 : mid_r0 + n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        marker_col = int(vals.pop())
        marker_mask = sub == marker_col
        assert marker_mask.any()

        matches = [j for j, tmask in enumerate(tmasks) if any(np.array_equal(marker_mask, np.rot90(tmask, rot)) for rot in range(4))]
        assert len(matches) == 1
        j = int(matches[0])
        assert j not in matched_templates
        matched_templates.add(j)

        for lr, lc in np.argwhere(marker_mask):
            rr, cc = _rotate_slot_cw(int(lr), int(lc), n)
            assert (rr, cc) not in slot_owner, "different templates must occupy different rotated slots"
            slot_owner[(rr, cc)] = j

            turns = int(rr % 4)
            key = (j, turns)
            if key not in stamp_cache:
                rotated = _rotate_mask_cw(tmasks[j], turns)
                stamp_cache[key] = np.where(rotated, pcols[j], bg)
            paste(out, stamp_cache[key], rr * n, cc * n)

    assert len(matched_templates) == k
    return out


# --------------- generator ---------------

_BG = 0
_SEP = 5
_N = 4
_K = 3
_PAL_H_CHOICES = [2, 3]
_PAL_H_WEIGHTS = [8, 1]
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _rotate_slot_cw(lr: int, lc: int, n: int) -> tuple[int, int]:
    return int(lc), int(n - 1 - lr)


def _rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -int(turns) % 4)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _rotation_signature_set(mask: np.ndarray) -> set[tuple[tuple[int, ...], ...]]:
    return {_mask_signature(np.rot90(mask, rot)) for rot in range(4)}


def _is_rotation_asymmetric(mask: np.ndarray) -> bool:
    return len(_rotation_signature_set(mask)) == 4


def _same_up_to_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    return not _rotation_signature_set(a).isdisjoint(_rotation_signature_set(b))


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


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)

_TEMPLATE_BANK: list[np.ndarray] = [
    _mask_from_rows(["1110", "0000", "0000", "0000"]),  # I3
    _mask_from_rows(["1000", "1100", "0000", "0000"]),  # L3
    _mask_from_rows(["1110", "0100", "0000", "0000"]),  # T4
    _mask_from_rows(["1000", "1110", "0000", "0000"]),  # L4
    _mask_from_rows(["1100", "0110", "0000", "0000"]),  # Z4
]


def _compatible_layouts() -> list[tuple[tuple[int, ...], list[tuple[int, ...]]]]:
    layouts: list[tuple[tuple[int, ...], list[tuple[int, ...]]]] = []
    for combo in itertools.combinations(range(len(_TEMPLATE_BANK)), _K):
        templates = [_TEMPLATE_BANK[i] for i in combo]
        if any(_same_up_to_rotation(templates[i], templates[j]) for i in range(_K) for j in range(i + 1, _K)):
            continue

        valid_rots: list[tuple[int, ...]] = []
        for rots in itertools.product(range(4), repeat=_K):
            marker_masks = [np.rot90(mask, rot) for mask, rot in zip(templates, rots)]
            used = np.zeros_like(marker_masks[0], dtype=bool)
            ok = True
            for mask in marker_masks:
                if bool((used & mask).any()):
                    ok = False
                    break
                used |= mask
            if not ok:
                continue

            if sum(int(rot != 0) for rot in rots) < 2:
                continue

            union_rows = {int(r) for mask in marker_masks for r, _ in np.argwhere(mask)}
            union_cols = {int(c) for mask in marker_masks for _, c in np.argwhere(mask)}
            if len(union_rows) < 4 or len(union_cols) < 4:
                continue

            valid_rots.append(tuple(int(rot) for rot in rots))

        if valid_rots:
            layouts.append((tuple(int(i) for i in combo), valid_rots))
    return layouts


_COMPATIBLE_LAYOUTS = _compatible_layouts()
assert _COMPATIBLE_LAYOUTS


def _build_expected(n: int, templates: list[np.ndarray], pcols: list[int], marker_masks: list[np.ndarray]) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    stamp_cache: dict[tuple[int, int], np.ndarray] = {}
    slot_owner: dict[tuple[int, int], int] = {}

    for j, marker_mask in enumerate(marker_masks):
        for lr, lc in np.argwhere(marker_mask):
            rr, cc = _rotate_slot_cw(int(lr), int(lc), n)
            assert (rr, cc) not in slot_owner
            slot_owner[(rr, cc)] = j
            turns = int(rr % 4)
            key = (j, turns)
            if key not in stamp_cache:
                stamp_cache[key] = np.where(_rotate_mask_cw(templates[j], turns), pcols[j], _BG)
            paste(out, stamp_cache[key], rr * n, cc * n)

    return out


def _gen_attempt() -> np.ndarray | None:
    n = _N
    k = _K
    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)

    colors = random.sample(_AVAILABLE_COLORS, 1 + 2 * k)
    marker_col = int(colors[0])
    tcols = [int(v) for v in colors[1 : 1 + k]]
    pcols = [int(v) for v in colors[1 + k : 1 + 2 * k]]

    combo, rot_options = random.choice(_COMPATIBLE_LAYOUTS)
    templates = [_TEMPLATE_BANK[i].copy() for i in combo]
    marker_rots = random.choice(rot_options)
    marker_masks = [np.rot90(mask, rot) for mask, rot in zip(templates, marker_rots)]
    expected = _build_expected(n, templates, pcols, marker_masks)

    grid = np.full((core_h, core_w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, marker_mask in enumerate(marker_masks):
        paste(grid, np.where(marker_mask, marker_col, _BG), n + 1, tile_assign[j] * (n + 1))

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        st = j * (n + 1)
        size = 2 if random.random() < 0.35 else 1
        c0 = st + (n - size) // 2
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
