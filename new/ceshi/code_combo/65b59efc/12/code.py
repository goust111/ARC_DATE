# Combo variant of 65b59efc (index 12)
# combo_spec: D9/0+D12/1
# Name: 列位置翻转混色标记

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task: the middle band effectively assigns one template color to one physical tile column.
#      If a blue marker appears anywhere inside column 1, every other marker in that same column is also
#      blue, so local slot (lr, lc) only tells us where to stamp and marker color tells us which template.
#    - Fused task here: the same physical tile column can contain several marker colors at once, and that
#      physical column now also determines the geometric action:
#        column 0 -> identity
#        column 1 -> horizontal flip
#        column 2 -> vertical flip
#      Example with N=4:
#        - top templates use key colors 2, 3, 7
#        - middle column 1 contains markers 2 at local (0, 3) and 7 at local (2, 1)
#        - those two markers no longer imply the same template; instead they mean
#            use template 2 at output block (0, 3) after horizontal flip
#            use template 7 at output block (2, 1) after horizontal flip
#      The color lookup is per marker, while the flip rule is per physical column.
#
# b) Complexity calibration
#    - The transform family is capped at three cases only: no flip, horizontal flip, vertical flip. This
#      is enough to be visible without exploding the hypothesis space.
#    - K is fixed to 3 so every physical column has a unique, repeated transform meaning and all three
#      cases can appear in one sample.
#    - The generator forces mixed-color evidence in every physical column, and at least one column contains
#      all three template colors. This prevents the task from collapsing back to the original one-color-per-
#      column behavior.
#
# c) Implicit-assumption audit against the original main()
#    - Original assumption "one tile column has one marker color" is false. The new solver must inspect
#      every marker independently and cannot cache a per-column template id.
#    - Original assumption "render tile = template recolored directly" is false for columns 1 and 2; the
#      solver must transform the template mask using the marker's physical column before stamping.
#    - Original assumption "K may be 3 or 4 without further meaning" is unsafe because D12/1 only defines
#      three flip states. The generator therefore fixes K=3 and main() expects exactly three tile columns.
#    - Original slot conflict check only needed to guard against different templates sharing one output
#      block. After fusion, even the same template in two physical columns would conflict because their
#      transforms differ, so the new solver forbids any duplicate slot at all.
#
# d) Placement legality
#    - The three template tiles, the three marker columns, and the three palette columns stay aligned in
#      the original vertical three-band layout with dashed separators, so parsing remains stable.
#    - Marker slots are still one-pixel placements within the N x N middle tiles. The generator chooses a
#      unique output slot for every marker, so no output block receives overlapping stamps.
#    - Palette swatches remain confined to the bottom band inside their own template columns.
#
# e) Visual observability
#    - Mixed columns are directly visible because a single marker tile column contains repeated pixels of
#      multiple colors.
#    - Column-dependent flipping is also visible because the same template color appears in different
#      physical columns, and the corresponding output blocks show mirrored versions of the same shape.
#    - To make those mirrors reliable to humans, the generator rejects templates that are invariant under
#      left-right or up-down flips.
#
# f) Visual cleanliness
#    - No extra object families are added; the puzzle still reads as templates on top, markers in the
#      middle, palette below.
#    - Only marker pixels become denser. Because markers are still isolated single pixels on a clean N x N
#      lattice, the legend remains readable even when colors mix inside one physical column.
#
# g) New assumptions introduced by main(), and generator support
#    - main() assumes exactly three physical marker columns exist and correspond to id/hflip/vflip in that
#      order; generate_input() fixes K=3 and keeps the original left-to-right column order.
#    - main() assumes every template key color is unique; generate_input() samples distinct template colors.
#    - main() assumes transformed stamps are visually distinct; generate_input() only keeps templates whose
#      identity, horizontal flip, and vertical flip are pairwise different.
#    - main() assumes each output slot is claimed by at most one marker; generate_input() uses unique slots.
#    - main() assumes mixed-color evidence actually exists; generate_input() forces every physical column to
#      contain at least two template colors and at least one column to contain all three.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (4, 5)
_K = 3
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.45, 0.78)
_SYMMETRY_OPTIONS = [None, "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.65
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where mixed marker colors share columns and column position controls flips.

    Input structure:
    - The grid is split into exactly three tile columns of width N, separated by dashed separator rows and columns.
    - The top band contains three binary templates, each drawn in its own template color.
    - The middle band contains marker pixels. Marker colors may mix inside the same physical tile column, so one column is no longer tied to one template.
    - The bottom band contains one palette swatch per template column, giving the repaint color for that template.

    Rule:
    - Use a marker's color to choose the template, and use the marker's local (row, col) inside its physical tile column to choose the output block.
    - Before stamping, transform the chosen template according to the marker's physical tile column: left column = unchanged, middle column = horizontal flip, right column = vertical flip.
    - Recolor the transformed template with that template's palette color and place it into the chosen N x N output block.

    Key insight: marker color and marker column now carry different pieces of information: color picks the template, while physical column picks the flip operation.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert len(sep_cols) == _K - 1 and len(sep_rows) >= 2

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    assert len(starts) == _K

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        key = vals.pop()
        mask = sub == key
        assert mask.any() and (~mask).any()
        tcols.append(key)
        tmasks.append(mask)
    assert len(set(tcols)) == _K
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: set[tuple[int, int]] = set()

    for phys, st in enumerate(starts):
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner, "each output block must be owned by exactly one marker"
                slot_owner.add((lr, lc))
                tile_mask = _transform_mask(tmasks[j], phys)
                tile = np.where(tile_mask, pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _transform_mask(mask: np.ndarray, phys: int) -> np.ndarray:
    if phys == 0:
        return mask.copy()
    if phys == 1:
        return np.fliplr(mask)
    if phys == 2:
        return np.flipud(mask)
    raise ValueError(f"unexpected physical column {phys}")


def _sample_template(n: int, color: int) -> np.ndarray | None:
    for _ in range(120):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry=random.choice(_SYMMETRY_OPTIONS),
            color_palette=[color],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == color
        if not mask.any() or not (~mask).any():
            continue
        hflip = np.fliplr(mask)
        vflip = np.flipud(mask)
        if np.array_equal(mask, hflip) or np.array_equal(mask, vflip):
            continue
        if np.array_equal(hflip, vflip):
            continue
        return mask
    return None


def _render_tile(mask: np.ndarray, color: int, phys: int) -> np.ndarray:
    return np.where(_transform_mask(mask, phys), color, _BG)


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc, phys in placements:
        paste(out, _render_tile(templates[j], palette_colors[j], phys), rr * n, cc * n)
    return out


def _build_original_baseline(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc, _phys in placements:
        paste(out, _render_tile(templates[j], palette_colors[j], 0), rr * n, cc * n)
    return out


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


def _sample_templates(n: int) -> tuple[list[int], list[np.ndarray]] | None:
    tcols = random.sample(_AVAILABLE_COLORS, _K)
    templates: list[np.ndarray] = []
    for col in tcols:
        mask = _sample_template(n, col)
        if mask is None:
            return None
        templates.append(mask)
    return tcols, templates


def _sample_placements(n: int) -> list[tuple[int, int, int, int]] | None:
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    min_marks = max(7, int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)

    for _ in range(160):
        random.shuffle(all_slots)
        chosen_slots = all_slots[:mark_count]
        support = [
            (0, 0),
            (1, 0),
            (1, 1),
            (2, 1),
            (2, 2),
            (0, 2),
            (2, 0),
        ]
        placements: list[tuple[int, int, int, int]] = []
        for (rr, cc), (j, phys) in zip(chosen_slots[: len(support)], support):
            placements.append((j, rr, cc, phys))

        for rr, cc in chosen_slots[len(support) :]:
            j = random.randrange(_K)
            phys = random.randrange(_K)
            placements.append((j, rr, cc, phys))

        by_phys = [set() for _ in range(_K)]
        by_template = [set() for _ in range(_K)]
        for j, _rr, _cc, phys in placements:
            by_phys[phys].add(j)
            by_template[j].add(phys)

        if min(len(cols) for cols in by_phys) < 2:
            continue
        if max(len(cols) for cols in by_phys) < 3:
            continue
        if min(len(cols) for cols in by_template) < 2:
            continue
        return placements

    return None


def _column_mix_counts(placements: list[tuple[int, int, int, int]]) -> list[int]:
    out = []
    for phys in range(_K):
        out.append(len({j for j, _rr, _cc, p in placements if p == phys}))
    return out


def _transformed_marker_count(
    templates: list[np.ndarray],
    placements: list[tuple[int, int, int, int]],
) -> int:
    count = 0
    for j, _rr, _cc, phys in placements:
        if phys == 0:
            continue
        if not np.array_equal(_transform_mask(templates[j], phys), templates[j]):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    sampled = _sample_templates(n)
    if sampled is None:
        return None
    tcols, templates = sampled

    palette_pool = [c for c in _AVAILABLE_COLORS if c not in tcols]
    if len(palette_pool) < _K:
        return None
    pcols = random.sample(palette_pool, _K)

    placements = _sample_placements(n)
    if placements is None:
        return None

    expected = _build_expected(n, templates, pcols, placements)
    baseline = _build_original_baseline(n, templates, pcols, placements)
    if np.array_equal(expected, baseline):
        return None

    if _transformed_marker_count(templates, placements) < 3:
        return None
    if min(_column_mix_counts(placements)) < 2:
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = _K * n + (_K - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, _K, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    for j, rr, cc, phys in placements:
        grid[n + 1 + rr, _raw_col(phys, cc, n)] = tcols[j]

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
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
