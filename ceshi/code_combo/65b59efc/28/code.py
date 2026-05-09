# Combo variant of 65b59efc (index 28)
# combo_spec: D16/1+D20/2
# Name: 干扰模板擦除印章

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


# Design analysis for combo D16/1 + D20/2
# ----------------------------------------
# a) Concrete behavior trace
#    - Original 65b59efc: every top tile is a real template; every middle-band non-bg pixel is a stamp
#      marker. Marker color picks the template, marker local slot (lr, lc) picks the output block, and the
#      bottom swatch recolors the chosen template.
#    - Fused rule: the top band now contains 1-2 extra template columns that never receive normal marker-color
#      references in the middle band, so they are pure distractors. In addition, the middle band contains one
#      extra color that does not match any top template color. Those extra-color pixels do not stamp; they
#      indicate output blocks that must be cleared after all normal stamps are rendered.
#    - Small example with N=3, five top templates A/B/C/D/E, but only A/C/E used:
#        normal markers: A at (0,0), C at (1,2), E at (2,1)
#        erase marker: special color X at local slot (1,2)
#      First render blocks (0,0)->A, (1,2)->C, (2,1)->E. Then erase block (1,2), so the final output keeps
#      only A and E. Templates B and D are visible in the legend but never contribute.
#
# b) Complexity calibration
#    - The distractor change is visible because at least one full template/palette column has no normal marker
#      references at all; these empty middle columns repeat within each example.
#    - The erase change is visible because erase markers always target already-stamped blocks, so the final
#      output differs from the no-erase ablation on nonzero pixels rather than being a no-op.
#    - The coding burden stays manageable: there is still only one extra semantic color in the middle band,
#      and normal markers still use direct template-color lookup.
#
# c) Implicit-assumption audit against the original main()
#    - Original assumption "all top templates are used" is false. The fused solver must allow unused template
#      columns and only stamp templates that are actually referenced by normal markers.
#    - Original assumption "each middle tile column has one uniform marker color" is false. Used columns may
#      contain both their normal template color and the extra erase color.
#    - Original assumption "every middle-band non-bg color names a template" is false. Exactly one middle-band
#      color is outside the top template-color set, and it means erase-after-render.
#    - Original assumption "any parsed slot stays stamped" is false. The solver must render all normal stamps
#      first and then blank every erase-marked destination block.
#
# d) Placement legality
#    - Normal placements still use disjoint local slots, so no two templates claim the same output block.
#    - Each erase mark is written into a used column that is different from the column owning that slot, so it
#      never overwrites the normal marker needed to create the stamped block it later removes.
#    - Distractor columns remain entirely empty in the middle band, which keeps them readable as unused legend
#      entries and prevents overlap with erase marks.
#
# e) Visual observability
#    - A human can see extra top templates whose colors never appear as normal middle markers.
#    - The erase color is visually singled out because it appears in the middle band but nowhere in the top
#      templates or palette swatches.
#    - Comparing input/output examples reveals a consistent two-stage story: some marked blocks would have been
#      stamped, but the extra-color marks remove them in the final output.
#
# f) Visual cleanliness
#    - The original three-band legend is preserved; no extra rows or hidden metadata are added.
#    - Distractor columns are cleanly empty in the middle band, rather than noisy, so the unused-template clue
#      is legible.
#    - Erase marks are sparse and use a single color, avoiding clutter inside the already dense middle band.
#
# g) New assumptions introduced by the fused solver, and generator support
#    - main() assumes exactly one extra middle-band color exists beyond the top template colors. generate_input()
#      reserves one dedicated erase color and forbids it from top/palette use.
#    - main() assumes erase marks target meaningful blocks. generate_input() samples erase slots only from
#      occupied normal placements and rejects samples where erasure would not change the output.
#    - main() assumes distractor templates are truly unused. generate_input() keeps their middle columns fully
#      empty and rejects any sample without at least one such column.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_LAYOUT_CHOICES = [(4, 1), (5, 1), (5, 2)]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.52, 0.88)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 600
_LAYOUT_CURSOR = 0


def _dash_cell(pos: int, n: int) -> bool:
    """True iff position `pos` receives a separator pip for tile size `n`."""
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    """Count separator pips across a row/column span."""
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    """Map a tile-column index and local column to an absolute input-grid column."""
    return tile_col * (n + 1) + local_c


def _sample_template(n: int, color: int) -> np.ndarray | None:
    """Sample a one-color binary template with visible holes."""
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
        area = int(mask.sum())
        if area < n:
            continue
        if area > n * n - 1:
            continue
        return mask
    return None


def _render_tile(mask: np.ndarray, palette_color: int, bg: int) -> np.ndarray:
    """Recolor a binary template mask with its palette color."""
    return np.where(mask, palette_color, bg)


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
    erase_slots: set[tuple[int, int]],
    bg: int,
) -> np.ndarray:
    """Render all normal stamps first, then erase the requested destination blocks."""
    expected = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        paste(expected, _render_tile(templates[j], pcols[j], bg), rr * n, cc * n)
    for rr, cc in erase_slots:
        expected[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = bg
    return expected


def _draw_separators(grid: np.ndarray, n: int, k_total: int, core_h: int, core_w: int) -> None:
    """Draw the original dashed separator rows and columns in-place."""
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k_total - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _infer_sep(g: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    """Infer separator color plus separator-row/column indices from the dashed layout."""
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    best = None
    for color, _ in counts.items():
        color = int(color)
        if color == bg:
            continue
        rows = [r for r in range(h) if int(np.count_nonzero(g[r, :] == color)) >= max(2, w // 2)]
        cols = [c for c in range(w) if int(np.count_nonzero(g[:, c] == color)) >= max(2, h // 2)]
        if len(rows) >= 2 and cols:
            score = (len(rows) + len(cols), counts[color])
            if best is None or score > best[0]:
                best = (score, color, rows, cols)
    if best is None:
        raise AssertionError("failed to infer separator structure")
    _, sep, rows, cols = best
    return int(sep), rows, cols


def _pick_template_layout() -> tuple[int, int]:
    """Cycle through the supported layout branches so repeated samples cover them all."""
    global _LAYOUT_CURSOR
    k_total, distractor_count = _LAYOUT_CHOICES[_LAYOUT_CURSOR % len(_LAYOUT_CHOICES)]
    _LAYOUT_CURSOR += 1
    return k_total, distractor_count


def _assign_slots(n: int, used_js: list[int]) -> list[tuple[int, int, int]] | None:
    """Assign disjoint local slots to used templates with moderate coverage."""
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    min_marks = max(len(used_js), int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    max_marks = n * n
    if min_marks > max_marks:
        return None

    for _ in range(120):
        mark_count = random.randint(min_marks, max_marks)
        chosen_slots = random.sample(all_slots, mark_count)
        js = (used_js * ((mark_count + len(used_js) - 1) // len(used_js)))[:mark_count]
        random.shuffle(js)
        placements = [(j, rr, cc) for (rr, cc), j in zip(chosen_slots, js)]
        counts = Counter(j for j, _, _ in placements)
        if all(counts[j] >= 1 for j in used_js):
            return placements
    return None


def _select_erase_slots(placements: list[tuple[int, int, int]], used_js: list[int]) -> set[tuple[int, int]] | None:
    """Choose occupied slots to erase without wiping out a template's only placement."""
    by_j: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for j, rr, cc in placements:
        by_j[j].append((rr, cc))

    candidates = [(j, rr, cc) for j, rr, cc in placements if len(by_j[j]) >= 2]
    if not candidates:
        return None

    erase_count = random.randint(1, min(2, len(candidates)))
    random.shuffle(candidates)
    chosen: list[tuple[int, int, int]] = []
    remaining = {j: len(by_j[j]) for j in used_js}
    used_slots: set[tuple[int, int]] = set()

    for j, rr, cc in candidates:
        slot = (rr, cc)
        if slot in used_slots:
            continue
        if remaining[j] <= 1:
            continue
        chosen.append((j, rr, cc))
        used_slots.add(slot)
        remaining[j] -= 1
        if len(chosen) == erase_count:
            break

    if not chosen:
        return None
    return {(rr, cc) for _, rr, cc in chosen}


def _combo_metrics(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
    erase_slots: set[tuple[int, int]],
    *,
    k_total: int,
    k_used: int,
) -> dict[str, int]:
    """Quantify whether both fused controls are active and visible."""
    combo = _build_expected(n, templates, pcols, placements, erase_slots, _BG)
    no_erase = _build_expected(n, templates, pcols, placements, set(), _BG)
    return {
        "distractor_count": k_total - k_used,
        "erase_count": len(erase_slots),
        "erase_pixel_delta": int(np.count_nonzero(combo != no_erase)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k_total, distractor_count = _pick_template_layout()
    k_used = k_total - distractor_count

    erase_color = random.choice(_AVAILABLE_COLORS)
    non_erase = [c for c in _AVAILABLE_COLORS if c != erase_color]
    tcols = random.sample(non_erase, k_total)
    pcols = [random.choice([c for c in non_erase if c != tcols[j]]) for j in range(k_total)]

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k_total * n + (k_total - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates: list[np.ndarray] = []
    for col in tcols:
        mask = _sample_template(n, col)
        if mask is None:
            return None
        templates.append(mask)

    template_ids = list(range(k_total))
    random.shuffle(template_ids)
    used_js = template_ids[:k_used]
    distractor_js = template_ids[k_used:]
    placements = _assign_slots(n, used_js)
    if placements is None:
        return None

    erase_slots = _select_erase_slots(placements, used_js)
    if erase_slots is None:
        return None

    used_phys = list(used_js)
    slot_owner_phys = {(rr, cc): j for j, rr, cc in placements}
    erase_placements: list[tuple[int, int, int]] = []
    for rr, cc in sorted(erase_slots):
        owner_phys = slot_owner_phys[(rr, cc)]
        choices = [phys for phys in used_phys if phys != owner_phys]
        if not choices:
            return None
        erase_placements.append((random.choice(choices), rr, cc))

    metrics = _combo_metrics(
        n,
        templates,
        pcols,
        placements,
        erase_slots,
        k_total=k_total,
        k_used=k_used,
    )
    if metrics["distractor_count"] < 1 or metrics["erase_count"] < 1 or metrics["erase_pixel_delta"] <= 0:
        return None

    expected = _build_expected(n, templates, pcols, placements, erase_slots, _BG)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k_total, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    mid_r0 = n + 1
    for j, rr, cc in placements:
        grid[mid_r0 + rr, _raw_col(j, cc, n)] = tcols[j]
    for phys, rr, cc in erase_placements:
        grid[mid_r0 + rr, _raw_col(phys, cc, n)] = erase_color

    pal_row = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[pal_row, c0 : c0 + size] = col

    middle_nonbg_by_col = []
    for phys in range(k_total):
        sub = grid[mid_r0 : mid_r0 + n, phys * (n + 1) : phys * (n + 1) + n]
        middle_nonbg_by_col.append(int(np.count_nonzero(sub != _BG)))
    if sum(v == 0 for v in middle_nonbg_by_col) != len(distractor_js):
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a stamped legend that mixes unused top templates with post-render erase marks.

    Input structure:
    - The grid is divided into K tile columns of width N separated by dashed 1-cell separator columns.
    - Top band (rows 0..N-1): K monochrome template masks, but 1-2 of them are distractors that receive no
      normal marker references.
    - Middle band (rows N+1..2N): normal marker pixels still use template colors, while one extra color marks
      output blocks that must be erased after stamping.
    - Bottom band (rows 2N+2..): K palette swatches, one aligned with each top template column.
    - Dashed separator rows remain at rows N and 2N+1.

    Rule:
    - Use each normal middle-band pixel's color to choose a top template, and use its local slot (lr, lc)
      inside its physical tile column to choose the destination N x N output block.
    - Render all such stamps with the palette color belonging to that template's top column.
    - After all normal stamps are placed, clear every output block whose local slot is marked by the unique
      extra middle-band erase color.

    Key insight: Only colors that appear in the top templates can create stamps; the lone middle-band color
    absent from the top band means "erase this block later", and some top templates are simply never used.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    sep, sep_rows, sep_cols = _infer_sep(g, bg)
    assert sep_rows and sep_cols

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k_total = len(starts)

    tcols: list[int] = []
    templates: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1, "each top tile must contain one template color"
        col = vals.pop()
        tcols.append(col)
        templates.append(sub == col)
    assert len(set(tcols)) == k_total, "template colors must be distinct"
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1, "each palette tile must contain one swatch color"
        pcols.append(vals.pop())

    mid_r0 = n + 1
    marker_cells: list[tuple[int, int, int, int]] = []
    extra_colors = set()
    template_color_set = set(tcols)
    for phys, st in enumerate(starts):
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                marker_cells.append((v, lr, lc, phys))
                if v not in template_color_set:
                    extra_colors.add(v)

    assert len(extra_colors) == 1, "expected exactly one erase color in the middle band"
    erase_color = extra_colors.pop()

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    erase_slots: set[tuple[int, int]] = set()

    for v, lr, lc, _phys in marker_cells:
        if v == erase_color:
            erase_slots.add((lr, lc))
            continue
        j = tcol_to_j[v]
        slot = (lr, lc)
        assert slot not in slot_owner or slot_owner[slot] == j, "different templates cannot share a slot"
        slot_owner[slot] = j
        out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = _render_tile(templates[j], pcols[j], bg)

    for rr, cc in erase_slots:
        assert (rr, cc) in slot_owner, "erase marks must target stamped blocks"
        out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = bg

    return out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
