# Combo variant of 65b59efc (index 25)
# combo_spec: D2/0+D5/2
# Name: 空间色彩双反转

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


# Design analysis for combo D2/0 + D5/2
# --------------------------------------
# a) Concrete behavior trace
#    - Original 65b59efc uses three aligned bands: top templates, middle markers, bottom palette swatches.
#      A marker color chooses one template, and its local marker slot (lr, lc) directly selects output block
#      (lr, lc). The rendered block is a binary recolor: template foreground -> palette color, template
#      background -> global background.
#    - The fused rule changes both halves of that decoding:
#        1. D2/0 reverses the slot map, so local slot (lr, lc) now stamps block
#           (N-1-lr, N-1-lc). Top-left marker position lands in the output's bottom-right block.
#        2. D5/2 reverses the recolor polarity, so the template keeps its own original template color on
#           active cells, while inactive cells inside the N x N stamp are filled with the palette color.
#    - Small N=3 example:
#        template mask/color =
#          [[2,2,0],
#           [0,2,0],
#           [0,2,2]]
#        palette color = 7
#        marker color = 2 at local slot (0, 2)
#      Original output would place
#          [[7,7,0],
#           [0,7,0],
#           [0,7,7]]
#      into output block (0, 2). The fused output instead places
#          [[2,2,7],
#           [7,2,7],
#           [7,2,2]]
#      into block (2, 0). The position flips diagonally and the foreground/background roles flip inside
#      the stamp itself.
#
# b) Complexity calibration
#    - The input layout is intentionally unchanged, so the only new burden is inferring two inversions from
#      repeated evidence rather than learning a new parsing scheme.
#    - To keep D5/2 visible, every template mask is required to leave at least two inactive cells and every
#      palette color differs from its paired template color. This prevents the fused stamps from collapsing
#      to a single flat color.
#    - To keep D2/0 visible, marker placements are not allowed to be perfectly invariant under
#      (rr, cc) -> (N-1-rr, N-1-cc). At least one occupied slot lacks its reversed partner, so a human can
#      see that the block arrangement is diagonally reversed rather than unchanged.
#
# c) Implicit-assumption audit against the original main()
#    - `assert len(vals) == 1` for top templates still holds because each top tile remains a one-color
#      binary mask. No change needed there.
#    - The original code assumed `tile = np.where(mask, palette, bg)`. That assumption is broken by D5/2;
#      the fused solver must render `tile = np.where(mask, template_color, palette_color)`.
#    - The original code wrote each tile at `(lr, lc)`. D2/0 breaks that direct mapping, so the solver must
#      write to `(N-1-lr, N-1-lc)` instead.
#    - The original collision check keyed by local slot can be kept because reversal is bijective, but the
#      more faithful fused assumption is "different templates may not claim the same reversed destination
#      block". The new solver therefore keys slot ownership on the reversed destination.
#    - A new D5/2-specific hidden risk is template solidity: if a mask is full, palette fill disappears.
#      The generator explicitly rejects full or almost-full masks to preserve the signal.
#
# d) Placement legality
#    - Templates still occupy disjoint N-wide columns in the top band, markers live only in the middle
#      band, and palette swatches live only in the bottom band. No new overlap channel is introduced.
#    - The reversed slot map is a bijection on the N x N slot lattice, so reversing coordinates cannot
#      create out-of-bounds writes or many-to-one collisions.
#    - Palette swatches remain 1-2 contiguous pixels centered inside each tile column, so palette decoding
#      is still unambiguous after adding the color inversion.
#
# e) Visual observability
#    - The same template color seen in the top band reappears as the foreground color of every stamped
#      block, while the bottom swatch color visibly fills the complementary cells. This makes the color
#      inversion readable from a few examples without relying on numeric color ids.
#    - Output occupancy is forced to be diagonally asymmetric, so the position rule cannot be mistaken for
#      the original direct mapping by simple coincidence.
#    - Because both cues repeat across several stamps in each example, a human can separately infer
#      "template color survives inside the stamp" and "marker coordinates land in the opposite block".
#
# f) Visual cleanliness
#    - The puzzle keeps the original clean three-band legend; no extra markers, overlays, or counting tasks
#      are added.
#    - Template colors and their paired palette colors are always different, so each stamped block is a
#      crisp two-color negative rather than muddled by same-color fills.
#    - The generator keeps slot coverage moderate rather than always full, which leaves visible background
#      gaps between some stamped blocks and helps the reversed arrangement read clearly.
#
# g) New assumptions introduced by main(), and generator support
#    - main() now assumes D5/2 matters: each template must have inactive cells and palette color different
#      from template color. generate_input() enforces both.
#    - main() now assumes D2/0 matters: the reversed placement should not accidentally match the direct
#      placement on the same input. generate_input() rejects any sample whose fused output equals the
#      D5/2-only ablation.
#    - main() still assumes each middle-band physical tile column contains only one marker color, and that
#      marker colors match top-band template colors one-to-one. generate_input() preserves that original
#      construction.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.40, 0.80)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500


def _dash_cell(pos: int, n: int) -> bool:
    """True iff position `pos` receives a separator pip for tile size `n`."""
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    """Count separator pips across a row/column span."""
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    """Map a tile-column index and local column to the absolute input-grid column."""
    return tile_col * (n + 1) + local_c


def _reverse_slot(n: int, rr: int, cc: int) -> tuple[int, int]:
    """D2/0 slot mapping: top-left local slots map to bottom-right output blocks."""
    return n - 1 - rr, n - 1 - cc


def _render_combo_tile(mask: np.ndarray, template_color: int, palette_color: int) -> np.ndarray:
    """D5/2 stamp rendering: keep template color on-mask, fill off-mask with palette color."""
    return np.where(mask, template_color, palette_color)


def _render_normal_tile(mask: np.ndarray, palette_color: int, bg: int) -> np.ndarray:
    """Base-task rendering for ablation checks."""
    return np.where(mask, palette_color, bg)


def _sample_template(n: int, color: int) -> np.ndarray | None:
    """Sample a one-color binary template that leaves visible negative space."""
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
        if area < n + 1:
            continue
        if area > n * n - 2:
            continue
        return mask
    return None


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    tcols: list[int],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
    *,
    reverse_slots: bool,
    color_swap: bool,
    bg: int,
) -> np.ndarray:
    """Render an output grid for the fused rule or one of its ablations."""
    expected = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        dr, dc = _reverse_slot(n, rr, cc) if reverse_slots else (rr, cc)
        if color_swap:
            tile = _render_combo_tile(templates[j], tcols[j], pcols[j])
        else:
            tile = _render_normal_tile(templates[j], pcols[j], bg)
        paste(expected, tile, dr * n, dc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    """Draw the original dashed separator rows and columns in-place."""
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _sample_slots(n: int, k: int) -> list[tuple[int, int]] | None:
    """Pick occupied local slots with enough coverage and visible asymmetry under reversal."""
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    max_marks = n * n - 1 if n * n > 1 else 1
    if min_marks > max_marks:
        max_marks = min_marks

    for _ in range(120):
        mark_count = random.randint(min_marks, max_marks)
        used_slots = random.sample(all_slots, mark_count)
        used_set = set(used_slots)
        asymmetric = any(_reverse_slot(n, rr, cc) not in used_set for rr, cc in used_slots)
        if not asymmetric:
            continue
        return used_slots
    return None


def _combo_metrics(
    n: int,
    templates: list[np.ndarray],
    tcols: list[int],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> dict[str, int]:
    """Quantify how strongly the fused sample differs from the original and single-dimension ablations."""
    combo = _build_expected(
        n,
        templates,
        tcols,
        pcols,
        placements,
        reverse_slots=True,
        color_swap=True,
        bg=_BG,
    )
    d2_only = _build_expected(
        n,
        templates,
        tcols,
        pcols,
        placements,
        reverse_slots=True,
        color_swap=False,
        bg=_BG,
    )
    d5_only = _build_expected(
        n,
        templates,
        tcols,
        pcols,
        placements,
        reverse_slots=False,
        color_swap=True,
        bg=_BG,
    )
    original = _build_expected(
        n,
        templates,
        tcols,
        pcols,
        placements,
        reverse_slots=False,
        color_swap=False,
        bg=_BG,
    )
    return {
        "vs_original_diff": int(np.count_nonzero(combo != original)),
        "vs_d2_only_diff": int(np.count_nonzero(combo != d2_only)),
        "vs_d5_only_diff": int(np.count_nonzero(combo != d5_only)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend whose output applies both a spatial inversion and a color negative.

    Input structure:
    - The grid is split into K tile columns of width N, separated by 1-column dashed separators.
    - Rows 0..N-1 hold K one-color binary templates, one template per tile column.
    - Rows N+1..2N hold marker pixels; each physical tile column uses one marker color only.
    - Rows 2N+2..end hold K palette swatches aligned with the template columns.
    - Dashed separator rows sit at rows N and 2N+1, and separator columns sit between tile columns.

    Rule:
    - A marker's color selects which top-band template to use.
    - A marker's local slot (lr, lc) inside its tile column maps to output block
      (N-1-lr, N-1-lc), so the slot lattice is reversed from top-left to bottom-right.
    - The chosen template is rendered as a two-color negative: active template cells keep the original
      template color, while inactive cells are filled with that template column's palette color.

    Key insight: The middle band still tells you "which template goes where", but both halves invert:
    marker coordinates map to the opposite output block, and the palette color fills the template's holes
    instead of replacing the template foreground.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == len(tcols)
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
                dr, dc = _reverse_slot(n, lr, lc)
                assert (dr, dc) not in slot_owner or slot_owner[(dr, dc)] == j, (
                    "different templates must occupy different reversed slots"
                )
                slot_owner[(dr, dc)] = j
                tile = _render_combo_tile(tmasks[j], tcols[j], pcols[j])
                out[dr * n : (dr + 1) * n, dc * n : (dc + 1) * n] = tile

    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)

    tcols = random.sample(_AVAILABLE_COLORS, k)
    remaining = [c for c in _AVAILABLE_COLORS if c not in tcols]
    pcols = random.sample(remaining, k)

    templates: list[np.ndarray] = []
    for col in tcols:
        mask = _sample_template(n, col)
        if mask is None:
            return None
        templates.append(mask)

    used_slots = _sample_slots(n, k)
    if used_slots is None:
        return None

    assign_pool = (list(range(k)) * ((len(used_slots) + k - 1) // k))[: len(used_slots)]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(
        n,
        templates,
        tcols,
        pcols,
        placements,
        reverse_slots=True,
        color_swap=True,
        bg=_BG,
    )
    metrics = _combo_metrics(n, templates, tcols, pcols, placements)
    if metrics["vs_d2_only_diff"] == 0 or metrics["vs_d5_only_diff"] == 0 or metrics["vs_original_diff"] == 0:
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

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
