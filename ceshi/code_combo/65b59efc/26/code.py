# Combo variant of 65b59efc (index 26)
# combo_spec: D1/0+D3/0+D12/1
# Name: 色块形状翻转匹配

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste  # noqa: E402


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original 65b59efc uses a single colored marker pixel. Its color directly selects one template and
#      its local slot directly selects one output block. The stamped tile is the raw template recolored by
#      the bottom palette swatch.
#    - Fused rule here replaces that with a 2x2 marker patch. Inside the patch, one fixed anchor color
#      marks the output slot, while the remaining three cells form a tiny shape. That tiny shape is the
#      same coarse 2x2 silhouette as one top-band template, so template lookup is now by shape matching.
#    - The physical marker column also matters: column 0 stamps the template as-is, column 1 stamps a
#      left-right flip, column 2 stamps an up-down flip, and later columns repeat that 3-way cycle.
#      The marker patch itself is flipped by the same column rule, so the solver must undo that flip to
#      recover the canonical tiny shape before matching it to the top template.
#    - Small example with N=4:
#        top template coarse shape = [[0,1],
#                                     [1,1]]
#        canonical marker patch    = [[A,S],
#                                     [S,S]]
#        if that template's markers live in tile column 1, the visible marker patch becomes
#                                     [[S,A],
#                                      [S,S]]
#        and the output stamp is the horizontal flip of the full 4x4 template.
#      The original task would have treated a single colored cell as enough. The fused task instead needs
#      "find anchor -> undo column flip on the 2x2 patch -> match the tiny shape -> stamp flipped tile".
#
# b) Complexity calibration
#    - The template family is intentionally restricted to four coarse 2x2 silhouettes, each expanded into
#      an NxN quadrant shape. That keeps shape matching legible to humans while still giving enough unique
#      templates for the three physical tile columns used here.
#    - Only three transform cases are used (`id`, `hflip`, `vflip`), one per physical tile column. Keeping
#      K fixed at 3 avoids introducing an undocumented "fourth column repeats some earlier flip" rule.
#    - Marker patches are 2x2 and non-overlapping, so the middle band stays sparse and readable. A denser
#      free-placement scheme would merge patches and destroy the "tiny shape" cue.
#
# c) Implicit-assumption audit against the original main()
#    - Original assumption "a non-bg middle-band cell is itself one marker" is false now. The solver must
#      group cells into 2x2 patches around a unique anchor-colored cell.
#    - Original assumption "marker color maps directly to template index" is false. All marker patches use
#      the same three dedicated marker colors; only the coarse shape matters.
#    - Original assumption "template orientation is fixed" is false. The output tile and the visible
#      marker patch both depend on the physical marker column's flip mode.
#    - Original assumption "any random template shape is fine" becomes unsafe. If a template were
#      horizontally or vertically symmetric, the flip rule could become invisible. The generator therefore
#      only uses coarse shapes whose expanded templates visibly change under both flips.
#
# d) Placement legality
#    - The three original bands and the dashed separators are preserved, so templates, markers, and
#      palette swatches still live in disjoint regions.
#    - Every marker occupies exactly one 2x2 box inside its own physical marker tile column. Generator
#      placement is built from disjoint 2x2 boxes, so marker patches never overlap and never cross
#      separators or tile bounds.
#    - Output block ownership remains unique: two templates are never assigned to the same decoded slot.
#
# e) Visual observability
#    - The top templates are literal enlarged versions of the same 2x2 silhouettes used by the markers, so
#      humans can compare them by eye without hidden numeric rules.
#    - The anchor color is fixed and visually unique. It gives a stable "this cell is the locator" cue,
#      while the surrounding shape cells identify the template.
#    - Because the same coarse shape appears in different physical columns, the corresponding output stamps
#      visibly flip in repeated, low-cardinality ways.
#
# f) Visual cleanliness
#    - There are still only three semantic layers: top templates, middle marker band, bottom palette band.
#      No extra legends or decoys are introduced.
#    - Marker colors are reserved and never reused by templates or palette swatches, so the middle band is
#      easy to isolate visually.
#    - The 2x2 patches sit on a coarse packing grid, which keeps the middle band tidy instead of noisy.
#
# g) New assumptions introduced by main(), and generator support
#    - main() assumes the marker anchor color appears exactly once per marker patch; generate_input() draws
#      every patch from one canonical 2x2 pattern with exactly one anchor cell.
#    - main() assumes the top template can be reduced to a unique coarse 2x2 silhouette; generate_input()
#      builds templates directly from those coarse silhouettes.
#    - main() assumes the visible marker patch is the canonical patch after undoing the column flip;
#      generate_input() applies exactly the same flip to both the patch and the stamped template.
#    - main() assumes marker patches do not overlap; generate_input() chooses them from disjoint 2x2 boxes.


_BG = 0
_SEP = 5

_MARK_ANCHOR = 1
_MARK_SHAPE = 2
_MARK_FILL = 3
_DRAW_COLORS = [4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_K_CHOICES = [3]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7
_MIN_SLOT_FRACTION = 0.55
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500

_PHYS_OPS = ["id", "hflip", "vflip"]

_CANONICAL_PATTERNS = [
    np.array([[0, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [0, 0]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
]


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _apply_op(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "id":
        return arr.copy()
    if op == "hflip":
        return np.fliplr(arr)
    if op == "vflip":
        return np.flipud(arr)
    raise ValueError(op)


def _quadrant_slices(n: int) -> list[tuple[slice, slice]]:
    mid = n // 2
    return [
        (slice(0, mid), slice(0, mid)),
        (slice(0, mid), slice(mid, n)),
        (slice(mid, n), slice(0, mid)),
        (slice(mid, n), slice(mid, n)),
    ]


def _pattern_key(pattern: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in pattern.astype(int).flatten())


def _template_signature(mask: np.ndarray) -> np.ndarray:
    n = mask.shape[0]
    sig = np.zeros((2, 2), dtype=bool)
    for idx, (rs, cs) in enumerate(_quadrant_slices(n)):
        sig[idx // 2, idx % 2] = bool(mask[rs, cs].any())
    return sig


def _build_template(pattern: np.ndarray, n: int, color: int) -> np.ndarray:
    out = np.full((n, n), _BG, dtype=int)
    for idx, (rs, cs) in enumerate(_quadrant_slices(n)):
        if pattern[idx // 2, idx % 2]:
            out[rs, cs] = int(color)
    return out


def _canonical_marker_patch(pattern: np.ndarray) -> np.ndarray:
    assert pattern.shape == (2, 2)
    assert not pattern[0, 0]
    patch = np.full((2, 2), _MARK_FILL, dtype=int)
    patch[0, 0] = int(_MARK_ANCHOR)
    patch[pattern] = int(_MARK_SHAPE)
    return patch


def _anchor_from_box(box_top: int, box_left: int, op: str) -> tuple[int, int]:
    if op == "id":
        return box_top, box_left
    if op == "hflip":
        return box_top, box_left + 1
    if op == "vflip":
        return box_top + 1, box_left
    raise ValueError(op)


def _box_from_anchor(anchor_r: int, anchor_c: int, op: str) -> tuple[int, int]:
    if op == "id":
        return anchor_r, anchor_c
    if op == "hflip":
        return anchor_r, anchor_c - 1
    if op == "vflip":
        return anchor_r - 1, anchor_c
    raise ValueError(op)


def _marker_boxes(n: int) -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(0, n - 1, 2) for c0 in range(0, n - 1, 2)]


def _extract_marker_signature(tile: np.ndarray, anchor_r: int, anchor_c: int, op: str) -> np.ndarray:
    box_r0, box_c0 = _box_from_anchor(anchor_r, anchor_c, op)
    assert 0 <= box_r0 <= tile.shape[0] - 2 and 0 <= box_c0 <= tile.shape[1] - 2
    patch = tile[box_r0 : box_r0 + 2, box_c0 : box_c0 + 2]
    canon = _apply_op(patch, op)
    assert int(canon[0, 0]) == _MARK_ANCHOR
    sig = canon == int(_MARK_SHAPE)
    assert not sig[0, 0]
    other_vals = {int(v) for v in canon.flat} - {_MARK_ANCHOR, _MARK_SHAPE, _MARK_FILL}
    assert not other_vals
    return sig


def _render_output(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
    ops_by_template: list[str],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        mask = templates[j] != int(_BG)
        stamped_mask = _apply_op(mask, ops_by_template[j])
        tile = np.where(stamped_mask, pcols[j], _BG)
        paste(out, tile, rr * n, cc * n)
    return out


def _behavior_metrics(
    n: int,
    placements: list[tuple[int, int, int]],
    patterns: list[np.ndarray],
    ops_by_template: list[str],
) -> dict[str, int]:
    total = len(placements)
    non_id = sum(1 for j, _rr, _cc in placements if ops_by_template[j] != "id")
    flip_columns_used = len({ops_by_template[j] for j, _rr, _cc in placements})
    raw_shape_changes = sum(1 for j, _rr, _cc in placements if ops_by_template[j] != "id")
    return {
        "total_markers": total,
        "non_id_markers": non_id,
        "flip_columns_used": flip_columns_used,
        "distinct_patterns_used": len({_pattern_key(patterns[j]) for j, _rr, _cc in placements}),
        "slot_capacity": len(_marker_boxes(n)) * len(_PHYS_OPS),
        "raw_shape_changes": raw_shape_changes,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend whose middle-band markers are 2x2 shape patches instead of single pixels.

    Input structure:
    - The grid is split into K tile columns of width N, separated by 1-column dashed separators.
    - Rows 0..N-1 hold K one-color templates, each template being a large NxN version of a coarse 2x2 shape.
    - Rows N+1..2N hold 2x2 marker patches built from three fixed marker colors: one anchor cell and a tiny
      shape occupying some of the other three cells.
    - Rows 2N+2..end hold palette swatches; each swatch still supplies the repaint color for the template in
      the same top-band column.

    Rule:
    - Each marker patch has exactly one anchor-colored cell; that anchor's local coordinate (lr, lc) inside
      its physical marker tile column selects the output block (lr, lc).
    - Undo the physical marker column's flip mode (`id`, `hflip`, `vflip`, repeating by column) on the 2x2
      patch, then compare its tiny shape to the coarse shape of the top templates to determine which template
      it refers to.
    - Stamp that matched template into the chosen output block, but apply the same physical-column flip to
      the full template before recoloring it with the matched palette color.

    Key insight: the middle band no longer uses template colors as direct labels. The unique anchor cell gives
    the block position, while the surrounding 2x2 shape must be unflipped according to its physical column and
    then matched to the enlarged top-band template shape.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if (g[:, c] == sep).sum() >= H // 2]
    sep_rows = [r for r in range(H) if (g[r, :] == sep).sum() >= W // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    mid_r0 = n + 1
    pal_r0 = sep_rows[-1] + 1

    templates: list[np.ndarray] = []
    sig_to_template: dict[tuple[int, int, int, int], int] = {}
    for j, st in enumerate(starts):
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        mask = sub == col
        sig = _template_signature(mask)
        key = _pattern_key(sig)
        assert key not in sig_to_template
        sig_to_template[key] = j
        templates.append(mask)

    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:H, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for phys_idx, st in enumerate(starts):
        op = _PHYS_OPS[phys_idx % len(_PHYS_OPS)]
        tile = g[mid_r0 : mid_r0 + n, st : st + n]
        anchors = np.argwhere(tile == int(_MARK_ANCHOR))
        for anchor_r, anchor_c in anchors:
            lr, lc = int(anchor_r), int(anchor_c)
            sig = _extract_marker_signature(tile, lr, lc, op)
            key = _pattern_key(sig)
            assert key in sig_to_template
            j = sig_to_template[key]
            assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
            slot_owner[(lr, lc)] = j
            stamped_mask = _apply_op(templates[j], op)
            tile_out = np.where(stamped_mask, pcols[j], bg)
            out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile_out

    return out


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = int(_SEP)
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = int(_SEP)


def _choose_placements(n: int, ops_by_template: list[str]) -> list[tuple[int, int, int]] | None:
    k = len(ops_by_template)
    boxes = _marker_boxes(n)
    candidates: list[list[tuple[int, int]]] = []
    for op in ops_by_template:
        cur = [_anchor_from_box(r0, c0, op) for r0, c0 in boxes]
        random.shuffle(cur)
        candidates.append(cur)

    placements: list[tuple[int, int, int]] = []
    used_slots: set[tuple[int, int]] = set()
    counts = [0] * k

    template_order = list(range(k))
    random.shuffle(template_order)
    for j in template_order:
        opts = [slot for slot in candidates[j] if slot not in used_slots]
        if not opts:
            return None
        slot = opts[0]
        placements.append((j, slot[0], slot[1]))
        used_slots.add(slot)
        counts[j] += 1

    max_total = len({slot for per_t in candidates for slot in per_t})
    min_total = max(k, int(np.ceil(_MIN_SLOT_FRACTION * max_total)))
    target = random.randint(min_total, max_total)

    while len(placements) < target:
        min_count = min(counts)
        pool = [
            (j, slot)
            for j in range(k)
            if counts[j] == min_count
            for slot in candidates[j]
            if slot not in used_slots
        ]
        if not pool:
            pool = [
                (j, slot)
                for j in range(k)
                for slot in candidates[j]
                if slot not in used_slots
            ]
        if not pool:
            break
        j, slot = random.choice(pool)
        placements.append((j, slot[0], slot[1]))
        used_slots.add(slot)
        counts[j] += 1

    random.shuffle(placements)
    return placements


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.choice(_N_CHOICES)
    k = random.choice(_K_CHOICES)

    template_patterns = random.sample(_CANONICAL_PATTERNS, k)
    tcols = random.sample(_DRAW_COLORS, k)
    pcols = random.sample(_DRAW_COLORS, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = [_build_template(pattern, n, color) for pattern, color in zip(template_patterns, tcols)]

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    ops_by_template = [_PHYS_OPS[tile_assign[j] % len(_PHYS_OPS)] for j in range(k)]

    placements = _choose_placements(n, ops_by_template)
    if placements is None:
        return None

    expected = _render_output(
        n=n,
        templates=templates,
        pcols=pcols,
        placements=placements,
        ops_by_template=ops_by_template,
    )

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, template in enumerate(templates):
        paste(grid, template, 0, j * (n + 1))

    mid_r0 = n + 1
    for j, rr, cc in placements:
        phys = tile_assign[j]
        op = ops_by_template[j]
        local_top, local_left = _box_from_anchor(rr, cc, op)
        patch = _apply_op(_canonical_marker_patch(template_patterns[j]), op)
        paste(grid, patch, mid_r0 + local_top, phys * (n + 1) + local_left)

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = int(col)

    metrics = _behavior_metrics(
        n=n,
        placements=placements,
        patterns=template_patterns,
        ops_by_template=ops_by_template,
    )
    if metrics["non_id_markers"] == 0 or metrics["flip_columns_used"] < 3:
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid, metrics
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
