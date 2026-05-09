# Combo variant of 65b59efc (index 20)
# combo_spec: D1/1+D17/2
# Name: 多色分隔边框匹配
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


# Design analysis:
# a) Concrete behavior trace:
#    - Original 65b59efc:
#      1. Read K single-color templates from the top band.
#      2. Read middle-band markers; marker color directly selects the template.
#      3. Marker local coordinates (lr, lc) choose the output N x N block.
#      4. Recolor the chosen template with its aligned palette swatch and stamp it.
#    - This fused variant changes two places together.
#      1. Horizontal and vertical separators no longer share one color role.
#         Horizontal separator rows are one repeated color `hsep`.
#         Vertical separator stripes are per-column keys.
#      2. Marker pixels no longer encode the template by their own color.
#         Every marker is colored `hsep`, so reading the marker color is a dead end.
#         The usable key is the vertical separator color of the tile column that contains
#         the marker; that key must be matched against the top-band vertical separator keys.
#    - Small example, N=3, K=3:
#      * top templates use colors [2, 4, 8]
#      * top vertical keys are [6, 1, 3]
#      * marker-band vertical keys are [1, 3, 6]
#      * horizontal separators and all marker pixels use color 5
#      A marker at local (2,0) in the third marker column does not mean template 5 and does not
#      mean "third template". Its column key is 6, which matches the first top key, so we stamp
#      template 0 into output block (2,0).
# b) Complexity calibration:
#    - The color roles stay small and repeated: one horizontal separator color, K vertical key
#      colors, K template colors, K palette colors. Vertical keys repeat as clean stripes, so the
#      clue is visible in 2-4 examples instead of being hidden in isolated pixels.
#    - Marker-band key order is always a derangement of the top-band key order. This prevents the
#      task from collapsing into "same physical column means same template".
# c) Original assumptions broken and replaced:
#    - `marker color == template color` is false; markers are all `hsep`.
#    - "all separators use one color" is false; rows and columns must be separated by axis.
#    - "template index == marker column index" is false; the column-key permutation is shuffled.
#    - The solver therefore first infers which separator color is horizontal-only, then uses the
#      per-column vertical keys as the actual matching channel.
# d) Placement legality:
#    - Template/palette contents occupy only the N-wide tile interiors; separator stripes live in
#      their own reserved columns, so objects never overlap.
#    - Marker slots remain unique across templates, so each output block has at most one owner.
#    - Any trailing padding is pure background, which keeps layout parsing stable.
# e) Visual observability:
#    - Across samples, the same marker color leads to different templates, so a human can reject
#      the old rule quickly.
#    - The top band and marker band visibly reuse the same set of vertical key colors in different
#      orders, which exposes the intended color-matching relation.
# f) Visual cleanliness:
#    - The puzzle still has the original three-band stacked layout.
#    - Only one extra cue is added: the vertical key stripe. No dense overlays or micro-markers.
# g) Non-degeneracy targets:
#    - The marker color is always the horizontal separator color and is excluded from template
#      colors, so the original direct-color solver cannot be right.
#    - The marker key order is always a full derangement, so same-column reading is always wrong.
#    - Validation later measures both properties explicitly.


_BG = 0
_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 5
_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 400
_COLORS = list(range(1, 10))


def _dash_on(local_idx: int) -> bool:
    return local_idx % 2 == 0


def _sample_template(n: int, color: int) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[color],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=_BG,
    )
    return sprite == color


def _sample_templates(n: int, colors: list[int]) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    seen: set[bytes] = set()
    for color in colors:
        for _ in range(80):
            mask = _sample_template(n, color)
            sig = mask.tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            templates.append(mask)
            break
        else:
            return None
    return templates


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], palette_colors[j], _BG)
        paste(out, tile, rr * n, cc * n)
    return out


def _draw_layout(
    grid: np.ndarray,
    *,
    n: int,
    k: int,
    core_h: int,
    hsep: int,
    top_keys: list[int],
    marker_keys: list[int],
) -> None:
    for r in (n, 2 * n + 1):
        for j in range(k):
            st = j * (n + 1)
            for lc in range(n):
                if _dash_on(lc):
                    grid[r, st + lc] = hsep
    for j, col in enumerate(top_keys):
        c = j * (n + 1) + n
        grid[: n + 1, c] = col
        grid[2 * n + 2 : core_h, c] = col
    for j, col in enumerate(marker_keys):
        c = j * (n + 1) + n
        grid[n + 1 : 2 * n + 2, c] = col


def _derangement(k: int) -> list[int]:
    order = list(range(k))
    while True:
        random.shuffle(order)
        if all(idx != val for idx, val in enumerate(order)):
            return order


def _infer_layout(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = g.shape
    for n in range(_N_RANGE[0], _N_RANGE[1] + 1):
        if 2 * n + 4 > h:
            continue
        sep_rows = (n, 2 * n + 1)
        for k in range(_K_RANGE[0], _K_RANGE[1] + 1):
            core_w = k * (n + 1)
            if core_w > w:
                continue
            if any(np.any(g[r, core_w:] != bg) for r in sep_rows):
                continue

            horiz_vals: list[int] = []
            ok = True
            for r in sep_rows:
                for j in range(k):
                    st = j * (n + 1)
                    for lc in range(n):
                        v = int(g[r, st + lc])
                        if _dash_on(lc):
                            if v == bg:
                                ok = False
                                break
                            horiz_vals.append(v)
                        elif v != bg:
                            ok = False
                            break
                    if not ok:
                        break
                    if int(g[r, st + n]) == bg:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            counts = Counter(horiz_vals)
            hsep, cnt = counts.most_common(1)[0]
            if cnt != len(horiz_vals):
                continue

            top_keys: list[int] = []
            marker_keys: list[int] = []
            for j in range(k):
                c = j * (n + 1) + n
                top_vals = {int(v) for v in g[: n + 1, c] if int(v) != bg}
                mid_vals = {int(v) for v in g[n + 1 : 2 * n + 2, c] if int(v) != bg}
                if len(top_vals) != 1 or len(mid_vals) != 1:
                    ok = False
                    break
                top_keys.append(top_vals.pop())
                marker_keys.append(mid_vals.pop())
            if not ok:
                continue
            if len(set(top_keys)) != k or len(set(marker_keys)) != k:
                continue
            if hsep in top_keys or hsep in marker_keys:
                continue
            if set(marker_keys) != set(top_keys):
                continue

            return n, k, hsep
    raise AssertionError("unable to infer combo layout")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where the useful template key comes from vertical separator colors.

    Input structure:
    - The grid has K tile columns of width N, and each tile column is followed by a 1-cell vertical
      separator stripe.
    - Top band (rows 0..N-1): one template mask per tile column.
    - Middle band (rows N+1..2N): sparse marker pixels whose local coordinates choose output blocks.
    - Bottom band (rows 2N+2..): one palette swatch per tile column.
    - Horizontal separator rows use one repeated dashed color, while vertical separator stripes use a
      set of K key colors. The top/bottom bands and the middle band show the same K keys but not in
      the same column order.

    Rule:
    - A marker at local position (lr, lc) stamps into output block (lr, lc), exactly as in the base
      task.
    - The marker's own color is not the template identifier. Instead, read the vertical separator key
      for that marker's tile column, match that key to the top-band separator key, and use the aligned
      template and palette color from that matched top tile.
    - Recolor the matched template with its palette swatch and stamp it into the selected output block.

    Key insight: Separate the two separator roles first: the horizontal separator color is just a
    repeated divider/marker color, while the per-column vertical separator colors are the true lookup
    keys that connect marker columns back to top-band templates.
    """
    g = np.array(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    n, k, hsep = _infer_layout(g, bg)
    starts = [j * (n + 1) for j in range(k)]

    templates: list[np.ndarray] = []
    template_colors: list[int] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat if int(v) != bg}
        assert len(vals) == 1
        color = vals.pop()
        template_colors.append(color)
        templates.append(sub == color)
    assert len(set(template_colors)) == k

    top_keys: list[int] = []
    marker_keys: list[int] = []
    for st in starts:
        c = st + n
        top_vals = {int(v) for v in g[: n + 1, c] if int(v) != bg}
        mid_vals = {int(v) for v in g[n + 1 : 2 * n + 2, c] if int(v) != bg}
        assert len(top_vals) == 1 and len(mid_vals) == 1
        top_keys.append(top_vals.pop())
        marker_keys.append(mid_vals.pop())
    assert len(set(top_keys)) == k
    assert len(set(marker_keys)) == k
    assert set(marker_keys) == set(top_keys)
    key_to_template = {key: j for j, key in enumerate(top_keys)}

    pal_r0 = 2 * n + 2
    palette_colors: list[int] = []
    for st in starts:
        sub = g[pal_r0:, st : st + n]
        vals = {int(v) for v in sub.flat if int(v) not in {bg, hsep}}
        assert len(vals) == 1
        palette_colors.append(vals.pop())

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    mid_r0 = n + 1

    for phys, st in enumerate(starts):
        j = key_to_template[marker_keys[phys]]
        tile = np.where(templates[j], palette_colors[j], bg)
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg:
                    continue
                assert v == hsep, "markers must reuse the horizontal separator color"
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)

    hsep = random.choice(_COLORS)
    remaining = [c for c in _COLORS if c != hsep]
    template_colors = random.sample(remaining, k)
    remaining = [c for c in remaining if c not in template_colors]
    top_keys = random.sample(remaining, k)
    palette_colors = random.sample(template_colors, k)
    marker_order = _derangement(k)
    marker_keys = [top_keys[j] for j in marker_order]
    column_for_template = {tmpl: phys for phys, tmpl in enumerate(marker_order)}

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * (n + 1)
    h = core_h + random.randint(0, _MAX_ROW_PAD)
    w = core_w + random.randint(0, _MAX_COL_PAD)

    templates = _sample_templates(n, template_colors)
    if templates is None:
        return None

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    for j in range(k):
        if j not in assign_pool:
            assign_pool[random.randrange(len(assign_pool))] = j
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, palette_colors, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_layout(
        grid,
        n=n,
        k=k,
        core_h=core_h,
        hsep=hsep,
        top_keys=top_keys,
        marker_keys=marker_keys,
    )

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, template_colors[j], _BG), 0, j * (n + 1))

    for j, rr, cc in placements:
        phys = column_for_template[j]
        grid[n + 1 + rr, phys * (n + 1) + cc] = hsep

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(palette_colors):
        st = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = st + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid, expected
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
