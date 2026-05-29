# Combo variant of 65b59efc (index 2)
# combo_spec: D1/1+D10/2
# Name: 双色边框匹配
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
# a) Concrete behavior:
#    - Original 65b59efc: marker color directly names the template, and the marker's local
#      (lr, lc) picks the output block.
#    - This combo removes template information from marker pixels themselves. Every marker uses
#      the horizontal-divider color, so color-reading on the marker now points to the wrong clue.
#      The actual template key is the vertical border color of the tile column that contains the
#      marker, and that color must be matched against the top legend rather than the same physical
#      column. A small 3x3 example:
#        top templates left->right use colors [2, 4, 8], with top border keys [6, 1, 3];
#        marker columns left->right use border keys [1, 3, 6]; both horizontal separator rows and
#        all marker pixels use color 5.
#        A marker at local (2,0) inside the third marker column does NOT mean template color 5 and
#        does NOT mean "use the third template"; its column border is 6, so it selects the template
#        whose top-band border key is 6 (here the first template), then stamps it into output block
#        (2,0). The key difference is "match border colors across bands, ignore marker color".
# b) Complexity calibration:
#    - Vertical border keys stay 1-to-1 with templates and repeat in clean banded strips, so the
#      clue is strong and visually recoverable without adding extra object types.
#    - Horizontal dividers all share one color and markers reuse that same color. This gives a
#      clear false lead without exploding the number of color roles.
# c) Broken assumptions from the original main():
#    - `marker color == template color` is false.
#    - "one separator color explains every divider" is false: horizontal rows and vertical columns
#      must be separated by axis.
#    - "template index == marker column index" is also false. The marker band uses a deranged
#      permutation of the top-band border keys, so solving by physical column position fails.
# d) Placement legality:
#    - Vertical border columns are reserved and never overlap templates, markers, or palette.
#    - Marker slots remain unique across templates, so each output block gets at most one stamp.
#    - Trailing padding is background-only, so layout inference is stable.
# e) Visual observability:
#    - Across examples, the same marker color appears under multiple different outputs, which makes
#      "marker color chooses template" visibly fail.
#    - The top band and marker band share the same small set of border-key colors, but in different
#      orders, so the human solver can see a repeatable color-matching relation instead of guessing
#      from absolute column position.
# f) Cleanliness:
#    - The scene keeps the original three-band legend structure; the only added visual load is one
#      border-key strip per tile column inside each legend band, which is easy to isolate.
# g) Non-degeneracy target:
#    - Inputs are constructed so the original rule cannot succeed: marker color equals the global
#      horizontal-divider color and is excluded from template colors.
#    - Inputs are also constructed so the same-column shortcut cannot succeed: the marker-band key
#      order is always a derangement of the top-band key order.


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

            top_keys = []
            marker_keys = []
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
    Decode a stamped legend where marker color is a decoy and the usable key lives on colored borders.

    Input structure:
    - The grid is split into K tile columns of width N, each followed by a 1-cell vertical border stripe.
    - Top band (rows 0..N-1): one template mask per tile column.
    - Middle band (rows N+1..2N): marker pixels whose local coordinates choose output blocks.
    - Bottom band (rows 2N+2..): one palette swatch per tile column.
    - The two separator rows use one horizontal-divider color. Vertical border stripes use a second
      family of colors: the top legend has one permutation of key colors and the marker band has a
      different permutation of the same keys.

    Rule:
    - A marker at local position (lr, lc) stamps into output block (lr, lc), as in the base task.
    - The marker's own color is not the template key. Instead, the vertical border color of that marker's
      tile column must be matched to the top-band border key color to select which template to use.
    - That template is recolored by the palette swatch from the matching tile column before stamping.

    Key insight: First separate horizontal-divider color from vertical-divider colors; only the vertical
    border colors carry template identity, while the shared marker/horizontal color is a distractor and
    the marker-band border order is intentionally different from the top-band order.
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
                assert v == hsep, "markers must use the horizontal-divider color"
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
