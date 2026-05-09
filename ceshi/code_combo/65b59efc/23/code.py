# Combo variant of 65b59efc (index 23)
# combo_spec: D3/3+D10/1
# Name: 无分隔嵌入标记
from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


# Design analysis:
# a) Concrete behavior trace
#    - Original 65b59efc:
#      1. The top band contains K templates, one per tile column.
#      2. The middle band contains sparse marker pixels. A marker's color directly picks the template.
#      3. The marker's local position (lr, lc) inside its tile column selects output block (lr, lc).
#      4. The bottom band gives one palette color per template; the chosen template is recolored and stamped.
#    - Fused variant:
#      1. All explicit separator rows and columns disappear. The legend width is now exactly K*N, split only
#         by latent N-cell tile blocks.
#      2. The middle band is completely blank. Marker information moves into the top template band.
#      3. Each top tile now contains two visible non-background colors:
#         - one template color, forming the actual template mask shown in that tile
#         - one embedded marker color, drawn only on background holes of that shown template
#      4. Embedded marker colors are palette colors, not template colors. A marker's color must be matched
#         against the bottom palette band to find which template index it refers to.
#      5. The physical tile that carries the markers is not the same as the referenced template tile; the
#         generator uses a full derangement. So "same tile means same template" is always wrong.
#    - Small example, N=3 and K=3:
#      * top displayed templates use colors [2, 4, 7]
#      * bottom palette colors are [1, 6, 8]
#      * top tile 0 shows template color 2, but its embedded marker pixels are color 6 at local slots
#        (0,2) and (2,1)
#      * because color 6 matches bottom palette swatch 1, those two marker positions stamp template 1,
#        recolored to 6, into output blocks (0,2) and (2,1)
#      Compared with the base task, the middle band contributes nothing, and compared with a simpler
#      no-separator variant, the visible second color inside each template tile is now semantically active.
#
# b) Complexity calibration
#    - The new cue is repeated K times per sample: every top tile contains a marker color and every palette
#      color is reused exactly once in the top band. That is enough support for a human to infer the
#      color-matching rule from only a few examples.
#    - The marker colors are drawn only from the K palette colors, while template colors are chosen from a
#      disjoint set. This keeps the two roles visually separable without relying on hidden numeric ordering.
#    - The tile-to-template assignment is a derangement rather than an arbitrary many-to-one mapping. That
#      makes the fusion visible and prevents a simpler "use the local template" shortcut from working.
#
# c) Original assumptions broken and replaced
#    - `N` can no longer be read from separator columns. The solver must infer a valid (N, K) layout from the
#      full grid shape and the blank middle band.
#    - `marker color == template color` is false. Marker colors now belong to the bottom palette set.
#    - "all marker pixels live in the middle band" is false. The middle band is required to be empty.
#    - "a top tile contains only the shown template" is false. The solver must split top-tile colors into a
#      dominant template color and an embedded palette-color marker set.
#    - "a marker tile stamps the template shown in that same tile" is false. The marker color determines the
#      actual template via the palette band, and the generator guarantees a non-identity permutation.
#
# d) Placement legality
#    - Marker pixels are only written onto background holes of the shown top tile, so the displayed template
#      mask itself is not destroyed.
#    - Different templates still own disjoint output slots. The generator assigns each logical slot to at most
#      one template and keeps all marker placements in-bounds.
#    - The middle band is left completely background and the palette band occupies only the bottom rows, so the
#      three functional zones never overlap.
#
# e) Visual observability
#    - A human sees the entire middle strip as empty, which immediately signals that the old middle-band rule
#      cannot apply.
#    - Each top tile visibly contains a sparse second color that also appears in exactly one bottom swatch.
#      Across examples, that repeated pairing exposes the embedded-marker lookup channel.
#    - Because the mapping is a derangement in every sample, the output cannot be explained by "stamp the
#      template displayed above the marker positions".
#
# f) Visual cleanliness
#    - The scene stays as three horizontal bands; nothing is overlaid onto the bottom palette band and no
#      auxiliary glyphs are introduced.
#    - Marker pixels live only in template holes, so the main silhouette of each shown template remains clear.
#    - The palette uses at most K colors and the template set uses a disjoint K-color set, which limits clutter.
#
# g) New solver assumptions and generator support
#    - main() assumes the full grid width equals K*N with no trailing column padding; generate_input() writes no
#      right padding so layout inference remains exact without separators.
#    - main() assumes each top tile contains exactly one template color outside the palette set and exactly one
#      embedded marker color inside the palette set; the generator enforces both conditions and gives every tile
#      at least one embedded marker.
#    - main() assumes each palette color appears once in the top band as a marker color; the generator uses a
#      derangement so the marker-color multiset equals the palette set exactly.
#    - main() assumes the blank middle band has height N; the generator always allocates that full empty strip.


_BG = 0
_AVAILABLE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_LAYOUT_CHOICES = [(3, 3), (4, 4), (5, 3), (5, 4)]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_DENSITY_RANGE = (0.30, 0.56)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.55
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 600


def _sample_template_mask(n: int) -> np.ndarray:
    for _ in range(160):
        try:
            sprite = random_sprite(
                n,
                n,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[1],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=_BG,
            )
        except AssertionError:
            continue
        mask = sprite == 1
        area = int(mask.sum())
        if area < 2 or area > n * n - 2:
            continue
        return mask
    raise RuntimeError("failed to sample template mask")


def _sample_templates(n: int, k: int) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    seen: set[bytes] = set()
    for _ in range(k):
        mask = None
        for _ in range(240):
            candidate = _sample_template_mask(n)
            key = np.ascontiguousarray(candidate.astype(np.uint8)).tobytes()
            if key in seen:
                continue
            mask = candidate
            seen.add(key)
            break
        if mask is None:
            return None
        templates.append(mask)
    return templates


def _render_expected(
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


def _render_same_column_baseline(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int]],
    carrier_for_template: list[int],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        i = carrier_for_template[j]
        tile = np.where(templates[i], palette_colors[i], _BG)
        paste(out, tile, rr * n, cc * n)
    return out


def _derangement(k: int) -> list[int]:
    order = list(range(k))
    while True:
        random.shuffle(order)
        if all(i != order[i] for i in range(k)):
            return order


def _seed_distinct_slots(allowed: list[set[tuple[int, int]]]) -> dict[int, tuple[int, int]] | None:
    order = sorted(range(len(allowed)), key=lambda j: len(allowed[j]))
    used: set[tuple[int, int]] = set()
    assigned: dict[int, tuple[int, int]] = {}

    def dfs(idx: int) -> bool:
        if idx == len(order):
            return True
        j = order[idx]
        choices = list(allowed[j])
        random.shuffle(choices)
        for slot in choices:
            if slot in used:
                continue
            used.add(slot)
            assigned[j] = slot
            if dfs(idx + 1):
                return True
            used.remove(slot)
            del assigned[j]
        return False

    if dfs(0):
        return assigned
    return None


def _choose_placements(
    n: int,
    allowed: list[set[tuple[int, int]]],
) -> list[tuple[int, int, int]] | None:
    if any(len(opts) == 0 for opts in allowed):
        return None

    seeded = _seed_distinct_slots(allowed)
    if seeded is None:
        return None

    union_slots = set().union(*allowed)
    max_marks = min(n * n, len(union_slots))
    min_marks = max(len(allowed), int(math.ceil(_MIN_SLOT_COVERAGE * n * n)))
    if max_marks < min_marks:
        return None

    total_marks = random.randint(min_marks, max_marks)
    slot_owner = {slot: j for j, slot in seeded.items()}
    remaining_slots = list(union_slots - set(slot_owner))
    random.shuffle(remaining_slots)

    for slot in remaining_slots[: total_marks - len(slot_owner)]:
        owners = [j for j, opts in enumerate(allowed) if slot in opts]
        slot_owner[slot] = random.choice(owners)

    placements = [(j, rr, cc) for (rr, cc), j in slot_owner.items()]
    random.shuffle(placements)
    return placements


def _infer_layout(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = g.shape
    candidates: list[tuple[int, int, int]] = []

    for n, k in _LAYOUT_CHOICES:
        if w != k * n:
            continue
        pal_h = h - 2 * n
        if pal_h not in _PAL_H_CHOICES:
            continue
        if np.any(g[n : 2 * n, :] != bg):
            continue

        palette_colors: list[int] = []
        ok = True
        for j in range(k):
            st = j * n
            sub = g[2 * n :, st : st + n]
            vals = {int(v) for v in sub.flat if int(v) != bg}
            if len(vals) != 1:
                ok = False
                break
            palette_colors.append(vals.pop())
        if not ok or len(set(palette_colors)) != k:
            continue
        palette_set = set(palette_colors)

        marker_colors: list[int] = []
        template_colors: list[int] = []
        for j in range(k):
            st = j * n
            sub = g[:n, st : st + n]
            vals = {int(v) for v in sub.flat if int(v) != bg}
            tpl_vals = vals - palette_set
            mark_vals = vals & palette_set
            if len(tpl_vals) != 1 or len(mark_vals) != 1:
                ok = False
                break
            template_colors.append(next(iter(tpl_vals)))
            marker_colors.append(next(iter(mark_vals)))
        if not ok:
            continue
        if len(set(template_colors)) != k:
            continue
        if set(marker_colors) != palette_set:
            continue

        candidates.append((n, k, pal_h))

    assert len(candidates) == 1, "layout must be uniquely inferable"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a contiguous stamp legend where the markers are hidden inside the top template strip.

    Input structure:
    - The grid has K adjacent tile blocks of width N with no explicit separator rows or columns.
    - Top band (rows 0..N-1): K displayed template masks, one per tile block, but each tile also contains
      sparse embedded marker pixels in a second color.
    - Middle band (rows N..2N-1): completely blank background; it no longer carries any marker data.
    - Bottom band (rows 2N..): one palette swatch per tile block. Marker colors are drawn from this palette set.

    Rule:
    - In each top tile, split the two active colors into:
      * the template color that forms the shown mask
      * the embedded marker color that matches one bottom palette swatch
    - Every embedded marker at local coordinate (lr, lc) stamps the template whose palette color matches
      that marker color, recolored with that same palette color, into output block (lr, lc).

    Key insight: The middle band is a red herring because it is entirely empty. The real control pixels are
    the sparse palette-colored dots embedded inside the top template area, and they must be separated from the
    displayed template before the palette-color match can reveal which template is actually being stamped.
    """
    g = np.array(grid, dtype=int)
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])

    n, k, pal_h = _infer_layout(g, bg)
    assert pal_h >= 2

    palette_colors: list[int] = []
    for j in range(k):
        st = j * n
        sub = g[2 * n :, st : st + n]
        vals = {int(v) for v in sub.flat if int(v) != bg}
        assert len(vals) == 1
        palette_colors.append(vals.pop())

    palette_set = set(palette_colors)
    palette_to_template = {col: j for j, col in enumerate(palette_colors)}

    templates: list[np.ndarray] = []
    marker_sets: list[list[tuple[int, int]]] = []

    for j in range(k):
        st = j * n
        sub = g[:n, st : st + n]
        vals = {int(v) for v in sub.flat if int(v) != bg}
        template_vals = vals - palette_set
        marker_vals = vals & palette_set
        assert len(template_vals) == 1 and len(marker_vals) == 1
        tcol = template_vals.pop()
        mcol = marker_vals.pop()
        templates.append(sub == tcol)
        marker_sets.append([(int(rr), int(cc)) for rr, cc in np.argwhere(sub == mcol)])

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for physical_tile, coords in enumerate(marker_sets):
        marker_color = int(g[coords[0][0], physical_tile * n + coords[0][1]]) if coords else None
        assert marker_color is not None
        j = palette_to_template[marker_color]
        tile = np.where(templates[j], palette_colors[j], bg)
        for rr, cc in coords:
            assert (rr, cc) not in slot_owner or slot_owner[(rr, cc)] == j
            slot_owner[(rr, cc)] = j
            out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n] = tile

    return out


def _gen_attempt() -> np.ndarray | None:
    n, k = random.choice(_LAYOUT_CHOICES)
    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    h = 2 * n + pal_h
    w = k * n

    colors = random.sample(_AVAILABLE_COLORS, 2 * k)
    template_colors = colors[:k]
    palette_colors = colors[k:]

    templates = _sample_templates(n, k)
    if templates is None:
        return None

    marker_target_by_tile = _derangement(k)
    carrier_for_template = [0] * k
    for tile_idx, template_idx in enumerate(marker_target_by_tile):
        carrier_for_template[template_idx] = tile_idx

    hole_sets = []
    for mask in templates:
        holes = {(int(r), int(c)) for r in range(n) for c in range(n) if not mask[r, c]}
        hole_sets.append(holes)

    allowed = [hole_sets[carrier_for_template[j]] for j in range(k)]
    placements = _choose_placements(n, allowed)
    if placements is None:
        return None

    expected = _render_expected(n, templates, palette_colors, placements)
    baseline = _render_same_column_baseline(n, templates, palette_colors, placements, carrier_for_template)
    if np.array_equal(expected, baseline):
        return None

    grid = np.full((h, w), _BG, dtype=int)

    for j, mask in enumerate(templates):
        st = j * n
        patch = np.where(mask, template_colors[j], _BG)
        paste(grid, patch, 0, st)

    for j, rr, cc in placements:
        carrier = carrier_for_template[j]
        st = carrier * n
        assert grid[rr, st + cc] == _BG
        grid[rr, st + cc] = palette_colors[j]

    swatch_r = h - 1
    for j, col in enumerate(palette_colors):
        st = j * n
        size = 2 if n >= 4 and random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = st + max(0, (n - size) // 2)
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
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
