# Combo variant of 65b59efc (index 18)
# combo_spec: D8/2+D12/0
# Name: 双源几何变换

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
#    - Original 65b59efc: a marker color selects one top-band template; the marker's local slot (lr, lc)
#      selects one output block; the bottom swatch only contributes the repaint color. The stamped tile is
#      the raw template mask recolored with that palette color, with no geometry change.
#    - Fused rule here: the bottom swatch still supplies the repaint color, but its pixel count also
#      selects a base transform:
#        1 pixel -> identity
#        2 pixels -> horizontal flip
#        3 pixels -> 90-degree clockwise rotation
#      Then the output block row adds an extra clockwise rotation: row 0 -> +0, row 1 -> +90, row 2
#      -> +180, row 3 -> +270, and higher rows continue modulo 4.
#    - Small example with N=4:
#        template =
#          [[1,1,0,0],
#           [0,1,0,0],
#           [0,1,1,0],
#           [0,0,1,0]]
#        swatch = three palette cells of color 8  => base op = rot90
#        marker local slot = (2, 1)               => block row 2 adds rot180
#      Original output tile at block (2,1) would be the unrotated mask recolored to 8. The fused output
#      first rotates the mask 90 clockwise from the swatch, then rotates that result another 180 because
#      the block row is 2. The difference is visible in both input bands and the stamped output.
#
# b) Complexity calibration
#    - The swatch-size channel uses exactly three cases, repeated within each sample. That is enough to
#      make the new symbol rule discoverable without creating too many discrete cases.
#    - The row-driven rotation uses only the output block row, not both row and column, so one axis stays
#      simple and repeated. With N in {4, 5}, rows 0..3 visibly realize the four quarter-turn states.
#    - Templates are required to have eight distinct D4/D8 orientations, so neither the swatch transform
#      nor the row rotation collapses into a visually identical stamp.
#
# c) Implicit-assumption audit against the original main()
#    - Original assumption "palette band only stores one color per template" is no longer sufficient.
#      main() now also counts how many swatch pixels are present in that template column.
#    - Original assumption "render tile = raw template mask recolored" is false. main() must compose two
#      geometric sources: swatch-coded base op and row-coded extra rotation.
#    - Original assumption "template symmetry does not matter" becomes unsafe. If a template were
#      horizontally symmetric or quarter-turn symmetric, the new rules could become invisible. The
#      generator therefore rejects any template whose D8 orbit is not fully distinct.
#    - Original assumption "any output row distribution is fine" is weaker now. To make the row rule
#      observable, generate_input() guarantees markers on rows 0,1,2,3 whenever those rows exist.
#
# d) Placement legality
#    - The three original bands and dashed separators are kept, so templates, markers, and palette swatch
#      areas remain disjoint and easy to parse.
#    - Marker slots are still unique across templates, so at most one stamp owns each output block.
#    - Swatches are horizontal runs of length 1..3 fully inside their own N-wide tile columns, and bottom
#      padding stays background-only, so swatch counting is unambiguous.
#
# e) Visual observability
#    - Humans can see that some palette swatches are 1, 2, or 3 cells wide even when the color itself is
#      unrelated to the transform. That gives a visible non-color cue for the first geometry source.
#    - The same template appears in multiple output rows, so the row-dependent extra rotation can be seen
#      holding the swatch size fixed while only the output row changes.
#    - Because templates are deliberately asymmetric, the composed transforms create clearly different
#      silhouettes rather than near-identical rotated blobs.
#
# f) Visual cleanliness
#    - No extra bands or overlay glyphs are added: the scene remains the base task's clean "template /
#      markers / palette" legend.
#    - Palette colors are chosen disjoint from template colors inside each sample, so the top-band lookup
#      cue and bottom-band repaint cue remain visually separated.
#    - Swatches are centered short horizontal bars, which keeps the extra encoding local and legible.
#
# g) New assumptions introduced by main(), and generator support
#    - main() assumes each palette area has exactly one non-background color and that its count is 1, 2,
#      or 3; generate_input() draws exactly one contiguous horizontal swatch per template column.
#    - main() assumes every template has a unique primary color for marker lookup; generate_input() keeps
#      the original one-color-per-template top band.
#    - main() assumes row rotation matters; generate_input() seeds placements so every active sample shows
#      multiple output rows, including rows 0..3 when available.
#    - main() assumes swatch transforms matter; generate_input() uses all three swatch sizes in each
#      sample and rejects templates with any D8 symmetry that would hide those transforms.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_K_CHOICES = [3, 4]
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.34, 0.72)
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.70
_MAX_RETRIES = 600

_MODE_TO_SIZE = {"id": 1, "hflip": 2, "rot90": 3}
_SIZE_TO_MODE = {1: "id", 2: "hflip", 3: "rot90"}


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _rot_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    k = quarter_turns % 4
    if k == 0:
        return arr.copy()
    return np.rot90(arr, 4 - k)


def _apply_base_transform(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "id":
        return mask.copy()
    if mode == "hflip":
        return np.fliplr(mask)
    if mode == "rot90":
        return _rot_cw(mask, 1)
    raise ValueError(mode)


def _apply_combo_transform(mask: np.ndarray, mode: str, block_row: int) -> np.ndarray:
    base = _apply_base_transform(mask, mode)
    return _rot_cw(base, block_row % 4)


def _d8_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [
        mask.copy(),
        _rot_cw(mask, 1),
        _rot_cw(mask, 2),
        _rot_cw(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T.copy(),
        np.flipud(np.fliplr(mask.T)),
    ]


def _mask_key(mask: np.ndarray) -> bytes:
    return np.ascontiguousarray(mask.astype(np.uint8)).tobytes()


def _d8_canonical(mask: np.ndarray) -> bytes:
    return min(_mask_key(v) for v in _d8_variants(mask))


def _is_d8_distinct(mask: np.ndarray) -> bool:
    return len({_mask_key(v) for v in _d8_variants(mask)}) == 8


def _sample_template_mask(n: int) -> np.ndarray | None:
    for _ in range(240):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == 1
        area = int(mask.sum())
        if area < n + 1 or area > n * n - 4:
            continue
        if not _is_d8_distinct(mask):
            continue
        return mask
    return None


def _sample_templates(n: int, colors: list[int]) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen_orbits: set[bytes] = set()
    for _color in colors:
        mask = None
        for _ in range(200):
            candidate = _sample_template_mask(n)
            if candidate is None:
                continue
            key = _d8_canonical(candidate)
            if key in seen_orbits:
                continue
            mask = candidate
            seen_orbits.add(key)
            break
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _render_output(
    n: int,
    bg: int,
    templates: list[np.ndarray],
    pcols: list[int],
    modes: list[str],
    placements: list[tuple[int, int, int]],
    *,
    include_swatch_transform: bool = True,
    include_row_rotation: bool = True,
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        mode = modes[j] if include_swatch_transform else "id"
        block_row = rr if include_row_rotation else 0
        tile_mask = _apply_combo_transform(templates[j], mode, block_row)
        tile = np.where(tile_mask, pcols[j], bg)
        paste(out, tile, rr * n, cc * n)
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


def _seeded_placements(n: int, k: int) -> list[tuple[int, int, int]]:
    active_rows = list(range(min(4, n)))
    available = {rr: list(range(n)) for rr in range(n)}
    seeds: list[tuple[int, int, int]] = []

    for j in range(k):
        row_pair = [active_rows[j % len(active_rows)], active_rows[(j + 1) % len(active_rows)]]
        for rr in row_pair:
            if not available[rr]:
                return []
            cc = random.choice(available[rr])
            available[rr].remove(cc)
            seeds.append((j, rr, cc))

    remaining_slots = [(rr, cc) for rr in range(n) for cc in range(n) if cc in available[rr]]
    random.shuffle(remaining_slots)
    min_marks = max(len(seeds), int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)

    placements = seeds[:]
    extra_needed = mark_count - len(seeds)
    if extra_needed > 0:
        for rr, cc in remaining_slots[:extra_needed]:
            placements.append((random.randrange(k), rr, cc))

    random.shuffle(placements)
    return placements


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_N_CHOICES)
    k = random.choice(_K_CHOICES)
    color_pool = random.sample(_AVAILABLE_COLORS, 2 * k)
    tcols = color_pool[:k]
    pcols = color_pool[k:]

    templates = _sample_templates(n, tcols)
    if templates is None:
        return None

    base_modes = ["id", "hflip", "rot90"]
    random.shuffle(base_modes)
    modes = base_modes[:]
    while len(modes) < k:
        modes.append(random.choice(base_modes))
    random.shuffle(modes)

    placements = _seeded_placements(n, k)
    if not placements:
        return None
    used_rows = {rr for _, rr, _ in placements}
    if not set(range(min(4, n))).issubset(used_rows):
        return None
    if {modes[j] for j, _, _ in placements} != set(base_modes):
        return None

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    expected = _render_output(n, _BG, templates, pcols, modes, placements)
    baseline = _render_output(
        n,
        _BG,
        templates,
        pcols,
        modes,
        placements,
        include_swatch_transform=False,
        include_row_rotation=False,
    )
    swatch_only = _render_output(
        n,
        _BG,
        templates,
        pcols,
        modes,
        placements,
        include_swatch_transform=True,
        include_row_rotation=False,
    )
    row_only = _render_output(
        n,
        _BG,
        templates,
        pcols,
        modes,
        placements,
        include_swatch_transform=False,
        include_row_rotation=True,
    )
    if np.array_equal(expected, baseline):
        return None
    if np.array_equal(expected, swatch_only):
        return None
    if np.array_equal(expected, row_only):
        return None

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
        size = _MODE_TO_SIZE[modes[j]]
        c0 = j * (n + 1) + (n - size) // 2
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where geometry comes from both palette swatches and output-row position.

    Input structure:
    - The grid is split into K tile columns of width N, separated by 1-column dashed separators.
    - Rows 0..N-1 hold K one-color template masks, one per tile column.
    - Rows N+1..2N hold marker pixels; a marker's color selects the template and its local slot selects
      the output block.
    - Rows 2N+2..end form the palette band; each tile column contains one horizontal swatch whose color
      is the repaint color and whose pixel count is 1, 2, or 3.
    - Dashed separator rows appear at row N and row 2N+1.

    Rule:
    - Read each top template and match marker colors to those template colors exactly as in the base task.
    - For template j, decode its palette swatch: 1 cell means no base transform, 2 cells mean horizontal
      flip, and 3 cells mean a 90-degree clockwise base rotation.
    - For a marker at local row lr and local column lc, stamp into output block (lr, lc), but first apply
      the swatch-coded transform and then an additional clockwise rotation of 90 * (lr mod 4) degrees.

    Key insight: The bottom swatch now carries two signals at once: its color still says how to repaint
    the template, while its size controls a base orientation. The marker row then adds a second geometric
    cue, so the final stamp is the composition of both sources.
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
    k = len(starts)

    template_colors: list[int] = []
    templates: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        template_colors.append(col)
        templates.append(sub == col)
    assert len(set(template_colors)) == k
    color_to_template = {col: j for j, col in enumerate(template_colors)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    modes: list[str] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        size = int(np.count_nonzero(sub == col))
        assert size in _SIZE_TO_MODE
        pcols.append(col)
        modes.append(_SIZE_TO_MODE[size])

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
                j = color_to_template[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                tile_mask = _apply_combo_transform(templates[j], modes[j], lr)
                tile = np.where(tile_mask, pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
