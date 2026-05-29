# Combo variant of 65b59efc (index 32)
# combo_spec: D1/2+D8/2
# Name: 连接器匹配变换编码
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_ROOT = ROOT / "solvers"
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from arc_utils import paste, random_sprite


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original task: a marker's own color selected the template, and the bottom
#    swatch color only recolored that template.
#    Fused task:
#    - The top band still holds K template masks, but each template now has a
#      one-pixel connector nub on one edge. Example for N=4:
#        template body color 2, connector color 8:
#            0 8 0 0
#            2 2 0 0
#            2 2 2 0
#            0 2 0 0
#      The nub sits above the unique anchor cell, so this template's code is
#      "up". In the middle band, every actual marker is a shared neutral color
#      (not the template color) plus the same connector nub beside it. A marker
#      at local slot (2,1) with an "up" nub therefore means: use the template
#      whose top-band connector also points up, then stamp it into output block
#      (2,1).
#    - The bottom swatch is no longer just a color sample. Its color still
#      gives the repaint color, but its size now encodes geometry:
#        1 pixel -> identity, 2 pixels -> horizontal flip, 3 pixels -> 90 deg
#        counterclockwise rotation.
#      So the decode chain is connector -> template index, then swatch size ->
#      transform, then recolor and paste.
# b) Complexity calibration:
#    Four templates maximum maps cleanly to four orthogonal connector
#    directions, which is enough variation to be visibly different without
#    becoming noisy. Swatch sizes are limited to three values so the transform
#    code repeats across samples and stays learnable.
# c) Broken original assumptions:
#    - "marker color == template color" no longer holds, so the old
#      tcol_to_j lookup must be removed.
#    - Top-band tiles no longer contain exactly one non-background color, so
#      template parsing must separate body pixels from the connector pixel.
#    - Bottom swatches no longer merely supply a color; their pixel count must
#      be read and mapped to a transform before rendering.
# d) Placement legality:
#    Each template uses one fixed connector direction. Marker slots for that
#    template are sampled only where the auxiliary connector stays inside the
#    N x N middle tile, and marker gadgets within a tile are kept disjoint so
#    no connector overwrites another marker.
# e/f/g) Observability and cleanliness:
#    The marker color is shared across all templates, so humans can see that
#    color no longer decides the template. The connector nub is visually small
#    but explicit in both the top and middle bands. Swatch lengths are drawn as
#    centered 1/2/3-pixel horizontal runs, making the transformation code easy
#    to compare across columns. Non-identity transforms are forced to act on
#    asymmetric templates so the size code has visible consequences.


_BG = 0
_SEP = 5
_MARK = 4
_CTRL = 8
_BODY_PALETTE = [1, 2, 3, 6, 7, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3]
_PAL_H_WEIGHTS = [5, 1]

_DENSITY_RANGE = (0.45, 0.82)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.50
_MAX_RETRIES = 500

_DIR_ORDER = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_COUNT_TO_TRANSFORM = {1: 1, 2: 2, 3: 3}


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _mask_signature(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8, copy=False).tobytes()


def _apply_transform(mask: np.ndarray, code: int) -> np.ndarray:
    if code == 1:
        return mask.copy()
    if code == 2:
        return np.fliplr(mask)
    if code == 3:
        return np.rot90(mask, 1)
    raise ValueError(f"unknown transform code: {code}")


def _find_template_connector(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    n = int(mask.shape[0])
    ctrl_pts = [tuple(map(int, pt)) for pt in np.argwhere(mask == _CTRL)]
    assert len(ctrl_pts) == 1
    cr, cc = ctrl_pts[0]
    body_neighbors = []
    for dr, dc in _DIR_ORDER:
        nr, nc = cr - dr, cc - dc
        if 0 <= nr < n and 0 <= nc < n and int(mask[nr, nc]) not in (_BG, _SEP, _CTRL):
            body_neighbors.append((int(nr), int(nc), int(dr), int(dc)))
    assert len(body_neighbors) == 1
    ar, ac, dr, dc = body_neighbors[0]
    return (ar, ac), (dr, dc)


def _read_marker_vector(sub: np.ndarray) -> tuple[int, int] | None:
    marks = [tuple(map(int, pt)) for pt in np.argwhere(sub == _MARK)]
    if not marks:
        return None
    vec = None
    for mr, mc in marks:
        here = []
        for dr, dc in _DIR_ORDER:
            nr, nc = mr + dr, mc + dc
            if 0 <= nr < sub.shape[0] and 0 <= nc < sub.shape[1] and int(sub[nr, nc]) == _CTRL:
                here.append((int(dr), int(dc)))
        assert len(here) == 1
        if vec is None:
            vec = here[0]
        else:
            assert vec == here[0], "tile column must use one connector pattern"
    return vec


def _extract_palette(sub: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere((sub != _BG) & (sub != _SEP))
    assert pts.size > 0
    vals = {int(sub[int(r), int(c)]) for r, c in pts}
    assert len(vals) == 1
    color = vals.pop()
    count = int(len(pts))
    assert count in _COUNT_TO_TRANSFORM
    return color, _COUNT_TO_TRANSFORM[count]


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    palette: list[tuple[int, int]],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        recolor, transform_code = palette[j]
        tile_mask = _apply_transform(templates[j], transform_code)
        tile = np.where(tile_mask, recolor, _BG)
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


def _candidate_anchor_cells(mask: np.ndarray, direction: tuple[int, int]) -> list[tuple[int, int]]:
    n = int(mask.shape[0])
    dr, dc = direction
    anchors = []
    for r in range(n):
        for c in range(n):
            if not bool(mask[r, c]):
                continue
            cr, cc = r + dr, c + dc
            if not (0 <= cr < n and 0 <= cc < n):
                continue
            if bool(mask[cr, cc]):
                continue
            neighbors = 0
            for tdr, tdc in _DIR_ORDER:
                nr, nc = cr + tdr, cc + tdc
                if 0 <= nr < n and 0 <= nc < n and bool(mask[nr, nc]):
                    neighbors += 1
            if neighbors == 1:
                anchors.append((int(r), int(c)))
    return anchors


def _sample_template_with_connector(
    n: int,
    body_color: int,
    direction: tuple[int, int],
    transform_code: int,
    forbidden_signatures: set[bytes],
) -> tuple[np.ndarray, tuple[int, int], bytes]:
    for _ in range(200):
        try:
            sprite = random_sprite(
                n,
                n,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[body_color],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=_BG,
            )
        except Exception:
            continue
        mask = sprite == body_color
        if transform_code != 1 and np.array_equal(_apply_transform(mask, transform_code), mask):
            continue
        sig = _mask_signature(mask)
        if sig in forbidden_signatures:
            continue
        anchors = _candidate_anchor_cells(mask, direction)
        if not anchors:
            continue
        anchor = random.choice(anchors)
        return mask, anchor, sig
    raise RuntimeError("failed to sample a connector-ready template")


def _valid_slots_for_direction(n: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    slots = []
    for r in range(n):
        for c in range(n):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n:
                slots.append((int(r), int(c)))
    return slots


def _slot_cells(slot: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = slot
    dr, dc = direction
    return {(int(r), int(c)), (int(r + dr), int(c + dc))}


def _choose_k(n: int) -> int:
    if n == 3:
        return 3
    return random.randint(*_K_RANGE)


def _build_balanced_quotas(n: int, k: int) -> list[int]:
    base = 2
    total = random.randint(max(int(np.ceil(_MIN_SLOT_COVERAGE * n * n)), base * k), n * n)
    quotas = [base] * k
    extra = total - base * k
    order = list(range(k))
    while extra > 0:
        random.shuffle(order)
        for j in order:
            if extra == 0:
                break
            quotas[j] += 1
            extra -= 1
    return quotas


def _assign_placements(n: int, directions: list[tuple[int, int]]) -> list[tuple[int, int, int]] | None:
    k = len(directions)
    quotas = _build_balanced_quotas(n, k)
    valid_slots = [_valid_slots_for_direction(n, direction) for direction in directions]
    slot_popularity: dict[tuple[int, int], int] = {}
    for slots in valid_slots:
        for slot in slots:
            slot_popularity[slot] = slot_popularity.get(slot, 0) + 1

    for _ in range(80):
        slot_used: set[tuple[int, int]] = set()
        occ_by_template = [set() for _ in directions]
        placements: list[tuple[int, int, int]] = []

        schedule: list[int] = []
        for level in range(max(quotas)):
            layer = [j for j, quota in enumerate(quotas) if quota > level]
            random.shuffle(layer)
            schedule.extend(layer)

        ok = True
        for j in schedule:
            direction = directions[j]
            choices = []
            for slot in valid_slots[j]:
                if slot in slot_used:
                    continue
                cells = _slot_cells(slot, direction)
                if cells & occ_by_template[j]:
                    continue
                choices.append(slot)
            if not choices:
                ok = False
                break
            choices.sort(key=lambda slot: (slot_popularity[slot], random.random()))
            slot = choices[0]
            slot_used.add(slot)
            occ_by_template[j] |= _slot_cells(slot, direction)
            placements.append((j, slot[0], slot[1]))

        if ok:
            random.shuffle(placements)
            return placements
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a connector-keyed stamp map: match marker gadgets to template gadgets,
    decode swatch size into a transform, then stamp the transformed template.

    Input structure:
    - Grid split into K tile columns of width N, with 1-column dashed separators.
    - Top band (rows 0..N-1): K template masks; each tile has one body color plus
      one connector pixel in shared connector color.
    - Middle band (rows N+1..2N): neutral marker pixels, each with one adjacent
      connector pixel; local marker position still chooses the output block.
    - Bottom band (rows 2N+2..): one horizontal swatch per tile column whose
      color is the repaint color and whose length is 1/2/3.

    Rule:
    - Read every top-band connector direction to index the templates.
    - For each tile column in the middle band, read the connector direction next
      to its marker pixels to decide which template that column refers to.
    - Decode the swatch length as a geometric transform
      (1=identity, 2=horizontal flip, 3=90-degree rotation), recolor the chosen
      template with the swatch color, and stamp it into every block named by the
      local marker coordinates.

    Key insight: The marker color is no longer informative. The tiny connector
    nubs identify the template, while the bottom swatch size controls how that
    template is reoriented before stamping.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if (g[:, c] == sep).sum() >= H // 2]
    sep_rows = [r for r in range(H) if (g[r, :] == sep).sum() >= W // 2]
    assert sep_cols and sep_rows

    n = int(sep_cols[0])
    starts = [0] + [int(c + 1) for c in sep_cols]
    k = len(starts)

    templates: list[np.ndarray] = []
    direction_to_template: dict[tuple[int, int], int] = {}
    ctrl_colors = set()
    for st in starts:
        sub = g[0:n, st:st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 2
        freq = Counter(map(int, sub.flatten()))
        ctrl_color = min(vals, key=lambda col: freq[col])
        body_color = max(vals, key=lambda col: freq[col])
        ctrl_colors.add(ctrl_color)
        body = sub == body_color
        (_anchor, direction) = _find_template_connector(np.where(sub == ctrl_color, _CTRL, np.where(body, body_color, _BG)))
        j = len(templates)
        assert direction not in direction_to_template
        direction_to_template[direction] = j
        templates.append(body)
    assert ctrl_colors == {_CTRL}

    pal_r0 = int(sep_rows[-1] + 1)
    palette = []
    for st in starts:
        palette.append(_extract_palette(g[pal_r0:H, st:st + n]))

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        sub = g[mid_r0:mid_r0 + n, st:st + n]
        direction = _read_marker_vector(sub)
        if direction is None:
            continue
        j = direction_to_template[direction]
        recolor, transform_code = palette[j]
        tile_mask = _apply_transform(templates[j], transform_code)
        tile = np.where(tile_mask, recolor, bg)
        for lr, lc in np.argwhere(sub == _MARK):
            rr, cc = int(lr), int(lc)
            assert (rr, cc) not in slot_owner or slot_owner[(rr, cc)] == j
            slot_owner[(rr, cc)] = j
            out[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n] = tile

    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = _choose_k(n)
    directions = random.sample(_DIR_ORDER, k)

    body_colors = random.sample(_BODY_PALETTE, k)
    palette_colors = random.sample(_BODY_PALETTE, k)

    transform_codes = [1, 2, 3]
    while len(transform_codes) < k:
        transform_codes.append(random.choice([1, 2, 3]))
    random.shuffle(transform_codes)
    if 2 not in transform_codes:
        transform_codes[random.randrange(k)] = 2
    if 3 not in transform_codes:
        transform_codes[random.randrange(k)] = 3
    palette = list(zip(palette_colors, transform_codes))

    templates = []
    anchors = []
    signatures: set[bytes] = set()
    try:
        for body_color, direction, (_pcol, transform_code) in zip(body_colors, directions, palette):
            mask, anchor, sig = _sample_template_with_connector(
                n,
                body_color,
                direction,
                transform_code,
                signatures,
            )
            templates.append(mask)
            anchors.append(anchor)
            signatures.add(sig)
    except RuntimeError:
        return None

    placements = _assign_placements(n, directions)
    if placements is None:
        return None
    expected = _build_expected(n, templates, palette, placements)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    grid = np.full((core_h, core_w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, (mask, direction, anchor, body_color) in enumerate(zip(templates, directions, anchors, body_colors)):
        tile = np.where(mask, body_color, _BG)
        ar, ac = anchor
        dr, dc = direction
        tile[ar + dr, ac + dc] = _CTRL
        paste(grid, tile, 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        tc = tile_assign[j]
        mr = n + 1 + rr
        mc = _raw_col(tc, cc, n)
        dr, dc = directions[j]
        grid[mr, mc] = _MARK
        grid[mr + dr, mc + dc] = _CTRL

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, (color, transform_code) in enumerate(palette):
        size = transform_code
        start = j * (n + 1)
        c0 = start + (n - size) // 2
        grid[swatch_r, c0:c0 + size] = color

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
