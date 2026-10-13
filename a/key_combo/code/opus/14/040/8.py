from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400
SELECTION_ATTEMPTS = 2500
TILE_ATTEMPTS = 200
OCCLUSION_ATTEMPTS = 150


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _inside(box: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _visible_mask(patch: np.ndarray) -> np.ndarray:
    return patch != BLACK


def _valid_row_period(patch: np.ndarray, period: int) -> bool:
    h, w = patch.shape
    visible = _visible_mask(patch)
    supported = False
    for c in range(w):
        for r in range(h - period):
            if visible[r, c] and visible[r + period, c]:
                supported = True
                if int(patch[r, c]) != int(patch[r + period, c]):
                    return False
    return supported


def _valid_col_period(patch: np.ndarray, period: int) -> bool:
    h, w = patch.shape
    visible = _visible_mask(patch)
    supported = False
    for r in range(h):
        for c in range(w - period):
            if visible[r, c] and visible[r, c + period]:
                supported = True
                if int(patch[r, c]) != int(patch[r, c + period]):
                    return False
    return supported


def _detect_periods(patch: np.ndarray) -> tuple[int, int]:
    h, w = patch.shape
    row_period = None
    col_period = None
    for p in range(1, min(5, h - 1) + 1):
        if _valid_row_period(patch, p):
            row_period = p
            break
    for p in range(1, min(5, w - 1) + 1):
        if _valid_col_period(patch, p):
            col_period = p
            break
    assert row_period is not None and col_period is not None, "failed to detect the translational period"
    return int(row_period), int(col_period)


def _recover_patch(pattern_patch: np.ndarray, occlusion_box: tuple[int, int, int, int], row_period: int, col_period: int) -> np.ndarray:
    recovered = pattern_patch.copy()
    h, w = recovered.shape
    r0, c0, r1, c1 = occlusion_box
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            assert int(recovered[rr, cc]) == BLACK
            residues = (
                rr % row_period,
                cc % col_period,
            )
            candidates = []
            for r2 in range(h):
                for c2 in range(w):
                    if int(recovered[r2, c2]) == BLACK:
                        continue
                    if (r2 % row_period, c2 % col_period) == residues:
                        candidates.append(int(recovered[r2, c2]))
            assert candidates, "each occluded orbit must have a visible representative"
            assert len(set(candidates)) == 1, "visible orbit representatives must agree"
            recovered[rr, cc] = candidates[0]
    return recovered


def _stamp_with_priorities(
    base: np.ndarray,
    sprite: np.ndarray,
    markers: Iterable[tuple[int, int]],
    occlusion_center: tuple[int, int],
    pattern_box: tuple[int, int, int, int],
) -> np.ndarray:
    out = base.copy()
    h, w = sprite.shape
    rr_center = h // 2
    cc_center = w // 2
    claimed = np.zeros(out.shape, dtype=bool)
    ordered = sorted((tuple(map(int, m)) for m in markers), key=lambda m: _manhattan(m, occlusion_center))
    for mr, mc in ordered:
        top = mr - rr_center
        left = mc - cc_center
        for sr in range(h):
            for sc in range(w):
                tr = top + sr
                tc = left + sc
                if not (0 <= tr < out.shape[0] and 0 <= tc < out.shape[1]):
                    continue
                if _inside(pattern_box, tr, tc):
                    continue
                if claimed[tr, tc]:
                    continue
                out[tr, tc] = int(sprite[sr, sc])
                claimed[tr, tc] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the occluded periodic patch, then use that restored patch as a sprite stamped onto each maroon marker.

    Input structure:
    - A grey background surrounds one rectangular region filled with a translationally periodic multicolor pattern.
    - Inside that patterned rectangle, one solid black rectangle occludes the true pattern colors.
    - Two to four maroon marker pixels lie on the grey background, and the black occlusion is the only black region.

    Rule:
    - Find the patterned rectangle and detect its row and column translation periods while ignoring the black occlusion.
    - Reconstruct every black cell from another visible cell in the same translation orbit, then crop the restored occlusion rectangle as sprite P.
    - Keep the restored original patterned rectangle unchanged, and stamp P centered on every maroon marker outside the pattern.
    - If stamped regions overlap, each overlapping cell belongs to the stamp whose marker has smaller Manhattan distance to the original occlusion center.

    Key insight:
    - The missing rectangle is simultaneously the defect that must be repaired and the only available template that can later be copied at the markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    pattern_mask = (g != GREY) & (g != MAROON)
    pattern_box = _bbox(pattern_mask)
    assert pattern_box is not None, "expected one patterned rectangle"
    pr0, pc0, pr1, pc1 = pattern_box
    pattern_patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    occlusion_mask = g == BLACK
    occlusion_box_global = _bbox(occlusion_mask)
    assert occlusion_box_global is not None, "expected one black occlusion rectangle"
    or0, oc0, or1, oc1 = occlusion_box_global
    assert pr0 <= or0 <= or1 <= pr1 and pc0 <= oc0 <= oc1 <= pc1, "occlusion must lie inside the patterned rectangle"
    assert np.all(g[or0 : or1 + 1, oc0 : oc1 + 1] == BLACK), "occlusion must be a solid rectangle"

    occlusion_box_local = (or0 - pr0, oc0 - pc0, or1 - pr0, oc1 - pc0)
    row_period, col_period = _detect_periods(pattern_patch)
    recovered_patch = _recover_patch(pattern_patch, occlusion_box_local, row_period, col_period)
    sprite = _crop(recovered_patch, occlusion_box_local)

    out = g.copy()
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = recovered_patch
    occlusion_center = ((or0 + or1) // 2, (oc0 + oc1) // 2)
    markers = [tuple(map(int, rc)) for rc in np.argwhere(g == MAROON)]
    assert 2 <= len(markers) <= 4, "expected 2 to 4 markers"
    return _stamp_with_priorities(out, sprite, markers, occlusion_center, pattern_box)


def _exact_row_period(tile: np.ndarray) -> int:
    h = tile.shape[0]
    for p in range(1, h + 1):
        if all(np.array_equal(tile[r], tile[r % p]) for r in range(h)):
            return p
    return h


def _exact_col_period(tile: np.ndarray) -> int:
    w = tile.shape[1]
    for p in range(1, w + 1):
        if all(np.array_equal(tile[:, c], tile[:, c % p]) for c in range(w)):
            return p
    return w


def _make_tile(period_h: int, period_w: int) -> np.ndarray | None:
    for _ in range(TILE_ATTEMPTS):
        color_count = random.randint(3, min(5, period_h * period_w))
        palette = random.sample(PATTERN_COLORS, color_count)
        tile = np.random.choice(palette, size=(period_h, period_w), replace=True).astype(int)
        if len(set(map(int, tile.flat))) < 3:
            continue
        if _exact_row_period(tile) != period_h or _exact_col_period(tile) != period_w:
            continue
        if len({tuple(map(int, row)) for row in tile}) == 1:
            continue
        if len({tuple(map(int, tile[:, c])) for c in range(tile.shape[1])}) == 1:
            continue
        return tile
    return None


def _choose_occlusion(pattern: np.ndarray, period_h: int, period_w: int) -> tuple[int, int, int, int] | None:
    ph, pw = pattern.shape
    heights = [s for s in (3, 5) if s <= ph]
    widths = [s for s in (3, 5) if s <= pw]
    if not heights or not widths:
        return None

    residue_counts: dict[tuple[int, int], int] = {}
    for r in range(ph):
        for c in range(pw):
            key = (r % period_h, c % period_w)
            residue_counts[key] = residue_counts.get(key, 0) + 1

    for _ in range(OCCLUSION_ATTEMPTS):
        oh = random.choice(heights)
        ow = random.choice(widths)
        if random.random() < 0.35:
            row_choices = [0, ph - oh] if ph > oh else [0]
            col_choices = [0, pw - ow] if pw > ow else [0]
            if random.random() < 0.5:
                r0 = random.choice(row_choices)
                c0 = random.randint(0, pw - ow)
            else:
                r0 = random.randint(0, ph - oh)
                c0 = random.choice(col_choices)
        else:
            r0 = random.randint(0, ph - oh)
            c0 = random.randint(0, pw - ow)

        sprite = pattern[r0 : r0 + oh, c0 : c0 + ow]
        if len(set(map(int, sprite.flat))) < 3:
            continue

        masked_counts: dict[tuple[int, int], int] = {}
        for rr in range(r0, r0 + oh):
            for cc in range(c0, c0 + ow):
                key = (rr % period_h, cc % period_w)
                masked_counts[key] = masked_counts.get(key, 0) + 1
        if any(masked_counts[key] >= residue_counts[key] for key in masked_counts):
            continue

        occluded = pattern.copy()
        occluded[r0 : r0 + oh, c0 : c0 + ow] = BLACK
        detected = _detect_periods(occluded)
        if detected != (period_h, period_w):
            continue
        return r0, c0, oh, ow
    return None


def _footprint_box(marker: tuple[int, int], sprite_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = sprite_shape
    ch = h // 2
    cw = w // 2
    mr, mc = marker
    return mr - ch, mc - cw, mr + ch, mc + cw


def _visible_overlap_cells(
    marker_a: tuple[int, int],
    marker_b: tuple[int, int],
    sprite: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    fa = _footprint_box(marker_a, sprite.shape)
    fb = _footprint_box(marker_b, sprite.shape)
    inter = _intersect(fa, fb)
    if inter is None:
        return []
    h, w = canvas_shape
    out = []
    for r in range(inter[0], inter[2] + 1):
        for c in range(inter[1], inter[3] + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if _inside(pattern_box, r, c):
                continue
            out.append((r, c, r - fa[0], c - fa[1]))
    return out


def _has_meaningful_stamp_overlap(
    markers: list[tuple[int, int]],
    sprite: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
) -> bool:
    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            fa = _footprint_box(markers[i], sprite.shape)
            fb = _footprint_box(markers[j], sprite.shape)
            inter = _intersect(fa, fb)
            if inter is None:
                continue
            for r in range(inter[0], inter[2] + 1):
                for c in range(inter[1], inter[3] + 1):
                    if not (0 <= r < canvas_shape[0] and 0 <= c < canvas_shape[1]):
                        continue
                    if _inside(pattern_box, r, c):
                        continue
                    a_sr = r - fa[0]
                    a_sc = c - fa[1]
                    b_sr = r - fb[0]
                    b_sc = c - fb[1]
                    if int(sprite[a_sr, a_sc]) != int(sprite[b_sr, b_sc]):
                        return True
    return False


def _has_pattern_conflict(
    marker: tuple[int, int],
    sprite: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    recovered_grid: np.ndarray,
) -> bool:
    footprint = _footprint_box(marker, sprite.shape)
    inter = _intersect(footprint, pattern_box)
    if inter is None:
        return False
    for r in range(inter[0], inter[2] + 1):
        for c in range(inter[1], inter[3] + 1):
            sr = r - footprint[0]
            sc = c - footprint[1]
            if int(sprite[sr, sc]) != int(recovered_grid[r, c]):
                return True
    return False


def _marker_candidates(
    grid: np.ndarray,
    sprite: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    recovered_grid: np.ndarray,
) -> list[dict]:
    h, w = grid.shape
    sh, sw = sprite.shape
    ch = sh // 2
    cw = sw // 2
    candidates = []
    for r in range(ch, h - ch):
        for c in range(cw, w - cw):
            if int(grid[r, c]) != GREY:
                continue
            if _inside(pattern_box, r, c):
                continue
            marker = (r, c)
            footprint = _footprint_box(marker, sprite.shape)
            touch_pattern = _intersect(footprint, pattern_box) is not None
            meaningful_pattern = _has_pattern_conflict(marker, sprite, pattern_box, recovered_grid)
            candidates.append(
                {
                    "marker": marker,
                    "footprint": footprint,
                    "touch_pattern": touch_pattern,
                    "meaningful_pattern": meaningful_pattern,
                }
            )
    return candidates


def _select_markers(
    candidates: list[dict],
    count: int,
    occlusion_center: tuple[int, int],
    sprite: np.ndarray,
    pattern_box: tuple[int, int, int, int],
    canvas_shape: tuple[int, int],
    need_overlap: bool,
    need_pattern_touch: bool,
) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    for _ in range(SELECTION_ATTEMPTS):
        picked = random.sample(candidates, count)
        markers = [item["marker"] for item in picked]
        distances = [_manhattan(marker, occlusion_center) for marker in markers]
        if len(set(distances)) != count:
            continue
        if need_pattern_touch:
            if not any(item["meaningful_pattern"] for item in picked):
                continue
        else:
            if any(item["touch_pattern"] for item in picked):
                continue
        overlap = _has_meaningful_stamp_overlap(markers, sprite, pattern_box, canvas_shape)
        if overlap != need_overlap:
            continue
        return markers
    return None


def _build_expected_output(
    recovered_grid: np.ndarray,
    sprite: np.ndarray,
    markers: list[tuple[int, int]],
    occlusion_center: tuple[int, int],
    pattern_box: tuple[int, int, int, int],
) -> np.ndarray:
    out = recovered_grid.copy()
    for mr, mc in markers:
        out[mr, mc] = MAROON
    return _stamp_with_priorities(out, sprite, markers, occlusion_center, pattern_box)


def _gen_attempt() -> np.ndarray | None:
    period_h = random.randint(2, 5)
    period_w = random.randint(2, 5)
    tile = _make_tile(period_h, period_w)
    if tile is None:
        return None

    rep_h = random.randint(3, 4)
    rep_w = random.randint(3, 4)
    pattern = np.tile(tile, (rep_h, rep_w))
    ph, pw = pattern.shape

    occlusion = _choose_occlusion(pattern, period_h, period_w)
    if occlusion is None:
        return None
    occ_r, occ_c, occ_h, occ_w = occlusion
    sprite = pattern[occ_r : occ_r + occ_h, occ_c : occ_c + occ_w].copy()

    margin_top = random.randint(3, 6)
    margin_bottom = random.randint(3, 6)
    margin_left = random.randint(3, 6)
    margin_right = random.randint(3, 6)
    grid_h = ph + margin_top + margin_bottom
    grid_w = pw + margin_left + margin_right
    if max(grid_h, grid_w) > 30:
        return None

    pattern_box = (margin_top, margin_left, margin_top + ph - 1, margin_left + pw - 1)
    recovered_grid = np.full((grid_h, grid_w), GREY, dtype=int)
    recovered_grid[margin_top : margin_top + ph, margin_left : margin_left + pw] = pattern

    inp = recovered_grid.copy()
    inp[margin_top + occ_r : margin_top + occ_r + occ_h, margin_left + occ_c : margin_left + occ_c + occ_w] = BLACK
    occlusion_center = (margin_top + occ_r + occ_h // 2, margin_left + occ_c + occ_w // 2)

    need_overlap = random.random() < 0.65
    need_pattern_touch = random.random() < 0.4
    marker_count = random.randint(2, 4)
    candidates = _marker_candidates(inp, sprite, pattern_box, recovered_grid)
    markers = _select_markers(
        candidates,
        marker_count,
        occlusion_center,
        sprite,
        pattern_box,
        inp.shape,
        need_overlap=need_overlap,
        need_pattern_touch=need_pattern_touch,
    )
    if markers is None:
        return None

    for mr, mc in markers:
        inp[mr, mc] = MAROON

    expected = _build_expected_output(recovered_grid, sprite, markers, occlusion_center, pattern_box)
    if np.array_equal(expected, inp):
        return None

    solved = main(inp)
    if not np.array_equal(solved, expected):
        return None

    detected_box = _bbox((inp != GREY) & (inp != MAROON))
    if detected_box != pattern_box:
        return None
    local_patch = _crop(inp, pattern_box)
    if _detect_periods(local_patch) != (period_h, period_w):
        return None
    recovered_patch = _recover_patch(local_patch, (occ_r, occ_c, occ_r + occ_h - 1, occ_c + occ_w - 1), period_h, period_w)
    if not np.array_equal(recovered_patch[occ_r : occ_r + occ_h, occ_c : occ_c + occ_w], sprite):
        return None

    distances = [_manhattan(marker, occlusion_center) for marker in markers]
    if len(set(distances)) != len(distances):
        return None
    if need_overlap and not _has_meaningful_stamp_overlap(markers, sprite, pattern_box, inp.shape):
        return None
    if need_pattern_touch and not any(_has_pattern_conflict(marker, sprite, pattern_box, recovered_grid) for marker in markers):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
