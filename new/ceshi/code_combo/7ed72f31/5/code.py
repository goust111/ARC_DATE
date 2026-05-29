# Combo variant of 7ed72f31 (index 5)
# combo_spec: D1/1+D11/1
# Name: 多方向平移广播

from __future__ import annotations

from collections import Counter
import random
import sys
from pathlib import Path

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import get_bg, label_mask


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The original task used one red geometry per group and produced one reflected copy.
#    This combo replaces reflection with size-encoded translation (D1) and also lets one
#    payload respond to two red lines at once (D11). The actionable object is therefore
#    a single 4-connected component made of:
#    - one horizontal red line of length Lh,
#    - one vertical red line of length Lv,
#    - one monochrome payload in exactly one quadrant near their intersection.
#    The horizontal line encodes a vertical translation of magnitude Lh. The vertical
#    line encodes a horizontal translation of magnitude Lv. The payload is broadcast
#    along both vectors at the same time:
#    - payload above the red row -> shift down by Lh,
#    - payload below the red row -> shift up by Lh,
#    - payload left  of the red column -> shift right by Lv,
#    - payload right of the red column -> shift left  by Lv.
#    Example: if the payload sits in the upper-left quadrant, then output paints three
#    copies: one below-left (row shift), one upper-right (col shift), and one lower-right
#    (row+col combined shift). The original task would either skip a red cross entirely or
#    reflect around it; this combo instead preserves orientation and uses the visible line
#    lengths as the displacement values.
# b) Complexity calibration
#    Every actionable group uses exactly two lines, not an arbitrary set of markers.
#    That keeps the rule small enough to infer while still visibly different from the
#    source task: the solver must notice both "line length matters" and "one source makes
#    several translated copies". We generate 3-4 groups per grid and force repeated but
#    non-identical length pairs; at least one group has Lh != Lv so the two encoded
#    distances cannot be collapsed into a single generic "copy across the cross" rule.
# c) Implicit assumptions review against the old solver
#    - Old code assumed the red geometry resolved to a single row / single column / point.
#      That fails here because the red pixels intentionally form a row+column union, so the
#      solver now has a dedicated cross classifier.
#    - Old reflection helpers assumed symmetric targets around an axis. That no longer
#      holds; we instead derive directed translation vectors from payload side and line
#      length.
#    - Old bbox-based payload eligibility could accidentally pull stray same-color pixels.
#      That is dangerous for broadcast because one wrong pixel would create several copies.
#      The combo solver therefore only translates the exact non-red pixels of the current
#      4-connected mixed component.
#    - Old isolated red pixel behavior is removed. Keeping it would mix two unrelated
#      marker grammars in one task and make the intended combo rule harder to read.
# d) Placement legality
#    Generation is output-first. Each group is built in local coordinates around a red
#    cross; the payload lives in one quadrant, and its box depth is always strictly smaller
#    than the translation distance induced by the relevant line. That guarantees:
#    - no translated copy overlaps the original payload,
#    - no translated copy lands on the red lines,
#    - the combined row+col copy occupies the diagonally opposite quadrant.
#    Local patches are then placed on the canvas with a 1-cell moat so groups cannot merge.
# e) Visual observability
#    Human solvers can compare input/output and see that identical shapes reappear in three
#    translated locations around the same cross, while the offsets match the visible red
#    line lengths. Because the payload shapes are irregular 2D sprites rather than bars,
#    "translated without flipping" is visually obvious.
# f) Visual cleanliness
#    Groups use distinct colors, occupy disjoint patches, and keep the red cross intact in
#    both input and output. The scene remains legible: marker, original payload, and the
#    three broadcast copies are separable by eye.
# g) New solver assumptions and generator guarantees
#    - Red markers are 1-cell-thick contiguous crosses only.
#    - All payload pixels of an actionable group lie in one quadrant relative to the cross.
#    - Each payload uses one color and is exactly the non-red subset of the mixed component.
#    The generator constructs only such inputs, and `main()` explicitly checks these
#    assumptions before painting any translated copies.

MARK = 2
LINE_LENGTHS = (3, 4, 5, 6)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= int(r) < int(h) and 0 <= int(c) < int(w)


def _is_consecutive(vals: list[int]) -> bool:
    return all(int(b) == int(a) + 1 for a, b in zip(vals, vals[1:]))


def _classify_cross_marker(reds: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    if len(reds) < 5:
        return None

    row_counts = Counter(int(r) for r, _ in reds)
    col_counts = Counter(int(c) for _, c in reds)
    row_anchor = max(sorted(row_counts), key=lambda r: int(row_counts[r]))
    col_anchor = max(sorted(col_counts), key=lambda c: int(col_counts[c]))

    row_cells = sorted(int(c) for r, c in reds if int(r) == int(row_anchor))
    col_cells = sorted(int(r) for r, c in reds if int(c) == int(col_anchor))
    if len(row_cells) < 2 or len(col_cells) < 2:
        return None
    if not _is_consecutive(row_cells) or not _is_consecutive(col_cells):
        return None

    red_set = {(int(r), int(c)) for r, c in reds}
    expected = {(int(row_anchor), int(c)) for c in row_cells}
    expected |= {(int(r), int(col_anchor)) for r in col_cells}
    if red_set != expected:
        return None

    return int(row_anchor), int(col_anchor), len(row_cells), len(col_cells)


def _extract_actionable_groups(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    lab, n = label_mask(g != int(bg), connectivity=4)
    groups: list[dict] = []

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue

        reds = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == int(MARK)]
        payload = [(int(r), int(c), int(g[r, c])) for r, c in coords if int(g[r, c]) != int(MARK)]
        geom = _classify_cross_marker(reds)
        if geom is None or not payload:
            continue

        row_anchor, col_anchor, row_len, col_len = map(int, geom)
        if any(int(r) == int(row_anchor) or int(c) == int(col_anchor) for r, c, _v in payload):
            continue

        if all(int(r) < int(row_anchor) for r, _c, _v in payload):
            row_sign = -1
            dr = int(row_len)
        elif all(int(r) > int(row_anchor) for r, _c, _v in payload):
            row_sign = 1
            dr = -int(row_len)
        else:
            continue

        if all(int(c) < int(col_anchor) for _r, c, _v in payload):
            col_sign = -1
            dc = int(col_len)
        elif all(int(c) > int(col_anchor) for _r, c, _v in payload):
            col_sign = 1
            dc = -int(col_len)
        else:
            continue

        groups.append(
            {
                "row_anchor": int(row_anchor),
                "col_anchor": int(col_anchor),
                "row_len": int(row_len),
                "col_len": int(col_len),
                "row_sign": int(row_sign),
                "col_sign": int(col_sign),
                "dr": int(dr),
                "dc": int(dc),
                "payload": payload,
                "payload_size": len(payload),
            }
        )

    return groups


def _paint_translation_targets(out: np.ndarray, groups: list[dict], include_combined: bool) -> np.ndarray:
    painted = np.array(out, dtype=int)
    shape = painted.shape

    for group in groups:
        dr = int(group["dr"])
        dc = int(group["dc"])
        for r, c, color in group["payload"]:
            targets = [(int(r + dr), int(c)), (int(r), int(c + dc))]
            if include_combined:
                targets.append((int(r + dr), int(c + dc)))
            for rr, cc in set(targets):
                if _in_bounds(shape, int(rr), int(cc)):
                    painted[int(rr), int(cc)] = int(color)

    return painted


def _reflection_baseline(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    groups = _extract_actionable_groups(g)

    for group in groups:
        r0 = int(group["row_anchor"])
        c0 = int(group["col_anchor"])
        for r, c, color in group["payload"]:
            targets = {
                (int(2 * r0 - r), int(c)),
                (int(r), int(2 * c0 - c)),
                (int(2 * r0 - r), int(2 * c0 - c)),
            }
            for rr, cc in targets:
                if _in_bounds(out.shape, int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    out = g.copy()
    groups = _extract_actionable_groups(g)
    return _paint_translation_targets(out, groups, include_combined=True)


def _grow_connected_shape(
    rng: random.Random, box_h: int, box_w: int, area: int, seed: tuple[int, int]
) -> set[tuple[int, int]]:
    shape = {seed}
    while len(shape) < int(area):
        frontier: list[tuple[int, int]] = []
        for r, c in shape:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= int(rr) < int(box_h) and 0 <= int(cc) < int(box_w) and (int(rr), int(cc)) not in shape:
                    frontier.append((int(rr), int(cc)))
        if not frontier:
            break
        shape.add(rng.choice(frontier))
    return shape


def _build_irregular_quadrant_shape(
    rng: random.Random, box_h: int, box_w: int, row_sign: int, col_sign: int
) -> set[tuple[int, int]]:
    near_r = int(box_h - 1) if int(row_sign) < 0 else 0
    near_c = int(box_w - 1) if int(col_sign) < 0 else 0
    seed = (int(near_r), int(near_c))
    max_area = min(6, int(box_h) * int(box_w))
    min_area = 3 if max_area <= 3 else 4

    for _ in range(60):
        area = rng.randint(int(min_area), int(max_area))
        shape = _grow_connected_shape(rng, int(box_h), int(box_w), int(area), seed)
        rows = {int(r) for r, _ in shape}
        cols = {int(c) for _, c in shape}
        if len(rows) < 2 or len(cols) < 2:
            continue
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(shape) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
            continue
        return shape

    fallback = {seed, (int(near_r), max(0, int(near_c) - 1)), (max(0, int(near_r) - 1), int(near_c)), (0, 0)}
    return {(int(r), int(c)) for r, c in fallback}


def _build_cross_group(
    rng: random.Random, color: int, bg: int, row_len: int, col_len: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    box_h = rng.randint(2, min(3, int(row_len) - 1))
    box_w = rng.randint(2, min(3, int(col_len) - 1))
    row_sign = rng.choice([-1, 1])
    col_sign = rng.choice([-1, 1])
    dr = int(row_len) if int(row_sign) < 0 else -int(row_len)
    dc = int(col_len) if int(col_sign) < 0 else -int(col_len)

    shape = _build_irregular_quadrant_shape(rng, int(box_h), int(box_w), int(row_sign), int(col_sign))

    left_len = int(row_len) // 2
    right_len = int(row_len) - 1 - int(left_len)
    up_len = int(col_len) // 2
    down_len = int(col_len) - 1 - int(up_len)

    red_cells = {(0, int(c)) for c in range(-int(left_len), int(right_len) + 1)}
    red_cells |= {(int(r), 0) for r in range(-int(up_len), int(down_len) + 1)}

    payload_cells: set[tuple[int, int]] = set()
    for br, bc in shape:
        rr = int(br - box_h) if int(row_sign) < 0 else int(1 + br)
        cc = int(bc - box_w) if int(col_sign) < 0 else int(1 + bc)
        payload_cells.add((int(rr), int(cc)))

    row_shift_cells = {(int(r + dr), int(c)) for r, c in payload_cells}
    col_shift_cells = {(int(r), int(c + dc)) for r, c in payload_cells}
    diag_shift_cells = {(int(r + dr), int(c + dc)) for r, c in payload_cells}

    all_cells = set(red_cells) | payload_cells | row_shift_cells | col_shift_cells | diag_shift_cells
    rows = [int(r) for r, _ in all_cells]
    cols = [int(c) for _, c in all_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    patch_h = int(max_r - min_r + 1)
    patch_w = int(max_c - min_c + 1)

    inp = np.full((int(patch_h), int(patch_w)), int(bg), dtype=int)
    out = np.full((int(patch_h), int(patch_w)), int(bg), dtype=int)

    def paint(cells: set[tuple[int, int]], value: int, dst: np.ndarray) -> None:
        for r, c in cells:
            dst[int(r - min_r), int(c - min_c)] = int(value)

    paint(red_cells, int(MARK), inp)
    paint(red_cells, int(MARK), out)
    paint(payload_cells, int(color), inp)
    paint(payload_cells, int(color), out)
    paint(row_shift_cells, int(color), out)
    paint(col_shift_cells, int(color), out)
    paint(diag_shift_cells, int(color), out)

    meta = {
        "row_len": int(row_len),
        "col_len": int(col_len),
        "row_sign": int(row_sign),
        "col_sign": int(col_sign),
        "payload_area": len(payload_cells),
        "clone_count": 3,
        "anisotropic": int(row_len) != int(col_len),
        "painted_new_pixels": int(np.count_nonzero((out != int(bg)) & (inp == int(bg)))),
    }
    return inp, out, meta


def _can_place(occupied: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for r, c in np.argwhere(patch != int(bg)):
        rr = int(top + r)
        cc = int(left + c)
        r0 = max(0, int(rr) - int(pad))
        r1 = min(int(h), int(rr) + int(pad) + 1)
        c0 = max(0, int(cc) - int(pad))
        c1 = min(int(w), int(cc) + int(pad) + 1)
        if occupied[int(r0) : int(r1), int(c0) : int(c1)].any():
            return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, bg: int, top: int, left: int) -> None:
    ph, pw = patch.shape
    window = dst[int(top) : int(top) + int(ph), int(left) : int(left) + int(pw)]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def analyze_variant(grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    groups = _extract_actionable_groups(g)
    out = main(g)
    no_diag = _paint_translation_targets(g.copy(), groups, include_combined=False)
    refl = _reflection_baseline(g)
    return {
        "groups": len(groups),
        "row_lengths": [int(group["row_len"]) for group in groups],
        "col_lengths": [int(group["col_len"]) for group in groups],
        "anisotropic_groups": sum(int(group["row_len"]) != int(group["col_len"]) for group in groups),
        "combined_only_pixels": int(np.count_nonzero((out != no_diag) & (out != g))),
        "delta_vs_reflection": int(np.count_nonzero(out != refl)),
        "painted_pixels": int(np.count_nonzero(out != g)),
    }


def generate_pair(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = rng or random.Random()
    palette = [c for c in range(10) if c != int(MARK)]

    for _attempt in range(300):
        h = rng.randint(22, 28)
        w = rng.randint(22, 28)
        bg = rng.choice(palette)
        colors = [c for c in palette if c != int(bg)]
        rng.shuffle(colors)
        n_groups = rng.randint(3, 4)
        if len(colors) < n_groups:
            continue

        row_lengths = [rng.choice(LINE_LENGTHS) for _ in range(n_groups)]
        col_lengths = [rng.choice(LINE_LENGTHS) for _ in range(n_groups)]
        if len(set(row_lengths)) == 1:
            row_lengths[0] = rng.choice([x for x in LINE_LENGTHS if x != row_lengths[0]])
        if len(set(col_lengths)) == 1:
            col_lengths[0] = rng.choice([x for x in LINE_LENGTHS if x != col_lengths[0]])
        if all(int(r) == int(c) for r, c in zip(row_lengths, col_lengths)):
            idx = rng.randrange(n_groups)
            col_lengths[idx] = rng.choice([x for x in LINE_LENGTHS if x != row_lengths[idx]])

        inp = np.full((int(h), int(w)), int(bg), dtype=int)
        out = np.full((int(h), int(w)), int(bg), dtype=int)
        occupied = np.zeros((int(h), int(w)), dtype=bool)
        group_meta: list[dict] = []
        ok = True

        for color, row_len, col_len in zip(colors[:n_groups], row_lengths, col_lengths):
            patch_in, patch_out, meta = _build_cross_group(rng, int(color), int(bg), int(row_len), int(col_len))
            ph, pw = patch_out.shape
            placed = False
            for _ in range(200):
                top = rng.randint(0, int(h) - int(ph))
                left = rng.randint(0, int(w) - int(pw))
                if not _can_place(occupied, patch_out, int(bg), int(top), int(left), pad=1):
                    continue
                _paste(inp, patch_in, int(bg), int(top), int(left))
                _paste(out, patch_out, int(bg), int(top), int(left))
                occupied[int(top) : int(top) + int(ph), int(left) : int(left) + int(pw)] |= patch_out != int(bg)
                group_meta.append(dict(meta))
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue
        if not np.array_equal(main(inp), out):
            continue

        stats = analyze_variant(inp)
        if stats["groups"] != n_groups:
            continue
        if stats["anisotropic_groups"] < 1:
            continue
        if stats["combined_only_pixels"] <= 0:
            continue
        if stats["delta_vs_reflection"] <= 0:
            continue

        meta = {"bg": int(bg), "groups": group_meta, "stats": stats}
        return inp, out, meta

    raise RuntimeError("failed to generate a valid combo sample")


def generate_input() -> np.ndarray:
    inp, out, _meta = generate_pair()
    assert np.array_equal(main(inp), out)
    return inp
