# Combo variant of 62593bfd (index 5)
# combo_spec: D2/1+D4/0
# Name: 水平分流 + 紧密排列

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for the combo variant
# =====================================
# 1) Concrete fused rule
#    - Objects are still grouped by D4-canonical shape frequency.
#    - Shared/duplicated shapes (freq >= 2) move to the LEFT side.
#    - Unique shapes (freq == 1) move to the RIGHT side.
#    - Rows are preserved: each object keeps its input bbox top row r0.
#    - New packing rule: within each side, objects are sorted by input r0 and
#      then packed tightly along the horizontal axis with no gaps.
#      left side  : higher-row object gets the leftmost available slot.
#      right side : higher-row object gets the rightmost available slot.
#
#    Small trace on a 20-column grid:
#    - shared objects:
#         A: input bbox (r0=2, c0=11, w=3)
#         B: input bbox (r0=9, c0=4,  w=4)
#      Original task output      -> A:(0,11), B:(0,4)
#      D2-only horizontal split  -> A:(2,0),  B:(9,0)
#      This combo                -> A:(2,0),  B:(9,3)
#      The side changes from vertical to horizontal, and same-side overlap is
#      resolved by row-ordered compact packing instead of stacking all objects
#      on the exact same edge column.
#
# 2) Complexity calibration
#    - If input columns accidentally increase with row order, a wrong solver
#      might think the packing order is driven by c0 rather than r0.
#      The generator therefore forces at least one multi-object side, always the
#      shared side, to have row-order != column-order.
#    - Colors stay distinct per object so adjacent packed placements remain
#      separable by eye even when horizontal gaps vanish.
#    - Object count, areas, and bbox sizes stay within the original task family
#      to avoid introducing extra concepts beyond the fused routing rule.
#
# 3) Implicit-assumption audit of the original main()
#    - Original extraction stored only c0 because row changed and column was
#      preserved. That premise is invalid here; r0 must be stored instead.
#    - Original destination logic `dest_r0 = 0 / H-h` encoded top/bottom
#      routing. The combo changes the routing axis to columns.
#    - Original `dest_c0 = c0` preserved independent positions. The combo makes
#      each object's destination column depend on earlier same-side objects, so
#      the solver now assumes a deterministic side-local ordering rule.
#    - `find_components(..., monochromatic=False, connectivity=8)` still assumes
#      all objects stay isolated 8-connected components. The generator must
#      guarantee a one-row blank band between every object's bbox strip so
#      differently colored objects never merge by touching diagonally.
#
# 4) Placement legality
#    - Output legality now depends on total width per side, not per object. The
#      generator explicitly rejects attempts where the shared-side packed width
#      or unique-side packed width exceeds W.
#    - Input legality is simpler because rows are non-overlapping bands; objects
#      may reuse columns without creating component merges.
#
# 5) Visual observability
#    - Human-visible cues are only shape-sharing, row order, and edge gravity.
#      No numeric-color ordering is used anywhere.
#    - Every generated sample forces the shared side to move nontrivially and to
#      have row-order different from column-order, so the "sort by row then pack"
#      rule is actually needed to explain the output.
#    - If there are multiple unique objects, the same mismatch is also forced on
#      the right side.
#
# 6) Visual cleanliness
#    - Each object occupies its own row band with at least one empty row between
#      neighbors. This keeps both input and output readable and prevents the
#      packed horizontal layout from collapsing into one multicolor component.
#
# 7) New solver assumptions and generator guarantees
#    - Assumption: each object is an isolated 8-connected foreground component.
#      Guarantee: row-band packing with a blank row gap.
#    - Assumption: per-side packed widths fit in-grid.
#      Guarantee: W is chosen only after widths are known.
#    - Assumption: row-order ties are rare / arbitrary.
#      Guarantee: generator gives distinct r0 values; solver still uses idx as a
#      deterministic tie-break on arbitrary valid inputs.


def _canonical_shape(coords):
    """Canonical shape key invariant under the D4 dihedral group."""
    pts = [(int(r), int(c)) for r, c in coords]

    def _norm(ps):
        mr = min(r for r, c in ps)
        mc = min(c for r, c in ps)
        return tuple(sorted((r - mr, c - mc) for r, c in ps))

    cur = list(pts)
    best = None
    for _ in range(4):
        n = _norm(cur)
        best = n if best is None or n < best else best
        n = _norm([(-r, c) for r, c in cur])
        best = n if n < best else best
        cur = [(c, -r) for r, c in cur]
    return best


def _mask_from_coords(coords, shape):
    m = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            m[rr, cc] = True
    return m


def _pack_side(entries, *, W, side):
    """Pack one side by input top-row order with side-anchored gravity."""
    placements = []
    ordered = sorted(entries, key=lambda d: (int(d["r0"]), int(d["idx"])))
    if side == "left":
        cursor = 0
        for entry in ordered:
            pat = entry["pat"]
            _h, w = map(int, pat.shape)
            placements.append((pat, int(entry["r0"]), int(cursor)))
            cursor += int(w)
        return placements

    cursor = int(W)
    for entry in ordered:
        pat = entry["pat"]
        _h, w = map(int, pat.shape)
        cursor -= int(w)
        placements.append((pat, int(entry["r0"]), int(cursor)))
    return placements


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route objects horizontally by D4-canonical shape frequency.

    Shared shapes  (freq >= 2) go to the left side.
    Unique shapes  (freq == 1) go to the right side.

    Fusion rule:
    - preserve each object's input row r0
    - within each side, sort by input r0
    - pack tightly from that side's edge with no horizontal gaps
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for idx, comp in enumerate(comps):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]

        extracted.append(
            {
                "idx": int(idx),
                "r0": int(r0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    left_entries = [d for d in extracted if int(freq[d["shape_key"]]) >= 2]
    right_entries = [d for d in extracted if int(freq[d["shape_key"]]) == 1]

    out = np.full((H, W), int(bg), dtype=int)
    for pat, dest_r0, dest_c0 in _pack_side(left_entries, W=W, side="left"):
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))
    for pat, dest_r0, dest_c0 in _pack_side(right_entries, W=W, side="right"):
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))
    return out


# ────────────────────── generator ──────────────────────


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` cells."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if max(all_r) - min(all_r) < max_dim and max(all_c) - min(all_c) < max_dim:
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    mr = min(r for r, c in cells)
    mc = min(c for r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, c in pts)
    mc = min(c for r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, c in coords)
    mc = max(c for r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _pack_rows_with_gaps(heights, H, gap=1):
    """Assign non-touching row bands so 8-neighbor component merges cannot occur."""
    n = len(heights)
    required = sum(int(h) for h in heights) + max(0, n - 1) * int(gap)
    if required > H:
        raise ValueError("insufficient height for row packing")

    slack = int(H - required)
    outer_gaps = [0] * (n + 1)
    for _ in range(slack):
        outer_gaps[random.randrange(n + 1)] += 1

    rows = []
    cur = outer_gaps[0]
    for i, h in enumerate(heights):
        rows.append(int(cur))
        cur += int(h)
        if i < n - 1:
            cur += int(gap) + outer_gaps[i + 1]
        else:
            cur += outer_gaps[-1]
    return rows


def _packed_cols(widths, W, side):
    cols = []
    if side == "left":
        cursor = 0
        for w in widths:
            cols.append(int(cursor))
            cursor += int(w)
        return cols

    cursor = int(W)
    for w in widths:
        cursor -= int(w)
        cols.append(int(cursor))
    return cols


def _assign_input_columns(entries, W, side):
    """
    Choose informative input columns.

    Requirements:
    - at least one object on the side moves horizontally in output
    - if side has multiple objects, column order must disagree with row order
      so "sort by row" is visually necessary and cannot be replaced by
      "sort by column".
    """
    widths = [int(entry["w"]) for entry in entries]
    packed = _packed_cols(widths, W, side)
    n = len(entries)

    for _ in range(500):
        cols = [random.randint(0, int(W - w)) for w in widths]
        if n > 1 and len(set(cols)) < n:
            continue
        if all(int(cols[i]) == int(packed[i]) for i in range(n)):
            continue
        if n > 1:
            col_order = sorted(range(n), key=lambda i: (int(cols[i]), int(i)))
            if col_order == list(range(n)):
                continue
        return cols

    raise ValueError("could not assign informative columns")


def _render_expected_output(objs, H, W, bg):
    out = np.full((H, W), bg, dtype=int)
    left_entries = []
    right_entries = []
    for obj in objs:
        entry = {"idx": obj["idx"], "r0": obj["r0"], "pat": obj["patch"]}
        (left_entries if obj["shared"] else right_entries).append(entry)

    for pat, dest_r0, dest_c0 in _pack_side(left_entries, W=W, side="left"):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    for pat, dest_r0, dest_c0 in _pack_side(right_entries, W=W, side="right"):
        paste(out, pat, dest_r0, dest_c0, transparent=bg)
    return out


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _gen_attempt():
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(
                base, random.randint(0, 3), random.choice([False, True])
            )
            patch = _coords_to_patch(transformed, avail[ci], bg)
            objs.append(
                {
                    "coords": transformed,
                    "patch": patch,
                    "color": avail[ci],
                    "shared": True,
                }
            )
            ci += 1

    for _ in range(n_unique):
        for _ in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            transformed = _transform_shape(
                shape, random.randint(0, 3), random.choice([False, True])
            )
            patch = _coords_to_patch(transformed, avail[ci], bg)
            objs.append(
                {
                    "coords": transformed,
                    "patch": patch,
                    "color": avail[ci],
                    "shared": False,
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    for obj in objs:
        h, w = map(int, obj["patch"].shape)
        obj["h"] = int(h)
        obj["w"] = int(w)

    min_H = max(19, sum(obj["h"] for obj in objs) + len(objs) - 1)
    if min_H > 30:
        raise ValueError("objects too tall for clean horizontal routing")
    H = random.randint(min_H, 30)

    shared_total_w = sum(obj["w"] for obj in objs if obj["shared"])
    unique_total_w = sum(obj["w"] for obj in objs if not obj["shared"])
    min_W = max(19, int(shared_total_w), int(unique_total_w))
    if min_W > 30:
        raise ValueError("objects too wide for side packing")
    W = random.randint(min_W, 30)

    random.shuffle(objs)
    packed_rows = _pack_rows_with_gaps([obj["h"] for obj in objs], H, gap=1)
    for idx, (obj, r0) in enumerate(zip(objs, packed_rows)):
        obj["idx"] = int(idx)
        obj["r0"] = int(r0)

    shared_by_row = sorted([obj for obj in objs if obj["shared"]], key=lambda o: int(o["r0"]))
    unique_by_row = sorted([obj for obj in objs if not obj["shared"]], key=lambda o: int(o["r0"]))

    shared_cols = _assign_input_columns(shared_by_row, W, "left")
    unique_cols = _assign_input_columns(unique_by_row, W, "right")
    for obj, c0 in zip(shared_by_row, shared_cols):
        obj["c0"] = int(c0)
    for obj, c0 in zip(unique_by_row, unique_cols):
        obj["c0"] = int(c0)

    out = _render_expected_output(objs, H, W, bg)

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        paste(inp, obj["patch"], int(obj["r0"]), int(obj["c0"]), transparent=bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D2/1+D4/0")
