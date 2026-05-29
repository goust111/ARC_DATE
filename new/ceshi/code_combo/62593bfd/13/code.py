# Combo variant of 62593bfd (index 13)
# combo_spec: D4/0+D11/1
# Name: 紧密排列 + 组内排序

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for this combo variant
# ======================================
# 0) Fusion rule chosen from the prompt
#    - Base task: group monochrome 8-connected objects by canonical shape modulo
#      D4; duplicated groups go to the top edge, unique groups go to the bottom.
#    - D4/0 contributes the horizontal compaction: objects on the same side are
#      no longer kept at their original c0, but are packed tightly from the left.
#    - D11/1 contributes a new intermediate ordering step: inside each canonical
#      shape group, members are ordered by their input row r0 from top to bottom.
#    - To make the two changes cooperate instead of merely coexist, this combo
#      treats each canonical group as a contiguous horizontal block. Side layout:
#         1. split into top(shared) / bottom(unique),
#         2. sort groups on the same side by the group's minimum input c0,
#         3. inside each group sort members by input r0,
#         4. flatten that order and pack left with no gaps.
#
# 1) Concrete trace
#    - Suppose the top side has two duplicated-shape groups.
#      Group A members in input:
#         A1 at (r0=3, c0=9), A2 at (r0=11, c0=2)
#      Group B members in input:
#         B1 at (r0=4, c0=15), B2 at (r0=8, c0=18)
#    - Original task:
#         all top objects keep c0, so outputs stay at columns [9, 2, 15, 18].
#    - D4-only variant:
#         top objects are globally sorted by c0, so order becomes A2, A1, B1, B2
#         and they are packed at columns [0, w(A2), ...].
#    - This combo:
#         group order is A then B because min c0(A)=2 < min c0(B)=15;
#         inside A, row order is A1 then A2 because 3 < 11;
#         inside B, row order is B1 then B2.
#         Final packed top order is A1, A2, B1, B2.
#    - Difference:
#         D11 changes the left-to-right order inside a same-shape block; D4 turns
#         that ordering into a visible compact arrangement.
#
# 2) Complexity calibration
#    - If every duplicated group happened to have row-order identical to c-order,
#      the combo would collapse to the D4-only variant. The generator therefore
#      forces at least one duplicated group to have row-order != c-order.
#    - If all chosen c0 values were already flush-left and gap-free, the combo
#      would visually collapse to the original solver. The generator therefore
#      also forces the combo output to differ from the original keep-c0 output.
#    - Colors stay distinct per object so order changes are easy to see even when
#      multiple transformed copies have the same canonical shape.
#
# 3) Implicit-assumption audit of the original main()
#    - `find_components(... connectivity=8, monochromatic=False)` assumes objects
#      do not touch even diagonally across colors. The generator preserves a
#      1-cell 8-neighbor margin in the input.
#    - The original loop used each object's own c0 independently. After fusion,
#      legality depends on cumulative packed width per side, so the generator now
#      guarantees every side's packed width fits in W.
#    - Sorting by row inside a group needs deterministic tie-breaking on arbitrary
#      valid inputs. The solver uses `(r0, c0, idx)` inside groups and
#      `(min_c0, min_r0, min_idx)` across groups.
#    - Group contiguity is a new solver convention. The generator therefore uses
#      explicit group-specific column bands so the intended group order is stable
#      and readable in examples.
#
# 4) Placement legality
#    - Output legality: no overlap occurs because packing uses bbox widths and
#      the generator rejects attempts whose side-wise packed width exceeds W.
#    - Input legality: objects are placed with `_can_place_with_margin`, keeping
#      every component isolated under 8-connectivity.
#    - The generator computes the exact expected combo output from placed input
#      metadata and checks `main(inp) == out`.
#
# 5) Visual observability / cleanliness / inferability
#    - Humans can see three cues in a few examples:
#         a. duplicated shapes go upward, unique shapes go downward;
#         b. same-side gaps disappear;
#         c. duplicated copies of one shape appear as left-to-right blocks ordered
#            by their original top-to-bottom positions, not by their columns.
#    - No hidden feature like numeric color id is used in any ordering rule.
#    - Unique groups are singletons, so only duplicated groups demonstrate the
#      row-order effect; the generator guarantees such a witness exists.
#
# 6) New assumptions introduced by the combo
#    - Assumption: there is enough horizontal room for packed sides.
#      Guarantee: `_assign_side_columns` rejects wider layouts.
#    - Assumption: at least one duplicated group makes row-order matter.
#      Guarantee: `_gen_attempt` rejects cases where combo output equals D4-only.
#    - Assumption: horizontal compaction is observable.
#      Guarantee: `_gen_attempt` rejects cases where combo output equals the
#      original keep-c0 output.


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


def _extract_entries(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    entries = []
    for idx, comp in enumerate(comps):
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, comp["bbox"])
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        entries.append(
            {
                "idx": int(idx),
                "r0": int(r0),
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return entries, bg


def _build_side_groups(entries):
    freq = Counter(d["shape_key"] for d in entries)
    groups_by_shape = {}
    for d in entries:
        groups_by_shape.setdefault(d["shape_key"], []).append(d)

    top_groups = []
    bottom_groups = []
    for shape_key, members in groups_by_shape.items():
        ordered = sorted(members, key=lambda d: (int(d["r0"]), int(d["c0"]), int(d["idx"])))
        group = {
            "shape_key": shape_key,
            "members": ordered,
            "min_c0": min(int(d["c0"]) for d in members),
            "min_r0": min(int(d["r0"]) for d in members),
            "min_idx": min(int(d["idx"]) for d in members),
        }
        if int(freq[shape_key]) >= 2:
            top_groups.append(group)
        else:
            bottom_groups.append(group)

    key_fn = lambda g: (int(g["min_c0"]), int(g["min_r0"]), int(g["min_idx"]))
    top_groups.sort(key=key_fn)
    bottom_groups.sort(key=key_fn)
    return top_groups, bottom_groups


def _pack_groups(groups, *, top_side, H, W, bg):
    out = np.full((H, W), int(bg), dtype=int)
    cursor = 0
    for group in groups:
        for entry in group["members"]:
            pat = entry["pat"]
            h, w = map(int, pat.shape)
            if cursor + w > W:
                raise ValueError("packed side wider than grid")
            dest_r0 = 0 if top_side else int(H - h)
            paste(out, pat, dest_r0, int(cursor), transparent=int(bg))
            cursor += int(w)
    return out


def _render_combo_output(entries, H, W, bg):
    top_groups, bottom_groups = _build_side_groups(entries)
    out = np.full((H, W), int(bg), dtype=int)
    top = _pack_groups(top_groups, top_side=True, H=H, W=W, bg=bg)
    bottom = _pack_groups(bottom_groups, top_side=False, H=H, W=W, bg=bg)
    out[top != bg] = top[top != bg]
    out[bottom != bg] = bottom[bottom != bg]
    return out


def _render_original_output(entries, H, W, bg):
    freq = Counter(d["shape_key"] for d in entries)
    out = np.full((H, W), int(bg), dtype=int)
    for entry in entries:
        pat = entry["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[entry["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(entry["c0"]), transparent=int(bg))
    return out


def _render_d4_only_output(entries, H, W, bg):
    freq = Counter(d["shape_key"] for d in entries)
    top_entries = [d for d in entries if int(freq[d["shape_key"]]) >= 2]
    bottom_entries = [d for d in entries if int(freq[d["shape_key"]]) == 1]
    out = np.full((H, W), int(bg), dtype=int)

    cursor = 0
    for entry in sorted(top_entries, key=lambda d: (int(d["c0"]), int(d["idx"]))):
        pat = entry["pat"]
        h, w = map(int, pat.shape)
        paste(out, pat, 0, int(cursor), transparent=int(bg))
        cursor += int(w)

    cursor = 0
    for entry in sorted(bottom_entries, key=lambda d: (int(d["c0"]), int(d["idx"]))):
        pat = entry["pat"]
        h, w = map(int, pat.shape)
        paste(out, pat, int(H - h), int(cursor), transparent=int(bg))
        cursor += int(w)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected monochrome objects by D4-canonical shape.

    - Shared shapes  (freq >= 2) move to the top edge.
    - Unique shapes  (freq == 1) move to the bottom edge.
    - On each side, canonical groups are ordered by the group's minimum input c0.
    - Inside each group, members are ordered by input row r0 from top to bottom.
    - The resulting sequence is packed tightly from the left with no gaps.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    entries, bg = _extract_entries(g)
    return _render_combo_output(entries, H, W, bg)


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
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
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(int(r - mr), int(c - mc)) for r, c in cells]


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(int(r - mr), int(c - mc)) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    p = np.full((mr + 1, mc + 1), int(bg), dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _can_place_with_margin(grid, patch, r0, c0, bg):
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    H, W = grid.shape
    for pr, pc in np.argwhere(patch != bg):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        r1 = max(0, rr - 1)
        r2 = min(H, rr + 2)
        c1 = max(0, cc - 1)
        c2 = min(W, cc + 2)
        if np.any(grid[r1:r2, c1:c2] != bg):
            return False
    return True


def _plan_group_members(group, *, witness=False):
    ordered_members = random.sample(group["members"], len(group["members"]))
    widths = [int(member["patch"].shape[1]) for member in ordered_members]
    if len(ordered_members) == 1:
        offsets = [0]
    else:
        pool_size = max(len(ordered_members) + 1, 4)
        pool = list(range(pool_size))
        sample = sorted(random.sample(pool, len(ordered_members)))
        if witness:
            offsets = list(reversed(sample))
        else:
            offsets = random.sample(sample, len(sample))
    span = max(int(off + w) for off, w in zip(offsets, widths))
    group["ordered_members"] = ordered_members
    group["local_offsets"] = offsets
    group["span"] = int(span)


def _assign_side_columns(groups, W):
    if not groups:
        return
    spans = [int(group["span"]) for group in groups]
    min_gap = 2
    need = sum(spans) + min_gap * max(0, len(groups) - 1)
    if need > W:
        raise ValueError("side bands wider than grid")

    extra = int(W - need)
    leading = random.randint(0, extra) if extra > 0 else 0
    extra -= int(leading)
    gap_extra = [0] * max(0, len(groups) - 1)
    for _ in range(extra):
        if gap_extra:
            gap_extra[random.randrange(len(gap_extra))] += 1
        else:
            leading += 1

    cursor = int(leading)
    for i, group in enumerate(groups):
        group["anchor"] = int(cursor)
        for member, off in zip(group["ordered_members"], group["local_offsets"]):
            member["c0"] = int(cursor + off)
        cursor += int(group["span"])
        if i + 1 < len(groups):
            cursor += int(min_gap + gap_extra[i])


def _place_group(inp, group, H, bg):
    prev_r0 = -1
    for member in group["ordered_members"]:
        patch = member["patch"]
        h, _w = map(int, patch.shape)
        lower = int(prev_r0 + 1)
        if lower > H - h:
            raise ValueError("row order cannot fit")
        rows = list(range(lower, H - h + 1))
        target_row = 0 if group["shared"] else int(H - h)
        rows = [r for r in rows if r != target_row] or rows
        if group["shared"]:
            candidates = rows
        else:
            candidates = list(reversed(rows))
        placed = False
        for r0 in candidates:
            if _can_place_with_margin(inp, patch, int(r0), int(member["c0"]), bg):
                paste(inp, patch, int(r0), int(member["c0"]), transparent=int(bg))
                member["r0"] = int(r0)
                prev_r0 = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("group placement failed")


def _collect_entries_from_groups(groups):
    entries = []
    idx = 0
    for group in groups:
        for member in group["members"]:
            entries.append(
                {
                    "idx": int(idx),
                    "r0": int(member["r0"]),
                    "c0": int(member["c0"]),
                    "pat": member["patch"],
                    "shape_key": group["shape_key"],
                }
            )
            idx += 1
    return entries


def _row_col_disagreement(entries):
    groups_by_shape = {}
    for entry in entries:
        groups_by_shape.setdefault(entry["shape_key"], []).append(entry)
    count = 0
    for members in groups_by_shape.values():
        if len(members) < 2:
            continue
        by_row = [d["idx"] for d in sorted(members, key=lambda d: (int(d["r0"]), int(d["c0"]), int(d["idx"])))]
        by_col = [d["idx"] for d in sorted(members, key=lambda d: (int(d["c0"]), int(d["r0"]), int(d["idx"])))]
        if by_row != by_col:
            count += 1
    return count


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
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)
    groups = []
    canon_used: set = set()

    for gsz in group_sizes:
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        members = []
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
            patch = _coords_to_patch(transformed, avail[ci], bg)
            members.append({"patch": patch, "color": int(avail[ci])})
            ci += 1
        groups.append({"shape_key": canon, "shared": True, "members": members})

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon in canon_used:
                continue
            canon_used.add(canon)
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
            patch = _coords_to_patch(transformed, avail[ci], bg)
            groups.append(
                {
                    "shape_key": canon,
                    "shared": False,
                    "members": [{"patch": patch, "color": int(avail[ci])}],
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    shared_groups = [group for group in groups if group["shared"]]
    unique_groups = [group for group in groups if not group["shared"]]
    random.shuffle(shared_groups)
    random.shuffle(unique_groups)

    witness_group = random.choice(shared_groups)
    for group in shared_groups:
        _plan_group_members(group, witness=(group is witness_group))
    for group in unique_groups:
        _plan_group_members(group, witness=False)

    _assign_side_columns(shared_groups, W)
    _assign_side_columns(unique_groups, W)

    inp = np.full((H, W), int(bg), dtype=int)
    for group in shared_groups:
        _place_group(inp, group, H, bg)
    for group in unique_groups:
        _place_group(inp, group, H, bg)

    entries = _collect_entries_from_groups(groups)
    if _row_col_disagreement(entries) == 0:
        raise ValueError("no row-vs-column disagreement witness")

    combo_out = _render_combo_output(entries, H, W, bg)
    original_out = _render_original_output(entries, H, W, bg)
    d4_out = _render_d4_only_output(entries, H, W, bg)
    if np.array_equal(combo_out, original_out):
        raise ValueError("combo degenerated to original behavior")
    if np.array_equal(combo_out, d4_out):
        raise ValueError("combo degenerated to D4-only behavior")

    result = main(inp)
    assert np.array_equal(result, combo_out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D4/0+D11/1")
