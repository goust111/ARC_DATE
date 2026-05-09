# Combo variant of 62593bfd (index 18)
# combo_spec: D1/1+D4/2+D5/1
# Name: 颜色分组 + 水平镜像 + unique重着色
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Original task:
#   extract 8-connected objects, classify them by D4-canonical shape, move
#   duplicated shapes to row 0 and unique shapes to row H-h, keep c0 unchanged,
#   and preserve colors.
# - Combo variant:
#   extract monochrome 8-connected objects, classify them by visible color
#   equality, move repeated colors to row 0 and singleton colors to row H-h,
#   mirror the destination column with dest_c0 = W - c0 - w, and recolor every
#   bottom/unique object to one common color.
# - Small trace:
#   with W=20, suppose a red object of width 3 starts at c0=2, another red
#   object of width 4 starts at c0=9, and a blue object of width 2 starts at
#   c0=15. Original logic might send all three to the bottom if their shapes are
#   all unique, leaving columns 2/9/15 unchanged. The combo instead counts
#   colors: the two red objects are duplicated and go to the top at mirrored
#   columns 15 and 7; the blue object is unique and goes to the bottom at
#   mirrored column 3 and is recolored to the shared unique color.
# - The visible difference is therefore threefold at once:
#   color equality controls top vs bottom, left/right destination is mirrored,
#   and only the bottom singleton-color objects lose their original colors.
#
# b) Complexity calibration
# - If repeated-color objects also shared shapes, a human could wrongly explain
#   the examples with the original shape-based rule. To avoid that, the
#   generator makes every object's D4-canonical shape globally unique.
# - To keep the recolor inferable, every sample contains at least two unique
#   color objects, so the common bottom recolor repeats inside one grid.
# - To keep the mirror visible, the generator forbids fixed-point placements
#   where c0 == W - c0 - w.
# - Repeated-color groups are limited to 1-2 groups of size 2-3, keeping object
#   count in the original 4-6 range and making the color grouping readable.
#
# c) Implicit assumption audit for main()
# - `find_components(... monochromatic=False)` from the original solver is no
#   longer valid because same-color disconnected objects must stay separate
#   counts. The combo must use `monochromatic=True`.
# - Grouping by `_canonical_shape(rel)` is replaced with grouping by
#   `comp["color"]`.
# - Destination placement can no longer use only `c0`; mirrored placement also
#   needs the bbox width `w`.
# - The old "paste patch unchanged" assumption no longer holds for unique
#   objects because they must be recolored before pasting.
# - These changes do not conflict: color grouping is computed from the input
#   component colors before any recolor happens, while mirror placement depends
#   only on geometry.
#
# d) Placement legality
# - Output-first generation is retained.
# - Objects are kept pairwise 8-disconnected in both input and output. This is
#   required twice: same-color objects must not merge in the input, and bottom
#   unique objects must not merge after they all share one output color.
# - Output placement chooses an input column c0, computes the mirrored output
#   column, and validates in-bounds non-contact placement there.
# - Input scattering preserves that chosen c0 and only changes the row to an
#   interior value, again checking 8-disconnection.
#
# e) Visual observability
# - Humans can directly see repeated colors, mirrored left-right movement, and
#   the shared recolor of bottom objects. No hidden ordering on color ids is
#   used; only visible equality and position matter.
# - Because shapes are all different, "same colors move together" is the only
#   stable explanation for the top group.
#
# f) Visual cleanliness
# - Object count remains 4-6, bbox sizes stay within 5x5, and a one-cell
#   8-neighborhood moat is enforced between all objects.
# - This prevents ambiguous touching clusters, especially at the bottom where
#   multiple recolored unique objects would otherwise merge visually.
#
# g) Color-change inferability
# - The recolor target is deterministic: `(bg + 1) % 10`.
# - The generator excludes that target color from all input objects, so the
#   recolor is always a genuine visible change rather than an accidental match.
# - The recolor applies only to the bottom group, giving a strong structural cue
#   for when the color change happens.
#
# 9) New main() assumptions and generator guarantees
# - Assumption: every foreground object is monochrome and 8-connected.
#   Guarantee: generator builds each object as one monochrome patch.
# - Assumption: same-color objects are disconnected.
#   Guarantee: generator enforces 8-neighborhood separation in input/output.
# - Assumption: at least one duplicated color group and at least two unique
#   color objects exist.
#   Guarantee: generator samples only from configs with both properties.
# - Assumption: mirrored destinations are in bounds and do not overlap.
#   Guarantee: output-first placement validates the mirrored destination before
#   accepting each object.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical D4-invariant key used only to keep generator shapes unique."""
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


def _mirror_c0(W: int, c0: int, w: int) -> int:
    return int(W - c0 - w)


def _target_unique_color(bg: int) -> int:
    return int((int(bg) + 1) % 10)


def _recolor_patch(patch: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    out[out != int(bg)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group monochrome 8-connected objects by color frequency.

    - Repeated colors  (freq >= 2): move to row 0 unchanged.
    - Unique colors    (freq == 1): move to row H-h, recolored to (bg + 1) % 10.
    - All destination columns are mirrored: dest_c0 = W - c0 - w.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    unique_color = _target_unique_color(bg)

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        return_format="dict",
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, tuple(map(int, comp["bbox"])))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append(
            {
                "c0": int(c0),
                "color": int(comp["color"]),
                "pat": pat,
            }
        )

    freq = Counter(d["color"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        is_dup = int(freq[d["color"]]) >= 2
        pat = d["pat"] if is_dup else _recolor_patch(d["pat"], bg, unique_color)
        h, w = map(int, pat.shape)
        dest_r0 = 0 if is_dup else int(H - h)
        dest_c0 = _mirror_c0(W, int(d["c0"]), w)
        paste(out, pat, dest_r0, dest_c0, transparent=int(bg))

    return out


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
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _clearance_ok(grid, patch, r0, c0, bg):
    """Require in-bounds placement with a one-cell 8-neighborhood moat."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for pr, pc in np.argwhere(patch != int(bg)):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for nr in range(max(0, rr - 1), min(H, rr + 2)):
            for nc in range(max(0, cc - 1), min(W, cc + 2)):
                if int(grid[nr, nc]) != int(bg):
                    return False
    return True


def _interior_row(H: int, h: int) -> int | None:
    lo = 1
    hi = H - h - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _gen_unique_shape(canon_used: set) -> list[tuple[int, int]]:
    for _ in range(400):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8, max_dim=5)
        canon = _canonical_shape(shape)
        if canon not in canon_used:
            canon_used.add(canon)
            return shape
    raise ValueError("unique shape not found")


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 2),
    ([3], 3),
    ([2, 2], 2),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    unique_color = _target_unique_color(bg)

    avail = [c for c in range(10) if c not in (bg, unique_color)]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used: set = set()

    for gsz in group_sizes:
        color = int(avail[ci])
        ci += 1
        for _ in range(gsz):
            coords = _gen_unique_shape(canon_used)
            objs.append(
                {
                    "coords": coords,
                    "color": color,
                    "is_dup": True,
                }
            )

    for _ in range(n_unique):
        color = int(avail[ci])
        ci += 1
        coords = _gen_unique_shape(canon_used)
        objs.append(
            {
                "coords": coords,
                "color": color,
                "is_dup": False,
            }
        )

    for obj in objs:
        obj["input_patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
        obj["output_patch"] = (
            obj["input_patch"]
            if obj["is_dup"]
            else _recolor_patch(obj["input_patch"], bg, unique_color)
        )

    out = np.full((H, W), bg, dtype=int)
    placement_order = sorted(
        objs,
        key=lambda obj: (
            0 if obj["is_dup"] else 1,
            -obj["output_patch"].shape[1],
            -obj["output_patch"].shape[0],
        ),
    )
    for obj in placement_order:
        patch = obj["output_patch"]
        h, w = patch.shape
        dest_r0 = 0 if obj["is_dup"] else int(H - h)
        placed = False
        for _ in range(800):
            c0 = random.randint(0, W - w)
            dest_c0 = _mirror_c0(W, c0, w)
            if dest_c0 == c0:
                continue
            if _clearance_ok(out, patch, dest_r0, dest_c0, bg):
                paste(out, patch, dest_r0, dest_c0, transparent=bg)
                obj["c0"] = int(c0)
                obj["dest_c0"] = int(dest_c0)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in sorted(
        objs,
        key=lambda obj: (-obj["input_patch"].shape[1], -obj["input_patch"].shape[0]),
    ):
        patch = obj["input_patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(800):
            r0 = _interior_row(H, h)
            if r0 is None:
                raise ValueError("no interior rows")
            if _clearance_ok(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    if any(int(obj["dest_c0"]) == int(obj["c0"]) for obj in objs):
        raise AssertionError("mirror placement degenerated to original columns")

    dup_count = sum(1 for obj in objs if obj["is_dup"])
    unique_count = len(objs) - dup_count
    if dup_count < 2 or unique_count < 2:
        raise AssertionError("insufficient support for combo behavior")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")
