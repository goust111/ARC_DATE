# Combo variant of 62593bfd (index 10)
# combo_spec: D2/0+D5/4
# Name: 方向反转 + unique擦除

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for combo D2/0 + D5/4
# --------------------------------------
# a) Concrete behavior and trace
# - Original 62593bfd groups 8-connected monochrome objects by D4-canonical shape,
#   keeps c0 fixed, sends shared-shape objects to row 0, and sends unique-shape objects
#   to row H-h.
# - This combo fuses two edits:
#   1. D2/0 inverts the direction mapping: shared -> bottom, unique -> top.
#   2. D5/4 erases unique objects instead of keeping them.
# - When fused, the unique branch still gets classified, but its final action is
#   "erase", so the visible output contains only shared-shape objects and they appear
#   on the bottom edge. Example: with canonical groups A, A, B in H=20, if both A
#   objects have height 4 and B has height 3:
#     original: A at row 0, B at row 17
#     combo:    A at row 16, B absent
#   The grouping step is unchanged; only the post-group action changes.
#
# b) Complexity calibration
# - Keep the original 4-6 object regime, 3-12 cell areas, and D4-equivalence grouping.
# - Every sample must contain at least one duplicated canonical shape and at least one
#   unique shape. Otherwise either the bottom-retained branch or the erase branch would
#   be invisible and the combo could degenerate to a simpler task.
# - Colors stay distinct per object to help a human track which input objects survive,
#   without making color itself part of the rule.
#
# c) Hidden-assumption audit of main()
# - find_components(..., connectivity=8, monochromatic=False) merges any 8-touching
#   non-background pixels even across colors. Therefore the generator must guarantee a
#   one-cell moat around every object both in input and among retained output objects.
# - crop_bbox + rel coords + _canonical_shape depend only on extracted geometry; these
#   remain valid because the combo changes only destination/retention, not recognition.
# - The old branch "unique -> bottom" is no longer valid. In the combo, the unique
#   branch must do nothing at paste time.
# - paste(..., transparent=bg) assumes retained objects do not overlap in output. The
#   generator constructs output first and checks that explicitly.
#
# d) Placement legality
# - Output-first generation is preserved, but output now contains only shared objects,
#   all aligned to row H-h with preserved c0. Unique objects are omitted from output.
# - Shared objects are then re-scattered into interior rows at the same c0. Unique
#   objects are scattered independently into interior rows because they will disappear.
# - `_no_touch` prevents both overlap and 8-neighbor contact, which keeps component
#   extraction stable under monochromatic=False.
#
# e) Visual observability
# - A solver can infer the combo from 2-3 examples because two independent signals are
#   visible at once: repeated-shape families survive and collect on the bottom edge,
#   while one-off shapes disappear entirely instead of appearing on the top or bottom.
# - No hidden factor such as numeric color id, component enumeration order, or bbox area
#   participates in the rule.
#
# f) Visual cleanliness
# - The output has a single active band at the bottom, making the surviving category
#   easy to read. Input objects are scattered away from the border rows when possible,
#   so the vertical move and the erasure are both visually obvious.
# - Touch-free placement prevents accidental fused blobs that would make shape counting
#   hard for humans and incorrect for the solver.
#
# g) Change inferability
# - There is no color remapping. Surviving shared objects preserve color/shape/orientation;
#   unique objects simply vanish into background. Because distinct colors are used per
#   object, a viewer can match repeated geometry across colors and notice which colors
#   disappear only when their shape is unique.
#
# h) Quantitative non-degeneracy check used after implementation
# - For generated samples, compare combo output against:
#   1. the original solver output (shared top, unique bottom)
#   2. the D5 erase-only output (shared top, unique erased)
# - Because every sample has at least one shared object and one unique object, and
#   H >= 19 while object heights <= 5, combo output should differ from both baselines on
#   every valid sample: versus original because unique objects disappear, and versus D5
#   because retained shared objects sit on the bottom instead of the top.
#
# i) New assumptions to re-check after editing
# - main() assumes shared objects can be pasted at row H-h without overlap; generator
#   guarantees this by output-first placement.
# - main() assumes unique objects may be skipped with no side effects; generator keeps at
#   least one shared object so the output is not trivially empty.
# - Generator assumes there is at least one interior row for scattering. With H in
#   [19, 30] and bbox height at most 5, rows 1..H-h-1 always exist.


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
    mask = np.zeros(shape, dtype=bool)
    H, W = shape
    for r, c in coords:
        rr, cc = int(r), int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected objects by D4-canonical shape.

    - Shared shapes (freq >= 2) are retained and translated to the BOTTOM edge.
    - Unique shapes (freq == 1) are erased completely.
    - Retained objects keep their shape, color, orientation, and left column c0.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, comp["bbox"])
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        if int(freq[d["shape_key"]]) < 2:
            continue
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        paste(out, pat, int(H - h), int(d["c0"]), transparent=bg)

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly area pixels within max_dim x max_dim."""
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
                if (
                    max(all_r) - min(all_r) < max_dim
                    and max(all_c) - min(all_c) < max_dim
                ):
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


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor contact with existing non-background pixels."""
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    mask = patch != bg
    if np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg):
        return False

    for pr, pc in np.argwhere(mask):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        rr0 = max(0, rr - 1)
        rr1 = min(H, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(W, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != bg):
            return False
    return True


def _interior_rows(H, h):
    max_r0 = int(H - h)
    rows = list(range(1, max_r0))
    return rows if rows else list(range(0, max_r0 + 1))


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
                base,
                random.randint(0, 3),
                random.choice([False, True]),
            )
            objs.append(
                {
                    "coords": transformed,
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
            if canon not in canon_used:
                canon_used.add(canon)
                objs.append(
                    {
                        "coords": shape,
                        "color": avail[ci],
                        "shared": False,
                    }
                )
                ci += 1
                break
        else:
            raise ValueError("unique shape not found")

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]
    assert shared_objs and unique_objs

    out = np.full((H, W), bg, dtype=int)
    for obj in sorted(
        shared_objs,
        key=lambda d: (-d["patch"].shape[1], -d["patch"].shape[0], -np.count_nonzero(d["patch"] != bg)),
    ):
        patch = obj["patch"]
        h, w = patch.shape
        dest_r0 = int(H - h)
        placed = False
        for _ in range(500):
            c0 = random.randint(0, W - w)
            if _no_touch(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("shared output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in sorted(
        shared_objs,
        key=lambda d: (-d["patch"].shape[1], -d["patch"].shape[0], -np.count_nonzero(d["patch"] != bg)),
    ):
        patch = obj["patch"]
        h, _w = patch.shape
        rows = _interior_rows(H, h)
        random.shuffle(rows)
        placed = False
        for r0 in rows:
            if _no_touch(inp, patch, r0, int(obj["c0"]), bg):
                paste(inp, patch, r0, int(obj["c0"]), transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("shared input scatter failed")

    for obj in sorted(
        unique_objs,
        key=lambda d: (-d["patch"].shape[1], -d["patch"].shape[0], -np.count_nonzero(d["patch"] != bg)),
    ):
        patch = obj["patch"]
        h, w = patch.shape
        rows = _interior_rows(H, h)
        placed = False
        for _ in range(700):
            r0 = random.choice(rows)
            c0 = random.randint(0, W - w)
            if _no_touch(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("unique input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant 10")
