# Combo variant of 62593bfd (index 2)
# combo_spec: D1/1+D2/0
# Name: 颜色分组 + 方向反转
#
# Design analysis
# ===============
# a) Concrete behavior and trace
# - Original task:
#   find each 8-connected object, classify it by D4-canonical shape, move repeated
#   shapes to the top edge and unique shapes to the bottom edge, preserving `c0`.
# - Fusion variant:
#   find each monochrome 8-connected object, classify it only by color frequency,
#   then reverse the original direction mapping:
#   duplicated color -> bottom edge, unique color -> top edge.
# - Small trace:
#   suppose H=20 and the input contains:
#   red L at c=2, red zigzag at c=10, blue T at c=16, green hook at c=22.
#   All four shapes are different.
#   Original task:
#     every shape is unique, so all four objects would move to row H-h.
#   D1-only color grouping:
#     both red objects would move to row 0; blue and green would move to row H-h.
#   Fusion here:
#     red is the only duplicated color, so both red objects move to row H-h;
#     blue and green are unique colors, so they move to row 0.
# - Visible difference:
#   the solver must infer both that color, not shape, defines the grouping and that
#   the repeated group is sent downward rather than upward.
#
# b) Complexity calibration
# - If repeated-color objects also shared shapes, the sample could accidentally admit
#   a shape-based explanation. To keep the fusion behavior visible, the generator
#   makes every object's D4-canonical shape globally unique.
# - At least one duplicated color group and at least one unique color are always
#   present, so both halves of the top/bottom split appear in every sample.
# - Repeated-color groups are kept small (size 2 or 3, optionally two groups) so the
#   visual evidence repeats without overcrowding the grid.
#
# c) Hidden-assumption audit of main()
# - `find_components(... monochromatic=False)` from the original task is invalid once
#   color equality matters. The fusion solver must use `monochromatic=True`, otherwise
#   same-color disconnected objects are not counted as separate color-group members.
# - The original `shape_key = _canonical_shape(rel)` no longer drives grouping inside
#   `main()`. Group membership is now `comp["color"]`.
# - `dest_r0` must flip direction relative to the original task:
#   duplicated color -> `H-h`, unique color -> `0`.
# - `paste(...)` still assumes the destination placements do not overlap. The generator
#   preserves that by constructing outputs first and validating each placement.
#
# d) Placement legality
# - Output-first generation is retained.
# - Objects are kept pairwise 8-disconnected in both output and input. This matters
#   more than in the original task because same-color objects must remain separate
#   components; if they touch even diagonally they merge and color counts collapse.
# - Top/bottom destinations remain safe because object bbox heights are at most 5 while
#   H is at least 19, so top and bottom bands cannot collide vertically.
#
# e) Visual observability
# - Humans can directly see repeated colors. By forcing globally unique shapes, the
#   stable explanation across examples is "same colors travel together".
# - The direction inversion is also visible because those same-color groups end up on
#   the bottom edge, contradicting the original task's repeated->top intuition.
# - No hidden ordering, numeric color ranking, or object-index rule is used.
#
# f) Visual neatness
# - Object count stays within 4-5, bbox sizes stay within the original 5x5 cap, and
#   all placements enforce at least one background cell of 8-neighborhood separation.
# - This keeps repeated-color objects visually distinct despite sharing a color.
#
# g) Change inferability
# - There is no recoloring. Colors are preserved exactly.
# - The only color-based inference is equality of color across disconnected objects,
#   and each sample contains both duplicated-color and unique-color evidence.
#
# h) Quantitative anti-degeneracy check
# - Because every object's canonical shape is unique, the original shape-based solver
#   would place every object in one band, while this fusion splits objects by color.
# - Because duplicated colors are sent to the bottom here, this fusion also differs
#   from the non-inverted color-grouping variant on every valid sample.
# - After implementation, we verify over many generated inputs that outputs differ from
#   both baselines and that both top and bottom categories appear.
#
# i) New assumptions introduced by the fusion
# - `main()` now assumes every foreground component is monochrome; the generator builds
#   each object as a single-color patch.
# - `main()` assumes at least one duplicated color and at least one unique color if the
#   task is to remain inferable; `_CONFIGS` enforces this.
# - `main()` assumes same-color objects are still disconnected components; the generator
#   enforces global 8-neighborhood separation in both output and input.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort objects to top or bottom edge by duplicated-vs-unique color, with the
    repeated-color group sent downward.

    Rule:
    - Find monochrome 8-connected components.
    - Count how many disconnected components use each color.
    - Duplicated colors (freq >= 2) move to the BOTTOM edge, row H-h.
    - Unique colors (freq == 1) move to the TOP edge, row 0.
    - Horizontal position (left bbox column c0) is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=True, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "color": int(comp["color"]),
            }
        )

    freq = Counter(d["color"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = int(H - h) if int(freq[d["color"]]) >= 2 else 0
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly `area` pixels."""
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
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[r, c] = color
    return p


def _no_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    mask = patch != bg
    return not np.any(grid[r0 : r0 + h, c0 : c0 + w][mask] != bg)


def _component_separated(grid, patch, r0, c0, bg):
    """True iff the placed patch stays 8-disconnected from existing objects."""
    if not _no_overlap(grid, patch, r0, c0, bg):
        return False
    H, W = grid.shape
    nz = np.argwhere(patch != bg)
    for pr, pc in nz:
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) != bg:
                    return False
    return True


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([2, 2], 1),
]


def _gen_unique_shape(canon_used):
    for _ in range(400):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        canon = _canonical_shape(shape)
        if canon not in canon_used:
            canon_used.add(canon)
            return shape
    raise ValueError("unique shape not found")


def _object_area(obj, bg):
    return int(np.count_nonzero(obj["patch"] != bg))


def _place_objects_on_rows(grid, objs, bg, row_getter):
    ordered = sorted(objs, key=lambda obj: (-_object_area(obj, bg), -obj["patch"].shape[1]))
    for obj in ordered:
        patch = obj["patch"]
        _h, w = patch.shape
        dest_r0 = int(row_getter(obj))
        placed = False
        for _ in range(500):
            c0 = random.randint(0, grid.shape[1] - w)
            if _component_separated(grid, patch, dest_r0, c0, bg):
                paste(grid, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                placed = True
                break
        if not placed:
            raise ValueError("row placement failed")


def _scatter_input(inp, objs, bg):
    ordered = sorted(objs, key=lambda obj: (-obj["patch"].shape[0], -_object_area(obj, bg)))
    for obj in ordered:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        candidate_rows = list(range(1, inp.shape[0] - h))
        if not candidate_rows:
            raise ValueError("no interior rows available")
        placed = False
        for _ in range(500):
            r0 = random.choice(candidate_rows)
            if _component_separated(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)
    ci = 0

    group_sizes, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used = set()

    for gsz in group_sizes:
        color = avail[ci]
        ci += 1
        for _ in range(gsz):
            coords = _gen_unique_shape(canon_used)
            objs.append({"coords": coords, "color": color, "duplicated_color": True})

    for _ in range(n_unique):
        color = avail[ci]
        ci += 1
        coords = _gen_unique_shape(canon_used)
        objs.append({"coords": coords, "color": color, "duplicated_color": False})

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)
    _place_objects_on_rows(
        out,
        objs,
        bg,
        row_getter=lambda obj: H - obj["patch"].shape[0] if obj["duplicated_color"] else 0,
    )

    inp = np.full((H, W), bg, dtype=int)
    _scatter_input(inp, objs, bg)

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")
