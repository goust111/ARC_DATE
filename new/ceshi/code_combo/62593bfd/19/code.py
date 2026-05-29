# Combo variant of 62593bfd (index 19)
# combo_spec: D2/0+D5/4+D14/1
# Name: 方向反转 + unique擦除 + 噪声像素

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

# Design analysis for combo D2/0 + D5/4 + D14/1
# ------------------------------------------------
# a) Concrete behavior and trace
# - Seed task:
#   1. Treat every non-background 8-connected component as a real object.
#   2. Group objects by D4-canonical shape.
#   3. Shared groups (freq >= 2) move to the top edge; unique groups move to the bottom edge.
# - Combo task:
#   1. First ignore area-1 isolated noise pixels by filtering to components of area >= 3.
#   2. Group only those real objects by D4-canonical shape.
#   3. Erase every unique-shape object completely.
#   4. Reverse the original direction for the surviving category: duplicated shapes move to the
#      bottom edge, preserving each object's left bbox column c0.
# - Small trace on H=20:
#   real objects = {A(area 5), A(area 5), B(area 4)}, noise = {n1(area 1), n2(area 1)}.
#   Seed output: A,A -> row 0; B -> row 16; the two noise pixels would also be treated as a
#   duplicated singleton shape and move to row 0.
#   Combo output: filter removes n1,n2; B is unique so it disappears; only the two A objects
#   remain, both translated to row 15/16 depending on their heights.
#   Net difference: top band becomes empty, bottom band contains only repeated large objects,
#   and all tiny dots vanish.
#
# b) Complexity calibration
# - We keep the original size regime for true objects: 4-6 components, each area 3-12, bbox <= 5.
# - Every sample contains at least one shared-shape family and at least one unique shape, so the
#   erase branch is always visible.
# - Noise count is kept small (2-5) and each noise component is exactly one pixel, which makes the
#   "ignore small isolated dots" rule legible without cluttering the scene.
# - Colors stay object-local and are not used in grouping. Distinct object colors help humans track
#   which larger shapes survive, while noise colors may repeat because the rule is geometric.
#
# c) Hidden-assumption audit of main()
# - `find_components(..., monochromatic=False, connectivity=8)` assumes foreground pieces do not
#   8-touch, or different objects would merge into one multicolor component. The generator must
#   enforce a one-cell moat between every pair of true objects and between objects and noise.
# - `min_area=3` is now essential: without it, singleton noise would be counted and would often
#   form a fake duplicated shape class.
# - `_canonical_shape(rel)` still only depends on geometry, so it remains valid after the noise
#   filter and after the erase rule.
# - The final loop no longer has a unique branch that pastes to an edge. Skipping freq==1 objects
#   is safe only because the intended output truly omits them.
# - `paste(out, pat, H-h, c0, ...)` assumes surviving duplicated objects do not overlap at the
#   bottom edge. The generator must reserve legal bottom placements output-first.
#
# d) Placement legality
# - Output-first generation places only shared objects on the bottom edge with preserved c0 and a
#   no-touch constraint.
# - Input then re-scatters shared objects into interior rows while keeping the same c0, so the
#   solver must move them back down.
# - Unique objects are placed only in the input, also with no-touch spacing; they may use arbitrary
#   interior c0 because they vanish in the output.
# - Noise pixels are added last, only in background cells whose 8-neighborhood is entirely blank,
#   guaranteeing they remain isolated area-1 distractors.
#
# e) Visual observability
# - Across 2-4 examples, a human sees three consistent cues:
#   1. tiny isolated dots disappear,
#   2. some larger one-off objects also disappear,
#   3. the repeated larger shapes are the only survivors and they all end up on the bottom edge.
# - No hidden property such as numeric color id, object creation order, or parity is used.
#
# f) Visual cleanliness
# - The output has a single, clean bottom band of survivors and otherwise background.
# - Input objects are kept off the exact top/bottom target rows when possible, so the movement and
#   deletion are visually active rather than already solved.
#
# g) Quantitative anti-degeneracy checks
# - The generator rejects any sample where combo output matches:
#   1. the seed solver,
#   2. the "reverse direction only" solver,
#   3. the "ignore noise + erase unique but keep shared at top" solver.
#   This ensures the noise filter, erase rule, and direction reversal all matter on generated data.
#
# h) New assumptions introduced by the combo
# - `main()` now assumes there is at least one real component of area >= 3; the generator always
#   creates multiple such objects before adding noise.
# - The bottom-only output assumes at least one duplicated family exists; `_CONFIGS` preserves that.
# - The erase rule is only learnable if at least one unique object exists; `_CONFIGS` also enforces
#   that in every sample.


MIN_OBJECT_AREA = 3
NOISE_COUNT_RANGE = (2, 5)


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


def _extract_objects(grid: np.ndarray, *, min_area: int):
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        min_area=int(min_area),
        return_format="dict",
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, _r1, _c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, tuple(map(int, comp["bbox"])))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
                "area": int(comp["area"]),
            }
        )
    return g, bg, extracted


def _solve_variant(
    grid: np.ndarray,
    *,
    min_area: int,
    shared_to_bottom: bool,
    erase_unique: bool,
) -> np.ndarray:
    g, bg, extracted = _extract_objects(grid, min_area=min_area)
    H, W = g.shape
    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        is_shared = int(freq[d["shape_key"]]) >= 2
        if not is_shared and erase_unique:
            continue
        pat = d["pat"]
        h, _w = pat.shape
        if is_shared:
            dest_r0 = int(H - h) if shared_to_bottom else 0
        else:
            dest_r0 = 0 if shared_to_bottom else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Ignore area-1/2 noise components; only area >= 3 components count as real objects.
    - Group real objects by D4-canonical shape.
    - Unique shapes are erased completely.
    - Duplicated shapes (freq >= 2) are translated to the bottom edge, preserving c0.
    """
    return _solve_variant(
        grid,
        min_area=MIN_OBJECT_AREA,
        shared_to_bottom=True,
        erase_unique=True,
    )


def _seed_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(
        grid,
        min_area=1,
        shared_to_bottom=False,
        erase_unique=False,
    )


def _reverse_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(
        grid,
        min_area=MIN_OBJECT_AREA,
        shared_to_bottom=True,
        erase_unique=False,
    )


def _erase_top_main(grid: np.ndarray) -> np.ndarray:
    return _solve_variant(
        grid,
        min_area=MIN_OBJECT_AREA,
        shared_to_bottom=False,
        erase_unique=True,
    )


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly area pixels within a max_dim bbox."""
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


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _c in coords)
    mc = max(c for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor contact with existing foreground."""
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


def _is_isolated_noise_slot(grid, r, c, bg):
    H, W = grid.shape
    rr = int(r)
    cc = int(c)
    if not (1 <= rr < H - 1):
        return False
    if grid[rr, cc] != bg:
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] != bg:
                return False
    return True


def _add_noise_pixels(grid, bg, palette):
    H, W = grid.shape
    count = random.randint(*NOISE_COUNT_RANGE)
    noise = []
    for _ in range(count):
        candidates = [
            (r, c)
            for r in range(1, H - 1)
            for c in range(W)
            if _is_isolated_noise_slot(grid, r, c, bg)
        ]
        if not candidates:
            if len(noise) >= NOISE_COUNT_RANGE[0]:
                break
            raise ValueError("not enough isolated slots for noise")
        r, c = random.choice(candidates)
        grid[int(r), int(c)] = int(random.choice(palette))
        noise.append((int(r), int(c)))
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("too little noise")
    return noise


def _count_singleton_components(grid: np.ndarray) -> int:
    bg = int(get_bg(grid))
    comps = find_components(
        np.array(grid, dtype=int),
        bg=bg,
        connectivity=8,
        monochromatic=False,
        min_area=1,
        return_format="dict",
    )
    return sum(1 for comp in comps if int(comp["area"]) == 1)


def _real_component_stats(grid: np.ndarray):
    _g, _bg, extracted = _extract_objects(grid, min_area=MIN_OBJECT_AREA)
    freq = Counter(d["shape_key"] for d in extracted)
    shared = sum(1 for d in extracted if int(freq[d["shape_key"]]) >= 2)
    unique = sum(1 for d in extracted if int(freq[d["shape_key"]]) == 1)
    return len(extracted), shared, unique


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
        area = random.randint(MIN_OBJECT_AREA, 12)
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
            area = random.randint(MIN_OBJECT_AREA, 12)
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

    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]
    assert shared_objs and unique_objs

    for obj in objs:
        obj["patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)

    out = np.full((H, W), bg, dtype=int)
    for obj in shared_objs:
        patch = obj["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            if _no_touch(out, patch, int(H - h), c0, bg):
                paste(out, patch, int(H - h), c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("shared output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in shared_objs:
        patch = obj["patch"]
        h, _w = patch.shape
        rows = _interior_rows(H, h)
        random.shuffle(rows)
        placed = False
        for r0 in rows:
            if _no_touch(inp, patch, int(r0), int(obj["c0"]), bg):
                paste(inp, patch, int(r0), int(obj["c0"]), transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("shared input scatter failed")

    for obj in unique_objs:
        patch = obj["patch"]
        h, w = patch.shape
        rows = _interior_rows(H, h)
        placed = False
        for _ in range(600):
            r0 = random.choice(rows)
            c0 = random.randint(0, W - w)
            if _no_touch(inp, patch, int(r0), c0, bg):
                paste(inp, patch, int(r0), c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("unique input scatter failed")

    noise = _add_noise_pixels(inp, bg, avail)
    if len(noise) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise missing")
    if _count_singleton_components(inp) < NOISE_COUNT_RANGE[0]:
        raise ValueError("noise merged unexpectedly")

    n_real, n_shared, n_unique_real = _real_component_stats(inp)
    if n_real != len(objs):
        raise ValueError("real object count changed")
    if n_shared < 2 or n_unique_real < 1:
        raise ValueError("combo categories not visible")

    result = main(inp)
    if not np.array_equal(result, out):
        raise AssertionError("round-trip check failed")
    if np.array_equal(_seed_main(inp), out):
        raise AssertionError("degenerated to seed behavior")
    if np.array_equal(_reverse_only_main(inp), out):
        raise AssertionError("unique erase not observable")
    if np.array_equal(_erase_top_main(inp), out):
        raise AssertionError("direction reversal not observable")

    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo 19")
