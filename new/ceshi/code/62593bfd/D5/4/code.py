# Variant of 62593bfd
# Dimension: D5 — 对象变换——移动过程中的形状/颜色变化 (transform)
# Change: 对象在平移过程中不做任何变换：形状、颜色、朝向完全保持原样（duplicated-size移顶部，singleton-s... → unique对象消失（擦除）

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste

# Design analysis for this D5 variant
# -----------------------------------
# spec_variant:
#   core_behavior:
#     input still contains 4-6 separated monochromatic 8-connected objects.
#     Objects are grouped by D4-canonical shape exactly as in 62593bfd.
#     Shared shapes (freq >= 2) still translate vertically to row 0 with c0 preserved.
#     Unique shapes (freq == 1) are no longer moved to the bottom; they are erased.
#   concrete_trace:
#     Example: if a blue L-shape at c0=2 shares its canonical shape with another object,
#     and a red T-shape at c0=11 is unique, then:
#       original output: blue appears at row 0, red appears at row H-h_red.
#       variant output: blue appears at row 0, red disappears entirely.
#     The changed step is the final placement decision after counting canonical-shape
#     frequencies. The extraction / grouping logic remains unchanged.
#   complexity_calibration:
#     Keep the original object counts, sizes, D4 variation, and distinct colors per object.
#     Require at least one duplicated group and at least one unique object in every sample.
#     This makes the erase branch always observable and avoids degeneration to the original task.
#   hidden_assumptions_review:
#     - main() still uses find_components(..., monochromatic=False), so different-colored
#       objects must not 8-touch. In the original code that was only probabilistically true.
#       The generator therefore enforces a one-cell 8-neighborhood moat between objects.
#     - The extracted patch and preserved c0 remain valid because shared objects are pasted
#       unchanged; only the unique branch is skipped.
#     - Paste order on the top row remains harmless because generator guarantees retained
#       objects do not overlap or touch there.
#   placement_legality:
#     Output is constructed only from shared objects, all on row 0, with no overlap/touch.
#     Input then scatters every object into interior rows; shared objects keep the same c0
#     they will use in output, while unique objects may use any c0 because they vanish.
#   visual_observability:
#     Every sample contains at least one unique object with its own color. In output those
#     colors disappear completely instead of reappearing at the bottom, so the erase rule is
#     visible from a few examples without relying on hidden numeric color identities.
#   cleanliness:
#     Interior scatter avoids row 0 and the exact bottom alignment when possible, making the
#     retained top movement and the unique-object disappearance easier to see.
#   color_change_inference:
#     No color mapping is introduced. Colors are preserved for retained objects; erased
#     objects revert to background only because they are removed.


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
    Group objects by D4-canonical shape.

    - Shared shapes  (canonical shape freq >= 2) -> translate to TOP (row 0).
    - Unique shapes  (canonical shape freq == 1) -> erase completely.
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
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
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
        paste(out, d["pat"], 0, int(d["c0"]), transparent=bg)

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


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor contact with already placed objects."""
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

    out = np.full((H, W), bg, dtype=int)
    shared_objs = [obj for obj in objs if obj["shared"]]
    unique_objs = [obj for obj in objs if not obj["shared"]]
    assert shared_objs and unique_objs

    for obj in shared_objs:
        patch = obj["patch"]
        _, w = patch.shape
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            if _no_touch(out, patch, 0, c0, bg):
                paste(out, patch, 0, c0, transparent=bg)
                obj["c0"] = int(c0)
                placed = True
                break
        if not placed:
            raise ValueError("shared output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in shared_objs:
        patch = obj["patch"]
        h, _ = patch.shape
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

    for obj in unique_objs:
        patch = obj["patch"]
        h, w = patch.shape
        rows = _interior_rows(H, h)
        placed = False
        for _ in range(500):
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
    raise RuntimeError("failed to generate input for 62593bfd D5 erase variant")
