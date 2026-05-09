# Combo variant of 62593bfd (index 24)
# combo_spec: D2/0+D10/0
# Name: 仅平移等价 + 方向反转

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

"""
Design analysis for combo D2/0 + D10/0
--------------------------------------

a) Concrete behavior and trace
- Original task:
  1. Group each 8-connected object by a D4-invariant canonical shape key.
  2. Duplicated groups (freq >= 2) move to the top edge.
  3. Unique groups (freq == 1) move to the bottom edge.
- Combo task:
  1. Group by translation-only exact normalized coordinates.
  2. Duplicated exact shapes move to the bottom edge.
  3. Unique exact shapes move to the top edge.
- Small trace:
  * Let A and A be two exact copies of one shape.
  * Let B and C be rotated copies of one D4 family but not exact copies.
  * Original: A,A,B,C are all "shared" under D4, so all go to the top.
  * Combo: A,A stay shared under exact matching, so both go to the bottom.
           B,C are unique under exact matching, so both go to the top.
  * Visible difference: exact duplicates gather at the bottom, while merely
    rotated/reflected relatives do not join them and remain at the top.

b) Complexity calibration
- If we only reversed direction, the task would be too close to the source.
- If we only changed D4 -> translation-only, then many samples would collapse
  to "almost everything is unique".
- The generator therefore forces two kinds of evidence in every sample:
  1. at least one exact-duplicate group, so the duplicated -> bottom branch is
     visible;
  2. one orientation-trap family with several D4-equivalent but exact-distinct
     shapes, so humans can see that rotation/flip no longer counts as equal.
- Object counts stay within the original 4-6 range and colors stay distinct
  per object to support correspondence without adding a color rule.

c) Hidden-assumption audit of main()
- get_bg(): still valid because background remains the most common color.
- find_components(..., connectivity=8, monochromatic=False):
  assumes intended objects never touch even diagonally, otherwise components
  merge across colors. The generator must forbid 8-neighbor contact.
- crop_bbox(...) + rel coords:
  still valid because each intended object is one isolated component.
- shape_key computation:
  the original D4 canonicalization assumption no longer holds. We replace it
  with translation-only normalization.
- dest_r0 assignment:
  the source assumed shared -> top and unique -> bottom. The combo inverts
  that mapping, so exact duplicates must go to H-h and exact uniques to 0.
- paste(..., transparent=bg):
  still assumes output placements do not overlap. The generator enforces this.

d) Placement legality
- Output-first generation is preserved.
- Objects are placed only after their final combo category is known.
- Placement rejects both overlap and 8-neighbor contact, preventing accidental
  component merges in either input or output.
- The preserved horizontal anchor c0 is safe because the same patch width is
  used in both scattered input and solved output.

e) Visual observability
- Humans can infer the rule from 2-4 examples because samples always include:
  1. bottom objects that are literally identical in orientation;
  2. top objects that look related by rotation/reflection but do not count as
     duplicates anymore.
- No invisible property such as color id, object order, or component area is
  used for grouping or placement.

f) Visual cleanliness
- There is only one layer of content: isolated monochromatic blobs on a single
  dominant background.
- Distinct colors and enforced separation keep object boundaries readable.
- Input scattering avoids solved rows when possible, so motion remains visible.

g) Change inferability
- There is no color remapping.
- The only latent classification is shape equality, and the generator supplies
  repeated exact copies plus repeated orientation variants so that classification
  can be inferred from visible geometry alone.

h) Quantitative anti-degeneracy checks used after implementation
- Compare exact-shared vs D4-shared status per object; every trap object must
  flip status between those two notions.
- Compare combo destinations against a D2-only baseline (D4 grouping with
  reversed direction); trap objects must land on opposite edges.
- Compare combo destinations against a D10-only baseline (exact grouping with
  original direction); all objects must differ because direction is reversed.

i) New assumptions after the edit
- main() now assumes exact duplicates can exist without touching, and that
  rotated/reflected relatives may intentionally be treated as unique.
- generate_input() guarantees at least one exact duplicate group, at least one
  orientation trap family, and full component isolation under 8-connectivity.
"""


def _translation_shape_key(coords):
    """Normalize a shape by translation only."""
    pts = [(int(r), int(c)) for r, c in coords]
    mr = min(r for r, _ in pts)
    mc = min(c for _, c in pts)
    return tuple(sorted((r - mr, c - mc) for r, c in pts))


def _canonical_shape_d4(coords):
    """D4-invariant key used only for generator-side observability checks."""
    pts = [(int(r), int(c)) for r, c in coords]
    best = None
    cur = list(pts)
    for _ in range(4):
        key = _translation_shape_key(cur)
        best = key if best is None or key < best else best
        key = _translation_shape_key([(-r, c) for r, c in cur])
        best = key if key < best else best
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
    Group objects by exact shape equality up to translation only.

    Rule:
    - Shared exact shapes (freq >= 2) move to the BOTTOM edge.
    - Unique exact shapes (freq == 1) move to the TOP edge.
    - Rotations/reflections do not count as equal.
    - Horizontal bbox anchor c0 is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    comps = find_components(
        g, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
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
                "shape_key": _translation_shape_key(rel),
            }
        )

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = int(H - h) if int(freq[d["shape_key"]]) >= 2 else 0
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)

    return out


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly area pixels."""
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
    return list(_translation_shape_key(cells))


def _transform_shape(coords, rot, flip):
    pts = list(coords)
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    return list(_translation_shape_key(pts))


def _distinct_d4_images(coords):
    seen = {}
    for rot in range(4):
        for flip in (False, True):
            transformed = _transform_shape(coords, rot, flip)
            key = _translation_shape_key(transformed)
            if key not in seen:
                seen[key] = transformed
    return list(seen.values())


def _coords_to_patch(coords, color, bg):
    mr = max(r for r, _ in coords)
    mc = max(c for _, c in coords)
    patch = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_contact_or_overlap(grid, patch, r0, c0, bg):
    h, w = patch.shape
    H, W = grid.shape
    r0 = int(r0)
    c0 = int(c0)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    coords = np.argwhere(patch != bg)
    for pr, pc in coords:
        rr = r0 + int(pr)
        cc = c0 + int(pc)
        if grid[rr, cc] != bg:
            return False
        rr0 = max(0, rr - 1)
        rr1 = min(H, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(W, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != bg):
            return False
    return True


def _pick_fresh_shape(exact_used, d4_used):
    for _ in range(300):
        area = random.randint(3, 12)
        shape = _grow_shape(area, 8)
        exact_key = _translation_shape_key(shape)
        d4_key = _canonical_shape_d4(shape)
        if exact_key in exact_used or d4_key in d4_used:
            continue
        return shape, exact_key, d4_key
    raise ValueError("fresh shape not found")


def _pick_orientation_family(size, exact_used, d4_used):
    for _ in range(300):
        area = random.randint(4, 10)
        base = _grow_shape(area, 8)
        d4_key = _canonical_shape_d4(base)
        if d4_key in d4_used:
            continue
        images = _distinct_d4_images(base)
        random.shuffle(images)
        chosen = []
        seen_exact = set()
        for shape in images:
            exact_key = _translation_shape_key(shape)
            if exact_key in exact_used or exact_key in seen_exact:
                continue
            chosen.append(shape)
            seen_exact.add(exact_key)
            if len(chosen) == size:
                return d4_key, chosen
    raise ValueError("orientation trap not found")


def _scatter_rows(H, h, solved_r0):
    interior = [r for r in range(1, H - h) if r != solved_r0]
    if interior:
        random.shuffle(interior)
        return interior
    rows = [r for r in range(0, H - h + 1) if r != solved_r0]
    random.shuffle(rows)
    return rows


def _dest_row(H, h, is_shared, reverse):
    if reverse:
        return int(H - h) if is_shared else 0
    return 0 if is_shared else int(H - h)


_CONFIGS = [
    ([2], 2, 0),
    ([2], 2, 1),
    ([2], 3, 0),
    ([3], 2, 0),
    ([3], 2, 1),
    ([2, 2], 2, 0),
]


def _gen_attempt():
    H = random.randint(19, 30)
    W = random.randint(19, 30)
    bg = random.choice([0, 1, 5, 8, 9])
    avail = [c for c in range(10) if c != bg]
    random.shuffle(avail)

    dup_group_sizes, trap_size, n_extra_unique = random.choice(_CONFIGS)
    total_objects = sum(dup_group_sizes) + trap_size + n_extra_unique
    if total_objects > len(avail):
        raise ValueError("not enough colors")

    objs = []
    exact_used = set()
    d4_used = set()
    color_idx = 0

    for gsz in dup_group_sizes:
        shape, exact_key, d4_key = _pick_fresh_shape(exact_used, d4_used)
        exact_used.add(exact_key)
        d4_used.add(d4_key)
        for _ in range(gsz):
            objs.append(
                {
                    "coords": shape,
                    "color": avail[color_idx],
                    "role": "exact_duplicate",
                    "exact_key": exact_key,
                    "d4_key": d4_key,
                }
            )
            color_idx += 1

    trap_d4_key, trap_shapes = _pick_orientation_family(trap_size, exact_used, d4_used)
    d4_used.add(trap_d4_key)
    for shape in trap_shapes:
        exact_key = _translation_shape_key(shape)
        exact_used.add(exact_key)
        objs.append(
            {
                "coords": shape,
                "color": avail[color_idx],
                "role": "orientation_trap",
                "exact_key": exact_key,
                "d4_key": trap_d4_key,
            }
        )
        color_idx += 1

    for _ in range(n_extra_unique):
        shape, exact_key, d4_key = _pick_fresh_shape(exact_used, d4_used)
        exact_used.add(exact_key)
        d4_used.add(d4_key)
        objs.append(
            {
                "coords": shape,
                "color": avail[color_idx],
                "role": "extra_unique",
                "exact_key": exact_key,
                "d4_key": d4_key,
            }
        )
        color_idx += 1

    exact_freq = Counter(obj["exact_key"] for obj in objs)
    d4_freq = Counter(obj["d4_key"] for obj in objs)

    out = np.full((H, W), bg, dtype=int)
    random.shuffle(objs)
    for obj in objs:
        patch = _coords_to_patch(obj["coords"], obj["color"], bg)
        h, w = patch.shape
        exact_shared = int(exact_freq[obj["exact_key"]]) >= 2
        d4_shared = int(d4_freq[obj["d4_key"]]) >= 2
        dest_r0 = _dest_row(H, h, exact_shared, reverse=True)
        obj["patch"] = patch
        obj["combo_r0"] = dest_r0
        obj["d2_r0"] = _dest_row(H, h, d4_shared, reverse=True)
        obj["d10_r0"] = _dest_row(H, h, exact_shared, reverse=False)
        placed = False
        cols = list(range(0, W - w + 1))
        random.shuffle(cols)
        for c0 in cols:
            if _no_contact_or_overlap(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    random.shuffle(objs)
    for obj in objs:
        patch = obj["patch"]
        h, _w = patch.shape
        c0 = obj["c0"]
        placed = False
        for r0 in _scatter_rows(H, h, obj["combo_r0"]):
            if _no_contact_or_overlap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"

    grouping_changed = sum(
        (int(exact_freq[obj["exact_key"]]) >= 2) != (int(d4_freq[obj["d4_key"]]) >= 2)
        for obj in objs
    )
    trap_changed = sum(
        ((int(exact_freq[obj["exact_key"]]) >= 2) != (int(d4_freq[obj["d4_key"]]) >= 2))
        for obj in objs
        if obj["role"] == "orientation_trap"
    )
    combo_vs_d2 = sum(obj["combo_r0"] != obj["d2_r0"] for obj in objs)
    combo_vs_d10 = sum(obj["combo_r0"] != obj["d10_r0"] for obj in objs)

    assert trap_changed >= trap_size, "orientation trap not expressed strongly enough"
    assert grouping_changed >= trap_size, "translation-only grouping not active enough"
    assert combo_vs_d2 >= trap_size, "combo collapsed toward D2-only behavior"
    assert combo_vs_d10 == len(objs), "direction reversal not expressed on all objects"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant 24")
