# Combo variant of 62593bfd (index 8)
# combo_spec: D4/2+D5/2
# Name: 水平镜像位置 + 水平翻转形状

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

spec_variant:
  inherited_core:
    - Grid size, bg choice, object count, area range, 8-connectivity, and D4
      canonical grouping all stay the same as the original 62593bfd task.
    - Objects are still routed by shared-vs-unique canonical shape:
      freq >= 2 -> top edge, freq == 1 -> bottom edge.

  fused_change:
    - D4/2 contributes a horizontal destination mirror:
      dest_c0 = W - c0 - w.
    - D5/2 contributes a horizontal reflection of each object's own patch:
      moved_pat = fliplr(pat).
    - Both changes use the same left-right axis, so the solver must infer that
      the object both moves to the mirrored column and also flips inside its
      own bbox.

  concrete_trace:
    - Suppose W = 12 and an input object has bbox-left c0 = 2, width w = 3,
      height h = 3, patch:
        700
        770
        070
    - Original task:
        shared -> paste unchanged patch at (0, 2)
        unique -> paste unchanged patch at (H-h, 2)
    - Combo task:
        mirrored column = 12 - 2 - 3 = 7
        flipped patch:
          007
          077
          070
        shared -> paste flipped patch at (0, 7)
        unique -> paste flipped patch at (H-h, 7)
    - Difference from original: both anchor column and interior geometry change.
      Difference from each single variant:
        * D4/2-only would move to column 7 but keep the old patch.
        * D5/2-only would flip the patch but keep column 2.

  complexity_calibration:
    - The generator forbids fixed-point horizontal placements where
      dest_c0 == c0; otherwise the D4/2 part becomes invisible.
    - It also rejects horizontally symmetric patches or width-1 patches;
      otherwise the D5/2 part becomes invisible.
    - The object count remains 4-6 with at least one shared-shape group and at
      least one unique shape, so the original top/bottom reasoning remains well
      supported rather than being drowned out by the new left-right cue.

  implicit_assumption_audit_main:
    - `_canonical_shape(rel)` still assumes grouping is computed from the input
      object shape up to D4 symmetry. This remains valid because the combo only
      changes rendering at the destination, not grouping semantics.
    - Extracting `pat` from the cropped bbox still assumes the bbox is the
      correct movable unit. This remains valid; the combo simply uses
      `fliplr(pat)` as the moved patch.
    - The original `dest_c0 = c0` assumption is broken. Destination placement
      must now depend on both the input column and the patch width.
    - `paste(out, ...)` assumes transformed destinations are legal and do not
      overlap. That is no longer automatic, so the generator must validate
      mirrored-column placements using the already-flipped patch.

  placement_legality:
    - Because `find_components(..., monochromatic=False, connectivity=8)` would
      merge diagonally touching objects even across colors, generator placement
      enforces a one-cell 8-neighborhood gap in both input and output.
    - Output-first generation now means: choose an input c0, derive the mirrored
      destination column from it, flip the patch, and verify the flipped patch
      can be pasted there without overlap.
    - Input scatter uses the original patch at the original c0 and only changes
      row, with rows restricted to the interior so the vertical motion remains
      visible.

  visual_observability:
    - Every object is constructed so its output differs from the original task,
      from the D4/2-only ablation, and from the D5/2-only ablation.
    - Distinct colors per object are preserved so humans can track a given
      component across the simultaneous position-mirror and shape-flip.
    - No invisible metadata such as raw color ids or component order is used in
      any routing decision.

  cleanliness:
    - No markers, frames, or extra clutter are added.
    - Objects stay sparse and 8-disconnected, so the grid still reads as a clean
      set of independent sprites rather than a merged mass.

  new_main_assumptions_and_generator_guarantees:
    - `main()` now assumes horizontal flip visibility is not needed for
      correctness, only for learnability. `generate_input()` guarantees visible
      flips so training examples do not degenerate.
    - `main()` assumes mirrored destinations remain in bounds; this is ensured by
      `dest_c0 = W - c0 - w` with `0 <= c0 <= W - w`.
    - `main()` assumes different transformed objects do not collide; the
      generator checks that explicitly at output placement time.
"""

import random
from collections import Counter

import numpy as np
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


def _flip_patch_horiz(patch: np.ndarray) -> np.ndarray:
    return np.fliplr(np.array(patch, dtype=int)).copy()


def _mirror_c0(W, c0, w):
    return int(W - c0 - w)


def _extract_objects(grid: np.ndarray, bg: int):
    comps = find_components(
        grid, bg=bg, connectivity=8, monochromatic=False, return_format="dict"
    )
    assert comps

    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        m = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[m] = patch[m]
        extracted.append(
            {
                "c0": int(c0),
                "pat": pat,
                "shape_key": _canonical_shape(rel),
            }
        )
    return extracted


def _render_output(
    grid: np.ndarray, *, mirror_position: bool, flip_shape: bool
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    extracted = _extract_objects(g, bg)
    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)

    for d in extracted:
        moved_pat = _flip_patch_horiz(d["pat"]) if flip_shape else d["pat"]
        h, w = map(int, moved_pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        dest_c0 = _mirror_c0(W, d["c0"], w) if mirror_position else int(d["c0"])
        paste(out, moved_pat, dest_r0, dest_c0, transparent=bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by D4-canonical shape, move shared shapes to the top edge and
    unique shapes to the bottom edge, while simultaneously mirroring each
    object's destination column and horizontally flipping the object's patch.
    """
    return _render_output(grid, mirror_position=True, flip_shape=True)


def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random 8-connected shape of exactly `area` pixels."""
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
        patch[r, c] = color
    return patch


def _is_hflip_visible_patch(patch: np.ndarray) -> bool:
    return patch.shape[1] > 1 and not np.array_equal(patch, _flip_patch_horiz(patch))


def _random_visible_transform(coords, color, bg):
    candidates = []
    for rot in range(4):
        for flip in (False, True):
            transformed = _transform_shape(coords, rot, flip)
            patch = _coords_to_patch(transformed, color, bg)
            if _is_hflip_visible_patch(patch):
                candidates.append((transformed, patch))
    if not candidates:
        raise ValueError("shape has no visibly flippable orientation")
    return random.choice(candidates)


def _can_place_with_gap(grid, patch, r0, c0, bg):
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
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if r0 <= nr < r0 + h and c0 <= nc < c0 + w and mask[nr - r0, nc - c0]:
                    continue
                if int(grid[nr, nc]) != int(bg):
                    return False
    return True


_CONFIGS = [
    ([2], 2),
    ([2], 3),
    ([3], 1),
    ([3], 2),
    ([3], 3),
    ([2, 2], 1),
    ([2, 2], 2),
]


def _interior_row(H, h):
    lo = 1
    hi = H - h - 1
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _combo_metrics(inp: np.ndarray):
    g = np.array(inp, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    extracted = _extract_objects(g, bg)
    moved_columns = 0
    visible_flips = 0
    for d in extracted:
        pat = d["pat"]
        h, w = map(int, pat.shape)
        del h
        if _mirror_c0(W, d["c0"], w) != int(d["c0"]):
            moved_columns += 1
        if _is_hflip_visible_patch(pat):
            visible_flips += 1
    return {
        "n_objects": len(extracted),
        "moved_columns": moved_columns,
        "visible_flips": visible_flips,
    }


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
        for _ in range(200):
            area = random.randint(3, 12)
            base = _grow_shape(area, 8)
            canon = _canonical_shape(base)
            if canon in canon_used:
                continue
            try:
                samples = []
                for j in range(gsz):
                    coords, patch = _random_visible_transform(base, avail[ci + j], bg)
                    samples.append((coords, patch))
            except ValueError:
                continue
            canon_used.add(canon)
            for j, (coords, patch) in enumerate(samples):
                objs.append(
                    {
                        "coords": coords,
                        "patch": patch,
                        "color": avail[ci + j],
                        "shared": True,
                    }
                )
            ci += gsz
            break
        else:
            raise ValueError("shared shape not found")

    for _ in range(n_unique):
        for _attempt in range(200):
            area = random.randint(3, 12)
            shape = _grow_shape(area, 8)
            canon = _canonical_shape(shape)
            if canon in canon_used:
                continue
            color = avail[ci]
            try:
                coords, patch = _random_visible_transform(shape, color, bg)
            except ValueError:
                continue
            canon_used.add(canon)
            objs.append(
                {
                    "coords": coords,
                    "patch": patch,
                    "color": color,
                    "shared": False,
                }
            )
            ci += 1
            break
        else:
            raise ValueError("unique shape not found")

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        out_patch = _flip_patch_horiz(obj["patch"])
        h, w = out_patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(400):
            c0 = random.randint(0, W - w)
            dest_c0 = _mirror_c0(W, c0, w)
            if dest_c0 == c0:
                continue
            if _can_place_with_gap(out, out_patch, dest_r0, dest_c0, bg):
                paste(out, out_patch, dest_r0, dest_c0, transparent=bg)
                obj["c0"] = c0
                obj["dest_c0"] = dest_c0
                obj["out_patch"] = out_patch
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["patch"]
        h, w = patch.shape
        c0 = int(obj["c0"])
        placed = False
        for _ in range(400):
            r0 = _interior_row(H, h)
            if r0 is None:
                raise ValueError("no interior rows")
            if _can_place_with_gap(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                obj["r0"] = r0
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    combo_out = main(inp)
    assert np.array_equal(combo_out, out), "round-trip check failed"

    base_out = _render_output(inp, mirror_position=False, flip_shape=False)
    dst_only_out = _render_output(inp, mirror_position=True, flip_shape=False)
    flip_only_out = _render_output(inp, mirror_position=False, flip_shape=True)
    if np.array_equal(combo_out, base_out):
        raise AssertionError("combo degenerated to original behavior")
    if np.array_equal(combo_out, dst_only_out):
        raise AssertionError("combo lost shape-flip behavior")
    if np.array_equal(combo_out, flip_only_out):
        raise AssertionError("combo lost mirrored-position behavior")

    metrics = _combo_metrics(inp)
    if metrics["moved_columns"] != metrics["n_objects"]:
        raise AssertionError("not all objects used mirrored destination columns")
    if metrics["visible_flips"] != metrics["n_objects"]:
        raise AssertionError("not all objects visibly flipped")

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo variant")
