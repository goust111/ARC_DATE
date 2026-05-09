# Variant of 62593bfd
# Dimension: D5 — 对象变换——移动过程中的形状/颜色变化 (transform)
# Change: 对象在平移过程中不做任何变换：形状、颜色、朝向完全保持原样（duplicated-size移顶部，singleton-s... → unique对象重着色为统一颜色
#
# Design analysis for this D5 variant
# -----------------------------------
# spec_variant:
#   core_rule:
#     shared_shape_objects:
#       condition: canonical D4 shape frequency >= 2
#       move: translate to top edge, preserve c0
#       transform: no change
#     unique_shape_objects:
#       condition: canonical D4 shape frequency == 1
#       move: translate to bottom edge, preserve c0
#       transform: recolor every non-bg pixel to one common color
#   recolor_rule:
#     formula: unique_color = (bg + 1) % 10
#     rationale:
#       - deterministic and visible across examples
#       - tied to a visible grid property (background color), not a hidden ordering
#       - mirrors the "subset gets recolored to a fixed role color" flavor from 20270e3b
#   concrete_example:
#     - Suppose bg=5, so unique_color=6.
#     - If three objects have canonical-shape frequencies [2,2,1], then:
#       shared pair move to row 0 unchanged;
#       the singleton moves to row H-h and every foreground pixel that was, say, color 3
#       becomes color 6.
#     - Original task would have kept that singleton color 3. The variant differs only in
#       this last rendering step.
#   complexity_calibration:
#     - Keep D4 grouping unchanged so the task still centers on shape equivalence.
#     - Force at least 2 unique objects in each generated example so the shared recolor is
#       supported by repetition inside a single grid, instead of being a one-off event.
#     - Exclude unique_color from input object colors so the recolor creates a genuine
#       visible change and does not accidentally match an original color.
#   implicit_assumptions_audit:
#     - find_components(... monochromatic=False) still works because input objects remain
#       disconnected; recolor happens only in output rendering, not before grouping.
#     - shape_key uses geometry only, so recoloring unique objects does not affect grouping.
#     - paste(... transparent=bg) remains safe because recolor changes only non-bg pixels.
#     - The old solver assumed "pat can be pasted unchanged for every object"; that is no
#       longer true for unique objects, so the patch must be recolored just before paste.
#   placement_legality:
#     - Input objects must stay 8-disconnected or the multicolor component finder would
#       merge them and break object extraction.
#     - Output unique objects now share one color, so generator must also avoid 8-touching
#       at the bottom edge; otherwise multiple unique objects could visually merge.
#     - The generator therefore enforces a one-cell 8-neighborhood clearance between
#       placed objects in both output and scattered input.
#   visual_observability:
#     - Humans can still see the original top-vs-bottom grouping by shared shape.
#     - The new D5 behavior is visible because all bottom singleton objects lose their
#       original distinct colors and adopt the same new color, while top shared objects do not.
#     - Varying bg across examples changes the target recolor value and exposes the
#       dependency on background rather than on a fixed absolute color id.

from __future__ import annotations

import random
from collections import Counter

import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste


def _canonical_shape(coords):
    """Canonical D4-invariant shape key."""
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


def _target_unique_color(bg: int) -> int:
    return int((int(bg) + 1) % 10)


def _recolor_patch(patch: np.ndarray, bg: int, color: int) -> np.ndarray:
    out = np.array(patch, copy=True)
    out[out != int(bg)] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected objects by D4-canonical shape.

    - Shared shapes  (freq >= 2): move to top edge unchanged.
    - Unique shapes  (freq == 1): move to bottom edge and recolor to (bg + 1) % 10.
    - Horizontal position (bbox c0) is preserved.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    unique_color = _target_unique_color(bg)

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

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), int(bg), dtype=int)

    for d in extracted:
        is_shared = int(freq[d["shape_key"]]) >= 2
        pat = d["pat"] if is_shared else _recolor_patch(d["pat"], bg, unique_color)
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if is_shared else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=int(bg))

    return out


# ────────────────────── generator ──────────────────────

def _grow_shape(area, connectivity=8, max_dim=5):
    """Grow a random connected shape of exactly *area* pixels."""
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
    p = np.full((mr + 1, mc + 1), bg, dtype=int)
    for r, c in coords:
        p[int(r), int(c)] = int(color)
    return p


def _clearance_ok(grid, patch, r0, c0, bg):
    """Require in-bounds placement with a 1-cell 8-neighborhood moat to existing fg."""
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
        area = random.randint(3, 12)
        base = _grow_shape(area, 8)
        canon = _canonical_shape(base)
        if canon in canon_used:
            raise ValueError("duplicate canonical")
        canon_used.add(canon)
        for _ in range(gsz):
            transformed = _transform_shape(base, random.randint(0, 3), random.choice([False, True]))
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

    out = np.full((H, W), bg, dtype=int)
    for obj in objs:
        out_color = int(obj["color"]) if obj["shared"] else int(unique_color)
        patch = _coords_to_patch(obj["coords"], out_color, bg)
        h, w = patch.shape
        dest_r0 = 0 if obj["shared"] else H - h
        placed = False
        for _ in range(500):
            c0 = random.randint(0, W - w)
            if _clearance_ok(out, patch, dest_r0, c0, bg):
                paste(out, patch, dest_r0, c0, transparent=bg)
                obj["c0"] = c0
                obj["input_patch"] = _coords_to_patch(obj["coords"], obj["color"], bg)
                placed = True
                break
        if not placed:
            raise ValueError("output placement failed")

    inp = np.full((H, W), bg, dtype=int)
    for obj in objs:
        patch = obj["input_patch"]
        h, _w = patch.shape
        c0 = int(obj["c0"])
        dest_r0 = 0 if obj["shared"] else H - h
        row_choices = [r for r in range(0, H - h + 1) if r != dest_r0]
        random.shuffle(row_choices)
        placed = False
        for r0 in row_choices[:500]:
            if _clearance_ok(inp, patch, r0, c0, bg):
                paste(inp, patch, r0, c0, transparent=bg)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(600):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd D5 variant")
