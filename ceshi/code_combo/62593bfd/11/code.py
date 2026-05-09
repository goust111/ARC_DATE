# Combo variant of 62593bfd (index 11)
# combo_spec: D2/2+D7/1
# Name: 三级频次分流 + 大量对象

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

# Design analysis for combo D2/2 + D7/1
# --------------------------------------
# a) Concrete behavior and trace
# - Original 62593bfd groups 8-connected foreground objects by D4-canonical shape
#   frequency, then uses only two routes:
#     freq >= 2 -> row 0
#     freq == 1 -> row H-h
# - This combo keeps the same grouping key but replaces the binary routing with
#   three visible frequency tiers:
#     freq >= 3 -> row 0
#     freq == 2 -> row H//2
#     freq == 1 -> row H-h
# - Small trace in H=28:
#     objects A,A,A share one canonical shape (top tier),
#     B,B share another shape (middle tier),
#     C,D,E are singletons (bottom tier).
#   Original task would send A,A,A and B,B all to row 0, leaving only C,D,E at the
#   bottom. The fused task instead splits them into three horizontal bands, so the
#   pair-group is no longer mixed with the 3-copy group.
#
# b) Complexity calibration
# - The object count rises to 8-10, but the number of frequency categories is still
#   just three. That keeps the rule inferable while making the new middle tier
#   genuinely necessary.
# - Each sample is forced to contain:
#     1 group of size 3 or 4,
#     1 or 2 groups of size 2,
#     2-4 unique groups.
#   This prevents degeneration back to the original two-tier task.
# - Shapes stay small (bbox 2-4, area 3-8) so 8-10 objects still fit cleanly in a
#   near-30x30 grid.
#
# c) Hidden-assumption audit of main()
# - `find_components(... connectivity=8, monochromatic=False)` assumes different
#   objects never touch even diagonally, regardless of color. The generator must now
#   enforce an 8-neighborhood moat because 8-10 objects make accidental mergers much
#   more likely than in the seed task.
# - `_canonical_shape(rel)` still depends only on geometry, not orientation or color.
#   This remains valid because the combo changes only the routing thresholds.
# - The original binary branch
#       `0 if freq >= 2 else H-h`
#   is no longer valid. It must become a 3-way branch, and the middle route must use
#   a fixed human-visible stripe rather than an arbitrary per-object formula.
# - `paste()` still assumes routed placements do not overlap. With three tiers, this
#   becomes a stronger generator responsibility: output-first packing must ensure each
#   tier fits at its destination row before the input is scattered.
#
# d) Placement legality
# - Output is constructed first from the intended final tiers. Objects within the same
#   tier are assigned non-touching horizontal slots with preserved `c0`.
# - Top objects share row 0, middle objects share row H//2, and bottom objects align
#   their bottom edge via row H-h.
# - Input generation then keeps those `c0` values and randomizes only the row, while
#   excluding the destination band for every object. This makes vertical motion visible
#   and preserves legality.
#
# e) Visual observability
# - A human can infer the fused rule from a few examples because three families are
#   visible simultaneously:
#     3-4 copies of one silhouette collect on top,
#     2 copies of another silhouette collect on the middle stripe,
#     singletons collect on the bottom.
# - No numeric color property, component order, or hidden tie-break participates.
#
# f) Visual cleanliness
# - The grid is enlarged to 27-30 in each dimension, and output placement uses one
#   blank column gap between bbox slots inside each band. This keeps the three bands
#   readable despite the higher object count.
# - Shapes are kept compact and irregular enough to be trackable without the scene
#   becoming a wall of touching blocks.
#
# g) Change inferability
# - Colors are not remapped. Objects keep color, shape, and orientation; only vertical
#   destination depends on the frequency tier. This keeps the new rule focused on
#   counting repeated canonical shapes rather than learning a second hidden code.
#
# h) Non-degeneracy check used after implementation
# - For generated samples, compare this combo output against the original solver
#   output. Because every sample contains at least one freq==2 shape group, the two
#   outputs must differ: the pair-group lands at row H//2 here but at row 0 in the
#   original task.
# - Also measure the per-sample frequency signature `(n_top, n_mid, n_bottom)` and
#   require all three counts to be positive.
#
# i) New solver assumptions that generate_input() must guarantee
# - There is at least one group in each tier {>=3, ==2, ==1}.
# - All routed objects fit in the three bands without overlap or 8-touching.
# - Input objects remain 8-disconnected under `monochromatic=False`.
# - The fixed middle band `row H//2` is free of any pre-existing input objects from
#   the pair tier, so the middle movement is always visible.


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
        rr = int(r)
        cc = int(c)
        if 0 <= rr < H and 0 <= cc < W:
            mask[rr, cc] = True
    return mask


def _dest_row(freq, H, h):
    if int(freq) >= 3:
        return 0
    if int(freq) == 2:
        return int(H // 2)
    return int(H - h)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group 8-connected objects by D4-canonical shape frequency.

    Routing:
    - freq >= 3 -> top edge      (row 0)
    - freq == 2 -> middle stripe (row H//2)
    - freq == 1 -> bottom edge   (row H-h)
    Horizontal position (bbox left column c0) is preserved.
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
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        paste(
            out,
            pat,
            _dest_row(int(freq[d["shape_key"]]), H, h),
            int(d["c0"]),
            transparent=bg,
        )

    return out


def _grow_shape(area, connectivity=8, max_dim=4):
    """Grow a random connected shape of exactly `area` pixels."""
    deltas = (
        [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
        if connectivity == 8
        else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    )
    cells = {(0, 0)}
    while len(cells) < int(area):
        frontier = []
        for r, c in cells:
            for dr, dc in deltas:
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) in cells:
                    continue
                all_r = [rr for rr, _ in cells] + [nr]
                all_c = [cc for _, cc in cells] + [nc]
                if max(all_r) - min(all_r) < int(max_dim) and max(all_c) - min(all_c) < int(max_dim):
                    frontier.append((nr, nc))
        if not frontier:
            raise ValueError("cannot grow shape")
        cells.add(random.choice(frontier))
    mr = min(r for r, _c in cells)
    mc = min(c for _r, c in cells)
    return [(r - mr, c - mc) for r, c in cells]


def _shape_dims(coords):
    rows = [int(r) for r, _c in coords]
    cols = [int(c) for _r, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _good_shape(coords):
    h, w = _shape_dims(coords)
    area = len(coords)
    return 2 <= h <= 4 and 2 <= w <= 4 and 3 <= area <= 8 and area < h * w


def _make_shape(canon_used):
    for _ in range(300):
        area = random.randint(3, 8)
        coords = _grow_shape(area, connectivity=8, max_dim=4)
        canon = _canonical_shape(coords)
        if canon in canon_used:
            continue
        if not _good_shape(coords):
            continue
        canon_used.add(canon)
        return coords
    raise ValueError("shape generation failed")


def _transform_shape(coords, rot, flip):
    pts = [(int(r), int(c)) for r, c in coords]
    for _ in range(int(rot) % 4):
        pts = [(c, -r) for r, c in pts]
    if flip:
        pts = [(-r, c) for r, c in pts]
    mr = min(r for r, _c in pts)
    mc = min(c for _r, c in pts)
    return [(r - mr, c - mc) for r, c in pts]


def _coords_to_patch(coords, color, bg):
    mr = max(int(r) for r, _c in coords)
    mc = max(int(c) for _r, c in coords)
    patch = np.full((mr + 1, mc + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[int(r), int(c)] = int(color)
    return patch


def _no_touch(grid, patch, r0, c0, bg):
    """Reject overlap and 8-neighbor contact with existing non-background pixels."""
    h, w = patch.shape
    H, W = grid.shape
    if int(r0) < 0 or int(c0) < 0 or int(r0) + h > H or int(c0) + w > W:
        return False

    mask = patch != int(bg)
    if np.any(grid[int(r0) : int(r0) + h, int(c0) : int(c0) + w][mask] != int(bg)):
        return False

    for pr, pc in np.argwhere(mask):
        rr = int(r0 + pr)
        cc = int(c0 + pc)
        rr0 = max(0, rr - 1)
        rr1 = min(H, rr + 2)
        cc0 = max(0, cc - 1)
        cc1 = min(W, cc + 2)
        if np.any(grid[rr0:rr1, cc0:cc1] != int(bg)):
            return False
    return True


def _assign_band_positions(objs, W, gap=1):
    band = list(objs)
    random.shuffle(band)
    widths = [int(obj["patch"].shape[1]) for obj in band]
    total = sum(widths) + max(0, len(widths) - 1) * int(gap)
    if total > int(W):
        raise ValueError("band too wide")

    slack = int(W - total)
    gaps = [0] * (len(widths) + 1)
    for _ in range(slack):
        gaps[random.randrange(len(gaps))] += 1

    cur = int(gaps[0])
    for idx, obj in enumerate(band):
        obj["c0"] = int(cur)
        cur += widths[idx]
        if idx + 1 < len(widths):
            cur += int(gap) + int(gaps[idx + 1])
    return band


_CONFIGS = [
    (3, 1, 3),
    (3, 1, 4),
    (4, 1, 2),
    (4, 1, 3),
    (3, 2, 2),
    (3, 2, 3),
    (4, 2, 2),
]


def _candidate_input_rows(H, h):
    rows = [r for r in range(2, int(H - h - 1)) if abs(r - int(H // 2)) > 1]
    if rows:
        return rows
    rows = [r for r in range(1, int(H - h)) if r != int(H // 2)]
    if rows:
        return rows
    return [r for r in range(0, int(H - h) + 1) if r != int(H // 2)]


def _original_output(inp):
    g = np.array(inp, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    extracted = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (r0, c0, r1, c1))
        rel = [(int(r) - r0, int(c) - c0) for r, c in comp["coords"]]
        mask = _mask_from_coords(rel, patch.shape)
        pat = np.full_like(patch, bg)
        pat[mask] = patch[mask]
        extracted.append({"c0": int(c0), "pat": pat, "shape_key": _canonical_shape(rel)})

    freq = Counter(d["shape_key"] for d in extracted)
    out = np.full((H, W), bg, dtype=int)
    for d in extracted:
        pat = d["pat"]
        h, _w = map(int, pat.shape)
        dest_r0 = 0 if int(freq[d["shape_key"]]) >= 2 else int(H - h)
        paste(out, pat, dest_r0, int(d["c0"]), transparent=bg)
    return out


def _gen_attempt():
    H = random.randint(27, 30)
    W = random.randint(27, 30)
    bg = random.choice([0, 1, 5, 8, 9])

    top_size, n_pair_groups, n_unique = random.choice(_CONFIGS)

    objs = []
    canon_used = set()

    top_base = _make_shape(canon_used)
    for _ in range(top_size):
        objs.append({"coords": _transform_shape(top_base, random.randint(0, 3), random.choice([False, True])), "freq": top_size})

    for _ in range(n_pair_groups):
        pair_base = _make_shape(canon_used)
        for _ in range(2):
            objs.append({"coords": _transform_shape(pair_base, random.randint(0, 3), random.choice([False, True])), "freq": 2})

    for _ in range(n_unique):
        uniq_base = _make_shape(canon_used)
        objs.append({"coords": _transform_shape(uniq_base, random.randint(0, 3), random.choice([False, True])), "freq": 1})

    n_objs = len(objs)
    assert 8 <= n_objs <= 10

    palette = [c for c in range(10) if c != int(bg)]
    random.shuffle(palette)
    colors = (palette * ((n_objs + len(palette) - 1) // len(palette)))[:n_objs]
    random.shuffle(colors)

    for obj, color in zip(objs, colors):
        obj["color"] = int(color)
        obj["patch"] = _coords_to_patch(obj["coords"], int(color), int(bg))

    bands = {
        "top": [obj for obj in objs if int(obj["freq"]) >= 3],
        "mid": [obj for obj in objs if int(obj["freq"]) == 2],
        "bot": [obj for obj in objs if int(obj["freq"]) == 1],
    }
    assert bands["top"] and bands["mid"] and bands["bot"]

    out = np.full((H, W), int(bg), dtype=int)
    for band_name in ("top", "mid", "bot"):
        for obj in _assign_band_positions(bands[band_name], W, gap=1):
            h, _w = map(int, obj["patch"].shape)
            obj["dest_r0"] = _dest_row(int(obj["freq"]), H, h)
            if not _no_touch(out, obj["patch"], int(obj["dest_r0"]), int(obj["c0"]), int(bg)):
                raise ValueError("output placement failed")
            paste(out, obj["patch"], int(obj["dest_r0"]), int(obj["c0"]), transparent=int(bg))

    inp = np.full((H, W), int(bg), dtype=int)
    place_order = list(objs)
    random.shuffle(place_order)
    place_order.sort(key=lambda obj: int(obj["patch"].shape[0] * obj["patch"].shape[1]), reverse=True)

    for obj in place_order:
        patch = obj["patch"]
        h, _w = map(int, patch.shape)
        c0 = int(obj["c0"])
        rows = _candidate_input_rows(H, h)
        random.shuffle(rows)

        placed = False
        for r0 in rows:
            if int(r0) == int(obj["dest_r0"]):
                continue
            if _no_touch(inp, patch, int(r0), c0, int(bg)):
                paste(inp, patch, int(r0), c0, transparent=int(bg))
                obj["input_r0"] = int(r0)
                placed = True
                break
        if not placed:
            raise ValueError("input scatter failed")

    comps = find_components(
        inp,
        bg=int(bg),
        connectivity=8,
        monochromatic=False,
        return_format="dict",
    )
    if len(comps) != n_objs:
        raise ValueError("component merge in input")

    result = main(inp)
    assert np.array_equal(result, out), "round-trip check failed"
    assert not np.array_equal(result, _original_output(inp)), "degenerated to original routing"
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        try:
            return _gen_attempt()
        except (ValueError, AssertionError, IndexError):
            continue
    raise RuntimeError("failed to generate input for 62593bfd combo D2/2+D7/1")
