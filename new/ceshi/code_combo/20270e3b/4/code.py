# Combo variant of 20270e3b (index 4)
# combo_spec: D1/1+D3/1
# Name: 多连接器嵌合匹配

from __future__ import annotations

import random
import sys

import numpy as np

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, is_contiguous, label_mask, paste


"""
Design analysis for combo D1/1 + D3/1
=====================================

a) Concrete behavior trace
--------------------------
- Original task:
  each sprite had exactly one flat horizontal ctrl bar. The solver could take
  all ctrl pixels from each sprite, assert the two bars had the same normalized
  shape, then translate the plug so its bar sat one row above the base bar.
- D1/1 change:
  a connector is no longer a flat bar. It is a 2 x w zigzag seam (here w=3 for
  readability), with exactly one ctrl pixel in each column. Base and plug match
  only when their column-wise row choices are complementary.
- D3/1 change:
  each sprite now contains two such connectors. Only one base connector and one
  plug connector form a valid complementary pair; the other two are decoys.
- Small concrete trace:
  base connectors: left profile [0,1,0], right profile [1,1,0]
  plug connectors: left profile [1,0,1], right profile [0,0,1]
  all four connectors have the same width, so width alone produces four cross-
  sprite candidates. Only base-left vs plug-left is complementary column by
  column, so only that pair may dock.

b) Complexity calibration
-------------------------
- Every sample visibly contains four orange zigzag connectors, not two.
- The connector width is kept small and fixed (w=3) so humans can read the seam
  shape quickly even with two connectors per sprite.
- Because all four connectors share the same width, the task does not collapse
  into "pick the only equal-length pair"; profile complementarity is necessary.
- The decoy count is still just one per sprite, so branching stays readable.

c) Implicit assumption audit against the original main()
--------------------------------------------------------
- Old assumption: "all ctrl pixels of one sprite belong to one connector".
  Broken. The new solver must segment ctrl pixels into two 8-connected zigzag
  components before matching anything.
- Old assumption: "matching means normalized shapes are equal".
  Broken. The new solver must decode a per-column 0/1 profile and require
  plug_profile == complement(base_profile).
- Old assumption: "plug row = base row - 1".
  Broken. For a shared 2-row seam, the correct translation aligns the connector
  bounding boxes' top rows, not a one-row offset.
- Old assumption: orientation checks can run on the union of all ctrl pixels.
  Broken. Orientation must be checked per connector component.
- New assumption introduced by the combo solver:
  there are exactly two valid downward connectors on the base and two upward
  connectors on the plug, and exactly one cross-pair is complementary.
  The generator explicitly validates these properties before returning a sample.

d) Placement legality
---------------------
- Base is still anchored at the grid origin in the input.
- Generator builds the sprites in their dock-ready local geometry first, using
  fixed connector rows that guarantee a valid seam and no out-of-bounds motion.
- Plug is then scattered elsewhere on the canvas with a non-touching placement,
  so the input always contains exactly two 8-connected foreground components.
- During docked-state validation, obj-obj overlap is forbidden; the only allowed
  overlaps are seam cells where at least one side is ctrl.

e) Visual observability
-----------------------
- A human sees four orange zigzag seams and one output seam becoming solid.
- Across examples, the active pair moves between left/right positions, so the
  rule cannot be inferred from a fixed location cue alone.
- Because decoys use the same width as the true pair, the visible zigzag pattern
  itself is the decisive clue.

f) Visual cleanliness
---------------------
- Connectors are separated either by rows or by at least one empty column, so
  the orange bands do not merge into unreadable blobs.
- The assembled output contains one clean interlocked seam and only bg/obj.

g) Color inference
------------------
- No new color semantics are added. `ctrl=7` still marks candidate seam cells.
- The only color change remains the original one: all ctrl pixels become obj in
  the output after docking.

h) Generator-side guarantees for new solver assumptions
-------------------------------------------------------
- `_profiles_for_sample()` chooses profiles so there is exactly one
  complementary base/plug pair among the four same-width candidates.
- `_build_base_sprite()` and `_build_plug_sprite()` are validated by the same
  connector detector used by `main()`, so the solver's structural assumptions
  are enforced at construction time.
- `_fusion_metrics()` is also used in generation to reject any accidental
  degeneration where profile complementarity would stop mattering.
"""


BG = 1
OBJ = 4
CTRL = 7
CONN_W = 3
SPRITE_H = 5
SPRITE_W = 7
LEFT_C0 = 0
RIGHT_C0 = 4


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _sprite_pts(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}


def _all_profiles(width: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for mask in range(1 << int(width)):
        prof = tuple((mask >> i) & 1 for i in range(int(width)))
        if set(prof) != {0, 1}:
            continue
        if sum(prof[i] != prof[i - 1] for i in range(1, len(prof))) < 1:
            continue
        out.append(tuple(int(x) for x in prof))
    return out


def _complement(profile: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 - int(x) for x in profile)


def _connector_components(
    g: np.ndarray, sprite: set[tuple[int, int]]
) -> list[dict[str, object]]:
    ctrl_mask = np.zeros_like(g, dtype=bool)
    for r, c in sprite:
        if int(g[int(r), int(c)]) == CTRL:
            ctrl_mask[int(r), int(c)] = True

    lab, n = label_mask(ctrl_mask, connectivity=8)
    body = {(int(r), int(c)) for (r, c) in sprite if int(g[int(r), int(c)]) == OBJ}
    out: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        assert coords.size > 0
        pts = {(int(r), int(c)) for r, c in coords}
        r0, c0, r1, c1 = _bbox_of_pts(pts)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        rel = {(int(r - r0), int(c - c0)) for r, c in pts}
        assert h == 2 and w >= 3
        assert len(rel) == w

        profile = []
        for dc in range(w):
            rows = [dr for dr, cc in rel if cc == dc]
            assert len(rows) == 1 and rows[0] in (0, 1)
            profile.append(int(rows[0]))
        profile_t = tuple(profile)
        assert set(profile_t) == {0, 1}

        body_above = all((int(r - 1), int(c)) in body for r, c in pts)
        body_below = all((int(r + 1), int(c)) in body for r, c in pts)
        free_above = all((int(r - 1), int(c)) not in body for r, c in pts)
        free_below = all((int(r + 1), int(c)) not in body for r, c in pts)
        if body_above and free_below and not body_below:
            orientation = "down"
        elif body_below and free_above and not body_above:
            orientation = "up"
        else:
            orientation = "other"

        out.append(
            {
                "pts": pts,
                "r0": int(r0),
                "c0": int(c0),
                "h": int(h),
                "w": int(w),
                "profile": profile_t,
                "orientation": orientation,
            }
        )

    out.sort(key=lambda d: (int(d["r0"]), int(d["c0"])))
    return out


def _pick_active_pair(
    base_connectors: list[dict[str, object]], plug_connectors: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    base_cands = [c for c in base_connectors if c["orientation"] == "down"]
    plug_cands = [c for c in plug_connectors if c["orientation"] == "up"]
    assert len(base_cands) == 2
    assert len(plug_cands) == 2

    matches = []
    for b in base_cands:
        for p in plug_cands:
            if int(b["w"]) != int(p["w"]):
                continue
            if _complement(tuple(int(x) for x in b["profile"])) == tuple(int(x) for x in p["profile"]):
                matches.append((b, p))
    assert len(matches) == 1
    return matches[0]


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
    else:
        base, plug = B, A

    base_conns = _connector_components(g, base)
    plug_conns = _connector_components(g, plug)
    same_width_pairs = 0
    complementary_pairs = 0
    for b in base_conns:
        for p in plug_conns:
            if int(b["w"]) == int(p["w"]):
                same_width_pairs += 1
            if int(b["w"]) == int(p["w"]) and _complement(tuple(int(x) for x in b["profile"])) == tuple(
                int(x) for x in p["profile"]
            ):
                complementary_pairs += 1
    return {
        "same_width_pairs": int(same_width_pairs),
        "complementary_pairs": int(complementary_pairs),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the plug sprite onto the base using the unique complementary zigzag
    connector pair, then recolor ctrl->obj and tight-crop.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != BG, connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, plug = A, B
    else:
        base, plug = B, A

    base_connectors = _connector_components(g, base)
    plug_connectors = _connector_components(g, plug)
    assert len(base_connectors) == 2
    assert len(plug_connectors) == 2
    active_base, active_plug = _pick_active_pair(base_connectors, plug_connectors)

    tr = int(active_base["r0"]) - int(active_plug["r0"])
    tc = int(active_base["c0"]) - int(active_plug["c0"])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_r0, base_c0, _br1, _bc1 = _bbox_of_pts(base)
    assert rr0 == base_r0 and cc0 == base_c0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        src_r, src_c = int(r - tr), int(c - tc)
        val = int(g[int(src_r), int(src_c)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != BG and val != BG:
            assert cur == CTRL or val == CTRL
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == CTRL] = OBJ
    cropped = _crop_non_bg(out, BG)
    assert cropped is not None
    return cropped


def _profiles_for_sample(width: int) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    profiles = _all_profiles(width)
    while True:
        active_base = random.choice(profiles)
        active_plug = _complement(active_base)
        decoy_base = random.choice(profiles)
        decoy_plug = random.choice(profiles)
        pairs = [
            (_complement(active_base) == active_plug),
            (_complement(active_base) == decoy_plug),
            (_complement(decoy_base) == active_plug),
            (_complement(decoy_base) == decoy_plug),
        ]
        if sum(bool(x) for x in pairs) != 1:
            continue
        if decoy_base == active_base:
            continue
        if decoy_plug == active_plug:
            continue
        return active_base, decoy_base, decoy_plug


def _layout_choice() -> tuple[int, int, int, int]:
    # Valid active-side combinations with tc >= 0.
    choices = [
        (LEFT_C0, RIGHT_C0, LEFT_C0, RIGHT_C0),
        (RIGHT_C0, LEFT_C0, LEFT_C0, RIGHT_C0),
        (RIGHT_C0, LEFT_C0, RIGHT_C0, LEFT_C0),
    ]
    return random.choice(choices)


def _add_base_connector(grid: np.ndarray, band_top: int, c0: int, profile: tuple[int, ...]) -> None:
    w = len(profile)
    grid[band_top - 1, c0 : c0 + w] = OBJ
    grid[:band_top, c0] = OBJ
    for i, bit in enumerate(profile):
        c = int(c0 + i)
        if int(bit) == 0:
            grid[band_top, c] = CTRL
        else:
            grid[band_top, c] = OBJ
            grid[band_top + 1, c] = CTRL


def _add_plug_connector(grid: np.ndarray, band_top: int, c0: int, profile: tuple[int, ...]) -> None:
    w = len(profile)
    grid[band_top + 2, c0 : c0 + w] = OBJ
    grid[band_top + 2 :, c0 + w - 1] = OBJ
    for i, bit in enumerate(profile):
        c = int(c0 + i)
        if int(bit) == 0:
            grid[band_top, c] = CTRL
            grid[band_top + 1, c] = OBJ
        else:
            grid[band_top + 1, c] = CTRL


def _build_base_sprite(
    active_profile: tuple[int, ...], decoy_profile: tuple[int, ...], active_c0: int, decoy_c0: int
) -> np.ndarray | None:
    grid = np.full((SPRITE_H, SPRITE_W), BG, dtype=int)
    grid[0, :] = OBJ
    _add_base_connector(grid, band_top=1, c0=int(decoy_c0), profile=decoy_profile)
    _add_base_connector(grid, band_top=3, c0=int(active_c0), profile=active_profile)

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if not (5 <= int(np.sum(grid != BG)) <= 29):
        return None
    sprite = _crop_non_bg(grid, BG)
    if sprite is None:
        return None
    conns = _connector_components(sprite, _sprite_pts(sprite))
    if len(conns) != 2 or any(c["orientation"] != "down" for c in conns):
        return None
    return sprite


def _build_plug_sprite(
    active_profile: tuple[int, ...], decoy_profile: tuple[int, ...], active_c0: int, decoy_c0: int
) -> np.ndarray | None:
    grid = np.full((SPRITE_H, SPRITE_W), BG, dtype=int)
    grid[SPRITE_H - 1, :] = OBJ
    _add_plug_connector(grid, band_top=0, c0=int(active_c0), profile=active_profile)
    _add_plug_connector(grid, band_top=2, c0=int(decoy_c0), profile=decoy_profile)

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if not (5 <= int(np.sum(grid != BG)) <= 29):
        return None
    sprite = _crop_non_bg(grid, BG)
    if sprite is None:
        return None
    conns = _connector_components(sprite, _sprite_pts(sprite))
    if len(conns) != 2 or any(c["orientation"] != "up" for c in conns):
        return None
    return sprite


def _non_touching_placement(
    canvas_h: int, canvas_w: int, base_sprite: np.ndarray, plug_sprite: np.ndarray
) -> tuple[int, int] | None:
    ph, pw = map(int, plug_sprite.shape)
    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != BG)}
    plug_pts = [(int(r), int(c)) for r, c in np.argwhere(plug_sprite != BG)]

    placements: list[tuple[int, int]] = []
    for pr in range(int(canvas_h - ph + 1)):
        for pc in range(int(canvas_w - pw + 1)):
            if int(pr + pc) == 0:
                continue
            shifted = [(int(r + pr), int(c + pc)) for r, c in plug_pts]
            shifted_set = set(shifted)
            if base_pts & shifted_set:
                continue
            touching = False
            min_dist = 10**9
            for br, bc in base_pts:
                for sr, sc in shifted:
                    dr = abs(int(br - sr))
                    dc = abs(int(bc - sc))
                    if dr <= 1 and dc <= 1:
                        touching = True
                        break
                    min_dist = min(min_dist, int(dr + dc))
                if touching:
                    break
            if touching:
                continue
            if 2 <= int(min_dist) <= 7:
                placements.append((int(pr), int(pc)))
    if not placements:
        return None
    return random.choice(placements)


def _gen_attempt() -> np.ndarray | None:
    active_base_profile, decoy_base_profile, decoy_plug_profile = _profiles_for_sample(CONN_W)
    active_plug_profile = _complement(active_base_profile)
    base_active_c0, base_decoy_c0, plug_active_c0, plug_decoy_c0 = _layout_choice()

    base_sprite = _build_base_sprite(active_base_profile, decoy_base_profile, base_active_c0, base_decoy_c0)
    if base_sprite is None:
        return None
    plug_sprite = _build_plug_sprite(active_plug_profile, decoy_plug_profile, plug_active_c0, plug_decoy_c0)
    if plug_sprite is None:
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = random.randint(max(8, bh + 2, ph + 2), 11)
    canvas_w = random.randint(max(11, bw + 3, pw + 3), 13)
    placement = _non_touching_placement(canvas_h, canvas_w, base_sprite, plug_sprite)
    if placement is None:
        return None
    pr, pc = placement

    canvas = np.full((int(canvas_h), int(canvas_w)), BG, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=BG)

    if np.sum(canvas == BG) <= canvas.size // 3:
        return None

    try:
        out = main(canvas)
        metrics = _fusion_metrics(canvas)
    except Exception:
        return None

    if out.size == 0 or not np.any(out != BG):
        return None
    if metrics["same_width_pairs"] != 4:
        return None
    if metrics["complementary_pairs"] != 1:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
