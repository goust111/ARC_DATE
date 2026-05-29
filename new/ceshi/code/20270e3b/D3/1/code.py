# Variant of 20270e3b
# Dimension: D3 — 源对象结构 (src)
# Change: 两个薄壁8连通精灵，各自包含obj像素构成的壁和一个水平ctrl连接器条
#   → 多连接器精灵。每个精灵现在都有两条水平 ctrl 连接器，solver 必须从中选出唯一可对接的一对。

from __future__ import annotations

import random
import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


"""
Design analysis for the D3 variant
==================================

a) Concrete behavior trace
--------------------------
- Preserve the original macro-rule: input has two 8-connected sprites, `base` stays at the
  grid origin, `plug` is translated so the chosen plug connector lands one row above the
  chosen base connector, then every `ctrl` pixel is recolored to `obj` and the result is
  tight-cropped.
- D3 change: each sprite now contains two horizontal `ctrl` bars instead of one.
  Both base connectors face downward (body above, free below).
  Both plug connectors face upward (body below, free above).
  Only one base length is shared with one plug length; that unique shared length defines
  the active docking pair. The other connector on each sprite is a decoy.
- Small trace:
  original: base has one length-2 connector, plug has one length-2 connector -> dock them.
  variant: base has lengths {2, 3}, plug has lengths {1, 2}. The only cross-sprite shared
  length is 2, so the length-2 pair is selected; the length-3 and length-1 bars are ignored.

b) Complexity calibration
-------------------------
- The extra decision is visible: every input shows four orange bars, not two.
- The rule still has low branching: connector lengths are restricted to {1,2,3}, and the
  generator uses a permutation so there is exactly one shared length across sprites.
- This avoids underconstrained matching while keeping the visual cue human-readable in 2-4
  examples.

c) Original-code assumptions that break
---------------------------------------
- `conn(sprite)` used to return all ctrl pixels as a single connector set. With two bars, that
  merges unrelated connectors and destroys the docking anchor.
- `assert shpA == shpB` used to assume one connector per sprite. Now we must split ctrl pixels
  into horizontal components and search for the unique matching pair.
- The original base/plug orientation check ran on every ctrl pixel at once. In the variant,
  orientation is checked per connector component.

d) Placement legality
---------------------
- The generator constructs base with its active connector on the bottom row and the plug with
  its active connector on the top row. This keeps the assembled overlap identical in spirit to
  the original task: the only overlap is plug body over base ctrl at the docking seam.
- The input placement step enforces non-overlap and non-touching separation, so the solver still
  sees exactly two foreground components.

e) Visual observability
-----------------------
- Humans can see that each sprite has two orange bars.
- Across examples, the active length varies among 1/2/3, so the new control variable is not
  frozen.
- The output shows only one seam location becoming attached, making the chosen pair recoverable
  from the examples.

f) Visual cleanliness
---------------------
- The sprites remain thin and line-like; decoys are on separate rows, so the four orange bars do
  not stack into unreadable blobs.
- Output still uses only `bg` and `obj`, so the post-docking image stays clean.

g) Color inference
------------------
- No new color semantics are introduced. `ctrl=7` remains the only special color; D3 changes only
  the number and arrangement of ctrl bars.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Solver assumption: each sprite has exactly two ctrl components, each a contiguous horizontal bar.
  Generator guarantee: bars are built explicitly and validated with the same detector used by `main`.
- Solver assumption: exactly one downward base connector length matches exactly one upward plug
  connector length. Generator guarantee: connector lengths are a permutation of {1,2,3} with one
  shared active length.
- Solver assumption: docking does not move the plug above or left of the base.
  Generator guarantee: base active connector is at the bottom, plug active connector starts at the
  cropped left edge, so the assembled bbox keeps base at the output top-left.
"""


BG = 1
OBJ = 4
CTRL = 7


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _sprite_connectors(
    g: np.ndarray, sprite: set[tuple[int, int]]
) -> list[dict[str, object]]:
    ctrl_mask = np.zeros_like(g, dtype=bool)
    for r, c in sprite:
        if int(g[r, c]) == CTRL:
            ctrl_mask[r, c] = True
    lab, n = label_mask(ctrl_mask, connectivity=4)
    out: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == int(k))
        assert coords.size > 0
        rows = sorted({int(r) for r, _c in coords})
        assert len(rows) == 1
        row = rows[0]
        cols = sorted(int(c) for _r, c in coords)
        assert cols == list(range(cols[0], cols[-1] + 1))
        pts = {(row, c) for c in cols}

        body_above = all((row - 1, c) in sprite and int(g[row - 1, c]) == OBJ for c in cols)
        body_below = all((row + 1, c) in sprite and int(g[row + 1, c]) == OBJ for c in cols)
        free_above = all((row - 1, c) not in sprite for c in cols)
        free_below = all((row + 1, c) not in sprite for c in cols)

        if body_above and free_below and not body_below:
            orientation = "down"
        elif body_below and free_above and not body_above:
            orientation = "up"
        else:
            orientation = "other"

        out.append(
            {
                "pts": pts,
                "row": int(row),
                "c0": int(cols[0]),
                "length": int(len(cols)),
                "shape": tuple((0, i) for i in range(len(cols))),
                "orientation": orientation,
            }
        )
    return out


def _pick_active_pair(
    base_connectors: list[dict[str, object]], plug_connectors: list[dict[str, object]]
) -> tuple[dict[str, object], dict[str, object]]:
    base_cands = [c for c in base_connectors if c["orientation"] == "down"]
    plug_cands = [c for c in plug_connectors if c["orientation"] == "up"]
    assert len(base_cands) == 2
    assert len(plug_cands) == 2
    matches = [(b, p) for b in base_cands for p in plug_cands if b["shape"] == p["shape"]]
    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
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

    base_connectors = _sprite_connectors(g, base)
    plug_connectors = _sprite_connectors(g, plug)
    assert len(base_connectors) == 2
    assert len(plug_connectors) == 2
    active_base, active_plug = _pick_active_pair(base_connectors, plug_connectors)

    tr = int(active_base["row"]) - 1 - int(active_plug["row"])
    tc = int(active_base["c0"]) - int(active_plug["c0"])
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    rr0, cc0, rr1, cc1 = _bbox_of_pts(pts_all)
    base_r0, base_c0, _base_r1, _base_c1 = _bbox_of_pts(base)
    assert rr0 == base_r0 and cc0 == base_c0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), BG, dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[r, c])

    for r, c in plug_shift:
        src_r, src_c = int(r - tr), int(c - tc)
        val = int(g[src_r, src_c])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != BG and val != BG:
            assert cur == CTRL or val == CTRL
        out[int(r - rr0), int(c - cc0)] = val

    out[out == CTRL] = OBJ
    nz = np.argwhere(out != BG)
    if nz.size == 0:
        return np.full((0, 0), BG, dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _crop(grid: np.ndarray) -> np.ndarray | None:
    nz = np.argwhere(grid != BG)
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _sprite_pts(grid: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(grid != BG)}


def _choose_disjoint_starts(width: int, len_a: int, len_b: int) -> tuple[int, int] | None:
    choices: list[tuple[int, int]] = []
    for a0 in range(width - len_a + 1):
        aset = set(range(a0, a0 + len_a))
        for b0 in range(width - len_b + 1):
            bset = set(range(b0, b0 + len_b))
            if aset.isdisjoint(bset):
                choices.append((int(a0), int(b0)))
    if not choices:
        return None
    return random.choice(choices)


def _build_base_sprite(active_len: int, decoy_len: int) -> np.ndarray | None:
    h = random.randint(4, 5)
    min_w = max(4, active_len + decoy_len + 1)
    if min_w > 8:
        return None
    w = random.randint(min_w, 8)
    starts = _choose_disjoint_starts(w, active_len, decoy_len)
    if starts is None:
        return None
    active_c0, decoy_c0 = starts
    active_row = h - 1
    decoy_row = random.randint(1, h - 3)

    mask = np.zeros((h, w), dtype=bool)
    active_end = active_c0 + active_len - 1
    decoy_end = decoy_c0 + decoy_len - 1
    backbone_end = max(active_end, decoy_end)
    extra = random.randint(0, w - 1 - backbone_end)
    mask[0, 0 : backbone_end + extra + 1] = True

    mask[decoy_row - 1, decoy_c0 : decoy_c0 + decoy_len] = True
    mask[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = True
    mask[active_row - 1, active_c0 : active_c0 + active_len] = True
    mask[active_row, active_c0 : active_c0 + active_len] = True

    for r in range(1, decoy_row):
        mask[r, decoy_c0] = True
    for r in range(1, active_row):
        mask[r, active_c0] = True

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    grid[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = CTRL
    grid[active_row, active_c0 : active_c0 + active_len] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if not (5 <= int(np.sum(grid != BG)) <= 29):
        return None
    conns = _sprite_connectors(grid, _sprite_pts(grid))
    if len(conns) != 2 or any(c["orientation"] != "down" for c in conns):
        return None
    lengths = sorted(int(c["length"]) for c in conns)
    if lengths != sorted([active_len, decoy_len]):
        return None
    return _crop(grid)


def _build_plug_sprite(active_len: int, decoy_len: int) -> np.ndarray | None:
    h = random.choice([4, 5])
    min_w = max(4, active_len + decoy_len + 1)
    if min_w > 8:
        return None
    w = random.randint(min_w, 8)

    active_c0 = 0
    decoy_choices = [c0 for c0 in range(w - decoy_len + 1) if set(range(c0, c0 + decoy_len)).isdisjoint(set(range(active_c0, active_c0 + active_len)))]
    if not decoy_choices:
        return None
    decoy_c0 = random.choice(decoy_choices)
    decoy_row = random.randint(2, h - 2)

    mask = np.zeros((h, w), dtype=bool)
    active_end = active_c0 + active_len - 1
    decoy_end = decoy_c0 + decoy_len - 1
    backbone_end = max(active_end, decoy_end)
    extra = random.randint(0, w - 1 - backbone_end)
    mask[h - 1, 0 : backbone_end + extra + 1] = True

    mask[0, active_c0 : active_c0 + active_len] = True
    mask[1, active_c0 : active_c0 + active_len] = True
    mask[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = True
    mask[decoy_row + 1, decoy_c0 : decoy_c0 + decoy_len] = True

    for r in range(1, h - 1):
        mask[r, active_c0] = True
    for r in range(decoy_row + 1, h - 1):
        mask[r, decoy_c0] = True

    grid = np.full((h, w), BG, dtype=int)
    grid[mask] = OBJ
    grid[0, active_c0 : active_c0 + active_len] = CTRL
    grid[decoy_row, decoy_c0 : decoy_c0 + decoy_len] = CTRL

    if not is_contiguous(grid != BG, background=0, connectivity=8):
        return None
    if not (5 <= int(np.sum(grid != BG)) <= 29):
        return None
    conns = _sprite_connectors(grid, _sprite_pts(grid))
    if len(conns) != 2 or any(c["orientation"] != "up" for c in conns):
        return None
    lengths = sorted(int(c["length"]) for c in conns)
    if lengths != sorted([active_len, decoy_len]):
        return None
    return _crop(grid)


def _non_touching_placement(
    canvas_h: int, canvas_w: int, base_sprite: np.ndarray, plug_sprite: np.ndarray
) -> tuple[int, int] | None:
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != BG)}
    plug_pts = [(int(r), int(c)) for r, c in np.argwhere(plug_sprite != BG)]

    placements: list[tuple[int, int]] = []
    for pr in range(canvas_h - ph + 1):
        for pc in range(canvas_w - pw + 1):
            if pr + pc == 0:
                continue
            shifted = [(r + pr, c + pc) for r, c in plug_pts]
            shifted_set = set(shifted)
            if base_pts & shifted_set:
                continue
            touching = False
            min_dist = 10**9
            for br, bc in base_pts:
                for sr, sc in shifted:
                    dr = abs(br - sr)
                    dc = abs(bc - sc)
                    if dr <= 1 and dc <= 1:
                        touching = True
                        break
                    min_dist = min(min_dist, dr + dc)
                if touching:
                    break
            if touching:
                continue
            if 2 <= min_dist <= 6:
                placements.append((int(pr), int(pc)))
    if not placements:
        return None
    return random.choice(placements)


def _gen_attempt() -> np.ndarray | None:
    active_len, base_decoy_len, plug_decoy_len = random.sample([1, 2, 3], 3)

    base_sprite = _build_base_sprite(active_len, base_decoy_len)
    if base_sprite is None:
        return None
    plug_sprite = _build_plug_sprite(active_len, plug_decoy_len)
    if plug_sprite is None:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    if bh + ph - 1 > 8:
        return None

    canvas_h = random.randint(max(7, bh, ph), 11)
    canvas_w = random.randint(max(6, bw, pw), 13)
    placement = _non_touching_placement(canvas_h, canvas_w, base_sprite, plug_sprite)
    if placement is None:
        return None
    pr, pc = placement

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=BG)
    paste(canvas, plug_sprite, pr, pc, transparent=BG)

    if np.sum(canvas == BG) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None

    if result.size == 0 or not np.any(result != BG):
        return None
    if not (5 <= result.shape[0] <= 8 and 3 <= result.shape[1] <= 12):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 400 attempts")
