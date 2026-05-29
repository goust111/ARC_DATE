# Combo variant of 20270e3b (index 25)
# combo_spec: D2/1+D6/1+D14/2
# Name: 分隔线自适应方向连接器判角色

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


"""
Design analysis for the fused variant
=====================================

This combo merges three changes into one coherent rule:
- D14/2: the input is split into two regions by one full ctrl-colored separator.
- D2/1: connector orientation matches the separator orientation, so docking is
  orthogonal to that line.
- D6/1: base is not chosen by global position. Instead, inspect each connector's
  free side: the sprite whose free side points toward the separator is the base.

a) Concretization and behavior trace
------------------------------------
- Original task: there are only two sprites, both use horizontal connectors, base
  is the one at the input origin, and plug is always moved to the row just above
  base's connector.
- Fused task: the separator is visible first. A horizontal separator implies both
  connectors are horizontal and the merge happens vertically; a vertical separator
  implies vertical connectors and a horizontal merge.
- Small concrete case:
    * horizontal separator
    * top sprite has a connector with body above / free side below
    * bottom sprite has body above / free side below as well
  The top sprite is base because "down" points toward the separator, while the
  bottom sprite is plug because the same "down" points away from the separator.
  During docking the plug connector is placed one cell below the base connector,
  so the plug body overlaps the base connector strip and both ctrl strips vanish
  into obj in the output.
- Rotating this whole construction by 90/180/270 degrees yields the other three
  visible cases without changing the abstract rule.

b) Complexity calibration
-------------------------
- Only four global situations are generated: separator horizontal/vertical and
  base on either side of that separator. That is enough to make the meta-rule
  visible, without introducing too many branches.
- Connector length is restricted to 2..3 so orientation is visually observable.
- The separator and connector share the same color, but their roles are still
  distinguishable geometrically: one is full-width/full-height, the others are
  short bars embedded in sprites.

c) Hidden assumption audit for main()
-------------------------------------
- Old assumption "foreground has exactly two components" is broken by the new
  separator, so the solver must first detect and remove the full ctrl line.
- Old assumption "base sits at the origin" is invalid. Role selection now depends
  on separator-relative connector direction.
- Old assumption "plug is always above or left of base" is invalid. After fusion,
  plug may end up above, below, left, or right of base depending on which region
  contains the base.
- Old assumption "base free side and plug free side are opposite" is also invalid
  here. In this combo they point in the same global direction, because one sprite
  points toward the separator and the other points away from it.

d) Placement legality
---------------------
- Generator first builds a valid docked pair in one canonical orientation:
  horizontal connectors, base above, plug below.
- It then rotates both sprites together, crops them, and places them into two
  regions separated by a full ctrl line, leaving at least one background row/col
  between every sprite and the separator.
- The canonical plug is grown only inside cells that do not overlap base obj
  pixels in the docked state; overlap is allowed only where plug obj covers the
  base ctrl bar.

e) Visual observability
-----------------------
- Across examples, humans can see the full separator line and infer that its
  orientation controls the docking axis.
- They can also compare which sprite's connector faces the separator versus away
  from it, and observe that only the separator-facing one remains fixed as base.
- Because the four rotated layouts are all generated, these cues are repeated in
  several observable forms rather than hidden in one rare corner case.

f) Visual cleanliness
---------------------
- Each region contains exactly one sprite.
- The separator is isolated by background and never touches a sprite.
- Only three visual primitives appear in the input: one sprite on each side plus
  one full separator line.

g) Change inferability
----------------------
- Color behavior stays simple: ctrl marks both the separator and connectors, and
  only the connector pixels disappear in the output because the separator is not
  copied into the merged crop at all.
- No hidden ordering by color id or component size is used; every decision is
  based on visible geometry.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: there is exactly one full ctrl separator row or column.
  Guarantee: generator inserts exactly one such line.
- Assumption: each side of the separator contains exactly one 8-connected sprite.
  Guarantee: generator pastes exactly one cropped sprite per region.
- Assumption: each connector is a straight 2..3 pixel bar whose free side can be
  classified unambiguously.
  Guarantee: generator constructs connectors with explicit body/free-side support.
- Assumption: docking overlap happens only at base ctrl cells.
  Guarantee: canonical plug growth uses an allowed-mask that forbids all other
  overlaps with base obj cells.
"""


DIR = tuple[int, int]


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in pts}
    return shape, (r0, c0)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _find_separator(grid: np.ndarray, ctrl: int) -> tuple[str, int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[int(r), :] == int(ctrl))]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, int(c)] == int(ctrl))]
    assert (len(rows) == 1) ^ (len(cols) == 1)
    if rows:
        return "h", rows[0]
    return "v", cols[0]


def _sprite_from_region(
    grid: np.ndarray, row_slice: slice, col_slice: slice, bg: int
) -> set[tuple[int, int]]:
    sub = grid[row_slice, col_slice]
    lab, n = label_mask(sub != int(bg), connectivity=8)
    assert int(n) == 1
    r_off = int(row_slice.start or 0)
    c_off = int(col_slice.start or 0)
    return {(int(r + r_off), int(c + c_off)) for r, c in np.argwhere(lab == 1)}


def _connector_axis(conn_pts: set[tuple[int, int]]) -> str:
    shape, _anchor = _norm_pts(conn_pts)
    rows = {r for r, _c in shape}
    cols = {c for _r, c in shape}
    if len(rows) == 1 and len(cols) >= 2:
        return "h"
    if len(cols) == 1 and len(rows) >= 2:
        return "v"
    raise AssertionError("connector is not a visible straight bar")


def _all_has(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) in body for r, c in pts)


def _all_free(body: set[tuple[int, int]], pts: set[tuple[int, int]], dr: int, dc: int) -> bool:
    return all((int(r + dr), int(c + dc)) not in body for r, c in pts)


def _free_dir(sprite: set[tuple[int, int]], conn_pts: set[tuple[int, int]], axis: str) -> DIR:
    body = sprite - conn_pts
    options: list[DIR] = []
    if axis == "h":
        if _all_has(body, conn_pts, -1, 0) and _all_free(body, conn_pts, 1, 0):
            options.append((1, 0))
        if _all_has(body, conn_pts, 1, 0) and _all_free(body, conn_pts, -1, 0):
            options.append((-1, 0))
    else:
        if _all_has(body, conn_pts, 0, -1) and _all_free(body, conn_pts, 0, 1):
            options.append((0, 1))
        if _all_has(body, conn_pts, 0, 1) and _all_free(body, conn_pts, 0, -1):
            options.append((0, -1))
    assert len(options) == 1
    return options[0]


def _toward_separator(region: str, sep_axis: str) -> DIR:
    if sep_axis == "h":
        return (1, 0) if region == "top" else (-1, 0)
    return (0, 1) if region == "left" else (0, -1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Detect the full ctrl separator.
    - Its orientation determines connector orientation and docking axis.
    - The sprite whose connector free side points toward the separator is the base.
    - Translate the other sprite so its connector lies one cell farther along the
      same free-side direction, then recolor ctrl->obj and tight-crop.
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    sep_axis, sep = _find_separator(g, ctrl)
    if sep_axis == "h":
        region_sprites = [
            ("top", _sprite_from_region(g, slice(0, int(sep)), slice(0, g.shape[1]), bg)),
            (
                "bottom",
                _sprite_from_region(g, slice(int(sep) + 1, g.shape[0]), slice(0, g.shape[1]), bg),
            ),
        ]
        conn_axis = "h"
    else:
        region_sprites = [
            ("left", _sprite_from_region(g, slice(0, g.shape[0]), slice(0, int(sep)), bg)),
            (
                "right",
                _sprite_from_region(g, slice(0, g.shape[0]), slice(int(sep) + 1, g.shape[1]), bg),
            ),
        ]
        conn_axis = "v"

    pieces = []
    for region, sprite in region_sprites:
        conn = {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}
        assert conn
        axis = _connector_axis(conn)
        assert axis == conn_axis
        shape, anchor = _norm_pts(conn)
        free = _free_dir(sprite, conn, axis)
        toward = _toward_separator(region, sep_axis)
        if free == toward:
            role = "base"
        else:
            assert free == (-toward[0], -toward[1])
            role = "plug"
        pieces.append(
            {
                "region": region,
                "sprite": sprite,
                "conn": conn,
                "shape": shape,
                "anchor": anchor,
                "free": free,
                "role": role,
            }
        )

    assert {piece["role"] for piece in pieces} == {"base", "plug"}
    base = next(piece for piece in pieces if piece["role"] == "base")
    plug = next(piece for piece in pieces if piece["role"] == "plug")
    assert base["shape"] == plug["shape"]
    assert base["free"] == plug["free"]

    d = base["free"]
    dr0, dc0 = base["anchor"]
    pr0, pc0 = plug["anchor"]
    tr = int(dr0 + d[0] - pr0)
    tc = int(dc0 + d[1] - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug["sprite"]}

    pts_all = base["sprite"] | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base["sprite"]:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return _crop_non_bg(out, bg)


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.append((nr, nc))
        count += 1


def _grow_shape_allowed(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allowed: np.ndarray,
) -> bool:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h
            and 0 <= c + dc < w
            and bool(allowed[r + dr, c + dc])
            and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.append((nr, nc))
        count += 1
    return count == int(target)


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        key = (int(r), int(c))
        if key in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _build_canonical_sprites() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Canonical docked geometry:
    - horizontal connectors
    - base above, free side down
    - plug below, also free side down
    - plug connector sits one row below base connector
    """
    bg, obj, ctrl = 1, 4, 7

    base_grid: np.ndarray | None = None
    conn_row = -1
    conn_cols: list[int] = []
    conn_len = -1
    for _ in range(40):
        base_h = _rng.randint(3, 6)
        base_w = _rng.randint(4, 8)
        base_target = _rng.randint(6, min(22, base_h * base_w))
        base = np.zeros((base_h, base_w), dtype=bool)
        base[0, 0] = True
        _grow_shape(base, base_target, [(0, 0)])
        if not is_contiguous(base, background=0, connectivity=8):
            continue

        conn_len = _rng.randint(2, min(3, base_w))
        candidates: list[tuple[int, list[int]]] = []
        for r in range(1, base_h):
            for c0 in range(0, base_w - conn_len + 1):
                cols = list(range(c0, c0 + conn_len))
                if not all(base[r, c] for c in cols):
                    continue
                if not all(base[r - 1, c] for c in cols):
                    continue
                if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                    continue
                candidates.append((int(r), cols))
        if not candidates:
            continue

        conn_row, conn_cols = _rng.choice(candidates)
        keep_base = {(conn_row, c) for c in conn_cols} | {(conn_row - 1, c) for c in conn_cols}
        _thin(base, keep=keep_base)

        if not all(base[conn_row, c] for c in conn_cols):
            continue
        if not all(base[conn_row - 1, c] for c in conn_cols):
            continue
        if any(conn_row + 1 < base_h and base[conn_row + 1, c] for c in conn_cols):
            continue

        base_grid = np.full((base_h, base_w), int(bg), dtype=int)
        base_grid[base] = int(obj)
        for c in conn_cols:
            base_grid[conn_row, c] = int(ctrl)
        if is_contiguous(base_grid != bg, background=0, connectivity=8):
            break
        base_grid = None
    if base_grid is None:
        return None

    for _ in range(40):
        plug_h = _rng.randint(2, 6)
        plug_w = _rng.randint(max(conn_len, 2), min(8, base_grid.shape[1] + 4))
        plug_conn_c0 = _rng.randint(0, plug_w - conn_len)

        plug = np.zeros((plug_h, plug_w), dtype=bool)
        keep_plug: set[tuple[int, int]] = set()
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug[plug_h - 1, c] = True
            plug[plug_h - 2, c] = True
            keep_plug.add((plug_h - 1, c))
            keep_plug.add((plug_h - 2, c))

        tr = int(conn_row + 1 - (plug_h - 1))
        tc = int(conn_cols[0] - plug_conn_c0)
        allowed = np.ones((plug_h, plug_w), dtype=bool)
        for r in range(plug_h):
            for c in range(plug_w):
                ar = int(tr + r)
                ac = int(tc + c)
                is_connector = r == plug_h - 1 and plug_conn_c0 <= c < plug_conn_c0 + conn_len
                if 0 <= ar < base_grid.shape[0] and 0 <= ac < base_grid.shape[1]:
                    if int(base_grid[ar, ac]) == int(obj):
                        allowed[r, c] = False
                    elif int(base_grid[ar, ac]) == int(ctrl) and is_connector:
                        allowed[r, c] = False

        if not all(bool(allowed[r, c]) for r, c in keep_plug):
            continue

        cap = min(24, int(np.sum(allowed)))
        if cap < 5:
            continue
        target = _rng.randint(max(5, conn_len + 2), cap)
        if not _grow_shape_allowed(plug, target, list(zip(*np.where(plug))), allowed):
            continue
        if not is_contiguous(plug, background=0, connectivity=8):
            continue

        _thin(plug, keep=keep_plug)
        if not all(plug[plug_h - 2, c] for c in range(plug_conn_c0, plug_conn_c0 + conn_len)):
            continue

        plug_grid = np.full((plug_h, plug_w), int(bg), dtype=int)
        plug_grid[plug] = int(obj)
        for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
            plug_grid[plug_h - 1, c] = int(ctrl)
        if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
            continue

        bad = False
        for r in range(plug_h):
            for c in range(plug_w):
                if int(plug_grid[r, c]) == int(bg):
                    continue
                ar = int(tr + r)
                ac = int(tc + c)
                if 0 <= ar < base_grid.shape[0] and 0 <= ac < base_grid.shape[1]:
                    if int(base_grid[ar, ac]) == int(obj):
                        bad = True
                    if (
                        r == plug_h - 1
                        and plug_conn_c0 <= c < plug_conn_c0 + conn_len
                        and int(base_grid[ar, ac]) != int(bg)
                    ):
                        bad = True
                if bad:
                    break
            if bad:
                break
        if bad:
            continue

        base_sprite = _crop_non_bg(base_grid, bg)
        plug_sprite = _crop_non_bg(plug_grid, bg)
        if int(np.sum(base_sprite != bg)) < 5 or int(np.sum(plug_sprite != bg)) < 5:
            continue
        return base_sprite, plug_sprite

    return None


def _rot_vec_ccw(v: DIR, k: int) -> DIR:
    out = v
    for _ in range(int(k) % 4):
        out = (-out[1], out[0])
    return out


def _place_with_separator(base_sprite: np.ndarray, plug_sprite: np.ndarray, d: DIR) -> np.ndarray | None:
    bg, ctrl = 1, 7
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    if d in {(1, 0), (-1, 0)}:
        top_sprite = base_sprite if d == (1, 0) else plug_sprite
        bottom_sprite = plug_sprite if d == (1, 0) else base_sprite
        th, tw = map(int, top_sprite.shape)
        lh, lw = map(int, bottom_sprite.shape)

        min_h = max(9, th + lh + 3)
        min_w = max(4, tw, lw)
        if min_h > 15 or min_w > 14:
            return None
        canvas_h = _rng.randint(min_h, 15)
        canvas_w = _rng.randint(min_w, 14)

        sep_min = th + 1
        sep_max = canvas_h - lh - 2
        if sep_min > sep_max:
            return None
        sep = _rng.randint(sep_min, sep_max)
        top_r = _rng.randint(0, sep - th - 1)
        bottom_r = _rng.randint(sep + 2, canvas_h - lh)
        top_c = _rng.randint(0, canvas_w - tw)
        bottom_c = _rng.randint(0, canvas_w - lw)

        canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
        canvas[sep, :] = int(ctrl)
        paste(canvas, top_sprite, top_r, top_c, transparent=bg)
        paste(canvas, bottom_sprite, bottom_r, bottom_c, transparent=bg)
        return canvas

    left_sprite = base_sprite if d == (0, 1) else plug_sprite
    right_sprite = plug_sprite if d == (0, 1) else base_sprite
    lh, lw = map(int, left_sprite.shape)
    rh, rw = map(int, right_sprite.shape)

    min_h = max(4, lh, rh)
    min_w = max(9, lw + rw + 3)
    if min_h > 14 or min_w > 15:
        return None
    canvas_h = _rng.randint(min_h, 14)
    canvas_w = _rng.randint(min_w, 15)

    sep_min = lw + 1
    sep_max = canvas_w - rw - 2
    if sep_min > sep_max:
        return None
    sep = _rng.randint(sep_min, sep_max)
    left_c = _rng.randint(0, sep - lw - 1)
    right_c = _rng.randint(sep + 2, canvas_w - rw)
    left_r = _rng.randint(0, canvas_h - lh)
    right_r = _rng.randint(0, canvas_h - rh)

    canvas = np.full((canvas_h, canvas_w), int(bg), dtype=int)
    canvas[:, sep] = int(ctrl)
    paste(canvas, left_sprite, left_r, left_c, transparent=bg)
    paste(canvas, right_sprite, right_r, right_c, transparent=bg)
    return canvas


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical_sprites()
    if canonical is None:
        return None
    base_sprite, plug_sprite = canonical

    k = _rng.choice([0, 1, 2, 3])
    base_sprite = np.rot90(base_sprite, k)
    plug_sprite = np.rot90(plug_sprite, k)
    d = _rot_vec_ccw((1, 0), k)

    canvas = _place_with_separator(base_sprite, plug_sprite, d)
    if canvas is None:
        return None
    if np.sum(canvas == 1) <= canvas.size // 2:
        return None

    try:
        out = main(canvas)
    except Exception:
        return None
    if out.size == 0 or not np.any(out != 1):
        return None
    if out.shape[0] > 12 or out.shape[1] > 12:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(800):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("generate_input: failed after 800 attempts")
