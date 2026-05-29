# Variant of 20270e3b
# Dimension: D11 — 对接时的旋转/翻转 (transform)
# Change: plug仅做平移对接，无旋转或翻转... → plug需旋转90度后对接

from __future__ import annotations

"""
Design analysis for the D11 variant.

spec_variant:
  core_change:
    original: plug sprite is already in its docking orientation; solver only translates it.
    variant: plug is shown in a non-identity 90-degree multiple rotation and must be
      rotated back (90/180/270) before the same docking translation is applied.
  concrete_visualization:
    - Colors stay fixed: bg=1, obj=4, ctrl=7.
    - Base stays at the input top-left and still carries a horizontal connector whose
      body is above and free side is below.
    - Plug is still the same kind of thin sprite, but in the input it has already been
      rotated by 1/2/3 quarter-turns, so its connector is either vertical or upside-down.
    - Output is still the tight crop of the assembled shape with ctrl recolored to obj.
  behavior_trace:
    original_small_case:
      - base connector row = 4, cols 2..3
      - plug connector is already top horizontal at local row 0, cols 1..2
      - solver shifts plug by (base_r-1-plug_r, base_c-plug_c)
    variant_small_case:
      - canonical plug is the same, but input stores rot90(plug)
      - solver must try rot90(input_plug, k), k in {1,2,3}
      - only one rotated candidate has:
          1. the same normalized connector bar as the base
          2. connector free side upward and body directly below
          3. overlap exactly on base connector cells after translation
      - then the old translation + recolor rule runs unchanged
    difference:
      - the changed step is not the final placement, but the pre-placement orientation
        search on the plug sprite.
  complexity_calibration:
    - Keep exactly two sprites, fixed colors, same docking concept.
    - Only one new degree of freedom is introduced: plug rotation in {90,180,270}.
    - To keep the rotation observable, connector length is restricted to 2-3 so a
      90/270 rotation visibly changes the connector from horizontal to vertical.
    - Plug shapes are filtered by oracle uniqueness so the rotation is not ambiguous.
  implicit_assumption_audit:
    - Old code assumed connA and connB have identical normalized shapes before any
      transform; that fails for 90/270-rotated plugs, so matching must move after
      rotation enumeration.
    - Old code assumed plug connector free side is already upward; that is no longer
      true on the raw input and must be checked on each rotated candidate instead.
    - Old code stamped the translated plug directly from input coordinates; now the
      rendered pixels must come from the rotated candidate array, not the raw input.
    - Old overlap assertion allowed any overlap touching ctrl. In this variant the
      generator guarantees a stricter assembled geometry: overlap must be exactly the
      base connector cells, with base ctrl covered by plug obj.
  placement_legality:
    - Generator still builds the canonical assembled pair first, then chooses a
      non-identity input rotation for the plug.
    - Plug placement on the input canvas uses padding=1 under 8-connectivity so the
      two sprites remain separate connected components.
    - Oracle check is strengthened to main(input) == expected_output, not merely
      “main does not crash”.
  visual_observability:
    - Humans can see the base connector is horizontal, while the plug connector is
      often vertical or upside-down in the input.
    - The output always shows the same compact assembled orientation, so the “rotate
      the plug until it fits, then dock it” rule is visible from examples.
  cleanliness:
    - No extra markers, colors, or distractor objects are added.
    - The only overlap in the assembled result is the original two-row seam:
      plug-connector over base-body and plug-body over base-connector.
  color_inference:
    - No new color mapping is introduced; ctrl still disappears by recoloring to obj.

New solver assumptions and generator guarantees:
  - exactly two 8-connected foreground components
  - base bbox top-left is at the input origin
  - each sprite has exactly one ctrl connector component
  - base connector is horizontal and downward-facing
  - among plug rotations {1,2,3}, exactly one yields a valid docking geometry
  - generate_input() explicitly filters for those conditions
"""

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


BG = 1
OBJ = 4
CTRL = 7


def _crop_non_bg(arr: np.ndarray, bg: int = BG) -> np.ndarray | None:
    nz = np.argwhere(arr != int(bg))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _sprite_from_points(grid: np.ndarray, pts: set[tuple[int, int]], bg: int = BG) -> tuple[np.ndarray, tuple[int, int]]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in pts:
        out[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return out, (int(r0), int(c0))


def _conn_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr == int(CTRL))}


def _body_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr == int(OBJ))}


def _all_fg_coords(arr: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in np.argwhere(arr != int(BG))}


def _norm_shape(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _assemble_docked(base_arr: np.ndarray, plug_arr: np.ndarray) -> np.ndarray | None:
    base_conn = _conn_coords(base_arr)
    plug_conn = _conn_coords(plug_arr)
    if not base_conn or not plug_conn:
        return None

    base_shape, (br0, bc0) = _norm_shape(base_conn)
    plug_shape, (pr0, pc0) = _norm_shape(plug_conn)
    if base_shape != plug_shape:
        return None

    base_body = _body_coords(base_arr)
    plug_body = _body_coords(plug_arr)

    for r, c in base_conn:
        if (int(r - 1), int(c)) not in base_body:
            return None
        if (int(r + 1), int(c)) in base_body:
            return None

    for r, c in plug_conn:
        if (int(r - 1), int(c)) in plug_body:
            return None
        if (int(r + 1), int(c)) not in plug_body:
            return None

    tr = int((br0 - 1) - pr0)
    tc = int(bc0 - pc0)

    shifted_vals: dict[tuple[int, int], int] = {}
    overlap_ctrl = set()
    overlap_body = set()
    base_above = {(int(r - 1), int(c)) for r, c in base_conn}
    for r, c in _all_fg_coords(plug_arr):
        rr, cc = int(r + tr), int(c + tc)
        val = int(plug_arr[int(r), int(c)])
        shifted_vals[(rr, cc)] = val
        if 0 <= rr < base_arr.shape[0] and 0 <= cc < base_arr.shape[1]:
            cur = int(base_arr[int(rr), int(cc)])
            if cur != int(BG):
                if cur == int(OBJ) and val == int(CTRL) and (rr, cc) in base_above:
                    overlap_ctrl.add((int(rr), int(cc)))
                    continue
                if cur == int(CTRL) and val == int(OBJ) and (rr, cc) in base_conn:
                    overlap_body.add((int(rr), int(cc)))
                    continue
                return None

    if overlap_ctrl != base_above or overlap_body != base_conn:
        return None

    pts_all = _all_fg_coords(base_arr) | set(shifted_vals)
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    if rr0 != 0 or cc0 != 0:
        return None

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)
    paste(out, base_arr, 0, 0, transparent=BG)

    for (rr, cc), val in shifted_vals.items():
        cur = int(out[int(rr - rr0), int(cc - cc0)])
        if cur != int(BG):
            if cur == int(OBJ) and val == int(CTRL) and (rr, cc) in base_above:
                out[int(rr - rr0), int(cc - cc0)] = int(val)
                continue
            if cur == int(CTRL) and val == int(OBJ) and (rr, cc) in base_conn:
                out[int(rr - rr0), int(cc - cc0)] = int(val)
                continue
            return None
        out[int(rr - rr0), int(cc - cc0)] = int(val)

    out[out == int(CTRL)] = int(OBJ)
    return _crop_non_bg(out, BG)


def _candidate_outputs(base_arr: np.ndarray, plug_input_arr: np.ndarray) -> list[tuple[int, np.ndarray]]:
    candidates: list[tuple[int, np.ndarray]] = []
    for rot_k in (1, 2, 3):
        rotated = _crop_non_bg(np.rot90(plug_input_arr, int(rot_k)), BG)
        if rotated is None:
            continue
        assembled = _assemble_docked(base_arr, rotated)
        if assembled is not None:
            candidates.append((int(rot_k), assembled))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the plug by a non-identity quarter-turn until its connector can dock,
    then apply the original vertical docking translation and tight-crop.
    """
    g = np.array(grid, dtype=int)

    lab, n = label_mask(g != int(BG), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = comps

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0

    if int(r0a + c0a) <= int(r0b + c0b):
        base_pts, plug_pts = A, B
    else:
        base_pts, plug_pts = B, A

    base_arr, (base_r0, base_c0) = _sprite_from_points(g, base_pts, BG)
    plug_arr, _ = _sprite_from_points(g, plug_pts, BG)
    assert int(base_r0) == 0 and int(base_c0) == 0

    candidates = _candidate_outputs(base_arr, plug_arr)
    assert len(candidates) == 1
    return candidates[0][1]


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[int(r + dr), int(c + dc)]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        count += 1
        frontier.append((int(nr), int(nc)))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove interior pixels to make a thin-walled shape."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r, c = int(r), int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[int(r + dr), int(c + dc)]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _gen_attempt() -> np.ndarray | None:
    bg, obj, ctrl = BG, OBJ, CTRL

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None

    conn_len = _rng.randint(2, min(3, base_w))
    candidates = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[int(r), int(c)] for c in cols):
                continue
            if not all(base[int(r - 1), int(c)] for c in cols):
                continue
            candidates.append((int(r), cols))
    if not candidates:
        return None

    conn_row, conn_cols = None, None
    _rng.shuffle(candidates)
    for cand_row, cand_cols in candidates:
        carved = base.copy()
        for c in cand_cols:
            if cand_row + 1 < base_h:
                carved[int(cand_row + 1), int(c)] = False
        if not carved[0, 0]:
            continue
        if not is_contiguous(carved, background=0, connectivity=8):
            continue
        if int(carved.sum()) < 5:
            continue
        base = carved
        conn_row, conn_cols = int(cand_row), cand_cols
        break
    if conn_row is None or conn_cols is None:
        return None

    keep_base = {(0, 0)}
    keep_base |= {(int(conn_row), int(c)) for c in conn_cols}
    keep_base |= {(int(conn_row - 1), int(c)) for c in conn_cols}
    _thin(base, keep=keep_base)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(3, 7)
    plug_w = _rng.randint(max(conn_len + 1, 3), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, min(plug_w - conn_len, conn_cols[0]))
    dc_offset = int(conn_cols[0] - plug_conn_c0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, int(c)] = True
        if plug_h < 2:
            return None
        plug[1, int(c)] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(max(5, conn_len + 2), plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    # Keep the seam geometry clean: row 0 is connector only, row 1 is the matching body only.
    for c in range(plug_w):
        if not (plug_conn_c0 <= c < plug_conn_c0 + conn_len):
            plug[0, int(c)] = False
            plug[1, int(c)] = False

    # Carve away any canonical-state collisions outside the intended 2-row docking seam.
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[int(r), int(c)]:
                continue
            dr = int(conn_row - 1 + r)
            dc = int(dc_offset + c)
            if not (0 <= dr < base_h and 0 <= dc < base_w):
                continue
            if not base[int(dr), int(dc)]:
                continue
            allowed_ctrl = r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len and dr == conn_row - 1
            allowed_body = r == 1 and plug_conn_c0 <= c < plug_conn_c0 + conn_len and dr == conn_row
            if not (allowed_ctrl or allowed_body):
                plug[int(r), int(c)] = False
    if not is_contiguous(plug, background=0, connectivity=8):
        return None

    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    keep_plug |= {(1, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, int(c)] = ctrl

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_grid[1, int(c)] != obj:
            return None
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    for r in range(plug_h):
        for c in range(plug_w):
            if plug_grid[int(r), int(c)] == bg:
                continue
            dr = int(conn_row - 1 + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[int(dr), int(dc)] != bg:
                if base_grid[int(dr), int(dc)] == obj and plug_grid[int(r), int(c)] == ctrl and r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                if base_grid[int(dr), int(dc)] == ctrl and plug_grid[int(r), int(c)] == obj and r == 1 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    if dc_offset < 0:
        return None

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid, bg)
    plug_sprite = _crop_non_bg(plug_grid, bg)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = base_sprite.shape
    ph0, pw0 = plug_sprite.shape
    plug_mask = plug_sprite != bg
    if ph0 == pw0:
        return None
    if np.array_equal(plug_mask, np.rot90(plug_mask, 2)):
        return None

    expected = _assemble_docked(base_sprite, plug_sprite)
    if expected is None:
        return None

    rot_choices: list[tuple[int, np.ndarray]] = []
    for input_rot in (1, 2, 3):
        rotated_input = _crop_non_bg(np.rot90(plug_sprite, int(input_rot)), bg)
        if rotated_input is None:
            continue
        found = _candidate_outputs(base_sprite, rotated_input)
        need_rot = int((-input_rot) % 4)
        if len(found) == 1 and found[0][0] == need_rot and np.array_equal(found[0][1], expected):
            rot_choices.append((int(input_rot), rotated_input))
    if not rot_choices:
        return None

    _input_rot, plug_input = _rng.choice(rot_choices)
    ph, pw = plug_input.shape

    h_min = max(7, bh, ph)
    w_min = max(3, bw, pw)
    if h_min > 11 or w_min > 13:
        return None
    canvas_h = _rng.randint(h_min, 11)
    canvas_w = _rng.randint(w_min, 13)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_input, transparent=bg, padding=1, padding_connectivity=8)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_input, int(pr), int(pc), transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 400 attempts")
