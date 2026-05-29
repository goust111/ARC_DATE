# Combo variant of 20270e3b (index 15)
# combo_spec: D3/2+D14/2
# Name: 干扰噪声分隔线

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete behavior and trace
#    - Start from the original docking task: two obj=4 sprites each carry a short
#      horizontal ctrl=7 connector. The plug docks above the base after connector
#      alignment.
#    - Apply D14/2 first at the scene-layout level: the input is split by one full
#      ctrl separator line. A horizontal separator means the real docking is vertical;
#      a vertical separator is handled as a 90-degree rotation of that same case.
#    - Apply D3/2 inside each separated region: besides the one real ctrl-bearing
#      sprite, the region may also contain a tiny detached noise blob in another
#      color. The solver must ignore those distractors and only keep the unique
#      legal sprite with obj/ctrl colors and a valid horizontal connector.
#    - This combo also gives the separator a post-transform role. In canonical
#      horizontal-separator form, after docking we keep one full separator row
#      between the two connector rows. All ctrl pixels from both connectors and the
#      separator recolor to obj, so the output visibly contains a solid body seam.
#    - Small trace:
#         old input  : [base, plug] with no divider, no noise
#         old output : plug shifted so plug-connector sits above base-connector;
#                      ctrl disappears.
#         combo input: [top region: plug + noise] + [full ctrl divider] +
#                      [bottom region: base + noise]
#         combo solve: detect divider -> canonicalize by rotation if needed ->
#                      in each region scan components, keep only the legal
#                      ctrl-bearing sprite -> align connectors -> insert divider
#                      as the seam row/col -> recolor all ctrl to obj -> crop.
#         key delta  : region membership replaces proximity; noise is filtered;
#                      separator orientation controls whether the merged result is
#                      tall or wide; separator survives as a recolored seam.
#
# b) Complexity calibration
#    - The separator is always present and sampled in both orientations, so the
#      D14 change is visually obvious across examples.
#    - Noise is always present in both regions, but kept tiny (1..3 cells) and in
#      colors outside {bg,obj,ctrl}; this makes the D3 change observable without
#      creating matching ambiguity.
#    - Connector length remains 1..3 with identical shape on both sprites. That
#      keeps the docking cue repeated and constrained.
#
# c) Old implicit assumptions that break and how the new solver handles them
#    - `len(best_comps) == 2` is false: foreground now includes the separator and
#      two noise blobs. New code detects/removes the full ctrl separator first, then
#      reasons region-by-region instead of globally counting components.
#    - "base is closest to (0,0)" is false: roles are now determined by region
#      placement relative to the separator. In canonical form, top = plug and
#      bottom = base.
#    - "all non-bg components use obj/ctrl colors" is false because noise uses other
#      colors. Candidate extraction therefore rejects any component containing colors
#      outside {obj,ctrl}.
#    - "separator is irrelevant to output" is false for this combo. The solver
#      explicitly inserts a full seam line during assembly and recolors it to obj.
#    - Rotation branch adds a new assumption: vertical-separator inputs are exactly a
#      rotated copy of the horizontal canonical case. The generator guarantees that.
#
# d) Placement legality
#    - Generator builds a canonical horizontal sample first:
#         * exactly one plug sprite above the separator and one base sprite below it
#         * at least one background row between each sprite/noise blob and separator
#         * noise blobs are placed with padding=1 so they stay detached under
#           8-connectivity
#    - Vertical-separator samples are produced by rotating a legal horizontal sample,
#      so left/right legality is inherited.
#    - Oracle verification checks `main(input) == expected_output`.
#
# e) Visual observability
#    - A human sees one ctrl line splitting the board into two regions and also sees
#      extra colored specks that vanish in the output.
#    - Across examples, horizontal input dividers produce tall outputs with a full
#      horizontal obj seam; vertical dividers produce wide outputs with a full
#      vertical obj seam. The separator direction is therefore recoverable from
#      2-3 demonstrations.
#
# f) Visual neatness
#    - Only one layer of symbolism is added per region: one main sprite and one tiny
#      noise blob. The separator is isolated by background and never touches those
#      objects in the input.
#    - Noise never uses obj or ctrl, so it is visually distinct from the real
#      docking structures.
#
# g) Color inferability
#    - Main rule colors remain fixed: bg=1, obj=4, ctrl=7.
#    - Noise uses a small palette {2,3,5,6,8,9} and is always discarded, which is
#      inferable because those colors disappear while the obj/ctrl structures drive
#      the transformation.
#
# h) New solver assumptions and generator guarantees
#    - Exactly one full-row or full-column ctrl separator exists.
#    - After canonicalization, each separator side contains exactly one legal
#      ctrl-bearing sprite and at least one detached non-{obj,ctrl} noise blob.
#    - The plug connector faces up (body below, free side above) and the base
#      connector faces down (body above, free side below).
#    - Matching connector shapes are unique.
#    - The canonical output seam row lies exactly between the two connector rows.
# -----------------------------------------------------------------------------


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [int(r) for r, _c in pts]
    cs = [int(c) for _r, c in pts]
    r0, c0 = min(rs), min(cs)
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (int(r0), int(c0))


def _find_separator(grid: np.ndarray, ctrl: int) -> tuple[str, int]:
    rows = [int(r) for r in range(grid.shape[0]) if np.all(grid[int(r), :] == int(ctrl))]
    cols = [int(c) for c in range(grid.shape[1]) if np.all(grid[:, int(c)] == int(ctrl))]
    assert (len(rows) == 1) ^ (len(cols) == 1)
    if rows:
        return "h", rows[0]
    return "v", cols[0]


def _connector_meta(
    g: np.ndarray, pts: set[tuple[int, int]], obj: int, ctrl: int
) -> dict[str, object] | None:
    if any(int(g[r, c]) not in (int(obj), int(ctrl)) for r, c in pts):
        return None
    ctrl_pts = {(int(r), int(c)) for (r, c) in pts if int(g[r, c]) == int(ctrl)}
    if not ctrl_pts:
        return None

    rows = {int(r) for r, _c in ctrl_pts}
    if len(rows) != 1:
        return None
    cols = sorted(int(c) for _r, c in ctrl_pts)
    if len(cols) > 3 or cols != list(range(cols[0], cols[-1] + 1)):
        return None

    body = pts - ctrl_pts
    faces_down = all((int(r - 1), int(c)) in body for r, c in ctrl_pts) and all(
        (int(r + 1), int(c)) not in body for r, c in ctrl_pts
    )
    faces_up = all((int(r + 1), int(c)) in body for r, c in ctrl_pts) and all(
        (int(r - 1), int(c)) not in body for r, c in ctrl_pts
    )
    if faces_down == faces_up:
        return None

    shape, anchor = _norm_pts(ctrl_pts)
    return {
        "pts": pts,
        "conn": ctrl_pts,
        "shape": shape,
        "anchor": anchor,
        "bbox": _bbox_of_pts(pts),
        "orientation": "base" if faces_down else "plug",
    }


def _region_sprite_meta(
    g: np.ndarray, row_slice: slice, col_slice: slice, bg: int, obj: int, ctrl: int
) -> dict[str, object]:
    sub = g[row_slice, col_slice]
    lab, n = label_mask(sub != int(bg), connectivity=8)
    metas: list[dict[str, object]] = []
    r_off = int(row_slice.start or 0)
    c_off = int(col_slice.start or 0)
    for k in range(1, int(n) + 1):
        pts = {
            (int(r + r_off), int(c + c_off))
            for r, c in np.argwhere(lab == int(k))
        }
        meta = _connector_meta(g, pts, obj, ctrl)
        if meta is not None:
            metas.append(meta)
    assert len(metas) == 1
    return metas[0]


def _assemble_with_separator(
    g: np.ndarray,
    plug_meta: dict[str, object],
    base_meta: dict[str, object],
    *,
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray:
    plug = set(plug_meta["pts"])
    base = set(base_meta["pts"])
    connP = set(plug_meta["conn"])
    connD = set(base_meta["conn"])

    shpP, (pr0, pc0) = _norm_pts(connP)
    shpD, (dr0, dc0) = _norm_pts(connD)
    assert shpP == shpD
    seam_row = int(dr0 - 1)
    tr = int((seam_row - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[r, c])
    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl) or int(r) == int(seam_row)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[int(seam_row - rr0), :] = int(ctrl)
    out[out == int(ctrl)] = int(obj)
    return _crop_non_bg(out, bg)


def _solve_canonical(grid: np.ndarray) -> np.ndarray:
    """
    Canonical form:
    - one horizontal full-ctrl separator row
    - plug region above, base region below
    - each region may also contain detached noise blobs
    - merged output keeps a recolored full seam row derived from the separator
    """
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    orient, sep = _find_separator(g, ctrl)
    assert orient == "h"
    assert 0 < int(sep) < g.shape[0] - 1

    plug_meta = _region_sprite_meta(g, slice(0, int(sep)), slice(0, g.shape[1]), bg, obj, ctrl)
    base_meta = _region_sprite_meta(
        g, slice(int(sep) + 1, g.shape[0]), slice(0, g.shape[1]), bg, obj, ctrl
    )

    assert plug_meta["orientation"] == "plug"
    assert base_meta["orientation"] == "base"
    return _assemble_with_separator(g, plug_meta, base_meta, bg=bg, obj=obj, ctrl=ctrl)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    _bg, _obj, ctrl = 1, 4, 7
    orient, _sep = _find_separator(g, ctrl)
    if orient == "h":
        return _solve_canonical(g)
    solved = _solve_canonical(np.rot90(g, -1))
    return _crop_non_bg(np.rot90(solved, 1), 1)


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


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
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


def _crop_sprite(grid: np.ndarray, bg: int) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return None, None
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return grid[r0 : r1 + 1, c0 : c1 + 1], (r0, c0)


def _expected_from_sprites(plug_sprite: np.ndarray, base_sprite: np.ndarray) -> np.ndarray:
    bg, obj, ctrl = 1, 4, 7
    ph, pw = map(int, plug_sprite.shape)
    bh, bw = map(int, base_sprite.shape)
    temp_h = int(ph + bh + 6)
    temp_w = int(max(pw, bw) + 6)
    temp = np.full((temp_h, temp_w), int(bg), dtype=int)

    plug_r, plug_c = 1, 2
    base_r, base_c = int(ph + 4), 2
    paste(temp, plug_sprite, plug_r, plug_c, transparent=bg)
    paste(temp, base_sprite, base_r, base_c, transparent=bg)

    plug_meta = {
        "pts": {(int(r), int(c)) for r, c in np.argwhere(temp == obj) if r < base_r}
        | {(int(r), int(c)) for r, c in np.argwhere(temp == ctrl) if r < base_r},
        "conn": {(int(r), int(c)) for r, c in np.argwhere(temp == ctrl) if r < base_r},
    }
    base_meta = {
        "pts": {(int(r), int(c)) for r, c in np.argwhere(temp == obj) if r >= base_r}
        | {(int(r), int(c)) for r, c in np.argwhere(temp == ctrl) if r >= base_r},
        "conn": {(int(r), int(c)) for r, c in np.argwhere(temp == ctrl) if r >= base_r},
    }
    return _assemble_with_separator(temp, plug_meta, base_meta, bg=bg, obj=obj, ctrl=ctrl)


def _build_canonical_sprites() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    bg, obj, ctrl = 1, 4, 7

    base_h = _rng.randint(3, 6)
    base_w = _rng.randint(3, 8)
    base_target = _rng.randint(5, min(22, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(1, base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            if any(r + 1 < base_h and base[r + 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 6)
    plug_w = _rng.randint(max(conn_len, 2), min(8, base_w + 3))
    if plug_h * plug_w < 5:
        return None
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = int(conn_cols[0] - plug_conn_c0)

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        plug[1, c] = True

    plug_target = _rng.randint(5, min(22, plug_h * plug_w))
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep)

    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if not plug[1, c]:
            return None

    seam_row = int(conn_row - 1)
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(seam_row - 1 + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == obj:
                if r == 1 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    base_sprite, base_off = _crop_sprite(base_grid, bg)
    plug_sprite, plug_off = _crop_sprite(plug_grid, bg)
    if base_sprite is None or plug_sprite is None or base_off is None or plug_off is None:
        return None
    if int(np.sum(base_sprite != bg)) < 5 or int(np.sum(plug_sprite != bg)) < 5:
        return None

    expected = _expected_from_sprites(plug_sprite, base_sprite)
    return plug_sprite, base_sprite, expected


def _make_noise_patch(color: int, bg: int) -> np.ndarray:
    options = [
        np.array([[color]], dtype=int),
        np.array([[color, color]], dtype=int),
        np.array([[color], [color]], dtype=int),
        np.array([[color, color], [color, bg]], dtype=int),
        np.array([[color, bg], [color, color]], dtype=int),
    ]
    return np.array(_rng.choice(options), dtype=int)


def _place_noise_in_region(
    canvas: np.ndarray, patch: np.ndarray, row_slice: slice, col_slice: slice, bg: int
) -> bool:
    r0 = int(row_slice.start or 0)
    r1 = int(row_slice.stop)
    c0 = int(col_slice.start or 0)
    c1 = int(col_slice.stop)
    ph, pw = map(int, patch.shape)
    if ph > r1 - r0 or pw > c1 - c0:
        return False

    sub = canvas[r0:r1, c0:c1]
    try:
        nr, nc = random_free_location(sub, patch, transparent=bg, padding=1)
    except ValueError:
        return False
    paste(canvas, patch, int(r0 + nr), int(c0 + nc), transparent=bg)
    return True


def _place_canonical_input(
    plug_sprite: np.ndarray, base_sprite: np.ndarray, expected: np.ndarray
) -> np.ndarray | None:
    bg, ctrl = 1, 7
    ph, pw = map(int, plug_sprite.shape)
    bh, bw = map(int, base_sprite.shape)

    canvas_h = _rng.randint(max(10, ph + bh + 5), min(16, ph + bh + 8))
    canvas_w = _rng.randint(max(5, max(pw, bw) + 1), min(15, max(pw, bw) + 5))
    if canvas_h <= ph + bh + 2 or canvas_w < max(pw, bw):
        return None

    sep_min = ph + 1
    sep_max = canvas_h - bh - 2
    if sep_min > sep_max:
        return None
    sep = _rng.randint(sep_min, sep_max)

    plug_r_max = sep - ph - 1
    base_r_min = sep + 2
    if plug_r_max < 0 or base_r_min > canvas_h - bh:
        return None

    plug_r = _rng.randint(0, plug_r_max)
    base_r = _rng.randint(base_r_min, canvas_h - bh)
    plug_c = _rng.randint(0, canvas_w - pw)
    base_c = _rng.randint(0, canvas_w - bw)

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    canvas[sep, :] = int(ctrl)
    paste(canvas, plug_sprite, plug_r, plug_c, transparent=bg)
    paste(canvas, base_sprite, base_r, base_c, transparent=bg)

    noise_palette = [2, 3, 5, 6, 8, 9]
    top_patch = _make_noise_patch(_rng.choice(noise_palette), bg)
    remaining = [c for c in noise_palette if c != int(top_patch[top_patch != bg][0])]
    bot_patch = _make_noise_patch(_rng.choice(remaining), bg)
    if not _place_noise_in_region(canvas, top_patch, slice(0, max(0, sep - 1)), slice(0, canvas_w), bg):
        return None
    if not _place_noise_in_region(
        canvas, bot_patch, slice(min(canvas_h, sep + 2), canvas_h), slice(0, canvas_w), bg
    ):
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, expected):
        return None
    return canvas


def _gen_attempt() -> np.ndarray | None:
    canonical = _build_canonical_sprites()
    if canonical is None:
        return None
    plug_sprite, base_sprite, expected = canonical
    canvas = _place_canonical_input(plug_sprite, base_sprite, expected)
    if canvas is None:
        return None

    if _rng.random() < 0.5:
        sample = canvas
        target = expected
    else:
        sample = np.rot90(canvas, 1)
        target = np.rot90(expected, 1)

    try:
        out = main(sample)
    except Exception:
        return None
    if not np.array_equal(out, target):
        return None
    return sample


def generate_input() -> np.ndarray:
    for _ in range(600):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 600 attempts")
