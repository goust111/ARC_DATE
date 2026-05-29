# Variant of 20270e3b
# Dimension: D3 — 源对象结构 (src)
# Change: 两个薄壁8连通精灵，各自包含obj像素构成的壁和一个水平ctrl连接器条... → 含干扰像素的精灵

from __future__ import annotations

import random
import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

# Design analysis for the D3 variant
# - Source task: None, so the baseline is the original 20270e3b solver/generator only.
# - a) Concrete behavior:
#   Input still contains one base sprite at the top-left corner and one plug sprite elsewhere.
#   Both sprites still use obj=4 for body and ctrl=7 for a horizontal connector of length 1..3.
#   The new visible change is that the canvas also contains 1 or 2 detached tiny blobs in other colors
#   (sampled from colors excluding bg/obj/ctrl). Those blobs never contain ctrl pixels and never touch
#   the two main sprites. Output ignores those blobs completely and still shows only the docked assembly.
#   Small trace:
#   old input: [base component, plug component] -> take the two components -> dock.
#   new input: [base, plug, noise A, noise B] -> filter components by “valid ctrl-bearing sprite” ->
#   keep only the matching base/plug pair -> dock -> noise disappears from output.
# - b) Complexity calibration:
#   Noise is always present (1 or 2 blobs), so the change is not accidental. The blobs are tiny
#   (1..3 cells) and use non-main colors, keeping the rule legible: humans can see “extra junk exists”
#   but the real action is still identified by the shared orange connector bars. Connector lengths remain
#   in 1..3 with repeated support across samples, so matching is not too free-form.
# - c) Old implicit assumptions that break:
#   `assert len(best_comps) == 2` and “sort components by size, then A/B are the sprites” both rely on
#   there being no extra foreground components. In this variant that is false. The new solver therefore
#   scans every 8-connected component, keeps only components containing a valid horizontal ctrl bar, and
#   selects the unique pair with matching connector shape and opposite docking orientation.
# - d) Placement legality:
#   Generator places base at (0,0), then places plug with padding=1 so the two sprites are definitely
#   separate under 8-connectivity. Each distractor blob is also placed with padding=1 against everything
#   already on the canvas, so no accidental merges occur. Oracle verification checks exact output equality.
# - e) Visual observability:
#   Across 2-4 examples a human can see that extra colored specks/mini-blocks appear in different places
#   and vanish in the output, while the two ctrl-bearing yellow sprites still dock the same way.
# - f) Visual neatness:
#   Noise blobs are tiny, detached, and use at most one color each. They do not overlap or diagonally
#   touch sprites, so the scene stays parseable.
# - g) Color inference:
#   Main-rule colors stay fixed: bg=1, obj=4, ctrl=7. Distractor colors come from a small pool and have
#   no transformation tied to their identity; they are simply ignored because they lack matching connectors.
# - New solver assumptions introduced here and guaranteed by the generator:
#   1. Exactly two components contain ctrl pixels and those two have the same normalized connector shape.
#   2. The base component is the ctrl-bearing component whose bbox top-left satisfies r+c==0.
#   3. Distractor blobs never touch the main sprites, so each blob remains its own non-ctrl component.


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _norm_pts(pts: set[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in pts]
    cs = [c for _r, c in pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = frozenset((int(r - r0), int(c - c0)) for r, c in pts)
    return shape, (r0, c0)


def _connector_meta(
    g: np.ndarray, pts: set[tuple[int, int]], obj: int, ctrl: int
) -> dict[str, object] | None:
    ctrl_pts = {(r, c) for (r, c) in pts if int(g[r, c]) == int(ctrl)}
    if not ctrl_pts:
        return None
    if any(int(g[r, c]) not in (int(obj), int(ctrl)) for r, c in pts):
        return None

    rows = {int(r) for r, _c in ctrl_pts}
    if len(rows) != 1:
        return None
    cols = sorted(int(c) for _r, c in ctrl_pts)
    if len(cols) > 3:
        return None
    if cols != list(range(cols[0], cols[-1] + 1)):
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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj, ctrl = 1, 4, 7

    lab, n = label_mask(g != bg, connectivity=8)
    metas: list[dict[str, object]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        meta = _connector_meta(g, pts, obj, ctrl)
        if meta is not None:
            metas.append(meta)

    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    for i in range(len(metas)):
        for j in range(i + 1, len(metas)):
            a = metas[i]
            b = metas[j]
            if a["shape"] != b["shape"]:
                continue
            a_bbox = a["bbox"]
            b_bbox = b["bbox"]
            assert isinstance(a_bbox, tuple) and isinstance(b_bbox, tuple)
            if a["orientation"] == "base" and b["orientation"] == "plug" and int(a_bbox[0] + a_bbox[1]) == 0:
                pairs.append((a, b))
            if b["orientation"] == "base" and a["orientation"] == "plug" and int(b_bbox[0] + b_bbox[1]) == 0:
                pairs.append((b, a))

    assert len(pairs) == 1
    base_meta, plug_meta = pairs[0]
    base = set(base_meta["pts"])
    plug = set(plug_meta["pts"])
    connD = set(base_meta["conn"])
    connP = set(plug_meta["conn"])

    _sd, (dr0, dc0) = _norm_pts(connD)
    _sp, (pr0, pc0) = _norm_pts(connP)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[r, c])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)

    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    h, w = mask.shape
    keep = keep or set()
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


def _crop_sprite(g: np.ndarray, bg: int) -> tuple[np.ndarray | None, tuple[int, int] | None]:
    nz = np.argwhere(g != int(bg))
    if nz.size == 0:
        return None, None
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return g[r0 : r1 + 1, c0 : c1 + 1], (r0, c0)


def _build_expected_output(
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    base_conn_row: int,
    base_conn_cols: list[int],
    plug_conn_c0: int,
    bg: int,
    obj: int,
    ctrl: int,
) -> np.ndarray:
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    pr = int(base_conn_row - 1)
    pc = int(base_conn_cols[0] - plug_conn_c0)
    assert pr >= 0 and pc >= 0

    out_h = max(int(bh), int(pr + ph))
    out_w = max(int(bw), int(pc + pw))
    out = np.full((out_h, out_w), int(bg), dtype=int)
    paste(out, base_sprite, 0, 0, transparent=bg)
    paste(out, plug_sprite, pr, pc, transparent=bg)
    out[out == int(ctrl)] = int(obj)
    nz = np.argwhere(out != int(bg))
    r0, c0, r1, c1 = map(int, bbox_coords(nz))
    return out[r0 : r1 + 1, c0 : c1 + 1]


def _make_noise_patch(color: int, bg: int) -> np.ndarray:
    options = [
        np.array([[color]], dtype=int),
        np.array([[color, color]], dtype=int),
        np.array([[color], [color]], dtype=int),
        np.array([[color, color], [color, bg]], dtype=int),
        np.array([[color, bg], [color, color]], dtype=int),
    ]
    return np.array(_rng.choice(options), dtype=int)


def _gen_attempt(debug: bool = False) -> np.ndarray | tuple[None, str] | None:
    bg, obj, ctrl = 1, 4, 7
    noise_palette = [2, 3, 5, 6, 8, 9]

    def _fail(reason: str) -> np.ndarray | tuple[None, str] | None:
        if debug:
            return None, reason
        return None

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return _fail("base_disconnected")
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates: list[tuple[int, list[int]]] = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if r == 0 or not all(base[r - 1, c] for c in cols):
                continue
            if r + 1 < base_h and any(base[r + 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return _fail("no_base_connector")
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return _fail("base_grid_disconnected")

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, min(plug_w - conn_len, conn_cols[0]))
    dc_offset = int(conn_cols[0] - plug_conn_c0)
    if dc_offset < 0:
        return _fail("plug_left_of_base")

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return _fail("plug_too_small")
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return _fail("plug_disconnected")
    _thin(plug, keep={(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)})

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(conn_row - 1 + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] != bg:
                allowed_overlap = False
                if c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                    if r == 0 and base_grid[dr, dc] == obj:
                        allowed_overlap = True
                    if r == 1 and base_grid[dr, dc] == ctrl:
                        allowed_overlap = True
                if not allowed_overlap:
                    return _fail("bad_overlap")

    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl
        if plug_h < 2 or plug_grid[1, c] != obj:
            return _fail("plug_connector_support")
    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return _fail("plug_grid_disconnected")

    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return _fail("pixel_count")

    base_sprite, base_off = _crop_sprite(base_grid, bg)
    plug_sprite, plug_off = _crop_sprite(plug_grid, bg)
    if base_sprite is None or plug_sprite is None or base_off is None or plug_off is None:
        return _fail("crop_failed")
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    base_conn_row_crop = int(conn_row - base_off[0])
    base_conn_cols_crop = [int(c - base_off[1]) for c in conn_cols]
    plug_conn_c0_crop = int(plug_conn_c0 - plug_off[1])
    expected = _build_expected_output(
        base_sprite,
        plug_sprite,
        base_conn_row_crop,
        base_conn_cols_crop,
        plug_conn_c0_crop,
        bg,
        obj,
        ctrl,
    )

    n_noise = _rng.randint(1, 2)
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(4, max(bw, pw) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return _fail("canvas_too_small")

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=1)
    except ValueError:
        return _fail("plug_placement")
    if pr + pc == 0:
        return _fail("plug_at_origin")
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    noise_colors = random.sample(noise_palette, n_noise)
    for color in noise_colors:
        patch = _make_noise_patch(color, bg)
        try:
            nr, nc = random_free_location(canvas, patch, transparent=bg, padding=1)
        except ValueError:
            return _fail("noise_placement")
        paste(canvas, patch, nr, nc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 3:
        return _fail("too_dense")

    try:
        result = main(canvas)
    except Exception:
        return _fail("oracle_exception")
    if not np.array_equal(result, expected):
        return _fail("oracle_mismatch")
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
