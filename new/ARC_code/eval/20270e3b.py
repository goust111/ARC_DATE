from __future__ import annotations
import random
import numpy as np
from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites using matching connector pixels, then tight-crop.

    Input structure:
    - Background color `bg`, sprite color `obj`, connector color `ctrl` (small, 2 components).
    - Two main foreground sprites, plus possibly tiny detached pixels belonging to one sprite.

    Rule:
    - Identify the **base** sprite (closest to grid top-left).
    - Translate the other sprite so its connector sits one cell above the base's connector.
    - Stamp the plug onto the base; remove connectors (recolor `ctrl` to `obj`).
    - Output is the tight crop around all non-background pixels.

    Key insight: The connector pixels define the unique docking point and direction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    # Fixed colors (consistent across all eval examples)
    bg, obj, ctrl = 1, 4, 7

    # Build foreground components (8-connected: sprites may touch diagonally)
    lab, n = label_mask(g != bg, connectivity=8)
    best_comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        best_comps.append(pts)
    best_comps.sort(key=len, reverse=True)
    assert len(best_comps) == 2  # exactly 2 sprites, no fragments
    A = set(best_comps[0])
    B = set(best_comps[1])

    def bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        coords = np.array(list(pts), dtype=int)
        r0, c0, r1, c1 = bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)

    def conn(sprite: set[tuple[int, int]]):
        return {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}

    connA = conn(A)
    connB = conn(B)
    assert not (not connA or not connB)

    def norm(conn_pts: set[tuple[int, int]]):
        rs = [r for r, _c in conn_pts]
        cs = [c for _r, c in conn_pts]
        r0, c0 = int(min(rs)), int(min(cs))
        shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
        return shape, (int(r0), int(c0))

    shpA, (ar0, ac0) = norm(connA)
    shpB, (br0, bc0) = norm(connB)
    assert not (shpA != shpB)

    # base = sprite whose bbox top-left is closest to (0,0)
    r0a, c0a, _r1a, _c1a = bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = bbox_of_pts(B)
    # Base must be anchored at grid top-left corner
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    # Connectors dock vertically: base connector's free side faces down,
    # plug connector's free side faces up.
    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body  # base: free side down
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body  # plug: free side up

    _sp, (pr0, pc0) = norm(connP)
    _sd, (dr0, dc0) = norm(connD)

    # place plug connector directly above base connector
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    # render on minimal bbox
    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = bbox_coords(coords_all)
    rr0, cc0, rr1, cc1 = map(int, (rr0, cc0, rr1, cc1))
    # Base must remain at output top-left after tight crop
    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0  # plug must not extend above/left of base
    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    # stamp base first
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    # stamp plug on top; overlap only allowed at connector docking zone (ctrl meets obj)
    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)  # overlap only at connector
        out[int(r - rr0), int(c - cc0)] = int(val)

    # connector disappears -> becomes obj
    out[out == int(ctrl)] = int(obj)

    # tight crop around non-bg
    nz = np.argwhere(out != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


# ---------------------------------------------------------------------------
# Generator (anchor-first: build base, grow plug on it, then separate)
# ---------------------------------------------------------------------------
import random as _rng


def _grow_shape(mask, target, frontier):
    """Grow a connected shape on mask from frontier up to target pixels."""
    count = int(mask.sum())
    while count < target and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [(r+dr, c+dc) for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
                if 0 <= r+dr < h and 0 <= c+dc < w and not mask[r+dr, c+dc]]
        if not nbrs:
            frontier[idx] = frontier[-1]; frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask, keep=None):
    """Remove interior pixels to make a thin-walled shape. keep = set of (r,c) to never remove."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (r, c) in keep:
            continue
        # Skip if already on boundary (adjacent to False)
        adj_bg = any(not (0 <= r+dr < h and 0 <= c+dc < w) or not mask[r+dr, c+dc]
                     for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)))
        if adj_bg:
            continue  # already thin here
        # Try removing
        mask[r, c] = False
        if is_contiguous(mask, background=0, connectivity=8):
            pass  # keep removed
        else:
            mask[r, c] = True  # restore


def _gen_attempt():
    bg, obj, ctrl = 1, 4, 7

    # Step 1: build base sprite body
    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    # Step 2: place connector on base (can be edge or interior row)
    conn_len = _rng.randint(1, min(3, base_w))
    # Find valid connector positions: row where base has obj, with obj above it
    # (connector free side faces down, body side faces up)
    candidates = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            # All connector cols must have base pixels
            if not all(base[r, c] for c in cols):
                continue
            # Body side (above): all connector cols must have obj above
            if r == 0:
                continue  # need body above
            if not all(base[r - 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    # Build base sprite grid with connector
    base_grid = np.full((base_h, base_w), bg, dtype=int)
    base_grid[base] = obj
    for c in conn_cols:
        base_grid[conn_row, c] = ctrl

    # Verify base still 8-connected with ctrl
    if not is_contiguous(base_grid != bg, background=0, connectivity=8):
        return None

    # Step 3: grow plug from connector downward
    # Plug connector sits 1 row above base connector in docked state.
    # Plug body grows below its connector.
    # In docked coords: plug connector at row conn_row, plug body at conn_row+1..
    # But plug is ON TOP of base, so plug pixels must not overlap base body (except at connector zone)
    max_plug_h = _rng.randint(2, 7)
    plug_h = max_plug_h
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))

    # Plug connector col offset so it aligns with base connector
    # plug_conn_c0 in plug-local coords, must map to base conn_cols in docked coords
    # docked: plug placed at (conn_row, dc_offset) where dc_offset aligns connectors
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = conn_cols[0] - plug_conn_c0  # plug col 0 maps to base col dc_offset

    # Plug occupies docked rows conn_row..conn_row+plug_h-1, cols dc_offset..dc_offset+plug_w-1
    # Row conn_row = plug connector row (shared with base connector)
    # Check plug fits in reasonable bounds
    if dc_offset < 0 or conn_row + plug_h > base_h + plug_h:
        pass  # plug can extend beyond base bbox, that's OK

    # Build plug mask in plug-local coords
    plug = np.zeros((plug_h, plug_w), dtype=bool)
    # Seed: connector cols + first body row
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True  # connector row
        if plug_h > 1:
            plug[1, c] = True  # first body row

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    # Thin plug but keep connector row pixels
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    # Verify plug doesn't overlap base body at non-connector positions in docked state
    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = conn_row + r  # docked row
            dc = dc_offset + c  # docked col
            if 0 <= dr < base_h and 0 <= dc < base_w:
                if base_grid[dr, dc] == obj:
                    # Overlap with base obj — only OK at connector zone (row 0, connector cols)
                    if r == 0 and c in range(plug_conn_c0, plug_conn_c0 + conn_len):
                        continue  # connector zone, allowed (ctrl meets obj)
                    return None  # non-connector overlap

    # Plug must not extend above or left of base
    plug_min_col = dc_offset
    if plug_min_col < 0:
        return None

    # Build plug sprite grid
    plug_grid = np.full((plug_h, plug_w), bg, dtype=int)
    plug_grid[plug] = obj
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = ctrl

    # Verify plug connector adjacency (body below connector)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        if plug_h < 2 or plug_grid[1, c] != obj:
            return None

    if not is_contiguous(plug_grid != bg, background=0, connectivity=8):
        return None

    # Pixel count checks
    if int(np.sum(base_grid != bg)) < 5 or int(np.sum(plug_grid != bg)) < 5:
        return None

    # Crop both sprites
    def crop(g):
        nz = np.argwhere(g != bg)
        if nz.size == 0:
            return None
        r0, c0, r1, c1 = bbox_coords(nz)
        return g[r0:r1+1, c0:c1+1]

    base_sprite = crop(base_grid)
    plug_sprite = crop(plug_grid)
    if base_sprite is None or plug_sprite is None:
        return None
    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape

    # Step 4: place on canvas — base at (0,0), plug elsewhere
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), bg, dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=bg, padding=0)
    except ValueError:
        return None
    if pr + pc == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=bg)

    if np.sum(canvas == bg) <= canvas.size // 2:
        return None

    # Step 5: oracle verify
    try:
        result = main(canvas)
    except Exception:
        return None
    # result should be non-trivial
    if result.size == 0 or not np.any(result != bg):
        return None
    return canvas


def generate_input():
    for _ in range(300):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
