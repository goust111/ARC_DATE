import numpy as np
from collections import deque


def _components(mask):
    h, w = mask.shape
    vis = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            q = deque([(r, c)])
            vis[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                        vis[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _corner_of_bbox(cells, h, w):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    top = r0 == 0
    left = c0 == 0
    right = c1 == w - 1
    bottom = r1 == h - 1
    if top and left:
        return "tl"
    if top and right:
        return "tr"
    if bottom and right:
        return "br"
    if bottom and left:
        return "bl"
    return None


def _paste(out, source, comp, dst_r0, dst_c0):
    rs = comp[:, 0].min(), comp[:, 0].max()
    cs = comp[:, 1].min(), comp[:, 1].max()
    h0 = rs[1] - rs[0] + 1
    w0 = cs[1] - cs[0] + 1
    for r in range(h0):
        for c in range(w0):
            if source[rs[0] + r, cs[0] + c] != 0:
                out[dst_r0 + r, dst_c0 + c] = source[rs[0] + r, cs[0] + c]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, copy=True)
    h, w = g.shape
    out = g.copy().astype(np.int64)
    fg = g != 0
    comps = _components(fg)

    corner_map = {"tl": None, "tr": None, "br": None, "bl": None}
    for comp in comps:
        corner = _corner_of_bbox(comp, h, w)
        if corner is not None:
            corner_map[corner] = comp

    nxt = {"tl": "tr", "tr": "br", "br": "bl", "bl": "tl"}
    copy_targets = {}
    for src_corner, src_cells in corner_map.items():
        if src_cells is None:
            continue
        r0, c0 = src_cells.min(axis=0)
        r1, c1 = src_cells.max(axis=0)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        dst = nxt[src_corner]
        if dst == "tl":
            dr, dc = 0, 0
        elif dst == "tr":
            dr, dc = 0, w - ww
        elif dst == "br":
            dr, dc = h - hh, w - ww
        else:
            dr, dc = h - hh, 0
        copy_targets[dst] = (src_cells, (dr, dc))

    for dst_corner, (cells, (dr, dc)) in copy_targets.items():
        _paste(out, g, cells, dr, dc)

    # fill gaps on each edge between adjacent corner objects
    # top edge
    comps_out = _components(out != 0)
    edge_comps = {"tl": None, "tr": None, "br": None, "bl": None}
    for comp in comps_out:
        corner = _corner_of_bbox(comp, h, w)
        if corner is not None:
            edge_comps[corner] = comp

    if edge_comps["tl"] is not None and edge_comps["tr"] is not None:
        col_left = int(edge_comps["tl"][:, 1].max())
        col_right = int(edge_comps["tr"][:, 1].min())
        for cc in range(col_left + 1, col_right):
            if out[0, cc] == 0:
                out[0, cc] = 8

    if edge_comps["bl"] is not None and edge_comps["br"] is not None:
        col_left = int(edge_comps["bl"][:, 1].max())
        col_right = int(edge_comps["br"][:, 1].min())
        for cc in range(col_left + 1, col_right):
            if out[h - 1, cc] == 0:
                out[h - 1, cc] = 8

    if edge_comps["tl"] is not None and edge_comps["bl"] is not None:
        row_top = int(edge_comps["tl"][:, 0].max())
        row_bottom = int(edge_comps["bl"][:, 0].min())
        for rr in range(row_top + 1, row_bottom):
            if out[rr, 0] == 0:
                out[rr, 0] = 8

    if edge_comps["tr"] is not None and edge_comps["br"] is not None:
        row_top = int(edge_comps["tr"][:, 0].max())
        row_bottom = int(edge_comps["br"][:, 0].min())
        for rr in range(row_top + 1, row_bottom):
            if out[rr, w - 1] == 0:
                out[rr, w - 1] = 8

    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 12
    grid = np.zeros((h, w), dtype=np.int64)
    used = np.zeros((h, w), dtype=bool)
    corners = ["tl", "tr", "bl", "br"]
    selected = list(rng.choice(corners, size=int(rng.integers(1, 4)), replace=False))
    palette = [1, 2, 3, 4, 7]
    for corner in selected:
        color = int(rng.choice(palette))
        if corner == "tl":
            hh = int(rng.integers(2, 5))
            ww = int(rng.integers(2, 5))
            r0, c0 = 0, 0
        elif corner == "tr":
            hh = int(rng.integers(2, 5))
            ww = int(rng.integers(2, 5))
            r0, c0 = 0, w - ww
        elif corner == "bl":
            hh = int(rng.integers(2, 5))
            ww = int(rng.integers(2, 5))
            r0, c0 = h - hh, 0
        else:
            hh = int(rng.integers(2, 5))
            ww = int(rng.integers(2, 5))
            r0, c0 = h - hh, w - ww

        for r in range(hh):
            for c in range(ww):
                if rng.random() < 0.85:
                    rr, cc = r0 + r, c0 + c
                    grid[rr, cc] = color
                    used[rr, cc] = True
        # guarantee corner touch
        if corner in ("tl", "bl"):
            grid[r0, c0] = color
            used[r0, c0] = True
            grid[min(r0 + hh - 1, h - 1), c0] = color
            used[min(r0 + hh - 1, h - 1), c0] = True
        if corner in ("tr", "br"):
            grid[r0, c0 + ww - 1] = color
            used[r0, c0 + ww - 1] = True
            grid[min(r0 + hh - 1, h - 1), c0 + ww - 1] = color
            used[min(r0 + hh - 1, h - 1), c0 + ww - 1] = True

    # optional interior distractors
    for _ in range(int(rng.integers(0, 3))):
        r = int(rng.integers(2, h - 2))
        c = int(rng.integers(2, w - 2))
        if used[r, c]:
            continue
        grid[r, c] = int(rng.choice([6, 9]))
    return grid

