from __future__ import annotations
import random
import numpy as np

BG, GREY = 0, 5


def _cells(g):
    rows = np.where(np.all(g == GREY, axis=1))[0]
    cols = np.where(np.all(g == GREY, axis=0))[0]
    rs = [-1] + rows.tolist() + [g.shape[0]]
    cs = [-1] + cols.tolist() + [g.shape[1]]
    cells=[]
    for i in range(len(rs)-1):
        for j in range(len(cs)-1):
            r0,r1 = rs[i]+1, rs[i+1]
            c0,c1 = cs[j]+1, cs[j+1]
            if r1-r0 > 0 and c1-c0 > 0:
                cells.append((i,j,r0,r1,c0,c1))
    return cells, len(cs)-1


def _norm(p):
    return tuple(tuple(int(x) for x in row) for row in p)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower-cell pattern to the dictionary row and rotate it by the matched dictionary index modulo four.

    Input structure:
    - Grey separator rows and columns split the grid into equal square cells.
    - The top cell row is a blue dictionary of distinct patterns.
    - Lower cells contain red copies of one dictionary pattern, possibly already rotated.

    Rule:
    - Match each lower cell to the dictionary pattern it equals under some rotation.
    - Let P be the matched dictionary column index.
    - Rotate the cell pattern to the orientation encoded by P mod 4, with the dictionary orientation as zero.
    - Leave the dictionary row and the separators unchanged.

    Key insight:
    - The dictionary column performs both identification and control of the target rotation.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    cells,ncols = _cells(g)
    dicts = {}
    for i,j,r0,r1,c0,c1 in cells:
        patch = g[r0:r1,c0:c1]
        fg = (patch != BG) & (patch != GREY)
        if i == 0: dicts[j] = fg
    for i,j,r0,r1,c0,c1 in cells:
        if i == 0: continue
        patch = g[r0:r1,c0:c1]
        fg = (patch != BG) & (patch != GREY)
        match = 0; rot0 = fg
        for idx,ref in dicts.items():
            cur = fg.copy()
            for k in range(4):
                if cur.shape == ref.shape and np.array_equal(cur, ref):
                    match = idx; rot0 = cur; break
                cur = np.rot90(cur)
            else:
                continue
            break
        target = np.rot90(dicts[match], -(match % 4))
        color = int(np.max(patch))
        out[r0:r1,c0:c1] = BG
        out[r0:r1,c0:c1][target] = color
    out[g == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    cell=4; rows=3; cols=4
    h = rows*cell + (rows-1)
    w = cols*cell + (cols-1)
    g = np.zeros((h,w), dtype=int)
    for r in range(cell, h, cell+1): g[r,:] = GREY
    for c in range(cell, w, cell+1): g[:,c] = GREY
    pats = [np.array([[1,0,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]], dtype=bool), np.array([[0,1,0,0],[1,1,1,0],[0,0,0,0],[0,0,0,0]], dtype=bool), np.array([[1,1,0,0],[0,1,0,0],[0,1,0,0],[0,0,0,0]], dtype=bool), np.array([[1,0,0,0],[1,0,0,0],[1,1,0,0],[0,0,0,0]], dtype=bool)]
    for j,p in enumerate(pats):
        r0,c0 = 0, j*(cell+1)
        g[r0:r0+cell, c0:c0+cell][p] = 1
    for i in range(1,rows):
        for j in range(cols):
            ref = random.randrange(cols)
            p = np.rot90(pats[ref], random.randrange(4))
            r0,c0 = i*(cell+1), j*(cell+1)
            g[r0:r0+cell, c0:c0+cell][p] = 2
    return g
