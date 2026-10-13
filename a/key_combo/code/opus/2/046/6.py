import random
from collections import deque
import numpy as np

Color = {
    "BLACK": 0,
    "GREEN": 3,
    "RED": 2,
    "BLUE": 1,
    "YELLOW": 4,
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_green_blocks(g: np.ndarray):
    h, w = g.shape
    visited = np.zeros_like(g, dtype=bool)
    blocks = []
    for r in range(h):
        for c in range(w):
            if g[r, c] != Color["GREEN"] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and g[nr, nc] == Color["GREEN"]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            if len(comp) == 4:
                rs = [p[0] for p in comp]
                cs = [p[1] for p in comp]
                r0, r1 = min(rs), max(rs)
                c0, c1 = min(cs), max(cs)
                if (r1 - r0 + 1) == 2 and (c1 - c0 + 1) == 2:
                    blocks.append((r0, c0, r1, c1))
    # sort stable for deterministic
    blocks = [(r0, c0, r1, c1) for r0, c0, r1, c1 in sorted(blocks)]
    return blocks


def _block_markers(block):
    r0, c0, r1, c1 = block
    return [
        (r0 - 1, c0 - 1, -1, -1),
        (r0 - 1, c1 + 1, -1, 1),
        (r1 + 1, c0 - 1, 1, -1),
        (r1 + 1, c1 + 1, 1, 1),
    ]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from green 2x2 anchors, then color them by global scan order.

    Input structure:
    - Black background and several green 2x2 blocks.
    - Each block has 1+ corner diagonal markers of the same color.
    - Global scan order is row-priority or column-priority depending on the anchors' center span.

    Rule:
    - Determine anchor blocks and choose scan order by center bounding box: row-priority if row span >= col span, otherwise column-priority.
    - Expand each marker in clockwise corner order (TL, TR, BR, BL) within that block order.
    - Draw ray cells in black as 45° diagonals until boundary or obstacle.
    - Color rays cyclically Red, Blue, Yellow by global ray index.

    Key insight:
    - Ray color assignment depends on a global ordering determined by all anchors together, not per-block local geometry.
    """
    g = np.asarray(grid, dtype=int)
    blocks = _find_green_blocks(g)
    if len(blocks) < 3:
        return g.copy()

    centers = [(b[0], b[1]) for b in blocks]  # top-left order proxy
    rs = [x[0] for x in centers]
    cs = [x[1] for x in centers]
    row_span = max(rs) - min(rs)
    col_span = max(cs) - min(cs)

    if row_span >= col_span:
        order = sorted(range(len(blocks)), key=lambda i: (blocks[i][0], blocks[i][1]))
    else:
        order = sorted(range(len(blocks)), key=lambda i: (blocks[i][1], blocks[i][0]))

    block_cells = set()
    for b in blocks:
        for rr in range(b[0], b[2] + 1):
            for cc in range(b[1], b[3] + 1):
                block_cells.add((rr, cc))

    out = g.copy()
    cycle = [Color["RED"], Color["BLUE"], Color["YELLOW"]]
    ray_idx = 0

    for idx in order:
        b = blocks[idx]
        for mr, mc, dr, dc in _block_markers(b):
            if not (0 <= mr < g.shape[0] and 0 <= mc < g.shape[1]):
                continue
            if g[mr, mc] != Color["GREEN"]:
                continue

            color = cycle[ray_idx % 3]
            ray_idx += 1
            r, c = mr + dr, mc + dc
            while 0 <= r < g.shape[0] and 0 <= c < g.shape[1]:
                if (r, c) in block_cells or g[r, c] != Color["BLACK"]:
                    break
                out[r, c] = color
                r += dr
                c += dc

    return out


def _random_connected_block(marked: set[tuple[int, int]]):
    # not needed: we only place plain 2x2 blocks
    return marked


def generate_input() -> np.ndarray:
    for _ in range(500):
        h = 15
        w = 15
        n_blocks = random.randint(3, 5)
        blocks = []
        block_set = []

        tries = 0
        while len(block_set) < n_blocks and tries < 1000:
            tries += 1
            r = random.randint(1, h - 3)
            c = random.randint(1, w - 3)
            if any(abs(r - rr) <= 4 and abs(c - cc) <= 4 for rr, cc in block_set):
                continue
            block_set.append((r, c))

        if len(block_set) != n_blocks:
            continue

        blocks = block_set
        g = np.zeros((h, w), dtype=int)
        for r, c in blocks:
            g[r : r + 2, c : c + 2] = Color["GREEN"]

        # add markers: at least one per block, 1..2 each
        candidate_markers = [
            [(-1, -1), (-1, 1), (1, -1), (1, 1)],
        ]

        marker_cells = set()
        for r, c in blocks:
            k = random.randint(1, 2)
            dirs = random.sample([(r - 1, c - 1), (r - 1, c + 2), (r + 2, c - 1), (r + 2, c + 2)], k)
            for mr, mc in dirs:
                if 0 <= mr < h and 0 <= mc < w and g[mr, mc] == Color["BLACK"]:
                    marker_cells.add((mr, mc))
                    g[mr, mc] = Color["GREEN"]

        # validate existence of rays + non-intersection
        blocks_cells = set()
        for r, c in blocks:
            for rr in range(r, r + 2):
                for cc in range(c, c + 2):
                    blocks_cells.add((rr, cc))

        all_rays = []
        occupied = set()
        ok = True
        for b in blocks:
            for mr, mc, dr, dc in _block_markers((b[0], b[1], b[0] + 1, b[1] + 1)):
                if not (0 <= mr < h and 0 <= mc < w):
                    continue
                if g[mr, mc] != Color["GREEN"]:
                    continue
                path = []
                r, c = mr + dr, mc + dc
                while 0 <= r < h and 0 <= c < w:
                    if (r, c) in blocks_cells:
                        break
                    if g[r, c] != Color["BLACK"]:
                        break
                    if (r, c) in occupied:
                        ok = False
                        break
                    path.append((r, c))
                    occupied.add((r, c))
                    r += dr
                    c += dc
                if not ok:
                    break
            if not ok:
                break

        if not ok:
            continue

        # enforce global orientation is not square
        rs = [x for x, _ in blocks]
        cs = [x[1] for x in blocks]
        if max(rs) - min(rs) == max(cs) - min(cs):
            continue

        return g

    raise RuntimeError("generate_input: failed to sample valid case for puzzle 6")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, y.max())
