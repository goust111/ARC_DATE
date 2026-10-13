import numpy as np

from arc_utils import DIR4, most_common as _most_common


def transform(grid: np.ndarray) -> np.ndarray:
    """
    Move the closest reachable maroon pixels into the 3×3 block around the unique red center.

    Input structure:
    - bg = orange floor, magenta walls, scattered maroon (9) pixels.
    - A unique red (2) pixel marks the center of a 3×3 destination block.

    Rule:
    - Maroon pixels connected to the destination through floor/maroon cells (not blocked by walls) are candidates.
    - Rank candidates by distance to the destination; the closest ones fill the block in row-major order.
    - Moved maroons are erased (replaced with floor color).

    Key insight: Wall barriers restrict connectivity — only maroon pixels reachable via flood-fill through the floor qualify, so proximity alone isn't enough.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    centers = np.argwhere(grid == 2)
    if len(centers) != 1:
        return out
    cr, cc = map(int, centers[0])

    # Destination = 3x3 block around the center.
    dest_cells: list[tuple[int, int]] = []
    for r in range(cr - 1, cr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < h and 0 <= c < w:
                dest_cells.append((r, c))
    dest_cells.sort()
    dest_set = set(dest_cells)

    # Infer floor color from the ring around the destination.
    ring_colors: list[int] = []
    for r, c in dest_cells:
        for dr, dc in DIR4:
            rr, cc2 = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) not in dest_set:
                ring_colors.append(int(grid[rr, cc2]))

    cand_ring = [x for x in ring_colors if x not in (2, 3, 9)]
    if not cand_ring:
        return out
    wall = _most_common(np.array(cand_ring, dtype=int))
    other = [int(x) for x in np.unique(grid) if int(x) not in (2, 3, 9, wall)]
    floor = other[0] if len(other) == 1 else (other and max(other, key=lambda c: int((grid == c).sum())))
    if not floor:
        floor = wall

    passable = {floor, 9}

    # BFS start = passable cells adjacent to the destination.
    starts: list[tuple[int, int]] = []
    seen = set()
    for r, c in dest_cells:
        for dr, dc in DIR4:
            rr, cc2 = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable:
                if (rr, cc2) not in seen:
                    seen.add((rr, cc2))
                    starts.append((rr, cc2))
    if not starts:
        return out

    INF = 10**9
    dist = np.full((h, w), INF, dtype=int)
    q = list(starts)
    for r, c in starts:
        dist[r, c] = 0

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR4:
            rr, cc2 = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc2 < w and int(grid[rr, cc2]) in passable and dist[rr, cc2] == INF:
                dist[rr, cc2] = d + 1
                q.append((rr, cc2))

    maroons = [(int(dist[r, c]), int(r), int(c)) for r, c in np.argwhere(grid == 9)]
    maroons = [t for t in maroons if t[0] < INF]
    maroons.sort()  # by distance then row then col

    k = min(len(dest_cells), len(maroons))
    chosen = {(r, c) for _, r, c in maroons[:k]}

    for r, c in chosen:
        out[r, c] = floor

    for r, c in dest_cells[:k]:
        out[r, c] = 9

    return out
