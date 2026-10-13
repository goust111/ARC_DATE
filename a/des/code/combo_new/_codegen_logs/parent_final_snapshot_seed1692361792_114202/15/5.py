import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5
BLUE = 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _candidate_gaps(grid: np.ndarray):
    h, w = grid.shape
    gaps = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BLACK:
                continue
            vertical = (
                grid[r - 1, c] == GREY
                and grid[r + 1, c] == GREY
                and grid[r, c - 1] != GREY
                and grid[r, c + 1] != GREY
            )
            horizontal = (
                grid[r, c - 1] == GREY
                and grid[r, c + 1] == GREY
                and grid[r - 1, c] != GREY
                and grid[r + 1, c] != GREY
            )
            if vertical ^ horizontal:
                gaps.append((r, c, "vertical" if vertical else "horizontal"))
    return gaps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    out = grid.copy()
    gap_info = _candidate_gaps(grid)

    wall_mask = grid == GREY
    for r, c, _ in gap_info:
        wall_mask[r, c] = True

    room_id = -np.ones(grid.shape, dtype=int)
    rooms = _components(~wall_mask)
    room_seeds = {}
    for idx, comp in enumerate(rooms):
        room_id[comp[:, 0], comp[:, 1]] = idx
        colors = set(grid[comp[:, 0], comp[:, 1]].tolist()) & {RED, BLUE}
        room_seeds[idx] = next(iter(colors))

    for r, c, orient in gap_info:
        if orient == "vertical":
            a = room_id[r, c - 1]
            b = room_id[r, c + 1]
        else:
            a = room_id[r - 1, c]
            b = room_id[r + 1, c]
        out[r, c] = GREY if room_seeds[a] == room_seeds[b] else GREEN
    return out


def generate_input() -> np.ndarray:
    h = random.randint(13, 17)
    w = random.randint(13, 17)
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    row_positions = list(range(3, h - 3))
    col_positions = list(range(3, w - 3))

    def pick_cuts(positions):
        target = random.randint(1, 2)
        random.shuffle(positions)
        picks = []
        for pos in positions:
            if all(abs(pos - prev) >= 3 for prev in picks):
                picks.append(pos)
            if len(picks) == target:
                break
        return sorted(picks or [positions[len(positions) // 2]])

    row_cuts = pick_cuts(row_positions)
    col_cuts = pick_cuts(col_positions)
    for r in row_cuts:
        grid[r, :] = GREY
    for c in col_cuts:
        grid[:, c] = GREY

    rooms = []
    r_edges = [0] + row_cuts + [h - 1]
    c_edges = [0] + col_cuts + [w - 1]
    for i in range(len(r_edges) - 1):
        for j in range(len(c_edges) - 1):
            r0, r1 = r_edges[i] + 1, r_edges[i + 1] - 1
            c0, c1 = c_edges[j] + 1, c_edges[j + 1] - 1
            if r0 > r1 or c0 > c1:
                continue
            rooms.append((r0, r1, c0, c1))
            sr = random.randint(r0, r1)
            sc = random.randint(c0, c1)
            grid[sr, sc] = random.choice([RED, BLUE])

    gap_choices = []
    for c in col_cuts:
        for r0, r1, _, _ in rooms:
            if r0 <= r1 and 1 < c < w - 2 and r0 <= (r0 + r1) // 2 <= r1:
                gap_choices.append(("v", (random.randint(r0, r1), c)))
    for r in row_cuts:
        for _, _, c0, c1 in rooms:
            if c0 <= c1 and 1 < r < h - 2 and c0 <= (c0 + c1) // 2 <= c1:
                gap_choices.append(("h", (r, random.randint(c0, c1))))
    random.shuffle(gap_choices)
    for kind, (r, c) in gap_choices[: random.randint(2, min(5, len(gap_choices)))]:
        if kind == "v" and grid[r - 1, c] == GREY and grid[r + 1, c] == GREY:
            grid[r, c] = BLACK
        if kind == "h" and grid[r, c - 1] == GREY and grid[r, c + 1] == GREY:
            grid[r, c] = BLACK
    return grid
