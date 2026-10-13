import random

import numpy as np


BG = 0
WALL = 5


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    wall_rows = [r for r in range(grid.shape[0]) if BG not in set(map(int, grid[r]))]
    wall_cols = [c for c in range(grid.shape[1]) if BG not in set(map(int, grid[:, c]))]
    r0, r1 = wall_rows[0], wall_rows[-1]
    rooms = []
    for i in range(len(wall_cols) - 1):
        c0, c1 = wall_cols[i], wall_cols[i + 1]
        if c1 - c0 > 1:
            rooms.append((r0, r1, c0, c1))
    keys = {}
    for idx, (ra, rb, ca, cb) in enumerate(rooms):
        inner = grid[ra + 1 : rb, ca + 1 : cb]
        pts = np.argwhere((inner != BG) & (inner != WALL))
        for rr, cc in pts:
            color = int(inner[rr, cc])
            if color != BG:
                keys[idx] = (color, (ra + 1 + int(rr), ca + 1 + int(cc)))
                break
    doors = {}
    indeg = {i: 0 for i in range(len(rooms))}
    for i in range(len(rooms) - 1):
        wall_c = rooms[i][3]
        door_row = next(r for r in range(r0 + 1, r1) if grid[r, wall_c] != WALL)
        color = int(grid[door_row, wall_c])
        src = i if keys[i][0] == color else i + 1
        dst = i + 1 if src == i else i
        doors[src] = (dst, (door_row, wall_c))
        indeg[dst] += 1
    cur = next(i for i in indeg if indeg[i] == 0)
    cur_pos = keys[cur][1]
    while cur in doors:
        color, key_pos = keys[cur]
        out[key_pos] = BG
        nxt, door = doors[cur]
        dr = 0 if rooms[nxt][0] == rooms[cur][0] else (1 if rooms[nxt][0] > rooms[cur][0] else -1)
        dc = 1 if nxt > cur else -1
        rr, cc = cur_pos
        while (rr, cc) != door:
            out[rr, cc] = color
            rr += dr
            cc += dc
        out[door] = color
        _, _, nca, ncb = rooms[nxt]
        end_c = ncb - 1 if dc == 1 else nca + 1
        cc += dc
        while cc != end_c + dc:
            out[rr, cc] = color
            cc += dc
        cur = nxt
        cur_pos = (rr, end_c)
    return out


def generate_input() -> np.ndarray:
    room_count = random.randint(3, 4)
    room_w = random.randint(4, 5)
    room_h = random.randint(4, 6)
    h = room_h + 2
    w = room_count * room_w + room_count + 1
    grid = np.zeros((h, w), dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    x = 0
    for _ in range(room_count + 1):
        grid[:, x] = WALL
        x += room_w + 1
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], room_count)
    key_row = random.randint(1, room_h)
    for i in range(room_count):
        c0 = i * (room_w + 1)
        color = colors[i]
        key_col = c0 + random.randint(1, room_w)
        grid[key_row, key_col] = color
        if i < room_count - 1:
            door_col = c0 + room_w + 1
            grid[key_row, door_col] = color
    return grid
