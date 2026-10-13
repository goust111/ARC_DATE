import random
import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each room's key color to find its matching door and fill the room on the far side of that door.

    Input structure:
    - Grey walls form a horizontal row of rectangular rooms.
    - Each room contains one uniquely colored key pixel.
    - One wall pixel somewhere on a room boundary is replaced by a matching door color.

    Rule:
    - For each room, read its key color and locate the door pixel with the same color.
    - If that door lies on the room's left wall, fill the room immediately to the left.
      If it lies on the right wall, fill the room immediately to the right.
    - Fill only black interior pixels; keep keys, walls, and doors unchanged.

    Key insight:
    - The key does not affect its own room; it remotely selects a neighboring room through
      a color-matched door embedded in one of the room's walls.
    """
    out = grid.copy()
    wall_cols = [c for c in range(grid.shape[1]) if np.count_nonzero(grid[:, c] == GREY) >= grid.shape[0] - 2]
    wall_cols = [0] + [c for c in wall_cols if c not in (0, grid.shape[1] - 1)] + [grid.shape[1] - 1]
    rooms = [(wall_cols[i], wall_cols[i + 1]) for i in range(len(wall_cols) - 1)]
    key_room = {}
    for i, (c0, c1) in enumerate(rooms):
        inner = grid[1:-1, c0 + 1 : c1]
        colors = [int(c) for c in np.unique(inner) if c != BG]
        if colors:
            key_room[colors[0]] = i
    for color, room_idx in key_room.items():
        pos = np.argwhere(grid == color)
        key_pos = None
        door_pos = None
        for r, c in pos:
            if 1 <= r < grid.shape[0] - 1 and rooms[room_idx][0] < c < rooms[room_idx][1]:
                key_pos = (r, c)
            else:
                door_pos = (r, c)
        if door_pos is None:
            continue
        _, dc = door_pos
        if dc == rooms[room_idx][0] and room_idx > 0:
            target = room_idx - 1
        elif dc == rooms[room_idx][1] and room_idx < len(rooms) - 1:
            target = room_idx + 1
        else:
            continue
        c0, c1 = rooms[target]
        region = out[1:-1, c0 + 1 : c1]
        region[region == BG] = color
    return out


def generate_input() -> np.ndarray:
    rooms = random.randint(3, 4)
    widths = [random.randint(4, 6) for _ in range(rooms)]
    W = sum(widths) + rooms + 1
    H = 9
    grid = np.zeros((H, W), dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    c = 0
    divs = [0]
    grid[:, 0] = GREY
    for w in widths:
        c += w + 1
        grid[:, c] = GREY
        divs.append(c)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], rooms)
    for i, color in enumerate(colors):
        c0, c1 = divs[i], divs[i + 1]
        kr = random.randint(2, H - 3)
        kc = random.randint(c0 + 1, c1 - 1)
        grid[kr, kc] = color
    for i, color in enumerate(colors):
        c0, c1 = divs[i], divs[i + 1]
        side = random.choice(["l", "r"])
        if side == "l" and i > 0:
            grid[random.randint(2, H - 3), c0] = color
        elif side == "r" and i < rooms - 1:
            grid[random.randint(2, H - 3), c1] = color
        elif i > 0:
            grid[random.randint(2, H - 3), c0] = color
        else:
            grid[random.randint(2, H - 3), c1] = color
    return grid
