import random
import numpy as np


BG = 0
GREY = 5


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each room's bar-shaped furniture out through its only exit and paint the whole corridor trail.

    Input structure:
    - Grey rectangular rooms each have exactly one wall gap as an exit.
    - Each room contains one colored 1-cell-thick bar object.

    Rule:
    - Detect the wall containing the gap to determine the push direction.
    - Remove the bar from the room and paint a straight same-colored trail in the corridor
      from just outside the exit until the next step would hit a wall or the grid boundary.
    - Leave all grey walls unchanged.

    Key insight:
    - Exit location, not bar orientation, determines the movement direction; the output is
      the swept trail, not just the final resting place.
    """
    out = grid.copy()
    for r0, r1, c0, c1 in _rooms(grid):
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(c) for c in np.unique(inner) if c != BG]
        if not colors:
            continue
        color = colors[0]
        pts = np.argwhere(inner == color)
        for r, c in pts:
            out[r0 + 1 + r, c0 + 1 + c] = BG
        if np.any(grid[r0, c0 + 1 : c1] == BG):
            col = int(np.where(grid[r0, c0 + 1 : c1] == BG)[0][0] + c0 + 1)
            r = r0 - 1
            while r >= 0 and grid[r, col] == BG:
                out[r, col] = color
                r -= 1
        elif np.any(grid[r1, c0 + 1 : c1] == BG):
            col = int(np.where(grid[r1, c0 + 1 : c1] == BG)[0][0] + c0 + 1)
            r = r1 + 1
            while r < grid.shape[0] and grid[r, col] == BG:
                out[r, col] = color
                r += 1
        elif np.any(grid[r0 + 1 : r1, c0] == BG):
            row = int(np.where(grid[r0 + 1 : r1, c0] == BG)[0][0] + r0 + 1)
            c = c0 - 1
            while c >= 0 and grid[row, c] == BG:
                out[row, c] = color
                c -= 1
        else:
            row = int(np.where(grid[r0 + 1 : r1, c1] == BG)[0][0] + r0 + 1)
            c = c1 + 1
            while c < grid.shape[1] and grid[row, c] == BG:
                out[row, c] = color
                c += 1
    return out


def _rooms(grid: np.ndarray):
    rooms = []
    for r in range(1, grid.shape[0] - 4):
        for c in range(1, grid.shape[1] - 4):
            if grid[r, c] != GREY:
                continue
            for h in range(5, 8):
                for w in range(5, 8):
                    if r + h > grid.shape[0] or c + w > grid.shape[1]:
                        continue
                    border = np.concatenate([grid[r, c : c + w], grid[r + h - 1, c : c + w], grid[r : r + h, c], grid[r : r + h, c + w - 1]])
                    if np.count_nonzero(border == GREY) >= len(border) - 2:
                        rooms.append((r, r + h - 1, c, c + w - 1))
    uniq = []
    for x in rooms:
        if x not in uniq:
            uniq.append(x)
    return uniq


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], random.randint(2, 4))
    used = []
    for color in colors:
        h = random.randint(5, 7)
        w = random.randint(5, 7)
        side = random.choice(["u", "d", "l", "r"])
        for _ in range(200):
            r = random.randint(2, 24 - h - 2)
            c = random.randint(2, 24 - w - 2)
            if all(r + h + 2 <= rr or rr + hh + 2 <= r or c + w + 2 <= cc or cc + ww + 2 <= c for rr, cc, hh, ww in used):
                used.append((r, c, h, w))
                grid[r, c : c + w] = GREY
                grid[r + h - 1, c : c + w] = GREY
                grid[r : r + h, c] = GREY
                grid[r : r + h, c + w - 1] = GREY
                if side == "u":
                    grid[r, c + w // 2] = BG
                    grid[r + h // 2, c + 1 : c + w - 1] = color
                elif side == "d":
                    grid[r + h - 1, c + w // 2] = BG
                    grid[r + h // 2, c + 1 : c + w - 1] = color
                elif side == "l":
                    grid[r + h // 2, c] = BG
                    grid[r + 1 : r + h - 1, c + w // 2] = color
                else:
                    grid[r + h // 2, c + w - 1] = BG
                    grid[r + 1 : r + h - 1, c + w // 2] = color
                break
    return grid
