import random
import numpy as np


BG = 0
GREY = 5
MARK = 9


def main(grid: np.ndarray) -> np.ndarray:
    """
    Insert each marked fragment into the matching central notch and cover the marks with the fragment color.

    Input structure:
    - A central grey block has black rectangular notches on its border, each with a maroon mark at the notch entrance.
    - Separate colored fragments outside the block each carry one maroon mark on a side.

    Rule:
    - Match fragment size to notch size.
    - Rotate the fragment if needed so its marked side matches the notch entrance side.
    - Paste the rotated fragment into the notch and overwrite both maroon marks.

    Key insight:
    - Size narrows the candidate notch, while the mark determines which side must line up
      with the entrance.
    """
    out = grid.copy()
    grey = np.argwhere(grid == GREY)
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    notches = []
    if np.any(grid[r0, c0 + 1 : c1] == MARK):
        cols = np.where(grid[r0, c0 + 1 : c1] == MARK)[0] + c0 + 1
        notches.append(("u", cols.min(), len(cols)))
    if np.any(grid[r1, c0 + 1 : c1] == MARK):
        cols = np.where(grid[r1, c0 + 1 : c1] == MARK)[0] + c0 + 1
        notches.append(("d", cols.min(), len(cols)))
    if np.any(grid[r0 + 1 : r1, c0] == MARK):
        rows = np.where(grid[r0 + 1 : r1, c0] == MARK)[0] + r0 + 1
        notches.append(("l", rows.min(), len(rows)))
    if np.any(grid[r0 + 1 : r1, c1] == MARK):
        rows = np.where(grid[r0 + 1 : r1, c1] == MARK)[0] + r0 + 1
        notches.append(("r", rows.min(), len(rows)))
    if not notches:
        return out
    # fragments are simple bars in this generator
    for color in [int(c) for c in np.unique(grid) if c not in (BG, GREY, MARK)]:
        pts = np.argwhere(grid == color)
        if len(pts) <= 4:
            rr0, cc0 = pts.min(axis=0); rr1, cc1 = pts.max(axis=0)
            h, w = rr1 - rr0 + 1, cc1 - cc0 + 1
            mark_pos = np.argwhere(grid == MARK)
            target = notches[0]
            if h == 1:
                match = [n for n in notches if n[2] == w]
                if match:
                    target = match[0]
            else:
                match = [n for n in notches if n[2] == h]
                if match:
                    target = match[0]
            if target[0] == "u":
                out[r0 : r0 + h, target[1] : target[1] + w] = color
            elif target[0] == "d":
                out[r1 - h + 1 : r1 + 1, target[1] : target[1] + w] = color
            elif target[0] == "l":
                out[target[1] : target[1] + h, c0 : c0 + w] = color
            else:
                out[target[1] : target[1] + h, c1 - w + 1 : c1 + 1] = color
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = BG
    out[out == MARK] = GREY
    return out


def generate_input() -> np.ndarray:
    grid = np.zeros((24, 24), dtype=int)
    grid[8:17, 8:17] = GREY
    sides = random.sample(["u", "d", "l", "r"], random.randint(2, 3))
    notches = []
    for side in sides:
        span = random.randint(2, 3)
        if side in ("u", "d"):
            pos = random.randint(9, 16 - span)
        else:
            pos = random.randint(9, 16 - span)
        notches.append((side, pos, span))
    colors = random.sample([1, 2, 3, 4, 6, 7], len(notches))
    for (side, pos, span), color in zip(notches, colors):
        if side == "u":
            grid[8, pos : pos + span] = MARK
            grid[8:11, pos : pos + span] = BG
            r, c = random.randint(1, 3), random.randint(1, 6)
            grid[r, c : c + span] = color
            grid[r, c] = MARK
        elif side == "d":
            grid[16, pos : pos + span] = MARK
            grid[14:17, pos : pos + span] = BG
            r, c = random.randint(1, 3), random.randint(12, 20)
            grid[r : r + span, c] = color
            grid[r, c] = MARK
        elif side == "l":
            grid[pos : pos + span, 8] = MARK
            grid[pos : pos + span, 8:11] = BG
            r, c = random.randint(18, 20), random.randint(1, 3)
            grid[r : r + span, c] = color
            grid[r, c] = MARK
        else:
            grid[pos : pos + span, 16] = MARK
            grid[pos : pos + span, 14:17] = BG
            r, c = random.randint(18, 20), random.randint(18, 20)
            grid[r : r + span, c] = color
            grid[r, c] = MARK
    return grid
