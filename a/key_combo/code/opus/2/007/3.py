import random
import numpy as np


def _grid_lines(grid):
    row_lines = [i for i in range(grid.shape[0]) if np.all(grid[i, :] == 5)]
    col_lines = [j for j in range(grid.shape[1]) if np.all(grid[:, j] == 5)]
    return row_lines, col_lines


def main(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    row_lines, col_lines = _grid_lines(grid)
    slots = []
    for sr in range(len(row_lines) - 1):
        for sc in range(len(col_lines) - 1):
            r0, r1 = row_lines[sr] + 1, row_lines[sr + 1]
            c0, c1 = col_lines[sc] + 1, col_lines[sc + 1]
            slots.append((sr, sc, r0, r1, c0, c1, ((r0 + r1 - 1) // 2, (c0 + c1 - 1) // 2)))

    proto = None
    masks = {}
    for slot in slots:
        _, _, r0, r1, c0, c1, _ = slot
        sub = grid[r0:r1, c0:c1]
        mask = (sub != 0) & (sub != 5)
        masks[(slot[0], slot[1])] = mask
        if np.any(sub == 8):
            proto = mask

    matches = []
    for slot in slots:
        key = (slot[0], slot[1])
        if masks[key].any() and np.array_equal(masks[key], proto):
            matches.append(slot)

    crossed = set()
    for a, b in zip(matches, matches[1:]):
        ar, ac = a[6]
        br, bc = b[6]
        step = 1 if bc >= ac else -1
        for c in range(ac, bc + step, step):
            out[ar, c] = 5
            for slot in slots:
                if slot[2] <= ar < slot[3] and slot[4] <= c < slot[5]:
                    crossed.add((slot[0], slot[1]))
        step = 1 if br >= ar else -1
        for r in range(ar, br + step, step):
            out[r, bc] = 5
            for slot in slots:
                if slot[2] <= r < slot[3] and slot[4] <= bc < slot[5]:
                    crossed.add((slot[0], slot[1]))

    match_keys = {(s[0], s[1]) for s in matches}
    for slot in slots:
        key = (slot[0], slot[1])
        r0, r1, c0, c1 = slot[2], slot[3], slot[4], slot[5]
        sym = (grid[r0:r1, c0:c1] != 0) & (grid[r0:r1, c0:c1] != 5)
        if key in match_keys:
            out[r0:r1, c0:c1][sym] = 8
        elif key in crossed:
            out[r0:r1, c0:c1][sym] = 7
    return out


def generate_input() -> np.ndarray:
    masks = [
        np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ]
    for _ in range(1000):
        grid = np.zeros((13, 17), dtype=int)
        grid[[0, 4, 8, 12], :] = 5
        grid[:, [0, 4, 8, 12, 16]] = 5
        proto_mask = random.choice(masks)
        all_slots = [(r, c) for r in range(3) for c in range(4)]
        match_slots = sorted(random.sample(all_slots, random.randint(3, 5)))
        bad = False
        for a, b in zip(match_slots, match_slots[1:]):
            ar, ac = 2 + 4 * a[0], 2 + 4 * a[1]
            br, bc = 2 + 4 * b[0], 2 + 4 * b[1]
            for m in match_slots:
                if m in (a, b):
                    continue
                mr, mc = 2 + 4 * m[0], 2 + 4 * m[1]
                if (mr == ar and min(ac, bc) <= mc <= max(ac, bc)) or (mc == bc and min(ar, br) <= mr <= max(ar, br)):
                    bad = True
        if bad:
            continue
        proto_slot = random.choice(match_slots)
        colors = [1, 2, 3, 4, 6, 7, 9]
        for slot in all_slots:
            r0, c0 = 1 + 4 * slot[0], 1 + 4 * slot[1]
            if slot in match_slots:
                color = 8 if slot == proto_slot else random.choice(colors)
                grid[r0:r0 + 3, c0:c0 + 3][proto_mask] = color
            elif random.random() < 0.75:
                mask = random.choice([m for m in masks if not np.array_equal(m, proto_mask)])
                grid[r0:r0 + 3, c0:c0 + 3][mask] = random.choice(colors)
        return grid
    raise RuntimeError("failed to generate input")
