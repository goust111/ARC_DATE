import random
from collections import deque
import numpy as np


def _gray_components(grid):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for sr, sc in zip(*np.where(grid == 5)):
        if seen[sr, sc]:
            continue
        q = deque([(int(sr), int(sc))])
        seen[sr, sc] = True
        cells = []
        while q:
            r, c = q.popleft()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == 5 and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(cells)
    return comps


def _frame_info(grid, cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
    return top, bottom, left, right


def _lines(out, top, bottom, left, right):
    width = right - left + 1
    height = bottom - top + 1
    row_lines = []
    for r in range(top, bottom + 1):
        if np.count_nonzero(out[r, left:right + 1] == 5) >= max(2, width // 2):
            row_lines.append(r)
    col_lines = []
    for c in range(left, right + 1):
        if np.count_nonzero(out[top:bottom + 1, c] == 5) >= max(2, height // 2):
            col_lines.append(c)
    return row_lines, col_lines


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    frames = []
    for cells in _gray_components(out):
        top, bottom, left, right = _frame_info(out, cells)
        if bottom - top < 4 or right - left < 4:
            continue
        out[top, left:right + 1] = 5
        out[bottom, left:right + 1] = 5
        out[top:bottom + 1, left] = 5
        out[top:bottom + 1, right] = 5
        row_lines, col_lines = _lines(out, top, bottom, left, right)
        frames.append((top, bottom, left, right, row_lines, col_lines))

    inferred = {}
    for _, _, _, _, row_lines, col_lines in frames:
        for i in range(len(row_lines) - 1):
            for j in range(len(col_lines) - 1):
                r0, r1 = row_lines[i] + 1, row_lines[i + 1]
                c0, c1 = col_lines[j] + 1, col_lines[j + 1]
                vals = [int(x) for x in out[r0:r1, c0:c1].reshape(-1) if x not in (0, 5)]
                if vals:
                    inferred[(i, j)] = vals[0]

    for _, _, _, _, row_lines, col_lines in frames:
        for i in range(len(row_lines) - 1):
            for j in range(len(col_lines) - 1):
                color = inferred.get((i, j))
                if color is None:
                    continue
                r0, r1 = row_lines[i] + 1, row_lines[i + 1]
                c0, c1 = col_lines[j] + 1, col_lines[j + 1]
                if np.count_nonzero((out[r0:r1, c0:c1] != 0) & (out[r0:r1, c0:c1] != 5)) == 0:
                    out[r0:r1, c0:c1] = color
    return out


def _draw_frame(grid, top, left, rows, cols, slot_h, slot_w):
    row_lines = [top + i * (slot_h + 1) for i in range(rows + 1)]
    col_lines = [left + j * (slot_w + 1) for j in range(cols + 1)]
    for r in row_lines:
        grid[r, col_lines[0]:col_lines[-1] + 1] = 5
    for c in col_lines:
        grid[row_lines[0]:row_lines[-1] + 1, c] = 5
    return row_lines, col_lines


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(1000):
        rows, cols = rng.choice([(2, 3), (3, 2)])
        slot_h, slot_w = 2, 2
        fh, fw = rows * slot_h + rows + 1, cols * slot_w + cols + 1
        positions = [(1, 1), (1, 14), (14, 1), (14, 14)]
        n = rng.choice([3, 4])
        grid = np.zeros((25, 25), dtype=int)
        colors = [1, 2, 3, 4, 6, 8]
        slot_colors = {(i, j): colors[i * cols + j] for i in range(rows) for j in range(cols)}
        coverage = {k: 0 for k in slot_colors}
        frame_slots = []
        for idx in range(n):
            keys = list(slot_colors)
            rng.shuffle(keys)
            take = set(keys[:rng.randint(2, len(keys) - 1)])
            frame_slots.append(take)
            for key in take:
                coverage[key] += 1
        for key, count in coverage.items():
            if count == 0:
                frame_slots[rng.randrange(n)].add(key)

        for idx in range(n):
            top, left = positions[idx]
            row_lines, col_lines = _draw_frame(grid, top, left, rows, cols, slot_h, slot_w)
            for i, j in frame_slots[idx]:
                grid[row_lines[i] + 1:row_lines[i + 1], col_lines[j] + 1:col_lines[j + 1]] = slot_colors[(i, j)]
            for _seg in range(rng.randint(1, 3)):
                edge = rng.choice(["top", "bottom", "left", "right"])
                if edge in ("top", "bottom"):
                    r = top if edge == "top" else top + fh - 1
                    c = rng.randint(left + 1, left + fw - 4)
                    ln = rng.randint(1, 3)
                    grid[r, c:c + ln] = 0
                else:
                    c = left if edge == "left" else left + fw - 1
                    r = rng.randint(top + 1, top + fh - 4)
                    ln = rng.randint(1, 3)
                    grid[r:r + ln, c] = 0
        return grid
    return generate_input()
