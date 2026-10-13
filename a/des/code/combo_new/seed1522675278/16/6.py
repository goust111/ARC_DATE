import random
from collections import deque

import numpy as np


BG = 0


def _components(mask):
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid)
    out = grid.copy()
    comps = _components(grid != BG)
    for cells in comps:
        colors = [int(grid[r, c]) for r, c in cells]
        base = max(set(colors), key=colors.count)
        marker = min(set(colors), key=colors.count)
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bbox = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        marker_pos = tuple(np.argwhere(bbox == marker)[0])
        mask = (bbox != BG).astype(int)
        for r, c in cells:
            out[r, c] = BG
        if np.array_equal(mask, np.fliplr(mask)):
            mid = bbox.shape[1] // 2
            keep_right = marker_pos[1] > mid
            half = mid
            if keep_right:
                keep = (bbox[:, mid:] != BG) * base
                keep[np.argwhere(bbox[:, mid:] == marker)[:, 0], np.argwhere(bbox[:, mid:] == marker)[:, 1]] = marker
                left_copy = np.fliplr(keep[:, 1:])
                out[r0 : r0 + left_copy.shape[0], c0 : c0 + left_copy.shape[1]][left_copy != 0] = left_copy[left_copy != 0]
                start_c = c0 + mid + half
                out[r0 : r0 + keep.shape[0], start_c : start_c + keep.shape[1]][keep != 0] = keep[keep != 0]
            else:
                keep = (bbox[:, : mid + 1] != BG) * base
                keep[np.argwhere(bbox[:, : mid + 1] == marker)[:, 0], np.argwhere(bbox[:, : mid + 1] == marker)[:, 1]] = marker
                right_copy = np.fliplr(keep[:, :-1])
                start_c = c0 - half
                out[r0 : r0 + keep.shape[0], start_c : start_c + keep.shape[1]][keep != 0] = keep[keep != 0]
                out[r0 : r0 + right_copy.shape[0], c0 + mid + 1 : c0 + mid + 1 + right_copy.shape[1]][right_copy != 0] = right_copy[right_copy != 0]
        else:
            mid = bbox.shape[0] // 2
            keep_down = marker_pos[0] > mid
            half = mid
            if keep_down:
                keep = (bbox[mid:, :] != BG) * base
                mr = np.argwhere(bbox[mid:, :] == marker)
                if len(mr):
                    keep[mr[:, 0], mr[:, 1]] = marker
                up_copy = np.flipud(keep[1:, :])
                out[r0 : r0 + up_copy.shape[0], c0 : c0 + up_copy.shape[1]][up_copy != 0] = up_copy[up_copy != 0]
                start_r = r0 + mid + half
                out[start_r : start_r + keep.shape[0], c0 : c0 + keep.shape[1]][keep != 0] = keep[keep != 0]
            else:
                keep = (bbox[: mid + 1, :] != BG) * base
                mr = np.argwhere(bbox[: mid + 1, :] == marker)
                if len(mr):
                    keep[mr[:, 0], mr[:, 1]] = marker
                down_copy = np.flipud(keep[:-1, :])
                start_r = r0 - half
                out[start_r : start_r + keep.shape[0], c0 : c0 + keep.shape[1]][keep != 0] = keep[keep != 0]
                out[r0 + mid + 1 : r0 + mid + 1 + down_copy.shape[0], c0 : c0 + down_copy.shape[1]][down_copy != 0] = down_copy[down_copy != 0]
    return out


def generate_input() -> np.ndarray:
    h, w = random.randint(22, 28), random.randint(22, 28)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7], 2)
    markers = random.sample([8, 9], len(colors))
    occupied = np.zeros((h, w), dtype=bool)
    for color, marker in zip(colors, markers):
        vertical = random.random() < 0.5
        if vertical:
            hh, ww = random.choice([(5, 5), (5, 7)])
            arr = np.zeros((hh, ww), dtype=int)
            arr[:, ww // 2] = color
            arr[hh // 2, :] = color
            side = random.choice(["left", "right"])
            mr = hh // 2
            mc = ww // 2 - 1 if side == "left" else ww // 2 + 1
        else:
            hh, ww = random.choice([(5, 5), (7, 5)])
            arr = np.zeros((hh, ww), dtype=int)
            arr[:, ww // 2] = color
            arr[hh // 2, :] = color
            side = random.choice(["up", "down"])
            mr = hh // 2 - 1 if side == "up" else hh // 2 + 1
            mc = ww // 2
        arr[mr, mc] = marker
        ah, aw = arr.shape
        pad = max(3, max(ah, aw) // 2 + 1)
        for _ in range(200):
            top = random.randint(pad, h - ah - pad - 1)
            left = random.randint(pad, w - aw - pad - 1)
            if occupied[top - 1 : top + ah + 1, left - 1 : left + aw + 1].any():
                continue
            grid[top : top + ah, left : left + aw][arr != 0] = arr[arr != 0]
            occupied[top : top + ah, left : left + aw] |= arr != 0
            break
    return grid
