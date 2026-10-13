from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5


def _frames(g: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = g.shape
    found = set()
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if g[r0, c0] != GREY:
                continue
            for r1 in range(r0 + 2, h):
                for c1 in range(c0 + 2, w):
                    if (
                        np.all(g[r0, c0 : c1 + 1] == GREY)
                        and np.all(g[r1, c0 : c1 + 1] == GREY)
                        and np.all(g[r0 : r1 + 1, c0] == GREY)
                        and np.all(g[r0 : r1 + 1, c1] == GREY)
                        and not np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GREY)
                    ):
                        found.add((r0, c0, r1, c1))
    return sorted(found)


def _trace(g: np.ndarray, a: tuple[int, int, int, int], bset: set[tuple[int, int, int, int]]) -> tuple[tuple[int, int, int, int] | None, list[int]]:
    r0, c0, r1, c1 = a
    mids = [((r0 + r1) // 2, c0 - 1, 0, -1), ((r0 + r1) // 2, c1 + 1, 0, 1), (r0 - 1, (c0 + c1) // 2, -1, 0), (r1 + 1, (c0 + c1) // 2, 1, 0)]
    for r, c, dr, dc in mids:
        if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
            continue
        if g[r, c] in {GREY, BG}:
            colors = []
            rr, cc = r, c
            while 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                if g[rr, cc] == BG:
                    break
                if g[rr, cc] != GREY:
                    colors.append(int(g[rr, cc]))
                for fr in bset:
                    fr0, fc0, fr1, fc1 = fr
                    if fr != a and fr0 <= rr <= fr1 and fc0 <= cc <= fc1 and (rr in {fr0, fr1} or cc in {fc0, fc1}):
                        return fr, colors
                rr += dr
                cc += dc
    return None, []


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send each template-frame pattern to its connected target frame, with line nodes choosing the copied color.

    Input structure:
    - Black background with small grey rectangular frames.
    - Some frames contain a colored template pattern; matching target frames are empty.
    - Each template frame connects to one target frame by a straight grey signal line that may contain colored nodes.

    Rule:
    - Follow the signal line from every non-empty template frame to its connected empty frame.
    - If the line has no colored nodes, copy the template pattern unchanged.
    - If the line has one node, recolor the template to that node color; if it has two, use the node closer to the target.
    - Paste the recolored pattern into the target frame at the same offset inside the frame.
    - Keep frames, signal lines, and colored nodes unchanged.

    Key insight:
    - The geometry routes the pattern, but the final copied color is controlled only by the last effective node on the connection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _frames(g)
    fset = set(frames)
    for fr in frames:
        r0, c0, r1, c1 = fr
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any((inner != BG) & (inner != GREY)):
            continue
        target, colors = _trace(g, fr, fset)
        if target is None:
            continue
        tr0, tc0, tr1, tc1 = target
        pattern = inner.copy()
        nz = pattern != BG
        if colors:
            pattern[nz] = colors[-1]
        th, tw = tr1 - tr0 - 1, tc1 - tc0 - 1
        ph, pw = pattern.shape
        h2, w2 = min(th, ph), min(tw, pw)
        view = out[tr0 + 1 : tr0 + 1 + h2, tc0 + 1 : tc0 + 1 + w2]
        mask = nz[:h2, :w2]
        view[mask] = pattern[:h2, :w2][mask]
    return out


def _draw_frame(g: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1, c1 = r0 + h - 1, c0 + w - 1
    g[r0, c0 : c1 + 1] = GREY
    g[r1, c0 : c1 + 1] = GREY
    g[r0 : r1 + 1, c0] = GREY
    g[r0 : r1 + 1, c1] = GREY


def generate_input() -> np.ndarray:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    g = np.zeros((h, w), dtype=int)
    for _ in range(random.randint(1, 2)):
        for _ in range(100):
            fh = random.randint(5, 7)
            fw = random.randint(5, 7)
            inner_h, inner_w = fh - 2, fw - 2
            gap = random.randint(3, 6)
            horiz = random.choice([True, False])
            if horiz and 2 * fw + gap + 2 < w and fh + 4 < h:
                row = random.randint(2, h - fh - 2)
                c0 = random.randint(1, w - (2 * fw + gap) - 1)
                t0 = c0 + fw + gap
                _draw_frame(g, row, c0, fh, fw)
                _draw_frame(g, row, t0, fh, fw)
                rr = row + inner_h // 2 + 1
                for cc in range(c0 + fw, t0):
                    g[rr, cc] = GREY
                line = [(rr, cc) for cc in range(c0 + fw, t0)]
                break
            if (not horiz) and 2 * fh + gap + 2 < h and fw + 4 < w:
                col = random.randint(2, w - fw - 2)
                r0 = random.randint(1, h - (2 * fh + gap) - 1)
                t0 = r0 + fh + gap
                _draw_frame(g, r0, col, fh, fw)
                _draw_frame(g, t0, col, fh, fw)
                cc = col + inner_w // 2 + 1
                for rr in range(r0 + fh, t0):
                    g[rr, cc] = GREY
                line = [(rr, cc) for rr in range(r0 + fh, t0)]
                row, c0 = r0, col
                break
        color = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
        pat = np.zeros((inner_h, inner_w), dtype=int)
        pat[:, 0] = color
        pat[-1, :] = color
        g[row + 1 : row + fh - 1, c0 + 1 : c0 + fw - 1][pat != 0] = color
        node_count = random.randint(0, 2)
        if node_count:
            picks = sorted(random.sample(range(len(line)), node_count))
            for idx in picks:
                rr, cc = line[idx]
                g[rr, cc] = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    return g
