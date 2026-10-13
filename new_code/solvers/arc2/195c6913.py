from __future__ import annotations
from collections import Counter
import numpy as np

def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate particles along a path with alternating turn direction at collisions.

    Input structure:
    - Top-left: 2x2 monochrome blocks define a repeating color cycle (left-to-right).
    - Isolated single pixels of a cycle color: start markers adjacent to the path.
    - A 2x2 monochrome block of a distinct color: end marker.

    Rule:
    - Each start marker launches a particle moving straight, painting cycle colors in order.
    - On hitting a boundary, paint that cell with the end color and turn 90°.
    - Turn direction alternates: odd collisions → left (CCW), even → right (CW).
    - If the scheduled turn is blocked, the particle stops.

    Key insight: The alternating left/right turn rule is the core mechanism; without it the path would be trivial.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    # Path is the most frequent color != template_bg.
    counts = Counter(map(int, g.flatten()))
    path_color = int(max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != int(template_bg)])[1])

    # Detect the repeating cycle from 2x2 blocks in the top-left region.
    pattern: list[int] = []
    pat_cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r in range(min(6, H - 1)):
        for c in range(min(25, W - 1)):
            if (int(r), int(c)) in seen:
                continue
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color):
                continue
            if int(g[int(r + 1), int(c)]) == v and int(g[int(r), int(c + 1)]) == v and int(g[int(r + 1), int(c + 1)]) == v:
                pattern.append(int(v))
                for rr, cc in [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]:
                    seen.add((int(rr), int(cc)))
                    pat_cells.append((int(rr), int(cc)))

    # Detect end color block (2x2).
    end_color = None
    end_pos = None
    pat_set = set(map(int, pattern))
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[int(r), int(c)])
            if v == int(template_bg) or v == int(path_color) or v in pat_set:
                continue
            if int(g[int(r + 1), int(c)]) == v and int(g[int(r), int(c + 1)]) == v and int(g[int(r + 1), int(c + 1)]) == v:
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    # Detect start markers: isolated pattern-color pixels adjacent to the path.
    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[int(r), int(c)])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == int(path_color):
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    out = g.copy()
    # Erase template pattern blocks and the end block.
    for r, c in pat_cells:
        out[int(r), int(c)] = int(template_bg)
    if end_pos is not None:
        er, ec = end_pos
        out[int(er) : int(er + 2), int(ec) : int(ec + 2)] = int(template_bg)

    # Initial direction preference (matches examples).
    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U

    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0  # collision count so far
        visited: set[tuple[int, int, int, int, int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == int(path_color):
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                r, c = nr, nc
                continue

            # Collision with non-path.
            if int(g[nr, nc]) == int(template_bg) and end_color is not None:
                out[nr, nc] = int(end_color)

            # Strict alternating turn: no fallback if blocked.
            left = (-int(dc), int(dr))  # CCW
            right = (int(dc), -int(dr))  # CW
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == int(path_color):
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

    return out

