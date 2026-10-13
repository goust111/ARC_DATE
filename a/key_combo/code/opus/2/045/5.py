import random
from collections import deque

import numpy as np

BG = 0
GRAY = 5
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _norm_key(cells):
    r0, r1, c0, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=np.int8)
    for r, c in cells:
        mask[r - r0, c - c0] = 1
    return tuple(mask.flatten())


def _find_legend_frame(grid: np.ndarray):
    h, w = grid.shape
    gray_mask = grid == GRAY
    candidates = []

    for comp in _components(gray_mask):
        if len(comp) < 12:
            continue
        r0, r1, c0, c1 = _bbox(comp)
        if r0 == 0 or c0 == 0:
            # prompt says frame in 左上角, but avoid frame hugging the outer edge entirely
            continue
        if not all((r0, c) in set(comp) for c in range(c0, c1 + 1)):
            pass
        if not all((r1, c) in set(comp) for c in range(c0, c1 + 1)):
            continue
        if not all((r, c0) in set(comp) for r in range(r0, r1 + 1)):
            continue
        if not all((r, c1) in set(comp) for r in range(r0, r1 + 1)):
            continue
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area < 25:
            continue
        candidates.append((r0, r1, c0, c1))

    if not candidates:
        # Fallback: first gray component with a rectangular hollow bbox.
        for comp in _components(gray_mask):
            if len(comp) >= 8:
                return _bbox(comp)
        return None
    return sorted(candidates, key=lambda t: (t[0], t[1], t[2]))[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape

    frame = _find_legend_frame(g)
    if frame is None:
        return g
    fr0, fr1, fc0, fc1 = frame
    frame_cells = set((r, c) for r in range(fr0, fr1 + 1) for c in range(fc0, fc1 + 1))

    legend_mask = np.zeros((h, w), dtype=bool)
    if fr1 - fr0 > 2 and fc1 - fc0 > 2:
        legend_mask[fr0 + 1 : fr1, fc0 + 1 : fc1] = True

    legend_shape_cells = legend_mask & (g != BG) & (g != GRAY)
    legend_map = {}
    for comp in _components(legend_shape_cells):
        if not comp:
            continue
        color = int(g[comp[0]])
        key = _norm_key(comp)
        legend_map.setdefault(key, color)

    marker_mask = (g == GRAY) & (~legend_mask)
    marker_comps = _components(marker_mask)
    marker_cells = set()
    for comp in marker_comps:
        marker_cells.update(comp)

    groups = {}
    for comp in marker_comps:
        if not comp:
            continue
        if len(comp) < 2 or len(comp) > 4:
            continue
        key = _norm_key(comp)
        groups.setdefault(key, []).append(comp)

    for comps in groups.values():
        if len(comps) != 2:
            continue
        c0, c1 = comps
        color = legend_map.get(_norm_key(c0))
        if color is None:
            continue

        r0a, r1a, c0a, c1a = _bbox(c0)
        r0b, r1b, c0b, c1b = _bbox(c1)
        start = ((r0a + r1a) // 2, (c0a + c1a) // 2)
        end = ((r0b + r1b) // 2, (c0b + c1b) // 2)

        sr, sc = start
        er, ec = end

        # Horizontal first, then vertical.
        if sc <= ec:
            h_cols = range(sc, ec + 1)
        else:
            h_cols = range(sc, ec - 1, -1)
        for c in h_cols:
            if (sr, c) in marker_cells or (sr, c) in frame_cells:
                continue
            g[sr, c] = color

        if sr <= er:
            v_rows = range(sr + 1, er + 1)
        else:
            v_rows = range(sr - 1, er - 1, -1)
        for r in v_rows:
            if (r, ec) in marker_cells or (r, ec) in frame_cells:
                continue
            g[r, ec] = color

    return g


def _place_shape(grid: np.ndarray, top: int, left: int, shape, color: int):
    h, w = grid.shape
    for dr, dc in shape:
        r = top + dr
        c = left + dc
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def generate_input() -> np.ndarray:
    h, w = 16, 16
    g = np.zeros((h, w), dtype=int)

    # Legend frame in top-left.
    fr0, fc0 = 0, 0
    fr1, fc1 = 7, 9  # half-open outer bounds below are +1
    # Draw 1-cell gray rectangle.
    for c in range(fc0, fc1 + 1):
        g[fr0, c] = GRAY
        g[fr1, c] = GRAY
    for r in range(fr0, fr1 + 1):
        g[r, fc0] = GRAY
        g[r, fc1] = GRAY

    inside_r0, inside_r1 = fr0 + 1, fr1
    inside_c0, inside_c1 = fc0 + 1, fc1

    legend_colors = [1, 2, 3, 4]
    random.shuffle(legend_colors)

    legend_shapes = [
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0)],
        [(0, 1), (1, 1), (1, 2)],
    ]
    random.shuffle(legend_shapes)
    chosen_shapes = []

    used = np.zeros((h, w), dtype=bool)
    used[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    for color in legend_colors:
        shape = legend_shapes.pop()
        chosen_shapes.append(shape)
        sh_h = max(r for r, _ in shape) + 1
        sh_w = max(c for _, c in shape) + 1

        success = False
        for _ in range(200):
            top = random.randint(inside_r0, inside_r1 - sh_h)
            left = random.randint(inside_c0, inside_c1 - sh_w)
            cells = {(top + dr, left + dc) for dr, dc in shape}
            if any(used[r, c] for r, c in cells):
                continue
            for r, c in cells:
                g[r, c] = color
                used[r, c] = True
            success = True
            break

        if not success:
            # Deterministic fallback in the legend area.
            top = inside_r0
            left = inside_c0
            for r, c in ((top + dr, left + dc) for dr, dc in shape):
                g[r, c] = color
                used[r, c] = True

    # Place six marker objects (3 pairs) outside the legend frame.
    marker_shapes = chosen_shapes[:]
    random.shuffle(marker_shapes)
    marker_used = used.copy()
    marker_occupied = set()

    # Candidate anchor columns avoid legend area and encourage short non-crossing paths.
    anchor_cols = [11, 12, 13]

    for i, shape in enumerate(marker_shapes):
        color_cells = shape
        sh_h = max(r for r, _ in color_cells) + 1
        sh_w = max(c for _, c in color_cells) + 1

        # Two matching markers of this shape.
        for j in range(2):
            placed = False
            for _ in range(300):
                if i == 0:
                    top_range = (1, 4)
                elif i == 1:
                    top_range = (6, 9)
                else:
                    top_range = (10, 13)

                top = random.randint(top_range[0], top_range[1])
                left = random.randint(10, w - sh_w - 1)
                if j == 1:
                    left = min(w - sh_w - 1, left + 3)

                cells = {(top + dr, left + dc) for dr, dc in color_cells}
                if any(marker_used[r, c] for r, c in cells):
                    continue
                if any(r >= h or c >= w or c < 0 for r, c in cells):
                    continue
                # Keep markers out of legend box.
                if any(fc0 <= c <= fc1 and fr0 <= r <= fr1 for r, c in cells):
                    continue

                for r, c in cells:
                    g[r, c] = GRAY
                    marker_used[r, c] = True
                    marker_occupied.add((r, c))
                placed = True
                break

            if not placed:
                # fallback nearby
                top = 10 + i
                left = 11 + j
                for r, c in ((top + dr, left + dc) for dr, dc in color_cells):
                    if 0 <= r < h and 0 <= c < w:
                        g[r, c] = GRAY
                        marker_used[r, c] = True
                        marker_occupied.add((r, c))

    return g
