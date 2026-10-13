import numpy as np

_RNG = np.random.default_rng(3109)


def _components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def main(grid: np.ndarray):
    """Task 9 transform.

    1) Detect the symmetry axis as the unique full non-black row or column.
    2) Choose minority color among non-axis colors and mirror its pixels across axis.
    3) For each mirrored pair, draw axis-aligned connection in anomaly color.
    4) Clear all non-axis non-anomaly pixels.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    axis_row = None
    axis_col = None
    axis_color = None

    for r in range(h):
        vals = np.unique(g[r])
        if len(vals) == 1 and vals[0] != 0:
            axis_row = r
            axis_color = int(vals[0])
            break

    if axis_row is None:
        for c in range(w):
            vals = np.unique(g[:, c])
            if len(vals) == 1 and vals[0] != 0:
                axis_col = c
                axis_color = int(vals[0])
                break

    if axis_color is None:
        return out

    mask = (g != axis_color) & (g != 0)
    colors, counts = np.unique(g[mask], return_counts=True)
    if colors.size == 0:
        return out
    min_count = counts.min()
    anomaly = int(colors[counts == min_count][0])

    # mirror anomaly pixels
    anm_mask = g == anomaly
    pts = np.argwhere(anm_mask)
    mirrored = []
    for r, c in pts:
        if axis_row is not None:
            rr = 2 * axis_row - r
            cc = c
            if 0 <= rr < h:
                out[rr, c] = anomaly
                mirrored.append((r, c, rr, c))
        else:
            rr = r
            cc = 2 * axis_col - c
            if 0 <= cc < w:
                out[r, cc] = anomaly
                mirrored.append((r, c, r, cc))

    # draw connections
    if axis_row is not None:
        for r0, c0, r1, c1 in mirrored:
            r0 = int(r0)
            c0 = int(c0)
            r1 = int(r1)
            c1 = int(c1)
            for rr in range(min(r0, r1), max(r0, r1) + 1):
                out[rr, c0] = anomaly
    else:
        for r0, c0, r1, c1 in mirrored:
            for cc in range(min(c0, c1), max(c0, c1) + 1):
                out[r0, cc] = anomaly

    # erase all non-axis non-anomaly pixels
    for r in range(h):
        for c in range(w):
            v = g[r, c]
            if v == axis_color or v == anomaly:
                continue
            out[r, c] = 0

    return out


def generate_input():
    """Task 9 input generator.

    1) Randomly choose horizontal or vertical axis, fill with one color.
    2) Place 2-3 non-axis colors, with anomaly color having least pixels.
    3) Ensure anomaly pixels are only on one side of axis and unique in orth coordinate.
    """

    h = w = 15
    g = np.zeros((h, w), dtype=int)

    axis_is_row = bool(_RNG.integers(2))
    axis_color = 4
    if axis_is_row:
        axis = int(_RNG.integers(4, 11))
        g[axis, :] = axis_color
        half = axis
    else:
        axis = int(_RNG.integers(4, 11))
        g[:, axis] = axis_color

    other_colors = [1, 2, 3]
    _RNG.shuffle(other_colors)
    counts = [2, 4, 5]
    colors = list(other_colors)
    # anomaly first is least count
    _RNG.shuffle(colors)
    count_map = {colors[0]: counts[0], colors[1]: counts[1], colors[2]: counts[2]}

    # anomaly side uniqueness condition
    used_coord = set()
    for color in colors:
        for _ in range(count_map[color]):
            placed = False
            for _ in range(60):
                if axis_is_row:
                    r = int(_RNG.integers(0, half))
                    c = int(_RNG.integers(0, w))
                    if color == colors[0]:
                        coord = c
                    else:
                        coord = c
                    if coord in used_coord and color == colors[0]:
                        continue
                else:
                    c = int(_RNG.integers(0, axis))
                    r = int(_RNG.integers(0, h))
                    coord = r
                    if coord in used_coord and color == colors[0]:
                        continue

                # non-axis line is reserved for anomalies, others can be both sides
                if axis_is_row:
                    if _RNG.random() < 0.5:
                        rr = int(_RNG.integers(0, half))
                        side = 'u'
                        r0 = rr
                    else:
                        rr = int(_RNG.integers(axis + 1, h))
                        r0 = rr
                    c0 = c
                    if r0 == axis:
                        continue
                    if g[r0, c0] != 0:
                        continue
                    g[r0, c0] = color
                else:
                    if _RNG.random() < 0.5:
                        cc = int(_RNG.integers(0, axis))
                    else:
                        cc = int(_RNG.integers(axis + 1, w))
                    r0 = r
                    c0 = cc
                    if c0 == axis:
                        continue
                    if g[r0, c0] != 0:
                        continue
                    g[r0, c0] = color

                if color == colors[0]:
                    used_coord.add(coord)
                placed = True
                break
            if not placed:
                continue

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
