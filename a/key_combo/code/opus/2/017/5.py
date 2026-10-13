import numpy as np

_RNG = np.random.default_rng(3105)


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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dr, y + dc
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            comps.append(comp)
    return comps


def _bbox(comp):
    rs = [r for r, c in comp]
    cs = [c for r, c in comp]
    return min(rs), max(rs), min(cs), max(cs)


def _is_rect_border(mask: np.ndarray, r0, r1, c0, c1):
    hh = (r1 - r0 + 1)
    ww = (c1 - c0 + 1)
    if hh < 3 or ww < 3:
        return False
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if (np.sum(mask[r0, c0:c1 + 1]) + np.sum(mask[r1, c0:c1 + 1]) +
            np.sum(mask[r0:r1 + 1, c0]) + np.sum(mask[r0:r1 + 1, c1]) -
            mask[r0, c0] - mask[r0, c1] - mask[r1, c0] - mask[r1, c1] <
            2 * (hh + ww) - 4):
        return False
    # perimeter should be mostly filled
    for c in range(c0, c1 + 1):
        if not (mask[r0, c] and mask[r1, c]):
            return False
    for r in range(r0, r1 + 1):
        if not (mask[r, c0] and mask[r, c1]):
            return False
    return True


def main(grid: np.ndarray):
    """Task 5 transform.

    1) Detect each gray rectangular frame from gray components.
    2) Count interior non-gray colors and choose key color where count equals number of color kinds.
    3) Recolor only the frame border to key color.
    4) Erase interior payload pixels and output the updated grid.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    for comp in _components(g == 5):
        if not comp:
            continue
        r0, r1, c0, c1 = _bbox(comp)
        if not _is_rect_border((g == 5), r0, r1, c0, c1):
            continue
        if (r1 - r0 < 3) or (c1 - c0 < 3):
            continue

        interior = g[r0 + 1:r1, c0 + 1:c1]
        vals = interior[interior != 0]
        vals = vals[vals != 5]
        if vals.size == 0:
            continue
        colors, counts = np.unique(vals, return_counts=True)
        N = len(colors)
        key = None
        for col, cnt in zip(colors, counts):
            if int(cnt) == N:
                key = int(col)
                break
        if key is None:
            continue

        out[r0, c0:c1 + 1] = key
        out[r1, c0:c1 + 1] = key
        out[r0:r1 + 1, c0] = key
        out[r0:r1 + 1, c1] = key
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                if out[rr, cc] != 0 and out[rr, cc] != 5:
                    out[rr, cc] = 0

    return out


def _draw_frame(g, r0, c0, r1, c1, color):
    g[r0, c0:c1 + 1] = color
    g[r1, c0:c1 + 1] = color
    g[r0:r1 + 1, c0] = color
    g[r0:r1 + 1, c1] = color


def generate_input():
    """Task 5 input generator.

    1) Place 3-5 disjoint gray 1-pixel frames.
    2) Fill each interior with 2-4 non-gray colors, total counts set so one color appears exactly N times.
    3) Ensure key colors are unique across frames for easy inversion.
    """

    h, w = 14, 14
    g = np.zeros((h, w), dtype=int)

    color_pool = [1, 2, 3, 4, 6, 7, 8, 9]
    frame_count = int(_RNG.integers(3, 6))
    _RNG.shuffle(color_pool)
    frame_colors = color_pool[:frame_count]
    key_colors = color_pool[frame_count:frame_count * 2]

    occupied = np.zeros((h, w), dtype=bool)

    for i in range(frame_count):
        for _ in range(80):
            hh = int(_RNG.integers(5, 10))
            ww = int(_RNG.integers(5, 10))
            if hh > h or ww > w:
                continue
            r0 = int(_RNG.integers(0, h - hh + 1))
            c0 = int(_RNG.integers(0, w - ww + 1))
            r1 = r0 + hh - 1
            c1 = c0 + ww - 1

            if np.any(occupied[max(0, r0 - 1):min(h, r1 + 2), max(0, c0 - 1):min(w, c1 + 2)]):
                continue

            _draw_frame(g, r0, c0, r1, c1, 5)
            occupied[max(0, r0 - 1):min(h, r1 + 2), max(0, c0 - 1):min(w, c1 + 2)] = True

            interior = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    interior.append((rr, cc))
            if len(interior) < 6:
                break

            N = int(_RNG.integers(2, min(4, len(interior) - 1) + 1))
            colors = color_pool[0:len(color_pool)]
            _RNG.shuffle(colors)
            used = colors[:N]
            key = key_colors[i % len(key_colors)]
            if key not in used:
                used[0] = key

            _RNG.shuffle(interior)
            # ensure exactly N appears N times
            for _ in range(N):
                rr, cc = interior.pop()
                g[rr, cc] = key

            for col in used:
                if col == key:
                    continue
                if not interior:
                    break
                cnt = int(_RNG.integers(1, min(5, len(interior) + 1)))
                for _ in range(cnt):
                    if not interior:
                        break
                    rr, cc = interior.pop()
                    if cnt == N and col == key:
                        continue
                    g[rr, cc] = col

            # leave remainder black
            break

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
