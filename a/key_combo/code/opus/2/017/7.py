import numpy as np

_RNG = np.random.default_rng(3107)


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
    """Task 7 transform.

    1) Read grayscale templates in legend and map each by number of gray pixels.
    2) Extract each cup, infer its tail direction, and get mouth content sequence.
    3) Count content to pick template; recolor template gray cells by ordered content colors.
    4) Erase cup contents and place colored template outward from the tail tip.
    """

    g = np.array(grid, dtype=int)
    h, w = g.shape

    # Parse legend templates.
    top = np.argwhere(g[:4] == 5)
    template_infos = {}
    for cc in np.unique(top[:, 1]):
        c0, c1 = cc, cc
        r0, r1 = 0, 3
        # find connected components in small legend area with bbox <= 3x3
    vis = np.zeros_like(g[:4], dtype=bool)
    comps = _components(g[:4, :4] == 5)
    # robust scan on full top rows
    vis = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if r >= 4 and c >= 4:
                pass
            if not (g[r, c] == 5) or vis[r, c]:
                continue
            stack = [(r, c)]
            vis[r, c] = True
            comp = []
            while stack:
                x, y = stack.pop()
                comp.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and g[nx, ny] == 5 and not vis[nx, ny]:
                        vis[nx, ny] = True
                        stack.append((nx, ny))
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r0 < 4 and (r1 - r0 + 1) <= 4 and (c1 - c0 + 1) <= 4:
                mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
                for rr, cc in comp:
                    mask[rr - r0, cc - c0] = True
                cnt = int(mask.sum())
                template_infos[cnt] = mask
                components.append((cnt, r0, c0, mask))

    # parse cups by color
    colors = [v for v in np.unique(g) if v not in (0, 5)]
    # colors maybe include content colors; find non-unique components with size>=5
    mask_red = np.zeros_like(g, dtype=bool)
    # find all components excluding 0 and 5 and infer probable cup colors by ring-like shapes
    out = g.copy()

    cup_pixels = np.zeros_like(g, dtype=bool)
    for color in colors:
        for comp in _components(g == color):
            if len(comp) < 8:
                continue
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            h0 = r1 - r0 + 1
            h1 = c1 - c0 + 1
            if h0 < 4 or h1 < 4:
                continue
            # tail must be one-cell outside ext.
            if (r0 - 1 >= 0 and out[r0 - 1, (c0 + c1) // 2] == color) or \
               (r1 + 1 < out.shape[0] and out[r1 + 1, (c0 + c1) // 2] == color) or \
               (c0 - 1 >= 0 and out[(r0 + r1) // 2, c0 - 1] == color) or \
               (c1 + 1 < out.shape[1] and out[(r0 + r1) // 2, c1 + 1] == color):
                cup_pixels[r0:r1 + 1, c0:c1 + 1] = g[r0:r1 + 1, c0:c1 + 1] == color

    comps = _components(cup_pixels)
    for comp in comps:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        color = int(g[comp[0][0], comp[0][1]])

        # detect tail pixel and direction
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        direction = None
        tail = None
        if r0 > 0 and out[r0 - 1, center[1]] == color:
            direction = (-1, 0)
            tail = (r0 - 1, center[1])
        elif r1 + 1 < h and out[r1 + 1, center[1]] == color:
            direction = (1, 0)
            tail = (r1 + 1, center[1])
        elif c0 > 0 and out[center[0], c0 - 1] == color:
            direction = (0, -1)
            tail = (center[0], c0 - 1)
        elif c1 + 1 < w and out[center[0], c1 + 1] == color:
            direction = (0, 1)
            tail = (center[0], c1 + 1)

        if direction is None or tail is None:
            continue

        mouth_is_vertical = direction in [(1, 0), (-1, 0)]
        if direction == (-1, 0):
            mouth_row = r1 - 1
            seq = [(r, c) for r, c in comp if r == mouth_row and out[r, c] == color]
        elif direction == (1, 0):
            mouth_row = r0 + 1
            seq = [(r, c) for r, c in comp if r == mouth_row and out[r, c] == color]
        elif direction == (0, -1):
            mouth_col = c1 - 1
            seq = [(r, c) for r, c in comp if c == mouth_col and out[r, c] == color]
        else:
            mouth_col = c0 + 1
            seq = [(r, c) for r, c in comp if c == mouth_col and out[r, c] == color]

        # use all inner non-frame pixels as content
        inner_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1) if out[r, c] != 0 and out[r, c] != color]
        if not inner_cells:
            inner_cells = [p for p in seq]

        if direction in [(-1, 0), (1, 0)]:
            inner_cells = sorted(inner_cells, key=lambda x: x[1])
        else:
            inner_cells = sorted(inner_cells, key=lambda x: x[0])

        colors_seq = [int(out[r, c]) for r, c in inner_cells]
        N = len(colors_seq)

        # erase content
        for rr, cc in inner_cells:
            out[rr, cc] = 0

        if N == 0 or N not in template_infos:
            continue
        tm = template_infos[N]
        t_h, t_w = tm.shape
        anchor_r = tail[0] + direction[0]
        anchor_c = tail[1] + direction[1]
        top = anchor_r - (t_h // 2)
        left = anchor_c - (t_w // 2)

        if top < 0 or left < 0 or top + t_h > h or left + t_w > w:
            continue

        idx = 0
        for rr in range(t_h):
            for cc in range(t_w):
                if not tm[rr, cc]:
                    continue
                if idx >= len(colors_seq):
                    break
                out[top + rr, left + cc] = colors_seq[idx % len(colors_seq)]
                idx += 1

    return out


def _draw_template(g, top, left, pattern, color):
    for r in range(pattern.shape[0]):
        for c in range(pattern.shape[1]):
            if pattern[r, c]:
                g[top + r, left + c] = 2


def generate_input():
    """Task 7 input generator.

    1) Generate 2-3 three-gray-pixel templates in a top legend.
    2) Generate 2-3 cups with one spike and one open side determined by tail direction.
    3) Choose content counts to match template gray counts and fill open mouth with content colors.
    """

    h, w = 16, 16
    g = np.zeros((h, w), dtype=int)

    cup_colors = [2, 6, 1]
    _RNG.shuffle(cup_colors)

    # templates
    templates = [
        np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 0, 0]], dtype=bool),
        np.array([[0, 1, 1],
                  [1, 1, 0],
                  [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1],
                  [1, 0, 1],
                  [1, 0, 1]], dtype=bool),
    ]
    counts = [int(t.sum()) for t in templates]
    for i, t in enumerate(templates):
        top = i * 5
        g[0:3, top:top + 3] = 0
        for r in range(3):
            for c in range(3):
                if t[r, c]:
                    g[r + 1, top + c] = 5
        g[0, top + 1] = 2 if i == 0 else (3 if i == 1 else 1)

    # cup regions
    count = int(_RNG.integers(2, 4))
    cup_specs = []
    used = []
    for i in range(count):
        cup_specs.append((cup_colors[i], counts[i]))

    for i, (ccolor, need) in enumerate(cup_specs):
        dir_idx = int(_RNG.integers(4))
        direction = [(-1, 0), (1, 0), (0, -1), (0, 1)][dir_idx]
        r0, c0 = 3 + i * 4, 2
        r1 = r0 + 4
        c1 = c0 + 6
        # outer border
        for c in range(c0, c1 + 1):
            g[r0, c] = ccolor
            g[r1, c] = ccolor
        for r in range(r0, r1 + 1):
            g[r, c0] = ccolor
            g[r, c1] = ccolor

        # open mouth opposite direction
        if direction == (-1, 0):
            g[r1, c0 + 1:c1] = 0
            mouth_cells = [(r1 - 1, c) for c in range(c0 + 1, c1)]
            anchor = (r0 - 1, (c0 + c1) // 2)
        elif direction == (1, 0):
            g[r0, c0 + 1:c1] = 0
            mouth_cells = [(r0 + 1, c) for c in range(c0 + 1, c1)]
            anchor = (r1 + 1, (c0 + c1) // 2)
        elif direction == (0, -1):
            g[r0 + 1:r1, c1] = 0
            mouth_cells = [(r, c1 - 1) for r in range(r0 + 1, r1)]
            anchor = ((r0 + r1) // 2, c0 - 1)
        else:
            g[r0 + 1:r1, c0] = 0
            mouth_cells = [(r, c0 + 1) for r in range(r0 + 1, r1)]
            anchor = ((r0 + r1) // 2, c1 + 1)

        _RNG.shuffle(mouth_cells)
        need = min(need, len(mouth_cells))
        palette = [1, 3, 4, 6, 7, 8, 2, 5]
        for j in range(need):
            rr, cc = mouth_cells[j]
            g[rr, cc] = palette[(i * 3 + j) % len(palette)]

    return g


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample))
