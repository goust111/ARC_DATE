import numpy as np


def _find_frames(grid):
    frames = []
    for color in [c for c in np.unique(grid) if c not in (0, 5, 7)]:
        ys, xs = np.where(grid == color)
        if len(ys) == 0:
            continue
        r1, r2 = int(ys.min()), int(ys.max())
        c1, c2 = int(xs.min()), int(xs.max())
        if r2 - r1 < 2 or c2 - c1 < 2:
            continue
        if not (np.all(grid[r1, c1 : c2 + 1] == color) and
                np.all(grid[r2, c1 : c2 + 1] == color) and
                np.all(grid[r1 : r2 + 1, c1] == color) and
                np.all(grid[r1 : r2 + 1, c2] == color)):
            continue
        frames.append((r1, r2, c1, c2))
    return frames


def main(grid):
    """
    1) Read all hollow frames.
    2) Odd marker count -> horizontal mirror, even -> vertical mirror.
    3) Cover erase: mirrored+original orange remove overlapping gray markers.
    4) Recolor frame border by remaining marker parity (4 if odd else 9).
    """
    g = np.array(grid, dtype=int).copy()
    for r1, r2, c1, c2 in _find_frames(g):
        inner = g[r1 + 1 : r2, c1 + 1 : c2]
        if inner.size == 0:
            continue

        markers = (inner == 5)
        N = int(markers.sum())
        odd = (N % 2 == 1)

        orange = (inner == 7)
        mirrored = np.zeros_like(inner)
        h, w = inner.shape
        oy, ox = np.where(orange)
        if odd:
            # horizontal mirror inside each row
            for y, x in zip(oy, ox):
                mirrored[y, w - 1 - x] = True
        else:
            # vertical mirror each column
            for y, x in zip(oy, ox):
                mirrored[h - 1 - y, x] = True

        all_orange = orange | mirrored
        # first keep existing orange, then erase markers under orange
        inner[all_orange] = 7
        inner[markers & (~all_orange)] = 5

        remaining = int(np.sum(inner == 5))
        border_color = 4 if (remaining % 2 == 1) else 9
        g[r1, c1 : c2 + 1] = border_color
        g[r2, c1 : c2 + 1] = border_color
        g[r1 : r2 + 1, c1] = border_color
        g[r1 : r2 + 1, c2] = border_color

    return g


def _gen_sprite(height, width, rng):
    length = int(rng.integers(3, 6))
    y = int(rng.integers(0, height))
    x = int(rng.integers(0, width))
    cells = {(y, x)}
    while len(cells) < length:
        yy, xx = list(cells)[-1]
        for _ in range(8):
            dy, dx = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            ny, nx = yy + dy, xx + dx
            if 0 <= ny < height and 0 <= nx < width:
                cells.add((ny, nx))
                yy, xx = ny, nx
                break
        else:
            y = int(rng.integers(0, height))
            x = int(rng.integers(0, width))
            cells.add((y, x))
            yy, xx = y, x
    return cells


def generate_input(seed=0, variant=None):
    if variant is None:
        if not hasattr(generate_input, "_variant"):
            generate_input._variant = 0
        variant = generate_input._variant
        generate_input._variant += 1
    rng = np.random.default_rng(int(seed) * 1000003 + int(variant) * 11 + 17)
    h = 24
    w = 24
    grid = np.zeros((h, w), dtype=int)

    frame_count = int(rng.integers(2, 5))
    colors = [1, 2, 3, 4, 6, 9]
    rng.shuffle(colors)
    colors = colors[:frame_count]
    frames = []
    slots = [(1, 1), (1, 13), (13, 1), (13, 13)]
    rng.shuffle(slots)

    for i, color in enumerate(colors):
        target_parity = i % 2
        hi = int(rng.integers(6, 10))
        wi = int(rng.integers(6, 10))
        fh = hi + 2
        fw = wi + 2
        base_r, base_c = slots[i]
        jr = int(rng.integers(0, max(1, min(3, h - fh - base_r + 1))))
        jc = int(rng.integers(0, max(1, min(3, w - fw - base_c + 1))))
        r1 = base_r + jr
        c1 = base_c + jc
        r2 = r1 + fh - 1
        c2 = c1 + fw - 1

        marker_n = int(rng.integers(2, 7))
        if marker_n % 2 != target_parity:
            marker_n = marker_n + 1 if marker_n < 6 else marker_n - 1

        grid[r1, c1 : c2 + 1] = color
        grid[r2, c1 : c2 + 1] = color
        grid[r1 : r2 + 1, c1] = color
        grid[r1 : r2 + 1, c2] = color

        sprite_cells = [(y, x) for y in range(hi) for x in range(wi)]
        rng.shuffle(sprite_cells)
        sprite = sprite_cells[:int(rng.integers(3, 7))]
        for y, x in sprite:
            grid[r1 + 1 + y, c1 + 1 + x] = 7

        interior_cells = [(r1 + 1 + y, c1 + 1 + x)
                          for y in range(hi) for x in range(wi)
                          if grid[r1 + 1 + y, c1 + 1 + x] == 0]
        rng.shuffle(interior_cells)
        for mr, mc in interior_cells[:marker_n]:
            grid[mr, mc] = 5

        frames.append((r1, r2, c1, c2))

    # ensure parity mix
    parities = [int(np.sum(grid[r1 + 1 : r2, c1 + 1 : c2] == 5) % 2) for r1, r2, c1, c2 in _find_frames(grid)]
    if len(set(parities)) <= 1 and parities:
        r1, r2, c1, c2 = _find_frames(grid)[0]
        target = r1 + 1
        grid[target, c1 + 1] = 5 if grid[target, c1 + 1] != 5 else 0

    return grid
