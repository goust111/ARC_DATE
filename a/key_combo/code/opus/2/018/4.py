import numpy as np


_BG = 0
_SEP = 5

_generate_calls = 0


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(5000 + _generate_calls)
    _generate_calls += 1

    h = 20
    w = 16
    grid = np.zeros((h, w), dtype=int)
    grid[:, 8] = _SEP

    # left panel glyphs
    colors = [2, 1, 3, 4]
    rng.shuffle(colors)
    areas = [3, 4, 5, 6]
    rng.shuffle(areas)
    glyph_count = 3 + rng.randint(0, 2)
    glyph_count = max(3, min(4, glyph_count))
    rows = [1, 3, 5, 7]

    for i in range(glyph_count):
        col = 1 + rng.randint(0, 6)
        color = colors[i]
        a = areas[i]
        shape = np.zeros((3, 3), dtype=int)
        if a == 3:
            shape[0, 0] = shape[0, 1] = shape[1, 0] = color
        elif a == 4:
            shape[:2, :2] = color
        elif a == 5:
            shape[0, :] = shape[1, :] = color
            shape[1, 1] = 0
            shape[2, 1] = color
        else:
            shape[:] = color
            idx = rng.randint(0, 9)
            shape.flat[idx] = 0
            shape[shape == color] = color
            while shape.sum() // color != 6:
                idx = rng.randint(0, 9)
                shape.flat[idx] = color
                if shape.sum() // color > 6:
                    shape.flat[idx // 2] = 0
        for r in range(3):
            for c in range(3):
                rr = rows[i] + r
                cc = col + c
                if shape[r, c] != 0:
                    grid[rr, cc] = color

    # right tokens on top row
    token_colors = colors[:glyph_count]
    cols = rng.choice(np.arange(9, 16), size=glyph_count, replace=False)
    rng.shuffle(cols)
    for c, color in zip(cols, token_colors):
        if c < w:
            grid[0, c] = color

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """1) Decode glyph areas in the left panel as instructions.
2) Match each instruction to the same color token in the right panel.
3) Paint floor cells in the right panel for every decoded instruction.
4) Let every token fall under gravity until it lands on its floor or another token; return only right panel."""

    g = np.array(grid)
    h, w = g.shape
    sep_col = int(np.where((g == _SEP).any(axis=0))[0][0])

    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    hr, wr = right.shape

    # decode glyphs by color and area
    visited = np.zeros_like(left, dtype=bool)
    areas = {}
    for r in range(left.shape[0]):
        for c in range(left.shape[1]):
            color = int(left[r, c])
            if color == _BG or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cnt = 0
            while stack:
                x, y = stack.pop()
                if left[x, y] != color:
                    continue
                cnt += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < left.shape[0] and 0 <= nc < left.shape[1] and not visited[nr, nc] and left[nr, nc] == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            areas[color] = h - cnt

    out = np.zeros((hr, wr), dtype=int)
    # place floor pixels
    for c in range(wr):
        color = int(right[0, c])
        if color == _BG:
            continue
        if color not in areas:
            continue
        floor = max(0, min(hr - 1, areas[color]))
        if 0 <= floor < hr:
            out[floor, c] = color

    # drop tokens
    tokens = []
    for c in range(wr):
        for r in range(1, hr):
            if right[r, c] != _BG:
                tokens.append((r, c, int(right[r, c])))

    for r0, c, color in tokens:
        rr = r0
        while rr + 1 < hr and out[rr + 1, c] == _BG:
            rr += 1
        out[rr, c] = color

    return out


if __name__ == "__main__":
    s = generate_input()
    print(main(s).shape)
