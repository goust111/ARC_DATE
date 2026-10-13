import random
import numpy as np

Color = {
    "BLACK": 0,
    "BLUE": 1,
    "RED": 2,
    "GREEN": 3,
    "YELLOW": 4,
    "PINK": 6,
    "ORANGE": 7,
    "GRAY": 5,
}


def _flip_template(mask: np.ndarray, mode: str):
    out = mask
    if mode in {"right", "180"}:
        out = np.fliplr(out)
    if mode in {"up", "180"}:
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each source pixel into a translated, recolored block using a templated stamp.

    Input structure:
    - One gray separator column splits left template panel and right source panel.
    - Left panel contains a blue template.
    - Right panel contains colored source pixels with one adjacent gray direction marker.

    Rule:
    - For each source pixel, infer placement block by marker direction:
      left=identity, right=horizontal flip, up=vertical flip, down=180.
    - Stamp the template (replacing blue by source color) into the output block
      indexed by source pixel position.
    - Keep black elsewhere.

    Key insight:
    - The source pixel encodes both color and local orientation; marker direction controls template orientation at each block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gray_cols = [c for c in range(w) if np.all(g[:, c] == Color["GRAY"])]
    if not gray_cols:
        return g.copy()
    sep = int(gray_cols[0])

    left = g[:, :sep]
    right = g[:, sep + 1 :]
    n, m = left.shape
    if right.shape != left.shape:
        return g.copy()

    out = np.zeros((n * n, n * n), dtype=int)

    for sr, sc in np.argwhere((right != Color["BLACK"]) & (right != Color["GRAY"])):
        sr = int(sr)
        sc = int(sc)
        src_color = int(right[sr, sc])

        marker = None
        for mr, mc in [(sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)]:
            if 0 <= mr < n and 0 <= mc < n and right[mr, mc] == Color["GRAY"]:
                if mr == sr - 1:
                    marker = "up"
                elif mr == sr + 1:
                    marker = "down"
                elif mc == sc - 1:
                    marker = "left"
                else:
                    marker = "right"
                break
        if marker is None:
            continue

        if marker == "left":
            mode = ""
        elif marker == "right":
            mode = "right"
        elif marker == "up":
            mode = "up"
        else:
            mode = "180"

        templ = (left == Color["BLUE"]).astype(int)
        templ = _flip_template(templ, mode)

        r0 = sr * n
        c0 = sc * n
        for rr in range(n):
            for cc in range(n):
                if templ[rr, cc] == 1:
                    out[r0 + rr, c0 + cc] = src_color

    return out


def _random_connected_mask(n: int, size: int):
    cells = {(random.randrange(n), random.randrange(n))}
    edge = list(cells)
    while len(cells) < size:
        r, c = random.choice(edge)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            if 0 <= nr < n and 0 <= nc < n and (nr, nc) not in cells:
                cells.add((nr, nc))
                edge.append((nr, nc))
                if len(cells) >= size:
                    break
    return cells


def generate_input() -> np.ndarray:
    for _ in range(500):
        n = random.randint(4, 5)
        w = 2 * n + 1
        left = np.zeros((n, n), dtype=int)

        # asymmetric template
        sz = random.randint(3, 6)
        mask_coords = _random_connected_mask(n, sz)
        for r, c in mask_coords:
            left[r, c] = Color["BLUE"]
        if np.array_equal(left, np.fliplr(left)) or np.array_equal(left, np.flipud(left)):
            continue

        right = np.zeros((n, n), dtype=int)
        palette = [Color["RED"], Color["GREEN"], Color["YELLOW"], Color["ORANGE"], Color["PINK"]]
        n_src = random.randint(2, 4)
        src_colors = random.sample(palette, n_src)
        dirs = ["left", "right", "up", "down"]
        random.shuffle(dirs)
        dirs = dirs[:n_src]

        used = set()
        placed = 0
        for col in range(n):
            if placed >= n_src:
                break
            for row in range(1, n - 1):
                if placed >= n_src:
                    break
                if (row, col) in used:
                    continue
                if any(right[row + dr, col + dc] != Color["BLACK"] for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)] if 0 <= row + dr < n and 0 <= col + dc < n):
                    continue
                used.add((row, col))

        slots = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)]
        random.shuffle(slots)

        for color, d, (sr, sc) in zip(src_colors, dirs, slots[:n_src]):
            if d == "left":
                mr, mc = sr, sc - 1
            elif d == "right":
                mr, mc = sr, sc + 1
            elif d == "up":
                mr, mc = sr - 1, sc
            else:
                mr, mc = sr + 1, sc

            if not (0 <= mr < n and 0 <= mc < n):
                break
            if right[sr, sc] != Color["BLACK"]:
                break
            if right[mr, mc] != Color["BLACK"]:
                break
            right[sr, sc] = color
            right[mr, mc] = Color["GRAY"]
            placed += 1

        if placed < n_src:
            continue

        if len(set(src_colors)) != n_src:
            continue
        if len(set(dirs)) < 2:
            continue

        out = np.concatenate([left, np.full((n, 1), Color["GRAY"], dtype=int), right], axis=1)
        return out

    raise RuntimeError("generate_input: failed to sample valid case for puzzle 7")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, y.shape, y.max())
