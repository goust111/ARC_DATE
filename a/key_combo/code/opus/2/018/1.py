import math
import numpy as np


_SOURCE = 5
_BACKGROUND = 0
_IGNORE = 7


_generate_calls = 0


def _shape_for_score(score):
    if score == 1:
        return np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    if score == 2:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    if score == 3:
        return np.array([[0, 0], [1, 0], [1, 1], [2, 1]])
    return np.array([[0, 0], [1, 0], [2, 0], [3, 0]])


def _build_valid_row_bands(h, rng):
    scores = [1, 2, 3, 4]
    bands = {}
    used = np.zeros(h, dtype=bool)
    for score in scores:
        span = score
        placed = False
        for _ in range(200):
            r0 = int(rng.randint(0, h - span + 1))
            if not used[r0 : r0 + span].any():
                bands[score] = r0
                used[r0 : r0 + span] = True
                placed = True
                break
        if not placed:
            for r0 in range(0, h - span + 1):
                if not used[r0 : r0 + span].any():
                    bands[score] = r0
                    used[r0 : r0 + span] = True
                    break
    return bands


def _find_components(grid):
    h, w = grid.shape
    colors = [1, 2, 3, 4, 6]
    comps = {}
    visited = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pix = []
            while stack:
                x, y = stack.pop()
                pix.append((x, y))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.setdefault(color, []).append(pix)

    out = []
    for color, c_list in comps.items():
        for pixels in c_list:
            out.append({"color": int(color), "pixels": pixels, "size": len(pixels), "side": int(math.isqrt(len(pixels)))})
    return out


def generate_input():
    global _generate_calls
    rng = np.random.RandomState(1000 + _generate_calls)
    _generate_calls += 1

    h = 18
    w = 18
    grid = np.zeros((h, w), dtype=int)
    grid[:, 0] = _SOURCE

    colors = [2, 1, 3, 4, 6]
    rng.shuffle(colors)
    scores = [1, 2, 3, 4]
    rng.shuffle(scores)
    bands = _build_valid_row_bands(h, rng)
    cols = np.array([1, 5, 9, 13])
    rng.shuffle(cols)

    for idx, score in enumerate(scores):
        color = colors[idx]
        base = _shape_for_score(score)
        top = bands[score]
        left = int(cols[idx])
        pts = base + np.array([top, left])
        for r, c in pts:
            if 0 <= r < h and 0 <= c < w - 1:
                grid[r, c] = color

    free = [(r, c) for r in range(h) for c in range(1, w) if grid[r, c] == 0]
    rng.shuffle(free)
    for r, c in free[:3 + rng.randint(0, 4)]:
        grid[r, c] = 7

    return grid


def _ray_scores(grid):
    h, w = grid.shape
    comps = _find_components(grid)
    color_to_index = {c["color"]: i for i, c in enumerate(comps)}
    hits = np.zeros(len(comps), dtype=int)
    for r in range(h):
        for c in range(1, w):
            p = int(grid[r, c])
            if p in (_BACKGROUND, _IGNORE):
                continue
            idx = color_to_index.get(p)
            if idx is None:
                continue
            hits[idx] += 1
            break

    for comp in comps:
        comp["score"] = int(hits[color_to_index[comp["color"]]])
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """1) Input has colored sprites on a gray source column.
2) Cast horizontal rays from each source row to score each sprite.
3) Convert each sprite into a solid square with side sqrt(area).
4) Sort by score and pack squares left-to-right, bottom-aligned."""

    g = np.array(grid)
    comps = _ray_scores(g)

    total_w = sum(int(math.isqrt(c["size"])) for c in comps) + max(0, len(comps) - 1)
    max_side = max((int(math.isqrt(c["size"])) for c in comps), default=0)
    out = np.zeros((max_side, total_w), dtype=int)

    for comp in sorted(comps, key=lambda it: (it["score"], it["color"])):
        s = int(math.isqrt(comp["size"]))
        x = int(comp.get("x", 0))
        if "x" not in comp:
            comp["x"] = x
    x = 0
    for comp in sorted(comps, key=lambda it: (it["score"], it["color"])):
        s = int(math.isqrt(comp["size"]))
        color = comp["color"]
        y = max_side - s
        out[y : y + s, x : x + s] = color
        x += s + 1

    return out


if __name__ == "__main__":
    sample = generate_input()
    print(main(sample).shape)
