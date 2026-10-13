import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(pts)
    return out


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_edge = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_edge = True
                        q.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _bbox_mask(component: list[tuple[int, int]]) -> np.ndarray:
    rs = [r for r, _ in component]
    cs = [c for _, c in component]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in component:
        mask[r - r0, c - c0] = True
    return mask


def _parse_legend_colors(grid: np.ndarray) -> list[int]:
    assert grid.shape[0] >= 3, "grid must contain two legend rows and a separator row"
    row0 = grid[0]
    row1 = grid[1]
    assert np.array_equal(row0, row1), "legend rows must match"

    colors: list[int] = []
    c = 0
    while c < grid.shape[1]:
        if int(row0[c]) == BG:
            c += 1
            continue
        color = int(row0[c])
        start = c
        while c < grid.shape[1] and int(row0[c]) == color:
            c += 1
        end = c
        assert end - start == 2, "each legend block must be 2 columns wide"
        assert np.all(row1[start:end] == color), "legend blocks must be solid 2x2"
        colors.append(color)
    assert colors, "legend must contain at least one color block"
    return colors


def _body_start(grid: np.ndarray) -> int:
    for r in range(2, grid.shape[0]):
        if np.all(grid[r] == BG):
            return r + 1
    raise AssertionError("missing separator row below the legend")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed shape from the top legend using its enclosed-hole count.

    Input structure:
    - The background is black.
    - The top of the grid contains a horizontal legend made of 1..K solid 2x2 color blocks,
      with one black column between adjacent blocks.
    - Below the legend and one black separator row are several grey 4-connected closed shapes;
      some shapes enclose one or more black holes.

    Rule:
    - Remove the legend rows and the separator row from the output.
    - For each grey shape, count the number of enclosed black holes inside its outer contour.
    - If the hole count is H >= 1, recolor all grey pixels of that shape with the color of the
      H-th legend block from the left; leave its holes black.
    - Shapes with no holes stay grey.

    Key insight:
    The legend is not keyed by object size or geometry: it is an ordered lookup table indexed by
    each shape's topological hole count.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors = _parse_legend_colors(g)
    body = g[_body_start(g) :].copy()

    for component in _components(body == GREY):
        holes = _hole_count(_bbox_mask(component))
        fill = GREY if holes == 0 or holes > len(legend_colors) else int(legend_colors[holes - 1])
        for r, c in component:
            body[r, c] = fill

    return body


def _normalize_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return sorted((r - r0, c - c0) for r, c in cells)


def _random_connected_cells(rng: random.Random, size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        r, c = rng.choice(tuple(cells))
        dr, dc = rng.choice(NEIGHBORS4)
        cells.add((r + dr, c + dc))
    return _normalize_cells(cells)


def _compact_polyomino(rng: random.Random, size: int, trials: int = 24) -> list[tuple[int, int]]:
    best = None
    best_score = None
    for _ in range(trials):
        cells = _random_connected_cells(rng, size)
        h = max(r for r, _ in cells) + 1
        w = max(c for _, c in cells) + 1
        score = (h * w, max(h, w), abs(h - w), rng.random())
        if best_score is None or score < best_score:
            best = cells
            best_score = score
    assert best is not None
    return best


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask
    if rng.random() < 0.5:
        out = np.flipud(out)
    if rng.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, rng.randrange(4))
    return out.copy()


def _make_holey_object(rng: random.Random, holes: int) -> np.ndarray:
    cells = _compact_polyomino(rng, holes)
    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    fine = np.zeros((2 * h + 1, 2 * w + 1), dtype=bool)
    for r, c in cells:
        rr, cc = 2 * r, 2 * c
        fine[rr, cc : cc + 3] = True
        fine[rr + 2, cc : cc + 3] = True
        fine[rr : rr + 3, cc] = True
        fine[rr : rr + 3, cc + 2] = True
    mask = np.kron(fine, np.ones((2, 2), dtype=bool))
    mask = _random_transform(mask, rng)
    assert _hole_count(mask) == holes
    return mask


def _make_solid_object(rng: random.Random) -> np.ndarray:
    for _ in range(32):
        cells = _compact_polyomino(rng, rng.randint(2, 6))
        h = max(r for r, _ in cells) + 1
        w = max(c for _, c in cells) + 1
        coarse = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            coarse[r, c] = True
        scale = rng.choice((2, 2, 3))
        mask = np.kron(coarse, np.ones((scale, scale), dtype=bool))
        mask = _random_transform(mask, rng)
        if _hole_count(mask) == 0:
            return mask
    raise RuntimeError("failed to generate a solid no-hole object")


def _place_objects(rng: random.Random, body_h: int, body_w: int, objects: list[np.ndarray]) -> np.ndarray | None:
    body = np.zeros((body_h, body_w), dtype=int)
    reserved = np.zeros((body_h, body_w), dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: objects[i].shape[0] * objects[i].shape[1], reverse=True)

    for idx in order:
        mask = objects[idx]
        h, w = mask.shape
        if h + 2 > body_h or w + 2 > body_w:
            return None
        positions = [(r, c) for r in range(1, body_h - h) for c in range(1, body_w - w)]
        rng.shuffle(positions)
        placed = False
        for r, c in positions:
            rr0, rr1 = max(0, r - 1), min(body_h, r + h + 1)
            cc0, cc1 = max(0, c - 1), min(body_w, c + w + 1)
            if reserved[rr0:rr1, cc0:cc1].any():
                continue
            body[r : r + h, c : c + w][mask] = GREY
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return body


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    k = rng.randint(1, 4)
    legend_colors = rng.sample(PALETTE_COLORS, k)

    counts = [0] + list(range(1, k + 1))
    if rng.random() < 0.8:
        counts.append(0)
    if rng.random() < 0.35:
        counts.append(rng.randint(1, k))
    rng.shuffle(counts)

    objects = [_make_solid_object(rng) if holes == 0 else _make_holey_object(rng, holes) for holes in counts]
    legend_span = 3 * k - 1
    needed_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for obj in objects)
    max_h = max(obj.shape[0] for obj in objects)
    max_w = max(obj.shape[1] for obj in objects)

    for _ in range(64):
        body_h = rng.randint(max(max_h + 2, 10), 27)
        body_w = rng.randint(max(max_w + 2, legend_span, 12), 30)
        if body_h * body_w < int(1.15 * needed_area):
            continue
        body = _place_objects(rng, body_h, body_w, objects)
        if body is None:
            continue

        grid = np.zeros((body_h + 3, body_w), dtype=int)
        legend_start = rng.randint(0, body_w - legend_span)
        for i, color in enumerate(legend_colors):
            c = legend_start + 3 * i
            grid[0:2, c : c + 2] = color
        grid[3:, :] = body
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
