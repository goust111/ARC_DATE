from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
PLATE_COLORS = [1, 2, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_plate_components(g: np.ndarray, frame_cells: set[tuple[int, int]]) -> list[dict]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    labels = -np.ones((h, w), dtype=int)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            color = int(g[sr, sc])
            if color in (BLACK, GREY) or seen[sr, sc] or (sr, sc) in frame_cells:
                continue

            cid = len(comps)
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                labels[r, c] = cid
                for rr, cc in _neighbors4(r, c):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or (rr, cc) in frame_cells:
                        continue
                    if int(g[rr, cc]) == color:
                        seen[rr, cc] = True
                        q.append((rr, cc))

            first = min(coords)
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "first": first,
                    "label": cid,
                }
            )

    return comps, labels


def _find_matrix_frame(g: np.ndarray) -> tuple[int, int, int, int, set[tuple[int, int]]]:
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    candidates = []

    for sr, sc in np.argwhere(g == GREY):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not seen[rr, cc] and int(g[rr, cc]) == GREY:
                    seen[rr, cc] = True
                    q.append((rr, cc))

        rs = [r for r, _ in coords]
        cs = [c for _, c in coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if side_h != side_w or side_h < 5:
            continue

        expected = set()
        for c in range(c0, c1 + 1):
            expected.add((r0, c))
            expected.add((r1, c))
        for r in range(r0, r1 + 1):
            expected.add((r, c0))
            expected.add((r, c1))
        actual = set(coords)
        if actual != expected:
            continue
        if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BLACK):
            continue

        # The puzzle marker is the square in the lower-right corner.
        score = (int(r1 == h - 1) + int(c1 == w - 1), r0 + c0, side_h)
        candidates.append((score, (r0, c0, r1, c1, actual)))

    assert candidates, "matrix frame not found"
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _adjacency(labels: np.ndarray, n: int) -> np.ndarray:
    h, w = map(int, labels.shape)
    adj = np.zeros((n, n), dtype=bool)
    for r in range(h):
        for c in range(w):
            a = int(labels[r, c])
            if a < 0:
                continue
            for rr, cc in ((r + 1, c), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w:
                    b = int(labels[rr, cc])
                    if b >= 0 and b != a:
                        adj[a, b] = True
                        adj[b, a] = True
    return adj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Encode map-region adjacencies inside the lower-right grey frame.

    Input structure:
    - Black background with several large single-color 4-connected map regions, each using a distinct non-grey color.
    - Some region pairs touch by a shared 4-neighbor edge, while other pairs are separated by black background.
    - A grey square frame sits in the lower-right corner; its black interior is (number of regions + 1) by (number of regions + 1).

    Rule:
    - Order the regions by the first pixel encountered in row-major scan order.
    - In the frame interior, write those region colors across the top header row and down the left header column.
    - For each off-diagonal matrix cell, write green when the corresponding two regions share an edge; otherwise leave it black.
    - Leave diagonal cells black and keep the original map, background, and grey frame unchanged.

    Key insight:
    - The colored shapes are a map: the needed information is the topology of shared borders, not size, color id, or distance.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, c1, frame_cells = _find_matrix_frame(g)
    inner_side = int(r1 - r0 - 1)

    comps, labels = _label_plate_components(g, frame_cells)
    colors = [int(comp["color"]) for comp in comps]
    assert len(colors) == len(set(colors)), "plate colors must be distinct"
    assert len(comps) + 1 == inner_side, "frame interior must be one larger than plate count"

    comps_sorted = sorted(comps, key=lambda comp: comp["first"])
    old_to_new = {int(comp["label"]): i for i, comp in enumerate(comps_sorted)}
    ordered_labels = -np.ones_like(labels)
    for old, new in old_to_new.items():
        ordered_labels[labels == old] = int(new)

    adj = _adjacency(ordered_labels, len(comps_sorted))

    ir0, ic0 = r0 + 1, c0 + 1
    out[ir0:r1, ic0:c1] = BLACK
    for i, comp in enumerate(comps_sorted):
        color = int(comp["color"])
        out[ir0, ic0 + i + 1] = color
        out[ir0 + i + 1, ic0] = color

    for i in range(len(comps_sorted)):
        for j in range(len(comps_sorted)):
            if i != j and bool(adj[i, j]):
                out[ir0 + i + 1, ic0 + j + 1] = GREEN

    return out


def _coarse_edges(coarse: np.ndarray, n: int):
    edges: set[tuple[int, int]] = set()
    vertical_contacts = 0
    horizontal_contacts = 0
    rows, cols = map(int, coarse.shape)
    for r in range(rows):
        for c in range(cols):
            a = int(coarse[r, c])
            if a < 0:
                continue
            if r + 1 < rows:
                b = int(coarse[r + 1, c])
                if b >= 0 and b != a:
                    edges.add(tuple(sorted((a, b))))
                    horizontal_contacts += 1
            if c + 1 < cols:
                b = int(coarse[r, c + 1])
                if b >= 0 and b != a:
                    edges.add(tuple(sorted((a, b))))
                    vertical_contacts += 1
    max_edges = n * (n - 1) // 2
    return edges, horizontal_contacts, vertical_contacts, max_edges


def _crop_coarse(coarse: np.ndarray) -> np.ndarray:
    pts = np.argwhere(coarse >= 0)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return coarse[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _make_coarse_map(rng: random.Random, n: int) -> np.ndarray | None:
    for _ in range(200):
        rows = rng.randint(max(4, n - 1), min(7, n + 3))
        cols = rng.randint(max(5, n), min(8, n + 4))
        coarse = -np.ones((rows, cols), dtype=int)

        all_cells = [(r, c) for r in range(rows) for c in range(cols)]
        first = rng.choice(all_cells)
        seeds = [first]
        coarse[first] = 0
        for label in range(1, n):
            empty = [(r, c) for r in range(rows) for c in range(cols) if int(coarse[r, c]) < 0]
            near = []
            for r, c in empty:
                if any(0 <= rr < rows and 0 <= cc < cols and int(coarse[rr, cc]) >= 0 for rr, cc in _neighbors4(r, c)):
                    near.append((r, c))
            pool = near if near and rng.random() < 0.75 else empty
            if not pool:
                return None
            cell = rng.choice(pool)
            coarse[cell] = label
            seeds.append(cell)

        targets = [rng.randint(2, 6) for _ in range(n)]
        if sum(targets) > int(rows * cols * 0.72):
            continue
        areas = [1 for _ in range(n)]
        failed = False
        for _grow_round in range(sum(targets) + rows * cols):
            pending = [i for i in range(n) if areas[i] < targets[i]]
            if not pending:
                break
            rng.shuffle(pending)
            progressed = False
            for label in pending:
                frontier = []
                for r, c in np.argwhere(coarse == label):
                    r, c = int(r), int(c)
                    for rr, cc in _neighbors4(r, c):
                        if 0 <= rr < rows and 0 <= cc < cols and int(coarse[rr, cc]) < 0:
                            frontier.append((rr, cc))
                if not frontier:
                    continue
                cell = rng.choice(frontier)
                coarse[cell] = label
                areas[label] += 1
                progressed = True
            if not progressed:
                failed = True
                break
        if failed or any(areas[i] < targets[i] for i in range(n)):
            continue

        coarse = _crop_coarse(coarse)
        edges, h_contacts, v_contacts, max_edges = _coarse_edges(coarse, n)
        if not (1 <= len(edges) < max_edges):
            continue
        if n >= 4 and (h_contacts == 0 or v_contacts == 0):
            continue
        return coarse
    return None


def _paint_map_from_coarse(coarse: np.ndarray, colors: list[int], scale: int) -> np.ndarray:
    rows, cols = map(int, coarse.shape)
    patch = np.zeros((rows * scale, cols * scale), dtype=int)
    for r in range(rows):
        for c in range(cols):
            label = int(coarse[r, c])
            if label >= 0:
                patch[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = int(colors[label])
    return patch


def _draw_frame(g: np.ndarray, n: int) -> None:
    outer = n + 3
    h, w = map(int, g.shape)
    r0 = h - outer
    c0 = w - outer
    g[r0 : h, c0] = GREY
    g[r0 : h, w - 1] = GREY
    g[r0, c0 : w] = GREY
    g[h - 1, c0 : w] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(3, 6)
    colors = rng.sample(PLATE_COLORS, n)
    coarse = _make_coarse_map(rng, n)
    if coarse is None:
        return None

    scale = rng.choice([2, 2, 3])
    patch = _paint_map_from_coarse(coarse, colors, scale)
    ph, pw = map(int, patch.shape)

    outer = n + 3
    top = rng.randint(1, 3)
    left = rng.randint(1, 3)
    gap = rng.randint(1, 3)
    right_margin = rng.randint(1, 4)
    extra_width = rng.randint(0, 4)

    h = top + ph + gap + outer
    w = max(left + pw + right_margin, outer + 2 + extra_width)
    if h > 30 or w > 30:
        return None

    g = np.zeros((h, w), dtype=int)
    g[top : top + ph, left : left + pw] = patch
    _draw_frame(g, n)

    try:
        out = main(g)
    except Exception:
        return None
    if out.shape != g.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, g):
        return None
    if int(np.count_nonzero(out == GREEN) - np.count_nonzero(g == GREEN)) <= 0:
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed to generate a valid input")
