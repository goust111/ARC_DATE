from __future__ import annotations

import numpy as np


BG = 8
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GEN_TRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _nearest_edge(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[str, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    distances = {
        "top": int(r0),
        "bottom": int(h - 1 - r1),
        "left": int(c0),
        "right": int(w - 1 - c1),
    }
    best = min(distances.values())
    winners = [edge for edge, dist in distances.items() if dist == best]
    assert len(winners) == 1, "nearest edge must be unique"
    return winners[0], int(best)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each monochrome foreground object to its nearest grid edge.

    The projection compresses the object into a one-pixel strip perpendicular
    to that edge, preserving the OR of occupied rows or columns.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    h, w = g.shape
    out = np.full_like(g, bg)
    projections: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []

    for comp in _components(g, bg):
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        bbox = tuple(map(int, comp["bbox"]))
        edge, distance = _nearest_edge(bbox, g.shape)
        r0, c0, r1, c1 = bbox
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        if edge == "top":
            strip = mask.any(axis=0)
            pts = np.array([(0, c0 + dc) for dc, on in enumerate(strip) if on], dtype=int)
        elif edge == "bottom":
            strip = mask.any(axis=0)
            pts = np.array([(h - 1, c0 + dc) for dc, on in enumerate(strip) if on], dtype=int)
        elif edge == "left":
            strip = mask.any(axis=1)
            pts = np.array([(r0 + dr, 0) for dr, on in enumerate(strip) if on], dtype=int)
        else:
            strip = mask.any(axis=1)
            pts = np.array([(r0 + dr, w - 1) for dr, on in enumerate(strip) if on], dtype=int)

        projections.append((distance, color, pts, bbox))

    # Nearer objects have priority, so draw farther ones first.
    projections.sort(key=lambda item: (-item[0], item[3]))
    for _distance, color, pts, _bbox in projections:
        if pts.size == 0:
            continue
        out[pts[:, 0], pts[:, 1]] = color
    return out


def _random_polyomino(rng: np.random.Generator, height: int, width: int, area: int) -> np.ndarray:
    cells = {(int(rng.integers(height)), int(rng.integers(width)))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_polyomino(rng, height, width, area)
        cells.add(frontier[int(rng.integers(len(frontier)))])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if len(set(rows)) < 2 or len(set(cols)) < 2:
        return _random_polyomino(rng, height, width, area)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _touches_existing(occ: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occ.shape
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                    return True
    return False


def _span_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _candidate_bbox(
    rng: np.random.Generator,
    edge: str,
    height: int,
    width: int,
    grid_h: int,
    grid_w: int,
    edge_spans: dict[str, list[tuple[int, int]]],
    want_overlap: bool,
) -> tuple[int, int] | None:
    if edge in ("top", "bottom"):
        max_d = max(1, (grid_h - height - 2) // 2)
        if max_d < 1 or grid_w - width < 5:
            return None
        for _ in range(80):
            if want_overlap and edge_spans[edge]:
                s0, s1 = edge_spans[edge][int(rng.integers(len(edge_spans[edge])))]
                lo = max(2, s0 - width + 1)
                hi = min(grid_w - width - 2, s1)
                if lo > hi:
                    continue
                c0 = int(rng.integers(lo, hi + 1))
            else:
                c0 = int(rng.integers(2, grid_w - width - 1))
            d = int(rng.integers(1, max_d + 1))
            r0 = d if edge == "top" else grid_h - height - d
            if not (d < c0 and d < grid_w - 1 - (c0 + width - 1)):
                continue
            if edge == "top" and not (d < grid_h - 1 - (r0 + height - 1)):
                continue
            if edge == "bottom" and not (d < r0):
                continue
            if edge_spans[edge] and want_overlap:
                hit = any(_span_overlap(c0, c0 + width - 1, s0, s1) for s0, s1 in edge_spans[edge])
                if not hit:
                    continue
            return r0, c0
    else:
        max_d = max(1, (grid_w - width - 2) // 2)
        if max_d < 1 or grid_h - height < 5:
            return None
        for _ in range(80):
            if want_overlap and edge_spans[edge]:
                s0, s1 = edge_spans[edge][int(rng.integers(len(edge_spans[edge])))]
                lo = max(2, s0 - height + 1)
                hi = min(grid_h - height - 2, s1)
                if lo > hi:
                    continue
                r0 = int(rng.integers(lo, hi + 1))
            else:
                r0 = int(rng.integers(2, grid_h - height - 1))
            d = int(rng.integers(1, max_d + 1))
            c0 = d if edge == "left" else grid_w - width - d
            if not (d < r0 and d < grid_h - 1 - (r0 + height - 1)):
                continue
            if edge == "left" and not (d < grid_w - 1 - (c0 + width - 1)):
                continue
            if edge == "right" and not (d < c0):
                continue
            if edge_spans[edge] and want_overlap:
                hit = any(_span_overlap(r0, r0 + height - 1, s0, s1) for s0, s1 in edge_spans[edge])
                if not hit:
                    continue
            return r0, c0
    return None


def _place_object(
    grid: np.ndarray,
    occ: np.ndarray,
    rng: np.random.Generator,
    edge: str,
    color: int,
    edge_spans: dict[str, list[tuple[int, int]]],
    want_overlap: bool,
) -> bool:
    grid_h, grid_w = grid.shape
    for _ in range(120):
        height = int(rng.integers(2, 6))
        width = int(rng.integers(2, 7))
        area = int(rng.integers(max(4, (height * width) // 2), height * width + 1))
        mask = _random_polyomino(rng, height, width, area)
        pos = _candidate_bbox(rng, edge, height, width, grid_h, grid_w, edge_spans, want_overlap)
        if pos is None:
            continue
        r0, c0 = pos
        cells = [(r0 + r, c0 + c) for r in range(height) for c in range(width) if mask[r, c]]
        if _touches_existing(occ, cells):
            continue
        for r, c in cells:
            grid[r, c] = color
            occ[r, c] = True
        if edge in ("top", "bottom"):
            edge_spans[edge].append((c0, c0 + width - 1))
        else:
            edge_spans[edge].append((r0, r0 + height - 1))
        return True
    return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_TRIES):
        grid_h = int(rng.integers(12, 23))
        grid_w = int(rng.integers(12, 23))
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        occ = np.zeros((grid_h, grid_w), dtype=bool)
        edge_spans = {edge: [] for edge in ("top", "bottom", "left", "right")}

        n_objs = int(rng.integers(3, 7))
        edges = [("top", "bottom", "left", "right")[int(rng.integers(4))] for _ in range(n_objs)]
        if len(set(edges)) == 1:
            edges[0] = "top"
            edges[1] = "left"
        force_overlap = bool(rng.integers(2))
        if force_overlap and n_objs >= 2:
            dup_edge = ("top", "bottom", "left", "right")[int(rng.integers(4))]
            i, j = sorted(rng.choice(n_objs, size=2, replace=False).tolist())
            edges[i] = dup_edge
            edges[j] = dup_edge

        if n_objs > len(PALETTE):
            continue
        colors = [PALETTE[i] for i in rng.permutation(len(PALETTE))[:n_objs]]

        placed = 0
        for i, (edge, color) in enumerate(zip(edges, colors, strict=False)):
            want_overlap = force_overlap and i > 0 and bool(edge_spans[edge]) and bool(rng.integers(2))
            if not _place_object(grid, occ, rng, edge, color, edge_spans, want_overlap):
                break
            placed += 1
        if placed != n_objs:
            continue

        comps = _components(grid, BG)
        if len(comps) != n_objs:
            continue
        try:
            nearest = [_nearest_edge(tuple(map(int, comp["bbox"])), grid.shape)[0] for comp in comps]
        except AssertionError:
            continue
        if sorted(nearest) != sorted(edges):
            continue

        out = main(grid)
        if np.array_equal(out, grid):
            continue
        if len(np.unique(grid)) < 4:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
