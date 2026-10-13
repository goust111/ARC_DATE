from __future__ import annotations

import numpy as np


BG = 7
GEN_ATTEMPTS = 400
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            touches = set()
            if np.any(arr[:, 0] == 0):
                touches.add("top")
            if np.any(arr[:, 0] == h - 1):
                touches.add("bottom")
            if np.any(arr[:, 1] == 0):
                touches.add("left")
            if np.any(arr[:, 1] == w - 1):
                touches.add("right")
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "touches": touches,
                }
            )
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Delete interior objects, compress single-edge objects to a border line,
    and fill the bbox of two-edge objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    vals, cnts = np.unique(g, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out = np.full((h, w), bg, dtype=int)

    for comp in _find_components(g, bg):
        color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        touches = comp["touches"]
        edge_count = len(touches)
        if edge_count == 0:
            continue
        if edge_count == 1:
            edge = next(iter(touches))
            if edge == "top":
                out[r0, c0 : c1 + 1] = color
            elif edge == "bottom":
                out[r1, c0 : c1 + 1] = color
            elif edge == "left":
                out[r0 : r1 + 1, c0] = color
            else:
                out[r0 : r1 + 1, c1] = color
            continue
        if edge_count == 2:
            out[r0 : r1 + 1, c0 : c1 + 1] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _random_grow(
    rng: np.random.Generator,
    height: int,
    width: int,
    target_area: int,
    seeds: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = set(seeds)
    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return set(seeds)
        uniq = list(dict.fromkeys(frontier))
        weights = []
        for r, c in uniq:
            edge_bias = 1.0
            if r in (0, height - 1) or c in (0, width - 1):
                edge_bias += 0.6
            weights.append(edge_bias)
        pick = uniq[int(rng.choice(len(uniq), p=np.asarray(weights) / np.sum(weights)))]
        cells.add(pick)
    return cells


def _make_interior_shape(rng: np.random.Generator) -> set[tuple[int, int]]:
    height = int(rng.integers(2, 6))
    width = int(rng.integers(2, 6))
    area = int(rng.integers(max(2, (height * width) // 3), height * width))
    seed = {(int(rng.integers(0, height)), int(rng.integers(0, width)))}
    return _normalize(_random_grow(rng, height, width, area, seed))


def _make_single_edge_shape(rng: np.random.Generator, edge: str) -> set[tuple[int, int]]:
    if edge in ("top", "bottom"):
        width = int(rng.integers(2, 7))
        height = int(rng.integers(2, 6))
        span = int(rng.integers(1, width + 1))
        start = int(rng.integers(0, width - span + 1))
        row = 0 if edge == "top" else height - 1
        seeds = {(row, c) for c in range(start, start + span)}
    else:
        height = int(rng.integers(2, 7))
        width = int(rng.integers(2, 6))
        span = int(rng.integers(1, height + 1))
        start = int(rng.integers(0, height - span + 1))
        col = 0 if edge == "left" else width - 1
        seeds = {(r, col) for r in range(start, start + span)}
    extra = int(rng.integers(0, max(1, (height * width) // 2) + 1))
    return _normalize(_random_grow(rng, height, width, len(seeds) + extra, seeds))


def _make_double_edge_shape(rng: np.random.Generator, edges: tuple[str, str], height: int, width: int) -> set[tuple[int, int]]:
    if set(edges) == {"top", "bottom"}:
        col = int(rng.integers(0, width))
        seeds = {(0, col), (height - 1, col)}
        for rr in range(1, height - 1):
            if rng.random() < 0.7:
                seeds.add((rr, col))
    elif set(edges) == {"left", "right"}:
        row = int(rng.integers(0, height))
        seeds = {(row, 0), (row, width - 1)}
        for cc in range(1, width - 1):
            if rng.random() < 0.7:
                seeds.add((row, cc))
    else:
        corner = {
            ("top", "left"): (0, 0),
            ("left", "top"): (0, 0),
            ("top", "right"): (0, width - 1),
            ("right", "top"): (0, width - 1),
            ("bottom", "left"): (height - 1, 0),
            ("left", "bottom"): (height - 1, 0),
            ("bottom", "right"): (height - 1, width - 1),
            ("right", "bottom"): (height - 1, width - 1),
        }[edges]
        rr, cc = corner
        seeds = {(rr, cc)}
        arm_v = int(rng.integers(1, height + 1))
        arm_h = int(rng.integers(1, width + 1))
        if "top" in edges:
            for r in range(arm_v):
                seeds.add((r, cc))
        if "bottom" in edges:
            for r in range(height - arm_v, height):
                seeds.add((r, cc))
        if "left" in edges:
            for c in range(arm_h):
                seeds.add((rr, c))
        if "right" in edges:
            for c in range(width - arm_h, width):
                seeds.add((rr, c))

    min_area = len(seeds)
    max_area = height * width
    target_area = int(rng.integers(min_area, max_area + 1))
    return _normalize(_random_grow(rng, height, width, target_area, seeds))


def _can_place(
    occupied: np.ndarray,
    cells: set[tuple[int, int]],
    top: int,
    left: int,
    pad: int = 1,
) -> bool:
    h, w = occupied.shape
    global_cells = [(top + r, left + c) for r, c in cells]
    for r, c in global_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in global_cells:
        r0 = max(0, r - pad)
        r1 = min(h - 1, r + pad)
        c0 = max(0, c - pad)
        c1 = min(w - 1, c + pad)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, cells: set[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in cells:
        grid[top + r, left + c] = color
        occupied[top + r, left + c] = True


def _sample_double_edges(rng: np.random.Generator) -> tuple[str, str]:
    choices = [
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
        ("top", "bottom"),
        ("left", "right"),
    ]
    return choices[int(rng.integers(0, len(choices)))]


def _place_object(
    rng: np.random.Generator,
    grid: np.ndarray,
    occupied: np.ndarray,
    color: int,
    kind: int,
) -> bool:
    h, w = grid.shape
    for _ in range(120):
        if kind == 0:
            cells = _make_interior_shape(rng)
            ph = 1 + max(r for r, _ in cells)
            pw = 1 + max(c for _, c in cells)
            if h - ph - 1 <= 0 or w - pw - 1 <= 0:
                continue
            top = int(rng.integers(1, h - ph))
            left = int(rng.integers(1, w - pw))
        elif kind == 1:
            edge = ["top", "bottom", "left", "right"][int(rng.integers(0, 4))]
            cells = _make_single_edge_shape(rng, edge)
            ph = 1 + max(r for r, _ in cells)
            pw = 1 + max(c for _, c in cells)
            if edge == "top":
                top = 0
                if w - pw - 1 <= 0:
                    continue
                left = int(rng.integers(1, w - pw))
            elif edge == "bottom":
                top = h - ph
                if top <= 0 or w - pw - 1 <= 0:
                    continue
                left = int(rng.integers(1, w - pw))
            elif edge == "left":
                left = 0
                if h - ph - 1 <= 0:
                    continue
                top = int(rng.integers(1, h - ph))
            else:
                left = w - pw
                if left <= 0 or h - ph - 1 <= 0:
                    continue
                top = int(rng.integers(1, h - ph))
        else:
            edges = _sample_double_edges(rng)
            if set(edges) == {"top", "bottom"}:
                height = h
                width = int(rng.integers(2, max(3, w - 1)))
                cells = _make_double_edge_shape(rng, edges, height, width)
                ph = h
                pw = 1 + max(c for _, c in cells)
                if w - pw - 1 <= 0:
                    continue
                top = 0
                left = int(rng.integers(1, w - pw))
            elif set(edges) == {"left", "right"}:
                height = int(rng.integers(2, max(3, h - 1)))
                width = w
                cells = _make_double_edge_shape(rng, edges, height, width)
                ph = 1 + max(r for r, _ in cells)
                pw = w
                if h - ph - 1 <= 0:
                    continue
                top = int(rng.integers(1, h - ph))
                left = 0
            else:
                height = int(rng.integers(2, min(7, h)))
                width = int(rng.integers(2, min(7, w)))
                cells = _make_double_edge_shape(rng, edges, height, width)
                ph = 1 + max(r for r, _ in cells)
                pw = 1 + max(c for _, c in cells)
                top = 0 if "top" in edges else h - ph
                left = 0 if "left" in edges else w - pw
                if ("bottom" in edges and top < 0) or ("right" in edges and left < 0):
                    continue

        if _can_place(occupied, cells, top, left):
            _stamp(grid, occupied, cells, top, left, color)
            return True
    return False


def _analyze_branch_counts(grid: np.ndarray) -> list[int]:
    counts = [0, 0, 0]
    for comp in _find_components(grid, BG):
        edges = len(comp["touches"])
        if edges in (0, 1, 2):
            counts[edges] += 1
    return counts


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(10, 19))
    w = int(rng.integers(10, 19))
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    base_kinds = [0, 1, 2]
    extra_count = int(rng.integers(0, 3))
    extras = [int(rng.integers(0, 3)) for _ in range(extra_count)]
    kinds = base_kinds + extras
    rng.shuffle(kinds)

    palette = [c for c in range(10) if c != BG]
    rng.shuffle(palette)
    if len(kinds) > len(palette):
        return None

    for color, kind in zip(palette[: len(kinds)], kinds, strict=False):
        if not _place_object(rng, grid, occupied, int(color), int(kind)):
            return None

    branch_counts = _analyze_branch_counts(grid)
    if min(branch_counts) < 1:
        return None
    if sum(branch_counts) != len(kinds):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
