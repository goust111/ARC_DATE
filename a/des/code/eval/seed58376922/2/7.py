import random
from typing import Iterable

import numpy as np


GEN_RETRIES = 300
BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bg_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _segment_infos(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    infos: list[dict] = []
    path_mask = np.zeros_like(grid, dtype=bool)

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        markers = [comp for comp in comps if len(comp) == 1]
        segments = [comp for comp in comps if len(comp) > 1]
        assert len(markers) == 1, "each color must have exactly one singleton marker"
        assert len(segments) == 1, "each color must have exactly one path segment"

        marker = tuple(map(int, markers[0][0]))
        segment = [tuple(map(int, cell)) for cell in segments[0]]
        for r, c in segment:
            path_mask[r, c] = True

        rows = sorted({r for r, _ in segment})
        cols = sorted({c for _, c in segment})
        if len(rows) == 1:
            infos.append(
                {
                    "color": color,
                    "marker": marker,
                    "orientation": "h",
                    "line": rows[0],
                    "coords": set(cols),
                }
            )
        elif len(cols) == 1:
            infos.append(
                {
                    "color": color,
                    "marker": marker,
                    "orientation": "v",
                    "line": cols[0],
                    "coords": set(rows),
                }
            )
        else:
            raise AssertionError("each colored path segment must be straight")

    return infos, path_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each interior marker onto the matching-colored path segment and fill
    the rectangle between them, leaving the loop pixels unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    infos, path_mask = _segment_infos(g, bg)

    out = g.copy()
    infos.sort(key=lambda item: item["marker"])
    for info in infos:
        color = int(info["color"])
        mr, mc = info["marker"]
        if info["orientation"] == "h":
            assert mc in info["coords"], "horizontal segment must share a column with its marker"
            pr, pc = int(info["line"]), int(mc)
        else:
            assert mr in info["coords"], "vertical segment must share a row with its marker"
            pr, pc = int(mr), int(info["line"])

        r0, r1 = sorted((mr, pr))
        c0, c1 = sorted((mc, pc))
        block = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = ~path_mask[r0 : r1 + 1, c0 : c1 + 1]
        block[mask] = color
        out[r0 : r1 + 1, c0 : c1 + 1] = block

    return out


def _edge_cells(a: tuple[int, int], b: tuple[int, int]) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc > ac else -1
        cells = [(ar, c) for c in range(ac + step, bc + step, step)]
        return cells, (0, step)
    if ac == bc:
        step = 1 if br > ar else -1
        cells = [(r, ac) for r in range(ar + step, br + step, step)]
        return cells, (step, 0)
    raise AssertionError("polygon edges must be orthogonal")


def _build_segments(vertices: list[tuple[int, int]]) -> list[dict]:
    segments: list[dict] = []
    n = len(vertices)
    for i in range(n):
        cells, direction = _edge_cells(vertices[i], vertices[(i + 1) % n])
        assert len(cells) >= 2, "every segment must have length at least 2"
        dr, dc = direction
        segments.append({"cells": cells, "inward": (dc, -dr)})
    return segments


def _candidate_positions(grid: np.ndarray, segment: dict) -> list[dict]:
    h, w = grid.shape
    cells = segment["cells"]
    if len(cells) > 3:
        cells = cells[1:-1]

    dr, dc = segment["inward"]
    cands: list[dict] = []
    for pr, pc in cells:
        ray: list[tuple[int, int]] = []
        r, c = pr + dr, pc + dc
        while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == BG:
            ray.append((r, c))
            r += dr
            c += dc
        if not ray:
            continue
        for depth, marker in enumerate(ray, start=1):
            cands.append({"marker": marker, "projection": (pr, pc), "depth": depth})
    return cands


def _choose_markers(grid: np.ndarray, segments: list[dict], rng: random.Random) -> list[tuple[int, int]]:
    pools = [_candidate_positions(grid, segment) for segment in segments]
    if any(not pool for pool in pools):
        raise ValueError("every segment needs at least one valid marker position")

    order = sorted(range(len(segments)), key=lambda idx: len(pools[idx]))
    placed: dict[int, tuple[int, int]] = {}
    used: set[tuple[int, int]] = set()

    for idx in order:
        pool = [cand for cand in pools[idx] if cand["marker"] not in used]
        if not pool:
            raise ValueError("marker positions must be unique")
        deep = [cand for cand in pool if cand["depth"] >= 2]
        choice = rng.choice(deep or pool)
        marker = tuple(map(int, choice["marker"]))
        placed[idx] = marker
        used.add(marker)

    return [placed[i] for i in range(len(segments))]


def _make_rectangle_vertices(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    outer_h = rng.randint(8, h - 3)
    outer_w = rng.randint(8, w - 3)
    top = rng.randint(1, h - outer_h - 1)
    left = rng.randint(1, w - outer_w - 1)
    bottom = top + outer_h - 1
    right = left + outer_w - 1
    return [(top, left), (top, right), (bottom, right), (bottom, left)]


def _make_notch_vertices(h: int, w: int, rng: random.Random) -> list[tuple[int, int]]:
    outer_h = rng.randint(11, h - 3)
    outer_w = rng.randint(11, w - 3)
    top = rng.randint(1, h - outer_h - 1)
    left = rng.randint(1, w - outer_w - 1)
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    notch_top = rng.randint(top + 3, bottom - 3)
    x1 = rng.randint(left + 3, right - 5)
    x2 = rng.randint(x1 + 2, right - 3)
    return [
        (top, left),
        (top, right),
        (bottom, right),
        (bottom, x2),
        (notch_top, x2),
        (notch_top, x1),
        (bottom, x1),
        (bottom, left),
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 22)
    w = rng.randint(14, 22)
    family = rng.choice(["rectangle", "notch"])
    if family == "rectangle":
        vertices = _make_rectangle_vertices(h, w, rng)
    else:
        if h < 14 or w < 14:
            return None
        vertices = _make_notch_vertices(h, w, rng)

    segments = _build_segments(vertices)
    colors = rng.sample(PALETTE, len(segments))
    grid = np.zeros((h, w), dtype=int)

    for color, segment in zip(colors, segments):
        for r, c in segment["cells"]:
            grid[r, c] = color

    try:
        markers = _choose_markers(grid, segments, rng)
    except ValueError:
        return None

    for color, marker in zip(colors, markers):
        mr, mc = marker
        if grid[mr, mc] != BG:
            return None
        grid[mr, mc] = color

    k = rng.randrange(4)
    if k:
        grid = np.rot90(grid, k).copy()

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
