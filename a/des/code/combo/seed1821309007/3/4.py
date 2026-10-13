import random
from collections import deque

import numpy as np


AZURE = 8
MAROON = 9
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

SHAPES = [
    [(-1, 0), (0, 0), (1, 0)],
    [(0, -1), (0, 0), (0, 1)],
    [(0, 0), (-1, 0), (0, -1), (0, 1)],
    [(0, 0), (-1, 0), (0, -1), (1, 0)],
    [(0, 0), (-1, 0), (0, 1), (1, 0)],
    [(0, 0), (0, -1), (0, 1), (1, 0)],
    [(0, 0), (-1, 0), (0, -1), (0, 1), (1, 0)],
]


def _components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == AZURE or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            bbox = (min(rows), min(cols), max(rows), max(cols))
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            objects.append({"color": color, "cells": cells, "bbox": bbox, "center": center})
    return objects


def _nearest_indices(objects: list[dict]) -> list[int]:
    nearest = []
    for i, obj in enumerate(objects):
        cr, cc = obj["center"]
        dists = []
        for j, other in enumerate(objects):
            if i == j:
                continue
            orr, occ = other["center"]
            dists.append((abs(cr - orr) + abs(cc - occ), j))
        dists.sort()
        assert len(dists) >= 1
        assert len(dists) == 1 or dists[0][0] != dists[1][0], "nearest neighbor must be unique"
        nearest.append(dists[0][1])
    return nearest


def _path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    cells = []
    step_c = 1 if bc >= ac else -1
    for c in range(ac, bc + step_c, step_c):
        cells.append((ar, c))
    step_r = 1 if br >= ar else -1
    for r in range(ar + step_r, br + step_r, step_r):
        cells.append((r, bc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each object to its nearest neighbor's color unless it is in a mutual-nearest pair,
    and draw maroon orthogonal connectors between mutual-nearest pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == AZURE), "expected azure background"

    objects = _components(g)
    assert len(objects) >= 3, "expected at least three foreground objects"

    nearest = _nearest_indices(objects)
    out = g.copy()

    for i, j in enumerate(nearest):
        if nearest[j] != i:
            new_color = int(objects[j]["color"])
            for r, c in objects[i]["cells"]:
                out[r, c] = new_color

    for i, j in enumerate(nearest):
        if i < j and nearest[j] == i:
            for r, c in _path_cells(objects[i]["center"], objects[j]["center"]):
                if int(g[r, c]) == AZURE:
                    out[r, c] = MAROON

    return out


def _shape_cells(center: tuple[int, int], shape: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in shape}


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _make_spec(centers: list[tuple[int, int]]) -> list[dict]:
    colors = random.sample(PALETTE, len(centers))
    specs = []
    for center, color in zip(centers, colors):
        shape = random.choice(SHAPES)
        cells = _shape_cells(center, shape)
        bbox = _bbox_from_cells(cells)
        specs.append({"center": center, "color": color, "cells": cells, "bbox": bbox})
    return specs


def _valid_specs(h: int, w: int, specs: list[dict], desired: list[int]) -> bool:
    for spec in specs:
        for r, c in spec["cells"]:
            if not (0 <= r < h and 0 <= c < w):
                return False

    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if not _bboxes_separated(specs[i]["bbox"], specs[j]["bbox"], gap=1):
                return False

    objects = []
    for spec in specs:
        bbox = spec["bbox"]
        center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        objects.append({"color": spec["color"], "cells": list(spec["cells"]), "bbox": bbox, "center": center})

    try:
        nearest = _nearest_indices(objects)
    except AssertionError:
        return False
    if nearest != desired:
        return False

    occupied = [set(spec["cells"]) for spec in specs]
    for i, j in enumerate(nearest):
        if i < j and nearest[j] == i:
            path = _path_cells(objects[i]["center"], objects[j]["center"])
            blockers = set()
            for k, occ in enumerate(occupied):
                if k not in (i, j):
                    blockers |= occ
            if any(cell in blockers for cell in path[1:-1]):
                return False
    return True


def _sample_centers(h: int, w: int, count: int, template: str) -> list[tuple[int, int]] | None:
    for _ in range(200):
        if template == "chain3":
            vertical = random.choice([False, True])
            d23 = random.randint(4, 6)
            d12 = d23 + random.randint(2, 4)
            if vertical:
                if h - 4 - d12 - d23 < 3:
                    continue
                c = random.randint(3, w - 4)
                r1 = random.randint(3, h - 4 - d12 - d23)
                centers = [(r1, c), (r1 + d12, c), (r1 + d12 + d23, c)]
            else:
                if w - 4 - d12 - d23 < 3:
                    continue
                r = random.randint(3, h - 4)
                c1 = random.randint(3, w - 4 - d12 - d23)
                centers = [(r, c1), (r, c1 + d12), (r, c1 + d12 + d23)]
        else:
            vertical_chain = random.choice([False, True])
            d23 = random.randint(4, 6)
            d12 = d23 + random.randint(2, 4)
            dp = random.randint(4, 6)
            if vertical_chain:
                if h - 4 - d12 - d23 < 3 or w < 16:
                    continue
                c_chain = random.randint(3, w // 2 - 1)
                r1 = random.randint(3, h - 4 - d12 - d23)
                chain = [(r1, c_chain), (r1 + d12, c_chain), (r1 + d12 + d23, c_chain)]
                left = max(c_chain + dp + 4, w // 2 + 1)
                right = w - 4
                if left > right or h - 4 - dp < 3:
                    continue
                c_pair = random.randint(left, right)
                r_pair = random.randint(3, h - 4 - dp)
                pair_vertical = random.choice([False, True])
                pair = [(r_pair, c_pair), (r_pair + dp, c_pair)] if pair_vertical else [(r_pair, c_pair), (r_pair, c_pair + dp)]
            else:
                if w - 4 - d12 - d23 < 3 or h < 16:
                    continue
                r_chain = random.randint(3, h // 2 - 1)
                c1 = random.randint(3, w - 4 - d12 - d23)
                chain = [(r_chain, c1), (r_chain, c1 + d12), (r_chain, c1 + d12 + d23)]
                top = max(r_chain + dp + 4, h // 2 + 1)
                bottom = h - 4
                if top > bottom or w - 4 - dp < 3:
                    continue
                r_pair = random.randint(top, bottom)
                c_pair = random.randint(3, w - 4 - dp)
                pair_vertical = random.choice([False, True])
                pair = [(r_pair, c_pair), (r_pair + dp, c_pair)] if pair_vertical else [(r_pair, c_pair), (r_pair, c_pair + dp)]
            centers = chain + pair

        ok = True
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                if max(abs(centers[i][0] - centers[j][0]), abs(centers[i][1] - centers[j][1])) < 4:
                    ok = False
        if ok and len(set(centers)) == count:
            return centers
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        h = random.randint(16, 22)
        w = random.randint(16, 22)
        use_extra_pair = random.choice([False, True])
        template = "chain5" if use_extra_pair else "chain3"
        centers = _sample_centers(h, w, 5 if use_extra_pair else 3, template)
        if centers is None:
            continue
        desired = [1, 2, 1] if not use_extra_pair else [1, 2, 1, 4, 3]

        for _ in range(80):
            specs = _make_spec(centers)
            if not _valid_specs(h, w, specs, desired):
                continue

            grid = np.full((h, w), AZURE, dtype=int)
            for spec in specs:
                for r, c in spec["cells"]:
                    grid[r, c] = spec["color"]
            return grid

    raise RuntimeError("failed to generate a valid input")
