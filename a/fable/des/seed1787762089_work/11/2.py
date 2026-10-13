from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
ORANGE = 7
MARKER_COLORS = [1, 2, 3, 4, 6]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray, *, conn8: bool = False) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    dirs = _neighbors4(0, 0) if not conn8 else [
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc
    ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    return int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())


def _find_containers(g: np.ndarray) -> list[dict[str, int]]:
    boxes = []
    for pts in _components(g == ORANGE):
        r0, c0, r1, c1 = _bbox(pts)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        expected = np.zeros((h, w), dtype=bool)
        expected[:, 0] = True
        expected[:, -1] = True
        expected[-1, :] = True
        if np.array_equal(g[r0 : r1 + 1, c0 : c1 + 1] == ORANGE, expected):
            boxes.append({"r0": r0, "c0": c0, "r1": r1, "c1": c1})
    assert 2 <= len(boxes) <= 3
    return boxes


def _top_markers(g: np.ndarray) -> list[tuple[int, int]]:
    markers = [(int(c), int(g[0, c])) for c in range(g.shape[1]) if int(g[0, c]) not in (BLACK, ORANGE)]
    assert 2 <= len(markers) <= 3
    assert len({c for _col, c in markers}) == len(markers)
    return markers


def _supply_counts(g: np.ndarray, marker_cols: set[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for color in marker_cols:
        pts = np.argwhere(g == color)
        counts[color] = int(sum(1 for r, _c in pts if int(r) != 0))
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _find_containers(g)
    markers = _top_markers(g)
    colors = {color for _col, color in markers}
    supplies = _supply_counts(g, colors)

    for col, color in markers:
        out[0, col] = BLACK
    for color in colors:
        out[(g == color) & (np.indices(g.shape)[0] != 0)] = BLACK

    for col, color in markers:
        hits = [box for box in containers if box["c0"] < col < box["c1"]]
        if len(hits) != 1:
            continue
        box = hits[0]
        width = box["c1"] - box["c0"] - 1
        height = box["r1"] - box["r0"]
        amount = supplies[color]
        assert amount <= width * height
        for k in range(amount):
            rr = box["r1"] - 1 - k // width
            cc = box["c0"] + 1 + k % width
            out[rr, cc] = color
    return out


def _can_place(occ: np.ndarray, cells: set[tuple[int, int]]) -> bool:
    h, w = occ.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w) or occ[r, c]:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                return False
    return True


def _place_blob(g: np.ndarray, occ: np.ndarray, color: int, amount: int) -> bool:
    h, w = g.shape
    for _ in range(120):
        r = random.randint(3, h - 8)
        c = random.randint(2, w - 4)
        cells = {(r, c)}
        while len(cells) < amount:
            frontier = set()
            for rr, cc in cells:
                for nr, nc in _neighbors4(rr, cc):
                    if 2 <= nr < h - 2 and 1 <= nc < w - 1 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) == amount and _can_place(occ, cells):
            for rr, cc in cells:
                g[rr, cc] = color
                occ[rr, cc] = True
            return True
    return False


def _draw_container(g: np.ndarray, occ: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int) -> dict[str, int]:
    r1, c1 = r0 + inner_h, c0 + inner_w + 1
    cells = {(r, c0) for r in range(r0, r1 + 1)}
    cells |= {(r, c1) for r in range(r0, r1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    if not _can_place(occ, cells):
        raise ValueError
    for r, c in cells:
        g[r, c] = ORANGE
        occ[r, c] = True
    return {"r0": r0, "c0": c0, "r1": r1, "c1": c1}


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(18, 24), random.randint(18, 24)
    g = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    n_cont = random.randint(2, 3)
    starts = sorted(random.sample(range(2, w - 7), n_cont))
    boxes = []
    try:
        for i, c0 in enumerate(starts):
            inner_w = random.randint(2, min(5, w - c0 - 3))
            inner_h = random.randint(3, 6)
            r0 = h - inner_h - 3 - random.randint(0, 2)
            boxes.append(_draw_container(g, occ, r0, c0, inner_h, inner_w))
    except ValueError:
        return None

    n_mark = random.randint(2, 3)
    hit_boxes = random.sample(boxes, random.randint(1, min(n_mark, len(boxes))))
    colors = random.sample(MARKER_COLORS, n_mark)
    marker_cols: list[int] = []
    for box in hit_boxes:
        marker_cols.append(random.randint(box["c0"] + 1, box["c1"] - 1))
    while len(marker_cols) < n_mark:
        c = random.randrange(1, w - 1)
        if all(not (box["c0"] < c < box["c1"]) for box in boxes):
            marker_cols.append(c)
    random.shuffle(marker_cols)
    for c, color in zip(marker_cols, colors):
        g[0, c] = color
        occ[0, c] = True

    for color in colors:
        hit = next((box for c, col in zip(marker_cols, colors) if col == color for box in boxes if box["c0"] < c < box["c1"]), None)
        cap = (hit["r1"] - hit["r0"]) * (hit["c1"] - hit["c0"] - 1) if hit else 9
        amount = random.randint(3, min(9, cap))
        if not _place_blob(g, occ, color, amount):
            return None

    try:
        main(g)
    except Exception:
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input")
