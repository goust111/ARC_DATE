from __future__ import annotations

import random
from collections import deque
from typing import Iterable

import numpy as np


BG = 5
INPUT_FG = 1
OPEN_TO_COLOR = {
    0: 2,  # red
    1: 3,  # green
    2: 4,  # yellow
    3: 7,  # orange
    4: 9,  # maroon
}
EDGES = ("T", "B", "L", "R")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _open_edge_count(mask: np.ndarray) -> int:
    open_edges = 0
    if not bool(mask[0, :].all()):
        open_edges += 1
    if not bool(mask[-1, :].all()):
        open_edges += 1
    if not bool(mask[:, 0].all()):
        open_edges += 1
    if not bool(mask[:, -1].all()):
        open_edges += 1
    return open_edges


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for comp in _components(g == INPUT_FG):
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        k = _open_edge_count(sub)
        out[comp[:, 0], comp[:, 1]] = OPEN_TO_COLOR[k]
    return out


def _edge_anchor(shape: tuple[int, int], edge: str) -> tuple[int, int]:
    h, w = shape
    if edge == "T":
        return 0, random.randint(1, w - 2)
    if edge == "B":
        return h - 1, random.randint(1, w - 2)
    if edge == "L":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _draw_line(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    r, c = a
    br, bc = b
    while r != br:
        mask[r, c] = True
        r += 1 if br > r else -1
    while c != bc:
        mask[r, c] = True
        c += 1 if bc > c else -1
    mask[r, c] = True


def _carve_gap(mask: np.ndarray, edge: str, keep: set[tuple[int, int]]) -> bool:
    h, w = mask.shape
    if edge in ("T", "B"):
        r = 0 if edge == "T" else h - 1
        candidates = [c for c in range(1, w - 1) if (r, c) not in keep]
        if not candidates:
            return False
        gap_len = random.randint(1, max(1, len(candidates) // 2))
        start = random.randint(0, len(candidates) - gap_len)
        for c in candidates[start : start + gap_len]:
            mask[r, c] = False
        return True
    c = 0 if edge == "L" else w - 1
    candidates = [r for r in range(1, h - 1) if (r, c) not in keep]
    if not candidates:
        return False
    gap_len = random.randint(1, max(1, len(candidates) // 2))
    start = random.randint(0, len(candidates) - gap_len)
    for r in candidates[start : start + gap_len]:
        mask[r, c] = False
    return True


def _trim(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_mask_for_k(k: int) -> np.ndarray:
    for _ in range(300):
        h = random.randint(4, 7)
        w = random.randint(4, 7)
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True

        center = (random.randint(1, h - 2), random.randint(1, w - 2))
        anchors = {edge: _edge_anchor((h, w), edge) for edge in EDGES}
        keep = set(anchors.values()) | {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
        for edge in EDGES:
            _draw_line(mask, anchors[edge], center)

        open_edges = set(random.sample(EDGES, k))
        ok = True
        for edge in open_edges:
            if not _carve_gap(mask, edge, keep):
                ok = False
                break
        if not ok:
            continue

        # Add a little interior variation without changing the bbox or the edge statuses.
        for _ in range(random.randint(0, 3)):
            rr = random.randint(1, h - 2)
            cc = random.randint(1, w - 2)
            mask[rr, cc] = True

        mask = _trim(mask)
        if _components(mask).__len__() != 1:
            continue
        if _open_edge_count(mask) != k:
            continue
        return mask
    raise RuntimeError(f"failed to build object with {k} open edges")


def _can_place(occupied: np.ndarray, patch: np.ndarray, top: int, left: int, margin: int = 1) -> bool:
    h, w = patch.shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(occupied.shape[0], top + h + margin)
    c1 = min(occupied.shape[1], left + w + margin)
    return not bool(occupied[r0:r1, c0:c1].any())


def _paste(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    region = grid[top : top + h, left : left + w]
    region[patch] = INPUT_FG
    occupied[top : top + h, left : left + w][patch] = True


def _categories_for_instance() -> list[int]:
    cats = [0, 1, 2, 3, 4]
    extras = random.randint(0, 3)
    for _ in range(extras):
        cats.append(random.randint(0, 4))
    random.shuffle(cats)
    return cats


def _try_generate() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    for k in _categories_for_instance():
        patch = _random_mask_for_k(k)
        ph, pw = patch.shape
        positions: list[tuple[int, int]] = []
        for top in range(1, h - ph):
            for left in range(1, w - pw):
                if _can_place(occupied, patch, top, left, margin=1):
                    positions.append((top, left))
        if not positions:
            return None
        top, left = random.choice(positions)
        _paste(grid, occupied, patch, top, left)

    out = main(grid)
    colors = set(map(int, np.unique(out))) - {BG}
    if not colors:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
