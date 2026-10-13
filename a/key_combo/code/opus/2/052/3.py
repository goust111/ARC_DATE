from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np

BLACK = 0


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                x, y = q.pop()
                comp.append((int(x), int(y)))
                for nx, ny in _neighbors4(x, y):
                    if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] and not visited[nx, ny]:
                        visited[nx, ny] = True
                        q.append((nx, ny))
            out.append(np.array(comp, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[int(r - r0), int(c - c0)] = True
    return local, int(r0), int(c0)


def _compress(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.uint8)
    # compress consecutive identical rows
    rows: list[np.ndarray] = []
    for r in range(m.shape[0]):
        if not rows or not np.array_equal(m[r], rows[-1]):
            rows.append(m[r])
    m = np.stack(rows, axis=0)
    # compress consecutive identical columns
    cols: list[np.ndarray] = []
    for c in range(m.shape[1]):
        col = m[:, c]
        if not cols or not np.array_equal(col, cols[-1]):
            cols.append(col)
    m2 = np.stack(cols, axis=1)
    return m2.astype(bool)


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    a = mask.astype(np.uint8)
    for k in range(4):
        r = np.rot90(a, k)
        out.append(r)
        out.append(np.fliplr(r))
    # deduplicate identical arrays
    uniq: list[np.ndarray] = []
    for t in out:
        if not any(np.array_equal(t, u) for u in uniq):
            uniq.append(t)
    return uniq


def _match_shape(a: np.ndarray, b: np.ndarray) -> bool:
    for t in _transforms(a):
        if t.shape == b.shape and np.array_equal(t, b):
            return True
    return False


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    mask = out != BLACK
    comps = _components(mask)
    objs = []
    for idx, coords in enumerate(comps):
        local, r0, c0 = _crop_mask(out != BLACK, coords)
        comp_mask = local
        comp_color = int(out[coords[0, 0], coords[0, 1]])
        if not all(out[coords[:, 0], coords[:, 1]] == comp_color):
            continue
        objs.append(
            {
                "idx": idx,
                "coords": coords,
                "origin": (r0, c0),
                "color": comp_color,
                "shape": _compress(comp_mask),
            }
        )

    if not objs:
        return out

    n = len(objs)
    dsu = _DSU(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _match_shape(objs[i]["shape"], objs[j]["shape"]):
                dsu.union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, obj in enumerate(objs):
        groups.setdefault(dsu.find(i), []).append(obj)

    for _, members in groups.items():
        if len(members) >= 2:
            count = Counter(m["color"] for m in members)
            majority = max(count.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            for m in members:
                rows = out[m["coords"][:, 0], m["coords"][:, 1]]
                out[m["coords"][:, 0], m["coords"][:, 1]] = int(majority)
        else:
            m = members[0]
            out[m["coords"][:, 0], m["coords"][:, 1]] = BLACK

    return out


def _make_mask(base: np.ndarray, sx: int, sy: int, rot: int, flip: bool) -> np.ndarray:
    m = np.kron(base.astype(np.uint8), np.ones((sx, sy), dtype=np.uint8))
    m = np.rot90(m, k=rot)
    if flip:
        m = np.fliplr(m)
    return m.astype(bool)


def _can_place(
    out: np.ndarray, occ: np.ndarray, top: int, left: int, mask: np.ndarray
) -> bool:
    h, w = out.shape
    mh, mw = mask.shape
    if top < 0 or left < 0 or top + mh > h or left + mw > w:
        return False
    slice_occ = occ[max(0, top - 1) : min(h, top + mh + 1), max(0, left - 1) : min(w, left + mw + 1)]
    if np.any(slice_occ):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = w = 20
    out = np.zeros((h, w), dtype=int)
    occ = np.zeros_like(out, dtype=bool)
    bases = [
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        np.array([[1, 1], [1, 1]], dtype=np.uint8),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=np.uint8),
    ]
    colors = [1, 2, 3, 4]
    created = 0
    groups = 0
    # generate at least one group with >=2 members
    for color in [1, 3]:
        reps = int(rng.integers(2, 4))
        base = bases[groups]
        for _ in range(reps):
            sx = int(rng.integers(1, 4))
            sy = int(rng.integers(1, 4))
            rot = int(rng.integers(0, 4))
            flip = bool(rng.random() < 0.5)
            m = _make_mask(base, sx, sy, rot, flip)
            mh, mw = m.shape
            placed = False
            for _attempt in range(200):
                top = int(rng.integers(1, h - mh - 1))
                left = int(rng.integers(1, w - mw - 1))
                if _can_place(out, occ, top, left, m):
                    rr, cc = np.where(m)
                    color = colors[groups] if color == 1 else colors[groups + 1]
                    out[top + rr, left + cc] = int(color)
                    occ[max(0, top - 1) : min(h, top + mh + 1), max(0, left - 1) : min(w, left + mw + 1)] = True
                    created += 1
                    placed = True
                    break
            if not placed:
                break
        groups += 1

    # unmatched objects
    for color in [2, 4]:
        base = bases[2]
        sx = int(rng.integers(1, 3))
        sy = int(rng.integers(1, 3))
        rot = int(rng.integers(0, 4))
        flip = bool(rng.random() < 0.5)
        m = _make_mask(base, sx, sy, rot, flip)
        mh, mw = m.shape
        for _ in range(2):
            placed = False
            for _attempt in range(200):
                top = int(rng.integers(1, h - mh - 1))
                left = int(rng.integers(1, w - mw - 1))
                if _can_place(out, occ, top, left, m):
                    rr, cc = np.where(m)
                    out[top + rr, left + cc] = int(color)
                    occ[max(0, top - 1) : min(h, top + mh + 1), max(0, left - 1) : min(w, left + mw + 1)] = True
                    created += 1
                    placed = True
                    break
            if not placed:
                break

    while created < 5:
        # fallback random fill to avoid empty outputs in rare cases
        r = int(rng.integers(1, h - 2))
        c = int(rng.integers(1, w - 2))
        if out[r, c] == 0:
            out[r, c] = int(rng.choice(colors))
            created += 1

    return out

