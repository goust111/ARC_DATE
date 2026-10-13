from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


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
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _pieces(grid: np.ndarray, bg: int) -> list[dict]:
    pieces: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        coords = np.argwhere(grid == color)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        pieces.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "mask": mask,
            }
        )
    return pieces


def _pair_indices(pieces: list[dict]) -> list[tuple[int, int]]:
    unused = set(range(len(pieces)))
    pairs: list[tuple[int, int]] = []
    while unused:
        i = min(unused)
        unused.remove(i)
        mask_i = pieces[i]["mask"]
        matches = []
        for j in unused:
            mask_j = pieces[j]["mask"]
            if mask_i.shape != mask_j.shape:
                continue
            if np.any(mask_i & mask_j):
                continue
            if np.all(mask_i | mask_j):
                matches.append(j)
        assert len(matches) == 1, "each piece must have one complementary same-bbox partner"
        j = matches[0]
        unused.remove(j)
        pairs.append((i, j))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-size complementary fragments and overlay each pair at the upper-left anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    pieces = _pieces(g, bg)
    assert len(pieces) % 2 == 0 and pieces, "expected an even nonzero number of pieces"

    out = g.copy()
    for i, j in _pair_indices(pieces):
        a = pieces[i]
        b = pieces[j]
        ar0, ac0, ar1, ac1 = a["bbox"]
        br0, bc0, br1, bc1 = b["bbox"]

        out[ar0 : ar1 + 1, ac0 : ac1 + 1][a["mask"]] = bg
        out[br0 : br1 + 1, bc0 : bc1 + 1][b["mask"]] = bg

        anchor = a if (ar0, ac0) <= (br0, bc0) else b
        other = b if anchor is a else a
        r0, c0, _r1, _c1 = anchor["bbox"]
        mh, mw = anchor["mask"].shape
        target = out[r0 : r0 + mh, c0 : c0 + mw]
        target[anchor["mask"]] = anchor["color"]
        target[other["mask"]] = other["color"]

    return out


def _connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _full_bbox(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    return (
        int(pts[:, 0].min()) == 0
        and int(pts[:, 0].max()) == h - 1
        and int(pts[:, 1].min()) == 0
        and int(pts[:, 1].max()) == w - 1
    )


def _make_pair_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(300):
        fill = random.uniform(0.38, 0.62)
        mask = np.random.random((h, w)) < fill
        other = ~mask
        if _full_bbox(mask) and _full_bbox(other):
            if random.random() < 0.5:
                return mask, other
            return other, mask
    raise RuntimeError("failed to make complementary masks")


def _place(grid: np.ndarray, mask: np.ndarray, color: int, occupied: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    mh, mw = mask.shape
    for _ in range(200):
        r0 = random.randint(1, h - mh - 1)
        c0 = random.randint(1, w - mw - 1)
        region = occupied[r0 - 1 : r0 + mh + 1, c0 - 1 : c0 + mw + 1]
        if np.any(region):
            continue
        grid[r0 : r0 + mh, c0 : c0 + mw][mask] = color
        occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
        return r0, c0
    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 4)
    h = random.randint(18, 28)
    w = random.randint(20, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(range(1, 10), pair_count * 2)
    used_dims: set[tuple[int, int]] = set()

    for pair_idx in range(pair_count):
        for _ in range(100):
            mh = random.randint(3, 6)
            mw = random.randint(4, 7)
            if (mh, mw) not in used_dims:
                used_dims.add((mh, mw))
                break
        else:
            return None

        m1, m2 = _make_pair_masks(mh, mw)
        c1 = colors[2 * pair_idx]
        c2 = colors[2 * pair_idx + 1]
        p1 = _place(grid, m1, c1, occupied)
        p2 = _place(grid, m2, c2, occupied)
        if p1 is None or p2 is None:
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate complementary-fragment input")
