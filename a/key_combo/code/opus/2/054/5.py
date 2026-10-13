"""Puzzle 5: parity gridline cuts connected components, then local interior recolor."""

from __future__ import annotations

from collections import deque

import numpy as np

BG = 0
FG = 5
MARKER_COLORS = {1, 2, 3}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    sr, sc = start
    q = deque([(sr, sc)])
    seen = {(sr, sc)}
    out: list[tuple[int, int]] = []

    while q:
        r, c = q.popleft()
        out.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds((h, w), nr, nc):
                continue
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    return out


def _all_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in zip(*np.where(mask)):
        if seen[sr, sc]:
            continue
        comp = _component(mask, (int(sr), int(sc)))
        for r, c in comp:
            seen[r, c] = True
        comps.append(comp)

    return comps


def _odd_neighbors8(r: int, c: int):
    for dr in (-2, 0, 2):
        for dc in (-2, 0, 2):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def main(grid: np.ndarray) -> np.ndarray:
    grid = np.array(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    comp_mask = grid == FG
    comps = _all_components(comp_mask)

    for comp in comps:
        marker_positions = [(r, c) for r, c in comp if grid[r, c] != FG]
        if len(marker_positions) != 1:
            continue

        marker_r, marker_c = marker_positions[0]
        marker_color = int(grid[marker_r, marker_c])
        if marker_color not in MARKER_COLORS:
            continue

        odd_cells = [(r, c) for r, c in comp if r % 2 == 1 and c % 2 == 1 and grid[r, c] == FG]
        if not odd_cells:
            continue

        odd_mask = np.zeros((h, w), dtype=bool)
        for r, c in odd_cells:
            odd_mask[r, c] = True

        subcomps = []
        seen = np.zeros((h, w), dtype=bool)
        for sr, sc in odd_cells:
            if seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cset: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cset.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if not _in_bounds((h, w), nr, nc):
                        continue
                    if not odd_mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            subcomps.append(cset)

        index = {p: idx for idx, comp_o in enumerate(subcomps) for p in comp_o}

        for part in subcomps:
            part_set = set(part)
            for r, c in part:
                ok = True
                if grid[r, c] != FG:
                    continue
                for nr, nc in _odd_neighbors8(r, c):
                    if not _in_bounds((h, w), nr, nc):
                        ok = False
                        break
                    if (nr, nc) not in part_set:
                        ok = False
                        break
                if ok:
                    # keep marker pixel intact if ever selected
                    if (r, c) == (marker_r, marker_c):
                        continue
                    out[r, c] = marker_color

    return out


def _random_blob(
    rng: np.random.Generator,
    H: int,
    W: int,
    occupied: np.ndarray,
    min_area: int,
    max_area: int,
) -> set[tuple[int, int]]:
    for _ in range(400):
        h0 = int(rng.integers(5, 8))
        w0 = int(rng.integers(5, 8))
        area = h0 * w0
        if area < min_area or area > max_area:
            continue

        top = int(rng.integers(1, H - h0 - 1))
        left = int(rng.integers(1, W - w0 - 1))

        cells = {(r, c) for r in range(top, top + h0) for c in range(left, left + w0)}
        if any(occupied[r, c] for r, c in cells):
            continue
        border = {
            (top - 1, left),
            (top - 1, left + w0),
            (top + h0, left),
            (top + h0, left + w0),
        }
        if any(not _in_bounds((H, W), r, c) for r, c in border):
            continue

        return cells

    return set()


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(200):
        H = 15
        W = 15
        grid = np.zeros((H, W), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        comp_num = int(rng.integers(2, 4))
        marker_colors = [1, 2, 3]
        rng.shuffle(marker_colors)
        created = 0

        for i in range(comp_num):
            for _ in range(200):
                cells = _random_blob(rng, H, W, occupied, 20, 40)
                if not cells:
                    continue

                # ensure marker inside and unique per component
                inner = [p for p in cells if 1 < p[0] < H - 2 and 1 < p[1] < W - 2]
                if not inner:
                    continue
                marker = inner[int(rng.integers(0, len(inner)))]

                for r, c in cells:
                    grid[r, c] = FG
                    occupied[r, c] = True

                mr, mc = marker
                grid[mr, mc] = marker_colors[i % len(marker_colors)]

                # small cavity so shape is not trivial
                carve = list(cells)
                rng.shuffle(carve)
                for rr, cc in carve[:2]:
                    nr, nc = rr, cc
                    if (nr, nc) != (mr, mc):
                        grid[nr, nc] = BG
                        occupied[nr, nc] = False
                created += 1
                break

        if created >= 2:
            return grid

    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 1, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 2, 5, 5, 5, 2, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )


if __name__ == "__main__":
    g = generate_input()
    print(main(g))
