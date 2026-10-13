from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _count_runs(values: list[bool]) -> int:
    runs = 0
    inside = False
    for v in values:
        if v and not inside:
            runs += 1
        inside = v
    return runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the missing border side of each rectangular frame to choose stripe orientation and stripe count.

    Input structure:
    - The background is the dominant color.
    - Each non-background object is a one-pixel-thick rectangular border with one or more gaps, all on the same side.
    - The frame interiors are background-colored and the frames do not overlap.

    Rule:
    - For each colored frame, find which border side contains the missing runs and count how many distinct gaps appear there.
    - If the gaps are on the top or bottom border, draw that many horizontal stripes inside the frame.
    - If the gaps are on the left or right border, draw that many vertical stripes inside the frame.
    - Place the stripes at evenly spaced interior layer positions and keep the original border and its gaps unchanged.

    Key insight:
    - The missing border encodes two independent signals at once: the affected side chooses the stripe direction, and the number of separate missing runs chooses the stripe count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        all_cells = [cell for comp in _components(g == color) for cell in comp]
        r0, c0, r1, c1 = _bbox(all_cells)
        cells = set(all_cells)
        top_missing = [((r0, c) not in cells) for c in range(c0, c1 + 1)]
        bottom_missing = [((r1, c) not in cells) for c in range(c0, c1 + 1)]
        left_missing = [((r, c0) not in cells) for r in range(r0, r1 + 1)]
        right_missing = [((r, c1) not in cells) for r in range(r0, r1 + 1)]
        gap_counts = {
            "top": _count_runs(top_missing),
            "bottom": _count_runs(bottom_missing),
            "left": _count_runs(left_missing),
            "right": _count_runs(right_missing),
        }
        side, count = max(gap_counts.items(), key=lambda item: item[1])
        assert count >= 1

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if side in ("top", "bottom"):
            for i in range(1, count + 1):
                offset = int(round(i * (inner_h + 1) / (count + 1)))
                offset = max(1, min(inner_h, offset))
                out[r0 + offset, c0 + 1 : c1] = color
        else:
            for i in range(1, count + 1):
                offset = int(round(i * (inner_w + 1) / (count + 1)))
                offset = max(1, min(inner_w, offset))
                out[r0 + 1 : r1, c0 + offset] = color

    return out


def _draw_broken_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, side: str, gaps: list[tuple[int, int]]) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if side == "top":
        for a, b in gaps:
            grid[r0, a : b + 1] = BG
    elif side == "bottom":
        for a, b in gaps:
            grid[r1, a : b + 1] = BG
    elif side == "left":
        for a, b in gaps:
            grid[a : b + 1, c0] = BG
    else:
        for a, b in gaps:
            grid[a : b + 1, c1] = BG


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BG, dtype=int)
        colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(2, 4))
        ok = True

        for color in colors:
            placed = False
            for _ in range(100):
                fh = random.randint(5, 9)
                fw = random.randint(5, 10)
                r0 = random.randint(1, h - fh - 1)
                c0 = random.randint(1, w - fw - 1)
                if np.any(grid[max(0, r0 - 1) : min(h, r0 + fh + 1), max(0, c0 - 1) : min(w, c0 + fw + 1)] != BG):
                    continue
                side = random.choice(["top", "bottom", "left", "right"])
                gap_count = random.randint(1, 3)
                gaps: list[tuple[int, int]] = []
                if side in ("top", "bottom"):
                    positions = list(range(c0 + 1, c0 + fw - 1))
                else:
                    positions = list(range(r0 + 1, r0 + fh - 1))
                random.shuffle(positions)
                positions.sort()
                cursor = 0
                while len(gaps) < gap_count and cursor < len(positions):
                    start = positions[cursor]
                    length = random.randint(1, 2)
                    end = min(start + length - 1, positions[-1])
                    if gaps and start <= gaps[-1][1] + 1:
                        cursor += 1
                        continue
                    gaps.append((start, end))
                    cursor += random.randint(1, 3)
                if len(gaps) != gap_count:
                    continue
                _draw_broken_frame(grid, r0, c0, fh, fw, color, side, gaps)
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid
    raise RuntimeError("failed to generate a valid instance")
