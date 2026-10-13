from __future__ import annotations

from collections import deque
import random
from typing import Iterable

import numpy as np

Color = 0


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
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
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs, cs = zip(*coords)
    return min(rs), min(cs), max(rs), max(cs)


def _perimeter_from_bbox(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    out: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        out.add((r0, c))
        out.add((r1, c))
    for r in range(r0 + 1, r1):
        out.add((r, c0))
        out.add((r, c1))
    return out


def _is_rect_ring(mask_set: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    for r, c in mask_set:
        r0, c0, r1, c1 = bbox
        if not (r == r0 or r == r1 or c == c0 or c == c1):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True).astype(int)
    h, w = out.shape
    wall_mask = out != Color
    rings: list[dict[str, object]] = []

    for comp in _components(wall_mask):
        r0, c0, r1, c1 = _bbox(comp)
        mask_set = set(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = _perimeter_from_bbox((r0, c0, r1, c1))
        if not _is_rect_ring(mask_set, (r0, c0, r1, c1)):
            continue
        if len(mask_set) < len(perim) - 2:
            continue
        missing = [p for p in perim if p not in mask_set]
        if len(missing) > 2:
            continue
        color = int(out[r0, c0])
        area = len(comp)
        rings.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "area": area,
                "color": color,
                "breached": len(missing) > 0,
            }
        )

    if not rings:
        return out

    rings.sort(key=lambda d: (d["area"], d["bbox"][2] - d["bbox"][0], d["bbox"][3] - d["bbox"][1]), reverse=True)
    breached = [d for d in rings if d["breached"]]
    n = len(breached)
    if n == 0:
        return out

    breached_colors = [int(d["color"]) for d in breached]
    if n % 2 == 1:
        new_colors = breached_colors[-1:] + breached_colors[:-1]
    else:
        new_colors = breached_colors[1:] + breached_colors[:1]

    for ring, color in zip(breached, new_colors):
        for r, c in ring["coords"]:
            out[r, c] = color

    return out


def _draw_perimeter(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
) -> None:
    for c in range(c0, c1 + 1):
        out[r0, c] = color
        out[r1, c] = color
    for r in range(r0 + 1, r1):
        out[r, c0] = color
        out[r, c1] = color


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    size = int(rng.integers(13, 18))
    side = size
    out = np.full((side, side), Color, dtype=int)
    max_rings = max(4, (min(side, side) - 2) // 2 - 1)
    n = int(rng.integers(4, min(max_rings, 6) + 1))
    colors = [1, 2, 3, 4, 5, 6]
    colors = rng.choice(colors, size=n, replace=True).tolist()
    unbroken = int(rng.integers(0, n))
    margin_step = 2

    for i in range(n):
        t = i * margin_step + 1
        r0 = t
        c0 = t
        r1 = side - 1 - t
        c1 = side - 1 - t
        if r1 - r0 < 2 or c1 - c0 < 2:
            break

        _draw_perimeter(out, r0, c0, r1, c1, int(colors[i % len(colors)]))
        if i != unbroken and rng.random() < 0.95:
            side_len = [r0, r1, c0, c1]
            side_choice = int(rng.integers(0, 4))
            if side_choice == 0:
                rr = r0
                cc = int(rng.integers(c0 + 1, c1))
                if cc == c0 or cc == c1:
                    cc = c0 + 1
            elif side_choice == 1:
                rr = r1
                cc = int(rng.integers(c0 + 1, c1))
                if cc == c0 or cc == c1:
                    cc = c0 + 1
            elif side_choice == 2:
                cc = c0
                rr = int(rng.integers(r0 + 1, r1))
                if rr == r0 or rr == r1:
                    rr = r0 + 1
            else:
                cc = c1
                rr = int(rng.integers(r0 + 1, r1))
                if rr == r0 or rr == r1:
                    rr = r0 + 1
            out[rr, cc] = Color

    return out

