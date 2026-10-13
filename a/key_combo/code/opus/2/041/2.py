from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import sys
from typing import Iterator

import numpy as np

if str(Path('/home/lixinyu/snap/new/solvers')) not in sys.path:
    sys.path.append(str(Path('/home/lixinyu/snap/new/solvers')))

from arc_utils import Color

BLACK = Color.BLACK
GRAY = Color.GREY

FRAME_COLORS = [Color.RED, Color.BLUE, Color.GREEN]


def _collect_coords(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    pts = np.argwhere(grid == int(color))
    return [(int(r), int(c)) for r, c in pts]


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _frame_candidates(grid: np.ndarray) -> dict[int, dict]:
    out: dict[int, dict] = {}
    h, w = grid.shape
    for color in FRAME_COLORS:
        pts = _collect_coords(grid, int(color))
        if len(pts) < 8:
            return {}
        r0, c0, r1, c1 = _bbox(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        perim = _frame_perimeter(r0, c0, r1, c1)
        # visible cells of this frame must lie on the bbox perimeter.
        if not all((r0 <= r <= r1 and c0 <= c <= c1 and (r in (r0, r1) or c in (c0, c1))) for r, c in pts):
            continue
        if not set(pts).issubset(perim):
            continue
        out[int(color)] = {
            "color": int(color),
            "bbox": (r0, c0, r1, c1),
            "perimeter": perim,
            "coords": set(pts),
        }
    return out


def _frame_perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _occlusion_order(grid: np.ndarray, frames: dict[int, dict]) -> list[int] | None:
    colors = sorted(frames.keys())
    idx = {c: i for i, c in enumerate(colors)}
    n = len(colors)
    if n != 3:
        return None

    g = grid
    edges: list[set[int]] = [set() for _ in range(n)]
    indeg = [0] * n
    for i, ca in enumerate(colors):
        for j, cb in enumerate(colors):
            if j <= i:
                continue
            pa = frames[ca]["perimeter"]
            pb = frames[cb]["perimeter"]
            cross = pa.intersection(pb)
            if not cross:
                continue
            cnt_a = 0
            cnt_b = 0
            for r, c in cross:
                v = int(g[r, c])
                if v == ca:
                    cnt_a += 1
                elif v == cb:
                    cnt_b += 1
            if cnt_a == 0 and cnt_b == 0:
                continue
            if cnt_b > cnt_a:
                edges[idx[ca]].add(idx[cb])
            elif cnt_a > cnt_b:
                edges[idx[cb]].add(idx[ca])

    for u in range(n):
        for v in edges[u]:
            indeg[v] += 1

    q: list[int] = [i for i, d in enumerate(indeg) if d == 0]
    order: list[int] = []
    while q:
        u = min(q)
        q.remove(u)
        order.append(u)
        for v in edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(order) != n:
        return None
    return [colors[i] for i in order]


def _components(grid: np.ndarray, color: int, min_area: int = 1) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    out: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: set[tuple[int, int]] = set()
            while stack:
                x, y = stack.pop()
                comp.add((x, y))
                for dr, dc in dirs:
                    nr, nc = x + dr, y + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == int(color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            if len(comp) >= min_area:
                out.append(comp)
    return out


def _find_sprite_components(grid: np.ndarray, frame_color: int, frame_bb: tuple[int, int, int, int]) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, frame_bb)
    comps = _components(grid, frame_color, min_area=3)
    sprites: list[set[tuple[int, int]]] = []
    for comp in comps:
        if len(comp) > 12:
            continue
        comp_r = [r for r, _ in comp]
        comp_c = [c for _, c in comp]
        if max(comp_r) < r0 or min(comp_r) > r1 or max(comp_c) < c0 or min(comp_c) > c1:
            sprites.append(comp)
    return sprites


def _nearest_side(comp: set[tuple[int, int]], frame_bb: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = map(int, frame_bb)
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    d_left = c0 - max(cs)
    d_right = min(cs) - c1
    d_top = r0 - max(rs)
    d_bottom = min(rs) - r1

    sides = [
        (d_left, "left"),
        (d_right, "right"),
        (d_top, "top"),
        (d_bottom, "bottom"),
    ]
    # choose the side the sprite is on and closest to frame
    valid = [(d, s) for d, s in sides if d >= 1]
    valid = valid if valid else sides
    _, side = sorted(valid, key=lambda t: (t[0], t[1]))[0]
    return side


def _mirror_sprite(out: np.ndarray, comp: set[tuple[int, int]], frame_bb: tuple[int, int, int, int], side: str) -> None:
    r0, c0, r1, c1 = map(int, frame_bb)
    for r, c in comp:
        if side == "left":
            nr, nc = r, 2 * c0 - c
        elif side == "right":
            nr, nc = r, 2 * c1 - c
        elif side == "top":
            nr, nc = 2 * r0 - r, c
        else:  # bottom
            nr, nc = 2 * r1 - r, c

        if not (0 <= nr < out.shape[0] and 0 <= nc < out.shape[1]):
            continue
        if not (r0 < nr < r1 and c0 < nc < c1):
            continue
        out[nr, nc] = int(GRAY)


def _iter_frame_cells(frame_bb: tuple[int, int, int, int]) -> Iterator[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame_bb)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                yield r, c


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each frame-linked sprite into its frame interior except the bottom frame.

    Input structure: three one-cell-thick overlapping single-color rectangular frames (red, blue, green)
    plus 4- to 6-cell monochrome sprites in frame color outside the frames.

    Rule: infer frame perimeters and occlusion order from crossing visibility, derive a unique bottom-to-top ordering,
    leave the bottom sprite unchanged, and for higher frames mirror the matching-color sprite across the nearest frame side
    that faces the sprite into the frame interior with gray color.

    Key insight: occlusion order controls mirror eligibility, so only sprites belonging to non-bottom layers are mirrored.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("main expects 2D ndarray")

    out = g.copy()
    frames = _frame_candidates(g)
    if len(frames) != 3:
        return out

    order = _occlusion_order(g, frames)
    if order is None or len(order) != 3:
        return out

    bottom = order[0]

    for color in order:
        if color == bottom:
            continue
        info = frames[color]
        sprites = [c for c in _find_sprite_components(g, color, info["bbox"]) if c]
        if len(sprites) != 1:
            continue
        comp = sprites[0]
        side = _nearest_side(comp, info["bbox"])
        _mirror_sprite(out, comp, info["bbox"], side)

    return out


def _draw_frame(grid: np.ndarray, color: int, r0: int, c0: int, r1: int, c1: int) -> None:
    for c in range(c0, c1 + 1):
        grid[r0, c] = color
        grid[r1, c] = color
    for r in range(r0 + 1, r1):
        grid[r, c0] = color
        grid[r, c1] = color


def _make_sprite(grid: np.ndarray, color: int, target_side: str, frame_bb: tuple[int, int, int, int]) -> None:
    h, w = grid.shape
    r0, c0, r1, c1 = map(int, frame_bb)
    if target_side == "left":
        sr = random.randint(r0 + 1, r1 - 1)
        sc = random.randint(1, max(1, c0 - 1))
    elif target_side == "right":
        sr = random.randint(r0 + 1, r1 - 1)
        sc = random.randint(c1 + 1, min(w - 2, c1 + 2))
    elif target_side == "top":
        sr = random.randint(1, max(1, r0 - 1))
        sc = random.randint(c0 + 1, c1 - 1)
    else:
        sr = random.randint(r1 + 1, min(h - 2, r1 + 2))
        sc = random.randint(c0 + 1, c1 - 1)

    candidates = [(sr, sc)]
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    size = random.randint(3, 6)
    for _ in range(size - 1):
        r, c = random.choice(candidates)
        random.shuffle(dirs)
        placed = False
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if grid[nr, nc] != int(BLACK):
                continue
            candidates.append((nr, nc))
            placed = True
            break
        if not placed:
            break

    for r, c in candidates:
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def generate_input() -> np.ndarray:
    h, w = 15, 15
    g = np.full((h, w), int(BLACK), dtype=int)
    colors = [int(c) for c in FRAME_COLORS]
    random.shuffle(colors)

    # 3 overlapping frames, order in paint stack is bottom -> top.
    while True:
        bboxes: list[tuple[int, int, int, int]] = []
        for _ in range(3):
            fh = random.randint(8, 11)
            fw = random.randint(8, 11)
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            bboxes.append((r0, c0, r0 + fh - 1, c0 + fw - 1))

        ok = True
        for i in range(3):
            for j in range(i + 1, 3):
                a = bboxes[i]
                b = bboxes[j]
                if not (_intersects(a, b)):
                    ok = False
        if not ok:
            continue

        # each pair must have perimeter overlap (at least one crossing pixel)
        b = True
        for i in range(3):
            per_i = _frame_perimeter(*bboxes[i])
            for j in range(i + 1, 3):
                if not per_i.intersection(_frame_perimeter(*bboxes[j])):
                    b = False
        if not b:
            continue
        break

    # draw frames in occlusion stack order: bottom -> top
    for color, bbox in zip(colors, bboxes):
        _draw_frame(g, color, *bbox)

    for color, bbox in zip(colors, bboxes):
        r0, c0, r1, c1 = bbox
        sides = []
        if c0 >= 3:
            sides.append("left")
        if c1 <= w - 4:
            sides.append("right")
        if r0 >= 3:
            sides.append("top")
        if r1 <= h - 4:
            sides.append("bottom")
        side = random.choice(sides or ["left"])
        _make_sprite(g, color, side, bbox)

    return g


def _intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)
