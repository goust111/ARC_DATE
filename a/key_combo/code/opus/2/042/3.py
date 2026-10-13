from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from collections import deque as deqmod

import numpy as np

import os
import sys
from pathlib import Path


_arc_utils_path = None
for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    for _cand_dir in (_p / "new" / "solvers", _p / "solvers"):
        _cand = _cand_dir / "arc_utils.py"
        if _cand.exists():
            _arc_utils_path = str(_cand_dir)
            break
    if _arc_utils_path is not None:
        break
if _arc_utils_path and _arc_utils_path not in sys.path:
    sys.path.insert(0, _arc_utils_path)

from arc_utils import find_components, get_bg



_BG = 0
_FRAME_COLOR_LIST = [1, 2, 3, 4, 6, 7]
_INNER_COLORS = [2, 3, 4, 6, 7, 1]


def _neighbors8(r: int, c: int, H: int, W: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out.append((rr, cc))
    return out


def _neighbors4(r: int, c: int, H: int, W: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < H and 0 <= cc < W:
            out.append((rr, cc))
    return out


def _extract_sprites(grid: np.ndarray, bg: int):
    mask = (grid != bg) & (grid != 5)
    comp_grid = np.where(mask, 1, 0).astype(int)
    comps = find_components(comp_grid, bg=0, connectivity=4, monochromatic=True, return_format="dict")

    sprites = []
    for comp in comps:
        coords = np.array(comp["coords"], dtype=int)
        if coords.shape[0] <= 1:
            continue

        r0, c0, r1, c1 = comp["bbox"]
        minr, minc, maxr, maxc = r0, c0, r1, c1
        inside = []
        border = []
        for r, c in coords:
            if minr < int(r) < maxr and minc < int(c) < maxc:
                inside.append((int(r), int(c)))
            else:
                border.append((int(r), int(c)))

        if not inside:
            continue

        border_colors = Counter(int(grid[r, c]) for r, c in border)
        interior_colors = Counter(int(grid[r, c]) for r, c in inside)

        norm_r = min(r for r, _ in inside)
        norm_c = min(c for _, c in inside)
        shape = tuple(sorted((r - norm_r, c - norm_c) for r, c in inside))

        sprites.append(
            {
                "coords": [tuple(map(int, p)) for p in coords],
                "bbox": comp["bbox"],
                "border_color": int(border_colors.most_common(1)[0][0]),
                "inner_color": int(interior_colors.most_common(1)[0][0]),
                "inner": inside,
                "shape": shape,
            }
        )
    return sprites


def _shortest_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int], blocked: np.ndarray) -> list[tuple[int, int]] | None:
    H, W = grid.shape
    if start == goal:
        return [start]
    q = deqmod([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in _neighbors8(r, c, H, W):
            if blocked[nr, nc] or (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            q.append((nr, nc))

    if goal not in parent:
        return None

    path = [goal]
    cur = goal
    while parent[cur] is not None:
        cur = parent[cur]
        path.append(cur)
    path.reverse()
    return path


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    H, W = g.shape
    out = g.copy()

    sprites = _extract_sprites(g, bg)
    if not sprites:
        return g.copy()

    shape_map: dict[tuple[tuple[int, int], ...], list[int]] = {}
    for i, s in enumerate(sprites):
        shape_map.setdefault(s["shape"], []).append(i)

    pairs: list[tuple[int, int]] = []
    for idxs in shape_map.values():
        if len(idxs) >= 2:
            for i in range(0, len(idxs), 2):
                if i + 1 < len(idxs):
                    pairs.append((idxs[i], idxs[i + 1]))

    paired = set(i for p in pairs for i in p)
    unpaired = [i for i in range(len(sprites)) if i not in paired]
    for idx in unpaired:
        r, c = np.array(sprites[idx]["coords"], dtype=int).T
        out[r, c] = bg

    coord_to_sprite: dict[tuple[int, int], int] = {}
    for i, s in enumerate(sprites):
        for coord in s["coords"]:
            coord_to_sprite[coord] = i

    gray_comps = find_components(
        g,
        bg=bg,
        connectivity=8,
        monochromatic=True,
        colors=[5],
        return_format="dict",
    )

    connected_pairs = set[frozenset[int]]()
    for comp in gray_comps:
        touch: set[int] = set()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors8(int(r), int(c), H, W):
                idx = coord_to_sprite.get((nr, nc))
                if idx is not None:
                    touch.add(int(idx))
        for a, b in combinations(sorted(touch), 2):
            connected_pairs.add(frozenset((int(a), int(b))))

    alive_sprites: set[int] = set()
    for a, b in pairs:
        conn = frozenset((a, b)) in connected_pairs
        inner_a = sprites[a]["inner"]
        inner_b = sprites[b]["inner"]
        if not inner_a or not inner_b:
            continue

        if conn:
            color_a = sprites[a]["inner_color"]
            color_b = sprites[b]["inner_color"]
            for rr, cc in inner_a:
                out[rr, cc] = color_b
            for rr, cc in inner_b:
                out[rr, cc] = color_a
            alive_sprites.add(a)
            alive_sprites.add(b)
        else:
            for rr, cc in sprites[a]["coords"]:
                out[rr, cc] = bg
            for rr, cc in sprites[b]["coords"]:
                out[rr, cc] = bg

    for idx in paired:
        if idx in alive_sprites:
            continue
        r, c = np.array(sprites[idx]["coords"], dtype=int).T
        out[r, c] = bg

    if gray_comps:
        for comp in gray_comps:
            comp_coords = np.array(comp["coords"], dtype=int)
            touches_alive = False
            for r, c in comp_coords:
                for nr, nc in _neighbors8(int(r), int(c), H, W):
                    sidx = coord_to_sprite.get((nr, nc))
                    if sidx is not None and sidx in alive_sprites:
                        touches_alive = True
                        break
                if touches_alive:
                    break
            if not touches_alive:
                rr = comp_coords[:, 0]
                cc = comp_coords[:, 1]
                out[rr, cc] = bg

    return out


def _random_shape(area: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    frontier = {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)}
    frontier -= {(0, 0)}

    while len(cells) < area:
        if not frontier:
            cells.add((0, 0))
            continue
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            continue
        cells.add(cell)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    return cells


def _paint_sprite(grid: np.ndarray, top: int, left: int, h: int, w: int, border: int, inner: int, shape: set[tuple[int, int]]) -> None:
    for r in range(h):
        for c in range(w):
            rr = top + r
            cc = left + c
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                grid[rr, cc] = border
            elif (r - 1, c - 1) in shape:
                grid[rr, cc] = inner


def _candidate_border_neighbors(
    grid: np.ndarray, top: int, left: int, h: int, w: int, bg: int
) -> list[tuple[int, int]]:
    H, W = grid.shape
    result = []
    r0, c0 = top, left
    r1, c1 = top + h - 1, left + w - 1
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if rr not in (r0, r1) and cc not in (c0, c1):
                continue
            for nr, nc in _neighbors8(rr, cc, H, W):
                if (r0 <= nr <= r1 and c0 <= nc <= c1):
                    continue
                if grid[nr, nc] == bg:
                    result.append((nr, nc))
    return result


def generate_input() -> np.ndarray:
    base_shapes = [
        {(0, 0), (1, 1)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
        {(0, 0), (0, 1), (1, 1), (2, 1)},
    ]
    frame_h, frame_w = 5, 5

    for _ in range(200):
        H = random.randint(16, 18)
        W = random.randint(16, 18)
        if H > 30 or W > 30:
            continue

        g = np.full((H, W), _BG, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        pair_count = random.randint(2, 3)
        shape_pool = random.sample(base_shapes, k=min(pair_count, len(base_shapes)))

        def free(top: int, left: int) -> bool:
            if top < 1 or left < 1 or top + frame_h >= H - 1 or left + frame_w >= W - 1:
                return False
            return not occupied[top - 1 : top + frame_h + 1, left - 1 : left + frame_w + 1].any()

        records: list[dict] = []
        sprite_colors = random.sample(_FRAME_COLOR_LIST, k=pair_count)
        inner_colors = random.sample(_INNER_COLORS, k=pair_count * 2)

        valid = True
        for pair_idx in range(pair_count):
            border = int(sprite_colors[pair_idx])
            inner_a = int(inner_colors[2 * pair_idx])
            inner_b = int(inner_colors[2 * pair_idx + 1])
            shape = set(shape_pool[pair_idx])

            pos_a = None
            pos_b = None
            for _pa in range(600):
                top = random.randint(1, H - frame_h - 1)
                left = random.randint(1, W - frame_w - 1)
                if free(top, left):
                    pos_a = (top, left)
                    break
            if pos_a is None:
                valid = False
                break

            _paint_sprite(g, pos_a[0], pos_a[1], frame_h, frame_w, border, inner_a, shape)
            occupied[pos_a[0] - 1 : pos_a[0] + frame_h + 1, pos_a[1] - 1 : pos_a[1] + frame_w + 1] = True
            records.append(
                {
                    "top": pos_a[0],
                    "left": pos_a[1],
                    "h": frame_h,
                    "w": frame_w,
                    "shape": tuple(sorted(shape)),
                }
            )

            for _pb in range(600):
                top2 = random.randint(1, H - frame_h - 1)
                left2 = random.randint(1, W - frame_w - 1)
                if free(top2, left2):
                    pos_b = (top2, left2)
                    break
            if pos_b is None:
                valid = False
                break

            _paint_sprite(g, pos_b[0], pos_b[1], frame_h, frame_w, border, inner_b, shape)
            occupied[pos_b[0] - 1 : pos_b[0] + frame_h + 1, pos_b[1] - 1 : pos_b[1] + frame_w + 1] = True
            records.append(
                {
                    "top": pos_b[0],
                    "left": pos_b[1],
                    "h": frame_h,
                    "w": frame_w,
                    "shape": tuple(sorted(shape)),
                }
            )

        if not valid or len(records) != pair_count * 2:
            continue

        blocked = g != _BG
        connected_pairs = 0
        for pair_idx in range(pair_count):
            a = records[2 * pair_idx]
            b = records[2 * pair_idx + 1]
            a_neighbors = _candidate_border_neighbors(g, a["top"], a["left"], a["h"], a["w"], _BG)
            b_neighbors = _candidate_border_neighbors(g, b["top"], b["left"], b["h"], b["w"], _BG)
            if not a_neighbors or not b_neighbors:
                continue

            path = None
            for s in a_neighbors:
                for t in b_neighbors:
                    path = _shortest_path(g, s, t, blocked)
                    if path is not None:
                        break
                if path is not None:
                    break

            if path is None:
                continue
            for rr, cc in path:
                g[rr, cc] = 5
                blocked[rr, cc] = True
            connected_pairs += 1

        if connected_pairs == 0:
            continue

        try:
            if main(g).ndim != 2:
                continue
        except Exception:
            continue

        return g

    raise RuntimeError("failed to generate valid puzzle-3 input")
