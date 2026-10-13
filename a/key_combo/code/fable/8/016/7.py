from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np


HOLE = 0
GEN_ATTEMPTS = 300
GEN_BG_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _infer_bg(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    bg, _ = counts.most_common(1)[0]
    if bg != HOLE or len(counts) == 1:
        return int(bg)
    non_hole = [(color, count) for color, count in counts.items() if color != HOLE]
    assert non_hole, "expected at least one non-hole color"
    return int(max(non_hole, key=lambda item: item[1])[0])


def _find_pieces(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    mask = g != int(bg)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    pieces: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            submask = mask[r0 : r1 + 1, c0 : c1 + 1]
            assert bool(submask.all()), "each fragment must occupy a full rectangle"
            pieces.append(g[r0 : r1 + 1, c0 : c1 + 1].copy())

    assert pieces, "expected at least one rectangular fragment"
    return pieces


def _build_orbits(size: int) -> tuple[np.ndarray, list[tuple[tuple[int, int], ...]]]:
    orbit_index = np.full((size, size), -1, dtype=int)
    orbits: list[tuple[tuple[int, int], ...]] = []
    lookup: dict[tuple[tuple[int, int], ...], int] = {}

    for r in range(size):
        for c in range(size):
            cells = tuple(
                sorted(
                    {
                        (r, c),
                        (size - 1 - r, c),
                        (r, size - 1 - c),
                        (size - 1 - r, size - 1 - c),
                    }
                )
            )
            oid = lookup.get(cells)
            if oid is None:
                oid = len(orbits)
                lookup[cells] = oid
                orbits.append(cells)
            orbit_index[r, c] = int(oid)

    return orbit_index, orbits


def _solve_outputs_from_pieces(pieces: list[np.ndarray], limit: int = 2) -> list[np.ndarray]:
    total_area = sum(int(piece.shape[0] * piece.shape[1]) for piece in pieces)
    side = int(math.isqrt(total_area))
    assert side * side == total_area, "fragment areas must tile a square"

    orbit_index, orbits = _build_orbits(side)
    orbit_sizes = np.array([len(cells) for cells in orbits], dtype=int)

    piece_infos = []
    for idx, piece in enumerate(pieces):
        arr = np.asarray(piece, dtype=int)
        colored = [(r, c, int(arr[r, c])) for r in range(arr.shape[0]) for c in range(arr.shape[1]) if int(arr[r, c]) != HOLE]
        piece_infos.append(
            {
                "idx": idx,
                "grid": arr,
                "h": int(arr.shape[0]),
                "w": int(arr.shape[1]),
                "area": int(arr.shape[0] * arr.shape[1]),
                "colored": colored,
            }
        )

    order = sorted(
        range(len(piece_infos)),
        key=lambda i: (
            -piece_infos[i]["area"],
            -len(piece_infos[i]["colored"]),
            -piece_infos[i]["h"],
            -piece_infos[i]["w"],
            piece_infos[i]["idx"],
        ),
    )

    occupied = np.zeros((side, side), dtype=bool)
    used = [False] * len(piece_infos)
    orbit_colors = np.full(len(orbits), -1, dtype=int)
    orbit_cover = np.zeros(len(orbits), dtype=int)

    solutions: list[np.ndarray] = []
    seen_outputs: set[bytes] = set()

    def try_place(piece_id: int, top: int, left: int):
        info = piece_infos[piece_id]
        h = int(info["h"])
        w = int(info["w"])
        if bool(occupied[top : top + h, left : left + w].any()):
            return None

        cover_add: dict[int, int] = {}
        for rr in range(h):
            for cc in range(w):
                oid = int(orbit_index[top + rr, left + cc])
                cover_add[oid] = cover_add.get(oid, 0) + 1

        color_add: dict[int, int] = {}
        for rr, cc, color in info["colored"]:
            oid = int(orbit_index[top + rr, left + cc])
            current = color_add.get(oid, int(orbit_colors[oid]))
            if current != -1 and current != color:
                return None
            color_add[oid] = int(color)

        for oid, add in cover_add.items():
            final_color = color_add.get(oid, int(orbit_colors[oid]))
            if int(orbit_cover[oid]) + add == int(orbit_sizes[oid]) and final_color == -1:
                return None

        return info, cover_add, color_add

    def place(piece_id: int, top: int, left: int, placement) -> list[tuple[int, int, int]]:
        info, cover_add, color_add = placement
        h = int(info["h"])
        w = int(info["w"])
        occupied[top : top + h, left : left + w] = True
        used[piece_id] = True
        updates: list[tuple[int, int, int]] = []
        for oid, add in cover_add.items():
            old_color = int(orbit_colors[oid])
            orbit_cover[oid] += int(add)
            if oid in color_add and old_color == -1:
                orbit_colors[oid] = int(color_add[oid])
            updates.append((oid, old_color, int(add)))
        return updates

    def unplace(piece_id: int, top: int, left: int, placement, updates: list[tuple[int, int, int]]) -> None:
        info, _cover_add, _color_add = placement
        h = int(info["h"])
        w = int(info["w"])
        occupied[top : top + h, left : left + w] = False
        used[piece_id] = False
        for oid, old_color, add in updates:
            orbit_cover[oid] -= int(add)
            orbit_colors[oid] = int(old_color)

    def build_output() -> np.ndarray | None:
        if bool((orbit_colors < 0).any()):
            return None
        out = np.zeros((side, side), dtype=int)
        for oid, cells in enumerate(orbits):
            color = int(orbit_colors[oid])
            if color < 0:
                return None
            for r, c in cells:
                out[r, c] = color
        return out

    def search() -> None:
        if len(solutions) >= limit:
            return
        flat = np.flatnonzero(~occupied.ravel())
        if flat.size == 0:
            out = build_output()
            if out is None:
                return
            key = out.tobytes()
            if key not in seen_outputs:
                seen_outputs.add(key)
                solutions.append(out)
            return

        first = int(flat[0])
        target_r, target_c = divmod(first, side)
        for piece_id in order:
            if used[piece_id]:
                continue
            info = piece_infos[piece_id]
            h = int(info["h"])
            w = int(info["w"])
            r_start = max(0, target_r - h + 1)
            r_end = min(target_r, side - h)
            c_start = max(0, target_c - w + 1)
            c_end = min(target_c, side - w)
            for top in range(r_start, r_end + 1):
                for left in range(c_start, c_end + 1):
                    placement = try_place(piece_id, top, left)
                    if placement is None:
                        continue
                    updates = place(piece_id, top, left, placement)
                    search()
                    unplace(piece_id, top, left, placement, updates)
                    if len(solutions) >= limit:
                        return

    search()
    return solutions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular fragments into the unique biaxially symmetric square and fill their hole cells by symmetry.

    Input structure:
    - A uniform non-hole background contains several separated rectangular fragments.
    - Each fragment is an axis-aligned rectangle made of visible colored cells plus some black hole cells.
    - Black cells inside fragments are missing values, not background, and fragments may not be rotated or flipped.

    Rule:
    - Translate every fragment to tile one square exactly, with no overlap and no gaps.
    - Among all tilings, keep the unique assembly whose completed square is symmetric across both the horizontal and vertical midlines when black holes are ignored.
    - For each hole cell, copy the unique visible color from its reflection orbit and output the completed square only.

    Key insight:
    - Local edge matching is irrelevant here; the real constraint is that every reflection orbit in the assembled square must agree on a single visible color, so assembly and hole-filling are solved together by global symmetry.
    """
    g = np.asarray(grid, dtype=int)
    bg = _infer_bg(g)
    pieces = _find_pieces(g, bg)
    solutions = _solve_outputs_from_pieces(pieces, limit=2)
    assert solutions, "expected at least one symmetric square assembly"
    assert len(solutions) == 1, "expected a unique completed symmetric output"
    return solutions[0]


def _random_partition(size: int, target_parts: int) -> list[tuple[int, int, int, int]]:
    rects = [(0, 0, size, size)]
    while len(rects) < target_parts:
        candidates = []
        for idx, (_r, _c, h, w) in enumerate(rects):
            if h >= 4:
                candidates.append((idx, "h"))
            if w >= 4:
                candidates.append((idx, "v"))
        if not candidates:
            break
        idx, axis = random.choice(candidates)
        r, c, h, w = rects.pop(idx)
        if axis == "h":
            cut = random.randint(2, h - 2)
            rects.extend([(r, c, cut, w), (r + cut, c, h - cut, w)])
        else:
            cut = random.randint(2, w - 2)
            rects.extend([(r, c, h, cut), (r, c + cut, h, w - cut)])
    return rects


def _make_symmetric_solution(size: int, palette: list[int]) -> np.ndarray:
    seed_h = (size + 1) // 2
    seed_w = (size + 1) // 2
    seed = np.full((seed_h, seed_w), random.choice(palette), dtype=int)

    paint_count = random.randint(4, 7)
    for _ in range(paint_count):
        color = random.choice(palette)
        h = random.randint(1, seed_h)
        w = random.randint(1, seed_w)
        r0 = random.randint(0, seed_h - h)
        c0 = random.randint(0, seed_w - w)
        seed[r0 : r0 + h, c0 : c0 + w] = int(color)

    if len(set(map(int, seed.ravel()))) < min(3, len(palette)):
        for color in random.sample(palette, k=min(3, len(palette))):
            seed[random.randrange(seed_h), random.randrange(seed_w)] = int(color)

    out = np.zeros((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            rr = min(r, size - 1 - r)
            cc = min(c, size - 1 - c)
            out[r, c] = int(seed[rr, cc])
    return out


def _choose_holes(
    solved: np.ndarray,
    rects: list[tuple[int, int, int, int]],
) -> np.ndarray | None:
    size = int(solved.shape[0])
    orbit_index, orbits = _build_orbits(size)
    orbit_holes = np.zeros(len(orbits), dtype=int)
    piece_map = np.full((size, size), -1, dtype=int)
    piece_areas: list[int] = []
    for pid, (r, c, h, w) in enumerate(rects):
        piece_map[r : r + h, c : c + w] = int(pid)
        piece_areas.append(int(h * w))

    max_holes_per_piece = [max(1, min(3, area // 5)) for area in piece_areas]
    piece_holes = [0] * len(rects)
    hole_mask = np.zeros_like(solved, dtype=bool)

    target_holes = random.randint(1, min(6, max(1, solved.size // 8)))
    candidates = [(r, c) for r in range(size) for c in range(size)]
    random.shuffle(candidates)

    for r, c in candidates:
        pid = int(piece_map[r, c])
        oid = int(orbit_index[r, c])
        if piece_holes[pid] >= max_holes_per_piece[pid]:
            continue
        if orbit_holes[oid] >= len(orbits[oid]) - 1:
            continue
        if piece_areas[pid] - piece_holes[pid] <= 2:
            continue
        if random.random() < 0.55:
            hole_mask[r, c] = True
            piece_holes[pid] += 1
            orbit_holes[oid] += 1
            if int(hole_mask.sum()) >= target_holes:
                break

    if not bool(hole_mask.any()):
        for r, c in candidates:
            pid = int(piece_map[r, c])
            oid = int(orbit_index[r, c])
            if orbit_holes[oid] < len(orbits[oid]) - 1 and piece_areas[pid] > 2:
                hole_mask[r, c] = True
                break

    if not bool(hole_mask.any()):
        return None

    for pid, (r, c, h, w) in enumerate(rects):
        piece = solved[r : r + h, c : c + w]
        visible = piece[~hole_mask[r : r + h, c : c + w]]
        if visible.size == 0 or len(set(map(int, visible))) < 2:
            return None

    return hole_mask


def _scatter_pieces(
    pieces: list[np.ndarray],
    bg: int,
) -> np.ndarray | None:
    max_piece_h = max(int(piece.shape[0]) for piece in pieces)
    max_piece_w = max(int(piece.shape[1]) for piece in pieces)
    min_h = max(16, max_piece_h + 4)
    min_w = max(16, max_piece_w + 4)
    height = random.randint(min_h, 28)
    width = random.randint(min_w, 28)

    grid = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for piece_id in order:
        piece = pieces[piece_id]
        h, w = map(int, piece.shape)
        placements = []
        for r in range(height - h + 1):
            for c in range(width - w + 1):
                r0 = max(0, r - 1)
                r1 = min(height, r + h + 1)
                c0 = max(0, c - 1)
                c1 = min(width, c + w + 1)
                if bool(occupied[r0:r1, c0:c1].any()):
                    continue
                placements.append((r, c))
        if not placements:
            return None
        r, c = random.choice(placements)
        grid[r : r + h, c : c + w] = piece
        occupied[r : r + h, c : c + w] = True

    return grid


def _generate_attempt() -> np.ndarray | None:
    size = random.randint(6, 10)
    target_parts = random.randint(3, 5)
    bg = random.choice(GEN_BG_CHOICES)
    palette_pool = [color for color in GEN_BG_CHOICES if color != bg]
    palette = random.sample(palette_pool, k=random.randint(3, min(5, len(palette_pool))))

    solved = _make_symmetric_solution(size, palette)
    rects = _random_partition(size, target_parts)
    if len(rects) < 3:
        return None

    hole_mask = _choose_holes(solved, rects)
    if hole_mask is None:
        return None

    pieces = []
    signatures = []
    for r, c, h, w in rects:
        piece = solved[r : r + h, c : c + w].copy()
        piece_holes = hole_mask[r : r + h, c : c + w]
        piece[piece_holes] = HOLE
        pieces.append(piece)
        signatures.append((piece.shape, piece.tobytes()))

    if len(set(signatures)) != len(signatures):
        return None

    scattered = _scatter_pieces(pieces, bg)
    if scattered is None:
        return None

    try:
        solutions = _solve_outputs_from_pieces(_find_pieces(scattered, bg), limit=2)
    except AssertionError:
        return None
    if len(solutions) != 1:
        return None
    if not np.array_equal(solutions[0], solved):
        return None
    return scattered


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetric-fragment puzzle")
