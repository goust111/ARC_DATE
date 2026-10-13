from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 500
GEN_TILE_H = (3, 5)
GEN_TILE_W = (3, 5)
GEN_COLOR_COUNT = (3, 5)
GEN_PIECE_COUNT = (4, 5)
GEN_MIN_PIECE_AREA = 3
GEN_CANVAS_H = (15, 24)
GEN_CANVAS_W = (15, 24)
GEN_GAP = 1


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()


def _piece_from_component(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    cells = []
    for r, c in coords:
        rr = int(r - r0)
        cc = int(c - c0)
        color = int(grid[r, c])
        patch[rr, cc] = color
        cells.append((rr, cc, color))
    cells.sort()
    return {
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "cells": tuple(cells),
        "area": len(cells),
        "h": patch.shape[0],
        "w": patch.shape[1],
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(area**0.5) + 1):
        if area % h == 0:
            w = area // h
            pairs.append((h, w))
            if h != w:
                pairs.append((w, h))
    return sorted(set(pairs))


def _tile_has_equal_halves(tile: np.ndarray) -> bool:
    h, w = tile.shape
    if w % 2 == 0 and np.array_equal(tile[:, : w // 2], tile[:, w // 2 :]):
        return True
    if h % 2 == 0 and np.array_equal(tile[: h // 2, :], tile[h // 2 :, :]):
        return True
    return False


def _extract_pieces(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    pieces = [_piece_from_component(grid, comp, bg) for comp in comps]
    pieces.sort(key=lambda p: (-p["area"], p["h"], p["w"], p["cells"]))
    return bg, pieces


def _candidate_placements(piece: dict, full_h: int, full_w: int, axis: str) -> tuple[list[dict], list[list[int]], int, tuple[int, int]]:
    if axis == "v":
        if full_w % 2:
            return [], [], 0, (0, 0)
        tile_h, tile_w = full_h, full_w // 2
        n = tile_h * tile_w
    else:
        if full_h % 2:
            return [], [], 0, (0, 0)
        tile_h, tile_w = full_h // 2, full_w
        n = tile_h * tile_w

    placements = []
    req_to_pls = [[] for _ in range(2 * n)]
    for r0 in range(full_h - piece["h"] + 1):
        for c0 in range(full_w - piece["w"] + 1):
            reqs = []
            cells = []
            used_reqs = set()
            pair_colors: dict[int, int] = {}
            ok = True
            for dr, dc, color in piece["cells"]:
                r = r0 + dr
                c = c0 + dc
                if axis == "v":
                    side = 0 if c < tile_w else 1
                    idx = r * tile_w + (c if side == 0 else c - tile_w)
                else:
                    side = 0 if r < tile_h else 1
                    idx = (r if side == 0 else r - tile_h) * tile_w + c
                req = side * n + idx
                if req in used_reqs:
                    ok = False
                    break
                used_reqs.add(req)
                prev = pair_colors.get(idx)
                if prev is not None and prev != color:
                    ok = False
                    break
                pair_colors[idx] = color
                reqs.append(req)
                cells.append((side, idx, color))
            if not ok:
                continue
            pl_index = len(placements)
            placements.append({"reqs": tuple(reqs), "cells": tuple(cells)})
            for req in reqs:
                req_to_pls[req].append(pl_index)
    return placements, req_to_pls, n, (tile_h, tile_w)


def _solve_shape_axis(pieces: list[dict], full_h: int, full_w: int, axis: str, limit: int = 2) -> list[np.ndarray]:
    prepared = []
    for piece in pieces:
        placements, req_to_pls, n, tile_shape = _candidate_placements(piece, full_h, full_w, axis)
        if not placements:
            return []
        prepared.append((placements, req_to_pls))

    if axis == "v":
        tile_h, tile_w = full_h, full_w // 2
    else:
        tile_h, tile_w = full_h // 2, full_w
    n = tile_h * tile_w
    occ = np.full((2, n), -1, dtype=int)
    solutions: list[np.ndarray] = []

    def compatible(pl: dict) -> bool:
        for side, idx, color in pl["cells"]:
            if occ[side, idx] != -1:
                return False
            other = occ[1 - side, idx]
            if other != -1 and other != color:
                return False
        return True

    def choose_requirement(remaining_mask: int):
        best_req = -1
        best_candidates: list[tuple[int, int]] | None = None
        for req in range(2 * n):
            side = req // n
            idx = req % n
            if occ[side, idx] != -1:
                continue
            candidates = []
            for pi, (placements, req_to_pls) in enumerate(prepared):
                if not (remaining_mask & (1 << pi)):
                    continue
                for pl_index in req_to_pls[req]:
                    if compatible(placements[pl_index]):
                        candidates.append((pi, pl_index))
            if not candidates:
                return req, []
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_req = req
                best_candidates = candidates
                if len(best_candidates) == 1:
                    break
        return best_req, best_candidates

    def place(pl: dict) -> list[tuple[int, int]]:
        changed = []
        for side, idx, color in pl["cells"]:
            occ[side, idx] = color
            changed.append((side, idx))
        return changed

    def undo(changed: list[tuple[int, int]]) -> None:
        for side, idx in changed:
            occ[side, idx] = -1

    def search(remaining_mask: int) -> None:
        if len(solutions) >= limit:
            return
        if remaining_mask == 0:
            if np.any(occ == -1) or not np.array_equal(occ[0], occ[1]):
                return
            solutions.append(occ[0].reshape(tile_h, tile_w).copy())
            return

        req, candidates = choose_requirement(remaining_mask)
        if req < 0 or not candidates:
            return
        for pi, pl_index in candidates:
            if not (remaining_mask & (1 << pi)):
                continue
            placement = prepared[pi][0][pl_index]
            changed = place(placement)
            search(remaining_mask & ~(1 << pi))
            undo(changed)
            if len(solutions) >= limit:
                return

    search((1 << len(pieces)) - 1)
    return solutions


def _solve_all_assemblies(grid: np.ndarray, limit: int = 2) -> list[np.ndarray]:
    _bg, pieces = _extract_pieces(grid)
    total_area = sum(piece["area"] for piece in pieces)
    if total_area % 2:
        return []

    assemblies: list[np.ndarray] = []
    for full_h, full_w in _factor_pairs(total_area):
        if not (2 <= full_h <= 30 and 2 <= full_w <= 30):
            continue
        for axis in ("v", "h"):
            sols = _solve_shape_axis(pieces, full_h, full_w, axis, limit=limit - len(assemblies))
            assemblies.extend(sols)
            if len(assemblies) >= limit:
                return assemblies
    return assemblies


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble scattered fragments into a doubled rectangle and return its single source half.

    Input structure:
    - A background canvas contains several disconnected multicolor fragments.
    - Fragments may only be translated; their orientation and internal colors stay fixed.
    - All fragments can be reassembled into one solid rectangle whose two adjacent halves are identical.

    Rule:
    - Translate every fragment so they tile one gapless rectangle without overlap.
    - The valid assembly is the unique one where the rectangle is exactly two equal copies joined horizontally or vertically.
    - Find that duplication axis and output the first half of the assembled rectangle.

    Key insight:
    - No target picture is given; the only thing that locks the jigsaw assembly is that the finished rectangle must itself be a perfect two-copy repetition, and the answer removes the redundant second copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    assemblies = _solve_all_assemblies(g, limit=2)
    assert len(assemblies) == 1, "expected exactly one valid doubled-rectangle assembly"
    return assemblies[0]


def _random_tile() -> np.ndarray:
    for _ in range(200):
        h = random.randint(*GEN_TILE_H)
        w = random.randint(*GEN_TILE_W)
        color_count = random.randint(*GEN_COLOR_COUNT)
        color_count = min(color_count, h * w)
        colors = random.sample(COLOR_POOL, color_count)
        tile = np.empty((h, w), dtype=int)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        for (r, c), color in zip(cells, colors):
            tile[r, c] = color
        for r, c in cells[len(colors) :]:
            tile[r, c] = random.choice(colors)
        if not _tile_has_equal_halves(tile):
            return tile
    raise RuntimeError("failed to sample a non-self-duplicable tile")


def _random_partition(h: int, w: int, piece_count: int, min_area: int) -> np.ndarray | None:
    area = h * w
    if piece_count * min_area > area:
        return None

    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, piece_count)
    labels = -np.ones((h, w), dtype=int)
    piece_cells = [set() for _ in range(piece_count)]
    frontiers = [set() for _ in range(piece_count)]

    for idx, (r, c) in enumerate(seeds):
        labels[r, c] = idx
        piece_cells[idx].add((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == -1:
                frontiers[idx].add((nr, nc))

    unassigned = area - piece_count
    while unassigned > 0:
        choices = [i for i in range(piece_count) if frontiers[i]]
        if not choices:
            return None
        needy = [i for i in choices if len(piece_cells[i]) < min_area]
        if needy:
            grow = min(needy, key=lambda i: (len(piece_cells[i]), random.random()))
        else:
            grow = min(choices, key=lambda i: (len(piece_cells[i]) + random.random(), random.random()))
        nr, nc = random.choice(tuple(frontiers[grow]))
        frontiers[grow].remove((nr, nc))
        if labels[nr, nc] != -1:
            continue
        labels[nr, nc] = grow
        piece_cells[grow].add((nr, nc))
        unassigned -= 1
        for rr, cc in _neighbors4(nr, nc):
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] == -1:
                frontiers[grow].add((rr, cc))
        for rr, cc in _neighbors4(nr, nc):
            if 0 <= rr < h and 0 <= cc < w and labels[rr, cc] != -1:
                frontiers[labels[rr, cc]].discard((nr, nc))

    if any(np.count_nonzero(labels == idx) < min_area for idx in range(piece_count)):
        return None
    return labels


def _piece_patches_from_labels(full: np.ndarray, labels: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    patches = []
    coord_sets = []
    for idx in range(labels.max() + 1):
        coords = np.argwhere(labels == idx)
        coord_sets.append(coords)
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in coords:
            patch[r - r0, c - c0] = full[r, c]
        patches.append(patch)
    return patches, coord_sets


def _piece_signature(patch: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return patch.shape, patch.tobytes()


def _crosses_axis(coords: np.ndarray, axis: str, tile_h: int, tile_w: int) -> bool:
    if axis == "v":
        cols = coords[:, 1]
        return bool(np.any(cols < tile_w) and np.any(cols >= tile_w))
    rows = coords[:, 0]
    return bool(np.any(rows < tile_h) and np.any(rows >= tile_h))


def _multicolor_piece_count(patches: list[np.ndarray]) -> int:
    total = 0
    for patch in patches:
        colors = set(map(int, patch[patch != BG].ravel()))
        if len(colors) >= 2:
            total += 1
    return total


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    total_area = sum(int(np.count_nonzero(patch != BG)) for patch in patches)
    ordered = sorted(patches, key=lambda patch: (-patch.shape[0] * patch.shape[1], random.random()))

    for _ in range(80):
        h = random.randint(*GEN_CANVAS_H)
        w = random.randint(*GEN_CANVAS_W)
        if total_area * 2 >= h * w:
            continue
        grid = np.full((h, w), BG, dtype=int)
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for patch in ordered:
            ph, pw = patch.shape
            options = [
                (r0, c0)
                for r0 in range(h - ph + 1)
                for c0 in range(w - pw + 1)
                if all(
                    _boxes_separate((r0, c0, r0 + ph - 1, c0 + pw - 1), box, GEN_GAP)
                    for box in boxes
                )
            ]
            if not options:
                ok = False
                break
            r0, c0 = random.choice(options)
            boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
            mask = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        if ok:
            return grid
    return None


def _generate_attempt() -> np.ndarray | None:
    tile = _random_tile()
    axis = random.choice(("v", "h"))
    full = np.concatenate([tile, tile], axis=1 if axis == "v" else 0)
    tile_h, tile_w = tile.shape
    full_h, full_w = full.shape

    max_piece_count = min(GEN_PIECE_COUNT[1], (full_h * full_w) // GEN_MIN_PIECE_AREA)
    if max_piece_count < GEN_PIECE_COUNT[0]:
        return None
    piece_count = random.randint(GEN_PIECE_COUNT[0], max_piece_count)

    labels = None
    for _ in range(120):
        candidate = _random_partition(full_h, full_w, piece_count, GEN_MIN_PIECE_AREA)
        if candidate is None:
            continue
        patches, coord_sets = _piece_patches_from_labels(full, candidate)
        if len({_piece_signature(patch) for patch in patches}) != len(patches):
            continue
        if not any(_crosses_axis(coords, axis, tile_h, tile_w) for coords in coord_sets):
            continue
        if _multicolor_piece_count(patches) < 2:
            continue
        labels = candidate
        break
    if labels is None:
        return None

    patches, _coord_sets = _piece_patches_from_labels(full, labels)
    grid = _scatter_patches(patches)
    if grid is None:
        return None

    if np.count_nonzero(grid == BG) <= np.count_nonzero(grid != BG):
        return None

    assemblies = _solve_all_assemblies(grid, limit=2)
    if len(assemblies) != 1 or not np.array_equal(assemblies[0], tile):
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, tile):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
