import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MARKER_COLORS = [1, 2, 3, 4, 6, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _cells_to_mask(cells: list[tuple[int, int]]) -> np.ndarray:
    norm = _normalize(cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _rotations(cells: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    base = _cells_to_mask(cells)
    out = []
    seen = set()
    for k in range(4):
        rot = np.rot90(base, k)
        pts = [(int(r), int(c)) for r, c in np.argwhere(rot)]
        norm = _normalize(pts)
        if norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def _shape_key(cells: list[tuple[int, int]]) -> tuple:
    reps = []
    for rot in _rotations(cells):
        mask = _cells_to_mask(list(rot))
        reps.append((mask.shape, mask.astype(np.uint8).tobytes()))
    return min(reps)


def _extract_cavities(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    cavities = []
    for comp in _components(grid == BG):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        neigh = set()
        marker_cells = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                v = int(grid[nr, nc])
                if v == BG:
                    continue
                neigh.add(v)
                if v not in (BG, GREY, ORANGE):
                    marker_cells.add((nr, nc))
        marker_colors = {int(grid[r, c]) for r, c in marker_cells}
        if GREY not in neigh or len(marker_colors) != 1 or len(marker_cells) != 1:
            continue
        cavities.append(
            {
                "cells": comp,
                "shape": _normalize(comp),
                "marker_color": next(iter(marker_colors)),
            }
        )
    return cavities


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    for comp in _components(grid == ORANGE):
        pieces.append({"cells": comp, "rots": _rotations(comp)})
    return pieces


def _match_cavities_to_pieces(cavities: list[dict], pieces: list[dict]) -> list[int]:
    by_shape: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        for rot in piece["rots"]:
            by_shape[rot].append(idx)

    order = sorted(range(len(cavities)), key=lambda i: len(set(by_shape[cavities[i]["shape"]])))
    used = [False] * len(pieces)
    assign = [-1] * len(cavities)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        ci = order[pos]
        candidates = []
        seen = set()
        for pi in by_shape[cavities[ci]["shape"]]:
            if used[pi] or pi in seen:
                continue
            seen.add(pi)
            candidates.append(pi)
        for pi in candidates:
            used[pi] = True
            assign[ci] = pi
            if dfs(pos + 1):
                return True
            used[pi] = False
            assign[ci] = -1
        return False

    assert dfs(0), "failed to find cavity-piece matching"
    return assign


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    cavities = _extract_cavities(g)
    pieces = _extract_pieces(g)
    assert cavities, "expected at least one enclosed cavity"
    assert len(cavities) == len(pieces), "piece count must match cavity count"
    _match_cavities_to_pieces(cavities, pieces)

    out = g.copy()
    out[out == ORANGE] = BG
    for cavity in cavities:
        for r, c in cavity["cells"]:
            out[r, c] = int(cavity["marker_color"])
    return out


def _random_shape(rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        area_max = min(h * w - 1, 9)
        area_min = min(area_max, max(3, (h * w) // 2))
        if area_min > area_max:
            continue
        target = rng.randint(area_min, area_max)
        cells = {(rng.randrange(h), rng.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        norm = _normalize(list(cells))
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        if hh < 2 or ww < 2:
            continue
        if len(norm) == hh * ww:
            continue
        return list(norm)
    raise RuntimeError("shape generation failed")


def _make_frame_patch(shape: list[tuple[int, int]], marker_color: int, rng: random.Random) -> np.ndarray:
    norm = _normalize(shape)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.full((h + 2, w + 2), GREY, dtype=int)
    for r, c in norm:
        patch[r + 1, c + 1] = BG
    candidates = []
    for r, c in norm:
        rr, cc = r + 1, c + 1
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and patch[nr, nc] == GREY:
                candidates.append((nr, nc))
    mr, mc = rng.choice(candidates)
    patch[mr, mc] = marker_color
    return patch


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int]:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _can_place(occ: np.ndarray, top: int, left: int, patch: np.ndarray, margin: int) -> bool:
    h, w = patch.shape
    hh, ww = occ.shape
    if top < 0 or left < 0 or top + h > hh or left + w > ww:
        return False
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(hh, top + h + margin)
    c1 = min(ww, left + w + margin)
    return not occ[r0:r1, c0:c1].any()


def _stamp(grid: np.ndarray, occ: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = grid[top : top + h, left : left + w]
    sub[mask] = patch[mask]
    occ[top : top + h, left : left + w] |= mask


def _piece_patch(shape: list[tuple[int, int]], rot_k: int) -> np.ndarray:
    mask = _cells_to_mask(shape)
    rot = np.rot90(mask, rot_k)
    out = np.full(rot.shape, BG, dtype=int)
    out[rot] = ORANGE
    return out


def _solution_count(grid: np.ndarray) -> int:
    cavities = _extract_cavities(grid)
    pieces = _extract_pieces(grid)
    if len(cavities) != len(pieces) or not cavities:
        return 0
    by_shape: dict[tuple[tuple[int, int], ...], list[int]] = defaultdict(list)
    for idx, piece in enumerate(pieces):
        for rot in piece["rots"]:
            by_shape[rot].append(idx)
    order = sorted(range(len(cavities)), key=lambda i: len(set(by_shape[cavities[i]["shape"]])))
    used = [False] * len(pieces)
    total = 0

    def dfs(pos: int) -> None:
        nonlocal total
        if total > 1:
            return
        if pos == len(order):
            total += 1
            return
        ci = order[pos]
        candidates = []
        seen = set()
        for pi in by_shape[cavities[ci]["shape"]]:
            if used[pi] or pi in seen:
                continue
            seen.add(pi)
            candidates.append(pi)
        for pi in candidates:
            used[pi] = True
            dfs(pos + 1)
            used[pi] = False

    dfs(0)
    return total


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_pairs = rng.randint(2, 4)
    shapes = []
    shape_keys = set()
    for _ in range(n_pairs):
        for _ in range(200):
            shape = _random_shape(rng)
            key = _shape_key(shape)
            if key not in shape_keys:
                shape_keys.add(key)
                shapes.append(shape)
                break
        else:
            return None

    marker_colors = rng.sample(MARKER_COLORS, n_pairs)
    objects = []
    for shape, marker_color in zip(shapes, marker_colors):
        frame = _make_frame_patch(shape, marker_color, rng)
        rot_k = rng.randrange(4)
        piece = _piece_patch(shape, rot_k)
        objects.append((frame, piece))

    rng.shuffle(objects)

    for frame, piece in objects:
        placed = False
        for _ in range(200):
            ft = rng.randint(1, h - frame.shape[0] - 1)
            fl = rng.randint(1, w - frame.shape[1] - 1)
            if not _can_place(occ, ft, fl, frame, margin=1):
                continue
            _stamp(grid, occ, ft, fl, frame)
            placed = True
            break
        if not placed:
            return None

        placed = False
        for _ in range(300):
            pt = rng.randint(1, h - piece.shape[0] - 1)
            pl = rng.randint(1, w - piece.shape[1] - 1)
            if not _can_place(occ, pt, pl, piece, margin=2):
                continue
            _stamp(grid, occ, pt, pl, piece)
            placed = True
            break
        if not placed:
            return None

    cavities = _extract_cavities(grid)
    pieces = _extract_pieces(grid)
    if len(cavities) != n_pairs or len(pieces) != n_pairs:
        return None
    if _solution_count(grid) != 1:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
