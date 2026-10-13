from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
COLORS = list(range(1, 10))
Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[Coord] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _proper_divisors(n: int) -> list[int]:
    return [p for p in range(1, int(n)) if n % p == 0]


def _has_axis_period(grid: np.ndarray, period: int, axis: int) -> bool:
    n = grid.shape[axis]
    if period <= 0 or period >= n or n % period != 0:
        return False
    if axis == 0:
        return bool(np.array_equal(grid, np.tile(grid[:period, :], (n // period, 1))))
    return bool(np.array_equal(grid, np.tile(grid[:, :period], (1, n // period))))


def _minimal_axis_period(grid: np.ndarray, axis: int) -> int:
    n = grid.shape[axis]
    for p in _proper_divisors(n):
        if _has_axis_period(grid, p, axis):
            return int(p)
    return int(n)


def _piece_signature(cells: tuple[tuple[int, int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(color)) for r, c, color in cells))


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    comps = _components_mask(g != bg)
    assert len(comps) >= 2, "expected several scattered fragments"
    pieces: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        cells = tuple(
            sorted(
                (int(r) - r0, int(c) - c0, int(g[int(r), int(c)]))
                for r, c in coords
            )
        )
        pieces.append(
            {
                "cells": cells,
                "h": int(r1 - r0 + 1),
                "w": int(c1 - c0 + 1),
                "area": int(len(cells)),
                "sig": _piece_signature(cells),
            }
        )
    pieces.sort(key=lambda p: (-int(p["area"]), -int(p["h"]) * int(p["w"]), p["sig"]))
    return pieces


def _factor_dims(area: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for h in range(2, min(30, int(area)) + 1):
        if area % h != 0:
            continue
        w = area // h
        if 2 <= w <= 30:
            dims.append((int(h), int(w)))
    return dims


def _candidate_placements(piece: dict, h: int, w: int, rp: int, cp: int) -> list[dict]:
    out: list[dict] = []
    ph = int(piece["h"])
    pw = int(piece["w"])
    if ph > h or pw > w:
        return out
    for r0 in range(h - ph + 1):
        for c0 in range(w - pw + 1):
            cells: list[tuple[int, int]] = []
            assigns: dict[int, int] = {}
            ok = True
            for dr, dc, color in piece["cells"]:
                r = r0 + int(dr)
                c = c0 + int(dc)
                color = int(color)
                idx = (r % rp) * cp + (c % cp)
                old = assigns.get(idx)
                if old is not None and old != color:
                    ok = False
                    break
                assigns[idx] = color
                cells.append((r * w + c, color))
            if ok:
                out.append(
                    {
                        "pos": (int(r0), int(c0)),
                        "cells": tuple(cells),
                        "assigns": tuple(sorted(assigns.items())),
                    }
                )
    return out


def _search_period(
    pieces: list[dict],
    h: int,
    w: int,
    rp: int,
    cp: int,
    limit: int,
) -> list[np.ndarray]:
    candidates = [_candidate_placements(piece, h, w, rp, cp) for piece in pieces]
    if any(len(cands) == 0 for cands in candidates):
        return []

    n = len(pieces)
    occupied = np.zeros(h * w, dtype=bool)
    board = np.full(h * w, -1, dtype=int)
    tile = np.full(rp * cp, -1, dtype=int)
    used = [False] * n
    chosen_pos: list[tuple[int, int] | None] = [None] * n
    solutions: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    def fits(cand: dict) -> bool:
        for idx, color in cand["assigns"]:
            old = int(tile[int(idx)])
            if old != -1 and old != int(color):
                return False
        for cell, _color in cand["cells"]:
            if bool(occupied[int(cell)]):
                return False
        return True

    def place(cand: dict) -> tuple[list[int], list[int]]:
        changed_cells: list[int] = []
        changed_tile: list[int] = []
        for idx, color in cand["assigns"]:
            idx = int(idx)
            if int(tile[idx]) == -1:
                tile[idx] = int(color)
                changed_tile.append(idx)
        for cell, color in cand["cells"]:
            cell = int(cell)
            occupied[cell] = True
            board[cell] = int(color)
            changed_cells.append(cell)
        return changed_cells, changed_tile

    def unplace(changed_cells: list[int], changed_tile: list[int]) -> None:
        for cell in changed_cells:
            occupied[int(cell)] = False
            board[int(cell)] = -1
        for idx in changed_tile:
            tile[int(idx)] = -1

    def backtrack(done: int) -> None:
        if len(solutions) >= limit:
            return
        if done == n:
            if not bool(np.all(occupied)):
                return
            out = board.reshape(h, w).copy()
            key = ((h, w), out.tobytes())
            if key not in seen:
                seen.add(key)
                solutions.append(out)
            return

        best_i = -1
        best_valid: list[dict] | None = None
        for i in range(n):
            if used[i]:
                continue
            valid = [cand for cand in candidates[i] if fits(cand)]
            if not valid:
                return
            if best_valid is None or len(valid) < len(best_valid):
                best_i = i
                best_valid = valid
                if len(best_valid) == 1:
                    break

        assert best_valid is not None
        used[best_i] = True
        prev_pos = chosen_pos[best_i - 1] if best_i > 0 and pieces[best_i]["sig"] == pieces[best_i - 1]["sig"] else None
        for cand in best_valid:
            pos = cand["pos"]
            if prev_pos is not None and pos < prev_pos:
                continue
            changed_cells, changed_tile = place(cand)
            chosen_pos[best_i] = pos
            backtrack(done + 1)
            chosen_pos[best_i] = None
            unplace(changed_cells, changed_tile)
            if len(solutions) >= limit:
                break
        used[best_i] = False

    backtrack(0)
    return solutions


def _solve_outputs(grid: np.ndarray, limit: int = 2) -> list[np.ndarray]:
    pieces = _extract_pieces(grid)
    area = sum(int(piece["area"]) for piece in pieces)
    outputs: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for h, w in _factor_dims(area):
        r_periods = _proper_divisors(h)
        c_periods = _proper_divisors(w)
        if not r_periods or not c_periods:
            continue
        for rp in r_periods:
            for cp in c_periods:
                remaining = limit - len(outputs)
                if remaining <= 0:
                    return outputs
                for out in _search_period(pieces, h, w, rp, cp, remaining):
                    key = ((int(out.shape[0]), int(out.shape[1])), out.tobytes())
                    if key in seen:
                        continue
                    seen.add(key)
                    outputs.append(out)
                    if len(outputs) >= limit:
                        return outputs
    return outputs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered colored fragments into the unique row-and-column periodic rectangle.

    Input structure:
    - The background is the most common color.
    - Every non-background 4-connected component is one multicolor jigsaw fragment.
    - The fragments are separated in the input and no complete target rectangle is shown.
    - The total fragment area equals the area of the hidden output rectangle.

    Rule:
    - Translate every fragment without rotation or reflection into one rectangle so the fragments
      do not overlap and their colored cells fill the rectangle exactly.
    - Among all factor-pair rectangle sizes, keep only assemblies whose completed rectangle has
      a strict smaller row period and a strict smaller column period.
    - Return the assembled rectangle; the valid periodic assembled image must be unique.

    Key insight:
    - Edge shape alone is underconstrained, but requiring all translated fragments to agree on one
      repeated row/column tile fixes the global jigsaw assembly.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"
    outputs = _solve_outputs(g, limit=2)
    assert len(outputs) == 1, "periodic jigsaw assembly must determine one output"
    return outputs[0].astype(int, copy=True)


# --------------- generator ---------------

_GEN_RETRIES = 500
_TILE_RETRIES = 80
_PARTITION_RETRIES = 120
_SCATTER_RETRIES = 120


def _random_tile(rp: int, cp: int) -> np.ndarray | None:
    for _ in range(_TILE_RETRIES):
        color_count = random.randint(3, min(5, len(COLORS)))
        palette = random.sample(COLORS, color_count)
        tile = np.array(random.choices(palette, k=rp * cp), dtype=int).reshape(rp, cp)
        for color in palette:
            tile[random.randrange(rp), random.randrange(cp)] = int(color)
        if set(map(int, np.unique(tile))) != set(map(int, palette)):
            continue
        if _minimal_axis_period(tile, 0) != rp or _minimal_axis_period(tile, 1) != cp:
            continue
        if any(len(set(map(int, tile[r, :]))) == 1 for r in range(rp)):
            continue
        if any(len(set(map(int, tile[:, c]))) == 1 for c in range(cp)):
            continue
        return tile
    return None


def _is_connected_cells(cells: list[Coord]) -> bool:
    if not cells:
        return False
    s = set(cells)
    q: deque[Coord] = deque([cells[0]])
    seen = {cells[0]}
    while q:
        r, c = q.popleft()
        for nb in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nb in s and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(s)


def _random_partition(h: int, w: int, n: int) -> list[list[Coord]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    seeds = random.sample(cells, n)
    labels = np.full((h, w), -1, dtype=int)
    frontiers: list[set[Coord]] = [set() for _ in range(n)]
    sizes = [0] * n

    def add_frontier(label: int, r: int, c: int) -> None:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(labels[nr, nc]) == -1:
                frontiers[label].add((nr, nc))

    for label, (r, c) in enumerate(seeds):
        labels[r, c] = label
        sizes[label] = 1
    for label, (r, c) in enumerate(seeds):
        add_frontier(label, r, c)

    remaining = h * w - n
    soft = max(4, int(round(h * w / n * 1.35)))
    while remaining > 0:
        available = [i for i, frontier in enumerate(frontiers) if frontier]
        if not available:
            return None
        weights = [max(1, soft - sizes[i]) for i in available]
        label = random.choices(available, weights=weights, k=1)[0]
        stale_guard = 0
        while frontiers[label]:
            cell = random.choice(tuple(frontiers[label]))
            frontiers[label].remove(cell)
            r, c = cell
            if int(labels[r, c]) == -1:
                break
            stale_guard += 1
            if stale_guard > h * w:
                return None
        else:
            continue
        labels[r, c] = label
        sizes[label] += 1
        remaining -= 1
        add_frontier(label, r, c)

    parts = [[(int(r), int(c)) for r, c in np.argwhere(labels == label)] for label in range(n)]
    if not all(_is_connected_cells(part) for part in parts):
        return None
    return parts


def _valid_parts(pattern: np.ndarray, parts: list[list[Coord]], rp: int, cp: int) -> bool:
    areas = [len(part) for part in parts]
    if min(areas) < 4 or max(areas) > 14:
        return False
    sigs = set()
    both_residue_spans = 0
    touches_outer_frame = 0
    h, w = pattern.shape
    for part in parts:
        arr = np.array(part, dtype=int)
        r0, c0, r1, c1 = _bbox_coords(arr)
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if bh < 2 or bw < 2:
            return False
        if len(part) == bh * bw:
            return False
        colors = {int(pattern[r, c]) for r, c in part}
        if len(colors) < 2:
            return False
        rel = tuple(sorted((r - r0, c - c0, int(pattern[r, c])) for r, c in part))
        if rel in sigs:
            return False
        sigs.add(rel)
        if len({r % rp for r, _c in part}) >= 2 and len({c % cp for _r, c in part}) >= 2:
            both_residue_spans += 1
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in part):
            touches_outer_frame += 1
    return both_residue_spans >= max(2, len(parts) // 2) and touches_outer_frame >= 3


def _patch_from_part(pattern: np.ndarray, part: list[Coord]) -> np.ndarray:
    arr = np.array(part, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(arr)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BACKGROUND, dtype=int)
    for r, c in part:
        patch[int(r) - r0, int(c) - c0] = int(pattern[int(r), int(c)])
    return patch


def _can_scatter_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = canvas.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    for dr, dc in np.argwhere(patch != BACKGROUND):
        r = r0 + int(dr)
        c = c0 + int(dc)
        if int(canvas[r, c]) != BACKGROUND:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if int(canvas[rr, cc]) != BACKGROUND:
                    return False
    return True


def _place_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    for dr, dc in np.argwhere(patch != BACKGROUND):
        canvas[r0 + int(dr), c0 + int(dc)] = int(patch[int(dr), int(dc)])


def _scatter_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    ordered = sorted(patches, key=lambda p: (-(p.shape[0] * p.shape[1]), -p.shape[0], -p.shape[1]))
    for _ in range(_SCATTER_RETRIES):
        random.shuffle(ordered)
        min_h = max(12, max(p.shape[0] for p in ordered) + 4)
        min_w = max(12, max(p.shape[1] for p in ordered) + 4)
        h = random.randint(min_h, 30)
        w = random.randint(min_w, 30)
        canvas = np.full((h, w), BACKGROUND, dtype=int)
        ok = True
        for patch in ordered:
            ph, pw = patch.shape
            candidates = [
                (r, c)
                for r in range(1, h - ph)
                for c in range(1, w - pw)
                if _can_scatter_place(canvas, patch, r, c)
            ]
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            _place_patch(canvas, patch, r0, c0)
        if not ok:
            continue
        coords = np.argwhere(canvas != BACKGROUND)
        r0, c0, r1, c1 = _bbox_coords(coords)
        top = random.randint(0, min(2, r0))
        left = random.randint(0, min(2, c0))
        bottom = random.randint(0, min(2, h - 1 - r1))
        right = random.randint(0, min(2, w - 1 - c1))
        cropped = canvas[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1].copy()
        if max(cropped.shape) <= 30 and int(np.count_nonzero(cropped != BACKGROUND)) < cropped.size // 2:
            return cropped
    return None


def _gen_attempt() -> np.ndarray | None:
    rp = random.choice([2, 3])
    cp = random.choice([2, 3, 4])
    reps_r = random.choice([2, 3])
    reps_c = random.choice([2, 3])
    h = rp * reps_r
    w = cp * reps_c
    if h * w > 72:
        return None

    tile = _random_tile(rp, cp)
    if tile is None:
        return None
    pattern = np.tile(tile, (reps_r, reps_c))
    assert _has_axis_period(pattern, rp, 0)
    assert _has_axis_period(pattern, cp, 1)
    assert _minimal_axis_period(pattern, 0) == rp
    assert _minimal_axis_period(pattern, 1) == cp

    max_piece_count = min(9, (h * w) // 4)
    min_piece_count = min(5, max_piece_count)
    if max_piece_count < 4:
        return None
    piece_count = random.randint(min_piece_count, max_piece_count)

    parts = None
    for _ in range(_PARTITION_RETRIES):
        candidate = _random_partition(h, w, piece_count)
        if candidate is not None and _valid_parts(pattern, candidate, rp, cp):
            parts = candidate
            break
    if parts is None:
        return None

    patches = [_patch_from_part(pattern, part) for part in parts]
    inp = _scatter_patches(patches)
    if inp is None:
        return None

    try:
        outputs = _solve_outputs(inp, limit=2)
    except AssertionError:
        return None
    if len(outputs) != 1 or not np.array_equal(outputs[0], pattern):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid periodic jigsaw input after {_GEN_RETRIES} attempts")
