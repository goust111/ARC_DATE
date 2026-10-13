from __future__ import annotations

from collections import Counter
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    TEAL = 8
    MAROON = 9

    ALL = tuple(range(10))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    cols = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        cols -= set(exclude)
    return sorted(cols)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _expand_tile(tile: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pr, pc = tile.shape
    h, w = shape
    rr = np.arange(h) % pr
    cc = np.arange(w) % pc
    return tile[rr[:, None], cc[None, :]]


def _all_periods(h: int, w: int) -> list[tuple[int, int]]:
    return sorted(
        [(pr, pc) for pr in range(1, h + 1) for pc in range(1, w + 1)],
        key=lambda rc: (rc[0] * rc[1], rc[0] + rc[1], rc[0], rc[1]),
    )


def _minimal_period(arr: np.ndarray) -> tuple[int, int]:
    h, w = arr.shape
    for pr, pc in _all_periods(h, w):
        tile = arr[:pr, :pc]
        if np.array_equal(_expand_tile(tile, (h, w)), arr):
            return pr, pc
    raise AssertionError("period search failed")


def _residue_visible(mismatch: np.ndarray, pr: int, pc: int) -> bool:
    seen = np.zeros((pr, pc), dtype=bool)
    for r, c in np.argwhere(~mismatch):
        seen[int(r) % pr, int(c) % pc] = True
    return bool(np.all(seen))


def _rectangles_from_mask(grid: np.ndarray, mask: np.ndarray) -> list[dict] | None:
    rects: list[dict] = []
    for coords in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        area = int(coords.shape[0])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if area != h * w:
            return None
        colors = set(map(int, grid[r0 : r1 + 1, c0 : c1 + 1].ravel()))
        if len(colors) != 1:
            return None
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "color": next(iter(colors)),
                "area": area,
            }
        )
    return rects


def _candidate_solution(grid: np.ndarray, pr: int, pc: int) -> dict | None:
    residue_modes = np.zeros((pr, pc), dtype=int)
    for rr in range(pr):
        for cc in range(pc):
            vals = list(map(int, grid[rr::pr, cc::pc].ravel()))
            counts = Counter(vals).most_common()
            if len(counts) >= 2 and counts[0][1] == counts[1][1]:
                return None
            residue_modes[rr, cc] = int(counts[0][0])

    pattern = _expand_tile(residue_modes, grid.shape)
    mismatch = grid != pattern
    if not np.any(mismatch):
        return None
    if not _residue_visible(mismatch, pr, pc):
        return None

    rects = _rectangles_from_mask(grid, mismatch)
    if not rects:
        return None

    palette = set(map(int, residue_modes.ravel()))
    if len(palette) < 2:
        return None

    recolors: list[tuple[tuple[int, int, int, int], int]] = []
    for rect in rects:
        if rect["color"] in palette:
            return None
        r0, c0, r1, c1 = rect["bbox"]
        hidden = set(map(int, pattern[r0 : r1 + 1, c0 : c1 + 1].ravel()))
        absent = palette - hidden
        if len(absent) != 1 or len(hidden) != len(palette) - 1:
            return None
        recolors.append((rect["bbox"], next(iter(absent))))

    return {"tile": residue_modes, "pattern": pattern, "recolors": recolors}


def _solve(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    for pr, pc in _all_periods(h, w):
        cand = _candidate_solution(g, pr, pc)
        if cand is not None:
            return cand
    raise AssertionError("no valid periodic-occluder interpretation found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each masking rectangle with the unique tile color absent from the pixels it hides.

    Input structure:
    - The full canvas is a repetition of a small row-column periodic tile.
    - One or more solid monochrome rectangles cover parts of that tiled pattern.
    - Rectangle colors do not belong to the tile palette, and uncovered cells still reveal the tile.

    Rule:
    - Infer the smallest periodic tile that explains the visible non-rectangle structure.
    - Reconstruct the tiled colors underneath every masking rectangle.
    - For each rectangle, collect the tile colors that would have appeared under it and find the one palette color that is entirely missing there.
    - Recolor the whole rectangle to that unique absent color, leaving every non-rectangle cell unchanged.

    Key insight:
    - The output color is decided by inversion of the hidden-color set: after reconstructing the periodic background, each occluder is painted by the only tile color it never covered.
    """
    g = np.asarray(grid, dtype=int)
    solution = _solve(g)
    out = g.copy()
    for (r0, c0, r1, c1), color in solution["recolors"]:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


_GEN_ATTEMPTS = 400
_GEN_TILE_H = (2, 5)
_GEN_TILE_W = (2, 5)
_GEN_GRID_H = (12, 24)
_GEN_GRID_W = (12, 24)
_GEN_PALETTE_SIZE = (3, 5)
_GEN_RECT_COUNT = (2, 4)
_GEN_RECT_SIDE = (2, 7)
_GEN_RECT_GAP = 1


def _random_tile(rng: random.Random, pr: int, pc: int, palette: list[int]) -> np.ndarray | None:
    for _ in range(_GEN_ATTEMPTS):
        tile = np.array(
            [[rng.choice(palette) for _ in range(pc)] for _ in range(pr)],
            dtype=int,
        )
        if set(map(int, tile.ravel())) != set(palette):
            continue
        if _minimal_period(tile) != (pr, pc):
            continue
        return tile
    return None


def _candidate_rectangles(base: np.ndarray, palette: set[int]) -> list[dict]:
    h, w = base.shape
    k = len(palette)
    rects: list[dict] = []
    for rh in range(_GEN_RECT_SIDE[0], min(_GEN_RECT_SIDE[1], h) + 1):
        for rw in range(_GEN_RECT_SIDE[0], min(_GEN_RECT_SIDE[1], w) + 1):
            if rh * rw < k - 1:
                continue
            for r0 in range(h - rh + 1):
                for c0 in range(w - rw + 1):
                    patch = base[r0 : r0 + rh, c0 : c0 + rw]
                    hidden = set(map(int, patch.ravel()))
                    absent = palette - hidden
                    if len(absent) != 1 or len(hidden) != k - 1:
                        continue
                    rects.append(
                        {
                            "bbox": (r0, c0, r0 + rh - 1, c0 + rw - 1),
                            "absent": next(iter(absent)),
                            "area": rh * rw,
                        }
                    )
    return rects


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0
    )


def _visible_after_selection(shape: tuple[int, int], rects: list[dict], pr: int, pc: int) -> bool:
    covered = np.zeros(shape, dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    return _residue_visible(covered, pr, pc)


def _select_rectangles(
    rng: random.Random,
    candidates: list[dict],
    shape: tuple[int, int],
    pr: int,
    pc: int,
) -> list[dict] | None:
    if not candidates:
        return None

    max_count = min(_GEN_RECT_COUNT[1], len(candidates))
    if max_count < _GEN_RECT_COUNT[0]:
        return None

    for _ in range(_GEN_ATTEMPTS):
        target = rng.randint(_GEN_RECT_COUNT[0], max_count)
        order = candidates[:]
        rng.shuffle(order)
        chosen: list[dict] = []
        used_absent: set[int] = set()

        for cand in order:
            if len(chosen) == target:
                break
            if any(_boxes_too_close(cand["bbox"], prev["bbox"], _GEN_RECT_GAP) for prev in chosen):
                continue
            if len(used_absent) < 2 and cand["absent"] in used_absent and len(candidates) > target:
                continue
            chosen.append(cand)
            used_absent.add(int(cand["absent"]))

        if len(chosen) != target:
            continue
        if len(used_absent) < min(2, target):
            continue
        if not _visible_after_selection(shape, chosen, pr, pc):
            continue
        return chosen
    return None


def _expected_output(inp: np.ndarray, rects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(rect["absent"])
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(_GEN_ATTEMPTS):
        pr = rng.randint(*_GEN_TILE_H)
        pc = rng.randint(*_GEN_TILE_W)
        h = rng.randint(max(pr * 3, _GEN_GRID_H[0]), _GEN_GRID_H[1])
        w = rng.randint(max(pc * 3, _GEN_GRID_W[0]), _GEN_GRID_W[1])

        max_palette = min(_GEN_PALETTE_SIZE[1], pr * pc, len(Color.ALL) - _GEN_RECT_COUNT[0])
        if max_palette < _GEN_PALETTE_SIZE[0]:
            continue
        palette_size = rng.randint(_GEN_PALETTE_SIZE[0], max_palette)
        palette = rng.sample(list(Color.ALL), palette_size)

        tile = _random_tile(rng, pr, pc, palette)
        if tile is None:
            continue

        base = _expand_tile(tile, (h, w))
        candidates = _candidate_rectangles(base, set(palette))
        rects = _select_rectangles(rng, candidates, (h, w), pr, pc)
        if rects is None:
            continue

        occluder_colors = rng.sample([c for c in Color.ALL if c not in palette], len(rects))
        inp = base.copy()
        for rect, occ_color in zip(rects, occluder_colors):
            r0, c0, r1, c1 = rect["bbox"]
            inp[r0 : r1 + 1, c0 : c1 + 1] = int(occ_color)

        expected = _expected_output(inp, rects)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue

        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
