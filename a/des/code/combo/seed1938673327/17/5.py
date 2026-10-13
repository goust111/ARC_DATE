from __future__ import annotations

from collections import Counter
import random

import numpy as np


MARKER_COLOR = 5
QUADRANTS = ("tl", "tr", "bl", "br")
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 20
GRID_MAX = 28
OBJECT_COUNT_MIN = 4
OBJECT_COUNT_MAX = 6
BBOX_MIN = 4
BBOX_MAX = 8
AREA_RATIO_MIN = 0.55
AREA_RATIO_MAX = 0.82
PADDING = 1
SHAPE_ATTEMPTS = 160
GEN_ATTEMPTS = 400

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _quadrant_of(local_r: int, local_c: int, h: int, w: int) -> str | None:
    if 2 * local_r == h - 1 or 2 * local_c == w - 1:
        return None
    vertical = "t" if 2 * local_r < h - 1 else "b"
    horizontal = "l" if 2 * local_c < w - 1 else "r"
    return vertical + horizontal


def _fold_index(idx: int, size: int, towards_high: bool) -> int:
    mirror = size - 1 - idx
    return max(idx, mirror) if towards_high else min(idx, mirror)


def _fold_mask(mask: np.ndarray, marker_quadrant: str) -> np.ndarray:
    h, w = mask.shape
    fold_down = marker_quadrant[0] == "t"
    fold_right = marker_quadrant[1] == "l"
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        rr = _fold_index(int(r), h, fold_down)
        cc = _fold_index(int(c), w, fold_right)
        out[rr, cc] = True
    return out


def _marker_info(values: list[int]) -> tuple[int, int]:
    counts = Counter(map(int, values))
    if MARKER_COLOR in counts and counts[MARKER_COLOR] == 1 and len(counts) >= 2:
        marker_color = MARKER_COLOR
    else:
        singleton_colors = [color for color, count in counts.items() if count == 1]
        assert len(singleton_colors) == 1, "expected exactly one marker-colored pixel"
        marker_color = int(singleton_colors[0])
    object_colors = [(count, color) for color, count in counts.items() if int(color) != int(marker_color)]
    assert object_colors, "component must contain non-marker pixels"
    _count, object_color = max(object_colors)
    return int(marker_color), int(object_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold each marked object into the corner opposite its marker's bbox quadrant.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = np.full_like(g, bg)

    for comp in _components(g != bg):
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        h = r1 - r0 + 1
        w = c1 - c0 + 1

        values = [int(g[int(r), int(c)]) for r, c in comp]
        marker_color, object_color = _marker_info(values)
        marker_cells = [(int(r), int(c)) for r, c in comp if int(g[int(r), int(c)]) == int(marker_color)]
        assert len(marker_cells) == 1, "expected one marker cell per object"
        mr, mc = marker_cells[0]
        marker_quadrant = _quadrant_of(mr - r0, mc - c0, h, w)
        assert marker_quadrant is not None, "marker cannot lie on the bbox centerline"

        mask = np.zeros((h, w), dtype=bool)
        mask[rows - r0, cols - c0] = True
        folded = _fold_mask(mask, marker_quadrant)
        out[r0 : r1 + 1, c0 : c1 + 1][folded] = object_color

    return out


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _marker_candidates(mask: np.ndarray, quadrant: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if rr in (0, h - 1) or cc in (0, w - 1):
            continue
        if _quadrant_of(rr, cc, h, w) != quadrant:
            continue
        degree = 0
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                degree += 1
        if degree >= 2:
            out.append((rr, cc))
    return out


def _target_bounds(h: int, w: int, marker_quadrant: str) -> tuple[int, int, int, int]:
    th = (h + 1) // 2
    tw = (w + 1) // 2
    r0 = h - th if marker_quadrant[0] == "t" else 0
    c0 = w - tw if marker_quadrant[1] == "l" else 0
    return r0, c0, th, tw


def _interesting_shape(mask: np.ndarray, quadrant: str) -> bool:
    r0, c0, th, tw = _target_bounds(mask.shape[0], mask.shape[1], quadrant)
    inside_target = np.zeros_like(mask, dtype=bool)
    inside_target[r0 : r0 + th, c0 : c0 + tw] = True
    return bool(np.any(mask & ~inside_target))


def _generate_shape(h: int, w: int, quadrant: str) -> tuple[np.ndarray, tuple[int, int]] | None:
    total = h * w
    lo = max(h + w - 1, int(np.ceil(total * AREA_RATIO_MIN)))
    hi = min(total - 1, int(np.floor(total * AREA_RATIO_MAX)))
    if lo > hi:
        return None

    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(SHAPE_ATTEMPTS):
        target_area = random.randint(lo, hi)
        mask = np.ones((h, w), dtype=bool)
        order = cells[:]
        random.shuffle(order)
        for r, c in order:
            if int(mask.sum()) <= target_area:
                break
            mask[r, c] = False
            if not _spans_bbox(mask) or not _is_connected(mask):
                mask[r, c] = True

        if int(mask.sum()) < lo:
            continue
        if not _interesting_shape(mask, quadrant):
            continue
        marker_candidates = _marker_candidates(mask, quadrant)
        if not marker_candidates:
            continue
        return mask, random.choice(marker_candidates)
    return None


def _quadrant_sequence(n_objects: int) -> list[str]:
    quads = list(QUADRANTS)
    random.shuffle(quads)
    extras = [random.choice(QUADRANTS) for _ in range(max(0, n_objects - len(QUADRANTS)))]
    quads.extend(extras)
    random.shuffle(quads)
    return quads[:n_objects]


def _place_specs(specs: list[dict], h: int, w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["mask"].size, reverse=True)
    for idx in order:
        mask = specs[idx]["mask"]
        box_h, box_w = mask.shape
        choices: list[tuple[int, int]] = []
        for r0 in range(h - box_h + 1):
            for c0 in range(w - box_w + 1):
                rs = max(0, r0 - PADDING)
                re = min(h, r0 + box_h + PADDING)
                cs = max(0, c0 - PADDING)
                ce = min(w, c0 + box_w + PADDING)
                if not occupied[rs:re, cs:ce].any():
                    choices.append((r0, c0))
        if not choices:
            return None
        r0, c0 = random.choice(choices)
        placements[idx] = (r0, c0)
        occupied[r0 : r0 + box_h, c0 : c0 + box_w] = True
    return placements


def _build_grid() -> np.ndarray | None:
    n_objects = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    h = random.randint(GRID_MIN, GRID_MAX) + max(0, n_objects - 4)
    w = random.randint(GRID_MIN, GRID_MAX)
    colors = random.sample(PALETTE, n_objects)
    quadrants = _quadrant_sequence(n_objects)

    specs: list[dict] = []
    for color, quadrant in zip(colors, quadrants):
        box_h = random.randint(BBOX_MIN, BBOX_MAX)
        box_w = random.randint(BBOX_MIN, BBOX_MAX)
        shape = _generate_shape(box_h, box_w, quadrant)
        if shape is None:
            return None
        mask, marker = shape
        specs.append({"color": int(color), "quadrant": quadrant, "mask": mask, "marker": marker})

    placements = _place_specs(specs, h, w)
    if placements is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec, (r0, c0) in zip(specs, placements):
        mask = spec["mask"]
        color = int(spec["color"])
        marker_r, marker_c = spec["marker"]
        box_h, box_w = mask.shape
        patch = np.zeros((box_h, box_w), dtype=int)
        patch[mask] = color
        patch[marker_r, marker_c] = MARKER_COLOR
        block = grid[r0 : r0 + box_h, c0 : c0 + box_w]
        assert not np.any((patch != 0) & (block != 0))
        block[patch != 0] = patch[patch != 0]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_grid()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(grid, out):
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
