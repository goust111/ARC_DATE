from __future__ import annotations

from collections import deque
import random

import numpy as np


RED = 2
MAGENTA = 6
TEAL = 8

# Input colors avoid the three output-only recolors for visual clarity.
INPUT_COLORS = tuple(c for c in range(10) if c not in {RED, MAGENTA, TEAL})

# --------------- shared helpers ---------------


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


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
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros((h, w), dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _interior_holes_in_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    holes: list[np.ndarray] = []
    for comp in _components(~mask):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == h - 1)
            | (comp[:, 1] == w - 1)
        )
        if not bool(touches_edge):
            holes.append(comp)
    return holes


def _interior_holes_in_grid(grid: np.ndarray, bg: int, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    sub_bg = grid[r0 : r1 + 1, c0 : c1 + 1] == bg
    holes: list[np.ndarray] = []
    for comp in _components(sub_bg):
        touches_edge = np.any(
            (comp[:, 0] == 0)
            | (comp[:, 1] == 0)
            | (comp[:, 0] == sub_bg.shape[0] - 1)
            | (comp[:, 1] == sub_bg.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = comp.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _reserve_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    reserve = np.zeros((h + 2, w + 2), dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            reserve[rows + dr + 1, cols + dc + 1] = True
    return reserve


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


# --------------- solver ---------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a red 8-neighborhood outline around every foreground object.

    Objects that enclose one or more background holes are recolored to teal, their holes
    are filled magenta, and the outline is drawn around the resulting solid footprint.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()

    for coords in _components(g != bg):
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes_in_grid(g, bg, coords)

        solid = mask.copy()
        if holes:
            out[mask] = TEAL
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = MAGENTA
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = RED

    return out


# --------------- generator ---------------

GRID_H_RANGE = (17, 26)
GRID_W_RANGE = (17, 26)
OBJECT_COUNT_RANGE = (2, 4)
MIN_BG_FRACTION = 0.5
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 200

SOLID_RETRIES = 120
SOLID_BOX_RANGE = (3, 7)
SOLID_MIN_AREA = 4
SOLID_MAX_AREA = 15

HOLLOW_RETRIES = 120
HOLLOW_OUTER_H_RANGE = (5, 9)
HOLLOW_OUTER_W_RANGE = (5, 9)
HOLLOW_HOLE_H_RANGE = (1, 3)
HOLLOW_HOLE_W_RANGE = (1, 3)
HOLLOW_SECOND_HOLE_PROB = 0.35
HOLLOW_PROTRUSION_PROB = 0.65
HOLLOW_PROTRUSION_SPAN_RANGE = (2, 5)
HOLLOW_PROTRUSION_DEPTH_RANGE = (1, 3)


def _make_solid_shape() -> np.ndarray | None:
    for _ in range(SOLID_RETRIES):
        h = random.randint(*SOLID_BOX_RANGE)
        w = random.randint(*SOLID_BOX_RANGE)
        max_area = min(SOLID_MAX_AREA, h * w - 1)
        if max_area < SOLID_MIN_AREA:
            continue
        target = random.randint(SOLID_MIN_AREA, max_area)

        mask = np.zeros((h, w), dtype=bool)
        sr = random.randrange(h)
        sc = random.randrange(w)
        mask[sr, sc] = True
        frontier = {(nr, nc) for nr, nc in _neighbors4(sr, sc) if 0 <= nr < h and 0 <= nc < w}

        while frontier and int(mask.sum()) < target:
            fr, fc = random.choice(tuple(frontier))
            frontier.remove((fr, fc))
            mask[fr, fc] = True
            for nr, nc in _neighbors4(fr, fc):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
            if int(mask.sum()) >= SOLID_MIN_AREA and random.random() < 0.12:
                break

        cropped = _crop_mask(mask)
        if cropped is None:
            continue
        if int(cropped.sum()) < SOLID_MIN_AREA:
            continue
        if min(cropped.shape) < 2:
            continue
        if _interior_holes_in_mask(cropped):
            continue
        return cropped
    return None


def _try_carve_hole(mask: np.ndarray, hole_h: int, hole_w: int) -> bool:
    h, w = mask.shape
    if hole_h + 2 > h or hole_w + 2 > w:
        return False

    for _ in range(80):
        r0 = random.randint(1, h - hole_h - 1)
        c0 = random.randint(1, w - hole_w - 1)
        ring = mask[r0 - 1 : r0 + hole_h + 1, c0 - 1 : c0 + hole_w + 1]
        if ring.shape != (hole_h + 2, hole_w + 2):
            continue
        if not ring.all():
            continue
        mask[r0 : r0 + hole_h, c0 : c0 + hole_w] = False
        return True
    return False


def _add_protrusion(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    side = random.choice(["top", "bottom", "left", "right"])

    if side in {"top", "bottom"}:
        depth = random.randint(*HOLLOW_PROTRUSION_DEPTH_RANGE)
        span = random.randint(2, min(HOLLOW_PROTRUSION_SPAN_RANGE[1], w))
        start = random.randint(0, w - span)
        out = np.zeros((h + depth, w), dtype=bool)
        if side == "top":
            out[depth:, :] = mask
            out[:depth, start : start + span] = True
        else:
            out[:h, :] = mask
            out[h:, start : start + span] = True
    else:
        depth = random.randint(*HOLLOW_PROTRUSION_DEPTH_RANGE)
        span = random.randint(2, min(HOLLOW_PROTRUSION_SPAN_RANGE[1], h))
        start = random.randint(0, h - span)
        out = np.zeros((h, w + depth), dtype=bool)
        if side == "left":
            out[:, depth:] = mask
            out[start : start + span, :depth] = True
        else:
            out[:, :w] = mask
            out[start : start + span, w:] = True

    return out


def _make_hollow_shape() -> np.ndarray | None:
    for _ in range(HOLLOW_RETRIES):
        h = random.randint(*HOLLOW_OUTER_H_RANGE)
        w = random.randint(*HOLLOW_OUTER_W_RANGE)
        mask = np.ones((h, w), dtype=bool)

        hole_count = 1
        if h >= 7 and w >= 7 and random.random() < HOLLOW_SECOND_HOLE_PROB:
            hole_count = 2

        carved = 0
        for _ in range(hole_count):
            hole_h = random.randint(*HOLLOW_HOLE_H_RANGE)
            hole_w = random.randint(*HOLLOW_HOLE_W_RANGE)
            carved += int(_try_carve_hole(mask, hole_h, hole_w))
        if carved == 0:
            continue

        if random.random() < HOLLOW_PROTRUSION_PROB:
            mask = _add_protrusion(mask)
        if random.random() < HOLLOW_PROTRUSION_PROB * 0.45:
            mask = _add_protrusion(mask)

        cropped = _crop_mask(mask)
        if cropped is None:
            continue
        if len(_interior_holes_in_mask(cropped)) == 0:
            continue
        return cropped
    return None


def _place_shape(grid: np.ndarray, reserved: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    reserve = _reserve_mask(mask)
    if mh + 2 > h or mw + 2 > w:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = random.randint(1, h - mh - 1)
        left = random.randint(1, w - mw - 1)
        region = reserved[top - 1 : top + mh + 1, left - 1 : left + mw + 1]
        if np.any(region & reserve):
            continue
        sub = grid[top : top + mh, left : left + mw]
        if np.any(sub[mask] != grid[0, 0]):
            continue

        sub[mask] = color
        region |= reserve
        return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _most_common_color(grid)
    if np.count_nonzero(grid == bg) / grid.size <= MIN_BG_FRACTION:
        return False

    comps = _components(grid != bg)
    if len(comps) < 2:
        return False

    has_hole = False
    has_solid = False
    out = main(grid)

    for coords in comps:
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes_in_grid(grid, bg, coords)
        rows = coords[:, 0]
        cols = coords[:, 1]

        if holes:
            has_hole = True
            if not np.all(out[rows, cols] == TEAL):
                return False
            solid = mask.copy()
            for hole in holes:
                if not np.all(out[hole[:, 0], hole[:, 1]] == MAGENTA):
                    return False
                solid[hole[:, 0], hole[:, 1]] = True
        else:
            has_solid = True
            if not np.array_equal(out[rows, cols], grid[rows, cols]):
                return False
            solid = mask

        outline = _outline_mask(solid)
        if not np.any(out[outline] == RED):
            return False

    return has_hole and has_solid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(INPUT_COLORS)
    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    obj_count = random.randint(*OBJECT_COUNT_RANGE)
    hollow_flags = [random.random() < 0.5 for _ in range(obj_count)]
    if all(hollow_flags):
        hollow_flags[random.randrange(obj_count)] = False
    if not any(hollow_flags):
        hollow_flags[random.randrange(obj_count)] = True

    palette = [c for c in INPUT_COLORS if c != bg]
    random.shuffle(palette)
    colors = palette[:obj_count]

    for is_hollow, color in zip(hollow_flags, colors, strict=False):
        shape = _make_hollow_shape() if is_hollow else _make_solid_shape()
        if shape is None:
            return None
        shape = _random_transform(shape)
        if not _place_shape(grid, reserved, shape, color):
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
