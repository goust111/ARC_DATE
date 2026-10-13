import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR8 if connectivity == 8 else DIR4
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask, connectivity=4):
        touches_border = any(
            int(r) == 0 or int(r) == h - 1 or int(c) == 0 or int(c) == w - 1
            for r, c in comp
        )
        if not touches_border:
            holes += 1
    return int(holes)


def _anchor_side(anchor: tuple[int, int], h: int, w: int) -> str:
    r, c = anchor
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    assert len(sides) == 1, "anchor must lie on exactly one edge"
    return sides[0]


def _place_on_track(
    side: str, track: int, shape_h: int, shape_w: int, hole_count: int, grid_h: int, grid_w: int
) -> tuple[int, int]:
    if side == "top":
        top = 1 + hole_count
        left = track - shape_w // 2
    elif side == "bottom":
        top = grid_h - 1 - hole_count - shape_h
        left = track - shape_w // 2
    elif side == "left":
        top = track - shape_h // 2
        left = 1 + hole_count
    else:
        top = track - shape_h // 2
        left = grid_w - 1 - hole_count - shape_w
    assert 0 <= top and 0 <= left
    assert top + shape_h <= grid_h and left + shape_w <= grid_w
    return int(top), int(left)


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    rows, cols = np.where(mask)
    return {(int(top + r), int(left + c)) for r, c in zip(rows, cols)}


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grid_h, grid_w = g.shape
    bg = _bg_color(g)

    sprites = []
    clear_mask = np.zeros_like(g, dtype=bool)
    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        color_cells = np.argwhere(g == color)
        boundary_cells = [
            (int(r), int(c))
            for r, c in color_cells
            if int(r) in (0, grid_h - 1) or int(c) in (0, grid_w - 1)
        ]
        assert len(boundary_cells) == 1, "each color must have exactly one boundary anchor"
        anchor = boundary_cells[0]
        side = _anchor_side(anchor, grid_h, grid_w)

        sprite_mask = g == color
        sprite_mask[anchor] = False
        comps = _components(sprite_mask, connectivity=8)
        assert len(comps) == 1, "each anchor color must match exactly one interior sprite"
        comp = comps[0]
        rows = comp[:, 0]
        cols = comp[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = []
        for r, c in comp:
            rr = int(r - r0)
            cc = int(c - c0)
            local[rr, cc] = True
            rel.append((rr, cc))
            clear_mask[int(r), int(c)] = True

        hole_count = _count_holes(local)
        track = anchor[1] if side in ("top", "bottom") else anchor[0]
        top, left = _place_on_track(side, int(track), local.shape[0], local.shape[1], hole_count, grid_h, grid_w)
        sprites.append((color, rel, top, left))

    out = g.copy()
    out[clear_mask] = bg

    occupied = set()
    for color, rel, top, left in sprites:
        for rr, cc in rel:
            r = top + rr
            c = left + cc
            assert out[r, c] == bg, "moved sprites must land on empty background"
            assert (r, c) not in occupied, "moved sprites must not overlap"
            out[r, c] = color
            occupied.add((r, c))
    return out


GEN_ATTEMPTS = 400
GEN_GRID_H = (18, 26)
GEN_GRID_W = (18, 26)
GEN_OBJECT_COUNT = (2, 4)
GEN_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_HOLE_COUNTS = [0, 1, 2, 3, 4]


def _connected8(mask: np.ndarray) -> bool:
    comps = _components(mask, connectivity=8)
    return len(comps) == 1


def _random_symmetry(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _make_thin_shape(side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        height = random.randint(2, 5)
        return np.ones((height, 1), dtype=bool)
    width = random.randint(2, 5)
    return np.ones((1, width), dtype=bool)


def _make_irregular_shape(hole_count: int) -> np.ndarray:
    for _ in range(200):
        min_dim = 3 if hole_count == 0 else 4
        shape_h = random.randint(min_dim, 8)
        shape_w = random.randint(min_dim, 8)
        mask = np.ones((shape_h, shape_w), dtype=bool)

        if hole_count > 0:
            candidates = [(r, c) for r in range(1, shape_h - 1) for c in range(1, shape_w - 1)]
            random.shuffle(candidates)
            holes = []
            for cell in candidates:
                if all(abs(cell[0] - r) + abs(cell[1] - c) > 1 for r, c in holes):
                    holes.append(cell)
                    if len(holes) == hole_count:
                        break
            if len(holes) != hole_count:
                continue
            for r, c in holes:
                mask[r, c] = False

        removals = random.randint(0, max(1, (shape_h + shape_w) // 2))
        for _ in range(removals):
            pts = [(int(r), int(c)) for r, c in np.argwhere(mask)]
            random.shuffle(pts)
            for r, c in pts:
                trial = mask.copy()
                trial[r, c] = False
                if trial.sum() < max(3, hole_count * 4 + 1):
                    continue
                trial = _trim_mask(trial)
                if _connected8(trial) and _count_holes(trial) == hole_count:
                    mask = trial
                    break

        mask = _trim_mask(mask)
        if _connected8(mask) and _count_holes(mask) == hole_count:
            return mask
    raise RuntimeError("failed to generate a sprite mask")


def _generate_shape(side: str, hole_count: int) -> np.ndarray:
    if hole_count == 0 and random.random() < 0.35:
        return _make_thin_shape(side)
    return _random_symmetry(_make_irregular_shape(hole_count))


def _expanded(cells: set[tuple[int, int]], grid_h: int, grid_w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if 0 <= nr < grid_h and 0 <= nc < grid_w:
                    out.add((int(nr), int(nc)))
    return out


def _candidate_tracks(side: str, shape: np.ndarray, grid_h: int, grid_w: int) -> list[int]:
    shape_h, shape_w = shape.shape
    if side in ("top", "bottom"):
        lo = max(1, shape_w // 2)
        hi = min(grid_w - 2, grid_w - (shape_w - shape_w // 2))
    else:
        lo = max(1, shape_h // 2)
        hi = min(grid_h - 2, grid_h - (shape_h - shape_h // 2))
    if lo > hi:
        return []
    vals = list(range(int(lo), int(hi) + 1))
    random.shuffle(vals)
    return vals


def _plan_final_placements(
    shapes: list[np.ndarray], sides: list[str], holes: list[int], grid_h: int, grid_w: int
) -> list[dict] | None:
    planned = []
    used_anchors: dict[str, set[int]] = {"top": set(), "bottom": set(), "left": set(), "right": set()}
    blocked: set[tuple[int, int]] = set()

    order = list(range(len(shapes)))
    order.sort(key=lambda i: max(shapes[i].shape), reverse=True)
    for idx in order:
        shape = shapes[idx]
        side = sides[idx]
        hole_count = holes[idx]
        placed = None
        for track in _candidate_tracks(side, shape, grid_h, grid_w):
            if any(abs(track - other) <= 1 for other in used_anchors[side]):
                continue
            anchor = (
                (0, track)
                if side == "top"
                else (grid_h - 1, track)
                if side == "bottom"
                else (track, 0)
                if side == "left"
                else (track, grid_w - 1)
            )
            top, left = _place_on_track(side, track, shape.shape[0], shape.shape[1], hole_count, grid_h, grid_w)
            cells = _mask_cells(shape, top, left)
            if cells & blocked:
                continue
            placed = {
                "index": idx,
                "side": side,
                "track": int(track),
                "anchor": anchor,
                "top": int(top),
                "left": int(left),
                "cells": cells,
                "shape": shape,
            }
            break
        if placed is None:
            return None
        planned.append(placed)
        used_anchors[side].add(int(placed["track"]))
        blocked |= _expanded(placed["cells"], grid_h, grid_w, pad=1)

    planned.sort(key=lambda item: item["index"])
    return planned


def _random_origins(
    planned: list[dict], grid_h: int, grid_w: int
) -> list[tuple[int, int]] | None:
    blocked = set()
    for item in planned:
        blocked |= _expanded(item["cells"], grid_h, grid_w, pad=1)
        blocked.add(item["anchor"])

    origins: list[tuple[int, int]] = [(-1, -1)] * len(planned)
    order = list(range(len(planned)))
    order.sort(key=lambda i: max(planned[i]["shape"].shape), reverse=True)

    for idx in order:
        shape = planned[idx]["shape"]
        shape_h, shape_w = shape.shape
        if grid_h - shape_h - 1 <= 1 or grid_w - shape_w - 1 <= 1:
            return None
        candidates = [
            (top, left)
            for top in range(1, grid_h - shape_h)
            for left in range(1, grid_w - shape_w)
        ]
        random.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            if (top, left) == (planned[idx]["top"], planned[idx]["left"]):
                continue
            cells = _mask_cells(shape, top, left)
            if cells & blocked:
                continue
            chosen = (int(top), int(left))
            blocked |= _expanded(cells, grid_h, grid_w, pad=1)
            break
        if chosen is None:
            return None
        origins[idx] = chosen
    return origins


def _paint(mask: np.ndarray, top: int, left: int, grid: np.ndarray, color: int) -> None:
    for r, c in _mask_cells(mask, top, left):
        grid[r, c] = int(color)


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GEN_GRID_H)
    grid_w = random.randint(*GEN_GRID_W)
    obj_count = random.randint(*GEN_OBJECT_COUNT)
    colors = random.sample(GEN_COLORS, obj_count)

    base_sides = random.sample(["top", "bottom", "left", "right"], k=min(4, obj_count))
    while len(base_sides) < obj_count:
        base_sides.append(random.choice(["top", "bottom", "left", "right"]))
    random.shuffle(base_sides)

    hole_counts = [random.choice(GEN_HOLE_COUNTS) for _ in range(obj_count)]
    shapes = [_generate_shape(base_sides[i], hole_counts[i]) for i in range(obj_count)]

    planned = _plan_final_placements(shapes, base_sides, hole_counts, grid_h, grid_w)
    if planned is None:
        return None

    origins = _random_origins(planned, grid_h, grid_w)
    if origins is None:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)

    for color, item, origin in zip(colors, planned, origins):
        ar, ac = item["anchor"]
        inp[ar, ac] = int(color)
        expected[ar, ac] = int(color)
        _paint(item["shape"], origin[0], origin[1], inp, color)
        _paint(item["shape"], item["top"], item["left"], expected, color)

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
