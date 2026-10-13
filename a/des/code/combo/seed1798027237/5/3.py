import random
from typing import Iterable

import numpy as np


BG = 0
AZURE = 8
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
ORIENTATIONS = ("left", "right", "up", "down")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _merge_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_collector_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in _components_of_color(grid, AZURE):
        r0, c0, r1, c1 = _bbox(comp)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh < 3 or ww < 3:
            continue
        pts = {tuple(map(int, xy)) for xy in comp}
        if len(pts) != 2 * hh + 2 * ww - 4:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            if (r, c0) not in pts or (r, c1) not in pts:
                ok = False
                break
        if not ok:
            continue
        for c in range(c0, c1 + 1):
            if (r0, c) not in pts or (r1, c) not in pts:
                ok = False
                break
        if ok:
            candidates.append((hh * ww, (r0, c0, r1, c1)))
    assert len(candidates) == 1, "expected exactly one azure collection frame"
    return candidates[0][1]


def _extract_holes(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((sub == color) | (sub == BG)), "mold bbox must contain only its color and black holes"
    assert np.all(sub[0, :] == color)
    assert np.all(sub[-1, :] == color)
    assert np.all(sub[:, 0] == color)
    assert np.all(sub[:, -1] == color)
    holes = np.argwhere(sub == BG)
    assert holes.size > 0, "mold must contain at least one internal black hole"
    return [tuple(map(int, rc)) for rc in holes]


def _project_holes(
    rect_bbox: tuple[int, int, int, int],
    mold_bbox: tuple[int, int, int, int],
    holes: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    rr0, rc0, rr1, rc1 = rect_bbox
    mr0, mc0, mr1, mc1 = mold_bbox
    mh = mr1 - mr0 + 1
    mw = mc1 - mc0 + 1

    row_overlap = max(rr0, mr0) <= min(rr1, mr1)
    col_overlap = max(rc0, mc0) <= min(rc1, mc1)
    left = row_overlap and mc1 < rc0
    right = row_overlap and mc0 > rc1
    up = col_overlap and mr1 < rr0
    down = col_overlap and mr0 > rr1
    assert sum((left, right, up, down)) == 1, "mold must align with its rectangle in exactly one direction"

    projected = []
    for hr, hc in holes:
        if left:
            pr = mr0 + hr
            pc = rc0 + (mw - 1 - hc)
        elif right:
            pr = mr0 + hr
            pc = rc1 - hc
        elif up:
            pr = rr0 + (mh - 1 - hr)
            pc = mc0 + hc
        else:
            pr = rr1 - hr
            pc = mc0 + hc
        assert rr0 <= pr <= rr1 and rc0 <= pc <= rc1, "projected hole must land inside the rectangle"
        projected.append((pr, pc))
    return projected


def _fragment_patch(cells: list[tuple[int, int]], color: int) -> np.ndarray:
    coords = np.array(cells, dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = color
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    collector_bbox = _find_collector_bbox(g)
    cr0, cc0, cr1, cc1 = collector_bbox
    collector_inner = (slice(cr0 + 1, cr1), slice(cc0 + 1, cc1))

    out = g.copy()
    out[collector_inner] = BG

    pairs = []
    for color in sorted(int(x) for x in np.unique(g) if int(x) not in (BG, AZURE)):
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each non-collector color must form one rectangle and one mold"

        solids = []
        molds = []
        for comp in comps:
            bbox = _bbox(comp)
            area = int(comp.shape[0])
            bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
            if area == bbox_area:
                solids.append((bbox, comp))
            else:
                molds.append((bbox, comp))

        assert len(solids) == 1 and len(molds) == 1, "color pairing must be unique"
        rect_bbox, rect_comp = solids[0]
        mold_bbox, mold_comp = molds[0]
        assert rect_comp.shape[0] > mold_comp.shape[0], "paired rectangle should be larger than its mold"

        holes = _extract_holes(g, mold_bbox, color)
        removed_cells = _project_holes(rect_bbox, mold_bbox, holes)
        pairs.append(
            {
                "color": color,
                "rect_bbox": rect_bbox,
                "mold_comp": mold_comp,
                "removed_cells": removed_cells,
                "fragment": _fragment_patch(removed_cells, color),
            }
        )

    pairs.sort(key=lambda item: (item["rect_bbox"][0], item["rect_bbox"][1]))

    for item in pairs:
        for r, c in item["mold_comp"]:
            out[int(r), int(c)] = BG
        for r, c in item["removed_cells"]:
            out[r, c] = BG

    cursor_r = cr0 + 1
    left_c = cc0 + 1
    inner_h = cr1 - cr0 - 1
    inner_w = cc1 - cc0 - 1
    assert inner_h > 0 and inner_w > 0
    for idx, item in enumerate(pairs):
        frag = item["fragment"]
        fh, fw = frag.shape
        assert fw <= inner_w, "fragment must fit inside the collection area"
        assert cursor_r + fh <= cr1, "fragments must fit vertically inside the collection area"
        target = out[cursor_r : cursor_r + fh, left_c : left_c + fw]
        target[frag != BG] = frag[frag != BG]
        cursor_r += fh
        if idx != len(pairs) - 1:
            cursor_r += 1

    return out


def _random_holes(mh: int, mw: int, rng: random.Random) -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(1, mh - 1) for c in range(1, mw - 1)]
    count = rng.randint(1, len(interior))
    chosen = {rng.choice(interior)}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for nr, nc in _neighbors4(r, c):
                if 1 <= nr < mh - 1 and 1 <= nc < mw - 1 and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        remaining = [cell for cell in interior if cell not in chosen]
        if frontier and rng.random() < 0.75:
            chosen.add(rng.choice(frontier))
        else:
            chosen.add(rng.choice(remaining))
    return sorted(chosen)


def _spec_fragment_shape(
    holes: list[tuple[int, int]],
    orientation: str,
    rect_h: int,
    rect_w: int,
    mold_h: int,
    mold_w: int,
) -> tuple[int, int]:
    rect_bbox = (10, 20, 10 + rect_h - 1, 20 + rect_w - 1)
    if orientation == "left":
        mold_bbox = (10, 13, 10 + mold_h - 1, 13 + mold_w - 1)
    elif orientation == "right":
        mold_bbox = (10, 31, 10 + mold_h - 1, 31 + mold_w - 1)
    elif orientation == "up":
        mold_bbox = (3, 20, 3 + mold_h - 1, 20 + mold_w - 1)
    else:
        mold_bbox = (21, 20, 21 + mold_h - 1, 20 + mold_w - 1)
    projected = _project_holes(rect_bbox, mold_bbox, holes)
    frag = _fragment_patch(projected, 1)
    return frag.shape


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0 = max(0, bbox[0] - pad)
    c0 = max(0, bbox[1] - pad)
    r1 = min(h - 1, bbox[2] + pad)
    c1 = min(w - 1, bbox[3] + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _bbox_clear(mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = mask.shape
    r0 = max(0, bbox[0] - pad)
    c0 = max(0, bbox[1] - pad)
    r1 = min(h - 1, bbox[2] + pad)
    c1 = min(w - 1, bbox[3] + pad)
    return not np.any(mask[r0 : r1 + 1, c0 : c1 + 1])


def _sample_pair_bboxes(
    grid_h: int,
    grid_w: int,
    rect_h: int,
    rect_w: int,
    mold_h: int,
    mold_w: int,
    orientation: str,
    gap: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if orientation == "left":
        if rect_w + mold_w + gap > grid_w or rect_h > grid_h:
            return None
        rect_r0 = rng.randint(0, grid_h - rect_h)
        rect_c0 = rng.randint(mold_w + gap, grid_w - rect_w)
        offset = rng.randint(0, rect_h - mold_h)
        mold_r0 = rect_r0 + offset
        mold_c0 = rect_c0 - gap - mold_w
    elif orientation == "right":
        if rect_w + mold_w + gap > grid_w or rect_h > grid_h:
            return None
        rect_r0 = rng.randint(0, grid_h - rect_h)
        rect_c0 = rng.randint(0, grid_w - rect_w - gap - mold_w)
        offset = rng.randint(0, rect_h - mold_h)
        mold_r0 = rect_r0 + offset
        mold_c0 = rect_c0 + rect_w + gap
    elif orientation == "up":
        if rect_h + mold_h + gap > grid_h or rect_w > grid_w:
            return None
        rect_r0 = rng.randint(mold_h + gap, grid_h - rect_h)
        rect_c0 = rng.randint(0, grid_w - rect_w)
        offset = rng.randint(0, rect_w - mold_w)
        mold_r0 = rect_r0 - gap - mold_h
        mold_c0 = rect_c0 + offset
    else:
        if rect_h + mold_h + gap > grid_h or rect_w > grid_w:
            return None
        rect_r0 = rng.randint(0, grid_h - rect_h - gap - mold_h)
        rect_c0 = rng.randint(0, grid_w - rect_w)
        offset = rng.randint(0, rect_w - mold_w)
        mold_r0 = rect_r0 + rect_h + gap
        mold_c0 = rect_c0 + offset

    rect_bbox = (rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c0 + rect_w - 1)
    mold_bbox = (mold_r0, mold_c0, mold_r0 + mold_h - 1, mold_c0 + mold_w - 1)
    return rect_bbox, mold_bbox


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    colors = rng.sample(PAIR_COLORS, pair_count)
    specs = []
    frag_heights = []
    frag_widths = []

    for color in colors:
        rect_h = rng.randint(5, 9)
        rect_w = rng.randint(5, 9)
        mold_h = rng.randint(3, min(5, rect_h))
        mold_w = rng.randint(3, min(5, rect_w))
        orientation = rng.choice(ORIENTATIONS)
        gap = rng.randint(1, 3)
        holes = _random_holes(mold_h, mold_w, rng)
        frag_h, frag_w = _spec_fragment_shape(holes, orientation, rect_h, rect_w, mold_h, mold_w)
        frag_heights.append(frag_h)
        frag_widths.append(frag_w)
        specs.append(
            {
                "color": color,
                "rect_h": rect_h,
                "rect_w": rect_w,
                "mold_h": mold_h,
                "mold_w": mold_w,
                "orientation": orientation,
                "gap": gap,
                "holes": holes,
            }
        )

    inner_h = sum(frag_heights) + (pair_count - 1) + rng.randint(1, 3)
    inner_w = max(frag_widths) + rng.randint(1, 3)
    collector_h = inner_h + 2
    collector_w = inner_w + 2
    min_h = max(collector_h + 5, 20)
    min_w = max(collector_w + 12, 24)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    collector_bbox = (grid_h - collector_h, 0, grid_h - 1, collector_w - 1)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_frame(grid, collector_bbox, AZURE)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _reserve(occupied, collector_bbox, pad=1)

    used_rect_tops: set[int] = set()
    specs.sort(
        key=lambda item: (
            max(item["rect_h"], item["mold_h"]),
            max(item["rect_w"], item["mold_w"]),
        ),
        reverse=True,
    )

    for spec in specs:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            sampled = _sample_pair_bboxes(
                grid_h,
                grid_w,
                spec["rect_h"],
                spec["rect_w"],
                spec["mold_h"],
                spec["mold_w"],
                spec["orientation"],
                spec["gap"],
                rng,
            )
            if sampled is None:
                return None
            rect_bbox, mold_bbox = sampled
            if rect_bbox[0] in used_rect_tops:
                continue
            pair_bbox = _merge_bbox(rect_bbox, mold_bbox)
            if not _bbox_clear(occupied, pair_bbox, pad=1):
                continue

            rr0, rc0, rr1, rc1 = rect_bbox
            mr0, mc0, mr1, mc1 = mold_bbox
            color = spec["color"]
            grid[rr0 : rr1 + 1, rc0 : rc1 + 1] = color
            grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = color
            for hr, hc in spec["holes"]:
                grid[mr0 + hr, mc0 + hc] = BG

            _reserve(occupied, pair_bbox, pad=1)
            spec["rect_bbox"] = rect_bbox
            spec["mold_bbox"] = mold_bbox
            used_rect_tops.add(rect_bbox[0])
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
