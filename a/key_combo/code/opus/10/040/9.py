from __future__ import annotations

import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
L_CORNERS = {(0, 0), (0, 1), (1, 0), (1, 1)}
MISSING_CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]

GEN_ATTEMPTS = 400
GEN_HEIGHTS = (15, 16, 17, 18)
GEN_WIDTHS = (15, 16, 17, 18)
GEN_RECT_SIZES = [(h, w) for h in range(2, 5) for w in range(3, 5)] + [
    (h, w) for h in range(3, 5) for w in range(2, 5)
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components_of_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components_of_mask(grid == ORANGE):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if len(comp) != h * w:
            continue
        rects.append(
            {
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert rects, "expected orange rectangular occluders"
    return rects


def _tri_cells(r0: int, c0: int, missing: tuple[int, int]) -> set[tuple[int, int]]:
    return {
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    }


def _detect_triangles(grid: np.ndarray) -> list[dict]:
    tris: list[dict] = []
    for color in (GREY, RED):
        for comp in _components_of_mask(grid == color):
            assert len(comp) == 3, "triomino must have area 3"
            r0, c0, r1, c1 = _bbox(comp)
            assert r1 == r0 + 1 and c1 == c0 + 1, "triomino must occupy a 2x2 box"
            rel = {(r - r0, c - c0) for r, c in comp}
            missing = list(L_CORNERS - rel)
            assert len(missing) == 1, "expected one missing corner"
            mr, mc = missing[0]
            tris.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "missing": (mr, mc),
                    "missing_global": (r0 + mr, c0 + mc),
                    "dir": (-1 if mr == 0 else 1, -1 if mc == 0 else 1),
                }
            )
    assert tris, "expected L triomino markers"
    return tris


def _infer_tile(grid: np.ndarray) -> np.ndarray:
    tile = np.full((2, 2), -1, dtype=int)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value not in (BLUE, YELLOW):
                continue
            rr, cc = r % 2, c % 2
            if tile[rr, cc] == -1:
                tile[rr, cc] = value
            else:
                assert tile[rr, cc] == value, "visible residues disagree"
    assert np.all(tile != -1), "all 2x2 residues must remain visible"
    return tile


def _ray_target_rect(
    start: tuple[int, int], direction: tuple[int, int], rects: list[dict], shape: tuple[int, int]
) -> int | None:
    rect_by_cell: dict[tuple[int, int], int] = {}
    for idx, rect in enumerate(rects):
        r0, c0, r1, c1 = rect["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                rect_by_cell[(r, c)] = idx

    r, c = start
    dr, dc = direction
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        hit = rect_by_cell.get((r, c))
        if hit is not None:
            return hit
        r += dr
        c += dc
    return None


def _render_restored(grid: np.ndarray, rect_ids: set[int], rects: list[dict], tile: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for idx in rect_ids:
        r0, c0, r1, c1 = rects[idx]["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = int(tile[r % 2, c % 2])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore only the occluders pointed to by grey L triomino missing corners using the hidden 2x2 periodic tile.

    Input structure:
    - The full grid is covered by a 2x2 periodic blue/yellow background pattern.
    - Several solid orange rectangles overwrite that pattern as occluders.
    - Each occluder has one nearby red or grey L triomino marker, where the triomino fills three cells of a 2x2 box.

    Rule:
    - Detect every orange filled rectangle and every red/grey L triomino.
    - Read the triomino's missing-corner direction from the empty corner of its 2x2 box.
    - For each grey triomino, follow that diagonal direction to the first orange rectangle and mark that rectangle for restoration.
    - Infer the underlying 2x2 periodic blue/yellow tile from the visible background cells, repaint only the marked rectangles with the tile, and leave red-linked rectangles plus all triominoes unchanged.

    Key insight:
    - The triomino color answers whether to restore at all, while the missing corner answers which neighboring occluder the restoration applies to.
    """
    g = np.asarray(grid, dtype=int)
    rects = _detect_rectangles(g)
    tris = _detect_triangles(g)
    tile = _infer_tile(g)

    restore_ids: set[int] = set()
    for tri in tris:
        if tri["color"] != GREY:
            continue
        hit = _ray_target_rect(tri["missing_global"], tri["dir"], rects, g.shape)
        assert hit is not None, "grey triomino must point to an orange rectangle"
        restore_ids.add(hit)

    return _render_restored(g, restore_ids, rects, tile)


def _tile_value(parity: int, inverse: bool) -> int:
    return BLUE if (parity == 0) ^ inverse else YELLOW


def _build_background(h: int, w: int, inverse: bool) -> tuple[np.ndarray, np.ndarray]:
    tile = np.empty((2, 2), dtype=int)
    for r in range(2):
        for c in range(2):
            tile[r, c] = _tile_value((r + c) % 2, inverse)
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = tile[r % 2, c % 2]
    return grid, tile


def _expanded_bbox(bbox: tuple[int, int, int, int], h: int, w: int, pad: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _group_bbox(rect_bbox: tuple[int, int, int, int], tri_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (
        min(rect_bbox[0], tri_bbox[0]),
        min(rect_bbox[1], tri_bbox[1]),
        max(rect_bbox[2], tri_bbox[2]),
        max(rect_bbox[3], tri_bbox[3]),
    )


def _compatible_configs(color: int, missing: tuple[int, int]) -> list[dict]:
    mr, mc = missing
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    configs = []
    if color == GREY:
        configs.append({"mode": "vertical", "side": dr})
        configs.append({"mode": "horizontal", "side": dc})
    else:
        configs.append({"mode": "vertical", "side": -dr})
        configs.append({"mode": "horizontal", "side": -dc})
    return configs


def _candidate_rect_positions(
    h: int,
    w: int,
    tri_r0: int,
    tri_c0: int,
    rect_h: int,
    rect_w: int,
    color: int,
    missing: tuple[int, int],
    config: dict,
) -> list[tuple[int, int, int, int]]:
    mr, mc = missing
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    results: list[tuple[int, int, int, int]] = []

    if config["mode"] == "vertical":
        if config["side"] == -1:
            rect_r1 = tri_r0 - 2
            rect_r0 = rect_r1 - rect_h + 1
        else:
            rect_r0 = tri_r0 + 3
            rect_r1 = rect_r0 + rect_h - 1
        if rect_r0 < 0 or rect_r1 >= h:
            return results

        if color == GREY:
            anchor = tri_c0 + mc + 2 * dc
        else:
            anchor = tri_c0 + (1 - mc)
        c_lo = max(0, anchor - rect_w + 1)
        c_hi = min(anchor, w - rect_w)
        for rect_c0 in range(c_lo, c_hi + 1):
            results.append((rect_r0, rect_c0, rect_r1, rect_c0 + rect_w - 1))
    else:
        if config["side"] == -1:
            rect_c1 = tri_c0 - 2
            rect_c0 = rect_c1 - rect_w + 1
        else:
            rect_c0 = tri_c0 + 3
            rect_c1 = rect_c0 + rect_w - 1
        if rect_c0 < 0 or rect_c1 >= w:
            return results

        if color == GREY:
            anchor = tri_r0 + mr + 2 * dr
        else:
            anchor = tri_r0 + (1 - mr)
        r_lo = max(0, anchor - rect_h + 1)
        r_hi = min(anchor, h - rect_h)
        for rect_r0 in range(r_lo, r_hi + 1):
            results.append((rect_r0, rect_c0, rect_r0 + rect_h - 1, rect_c1))

    return results


def _bbox_clear(
    bbox: tuple[int, int, int, int], reserved: np.ndarray
) -> bool:
    r0, c0, r1, c1 = bbox
    return not reserved[r0 : r1 + 1, c0 : c1 + 1].any()


def _grey_ray_hits_own_rect(
    tri_r0: int,
    tri_c0: int,
    missing: tuple[int, int],
    rect_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> bool:
    rect_set = {
        (r, c)
        for r in range(rect_bbox[0], rect_bbox[2] + 1)
        for c in range(rect_bbox[1], rect_bbox[3] + 1)
    }
    mr, mc = missing
    r, c = tri_r0 + mr, tri_c0 + mc
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    r += dr
    c += dc
    while _in_bounds(shape, r, c):
        if (r, c) in rect_set:
            return True
        r += dr
        c += dc
    return False


def _visible_residues_ok(grid: np.ndarray) -> bool:
    seen = set()
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) in (BLUE, YELLOW):
                seen.add((r % 2, c % 2))
    return len(seen) == 4


def _place_objects(
    h: int, w: int, colors: list[int], missings: list[tuple[int, int]]
) -> list[dict] | None:
    reserved = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for color, missing in zip(colors, missings):
        rect_h, rect_w = random.choice(GEN_RECT_SIZES)
        tri_positions = [(r, c) for r in range(h - 1) for c in range(w - 1)]
        random.shuffle(tri_positions)
        configs = _compatible_configs(color, missing)
        random.shuffle(configs)

        placed = None
        for config in configs:
            for tri_r0, tri_c0 in tri_positions:
                tri_bbox = (tri_r0, tri_c0, tri_r0 + 1, tri_c0 + 1)
                tri_cells = _tri_cells(tri_r0, tri_c0, missing)
                rect_positions = _candidate_rect_positions(
                    h, w, tri_r0, tri_c0, rect_h, rect_w, color, missing, config
                )
                random.shuffle(rect_positions)
                for rect_bbox in rect_positions:
                    if color == GREY and not _grey_ray_hits_own_rect(
                        tri_r0, tri_c0, missing, rect_bbox, (h, w)
                    ):
                        continue
                    group_box = _group_bbox(rect_bbox, tri_bbox)
                    padded = _expanded_bbox(group_box, h, w, pad=1)
                    if not _bbox_clear(padded, reserved):
                        continue
                    placed = {
                        "color": color,
                        "missing": missing,
                        "rect_bbox": rect_bbox,
                        "tri_bbox": tri_bbox,
                        "tri_cells": tri_cells,
                    }
                    break
                if placed is not None:
                    break
            if placed is not None:
                break
        if placed is None:
            return None

        objects.append(placed)
        pr0, pc0, pr1, pc1 = _expanded_bbox(
            _group_bbox(placed["rect_bbox"], placed["tri_bbox"]), h, w, pad=1
        )
        reserved[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
    return objects


def _paint_input(base: np.ndarray, objects: list[dict]) -> np.ndarray:
    inp = base.copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["rect_bbox"]
        inp[r0 : r1 + 1, c0 : c1 + 1] = ORANGE
        for r, c in obj["tri_cells"]:
            inp[r, c] = obj["color"]
    return inp


def _expected_output(base: np.ndarray, inp: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = inp.copy()
    for obj in objects:
        if obj["color"] != GREY:
            continue
        r0, c0, r1, c1 = obj["rect_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = base[r0 : r1 + 1, c0 : c1 + 1]
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, objects: list[dict]) -> bool:
    rects = _detect_rectangles(inp)
    tris = _detect_triangles(inp)
    if len(rects) != len(objects) or len(tris) != len(objects):
        return False
    if not _visible_residues_ok(inp):
        return False

    grey_count = sum(obj["color"] == GREY for obj in objects)
    red_count = sum(obj["color"] == RED for obj in objects)
    if grey_count == 0 or red_count == 0:
        return False
    if np.array_equal(inp, expected):
        return False

    actual = main(inp)
    if not np.array_equal(actual, expected):
        return False

    changed_boxes = 0
    unchanged_red = 0
    for obj in objects:
        r0, c0, r1, c1 = obj["rect_bbox"]
        before = inp[r0 : r1 + 1, c0 : c1 + 1]
        after = expected[r0 : r1 + 1, c0 : c1 + 1]
        if obj["color"] == GREY:
            if np.array_equal(before, after):
                return False
            changed_boxes += 1
        else:
            if not np.array_equal(before, after):
                return False
            unchanged_red += 1
    return changed_boxes == grey_count and unchanged_red == red_count


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GEN_HEIGHTS)
    w = random.choice(GEN_WIDTHS)
    inverse = bool(random.getrandbits(1))
    base, _tile = _build_background(h, w, inverse)

    count = random.choice((3, 4))
    missings = random.sample(MISSING_CORNERS, count)
    grey_count = random.randint(1, count - 1)
    colors = [GREY] * grey_count + [RED] * (count - grey_count)
    random.shuffle(colors)

    objects = _place_objects(h, w, colors, missings)
    if objects is None:
        return None

    inp = _paint_input(base, objects)
    expected = _expected_output(base, inp, objects)

    if not _validate_semantics(inp, expected, objects):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
