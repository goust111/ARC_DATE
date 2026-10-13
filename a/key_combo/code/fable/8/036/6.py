from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


CORNERS = ("upper_left", "upper_right", "lower_left", "lower_right")
GEN_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 30
FRAME_COUNT_CHOICES = (2, 3)
FRAME_SIZE_RANGE = (5, 8)
MARGIN_RANGE = (2, 4)
ZONE_GAP = 2


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records = []
    for color in colors_present(grid, exclude={bg}):
        for coords in components_mask(grid == int(color), connectivity=4):
            records.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return records


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or comp["area"] != 2 * (h + w) - 4:
        return False
    coords = {tuple(map(int, rc)) for rc in comp["coords"]}
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_perimeter = r in (r0, r1) or c in (c0, c1)
            if on_perimeter != ((r, c) in coords):
                return False
    return True


def _cross_info(coords: np.ndarray, color: int) -> dict | None:
    rows = Counter(map(int, coords[:, 0]))
    cols = Counter(map(int, coords[:, 1]))
    repeated_rows = [r for r, count in rows.items() if count > 1]
    repeated_cols = [c for c, count in cols.items() if count > 1]
    if len(repeated_rows) != 1 or len(repeated_cols) != 1:
        return None

    cross_r = repeated_rows[0]
    cross_c = repeated_cols[0]
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    if not all(int(r) == cross_r or int(c) == cross_c for r, c in coords):
        return None
    coord_set = {tuple(map(int, rc)) for rc in coords}
    if (cross_r, c0) not in coord_set or (cross_r, c1) not in coord_set:
        return None
    if (r0, cross_c) not in coord_set or (r1, cross_c) not in coord_set:
        return None
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "row": int(cross_r),
        "col": int(cross_c),
    }


def _corner_points(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _nearest_corner(box: tuple[int, int, int, int], marker: tuple[int, int]) -> tuple[int, tuple[int, int]]:
    mr, mc = marker
    corners = _corner_points(box)
    dists = [abs(mr - r) + abs(mc - c) for r, c in corners]
    best = min(dists)
    assert dists.count(best) == 1, "marker must have a unique nearest corner"
    idx = dists.index(best)
    return idx, corners[idx]


def _pair_frames_and_crosses(grid: np.ndarray) -> tuple[int, tuple[int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = _component_records(g, bg)

    marker_candidates = [comp for comp in comps if comp["area"] == 1]
    assert len(marker_candidates) == 1, "expected one isolated marker pixel"
    marker_comp = marker_candidates[0]
    marker = tuple(map(int, marker_comp["coords"][0]))

    frames = [comp for comp in comps if _is_rect_frame(comp)]
    crosses = []
    for color in colors_present(g, exclude={bg}):
        if int(color) == marker_comp["color"]:
            continue
        coords = np.argwhere(g == int(color))
        info = _cross_info(coords, int(color))
        if info is not None:
            crosses.append(info)
    assert frames, "expected at least one rectangular frame"
    assert len(frames) == len(crosses), "expected one cross per frame"

    pairs = []
    used = set()
    for frame in frames:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        matches = []
        for idx, cross in enumerate(crosses):
            zr0, zc0, zr1, zc1 = cross["bbox"]
            if not (zr0 < fr0 < fr1 < zr1 and zc0 < fc0 < fc1 < zc1):
                continue
            if not (fr0 < cross["row"] < fr1 and fc0 < cross["col"] < fc1):
                continue
            matches.append(idx)
        assert len(matches) == 1, "each frame must have exactly one enclosing cross"
        cross_idx = matches[0]
        assert cross_idx not in used, "crosses must pair one-to-one with frames"
        used.add(cross_idx)
        pairs.append({"frame": frame, "cross": crosses[cross_idx]})

    return bg, marker + (marker_comp["color"],), pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each corridor cross to the frame corner nearest the shared marker while keeping the frame and marker fixed.

    Input structure:
    - The grid has several disjoint one-pixel-thick rectangular frames, each sitting inside its own empty corridor area.
    - Each frame is pierced by a differently colored cross: one full row and one full column of that corridor pass through the frame interior, with frame pixels covering the border overlaps.
    - One isolated marker pixel lies elsewhere on the canvas; for every frame, its Manhattan distance to the four frame corners has a unique minimum.

    Rule:
    - For each frame, erase its original cross color from the old corridor row and corridor column.
    - Find the frame corner nearest to the shared marker, then redraw that cross color across the whole corridor row and column aligned to that corner.
    - Restore every frame on top of the moved cross, and finally restore the marker pixel on top.

    Key insight:
    - The marker is the only direction code: different frames can move to different corners because each one compares the same marker against its own local corner geometry.
    """
    g = np.asarray(grid, dtype=int)
    bg, (marker_r, marker_c, marker_color), pairs = _pair_frames_and_crosses(g)

    out = g.copy()
    for pair in pairs:
        cross = pair["cross"]
        out[tuple(cross["coords"].T)] = bg

    for pair in pairs:
        frame = pair["frame"]
        cross = pair["cross"]
        _corner_idx, (target_r, target_c) = _nearest_corner(frame["bbox"], (marker_r, marker_c))
        zr0, zc0, zr1, zc1 = cross["bbox"]
        out[target_r, zc0 : zc1 + 1] = cross["color"]
        out[zr0 : zr1 + 1, target_c] = cross["color"]

    for pair in pairs:
        frame = pair["frame"]
        out[tuple(frame["coords"].T)] = frame["color"]

    out[marker_r, marker_c] = marker_color
    return out


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_input(meta: dict) -> np.ndarray:
    grid = np.full(meta["shape"], meta["bg"], dtype=int)
    for obj in meta["objects"]:
        zr0, zc0, zr1, zc1 = obj["zone"]
        grid[obj["cross_row"], zc0 : zc1 + 1] = obj["cross_color"]
        grid[zr0 : zr1 + 1, obj["cross_col"]] = obj["cross_color"]
        _paint_frame(grid, obj["frame"], obj["frame_color"])
    mr, mc = meta["marker"]
    grid[mr, mc] = meta["marker_color"]
    return grid


def _render_output(meta: dict, forced_corner: int | None = None) -> np.ndarray:
    grid = np.full(meta["shape"], meta["bg"], dtype=int)
    for obj in meta["objects"]:
        zr0, zc0, zr1, zc1 = obj["zone"]
        corner_idx = obj["target_corner"] if forced_corner is None else forced_corner
        target_r, target_c = _corner_points(obj["frame"])[corner_idx]
        grid[target_r, zc0 : zc1 + 1] = obj["cross_color"]
        grid[zr0 : zr1 + 1, target_c] = obj["cross_color"]
        _paint_frame(grid, obj["frame"], obj["frame_color"])
    mr, mc = meta["marker"]
    grid[mr, mc] = meta["marker_color"]
    return grid


def _nonoverlap(box: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in others:
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _sample_layout(h: int, w: int, count: int) -> list[dict] | None:
    boxes = []
    objects = []
    for _ in range(count):
        fh = random.randint(*FRAME_SIZE_RANGE)
        fw = random.randint(*FRAME_SIZE_RANGE)
        top = random.randint(*MARGIN_RANGE)
        bottom = random.randint(*MARGIN_RANGE)
        left = random.randint(*MARGIN_RANGE)
        right = random.randint(*MARGIN_RANGE)
        zh = fh + top + bottom
        zw = fw + left + right
        placed = False
        for _ in range(80):
            zr0 = random.randint(0, h - zh)
            zc0 = random.randint(0, w - zw)
            zone = (zr0, zc0, zr0 + zh - 1, zc0 + zw - 1)
            if not _nonoverlap(zone, boxes, ZONE_GAP):
                continue
            fr0 = zr0 + top
            fc0 = zc0 + left
            frame = (fr0, fc0, fr0 + fh - 1, fc0 + fw - 1)
            boxes.append(zone)
            objects.append({"zone": zone, "frame": frame})
            placed = True
            break
        if not placed:
            return None
    return objects


def _marker_candidates(meta: dict) -> list[tuple[int, int]]:
    h, w = meta["shape"]
    blocked = np.zeros((h, w), dtype=bool)
    for obj in meta["objects"]:
        zr0, zc0, zr1, zc1 = obj["zone"]
        blocked[zr0 : zr1 + 1, zc0 : zc1 + 1] = True

    candidates = []
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            ok = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if blocked[rr, cc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((r, c))
    return candidates


def _annotate_targets(meta: dict) -> bool:
    chosen = []
    for obj in meta["objects"]:
        try:
            idx, _corner = _nearest_corner(obj["frame"], meta["marker"])
        except AssertionError:
            return False
        obj["target_corner"] = idx
        chosen.append(idx)

    distinct = len(set(chosen))
    if distinct < 2:
        return False
    if len(meta["objects"]) == 3 and random.random() < 0.45 and distinct < 3:
        return False
    return True


def _semantic_checks(meta: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    if not np.array_equal(main(inp), out):
        return False
    if any(np.array_equal(out, _render_output(meta, forced_corner=i)) for i in range(4)):
        return False
    for obj in meta["objects"]:
        fr0, fc0, fr1, fc1 = obj["frame"]
        if not (fr0 < obj["cross_row"] < fr1 and fc0 < obj["cross_col"] < fc1):
            return False
        zr0, zc0, zr1, zc1 = obj["zone"]
        if not (zr0 < fr0 < fr1 < zr1 and zc0 < fc0 < fc1 < zc1):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.choice(FRAME_COUNT_CHOICES)
    layout = _sample_layout(h, w, count)
    if layout is None:
        return None

    color_pool = list(range(10))
    random.shuffle(color_pool)
    if len(color_pool) < 2 * count + 2:
        return None
    bg = color_pool.pop()
    marker_color = color_pool.pop()

    meta = {
        "shape": (h, w),
        "bg": bg,
        "marker_color": marker_color,
        "objects": [],
    }

    for obj in layout:
        cross_color = color_pool.pop()
        frame_color = color_pool.pop()
        fr0, fc0, fr1, fc1 = obj["frame"]
        meta["objects"].append(
            {
                **obj,
                "frame_color": frame_color,
                "cross_color": cross_color,
                "cross_row": random.randint(fr0 + 1, fr1 - 1),
                "cross_col": random.randint(fc0 + 1, fc1 - 1),
            }
        )

    candidates = _marker_candidates(meta)
    random.shuffle(candidates)
    for marker in candidates:
        meta["marker"] = marker
        if not _annotate_targets(meta):
            continue
        inp = _render_input(meta)
        out = _render_output(meta)
        if _semantic_checks(meta, inp, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/8/036/6")
