import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GRAY = 5
ORANGE = 7

PATTERN_COLORS = (4, 6, 8, 9)
KINDS = ("horizontal", "vertical", "rotation")
LEGEND_COLOR_BY_KIND = {
    "horizontal": BLUE,
    "vertical": GREEN,
    "rotation": RED,
}

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

CORNERS = ("tl", "tr", "bl", "br")


def _bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_cells(bbox):
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _components_of_color(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _tight_crop(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_connected(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(coords[0])
    q = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _classify_symmetry(mask):
    tight = _tight_crop(mask.astype(bool))
    horizontal = np.array_equal(tight, np.flipud(tight))
    vertical = np.array_equal(tight, np.fliplr(tight))
    rotation = np.array_equal(tight, np.rot90(tight, 2))
    kinds = []
    if horizontal:
        kinds.append("horizontal")
    if vertical:
        kinds.append("vertical")
    if rotation:
        kinds.append("rotation")
    if len(kinds) != 1:
        return None
    return kinds[0]


def _symmetry_orbits(size, kind):
    seen = set()
    orbits = []
    for r in range(size):
        for c in range(size):
            if (r, c) in seen:
                continue
            if kind == "horizontal":
                orbit = {(r, c), (size - 1 - r, c)}
            elif kind == "vertical":
                orbit = {(r, c), (r, size - 1 - c)}
            else:
                orbit = {(r, c), (size - 1 - r, size - 1 - c)}
            frozen = tuple(sorted(orbit))
            seen.update(orbit)
            orbits.append(frozen)
    return orbits


def _random_symmetric_mask(kind, size):
    area_lo = max(3, size)
    area_hi = size * size - 1
    orbits = _symmetry_orbits(size, kind)
    for _ in range(300):
        prob = random.uniform(0.35, 0.7)
        mask = np.zeros((size, size), dtype=bool)
        for orbit in orbits:
            if random.random() < prob:
                for r, c in orbit:
                    mask[r, c] = True
        if int(mask.sum()) < area_lo or int(mask.sum()) > area_hi:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not _mask_connected(mask):
            continue
        if _classify_symmetry(mask) != kind:
            continue
        return mask
    raise RuntimeError(f"failed to generate {kind} mask of size {size}")


def _extract_frames(grid):
    frames = []
    for coords in _components_of_color(grid, GRAY):
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        border = _border_cells(bbox)
        comp_set = set(coords)
        if not comp_set.issubset(border):
            continue
        missing = list(border - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        if gap in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            continue
        if len(comp_set) != len(border) - 1:
            continue
        if gap[0] == r0:
            gap_dir = "up"
        elif gap[0] == r1:
            gap_dir = "down"
        elif gap[1] == c0:
            gap_dir = "left"
        else:
            gap_dir = "right"
        frames.append(
            {
                "bbox": bbox,
                "cells": comp_set,
                "gap": gap,
                "gap_dir": gap_dir,
            }
        )
    return frames


def _pattern_mask_in_frame(grid, frame):
    r0, c0, r1, c1 = frame["bbox"]
    inside = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = inside != BLACK
    if not mask.any():
        raise AssertionError("frame interior must contain a pattern")
    return _tight_crop(mask)


def _orange_bar_dirs(grid, br, bc):
    h, w = grid.shape
    dirs = []
    if br - 1 >= 0 and np.all(grid[br - 1, bc : bc + 2] == ORANGE):
        dirs.append("up")
    if br + 2 < h and np.all(grid[br + 2, bc : bc + 2] == ORANGE):
        dirs.append("down")
    if bc - 1 >= 0 and np.all(grid[br : br + 2, bc - 1] == ORANGE):
        dirs.append("left")
    if bc + 2 < w and np.all(grid[br : br + 2, bc + 2] == ORANGE):
        dirs.append("right")
    return dirs


def _find_legend_block(grid, frame):
    h, w = grid.shape
    gap_r, gap_c = frame["gap"]
    direction = frame["gap_dir"]
    candidates = []
    for br in range(h - 1):
        for bc in range(w - 1):
            patch = grid[br : br + 2, bc : bc + 2]
            color = int(patch[0, 0])
            if color in (BLACK, GRAY, ORANGE):
                continue
            if not np.all(patch == color):
                continue
            if len(_orange_bar_dirs(grid, br, bc)) != 1:
                continue
            if direction in ("left", "right"):
                if not (br <= gap_r <= br + 1):
                    continue
                if direction == "left":
                    if bc + 1 >= gap_c:
                        continue
                    forward = gap_c - (bc + 1)
                else:
                    if bc <= gap_c:
                        continue
                    forward = bc - gap_c
            else:
                if not (bc <= gap_c <= bc + 1):
                    continue
                if direction == "up":
                    if br + 1 >= gap_r:
                        continue
                    forward = gap_r - (br + 1)
                else:
                    if br <= gap_r:
                        continue
                    forward = br - gap_r
            if 1 <= forward <= 6:
                candidates.append((forward, color))
    if not candidates:
        return None
    candidates.sort()
    return {"color": candidates[0][1]}


def _ray_cells(grid, frame):
    h, w = grid.shape
    r, c = frame["gap"]
    dr, dc = DIRS[frame["gap_dir"]]
    cells = []
    while 0 <= r < h and 0 <= c < w and grid[r, c] != GRAY:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Corner legends are rendered as miniature framed examples because ARC grids
    cannot carry text labels. Each legend frame contains a sample motif of one
    symmetry class and points toward a 2x2 color block with an orange stub.

    Ordinary gray frames use the same motif logic. Classify each interior mask
    as horizontal-only, vertical-only, or 180-rotation-only, then extend a ray
    from the frame gap using the legend color for that class.
    """
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    mapping = {}
    legend_ids = set()
    for idx, frame in enumerate(frames):
        block = _find_legend_block(g, frame)
        if block is None:
            continue
        kind = _classify_symmetry(_pattern_mask_in_frame(g, frame))
        assert kind is not None
        assert kind not in mapping
        mapping[kind] = block["color"]
        legend_ids.add(idx)

    assert set(mapping) == set(KINDS), "expected exactly one legend for each symmetry class"

    out = g.copy()
    for idx, frame in enumerate(frames):
        if idx in legend_ids:
            continue
        kind = _classify_symmetry(_pattern_mask_in_frame(g, frame))
        assert kind is not None
        color = mapping[kind]
        for r, c in _ray_cells(g, frame):
            out[r, c] = color
    return out


# --------------- generator ---------------

GRID_H_RANGE = (26, 35)
GRID_W_RANGE = (26, 35)
REGULAR_FRAME_COUNT_RANGE = (3, 5)
REGULAR_FRAME_PAD = 1
PLACE_RETRIES = 80
GEN_RETRIES = 400


def _draw_frame(grid, bbox, gap):
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GRAY
    grid[r1, c0 : c1 + 1] = GRAY
    grid[r0 : r1 + 1, c0] = GRAY
    grid[r0 : r1 + 1, c1] = GRAY
    grid[gap] = BLACK


def _draw_pattern(grid, bbox, mask, color, offset_r, offset_c):
    r0, c0, _, _ = bbox
    pr0 = r0 + 1 + offset_r
    pc0 = c0 + 1 + offset_c
    grid[pr0 : pr0 + mask.shape[0], pc0 : pc0 + mask.shape[1]][mask] = color


def _legend_layout(corner, h, w):
    if corner == "tl":
        return {
            "frame_bbox": (1, 4, 5, 8),
            "gap": (3, 4),
            "block_bbox": (2, 1, 3, 2),
            "bar_dirs": ("up", "left"),
            "zone": (0, 0, 6, 9),
        }
    if corner == "tr":
        return {
            "frame_bbox": (1, w - 9, 5, w - 5),
            "gap": (3, w - 5),
            "block_bbox": (2, w - 3, 3, w - 2),
            "bar_dirs": ("up", "right"),
            "zone": (0, w - 10, 6, w - 1),
        }
    if corner == "bl":
        return {
            "frame_bbox": (h - 6, 4, h - 2, 8),
            "gap": (h - 4, 4),
            "block_bbox": (h - 4, 1, h - 3, 2),
            "bar_dirs": ("down", "left"),
            "zone": (h - 7, 0, h - 1, 9),
        }
    return {
        "frame_bbox": (h - 6, w - 9, h - 2, w - 5),
        "gap": (h - 4, w - 5),
        "block_bbox": (h - 4, w - 3, h - 3, w - 2),
        "bar_dirs": ("down", "right"),
        "zone": (h - 7, w - 10, h - 1, w - 1),
    }


def _draw_orange_bar(grid, block_bbox, direction):
    r0, c0, r1, c1 = block_bbox
    if direction == "up":
        grid[r0 - 1, c0 : c1 + 1] = ORANGE
    elif direction == "down":
        grid[r1 + 1, c0 : c1 + 1] = ORANGE
    elif direction == "left":
        grid[r0 : r1 + 1, c0 - 1] = ORANGE
    else:
        grid[r0 : r1 + 1, c1 + 1] = ORANGE


def _paint_legend(grid, corner, kind, exemplar_color):
    layout = _legend_layout(corner, *grid.shape)
    _draw_frame(grid, layout["frame_bbox"], layout["gap"])
    sample = _random_symmetric_mask(kind, 3)
    _draw_pattern(grid, layout["frame_bbox"], sample, exemplar_color, 0, 0)
    br0, bc0, br1, bc1 = layout["block_bbox"]
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = LEGEND_COLOR_BY_KIND[kind]
    _draw_orange_bar(grid, layout["block_bbox"], random.choice(layout["bar_dirs"]))
    return layout["zone"]


def _mark_rect(mask, bbox, pad=0):
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(mask.shape[0] - 1, r1 + pad)
    cc1 = min(mask.shape[1] - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_regular_spec(kind):
    size = random.randint(3, 5)
    mask = _random_symmetric_mask(kind, size)
    inner_h = size + random.randint(2, 3)
    inner_w = size + random.randint(2, 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    offset_r = random.randint(1, inner_h - size - 1)
    offset_c = random.randint(1, inner_w - size - 1)
    gap_dir = random.choice(tuple(DIRS))
    gap_pos = random.randint(1, outer_w - 2) if gap_dir in ("up", "down") else random.randint(1, outer_h - 2)
    return {
        "kind": kind,
        "mask": mask,
        "color": random.choice(PATTERN_COLORS),
        "outer_h": outer_h,
        "outer_w": outer_w,
        "offset_r": offset_r,
        "offset_c": offset_c,
        "gap_dir": gap_dir,
        "gap_pos": gap_pos,
    }


def _gap_from_bbox(bbox, gap_dir, gap_pos):
    r0, c0, r1, c1 = bbox
    if gap_dir == "up":
        return (r0, c0 + gap_pos)
    if gap_dir == "down":
        return (r1, c0 + gap_pos)
    if gap_dir == "left":
        return (r0 + gap_pos, c0)
    return (r0 + gap_pos, c1)


def _place_regular_frames(grid, specs, legend_reserved):
    h, w = grid.shape
    occupied = legend_reserved.copy()
    placed = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: -(specs[i]["outer_h"] * specs[i]["outer_w"]))
    for idx in order:
        spec = specs[idx]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        if h - oh - 2 < 1 or w - ow - 2 < 1:
            return None
        for _ in range(PLACE_RETRIES):
            r0 = random.randint(1, h - oh - 2)
            c0 = random.randint(1, w - ow - 2)
            bbox = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            rr0 = r0 - REGULAR_FRAME_PAD
            cc0 = c0 - REGULAR_FRAME_PAD
            rr1 = r0 + oh - 1 + REGULAR_FRAME_PAD
            cc1 = c0 + ow - 1 + REGULAR_FRAME_PAD
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            gap = _gap_from_bbox(bbox, spec["gap_dir"], spec["gap_pos"])
            placed[idx] = {
                **spec,
                "bbox": bbox,
                "gap": gap,
            }
            _mark_rect(occupied, bbox, REGULAR_FRAME_PAD)
            break
        else:
            return None
    for spec in placed:
        _draw_frame(grid, spec["bbox"], spec["gap"])
        _draw_pattern(grid, spec["bbox"], spec["mask"], spec["color"], spec["offset_r"], spec["offset_c"])
    return placed


def _inside_bbox(cell, bbox):
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _validate_paths(grid, regular_frames, legend_zone_mask):
    path_cells = set()
    for i, frame in enumerate(regular_frames):
        cells = _ray_cells(grid, frame)
        if len(cells) < 2:
            return False
        for j, cell in enumerate(cells):
            if grid[cell] != BLACK:
                return False
            if j > 0 and legend_zone_mask[cell]:
                return False
            for k, other in enumerate(regular_frames):
                if i != k and _inside_bbox(cell, other["bbox"]):
                    return False
            if cell in path_cells:
                return False
            path_cells.add(cell)
    return True


def _expected_output(inp, regular_frames):
    out = inp.copy()
    for frame in regular_frames:
        color = LEGEND_COLOR_BY_KIND[frame["kind"]]
        for r, c in _ray_cells(inp, frame):
            out[r, c] = color
    return out


def _gen_attempt():
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    corners = random.sample(CORNERS, 3)
    corner_by_kind = dict(zip(random.sample(KINDS, len(KINDS)), corners))
    legend_zone_mask = np.zeros((h, w), dtype=bool)
    for kind in KINDS:
        corner = corner_by_kind[kind]
        exemplar_color = random.choice(PATTERN_COLORS)
        zone = _paint_legend(grid, corner, kind, exemplar_color)
        _mark_rect(legend_zone_mask, zone, 0)

    count = random.randint(*REGULAR_FRAME_COUNT_RANGE)
    classes = list(KINDS) + [random.choice(KINDS) for _ in range(count - len(KINDS))]
    random.shuffle(classes)
    specs = [_sample_regular_spec(kind) for kind in classes]
    regular_frames = _place_regular_frames(grid, specs, legend_zone_mask)
    if regular_frames is None:
        return None

    if not _validate_paths(grid, regular_frames, legend_zone_mask):
        return None

    expected = _expected_output(grid, regular_frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
