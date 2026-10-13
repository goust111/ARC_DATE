from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

RED = 2
BLUE = 1
GREY = 5

DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return int(min(rows)), int(min(cols)), int(max(rows)), int(max(cols))


def _border_cells(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    cells: list[Coord] = []
    for c in range(c0, c1 + 1):
        cells.append((int(r0), int(c)))
        cells.append((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        cells.append((int(r), int(c0)))
        cells.append((int(r), int(c1)))
    return cells


def _frame_bbox(grid: np.ndarray, coords: list[Coord], color: int, bg: int) -> BBox | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    expected = set(_border_cells((r0, c0, r1, c1)))
    if set(coords) != expected:
        return None
    if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
        return None
    if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1][grid[r0 : r1 + 1, c0 : c1 + 1] == color] == color):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _solid_rect_bbox(coords: list[Coord]) -> BBox | None:
    r0, c0, r1, c1 = _bbox(coords)
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    if len(coords) != area:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[dict], np.ndarray]:
    colors = [int(v) for v in np.unique(grid) if int(v) not in {bg, RED, BLUE}]
    candidates: list[tuple[int, list[BBox]]] = []
    for color in colors:
        comps = _components(grid == color)
        frame_boxes: list[BBox] = []
        ok = True
        for comp in comps:
            box = _frame_bbox(grid, comp, color, bg)
            if box is None:
                ok = False
                break
            frame_boxes.append(box)
        if ok and frame_boxes:
            candidates.append((int(color), frame_boxes))
    assert len(candidates) == 1, "expected a unique frame color"

    frame_color, boxes = candidates[0]
    owner = np.full(grid.shape, -1, dtype=int)
    frames: list[dict] = []
    for fid, box in enumerate(boxes):
        for r, c in _border_cells(box):
            owner[r, c] = int(fid)
        frames.append({"id": int(fid), "bbox": box})
    return int(frame_color), frames, owner


def _find_sources(grid: np.ndarray, bg: int, frame_color: int) -> tuple[list[dict], np.ndarray]:
    mask = (grid != bg) & (grid != frame_color) & (grid != RED) & (grid != BLUE)
    comps = _components(mask)
    assert comps, "expected at least one source region"
    source_id = np.full(grid.shape, -1, dtype=int)
    sources: list[dict] = []
    for sid, comp in enumerate(comps):
        box = _solid_rect_bbox(comp)
        assert box is not None, "source regions must be solid rectangles"
        r0, c0, r1, c1 = box
        source_id[r0 : r1 + 1, c0 : c1 + 1] = int(sid)
        sources.append(
            {
                "id": int(sid),
                "bbox": box,
                "cells": set(comp),
                "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
            }
        )
    return sources, source_id


def _find_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for color in (RED, BLUE):
        for comp in _components(grid == color):
            assert len(comp) == 3, "arrow components must have exactly three pixels"
            r0, c0, r1, c1 = _bbox(comp)
            assert r1 - r0 == 1 and c1 - c0 == 1, "arrows must be 2x2 L-shapes"
            arrows.append({"color": int(color), "cells": set(comp)})
    assert arrows, "expected at least one arrow"
    return arrows


def _adjacent_source_ids(arrow_cells: set[Coord], source_id: np.ndarray) -> set[int]:
    h, w = source_id.shape
    ids: set[int] = set()
    for r, c in arrow_cells:
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and source_id[nr, nc] >= 0:
                ids.add(int(source_id[nr, nc]))
    return ids


def _touches_from_behind(arrow_cells: set[Coord], source_cells: set[Coord], dr: int, dc: int) -> bool:
    return any((int(r - dr), int(c - dc)) in source_cells for r, c in arrow_cells)


def _frontier_cells(cells: set[Coord], dr: int, dc: int) -> list[Coord]:
    if dr > 0:
        edge = max(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    if dr < 0:
        edge = min(r for r, _ in cells)
        return [(r, c) for r, c in cells if r == edge]
    if dc > 0:
        edge = max(c for _, c in cells)
        return [(r, c) for r, c in cells if c == edge]
    edge = min(c for _, c in cells)
    return [(r, c) for r, c in cells if c == edge]


def _ray_hit_frame(
    grid: np.ndarray, bg: int, frame_owner: np.ndarray, arrow_cells: set[Coord], dr: int, dc: int
) -> int | None:
    h, w = grid.shape
    hits: set[int] = set()
    for r, c in _frontier_cells(arrow_cells, dr, dc):
        nr, nc = r + dr, c + dc
        while 0 <= nr < h and 0 <= nc < w:
            fid = int(frame_owner[nr, nc])
            if fid >= 0:
                hits.add(fid)
                break
            if int(grid[nr, nc]) != bg:
                break
            nr += dr
            nc += dc
    if len(hits) != 1:
        return None
    return next(iter(hits))


def _transform_patch(patch: np.ndarray, arrow_color: int, direction: str) -> np.ndarray:
    if arrow_color == RED:
        return np.fliplr(patch) if direction in {"left", "right"} else np.flipud(patch)
    assert arrow_color == BLUE, "arrow color must be red or blue"
    return np.rot90(patch, 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each source rectangle into the frame hit by its arrow.

    Red arrows mirror the pattern across the axis perpendicular to the movement
    direction. Blue arrows rotate the pattern by 180 degrees. The source and
    the arrow are erased after projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    frame_color, frames, frame_owner = _find_frames(g, bg)
    arrows = _find_arrows(g)
    sources, source_id = _find_sources(g, bg, frame_color)

    out = g.copy()
    for source in sources:
        r0, c0, r1, c1 = source["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg
    for arrow in arrows:
        for r, c in arrow["cells"]:
            out[r, c] = bg

    used_sources: set[int] = set()
    used_frames: set[int] = set()
    for arrow in arrows:
        source_ids = _adjacent_source_ids(arrow["cells"], source_id)
        assert len(source_ids) == 1, "each arrow must touch exactly one source"
        sid = next(iter(source_ids))
        assert sid not in used_sources, "sources must be used exactly once"
        source = sources[sid]

        matches: list[tuple[str, int]] = []
        for direction, (dr, dc) in DIRS.items():
            if not _touches_from_behind(arrow["cells"], source["cells"], dr, dc):
                continue
            fid = _ray_hit_frame(g, bg, frame_owner, arrow["cells"], dr, dc)
            if fid is not None:
                matches.append((direction, int(fid)))
        assert len(matches) == 1, "arrow direction must resolve to one frame"
        direction, fid = matches[0]
        assert fid not in used_frames, "frames must be used exactly once"

        patch = _transform_patch(source["patch"], arrow["color"], direction)
        ph, pw = patch.shape
        fr0, fc0, fr1, fc1 = frames[fid]["bbox"]
        inner_r0, inner_c0 = fr0 + 1, fc0 + 1
        inner_r1, inner_c1 = fr1 - 1, fc1 - 1

        sr0, sc0, _sr1, _sc1 = source["bbox"]
        if direction == "right":
            dst_r, dst_c = sr0, inner_c0
        elif direction == "left":
            dst_r, dst_c = sr0, inner_c1 - pw + 1
        elif direction == "down":
            dst_r, dst_c = inner_r0, sc0
        else:
            dst_r, dst_c = inner_r1 - ph + 1, sc0

        assert inner_r0 <= dst_r <= inner_r1 - ph + 1
        assert inner_c0 <= dst_c <= inner_c1 - pw + 1
        assert np.all(out[dst_r : dst_r + ph, dst_c : dst_c + pw] == bg)
        out[dst_r : dst_r + ph, dst_c : dst_c + pw] = patch

        used_sources.add(int(sid))
        used_frames.add(int(fid))

    assert len(used_sources) == len(sources) == len(arrows) == len(frames)
    return out


GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
PAIR_COUNT_RANGE = (1, 3)
PLACEMENT_ATTEMPTS = 90
GEN_ATTEMPTS = 400
BG_CHOICES = (0, 7, 8)
SOURCE_SIZE_RANGE = (2, 4)
SOURCE_COLORS = (0, 3, 4, 6, 7, 8, 9)


def _random_pattern(rng: random.Random, h: int, w: int, bg: int) -> np.ndarray:
    palette = [c for c in SOURCE_COLORS if c not in {bg, RED, BLUE, GREY}]
    for _ in range(120):
        color_count = rng.randint(2, min(4, len(palette)))
        colors = rng.sample(palette, color_count)
        arr = np.array([[rng.choice(colors) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(np.unique(arr)) < 2:
            continue
        if np.array_equal(arr, np.fliplr(arr)):
            continue
        if np.array_equal(arr, np.flipud(arr)):
            continue
        if np.array_equal(arr, np.rot90(arr, 2)):
            continue
        return arr
    raise RuntimeError("failed to build a non-degenerate source pattern")


def _arrow_cells_rel(direction: str, tail_offset: int, head_variant: int, sh: int, sw: int) -> list[Coord]:
    if direction == "right":
        return [(tail_offset, sw), (tail_offset + 1, sw), (tail_offset + head_variant, sw + 1)]
    if direction == "left":
        return [(tail_offset, -1), (tail_offset + 1, -1), (tail_offset + head_variant, -2)]
    if direction == "down":
        return [(sh, tail_offset), (sh, tail_offset + 1), (sh + 1, tail_offset + head_variant)]
    return [(-1, tail_offset), (-1, tail_offset + 1), (-2, tail_offset + head_variant)]


def _frame_origin_rel(direction: str, sh: int, sw: int, ih: int, iw: int, gap: int, orth_offset: int) -> tuple[int, int]:
    if direction == "right":
        return -orth_offset - 1, sw + 2 + gap
    if direction == "left":
        return -orth_offset - 1, -iw - gap - 4
    if direction == "down":
        return sh + 2 + gap, -orth_offset - 1
    return -ih - gap - 4, -orth_offset - 1


def _frame_cells_rel(fr0: int, fc0: int, ih: int, iw: int) -> list[Coord]:
    return _border_cells((fr0, fc0, fr0 + ih + 1, fc0 + iw + 1))


def _dilate(coords: list[Coord], h: int, w: int) -> set[Coord]:
    padded: set[Coord] = set()
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((int(nr), int(nc)))
    return padded


def _pair_layout(
    rng: random.Random, bg: int, direction: str, arrow_color: int
) -> dict:
    sh = rng.randint(*SOURCE_SIZE_RANGE)
    sw = rng.randint(*SOURCE_SIZE_RANGE)
    pattern = _random_pattern(rng, sh, sw, bg)

    if direction in {"left", "right"}:
        ih = sh + rng.randint(0, 2)
        iw = sw + rng.randint(1, 3)
        orth_offset = rng.randint(0, ih - sh)
        tail_offset = rng.randint(0, sh - 2)
    else:
        ih = sh + rng.randint(1, 3)
        iw = sw + rng.randint(0, 2)
        orth_offset = rng.randint(0, iw - sw)
        tail_offset = rng.randint(0, sw - 2)

    head_variant = rng.randint(0, 1)
    gap = rng.randint(1, 4)
    arrow_cells = _arrow_cells_rel(direction, tail_offset, head_variant, sh, sw)
    fr0, fc0 = _frame_origin_rel(direction, sh, sw, ih, iw, gap, orth_offset)
    frame_cells = _frame_cells_rel(fr0, fc0, ih, iw)
    source_cells = [(r, c) for r in range(sh) for c in range(sw)]
    all_cells = source_cells + arrow_cells + frame_cells
    r0, c0, r1, c1 = _bbox(all_cells)

    return {
        "pattern": pattern,
        "direction": direction,
        "arrow_color": int(arrow_color),
        "source_shape": (int(sh), int(sw)),
        "source_cells_rel": source_cells,
        "arrow_cells_rel": arrow_cells,
        "frame_cells_rel": frame_cells,
        "frame_bbox_rel": (int(fr0), int(fc0), int(fr0 + ih + 1), int(fc0 + iw + 1)),
        "bbox_rel": (int(r0), int(c0), int(r1), int(c1)),
    }


def _place_pair(
    grid: np.ndarray, occupied: np.ndarray, layout: dict, rng: random.Random
) -> bool:
    h, w = grid.shape
    min_r, min_c, max_r, max_c = layout["bbox_rel"]
    base_r_min, base_r_max = -min_r, h - 1 - max_r
    base_c_min, base_c_max = -min_c, w - 1 - max_c
    if base_r_min > base_r_max or base_c_min > base_c_max:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        base_r = rng.randint(base_r_min, base_r_max)
        base_c = rng.randint(base_c_min, base_c_max)

        frame_cells = [(base_r + r, base_c + c) for r, c in layout["frame_cells_rel"]]
        source_cells = [(base_r + r, base_c + c) for r, c in layout["source_cells_rel"]]
        arrow_cells = [(base_r + r, base_c + c) for r, c in layout["arrow_cells_rel"]]
        object_cells = frame_cells + source_cells + arrow_cells
        padded = _dilate(object_cells, h, w)
        if any(occupied[r, c] for r, c in padded):
            continue

        pattern = layout["pattern"]
        sh, sw = layout["source_shape"]
        grid[[r for r, _ in frame_cells], [c for _, c in frame_cells]] = GREY
        grid[base_r : base_r + sh, base_c : base_c + sw] = pattern
        for r, c in arrow_cells:
            grid[r, c] = layout["arrow_color"]
        for r, c in padded:
            occupied[r, c] = True
        return True
    return False


def _sample_arrow_colors(rng: random.Random, count: int) -> list[int]:
    if count == 1:
        return [rng.choice([RED, BLUE])]
    colors = [RED, BLUE]
    while len(colors) < count:
        colors.append(rng.choice([RED, BLUE]))
    rng.shuffle(colors)
    return colors


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    bg = rng.choice(BG_CHOICES)
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    pair_count = rng.randint(*PAIR_COUNT_RANGE)
    directions = rng.sample(list(DIRS.keys()), pair_count)
    arrow_colors = _sample_arrow_colors(rng, pair_count)
    rng.shuffle(directions)

    for direction, arrow_color in zip(directions, arrow_colors):
        placed = False
        for _ in range(80):
            layout = _pair_layout(rng, bg, direction, arrow_color)
            if _place_pair(grid, occupied, layout, rng):
                placed = True
                break
        if not placed:
            return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
