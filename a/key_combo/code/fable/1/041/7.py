from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg, random_free_location


BG = 0
CANVAS_SHAPE = (30, 30)
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
BOX_SIZE_RANGE = (4, 7)
PATH_LEN_RANGE = (2, 6)
CHAIN_LENGTHS = (1, 2, 3, 4)
MAX_ATTEMPTS = 500


def _side_from_dir(direction: tuple[int, int]) -> str:
    for side, vec in SIDE_TO_DIR.items():
        if tuple(direction) == vec:
            return side
    raise ValueError(f"unknown direction: {direction}")


def _offset_choices(side: str, height: int, width: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(1, width - 1))
    return list(range(1, height - 1))


def _gap_coord(bbox: tuple[int, int, int, int], side: str, offset: int) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    if side == "right":
        return r0 + offset, c1
    raise ValueError(side)


def _bbox_from_edge_point(
    point: tuple[int, int],
    side: str,
    height: int,
    width: int,
    offset: int,
) -> tuple[int, int, int, int]:
    r, c = map(int, point)
    if side == "top":
        r0 = r
        c0 = c - offset
    elif side == "bottom":
        r0 = r - height + 1
        c0 = c - offset
    elif side == "left":
        r0 = r - offset
        c0 = c
    elif side == "right":
        r0 = r - offset
        c0 = c - width + 1
    else:
        raise ValueError(side)
    return int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError(f"not axis-aligned: {a} -> {b}")


def _trace_to_edge(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = direction
    H, W = shape
    r, c = start
    out = [start]
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            break
        out.append((r, c))
    return out


def _side_of_gap(
    pos: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> str | None:
    r, c = pos
    r0, c0, r1, c1 = bbox
    if (r in (r0, r1)) and (c in (c0, c1)):
        return None
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    return None


def _box_patch(box: dict, *, fill: int | None = None) -> np.ndarray:
    r0, c0, r1, c1 = box["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    patch = np.full((height, width), BG, dtype=int)
    patch[0, :] = patch[-1, :] = patch[:, 0] = patch[:, -1] = int(box["border"])
    for gap in box["gaps"]:
        gr, gc = gap["pos"]
        patch[gr - r0, gc - c0] = BG
    if fill is not None:
        patch[1:-1, 1:-1] = int(fill)
    return patch


def _gaps_keep_border_connected(
    bbox: tuple[int, int, int, int],
    gaps: list[tuple[str, int]],
) -> bool:
    height = bbox[2] - bbox[0] + 1
    width = bbox[3] - bbox[1] + 1
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for side, offset in gaps:
        rr, cc = _gap_coord(bbox, side, offset)
        border[rr - bbox[0], cc - bbox[1]] = False
    coords = np.argwhere(border)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and border[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(border.sum())


def _box_area_clear(
    bbox: tuple[int, int, int, int],
    area_mask: np.ndarray,
    reserved_mask: np.ndarray,
    allowed_reserved: set[tuple[int, int]] | None = None,
    padding: int = 1,
) -> bool:
    H, W = area_mask.shape
    r0, c0, r1, c1 = bbox
    if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H - 1, r1 + padding)
    cc1 = min(W - 1, c1 + padding)
    if area_mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
        return False
    allowed = set() if allowed_reserved is None else set(allowed_reserved)
    for cell in _bbox_cells(bbox):
        if cell in allowed:
            continue
        if reserved_mask[cell]:
            return False
    return True


def _add_box(box: dict, grid: np.ndarray, area_mask: np.ndarray) -> None:
    fill = box.get("fill")
    patch = _box_patch(box, fill=fill)
    r0, c0, r1, c1 = box["bbox"]
    grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(patch != BG, patch, grid[r0 : r1 + 1, c0 : c1 + 1])
    area_mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _sample_box(
    rng: random.Random,
    *,
    bbox: tuple[int, int, int, int],
    border: int,
    required_gaps: list[tuple[str, int]],
    fill: int | None = None,
) -> dict | None:
    height = bbox[2] - bbox[0] + 1
    width = bbox[3] - bbox[1] + 1
    used_positions: set[tuple[int, int]] = set()
    gaps = []
    for side, offset in required_gaps:
        if offset not in _offset_choices(side, height, width):
            return None
    if not _gaps_keep_border_connected(bbox, required_gaps):
        return None
    for side, offset in required_gaps:
        pos = _gap_coord(bbox, side, offset)
        if pos in used_positions:
            return None
        used_positions.add(pos)
        gaps.append({"side": side, "offset": int(offset), "pos": pos, "dir": SIDE_TO_DIR[side]})
    if len(gaps) > 2:
        return None
    return {
        "bbox": tuple(map(int, bbox)),
        "border": int(border),
        "gaps": gaps,
        "fill": None if fill is None else int(fill),
    }


def _random_box_in_place(
    rng: random.Random,
    area_mask: np.ndarray,
    reserved_mask: np.ndarray,
    border: int,
    *,
    gap_count: int,
) -> dict | None:
    height = rng.randint(*BOX_SIZE_RANGE)
    width = rng.randint(*BOX_SIZE_RANGE)
    blocker = area_mask.astype(int) | reserved_mask.astype(int)
    patch = np.ones((height, width), dtype=int)
    try:
        r0, c0 = random_free_location(blocker, patch, transparent=BG, padding=1)
    except ValueError:
        return None
    bbox = (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1))
    choices = []
    used = set()
    while len(choices) < gap_count:
        side = rng.choice(SIDES)
        offset = rng.choice(_offset_choices(side, height, width))
        key = (side, offset)
        if key in used:
            continue
        used.add(key)
        choices.append(key)
    return _sample_box(rng, bbox=bbox, border=border, required_gaps=choices)


def _analyze_box(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    color = int(comp["color"])
    bbox = tuple(map(int, comp["bbox"]))
    r0, c0, r1, c1 = bbox
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    coords = {tuple(map(int, coord)) for coord in np.asarray(comp["coords"], dtype=int)}
    perimeter = set(_perimeter_cells(bbox))
    if not coords <= perimeter:
        return None
    gaps = sorted(perimeter - coords)
    if not (1 <= len(gaps) <= 2):
        return None
    gap_infos = []
    for pos in gaps:
        if int(grid[pos]) != bg:
            return None
        side = _side_of_gap(pos, bbox)
        if side is None:
            return None
        gap_infos.append({"pos": pos, "side": side, "dir": SIDE_TO_DIR[side]})
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    vals = set(map(int, inner.reshape(-1)))
    if vals == {bg}:
        fill = None
    elif len(vals) == 1:
        fill = next(iter(vals))
        if fill in (bg, color):
            return None
    else:
        return None
    return {
        "bbox": bbox,
        "border": color,
        "gaps": gap_infos,
        "fill": fill,
    }


def _extract_boxes(grid: np.ndarray) -> tuple[list[dict], int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    boxes = []
    for comp in comps:
        box = _analyze_box(g, comp, bg)
        if box is not None:
            boxes.append(box)
    sources = [i for i, box in enumerate(boxes) if box["fill"] is not None]
    assert len(sources) == 1, "expected exactly one filled source box"
    return boxes, sources[0], bg


def _cast_ray(
    grid: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
    gap_lookup: dict[tuple[int, int], tuple[int, int]],
    boxes: list[dict],
    source_box_idx: int,
    bg: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    dr, dc = direction
    H, W = grid.shape
    r, c = start
    path = [start]
    while True:
        r += dr
        c += dc
        if not (0 <= r < H and 0 <= c < W):
            return path, None
        pos = (r, c)
        hit = gap_lookup.get(pos)
        if hit is not None and hit[0] != source_box_idx:
            hit_box_idx, hit_gap_idx = hit
            gap = boxes[hit_box_idx]["gaps"][hit_gap_idx]
            if gap["dir"] == (-dr, -dc):
                path.append(pos)
                return path, (hit_box_idx, hit_gap_idx)
        if int(grid[r, c]) != bg:
            return path, None
        path.append(pos)


def _propagate(grid: np.ndarray, boxes: list[dict], source_idx: int, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    paint = int(boxes[source_idx]["fill"])
    gap_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for box_idx, box in enumerate(boxes):
        for gap_idx, gap in enumerate(box["gaps"]):
            gap_lookup[gap["pos"]] = (box_idx, gap_idx)

    visited = {source_idx}
    queue = [(source_idx, gap_idx) for gap_idx in range(len(boxes[source_idx]["gaps"]))]
    while queue:
        box_idx, exit_gap_idx = queue.pop(0)
        exit_gap = boxes[box_idx]["gaps"][exit_gap_idx]
        path, hit = _cast_ray(g, exit_gap["pos"], exit_gap["dir"], gap_lookup, boxes, box_idx, bg)
        _paint_cells(out, path, paint)
        if hit is None:
            continue
        target_box_idx, target_gap_idx = hit
        target_box = boxes[target_box_idx]
        if target_box_idx not in visited:
            visited.add(target_box_idx)
            r0, c0, r1, c1 = target_box["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = paint
            if len(target_box["gaps"]) == 2:
                next_gap_idx = 1 - target_gap_idx
                queue.append((target_box_idx, next_gap_idx))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Flood paint through aligned box gaps, filling each downstream hollow rectangle and the straight pipe segments between them.

    Input structure:
    - Black background with several one-cell-thick hollow rectangles whose borders have one or two single-cell gaps.
    - Exactly one rectangle has its whole interior already filled with a paint color.
    - Some gaps face another gap in the same row or column with only background between them; other boxes are distractors.

    Rule:
    - Treat each gap as a port whose flow direction is perpendicular to its border side.
    - Paint leaves the filled source box through its gap ports, travels straight across background, and enters any directly facing gap it reaches.
    - When a ray enters a new box, fill that box interior with the same paint and continue from its other gap if it has one; if a ray misses every port or reaches a one-gap box, that branch stops.
    - Color every traversed background segment and every traversed gap cell with the paint color; untouched boxes stay hollow.

    Key insight:
    - Connectivity is defined by gap-to-gap alignment along unobstructed straight rays, not by object proximity, size, or color identity.
    """
    boxes, source_idx, bg = _extract_boxes(np.asarray(grid, dtype=int))
    return _propagate(np.asarray(grid, dtype=int), boxes, source_idx, bg)


def _build_chain(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    grid = np.full(CANVAS_SHAPE, BG, dtype=int)
    expected = np.full(CANVAS_SHAPE, BG, dtype=int)
    area_mask = np.zeros(CANVAS_SHAPE, dtype=bool)
    reserved_mask = np.zeros(CANVAS_SHAPE, dtype=bool)

    mode = rng.choices(["sink", "open", "blocked"], weights=[4, 3, 3])[0]
    if mode == "sink":
        chain_len = rng.choice([2, 3, 4])
    else:
        chain_len = rng.choice(CHAIN_LENGTHS)

    paint = rng.choice(PALETTE)
    border_pool = [c for c in PALETTE if c != paint]
    rng.shuffle(border_pool)

    height = rng.randint(*BOX_SIZE_RANGE)
    width = rng.randint(*BOX_SIZE_RANGE)
    for _ in range(80):
        r0 = rng.randint(2, CANVAS_SHAPE[0] - height - 3)
        c0 = rng.randint(2, CANVAS_SHAPE[1] - width - 3)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        source_side = rng.choice(SIDES)
        source_offset = rng.choice(_offset_choices(source_side, height, width))
        source = _sample_box(
            rng,
            bbox=bbox,
            border=border_pool[0],
            required_gaps=[(source_side, source_offset)],
            fill=paint,
        )
        if source is None:
            continue
        if not _box_area_clear(source["bbox"], area_mask, reserved_mask):
            continue
        break
    else:
        return None

    boxes = [source]
    _add_box(source, grid, area_mask)

    current_gap = source["gaps"][0]
    chain_lines: list[list[tuple[int, int]]] = []
    chain_targets: list[int] = []

    for step_idx in range(1, chain_len):
        placed = False
        for _ in range(100):
            next_side = OPPOSITE_SIDE[_side_from_dir(current_gap["dir"])]
            next_h = rng.randint(*BOX_SIZE_RANGE)
            next_w = rng.randint(*BOX_SIZE_RANGE)
            next_offset = rng.choice(_offset_choices(next_side, next_h, next_w))
            distance = rng.randint(*PATH_LEN_RANGE)
            q = (
                current_gap["pos"][0] + current_gap["dir"][0] * distance,
                current_gap["pos"][1] + current_gap["dir"][1] * distance,
            )
            bbox = _bbox_from_edge_point(q, next_side, next_h, next_w, next_offset)
            gap_specs = [(next_side, next_offset)]
            if step_idx < chain_len - 1 or mode in ("open", "blocked"):
                exit_choices = [side for side in SIDES if side != next_side]
                exit_side = rng.choice(exit_choices)
                exit_offset = rng.choice(_offset_choices(exit_side, next_h, next_w))
                gap_specs.append((exit_side, exit_offset))
            border = border_pool[step_idx % len(border_pool)]
            box = _sample_box(rng, bbox=bbox, border=border, required_gaps=gap_specs, fill=None)
            if box is None:
                continue
            if not _box_area_clear(box["bbox"], area_mask, reserved_mask):
                continue
            segment = _line_cells(current_gap["pos"], q)
            if any(area_mask[r, c] or reserved_mask[r, c] for r, c in segment[1:-1]):
                continue
            _add_box(box, grid, area_mask)
            boxes.append(box)
            chain_lines.append(segment)
            chain_targets.append(len(boxes) - 1)
            for r, c in segment:
                reserved_mask[r, c] = True
            current_gap = box["gaps"][1] if len(box["gaps"]) == 2 else box["gaps"][0]
            placed = True
            break
        if not placed:
            return None

    final_line: list[tuple[int, int]] | None = None
    if mode == "open":
        final_line = _trace_to_edge(current_gap["pos"], current_gap["dir"], CANVAS_SHAPE)
        if len(final_line) <= 1:
            return None
        if any(area_mask[r, c] or reserved_mask[r, c] for r, c in final_line[1:]):
            return None
        for r, c in final_line:
            reserved_mask[r, c] = True
    elif mode == "blocked":
        placed = False
        for _ in range(120):
            impact_side = OPPOSITE_SIDE[_side_from_dir(current_gap["dir"])]
            block_h = rng.randint(*BOX_SIZE_RANGE)
            block_w = rng.randint(*BOX_SIZE_RANGE)
            impact_offset = rng.choice(_offset_choices(impact_side, block_h, block_w))
            distance = rng.randint(*PATH_LEN_RANGE)
            impact_point = (
                current_gap["pos"][0] + current_gap["dir"][0] * distance,
                current_gap["pos"][1] + current_gap["dir"][1] * distance,
            )
            bbox = _bbox_from_edge_point(impact_point, impact_side, block_h, block_w, impact_offset)
            gap_count = rng.choice([1, 2])
            specs = []
            used = set()
            while len(specs) < gap_count:
                side = rng.choice(SIDES)
                offset = rng.choice(_offset_choices(side, block_h, block_w))
                if (side, offset) == (impact_side, impact_offset):
                    continue
                if (side, offset) in used:
                    continue
                used.add((side, offset))
                specs.append((side, offset))
            blocker = _sample_box(
                rng,
                bbox=bbox,
                border=border_pool[(chain_len + 1) % len(border_pool)],
                required_gaps=specs,
                fill=None,
            )
            if blocker is None:
                continue
            if not _box_area_clear(blocker["bbox"], area_mask, reserved_mask):
                continue
            segment = _line_cells(current_gap["pos"], impact_point)
            if len(segment) <= 1:
                continue
            if any(area_mask[r, c] or reserved_mask[r, c] for r, c in segment[1:-1]):
                continue
            _add_box(blocker, grid, area_mask)
            boxes.append(blocker)
            final_line = segment[:-1]
            for r, c in final_line:
                reserved_mask[r, c] = True
            placed = True
            break
        if not placed:
            return None

    maybe_pair = rng.random() < 0.65
    if maybe_pair:
        for _ in range(60):
            left_h = rng.randint(*BOX_SIZE_RANGE)
            left_w = rng.randint(*BOX_SIZE_RANGE)
            blocker = area_mask.astype(int) | reserved_mask.astype(int)
            patch = np.ones((left_h, left_w), dtype=int)
            try:
                r0, c0 = random_free_location(blocker, patch, transparent=BG, padding=1)
            except ValueError:
                break
            side = rng.choice(SIDES)
            offset = rng.choice(_offset_choices(side, left_h, left_w))
            left_bbox = (int(r0), int(c0), int(r0 + left_h - 1), int(c0 + left_w - 1))
            left_box = _sample_box(
                rng,
                bbox=left_bbox,
                border=border_pool[(len(boxes) + 1) % len(border_pool)],
                required_gaps=[(side, offset)],
                fill=None,
            )
            if left_box is None:
                continue
            right_side = OPPOSITE_SIDE[side]
            right_h = rng.randint(*BOX_SIZE_RANGE)
            right_w = rng.randint(*BOX_SIZE_RANGE)
            right_offset = rng.choice(_offset_choices(right_side, right_h, right_w))
            gap_a = left_box["gaps"][0]
            distance = rng.randint(*PATH_LEN_RANGE)
            gap_b = (
                gap_a["pos"][0] + gap_a["dir"][0] * distance,
                gap_a["pos"][1] + gap_a["dir"][1] * distance,
            )
            right_bbox = _bbox_from_edge_point(gap_b, right_side, right_h, right_w, right_offset)
            right_box = _sample_box(
                rng,
                bbox=right_bbox,
                border=border_pool[(len(boxes) + 2) % len(border_pool)],
                required_gaps=[(right_side, right_offset)],
                fill=None,
            )
            if right_box is None:
                continue
            if not _box_area_clear(left_box["bbox"], area_mask, reserved_mask):
                continue
            temp_mask = area_mask.copy()
            lr0, lc0, lr1, lc1 = left_box["bbox"]
            temp_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True
            if not _box_area_clear(right_box["bbox"], temp_mask, reserved_mask):
                continue
            segment = _line_cells(gap_a["pos"], gap_b)
            if any(area_mask[r, c] or reserved_mask[r, c] for r, c in segment[1:-1]):
                continue
            _add_box(left_box, grid, area_mask)
            _add_box(right_box, grid, area_mask)
            boxes.extend([left_box, right_box])
            for r, c in segment:
                reserved_mask[r, c] = True
            break

    min_distractors = max(1, 3 - len(boxes))
    distractor_count = rng.randint(min_distractors, 3)
    for i in range(distractor_count):
        box = _random_box_in_place(
            rng,
            area_mask,
            reserved_mask,
            border=border_pool[(len(boxes) + i) % len(border_pool)],
            gap_count=rng.choice([1, 2]),
        )
        if box is None:
            continue
        _add_box(box, grid, area_mask)
        boxes.append(box)

    if len(boxes) < 3:
        return None

    expected = grid.copy()
    for segment, target_idx in zip(chain_lines, chain_targets):
        _paint_cells(expected, segment, paint)
        r0, c0, r1, c1 = boxes[target_idx]["bbox"]
        expected[r0 + 1 : r1, c0 + 1 : c1] = paint

    if final_line is not None:
        _paint_cells(expected, final_line, paint)

    used = np.argwhere(grid != BG)
    if used.size == 0:
        return None
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top_margin = rng.randint(1, min(3, int(r0)) + 1) if r0 > 0 else 0
    left_margin = rng.randint(1, min(3, int(c0)) + 1) if c0 > 0 else 0
    bottom_margin = rng.randint(1, min(3, int(CANVAS_SHAPE[0] - 1 - r1)) + 1) if r1 < CANVAS_SHAPE[0] - 1 else 0
    right_margin = rng.randint(1, min(3, int(CANVAS_SHAPE[1] - 1 - c1)) + 1) if c1 < CANVAS_SHAPE[1] - 1 else 0
    crop = (
        max(0, int(r0) - top_margin),
        max(0, int(c0) - left_margin),
        min(CANVAS_SHAPE[0] - 1, int(r1) + bottom_margin),
        min(CANVAS_SHAPE[1] - 1, int(c1) + right_margin),
    )
    rr0, cc0, rr1, cc1 = crop
    grid = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    expected = expected[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()

    try:
        boxes_found, source_idx, _ = _extract_boxes(grid)
        if len(boxes_found) < 3:
            return None
        if not np.array_equal(main(grid), expected):
            return None
        if np.array_equal(grid, expected):
            return None
        source_gaps = len(boxes_found[source_idx]["gaps"])
        if source_gaps != 1:
            return None
    except Exception:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _build_chain(random.Random())
        if result is None:
            continue
        grid, _expected = result
        return grid
    raise RuntimeError("failed to generate a valid gap-port chain input")
