from __future__ import annotations

from collections import Counter
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return bool(np.any((~mask) & (~outside)))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    objects: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        seen = np.zeros((h, w), dtype=bool)
        for r in range(h):
            for c in range(w):
                if seen[r, c] or int(grid[r, c]) != color:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for nr, nc in _neighbors4(rr, cc):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))

                coords = np.array(cells, dtype=int)
                bbox = _bbox(coords)
                mask = _component_mask(coords, bbox)
                patch = np.full(mask.shape, bg, dtype=int)
                patch[mask] = color
                objects.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": bbox,
                        "mask": mask,
                        "patch": patch,
                        "h": int(mask.shape[0]),
                        "w": int(mask.shape[1]),
                        "closed": _has_enclosed_hole(mask),
                    }
                )
    return objects


def _settle_group(objects: list[dict], grid_shape: tuple[int, int], upward: bool) -> list[dict]:
    h, _w = grid_shape
    occupied = np.zeros(grid_shape, dtype=bool)
    settled: list[dict] = []
    if upward:
        ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]))
    else:
        ordered = sorted(objects, key=lambda obj: (-obj["bbox"][2], obj["bbox"][1], -obj["bbox"][0], obj["bbox"][3]))

    for obj in ordered:
        c0 = int(obj["bbox"][1])
        rows = range(0, h - obj["h"] + 1) if upward else range(h - obj["h"], -1, -1)
        row = None
        for candidate in rows:
            region = occupied[candidate : candidate + obj["h"], c0 : c0 + obj["w"]]
            if not bool(np.any(region & obj["mask"])):
                row = int(candidate)
                region[obj["mask"]] = True
                break
        assert row is not None, "expected a valid slide destination"
        new_bbox = (row, c0, row + obj["h"] - 1, c0 + obj["w"] - 1)
        new_obj = dict(obj)
        new_obj["settled_row"] = int(row)
        new_obj["settled_bbox"] = new_bbox
        settled.append(new_obj)
    return settled


def _paint_objects(shape: tuple[int, int], objects: list[dict], bg: int) -> np.ndarray:
    canvas = np.full(shape, bg, dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["settled_bbox"]
        region = canvas[r0 : r0 + obj["h"], c0 : c0 + obj["w"]]
        region[obj["mask"]] = obj["color"]
    return canvas


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects, "expected foreground objects"

    closed = [obj for obj in objects if obj["closed"]]
    open_objects = [obj for obj in objects if not obj["closed"]]
    assert 2 <= len(closed) <= 3, "expected 2-3 closed objects"
    assert 3 <= len(open_objects) <= 5, "expected 3-5 open objects"

    settled_closed = _settle_group(closed, g.shape, upward=True)
    settled_open = _settle_group(open_objects, g.shape, upward=False)
    open_canvas = _paint_objects(g.shape, settled_open, bg)

    counts: list[int] = []
    for obj in settled_closed:
        r0, c0, r1, c1 = obj["settled_bbox"]
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "closed objects must have a non-empty strict interior"
        interior = open_canvas[r0 + 1 : r1, c0 + 1 : c1]
        counts.append(int(np.count_nonzero(interior != bg)))

    best = max(counts)
    winners = [idx for idx, count in enumerate(counts) if count == best]
    assert len(winners) == 1, "expected a unique frame with maximal interior overlap"
    winner = settled_closed[winners[0]]
    wr0, wc0, wr1, wc1 = winner["settled_bbox"]
    out = open_canvas[wr0 + 1 : wr1, wc0 + 1 : wc1].copy()

    return out, {
        "bg": bg,
        "closed": settled_closed,
        "open": settled_open,
        "open_canvas": open_canvas,
        "counts": counts,
        "winner_index": int(winners[0]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the interior of the frame that contains the most sprite pixels after independent gravity stacking.

    Input structure:
    - The background is black and the foreground is split into monochrome 4-connected objects.
    - Closed objects are one-cell-thick hollow rectangular frames whose bounding boxes contain an enclosed background hole.
    - Open objects are small hole-free sprites that keep their shapes and horizontal positions.

    Rule:
    - Classify each object by whether it encloses a hole.
    - Slide closed objects upward, processing them from top to bottom, until each hits the top edge or a previously settled closed object.
    - Slide open objects downward, processing them from bottom to top, until each hits the bottom edge or a previously settled open object.
    - On the stacked layout, count how many open-object pixels lie inside the strict interior of each closed frame, select the unique maximum, and output that frame's strict interior with the border removed.

    Key insight:
    - The winning crop cannot be read from the input directly because the decisive sprite/frame overlap is created only after the two object classes are stacked in opposite directions.
    """
    return _solve(grid)[0]


_OPEN_BASE_SHAPES = [
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (2, 0), (3, 0), (3, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)},
]

_GEN_ATTEMPTS = 800


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_cells(cells: set[tuple[int, int]], rot: int, flip: bool) -> set[tuple[int, int]]:
    cur = set(cells)
    for _ in range(rot):
        cur = {(-c, r) for r, c in cur}
        cur = _normalize(cur)
    if flip:
        max_c = max(c for _r, c in cur)
        cur = {(r, max_c - c) for r, c in cur}
        cur = _normalize(cur)
    return cur


def _shape_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_open_mask() -> np.ndarray:
    base = random.choice(_OPEN_BASE_SHAPES)
    variants = []
    for rot in range(4):
        for flip in (False, True):
            cells = _transform_cells(base, rot, flip)
            if (0, 0) in cells:
                variants.append(cells)
    cells = random.choice(variants)
    mask = _shape_to_mask(cells)
    assert not _has_enclosed_hole(mask)
    return mask


def _frame_patch(height: int, width: int, color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch, patch != 0


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _can_place_mask(occupied: np.ndarray, spec: dict, row: int) -> bool:
    region = occupied[row : row + spec["h"], spec["col"] : spec["col"] + spec["w"]]
    return not bool(np.any(region & spec["mask"]))


def _place_mask(occupied: np.ndarray, spec: dict, row: int) -> None:
    region = occupied[row : row + spec["h"], spec["col"] : spec["col"] + spec["w"]]
    region[spec["mask"]] = True


def _search_input_rows(h: int, frame_specs: list[dict], open_specs: list[dict]) -> bool:
    occupied = np.zeros((h, max([spec["col"] + spec["w"] for spec in frame_specs + open_specs], default=0)), dtype=bool)
    used_change = {"value": False}

    def assign_group(specs: list[dict], prefer_desc: bool, allow_same_final: bool) -> bool:
        def rec(idx: int, prev_row: int) -> bool:
            if idx == len(specs):
                return True
            spec = specs[idx]
            start = spec["final_row"] if allow_same_final else (spec["final_row"] + 1 if prefer_desc else spec["final_row"] - 1)
            if prefer_desc:
                lo = max(prev_row + 1, spec["final_row"])
                rows = list(range(h - spec["h"], lo - 1, -1))
            else:
                hi = min(spec["final_row"], h - spec["h"])
                if start < 0:
                    start = hi
                rows = list(range(0, hi + 1))
            random.shuffle(rows)
            rows.sort(reverse=prefer_desc)
            for row in rows:
                if row <= prev_row:
                    continue
                if not _can_place_mask(occupied, spec, row):
                    continue
                _place_mask(occupied, spec, row)
                spec["input_row"] = int(row)
                moved_now = row != spec["final_row"]
                old_changed = used_change["value"]
                used_change["value"] = old_changed or moved_now
                if rec(idx + 1, row):
                    return True
                region = occupied[row : row + spec["h"], spec["col"] : spec["col"] + spec["w"]]
                region[spec["mask"]] = False
                used_change["value"] = old_changed
            return False

        return rec(0, -1)

    if not assign_group(frame_specs, prefer_desc=True, allow_same_final=True):
        return False
    if not assign_group(open_specs, prefer_desc=False, allow_same_final=True):
        return False
    return bool(used_change["value"])


def _render_input(shape: tuple[int, int], frame_specs: list[dict], open_specs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in frame_specs + open_specs:
        row = spec["input_row"]
        region = grid[row : row + spec["h"], spec["col"] : spec["col"] + spec["w"]]
        region[spec["mask"]] = spec["color"]
    return grid


def _semantic_check(inp: np.ndarray, expected: np.ndarray, counts: list[int]) -> bool:
    try:
        out, meta = _solve(inp)
    except Exception:
        return False
    if not np.array_equal(out, expected):
        return False
    solved_counts = meta["counts"]
    if solved_counts != counts:
        return False
    positive = [count for count in solved_counts if count > 0]
    if len(positive) < 2 or len(set(positive)) < 2:
        return False
    if solved_counts.count(max(solved_counts)) != 1:
        return False
    return True


def _random_hostable_open_mask() -> np.ndarray:
    candidates = []
    tall_candidates = []
    for _ in range(80):
        mask = _random_open_mask()
        if mask.shape[0] <= 2 and mask.shape[1] <= 4:
            candidates.append(mask)
            if mask.shape[0] == 2:
                tall_candidates.append(mask)
    assert candidates
    if tall_candidates and random.random() < 0.75:
        return random.choice(tall_candidates)
    return random.choice(candidates)


def _build_specs() -> tuple[int, int, list[dict], list[dict]] | None:
    n_frames = 3 if random.random() < 0.85 else 2
    n_open = random.randint(4, 5) if n_frames == 2 else random.randint(3, 5)

    open_colors = random.sample(range(1, 10), n_open)
    open_specs = []
    max_open_w = 0
    total_open_h = 0
    for color in open_colors:
        mask = _random_hostable_open_mask()
        patch = _mask_to_patch(mask, color)
        oh, ow = mask.shape
        open_specs.append({"color": color, "patch": patch, "mask": mask, "h": oh, "w": ow})
        max_open_w = max(max_open_w, ow)
        total_open_h += oh

    frame_colors = random.sample([color for color in range(1, 10) if color not in open_colors], n_frames)
    frame_specs = []
    total_frame_h = 0
    min_frame_w = 99
    for color in frame_colors:
        fh = random.randint(4, 5)
        fw = random.randint(max(5, max_open_w + 2), 7)
        patch, mask = _frame_patch(fh, fw, color)
        frame_specs.append({"color": color, "patch": patch, "mask": mask, "h": fh, "w": fw})
        total_frame_h += fh
        min_frame_w = min(min_frame_w, fw)

    overlap = random.randint(5, 7) if n_frames == 3 else random.randint(4, 6)
    h = total_frame_h + total_open_h - overlap
    if not (14 <= h <= 20):
        return None
    min_w = max(14, max(max(spec["w"] for spec in frame_specs), max_open_w + 2) + 1)
    if min_w > 18:
        return None
    w = random.randint(min_w, 18)

    return h, w, frame_specs, open_specs


def _place_frames(h: int, w: int, frame_specs: list[dict], open_specs: list[dict]) -> bool:
    max_frame_w = max(spec["w"] for spec in frame_specs)
    max_open_w = max(spec["w"] for spec in open_specs)
    min_frame_w = min(spec["w"] for spec in frame_specs)

    frame_col = random.randint(0, w - max_frame_w)
    open_min = frame_col + 1
    open_max = frame_col + min_frame_w - max_open_w - 1
    if open_min > open_max:
        return False
    open_col = random.randint(open_min, open_max)

    total_frame_span = sum(spec["h"] for spec in frame_specs)
    top_slack = h - total_frame_span
    if top_slack < 1:
        return False
    start_row = random.randint(max(1, top_slack - 2), top_slack)

    cursor = start_row
    random.shuffle(frame_specs)
    for spec in frame_specs:
        spec["col"] = frame_col
        spec["input_row"] = cursor
        cursor += spec["h"]

    random.shuffle(open_specs)
    for spec in open_specs:
        spec["col"] = open_col
    return True


def _candidate_open_rows(h: int, frame_specs: list[dict], spec: dict, occupied: np.ndarray) -> list[int]:
    rows = []
    for row in range(0, h - spec["h"] + 1):
        if _can_place_mask(occupied, spec, row):
            rows.append(row)

    def score(row: int) -> tuple[int, int, int]:
        inside = 0
        margin = 9
        for frame in frame_specs:
            lo = frame["input_row"] + 1
            hi = frame["input_row"] + frame["h"] - spec["h"] - 1
            if lo <= row <= hi:
                inside = 1
                margin = row - lo
                break
        return (-inside, margin, row)

    rows.sort(key=score)
    return rows


def _place_opens(h: int, w: int, frame_specs: list[dict], open_specs: list[dict]) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for spec in frame_specs:
        _place_mask(occupied, spec, spec["input_row"])

    def rec(idx: int, prev_row: int) -> bool:
        if idx == len(open_specs):
            return True
        spec = open_specs[idx]
        for row in _candidate_open_rows(h, frame_specs, spec, occupied):
            if row <= prev_row:
                continue
            _place_mask(occupied, spec, row)
            spec["input_row"] = row
            if rec(idx + 1, row):
                return True
            region = occupied[row : row + spec["h"], spec["col"] : spec["col"] + spec["w"]]
            region[spec["mask"]] = False
        return False

    return rec(0, -1)


def _movement_checks(inp: np.ndarray, meta: dict) -> bool:
    by_color_in = {}
    for obj in _find_components(inp, 0):
        by_color_in[obj["color"]] = obj

    closed_moved = 0
    open_moved = 0
    for obj in meta["closed"]:
        if by_color_in[obj["color"]]["bbox"][0] != obj["settled_row"]:
            closed_moved += 1
    for obj in meta["open"]:
        if by_color_in[obj["color"]]["bbox"][0] != obj["settled_row"]:
            open_moved += 1
    return closed_moved >= 1 and open_moved >= 1


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        built = _build_specs()
        if built is None:
            continue
        h, w, frame_specs, open_specs = built
        if not _place_frames(h, w, frame_specs, open_specs):
            continue
        if not _place_opens(h, w, frame_specs, open_specs):
            continue
        inp = _render_input((h, w), frame_specs, open_specs)
        try:
            out, meta = _solve(inp)
        except Exception:
            continue
        counts = meta["counts"]
        positive = [count for count in counts if count > 0]
        if len(positive) < 2 or len(set(positive)) < 2:
            continue
        if counts.count(max(counts)) != 1:
            continue
        if out.size == 0 or not np.any(out != 0):
            continue
        if max(out.shape) > 30:
            continue
        if not _movement_checks(inp, meta):
            continue
        if _semantic_check(inp, out, counts):
            return inp
    raise RuntimeError("failed to generate a valid input for ARC puzzle 046/3")
