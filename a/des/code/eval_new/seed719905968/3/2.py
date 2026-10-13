import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
SCENE_MIN = 18
SCENE_MAX = 30
MAX_GENERATE_ATTEMPTS = 400
SOURCE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _make_source_specs():
    specs = []
    raw = {
        "right": [
            {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (0, 3)},
            {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (1, 3)},
        ],
        "left": [
            {(0, 2), (0, 3), (1, 2), (1, 3), (0, 1), (1, 1), (0, 0)},
            {(0, 2), (0, 3), (1, 2), (1, 3), (0, 1), (1, 1), (1, 0)},
        ],
        "down": [
            {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)},
            {(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 1)},
        ],
        "up": [
            {(2, 0), (2, 1), (3, 0), (3, 1), (1, 0), (1, 1), (0, 0)},
            {(2, 0), (2, 1), (3, 0), (3, 1), (1, 0), (1, 1), (0, 1)},
        ],
    }
    for direction, variants in raw.items():
        for variant_idx, cells in enumerate(variants):
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, c0 = min(rows), min(cols)
            normalized = {(r - r0, c - c0) for r, c in cells}
            h = max(r for r, _ in normalized) + 1
            w = max(c for _, c in normalized) + 1
            tip = None
            source_square = None
            for rr in range(h - 1):
                for cc in range(w - 1):
                    square = {(rr, cc), (rr + 1, cc), (rr, cc + 1), (rr + 1, cc + 1)}
                    if square <= normalized:
                        remaining = normalized - square
                        if len(remaining) != 3:
                            continue
                        if direction in ("left", "right"):
                            rear_cols = {c for _, c in remaining}
                            if len(rear_cols) != 2:
                                continue
                            tip_col = max(rear_cols) if direction == "right" else min(rear_cols)
                            tip_candidates = [p for p in remaining if p[1] == tip_col]
                        else:
                            rear_rows = {r for r, _ in remaining}
                            if len(rear_rows) != 2:
                                continue
                            tip_row = max(rear_rows) if direction == "down" else min(rear_rows)
                            tip_candidates = [p for p in remaining if p[0] == tip_row]
                        if len(tip_candidates) == 1:
                            tip = tip_candidates[0]
                            source_square = square
                            break
                if tip is not None:
                    break
            assert tip is not None and source_square is not None
            specs.append(
                {
                    "direction": direction,
                    "variant": variant_idx,
                    "cells": normalized,
                    "tip": tip,
                    "shape": (h, w),
                    "source_square": source_square,
                }
            )
    return specs


SOURCE_SPECS = _make_source_specs()
SPEC_BY_KEY = {(spec["direction"], spec["variant"]): spec for spec in SOURCE_SPECS}


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int, mask: np.ndarray | None = None):
    h, w = grid.shape
    allowed = np.ones((h, w), dtype=bool) if mask is None else mask.copy()
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not allowed[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR_VECTORS.values():
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not allowed[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _parse_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    assert h >= 4, "grid too short for bottom legend"
    legend_top = h - 2
    assert np.all(grid[legend_top - 1] == bg), "legend must be separated from the scene"
    row0 = grid[legend_top]
    row1 = grid[legend_top + 1]
    assert np.array_equal(row0, row1), "legend rows must match"
    mapping = {}
    c = 0
    while c < w:
        if int(row0[c]) == bg:
            c += 1
            continue
        assert c + 1 < w and int(row0[c + 1]) != bg, "legend entries must be adjacent pairs"
        left = int(row0[c])
        right = int(row0[c + 1])
        mapping[left] = right
        c += 2
        while c < w and int(row0[c]) == bg:
            c += 1
    assert len(mapping) >= 2, "legend must contain at least two color mappings"
    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[legend_top : legend_top + 2, :] = True
    return mapping, legend_mask, legend_top


def _match_source_component(comp):
    coords = comp["coords"]
    if len(coords) != 7:
        return None
    r0, c0, _r1, _c1 = comp["bbox"]
    normalized = {(r - r0, c - c0) for r, c in coords}
    for spec in SOURCE_SPECS:
        if normalized == spec["cells"]:
            tip = (r0 + spec["tip"][0], c0 + spec["tip"][1])
            return {
                "color": comp["color"],
                "direction": spec["direction"],
                "tip": tip,
                "cells": set(coords),
            }
    return None


def _match_frame_component(comp, grid: np.ndarray, bg: int):
    coords = set(comp["coords"])
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    if coords != border:
        return None
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or np.any(interior != bg):
        return None
    return {
        "color": comp["color"],
        "bbox": comp["bbox"],
        "border": coords,
        "interior_slice": (slice(r0 + 1, r1), slice(c0 + 1, c1)),
    }


def _match_filter_component(comp):
    coords = comp["coords"]
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(coords) < 3:
        return None
    if len(rows) == 1 or len(cols) == 1:
        return {"coords": set(coords), "color": comp["color"]}
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate colored rays from arrow-marked source blocks, transform them at filter lines, and fill the hit containers.

    Input structure:
    - bg = most common color.
    - Several 2x2 monochrome source blocks each belong to a 7-cell same-color source+arrow sprite; the arrow tip points up, down, left, or right.
    - Straight monochrome line segments of other colors act as filters, and several hollow rectangular frames act as target containers.
    - A two-row legend sits at the bottom: each entry shows an input color immediately left of its transformed output color.

    Rule:
    - For every source sprite, launch a ray from the arrow tip in the pointing direction.
    - Moving through background keeps the current ray color unchanged.
    - Each time the ray crosses any filter segment, update the current color by the bottom legend mapping, applying the mapping once per crossed segment.
    - When the ray first reaches a hollow frame, fill that frame's interior with the ray's current color. Do not draw the ray itself; keep the rest of the scene unchanged.

    Key insight:
    - A filter line's own color is irrelevant: every filter merely triggers one application of the shared bottom color-mapping legend, so the final fill depends on arrow direction plus how many filters were crossed.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    mapping, legend_mask, legend_top = _parse_legend(g, bg)

    scene_mask = np.ones_like(g, dtype=bool)
    scene_mask[legend_mask] = False
    if legend_top - 1 >= 0:
        scene_mask[legend_top - 1, :] = False

    comps = _components(g, bg=bg, mask=scene_mask)
    sources = []
    source_cells = set()
    frames = []
    frame_cells = set()
    used = set()

    for idx, comp in enumerate(comps):
        src = _match_source_component(comp)
        if src is not None:
            sources.append(src)
            source_cells |= src["cells"]
            used.add(idx)

    for idx, comp in enumerate(comps):
        if idx in used:
            continue
        frame = _match_frame_component(comp, g, bg)
        if frame is not None:
            frames.append(frame)
            frame_cells |= frame["border"]
            used.add(idx)

    filters = []
    for idx, comp in enumerate(comps):
        if idx in used:
            continue
        filt = _match_filter_component(comp)
        if filt is not None:
            filters.append(filt)
            used.add(idx)

    assert sources, "expected at least one source-arrow sprite"
    assert frames, "expected at least one hollow target frame"
    assert filters, "expected at least one filter line"

    frame_lookup = {}
    for frame_id, frame in enumerate(frames):
        for cell in frame["border"]:
            frame_lookup[cell] = frame_id

    filter_lookup = {}
    for filter_id, filt in enumerate(filters):
        for cell in filt["coords"]:
            filter_lookup[cell] = filter_id

    out = g.copy()
    filled_frames = set()
    h, w = g.shape

    for source in sources:
        cur = source["color"]
        dr, dc = DIR_VECTORS[source["direction"]]
        r, c = source["tip"]
        last_filter = None
        while True:
            r += dr
            c += dc
            assert 0 <= r < h and 0 <= c < w, "ray exited the grid before reaching a target frame"
            cell = (r, c)

            if cell in filter_lookup:
                filter_id = filter_lookup[cell]
                if filter_id != last_filter:
                    assert cur in mapping, "ray color must be present in the legend mapping"
                    cur = mapping[cur]
                last_filter = filter_id
                continue

            last_filter = None

            if cell in frame_lookup:
                frame_id = frame_lookup[cell]
                rs, cs = frames[frame_id]["interior_slice"]
                out[rs, cs] = cur
                filled_frames.add(frame_id)
                break

            val = int(g[cell])
            assert val == bg, "ray path must only contain background, filters, and target frames"

    assert len(filled_frames) == len(frames), "every frame must be reached by exactly one ray"
    return out


def _build_mapping(source_colors, rng: random.Random):
    order = list(source_colors)
    rng.shuffle(order)
    return {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}


def _paint_cells(grid: np.ndarray, cells, color: int):
    for r, c in cells:
        grid[r, c] = color


def _frame_cells(r0: int, c0: int, height: int, width: int):
    return {
        (r, c)
        for r in range(r0, r0 + height)
        for c in range(c0, c0 + width)
        if r in (r0, r0 + height - 1) or c in (c0, c0 + width - 1)
    }


def _can_place(cells, occupied: np.ndarray):
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in cells:
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _mark_occupied(cells, occupied: np.ndarray):
    for r, c in cells:
        occupied[r, c] = True


def _lane_positions(limit: int, rng: random.Random):
    bases = list(range(2, limit - 2, 5))
    lanes = []
    for base in bases:
        shift_max = 1 if base + 1 <= limit - 3 else 0
        lanes.append(base + rng.randint(0, shift_max))
    return lanes


def _lane_capacity(limit: int):
    return len(range(2, limit - 2, 5))


def _choose_filter_positions(start: int, stop: int, count: int, rng: random.Random):
    candidates = []
    for offset in (0, 1):
        seq = list(range(start + offset, stop + 1, 2))
        if len(seq) > len(candidates):
            candidates = seq
    if len(candidates) < count:
        return None
    chosen = sorted(rng.sample(candidates, count))
    return chosen


def _filter_slot_capacity(start: int, stop: int):
    best = 0
    for offset in (0, 1):
        seq = list(range(start + offset, stop + 1, 2))
        best = max(best, len(seq))
    return best


def _paint_legend(grid: np.ndarray, mapping: dict[int, int], rng: random.Random):
    h, w = grid.shape
    entries = list(mapping.items())
    rng.shuffle(entries)
    gaps = [rng.randint(1, 2) for _ in range(max(0, len(entries) - 1))]
    total_width = 2 * len(entries) + sum(gaps)
    left_margin = rng.randint(1, max(1, w - total_width - 1))
    c = left_margin
    for idx, (src, dst) in enumerate(entries):
        grid[h - 2 : h, c] = src
        grid[h - 2 : h, c + 1] = dst
        c += 2
        if idx < len(gaps):
            c += gaps[idx]


def _place_horizontal_family(grid: np.ndarray, mapping: dict[int, int], rng: random.Random):
    h, w = grid.shape
    scene_h = h - 3
    lanes = _lane_positions(scene_h, rng)
    if len(lanes) < 2:
        return None
    n_sources = len(mapping)
    lane_rows = sorted(rng.sample(lanes, n_sources))
    colors = list(mapping.keys())
    rng.shuffle(colors)
    occupied = np.zeros((scene_h, w), dtype=bool)
    filter_counts = []

    for lane, color in zip(lane_rows, colors):
        direction = rng.choice(["left", "right"])
        variant = rng.randint(0, 1)
        spec = SPEC_BY_KEY[(direction, variant)]
        tip_r_rel, tip_c_rel = spec["tip"]
        comp_h, comp_w = spec["shape"]
        bbox_r0 = lane - tip_r_rel
        if not (0 <= bbox_r0 and bbox_r0 + comp_h <= scene_h):
            return None

        if direction == "right":
            bbox_c0 = rng.randint(1, 3)
        else:
            bbox_c0 = rng.randint(w - comp_w - 4, w - comp_w - 2)
        source_cells = {(bbox_r0 + r, bbox_c0 + c) for r, c in spec["cells"]}
        if not _can_place(source_cells, occupied):
            return None

        tip_r = bbox_r0 + tip_r_rel
        tip_c = bbox_c0 + tip_c_rel
        frame_h = rng.randint(3, 4)
        frame_w = rng.randint(3, 5)
        inside_row = rng.randint(1, frame_h - 2)
        frame_r0 = lane - inside_row
        if not (0 <= frame_r0 and frame_r0 + frame_h <= scene_h):
            return None

        if direction == "right":
            min_frame_c0 = max(tip_c + 6, w // 2)
            max_frame_c0 = w - frame_w - 2
        else:
            min_frame_c0 = 1
            max_frame_c0 = min(w // 2 - frame_w, tip_c - frame_w - 6)
        if min_frame_c0 > max_frame_c0:
            return None
        frame_c0 = rng.randint(min_frame_c0, max_frame_c0)
        frame_cells = _frame_cells(frame_r0, frame_c0, frame_h, frame_w)
        if not _can_place(frame_cells, occupied):
            return None

        if direction == "right":
            filter_start = tip_c + 2
            filter_stop = frame_c0 - 2
        else:
            filter_start = frame_c0 + frame_w + 1
            filter_stop = tip_c - 2
        if filter_start > filter_stop:
            return None

        max_filters = min(3, _filter_slot_capacity(filter_start, filter_stop))
        count = rng.randint(0, max_filters)
        positions = _choose_filter_positions(filter_start, filter_stop, count, rng)
        if positions is None:
            return None

        filter_cells = []
        for col in positions:
            filter_cells.extend([(lane - 1, col), (lane, col), (lane + 1, col)])
        if filter_cells and not _can_place(filter_cells, occupied):
            return None

        _paint_cells(grid, source_cells, color)
        _mark_occupied(source_cells, occupied)
        _paint_cells(grid, frame_cells, FRAME)
        _mark_occupied(frame_cells, occupied)
        for col in positions:
            filter_color = rng.choice([v for v in SOURCE_COLOR_POOL if v not in mapping and v != FRAME])
            cells = [(lane - 1, col), (lane, col), (lane + 1, col)]
            _paint_cells(grid, cells, filter_color)
            _mark_occupied(cells, occupied)
        filter_counts.append(count)

    return filter_counts


def _place_vertical_family(grid: np.ndarray, mapping: dict[int, int], rng: random.Random):
    h, w = grid.shape
    scene_h = h - 3
    lanes = _lane_positions(w, rng)
    if len(lanes) < 2:
        return None
    n_sources = len(mapping)
    lane_cols = sorted(rng.sample(lanes, n_sources))
    colors = list(mapping.keys())
    rng.shuffle(colors)
    occupied = np.zeros((scene_h, w), dtype=bool)
    filter_counts = []

    for lane, color in zip(lane_cols, colors):
        direction = rng.choice(["up", "down"])
        variant = rng.randint(0, 1)
        spec = SPEC_BY_KEY[(direction, variant)]
        tip_r_rel, tip_c_rel = spec["tip"]
        comp_h, comp_w = spec["shape"]
        bbox_c0 = lane - tip_c_rel
        if not (0 <= bbox_c0 and bbox_c0 + comp_w <= w):
            return None

        if direction == "down":
            bbox_r0 = rng.randint(1, 3)
        else:
            bbox_r0 = rng.randint(scene_h - comp_h - 4, scene_h - comp_h - 2)
        source_cells = {(bbox_r0 + r, bbox_c0 + c) for r, c in spec["cells"]}
        if not _can_place(source_cells, occupied):
            return None

        tip_r = bbox_r0 + tip_r_rel
        tip_c = bbox_c0 + tip_c_rel
        frame_h = rng.randint(3, 5)
        frame_w = rng.randint(3, 4)
        inside_col = rng.randint(1, frame_w - 2)
        frame_c0 = lane - inside_col
        if not (0 <= frame_c0 and frame_c0 + frame_w <= w):
            return None

        if direction == "down":
            min_frame_r0 = max(tip_r + 6, scene_h // 2 - 1)
            max_frame_r0 = scene_h - frame_h - 2
        else:
            min_frame_r0 = 1
            max_frame_r0 = min(scene_h // 2 - frame_h, tip_r - frame_h - 6)
        if min_frame_r0 > max_frame_r0:
            return None
        frame_r0 = rng.randint(min_frame_r0, max_frame_r0)
        frame_cells = _frame_cells(frame_r0, frame_c0, frame_h, frame_w)
        if not _can_place(frame_cells, occupied):
            return None

        if direction == "down":
            filter_start = tip_r + 2
            filter_stop = frame_r0 - 2
        else:
            filter_start = frame_r0 + frame_h + 1
            filter_stop = tip_r - 2
        if filter_start > filter_stop:
            return None

        max_filters = min(3, _filter_slot_capacity(filter_start, filter_stop))
        count = rng.randint(0, max_filters)
        positions = _choose_filter_positions(filter_start, filter_stop, count, rng)
        if positions is None:
            return None

        filter_cells = []
        for row in positions:
            filter_cells.extend([(row, lane - 1), (row, lane), (row, lane + 1)])
        if filter_cells and not _can_place(filter_cells, occupied):
            return None

        _paint_cells(grid, source_cells, color)
        _mark_occupied(source_cells, occupied)
        _paint_cells(grid, frame_cells, FRAME)
        _mark_occupied(frame_cells, occupied)
        for row in positions:
            filter_color = rng.choice([v for v in SOURCE_COLOR_POOL if v not in mapping and v != FRAME])
            cells = [(row, lane - 1), (row, lane), (row, lane + 1)]
            _paint_cells(grid, cells, filter_color)
            _mark_occupied(cells, occupied)
        filter_counts.append(count)

    return filter_counts


def _gen_attempt(rng: random.Random):
    h = rng.randint(SCENE_MIN, SCENE_MAX)
    w = rng.randint(SCENE_MIN, SCENE_MAX)
    grid = np.full((h, w), BG, dtype=int)

    family = "horizontal" if rng.random() < 0.5 else "vertical"
    if family == "horizontal":
        max_sources = min(4, _lane_capacity(h - 3))
    else:
        max_sources = min(4, _lane_capacity(w))
    if max_sources < 2:
        return None
    n_sources = rng.randint(2, max_sources)
    source_colors = rng.sample(SOURCE_COLOR_POOL, n_sources)
    mapping = _build_mapping(source_colors, rng)
    _paint_legend(grid, mapping, rng)

    if family == "horizontal":
        filter_counts = _place_horizontal_family(grid, mapping, rng)
    else:
        filter_counts = _place_vertical_family(grid, mapping, rng)
    if filter_counts is None or max(filter_counts, default=0) == 0:
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
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
