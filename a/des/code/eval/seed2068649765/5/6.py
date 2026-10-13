from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


Coord = tuple[int, int]


def _neighbors(r: int, c: int, connectivity: int) -> list[Coord]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bg_color(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coord_set(coords: np.ndarray) -> set[Coord]:
    return {tuple(map(int, rc)) for rc in coords}


def _segment_info(g: np.ndarray, bg: int, color: int, coords: np.ndarray) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    size = int(coords.shape[0])
    if size < 4:
        return None
    orientation: str
    step: Coord
    end0: Coord
    end1: Coord
    if r0 == r1 and size == c1 - c0 + 1:
        orientation = "h"
        step = (0, 1)
        end0 = (r0, c0)
        end1 = (r0, c1)
    elif c0 == c1 and size == r1 - r0 + 1:
        orientation = "v"
        step = (1, 0)
        end0 = (r0, c0)
        end1 = (r1, c0)
    else:
        return None

    ext0 = (end0[0] - step[0], end0[1] - step[1])
    ext1 = (end1[0] + step[0], end1[1] + step[1])
    h, w = g.shape

    ext_cells: list[tuple[Coord, Coord]] = []
    for endpoint, ext in ((end0, ext0), (end1, ext1)):
        er, ec = ext
        if 0 <= er < h and 0 <= ec < w and int(g[er, ec]) != bg:
            ext_cells.append((endpoint, ext))

    if len(ext_cells) != 1:
        return None

    marked, marker = ext_cells[0]
    unmarked = end1 if marked == end0 else end0
    direction = (marked[0] - unmarked[0], marked[1] - unmarked[1])

    return {
        "color": int(color),
        "coords": coords.copy(),
        "cells": _coord_set(coords),
        "orientation": orientation,
        "length": size,
        "marked": marked,
        "unmarked": unmarked,
        "marker": marker,
        "dir": direction,
    }


def _parse_scene(g: np.ndarray) -> dict[str, object]:
    bg = _bg_color(g)
    h, w = g.shape

    segments: list[dict[str, object]] = []
    segment_cells: set[Coord] = set()
    marker_cells: set[Coord] = set()

    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        for comp in _components(g == color, connectivity=4):
            info = _segment_info(g, bg, color, comp)
            if info is None:
                continue
            segments.append(info)
            segment_cells |= info["cells"]
            marker_cells.add(tuple(map(int, info["marker"])))

    assert segments, "expected conveyor segments"

    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)
    for seg in segments:
        by_color[int(seg["color"])].append(seg)
    assert all(len(items) == 2 for items in by_color.values()), "each conveyor color must have exactly two segments"

    protected = np.zeros((h, w), dtype=bool)
    for r, c in segment_cells | marker_cells:
        protected[r, c] = True

    sprite_mask = (g != bg) & (~protected)
    sprite_components = _components(sprite_mask, connectivity=8)
    assert sprite_components, "expected at least one sprite"

    sprites: list[dict[str, object]] = []
    for idx, coords in enumerate(sprite_components):
        r0, c0, r1, c1 = _bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        assert side_h == side_w and side_h in (2, 3), "sprites must fit a 2x2 or 3x3 square"

        attachments: list[tuple[int, Coord]] = []
        for seg_idx, seg in enumerate(segments):
            sr, sc = map(int, seg["unmarked"])
            for pr, pc in coords:
                if abs(int(pr) - sr) + abs(int(pc) - sc) == 1:
                    attachments.append((seg_idx, (int(pr), int(pc))))

        assert len(attachments) == 1, "each sprite must attach to exactly one unmarked endpoint"
        seg_idx, contact = attachments[0]
        seg = segments[seg_idx]
        values = g[coords[:, 0], coords[:, 1]].astype(int).copy()
        offset = (contact[0] - int(seg["unmarked"][0]), contact[1] - int(seg["unmarked"][1]))

        sprites.append(
            {
                "id": idx,
                "coords": coords.copy(),
                "values": values,
                "contact": contact,
                "offset": offset,
                "source_index": seg_idx,
                "source_color": int(seg["color"]),
                "bbox": (r0, c0, r1, c1),
                "arrival": int(seg["length"]),
            }
        )

    return {
        "bg": bg,
        "segments": segments,
        "by_color": by_color,
        "sprites": sprites,
        "segment_cells": segment_cells,
        "marker_cells": marker_cells,
    }


def _translate(coords: np.ndarray, dr: int, dc: int) -> np.ndarray:
    return coords + np.array([dr, dc], dtype=int)


def _valid_position(coords: np.ndarray, fixed: np.ndarray, stopped: np.ndarray) -> tuple[bool, bool]:
    h, w = fixed.shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    if bool(np.any(rows < 0) or np.any(rows >= h) or np.any(cols < 0) or np.any(cols >= w)):
        return False, False
    if bool(np.any(fixed[rows, cols])):
        return False, False
    if bool(np.any(stopped[rows, cols])):
        return False, True
    return True, False


def _solve(g: np.ndarray, return_meta: bool = False):
    scene = _parse_scene(g)
    bg = int(scene["bg"])
    segments = scene["segments"]
    by_color = scene["by_color"]
    sprites = scene["sprites"]

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    fixed = out != bg
    stopped = np.zeros_like(fixed, dtype=bool)
    had_collision = False

    def target_segment(source_index: int, color: int) -> dict[str, object]:
        pair = by_color[int(color)]
        if pair[0] is segments[source_index]:
            return pair[1]
        return pair[0]

    order = sorted(
        sprites,
        key=lambda s: (
            int(s["arrival"]),
            int(s["bbox"][0]),
            int(s["bbox"][1]),
            int(s["id"]),
        ),
    )

    for sprite in order:
        target = target_segment(int(sprite["source_index"]), int(sprite["source_color"]))
        move = (-int(target["dir"][0]), -int(target["dir"][1]))
        contact0 = tuple(map(int, sprite["contact"]))
        offset = tuple(map(int, sprite["offset"]))

        last_valid: np.ndarray | None = None
        for step_idx in range(int(target["length"])):
            line_cell = (
                int(target["marked"][0]) + step_idx * move[0],
                int(target["marked"][1]) + step_idx * move[1],
            )
            new_contact = (line_cell[0] + offset[0], line_cell[1] + offset[1])
            dr = new_contact[0] - contact0[0]
            dc = new_contact[1] - contact0[1]
            shifted = _translate(sprite["coords"], dr, dc)
            ok, hit_stopped = _valid_position(shifted, fixed, stopped)
            if not ok:
                had_collision = had_collision or hit_stopped
                break
            last_valid = shifted

        assert last_valid is not None, "sprite must be placeable on paired segment"
        vals = sprite["values"]
        out[last_valid[:, 0], last_valid[:, 1]] = vals
        stopped[last_valid[:, 0], last_valid[:, 1]] = True

    if return_meta:
        return out, {"had_collision": had_collision, "scene": scene}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Conveyor segments carry nearby sprites toward grey-marked endpoints, then teleport
    them to the other same-colored segment and send them back toward that segment's
    unmarked end. Stopped sprites block later arrivals.
    """
    g = np.asarray(grid, dtype=int)
    return _solve(g, return_meta=False)


# --------------- generator ---------------

BG = 0
GREY = 5
LINE_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
PAIR_COUNT_RANGE = (2, 3)
LINE_LEN_RANGE = (4, 8)
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


def _expand(cells: set[Coord], h: int, w: int, radius: int) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _paint_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray) -> set[Coord]:
    h, w = patch.shape
    grid[top : top + h, left : left + w] = patch
    return {(top + r, left + c) for r in range(h) for c in range(w)}


def _make_patch(size: int, palette: list[int]) -> np.ndarray:
    k = random.randint(2, min(4, len(palette)))
    colors = random.sample(palette, k)
    patch = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            patch[r, c] = random.choice(colors)
    if len(set(map(int, patch.ravel()))) == 1:
        patch[size - 1, size - 1] = colors[1]
    return patch


def _segment_cells(
    orientation: str,
    anchor_r: int,
    anchor_c: int,
    length: int,
    marked_side: str,
) -> tuple[list[Coord], Coord, Coord, Coord]:
    if orientation == "h":
        cells = [(anchor_r, anchor_c + i) for i in range(length)]
        left = cells[0]
        right = cells[-1]
        if marked_side == "left":
            return cells, (anchor_r, anchor_c - 1), right, left
        return cells, (anchor_r, anchor_c + length), left, right

    cells = [(anchor_r + i, anchor_c) for i in range(length)]
    top = cells[0]
    bottom = cells[-1]
    if marked_side == "top":
        return cells, (anchor_r - 1, anchor_c), bottom, top
    return cells, (anchor_r + length, anchor_c), top, bottom


def _source_offset_and_contact(size: int, orientation: str, preferred: Coord | None = None) -> tuple[Coord, Coord]:
    if orientation == "h":
        choices = [((-1, 0), (size - 1, c)) for c in range(size)] + [((1, 0), (0, c)) for c in range(size)]
    else:
        choices = [((0, -1), (r, size - 1)) for r in range(size)] + [((0, 1), (r, 0)) for r in range(size)]
    if preferred is not None:
        choices = [item for item in choices if item[0] == preferred]
    assert choices
    return random.choice(choices)


def _line_spec(
    color: int,
    orientation: str,
    anchor_r: int,
    anchor_c: int,
    length: int,
    marked_side: str,
) -> dict[str, object]:
    cells, marker, unmarked, marked = _segment_cells(orientation, anchor_r, anchor_c, length, marked_side)
    direction = (marked[0] - unmarked[0], marked[1] - unmarked[1])
    return {
        "color": color,
        "orientation": orientation,
        "length": length,
        "cells": cells,
        "marker": marker,
        "unmarked": unmarked,
        "marked": marked,
        "dir": direction,
    }


def _draw_line(grid: np.ndarray, line: dict[str, object]) -> set[Coord]:
    cells = set(line["cells"])
    for r, c in line["cells"]:
        grid[r, c] = int(line["color"])
    mr, mc = map(int, line["marker"])
    grid[mr, mc] = GREY
    cells.add((mr, mc))
    return cells


def _fits_cells(cells: set[Coord], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _place_special_targets(
    h: int,
    w: int,
    colors: list[int],
    vertical: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]], set[Coord], set[Coord]] | None:
    size = random.choice([2, 3])
    patch_a = _make_patch(size, [c for c in LINE_POOL if c not in colors][:4] or [1, 2, 3, 4])
    patch_b = _make_patch(size, [c for c in LINE_POOL if c not in colors][-4:] or [6, 7, 8, 9])

    target_lines: list[dict[str, object]] = []
    sprite_specs: list[dict[str, object]] = []
    occupied: set[Coord] = set()

    if not vertical:
        row_lo = h // 2 + 1
        row_hi = h - size - 3
        if row_lo > row_hi:
            return None
        row0 = random.randint(row_lo, row_hi)
        row1 = row0 + size + 1
        length0 = random.randint(5, min(7, w - 6))
        length1 = random.randint(5, min(7, w - 6))
        start = random.randint(3, w - max(length0, length1) - 4)
        line0 = _line_spec(colors[0], "h", row0, start, length0, "right")
        line1 = _line_spec(colors[1], "h", row1, start, length1, "right")
        target_lines.extend([line0, line1])
        sprite_specs.extend(
            [
                {"patch": patch_a, "offset": (1, 0), "contact_local": (0, 0)},
                {"patch": patch_b, "offset": (-1, 0), "contact_local": (size - 1, 0)},
            ]
        )
    else:
        col0_lo = 2
        col0_hi = w - size - 4
        if col0_lo > col0_hi:
            return None
        col0 = random.randint(col0_lo, col0_hi)
        col1 = col0 + size + 1
        length0 = random.randint(5, min(7, h - 6))
        length1 = random.randint(5, min(7, h - 6))
        row_lo = h // 2 + 1
        row_hi = h - max(length0, length1) - 1
        if row_lo > row_hi:
            return None
        start = random.randint(row_lo, row_hi)
        line0 = _line_spec(colors[0], "v", start, col0, length0, "bottom")
        line1 = _line_spec(colors[1], "v", start, col1, length1, "bottom")
        target_lines.extend([line0, line1])
        sprite_specs.extend(
            [
                {"patch": patch_a, "offset": (0, 1), "contact_local": (0, 0)},
                {"patch": patch_b, "offset": (0, -1), "contact_local": (0, size - 1)},
            ]
        )

    for line in target_lines:
        occupied |= set(line["cells"])
        occupied.add(tuple(map(int, line["marker"])))

    reserved = _expand(occupied, h, w, radius=4)
    return target_lines, sprite_specs, occupied, reserved


def _place_random_target(
    h: int,
    w: int,
    color: int,
    reserved: set[Coord],
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_RETRIES):
        orientation = random.choice(["h", "v"])
        length = random.randint(*LINE_LEN_RANGE)
        if orientation == "h":
            if w < length + 4:
                continue
            row = random.randint(h // 2 + 1, h - 2)
            start = random.randint(2, w - length - 3)
            marked_side = random.choice(["left", "right"])
            line = _line_spec(color, "h", row, start, length, marked_side)
        else:
            if h // 2 + 1 > h - length - 1:
                continue
            start_r = random.randint(h // 2 + 1, h - length - 1)
            col = random.randint(2, w - 3)
            marked_side = random.choice(["top", "bottom"])
            line = _line_spec(color, "v", start_r, col, length, marked_side)

        cells = set(line["cells"])
        cells.add(tuple(map(int, line["marker"])))
        if not _fits_cells(cells, h, w):
            continue
        if cells & reserved:
            continue
        return line
    return None


def _place_source_with_sprite(
    h: int,
    w: int,
    color: int,
    sprite_spec: dict[str, object],
    reserved: set[Coord],
    length: int | None = None,
    forced_orientation: str | None = None,
) -> tuple[dict[str, object], int, int, set[Coord]] | None:
    patch = np.asarray(sprite_spec["patch"], dtype=int)
    size = int(patch.shape[0])
    raw_offset = tuple(map(int, sprite_spec["offset"]))
    preferred_offset = raw_offset if raw_offset != (0, 0) else None

    for _ in range(PLACEMENT_RETRIES):
        orientation = forced_orientation or random.choice(["h", "v"])
        line_len = length if length is not None else random.randint(*LINE_LEN_RANGE)
        offset, contact_local = _source_offset_and_contact(size, orientation, preferred=preferred_offset)

        if orientation == "h":
            if w < line_len + 4:
                continue
            max_row = h // 2 - 1
            if 2 > max_row:
                continue
            row = random.randint(2, max_row)
            start = random.randint(2, w - line_len - 3)
            marked_side = random.choice(["left", "right"])
            line = _line_spec(color, "h", row, start, line_len, marked_side)
        else:
            max_start = h // 2 - line_len
            if max_start < 2:
                continue
            start_r = random.randint(2, max_start)
            col = random.randint(2, w - 3)
            marked_side = random.choice(["top", "bottom"])
            line = _line_spec(color, "v", start_r, col, line_len, marked_side)

        ur, uc = map(int, line["unmarked"])
        top = ur + offset[0] - int(contact_local[0])
        left = uc + offset[1] - int(contact_local[1])
        sprite_cells = {(top + r, left + c) for r in range(size) for c in range(size)}
        cells = set(line["cells"]) | {tuple(map(int, line["marker"]))} | sprite_cells
        if not _fits_cells(cells, h, w):
            continue
        if any(r > h // 2 for r, _c in sprite_cells):
            continue
        if cells & reserved:
            continue

        updated = dict(sprite_spec)
        updated["offset"] = offset
        updated["contact_local"] = contact_local
        return line, top, left, cells
    return None


def _assemble_grid(
    h: int,
    w: int,
    pair_specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in pair_specs:
        _draw_line(grid, spec["source"])
        _draw_line(grid, spec["target"])
        _paint_patch(grid, int(spec["sprite_top"]), int(spec["sprite_left"]), np.asarray(spec["sprite"]["patch"], dtype=int))
    return grid


def _make_pair_specs(force_collision: bool) -> list[dict[str, object]] | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(LINE_POOL, pair_count)
    collision_vertical = force_collision and bool(random.randrange(2))

    pair_specs: list[dict[str, object]] = []
    reserved: set[Coord] = set()

    sprite_palette = [c for c in LINE_POOL if c not in colors]
    assert len(sprite_palette) >= 4

    start_idx = 0
    if force_collision:
        special_colors = colors[:2]
        placed = _place_special_targets(h, w, special_colors, vertical=collision_vertical)
        if placed is None:
            return None
        targets, sprites, used, block = placed
        reserved |= block
        source_lengths = [4, 6] if random.randrange(2) == 0 else [5, 7]
        for i in range(2):
            result = _place_source_with_sprite(
                h,
                w,
                special_colors[i],
                sprites[i],
                reserved,
                length=source_lengths[i],
                forced_orientation="v" if collision_vertical else "h",
            )
            if result is None:
                return None
            source, top, left, cells = result
            pair_specs.append(
                {
                    "color": special_colors[i],
                    "source": source,
                    "target": targets[i],
                    "sprite": sprites[i],
                    "sprite_top": top,
                    "sprite_left": left,
                }
            )
            reserved |= _expand(cells, h, w, radius=1)
        start_idx = 2

    for color in colors[start_idx:]:
        target = _place_random_target(h, w, color, reserved)
        if target is None:
            return None
        reserved |= _expand(set(target["cells"]) | {tuple(map(int, target["marker"]))}, h, w, radius=4)

        target_orientation = str(target["orientation"])
        forced_orientation = target_orientation if random.random() < 0.5 else ("v" if target_orientation == "h" else "h")
        size = random.choice([2, 3])
        patch = _make_patch(size, sprite_palette)
        sprite = {"patch": patch, "offset": (0, 0), "contact_local": (0, 0)}
        result = _place_source_with_sprite(h, w, color, sprite, reserved, forced_orientation=forced_orientation)
        if result is None:
            return None
        source, top, left, cells = result
        pair_specs.append(
            {
                "color": color,
                "source": source,
                "target": target,
                "sprite": sprite,
                "sprite_top": top,
                "sprite_left": left,
            }
        )
        reserved |= _expand(cells, h, w, radius=1)

    grid = _assemble_grid(h, w, pair_specs)
    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return None
    assert isinstance(out, np.ndarray)

    if force_collision != bool(meta["had_collision"]):
        return None

    scene = meta["scene"]
    if len(scene["sprites"]) != pair_count:
        return None

    return pair_specs


def generate_input() -> np.ndarray:
    want_collision = bool(random.randrange(2))
    for _ in range(GEN_RETRIES):
        specs = _make_pair_specs(want_collision)
        if specs is None:
            continue
        h = 0
        w = 0
        for spec in specs:
            coords = set(spec["source"]["cells"]) | set(spec["target"]["cells"])
            coords.add(tuple(map(int, spec["source"]["marker"])))
            coords.add(tuple(map(int, spec["target"]["marker"])))
            patch = np.asarray(spec["sprite"]["patch"], dtype=int)
            top = int(spec["sprite_top"])
            left = int(spec["sprite_left"])
            coords |= {(top + r, left + c) for r in range(patch.shape[0]) for c in range(patch.shape[1])}
            h = max(h, max(r for r, _ in coords) + 1)
            w = max(w, max(c for _r, c in coords) + 1)
        h = max(h, random.randint(*GRID_H_RANGE))
        w = max(w, random.randint(*GRID_W_RANGE))
        grid = np.full((h, w), BG, dtype=int)
        for spec in specs:
            _draw_line(grid, spec["source"])
            _draw_line(grid, spec["target"])
            _paint_patch(grid, int(spec["sprite_top"]), int(spec["sprite_left"]), np.asarray(spec["sprite"]["patch"], dtype=int))
        try:
            _solve(grid, return_meta=True)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid conveyor puzzle input")
