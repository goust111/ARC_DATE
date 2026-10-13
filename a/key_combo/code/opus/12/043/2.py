from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
RED = 2
YELLOW = 4
MARKER_COLORS = [1, 3, 5, 6, 7, 8, 9]

GRID_ROWS = (14, 17)
GRID_COLS = (16, 19)
NUM_OBJECTS = (3, 4)
UNIT_SEARCH_SIZE = 11
PADDING = 1
MAX_GENERATE_ATTEMPTS = 400
MAX_UNIT_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_bends(mask: np.ndarray) -> int:
    h, w = mask.shape
    bends = 0
    for r in range(h - 1):
        for c in range(w - 1):
            bends += int(mask[r : r + 2, c : c + 2].sum() == 3)
    return bends


def _segment_info(coords: np.ndarray) -> dict:
    rows = sorted(int(r) for r in coords[:, 0])
    cols = sorted(int(c) for c in coords[:, 1])
    if len(set(rows)) == 1:
        row = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1))
        return {
            "orientation": "horizontal",
            "row": row,
            "col_min": cols[0],
            "col_max": cols[-1],
            "col_sum": cols[0] + cols[-1],
            "row_min": row,
            "row_max": row,
        }
    if len(set(cols)) == 1:
        col = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1))
        return {
            "orientation": "vertical",
            "col": col,
            "row_min": rows[0],
            "row_max": rows[-1],
            "row_sum": rows[0] + rows[-1],
            "col_min": col,
            "col_max": col,
        }
    raise AssertionError("red component must be a straight contiguous segment")


def _transform_coords(coords: np.ndarray, segment: dict, bends: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if bends == 1:
            if segment["orientation"] == "horizontal":
                out.append((2 * int(segment["row"]) - rr, cc))
            else:
                out.append((rr, 2 * int(segment["col"]) - cc))
        elif bends == 2:
            if segment["orientation"] == "horizontal":
                out.append((rr, int(segment["col_sum"]) - cc))
            else:
                out.append((int(segment["row_sum"]) - rr, cc))
        elif bends == 3:
            if segment["orientation"] == "horizontal":
                out.append((2 * int(segment["row"]) - rr, int(segment["col_sum"]) - cc))
            else:
                out.append((int(segment["row_sum"]) - rr, 2 * int(segment["col"]) - cc))
        else:
            raise AssertionError("bend count must be 1, 2, or 3")
    return out


def _all_on_side(coords: np.ndarray, segment: dict, direction: tuple[int, int]) -> bool:
    if direction == (1, 0):
        return all(int(r) < int(segment["row_min"]) for r, _c in coords)
    if direction == (-1, 0):
        return all(int(r) > int(segment["row_max"]) for r, _c in coords)
    if direction == (0, 1):
        return all(int(c) < int(segment["col_min"]) for _r, c in coords)
    if direction == (0, -1):
        return all(int(c) > int(segment["col_max"]) for _r, c in coords)
    return False


def _all_on_opposite_side(coords: list[tuple[int, int]], segment: dict, direction: tuple[int, int]) -> bool:
    if direction == (1, 0):
        return all(int(r) > int(segment["row_max"]) for r, _c in coords)
    if direction == (-1, 0):
        return all(int(r) < int(segment["row_min"]) for r, _c in coords)
    if direction == (0, 1):
        return all(int(c) > int(segment["col_max"]) for _r, c in coords)
    if direction == (0, -1):
        return all(int(c) < int(segment["col_min"]) for _r, c in coords)
    return False


def _analyze_input(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    red_segments = _components_of_color(g, RED)
    yellow_objects = _components_of_color(g, YELLOW)
    assert len(yellow_objects) >= 1
    assert len(red_segments) == len(yellow_objects)

    red_by_cell: dict[tuple[int, int], int] = {}
    red_info = []
    for idx, coords in enumerate(red_segments):
        info = _segment_info(coords)
        red_info.append(info)
        for r, c in coords:
            red_by_cell[(int(r), int(c))] = idx

    used_red: set[int] = set()
    target_footprints: list[set[tuple[int, int]]] = []
    analyzed: list[dict] = []
    for coords in yellow_objects:
        adjacent_segments: set[int] = set()
        contact_dirs: set[tuple[int, int]] = set()
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            for nr, nc in _neighbors4(rr, cc):
                seg_idx = red_by_cell.get((nr, nc))
                if seg_idx is not None:
                    adjacent_segments.add(seg_idx)
                    contact_dirs.add((nr - rr, nc - cc))

        assert len(adjacent_segments) == 1
        assert len(contact_dirs) == 1
        seg_idx = next(iter(adjacent_segments))
        direction = next(iter(contact_dirs))
        segment = red_info[seg_idx]
        assert _all_on_side(coords, segment, direction)

        r0, c0 = coords.min(axis=0)
        local_mask = np.zeros((int(coords[:, 0].max() - r0 + 1), int(coords[:, 1].max() - c0 + 1)), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        bends = _count_bends(local_mask)
        assert bends in {1, 2, 3}

        transformed = _transform_coords(coords, segment, bends)
        transformed_set = {(int(r), int(c)) for r, c in transformed}
        assert len(transformed_set) == len(transformed)
        h, w = g.shape
        assert all(0 <= r < h and 0 <= c < w for r, c in transformed)
        assert transformed_set.isdisjoint({(int(r), int(c)) for r, c in coords})
        assert transformed_set.isdisjoint(set(red_by_cell))
        assert _all_on_opposite_side(transformed, segment, direction)

        marker_cells = [(r, c) for r, c in transformed if int(g[r, c]) not in {BG, RED, YELLOW}]
        assert len(marker_cells) == 1
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])
        assert all(int(g[r, c]) in {BG, marker_color} for r, c in transformed)

        for footprint in target_footprints:
            assert footprint.isdisjoint(transformed_set)
        target_footprints.append(transformed_set)

        used_red.add(seg_idx)
        analyzed.append(
            {
                "yellow": coords,
                "segment_idx": seg_idx,
                "segment": segment,
                "bends": bends,
                "marker_color": marker_color,
                "transformed": transformed,
            }
        )

    assert len(used_red) == len(red_segments)
    return analyzed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each yellow sprite by the mirror type selected from its bend count, then paint the reflected copy with the footprint marker color.

    Input structure:
    - The grid is black background with 3-4 disconnected yellow 4-connected sprites, one straight red segment adjacent to each sprite, and one colored marker pixel inside each sprite's reflected footprint.
    - Red components are contiguous horizontal or vertical line segments, and each yellow sprite touches exactly one of them from a single side.
    - Marker pixels use non-black, non-red, non-yellow colors and the reflected footprints for different sprites do not overlap.

    Rule:
    - For each yellow sprite, crop its mask conceptually and count the number of 2x2 L-corners, i.e. 2x2 windows containing exactly three yellow cells; the count is always 1, 2, or 3.
    - If the bend count is 1, reflect the sprite across the red segment's long axis; if it is 2, reflect across the segment's short-axis bisector; if it is 3, apply a 180-degree point reflection around the red segment center.
    - Find the unique marker pixel inside the reflected footprint, paint the whole reflected copy with that marker color, and preserve all original pixels; reflected pixels only fill background cells and never overwrite existing non-background content.

    Key insight:
    - The bend count does not map to a color directly; it selects a geometric transform, and that transform determines which footprint contains the marker color to reuse.
    """
    g = np.asarray(grid, dtype=int)
    analyzed = _analyze_input(g)
    out = g.copy()
    for item in analyzed:
        color = int(item["marker_color"])
        for r, c in item["transformed"]:
            if int(out[r, c]) == BG:
                out[r, c] = color
    return out


def _random_connected_mask(area: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r = random.randrange(h)
    c = random.randrange(w)
    mask[r, c] = True
    while int(mask.sum()) < area:
        frontier = []
        for rr, cc in np.argwhere(mask):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = random.choice(frontier)
        mask[nr, nc] = True
    return _crop_mask(mask)


def _random_shape_with_bends(target_bends: int) -> np.ndarray:
    for _ in range(300):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(4, min(7, h * w))
        mask = _random_connected_mask(area, h, w)
        if int(mask.sum()) != area:
            continue
        if _count_bends(mask) == target_bends:
            if random.random() < 0.5:
                mask = np.flipud(mask)
            if random.random() < 0.5:
                mask = np.fliplr(mask)
            mask = np.rot90(mask, random.randrange(4))
            return mask.copy()
    raise RuntimeError(f"failed to build a shape with {target_bends} bends")


def _build_unit(target_bends: int, marker_color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_UNIT_ATTEMPTS):
        mask = _random_shape_with_bends(target_bends)
        h, w = mask.shape
        coords_local = np.argwhere(mask)

        canvas_h = UNIT_SEARCH_SIZE
        canvas_w = UNIT_SEARCH_SIZE
        orientation = random.choice(["horizontal", "vertical"])
        seg_len = random.randint(2, 4)
        if orientation == "horizontal":
            row = canvas_h // 2
            col0 = canvas_w // 2 - seg_len // 2
            red_coords = np.array([(row, col0 + i) for i in range(seg_len)], dtype=int)
        else:
            col = canvas_w // 2
            row0 = canvas_h // 2 - seg_len // 2
            red_coords = np.array([(row0 + i, col) for i in range(seg_len)], dtype=int)
        segment = _segment_info(red_coords)
        red_set = {(int(r), int(c)) for r, c in red_coords}

        candidates: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
        for r0 in range(canvas_h - h + 1):
            for c0 in range(canvas_w - w + 1):
                src = coords_local + np.array([r0, c0])
                src_set = {(int(r), int(c)) for r, c in src}
                if src_set & red_set:
                    continue

                adjacent_segments = set()
                contact_dirs = set()
                for r, c in src:
                    for nr, nc in _neighbors4(int(r), int(c)):
                        if (nr, nc) in red_set:
                            adjacent_segments.add(0)
                            contact_dirs.add((nr - int(r), nc - int(c)))
                if len(adjacent_segments) != 1 or len(contact_dirs) != 1:
                    continue
                direction = next(iter(contact_dirs))
                if not _all_on_side(src, segment, direction):
                    continue

                dst = _transform_coords(src, segment, target_bends)
                dst_set = {(int(r), int(c)) for r, c in dst}
                if len(dst_set) != len(dst):
                    continue
                if not all(0 <= r < canvas_h and 0 <= c < canvas_w for r, c in dst):
                    continue
                if dst_set & src_set or dst_set & red_set:
                    continue
                if not _all_on_opposite_side(dst, segment, direction):
                    continue
                candidates.append((src.copy(), dst))

        if not candidates:
            continue

        src, dst = random.choice(candidates)
        all_coords = np.vstack([src, red_coords, np.array(dst, dtype=int)])
        r0, c0 = all_coords.min(axis=0)
        r1, c1 = all_coords.max(axis=0)
        shape = (int(r1 - r0 + 1), int(c1 - c0 + 1))
        src_rel = src - np.array([r0, c0])
        red_rel = red_coords - np.array([r0, c0])
        dst_rel = np.array(dst, dtype=int) - np.array([r0, c0])

        output_unit = np.zeros(shape, dtype=int)
        output_unit[src_rel[:, 0], src_rel[:, 1]] = YELLOW
        output_unit[red_rel[:, 0], red_rel[:, 1]] = RED
        output_unit[dst_rel[:, 0], dst_rel[:, 1]] = marker_color

        input_unit = output_unit.copy()
        input_unit[dst_rel[:, 0], dst_rel[:, 1]] = BG
        mr, mc = dst_rel[random.randrange(len(dst_rel))]
        input_unit[int(mr), int(mc)] = marker_color

        assert np.array_equal(main(input_unit), output_unit)
        return input_unit, output_unit

    raise RuntimeError("failed to build a valid unit")


def _can_place(occupied: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    mask = patch != BG
    for dr, dc in np.argwhere(mask):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - PADDING)
        r2 = min(hh, rr + PADDING + 1)
        c1 = max(0, cc - PADDING)
        c2 = min(ww, cc + PADDING + 1)
        if occupied[r1:r2, c1:c2].any():
            return False
    return True


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = patch[mask]


def _generate_scene() -> np.ndarray:
    n_obj = random.randint(*NUM_OBJECTS)
    bend_counts = [1, 2, 3]
    if n_obj == 4:
        bend_counts.append(random.choice([1, 2, 3]))
    random.shuffle(bend_counts)
    marker_colors = random.sample(MARKER_COLORS, n_obj)

    units = [_build_unit(bends, color) for bends, color in zip(bend_counts, marker_colors)]
    order = list(range(n_obj))
    order.sort(key=lambda idx: int((units[idx][1] != BG).sum()), reverse=True)

    for _ in range(120):
        h = random.randint(*GRID_ROWS)
        w = random.randint(*GRID_COLS)
        inp = np.zeros((h, w), dtype=int)
        out = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        ok = True

        for idx in order:
            input_unit, output_unit = units[idx]
            ph, pw = output_unit.shape
            positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
            random.shuffle(positions)
            placed = False
            for r0, c0 in positions:
                if _can_place(occupied, output_unit, r0, c0):
                    _paste(inp, input_unit, r0, c0)
                    _paste(out, output_unit, r0, c0)
                    occupied[r0 : r0 + ph, c0 : c0 + pw] |= output_unit != BG
                    placed = True
                    break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        analyzed = _analyze_input(inp)
        assert {item["bends"] for item in analyzed} == {1, 2, 3}
        assert np.array_equal(main(inp), out)
        return inp

    raise RuntimeError("failed to place generated units")


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        try:
            return _generate_scene()
        except Exception:
            continue
    raise RuntimeError("failed to generate valid input")
