from __future__ import annotations

from collections import Counter
import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
WALL = 9
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _infer_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (int(bg), GREY)]
    candidates = []
    for color in colors:
        comps = _find_color_components(grid, color)
        if any(comp["area"] >= 2 for comp in comps):
            candidates.append(color)
    assert len(candidates) == 1, f"expected one wall color, got {candidates}"
    return int(candidates[0])


def _horizontal_hole(grid: np.ndarray, wall_color: int, r: int, c: int) -> dict | None:
    h, w = grid.shape
    if c - 1 < 0 or c + 1 >= w:
        return None
    if int(grid[r, c - 1]) != wall_color or int(grid[r, c + 1]) != wall_color:
        return None
    if (r > 0 and int(grid[r - 1, c]) == wall_color) or (r + 1 < h and int(grid[r + 1, c]) == wall_color):
        return None

    left = c - 1
    while left - 1 >= 0 and int(grid[r, left - 1]) == wall_color:
        left -= 1
    right = c + 1
    while right + 1 < w and int(grid[r, right + 1]) == wall_color:
        right += 1

    if c - left < 2 or right - c < 2:
        return None
    for cc in range(left, right + 1):
        if cc == c:
            continue
        if int(grid[r, cc]) != wall_color:
            return None
        if (r > 0 and int(grid[r - 1, cc]) == wall_color) or (r + 1 < h and int(grid[r + 1, cc]) == wall_color):
            return None
    return {
        "hole": (int(r), int(c)),
        "axis": "h",
        "wall_span": (int(r), int(left), int(r), int(right)),
        "wall_cells": {(int(r), int(cc)) for cc in range(left, right + 1) if cc != c},
    }


def _vertical_hole(grid: np.ndarray, wall_color: int, r: int, c: int) -> dict | None:
    h, w = grid.shape
    if r - 1 < 0 or r + 1 >= h:
        return None
    if int(grid[r - 1, c]) != wall_color or int(grid[r + 1, c]) != wall_color:
        return None
    if (c > 0 and int(grid[r, c - 1]) == wall_color) or (c + 1 < w and int(grid[r, c + 1]) == wall_color):
        return None

    top = r - 1
    while top - 1 >= 0 and int(grid[top - 1, c]) == wall_color:
        top -= 1
    bottom = r + 1
    while bottom + 1 < h and int(grid[bottom + 1, c]) == wall_color:
        bottom += 1

    if r - top < 2 or bottom - r < 2:
        return None
    for rr in range(top, bottom + 1):
        if rr == r:
            continue
        if int(grid[rr, c]) != wall_color:
            return None
        if (c > 0 and int(grid[rr, c - 1]) == wall_color) or (c + 1 < w and int(grid[rr, c + 1]) == wall_color):
            return None
    return {
        "hole": (int(r), int(c)),
        "axis": "v",
        "wall_span": (int(top), int(c), int(bottom), int(c)),
        "wall_cells": {(int(rr), int(c)) for rr in range(top, bottom + 1) if rr != r},
    }


def _find_holes(grid: np.ndarray, bg: int, wall_color: int) -> list[dict]:
    holes = []
    for r, c in np.argwhere(grid == bg):
        cand_h = _horizontal_hole(grid, wall_color, int(r), int(c))
        cand_v = _vertical_hole(grid, wall_color, int(r), int(c))
        cands = [cand for cand in (cand_h, cand_v) if cand is not None]
        assert len(cands) <= 1, f"ambiguous hole orientation at {(int(r), int(c))}"
        if cands:
            holes.append(cands[0])
    holes.sort(key=lambda item: item["hole"])
    assert holes, "expected at least one pinhole wall"
    return holes


def _collect_markers(grid: np.ndarray, bg: int, wall_color: int) -> dict[tuple[int, int], int]:
    marker_map: dict[tuple[int, int], int] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (int(bg), GREY, int(wall_color)):
            continue
        comps = _find_color_components(grid, color)
        for comp in comps:
            assert comp["area"] == 1, "non-wall non-gray colors must be singleton markers"
            pos = tuple(map(int, comp["coords"][0]))
            marker_map[pos] = color
    assert marker_map, "expected marker pixels"
    return marker_map


def _component_side(component: dict, hole: tuple[int, int], axis: str) -> int | None:
    hr, hc = hole
    coords = component["coords"]
    values = coords[:, 0] - hr if axis == "h" else coords[:, 1] - hc
    if np.any(values == 0):
        return None
    side = 1 if int(values[0]) > 0 else -1
    if np.all(values * side > 0):
        return side
    return None


def _reflect_coords(coords: np.ndarray, hole: tuple[int, int]) -> list[tuple[int, int]]:
    hr, hc = hole
    return [(int(2 * hr - r), int(2 * hc - c)) for r, c in coords]


def _extract_scene(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    wall_color = _infer_wall_color(g, bg)
    holes = _find_holes(g, bg, wall_color)
    gray_components = _find_color_components(g, GREY)
    marker_map = _collect_markers(g, bg, wall_color)

    pairs = []
    used_components = set()
    used_markers = set()

    for hole_info in holes:
        hole = hole_info["hole"]
        axis = hole_info["axis"]
        matches = set()
        for idx, component in enumerate(gray_components):
            if _component_side(component, hole, axis) is None:
                continue
            for reflected in _reflect_coords(component["coords"], hole):
                if reflected in marker_map:
                    matches.add((idx, reflected, int(marker_map[reflected])))
        assert len(matches) == 1, f"expected one object-marker match for hole {hole}, got {matches}"
        component_idx, marker_pos, marker_color = next(iter(matches))
        assert component_idx not in used_components, "gray component cannot serve two pinholes"
        assert marker_pos not in used_markers, "marker cannot serve two pinholes"
        used_components.add(component_idx)
        used_markers.add(marker_pos)

        component = gray_components[component_idx]
        reflected_coords = _reflect_coords(component["coords"], hole)
        for rr, cc in reflected_coords:
            assert _in_bounds(g.shape, rr, cc), "reflected image must stay in bounds"
            assert (rr, cc) != hole, "reflected image cannot include the pinhole"
            assert int(g[rr, cc]) in (int(bg), int(marker_color)), "image region must be empty except the seed marker"

        pairs.append(
            {
                "hole": hole,
                "axis": axis,
                "wall_cells": set(hole_info["wall_cells"]),
                "component_idx": int(component_idx),
                "source_coords": [tuple(map(int, p)) for p in component["coords"]],
                "marker_pos": tuple(map(int, marker_pos)),
                "marker_color": int(marker_color),
                "reflected_coords": reflected_coords,
            }
        )

    assert len(pairs) == len(gray_components) == len(marker_map), "expected one gray object and one marker per pinhole"
    return int(bg), int(wall_color), sorted(pairs, key=lambda item: item["hole"])


def _render_output(grid: np.ndarray, pairs: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for pair in pairs:
        color = int(pair["marker_color"])
        for r, c in pair["reflected_coords"]:
            out[r, c] = color
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Project each gray object through its wall pinhole as a recolored 180-degree point reflection.

    Input structure:
    - The background is black.
    - Several maroon one-cell-thick straight wall segments each contain exactly one black gap cell.
    - For every gap, one side contains a 4-connected gray object and the opposite side contains a colored singleton marker.

    Rule:
    - Find each wall gap that acts as a pinhole.
    - Match the gray object to the marker whose position is the point reflection of one object pixel across that pinhole.
    - Reflect every pixel of the whole gray object through the pinhole onto the other side, recolor the reflected copy with the marker color, keep the wall and gray source object unchanged, and keep the pinhole black.

    Key insight:
    - The marker is not an extra target to move; it is one pixel of the full reflected image, and the image is a 180-degree point reflection around the gap cell rather than a translation or a mirror across the whole wall.
    """
    grid = np.asarray(input_grid, dtype=int)
    bg, _wall_color, pairs = _extract_scene(grid)
    out = _render_output(grid, pairs)
    for pair in pairs:
        hr, hc = pair["hole"]
        out[hr, hc] = bg
    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    while True:
        area_min = max(3, (height * width + 2) // 3)
        target = random.randint(area_min, height * width)
        cells = {(random.randrange(height), random.randrange(width))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_coords(coords)
        mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        if mask.shape[0] >= 2 and mask.shape[1] >= 2 and int(mask.sum()) >= 3:
            return mask


def _local_case(slot_h: int, slot_w: int, marker_color: int) -> tuple[np.ndarray, dict]:
    for _ in range(200):
        grid = np.full((slot_h, slot_w), BG, dtype=int)
        axis = random.choice(("h", "v"))
        side = random.choice((-1, 1))

        if axis == "h":
            hr = random.randint(3, slot_h - 4)
            wall_left = random.randint(1, slot_w - 6)
            wall_right = random.randint(wall_left + 4, slot_w - 2)
            hc = random.randint(wall_left + 2, wall_right - 2)
            radial_limit = min(hr - 1, slot_h - 2 - hr)
            lateral_limit = min(hc - 1, slot_w - 2 - hc)
            if radial_limit < 2 or lateral_limit < 1:
                continue

            obj_h = random.randint(2, min(4, radial_limit))
            base_dist = random.randint(1, radial_limit - obj_h + 1)
            obj_w = random.randint(2, min(5, 2 * lateral_limit + 1))
            offset_c0 = random.randint(-lateral_limit, lateral_limit - obj_w + 1)
            mask = _random_connected_mask(obj_h, obj_w)

            source = []
            for rr, cc in np.argwhere(mask):
                r = hr + side * (base_dist + int(rr))
                c = hc + offset_c0 + int(cc)
                source.append((int(r), int(c)))
            wall_cells = {(hr, c) for c in range(wall_left, wall_right + 1) if c != hc}
            hole = (hr, hc)
        else:
            hc = random.randint(3, slot_w - 4)
            wall_top = random.randint(1, slot_h - 6)
            wall_bottom = random.randint(wall_top + 4, slot_h - 2)
            hr = random.randint(wall_top + 2, wall_bottom - 2)
            radial_limit = min(hc - 1, slot_w - 2 - hc)
            lateral_limit = min(hr - 1, slot_h - 2 - hr)
            if radial_limit < 2 or lateral_limit < 1:
                continue

            obj_w = random.randint(2, min(4, radial_limit))
            base_dist = random.randint(1, radial_limit - obj_w + 1)
            obj_h = random.randint(2, min(5, 2 * lateral_limit + 1))
            offset_r0 = random.randint(-lateral_limit, lateral_limit - obj_h + 1)
            mask = _random_connected_mask(obj_h, obj_w)

            source = []
            for rr, cc in np.argwhere(mask):
                r = hr + offset_r0 + int(rr)
                c = hc + side * (base_dist + int(cc))
                source.append((int(r), int(c)))
            wall_cells = {(r, hc) for r in range(wall_top, wall_bottom + 1) if r != hr}
            hole = (hr, hc)

        reflected = [(int(2 * hole[0] - r), int(2 * hole[1] - c)) for r, c in source]
        source_set = set(source)
        reflected_set = set(reflected)
        if len(source_set) != len(source) or len(reflected_set) != len(reflected):
            continue
        if hole in source_set or hole in reflected_set:
            continue
        if source_set & wall_cells or reflected_set & wall_cells:
            continue
        if any(not _in_bounds((slot_h, slot_w), r, c) for r, c in source_set | reflected_set):
            continue

        marker_pos = random.choice(reflected)
        for r, c in wall_cells:
            grid[r, c] = WALL
        for r, c in source:
            grid[r, c] = GREY
        grid[marker_pos] = marker_color

        for r, c in reflected:
            if (r, c) == marker_pos:
                continue
            if int(grid[r, c]) != BG:
                break
        else:
            return grid, {
                "hole": hole,
                "axis": axis,
                "wall_cells": set(map(tuple, wall_cells)),
                "source_coords": sorted(map(tuple, source)),
                "reflected_coords": sorted(map(tuple, reflected)),
                "marker_pos": tuple(marker_pos),
                "marker_color": int(marker_color),
            }
    raise RuntimeError("failed to generate a local pinhole case")


def _choose_layout(case_count: int) -> tuple[int, int]:
    if case_count == 2:
        return random.choice(((1, 2), (2, 1), (2, 2)))
    if case_count == 3:
        return 2, 2
    return 2, 2


def _expected_from_meta(grid: np.ndarray, meta: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for item in meta:
        for r, c in item["reflected_coords"]:
            out[r, c] = int(item["marker_color"])
        hr, hc = item["hole"]
        out[hr, hc] = BG
    return out


def _canonical_meta(meta: Iterable[dict]) -> list[dict]:
    normalized = []
    for item in meta:
        normalized.append(
            {
                "hole": tuple(item["hole"]),
                "axis": str(item["axis"]),
                "wall_cells": sorted(map(tuple, item["wall_cells"])),
                "source_coords": sorted(map(tuple, item["source_coords"])),
                "reflected_coords": sorted(map(tuple, item["reflected_coords"])),
                "marker_pos": tuple(item["marker_pos"]),
                "marker_color": int(item["marker_color"]),
            }
        )
    return sorted(normalized, key=lambda item: item["hole"])


def _validate_meta_scene(grid: np.ndarray, meta: list[dict]) -> None:
    used_source = set()
    used_reflected = set()
    used_markers = set()
    for item in meta:
        hole = tuple(item["hole"])
        assert int(grid[hole]) == BG
        assert hole not in used_source and hole not in used_reflected

        for cell in item["wall_cells"]:
            assert int(grid[cell]) == WALL
        for cell in item["source_coords"]:
            assert int(grid[cell]) == GREY
            assert cell not in used_source
            used_source.add(cell)
        for cell in item["reflected_coords"]:
            assert cell not in used_reflected
            used_reflected.add(cell)
            if cell == tuple(item["marker_pos"]):
                assert int(grid[cell]) == int(item["marker_color"])
            else:
                assert int(grid[cell]) == BG
        assert tuple(item["marker_pos"]) in map(tuple, item["reflected_coords"])
        assert tuple(item["marker_pos"]) not in used_markers
        used_markers.add(tuple(item["marker_pos"]))


def _generate_case(return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    case_count = random.randint(2, 4)
    rows, cols = _choose_layout(case_count)
    slot_h = random.randint(9, 11)
    slot_w = random.randint(9, 11)
    gap = 1
    height = rows * slot_h + (rows + 1) * gap
    width = cols * slot_w + (cols + 1) * gap
    if height > 30 or width > 30:
        raise RuntimeError("layout exceeds ARC size limit")

    grid = np.full((height, width), BG, dtype=int)
    positions = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    chosen_positions = random.sample(positions, case_count)
    marker_colors = random.sample(MARKER_COLORS, case_count)
    meta = []

    for marker_color, (slot_r, slot_c) in zip(marker_colors, chosen_positions):
        local, local_meta = _local_case(slot_h, slot_w, marker_color)
        top = gap + slot_r * (slot_h + gap)
        left = gap + slot_c * (slot_w + gap)
        grid[top : top + slot_h, left : left + slot_w] = local

        meta.append(
            {
                "hole": (top + local_meta["hole"][0], left + local_meta["hole"][1]),
                "axis": local_meta["axis"],
                "wall_cells": {(top + r, left + c) for r, c in local_meta["wall_cells"]},
                "source_coords": [(top + r, left + c) for r, c in local_meta["source_coords"]],
                "reflected_coords": [(top + r, left + c) for r, c in local_meta["reflected_coords"]],
                "marker_pos": (top + local_meta["marker_pos"][0], left + local_meta["marker_pos"][1]),
                "marker_color": local_meta["marker_color"],
            }
        )

    _validate_meta_scene(grid, meta)
    expected = _expected_from_meta(grid, meta)
    parsed = _extract_scene(grid)[2]
    assert _canonical_meta(parsed) == _canonical_meta(meta)
    assert np.array_equal(main(grid), expected)
    if return_meta:
        return grid, meta
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_case(return_meta=False)
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid pinhole inversion instance")
