from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, bbox_h, bbox_w, components_mask, in_bounds, neighbors4


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

TEMPLATE_SHAPES = (
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
)


def _coords_from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _is_wall_adjacent(grid: np.ndarray, coords: set[tuple[int, int]]) -> bool:
    height, width = grid.shape
    for r, c in coords:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) == GREY:
                return True
    return False


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coords)


def _candidate_chambers(grid: np.ndarray, template_coords: set[tuple[int, int]]) -> list[dict] | None:
    traversable = grid != GREY
    for r, c in template_coords:
        traversable[r, c] = False
    chambers = []
    for comp in components_mask(traversable, connectivity=4):
        coords = {tuple(map(int, rc)) for rc in comp}
        marker_coords = [(r, c) for r, c in coords if int(grid[r, c]) in MARKER_COLORS]
        if not marker_coords:
            continue
        counts = Counter(int(grid[r, c]) for r, c in marker_coords)
        major_color, major_count = counts.most_common(1)[0]
        tied = [color for color, cnt in counts.items() if cnt == major_count]
        if len(tied) != 1:
            return None
        chambers.append(
            {
                "coords": coords,
                "bbox": bbox_coords(comp),
                "centroid": tuple(np.mean(comp, axis=0)),
                "markers": marker_coords,
                "majority": int(major_color),
            }
        )
    return chambers if len(chambers) == 4 else None


def _find_template(grid: np.ndarray) -> dict:
    candidates = []
    height, width = grid.shape
    for h in (2, 3):
        for w in (2, 3):
            for r0 in range(height - h + 1):
                for c0 in range(width - w + 1):
                    sub = grid[r0 : r0 + h, c0 : c0 + w]
                    area = int((sub == BLACK).sum())
                    if not 3 <= area <= 5:
                        continue
                    coord_set = {
                        (int(r0 + rr), int(c0 + cc))
                        for rr, cc in np.argwhere(sub == BLACK)
                    }
                    if h * w == area or not _is_connected(coord_set):
                        continue
                    if not _is_wall_adjacent(grid, coord_set):
                        continue
                    chambers = _candidate_chambers(grid, coord_set)
                    if chambers is None:
                        continue
                    center = ((r0 + r0 + h - 1) / 2.0, (c0 + c0 + w - 1) / 2.0)
                    dist2 = (center[0] - (height - 1) / 2.0) ** 2 + (center[1] - (width - 1) / 2.0) ** 2
                    candidates.append(
                        {
                            "coords": coord_set,
                            "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                            "center": center,
                            "dist2": dist2,
                            "chambers": chambers,
                        }
                    )
    assert candidates, "expected a small wall-adjacent black template"
    candidates.sort(key=lambda item: (item["dist2"], item["bbox"]))
    template = candidates[0]
    assert len(candidates) == 1 or candidates[1]["dist2"] > template["dist2"], "template identification is ambiguous"
    return template


def _find_chambers(grid: np.ndarray, template: dict) -> list[dict]:
    chambers = template.get("chambers")
    if chambers is None:
        chambers = _candidate_chambers(grid, template["coords"])
    assert chambers is not None, "expected exactly four marked chambers"
    assert len(chambers) == 4, "expected exactly four marked chambers"
    return chambers


def _direction_from_center(center: tuple[float, float], centroid: tuple[float, float]) -> str:
    dr = float(centroid[0] - center[0])
    dc = float(centroid[1] - center[1])
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _template_mask(template: dict) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = template["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in template["coords"]:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0)


def _tiling_cells(
    grid: np.ndarray,
    template: dict,
    direction: str,
    color: int,
) -> list[list[tuple[int, int]]]:
    mask, (r0, c0) = _template_mask(template)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    step = bbox_h(template["bbox"]) if direction in ("up", "down") else bbox_w(template["bbox"])
    dr, dc = DIRS[direction]
    placements = []
    k = 1
    while True:
        shifted = []
        valid = True
        for rr, cc in coords:
            nr = r0 + rr + k * step * dr
            nc = c0 + cc + k * step * dc
            if not in_bounds(grid.shape, nr, nc):
                valid = False
                break
            if int(grid[nr, nc]) == GREY or int(grid[nr, nc]) in MARKER_COLORS:
                valid = False
                break
            shifted.append((int(nr), int(nc)))
        if not valid:
            break
        placements.append(shifted)
        k += 1
    return placements


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    template = _find_template(g)
    chambers = _find_chambers(g, template)
    direction_map = {}
    for chamber in chambers:
        direction = _direction_from_center(template["center"], chamber["centroid"])
        assert direction not in direction_map, "multiple chambers mapped to one direction"
        direction_map[direction] = chamber
    assert set(direction_map) == set(DIRS), "expected one chamber in each cardinal direction"

    out = g.copy()
    painted_cells: dict[str, list[tuple[int, int]]] = {}
    for direction, chamber in direction_map.items():
        placements = _tiling_cells(g, template, direction, chamber["majority"])
        painted_cells[direction] = [cell for placement in placements for cell in placement]
        for placement in placements:
            for r, c in placement:
                out[r, c] = int(chamber["majority"])
    return out, {"template": template, "chambers": direction_map, "painted": painted_cells}


def main(grid: np.ndarray) -> np.ndarray:
    """Tile recolored copies of the central black template into the four voted chambers.

    Input structure:
    - The grid contains grey walls, a small non-rectangular black template touching the wall region, and four separate chambers containing black floor plus colored marker pixels.
    - Each chamber corresponds to one cardinal direction from the template and contains 2-4 marker pixels drawn from blue, red, green, and yellow, with a unique majority color.

    Rule:
    - Identify the unique small black wall-adjacent template component and the four non-grey chambers around it.
    - For each chamber, take the majority marker color and map that chamber to up, down, left, or right by comparing the chamber centroid with the template center.
    - Recolor the template to that chamber's majority color and repeat it outward in the chamber direction using the template bbox height for vertical tiling or bbox width for horizontal tiling, stopping before a copy would hit a grey wall, a marker pixel, or the grid boundary.
    - Keep the original black template, all grey walls, and all marker pixels unchanged.

    Key insight:
    - The chamber markers do not fill their own chamber; instead, each chamber votes for the color of the template copies launched toward that chamber.
    """
    return _solve(grid)[0]


def _place_markers(
    grid: np.ndarray,
    room_cells: list[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    majority: int,
) -> list[tuple[int, int, int]]:
    candidates = [cell for cell in room_cells if cell not in forbidden]
    random.shuffle(candidates)
    assert len(candidates) >= 4, "room needs spare cells away from the tiling lane"

    total = random.randint(2, 4)
    majority_count = random.randint(total // 2 + 1, total)
    minority_count = total - majority_count
    markers = []
    for cell in candidates[:majority_count]:
        markers.append((cell[0], cell[1], majority))

    other_colors = [color for color in MARKER_COLORS if color != majority]
    for cell in candidates[majority_count : majority_count + minority_count]:
        markers.append((cell[0], cell[1], random.choice(other_colors)))
    return markers


def _shift_shape(shape: np.ndarray, top: int, left: int) -> set[tuple[int, int]]:
    coords = set()
    for r, c in np.argwhere(shape):
        coords.add((int(top + r), int(left + c)))
    return coords


def _room_cells(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _expected_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    for direction, color in meta["majorities"].items():
        for placement in meta["placements"][direction]:
            for r, c in placement:
                out[r, c] = int(color)
    return out


def _build_case() -> tuple[np.ndarray, dict]:
    shape = random.choice(TEMPLATE_SHAPES)
    th, tw = shape.shape
    spare_per_tile = int(th * tw - shape.sum())
    min_copies = max(1, (4 + spare_per_tile - 1) // spare_per_tile)

    max_copies = 4
    up_copies = random.randint(min_copies, max_copies)
    down_copies = random.randint(min_copies, max_copies)
    left_copies = random.randint(min_copies, max_copies)
    right_copies = random.randint(min_copies, max_copies)

    r0 = 1 + up_copies * th
    c0 = 1 + left_copies * tw
    bottom_margin = 1 + down_copies * th
    right_margin = 1 + right_copies * tw

    height = r0 + th + bottom_margin
    width = c0 + tw + right_margin
    if max(height, width) > 30:
        raise AssertionError("constructed grid exceeds ARC limits")

    grid = np.full((height, width), GREY, dtype=int)
    template_coords = _shift_shape(shape, r0, c0)
    for r, c in template_coords:
        grid[r, c] = BLACK

    rooms = {
        "up": (1, c0, r0 - 1, c0 + tw - 1),
        "down": (r0 + th, c0, height - 2, c0 + tw - 1),
        "left": (r0, 1, r0 + th - 1, c0 - 1),
        "right": (r0, c0 + tw, r0 + th - 1, width - 2),
    }

    for bounds in rooms.values():
        rr0, cc0, rr1, cc1 = bounds
        grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = BLACK

    color_perm = list(MARKER_COLORS)
    random.shuffle(color_perm)
    majorities = dict(zip(("up", "down", "left", "right"), color_perm))

    placements: dict[str, list[list[tuple[int, int]]]] = {}
    occupied_markers: set[tuple[int, int]] = set()
    marker_records = []
    for direction, bounds in rooms.items():
        room_list = _room_cells(bounds)
        lane = set()
        step = th if direction in ("up", "down") else tw
        dr, dc = DIRS[direction]
        shape_cells = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
        origin = (r0, c0)
        direction_placements = []
        k = 1
        while True:
            shifted = []
            valid = True
            for sr, sc in shape_cells:
                nr = origin[0] + sr + k * step * dr
                nc = origin[1] + sc + k * step * dc
                if not (bounds[0] <= nr <= bounds[2] and bounds[1] <= nc <= bounds[3]):
                    valid = False
                    break
                shifted.append((int(nr), int(nc)))
            if not valid:
                break
            direction_placements.append(shifted)
            lane.update(shifted)
            k += 1
        assert direction_placements, "each chamber must receive at least one tiled copy"
        placements[direction] = direction_placements

        markers = _place_markers(grid, room_list, lane | occupied_markers, majorities[direction])
        for mr, mc, color in markers:
            grid[mr, mc] = int(color)
            occupied_markers.add((mr, mc))
            marker_records.append((direction, mr, mc, int(color)))

    meta = {
        "template_coords": template_coords,
        "rooms": rooms,
        "majorities": majorities,
        "placements": placements,
        "markers": marker_records,
    }
    return grid, meta


def _validate_case(grid: np.ndarray, meta: dict) -> None:
    out = main(grid)
    expected = _expected_output(grid, meta)
    assert np.array_equal(out, expected), "solver output disagrees with generator metadata"

    _, parsed = _solve(grid)
    assert all(parsed["painted"][direction] for direction in DIRS), "every direction must paint at least one cell"
    assert any(len(meta["placements"][direction]) >= 2 for direction in DIRS), "need multi-copy examples to expose stride logic"
    assert len(set(meta["majorities"].values())) == 4, "all chamber majorities must differ"

    marker_map: dict[str, list[int]] = {direction: [] for direction in DIRS}
    for direction, r, c, color in meta["markers"]:
        assert int(out[r, c]) == int(color), "markers must remain unchanged in output"
        marker_map[direction].append(int(color))
    for direction, colors in marker_map.items():
        assert 2 <= len(colors) <= 4
        counts = Counter(colors)
        major = meta["majorities"][direction]
        assert counts[major] == max(counts.values())
        assert list(counts.values()).count(counts[major]) == 1, "majority must be unique"

    template_cells = meta["template_coords"]
    for direction in DIRS:
        for cell in parsed["painted"][direction]:
            assert cell not in template_cells, "tiled copies must not overwrite the source template"
    for direction, chamber in parsed["chambers"].items():
        assert int(chamber["majority"]) == int(meta["majorities"][direction]), "parsed chamber vote mismatch"


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid, meta = _build_case()
        try:
            _validate_case(grid, meta)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid chamber-voting tiling input")
