import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
GREY = 5
MAROON = 9
INPUT_COLORS = [2, 3, 4, 6, 7]


SHAPES = (
    {"name": "dot", "cells": ((0, 0),)},
    {"name": "v2", "cells": ((0, 0), (1, 0))},
    {"name": "h2", "cells": ((0, 0), (0, 1))},
    {"name": "v3", "cells": ((0, 0), (1, 0), (2, 0))},
    {"name": "l_bl", "cells": ((0, 0), (1, 0), (1, 1))},
    {"name": "l_br", "cells": ((0, 1), (1, 0), (1, 1))},
    {"name": "l_tl", "cells": ((0, 0), (0, 1), (1, 0))},
    {"name": "l_tr", "cells": ((0, 0), (0, 1), (1, 1))},
)
WIDTH1_SHAPES = tuple(shape for shape in SHAPES if max(c for _, c in shape["cells"]) == 0)
WIDTH2_SHAPES = tuple(shape for shape in SHAPES if max(c for _, c in shape["cells"]) == 1)
OVERLAP_SHAPES = tuple(shape for shape in SHAPES if max(r for r, _ in shape["cells"]) >= 1)

GEN_RETRIES = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bar_endpoints(grid: np.ndarray) -> tuple[int, int]:
    bottom = np.asarray(grid[-1], dtype=int)
    cols = np.flatnonzero(bottom == BLUE)
    assert cols.size >= 3, "expected a blue ruler of length at least 3"
    assert np.array_equal(cols, np.arange(cols[0], cols[-1] + 1)), "blue ruler must be contiguous"
    assert int(cols[0]) > 0 and int(cols[-1]) < grid.shape[1] - 1, "ruler must have black flanks"
    return int(cols[0]), int(cols[-1])


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    active = (g != BG) & (g != BLUE)
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r in range(h):
        for c in range(w):
            if not active[r, c] or seen[r, c]:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            bbox = _bbox(coords)
            cols = sorted({cc for _, cc in coords})
            objects.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "coord_set": set(coords),
                    "bbox": bbox,
                    "cols": tuple(cols),
                    "height": bbox[2] - bbox[0] + 1,
                    "width": bbox[3] - bbox[1] + 1,
                }
            )

    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0], obj["color"]))
    return objects


def _distance_and_side(obj: dict, left: int, right: int) -> tuple[int, str]:
    center_col = (obj["bbox"][1] + obj["bbox"][3]) // 2
    dist_left = abs(center_col - left)
    dist_right = abs(center_col - right)
    if dist_left < dist_right:
        side = "left"
    elif dist_right < dist_left:
        side = "right"
    else:
        side = "tie"
    return min(dist_left, dist_right), side


def _render_from_objects(grid: np.ndarray, objects: list[dict], left: int, right: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()

    for obj in objects:
        distance, _side = _distance_and_side(obj, left, right)
        copy_coords = {(r - distance, c) for r, c in obj["coords"]}
        top_row = obj["bbox"][0]
        copy_bottom = obj["bbox"][2] - distance
        occupied = obj["coord_set"] | copy_coords

        for col in obj["cols"]:
            for row in range(copy_bottom, top_row + 1):
                if (row, col) in occupied:
                    continue
                if out[row, col] == BG:
                    out[row, col] = MAROON

    for obj in objects:
        distance, _side = _distance_and_side(obj, left, right)
        for r, c in obj["coords"]:
            rr = r - distance
            if out[rr, c] in (BG, MAROON, GREY):
                out[rr, c] = GREY

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bottom blue ruler to lift each floating object upward by its endpoint distance, then connect original and copy with vertical trails.
    
    Input structure:
    - The grid is black background with one contiguous blue ruler on the bottom row and 3-5 small colored 4-connected objects above it.
    - Each object is monochromatic, uses a color different from blue, and occupies one column or two adjacent columns.
    - The ruler has black flanks on both sides, so its left and right endpoints are visually identifiable.
    
    Rule:
    - Read the bottom blue segment and record its left endpoint L and right endpoint R.
    - For each non-background, non-blue object, compute the horizontal distance D from the object's bbox center column to the nearer endpoint.
    - Create a grey copy of that object shifted straight upward by D rows, keeping its columns fixed.
    - In each occupied object column, fill background cells between the copy and the original with maroon vertical trail pixels, while leaving blue ruler cells and original colored object cells unchanged.
    
    Key insight:
    - The blue bar is a distance ruler: the same endpoint distance simultaneously determines how far each copy rises and how long its connecting trail becomes.
    """
    g = np.asarray(grid, dtype=int)
    left, right = _bar_endpoints(g)
    objects = _find_objects(g)
    return _render_from_objects(g, objects, left, right)


def _shape_size(shape: dict) -> tuple[int, int]:
    return max(r for r, _ in shape["cells"]) + 1, max(c for _, c in shape["cells"]) + 1


def _bar_geometry(width: int) -> tuple[int, int]:
    bar_len = random.randint(3, width // 2)
    centered = (width - bar_len) // 2
    lo = max(1, centered - 2)
    hi = min(width - bar_len - 2, centered + 2)
    if lo > hi:
        start = max(1, min(width - bar_len - 2, centered))
    else:
        start = random.randint(lo, hi)
    return start, start + bar_len - 1


def _band_clear(bands: list[tuple[int, int]], start: int, width: int, gap: int = 1) -> bool:
    end = start + width - 1
    for other_start, other_end in bands:
        if not (end < other_start - gap or start > other_end + gap):
            return False
    return True


def _sample_roles(count: int, require_overlap: bool) -> list[dict]:
    roles: list[dict] = []
    if require_overlap:
        roles.append({"shape": random.choice(OVERLAP_SHAPES), "require_overlap": True})
    if not any(_shape_size(role["shape"])[1] == 2 for role in roles):
        roles.append({"shape": random.choice(WIDTH2_SHAPES), "require_overlap": False})
    if not any(_shape_size(role["shape"])[1] == 1 for role in roles):
        roles.append({"shape": random.choice(WIDTH1_SHAPES), "require_overlap": False})
    while len(roles) < count:
        roles.append({"shape": random.choice(SHAPES), "require_overlap": False})
    random.shuffle(roles)
    return roles[:count]


def _build_case() -> tuple[np.ndarray, dict] | None:
    height = random.randint(10, 14)
    width = random.randint(15, 19)
    left, right = _bar_geometry(width)

    object_count = random.randint(3, 5)
    require_overlap = random.random() < 0.65
    roles = _sample_roles(object_count, require_overlap)
    bands: list[tuple[int, int]] = []
    objects: list[dict] = []

    for role in roles:
        shape = role["shape"]
        shape_h, shape_w = _shape_size(shape)
        feasible = []
        for start in range(0, width - shape_w + 1):
            if not _band_clear(bands, start, shape_w, gap=1):
                continue
            dist_left = abs(start - left)
            dist_right = abs(start - right)
            distance = min(dist_left, dist_right)
            if distance < 1:
                continue
            if distance > height - shape_h - 3:
                continue
            if role["require_overlap"] and shape_h <= distance:
                continue
            if dist_left < dist_right:
                side = "left"
            elif dist_right < dist_left:
                side = "right"
            else:
                side = "tie"
            feasible.append((start, distance, side))

        if not feasible:
            return None

        start, distance, side = random.choice(feasible)
        top_min = distance
        top_max = height - shape_h - 3
        top = random.randint(top_min, top_max)
        coords = tuple(sorted((top + dr, start + dc) for dr, dc in shape["cells"]))
        bbox = _bbox(list(coords))

        objects.append(
            {
                "shape": shape["name"],
                "color": None,
                "coords": coords,
                "bbox": bbox,
                "cols": tuple(sorted({c for _, c in coords})),
                "distance": distance,
                "side": side,
                "height": shape_h,
                "width": shape_w,
                "require_overlap": role["require_overlap"],
            }
        )
        bands.append((start, start + shape_w - 1))

    sides = {obj["side"] for obj in objects if obj["side"] in {"left", "right"}}
    if sides != {"left", "right"}:
        return None
    if len({obj["distance"] for obj in objects}) < 2:
        return None
    if require_overlap and not any(obj["require_overlap"] and obj["height"] > obj["distance"] for obj in objects):
        return None

    colors = random.sample(INPUT_COLORS, len(objects))
    grid = np.zeros((height, width), dtype=int)
    grid[-1, left : right + 1] = BLUE

    for obj, color in zip(objects, colors):
        obj["color"] = color
        for r, c in obj["coords"]:
            grid[r, c] = color

    metadata = {"left": left, "right": right, "objects": objects, "require_overlap": require_overlap}
    _validate_case(grid, metadata)
    return grid, metadata


def _validate_case(grid: np.ndarray, metadata: dict) -> None:
    left, right = _bar_endpoints(grid)
    assert (left, right) == (metadata["left"], metadata["right"])

    parsed = _find_objects(grid)
    expected = {obj["color"]: obj for obj in metadata["objects"]}
    actual = {obj["color"]: obj for obj in parsed}
    assert set(expected) == set(actual)

    for color, spec in expected.items():
        obj = actual[color]
        assert obj["coords"] == spec["coords"]
        assert obj["cols"] == spec["cols"]
        distance, side = _distance_and_side(obj, left, right)
        assert distance == spec["distance"]
        assert side == spec["side"]
        assert 1 <= distance <= obj["bbox"][0]
        assert obj["bbox"][2] <= grid.shape[0] - 4

    parsed_bands = sorted((obj["bbox"][1], obj["bbox"][3]) for obj in parsed)
    for (_, prev_end), (cur_start, _) in zip(parsed_bands, parsed_bands[1:]):
        assert prev_end < cur_start

    out = main(grid)
    expected_out = _render_from_objects(grid, parsed, left, right)
    assert np.array_equal(out, expected_out)

    if metadata["require_overlap"]:
        overlap_cells = []
        for obj in parsed:
            distance, _side = _distance_and_side(obj, left, right)
            copy_coords = {(r - distance, c) for r, c in obj["coords"]}
            shared = copy_coords & obj["coord_set"]
            if shared:
                overlap_cells.extend((r, c, obj["color"]) for r, c in shared)
        assert overlap_cells, "expected at least one self-overlap example"
        for r, c, color in overlap_cells:
            assert int(out[r, c]) == int(color), "original color must beat the grey copy on overlap"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _build_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid distance-ruler offset-copy input")
