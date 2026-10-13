from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import bbox_coords as _bbox_coords  # type: ignore
    from arc_utils import find_components as _find_components  # type: ignore
    from arc_utils import get_bg as _get_bg  # type: ignore
except Exception:  # pragma: no cover - fallback keeps SourceFileLoader robust
    _bbox_coords = None
    _find_components = None
    _get_bg = None


BG = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
AREAS = (4, 9, 16)
AREA_TO_SIDE = {4: 2, 9: 3, 16: 4}
AREA_DIMS = {
    4: ((2, 3), (3, 2)),
    9: ((2, 5), (5, 2), (3, 4), (4, 3)),
    16: ((3, 6), (6, 3), (4, 5), (5, 4)),
}
GRID_SIDE_RANGE = (16, 22)
GENERATION_RETRIES = 400
SHAPE_ATTEMPTS = 250
PLACEMENT_ATTEMPTS = 250


def get_bg(grid: np.ndarray) -> int:
    if _get_bg is not None:
        return int(_get_bg(grid))
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    if _bbox_coords is not None:
        r0, c0, r1, c1 = _bbox_coords(coords)
        return int(r0), int(c0), int(r1), int(c1)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    if _find_components is not None:
        return list(
            _find_components(
                np.asarray(grid, dtype=int),
                bg=int(bg),
                connectivity=4,
                monochromatic=True,
                return_format="dict",
            )
        )

    g = np.asarray(grid, dtype=int)
    seen = np.zeros_like(g, dtype=bool)
    out: list[dict] = []
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return out


def _component_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _has_flanked_boundary_gap(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for c in range(w):
        if not mask[0, c] and mask[0, :c].any() and mask[0, c + 1 :].any():
            return True
        if h > 1 and not mask[h - 1, c] and mask[h - 1, :c].any() and mask[h - 1, c + 1 :].any():
            return True
    for r in range(h):
        if not mask[r, 0] and mask[:r, 0].any() and mask[r + 1 :, 0].any():
            return True
        if w > 1 and not mask[r, w - 1] and mask[:r, w - 1].any() and mask[r + 1 :, w - 1].any():
            return True
    return False


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.any(g == BG), "yellow background must be present"
    assert int(get_bg(g)) == BG, "yellow must be the dominant background"

    objects = find_components(g, BG)
    assert 5 <= len(objects) <= 7, "expected 5 to 7 foreground objects"

    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors)), "each object must use a unique color"

    enriched: list[dict] = []
    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        bbox = tuple(map(int, obj["bbox"]))
        area = int(obj["area"])
        side = int(math.isqrt(area))
        assert area in AREAS and side * side == area, "pixel count must be a perfect square from {4,9,16}"
        mask = _component_mask(coords, bbox)
        assert mask.sum() == area
        assert not (mask.shape[0] == side and mask.shape[1] == side and mask.all()), "input shapes must not already be solid squares"
        enriched.append(
            {
                "color": int(obj["color"]),
                "coords": coords,
                "bbox": bbox,
                "area": area,
                "side": side,
                "row": bbox[0],
                "has_notch": _has_flanked_boundary_gap(mask),
            }
        )

    for has_notch in (False, True):
        members = [obj for obj in enriched if obj["has_notch"] == has_notch]
        by_area: dict[int, list[int]] = {}
        for obj in members:
            by_area.setdefault(int(obj["area"]), []).append(int(obj["row"]))
        for rows in by_area.values():
            assert len(rows) == len(set(rows)), "same-area ties must be broken uniquely by bbox top row"

    return enriched


def _render_output_from_objects(objects: list[dict]) -> np.ndarray:
    assert objects
    group_a = sorted(
        (obj for obj in objects if obj["has_notch"]),
        key=lambda obj: (int(obj["area"]), int(obj["row"]), int(obj["bbox"][1])),
    )
    group_b = sorted(
        (obj for obj in objects if not obj["has_notch"]),
        key=lambda obj: (int(obj["area"]), int(obj["row"]), int(obj["bbox"][1])),
    )
    assert group_a and group_b, "both notch classes must be present"

    height = max(int(obj["side"]) for obj in objects)
    width = sum(int(obj["side"]) for obj in objects) + len(objects) - 1
    out = np.full((height, width), BG, dtype=int)

    cursor = 0
    for group_index, group in enumerate((group_a, group_b)):
        if group_index == 1:
            cursor += 1
        for i, obj in enumerate(group):
            side = int(obj["side"])
            top = height - side
            out[top:height, cursor : cursor + side] = int(obj["color"])
            cursor += side
            if i + 1 < len(group):
                cursor += 1

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by whether their bbox border has a flanked background notch, then repack each group into ascending solid squares.

    Input structure:
    - The background is yellow and the foreground consists of 5 to 7 separated 4-connected monochrome objects in distinct colors.
    - Every object has area 4, 9, or 16 pixels, so its area is a perfect square, but the input shape itself is not a solid square.
    - Some objects have a flanked background gap on a bbox border, while the others have no such bbox-border notch.

    Rule:
    - Find each foreground object and inspect its own bbox border for a background gap whose two same-border neighbors are object pixels.
    - Put notched objects in group A and non-notched objects in group B.
    - Within each group, sort by pixel count ascending; if counts tie, sort by the object's input bbox top-row coordinate ascending.
    - Replace every object by a same-color solid square whose side length is the square root of its area.
    - Place group A from left to right with bottoms aligned and one yellow column between adjacent squares; after one more yellow separator column, place group B the same way.

    Key insight:
    - The notch test is not a motion direction cue here; it is a binary classifier that decides which objects share a repacked lane before the area-based square conversion and ordering.
    """
    objects = _extract_objects(np.asarray(grid, dtype=int))
    return _render_output_from_objects(objects)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _removable_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = list(map(tuple, np.argwhere(mask)))
    random.shuffle(coords)
    candidates: list[tuple[int, int]] = []
    for r, c in coords:
        trial = mask.copy()
        trial[r, c] = False
        if _spans_bbox(trial) and _is_connected(trial):
            candidates.append((int(r), int(c)))
    return candidates


def _sample_shape(area: int, want_notch: bool) -> np.ndarray | None:
    dims = list(AREA_DIMS[area])
    for _ in range(SHAPE_ATTEMPTS):
        h, w = random.choice(dims)
        mask = np.ones((h, w), dtype=bool)
        remove_count = h * w - area
        ok = True
        for _step in range(remove_count):
            candidates = _removable_cells(mask)
            if not candidates:
                ok = False
                break

            def priority(cell: tuple[int, int]) -> int:
                r, c = cell
                on_edge = r in (0, h - 1) or c in (0, w - 1)
                corner = (r in (0, h - 1)) and (c in (0, w - 1))
                middle_edge = on_edge and not corner
                if want_notch:
                    return 2 if middle_edge else 1 if on_edge else 0
                return 2 if corner else 1 if not on_edge else 0

            best_rank = max(priority(cell) for cell in candidates)
            best = [cell for cell in candidates if priority(cell) == best_rank]
            r, c = random.choice(best)
            mask[r, c] = False

        if not ok:
            continue
        if int(mask.sum()) != area:
            continue
        if not _spans_bbox(mask):
            continue
        if not _is_connected(mask):
            continue
        if _has_flanked_boundary_gap(mask) != want_notch:
            continue
        side = AREA_TO_SIDE[area]
        if mask.shape == (side, side) and mask.all():
            continue

        variant = mask.copy()
        if random.random() < 0.5:
            variant = np.flipud(variant)
        if random.random() < 0.5:
            variant = np.fliplr(variant)
        variant = np.rot90(variant, random.randrange(4))
        if _has_flanked_boundary_gap(variant) == want_notch and _is_connected(variant):
            return variant
    return None


def _sample_specs() -> list[dict] | None:
    n = random.randint(5, 7)
    notch_count = random.randint(2, n - 2)
    groups = [True] * notch_count + [False] * (n - notch_count)
    random.shuffle(groups)

    areas = [0] * n
    if random.random() < 0.7:
        indices = [i for i in range(n) if (groups[i] or random.random() < 0.6)]
        if len(indices) >= 2:
            pair = random.sample(indices, 2)
            forced_area = random.choice((9, 16) if groups[pair[0]] and groups[pair[1]] else AREAS)
            if groups[pair[0]] != groups[pair[1]] and forced_area == 4 and (groups[pair[0]] or groups[pair[1]]):
                forced_area = 9
            areas[pair[0]] = forced_area
            areas[pair[1]] = forced_area

    for i, has_notch in enumerate(groups):
        if areas[i]:
            continue
        pool = (9, 16) if has_notch else AREAS
        areas[i] = random.choice(pool)

    if sum(AREA_TO_SIDE[area] for area in areas) + n - 1 > 30:
        return None
    if len(set(areas)) == 1 and len(set(groups)) == 1:
        return None

    colors = random.sample(PALETTE, n)
    specs = []
    for color, area, has_notch in zip(colors, areas, groups):
        mask = _sample_shape(area, has_notch)
        if mask is None:
            return None
        specs.append({"color": color, "area": area, "has_notch": has_notch, "mask": mask})
    return specs


def _place_specs(specs: list[dict]) -> np.ndarray | None:
    max_h = max(int(spec["mask"].shape[0]) for spec in specs)
    max_w = max(int(spec["mask"].shape[1]) for spec in specs)
    min_side = max(GRID_SIDE_RANGE[0], max(max_h, max_w) + 2)
    for _ in range(PLACEMENT_ATTEMPTS):
        h = random.randint(min_side, GRID_SIDE_RANGE[1])
        w = random.randint(min_side, GRID_SIDE_RANGE[1])
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        placements: list[tuple[int, int, int]] = []
        order = sorted(range(len(specs)), key=lambda i: specs[i]["mask"].size, reverse=True)
        ok = True
        for idx in order:
            mask = np.asarray(specs[idx]["mask"], dtype=bool)
            mh, mw = mask.shape
            placed = False
            for _attempt in range(PLACEMENT_ATTEMPTS):
                r0 = random.randint(0, h - mh)
                c0 = random.randint(0, w - mw)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + mh + 1)
                cc1 = min(w, c0 + mw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                patch = grid[r0 : r0 + mh, c0 : c0 + mw]
                patch[mask] = int(specs[idx]["color"])
                occ_patch = occupied[r0 : r0 + mh, c0 : c0 + mw]
                occ_patch[mask] = True
                placements.append((idx, r0, c0))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        placements.sort()
        for idx, r0, c0 in placements:
            specs[idx]["placed_row"] = int(r0)
            specs[idx]["placed_col"] = int(c0)
        return grid
    return None


def _validate_semantics(grid: np.ndarray, intended: list[dict]) -> bool:
    try:
        detected = _extract_objects(grid)
        out = main(grid)
    except AssertionError:
        return False

    if len(detected) != len(intended):
        return False

    by_color_detected = {int(obj["color"]): obj for obj in detected}
    by_color_intended = {int(spec["color"]): spec for spec in intended}
    if set(by_color_detected) != set(by_color_intended):
        return False

    for color, spec in by_color_intended.items():
        obj = by_color_detected[color]
        if bool(obj["has_notch"]) != bool(spec["has_notch"]):
            return False
        if int(obj["area"]) != int(spec["area"]):
            return False

    group_a = sorted(
        (obj for obj in detected if obj["has_notch"]),
        key=lambda obj: (int(obj["area"]), int(obj["row"]), int(obj["bbox"][1])),
    )
    group_b = sorted(
        (obj for obj in detected if not obj["has_notch"]),
        key=lambda obj: (int(obj["area"]), int(obj["row"]), int(obj["bbox"][1])),
    )
    if not group_a or not group_b:
        return False

    expected_h = max(int(obj["side"]) for obj in detected)
    expected_w = sum(int(obj["side"]) for obj in detected) + len(detected) - 1
    if out.shape != (expected_h, expected_w):
        return False

    cursor = 0
    for group_index, group in enumerate((group_a, group_b)):
        if group_index == 1:
            if cursor <= 0 or not np.all(out[:, cursor] == BG):
                return False
            cursor += 1
        for i, obj in enumerate(group):
            side = int(obj["side"])
            top = expected_h - side
            block = out[top:expected_h, cursor : cursor + side]
            if block.shape != (side, side) or not np.all(block == int(obj["color"])):
                return False
            if top > 0 and np.any(out[:top, cursor : cursor + side] != BG):
                return False
            cursor += side
            if i + 1 < len(group):
                if not np.all(out[:, cursor] == BG):
                    return False
                cursor += 1
    if cursor != expected_w:
        return False

    if np.count_nonzero(out == BG) == 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        specs = _sample_specs()
        if specs is None:
            continue
        grid = _place_specs(specs)
        if grid is None:
            continue
        if _validate_semantics(grid, specs):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
