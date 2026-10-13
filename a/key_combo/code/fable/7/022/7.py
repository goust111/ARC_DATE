from __future__ import annotations

import importlib.util
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load arc_utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
bbox_h = _ARC.bbox_h
bbox_w = _ARC.bbox_w
colors_present = _ARC.colors_present
crop_bbox = _ARC.crop_bbox
find_components = _ARC.find_components
get_bg = _ARC.get_bg
label_mask = _ARC.label_mask


_QUADRANTS = ("nw", "ne", "sw", "se")
_STEPS = {
    "nw": (-1, -1),
    "ne": (-1, 1),
    "sw": (1, -1),
    "se": (1, 1),
}
_PALETTE = tuple(int(c) for c in Color.NOT_BLACK)
_GEN_ATTEMPTS = 400


def _quadrant_of(r: int, c: int, cross_r: int, cross_c: int) -> str:
    if r == cross_r or c == cross_c:
        raise AssertionError("outside singleton lies on a cross axis")
    if r < cross_r and c < cross_c:
        return "nw"
    if r < cross_r and c > cross_c:
        return "ne"
    if r > cross_r and c < cross_c:
        return "sw"
    return "se"


def _frame_perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rectangular_frame(grid: np.ndarray, component: dict) -> bool:
    frame_bbox = component["bbox"]
    height = bbox_h(frame_bbox)
    width = bbox_w(frame_bbox)
    if height < 5 or width < 5:
        return False
    if int(component["area"]) != 2 * (height + width) - 4:
        return False
    patch = crop_bbox(grid == int(component["color"]), frame_bbox)
    return np.array_equal(patch, _frame_perimeter_mask(height, width))


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = int(get_bg(g))
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    frame_candidates = [comp for comp in components if _is_rectangular_frame(g, comp)]
    assert len(frame_candidates) == 1, "expected exactly one one-pixel rectangular frame"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape[0] >= 3 and inner.shape[1] >= 3, "frame interior must support four rooms"

    cross_candidates: list[tuple[int, int, int]] = []
    for color in colors_present(g, exclude={bg, frame_color}):
        mask = inner == int(color)
        if not mask.any():
            continue
        full_rows = np.where(mask.sum(axis=1) == inner.shape[1])[0]
        full_cols = np.where(mask.sum(axis=0) == inner.shape[0])[0]
        if len(full_rows) != 1 or len(full_cols) != 1:
            continue
        rel_r = int(full_rows[0])
        rel_c = int(full_cols[0])
        expected = np.zeros_like(mask, dtype=bool)
        expected[rel_r, :] = True
        expected[:, rel_c] = True
        if np.array_equal(mask, expected):
            cross_candidates.append((int(color), r0 + 1 + rel_r, c0 + 1 + rel_c))

    assert len(cross_candidates) == 1, "expected exactly one interior cross"
    cross_color, cross_r, cross_c = cross_candidates[0]

    reserved = np.zeros_like(g, dtype=bool)
    reserved[r0, c0 : c1 + 1] = True
    reserved[r1, c0 : c1 + 1] = True
    reserved[r0 : r1 + 1, c0] = True
    reserved[r0 : r1 + 1, c1] = True
    reserved[cross_r, c0 + 1 : c1] = True
    reserved[r0 + 1 : r1, cross_c] = True

    frame_region = g[r0 : r1 + 1, c0 : c1 + 1]
    reserved_region = reserved[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((frame_region == bg) | reserved_region), "frame interior contains unexpected foreground"

    extras_mask = (g != bg) & ~reserved
    lab, count = label_mask(extras_mask, connectivity=8)
    points = []
    for label_id in range(1, count + 1):
        coords = np.argwhere(lab == label_id)
        assert coords.shape[0] == 1, "outside objects must be singleton pixels"
        r, c = map(int, coords[0])
        assert not (r0 <= r <= r1 and c0 <= c <= c1), "singleton must lie outside the frame"
        quadrant = _quadrant_of(r, c, cross_r, cross_c)
        points.append(
            {
                "coord": (r, c),
                "color": int(g[r, c]),
                "quadrant": quadrant,
                "distance": max(abs(r - cross_r), abs(c - cross_c)),
            }
        )

    room_caps = {
        "nw": min(cross_r - r0 - 1, cross_c - c0 - 1),
        "ne": min(cross_r - r0 - 1, c1 - cross_c - 1),
        "sw": min(r1 - cross_r - 1, cross_c - c0 - 1),
        "se": min(r1 - cross_r - 1, c1 - cross_c - 1),
    }
    assert all(cap >= 1 for cap in room_caps.values()), "cross must split the frame into four non-empty rooms"

    grouped = defaultdict(list)
    for point in points:
        grouped[point["quadrant"]].append(int(point["distance"]))
    for quadrant in _QUADRANTS:
        distances = grouped[quadrant]
        assert len(distances) == len(set(distances)), "same-quadrant distances must be unique"
        assert len(distances) <= room_caps[quadrant], "room must have enough diagonal slots"

    return {
        "grid": g,
        "bg": bg,
        "frame_color": frame_color,
        "cross_color": int(cross_color),
        "frame_bbox": (r0, c0, r1, c1),
        "cross": (int(cross_r), int(cross_c)),
        "points": points,
        "room_caps": room_caps,
    }


def _render_output(spec: dict) -> np.ndarray:
    bg = int(spec["bg"])
    frame_color = int(spec["frame_color"])
    cross_color = int(spec["cross_color"])
    r0, c0, r1, c1 = map(int, spec["frame_bbox"])
    cross_r, cross_c = map(int, spec["cross"])

    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color

    rel_r = cross_r - r0
    rel_c = cross_c - c0
    out[rel_r, 1:-1] = cross_color
    out[1:-1, rel_c] = cross_color

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for point in spec["points"]:
        grouped[point["quadrant"]].append((int(point["distance"]), int(point["color"])))

    for quadrant, items in grouped.items():
        dr, dc = _STEPS[quadrant]
        for offset, (_distance, color) in enumerate(sorted(items), start=1):
            rr = cross_r + dr * offset
            cc = cross_c + dc * offset
            out[rr - r0, cc - c0] = int(color)

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Pack outside singleton pixels into the frame room matching their quadrant relative to the cross center.

    Input structure:
    - The grid has a black background, one one-pixel hollow rectangular frame, and one interior cross made from a full row and full column of another color.
    - The cross intersection lies strictly inside the frame and splits the interior into four rooms.
    - Outside the frame there are scattered colored singleton pixels, with unique Chebyshev distances per quadrant relative to the cross intersection.

    Rule:
    - Infer each outside singleton's quadrant from its position relative to the cross intersection, not from the frame center.
    - Within each quadrant, sort the singleton colors by increasing Chebyshev distance to the intersection.
    - Crop to the frame, preserve the frame and cross, then place each quadrant's sorted colors along that room's inward-to-outward diagonal starting from the cell diagonally adjacent to the intersection.

    Key insight:
    - The intersection of the biased cross is the only valid origin for both quadrant assignment and distance ordering, and every room uses the same local diagonal-packing rule mirrored by direction.
    """
    return _render_output(_extract_structure(np.asarray(input_grid, dtype=int)))


def _sort_key_for_quadrant(quadrant: str, coord: tuple[int, int]) -> tuple[int, int]:
    r, c = coord
    if quadrant == "nw":
        return (r, c)
    if quadrant == "ne":
        return (r, -c)
    if quadrant == "sw":
        return (-r, c)
    return (-r, -c)


def _candidate_distances(
    height: int,
    width: int,
    frame_bbox: tuple[int, int, int, int],
    cross: tuple[int, int],
) -> dict[str, dict[int, list[tuple[int, int]]]]:
    r0, c0, r1, c1 = map(int, frame_bbox)
    cross_r, cross_c = map(int, cross)
    buckets = {quadrant: defaultdict(list) for quadrant in _QUADRANTS}

    for r in range(height):
        for c in range(width):
            if r == cross_r or c == cross_c:
                continue
            if r0 - 1 <= r <= r1 + 1 and c0 - 1 <= c <= c1 + 1:
                continue
            quadrant = _quadrant_of(r, c, cross_r, cross_c)
            distance = max(abs(r - cross_r), abs(c - cross_c))
            buckets[quadrant][distance].append((r, c))

    out: dict[str, dict[int, list[tuple[int, int]]]] = {}
    for quadrant in _QUADRANTS:
        out[quadrant] = {}
        for distance, coords in buckets[quadrant].items():
            out[quadrant][int(distance)] = sorted(coords, key=lambda coord: _sort_key_for_quadrant(quadrant, coord))
    return out


def _coords_are_separated(coord: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    r, c = coord
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in used)


def _expected_from_meta(meta: dict) -> np.ndarray:
    spec = {
        "bg": meta["bg"],
        "frame_color": meta["frame_color"],
        "cross_color": meta["cross_color"],
        "frame_bbox": meta["frame_bbox"],
        "cross": meta["cross"],
        "points": meta["points"],
    }
    return _render_output(spec)


def _generate_case(return_meta: bool = False):
    for _ in range(_GEN_ATTEMPTS):
        frame_height = random.randint(7, 18)
        frame_width = random.randint(7, 18)
        top_margin = random.randint(2, 6)
        bottom_margin = random.randint(2, 6)
        left_margin = random.randint(2, 6)
        right_margin = random.randint(2, 6)
        height = frame_height + top_margin + bottom_margin
        width = frame_width + left_margin + right_margin
        if height > 30 or width > 30:
            continue

        r0 = top_margin
        c0 = left_margin
        r1 = r0 + frame_height - 1
        c1 = c0 + frame_width - 1
        cross_r = random.randint(r0 + 2, r1 - 2)
        cross_c = random.randint(c0 + 2, c1 - 2)
        frame_bbox = (r0, c0, r1, c1)
        cross = (cross_r, cross_c)

        capacities = {
            "nw": min(cross_r - r0 - 1, cross_c - c0 - 1),
            "ne": min(cross_r - r0 - 1, c1 - cross_c - 1),
            "sw": min(r1 - cross_r - 1, cross_c - c0 - 1),
            "se": min(r1 - cross_r - 1, c1 - cross_c - 1),
        }
        if min(capacities.values()) < 1:
            continue

        candidates = _candidate_distances(height, width, frame_bbox, cross)
        available_quadrants = [quadrant for quadrant in _QUADRANTS if candidates[quadrant]]
        if len(available_quadrants) < 2:
            continue

        desired_nonempty = min(len(available_quadrants), random.choices([2, 3, 4], weights=[3, 3, 2])[0])
        occupied = random.sample(available_quadrants, desired_nonempty)
        counts = {quadrant: 0 for quadrant in _QUADRANTS}
        for quadrant in occupied:
            max_count = min(capacities[quadrant], len(candidates[quadrant]))
            if max_count < 1:
                break
            if random.random() < 0.3:
                counts[quadrant] = max_count
            else:
                counts[quadrant] = random.randint(1, max_count)
        else:
            if sum(counts.values()) < 3:
                continue
            if max(counts.values()) == 1 and sum(1 for value in counts.values() if value > 0) < 3:
                continue

            used_coords: list[tuple[int, int]] = []
            points = []
            for quadrant in _QUADRANTS:
                count = counts[quadrant]
                if count == 0:
                    continue
                distances = sorted(random.sample(list(candidates[quadrant]), count))
                for distance in distances:
                    choice = None
                    for coord in candidates[quadrant][distance]:
                        if _coords_are_separated(coord, used_coords):
                            choice = coord
                            break
                    if choice is None:
                        break
                    used_coords.append(choice)
                    points.append(
                        {
                            "coord": choice,
                            "quadrant": quadrant,
                            "distance": int(distance),
                        }
                    )
                else:
                    continue
                break
            else:
                frame_color, cross_color = random.sample(_PALETTE, 2)
                point_palette = random.sample([c for c in _PALETTE if c not in {frame_color, cross_color}], random.randint(2, 5))
                for point in points:
                    point["color"] = random.choice(point_palette)
                if len(points) >= 2 and len({point["color"] for point in points}) == len(points) and random.random() < 0.6:
                    points[-1]["color"] = points[0]["color"]

                grid = np.zeros((height, width), dtype=int)
                grid[r0, c0 : c1 + 1] = frame_color
                grid[r1, c0 : c1 + 1] = frame_color
                grid[r0 : r1 + 1, c0] = frame_color
                grid[r0 : r1 + 1, c1] = frame_color
                grid[cross_r, c0 + 1 : c1] = cross_color
                grid[r0 + 1 : r1, cross_c] = cross_color
                for point in points:
                    pr, pc = point["coord"]
                    grid[pr, pc] = int(point["color"])

                meta = {
                    "bg": 0,
                    "frame_color": int(frame_color),
                    "cross_color": int(cross_color),
                    "frame_bbox": frame_bbox,
                    "cross": cross,
                    "points": [dict(point) for point in points],
                }
                expected = _expected_from_meta(meta)
                parsed = _extract_structure(grid)
                if not np.array_equal(_render_output(parsed), expected):
                    continue
                if return_meta:
                    return grid, meta
                return grid

    raise RuntimeError("failed to generate a valid input for task 7")


def generate_input() -> np.ndarray:
    return _generate_case(return_meta=False)


def _validate_output_semantics(input_grid: np.ndarray, output_grid: np.ndarray) -> None:
    spec = _extract_structure(input_grid)
    expected = _render_output(spec)
    assert np.array_equal(output_grid, expected), "output does not match the parsed quadrant-packing rule"
