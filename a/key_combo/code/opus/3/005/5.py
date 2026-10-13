from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("_arc_utils_shared", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load arc_utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
get_bg = _ARC.get_bg
in_bounds = _ARC.in_bounds

RING_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW]
HORIZONTAL_SIDES = ("left", "right")
VERTICAL_SIDES = ("top", "bottom")
ALL_SIDES = HORIZONTAL_SIDES + VERTICAL_SIDES


def _ring_mask_from_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _rect_ring_bbox(comp: dict) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected_area = 2 * (h + w) - 4
    if int(comp["area"]) != expected_area:
        return None
    coords = comp["coords"]
    if not np.all(
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    ):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _extract_structure(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    bg = int(get_bg(grid))
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True)
    rings: list[dict] = []
    pixels: list[dict] = []

    for comp in comps:
        ring_bbox = _rect_ring_bbox(comp)
        if ring_bbox is not None:
            cells = {tuple(map(int, rc)) for rc in comp["coords"]}
            rings.append(
                {
                    "color": int(comp["color"]),
                    "bbox": ring_bbox,
                    "cells": cells,
                }
            )
            continue

        if int(comp["area"]) == 1:
            r, c = map(int, comp["coords"][0])
            pixels.append({"pos": (r, c), "color": int(comp["color"])})

    rings.sort(
        key=lambda ring: (
            ring["bbox"][0],
            ring["bbox"][1],
            -(ring["bbox"][2] - ring["bbox"][0]),
        )
    )
    return rings, pixels, bg


def _movement_step(pos: tuple[int, int], center: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    cr, cc = center
    if abs(c - cc) >= abs(r - cr):
        return (0, 1 if c < cc else -1)
    return (1 if r < cr else -1, 0)


def _scan_base_stop(
    shape: tuple[int, int],
    pos: tuple[int, int],
    target_cells: set[tuple[int, int]],
    center: tuple[int, int],
) -> tuple[int, int] | None:
    dr, dc = _movement_step(pos, center)
    prev = pos
    cur = pos
    while True:
        cur = (cur[0] + dr, cur[1] + dc)
        if not in_bounds(shape, cur[0], cur[1]):
            return None
        if cur in target_cells:
            return prev
        prev = cur
        if (dr and cur[0] == center[0]) or (dc and cur[1] == center[1]):
            return None


def _settle_pixels(
    shape: tuple[int, int],
    rings: list[dict],
    pixels: list[dict],
    center: tuple[int, int],
) -> tuple[list[dict], list[int]]:
    color_to_ring = {ring["color"]: idx for idx, ring in enumerate(rings)}
    settled: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    activated: set[int] = set()

    ordered_pixels = sorted(pixels, key=lambda pixel: (abs(pixel["pos"][0] - center[0]) + abs(pixel["pos"][1] - center[1]), pixel["pos"][0], pixel["pos"][1]))
    for pixel in ordered_pixels:
        ring_idx = color_to_ring.get(pixel["color"])
        if ring_idx is None:
            continue
        target_ring = rings[ring_idx]
        base_stop = _scan_base_stop(shape, pixel["pos"], target_ring["cells"], center)
        if base_stop is None:
            continue
        dr, dc = _movement_step(pixel["pos"], center)
        stop = base_stop
        while stop in occupied:
            stop = (stop[0] - dr, stop[1] - dc)
            if not in_bounds(shape, stop[0], stop[1]):
                stop = None
                break
        if stop is None:
            continue
        occupied.add(stop)
        activated.add(ring_idx)
        settled.append(
            {
                "start": pixel["pos"],
                "stop": stop,
                "color": pixel["color"],
                "ring_idx": ring_idx,
            }
        )

    return settled, sorted(activated)


def _rotated_ring_colors(rings: list[dict], active_ring_indices: list[int]) -> list[int]:
    new_colors = [ring["color"] for ring in rings]
    if not active_ring_indices:
        return new_colors
    for i, ring_idx in enumerate(active_ring_indices):
        next_idx = active_ring_indices[(i + 1) % len(active_ring_indices)]
        new_colors[ring_idx] = rings[next_idx]["color"]
    return new_colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide loose pixels inward to their matching concentric rings, then rotate colors only among the hit rings.

    Input structure:
    - The background is black.
    - The center contains 3 or 4 one-cell-thick concentric rectangular rings with different colors.
    - Several isolated single pixels sit outside the outer ring, and each such pixel color matches one ring color.

    Rule:
    - Each loose pixel moves toward the grid center along its dominant axis: horizontal if its horizontal center-distance is at least its vertical center-distance, otherwise vertical.
    - The pixel ignores background and non-matching ring colors while moving, and stops on the background cell immediately outside the first same-color ring cell on that path; if that stop cell is already occupied by an earlier settled pixel, it stacks one cell farther outward.
    - Pixels settle from near the center to far from the center, using Manhattan distance to the grid center.
    - After all pixels settle, only the rings that received at least one pixel are activated; ordered from outer to inner, the activated rings cyclically take the next activated ring's color, while non-activated rings keep their original colors.
    - Settled loose pixels keep their original colors.

    Key insight:
    - The ring recoloring cannot be decided from the rings alone: first resolve the selective same-color inward slides, then rotate colors only across the activated nesting subsequence.
    """
    g = np.asarray(grid, dtype=int)
    rings, pixels, bg = _extract_structure(g)
    out = g.copy()
    for pixel in pixels:
        out[pixel["pos"]] = bg

    center = (g.shape[0] // 2, g.shape[1] // 2)
    settled, active_ring_indices = _settle_pixels(g.shape, rings, pixels, center)
    ring_colors = _rotated_ring_colors(rings, active_ring_indices)

    for ring_idx, ring in enumerate(rings):
        for cell in ring["cells"]:
            out[cell] = ring_colors[ring_idx]
    for item in settled:
        out[item["stop"]] = item["color"]
    return out


def _draw_ring(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    mask = _ring_mask_from_bbox(grid.shape, bbox)
    grid[mask] = color


def _is_isolated(pos: tuple[int, int], others: set[tuple[int, int]]) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > 1 for r, c in others)


def _side_slots(side: str, shape: tuple[int, int], outer_margin: int) -> list[int]:
    h, w = shape
    if side == "left":
        return [c for c in (0, 2, 1, 3) if c < outer_margin]
    if side == "right":
        return [c for c in (w - 1, w - 3, w - 2, w - 4) if c >= w - outer_margin]
    if side == "top":
        return [r for r in (0, 2, 1, 3) if r < outer_margin]
    if side == "bottom":
        return [r for r in (h - 1, h - 3, h - 2, h - 4) if r >= h - outer_margin]
    raise ValueError(side)


def _candidate_lanes(ring_bbox: tuple[int, int, int, int], side: str) -> list[int]:
    r0, c0, r1, c1 = ring_bbox
    if side in HORIZONTAL_SIDES:
        lanes = list(range(r0, r1 + 1))
    else:
        lanes = list(range(c0, c1 + 1))
    random.shuffle(lanes)
    return lanes


def _place_group(
    shape: tuple[int, int],
    center: tuple[int, int],
    outer_margin: int,
    ring_bbox: tuple[int, int, int, int],
    ring_idx: int,
    side: str,
    count: int,
    used_starts: set[tuple[int, int]],
    used_lanes: set[tuple[int, str, int]],
) -> tuple[list[tuple[int, int]], int] | None:
    for lane in _candidate_lanes(ring_bbox, side):
        if (ring_idx, side, lane) in used_lanes:
            continue
        starts: list[tuple[int, int]] = []
        for slot in _side_slots(side, shape, outer_margin):
            if side == "left":
                candidate = (lane, slot)
            elif side == "right":
                candidate = (lane, slot)
            elif side == "top":
                candidate = (slot, lane)
            else:
                candidate = (slot, lane)
            if not _is_isolated(candidate, used_starts | set(starts)):
                continue
            dr, dc = _movement_step(candidate, center)
            if side == "left" and (dr, dc)[1] != 1:
                continue
            if side == "right" and (dr, dc)[1] != -1:
                continue
            if side == "top" and (dr, dc)[0] != 1:
                continue
            if side == "bottom" and (dr, dc)[0] != -1:
                continue
            starts.append(candidate)
            if len(starts) == count:
                return starts, lane
    return None


def _choose_active_subset(ring_count: int, require_outer: bool) -> list[int]:
    candidates: list[list[int]] = []
    weights: list[int] = []
    indices = list(range(ring_count))
    for mask in range(1 << ring_count):
        subset = [idx for idx in indices if mask & (1 << idx)]
        if len(subset) < 2 or len(subset) >= ring_count:
            continue
        if require_outer and 0 not in subset:
            continue
        contiguous = subset == list(range(subset[0], subset[-1] + 1))
        candidates.append(subset)
        weights.append(1 if contiguous else 3)
    return random.choices(candidates, weights=weights, k=1)[0]


def _planned_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = grid.copy()
    for pixel in meta["pixels"]:
        out[pixel["start"]] = Color.BLACK
    rotated_colors = _rotated_ring_colors(meta["rings"], meta["active_ring_indices"])
    for ring_idx, ring in enumerate(meta["rings"]):
        for cell in ring["cells"]:
            out[cell] = rotated_colors[ring_idx]
    for pixel in meta["pixels"]:
        out[pixel["stop"]] = pixel["color"]
    return out


def _validate_case(grid: np.ndarray, meta: dict) -> bool:
    out = main(grid)
    expected = _planned_output(grid, meta)
    if not np.array_equal(out, expected):
        return False
    if np.array_equal(out, grid):
        return False
    active = meta["active_ring_indices"]
    if len(active) < 2 or len(active) >= len(meta["rings"]):
        return False
    rotated_colors = _rotated_ring_colors(meta["rings"], active)
    changed = 0
    unchanged = 0
    for ring_idx, ring in enumerate(meta["rings"]):
        sample_cell = next(iter(ring["cells"]))
        if int(out[sample_cell]) == int(meta["rings"][ring_idx]["color"]):
            unchanged += 1
        if int(out[sample_cell]) == int(rotated_colors[ring_idx]) and rotated_colors[ring_idx] != ring["color"]:
            changed += 1
    if changed == 0 or unchanged == 0:
        return False
    ring_cells = set().union(*(ring["cells"] for ring in meta["rings"]))
    if len({pixel["stop"] for pixel in meta["pixels"]}) != len(meta["pixels"]):
        return False
    for pixel in meta["pixels"]:
        if pixel["stop"] in ring_cells:
            return False
        if tuple(pixel["start"]) == tuple(pixel["stop"]):
            continue
        if int(out[pixel["start"]]) != Color.BLACK:
            return False
        if int(out[pixel["stop"]]) != pixel["color"]:
            return False
    return True


def _generate_case() -> tuple[np.ndarray, dict]:
    for _ in range(400):
        ring_count = random.choices([3, 4], weights=[3, 2], k=1)[0]
        want_stack = random.random() < 0.65
        outer_margin = 4 if want_stack else random.choice([3, 4])
        outer_size = 4 * ring_count - 1
        size = outer_size + 2 * outer_margin
        center = (size // 2, size // 2)
        grid = np.zeros((size, size), dtype=int)

        ring_colors = random.sample(RING_COLORS, k=ring_count)
        rings: list[dict] = []
        for idx in range(ring_count):
            offset = outer_margin + 2 * idx
            ring_size = outer_size - 4 * idx
            bbox = (offset, offset, offset + ring_size - 1, offset + ring_size - 1)
            _draw_ring(grid, bbox, ring_colors[idx])
            rings.append(
                {
                    "color": ring_colors[idx],
                    "bbox": bbox,
                    "cells": {tuple(map(int, rc)) for rc in np.argwhere(_ring_mask_from_bbox(grid.shape, bbox))},
                }
            )

        active_ring_indices = _choose_active_subset(ring_count, require_outer=want_stack)

        groups: list[dict] = []
        if want_stack:
            groups.append({"ring_idx": 0, "side": random.choice(ALL_SIDES), "count": 2})
        if not any(group["side"] in HORIZONTAL_SIDES for group in groups):
            groups.append(
                {
                    "ring_idx": random.choice(active_ring_indices),
                    "side": random.choice(HORIZONTAL_SIDES),
                    "count": 1,
                }
            )
        if not any(group["side"] in VERTICAL_SIDES for group in groups):
            groups.append(
                {
                    "ring_idx": random.choice(active_ring_indices),
                    "side": random.choice(VERTICAL_SIDES),
                    "count": 1,
                }
            )
        represented = {group["ring_idx"] for group in groups}
        for ring_idx in active_ring_indices:
            if ring_idx not in represented:
                groups.append(
                    {
                        "ring_idx": ring_idx,
                        "side": random.choice(ALL_SIDES),
                        "count": 1,
                    }
                )

        min_pixels = max(4, sum(group["count"] for group in groups))
        total_pixels = random.randint(min_pixels, 8)
        while sum(group["count"] for group in groups) < total_pixels:
            groups.append(
                {
                    "ring_idx": random.choice(active_ring_indices),
                    "side": random.choice(ALL_SIDES),
                    "count": 1,
                }
            )

        random.shuffle(groups)
        used_starts: set[tuple[int, int]] = set()
        used_lanes: set[tuple[int, str, int]] = set()
        pixels_meta: list[dict] = []
        placement_ok = True
        for group in groups:
            ring_idx = group["ring_idx"]
            side = group["side"]
            count = group["count"]
            placed = _place_group(
                grid.shape,
                center,
                outer_margin,
                rings[ring_idx]["bbox"],
                ring_idx,
                side,
                count,
                used_starts,
                used_lanes,
            )
            if placed is None:
                placement_ok = False
                break
            starts, lane = placed
            used_lanes.add((ring_idx, side, lane))
            for start in starts:
                grid[start] = rings[ring_idx]["color"]
                used_starts.add(start)

        if not placement_ok:
            continue

        loose_pixels = [{"pos": start, "color": int(grid[start])} for start in sorted(used_starts)]
        settled, solved_active = _settle_pixels(grid.shape, rings, loose_pixels, center)
        if solved_active != sorted(active_ring_indices):
            continue

        stop_map = {item["start"]: item for item in settled}
        for start in used_starts:
            item = stop_map.get(start)
            if item is None:
                placement_ok = False
                break
            pixels_meta.append(item)
        if not placement_ok:
            continue

        meta = {
            "rings": rings,
            "pixels": pixels_meta,
            "active_ring_indices": sorted(active_ring_indices),
        }
        if _validate_case(grid, meta):
            return grid, meta

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    grid, _meta = _generate_case()
    return grid
