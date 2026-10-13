from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}

# The 3-pixel L marker occupies three cells of a 2x2 box.
# The missing corner of that 2x2 box encodes the ray direction.
MISSING_TO_DIR = {
    (0, 0): DIRS["U"],  # missing NW
    (0, 1): DIRS["R"],  # missing NE
    (1, 1): DIRS["D"],  # missing SE
    (1, 0): DIRS["L"],  # missing SW
}
DIR_TO_MISSING = {value: key for key, value in MISSING_TO_DIR.items()}

MAX_RETRIES = 400


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _neighbors8(r: int, c: int, shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    h, w = shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _marker_direction(inner_coords: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(inner_coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(inner_coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    local = {(r - r0, c - c0) for r, c in inner_coords}
    all_cells = {(0, 0), (0, 1), (1, 0), (1, 1)}
    missing = tuple(next(iter(all_cells - local)))
    return MISSING_TO_DIR.get(missing)


def _frame_info_from_component(
    grid: np.ndarray, color: int, comp: list[tuple[int, int]]
) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 4 or outer_w < 4:
        return None

    comp_set = set(comp)
    border = _border_coords(r0, c0, r1, c1)
    if comp_set != border:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.shape[0] < 2 or inner.shape[1] < 2:
        return None

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(inner != BG)]
    if len(marker_coords) != 3:
        return None

    marker_values = {int(inner[r, c]) for r, c in marker_coords}
    if len(marker_values) != 1:
        return None
    marker_color = int(next(iter(marker_values)))
    if marker_color == int(color):
        return None

    direction = _marker_direction(marker_coords)
    if direction is None:
        return None

    return {
        "frame_color": int(color),
        "marker_color": marker_color,
        "direction": direction,
        "ray_length": int(min(inner.shape)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "component_key": frozenset(comp_set),
    }


def _ray_cells(
    start: tuple[int, int], direction: tuple[int, int], length: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    for step in range(length):
        rr = int(start[0] + dr * step)
        cc = int(start[1] + dc * step)
        if not (0 <= rr < h and 0 <= cc < w):
            break
        cells.append((rr, cc))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seed pixel to its same-colored frame and emit a marker-colored ray from the seed.

    Input structure:
    - The background is black.
    - Several distinct colors form hollow rectangular frames with 1-pixel borders; each frame interior is otherwise black.
    - Inside every frame there is exactly one 3-pixel L marker occupying three cells of a 2x2 box, in a color different from the frame.
    - Outside the frames, singleton seed pixels reuse frame colors.

    Rule:
    - For each seed pixel, find the unique frame with the same color.
    - Read the ray direction from the frame's L marker: the missing corner of the marker's 2x2 box means up/right/down/left for missing NW/NE/SE/SW.
    - Paint a straight ray starting at the seed cell itself, using the marker color.
    - The ray length equals the frame interior's shorter side length, and any part beyond the grid edge is clipped away.

    Key insight:
    - The frame color links seeds to frames, while the frame geometry gives the ray length and the internal L marker gives the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    components_by_color = {color: _connected_components(g == color) for color in colors}

    frames: dict[int, dict] = {}
    for color, comps in components_by_color.items():
        matches = []
        for comp in comps:
            info = _frame_info_from_component(g, color, comp)
            if info is not None:
                matches.append(info)
        if matches:
            assert len(matches) == 1, "each frame color should identify one frame"
            frames[color] = matches[0]

    assert frames, "expected at least one frame"

    for frame_color, frame in frames.items():
        for comp in components_by_color[frame_color]:
            if frozenset(comp) == frame["component_key"]:
                continue
            if len(comp) != 1:
                continue
            seed = comp[0]
            for rr, cc in _ray_cells(seed, frame["direction"], frame["ray_length"], g.shape):
                out[rr, cc] = int(frame["marker_color"])

    return out


def _place_frames(frame_specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placed: list[dict] = []
    for spec in sorted(frame_specs, key=lambda item: -(item["outer_h"] * item["outer_w"])):
        fh = int(spec["outer_h"])
        fw = int(spec["outer_w"])
        if fh + 2 > h or fw + 2 > w:
            return None
        options = []
        for r0 in range(1, h - fh):
            for c0 in range(1, w - fw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + fh + 1)
                cc1 = min(w, c0 + fw + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        occupied[r0 : r0 + fh, c0 : c0 + fw] = True
        placed.append({**spec, "r0": int(r0), "c0": int(c0)})
    return placed


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    outer_h = int(frame["outer_h"])
    outer_w = int(frame["outer_w"])
    frame_color = int(frame["frame_color"])
    marker_color = int(frame["marker_color"])

    grid[r0, c0 : c0 + outer_w] = frame_color
    grid[r0 + outer_h - 1, c0 : c0 + outer_w] = frame_color
    grid[r0 : r0 + outer_h, c0] = frame_color
    grid[r0 : r0 + outer_h, c0 + outer_w - 1] = frame_color

    box_r = r0 + 1 + int(frame["marker_box_r"])
    box_c = c0 + 1 + int(frame["marker_box_c"])
    missing = DIR_TO_MISSING[frame["direction"]]
    for dr in range(2):
        for dc in range(2):
            if (dr, dc) == missing:
                continue
            grid[box_r + dr, box_c + dc] = marker_color


def _make_frame_specs() -> list[dict]:
    n_frames = random.randint(2, 4)
    palette = random.sample(range(1, 10), 2 * n_frames)
    frame_colors = palette[:n_frames]
    marker_colors = palette[n_frames:]
    directions = list(DIRS.values())

    specs = []
    for frame_color, marker_color in zip(frame_colors, marker_colors):
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 6)
        specs.append(
            {
                "frame_color": int(frame_color),
                "marker_color": int(marker_color),
                "direction": random.choice(directions),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "marker_box_r": random.randint(0, inner_h - 2),
                "marker_box_c": random.randint(0, inner_w - 2),
            }
        )
    return specs


def _frame_box_mask(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        mask[
            r0 : r0 + int(frame["outer_h"]),
            c0 : c0 + int(frame["outer_w"]),
        ] = True
    return mask


def _has_nonzero_8_neighbor(grid: np.ndarray, r: int, c: int) -> bool:
    return any(int(grid[rr, cc]) != BG for rr, cc in _neighbors8(r, c, grid.shape))


def _seed_candidates(
    grid: np.ndarray,
    reserved_output: np.ndarray,
    frame: dict,
) -> tuple[list[tuple[int, int, list[tuple[int, int]]]], list[tuple[int, int, list[tuple[int, int]]]]]:
    unclipped = []
    clipped = []
    frame_color = int(frame["frame_color"])
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BG:
                continue
            if reserved_output[r, c]:
                continue
            if _has_nonzero_8_neighbor(grid, r, c):
                continue
            if any(int(grid[nr, nc]) == frame_color for nr, nc in _neighbors4(r, c, grid.shape)):
                continue
            path = _ray_cells((r, c), frame["direction"], int(frame["ray_length"]), grid.shape)
            if len(path) < 2:
                continue
            if any(bool(reserved_output[rr, cc]) for rr, cc in path):
                continue
            bucket = clipped if len(path) < int(frame["ray_length"]) else unclipped
            bucket.append((int(r), int(c), path))
    return unclipped, clipped


def _build_expected_output(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        for seed in frame["seeds"]:
            for rr, cc in seed["path"]:
                out[rr, cc] = int(frame["marker_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    frame_specs = _make_frame_specs()
    total_area = sum(int(spec["outer_h"] * spec["outer_w"]) for spec in frame_specs)
    min_side = max(
        14,
        int(max(spec["outer_h"] for spec in frame_specs) + 3),
        int(max(spec["outer_w"] for spec in frame_specs) + 3),
        int(np.ceil(np.sqrt(total_area * 3.0))),
    )
    if min_side > 30:
        return None

    grid_h = random.randint(min_side, 30)
    grid_w = random.randint(min_side, 30)
    frames = _place_frames(frame_specs, (grid_h, grid_w))
    if frames is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for frame in frames:
        frame["ray_length"] = int(min(frame["inner_h"], frame["inner_w"]))
        frame["seeds"] = []
        _draw_frame(grid, frame)

    reserved_output = _frame_box_mask(grid.shape, frames)

    seed_counts = [1] * len(frames)
    extra_slots = random.randint(0, len(frames))
    if extra_slots:
        for idx in random.sample(range(len(frames)), extra_slots):
            seed_counts[idx] += 1

    order = sorted(
        range(len(frames)),
        key=lambda idx: (-seed_counts[idx], -int(frames[idx]["ray_length"])),
    )

    for idx in order:
        frame = frames[idx]
        for _ in range(seed_counts[idx]):
            unclipped, clipped = _seed_candidates(grid, reserved_output, frame)
            prefer_clipped = random.random() < 0.5
            if prefer_clipped and clipped:
                pool = clipped
            elif (not prefer_clipped) and unclipped:
                pool = unclipped
            else:
                pool = unclipped or clipped
            if not pool:
                return None
            r, c, path = random.choice(pool)
            grid[r, c] = int(frame["frame_color"])
            frame["seeds"].append({"coord": (int(r), int(c)), "path": path})
            for rr, cc in path:
                reserved_output[rr, cc] = True

    expected = _build_expected_output(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-and-ray sample")
