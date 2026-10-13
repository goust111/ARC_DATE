from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components, random_sprite


OBJECT_COLORS = (Color.BLUE, Color.GREEN, Color.YELLOW, Color.ORANGE, Color.TEAL, Color.MAROON)
OBJECT_DIMS = ((2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2))
MAX_GEN_ATTEMPTS = 600


def _is_frame(comp: dict) -> bool:
    if int(comp["color"]) != Color.GREY:
        return False
    h = bbox_h(comp["bbox"])
    w = bbox_w(comp["bbox"])
    return h >= 5 and w >= 5 and int(comp["area"]) == 2 * (h + w) - 4


def _center_col(comp: dict) -> int:
    _r0, c0, _r1, c1 = map(int, comp["bbox"])
    return (c0 + c1) // 2


def _interior_cells(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, frame["bbox"])
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _scene_parts(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    comps = find_components(grid, bg=Color.BLACK, connectivity=4, monochromatic=True)
    frames = [comp for comp in comps if _is_frame(comp)]
    objects = [
        comp
        for comp in comps
        if int(comp["color"]) not in {Color.BLACK, Color.RED, Color.GREY}
    ]
    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    objects.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    return objects, frames


def main(grid: np.ndarray) -> np.ndarray:
    """Project each upper object downward and render, inside every gray frame, how many rays pass through it.

    Input structure:
    - The background is black, with several small single-color foreground objects in the upper part.
    - The lower part contains 2-3 gray 1-pixel-thick rectangular frames whose interiors may contain red distractor pixels.
    - Additional red distractor pixels may also appear outside the frames.

    Rule:
    - For each upper foreground object, compute the horizontal center column of its bounding box using floor((left + right) / 2).
    - Treat that center column as a vertical downward ray and, for each gray frame, count whether the ray column lies inside the frame's interior column range.
    - In each frame interior, fill the first N cells in row-major order with yellow, where N is the number of crossing rays; fill the remaining interior cells black.
    - Erase all red distractors, while preserving the upper objects and the gray frame borders.

    Key insight:
    - The visible red pixels are decoys: the true count comes from projecting object-center columns into frame interiors, not from counting any marks already inside a frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[out == Color.RED] = Color.BLACK

    objects, frames = _scene_parts(g)
    center_cols = [_center_col(comp) for comp in objects]

    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        count = sum(c0 + 1 <= col <= c1 - 1 for col in center_cols)
        cells = _interior_cells(frame)
        for idx, (r, c) in enumerate(cells):
            out[r, c] = Color.YELLOW if idx < count else Color.BLACK

    return out


def _random_split(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _make_frame_patch(inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), Color.BLACK, dtype=int)
    patch[0, :] = Color.GREY
    patch[-1, :] = Color.GREY
    patch[:, 0] = Color.GREY
    patch[:, -1] = Color.GREY
    return patch


def _choose_frame_specs(height: int, width: int, n_frames: int) -> list[dict] | None:
    for _ in range(200):
        inners = [(random.randint(3, 5), random.randint(3, 5)) for _ in range(n_frames)]
        outer_heights = [h + 2 for h, _w in inners]
        outer_widths = [w + 2 for _h, w in inners]
        min_gap = 2
        span = sum(outer_widths) + min_gap * (n_frames - 1)
        if span + 2 > width:
            continue

        top_min = max(8, height // 2)
        if any(top_min > height - oh - 1 for oh in outer_heights):
            continue

        extras = _random_split(width - span - 2, n_frames + 1)
        left = 1 + extras[0]
        gap_extras = extras[1:-1]

        specs = []
        c = left
        for idx, ((inner_h, inner_w), outer_h, outer_w) in enumerate(
            zip(inners, outer_heights, outer_widths, strict=False)
        ):
            top = random.randint(top_min, height - outer_h - 1)
            specs.append(
                {
                    "bbox": (top, c, top + outer_h - 1, c + outer_w - 1),
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                }
            )
            c += outer_w
            if idx < n_frames - 1:
                c += min_gap + gap_extras[idx]
        return specs
    return None


def _sample_centers(width: int, n_objects: int, intervals: list[tuple[int, int]]) -> list[int] | None:
    for _ in range(400):
        available = set(range(1, width - 1))
        centers = []
        while available and len(centers) < n_objects:
            c = random.choice(tuple(available))
            centers.append(c)
            for banned in range(c - 1, c + 2):
                available.discard(banned)
        if len(centers) != n_objects:
            continue

        centers.sort()
        counts = [sum(lo <= c <= hi for c in centers) for lo, hi in intervals]
        if any(count == 0 for count in counts):
            continue
        if len(set(counts)) == 1:
            continue
        if sum(counts) >= n_objects:
            continue
        return centers
    return None


def _make_object_patch(color: int, height: int, width: int) -> np.ndarray | None:
    for _ in range(120):
        patch = random_sprite(
            height,
            width,
            color_palette=[int(color)],
            background=Color.BLACK,
            connectivity=4,
        )
        area = int(np.sum(patch != Color.BLACK))
        if 3 <= area <= 8:
            return patch.astype(int)
    return None


def _has_clearance(occupied: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    mask = patch != Color.BLACK
    coords = np.argwhere(mask)
    for dr, dc in coords:
        rr = int(top + dr)
        cc = int(left + dc)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = grid[top : top + h, left : left + w]
    mask = patch != Color.BLACK
    view[mask] = patch[mask]
    occupied[top : top + h, left : left + w] |= mask


def _place_objects(grid: np.ndarray, centers: list[int], frame_top: int) -> bool:
    occupied = grid != Color.BLACK
    max_bottom = frame_top - 3
    colors = random.sample(OBJECT_COLORS, len(centers))

    for center, color in zip(centers, colors, strict=False):
        success = False
        nearest_gap = min((abs(center - other) for other in centers if other != center), default=99)
        dims = list(OBJECT_DIMS)
        if nearest_gap <= 2:
            dims = [dim for dim in dims if dim[1] <= 3]
        random.shuffle(dims)

        for _ in range(200):
            height, width = random.choice(dims)
            left = center - (width - 1) // 2
            if left < 1 or left + width > grid.shape[1] - 1:
                continue
            top_max = max_bottom - height + 1
            if top_max < 1:
                continue
            top = random.randint(1, top_max)
            patch = _make_object_patch(color, height, width)
            if patch is None:
                continue
            if not _has_clearance(occupied, top, left, patch):
                continue
            _stamp(grid, occupied, patch, top, left)
            success = True
            break

        if not success:
            return False

    return True


def _paint_frame(grid: np.ndarray, spec: dict) -> None:
    patch = _make_frame_patch(spec["inner_h"], spec["inner_w"])
    r0, c0, _r1, _c1 = map(int, spec["bbox"])
    grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch


def _frame_intervals_from_specs(specs: list[dict]) -> list[tuple[int, int]]:
    return [(int(spec["bbox"][1]) + 1, int(spec["bbox"][3]) - 1) for spec in specs]


def _place_red_distractors(grid: np.ndarray, frames: list[dict], ray_counts: list[int]) -> None:
    red_counts = []
    interior_lists = [_interior_cells(frame) for frame in frames]
    for cells in interior_lists:
        red_counts.append(random.randint(1, min(len(cells), 6)))

    if all(red_n == ray_n for red_n, ray_n in zip(red_counts, ray_counts, strict=False)):
        first_cap = len(interior_lists[0])
        red_counts[0] = 1 if ray_counts[0] != 1 else min(first_cap, 2)

    for cells, red_n in zip(interior_lists, red_counts, strict=False):
        for r, c in random.sample(cells, red_n):
            grid[r, c] = Color.RED

    inside = {cell for cells in interior_lists for cell in cells}
    outside_cells = [
        (r, c)
        for r, c in np.argwhere(grid == Color.BLACK)
        if (int(r), int(c)) not in inside
    ]
    outside_red = random.randint(1, min(4, len(outside_cells)))
    for r, c in random.sample(outside_cells, outside_red):
        grid[r, c] = Color.RED


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    objects, frames = _scene_parts(grid)
    if not (2 <= len(frames) <= 3 and 4 <= len(objects) <= 6):
        return False

    centers = [_center_col(comp) for comp in objects]
    if len(set(centers)) != len(centers):
        return False

    max_object_bottom = max(int(comp["bbox"][2]) for comp in objects)
    min_frame_top = min(int(frame["bbox"][0]) for frame in frames)
    if min_frame_top - max_object_bottom - 1 < 2:
        return False

    ray_counts = []
    red_inside_counts = []
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        ray_count = sum(c0 + 1 <= col <= c1 - 1 for col in centers)
        if ray_count < 1:
            return False
        cells = _interior_cells(frame)
        red_inside = sum(int(grid[r, c] == Color.RED) for r, c in cells)
        if red_inside < 1:
            return False
        ray_counts.append(ray_count)
        red_inside_counts.append(red_inside)

    if len(set(ray_counts)) == 1:
        return False
    if sum(ray_counts) >= len(objects):
        return False
    if all(a == b for a, b in zip(ray_counts, red_inside_counts, strict=False)):
        return False

    frame_interiors = {cell for frame in frames for cell in _interior_cells(frame)}
    outside_reds = [
        (int(r), int(c))
        for r, c in np.argwhere(grid == Color.RED)
        if (int(r), int(c)) not in frame_interiors
    ]
    if not outside_reds:
        return False

    out = main(grid)
    if np.any(out == Color.RED):
        return False

    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:
            if int(out[r, c]) != color:
                return False

    for frame, ray_count in zip(frames, ray_counts, strict=False):
        r0, c0, r1, c1 = map(int, frame["bbox"])
        border_cells = []
        for c in range(c0, c1 + 1):
            border_cells.append((r0, c))
            border_cells.append((r1, c))
        for r in range(r0 + 1, r1):
            border_cells.append((r, c0))
            border_cells.append((r, c1))
        if any(int(out[r, c]) != Color.GREY for r, c in border_cells):
            return False

        cells = _interior_cells(frame)
        for idx, (r, c) in enumerate(cells):
            expected = Color.YELLOW if idx < ray_count else Color.BLACK
            if int(out[r, c]) != expected:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 22)
    width = random.randint(20, 28)
    n_frames = random.randint(2, 3)
    n_objects = random.randint(4, 6)

    frame_specs = _choose_frame_specs(height, width, n_frames)
    if frame_specs is None:
        return None

    intervals = _frame_intervals_from_specs(frame_specs)
    centers = _sample_centers(width, n_objects, intervals)
    if centers is None:
        return None

    grid = np.full((height, width), Color.BLACK, dtype=int)
    for spec in frame_specs:
        _paint_frame(grid, spec)

    min_frame_top = min(int(spec["bbox"][0]) for spec in frame_specs)
    if not _place_objects(grid, centers, min_frame_top):
        return None

    _place_red_distractors(
        grid,
        [{"bbox": tuple(map(int, spec["bbox"]))} for spec in frame_specs],
        [sum(lo <= c <= hi for c in centers) for lo, hi in intervals],
    )

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
