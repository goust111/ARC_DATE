from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 7  # orange
COLOR_POOL = tuple(color for color in range(10) if color != BG)

FRAME_COUNT_OPTIONS = (2, 3, 4)
INTERIOR_MIN = 4
INTERIOR_MAX = 12
MAX_DROP = 3
MIN_MASTER_POINTS = 5
MAX_MASTER_POINTS = 10
MIN_VISIBLE_POINTS = 3
PLACEMENT_GAP = 1
GRID_SIDE_MAX = 30
MAX_ATTEMPTS = 600


def _map_index(idx: int, src_size: int, dst_size: int) -> int:
    assert 0 <= idx < src_size
    assert src_size > 0 and dst_size > 0
    return ((2 * int(idx) + 1) * int(dst_size)) // (2 * int(src_size))


def _boxes_separated(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    gap: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = map(int, box_a)
    br0, bc0, br1, bc1 = map(int, box_b)
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(counts.most_common(1)[0][0])


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one non-background frame color"

    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h >= 3 and w >= 3, "each color must define a rectangular frame"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros((h, w), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        assert np.all(patch[border_mask] == color), "frame border must be complete"
        assert np.all(
            (patch[~border_mask] == bg) | (patch[~border_mask] == color)
        ), "frame interior may only contain background or same-color dots"

        interior = patch[1:-1, 1:-1] == color
        assert interior.any(), "each frame must contain interior dots"

        points = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "outer_area": int(h * w),
                "interior_shape": (int(h - 2), int(w - 2)),
                "points": points,
            }
        )

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            assert _boxes_separated(
                frames[i]["bbox"], frames[j]["bbox"], gap=0
            ), "frames must not overlap or nest"

    frames.sort(key=lambda item: int(item["outer_area"]), reverse=True)
    assert len(frames) >= 2, "expected multiple frames"
    for a, b in zip(frames, frames[1:]):
        assert int(a["outer_area"]) > int(
            b["outer_area"]
        ), "frame sizes must have a unique descending order"
    return frames


def _project_points(
    points: list[tuple[int, int]] | set[tuple[int, int]],
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
) -> set[tuple[int, int]]:
    src_h, src_w = map(int, src_shape)
    dst_h, dst_w = map(int, dst_shape)
    return {
        (_map_index(int(r), src_h, dst_h), _map_index(int(c), src_w, dst_w))
        for r, c in points
    }


def _render_output(frames: list[dict], bg: int) -> np.ndarray:
    out_h, out_w = map(int, frames[0]["interior_shape"])
    out = np.full((out_h, out_w), int(bg), dtype=int)
    for frame in frames:
        color = int(frame["color"])
        projected = _project_points(
            frame["points"], frame["interior_shape"], (out_h, out_w)
        )
        for r, c in projected:
            out[int(r), int(c)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge same-pattern scatter dots from multiple rectangular frames into the
    largest frame's interior coordinate system.

    Input structure:
    - Background is orange.
    - Several non-overlapping one-pixel-thick rectangular frames appear in
      different colors.
    - Inside each frame, same-color scatter pixels use the same normalized
      coordinates across frame sizes.

    Rule:
    - Sort frames by area from largest to smallest.
    - Create an orange canvas the size of the largest frame's interior.
    - Project each frame's interior dots into that canvas by normalized position.
    - Draw them in order so later, smaller frames overwrite earlier ones.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    frames = _extract_frames(g, bg)
    out_h, out_w = map(int, frames[0]["interior_shape"])
    for frame in frames[1:]:
        h, w = map(int, frame["interior_shape"])
        assert h <= out_h and w <= out_w, "largest frame must define output size"

    return _render_output(frames, bg)


def _gen_dim_chain(count: int) -> list[int]:
    largest = random.randint(INTERIOR_MIN + count - 1, INTERIOR_MAX)
    dims = [largest]
    current = largest
    for i in range(1, count):
        min_next = INTERIOR_MIN + (count - 1 - i)
        max_drop = min(MAX_DROP, current - min_next)
        drop = random.randint(1, max_drop)
        current -= drop
        dims.append(current)
    return dims


def _sample_master_points(height: int, width: int) -> list[tuple[int, int]]:
    all_cells = [(r, c) for r in range(height) for c in range(width)]
    max_points = min(MAX_MASTER_POINTS, max(MIN_MASTER_POINTS, (height * width) // 4))
    point_count = random.randint(MIN_MASTER_POINTS, max_points)
    points = random.sample(all_cells, point_count)
    rows = {r for r, _ in points}
    cols = {c for _, c in points}
    if len(rows) < 2 or len(cols) < 2:
        return []
    return points


def _place_boxes(shapes: list[tuple[int, int]], grid_h: int, grid_w: int) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for box_h, box_w in shapes:
        placed = False
        for _ in range(160):
            r0 = random.randint(1, grid_h - box_h - 1)
            c0 = random.randint(1, grid_w - box_w - 1)
            candidate = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
            if all(
                _boxes_separated(candidate, existing, gap=PLACEMENT_GAP)
                for existing in boxes
            ):
                boxes.append(candidate)
                placed = True
                break
        if not placed:
            return None
    return boxes


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    heights = _gen_dim_chain(frame_count)
    widths = _gen_dim_chain(frame_count)
    largest_shape = (int(heights[0]), int(widths[0]))

    master_points = _sample_master_points(*largest_shape)
    if not master_points:
        return None

    colors = random.sample(COLOR_POOL, frame_count)
    frames: list[dict] = []
    projected_sets: list[frozenset[tuple[int, int]]] = []
    for color, h, w in zip(colors, heights, widths):
        local_points = {
            (
                _map_index(int(r), largest_shape[0], int(h)),
                _map_index(int(c), largest_shape[1], int(w)),
            )
            for r, c in master_points
        }
        if len(local_points) < MIN_VISIBLE_POINTS:
            return None

        frame = {
            "color": int(color),
            "outer_area": int((h + 2) * (w + 2)),
            "interior_shape": (int(h), int(w)),
            "points": sorted(local_points),
        }
        frames.append(frame)
        projected_sets.append(
            frozenset(_project_points(frame["points"], frame["interior_shape"], largest_shape))
        )

    if len(set(projected_sets)) != frame_count:
        return None

    out = _render_output(frames, BG)
    visible_colors = {int(color) for color in np.unique(out) if int(color) != BG}
    if visible_colors != {int(frame["color"]) for frame in frames}:
        return None

    outer_shapes = [
        (int(frame["interior_shape"][0] + 2), int(frame["interior_shape"][1] + 2))
        for frame in frames
    ]
    area_sum = sum(int(h * w) for h, w in outer_shapes)
    min_h = max(max(h for h, _ in outer_shapes) + 4, int(area_sum**0.5) + 4)
    min_w = max(max(w for _, w in outer_shapes) + 4, int(area_sum**0.5) + 4)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    grid_h = random.randint(min_h, GRID_SIDE_MAX)
    grid_w = random.randint(min_w, GRID_SIDE_MAX)
    boxes = _place_boxes(outer_shapes, grid_h, grid_w)
    if boxes is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for frame, (r0, c0, r1, c1) in zip(frames, boxes):
        color = int(frame["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        for pr, pc in frame["points"]:
            grid[r0 + 1 + int(pr), c0 + 1 + int(pc)] = color
        frame["bbox"] = (int(r0), int(c0), int(r1), int(c1))

    solved = main(grid)
    if not np.array_equal(solved, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
