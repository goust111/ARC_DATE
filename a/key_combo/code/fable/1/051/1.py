from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, in_bounds, neighbors4


SIDES = ("top", "bottom", "left", "right")
OUT_STEPS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

FRAME_SIZE_RANGE = (5, 8)
RAY_LEN_RANGE = (2, 5)
DOT_COUNT_RANGE = (2, 5)
GRID_SIDE_MIN = 14
MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 120
DOT_TRIES = 160
FRAME_COUNT_CHOICES = [1, 2, 3, 4]
FRAME_COUNT_WEIGHTS = [1, 3, 4, 2]


def _bbox_area_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }


def _expand_cells(
    cells: set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    height, width = shape
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width:
                    out.add((rr, cc))
    return out


def _ray_cells(
    shape: tuple[int, int],
    gap: tuple[int, int],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    height, width = shape
    dr, dc = step
    r = gap[0] + dr
    c = gap[1] + dc
    out = []
    while 0 <= r < height and 0 <= c < width:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _gap_side_and_step(
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
) -> tuple[str, tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top", OUT_STEPS["top"]
    if gr == r1 and c0 < gc < c1:
        return "bottom", OUT_STEPS["bottom"]
    if gc == c0 and r0 < gr < r1:
        return "left", OUT_STEPS["left"]
    if gc == c1 and r0 < gr < r1:
        return "right", OUT_STEPS["right"]
    raise AssertionError("gap must be a non-corner border cell")


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }


def _bfs_distances(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
) -> dict[tuple[int, int], int]:
    r0, c0, r1, c1 = bbox
    traversable = np.zeros(shape, dtype=bool)
    traversable[r0 + 1 : r1, c0 + 1 : c1] = True
    traversable[gap] = True

    distances = {gap: 0}
    queue: deque[tuple[int, int]] = deque([gap])
    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(shape, nr, nc):
                continue
            if not traversable[nr, nc]:
                continue
            if (nr, nc) in distances:
                continue
            distances[(nr, nc)] = distances[(r, c)] + 1
            queue.append((nr, nc))
    return distances


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    frames = []
    frame_cells: set[tuple[int, int]] = set()
    for comp in components:
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        coords = {tuple(map(int, rc)) for rc in comp["coords"]}
        border = _border_cells(bbox)
        if len(coords) != len(border) - 1:
            continue
        gaps = list(border - coords)
        if len(gaps) != 1:
            continue

        gap = gaps[0]
        try:
            side, out_step = _gap_side_and_step(bbox, gap)
        except AssertionError:
            continue

        frame = {
            "bbox": bbox,
            "color": color,
            "frame_cells": coords,
            "gap": gap,
            "side": side,
            "out_step": out_step,
        }
        frames.append(frame)
        frame_cells |= coords

    assert frames, "expected one or more one-gap rectangular frames"
    assert len({frame["color"] for frame in frames}) == 1, "all frames must share one color"

    dot_cells: set[tuple[int, int]] = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        dots = []
        distances = _bfs_distances(g.shape, frame["bbox"], frame["gap"])
        expected_reachable = (r1 - r0 - 1) * (c1 - c0 - 1) + 1
        assert len(distances) == expected_reachable, "frame interior must be open to the gap"

        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                assert value != color, "interior scatter must not reuse the frame color"
                if value == bg:
                    continue
                assert (r, c) in distances, "interior scatter must reach the gap through the interior"
                dots.append({"pos": (r, c), "color": value, "dist": distances[(r, c)]})
                dot_cells.add((r, c))

        assert dots, "each frame must contain at least one interior scatter pixel"
        dists = [dot["dist"] for dot in dots]
        assert len(set(dists)) == len(dists), "distances to the gap must be unique within a frame"

        ray = _ray_cells(g.shape, frame["gap"], frame["out_step"])
        assert len(ray) >= len(dots), "the outward ray must fit the whole sequence"

        frame["dots"] = sorted(dots, key=lambda dot: int(dot["dist"]))
        frame["ray"] = ray

    valid = frame_cells | dot_cells
    for r, c in np.argwhere(g != bg):
        assert (int(r), int(c)) in valid, "all non-background cells must be frame cells or interior scatter"

    return bg, frames


def _render_from_records(
    grid: np.ndarray,
    bg: int,
    frames: list[dict],
    *,
    order_key=None,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    paint_cells: set[tuple[int, int]] = set()

    for frame in frames:
        for dot in frame["dots"]:
            out[dot["pos"]] = bg

        ordered = frame["dots"]
        if order_key is not None:
            ordered = sorted(frame["dots"], key=order_key)
        for index, dot in enumerate(ordered):
            target = frame["ray"][index]
            assert target not in paint_cells, "different frames must not paint the same output cell"
            assert int(out[target]) == bg, "sequence must flow only into background cells"
            out[target] = int(dot["color"])
            paint_cells.add(target)

    return out


def _nontrivial_distance_order(frame: dict) -> bool:
    cells = [tuple(dot["pos"]) for dot in frame["dots"]]
    distance_order = cells
    alternatives = [
        sorted(cells),
        sorted(cells, key=lambda p: (-p[0], p[1])),
        sorted(cells, key=lambda p: (p[1], p[0])),
        sorted(cells, key=lambda p: (-p[1], p[0])),
    ]
    return all(distance_order != alt for alt in alternatives)


def _validate_generated_grid(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg, frames = _extract_frames(g)
    expected = _render_from_records(g, bg, frames)
    actual = main(g)
    assert np.array_equal(actual, expected), "solver output disagrees with the scene semantics"

    assert any(_nontrivial_distance_order(frame) for frame in frames), (
        "need at least one frame whose distance order differs from simple row/column scanning"
    )

    row_major = _render_from_records(g, bg, frames, order_key=lambda dot: tuple(dot["pos"]))
    col_major = _render_from_records(
        g,
        bg,
        frames,
        order_key=lambda dot: (int(dot["pos"][1]), int(dot["pos"][0])),
    )
    assert not np.array_equal(expected, row_major), "scene should reject row-major shortcuts"
    assert not np.array_equal(expected, col_major), "scene should reject column-major shortcuts"

    if len(frames) > 1:
        assert len({frame["side"] for frame in frames}) >= 2, "multi-frame scenes should vary gap direction"


def main(grid: np.ndarray) -> np.ndarray:
    """Queue the interior dots by distance to the frame gap, then stream them straight out of the gap.

    Input structure:
    - The grid has a dominant background and one or more non-touching hollow rectangular frames of one shared color.
    - Each frame border is missing exactly one non-corner cell, which acts as the gap.
    - Inside each frame are colored scatter pixels on the empty interior cells.

    Rule:
    - For each frame, measure every interior scatter pixel's shortest 4-connected path length to the gap while staying inside the frame interior plus the gap cell.
    - Starting from the first cell outside the gap and continuing straight outward perpendicular to that border side, place the scatter colors in increasing distance order.
    - Erase the original interior scatter pixels to background, while keeping the frame border and the gap unchanged.

    Key insight:
    - The missing border cell does three jobs at once: it marks the exit point, fixes the outward direction, and defines the only correct ordering by geodesic distance rather than by reading order.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames = _extract_frames(g)
    return _render_from_records(g, bg, frames)


def _sample_specs() -> list[dict]:
    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    sides = random.sample(SIDES, frame_count)

    specs = []
    for side in sides:
        specs.append(
            {
                "side": side,
                "h": random.randint(*FRAME_SIZE_RANGE),
                "w": random.randint(*FRAME_SIZE_RANGE),
                "ray_len": random.randint(*RAY_LEN_RANGE),
            }
        )
    return specs


def _min_scene_side(specs: list[dict], axis: str) -> int:
    if axis == "rows":
        top_need = max((spec["ray_len"] + spec["h"] for spec in specs if spec["side"] == "top"), default=0)
        bottom_need = max((spec["ray_len"] + spec["h"] for spec in specs if spec["side"] == "bottom"), default=0)
        longest = max(spec["h"] for spec in specs)
    else:
        top_need = max((spec["ray_len"] + spec["w"] for spec in specs if spec["side"] == "left"), default=0)
        bottom_need = max((spec["ray_len"] + spec["w"] for spec in specs if spec["side"] == "right"), default=0)
        longest = max(spec["w"] for spec in specs)

    gap = 2 if top_need and bottom_need else 0
    return max(GRID_SIDE_MIN, top_need + bottom_need + gap, longest + 4)


def _record_from_spec(spec: dict, shape: tuple[int, int], origin: tuple[int, int], offset: int) -> dict:
    height, width = shape
    r0, c0 = origin
    h = int(spec["h"])
    w = int(spec["w"])
    bbox = (r0, c0, r0 + h - 1, c0 + w - 1)

    if spec["side"] == "top":
        gap = (r0, c0 + offset)
    elif spec["side"] == "bottom":
        gap = (r0 + h - 1, c0 + offset)
    elif spec["side"] == "left":
        gap = (r0 + offset, c0)
    else:
        gap = (r0 + offset, c0 + w - 1)

    side, out_step = _gap_side_and_step(bbox, gap)
    frame_cells = _border_cells(bbox) - {gap}
    ray = _ray_cells(shape, gap, out_step)
    assert len(ray) == int(spec["ray_len"])

    area = _bbox_area_cells(bbox)
    padded = _expand_cells(area, shape, radius=1) | _expand_cells(set(ray), shape, radius=1)
    return {
        "bbox": bbox,
        "gap": gap,
        "side": side,
        "out_step": out_step,
        "frame_cells": frame_cells,
        "ray": ray,
        "blocked": padded,
    }


def _place_records(specs: list[dict], shape: tuple[int, int]) -> list[dict] | None:
    height, width = shape
    blocked = np.zeros(shape, dtype=bool)
    placed = []

    order = sorted(specs, key=lambda spec: int(spec["h"]) * int(spec["w"]), reverse=True)
    for spec in order:
        h = int(spec["h"])
        w = int(spec["w"])
        side = str(spec["side"])
        ray_len = int(spec["ray_len"])

        if side == "top":
            starts = [(ray_len, c0) for c0 in range(width - w + 1)]
            offsets = list(range(1, w - 1))
        elif side == "bottom":
            starts = [(height - ray_len - h, c0) for c0 in range(width - w + 1)]
            offsets = list(range(1, w - 1))
        elif side == "left":
            starts = [(r0, ray_len) for r0 in range(height - h + 1)]
            offsets = list(range(1, h - 1))
        else:
            starts = [(r0, width - ray_len - w) for r0 in range(height - h + 1)]
            offsets = list(range(1, h - 1))

        random.shuffle(starts)
        found = None
        for origin in starts[:PLACEMENT_TRIES]:
            random.shuffle(offsets)
            for offset in offsets:
                record = _record_from_spec(spec, shape, origin, offset)
                if any(blocked[r, c] for r, c in record["blocked"]):
                    continue
                found = record
                break
            if found is not None:
                break

        if found is None:
            return None

        for r, c in found["blocked"]:
            blocked[r, c] = True
        placed.append(found)

    return placed


def _distance_buckets(record: dict, shape: tuple[int, int]) -> dict[int, list[tuple[int, int]]]:
    distances = _bfs_distances(shape, record["bbox"], record["gap"])
    r0, c0, r1, c1 = record["bbox"]
    buckets: dict[int, list[tuple[int, int]]] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            dist = int(distances[(r, c)])
            buckets.setdefault(dist, []).append((r, c))
    return buckets


def _choose_dots(
    record: dict,
    shape: tuple[int, int],
    palette: list[int],
    *,
    hard: bool,
) -> list[dict] | None:
    buckets = _distance_buckets(record, shape)
    distance_values = sorted(buckets)
    max_count = min(DOT_COUNT_RANGE[1], len(record["ray"]), len(distance_values), len(palette))
    min_count = DOT_COUNT_RANGE[0]
    if hard:
        min_count = max(min_count, 3)
    if max_count < min_count:
        return None

    for _ in range(DOT_TRIES):
        count = random.randint(min_count, max_count)
        chosen_distances = sorted(random.sample(distance_values, count))
        cells = [random.choice(buckets[dist]) for dist in chosen_distances]
        colors = random.sample(palette, count)
        dots = [
            {"pos": cells[i], "dist": int(chosen_distances[i]), "color": int(colors[i])}
            for i in range(count)
        ]
        if hard and not _nontrivial_distance_order({"dots": dots}):
            continue
        return dots

    return None


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    frame_color: int,
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    for record in records:
        for r, c in record["frame_cells"]:
            inp[r, c] = frame_color
        for dot in record["dots"]:
            inp[dot["pos"]] = int(dot["color"])
    out = _render_from_records(inp, bg, records)
    return inp, out


def _generate_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    min_h = _min_scene_side(specs, "rows")
    min_w = _min_scene_side(specs, "cols")
    if min_h > 30 or min_w > 30:
        return None

    height = random.randint(min_h, min(30, min_h + 6))
    width = random.randint(min_w, min(30, min_w + 6))
    records = _place_records(specs, (height, width))
    if records is None:
        return None

    bg, frame_color = random.sample(Color.ALL_COLORS, 2)
    palette = [color for color in Color.ALL_COLORS if color not in {bg, frame_color}]

    hard_choices = []
    for index, record in enumerate(records):
        dots = _choose_dots(record, (height, width), palette, hard=True)
        if dots is not None:
            hard_choices.append((index, dots))
    if not hard_choices:
        return None

    hard_index, hard_dots = random.choice(hard_choices)
    for index, record in enumerate(records):
        if index == hard_index:
            record["dots"] = hard_dots
        else:
            dots = _choose_dots(record, (height, width), palette, hard=False)
            if dots is None:
                return None
            record["dots"] = dots

    inp, expected = _render_scene((height, width), bg, frame_color, records)
    if not np.array_equal(main(inp), expected):
        return None

    try:
        _validate_generated_grid(inp)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-flow scene")
