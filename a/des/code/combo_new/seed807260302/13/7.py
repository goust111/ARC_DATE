from __future__ import annotations

import random
from collections import deque

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
VERTICAL_DIRS = ("up", "down")
HORIZONTAL_DIRS = ("left", "right")
BG_CHOICES = (0, 1, 7)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _candidate_shape(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: str,
    stretch: int,
) -> tuple[set[tuple[int, int]], tuple[int, int, int, int]] | None:
    if direction == "up":
        body_r0, body_c0, body_r1, body_c1 = r0 + stretch, c0, r1, c1
        if body_r0 > body_r1:
            return None
        width = body_c1 - body_c0 + 1
        if width % 2 == 0:
            return None
        tail_col = body_c0 + width // 2
        cells = {(r, c) for r in range(body_r0, body_r1 + 1) for c in range(body_c0, body_c1 + 1)}
        for r in range(r0, body_r0):
            cells.add((r, tail_col))
        return cells, (body_r0, body_c0, body_r1, body_c1)

    if direction == "down":
        body_r0, body_c0, body_r1, body_c1 = r0, c0, r1 - stretch, c1
        if body_r0 > body_r1:
            return None
        width = body_c1 - body_c0 + 1
        if width % 2 == 0:
            return None
        tail_col = body_c0 + width // 2
        cells = {(r, c) for r in range(body_r0, body_r1 + 1) for c in range(body_c0, body_c1 + 1)}
        for r in range(body_r1 + 1, r1 + 1):
            cells.add((r, tail_col))
        return cells, (body_r0, body_c0, body_r1, body_c1)

    if direction == "left":
        body_r0, body_c0, body_r1, body_c1 = r0, c0 + stretch, r1, c1
        if body_c0 > body_c1:
            return None
        height = body_r1 - body_r0 + 1
        if height % 2 == 0:
            return None
        tail_row = body_r0 + height // 2
        cells = {(r, c) for r in range(body_r0, body_r1 + 1) for c in range(body_c0, body_c1 + 1)}
        for c in range(c0, body_c0):
            cells.add((tail_row, c))
        return cells, (body_r0, body_c0, body_r1, body_c1)

    if direction == "right":
        body_r0, body_c0, body_r1, body_c1 = r0, c0, r1, c1 - stretch
        if body_c0 > body_c1:
            return None
        height = body_r1 - body_r0 + 1
        if height % 2 == 0:
            return None
        tail_row = body_r0 + height // 2
        cells = {(r, c) for r in range(body_r0, body_r1 + 1) for c in range(body_c0, body_c1 + 1)}
        for c in range(body_c1 + 1, c1 + 1):
            cells.add((tail_row, c))
        return cells, (body_r0, body_c0, body_r1, body_c1)

    raise ValueError(direction)


def _parse_object(color: int, coords: list[tuple[int, int]]) -> dict[str, object]:
    cells = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    candidates: list[dict[str, object]] = []

    for direction in DIRECTIONS:
        limit = (r1 - r0) if direction in VERTICAL_DIRS else (c1 - c0)
        for stretch in range(1, limit + 1):
            candidate = _candidate_shape(r0, c0, r1, c1, direction, stretch)
            if candidate is None:
                continue
            expected, body_bbox = candidate
            if expected != cells:
                continue
            br0, bc0, br1, bc1 = body_bbox
            body_h = br1 - br0 + 1
            body_w = bc1 - bc0 + 1
            if body_h < 2 or body_w < 2:
                continue
            candidates.append(
                {
                    "color": color,
                    "direction": direction,
                    "stretch": stretch,
                    "body_bbox": body_bbox,
                    "expanded_bbox": (r0, c0, r1, c1),
                    "body_area": body_h * body_w,
                    "expanded_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )

    assert len(candidates) == 1, "expected each component to be one rectangle with one tail"
    return candidates[0]


def _render_output(shape: tuple[int, int], bg: int, objects: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    order = sorted(
        objects,
        key=lambda obj: (
            int(obj["expanded_area"]),
            int(obj["body_area"]),
            int(obj["expanded_bbox"][0]),
            int(obj["expanded_bbox"][1]),
        ),
    )
    for obj in order:
        r0, c0, r1, c1 = obj["expanded_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each colored rectangle in the direction encoded by its one-pixel tail.

    Input structure:
    - The background is the dominant color.
    - Each non-background component is monochrome and consists of a solid axis-aligned rectangle plus one same-color one-pixel-wide tail.
    - Every tail extends straight outward from the midpoint of one rectangle side, with direction up, down, left, or right.

    Rule:
    - For each component, detect the rectangle body, the tail direction, and the tail length L.
    - Expand the whole rectangle by L cells in that direction, so the final object is the full bounding rectangle of the original body-plus-tail shape.
    - Remove the original thin-tail silhouette by repainting only the stretched rectangles on background.
    - If stretched rectangles overlap, the one with larger stretched area remains visible on top of the smaller one.

    Key insight:
    - The tail is not decoration; its axis gives the stretch direction and its length gives the exact amount, so the finished rectangle is just the component's outer bbox filled solid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _components(g, bg)
    assert comps, "expected at least one non-background object"

    objects = [_parse_object(int(comp["color"]), comp["coords"]) for comp in comps]
    return _render_output(g.shape, bg, objects)


def _build_spec(
    color: int,
    direction: str,
    stretch: int,
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> dict[str, object] | None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    candidate = _candidate_shape(r0, c0, r1, c1, direction, stretch)
    if candidate is None:
        return None
    input_cells, body_bbox = candidate
    br0, bc0, br1, bc1 = body_bbox
    body_h = br1 - br0 + 1
    body_w = bc1 - bc0 + 1
    if body_h < 2 or body_w < 2:
        return None
    return {
        "color": color,
        "direction": direction,
        "stretch": stretch,
        "body_bbox": body_bbox,
        "expanded_bbox": (r0, c0, r1, c1),
        "body_area": body_h * body_w,
        "expanded_area": height * width,
        "height": height,
        "width": width,
        "input_cells": input_cells,
    }


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _direction_pool(rng: random.Random, count: int) -> list[str]:
    base = list(DIRECTIONS)
    rng.shuffle(base)
    out = base[: min(count, len(base))]
    while len(out) < count:
        out.append(rng.choice(DIRECTIONS))
    rng.shuffle(out)
    return out


def _sample_nonoverlap_spec(
    rng: random.Random,
    h: int,
    w: int,
    color: int,
    direction: str,
    blocked_input: set[tuple[int, int]],
    existing_rects: list[tuple[int, int, int, int]],
) -> dict[str, object] | None:
    for _ in range(300):
        stretch = rng.randint(1, 4)
        if direction in VERTICAL_DIRS:
            width_choices = [v for v in (3, 5, 7) if v <= w - 2]
            if not width_choices:
                return None
            width = rng.choice(width_choices)
            body_h = rng.randint(2, 6)
            height = body_h + stretch
        else:
            height_choices = [v for v in (3, 5, 7) if v <= h - 2]
            if not height_choices:
                return None
            height = rng.choice(height_choices)
            body_w = rng.randint(2, 6)
            width = body_w + stretch

        if height > h - 2 or width > w - 2:
            continue

        r0 = rng.randint(1, h - height - 1)
        c0 = rng.randint(1, w - width - 1)
        spec = _build_spec(color, direction, stretch, r0, c0, height, width)
        if spec is None:
            continue
        rect = spec["expanded_bbox"]
        if any(_rect_overlap(rect, other, margin=1) for other in existing_rects):
            continue
        cells = spec["input_cells"]
        if any(cell in blocked_input for cell in cells):
            continue
        return spec
    return None


def _make_overlap_pair(
    rng: random.Random,
    h: int,
    w: int,
    color_a: int,
    color_b: int,
) -> tuple[dict[str, object], dict[str, object]] | None:
    direction_a = rng.choice(DIRECTIONS)
    if direction_a in VERTICAL_DIRS:
        stretch_a = 4
        width_a = 9
        body_h_a = rng.randint(3, 6)
        height_a = body_h_a + stretch_a
        if height_a > h - 2 or width_a > w - 2:
            return None
        r0_a = rng.randint(1, h - height_a - 1)
        c0_a = rng.randint(1, w - width_a - 1)
        spec_a = _build_spec(color_a, direction_a, stretch_a, r0_a, c0_a, height_a, width_a)
        if spec_a is None:
            return None

        direction_b = rng.choice(HORIZONTAL_DIRS)
        r0_b = r0_a if direction_a == "up" else r0_a + height_a - 3
        c0_b = c0_a if rng.random() < 0.5 else c0_a + width_a - 3
        spec_b = _build_spec(color_b, direction_b, 1, r0_b, c0_b, 3, 3)
    else:
        stretch_a = 4
        height_a = 9
        body_w_a = rng.randint(3, 6)
        width_a = body_w_a + stretch_a
        if height_a > h - 2 or width_a > w - 2:
            return None
        r0_a = rng.randint(1, h - height_a - 1)
        c0_a = rng.randint(1, w - width_a - 1)
        spec_a = _build_spec(color_a, direction_a, stretch_a, r0_a, c0_a, height_a, width_a)
        if spec_a is None:
            return None

        direction_b = rng.choice(VERTICAL_DIRS)
        r0_b = r0_a if rng.random() < 0.5 else r0_a + height_a - 3
        c0_b = c0_a if direction_a == "left" else c0_a + width_a - 3
        spec_b = _build_spec(color_b, direction_b, 1, r0_b, c0_b, 3, 3)

    if spec_b is None:
        return None
    if int(spec_a["expanded_area"]) <= int(spec_b["expanded_area"]):
        return None
    if spec_a["input_cells"] & _halo(spec_b["input_cells"], h, w):
        return None
    if spec_b["input_cells"] & _halo(spec_a["input_cells"], h, w):
        return None
    if not _rect_overlap(spec_a["expanded_bbox"], spec_b["expanded_bbox"]):
        return None
    return spec_a, spec_b


def _render_input(shape: tuple[int, int], bg: int, specs: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for spec in specs:
        for r, c in spec["input_cells"]:
            grid[r, c] = int(spec["color"])
    return grid


def _count_output_overlaps(specs: list[dict[str, object]]) -> int:
    count = 0
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            if _rect_overlap(specs[i]["expanded_bbox"], specs[j]["expanded_bbox"]):
                count += 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 30)
    w = rng.randint(15, 30)
    bg = rng.choice(BG_CHOICES)
    overlap_mode = bool(rng.randrange(2))
    num_objects = rng.randint(2, 5)
    palette = [c for c in range(10) if c != bg]
    colors = rng.sample(palette, num_objects)

    specs: list[dict[str, object]] = []
    blocked_input: set[tuple[int, int]] = set()
    blocked_rects: list[tuple[int, int, int, int]] = []

    if overlap_mode:
        pair = _make_overlap_pair(rng, h, w, colors[0], colors[1])
        if pair is None:
            return None
        spec_a, spec_b = pair
        specs.extend([spec_a, spec_b])
        blocked_input |= _halo(spec_a["input_cells"], h, w)
        blocked_input |= _halo(spec_b["input_cells"], h, w)
        blocked_rects.append(spec_a["expanded_bbox"])

        extra_dirs = _direction_pool(rng, num_objects - 2)
        for i in range(num_objects - 2):
            spec = _sample_nonoverlap_spec(
                rng,
                h,
                w,
                colors[i + 2],
                extra_dirs[i],
                blocked_input,
                blocked_rects,
            )
            if spec is None:
                return None
            specs.append(spec)
            blocked_input |= _halo(spec["input_cells"], h, w)
            blocked_rects.append(spec["expanded_bbox"])
    else:
        dirs = _direction_pool(rng, num_objects)
        for i in range(num_objects):
            spec = _sample_nonoverlap_spec(
                rng,
                h,
                w,
                colors[i],
                dirs[i],
                blocked_input,
                blocked_rects,
            )
            if spec is None:
                return None
            specs.append(spec)
            blocked_input |= _halo(spec["input_cells"], h, w)
            blocked_rects.append(spec["expanded_bbox"])

    overlap_count = _count_output_overlaps(specs)
    if overlap_mode and overlap_count == 0:
        return None
    if not overlap_mode and overlap_count != 0:
        return None

    grid = _render_input((h, w), bg, specs)
    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[int(np.argmax(counts))]) != bg:
        return None
    if int((counts == counts.max()).sum()) != 1:
        return None

    expected = _render_output((h, w), bg, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
