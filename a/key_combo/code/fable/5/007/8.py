"""ARC task 8: diagonal leak chain reaction."""

from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_GENERATION_ATTEMPTS = 500
_GENERATION_COUNT = 0


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (row, column) coordinate arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _outlets(frame: dict) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    r0, c0, r1, c1 = frame["bbox"]
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    return list(zip(corners, DIAGONALS))


def _parse_frames(grid: np.ndarray) -> tuple[list[dict], tuple[int, int], int]:
    """Identify exact hollow rectangular borders, their contents, and the spark."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert max(g.shape) <= 30

    candidates: list[dict] = []
    for color in (int(v) for v in np.unique(g) if int(v) != BACKGROUND):
        for coords in _components(g == color):
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            box = (int(r0), int(c0), int(r1), int(c1))
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            actual = {tuple(map(int, p)) for p in coords}
            wall = _border_cells(box)
            if actual != wall:
                continue

            inside = g[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
            pressure_colors = {int(v) for v in np.unique(inside) if int(v) != BACKGROUND}
            if len(pressure_colors) != 1:
                continue
            pressure_color = pressure_colors.pop()
            if pressure_color == color:
                continue
            pressure_local = np.argwhere(inside == pressure_color)
            if pressure_local.size == 0:
                continue
            pressure = [
                (int(r0) + 1 + int(r), int(c0) + 1 + int(c))
                for r, c in pressure_local
            ]
            if len(_components(inside == pressure_color)) != 1:
                continue
            candidates.append(
                {
                    "bbox": box,
                    "wall_color": int(color),
                    "wall": wall,
                    "pressure_color": int(pressure_color),
                    "pressure": pressure,
                }
            )

    candidates.sort(key=lambda f: f["bbox"])
    assert candidates, "expected at least one hollow frame"

    # Valid frames are separate rather than nested or overlapping.
    for i, a in enumerate(candidates):
        ar0, ac0, ar1, ac1 = a["bbox"]
        for b in candidates[i + 1 :]:
            br0, bc0, br1, bc1 = b["bbox"]
            overlap = not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)
            assert not overlap, "frames must have disjoint bounding boxes"

    claimed: set[tuple[int, int]] = set()
    for frame in candidates:
        r0, c0, r1, c1 = frame["bbox"]
        claimed.update((r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
    outside = [
        (int(r), int(c))
        for r, c in np.argwhere(g != BACKGROUND)
        if (int(r), int(c)) not in claimed
    ]
    assert len(outside) == 1, "expected exactly one isolated spark outside all frames"
    spark = outside[0]

    starters: list[int] = []
    for i, frame in enumerate(candidates):
        for corner, direction in _outlets(frame):
            if spark == (corner[0] + direction[0], corner[1] + direction[1]):
                starters.append(i)
    assert len(starters) == 1, "the spark must identify exactly one frame corner"
    return candidates, spark, starters[0]


def _transform(grid: np.ndarray, *, with_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    frames, spark, start = _parse_frames(g)
    out = g.copy()
    h, w = out.shape
    wall_owner = {cell: i for i, frame in enumerate(frames) for cell in frame["wall"]}

    queue: deque[int] = deque([start])
    scheduled = {start}
    erupted: list[int] = []
    triggered_by: dict[int, int] = {}
    painted_by: dict[tuple[int, int], int] = {}

    while queue:
        frame_id = queue.popleft()
        frame = frames[frame_id]
        erupted.append(frame_id)

        for r, c in frame["pressure"]:
            out[r, c] = BACKGROUND

        newly_hit: list[int] = []
        for corner, (dr, dc) in _outlets(frame):
            r, c = corner[0] + dr, corner[1] + dc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) != BACKGROUND:
                    hit = wall_owner.get((r, c))
                    if hit is not None and hit not in scheduled:
                        scheduled.add(hit)
                        triggered_by[hit] = frame_id
                        newly_hit.append(hit)
                    break
                out[r, c] = frame["pressure_color"]
                painted_by[(r, c)] = frame_id
                r += dr
                c += dc
        queue.extend(newly_hit)

    if with_trace:
        return out, {
            "frames": frames,
            "spark": spark,
            "start": start,
            "erupted": erupted,
            "triggered_by": triggered_by,
            "painted_by": painted_by,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate diagonally emitted pressure-color rays through a spark-started chain of hollow frames.

    Input structure:
    - A black background contains separated monochrome hollow rectangular frames.
    - Every frame encloses one 4-connected pressure cluster of a different color.
    - Exactly one isolated spark occupies the diagonal outside cell of one frame corner.

    Rule:
    - The spark-adjacent frame erupts first. From each corner's diagonally outside
      cell, cast a ray away from the frame in that frame's pressure color, painting
      background until the first non-background cell or the grid edge.
    - A ray whose first obstacle is another frame wall schedules that frame to erupt;
      each frame erupts at most once. Clear the pressure cluster of every erupted
      frame, while leaving the spark and all untriggered frames unchanged.

    Key insight:
    - The unique spark fixes the causal start, and wall hits—not mere diagonal
      alignment or proximity—determine the chain; every triggered frame supplies
      both the color and the four corner-relative ray origins for the next step.
    """
    return _transform(grid)


def _grow_pressure(
    box: tuple[int, int, int, int], rng: random.Random
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    safe = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
    if not safe:
        safe = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    seed = rng.choice(safe)
    cells = {seed}
    target = rng.randint(1, min(3, len(safe)))
    safe_set = set(safe)
    while len(cells) < target:
        options = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (r + dr, c + dc) in safe_set and (r + dr, c + dc) not in cells
        }
        if not options:
            break
        cells.add(rng.choice(sorted(options)))
    return sorted(cells)


def _draw_frame(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    wall_color: int,
    pressure_color: int,
    pressure: list[tuple[int, int]],
) -> None:
    for r, c in _border_cells(box):
        grid[r, c] = wall_color
    for r, c in pressure:
        grid[r, c] = pressure_color


def _boxes_clear(
    box: tuple[int, int, int, int], existing: list[tuple[int, int, int, int]]
) -> bool:
    """Require at least one background cell between every pair of frame bboxes."""
    r0, c0, r1, c1 = box
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + 1 < er0 or er1 + 1 < r0 or c1 + 1 < ec0 or ec1 + 1 < c0):
            return False
    return True


def _first_obstacle(
    grid: np.ndarray, corner: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, int] | None:
    r, c = corner[0] + direction[0], corner[1] + direction[1]
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        if int(grid[r, c]) != BACKGROUND:
            return r, c
        r += direction[0]
        c += direction[1]
    return None


def _semantic_check(
    grid: np.ndarray,
    active_boxes: list[tuple[int, int, int, int]],
    spectator_boxes: list[tuple[int, int, int, int]],
) -> bool:
    """Check intended roles and causal dependencies, not just solver closure."""
    try:
        out, trace = _transform(grid, with_trace=True)
    except (AssertionError, ValueError, IndexError):
        return False

    frames = trace["frames"]
    by_box = {frame["bbox"]: i for i, frame in enumerate(frames)}
    if set(by_box) != set(active_boxes + spectator_boxes):
        return False
    active_ids = [by_box[b] for b in active_boxes]
    spectator_ids = {by_box[b] for b in spectator_boxes}
    if trace["erupted"] != active_ids or trace["start"] != active_ids[0]:
        return False
    if spectator_ids & set(trace["erupted"]):
        return False
    if any(trace["triggered_by"].get(active_ids[i]) != active_ids[i - 1]
           for i in range(1, len(active_ids))):
        return False

    # The forward SE ray must encounter each immediate successor first. Thus the
    # third frame cannot be reached by a shortcut ray from the spark frame.
    for i in range(len(active_boxes) - 1):
        _r0, _c0, r1, c1 = active_boxes[i]
        hit = _first_obstacle(grid, (r1, c1), (1, 1))
        nr0, nc0, _nr1, _nc1 = active_boxes[i + 1]
        if hit != (nr0, nc0):
            return False

    painted_counts = {i: 0 for i in active_ids}
    for cell, owner in trace["painted_by"].items():
        if owner in painted_counts:
            painted_counts[owner] += 1
            if int(out[cell]) != int(frames[owner]["pressure_color"]):
                return False
    if any(count == 0 for count in painted_counts.values()):
        return False

    for i in active_ids:
        if any(int(out[r, c]) != BACKGROUND for r, c in frames[i]["pressure"]):
            return False
    for i in spectator_ids:
        if any(int(out[r, c]) != int(frames[i]["pressure_color"])
               for r, c in frames[i]["pressure"]):
            return False
    return not np.array_equal(out, grid)


def _attempt_generate(scenario: int) -> np.ndarray | None:
    rng = random.Random()
    active_count = 3 + (scenario % 2)
    spectator_count = 1 + ((scenario // 2) % 2)

    # Build a southeast causal chain. A later dihedral transform varies all four
    # directions without changing the corner-relative reference system.
    for _ in range(80):
        sizes = [(rng.randint(5, 6), rng.randint(5, 6)) for _ in range(active_count)]
        gaps = [rng.randint(2, 3) for _ in range(active_count - 1)]
        extent_h = sum(h for h, _w in sizes) + sum(g - 1 for g in gaps)
        extent_w = sum(w for _h, w in sizes) + sum(g - 1 for g in gaps)
        if extent_h <= 25 and extent_w <= 25:
            break
    else:
        return None

    min_h, min_w = max(24, extent_h + 4), max(24, extent_w + 4)
    if min_h > 30 or min_w > 30:
        return None
    h, w = rng.randint(min_h, 30), rng.randint(min_w, 30)
    top = rng.randint(2, h - extent_h - 2)
    left = rng.randint(2, w - extent_w - 2)

    active_boxes: list[tuple[int, int, int, int]] = []
    r0, c0 = top, left
    for i, (fh, fw) in enumerate(sizes):
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        active_boxes.append(box)
        if i < active_count - 1:
            gap = gaps[i]
            r0, c0 = box[2] + gap, box[3] + gap

    palette = rng.sample(list(range(1, 10)), 2 + active_count)
    wall_color, spark_color = palette[:2]
    active_colors = palette[2:]
    grid = np.zeros((h, w), dtype=int)
    for box, pressure_color in zip(active_boxes, active_colors):
        _draw_frame(grid, box, wall_color, pressure_color, _grow_pressure(box, rng))

    first = active_boxes[0]
    spark_choices = [
        (first[0] - 1, first[1] - 1),
        (first[0] - 1, first[3] + 1),
        (first[2] + 1, first[1] - 1),
    ]
    spark = rng.choice(spark_choices)
    grid[spark] = spark_color

    # Obtain the unobstructed active-ray footprint, then put spectator frames
    # beside (never on) it. This creates explicit near-miss counterexamples.
    try:
        _base_out, base_trace = _transform(grid, with_trace=True)
    except AssertionError:
        return None
    base_active_boxes = [base_trace["frames"][i]["bbox"] for i in base_trace["erupted"]]
    if base_active_boxes != active_boxes:
        return None
    ray_cells = set(base_trace["painted_by"])

    spectator_boxes: list[tuple[int, int, int, int]] = []
    existing = list(active_boxes)
    for spectator_i in range(spectator_count):
        candidates: list[tuple[int, int, int, int]] = []
        near: list[tuple[int, int, int, int]] = []
        dims = [(5, 5), (5, 6), (6, 5)]
        rng.shuffle(dims)
        for fh, fw in dims:
            for sr in range(1, h - fh):
                for sc in range(1, w - fw):
                    box = (sr, sc, sr + fh - 1, sc + fw - 1)
                    if not _boxes_clear(box, existing) or spark in _border_cells(box):
                        continue
                    footprint = {
                        (r, c)
                        for r in range(box[0], box[2] + 1)
                        for c in range(box[1], box[3] + 1)
                    }
                    if footprint & ray_cells:
                        continue
                    candidates.append(box)
                    wall = _border_cells(box)
                    if any(
                        max(abs(r - pr), abs(c - pc)) == 1
                        for r, c in wall
                        for pr, pc in ray_cells
                    ):
                        near.append(box)
        pool = near if spectator_i == 0 and near else candidates
        if not pool:
            return None
        box = rng.choice(pool)
        spectator_boxes.append(box)
        existing.append(box)
        pressure_color = rng.choice(active_colors)
        _draw_frame(grid, box, wall_color, pressure_color, _grow_pressure(box, rng))

    if not _semantic_check(grid, active_boxes, spectator_boxes):
        return None

    # Cycle the principal chain direction across consecutive generated examples;
    # add a random reflection for further layout and rectangular-shape variety.
    transformed = np.rot90(grid, k=scenario % 4)
    if rng.random() < 0.5:
        transformed = np.fliplr(transformed)
    transformed = transformed.copy()
    try:
        _out, trace = _transform(transformed, with_trace=True)
    except AssertionError:
        return None
    if len(trace["frames"]) != active_count + spectator_count:
        return None
    if len(trace["erupted"]) != active_count:
        return None
    return transformed


def generate_input() -> np.ndarray:
    """Generate varied, semantically checked chain reactions and near-miss spectators."""
    global _GENERATION_COUNT
    scenario = _GENERATION_COUNT
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _attempt_generate(scenario)
        if result is not None:
            _GENERATION_COUNT += 1
            return result
    raise RuntimeError("failed to generate a valid diagonal leak chain")
