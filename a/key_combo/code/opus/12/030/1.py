from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME_PALETTE = [1, 2, 3, 4, 8, 9]
BAR_COLOR = 5
CONTROL_COLOR = 6
STAMP_COLOR = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 16:
        return False
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expected = 2 * (h + w) - 4
    cells = set(coords)
    if len(cells) != expected:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in cells):
                return False
    return True


def _find_marker(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]]:
    grey = [tuple(map(int, rc)) for rc in np.argwhere(grid == BAR_COLOR)]
    magenta = [tuple(map(int, rc)) for rc in np.argwhere(grid == CONTROL_COLOR)]
    assert len(grey) == 3, "expected one three-pixel grey marker bar"
    assert len(magenta) == 1, "expected one magenta control pixel"

    rows = sorted({r for r, _ in grey})
    cols = sorted({c for _, c in grey})
    if len(rows) == 3 and len(cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        center = (rows[1], cols[0])
    elif len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
        center = (rows[0], cols[1])
    else:
        raise AssertionError("marker bar must be a straight 3-pixel segment")

    control = magenta[0]
    dr = center[0] - control[0]
    dc = center[1] - control[1]
    assert abs(dr) + abs(dc) == 1, "control must be 4-adjacent to the marker center"

    ray: list[tuple[int, int]] = []
    r = center[0] + dr
    c = center[1] + dc
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w:
        ray.append((r, c))
        r += dr
        c += dc
    return center, (dr, dc), ray


def _frame_line_cells(
    bbox: tuple[int, int, int, int],
    step: tuple[int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if step[0] == 0:
        mid_r = (r0 + r1) // 2
        cells = [(mid_r, c) for c in range(c0 + 1, c1)]
        return cells if step[1] > 0 else list(reversed(cells))
    mid_c = (c0 + c1) // 2
    cells = [(r, mid_c) for r in range(r0 + 1, r1)]
    return cells if step[0] > 0 else list(reversed(cells))


def _stamp_cells(
    bbox: tuple[int, int, int, int],
    step: tuple[int, int],
    count: int,
) -> list[tuple[int, int]]:
    line = _frame_line_cells(bbox, step)
    if not line or count <= 0:
        return []
    max_with_gap2 = (len(line) + 1) // 2
    if count <= max_with_gap2:
        stride = 2
        usable = count
    else:
        stride = 1
        usable = min(count, len(line))
    return [line[i * stride] for i in range(usable)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate only the ray-penetrated hollow frames, count their same-color sprites, and stamp orange tally dots inside.

    Input structure:
    - The background is black, with exactly one grey three-pixel marker bar and one adjacent magenta control pixel near an outer edge.
    - Several distinct-color hollow rectangular frames sit apart from one another, each with a black interior.
    - Small monochrome 4-connected sprites of matching frame colors are scattered outside the frames.

    Rule:
    - Fire a straight ray from the marker-bar center in the direction opposite the control pixel, but do not draw the ray itself.
    - A frame is activated only if the ray passes through its interior cells; touching only the border does not count.
    - For every activated frame, count the outside sprites of that frame's color, erase those sprites, and place that many orange dots on the frame's interior center row or center column, starting from the ray-entry side with spacing 2, compressing to spacing 1 only when needed.
    - Frames not penetrated by the ray, along with their same-color sprites, stay unchanged; the marker bar and control pixel also stay unchanged.

    Key insight:
    - The ray is purely a selector: it decides which color-frames absorb their matching external sprites and convert that count into an internal orange tally.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _center, step, ray = _find_marker(g)
    ray_set = set(ray)

    colors = [
        int(color)
        for color in np.unique(g)
        if int(color) not in (BG, BAR_COLOR, CONTROL_COLOR, STAMP_COLOR)
    ]

    frames: dict[int, dict[str, object]] = {}
    patterns: dict[int, list[list[tuple[int, int]]]] = {}
    for color in colors:
        comps = _components_of_color(g, color)
        for coords in comps:
            if _is_hollow_frame(coords):
                frames[color] = {"coords": coords, "bbox": _bbox(coords)}
            else:
                patterns.setdefault(color, []).append(coords)

    assert frames, "expected at least one hollow frame"

    for color, frame in frames.items():
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        interior_hit = any((r, c) in ray_set for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
        if not interior_hit:
            continue

        same_color_patterns = patterns.get(color, [])
        for coords in same_color_patterns:
            rr, cc = zip(*coords)
            out[list(rr), list(cc)] = BG

        for r, c in _stamp_cells(frame["bbox"], step, len(same_color_patterns)):  # type: ignore[arg-type]
            out[r, c] = STAMP_COLOR

    return out


def _rects_conflict(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    padding: int = 1,
) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _marker_spec(h: int, w: int) -> dict[str, object]:
    direction = random.choice(["E", "W", "S", "N"])
    if direction == "E":
        row = random.randint(4, h - 5)
        center = (row, 1)
        control = (row, 0)
        bar = [(row - 1, 1), (row, 1), (row + 1, 1)]
        step = (0, 1)
    elif direction == "W":
        row = random.randint(4, h - 5)
        center = (row, w - 2)
        control = (row, w - 1)
        bar = [(row - 1, w - 2), (row, w - 2), (row + 1, w - 2)]
        step = (0, -1)
    elif direction == "S":
        col = random.randint(4, w - 5)
        center = (1, col)
        control = (0, col)
        bar = [(1, col - 1), (1, col), (1, col + 1)]
        step = (1, 0)
    else:
        col = random.randint(4, w - 5)
        center = (h - 2, col)
        control = (h - 1, col)
        bar = [(h - 2, col - 1), (h - 2, col), (h - 2, col + 1)]
        step = (-1, 0)

    ray: list[tuple[int, int]] = []
    r = center[0] + step[0]
    c = center[1] + step[1]
    while 0 <= r < h and 0 <= c < w:
        ray.append((r, c))
        r += step[0]
        c += step[1]

    marker_cells = bar + [control]
    return {
        "center": center,
        "control": control,
        "bar": bar,
        "step": step,
        "ray": ray,
        "bbox": _bbox(marker_cells),
    }


def _frame_candidates(
    h: int,
    w: int,
    fh: int,
    fw: int,
    step: tuple[int, int],
    ray_index: int,
    relation: str,
) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for r0 in range(1, h - fh):
        for c0 in range(1, w - fw):
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            if step[0] == 0:
                if relation == "hit" and not (r0 + 1 <= ray_index <= r1 - 1):
                    continue
                if relation == "graze" and ray_index not in (r0, r1):
                    continue
                if relation == "above" and not (r1 < ray_index):
                    continue
                if relation == "below" and not (r0 > ray_index):
                    continue
            else:
                if relation == "hit" and not (c0 + 1 <= ray_index <= c1 - 1):
                    continue
                if relation == "graze" and ray_index not in (c0, c1):
                    continue
                if relation == "left" and not (c1 < ray_index):
                    continue
                if relation == "right" and not (c0 > ray_index):
                    continue
            boxes.append((r0, c0, r1, c1))
    random.shuffle(boxes)
    return boxes


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _random_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                candidate = (r + dr, c + dc)
                if candidate not in cells:
                    frontier.append(candidate)
        cells.add(random.choice(frontier))

    r0, c0, r1, c1 = _bbox(list(cells))
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = 1
    return patch


def _can_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    color: int,
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    patch_cells = {
        (top + r, left + c)
        for r in range(ph)
        for c in range(pw)
        if int(patch[r, c]) == 1
    }
    for r, c in patch_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in blocked or int(grid[r, c]) != BG:
            return False
    for r, c in patch_cells:
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in patch_cells and int(grid[nr, nc]) != BG:
                return False
    return True


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    color: int,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - ph + 1):
        for left in range(w - pw + 1):
            if _can_place_patch(grid, patch, top, left, color, blocked):
                candidates.append((top, left))
    if not candidates:
        return None
    top, left = random.choice(candidates)
    coords: list[tuple[int, int]] = []
    for r in range(ph):
        for c in range(pw):
            if int(patch[r, c]) == 1:
                rr = top + r
                cc = left + c
                grid[rr, cc] = color
                coords.append((rr, cc))
    return coords


def _expected_output(
    inp: np.ndarray,
    frame_meta: list[dict[str, object]],
    pattern_meta: dict[int, list[list[tuple[int, int]]]],
    step: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    for frame in frame_meta:
        if not frame["hit"]:
            continue
        color = int(frame["color"])
        for coords in pattern_meta[color]:
            rr, cc = zip(*coords)
            out[list(rr), list(cc)] = BG
        for r, c in _stamp_cells(frame["bbox"], step, len(pattern_meta[color])):  # type: ignore[arg-type]
            out[r, c] = STAMP_COLOR
    return out


def _validate_metadata(
    grid: np.ndarray,
    frame_meta: list[dict[str, object]],
    pattern_meta: dict[int, list[list[tuple[int, int]]]],
    marker: dict[str, object],
    want_dense: bool,
) -> None:
    ray_set = set(marker["ray"])  # type: ignore[arg-type]
    step = marker["step"]  # type: ignore[assignment]
    hits = [frame for frame in frame_meta if frame["hit"]]
    misses = [frame for frame in frame_meta if not frame["hit"]]
    assert hits and misses

    saw_dense = False
    for frame in frame_meta:
        box = frame["bbox"]  # type: ignore[assignment]
        r0, c0, r1, c1 = box
        interior_hit = any((r, c) in ray_set for r in range(r0 + 1, r1) for c in range(c0 + 1, c1))
        assert interior_hit == bool(frame["hit"])
        color = int(frame["color"])
        assert 1 <= len(pattern_meta[color]) <= 3
        if frame["hit"]:
            line_len = len(_frame_line_cells(box, step))
            if len(pattern_meta[color]) > (line_len + 1) // 2:
                saw_dense = True
        if frame["relation"] == "graze":
            if step[0] == 0:
                assert marker["center"][0] not in range(r0 + 1, r1)  # type: ignore[index]
            else:
                assert marker["center"][1] not in range(c0 + 1, c1)  # type: ignore[index]

    if want_dense:
        assert saw_dense, "wanted at least one spacing-1 stamping case"

    expected = _expected_output(grid, frame_meta, pattern_meta, step)
    actual = main(grid)
    assert not np.array_equal(grid, expected), "input and output should differ"
    assert np.array_equal(actual, expected), "solver must match generator semantics"


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    grid = np.zeros((h, w), dtype=int)

    marker = _marker_spec(h, w)
    for r, c in marker["bar"]:  # type: ignore[index]
        grid[r, c] = BAR_COLOR
    control_r, control_c = marker["control"]  # type: ignore[index]
    grid[control_r, control_c] = CONTROL_COLOR

    num_frames = random.randint(4, 6)
    num_hits = random.randint(1, min(3, num_frames - 1))
    num_misses = num_frames - num_hits
    colors = random.sample(FRAME_PALETTE, num_frames)

    horizontal_ray = marker["step"][0] == 0  # type: ignore[index]
    ray_index = marker["center"][0] if horizontal_ray else marker["center"][1]  # type: ignore[index]
    want_dense = random.random() < 0.5
    dense_hit_index = random.randrange(num_hits) if want_dense else -1

    miss_relations: list[str] = []
    if num_misses > 0 and random.random() < 0.55:
        miss_relations.append("graze")
    while len(miss_relations) < num_misses:
        if horizontal_ray:
            miss_relations.append(random.choice(["above", "below"]))
        else:
            miss_relations.append(random.choice(["left", "right"]))
    random.shuffle(miss_relations)

    specs: list[dict[str, object]] = []
    for idx in range(num_hits):
        if horizontal_ray:
            fh = random.choice([5, 7])
            fw = 5 if idx == dense_hit_index else random.choice([5, 7])
        else:
            fh = 5 if idx == dense_hit_index else random.choice([5, 7])
            fw = random.choice([5, 7])
        specs.append(
            {
                "color": colors[idx],
                "fh": fh,
                "fw": fw,
                "hit": True,
                "relation": "hit",
                "count": 3 if idx == dense_hit_index else random.randint(1, 3),
            }
        )
    for idx, relation in enumerate(miss_relations, start=num_hits):
        specs.append(
            {
                "color": colors[idx],
                "fh": random.choice([5, 7]),
                "fw": random.choice([5, 7]),
                "hit": False,
                "relation": relation,
                "count": random.randint(1, 3),
            }
        )

    existing_boxes = [marker["bbox"]]  # type: ignore[list-item]
    frame_meta: list[dict[str, object]] = []
    random.shuffle(specs)
    specs.sort(key=lambda spec: (0 if spec["hit"] else 1, -(int(spec["fh"]) * int(spec["fw"]))))

    for spec in specs:
        candidates = _frame_candidates(
            h,
            w,
            int(spec["fh"]),
            int(spec["fw"]),
            marker["step"],  # type: ignore[arg-type]
            int(ray_index),
            str(spec["relation"]),
        )
        placed = None
        for box in candidates:
            if any(_rects_conflict(box, prev, padding=1) for prev in existing_boxes):
                continue
            placed = box
            break
        if placed is None:
            return None
        _draw_frame(grid, placed, int(spec["color"]))
        existing_boxes.append(placed)
        frame_meta.append(
            {
                "color": int(spec["color"]),
                "bbox": placed,
                "hit": bool(spec["hit"]),
                "relation": str(spec["relation"]),
                "count": int(spec["count"]),
            }
        )

    blocked = set(marker["ray"])  # type: ignore[arg-type]
    blocked.add(marker["control"])  # type: ignore[arg-type]
    blocked.update(marker["bar"])  # type: ignore[arg-type]
    for frame in frame_meta:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                blocked.add((r, c))

    pattern_meta: dict[int, list[list[tuple[int, int]]]] = {}
    for frame in frame_meta:
        color = int(frame["color"])
        count = int(frame["count"])
        pattern_meta[color] = []
        for _ in range(count):
            placed_coords = None
            for _attempt in range(20):
                area = random.randint(3, 5)
                patch = _random_polyomino(area)
                placed_coords = _place_patch(grid, patch, color, blocked)
                if placed_coords is not None:
                    pattern_meta[color].append(placed_coords)
                    break
            if placed_coords is None:
                return None

    _validate_metadata(grid, frame_meta, pattern_meta, marker, want_dense)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-frame counting puzzle")
