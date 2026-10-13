from __future__ import annotations

import random

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "cell_set": set(cells),
                    "area": int(coords.shape[0]),
                }
            )

    return components


def _expected_u_cells(bbox: tuple[int, int, int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    rows = range(r0, r1 + 1)
    cols = range(c0, c1 + 1)

    if direction == "up":
        return {(r, c0) for r in rows} | {(r, c1) for r in rows} | {(r1, c) for c in cols}
    if direction == "down":
        return {(r, c0) for r in rows} | {(r, c1) for r in rows} | {(r0, c) for c in cols}
    if direction == "left":
        return {(r0, c) for c in cols} | {(r1, c) for c in cols} | {(r, c1) for r in rows}
    assert direction == "right"
    return {(r0, c) for c in cols} | {(r1, c) for c in cols} | {(r, c0) for r in rows}


def _detect_u_direction(component: dict) -> str | None:
    r0, c0, r1, c1 = component["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    matches = [
        direction
        for direction in DIRECTIONS
        if component["cell_set"] == _expected_u_cells(component["bbox"], direction)
    ]
    return matches[0] if len(matches) == 1 else None


def _frame_interior_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
    }
    return cells - frame["cell_set"]


def _detect_scene(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    by_color: dict[int, list[int]] = {}
    for idx, component in enumerate(components):
        by_color.setdefault(int(component["color"]), []).append(idx)

    frames: list[dict] = []
    wall_mask = np.zeros(g.shape, dtype=bool)
    for idx, component in enumerate(components):
        direction = _detect_u_direction(component)
        if direction is None:
            continue
        frame = {
            "id": idx,
            "color": int(component["color"]),
            "coords": component["coords"],
            "bbox": component["bbox"],
            "cell_set": component["cell_set"],
            "direction": direction,
        }
        frames.append(frame)
        wall_mask[component["coords"][:, 0], component["coords"][:, 1]] = True

    assert frames, "expected at least one U-shaped frame"

    used_bullets: set[int] = set()
    for frame in frames:
        interior = _frame_interior_cells(frame)
        candidates = []
        for idx in by_color[frame["color"]]:
            if idx == frame["id"]:
                continue
            component = components[idx]
            if component["cell_set"] <= interior:
                candidates.append(idx)
        assert len(candidates) == 1, "each U frame must contain exactly one same-color bullet"
        bullet_idx = candidates[0]
        assert bullet_idx not in used_bullets, "bullets must map to exactly one frame"
        used_bullets.add(bullet_idx)
        frame["bullet"] = components[bullet_idx]

    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return bg, frames, wall_mask


def _trace_ray(
    frame: dict,
    grid: np.ndarray,
    bg: int,
    wall_mask: np.ndarray,
) -> tuple[list[tuple[int, int]], str]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r0, c0, r1, c1 = frame["bbox"]
    br0, bc0, br1, bc1 = frame["bullet"]["bbox"]
    cells: list[tuple[int, int]] = []

    if frame["direction"] == "up":
        cols = slice(c0 + 1, c1)
        row = br0 - 1
        while row >= 0:
            if np.any(wall_mask[row, cols]):
                return cells, "wall"
            bg_cols = np.where(g[row, cols] == bg)[0]
            cells.extend((row, c0 + 1 + int(offset)) for offset in bg_cols)
            row -= 1
        return cells, "edge"

    if frame["direction"] == "down":
        cols = slice(c0 + 1, c1)
        row = br1 + 1
        while row < h:
            if np.any(wall_mask[row, cols]):
                return cells, "wall"
            bg_cols = np.where(g[row, cols] == bg)[0]
            cells.extend((row, c0 + 1 + int(offset)) for offset in bg_cols)
            row += 1
        return cells, "edge"

    if frame["direction"] == "left":
        rows = slice(r0 + 1, r1)
        col = bc0 - 1
        while col >= 0:
            if np.any(wall_mask[rows, col]):
                return cells, "wall"
            bg_rows = np.where(g[rows, col] == bg)[0]
            cells.extend((r0 + 1 + int(offset), col) for offset in bg_rows)
            col -= 1
        return cells, "edge"

    assert frame["direction"] == "right"
    rows = slice(r0 + 1, r1)
    col = bc1 + 1
    while col < w:
        if np.any(wall_mask[rows, col]):
            return cells, "wall"
        bg_rows = np.where(g[rows, col] == bg)[0]
        cells.extend((r0 + 1 + int(offset), col) for offset in bg_rows)
        col += 1
    return cells, "edge"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each U-shaped frame, use its opening direction as a ray direction,
    and emit a same-color ray whose width matches the frame's interior width.
    The ray starts from the bullet side nearest the opening and stops before the
    first U wall it would hit, or at the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, wall_mask = _detect_scene(g)
    planned: dict[tuple[int, int], int] = {}

    for frame in frames:
        ray_cells, _stop = _trace_ray(frame, g, bg, wall_mask)
        for cell in ray_cells:
            planned.setdefault(cell, frame["color"])

    out = g.copy()
    for (r, c), color in planned.items():
        if int(out[r, c]) == bg:
            out[r, c] = int(color)
    return out


def _sample_size(direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        return random.randint(4, 7), random.randint(5, 9)
    return random.randint(5, 9), random.randint(4, 7)


def _bullet_room(direction: str, h: int, w: int) -> tuple[int, int, int, int]:
    if direction == "up":
        return 1, h - 3, 2, w - 3
    if direction == "down":
        return 2, h - 2, 2, w - 3
    if direction == "left":
        return 2, h - 3, 1, w - 3
    assert direction == "right"
    return 2, h - 3, 2, w - 2


def _sample_bullet_bbox(
    bbox: tuple[int, int, int, int],
    direction: str,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    room_r0, room_r1, room_c0, room_c1 = _bullet_room(direction, h, w)
    room_h = room_r1 - room_r0 + 1
    room_w = room_c1 - room_c0 + 1
    assert room_h >= 1 and room_w >= 1

    bullet_h = random.randint(1, min(2, room_h))
    bullet_w = random.randint(1, min(2, room_w))
    top = random.randint(room_r0, room_r1 - bullet_h + 1)
    left = random.randint(room_c0, room_c1 - bullet_w + 1)
    return (
        r0 + top,
        c0 + left,
        r0 + top + bullet_h - 1,
        c0 + left + bullet_w - 1,
    )


def _bbox_separated(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 1,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _draw_spec(grid: np.ndarray, spec: dict) -> None:
    color = int(spec["color"])
    for r, c in _expected_u_cells(spec["bbox"], spec["direction"]):
        grid[r, c] = color
    br0, bc0, br1, bc1 = spec["bullet_bbox"]
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = color


def _random_bbox(
    H: int,
    W: int,
    h: int,
    w: int,
    direction: str,
) -> tuple[int, int, int, int] | None:
    r_lo = 0
    r_hi = H - h
    c_lo = 0
    c_hi = W - w
    if direction == "up":
        r_lo = 1
    elif direction == "down":
        r_hi -= 1
    elif direction == "left":
        c_lo = 1
    else:
        c_hi -= 1

    if r_lo > r_hi or c_lo > c_hi:
        return None

    r0 = random.randint(r_lo, r_hi)
    c0 = random.randint(c_lo, c_hi)
    return r0, c0, r0 + h - 1, c0 + w - 1


def _make_spec(
    bbox: tuple[int, int, int, int],
    direction: str,
    color: int,
) -> dict:
    return {
        "bbox": bbox,
        "direction": direction,
        "color": int(color),
        "bullet_bbox": _sample_bullet_bbox(bbox, direction),
    }


def _place_free_object(
    H: int,
    W: int,
    direction: str,
    color: int,
    placed: list[dict],
) -> dict | None:
    for _ in range(80):
        h, w = _sample_size(direction)
        bbox = _random_bbox(H, W, h, w, direction)
        if bbox is None:
            continue
        if all(_bbox_separated(bbox, other["bbox"]) for other in placed):
            return _make_spec(bbox, direction, color)
    return None


def _place_blocked_pair(
    H: int,
    W: int,
    direction: str,
    color_a: int,
    color_b: int,
) -> list[dict] | None:
    for _ in range(80):
        if direction in ("up", "down"):
            width = random.randint(5, min(9, W))
            h_a = random.randint(4, 7)
            h_b = random.randint(4, 7)
            gap = random.randint(2, 5)
            c_bbox = _random_bbox(H, W, 1, width, "left")
            if c_bbox is None:
                continue
            c0 = c_bbox[1]
            if direction == "up":
                total = h_a + h_b + gap
                if total >= H:
                    continue
                blocker_r0 = random.randint(1, H - total)
                source_r0 = blocker_r0 + h_b + gap
                bbox_b = (blocker_r0, c0, blocker_r0 + h_b - 1, c0 + width - 1)
                bbox_a = (source_r0, c0, source_r0 + h_a - 1, c0 + width - 1)
            else:
                total = h_a + h_b + gap + 1
                if total > H:
                    continue
                source_r0 = random.randint(0, H - total)
                blocker_r0 = source_r0 + h_a + gap
                bbox_a = (source_r0, c0, source_r0 + h_a - 1, c0 + width - 1)
                bbox_b = (blocker_r0, c0, blocker_r0 + h_b - 1, c0 + width - 1)
        else:
            height = random.randint(5, min(9, H))
            w_a = random.randint(4, 7)
            w_b = random.randint(4, 7)
            gap = random.randint(2, 5)
            r_bbox = _random_bbox(H, W, height, 1, "up")
            if r_bbox is None:
                continue
            r0 = r_bbox[0]
            if direction == "left":
                total = w_a + w_b + gap
                if total >= W:
                    continue
                blocker_c0 = random.randint(1, W - total)
                source_c0 = blocker_c0 + w_b + gap
                bbox_b = (r0, blocker_c0, r0 + height - 1, blocker_c0 + w_b - 1)
                bbox_a = (r0, source_c0, r0 + height - 1, source_c0 + w_a - 1)
            else:
                total = w_a + w_b + gap + 1
                if total > W:
                    continue
                source_c0 = random.randint(0, W - total)
                blocker_c0 = source_c0 + w_a + gap
                bbox_a = (r0, source_c0, r0 + height - 1, source_c0 + w_a - 1)
                bbox_b = (r0, blocker_c0, r0 + height - 1, blocker_c0 + w_b - 1)

        if _bbox_separated(bbox_a, bbox_b):
            return [
                _make_spec(bbox_a, direction, color_a),
                _make_spec(bbox_b, direction, color_b),
            ]
    return None


def _scene_is_valid(grid: np.ndarray, mode: str, expected_count: int) -> bool:
    g = np.asarray(grid, dtype=int)
    if np.mean(g == 0) < 0.55:
        return False

    try:
        bg, frames, wall_mask = _detect_scene(g)
    except Exception:
        return False

    if len(frames) != expected_count:
        return False

    seen: set[tuple[int, int]] = set()
    stop_reasons = []
    for frame in frames:
        cells, stop = _trace_ray(frame, g, bg, wall_mask)
        if not cells:
            return False
        for cell in cells:
            if cell in seen:
                return False
            seen.add(cell)
        stop_reasons.append(stop)

    if mode == "edge" and any(stop != "edge" for stop in stop_reasons):
        return False
    if mode == "blocked" and not ("wall" in stop_reasons and "edge" in stop_reasons):
        return False

    try:
        out = main(g)
    except Exception:
        return False
    return not np.array_equal(out, g)


def _render_specs(H: int, W: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((H, W), dtype=int)
    for spec in specs:
        _draw_spec(grid, spec)
    return grid


def _gen_attempt(mode: str) -> np.ndarray | None:
    H = random.randint(16, 24)
    W = random.randint(16, 24)
    n_objects = random.randint(2, 4)
    colors = random.sample(PALETTE, n_objects)

    specs: list[dict] = []
    if mode == "blocked":
        pair_direction = random.choice(DIRECTIONS)
        pair = _place_blocked_pair(H, W, pair_direction, colors[0], colors[1])
        if pair is None:
            return None
        specs.extend(pair)
        remaining_directions = [d for d in DIRECTIONS if d != pair_direction]
        for color in colors[2:]:
            direction = random.choice(remaining_directions or list(DIRECTIONS))
            spec = _place_free_object(H, W, direction, color, specs)
            if spec is None:
                return None
            specs.append(spec)
    else:
        directions = random.sample(list(DIRECTIONS), k=n_objects)
        for color, direction in zip(colors, directions):
            spec = _place_free_object(H, W, direction, color, specs)
            if spec is None:
                return None
            specs.append(spec)

    grid = _render_specs(H, W, specs)
    return grid if _scene_is_valid(grid, mode, n_objects) else None


def generate_input() -> np.ndarray:
    mode = random.choice(("edge", "blocked"))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
