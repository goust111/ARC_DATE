from __future__ import annotations

import random

import numpy as np


SIDE_TO_OUTWARD = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SIDE_TO_INWARD = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

CARDINAL_DIRECTIONS = ("right", "left", "down", "up")
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = np.asarray(grid == int(color), dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _opening_side(opening: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = opening
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _deepest_cell(bbox: tuple[int, int, int, int], side: str, opening: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    r, c = opening
    if side == "top":
        return (r1 - 1, c)
    if side == "bottom":
        return (r0 + 1, c)
    if side == "left":
        return (r, c1 - 1)
    return (r, c0 + 1)


def _cup_from_component(
    grid: np.ndarray, color: int, coords: np.ndarray, bg: int
) -> dict[str, object] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    bbox = (int(r0), int(c0), int(r1), int(c1))
    perimeter = _perimeter_cells(*bbox)
    comp_set = {tuple(map(int, coord)) for coord in coords.tolist()}
    if len(comp_set) != len(perimeter) - 1:
        return None
    if not comp_set.issubset(perimeter):
        return None

    missing = list(perimeter - comp_set)
    if len(missing) != 1:
        return None
    opening = missing[0]
    side = _opening_side(opening, bbox)
    if side is None:
        return None

    return {
        "color": int(color),
        "bbox": bbox,
        "opening": tuple(map(int, opening)),
        "side": side,
        "outward": SIDE_TO_OUTWARD[side],
        "inward": SIDE_TO_INWARD[side],
        "deepest": _deepest_cell(bbox, side, opening),
        "ball": None,
    }


def _ball_is_isolated(grid: np.ndarray, r: int, c: int, bg: int) -> bool:
    height, width = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width and int(grid[nr, nc]) != bg:
            return False
    return True


def _parse_cups(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(v) for v in np.unique(g) if int(v) != bg)

    cups: list[dict[str, object]] = []
    wall_mask = np.zeros_like(g, dtype=bool)
    for color in colors:
        for coords in _components_of_color(g, color):
            cup = _cup_from_component(g, color, coords, bg)
            if cup is None:
                continue
            cups.append(cup)
            wall_coords = np.asarray(coords, dtype=int)
            wall_mask[wall_coords[:, 0], wall_coords[:, 1]] = True

    assert cups, "expected at least one cup outline"

    ball_cells: set[tuple[int, int]] = set()
    for cup in cups:
        r0, c0, r1, c1 = cup["bbox"]
        interior = [
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
            if int(g[r, c]) != bg
        ]
        assert len(interior) <= 1, "each cup interior may contain at most one ball"
        if not interior:
            continue

        br, bc = interior[0]
        side = str(cup["side"])
        opening_r, opening_c = cup["opening"]
        if side in {"left", "right"}:
            assert br == opening_r, "source ball must align with the opening row"
        else:
            assert bc == opening_c, "source ball must align with the opening column"
        assert _ball_is_isolated(g, br, bc, bg), "source balls must be isolated pixels"

        cup["ball"] = (int(br), int(bc), int(g[br, bc]))
        ball_cells.add((int(br), int(bc)))

    all_fg = {(int(r), int(c)) for r, c in np.argwhere(g != bg)}
    wall_cells = {(int(r), int(c)) for r, c in np.argwhere(wall_mask)}
    assert all_fg == wall_cells | ball_cells, "foreground must consist only of cups and balls"
    return cups, wall_mask, bg


def _trace_ball(
    start: tuple[int, int],
    color: int,
    direction: tuple[int, int],
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    opening_map: dict[tuple[int, int], dict[str, object]],
) -> tuple[int, int]:
    height, width = shape
    dr, dc = direction
    current = start
    while True:
        nr = current[0] + dr
        nc = current[1] + dc
        if not (0 <= nr < height and 0 <= nc < width):
            return current

        gate = opening_map.get((nr, nc))
        if gate is not None and gate["ball"] is None and tuple(gate["inward"]) == (dr, dc):
            if int(gate["color"]) == int(color):
                return tuple(map(int, gate["deepest"]))
            return current

        if wall_mask[nr, nc]:
            return current
        current = (nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each cup-held ball through its opening and let matching empty cups capture it while mismatching cups reject it.

    Input structure:
    - A black background contains several monochrome rectangular cup outlines, each missing exactly one non-corner wall cell as its opening.
    - Some cups contain one isolated colored ball aligned with that cup opening; other cups are empty.
    - Ball paths are straight and non-intersecting, and every empty cup receives at most one incoming path.

    Rule:
    - Each ball leaves its source cup along the opening direction.
    - If the ray reaches an empty cup exactly through that cup opening, a same-colored cup captures the ball at the deepest interior cell on that line, while a different-colored cup rejects it one cell outside the opening.
    - Otherwise the ball stops one cell before the first cup wall it would hit, or at the grid edge if nothing blocks the ray; all cup outlines stay unchanged.

    Key insight:
    - The missing wall cell is both launcher and gate: source cups use it as an exit, target cups use it as a color-sensitive entrance, and every final resting place follows from that single straight ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    cups, wall_mask, bg = _parse_cups(g)
    opening_map = {tuple(cup["opening"]): cup for cup in cups}

    out = g.copy()
    launches: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
    for cup in cups:
        ball = cup["ball"]
        if ball is None:
            continue
        br, bc, color = ball
        launches.append(((int(br), int(bc)), int(color), tuple(cup["outward"])))
        out[br, bc] = bg

    assert launches, "expected at least one source ball"

    for start, color, direction in launches:
        stop = _trace_ball(start, color, direction, out.shape, wall_mask, opening_map)
        assert out[stop] == bg, "ball landing cells must be empty in the output"
        out[stop] = color
    return out


def _draw_cup(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    color: int,
    side: str,
    offset: int,
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if side == "top":
        grid[top, left + offset] = 0
    elif side == "bottom":
        grid[bottom, left + offset] = 0
    elif side == "left":
        grid[top + offset, left] = 0
    else:
        grid[top + offset, right] = 0


def _sample_cup_size() -> tuple[int, int]:
    return random.randint(5, 7), random.randint(5, 7)


def _sample_open_offset(span: int) -> int:
    return random.randint(2, span - 3)


def _sample_ball_axis_index(span: int) -> int:
    return random.randint(2, span - 3)


def _build_base_gate_patch(match: bool) -> np.ndarray:
    src_h, src_w = _sample_cup_size()
    dst_h, dst_w = _sample_cup_size()
    src_row = _sample_open_offset(src_h)
    dst_row = _sample_open_offset(dst_h)
    gap = random.randint(3, 6)

    lane = max(src_row, dst_row) + 1
    src_top = lane - src_row
    dst_top = lane - dst_row
    src_left = 1
    dst_left = src_left + src_w + gap

    patch_h = max(src_top + src_h, dst_top + dst_h) + 1
    patch_w = dst_left + dst_w + 1
    patch = np.zeros((patch_h, patch_w), dtype=int)

    src_color, ball_color, spare = random.sample(PALETTE, 3)
    dst_color = ball_color if match else spare

    _draw_cup(patch, src_top, src_left, src_h, src_w, src_color, "right", src_row)
    _draw_cup(patch, dst_top, dst_left, dst_h, dst_w, dst_color, "left", dst_row)
    patch[lane, src_left + _sample_ball_axis_index(src_w)] = ball_color
    return patch


def _build_base_wall_stop_patch() -> np.ndarray:
    src_h, src_w = _sample_cup_size()
    wall_h, wall_w = _sample_cup_size()
    src_row = _sample_open_offset(src_h)
    hit_row = _sample_open_offset(wall_h)
    gap = random.randint(3, 6)

    lane = max(src_row, hit_row) + 1
    src_top = lane - src_row
    wall_top = lane - hit_row
    src_left = 1
    wall_left = src_left + src_w + gap

    patch_h = max(src_top + src_h, wall_top + wall_h) + 1
    patch_w = wall_left + wall_w + 1
    patch = np.zeros((patch_h, patch_w), dtype=int)

    src_color, ball_color, wall_color = random.sample(PALETTE, 3)
    wall_side = random.choice(("top", "bottom", "right"))
    wall_offset = _sample_open_offset(wall_w if wall_side in {"top", "bottom"} else wall_h)

    _draw_cup(patch, src_top, src_left, src_h, src_w, src_color, "right", src_row)
    _draw_cup(patch, wall_top, wall_left, wall_h, wall_w, wall_color, wall_side, wall_offset)
    patch[lane, src_left + _sample_ball_axis_index(src_w)] = ball_color
    return patch


def _build_base_edge_stop_patch() -> np.ndarray:
    src_h, src_w = _sample_cup_size()
    src_row = _sample_open_offset(src_h)
    travel = random.randint(4, 8)

    src_top = 1
    src_left = 1
    lane = src_top + src_row
    patch_h = src_top + src_h + 1
    patch_w = src_left + src_w + travel
    patch = np.zeros((patch_h, patch_w), dtype=int)

    src_color, ball_color = random.sample(PALETTE, 2)
    _draw_cup(patch, src_top, src_left, src_h, src_w, src_color, "right", src_row)
    patch[lane, src_left + _sample_ball_axis_index(src_w)] = ball_color
    return patch


def _build_decoy_patch() -> np.ndarray:
    cup_h, cup_w = _sample_cup_size()
    side = random.choice(("top", "bottom", "left", "right"))
    offset = _sample_open_offset(cup_w if side in {"top", "bottom"} else cup_h)
    patch = np.zeros((cup_h + 2, cup_w + 2), dtype=int)
    _draw_cup(patch, 1, 1, cup_h, cup_w, random.choice(PALETTE), side, offset)
    return patch


def _transform_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return patch
    if direction == "left":
        return np.fliplr(patch)
    if direction == "down":
        return patch.T.copy()
    return np.flipud(patch.T).copy()


def _can_place_bbox(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    touch_side: str | None,
) -> bool:
    del touch_side
    return not bool(occupied[top : top + height, left : left + width].any())


def _mark_bbox(
    occupied: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    *,
    touch_side: str | None,
) -> None:
    del touch_side
    occupied[top : top + height, left : left + width] = True


def _paste_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    view = dst[top : top + h, left : left + w]
    mask = patch != 0
    assert not np.any(view[mask] != 0), "patches must not overlap"
    view[mask] = patch[mask]


def _candidate_positions(
    canvas_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    touch_side: str | None,
) -> list[tuple[int, int]]:
    height, width = canvas_shape
    ph, pw = patch_shape
    if ph > height or pw > width:
        return []

    positions: list[tuple[int, int]] = []
    if touch_side is None:
        for r in range(height - ph + 1):
            for c in range(width - pw + 1):
                positions.append((r, c))
    elif touch_side == "left":
        for r in range(height - ph + 1):
            positions.append((r, 0))
    elif touch_side == "right":
        c = width - pw
        for r in range(height - ph + 1):
            positions.append((r, c))
    elif touch_side == "top":
        for c in range(width - pw + 1):
            positions.append((0, c))
    else:
        r = height - ph
        for c in range(width - pw + 1):
            positions.append((r, c))
    random.shuffle(positions)
    return positions


def _gen_attempt() -> np.ndarray | None:
    stop_kind = random.choice(("stop_wall", "stop_edge"))
    scenarios = [
        {"kind": "same", "touch_side": None},
        {"kind": "diff", "touch_side": None},
        {"kind": stop_kind, "touch_side": None},
    ]
    if random.random() < 0.5:
        scenarios.append({"kind": "decoy", "touch_side": None})

    patches: list[dict[str, object]] = []
    for spec in scenarios:
        kind = str(spec["kind"])
        if kind == "same":
            patch = _build_base_gate_patch(match=True)
            direction = random.choice(CARDINAL_DIRECTIONS)
            touch_side = None
        elif kind == "diff":
            patch = _build_base_gate_patch(match=False)
            direction = random.choice(CARDINAL_DIRECTIONS)
            touch_side = None
        elif kind == "stop_wall":
            patch = _build_base_wall_stop_patch()
            direction = random.choice(CARDINAL_DIRECTIONS)
            touch_side = None
        elif kind == "stop_edge":
            direction = random.choice(CARDINAL_DIRECTIONS)
            patch = _build_base_edge_stop_patch()
            touch_side = {"right": "right", "left": "left", "down": "bottom", "up": "top"}[direction]
        else:
            patch = _build_decoy_patch()
            direction = "right"
            touch_side = None

        transformed = _transform_patch(patch, direction)
        patches.append({"patch": transformed, "touch_side": touch_side})

    total_area = sum(int(p["patch"].shape[0] * p["patch"].shape[1]) for p in patches)
    max_patch_h = max(int(p["patch"].shape[0]) for p in patches)
    max_patch_w = max(int(p["patch"].shape[1]) for p in patches)
    min_h = max(20, max_patch_h)
    min_w = max(20, max_patch_w)
    if min_h > 30 or min_w > 30 or total_area > 760:
        return None

    canvas_h = random.randint(min_h, 30)
    canvas_w = random.randint(min_w, 30)

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)

    patches.sort(
        key=lambda item: (
            item["touch_side"] is None,
            -int(item["patch"].shape[0] * item["patch"].shape[1]),
        )
    )

    for item in patches:
        patch = item["patch"]
        touch_side = item["touch_side"]
        positions = _candidate_positions(canvas.shape, patch.shape, touch_side)
        placed = False
        for top, left in positions:
            if not _can_place_bbox(occupied, top, left, patch.shape[0], patch.shape[1], touch_side=touch_side):
                continue
            _paste_patch(canvas, patch, top, left)
            _mark_bbox(occupied, top, left, patch.shape[0], patch.shape[1], touch_side=touch_side)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(canvas)
    except AssertionError:
        return None
    if np.array_equal(canvas, out):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cup-gate puzzle input")
