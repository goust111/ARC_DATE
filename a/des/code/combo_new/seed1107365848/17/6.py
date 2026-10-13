import random
from collections import defaultdict, deque

import numpy as np


ARC_LIMIT = 30
GEN_CANVAS = 30
GEN_RETRIES = 400
GEN_PLACEMENT_TRIES = 200
GEN_PAIR_COUNTS = (2, 3, 4)
GEN_INTERIOR_MIN = 2
GEN_INTERIOR_MAX = 5
GEN_MIN_MARGIN = 1
GEN_MAX_EXTRA_MARGIN = 4
NEIGHBORS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ROTATION_K = {"up": 0, "right": 3, "down": 2, "left": 1}
CARDINALS = ("up", "right", "down", "left")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) != color:
                continue
            visited[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS_4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            components.append(coords)

    return components


def _frame_signature(
    coords: list[tuple[int, int]]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    perimeter = np.zeros((h, w), dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    if np.any(mask & ~perimeter):
        return None

    missing = np.argwhere(perimeter & ~mask)
    if len(missing) > 1:
        return None

    gap_side = None
    if len(missing) == 1:
        mr, mc = map(int, missing[0])
        if (mr in (0, h - 1)) and (mc in (0, w - 1)):
            return None
        if mr == 0:
            gap_side = "up"
        elif mr == h - 1:
            gap_side = "down"
        elif mc == 0:
            gap_side = "left"
        elif mc == w - 1:
            gap_side = "right"
        else:
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": gap_side,
        "shape": (h, w),
    }


def _crop_interior(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 + 1 : r1, c0 + 1 : c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each window's interior pattern into the same-colored empty room, rotating it to match the doorway gap side.

    Input structure:
    - The background is the dominant color.
    - Several colored rectangular border components appear in color-matched pairs.
      One rectangle in a pair is a complete frame whose interior contains a
      multicolor pattern (the window), and the other is a frame with exactly one
      missing border pixel and an all-background interior (the empty room).
    - The missing border pixel of each empty room lies on exactly one side of the
      rectangle and acts as a directional doorway marker.

    Rule:
    - Group frame components by their border color and pair each complete frame
      with the same-colored frame that has one doorway gap.
    - Read the gap side of the empty room: top means no rotation, right means a
      90-degree clockwise rotation, bottom means 180 degrees, and left means a
      90-degree counterclockwise rotation.
    - Copy the complete frame's interior patch into the empty room's interior
      after applying that rotation. Leave every border and every source window
      unchanged.

    Key insight:
    - Border color decides which patterned window supplies each room, while the
      location of the room's single border gap independently encodes the rotation
      that must be applied before stamping the pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"

    bg = _get_bg(g)
    by_color: dict[int, list[dict[str, object]]] = defaultdict(list)

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for coords in _components_of_color(g, color):
            sig = _frame_signature(coords)
            if sig is not None:
                sig["color"] = color
                by_color[color].append(sig)

    out = np.array(g, copy=True)
    paired = 0

    for color, frames in by_color.items():
        windows = [frame for frame in frames if frame["gap_side"] is None]
        rooms = [frame for frame in frames if frame["gap_side"] is not None]
        if len(windows) != 1 or len(rooms) != 1:
            continue

        window = windows[0]
        room = rooms[0]
        pattern = _crop_interior(g, window["bbox"])
        room_interior = _crop_interior(g, room["bbox"])
        assert np.any(pattern != bg), f"window color {color} must contain a visible interior pattern"
        assert np.all(room_interior == bg), f"room color {color} must start empty"

        rotated = np.rot90(pattern, ROTATION_K[str(room["gap_side"])])
        target = _crop_interior(out, room["bbox"])
        assert rotated.shape == target.shape, f"rotated pattern shape {rotated.shape} does not fit room {target.shape}"
        target[:, :] = rotated
        paired += 1

    assert paired > 0, "expected at least one color-matched window/room pair"
    return out


def _choose_directions(rng: random.Random, n_pairs: int) -> list[str]:
    if n_pairs == 2:
        directions = [rng.choice(("up", "down")), rng.choice(("left", "right"))]
        rng.shuffle(directions)
        return directions
    if n_pairs == 3:
        directions = rng.sample(CARDINALS, 3)
        rng.shuffle(directions)
        return directions
    directions = list(CARDINALS)
    rng.shuffle(directions)
    return directions


def _grown_mask(h: int, w: int, target: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True

    attempts = 0
    while int(mask.sum()) < target and attempts < h * w * 20:
        attempts += 1
        cells = list(zip(*np.where(mask)))
        r, c = cells[rng.randrange(len(cells))]
        dr, dc = NEIGHBORS_4[rng.randrange(len(NEIGHBORS_4))]
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            mask[nr, nc] = True

    return mask


def _random_pattern(
    h: int,
    w: int,
    bg: int,
    palette: list[int],
    rotation_k: int,
    rng: random.Random,
) -> np.ndarray | None:
    min_cells = max(3, (h * w + 3) // 4)
    max_cells = max(min_cells, h * w - 1)

    for _ in range(120):
        target = rng.randint(min_cells, max_cells)
        support = _grown_mask(h, w, target, rng)
        rows, cols = np.where(support)
        if len(rows) < 3 or np.all(support):
            continue
        if h > 1 and len(set(map(int, rows))) == 1:
            continue
        if w > 1 and len(set(map(int, cols))) == 1:
            continue

        pattern = np.full((h, w), bg, dtype=int)
        n_colors = rng.randint(1, min(3, len(palette)))
        colors = rng.sample(palette, n_colors)
        support_cells = [(int(r), int(c)) for r, c in zip(rows, cols)]
        rng.shuffle(support_cells)

        for color, (r, c) in zip(colors, support_cells):
            pattern[r, c] = color
        for r, c in support_cells[n_colors:]:
            pattern[r, c] = rng.choice(colors)

        if rotation_k != 0 and np.array_equal(pattern, np.rot90(pattern, rotation_k)):
            continue
        return pattern

    return None


def _draw_window(
    grid: np.ndarray,
    top: int,
    left: int,
    border_color: int,
    pattern: np.ndarray,
) -> tuple[int, int, int, int]:
    ih, iw = pattern.shape
    bbox = (top, left, top + ih + 1, left + iw + 1)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = pattern
    return bbox


def _draw_room(
    grid: np.ndarray,
    top: int,
    left: int,
    border_color: int,
    interior_shape: tuple[int, int],
    gap_side: str,
    bg: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    ih, iw = interior_shape
    bbox = (top, left, top + ih + 1, left + iw + 1)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = bg

    if gap_side in ("up", "down"):
        gap_c = c0 + rng.randint(1, iw)
        gap_r = r0 if gap_side == "up" else r1
    else:
        gap_r = r0 + rng.randint(1, ih)
        gap_c = c0 if gap_side == "left" else c1
    grid[gap_r, gap_c] = bg
    return bbox


def _place_rectangles(rectangles: list[dict[str, int]], rng: random.Random) -> dict[int, tuple[int, int]] | None:
    occupied = np.zeros((GEN_CANVAS, GEN_CANVAS), dtype=bool)
    placements: dict[int, tuple[int, int]] = {}
    order = list(range(len(rectangles)))
    rng.shuffle(order)

    for idx in order:
        rect = rectangles[idx]
        height = rect["h"]
        width = rect["w"]
        placed = False
        for _ in range(GEN_PLACEMENT_TRIES):
            top = rng.randint(GEN_MIN_MARGIN, GEN_CANVAS - height - GEN_MIN_MARGIN)
            left = rng.randint(GEN_MIN_MARGIN, GEN_CANVAS - width - GEN_MIN_MARGIN)
            er0 = top - 1
            ec0 = left - 1
            er1 = top + height
            ec1 = left + width
            if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None

    return placements


def _crop_canvas(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    rng: random.Random,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    coords = np.argwhere(grid != int(specs[0]["bg"]))
    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    top = rng.randint(max(0, int(r0) - GEN_MAX_EXTRA_MARGIN), int(r0) - 1)
    left = rng.randint(max(0, int(c0) - GEN_MAX_EXTRA_MARGIN), int(c0) - 1)
    bottom = rng.randint(int(r1) + 1, min(GEN_CANVAS - 1, int(r1) + GEN_MAX_EXTRA_MARGIN))
    right = rng.randint(int(c1) + 1, min(GEN_CANVAS - 1, int(c1) + GEN_MAX_EXTRA_MARGIN))

    cropped = np.array(grid[top : bottom + 1, left : right + 1], dtype=int, copy=True)
    adjusted: list[dict[str, object]] = []
    for spec in specs:
        wr0, wc0, wr1, wc1 = spec["window_bbox"]
        rr0, rc0, rr1, rc1 = spec["room_bbox"]
        adjusted.append(
            {
                "bg": spec["bg"],
                "direction": spec["direction"],
                "pattern": spec["pattern"],
                "window_bbox": (wr0 - top, wc0 - left, wr1 - top, wc1 - left),
                "room_bbox": (rr0 - top, rc0 - left, rr1 - top, rc1 - left),
            }
        )
    return cropped, adjusted


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for spec in specs:
        rr0, rc0, rr1, rc1 = spec["room_bbox"]
        rotated = np.rot90(spec["pattern"], ROTATION_K[str(spec["direction"])])
        target = out[rr0 + 1 : rr1, rc0 + 1 : rc1]
        assert target.shape == rotated.shape
        target[:, :] = rotated
    return out


def _is_bg_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)]) == bg


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    n_pairs = rng.choice(GEN_PAIR_COUNTS)
    border_colors = rng.sample([c for c in range(10) if c != bg], n_pairs)
    pattern_palette = [c for c in range(10) if c != bg and c not in border_colors]
    directions = _choose_directions(rng, n_pairs)

    pair_specs: list[dict[str, object]] = []
    rectangles: list[dict[str, int]] = []

    for pair_idx in range(n_pairs):
        direction = directions[pair_idx]
        if direction in ("left", "right"):
            ph = rng.randint(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX)
            pw_choices = [w for w in range(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX + 1) if w != ph]
            pw = rng.choice(pw_choices)
        else:
            ph = rng.randint(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX)
            pw = rng.randint(GEN_INTERIOR_MIN, GEN_INTERIOR_MAX)

        pattern = _random_pattern(ph, pw, bg, pattern_palette, ROTATION_K[direction], rng)
        if pattern is None:
            return None

        room_shape = np.rot90(pattern, ROTATION_K[direction]).shape
        pair_specs.append(
            {
                "bg": bg,
                "border_color": border_colors[pair_idx],
                "direction": direction,
                "pattern": pattern,
            }
        )
        rectangles.append({"pair_idx": pair_idx, "kind": 0, "h": ph + 2, "w": pw + 2})
        rectangles.append({"pair_idx": pair_idx, "kind": 1, "h": int(room_shape[0]) + 2, "w": int(room_shape[1]) + 2})

    placements = _place_rectangles(rectangles, rng)
    if placements is None:
        return None

    grid = np.full((GEN_CANVAS, GEN_CANVAS), bg, dtype=int)
    full_specs: list[dict[str, object]] = []
    for rect_idx, rect in enumerate(rectangles):
        top, left = placements[rect_idx]
        spec = pair_specs[rect["pair_idx"]]
        if rect["kind"] == 0:
            window_bbox = _draw_window(grid, top, left, int(spec["border_color"]), spec["pattern"])
            spec["window_bbox"] = window_bbox
        else:
            room_bbox = _draw_room(
                grid,
                top,
                left,
                int(spec["border_color"]),
                np.rot90(spec["pattern"], ROTATION_K[str(spec["direction"])]).shape,
                str(spec["direction"]),
                bg,
                rng,
            )
            spec["room_bbox"] = room_bbox

    for spec in pair_specs:
        full_specs.append(
            {
                "bg": spec["bg"],
                "direction": spec["direction"],
                "pattern": spec["pattern"],
                "window_bbox": spec["window_bbox"],
                "room_bbox": spec["room_bbox"],
            }
        )

    grid, cropped_specs = _crop_canvas(grid, full_specs, rng)
    if max(grid.shape) > ARC_LIMIT or not _is_bg_dominant(grid, bg):
        return None

    expected = _expected_output(grid, cropped_specs)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
