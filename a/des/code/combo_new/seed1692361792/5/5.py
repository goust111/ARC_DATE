import random
from collections import defaultdict

import numpy as np


BG = 0
SIDES = ("top", "left", "bottom", "right")
MAX_GENERATION_ATTEMPTS = 400
POINT_COUNT_VALUES = (1, 2, 3, 4)
POINT_COUNT_WEIGHTS = (1, 3, 3, 2)
FRAME_COUNT_CHOICES = (2, 3, 3, 4)
GRID_MIN_BY_FRAME_COUNT = {2: 14, 3: 18, 4: 20}
FRAME_MAX_OUTER_BY_COUNT = {2: 12, 3: 10, 4: 8}
PLACEMENT_TRIES = 120
TIE_PROBABILITY = 0.35
COLLISION_PROBABILITY = 0.4

_CANDIDATE_CACHE: dict[tuple[int, int], tuple[list[dict], dict[str, list[dict]], dict[str, dict[tuple[int, int], list[dict]]]]] = {}


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = map(int, grid.shape)
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or int(grid[row, col]) != color:
                continue
            stack = [(row, col)]
            visited[row, col] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for comp in _color_components(grid, color):
            if len(comp) < 8:
                continue
            pts = np.array(comp, dtype=int)
            r0, c0 = map(int, pts.min(axis=0))
            r1, c1 = map(int, pts.max(axis=0))
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            if height < 3 or width < 3:
                continue
            if len(comp) != 2 * height + 2 * width - 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _border_mask(height, width)
            if np.all(sub[border] == color) and not np.any(sub[~border] == color):
                frames.append({"color": color, "bbox": (r0, c0, r1, c1)})
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _project_point(row: int, col: int, r0: int, c0: int, r1: int, c1: int) -> tuple[str, int, int]:
    distances = (
        ("top", int(row - r0)),
        ("left", int(col - c0)),
        ("bottom", int(r1 - row)),
        ("right", int(c1 - col)),
    )
    best = min(distance for _side, distance in distances)
    side = next(side for side, distance in distances if distance == best)
    if side == "top":
        return side, int(r0), int(col)
    if side == "left":
        return side, int(row), int(c0)
    if side == "bottom":
        return side, int(r1), int(col)
    return side, int(row), int(c1)


def _candidate_entries(height: int, width: int) -> tuple[list[dict], dict[str, list[dict]], dict[str, dict[tuple[int, int], list[dict]]]]:
    key = (height, width)
    if key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[key]

    entries: list[dict] = []
    by_side: dict[str, list[dict]] = {side: [] for side in SIDES}
    by_target: dict[str, dict[tuple[int, int], list[dict]]] = {side: defaultdict(list) for side in SIDES}
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            distances = (
                ("top", row),
                ("left", col),
                ("bottom", height - 1 - row),
                ("right", width - 1 - col),
            )
            best = min(distance for _side, distance in distances)
            tied = [side for side, distance in distances if distance == best]
            side, tr, tc = _project_point(row, col, 0, 0, height - 1, width - 1)
            entry = {
                "pos": (row, col),
                "side": side,
                "target": (tr, tc),
                "tie": len(tied) > 1,
            }
            entries.append(entry)
            by_side[side].append(entry)
            by_target[side][entry["target"]].append(entry)

    grouped_targets = {side: dict(targets) for side, targets in by_target.items()}
    _CANDIDATE_CACHE[key] = (entries, by_side, grouped_targets)
    return _CANDIDATE_CACHE[key]


def _pick_entry(rng: random.Random, pool: list[dict], used_positions: set[tuple[int, int]], prefer_tie: bool) -> dict | None:
    available = [entry for entry in pool if entry["pos"] not in used_positions]
    if not available:
        return None
    if prefer_tie:
        tied = [entry for entry in available if entry["tie"]]
        if tied:
            available = tied
    return rng.choice(available)


def _choose_frame_points(
    rng: random.Random,
    height: int,
    width: int,
    side_order: list[str],
    must_include_tie: bool,
    must_include_collision: bool,
) -> list[dict] | None:
    entries, by_side, by_target = _candidate_entries(height, width)
    point_count = rng.choices(POINT_COUNT_VALUES, weights=POINT_COUNT_WEIGHTS, k=1)[0]
    if must_include_collision:
        point_count = max(point_count, 2)

    selected: list[dict] = []
    used_positions: set[tuple[int, int]] = set()

    if must_include_collision:
        pair_options: list[list[dict]] = []
        for side in SIDES:
            for group in by_target[side].values():
                if len(group) >= 2:
                    pair_options.append(group)
        if not pair_options:
            return None
        group = rng.choice(pair_options)
        pair = rng.sample(group, 2)
        for entry in pair:
            selected.append(entry)
            used_positions.add(entry["pos"])

    remaining = point_count - len(selected)
    need_tie = must_include_tie
    for idx in range(remaining):
        desired_side = side_order[idx % len(side_order)]
        prefer_tie = need_tie or rng.random() < TIE_PROBABILITY
        entry = _pick_entry(rng, by_side[desired_side], used_positions, prefer_tie)
        if entry is None:
            entry = _pick_entry(rng, entries, used_positions, prefer_tie)
        if entry is None:
            return None
        selected.append(entry)
        used_positions.add(entry["pos"])
        if entry["tie"]:
            need_tie = False

    if need_tie:
        return None
    return selected


def _assign_point_colors(rng: random.Random, selected: list[dict], palette: list[int]) -> dict[tuple[int, int], int]:
    assigned: dict[tuple[int, int], int] = {}
    by_target: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for entry in selected:
        by_target[entry["target"]].append(entry)

    for group in by_target.values():
        if len(group) <= 1 or len(palette) <= 1:
            continue
        colors = rng.sample(palette, k=min(len(group), len(palette)))
        for entry, color in zip(group, colors):
            assigned[entry["pos"]] = color

    for entry in selected:
        pos = entry["pos"]
        if pos in assigned:
            continue
        row, col = pos
        blocked = set()
        for (other_row, other_col), other_color in assigned.items():
            if abs(other_row - row) + abs(other_col - col) == 1:
                blocked.add(other_color)
        choices = [color for color in palette if color not in blocked]
        if not choices:
            choices = palette
        assigned[pos] = rng.choice(choices)
    return assigned


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _validate_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    try:
        frames = _find_frames(grid)
        if len(frames) != expected_frames:
            return False

        frame_colors = {int(frame["color"]) for frame in frames}
        border_mask = np.zeros_like(grid, dtype=bool)
        interior_mask = np.zeros_like(grid, dtype=bool)
        seen_sides: set[str] = set()
        point_total = 0

        for frame in frames:
            r0, c0, r1, c1 = frame["bbox"]
            local_border = _border_mask(r1 - r0 + 1, c1 - c0 + 1)
            border_mask[r0 : r1 + 1, c0 : c1 + 1] |= local_border
            interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

            points = []
            for row in range(r0 + 1, r1):
                for col in range(c0 + 1, c1):
                    color = int(grid[row, col])
                    if color == BG:
                        continue
                    if color in frame_colors:
                        return False
                    points.append((row, col))
                    side, _tr, _tc = _project_point(row, col, r0, c0, r1, c1)
                    seen_sides.add(side)

            if not 1 <= len(points) <= 4:
                return False
            point_total += len(points)

        non_bg = grid != BG
        extra_pixels = non_bg & ~border_mask
        if np.any(extra_pixels & ~interior_mask):
            return False
        if point_total == 0 or len(seen_sides) < min(3, point_total):
            return False

        out = main(grid)
        return isinstance(out, np.ndarray) and out.shape == grid.shape
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    min_dim = GRID_MIN_BY_FRAME_COUNT[frame_count]
    height = rng.randint(min_dim, 30)
    width = rng.randint(min_dim, 30)
    frame_max_outer = min(FRAME_MAX_OUTER_BY_COUNT[frame_count], height - 2, width - 2)
    if frame_max_outer < 5:
        return None

    colors = list(range(1, 10))
    rng.shuffle(colors)
    frame_colors = colors[:frame_count]
    palette_pool = colors[frame_count:]
    palette_size = rng.randint(2, min(4, len(palette_pool)))
    point_palette = palette_pool[:palette_size]

    placements: list[dict] = []
    blocked = np.zeros((height, width), dtype=bool)
    for idx in range(frame_count):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            outer_h = rng.randint(5, frame_max_outer)
            outer_w = rng.randint(5, frame_max_outer)
            if height - outer_h - 1 < 1 or width - outer_w - 1 < 1:
                continue
            r0 = rng.randint(1, height - outer_h - 1)
            c0 = rng.randint(1, width - outer_w - 1)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c1 + 2)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            blocked[rr0:rr1, cc0:cc1] = True
            placements.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "color": frame_colors[idx],
                }
            )
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    for frame in placements:
        _draw_frame(grid, frame["bbox"], int(frame["color"]))

    base_order = list(SIDES)
    rng.shuffle(base_order)
    tie_frame = rng.randrange(frame_count) if rng.random() < 0.5 else None
    collision_frame = rng.randrange(frame_count) if rng.random() < COLLISION_PROBABILITY else None
    if tie_frame is not None and collision_frame is not None and tie_frame == collision_frame and frame_count > 1:
        collision_frame = (collision_frame + 1) % frame_count

    for idx, frame in enumerate(placements):
        side_order = base_order[idx % 4 :] + base_order[: idx % 4]
        selected = _choose_frame_points(
            rng,
            int(frame["outer_h"]),
            int(frame["outer_w"]),
            side_order,
            must_include_tie=idx == tie_frame,
            must_include_collision=idx == collision_frame,
        )
        if not selected:
            return None
        colors_by_pos = _assign_point_colors(rng, selected, point_palette)
        r0, c0, _r1, _c1 = frame["bbox"]
        for entry in sorted(selected, key=lambda item: item["pos"]):
            lr, lc = entry["pos"]
            grid[r0 + lr, c0 + lc] = int(colors_by_pos[entry["pos"]])

    if not _validate_generated_grid(grid, expected_frames=frame_count):
        return None
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored point inside a hollow rectangle to its nearest frame edge.

    Input structure:
    - The background is black.
    - The grid contains several non-overlapping 1-pixel-thick hollow rectangles, each in its own color.
    - Inside each rectangle there are 1 to 4 colored single-pixel markers on black cells, and these marker colors differ from every frame color.

    Rule:
    - For each marker, compare its distances to the top, left, bottom, and right inner walls of its containing rectangle, using priority top > left > bottom > right to break ties.
    - Recolor the aligned border pixel on that chosen side with the marker color, preserving the marker's column for horizontal sides or row for vertical sides.
    - Process markers in row-major order so later markers overwrite earlier ones when they project to the same border cell.
    - Clear all interior markers back to black while leaving every other frame pixel unchanged.

    Key insight:
    - The transformation is a nearest-edge projection inside each frame, with orthogonal alignment deciding where the color lands on the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one hollow rectangle frame"

    frame_colors = {int(frame["color"]) for frame in frames}
    border_mask = np.zeros_like(g, dtype=bool)
    interior_mask = np.zeros_like(g, dtype=bool)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        border_mask[r0 : r1 + 1, c0 : c1 + 1] |= _border_mask(r1 - r0 + 1, c1 - c0 + 1)
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

        points: list[tuple[int, int, int]] = []
        for row in range(r0 + 1, r1):
            for col in range(c0 + 1, c1):
                color = int(g[row, col])
                if color == BG:
                    continue
                assert color not in frame_colors, "marker colors must differ from frame colors"
                points.append((row, col, color))
                out[row, col] = BG

        assert 1 <= len(points) <= 4, "each frame must contain 1 to 4 markers"
        for row, col, color in points:
            _side, target_row, target_col = _project_point(row, col, r0, c0, r1, c1)
            out[target_row, target_col] = color

    extras = (g != BG) & ~border_mask
    assert not np.any(extras & ~interior_mask), "all non-frame colors must lie inside a frame"
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
