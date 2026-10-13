from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
PALETTE = tuple(range(1, 10))

GEN_RETRIES = 400
SPEC_RETRIES = 120
PLACEMENT_RETRIES = 220
SEED_RETRIES = 120

OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (1, 3, 3, 1)

FRAME_H_RANGE = (6, 12)
FRAME_W_RANGE = (6, 12)
SEED_MAX_DIM = 5


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(np.asarray(grid, dtype=int).ravel().tolist()).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _coords_to_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _is_asymmetric(mask: np.ndarray) -> bool:
    variants: list[np.ndarray] = [
        np.flipud(mask),
        np.fliplr(mask),
        np.rot90(mask, 2),
    ]
    if mask.shape[0] == mask.shape[1]:
        variants.extend([np.rot90(mask, 1), np.rot90(mask, 3), mask.T])
    return all(not np.array_equal(mask, variant) for variant in variants if variant.shape == mask.shape)


def _monochrome_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append((color, coords))
    return comps


def _frame_border_pixels_local(height: int, width: int, direction: str, gap_start: int, gap_end: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(width):
        if not (direction == "up" and gap_start <= c <= gap_end):
            border.add((0, c))
        if not (direction == "down" and gap_start <= c <= gap_end):
            border.add((height - 1, c))
    for r in range(1, height - 1):
        if not (direction == "left" and gap_start <= r <= gap_end):
            border.add((r, 0))
        if not (direction == "right" and gap_start <= r <= gap_end):
            border.add((r, width - 1))
    return border


def _project_seed(seed_coords: list[tuple[int, int]], direction: str, frame_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_box
    rows = [r for r, _ in seed_coords]
    cols = [c for _, c in seed_coords]
    if direction == "up":
        dr, dc = (r0 - 1) - max(rows), 0
    elif direction == "down":
        dr, dc = (r1 + 1) - min(rows), 0
    elif direction == "left":
        dr, dc = 0, (c0 - 1) - max(cols)
    else:
        dr, dc = 0, (c1 + 1) - min(cols)
    return [(r + dr, c + dc) for r, c in seed_coords]


def _parse_frame_component(grid: np.ndarray, bg: int, color: int, coords: list[tuple[int, int]]) -> dict | None:
    coord_set = set(coords)
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return None

    border = {(r0, c) for c in range(c0, c1 + 1)}
    border |= {(r1, c) for c in range(c0, c1 + 1)}
    border |= {(r, c0) for r in range(r0, r1 + 1)}
    border |= {(r, c1) for r in range(r0, r1 + 1)}
    if not coord_set <= border:
        return None

    missing = sorted(border - coord_set)
    if len(missing) < 2 or any(int(grid[r, c]) != bg for r, c in missing):
        return None

    direction: str | None = None
    axis = []
    if all(r == r0 for r, c in missing):
        direction = "up"
        axis = sorted(c for _, c in missing)
    elif all(r == r1 for r, c in missing):
        direction = "down"
        axis = sorted(c for _, c in missing)
    elif all(c == c0 for r, c in missing):
        direction = "left"
        axis = sorted(r for r, _ in missing)
    elif all(c == c1 for r, c in missing):
        direction = "right"
        axis = sorted(r for r, _ in missing)
    if direction is None or axis != list(range(axis[0], axis[-1] + 1)):
        return None

    seed_coords = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) not in (bg, color)
    ]
    if not seed_coords:
        return None

    seed_colors = {int(grid[r, c]) for r, c in seed_coords}
    if len(seed_colors) != 1:
        return None

    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "direction": direction,
        "gap": (int(axis[0]), int(axis[-1])),
        "seed_coords": sorted(seed_coords),
        "seed_color": int(next(iter(seed_colors))),
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = []
    for color, coords in _monochrome_components(g, bg):
        parsed = _parse_frame_component(g, bg, color, coords)
        if parsed is not None:
            frames.append(parsed)
    return sorted(frames, key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each seed pattern through its frame gap and repaint the projection with the frame color.

    Input structure:
    - The background is black.
    - Several monochrome rectangular border frames appear on the grid; each border is one cell thick and has one contiguous gap on exactly one side.
    - Each frame contains one small monochrome seed pattern of a different color, and the seed does not touch the border.

    Rule:
    - Detect each rectangular border and identify which side contains the contiguous gap; that side defines the projection direction.
    - Copy the full seed pattern straight through the gap until the copied pattern sits just outside the frame, keeping its orthogonal offset unchanged.
    - Draw the outside copy with the frame color, while leaving the original border and the original seed unchanged.

    Key insight:
    The missing border segment encodes direction, but the projected copy inherits the border color rather than the seed color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    for frame in _extract_frames(g):
        projected = _project_seed(
            frame["seed_coords"],
            frame["direction"],
            (frame["r0"], frame["c0"], frame["r1"], frame["c1"]),
        )
        for r, c in projected:
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == bg:
                out[r, c] = frame["color"]
    return out


def _random_seed_coords(rng: random.Random, max_h: int, max_w: int) -> list[tuple[int, int]]:
    for _ in range(SEED_RETRIES):
        height = rng.randint(2, max_h)
        width = rng.randint(2, max_w)
        max_area = height * width - 1
        if max_area < 3:
            continue
        target_area = rng.randint(3, max_area)
        cells = {(rng.randrange(height), rng.randrange(width))}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue
        coords = _normalize_coords(cells)
        mask = _coords_to_mask(coords)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) == mask.shape[0] * mask.shape[1]:
            continue
        if _is_asymmetric(mask):
            return coords
    raise RuntimeError("failed to create a non-symmetric seed")


def _make_object_spec(rng: random.Random, border_color: int, seed_color: int) -> dict:
    direction = rng.choice(DIRECTIONS)
    for _ in range(SPEC_RETRIES):
        frame_h = rng.randint(*FRAME_H_RANGE)
        frame_w = rng.randint(*FRAME_W_RANGE)
        max_seed_h = min(SEED_MAX_DIM, frame_h - 3)
        max_seed_w = min(SEED_MAX_DIM, frame_w - 3)
        if max_seed_h < 2 or max_seed_w < 2:
            continue

        seed_shape = _random_seed_coords(rng, max_seed_h, max_seed_w)
        seed_h = max(r for r, _ in seed_shape) + 1
        seed_w = max(c for _, c in seed_shape) + 1
        seed_top_max = frame_h - 1 - seed_h
        seed_left_max = frame_w - 1 - seed_w
        if seed_top_max < 1 or seed_left_max < 1:
            continue

        seed_top = rng.randint(1, seed_top_max)
        seed_left = rng.randint(1, seed_left_max)
        seed_local = [(seed_top + r, seed_left + c) for r, c in seed_shape]

        seed_rows = [r for r, _ in seed_local]
        seed_cols = [c for _, c in seed_local]
        if direction in ("up", "down"):
            seed_min = min(seed_cols)
            seed_max = max(seed_cols)
            gap_start = rng.randint(1, seed_min)
            gap_end_min = max(seed_max, gap_start + 1)
            if gap_end_min > frame_w - 2:
                continue
            gap_end = rng.randint(gap_end_min, frame_w - 2)
        else:
            seed_min = min(seed_rows)
            seed_max = max(seed_rows)
            gap_start = rng.randint(1, seed_min)
            gap_end_min = max(seed_max, gap_start + 1)
            if gap_end_min > frame_h - 2:
                continue
            gap_end = rng.randint(gap_end_min, frame_h - 2)

        projected_local = _project_seed(seed_local, direction, (0, 0, frame_h - 1, frame_w - 1))
        pr0, pc0, pr1, pc1 = _bbox(projected_local)
        return {
            "border_color": int(border_color),
            "seed_color": int(seed_color),
            "direction": direction,
            "frame_h": int(frame_h),
            "frame_w": int(frame_w),
            "gap_start": int(gap_start),
            "gap_end": int(gap_end),
            "frame_pixels_local": _frame_border_pixels_local(frame_h, frame_w, direction, gap_start, gap_end),
            "seed_local": seed_local,
            "projected_local": projected_local,
            "project_box_local": (int(pr0), int(pc0), int(pr1), int(pc1)),
            "reserve_score": frame_h * frame_w + (pr1 - pr0 + 1) * (pc1 - pc0 + 1),
        }
    raise RuntimeError("failed to create an object spec")


def _rect_with_pad(box: tuple[int, int, int, int], pad: int, height: int, width: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _place_object(grid: np.ndarray, reserved: np.ndarray, spec: dict, rng: random.Random) -> dict | None:
    height, width = grid.shape
    pr0, pc0, pr1, pc1 = spec["project_box_local"]

    min_row = min(0, pr0)
    min_col = min(0, pc0)
    max_row = max(spec["frame_h"] - 1, pr1)
    max_col = max(spec["frame_w"] - 1, pc1)

    top_lo = -min_row
    top_hi = height - 1 - max_row
    left_lo = -min_col
    left_hi = width - 1 - max_col
    if top_lo > top_hi or left_lo > left_hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(top_lo, top_hi)
        left = rng.randint(left_lo, left_hi)
        frame_box = (top, left, top + spec["frame_h"] - 1, left + spec["frame_w"] - 1)
        project_box = (top + pr0, left + pc0, top + pr1, left + pc1)
        blocked = _rect_with_pad(frame_box, 1, height, width) | _rect_with_pad(project_box, 1, height, width)
        if any(reserved[r, c] for r, c in blocked):
            continue

        for r, c in spec["frame_pixels_local"]:
            grid[top + r, left + c] = spec["border_color"]
        for r, c in spec["seed_local"]:
            grid[top + r, left + c] = spec["seed_color"]
        for r, c in blocked:
            reserved[r, c] = True

        return {
            **spec,
            "top": int(top),
            "left": int(left),
            "frame_box": frame_box,
            "projected_abs": [(top + r, left + c) for r, c in spec["projected_local"]],
        }
    return None


def _pick_grid_shape(rng: random.Random, object_count: int) -> tuple[int, int]:
    min_dim = 16 + max(0, object_count - 2) * 2
    min_area = {1: 150, 2: 240, 3: 330, 4: 430}[object_count]
    for _ in range(60):
        height = rng.randint(min_dim, 30)
        width = rng.randint(min_dim, 30)
        if height * width >= min_area:
            return height, width
    return 30, 30


def _validate_grid(grid: np.ndarray, placed_specs: list[dict]) -> bool:
    detected = _extract_frames(grid)
    if len(detected) != len(placed_specs):
        return False

    detected = sorted(detected, key=lambda item: (item["r0"], item["c0"]))
    placed_specs = sorted(placed_specs, key=lambda item: (item["top"], item["left"]))
    for frame, spec in zip(detected, placed_specs):
        if (frame["r0"], frame["c0"], frame["r1"], frame["c1"]) != spec["frame_box"]:
            return False
        if frame["direction"] != spec["direction"]:
            return False
        if frame["color"] != spec["border_color"]:
            return False
        if frame["seed_color"] != spec["seed_color"]:
            return False

    out = main(grid)
    expected = {pt for spec in placed_specs for pt in spec["projected_abs"]}
    changed = {(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if int(out[r, c]) != int(grid[r, c])}
    if changed != expected:
        return False

    for spec in placed_specs:
        for r, c in spec["projected_abs"]:
            if int(grid[r, c]) != 0 or int(out[r, c]) != spec["border_color"]:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    height, width = _pick_grid_shape(rng, object_count)
    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    unique_border_count = rng.randint(1, min(3, object_count))
    shuffled = list(PALETTE)
    rng.shuffle(shuffled)
    border_pool = shuffled[:unique_border_count]
    seed_pool = shuffled[unique_border_count:]
    if len(seed_pool) < object_count:
        return None

    border_colors = [rng.choice(border_pool) for _ in range(object_count)]
    seed_colors = rng.sample(seed_pool, object_count)

    specs: list[dict] = []
    try:
        for border_color, seed_color in zip(border_colors, seed_colors):
            specs.append(_make_object_spec(rng, border_color, seed_color))
    except RuntimeError:
        return None

    placed: list[dict] = []
    for spec in sorted(specs, key=lambda item: item["reserve_score"], reverse=True):
        placed_spec = _place_object(grid, reserved, spec, rng)
        if placed_spec is None:
            return None
        placed.append(placed_spec)

    if not _validate_grid(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-frame input")
