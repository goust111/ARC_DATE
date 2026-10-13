import random
from collections import deque

import numpy as np


BLACK = 0
WALL_COLOR = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 25
GRID_MAX = 30
ROOM_COUNT_CHOICES = [3, 4]
CENTER_SPAN_CHOICES = [1, 2, 3, 4]
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rectangular_frame(points: list[tuple[int, int]]) -> tuple[bool, tuple[int, int, int, int]]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False, (r0, c0, r1, c1)

    expected_area = 2 * (height + width) - 4
    if len(points) != expected_area:
        return False, (r0, c0, r1, c1)

    point_set = set(points)
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False, (r0, c0, r1, c1)
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False, (r0, c0, r1, c1)
    for r, c in points:
        if r not in (r0, r1) and c not in (c0, c1):
            return False, (r0, c0, r1, c1)
    return True, (r0, c0, r1, c1)


def _find_rooms(grid: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    rooms: list[tuple[int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BLACK:
            continue
        for comp in _components(grid == color):
            is_room, bbox = _is_rectangular_frame(comp)
            if is_room:
                r0, c0, r1, c1 = bbox
                rooms.append((r0, c0, r1, c1, color))
    rooms.sort()
    return rooms


def _carpet_mask(inner_h: int, inner_w: int, thickness: int) -> np.ndarray:
    rows = np.arange(inner_h)[:, None]
    cols = np.arange(inner_w)[None, :]
    dist = rows.copy()
    dist = np.minimum(dist, cols)
    dist = np.minimum(dist, inner_h - 1 - rows)
    dist = np.minimum(dist, inner_w - 1 - cols)
    return dist < thickness


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each room's inner border with a color ring whose thickness matches its seed count.

    Input structure:
    - The background is black.
    - One-pixel-thick hollow rectangular rooms appear as disconnected gray frames.
    - Each room contains 1, 2, or 3 same-colored seed pixels somewhere inside the frame.

    Rule:
    - For each hollow room, count the non-black non-wall seed pixels inside it.
    - Use that room's seed color to paint the N innermost layers adjacent to the room's walls,
      where N is the seed count.
    - Seed pixels that lie in the painted border are overwritten by the carpet; seed pixels deeper
      in the untouched center remain as they are. Walls and the outside background stay unchanged.

    Key insight:
    - The seed color supplies the fill color, while the number of seeds converts directly into the
      thickness of a rectangular inner ring.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1, wall_color in _find_rooms(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (inner != BLACK) & (inner != wall_color)
        seed_colors = np.unique(inner[seed_mask])
        assert len(seed_colors) == 1, "each room must contain exactly one seed color"
        seed_color = int(seed_colors[0])
        seed_count = int(seed_mask.sum())
        assert seed_count in (1, 2, 3), "seed count must be 1, 2, or 3"

        carpet = _carpet_mask(inner.shape[0], inner.shape[1], seed_count)
        out[r0 + 1 : r1, c0 + 1 : c1][carpet] = seed_color

    return out


def _sample_room_counts(rng: random.Random, room_count: int) -> list[int]:
    counts = [1, 2, 3]
    if room_count == 4:
        counts.append(rng.choice([1, 2, 3]))
    rng.shuffle(counts)
    return counts


def _assign_modes(rng: random.Random, counts: list[int]) -> list[str]:
    room_count = len(counts)
    modes = [""] * room_count
    mixed_idx = rng.choice([idx for idx, count in enumerate(counts) if count >= 2])
    remaining = [idx for idx in range(room_count) if idx != mixed_idx]
    carpet_idx, center_idx = rng.sample(remaining, 2)
    modes[mixed_idx] = "mixed"
    modes[carpet_idx] = "carpet"
    modes[center_idx] = "center"

    for idx in range(room_count):
        if modes[idx]:
            continue
        options = ["carpet", "center"]
        if counts[idx] >= 2:
            options.append("mixed")
        modes[idx] = rng.choice(options)
    return modes


def _sample_center_dims(rng: random.Random, min_area: int, max_h: int, max_w: int) -> tuple[int, int] | None:
    options = [
        (center_h, center_w)
        for center_h in CENTER_SPAN_CHOICES
        for center_w in CENTER_SPAN_CHOICES
        if center_h <= max_h and center_w <= max_w and center_h * center_w >= min_area
    ]
    if not options:
        return None
    return rng.choice(options)


def _pick_seed_positions(
    rng: random.Random, inner_h: int, inner_w: int, seed_count: int, mode: str
) -> list[tuple[int, int]]:
    carpet = _carpet_mask(inner_h, inner_w, seed_count)
    carpet_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if carpet[r, c]]
    center_cells = [(r, c) for r in range(inner_h) for c in range(inner_w) if not carpet[r, c]]

    if mode == "carpet":
        return rng.sample(carpet_cells, seed_count)
    if mode == "center":
        return rng.sample(center_cells, seed_count)

    center_count = rng.randint(1, min(seed_count - 1, len(center_cells)))
    carpet_count = seed_count - center_count
    return rng.sample(center_cells, center_count) + rng.sample(carpet_cells, carpet_count)


def _sample_room_geometry(
    rng: random.Random, seed_count: int, mode: str, slot_h: int, slot_w: int
) -> tuple[int, int, list[tuple[int, int]]] | None:
    options: list[tuple[int, int, int, int]] = []
    for center_h in CENTER_SPAN_CHOICES:
        for center_w in CENTER_SPAN_CHOICES:
            center_area = center_h * center_w
            if mode == "center" and center_area < seed_count:
                continue
            inner_h = 2 * seed_count + center_h
            inner_w = 2 * seed_count + center_w
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h <= slot_h and outer_w <= slot_w:
                options.append((outer_h, outer_w, inner_h, inner_w))
    if not options:
        return None

    outer_h, outer_w, inner_h, inner_w = rng.choice(options)
    seeds = _pick_seed_positions(rng, inner_h, inner_w, seed_count, mode)
    return outer_h, outer_w, seeds


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    seed_color: int,
    seed_positions: list[tuple[int, int]],
) -> None:
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = WALL_COLOR
    grid[r1, c0 : c1 + 1] = WALL_COLOR
    grid[r0 : r1 + 1, c0] = WALL_COLOR
    grid[r0 : r1 + 1, c1] = WALL_COLOR
    for rr, cc in seed_positions:
        grid[r0 + 1 + rr, c0 + 1 + cc] = seed_color


def _slot_bounds(h: int, w: int, split_r: int, split_c: int) -> list[tuple[int, int, int, int]]:
    return [
        (1, 1, split_r - 1, split_c - 1),
        (1, split_c + 1, split_r - 1, w - 2),
        (split_r + 1, 1, h - 2, split_c - 1),
        (split_r + 1, split_c + 1, h - 2, w - 2),
    ]


def _validate_grid(grid: np.ndarray, expected_room_count: int) -> bool:
    rooms = _find_rooms(grid)
    if len(rooms) != expected_room_count:
        return False

    used_seed_colors: set[int] = set()
    saw_carpet_seed = False
    saw_center_seed = False
    saw_mixed_room = False
    saw_counts = set()

    for r0, c0, r1, c1, wall_color in rooms:
        if wall_color != WALL_COLOR:
            return False
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_mask = (inner != BLACK) & (inner != wall_color)
        if not np.any(seed_mask):
            return False
        seed_colors = np.unique(inner[seed_mask])
        if len(seed_colors) != 1:
            return False
        seed_color = int(seed_colors[0])
        if seed_color in used_seed_colors:
            return False
        used_seed_colors.add(seed_color)

        seed_count = int(seed_mask.sum())
        if seed_count not in (1, 2, 3):
            return False
        saw_counts.add(seed_count)

        carpet = _carpet_mask(inner.shape[0], inner.shape[1], seed_count)
        room_has_carpet_seed = bool(np.any(seed_mask & carpet))
        room_has_center_seed = bool(np.any(seed_mask & ~carpet))
        saw_carpet_seed |= room_has_carpet_seed
        saw_center_seed |= room_has_center_seed
        saw_mixed_room |= room_has_carpet_seed and room_has_center_seed

    if not {1, 2, 3}.issubset(saw_counts):
        return False
    if not saw_carpet_seed or not saw_center_seed or not saw_mixed_room:
        return False

    out = main(grid)
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    room_count = rng.choice(ROOM_COUNT_CHOICES)
    counts = _sample_room_counts(rng, room_count)
    modes = _assign_modes(rng, counts)

    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    split_r = rng.randint(12, h - 13)
    split_c = rng.randint(12, w - 13)
    slots = _slot_bounds(h, w, split_r, split_c)
    chosen_slots = rng.sample(slots, room_count)
    room_colors = rng.sample(SEED_COLORS, room_count)

    grid = np.zeros((h, w), dtype=int)

    for (slot_r0, slot_c0, slot_r1, slot_c1), seed_count, mode, seed_color in zip(
        chosen_slots, counts, modes, room_colors
    ):
        slot_h = slot_r1 - slot_r0 + 1
        slot_w = slot_c1 - slot_c0 + 1
        geometry = _sample_room_geometry(rng, seed_count, mode, slot_h, slot_w)
        if geometry is None:
            return None
        outer_h, outer_w, seeds = geometry

        r0 = rng.randint(slot_r0, slot_r1 - outer_h + 1)
        c0 = rng.randint(slot_c0, slot_c1 - outer_w + 1)
        _draw_room(grid, r0, c0, outer_h, outer_w, seed_color, seeds)

    if not _validate_grid(grid, room_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-carpet puzzle input")
