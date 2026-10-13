from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color in the grid."""
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    """Yield 4-neighbors using the canonical (row, col) convention."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = bbox
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Upscale a sprite by integer nearest-neighbor replication."""
    return np.kron(np.asarray(sprite, dtype=int), np.ones((factor, factor), dtype=int))


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _grow_polyomino(h: int, w: int, size: int) -> np.ndarray | None:
    """Grow a random connected set of exactly `size` cells inside an h x w box."""
    for _ in range(80):
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = set()
        for nr, nc in neighbors4(*start):
            if 0 <= nr < h and 0 <= nc < w:
                frontier.add((nr, nc))
        while len(cells) < size and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            cells.add(nxt)
            for nr, nc in neighbors4(*nxt):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != size:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _fill_two_color_pattern(h: int, w: int, a: int, b: int) -> np.ndarray:
    """Create a clear two-region full rectangle using two filler colors."""
    rr, cc = np.indices((h, w))
    canvas = np.full((h, w), a, dtype=int)
    style = random.choice(("vertical", "horizontal", "diag", "corner", "stripe"))
    if style == "vertical":
        cut = random.randint(1, w - 1)
        canvas[:, cut:] = b
    elif style == "horizontal":
        cut = random.randint(1, h - 1)
        canvas[cut:, :] = b
    elif style == "diag":
        threshold = random.randint(1, h + w - 2)
        canvas[rr + cc >= threshold] = b
    elif style == "corner":
        bh = random.randint(1, h - 1)
        bw = random.randint(1, w - 1)
        corner = random.choice(("tl", "tr", "bl", "br"))
        if corner == "tl":
            canvas[:bh, :bw] = b
        elif corner == "tr":
            canvas[:bh, w - bw :] = b
        elif corner == "bl":
            canvas[h - bh :, :bw] = b
        else:
            canvas[h - bh :, w - bw :] = b
    else:
        if random.choice((False, True)):
            col = random.randrange(w)
            canvas[:, col] = b
        else:
            row = random.randrange(h)
            canvas[row, :] = b
    return canvas


def _analyze_grid(grid: np.ndarray) -> dict:
    """Parse the puzzle semantics from a candidate grid and assert uniqueness."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg = get_bg(g)
    fg_components = components_mask(g != bg)
    assert fg_components, "expected foreground walls and one sprite"

    wall_components = [coords for coords in fg_components if _touches_border(coords, g.shape)]
    sprite_components = [coords for coords in fg_components if not _touches_border(coords, g.shape)]
    assert len(sprite_components) == 1, "expected exactly one non-border sprite"
    assert wall_components, "expected at least one border-touching wall component"

    wall_coords = np.concatenate(wall_components, axis=0)
    wall_colors = set(map(int, g[wall_coords[:, 0], wall_coords[:, 1]]))
    assert len(wall_colors) == 1, "wall must use exactly one visible color"
    wall_color = next(iter(wall_colors))

    sprite_coords = sprite_components[0]
    sprite_vals = g[sprite_coords[:, 0], sprite_coords[:, 1]]
    sprite_colors = set(map(int, sprite_vals))
    assert len(sprite_colors) >= 2, "sprite must be multicolor"

    sprite_mask = np.zeros_like(g, dtype=bool)
    sprite_mask[sprite_coords[:, 0], sprite_coords[:, 1]] = True
    erased = g.copy()
    erased[sprite_mask] = bg
    rooms = components_mask(erased == bg)
    room_count = len(rooms)
    assert room_count >= 2, "need one sprite room plus at least one empty room"
    n = room_count - 1

    color_counts = Counter(map(int, sprite_vals))
    matches = [color for color, count in color_counts.items() if count == n]
    assert len(matches) == 1, "sprite must have exactly one color whose count equals empty-room count"
    target_color = matches[0]

    bbox = bbox_coords(sprite_coords)
    return {
        "bg": int(bg),
        "wall_color": int(wall_color),
        "sprite_coords": sprite_coords,
        "sprite_mask": sprite_mask,
        "sprite_bbox": bbox,
        "room_count": int(room_count),
        "n": int(n),
        "target_color": int(target_color),
        "color_counts": color_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the sprite color whose population matches the empty-room count, then upscale the cleaned sprite bbox by that same factor.

    Input structure:
    - The background is the most common color.
    - Border-touching foreground components are one-color walls that partition the background into 4-connected rooms.
    - Exactly one multicolor sprite lies fully inside one room and does not touch the canvas border.

    Rule:
    - Distinguish walls from the sprite by border contact: touching components are walls, the unique non-touching component is the sprite.
    - Temporarily erase the sprite to recover the wall-defined room layout, count the number of empty rooms, and call that number n.
    - Inside the sprite, recolor the unique color that appears exactly n times to the background color, then crop the sprite bbox and upscale it n times per pixel.

    Key insight:
    - The hidden number n carries both semantic loads at once: it is inferred from the wall-made room count, and that same count singles out which sprite color is disposable before scaling.
    """
    info = _analyze_grid(grid)
    g = np.asarray(grid, dtype=int)
    bg = info["bg"]
    target_color = info["target_color"]
    n = info["n"]

    sprite_box = crop_bbox(g, info["sprite_bbox"])
    sprite_box[sprite_box == target_color] = bg
    return scale_sprite(sprite_box, n)


ROOM_COUNT_RANGE = (2, 5)
GRID_SIDE_MAX = 28
ROOM_SPLIT_MIN = 5
SPRITE_MIN_DIM = 3
SPRITE_MAX_DIM = 7
GEN_RETRIES = 400


def _split_rooms(h: int, w: int, split_count: int) -> tuple[list[tuple[int, int, int, int]], np.ndarray] | tuple[None, None]:
    """Randomly partition the canvas into `split_count + 1` rectangular rooms using connected walls."""
    rooms = [(0, h - 1, 0, w - 1)]
    walls = np.zeros((h, w), dtype=bool)
    for _ in range(split_count):
        options: list[tuple[int, str]] = []
        for idx, (r0, r1, c0, c1) in enumerate(rooms):
            if c1 - c0 + 1 >= 2 * ROOM_SPLIT_MIN + 1:
                options.append((idx, "v"))
            if r1 - r0 + 1 >= 2 * ROOM_SPLIT_MIN + 1:
                options.append((idx, "h"))
        if not options:
            return None, None
        idx, orient = random.choice(options)
        r0, r1, c0, c1 = rooms.pop(idx)
        if orient == "v":
            cut = random.randint(c0 + ROOM_SPLIT_MIN, c1 - ROOM_SPLIT_MIN)
            walls[r0 : r1 + 1, cut] = True
            rooms.extend(((r0, r1, c0, cut - 1), (r0, r1, cut + 1, c1)))
        else:
            cut = random.randint(r0 + ROOM_SPLIT_MIN, r1 - ROOM_SPLIT_MIN)
            walls[cut, c0 : c1 + 1] = True
            rooms.extend(((r0, cut - 1, c0, c1), (cut + 1, r1, c0, c1)))
    return rooms, walls


def _build_sprite(h: int, w: int, n: int, bg: int, wall_color: int) -> tuple[np.ndarray, int] | tuple[None, None]:
    """Create a full rectangular multicolor sprite with a unique n-count color."""
    palette = [color for color in range(10) if color != bg]
    area = h * w
    for _ in range(200):
        use_wall_inside = random.random() < 0.6
        if use_wall_inside and random.random() < 0.45:
            scaffold_color = wall_color
        else:
            scaffold_choices = [color for color in palette if color != wall_color]
            scaffold_color = random.choice(scaffold_choices)

        filler_count = 2 if area - n >= 6 and random.random() < 0.65 else 1
        fillers: list[int] = []
        if use_wall_inside and wall_color != scaffold_color and filler_count >= 1 and random.random() < 0.55:
            fillers.append(wall_color)
        filler_pool = [color for color in palette if color != scaffold_color and color not in fillers]
        if len(filler_pool) < filler_count - len(fillers):
            continue
        fillers.extend(random.sample(filler_pool, filler_count - len(fillers)))

        scaffold_mask = _grow_polyomino(h, w, n)
        if scaffold_mask is None:
            continue

        sprite = np.full((h, w), fillers[0], dtype=int)
        if filler_count == 2:
            sprite = _fill_two_color_pattern(h, w, fillers[0], fillers[1])
        sprite[scaffold_mask] = scaffold_color

        counts = Counter(map(int, sprite.ravel()))
        if len(counts) < 2:
            continue
        if counts[scaffold_color] != n:
            continue
        if sum(count == n for count in counts.values()) != 1:
            continue
        if any(count == n + 1 for count in counts.values()):
            continue
        return sprite, scaffold_color
    return None, None


def _expected_output(sprite: np.ndarray, bg: int, target_color: int, n: int) -> np.ndarray:
    cleaned = np.asarray(sprite, dtype=int).copy()
    cleaned[cleaned == target_color] = bg
    return scale_sprite(cleaned, n)


def _validate_instance(grid: np.ndarray, sprite: np.ndarray, target_color: int, n: int) -> bool:
    """Check structure, uniqueness, and that both counting and erasure are indispensable."""
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if info["n"] != n or info["target_color"] != target_color:
        return False
    if info["room_count"] != n + 1 or n <= 1:
        return False
    if any(count == n + 1 for count in info["color_counts"].values()):
        return False

    expected = _expected_output(sprite, info["bg"], target_color, n)
    predicted = main(grid)
    if not np.array_equal(predicted, expected):
        return False

    original_box = crop_bbox(grid, info["sprite_bbox"])
    if np.array_equal(scale_sprite(original_box, n), predicted):
        return False
    if predicted.shape != (sprite.shape[0] * n, sprite.shape[1] * n):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    """Sample one valid puzzle instance or return None if the attempt degenerates."""
    n = random.randint(*ROOM_COUNT_RANGE)
    min_side = max(18, 12 + 2 * n)
    h = random.randint(min_side, GRID_SIDE_MAX)
    w = random.randint(min_side, GRID_SIDE_MAX)

    bg = random.randrange(10)
    wall_choices = [color for color in range(10) if color != bg]
    wall_color = random.choice(wall_choices)

    rooms, walls = _split_rooms(h, w, n)
    if rooms is None:
        return None

    candidates = []
    for room in rooms:
        r0, r1, c0, c1 = room
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        max_sprite_h = min(SPRITE_MAX_DIM, room_h - 2, 30 // n)
        max_sprite_w = min(SPRITE_MAX_DIM, room_w - 2, 30 // n)
        if max_sprite_h >= SPRITE_MIN_DIM and max_sprite_w >= SPRITE_MIN_DIM:
            candidates.append((room, max_sprite_h, max_sprite_w))
    if not candidates:
        return None

    room, max_sprite_h, max_sprite_w = random.choice(candidates)
    sprite_h = random.randint(SPRITE_MIN_DIM, max_sprite_h)
    sprite_w = random.randint(SPRITE_MIN_DIM, max_sprite_w)
    if sprite_h * sprite_w <= n + 1:
        return None

    sprite, target_color = _build_sprite(sprite_h, sprite_w, n, bg, wall_color)
    if sprite is None:
        return None

    r0, r1, c0, c1 = room
    top = random.randint(r0 + 1, r1 - sprite_h)
    left = random.randint(c0 + 1, c1 - sprite_w)

    grid = np.full((h, w), bg, dtype=int)
    grid[walls] = wall_color
    grid[top : top + sprite_h, left : left + sprite_w] = sprite

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    if not _validate_instance(grid, sprite, target_color, n):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a random non-degenerate input that satisfies the puzzle semantics."""
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid empty-room upscale puzzle")
