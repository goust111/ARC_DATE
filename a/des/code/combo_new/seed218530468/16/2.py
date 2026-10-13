from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))

# Generator configuration.
GEN_ATTEMPTS = 400
PARTITION_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 120
PAIR_COUNT_RANGE = (2, 4)
BOX_HEIGHT_RANGE = (3, 6)
BOX_WIDTH_RANGE = (3, 7)
MAX_GRID_SIZE = 30
SEPARATOR = 1


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < height:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < width:
        out.append((r, c + 1))
    return out


def _connected(mask: np.ndarray) -> bool:
    mask = np.asarray(mask, dtype=bool)
    points = np.argwhere(mask)
    if points.size == 0:
        return False

    height, width = mask.shape
    start = tuple(map(int, points[0]))
    queue = deque([start])
    seen = {start}

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, height, width):
            if mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _find_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    seen = np.zeros_like(g, dtype=bool)
    components: list[dict] = []

    for r in range(height):
        for c in range(width):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, height, width):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True

            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return components


def _aligned_rectangle_dims(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    ha, wa = map(int, mask_a.shape)
    hb, wb = map(int, mask_b.shape)
    height = max(ha, hb)
    width = max(wa, wb)

    canvas_a = np.zeros((height, width), dtype=bool)
    canvas_b = np.zeros((height, width), dtype=bool)
    canvas_a[:ha, :wa] = mask_a
    canvas_b[:hb, :wb] = mask_b

    assert not np.any(canvas_a & canvas_b), "paired fragments must not overlap when aligned"
    assert np.all(canvas_a | canvas_b), "paired fragments must fill a rectangle when aligned"
    return height, width


def _reconstructed_rectangles(grid: np.ndarray) -> list[dict]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for component in _find_components(grid):
        by_color[int(component["color"])].append(component)

    rectangles: list[dict] = []
    for color, parts in by_color.items():
        assert len(parts) == 2, "each foreground color must appear in exactly two fragments"
        height, width = _aligned_rectangle_dims(parts[0]["mask"], parts[1]["mask"])
        top = min(int(parts[0]["bbox"][0]), int(parts[1]["bbox"][0]))
        left = min(int(parts[0]["bbox"][1]), int(parts[1]["bbox"][1]))
        rectangles.append(
            {
                "color": int(color),
                "h": int(height),
                "w": int(width),
                "area": int(height * width),
                "anchor": (top, left),
            }
        )
    return rectangles


def _pack_layout(rectangles: list[dict], canvas_width: int) -> tuple[list[tuple[int, int]], int]:
    placements: list[tuple[int, int]] = []
    row = 0
    col_end = 0
    row_height = 0
    used_height = 0

    for rect in rectangles:
        h = int(rect["h"])
        w = int(rect["w"])
        assert 1 <= w <= canvas_width

        needs_wrap = placements and col_end + SEPARATOR + w > canvas_width
        if needs_wrap:
            row += row_height + SEPARATOR
            col_end = 0
            row_height = 0

        col = 0 if col_end == 0 else col_end + SEPARATOR
        placements.append((row, col))
        col_end = col + w
        row_height = max(row_height, h)
        used_height = max(used_height, row + h)

    return placements, used_height


def _render_rectangles(rectangles: list[dict], canvas_width: int) -> np.ndarray:
    placements, used_height = _pack_layout(rectangles, canvas_width)
    out = np.full((used_height, canvas_width), BG, dtype=int)
    for rect, (row, col) in zip(rectangles, placements):
        out[row : row + int(rect["h"]), col : col + int(rect["w"])] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored fragments, rebuild their rectangles, and pack those rectangles by descending area.

    Input structure:
    - Background is black.
    - Non-black pixels form monochrome 4-connected fragments, and each foreground color appears in exactly two separate fragments.
    - For each color, the two fragments become complementary parts of one rectangle after cropping each to its own bbox and top-left aligning the cropped masks.

    Rule:
    - Group fragments by color.
    - For each color, crop both fragments to their bboxes, top-left align those cropped masks, and merge them; the aligned union is a solid rectangle of that color.
    - Sort the reconstructed rectangles by area from largest to smallest.
    - Pack the solid rectangles onto a fresh black canvas using the input width as the row width: place rectangles left to right with one blank column between neighbors, and when the next rectangle would overflow, start a new row one blank row below the tallest rectangle in the previous row.

    Key insight:
    - The scattered irregular pieces are a distraction: the real task is first to recognize that each same-color pair complements into a rectangle, and then to forget the original locations and repack only the recovered rectangle sizes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles = _reconstructed_rectangles(g)
    rectangles.sort(key=lambda rect: (-int(rect["area"]), int(rect["anchor"][0]), int(rect["anchor"][1])))
    return _render_rectangles(rectangles, int(g.shape[1]))


def _sample_partition_box(height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    top_choices = [(0, c) for c in range(1, width)]
    left_choices = [(r, 0) for r in range(1, height)]
    area = height * width
    min_target = max(3, area // 3)
    max_target = min(area - 3, (2 * area) // 3)
    if min_target > max_target:
        return None

    for _ in range(PARTITION_ATTEMPTS):
        top_seed = random.choice(top_choices)
        left_seed = random.choice(left_choices)
        reserved = {top_seed, left_seed}
        target = random.randint(min_target, max_target)

        part_a = {(0, 0)}
        frontier = {
            nbr
            for nbr in _neighbors4(0, 0, height, width)
            if nbr not in reserved
        }

        while len(part_a) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            part_a.add(cell)
            for nbr in _neighbors4(cell[0], cell[1], height, width):
                if nbr not in part_a and nbr not in reserved:
                    frontier.add(nbr)

        if len(part_a) != target:
            continue

        mask_a = np.zeros((height, width), dtype=bool)
        for r, c in part_a:
            mask_a[r, c] = True
        mask_b = ~mask_a

        if not (mask_b[top_seed] and mask_b[left_seed]):
            continue
        if not _connected(mask_a) or not _connected(mask_b):
            continue

        crop_a = _crop_mask(mask_a)
        crop_b = _crop_mask(mask_b)
        if int(crop_a.sum()) == crop_a.size or int(crop_b.sum()) == crop_b.size:
            continue
        if min(crop_a.shape) < 2 or min(crop_b.shape) < 2:
            continue

        aligned_h, aligned_w = _aligned_rectangle_dims(crop_a, crop_b)
        if aligned_h != height or aligned_w != width:
            continue
        return crop_a, crop_b

    return None


def _choose_rectangles() -> list[dict] | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(PALETTE, pair_count)

    choices = [
        (h, w)
        for h in range(BOX_HEIGHT_RANGE[0], BOX_HEIGHT_RANGE[1] + 1)
        for w in range(BOX_WIDTH_RANGE[0], BOX_WIDTH_RANGE[1] + 1)
    ]
    random.shuffle(choices)

    used_areas: set[int] = set()
    rectangles: list[dict] = []
    for color in colors:
        random.shuffle(choices)
        picked = False
        for height, width in choices:
            if height * width in used_areas:
                continue
            pieces = _sample_partition_box(height, width)
            if pieces is None:
                continue
            used_areas.add(height * width)
            rectangles.append(
                {
                    "color": int(color),
                    "h": int(height),
                    "w": int(width),
                    "area": int(height * width),
                    "piece_a": pieces[0],
                    "piece_b": pieces[1],
                }
            )
            picked = True
            break
        if not picked:
            return None

    rectangles.sort(key=lambda rect: -int(rect["area"]))
    return rectangles


def _wrap_count(rectangles: list[dict], canvas_width: int) -> int:
    placements, _ = _pack_layout(rectangles, canvas_width)
    return len({row for row, _ in placements})


def _choose_canvas_width(rectangles: list[dict], want_wrap: bool) -> int | None:
    max_width = max(int(rect["w"]) for rect in rectangles)
    one_row_width = sum(int(rect["w"]) for rect in rectangles) + SEPARATOR * (len(rectangles) - 1)

    candidates = []
    for width in range(max_width, MAX_GRID_SIZE + 1):
        rows = _wrap_count(rectangles, width)
        if want_wrap and rows >= 2:
            candidates.append(width)
        if not want_wrap and rows == 1:
            candidates.append(width)

    if not candidates:
        return None

    if want_wrap:
        return random.choice(candidates[: max(1, len(candidates) // 2 + 1)])
    return random.choice(candidates)


def _sprite_grid(mask: np.ndarray, color: int) -> np.ndarray:
    sprite = np.full(mask.shape, BG, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _place_sprites(
    height: int,
    width: int,
    sprites: list[dict],
) -> np.ndarray | None:
    occupied = np.zeros((height, width), dtype=bool)
    grid = np.full((height, width), BG, dtype=int)

    order = list(range(len(sprites)))
    order.sort(key=lambda idx: sprites[idx]["sprite"].shape[0] * sprites[idx]["sprite"].shape[1], reverse=True)

    for idx in order:
        sprite = sprites[idx]["sprite"]
        sh, sw = map(int, sprite.shape)
        positions: list[tuple[int, int]] = []
        for r in range(height - sh + 1):
            for c in range(width - sw + 1):
                r0 = max(0, r - 1)
                c0 = max(0, c - 1)
                r1 = min(height, r + sh + 1)
                c1 = min(width, c + sw + 1)
                if not occupied[r0:r1, c0:c1].any():
                    positions.append((r, c))
        if not positions:
            return None

        row, col = random.choice(positions)
        occupied[row : row + sh, col : col + sw] = True
        patch = sprite != BG
        grid[row : row + sh, col : col + sw][patch] = sprite[patch]

    return grid


def _gen_attempt() -> np.ndarray | None:
    rectangles = _choose_rectangles()
    if rectangles is None:
        return None

    want_wrap = random.random() < 0.5
    canvas_width = _choose_canvas_width(rectangles, want_wrap)
    if canvas_width is None:
        return None

    expected = _render_rectangles(rectangles, canvas_width)
    if max(expected.shape) > MAX_GRID_SIZE:
        return None
    if want_wrap and _wrap_count(rectangles, canvas_width) < 2:
        return None
    if not want_wrap and _wrap_count(rectangles, canvas_width) != 1:
        return None

    sprites: list[dict] = []
    for rect in rectangles:
        color = int(rect["color"])
        sprites.append({"color": color, "sprite": _sprite_grid(rect["piece_a"], color)})
        sprites.append({"color": color, "sprite": _sprite_grid(rect["piece_b"], color)})
    random.shuffle(sprites)

    sprite_area = sum(int(item["sprite"].shape[0] * item["sprite"].shape[1]) for item in sprites)
    tallest = max(int(item["sprite"].shape[0]) for item in sprites)
    min_height = max(tallest + 2, (sprite_area // max(1, canvas_width - 2)) + 6, expected.shape[0] + 2)
    if min_height > MAX_GRID_SIZE:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        height = random.randint(min_height, MAX_GRID_SIZE)
        inp = _place_sprites(height, canvas_width, sprites)
        if inp is None:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
