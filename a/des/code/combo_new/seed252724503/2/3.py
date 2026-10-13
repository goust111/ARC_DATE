from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))

SPRITE_COUNT_RANGE = (3, 4)
BASE_SIDE_RANGE = (5, 6)
EDGE_REMOVAL_RANGE = (1, 4)
PLAYFIELD_EXTRA_HEIGHT_RANGE = (2, 8)
HORIZONTAL_GAP_RANGE = (1, 3)
TIE_CASE_PROBABILITY = 0.5
MAX_HOLE_SLOTS = 5

BASE_SHAPE_ATTEMPTS = 200
SLOT_PACKING_ATTEMPTS = 24
COUNT_ATTEMPTS = 200
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _count_enclosed_background(mask_crop: np.ndarray) -> int:
    bg_mask = ~np.asarray(mask_crop, dtype=bool)
    h, w = map(int, bg_mask.shape)
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def seed(rr: int, cc: int) -> None:
        if bg_mask[rr, cc] and not outside[rr, cc]:
            outside[rr, cc] = True
            queue.append((int(rr), int(cc)))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bg_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((int(nr), int(nc)))

    return int(np.count_nonzero(bg_mask & ~outside))


def _sprite_infos(playfield: np.ndarray) -> list[dict[str, object]]:
    sprites: list[dict[str, object]] = []
    for coords in _connected_components(playfield != BG):
        colors = {int(playfield[r, c]) for r, c in coords}
        assert len(colors) == 1
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(coords)
        mask_crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            mask_crop[r - r0, c - c0] = True
        sprites.append(
            {
                "coords": coords,
                "color": int(color),
                "holes": _count_enclosed_background(mask_crop),
                "r0": int(r0),
                "c0": int(c0),
            }
        )
    return sprites


def _strip_colors(strip_row: np.ndarray) -> list[int]:
    cols = np.flatnonzero(strip_row != BG)
    assert len(cols) >= 2
    assert bool(np.all(np.diff(cols) == 1))
    colors = [int(strip_row[c]) for c in cols]
    assert len(set(colors)) == len(colors)
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor same-outline sprites by sorting them with their enclosed-hole counts and the bottom strip.

    Input structure:
    - The background is black.
    - The last row contains one contiguous left-to-right strip of distinct non-black colors.
    - Above that strip are several non-touching monochrome sprites with the same outer silhouette but different internal black holes.

    Rule:
    - Remove the bottom strip row from the working canvas.
    - For each sprite, count the background pixels inside its bounding box that are enclosed by the sprite and unreachable from the bbox edge.
    - Sort sprites from fewer holes to more holes; break ties by leftmost column.
    - Recolor every pixel of the i-th sorted sprite with the i-th strip color, leaving holes black.

    Key insight:
    - The shared outline is a distraction: the ordering key is the number of enclosed background pixels, and the bottom strip supplies the replacement colors in sorted order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    strip_colors = _strip_colors(g[-1])
    playfield = g[:-1].copy()

    sprites = _sprite_infos(playfield)
    assert len(sprites) == len(strip_colors)
    sprites.sort(key=lambda sprite: (int(sprite["holes"]), int(sprite["c0"]), int(sprite["r0"])))

    out = playfield.copy()
    for sprite, new_color in zip(sprites, strip_colors):
        for r, c in sprite["coords"]:
            out[r, c] = int(new_color)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_connected_components(mask)) == 1


def _border_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                cells.append((int(r), int(c)))
    return cells


def _hole_slot_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = map(int, mask.shape)
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    if not mask[r + dr, c + dc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((int(r), int(c)))
    return candidates


def _pack_hole_slots(candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    for _ in range(SLOT_PACKING_ATTEMPTS):
        order = candidates[:]
        random.shuffle(order)
        slots: list[tuple[int, int]] = []
        for r, c in order:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in slots):
                slots.append((int(r), int(c)))
        if len(slots) > len(best):
            best = slots
    return best


def _randomly_transform(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = np.fliplr(out)
    out = np.rot90(out, random.randrange(4))
    return np.asarray(out, dtype=bool)


def _make_base_shape(min_slots: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(BASE_SHAPE_ATTEMPTS):
        height = random.randint(*BASE_SIDE_RANGE)
        width = random.randint(*BASE_SIDE_RANGE)
        mask = np.ones((height, width), dtype=bool)

        border = _border_cells(mask)
        random.shuffle(border)
        max_removals = min(EDGE_REMOVAL_RANGE[1], len(border) - 1)
        target_removals = random.randint(EDGE_REMOVAL_RANGE[0], max_removals)
        removed = 0
        for r, c in border:
            if removed >= target_removals:
                break
            trial = mask.copy()
            trial[r, c] = False
            if not bool(np.all(np.any(trial, axis=0))) or not bool(np.all(np.any(trial, axis=1))):
                continue
            if not _is_connected(trial):
                continue
            mask = trial
            removed += 1
        if removed == 0:
            continue

        mask = _randomly_transform(mask)
        slots = _pack_hole_slots(_hole_slot_candidates(mask))
        if len(slots) < min_slots:
            continue
        slot_count = random.randint(min_slots, min(MAX_HOLE_SLOTS, len(slots)))
        return mask, random.sample(slots, slot_count)

    raise RuntimeError("failed to build a valid base shape")


def _sample_counts(sprite_count: int, max_holes: int) -> list[int]:
    values = list(range(max_holes + 1))
    use_tie_case = random.random() < TIE_CASE_PROBABILITY

    for _ in range(COUNT_ATTEMPTS):
        if use_tie_case:
            counts = [random.choice(values) for _ in range(sprite_count)]
            distinct = len(set(counts))
            if distinct in (1, sprite_count):
                continue
        else:
            counts = random.sample(values, sprite_count)
        if counts == sorted(counts):
            continue
        return [int(v) for v in counts]

    if use_tie_case:
        fallback = [0, 1, 0, 2][:sprite_count]
    else:
        fallback = list(range(sprite_count - 1, -1, -1))
    assert max(fallback) <= max_holes
    return [int(v) for v in fallback]


def _sample_gaps(count: int, capacity: int) -> list[int]:
    if count == 0:
        return []
    min_total = count * HORIZONTAL_GAP_RANGE[0]
    max_total = min(count * HORIZONTAL_GAP_RANGE[1], capacity)
    total = random.randint(min_total, max_total)
    gaps = [HORIZONTAL_GAP_RANGE[0]] * count
    remaining = total - min_total
    while remaining > 0:
        choices = [i for i, gap in enumerate(gaps) if gap < HORIZONTAL_GAP_RANGE[1]]
        idx = random.choice(choices)
        gaps[idx] += 1
        remaining -= 1
    return gaps


def _horizontal_layout(sprite_width: int, sprite_count: int) -> tuple[int, list[int]] | None:
    gap_capacity = 30 - sprite_count * sprite_width
    if gap_capacity < (sprite_count - 1) * HORIZONTAL_GAP_RANGE[0]:
        return None

    gaps = _sample_gaps(sprite_count - 1, gap_capacity)
    span = sprite_count * sprite_width + sum(gaps)
    grid_width = random.randint(span, 30)
    left_margin = random.randint(0, grid_width - span)

    cols = [left_margin]
    current = left_margin
    for gap in gaps:
        current += sprite_width + gap
        cols.append(current)
    return int(grid_width), [int(c) for c in cols]


def _sample_rows(playfield_height: int, sprite_height: int, sprite_count: int) -> list[int]:
    row_limit = playfield_height - sprite_height - 1
    assert row_limit >= 0

    if row_limit == 0 or random.choice(["aligned", "staggered"]) == "aligned":
        row = random.randint(0, row_limit)
        return [int(row)] * sprite_count

    rows = [random.randint(0, row_limit) for _ in range(sprite_count)]
    if len(set(rows)) == 1:
        idx = random.randrange(sprite_count)
        choices = [r for r in range(row_limit + 1) if r != rows[idx]]
        rows[idx] = random.choice(choices)
    return [int(r) for r in rows]


def _paint_sprite(
    grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(mask):
        r = int(top + int(rr))
        c = int(left + int(cc))
        grid[r, c] = int(color)
        coords.append((r, c))
    return coords


def _expected_output(
    playfield: np.ndarray, sprite_infos: list[dict[str, object]], strip_colors: list[int]
) -> np.ndarray:
    out = playfield.copy()
    ordered = sorted(sprite_infos, key=lambda sprite: (int(sprite["holes"]), int(sprite["c0"]), int(sprite["r0"])))
    for sprite, new_color in zip(ordered, strip_colors):
        for r, c in sprite["coords"]:
            out[r, c] = int(new_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    base_mask, hole_order = _make_base_shape(min_slots=sprite_count - 1)
    sprite_height, sprite_width = map(int, base_mask.shape)

    layout = _horizontal_layout(sprite_width, sprite_count)
    if layout is None:
        return None
    grid_width, cols = layout

    playfield_height = random.randint(
        sprite_height + PLAYFIELD_EXTRA_HEIGHT_RANGE[0],
        min(29, sprite_height + PLAYFIELD_EXTRA_HEIGHT_RANGE[1]),
    )
    rows = _sample_rows(playfield_height, sprite_height, sprite_count)

    counts = _sample_counts(sprite_count, len(hole_order))
    hole_sets = {count: set(hole_order[:count]) for count in set(counts)}

    playfield = np.zeros((playfield_height, grid_width), dtype=int)
    sprite_colors = random.sample(COLOR_POOL, sprite_count)
    strip_colors = random.sample(COLOR_POOL, sprite_count)
    strip_start = random.randint(0, grid_width - sprite_count)

    sprite_infos: list[dict[str, object]] = []
    for color, holes, top, left in zip(sprite_colors, counts, rows, cols):
        sprite_mask = base_mask.copy()
        for rr, cc in hole_sets[holes]:
            sprite_mask[rr, cc] = False
        coords = _paint_sprite(playfield, sprite_mask, color, top, left)
        sprite_infos.append(
            {
                "coords": coords,
                "color": int(color),
                "holes": int(holes),
                "r0": int(top),
                "c0": int(left),
            }
        )

    expected = _expected_output(playfield, sprite_infos, strip_colors)
    ordered = sorted(sprite_infos, key=lambda sprite: (int(sprite["holes"]), int(sprite["c0"]), int(sprite["r0"])))
    if all(int(color) == int(sprite["color"]) for color, sprite in zip(strip_colors, ordered)):
        return None

    grid = np.vstack([playfield, np.zeros((1, grid_width), dtype=int)])
    grid[-1, strip_start : strip_start + sprite_count] = np.array(strip_colors, dtype=int)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
