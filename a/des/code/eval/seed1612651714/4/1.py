from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_CHOICES = tuple(DIRS)

# --------------- solver helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _crop_component(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frame_boundary(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _seed_sort_key(seed: tuple[int, int, int], direction: str) -> tuple[int, int]:
    r, c, _ = seed
    if direction in ("left", "right"):
        return (r, c)
    return (c, r)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        coords = np.argwhere(grid == color)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 5 or w < 5:
            continue

        boundary = _frame_boundary(r0, c0, r1, c1)
        color_cells = {tuple(map(int, xy)) for xy in coords}
        if not color_cells <= boundary:
            continue

        missing = list(boundary - color_cells)
        if len(missing) != 1:
            continue
        gap_r, gap_c = map(int, missing[0])
        if gap_r == r0 and c0 < gap_c < c1:
            direction = "up"
        elif gap_r == r1 and c0 < gap_c < c1:
            direction = "down"
        elif gap_c == c0 and r0 < gap_r < r1:
            direction = "left"
        elif gap_c == c1 and r0 < gap_r < r1:
            direction = "right"
        else:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        seed_coords = np.argwhere((interior != bg) & (interior != color))
        if not (1 <= len(seed_coords) <= 3):
            continue

        seeds = []
        seed_colors = set()
        ok = True
        for rr, cc in seed_coords:
            abs_r = int(r0 + 1 + rr)
            abs_c = int(c0 + 1 + cc)
            seed_color = int(grid[abs_r, abs_c])
            if seed_color in seed_colors or np.count_nonzero(interior == seed_color) != 1:
                ok = False
                break
            seed_colors.add(seed_color)
            seeds.append((abs_r, abs_c, seed_color))
        if not ok:
            continue

        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "gap": (gap_r, gap_c),
                "direction": direction,
                "seeds": seeds,
            }
        )

    assert frames, "expected at least one valid frame"
    return frames


def _sprite_attachment(mask: np.ndarray, direction: str) -> tuple[int, int]:
    h, w = mask.shape
    pts = np.argwhere(mask)
    if direction == "right":
        edge = pts[pts[:, 1] == 0]
    elif direction == "left":
        edge = pts[pts[:, 1] == w - 1]
    elif direction == "down":
        edge = pts[pts[:, 0] == 0]
    else:
        edge = pts[pts[:, 0] == h - 1]
    assert len(edge) == 1, "sprite attachment edge must contain one anchor pixel"
    return int(edge[0, 0]), int(edge[0, 1])


def _extract_sprites(grid: np.ndarray, frames: list[dict]) -> dict[int, dict]:
    h, w = grid.shape
    inside_frame = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inside_frame[r0 : r1 + 1, c0 : c1 + 1] = True

    seed_colors = {seed_color for frame in frames for _, _, seed_color in frame["seeds"]}
    sprites: dict[int, dict] = {}
    for color in seed_colors:
        comps = _component_coords((grid == color) & (~inside_frame))
        assert len(comps) == 1, "each seed color must have exactly one outside sprite"
        coords = comps[0]
        r0, c0, r1, c1 = _bbox(coords)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        assert mask.shape[0] <= 3 and mask.shape[1] <= 3, "sprites must fit within 3x3"
        sprites[color] = {
            "coords": coords,
            "mask": mask,
        }
    return sprites


def _paint_sprite(
    out: np.ndarray,
    color: int,
    mask: np.ndarray,
    top: int,
    left: int,
    bg: int,
) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        r = int(top + rr)
        c = int(left + cc)
        assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
        assert int(out[r, c]) == bg
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow template-shaped colored branches from each frame gap.

    Input structure
    - The most common color is the background.
    - One or more hollow rectangular frames sit on the canvas; each frame has
      exactly one single-cell border gap.
    - Inside each frame are 1-3 colored seed pixels, and outside the frames are
      small monochrome sprites whose colors match the seeds.

    Rule
    - The border gap determines the growth direction.
    - For every seed, find the outside sprite with the same color and use its
      shape as a growth unit.
    - Sort a frame's seeds by the axis perpendicular to the growth direction.
    - Starting just outside the gap, place the matching growth units in that
      order, tightly packed along the growth axis.
    - Keep the frames and seeds, remove the original outside sprites, and keep
      only the newly grown branches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    sprites = _extract_sprites(g, frames)

    out = g.copy()
    for info in sprites.values():
        coords = info["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for frame in frames:
        gap_r, gap_c = frame["gap"]
        direction = frame["direction"]
        ordered = sorted(frame["seeds"], key=lambda seed: _seed_sort_key(seed, direction))

        if direction == "right":
            cursor = gap_c + 1
            base = gap_r
            for _, _, color in ordered:
                mask = sprites[color]["mask"]
                ar, ac = _sprite_attachment(mask, direction)
                _paint_sprite(out, color, mask, base - ar, cursor - ac, bg)
                cursor += mask.shape[1]
        elif direction == "left":
            cursor = gap_c - 1
            base = gap_r
            for _, _, color in ordered:
                mask = sprites[color]["mask"]
                ar, ac = _sprite_attachment(mask, direction)
                _paint_sprite(out, color, mask, base - ar, cursor - ac, bg)
                cursor -= mask.shape[1]
        elif direction == "down":
            cursor = gap_r + 1
            base = gap_c
            for _, _, color in ordered:
                mask = sprites[color]["mask"]
                ar, ac = _sprite_attachment(mask, direction)
                _paint_sprite(out, color, mask, cursor - ar, base - ac, bg)
                cursor += mask.shape[0]
        else:
            cursor = gap_r - 1
            base = gap_c
            for _, _, color in ordered:
                mask = sprites[color]["mask"]
                ar, ac = _sprite_attachment(mask, direction)
                _paint_sprite(out, color, mask, cursor - ar, base - ac, bg)
                cursor -= mask.shape[0]

    return out


# --------------- generator helpers ---------------

GRID_H_RANGE = (20, 29)
GRID_W_RANGE = (20, 29)
FRAME_COUNT_CHOICES = (1, 2)
INNER_SIZE_RANGE = (3, 5)
CANVAS_COLORS = tuple(range(10))
RETRIES = 400
PLACEMENT_TRIES = 250
SPRITE_PLACEMENT_TRIES = 250


def _bool_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def _valid_sprite(mask: np.ndarray, direction: str) -> bool:
    h, w = mask.shape
    if not mask.any():
        return False
    if not (mask[0, :].any() and mask[h - 1, :].any() and mask[:, 0].any() and mask[:, w - 1].any()):
        return False
    if not _bool_connected(mask):
        return False

    if direction == "right":
        rows = np.flatnonzero(mask[:, 0])
        if len(rows) != 1:
            return False
        return bool(mask[int(rows[0]), w - 1])
    if direction == "left":
        rows = np.flatnonzero(mask[:, w - 1])
        if len(rows) != 1:
            return False
        return bool(mask[int(rows[0]), 0])
    if direction == "down":
        cols = np.flatnonzero(mask[0, :])
        if len(cols) != 1:
            return False
        return bool(mask[h - 1, int(cols[0])])
    cols = np.flatnonzero(mask[h - 1, :])
    if len(cols) != 1:
        return False
    return bool(mask[0, int(cols[0])])


@lru_cache(maxsize=None)
def _sprite_library(direction: str) -> tuple[np.ndarray, ...]:
    masks: list[np.ndarray] = []
    for h in range(1, 4):
        for w in range(1, 4):
            for bits in range(1, 1 << (h * w)):
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(h * w):
                    if (bits >> idx) & 1:
                        mask[idx // w, idx % w] = True
                if _valid_sprite(mask, direction):
                    masks.append(mask)
    return tuple(masks)


def _random_sprite(direction: str) -> np.ndarray:
    return random.choice(_sprite_library(direction)).copy()


def _add_cell(cells: dict[tuple[int, int], int], r: int, c: int, color: int) -> None:
    key = (int(r), int(c))
    if key in cells:
        assert cells[key] == int(color)
    cells[key] = int(color)


def _build_frame_plan(direction: str, border_color: int, seed_colors: list[int]) -> dict:
    n = len(seed_colors)
    if direction in ("left", "right"):
        inner_h = random.randint(max(INNER_SIZE_RANGE[0], n), INNER_SIZE_RANGE[1])
        inner_w = random.randint(*INNER_SIZE_RANGE)
    else:
        inner_h = random.randint(*INNER_SIZE_RANGE)
        inner_w = random.randint(max(INNER_SIZE_RANGE[0], n), INNER_SIZE_RANGE[1])
    outer_h, outer_w = inner_h + 2, inner_w + 2

    if direction == "up":
        gap = (0, random.randint(1, outer_w - 2))
        orth_positions = sorted(random.sample(range(1, outer_w - 1), n))
        seed_positions = [(random.randint(1, outer_h - 2), col) for col in orth_positions]
    elif direction == "down":
        gap = (outer_h - 1, random.randint(1, outer_w - 2))
        orth_positions = sorted(random.sample(range(1, outer_w - 1), n))
        seed_positions = [(random.randint(1, outer_h - 2), col) for col in orth_positions]
    elif direction == "left":
        gap = (random.randint(1, outer_h - 2), 0)
        orth_positions = sorted(random.sample(range(1, outer_h - 1), n))
        seed_positions = [(row, random.randint(1, outer_w - 2)) for row in orth_positions]
    else:
        gap = (random.randint(1, outer_h - 2), outer_w - 1)
        orth_positions = sorted(random.sample(range(1, outer_h - 1), n))
        seed_positions = [(row, random.randint(1, outer_w - 2)) for row in orth_positions]

    shuffled_colors = seed_colors[:]
    random.shuffle(shuffled_colors)
    seeds = [(r, c, color) for (r, c), color in zip(seed_positions, shuffled_colors, strict=False)]
    sprites = {color: _random_sprite(direction) for color in seed_colors}

    input_cells: dict[tuple[int, int], int] = {}
    for rr in range(outer_h):
        for cc in range(outer_w):
            on_boundary = rr in (0, outer_h - 1) or cc in (0, outer_w - 1)
            if on_boundary and (rr, cc) != gap:
                _add_cell(input_cells, rr, cc, border_color)
    for rr, cc, color in seeds:
        _add_cell(input_cells, rr, cc, color)

    output_cells = dict(input_cells)
    ordered = sorted(seeds, key=lambda seed: _seed_sort_key(seed, direction))
    gap_r, gap_c = gap
    if direction == "right":
        cursor = gap_c + 1
        base = gap_r
        for _, _, color in ordered:
            mask = sprites[color]
            ar, ac = _sprite_attachment(mask, direction)
            top, left = base - ar, cursor - ac
            for rr, cc in np.argwhere(mask):
                _add_cell(output_cells, top + int(rr), left + int(cc), color)
            cursor += mask.shape[1]
    elif direction == "left":
        cursor = gap_c - 1
        base = gap_r
        for _, _, color in ordered:
            mask = sprites[color]
            ar, ac = _sprite_attachment(mask, direction)
            top, left = base - ar, cursor - ac
            for rr, cc in np.argwhere(mask):
                _add_cell(output_cells, top + int(rr), left + int(cc), color)
            cursor -= mask.shape[1]
    elif direction == "down":
        cursor = gap_r + 1
        base = gap_c
        for _, _, color in ordered:
            mask = sprites[color]
            ar, ac = _sprite_attachment(mask, direction)
            top, left = cursor - ar, base - ac
            for rr, cc in np.argwhere(mask):
                _add_cell(output_cells, top + int(rr), left + int(cc), color)
            cursor += mask.shape[0]
    else:
        cursor = gap_r - 1
        base = gap_c
        for _, _, color in ordered:
            mask = sprites[color]
            ar, ac = _sprite_attachment(mask, direction)
            top, left = cursor - ar, base - ac
            for rr, cc in np.argwhere(mask):
                _add_cell(output_cells, top + int(rr), left + int(cc), color)
            cursor -= mask.shape[0]

    occupied = set(output_cells)
    padded = set()
    for r, c in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                padded.add((r + dr, c + dc))

    return {
        "direction": direction,
        "gap": gap,
        "seeds": seeds,
        "sprites": sprites,
        "input_cells": input_cells,
        "output_cells": output_cells,
        "occupied": occupied,
        "padded": padded,
    }


def _place_structure(plan: dict, reserved: np.ndarray) -> tuple[int, int] | None:
    h, w = reserved.shape
    coords = np.array(list(plan["occupied"]), dtype=int)
    min_r, min_c = map(int, coords.min(axis=0))
    max_r, max_c = map(int, coords.max(axis=0))
    r_lo, r_hi = -min_r, h - 1 - max_r
    c_lo, c_hi = -min_c, w - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(PLACEMENT_TRIES):
        shift_r = random.randint(r_lo, r_hi)
        shift_c = random.randint(c_lo, c_hi)
        ok = True
        for rr, cc in plan["padded"]:
            ar = rr + shift_r
            ac = cc + shift_c
            if 0 <= ar < h and 0 <= ac < w and reserved[ar, ac]:
                ok = False
                break
        if not ok:
            continue
        for rr, cc in plan["padded"]:
            ar = rr + shift_r
            ac = cc + shift_c
            if 0 <= ar < h and 0 <= ac < w:
                reserved[ar, ac] = True
        return shift_r, shift_c
    return None


def _paint_cells(grid: np.ndarray, cells: dict[tuple[int, int], int], shift: tuple[int, int]) -> None:
    shift_r, shift_c = shift
    for (rr, cc), color in cells.items():
        grid[rr + shift_r, cc + shift_c] = color


def _place_sprite_instance(
    inp: np.ndarray,
    mask: np.ndarray,
    color: int,
    reserved: np.ndarray,
) -> bool:
    h, w = inp.shape
    sh, sw = mask.shape
    for _ in range(SPRITE_PLACEMENT_TRIES):
        top = random.randint(0, h - sh)
        left = random.randint(0, w - sw)
        ok = True
        for rr in range(top - 1, top + sh + 1):
            for cc in range(left - 1, left + sw + 1):
                if 0 <= rr < h and 0 <= cc < w and reserved[rr, cc]:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        coords = np.argwhere(mask)
        for rr, cc in coords:
            inp[top + int(rr), left + int(cc)] = color
        for rr in range(top - 1, top + sh + 1):
            for cc in range(left - 1, left + sw + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    reserved[rr, cc] = True
        return True
    return False


def _oracle_ok(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        got = main(inp)
    except Exception:
        return False
    return np.array_equal(got, expected)


def _sample_seed_counts(frame_count: int) -> list[int]:
    if frame_count == 1:
        return [random.randint(1, 3)]
    return [random.randint(1, 3), random.randint(1, 3)]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.choice(CANVAS_COLORS)
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    seed_counts = _sample_seed_counts(frame_count)
    total_seeds = sum(seed_counts)

    colors = [color for color in CANVAS_COLORS if color != bg]
    random.shuffle(colors)
    frame_colors = colors[:frame_count]
    seed_palette = colors[frame_count : frame_count + total_seeds]

    plans = []
    cursor = 0
    for frame_index, count in enumerate(seed_counts):
        seed_colors = seed_palette[cursor : cursor + count]
        cursor += count
        plans.append(_build_frame_plan(random.choice(DIR_CHOICES), frame_colors[frame_index], seed_colors))

    reserved = np.zeros((h, w), dtype=bool)
    placed_plans = []
    for plan in sorted(plans, key=lambda item: len(item["occupied"]), reverse=True):
        shift = _place_structure(plan, reserved)
        if shift is None:
            return None
        placed_plans.append((plan, shift))

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    sprite_reserved = reserved.copy()
    for plan, shift in placed_plans:
        _paint_cells(inp, plan["input_cells"], shift)
        _paint_cells(expected, plan["output_cells"], shift)

    sprite_items: list[tuple[int, np.ndarray]] = []
    for plan, _shift in placed_plans:
        for color, mask in plan["sprites"].items():
            sprite_items.append((color, mask))
    random.shuffle(sprite_items)
    for color, mask in sprite_items:
        if not _place_sprite_instance(inp, mask, color, sprite_reserved):
            return None

    if np.array_equal(inp, expected):
        return None
    if _get_bg(inp) != bg:
        return None
    if not _oracle_ok(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid sample")
