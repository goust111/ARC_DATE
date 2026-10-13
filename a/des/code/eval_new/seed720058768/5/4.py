import random
from typing import Iterable

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
PLACE_RETRIES = 120
GRID_H_RANGE = (14, 30)
GRID_W_RANGE = (14, 30)
RECT_COUNT_WEIGHTS = (1, 2, 2, 3)
RECT_H_RANGE = (7, 12)
RECT_W_RANGE = (7, 12)
SEGMENT_COUNT_RANGE = (2, 4)
PALETTE_SIZE_RANGE = (2, 3)
FRAME_MIN_SIZE = 4


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_frame_component(
    grid: np.ndarray, comp: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int] | None, list[tuple[int, int]] | None]:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < FRAME_MIN_SIZE or w < FRAME_MIN_SIZE:
        return None, None
    pts = set(comp)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border and (rr, cc) not in pts:
                return None, None
    interior = [(rr, cc) for rr, cc in comp if r0 < rr < r1 and c0 < cc < c1]
    if len(interior) == 0:
        if len(comp) != 2 * (h + w) - 4:
            return None, None
        return (r0, c0, r1, c1), None
    if len(interior) != 2:
        return None, None
    if len(comp) != 2 * (h + w) - 2:
        return None, None
    if not _is_pointer_component(grid, interior):
        return None, None
    return (r0, c0, r1, c1), sorted(interior)


def _is_pointer_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 2:
        return False
    (r0, c0), (r1, c1) = sorted(comp)
    if abs(r0 - r1) + abs(c0 - c1) != 1:
        return False
    color = int(grid[r0, c0])
    return color != BG and int(grid[r1, c1]) == color


def _pointer_direction(pointer: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str:
    (r0, c0), (r1, c1) = sorted(pointer)
    fr0, fc0, fr1, fc1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    frame_center_r = (fr0 + fr1) / 2.0
    frame_center_c = (fc0 + fc1) / 2.0
    if r0 == r1:
        if center_c < frame_center_c:
            return "left"
        if center_c > frame_center_c:
            return "right"
        return "right"
    if center_r < frame_center_r:
        return "up"
    if center_r > frame_center_r:
        return "down"
    return "down"


def _extract_objects(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]]:
    comps = _components_mask(grid != BG)
    frames: list[tuple[tuple[int, int, int, int], list[tuple[int, int]] | None]] = []
    pointers: list[list[tuple[int, int]]] = []
    for comp in comps:
        bbox, embedded_pointer = _classify_frame_component(grid, comp)
        if bbox is not None:
            frames.append((bbox, embedded_pointer))
            continue
        if _is_pointer_component(grid, comp):
            pointers.append(sorted(comp))
            continue
        raise AssertionError("unexpected foreground component")

    assignments: list[tuple[tuple[int, int, int, int], list[tuple[int, int]]]] = []
    used = set()
    for bbox, embedded_pointer in sorted(frames, key=lambda item: item[0]):
        if embedded_pointer is not None:
            assignments.append((bbox, embedded_pointer))
            continue
        r0, c0, r1, c1 = bbox
        inside = [
            idx
            for idx, pointer in enumerate(pointers)
            if all(r0 < r < r1 and c0 < c < c1 for r, c in pointer)
        ]
        assert len(inside) == 1, "each frame must contain exactly one pointer"
        used.add(inside[0])
        assignments.append((bbox, pointers[inside[0]]))
    assert len(used) == len(pointers), "every pointer must belong to one frame"
    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each multicolored rectangular frame with stripes copied from the side indicated by its internal pointer.

    Input structure:
    - Black background.
    - Several one-pixel-thick rectangular border loops made of non-black colors; each side can contain multiple color segments.
    - Inside each frame there is exactly one two-pixel pointer of a single color.
    - A vertical pointer lies in the upper or lower half of its frame; a horizontal pointer lies in the left or right half.

    Rule:
    - Detect each rectangular border loop and the unique two-pixel pointer strictly inside it.
    - The pointer axis chooses opposite border sides: vertical means top/bottom, horizontal means left/right. The half containing the pointer chooses which side of that pair to sample.
    - Read the chosen side excluding the two corner pixels. Top/bottom are read left-to-right; left/right are read top-to-bottom.
    - Fill the whole frame interior with the sampled sequence repeated as stripes. Sampling from top/bottom repeats across columns in every row; sampling from left/right repeats across rows in every column. The pointer pixels are overwritten.

    Key insight:
    - The same internal two-pixel marker determines both which border sequence to copy and whether the repeated pattern runs row-wise or column-wise.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for bbox, pointer in _extract_objects(g):
        r0, c0, r1, c1 = bbox
        direction = _pointer_direction(pointer, bbox)
        if direction == "up":
            seq = list(map(int, g[r0, c0 + 1 : c1]))
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = seq[(cc - (c0 + 1)) % len(seq)]
        elif direction == "down":
            seq = list(map(int, g[r1, c0 + 1 : c1]))
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = seq[(cc - (c0 + 1)) % len(seq)]
        elif direction == "left":
            seq = list(map(int, g[r0 + 1 : r1, c0]))
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = seq[(rr - (r0 + 1)) % len(seq)]
        else:
            seq = list(map(int, g[r0 + 1 : r1, c1]))
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr, cc] = seq[(rr - (r0 + 1)) % len(seq)]
    return out


def _random_partition(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    bounds = [0] + cuts + [total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _edge_sequence(length: int, palette: list[int], rng: random.Random) -> list[int]:
    n_segments = rng.randint(SEGMENT_COUNT_RANGE[0], min(SEGMENT_COUNT_RANGE[1], length))
    seg_lengths = _random_partition(length, n_segments, rng)
    seg_colors = [rng.choice(palette)]
    while len(seg_colors) < n_segments:
        choices = [color for color in palette if color != seg_colors[-1]]
        seg_colors.append(rng.choice(choices))
    if len(set(seg_colors)) == 1:
        seg_colors[-1] = next(color for color in palette if color != seg_colors[-1])
    seq: list[int] = []
    for color, run in zip(seg_colors, seg_lengths):
        seq.extend([int(color)] * run)
    return seq


def _paint_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    palette: list[int],
    pointer_color: int,
    direction: str,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    top_seq = _edge_sequence(c1 - c0 - 1, palette, rng)
    bottom_seq = _edge_sequence(c1 - c0 - 1, palette, rng)
    left_seq = _edge_sequence(r1 - r0 - 1, palette, rng)
    right_seq = _edge_sequence(r1 - r0 - 1, palette, rng)

    grid[r0, c0] = top_seq[0]
    grid[r0, c0 + 1 : c1] = top_seq
    grid[r0, c1] = top_seq[-1]
    grid[r1, c0] = bottom_seq[0]
    grid[r1, c0 + 1 : c1] = bottom_seq
    grid[r1, c1] = bottom_seq[-1]
    grid[r0 + 1 : r1, c0] = left_seq
    grid[r0 + 1 : r1, c1] = right_seq

    candidates = _pointer_candidates(bbox)
    usable = [cells for cells in candidates if _pointer_direction(cells, bbox) == direction]
    assert usable, f"no pointer placement for {bbox} -> {direction}"
    pointer = rng.choice(usable)
    for rr, cc in pointer:
        grid[rr, cc] = int(pointer_color)


def _pointer_candidates(bbox: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    candidates: list[list[tuple[int, int]]] = []
    for rr in range(r0 + 2, r1 - 1):
        for cc in range(c0 + 2, c1 - 1):
            if rr + 1 <= r1 - 2:
                candidates.append([(rr, cc), (rr + 1, cc)])
            if cc + 1 <= c1 - 2:
                candidates.append([(rr, cc), (rr, cc + 1)])
    return candidates


def _expanded_free(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    return not occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    h, w = occupied.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_rects = rng.choice(RECT_COUNT_WEIGHTS)
    directions = [rng.choice(DIRECTIONS) for _ in range(n_rects)]
    for direction in directions:
        placed = False
        for _ in range(PLACE_RETRIES):
            rect_h = rng.randint(*RECT_H_RANGE)
            rect_w = rng.randint(*RECT_W_RANGE)
            if rect_h > h - 2 or rect_w > w - 2:
                continue
            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            if not _expanded_free(occupied, bbox):
                continue

            palette_size = rng.randint(*PALETTE_SIZE_RANGE)
            palette = rng.sample(COLORS, palette_size)
            pointer_color = rng.choice([color for color in COLORS if color not in palette])
            _paint_frame(grid, bbox, palette, pointer_color, direction, rng)
            _mark_occupied(occupied, bbox)
            placed = True
            break
        if not placed:
            return None

    objects = _extract_objects(grid)
    if len(objects) != n_rects:
        return None
    for bbox, pointer in objects:
        direction = _pointer_direction(pointer, bbox)
        if direction not in DIRECTIONS:
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is None:
            continue
        _ = main(grid)
        return grid
    raise RuntimeError("generate_input: failed to create a valid grid")
