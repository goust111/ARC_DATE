import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "right", "bottom", "left")

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
RECT_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 8)
OUTER_MARGIN = 1
RECT_GAP = 2
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _dominant_non_black_color(grid: np.ndarray) -> int | None:
    values = grid[grid != BLACK]
    if values.size == 0:
        return None
    counts = Counter(int(v) for v in values.flat)
    return counts.most_common(1)[0][0]


def _boundary_cycle(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cycle: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cycle.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        cycle.append((r, c1))
    for c in range(c1 - 1, c0 - 1, -1):
        cycle.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        cycle.append((r, c0))
    return cycle


def _extract_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frame_color = _dominant_non_black_color(grid)
    if frame_color is None:
        return []

    frames: list[dict[str, object]] = []
    for coords in _components4(grid == frame_color):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        boundary = set(_boundary_cycle(r0, c0, r1, c1))
        if set(coords) != boundary:
            continue

        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        filled = np.argwhere(inner != BLACK)
        if filled.shape[0] != 1:
            continue

        seed_r_rel, seed_c_rel = map(int, filled[0])
        seed_color = int(inner[seed_r_rel, seed_c_rel])
        if seed_color == frame_color:
            continue

        seed_r = r0 + 1 + seed_r_rel
        seed_c = c0 + 1 + seed_c_rel
        touching: list[str] = []
        if seed_r == r0 + 1:
            touching.append("top")
        if seed_r == r1 - 1:
            touching.append("bottom")
        if seed_c == c0 + 1:
            touching.append("left")
        if seed_c == c1 - 1:
            touching.append("right")
        if len(touching) != 1:
            continue

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (seed_r, seed_c),
                "seed_color": seed_color,
                "side": touching[0],
                "frame_color": frame_color,
            }
        )

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a clockwise ribbon on each rectangular frame using the enclosed seed.

    The start point is the frame pixel directly opposite the seed on the wall
    it touches, and the ribbon length is the perimeter of the interior space.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _extract_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        seed_r, seed_c = frame["seed"]
        seed_color = int(frame["seed_color"])
        side = frame["side"]

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ribbon_len = max(0, 2 * (inner_h + inner_w) - 4)

        if side == "top":
            start = (r0, seed_c)
        elif side == "right":
            start = (seed_r, c1)
        elif side == "bottom":
            start = (r1, seed_c)
        else:
            start = (seed_r, c0)

        cycle = _boundary_cycle(r0, c0, r1, c1)
        start_idx = cycle.index(start)
        for step in range(ribbon_len):
            rr, cc = cycle[(start_idx + step) % len(cycle)]
            out[rr, cc] = seed_color
        out[seed_r, seed_c] = BLACK

    return out


def _side_sequence(rng: random.Random, count: int) -> list[str]:
    sequence: list[str] = []
    while len(sequence) < count:
        block = list(SIDES)
        rng.shuffle(block)
        sequence.extend(block)
    return sequence[:count]


def _well_separated(
    boxes: list[tuple[int, int, int, int]],
    candidate: tuple[int, int, int, int],
    gap: int,
) -> bool:
    r0, c0, r1, c1 = candidate
    for er0, ec0, er1, ec1 in boxes:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _place_seed(
    rng: random.Random,
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        seed_r = r0 + 1
        seed_c = rng.randint(c0 + 2, c1 - 2)
    elif side == "right":
        seed_r = rng.randint(r0 + 2, r1 - 2)
        seed_c = c1 - 1
    elif side == "bottom":
        seed_r = r1 - 1
        seed_c = rng.randint(c0 + 2, c1 - 2)
    else:
        seed_r = rng.randint(r0 + 2, r1 - 2)
        seed_c = c0 + 1
    grid[seed_r, seed_c] = color


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rect_count = rng.choice(RECT_COUNT_CHOICES)
    min_side = max(GRID_H_RANGE[0], 10 + 4 * (rect_count - 1))
    h = rng.randint(min_side, GRID_H_RANGE[1])
    w = rng.randint(max(GRID_W_RANGE[0], min_side), GRID_W_RANGE[1])
    grid = np.zeros((h, w), dtype=int)

    seed_colors = rng.sample(SEED_COLORS, rect_count)
    sides = _side_sequence(rng, rect_count)
    boxes: list[tuple[int, int, int, int]] = []

    for _idx in range(rect_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            inner_h = rng.randint(*INNER_H_RANGE)
            inner_w = rng.randint(*INNER_W_RANGE)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 * OUTER_MARGIN > h or outer_w + 2 * OUTER_MARGIN > w:
                continue

            r0 = rng.randint(OUTER_MARGIN, h - OUTER_MARGIN - outer_h)
            c0 = rng.randint(OUTER_MARGIN, w - OUTER_MARGIN - outer_w)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _well_separated(boxes, bbox, RECT_GAP):
                continue

            boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(boxes)
    for bbox, side, color in zip(boxes, sides, seed_colors):
        _draw_frame(grid, bbox)
        _place_seed(rng, grid, bbox, side, color)

    frames = _extract_frames(grid)
    if len(frames) != rect_count:
        return None
    if len({tuple(frame["bbox"]) for frame in frames}) != rect_count:
        return None
    if sorted(int(frame["seed_color"]) for frame in frames) != sorted(seed_colors):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
