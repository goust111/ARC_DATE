import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 80
FRAME_COUNT_RANGE = (1, 4)
GRID_MIN_BY_FRAME_COUNT = {
    1: 10,
    2: 14,
    3: 18,
    4: 22,
}
SIZE_CHOICES_BY_FRAME_COUNT = {
    1: (5, 7, 9, 11),
    2: (5, 7, 9, 11),
    3: (5, 7, 9),
    4: (5, 7),
}
POINT_COUNT_RANGE = (1, 5)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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


def _is_rect_frame(mask: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if len(comp) != 2 * (h + w) - 4:
        return None
    for r, c in comp:
        if r not in (r0, r1) and c not in (c0, c1):
            return None
    if not np.all(mask[r0, c0 : c1 + 1]) or not np.all(mask[r1, c0 : c1 + 1]):
        return None
    if not np.all(mask[r0 : r1 + 1, c0]) or not np.all(mask[r0 : r1 + 1, c1]):
        return None
    if h > 2 and w > 2 and np.any(mask[r0 + 1 : r1, c0 + 1 : c1]):
        return None
    return (r0, c0, r1, c1)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_mask = grid == GREY
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grey_mask):
        bbox = _is_rect_frame(grey_mask, comp)
        if bbox is not None:
            frames.append(bbox)
    frames.sort()
    return frames


def _unique_winner(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in values.tolist())
    if not counts:
        raise ValueError("expected at least one vote pixel inside every frame")
    max_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == max_count]
    if len(winners) != 1:
        raise ValueError("expected a unique winning vote color")
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey frame by its winning interior vote color and draw a centered cross.

    Input structure:
    - Black background.
    - Several non-overlapping one-pixel-thick grey rectangular frames.
    - Each frame encloses a black interior containing a few isolated colored vote dots.
    - Within every frame, one dot color appears strictly more often than any other.

    Rule:
    - Detect each grey rectangular frame and inspect only the pixels strictly inside it.
    - Count the interior dot colors and pick the unique most frequent color.
    - Clear the entire frame interior back to black, removing all original vote dots.
    - Recolor the full border of the frame with the winning color.
    - Draw a cross in the frame interior with that same color: fill the interior middle row and the interior middle column.

    Key insight:
    - The scattered dots are votes rather than output content; each closed frame is redrawn from scratch using only the winning color, which controls both the recolored border and the centered cross.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        votes = interior[(interior != BLACK) & (interior != GREY)]
        winner = _unique_winner(votes)

        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        out[r0, c0 : c1 + 1] = winner
        out[r1, c0 : c1 + 1] = winner
        out[r0 : r1 + 1, c0] = winner
        out[r0 : r1 + 1, c1] = winner

        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        out[mid_r, c0 + 1 : c1] = winner
        out[r0 + 1 : r1, mid_c] = winner

    return out


def _compositions(total: int, parts: int) -> list[list[int]]:
    if parts == 1:
        return [[total]]
    out: list[list[int]] = []
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            out.append([first] + rest)
    return out


def _sample_vote_counts(rng: random.Random, total: int) -> list[int]:
    if total <= 2 or rng.random() < 0.45:
        return [total]

    winner_count = rng.randint(2, total - 1)
    remainder = total - winner_count
    candidates: list[list[int]] = []
    for num_losers in range(1, min(3, remainder) + 1):
        for parts in _compositions(remainder, num_losers):
            if max(parts) < winner_count:
                candidates.append(parts)
    loser_counts = rng.choice(candidates)
    counts = [winner_count] + loser_counts
    return counts


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _place_votes(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    n_points = rng.randint(POINT_COUNT_RANGE[0], min(POINT_COUNT_RANGE[1], len(cells)))
    counts = _sample_vote_counts(rng, n_points)
    colors = rng.sample(COLOR_POOL, len(counts))

    chosen_cells = rng.sample(cells, n_points)
    rng.shuffle(chosen_cells)

    idx = 0
    for color, count in zip(colors, counts):
        for _ in range(count):
            r, c = chosen_cells[idx]
            grid[r, c] = color
            idx += 1


def _try_place_frames(
    h: int,
    w: int,
    frame_count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    size_choices = [s for s in SIZE_CHOICES_BY_FRAME_COUNT[frame_count] if s <= h and s <= w]
    if not size_choices:
        return None

    specs = [(rng.choice(size_choices), rng.choice(size_choices)) for _ in range(frame_count)]
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)

    blocked = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for outer_h, outer_w in specs:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, h - outer_h)
            c0 = rng.randint(0, w - outer_w)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(h, r1 + 2)
            pc1 = min(w, c1 + 2)
            if blocked[pr0:pr1, pc0:pc1].any():
                continue
            blocked[pr0:pr1, pc0:pc1] = True
            boxes.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None
    return boxes


def _validate_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    frames = _find_frames(grid)
    if len(frames) != expected_frames:
        return False
    for r0, c0, r1, c1 in frames:
        if (r1 - r0) < 4 or (c1 - c0) < 4:
            return False
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        votes = interior[(interior != BLACK) & (interior != GREY)]
        if votes.size == 0:
            return False
        try:
            _unique_winner(votes)
        except ValueError:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    min_side = GRID_MIN_BY_FRAME_COUNT[frame_count]
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)

    boxes = _try_place_frames(h, w, frame_count, rng)
    if boxes is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for box in boxes:
        _paint_frame(grid, box)
        _place_votes(grid, box, rng)

    if not _validate_generated_grid(grid, frame_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
