from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r0 in range(height):
        for c0 in range(width):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            comp: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((int(r), int(c)))
                for nr, nc in _neighbors4(r, c):
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not seen[nr, nc]
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_exact_rect_frame(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    frame_h = r1 - r0 + 1
    frame_w = c1 - c0 + 1
    if frame_h < 3 or frame_w < 3:
        return False

    local = np.zeros((frame_h, frame_w), dtype=bool)
    for r, c in points:
        local[r - r0, c - c0] = True

    expected = np.zeros((frame_h, frame_w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _dominant_interior_color(interior: np.ndarray) -> int:
    counts = Counter(
        int(v) for v in interior.ravel() if int(v) not in (BLACK, GREY)
    )
    assert counts, "each frame interior must contain colored vote pixels"
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    assert len(winners) == 1, "each frame must have a unique vote winner"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular frame by the majority color inside it.

    Input structure:
    - Black background contains several non-overlapping one-cell-thick grey rectangular frames.
    - Each frame is away from the grid edge and encloses black cells with scattered non-black, non-grey color pixels.
    - Every frame interior contains at least two colors, and one color has a unique largest count.

    Rule:
    - For each grey frame, count the non-black, non-grey pixels strictly inside its rectangle.
    - Recolor every grey pixel of that frame to the uniquely most frequent interior color.
    - Inside the same frame, erase every non-black pixel whose color is not that winner; keep winner pixels unchanged.
    - Leave the outside black background and any pixels outside frames unchanged.

    Key insight:
    The neutral grey border is a container whose final color is decided by a vote among its contents.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    grey_components = _components(g == GREY)
    assert grey_components, "expected at least one grey frame"

    for points in grey_components:
        box = _bbox(points)
        assert _is_exact_rect_frame(points, box), "grey components must be rectangular frames"
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        winner = _dominant_interior_color(interior)

        for r, c in points:
            out[r, c] = winner

        interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
        erase = (interior_out != BLACK) & (interior_out != winner)
        interior_out[erase] = BLACK

    return out


# ---------------- generator ----------------

GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
FRAME_COUNT_RANGE = (2, 5)
FRAME_H_RANGE = (4, 10)
FRAME_W_RANGE = (4, 10)
PLACE_TRIES_PER_FRAME = 120
GENERATION_ATTEMPTS = 500


def _paint_frame(grid: np.ndarray, r0: int, c0: int, frame_h: int, frame_w: int) -> None:
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _sample_vote_plan(area: int) -> list[tuple[int, int]]:
    n_colors = random.randint(2, min(4, area - 1, len(PALETTE)))
    colors = random.sample(PALETTE, n_colors)
    dominant = colors[0]
    others = colors[1:]

    for _ in range(100):
        non_counts = [random.randint(1, min(3, max(1, area // 3))) for _ in others]
        dominant_count = max(non_counts) + random.randint(1, 3)
        total = dominant_count + sum(non_counts)
        if total <= area and total <= max(n_colors + 1, int(area * 0.75)):
            plan = [(dominant, dominant_count)]
            plan.extend((color, count) for color, count in zip(others, non_counts))
            random.shuffle(plan)
            return plan

    plan = [(dominant, 2)]
    plan.extend((color, 1) for color in others)
    random.shuffle(plan)
    return plan


def _scatter_votes(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_h: int,
    frame_w: int,
    plan: list[tuple[int, int]],
) -> int:
    positions = [
        (r, c)
        for r in range(r0 + 1, r0 + frame_h - 1)
        for c in range(c0 + 1, c0 + frame_w - 1)
    ]
    random.shuffle(positions)

    colors: list[int] = []
    for color, count in plan:
        colors.extend([int(color)] * int(count))
    random.shuffle(colors)

    for (r, c), color in zip(positions, colors):
        grid[r, c] = int(color)

    counts = Counter()
    for color, count in plan:
        counts[int(color)] += int(count)
    best_count = max(counts.values())
    winners = [color for color, count in counts.items() if count == best_count]
    assert len(winners) == 1
    return int(winners[0])


def _expected_output(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    out = grid.copy()
    for r0, c0, frame_h, frame_w, winner in frames:
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        out[r0, c0 : c1 + 1] = winner
        out[r1, c0 : c1 + 1] = winner
        out[r0 : r1 + 1, c0] = winner
        out[r0 : r1 + 1, c1] = winner
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[(interior != BLACK) & (interior != winner)] = BLACK
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    max_frames = min(FRAME_COUNT_RANGE[1], max(FRAME_COUNT_RANGE[0], (height * width) // 55))
    n_frames = random.randint(FRAME_COUNT_RANGE[0], max_frames)

    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    frames: list[tuple[int, int, int, int, int]] = []

    for _ in range(n_frames):
        placed = False
        for _try in range(PLACE_TRIES_PER_FRAME):
            frame_h = random.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], height - 2))
            frame_w = random.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], width - 2))
            if (frame_h - 2) * (frame_w - 2) < 4:
                continue

            r0 = random.randint(1, height - frame_h - 1)
            c0 = random.randint(1, width - frame_w - 1)
            r1 = r0 + frame_h - 1
            c1 = c0 + frame_w - 1
            if blocked[r0 : r1 + 1, c0 : c1 + 1].any():
                continue

            _paint_frame(grid, r0, c0, frame_h, frame_w)
            plan = _sample_vote_plan((frame_h - 2) * (frame_w - 2))
            winner = _scatter_votes(grid, r0, c0, frame_h, frame_w, plan)

            hr0 = max(0, r0 - 1)
            hc0 = max(0, c0 - 1)
            hr1 = min(height, r1 + 2)
            hc1 = min(width, c1 + 2)
            blocked[hr0:hr1, hc0:hc1] = True
            frames.append((r0, c0, frame_h, frame_w, winner))
            placed = True
            break

        if not placed:
            return None

    expected = _expected_output(grid, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
