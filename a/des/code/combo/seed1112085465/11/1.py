import math
import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (16, 30)
GRID_W_RANGE = (16, 30)
CONTAINER_GAP = 2
MAX_GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 100


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(grid: np.ndarray, comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    pts = np.array(comp, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 3:
        return None

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((int(r0), int(c)))
        expected.add((int(r1), int(c)))
    for r in range(r0 + 1, r1):
        expected.add((int(r), int(c0)))
        expected.add((int(r), int(c1)))

    if set(comp) != expected:
        return None
    if bool((grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME).any()):
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        bbox = _frame_bbox(grid, comp)
        if bbox is not None:
            frames.append(bbox)
    frames.sort()
    return frames


def _ranked_colors(interior: np.ndarray) -> list[int]:
    counts: dict[int, int] = {}
    first_seen: dict[int, tuple[int, int]] = {}
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            color = int(interior[r, c])
            if color in (BG, FRAME):
                continue
            counts[color] = counts.get(color, 0) + 1
            if color not in first_seen:
                first_seen[color] = (r, c)

    ranked = sorted(
        counts,
        key=lambda color: (-counts[color], first_seen[color][0], first_seen[color][1]),
    )
    return ranked


def _paint_ranked_border(out: np.ndarray, bbox: tuple[int, int, int, int], ranked: list[int]) -> None:
    r0, c0, r1, c1 = bbox

    # Split the perimeter into one clockwise traversal: top row, right edge,
    # bottom edge, then left edge. This assigns each corner to exactly one side.
    out[r0, c0 : c1 + 1] = FRAME
    out[r0 + 1 : r1 + 1, c1] = FRAME
    out[r1, c0:c1] = FRAME
    out[r0 + 1 : r1, c0] = FRAME

    if len(ranked) >= 1:
        out[r0, c0 : c1 + 1] = ranked[0]
    if len(ranked) >= 2:
        out[r0 + 1 : r1 + 1, c1] = ranked[1]
    if len(ranked) >= 3:
        out[r1, c0:c1] = ranked[2]
    if len(ranked) >= 4:
        out[r0 + 1 : r1, c0] = ranked[3]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each grey square frame using the frequency ranking of the colors inside it.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for bbox in _find_frames(g):
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ranked = _ranked_colors(interior)
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        _paint_ranked_border(out, bbox, ranked)

    return out


def _sample_container_count(rng: random.Random, h: int, w: int) -> int:
    min_side = min(h, w)
    if min_side < 18:
        return rng.choice([1, 2])
    if min_side < 23:
        return rng.choice([1, 2, 3])
    return rng.choice([2, 3, 4])


def _sample_container_spec(rng: random.Random, h: int, w: int) -> dict[str, object] | None:
    n_colors = rng.choice([2, 3, 4])
    counts = rng.sample([1, 2, 3, 4], n_colors)
    colors = rng.sample(PALETTE, n_colors)
    total_markers = sum(counts)
    min_inner_side = max(3, int(math.ceil(math.sqrt(total_markers))))
    min_side = max(5, min_inner_side + 2)
    max_side = min(10, h - 2, w - 2)
    if max_side < min_side:
        return None
    side = rng.randint(min_side, max_side)
    return {"side": side, "colors": colors, "counts": counts}


def _can_place(used: np.ndarray, r0: int, c0: int, side: int, gap: int) -> bool:
    h, w = used.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r0 + side + gap)
    cc1 = min(w, c0 + side + gap)
    return not bool(used[rr0:rr1, cc0:cc1].any())


def _mark_used(used: np.ndarray, r0: int, c0: int, side: int, gap: int) -> None:
    h, w = used.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h, r0 + side + gap)
    cc1 = min(w, c0 + side + gap)
    used[rr0:rr1, cc0:cc1] = True


def _draw_container(
    grid: np.ndarray,
    rng: random.Random,
    r0: int,
    c0: int,
    side: int,
    colors: list[int],
    counts: list[int],
) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME

    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    picks = rng.sample(cells, sum(counts))
    cursor = 0
    for color, count in zip(colors, counts):
        for r, c in picks[cursor : cursor + count]:
            grid[r, c] = color
        cursor += count


def _container_counts(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    counts: dict[int, int] = {}
    for color in interior.ravel():
        value = int(color)
        if value in (BG, FRAME):
            continue
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.values(), reverse=True)


def _valid_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    frames = _find_frames(grid)
    if len(frames) != expected_frames:
        return False
    for bbox in frames:
        counts = _container_counts(grid, bbox)
        if not (2 <= len(counts) <= 4):
            return False
        if len(set(counts)) != len(counts):
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    target_frames = _sample_container_count(rng, h, w)
    specs: list[dict[str, object]] = []
    for _ in range(target_frames):
        spec = _sample_container_spec(rng, h, w)
        if spec is None:
            return None
        specs.append(spec)
    specs.sort(key=lambda spec: int(spec["side"]), reverse=True)

    placements: list[tuple[int, int, dict[str, object]]] = []
    for spec in specs:
        side = int(spec["side"])
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, h - side - 1)
            c0 = rng.randint(1, w - side - 1)
            if not _can_place(used, r0, c0, side, CONTAINER_GAP):
                continue
            _mark_used(used, r0, c0, side, CONTAINER_GAP)
            placements.append((r0, c0, spec))
            placed = True
            break
        if not placed:
            return None

    rng.shuffle(placements)
    for r0, c0, spec in placements:
        _draw_container(
            grid,
            rng,
            r0,
            c0,
            int(spec["side"]),
            list(spec["colors"]),
            list(spec["counts"]),
        )

    if not _valid_generated_grid(grid, len(placements)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated retries")
