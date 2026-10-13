from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = tuple(range(10))

# Generator configuration.
GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
FRAME_H_RANGE = (3, 8)
FRAME_W_RANGE = (3, 8)
FRAME_COUNT_BY_AREA = (
    (220, 2),
    (320, 3),
)
FRAME_COUNT_MAX = 4
MAX_RAY_LENGTH = 8
PLACEMENT_ATTEMPTS = 250
PADDING = 1


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _same_color_components(grid: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    queue.append((nr, nc))
            components.append((color, coords))
    return components


def _border_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _gap_direction(r0: int, c0: int, r1: int, c1: int, gap: tuple[int, int]) -> tuple[int, int] | None:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return (-1, 0)
    if gr == r1 and c0 < gc < c1:
        return (1, 0)
    if gc == c0 and r0 < gr < r1:
        return (0, -1)
    if gc == c1 and r0 < gr < r1:
        return (0, 1)
    return None


def _detect_frames(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g) if bg is None else int(bg)
    frames: list[dict] = []

    for color, coords in _same_color_components(g, bg):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue

        coord_set = set(coords)
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in coord_set):
            continue

        border = _border_positions(r0, c0, r1, c1)
        if len(coord_set) != len(border) - 1:
            continue

        missing: list[tuple[int, int]] = []
        valid = True
        for pos in border:
            value = int(g[pos])
            if pos in coord_set:
                if value != color:
                    valid = False
                    break
            else:
                if value != bg:
                    valid = False
                    break
                missing.append(pos)
        if not valid or len(missing) != 1:
            continue

        gap = missing[0]
        direction = _gap_direction(r0, c0, r1, c1, gap)
        if direction is None:
            continue

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_values = [int(v) for v in interior.flatten() if int(v) != bg]
        if not seed_values:
            continue
        counts = Counter(seed_values)
        top_count = max(counts.values())
        mode_colors = [col for col, count in counts.items() if count == top_count]
        if len(mode_colors) != 1:
            continue

        frames.append(
            {
                "frame_color": int(color),
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "ray_length": len(seed_values),
                "ray_color": int(mode_colors[0]),
            }
        )

    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each single-gap rectangular frame as a cannon.

    - Detect one-pixel-thick monochrome rectangular frames with exactly one
      background gap on a non-corner border position.
    - Count the non-background seed pixels inside the frame.
    - Paint a ray from the gap outward, using the interior majority color and a
      length equal to the seed count.
    - Only original background cells may be overwritten.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    h, w = g.shape

    for frame in _detect_frames(g, bg):
        r, c = frame["gap"]
        dr, dc = frame["direction"]
        for _ in range(int(frame["ray_length"])):
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(g[r, c]) == bg:
                out[r, c] = int(frame["ray_color"])
            r += dr
            c += dc

    return out


def _sample_frame_count(rng: random.Random, area: int) -> int:
    max_frames = FRAME_COUNT_MAX
    for threshold, limit in FRAME_COUNT_BY_AREA:
        if area < threshold:
            max_frames = limit
            break
    return rng.randint(1, max_frames)


def _direction_specs(rng: random.Random, count: int) -> list[str]:
    dirs = ["up", "down", "left", "right"]
    if count <= 4 and rng.random() < 0.5:
        return rng.sample(dirs, count)
    return [rng.choice(dirs) for _ in range(count)]


def _candidate_seed_positions(height: int, width: int, parity: int) -> list[tuple[int, int]]:
    pts = []
    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (r + c) % 2 == parity:
                pts.append((r, c))
    return pts


def _random_composition(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _seed_color_assignment(
    rng: random.Random,
    n_seeds: int,
    bg: int,
    frame_color: int,
) -> list[int]:
    usable = [c for c in PALETTE if c not in (bg, frame_color)]
    max_colors = min(3, len(usable), n_seeds, (n_seeds + 1) // 2)
    color_count = rng.randint(1, max_colors)
    colors = rng.sample(usable, color_count)
    if color_count == 1:
        return [colors[0]] * n_seeds

    mode_count = rng.randint((n_seeds // 2) + 1, n_seeds - (color_count - 1))
    other_counts = _random_composition(rng, n_seeds - mode_count, color_count - 1)
    seeds = [colors[0]] * mode_count
    for color, count in zip(colors[1:], other_counts):
        seeds.extend([color] * count)
    rng.shuffle(seeds)
    return seeds


def _ray_cells(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    direction: str,
    gap_index: int,
    length: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if direction == "up":
        gap = (r0, c0 + gap_index)
        cells = [(r0 - step, gap[1]) for step in range(length)]
    elif direction == "down":
        gap = (r1, c0 + gap_index)
        cells = [(r1 + step, gap[1]) for step in range(length)]
    elif direction == "left":
        gap = (r0 + gap_index, c0)
        cells = [(gap[0], c0 - step) for step in range(length)]
    else:
        gap = (r0 + gap_index, c1)
        cells = [(gap[0], c1 + step) for step in range(length)]
    return gap, cells


def _footprint_cells(
    h: int,
    w: int,
    r0: int,
    c0: int,
    ray: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}
    cells.update(ray)
    return cells


def _padded_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int) -> set[tuple[int, int]]:
    padded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - pad, r + pad + 1):
            for nc in range(c - pad, c + pad + 1):
                if 0 <= nr < h and 0 <= nc < w:
                    padded.add((nr, nc))
    return padded


def _place_frame(
    rng: random.Random,
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    frame_color: int,
    direction: str,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACEMENT_ATTEMPTS):
        box_h = rng.randint(*FRAME_H_RANGE)
        box_w = rng.randint(*FRAME_W_RANGE)
        if box_h >= h - 2 or box_w >= w - 2:
            continue

        r0 = rng.randint(1, h - box_h - 1)
        c0 = rng.randint(1, w - box_w - 1)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        gap_limit = box_w - 2 if direction in ("up", "down") else box_h - 2
        if gap_limit < 1:
            continue
        gap_index = rng.randint(1, gap_limit)

        parity0 = _candidate_seed_positions(box_h, box_w, 0)
        parity1 = _candidate_seed_positions(box_h, box_w, 1)
        seed_cap = max(len(parity0), len(parity1))
        if seed_cap < 1:
            continue

        gap, ray_probe = _ray_cells(r0, c0, r1, c1, direction, gap_index, 1)
        if direction == "up":
            exterior_cap = gap[0] + 1
        elif direction == "down":
            exterior_cap = h - gap[0]
        elif direction == "left":
            exterior_cap = gap[1] + 1
        else:
            exterior_cap = w - gap[1]

        max_length = min(seed_cap, exterior_cap, MAX_RAY_LENGTH)
        if max_length < 1:
            continue
        ray_length = rng.randint(1, max_length)
        gap, ray_cells = _ray_cells(r0, c0, r1, c1, direction, gap_index, ray_length)

        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in ray_cells):
            continue

        footprint = _footprint_cells(box_h, box_w, r0, c0, ray_cells)
        padded = _padded_cells(footprint, h, w, PADDING)
        if any(bool(blocked[rr, cc]) for rr, cc in padded):
            continue

        valid_parities = [pts for pts in (parity0, parity1) if len(pts) >= ray_length]
        if not valid_parities:
            continue
        seed_positions = rng.sample(rng.choice(valid_parities), ray_length)
        seed_colors = _seed_color_assignment(rng, ray_length, bg, frame_color)

        border = _border_positions(r0, c0, r1, c1)
        for rr, cc in border:
            if (rr, cc) != gap:
                grid[rr, cc] = frame_color
        for (sr, sc), color in zip(seed_positions, seed_colors):
            grid[r0 + sr, c0 + sc] = color
        for rr, cc in padded:
            blocked[rr, cc] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(PALETTE)
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    frame_count = _sample_frame_count(rng, h * w)
    frame_colors = rng.sample([c for c in PALETTE if c != bg], frame_count)
    directions = _direction_specs(rng, frame_count)

    for frame_color, direction in zip(frame_colors, directions):
        if not _place_frame(rng, grid, blocked, bg, frame_color, direction):
            return None

    if get_bg(grid) != bg:
        return None

    detected = _detect_frames(grid, bg)
    if len(detected) != frame_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
