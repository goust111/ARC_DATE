import itertools
import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
YELLOW = 4
TILE = 3
GEN_ATTEMPTS = 400


def _checkerboard(shape: tuple[int, int], tile: int = TILE) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = BLUE if ((r // tile) + (c // tile)) % 2 == 0 else YELLOW
    return grid


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _infer_tile_size(grid: np.ndarray, bg_colors: set[int]) -> int:
    h, w = grid.shape
    run_lengths = []
    for r in range(h):
        if any(int(grid[r, c]) not in bg_colors for c in range(w)):
            continue
        run = 1
        for c in range(1, w):
            if int(grid[r, c]) == int(grid[r, c - 1]):
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
        break
    for c in range(w):
        if any(int(grid[r, c]) not in bg_colors for r in range(h)):
            continue
        run = 1
        for r in range(1, h):
            if int(grid[r, c]) == int(grid[r - 1, c]):
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
        break
    assert run_lengths, "could not infer checkerboard tile size"
    counts: dict[int, int] = {}
    for run in run_lengths:
        counts[run] = counts.get(run, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _find_reference_bg(grid: np.ndarray, bg_colors: set[int]) -> tuple[int, int, int]:
    h, w = grid.shape
    for s in range(h + w - 1):
        for r in range(min(s, h - 1), max(-1, s - w), -1):
            c = s - r
            if 0 <= c < w and int(grid[r, c]) in bg_colors:
                return r, c, int(grid[r, c])
    raise AssertionError("expected at least one visible background cell")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red object according to the checkerboard phase underneath it.

    Objects resting on blue tiles move right by their area.
    Objects resting on yellow tiles move down by their area.
    Parts shifted outside the canvas are clipped.
    """
    g = np.asarray(grid, dtype=int)
    values, counts = np.unique(g, return_counts=True)
    order = np.argsort(-counts)
    assert len(order) >= 3, "expected two background colors and one foreground color"
    bg_a = int(values[order[0]])
    bg_b = int(values[order[1]])
    fg_colors = [int(v) for v in values if int(v) not in {bg_a, bg_b}]
    assert len(fg_colors) == 1, "expected a single foreground color"
    fg = fg_colors[0]

    tile = _infer_tile_size(g, {bg_a, bg_b})
    ref_r, ref_c, ref_color = _find_reference_bg(g, {bg_a, bg_b})
    if ((ref_r // tile) + (ref_c // tile)) % 2 == 0:
        top_left = ref_color
        other = bg_b if ref_color == bg_a else bg_a
    else:
        other = ref_color
        top_left = bg_b if ref_color == bg_a else bg_a

    bg = np.empty_like(g)
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            bg[r, c] = top_left if ((r // tile) + (c // tile)) % 2 == 0 else other

    out = bg.copy()
    for comp in _find_components(g == fg):
        n = len(comp)
        under = {int(bg[r, c]) for r, c in comp}
        assert len(under) == 1, "each object must lie on a single checkerboard phase"
        phase = next(iter(under))
        dr, dc = (0, n) if phase == top_left else (n, 0)
        for r, c in comp:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = fg
    return out


def _shape_connected(cells: tuple[tuple[int, int], ...]) -> bool:
    cell_set = set(cells)
    queue = deque([cells[0]])
    seen = {cells[0]}
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return len(seen) == len(cells)


def _build_shape_library() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    cells = [(r, c) for r in range(TILE) for c in range(TILE)]
    library: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for size in range(2, 6):
        shapes = []
        for combo in itertools.combinations(cells, size):
            if _shape_connected(combo):
                shapes.append(combo)
        library[size] = shapes
    return library


SHAPES = _build_shape_library()


def _touches_existing(coords: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        if (r, c) in occupied:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in occupied:
                return True
    return False


def _object_target(coords: set[tuple[int, int]], area: int, phase: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = (0, area) if phase == BLUE else (area, 0)
    return {
        (r + dr, c + dc)
        for r, c in coords
        if 0 <= r + dr < h and 0 <= c + dc < w
    }


def _candidate_blocks(h: int, w: int, phase: int, clipped: bool) -> list[tuple[int, int]]:
    blocks = []
    bh = h // TILE
    bw = w // TILE
    for br in range(bh):
        for bc in range(bw):
            if (br + bc) % 2 == (0 if phase == BLUE else 1):
                row0 = br * TILE
                col0 = bc * TILE
                if clipped:
                    if phase == BLUE and col0 + TILE < w - 1:
                        continue
                    if phase == YELLOW and row0 + TILE < h - 1:
                        continue
                blocks.append((row0, col0))
    random.shuffle(blocks)
    return blocks


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 21)
    w = random.randint(12, 21)
    grid = _checkerboard((h, w))

    obj_count = random.randint(2, 5)
    phases = [BLUE, YELLOW]
    while len(phases) < obj_count:
        phases.append(random.choice((BLUE, YELLOW)))
    random.shuffle(phases)

    if BLUE not in phases or YELLOW not in phases:
        return None

    clip_flags = [False] * obj_count
    if random.random() < 0.5:
        clip_flags[random.randrange(obj_count)] = True

    occupied: set[tuple[int, int]] = set()
    targets: set[tuple[int, int]] = set()

    for phase, clipped in zip(phases, clip_flags):
        area = random.randint(2, 5)
        shape = set(random.choice(SHAPES[area]))
        placed = False
        blocks = _candidate_blocks(h, w, phase, clipped)
        for row0, col0 in blocks:
            coords = {(row0 + r, col0 + c) for r, c in shape}
            if _touches_existing(coords, occupied):
                continue
            if any(grid[r, c] != phase for r, c in coords):
                continue
            target = _object_target(coords, area, phase, (h, w))
            if not target:
                continue
            if target & targets:
                continue
            occupied.update(coords)
            targets.update(target)
            for r, c in coords:
                grid[r, c] = RED
            placed = True
            break
        if not placed:
            return None

    if len(_find_components(grid == RED)) != obj_count:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
