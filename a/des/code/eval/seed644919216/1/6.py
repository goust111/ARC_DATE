import random

import numpy as np


DIR_TO_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_ATTEMPTS = 400
PLACEMENT_TRIES = 250
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
FRAME_COUNT_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_WEIGHTS = (2, 3, 2, 1)
FRAME_H_RANGE = (5, 9)
FRAME_W_RANGE = (5, 9)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_frame(
    grid: np.ndarray, comp: list[tuple[int, int]], frame_color: int, bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None

    comp_set = set(comp)
    perimeter = _perimeter_cells(r0, c0, r1, c1)
    if len(comp_set) != len(perimeter) - 1:
        return None
    if any(cell not in perimeter for cell in comp_set):
        return None

    missing = list(perimeter - comp_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if int(grid[gap]) != bg:
        return None

    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        direction = "up"
    elif gr == r1 and c0 < gc < c1:
        direction = "down"
    elif gc == c0 and r0 < gr < r1:
        direction = "left"
    elif gc == c1 and r0 < gr < r1:
        direction = "right"
    else:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_non_bg = np.argwhere(inner != bg)
    if inner_non_bg.shape[0] != 1:
        return None

    sr, sc = map(int, inner_non_bg[0])
    sr += r0 + 1
    sc += c0 + 1
    seed_color = int(grid[sr, sc])
    if seed_color == frame_color:
        return None

    dr, dc = DIR_TO_VEC[direction]
    if dr != 0 and sc != gc:
        return None
    if dc != 0 and sr != gr:
        return None

    if direction == "up" and sr <= r0 + 1:
        return None
    if direction == "down" and sr >= r1 - 1:
        return None
    if direction == "left" and sc <= c0 + 1:
        return None
    if direction == "right" and sc >= c1 - 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "seed": (sr, sc),
        "seed_color": seed_color,
        "frame_color": frame_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each near-closed frame from its seed toward the side-gap, then extend
    outward from the gap to the grid edge using the frame color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    out = g.copy()

    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(g == color):
            frame = _parse_frame(g, comp, color, bg)
            if frame is not None:
                frames.append(frame)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gr, gc = frame["gap"]
        sr, sc = frame["seed"]
        seed_color = int(frame["seed_color"])
        frame_color = int(frame["frame_color"])
        dr, dc = DIR_TO_VEC[str(frame["direction"])]

        rr, cc = sr + dr, sc + dc
        while r0 < rr < r1 and c0 < cc < c1:
            out[rr, cc] = seed_color
            rr += dr
            cc += dc

        rr, cc = gr, gc
        while 0 <= rr < h and 0 <= cc < w:
            out[rr, cc] = frame_color
            rr += dr
            cc += dc

    return out


def _sample_directions(count: int, rng: random.Random) -> list[str]:
    dirs = list(DIR_TO_VEC)
    rng.shuffle(dirs)
    return dirs[:count]


def _ray_to_edge(
    gap: tuple[int, int], direction: str, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_TO_VEC[direction]
    r, c = gap
    cells = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _expanded_bbox_cells(
    r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            cells.add((r, c))
    return cells


def _place_frame(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    direction: str,
    frame_color: int,
    seed_color: int,
    rng: random.Random,
) -> dict[str, object] | None:
    min_h, max_h = FRAME_H_RANGE
    min_w, max_w = FRAME_W_RANGE
    max_h = min(max_h, h - 2)
    max_w = min(max_w, w - 2)
    if max_h < min_h or max_w < min_w:
        return None

    for _ in range(PLACEMENT_TRIES):
        fh = rng.randint(min_h, max_h)
        fw = rng.randint(min_w, max_w)
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        r1 = r0 + fh - 1
        c1 = c0 + fw - 1

        bbox_cells = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
        }
        if any(cell in blocked for cell in bbox_cells):
            continue

        if direction in ("up", "down"):
            gap_c = rng.randint(c0 + 1, c1 - 1)
            gap = (r0, gap_c) if direction == "up" else (r1, gap_c)
            if direction == "up":
                seed_r = rng.randint(r0 + 2, r1 - 1)
            else:
                seed_r = rng.randint(r0 + 1, r1 - 2)
            seed = (seed_r, gap_c)
        else:
            gap_r = rng.randint(r0 + 1, r1 - 1)
            gap = (gap_r, c0) if direction == "left" else (gap_r, c1)
            if direction == "left":
                seed_c = rng.randint(c0 + 2, c1 - 1)
            else:
                seed_c = rng.randint(c0 + 1, c1 - 2)
            seed = (gap_r, seed_c)

        ray = _ray_to_edge(gap, direction, (h, w))
        if any(cell in blocked for cell in ray):
            continue

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "seed": seed,
            "frame_color": frame_color,
            "seed_color": seed_color,
            "blockers": _expanded_bbox_cells(r0, c0, r1, c1, (h, w)),
            "ray": ray,
        }
    return None


def _draw_frame(grid: np.ndarray, frame: dict[str, object], bg: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    gr, gc = frame["gap"]
    sr, sc = frame["seed"]
    frame_color = int(frame["frame_color"])
    seed_color = int(frame["seed_color"])

    for c in range(c0, c1 + 1):
        grid[r0, c] = frame_color
        grid[r1, c] = frame_color
    for r in range(r0, r1 + 1):
        grid[r, c0] = frame_color
        grid[r, c1] = frame_color
    grid[gr, gc] = bg
    grid[sr, sc] = seed_color


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        h = rng.randint(*GRID_H_RANGE)
        w = rng.randint(*GRID_W_RANGE)
        bg = rng.randint(0, 9)
        grid = np.full((h, w), bg, dtype=int)

        frame_count = rng.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
        directions = _sample_directions(frame_count, rng)

        palette = [c for c in range(10) if c != bg]
        rng.shuffle(palette)
        border_pool_size = rng.randint(1, min(3, frame_count))
        border_pool = palette[:border_pool_size]
        seed_colors = palette[border_pool_size : border_pool_size + frame_count]
        if len(seed_colors) < frame_count:
            continue

        blocked: set[tuple[int, int]] = set()
        frames: list[dict[str, object]] = []
        ok = True
        for i, direction in enumerate(directions):
            frame = _place_frame(
                h=h,
                w=w,
                blocked=blocked,
                direction=direction,
                frame_color=rng.choice(border_pool),
                seed_color=seed_colors[i],
                rng=rng,
            )
            if frame is None:
                ok = False
                break
            frames.append(frame)
            blocked.update(frame["blockers"])
            blocked.update(frame["ray"])

        if not ok:
            continue

        for frame in frames:
            _draw_frame(grid, frame, bg)

        parsed = []
        for color in np.unique(grid):
            color = int(color)
            if color == bg:
                continue
            for comp in _components(grid == color):
                candidate = _parse_frame(grid, comp, color, bg)
                if candidate is not None:
                    parsed.append(candidate)
        if len(parsed) != frame_count:
            continue

        return grid

    raise RuntimeError("generate_input: exhausted retries")
