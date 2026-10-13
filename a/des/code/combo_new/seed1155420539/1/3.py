from __future__ import annotations

import random

import numpy as np


BG = 5
BLACK = 0
RED = 2

DIRS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

ARROW_PATTERNS: dict[str, tuple[tuple[int, int], ...]] = {
    "up": ((0, 0), (1, -1), (1, 1)),
    "down": ((0, 0), (-1, -1), (-1, 1)),
    "left": ((0, 0), (-1, 1), (1, 1)),
    "right": ((0, 0), (-1, -1), (1, -1)),
}


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and bool(mask[nr, nc])
                    and not bool(seen[nr, nc])
                ):
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _arrow_direction(red_cells: np.ndarray) -> tuple[int, int]:
    pts = {tuple(map(int, p)) for p in red_cells}
    assert len(pts) == 3, "each object must contain exactly one 3-pixel red arrow"
    matches: list[tuple[int, int]] = []
    for tr, tc in pts:
        for name, rels in ARROW_PATTERNS.items():
            shaped = {(int(tr + dr), int(tc + dc)) for dr, dc in rels}
            if shaped == pts:
                matches.append(DIRS[name])
    assert len(matches) == 1, "red arrow shape must determine a unique direction"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each colored object repeatedly in the direction encoded by its internal red arrow.

    Input structure:
    - A gray background contains several separated connected objects.
    - Each object has one visible main color, one 3-pixel red arrow embedded in that object, and 1 to 4 black marker pixels.
    - The red arrow has one tip and two diagonal wings, pointing up, down, left, or right.

    Rule:
    - For each object, count its black marker pixels N and decode the red arrow direction.
    - Starting from the object's original bounding box, draw N adjacent copies in that direction, with each step equal to the bounding-box span along the direction.
    - Copies contain only the object's original main-color pixels; the red arrow and black markers are not copied. Copies that leave the grid are clipped.
    - The original objects remain unchanged and the grid shape is preserved.

    Key insight:
    The arrow shape supplies the direction, while the number of black marker pixels supplies how many clean main-color stamps to place.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    out = g.copy()

    for comp in _components(g != BG, connectivity=8):
        vals = [int(g[r, c]) for r, c in comp]
        main_colors = sorted({v for v in vals if v not in (BG, BLACK, RED)})
        assert len(main_colors) == 1, "each object must have one main color"
        color = int(main_colors[0])

        red = np.array([(int(r), int(c)) for r, c in comp if int(g[r, c]) == RED], dtype=int)
        black = np.array([(int(r), int(c)) for r, c in comp if int(g[r, c]) == BLACK], dtype=int)
        assert 1 <= len(black) <= 4, "black marker count must be between one and four"
        dr, dc = _arrow_direction(red)

        r0, c0, r1, c1 = _bbox(comp)
        span = int(r1 - r0 + 1) if dr else int(c1 - c0 + 1)
        main_rel = [(int(r - r0), int(c - c0)) for r, c in comp if int(g[r, c]) == color]
        assert main_rel, "object must contain main-color pixels"

        for step in range(1, len(black) + 1):
            br = r0 + dr * span * step
            bc = c0 + dc * span * step
            for rr, cc in main_rel:
                r = int(br + rr)
                c = int(bc + cc)
                if 0 <= r < h and 0 <= c < w:
                    out[r, c] = color

    return out


# --------------- generator ---------------

GEN_COLORS = [1, 3, 4, 6, 7, 8, 9]
GEN_GRID_MIN = 16
GEN_GRID_MAX = 30
GEN_OBJECT_COUNTS = (1, 2, 3)
GEN_ATTEMPTS = 400


def _local_arrow_cells(name: str, height: int, width: int) -> list[tuple[tuple[int, int], ...]]:
    out: list[tuple[tuple[int, int], ...]] = []
    for tr in range(height):
        for tc in range(width):
            cells = tuple((tr + dr, tc + dc) for dr, dc in ARROW_PATTERNS[name])
            if all(1 <= r < height - 1 and 1 <= c < width - 1 for r, c in cells):
                out.append(cells)
    return out


def _has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _connected(mask: np.ndarray) -> bool:
    return bool(mask.any() and len(_components(mask, connectivity=4)) == 1)


def _make_object_spec(color: int) -> dict | None:
    height = random.randint(5, 7)
    width = random.randint(5, 7)
    direction_name = random.choice(list(DIRS))
    marker_count = random.randint(1, 4)

    arrow_options = _local_arrow_cells(direction_name, height, width)
    if not arrow_options:
        return None
    arrow = set(random.choice(arrow_options))

    interior = [
        (r, c)
        for r in range(1, height - 1)
        for c in range(1, width - 1)
        if (r, c) not in arrow
    ]
    if len(interior) < marker_count:
        return None
    black = set(random.sample(interior, marker_count))

    mask = np.ones((height, width), dtype=bool)
    edge_cells = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if r in (0, height - 1) or c in (0, width - 1)
    ]
    random.shuffle(edge_cells)
    target_notches = random.randint(0, min(5, len(edge_cells)))
    for r, c in edge_cells:
        if target_notches <= 0:
            break
        trial = mask.copy()
        trial[r, c] = False
        main_trial = trial.copy()
        for rr, cc in arrow | black:
            main_trial[rr, cc] = False
        if _has_full_bbox(trial) and _connected(trial) and _connected(main_trial):
            mask = trial
            target_notches -= 1

    main_mask = mask.copy()
    for r, c in arrow | black:
        main_mask[r, c] = False
    if int(main_mask.sum()) < 10 or not _connected(main_mask):
        return None

    return {
        "color": int(color),
        "height": int(height),
        "width": int(width),
        "direction_name": direction_name,
        "direction": DIRS[direction_name],
        "marker_count": int(marker_count),
        "mask": mask,
        "main_mask": main_mask,
        "arrow": arrow,
        "black": black,
    }


def _stamp_cells(
    spec: dict, top: int, left: int, rows: int, cols: int
) -> tuple[set[tuple[int, int]], bool, bool]:
    dr, dc = spec["direction"]
    span = int(spec["height"]) if dr else int(spec["width"])
    cells: set[tuple[int, int]] = set()
    any_outside = False
    partial_copy = False

    for step in range(int(spec["marker_count"]) + 1):
        br = top + dr * span * step
        bc = left + dc * span * step
        local = spec["mask"] if step == 0 else spec["main_mask"]
        step_inside = False
        step_outside = False
        for rr, cc in np.argwhere(local):
            r = int(br + rr)
            c = int(bc + cc)
            if 0 <= r < rows and 0 <= c < cols:
                cells.add((r, c))
                if step > 0:
                    step_inside = True
            else:
                if step > 0:
                    step_outside = True
        any_outside = any_outside or step_outside
        partial_copy = partial_copy or (step > 0 and step_inside and step_outside)

    return cells, any_outside, partial_copy


def _place_object(
    spec: dict, grid: np.ndarray, forbidden: np.ndarray, want_clip: bool
) -> bool:
    rows, cols = grid.shape
    height = int(spec["height"])
    width = int(spec["width"])
    positions = [(r, c) for r in range(rows - height + 1) for c in range(cols - width + 1)]
    random.shuffle(positions)

    for top, left in positions:
        cells, any_outside, partial_copy = _stamp_cells(spec, top, left, rows, cols)
        if want_clip and not partial_copy:
            continue
        if not want_clip and any_outside:
            continue
        if not cells or any(bool(forbidden[r, c]) for r, c in cells):
            continue

        for rr, cc in np.argwhere(spec["main_mask"]):
            grid[top + int(rr), left + int(cc)] = int(spec["color"])
        for rr, cc in spec["black"]:
            grid[top + int(rr), left + int(cc)] = BLACK
        for rr, cc in spec["arrow"]:
            grid[top + int(rr), left + int(cc)] = RED

        for r, c in cells:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        forbidden[nr, nc] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    cols = random.randint(GEN_GRID_MIN, GEN_GRID_MAX)
    count = random.choice(GEN_OBJECT_COUNTS)
    colors = random.sample(GEN_COLORS, count)
    grid = np.full((rows, cols), BG, dtype=int)
    forbidden = np.zeros((rows, cols), dtype=bool)

    specs: list[dict] = []
    for color in colors:
        spec = _make_object_spec(color)
        if spec is None:
            return None
        specs.append(spec)

    clip_flags = [random.choice([False, True]) for _ in specs]
    if len(specs) > 1 and not any(clip_flags) and random.random() < 0.5:
        clip_flags[random.randrange(len(clip_flags))] = True

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        if not _place_object(specs[idx], grid, forbidden, clip_flags[idx]):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed1155420539 puzzle 3 input")
