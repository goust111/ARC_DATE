import random
from collections import Counter, deque

import numpy as np


BG = 0
FRAME = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
BRANCHES = ("short", "medium", "long")
DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
CLOCKWISE = {
    "up": "right",
    "right": "down",
    "down": "left",
    "left": "up",
}
SIDE_TO_DIR = {
    "top": "down",
    "bottom": "up",
    "left": "right",
    "right": "left",
}
ARROW_TO_SIDE = {
    "down": "top",
    "up": "bottom",
    "right": "left",
    "left": "right",
}
ARROW_TEMPLATES = {
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
}
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    return len(coords) == h * w


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_frame(grid: np.ndarray, bg: int) -> tuple[int, int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(coords)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            border = _border_cells(r0, c0, r1, c1)
            if len(coords) != len(border):
                continue
            if set(map(tuple, coords.tolist())) != border:
                continue
            candidates.append((len(coords), color, r0, c0, r1, c1))
    assert candidates, "expected one hollow rectangular frame"
    candidates.sort(reverse=True)
    _area, color, r0, c0, r1, c1 = candidates[0]
    return int(color), int(r0), int(c0), int(r1), int(c1)


def _arrow_direction(coords: np.ndarray) -> str | None:
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3) or len(coords) != 5:
        return None
    patch = np.zeros((3, 3), dtype=int)
    rel = coords - np.array([r0, c0], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = 1
    for direction, template in ARROW_TEMPLATES.items():
        if np.array_equal(patch, template):
            return direction
    return None


def _source_side(block_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> str:
    br0, bc0, br1, bc1 = block_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if br1 == fr0 - 1 and fc0 + 1 <= bc0 <= bc1 <= fc1 - 1:
        return "top"
    if br0 == fr1 + 1 and fc0 + 1 <= bc0 <= bc1 <= fc1 - 1:
        return "bottom"
    if bc1 == fc0 - 1 and fr0 + 1 <= br0 <= br1 <= fr1 - 1:
        return "left"
    if bc0 == fc1 + 1 and fr0 + 1 <= br0 <= br1 <= fr1 - 1:
        return "right"
    raise AssertionError("source block must sit flush against exactly one frame side")


def _entry_point(block_bbox: tuple[int, int, int, int], side: str, frame_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    br0, bc0, br1, bc1 = block_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if side in ("top", "bottom"):
        mid = (bc0 + bc1) // 2
        assert (bc1 - bc0 + 1) % 2 == 1, "contact span must have a unique midpoint"
        return (fr0 + 1, mid) if side == "top" else (fr1 - 1, mid)
    mid = (br0 + br1) // 2
    assert (br1 - br0 + 1) % 2 == 1, "contact span must have a unique midpoint"
    return (mid, fc0 + 1) if side == "left" else (mid, fc1 - 1)


def _trace_path(
    start: tuple[int, int],
    direction: str,
    steps: int,
    inner_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = inner_bbox
    r, c = start
    out: list[tuple[int, int]] = []
    for i in range(int(steps)):
        assert r0 <= r <= r1 and c0 <= c <= c1
        out.append((r, c))
        if i == steps - 1:
            break
        dr, dc = DELTA[direction]
        nr, nc = r + dr, c + dc
        if not (r0 <= nr <= r1 and c0 <= nc <= c1):
            direction = CLOCKWISE[direction]
            dr, dc = DELTA[direction]
            nr, nc = r + dr, c + dc
        r, c = nr, nc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each outer source block into the hollow frame, using the block area as path length and turning at inner walls.

    Input structure:
    - A dominant black background surrounds one large one-pixel-thick hollow rectangular frame.
    - Outside the frame, each active color appears as one solid rectangle flush against a frame side and one separated same-colored arrow.
    - Each arrow points from the outside toward the rectangle and therefore toward the targeted frame side.

    Rule:
    - Detect the frame, then pair each colored rectangle with its same-colored arrow.
    - The rectangle area is the path length N.
    - Start from the midpoint of the rectangle's contact span and paint a one-pixel-wide path of N cells inside the frame.
    - The first leg moves perpendicular to the contacted side; whenever the next step would hit the frame border, turn 90 degrees clockwise and continue.
    - Keep the original frame, source rectangles, and arrows unchanged.

    Key insight:
    - The arrow tells which side launches the projection, the rectangle's 2D area becomes a 1D path length, and wall hits convert one straight ray into a turning bounce path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    frame_color, fr0, fc0, fr1, fc1 = _detect_frame(g, bg)
    frame_bbox = (fr0, fc0, fr1, fc1)
    inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)

    objects_by_color: dict[int, list[np.ndarray]] = {}
    for color in np.unique(g):
        color = int(color)
        if color in (bg, frame_color):
            continue
        comps = _components_mask(g == color)
        if comps:
            objects_by_color[color] = comps

    out = g.copy()
    claimed: dict[tuple[int, int], int] = {}

    for color, comps in objects_by_color.items():
        rectangles = [coords for coords in comps if _is_solid_rectangle(coords)]
        arrows = [coords for coords in comps if _arrow_direction(coords) is not None]
        assert len(rectangles) == 1 and len(arrows) == 1, "each active color must have one rectangle and one arrow"
        block = rectangles[0]
        arrow = arrows[0]
        block_bbox = _bbox(block)
        side = _source_side(block_bbox, frame_bbox)
        arrow_dir = _arrow_direction(arrow)
        assert arrow_dir is not None and ARROW_TO_SIDE[arrow_dir] == side, "arrow direction must match the contacted side"

        start = _entry_point(block_bbox, side, frame_bbox)
        path = _trace_path(start, SIDE_TO_DIR[side], int(len(block)), inner_bbox)
        for cell in path:
            if cell in claimed:
                assert claimed[cell] == color, "projected paths must not overlap across colors"
            claimed[cell] = color

    for (r, c), color in claimed.items():
        out[r, c] = color
    return out


def _candidate_specs(
    side: str,
    frame_bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    branch: str,
) -> list[dict[str, object]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    path_inner_bbox = (fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1)
    h, w = grid_shape
    specs: list[dict[str, object]] = []
    if side in ("top", "bottom"):
        usable = inner_w - 2
        spans = [s for s in (1, 3, 5, 7) if s <= usable]
        for span in spans:
            c_lo = fc0 + 2 + span // 2
            c_hi = fc1 - 2 - span // 2
            for center in range(c_lo, c_hi + 1):
                extra2 = center - (fc0 + 1) if side == "top" else (fc1 - 1) - center
                first_leg = inner_h
                for depth in (1, 2, 3):
                    area = span * depth
                    cycle_len = 2 * (inner_h + inner_w) - 4
                    if area > cycle_len:
                        continue
                    if branch == "short" and not (area <= first_leg):
                        continue
                    if branch == "medium" and not (first_leg < area <= first_leg + extra2):
                        continue
                    if branch == "long" and not (area > first_leg + extra2):
                        continue
                    bc0, bc1 = center - span // 2, center + span // 2
                    br0 = fr0 - depth if side == "top" else fr1 + 1
                    br1 = fr0 - 1 if side == "top" else fr1 + depth
                    if br0 < 0 or br1 >= h:
                        continue
                    if side == "top":
                        ar0 = br0 - 4
                        ac0 = center - 1
                    else:
                        ar0 = br1 + 2
                        ac0 = center - 1
                    if ar0 < 0 or ar0 + 2 >= h or ac0 < 0 or ac0 + 2 >= w:
                        continue
                    block_bbox = (br0, bc0, br1, bc1)
                    start = _entry_point(block_bbox, side, frame_bbox)
                    path = _trace_path(start, SIDE_TO_DIR[side], area, path_inner_bbox)
                    specs.append(
                        {
                            "side": side,
                            "span": span,
                            "depth": depth,
                            "area": area,
                            "center": center,
                            "block_bbox": block_bbox,
                            "arrow_top_left": (ar0, ac0),
                            "arrow_dir": SIDE_TO_DIR[side],
                            "path": path,
                            "branch": branch,
                        }
                    )
    else:
        usable = inner_h - 2
        spans = [s for s in (1, 3, 5, 7) if s <= usable]
        for span in spans:
            r_lo = fr0 + 2 + span // 2
            r_hi = fr1 - 2 - span // 2
            for center in range(r_lo, r_hi + 1):
                extra2 = center - (fr0 + 1) if side == "left" else (fr1 - 1) - center
                first_leg = inner_w
                for depth in (1, 2, 3):
                    area = span * depth
                    cycle_len = 2 * (inner_h + inner_w) - 4
                    if area > cycle_len:
                        continue
                    if branch == "short" and not (area <= first_leg):
                        continue
                    if branch == "medium" and not (first_leg < area <= first_leg + extra2):
                        continue
                    if branch == "long" and not (area > first_leg + extra2):
                        continue
                    br0, br1 = center - span // 2, center + span // 2
                    bc0 = fc0 - depth if side == "left" else fc1 + 1
                    bc1 = fc0 - 1 if side == "left" else fc1 + depth
                    if bc0 < 0 or bc1 >= w:
                        continue
                    if side == "left":
                        ar0 = center - 1
                        ac0 = bc0 - 4
                    else:
                        ar0 = center - 1
                        ac0 = bc1 + 2
                    if ar0 < 0 or ar0 + 2 >= h or ac0 < 0 or ac0 + 2 >= w:
                        continue
                    block_bbox = (br0, bc0, br1, bc1)
                    start = _entry_point(block_bbox, side, frame_bbox)
                    path = _trace_path(start, SIDE_TO_DIR[side], area, path_inner_bbox)
                    specs.append(
                        {
                            "side": side,
                            "span": span,
                            "depth": depth,
                            "area": area,
                            "center": center,
                            "block_bbox": block_bbox,
                            "arrow_top_left": (ar0, ac0),
                            "arrow_dir": SIDE_TO_DIR[side],
                            "path": path,
                            "branch": branch,
                        }
                    )
    random.shuffle(specs)
    return specs


def _cells_for_block(block_bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = block_bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _cells_for_arrow(top_left: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    r0, c0 = top_left
    patch = ARROW_TEMPLATES[direction]
    cells = set()
    for dr in range(3):
        for dc in range(3):
            if patch[dr, dc]:
                cells.add((r0 + dr, c0 + dc))
    return cells


def _branch_plan(count: int) -> list[str]:
    if count == 2:
        plan = random.sample(list(BRANCHES), 2)
    elif count == 3:
        plan = list(BRANCHES)
    else:
        plan = list(BRANCHES) + [random.choice(BRANCHES)]
    random.shuffle(plan)
    return plan


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.randint(7, 12)
    frame_w = random.randint(7, 12)

    extra_h = random.randint(0, min(30 - (frame_h + 14), 4))
    extra_w = random.randint(0, min(30 - (frame_w + 14), 4))
    top_margin = 7 + random.randint(0, extra_h)
    bottom_margin = 7 + (extra_h - (top_margin - 7))
    left_margin = 7 + random.randint(0, extra_w)
    right_margin = 7 + (extra_w - (left_margin - 7))

    height = top_margin + frame_h + bottom_margin
    width = left_margin + frame_w + right_margin
    assert height <= 30 and width <= 30

    fr0, fc0 = top_margin, left_margin
    fr1, fc1 = fr0 + frame_h - 1, fc0 + frame_w - 1
    frame_bbox = (fr0, fc0, fr1, fc1)

    grid = np.full((height, width), BG, dtype=int)
    for r, c in _border_cells(fr0, fc0, fr1, fc1):
        grid[r, c] = FRAME

    source_count = random.randint(2, 4)
    sides = random.sample(list(SIDES), source_count)
    branches = _branch_plan(source_count)
    colors = random.sample(COLOR_POOL, source_count)
    side_to_branch = dict(zip(sides, branches))
    random.shuffle(sides)

    occupied_outside: set[tuple[int, int]] = set(_border_cells(fr0, fc0, fr1, fc1))
    occupied_paths: dict[tuple[int, int], int] = {}
    chosen_specs: list[tuple[int, dict[str, object]]] = []

    for side, color in zip(sides, colors):
        branch = side_to_branch[side]
        candidates = _candidate_specs(side, frame_bbox, grid.shape, branch)
        if not candidates:
            return None

        placed = False
        for spec in candidates:
            block_cells = _cells_for_block(spec["block_bbox"])
            arrow_cells = _cells_for_arrow(spec["arrow_top_left"], spec["arrow_dir"])
            if occupied_outside & block_cells:
                continue
            if occupied_outside & arrow_cells:
                continue
            if block_cells & arrow_cells:
                continue

            path_cells = spec["path"]
            conflict = False
            for cell in path_cells:
                existing = occupied_paths.get(cell)
                if existing is not None and existing != color:
                    conflict = True
                    break
            if conflict:
                continue

            occupied_outside |= block_cells | arrow_cells
            for cell in path_cells:
                occupied_paths[cell] = color
            chosen_specs.append((color, spec))
            placed = True
            break
        if not placed:
            return None

    if not any(spec["branch"] == "long" for _, spec in chosen_specs):
        return None

    for color, spec in chosen_specs:
        br0, bc0, br1, bc1 = spec["block_bbox"]
        grid[br0 : br1 + 1, bc0 : bc1 + 1] = color
        ar0, ac0 = spec["arrow_top_left"]
        patch = ARROW_TEMPLATES[spec["arrow_dir"]] * color
        mask = patch != 0
        view = grid[ar0 : ar0 + 3, ac0 : ac0 + 3]
        if np.any(view[mask] != BG):
            return None
        view[mask] = patch[mask]

    try:
        out = main(grid)
    except Exception:
        return None

    frame_window = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
    if not np.any(frame_window != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-bounce input")
