import random
from collections import deque

import numpy as np


BG = 0
ARROW = 2
BODY_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PATTERN_A = {(0, 0), (1, 0), (1, 1)}  # up or right
PATTERN_B = {(0, 0), (0, 1), (1, 1)}  # left or down

GEN_ATTEMPTS = 400
SHAPE_ATTEMPTS = 200
PLACE_ATTEMPTS = 200
GRID_MIN = 18
GRID_MAX = 30
OBJ_COUNT_RANGE = (2, 4)
SHAPE_H_RANGE = (5, 8)
SHAPE_W_RANGE = (5, 8)
MAX_BOUNDARY_REMOVALS = 5
MARGIN = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    total = int(mask.sum())
    if total == 0:
        return False
    comps = _components(mask)
    return len(comps) == 1 and len(comps[0]) == total


def _crop(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_cells_from_block(top: int, left: int, direction: str) -> list[tuple[int, int]]:
    if direction in ("up", "right"):
        return [(top, left), (top + 1, left), (top + 1, left + 1)]
    return [(top, left), (top, left + 1), (top + 1, left + 1)]


def _candidate_arrow_blocks(mask: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(0, h - 1):
        for c in range(0, w - 1):
            if not mask[r : r + 2, c : c + 2].all():
                continue
            if r == 0 or c == 0 or r + 2 >= h or c + 2 >= w:
                continue
            if not mask[r - 1 : r + 3, c - 1 : c + 3].all():
                continue

            top_dist = r
            left_dist = c
            bottom_dist = h - r - 2
            right_dist = w - c - 2

            if direction == "up" and top_dist < right_dist:
                candidates.append((r, c))
            elif direction == "right" and right_dist < top_dist:
                candidates.append((r, c))
            elif direction == "left" and left_dist < bottom_dist:
                candidates.append((r, c))
            elif direction == "down" and bottom_dist < left_dist:
                candidates.append((r, c))
    return candidates


def _make_shape(rng: random.Random, direction: str) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(*SHAPE_H_RANGE)
        w = rng.randint(*SHAPE_W_RANGE)
        mask = np.ones((h, w), dtype=bool)

        boundary = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        ]
        rng.shuffle(boundary)
        removals = rng.randint(0, MAX_BOUNDARY_REMOVALS)
        removed = 0
        for r, c in boundary:
            if removed >= removals:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < 14:
                continue
            if not _is_connected(trial):
                continue
            mask = trial
            removed += 1

        mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        mask = _crop(mask)

        arrow_blocks = _candidate_arrow_blocks(mask, direction)
        rng.shuffle(arrow_blocks)
        for ar, ac in arrow_blocks:
            arrow_cells = _arrow_cells_from_block(ar, ac, direction)
            body = mask.copy()
            for rr, cc in arrow_cells:
                body[rr, cc] = False
            if body.sum() < 8 or not _is_connected(body):
                continue
            return mask, arrow_cells
    return None


def _infer_direction(component: list[tuple[int, int]], red_cells: list[tuple[int, int]]) -> str:
    r0, c0, r1, c1 = _bbox(component)
    red_r0, red_c0, red_r1, red_c1 = _bbox(red_cells)
    norm = {(r - red_r0, c - red_c0) for r, c in red_cells}

    if norm == PATTERN_A:
        top_dist = red_r0 - r0
        right_dist = c1 - red_c1
        return "right" if right_dist <= top_dist else "up"
    if norm == PATTERN_B:
        left_dist = red_c0 - c0
        bottom_dist = r1 - red_r1
        return "left" if left_dist <= bottom_dist else "down"
    raise ValueError(f"Unrecognized arrow pattern: {sorted(norm)}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored shape by one of its own spans, using the enclosed red arrow as the direction cue.

    Input structure:
    - The background is black.
    - The foreground contains several separated filled shapes; each shape is a 4-connected non-black component.
    - Every shape uses one body color plus exactly one 3-pixel red arrow marker inside it.
    - The red marker is an L-shaped triomino that encodes one of the four cardinal directions.

    Rule:
    - For each non-black component, isolate its three red pixels and decode the arrow direction.
    - Compute the component's bounding-box span along that direction: width for left/right, height for up/down.
    - Move the entire component, including the red arrow, by exactly that span in the decoded direction.
    - Clear the original position to black and keep the grid size unchanged.

    Key insight:
    - Each object moves by "one body-length" in the arrow direction, so the needed distance is not constant and must be derived from that object's own bounding box.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    moves: list[tuple[list[tuple[int, int]], int, int]] = []
    for comp in _components(g != BG):
        red_cells = [(r, c) for r, c in comp if int(g[r, c]) == ARROW]
        if len(red_cells) != 3:
            raise ValueError("Each object must contain exactly one 3-pixel red arrow.")

        body_colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) not in (BG, ARROW)}
        if len(body_colors) != 1:
            raise ValueError("Each object must have exactly one non-red body color.")

        direction = _infer_direction(comp, red_cells)
        r0, c0, r1, c1 = _bbox(comp)
        span = (c1 - c0 + 1) if direction in ("left", "right") else (r1 - r0 + 1)
        dr, dc = DIRS[direction]
        moves.append((comp, dr * span, dc * span))

    h, w = g.shape
    for comp, dr, dc in moves:
        for r, c in comp:
            rr = r + dr
            cc = c + dc
            if not (0 <= rr < h and 0 <= cc < w):
                raise ValueError("Translated shape left the grid.")
            if out[rr, cc] != BG:
                raise ValueError("Translated shapes overlapped.")
            out[rr, cc] = int(g[r, c])
    return out


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserved[rr, cc] = True


def _manual_output(h: int, w: int, specs: list[dict]) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for spec in specs:
        mask = spec["mask"]
        arrow = set(spec["arrow"])
        body_color = spec["color"]
        sr = spec["row"]
        sc = spec["col"]
        direction = spec["direction"]
        span = mask.shape[1] if direction in ("left", "right") else mask.shape[0]
        dr, dc = DIRS[direction]
        shift_r = dr * span
        shift_c = dc * span

        for lr, lc in np.argwhere(mask):
            rr = sr + int(lr) + shift_r
            cc = sc + int(lc) + shift_c
            color = ARROW if (int(lr), int(lc)) in arrow else body_color
            if out[rr, cc] != BG:
                raise ValueError("Manual output overlap detected.")
            out[rr, cc] = color
    return out


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    n_objects = rng.randint(*OBJ_COUNT_RANGE)
    directions = ["up", "down", "left", "right"]
    rng.shuffle(directions)
    directions = directions[:n_objects]
    colors = rng.sample(BODY_COLORS, n_objects)

    reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []

    for direction, color in zip(directions, colors):
        placed = False
        for _ in range(PLACE_ATTEMPTS):
            shape = _make_shape(rng, direction)
            if shape is None:
                continue
            mask, arrow = shape
            sh, sw = mask.shape

            if direction == "right":
                row_lo, row_hi = MARGIN, h - sh - MARGIN
                col_lo, col_hi = MARGIN, w - 2 * sw - MARGIN
            elif direction == "left":
                row_lo, row_hi = MARGIN, h - sh - MARGIN
                col_lo, col_hi = sw + MARGIN, w - sw - MARGIN
            elif direction == "down":
                row_lo, row_hi = MARGIN, h - 2 * sh - MARGIN
                col_lo, col_hi = MARGIN, w - sw - MARGIN
            else:
                row_lo, row_hi = sh + MARGIN, h - sh - MARGIN
                col_lo, col_hi = MARGIN, w - sw - MARGIN

            if row_lo > row_hi or col_lo > col_hi:
                continue

            sr = rng.randint(row_lo, row_hi)
            sc = rng.randint(col_lo, col_hi)
            dr, dc = DIRS[direction]
            span = sw if direction in ("left", "right") else sh
            shift_r = dr * span
            shift_c = dc * span

            src_cells = [(sr + r, sc + c) for r, c in np.argwhere(mask)]
            dst_cells = [(r + shift_r, c + shift_c) for r, c in src_cells]

            if any(reserved[r, c] for r, c in src_cells + dst_cells):
                continue

            specs.append(
                {
                    "mask": mask,
                    "arrow": arrow,
                    "color": color,
                    "row": sr,
                    "col": sc,
                    "direction": direction,
                }
            )
            _mark_reserved(reserved, src_cells + dst_cells)
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        mask = spec["mask"]
        arrow = set(spec["arrow"])
        body_color = spec["color"]
        sr = spec["row"]
        sc = spec["col"]
        for lr, lc in np.argwhere(mask):
            grid[sr + int(lr), sc + int(lc)] = body_color
        for rr, cc in arrow:
            grid[sr + rr, sc + cc] = ARROW

    comps = _components(grid != BG)
    if len(comps) != n_objects:
        return None

    for spec in specs:
        shape_cells = int(spec["mask"].sum())
        found = False
        for comp in comps:
            if len(comp) != shape_cells:
                continue
            red_cells = [(r, c) for r, c in comp if int(grid[r, c]) == ARROW]
            if len(red_cells) != 3:
                continue
            if _infer_direction(comp, red_cells) == spec["direction"]:
                found = True
                break
        if not found:
            return None

    expected = _manual_output(h, w, specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid puzzle instance.")
