from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
MASK_COLOR = 9
FRAME_COLOR_CHOICES = (5, 6, 7, 8)
POINTER_COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8)
DIRECTIONS = ("up", "down", "left", "right")
PAIR_COUNT_CHOICES = (2, 3, 4, 4)
POINTER_SHAPES = {
    "up": (((0, 0), (1, 0), (1, 1), (1, 2)), ((0, 2), (1, 0), (1, 1), (1, 2))),
    "down": (((0, 0), (0, 1), (0, 2), (1, 0)), ((0, 0), (0, 1), (0, 2), (1, 2))),
    "left": (((0, 0), (0, 1), (1, 1), (2, 1)), ((2, 0), (0, 1), (1, 1), (2, 1))),
    "right": (((0, 0), (1, 0), (2, 0), (0, 1)), ((0, 0), (1, 0), (2, 0), (2, 1))),
}
POINTER_BOX = {"up": (2, 3), "down": (2, 3), "left": (3, 2), "right": (3, 2)}
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _monochromatic_components(grid: np.ndarray, active: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not active[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and active[nr, nc] and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "bbox": _bbox(coords),
                }
            )

    return comps


def _frame_info(comp: dict) -> dict | None:
    coords = list(comp["coords"])
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected = set()
    for rr in range(r0, r1 + 1):
        expected.add((rr, c0))
        expected.add((rr, c1))
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))

    coord_set = set(coords)
    if coord_set != expected:
        return None

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "coord_set": coord_set,
        "bbox": comp["bbox"],
    }


def _pointer_direction(comp: dict) -> str | None:
    coords = list(comp["coords"])
    if len(coords) != 4:
        return None
    r0, c0, r1, c1 = comp["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if (height, width) == (2, 3):
        row_counts = [sum(1 for r, _ in coords if r == rr) for rr in range(r0, r1 + 1)]
        if sorted(row_counts) != [1, 3]:
            return None
        return "up" if row_counts[0] == 1 else "down"

    if (height, width) == (3, 2):
        col_counts = [sum(1 for _, c in coords if c == cc) for cc in range(c0, c1 + 1)]
        if sorted(col_counts) != [1, 3]:
            return None
        return "left" if col_counts[0] == 1 else "right"

    return None


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    active = g != bg

    frames: list[dict] = []
    border_mask = np.zeros(g.shape, dtype=bool)
    for comp in _monochromatic_components(g, active):
        frame = _frame_info(comp)
        if frame is None:
            continue
        frames.append(frame)
        for r, c in frame["coords"]:
            border_mask[r, c] = True

    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one container frame"

    interior_mask = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask_coords: list[tuple[int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) != bg:
                    interior_mask[r, c] = True
                    mask_coords.append((r, c))
        assert mask_coords, "each frame must contain an internal mask"
        frame["mask_coords"] = tuple(mask_coords)

    pointer_mask = active & ~border_mask & ~interior_mask
    pointers: list[dict] = []
    for comp in _monochromatic_components(g, pointer_mask):
        direction = _pointer_direction(comp)
        assert direction is not None, "remaining objects must be valid L-shaped pointers"
        pointers.append(
            {
                "color": int(comp["color"]),
                "coords": comp["coords"],
                "bbox": comp["bbox"],
                "direction": direction,
            }
        )

    pointers.sort(key=lambda pointer: pointer["bbox"])
    assert pointers, "expected at least one pointer"
    return bg, frames, pointers


def _target_frame_index(pointer: dict, frames: list[dict]) -> int:
    pr0, pc0, pr1, pc1 = pointer["bbox"]
    direction = pointer["direction"]
    candidates: list[tuple[int, int]] = []

    for idx, frame in enumerate(frames):
        fr0, fc0, fr1, fc1 = frame["bbox"]
        if direction == "up":
            if fr1 < pr0 and _overlap(pc0, pc1, fc0, fc1):
                candidates.append((pr0 - fr1, idx))
        elif direction == "down":
            if fr0 > pr1 and _overlap(pc0, pc1, fc0, fc1):
                candidates.append((fr0 - pr1, idx))
        elif direction == "left":
            if fc1 < pc0 and _overlap(pr0, pr1, fr0, fr1):
                candidates.append((pc0 - fc1, idx))
        else:
            if fc0 > pc1 and _overlap(pr0, pr1, fr0, fr1):
                candidates.append((fc0 - pc1, idx))

    assert candidates, "each pointer must hit a frame in its facing direction"
    candidates.sort()
    best_distance = candidates[0][0]
    best = [idx for dist, idx in candidates if dist == best_distance]
    assert len(best) == 1, "nearest frame must be unique"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each L-shaped pointer into the first frame in its facing direction and recolor that frame's internal mask.

    Input structure:
    - The background is black and dominates the grid.
    - Several hollow rectangular container frames appear as one-pixel borders, each enclosing a smaller non-black mask shape inside a black interior.
    - Several other monochrome objects are 4-cell L-shaped pointers made from a 3-cell bar plus one perpendicular nub, and the nub direction indicates up, down, left, or right.

    Rule:
    - Detect each pointer's facing direction from the side on which its single protruding pixel appears.
    - Along that direction, find the first container frame whose row or column span overlaps the pointer's projected lane.
    - Recolor every non-black mask pixel inside that frame to the pointer's color, keep the frame border unchanged, and leave the frame's black interior pixels black.
    - Erase every pointer itself to black.

    Key insight:
    - The L-shaped asymmetry is the directional cue, while the colored pixels already inside each frame are not noise: they are a stencil telling which interior cells should inherit the pointer's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg, frames, pointers = _parse_scene(g)
    out = g.copy()

    for pointer in pointers:
        for r, c in pointer["coords"]:
            out[r, c] = bg

    for pointer in pointers:
        frame = frames[_target_frame_index(pointer, frames)]
        for r, c in frame["mask_coords"]:
            out[r, c] = int(pointer["color"])

    return out


def _is_asymmetric(mask: np.ndarray) -> bool:
    return not (
        np.array_equal(mask, mask[::-1, :])
        or np.array_equal(mask, mask[:, ::-1])
        or np.array_equal(mask, mask[::-1, ::-1])
    )


def _random_mask(inner_h: int, inner_w: int) -> np.ndarray:
    max_cells = min(inner_h * inner_w - 1, max(4, inner_h + inner_w))
    min_cells = 3 if inner_h * inner_w >= 6 else 2

    for _ in range(80):
        target = random.randint(min_cells, max_cells)
        start = (random.randrange(inner_h), random.randrange(inner_w))
        shape = {start}
        frontier = {start}
        while len(shape) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            r, c = cell
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < inner_h and 0 <= nc < inner_w and (nr, nc) not in shape:
                    neighbors.append((nr, nc))
            random.shuffle(neighbors)
            for neighbor in neighbors:
                shape.add(neighbor)
                frontier.add(neighbor)
                if len(shape) == target:
                    break

        if len(shape) != target:
            continue

        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for r, c in shape:
            mask[r, c] = True
        if mask.all():
            continue
        if _is_asymmetric(mask):
            return mask

    mask = np.zeros((inner_h, inner_w), dtype=bool)
    fallback = {(0, 0), (0, min(1, inner_w - 1)), (min(1, inner_h - 1), 0)}
    if inner_h >= 3 and inner_w >= 2:
        fallback.add((2, 1))
    for r, c in fallback:
        mask[r, c] = True
    return mask


def _rect_is_clear(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not bool(blocked[rr0:rr1, cc0:cc1].any())


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _mark_projection_lane(blocked: np.ndarray, direction: str, frame_bbox: tuple[int, int, int, int], pointer_bbox: tuple[int, int, int, int]) -> None:
    pr0, pc0, pr1, pc1 = pointer_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if direction == "up":
        blocked[fr1 + 1 : pr0, pc0 : pc1 + 1] = True
    elif direction == "down":
        blocked[pr1 + 1 : fr0, pc0 : pc1 + 1] = True
    elif direction == "left":
        blocked[pr0 : pr1 + 1, fc1 + 1 : pc0] = True
    else:
        blocked[pr0 : pr1 + 1, pc1 + 1 : fc0] = True


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, frame_color: int) -> tuple[int, int, int, int]:
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    return r0, c0, r1, c1


def _paint_mask(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> None:
    rows, cols = np.where(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist()):
        grid[r0 + 1 + dr, c0 + 1 + dc] = MASK_COLOR


def _paint_pointer(grid: np.ndarray, r0: int, c0: int, color: int, direction: str) -> tuple[int, int, int, int]:
    shape = random.choice(POINTER_SHAPES[direction])
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color
    h, w = POINTER_BOX[direction]
    return r0, c0, r0 + h - 1, c0 + w - 1


def _place_pair(grid: np.ndarray, blocked: np.ndarray, direction: str, color: int, frame_color: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = grid.shape

    for _ in range(120):
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)
        frame_h = inner_h + 2
        frame_w = inner_w + 2
        gap = random.randint(2, 6)
        pointer_h, pointer_w = POINTER_BOX[direction]

        if direction == "up":
            max_r0 = h - (frame_h + gap + pointer_h)
            if max_r0 < 0 or w < frame_w:
                continue
            fr0 = random.randint(0, max_r0)
            fc0 = random.randint(0, w - frame_w)
            pr0 = fr0 + frame_h + gap
            pc_choices = list(range(fc0, fc0 + frame_w - pointer_w + 1))
        elif direction == "down":
            min_r0 = gap + pointer_h
            if min_r0 > h - frame_h or w < frame_w:
                continue
            fr0 = random.randint(min_r0, h - frame_h)
            fc0 = random.randint(0, w - frame_w)
            pr0 = fr0 - gap - pointer_h
            pc_choices = list(range(fc0, fc0 + frame_w - pointer_w + 1))
        elif direction == "left":
            max_c0 = w - (frame_w + gap + pointer_w)
            if max_c0 < 0 or h < frame_h:
                continue
            fr0 = random.randint(0, h - frame_h)
            fc0 = random.randint(0, max_c0)
            pc0 = fc0 + frame_w + gap
            pr_choices = list(range(fr0, fr0 + frame_h - pointer_h + 1))
        else:
            min_c0 = gap + pointer_w
            if min_c0 > w - frame_w or h < frame_h:
                continue
            fr0 = random.randint(0, h - frame_h)
            fc0 = random.randint(min_c0, w - frame_w)
            pc0 = fc0 - gap - pointer_w
            pr_choices = list(range(fr0, fr0 + frame_h - pointer_h + 1))

        if direction in ("up", "down"):
            random.shuffle(pc_choices)
            for pc0 in pc_choices:
                if not _rect_is_clear(blocked, fr0, fc0, frame_h, frame_w):
                    continue
                if not _rect_is_clear(blocked, pr0, pc0, pointer_h, pointer_w):
                    continue
                frame_bbox = _paint_frame(grid, fr0, fc0, inner_h, inner_w, frame_color)
                _paint_mask(grid, fr0, fc0, _random_mask(inner_h, inner_w))
                pointer_bbox = _paint_pointer(grid, pr0, pc0, color, direction)
                _mark_rect(blocked, fr0, fc0, frame_h, frame_w)
                _mark_rect(blocked, pr0, pc0, pointer_h, pointer_w)
                _mark_projection_lane(blocked, direction, frame_bbox, pointer_bbox)
                return frame_bbox, pointer_bbox
        else:
            random.shuffle(pr_choices)
            for pr0 in pr_choices:
                if not _rect_is_clear(blocked, fr0, fc0, frame_h, frame_w):
                    continue
                if not _rect_is_clear(blocked, pr0, pc0, pointer_h, pointer_w):
                    continue
                frame_bbox = _paint_frame(grid, fr0, fc0, inner_h, inner_w, frame_color)
                _paint_mask(grid, fr0, fc0, _random_mask(inner_h, inner_w))
                pointer_bbox = _paint_pointer(grid, pr0, pc0, color, direction)
                _mark_rect(blocked, fr0, fc0, frame_h, frame_w)
                _mark_rect(blocked, pr0, pc0, pointer_h, pointer_w)
                _mark_projection_lane(blocked, direction, frame_bbox, pointer_bbox)
                return frame_bbox, pointer_bbox

    return None


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    directions = random.sample(DIRECTIONS, pair_count)

    min_size = 16 + 3 * pair_count
    height = random.randint(min_size, 30)
    width = random.randint(min_size, 30)
    grid = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    frame_color = random.choice(FRAME_COLOR_CHOICES)
    pointer_colors = random.sample([c for c in POINTER_COLOR_POOL if c not in {BG, MASK_COLOR, frame_color}], pair_count)

    intended_targets: dict[int, tuple[int, int, int, int]] = {}
    for direction, color in zip(directions, pointer_colors):
        placed = _place_pair(grid, blocked, direction, color, frame_color)
        if placed is None:
            return None
        frame_bbox, _pointer_bbox = placed
        intended_targets[color] = frame_bbox

    try:
        bg, frames, pointers = _parse_scene(grid)
        assert bg == BG
        assert len(frames) == pair_count
        assert len(pointers) == pair_count
        seen_targets = set()
        for pointer in pointers:
            target = frames[_target_frame_index(pointer, frames)]["bbox"]
            assert intended_targets[int(pointer["color"])] == target
            seen_targets.add(target)
        assert len(seen_targets) == pair_count
        out = main(grid)
        assert out.shape == grid.shape
        assert np.count_nonzero(grid != out) > 0
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
