from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5

HORIZONTAL = "horizontal"
VERTICAL = "vertical"

GLYPH_HORIZONTAL = np.array(
    [
        [GRAY, BLACK, BLACK],
        [GRAY, BLACK, BLACK],
        [GRAY, GRAY, BLACK],
    ],
    dtype=int,
)

GLYPH_VERTICAL = np.array(
    [
        [BLACK, GRAY, GRAY],
        [GRAY, GRAY, BLACK],
        [GRAY, BLACK, BLACK],
    ],
    dtype=int,
)

GLYPH_TO_AXIS = {
    tuple(map(tuple, (GLYPH_HORIZONTAL == GRAY).astype(int))): HORIZONTAL,
    tuple(map(tuple, (GLYPH_VERTICAL == GRAY).astype(int))): VERTICAL,
}

SIDE_RANGE = (7, 9)
FRAME_COUNT_CHOICES = (2, 3)
GRID_SIZE_RANGE = (20, 28)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    if connectivity == 8:
        offsets = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _glyph_key(patch: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(map(tuple, (patch == GRAY).astype(int)))


def _mirror_cells(cells: list[tuple[int, int]], axis: str, interior_size: int) -> list[tuple[int, int]]:
    if axis == HORIZONTAL:
        return [(r, interior_size - 1 - c) for r, c in cells]
    if axis == VERTICAL:
        return [(interior_size - 1 - r, c) for r, c in cells]
    raise ValueError(f"unknown axis: {axis}")


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for coords in _component_coords(grid == BLUE, connectivity=8):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        side = r1 - r0 + 1
        assert side == c1 - c0 + 1, "frame bbox must be square"
        assert SIDE_RANGE[0] <= side <= SIDE_RANGE[1], "unexpected frame size"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(patch, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        assert np.all(patch[border] == BLUE), "frame border must be blue"
        assert not np.any(patch[1:-1, 1:-1] == BLUE), "frame interior must not contain blue"
        frames.append(box)
    return sorted(frames, key=lambda box: (box[0], box[1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each frame's gray glyph to choose the mirror axis, then complete the red motif inside the frame.

    Input structure:
    - A black background contains 2-3 separated blue hollow square frames.
    - The top-left 3x3 interior corner of each frame contains one gray/black glyph.
    - The remaining frame interior contains one connected red motif drawn only on one side of the future symmetry axis.

    Rule:
    - Detect the blue square frames and process them in top-to-bottom, then left-to-right order.
    - Read the gray pattern in each frame's interior top-left 3x3 area and decode it as either horizontal or vertical mirroring.
    - Mirror every red motif cell across the center line of that frame's interior and keep the union of original and mirrored red cells.
    - Remove the gray glyph from the output while preserving the blue frame and the black background.

    Key insight:
    - Because every frame is square, geometry alone does not reveal the mirror direction; the gray glyph is the only code for the axis.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _frame_bboxes(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        n = interior.shape[0]
        glyph_patch = interior[:3, :3]
        assert set(map(int, glyph_patch.flatten())).issubset({BLACK, GRAY}), "glyph area must be gray/black only"
        axis = GLYPH_TO_AXIS.get(_glyph_key(glyph_patch))
        assert axis is not None, "unknown glyph pattern"

        red_cells = [
            (int(rr), int(cc))
            for rr, cc in np.argwhere(interior == RED)
        ]
        assert red_cells, "each frame must contain a red motif"

        for rr, cc in _mirror_cells(red_cells, axis, n):
            out[r0 + 1 + rr, c0 + 1 + cc] = RED

        out[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4][glyph_patch == GRAY] = BLACK

    return out


def _allowed_seed_mask(interior_size: int, axis: str) -> np.ndarray:
    allowed = np.zeros((interior_size, interior_size), dtype=bool)
    if axis == HORIZONTAL:
        allowed[3:, : interior_size // 2] = True
    else:
        allowed[: interior_size // 2, 3:] = True
    return allowed


def _grow_connected_shape(allowed: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
    if len(cells) < size:
        return None

    for _ in range(80):
        seed = random.choice(cells)
        chosen = {seed}
        frontier = {nb for nb in _neighbors4(*seed) if 0 <= nb[0] < allowed.shape[0] and 0 <= nb[1] < allowed.shape[1] and allowed[nb]}
        while len(chosen) < size and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            chosen.add(cell)
            for nb in _neighbors4(*cell):
                if 0 <= nb[0] < allowed.shape[0] and 0 <= nb[1] < allowed.shape[1] and allowed[nb] and nb not in chosen:
                    frontier.add(nb)
        if len(chosen) != size:
            continue

        coords = np.array(sorted(chosen), dtype=int)
        if len(coords) >= 4:
            rspan = int(coords[:, 0].max() - coords[:, 0].min() + 1)
            cspan = int(coords[:, 1].max() - coords[:, 1].min() + 1)
            if rspan == 1 or cspan == 1:
                continue
        return [tuple(map(int, rc)) for rc in coords]
    return None


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_boxes(height: int, width: int, sides: list[int]) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for side in sorted(sides, reverse=True):
        for _ in range(200):
            r0 = random.randint(0, height - side)
            c0 = random.randint(0, width - side)
            box = (r0, c0, r0 + side - 1, c0 + side - 1)
            if all(_boxes_separated(box, old) for old in boxes):
                boxes.append(box)
                break
        else:
            return None
    return boxes


def _glyph_for_axis(axis: str) -> np.ndarray:
    if axis == HORIZONTAL:
        return GLYPH_HORIZONTAL
    return GLYPH_VERTICAL


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    axes = [HORIZONTAL, VERTICAL]
    if frame_count == 3:
        axes.append(random.choice([HORIZONTAL, VERTICAL]))
    random.shuffle(axes)

    sides = [random.randint(*SIDE_RANGE) for _ in range(frame_count)]
    min_side = max(sides)
    height = random.randint(max(GRID_SIZE_RANGE[0], min_side + 2, 22 if frame_count == 3 else 20), GRID_SIZE_RANGE[1])
    width = random.randint(max(GRID_SIZE_RANGE[0], min_side + 2, 22 if frame_count == 3 else 20), GRID_SIZE_RANGE[1])
    boxes = _place_boxes(height, width, sides)
    if boxes is None:
        return None

    solved = np.zeros((height, width), dtype=int)
    inp = np.zeros((height, width), dtype=int)

    for box, axis in zip(boxes, axes):
        _draw_frame(solved, box)
        _draw_frame(inp, box)
        r0, c0, r1, c1 = box
        n = r1 - r0 - 1
        allowed = _allowed_seed_mask(n, axis)
        max_cells = min(8, int(allowed.sum()))
        size = random.randint(3, max_cells)
        motif = _grow_connected_shape(allowed, size)
        if motif is None:
            return None
        mirrored = _mirror_cells(motif, axis, n)

        for rr, cc in motif + mirrored:
            solved[r0 + 1 + rr, c0 + 1 + cc] = RED
        for rr, cc in motif:
            inp[r0 + 1 + rr, c0 + 1 + cc] = RED

        glyph = _glyph_for_axis(axis)
        glyph_mask = glyph == GRAY
        inp[r0 + 1 : r0 + 4, c0 + 1 : c0 + 4][glyph_mask] = GRAY

    if not np.array_equal(main(inp), solved):
        return None
    return inp, solved


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input grid")
