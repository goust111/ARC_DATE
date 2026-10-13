import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

FRAME_SIZE = 7
TEMPLATE_SIZE = 5
SIDE_TO_ROTATION = {"top": 0, "right": 1, "bottom": 2, "left": 3}
FIXED_SIDE_COLORS = {"top": RED, "right": GREEN, "bottom": BLUE, "left": YELLOW}


def _rotations_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k=k)) for k in (1, 2, 3))


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _is_connected(mask: np.ndarray) -> bool:
    cells = [tuple(cell) for cell in np.argwhere(mask)]
    if not cells:
        return False
    seen = {cells[0]}
    stack = [cells[0]]
    cell_set = set(cells)
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cell_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _find_frame(grid: np.ndarray) -> tuple[int, int, dict[int, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, dict[int, int], np.ndarray]] = []
    for r0 in range(h - FRAME_SIZE + 1):
        for c0 in range(w - FRAME_SIZE + 1):
            sub = grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE]

            top = int(sub[0, 1])
            right = int(sub[1, -1])
            bottom = int(sub[-1, 0])
            left = int(sub[0, 0])
            side_colors = [top, right, bottom, left]
            if any(color == BACKGROUND for color in side_colors):
                continue
            if len(set(side_colors)) != 4:
                continue

            if not np.all(sub[0, 1:] == top):
                continue
            if not np.all(sub[1:, -1] == right):
                continue
            if not np.all(sub[-1, :-1] == bottom):
                continue
            if not np.all(sub[:-1, 0] == left):
                continue

            inner = sub[1:-1, 1:-1]
            if not np.isin(inner, [BACKGROUND, GREY]).all():
                continue
            template = inner == GREY
            grey_count = int(template.sum())
            if grey_count < 1 or not _is_connected(template):
                continue

            mapping = {
                top: SIDE_TO_ROTATION["top"],
                right: SIDE_TO_ROTATION["right"],
                bottom: SIDE_TO_ROTATION["bottom"],
                left: SIDE_TO_ROTATION["left"],
            }
            candidates.append((r0, c0, mapping, template))

    assert len(candidates) == 1, "expected exactly one valid color-coded frame"
    return candidates[0]


def _stamp_only_background(out: np.ndarray, center: tuple[int, int], template: np.ndarray) -> None:
    cr, cc = center
    for tr, tc in np.argwhere(template):
        rr = cr + int(tr) - 2
        cc2 = cc + int(tc) - 2
        if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1] and out[rr, cc2] == BACKGROUND:
            out[rr, cc2] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the frame-side colors into rotations and stamp the extracted template at matching scattered pixels.

    Input structure:
    - A black background contains one 7x7 rectangular frame whose four sides have distinct colors.
    - The 5x5 interior of that frame holds a grey template on black.
    - Elsewhere, isolated single pixels use one of the frame-side colors as a rotation key.

    Rule:
    - Locate the unique 7x7 colored frame and extract its 5x5 grey template.
    - Build a color-to-rotation map from the frame sides: top->0 degrees, right->90 degrees counterclockwise,
      bottom->180 degrees, left->90 degrees clockwise.
    - For every scattered pixel outside the frame, rotate the template by the mapped angle and place the 5x5 stamp
      centered on that pixel, filling only black cells with grey.
    - After all stamps are applied, erase the entire original 7x7 frame region to black.

    Key insight:
    - The frame is both the stencil source and a visible lookup table: each side color encodes the rotation to use for
      scattered pixels of that color.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, color_to_rotation, template = _find_frame(g)

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r0 <= r < r0 + FRAME_SIZE and c0 <= c < c0 + FRAME_SIZE:
                continue
            color = int(g[r, c])
            if color not in color_to_rotation:
                continue
            rotated = np.rot90(template, k=color_to_rotation[color])
            _stamp_only_background(out, (r, c), rotated)

    out[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = BACKGROUND
    return out


def _random_template(rng: random.Random) -> np.ndarray:
    for _ in range(400):
        mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)
        target_size = rng.randint(4, 8)
        start = (2, 2) if rng.random() < 0.5 else (rng.randrange(5), rng.randrange(5))
        mask[start] = True
        frontier = {start}

        while int(mask.sum()) < target_size:
            candidates = []
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < 5 and 0 <= nc < 5 and not mask[nr, nc]:
                        candidates.append((nr, nc))
            if not candidates:
                break
            cell = rng.choice(candidates)
            mask[cell] = True
            frontier.add(cell)

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if int(mask.sum()) != target_size:
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        if not _is_connected(mask):
            continue
        if not _rotations_distinct(mask):
            continue
        return mask
    raise RuntimeError("failed to generate asymmetric connected template")


def _stamp_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - 2, c - 2, r + 2, c + 2


def _bboxes_disjoint(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _choose_scattered_centers(
    rng: random.Random,
    h: int,
    w: int,
    frame_bbox: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]]:
    candidates = [(r, c) for r in range(2, h - 2) for c in range(2, w - 2)]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for cell in candidates:
        bbox = _stamp_bbox(cell)
        if not _bboxes_disjoint(bbox, frame_bbox):
            continue
        if any(abs(cell[0] - r) + abs(cell[1] - c) < 6 for r, c in chosen):
            continue
        if any(max(abs(cell[0] - r), abs(cell[1] - c)) < 5 for r, c in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    raise RuntimeError("failed to place scattered centers")


def _build_input_from_parts(
    h: int,
    w: int,
    frame_origin: tuple[int, int],
    template: np.ndarray,
    centers: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    r0, c0 = frame_origin

    grid[r0, c0] = FIXED_SIDE_COLORS["left"]
    grid[r0, c0 + 1 : c0 + FRAME_SIZE] = FIXED_SIDE_COLORS["top"]
    grid[r0 + 1 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = FIXED_SIDE_COLORS["right"]
    grid[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE - 1] = FIXED_SIDE_COLORS["bottom"]
    grid[r0 + 1 : r0 + FRAME_SIZE - 1, c0] = FIXED_SIDE_COLORS["left"]
    grid[r0 + 1 : r0 + FRAME_SIZE - 1, c0 + 1 : c0 + FRAME_SIZE - 1][template] = GREY

    for (r, c), color in zip(centers, colors):
        grid[r, c] = color
    return grid


def _validate_instance(
    inp: np.ndarray,
    template: np.ndarray,
    frame_origin: tuple[int, int],
    centers: list[tuple[int, int]],
    colors: list[int],
) -> bool:
    r0, c0 = frame_origin
    frame_bbox = (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1)
    if len(set(colors)) != 4:
        return False
    if not _rotations_distinct(template):
        return False
    if not _is_connected(template):
        return False
    for center in centers:
        bbox = _stamp_bbox(center)
        if not _bboxes_disjoint(bbox, frame_bbox):
            return False
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) < 6:
                return False
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) < 5:
                return False

    out = main(inp)
    if not np.all(out[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] == BACKGROUND):
        return False
    grey_outside_frame = np.argwhere(out == GREY)
    if len(grey_outside_frame) == 0:
        return False
    for (r, c), color in zip(centers, colors):
        if out[r, c] != color:
            return False
        rotated = np.rot90(template, k=SIDE_TO_ROTATION[[k for k, v in FIXED_SIDE_COLORS.items() if v == color][0]])
        for tr, tc in np.argwhere(rotated):
            rr = r + int(tr) - 2
            cc = c + int(tc) - 2
            if inp[rr, cc] == BACKGROUND and out[rr, cc] != GREY:
                return False
    return True


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(500):
        h = rng.randint(18, 24)
        w = rng.randint(18, 24)
        frame_origin = (rng.randint(0, 2), rng.randint(0, 2))
        template = _random_template(rng)
        frame_bbox = (
            frame_origin[0],
            frame_origin[1],
            frame_origin[0] + FRAME_SIZE - 1,
            frame_origin[1] + FRAME_SIZE - 1,
        )
        scattered_count = rng.randint(4, 8)
        try:
            centers = _choose_scattered_centers(rng, h, w, frame_bbox, scattered_count)
        except RuntimeError:
            continue

        colors = [RED, GREEN, BLUE, YELLOW]
        extras = [rng.choice(colors) for _ in range(scattered_count - 4)]
        colors = colors + extras
        rng.shuffle(colors)

        inp = _build_input_from_parts(h, w, frame_origin, template, centers, colors)
        if _validate_instance(inp, template, frame_origin, centers, colors):
            return inp

    raise RuntimeError("failed to generate valid input")
