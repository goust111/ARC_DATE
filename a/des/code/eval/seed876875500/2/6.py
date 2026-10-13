import random

import numpy as np


BACKGROUND = 0
TEMPLATE_COLOR = 5
SEPARATOR_COLOR = 9
MARKER_PALETTE = (1, 2, 3, 4, 7)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
GENERATE_RETRIES = 400

UP_PROFILES = {
    3: (
        (1, 1, 3),
        (1, 3, 3),
    ),
    5: (
        (1, 1, 3, 5, 5),
        (1, 3, 3, 5, 5),
        (1, 3, 5, 5, 3),
        (1, 3, 5, 5, 5),
    ),
}


def get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def rotate_mask_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, -turns)


def nearest_edge_direction(r: int, c: int, h: int, w: int) -> int:
    candidates = (
        (c, 0, 0, LEFT),
        (w - 1 - c, 0, 1, RIGHT),
        (r, 1, 0, UP),
        (h - 1 - r, 1, 1, DOWN),
    )
    return min(candidates)[3]


def has_horizontal_priority_tie(r: int, c: int, h: int, w: int) -> bool:
    top = r
    bottom = h - 1 - r
    left = c
    right = w - 1 - c
    best = min(top, bottom, left, right)
    return (left == best or right == best) and (top == best or bottom == best)


def infer_tip_direction(mask: np.ndarray) -> int:
    coords = np.argwhere(mask)
    center_r = (mask.shape[0] - 1) / 2.0
    center_c = (mask.shape[1] - 1) / 2.0
    dr = float(coords[:, 0].mean() - center_r)
    dc = float(coords[:, 1].mean() - center_c)

    if abs(dr) >= abs(dc) and abs(dr) > 1e-9:
        return UP if dr > 0 else DOWN
    if abs(dc) > 1e-9:
        return LEFT if dc > 0 else RIGHT

    edge_scores = {
        UP: 10 * int(mask[0].sum()) + int(mask[1].sum()) if mask.shape[0] > 1 else int(mask[0].sum()),
        DOWN: 10 * int(mask[-1].sum()) + int(mask[-2].sum()) if mask.shape[0] > 1 else int(mask[-1].sum()),
        LEFT: 10 * int(mask[:, 0].sum()) + int(mask[:, 1].sum()) if mask.shape[1] > 1 else int(mask[:, 0].sum()),
        RIGHT: 10 * int(mask[:, -1].sum()) + int(mask[:, -2].sum()) if mask.shape[1] > 1 else int(mask[:, -1].sum()),
    }
    return min(edge_scores, key=edge_scores.get)


def paint_stamp(canvas: np.ndarray, mask: np.ndarray, center_r: int, center_c: int, color: int) -> None:
    h, w = canvas.shape
    mr = mask.shape[0] // 2
    mc = mask.shape[1] // 2
    for r, c in np.argwhere(mask):
        rr = center_r + int(r) - mr
        cc = center_c + int(c) - mc
        if 0 <= rr < h and 0 <= cc < w:
            canvas[rr, cc] = color


def stamp_cells(mask: np.ndarray, center_r: int, center_c: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    mr = mask.shape[0] // 2
    mc = mask.shape[1] // 2
    for r, c in np.argwhere(mask):
        rr = center_r + int(r) - mr
        cc = center_c + int(c) - mc
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
    return cells


def make_up_mask(profile: tuple[int, ...]) -> np.ndarray:
    size = len(profile)
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    for r, width in enumerate(profile):
        start = center - width // 2
        mask[r, start : start + width] = True
    return mask


def extract_template_mask(grid: np.ndarray) -> tuple[np.ndarray, int]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == SEPARATOR_COLOR)]
    assert len(sep_rows) == 1
    sep = sep_rows[0]

    top = grid[:sep]
    grey = np.argwhere(top == TEMPLATE_COLOR)
    assert len(grey) > 0
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    mask = top[r0 : r1 + 1, c0 : c1 + 1] == TEMPLATE_COLOR
    assert mask.shape[0] % 2 == 1 and mask.shape[1] % 2 == 1
    return mask, sep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reuse the upper grey stamp for each lower marker.

    The stamp is rotated so its tip points to the nearest canvas edge.
    Ties prefer horizontal edges (left/right).
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    template, sep = extract_template_mask(g)
    source_dir = infer_tip_direction(template)

    canvas = g[sep + 1 :].copy()
    h, w = canvas.shape
    markers = sorted((int(r), int(c)) for r, c in np.argwhere(canvas != bg))

    for r, c in markers:
        color = int(canvas[r, c])
        target_dir = nearest_edge_direction(r, c, h, w)
        turns = (target_dir - source_dir) % 4
        rotated = rotate_mask_cw(template, turns)
        paint_stamp(canvas, rotated, r, c, color)

    return canvas


def _choose_template() -> tuple[np.ndarray, int]:
    size = random.choice(sorted(UP_PROFILES))
    mask = make_up_mask(random.choice(UP_PROFILES[size]))
    direction = random.randrange(4)
    rotated = rotate_mask_cw(mask, direction)
    if infer_tip_direction(rotated) != direction:
        return _choose_template()
    return rotated, direction


def _desired_directions(count: int) -> list[int]:
    dirs = [UP, RIGHT, DOWN, LEFT]
    random.shuffle(dirs)
    chosen = dirs[: min(count, 4)]
    while len(chosen) < count:
        chosen.append(random.choice(dirs))
    random.shuffle(chosen)
    return chosen


def _place_markers(
    canvas_h: int,
    canvas_w: int,
    template: np.ndarray,
    source_dir: int,
    directions: list[int],
) -> list[tuple[int, int, int]] | None:
    colors = list(random.sample(MARKER_PALETTE, len(directions)))
    tie_index = None
    horizontal_slots = [i for i, direction in enumerate(directions) if direction in (LEFT, RIGHT)]
    if horizontal_slots and random.random() < 0.45:
        tie_index = random.choice(horizontal_slots)

    rotated_masks = {
        direction: rotate_mask_cw(template, (direction - source_dir) % 4)
        for direction in (UP, RIGHT, DOWN, LEFT)
    }
    placed: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()

    order = list(range(len(directions)))
    if tie_index is not None:
        order.remove(tie_index)
        order.insert(0, tie_index)

    for idx in order:
        direction = directions[idx]
        require_tie = idx == tie_index
        candidates = [
            (r, c)
            for r in range(canvas_h)
            for c in range(canvas_w)
            if nearest_edge_direction(r, c, canvas_h, canvas_w) == direction
            and (not require_tie or has_horizontal_priority_tie(r, c, canvas_h, canvas_w))
        ]
        random.shuffle(candidates)

        mask = rotated_masks[direction]
        placed_here = False
        for r, c in candidates:
            if any(max(abs(r - pr), abs(c - pc)) <= 1 for pr, pc, _ in placed):
                continue
            cells = stamp_cells(mask, r, c, canvas_h, canvas_w)
            if len(cells) < max(3, int(mask.sum()) // 2):
                continue
            if cells & occupied:
                continue
            placed.append((r, c, colors[idx]))
            occupied |= cells
            placed_here = True
            break

        if not placed_here:
            return None

    return placed


def _render_expected(
    canvas_h: int,
    canvas_w: int,
    template: np.ndarray,
    source_dir: int,
    markers: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.full((canvas_h, canvas_w), BACKGROUND, dtype=int)
    for r, c, color in markers:
        canvas[r, c] = color

    out = canvas.copy()
    for r, c, color in markers:
        direction = nearest_edge_direction(r, c, canvas_h, canvas_w)
        rotated = rotate_mask_cw(template, (direction - source_dir) % 4)
        paint_stamp(out, rotated, r, c, color)
    return canvas, out


def _gen_attempt() -> np.ndarray | None:
    template, source_dir = _choose_template()
    template_h, template_w = template.shape

    canvas_h = random.randint(max(8, template_h + 3), 14)
    canvas_w = random.randint(max(9, template_w + 4), 15)
    marker_count = random.randint(2, 5)
    directions = _desired_directions(marker_count)
    markers = _place_markers(canvas_h, canvas_w, template, source_dir, directions)
    if markers is None:
        return None

    lower_inp, expected = _render_expected(canvas_h, canvas_w, template, source_dir, markers)

    top_h = random.randint(template_h + 2, template_h + 5)
    grid = np.full((top_h + 1 + canvas_h, canvas_w), BACKGROUND, dtype=int)
    grid[top_h, :] = SEPARATOR_COLOR

    top_r = random.randint(1, top_h - template_h - 1)
    left_c = random.randint(1, canvas_w - template_w - 1)
    grid[top_r : top_r + template_h, left_c : left_c + template_w][template] = TEMPLATE_COLOR
    grid[top_h + 1 :, :] = lower_inp

    if get_bg(grid) != BACKGROUND:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
