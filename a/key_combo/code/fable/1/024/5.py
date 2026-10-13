from __future__ import annotations

import numpy as np


# Lightweight helpers adapted from the shared arc_utils toolbox.
BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
CHAIN_LENGTH_CHOICES = (1, 2, 3)
CHAIN_LENGTH_PROBS = (0.2, 0.45, 0.35)
BOX_SIDE_RANGE_BY_COUNT = {
    2: (6, 9),
    3: (5, 9),
    4: (5, 8),
    5: (5, 7),
}
BOX_GAP = 1
EDGE_MARGIN = 1
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(np.asarray(grid, dtype=int))))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(g == int(color)):
        rr = int(r)
        cc = int(c)
        if visited[rr, cc]:
            continue
        stack = [(rr, cc)]
        visited[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == int(color):
                    visited[nr, nc] = True
                    stack.append((nr, nc))
        coords_arr = np.array(coords, dtype=int)
        comps.append(
            {
                "color": int(color),
                "coords": coords_arr,
                "bbox": bbox_coords(coords_arr),
                "area": int(coords_arr.shape[0]),
            }
        )
    return comps


def _frame_from_component(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 3:
        return None

    expected_area = 4 * side_h - 4
    if int(comp["area"]) != expected_area:
        return None

    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    if not np.all(patch[0, :] == color):
        return None
    if not np.all(patch[-1, :] == color):
        return None
    if not np.all(patch[:, 0] == color):
        return None
    if not np.all(patch[:, -1] == color):
        return None

    interior = patch[1:-1, 1:-1].copy()
    if interior.size == 0 or np.any(interior == color):
        return None

    non_bg_inside = set(map(int, np.unique(interior))) - {int(bg)}
    if len(non_bg_inside) != 1:
        return None

    content_color = next(iter(non_bg_inside))
    return {
        "border_color": color,
        "content_color": int(content_color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "interior": interior,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames: list[dict] = []
    for color in colors_present(g, exclude={bg}):
        for comp in _components_for_color(g, int(color)):
            frame = _frame_from_component(g, bg, comp)
            if frame is not None:
                frames.append(frame)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the color-pointer chain from the unique solid row and output the final box interior.

    Input structure:
    - A noisy background grid contains exactly one row whose every cell has the same color.
    - Several hollow same-thickness square frames are scattered elsewhere, and each frame uses a unique border color.
    - Inside every frame is a monochrome pattern on the frame interior; some pattern colors match other frame-border colors.

    Rule:
    - Read the color of the unique solid row as the starting color.
    - Repeatedly locate the frame whose border has the current color, then replace the current color with that frame's interior pattern color.
    - Stop when no frame border matches the current color, and output the full interior crop of the last frame entered, excluding its border.

    Key insight:
    - A frame's content is not the final answer by itself: it is a visible pointer to the next frame, so the task is to resolve the entire border-color to content-color chain before cropping the terminal frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    solid_rows = [r for r in range(g.shape[0]) if np.all(g[r] == g[r, 0])]
    assert len(solid_rows) == 1, "expected exactly one solid-color row"
    start_color = int(g[solid_rows[0], 0])

    frames = _find_frames(g)
    by_border: dict[int, dict] = {}
    for frame in frames:
        border_color = int(frame["border_color"])
        assert border_color not in by_border, "frame border colors must be unique"
        by_border[border_color] = frame

    current = int(start_color)
    visited: set[int] = set()
    last_frame: dict | None = None
    while current in by_border:
        assert current not in visited, "pointer chain must terminate"
        visited.add(current)
        last_frame = by_border[current]
        current = int(last_frame["content_color"])

    assert last_frame is not None, "the solid-row color must point to at least one frame"
    return np.asarray(last_frame["interior"], dtype=int).copy()


def _random_connected_cells(side: int, area: int, rng: np.random.Generator) -> set[tuple[int, int]]:
    start = (int(rng.integers(0, side)), int(rng.integers(0, side)))
    cells = {start}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        frontier = list(dict.fromkeys(frontier))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return cells


def _make_inner_patch(side: int, color: int, rng: np.random.Generator) -> np.ndarray:
    patch = np.full((side, side), BG, dtype=int)
    mode = str(rng.choice(["poly", "block", "full"], p=[0.55, 0.25, 0.20]))
    area = side * side
    if mode == "full":
        patch[:, :] = int(color)
        return patch
    if mode == "block":
        block_h = int(rng.integers(1, side + 1))
        block_w = int(rng.integers(1, side + 1))
        if block_h == side and block_w == side:
            if side > 1:
                if rng.random() < 0.5:
                    block_h -= 1
                else:
                    block_w -= 1
        r0 = int(rng.integers(0, side - block_h + 1))
        c0 = int(rng.integers(0, side - block_w + 1))
        patch[r0 : r0 + block_h, c0 : c0 + block_w] = int(color)
        return patch

    max_area = max(1, area - 1)
    cap = max(1, int(np.ceil(area * 0.7)))
    target_area = int(rng.integers(1, min(max_area, cap) + 1))
    for r, c in _random_connected_cells(side, target_area, rng):
        patch[r, c] = int(color)
    return patch


def _bbox_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_boxes(
    h: int,
    w: int,
    entry_row: int,
    boxes: list[dict],
    rng: np.random.Generator,
) -> bool:
    order = sorted(range(len(boxes)), key=lambda i: int(boxes[i]["side"]), reverse=True)
    placed_bboxes: list[tuple[int, int, int, int]] = []
    for idx in order:
        side = int(boxes[idx]["side"])
        if h - side - EDGE_MARGIN < EDGE_MARGIN or w - side - EDGE_MARGIN < EDGE_MARGIN:
            return False
        success = False
        for _ in range(300):
            r0 = int(rng.integers(EDGE_MARGIN, h - side - EDGE_MARGIN + 1))
            c0 = int(rng.integers(EDGE_MARGIN, w - side - EDGE_MARGIN + 1))
            r1 = r0 + side - 1
            c1 = c0 + side - 1
            if not (r1 < entry_row - 1 or r0 > entry_row + 1):
                continue
            bbox = (r0, c0, r1, c1)
            if any(_bbox_overlaps(bbox, other, pad=BOX_GAP) for other in placed_bboxes):
                continue
            boxes[idx]["bbox"] = bbox
            placed_bboxes.append(bbox)
            success = True
            break
        if not success:
            return False
    return True


def _paint_box(grid: np.ndarray, bbox: tuple[int, int, int, int], border_color: int, interior: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = int(border_color)
    grid[r1, c0 : c1 + 1] = int(border_color)
    grid[r0 : r1 + 1, c0] = int(border_color)
    grid[r0 : r1 + 1, c1] = int(border_color)
    grid[r0 + 1 : r1, c0 + 1 : c1] = np.asarray(interior, dtype=int)


def _build_noise_mask(h: int, w: int, entry_row: int, boxes: list[dict]) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    mask[entry_row, :] = False
    for box in boxes:
        r0, c0, r1, c1 = box["bbox"]
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = False
    return mask


def _place_noise(
    grid: np.ndarray,
    entry_row: int,
    boxes: list[dict],
    noise_colors: list[int],
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    allowed = _build_noise_mask(h, w, entry_row, boxes)
    used: set[tuple[int, int]] = set()

    def pick_cell(candidates: list[tuple[int, int]]) -> tuple[int, int] | None:
        free = [cell for cell in candidates if cell not in used]
        if not free:
            return None
        return free[int(rng.integers(0, len(free)))]

    # Ensure each non-entry row is visibly mixed.
    for r in range(h):
        if r == entry_row:
            continue
        if np.unique(grid[r]).size > 1:
            continue
        row_cells = [(r, c) for c in range(w) if allowed[r, c]]
        cell = pick_cell(row_cells)
        if cell is None:
            return False
        used.add(cell)
        grid[cell] = int(rng.choice(noise_colors))

    # Ensure every terminal/noise color appears outside the boxes at least once.
    for color in noise_colors:
        free_cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed)]
        cell = pick_cell(free_cells)
        if cell is None:
            return False
        used.add(cell)
        grid[cell] = int(color)

    extra_count = int(rng.integers(3, max(4, 3 + len(noise_colors)) + 1))
    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(allowed) if tuple(map(int, rc)) not in used]
    rng.shuffle(free_cells)
    for cell in free_cells[:extra_count]:
        used.add(cell)
        grid[cell] = int(rng.choice(noise_colors))
    return True


def _sample_grid_shape(n_boxes: int, rng: np.random.Generator) -> tuple[int, int]:
    minimum = max(GRID_SIDE_MIN, 14 + 2 * n_boxes)
    h = int(rng.integers(minimum, GRID_SIDE_MAX + 1))
    w = int(rng.integers(minimum, GRID_SIDE_MAX + 1))
    return h, w


def _build_spec(rng: np.random.Generator) -> tuple[list[dict], int, list[int]]:
    chain_len = int(rng.choice(CHAIN_LENGTH_CHOICES, p=CHAIN_LENGTH_PROBS))
    max_distractors = (8 - chain_len) // 2
    distractors = int(rng.integers(1, max_distractors + 1))
    n_boxes = chain_len + distractors

    palette = list(map(int, rng.choice(PALETTE, size=n_boxes + distractors + 1, replace=False)))
    border_colors = palette[:n_boxes]
    terminal_colors = palette[n_boxes:]

    lo, hi = BOX_SIDE_RANGE_BY_COUNT[n_boxes]
    boxes: list[dict] = []
    for i in range(chain_len):
        border_color = int(border_colors[i])
        content_color = int(border_colors[i + 1]) if i + 1 < chain_len else int(terminal_colors[0])
        side = int(rng.integers(lo, hi + 1))
        inner_side = side - 2
        boxes.append(
            {
                "role": "chain",
                "border_color": border_color,
                "content_color": content_color,
                "side": side,
                "interior": _make_inner_patch(inner_side, content_color, rng),
            }
        )

    for j in range(distractors):
        border_color = int(border_colors[chain_len + j])
        content_color = int(terminal_colors[j + 1])
        side = int(rng.integers(lo, hi + 1))
        inner_side = side - 2
        boxes.append(
            {
                "role": "distractor",
                "border_color": border_color,
                "content_color": content_color,
                "side": side,
                "interior": _make_inner_patch(inner_side, content_color, rng),
            }
        )
    return boxes, chain_len, terminal_colors


def _validate_generated(
    grid: np.ndarray,
    entry_row: int,
    boxes: list[dict],
    chain_len: int,
) -> bool:
    if get_bg(grid) != BG:
        return False
    solid_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == grid[r, 0])]
    if solid_rows != [int(entry_row)]:
        return False
    if any(np.unique(grid[r]).size == 1 for r in range(grid.shape[0]) if r != entry_row):
        return False

    border_colors = [int(box["border_color"]) for box in boxes]
    if len(border_colors) != len(set(border_colors)):
        return False
    border_set = set(border_colors)
    chain_boxes = boxes[:chain_len]
    if any(int(box["content_color"]) in border_set for box in boxes[chain_len:]):
        return False
    if int(chain_boxes[-1]["content_color"]) in border_set:
        return False

    frames = _find_frames(grid)
    if len(frames) != len(boxes):
        return False
    frame_bboxes = {tuple(map(int, frame["bbox"])) for frame in frames}
    spec_bboxes = {tuple(map(int, box["bbox"])) for box in boxes}
    if frame_bboxes != spec_bboxes:
        return False

    expected = np.asarray(chain_boxes[-1]["interior"], dtype=int)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    boxes, chain_len, terminal_colors = _build_spec(rng)
    h, w = _sample_grid_shape(len(boxes), rng)
    entry_row = int(rng.integers(3, h - 3))

    if not _place_boxes(h, w, entry_row, boxes, rng):
        return None

    grid = np.full((h, w), BG, dtype=int)
    grid[entry_row, :] = int(boxes[0]["border_color"])
    for box in boxes:
        _paint_box(grid, tuple(box["bbox"]), int(box["border_color"]), np.asarray(box["interior"], dtype=int))

    if not _place_noise(grid, entry_row, boxes, [int(c) for c in terminal_colors], rng):
        return None
    if not _validate_generated(grid, entry_row, boxes, chain_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
