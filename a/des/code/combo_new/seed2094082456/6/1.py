import random

import numpy as np


ORTH_DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HORIZONTAL_DIRS = {"left", "right"}
VERTICAL_DIRS = {"up", "down"}
ARROW_MASKS = {
    "up": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}
ARROW_COMPONENT_MASKS = {
    "up": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [0, 1],
            [1, 1],
            [0, 1],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [1, 0],
            [1, 1],
            [1, 0],
        ],
        dtype=bool,
    ),
}
GEN_RETRIES = 400
PLACEMENT_PADDING = 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    return comps


def _is_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 5 or w < 5:
        return False
    return bool(
        np.all(mask[0, :])
        and np.all(mask[-1, :])
        and np.all(mask[:, 0])
        and np.all(mask[:, -1])
        and not np.any(mask[1:-1, 1:-1])
    )


def _arrow_direction(mask: np.ndarray) -> str | None:
    if int(mask.sum()) != 4:
        return None
    for name, pattern in ARROW_COMPONENT_MASKS.items():
        if np.array_equal(mask, pattern):
            return name
    return None


def _choose_direction_from_clue(coords: np.ndarray, ih: int, iw: int) -> str:
    rows = coords[:, 0]
    cols = coords[:, 1]
    if int(rows.min()) == int(rows.max()):
        left_margin = int(cols.min())
        right_margin = int(iw - 1 - cols.max())
        if left_margin == right_margin:
            center = (iw - 1) / 2.0
            return "left" if float(cols.mean()) < center else "right"
        return "left" if left_margin < right_margin else "right"
    if int(cols.min()) == int(cols.max()):
        top_margin = int(rows.min())
        bottom_margin = int(ih - 1 - rows.max())
        if top_margin == bottom_margin:
            center = (ih - 1) / 2.0
            return "up" if float(rows.mean()) < center else "down"
        return "up" if top_margin < bottom_margin else "down"
    raise AssertionError("expected a one-cell-thick horizontal or vertical clue line")


def _scale_mask(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_shape
    scaled = np.zeros(out_shape, dtype=bool)
    src_h, src_w = mask.shape
    for r in range(src_h):
        rr0 = r * out_h // src_h
        rr1 = (r + 1) * out_h // src_h
        for c in range(src_w):
            if not mask[r, c]:
                continue
            cc0 = c * out_w // src_w
            cc1 = (c + 1) * out_w // src_w
            scaled[rr0:rr1, cc0:cc1] = True
    return scaled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each framed clue line with the matching scaled arrow chosen from same-color external arrows.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick rectangular hollow frames of different colors are scattered on the grid.
    - Inside each frame is a detached same-color clue made of a short horizontal row or vertical column.
    - Outside the frames are detached 3x3 same-color T-shaped arrows pointing up, down, left, or right.

    Rule:
    - For each frame, read the clue orientation: a horizontal clue chooses between left/right arrows, and a vertical clue chooses between up/down arrows.
    - Break that tie using the clue offset inside the frame: clues closer to the left, right, top, or bottom select the arrow pointing toward that side.
    - Scale the chosen 3x3 arrow to the frame interior and repaint that interior with the frame color, replacing the old clue and blank cells.
    - Keep the frames themselves and every external arrow unchanged.

    Key insight:
    - Color links arrows to frames, while the clue uses two visible signals at once: its row/column orientation picks the axis and its off-center position picks the final direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _color_components(g, bg)

    frames: list[dict] = []
    arrows_by_color: dict[int, set[str]] = {}
    for comp in comps:
        if _is_frame(comp["mask"]):
            frames.append(comp)
            continue
        direction = _arrow_direction(comp["mask"])
        if direction is not None:
            arrows_by_color.setdefault(int(comp["color"]), set()).add(direction)

    out = g.copy()
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        clue_coords = np.argwhere(interior == color)
        assert len(clue_coords) >= 2, "each frame needs an internal clue line"
        ih, iw = interior.shape
        direction = _choose_direction_from_clue(clue_coords, ih, iw)
        assert color in arrows_by_color and direction in arrows_by_color[color], "missing matching external arrow"

        fill = out[r0 + 1 : r1, c0 + 1 : c1]
        fill[:] = bg
        scaled = _scale_mask(ARROW_MASKS[direction], (ih, iw))
        fill[scaled] = color

    return out


def _draw_frame_border(patch: np.ndarray, color: int) -> None:
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color


def _sample_canvas_shape(n_frames: int) -> tuple[int, int]:
    low = 22 if n_frames == 2 else 24
    rows = random.randint(low, 30)
    cols = random.randint(low, 30)
    if random.random() < 0.65:
        cols = min(30, max(low, rows + random.choice([-3, -2, -1, 0, 1, 2, 3])))
    return rows, cols


def _target_directions(n_frames: int) -> list[str]:
    dirs = [random.choice(["left", "right"]), random.choice(["up", "down"])]
    if n_frames == 3:
        dirs.append(random.choice(list(ORTH_DIRS)))
    random.shuffle(dirs)
    return dirs


def _sample_horizontal_clue(iw: int, direction: str) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for length in range(2, min(4, iw - 2) + 1):
        for start in range(1, iw - length):
            left = start
            right = iw - length - start
            if right < 1:
                continue
            if direction == "left" and left < right:
                choices.append((length, start))
            if direction == "right" and right < left:
                choices.append((length, start))
    if not choices:
        raise AssertionError("no valid horizontal clue placement")
    return random.choice(choices)


def _sample_vertical_clue(ih: int, direction: str) -> tuple[int, int]:
    choices: list[tuple[int, int]] = []
    for length in range(2, min(4, ih - 2) + 1):
        for start in range(1, ih - length):
            top = start
            bottom = ih - length - start
            if bottom < 1:
                continue
            if direction == "up" and top < bottom:
                choices.append((length, start))
            if direction == "down" and bottom < top:
                choices.append((length, start))
    if not choices:
        raise AssertionError("no valid vertical clue placement")
    return random.choice(choices)


def _make_frame_patches(bg: int, color: int, direction: str) -> tuple[np.ndarray, np.ndarray]:
    if direction in HORIZONTAL_DIRS:
        ih = random.choice([3, 6, 9])
        iw = random.choice([6, 9])
    else:
        ih = random.choice([6, 9])
        iw = random.choice([3, 6, 9])

    inp = np.full((ih + 2, iw + 2), bg, dtype=int)
    out = np.full_like(inp, bg)
    _draw_frame_border(inp, color)
    _draw_frame_border(out, color)

    if direction in HORIZONTAL_DIRS:
        length, start = _sample_horizontal_clue(iw, direction)
        row = random.randint(1, ih - 2)
        inp[row + 1, start + 1 : start + 1 + length] = color
    else:
        length, start = _sample_vertical_clue(ih, direction)
        col = random.randint(1, iw - 2)
        inp[start + 1 : start + 1 + length, col + 1] = color

    scaled = _scale_mask(ARROW_MASKS[direction], (ih, iw))
    out[1:-1, 1:-1][scaled] = color
    return inp, out


def _make_arrow_patch(bg: int, color: int, direction: str) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    patch[ARROW_MASKS[direction]] = color
    return patch


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_objects(rows: int, cols: int, objects: list[dict]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(objects)), key=lambda idx: objects[idx]["input"].shape[0] * objects[idx]["input"].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(objects)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch = objects[idx]["input"]
        ph, pw = patch.shape
        candidates = [(r, c) for r in range(rows - ph + 1) for c in range(cols - pw + 1)]
        random.shuffle(candidates)
        chosen = None
        for r, c in candidates:
            box = (r, c, r + ph - 1, c + pw - 1)
            if any(_boxes_overlap(box, other, PLACEMENT_PADDING) for other in boxes):
                continue
            chosen = (r, c)
            boxes.append(box)
            break
        if chosen is None:
            return None
        placements[idx] = chosen

    return [p for p in placements if p is not None]


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    region = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != bg
    region[mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.choice([2, 3])
    rows, cols = _sample_canvas_shape(n_frames)
    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    frame_colors = random.sample(palette, n_frames)
    directions = _target_directions(n_frames)

    objects: list[dict] = []
    for color, direction in zip(frame_colors, directions):
        frame_inp, frame_out = _make_frame_patches(bg, color, direction)
        objects.append({"input": frame_inp, "output": frame_out})
        for arrow_direction in ("up", "down", "left", "right"):
            arrow = _make_arrow_patch(bg, color, arrow_direction)
            objects.append({"input": arrow, "output": arrow})

    placements = _place_objects(rows, cols, objects)
    if placements is None:
        return None

    inp = np.full((rows, cols), bg, dtype=int)
    out = np.full((rows, cols), bg, dtype=int)
    for obj, (r, c) in zip(objects, placements):
        _paint_patch(inp, obj["input"], r, c, bg)
        _paint_patch(out, obj["output"], r, c, bg)

    if _get_bg(inp) != bg:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
