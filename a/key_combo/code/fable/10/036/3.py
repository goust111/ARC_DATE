from __future__ import annotations

import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_height_separator_cols(grid: np.ndarray, bg: int) -> list[int]:
    cols: list[int] = []
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != bg and np.all(col == col[0]):
            cols.append(c)
    return cols


def _split_panels(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    bg = _get_bg(grid)
    sep_cols = _full_height_separator_cols(grid, bg)
    assert sep_cols, "expected full-height separator columns"
    sep_color = int(grid[0, sep_cols[0]])
    assert all(int(grid[0, c]) == sep_color for c in sep_cols)

    panels: list[np.ndarray] = []
    start = 0
    for c in sep_cols:
        panel = grid[:, start:c]
        assert panel.shape[1] > 0
        panels.append(panel.copy())
        start = c + 1
    last = grid[:, start:]
    assert last.shape[1] > 0
    panels.append(last.copy())

    widths = {panel.shape[1] for panel in panels}
    assert len(widths) == 1, "all panels must have equal width"
    assert len(panels) >= 2, "expected at least two panels"
    return panels, sep_color


def _infer_direction(dot_coords: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    rows = dot_coords[:, 0]
    cols = dot_coords[:, 1]
    matches: list[str] = []
    if np.all(rows < r0) and np.all((c0 <= cols) & (cols <= c1)):
        matches.append("up")
    if np.all(rows > r1) and np.all((c0 <= cols) & (cols <= c1)):
        matches.append("down")
    if np.all(cols < c0) and np.all((r0 <= rows) & (rows <= r1)):
        matches.append("left")
    if np.all(cols > c1) and np.all((r0 <= rows) & (rows <= r1)):
        matches.append("right")
    assert len(matches) == 1, "dots must lie on exactly one bbox side"
    return matches[0]


def _shift_panel_shape(panel: np.ndarray, shape_color: int, direction: str, steps: int) -> np.ndarray:
    out = np.zeros_like(panel)
    dr, dc = DIRS[direction]
    for r, c in np.argwhere(panel == shape_color):
        nr = int(r) + dr * steps
        nc = int(c) + dc * steps
        if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1]:
            out[nr, nc] = shape_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each panel's foreground shape by the count-and-side dots, then overlay the shifted shapes into one panel.

    Input structure:
    - Full-height separator columns split the grid into several equal-size panels on a shared background.
    - Each panel contains one unique foreground-shape color and one shared dot color.
    - In every panel, all dots lie strictly on one side of that shape's bounding box.

    Rule:
    - For each panel, count the dots to get a distance k and use the dot side of the shape bbox to get a direction.
    - Translate that panel's shape by k cells in that direction, clipping any shifted pixels that leave the panel canvas.
    - Overlay the translated shapes onto one output canvas in left-to-right panel order, with later panels overwriting earlier ones.
    - Discard the dots and separator columns from the output.

    Key insight:
    - The shared dot color is not a drawable object; it encodes a push vector where side gives direction and count gives distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    panels, sep_color = _split_panels(g)
    bg = _get_bg(g)

    color_sets: list[set[int]] = []
    for panel in panels:
        colors = set(map(int, np.unique(panel))) - {bg, sep_color}
        assert len(colors) == 2, "each panel must have exactly one shape color and one dot color"
        color_sets.append(colors)

    common = set.intersection(*color_sets)
    assert len(common) == 1, "expected one shared dot color across panels"
    dot_color = common.pop()

    shape_colors: list[int] = []
    for colors in color_sets:
        shape_color = next(iter(colors - {dot_color}))
        shape_colors.append(shape_color)
    assert len(set(shape_colors)) == len(shape_colors), "shape colors must be panel-unique"

    out = np.full(panels[0].shape, bg, dtype=int)
    for panel, shape_color in zip(panels, shape_colors):
        shape_mask = panel == shape_color
        dot_coords = np.argwhere(panel == dot_color)
        assert shape_mask.any() and dot_coords.size > 0
        box = _bbox(shape_mask)
        assert box is not None
        direction = _infer_direction(dot_coords, box)
        steps = int(dot_coords.shape[0])
        shifted = _shift_panel_shape(panel, shape_color, direction, steps)
        mask = shifted != bg
        out[mask] = shifted[mask]

    return out


def _random_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    rr, cc = b
    cells = [(r, c)]
    moves = [0] * abs(rr - r) + [1] * abs(cc - c)
    random.shuffle(moves)
    for axis in moves:
        if axis == 0:
            r += 1 if rr > r else -1
        else:
            c += 1 if cc > c else -1
        cells.append((r, c))
    return cells


def _make_connected_shape(h: int, w: int) -> np.ndarray:
    for _ in range(60):
        target = random.randint(max(4, (h * w + 1) // 2), h * w - 1)
        anchors = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        hub = (random.randrange(h), random.randrange(w))
        cells: set[tuple[int, int]] = set()
        for anchor in anchors:
            cells.update(_random_path(anchor, hub))

        while len(cells) < target:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) >= h * w:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _bbox(mask) == (0, 0, h - 1, w - 1):
            return mask

    mask = np.ones((h, w), dtype=bool)
    mask[h // 2, w // 2] = False
    return mask


def _place_dots(
    panel: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    count: int,
    dot_color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        cols = random.sample(list(range(c0, c1 + 1)), count)
        panel[r0 - 1, cols] = dot_color
    elif direction == "down":
        cols = random.sample(list(range(c0, c1 + 1)), count)
        panel[r1 + 1, cols] = dot_color
    elif direction == "left":
        rows = random.sample(list(range(r0, r1 + 1)), count)
        panel[rows, c0 - 1] = dot_color
    else:
        rows = random.sample(list(range(r0, r1 + 1)), count)
        panel[rows, c1 + 1] = dot_color


def _panel_shift_variant(panels: list[dict], order: list[int]) -> np.ndarray:
    H = panels[0]["panel"].shape[0]
    W = panels[0]["panel"].shape[1]
    out = np.zeros((H, W), dtype=int)
    for idx in order:
        info = panels[idx]
        shifted = _shift_panel_shape(
            info["panel"],
            info["shape_color"],
            info["direction"],
            info["count"],
        )
        mask = shifted != 0
        out[mask] = shifted[mask]
    return out


def _visible_color_counts(grid: np.ndarray) -> dict[int, int]:
    colors = [int(c) for c in np.unique(grid) if int(c) != 0]
    return {color: int(np.count_nonzero(grid == color)) for color in colors}


def _alt_output(panels: list[dict], order: list[int], overrides: dict[int, tuple[str, int]]) -> np.ndarray:
    H = panels[0]["panel"].shape[0]
    W = panels[0]["panel"].shape[1]
    out = np.zeros((H, W), dtype=int)
    for idx in order:
        info = panels[idx]
        direction, count = overrides.get(idx, (info["direction"], info["count"]))
        shifted = _shift_panel_shape(info["panel"], info["shape_color"], direction, count)
        mask = shifted != 0
        out[mask] = shifted[mask]
    return out


def _sample_panel_count() -> int:
    return 4 if random.random() < 0.6 else 3


def _sample_panel_shape_size(panel_h: int, panel_w: int, direction: str, count: int) -> tuple[int, int]:
    max_h = min(5, panel_h - 1)
    max_w = min(5, panel_w - 1)
    choices: list[tuple[int, int]] = []
    for h in range(3, max_h + 1):
        for w in range(3, max_w + 1):
            if direction in ("up", "down") and w >= count:
                choices.append((h, w))
            if direction in ("left", "right") and h >= count:
                choices.append((h, w))
    assert choices
    return random.choice(choices)


def _final_range(panel_h: int, panel_w: int, shape_h: int, shape_w: int, direction: str, count: int) -> tuple[range, range]:
    if direction == "up":
        row_range = range(1 - count, panel_h - shape_h - count + 1)
        col_range = range(0, panel_w - shape_w + 1)
    elif direction == "down":
        row_range = range(count, panel_h - shape_h + count)
        col_range = range(0, panel_w - shape_w + 1)
    elif direction == "left":
        row_range = range(0, panel_h - shape_h + 1)
        col_range = range(1 - count, panel_w - shape_w - count + 1)
    else:
        row_range = range(0, panel_h - shape_h + 1)
        col_range = range(count, panel_w - shape_w + count)
    assert len(row_range) > 0 and len(col_range) > 0
    return row_range, col_range


def _orig_top_left(final_r: int, final_c: int, direction: str, count: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return final_r - dr * count, final_c - dc * count


def _build_panel(
    panel_h: int,
    panel_w: int,
    shape_color: int,
    dot_color: int,
    direction: str,
    count: int,
    center_r: int,
    center_c: int,
    force_clip: bool,
) -> dict | None:
    shape_h, shape_w = _sample_panel_shape_size(panel_h, panel_w, direction, count)
    mask = _make_connected_shape(shape_h, shape_w)
    row_range, col_range = _final_range(panel_h, panel_w, shape_h, shape_w, direction, count)

    def score(fr: int, fc: int) -> tuple[int, float]:
        clip_penalty = 0
        if force_clip:
            fully_inside = 0 <= fr and fr + shape_h <= panel_h and 0 <= fc and fc + shape_w <= panel_w
            clip_penalty = 0 if not fully_inside else 4
        else:
            fully_inside = 0 <= fr and fr + shape_h <= panel_h and 0 <= fc and fc + shape_w <= panel_w
            clip_penalty = 0 if fully_inside else 4
        dist = abs((fr + shape_h / 2) - center_r) + abs((fc + shape_w / 2) - center_c)
        return clip_penalty, dist + random.random()

    candidates = [(fr, fc) for fr in row_range for fc in col_range]
    candidates.sort(key=lambda pos: score(pos[0], pos[1]))
    for final_r, final_c in candidates[:12]:
        orig_r, orig_c = _orig_top_left(final_r, final_c, direction, count)
        if not (0 <= orig_r <= panel_h - shape_h and 0 <= orig_c <= panel_w - shape_w):
            continue
        bbox = (orig_r, orig_c, orig_r + shape_h - 1, orig_c + shape_w - 1)
        if direction == "up" and orig_r < 1:
            continue
        if direction == "down" and bbox[2] >= panel_h - 1:
            continue
        if direction == "left" and orig_c < 1:
            continue
        if direction == "right" and bbox[3] >= panel_w - 1:
            continue

        panel = np.zeros((panel_h, panel_w), dtype=int)
        panel[orig_r : orig_r + shape_h, orig_c : orig_c + shape_w][mask] = shape_color
        if any(np.all(panel[:, c] == shape_color) for c in range(panel_w)):
            continue
        if any(np.all(panel[r, :] == shape_color) for r in range(panel_h)):
            continue
        _place_dots(panel, bbox, direction, count, dot_color)
        return {
            "panel": panel,
            "shape_color": shape_color,
            "direction": direction,
            "count": count,
            "final_top_left": (final_r, final_c),
            "bbox": bbox,
            "clip": not (0 <= final_r and final_r + shape_h <= panel_h and 0 <= final_c and final_c + shape_w <= panel_w),
        }
    return None


def _render_input(panels: list[dict], sep_color: int) -> np.ndarray:
    panel_h, panel_w = panels[0]["panel"].shape
    panel_count = len(panels)
    grid = np.zeros((panel_h, panel_count * panel_w + (panel_count - 1)), dtype=int)
    col = 0
    for idx, info in enumerate(panels):
        grid[:, col : col + panel_w] = info["panel"]
        col += panel_w
        if idx != panel_count - 1:
            grid[:, col] = sep_color
            col += 1
    return grid


def _validate_semantics(inp: np.ndarray, panels: list[dict]) -> bool:
    true_out = main(inp)
    counts = _visible_color_counts(true_out)
    if len(counts) != len(panels):
        return False
    if any(v < 2 for v in counts.values()):
        return False

    no_shift = _alt_output(panels, list(range(len(panels))), {i: (panels[i]["direction"], 0) for i in range(len(panels))})
    if np.array_equal(true_out, no_shift):
        return False

    reverse = _panel_shift_variant(panels, list(reversed(range(len(panels)))))
    if np.array_equal(true_out, reverse):
        return False

    opposite = _alt_output(
        panels,
        list(range(len(panels))),
        {i: (OPPOSITE[panels[i]["direction"]], panels[i]["count"]) for i in range(len(panels))},
    )
    if np.array_equal(true_out, opposite):
        return False

    for i in range(len(panels)):
        original = _alt_output(panels, list(range(len(panels))), {i: (panels[i]["direction"], 0)})
        if np.array_equal(true_out, original):
            return False
        if panels[i]["count"] > 1:
            shorter = _alt_output(
                panels,
                list(range(len(panels))),
                {i: (panels[i]["direction"], panels[i]["count"] - 1)},
            )
            if np.array_equal(true_out, shorter):
                return False
        wrong_side = _alt_output(
            panels,
            list(range(len(panels))),
            {i: (OPPOSITE[panels[i]["direction"]], panels[i]["count"])},
        )
        if np.array_equal(true_out, wrong_side):
            return False

    if any(np.all(true_out == panels[i]["shape_color"], axis=0).any() for i in range(len(panels))):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    panel_count = _sample_panel_count()
    panel_h = random.randint(6, 8)
    max_panel_w = min(8, (30 - (panel_count - 1)) // panel_count)
    min_panel_w = 5 if max_panel_w >= 5 else max_panel_w
    if min_panel_w < 5:
        return None
    panel_w = random.randint(min_panel_w, max_panel_w)

    palette = random.sample(list(range(1, 10)), panel_count + 2)
    sep_color = palette[0]
    dot_color = palette[1]
    shape_colors = palette[2:]

    if panel_count == 4:
        directions = list(DIRS)
        random.shuffle(directions)
    else:
        directions = random.sample(list(DIRS), panel_count)
        random.shuffle(directions)

    counts = [random.randint(1, 3) for _ in range(panel_count)]
    if max(counts) == 1:
        counts[random.randrange(panel_count)] = 2

    center_r = random.uniform(panel_h * 0.35, panel_h * 0.65)
    center_c = random.uniform(panel_w * 0.35, panel_w * 0.65)
    clip_index = random.randrange(panel_count) if random.random() < 0.3 else -1

    panels: list[dict] = []
    for idx in range(panel_count):
        info = _build_panel(
            panel_h=panel_h,
            panel_w=panel_w,
            shape_color=shape_colors[idx],
            dot_color=dot_color,
            direction=directions[idx],
            count=counts[idx],
            center_r=center_r + random.uniform(-0.8, 0.8),
            center_c=center_c + random.uniform(-0.8, 0.8),
            force_clip=(idx == clip_index),
        )
        if info is None:
            return None
        panels.append(info)

    inp = _render_input(panels, sep_color)
    try:
        if not _validate_semantics(inp, panels):
            return None
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
