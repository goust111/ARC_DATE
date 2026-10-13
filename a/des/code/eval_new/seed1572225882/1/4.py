import numpy as np


TEMPLATE_PATTERNS = [
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
]



def _bg_color(grid: np.ndarray) -> int:
    """Return the most frequent color in a grid."""
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _read_templates(grid: np.ndarray, bg: int):
    """
    Return `(top_row, [4 3x3 bool masks])` for the template panel.
    """
    h, w = grid.shape
    rows_with_color = np.where(np.any(grid != bg, axis=1))[0]
    if len(rows_with_color) == 0:
        return 0, None
    top = int(rows_with_color[0])
    if top + 3 > h:
        return top, None

    panel = grid[top : top + 3]
    best_score = -1
    best_start = None

    for start in range(0, w - 14):
        ok = True
        score = 0
        for i in range(4):
            l = start + i * 4
            block = panel[:, l : l + 3]
            if block.shape[1] != 3 or not np.any(block != bg):
                ok = False
                break
            score += int(np.count_nonzero(block != bg))
        if not ok:
            continue
        gap_cols = [start + 3, start + 7, start + 11]
        for g in gap_cols:
            if g >= w or np.any(panel[:, g] != bg):
                ok = False
                break
        if not ok:
            continue
        if score > best_score:
            best_score = score
            best_start = start

    if best_start is None:
        non_bg_cols = np.where(np.any(panel != bg, axis=0))[0]
        if len(non_bg_cols) < 15:
            return top, None
        start = int(non_bg_cols[0])
        if start + 15 > w:
            start = w - 15
        templates = []
        for i in range(4):
            masks = panel[:, start + i * 4 : start + i * 4 + 3] != bg
            if masks.shape != (3, 3):
                return top, None
            templates.append(masks)
        return top, templates

    templates = []
    for i in range(4):
        l = best_start + i * 4
        masks = panel[:, l : l + 3] != bg
        templates.append(masks)
    return top, templates


def _ring_info(window: np.ndarray, bg: int):
    """Identify a 5x5 C-ring and return (color, open_dir)."""
    if window.shape != (5, 5):
        return None
    if not np.all(window[1:4, 1:4] == bg):
        return None

    open_top = np.all(window[0, :] == bg)
    open_right = np.all(window[:, 4] == bg)
    open_bottom = np.all(window[4, :] == bg)
    open_left = np.all(window[:, 0] == bg)

    open_dirs = [open_top, open_right, open_bottom, open_left]
    if sum(open_dirs) != 1:
        return None

    border_mask = np.ones((5, 5), dtype=bool)
    border_mask[1:4, 1:4] = False
    border = window[border_mask]
    fg = border[border != bg]

    if fg.size != 11:
        return None
    if np.any(fg != fg[0]):
        return None

    return int(fg[0]), int(np.argmax(open_dirs))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each C-shaped ring cavity with the template that matches its opening direction.
    Input structure:
    - bg is a dominant background color, the top 3x3 template strip contains four 3x3 masks separated by one-column gaps (left to right: up, right, down, left), and below it a field of colored rings with one missing border side.
    - each ring is a 5x5 hollow frame with a 3x3 empty interior, and interior bg is empty before filling.
    Rule:
    - detect and extract the four 3x3 template masks from the top strip, keeping the left-to-right order as up/right/down/left.
    - locate each ring, determine its missing side (open direction) from its border pixels, and stamp the corresponding template into its 3x3 interior.
    - recolor template foreground pixels with the ring's own color while template background pixels stay unchanged.
    - clear the top 3-row template strip in the output, leaving the transformed rings and the rest unchanged.
    Key insight:
    - infer direction from frame geometry, then apply the position-encoded template library to recolor each ring's cavity by the ring color.
    """

    bg = _bg_color(grid)
    out = grid.copy()

    top_row, templates = _read_templates(grid, bg)
    if templates is None or len(templates) != 4:
        if top_row < out.shape[0]:
            out[top_row : min(top_row + 3, out.shape[0]), :] = bg
        return out

    h, w = out.shape
    for r in range(top_row + 4, h - 4):
        for c in range(0, w - 4):
            window = grid[r : r + 5, c : c + 5]
            info = _ring_info(window, bg)
            if info is None:
                continue
            color, open_dir = info
            inner = out[r + 1 : r + 4, c + 1 : c + 4]
            inner[templates[open_dir]] = color

    if top_row < out.shape[0]:
        out[top_row : min(top_row + 3, out.shape[0]), :] = bg
    return out


_rng = np.random.default_rng()


def generate_input() -> np.ndarray:
    """Generate random valid inputs for the C-ring templating task."""
    h = int(_rng.integers(12, 31))
    w = int(_rng.integers(17, 31))
    bg = 0
    grid = np.full((h, w), bg, dtype=int)

    left_pad = int(_rng.integers(0, 3))
    if left_pad + 14 >= w:
        left_pad = max(0, w - 15)

    top_strip_top = 0
    top_strip_h = 3

    # Build palette
    palette = list(range(1, 30))
    _rng.shuffle(palette)
    pattern_masks = [m.copy() for m in TEMPLATE_PATTERNS]
    _rng.shuffle(pattern_masks)
    chosen_masks = pattern_masks[:4]

    used = set()
    colors = []
    for _ in range(4):
        c = palette[len(colors)]
        while c in used:
            c = int(_rng.integers(1, 30))
        colors.append(c)
        used.add(c)

    # Templates in order: up, right, down, left
    for i in range(4):
        l = left_pad + i * 4
        grid[top_strip_top : top_strip_top + top_strip_h, l : l + 3][chosen_masks[i]] = colors[i]

    # Candidate slots for ring top-left coordinates in a 6-cell lattice (spacing 1 cell gap)
    ring_y_candidates = [y for y in range(top_strip_top + 4, h - 4) if len(range(y, h - 4, 6)) >= 1]
    ring_x_candidates = [x for x in range(0, w - 4) if len(range(x, w - 4, 6)) >= 1]
    if not ring_y_candidates:
        ring_y_candidates = [top_strip_top + 4]
    if not ring_x_candidates:
        ring_x_candidates = [0]

    max_cols = max(1, len(range(0, w - 4, 6)))
    max_rows = max(1, len(range(top_strip_top + 4, h - 4, 6)))

    cols = int(_rng.integers(1, max_cols + 1))
    rows = int(_rng.integers(1, max_rows + 1))

    start_x_candidates = [x for x in range(0, w - 4) if len(range(x, w - 4, 6)) >= cols]
    start_y_candidates = [
        y
        for y in range(top_strip_top + 4, h - 4)
        if len(range(y, h - 4, 6)) >= rows
    ]

    if not start_x_candidates:
        cols = 1
        start_x_candidates = [x for x in range(0, w - 4)] or [0]
    if not start_y_candidates:
        rows = 1
        start_y_candidates = [y for y in range(top_strip_top + 4, h - 4)] or [top_strip_top + 4]

    start_x = int(_rng.choice(start_x_candidates))
    start_y = int(_rng.choice(start_y_candidates))

    ys = [start_y + i * 6 for i in range(rows) if start_y + i * 6 <= h - 5]
    xs = [start_x + j * 6 for j in range(cols) if start_x + j * 6 <= w - 5]
    if len(xs) < cols:
        cols = len(xs)
    if len(ys) < rows:
        rows = len(ys)

    slot_positions = [(yy, xx) for yy in ys for xx in xs]
    total_slots = len(slot_positions)
    if total_slots == 0:
        return grid

    count = int(_rng.integers(1, total_slots + 1))
    remaining_colors = [c for c in palette if c not in used]
    if len(remaining_colors) < count:
        # Safety fallback; never triggers with current bounds.
        remaining_colors = list(range(1, 30))
        remaining_colors = [c for c in remaining_colors if c != bg]

    ring_colors = _rng.choice(remaining_colors, size=count, replace=False).tolist()
    selected = [slot_positions[i] for i in _rng.choice(len(slot_positions), size=count, replace=False)]
    opens = [0, 1, 2, 3]

    for (r, c), color in zip(selected, ring_colors):
        open_dir = int(_rng.choice(opens))
        # Draw outer 5x5 frame in `color`
        grid[r : r + 5, c : c + 5] = bg
        grid[r, c : c + 5] = color
        grid[r + 4, c : c + 5] = color
        grid[r : r + 5, c] = color
        grid[r : r + 5, c + 4] = color

        # Remove one side (opening)
        if open_dir == 0:
            grid[r, c : c + 5] = bg
        elif open_dir == 1:
            grid[r : r + 5, c + 4] = bg
        elif open_dir == 2:
            grid[r + 4, c : c + 5] = bg
        elif open_dir == 3:
            grid[r : r + 5, c] = bg

    return grid
