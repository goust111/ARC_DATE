import numpy as np


LEGEND_HEIGHT = 3
LEGEND_GAP_ROW = 1
ENTRY_WIDTH = 2
ENTRY_GAP = 1
MAX_RETRIES = 300
PLACEMENT_ATTEMPTS = 120

GRID_H_RANGE = (13, 21)
GRID_W_RANGE = (18, 28)
FRAME_COUNT_RANGE = (2, 5)
FRAME_SIZE_MIN = 4
FRAME_H_MAX = 8
FRAME_W_MAX = 9
EXTRA_LEGEND_MAX = 2

BG_COLOR = 0
COLOR_POOL = tuple(range(1, 10))
SOLID_TEMPLATE = np.ones((2, 2), dtype=bool)
CHECKER_A = np.array([[1, 0], [0, 1]], dtype=bool)
CHECKER_B = np.array([[0, 1], [1, 0]], dtype=bool)
VALID_TEMPLATES = (SOLID_TEMPLATE, CHECKER_A, CHECKER_B)


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _template_kind(template: np.ndarray) -> str:
    if np.array_equal(template, SOLID_TEMPLATE):
        return "solid"
    if np.array_equal(template, CHECKER_A) or np.array_equal(template, CHECKER_B):
        return "checker"
    raise AssertionError("legend template must be solid or checkerboard")


def _parse_legend(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= LEGEND_HEIGHT, "grid is too short to contain a bottom legend"
    label_row = g[h - LEGEND_HEIGHT]
    pattern_rows = g[h - 2 : h]

    mapping: dict[int, np.ndarray] = {}
    c = 0
    while c < w:
        color = int(label_row[c])
        if color == bg:
            c += 1
            continue
        c1 = c
        while c1 < w and int(label_row[c1]) == color:
            c1 += 1
        run = c1 - c
        assert run == ENTRY_WIDTH, f"legend label width must be {ENTRY_WIDTH}"
        template = pattern_rows[:, c : c + ENTRY_WIDTH] == color
        assert template.shape == (2, 2), "legend pattern must be 2x2"
        allowed = (pattern_rows[:, c : c + ENTRY_WIDTH] == bg) | (pattern_rows[:, c : c + ENTRY_WIDTH] == color)
        assert np.all(allowed), "legend template may only use its own color and background"
        _template_kind(template)
        assert color not in mapping, "legend colors must be unique"
        mapping[color] = template.copy()
        c = c1
    assert mapping, "legend mapping must be non-empty"
    return mapping


def _apply_template(out: np.ndarray, bbox: tuple[int, int, int, int], color: int, template: np.ndarray, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    ih, iw = interior.shape
    assert ih > 0 and iw > 0, "frame interior must be non-empty"
    interior[:, :] = bg
    rr = np.arange(ih) % 2
    cc = np.arange(iw) % 2
    mask = template[rr[:, None], cc[None, :]]
    interior[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame interior according to its color's bottom legend template.

    The last three rows form the legend band: a 1x2 color key above a 2x2
    template block. The template is either solid or checkerboard in the same
    color, and that exact 2x2 template is tiled across the matching frame's
    interior while the border and legend remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping = _parse_legend(g, bg)

    body = g[: g.shape[0] - LEGEND_HEIGHT]
    out = g.copy()
    found_frames = 0
    for color in np.unique(body):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(body == color):
            r0, c0, r1, c1 = _bbox(coords)
            crop = body[r0 : r1 + 1, c0 : c1 + 1] == color
            if not _is_rect_frame(crop):
                continue
            assert color in mapping, f"missing legend template for frame color {color}"
            _apply_template(out, (r0, c0, r1, c1), color, mapping[color], bg)
            found_frames += 1
    assert found_frames > 0, "expected at least one rectangular frame"
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _choose_frame_templates(frame_colors: list[int], rng: np.random.Generator) -> dict[int, np.ndarray]:
    n = len(frame_colors)
    assert n >= 2, "generator always uses both rule branches"
    n_solid = int(rng.integers(1, n))
    kinds = ["solid"] * n_solid + ["checker"] * (n - n_solid)
    rng.shuffle(kinds)
    templates: dict[int, np.ndarray] = {}
    for color, kind in zip(frame_colors, kinds):
        if kind == "solid":
            templates[color] = SOLID_TEMPLATE.copy()
        else:
            templates[color] = CHECKER_A.copy()
    assert {"solid", "checker"} <= {_template_kind(t) for t in templates.values()}
    return templates


def _random_extra_template(rng: np.random.Generator) -> np.ndarray:
    if bool(rng.integers(0, 2)):
        return SOLID_TEMPLATE.copy()
    return CHECKER_A.copy() if bool(rng.integers(0, 2)) else CHECKER_B.copy()


def _sample_bboxes(h: int, w: int, count: int, rng: np.random.Generator) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    body_h = h - LEGEND_HEIGHT - LEGEND_GAP_ROW
    if body_h < FRAME_SIZE_MIN:
        return None
    max_h = min(FRAME_H_MAX, body_h)
    max_w = min(FRAME_W_MAX, w)
    if max_h < FRAME_SIZE_MIN or max_w < FRAME_SIZE_MIN:
        return None

    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            fh = int(rng.integers(FRAME_SIZE_MIN, max_h + 1))
            fw = int(rng.integers(FRAME_SIZE_MIN, max_w + 1))
            if body_h - fh < 0 or w - fw < 0:
                continue
            r0 = int(rng.integers(0, body_h - fh + 1))
            c0 = int(rng.integers(0, w - fw + 1))
            box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_too_close(box, other, pad=1) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _legend_capacity(width: int) -> int:
    return (width + ENTRY_GAP) // (ENTRY_WIDTH + ENTRY_GAP)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    max_entries = min(len(COLOR_POOL), _legend_capacity(w))
    if max_entries < FRAME_COUNT_RANGE[0]:
        return None

    max_frames = min(FRAME_COUNT_RANGE[1], max_entries)
    n_frames = int(rng.integers(FRAME_COUNT_RANGE[0], max_frames + 1))
    max_extra = min(EXTRA_LEGEND_MAX, len(COLOR_POOL) - n_frames, max_entries - n_frames)
    n_extra = int(rng.integers(0, max_extra + 1))
    n_entries = n_frames + n_extra
    legend_width = n_entries * ENTRY_WIDTH + (n_entries - 1) * ENTRY_GAP
    left = int(rng.integers(0, w - legend_width + 1))

    palette = rng.permutation(np.asarray(COLOR_POOL, dtype=int)).tolist()
    frame_colors = [int(x) for x in palette[:n_frames]]
    extra_colors = [int(x) for x in palette[n_frames : n_frames + n_extra]]
    frame_templates = _choose_frame_templates(frame_colors, rng)

    legend_templates = {color: tmpl.copy() for color, tmpl in frame_templates.items()}
    for color in extra_colors:
        legend_templates[color] = _random_extra_template(rng)

    legend_colors = frame_colors + extra_colors
    rng.shuffle(legend_colors)
    bboxes = _sample_bboxes(h, w, n_frames, rng)
    if bboxes is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    for color, bbox in zip(frame_colors, bboxes):
        _draw_frame(grid, bbox, color)

    label_row = h - LEGEND_HEIGHT
    pattern_row = h - 2
    for i, color in enumerate(legend_colors):
        c0 = left + i * (ENTRY_WIDTH + ENTRY_GAP)
        grid[label_row, c0 : c0 + ENTRY_WIDTH] = color
        template = legend_templates[color]
        grid[pattern_row : pattern_row + 2, c0 : c0 + ENTRY_WIDTH] = np.where(template, color, BG_COLOR)

    expected = grid.copy()
    for color, bbox in zip(frame_colors, bboxes):
        _apply_template(expected, bbox, color, frame_templates[color], BG_COLOR)

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
