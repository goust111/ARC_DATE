import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, colors_present, get_bg, in_bounds
import random


def _is_frame_corner(frame_mask: np.ndarray, r: int, c: int) -> bool:
    h, w = frame_mask.shape
    up = r > 0 and frame_mask[r - 1, c]
    down = r + 1 < h and frame_mask[r + 1, c]
    left = c > 0 and frame_mask[r, c - 1]
    right = c + 1 < w and frame_mask[r, c + 1]
    return (up or down) and (left or right)


def main(grid: np.ndarray) -> np.ndarray:
    """Extend a solid rectangle's color along outward diagonals after they pass through frame corners.
    Input structure
    The grid has a dominant background, one solid rectangular foreground object, and one one-cell-thick orthogonal frame object in a different color.
    Rule
    From each outer corner of the solid rectangle, trace the outward diagonal; if the first non-background pixel on that ray is a corner of the frame, continue past that frame pixel and paint every remaining background pixel on the same diagonal with the rectangle color.
    Key insight
    The frame corner selects which rectangle-corner diagonals are active, and the interior/background part of the ray before the frame is left unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2

    bboxes = {}
    solid_colors = []
    for color in colors:
        mask = g == color
        box = bbox(mask)
        assert box is not None
        bboxes[color] = box
        if int(mask.sum()) == bbox_h(box) * bbox_w(box):
            solid_colors.append(color)
    assert len(solid_colors) == 1

    src_color = solid_colors[0]
    frame_color = next(color for color in colors if color != src_color)
    frame_mask = g == frame_color
    out = g.copy()

    r0, c0, r1, c1 = bboxes[src_color]
    corner_rays = [
        ((r0, c0), (-1, -1)),
        ((r0, c1), (-1, 1)),
        ((r1, c0), (1, -1)),
        ((r1, c1), (1, 1)),
    ]

    active_rays = 0
    for (r, c), (dr, dc) in corner_rays:
        rr, cc = r + dr, c + dc
        while in_bounds(g.shape, rr, cc) and g[rr, cc] == bg:
            rr += dr
            cc += dc

        if not in_bounds(g.shape, rr, cc):
            continue
        if g[rr, cc] != frame_color or not _is_frame_corner(frame_mask, rr, cc):
            continue

        active_rays += 1
        rr += dr
        cc += dc
        painted = 0
        while in_bounds(g.shape, rr, cc):
            assert g[rr, cc] == bg
            out[rr, cc] = src_color
            painted += 1
            rr += dr
            cc += dc
        assert painted > 0

    assert active_rays > 0
    return out

# --------------- generator (o5x) ---------------
# Canvas and object bounds.
GRID_N_MIN, GRID_N_MAX = 6, 12
FRAME_H_MIN, FRAME_H_MAX = 3, 7
FRAME_W_MIN, FRAME_W_MAX = 4, 6
SOLID_H_MIN, SOLID_H_MAX = 1, 5
SOLID_W_MIN, SOLID_W_MAX = 1, 4
TRAIL_LEN_MIN, TRAIL_LEN_MAX = 1, 5

# Fixed frame/solid geometry and placement padding.
SOLID_NEAR_OFFSET, SOLID_FAR_MARGIN = 2, 3
MIN_U_SPAN_FOR_SOLID, U_SIDE_MARGIN = 5, 1

# Color, dominance, and retry controls.
BG_COLOR, COLOR_COUNT_MINLEN, GEN_RETRY_LIMIT = 0, 10, 300
NON_BG_COLORS = range(1, 10)

# Config choice order: grid size, frame kind, orientation, frame height, frame width.
CHOICE_FIELD_ORDER = (2, 0, 1, 3, 4)


def _gen_solid_shape(solid):
    sr0, sc0, sr1, sc1 = solid
    return sr1 - sr0 + 1, sc1 - sc0 + 1


def _gen_trail_len(n, r, c, dr, dc):
    length = 0
    r += dr
    c += dc
    while 0 <= r < n and 0 <= c < n:
        length += 1
        r += dr
        c += dc
    return length


def _gen_valid_config(n, r0, c0, solid, active):
    sh, sw = _gen_solid_shape(solid)
    trails = [_gen_trail_len(n, r0 + rr, c0 + cc, dr, dc) for rr, cc, dr, dc in active]
    return (
        SOLID_H_MIN <= sh <= SOLID_H_MAX
        and SOLID_W_MIN <= sw <= SOLID_W_MAX
        and all(TRAIL_LEN_MIN <= t <= TRAIL_LEN_MAX for t in trails)
    )


def _gen_add_config(configs, kind, orient, n, h, w, r0, c0, solid, active):
    if _gen_valid_config(n, r0, c0, solid, active):
        configs.append((kind, orient, n, h, w, r0, c0, solid, active))


def _gen_l_configs(n, h, w):
    if h > n or w > n:
        return []
    configs = []
    last_r, last_c = h - 1, w - 1
    specs = [
        ("tl", 0, 0, (0, 0, h - SOLID_FAR_MARGIN, w - SOLID_FAR_MARGIN), [(last_r, last_c, 1, 1)]),
        ("tr", 0, n - w, (0, SOLID_NEAR_OFFSET, h - SOLID_FAR_MARGIN, last_c), [(last_r, 0, 1, -1)]),
        ("bl", n - h, 0, (SOLID_NEAR_OFFSET, 0, last_r, w - SOLID_FAR_MARGIN), [(0, last_c, -1, 1)]),
        ("br", n - h, n - w, (SOLID_NEAR_OFFSET, SOLID_NEAR_OFFSET, last_r, last_c), [(0, 0, -1, -1)]),
    ]
    for orient, r0, c0, solid, active in specs:
        _gen_add_config(configs, "L", orient, n, h, w, r0, c0, solid, active)
    return configs


def _gen_u_configs(n, h, w):
    if h > n or w > n:
        return []
    configs = []
    last_r, last_c = h - 1, w - 1
    if w >= MIN_U_SPAN_FOR_SOLID:
        for c0 in range(U_SIDE_MARGIN, n - w):
            for orient, r0, solid, active in [
                ("top", 0, (0, SOLID_NEAR_OFFSET, h - SOLID_FAR_MARGIN, w - SOLID_FAR_MARGIN), [(last_r, 0, 1, -1), (last_r, last_c, 1, 1)]),
                ("bottom", n - h, (SOLID_NEAR_OFFSET, SOLID_NEAR_OFFSET, last_r, w - SOLID_FAR_MARGIN), [(0, 0, -1, -1), (0, last_c, -1, 1)]),
            ]:
                _gen_add_config(configs, "U", orient, n, h, w, r0, c0, solid, active)
    if h >= MIN_U_SPAN_FOR_SOLID:
        for r0 in range(U_SIDE_MARGIN, n - h):
            for orient, c0, solid, active in [
                ("left", 0, (SOLID_NEAR_OFFSET, 0, h - SOLID_FAR_MARGIN, w - SOLID_FAR_MARGIN), [(0, last_c, -1, 1), (last_r, last_c, 1, 1)]),
                ("right", n - w, (SOLID_NEAR_OFFSET, SOLID_NEAR_OFFSET, h - SOLID_FAR_MARGIN, last_c), [(0, 0, -1, -1), (last_r, 0, 1, -1)]),
            ]:
                _gen_add_config(configs, "U", orient, n, h, w, r0, c0, solid, active)
    return configs


def _gen_frame_pixel_count(kind, orient, h, w):
    if kind == "L":
        return h + w - 1
    if orient in ("top", "bottom"):
        return 2 * h + w - 2
    return h + 2 * w - 2


def _gen_configs():
    configs = []
    for n in range(GRID_N_MIN, GRID_N_MAX + 1):
        for h in range(FRAME_H_MIN, FRAME_H_MAX + 1):
            for w in range(FRAME_W_MIN, FRAME_W_MAX + 1):
                for cfg in _gen_l_configs(n, h, w) + _gen_u_configs(n, h, w):
                    kind, orient, _, _, _, _, _, solid, _ = cfg
                    sh, sw = _gen_solid_shape(solid)
                    solid_count = sh * sw
                    frame_count = _gen_frame_pixel_count(kind, orient, h, w)
                    if n * n - frame_count - solid_count > max(frame_count, solid_count):
                        configs.append(cfg)
    return configs
GEN_CONFIGS = _gen_configs()

def _gen_filter_by_random_field(configs, field):
    value = random.choice(sorted({cfg[field] for cfg in configs}))
    return [cfg for cfg in configs if cfg[field] == value]


def _gen_choose_config():
    configs = GEN_CONFIGS
    for field in CHOICE_FIELD_ORDER:
        configs = _gen_filter_by_random_field(configs, field)
    return random.choice(configs)


def _gen_draw_frame(grid, kind, orient, r0, c0, h, w, color):
    last_r = r0 + h - 1
    last_c = c0 + w - 1
    if kind == "L":
        wall_r = last_r if orient in ("tl", "tr") else r0
        wall_c = last_c if orient in ("tl", "bl") else c0
        grid[wall_r, c0 : c0 + w] = color
        grid[r0 : r0 + h, wall_c] = color
    elif orient in ("top", "bottom"):
        wall_r = last_r if orient == "top" else r0
        grid[wall_r, c0 : c0 + w] = color
        grid[r0 : r0 + h, c0] = color
        grid[r0 : r0 + h, last_c] = color
    else:
        wall_c = last_c if orient == "left" else c0
        grid[r0, c0 : c0 + w] = color
        grid[last_r, c0 : c0 + w] = color
        grid[r0 : r0 + h, wall_c] = color


def _gen_paint_trails(out, n, r0, c0, active, color):
    trail_cells = []
    for rr, cc, dr, dc in active:
        r, c = r0 + rr + dr, c0 + cc + dc
        while 0 <= r < n and 0 <= c < n:
            out[r, c] = color
            trail_cells.append((r, c))
            r += dr
            c += dc
    return trail_cells


def _gen_render_pair(cfg, solid_color, frame_color):
    kind, orient, n, h, w, r0, c0, solid, active = cfg
    out = np.zeros((n, n), dtype=int)
    _gen_draw_frame(out, kind, orient, r0, c0, h, w, frame_color)

    sr0, sc0, sr1, sc1 = solid
    out[r0 + sr0 : r0 + sr1 + 1, c0 + sc0 : c0 + sc1 + 1] = solid_color

    trail_cells = _gen_paint_trails(out, n, r0, c0, active, solid_color)

    inp = out.copy()
    for r, c in trail_cells:
        inp[r, c] = BG_COLOR
    return inp, out


def _gen_pair_is_valid(inp, out, solid_color, frame_color):
    counts = np.bincount(inp.ravel(), minlength=COLOR_COUNT_MINLEN)
    return counts[BG_COLOR] > max(counts[solid_color], counts[frame_color]) and np.array_equal(main(inp), out)


def _gen_attempt():
    cfg = _gen_choose_config()
    solid_color, frame_color = random.sample(NON_BG_COLORS, 2)
    inp, out = _gen_render_pair(cfg, solid_color, frame_color)
    return inp if _gen_pair_is_valid(inp, out, solid_color, frame_color) else None


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate ec883f72 input")
