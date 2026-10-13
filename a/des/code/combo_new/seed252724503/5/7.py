from __future__ import annotations

import random

import numpy as np


ORANGE = 7
BLUE = 1
RED = 2

ALL_COLORS = tuple(range(10))
GEN_BG_CHOICES = (0, 3, 4, 5, 6, 8, 9)
GEN_TEMPLATE_INNER_RANGE = (5, 7)
GEN_TEMPLATE_EXTRA_SHAPES = (2, 4)
GEN_ENVELOPE_COUNT_CHOICES = (2, 3, 4)
GEN_GRID_MIN = 18
GEN_GRID_MAX = 30
GEN_PLACEMENT_RETRIES = 200
GEN_LAYOUT_RETRIES = 120
GEN_ATTEMPTS = 400

SHAPES = (
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for cells in _components(grid == color):
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border_area = 2 * (h + w) - 4
        if len(cells) != border_area:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == color):
            continue
        if np.any(sub[~border] == color):
            continue
        frames.append((r0, c0, r1, c1))
    return frames


def _recover_template(template_inner: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    red_locs = np.argwhere(template_inner == RED)
    assert red_locs.shape[0] == 1, "template must contain exactly one red anchor"
    ar, ac = map(int, red_locs[0])
    h, w = template_inner.shape
    assert 0 < ar < h - 1 and 0 < ac < w - 1, "template anchor must be interior"
    neighbor_colors = {
        int(template_inner[nr, nc])
        for nr, nc in _neighbors4(ar, ac)
        if 0 <= nr < h and 0 <= nc < w
    }
    assert len(neighbor_colors) == 1, "template anchor color must be recoverable from its 4-neighbors"
    anchor_color = neighbor_colors.pop()
    assert anchor_color not in (bg, RED), "template anchor must recover to a visible pattern color"
    pattern = template_inner.copy()
    pattern[ar, ac] = anchor_color
    return pattern, (ar, ac)


def _stamp_pattern(
    out: np.ndarray,
    pattern: np.ndarray,
    anchor: tuple[int, int],
    inner_bbox: tuple[int, int, int, int],
    marker: tuple[int, int],
    bg: int,
) -> None:
    ir0, ic0, ir1, ic1 = inner_bbox
    ar, ac = anchor
    mr, mc = marker
    top = ir0 + mr - ar
    left = ic0 + mc - ac
    ph, pw = pattern.shape
    for pr in range(ph):
        tr = top + pr
        if tr < ir0 or tr > ir1:
            continue
        for pc in range(pw):
            tc = left + pc
            if tc < ic0 or tc > ic1:
                continue
            color = int(pattern[pr, pc])
            if color == bg:
                continue
            out[tr, tc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the orange-framed template into each blue-framed envelope by aligning their red anchor markers.

    Input structure:
    - The background is the dominant color.
    - One small orange rectangular frame contains a multicolor template pattern, with one red anchor pixel replacing a template cell.
    - Several larger blue rectangular frames act as envelopes; each envelope interior is otherwise background except for one red marker pixel.

    Rule:
    - Identify the unique orange frame and extract its interior template without the border.
    - Recover the template's true anchor color, record the red anchor position, and treat that position as part of the stamp pattern.
    - For each blue frame, stamp the template into the envelope so the template anchor lands on the envelope's red marker.
    - Clip any stamped cells that would fall outside the envelope interior, and preserve the orange/blue borders.

    Key insight:
    - The red pixels are not output colors: they form an alignment system linking the template anchor to each envelope marker, while cropping is determined only by the destination frame interior.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    orange_frames = _find_frames(g, ORANGE)
    blue_frames = _find_frames(g, BLUE)
    assert len(orange_frames) == 1, "expected exactly one orange template frame"
    assert blue_frames, "expected at least one blue envelope frame"

    out = g.copy()

    tr0, tc0, tr1, tc1 = orange_frames[0]
    template_inner = g[tr0 + 1 : tr1, tc0 + 1 : tc1]
    pattern, anchor = _recover_template(template_inner, bg)
    ar, ac = anchor
    out[tr0 + 1 + ar, tc0 + 1 + ac] = pattern[ar, ac]

    for r0, c0, r1, c1 in blue_frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        red_locs = np.argwhere(inner == RED)
        assert red_locs.shape[0] == 1, "each envelope must contain exactly one red marker"
        mr, mc = map(int, red_locs[0])
        _stamp_pattern(out, pattern, anchor, (r0 + 1, c0 + 1, r1 - 1, c1 - 1), (mr, mc), bg)

    return out


def _paint_shape(
    canvas: np.ndarray,
    shape: np.ndarray,
    color: int,
    top: int,
    left: int,
) -> None:
    h, w = shape.shape
    patch = canvas[top : top + h, left : left + w]
    patch[shape] = color


def _can_place_shape(canvas: np.ndarray, shape: np.ndarray, top: int, left: int, bg: int) -> bool:
    h, w = shape.shape
    if top < 0 or left < 0 or top + h > canvas.shape[0] or left + w > canvas.shape[1]:
        return False
    sub = canvas[top : top + h, left : left + w]
    return bool(np.all(sub[shape] == bg))


def _make_template_inner(
    rng: random.Random, bg: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], int]:
    h = rng.randint(*GEN_TEMPLATE_INNER_RANGE)
    w = rng.randint(*GEN_TEMPLATE_INNER_RANGE)
    inner = np.full((h, w), bg, dtype=int)
    palette = [c for c in ALL_COLORS if c not in (bg, ORANGE, BLUE, RED)]

    anchor_color = rng.choice(palette)
    ar = rng.randint(1, h - 2)
    ac = rng.randint(1, w - 2)
    for nr, nc in ((ar, ac), (ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)):
        inner[nr, nc] = anchor_color

    extension_options = []
    if ar - 2 >= 0:
        extension_options.append((ar - 2, ac))
    if ar + 2 < h:
        extension_options.append((ar + 2, ac))
    if ac - 2 >= 0:
        extension_options.append((ar, ac - 2))
    if ac + 2 < w:
        extension_options.append((ar, ac + 2))
    rng.shuffle(extension_options)
    for nr, nc in extension_options[: rng.randint(0, min(2, len(extension_options)))]:
        inner[nr, nc] = anchor_color

    extra_colors = [c for c in palette if c != anchor_color]
    rng.shuffle(extra_colors)
    used_colors: set[int] = {anchor_color}
    n_extra = rng.randint(*GEN_TEMPLATE_EXTRA_SHAPES)
    shapes = list(SHAPES)
    for idx in range(n_extra):
        color = extra_colors[idx % len(extra_colors)]
        used_colors.add(color)
        rng.shuffle(shapes)
        placed = False
        for shape in shapes:
            sh, sw = shape.shape
            candidates = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
            rng.shuffle(candidates)
            for top, left in candidates:
                cells = np.argwhere(shape)
                if any((top + int(rr), left + int(cc)) == (ar, ac) for rr, cc in cells):
                    continue
                if not _can_place_shape(inner, shape, top, left, bg):
                    continue
                _paint_shape(inner, shape, color, top, left)
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError("failed to place template extra shape")

    assert len(used_colors) >= 2, "template must be multicolor"
    restored = inner.copy()
    inner[ar, ac] = RED
    return inner, restored, (ar, ac), anchor_color


def _full_fit_ranges(
    inner_shape: tuple[int, int], pattern_shape: tuple[int, int], anchor: tuple[int, int]
) -> tuple[range, range] | None:
    ih, iw = inner_shape
    ph, pw = pattern_shape
    ar, ac = anchor
    row_start = ar
    row_stop = ih - (ph - ar) + 1
    col_start = ac
    col_stop = iw - (pw - ac) + 1
    if row_start >= row_stop or col_start >= col_stop:
        return None
    return range(row_start, row_stop), range(col_start, col_stop)


def _choose_marker(
    rng: random.Random,
    inner_shape: tuple[int, int],
    pattern_shape: tuple[int, int],
    anchor: tuple[int, int],
    want_full: bool,
) -> tuple[int, int]:
    ih, iw = inner_shape
    fit_ranges = _full_fit_ranges(inner_shape, pattern_shape, anchor)
    if want_full:
        assert fit_ranges is not None, "full-fit envelopes must admit the whole pattern"
        row_range, col_range = fit_ranges
        return rng.choice(list(row_range)), rng.choice(list(col_range))

    fit_set: set[tuple[int, int]] = set()
    if fit_ranges is not None:
        row_range, col_range = fit_ranges
        fit_set = {(r, c) for r in row_range for c in col_range}

    clipped = [(r, c) for r in range(ih) for c in range(iw) if (r, c) not in fit_set]
    assert clipped, "clipped envelopes must have at least one clipping marker choice"
    corner_like = [
        (r, c)
        for r, c in clipped
        if min(r, ih - 1 - r) <= 1 and min(c, iw - 1 - c) <= 1
    ]
    pool = corner_like if corner_like and rng.random() < 0.8 else clipped
    return rng.choice(pool)


def _make_envelope_patch(
    rng: random.Random,
    bg: int,
    pattern_shape: tuple[int, int],
    anchor: tuple[int, int],
    want_full: bool,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    ph, pw = pattern_shape
    if want_full:
        ih = rng.randint(ph, min(9, ph + 3))
        iw = rng.randint(pw, min(9, pw + 3))
    else:
        if rng.random() < 0.5:
            ih = rng.randint(max(3, ph - 3), max(3, ph - 1))
            iw = rng.randint(max(3, pw - 3), max(3, pw - 1))
        else:
            ih = rng.randint(ph, min(9, ph + 3))
            iw = rng.randint(pw, min(9, pw + 3))

    marker = _choose_marker(rng, (ih, iw), pattern_shape, anchor, want_full)
    patch = np.full((ih + 2, iw + 2), bg, dtype=int)
    patch[0, :] = BLUE
    patch[-1, :] = BLUE
    patch[:, 0] = BLUE
    patch[:, -1] = BLUE
    patch[1 + marker[0], 1 + marker[1]] = RED
    return patch, marker, (ih, iw)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int = 1) -> bool:
    return not (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _place_patches(
    rng: random.Random, shapes: list[tuple[int, int]], grid_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    order = sorted(range(len(shapes)), key=lambda idx: shapes[idx][0] * shapes[idx][1], reverse=True)
    boxes: list[tuple[int, int, int, int]] = []
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    for idx in order:
        ph, pw = shapes[idx]
        placed = False
        for _ in range(GEN_PLACEMENT_RETRIES):
            top = rng.randint(0, h - ph)
            left = rng.randint(0, w - pw)
            box = (top, left, top + ph - 1, left + pw - 1)
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            boxes.append(box)
            positions[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _apply_expected(
    grid: np.ndarray,
    template_pos: tuple[int, int],
    template_restored: np.ndarray,
    anchor: tuple[int, int],
    envelopes: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
    bg: int,
) -> np.ndarray:
    out = grid.copy()
    tt, tl = template_pos
    ar, ac = anchor
    out[tt + 1 + ar, tl + 1 + ac] = template_restored[ar, ac]
    for (top, left), marker, inner_shape in envelopes:
        ih, iw = inner_shape
        _stamp_pattern(
            out,
            template_restored,
            anchor,
            (top + 1, left + 1, top + ih, left + iw),
            marker,
            bg,
        )
    return out


def _branch_plan(rng: random.Random, env_count: int) -> list[bool]:
    if env_count == 2:
        plan = [True, False]
    elif env_count == 3:
        plan = [True, True, False] if rng.random() < 0.5 else [True, False, False]
    else:
        plan = [True, True, False, False]
    rng.shuffle(plan)
    return plan


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(GEN_BG_CHOICES)
    template_inner, template_restored, anchor, _anchor_color = _make_template_inner(rng, bg)
    ph, pw = template_inner.shape

    template_patch = np.full((ph + 2, pw + 2), bg, dtype=int)
    template_patch[0, :] = ORANGE
    template_patch[-1, :] = ORANGE
    template_patch[:, 0] = ORANGE
    template_patch[:, -1] = ORANGE
    template_patch[1:-1, 1:-1] = template_inner

    env_count = rng.choices(GEN_ENVELOPE_COUNT_CHOICES, weights=(3, 4, 1), k=1)[0]
    plan = _branch_plan(rng, env_count)
    envelope_specs: list[tuple[np.ndarray, tuple[int, int], tuple[int, int], bool]] = []
    for want_full in plan:
        patch, marker, inner_shape = _make_envelope_patch(rng, bg, (ph, pw), anchor, want_full)
        envelope_specs.append((patch, marker, inner_shape, want_full))

    patches = [template_patch] + [spec[0] for spec in envelope_specs]
    shapes = [patch.shape for patch in patches]
    total_area = sum(h * w for h, w in shapes)
    min_span = max(max(h for h, _ in shapes), max(w for _, w in shapes), int(np.ceil(np.sqrt(total_area * 1.35))))

    for _ in range(GEN_LAYOUT_RETRIES):
        grid_h = rng.randint(max(GEN_GRID_MIN, min_span), GEN_GRID_MAX)
        grid_w = rng.randint(max(GEN_GRID_MIN, min_span), GEN_GRID_MAX)
        placements = _place_patches(rng, shapes, (grid_h, grid_w))
        if placements is None:
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        for patch, (top, left) in zip(patches, placements, strict=True):
            ph2, pw2 = patch.shape
            grid[top : top + ph2, left : left + pw2] = patch

        values, counts = np.unique(grid, return_counts=True)
        if int(values[np.argmax(counts)]) != bg:
            continue

        expected_envs = [
            (placements[idx + 1], envelope_specs[idx][1], envelope_specs[idx][2])
            for idx in range(env_count)
        ]
        expected = _apply_expected(grid, placements[0], template_restored, anchor, expected_envs, bg)
        if np.count_nonzero(grid == RED) != env_count + 1:
            continue
        if np.count_nonzero(expected == RED) != 0:
            continue
        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-envelope puzzle")
