import random
import numpy as np


BLACK = 0
GREY = 5
MAGENTA = 6
USABLE_COLORS = [1, 2, 3, 4, 7, 8, 9]
SIDE_TO_ROT_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert rows, "expected a full grey separator row"
    return int(rows[0])


def _extract_palette(top: np.ndarray) -> dict[int, np.ndarray]:
    mapping: dict[int, np.ndarray] = {}
    for pts in _components(top != BLACK):
        r0, c0, r1, c1 = _bbox(pts)
        sub = top[r0 : r1 + 1, c0 : c1 + 1]
        if sub.shape[0] < 3 or sub.shape[1] < 3:
            continue
        perimeter = _perimeter_mask(*sub.shape)
        border_vals = np.unique(sub[perimeter])
        if len(border_vals) != 1 or int(border_vals[0]) == BLACK:
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or np.any(inner == BLACK):
            continue
        mapping[int(border_vals[0])] = inner.copy()
    assert mapping, "palette must contain at least one keyed pattern"
    return mapping


def _marker_side(marker_rc: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = marker_rc
    h, w = shape
    if r == 0:
        return "top"
    if c == w - 1:
        return "right"
    if r == h - 1:
        return "bottom"
    if c == 0:
        return "left"
    raise AssertionError("marker must lie on a border")


def _tile_to_shape(pattern: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    ph, pw = pattern.shape
    reps_h = (out_h + ph - 1) // ph
    reps_w = (out_w + pw - 1) // pw
    tiled = np.tile(pattern, (reps_h, reps_w))
    return tiled[:out_h, :out_w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each target frame's border color to a palette entry, rotate the entry's
    interior pattern by the magenta marker side, and tile-fill the target cavity.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_row(g)
    palette = _extract_palette(g[:sep])

    out = g.copy()
    bottom = g[sep + 1 :]
    for pts in _components(bottom != BLACK):
        r0, c0, r1, c1 = _bbox(pts)
        sub = bottom[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        if h < 3 or w < 3:
            continue

        perimeter = _perimeter_mask(h, w)
        marker_locs = np.argwhere(sub == MAGENTA)
        if len(marker_locs) != 1:
            continue

        perim_vals = sub[perimeter]
        border_vals = [int(v) for v in perim_vals if int(v) != MAGENTA]
        if not border_vals:
            continue
        border_color = border_vals[0]
        if any(v != border_color for v in border_vals):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0 or np.any(inner != BLACK):
            continue

        side = _marker_side(tuple(map(int, marker_locs[0])), sub.shape)
        assert border_color in palette, "every target border must match a palette key"
        pattern = np.rot90(palette[border_color], SIDE_TO_ROT_K[side])
        fill = _tile_to_shape(pattern, h - 2, w - 2)

        gr0 = sep + 1 + r0
        gc0 = c0
        out[gr0 + 1 : gr0 + h - 1, gc0 + 1 : gc0 + w - 1] = fill
        mr, mc = map(int, marker_locs[0])
        out[gr0 + mr, gc0 + mc] = border_color

    return out


def _random_pattern(rng: random.Random) -> np.ndarray:
    while True:
        c1, c2 = rng.sample(USABLE_COLORS, 2)
        mask = np.zeros((3, 3), dtype=bool)
        count = rng.randint(2, 7)
        for idx in rng.sample(range(9), count):
            mask[idx // 3, idx % 3] = True
        rots = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rots) != 4:
            continue
        pattern = np.full((3, 3), c1, dtype=int)
        pattern[mask] = c2
        return pattern


def _make_palette_box(border_color: int, pattern: np.ndarray) -> np.ndarray:
    box = np.full((5, 5), border_color, dtype=int)
    box[1:4, 1:4] = pattern
    return box


def _make_target_box(
    border_color: int,
    inner_h: int,
    inner_w: int,
    side: str,
    rng: random.Random,
) -> np.ndarray:
    box = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    box[0, :] = border_color
    box[-1, :] = border_color
    box[:, 0] = border_color
    box[:, -1] = border_color
    if side == "top":
        box[0, rng.randint(1, inner_w)] = MAGENTA
    elif side == "right":
        box[rng.randint(1, inner_h), -1] = MAGENTA
    elif side == "bottom":
        box[-1, rng.randint(1, inner_w)] = MAGENTA
    elif side == "left":
        box[rng.randint(1, inner_h), 0] = MAGENTA
    else:
        raise ValueError(side)
    return box


def _rects_clear(
    rect_a: tuple[int, int, int, int],
    rect_b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    ar0, ac0, ar1, ac1 = rect_a
    br0, bc0, br1, bc1 = rect_b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rectangles(
    sizes: list[tuple[int, int]],
    region_h: int,
    region_w: int,
    rng: random.Random,
    margin: int = 1,
    gap: int = 1,
    attempts_per_rect: int = 300,
) -> list[tuple[int, int]] | None:
    placed: list[tuple[int, int] | None] = [None] * len(sizes)
    rects: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for idx in order:
        h, w = sizes[idx]
        max_r = region_h - margin - h
        max_c = region_w - margin - w
        if max_r < margin or max_c < margin:
            return None
        for _ in range(attempts_per_rect):
            r = rng.randint(margin, max_r)
            c = rng.randint(margin, max_c)
            rect = (r, c, r + h - 1, c + w - 1)
            if all(_rects_clear(rect, other, gap) for other in rects):
                placed[idx] = (r, c)
                rects.append(rect)
                break
        else:
            return None
    return [pos for pos in placed if pos is not None]


def _fill_target_in_output(
    out: np.ndarray,
    top: int,
    left: int,
    box_h: int,
    box_w: int,
    border_color: int,
    pattern: np.ndarray,
    side: str,
) -> None:
    rotated = np.rot90(pattern, SIDE_TO_ROT_K[side])
    fill = _tile_to_shape(rotated, box_h - 2, box_w - 2)
    out[top + 1 : top + box_h - 1, left + 1 : left + box_w - 1] = fill
    if side == "top":
        row = top
        col = None
        for c in range(left + 1, left + box_w - 1):
            if out[row, c] == MAGENTA:
                col = c
                break
        if col is None:
            raise AssertionError("missing marker")
        out[row, col] = border_color
    elif side == "right":
        col = left + box_w - 1
        row = None
        for r in range(top + 1, top + box_h - 1):
            if out[r, col] == MAGENTA:
                row = r
                break
        if row is None:
            raise AssertionError("missing marker")
        out[row, col] = border_color
    elif side == "bottom":
        row = top + box_h - 1
        col = None
        for c in range(left + 1, left + box_w - 1):
            if out[row, c] == MAGENTA:
                col = c
                break
        if col is None:
            raise AssertionError("missing marker")
        out[row, col] = border_color
    elif side == "left":
        col = left
        row = None
        for r in range(top + 1, top + box_h - 1):
            if out[r, col] == MAGENTA:
                row = r
                break
        if row is None:
            raise AssertionError("missing marker")
        out[row, col] = border_color
    else:
        raise ValueError(side)


def _gen_attempt(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    n_palette = rng.randint(2, 4)
    border_colors = rng.sample(USABLE_COLORS, n_palette)
    patterns = {color: _random_pattern(rng) for color in border_colors}

    n_targets = rng.randint(1, n_palette)
    target_colors = rng.sample(border_colors, n_targets)
    sides = rng.sample(["top", "right", "bottom", "left"], n_targets)

    max_inner = {1: 9, 2: 8, 3: 7, 4: 6}[n_targets]
    target_specs = []
    for color, side in zip(target_colors, sides):
        inner_h = rng.randint(3, max_inner)
        inner_w = rng.randint(3, max_inner)
        box = _make_target_box(color, inner_h, inner_w, side, rng)
        target_specs.append(
            {
                "color": color,
                "side": side,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "box": box,
            }
        )

    top_band_w = 5 * n_palette + (n_palette - 1)
    max_target_w = max(spec["box"].shape[1] for spec in target_specs)
    max_target_h = max(spec["box"].shape[0] for spec in target_specs)
    width_min = max(top_band_w + 2, max_target_w + 4, 16)
    width_max = max(width_min, 30)
    if width_min > 30:
        return None
    width = rng.randint(width_min, width_max)

    top_h = rng.randint(5, 9)
    bottom_h_min = max(max_target_h + 2, 10)
    bottom_h_max = max(bottom_h_min, 20)
    bottom_h = rng.randint(bottom_h_min, bottom_h_max)

    grid = np.full((top_h + 1 + bottom_h, width), BLACK, dtype=int)
    sep = top_h
    grid[sep, :] = GREY

    order = rng.sample(border_colors, len(border_colors))
    left_margin = rng.randint(1, width - top_band_w - 1)
    top_row = rng.randint(0, top_h - 5)
    for i, color in enumerate(order):
        c0 = left_margin + i * 6
        grid[top_row : top_row + 5, c0 : c0 + 5] = _make_palette_box(color, patterns[color])

    sizes = [tuple(spec["box"].shape) for spec in target_specs]
    positions = _place_rectangles(sizes, bottom_h, width, rng, margin=1, gap=1)
    if positions is None:
        return None

    for spec, (r, c) in zip(target_specs, positions):
        box = spec["box"]
        h, w = box.shape
        grid[sep + 1 + r : sep + 1 + r + h, c : c + w] = box
        spec["top"] = sep + 1 + r
        spec["left"] = c

    expected = grid.copy()
    for spec in target_specs:
        box_h, box_w = spec["box"].shape
        _fill_target_in_output(
            expected,
            int(spec["top"]),
            int(spec["left"]),
            int(box_h),
            int(box_w),
            int(spec["color"]),
            patterns[int(spec["color"])],
            str(spec["side"]),
        )

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
