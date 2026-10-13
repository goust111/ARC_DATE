import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (1, 2)
PATTERN_COLOR = 3

GEN_GRID_ROWS = (15, 24)
GEN_GRID_COLS = (18, 28)
GEN_SMALL_INNER_H = (2, 5)
GEN_SMALL_INNER_W = (2, 5)
GEN_FRAME_GAP = (1, 4)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _rotate_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h = max(r for r, _ in cells) + 1
    return _normalize_shape(tuple((c, h - 1 - r) for r, c in cells))


def _shape_variants() -> list[tuple[tuple[int, int], ...]]:
    base = [
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ]
    variants = set()
    for shape in base:
        cur = _normalize_shape(shape)
        for _ in range(4):
            variants.add(cur)
            cur = _rotate_shape(cur)
    return sorted(variants)


SHAPES = _shape_variants()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _paint_shape(canvas: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for r, c in shape:
        canvas[top + r, left + c] = color


def _frame_info(coords: np.ndarray, color: int) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None
    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not local[0, :].all() or not local[-1, :].all() or not local[:, 0].all() or not local[:, -1].all():
        return None
    if h > 2 and w > 2 and local[1:-1, 1:-1].any():
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "coords": coords,
        "inner_shape": (h - 2, w - 2),
        "inner_area": (h - 2) * (w - 2),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            info = _frame_info(coords, color)
            if info is not None:
                frames.append(info)
    return frames


def _inner_patch(grid: np.ndarray, frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    return grid[r0 + 1 : r1, c0 + 1 : c1].copy()


def _scaled_source_pattern(grid: np.ndarray, source: dict, bg: int) -> np.ndarray:
    interior = _inner_patch(grid, source)
    pattern_bbox = _bbox(interior != bg)
    assert pattern_bbox is not None, "expected a non-background pattern in the source frame"
    pattern = _crop(interior, pattern_bbox)
    return np.kron(pattern, np.ones((2, 2), dtype=int))


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert len(frames) == 2, "expected exactly two hollow rectangular frames"
    areas = [frame["inner_area"] for frame in frames]
    assert areas[0] != areas[1], "expected frames with different interior areas"

    container = frames[0] if frames[0]["inner_area"] > frames[1]["inner_area"] else frames[1]
    source = frames[1] if container is frames[0] else frames[0]
    scaled = _scaled_source_pattern(g, source, bg)

    out = _crop(g, container["bbox"])
    out[1:-1, 1:-1] = bg
    inner_h, inner_w = container["inner_shape"]
    assert scaled.shape[0] <= inner_h and scaled.shape[1] <= inner_w, "scaled source must fit container interior"
    out[1 : 1 + scaled.shape[0], 1 : 1 + scaled.shape[1]] = scaled
    return out, source, container


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two framed interiors, upscale the smaller frame's pattern, and crop to the larger frame.

    Input structure:
    - The background is black.
    - Exactly two non-touching 1-cell-thick hollow rectangular frames appear, each in a different color.
    - Each frame encloses a small green pattern on black, and the larger frame's interior dimensions are exactly twice the smaller frame's interior dimensions.

    Rule:
    - Measure each frame by the area strictly inside its border and assign the larger one as the output container.
    - Extract the tight bounding box of the smaller frame's interior green pattern, including black cells inside that tight box.
    - Upscale that cropped pattern by a factor of two, place it at the top-left of the larger frame's interior, clear the rest of that interior to black, and discard the smaller frame.

    Key insight:
    - The size comparison does not choose a winner color directly; it assigns roles, so one frame contributes only its border while the other contributes only its cropped interior pattern.
    """
    out, _source, _container = _solve(grid)
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _choose_shape(max_h: int, max_w: int, *, prefer_smaller: bool, prefer_holey: bool) -> tuple[tuple[int, int], ...]:
    candidates = [shape for shape in SHAPES if _shape_dims(shape)[0] <= max_h and _shape_dims(shape)[1] <= max_w]
    assert candidates, "no shape fits the chosen interior"
    if prefer_smaller:
        smaller = [shape for shape in candidates if _shape_dims(shape) != (max_h, max_w)]
        if smaller:
            candidates = smaller
    if prefer_holey:
        holey = [shape for shape in candidates if len(shape) < _shape_dims(shape)[0] * _shape_dims(shape)[1]]
        if holey:
            candidates = holey
    return random.choice(candidates)


def _place_shape_interior(inner_h: int, inner_w: int, *, require_leading_margin: bool) -> tuple[tuple[tuple[int, int], ...], int, int]:
    shape = _choose_shape(
        inner_h,
        inner_w,
        prefer_smaller=require_leading_margin,
        prefer_holey=random.random() < 0.6,
    )
    shape_h, shape_w = _shape_dims(shape)
    row_choices = list(range(inner_h - shape_h + 1))
    col_choices = list(range(inner_w - shape_w + 1))
    placements = [(r, c) for r in row_choices for c in col_choices]
    if require_leading_margin:
        shifted = [(r, c) for r, c in placements if r > 0 or c > 0]
        if shifted:
            placements = shifted
    top, left = random.choice(placements)
    return shape, top, left


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_attempt() -> np.ndarray:
    small_h = random.randint(*GEN_SMALL_INNER_H)
    small_w = random.randint(*GEN_SMALL_INNER_W)
    big_h, big_w = 2 * small_h, 2 * small_w
    small_outer = (small_h + 2, small_w + 2)
    big_outer = (big_h + 2, big_w + 2)

    gap = random.randint(*GEN_FRAME_GAP)
    layout = random.choice(("horizontal", "vertical"))
    if layout == "horizontal":
        min_rows = max(big_outer[0], small_outer[0]) + 2
        min_cols = big_outer[1] + small_outer[1] + gap + 2
    else:
        min_rows = big_outer[0] + small_outer[0] + gap + 2
        min_cols = max(big_outer[1], small_outer[1]) + 2

    max_rows = min(30, max(min_rows, GEN_GRID_ROWS[0] + 5))
    max_cols = min(30, max(min_cols, GEN_GRID_COLS[0] + 8))
    if min_rows > max_rows or min_cols > max_cols:
        raise AssertionError("chosen dimensions exceed ARC limits")

    grid_h = random.randint(max(GEN_GRID_ROWS[0], min_rows), max_rows)
    grid_w = random.randint(max(GEN_GRID_COLS[0], min_cols), max_cols)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    colors = list(FRAME_COLORS)
    container_color = random.choice(colors)
    source_color = colors[0] if colors[1] == container_color else colors[1]

    specs = [
        {"role": "container", "outer": big_outer, "color": container_color},
        {"role": "source", "outer": small_outer, "color": source_color},
    ]
    first, second = random.sample(specs, 2)

    if layout == "horizontal":
        total_w = first["outer"][1] + gap + second["outer"][1]
        left_margin = random.randint(0, grid_w - total_w)
        first_r = random.randint(0, grid_h - first["outer"][0])
        second_r = random.randint(0, grid_h - second["outer"][0])
        first_c = left_margin
        second_c = left_margin + first["outer"][1] + gap
    else:
        total_h = first["outer"][0] + gap + second["outer"][0]
        top_margin = random.randint(0, grid_h - total_h)
        first_r = top_margin
        second_r = top_margin + first["outer"][0] + gap
        first_c = random.randint(0, grid_w - first["outer"][1])
        second_c = random.randint(0, grid_w - second["outer"][1])

    placements = {
        first["role"]: (first_r, first_c, first_r + first["outer"][0] - 1, first_c + first["outer"][1] - 1),
        second["role"]: (second_r, second_c, second_r + second["outer"][0] - 1, second_c + second["outer"][1] - 1),
    }
    assert _boxes_separated(placements["container"], placements["source"], pad=1)

    _draw_frame(grid, placements["container"], container_color)
    _draw_frame(grid, placements["source"], source_color)

    require_crop = random.random() < 0.75 and (small_h > 1 or small_w > 1)
    source_shape, src_top, src_left = _place_shape_interior(
        small_h,
        small_w,
        require_leading_margin=require_crop and (small_h > 1 or small_w > 1),
    )
    src_r0, src_c0, _src_r1, _src_c1 = placements["source"]
    _paint_shape(grid, source_shape, src_r0 + 1 + src_top, src_c0 + 1 + src_left, PATTERN_COLOR)

    container_shape, cont_top, cont_left = _place_shape_interior(
        big_h,
        big_w,
        require_leading_margin=(big_h > 2 or big_w > 2) and random.random() < 0.7,
    )
    cont_r0, cont_c0, _cont_r1, _cont_c1 = placements["container"]
    _paint_shape(grid, container_shape, cont_r0 + 1 + cont_top, cont_c0 + 1 + cont_left, PATTERN_COLOR)
    return grid


def _validate_instance(grid: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert _get_bg(g) == BACKGROUND

    frames = _find_frames(g, BACKGROUND)
    assert len(frames) == 2
    assert {frame["color"] for frame in frames} == set(FRAME_COLORS)
    assert frames[0]["inner_area"] != frames[1]["inner_area"]

    source = min(frames, key=lambda frame: frame["inner_area"])
    container = max(frames, key=lambda frame: frame["inner_area"])
    sh, sw = source["inner_shape"]
    ch, cw = container["inner_shape"]
    assert (ch, cw) == (2 * sh, 2 * sw)
    assert _boxes_separated(source["bbox"], container["bbox"], pad=1)

    for frame in frames:
        interior = _inner_patch(g, frame)
        green = interior == PATTERN_COLOR
        assert 2 <= int(green.sum()) <= 4
        assert np.all((interior == BACKGROUND) | green)
        assert len(_components(green)) == 1

    source_interior = _inner_patch(g, source)
    pattern_bbox = _bbox(source_interior == PATTERN_COLOR)
    assert pattern_bbox is not None
    pr0, pc0, pr1, pc1 = pattern_bbox
    assert 0 <= pr0 <= pr1 < sh and 0 <= pc0 <= pc1 < sw

    scaled = _scaled_source_pattern(g, source, BACKGROUND)
    assert scaled.shape[0] <= ch and scaled.shape[1] <= cw

    container_interior = _inner_patch(g, container)
    expected_interior = np.full((ch, cw), BACKGROUND, dtype=int)
    expected_interior[: scaled.shape[0], : scaled.shape[1]] = scaled
    assert not np.array_equal(container_interior, expected_interior), "container distractor must not already equal the answer"

    out = main(g)
    assert out.shape == (ch + 2, cw + 2)
    assert np.all(out[0, :] == container["color"])
    assert np.all(out[-1, :] == container["color"])
    assert np.all(out[:, 0] == container["color"])
    assert np.all(out[:, -1] == container["color"])
    assert np.array_equal(out[1:-1, 1:-1], expected_interior)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        try:
            _validate_instance(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
