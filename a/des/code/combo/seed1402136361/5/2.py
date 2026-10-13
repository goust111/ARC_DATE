import random
from collections import Counter, deque

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _gray_color(grid: np.ndarray, bg: int) -> int:
    best_color = None
    best_score = None
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        if not comps:
            continue
        score = (max(len(comp) for comp in comps), sum(len(comp) for comp in comps))
        if best_score is None or score > best_score:
            best_color = color
            best_score = score
    assert best_color is not None and best_score is not None and best_score[0] > 1
    return int(best_color)


def _legend_info(grid: np.ndarray, bg: int, gray: int) -> tuple[int, dict[tuple[tuple[int, int], ...], tuple[int, int]]]:
    marker_cells = [(int(r), int(c), int(grid[r, c]))
                    for r, c in np.argwhere((grid != bg) & (grid != gray))]
    assert marker_cells
    marker_rows = {r for r, _c, _color in marker_cells}
    assert len(marker_rows) == 1
    marker_row = next(iter(marker_rows))

    template_comps = _components(grid[:marker_row, :] == gray)
    assert template_comps
    mapping: dict[tuple[tuple[int, int], ...], tuple[int, int]] = {}

    for mr, mc, color in sorted(marker_cells, key=lambda item: item[1]):
        candidates = []
        for comp in template_comps:
            r0, c0, r1, c1 = _bbox(comp)
            if r1 < mr and c0 <= mc <= c1:
                candidates.append(comp)
        assert len(candidates) == 1
        comp = candidates[0]
        r0, c0, r1, c1 = _bbox(comp)
        assert (c1 - c0 + 1) % 2 == 1
        center_c = c0 + (c1 - c0) // 2
        if mc < center_c:
            direction = -1
        elif mc > center_c:
            direction = 1
        else:
            direction = 0
        shape = _normalize(comp)
        assert shape not in mapping
        mapping[shape] = (color, direction)

    return marker_row + 1, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object using the matching legend marker color, move it
    horizontally according to the marker offset, and remove the top legend band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _mode_color(g)
    gray = _gray_color(g, bg)
    main_r0, legend_map = _legend_info(g, bg, gray)

    scene = g[main_r0:, :]
    h, w = scene.shape
    assert h > 0 and w > 0

    out = np.full((h, w), bg, dtype=int)
    obj_comps = _components(scene == gray)
    assert obj_comps

    for comp in obj_comps:
        shape = _normalize(comp)
        assert shape in legend_map
        color, direction = legend_map[shape]
        _r0, c0, _r1, c1 = _bbox(comp)
        if direction < 0:
            dc = -c0
        elif direction > 0:
            dc = (w - 1) - c1
        else:
            dc = 0

        for r, c in comp:
            rr, cc = r, c + dc
            assert 0 <= rr < h and 0 <= cc < w
            assert out[rr, cc] == bg
            out[rr, cc] = color

    return out


# --------------- generator ---------------

_GRAY = 5
_BASE_SHAPES = [
    ("###", ".#."),
    ("##.", ".##"),
    ("#..", "#..", "###"),
    ("###", "..#", "..#"),
    ("##.", "###"),
    ("#####", "..#.."),
    ("..#..", "#####"),
    ("##...", "#####"),
    ("#....", "#####"),
    ("###..", ".#...", ".#..."),
]
_MAX_ATTEMPTS = 400


def _parse_shape(rows: tuple[str, ...] | list[str]) -> tuple[tuple[int, int], ...]:
    cells = [(r, c) for r, row in enumerate(rows) for c, ch in enumerate(row) if ch == "#"]
    assert cells
    return _normalize(cells)


_BASE_SHAPES = [_parse_shape(shape) for shape in _BASE_SHAPES]


def _transform_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c)) for r, c in shape}
    h = 1 + max(r for r, _ in cells)
    w = 1 + max(c for _r, c in cells)
    if random.random() < 0.5:
        cells = {(r, w - 1 - c) for r, c in cells}
    if random.random() < 0.5:
        cells = {(h - 1 - r, c) for r, c in cells}
    return _normalize(list(cells))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _r, c in shape)
    return h, w


def _sample_templates() -> list[tuple[tuple[tuple[int, int], ...], int, int]]:
    n_templates = random.randint(3, 4)
    directions = [-1, 0, 1]
    if n_templates == 4:
        directions.append(random.choice(directions))
    random.shuffle(directions)

    templates: list[tuple[tuple[tuple[int, int], ...], int, int]] = []
    seen_shapes: set[tuple[tuple[int, int], ...]] = set()
    base_indices = list(range(len(_BASE_SHAPES)))
    random.shuffle(base_indices)
    for idx in base_indices:
        shape = _transform_shape(_BASE_SHAPES[idx])
        h, w = _shape_size(shape)
        if w not in (3, 5) or w % 2 == 0:
            continue
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        templates.append((shape, directions[len(templates)], 0))
        if len(templates) == n_templates:
            break
    assert len(templates) == n_templates
    return templates


def _distribute_positions(widths: list[int], left_margin: int, gaps: list[int]) -> list[int]:
    xs = [left_margin]
    for i in range(1, len(widths)):
        xs.append(xs[-1] + widths[i - 1] + gaps[i - 1])
    return xs


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    for dr, dc in shape:
        grid[r0 + dr, c0 + dc] = color


def _build_sample() -> tuple[np.ndarray, np.ndarray]:
    templates = _sample_templates()
    bg_choices = [c for c in range(10) if c != _GRAY]
    bg = random.choice(bg_choices)
    marker_colors = random.sample([c for c in range(10) if c not in {bg, _GRAY}], len(templates))
    templates = [(shape, direction, color) for (shape, direction, _), color in zip(templates, marker_colors)]

    widths = [_shape_size(shape)[1] for shape, _direction, _color in templates]
    heights = [_shape_size(shape)[0] for shape, _direction, _color in templates]
    band_h = max(heights)
    gaps = [random.randint(2, 4) for _ in range(len(templates) - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    legend_xs = _distribute_positions(widths, left_margin, gaps)
    min_width = legend_xs[-1] + widths[-1] + right_margin
    width = max(14, min_width + random.randint(0, 6))

    object_template_ids = list(range(len(templates)))
    object_template_ids.extend(random.choices(range(len(templates)), k=random.randint(0, 2)))
    random.shuffle(object_template_ids)

    object_heights = [_shape_size(templates[idx][0])[0] for idx in object_template_ids]
    top_gap = random.randint(1, 3)
    bottom_gap = random.randint(1, 3)
    row_gaps = [random.randint(1, 3) for _ in range(max(0, len(object_template_ids) - 1))]
    main_h = top_gap + bottom_gap + sum(object_heights) + sum(row_gaps) + random.randint(0, 4)

    legend_h = band_h + 1
    inp = np.full((legend_h + main_h, width), bg, dtype=int)
    out = np.full((main_h, width), bg, dtype=int)

    for x, (shape, direction, color) in zip(legend_xs, templates):
        sh, sw = _shape_size(shape)
        y = band_h - sh
        _paint_shape(inp, shape, y, x, _GRAY)
        marker_c = x + sw // 2 + direction
        assert x <= marker_c < x + sw
        inp[band_h, marker_c] = color

    row = top_gap
    for i, template_idx in enumerate(object_template_ids):
        shape, direction, color = templates[template_idx]
        sh, sw = _shape_size(shape)
        max_c = width - sw - 1
        assert max_c >= 1
        col = random.randint(1, max_c)
        _paint_shape(inp, shape, legend_h + row, col, _GRAY)

        if direction < 0:
            target_c = 0
        elif direction > 0:
            target_c = width - sw
        else:
            target_c = col
        _paint_shape(out, shape, row, target_c, color)

        row += sh
        if i < len(row_gaps):
            row += row_gaps[i]

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    try:
        inp, expected = _build_sample()
    except Exception:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
