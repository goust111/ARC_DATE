import random
from collections import Counter, deque

import numpy as np


BG = 0
GEN_ATTEMPTS = 500
GRID_SIZE_RANGE = (13, 19)
K_RANGE = (3, 6)
GUIDE_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
OBJECT_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _paint_bars(
    out: np.ndarray,
    orientation: str,
    origin: tuple[int, int],
    guide_color: int,
    k: int,
) -> None:
    h, w = out.shape
    r0, c0 = origin
    if orientation == "vertical":
        top = r0 - (k - 1)
        bottom = r0 + (k - 1)
        for rr in range(top, bottom + 1):
            if 0 <= rr < h:
                if 0 <= c0 - k < w:
                    out[rr, c0 - k] = guide_color
                if 0 <= c0 + k < w:
                    out[rr, c0 + k] = guide_color
    else:
        left = c0 - (k - 1)
        right = c0 + (k - 1)
        for cc in range(left, right + 1):
            if 0 <= cc < w:
                if 0 <= r0 - k < h:
                    out[r0 - k, cc] = guide_color
                if 0 <= r0 + k < h:
                    out[r0 + k, cc] = guide_color


def _render_output(
    shape: tuple[int, int],
    bg: int,
    guide_color: int,
    orientation: str,
    center_segment: list[tuple[int, int]],
    origin: tuple[int, int],
    palette: list[int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    protected = np.zeros(shape, dtype=bool)
    for r, c in center_segment:
        protected[r, c] = True
        out[r, c] = guide_color

    rows, cols = np.indices(shape)
    r0, c0 = origin
    dist = np.maximum(np.abs(rows - r0), np.abs(cols - c0))
    for d, color in enumerate(palette):
        mask = (dist == d) & ~protected
        out[mask] = int(color)

    _paint_bars(out, orientation, origin, guide_color, len(palette))
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected at least one non-background color"

    scene = None
    for color in non_bg:
        mask = g == color
        full_rows = [int(r) for r in np.flatnonzero(np.all(mask, axis=1))]
        full_cols = [int(c) for c in np.flatnonzero(np.all(mask, axis=0))]
        if len(full_cols) == 2 and len(full_rows) != 2:
            row_counts = np.count_nonzero(mask, axis=1)
            center_rows = [r for r in range(h) if row_counts[r] > 2]
            if len(center_rows) != 1:
                continue
            center_row = int(center_rows[0])
            seg_cols = [c for c in np.flatnonzero(mask[center_row]) if c not in full_cols]
            if not seg_cols:
                continue
            if seg_cols != list(range(seg_cols[0], seg_cols[-1] + 1)):
                continue
            if len(seg_cols) % 2 != 1:
                continue
            center_col = int(seg_cols[len(seg_cols) // 2])
            if center_col * 2 != full_cols[0] + full_cols[1]:
                continue
            scene = {
                "bg": bg,
                "guide_color": int(color),
                "orientation": "vertical",
                "pair_positions": tuple(full_cols),
                "center_axis": center_row,
                "center_segment": [(center_row, int(c)) for c in seg_cols],
                "origin": (center_row, center_col),
            }
            break
        if len(full_rows) == 2 and len(full_cols) != 2:
            col_counts = np.count_nonzero(mask, axis=0)
            center_cols = [c for c in range(w) if col_counts[c] > 2]
            if len(center_cols) != 1:
                continue
            center_col = int(center_cols[0])
            seg_rows = [r for r in np.flatnonzero(mask[:, center_col]) if r not in full_rows]
            if not seg_rows:
                continue
            if seg_rows != list(range(seg_rows[0], seg_rows[-1] + 1)):
                continue
            if len(seg_rows) % 2 != 1:
                continue
            center_row = int(seg_rows[len(seg_rows) // 2])
            if center_row * 2 != full_rows[0] + full_rows[1]:
                continue
            scene = {
                "bg": bg,
                "guide_color": int(color),
                "orientation": "horizontal",
                "pair_positions": tuple(full_rows),
                "center_axis": center_col,
                "center_segment": [(int(r), center_col) for r in seg_rows],
                "origin": (center_row, center_col),
            }
            break

    assert scene is not None, "failed to identify guide structure"

    object_mask = (g != scene["bg"]) & (g != scene["guide_color"])
    comps = _components(object_mask)
    assert len(comps) == 1, "expected one 4-connected internal object"
    scene["object_coords"] = comps[0]

    palette: list[int] = []
    wrong_axis_ambiguous = False
    if scene["orientation"] == "vertical":
        for r in range(h):
            colors = sorted({int(v) for v in g[r] if int(v) not in (scene["bg"], scene["guide_color"])})
            assert len(colors) <= 1, "row scan must be unambiguous"
            if colors:
                palette.append(colors[0])
        for c in range(w):
            colors = {int(v) for v in g[:, c] if int(v) not in (scene["bg"], scene["guide_color"])}
            if len(colors) > 1:
                wrong_axis_ambiguous = True
    else:
        for c in range(w):
            colors = sorted({int(v) for v in g[:, c] if int(v) not in (scene["bg"], scene["guide_color"])})
            assert len(colors) <= 1, "column scan must be unambiguous"
            if colors:
                palette.append(colors[0])
        for r in range(h):
            colors = {int(v) for v in g[r] if int(v) not in (scene["bg"], scene["guide_color"])}
            if len(colors) > 1:
                wrong_axis_ambiguous = True

    assert len(palette) >= 1, "empty palette"
    scene["palette"] = palette
    scene["wrong_axis_ambiguous"] = wrong_axis_ambiguous
    return scene


def _wrong_axis_palette(grid: np.ndarray, scene: dict) -> list[int]:
    g = np.asarray(grid, dtype=int)
    bg = scene["bg"]
    guide = scene["guide_color"]
    palette: list[int] = []
    if scene["orientation"] == "vertical":
        for c in range(g.shape[1]):
            cells = [(r, c) for r in range(g.shape[0]) if int(g[r, c]) not in (bg, guide)]
            if cells:
                palette.append(int(g[cells[0][0], c]))
    else:
        for r in range(g.shape[0]):
            cells = [(r, c) for c in range(g.shape[1]) if int(g[r, c]) not in (bg, guide)]
            if cells:
                palette.append(int(g[r, cells[0][1]]))
    return palette


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the internal object's color order along the guide direction and re-encode it as concentric square rings.

    Input structure:
    - The background is black.
    - One guide color forms exactly two full parallel bars plus one centered perpendicular segment.
    - A single 4-connected multicolor object lies strictly between the parallel bars and never touches the guide.

    Rule:
    - If the full bar pair is vertical, scan the object from top to bottom; if the pair is horizontal, scan from left to right.
    - At each scanned row or column, record the unique non-background non-guide color that appears there, skipping empty positions, to obtain an ordered palette.
    - Erase the object, keep the perpendicular center segment fixed, and color Chebyshev rings around that segment's midpoint with palette[0], palette[1], ..., clipping at the grid boundary.
    - Move the two parallel guide bars so they sit immediately outside the outermost ring, with length matching that ring's side span; center-segment cells stay guide-colored if a ring would overlap them.

    Key insight:
    - The guide bars determine the reading axis and the expansion center, while the internal object's ordered projection supplies the ring palette and therefore the number of concentric layers.
    """
    scene = _analyze(np.asarray(grid, dtype=int))
    return _render_output(
        np.asarray(grid).shape,
        scene["bg"],
        scene["guide_color"],
        scene["orientation"],
        scene["center_segment"],
        scene["origin"],
        scene["palette"],
    )


def _sample_palette(k: int, guide_color: int) -> list[int]:
    choices = [c for c in OBJECT_COLORS if c != guide_color]
    n_unique = random.randint(3, min(5, k, len(choices)))
    unique = random.sample(choices, n_unique)
    palette = unique[:]
    while len(palette) < k:
        palette.append(random.choice(unique))
    for _ in range(100):
        random.shuffle(palette)
        if len(set(palette)) >= 3 and all(palette[i] != palette[i - 1] for i in range(1, len(palette))):
            return [int(v) for v in palette]
    return [int(v) for v in palette]


def _sample_widths(k: int, dense_object: bool) -> list[int]:
    if not dense_object:
        assert k >= 5
        return [1] * k
    widths = [1] * k
    target_area = random.randint(max(5, k + 1), min(10, 2 * k))
    extra = target_area - k
    for idx in random.sample(range(k), extra):
        widths[idx] = 2
    return widths


def _sample_segments(k: int, widths: list[int], min_c: int, max_c: int) -> list[int] | None:
    starts: list[int] = []
    first_max = max_c - widths[0] + 1
    if min_c > first_max:
        return None
    starts.append(random.randint(min_c, first_max))
    for i in range(1, k):
        prev_start = starts[-1]
        prev_end = prev_start + widths[i - 1] - 1
        low = max(min_c, prev_start - (widths[i] - 1))
        high = min(max_c - widths[i] + 1, prev_end)
        if low > high:
            return None
        starts.append(random.randint(low, high))
    return starts


def _build_vertical_scene(
    h: int,
    w: int,
    guide_color: int,
    k: int,
    clip_branch: bool,
    dense_object: bool,
) -> np.ndarray | None:
    d_options = [d for d in range(3, min(5, (w - 1) // 2) + 1)]
    if not d_options:
        return None
    d = random.choice(d_options)

    layouts: list[tuple[int, int, str, bool]] = []
    for center_row in range(h):
        for center_col in range(d, w - d):
            if min(center_row, h - 1 - center_row, center_col, w - 1 - center_col) < 2:
                continue
            fully_visible = (
                center_row >= k - 1
                and h - 1 - center_row >= k - 1
                and center_col >= k
                and w - 1 - center_col >= k
            )
            if center_row >= k + 2:
                layouts.append((center_row, center_col, "up", fully_visible))
            if center_row <= h - k - 3:
                layouts.append((center_row, center_col, "down", fully_visible))

    layouts = [layout for layout in layouts if layout[3] != clip_branch]
    if not layouts:
        return None
    center_row, center_col, side, _ = random.choice(layouts)

    left = center_col - d
    right = center_col + d
    usable_min = left + 2
    usable_max = right - 2
    if usable_min > usable_max:
        return None

    widths = _sample_widths(k, dense_object)
    if max(widths) > usable_max - usable_min + 1:
        return None
    starts = _sample_segments(k, widths, usable_min, usable_max)
    if starts is None:
        return None

    if side == "up":
        row_start_max = center_row - 2 - k + 1
        if row_start_max < 1:
            return None
        row_start = random.randint(1, row_start_max)
    else:
        row_start_min = center_row + 2
        row_start_max = h - 2 - k + 1
        if row_start_min > row_start_max:
            return None
        row_start = random.randint(row_start_min, row_start_max)

    palette = _sample_palette(k, guide_color)
    grid = np.zeros((h, w), dtype=int)
    grid[:, left] = guide_color
    grid[:, right] = guide_color
    grid[center_row, left + 1 : right] = guide_color

    for i in range(k):
        rr = row_start + i
        cc = starts[i]
        grid[rr, cc : cc + widths[i]] = palette[i]
    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _analyze(grid)
        palette = scene["palette"]
        if len(palette) < 3:
            return False
        if len(set(palette)) < 3:
            return False
        if not scene["wrong_axis_ambiguous"]:
            return False

        object_area = len(scene["object_coords"])
        if not (5 <= object_area <= 10):
            return False

        out = main(grid)
        wrong_palette = _wrong_axis_palette(grid, scene)
        if wrong_palette == palette:
            return False
        wrong_axis_out = _render_output(
            grid.shape,
            scene["bg"],
            scene["guide_color"],
            scene["orientation"],
            scene["center_segment"],
            scene["origin"],
            wrong_palette,
        )
        if np.array_equal(out, wrong_axis_out):
            return False

        if object_area > len(palette):
            area_palette = palette + [palette[-1]] * (object_area - len(palette))
            area_out = _render_output(
                grid.shape,
                scene["bg"],
                scene["guide_color"],
                scene["orientation"],
                scene["center_segment"],
                scene["origin"],
                area_palette,
            )
            if np.array_equal(out, area_out):
                return False

        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        orientation = random.choice(["vertical", "horizontal"])
        clip_branch = random.random() < 0.4
        dense_object = random.random() < 0.7
        h = random.randint(*GRID_SIZE_RANGE)
        w = random.randint(*GRID_SIZE_RANGE)
        if not dense_object:
            k = random.randint(5, K_RANGE[1])
        else:
            k = random.randint(*K_RANGE)
        guide_color = random.choice(GUIDE_COLORS)

        if orientation == "vertical":
            grid = _build_vertical_scene(h, w, guide_color, k, clip_branch, dense_object)
        else:
            grid_t = _build_vertical_scene(w, h, guide_color, k, clip_branch, dense_object)
            grid = None if grid_t is None else grid_t.T

        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
