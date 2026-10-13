import random
from collections import deque

import numpy as np


BG = 0
SEP = 6

PANEL_COUNT = 2
PANEL_WIDTH_RANGE = (12, 14)
GRID_HEIGHT_RANGE = (15, 24)
MAX_GENERATION_ATTEMPTS = 300
PANEL_BUILD_ATTEMPTS = 120
STAMP_PLACEMENT_ATTEMPTS = 80

TEMPLATE_AREA_MIN = 7
TEMPLATE_AREA_MAX = 18
TEMPLATE_FILL_RATIO_MAX = 0.5
STAMP_FILL_RATIO_MIN = 0.5

_RNG = random.Random()

_STAMP_BASES = (
    np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    ),
)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each sparse template point with a centered copy of the smaller compact stamp inside the same panel.

    Input structure:
    - Black background, with one or more full-height magenta separator columns splitting the canvas into equal-width panels.
    - Each panel contains exactly two non-overlapping 4-connected monochrome objects that do not touch the panel boundary.
    - The larger object is a sparse template with more pixels; the smaller object is a compact stamp whose bounding box has a single center cell.

    Rule:
    - In every panel, identify the template as the larger component by pixel count and the stamp as the smaller one.
    - Remove both original objects, then for each template pixel place one centered copy of the stamp in column-major order (left-to-right, then top-to-bottom).
    - The copied pixels keep the stamp's color; overlapping copies simply overwrite earlier copies with the same color.
    - Keep the magenta separator columns unchanged.

    Key insight:
    - The larger object is not preserved as geometry; it is only a coordinate map of stamp centers, so the task is a per-panel stamping transform rather than ordinary object retention.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    bg = _most_common_color(g)
    sep_cols, sep_color = _separator_columns(g, bg)
    panels = _panel_spans(sep_cols, w)

    out = np.full_like(g, bg)
    out[:, sep_cols] = sep_color

    for c0, c1 in panels:
        comps = _panel_components(g, c0, c1, bg, sep_color)
        assert len(comps) == 2
        comps.sort(key=lambda comp: (comp["area"], comp["bbox"][0], comp["bbox"][1]))
        stamp, template = comps
        assert template["area"] > stamp["area"]

        stamp_offsets = _centered_offsets(stamp["coords"])
        template_points = sorted(
            ((int(r), int(c)) for r, c in template["coords"]),
            key=lambda rc: (rc[1], rc[0]),
        )

        for tr, tc in template_points:
            for dr, dc in stamp_offsets:
                rr = tr + dr
                cc = tc + dc
                if 0 <= rr < h and c0 <= cc <= c1:
                    out[rr, cc] = stamp["color"]

    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    panel_w = rng.randint(*PANEL_WIDTH_RANGE)
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = PANEL_COUNT * panel_w + (PANEL_COUNT - 1)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BG, dtype=int)
    sep_col = panel_w
    grid[:, sep_col] = SEP

    color_pool = [1, 2, 3, 4, 5, 7, 8, 9]
    panel_colors = rng.sample(color_pool, PANEL_COUNT * 2)

    spans = _panel_spans([sep_col], width)
    for idx, (c0, c1) in enumerate(spans):
        template_color = panel_colors[2 * idx]
        stamp_color = panel_colors[2 * idx + 1]
        built = _build_panel(rng, height, panel_w)
        if built is None:
            return None

        template_cells, stamp_mask, stamp_top_left = built
        for r, c in template_cells:
            grid[r, c0 + c] = template_color

        sr, sc = stamp_top_left
        for r, c in np.argwhere(stamp_mask):
            grid[sr + int(r), c0 + sc + int(c)] = stamp_color

    if not _validate_generated(grid):
        return None
    return grid


def _build_panel(
    rng: random.Random,
    panel_h: int,
    panel_w: int,
) -> tuple[set[tuple[int, int]], np.ndarray, tuple[int, int]] | None:
    for _ in range(PANEL_BUILD_ATTEMPTS):
        stamp_mask = _random_stamp(rng)
        sh, sw = stamp_mask.shape
        half_r = sh // 2
        half_c = sw // 2

        safe_r0 = 1 + half_r
        safe_r1 = panel_h - 2 - half_r
        safe_c0 = 1 + half_c
        safe_c1 = panel_w - 2 - half_c
        if safe_r0 > safe_r1 or safe_c0 > safe_c1:
            continue

        safe_area = (safe_r1 - safe_r0 + 1) * (safe_c1 - safe_c0 + 1)
        stamp_area = int(stamp_mask.sum())
        area_min = max(TEMPLATE_AREA_MIN, stamp_area + 1)
        area_max = min(TEMPLATE_AREA_MAX, safe_area // 2)
        if area_min > area_max:
            continue

        target_area = rng.randint(area_min, area_max)
        template_cells = _random_sparse_component(
            rng,
            safe_r0,
            safe_r1,
            safe_c0,
            safe_c1,
            target_area,
        )
        if template_cells is None:
            continue

        for _ in range(STAMP_PLACEMENT_ATTEMPTS):
            top = rng.randint(1, panel_h - sh - 1)
            left = rng.randint(1, panel_w - sw - 1)
            stamp_cells = {(top + int(r), left + int(c)) for r, c in np.argwhere(stamp_mask)}
            if _touching_or_overlapping(template_cells, stamp_cells):
                continue
            return template_cells, stamp_mask, (top, left)

    return None


def _random_stamp(rng: random.Random) -> np.ndarray:
    mask = np.array(rng.choice(_STAMP_BASES), dtype=int)
    k = rng.randrange(4)
    mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _random_sparse_component(
    rng: random.Random,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    target_area: int,
) -> set[tuple[int, int]] | None:
    cells = {(rng.randint(r0, r1), rng.randint(c0, c1))}

    for _ in range(target_area * 10):
        if len(cells) == target_area:
            break

        br0, bc0, br1, bc1 = _bbox(cells)
        weights: dict[tuple[int, int], int] = {}
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if nr < r0 or nr > r1 or nc < c0 or nc > c1 or (nr, nc) in cells:
                    continue
                expands_bbox = nr < br0 or nr > br1 or nc < bc0 or nc > bc1
                weights[(nr, nc)] = max(weights.get((nr, nc), 0), 3 if expands_bbox else 1)

        if not weights:
            return None

        choices = list(weights)
        pick = rng.choices(choices, weights=[weights[cell] for cell in choices], k=1)[0]
        cells.add(pick)

    if len(cells) != target_area:
        return None

    br0, bc0, br1, bc1 = _bbox(cells)
    bbox_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    if bbox_area <= 0:
        return None
    if len({r for r, _ in cells}) < 3 or len({c for _, c in cells}) < 3:
        return None
    if target_area / bbox_area > TEMPLATE_FILL_RATIO_MAX:
        return None
    if bbox_area < target_area + 4:
        return None

    return cells


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    bg = _most_common_color(grid)
    if bg != BG:
        return False

    try:
        sep_cols, sep_color = _separator_columns(grid, bg)
        if sep_color != SEP:
            return False
        panels = _panel_spans(sep_cols, grid.shape[1])
        if len(panels) != PANEL_COUNT:
            return False

        for c0, c1 in panels:
            comps = _panel_components(grid, c0, c1, bg, sep_color)
            if len(comps) != 2:
                return False
            comps.sort(key=lambda comp: comp["area"])
            stamp, template = comps
            if template["area"] <= stamp["area"]:
                return False

            sr0, sc0, sr1, sc1 = stamp["bbox"]
            sh = sr1 - sr0 + 1
            sw = sc1 - sc0 + 1
            if sh % 2 == 0 or sw % 2 == 0:
                return False
            stamp_cells = {tuple(map(int, rc)) for rc in stamp["coords"]}
            center = (sr0 + sh // 2, sc0 + sw // 2)
            if center not in stamp_cells:
                return False
            if stamp["area"] / (sh * sw) < STAMP_FILL_RATIO_MIN:
                return False

            tr0, tc0, tr1, tc1 = template["bbox"]
            bbox_area = (tr1 - tr0 + 1) * (tc1 - tc0 + 1)
            if template["area"] / bbox_area > TEMPLATE_FILL_RATIO_MAX:
                return False

            half_r = sh // 2
            half_c = sw // 2
            for r, c in template["coords"]:
                if r - half_r < 0 or r + half_r >= grid.shape[0]:
                    return False
                if c - half_c < c0 or c + half_c > c1:
                    return False

        out = main(grid)
        if out.shape != grid.shape:
            return False
        if np.array_equal(out, grid):
            return False
    except AssertionError:
        return False

    return True


def _separator_columns(grid: np.ndarray, bg: int) -> tuple[list[int], int]:
    h, w = grid.shape
    sep_cols: list[int] = []
    sep_colors: set[int] = set()
    for c in range(w):
        col = grid[:, c]
        value = int(col[0])
        if value == bg:
            continue
        if bool(np.all(col == value)):
            sep_cols.append(c)
            sep_colors.add(value)

    assert sep_cols
    assert len(sep_colors) == 1
    for a, b in zip(sep_cols, sep_cols[1:]):
        assert b - a > 1
    assert len(sep_cols) == PANEL_COUNT - 1
    return sep_cols, next(iter(sep_colors))


def _panel_spans(sep_cols: list[int], width: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for col in sorted(sep_cols):
        assert start <= col - 1
        spans.append((start, col - 1))
        start = col + 1
    assert start <= width - 1
    spans.append((start, width - 1))

    panel_widths = [c1 - c0 + 1 for c0, c1 in spans]
    assert panel_widths
    assert len(set(panel_widths)) == 1
    assert panel_widths[0] >= 10
    return spans


def _panel_components(
    grid: np.ndarray,
    c0: int,
    c1: int,
    bg: int,
    sep_color: int,
) -> list[dict]:
    panel = grid[:, c0 : c1 + 1]
    h, w = panel.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(panel[r, c])
            if color in (bg, sep_color) or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, c0 + cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(panel[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            assert min(rows) > 0 and max(rows) < h - 1
            assert min(cols) > c0 and max(cols) < c1
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(cells),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _centered_offsets(coords: np.ndarray) -> list[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h % 2 == 1 and w % 2 == 1

    center = (r0 + h // 2, c0 + w // 2)
    coord_set = {tuple(map(int, rc)) for rc in coords}
    assert center in coord_set

    offsets = [(int(r) - center[0], int(c) - center[1]) for r, c in coords]
    offsets.sort()
    return offsets


def _touching_or_overlapping(
    template_cells: set[tuple[int, int]],
    stamp_cells: set[tuple[int, int]],
) -> bool:
    blocked = set(template_cells)
    for r, c in template_cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                blocked.add((r + dr, c + dc))
    return any(cell in blocked for cell in stamp_cells)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])
