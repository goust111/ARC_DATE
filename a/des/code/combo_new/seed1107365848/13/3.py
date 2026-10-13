import random
from collections import Counter, deque

import numpy as np


BG = 0
SEPARATOR_COLOR = 9
GEN_RETRIES = 300
PATTERN_LIBRARY = (
    np.array([[1]], dtype=int),
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(np.min(coords[:, 0])),
        int(np.min(coords[:, 1])),
        int(np.max(coords[:, 0])),
        int(np.max(coords[:, 1])),
    )


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(pts)),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            visited[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and bool(mask[nr, nc]):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_patch(grid: np.ndarray, coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _count_holes(panel: np.ndarray, bg: int) -> int:
    holes = 0
    for comp in _mask_components(panel == bg):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == panel.shape[0] - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == panel.shape[1] - 1):
            continue
        holes += 1
    return holes


def _shape_bbox(panel: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(panel != bg)
    assert len(coords) > 0
    return _bbox(coords)


def _stamp_pattern(panel: np.ndarray, pattern: np.ndarray, center_r: int, center_c: int, bg: int) -> None:
    ph, pw = pattern.shape
    top = center_r - ph // 2
    left = center_c - pw // 2
    for pr in range(ph):
        for pc in range(pw):
            if int(pattern[pr, pc]) == bg:
                continue
            rr = top + pr
            cc = left + pc
            assert 0 <= rr < panel.shape[0] and 0 <= cc < panel.shape[1]
            panel[rr, cc] = int(pattern[pr, pc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay each panel with the legend sprite indexed by the hole count of its enclosed shape.

    Input structure:
    - A dominant black background surrounds two equal-size panels separated by a single maroon line, either vertically or horizontally.
    - Each panel contains one monochrome closed shape; the black regions fully enclosed by that shape form 0, 1, or 2 holes.
    - A legend band outside the panels contains exactly three small colored patterns laid out from left to right.

    Rule:
    - Detect the separator to recover the two panel regions and remove the legend band from the output.
    - Count, for each panel, how many black connected components are enclosed by the panel's foreground shape.
    - Read the three legend patterns from left to right, choose pattern H for a panel with H holes, and stamp that pattern at the center of the shape's bounding box.

    Key insight:
    - The hole count is not painted directly; it must be inferred from enclosed background structure and then used as an index into the external left-to-right legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _monochrome_components(g, bg)
    assert len(comps) >= 6

    line_candidates = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if w == 1 and comp["area"] == h and h >= 5:
            line_candidates.append((comp["area"], "vertical", comp))
        if h == 1 and comp["area"] == w and w >= 5:
            line_candidates.append((comp["area"], "horizontal", comp))
    assert line_candidates
    _, orientation, separator = max(line_candidates, key=lambda item: item[0])

    separator_ids = {id(separator)}
    non_separator = [comp for comp in comps if id(comp) not in separator_ids]
    legend_comps = sorted(non_separator, key=lambda comp: (comp["area"], comp["bbox"][0], comp["bbox"][1]))[:3]
    assert len(legend_comps) == 3
    legend_top = min(comp["bbox"][0] for comp in legend_comps)
    main_region = g[:legend_top, :].copy()
    out = main_region.copy()

    legend_patterns = []
    for comp in sorted(legend_comps, key=lambda comp: comp["bbox"][1]):
        legend_patterns.append(_extract_patch(g, comp["coords"]))
    assert len(legend_patterns) == 3

    sr0, sc0, sr1, sc1 = separator["bbox"]
    if orientation == "vertical":
        assert sr0 == 0
        split = sc0
        panels = [
            out[:, :split],
            out[:, split + 1 :],
        ]
    else:
        assert sc0 == 0
        split = sr0
        panels = [
            out[:split, :],
            out[split + 1 :, :],
        ]

    for panel in panels:
        hole_count = _count_holes(panel, bg)
        assert 0 <= hole_count <= 2
        r0, c0, r1, c1 = _shape_bbox(panel, bg)
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        _stamp_pattern(panel, legend_patterns[hole_count], center_r, center_c, bg)

    return out


def _split_integer(total: int, parts: int, rng: random.Random) -> list[int]:
    cuts = sorted(rng.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _make_colored_pattern(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask > 0, color, BG).astype(int)


def _rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _build_panel(panel_h: int, panel_w: int, color: int, hole_count: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(120):
        height_choices = [v for v in range(5, panel_h - 1) if v % 2 == 1]
        width_choices = [v for v in range(5, panel_w - 1) if v % 2 == 1]
        if not height_choices or not width_choices:
            return None
        outer_h = rng.choice(height_choices)
        outer_w = rng.choice(width_choices)
        if hole_count == 2 and max(outer_h, outer_w) < 7:
            continue

        top_max = panel_h - outer_h - 1
        left_max = panel_w - outer_w - 1
        if top_max < 1 or left_max < 1:
            continue
        top = rng.randint(1, top_max)
        left = rng.randint(1, left_max)

        panel = np.zeros((panel_h, panel_w), dtype=int)
        panel[top : top + outer_h, left : left + outer_w] = color

        holes: list[tuple[int, int, int, int]] = []
        ok = True
        for _hole_idx in range(hole_count):
            placed = False
            for _ in range(80):
                hole_h = rng.randint(1, min(2, outer_h - 2))
                hole_w = rng.randint(1, min(2, outer_w - 2))
                hole_top = top + rng.randint(1, outer_h - hole_h - 1)
                hole_left = left + rng.randint(1, outer_w - hole_w - 1)
                rect = (hole_top, hole_left, hole_top + hole_h - 1, hole_left + hole_w - 1)
                if any(_rects_too_close(rect, other) for other in holes):
                    continue
                holes.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        for r0, c0, r1, c1 in holes:
            panel[r0 : r1 + 1, c0 : c1 + 1] = BG
        if _count_holes(panel, BG) != hole_count:
            continue
        return panel
    return None


def _layout_patterns(width: int, patterns: list[np.ndarray], rng: random.Random) -> list[int] | None:
    widths = [pattern.shape[1] for pattern in patterns]
    min_gap = 2
    side_margin = 1
    required = 2 * side_margin + sum(widths) + min_gap * (len(patterns) - 1)
    if required > width:
        return None
    extras = _split_integer(width - required, 4, rng)
    positions = []
    x = side_margin + extras[0]
    positions.append(x)
    x += widths[0] + min_gap + extras[1]
    positions.append(x)
    x += widths[1] + min_gap + extras[2]
    positions.append(x)
    return positions


def _compose_expected(main_region: np.ndarray, orientation: str, split: int, patterns: list[np.ndarray], bg: int) -> np.ndarray:
    out = main_region.copy()
    if orientation == "vertical":
        panels = [out[:, :split], out[:, split + 1 :]]
    else:
        panels = [out[:split, :], out[split + 1 :, :]]
    for panel in panels:
        hole_count = _count_holes(panel, bg)
        r0, c0, r1, c1 = _shape_bbox(panel, bg)
        _stamp_pattern(panel, patterns[hole_count], (r0 + r1) // 2, (c0 + c1) // 2, bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    orientation = rng.choice(["vertical", "horizontal"])
    legend_h = 3
    if orientation == "vertical":
        panel_h = rng.randint(9, 14)
        panel_w = rng.randint(8, 13)
        main_h = panel_h
        total_w = 2 * panel_w + 1
        total_h = main_h + legend_h
        split = panel_w
    else:
        panel_h = rng.randint(8, 11)
        panel_w = rng.randint(15, 24)
        main_h = 2 * panel_h + 1
        total_w = panel_w
        total_h = main_h + legend_h
        split = panel_h
    if total_h > 30 or total_w > 30:
        return None

    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8], 5)
    shape_colors = colors[:2]
    pattern_colors = colors[2:]
    pattern_masks = rng.sample(list(PATTERN_LIBRARY), 3)
    patterns = [_make_colored_pattern(mask, color) for mask, color in zip(pattern_masks, pattern_colors)]

    x_positions = _layout_patterns(total_w, patterns, rng)
    if x_positions is None:
        return None

    panels = []
    hole_counts = []
    for color in shape_colors:
        hole_count = rng.randint(0, 2)
        panel = _build_panel(panel_h, panel_w, color, hole_count, rng)
        if panel is None:
            return None
        panels.append(panel)
        hole_counts.append(hole_count)

    grid = np.zeros((total_h, total_w), dtype=int)
    main_region = grid[:main_h, :]
    if orientation == "vertical":
        main_region[:, split] = SEPARATOR_COLOR
        main_region[:, :split] = panels[0]
        main_region[:, split + 1 :] = panels[1]
    else:
        main_region[split, :] = SEPARATOR_COLOR
        main_region[:split, :] = panels[0]
        main_region[split + 1 :, :] = panels[1]

    legend = grid[main_h:, :]
    for pattern, x in zip(patterns, x_positions):
        ph, pw = pattern.shape
        legend[:ph, x : x + pw] = np.maximum(legend[:ph, x : x + pw], pattern)

    for pattern, x in zip(patterns, x_positions):
        ph, pw = pattern.shape
        block = legend[:ph, x : x + pw]
        if not np.array_equal(block[pattern != BG], pattern[pattern != BG]):
            return None

    for hole_count, panel in zip(hole_counts, panels):
        if _count_holes(panel, BG) != hole_count:
            return None

    expected = _compose_expected(grid[:main_h, :], orientation, split, patterns, BG)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-hole legend task")
