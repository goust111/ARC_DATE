from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


BLACK = 0
BLUE = 1
GREEN = 3
GREY = 5
ORANGE = 7

_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
try:
    _ARC_UTILS = SourceFileLoader("_arc_utils_for_opus_11_047_3", str(_ARC_UTILS_PATH)).load_module()
except Exception:
    _ARC_UTILS = None


if _ARC_UTILS is not None:
    bbox_coords = _ARC_UTILS.bbox_coords
else:
    def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)


def _norm_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return {(r - r0, c - c0) for r, c in cells}


def _component_cells(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _render_panel(
    panel_shape: tuple[int, int],
    sprite_shape: set[tuple[int, int]],
    origin: tuple[int, int],
) -> np.ndarray:
    h, w = panel_shape
    r0, c0 = origin
    out = np.full((h, w), BLACK, dtype=int)
    for dr, dc in sprite_shape:
        rr = r0 + dr
        cc = c0 + dc
        assert 0 <= rr < h and 0 <= cc < w, "sprite must stay inside the panel"
        out[rr, cc] = GREEN
    return out


def _find_divider_col(grid: np.ndarray) -> int:
    h, w = grid.shape
    candidates = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    for c in candidates:
        if c <= 0 or c >= w - 3:
            continue
        left = grid[:, :c]
        sep_rows = [r for r in range(h) if np.all(left[r, :] == GREY)]
        if len(sep_rows) != 3:
            continue
        panel_h = sep_rows[0]
        expected = [panel_h, 2 * panel_h + 1, 3 * panel_h + 2]
        if panel_h > 0 and sep_rows == expected and h == 4 * panel_h + 3:
            return int(c)
    raise AssertionError("expected a unique full-height grey divider")


def _split_left_panels(grid: np.ndarray, divider_c: int) -> list[np.ndarray]:
    left = grid[:, :divider_c]
    panel_h = (grid.shape[0] - 3) // 4
    panels: list[np.ndarray] = []
    start = 0
    for i in range(4):
        panels.append(left[start : start + panel_h, :].copy())
        start += panel_h
        if i < 3:
            assert np.all(left[start, :] == GREY), "separator rows must be grey"
            start += 1
    assert len({panel.shape for panel in panels}) == 1
    return panels


def _panel_sprite_data(panel: np.ndarray) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    colors = set(map(int, np.unique(panel)))
    assert colors <= {BLACK, GREEN}, "panel should only contain black background and green sprite"
    coords = np.argwhere(panel == GREEN)
    assert 3 <= len(coords) <= 4, "sprite must contain 3-4 cells"
    comps = _component_cells(panel == GREEN)
    assert len(comps) == 1, "sprite must be 4-connected"
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shape = {(int(r - r0), int(c - c0)) for r, c in coords}
    return shape, (int(r0), int(c0))


def _infer_sequence(panels: list[np.ndarray]) -> tuple[tuple[int, int], set[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    shapes: list[set[tuple[int, int]]] = []
    origins: list[tuple[int, int]] = []
    for panel in panels:
        shape, origin = _panel_sprite_data(panel)
        shapes.append(shape)
        origins.append(origin)
    assert len({frozenset(shape) for shape in shapes}) == 1, "all panels must show the same sprite"
    dr = origins[1][0] - origins[0][0]
    dc = origins[1][1] - origins[0][1]
    assert (dr, dc) != (0, 0), "animation must move"
    for i in range(1, 3):
        assert origins[i + 1][0] - origins[i][0] == dr
        assert origins[i + 1][1] - origins[i][1] == dc
    return panels[0].shape, shapes[0], origins[-1], (int(dr), int(dc))


def _parse_frames(grid: np.ndarray, divider_c: int, panel_shape: tuple[int, int]) -> list[dict]:
    right = grid[:, divider_c + 1 :]
    frame_comps = _component_cells(right == BLUE)
    frames: list[dict] = []
    for coords in frame_comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(sub[border] == BLUE):
            continue
        if np.any(sub[~border] == BLUE):
            continue
        interior = sub[1:-1, 1:-1]
        if interior.shape != panel_shape:
            continue
        bar_rows = [r for r in range(interior.shape[0]) if np.all(interior[r, :] == ORANGE)]
        cleaned = interior.copy()
        if bar_rows:
            cleaned[np.array(bar_rows, dtype=int), :] = BLACK
        if np.any((cleaned != BLACK) & (cleaned != ORANGE)):
            continue
        if np.any(cleaned == ORANGE):
            continue
        frames.append(
            {
                "bbox": (int(r0), int(divider_c + 1 + c0), int(r1), int(divider_c + 1 + c1)),
                "n_bars": len(bar_rows),
            }
        )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    assert 2 <= len(frames) <= 3, "expected 2-3 blue frames"
    counts = [frame["n_bars"] for frame in frames]
    assert len(set(counts)) == len(counts), "frame bar counts must be distinct"
    return frames


def _future_panel(
    panel_shape: tuple[int, int],
    sprite_shape: set[tuple[int, int]],
    last_origin: tuple[int, int],
    delta: tuple[int, int],
    steps: int,
) -> np.ndarray:
    r0 = last_origin[0] + steps * delta[0]
    c0 = last_origin[1] + steps * delta[1]
    return _render_panel(panel_shape, sprite_shape, (int(r0), int(c0)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the bar count inside each frame to choose a future animation step, then place that future sprite back inside the same frame.

    Input structure:
    - A full-height grey divider splits the grid into a left animation strip and a right frame area.
    - The left side contains four equal green-sprite panels stacked top-to-bottom, separated by grey rows.
    - The right side contains 2-3 blue hollow frames; each frame interior is blank black except for 0-3 full-width orange horizontal bars.

    Rule:
    - Read the four left panels as consecutive animation frames and infer the unique constant sprite translation.
    - For each blue frame, count its orange interior bars to get N.
    - Extrapolate the left animation N more steps beyond the fourth panel and render that future panel inside the frame interior.
    - Erase the frame's orange bars while leaving the grey divider, the left strip, and the blue borders unchanged.

    Key insight:
    - The orange bar count is not decoration: it parameterizes how far to push the observed animation, and each frame receives a different future state of the same moving sprite.
    """
    g = np.asarray(grid, dtype=int)
    divider_c = _find_divider_col(g)
    panels = _split_left_panels(g, divider_c)
    panel_shape, sprite_shape, last_origin, delta = _infer_sequence(panels)
    frames = _parse_frames(g, divider_c, panel_shape)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = _future_panel(
            panel_shape,
            sprite_shape,
            last_origin,
            delta,
            frame["n_bars"],
        )
    return out


_BASE_SHAPES = [
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (0, 2), (0, 3)},
    {(0, 0), (1, 0), (2, 0), (3, 0)},
    {(0, 0), (1, 0), (2, 0), (2, 1)},
    {(0, 1), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
]


def _shape_dims(shape: set[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _choose_sprite_shape(inner_h: int, inner_w: int, max_count: int) -> set[tuple[int, int]] | None:
    max_shape_w = inner_w - (3 + max_count)
    candidates = []
    for shape in _BASE_SHAPES:
        shape = _norm_shape(shape)
        h, w = _shape_dims(shape)
        if h <= inner_h and w <= max_shape_w:
            candidates.append(shape)
    if not candidates:
        return None
    return random.choice(candidates)


def _allocate_right_layout(total_h: int, outer_h: int, n_frames: int) -> list[int] | None:
    min_needed = n_frames * outer_h + (n_frames - 1)
    if min_needed > total_h:
        return None
    extras = [0] * (n_frames + 1)
    for i in range(1, n_frames):
        extras[i] = 1
    slack = total_h - min_needed
    for _ in range(slack):
        extras[random.randrange(n_frames + 1)] += 1
    tops: list[int] = []
    cur = extras[0]
    for i in range(n_frames):
        tops.append(cur)
        cur += outer_h
        if i < n_frames - 1:
            cur += extras[i + 1]
    return tops


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.choice([2, 3])
    inner_h = random.randint(5, 6) if n_frames == 3 else random.randint(4, 6)
    inner_w = random.randint(8, 10)
    counts = random.sample([0, 1, 2, 3], k=n_frames)
    max_count = max(counts)
    sprite_shape = _choose_sprite_shape(inner_h, inner_w, max_count)
    if sprite_shape is None:
        return None

    shape_h, shape_w = _shape_dims(sprite_shape)
    direction = random.choice([-1, 1])
    row0 = random.randint(0, inner_h - shape_h)
    if direction == 1:
        col_min = 0
        col_max = inner_w - shape_w - (3 + max_count)
    else:
        col_min = 3 + max_count
        col_max = inner_w - shape_w
    if col_min > col_max:
        return None
    col0 = random.randint(col_min, col_max)

    panel_shape = (inner_h, inner_w)
    panels = [
        _render_panel(panel_shape, sprite_shape, (row0, col0 + direction * step))
        for step in range(4)
    ]
    future_by_count = {
        count: _future_panel(panel_shape, sprite_shape, (row0, col0 + 3 * direction), (0, direction), count)
        for count in counts
    }
    assert len({future.tobytes() for future in future_by_count.values()}) == len(counts)

    total_h = 4 * inner_h + 3
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    frame_tops = _allocate_right_layout(total_h, outer_h, n_frames)
    if frame_tops is None:
        return None

    total_w = inner_w + 1 + outer_w
    grid = np.full((total_h, total_w), BLACK, dtype=int)
    divider_c = inner_w
    grid[:, divider_c] = GREY

    row = 0
    for i, panel in enumerate(panels):
        grid[row : row + inner_h, :inner_w] = panel
        row += inner_h
        if i < 3:
            grid[row, :inner_w] = GREY
            row += 1

    frame_left = divider_c + 1
    for top, count in zip(frame_tops, counts):
        bottom = top + outer_h
        right = frame_left + outer_w
        grid[top:bottom, frame_left:right] = BLUE
        grid[top + 1 : bottom - 1, frame_left + 1 : right - 1] = BLACK
        if count:
            bar_rows = random.sample(range(inner_h), k=count)
            for br in bar_rows:
                grid[top + 1 + br, frame_left + 1 : right - 1] = ORANGE

    expected = grid.copy()
    for top, count in zip(frame_tops, counts):
        expected[top + 1 : top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w] = future_by_count[count]

    parsed_divider = _find_divider_col(grid)
    parsed_panels = _split_left_panels(grid, parsed_divider)
    parsed_shape, parsed_sprite, parsed_last, parsed_delta = _infer_sequence(parsed_panels)
    parsed_frames = _parse_frames(grid, parsed_divider, parsed_shape)
    assert parsed_delta == (0, direction)
    assert parsed_sprite == sprite_shape
    assert parsed_last == (row0, col0 + 3 * direction)
    assert sorted(frame["n_bars"] for frame in parsed_frames) == sorted(counts)
    assert any(frame["n_bars"] > 0 for frame in parsed_frames)

    naive = grid.copy()
    for frame in parsed_frames:
        r0, c0, r1, c1 = frame["bbox"]
        naive[r0 + 1 : r1, c0 + 1 : c1] = panels[-1]
    assert not np.array_equal(naive, expected), "bar count must matter"
    assert np.array_equal(main(grid), expected)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_case()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid ARC sample")
