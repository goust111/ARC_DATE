from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
SEP = 5
PANEL_COUNT = 5
VISIBLE_COUNT = 4
MAX_RETRIES = 600

COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
PERIOD_SET_OPTIONS = (
    (1, 2, 3),
    (1, 3),
    (2, 3),
    (1, 2),
)
PERIOD_SET_WEIGHTS = (0.40, 0.25, 0.20, 0.15)


@dataclass(frozen=True)
class RectObject:
    color: int
    depth: int
    height: int
    width: int
    dr: int
    dc: int
    base_r: int
    base_c: int

    def top_left_at(self, panel_idx: int) -> tuple[int, int]:
        step_idx = panel_idx // self.depth
        return self.base_r + step_idx * self.dr, self.base_c + step_idx * self.dc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    sep_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == SEP)]
    assert len(sep_cols) == PANEL_COUNT - 1, "expected four separator columns"
    assert all(sep_cols[i] + 1 < sep_cols[i + 1] for i in range(len(sep_cols) - 1))

    starts = [0] + [c + 1 for c in sep_cols]
    ends = sep_cols + [g.shape[1]]
    widths = [end - start for start, end in zip(starts, ends)]
    assert len(set(widths)) == 1, "panels must have equal width"
    assert widths[0] >= 4

    return [g[:, start:end].copy() for start, end in zip(starts, ends)]


def _solid_rect_bbox(panel: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = panel == color
    box = _bbox(mask)
    assert box is not None, f"missing color {color}"
    r0, c0, r1, c1 = box
    assert np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == color), "color must form a solid rectangle"
    assert int(mask.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "color must be one rectangle"
    return box


def _analyze(grid: np.ndarray) -> tuple[list[np.ndarray], int, dict[int, dict[str, object]]]:
    panels = _split_panels(grid)
    template = panels[-1]
    assert np.all(template == template[0, 0]), "template panel must be blank"
    bg = int(template[0, 0])

    prior = panels[:-1]
    color_sets = [set(map(int, np.unique(panel))) - {bg} for panel in prior]
    assert all(color_sets), "every visible panel must contain rectangles"
    first_colors = color_sets[0]
    assert 2 <= len(first_colors) <= 3, "expected 2-3 rectangle colors"
    assert all(cs == first_colors for cs in color_sets[1:]), "all visible panels must show the same colors"

    info: dict[int, dict[str, object]] = {}
    depths = []
    for color in sorted(first_colors):
        boxes = [_solid_rect_bbox(panel, color) for panel in prior]
        shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in boxes}
        assert len(shapes) == 1, "rectangle size must stay constant"
        height, width = next(iter(shapes))
        depth = min(height, width)
        assert 1 <= depth <= 3, "depth must be inferable from visible panels"

        positions = [(r0, c0) for r0, c0, _r1, _c1 in boxes]
        step = (positions[depth][0] - positions[0][0], positions[depth][1] - positions[0][1])
        assert step != (0, 0), "each rectangle must move on active steps"

        for t in range(VISIBLE_COUNT):
            expected_idx = (t // depth) * depth
            assert positions[t] == positions[expected_idx], "inactive panels must keep the previous active position"

        info[color] = {
            "height": height,
            "width": width,
            "depth": depth,
            "positions": positions,
            "step": step,
        }
        depths.append(depth)

    assert len(set(depths)) == len(depths), "depths must be unique"
    return panels, bg, info


def _paint_clipped(dst: np.ndarray, bg: int, color: int, top: int, left: int, height: int, width: int) -> None:
    h, w = dst.shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + height)
    c1 = min(w, left + width)
    if r0 >= r1 or c0 >= c1:
        return
    region = dst[r0:r1, c0:c1]
    assert np.all(region == bg), "predicted rectangles must not overlap"
    region[:, :] = color


def _predict_from_analysis(panels: list[np.ndarray], bg: int, info: dict[int, dict[str, object]]) -> np.ndarray:
    out = np.full_like(panels[-1], bg)
    for color, spec in info.items():
        height = int(spec["height"])
        width = int(spec["width"])
        depth = int(spec["depth"])
        step_r, step_c = spec["step"]  # type: ignore[misc]
        last_r, last_c = spec["positions"][-1]  # type: ignore[index]
        if 4 % depth == 0:
            top, left = last_r + step_r, last_c + step_c
        else:
            top, left = last_r, last_c
        _paint_clipped(out, bg, int(color), int(top), int(left), height, width)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Predict the next panel by using each rectangle's shortest side as its motion period.

    Input structure:
    - The grid is a horizontal strip of five equal panels separated by gray columns.
    - Panels 0-3 each contain the same 2-3 colored solid rectangles on a blank background; panel 4 is an empty template.
    - Each color identifies one rectangle whose bounding-box size stays constant while its position may change across panels.

    Rule:
    - For each rectangle color, compute depth d = min(height, width) from its solid bounding box.
    - The rectangle only moves on panel indices that are multiples of d; between active indices it keeps the most recent active position.
    - Measure the constant translation vector between two adjacent visible active panels, then apply it once more only if panel 4 is also active for that depth.
    - Draw every predicted rectangle in the template panel, clipping any part that would fall outside the panel bounds.

    Key insight:
    - Rectangle geometry encodes time: the shortest side is not for recoloring, but for deciding how often that rectangle advances in the panel sequence.
    """
    panels, bg, info = _analyze(np.asarray(grid, dtype=int))
    return _predict_from_analysis(panels, bg, info)


def _period_dims(depth: int, panel_h: int, panel_w: int) -> list[tuple[int, int]]:
    out = []
    if depth == 1:
        for width in range(2, panel_w + 1):
            out.append((1, width))
        for height in range(2, panel_h + 1):
            out.append((height, 1))
    else:
        max_h = min(panel_h, depth + 2)
        max_w = min(panel_w, depth + 2)
        for height in range(depth, max_h + 1):
            for width in range(depth, max_w + 1):
                if min(height, width) == depth:
                    out.append((height, width))
    random.shuffle(out)
    return out


def _step_options(depth: int) -> list[tuple[int, int]]:
    limit = 1 if depth == 1 else 2
    options = [
        (dr, dc)
        for dr in range(-limit, limit + 1)
        for dc in range(-limit, limit + 1)
        if (dr, dc) != (0, 0)
    ]
    random.shuffle(options)
    return options


def _visible_full(obj: RectObject, panel_h: int, panel_w: int) -> bool:
    for t in range(VISIBLE_COUNT):
        r, c = obj.top_left_at(t)
        if not (0 <= r <= panel_h - obj.height and 0 <= c <= panel_w - obj.width):
            return False
    return True


def _output_visible(obj: RectObject, panel_h: int, panel_w: int) -> bool:
    r, c = obj.top_left_at(4)
    return r < panel_h and c < panel_w and r + obj.height > 0 and c + obj.width > 0


def _output_clipped(obj: RectObject, panel_h: int, panel_w: int) -> bool:
    r, c = obj.top_left_at(4)
    return not (0 <= r <= panel_h - obj.height and 0 <= c <= panel_w - obj.width)


def _mask_for(obj: RectObject, panel_h: int, panel_w: int, panel_idx: int) -> np.ndarray:
    mask = np.zeros((panel_h, panel_w), dtype=bool)
    top, left = obj.top_left_at(panel_idx)
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(panel_h, top + obj.height)
    c1 = min(panel_w, left + obj.width)
    if r0 < r1 and c0 < c1:
        mask[r0:r1, c0:c1] = True
    return mask


def _candidate_objects(
    color: int,
    depth: int,
    panel_h: int,
    panel_w: int,
    require_clip: bool | None,
) -> list[RectObject]:
    candidates: list[RectObject] = []
    for height, width in _period_dims(depth, panel_h, panel_w):
        for dr, dc in _step_options(depth):
            for base_r in range(-height + 1, panel_h):
                for base_c in range(-width + 1, panel_w):
                    obj = RectObject(color, depth, height, width, dr, dc, base_r, base_c)
                    if not _visible_full(obj, panel_h, panel_w):
                        continue
                    if not _output_visible(obj, panel_h, panel_w):
                        continue
                    clip = _output_clipped(obj, panel_h, panel_w)
                    if require_clip is not None and clip != require_clip:
                        continue
                    if obj.top_left_at(depth) == obj.top_left_at(0):
                        continue
                    candidates.append(obj)
    random.shuffle(candidates)
    return candidates


def _choose_periods() -> tuple[int, ...]:
    return tuple(random.choices(PERIOD_SET_OPTIONS, weights=PERIOD_SET_WEIGHTS, k=1)[0])


def _pick_panel_shape(periods: tuple[int, ...]) -> tuple[int, int]:
    if 3 in periods:
        return random.randint(6, 8), 5
    return random.randint(6, 8), random.choice((4, 5))


def _search_layout(periods: tuple[int, ...], panel_h: int, panel_w: int) -> list[RectObject] | None:
    colors = random.sample(COLOR_POOL, len(periods))
    active_on_4 = [depth for depth in periods if 4 % depth == 0]
    force_clip = random.random() < 0.45 and bool(active_on_4)
    clip_depth = random.choice(active_on_4) if force_clip else None

    candidate_lists: list[list[RectObject]] = []
    for color, depth in zip(colors, periods):
        require_clip = True if depth == clip_depth else None
        candidates = _candidate_objects(color, depth, panel_h, panel_w, require_clip=require_clip)
        if not candidates:
            return None
        candidate_lists.append(candidates)

    order = sorted(range(len(periods)), key=lambda i: len(candidate_lists[i]))
    occupancy = [np.zeros((panel_h, panel_w), dtype=bool) for _ in range(PANEL_COUNT)]
    chosen: list[RectObject | None] = [None] * len(periods)

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        obj_idx = order[idx]
        for obj in candidate_lists[obj_idx]:
            masks = [_mask_for(obj, panel_h, panel_w, t) for t in range(PANEL_COUNT)]
            if any(np.any(occupancy[t] & masks[t]) for t in range(PANEL_COUNT)):
                continue
            for t in range(PANEL_COUNT):
                occupancy[t] |= masks[t]
            chosen[obj_idx] = obj
            if backtrack(idx + 1):
                return True
            chosen[obj_idx] = None
            for t in range(PANEL_COUNT):
                occupancy[t] ^= masks[t]
        return False

    if not backtrack(0):
        return None
    result = [obj for obj in chosen if obj is not None]
    assert len(result) == len(periods)
    return result


def _render_input(objects: list[RectObject], panel_h: int, panel_w: int) -> np.ndarray:
    total_w = PANEL_COUNT * panel_w + (PANEL_COUNT - 1)
    grid = np.full((panel_h, total_w), BG, dtype=int)
    for k in range(PANEL_COUNT - 1):
        grid[:, (k + 1) * panel_w + k] = SEP

    for panel_idx in range(VISIBLE_COUNT):
        col0 = panel_idx * (panel_w + 1)
        panel = grid[:, col0 : col0 + panel_w]
        for obj in objects:
            top, left = obj.top_left_at(panel_idx)
            panel[top : top + obj.height, left : left + obj.width] = obj.color
    return grid


def _render_expected_output(objects: list[RectObject], panel_h: int, panel_w: int) -> np.ndarray:
    out = np.full((panel_h, panel_w), BG, dtype=int)
    for obj in objects:
        top, left = obj.top_left_at(4)
        _paint_clipped(out, BG, obj.color, top, left, obj.height, obj.width)
    return out


def _panel_slice(grid: np.ndarray, panel_idx: int, panel_w: int) -> np.ndarray:
    col0 = panel_idx * (panel_w + 1)
    return grid[:, col0 : col0 + panel_w]


def _validate_generated(grid: np.ndarray, objects: list[RectObject], expected: np.ndarray) -> bool:
    try:
        panels, bg, info = _analyze(grid)
        solved = main(grid)
    except AssertionError:
        return False

    if bg != BG:
        return False
    if not np.array_equal(solved, expected):
        return False
    if not np.array_equal(_predict_from_analysis(panels, bg, info), expected):
        return False

    panel_w = panels[0].shape[1]
    for obj in objects:
        spec = info.get(obj.color)
        if spec is None:
            return False
        positions = spec["positions"]  # type: ignore[index]
        if spec["height"] != obj.height or spec["width"] != obj.width or spec["depth"] != obj.depth:
            return False
        if tuple(spec["step"]) != (obj.dr, obj.dc):
            return False
        if positions != [obj.top_left_at(t) for t in range(VISIBLE_COUNT)]:
            return False

    if not any(4 % obj.depth == 0 for obj in objects):
        return False
    if not any(_mask_for(obj, panels[0].shape[0], panel_w, 4).sum() != _mask_for(obj, panels[0].shape[0], panel_w, 3).sum() or obj.top_left_at(4) != obj.top_left_at(3) for obj in objects if 4 % obj.depth == 0):
        return False

    visible_panels = [_panel_slice(grid, i, panel_w) for i in range(VISIBLE_COUNT)]
    if any(np.array_equal(expected, panel) for panel in visible_panels):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    periods = _choose_periods()
    panel_h, panel_w = _pick_panel_shape(periods)
    objects = _search_layout(periods, panel_h, panel_w)
    if objects is None:
        return None

    grid = _render_input(objects, panel_h, panel_w)
    expected = _render_expected_output(objects, panel_h, panel_w)

    if not _validate_generated(grid, objects, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
