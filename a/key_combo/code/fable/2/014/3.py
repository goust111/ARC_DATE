from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _ARC_UTILS_DIR.exists():
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import Color, get_bg  # type: ignore
except Exception:
    class Color:
        BLACK = 0
        GREY = 5

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


BG = int(Color.BLACK)
OBSTACLE = int(Color.GREY)
NON_OBSTACLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

PANEL_COUNT_RANGE = (3, 5)
PANEL_WIDTH_RANGE = (4, 6)
HEIGHT_RANGE = (6, 11)
MAX_GENERATION_ATTEMPTS = 400


def _full_height_non_bg_columns(grid: np.ndarray, bg: int) -> list[tuple[int, int]]:
    cols: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != int(bg):
            cols.append((int(c), int(vals[0])))
    return cols


def _parse_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = int(get_bg(g))

    full_cols = _full_height_non_bg_columns(g, bg)
    marker_candidates = [(c, color) for c, color in full_cols if c in (0, w - 1)]
    assert len(marker_candidates) == 1, "expected exactly one full-height edge marker line"
    marker_col, marker_color = marker_candidates[0]
    marker_side = "left" if marker_col == 0 else "right"

    divider_cols = sorted(c for c, _color in full_cols if c != marker_col)
    divider_colors = {color for c, color in full_cols if c != marker_col}
    assert len(divider_colors) == 1, "expected one divider color shared by all divider columns"
    divider_color = int(next(iter(divider_colors)))

    panel_count = len(divider_cols) + 1
    assert panel_count >= 2, "expected the divider columns to split the grid into multiple panels"

    data_start = 1 if marker_side == "left" else 0
    data_end = w if marker_side == "left" else w - 1
    starts = [data_start] + [c + 1 for c in divider_cols]
    ends = divider_cols + [data_end]
    widths = [int(end - start) for start, end in zip(starts, ends)]
    assert widths and len(set(widths)) == 1 and widths[0] > 0, "panel widths must be equal"
    panel_w = widths[0]

    panels_lr = [g[:, start:end].copy() for start, end in zip(starts, ends)]
    panels_from_marker = panels_lr if marker_side == "left" else list(reversed(panels_lr))

    return {
        "bg": bg,
        "h": h,
        "w": w,
        "marker_side": marker_side,
        "marker_color": int(marker_color),
        "marker_col": int(marker_col),
        "divider_color": divider_color,
        "divider_cols": divider_cols,
        "panel_w": int(panel_w),
        "panel_count": int(panel_count),
        "segments_lr": list(zip(starts, ends)),
        "panels_lr": panels_lr,
        "panels_from_marker": panels_from_marker,
    }


def _unroll_to_strip(panels_from_marker: list[np.ndarray]) -> np.ndarray:
    strip_panels = []
    for idx, panel in enumerate(panels_from_marker):
        strip_panels.append(panel if idx % 2 == 0 else panel[:, ::-1])
    return np.concatenate(strip_panels, axis=1)


def _fold_strip_to_grid(
    strip: np.ndarray,
    *,
    marker_side: str,
    marker_color: int,
    divider_color: int,
    panel_w: int,
    bg: int = BG,
) -> np.ndarray:
    h, strip_w = strip.shape
    assert strip_w % int(panel_w) == 0
    panel_count = strip_w // int(panel_w)

    panels_from_marker = []
    for idx in range(panel_count):
        panel = strip[:, idx * panel_w : (idx + 1) * panel_w].copy()
        panels_from_marker.append(panel if idx % 2 == 0 else panel[:, ::-1])

    panels_lr = panels_from_marker if marker_side == "left" else list(reversed(panels_from_marker))
    out_w = 1 + strip_w + (panel_count - 1)
    out = np.full((h, out_w), int(bg), dtype=int)

    c = 1 if marker_side == "left" else 0
    for idx, panel in enumerate(panels_lr):
        out[:, c : c + panel_w] = panel
        c += panel_w
        if idx != panel_count - 1:
            out[:, c] = int(divider_color)
            c += 1

    if marker_side == "left":
        out[:, 0] = int(marker_color)
    else:
        out[:, -1] = int(marker_color)
    return out


def _ray_events(strip: np.ndarray, panel_w: int, *, bg: int = BG, obstacle: int = OBSTACLE) -> list[dict]:
    g = np.asarray(strip, dtype=int)
    h, w = g.shape
    source_mask = (g != int(bg)) & (g != int(obstacle))
    blocker_mask = source_mask | (g == int(obstacle))

    events: list[dict] = []
    for r in range(h):
        blockers = np.flatnonzero(blocker_mask[r])
        if blockers.size == 0:
            continue
        for c in np.flatnonzero(source_mask[r]):
            idx = int(np.searchsorted(blockers, c, side="right"))
            if idx < blockers.size:
                stop = int(blockers[idx])
                stop_type = "obstacle" if int(g[r, stop]) == int(obstacle) else "source"
            else:
                stop = w
                stop_type = "end"
            fill_len = int(stop - c - 1)
            crosses_fold = fill_len > 0 and (stop - 1) // int(panel_w) > c // int(panel_w)
            events.append(
                {
                    "row": int(r),
                    "col": int(c),
                    "stop": int(stop),
                    "stop_type": stop_type,
                    "fill_len": fill_len,
                    "crosses_fold": bool(crosses_fold),
                    "color": int(g[r, c]),
                }
            )
    return events


def _extend_strip_rays(strip: np.ndarray, panel_w: int, *, bg: int = BG, obstacle: int = OBSTACLE) -> np.ndarray:
    out = np.asarray(strip, dtype=int).copy()
    for event in _ray_events(strip, panel_w, bg=bg, obstacle=obstacle):
        if event["fill_len"] > 0:
            r = int(event["row"])
            c = int(event["col"])
            stop = int(event["stop"])
            out[r, c + 1 : stop] = int(event["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Unroll equal-width folded panels into one strip, cast rays away from the marker edge, then fold the strip back.

    Input structure:
    - A full-height marker line occupies exactly the left or right edge, and full-height divider columns split the rest into equal-width panels.
    - Divider color appears only on divider columns; the panels contain background, colored source pixels, and gray obstacle pixels.
    - The panel sequence is ordered from the marker edge toward the opposite side.

    Rule:
    - Treat the panels as a folded strip: from the marker side, panels 2, 4, 6, ... must be mirrored horizontally when unrolled.
    - On the unrolled strip, every non-gray source pixel emits a horizontal ray away from the marker edge, filling background until the next obstacle, the next source pixel, or the strip end.
    - Fold the painted strip back with the same alternating mirror pattern and restore the original marker and divider columns unchanged.

    Key insight:
    - The divider columns are fold seams, not walls. A ray that leaves one panel continues inside the next panel from the same physical side, producing a snake-like path rather than a straight pass through divider columns.
    """
    g = np.asarray(grid, dtype=int)
    layout = _parse_layout(g)
    strip = _unroll_to_strip(layout["panels_from_marker"])
    painted = _extend_strip_rays(strip, layout["panel_w"], bg=layout["bg"], obstacle=OBSTACLE)

    refolded = []
    panel_w = int(layout["panel_w"])
    for idx in range(int(layout["panel_count"])):
        panel = painted[:, idx * panel_w : (idx + 1) * panel_w].copy()
        refolded.append(panel if idx % 2 == 0 else panel[:, ::-1])
    panels_lr = refolded if layout["marker_side"] == "left" else list(reversed(refolded))

    out = g.copy()
    for (start, end), panel in zip(layout["segments_lr"], panels_lr):
        out[:, start:end] = panel
    return out


def _random_palette() -> tuple[int, int, list[int]]:
    colors = NON_OBSTACLE_COLORS[:]
    random.shuffle(colors)
    marker = colors.pop()
    divider = colors.pop()
    source_colors = colors[:]
    random.shuffle(source_colors)
    return int(marker), int(divider), [int(c) for c in source_colors]


def _place_required_patterns(strip: np.ndarray, panel_w: int, source_colors: list[int]) -> bool:
    h, strip_w = strip.shape
    panel_count = strip_w // int(panel_w)
    if h < 3 or panel_count < 3:
        return False

    rows = random.sample(range(h), k=3)
    color_pool = source_colors[:]
    random.shuffle(color_pool)
    if len(color_pool) < 4:
        return False
    c1, c2, c3, c4 = color_pool[:4]

    r0 = rows[0]
    src_panel = random.randint(0, panel_count - 2)
    obs_panel = random.randint(src_panel + 1, panel_count - 1)
    x_src = src_panel * panel_w + random.randint(0, panel_w - 3)
    x_obs = obs_panel * panel_w + random.randint(2, panel_w - 1)
    strip[r0, x_src] = c1
    strip[r0, x_obs] = OBSTACLE

    r1 = rows[1]
    near_panel = random.randint(0, panel_count - 2)
    far_panel = random.randint(near_panel + 1, panel_count - 1)
    x_near = near_panel * panel_w + random.randint(0, panel_w - 3)
    x_far = far_panel * panel_w + random.randint(1, panel_w - 2)
    strip[r1, x_near] = c2
    strip[r1, x_far] = c3

    r2 = rows[2]
    tail_panel = random.randint(0, panel_count - 2)
    x_tail = tail_panel * panel_w + random.randint(0, panel_w - 2)
    strip[r2, x_tail] = c4
    return True


def _add_optional_items(strip: np.ndarray, source_colors: list[int]) -> None:
    h, strip_w = strip.shape
    free_rows = [r for r in range(h) if np.count_nonzero(strip[r] != BG) <= 1]
    random.shuffle(free_rows)

    for r in free_rows[: random.randint(0, min(2, len(free_rows)))]:
        src = random.randint(0, strip_w - 2)
        if int(strip[r, src]) != BG:
            continue
        color = random.choice(source_colors)
        strip[r, src] = int(color)
        if random.random() < 0.55:
            obs = random.randint(src + 1, strip_w - 1)
            if int(strip[r, obs]) == BG:
                strip[r, obs] = OBSTACLE

    free_cells = [(r, c) for r in range(h) for c in range(strip_w) if int(strip[r, c]) == BG]
    random.shuffle(free_cells)
    for r, c in free_cells[: random.randint(0, 2)]:
        strip[r, c] = OBSTACLE


def _good_events(strip: np.ndarray, panel_w: int) -> bool:
    events = _ray_events(strip, panel_w)
    if not events:
        return False
    meaningful = [event for event in events if event["fill_len"] >= 1]
    if len(meaningful) < 4:
        return False
    if sum(event["crosses_fold"] for event in meaningful) < 2:
        return False
    if not any(event["stop_type"] == "obstacle" and event["fill_len"] >= 2 for event in events):
        return False
    if not any(event["stop_type"] == "source" and event["fill_len"] >= 1 for event in events):
        return False
    if not any(event["stop_type"] == "end" and event["fill_len"] >= 1 for event in events):
        return False
    return sum(event["fill_len"] for event in events) >= 10


def _gen_attempt() -> np.ndarray | None:
    marker_color, divider_color, source_colors = _random_palette()
    panel_count = random.randint(*PANEL_COUNT_RANGE)
    panel_w_max = min(PANEL_WIDTH_RANGE[1], (30 - panel_count) // panel_count)
    panel_w_min = PANEL_WIDTH_RANGE[0]
    if panel_w_max < panel_w_min:
        return None
    panel_w = random.randint(panel_w_min, panel_w_max)

    h = random.randint(*HEIGHT_RANGE)
    strip = np.full((h, panel_count * panel_w), BG, dtype=int)
    if not _place_required_patterns(strip, panel_w, source_colors):
        return None
    _add_optional_items(strip, source_colors)

    if not _good_events(strip, panel_w):
        return None

    marker_side = random.choice(["left", "right"])
    grid = _fold_strip_to_grid(
        strip,
        marker_side=marker_side,
        marker_color=marker_color,
        divider_color=divider_color,
        panel_w=panel_w,
        bg=BG,
    )

    try:
        parsed = _parse_layout(grid)
    except AssertionError:
        return None

    if int(parsed["panel_count"]) != int(panel_count) or int(parsed["panel_w"]) != int(panel_w):
        return None

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.count_nonzero(out != grid) < max(8, panel_w + 2):
        return None
    if np.count_nonzero((grid != BG) & (grid != OBSTACLE)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid folded-panel snake-ray input")
