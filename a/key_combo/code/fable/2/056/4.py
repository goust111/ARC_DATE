from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


FOUR_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 500
PLACEMENT_RETRIES = 250


def _region_fill(region: np.ndarray) -> int:
    return int(Counter(map(int, region.ravel())).most_common(1)[0][0])


def _marker_color(windows: list[dict], bg: int, grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    candidates = [
        color
        for color, total in counts.items()
        if color != bg
        and total == len(windows)
        and all(np.count_nonzero(window["region"] == color) == 1 for window in windows)
    ]
    assert len(candidates) == 1, "expected exactly one shared marker color"
    return int(candidates[0])


def _window_records(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = int(get_bg(grid))
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(comps) == 2, "expected exactly two windows"

    windows = []
    for comp in comps:
        region = crop_bbox(grid, comp["bbox"])
        assert int(comp["area"]) == int(region.size), "window must fill its bbox"
        windows.append({"bbox": tuple(map(int, comp["bbox"])), "region": region})

    shapes = {tuple(window["region"].shape) for window in windows}
    assert len(shapes) == 1, "windows must be equal-sized"

    marker = _marker_color(windows, bg, grid)
    for window in windows:
        region = window["region"]
        fill = _region_fill(region)
        marker_coords = np.argwhere(region == marker)
        assert marker_coords.shape == (1, 2), "each window must contain one marker"
        window["fill"] = fill
        window["marker"] = (int(marker_coords[0, 0]), int(marker_coords[0, 1]))
        window["has_sprite"] = bool(np.any((region != fill) & (region != marker)))

    assert sum(window["has_sprite"] for window in windows) == 1, "expected one sprite window and one empty window"
    return bg, marker, windows


def _mirror_axis(src_bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]) -> str:
    dr = int(dst_bbox[0] - src_bbox[0])
    dc = int(dst_bbox[1] - src_bbox[1])
    assert abs(dr) != abs(dc), "window offset must have a unique dominant axis"
    return "horizontal" if abs(dc) > abs(dr) else "vertical"


def _transform_offsets(offset_to_color: dict[tuple[int, int], int], axis: str) -> dict[tuple[int, int], int]:
    transformed = {}
    for (dr, dc), color in offset_to_color.items():
        key = (dr, -dc) if axis == "horizontal" else (-dr, dc)
        transformed[key] = int(color)
    return transformed


def _render_window(
    shape: tuple[int, int],
    fill: int,
    marker_pos: tuple[int, int],
    offset_to_color: dict[tuple[int, int], int],
    *,
    clip: bool = False,
) -> np.ndarray:
    out = np.full(shape, int(fill), dtype=int)
    mr, mc = map(int, marker_pos)
    for (dr, dc), color in offset_to_color.items():
        rr = mr + int(dr)
        cc = mc + int(dc)
        if clip and not (0 <= rr < shape[0] and 0 <= cc < shape[1]):
            continue
        assert 0 <= rr < shape[0] and 0 <= cc < shape[1]
        out[rr, cc] = int(color)
    return out


def _paint_transformed(
    source_region: np.ndarray,
    src_fill: int,
    src_marker: tuple[int, int],
    dst_shape: tuple[int, int],
    dst_fill: int,
    dst_marker: tuple[int, int],
    axis: str,
) -> np.ndarray:
    out = np.full(dst_shape, int(dst_fill), dtype=int)
    sr, sc = map(int, src_marker)
    dr0, dc0 = map(int, dst_marker)
    for r, c in np.argwhere(source_region != src_fill):
        r = int(r)
        c = int(c)
        rel_r = r - sr
        rel_c = c - sc
        if axis == "horizontal":
            rr = dr0 + rel_r
            cc = dc0 - rel_c
        else:
            rr = dr0 - rel_r
            cc = dc0 + rel_c
        if 0 <= rr < dst_shape[0] and 0 <= cc < dst_shape[1]:
            out[rr, cc] = int(source_region[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the sprite window across its shared marker axis, then paste it into the empty window.

    Input structure:
    - The dominant outer background surrounds exactly two solid rectangular windows of equal size.
    - Each window contains exactly one pixel of the same marker color.
    - One window also contains a sparse sprite on top of its fill color, while the other window is empty except for the marker.

    Rule:
    - Crop the output to the empty window and fill it with that window's base color.
    - Compare the two window positions: if the empty window is offset more horizontally than vertically from the sprite window, mirror the sprite left-right across the marker column; otherwise mirror it up-down across the marker row.
    - Translate the mirrored non-fill pixels so the two markers coincide, and keep the marker color in the pasted result.

    Key insight:
    - The shared marker has two roles at once: it is both the alignment anchor and the mirror axis, while the dominant direction of the window-to-window displacement chooses which mirror to use.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, marker, windows = _window_records(g)
    source = next(window for window in windows if window["has_sprite"])
    destination = next(window for window in windows if not window["has_sprite"])
    axis = _mirror_axis(source["bbox"], destination["bbox"])

    out = _paint_transformed(
        source_region=source["region"],
        src_fill=int(source["fill"]),
        src_marker=source["marker"],
        dst_shape=destination["region"].shape,
        dst_fill=int(destination["fill"]),
        dst_marker=destination["marker"],
        axis=axis,
    )
    assert np.count_nonzero(out == marker) == 1
    return out


def _mirror_offset(offset: tuple[int, int], axis: str) -> tuple[int, int]:
    dr, dc = map(int, offset)
    return (dr, -dc) if axis == "horizontal" else (-dr, dc)


def _opposite_axis(axis: str) -> str:
    return "vertical" if axis == "horizontal" else "horizontal"


def _valid_offsets(
    shape: tuple[int, int],
    src_marker: tuple[int, int],
    dst_marker: tuple[int, int],
    axis: str,
) -> set[tuple[int, int]]:
    h, w = map(int, shape)
    src_r, src_c = map(int, src_marker)
    dst_r, dst_c = map(int, dst_marker)
    allowed: set[tuple[int, int]] = set()
    for dr in range(-src_r, h - src_r):
        for dc in range(-src_c, w - src_c):
            if axis == "horizontal":
                rr, cc = dst_r + dr, dst_c - dc
            else:
                rr, cc = dst_r - dr, dst_c + dc
            if 0 <= rr < h and 0 <= cc < w:
                allowed.add((int(dr), int(dc)))
    return allowed


def _reachable_offsets(allowed: set[tuple[int, int]]) -> set[tuple[int, int]]:
    if (0, 0) not in allowed:
        return set()
    stack = [(0, 0)]
    seen = {(0, 0)}
    while stack:
        dr, dc = stack.pop()
        for step_r, step_c in FOUR_NEIGHBORS:
            nxt = (dr + step_r, dc + step_c)
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _grow_sprite_offsets(component: set[tuple[int, int]], target_size: int) -> set[tuple[int, int]] | None:
    cells = {(0, 0)}
    while len(cells) < target_size:
        frontier = []
        for dr, dc in cells:
            for step_r, step_c in FOUR_NEIGHBORS:
                nxt = (dr + step_r, dc + step_c)
                if nxt in component and nxt not in cells:
                    frontier.append(nxt)
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _random_marker_coord(limit: int) -> int:
    if limit <= 2:
        return random.randrange(limit)
    if random.random() < 0.75:
        return random.randrange(1, limit - 1)
    return random.randrange(limit)


def _sample_sprite_offsets(
    shape: tuple[int, int],
    src_marker: tuple[int, int],
    dst_marker: tuple[int, int],
    axis: str,
) -> set[tuple[int, int]] | None:
    reachable = _reachable_offsets(_valid_offsets(shape, src_marker, dst_marker, axis))
    if len(reachable) < 5:
        return None

    max_size = min(len(reachable), max(5, (shape[0] * shape[1]) // 3))
    if max_size < 5:
        return None
    target_size = random.randint(5, max_size)
    return _grow_sprite_offsets(reachable, target_size)


def _assign_sprite_colors(
    offsets: set[tuple[int, int]],
    marker_color: int,
    sprite_palette: list[int],
) -> dict[tuple[int, int], int]:
    non_marker = [offset for offset in offsets if offset != (0, 0)]
    random.shuffle(non_marker)
    color_count = min(len(sprite_palette), max(1, min(3, len(non_marker))))
    active = sprite_palette[:color_count]

    assigned = {(0, 0): int(marker_color)}
    color_stream = active.copy()
    while len(color_stream) < len(non_marker):
        color_stream.append(random.choice(active))
    random.shuffle(color_stream)

    for offset, color in zip(non_marker, color_stream, strict=False):
        assigned[offset] = int(color)
    return assigned


def _pick_window_positions(
    grid_shape: tuple[int, int],
    window_shape: tuple[int, int],
    axis: str,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    gh, gw = map(int, grid_shape)
    wh, ww = map(int, window_shape)
    for _ in range(PLACEMENT_RETRIES):
        src_top = random.randint(1, gh - wh - 1)
        src_left = random.randint(1, gw - ww - 1)
        dst_top = random.randint(1, gh - wh - 1)
        dst_left = random.randint(1, gw - ww - 1)

        if not (src_top + wh < dst_top or dst_top + wh < src_top or src_left + ww < dst_left or dst_left + ww < src_left):
            continue

        dr = dst_top - src_top
        dc = dst_left - src_left
        if abs(dr) == abs(dc):
            continue
        if axis == "horizontal" and abs(dc) <= abs(dr):
            continue
        if axis == "vertical" and abs(dr) <= abs(dc):
            continue
        return (src_top, src_left), (dst_top, dst_left)
    return None


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.choice(("horizontal", "vertical"))
    window_shape = (random.randint(5, 8), random.randint(6, 9))
    wh, ww = window_shape

    bg, src_fill, dst_fill, marker_color, *sprite_colors = random.sample(ALL_COLORS, 7)
    src_marker = (_random_marker_coord(wh), _random_marker_coord(ww))
    dst_marker = (_random_marker_coord(wh), _random_marker_coord(ww))

    offsets = _sample_sprite_offsets(window_shape, src_marker, dst_marker, axis)
    if offsets is None:
        return None
    if axis == "horizontal" and all(dc == 0 for _, dc in offsets):
        return None
    if axis == "vertical" and all(dr == 0 for dr, _ in offsets):
        return None

    offset_to_color = _assign_sprite_colors(offsets, marker_color, sprite_colors)
    mirrored = _transform_offsets(offset_to_color, axis)
    plain = offset_to_color
    wrong = _transform_offsets(offset_to_color, _opposite_axis(axis))

    source_region = _render_window(window_shape, src_fill, src_marker, offset_to_color)
    dest_region = np.full(window_shape, dst_fill, dtype=int)
    dest_region[dst_marker] = marker_color

    expected = _render_window(window_shape, dst_fill, dst_marker, mirrored)
    translated = _render_window(window_shape, dst_fill, dst_marker, plain, clip=True)
    wrong_axis = _render_window(window_shape, dst_fill, dst_marker, wrong, clip=True)

    if np.array_equal(expected, translated):
        return None
    if np.array_equal(expected, wrong_axis):
        return None
    if np.array_equal(translated, wrong_axis):
        return None
    if _region_fill(source_region) != src_fill or _region_fill(dest_region) != dst_fill:
        return None

    grid_shape = (
        random.randint(max(wh + 4, 16), 30),
        random.randint(max(ww + 4, 16), 30),
    )
    positions = _pick_window_positions(grid_shape, window_shape, axis)
    if positions is None:
        return None
    (src_top, src_left), (dst_top, dst_left) = positions

    grid = np.full(grid_shape, bg, dtype=int)
    paste(grid, source_region, src_top, src_left)
    paste(grid, dest_region, dst_top, dst_left)

    if np.array_equal(main(grid), expected):
        return grid, expected
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _build_candidate()
        if candidate is None:
            continue
        grid, _expected = candidate
        return grid
    raise RuntimeError("failed to generate a valid mirror-transport input")
