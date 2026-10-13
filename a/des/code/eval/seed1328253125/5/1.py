from __future__ import annotations

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _gap_side(missing_r: int, missing_c: int, height: int, width: int) -> str | None:
    if missing_r == 0 and 0 < missing_c < width - 1:
        return "top"
    if missing_r == height - 1 and 0 < missing_c < width - 1:
        return "bottom"
    if missing_c == 0 and 0 < missing_r < height - 1:
        return "left"
    if missing_c == width - 1 and 0 < missing_r < height - 1:
        return "right"
    return None


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    candidates: list[dict[str, int | str]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 4 or width < 4:
                continue

            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            comp_mask = sub == color
            if np.any(comp_mask & ~border):
                continue
            missing = border & ~comp_mask
            if int(comp_mask.sum()) != int(border.sum()) - 1 or int(missing.sum()) != 1:
                continue

            missing_r, missing_c = map(int, np.argwhere(missing)[0])
            side = _gap_side(missing_r, missing_c, height, width)
            if side is None:
                continue

            interior = sub[1:-1, 1:-1]
            if not np.any(interior != bg):
                continue

            candidates.append(
                {
                    "color": color,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "height": height,
                    "width": width,
                    "side": side,
                }
            )

    assert candidates, "expected at least one frame candidate"

    groups: dict[tuple[int, int, int], list[dict[str, int | str]]] = {}
    for cand in candidates:
        key = (int(cand["color"]), int(cand["height"]), int(cand["width"]))
        groups.setdefault(key, []).append(cand)

    best = max(groups.values(), key=lambda group: (len(group), int(group[0]["height"]) * int(group[0]["width"])))
    return sorted(best, key=lambda frame: (int(frame["r0"]), int(frame["c0"])))


def _is_singleton_marker(grid: np.ndarray, row: int, col: int, color: int) -> bool:
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = row + dr
        cc = col + dc
        if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and int(grid[rr, cc]) == int(color):
            return False
    return True


def _find_legend(grid: np.ndarray, bg: int, frames: list[dict[str, int | str]]) -> dict[int, int]:
    temp = np.array(grid, copy=True)
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        temp[r0 : r1 + 1, c0 : c1 + 1] = bg

    groups: dict[int, dict[int, int]] = {}
    for color0 in np.unique(temp):
        bar_color = int(color0)
        if bar_color == bg:
            continue
        for coords in _components_of_color(temp, bar_color):
            r0, c0, r1, c1 = _bbox(coords)
            if r0 != r1:
                continue
            width = c1 - c0 + 1
            if width < 1 or width > 5 or int(len(coords)) != width:
                continue
            marker_c = c1 + 1
            if marker_c >= temp.shape[1]:
                continue
            marker_color = int(temp[r0, marker_c])
            if marker_color in (bg, bar_color):
                continue
            if not _is_singleton_marker(temp, r0, marker_c, marker_color):
                continue
            groups.setdefault(bar_color, {})
            groups[bar_color][width] = marker_color

    assert groups, "expected a legend"
    best = max(groups.values(), key=lambda mapping: (len(mapping), sum(mapping)))
    assert best, "legend mapping must be nonempty"
    return {int(length): int(color) for length, color in best.items()}


def _recolor_pattern(mask: np.ndarray, mode: str, legend: dict[int, int], bg: int) -> np.ndarray:
    out = np.full(mask.shape, bg, dtype=int)
    if mode == "rows":
        for r in range(mask.shape[0]):
            count = int(mask[r].sum())
            if count == 0:
                continue
            assert count in legend, "missing legend entry for row count"
            out[r, mask[r]] = int(legend[count])
    else:
        for c in range(mask.shape[1]):
            count = int(mask[:, c].sum())
            if count == 0:
                continue
            assert count in legend, "missing legend entry for column count"
            out[mask[:, c], c] = int(legend[count])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Determine the projection axis from each frame's gap, count pattern pixels along that axis,
    then recolor each populated row/column using the legend's count-to-color mapping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    legend = _find_legend(g, bg, frames)

    out = g.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        r1 = int(frame["r1"])
        c1 = int(frame["c1"])
        side = str(frame["side"])
        mode = "cols" if side in ("top", "bottom") else "rows"

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        pattern_mask = interior != bg
        colored = _recolor_pattern(pattern_mask, mode, legend, bg)

        target = out[r0 + 1 : r1, c0 + 1 : c1]
        target[pattern_mask] = colored[pattern_mask]

    return out


LEGEND_LENGTHS = (1, 2, 3, 4, 5)
GEN_ATTEMPTS = 400
GEN_GRID_H_RANGE = (24, 31)
GEN_GRID_W_RANGE = (24, 31)
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_FRAME_OUTER_H_RANGE = (7, 10)
GEN_FRAME_OUTER_W_RANGE = (7, 10)
GEN_FRAME_CLEARANCE = 1
GEN_FRAME_MARGIN = 1
GEN_LEGEND_MARGIN = 1
GEN_LEGEND_ROW_GAPS = (1, 2)
GEN_FRAME_TRIES = 200
GEN_PATTERN_TRIES = 200


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _legend_box(height: int, width: int, corner: str, row_gap: int) -> tuple[int, int, int, int]:
    legend_h = 1 + (len(LEGEND_LENGTHS) - 1) * (1 + row_gap)
    legend_w = max(LEGEND_LENGTHS) + 1
    if "top" in corner:
        r0 = GEN_LEGEND_MARGIN
    else:
        r0 = height - GEN_LEGEND_MARGIN - legend_h
    if "left" in corner:
        c0 = GEN_LEGEND_MARGIN
    else:
        c0 = width - GEN_LEGEND_MARGIN - legend_w
    return r0, c0, r0 + legend_h - 1, c0 + legend_w - 1


def _draw_legend(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    row_gap: int,
    bar_color: int,
    legend_map: dict[int, int],
    row_order: list[int],
) -> None:
    r0, c0, _r1, _c1 = box
    for idx, length in enumerate(row_order):
        rr = r0 + idx * (1 + row_gap)
        grid[rr, c0 : c0 + length] = int(bar_color)
        grid[rr, c0 + length] = int(legend_map[length])


def _make_frame_shell(height: int, width: int, frame_color: int, bg: int, side: str, gap_pos: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    if side == "top":
        patch[0, gap_pos] = bg
    elif side == "bottom":
        patch[-1, gap_pos] = bg
    elif side == "left":
        patch[gap_pos, 0] = bg
    else:
        patch[gap_pos, -1] = bg
    return patch


def _random_pattern_mask(
    mode: str,
    inner_h: int,
    inner_w: int,
    rng: np.random.Generator,
    preferred_counts: list[int],
) -> tuple[np.ndarray, set[int]] | None:
    primary = inner_h if mode == "rows" else inner_w
    secondary = inner_w if mode == "rows" else inner_h
    max_count = min(5, secondary)
    if primary < 3 or max_count < 2:
        return None

    demo_legend = {count: count for count in range(1, max(inner_h, inner_w) + 1)}
    min_active = min(primary, max(4, primary - 2))

    for _ in range(GEN_PATTERN_TRIES):
        active_count = int(rng.integers(min_active, primary + 1))
        active_idx = rng.choice(primary, size=active_count, replace=False)

        counts: list[int] = []
        for value in preferred_counts:
            if 1 <= value <= max_count and len(counts) < active_count:
                counts.append(int(value))
        while len(counts) < active_count:
            counts.append(int(rng.integers(1, max_count + 1)))
        rng.shuffle(counts)

        if len(set(counts)) < 2:
            continue

        mask = np.zeros((inner_h, inner_w), dtype=bool)
        for idx, count in zip(active_idx, counts):
            positions = rng.choice(secondary, size=int(count), replace=False)
            if mode == "rows":
                mask[int(idx), positions] = True
            else:
                mask[positions, int(idx)] = True

        by_rows = _recolor_pattern(mask, "rows", demo_legend, -1)
        by_cols = _recolor_pattern(mask, "cols", demo_legend, -1)
        if np.array_equal(by_rows, by_cols):
            continue

        return mask, set(int(x) for x in counts)

    return None


def _place_frames(
    height: int,
    width: int,
    frame_h: int,
    frame_w: int,
    count: int,
    legend_box: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(GEN_FRAME_TRIES):
            r0 = int(rng.integers(GEN_FRAME_MARGIN, height - frame_h - GEN_FRAME_MARGIN + 1))
            c0 = int(rng.integers(GEN_FRAME_MARGIN, width - frame_w - GEN_FRAME_MARGIN + 1))
            box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
            if _boxes_overlap(box, legend_box, pad=GEN_FRAME_CLEARANCE):
                continue
            if any(_boxes_overlap(box, other, pad=GEN_FRAME_CLEARANCE) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _make_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    grid_h = int(rng.integers(*GEN_GRID_H_RANGE))
    grid_w = int(rng.integers(*GEN_GRID_W_RANGE))
    frame_count = int(rng.integers(*GEN_FRAME_COUNT_RANGE))
    frame_h = int(rng.integers(*GEN_FRAME_OUTER_H_RANGE))
    frame_w = int(rng.integers(*GEN_FRAME_OUTER_W_RANGE))
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    if inner_h < 5 or inner_w < 5:
        return None

    palette = [int(x) for x in rng.permutation(10)]
    bg = palette[0]
    frame_color = palette[1]
    pattern_color = palette[2]
    bar_color = palette[3]
    legend_colors = palette[4:9]
    legend_map = {length: int(color) for length, color in zip(LEGEND_LENGTHS, legend_colors)}

    row_gap = int(rng.choice(GEN_LEGEND_ROW_GAPS))
    corner = str(rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"]))
    legend_box = _legend_box(grid_h, grid_w, corner, row_gap)
    if legend_box[0] < 0 or legend_box[1] < 0:
        return None

    frame_boxes = _place_frames(grid_h, grid_w, frame_h, frame_w, frame_count, legend_box, rng)
    if frame_boxes is None:
        return None

    modes = ["rows", "cols"]
    while len(modes) < frame_count:
        modes.append(str(rng.choice(["rows", "cols"])))
    rng.shuffle(modes)

    input_grid = np.full((grid_h, grid_w), bg, dtype=int)
    output_grid = np.full((grid_h, grid_w), bg, dtype=int)
    row_order = [int(x) for x in rng.permutation(LEGEND_LENGTHS)]
    _draw_legend(input_grid, legend_box, row_gap, bar_color, legend_map, row_order)
    _draw_legend(output_grid, legend_box, row_gap, bar_color, legend_map, row_order)

    used_counts: set[int] = set()
    for box, mode in zip(frame_boxes, modes):
        missing = [count for count in LEGEND_LENGTHS if count not in used_counts]
        preferred = [int(x) for x in rng.permutation(missing)] if missing else []
        mask_info = _random_pattern_mask(mode, inner_h, inner_w, rng, preferred)
        if mask_info is None:
            return None
        mask, counts = mask_info
        used_counts |= counts

        if mode == "rows":
            side = str(rng.choice(["left", "right"]))
            gap_pos = int(rng.integers(1, frame_h - 1))
        else:
            side = str(rng.choice(["top", "bottom"]))
            gap_pos = int(rng.integers(1, frame_w - 1))

        shell = _make_frame_shell(frame_h, frame_w, frame_color, bg, side, gap_pos)
        inp_patch = shell.copy()
        inp_patch[1:-1, 1:-1][mask] = pattern_color

        out_patch = shell.copy()
        out_patch[1:-1, 1:-1] = _recolor_pattern(mask, mode, legend_map, bg)

        r0, c0, r1, c1 = box
        input_grid[r0 : r1 + 1, c0 : c1 + 1] = inp_patch
        output_grid[r0 : r1 + 1, c0 : c1 + 1] = out_patch

    if used_counts != set(LEGEND_LENGTHS):
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
