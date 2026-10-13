from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "points": pts, "bbox": _bbox(pts)})
    return comps


def _is_rect_frame(grid: np.ndarray, comp: dict[str, object]) -> bool:
    color = int(comp["color"])
    points = comp["points"]
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(points) != 2 * (h + w) - 4:
        return False

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(window[0, :] == color):
        return False
    if not np.all(window[-1, :] == color):
        return False
    if not np.all(window[:, 0] == color):
        return False
    if not np.all(window[:, -1] == color):
        return False
    if np.any(window[1:-1, 1:-1] == color):
        return False
    return True


def _frame_groups(grid: np.ndarray, bg: int) -> list[list[dict[str, object]]]:
    frames = [comp for comp in _color_components(grid, bg) if _is_rect_frame(grid, comp)]
    by_center: dict[tuple[int, int], list[dict[str, object]]] = {}
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        by_center.setdefault((r0 + r1, c0 + c1), []).append(frame)

    groups: list[list[dict[str, object]]] = []
    for group in by_center.values():
        if len(group) < 2:
            continue
        ordered = sorted(
            group,
            key=lambda frame: (-(frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1)),
        )
        ok = True
        for outer, inner in zip(ordered, ordered[1:], strict=False):
            or0, oc0, or1, oc1 = outer["bbox"]
            ir0, ic0, ir1, ic1 = inner["bbox"]
            if not (or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1):
                ok = False
                break
        if ok:
            groups.append(ordered)
    return groups


def _fill_interval(
    out: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    inner_bbox: tuple[int, int, int, int] | None,
    fill_color: int,
    bg: int,
    checker: bool,
) -> None:
    r0, c0, r1, c1 = outer_bbox
    anchor_r = r0 + 1
    anchor_c = c0 + 1
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if inner_bbox is not None:
                ir0, ic0, ir1, ic1 = inner_bbox
                if ir0 <= r <= ir1 and ic0 <= c <= ic1:
                    continue
            if checker:
                out[r, c] = fill_color if (r - anchor_r + c - anchor_c) % 2 == 0 else bg
            else:
                out[r, c] = fill_color


def _apply_group_fill(out: np.ndarray, frames_outer_to_inner: list[dict[str, object]], bg: int) -> None:
    frames_inner_to_outer = list(reversed(frames_outer_to_inner))
    fill_color = int(frames_inner_to_outer[0]["color"])
    previous_bbox: tuple[int, int, int, int] | None = None
    for interval_idx, frame in enumerate(frames_inner_to_outer, start=1):
        _fill_interval(
            out,
            frame["bbox"],
            previous_bbox,
            fill_color,
            bg,
            checker=interval_idx % 2 == 0,
        )
        previous_bbox = frame["bbox"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested rectangular frame groups from the inside outward with alternating solid and checker intervals.

    Input structure:
    - bg is the most common color.
    - The grid contains several groups of concentric 1-pixel-thick hollow rectangular frames, with 2 to 4 layers per group and background gaps between adjacent layers.
    - Each layer in a group has its own border color, and a small corner legend shows a solid 2x2 block beside a 2x2 checker block.

    Rule:
    - Detect each concentric frame group and order its frames from outermost to innermost.
    - Let F be the color of the innermost frame in that group.
    - Fill the innermost center region and each successive ring between frames from inside to outside: odd-numbered intervals are filled solid with F, even-numbered intervals are filled as a 1x1 checkerboard alternating F and bg.
    - Keep every frame border unchanged.

    Key insight:
    - The legend only indicates the two fill modes; the actual fill color is determined separately by the innermost frame of each nested group.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    out = g.copy()
    for group in _frame_groups(g, bg):
        _apply_group_fill(out, group, bg)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_group_spec(frame_palette: list[int], group_count: int) -> dict[str, object]:
    max_outer = {1: 18, 2: 14, 3: 12}[group_count]
    frame_count_options = [2, 3, 4] if group_count < 3 else [2, 3]
    for _ in range(200):
        frame_count = random.choice(frame_count_options)
        inner_h = random.randint(3, 6)
        inner_w = random.randint(3, 6)

        delta_rows: list[int] = []
        delta_cols: list[int] = []
        for _layer in range(frame_count - 1):
            if frame_count == 4:
                row_step = 2
                col_step = 2
            elif frame_count == 3:
                row_step = random.choice([2, 2, 3])
                col_step = random.choice([2, 2, 3])
            else:
                row_step = random.choice([2, 3, 4])
                col_step = random.choice([2, 3, 4])
            delta_rows.append(row_step)
            delta_cols.append(col_step)

        outer_h = inner_h + 2 * sum(delta_rows)
        outer_w = inner_w + 2 * sum(delta_cols)
        if outer_h > max_outer or outer_w > max_outer:
            continue

        colors = random.sample(frame_palette, frame_count)
        frames: list[dict[str, object]] = []
        r0 = 0
        c0 = 0
        r1 = outer_h - 1
        c1 = outer_w - 1
        frames.append({"bbox": (r0, c0, r1, c1), "color": colors[0]})
        for idx in range(frame_count - 1):
            r0 += delta_rows[idx]
            c0 += delta_cols[idx]
            r1 -= delta_rows[idx]
            c1 -= delta_cols[idx]
            frames.append({"bbox": (r0, c0, r1, c1), "color": colors[idx + 1]})

        return {"size": (outer_h, outer_w), "frames": frames}
    raise RuntimeError("failed to sample a valid frame group")


def _legend_patch(solid: int, checker_a: int, checker_b: int, phase: int) -> np.ndarray:
    patch = np.zeros((2, 4), dtype=int)
    patch[:, :2] = solid
    for r in range(2):
        for c in range(2):
            patch[r, c + 2] = checker_a if (r + c + phase) % 2 == 0 else checker_b
    return patch


def _legend_position(height: int, width: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, width - 4
    if corner == "bl":
        return height - 2, 0
    return height - 2, width - 4


def _box_free(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(used.shape[0], r0 + h + pad)
    cc1 = min(used.shape[1], c0 + w + pad)
    return not used[rr0:rr1, cc0:cc1].any()


def _mark_box(used: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(used.shape[0], r0 + h + pad)
    cc1 = min(used.shape[1], c0 + w + pad)
    used[rr0:rr1, cc0:cc1] = True


def _place_groups(
    height: int,
    width: int,
    groups: list[dict[str, object]],
    legend_corner: str,
) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    used = np.zeros((height, width), dtype=bool)
    legend_r, legend_c = _legend_position(height, width, legend_corner)
    _mark_box(used, legend_r, legend_c, 2, 4, pad=1)

    placements: list[tuple[int, int] | None] = [None] * len(groups)
    order = sorted(
        range(len(groups)),
        key=lambda idx: groups[idx]["size"][0] * groups[idx]["size"][1],
        reverse=True,
    )
    for idx in order:
        gh, gw = groups[idx]["size"]
        if height - gh - 2 < 1 or width - gw - 2 < 1:
            return None
        placed = False
        for _ in range(300):
            r0 = random.randint(1, height - gh - 1)
            c0 = random.randint(1, width - gw - 1)
            if _box_free(used, r0, c0, gh, gw, pad=1):
                placements[idx] = (r0, c0)
                _mark_box(used, r0, c0, gh, gw, pad=1)
                placed = True
                break
        if not placed:
            return None
    return [pos for pos in placements if pos is not None], (legend_r, legend_c)


def _draw_group(grid: np.ndarray, group: dict[str, object], top: int, left: int) -> list[dict[str, object]]:
    absolute_frames: list[dict[str, object]] = []
    for frame in group["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        abs_bbox = (top + r0, left + c0, top + r1, left + c1)
        _draw_frame(grid, abs_bbox, int(frame["color"]))
        absolute_frames.append({"bbox": abs_bbox, "color": int(frame["color"])})
    return absolute_frames


def _expected_output(inp: np.ndarray, placed_groups: list[list[dict[str, object]]]) -> np.ndarray:
    out = inp.copy()
    bg = _most_common_color(inp)
    for frames in placed_groups:
        _apply_group_fill(out, frames, bg)
    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)
    remaining = [color for color in range(10) if color != bg]
    legend_colors = random.sample(remaining, 3)
    frame_palette = [color for color in remaining if color not in legend_colors]

    group_count = random.randint(1, 3)
    groups = [_sample_group_spec(frame_palette, group_count) for _ in range(group_count)]

    min_height = max(18, max(group["size"][0] for group in groups) + 4)
    min_width = max(18, max(group["size"][1] for group in groups) + 4)
    if group_count == 3:
        min_height = max(min_height, 24)
        min_width = max(min_width, 24)
    elif group_count == 2:
        min_height = max(min_height, 22)
        min_width = max(min_width, 22)

    if min_height > 30 or min_width > 30:
        return None

    height = random.randint(min_height, 30)
    width = random.randint(min_width, 30)
    legend_corner = random.choice(["tl", "tr", "bl", "br"])
    placed = _place_groups(height, width, groups, legend_corner)
    if placed is None:
        return None
    placements, (legend_r, legend_c) = placed

    inp = np.full((height, width), bg, dtype=int)
    legend = _legend_patch(
        legend_colors[0],
        legend_colors[1],
        legend_colors[2],
        phase=random.randint(0, 1),
    )
    inp[legend_r : legend_r + 2, legend_c : legend_c + 4] = legend

    absolute_groups: list[list[dict[str, object]]] = []
    for group, (top, left) in zip(groups, placements, strict=False):
        absolute_groups.append(_draw_group(inp, group, top, left))

    out = _expected_output(inp, absolute_groups)
    if not np.array_equal(main(inp), out):
        return None

    values, counts = np.unique(inp, return_counts=True)
    if int(values[counts.argmax()]) != bg:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
