from __future__ import annotations

import random
from typing import Any

import numpy as np


BG_COLOR = 0
DIRECTIONS = ("up", "down", "left", "right")
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _color_components(grid: np.ndarray, bg: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for coords in _components(grid == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _is_symmetric(mask: np.ndarray) -> bool:
    variants = [np.flipud(mask), np.fliplr(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        variants.extend([np.rot90(mask, 1), np.rot90(mask, 3)])
    return any(mask.shape == other.shape and np.array_equal(mask, other) for other in variants)


def _extract_frame(grid: np.ndarray, bg: int, comp: dict[str, Any]) -> dict[str, Any] | None:
    frame_color = int(comp["color"])
    coords = comp["coords"]
    r0, c0, r1, c1 = map(int, comp["bbox"])
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 4 or width < 4:
        return None

    border = np.zeros((height, width), dtype=bool)
    border[coords[:, 0] - r0, coords[:, 1] - c0] = True
    for rr, cc in coords:
        if int(rr) not in (r0, r1) and int(cc) not in (c0, c1):
            return None

    if not (border[0, 0] and border[0, -1] and border[-1, 0] and border[-1, -1]):
        return None

    sides = {
        "up": border[0, :],
        "down": border[-1, :],
        "left": border[:, 0],
        "right": border[:, -1],
    }
    missing = [name for name, arr in sides.items() if not bool(arr.all())]
    if len(missing) != 1:
        return None

    side = missing[0]
    gap_arr = sides[side]
    gap_idx = np.flatnonzero(~gap_arr)
    if gap_idx.size == 0:
        return None
    if int(gap_idx[0]) == 0 or int(gap_idx[-1]) == int(len(gap_arr) - 1):
        return None
    if int(gap_idx[-1] - gap_idx[0] + 1) != int(gap_idx.size):
        return None

    for other_side, arr in sides.items():
        if other_side != side and not bool(arr.all()):
            return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0 or np.any(inner == frame_color):
        return None
    inner_colors = set(int(v) for v in np.unique(inner) if int(v) not in (bg, frame_color))
    if len(inner_colors) != 1:
        return None

    pattern_color = int(next(iter(inner_colors)))
    pattern_mask = inner == pattern_color
    if np.any((inner != bg) & (inner != pattern_color)):
        return None

    pattern_parts = _components(pattern_mask)
    if len(pattern_parts) != 1:
        return None
    rel_coords = pattern_parts[0]
    pr0_rel, pc0_rel, pr1_rel, pc1_rel = _bbox(rel_coords)

    abs_coords = rel_coords + np.array([r0 + 1, c0 + 1], dtype=int)
    pr0, pc0, pr1, pc1 = _bbox(abs_coords)

    if side in ("up", "down"):
        if (c0 + int(gap_idx[0]), c0 + int(gap_idx[-1])) != (pc0, pc1):
            return None
    else:
        if (r0 + int(gap_idx[0]), r0 + int(gap_idx[-1])) != (pr0, pr1):
            return None

    return {
        "bbox": (r0, c0, r1, c1),
        "frame_color": frame_color,
        "pattern_color": pattern_color,
        "pattern_coords": [(int(r), int(c)) for r, c in abs_coords],
        "pattern_bbox": (pr0, pc0, pr1, pc1),
        "side": side,
    }


def _shift_for_frame(info: dict[str, Any]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, info["bbox"])
    pr0, pc0, pr1, pc1 = map(int, info["pattern_bbox"])
    side = info["side"]
    if side == "up":
        return int(r0 - 1 - pr1), 0
    if side == "down":
        return int(r1 + 1 - pr0), 0
    if side == "left":
        return 0, int(c0 - 1 - pc1)
    return 0, int(c1 + 1 - pc0)


def _copy_bbox(info: dict[str, Any]) -> tuple[int, int, int, int]:
    pr0, pc0, pr1, pc1 = map(int, info["pattern_bbox"])
    dr, dc = _shift_for_frame(info)
    return int(pr0 + dr), int(pc0 + dc), int(pr1 + dr), int(pc1 + dc)


def _ray_marker_color(
    grid: np.ndarray,
    bg: int,
    singleton_cells: set[tuple[int, int]],
    side: str,
    copy_bbox: tuple[int, int, int, int],
    line: int,
) -> int | None:
    r0, c0, r1, c1 = map(int, copy_bbox)
    if side == "up":
        for rr in range(r0 - 1, -1, -1):
            if int(grid[rr, line]) != bg:
                return int(grid[rr, line]) if (rr, line) in singleton_cells else None
        return None
    if side == "down":
        for rr in range(r1 + 1, grid.shape[0]):
            if int(grid[rr, line]) != bg:
                return int(grid[rr, line]) if (rr, line) in singleton_cells else None
        return None
    if side == "left":
        for cc in range(c0 - 1, -1, -1):
            if int(grid[line, cc]) != bg:
                return int(grid[line, cc]) if (line, cc) in singleton_cells else None
        return None
    for cc in range(c1 + 1, grid.shape[1]):
        if int(grid[line, cc]) != bg:
            return int(grid[line, cc]) if (line, cc) in singleton_cells else None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each inner pattern through its frame gap and recolor the projected copy from outward marker rays.

    Input structure:
    - The background is black.
    - Several separate monochrome hollow rectangular frames appear; each frame has exactly one contiguous gap on one side.
    - Each frame encloses one connected monochrome pattern of a different color.
    - Outside each gap direction, single-pixel marker dots may appear along the outward rows or columns.

    Rule:
    - Detect every rectangular frame and determine which side contains the gap.
    - Translate the enclosed pattern straight outward in that direction until the copied pattern sits immediately outside the frame.
    - Recolor each copied pixel by scanning farther outward on its row or column: if the first non-background cell on that ray is a singleton marker, use that marker color; otherwise use the frame color.
    - Keep the original frames, inner patterns, and marker dots unchanged.

    Key insight:
    - The missing border segment encodes a projection direction, and the exterior singleton dots act as a row-wise or column-wise palette for the translated copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_most_common_color(g))
    components = _color_components(g, bg)
    singleton_cells = {
        (int(comp["coords"][0, 0]), int(comp["coords"][0, 1]))
        for comp in components
        if int(comp["area"]) == 1
    }

    frames: list[dict[str, Any]] = []
    for comp in components:
        info = _extract_frame(g, bg, comp)
        if info is not None:
            frames.append(info)
    assert frames, "expected at least one valid frame"

    out = g.copy()
    for info in frames:
        side = info["side"]
        copy_box = _copy_bbox(info)
        dr, dc = _shift_for_frame(info)
        line_colors: dict[int, int] = {}

        if side in ("up", "down"):
            for col in range(int(copy_box[1]), int(copy_box[3]) + 1):
                marker_color = _ray_marker_color(g, bg, singleton_cells, side, copy_box, int(col))
                if marker_color is not None:
                    line_colors[int(col)] = int(marker_color)
        else:
            for row in range(int(copy_box[0]), int(copy_box[2]) + 1):
                marker_color = _ray_marker_color(g, bg, singleton_cells, side, copy_box, int(row))
                if marker_color is not None:
                    line_colors[int(row)] = int(marker_color)

        for rr, cc in info["pattern_coords"]:
            nr = int(rr + dr)
            nc = int(cc + dc)
            assert 0 <= nr < out.shape[0] and 0 <= nc < out.shape[1], "projected copy must fit in grid"
            if side in ("up", "down"):
                out[nr, nc] = int(line_colors.get(nc, int(info["frame_color"])))
            else:
                out[nr, nc] = int(line_colors.get(nr, int(info["frame_color"])))

    return out


def _random_pattern(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        height = rng.randint(2, 4)
        width = rng.randint(2, 4)
        target_area = rng.randint(max(3, height + width - 2), height * width - 1)
        start = (rng.randrange(height), rng.randrange(width))
        cells = {start}
        frontier = set()
        while len(cells) < target_area:
            frontier.clear()
            for rr, cc in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(tuple(frontier)))
        if len(cells) != target_area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True

        if mask.all():
            continue
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if _is_symmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to build a valid asymmetric pattern")


def _rects_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _make_object_spec(rng: random.Random) -> dict[str, Any]:
    pattern = _random_pattern(rng)
    ph, pw = map(int, pattern.shape)
    direction = rng.choice(DIRECTIONS)

    top_pad = rng.randint(0, 2)
    bottom_pad = rng.randint(0, 2)
    left_pad = rng.randint(0, 2)
    right_pad = rng.randint(0, 2)
    frame_h = int(ph + top_pad + bottom_pad + 2)
    frame_w = int(pw + left_pad + right_pad + 2)
    marker_strip = rng.randint(1, 3)

    if direction == "up":
        env_h, env_w = int(marker_strip + ph + frame_h), int(frame_w)
        frame_offset = (int(marker_strip + ph), 0)
        span = int(pw)
    elif direction == "down":
        env_h, env_w = int(frame_h + ph + marker_strip), int(frame_w)
        frame_offset = (0, 0)
        span = int(pw)
    elif direction == "left":
        env_h, env_w = int(frame_h), int(marker_strip + pw + frame_w)
        frame_offset = (0, int(marker_strip + pw))
        span = int(ph)
    else:
        env_h, env_w = int(frame_h), int(frame_w + pw + marker_strip)
        frame_offset = (0, 0)
        span = int(ph)

    if span > 1 and rng.random() < 0.8:
        marker_count = rng.randint(1, span - 1)
    else:
        marker_count = span
    line_offsets = sorted(rng.sample(range(span), marker_count))
    colors = rng.sample(COLOR_POOL, 2 + marker_count)

    return {
        "direction": direction,
        "pattern": pattern,
        "pattern_h": ph,
        "pattern_w": pw,
        "top_pad": top_pad,
        "bottom_pad": bottom_pad,
        "left_pad": left_pad,
        "right_pad": right_pad,
        "frame_h": frame_h,
        "frame_w": frame_w,
        "marker_strip": marker_strip,
        "env_h": env_h,
        "env_w": env_w,
        "frame_offset": frame_offset,
        "marker_offsets": line_offsets,
        "frame_color": int(colors[0]),
        "pattern_color": int(colors[1]),
        "marker_colors": [int(v) for v in colors[2:]],
    }


def _finalize_object(rng: random.Random, spec: dict[str, Any], env_origin: tuple[int, int]) -> None:
    env_r, env_c = map(int, env_origin)
    off_r, off_c = map(int, spec["frame_offset"])
    frame_h = int(spec["frame_h"])
    frame_w = int(spec["frame_w"])
    ph = int(spec["pattern_h"])
    pw = int(spec["pattern_w"])

    r0 = int(env_r + off_r)
    c0 = int(env_c + off_c)
    r1 = int(r0 + frame_h - 1)
    c1 = int(c0 + frame_w - 1)
    pr0 = int(r0 + 1 + int(spec["top_pad"]))
    pc0 = int(c0 + 1 + int(spec["left_pad"]))
    pr1 = int(pr0 + ph - 1)
    pc1 = int(pc0 + pw - 1)

    direction = spec["direction"]
    if direction == "up":
        copy_bbox = (int(r0 - ph), pc0, int(r0 - 1), pc1)
        marker_band = list(range(int(copy_bbox[0] - int(spec["marker_strip"])), int(copy_bbox[0])))
    elif direction == "down":
        copy_bbox = (int(r1 + 1), pc0, int(r1 + ph), pc1)
        marker_band = list(range(int(copy_bbox[2] + 1), int(copy_bbox[2] + int(spec["marker_strip"]) + 1)))
    elif direction == "left":
        copy_bbox = (pr0, int(c0 - pw), pr1, int(c0 - 1))
        marker_band = list(range(int(copy_bbox[1] - int(spec["marker_strip"])), int(copy_bbox[1])))
    else:
        copy_bbox = (pr0, int(c1 + 1), pr1, int(c1 + pw))
        marker_band = list(range(int(copy_bbox[3] + 1), int(copy_bbox[3] + int(spec["marker_strip"]) + 1)))

    marker_lines: dict[int, int] = {}
    marker_cells: list[tuple[int, int, int]] = []
    for offset, color in zip(spec["marker_offsets"], spec["marker_colors"]):
        if direction in ("up", "down"):
            col = int(pc0 + int(offset))
            row = int(rng.choice(marker_band))
            marker_lines[col] = int(color)
            marker_cells.append((row, col, int(color)))
        else:
            row = int(pr0 + int(offset))
            col = int(rng.choice(marker_band))
            marker_lines[row] = int(color)
            marker_cells.append((row, col, int(color)))

    spec["frame_bbox"] = (r0, c0, r1, c1)
    spec["pattern_origin"] = (pr0, pc0)
    spec["pattern_bbox_abs"] = (pr0, pc0, pr1, pc1)
    spec["copy_bbox"] = copy_bbox
    spec["marker_lines"] = marker_lines
    spec["marker_cells"] = marker_cells
    spec["envelope"] = (env_r, env_c, env_r + int(spec["env_h"]) - 1, env_c + int(spec["env_w"]) - 1)


def _render_input(shape: tuple[int, int], specs: list[dict[str, Any]]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = map(int, spec["frame_bbox"])
        pr0, pc0 = map(int, spec["pattern_origin"])
        ph = int(spec["pattern_h"])
        pw = int(spec["pattern_w"])
        frame_color = int(spec["frame_color"])
        pattern_color = int(spec["pattern_color"])

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        if spec["direction"] == "up":
            grid[r0, pc0 : pc0 + pw] = BG_COLOR
        elif spec["direction"] == "down":
            grid[r1, pc0 : pc0 + pw] = BG_COLOR
        elif spec["direction"] == "left":
            grid[pr0 : pr0 + ph, c0] = BG_COLOR
        else:
            grid[pr0 : pr0 + ph, c1] = BG_COLOR

        pattern = spec["pattern"]
        for rr, cc in np.argwhere(pattern):
            grid[int(pr0 + rr), int(pc0 + cc)] = pattern_color

        for rr, cc, color in spec["marker_cells"]:
            grid[int(rr), int(cc)] = int(color)

    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, Any]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        pr0, pc0 = map(int, spec["pattern_origin"])
        frame_color = int(spec["frame_color"])
        pattern = spec["pattern"]
        line_colors = spec["marker_lines"]
        direction = spec["direction"]
        copy_r0, copy_c0, _copy_r1, _copy_c1 = map(int, spec["copy_bbox"])

        for rr, cc in np.argwhere(pattern):
            src_r = int(pr0 + rr)
            src_c = int(pc0 + cc)
            if direction == "up":
                dst_r = int(src_r + (copy_r0 - pr0))
                dst_c = src_c
                out[dst_r, dst_c] = int(line_colors.get(dst_c, frame_color))
            elif direction == "down":
                dst_r = int(src_r + (copy_r0 - pr0))
                dst_c = src_c
                out[dst_r, dst_c] = int(line_colors.get(dst_c, frame_color))
            elif direction == "left":
                dst_r = src_r
                dst_c = int(src_c + (copy_c0 - pc0))
                out[dst_r, dst_c] = int(line_colors.get(dst_r, frame_color))
            else:
                dst_r = src_r
                dst_c = int(src_c + (copy_c0 - pc0))
                out[dst_r, dst_c] = int(line_colors.get(dst_r, frame_color))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice([1, 2])
    specs = [_make_object_spec(rng) for _ in range(object_count)]
    if not any(len(spec["marker_offsets"]) < (spec["pattern_w"] if spec["direction"] in ("up", "down") else spec["pattern_h"]) for spec in specs):
        return None

    min_h = max(int(spec["env_h"]) for spec in specs)
    min_w = max(int(spec["env_w"]) for spec in specs)
    grid_h = rng.randint(max(14, min_h), 30)
    grid_w = rng.randint(max(14, min_w), 30)

    order = sorted(range(object_count), key=lambda i: int(specs[i]["env_h"]) * int(specs[i]["env_w"]), reverse=True)
    placed_rects: list[tuple[int, int, int, int]] = []
    origins: dict[int, tuple[int, int]] = {}
    for idx in order:
        spec = specs[idx]
        placed = False
        for _ in range(160):
            env_h = int(spec["env_h"])
            env_w = int(spec["env_w"])
            env_r = rng.randint(0, grid_h - env_h)
            env_c = rng.randint(0, grid_w - env_w)
            rect = (env_r, env_c, env_r + env_h - 1, env_c + env_w - 1)
            if any(_rects_overlap(rect, other, pad=1) for other in placed_rects):
                continue
            origins[idx] = (env_r, env_c)
            placed_rects.append(rect)
            placed = True
            break
        if not placed:
            return None

    for idx, spec in enumerate(specs):
        _finalize_object(rng, spec, origins[idx])

    inp = _render_input((grid_h, grid_w), specs)
    expected = _expected_output(inp, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
