from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
DIRS = ("top", "bottom", "left", "right")
STATUS_ORDER = ("outer_only", "inner_only", "both", "decoy")
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _segment_from_component(comp: dict) -> dict | None:
    coords = comp["coords"]
    color = int(comp["color"])
    if coords.shape[0] == 1:
        r, c = map(int, coords[0])
        return {"kind": "seed", "color": color, "r": r, "c": c}

    rows = set(map(int, coords[:, 0]))
    cols = set(map(int, coords[:, 1]))
    if len(rows) == 1 and coords.shape[0] >= 3:
        row = next(iter(rows))
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        if c1 - c0 + 1 == coords.shape[0]:
            return {
                "kind": "horizontal",
                "color": color,
                "row": row,
                "c0": c0,
                "c1": c1,
                "coords": coords,
            }
    if len(cols) == 1 and coords.shape[0] >= 3:
        col = next(iter(cols))
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        if r1 - r0 + 1 == coords.shape[0]:
            return {
                "kind": "vertical",
                "color": color,
                "col": col,
                "r0": r0,
                "r1": r1,
                "coords": coords,
            }
    return None


def _rectangles_from_segments(segments: list[dict]) -> list[dict]:
    horizontals = [seg for seg in segments if seg["kind"] == "horizontal"]
    verticals = [seg for seg in segments if seg["kind"] == "vertical"]
    vertical_map = {
        (int(seg["col"]), int(seg["r0"]), int(seg["r1"])): seg for seg in verticals
    }

    rectangles: dict[tuple[int, int, int, int], dict] = {}
    for i in range(len(horizontals)):
        for j in range(i + 1, len(horizontals)):
            top_like = horizontals[i]
            bottom_like = horizontals[j]
            if int(top_like["c0"]) != int(bottom_like["c0"]) or int(top_like["c1"]) != int(bottom_like["c1"]):
                continue
            r0, r1 = sorted((int(top_like["row"]), int(bottom_like["row"])))
            c0 = int(top_like["c0"]) - 1
            c1 = int(top_like["c1"]) + 1
            left = vertical_map.get((c0, r0, r1))
            right = vertical_map.get((c1, r0, r1))
            if left is None or right is None:
                continue
            top = top_like if int(top_like["row"]) == r0 else bottom_like
            bottom = bottom_like if int(bottom_like["row"]) == r1 else top_like
            bbox = (r0, c0, r1, c1)
            rectangles[bbox] = {
                "bbox": bbox,
                "top": int(top["color"]),
                "bottom": int(bottom["color"]),
                "left": int(left["color"]),
                "right": int(right["color"]),
            }

    rects = list(rectangles.values())
    assert len(rects) == 2, "expected exactly two rectangular frames"
    return rects


def _contains(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[0] and a[1] < b[1] and a[2] > b[2] and a[3] > b[3]


def _parse_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg)
    parsed = [_segment_from_component(comp) for comp in components]
    assert all(item is not None for item in parsed), "unexpected non-frame object"
    items = [item for item in parsed if item is not None]
    segments = [item for item in items if item["kind"] != "seed"]
    seeds = [item for item in items if item["kind"] == "seed"]
    assert segments, "expected frame segments"
    rectangles = _rectangles_from_segments(segments)
    rect_a, rect_b = rectangles
    if _contains(rect_a["bbox"], rect_b["bbox"]):
        outer, inner = rect_a, rect_b
    else:
        assert _contains(rect_b["bbox"], rect_a["bbox"]), "frames must be nested"
        outer, inner = rect_b, rect_a

    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    assert (or0 + or1) % 2 == 0 and (oc0 + oc1) % 2 == 0, "outer frame must have an integer center"
    assert (or0 + or1) == (ir0 + ir1) and (oc0 + oc1) == (ic0 + ic1), "frames must share one center"
    center = ((or0 + or1) // 2, (oc0 + oc1) // 2)

    frame_mask = np.zeros(g.shape, dtype=bool)
    for seg in segments:
        coords = seg["coords"]
        frame_mask[coords[:, 0], coords[:, 1]] = True

    return outer, inner, seeds, frame_mask, bg, center


def _seed_direction(seed: dict, outer: dict, center: tuple[int, int]) -> str:
    r = int(seed["r"])
    c = int(seed["c"])
    cr, cc = center
    r0, c0, r1, c1 = outer["bbox"]
    if c == cc and r < r0:
        return "top"
    if c == cc and r > r1:
        return "bottom"
    if r == cr and c < c0:
        return "left"
    if r == cr and c > c1:
        return "right"
    raise AssertionError("seed must lie on one of the four rays pointing to the shared center")


def _direction_params(direction: str, outer: dict, inner: dict) -> dict:
    or0, oc0, or1, oc1 = outer["bbox"]
    ir0, ic0, ir1, ic1 = inner["bbox"]
    if direction == "top":
        return {"axis": "row", "step": 1, "outer_hit": or0, "inner_hit": ir0, "far_inner": ir1}
    if direction == "bottom":
        return {"axis": "row", "step": -1, "outer_hit": or1, "inner_hit": ir1, "far_inner": ir0}
    if direction == "left":
        return {"axis": "col", "step": 1, "outer_hit": oc0, "inner_hit": ic0, "far_inner": ic1}
    if direction == "right":
        return {"axis": "col", "step": -1, "outer_hit": oc1, "inner_hit": ic1, "far_inner": ic0}
    raise ValueError(f"unknown direction: {direction}")


def _paint_segment(out: np.ndarray, seed: dict, direction: str, stop_scalar: int) -> None:
    color = int(seed["color"])
    r = int(seed["r"])
    c = int(seed["c"])
    if direction == "top":
        for rr in range(r, stop_scalar + 1):
            out[rr, c] = color
    elif direction == "bottom":
        for rr in range(r, stop_scalar - 1, -1):
            out[rr, c] = color
    elif direction == "left":
        for cc in range(c, stop_scalar + 1):
            out[r, cc] = color
    else:
        for cc in range(c, stop_scalar - 1, -1):
            out[r, cc] = color


def _stop_scalar(seed: dict, direction: str, outer: dict, inner: dict) -> int:
    color = int(seed["color"])
    params = _direction_params(direction, outer, inner)
    step = int(params["step"])
    outer_side = int(outer[direction])
    inner_side = int(inner[direction])
    if color != outer_side:
        return int(params["outer_hit"] - step)
    if color != inner_side:
        return int(params["inner_hit"] - step)
    return int(params["far_inner"] - step)


def _render_with_structure(grid: np.ndarray, outer: dict, inner: dict, seeds: list[dict], frame_mask: np.ndarray, center: tuple[int, int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    seen_dirs = set()
    for seed in seeds:
        direction = _seed_direction(seed, outer, center)
        assert direction not in seen_dirs, "expected at most one seed on each center ray"
        seen_dirs.add(direction)
        stop = _stop_scalar(seed, direction, outer, inner)
        _paint_segment(out, seed, direction, stop)
    out[frame_mask] = np.asarray(grid, dtype=int)[frame_mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot each outside seed straight toward the shared center of two nested frames, letting each gate color decide whether the ray passes or stops.

    Input structure:
    - A black background contains one outer hollow rectangle and one inner hollow rectangle sharing the same center.
    - Each rectangle side is monochromatic, and single colored seeds sit outside the outer frame on the four row-or-column rays pointing directly at that center.
    - The seeds are separate from the frames and may match the color of a gate on their own path or of some unrelated side elsewhere.

    Rule:
    - From each seed, draw a straight line of the seed's color toward the shared center.
    - When the line reaches the outer side on its path, it passes only if the seed color matches that side; otherwise it stops one cell before the side.
    - A line that passes the outer gate is tested again at the corresponding inner side; if it also passes there, it continues across the inner cavity and stops at the opposite inner wall's inside-adjacent cell.
    - Frame cells themselves keep their original side colors.

    Key insight:
    - Side colors do not tell the line where to connect; they act as sequential permissions on the fixed ray toward the center.
    """
    outer, inner, seeds, frame_mask, _bg, center = _parse_structure(np.asarray(grid, dtype=int))
    return _render_with_structure(np.asarray(grid, dtype=int), outer, inner, seeds, frame_mask, center)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], side_colors: dict[str, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 + 1 : c1] = int(side_colors["top"])
    grid[r1, c0 + 1 : c1] = int(side_colors["bottom"])
    grid[r0 : r1 + 1, c0] = int(side_colors["left"])
    grid[r0 : r1 + 1, c1] = int(side_colors["right"])


def _sample_bbox() -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    for _ in range(200):
        outer_h = random.choice((9, 11, 13, 15, 17))
        outer_w = random.choice((9, 11, 13, 15, 17))
        inner_h = random.choice(tuple(range(5, outer_h - 3, 2)))
        inner_w = random.choice(tuple(range(5, outer_w - 3, 2)))
        top_margin = random.randint(2, 6)
        bottom_margin = random.randint(2, 6)
        left_margin = random.randint(2, 6)
        right_margin = random.randint(2, 6)
        height = top_margin + outer_h + bottom_margin
        width = left_margin + outer_w + right_margin
        if max(height, width) > 30:
            continue
        outer = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
        dr = (outer_h - inner_h) // 2
        dc = (outer_w - inner_w) // 2
        inner = (outer[0] + dr, outer[1] + dc, outer[2] - dr, outer[3] - dc)
        center = ((outer[0] + outer[2]) // 2, (outer[1] + outer[3]) // 2)
        return outer, inner, center, (height, width)
    raise RuntimeError("failed to sample nested frame boxes")


def _sample_colors() -> tuple[dict[str, str], dict[str, int], dict[str, int], dict[str, int]]:
    directions = list(DIRS)
    random.shuffle(directions)
    status_by_dir = dict(zip(directions, STATUS_ORDER))

    outer_palette = list(random.sample(PALETTE, 4))
    outer_colors = {direction: color for direction, color in zip(DIRS, outer_palette)}

    remaining = [color for color in PALETTE if color not in outer_palette]
    extra_inner = random.sample(remaining, 3)
    inner_colors: dict[str, int] = {}
    extra_iter = iter(extra_inner)
    both_dir = next(direction for direction, status in status_by_dir.items() if status == "both")
    for direction in DIRS:
        if direction == both_dir:
            inner_colors[direction] = outer_colors[direction]
        else:
            inner_colors[direction] = next(extra_iter)

    outer_only_dir = next(direction for direction, status in status_by_dir.items() if status == "outer_only")
    decoy_dir = next(direction for direction, status in status_by_dir.items() if status == "decoy")
    seed_colors: dict[str, int] = {}
    for direction, status in status_by_dir.items():
        if status == "outer_only":
            seed_colors[direction] = outer_colors[direction]
        elif status == "inner_only":
            seed_colors[direction] = inner_colors[direction]
        elif status == "both":
            seed_colors[direction] = outer_colors[direction]
        else:
            seed_colors[direction] = inner_colors[outer_only_dir]

    assert seed_colors[decoy_dir] != outer_colors[decoy_dir]
    assert seed_colors[decoy_dir] != inner_colors[decoy_dir]
    return status_by_dir, outer_colors, inner_colors, seed_colors


def _seed_coord(direction: str, outer: tuple[int, int, int, int], center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = outer
    cr, cc = center
    height, width = shape
    if direction == "top":
        return random.randint(0, r0 - 1), cc
    if direction == "bottom":
        return random.randint(r1 + 1, height - 1), cc
    if direction == "left":
        return cr, random.randint(0, c0 - 1)
    if direction == "right":
        return cr, random.randint(c1 + 1, width - 1)
    raise ValueError(f"unknown direction: {direction}")


def _blueprint_expected(shape: tuple[int, int], outer: dict, inner: dict, seeds: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    _paint_frame(grid, outer["bbox"], outer)
    _paint_frame(grid, inner["bbox"], inner)
    frame_input = grid.copy()
    frame_mask = frame_input != BACKGROUND
    for seed in seeds:
        grid[int(seed["r"]), int(seed["c"])] = int(seed["color"])
    for seed in seeds:
        direction = seed["direction"]
        stop = _stop_scalar(seed, direction, outer, inner)
        _paint_segment(grid, seed, direction, stop)
    grid[frame_mask] = frame_input[frame_mask]
    return grid


def _validate_blueprint(
    outer: dict,
    inner: dict,
    seeds: list[dict],
    status_by_dir: dict[str, str],
    seed_colors: dict[str, int],
) -> None:
    assert {seed["direction"] for seed in seeds} == set(DIRS)
    stop_cells = set()
    matched_statuses = set()
    for seed in seeds:
        direction = seed["direction"]
        color = int(seed["color"])
        outer_side = int(outer[direction])
        inner_side = int(inner[direction])
        stop = _stop_scalar(seed, direction, outer, inner)
        if direction in ("top", "bottom"):
            stop_cells.add((stop, int(seed["c"])))
        else:
            stop_cells.add((int(seed["r"]), stop))
        if color == outer_side and color == inner_side:
            matched_statuses.add("both")
        elif color == outer_side and color != inner_side:
            matched_statuses.add("outer_only")
        elif color != outer_side and color == inner_side:
            matched_statuses.add("inner_only")
        else:
            matched_statuses.add("decoy")
    assert matched_statuses == set(STATUS_ORDER)
    assert len(stop_cells) == 4

    decoy_dir = next(direction for direction, status in status_by_dir.items() if status == "decoy")
    outer_only_dir = next(direction for direction, status in status_by_dir.items() if status == "outer_only")
    assert seed_colors[decoy_dir] == inner[outer_only_dir]
    assert seed_colors[decoy_dir] not in {outer[decoy_dir], inner[decoy_dir]}


def _gen_attempt() -> np.ndarray | None:
    outer_bbox, inner_bbox, center, shape = _sample_bbox()
    status_by_dir, outer_colors, inner_colors, seed_colors = _sample_colors()
    height, width = shape

    outer = {"bbox": outer_bbox, **outer_colors}
    inner = {"bbox": inner_bbox, **inner_colors}

    seeds = []
    for direction in DIRS:
        r, c = _seed_coord(direction, outer_bbox, center, (height, width))
        seeds.append(
            {
                "direction": direction,
                "color": int(seed_colors[direction]),
                "r": int(r),
                "c": int(c),
            }
        )

    _validate_blueprint(outer, inner, seeds, status_by_dir, seed_colors)

    inp = np.zeros((height, width), dtype=int)
    _paint_frame(inp, outer_bbox, outer_colors)
    _paint_frame(inp, inner_bbox, inner_colors)
    for seed in seeds:
        inp[int(seed["r"]), int(seed["c"])] = int(seed["color"])

    expected = _blueprint_expected(inp.shape, outer, inner, seeds)

    if len(set(map(int, inp.ravel())) - {BACKGROUND}) < 6:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    parsed_outer, parsed_inner, parsed_seeds, _frame_mask, _bg, parsed_center = _parse_structure(inp)
    if parsed_outer["bbox"] != outer_bbox or parsed_inner["bbox"] != inner_bbox or parsed_center != center:
        return None
    if {(_seed_direction(seed, parsed_outer, parsed_center), int(seed["color"])) for seed in parsed_seeds} != {
        (seed["direction"], int(seed["color"])) for seed in seeds
    }:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-gate ARC input")
