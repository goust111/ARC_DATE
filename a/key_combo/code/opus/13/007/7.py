import numpy as np


BACKGROUND = 0
COLOR_POOL = [2, 1, 3, 4, 6, 7]
MAX_GENERATION_ATTEMPTS = 400


def _bbox_h(bbox):
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox):
    return int(bbox[3] - bbox[1] + 1)


def _bbox_area(bbox):
    return _bbox_h(bbox) * _bbox_w(bbox)


def _strictly_contains(outer, inner):
    return (
        int(outer[0]) < int(inner[0]) < int(inner[2]) < int(outer[2])
        and int(outer[1]) < int(inner[1]) < int(inner[3]) < int(outer[3])
    )


def _border_coords(bbox, shape=None):
    r0, c0, r1, c1 = map(int, bbox)
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0 + 1, r1):
        coords.add((r, c0))
        coords.add((r, c1))
    if shape is None:
        return coords
    h, w = map(int, shape)
    return {(r, c) for (r, c) in coords if 0 <= r < h and 0 <= c < w}


def _draw_frame(grid, bbox, color):
    for r, c in _border_coords(bbox):
        grid[r, c] = int(color)


def _randint(rng, lo, hi):
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_origin(rng, limit, touch_prob=0.38):
    limit = int(limit)
    if limit <= 0:
        return 0
    roll = float(rng.random())
    if roll < touch_prob / 2.0:
        return 0
    if roll < touch_prob:
        return limit
    return _randint(rng, 0, limit)


def _random_inner_bbox(rng, parent_bbox, min_h, min_w):
    pr0, pc0, pr1, pc1 = map(int, parent_bbox)
    max_top = pr1 - min_h
    max_left = pc1 - min_w
    top_margin = _randint(rng, 1, min(2, max_top - pr0))
    left_margin = _randint(rng, 1, min(2, max_left - pc0))
    max_bottom_margin = pr1 - (pr0 + top_margin + min_h - 1)
    max_right_margin = pc1 - (pc0 + left_margin + min_w - 1)
    bottom_margin = _randint(rng, 1, min(2, max_bottom_margin))
    right_margin = _randint(rng, 1, min(2, max_right_margin))
    return (
        pr0 + top_margin,
        pc0 + left_margin,
        pr1 - bottom_margin,
        pc1 - right_margin,
    )


def _detect_frames(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    frames = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, f"missing pixels for color {color}"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        assert _bbox_h(bbox) >= 3 and _bbox_w(bbox) >= 3, f"frame too small: {bbox}"
        actual = {tuple(map(int, rc)) for rc in coords}
        expected = _border_coords(bbox)
        assert actual == expected, f"color {color} is not a single hollow rectangle"
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "coords": actual,
                "area": _bbox_area(bbox),
            }
        )

    areas = [frame["area"] for frame in frames]
    assert len(set(areas)) == len(areas), "frame areas must be unique"

    for frame in frames:
        depth = 0
        for other in frames:
            if other is frame:
                continue
            if _strictly_contains(other["bbox"], frame["bbox"]):
                depth += 1
        frame["depth"] = depth

    return frames


def _solve(grid, return_meta=False):
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    survivors = [frame for frame in frames if frame["depth"] % 2 == 1]
    erased = [frame for frame in frames if frame["depth"] % 2 == 0]
    assert survivors, "expected at least one odd-depth frame"
    assert erased, "expected at least one even-depth frame"

    ghost_union = set()
    for frame in erased:
        ghost_union |= frame["coords"]

    out = g.copy()
    for frame in erased:
        for r, c in frame["coords"]:
            out[r, c] = BACKGROUND

    original_survivor_pixels = set()
    for frame in survivors:
        original_survivor_pixels |= frame["coords"]

    claims = {}
    stop_records = []
    for frame in survivors:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        layer = 1
        while True:
            ring = _border_coords((r0 - layer, c0 - layer, r1 + layer, c1 + layer), shape=g.shape)
            assert ring, "growth ring should reach the grid before disappearing"
            edge_hit = any(
                r == 0 or c == 0 or r == g.shape[0] - 1 or c == g.shape[1] - 1
                for r, c in ring
            )
            ghost_hit = bool(ring & ghost_union)
            for coord in ring:
                claims.setdefault(coord, []).append((int(frame["area"]), int(frame["color"])))
            if edge_hit or ghost_hit:
                stop_records.append(
                    {
                        "color": int(frame["color"]),
                        "depth": int(frame["depth"]),
                        "area": int(frame["area"]),
                        "layers": int(layer),
                        "edge_hit": bool(edge_hit),
                        "ghost_hit": bool(ghost_hit),
                    }
                )
                break
            layer += 1

    collision_pixels = 0
    for coord, options in claims.items():
        if coord in original_survivor_pixels:
            continue
        if len(options) > 1:
            collision_pixels += 1
        area, color = min(options, key=lambda item: (int(item[0]), int(item[1])))
        out[coord] = int(color)

    if return_meta:
        return out, {
            "frame_count": len(frames),
            "survivor_count": len(survivors),
            "erased_count": len(erased),
            "collision_pixels": int(collision_pixels),
            "stop_records": stop_records,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase even-depth nested frames and let odd-depth frames grow until ghost borders or the grid edge.

    Input structure:
    - The grid is a black background containing 4-6 one-pixel-thick hollow rectangular frames.
    - Each frame is monochrome, axis-aligned, and uses a distinct foreground color.
    - Some frames are strictly nested inside others, creating multiple containment depths.

    Rule:
    - Compute each frame's nesting depth by counting how many other frames strictly contain it.
    - Erase every even-depth frame, but remember its original border as an invisible ghost boundary.
    - Keep every odd-depth frame and grow outward concentric rectangular rings of the same color.
    - Draw each stopping layer too; stop a frame once a new ring touches the grid edge or any ghost boundary.
    - If multiple growing rings claim the same pixel, keep the color of the frame with the smaller original bbox area.
    - Preserve the original pixels of surviving frames, and never redraw erased frames in their old colors.

    Key insight:
    - Deleted frames still matter because their invisible borders cap later growth, so the output depends on containment depth and remembered geometry rather than only visible survivors.
    """
    return _solve(grid)


def _build_layout(rng, height, width):
    orientation = "horizontal" if float(rng.random()) < 0.5 else "vertical"
    child_count = 3 if float(rng.random()) < 0.5 else 2
    deep_len = 3 if float(rng.random()) < 0.55 else 2

    if orientation == "horizontal":
        common_h = _randint(rng, 7, 9)
        widths = [_randint(rng, 7, 9)]
        for _ in range(child_count - 1):
            widths.append(_randint(rng, 3, 6))
        gaps = [_randint(rng, 1, 3) for _ in range(child_count - 1)]
        mt, mb, ml, mr = (_randint(rng, 2, 4) for _ in range(4))
        outer_h = common_h + mt + mb
        outer_w = sum(widths) + sum(gaps) + ml + mr
        if outer_h > height or outer_w > width:
            return None, None
        outer_r0 = _sample_origin(rng, height - outer_h)
        outer_c0 = _sample_origin(rng, width - outer_w)
        outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
        frames = [outer]
        child_r0 = outer_r0 + mt
        child_r1 = child_r0 + common_h - 1
        cur_c = outer_c0 + ml
        children = []
        for idx, child_w in enumerate(widths):
            child = (child_r0, cur_c, child_r1, cur_c + child_w - 1)
            children.append(child)
            frames.append(child)
            cur_c += child_w
            if idx < len(gaps):
                cur_c += gaps[idx]
    else:
        common_w = _randint(rng, 7, 9)
        heights = [_randint(rng, 7, 9)]
        for _ in range(child_count - 1):
            heights.append(_randint(rng, 3, 6))
        gaps = [_randint(rng, 1, 3) for _ in range(child_count - 1)]
        mt, mb, ml, mr = (_randint(rng, 2, 4) for _ in range(4))
        outer_h = sum(heights) + sum(gaps) + mt + mb
        outer_w = common_w + ml + mr
        if outer_h > height or outer_w > width:
            return None, None
        outer_r0 = _sample_origin(rng, height - outer_h)
        outer_c0 = _sample_origin(rng, width - outer_w)
        outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
        frames = [outer]
        child_c0 = outer_c0 + ml
        child_c1 = child_c0 + common_w - 1
        cur_r = outer_r0 + mt
        children = []
        for idx, child_h in enumerate(heights):
            child = (cur_r, child_c0, cur_r + child_h - 1, child_c1)
            children.append(child)
            frames.append(child)
            cur_r += child_h
            if idx < len(gaps):
                cur_r += gaps[idx]

    host = children[0]
    inner_even = _random_inner_bbox(rng, host, 5, 5)
    frames.append(inner_even)
    if deep_len == 3:
        inner_odd = _random_inner_bbox(rng, inner_even, 3, 3)
        frames.append(inner_odd)

    areas = [_bbox_area(bbox) for bbox in frames]
    if len(set(areas)) != len(areas):
        return None, None

    return frames, {"orientation": orientation, "child_count": child_count, "deep_len": deep_len}


def _render_input(height, width, frames, colors):
    grid = np.full((int(height), int(width)), BACKGROUND, dtype=int)
    for bbox, color in zip(frames, colors):
        _draw_frame(grid, bbox, color)
    return grid


def _valid_generated_example(grid, layout_meta):
    try:
        out, meta = _solve(grid, return_meta=True)
    except AssertionError:
        return False

    if not np.any(out != grid):
        return False
    if meta["frame_count"] != len([color for color in np.unique(grid) if int(color) != BACKGROUND]):
        return False
    if meta["collision_pixels"] <= 0:
        return False
    if not any(record["ghost_hit"] for record in meta["stop_records"]):
        return False
    if layout_meta["deep_len"] == 3 and not any(record["depth"] == 3 for record in meta["stop_records"]):
        return False
    if layout_meta["child_count"] == 3 and meta["survivor_count"] < 3:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        height = _randint(rng, 17, 22)
        width = _randint(rng, 17, 22)
        frames, layout_meta = _build_layout(rng, height, width)
        if frames is None:
            continue
        colors = list(map(int, rng.choice(COLOR_POOL, size=len(frames), replace=False)))
        grid = _render_input(height, width, frames, colors)
        if _valid_generated_example(grid, layout_meta):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
