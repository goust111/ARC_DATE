from __future__ import annotations

import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
SOURCE_COLORS = [1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _detect_axis(grid: np.ndarray) -> tuple[str, int]:
    h, w = grid.shape
    yellow = grid == YELLOW
    row_hits = [r for r in range(h) if np.all(grid[r, :] == YELLOW)]
    col_hits = [c for c in range(w) if np.all(grid[:, c] == YELLOW)]
    candidates = [("h", int(r)) for r in row_hits] + [("v", int(c)) for c in col_hits]
    assert len(candidates) == 1, "expected exactly one full yellow mirror line"
    orient, axis = candidates[0]
    expected_len = w if orient == "h" else h
    assert int(yellow.sum()) == expected_len, "yellow pixels must belong only to the mirror line"
    return orient, axis


def _source_coords(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid != BLACK) & (grid != YELLOW) & (grid != GREY)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    assert coords, "expected at least one colored source pixel"
    return sorted(coords)


def _source_side(orient: str, axis: int, source: tuple[int, int]) -> int:
    r, c = source
    pos = c if orient == "v" else r
    side = -1 if pos < axis else 1 if pos > axis else 0
    assert side != 0, "source pixels may not lie on the mirror line"
    return side


def _trace_source(
    grid: np.ndarray,
    orient: str,
    axis: int,
    source: tuple[int, int],
    *,
    block_before: bool,
    stop_after: bool,
) -> tuple[int, int, int] | None:
    r, c = source
    color = int(grid[r, c])
    side = _source_side(orient, axis, source)
    step = -side
    distance = abs((c if orient == "v" else r) - axis)

    if orient == "v":
        if block_before:
            for cc in range(c + step, axis, step):
                if int(grid[r, cc]) == GREY:
                    return None
        target_c = axis + step * distance
        if not (0 <= target_c < grid.shape[1]):
            return None
        final_c = target_c
        if stop_after:
            for cc in range(axis + step, target_c + step, step):
                if int(grid[r, cc]) == GREY:
                    final_c = cc - step
                    break
        if final_c == axis:
            return None
        return r, int(final_c), color

    if block_before:
        for rr in range(r + step, axis, step):
            if int(grid[rr, c]) == GREY:
                return None
    target_r = axis + step * distance
    if not (0 <= target_r < grid.shape[0]):
        return None
    final_r = target_r
    if stop_after:
        for rr in range(axis + step, target_r + step, step):
            if int(grid[rr, c]) == GREY:
                final_r = rr - step
                break
    if final_r == axis:
        return None
    return int(final_r), c, color


def _solve_grid(grid: np.ndarray, *, block_before: bool, stop_after: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    orient, axis = _detect_axis(g)
    sources = _source_coords(g)
    side_set = {_source_side(orient, axis, source) for source in sources}
    assert len(side_set) == 1, "all source pixels must lie on the same side of the mirror"

    out = g.copy()
    for source in sources:
        traced = _trace_source(g, orient, axis, source, block_before=block_before, stop_after=stop_after)
        if traced is None:
            continue
        rr, cc, color = traced
        if int(g[rr, cc]) == BLACK:
            out[rr, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror single colored source pixels across a full yellow axis, with grey walls blocking before the mirror and truncating after it.

    Input structure:
    - A black background contains exactly one complete yellow horizontal or vertical line as the mirror axis.
    - Several non-yellow, non-grey source pixels lie strictly on one side of that axis.
    - Grey wall cells may appear on both sides of the axis.

    Rule:
    - For each source pixel, follow the perpendicular ray from the source toward the mirror.
    - If any grey wall lies between the source and the yellow line, that source creates no image.
    - Otherwise reflect the source to the equally distant cell on the other side; if a grey wall appears on that post-mirror segment, place the image on the last cell before the first such wall.
    - Keep the original sources, the yellow line, and all walls unchanged; only add the reflected image pixels.

    Key insight:
    - The same grey wall color has two roles depending on which side of the mirror it occupies along a source ray: before the mirror it cancels the reflection, after the mirror it shortens the reflected landing position.
    """
    return _solve_grid(grid, block_before=True, stop_after=True)


def _expected_from_meta(grid: np.ndarray, meta: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for item in meta["sources"]:
        image = item["image"]
        if image is None:
            continue
        rr, cc = image
        assert int(grid[rr, cc]) == BLACK
        out[rr, cc] = int(item["color"])
    return out


def _build_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    orient = random.choice(["h", "v"])
    height = random.randint(10, 20)
    width = random.randint(10, 20)

    if orient == "v":
        if width < 10:
            return None
        axis = random.randint(4, width - 5)
        max_dist = min(axis, width - axis - 1, 6)
        max_sources = min(6, height - 2, len(SOURCE_COLORS))
        if max_dist < 2 or max_sources < 3:
            return None
        n_sources = random.randint(3, max_sources)
        rays = sorted(random.sample(range(1, height - 1), n_sources))
    else:
        if height < 10:
            return None
        axis = random.randint(4, height - 5)
        max_dist = min(axis, height - axis - 1, 6)
        max_sources = min(6, width - 2, len(SOURCE_COLORS))
        if max_dist < 2 or max_sources < 3:
            return None
        n_sources = random.randint(3, max_sources)
        rays = sorted(random.sample(range(1, width - 1), n_sources))

    grid = np.zeros((height, width), dtype=int)
    if orient == "v":
        grid[:, axis] = YELLOW
    else:
        grid[axis, :] = YELLOW

    source_side = random.choice([-1, 1])
    colors = random.sample(SOURCE_COLORS, n_sources)
    branches = ["clear", "pre", "post"]
    while len(branches) < n_sources:
        branches.append(random.choice(["clear", "pre", "post"]))
    random.shuffle(branches)

    sources_meta: list[dict] = []
    for ray, color, branch in zip(rays, colors, branches):
        distance = random.randint(2, max_dist)
        step = -source_side

        if orient == "v":
            source = (ray, axis + source_side * distance)
            target = (ray, axis + step * distance)
        else:
            source = (axis + source_side * distance, ray)
            target = (axis + step * distance, ray)

        if not _in_bounds(grid.shape, *source) or not _in_bounds(grid.shape, *target):
            return None
        if int(grid[source]) != BLACK or int(grid[target]) != BLACK:
            return None

        grid[source] = color
        pre_wall = None
        post_wall = None
        image = target

        if branch == "pre":
            offset = random.randint(1, distance - 1)
            if orient == "v":
                pre_wall = (source[0], source[1] + step * offset)
            else:
                pre_wall = (source[0] + step * offset, source[1])
            if int(grid[pre_wall]) != BLACK:
                return None
            grid[pre_wall] = GREY
            image = None
        elif branch == "post":
            wall_dist = random.randint(2, distance)
            if orient == "v":
                post_wall = (source[0], axis + step * wall_dist)
                image = (source[0], axis + step * (wall_dist - 1))
            else:
                post_wall = (axis + step * wall_dist, source[1])
                image = (axis + step * (wall_dist - 1), source[1])
            if int(grid[post_wall]) != BLACK or image == source:
                return None
            grid[post_wall] = GREY

        sources_meta.append(
            {
                "branch": branch,
                "color": int(color),
                "source": source,
                "target": target,
                "image": image,
                "pre_wall": pre_wall,
                "post_wall": post_wall,
                "distance": int(distance),
                "ray": int(ray),
            }
        )

    unused_rays = [r for r in (range(1, height - 1) if orient == "v" else range(1, width - 1)) if r not in set(rays)]
    random.shuffle(unused_rays)
    extra_count = random.randint(0, min(3, len(unused_rays)))
    for ray in unused_rays[:extra_count]:
        side = random.choice([-1, 1])
        if orient == "v":
            max_wall_dist = axis - 1 if side < 0 else width - axis - 2
        else:
            max_wall_dist = axis - 1 if side < 0 else height - axis - 2
        if max_wall_dist < 1:
            continue
        wall_dist = random.randint(1, max_wall_dist)
        if orient == "v":
            wall = (ray, axis + side * wall_dist)
        else:
            wall = (axis + side * wall_dist, ray)
        if int(grid[wall]) == BLACK:
            grid[wall] = GREY

    meta = {"orient": orient, "axis": int(axis), "source_side": int(source_side), "sources": sources_meta}
    expected = _expected_from_meta(grid, meta)
    return grid, expected, meta


def _validate_case(grid: np.ndarray, expected: np.ndarray, meta: dict) -> bool:
    g = np.asarray(grid, dtype=int)
    out = np.asarray(expected, dtype=int)
    if g.shape != out.shape or g.ndim != 2 or max(g.shape) > 30:
        return False
    if not np.array_equal(out, _expected_from_meta(g, meta)):
        return False

    try:
        orient, axis = _detect_axis(g)
    except AssertionError:
        return False
    if orient != meta["orient"] or axis != meta["axis"]:
        return False

    if len(meta["sources"]) < 3:
        return False
    branch_counts = {name: 0 for name in ("clear", "pre", "post")}
    added_images = []

    try:
        actual_sources = _source_coords(g)
    except AssertionError:
        return False

    if sorted(item["source"] for item in meta["sources"]) != actual_sources:
        return False

    side_set = {_source_side(orient, axis, source) for source in actual_sources}
    if side_set != {meta["source_side"]}:
        return False

    for item in meta["sources"]:
        branch = item["branch"]
        branch_counts[branch] += 1
        source = item["source"]
        target = item["target"]
        image = item["image"]
        pre_wall = item["pre_wall"]
        post_wall = item["post_wall"]

        if int(g[source]) != int(item["color"]):
            return False
        if source == target or not _in_bounds(g.shape, *target):
            return False

        sr, sc = source
        tr, tc = target
        if orient == "v" and sr != tr:
            return False
        if orient == "h" and sc != tc:
            return False

        side = _source_side(orient, axis, source)
        step = -side
        source_to_axis = []
        axis_to_target = []
        if orient == "v":
            source_to_axis = [(sr, cc) for cc in range(sc + step, axis, step)]
            axis_to_target = [(sr, cc) for cc in range(axis + step, tc + step, step)]
        else:
            source_to_axis = [(rr, sc) for rr in range(sr + step, axis, step)]
            axis_to_target = [(rr, sc) for rr in range(axis + step, tr + step, step)]

        if branch == "clear":
            if any(int(g[cell]) == GREY for cell in source_to_axis + axis_to_target):
                return False
            if image != target or int(out[image]) != int(item["color"]):
                return False
            added_images.append(image)
        elif branch == "pre":
            if pre_wall is None or post_wall is not None or image is not None:
                return False
            if int(g[pre_wall]) != GREY:
                return False
            if pre_wall not in source_to_axis:
                return False
            if any(int(g[cell]) == GREY for cell in axis_to_target):
                return False
            if any(int(out[cell]) != int(g[cell]) for cell in axis_to_target):
                return False
        else:
            if post_wall is None or pre_wall is not None or image is None:
                return False
            if int(g[post_wall]) != GREY:
                return False
            if any(int(g[cell]) == GREY for cell in source_to_axis):
                return False
            if post_wall not in axis_to_target:
                return False
            first_wall = next((cell for cell in axis_to_target if int(g[cell]) == GREY), None)
            if first_wall != post_wall:
                return False
            if image == target and post_wall == target:
                return False
            if int(out[image]) != int(item["color"]):
                return False
            added_images.append(image)

    if any(count == 0 for count in branch_counts.values()):
        return False
    if len(added_images) != len(set(added_images)):
        return False
    if len(added_images) < 2:
        return False

    if not np.array_equal(main(g), out):
        return False
    if not np.array_equal(_solve_grid(g, block_before=True, stop_after=True), out):
        return False
    if np.array_equal(_solve_grid(g, block_before=False, stop_after=True), out):
        return False
    if np.array_equal(_solve_grid(g, block_before=True, stop_after=False), out):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, expected, meta = built
        if _validate_case(grid, expected, meta):
            return grid
    raise RuntimeError("failed to generate a valid mirror-with-walls ARC input")
