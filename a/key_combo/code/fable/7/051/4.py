import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    """Return the most common grid value."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude=()) -> list[int]:
    """Return sorted unique colors, excluding any requested values."""
    return sorted(set(map(int, grid.ravel())) - set(map(int, exclude)))


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return inclusive bbox (r0, c0, r1, c1) for an (N,2) coord array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_coords(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _is_corner(rect: tuple[int, int, int, int], coord: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = rect
    return coord in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}


def _gap_side(rect: tuple[int, int, int, int], gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = rect
    r, c = gap
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on the frame border")


def _inside_target(rect: tuple[int, int, int, int], gap: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    r, c = gap
    side = _gap_side(rect, gap)
    if side == "top":
        return r1 - 1, c
    if side == "bottom":
        return r0 + 1, c
    if side == "left":
        return r, c1 - 1
    return r, c0 + 1


def _outside_target(
    rect: tuple[int, int, int, int], gap: tuple[int, int], rank: int
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    r, c = gap
    side = _gap_side(rect, gap)
    if side == "top":
        return r0 - rank, c
    if side == "bottom":
        return r1 + rank, c
    if side == "left":
        return r, c0 - rank
    return r, c1 + rank


def _marker_match(
    rect: tuple[int, int, int, int], gap: tuple[int, int], marker: tuple[int, int]
) -> tuple[bool, int]:
    gr, gc = gap
    mr, mc = marker
    side = _gap_side(rect, gap)
    if side == "top" and mr < gr and mc == gc:
        return True, gr - mr
    if side == "bottom" and mr > gr and mc == gc:
        return True, mr - gr
    if side == "left" and mc < gc and mr == gr:
        return True, gc - mc
    if side == "right" and mc > gc and mr == gr:
        return True, mc - gc
    return False, -1


def _try_parse_with_wall_color(
    grid: np.ndarray, bg: int, wall_color: int, marker_color: int
):
    wall_coords = np.argwhere(grid == wall_color)
    if wall_coords.size == 0:
        return None
    rect = bbox_coords(wall_coords)
    r0, c0, r1, c1 = rect
    if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 7:
        return None

    border = set(_border_coords(rect))
    if any((int(r), int(c)) not in border for r, c in wall_coords):
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    gaps = []
    for cell in _border_coords(rect):
        value = int(grid[cell])
        if value == wall_color:
            continue
        if value != bg or _is_corner(rect, cell):
            return None
        gaps.append(cell)
    if not gaps:
        return None
    if len({r for r, _ in gaps}) != len(gaps):
        return None
    if len({c for _, c in gaps}) != len(gaps):
        return None

    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(grid == marker_color)]
    if not marker_coords:
        return None
    for mr, mc in marker_coords:
        if r0 <= mr <= r1 and c0 <= mc <= c1:
            return None
        if (
            (mr > 0 and int(grid[mr - 1, mc]) == marker_color)
            or (mr + 1 < grid.shape[0] and int(grid[mr + 1, mc]) == marker_color)
            or (mc > 0 and int(grid[mr, mc - 1]) == marker_color)
            or (mc + 1 < grid.shape[1] and int(grid[mr, mc + 1]) == marker_color)
        ):
            return None

    groups = {gap: [] for gap in gaps}
    for marker in marker_coords:
        matches = []
        for gap in gaps:
            ok, dist = _marker_match(rect, gap, marker)
            if ok:
                matches.append((gap, dist))
        if len(matches) != 1:
            return None
        gap, dist = matches[0]
        if dist < 2:
            return None
        groups[gap].append((dist, marker))

    if any(not entries for entries in groups.values()):
        return None

    for entries in groups.values():
        dists = sorted(dist for dist, _ in entries)
        for i in range(1, len(dists)):
            if dists[i] - dists[i - 1] < 2:
                return None

    seen_targets = set()
    for gap, entries in groups.items():
        inside = _inside_target(rect, gap)
        if inside in seen_targets:
            return None
        seen_targets.add(inside)
        for rank in range(1, len(entries)):
            outside = _outside_target(rect, gap, rank)
            if not (0 <= outside[0] < grid.shape[0] and 0 <= outside[1] < grid.shape[1]):
                return None
            if outside in seen_targets:
                return None
            seen_targets.add(outside)

    return {
        "bg": bg,
        "wall_color": wall_color,
        "marker_color": marker_color,
        "rect": rect,
        "gaps": tuple(sorted(gaps)),
        "groups": {gap: tuple(sorted(entries)) for gap, entries in groups.items()},
    }


def _parse_puzzle(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly one frame color and one marker color"

    parses = []
    for wall_color in colors:
        marker_color = colors[0] if colors[1] == wall_color else colors[1]
        parsed = _try_parse_with_wall_color(g, bg, wall_color, marker_color)
        if parsed is not None:
            parses.append(parsed)
    assert len(parses) == 1, "expected a unique frame interpretation"
    return parses[0]


def _solve_from_parsed(parsed: dict) -> np.ndarray:
    rect = parsed["rect"]
    wall_color = parsed["wall_color"]
    marker_color = parsed["marker_color"]
    bg = parsed["bg"]
    grid = parsed["grid"]

    out = grid.copy()
    out[out == marker_color] = bg

    events = []
    for gap, entries in parsed["groups"].items():
        for dist, marker in entries:
            events.append((int(dist), gap, marker))

    outside_counts = {gap: 0 for gap in parsed["gaps"]}
    passed = set()
    final_positions = []
    for _, gap, marker in sorted(events, key=lambda item: (item[0], item[1][0], item[1][1], item[2][0], item[2][1])):
        if gap not in passed:
            final_positions.append(_inside_target(rect, gap))
            passed.add(gap)
        else:
            outside_counts[gap] += 1
            final_positions.append(_outside_target(rect, gap, outside_counts[gap]))

    assert len(set(final_positions)) == len(final_positions), "final marker positions must be unique"
    for gap in parsed["gaps"]:
        out[gap] = wall_color
    for r, c in final_positions:
        out[r, c] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marker through its aligned one-use gap, then seal the frame and queue the rest outside.

    Input structure:
    - The grid has a dominant background color, one frame color, and one marker color.
    - The frame color forms a single-cell-thick hollow rectangle whose border contains one or more single-cell gaps.
    - Every marker pixel lies on the unique outward row or column ray of exactly one gap, with only background between neighboring markers and between the nearest marker and the frame.

    Rule:
    - Process all markers from nearest to farthest by their distance to the frame along their own gap ray.
    - The first marker for an unsealed gap passes through that gap and stops one cell before the opposite inner wall, then that gap becomes frame-colored forever.
    - Any later marker for the same gap can no longer enter, so it stops outside the frame, packed tightly against the sealed wall or the previous stopped marker on that same outward ray.
    - Erase all original marker positions; keep the completed solid frame and only the final marker positions.

    Key insight:
    - A gap is a single-use resource: the nearest aligned marker changes the fate of every farther marker on that same ray by sealing the door after its one pass.
    """
    parsed = _parse_puzzle(grid)
    parsed["grid"] = np.asarray(grid, dtype=int)
    return _solve_from_parsed(parsed)


def _render_input(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), spec["bg"], dtype=int)
    rect = spec["rect"]
    for cell in _border_coords(rect):
        if cell not in spec["gaps"]:
            grid[cell] = spec["wall_color"]
    for markers in spec["markers"].values():
        for marker in markers:
            grid[marker] = spec["marker_color"]
    return grid


def _render_output_from_spec(spec: dict) -> np.ndarray:
    grid = np.full((spec["height"], spec["width"]), spec["bg"], dtype=int)
    rect = spec["rect"]
    for cell in _border_coords(rect):
        grid[cell] = spec["wall_color"]

    final_positions = []
    for gap in spec["gaps"]:
        final_positions.append(_inside_target(rect, gap))
        for rank in range(1, len(spec["markers"][gap])):
            final_positions.append(_outside_target(rect, gap, rank))
    assert len(set(final_positions)) == len(final_positions)
    for cell in final_positions:
        grid[cell] = spec["marker_color"]
    return grid


def _sample_spaced_distances(limit: int, count: int) -> list[int] | None:
    choices = list(range(2, limit + 1))
    for _ in range(80):
        picked = sorted(random.sample(choices, count))
        if all(picked[i] - picked[i - 1] >= 2 for i in range(1, count)):
            return picked
    return None


def _marker_coord_for_gap(gap: tuple[int, int], side: str, dist: int) -> tuple[int, int]:
    r, c = gap
    if side == "top":
        return r - dist, c
    if side == "bottom":
        return r + dist, c
    if side == "left":
        return r, c - dist
    return r, c + dist


def _generate_spec() -> dict | None:
    bg, wall_color, marker_color = random.sample(PALETTE, 3)
    side_count = random.choices([2, 3, 4], weights=[2, 3, 3])[0]
    chosen_sides = random.sample(list(SIDES), side_count)
    chosen_sides.sort(key=SIDES.index)

    marker_counts = {side: 1 for side in chosen_sides}
    multi_side = random.choice(chosen_sides)
    marker_counts[multi_side] = random.randint(2, 4)
    for side in chosen_sides:
        if side == multi_side:
            continue
        marker_counts[side] = random.choices([1, 2, 3, 4], weights=[4, 3, 2, 1])[0]

    top_req = 2 * marker_counts.get("top", 0) + 1
    bottom_req = 2 * marker_counts.get("bottom", 0) + 1
    left_req = 2 * marker_counts.get("left", 0) + 1
    right_req = 2 * marker_counts.get("right", 0) + 1

    max_h = 30 - top_req - bottom_req
    max_w = 30 - left_req - right_req
    if max_h < 7 or max_w < 7:
        return None

    frame_h = random.randint(7, min(13, max_h))
    frame_w = random.randint(7, min(13, max_w))
    total_h = frame_h + top_req + bottom_req + random.randint(0, min(4, 30 - frame_h - top_req - bottom_req))
    total_w = frame_w + left_req + right_req + random.randint(0, min(4, 30 - frame_w - left_req - right_req))

    top_margin = top_req + random.randint(0, total_h - frame_h - top_req - bottom_req)
    left_margin = left_req + random.randint(0, total_w - frame_w - left_req - right_req)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    rect = (r0, c0, r1, c1)

    gap_positions = {}
    top_cols = list(range(c0 + 2, c1 - 1))
    side_rows = list(range(r0 + 2, r1 - 1))
    if "top" in chosen_sides:
        gap_positions["top"] = (r0, random.choice(top_cols))
    if "bottom" in chosen_sides:
        cols = [c for c in top_cols if c != gap_positions.get("top", (None, None))[1]]
        if not cols:
            return None
        gap_positions["bottom"] = (r1, random.choice(cols))
    if "left" in chosen_sides:
        gap_positions["left"] = (random.choice(side_rows), c0)
    if "right" in chosen_sides:
        rows = [r for r in side_rows if r != gap_positions.get("left", (None, None))[0]]
        if not rows:
            return None
        gap_positions["right"] = (random.choice(rows), c1)

    gaps = tuple(gap_positions[side] for side in chosen_sides)
    inside_targets = [_inside_target(rect, gap) for gap in gaps]
    if len(set(inside_targets)) != len(inside_targets):
        return None

    height, width = total_h, total_w
    markers = {}
    for side in chosen_sides:
        gap = gap_positions[side]
        if side == "top":
            limit = r0
        elif side == "bottom":
            limit = height - 1 - r1
        elif side == "left":
            limit = c0
        else:
            limit = width - 1 - c1
        dists = _sample_spaced_distances(limit, marker_counts[side])
        if dists is None:
            return None
        markers[gap] = tuple(_marker_coord_for_gap(gap, side, dist) for dist in dists)

    spec = {
        "bg": bg,
        "wall_color": wall_color,
        "marker_color": marker_color,
        "height": height,
        "width": width,
        "rect": rect,
        "gaps": gaps,
        "markers": markers,
        "side_order": tuple(chosen_sides),
    }
    return spec


def _validate_generated_instance(grid: np.ndarray, spec: dict) -> bool:
    try:
        parsed = _parse_puzzle(grid)
    except AssertionError:
        return False

    if parsed["rect"] != spec["rect"]:
        return False
    if set(parsed["gaps"]) != set(spec["gaps"]):
        return False
    if parsed["wall_color"] != spec["wall_color"] or parsed["marker_color"] != spec["marker_color"]:
        return False
    if not any(len(spec["markers"][gap]) >= 2 for gap in spec["gaps"]):
        return False

    expected_markers = {gap: tuple(sorted(spec["markers"][gap])) for gap in spec["gaps"]}
    parsed_markers = {
        gap: tuple(sorted(marker for _, marker in parsed["groups"][gap])) for gap in parsed["gaps"]
    }
    if expected_markers != parsed_markers:
        return False

    oracle = _render_output_from_spec(spec)
    actual = main(grid)
    if not np.array_equal(actual, oracle):
        return False

    out_marker_coords = np.argwhere(actual == spec["marker_color"])
    if out_marker_coords.shape[0] != sum(len(spec["markers"][gap]) for gap in spec["gaps"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        spec = _generate_spec()
        if spec is None:
            continue
        grid = _render_input(spec)
        if _validate_generated_instance(grid, spec):
            return grid
    raise RuntimeError("failed to generate a valid one-pass sealed-door instance")
