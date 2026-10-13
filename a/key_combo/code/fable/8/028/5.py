from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, find_components


GREY = Color.GREY
BLACK = Color.BLACK
MARKER_COLORS = tuple(color for color in Color.ALL_COLORS if color not in (BLACK, GREY))
SIDES = ("top", "bottom", "left", "right")

GEN_GRID_H_RANGE = (12, 24)
GEN_GRID_W_RANGE = (12, 24)
GEN_RECT_COUNT_RANGE = (3, 6)
GEN_RECT_H_RANGE = (2, 5)
GEN_RECT_W_RANGE = (2, 6)
GEN_MIN_GAP = 1
GEN_MAX_PLACEMENT_TRIES = 120
GEN_MAX_ATTEMPTS = 400


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    """Return all solid interior grey rectangles as ordered component records."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = find_components(
        g,
        bg=BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[GREY],
        return_format="dict",
    )
    assert comps, "expected at least one grey rectangle"

    rects = []
    for rect_id, comp in enumerate(sorted(comps, key=lambda obj: obj["bbox"])):
        r0, c0, r1, c1 = map(int, comp["bbox"])
        height = bbox_h(comp["bbox"])
        width = bbox_w(comp["bbox"])
        assert int(comp["area"]) == height * width, "grey objects must be solid rectangles"
        assert 0 < r0 <= r1 < h - 1 and 0 < c0 <= c1 < w - 1, "rectangles must stay off the border"
        rects.append(
            {
                "id": rect_id,
                "bbox": (r0, c0, r1, c1),
                "coords": comp["coords"],
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "height": height,
                "width": width,
            }
        )
    return rects


def _build_rect_index_grid(shape: tuple[int, int], rects: list[dict]) -> np.ndarray:
    rect_index = np.full(shape, -1, dtype=int)
    for rect in rects:
        rect_index[rect["coords"][:, 0], rect["coords"][:, 1]] = int(rect["id"])
    return rect_index


def _marker_side_and_pos(r: int, c: int, height: int, width: int) -> tuple[str, int]:
    if r == 0 and 0 < c < width - 1:
        return "top", int(c)
    if r == height - 1 and 0 < c < width - 1:
        return "bottom", int(c)
    if c == 0 and 0 < r < height - 1:
        return "left", int(r)
    if c == width - 1 and 0 < r < height - 1:
        return "right", int(r)
    raise AssertionError("colored border pixels may not occupy corners")


def _ray_to_coord(ray: tuple[str, int], height: int, width: int) -> tuple[int, int]:
    side, pos = ray
    if side == "top":
        return 0, int(pos)
    if side == "bottom":
        return height - 1, int(pos)
    if side == "left":
        return int(pos), 0
    if side == "right":
        return int(pos), width - 1
    raise AssertionError(side)


def _ray_direction(side: str) -> tuple[int, int]:
    if side == "top":
        return 1, 0
    if side == "bottom":
        return -1, 0
    if side == "left":
        return 0, 1
    if side == "right":
        return 0, -1
    raise AssertionError(side)


def _boundary_markers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers = []
    perimeter = (
        [(0, c) for c in range(w)]
        + [(h - 1, c) for c in range(w)]
        + [(r, 0) for r in range(1, h - 1)]
        + [(r, w - 1) for r in range(1, h - 1)]
    )
    for r, c in perimeter:
        color = int(g[r, c])
        if color == BLACK:
            continue
        assert color != GREY, "grey belongs only to rectangles"
        side, pos = _marker_side_and_pos(int(r), int(c), h, w)
        markers.append({"coord": (int(r), int(c)), "side": side, "pos": pos, "color": color})
    return markers


def _scan_first_hit(rect_index: np.ndarray, ray: tuple[str, int]) -> int | None:
    h, w = rect_index.shape
    r, c = _ray_to_coord(ray, h, w)
    dr, dc = _ray_direction(ray[0])
    r += dr
    c += dc
    while 0 < r < h - 1 and 0 < c < w - 1:
        rect_id = int(rect_index[r, c])
        if rect_id != -1:
            return rect_id
        r += dr
        c += dc
    return None


def _eligible_rectangles(rects: list[dict], side: str, pos: int) -> list[int]:
    if side in {"top", "bottom"}:
        hits = [rect["id"] for rect in rects if rect["c0"] <= pos <= rect["c1"]]
        if side == "top":
            return sorted(hits, key=lambda rect_id: rects[rect_id]["r0"])
        return sorted(hits, key=lambda rect_id: rects[rect_id]["r1"], reverse=True)

    hits = [rect["id"] for rect in rects if rect["r0"] <= pos <= rect["r1"]]
    if side == "left":
        return sorted(hits, key=lambda rect_id: rects[rect_id]["c0"])
    return sorted(hits, key=lambda rect_id: rects[rect_id]["c1"], reverse=True)


def _visibility_catalog(rects: list[dict], height: int, width: int) -> dict:
    candidates_by_rect = [[] for _ in rects]
    empty_rays = []
    blocked_lines = 0

    for side in SIDES:
        positions = range(1, width - 1) if side in {"top", "bottom"} else range(1, height - 1)
        for pos in positions:
            ray = (side, int(pos))
            eligible = _eligible_rectangles(rects, side, int(pos))
            if not eligible:
                empty_rays.append(ray)
                continue
            candidates_by_rect[eligible[0]].append(ray)
            if len(eligible) > 1:
                blocked_lines += 1

    return {
        "candidates_by_rect": candidates_by_rect,
        "empty_rays": empty_rays,
        "blocked_lines": blocked_lines,
    }


def _winner(votes: Counter[int]) -> int:
    top_count = max(votes.values())
    winners = [color for color, count in votes.items() if count == top_count]
    assert len(winners) == 1, "majority color must be unique"
    return int(winners[0])


def _solve_by_geometry(grid: np.ndarray) -> tuple[np.ndarray, list[Counter[int]], dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rects = _extract_rectangles(g)
    votes = [Counter() for _ in rects]

    for marker in _boundary_markers(g):
        ray = (marker["side"], marker["pos"])
        eligible = _eligible_rectangles(rects, marker["side"], marker["pos"])
        if eligible:
            votes[eligible[0]][int(marker["color"])] += 1

    out = g.copy()
    out[0, :] = BLACK
    out[-1, :] = BLACK
    out[:, 0] = BLACK
    out[:, -1] = BLACK
    for rect in rects:
        fill = GREY
        rect_votes = votes[rect["id"]]
        if rect_votes:
            fill = _winner(rect_votes)
        out[rect["coords"][:, 0], rect["coords"][:, 1]] = fill

    return out, votes, _visibility_catalog(rects, h, w)


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each interior grey rectangle by the unique majority color among border rays that hit it first.

    Input structure:
    - A black background contains one or more non-overlapping grey solid rectangles fully inside the border.
    - Additional colored single pixels may appear on the outer border, but never in the corners.
    - Each border pixel looks straight inward along its row or column.

    Rule:
    - For every colored border pixel, trace its inward row or column and find the first grey rectangle on that sight line.
    - Add one vote of the border pixel's color to that first-hit rectangle; farther rectangles on the same line receive nothing because they are blocked.
    - Recolor each rectangle to its unique majority vote color, keep unvoted rectangles grey, and erase all border pixels to black.

    Key insight:
    - Votes are assigned by first visibility along straight sight lines, so blocking by nearer rectangles determines which rectangle each border pixel can affect.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rects = _extract_rectangles(g)
    rect_index = _build_rect_index_grid(g.shape, rects)
    votes = [Counter() for _ in rects]

    for marker in _boundary_markers(g):
        ray = (marker["side"], marker["pos"])
        rect_id = _scan_first_hit(rect_index, ray)
        if rect_id is not None:
            votes[rect_id][int(marker["color"])] += 1

    out = g.copy()
    out[0, :] = BLACK
    out[-1, :] = BLACK
    out[:, 0] = BLACK
    out[:, -1] = BLACK
    for rect in rects:
        fill = GREY
        rect_votes = votes[rect["id"]]
        if rect_votes:
            fill = _winner(rect_votes)
        out[rect["coords"][:, 0], rect["coords"][:, 1]] = fill
    return out


def _rect_ok(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], height: int, width: int) -> bool:
    r0, c0, rh, rw = rect
    r1 = r0 + rh - 1
    c1 = c0 + rw - 1
    if not (1 <= r0 <= r1 < height - 1 and 1 <= c0 <= c1 < width - 1):
        return False
    for er0, ec0, eh, ew in placed:
        er1 = er0 + eh - 1
        ec1 = ec0 + ew - 1
        separated = r1 + GEN_MIN_GAP < er0 or er1 + GEN_MIN_GAP < r0 or c1 + GEN_MIN_GAP < ec0 or ec1 + GEN_MIN_GAP < c0
        if not separated:
            return False
    return True


def _random_rect(height: int, width: int, placed: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int] | None:
    for _ in range(GEN_MAX_PLACEMENT_TRIES):
        rh = random.randint(*GEN_RECT_H_RANGE)
        rw = random.randint(*GEN_RECT_W_RANGE)
        if rh >= height - 2 or rw >= width - 2:
            continue
        r0 = random.randint(1, height - rh - 1)
        c0 = random.randint(1, width - rw - 1)
        rect = (r0, c0, rh, rw)
        if _rect_ok(rect, placed, height, width):
            return rect
    return None


def _aligned_rect(
    anchor: tuple[int, int, int, int],
    orientation: str,
    height: int,
    width: int,
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    ar0, ac0, ah, aw = anchor
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    for _ in range(GEN_MAX_PLACEMENT_TRIES):
        rh = random.randint(*GEN_RECT_H_RANGE)
        rw = random.randint(*GEN_RECT_W_RANGE)
        if rh >= height - 2 or rw >= width - 2:
            continue
        if orientation == "vertical":
            c_low = max(1, ac0 - rw + 1)
            c_high = min(width - rw - 1, ac1)
            if c_low > c_high:
                continue
            c0 = random.randint(c_low, c_high)
            row_ranges = []
            above_max = ar0 - rh - 1
            below_min = ar1 + 2
            if 1 <= above_max:
                row_ranges.append((1, above_max))
            if below_min <= height - rh - 1:
                row_ranges.append((below_min, height - rh - 1))
            if not row_ranges:
                continue
            lo, hi = random.choice(row_ranges)
            r0 = random.randint(lo, hi)
        elif orientation == "horizontal":
            r_low = max(1, ar0 - rh + 1)
            r_high = min(height - rh - 1, ar1)
            if r_low > r_high:
                continue
            r0 = random.randint(r_low, r_high)
            col_ranges = []
            left_max = ac0 - rw - 1
            right_min = ac1 + 2
            if 1 <= left_max:
                col_ranges.append((1, left_max))
            if right_min <= width - rw - 1:
                col_ranges.append((right_min, width - rw - 1))
            if not col_ranges:
                continue
            lo, hi = random.choice(col_ranges)
            c0 = random.randint(lo, hi)
        else:
            raise AssertionError(orientation)

        rect = (r0, c0, rh, rw)
        if _rect_ok(rect, placed, height, width):
            return rect
    return None


def _materialize_rectangles(rect_specs: list[tuple[int, int, int, int]]) -> list[dict]:
    rects = []
    for rect_id, (r0, c0, rh, rw) in enumerate(sorted(rect_specs)):
        r1 = r0 + rh - 1
        c1 = c0 + rw - 1
        rows, cols = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        coords = np.column_stack((rows.ravel(), cols.ravel()))
        rects.append(
            {
                "id": rect_id,
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
            }
        )
    return rects


def _build_layout() -> tuple[int, int, list[tuple[int, int, int, int]], dict] | None:
    height = random.randint(*GEN_GRID_H_RANGE)
    width = random.randint(*GEN_GRID_W_RANGE)
    target_count = random.randint(*GEN_RECT_COUNT_RANGE)

    placed = []
    first = _random_rect(height, width, placed)
    if first is None:
        return None
    placed.append(first)

    mode = random.choices(["vertical", "horizontal", "both"], weights=[2, 2, 4])[0]
    if mode in {"vertical", "both"} and len(placed) < target_count:
        rect = _aligned_rect(placed[0], "vertical", height, width, placed)
        if rect is not None:
            placed.append(rect)
    if mode in {"horizontal", "both"} and len(placed) < target_count:
        anchor = random.choice(placed)
        rect = _aligned_rect(anchor, "horizontal", height, width, placed)
        if rect is not None:
            placed.append(rect)

    while len(placed) < target_count:
        if random.random() < 0.35:
            orientation = random.choice(("vertical", "horizontal"))
            rect = _aligned_rect(random.choice(placed), orientation, height, width, placed)
        else:
            rect = _random_rect(height, width, placed)
        if rect is None:
            return None
        placed.append(rect)

    rects = _materialize_rectangles(placed)
    catalog = _visibility_catalog(rects, height, width)
    hittable = [len(rays) for rays in catalog["candidates_by_rect"]]
    if catalog["blocked_lines"] < 1:
        return None
    if max(hittable, default=0) < 3:
        return None
    if sum(count > 0 for count in hittable) < 2:
        return None
    if len(catalog["empty_rays"]) < 2:
        return None
    return height, width, placed, catalog


def _choose_distinct_colors(count: int) -> list[int]:
    if count <= len(MARKER_COLORS):
        return random.sample(MARKER_COLORS, count)
    return [random.choice(MARKER_COLORS) for _ in range(count)]


def _vote_colors(total_votes: int, majority_color: int, force_mixed: bool) -> list[int]:
    if force_mixed and total_votes >= 3:
        majority_count = random.randint(total_votes // 2 + 1, total_votes - 1)
        others = [random.choice([color for color in MARKER_COLORS if color != majority_color]) for _ in range(total_votes - majority_count)]
        colors = [majority_color] * majority_count + others
        random.shuffle(colors)
        return colors
    return [majority_color] * total_votes


def _plan_markers(rects: list[dict], catalog: dict) -> dict[tuple[str, int], int] | None:
    candidates_by_rect = catalog["candidates_by_rect"]
    hittable = [rect_id for rect_id, rays in enumerate(candidates_by_rect) if rays]
    mixed_candidates = [rect_id for rect_id in hittable if len(candidates_by_rect[rect_id]) >= 3]
    if not mixed_candidates:
        return None

    mixed_rect = random.choice(mixed_candidates)
    active_count = random.randint(1, len(hittable))
    if len(rects) >= 4 and active_count == len(hittable) and random.random() < 0.7:
        active_count -= 1
    active = {mixed_rect}
    others = [rect_id for rect_id in hittable if rect_id != mixed_rect]
    random.shuffle(others)
    active.update(others[: max(0, active_count - 1)])

    assignments: dict[tuple[str, int], int] = {}
    majority_palette = _choose_distinct_colors(len(active))
    for color_index, rect_id in enumerate(sorted(active)):
        rays = candidates_by_rect[rect_id][:]
        random.shuffle(rays)
        max_votes = min(6, len(rays))
        if rect_id == mixed_rect:
            total_votes = random.randint(3, max_votes)
            colors = _vote_colors(total_votes, majority_palette[color_index], force_mixed=True)
        else:
            if max_votes >= 2 and random.random() < 0.65:
                total_votes = random.randint(2, max_votes)
            else:
                total_votes = random.randint(1, max_votes)
            colors = _vote_colors(
                total_votes,
                majority_palette[color_index],
                force_mixed=total_votes >= 3 and random.random() < 0.35,
            )
        for ray, color in zip(rays[:total_votes], colors):
            assignments[ray] = int(color)

    empty_rays = catalog["empty_rays"][:]
    random.shuffle(empty_rays)
    if empty_rays and random.random() < 0.8:
        distractor_count = random.randint(1, min(4, len(empty_rays)))
        for ray in empty_rays[:distractor_count]:
            assignments[ray] = int(random.choice(MARKER_COLORS))

    if len(set(assignments.values())) < 2:
        extra_choices = [ray for ray in empty_rays if ray not in assignments]
        if extra_choices:
            assignments[extra_choices[0]] = int(random.choice([color for color in MARKER_COLORS if color != next(iter(assignments.values()))]))
    return assignments


def _render_input(height: int, width: int, rect_specs: list[tuple[int, int, int, int]], assignments: dict[tuple[str, int], int]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for r0, c0, rh, rw in rect_specs:
        grid[r0 : r0 + rh, c0 : c0 + rw] = GREY
    for ray, color in assignments.items():
        r, c = _ray_to_coord(ray, height, width)
        grid[r, c] = int(color)
    return grid


def _validate_example(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    rects = _extract_rectangles(g)
    rect_index = _build_rect_index_grid(g.shape, rects)
    markers = _boundary_markers(g)
    geom_votes = [Counter() for _ in rects]
    scan_votes = [Counter() for _ in rects]

    for marker in markers:
        ray = (marker["side"], marker["pos"])
        eligible = _eligible_rectangles(rects, marker["side"], marker["pos"])
        geom_hit = eligible[0] if eligible else None
        scan_hit = _scan_first_hit(rect_index, ray)
        assert geom_hit == scan_hit, "scan and geometry must agree on first-hit blocking"
        if geom_hit is not None:
            geom_votes[geom_hit][int(marker["color"])] += 1
            scan_votes[scan_hit][int(marker["color"])] += 1

    out = main(g)
    ref, ref_votes, catalog = _solve_by_geometry(g)
    assert np.array_equal(out, ref), "main() must match the independent geometric reference"
    assert all(v1 == v2 == v3 for v1, v2, v3 in zip(geom_votes, scan_votes, ref_votes))
    assert np.all(out[0, :] == BLACK)
    assert np.all(out[-1, :] == BLACK)
    assert np.all(out[:, 0] == BLACK)
    assert np.all(out[:, -1] == BLACK)

    mixed_vote_rects = 0
    grey_rects = 0
    for rect in rects:
        rect_votes = geom_votes[rect["id"]]
        expected = GREY
        if rect_votes:
            expected = _winner(rect_votes)
            if len(rect_votes) > 1:
                mixed_vote_rects += 1
        else:
            grey_rects += 1
        rect_pixels = out[rect["coords"][:, 0], rect["coords"][:, 1]]
        assert np.all(rect_pixels == expected)

    return {
        "marker_count": len(markers),
        "hit_count": sum(sum(counter.values()) for counter in geom_votes),
        "miss_count": len(markers) - sum(sum(counter.values()) for counter in geom_votes),
        "grey_rects": grey_rects,
        "mixed_vote_rects": mixed_vote_rects,
        "blocked_lines": int(catalog["blocked_lines"]),
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        layout = _build_layout()
        if layout is None:
            continue
        height, width, rect_specs, catalog = layout
        rects = _materialize_rectangles(rect_specs)
        assignments = _plan_markers(rects, catalog)
        if assignments is None:
            continue
        grid = _render_input(height, width, rect_specs, assignments)
        try:
            stats = _validate_example(grid)
        except Exception:
            continue
        if stats["marker_count"] < 4:
            continue
        if stats["blocked_lines"] < 1:
            continue
        if stats["miss_count"] < 1:
            continue
        if stats["mixed_vote_rects"] < 1:
            continue
        return grid
    raise RuntimeError("failed to generate a valid line-of-sight voting example")
