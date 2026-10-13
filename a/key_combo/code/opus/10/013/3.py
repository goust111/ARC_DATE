import random
from collections import Counter

import numpy as np


BG = 0
MARKER_COLORS = (2, 4)
FRAME_COLORS = (1, 3, 5, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
GEN_ATTEMPTS = 800


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _ring_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _is_frame(coords: np.ndarray) -> bool:
    box = _bbox(coords)
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expected = _ring_cells(box)
    actual = {tuple(map(int, xy)) for xy in coords}
    return actual == expected


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _component_coords(grid == color):
            if _is_frame(coords):
                box = _bbox(coords)
                frames.append(
                    {
                        "color": int(color),
                        "coords": coords,
                        "bbox": box,
                        "area": _bbox_area(box),
                    }
                )
    frames.sort(key=lambda item: (item["area"], item["bbox"]))
    return frames


def _singleton_pixels(grid: np.ndarray, frame_mask: np.ndarray) -> list[dict]:
    singles = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _component_coords(grid == color):
            if len(coords) != 1:
                continue
            r, c = map(int, coords[0])
            if frame_mask[r, c]:
                continue
            singles.append({"coord": (r, c), "color": int(color)})
    return singles


def _marker_matches_frame(
    grid: np.ndarray, frame: dict, marker: dict
) -> tuple[str, int] | None:
    r0, c0, r1, c1 = frame["bbox"]
    mr, mc = marker["coord"]
    if c0 <= mc <= c1 and mr < r0 and np.all(grid[mr + 1 : r0, mc] == BG):
        return "up", r0 - mr
    if c0 <= mc <= c1 and mr > r1 and np.all(grid[r1 + 1 : mr, mc] == BG):
        return "down", mr - r1
    if r0 <= mr <= r1 and mc < c0 and np.all(grid[mr, mc + 1 : c0] == BG):
        return "left", c0 - mc
    if r0 <= mr <= r1 and mc > c1 and np.all(grid[mr, c1 + 1 : mc] == BG):
        return "right", mc - c1
    return None


def _match_markers(grid: np.ndarray, frames: list[dict]) -> list[dict]:
    frame_mask = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        for r, c in frame["coords"]:
            frame_mask[int(r), int(c)] = True
    singles = _singleton_pixels(grid, frame_mask)

    matched = []
    used = set()
    for frame in frames:
        candidates = []
        for idx, marker in enumerate(singles):
            if idx in used:
                continue
            rel = _marker_matches_frame(grid, frame, marker)
            if rel is None:
                continue
            direction, dist = rel
            if not (1 <= dist <= 3):
                continue
            candidates.append((idx, direction, dist, marker))
        assert len(candidates) == 1, "expected exactly one aligned singleton marker per frame"
        idx, direction, dist, marker = candidates[0]
        used.add(idx)
        matched.append(
            {
                **frame,
                "marker": marker["coord"],
                "marker_color": int(marker["color"]),
                "direction": direction,
                "marker_dist": dist,
            }
        )
    assert len(used) == len(frames), "expected every marker to match exactly one frame"
    return matched


def _stretch_box(box: tuple[int, int, int, int], direction: str, amount: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    if direction == "up":
        return r0 - amount, c0, r1, c1
    if direction == "down":
        return r0, c0, r1 + amount, c1
    if direction == "left":
        return r0, c0 - amount, r1, c1
    if direction == "right":
        return r0, c0, r1, c1 + amount
    raise ValueError(direction)


def _draw_frame(out: np.ndarray, box: tuple[int, int, int, int], border_color: int, fill_color: int) -> None:
    r0, c0, r1, c1 = box
    out[r0 : r1 + 1, c0 : c1 + 1] = fill_color
    out[r0, c0 : c1 + 1] = border_color
    out[r1, c0 : c1 + 1] = border_color
    out[r0 : r1 + 1, c0] = border_color
    out[r0 : r1 + 1, c1] = border_color


def _solve_objects(grid: np.ndarray) -> list[dict]:
    frames = _extract_frames(grid)
    assert 2 <= len(frames) <= 3, "expected 2-3 hollow frames"
    objects = _match_markers(grid, frames)
    used_dirs = {obj["direction"] for obj in objects}
    assert len(used_dirs) == len(objects), "expected distinct marker directions"
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        colors = {int(c) for c in np.unique(inner) if int(c) != BG}
        assert 1 <= len(colors) <= 4
        obj["stretch"] = len(colors)
        obj["final_bbox"] = _stretch_box(obj["bbox"], obj["direction"], obj["stretch"])
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch each hollow rectangle by the number of distinct interior colors and refill the new interior with the marker color.

    Input structure:
    - The grid has a black background plus 2-3 non-overlapping hollow rectangular frames with 1-pixel monochrome borders.
    - Each frame contains a few scattered interior pixels whose distinct color count encodes a stretch amount.
    - Exactly one singleton marker pixel lies outside each frame, aligned above, below, left, or right of that frame.

    Rule:
    - For each frame, count the number of distinct non-black colors inside the frame interior.
    - Use the aligned singleton marker to determine which side of the frame moves outward by that count.
    - Redraw the stretched frame in the original border color, repaint the entire new interior with the marker color, and erase the standalone marker pixel unless it lands on the new border.

    Key insight:
    - The scattered interior pixels encode distance through color diversity, while the external marker encodes direction through its relative side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert int(np.count_nonzero(g == BG)) > 0

    objects = _solve_objects(g)
    out = g.copy()
    for obj in sorted(objects, key=lambda item: (item["area"], item["bbox"])):
        final_box = obj["final_bbox"]
        _draw_frame(out, final_box, obj["color"], obj["marker_color"])

        mr, mc = obj["marker"]
        r0, c0, r1, c1 = final_box
        on_border = r0 <= mr <= r1 and c0 <= mc <= c1 and (mr in (r0, r1) or mc in (c0, c1))
        out[mr, mc] = obj["color"] if on_border else BG
    return out


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _corridor_cells(box: tuple[int, int, int, int], direction: str, marker: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    mr, mc = marker
    pts = {(mr, mc)}
    if direction == "up":
        for r in range(mr + 1, r0):
            pts.add((r, mc))
    elif direction == "down":
        for r in range(r1 + 1, mr):
            pts.add((r, mc))
    elif direction == "left":
        for c in range(mc + 1, c0):
            pts.add((mr, c))
    else:
        for c in range(c1 + 1, mc):
            pts.add((mr, c))
    return pts


def _choose_marker(box: tuple[int, int, int, int], direction: str, dist: int, rng: random.Random) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if direction == "up":
        return r0 - dist, rng.randint(c0 + 1, c1 - 1)
    if direction == "down":
        return r1 + dist, rng.randint(c0 + 1, c1 - 1)
    if direction == "left":
        return rng.randint(r0 + 1, r1 - 1), c0 - dist
    return rng.randint(r0 + 1, r1 - 1), c1 + dist


def _candidate_box_bounds(
    h: int, w: int, direction: str, stretch: int, marker_dist: int, shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    rows, cols = shape
    if direction == "up":
        return max(stretch, marker_dist), 0, rows - h, cols - w
    if direction == "down":
        return 0, 0, rows - h - max(stretch, marker_dist), cols - w
    if direction == "left":
        return 0, max(stretch, marker_dist), rows - h, cols - w
    return 0, 0, rows - h, cols - w - max(stretch, marker_dist)


def _valid_candidate(candidate: dict, placed: list[dict]) -> bool:
    box = candidate["bbox"]
    marker = candidate["marker"]
    corridor = candidate["corridor"]
    for other in placed:
        if _boxes_intersect(box, other["bbox"], pad=1):
            return False
        if marker in _ring_cells(other["bbox"]):
            return False
        if other["marker"] in _ring_cells(box):
            return False
        if marker == other["marker"]:
            return False
        if any(cell in corridor for cell in other["corridor"]):
            return False
        or0, oc0, or1, oc1 = other["bbox"]
        if any(or0 <= r <= or1 and oc0 <= c <= oc1 for r, c in corridor):
            return False
        r0, c0, r1, c1 = box
        if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in other["corridor"]):
            return False
    return True


def _scatter_colors(border_color: int, k: int, rng: random.Random) -> list[int]:
    palette = [c for c in FRAME_COLORS if c != border_color]
    colors = rng.sample(palette, k)
    total = rng.randint(max(2, k), 6)
    scatter = list(colors)
    while len(scatter) < total:
        scatter.append(rng.choice(colors))
    rng.shuffle(scatter)
    return scatter


def _place_scatter(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    border_color: int,
    stretch: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = box
    interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    scatter = _scatter_colors(border_color, stretch, rng)
    positions = rng.sample(interior, len(scatter))
    for (r, c), color in zip(positions, scatter):
        grid[r, c] = color


def _final_box(spec: dict) -> tuple[int, int, int, int]:
    return _stretch_box(spec["bbox"], spec["direction"], spec["stretch"])


def _has_required_overlap(specs: list[dict], should_overlap: bool) -> bool:
    overlaps = []
    for i in range(len(specs)):
        for j in range(i + 1, len(specs)):
            area = _intersection_area(_final_box(specs[i]), _final_box(specs[j]))
            if area > 0:
                overlaps.append(area)
    if should_overlap:
        return bool(overlaps) and max(overlaps) <= 12
    return not overlaps


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(18, 24)
    cols = rng.randint(18, 24)
    n_objs = 3 if rng.random() < 0.65 else 2
    directions = rng.sample(DIRS, n_objs)
    stretches = rng.sample([1, 2, 3, 4], n_objs)
    overlap_branch = rng.random() < 0.4

    specs: list[dict] = []
    border_colors = rng.sample(FRAME_COLORS, n_objs)
    marker_colors = [rng.choice(MARKER_COLORS) for _ in range(n_objs)]

    for idx in range(n_objs):
        direction = directions[idx]
        stretch = stretches[idx]
        border_color = border_colors[idx]
        marker_color = marker_colors[idx]
        placed = False
        for _ in range(300):
            h = rng.randint(5, 8)
            w = rng.randint(5, 8)
            marker_dist = rng.randint(1, 3)
            rmin, cmin, rmax, cmax = _candidate_box_bounds(h, w, direction, stretch, marker_dist, (rows, cols))
            if rmin > rmax or cmin > cmax:
                continue
            r0 = rng.randint(rmin, rmax)
            c0 = rng.randint(cmin, cmax)
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            marker = _choose_marker(box, direction, marker_dist, rng)
            candidate = {
                "bbox": box,
                "direction": direction,
                "stretch": stretch,
                "color": border_color,
                "marker_color": marker_color,
                "marker": marker,
                "corridor": _corridor_cells(box, direction, marker),
            }
            if _valid_candidate(candidate, specs):
                specs.append(candidate)
                placed = True
                break
        if not placed:
            return None

    if not _has_required_overlap(specs, overlap_branch):
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for spec in specs:
        _draw_frame(grid, spec["bbox"], spec["color"], BG)
        _place_scatter(grid, spec["bbox"], spec["color"], spec["stretch"], rng)
        mr, mc = spec["marker"]
        if grid[mr, mc] != BG:
            return None
        grid[mr, mc] = spec["marker_color"]

    try:
        solved = main(grid)
        objects = _solve_objects(grid)
    except AssertionError:
        return None

    if len(objects) != n_objs:
        return None
    if np.array_equal(grid, solved):
        return None
    if Counter(obj["stretch"] for obj in objects) != Counter(stretches):
        return None
    if {obj["direction"] for obj in objects} != set(directions):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
