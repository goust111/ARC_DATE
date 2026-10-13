from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
COLORS = list(range(10))

# Generator configuration.
PAIR_COUNT_RANGE = (1, 4)
SCENE_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 160
GENERATION_ATTEMPTS = 320
GRID_BASE = 13
GRID_EXTRA = 5


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _scan_colors(grid: np.ndarray, excluded: set[int] | None = None) -> list[int]:
    seen = set() if excluded is None else set(map(int, excluded))
    colors: list[int] = []
    for raw in grid.ravel().tolist():
        color = int(raw)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _component_sort_key(comp: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue

            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rows = arr[:, 0]
            cols = arr[:, 1]
            bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox,
                    "area": int(arr.shape[0]),
                }
            )

    comps.sort(key=_component_sort_key)
    return comps


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _round_nearest(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _segment_from_component(comp: dict) -> dict | None:
    coords = comp["coords"]
    rows = sorted(set(int(v) for v in coords[:, 0].tolist()))
    cols = sorted(set(int(v) for v in coords[:, 1].tolist()))

    if len(rows) == 1:
        lo, hi = cols[0], cols[-1]
        if cols != list(range(lo, hi + 1)):
            return None
        return {"orient": "h", "fixed": rows[0], "lo": lo, "hi": hi, "length": hi - lo + 1}

    if len(cols) == 1:
        lo, hi = rows[0], rows[-1]
        if rows != list(range(lo, hi + 1)):
            return None
        return {"orient": "v", "fixed": cols[0], "lo": lo, "hi": hi, "length": hi - lo + 1}

    return None


def _component_side(comp: dict, segment: dict) -> int:
    r0, c0, r1, c1 = comp["bbox"]
    if segment["orient"] == "v":
        axis = int(segment["fixed"])
        if c1 < axis:
            return -1
        if c0 > axis:
            return 1
        return 0

    axis = int(segment["fixed"])
    if r1 < axis:
        return -1
    if r0 > axis:
        return 1
    return 0


def _gap_for_pair(comp_a: dict, comp_b: dict, segment: dict) -> tuple[int, int] | None:
    side_a = _component_side(comp_a, segment)
    side_b = _component_side(comp_b, segment)
    if side_a == 0 or side_b == 0 or side_a == side_b:
        return None

    center_a = _bbox_center(comp_a["bbox"])
    center_b = _bbox_center(comp_b["bbox"])

    if segment["orient"] == "v":
        x = float(segment["fixed"])
        x0 = center_a[1]
        x1 = center_b[1]
        if x0 == x1:
            return None
        t = (x - x0) / (x1 - x0)
        if not 0.0 <= t <= 1.0:
            return None
        y = center_a[0] + (center_b[0] - center_a[0]) * t
        if y < segment["lo"] - 0.5 or y > segment["hi"] + 0.5:
            return None
        gap_r = min(max(_round_nearest(y), int(segment["lo"])), int(segment["hi"]))
        return int(gap_r), int(segment["fixed"])

    y = float(segment["fixed"])
    y0 = center_a[0]
    y1 = center_b[0]
    if y0 == y1:
        return None
    t = (y - y0) / (y1 - y0)
    if not 0.0 <= t <= 1.0:
        return None
    x = center_a[1] + (center_b[1] - center_a[1]) * t
    if x < segment["lo"] - 0.5 or x > segment["hi"] + 0.5:
        return None
    gap_c = min(max(_round_nearest(x), int(segment["lo"])), int(segment["hi"]))
    return int(segment["fixed"]), int(gap_c)


def _same_side(segment: dict, side: int, r: int, c: int, goal: tuple[int, int]) -> bool:
    if (r, c) == goal:
        return True
    if segment["orient"] == "v":
        return c < int(segment["fixed"]) if side < 0 else c > int(segment["fixed"])
    return r < int(segment["fixed"]) if side < 0 else r > int(segment["fixed"])


def _shortest_side_path(
    grid: np.ndarray,
    comp: dict,
    color: int,
    bg: int,
    wall_color: int,
    segment: dict,
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    side = _component_side(comp, segment)
    assert side != 0, "component must lie on exactly one side of its separating wall"

    starts = sorted((int(r), int(c)) for r, c in comp["coords"].tolist())
    parent: dict[tuple[int, int], tuple[int, int] | None] = {cell: None for cell in starts}
    q = deque(starts)

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nxt in parent or not _same_side(segment, side, nr, nc, goal):
                continue

            value = int(grid[nr, nc])
            if nxt == goal:
                if value not in (bg, wall_color, color):
                    continue
            else:
                if value == wall_color or value not in (bg, color):
                    continue

            parent[nxt] = (r, c)
            q.append(nxt)

    assert goal in parent, "gap must be reachable from each object on its own side"

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _bridge_info_for_pair(
    grid: np.ndarray,
    color: int,
    bg: int,
    wall_color: int,
    segments: list[dict],
    comp_a: dict,
    comp_b: dict,
) -> dict:
    best: dict | None = None

    for idx, segment in enumerate(segments):
        gap = _gap_for_pair(comp_a, comp_b, segment)
        if gap is None:
            continue
        try:
            path_a = _shortest_side_path(grid, comp_a, color, bg, wall_color, segment, gap)
            path_b = _shortest_side_path(grid, comp_b, color, bg, wall_color, segment, gap)
        except AssertionError:
            continue

        score = (
            len(path_a) + len(path_b) - 1,
            len(path_a),
            len(path_b),
            gap[0],
            gap[1],
            idx,
        )
        if best is None or score < best["score"]:
            best = {
                "score": score,
                "segment": segment,
                "gap": gap,
                "path_a": path_a,
                "path_b": path_b,
            }

    assert best is not None, f"expected exactly one separating wall for color {color}"
    return best


def _find_wall_color(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    colors = _scan_colors(grid, excluded={bg})
    candidates: list[tuple[int, int, int, list[dict]]] = []

    for order, color in enumerate(colors):
        comps = _components_of_color(grid, color)
        if not comps:
            continue

        segments: list[dict] = []
        total_length = 0
        ok = True
        for comp in comps:
            segment = _segment_from_component(comp)
            if segment is None or int(segment["length"]) < 2:
                ok = False
                break
            segments.append(segment)
            total_length += int(segment["length"])
        if not ok:
            continue

        other_colors = [other for other in colors if other != color]
        if not other_colors:
            continue

        for other in other_colors:
            pair = _components_of_color(grid, other)
            if len(pair) != 2:
                ok = False
                break
            try:
                _bridge_info_for_pair(grid, other, bg, color, segments, pair[0], pair[1])
            except AssertionError:
                ok = False
                break

        if ok:
            candidates.append((len(other_colors), total_length, order, color, segments))

    assert candidates, "failed to identify the wall color"
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best = candidates[0]
    if len(candidates) > 1:
        runner_up = candidates[1]
        assert best[:3] != runner_up[:3], "wall color must be unique"
    return int(best[3]), best[4]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-colored object pairs separated by straight wall segments, open one-pixel
    gaps on those walls near the bbox-center intersection, and connect each pair to the
    gap with a shortest orthogonal bridge in the pair's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    wall_color, wall_segments = _find_wall_color(g, bg)
    pair_colors = _scan_colors(g, excluded={bg, wall_color})
    assert pair_colors, "expected at least one pair of bridged objects"

    plans: list[tuple[int, dict]] = []
    for color in pair_colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, f"expected exactly two objects for color {color}"
        info = _bridge_info_for_pair(g, color, bg, wall_color, wall_segments, comps[0], comps[1])
        plans.append((color, info))

    out = g.copy()
    for color, info in plans:
        for path in (info["path_a"], info["path_b"]):
            for r, c in path:
                current = int(out[r, c])
                assert current in (bg, wall_color, color), "bridges from different pairs may not overlap"
                out[r, c] = int(color)

    return out


def _random_blob_mask() -> np.ndarray:
    for _ in range(60):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        area = random.randint(max(3, (h * w + 1) // 2), h * w)

        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if len(rows) < 2 or len(cols) < 2:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if random.choice((False, True)):
            mask = np.flipud(mask)
        if random.choice((False, True)):
            mask = np.fliplr(mask)
        return mask.copy()

    return np.array([[1, 1], [1, 0]], dtype=bool)


def _all_positions(mask: np.ndarray, patch_shape: tuple[int, int], row_range: range, col_range: range) -> list[tuple[int, int]]:
    ph, pw = patch_shape
    mh, mw = mask.shape
    positions = [
        (r0, c0)
        for r0 in row_range
        for c0 in col_range
        if 0 <= r0 and 0 <= c0 and r0 + mh <= ph and c0 + mw <= pw
    ]
    random.shuffle(positions)
    return positions


def _place_mask(patch: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    patch[top + rows, left + cols] = int(color)


def _scene_valid(patch: np.ndarray, bg: int, wall_color: int, pair_color: int) -> bool:
    try:
        found_wall, segments = _find_wall_color(patch, bg)
        if found_wall != wall_color:
            return False
        colors = _scan_colors(patch, excluded={bg, wall_color})
        if colors != [pair_color]:
            return False
        comps = _components_of_color(patch, pair_color)
        if len(comps) != 2:
            return False
        info = _bridge_info_for_pair(patch, pair_color, bg, wall_color, segments, comps[0], comps[1])
        segment = info["segment"]
        gap = info["gap"]
        if int(segment["length"]) >= 5:
            variable = gap[0] if segment["orient"] == "v" else gap[1]
            if variable in (int(segment["lo"]), int(segment["hi"])):
                return False
        if len(info["path_a"]) < 2 or len(info["path_b"]) < 2:
            return False
        out = main(patch)
        return not np.array_equal(out, patch)
    except AssertionError:
        return False


def _generate_scene(orientation: str, bg: int, wall_color: int, pair_color: int) -> np.ndarray:
    for _ in range(SCENE_ATTEMPTS):
        if orientation == "v":
            ph = random.randint(8, 12)
            pw = random.randint(11, 16)
            wall_axis = random.randint(4, pw - 5)
            wall_len = random.randint(4, ph - 2)
            wall_start = random.randint(1, ph - wall_len - 1)
        else:
            ph = random.randint(11, 16)
            pw = random.randint(8, 12)
            wall_axis = random.randint(4, ph - 5)
            wall_len = random.randint(4, pw - 2)
            wall_start = random.randint(1, pw - wall_len - 1)

        patch = np.full((ph, pw), int(bg), dtype=int)
        if orientation == "v":
            patch[wall_start : wall_start + wall_len, wall_axis] = int(wall_color)
        else:
            patch[wall_axis, wall_start : wall_start + wall_len] = int(wall_color)

        first = _random_blob_mask()
        second = _random_blob_mask()

        if orientation == "v":
            left_positions = _all_positions(
                first,
                patch.shape,
                range(1, ph - first.shape[0]),
                range(1, wall_axis - first.shape[1]),
            )
            right_positions = _all_positions(
                second,
                patch.shape,
                range(1, ph - second.shape[0]),
                range(wall_axis + 2, pw - second.shape[1]),
            )
        else:
            left_positions = _all_positions(
                first,
                patch.shape,
                range(1, wall_axis - first.shape[0]),
                range(1, pw - first.shape[1]),
            )
            right_positions = _all_positions(
                second,
                patch.shape,
                range(wall_axis + 2, ph - second.shape[0]),
                range(1, pw - second.shape[1]),
            )

        if not left_positions or not right_positions:
            continue

        for top_a, left_a in left_positions[:40]:
            candidate = patch.copy()
            _place_mask(candidate, first, top_a, left_a, pair_color)
            for top_b, left_b in right_positions[:40]:
                test = candidate.copy()
                _place_mask(test, second, top_b, left_b, pair_color)
                if _scene_valid(test, bg, wall_color, pair_color):
                    return test

    raise AssertionError("failed to generate a valid local bridge scene")


def _orientations_for_pairs(pair_count: int) -> list[str]:
    if pair_count == 1:
        return [random.choice(("v", "h"))]
    if random.choice((False, True)):
        out = ["v", "h"] + [random.choice(("v", "h")) for _ in range(pair_count - 2)]
        random.shuffle(out)
        return out
    return [random.choice(("v", "h")) for _ in range(pair_count)]


def _paste_scene(grid: np.ndarray, scene: np.ndarray, top: int, left: int, bg: int) -> None:
    rows, cols = np.where(scene != bg)
    grid[top + rows, left + cols] = scene[rows, cols]


def _place_scenes(shape: tuple[int, int], bg: int, scenes: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    grid = np.full((h, w), int(bg), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = list(range(len(scenes)))
    order.sort(key=lambda idx: scenes[idx].shape[0] * scenes[idx].shape[1], reverse=True)

    for idx in order:
        scene = scenes[idx]
        sh, sw = scene.shape
        spots = [
            (r0, c0)
            for r0 in range(0, h - sh + 1)
            for c0 in range(0, w - sw + 1)
            if not np.any(
                blocked[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)]
            )
        ]
        random.shuffle(spots)
        if not spots:
            return None

        top, left = spots[0]
        _paste_scene(grid, scene, top, left, bg)
        blocked[max(0, top - 1) : min(h, top + sh + 1), max(0, left - 1) : min(w, left + sw + 1)] = True

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair_count = random.randint(*PAIR_COUNT_RANGE)
        bg = random.choice(COLORS)
        remaining = [color for color in COLORS if color != bg]
        wall_color = random.choice(remaining)
        pair_colors = random.sample([color for color in remaining if color != wall_color], pair_count)

        scenes = [
            _generate_scene(orientation, bg, wall_color, color)
            for orientation, color in zip(_orientations_for_pairs(pair_count), pair_colors, strict=False)
        ]

        min_h = max(scene.shape[0] for scene in scenes) + 2
        min_w = max(scene.shape[1] for scene in scenes) + 2
        h = random.randint(max(min_h, GRID_BASE + pair_count * 3), max(min_h, GRID_BASE + pair_count * 3 + GRID_EXTRA))
        w = random.randint(max(min_w, GRID_BASE + pair_count * 4), max(min_w, GRID_BASE + pair_count * 4 + GRID_EXTRA))

        grid = None
        for _ in range(PLACEMENT_ATTEMPTS):
            grid = _place_scenes((h, w), bg, scenes)
            if grid is not None:
                break
        if grid is None:
            continue

        try:
            if _bg_color(grid) != bg:
                continue
            found_wall, _segments = _find_wall_color(grid, bg)
            if found_wall != wall_color:
                continue
            out = main(grid)
        except AssertionError:
            continue

        if np.array_equal(out, grid):
            continue
        return grid

    raise AssertionError("generator exhausted its retry budget")
