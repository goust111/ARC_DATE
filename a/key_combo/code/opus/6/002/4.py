import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

SEED_COLORS = (RED, GREEN, BLUE, YELLOW)
KEY_ORDER = (RED, GREEN, BLUE, YELLOW)
DIRS = ("up", "down", "left", "right")


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_frame_mask(h: int, w: int) -> np.ndarray:
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return frame


def _is_legend_component(grid: np.ndarray, comp: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 7:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(sub == GREY, _rect_frame_mask(h, w))


def _opening_direction(submask: np.ndarray) -> str:
    top = np.all(submask[0, 1:-1])
    bottom = np.all(submask[-1, 1:-1])
    left = np.all(submask[1:-1, 0])
    right = np.all(submask[1:-1, -1])
    states = {"up": top, "down": bottom, "left": left, "right": right}
    missing = [name for name, present in states.items() if not present]
    assert len(missing) == 1, f"expected exactly one opening, got {states}"
    return missing[0]


def _parse_wall(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(coords)
    submask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local = coords - np.array([[r0, c0]])
    submask[local[:, 0], local[:, 1]] = True
    direction = _opening_direction(submask)

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_positions = []
    for rr, cc in np.argwhere(np.isin(inner, SEED_COLORS)):
        seed_positions.append((int(rr + r0 + 1), int(cc + c0 + 1)))
    assert len(seed_positions) == 1, "each wall must contain exactly one seed"
    seed_r, seed_c = seed_positions[0]
    color = int(grid[seed_r, seed_c])
    return {
        "bbox": (r0, c0, r1, c1),
        "direction": direction,
        "seed": (seed_r, seed_c),
        "color": color,
    }


def _parse_legend(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict:
    r0, c0, r1, c1 = bbox
    keys = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color in SEED_COLORS:
                assert color not in keys, "legend color key must be unique"
                keys[color] = (r, c)
    assert 3 <= len(keys) <= 4
    return {"bbox": bbox, "keys": keys}


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], dict]:
    grey_comps = components_mask(grid == GREY)
    legend = None
    walls = []
    for comp in grey_comps:
        if _is_legend_component(grid, comp):
            assert legend is None, "expected one legend box"
            legend = _parse_legend(grid, bbox_coords(comp))
        else:
            walls.append(_parse_wall(grid, comp))
    assert legend is not None
    assert 3 <= len(walls) <= 4
    wall_colors = {wall["color"] for wall in walls}
    assert wall_colors == set(legend["keys"]), "legend keys must match wall seed colors"
    return walls, legend


def _ray_lanes(bbox: tuple[int, int, int, int], direction: str, shape: tuple[int, int]) -> list[list[tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    lanes = []
    if direction == "up":
        for c in range(c0 + 1, c1):
            lanes.append([(r, c) for r in range(r0 - 1, -1, -1)])
    elif direction == "down":
        for c in range(c0 + 1, c1):
            lanes.append([(r, c) for r in range(r1 + 1, h)])
    elif direction == "left":
        for r in range(r0 + 1, r1):
            lanes.append([(r, c) for c in range(c0 - 1, -1, -1)])
    else:
        for r in range(r0 + 1, r1):
            lanes.append([(r, c) for c in range(c1 + 1, w)])
    return lanes


def _filled_cells(grid: np.ndarray, wall: dict) -> set[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = wall["bbox"]
    seed_r, seed_c = wall["seed"]
    color = wall["color"]
    direction = wall["direction"]

    allowed = np.zeros_like(grid, dtype=bool)
    allowed[r0 + 1 : r1, c0 + 1 : c1] = True
    q = deque([(seed_r, seed_c)])
    seen = {(seed_r, seed_c)}
    while q:
        rr, cc = q.popleft()
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and (nr, nc) not in seen:
                if int(grid[nr, nc]) in (BG, color):
                    seen.add((nr, nc))
                    q.append((nr, nc))

    for lane in _ray_lanes(wall["bbox"], direction, grid.shape):
        for rr, cc in lane:
            if int(grid[rr, cc]) != BG:
                break
            seen.add((rr, cc))
    return seen


def _paint_output(grid: np.ndarray, walls: list[dict], legend: dict) -> np.ndarray:
    out = np.array(grid, copy=True)
    counts: dict[int, int] = {}
    for wall in walls:
        filled = _filled_cells(grid, wall)
        counts[wall["color"]] = len(filled)
        for r, c in filled:
            out[r, c] = wall["color"]

    r0, c0, r1, c1 = legend["bbox"]
    for color, (r, c) in legend["keys"].items():
        count = counts[color]
        for i in range(count):
            dot_c = c + 2 + 2 * i
            assert dot_c < c1, "legend box too narrow for dot count"
            out[r, dot_c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each partial grey enclosure, radiate through its opening, then encode the shadow-reduced fill counts in the legend.

    Input structure:
    - The background is black. Several grey one-cell-thick U/C walls sit on the left, each with exactly one colored seed inside.
    - Some openings have orange obstacle pixels somewhere along the outward ray lanes between the opening and the grid edge.
    - On the right there is one grey rectangular legend frame whose interior contains one color-key pixel for each seed color.

    Rule:
    - For every grey wall, flood-fill its interior from the seed color while staying inside the wall bbox interior.
    - From the opening side, extend straight outward along each opening lane, filling black cells until the first non-black cell or the boundary; orange obstacles therefore block their own cell and every farther cell on that lane.
    - Count the total filled cells for each seed color, including the original seed cell, then write that many same-color dots in the matching legend row starting two cells to the right of the key and spaced every other cell.

    Key insight:
    - The legend does not report the unobstructed fan size: each orange obstacle shortens exactly one outward lane, so the final count must be computed from the actual filled footprint after ray blocking.
    """
    walls, legend = _parse_scene(np.asarray(grid))
    return _paint_output(np.asarray(grid), walls, legend)


def _wall_cells(r0: int, c0: int, h: int, w: int, direction: str) -> set[tuple[int, int]]:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    cells: set[tuple[int, int]] = set()
    if direction != "up":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
    if direction != "down":
        for c in range(c0, c1 + 1):
            cells.add((r1, c))
    if direction != "left":
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
    if direction != "right":
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    return cells


def _visible_tuples(span: int, full_len: int, target_sum: int) -> list[tuple[int, ...]]:
    out = []
    if full_len < 0:
        return out

    def rec(idx: int, rem: int, cur: list[int]) -> None:
        if idx == span:
            if rem == 0:
                tpl = tuple(cur)
                if any(v < full_len for v in tpl):
                    if not any(v == full_len for v in tpl):
                        return
                out.append(tpl)
            return
        for v in range(full_len + 1):
            if v > rem:
                break
            cur.append(v)
            rec(idx + 1, rem - v, cur)
            cur.pop()

    rec(0, target_sum, [])
    return out


def _candidate_specs(direction: str, target_count: int, height: int, play_w: int) -> list[dict]:
    specs = []
    for h in range(3, 6):
        for w in range(3, 6):
            area = (h - 2) * (w - 2)
            if area <= 0 or area > target_count or area > 6:
                continue

            if direction == "up":
                lane_span = w - 2
                col_starts = [c for c in (1, 2, 3) if c + w <= play_w]
                for full_len in (1, 2, 3):
                    r0 = full_len
                    if r0 + h > height:
                        continue
                    for c0 in col_starts:
                        for vis in _visible_tuples(lane_span, full_len, target_count - area):
                            specs.append({"r0": r0, "c0": c0, "h": h, "w": w, "direction": direction, "visible": vis})
            elif direction == "down":
                lane_span = w - 2
                col_starts = [c for c in (max(1, play_w - w - 2), max(1, play_w - w - 3)) if c + w <= play_w]
                for full_len in (1, 2, 3):
                    r1 = height - 1 - full_len
                    r0 = r1 - h + 1
                    if r0 < 0:
                        continue
                    for c0 in sorted(set(col_starts)):
                        for vis in _visible_tuples(lane_span, full_len, target_count - area):
                            specs.append({"r0": r0, "c0": c0, "h": h, "w": w, "direction": direction, "visible": vis})
            elif direction == "left":
                lane_span = h - 2
                row_starts = [height // 2 - 4, height // 2 - 3, height // 2 - 2]
                for full_len in (1, 2, 3):
                    c0 = full_len
                    if c0 + w > play_w:
                        continue
                    for r0 in row_starts:
                        if 0 <= r0 and r0 + h <= height:
                            for vis in _visible_tuples(lane_span, full_len, target_count - area):
                                specs.append({"r0": r0, "c0": c0, "h": h, "w": w, "direction": direction, "visible": vis})
            else:
                lane_span = h - 2
                row_starts = [height // 2 - 1, height // 2]
                for full_len in (2, 3, 4):
                    c1 = play_w + 1 - full_len
                    c0 = c1 - w + 1
                    if c0 < 0:
                        continue
                    for r0 in row_starts:
                        if 0 <= r0 and r0 + h <= height:
                            for vis in _visible_tuples(lane_span, full_len, target_count - area):
                                specs.append({"r0": r0, "c0": c0, "h": h, "w": w, "direction": direction, "visible": vis})
    return specs


def _obstacles_from_visible(spec: dict) -> set[tuple[int, int]]:
    r0 = spec["r0"]
    c0 = spec["c0"]
    h = spec["h"]
    w = spec["w"]
    direction = spec["direction"]
    visible = spec["visible"]
    obs = set()
    if direction == "up":
        for i, vis in enumerate(visible):
            full_len = r0
            if vis < full_len:
                obs.add((r0 - 1 - vis, c0 + 1 + i))
    elif direction == "down":
        r1 = r0 + h - 1
        for i, vis in enumerate(visible):
            full_len = spec["grid_h"] - 1 - r1
            if vis < full_len:
                obs.add((r1 + 1 + vis, c0 + 1 + i))
    elif direction == "left":
        for i, vis in enumerate(visible):
            full_len = c0
            if vis < full_len:
                obs.add((r0 + 1 + i, c0 - 1 - vis))
    else:
        c1 = c0 + w - 1
        for i, vis in enumerate(visible):
            full_len = spec["play_w"] + 1 - c1
            if vis < full_len:
                obs.add((r0 + 1 + i, c1 + 1 + vis))
    return obs


def _spec_count(spec: dict) -> int:
    return (spec["h"] - 2) * (spec["w"] - 2) + sum(spec["visible"])


def _legend_width(max_count: int) -> int:
    return 3 + 2 * max_count


def _touches_occupied(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for cell in cells:
        if cell in occupied:
            return True
        for nbr in neighbors4(*cell):
            if nbr in occupied:
                return True
    return False


def _build_input(specs: list[dict], legend_rows: list[int], legend_c0: int, legend_w: int, height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    legend_h = 2 * len(legend_rows) + 1
    legend_r0 = legend_rows[0] - 1
    legend_r1 = legend_r0 + legend_h - 1
    legend_c1 = legend_c0 + legend_w - 1
    frame = _rect_frame_mask(legend_h, legend_w)
    grid[legend_r0 : legend_r1 + 1, legend_c0 : legend_c1 + 1][frame] = GREY

    for row, color in zip(legend_rows, [color for color in KEY_ORDER if color in {spec["color"] for spec in specs}]):
        grid[row, legend_c0 + 1] = color

    for spec in specs:
        for r, c in _wall_cells(spec["r0"], spec["c0"], spec["h"], spec["w"], spec["direction"]):
            grid[r, c] = GREY
        seed_r = spec["r0"] + 1 + spec["seed_inner"][0]
        seed_c = spec["c0"] + 1 + spec["seed_inner"][1]
        grid[seed_r, seed_c] = spec["color"]
        for r, c in spec["obstacles"]:
            grid[r, c] = ORANGE
    return grid


def _validate_input(grid: np.ndarray, expected_counts: dict[int, int]) -> None:
    walls, legend = _parse_scene(grid)
    counts = {wall["color"]: len(_filled_cells(grid, wall)) for wall in walls}
    assert counts == expected_counts
    values = list(counts.values())
    assert len(values) == len(set(values))
    assert all(2 <= value <= 8 for value in values)
    obstacle_flags = []
    for wall in walls:
        obstacle_flags.append(
            any(int(grid[r, c]) == ORANGE for lane in _ray_lanes(wall["bbox"], wall["direction"], grid.shape) for r, c in lane)
        )
    assert any(obstacle_flags)
    assert any(not flag for flag in obstacle_flags)
    assert set(legend["keys"]) == set(expected_counts)


def generate_input() -> np.ndarray:
    for _ in range(2000):
        count = random.choice((3, 4))
        directions = random.sample(DIRS, count)
        target_counts = random.sample(list(range(2, 9)), count)
        height = random.randint(18, 23)
        play_w = 9

        candidate_lists: dict[str, list[dict]] = {}
        if any(not (candidate_lists.setdefault(direction, _candidate_specs(direction, target, height, play_w)) or True) for direction, target in zip(directions, target_counts)):
            continue
        if any(len(candidate_lists[direction]) == 0 for direction in directions):
            continue

        chosen_specs = []
        for direction, target in zip(directions, target_counts):
            pool = []
            for spec in candidate_lists[direction]:
                spec = dict(spec)
                spec["grid_h"] = height
                spec["play_w"] = play_w
                spec["obstacles"] = _obstacles_from_visible(spec)
                spec["has_obstacle"] = bool(spec["obstacles"])
                if _spec_count(spec) == target:
                    pool.append(spec)
            if not pool:
                chosen_specs = []
                break
            chosen_specs.append(random.choice(pool))
        if not chosen_specs:
            continue

        if not any(spec["has_obstacle"] for spec in chosen_specs):
            continue
        if not any(not spec["has_obstacle"] for spec in chosen_specs):
            continue

        colors = random.sample(SEED_COLORS, count)
        for spec, color in zip(chosen_specs, colors):
            spec["color"] = color
            inner_h = spec["h"] - 2
            inner_w = spec["w"] - 2
            seed_options = [(r, c) for r in range(inner_h) for c in range(inner_w)]
            spec["seed_inner"] = random.choice(seed_options)

        expected_counts = {spec["color"]: _spec_count(spec) for spec in chosen_specs}
        if len(set(expected_counts.values())) != count:
            continue

        max_count = max(expected_counts.values())
        legend_w = _legend_width(max_count)
        width = play_w + 2 + legend_w
        if width > 30:
            continue
        legend_c0 = play_w + 2
        ordered_colors = [color for color in KEY_ORDER if color in expected_counts]
        legend_rows = []
        legend_h = 2 * len(ordered_colors) + 1
        legend_r0 = max(1, min(height - legend_h - 1, height // 2 - legend_h // 2))
        for i in range(len(ordered_colors)):
            legend_rows.append(legend_r0 + 1 + 2 * i)

        grid = np.zeros((height, width), dtype=int)
        frame = _rect_frame_mask(legend_h, legend_w)
        grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + legend_w][frame] = GREY
        for row, color in zip(legend_rows, ordered_colors):
            grid[row, legend_c0 + 1] = color

        occupied = set()
        for r in range(legend_r0, legend_r0 + legend_h):
            for c in range(legend_c0, legend_c0 + legend_w):
                if grid[r, c] != BG:
                    occupied.add((r, c))
        ok = True
        for spec in chosen_specs:
            wall = _wall_cells(spec["r0"], spec["c0"], spec["h"], spec["w"], spec["direction"])
            seed_r = spec["r0"] + 1 + spec["seed_inner"][0]
            seed_c = spec["c0"] + 1 + spec["seed_inner"][1]
            body = set(wall) | {(seed_r, seed_c)} | set(spec["obstacles"])
            if _touches_occupied(body, occupied):
                ok = False
                break
            occupied |= body
            for r, c in wall:
                grid[r, c] = GREY
            grid[seed_r, seed_c] = spec["color"]
            for r, c in spec["obstacles"]:
                grid[r, c] = ORANGE
        if not ok:
            continue

        _validate_input(grid, expected_counts)
        expected = _paint_output(grid, _parse_scene(grid)[0], _parse_scene(grid)[1])
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
