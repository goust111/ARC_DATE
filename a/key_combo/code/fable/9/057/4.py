from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(int(v) for v in np.unique(grid))
    if exclude is None:
        return colors
    return [c for c in colors if c not in exclude]


def _components8(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            comps.append(comp)
    return comps


def _closed_u_containers(grid: np.ndarray, color: int, bg: int) -> list[BBox]:
    h, w = grid.shape
    boxes: list[BBox] = []
    for bottom in range(1, h):
        for left in range(w - 2):
            if int(grid[bottom, left]) != color:
                continue
            for right in range(left + 2, w):
                if int(grid[bottom, right]) != color:
                    continue
                if not np.all(grid[bottom, left : right + 1] == color):
                    continue
                top = bottom
                while top > 0 and int(grid[top - 1, left]) == color and int(grid[top - 1, right]) == color:
                    top -= 1
                if top == bottom:
                    continue
                interior = grid[top:bottom, left + 1 : right]
                if interior.size == 0 or not np.all(interior == bg):
                    continue
                boxes.append((top, bottom, left, right))
    return sorted(set(boxes), key=lambda box: (box[2], box[0], box[1], box[3]))


def _segment_from_piece(piece: list[Coord]) -> dict:
    piece = sorted(piece)
    assert len(piece) >= 2, "every chute piece must reveal its diagonal direction"
    diffs_a = {r - c for r, c in piece}
    diffs_b = {r + c for r, c in piece}
    if len(diffs_a) == 1:
        direction = 1
        line_key = ("backslash", int(next(iter(diffs_a))))
    elif len(diffs_b) == 1:
        direction = -1
        line_key = ("slash", int(next(iter(diffs_b))))
    else:
        raise AssertionError("chute piece must be a straight 45-degree diagonal")
    top = min(piece, key=lambda p: (p[0], p[1]))
    bottom = max(piece, key=lambda p: (p[0], p[1]))
    return {
        "top": top,
        "bottom": bottom,
        "visible": list(piece),
        "visible_set": set(piece),
        "gap": None,
        "dir": direction,
        "line_key": line_key,
    }


def _merge_chute_pieces(pieces: list[dict]) -> list[dict]:
    groups: dict[tuple[str, int], list[dict]] = {}
    for piece in pieces:
        groups.setdefault(piece["line_key"], []).append(piece)

    chutes: list[dict] = []
    for key, group in groups.items():
        group.sort(key=lambda item: item["top"])
        i = 0
        while i < len(group):
            current = group[i]
            merged = {
                "top": current["top"],
                "bottom": current["bottom"],
                "visible": list(current["visible"]),
                "visible_set": set(current["visible_set"]),
                "gap": None,
                "dir": current["dir"],
                "line_key": key,
            }
            if i + 1 < len(group):
                nxt = group[i + 1]
                dr, dc = 1, current["dir"]
                gap = (current["bottom"][0] + dr, current["bottom"][1] + dc)
                expected_top = (current["bottom"][0] + 2 * dr, current["bottom"][1] + 2 * dc)
                if nxt["top"] == expected_top:
                    merged["bottom"] = nxt["bottom"]
                    merged["visible"].extend(nxt["visible"])
                    merged["visible_set"].update(nxt["visible_set"])
                    merged["gap"] = gap
                    i += 1
            merged["visible"].sort()
            merged["bottom_t"] = merged["bottom"][0] - merged["top"][0]
            merged["gap_t"] = None if merged["gap"] is None else merged["gap"][0] - merged["top"][0]
            chutes.append(merged)
            i += 1
    return sorted(chutes, key=lambda chute: (chute["top"][0], chute["top"][1]))


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    fg = _colors_present(g, exclude={bg})
    assert len(fg) == 3, "expected marker, chute, and container colors"

    marker_colors = [c for c in fg if int(np.count_nonzero(g == c)) == 1]
    assert len(marker_colors) == 1, "expected one singleton marker color"
    marker_color = marker_colors[0]
    marker_pos = tuple(map(int, np.argwhere(g == marker_color)[0]))
    assert marker_pos[0] == 0, "marker must sit on the top row"

    remaining = [c for c in fg if c != marker_color]
    container_candidates = []
    for color in remaining:
        boxes = _closed_u_containers(g, color, bg)
        if boxes:
            container_candidates.append((color, boxes))
    assert len(container_candidates) == 1, "expected exactly one container color"
    container_color, containers = container_candidates[0]
    chute_color = next(c for c in remaining if c != container_color)

    pieces = [_segment_from_piece(comp) for comp in _components8(g == chute_color)]
    chutes = _merge_chute_pieces(pieces)

    pos_to_chute: dict[Coord, int] = {}
    for idx, chute in enumerate(chutes):
        for cell in chute["visible"]:
            assert cell not in pos_to_chute, "visible chute cells must not overlap"
            pos_to_chute[cell] = idx

    return {
        "grid": g,
        "bg": bg,
        "marker_color": marker_color,
        "marker_pos": marker_pos,
        "container_color": container_color,
        "containers": containers,
        "chute_color": chute_color,
        "chutes": chutes,
        "pos_to_chute": pos_to_chute,
    }


def _first_hit(scene: dict, row: int, col: int) -> tuple[int, dict] | None:
    h = scene["grid"].shape[0]
    for r in range(row + 1, h):
        idx = scene["pos_to_chute"].get((r, col))
        if idx is not None:
            return r, scene["chutes"][idx]
    return None


def _simulate_fall(scene: dict) -> tuple[Coord, list[dict]]:
    row, col = scene["marker_pos"]
    events: list[dict] = []
    while True:
        hit = _first_hit(scene, row, col)
        if hit is None:
            return (row, col), events
        entry_row, chute = hit
        entry_t = entry_row - chute["top"][0]
        leak = chute["gap_t"] is not None and entry_t < chute["gap_t"]
        landing = chute["gap"] if leak else chute["bottom"]
        events.append(
            {
                "entry": (entry_row, col),
                "landing": landing,
                "leak": leak,
                "line_key": chute["line_key"],
            }
        )
        row, col = landing


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route the top marker through diagonal chutes, leaking at single-cell breaks, then fill the caught container or the whole floor.

    Input structure:
    - A dominant background contains exactly one singleton marker on the top row,
      one chute color forming several non-intersecting 45-degree diagonal segments,
      and one container color forming one or more upward-open U-shaped containers
      near the bottom.
    - Some chutes are intact diagonal segments; others are split into two collinear
      pieces by exactly one erased background cell in the middle.
    - The marker's initial column does not directly fall into any container interior.

    Rule:
    - Erase the original marker.
    - Let the marker fall vertically. Whenever that falling column first enters a
      chute cell, follow that chute toward its lower end.
    - If the chute has a one-cell break below the entry point, leak from the break
      cell and resume vertical falling there; otherwise slide to the chute's lower
      endpoint and continue falling from that endpoint.
    - Repeat until no chute is hit. If the final fall column lies inside exactly
      one container opening, fill that container's empty interior with the marker
      color; otherwise fill the entire bottom row with the marker color.
    - Keep all chute and container pixels unchanged.

    Key insight:
    - The answer depends on repeatedly rewriting the drop column: intact chutes
      hand the marker to their lower endpoints, while broken chutes divert it only
      as far as the missing cell before it leaks straight down.
    """
    scene = _parse_scene(grid)
    out = scene["grid"].copy()
    marker_color = scene["marker_color"]
    bg = scene["bg"]
    out[scene["marker_pos"]] = bg

    final_pos, _events = _simulate_fall(scene)
    final_col = final_pos[1]
    catches = [box for box in scene["containers"] if box[2] < final_col < box[3]]
    assert len(catches) <= 1, "final fall column must choose at most one container"

    if catches:
        top, bottom, left, right = catches[0]
        out[top:bottom, left + 1 : right] = marker_color
    else:
        assert np.all(out[-1, :] == bg), "uncaught marker requires an empty bottom row"
        out[-1, :] = marker_color
    return out


GEN_BG = 0
GEN_RETRIES = 600
GEN_WIDTH_RANGE = (17, 24)
GEN_PATH_COUNT_CHOICES = (2, 2, 3)
GEN_DELTA_CHOICES = (3, 4)
GEN_EXTRA_CONTAINER_CHOICES = (0, 1, 1, 2)
GEN_EXTRA_CHUTE_CHOICES = (0, 1, 1, 2)


def _segment_spec(top_row: int, top_col: int, direction: int, delta: int, broken: bool, lower_len: int) -> dict:
    total_t = delta + lower_len
    visible: list[Coord] = []
    reserved: list[Coord] = []
    gap = None
    for t in range(total_t + 1):
        cell = (top_row + t, top_col + direction * t)
        reserved.append(cell)
        if broken and t == delta:
            gap = cell
        else:
            visible.append(cell)
    landing = (top_row + delta, top_col + direction * delta)
    bottom_visible = (top_row + total_t, top_col + direction * total_t)
    return {
        "top": (top_row, top_col),
        "dir": direction,
        "delta": delta,
        "broken": broken,
        "lower_len": lower_len,
        "gap": gap,
        "landing": landing,
        "bottom_visible": bottom_visible,
        "visible": visible,
        "visible_set": set(visible),
        "reserved": reserved,
        "reserved_set": set(reserved),
        "line_key": ("backslash", top_row - top_col) if direction == 1 else ("slash", top_row + top_col),
    }


def _compatible_with(existing: list[dict], spec: dict) -> bool:
    for other in existing:
        if other["line_key"] == spec["line_key"]:
            return False
        if other["reserved_set"] & spec["reserved_set"]:
            return False
        for a in spec["visible"]:
            for b in other["visible"]:
                if max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1:
                    return False
        if spec["gap"] is not None and spec["gap"] in other["visible_set"]:
            return False
        if other["gap"] is not None and other["gap"] in spec["visible_set"]:
            return False
    return True


def _path_columns_are_clean(start_col: int, specs: list[dict]) -> bool:
    all_visible = [cell for spec in specs for cell in spec["visible"]]
    phase_row, phase_col = 0, start_col
    for spec in specs:
        hits = sorted(r for r, c in all_visible if c == phase_col and r > phase_row)
        if not hits or hits[0] != spec["top"][0]:
            return False
        phase_row, phase_col = spec["landing"]
    if any(c == phase_col and r > phase_row for r, c in all_visible):
        return False
    return True


def _make_container_box(col: int, bottom: int, interior_h: int) -> BBox:
    return (bottom - interior_h, bottom, col - 1, col + 1)


def _draw_container(grid: np.ndarray, box: BBox, color: int) -> None:
    top, bottom, left, right = box
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color
    grid[bottom, left : right + 1] = color


def _draw_chute(grid: np.ndarray, spec: dict, color: int) -> None:
    for r, c in spec["visible"]:
        grid[r, c] = color


def _build_path(width: int) -> dict | None:
    n_path = random.choice(GEN_PATH_COUNT_CHOICES)
    broken_idx = random.randrange(n_path) if random.random() < 0.65 else None
    start_col = random.randint(4, width - 5)

    specs: list[dict] = []
    milestone_cols: list[int] = []
    landings: list[Coord] = []
    current_row = 0
    current_col = start_col

    for i in range(n_path):
        placed = False
        for direction in random.sample([-1, 1], 2):
            delta = random.choice(GEN_DELTA_CHOICES)
            new_col = current_col + direction * delta
            if not (2 <= new_col <= width - 3):
                continue
            if any(abs(new_col - old) < 3 for old in milestone_cols):
                continue
            gap_before = random.randint(2, 3)
            top_row = current_row + gap_before
            broken = i == broken_idx
            lower_len = random.randint(2, 3) if broken else 0
            spec = _segment_spec(top_row, current_col, direction, delta, broken, lower_len)
            if spec["bottom_visible"][0] > 20:
                continue
            if not (1 <= spec["bottom_visible"][1] <= width - 2):
                continue
            if not _compatible_with(specs, spec):
                continue
            specs.append(spec)
            milestone_cols.append(spec["landing"][1])
            landings.append(spec["landing"])
            current_row, current_col = spec["landing"]
            placed = True
            break
        if not placed:
            return None

    if not _path_columns_are_clean(start_col, specs):
        return None

    return {
        "start_col": start_col,
        "specs": specs,
        "milestone_cols": milestone_cols,
        "landings": landings,
        "max_visual_row": max(spec["bottom_visible"][0] for spec in specs),
        "has_broken": broken_idx is not None,
    }


def _add_distractors(height: int, width: int, ceiling_row: int, all_specs: list[dict], fall_cols: set[int]) -> list[dict] | None:
    target = random.choice(GEN_EXTRA_CHUTE_CHOICES)
    distractors: list[dict] = []
    if target == 0:
        return distractors

    top_limit = ceiling_row - 4
    if top_limit < 2:
        return None

    for _ in range(target):
        placed = False
        for _attempt in range(80):
            top_row = random.randint(2, top_limit)
            top_col = random.randint(2, width - 3)
            direction = random.choice([-1, 1])
            delta = random.randint(2, 4)
            end_col = top_col + direction * delta
            if not (2 <= end_col <= width - 3):
                continue
            broken = random.random() < 0.35
            lower_len = random.randint(2, 3) if broken else 0
            spec = _segment_spec(top_row, top_col, direction, delta, broken, lower_len)
            if spec["bottom_visible"][0] >= ceiling_row:
                continue
            if not (1 <= spec["bottom_visible"][1] <= width - 2):
                continue
            if any(c in fall_cols for _, c in spec["visible"]):
                continue
            if not _compatible_with(all_specs + distractors, spec):
                continue
            distractors.append(spec)
            placed = True
            break
        if not placed:
            return None
    return distractors


def _expected_output(input_grid: np.ndarray, marker_color: int, catch_col: int | None) -> np.ndarray:
    out = input_grid.copy()
    out[np.argwhere(out == marker_color)[0][0], np.argwhere(out == marker_color)[0][1]] = GEN_BG
    if catch_col is None:
        out[-1, :] = marker_color
        return out

    scene = _parse_scene(input_grid)
    for top, bottom, left, right in scene["containers"]:
        if left < catch_col < right:
            out[top:bottom, left + 1 : right] = marker_color
            return out
    raise AssertionError("missing target container")


def _validate_generated(inp: np.ndarray, expected: np.ndarray, design: dict) -> bool:
    try:
        scene = _parse_scene(inp)
        final_pos, events = _simulate_fall(scene)
        actual = main(inp)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if len(scene["chutes"]) != design["total_chutes"]:
        return False
    if [event["landing"] for event in events] != design["landings"]:
        return False
    if [event["leak"] for event in events] != design["leak_pattern"]:
        return False
    if final_pos[1] != design["final_col"]:
        return False
    if any(box[2] < design["start_col"] < box[3] for box in scene["containers"]):
        return False

    for col in design["decoy_cols"]:
        if sum(1 for box in scene["containers"] if box[2] < col < box[3]) != 1:
            return False
    if design["catch"]:
        if sum(1 for box in scene["containers"] if box[2] < final_pos[1] < box[3]) != 1:
            return False
    else:
        if any(box[2] < final_pos[1] < box[3] for box in scene["containers"]):
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    width = random.randint(*GEN_WIDTH_RANGE)
    path = _build_path(width)
    if path is None:
        return None

    catch = random.random() < 0.5
    final_col = path["milestone_cols"][-1]
    decoy_cols = path["milestone_cols"][:-1] if catch else path["milestone_cols"][:-1]
    target_cols = path["milestone_cols"] if catch else path["milestone_cols"][:-1]

    container_bottom = path["max_visual_row"] + random.randint(6, 7)
    if container_bottom > 27:
        return None
    height = container_bottom + 2

    boxes = [_make_container_box(col, container_bottom, random.randint(2, 4)) for col in target_cols]
    used_container_cols = set(target_cols)
    fall_cols = {path["start_col"], *path["milestone_cols"]}

    extra_count = random.choice(GEN_EXTRA_CONTAINER_CHOICES)
    extra_candidates = [
        col
        for col in range(2, width - 2)
        if col not in used_container_cols
        and col not in fall_cols
        and all(abs(col - other) >= 3 for other in used_container_cols | fall_cols)
    ]
    random.shuffle(extra_candidates)
    for col in extra_candidates[:extra_count]:
        boxes.append(_make_container_box(col, container_bottom, random.randint(2, 4)))
        used_container_cols.add(col)

    min_container_top = min(box[0] for box in boxes)
    distractors = _add_distractors(height, width, min_container_top - 1, list(path["specs"]), fall_cols)
    if distractors is None:
        return None

    colors = random.sample([c for c in range(1, 10)], 3)
    marker_color, chute_color, container_color = colors
    grid = np.full((height, width), GEN_BG, dtype=int)
    for box in boxes:
        _draw_container(grid, box, container_color)
    for spec in path["specs"] + distractors:
        _draw_chute(grid, spec, chute_color)
    grid[0, path["start_col"]] = marker_color

    expected = _expected_output(grid, marker_color, final_col if catch else None)
    design = {
        "start_col": path["start_col"],
        "landings": path["landings"],
        "leak_pattern": [spec["broken"] for spec in path["specs"]],
        "final_col": final_col,
        "catch": catch,
        "decoy_cols": decoy_cols,
        "total_chutes": len(path["specs"]) + len(distractors),
    }
    if not _validate_generated(grid, expected, design):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chute scene")
