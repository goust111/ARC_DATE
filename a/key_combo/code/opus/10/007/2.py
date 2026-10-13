import random
from collections import deque

import numpy as np


BG = 0
TOP_BAR = 1
BOTTOM_BAR = 2
TRACK = 3
OBJECT_COLORS = (4, 5, 6, 7, 8, 9)
GEN_RETRIES = 400

SHAPE_TEMPLATES = {
    2: [
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
    ],
    3: [
        ((0, -2), (0, -1), (0, 0)),
        ((0, -1), (0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
    ],
    4: [
        ((0, -2), (0, -1), (0, 0), (0, 1)),
        ((0, -1), (0, 0), (0, 1), (0, 2)),
    ],
    5: [
        ((0, -2), (0, -1), (0, 0), (0, 1), (0, 2)),
    ],
}


def _is_4_connected(coords):
    cells = {tuple(map(int, rc)) for rc in coords}
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == cells


def _find_track(grid):
    h, w = grid.shape
    candidates = []
    for c in range(w):
        rows = [r for r in range(1, h - 1) if int(grid[r, c]) != BG]
        if len(rows) < 5:
            continue
        values = [int(grid[r, c]) for r in rows]
        if any(v in (BG, TOP_BAR, BOTTOM_BAR) for v in values):
            continue
        diffs = {rows[i + 1] - rows[i] for i in range(len(rows) - 1)}
        if len(diffs) == 1 and next(iter(diffs)) in (2, 3):
            candidates.append((c, rows))
    assert len(candidates) == 1, "expected exactly one vertical track"
    track_col, slot_rows = candidates[0]
    return int(track_col), tuple(int(r) for r in slot_rows)


def _extract_scene(grid):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30
    assert np.all(g[0] == TOP_BAR)
    assert np.all(g[-1] == BOTTOM_BAR)

    track_col, slot_rows = _find_track(g)
    spacing = slot_rows[1] - slot_rows[0]
    assert all(slot_rows[i + 1] - slot_rows[i] == spacing for i in range(len(slot_rows) - 1))

    slot_index_by_row = {row: idx for idx, row in enumerate(slot_rows)}
    objects = []
    seen_colors = set()
    for color in np.unique(g):
        color = int(color)
        if color not in OBJECT_COLORS:
            continue
        coords = np.argwhere(g == color)
        assert coords.size > 0
        assert color not in seen_colors
        seen_colors.add(color)
        assert _is_4_connected(coords), "each object must be one 4-connected component"
        area = int(coords.shape[0])
        assert 2 <= area <= 5

        track_hits = [
            (int(r), int(c))
            for r, c in coords
            if int(c) == track_col and int(r) in slot_index_by_row
        ]
        assert len(track_hits) == 1, "each object must cover exactly one track slot"
        anchor_row = track_hits[0][0]
        slot_index = slot_index_by_row[anchor_row]

        center_row = float(coords[:, 0].mean())
        top_distance = center_row
        bottom_distance = (h - 1) - center_row
        direction = -1 if top_distance <= bottom_distance else 1
        target_color = TOP_BAR if direction == -1 else BOTTOM_BAR
        target_distance = top_distance if direction == -1 else bottom_distance
        raw_target_index = slot_index + direction * area
        assert 0 <= raw_target_index < len(slot_rows), "raw movement must stay on track"

        objects.append(
            {
                "color": color,
                "coords": [tuple(map(int, rc)) for rc in coords],
                "area": area,
                "anchor_row": anchor_row,
                "slot_index": slot_index,
                "direction": direction,
                "target_color": target_color,
                "target_distance": target_distance,
                "raw_target_index": raw_target_index,
            }
        )

    assert 2 <= len(objects) <= 3
    for row in slot_rows:
        v = int(g[row, track_col])
        assert v == TRACK or v in OBJECT_COLORS

    return {
        "grid": g,
        "height": h,
        "width": w,
        "track_col": track_col,
        "slot_rows": slot_rows,
        "spacing": spacing,
        "objects": objects,
    }


def _placement_order(objects):
    return sorted(
        objects,
        key=lambda obj: (
            obj["target_distance"],
            0 if obj["direction"] == -1 else 1,
            obj["anchor_row"] if obj["direction"] == -1 else -obj["anchor_row"],
            obj["color"],
        ),
    )


def _simulate_objects(scene):
    slot_rows = scene["slot_rows"]
    spacing = scene["spacing"]
    occupied = set()
    placements = []

    for obj in _placement_order(scene["objects"]):
        last_shift_slots = 0
        collision = False
        for step in range(1, obj["area"] + 1):
            shift_slots = obj["direction"] * step
            target_index = obj["slot_index"] + shift_slots
            if not 0 <= target_index < len(slot_rows):
                collision = True
                break
            dr = shift_slots * spacing
            trial = [(r + dr, c) for r, c in obj["coords"]]
            if any((rr, cc) in occupied for rr, cc in trial):
                collision = True
                break
            last_shift_slots = shift_slots

        dr = last_shift_slots * spacing
        final_coords = [(r + dr, c) for r, c in obj["coords"]]
        for cell in final_coords:
            occupied.add(cell)
        placements.append(
            {
                **obj,
                "collision": collision,
                "moved_slots": abs(last_shift_slots),
                "final_slot_index": obj["slot_index"] + last_shift_slots,
                "final_coords": final_coords,
            }
        )

    return placements


def _render_output(scene, placements):
    out = np.full_like(scene["grid"], BG)
    out[0, :] = TOP_BAR
    out[-1, :] = BOTTOM_BAR
    for row in scene["slot_rows"]:
        out[row, scene["track_col"]] = TRACK
    for placement in placements:
        for r, c in placement["final_coords"]:
            out[r, c] = placement["target_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each track-covering object by its own area toward the nearer edge bar.

    Input structure:
    - The background is black, with a full-width blue bar on the top row and a full-width red bar on the bottom row.
    - A single vertical track appears as evenly spaced green dots in one column, except where 2-3 small monochrome objects cover individual track slots.
    - Each foreground object is 4-connected, uses a distinct non-bar color, and overlaps exactly one track dot position.

    Rule:
    - For each object, count its pixels to get N, then compare the object's center row to the top and bottom bars to choose the nearer bar; ties go upward.
    - Translate the whole object vertically by N track slots toward that nearer bar, with one slot equal to the green-dot spacing.
    - Process objects in order of increasing initial distance to their chosen bar; if a later object would land on an already occupied slot, it stops at the last empty slot before the collision.
    - Restore vacated track cells to green and recolor each moved object to the color of the bar it moved toward.

    Key insight:
    - Area determines how many track slots an object tries to travel, but the movement direction and recolor both come from the nearer edge bar, and collision resolution depends on sorting by initial bar distance rather than by area.
    """
    scene = _extract_scene(grid)
    placements = _simulate_objects(scene)
    return _render_output(scene, placements)


def _simulate_specs(specs, slot_rows, height):
    objects = []
    for spec in specs:
        row = slot_rows[spec["slot_index"]]
        direction = -1 if (2 * row) <= (height - 1) else 1
        distance = row if direction == -1 else (height - 1 - row)
        objects.append(
            {
                "slot_index": spec["slot_index"],
                "area": spec["area"],
                "direction": direction,
                "distance": distance,
                "is_tie": 2 * row == (height - 1),
            }
        )

    order = sorted(
        objects,
        key=lambda obj: (
            obj["distance"],
            0 if obj["direction"] == -1 else 1,
            obj["slot_index"] if obj["direction"] == -1 else -obj["slot_index"],
        ),
    )

    occupied = set()
    placements = []
    for obj in order:
        last_index = obj["slot_index"]
        collision = False
        for step in range(1, obj["area"] + 1):
            next_index = obj["slot_index"] + obj["direction"] * step
            if next_index in occupied:
                collision = True
                break
            last_index = next_index
        occupied.add(last_index)
        placements.append(
            {
                **obj,
                "collision": collision,
                "raw_target_index": obj["slot_index"] + obj["direction"] * obj["area"],
                "final_slot_index": last_index,
                "moved_slots": abs(last_index - obj["slot_index"]),
            }
        )
    return placements


def _valid_specs(specs, slot_rows, height):
    if len({spec["slot_index"] for spec in specs}) != len(specs):
        return False
    if max(spec["area"] for spec in specs) < 3:
        return False
    distances = []
    for spec in specs:
        row = slot_rows[spec["slot_index"]]
        distance = row if (2 * row) <= (height - 1) else (height - 1 - row)
        distances.append(distance)
    return len(set(distances)) == len(distances)


def _choose_top_collision(slot_rows, height, slot_count, max_area, rng):
    up_slots = [i for i, row in enumerate(slot_rows) if (2 * row) <= (height - 1)]
    options = []
    for i1 in up_slots:
        for i2 in up_slots:
            if i1 >= i2:
                continue
            for a1 in range(2, min(max_area, i1) + 1):
                p1 = i1 - a1
                for a2 in range(2, min(max_area, i2) + 1):
                    raw2 = i2 - a2
                    if raw2 <= p1 <= i2 - 1:
                        specs = [{"slot_index": i1, "area": a1}, {"slot_index": i2, "area": a2}]
                        if _valid_specs(specs, slot_rows, height):
                            options.append(specs)
    if not options:
        return None
    specs = [dict(spec) for spec in rng.choice(options)]
    if rng.random() < 0.55:
        extras = _candidate_extra_specs(slot_rows, height, slot_count, max_area, used_slots={spec["slot_index"] for spec in specs})
        rng.shuffle(extras)
        for extra in extras:
            trial = specs + [extra]
            sim = _simulate_specs(trial, slot_rows, height)
            if _valid_specs(trial, slot_rows, height) and any(p["collision"] and p["direction"] == -1 for p in sim):
                specs = trial
                break
    return specs


def _choose_bottom_collision(slot_rows, height, slot_count, max_area, rng):
    down_slots = [i for i, row in enumerate(slot_rows) if (2 * row) > (height - 1)]
    options = []
    for i1 in down_slots:
        for i2 in down_slots:
            if i1 <= i2:
                continue
            for a1 in range(2, min(max_area, slot_count - 1 - i1) + 1):
                p1 = i1 + a1
                for a2 in range(2, min(max_area, slot_count - 1 - i2) + 1):
                    raw2 = i2 + a2
                    if i2 + 1 <= p1 <= raw2:
                        specs = [{"slot_index": i1, "area": a1}, {"slot_index": i2, "area": a2}]
                        if _valid_specs(specs, slot_rows, height):
                            options.append(specs)
    if not options:
        return None
    specs = [dict(spec) for spec in rng.choice(options)]
    if rng.random() < 0.55:
        extras = _candidate_extra_specs(slot_rows, height, slot_count, max_area, used_slots={spec["slot_index"] for spec in specs})
        rng.shuffle(extras)
        for extra in extras:
            trial = specs + [extra]
            sim = _simulate_specs(trial, slot_rows, height)
            if _valid_specs(trial, slot_rows, height) and any(p["collision"] and p["direction"] == 1 for p in sim):
                specs = trial
                break
    return specs


def _candidate_extra_specs(slot_rows, height, slot_count, max_area, used_slots):
    extras = []
    for i, row in enumerate(slot_rows):
        if i in used_slots:
            continue
        direction = -1 if (2 * row) <= (height - 1) else 1
        limit = i if direction == -1 else (slot_count - 1 - i)
        for area in range(2, min(max_area, limit) + 1):
            extras.append({"slot_index": i, "area": area})
    return extras


def _choose_tie_specs(slot_rows, height, slot_count, max_area, rng):
    tie_slots = [i for i, row in enumerate(slot_rows) if 2 * row == (height - 1)]
    if not tie_slots:
        return None
    tie_slot = rng.choice(tie_slots)
    specs = [{"slot_index": tie_slot, "area": rng.randint(2, min(max_area, tie_slot))}]
    extras = _candidate_extra_specs(slot_rows, height, slot_count, max_area, used_slots={tie_slot})
    rng.shuffle(extras)
    for extra1 in extras:
        trial = specs + [extra1]
        if not _valid_specs(trial, slot_rows, height):
            continue
        sim = _simulate_specs(trial, slot_rows, height)
        if any(p["is_tie"] for p in sim) and len({p["direction"] for p in sim}) >= 2:
            return trial
    for extra1 in extras:
        for extra2 in extras:
            if extra1["slot_index"] == extra2["slot_index"]:
                continue
            trial = specs + [extra1, extra2]
            if not _valid_specs(trial, slot_rows, height):
                continue
            sim = _simulate_specs(trial, slot_rows, height)
            if any(p["is_tie"] for p in sim) and len({p["direction"] for p in sim}) >= 2:
                return trial
    return None


def _choose_mixed_specs(slot_rows, height, slot_count, max_area, rng):
    candidates = _candidate_extra_specs(slot_rows, height, slot_count, max_area, used_slots=set())
    for _ in range(600):
        count = 3 if rng.random() < 0.6 else 2
        specs = []
        used = set()
        rng.shuffle(candidates)
        for candidate in candidates:
            if candidate["slot_index"] in used:
                continue
            specs.append(dict(candidate))
            used.add(candidate["slot_index"])
            if len(specs) == count:
                break
        if len(specs) != count or not _valid_specs(specs, slot_rows, height):
            continue
        sim = _simulate_specs(specs, slot_rows, height)
        if len({p["direction"] for p in sim}) < 2:
            continue
        if any(p["collision"] for p in sim):
            continue
        return specs
    return None


def _build_layout(mode, rng):
    if mode == "tie":
        spacing = 2
        slot_count = rng.choice((7, 9))
        top_gap = bottom_gap = rng.choice((1, 2))
    elif mode == "bottom_collision":
        spacing = 2
        slot_count = rng.randint(8, 10)
        top_gap = rng.randint(1, 2)
        bottom_gap = rng.randint(1, 2)
    elif mode == "top_collision":
        spacing = 2 if rng.random() < 0.75 else 3
        slot_count = rng.randint(7, 10) if spacing == 2 else rng.randint(6, 8)
        top_gap = rng.randint(1, 2)
        bottom_gap = rng.randint(1, 2)
    else:
        spacing = 2 if rng.random() < 0.7 else 3
        slot_count = rng.randint(7, 10) if spacing == 2 else rng.randint(6, 8)
        top_gap = rng.randint(1, 2)
        bottom_gap = rng.randint(1, 2)

    slot_start = 1 + top_gap
    slot_rows = [slot_start + i * spacing for i in range(slot_count)]
    height = slot_rows[-1] + bottom_gap + 2
    width = rng.randint(9, 13)
    return height, width, spacing, tuple(slot_rows)


def _choose_specs(mode, slot_rows, height, rng):
    slot_count = len(slot_rows)
    max_area = min(5, slot_count // 2)
    if max_area < 2:
        return None

    choosers = {
        "top_collision": _choose_top_collision,
        "bottom_collision": _choose_bottom_collision,
        "tie": _choose_tie_specs,
        "mixed": _choose_mixed_specs,
    }
    return choosers[mode](slot_rows, height, slot_count, max_area, rng)


def _pick_shapes(specs, width, rng):
    shapes = []
    min_col = 0
    max_col = width - 1
    low_needed = 0
    high_needed = width - 1
    for spec in specs:
        shape = [tuple(cell) for cell in rng.choice(SHAPE_TEMPLATES[spec["area"]])]
        shape_min = min(c for _, c in shape)
        shape_max = max(c for _, c in shape)
        low_needed = max(low_needed, -shape_min)
        high_needed = min(high_needed, max_col - shape_max)
        shapes.append(shape)
    if low_needed > high_needed:
        return None
    track_col = rng.randint(low_needed, high_needed)
    return shapes, track_col


def _render_input(height, width, slot_rows, track_col, specs, shapes, colors):
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = TOP_BAR
    grid[-1, :] = BOTTOM_BAR
    for row in slot_rows:
        grid[row, track_col] = TRACK
    for spec, shape, color in zip(specs, shapes, colors):
        row = slot_rows[spec["slot_index"]]
        for dr, dc in shape:
            grid[row + dr, track_col + dc] = color
    return grid


def _validate_generated(grid, mode):
    scene = _extract_scene(grid)
    placements = _simulate_objects(scene)
    out = main(grid)
    ref = _render_output(scene, placements)
    assert np.array_equal(out, ref)
    assert len({obj["color"] for obj in scene["objects"]}) == len(scene["objects"])
    assert len(scene["slot_rows"]) > max(obj["area"] for obj in scene["objects"])
    assert any(obj["area"] >= 3 for obj in scene["objects"])

    for placement in placements:
        assert placement["target_color"] in (TOP_BAR, BOTTOM_BAR)
        if not placement["collision"]:
            assert placement["moved_slots"] == placement["area"]
        else:
            assert placement["moved_slots"] < placement["area"]
            blocked_index = placement["final_slot_index"] + placement["direction"]
            blockers = {
                other["final_slot_index"]
                for other in placements
                if other["target_distance"] < placement["target_distance"]
            }
            assert blocked_index in blockers

    if mode == "top_collision":
        assert any(p["collision"] and p["direction"] == -1 for p in placements)
    elif mode == "bottom_collision":
        assert any(p["collision"] and p["direction"] == 1 for p in placements)
    elif mode == "tie":
        assert any(2 * p["anchor_row"] == scene["height"] - 1 for p in placements)
    elif mode == "mixed":
        assert len({p["direction"] for p in placements}) >= 2
        assert not any(p["collision"] for p in placements)


def _gen_attempt(rng):
    mode = rng.choices(
        ["top_collision", "bottom_collision", "tie", "mixed"],
        weights=[3, 2, 2, 3],
        k=1,
    )[0]
    height, width, _spacing, slot_rows = _build_layout(mode, rng)
    specs = _choose_specs(mode, slot_rows, height, rng)
    if not specs:
        return None
    shapes_and_track = _pick_shapes(specs, width, rng)
    if shapes_and_track is None:
        return None
    shapes, track_col = shapes_and_track
    colors = rng.sample(OBJECT_COLORS, len(specs))
    grid = _render_input(height, width, slot_rows, track_col, specs, shapes, colors)
    _validate_generated(grid, mode)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area-driven track slide input")
