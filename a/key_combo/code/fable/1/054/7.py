from __future__ import annotations

from collections import Counter, deque
import math
import random

import numpy as np


GREEN = 3
MAX_GENERATION_ATTEMPTS = 400
BG_CHOICES = [0, 1, 4, 5, 6, 7, 8, 9]
PITCH_CHOICES = [2, 3, 4]
TRACK_COUNT_RANGE = (2, 4)
SLOT_COUNT_RANGE = (4, 6)
STEP_COUNT_MAX = 6


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return int(round((r0 + r1) / 2)), int(round((c0 + c1) / 2))


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _build_track_positions(anchor_axis: int, green_axis_values: list[int]) -> tuple[list[int], int] | None:
    if len(green_axis_values) < 2:
        return None
    diffs = [abs(v - anchor_axis) for v in green_axis_values if v != anchor_axis]
    if not diffs:
        return None
    pitch = 0
    for diff in diffs:
        pitch = math.gcd(pitch, diff)
    if pitch <= 0:
        return None
    positions = sorted(green_axis_values + [anchor_axis])
    if len(positions) != len(set(positions)):
        return None
    if any(b - a != pitch for a, b in zip(positions, positions[1:])):
        return None
    return positions, pitch


def _direction_toward_far_end(index: int, n_slots: int) -> int:
    left_dist = index
    right_dist = n_slots - 1 - index
    assert left_dist != right_dist, "object may not start at the exact track midpoint"
    return 1 if right_dist > left_dist else -1


def _advance_with_bounce(index: int, n_slots: int, steps: int) -> int:
    pos = int(index)
    direction = _direction_toward_far_end(pos, int(n_slots))
    for _ in range(int(steps)):
        nxt = pos + direction
        if nxt < 0 or nxt >= n_slots:
            direction *= -1
            nxt = pos + direction
        pos = nxt
    return int(pos)


def _extract_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 3
    bg = _get_bg(g)
    palette_counts = Counter(int(v) for v in g[0] if int(v) != bg)

    body = g.copy()
    body[0, :] = bg
    green_points = {tuple(map(int, rc)) for rc in np.argwhere(body == GREEN)}
    components = [comp for comp in _find_monochrome_components(body, bg) if comp["color"] != GREEN]

    object_colors = [int(comp["color"]) for comp in components]
    assert components, "expected at least one movable object"
    assert len(object_colors) == len(set(object_colors)), "object colors must be unique"

    objects = []
    for comp in components:
        color = int(comp["color"])
        assert color in palette_counts, "every object color must appear in the palette row"
        anchor_r, anchor_c = _bbox_center(comp["bbox"])

        row_green = sorted(c for r, c in green_points if r == anchor_r)
        col_green = sorted(r for r, c in green_points if c == anchor_c)
        horiz = _build_track_positions(anchor_c, row_green)
        vert = _build_track_positions(anchor_r, col_green)

        candidates = []
        if horiz is not None:
            candidates.append(("horizontal", horiz))
        if vert is not None:
            candidates.append(("vertical", vert))
        assert len(candidates) == 1, "each object must align to exactly one track"

        orientation, (positions, pitch) = candidates[0]
        axis_value = anchor_c if orientation == "horizontal" else anchor_r
        slot_index = positions.index(axis_value)
        objects.append(
            {
                "color": color,
                "coords": comp["coords"],
                "anchor": (anchor_r, anchor_c),
                "orientation": orientation,
                "positions": positions,
                "pitch": int(pitch),
                "slot_index": int(slot_index),
                "steps": int(palette_counts[color]),
            }
        )

    return {
        "grid": g,
        "bg": bg,
        "palette_counts": palette_counts,
        "green_points": green_points,
        "objects": objects,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each palette-colored object along its own green-dot track by the number of palette occurrences.

    Input structure:
    - The top row is a palette row containing repeated colored pixels on the background.
    - Below it are several straight, non-intersecting tracks made of evenly spaced green dots.
    - Each track has exactly one non-green object resting on one slot, and object colors are unique.

    Rule:
    - Count how many times each object color appears in the top palette row.
    - Move that object along its track toward the farther endpoint by exactly that many slots.
    - If the object reaches the endpoint before using all steps, reverse direction and continue; then clear the palette row, redraw all green track dots, and place the moved objects.

    Key insight:
    - The top row is not decoration: its color multiplicities are the step counts, and the terminal slot is a bounce wall rather than a stopping point.
    """
    info = _extract_instance(np.asarray(grid, dtype=int))
    g = info["grid"]
    bg = info["bg"]
    out = np.full_like(g, bg)

    for r, c in info["green_points"]:
        out[r, c] = GREEN

    occupied_by_objects: set[tuple[int, int]] = set()
    for obj in info["objects"]:
        anchor_r, anchor_c = obj["anchor"]
        out[anchor_r, anchor_c] = GREEN

        final_index = _advance_with_bounce(obj["slot_index"], len(obj["positions"]), obj["steps"])
        if obj["orientation"] == "horizontal":
            target_anchor = (anchor_r, obj["positions"][final_index])
        else:
            target_anchor = (obj["positions"][final_index], anchor_c)

        offsets = obj["coords"] - np.array([[anchor_r, anchor_c]], dtype=int)
        target_coords = offsets + np.array([target_anchor], dtype=int)
        for rr, cc in target_coords:
            rr = int(rr)
            cc = int(cc)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "moved object must stay in bounds"
            assert (rr, cc) not in occupied_by_objects, "moved objects may not overlap"
            occupied_by_objects.add((rr, cc))
        for rr, cc in target_coords:
            out[int(rr), int(cc)] = int(obj["color"])

    return out


def _choose_side_index(n_slots: int, side: str) -> int:
    center = (n_slots - 1) / 2
    if side == "low":
        candidates = [i for i in range(n_slots) if i < center]
    else:
        candidates = [i for i in range(n_slots) if i > center]
    assert candidates
    return int(random.choice(candidates))


def _row_candidates(height: int, used_rows: list[int]) -> list[int]:
    return [r for r in range(1, height - 1) if all(abs(r - u) >= 2 for u in used_rows)]


def _col_candidates(width: int, used_cols: list[int], forbidden_cols: set[int]) -> list[int]:
    return [
        c
        for c in range(1, width - 1)
        if c not in forbidden_cols and all(abs(c - u) >= 2 for u in used_cols)
    ]


def _sample_horizontal_track(
    height: int,
    width: int,
    used_rows: list[int],
    forbidden_slot_cols: set[int],
) -> dict | None:
    rows = _row_candidates(height, used_rows)
    if not rows:
        return None
    for _ in range(100):
        n_slots = random.randint(*SLOT_COUNT_RANGE)
        pitch = random.choice(PITCH_CHOICES)
        max_start = width - 2 - pitch * (n_slots - 1)
        if max_start < 1:
            continue
        row = int(random.choice(rows))
        start = random.randint(1, max_start)
        cols = [start + pitch * i for i in range(n_slots)]
        if any(c in forbidden_slot_cols for c in cols):
            continue
        return {
            "orientation": "horizontal",
            "line": row,
            "positions": cols,
            "slots": [(row, c) for c in cols],
        }
    return None


def _sample_vertical_track(
    height: int,
    width: int,
    used_cols: list[int],
    forbidden_cols: set[int],
    forbidden_rows: set[int],
    forbidden_slot_rows: set[int],
) -> dict | None:
    cols = _col_candidates(width, used_cols, forbidden_cols)
    if not cols:
        return None
    for _ in range(120):
        n_slots = random.randint(*SLOT_COUNT_RANGE)
        pitch = random.choice(PITCH_CHOICES)
        max_start = height - 2 - pitch * (n_slots - 1)
        if max_start < 1:
            continue
        col = int(random.choice(cols))
        starts = []
        for start in range(1, max_start + 1):
            rows = [start + pitch * i for i in range(n_slots)]
            if any(r in forbidden_rows for r in rows):
                continue
            if any(r in forbidden_slot_rows for r in rows):
                continue
            starts.append(start)
        if not starts:
            continue
        start = int(random.choice(starts))
        rows = [start + pitch * i for i in range(n_slots)]
        return {
            "orientation": "vertical",
            "line": col,
            "positions": rows,
            "slots": [(r, col) for r in rows],
        }
    return None


def _assign_steps(tracks: list[dict]) -> bool:
    n_tracks = len(tracks)
    role_by_index = ["direct"] * n_tracks
    bounce_index = random.randrange(n_tracks)
    role_by_index[bounce_index] = "bounce"
    remaining = [i for i in range(n_tracks) if i != bounce_index]
    if remaining:
        keep_direct = random.choice(remaining)
        for i in remaining:
            if i == keep_direct:
                role_by_index[i] = "direct"
            else:
                role_by_index[i] = random.choice(["direct", "bounce"])

    order = list(range(n_tracks))
    random.shuffle(order)
    side_by_index = ["low"] * n_tracks
    side_by_index[order[0]] = "low"
    if n_tracks >= 2:
        side_by_index[order[1]] = "high"
    for i in order[2:]:
        side_by_index[i] = random.choice(["low", "high"])

    for i, track in enumerate(tracks):
        n_slots = len(track["positions"])
        side = side_by_index[i]
        obj_index = _choose_side_index(n_slots, side)
        far_dist = max(obj_index, n_slots - 1 - obj_index)
        if role_by_index[i] == "bounce":
            lo = far_dist + 1
            hi = min(STEP_COUNT_MAX, far_dist + n_slots - 1)
        else:
            lo = 1
            hi = min(STEP_COUNT_MAX, far_dist)
        if lo > hi:
            return False
        track["object_index"] = int(obj_index)
        track["steps"] = int(random.randint(lo, hi))
    return True


def _paint_palette(grid: np.ndarray, color_steps: dict[int, int], bg: int) -> bool:
    items = list(color_steps.items())
    random.shuffle(items)
    min_required = sum(count for _color, count in items) + max(0, len(items) - 1)
    width = grid.shape[1]
    if min_required > width:
        return False

    gaps = [0] + [1] * max(0, len(items) - 1) + [0]
    extra = width - min_required
    for _ in range(extra):
        gaps[random.randrange(len(gaps))] += 1

    pos = gaps[0]
    for i, (color, count) in enumerate(items):
        grid[0, pos : pos + count] = int(color)
        pos += count
        if i < len(items) - 1:
            pos += gaps[i + 1]
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 30)
    width = random.randint(24, 30)
    n_tracks = random.randint(*TRACK_COUNT_RANGE)
    mode = random.choices(["all_h", "all_v", "mixed"], weights=[3, 3, 4], k=1)[0]

    if mode == "all_h":
        n_h = n_tracks
        n_v = 0
    elif mode == "all_v":
        n_h = 0
        n_v = n_tracks
    else:
        n_h = random.randint(1, n_tracks - 1)
        n_v = n_tracks - n_h

    bg = int(random.choice(BG_CHOICES))
    color_pool = [c for c in range(10) if c not in (bg, GREEN)]
    if len(color_pool) < n_tracks:
        return None
    obj_colors = random.sample(color_pool, n_tracks)

    tracks: list[dict] = []
    horizontal_rows: list[int] = []
    horizontal_slot_cols: set[int] = set()
    for _ in range(n_h):
        track = _sample_horizontal_track(height, width, horizontal_rows, horizontal_slot_cols)
        if track is None:
            return None
        tracks.append(track)
        horizontal_rows.append(track["line"])
        horizontal_slot_cols.update(track["positions"])

    vertical_cols: list[int] = []
    vertical_slot_rows: set[int] = set()
    for _ in range(n_v):
        track = _sample_vertical_track(
            height,
            width,
            vertical_cols,
            horizontal_slot_cols,
            set(horizontal_rows),
            vertical_slot_rows,
        )
        if track is None:
            return None
        tracks.append(track)
        vertical_cols.append(track["line"])
        vertical_slot_rows.update(track["positions"])

    random.shuffle(tracks)
    if not _assign_steps(tracks):
        return None

    color_steps: dict[int, int] = {}
    for color, track in zip(obj_colors, tracks):
        track["color"] = int(color)
        color_steps[int(color)] = int(track["steps"])

    grid = np.full((height, width), bg, dtype=int)
    if not _paint_palette(grid, color_steps, bg):
        return None

    for track in tracks:
        slots = track["slots"]
        obj_slot = slots[track["object_index"]]
        for r, c in slots:
            grid[r, c] = GREEN
        grid[obj_slot] = int(track["color"])

    parsed = _extract_instance(grid)
    if len(parsed["objects"]) != n_tracks:
        return None
    bounce_flags = []
    direct_flags = []
    orientations = set()
    for obj in parsed["objects"]:
        n_slots = len(obj["positions"])
        far_dist = max(obj["slot_index"], n_slots - 1 - obj["slot_index"])
        bounced = obj["steps"] > far_dist
        bounce_flags.append(bounced)
        direct_flags.append(not bounced)
        orientations.add(obj["orientation"])
    if not any(bounce_flags):
        return None
    if n_tracks > 1 and not any(direct_flags):
        return None
    if mode == "mixed" and orientations != {"horizontal", "vertical"}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
