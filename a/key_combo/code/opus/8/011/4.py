import random
from collections import Counter

import numpy as np


BG = 0
OBJECT_COLORS = (1, 2, 3, 4)
TRACK_COLOR = 5
NOISE_COLOR = 7

BASE_SHAPES = (
    ((1, 0), (2, 0)),
    ((1, 0), (2, 0), (2, 1)),
    ((1, 0), (2, 0), (2, -1)),
    ((1, 0), (2, 0), (3, 0)),
    ((1, 0), (2, 0), (3, 0), (2, 1)),
    ((1, 0), (2, 0), (3, 0), (2, -1)),
)
END_SAFE_SHAPES = (
    ((1, 0), (2, 0)),
    ((1, 0), (2, 0), (3, 0)),
)

MAX_GENERATION_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _find_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    allowed = set(int(color) for color in colors)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color not in allowed:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _detect_track(grid: np.ndarray) -> dict:
    coords = np.argwhere(grid == TRACK_COLOR)
    assert coords.size > 0, "expected a grey-dot track"
    rows = sorted(set(int(r) for r in coords[:, 0]))
    cols = sorted(set(int(c) for c in coords[:, 1]))
    if len(rows) == 1:
        orientation = "h"
        fixed = rows[0]
        slots = sorted(int(c) for c in coords[:, 1])
    else:
        assert len(cols) == 1, "track must be horizontal or vertical"
        orientation = "v"
        fixed = cols[0]
        slots = sorted(int(r) for r in coords[:, 0])
    assert len(slots) >= 2, "track needs multiple slots"
    diffs = [b - a for a, b in zip(slots, slots[1:])]
    assert len(set(diffs)) == 1, "track slots must be equally spaced"
    return {
        "orientation": orientation,
        "fixed": fixed,
        "slots": slots,
        "step": diffs[0],
        "slot_to_index": {value: idx for idx, value in enumerate(slots)},
    }


def _slot_anchor(component: dict, track: dict) -> int:
    coords = component["coords"]
    if track["orientation"] == "h":
        dists = np.abs(coords[:, 0] - int(track["fixed"]))
        nearest = coords[dists == int(dists.min())]
        return int(round(float(nearest[:, 1].mean())))
    dists = np.abs(coords[:, 1] - int(track["fixed"]))
    nearest = coords[dists == int(dists.min())]
    return int(round(float(nearest[:, 0].mean())))


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    track = _detect_track(g)
    objects = _find_components(g, OBJECT_COLORS)
    assert 4 <= len(objects) <= 5, "expected 4-5 track objects"

    slot_to_object: dict[int, int] = {}
    for idx, obj in enumerate(objects):
        assert 2 <= int(obj["area"]) <= 4, "objects must have 2-4 pixels"
        anchor = _slot_anchor(obj, track)
        assert anchor in track["slot_to_index"], "object must align with a track slot"
        slot_idx = int(track["slot_to_index"][anchor])
        assert slot_idx not in slot_to_object, "each slot may hold at most one object"
        slot_to_object[slot_idx] = idx
        obj["slot_idx"] = slot_idx

    colors = [int(obj["color"]) for obj in objects]
    freq = Counter(colors)
    assert any(count > 1 for count in freq.values()), "need repeated colors"
    assert any(count == 1 for count in freq.values()), "need unique colors"

    for obj in objects:
        repeated = int(freq[int(obj["color"])]) > 1
        obj["repeated"] = repeated
        obj["target_idx"] = int(obj["slot_idx"]) - 1 if repeated else int(obj["slot_idx"]) + 1

    proposals: dict[int, list[int]] = {}
    for idx, obj in enumerate(objects):
        target = int(obj["target_idx"])
        if 0 <= target < len(track["slots"]):
            proposals.setdefault(target, []).append(idx)

    locked = {idx for idx, obj in enumerate(objects) if not (0 <= int(obj["target_idx"]) < len(track["slots"]))}
    locked.update(
        idx
        for target, indices in proposals.items()
        if len(indices) > 1
        for idx in indices
    )

    changed = True
    while changed:
        changed = False
        occupied_by_locked = {int(objects[idx]["slot_idx"]) for idx in locked}
        for idx, obj in enumerate(objects):
            if idx in locked:
                continue
            if int(obj["target_idx"]) in occupied_by_locked:
                locked.add(idx)
                changed = True

    occupancy = ["." for _ in track["slots"]]
    for obj in objects:
        occupancy[int(obj["slot_idx"])] = "R" if obj["repeated"] else "U"

    for obj in objects:
        obj["moves"] = False
        obj["final_idx"] = int(obj["slot_idx"])
    for idx, obj in enumerate(objects):
        if idx in locked:
            continue
        obj["moves"] = True
        obj["final_idx"] = int(obj["target_idx"])

    return {
        "track": track,
        "objects": objects,
        "occupancy": "".join(occupancy),
        "color_freq": freq,
        "locked": locked,
        "proposals": proposals,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide repeated-color objects one slot toward the track start and unique-color objects one slot toward the track end.

    Input structure:
    - The background is black, with a straight horizontal or vertical grey dotted track made of equally spaced single pixels.
    - Each occupied slot has one small 4-connected monochrome object in red, blue, green, or yellow; the grey dots stay visible and orange single pixels are distractors.
    - The track has 4-5 such objects plus at least one empty slot at each end.

    Rule:
    - Ignore the orange distractors and count color frequencies only among the track objects.
    - Objects whose color appears more than once target the previous slot toward the track start; objects whose color appears once target the next slot toward the track end.
    - Compute all target slots from the initial state, cancel moves that go out of range or that share a target slot, then keep cancelling any move whose target stays occupied by a cancelled object.
    - Redraw the surviving moves simultaneously; the grey track dots and orange distractors remain unchanged.

    Key insight:
    - Direction is encoded by frequency, not by position: same-frequency objects move the same way even when they sit on opposite sides of the track.
    """
    analysis = _analyze(grid)
    out = np.asarray(grid, dtype=int).copy()
    for obj in analysis["objects"]:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    track = analysis["track"]
    for obj in analysis["objects"]:
        delta_slots = int(obj["final_idx"]) - int(obj["slot_idx"])
        if track["orientation"] == "h":
            dr, dc = 0, delta_slots * int(track["step"])
        else:
            dr, dc = delta_slots * int(track["step"]), 0
        moved = obj["coords"] + np.array([dr, dc], dtype=int)
        out[moved[:, 0], moved[:, 1]] = int(obj["color"])
    return out


def _pattern_move_info(pattern: str) -> tuple[set[int], dict[int, list[int]]]:
    targets: dict[int, list[int]] = {}
    locked = set()
    for idx, ch in enumerate(pattern):
        if ch == ".":
            continue
        target = idx - 1 if ch == "R" else idx + 1
        if not (0 <= target < len(pattern)):
            locked.add(idx)
            continue
        targets.setdefault(target, []).append(idx)
    for indices in targets.values():
        if len(indices) > 1:
            locked.update(indices)
    changed = True
    while changed:
        changed = False
        occupied_by_locked = {idx for idx, ch in enumerate(pattern) if ch != "." and idx in locked}
        for target, indices in targets.items():
            if target not in occupied_by_locked:
                continue
            for idx in indices:
                if idx not in locked:
                    locked.add(idx)
                    changed = True
    return locked, targets


def _pattern_ok(pattern: str) -> bool:
    n = len(pattern)
    if pattern[0] != "." or pattern[-1] != ".":
        return False
    if not (4 <= sum(ch != "." for ch in pattern) <= 5):
        return False
    if pattern.count("R") < 2 or pattern.count("U") < 1:
        return False
    if not ("R" in pattern and "U" in pattern):
        return False
    if not (pattern.index("R") < pattern.rindex("U") and pattern.index("U") < pattern.rindex("R")):
        return False

    locked, targets = _pattern_move_info(pattern)
    success_r = 0
    success_u = 0
    for target, indices in targets.items():
        if len(indices) == 1 and indices[0] not in locked:
            if pattern[indices[0]] == "R":
                success_r += 1
            else:
                success_u += 1
    return success_r >= 1 and success_u >= 1


def _all_patterns(n_slots: int) -> list[str]:
    states = []
    symbols = (".", "R", "U")
    for code in range(3 ** n_slots):
        x = code
        chars = []
        for _ in range(n_slots):
            x, rem = divmod(x, 3)
            chars.append(symbols[rem])
        pattern = "".join(chars)
        if _pattern_ok(pattern):
            states.append(pattern)
    return states


PATTERNS_BY_SIZE = {n: _all_patterns(n) for n in (7, 8)}


def _pattern_has_conflict(pattern: str) -> bool:
    _locked, targets = _pattern_move_info(pattern)
    return any(len(indices) > 1 for indices in targets.values())


def _choose_pattern() -> str:
    n_slots = random.choice((7, 8))
    patterns = PATTERNS_BY_SIZE[n_slots]
    want_conflict = random.random() < 0.45
    subset = [pattern for pattern in patterns if _pattern_has_conflict(pattern) == want_conflict]
    if not subset:
        subset = patterns
    return random.choice(subset)


def _repeated_partitions(n_repeated: int) -> list[tuple[int, ...]]:
    options = {
        2: [(2,)],
        3: [(3,)],
        4: [(4,), (2, 2)],
        5: [(5,), (3, 2)],
    }
    return options.get(n_repeated, [])


def _assign_colors(pattern: str) -> dict[int, int]:
    repeated_positions = [idx for idx, ch in enumerate(pattern) if ch == "R"]
    unique_positions = [idx for idx, ch in enumerate(pattern) if ch == "U"]
    partitions = [
        part
        for part in _repeated_partitions(len(repeated_positions))
        if len(part) + len(unique_positions) <= len(OBJECT_COLORS)
    ]
    assert partitions, "repeated/unique counts must fit the palette"
    repeated_partition = random.choice(partitions)
    palette = random.sample(list(OBJECT_COLORS), len(repeated_partition) + len(unique_positions))
    repeated_colors = palette[: len(repeated_partition)]
    unique_colors = palette[len(repeated_partition) :]

    repeated_assignment = []
    for color, size in zip(repeated_colors, repeated_partition):
        repeated_assignment.extend([color] * size)
    random.shuffle(repeated_assignment)
    random.shuffle(unique_colors)

    out = {}
    for pos, color in zip(repeated_positions, repeated_assignment):
        out[pos] = int(color)
    for pos, color in zip(unique_positions, unique_colors):
        out[pos] = int(color)
    return out


def _successful_endpoint_movers(pattern: str) -> set[int]:
    locked, targets = _pattern_move_info(pattern)
    endpoint_movers = set()
    last = len(pattern) - 1
    for target, indices in targets.items():
        if len(indices) == 1 and indices[0] not in locked and target in {0, last}:
            endpoint_movers.add(indices[0])
    return endpoint_movers


def _orient_shape(base_shape: tuple[tuple[int, int], ...], orientation: str, side: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(side * r, c) for r, c in base_shape]
    return [(c, side * r) for r, c in base_shape]


def _place_track_and_objects(pattern: str) -> np.ndarray | None:
    orientation = random.choice(("h", "v"))
    n_slots = len(pattern)
    step = 4
    span = step * (n_slots - 1)

    if orientation == "h":
        width = random.randint(span + 2, 30)
        start = random.randint(1, width - 1 - span)
        height = random.randint(11, 16)
        track_line = random.randint(3, height - 4)
        grid = np.full((height, width), BG, dtype=int)
        slots = [start + step * idx for idx in range(n_slots)]
        for col in slots:
            grid[track_line, col] = TRACK_COLOR
    else:
        height = random.randint(span + 2, 30)
        start = random.randint(1, height - 1 - span)
        width = random.randint(11, 16)
        track_line = random.randint(3, width - 4)
        grid = np.full((height, width), BG, dtype=int)
        slots = [start + step * idx for idx in range(n_slots)]
        for row in slots:
            grid[row, track_line] = TRACK_COLOR

    colors_by_slot = _assign_colors(pattern)
    endpoint_movers = _successful_endpoint_movers(pattern)
    for slot_idx, marker in enumerate(pattern):
        if marker == ".":
            continue
        color = colors_by_slot[slot_idx]
        shape_pool = END_SAFE_SHAPES if slot_idx in endpoint_movers else BASE_SHAPES
        base_shape = random.choice(shape_pool)
        side = random.choice((-1, 1))
        rel = _orient_shape(base_shape, orientation, side)
        coords = []
        if orientation == "h":
            anchor_r, anchor_c = track_line, slots[slot_idx]
        else:
            anchor_r, anchor_c = slots[slot_idx], track_line
        for dr, dc in rel:
            rr = anchor_r + dr
            cc = anchor_c + dc
            if not _in_bounds(grid.shape, rr, cc) or int(grid[rr, cc]) != BG:
                return None
            coords.append((rr, cc))
        for rr, cc in coords:
            grid[rr, cc] = color

    if len(_find_components(grid, OBJECT_COLORS)) != sum(ch != "." for ch in pattern):
        return None
    return grid


def _scatter_noise(grid: np.ndarray) -> bool:
    track = _detect_track(grid)
    forbidden = grid != BG
    h, w = grid.shape
    candidates = []
    for r in range(h):
        for c in range(w):
            if forbidden[r, c]:
                continue
            if track["orientation"] == "h" and abs(r - int(track["fixed"])) < 2:
                continue
            if track["orientation"] == "v" and abs(c - int(track["fixed"])) < 2:
                continue
            ok = True
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if _in_bounds(grid.shape, nr, nc) and forbidden[nr, nc]:
                    ok = False
                    break
            if ok:
                candidates.append((r, c))
    if len(candidates) < 3:
        return False
    for r, c in random.sample(candidates, random.randint(3, min(7, len(candidates)))):
        grid[r, c] = NOISE_COLOR
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
        track = analysis["track"]
        objects = analysis["objects"]
        slots_used = {int(obj["slot_idx"]) for obj in objects}
        if 0 in slots_used or len(track["slots"]) - 1 in slots_used:
            return False
        if not any(obj["repeated"] and obj["moves"] for obj in objects):
            return False
        if not any((not obj["repeated"]) and obj["moves"] for obj in objects):
            return False
        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if not np.array_equal(out[grid == TRACK_COLOR], grid[grid == TRACK_COLOR]):
            return False
        if not np.array_equal(out[grid == NOISE_COLOR], grid[grid == NOISE_COLOR]):
            return False
        if len(_find_components(grid, OBJECT_COLORS)) != len(_find_components(out, OBJECT_COLORS)):
            return False
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pattern = _choose_pattern()
        grid = _place_track_and_objects(pattern)
        if grid is None:
            continue
        if not _scatter_noise(grid):
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid frequency-track puzzle")
