from __future__ import annotations

import random

import numpy as np


BG = 0
SIDE_TO_OUT = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OUT_TO_SIDE = {value: key for key, value in SIDE_TO_OUT.items()}
OPPOSITE = {
    (-1, 0): (1, 0),
    (1, 0): (-1, 0),
    (0, -1): (0, 1),
    (0, 1): (0, -1),
}


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    return mask.astype(int).cumsum(axis=0).cumsum(axis=1)


def _rect_sum(ps: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    if r0 > r1 or c0 > c1:
        return 0
    out = int(ps[r1, c1])
    if r0 > 0:
        out -= int(ps[r0 - 1, c1])
    if c0 > 0:
        out -= int(ps[r1, c0 - 1])
    if r0 > 0 and c0 > 0:
        out += int(ps[r0 - 1, c0 - 1])
    return out


def _side_offsets(height: int, width: int, side: str) -> list[int]:
    limit = width if side in {"top", "bottom"} else height
    return list(range(1, limit - 1))


def _gap_coord(r0: int, c0: int, height: int, width: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return int(r0), int(c0 + offset)
    if side == "bottom":
        return int(r0 + height - 1), int(c0 + offset)
    if side == "left":
        return int(r0 + offset), int(c0)
    if side == "right":
        return int(r0 + offset), int(c0 + width - 1)
    raise ValueError("invalid side")


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c0, c1 + 1):
            coords.append((r1, c))
    return coords


def _build_frame(
    r0: int,
    c0: int,
    height: int,
    width: int,
    gap_specs: list[tuple[str, int]],
) -> dict:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    center = (r0 + height // 2, c0 + width // 2)
    gaps = []
    gap_cells = set()
    for side, offset in gap_specs:
        coord = _gap_coord(r0, c0, height, width, side, offset)
        gap_cells.add(coord)
        out_dir = SIDE_TO_OUT[side]
        gaps.append(
            {
                "coord": coord,
                "side": side,
                "offset": int(offset),
                "out_dir": out_dir,
                "in_dir": OPPOSITE[out_dir],
            }
        )

    wall = set(_border_coords(r0, c0, r1, c1)) - gap_cells
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "center": center,
        "gaps": gaps,
        "wall": wall,
    }


def _frame_signature(frame: dict) -> tuple:
    return (
        tuple(map(int, frame["bbox"])),
        tuple(sorted(tuple(map(int, gap["coord"])) for gap in frame["gaps"])),
    )


def _build_maps(frames: list[dict]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]]]:
    wall_cells: set[tuple[int, int]] = set()
    gap_entries: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]] = {}
    for frame_id, frame in enumerate(frames):
        wall_cells.update(frame["wall"])
        for gap_id, gap in enumerate(frame["gaps"]):
            gap_entries.setdefault(gap["coord"], []).append((frame_id, gap_id, gap["in_dir"]))
    return wall_cells, gap_entries


def _cast_ray(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    gap_entries: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]],
    start_gap: tuple[int, int],
    direction: tuple[int, int],
) -> tuple:
    height, width = map(int, shape)
    dr, dc = map(int, direction)
    r = int(start_gap[0] + dr)
    c = int(start_gap[1] + dc)
    last = (int(start_gap[0]), int(start_gap[1]))
    while 0 <= r < height and 0 <= c < width:
        for frame_id, gap_id, in_dir in gap_entries.get((r, c), []):
            if in_dir == direction:
                return ("enter", int(frame_id), int(gap_id))
        if (r, c) in wall_cells:
            return ("wall", last)
        last = (int(r), int(c))
        r += dr
        c += dc
    return ("edge", last)


def _trace_message(shape: tuple[int, int], frames: list[dict], start_id: int) -> dict:
    wall_cells, gap_entries = _build_maps(frames)
    start_frame = frames[int(start_id)]
    assert len(start_frame["gaps"]) == 1, "start frame must have exactly one gap"

    visited = {int(start_id)}
    transit_ids: list[int] = []
    current_gap = start_frame["gaps"][0]["coord"]
    direction = start_frame["gaps"][0]["out_dir"]

    while True:
        event = _cast_ray(shape, wall_cells, gap_entries, current_gap, direction)
        kind = event[0]
        if kind == "wall":
            return {
                "final": event[1],
                "transit_ids": transit_ids,
                "stop_kind": "wall",
            }
        if kind == "edge":
            return {
                "final": event[1],
                "transit_ids": transit_ids,
                "stop_kind": "edge",
            }

        _, frame_id, gap_id = event
        assert frame_id not in visited, "pipe path must be acyclic"
        visited.add(frame_id)
        frame = frames[frame_id]
        if len(frame["gaps"]) == 1:
            return {
                "final": frame["center"],
                "transit_ids": transit_ids,
                "stop_kind": "endpoint",
                "stop_frame_id": int(frame_id),
            }

        assert len(frame["gaps"]) == 2, "every non-endpoint frame must have exactly two gaps"
        transit_ids.append(int(frame_id))
        exit_id = 1 - int(gap_id)
        current_gap = frame["gaps"][exit_id]["coord"]
        direction = frame["gaps"][exit_id]["out_dir"]


def _detect_colors(grid: np.ndarray) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    assert len(colors) >= 2, "expected one wall color and one letter color"
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    wall_color = max(colors, key=lambda color: (counts[color], -color))
    letter_cells = np.argwhere((grid != BG) & (grid != wall_color))
    assert letter_cells.shape[0] == 1, "expected exactly one colored letter pixel"
    letter_color = int(grid[tuple(letter_cells[0])])
    return int(wall_color), int(letter_color)


def _enumerate_frame_candidates(grid: np.ndarray, wall_color: int) -> list[dict]:
    height, width = map(int, grid.shape)
    wall_mask = grid == int(wall_color)
    nonblack_mask = grid != BG
    wall_ps = _prefix_sum(wall_mask)
    nonblack_ps = _prefix_sum(nonblack_mask)
    candidates = []

    for frame_h in range(3, height + 1, 2):
        for frame_w in range(3, width + 1, 2):
            perimeter = 2 * (frame_h + frame_w) - 4
            for r0 in range(height - frame_h + 1):
                r1 = r0 + frame_h - 1
                for c0 in range(width - frame_w + 1):
                    c1 = c0 + frame_w - 1
                    total_wall = _rect_sum(wall_ps, r0, c0, r1, c1)
                    inner_wall = _rect_sum(wall_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if inner_wall != 0:
                        continue

                    total_nonblack = _rect_sum(nonblack_ps, r0, c0, r1, c1)
                    inner_nonblack = _rect_sum(nonblack_ps, r0 + 1, c0 + 1, r1 - 1, c1 - 1)
                    if inner_nonblack not in {0, 1}:
                        continue

                    border_wall = total_wall
                    gaps = perimeter - border_wall
                    if gaps not in {1, 2}:
                        continue
                    if total_nonblack - inner_nonblack != border_wall:
                        continue

                    center = (r0 + frame_h // 2, c0 + frame_w // 2)
                    center_value = int(grid[center])
                    if inner_nonblack == 1 and (center_value == BG or center_value == wall_color):
                        continue
                    if inner_nonblack == 0 and center_value != BG:
                        continue

                    gap_specs = []
                    valid = True
                    for r, c in _border_coords(r0, c0, r1, c1):
                        value = int(grid[r, c])
                        if value == wall_color:
                            continue
                        if value != BG:
                            valid = False
                            break
                        if (r in {r0, r1}) and (c in {c0, c1}):
                            valid = False
                            break
                        if r == r0:
                            gap_specs.append(("top", c - c0))
                        elif r == r1:
                            gap_specs.append(("bottom", c - c0))
                        elif c == c0:
                            gap_specs.append(("left", r - r0))
                        else:
                            gap_specs.append(("right", r - r0))
                    if not valid or len(gap_specs) != gaps:
                        continue

                    frame = _build_frame(r0, c0, frame_h, frame_w, gap_specs)
                    frame["center_has_letter"] = center_value not in {BG, wall_color}
                    frame["wall_set"] = frozenset(r * width + c for r, c in frame["wall"])
                    candidates.append(frame)
    return candidates


def _solve_exact_cover(candidates: list[dict], wall_mask: np.ndarray) -> list[dict]:
    width = wall_mask.shape[1]
    universe = {int(r * width + c) for r, c in np.argwhere(wall_mask)}
    by_cell: dict[int, list[int]] = {cell: [] for cell in universe}
    usable = []
    for candidate in candidates:
        if candidate["wall_set"] and candidate["wall_set"].issubset(universe):
            usable.append(candidate)
    for idx, candidate in enumerate(usable):
        for cell in candidate["wall_set"]:
            by_cell[cell].append(idx)

    solutions: list[list[int]] = []

    def backtrack(remaining: set[int], chosen: list[int]) -> None:
        if not remaining:
            solutions.append(chosen.copy())
            return
        pivot = min(
            remaining,
            key=lambda cell: sum(1 for idx in by_cell[cell] if usable[idx]["wall_set"].issubset(remaining)),
        )
        options = [idx for idx in by_cell[pivot] if usable[idx]["wall_set"].issubset(remaining)]
        if not options:
            return
        options.sort(key=lambda idx: -len(usable[idx]["wall_set"]))
        for idx in options:
            chosen.append(idx)
            backtrack(remaining - set(usable[idx]["wall_set"]), chosen)
            chosen.pop()
            if len(solutions) > 1:
                return

    backtrack(set(universe), [])
    assert solutions, "failed to cover all wall pixels with frames"
    assert len(solutions) == 1, "frame cover must be unique"
    frames = [usable[idx] for idx in solutions[0]]
    frames.sort(key=_frame_signature)
    return frames


def _parse_frames(grid: np.ndarray) -> tuple[int, int, list[dict], int]:
    wall_color, letter_color = _detect_colors(grid)
    candidates = _enumerate_frame_candidates(grid, wall_color)
    frames = _solve_exact_cover(candidates, grid == wall_color)
    start_ids = [idx for idx, frame in enumerate(frames) if frame.get("center_has_letter", False)]
    assert len(start_ids) == 1, "expected exactly one start frame"
    start_id = int(start_ids[0])
    assert len(frames[start_id]["gaps"]) == 1, "the letter must start in a one-gap endpoint frame"
    return wall_color, letter_color, frames, start_id


def _paint_output(
    grid: np.ndarray,
    frames: list[dict],
    start_id: int,
    transit_ids: list[int],
    final_pos: tuple[int, int],
    letter_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[frames[int(start_id)]["center"]] = BG
    for frame_id in transit_ids:
        out[frames[int(frame_id)]["center"]] = int(letter_color)
    out[tuple(final_pos)] = int(letter_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the hidden letter through hollow frames and stamp each relay frame center.

    Input structure:
    - Black is the background.
    - Several same-colored hollow odd-sized rectangular frames are scattered apart; each frame has either one single-cell gap or two single-cell gaps on its border.
    - Exactly one one-gap endpoint frame contains a single colored letter pixel at its center; every other frame interior is empty.

    Rule:
    - Identify every hollow frame together with its gap cells.
    - Launch the letter out of the start frame through its only gap and follow the outward row or column ray.
    - On each ray, the first obstacle wins: an enterable gap cell lets the letter enter that frame, a wall cell stops it on the adjacent outside cell, and an unobstructed ray stops on the last grid-edge cell.
    - Entering a two-gap frame stamps its center with the letter color and immediately re-emits the letter through the other gap; entering a one-gap endpoint frame ends the route at that frame center.
    - Preserve all original frames and gaps, erase the start letter from its source center, and add only the relay stamps plus the final resting letter pixel.

    Key insight:
    - The gaps define an invisible directed pipe network, but every hop still obeys a first-obstacle ray rule, so solving the task requires both frame/gap parsing and ordered ray tracing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _wall_color, letter_color, frames, start_id = _parse_frames(g)
    trace = _trace_message(g.shape, frames, start_id)
    return _paint_output(g, frames, start_id, trace["transit_ids"], trace["final"], letter_color)


GEN_ATTEMPTS = 500
FRAME_SIZE_CHOICES = (3, 5, 7)
GRID_HEIGHT_RANGE = (18, 29)
GRID_WIDTH_RANGE = (18, 29)
TRANSIT_COUNT_RANGE = (1, 3)
EXTRA_FRAME_COUNT_RANGE = (0, 2)
PLACEMENT_ATTEMPTS = 200


def _sample_dims(min_size: int = 3) -> tuple[int, int]:
    choices = [size for size in FRAME_SIZE_CHOICES if size >= int(min_size)]
    return int(random.choice(choices)), int(random.choice(choices))


def _bbox_clear(bbox: tuple[int, int, int, int], frames: list[dict], margin: int = 1) -> bool:
    r0, c0, r1, c1 = map(int, bbox)
    for frame in frames:
        a0, b0, a1, b1 = map(int, frame["bbox"])
        separated = (
            r1 + margin < a0
            or a1 + margin < r0
            or c1 + margin < b0
            or b1 + margin < c0
        )
        if not separated:
            return False
    return True


def _distance_to_edge(coord: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> int:
    height, width = map(int, shape)
    r, c = map(int, coord)
    dr, dc = map(int, direction)
    count = 0
    r += dr
    c += dc
    while 0 <= r < height and 0 <= c < width:
        count += 1
        r += dr
        c += dc
    return int(count)


def _segment_clear(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    gap_entries: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]],
    start_gap: tuple[int, int],
    direction: tuple[int, int],
    stop_coord: tuple[int, int],
) -> bool:
    dr, dc = map(int, direction)
    r = int(start_gap[0] + dr)
    c = int(start_gap[1] + dc)
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        if (r, c) == stop_coord:
            return True
        if (r, c) in wall_cells:
            return False
        for _frame_id, _gap_id, in_dir in gap_entries.get((r, c), []):
            if in_dir == direction:
                return False
        r += dr
        c += dc
    return False


def _ray_clear_to_edge(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    gap_entries: dict[tuple[int, int], list[tuple[int, int, tuple[int, int]]]],
    start_gap: tuple[int, int],
    direction: tuple[int, int],
) -> bool:
    dr, dc = map(int, direction)
    r = int(start_gap[0] + dr)
    c = int(start_gap[1] + dc)
    seen = False
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        seen = True
        if (r, c) in wall_cells:
            return False
        for _frame_id, _gap_id, in_dir in gap_entries.get((r, c), []):
            if in_dir == direction:
                return False
        r += dr
        c += dc
    return seen


def _edge_endpoint(start_gap: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    dr, dc = map(int, direction)
    r = int(start_gap[0])
    c = int(start_gap[1])
    last = (r, c)
    while True:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
            return last
        last = (nr, nc)
        r, c = nr, nc


def _place_start_frame(shape: tuple[int, int], frames: list[dict]) -> dict | None:
    height, width = map(int, shape)
    for _ in range(PLACEMENT_ATTEMPTS):
        frame_h, frame_w = _sample_dims()
        side = random.choice(list(SIDE_TO_OUT))
        offset = random.choice(_side_offsets(frame_h, frame_w, side))
        r0 = random.randint(1, height - frame_h - 1)
        c0 = random.randint(1, width - frame_w - 1)
        frame = _build_frame(r0, c0, frame_h, frame_w, [(side, offset)])
        if not _bbox_clear(frame["bbox"], frames):
            continue
        if _distance_to_edge(frame["gaps"][0]["coord"], frame["gaps"][0]["out_dir"], shape) < 4:
            continue
        return frame
    return None


def _sample_aligned_bbox(
    shape: tuple[int, int],
    current_gap: tuple[int, int],
    direction: tuple[int, int],
    frame_h: int,
    frame_w: int,
    offset: int,
) -> tuple[int, int] | None:
    height, width = map(int, shape)
    row, col = map(int, current_gap)
    if direction == (0, 1):
        r0 = row - offset
        if not (1 <= r0 <= height - frame_h - 1):
            return None
        c_min = col + 2
        c_max = width - frame_w - 1
        if c_min > c_max:
            return None
        return int(r0), int(random.randint(c_min, c_max))
    if direction == (0, -1):
        r0 = row - offset
        if not (1 <= r0 <= height - frame_h - 1):
            return None
        c_min = 1
        c_max = col - frame_w - 1
        if c_min > c_max:
            return None
        return int(r0), int(random.randint(c_min, c_max))
    if direction == (1, 0):
        c0 = col - offset
        if not (1 <= c0 <= width - frame_w - 1):
            return None
        r_min = row + 2
        r_max = height - frame_h - 1
        if r_min > r_max:
            return None
        return int(random.randint(r_min, r_max)), int(c0)
    if direction == (-1, 0):
        c0 = col - offset
        if not (1 <= c0 <= width - frame_w - 1):
            return None
        r_min = 1
        r_max = row - frame_h - 1
        if r_min > r_max:
            return None
        return int(random.randint(r_min, r_max)), int(c0)
    return None


def _choose_exit_side(entry_side: str) -> str:
    choices = []
    for side in SIDE_TO_OUT:
        if side == entry_side:
            continue
        if {
            frozenset({"left", "right"}),
            frozenset({"top", "bottom"}),
        } and frozenset({entry_side, side}) in {frozenset({"left", "right"}), frozenset({"top", "bottom"})}:
            choices.append(side)
        else:
            choices.extend([side, side])
    return random.choice(choices)


def _place_target_frame(
    shape: tuple[int, int],
    frames: list[dict],
    current_gap: tuple[int, int],
    direction: tuple[int, int],
    kind: str,
    *,
    require_clear: bool,
) -> dict | None:
    wall_cells, gap_entries = _build_maps(frames)
    entry_side = OUT_TO_SIDE[OPPOSITE[direction]]
    for _ in range(PLACEMENT_ATTEMPTS):
        frame_h, frame_w = _sample_dims(5 if kind == "transit" else 3)
        entry_offset = random.choice(_side_offsets(frame_h, frame_w, entry_side))
        top_left = _sample_aligned_bbox(shape, current_gap, direction, frame_h, frame_w, entry_offset)
        if top_left is None:
            continue
        r0, c0 = top_left
        gap_specs = [(entry_side, entry_offset)]
        if kind == "transit":
            exit_side = _choose_exit_side(entry_side)
            exit_offset = random.choice(_side_offsets(frame_h, frame_w, exit_side))
            gap_specs.append((exit_side, exit_offset))
        frame = _build_frame(r0, c0, frame_h, frame_w, gap_specs)
        if not _bbox_clear(frame["bbox"], frames):
            continue
        if require_clear and not _segment_clear(
            shape,
            wall_cells,
            gap_entries,
            current_gap,
            direction,
            frame["gaps"][0]["coord"],
        ):
            continue
        return frame
    return None


def _all_gap_positions(frame_h: int, frame_w: int) -> list[tuple[str, int]]:
    positions = []
    for side in SIDE_TO_OUT:
        for offset in _side_offsets(frame_h, frame_w, side):
            positions.append((side, offset))
    return positions


def _place_blocker_frame(
    shape: tuple[int, int],
    frames: list[dict],
    current_gap: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[dict, tuple[int, int]] | None:
    wall_cells, gap_entries = _build_maps(frames)
    hit_side = OUT_TO_SIDE[OPPOSITE[direction]]
    for _ in range(PLACEMENT_ATTEMPTS):
        frame_h, frame_w = _sample_dims(5)
        hit_offset = random.choice(_side_offsets(frame_h, frame_w, hit_side))
        top_left = _sample_aligned_bbox(shape, current_gap, direction, frame_h, frame_w, hit_offset)
        if top_left is None:
            continue
        r0, c0 = top_left
        positions = [pos for pos in _all_gap_positions(frame_h, frame_w) if pos != (hit_side, hit_offset)]
        if not positions:
            continue
        gap_count = random.choice([1, 2])
        gap_specs = random.sample(positions, gap_count)
        frame = _build_frame(r0, c0, frame_h, frame_w, gap_specs)
        hit_coord = _gap_coord(r0, c0, frame_h, frame_w, hit_side, hit_offset)
        if not _bbox_clear(frame["bbox"], frames):
            continue
        if not _segment_clear(shape, wall_cells, gap_entries, current_gap, direction, hit_coord):
            continue
        if hit_coord in {gap["coord"] for gap in frame["gaps"]}:
            continue
        return frame, hit_coord
    return None


def _place_random_frame(shape: tuple[int, int], frames: list[dict]) -> dict | None:
    height, width = map(int, shape)
    for _ in range(PLACEMENT_ATTEMPTS):
        gap_count = random.choice([1, 2])
        frame_h, frame_w = _sample_dims(5 if gap_count == 2 else 3)
        r0 = random.randint(1, height - frame_h - 1)
        c0 = random.randint(1, width - frame_w - 1)
        gap_specs = random.sample(_all_gap_positions(frame_h, frame_w), gap_count)
        frame = _build_frame(r0, c0, frame_h, frame_w, gap_specs)
        if _bbox_clear(frame["bbox"], frames):
            return frame
    return None


def _render_input(shape: tuple[int, int], frames: list[dict], wall_color: int, letter_color: int, start_id: int) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for frame in frames:
        for r, c in frame["wall"]:
            grid[r, c] = int(wall_color)
    grid[frames[int(start_id)]["center"]] = int(letter_color)
    return grid


def _expected_from_meta(
    grid: np.ndarray,
    frames: list[dict],
    start_id: int,
    transit_ids: list[int],
    final_pos: tuple[int, int],
    letter_color: int,
) -> np.ndarray:
    return _paint_output(grid, frames, start_id, transit_ids, final_pos, letter_color)


def _validate_case(
    grid: np.ndarray,
    frames: list[dict],
    wall_color: int,
    letter_color: int,
    start_id: int,
    transit_ids: list[int],
    final_kind: str,
    final_pos: tuple[int, int],
    *,
    pre_final_gap: tuple[int, int] | None = None,
    pre_final_dir: tuple[int, int] | None = None,
    blocker_sig: tuple | None = None,
    false_target_sig: tuple | None = None,
) -> bool:
    expected = _expected_from_meta(grid, frames, start_id, transit_ids, final_pos, letter_color)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    parsed_wall, parsed_letter, parsed_frames, parsed_start = _parse_frames(grid)
    if parsed_wall != wall_color or parsed_letter != letter_color:
        return False
    if {_frame_signature(frame) for frame in parsed_frames} != {_frame_signature(frame) for frame in frames}:
        return False
    if _frame_signature(parsed_frames[parsed_start]) != _frame_signature(frames[start_id]):
        return False

    trace = _trace_message(grid.shape, parsed_frames, parsed_start)
    parsed_transits = [_frame_signature(parsed_frames[idx]) for idx in trace["transit_ids"]]
    expected_transits = [_frame_signature(frames[idx]) for idx in transit_ids]
    if parsed_transits != expected_transits:
        return False
    if trace["stop_kind"] != final_kind or tuple(trace["final"]) != tuple(final_pos):
        return False

    if len(transit_ids) < 1:
        return False
    if final_kind == "edge":
        r, c = map(int, final_pos)
        if r not in {0, grid.shape[0] - 1} and c not in {0, grid.shape[1] - 1}:
            return False
        if int(grid[r, c]) != BG:
            return False
    if final_kind == "endpoint":
        if tuple(final_pos) == tuple(frames[start_id]["center"]):
            return False
    if final_kind == "wall":
        if None in {pre_final_gap, pre_final_dir, blocker_sig, false_target_sig}:
            return False
        frames_wo_blocker = [frame for frame in frames if _frame_signature(frame) != blocker_sig]
        wall_cells, gap_entries = _build_maps(frames_wo_blocker)
        event = _cast_ray(grid.shape, wall_cells, gap_entries, pre_final_gap, pre_final_dir)
        if event[0] != "enter":
            return False
        entered_sig = _frame_signature(frames_wo_blocker[event[1]])
        if entered_sig != false_target_sig:
            return False
    return True


def _generate_case() -> np.ndarray | None:
    shape = (
        int(random.randint(*GRID_HEIGHT_RANGE)),
        int(random.randint(*GRID_WIDTH_RANGE)),
    )
    wall_color, letter_color = random.sample(range(1, 10), 2)
    final_kind = random.choice(["endpoint", "wall", "edge"])
    transit_count = int(random.randint(*TRANSIT_COUNT_RANGE))

    frames: list[dict] = []
    start = _place_start_frame(shape, frames)
    if start is None:
        return None
    frames.append(start)
    start_id = 0

    current_gap = start["gaps"][0]["coord"]
    current_dir = start["gaps"][0]["out_dir"]
    transit_ids: list[int] = []

    for _ in range(transit_count):
        transit = _place_target_frame(shape, frames, current_gap, current_dir, "transit", require_clear=True)
        if transit is None:
            return None
        frames.append(transit)
        transit_ids.append(len(frames) - 1)
        current_gap = transit["gaps"][1]["coord"]
        current_dir = transit["gaps"][1]["out_dir"]

    pre_final_gap = current_gap
    pre_final_dir = current_dir
    blocker_sig = None
    false_target_sig = None

    if final_kind == "endpoint":
        endpoint = _place_target_frame(shape, frames, current_gap, current_dir, "endpoint", require_clear=True)
        if endpoint is None:
            return None
        frames.append(endpoint)
        final_pos = endpoint["center"]
    elif final_kind == "edge":
        wall_cells, gap_entries = _build_maps(frames)
        if not _ray_clear_to_edge(shape, wall_cells, gap_entries, current_gap, current_dir):
            return None
        final_pos = _edge_endpoint(current_gap, current_dir, shape)
    else:
        blocker = _place_blocker_frame(shape, frames, current_gap, current_dir)
        if blocker is None:
            return None
        blocker_frame, hit_coord = blocker
        frames.append(blocker_frame)
        blocker_sig = _frame_signature(blocker_frame)
        final_pos = (int(hit_coord[0] - current_dir[0]), int(hit_coord[1] - current_dir[1]))

        false_target = _place_target_frame(shape, frames, hit_coord, current_dir, "endpoint", require_clear=False)
        if false_target is None:
            return None
        frames.append(false_target)
        false_target_sig = _frame_signature(false_target)

    extra_count = int(random.randint(*EXTRA_FRAME_COUNT_RANGE))
    for _ in range(extra_count):
        extra = _place_random_frame(shape, frames)
        if extra is not None:
            frames.append(extra)

    grid = _render_input(shape, frames, wall_color, letter_color, start_id)
    if np.count_nonzero(grid == wall_color) <= 12:
        return None
    try:
        if not _validate_case(
            grid,
            frames,
            wall_color,
            letter_color,
            start_id,
            transit_ids,
            final_kind,
            final_pos,
            pre_final_gap=pre_final_gap,
            pre_final_dir=pre_final_dir,
            blocker_sig=blocker_sig,
            false_target_sig=false_target_sig,
        ):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible relay pipe case")
