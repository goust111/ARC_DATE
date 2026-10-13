import random
from collections import deque

import numpy as np


BG = 0
FRAME = 1
ORANGE = 7
TEAL = 8

MIN_FRAME_SIZE = 3
MAX_FRAME_SIZE = 6
MAX_GENERATION_ATTEMPTS = 400

PLACEMENTS = ("north", "south", "west", "east")
VERTICAL_PLACEMENTS = ("north", "south")
HORIZONTAL_PLACEMENTS = ("west", "east")


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _inside(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _inside(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert np.all((g == BG) | (g == FRAME)), "expected only black background and blue frame pixels"

    components = _find_components(g == FRAME)
    assert components, "expected at least one blue frame"

    frames: list[dict] = []
    for comp in components:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height >= MIN_FRAME_SIZE and width >= MIN_FRAME_SIZE, "frame bbox too small"

        expected = _border_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        assert comp_set <= expected, "frame pixels must lie on a rectangle border"
        missing = expected - comp_set
        assert len(missing) == 1, "each frame must have exactly one missing border cell"
        gap = next(iter(missing))
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        assert gap not in corners, "gap may not be a corner"

        if gap[0] == r0:
            direction = (-1, 0)
            side = "top"
        elif gap[0] == r1:
            direction = (1, 0)
            side = "bottom"
        elif gap[1] == c0:
            direction = (0, -1)
            side = "left"
        else:
            direction = (0, 1)
            side = "right"

        frames.append(
            {
                "r0": r0,
                "r1": r1,
                "c0": c0,
                "c1": c1,
                "gap": gap,
                "dir": direction,
                "side": side,
            }
        )
    return frames


def _render_input(shape: tuple[int, int], frames: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for frame in frames:
        r0, r1, c0, c1 = frame["r0"], frame["r1"], frame["c0"], frame["c1"]
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        gr, gc = frame["gap"]
        grid[gr, gc] = BG
    return grid


def _apply_rule(shape: tuple[int, int], frames: list[dict]) -> tuple[np.ndarray, dict]:
    base = _render_input(shape, frames)
    out = base.copy()
    interior = np.zeros(shape, dtype=bool)

    for frame in frames:
        r0, r1, c0, c1 = frame["r0"], frame["r1"], frame["c0"], frame["c1"]
        if r0 + 1 <= r1 - 1 and c0 + 1 <= c1 - 1:
            interior[r0 + 1 : r1, c0 + 1 : c1] = True
            out[r0 + 1 : r1, c0 + 1 : c1] = TEAL

    border_mask = base == FRAME
    counts = np.zeros(shape, dtype=int)
    rays: list[set[tuple[int, int]]] = []
    stop_modes: list[str] = []

    for frame in frames:
        r, c = frame["gap"]
        dr, dc = frame["dir"]
        ray_cells: set[tuple[int, int]] = set()
        while _inside(shape, r, c):
            if not interior[r, c] and not border_mask[r, c]:
                counts[r, c] += 1
                ray_cells.add((r, c))
            nr, nc = r + dr, c + dc
            if not _inside(shape, nr, nc):
                stop_modes.append("edge")
                break
            if border_mask[nr, nc]:
                stop_modes.append("block")
                break
            r, c = nr, nc
        rays.append(ray_cells)

    odd_mask = counts % 2 == 1
    even_mask = (counts >= 2) & (counts % 2 == 0)
    out[odd_mask] = TEAL
    out[even_mask] = ORANGE
    out[border_mask] = FRAME
    out[interior] = TEAL

    debug = {
        "counts": counts,
        "interior": interior,
        "rays": rays,
        "stops": stop_modes,
    }
    return out, debug


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the inside of each blue gap-frame and parity-color the rays emitted from its missing border cell.

    Input structure:
    - Black background with several blue hollow rectangular frames.
    - Each frame is axis-aligned, has a strict interior, and is missing exactly one non-corner border pixel.
    - The side containing that gap determines the outward direction of the frame's ray.

    Rule:
    - Fill every frame's strict interior with teal.
    - From each gap cell, extend a straight ray outward along the gap side's normal direction, counting that gap cell and later non-frame, non-interior cells until the grid edge or until the next step would hit a blue border pixel.
    - A ray-covered cell stays teal when covered by an odd number of rays, flips to orange when covered by an even number of rays of at least two, and untouched background stays black.

    Key insight:
    - The output is determined by combining all frames at once: each gap encodes one ray direction, and only the overlap parity of the resulting ray cover counts decides which cells turn orange.
    """
    g = np.asarray(grid, dtype=int)
    frames = _detect_frames(g)
    out, _debug = _apply_rule(g.shape, frames)
    return out


def _axis_values(lo: int, hi: int, count: int) -> list[int] | None:
    assert count in (1, 2)
    if lo > hi:
        return None
    if count == 1:
        return [random.randint(lo, hi)]

    deltas = (2, 3, 4, -2, -3, -4)
    for _ in range(80):
        first = random.randint(lo, hi)
        valid = [delta for delta in deltas if lo <= first + delta <= hi]
        if valid:
            second = first + random.choice(valid)
            return [first, second]
    return None


def _horizontal_starts(
    width_limit: int, frame_width: int, gap_col: int, include: int | None, exclude: int | None
) -> list[int]:
    lo = max(0, gap_col - (frame_width - 2))
    hi = min(width_limit - frame_width, gap_col - 1)
    starts = []
    for c0 in range(lo, hi + 1):
        if include is not None and not (c0 <= include <= c0 + frame_width - 1):
            continue
        if exclude is not None and c0 <= exclude <= c0 + frame_width - 1:
            continue
        starts.append(c0)
    return starts


def _vertical_starts(
    height_limit: int, frame_height: int, gap_row: int, include: int | None, exclude: int | None
) -> list[int]:
    lo = max(0, gap_row - (frame_height - 2))
    hi = min(height_limit - frame_height, gap_row - 1)
    starts = []
    for r0 in range(lo, hi + 1):
        if include is not None and not (r0 <= include <= r0 + frame_height - 1):
            continue
        if exclude is not None and r0 <= exclude <= r0 + frame_height - 1:
            continue
        starts.append(r0)
    return starts


def _frame_options(
    shape: tuple[int, int],
    placement: str,
    ray_value: int,
    row_limits: tuple[int, int],
    col_limits: tuple[int, int],
    include: int | None,
    exclude: int | None,
) -> list[dict]:
    h, w = shape
    options: list[dict] = []

    if placement == "north":
        top_gap_max = row_limits[0]
        for frame_height in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
            for frame_width in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
                starts = _horizontal_starts(w, frame_width, ray_value, include, exclude)
                if not starts:
                    continue
                for gap_row in range(frame_height - 1, top_gap_max + 1):
                    r0 = gap_row - frame_height + 1
                    for c0 in starts:
                        options.append(
                            {
                                "placement": placement,
                                "r0": r0,
                                "r1": gap_row,
                                "c0": c0,
                                "c1": c0 + frame_width - 1,
                                "gap": (gap_row, ray_value),
                                "dir": (1, 0),
                                "side": "bottom",
                            }
                        )
        return options

    if placement == "south":
        top_gap_min = row_limits[1]
        for frame_height in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
            for frame_width in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
                starts = _horizontal_starts(w, frame_width, ray_value, include, exclude)
                if not starts:
                    continue
                for gap_row in range(top_gap_min, h - frame_height + 1):
                    for c0 in starts:
                        options.append(
                            {
                                "placement": placement,
                                "r0": gap_row,
                                "r1": gap_row + frame_height - 1,
                                "c0": c0,
                                "c1": c0 + frame_width - 1,
                                "gap": (gap_row, ray_value),
                                "dir": (-1, 0),
                                "side": "top",
                            }
                        )
        return options

    if placement == "west":
        right_gap_max = col_limits[0]
        for frame_height in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
            starts = _vertical_starts(h, frame_height, ray_value, include, exclude)
            if not starts:
                continue
            for frame_width in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
                for gap_col in range(frame_width - 1, right_gap_max + 1):
                    c0 = gap_col - frame_width + 1
                    for r0 in starts:
                        options.append(
                            {
                                "placement": placement,
                                "r0": r0,
                                "r1": r0 + frame_height - 1,
                                "c0": c0,
                                "c1": gap_col,
                                "gap": (ray_value, gap_col),
                                "dir": (0, 1),
                                "side": "right",
                            }
                        )
        return options

    assert placement == "east"
    left_gap_min = col_limits[1]
    for frame_height in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
        starts = _vertical_starts(h, frame_height, ray_value, include, exclude)
        if not starts:
            continue
        for frame_width in range(MIN_FRAME_SIZE, MAX_FRAME_SIZE + 1):
            for gap_col in range(left_gap_min, w - frame_width + 1):
                for r0 in starts:
                    options.append(
                        {
                            "placement": placement,
                            "r0": r0,
                            "r1": r0 + frame_height - 1,
                            "c0": gap_col,
                            "c1": gap_col + frame_width - 1,
                            "gap": (ray_value, gap_col),
                            "dir": (0, -1),
                            "side": "left",
                        }
                    )
    return options


def _bboxes_apart(a: dict, b: dict, pad: int = 1) -> bool:
    return (
        a["r1"] + pad < b["r0"]
        or b["r1"] + pad < a["r0"]
        or a["c1"] + pad < b["c0"]
        or b["c1"] + pad < a["c0"]
    )


def _choose_frames(options_by_place: dict[str, list[dict]]) -> list[dict] | None:
    ordered = sorted(options_by_place, key=lambda key: len(options_by_place[key]))
    chosen: list[dict] = []

    def backtrack(index: int) -> bool:
        if index == len(ordered):
            return True
        place = ordered[index]
        options = list(options_by_place[place])
        random.shuffle(options)
        for frame in options:
            if all(_bboxes_apart(frame, other) for other in chosen):
                chosen.append(frame)
                if backtrack(index + 1):
                    return True
                chosen.pop()
        return False

    if backtrack(0):
        return list(chosen)
    return None


def _bbox_mask(shape: tuple[int, int], frame: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1] = True
    return mask


def _validate_candidate(shape: tuple[int, int], frames: list[dict], require_blocked: bool) -> bool:
    grid = _render_input(shape, frames)
    try:
        detected = _detect_frames(grid)
        expected, debug = _apply_rule(shape, detected)
        actual = main(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False

    if len({frame["side"] for frame in detected}) != len(detected):
        return False

    pairwise_hits = 0
    for i in range(len(debug["rays"])):
        for j in range(i + 1, len(debug["rays"])):
            if debug["rays"][i] & debug["rays"][j]:
                pairwise_hits += 1
    if pairwise_hits < 2:
        return False

    counts = debug["counts"]
    if int(np.count_nonzero(counts >= 2)) < 2:
        return False
    if int(np.count_nonzero(counts == 1)) < len(frames) + 2:
        return False

    if require_blocked and "block" not in debug["stops"]:
        return False

    full_out = expected
    for idx, frame in enumerate(detected):
        reduced = detected[:idx] + detected[idx + 1 :]
        reduced_out, _ = _apply_rule(shape, reduced)
        outside = ~_bbox_mask(shape, frame)
        if not np.any(full_out[outside] != reduced_out[outside]):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(20, 26)
    width = random.randint(20, 26)
    shape = (height, width)

    if random.random() < 0.45:
        missing = random.choice(PLACEMENTS)
        present = [place for place in PLACEMENTS if place != missing]
    else:
        present = list(PLACEMENTS)

    vertical_places = [place for place in VERTICAL_PLACEMENTS if place in present]
    horizontal_places = [place for place in HORIZONTAL_PLACEMENTS if place in present]

    col_values = _axis_values(7, width - 8, len(vertical_places))
    row_values = _axis_values(7, height - 8, len(horizontal_places))
    if col_values is None or row_values is None:
        return None

    random.shuffle(col_values)
    random.shuffle(row_values)
    vertical_lines = dict(zip(vertical_places, col_values))
    horizontal_lines = dict(zip(horizontal_places, row_values))

    min_row = min(horizontal_lines.values())
    max_row = max(horizontal_lines.values())
    min_col = min(vertical_lines.values())
    max_col = max(vertical_lines.values())

    vertical_block = None
    if set(vertical_places) == set(VERTICAL_PLACEMENTS) and random.random() < 0.55:
        vertical_block = random.choice(("north_hits_south", "south_hits_north"))

    horizontal_block = None
    if set(horizontal_places) == set(HORIZONTAL_PLACEMENTS) and random.random() < 0.55:
        horizontal_block = random.choice(("west_hits_east", "east_hits_west"))

    require_blocked = vertical_block is not None or horizontal_block is not None

    options_by_place: dict[str, list[dict]] = {}
    for place in present:
        include = None
        exclude = None
        if place == "north":
            if "south" in vertical_lines:
                other = vertical_lines["south"]
                if vertical_block == "south_hits_north":
                    include = other
                else:
                    exclude = other
            ray_value = vertical_lines["north"]
        elif place == "south":
            if "north" in vertical_lines:
                other = vertical_lines["north"]
                if vertical_block == "north_hits_south":
                    include = other
                else:
                    exclude = other
            ray_value = vertical_lines["south"]
        elif place == "west":
            if "east" in horizontal_lines:
                other = horizontal_lines["east"]
                if horizontal_block == "east_hits_west":
                    include = other
                else:
                    exclude = other
            ray_value = horizontal_lines["west"]
        else:
            if "west" in horizontal_lines:
                other = horizontal_lines["west"]
                if horizontal_block == "west_hits_east":
                    include = other
                else:
                    exclude = other
            ray_value = horizontal_lines["east"]

        options = _frame_options(
            shape,
            place,
            ray_value,
            row_limits=(min_row - 2, max_row + 2),
            col_limits=(min_col - 2, max_col + 2),
            include=include,
            exclude=exclude,
        )
        if not options:
            return None
        options_by_place[place] = options

    frames = _choose_frames(options_by_place)
    if frames is None:
        return None

    if not _validate_candidate(shape, frames, require_blocked):
        return None

    return _render_input(shape, frames)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
