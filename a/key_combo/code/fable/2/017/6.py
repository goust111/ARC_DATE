from __future__ import annotations

from collections import Counter

import numpy as np


SLASH_COLOR = 8
BACKSLASH_COLOR = 2
FRAME_COLOR_CHOICES = [1, 3, 4, 5, 6, 7, 9]
EMITTER_COLOR_CHOICES = [1, 3, 4, 5, 6, 7, 9]

DIRS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GENERATION_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_border_coords(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _is_edge_local(cell: tuple[int, int], height: int, width: int) -> bool:
    r, c = cell
    return r == 0 or r == height - 1 or c == 0 or c == width - 1


def _perp_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (-dc, -dr), (dc, dr)


def _reflect(
    direction: tuple[int, int],
    mirror_color: int,
    *,
    slash_color: int = SLASH_COLOR,
    backslash_color: int = BACKSLASH_COLOR,
) -> tuple[int, int]:
    dr, dc = direction
    if int(mirror_color) == int(slash_color):
        return -dc, -dr
    if int(mirror_color) == int(backslash_color):
        return dc, dr
    raise AssertionError("unexpected mirror color")


def _mirror_color_for_turn(direction: tuple[int, int], new_direction: tuple[int, int]) -> int:
    slash_dir, backslash_dir = _perp_dirs(direction)
    if tuple(new_direction) == slash_dir:
        return SLASH_COLOR
    if tuple(new_direction) == backslash_dir:
        return BACKSLASH_COLOR
    raise AssertionError("turn is not a valid 90-degree mirror reflection")


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    candidates = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        coords = np.argwhere(g == int(color))
        if coords.shape[0] < 8:
            continue
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        if r0 <= 0 or c0 <= 0 or r1 >= height - 1 or c1 >= width - 1:
            continue
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if {tuple(map(int, rc)) for rc in coords} != _rectangle_border_coords(bbox):
            continue
        candidates.append((int(color), bbox))
    assert len(candidates) == 1, "expected exactly one framed rectangle"
    return candidates[0]


def _collect_emitters(
    grid: np.ndarray,
    frame_color: int,
    frame_bbox: tuple[int, int, int, int],
    bg: int,
    *,
    slash_color: int = SLASH_COLOR,
    backslash_color: int = BACKSLASH_COLOR,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    emitters = []
    used_colors = set()
    for color in sorted(set(map(int, g.ravel())) - {int(bg), int(frame_color), int(slash_color), int(backslash_color)}):
        coords = np.argwhere(g == int(color))
        assert coords.shape == (1, 2), "each emitter color must appear exactly once"
        r, c = map(int, coords[0])
        if r == r0 - 1 and c0 < c < c1:
            emitters.append(
                {
                    "color": int(color),
                    "side": "top",
                    "slot": c - (c0 + 1),
                    "start": (r0 + 1, c),
                    "direction": DIRS["D"],
                    "emitter": (r, c),
                }
            )
        elif r == r1 + 1 and c0 < c < c1:
            emitters.append(
                {
                    "color": int(color),
                    "side": "bottom",
                    "slot": c - (c0 + 1),
                    "start": (r1 - 1, c),
                    "direction": DIRS["U"],
                    "emitter": (r, c),
                }
            )
        elif c == c0 - 1 and r0 < r < r1:
            emitters.append(
                {
                    "color": int(color),
                    "side": "left",
                    "slot": r - (r0 + 1),
                    "start": (r, c0 + 1),
                    "direction": DIRS["R"],
                    "emitter": (r, c),
                }
            )
        elif c == c1 + 1 and r0 < r < r1:
            emitters.append(
                {
                    "color": int(color),
                    "side": "right",
                    "slot": r - (r0 + 1),
                    "start": (r, c1 - 1),
                    "direction": DIRS["L"],
                    "emitter": (r, c),
                }
            )
        else:
            raise AssertionError("non-mirror foreground outside the frame must be an emitter")
        assert int(color) not in used_colors
        used_colors.add(int(color))
    assert emitters, "expected at least one emitter"
    return emitters


def _trace_all(
    grid: np.ndarray,
    *,
    slash_color: int = SLASH_COLOR,
    backslash_color: int = BACKSLASH_COLOR,
) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frame_color, frame_bbox = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = True
    interior_values = set(map(int, g[interior_mask]))
    assert interior_values <= {int(bg), int(slash_color), int(backslash_color)}, "interior may only contain background or mirrors"

    emitters = _collect_emitters(g, frame_color, frame_bbox, bg, slash_color=slash_color, backslash_color=backslash_color)

    mirror_mask = interior_mask & ((g == int(slash_color)) | (g == int(backslash_color)))
    mirror_coords = {tuple(map(int, rc)) for rc in np.argwhere(mirror_mask)}
    used_mirror_coords = set()
    used_mirror_colors = set()

    out = g.copy()
    traces = []
    for emitter in sorted(emitters, key=lambda item: (item["side"], item["slot"], item["color"])):
        pos = tuple(map(int, emitter["start"]))
        direction = tuple(map(int, emitter["direction"]))
        color = int(emitter["color"])
        seen_states = set()
        bg_cells = []
        mirrors_on_path = []

        while True:
            state = (pos[0], pos[1], direction[0], direction[1])
            assert state not in seen_states, "ray tracing must not loop"
            seen_states.add(state)

            r, c = pos
            value = int(g[r, c])
            if value == int(bg):
                assert out[r, c] in (int(bg), color), "paths must not overlap"
                out[r, c] = color
                bg_cells.append((r, c))
            elif value in (int(slash_color), int(backslash_color)):
                used_mirror_coords.add((r, c))
                used_mirror_colors.add(value)
                mirrors_on_path.append((r, c, value))
                direction = _reflect(direction, value, slash_color=slash_color, backslash_color=backslash_color)
            else:
                raise AssertionError("ray hit an unexpected interior object")

            nr = r + direction[0]
            nc = c + direction[1]
            if not (ir0 <= nr <= ir1 and ic0 <= nc <= ic1):
                break
            pos = (nr, nc)

        traces.append(
            {
                "color": color,
                "side": emitter["side"],
                "slot": emitter["slot"],
                "bg_cells": bg_cells,
                "mirrors": mirrors_on_path,
            }
        )

    return {
        "out": out,
        "bg": int(bg),
        "frame_color": int(frame_color),
        "frame_bbox": frame_bbox,
        "emitters": emitters,
        "mirror_coords": mirror_coords,
        "used_mirror_coords": used_mirror_coords,
        "used_mirror_colors": used_mirror_colors,
        "traces": traces,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each exterior emitter ray through a framed mirror box and draw its bent path.

    Input structure:
    - A single hollow rectangular frame sits away from the outer grid boundary.
    - Inside the frame, the background is empty except for single-pixel teal and red diagonal mirrors.
    - Outside the frame, unique one-pixel emitters touch the frame on one side and therefore define an inward normal entry direction.

    Rule:
    - For each emitter, start from the first interior cell behind the touched frame side and move straight inward.
    - Teal mirrors behave like "/" and red mirrors behave like "\\"; when a ray reaches a mirror cell it turns and continues, but the mirror pixel itself keeps its original color.
    - Paint every traversed background cell with that emitter's color, and stop when the next step would leave the frame interior and hit the inner wall.

    Key insight:
    - The visible output path is not present in the input; it is reconstructed by combining emitter-side normals, the frame's interior coordinate system, and the fixed diagonal reflection rule of the two mirror colors.
    """
    return _trace_all(grid)["out"]


def _edge_slots(side: str, height: int, width: int) -> list[int]:
    if side in ("top", "bottom"):
        return list(range(1, width - 1))
    return list(range(1, height - 1))


def _start_from_slot(side: str, slot: int, height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (0, int(slot)), DIRS["D"]
    if side == "bottom":
        return (height - 1, int(slot)), DIRS["U"]
    if side == "left":
        return (int(slot), 0), DIRS["R"]
    if side == "right":
        return (int(slot), width - 1), DIRS["L"]
    raise ValueError(side)


def _slot_is_clear(side: str, slot: int, used_slots: set[tuple[str, int]]) -> bool:
    for old_side, old_slot in used_slots:
        if side == old_side and abs(int(slot) - int(old_slot)) <= 1:
            return False
    return True


def _local_to_global(
    cell: tuple[int, int],
    frame_bbox: tuple[int, int, int, int],
) -> tuple[int, int]:
    r0, c0, _r1, _c1 = frame_bbox
    return r0 + 1 + int(cell[0]), c0 + 1 + int(cell[1])


def _emitter_global(
    side: str,
    slot: int,
    frame_bbox: tuple[int, int, int, int],
    height: int,
    width: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return r0 - 1, c0 + 1 + int(slot)
    if side == "bottom":
        return r1 + 1, c0 + 1 + int(slot)
    if side == "left":
        return r0 + 1 + int(slot), c0 - 1
    if side == "right":
        return r0 + 1 + int(slot), c1 + 1
    raise ValueError(side)


def _build_path(
    rng: np.random.Generator,
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    side: str,
    slot: int,
    turn_target: int,
) -> dict | None:
    start, start_dir = _start_from_slot(side, slot, height, width)
    if start in occupied:
        return None

    def dfs(
        pos: tuple[int, int],
        direction: tuple[int, int],
        turns_left: int,
        used: set[tuple[int, int]],
    ) -> dict | None:
        line = []
        rr, cc = pos
        while True:
            rr += direction[0]
            cc += direction[1]
            if not (0 <= rr < height and 0 <= cc < width):
                break
            cell = (rr, cc)
            if cell in occupied or cell in used:
                break
            line.append(cell)

        candidates = []
        if turns_left == 0:
            for idx, cell in enumerate(line, start=1):
                if _is_edge_local(cell, height, width):
                    candidates.append(("end", idx, cell))
        else:
            for idx, cell in enumerate(line, start=1):
                if _is_edge_local(cell, height, width):
                    break
                if idx < 2:
                    continue
                for new_direction in _perp_dirs(direction):
                    nr = cell[0] + new_direction[0]
                    nc = cell[1] + new_direction[1]
                    next_cell = (nr, nc)
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if next_cell in occupied or next_cell in used:
                        continue
                    candidates.append(
                        (
                            "turn",
                            idx,
                            cell,
                            new_direction,
                            _mirror_color_for_turn(direction, new_direction),
                        )
                    )
        if not candidates:
            return None

        order = list(range(len(candidates)))
        rng.shuffle(order)
        for candidate_index in order:
            candidate = candidates[candidate_index]
            if candidate[0] == "end":
                _, steps, _cell = candidate
                segment = line[:steps]
                return {"bg_cells": segment, "mirrors": []}

            _, steps, mirror_cell, new_direction, mirror_color = candidate
            used_now = used | set(line[:steps])
            suffix = dfs(mirror_cell, new_direction, turns_left - 1, used_now)
            if suffix is None:
                continue
            return {
                "bg_cells": line[: steps - 1] + suffix["bg_cells"],
                "mirrors": [(mirror_cell, mirror_color)] + suffix["mirrors"],
            }
        return None

    suffix = dfs(start, start_dir, int(turn_target), {start})
    if suffix is None:
        return None
    return {
        "side": side,
        "slot": int(slot),
        "start": start,
        "direction": start_dir,
        "bg_cells": [start] + suffix["bg_cells"],
        "mirrors": suffix["mirrors"],
    }


def _paint_expected_output(
    grid: np.ndarray,
    paths: list[dict],
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for path in paths:
        color = int(path["color"])
        for cell in path["bg_cells"]:
            rr, cc = _local_to_global(cell, frame_bbox)
            out[rr, cc] = color
    return out


def _attempt_generate(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    interior_height = int(rng.integers(8, 15))
    interior_width = int(rng.integers(8, 15))
    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))

    frame_top = top_margin
    frame_left = left_margin
    frame_bottom = frame_top + interior_height + 1
    frame_right = frame_left + interior_width + 1
    grid_height = frame_bottom + bottom_margin + 1
    grid_width = frame_right + right_margin + 1
    if max(grid_height, grid_width) > 30:
        return None

    frame_bbox = (frame_top, frame_left, frame_bottom, frame_right)
    bg = 0
    frame_color = int(rng.choice(FRAME_COLOR_CHOICES))
    emitter_palette = [color for color in EMITTER_COLOR_CHOICES if color not in {frame_color, SLASH_COLOR, BACKSLASH_COLOR}]
    path_count = int(rng.integers(2, 5))
    if len(emitter_palette) < path_count:
        return None
    emitter_colors = list(map(int, rng.choice(emitter_palette, size=path_count, replace=False)))

    turn_targets = [int(rng.integers(1, 4)) for _ in range(path_count)]
    if path_count >= 3 and rng.random() < 0.45:
        turn_targets[int(rng.integers(0, path_count))] = 0

    occupied: set[tuple[int, int]] = set()
    used_slots: set[tuple[str, int]] = set()
    paths = []

    for color, turn_target in zip(emitter_colors, turn_targets):
        starts = []
        for side in ("top", "bottom", "left", "right"):
            for slot in _edge_slots(side, interior_height, interior_width):
                if not _slot_is_clear(side, slot, used_slots):
                    continue
                starts.append((side, slot))
        if not starts:
            return None

        path = None
        for index in rng.permutation(len(starts)):
            side, slot = starts[int(index)]
            candidate = _build_path(
                rng,
                interior_height,
                interior_width,
                occupied,
                side,
                slot,
                turn_target,
            )
            if candidate is None:
                continue
            candidate["color"] = int(color)
            path = candidate
            break
        if path is None:
            return None

        occupied.update(path["bg_cells"])
        occupied.update(cell for cell, _mirror_color in path["mirrors"])
        if len(occupied) > (interior_height * interior_width) // 2:
            return None
        used_slots.add((path["side"], path["slot"]))
        paths.append(path)

    if len({path["side"] for path in paths}) < 2:
        return None

    total_mirror_colors = {int(mirror_color) for path in paths for _cell, mirror_color in path["mirrors"]}
    if total_mirror_colors != {SLASH_COLOR, BACKSLASH_COLOR}:
        return None

    grid = np.full((grid_height, grid_width), bg, dtype=int)
    for r, c in _rectangle_border_coords(frame_bbox):
        grid[r, c] = frame_color
    for path in paths:
        emitter_r, emitter_c = _emitter_global(path["side"], path["slot"], frame_bbox, interior_height, interior_width)
        grid[emitter_r, emitter_c] = int(path["color"])
        for cell, mirror_color in path["mirrors"]:
            rr, cc = _local_to_global(cell, frame_bbox)
            assert grid[rr, cc] == bg
            grid[rr, cc] = int(mirror_color)

    expected = _paint_expected_output(grid, paths, frame_bbox)
    return grid, expected


def _validate_generated_pair(grid: np.ndarray, expected: np.ndarray) -> bool:
    if np.array_equal(grid, expected):
        return False

    try:
        traced = _trace_all(grid)
        alt = _trace_all(grid, slash_color=BACKSLASH_COLOR, backslash_color=SLASH_COLOR)
    except AssertionError:
        return False

    if not np.array_equal(traced["out"], expected):
        return False
    if traced["mirror_coords"] != traced["used_mirror_coords"]:
        return False
    if traced["used_mirror_colors"] != {SLASH_COLOR, BACKSLASH_COLOR}:
        return False
    if len(traced["emitters"]) < 2:
        return False
    if len({item["side"] for item in traced["emitters"]}) < 2:
        return False
    if len(traced["mirror_coords"]) < 2:
        return False
    if np.array_equal(alt["out"], expected):
        return False
    if sum(len(path["bg_cells"]) for path in traced["traces"]) < 8:
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        pair = _attempt_generate(rng)
        if pair is None:
            continue
        grid, expected = pair
        if _validate_generated_pair(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid slanted-mirror ray-tracing puzzle")
