from __future__ import annotations

import random
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
try:
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import bbox as _arc_bbox
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import get_bg as _arc_get_bg
except Exception:
    _arc_bbox = None
    _arc_crop_bbox = None
    _arc_get_bg = None


Coord = tuple[int, int]

BG = 0
GLYPH = 5
OBSTACLE = 8
SEPARATOR_POOL = (1, 2, 3, 4, 6, 7, 9)
START_POOL = (1, 2, 3, 4, 6, 7, 9)
TURN_SIGNS = (-1, 1)
DIRS = ("up", "right", "down", "left")
DIR_TO_VEC = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_TO_SIDE = {
    "up": "top",
    "right": "right",
    "down": "bottom",
    "left": "left",
}
BASE_GLYPHS = {
    "up": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    "right": np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    "down": np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    "left": np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
}
GLYPH_TO_DIR = {tuple(mask.ravel().tolist()): direction for direction, mask in BASE_GLYPHS.items()}

MAIN_HEIGHT_LIMIT = 26
GRID_LIMIT = 30
GLYPH_STRIP_HEIGHT = 3
GEN_ATTEMPTS = 600
CMD_COUNT_CHOICES = (4, 5, 6)
SEGMENT_LEN_MAX = 8
NONBOUNDARY_MARGIN_RANGE = (1, 3)
EXTRA_MARGIN_RANGE = (0, 2)


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    counts = Counter(map(int, np.asarray(grid, dtype=int).ravel()))
    return int(counts.most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if _arc_bbox is not None:
        box = _arc_bbox(mask)
        return None if box is None else tuple(map(int, box))
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return np.asarray(_arc_crop_bbox(grid, box), dtype=grid.dtype)
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _separator_row(g: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        row_colors = set(map(int, g[r, :].tolist()))
        if len(row_colors) != 1:
            continue
        color = next(iter(row_colors))
        if color == bg:
            continue
        candidates.append((r, color))
    assert len(candidates) == 1, "expected one solid non-background separator row"
    return candidates[0]


def _decode_commands(strip: np.ndarray, bg: int) -> list[str]:
    assert strip.shape[0] == GLYPH_STRIP_HEIGHT, "glyph strip must be exactly three rows tall"
    glyph_mask = np.any(strip != bg, axis=0)
    runs = _runs(glyph_mask)
    assert runs, "expected at least one glyph"

    colors = sorted(set(map(int, strip[strip != bg].ravel())))
    assert len(colors) == 1, "glyph strip must use one foreground color"

    commands: list[str] = []
    for c0, c1 in runs:
        assert c1 - c0 + 1 == 3, "each glyph must occupy a 3-column slot"
        patch = strip[:, c0 : c1 + 1]
        mask = patch != bg
        box = _bbox(mask)
        assert box is not None, "glyph patch cannot be empty"
        cropped = _crop_bbox(mask.astype(int), box).astype(bool)
        assert cropped.shape == (3, 3), "each glyph must touch its 3x3 slot"
        key = tuple(cropped.astype(int).ravel().tolist())
        assert key in GLYPH_TO_DIR, "unknown glyph pattern"
        commands.append(GLYPH_TO_DIR[key])
    return commands


def _parse_main_region(main_region: np.ndarray, bg: int) -> tuple[Coord, int, int]:
    fg_colors = [int(color) for color in np.unique(main_region) if int(color) != bg]
    counts = {color: int(np.count_nonzero(main_region == color)) for color in fg_colors}
    start_candidates = [color for color, count in counts.items() if count == 1]
    assert len(start_candidates) == 1, "expected one singleton start color"
    start_color = start_candidates[0]
    obstacle_candidates = [color for color in fg_colors if color != start_color]
    assert len(obstacle_candidates) == 1, "expected one obstacle color"
    obstacle_color = obstacle_candidates[0]
    start_coords = np.argwhere(main_region == start_color)
    assert start_coords.shape[0] == 1
    start = tuple(map(int, start_coords[0]))
    return start, start_color, obstacle_color


def _simulate_path(
    main_region: np.ndarray,
    commands: list[str],
    start: Coord,
    start_color: int,
    obstacle_color: int,
) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(main_region, dtype=int).copy()
    blocked = main_region == obstacle_color
    height, width = map(int, main_region.shape)
    cur = tuple(map(int, start))
    traces: list[dict] = []

    for command in commands:
        dr, dc = DIR_TO_VEC[command]
        cells: list[Coord] = []
        while True:
            nr = cur[0] + dr
            nc = cur[1] + dc
            if not (0 <= nr < height and 0 <= nc < width):
                stop_reason = "boundary"
                break
            if blocked[nr, nc]:
                stop_reason = "obstacle"
                break
            cur = (nr, nc)
            out[nr, nc] = start_color
            cells.append(cur)
        traces.append(
            {
                "dir": command,
                "cells": cells,
                "end": cur,
                "stop": stop_reason,
            }
        )
    return out, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the bottom glyph strip into ray directions and draw the resulting multi-segment trail in the upper field.

    Input structure:
    - A single solid separator row splits the canvas into an upper main region and a lower three-row glyph strip.
    - The glyph strip contains left-to-right 3x3 grey-black symbols chosen from four fixed directional patterns.
    - The main region contains one singleton colored start pixel and several single-cell obstacle pixels of one other color.

    Rule:
    - Decode each glyph to one of the four cardinal directions and execute the commands in strip order.
    - For each command, move the current point cell by cell in that direction, painting every visited cell with the start color until the next cell would be an obstacle or would leave the main region.
    - After all commands finish, discard the separator row and glyph strip and return only the painted main region.

    Key insight:
    - The glyphs are not decorative objects but a program: the symbols supply direction while the obstacle field and borders supply each segment's run length, so neither half of the input is sufficient on its own.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 0 < g.shape[0] <= GRID_LIMIT and 0 < g.shape[1] <= GRID_LIMIT

    bg = _get_bg(g)
    sep_row, _separator_color = _separator_row(g, bg)
    assert 0 < sep_row < g.shape[0] - 1, "separator must split non-empty regions"

    main_region = g[:sep_row, :]
    glyph_strip = g[sep_row + 1 :, :]
    assert main_region.size > 0
    assert glyph_strip.shape[0] == GLYPH_STRIP_HEIGHT

    commands = _decode_commands(glyph_strip, bg)
    start, start_color, obstacle_color = _parse_main_region(main_region, bg)
    out, _traces = _simulate_path(main_region, commands, start, start_color, obstacle_color)
    return out


def _rotating_commands() -> list[str]:
    start_idx = random.randrange(4)
    turn_sign = random.choice(TURN_SIGNS)
    count = random.choice(CMD_COUNT_CHOICES)
    return [DIRS[(start_idx + i * turn_sign) % 4] for i in range(count)]


def _descending_lengths(count: int) -> list[int] | None:
    if count <= 0:
        return []
    if SEGMENT_LEN_MAX < count:
        return None
    return sorted(random.sample(range(1, SEGMENT_LEN_MAX + 1), count), reverse=True)


def _segment_step(a: Coord, b: Coord) -> list[Coord]:
    r0, c0 = map(int, a)
    r1, c1 = map(int, b)
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    assert c0 == c1
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _build_relative_path(commands: list[str]) -> tuple[list[dict], set[Coord]] | None:
    first_axis = commands[0] in ("up", "down")
    count_a = (len(commands) + 1) // 2
    count_b = len(commands) // 2
    lengths_a = _descending_lengths(count_a)
    lengths_b = _descending_lengths(count_b)
    if lengths_a is None or lengths_b is None:
        return None

    idx_a = 0
    idx_b = 0
    cur = (0, 0)
    occupied = {cur}
    segments: list[dict] = []

    for i, command in enumerate(commands):
        if (command in ("up", "down")) == first_axis:
            length = lengths_a[idx_a]
            idx_a += 1
        else:
            length = lengths_b[idx_b]
            idx_b += 1

        dr, dc = DIR_TO_VEC[command]
        cells: list[Coord] = []
        for _ in range(length):
            nxt = (cur[0] + dr, cur[1] + dc)
            if nxt in occupied:
                return None
            cells.append(nxt)
            occupied.add(nxt)
            cur = nxt

        segments.append(
            {
                "dir": command,
                "length": length,
                "cells": cells,
                "start": cells[0] if cells else cur,
                "end": cur,
            }
        )

    return segments, occupied


def _path_bbox(cells: set[Coord]) -> tuple[int, int, int, int]:
    arr = np.array(sorted(cells), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _candidate_boundary_sides(segments: list[dict], occupied: set[Coord]) -> set[str]:
    r0, c0, r1, c1 = _path_bbox(occupied)
    sides: set[str] = set()
    for segment in segments:
        er, ec = map(int, segment["end"])
        direction = segment["dir"]
        if direction == "up" and er == r0:
            sides.add("top")
        elif direction == "down" and er == r1:
            sides.add("bottom")
        elif direction == "left" and ec == c0:
            sides.add("left")
        elif direction == "right" and ec == c1:
            sides.add("right")
    return sides


def _choose_boundary_sides(candidate_sides: set[str]) -> set[str] | None:
    if not candidate_sides:
        return None
    chosen: set[str] = set()
    side_list = sorted(candidate_sides)
    random.shuffle(side_list)
    chosen.add(side_list[0])
    if len(side_list) > 1 and random.random() < 0.45:
        chosen.add(side_list[1])
    return chosen


def _min_margin(side: str, boundary_sides: set[str]) -> int:
    return 0 if side in boundary_sides else random.randint(*NONBOUNDARY_MARGIN_RANGE)


def _strip_width(command_count: int) -> tuple[int, int, int]:
    left_pad = random.randint(0, 2)
    right_pad = random.randint(0, 2)
    gap = 1
    width = left_pad + command_count * 3 + max(0, command_count - 1) * gap + right_pad
    return width, left_pad, gap


def _allocate_margins(
    path_width: int,
    path_height: int,
    boundary_sides: set[str],
    glyph_width: int,
) -> tuple[dict[str, int], tuple[int, int]] | None:
    margins = {
        "top": _min_margin("top", boundary_sides),
        "bottom": _min_margin("bottom", boundary_sides),
        "left": _min_margin("left", boundary_sides),
        "right": _min_margin("right", boundary_sides),
    }

    expandable_h = [side for side in ("left", "right") if side not in boundary_sides]
    needed_width = glyph_width - (path_width + margins["left"] + margins["right"])
    if needed_width > 0:
        if not expandable_h:
            return None
        for _ in range(needed_width):
            side = random.choice(expandable_h)
            margins[side] += 1

    for side in expandable_h:
        margins[side] += random.randint(*EXTRA_MARGIN_RANGE)

    expandable_v = [side for side in ("top", "bottom") if side not in boundary_sides]
    for side in expandable_v:
        margins[side] += random.randint(*EXTRA_MARGIN_RANGE)

    main_height = path_height + margins["top"] + margins["bottom"]
    main_width = path_width + margins["left"] + margins["right"]
    if main_height > MAIN_HEIGHT_LIMIT or main_width > GRID_LIMIT:
        return None
    return margins, (main_height, main_width)


def _translate(coord: Coord, row_offset: int, col_offset: int) -> Coord:
    return (int(coord[0] + row_offset), int(coord[1] + col_offset))


def _command_sides_from_segments(
    segments: list[dict],
    row_offset: int,
    col_offset: int,
    main_shape: tuple[int, int],
) -> dict[int, str]:
    height, width = map(int, main_shape)
    result: dict[int, str] = {}
    for i, segment in enumerate(segments):
        er, ec = _translate(segment["end"], row_offset, col_offset)
        direction = segment["dir"]
        if direction == "up" and er == 0:
            result[i] = "top"
        elif direction == "down" and er == height - 1:
            result[i] = "bottom"
        elif direction == "left" and ec == 0:
            result[i] = "left"
        elif direction == "right" and ec == width - 1:
            result[i] = "right"
    return result


def _obstacle_positions(
    segments: list[dict],
    row_offset: int,
    col_offset: int,
    boundary_indices: set[int],
    path_cells: set[Coord],
    main_shape: tuple[int, int],
) -> set[Coord] | None:
    height, width = map(int, main_shape)
    obstacles: set[Coord] = set()
    translated_path = {_translate(cell, row_offset, col_offset) for cell in path_cells}
    for i, segment in enumerate(segments):
        if i in boundary_indices:
            continue
        end = _translate(segment["end"], row_offset, col_offset)
        dr, dc = DIR_TO_VEC[segment["dir"]]
        blocker = (end[0] + dr, end[1] + dc)
        if not (0 <= blocker[0] < height and 0 <= blocker[1] < width):
            return None
        if blocker in translated_path or blocker in obstacles:
            return None
        obstacles.add(blocker)
    return obstacles


def _render_main_region(
    main_shape: tuple[int, int],
    start: Coord,
    start_color: int,
    obstacles: set[Coord],
) -> np.ndarray:
    out = np.full(main_shape, BG, dtype=int)
    out[start] = start_color
    for r, c in obstacles:
        out[r, c] = OBSTACLE
    return out


def _render_glyph_strip(commands: list[str], width: int, left_pad: int, gap: int) -> np.ndarray:
    strip = np.full((GLYPH_STRIP_HEIGHT, width), BG, dtype=int)
    c = left_pad
    for i, command in enumerate(commands):
        strip[:, c : c + 3] = np.where(BASE_GLYPHS[command], GLYPH, BG)
        c += 3
        if i + 1 < len(commands):
            c += gap
    return strip


def _render_input(
    main_region: np.ndarray,
    commands: list[str],
    separator_color: int,
    left_pad: int,
    gap: int,
) -> np.ndarray:
    width = main_region.shape[1]
    separator = np.full((1, width), separator_color, dtype=int)
    strip = _render_glyph_strip(commands, width, left_pad, gap)
    return np.vstack([main_region, separator, strip])


def _render_expected(main_region: np.ndarray, commands: list[str]) -> tuple[np.ndarray, list[dict], Coord, int]:
    bg = _get_bg(main_region)
    start, start_color, obstacle_color = _parse_main_region(main_region, bg)
    out, traces = _simulate_path(main_region, commands, start, start_color, obstacle_color)
    return out, traces, start, start_color


def _all_four_directions(commands: list[str]) -> bool:
    return set(commands) == set(DIRS)


def _changed_output(main_region: np.ndarray, commands: list[str], expected: np.ndarray) -> bool:
    for i, command in enumerate(commands):
        shorter = commands[:i] + commands[i + 1 :]
        alt_out, _alt_traces, _start, _start_color = _render_expected(main_region, shorter)
        if np.array_equal(alt_out, expected):
            return False
        for other in DIRS:
            if other == command:
                continue
            mutated = commands.copy()
            mutated[i] = other
            alt_out, _alt_traces, _start, _start_color = _render_expected(main_region, mutated)
            if np.array_equal(alt_out, expected):
                return False
    return True


def _mapping_is_unique(main_region: np.ndarray, command_symbols: list[str], expected: np.ndarray) -> bool:
    symbol_order = list(DIRS)
    identity = tuple(symbol_order)
    for perm in permutations(DIRS):
        if perm == identity:
            continue
        remap = {symbol_order[i]: perm[i] for i in range(4)}
        remapped = [remap[symbol] for symbol in command_symbols]
        alt_out, _alt_traces, _start, _start_color = _render_expected(main_region, remapped)
        if np.array_equal(alt_out, expected):
            return False
    return True


def _semantic_ok(
    inp: np.ndarray,
    planned_commands: list[str],
    boundary_indices: set[int],
) -> bool:
    try:
        predicted = main(inp)
    except Exception:
        return False

    bg = _get_bg(inp)
    sep_row, separator_color = _separator_row(inp, bg)
    if separator_color == BG or sep_row <= 0:
        return False

    main_region = inp[:sep_row, :]
    glyph_strip = inp[sep_row + 1 :, :]
    if glyph_strip.shape[0] != GLYPH_STRIP_HEIGHT:
        return False

    try:
        decoded = _decode_commands(glyph_strip, bg)
        expected, traces, start, start_color = _render_expected(main_region, decoded)
    except Exception:
        return False

    if decoded != planned_commands:
        return False
    if not np.array_equal(predicted, expected):
        return False
    if not _all_four_directions(decoded):
        return False
    if any(len(trace["cells"]) == 0 for trace in traces):
        return False
    if len({trace["stop"] for trace in traces}) != 2:
        return False
    if not {"boundary", "obstacle"} <= {trace["stop"] for trace in traces}:
        return False
    if any(decoded[i] == decoded[i + 1] for i in range(len(decoded) - 1)):
        return False

    visited = {start}
    for trace in traces:
        for cell in trace["cells"]:
            if cell in visited:
                return False
            visited.add(cell)

    actual_boundary_indices = {i for i, trace in enumerate(traces) if trace["stop"] == "boundary"}
    if actual_boundary_indices != boundary_indices:
        return False

    obstacle_count = int(np.count_nonzero(main_region == OBSTACLE))
    if obstacle_count < 2:
        return False
    if not _changed_output(main_region, decoded, expected):
        return False
    if not _mapping_is_unique(main_region, decoded, expected):
        return False

    counts = Counter(map(int, main_region.ravel()))
    if counts[start_color] != 1:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    commands = _rotating_commands()
    if not _all_four_directions(commands):
        return None

    built = _build_relative_path(commands)
    if built is None:
        return None
    segments, path_cells = built
    path_box = _path_bbox(path_cells)
    path_h = path_box[2] - path_box[0] + 1
    path_w = path_box[3] - path_box[1] + 1

    candidate_sides = _candidate_boundary_sides(segments, path_cells)
    boundary_sides = _choose_boundary_sides(candidate_sides)
    if boundary_sides is None:
        return None

    strip_width, left_pad, gap = _strip_width(len(commands))
    margin_info = _allocate_margins(path_w, path_h, boundary_sides, strip_width)
    if margin_info is None:
        return None
    margins, main_shape = margin_info

    row_offset = margins["top"] - path_box[0]
    col_offset = margins["left"] - path_box[1]

    boundary_by_segment = _command_sides_from_segments(segments, row_offset, col_offset, main_shape)
    boundary_indices = {
        i for i, side in boundary_by_segment.items() if side in boundary_sides
    }
    if not boundary_indices or len(boundary_indices) == len(commands):
        return None

    obstacles = _obstacle_positions(
        segments,
        row_offset,
        col_offset,
        boundary_indices,
        path_cells,
        main_shape,
    )
    if obstacles is None:
        return None

    separator_color = random.choice(SEPARATOR_POOL)
    start_color_choices = [color for color in START_POOL if color not in (separator_color, GLYPH, OBSTACLE)]
    if not start_color_choices:
        return None
    start_color = random.choice(start_color_choices)
    start = _translate((0, 0), row_offset, col_offset)
    main_region = _render_main_region(main_shape, start, start_color, obstacles)

    inp = _render_input(main_region, commands, separator_color, left_pad, gap)
    if inp.shape[0] > GRID_LIMIT or inp.shape[1] > GRID_LIMIT:
        return None
    return inp if _semantic_ok(inp, commands, boundary_indices) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid glyph-ray program input")
