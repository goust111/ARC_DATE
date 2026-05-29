# Combo variant of 8b9c3697 (index 14)
# combo_spec: D7/2+D8/2
# Name: 颜色编码方向多色齿

from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import Color, components_mask, get_bg, paste, random_free_location
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, components_mask, get_bg, paste, random_free_location


RED = int(Color.RED)
BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
GREEN = int(Color.GREEN)
YELLOW = int(Color.YELLOW)
PINK = int(Color.PINK)
GREY = int(Color.GREY)
ORANGE = int(Color.ORANGE)
TEAL = int(Color.TEAL)
MAROON = int(Color.MAROON)


# Tooth color -> motion direction of the matched red singleton.
DIR_BY_TOOTH_COLOR = {
    BLUE: (0, 1),     # enter from the left
    YELLOW: (0, -1),  # enter from the right
    GREEN: (1, 0),    # enter from above
    PINK: (-1, 0),    # enter from below
}

TOOTH_COLORS = tuple(DIR_BY_TOOTH_COLOR.keys())
TOOTH_COUNTS = (2, 3)

GRID_H_RANGE = (19, 24)
GRID_W_RANGE = (23, 29)
BG_COLORS = (GREY, TEAL, MAROON)
WALL_COLORS = (GREY, ORANGE, TEAL, MAROON)

STRUCT_COUNT_RANGE = (2, 3)
SLIDE_RANGE = (3, 6)
STRUCT_PADDING = 2

DISTRACTOR_SHAPES = ((1, 2), (2, 1), (2, 2), (1, 3), (3, 1))
DISTRACTOR_COUNT_RANGE = (2, 5)
MAX_RETRIES = 500


def _structure_width(tooth_count: int) -> int:
    return int(3 * tooth_count + 1)


def _tooth_cols(tooth_count: int) -> list[int]:
    return [int(1 + 3 * idx) for idx in range(tooth_count)]


def _socket_cols(tooth_count: int) -> list[int]:
    return [int(col + 1) for col in _tooth_cols(tooth_count)]


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _singleton_reds(grid: np.ndarray) -> set[tuple[int, int]]:
    singles: set[tuple[int, int]] = set()
    for coords in components_mask(grid == int(RED), connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) == 1:
            singles.add(pts[0])
    return singles


def _scan_structures(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    infos: list[dict] = []

    for tooth_count in sorted(TOOTH_COUNTS, reverse=True):
        width = _structure_width(tooth_count)
        tooth_cols = set(_tooth_cols(tooth_count))
        socket_cols = set(_socket_cols(tooth_count))

        for r0 in range(h - 2):
            for c0 in range(w - width + 1):
                if c0 > 0:
                    if int(grid[r0, c0 - 1]) != int(bg) or int(grid[r0 + 1, c0 - 1]) != int(bg):
                        continue
                if c0 + width < w:
                    if int(grid[r0, c0 + width]) != int(bg) or int(grid[r0 + 1, c0 + width]) != int(bg):
                        continue

                wall_color = int(grid[r0, c0])
                if wall_color in (int(bg), int(RED)):
                    continue

                ok = True
                slots: list[dict] = []
                for rel_c in range(width):
                    top = int(grid[r0, c0 + rel_c])
                    bottom = int(grid[r0 + 1, c0 + rel_c])

                    if rel_c in socket_cols:
                        if top != int(bg) or bottom != int(bg):
                            ok = False
                            break
                        continue

                    if top != int(wall_color):
                        ok = False
                        break

                    if rel_c in tooth_cols:
                        if bottom not in TOOTH_COLORS:
                            ok = False
                            break
                        slots.append(
                            {
                                "tooth_color": int(bottom),
                                "direction": tuple(map(int, DIR_BY_TOOTH_COLOR[int(bottom)])),
                                "target": (int(r0 + 2), int(c0 + rel_c + 1)),
                            }
                        )
                    else:
                        if bottom != int(wall_color):
                            ok = False
                            break

                if not ok:
                    continue

                if any(int(grid[tr, tc]) != int(bg) for tr, tc in (slot["target"] for slot in slots)):
                    continue

                infos.append(
                    {
                        "top_left": (int(r0), int(c0)),
                        "width": int(width),
                        "tooth_count": int(tooth_count),
                        "wall_color": int(wall_color),
                        "slots": slots,
                    }
                )

    infos.sort(key=lambda item: (item["top_left"][0], item["top_left"][1], -item["width"]))
    chosen: list[dict] = []
    occupied = np.zeros(grid.shape, dtype=bool)
    for info in infos:
        r0, c0 = info["top_left"]
        width = int(info["width"])
        probe = occupied[r0:r0 + 2, c0:c0 + width]
        if probe.any():
            continue
        occupied[r0:r0 + 2, c0:c0 + width] = True
        chosen.append(info)

    return chosen


def _find_slot_source(
    grid: np.ndarray,
    bg: int,
    red_singletons: set[tuple[int, int]],
    target: tuple[int, int],
    direction: tuple[int, int],
) -> dict | None:
    h, w = grid.shape
    tr, tc = map(int, target)
    dr, dc = map(int, direction)

    rr = int(tr - dr)
    cc = int(tc - dc)
    slide = 1
    while 0 <= rr < h and 0 <= cc < w:
        val = int(grid[rr, cc])
        if val == int(bg):
            rr -= int(dr)
            cc -= int(dc)
            slide += 1
            continue
        if val == int(RED) and (int(rr), int(cc)) in red_singletons:
            return {
                "source": (int(rr), int(cc)),
                "target": (int(tr), int(tc)),
                "direction": (int(dr), int(dc)),
                "slide": int(slide),
            }
        return None
    return None


def _find_assignments(grid: np.ndarray, bg: int) -> list[dict]:
    red_singletons = _singleton_reds(grid)
    structures = _scan_structures(grid, bg)

    assignments: list[dict] = []
    used_sources: set[tuple[int, int]] = set()
    for structure in structures:
        for slot in structure["slots"]:
            match = _find_slot_source(grid, bg, red_singletons, slot["target"], slot["direction"])
            if match is None:
                continue
            if match["source"] in used_sources:
                continue
            used_sources.add(match["source"])
            assignments.append(match)

    return assignments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored teeth encode the docking direction of individual red pieces.

    Every structure keeps the same canonical comb-like silhouette. Each tooth has its own color, and that color
    determines from which side a red singleton must slide into the tooth's dedicated socket cell:
    - blue: from the left
    - yellow: from the right
    - green: from above
    - pink: from below

    The travelled cells become black. Any red component that is not one of these color-coded matches disappears.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    out = g.copy()
    out[out == int(RED)] = int(bg)

    for item in _find_assignments(g, bg):
        sr, sc = map(int, item["source"])
        tr, tc = map(int, item["target"])
        dr, dc = map(int, item["direction"])
        slide = int(item["slide"])

        for step in range(slide):
            rr = int(sr + step * dr)
            cc = int(sc + step * dc)
            out[rr, cc] = int(BLACK)
        out[tr, tc] = int(RED)

    return out


def _allowed_colors_for_tooth(tooth_count: int, tooth_idx: int) -> tuple[int, ...]:
    allowed = {GREEN, PINK}
    if tooth_idx == 0:
        allowed.add(BLUE)
    if tooth_idx == tooth_count - 1:
        allowed.add(YELLOW)
    return tuple(sorted(allowed))


def _assign_tooth_colors(tooth_counts: list[int]) -> list[tuple[int, ...]] | None:
    positions: list[tuple[int, int, tuple[int, ...]]] = []
    for sid, tooth_count in enumerate(tooth_counts):
        for tid in range(tooth_count):
            positions.append((int(sid), int(tid), _allowed_colors_for_tooth(tooth_count, tid)))

    left_positions = [(sid, tid) for sid, tid, allowed in positions if BLUE in allowed]
    right_positions = [(sid, tid) for sid, tid, allowed in positions if YELLOW in allowed]
    if not left_positions or not right_positions or len(positions) < 4:
        return None

    blue_pos = random.choice(left_positions)
    yellow_choices = [pos for pos in right_positions if pos != blue_pos]
    if not yellow_choices:
        return None
    yellow_pos = random.choice(yellow_choices)

    remaining = [(sid, tid) for sid, tid, _allowed in positions if (sid, tid) not in {blue_pos, yellow_pos}]
    green_pos = random.choice(remaining)
    remaining = [pos for pos in remaining if pos != green_pos]
    pink_pos = random.choice(remaining)

    by_pos: dict[tuple[int, int], int] = {
        blue_pos: int(BLUE),
        yellow_pos: int(YELLOW),
        green_pos: int(GREEN),
        pink_pos: int(PINK),
    }

    tooth_colors: list[list[int]] = []
    for sid, tooth_count in enumerate(tooth_counts):
        row: list[int] = []
        for tid in range(tooth_count):
            pos = (int(sid), int(tid))
            if pos in by_pos:
                row.append(int(by_pos[pos]))
            else:
                row.append(int(random.choice(_allowed_colors_for_tooth(tooth_count, tid))))
        tooth_colors.append(row)

    return [tuple(row) for row in tooth_colors]


def _build_structure_spec(
    tooth_count: int,
    tooth_colors: tuple[int, ...],
    slides: tuple[int, ...],
    bg: int,
    wall_color: int,
) -> dict | None:
    width = _structure_width(tooth_count)
    tooth_cols = _tooth_cols(tooth_count)
    socket_cols = _socket_cols(tooth_count)

    left_pad = max([int(slides[idx]) for idx, color in enumerate(tooth_colors) if color == int(BLUE)] + [1])
    right_pad = max([int(slides[idx]) for idx, color in enumerate(tooth_colors) if color == int(YELLOW)] + [1])
    up_pad = max([int(slides[idx]) for idx, color in enumerate(tooth_colors) if color == int(GREEN)] + [1])
    down_pad = max([int(slides[idx]) for idx, color in enumerate(tooth_colors) if color == int(PINK)] + [1])

    patch_h = int(up_pad + 3 + down_pad)
    patch_w = int(left_pad + width + right_pad)
    base_r = int(up_pad)
    base_c = int(left_pad)

    structure = np.full((patch_h, patch_w), bg, dtype=int)
    input_patch = np.full((patch_h, patch_w), bg, dtype=int)
    output_patch = np.full((patch_h, patch_w), bg, dtype=int)

    structure_mask = np.zeros((patch_h, patch_w), dtype=bool)
    for rel_c in range(width):
        rr = int(base_r)
        cc = int(base_c + rel_c)
        if rel_c not in socket_cols:
            structure[rr, cc] = int(wall_color)
            input_patch[rr, cc] = int(wall_color)
            output_patch[rr, cc] = int(wall_color)
            structure_mask[rr, cc] = True

        rr = int(base_r + 1)
        if rel_c in socket_cols:
            continue
        if rel_c in tooth_cols:
            tooth_idx = tooth_cols.index(rel_c)
            color = int(tooth_colors[tooth_idx])
        else:
            color = int(wall_color)
        structure[rr, cc] = int(color)
        input_patch[rr, cc] = int(color)
        output_patch[rr, cc] = int(color)
        structure_mask[rr, cc] = True

    slots: list[dict] = []
    seen_input: set[tuple[int, int]] = set()
    seen_output_nonbg: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(structure_mask):
        seen_output_nonbg.add((int(rr), int(cc)))

    for tooth_idx, (tooth_col, socket_col, tooth_color, slide) in enumerate(zip(tooth_cols, socket_cols, tooth_colors, slides)):
        target = (int(base_r + 2), int(base_c + socket_col))
        dr, dc = map(int, DIR_BY_TOOTH_COLOR[int(tooth_color)])
        source = (int(target[0] - slide * dr), int(target[1] - slide * dc))

        if not (0 <= source[0] < patch_h and 0 <= source[1] < patch_w):
            return None

        path_cells = [(int(source[0] + step * dr), int(source[1] + step * dc)) for step in range(int(slide))]
        if any(not (0 <= rr < patch_h and 0 <= cc < patch_w) for rr, cc in path_cells):
            return None

        for rr, cc in path_cells:
            if (rr, cc) in seen_output_nonbg:
                return None
        if source in seen_input:
            return None

        input_patch[source] = int(RED)
        seen_input.add(source)

        for rr, cc in path_cells:
            output_patch[rr, cc] = int(BLACK)
            seen_output_nonbg.add((int(rr), int(cc)))
        output_patch[target] = int(RED)
        seen_output_nonbg.add(target)

        slots.append(
            {
                "tooth_index": int(tooth_idx),
                "tooth_color": int(tooth_color),
                "direction": (int(dr), int(dc)),
                "slide": int(slide),
                "source": source,
                "target": target,
            }
        )

    return {
        "input": input_patch,
        "output": output_patch,
        "slots": slots,
        "tooth_colors": tuple(map(int, tooth_colors)),
    }


def _place_structures(specs: list[dict], h: int, w: int, bg: int) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    global_slots: list[dict] = []

    for spec in sorted(specs, key=lambda item: np.count_nonzero(item["output"] != int(bg)), reverse=True):
        try:
            loc = random_free_location(out, spec["output"], transparent=bg, padding=STRUCT_PADDING)
        except ValueError:
            return None

        r0, c0 = map(int, loc)
        paste(inp, spec["input"], r0, c0, transparent=bg)
        paste(out, spec["output"], r0, c0, transparent=bg)

        for slot in spec["slots"]:
            global_slots.append(
                {
                    "tooth_color": int(slot["tooth_color"]),
                    "direction": tuple(map(int, slot["direction"])),
                    "slide": int(slot["slide"]),
                    "source": (int(r0 + slot["source"][0]), int(c0 + slot["source"][1])),
                    "target": (int(r0 + slot["target"][0]), int(c0 + slot["target"][1])),
                }
            )

    return inp, out, global_slots


def _place_distractors(inp: np.ndarray, out: np.ndarray, bg: int, wall_color: int, count: int) -> bool:
    h, w = inp.shape
    shapes = list(DISTRACTOR_SHAPES)
    red_mask = inp == int(RED)

    for _ in range(count):
        blocked = (inp != int(bg)) | (out != int(bg)) | _dilate4(red_mask)
        canvas = np.full((h, w), bg, dtype=int)
        canvas[blocked] = int(wall_color)

        random.shuffle(shapes)
        placed = False
        for rh, rw in shapes:
            patch = np.full((rh, rw), int(RED), dtype=int)
            try:
                loc = random_free_location(canvas, patch, transparent=bg, padding=0)
            except ValueError:
                continue

            rr, cc = map(int, loc)
            paste(inp, patch, rr, cc, transparent=bg)
            red_mask[rr:rr + rh, cc:cc + rw] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bg = int(random.choice(BG_COLORS))
    wall_choices = [color for color in WALL_COLORS if int(color) != int(bg)]
    wall_color = int(random.choice(wall_choices))
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    struct_count = random.randint(*STRUCT_COUNT_RANGE)
    tooth_counts = [int(random.choice(TOOTH_COUNTS)) for _ in range(struct_count)]
    tooth_colors = _assign_tooth_colors(tooth_counts)
    if tooth_colors is None:
        return None

    specs: list[dict] = []
    for tooth_count, colors in zip(tooth_counts, tooth_colors):
        slides = tuple(random.randint(*SLIDE_RANGE) for _ in range(tooth_count))
        spec = _build_structure_spec(tooth_count, colors, slides, bg, wall_color)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_structures(specs, h, w, bg)
    if placed is None:
        return None
    inp, out, slots = placed

    if {int(slot["tooth_color"]) for slot in slots} != set(map(int, TOOTH_COLORS)):
        return None

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    if not _place_distractors(inp, out, bg, wall_color, distractor_count):
        return None

    if not np.array_equal(main(inp), out):
        return None

    assignments = _find_assignments(inp, bg)
    if len(assignments) != len(slots):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for 8b9c3697 combo D7/2+D8/2")
