from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox, crop_bbox, find_components, get_bg, in_bounds, paste


BG = 0
PATH_COLOR = 5
GLYPH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
DIR_ORDER = ("up", "right", "down", "left")
LENGTHS = (6, 7)
CANVAS_SIZE = 30
MAX_GENERATION_ATTEMPTS = 800
_SEEN_INPUT_SIGNATURES: set[str] = set()


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns))


BASE_TEMPLATES = {
    6: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=int,
    ),
    7: np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 1, 1],
        ],
        dtype=int,
    ),
}

GLYPH_TEMPLATES: dict[tuple[str, int], np.ndarray] = {}
for length, base in BASE_TEMPLATES.items():
    for turns, direction in enumerate(DIR_ORDER):
        GLYPH_TEMPLATES[(direction, length)] = _rotate_cw(base, turns)


def _point_to_coords_chebyshev(point: tuple[int, int], coords: np.ndarray) -> int:
    rr = np.abs(coords[:, 0] - int(point[0]))
    cc = np.abs(coords[:, 1] - int(point[1]))
    return int(np.min(np.maximum(rr, cc)))


def _box_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    return {(r, c) for r in range(r0, r0 + 3) for c in range(c0, c0 + 3)}


def _box_with_margin(top_left: tuple[int, int], shape: tuple[int, int], margin: int = 1) -> set[tuple[int, int]]:
    r0, c0 = map(int, top_left)
    h, w = map(int, shape)
    cells = set()
    for r in range(r0 - margin, r0 + h + margin):
        for c in range(c0 - margin, c0 + w + margin):
            if in_bounds((CANVAS_SIZE, CANVAS_SIZE), r, c):
                cells.add((r, c))
    return cells


def _template_cells(top_left: tuple[int, int], template: np.ndarray) -> np.ndarray:
    r0, c0 = map(int, top_left)
    pts = []
    for r, c in np.argwhere(template != 0):
        pts.append((r0 + int(r), c0 + int(c)))
    return np.array(pts, dtype=int)


def _decode_direction(patch: np.ndarray, bg: int) -> str:
    mask = patch != int(bg)
    top = int(mask[0, :].sum())
    bottom = int(mask[2, :].sum())
    left = int(mask[:, 0].sum())
    right = int(mask[:, 2].sum())
    candidates = []
    if top == 1 and bottom >= 2:
        candidates.append("up")
    if bottom == 1 and top >= 2:
        candidates.append("down")
    if left == 1 and right >= 2:
        candidates.append("left")
    if right == 1 and left >= 2:
        candidates.append("right")
    assert len(candidates) == 1, f"expected one arrow direction, got {candidates}"
    return candidates[0]


def _extract_glyphs(grid: np.ndarray) -> tuple[tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    start_pts = np.argwhere(g == PATH_COLOR)
    assert start_pts.shape == (1, 2), "expected one grey start pixel"
    start = (int(start_pts[0, 0]), int(start_pts[0, 1]))

    glyphs: list[dict] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {bg, PATH_COLOR}:
            continue
        b = comp["bbox"]
        assert b is not None
        r0, c0, r1, c1 = map(int, b)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "glyphs must occupy a 3x3 bbox"
        patch = crop_bbox(g, b)
        direction = _decode_direction(patch, bg)
        length = int(comp["area"])
        glyphs.append(
            {
                "bbox": b,
                "coords": comp["coords"].astype(int),
                "direction": direction,
                "length": length,
                "color": color,
            }
        )
    assert glyphs, "expected at least one glyph"
    return start, glyphs


def _nearest_unused_glyph(current: tuple[int, int], glyphs: list[dict], used: set[int]) -> int:
    scored = []
    for idx, glyph in enumerate(glyphs):
        if idx in used:
            continue
        dist = _point_to_coords_chebyshev(current, glyph["coords"])
        scored.append((dist, idx))
    assert scored, "expected an unused glyph"
    scored.sort()
    assert len(scored) == 1 or scored[0][0] < scored[1][0], "nearest glyph must be unique"
    return int(scored[0][1])


def _draw_segment(out: np.ndarray, current: tuple[int, int], direction: str, length: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    bg = int(get_bg(out))
    r, c = map(int, current)
    for _ in range(int(length)):
        r += int(dr)
        c += int(dc)
        assert in_bounds(out.shape, r, c), "segment exits the grid"
        if int(out[r, c]) in {bg, PATH_COLOR}:
            out[r, c] = PATH_COLOR
    return int(r), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a gray polyline by repeatedly reading the nearest unused 3x3 arrow glyph from the current endpoint.

    Input structure:
    - The grid has a uniform background, one gray start pixel, and several separated monochrome glyphs.
    - Every glyph occupies a 3x3 bounding box and looks like a cardinal arrow.
    - The glyphs are mutually non-adjacent and do not touch the start pixel.

    Rule:
    - Start at the gray pixel.
    - Repeatedly select the unused glyph whose filled pixels are at the smallest Chebyshev distance from the current endpoint.
    - The glyph's arrow direction gives the next segment direction, and the glyph's pixel count gives the segment length.
    - Draw that many gray cells, update the endpoint, and continue until all glyphs are consumed; keep the glyphs unchanged.

    Key insight:
    - The glyphs do not have a fixed global reading order: every new segment changes which instruction is nearest next, so the polyline itself determines the decoding sequence.
    """
    g = np.asarray(grid, dtype=int)
    start, glyphs = _extract_glyphs(g)

    out = g.copy()
    out[start] = PATH_COLOR
    current = start
    used: set[int] = set()
    while len(used) < len(glyphs):
        idx = _nearest_unused_glyph(current, glyphs, used)
        glyph = glyphs[idx]
        current = _draw_segment(out, current, glyph["direction"], glyph["length"])
        used.add(idx)
    return out


def _simulate_path(instructions: list[tuple[str, int]]) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    current = (0, 0)
    endpoints = [current]
    visited = {current}
    path_cells = {current}
    for direction, length in instructions:
        dr, dc = DIRS[direction]
        for _ in range(int(length)):
            nxt = (int(current[0] + dr), int(current[1] + dc))
            if nxt in visited:
                raise ValueError("path self-intersects")
            visited.add(nxt)
            path_cells.add(nxt)
            current = nxt
        endpoints.append(current)
    return endpoints, path_cells


def _direction_changes(instructions: list[tuple[str, int]]) -> int:
    return sum(1 for i in range(1, len(instructions)) if instructions[i][0] != instructions[i - 1][0])


def _build_instruction_sequence() -> tuple[list[tuple[str, int]], list[tuple[int, int]], set[tuple[int, int]]]:
    for _ in range(400):
        n_steps = random.randint(3, 4)
        instructions = []
        for i in range(n_steps):
            options = list(DIR_ORDER)
            if i > 0:
                prev = instructions[-1][0]
                opposite = DIR_ORDER[(DIR_ORDER.index(prev) + 2) % 4]
                options = [d for d in options if d != prev and d != opposite]
            direction = random.choice(options)
            instructions.append((direction, random.choice(LENGTHS)))

        if _direction_changes(instructions) < 2:
            continue

        try:
            endpoints, path_cells = _simulate_path(instructions)
        except ValueError:
            continue

        rs = [r for r, _ in path_cells]
        cs = [c for _r, c in path_cells]
        if max(rs) - min(rs) > 15 or max(cs) - min(cs) > 15:
            continue
        if len({d in ("up", "down") for d, _ in instructions}) == 1:
            continue
        return instructions, endpoints, path_cells
    raise RuntimeError("failed to build a non-degenerate instruction sequence")


def _translate_cells(
    endpoints: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]]:
    rs = [r for r, _ in path_cells]
    cs = [c for _r, c in path_cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    r_shift = random.randint(4 - min_r, CANVAS_SIZE - 5 - max_r)
    c_shift = random.randint(4 - min_c, CANVAS_SIZE - 5 - max_c)
    endpoints2 = [(int(r + r_shift), int(c + c_shift)) for r, c in endpoints]
    path2 = {(int(r + r_shift), int(c + c_shift)) for r, c in path_cells}
    return endpoints2, path2


def _candidate_positions(
    template: np.ndarray,
    forbidden_boxes: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], np.ndarray]]:
    candidates: list[tuple[tuple[int, int], np.ndarray]] = []
    for r0 in range(CANVAS_SIZE - 2):
        for c0 in range(CANVAS_SIZE - 2):
            box = _box_cells((r0, c0))
            if box & forbidden_boxes:
                continue
            if box & path_cells:
                continue
            coords = _template_cells((r0, c0), template)
            candidates.append(((r0, c0), coords))
    random.shuffle(candidates)
    return candidates


def _place_glyphs(
    instructions: list[tuple[str, int]],
    endpoints: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> list[dict] | None:
    start = endpoints[0]
    forbidden_boxes = _box_with_margin(start, (1, 1), margin=1)
    placed: list[dict | None] = [None] * len(instructions)

    for idx in range(len(instructions) - 1, -1, -1):
        direction, length = instructions[idx]
        template = GLYPH_TEMPLATES[(direction, length)]
        future = [g for g in placed[idx + 1 :] if g is not None]
        best_choices = []
        for top_left, coords in _candidate_positions(template, forbidden_boxes, path_cells):
            d_cur = _point_to_coords_chebyshev(endpoints[idx], coords)
            if future:
                future_dists = [_point_to_coords_chebyshev(endpoints[idx], g["coords"]) for g in future]
                if not all(d_cur < d for d in future_dists):
                    continue
            best_choices.append((d_cur, top_left, coords, template))

        if not best_choices:
            return None

        best_choices.sort(key=lambda item: item[0])
        cutoff = min(best_choices[0][0] + 2, best_choices[-1][0])
        options = [item for item in best_choices if item[0] <= cutoff]
        d_cur, top_left, coords, template = random.choice(options)
        placed[idx] = {
            "top_left": top_left,
            "coords": coords,
            "direction": direction,
            "length": length,
            "color": random.choice(GLYPH_COLORS),
            "template": template,
            "distance": d_cur,
        }
        forbidden_boxes |= _box_with_margin(top_left, (3, 3), margin=1)

    return [g for g in placed if g is not None]


def _dynamic_order(start: tuple[int, int], glyphs: list[dict]) -> list[int]:
    order = []
    current = start
    used: set[int] = set()
    while len(used) < len(glyphs):
        idx = _nearest_unused_glyph(current, glyphs, used)
        order.append(idx)
        dr, dc = DIRS[glyphs[idx]["direction"]]
        current = (int(current[0] + dr * glyphs[idx]["length"]), int(current[1] + dc * glyphs[idx]["length"]))
        used.add(idx)
    return order


def _naive_orders(start: tuple[int, int], glyphs: list[dict]) -> list[list[int]]:
    def anchor(i: int) -> tuple[int, int]:
        if "top_left" in glyphs[i]:
            return tuple(map(int, glyphs[i]["top_left"]))
        r0, c0, _r1, _c1 = map(int, glyphs[i]["bbox"])
        return int(r0), int(c0)

    row_major = sorted(range(len(glyphs)), key=anchor)
    col_major = sorted(range(len(glyphs)), key=lambda i: (anchor(i)[1], anchor(i)[0]))
    start_dist = sorted(
        range(len(glyphs)),
        key=lambda i: (_point_to_coords_chebyshev(start, glyphs[i]["coords"]), anchor(i)),
    )
    return [row_major, col_major, start_dist]


def _render_example(
    start: tuple[int, int],
    glyphs: list[dict],
    path_cells: set[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((CANVAS_SIZE, CANVAS_SIZE), BG, dtype=int)
    inp[start] = PATH_COLOR
    for glyph in glyphs:
        patch = (glyph["template"] * int(glyph["color"])).astype(int)
        paste(inp, patch, glyph["top_left"][0], glyph["top_left"][1], transparent=0)

    out = inp.copy()
    for r, c in path_cells:
        out[int(r), int(c)] = PATH_COLOR
    return inp, out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    occ = (inp != BG) | (out != BG)
    b = bbox(occ)
    assert b is not None
    r0, c0, r1, c1 = map(int, b)
    extra_top = random.randint(0, min(3, r0))
    extra_left = random.randint(0, min(3, c0))
    extra_bottom = random.randint(0, min(3, inp.shape[0] - 1 - r1))
    extra_right = random.randint(0, min(3, inp.shape[1] - 1 - c1))
    rr0 = r0 - extra_top
    cc0 = c0 - extra_left
    rr1 = r1 + extra_bottom
    cc1 = c1 + extra_right
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _validate_semantics(
    inp: np.ndarray,
    expected: np.ndarray,
    instructions: list[tuple[str, int]],
) -> bool:
    try:
        start, glyphs = _extract_glyphs(inp)
        got = main(inp)
    except Exception:
        return False

    if not np.array_equal(got, expected):
        return False
    if len(glyphs) != len(instructions):
        return False

    current = start
    used: set[int] = set()
    for step, (direction, length) in enumerate(instructions):
        scores = []
        for idx, glyph in enumerate(glyphs):
            if idx in used:
                continue
            scores.append((_point_to_coords_chebyshev(current, glyph["coords"]), idx))
        scores.sort()
        if len(scores) > 1 and scores[0][0] >= scores[1][0]:
            return False
        glyph = glyphs[scores[0][1]]
        if glyph["direction"] != direction or glyph["length"] != length:
            return False
        dr, dc = DIRS[direction]
        current = (int(current[0] + dr * length), int(current[1] + dc * length))
        used.add(scores[0][1])

    if _dynamic_order(start, glyphs) in _naive_orders(start, glyphs):
        return False
    if _direction_changes(instructions) < 2:
        return False
    if not (inp.ndim == 2 and expected.ndim == 2 and max(inp.shape) <= 30 and max(expected.shape) <= 30):
        return False
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    instructions, endpoints, path_cells = _build_instruction_sequence()
    endpoints, path_cells = _translate_cells(endpoints, path_cells)
    glyphs = _place_glyphs(instructions, endpoints, path_cells)
    if glyphs is None:
        return None

    if _dynamic_order(endpoints[0], glyphs) != list(range(len(glyphs))):
        return None
    if _dynamic_order(endpoints[0], glyphs) in _naive_orders(endpoints[0], glyphs):
        return None

    inp, expected = _render_example(endpoints[0], glyphs, path_cells)
    inp, expected = _crop_pair(inp, expected)
    if not _validate_semantics(inp, expected, instructions):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        pair = _gen_attempt()
        if pair is None:
            continue
        inp, _expected = pair
        sig = hashlib.md5(inp.tobytes() + str(inp.shape).encode()).hexdigest()
        if sig in _SEEN_INPUT_SIGNATURES:
            continue
        _SEEN_INPUT_SIGNATURES.add(sig)
        return inp
    raise RuntimeError("failed to generate a valid input")
