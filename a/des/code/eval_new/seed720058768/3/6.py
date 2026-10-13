import numpy as np


BG_COLOR = 0
START_COLOR = 3  # green
END_COLOR = 2    # red
MID_COLORS = [1, 4, 5, 6, 7, 8, 9]

SEGMENT_COUNT_MIN = 3
SEGMENT_COUNT_MAX_EXCLUSIVE = 7
SEGMENT_LENGTH_MIN = 3
SEGMENT_LENGTH_MAX_EXCLUSIVE = 6
INPUT_SIZE_MIN = 18
INPUT_SIZE_MAX_EXCLUSIVE = 31
OUTPUT_SPAN_LIMIT = 22
ROTATE_PROBABILITY = 0.5
PATH_RETRIES = 120
PLACEMENT_RETRIES = 120
MAX_GENERATION_ATTEMPTS = 400
PAD = 1

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered two-tone segments into one continuous start-to-end polyline.

    Input structure:
    - The background is black.
    - Non-background objects are short 1-cell-thick horizontal or vertical segments of length 3-5, plus one green singleton start marker and one red singleton end marker.
    - Each segment uses exactly two colors: one endpoint is the segment's key color, while every remaining cell is the body color.

    Rule:
    - Identify the unique chain of segments by color relay: the first segment is the one whose key endpoint is green, and each segment's body color is the next segment's key color, until a segment with red body color ends the chain.
    - Rebuild the path in that order. The first segment keeps its input direction, starting with its green endpoint. Every later segment is placed with its key endpoint one step beyond the previous free end, and its orientation must alternate horizontal/vertical relative to the previous segment. If a segment must turn, rotate it by 90 degrees while preserving the endpoint-to-body order.
    - Choose the unique non-overlapping assembly with no extra side-contacts, then crop the output to the path's bounding box.

    Key insight:
    - The special endpoint color is a receive key and the repeated body color is a send key, so color matching determines the segment order while alternating orientation determines the final geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    segments = _extract_segments(g)
    chain = _order_chain(segments)
    return _assemble_chain(chain)


def _extract_segments(grid: np.ndarray) -> list[dict]:
    bg = _background_color(grid)
    comps = _components4(grid != bg)

    start_markers = []
    end_markers = []
    segments = []

    for coords in comps:
        if len(coords) == 1:
            r, c = coords[0]
            color = int(grid[r, c])
            if color == START_COLOR:
                start_markers.append((r, c))
            elif color == END_COLOR:
                end_markers.append((r, c))
            else:
                raise AssertionError("unexpected singleton color")
            continue
        segments.append(_parse_segment(grid, coords))

    assert len(start_markers) == 1, "expected exactly one green singleton start marker"
    assert len(end_markers) == 1, "expected exactly one red singleton end marker"
    assert segments, "expected at least one segment"
    return segments


def _parse_segment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    if r0 == r1:
        axis = "H"
        ordered = sorted(coords, key=lambda p: p[1])
        pos_v = (0, 1)
        neg_v = (0, -1)
    elif c0 == c1:
        axis = "V"
        ordered = sorted(coords, key=lambda p: p[0])
        pos_v = (1, 0)
        neg_v = (-1, 0)
    else:
        raise AssertionError("segment must be horizontal or vertical")

    colors = [int(grid[r, c]) for r, c in ordered]
    length = len(ordered)
    assert SEGMENT_LENGTH_MIN <= length < SEGMENT_LENGTH_MAX_EXCLUSIVE, "segment length out of range"
    assert len(set(colors)) == 2, "segment must use exactly two colors"

    if colors[0] != colors[1] and all(color == colors[1] for color in colors[1:]):
        key_color = int(colors[0])
        body_color = int(colors[1])
        orig_v = pos_v
    elif colors[-1] != colors[-2] and all(color == colors[0] for color in colors[:-1]):
        key_color = int(colors[-1])
        body_color = int(colors[0])
        orig_v = neg_v
    else:
        raise AssertionError("segment must have one differently colored endpoint")

    return {
        "axis": axis,
        "length": int(length),
        "key": key_color,
        "body": body_color,
        "orig_v": orig_v,
    }


def _order_chain(segments: list[dict]) -> list[dict]:
    by_key = {}
    for seg in segments:
        assert seg["key"] not in by_key, "key colors must identify segments uniquely"
        by_key[seg["key"]] = seg

    ordered = []
    used = set()
    current_key = START_COLOR

    while True:
        assert current_key in by_key, "broken color relay chain"
        seg = by_key[current_key]
        ident = id(seg)
        assert ident not in used, "relay chain must not loop"
        used.add(ident)
        ordered.append(seg)
        current_key = int(seg["body"])
        if current_key == END_COLOR:
            break

    assert len(ordered) == len(segments), "every segment must belong to the unique relay chain"
    return ordered


def _assemble_chain(chain: list[dict]) -> np.ndarray:
    first = chain[0]
    first_v = tuple(first["orig_v"])
    first_cells = _segment_cells((0, 0), first_v, int(first["length"]))

    occupied = {}
    for i, cell in enumerate(first_cells):
        occupied[cell] = int(first["key"]) if i == 0 else int(first["body"])

    solutions: list[dict[tuple[int, int], int]] = []

    def dfs(index: int, prev_v: tuple[int, int], prev_free: tuple[int, int]) -> None:
        if len(solutions) > 1:
            return
        if index == len(chain):
            solutions.append(dict(occupied))
            return

        seg = chain[index]
        required_axis = "V" if _axis(prev_v) == "H" else "H"
        candidate_vs = _candidate_vectors(seg, required_axis)

        for v in candidate_vs:
            key_pos = _add(prev_free, v)
            cells = _segment_cells(key_pos, v, int(seg["length"]))
            if not _placement_is_clean(cells, occupied, key_pos, prev_free):
                continue

            for i, cell in enumerate(cells):
                occupied[cell] = int(seg["key"]) if i == 0 else int(seg["body"])

            dfs(index + 1, v, cells[-1])

            for cell in cells:
                del occupied[cell]

    dfs(1, first_v, first_cells[-1])
    assert len(solutions) == 1, "path assembly must be unique"
    return _render_occupied(solutions[0])


def _candidate_vectors(seg: dict, required_axis: str) -> list[tuple[int, int]]:
    orig_v = tuple(seg["orig_v"])
    if _axis(orig_v) == required_axis:
        return [orig_v]

    out = []
    for v in (_rot_cw(orig_v), _rot_ccw(orig_v)):
        if _axis(v) == required_axis and v not in out:
            out.append(v)
    assert out, "segment must have at least one valid orientation"
    return out


def _placement_is_clean(
    cells: list[tuple[int, int]],
    occupied: dict[tuple[int, int], int],
    key_pos: tuple[int, int],
    prev_free: tuple[int, int],
) -> bool:
    if len(set(cells)) != len(cells):
        return False
    allowed = frozenset((key_pos, prev_free))

    for cell in cells:
        if cell in occupied:
            return False
        for nbr in _neighbors4(cell):
            if nbr not in occupied:
                continue
            if frozenset((cell, nbr)) == allowed:
                continue
            return False
    return True


def _render_occupied(occupied: dict[tuple[int, int], int]) -> np.ndarray:
    rows = [r for r, _ in occupied]
    cols = [c for _, c in occupied]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BG_COLOR, dtype=int)
    for (r, c), color in occupied.items():
        out[r - min_r, c - min_c] = int(color)
    return out


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _axis(v: tuple[int, int]) -> str:
    return "H" if v[0] == 0 else "V"


def _rot_cw(v: tuple[int, int]) -> tuple[int, int]:
    return (v[1], -v[0])


def _rot_ccw(v: tuple[int, int]) -> tuple[int, int]:
    return (-v[1], v[0])


def _segment_cells(start: tuple[int, int], v: tuple[int, int], length: int) -> list[tuple[int, int]]:
    return [(start[0] + step * v[0], start[1] + step * v[1]) for step in range(length)]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")


def _gen_attempt() -> np.ndarray | None:
    seg_count = int(RNG.integers(SEGMENT_COUNT_MIN, SEGMENT_COUNT_MAX_EXCLUSIVE))
    lengths = [int(x) for x in RNG.integers(SEGMENT_LENGTH_MIN, SEGMENT_LENGTH_MAX_EXCLUSIVE, size=seg_count)]

    mid_count = seg_count - 1
    mids = [int(x) for x in RNG.choice(MID_COLORS, size=mid_count, replace=False)]
    chain_colors = [START_COLOR, *mids, END_COLOR]

    specs = []
    for i in range(seg_count):
        specs.append(
            {
                "length": lengths[i],
                "key": int(chain_colors[i]),
                "body": int(chain_colors[i + 1]),
            }
        )

    if not _sample_output_vectors(specs):
        return None

    later = list(range(1, seg_count))
    rotate_flags = RNG.random(len(later)) < ROTATE_PROBABILITY
    if later and not rotate_flags.any():
        rotate_flags[int(RNG.integers(0, len(later)))] = True
    if len(later) >= 2 and rotate_flags.all():
        rotate_flags[int(RNG.integers(0, len(later)))] = False

    specs[0]["orig_v"] = specs[0]["out_v"]
    for idx, seg_index in enumerate(later):
        out_v = tuple(specs[seg_index]["out_v"])
        if rotate_flags[idx]:
            specs[seg_index]["orig_v"] = (_rot_cw(out_v) if RNG.random() < 0.5 else _rot_ccw(out_v))
        else:
            specs[seg_index]["orig_v"] = out_v

    target = _render_from_vectors(specs, "out_v")
    assert max(target.shape) <= 30, "generated target exceeded ARC size limit"

    grid = _scatter_objects(specs)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(solved, target):
        return None
    return grid


def _sample_output_vectors(specs: list[dict]) -> bool:
    for _ in range(PATH_RETRIES):
        occupied: dict[tuple[int, int], int] = {}
        min_r = max_r = min_c = max_c = 0

        first_v = DIRS4[int(RNG.integers(0, len(DIRS4)))]
        specs[0]["out_v"] = first_v
        first_cells = _segment_cells((0, 0), first_v, int(specs[0]["length"]))
        for cell in first_cells:
            occupied[cell] = 1
        min_r, max_r, min_c, max_c = _bbox_update(0, 0, 0, 0, first_cells)
        prev_free = first_cells[-1]
        prev_v = first_v
        ok = True

        for i in range(1, len(specs)):
            options = [(1, 0), (-1, 0)] if _axis(prev_v) == "H" else [(0, 1), (0, -1)]
            options = [options[j] for j in RNG.permutation(len(options))]

            chosen_v = None
            chosen_cells = None
            chosen_bbox = None

            for v in options:
                key_pos = _add(prev_free, v)
                cells = _segment_cells(key_pos, v, int(specs[i]["length"]))
                if not _placement_is_clean(cells, occupied, key_pos, prev_free):
                    continue

                cand_min_r, cand_max_r, cand_min_c, cand_max_c = _bbox_update(
                    min_r, max_r, min_c, max_c, cells
                )
                if cand_max_r - cand_min_r + 1 > OUTPUT_SPAN_LIMIT:
                    continue
                if cand_max_c - cand_min_c + 1 > OUTPUT_SPAN_LIMIT:
                    continue

                chosen_v = v
                chosen_cells = cells
                chosen_bbox = (cand_min_r, cand_max_r, cand_min_c, cand_max_c)
                break

            if chosen_v is None:
                ok = False
                break

            specs[i]["out_v"] = chosen_v
            for cell in chosen_cells:
                occupied[cell] = 1
            min_r, max_r, min_c, max_c = chosen_bbox
            prev_free = chosen_cells[-1]
            prev_v = chosen_v

        if ok:
            return True
    return False


def _render_from_vectors(specs: list[dict], vector_key: str) -> np.ndarray:
    occupied: dict[tuple[int, int], int] = {}
    prev_free = None

    for i, seg in enumerate(specs):
        v = tuple(seg[vector_key])
        if i == 0:
            key_pos = (0, 0)
        else:
            key_pos = _add(prev_free, v)
            assert _placement_is_clean(
                _segment_cells(key_pos, v, int(seg["length"])),
                occupied,
                key_pos,
                prev_free,
            ), "reference path must stay simple"

        cells = _segment_cells(key_pos, v, int(seg["length"]))
        for j, cell in enumerate(cells):
            occupied[cell] = int(seg["key"]) if j == 0 else int(seg["body"])
        prev_free = cells[-1]

    return _render_occupied(occupied)


def _bbox_update(
    min_r: int,
    max_r: int,
    min_c: int,
    max_c: int,
    cells: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return (
        min(min_r, min(rows)),
        max(max_r, max(rows)),
        min(min_c, min(cols)),
        max(max_c, max(cols)),
    )


def _scatter_objects(specs: list[dict]) -> np.ndarray | None:
    h = int(RNG.integers(INPUT_SIZE_MIN, INPUT_SIZE_MAX_EXCLUSIVE))
    w = int(RNG.integers(INPUT_SIZE_MIN, INPUT_SIZE_MAX_EXCLUSIVE))
    grid = np.full((h, w), BG_COLOR, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    objects = []
    for seg in specs:
        cells, colors, shape = _segment_sprite(int(seg["length"]), tuple(seg["orig_v"]), int(seg["key"]), int(seg["body"]))
        objects.append({"cells": cells, "colors": colors, "shape": shape})
    objects.append({"cells": [(0, 0)], "colors": [START_COLOR], "shape": (1, 1)})
    objects.append({"cells": [(0, 0)], "colors": [END_COLOR], "shape": (1, 1)})

    order = sorted(range(len(objects)), key=lambda i: (-len(objects[i]["cells"]), i))
    for obj_index in order:
        obj = objects[obj_index]
        oh, ow = obj["shape"]
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = int(RNG.integers(0, h - oh + 1))
            left = int(RNG.integers(0, w - ow + 1))
            translated = [(top + r, left + c) for r, c in obj["cells"]]
            if any(blocked[r, c] for r, c in translated):
                continue
            for (r, c), color in zip(translated, obj["colors"]):
                grid[r, c] = int(color)
            _block_cells(blocked, translated)
            placed = True
            break
        if not placed:
            return None

    return grid


def _segment_sprite(
    length: int,
    v: tuple[int, int],
    key_color: int,
    body_color: int,
) -> tuple[list[tuple[int, int]], list[int], tuple[int, int]]:
    if v == (0, 1):
        cells = [(0, c) for c in range(length)]
        colors = [key_color] + [body_color] * (length - 1)
        shape = (1, length)
    elif v == (0, -1):
        cells = [(0, c) for c in range(length)]
        colors = [body_color] * (length - 1) + [key_color]
        shape = (1, length)
    elif v == (1, 0):
        cells = [(r, 0) for r in range(length)]
        colors = [key_color] + [body_color] * (length - 1)
        shape = (length, 1)
    elif v == (-1, 0):
        cells = [(r, 0) for r in range(length)]
        colors = [body_color] * (length - 1) + [key_color]
        shape = (length, 1)
    else:
        raise AssertionError("invalid unit vector")
    return cells, colors, shape


def _block_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - PAD), min(h, r + PAD + 1)):
            for cc in range(max(0, c - PAD), min(w, c + PAD + 1)):
                blocked[rr, cc] = True

