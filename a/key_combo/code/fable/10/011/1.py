from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CORNER_NAMES = ("tl", "tr", "bl", "br")

GEN_ATTEMPTS = 500
STACK_ATTEMPTS = 200
PRIMARY_LENGTHS = (2, 3)
SECONDARY_LENGTHS = (1, 2)
SECONDARY_PROB = 0.55
TOP_HEIGHTS = (4, 5)
TOP_WIDTHS = (4, 5)
EXT_CHOICES = (1, 2)
MAX_STACK_SPAN = 11
MARGINS = (1, 2, 3)
GAPS = (2, 3, 4)


def _corners(bbox: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _shift_bbox(bbox: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return r0 + dr, c0 + dc, r1 + dr, c1 + dc


def _normalize_bboxes(
    bboxes: list[tuple[int, int, int, int]]
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]]:
    min_r = min(r0 for r0, _c0, _r1, _c1 in bboxes)
    min_c = min(c0 for _r0, c0, _r1, _c1 in bboxes)
    shifted = [_shift_bbox(bbox, -min_r, -min_c) for bbox in bboxes]
    max_r = max(r1 for _r0, _c0, r1, _c1 in shifted)
    max_c = max(c1 for _r0, _c0, _r1, c1 in shifted)
    return shifted, (max_r + 1, max_c + 1)


def _point_interior(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = map(int, point)
    r0, c0, r1, c1 = map(int, bbox)
    return r0 < r < r1 and c0 < c < c1


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return max(ar0, br0) <= min(ar1, br1) and max(ac0, bc0) <= min(ac1, bc1)


def _fill_rect(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _render_output(shape: tuple[int, int], rects: list[dict], order: list[int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for idx in order:
        spec = rects[int(idx)]
        _fill_rect(out, spec["bbox"], int(spec["color"]))
    return out


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    rects: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {bg}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] in (3, 4), "each color must have three or four corner pixels"
        rows = sorted({int(r) for r, _c in coords})
        cols = sorted({int(c) for _r, c in coords})
        assert len(rows) == 2 and len(cols) == 2, "corner pixels must span exactly two rows and two cols"
        bbox = (rows[0], cols[0], rows[1], cols[1])
        assert rows[1] - rows[0] + 1 >= 3 and cols[1] - cols[0] + 1 >= 3

        corner_map = _corners(bbox)
        full_set = set(corner_map.values())
        present = {tuple(map(int, rc)) for rc in coords}
        assert present <= full_set, "pixels must lie on the rectangle corners"
        missing = list(full_set - present)
        assert len(missing) in (0, 1), "each color may miss at most one corner"
        rects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "missing": None if not missing else missing[0],
            }
        )
    assert rects, "expected at least one rectangle generator"
    return rects


def _cover_edges(rects: list[dict]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, rect in enumerate(rects):
        missing = rect["missing"]
        if missing is None:
            continue
        parents = [
            j
            for j, other in enumerate(rects)
            if i != j and _point_interior(missing, other["bbox"])
        ]
        assert parents, "a missing corner must lie inside at least one upper rectangle"
        for j in parents:
            edges.add((i, j))
    return edges


def _topological_order(rects: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    n = len(rects)
    outgoing: list[list[int]] = [[] for _ in range(n)]
    indeg = [0] * n
    for src, dst in edges:
        outgoing[int(src)].append(int(dst))
        indeg[int(dst)] += 1

    key = lambda i: (*rects[int(i)]["bbox"], int(rects[int(i)]["color"]))
    ready = sorted([i for i in range(n) if indeg[i] == 0], key=key)
    order: list[int] = []
    while ready:
        cur = ready.pop(0)
        order.append(int(cur))
        for nxt in sorted(outgoing[cur], key=key):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=key)

    assert len(order) == n, "cover graph must be acyclic"
    return order


def _reachability(rects: list[dict], edges: set[tuple[int, int]]) -> list[list[bool]]:
    n = len(rects)
    adj: list[list[int]] = [[] for _ in range(n)]
    for src, dst in edges:
        adj[int(src)].append(int(dst))
    reach = [[False] * n for _ in range(n)]
    for start in range(n):
        stack = list(adj[start])
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            reach[start][cur] = True
            stack.extend(adj[cur])
    return reach


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing solid rectangles and paint them in the unique cover order encoded by missing corners.

    Input structure:
    - A black background contains several colors, one color per rectangle generator.
    - Each color shows either the four corners of one axis-aligned solid rectangle, or exactly three corners.
    - Any missing corner lies strictly inside a higher rectangle, so missing corners encode the overlap order.

    Rule:
    - Reconstruct each rectangle from its corner set; three corners uniquely determine the fourth.
    - If rectangle A misses a corner and that corner lies inside rectangle B, then B is above A.
    - Take the transitive closure of those cover relations and draw all solid rectangles from bottom to top.

    Key insight:
    - The input never shows the rectangles themselves; the one absent corner per lower layer is the only clue that recovers the stack order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rects = _extract_rectangles(g)
    edges = _cover_edges(rects)
    reach = _reachability(rects, edges)

    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            if not _overlap(rects[i]["bbox"], rects[j]["bbox"]):
                continue
            ij = reach[i][j]
            ji = reach[j][i]
            assert ij ^ ji, "every overlapping pair must have a unique layer order"

    order = _topological_order(rects, edges)
    return _render_output(g.shape, rects, order)


def _hidden_corner_names(
    bbox: tuple[int, int, int, int], higher_bboxes: list[tuple[int, int, int, int]]
) -> list[str]:
    hidden: list[str] = []
    for name, point in _corners(bbox).items():
        if any(_point_interior(point, other) for other in higher_bboxes):
            hidden.append(name)
    return hidden


def _make_lower_bbox(
    upper: tuple[int, int, int, int], corner_name: str
) -> tuple[int, int, int, int]:
    ur0, uc0, ur1, uc1 = map(int, upper)
    miss_r = random.randint(ur0 + 1, ur1 - 1)
    miss_c = random.randint(uc0 + 1, uc1 - 1)
    row_ext = random.choice(EXT_CHOICES)
    col_ext = random.choice(EXT_CHOICES)

    if corner_name == "tl":
        return miss_r, miss_c, ur1 + row_ext, uc1 + col_ext
    if corner_name == "tr":
        return miss_r, uc0 - col_ext, ur1 + row_ext, miss_c
    if corner_name == "bl":
        return ur0 - row_ext, miss_c, miss_r, uc1 + col_ext
    if corner_name == "br":
        return ur0 - row_ext, uc0 - col_ext, miss_r, miss_c
    raise ValueError(f"unknown corner name: {corner_name}")


def _orientation_sequence(n: int) -> list[str]:
    if n <= len(CORNER_NAMES) and random.random() < 0.75:
        return random.sample(CORNER_NAMES, n)
    seq: list[str] = []
    prev = None
    for _ in range(n):
        choices = [name for name in CORNER_NAMES if name != prev]
        nxt = random.choice(choices)
        seq.append(nxt)
        prev = nxt
    return seq


def _stack_semantics_ok(rects_top_down: list[tuple[int, int, int, int]]) -> bool:
    if len(set(rects_top_down)) != len(rects_top_down):
        return False
    _, shape = _normalize_bboxes(rects_top_down)
    if max(shape) > MAX_STACK_SPAN:
        return False

    for idx, bbox in enumerate(rects_top_down):
        higher = rects_top_down[:idx]
        hidden = _hidden_corner_names(bbox, higher)
        if idx == 0:
            if hidden:
                return False
            continue
        if len(hidden) != 1:
            return False
        missing = _corners(bbox)[hidden[0]]
        parent_hits = [hb for hb in higher if _point_interior(missing, hb)]
        if len(parent_hits) != 1 or parent_hits[0] != rects_top_down[idx - 1]:
            return False
    return True


def _generate_stack(length: int) -> list[tuple[int, int, int, int]]:
    for _ in range(STACK_ATTEMPTS):
        top = (0, 0, random.choice(TOP_HEIGHTS) - 1, random.choice(TOP_WIDTHS) - 1)
        rects_top_down = [top]
        for corner_name in _orientation_sequence(length - 1):
            rects_top_down.append(_make_lower_bbox(rects_top_down[-1], corner_name))
        rects_top_down, _shape = _normalize_bboxes(rects_top_down)
        if _stack_semantics_ok(rects_top_down):
            return rects_top_down
    raise RuntimeError("failed to generate a valid rectangle stack")


def _shift_stack(
    stack: list[tuple[int, int, int, int]], dr: int, dc: int
) -> list[tuple[int, int, int, int]]:
    return [_shift_bbox(bbox, dr, dc) for bbox in stack]


def _layout_stacks(
    stacks_top_down: list[list[tuple[int, int, int, int]]]
) -> tuple[list[list[tuple[int, int, int, int]]], tuple[int, int]] | None:
    stacks = stacks_top_down[:]
    random.shuffle(stacks)
    shapes = [_normalize_bboxes(stack)[1] for stack in stacks]

    if len(stacks) == 1:
        h, w = shapes[0]
        top = random.choice(MARGINS)
        left = random.choice(MARGINS)
        bottom = random.choice(MARGINS)
        right = random.choice(MARGINS)
        extra_h = random.randint(0, 2)
        extra_w = random.randint(0, 2)
        H = h + top + bottom + extra_h
        W = w + left + right + extra_w
        if max(H, W) > 30:
            return None
        dr = top + random.randint(0, extra_h)
        dc = left + random.randint(0, extra_w)
        return [_shift_stack(stacks[0], dr, dc)], (H, W)

    for mode in random.sample(["horizontal", "vertical"], 2):
        gap = random.choice(GAPS)
        top = random.choice(MARGINS)
        left = random.choice(MARGINS)
        bottom = random.choice(MARGINS)
        right = random.choice(MARGINS)
        extra = random.randint(0, 2)

        if mode == "horizontal":
            H = top + bottom + max(h for h, _w in shapes) + extra
            W = left + right + sum(w for _h, w in shapes) + gap
            if max(H, W) > 30:
                continue
            placed: list[list[tuple[int, int, int, int]]] = []
            cur_c = left
            for stack, (h, w) in zip(stacks, shapes):
                spare = H - top - bottom - h
                dr = top + random.randint(0, spare)
                placed.append(_shift_stack(stack, dr, cur_c))
                cur_c += w + gap
            return placed, (H, W)

        H = top + bottom + sum(h for h, _w in shapes) + gap
        W = left + right + max(w for _h, w in shapes) + extra
        if max(H, W) > 30:
            continue
        placed = []
        cur_r = top
        for stack, (h, w) in zip(stacks, shapes):
            spare = W - left - right - w
            dc = left + random.randint(0, spare)
            placed.append(_shift_stack(stack, cur_r, dc))
            cur_r += h + gap
        return placed, (H, W)
    return None


def _expected_missing_by_color(stack_specs_top_down: list[list[dict]]) -> dict[int, tuple[int, int] | None]:
    result: dict[int, tuple[int, int] | None] = {}
    for stack in stack_specs_top_down:
        higher: list[tuple[int, int, int, int]] = []
        for spec in stack:
            hidden = _hidden_corner_names(spec["bbox"], higher)
            assert len(hidden) in (0, 1)
            result[int(spec["color"])] = None if not hidden else _corners(spec["bbox"])[hidden[0]]
            higher.append(spec["bbox"])
    return result


def _anti_shortcuts_ok(rects: list[dict], expected: np.ndarray) -> bool:
    edges = _cover_edges(rects)
    order = _topological_order(rects, edges)
    heuristics = [
        list(reversed(order)),
        sorted(range(len(rects)), key=lambda i: _bbox_area(rects[i]["bbox"])),
        sorted(range(len(rects)), key=lambda i: -_bbox_area(rects[i]["bbox"])),
        sorted(range(len(rects)), key=lambda i: (*rects[i]["bbox"][:2], int(rects[i]["color"]))),
    ]
    for candidate in heuristics:
        if np.array_equal(_render_output(expected.shape, rects, candidate), expected):
            return False
    return True


def _validate_generated_case(
    inp: np.ndarray, expected: np.ndarray, stack_specs_top_down: list[list[dict]]
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    intended = {
        int(spec["color"]): tuple(spec["bbox"])
        for stack in stack_specs_top_down
        for spec in stack
    }
    intended_missing = _expected_missing_by_color(stack_specs_top_down)

    try:
        parsed = _extract_rectangles(inp)
        parsed_by_color = {int(rect["color"]): rect for rect in parsed}
        if set(parsed_by_color) != set(intended):
            return False
        for color, bbox in intended.items():
            if tuple(parsed_by_color[color]["bbox"]) != bbox:
                return False
            if parsed_by_color[color]["missing"] != intended_missing[color]:
                return False

        edges = _cover_edges(parsed)
        reach = _reachability(parsed, edges)
        for i in range(len(parsed)):
            for j in range(i + 1, len(parsed)):
                if not _overlap(parsed[i]["bbox"], parsed[j]["bbox"]):
                    continue
                if not (reach[i][j] ^ reach[j][i]):
                    return False
    except Exception:
        return False

    return _anti_shortcuts_ok(parsed, expected)


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    lengths = [random.choice(PRIMARY_LENGTHS)]
    if random.random() < SECONDARY_PROB:
        lengths.append(random.choice(SECONDARY_LENGTHS))

    try:
        stacks_top_down = [_generate_stack(length) for length in lengths]
    except RuntimeError:
        return None

    layout = _layout_stacks(stacks_top_down)
    if layout is None:
        return None
    placed_stacks, shape = layout

    total_rects = sum(len(stack) for stack in placed_stacks)
    colors = random.sample(PALETTE, total_rects)

    stack_specs_top_down: list[list[dict]] = []
    color_idx = 0
    for stack in placed_stacks:
        specs: list[dict] = []
        for bbox in stack:
            specs.append({"bbox": tuple(bbox), "color": colors[color_idx]})
            color_idx += 1
        stack_specs_top_down.append(specs)

    flat_specs: list[dict] = []
    for stack in stack_specs_top_down:
        for spec in stack:
            flat_specs.append({"bbox": tuple(spec["bbox"]), "color": int(spec["color"])})

    expected_order: list[int] = []
    flat_index = 0
    stack_id_to_indices: list[list[int]] = []
    for stack in stack_specs_top_down:
        indices: list[int] = []
        for _spec in stack:
            indices.append(flat_index)
            flat_index += 1
        stack_id_to_indices.append(indices)
    for indices in stack_id_to_indices:
        expected_order.extend(reversed(indices))

    expected = _render_output(shape, flat_specs, expected_order)
    inp = np.full(shape, BG, dtype=int)

    for stack in stack_specs_top_down:
        higher: list[tuple[int, int, int, int]] = []
        for spec in stack:
            hidden = _hidden_corner_names(spec["bbox"], higher)
            if len(hidden) not in (0, 1):
                return None
            hidden_name = None if not hidden else hidden[0]
            for name, point in _corners(spec["bbox"]).items():
                if name == hidden_name:
                    continue
                r, c = point
                if inp[r, c] != BG:
                    return None
                inp[r, c] = int(spec["color"])
            higher.append(spec["bbox"])

    if np.array_equal(expected, inp):
        return None
    if not _validate_generated_case(inp, expected, stack_specs_top_down):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid missing-corner rectangle input")
