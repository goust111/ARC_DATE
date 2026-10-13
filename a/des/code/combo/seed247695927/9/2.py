from __future__ import annotations

from collections import deque
import random

import numpy as np


SOLID_MASK = np.ones((3, 3), dtype=bool)
X_MASK = np.array(
    [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ],
    dtype=bool,
)
DIAMOND_MASK = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=bool,
)

NEXT_SHAPE = {"solid": "x", "x": "diamond", "diamond": "solid"}
PREV_SHAPE = {"solid": "diamond", "x": "solid", "diamond": "x"}
SHAPE_MASKS = {"solid": SOLID_MASK, "x": X_MASK, "diamond": DIAMOND_MASK}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _monochrome_components(grid: np.ndarray, bg: int, connectivity: int) -> list[list[tuple[int, int]]]:
    out = []
    for color in [int(c) for c in np.unique(grid) if int(c) != int(bg)]:
        out.extend(_components(grid == color, connectivity))
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _classify_token(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None
    color = int(grid[coords[0]])
    if any(int(grid[r, c]) != color for r, c in coords):
        return None

    mask = np.zeros((3, 3), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True

    for name, shape_mask in SHAPE_MASKS.items():
        if np.array_equal(mask, shape_mask):
            return {
                "shape": name,
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
            }
    return None


def _paint_shape(grid: np.ndarray, box: tuple[int, int, int, int], color: int, shape: str, bg: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    mask = SHAPE_MASKS[shape]
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(color)


def _slot_for_endpoint(endpoint: tuple[int, int], neighbor: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    h, w = shape
    er, ec = endpoint
    nr, nc = neighbor
    dr, dc = er - nr, ec - nc
    center_r = er + 2 * dr
    center_c = ec + 2 * dc
    r0, c0 = center_r - 1, center_c - 1
    r1, c1 = center_r + 1, center_c + 1
    if 0 <= r0 and 0 <= c0 and r1 < h and c1 < w:
        return (r0, c0, r1, c1)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transmit endpoint tokens through each wire and cycle their shapes.

    Input structure:
    - Background pixels dominate the grid.
    - Non-background pixels form 1-cell-wide wire components and 3x3 tokens.
    - Tokens are monochrome 3x3 solid, X, or diamond patterns placed one cell
      beyond a wire endpoint.

    Rule:
    - A token at a wire endpoint is replaced in place by the previous shape in
      the cycle solid -> X -> diamond -> solid.
    - The next shape in that same cycle is painted at the opposite endpoint's
      token slot.
    - Wire pixels stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)

    token_components = []
    for comp in _monochrome_components(g, bg=bg, connectivity=8):
        token = _classify_token(g, comp)
        if token is not None:
            token_components.append(token)

    token_by_box = {token["bbox"]: token for token in token_components}
    used_boxes: set[tuple[int, int, int, int]] = set()

    wire_mask = g != bg
    for token in token_components:
        for r, c in token["coords"]:
            wire_mask[r, c] = False

    wire_components = _components(wire_mask, connectivity=4)
    assert wire_components, "expected at least one wire component"

    out = g.copy()
    for token in token_components:
        r0, c0, r1, c1 = token["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for comp in wire_components:
        comp_set = set(comp)
        endpoints = []
        neighbors = {}
        for cell in comp:
            nbs = [(r + dr, c + dc) for (r, c) in [cell] for dr, dc in DIR4 if (r + dr, c + dc) in comp_set]
            if len(nbs) == 1:
                endpoints.append(cell)
                neighbors[cell] = nbs[0]
        assert len(endpoints) == 2, "each wire must have exactly two endpoints"

        slot_info = []
        for endpoint in endpoints:
            slot = _slot_for_endpoint(endpoint, neighbors[endpoint], (h, w))
            assert slot is not None, "endpoint token slot must fit in bounds"
            slot_info.append((endpoint, slot))

        present = []
        for endpoint, slot in slot_info:
            token = token_by_box.get(slot)
            if token is not None:
                present.append((endpoint, slot, token))
                used_boxes.add(slot)

        assert len(present) == 1, "each wire must have exactly one attached token"
        src_endpoint, src_box, token = present[0]
        dst_box = [slot for endpoint, slot in slot_info if endpoint != src_endpoint][0]
        assert dst_box not in token_by_box, "destination slot must be empty in the input"

        _paint_shape(out, src_box, token["color"], PREV_SHAPE[token["shape"]], bg)
        _paint_shape(out, dst_box, token["color"], NEXT_SHAPE[token["shape"]], bg)

    assert used_boxes == set(token_by_box), "every token must be attached to exactly one wire endpoint"
    return out


GRID_MIN = 18
GRID_MAX = 26
WIRE_COUNT_RANGE = (1, 4)
PATH_MIN_LEN = 6
MAX_PATH_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(cell: tuple[int, int], shape: tuple[int, int]):
    r, c = cell
    h, w = shape
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _find_path(free: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        nbs = list(_neighbors4(cur, free.shape))
        random.shuffle(nbs)
        for nb in nbs:
            if not free[nb] or nb in prev:
                continue
            prev[nb] = cur
            q.append(nb)
    if end not in prev:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _box_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _paint_mask(grid: np.ndarray, box: tuple[int, int, int, int], color: int, mask: np.ndarray) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = int(color)


def _sample_endpoint(rng_r: range, rng_c: range) -> tuple[int, int]:
    return random.choice(list(rng_r)), random.choice(list(rng_c))


def _make_wire_candidate(blocked: np.ndarray) -> tuple[list[tuple[int, int]], tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    h, w = blocked.shape
    inner_rows = range(3, h - 3)
    inner_cols = range(3, w - 3)
    if len(inner_rows) == 0 or len(inner_cols) == 0:
        return None

    for _ in range(MAX_PATH_ATTEMPTS):
        start = _sample_endpoint(inner_rows, inner_cols)
        end = _sample_endpoint(inner_rows, inner_cols)
        if start == end:
            continue
        if abs(start[0] - end[0]) + abs(start[1] - end[1]) < PATH_MIN_LEN:
            continue

        free = ~blocked
        free[start] = True
        free[end] = True
        path = _find_path(free, start, end)
        if path is None or len(path) < PATH_MIN_LEN:
            continue

        start_slot = _slot_for_endpoint(path[0], path[1], blocked.shape)
        end_slot = _slot_for_endpoint(path[-1], path[-2], blocked.shape)
        if start_slot is None or end_slot is None:
            continue

        path_cells = set(path)
        start_box_cells = _box_cells(start_slot)
        end_box_cells = _box_cells(end_slot)
        if path_cells & start_box_cells or path_cells & end_box_cells or start_box_cells & end_box_cells:
            continue

        footprint = path_cells | start_box_cells | end_box_cells
        if any(blocked[r, c] for r, c in footprint):
            continue

        return path, start_slot, end_slot
    return None


def _manual_output(inp: np.ndarray, wires: list[dict], bg: int) -> np.ndarray:
    out = inp.copy()
    for wire in wires:
        src_box = wire["src_box"]
        dst_box = wire["dst_box"]
        color = wire["token_color"]
        shape = wire["token_shape"]
        sr0, sc0, sr1, sc1 = src_box
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = bg
        _paint_mask(out, src_box, color, SHAPE_MASKS[PREV_SHAPE[shape]])
        _paint_mask(out, dst_box, color, SHAPE_MASKS[NEXT_SHAPE[shape]])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randrange(10)
    palette = [c for c in range(10) if c != bg]
    wire_color = random.choice(palette)
    token_palette = [c for c in palette if c != wire_color]
    if len(token_palette) < 2:
        return None

    inp = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    wires = []

    for _ in range(random.randint(*WIRE_COUNT_RANGE)):
        candidate = _make_wire_candidate(blocked)
        if candidate is None:
            if wires:
                break
            return None

        path, box_a, box_b = candidate
        src_is_a = random.random() < 0.5
        src_box, dst_box = (box_a, box_b) if src_is_a else (box_b, box_a)
        token_shape = random.choice(["solid", "x"])
        token_color = random.choice(token_palette)

        for r, c in path:
            inp[r, c] = wire_color
        _paint_mask(inp, src_box, token_color, SHAPE_MASKS[token_shape])

        all_cells = set(path) | _box_cells(box_a) | _box_cells(box_b)
        for r, c in _expand_cells(all_cells, (h, w), radius=1):
            blocked[r, c] = True

        wires.append(
            {
                "src_box": src_box,
                "dst_box": dst_box,
                "token_shape": token_shape,
                "token_color": token_color,
            }
        )

    if not wires:
        return None

    expected = _manual_output(inp, wires, bg)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
