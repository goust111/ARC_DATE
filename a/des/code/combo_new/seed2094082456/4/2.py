from __future__ import annotations

import random

import numpy as np

BG = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    comps.sort(key=lambda comp: min(comp))
    return comps


def _connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _shift_cells(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((int(rr), int(cc)))
    return expanded


def _decode_arrow(gray_cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], int]:
    pts = [(int(r), int(c)) for r, c in gray_cells]
    n = len(pts)
    if n < 4:
        raise ValueError("arrow is too short to decode unambiguously")

    rows: dict[int, list[int]] = {}
    cols: dict[int, list[int]] = {}
    for r, c in pts:
        rows.setdefault(int(r), []).append(int(c))
        cols.setdefault(int(c), []).append(int(r))

    candidates: list[tuple[int, tuple[int, int]]] = []

    for row, row_cols in rows.items():
        if len(row_cols) != n - 1:
            continue
        ordered = sorted(row_cols)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            continue
        heads = [(r, c) for r, c in pts if r != row]
        if len(heads) != 1:
            continue
        hr, hc = heads[0]
        if abs(hr - row) != 1 or hc not in (ordered[0], ordered[-1]):
            continue
        direction = (0, 1) if hc == ordered[0] else (0, -1)
        candidates.append((len(ordered), direction))

    for col, col_rows in cols.items():
        if len(col_rows) != n - 1:
            continue
        ordered = sorted(col_rows)
        if ordered != list(range(ordered[0], ordered[-1] + 1)):
            continue
        heads = [(r, c) for r, c in pts if c != col]
        if len(heads) != 1:
            continue
        hr, hc = heads[0]
        if abs(hc - col) != 1 or hr not in (ordered[0], ordered[-1]):
            continue
        direction = (1, 0) if hr == ordered[0] else (-1, 0)
        candidates.append((len(ordered), direction))

    if not candidates:
        raise ValueError("no valid arrow found")

    best_len = max(length for length, _ in candidates)
    best_dirs = [direction for length, direction in candidates if length == best_len]
    return best_dirs[0], int(best_len)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate each colored object by the direction and distance encoded by its internal gray arrow.

    Input structure:
    - The background is black.
    - Each foreground object is a separate 4-connected non-black component containing one body color and one gray arrow.
    - The gray arrow is an L-shape: a straight horizontal or vertical shaft plus one orthogonally offset head pixel at one shaft end.

    Rule:
    - For each connected foreground object, isolate its gray arrow and decode the direction from the head-bearing end toward the opposite shaft end.
    - The translation distance equals the shaft length in pixels.
    - Remove the gray arrow, clear the original object location, and move only the non-gray body pixels by that displacement.

    Key insight:
    - The embedded gray arrow is a complete instruction: its endpoint-with-offset gives the movement direction, and the shaft length gives the movement distance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    for comp in _components(g != BG):
        gray_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        if not gray_cells:
            continue
        direction, length = _decode_arrow(gray_cells)
        dr = direction[0] * length
        dc = direction[1] * length
        for r, c in comp:
            color = int(g[r, c])
            if color == GREY:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _base_shape(direction: tuple[int, int], shaft_len: int, rng: random.Random) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    horizontal = direction[0] == 0
    for _ in range(120):
        if horizontal:
            height = rng.randint(4, 8)
            width = rng.randint(shaft_len + 2, min(shaft_len + 6, 10))
            head_side = rng.choice([-1, 1])
            shaft_r = rng.randint(2, height - 2) if head_side == -1 else rng.randint(1, height - 3)
            start_c = rng.randint(1, width - shaft_len - 1)
            end_c = start_c + shaft_len - 1
            head_r = shaft_r + head_side
            head_c = start_c if direction == (0, 1) else end_c
            gray = {(shaft_r, c) for c in range(start_c, end_c + 1)}
            gray.add((head_r, head_c))
            host = {(r, c) for r in range(height) for c in range(width)}
        else:
            height = rng.randint(shaft_len + 2, min(shaft_len + 6, 10))
            width = rng.randint(4, 8)
            head_side = rng.choice([-1, 1])
            shaft_c = rng.randint(2, width - 2) if head_side == -1 else rng.randint(1, width - 3)
            start_r = rng.randint(1, height - shaft_len - 1)
            end_r = start_r + shaft_len - 1
            head_r = start_r if direction == (1, 0) else end_r
            head_c = shaft_c + head_side
            gray = {(r, shaft_c) for r in range(start_r, end_r + 1)}
            gray.add((head_r, head_c))
            host = {(r, c) for r in range(height) for c in range(width)}

        color = host - gray
        if not _connected(color):
            continue
        return color, gray
    raise RuntimeError("failed to build a base shape")


def _add_tabs(color: set[tuple[int, int]], host_h: int, host_w: int, rng: random.Random) -> set[tuple[int, int]]:
    out = set(color)
    sides = ["top", "bottom", "left", "right"]
    rng.shuffle(sides)
    for side in sides[: rng.randint(0, 2)]:
        if side in {"top", "bottom"}:
            length = rng.randint(1, min(3, host_w))
            start = rng.randint(0, host_w - length)
            row = -1 if side == "top" else host_h
            for c in range(start, start + length):
                out.add((row, c))
        else:
            length = rng.randint(1, min(3, host_h))
            start = rng.randint(0, host_h - length)
            col = -1 if side == "left" else host_w
            for r in range(start, start + length):
                out.add((r, col))
    return out


def _build_object(direction: tuple[int, int], rng: random.Random) -> dict[str, object]:
    shaft_len = rng.randint(3, 6)
    color, gray = _base_shape(direction, shaft_len, rng)
    host_h = max(r for r, _ in color | gray) + 1
    host_w = max(c for _, c in color | gray) + 1
    color = _add_tabs(color, host_h, host_w, rng)

    # Normalize both sets together so their shared coordinate system is preserved.
    min_r = min(r for r, _ in color | gray)
    min_c = min(c for _, c in color | gray)
    color = {(r - min_r, c - min_c) for r, c in color}
    gray = {(r - min_r, c - min_c) for r, c in gray}

    if not _connected(color):
        raise RuntimeError("generated color shape became disconnected")
    if not _connected(color | gray):
        raise RuntimeError("generated object became disconnected")

    shift = (direction[0] * shaft_len, direction[1] * shaft_len)
    input_rel = color | gray
    output_rel = _shift_cells(color, shift[0], shift[1])
    return {
        "color": color,
        "gray": gray,
        "input_rel": input_rel,
        "output_rel": output_rel,
        "shift": shift,
        "direction": direction,
        "shaft_len": shaft_len,
        "area": len(input_rel),
    }


def _place_object(
    spec: dict[str, object],
    h: int,
    w: int,
    input_forbidden: set[tuple[int, int]],
    output_forbidden: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    input_rel = spec["input_rel"]
    output_rel = spec["output_rel"]

    min_in_r = min(r for r, _ in input_rel)
    max_in_r = max(r for r, _ in input_rel)
    min_in_c = min(c for _, c in input_rel)
    max_in_c = max(c for _, c in input_rel)
    min_out_r = min(r for r, _ in output_rel)
    max_out_r = max(r for r, _ in output_rel)
    min_out_c = min(c for _, c in output_rel)
    max_out_c = max(c for _, c in output_rel)

    top_min = max(-min_in_r, -min_out_r)
    top_max = min(h - 1 - max_in_r, h - 1 - max_out_r)
    left_min = max(-min_in_c, -min_out_c)
    left_max = min(w - 1 - max_in_c, w - 1 - max_out_c)
    if top_min > top_max or left_min > left_max:
        return None

    for _ in range(180):
        top = rng.randint(top_min, top_max)
        left = rng.randint(left_min, left_max)
        input_abs = _shift_cells(input_rel, top, left)
        output_abs = _shift_cells(output_rel, top, left)
        if any(cell in input_forbidden for cell in input_abs):
            continue
        if any(cell in output_forbidden for cell in output_abs):
            continue
        color_abs = _shift_cells(spec["color"], top, left)
        gray_abs = _shift_cells(spec["gray"], top, left)
        return color_abs, gray_abs, output_abs
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 30)
    w = rng.randint(16, 30)
    max_objects = min(5, len(PALETTE), max(2, (h * w) // 120))
    count = rng.randint(2, max_objects)

    directions = [rng.choice(DIRS) for _ in range(count)]
    if len(set(directions)) == 1:
        directions[rng.randrange(count)] = rng.choice([d for d in DIRS if d != directions[0]])

    specs = [_build_object(direction, rng) for direction in directions]
    specs.sort(key=lambda spec: int(spec["area"]), reverse=True)
    colors = rng.sample(PALETTE, count)

    placed: list[tuple[dict[str, object], int, set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]] = []
    input_forbidden: set[tuple[int, int]] = set()
    output_forbidden: set[tuple[int, int]] = set()

    for spec, color_id in zip(specs, colors):
        placed_obj = _place_object(spec, h, w, input_forbidden, output_forbidden, rng)
        if placed_obj is None:
            return None
        color_abs, gray_abs, output_abs = placed_obj
        placed.append((spec, color_id, color_abs, gray_abs, output_abs))
        input_forbidden |= _expand8(color_abs | gray_abs, h, w)
        output_forbidden |= _expand8(output_abs, h, w)

    inp = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    for _spec, color_id, color_abs, gray_abs, output_abs in placed:
        for r, c in color_abs:
            inp[r, c] = int(color_id)
        for r, c in gray_abs:
            inp[r, c] = GREY
        for r, c in output_abs:
            expected[r, c] = int(color_id)

    if np.count_nonzero(inp == BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
