import random
from collections import Counter, deque

import numpy as np


BG = 0
BORDER = 5
OBJECT_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield (r - 1, c)
    if r + 1 < h:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < w:
        yield (r, c + 1)


def _most_common(values) -> int:
    return int(Counter(int(v) for v in values).most_common(1)[0][0])


def _outer_ring(grid: np.ndarray) -> np.ndarray:
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[1:-1, 0]
    right = grid[1:-1, -1]
    return np.concatenate([top, bottom, left, right])


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(2, area + 1):
        if area % h != 0:
            continue
        w = area // h
        if w >= 2:
            pairs.append((h, w))
    return pairs


def _components_by_color(grid: np.ndarray, excluded: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in excluded or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def _detect_notches(grid: np.ndarray, bg: int, border: int) -> list[dict]:
    h, w = grid.shape
    notches: list[dict] = []

    c = 1
    while c < w - 1:
        if int(grid[0, c]) != bg:
            c += 1
            continue
        c0 = c
        while c + 1 < w - 1 and int(grid[0, c + 1]) == bg:
            c += 1
        c1 = c
        depth = 0
        while (
            depth < h
            and bool(np.all(grid[depth, c0 : c1 + 1] == bg))
            and int(grid[depth, c0 - 1]) == border
            and int(grid[depth, c1 + 1]) == border
        ):
            depth += 1
        if depth > 0:
            notches.append(
                {
                    "edge": "up",
                    "r0": 0,
                    "r1": depth - 1,
                    "c0": c0,
                    "c1": c1,
                    "area": depth * (c1 - c0 + 1),
                    "shape": (depth, c1 - c0 + 1),
                }
            )
        c += 1

    c = 1
    while c < w - 1:
        if int(grid[h - 1, c]) != bg:
            c += 1
            continue
        c0 = c
        while c + 1 < w - 1 and int(grid[h - 1, c + 1]) == bg:
            c += 1
        c1 = c
        depth = 0
        while (
            depth < h
            and bool(np.all(grid[h - 1 - depth, c0 : c1 + 1] == bg))
            and int(grid[h - 1 - depth, c0 - 1]) == border
            and int(grid[h - 1 - depth, c1 + 1]) == border
        ):
            depth += 1
        if depth > 0:
            notches.append(
                {
                    "edge": "down",
                    "r0": h - depth,
                    "r1": h - 1,
                    "c0": c0,
                    "c1": c1,
                    "area": depth * (c1 - c0 + 1),
                    "shape": (depth, c1 - c0 + 1),
                }
            )
        c += 1

    r = 1
    while r < h - 1:
        if int(grid[r, 0]) != bg:
            r += 1
            continue
        r0 = r
        while r + 1 < h - 1 and int(grid[r + 1, 0]) == bg:
            r += 1
        r1 = r
        depth = 0
        while (
            depth < w
            and bool(np.all(grid[r0 : r1 + 1, depth] == bg))
            and int(grid[r0 - 1, depth]) == border
            and int(grid[r1 + 1, depth]) == border
        ):
            depth += 1
        if depth > 0:
            notches.append(
                {
                    "edge": "left",
                    "r0": r0,
                    "r1": r1,
                    "c0": 0,
                    "c1": depth - 1,
                    "area": depth * (r1 - r0 + 1),
                    "shape": (r1 - r0 + 1, depth),
                }
            )
        r += 1

    r = 1
    while r < h - 1:
        if int(grid[r, w - 1]) != bg:
            r += 1
            continue
        r0 = r
        while r + 1 < h - 1 and int(grid[r + 1, w - 1]) == bg:
            r += 1
        r1 = r
        depth = 0
        while (
            depth < w
            and bool(np.all(grid[r0 : r1 + 1, w - 1 - depth] == bg))
            and int(grid[r0 - 1, w - 1 - depth]) == border
            and int(grid[r1 + 1, w - 1 - depth]) == border
        ):
            depth += 1
        if depth > 0:
            notches.append(
                {
                    "edge": "right",
                    "r0": r0,
                    "r1": r1,
                    "c0": w - depth,
                    "c1": w - 1,
                    "area": depth * (r1 - r0 + 1),
                    "shape": (r1 - r0 + 1, depth),
                }
            )
        r += 1

    return notches


def _find_arrow_head(coords: set[tuple[int, int]], h: int, w: int) -> tuple[tuple[int, int], list[tuple[int, int]], str]:
    degrees = {cell: sum((nbr in coords) for nbr in _neighbors4(cell[0], cell[1], h, w)) for cell in coords}
    found = []
    for r, c in coords:
        neighbors = [nbr for nbr in _neighbors4(r, c, h, w) if nbr in coords]
        if len(neighbors) != 3:
            continue
        leaves = [nbr for nbr in neighbors if degrees[nbr] == 1]
        if len(leaves) != 2:
            continue
        shaft = [nbr for nbr in neighbors if nbr not in leaves]
        if len(shaft) != 1:
            continue
        (sr, sc) = shaft[0]
        vertical_leaves = sorted(leaves) == [(r - 1, c), (r + 1, c)] if 0 < r < h - 1 else False
        horizontal_leaves = sorted(leaves) == [(r, c - 1), (r, c + 1)] if 0 < c < w - 1 else False
        if vertical_leaves and sr == r:
            direction = "right" if sc == c - 1 else "left"
            found.append(((r, c), leaves, direction))
        elif horizontal_leaves and sc == c:
            direction = "down" if sr == r - 1 else "up"
            found.append(((r, c), leaves, direction))
    if len(found) != 1:
        raise ValueError("expected exactly one arrow head")
    return found[0]


def _extract_block(component: dict, grid_shape: tuple[int, int]) -> dict:
    h, w = grid_shape
    coords = {tuple(map(int, cell)) for cell in component["coords"]}
    head, leaves, direction = _find_arrow_head(coords, h, w)
    dr, dc = DIRS[OPPOSITE[direction]]
    arrow = {head, *leaves}
    cur = (head[0] + dr, head[1] + dc)
    while cur in coords:
        if direction in ("left", "right"):
            perp = [(cur[0] - 1, cur[1]), (cur[0] + 1, cur[1])]
        else:
            perp = [(cur[0], cur[1] - 1), (cur[0], cur[1] + 1)]
        if any(p in coords for p in perp):
            break
        arrow.add(cur)
        cur = (cur[0] + dr, cur[1] + dc)
    block = coords - arrow
    if not block:
        raise ValueError("empty block")
    rs = [r for r, _ in block]
    cs = [c for _, c in block]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if len(block) != (r1 - r0 + 1) * (c1 - c0 + 1):
        raise ValueError("block must be solid rectangle")
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr, cc) not in block:
                raise ValueError("block rectangle has hole")
    return {
        "color": int(component["color"]),
        "component_coords": component["coords"],
        "direction": direction,
        "lane": head[1] if direction in ("up", "down") else head[0],
        "block_bbox": (r0, c0, r1, c1),
        "area": len(block),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow-linked colored block into the matching edge notch and reshape it to fill the notch.

    Input structure:
    - The background is black, and a single-pixel-thick grey border surrounds the canvas while leaving rectangular black notches on the edges.
    - Each movable object is monochrome and 4-connected: a solid rectangular block is attached to a same-colored arrow whose shaft is horizontal or vertical and whose head is a 3-pixel wedge.
    - Every notch area matches exactly one block area, and the arrow's lane points to a unique notch on that edge.

    Rule:
    - Detect each arrow direction from its wedge head, then separate the arrow pixels from the attached solid rectangle.
    - Erase the original object and choose the unique notch on the indicated edge whose area matches the block and whose opening contains the arrow lane.
    - Fill that notch's black cells with the block color; because the block is monochrome, any required scan-order rearrangement becomes a full monochrome fill.

    Key insight:
    - The arrow head identifies both direction and motion lane, so area plus edge alignment is enough to pair each rectangular block with its destination notch even when the notch shape differs.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    bg = _most_common(g.flatten())
    border = _most_common(v for v in _outer_ring(g) if int(v) != bg)
    notches = _detect_notches(g, bg, border)
    by_edge: dict[str, list[dict]] = {k: [] for k in DIRS}
    for notch in notches:
        by_edge[notch["edge"]].append(notch)

    components = _components_by_color(g, excluded={bg, border})
    objects = [_extract_block(comp, g.shape) for comp in components]

    out = g.copy()
    for obj in objects:
        coords = obj["component_coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for obj in objects:
        candidates = []
        for notch in by_edge[obj["direction"]]:
            if int(notch["area"]) != int(obj["area"]):
                continue
            if obj["direction"] in ("up", "down"):
                if int(notch["c0"]) <= int(obj["lane"]) <= int(notch["c1"]):
                    candidates.append(notch)
            else:
                if int(notch["r0"]) <= int(obj["lane"]) <= int(notch["r1"]):
                    candidates.append(notch)
        if len(candidates) != 1:
            raise ValueError("object-to-notch match must be unique")
        notch = candidates[0]
        out[notch["r0"] : notch["r1"] + 1, notch["c0"] : notch["c1"] + 1] = int(obj["color"])
    return out


def _draw_frame(h: int, w: int) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = BORDER
    grid[-1, :] = BORDER
    grid[:, 0] = BORDER
    grid[:, -1] = BORDER
    return grid


def _notch_shape_options(direction: str, h: int, w: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    if direction in ("up", "down"):
        for depth in range(2, min(6, h - 7)):
            for width in range(2, min(8, w - 5)):
                shapes.append((depth, width))
    else:
        for height in range(2, min(8, h - 5)):
            for depth in range(2, min(6, w - 7)):
                shapes.append((height, depth))
    return shapes


def _block_shape_options(area: int, direction: str) -> list[tuple[int, int]]:
    options = []
    for bh, bw in _factor_pairs(area):
        if bh > 7 or bw > 7:
            continue
        if direction in ("up", "down") and bw >= 3:
            options.append((bh, bw))
        if direction in ("left", "right") and bh >= 3:
            options.append((bh, bw))
    return options


def _carve_notch(grid: np.ndarray, direction: str, top: int, left: int, shape: tuple[int, int]) -> dict:
    h, w = grid.shape
    if direction == "up":
        depth, width = shape
        c0 = left
        c1 = left + width - 1
        grid[0:depth, c0 : c1 + 1] = BG
        grid[0:depth, c0 - 1] = BORDER
        grid[0:depth, c1 + 1] = BORDER
        return {"edge": direction, "r0": 0, "r1": depth - 1, "c0": c0, "c1": c1, "area": depth * width, "shape": shape}
    if direction == "down":
        depth, width = shape
        c0 = left
        c1 = left + width - 1
        grid[h - depth : h, c0 : c1 + 1] = BG
        grid[h - depth : h, c0 - 1] = BORDER
        grid[h - depth : h, c1 + 1] = BORDER
        return {"edge": direction, "r0": h - depth, "r1": h - 1, "c0": c0, "c1": c1, "area": depth * width, "shape": shape}
    if direction == "left":
        height, depth = shape
        r0 = top
        r1 = top + height - 1
        grid[r0 : r1 + 1, 0:depth] = BG
        grid[r0 - 1, 0:depth] = BORDER
        grid[r1 + 1, 0:depth] = BORDER
        return {"edge": direction, "r0": r0, "r1": r1, "c0": 0, "c1": depth - 1, "area": height * depth, "shape": shape}
    height, depth = shape
    r0 = top
    r1 = top + height - 1
    grid[r0 : r1 + 1, w - depth : w] = BG
    grid[r0 - 1, w - depth : w] = BORDER
    grid[r1 + 1, w - depth : w] = BORDER
    return {"edge": direction, "r0": r0, "r1": r1, "c0": w - depth, "c1": w - 1, "area": height * depth, "shape": shape}


def _place_notches(grid: np.ndarray, directions: list[str]) -> list[dict] | None:
    h, w = grid.shape
    specs = []
    for direction in directions:
        choices = []
        for shape in _notch_shape_options(direction, h, w):
            area = int(shape[0] * shape[1])
            if not _block_shape_options(area, direction):
                continue
            choices.append(shape)
        if not choices:
            return None
        random.shuffle(choices)
        placed = None
        for shape in choices:
            if direction in ("up", "down"):
                width = shape[1]
                if w - width - 6 < 0:
                    continue
                c0 = random.randint(3, w - width - 4)
                placed = _carve_notch(grid, direction, 0, c0, shape)
                break
            height = shape[0]
            if h - height - 6 < 0:
                continue
            r0 = random.randint(3, h - height - 4)
            placed = _carve_notch(grid, direction, r0, 0, shape)
            break
        if placed is None:
            return None
        specs.append(placed)
    return specs


def _object_cells(direction: str, block_r0: int, block_c0: int, bh: int, bw: int, offset: int, shaft_len: int) -> set[tuple[int, int]]:
    cells = {(r, c) for r in range(block_r0, block_r0 + bh) for c in range(block_c0, block_c0 + bw)}
    if direction == "up":
        lane = block_c0 + offset
        head_r = block_r0 - shaft_len
        for r in range(head_r, block_r0):
            cells.add((r, lane))
        cells.add((head_r, lane - 1))
        cells.add((head_r, lane + 1))
        return cells
    if direction == "down":
        lane = block_c0 + offset
        head_r = block_r0 + bh - 1 + shaft_len
        for r in range(block_r0 + bh, head_r + 1):
            cells.add((r, lane))
        cells.add((head_r, lane - 1))
        cells.add((head_r, lane + 1))
        return cells
    if direction == "left":
        lane = block_r0 + offset
        head_c = block_c0 - shaft_len
        for c in range(head_c, block_c0):
            cells.add((lane, c))
        cells.add((lane - 1, head_c))
        cells.add((lane + 1, head_c))
        return cells
    lane = block_r0 + offset
    head_c = block_c0 + bw - 1 + shaft_len
    for c in range(block_c0 + bw, head_c + 1):
        cells.add((lane, c))
    cells.add((lane - 1, head_c))
    cells.add((lane + 1, head_c))
    return cells


def _touches_occupied(cells: set[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return True
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return True
    return False


def _try_place_object(grid: np.ndarray, occupied: np.ndarray, notch: dict, color: int) -> tuple[bool, bool]:
    h, w = grid.shape
    direction = notch["edge"]
    area = int(notch["area"])
    notch_shape = tuple(int(x) for x in notch["shape"])
    block_choices = _block_shape_options(area, direction)
    if not block_choices:
        return False, False
    prefer_diff = random.random() < 0.5
    if prefer_diff:
        diff_choices = [shape for shape in block_choices if shape != notch_shape]
        if diff_choices:
            block_choices = diff_choices
    else:
        same_choices = [shape for shape in block_choices if shape == notch_shape]
        if same_choices:
            block_choices = same_choices
    for _ in range(120):
        bh, bw = random.choice(block_choices)
        shaft_len = random.randint(2, 4)
        if direction in ("up", "down"):
            offset = random.randint(1, bw - 2)
            lane_options = [
                lane
                for lane in range(int(notch["c0"]), int(notch["c1"]) + 1)
                if 1 <= lane - offset and lane - offset + bw - 1 <= w - 2
            ]
            if not lane_options:
                continue
            lane = random.choice(lane_options)
            block_c0 = lane - offset
            if direction == "up":
                head_rows = list(range(int(notch["shape"][0]), h - bh - shaft_len))
                if not head_rows:
                    continue
                head_r = random.choice(head_rows)
                block_r0 = head_r + shaft_len
            else:
                head_rows = list(range(bh + shaft_len, h - int(notch["shape"][0])))
                if not head_rows:
                    continue
                head_r = random.choice(head_rows)
                block_r0 = head_r - shaft_len - bh + 1
        else:
            offset = random.randint(1, bh - 2)
            lane_options = [
                lane
                for lane in range(int(notch["r0"]), int(notch["r1"]) + 1)
                if 1 <= lane - offset and lane - offset + bh - 1 <= h - 2
            ]
            if not lane_options:
                continue
            lane = random.choice(lane_options)
            block_r0 = lane - offset
            if direction == "left":
                head_cols = list(range(int(notch["shape"][1]), w - bw - shaft_len))
                if not head_cols:
                    continue
                head_c = random.choice(head_cols)
                block_c0 = head_c + shaft_len
            else:
                head_cols = list(range(bw + shaft_len, w - int(notch["shape"][1])))
                if not head_cols:
                    continue
                head_c = random.choice(head_cols)
                block_c0 = head_c - shaft_len - bw + 1

        cells = _object_cells(direction, block_r0, block_c0, bh, bw, offset, shaft_len)
        if _touches_occupied(cells, occupied):
            continue
        if any(int(grid[r, c]) != BG for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = color
            occupied[r, c] = True
        return True, (bh, bw) != notch_shape
    return False, False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 26)
    w = random.randint(18, 26)
    directions = random.sample(list(DIRS), k=random.randint(2, 4))
    colors = random.sample(OBJECT_PALETTE, k=len(directions))

    base = _draw_frame(h, w)
    notches = _place_notches(base, directions)
    if notches is None:
        return None

    expected = base.copy()
    for notch, color in zip(notches, colors):
        expected[notch["r0"] : notch["r1"] + 1, notch["c0"] : notch["c1"] + 1] = color

    inp = base.copy()
    occupied = np.zeros((h, w), dtype=bool)
    diff_flags = []
    for notch, color in zip(notches, colors):
        ok, diff = _try_place_object(inp, occupied, notch, color)
        if not ok:
            return None
        diff_flags.append(diff)

    if len(diff_flags) >= 2 and not (any(diff_flags) and any(not x for x in diff_flags)):
        return None
    if np.array_equal(inp, expected):
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    detected = _detect_notches(inp, BG, BORDER)
    if len(detected) != len(notches):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
