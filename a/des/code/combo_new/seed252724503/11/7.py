import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
AZURE = 8


def _components(mask):
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize(coords):
    rs = [p[0] for p in coords]
    cs = [p[1] for p in coords]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - min(rs), c - min(cs)] = True
    return mask


def _variants(mask):
    vars_ = []
    cur = mask
    for _ in range(4):
        vars_.append(cur)
        vars_.append(np.fliplr(cur))
        cur = np.rot90(cur)
    return vars_


def _room_bboxes(grid):
    boxes = []
    for comp in _components(grid == GREY):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r1 - r0 >= 4 and c1 - c0 >= 4:
            boxes.append((r0, r1, c0, c1))
    boxes.sort()
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each room object from its door side and update the door state from a shape-action legend.

    Input structure:
    - Grey rectangular rooms each have exactly one missing wall pixel acting as a door.
    - Each room contains one colored 4-connected object.
    - Away from the rooms, several reference shapes are paired with a right-adjacent orange or azure marker.

    Rule:
    - Recolor every room object by the room's door side: top to red, right to blue, bottom to yellow, left to green.
    - Normalize each room object's shape and compare it to the legend shapes up to rotation and reflection.
    - If the matched legend marker is orange, fill the room's door gap with grey; if it is azure or unmatched, leave the gap open.

    Key insight:
    - Door position and object shape encode different outputs at the same time: side controls recoloring, while shape controls whether the room structure changes.
    """
    out = grid.copy()
    rooms = _room_bboxes(grid)
    room_mask = np.zeros_like(grid, dtype=bool)
    for r0, r1, c0, c1 in rooms:
        room_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    legends = []
    legend_mask = (~room_mask) & (grid != BLACK) & (grid != GREY) & (grid != ORANGE) & (grid != AZURE)
    for comp in _components(legend_mask):
        mask = _normalize(comp)
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        r0, r1 = min(rs), max(rs)
        c1 = max(cs)
        action = None
        for rr in range(r0, r1 + 1):
            if c1 + 1 < grid.shape[1] and grid[rr, c1 + 1] in (ORANGE, AZURE):
                action = int(grid[rr, c1 + 1])
                break
        legends.append((mask, action))

    door_colors = {"top": RED, "right": BLUE, "bottom": YELLOW, "left": GREEN}
    for r0, r1, c0, c1 in rooms:
        gap = None
        for c in range(c0, c1 + 1):
            if grid[r0, c] == BLACK:
                gap = ("top", r0, c)
            if grid[r1, c] == BLACK:
                gap = ("bottom", r1, c)
        for r in range(r0, r1 + 1):
            if grid[r, c0] == BLACK:
                gap = ("left", r, c0)
            if grid[r, c1] == BLACK:
                gap = ("right", r, c1)
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        pts = np.argwhere((inner != BLACK) & (inner != GREY))
        color = door_colors[gap[0]]
        out[r0 + 1 : r1, c0 + 1 : c1][(inner != BLACK) & (inner != GREY)] = color
        shape = _normalize([tuple(p) for p in pts])
        matched_action = None
        for legend_mask_arr, action in legends:
            if any(v.shape == shape.shape and np.array_equal(v, shape) for v in _variants(legend_mask_arr)):
                matched_action = action
                break
        if matched_action == ORANGE:
            out[gap[1], gap[2]] = GREY
    return out


def _shape_library():
    return [
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    ]


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    grid = np.zeros((28, 28), dtype=int)
    shapes = _shape_library()
    legend_indices = rng.choice(len(shapes), size=int(rng.integers(2, 5)), replace=False)
    legend_actions = rng.choice([ORANGE, AZURE], size=len(legend_indices))
    r = 1
    for idx, action in zip(legend_indices, legend_actions):
        shape = shapes[int(idx)]
        h, w = shape.shape
        color = int(rng.choice([1, 2, 3, 4, 6]))
        grid[r : r + h, 1 : 1 + w][shape] = color
        grid[r + h // 2, 2 + w] = int(action)
        r += h + 2

    room_count = int(rng.integers(2, 5))
    door_sides = ["top", "right", "bottom", "left"]
    for i in range(room_count):
        h = 6
        w = 6
        r0 = 2 + 8 * (i // 2)
        c0 = 10 + 8 * (i % 2)
        grid[r0 : r0 + h, c0 : c0 + w] = GREY
        grid[r0 + 1 : r0 + h - 1, c0 + 1 : c0 + w - 1] = 0
        side = rng.choice(door_sides)
        if side == "top":
            gap = (r0, c0 + w // 2)
        elif side == "bottom":
            gap = (r0 + h - 1, c0 + w // 2)
        elif side == "left":
            gap = (r0 + h // 2, c0)
        else:
            gap = (r0 + h // 2, c0 + w - 1)
        grid[gap] = BLACK
        if rng.random() < 0.8:
            idx = int(rng.choice(legend_indices))
            shape = shapes[idx]
        else:
            shape = np.array([[1, 1, 1], [1, 0, 0]], dtype=bool)
        color = int(rng.choice([1, 2, 3, 4, 6]))
        sh, sw = shape.shape
        rr = r0 + 1 + (h - 2 - sh) // 2
        cc = c0 + 1 + (w - 2 - sw) // 2
        grid[rr : rr + sh, cc : cc + sw][shape] = color
    return grid
