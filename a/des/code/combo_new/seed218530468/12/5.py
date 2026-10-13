from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
BODY_COLORS = (1, 2, 3, 6, 7, 8, 9)

DIR4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    dirs = DIR4 if connectivity == 4 else DIR8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _rotate_cw(mask: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(mask, -steps % 4)


def _rotation_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    variants = []
    for steps in range(4):
        rotated = _rotate_cw(mask, steps)
        variants.append((tuple(rotated.shape), rotated.tobytes()))
    return min(variants)


def _rotation_variant_count(mask: np.ndarray) -> int:
    seen = set()
    for steps in range(4):
        rotated = _rotate_cw(mask, steps)
        seen.add((tuple(rotated.shape), rotated.tobytes()))
    return len(seen)


def _direction_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = marker
    r0, c0, r1, c1 = bbox
    side_hits = [
        r == r0,
        c == c1,
        r == r1,
        c == c0,
    ]
    assert sum(side_hits) == 1, "marker must lie on exactly one bbox side"
    return side_hits.index(True)


def _extract_interior(local_frame_mask: np.ndarray) -> np.ndarray:
    h, w = local_frame_mask.shape
    open_mask = ~local_frame_mask
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if open_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return open_mask & ~outside


def _parse_object(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert YELLOW in colors and GREY not in colors and len(colors) == 2
    yellow_cells = [(r, c) for r, c in coords if int(grid[r, c]) == YELLOW]
    assert len(yellow_cells) == 1
    marker = yellow_cells[0]
    bbox = _bbox(coords)
    body_cells = [(r, c) for r, c in coords if int(grid[r, c]) != YELLOW]
    body_color = int(grid[body_cells[0]])
    assert all(int(grid[r, c]) == body_color for r, c in body_cells)
    body_mask = _mask_from_coords(body_cells)
    assert len(_components(body_mask, connectivity=4)) == 1
    return {
        "coords": coords,
        "dir": _direction_from_marker(marker, bbox),
        "color": body_color,
        "mask": body_mask,
    }


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert colors <= {GREY, YELLOW} and YELLOW in colors
    yellow_cells = [(r, c) for r, c in coords if int(grid[r, c]) == YELLOW]
    assert len(yellow_cells) == 1
    marker = yellow_cells[0]
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    interior = _extract_interior(local)
    parts = _components(interior, connectivity=4)
    assert len(parts) == 1, "frame must enclose exactly one interior region"
    inner_mask = _crop_mask(interior)
    inner_coords = [(r0 + r, c0 + c) for r, c in np.argwhere(interior)]
    return {
        "dir": _direction_from_marker(marker, bbox),
        "mask": inner_mask,
        "fill_coords": inner_coords,
    }


def _perfect_matching(options: list[list[int]], n_right: int) -> list[int]:
    order = sorted(range(len(options)), key=lambda i: len(options[i]))
    used = [False] * n_right
    assign = [-1] * len(options)

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        left = order[pos]
        for right in options[left]:
            if used[right]:
                continue
            used[right] = True
            assign[left] = right
            if dfs(pos + 1):
                return True
            assign[left] = -1
            used[right] = False
        return False

    assert dfs(0), "object-frame matching must be unique and feasible"
    return assign


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate marked colored objects and place them into the matching marked grey outline frames.

    Input structure:
    - The background is black.
    - Several small colored objects appear in the grid; each object is one body color plus one yellow marker pixel.
    - Several hollow grey outline frames also appear; each frame keeps one yellow marker on one side of its border.

    Rule:
    - Read each object's current direction from the side of its yellow marker and each frame's target direction from the frame marker.
    - For an object-frame pair, rotate the object's body clockwise by the direction difference, then compare the rotated body shape against the frame's enclosed interior shape.
    - Fill the uniquely matching frame interior with the object's body color, clear the object's original pixels to black, and keep the grey border and yellow frame marker unchanged.

    Key insight:
    - The yellow pixels carry two jobs at once: they encode the required rotation and they help disambiguate which rotated body belongs in which frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    comps = _components(g != BLACK, connectivity=8)
    objects = []
    frames = []
    for coords in comps:
        colors = {int(g[r, c]) for r, c in coords}
        if YELLOW in colors and GREY in colors and colors <= {GREY, YELLOW}:
            frames.append(_parse_frame(g, coords))
        elif YELLOW in colors and GREY not in colors and len(colors) == 2:
            objects.append(_parse_object(g, coords))

    assert objects and frames and len(objects) == len(frames)

    options: list[list[int]] = []
    for obj in objects:
        compatible = []
        for j, frame in enumerate(frames):
            steps = (frame["dir"] - obj["dir"]) % 4
            rotated = _rotate_cw(obj["mask"], steps)
            if rotated.shape == frame["mask"].shape and np.array_equal(rotated, frame["mask"]):
                compatible.append(j)
        assert compatible, "every object must match at least one frame"
        options.append(compatible)

    assignment = _perfect_matching(options, len(frames))

    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BLACK
    for i, frame_idx in enumerate(assignment):
        color = int(objects[i]["color"])
        for r, c in frames[frame_idx]["fill_coords"]:
            out[r, c] = color
    return out


GRID_SIDE_MAX = 30
OUTER_MARGIN = 1
PLACEMENT_GAP = 1
NUM_PAIRS_CHOICES = (2, 3, 4)
BODY_DIM_CHOICES = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_ATTEMPTS = 400


def _body_seed(h: int, w: int) -> set[tuple[int, int]]:
    cells = {(h // 2, c) for c in range(w)}
    cells |= {(r, w // 2) for r in range(h)}
    return cells


def _frontier(cells: set[tuple[int, int]], h: int, w: int) -> list[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                out.add((nr, nc))
    return list(out)


def _body_hole_free(mask: np.ndarray) -> bool:
    empty = ~mask
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if empty[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and empty[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    return not bool((empty & ~outside).any())


def _frame_border_from_body(body_mask: np.ndarray) -> np.ndarray:
    h, w = body_mask.shape
    inner = np.zeros((h + 2, w + 2), dtype=bool)
    inner[1 : h + 1, 1 : w + 1] = body_mask
    border = np.zeros_like(inner)
    for r, c in np.argwhere(inner):
        for dr, dc in DIR8:
            nr, nc = int(r) + dr, int(c) + dc
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not inner[nr, nc]:
                border[nr, nc] = True
    return border


def _valid_body_shape(mask: np.ndarray) -> bool:
    if len(_components(mask, connectivity=4)) != 1:
        return False
    if not _body_hole_free(mask):
        return False
    if _rotation_variant_count(mask) == 1:
        return False
    border = _frame_border_from_body(mask)
    if len(_components(border, connectivity=8)) != 1:
        return False
    interior = _extract_interior(border)
    recovered = interior[1:-1, 1:-1]
    return np.array_equal(recovered, mask)


def _random_body_shape() -> np.ndarray:
    for _ in range(200):
        h, w = random.choice(BODY_DIM_CHOICES)
        cells = _body_seed(h, w)
        target = random.randint(len(cells), h * w)
        while len(cells) < target:
            choices = _frontier(cells, h, w)
            if not choices:
                break
            cells.add(random.choice(choices))
            if random.random() < 0.35:
                extra_choices = _frontier(cells, h, w)
                if extra_choices:
                    cells.add(random.choice(extra_choices))
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _valid_body_shape(mask):
            return mask
    raise RuntimeError("failed to generate a valid body shape")


def _object_patch(body_mask: np.ndarray, color: int, direction: int) -> np.ndarray:
    h, w = body_mask.shape
    if direction == UP:
        patch = np.full((h + 1, w), BLACK, dtype=int)
        patch[1:][body_mask] = color
        patch[0, w // 2] = YELLOW
        return patch
    if direction == RIGHT:
        patch = np.full((h, w + 1), BLACK, dtype=int)
        patch[:, :w][body_mask] = color
        patch[h // 2, w] = YELLOW
        return patch
    if direction == DOWN:
        patch = np.full((h + 1, w), BLACK, dtype=int)
        patch[:h][body_mask] = color
        patch[h, w // 2] = YELLOW
        return patch
    patch = np.full((h, w + 1), BLACK, dtype=int)
    patch[:, 1:][body_mask] = color
    patch[h // 2, 0] = YELLOW
    return patch


def _frame_patch(inner_mask: np.ndarray, direction: int) -> np.ndarray:
    border = _frame_border_from_body(inner_mask)
    patch = np.full(border.shape, BLACK, dtype=int)
    patch[border] = GREY
    h, w = inner_mask.shape
    marker = {
        UP: (0, 1 + w // 2),
        RIGHT: (1 + h // 2, w + 1),
        DOWN: (h + 1, 1 + w // 2),
        LEFT: (1 + h // 2, 0),
    }[direction]
    assert patch[marker] == GREY
    patch[marker] = YELLOW
    return patch


def _paste(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BLACK
    canvas[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]][mask] = patch[mask]


def _can_place(occupied: np.ndarray, patch_mask: np.ndarray, r0: int, c0: int, gap: int) -> bool:
    h, w = occupied.shape
    for pr, pc in np.argwhere(patch_mask):
        gr, gc = r0 + int(pr), c0 + int(pc)
        if occupied[gr, gc]:
            return False
        rr0 = max(0, gr - gap)
        rr1 = min(h, gr + gap + 1)
        cc0 = max(0, gc - gap)
        cc1 = min(w, gc + gap + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            return False
    return True


def _place_patch_randomly(
    canvas: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
) -> tuple[int, int] | None:
    ph, pw = patch.shape
    h, w = canvas.shape
    if ph > h - 2 * OUTER_MARGIN or pw > w - 2 * OUTER_MARGIN:
        return None
    rows = list(range(OUTER_MARGIN, h - ph - OUTER_MARGIN + 1))
    cols = list(range(OUTER_MARGIN, w - pw - OUTER_MARGIN + 1))
    origins = [(r, c) for r in rows for c in cols]
    random.shuffle(origins)
    patch_mask = patch != BLACK
    for r0, c0 in origins:
        if _can_place(occupied, patch_mask, r0, c0, PLACEMENT_GAP):
            _paste(canvas, patch, r0, c0)
            occupied[r0 : r0 + ph, c0 : c0 + pw] |= patch_mask
            return r0, c0
    return None


def _sample_pairs() -> list[dict]:
    n_pairs = random.choice(NUM_PAIRS_CHOICES)
    colors = random.sample(BODY_COLORS, n_pairs)
    used_keys = set()
    pairs = []
    for color in colors:
        for _ in range(200):
            body = _random_body_shape()
            key = _rotation_key(body)
            if key in used_keys:
                continue
            used_keys.add(key)
            current_dir = random.randrange(4)
            target_dir = random.randrange(4)
            steps = (target_dir - current_dir) % 4
            rotated = _rotate_cw(body, steps)
            pairs.append(
                {
                    "color": color,
                    "object_patch": _object_patch(body, color, current_dir),
                    "frame_patch": _frame_patch(rotated, target_dir),
                    "filled_mask": rotated,
                }
            )
            break
        else:
            return []
    return pairs


def _grid_side_bounds(n_pairs: int) -> tuple[int, int]:
    low = max(14, 8 + 4 * n_pairs)
    return low, GRID_SIDE_MAX


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    pairs = _sample_pairs()
    if not pairs:
        return None
    low, high = _grid_side_bounds(len(pairs))
    h = random.randint(low, high)
    w = random.randint(low, high)
    inp = np.full((h, w), BLACK, dtype=int)
    out = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_order = list(range(len(pairs)))
    random.shuffle(frame_order)
    frame_order.sort(key=lambda i: pairs[i]["frame_patch"].size, reverse=True)
    frame_positions: dict[int, tuple[int, int]] = {}
    for idx in frame_order:
        pos = _place_patch_randomly(inp, occupied, pairs[idx]["frame_patch"])
        if pos is None:
            return None
        frame_positions[idx] = pos
        _paste(out, pairs[idx]["frame_patch"], pos[0], pos[1])
        for r, c in np.argwhere(pairs[idx]["filled_mask"]):
            out[pos[0] + 1 + int(r), pos[1] + 1 + int(c)] = int(pairs[idx]["color"])

    object_order = list(range(len(pairs)))
    random.shuffle(object_order)
    object_order.sort(key=lambda i: pairs[i]["object_patch"].size, reverse=True)
    for idx in object_order:
        pos = _place_patch_randomly(inp, occupied, pairs[idx]["object_patch"])
        if pos is None:
            return None
        # Keep output free of the movable object.
        _ = pos

    if np.array_equal(inp, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC instance")
