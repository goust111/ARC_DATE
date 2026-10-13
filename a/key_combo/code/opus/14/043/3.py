import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
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


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    steps = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _loop_neighbors(loop_mask: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    cells = {tuple(map(int, rc)) for rc in np.argwhere(loop_mask)}
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in cells:
        nbs = [(nr, nc) for nr, nc in neighbors4(r, c) if (nr, nc) in cells]
        assert len(nbs) == 2, "loop pixels must have degree 2"
        neighbors[(r, c)] = nbs
    return neighbors


def _trace_cycle(loop_mask: np.ndarray) -> list[tuple[int, int]]:
    neighbors = _loop_neighbors(loop_mask)
    start = min(neighbors)
    path = [start]
    prev = None
    cur = start
    while True:
        nbs = neighbors[cur]
        if prev is None:
            nxt = min(nbs)
        else:
            nxt = nbs[0] if nbs[0] != prev else nbs[1]
        if nxt == start:
            assert len(path) == len(neighbors), "cycle did not visit every loop pixel exactly once"
            return path
        path.append(nxt)
        prev, cur = cur, nxt


def _turn_data(path: list[tuple[int, int]]) -> tuple[dict[tuple[int, int], int], Counter]:
    n = len(path)
    signs: dict[tuple[int, int], int] = {}
    counts: Counter = Counter()
    for i, (r, c) in enumerate(path):
        pr, pc = path[i - 1]
        nr, nc = path[(i + 1) % n]
        incoming = (r - pr, c - pc)
        outgoing = (nr - r, nc - c)
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        if cross == 0:
            continue
        sign = 1 if cross > 0 else -1
        signs[(r, c)] = sign
        counts[sign] += 1
    return signs, counts


def analyze_loop(loop_mask: np.ndarray) -> dict:
    path = _trace_cycle(loop_mask)
    signs, counts = _turn_data(path)
    assert signs, "expected at least one bend"
    assert abs(counts[1] - counts[-1]) == 4, "simple orthogonal loop must satisfy convex-concave = 4"
    if counts[1] > counts[-1]:
        path = list(reversed(path))
        signs, counts = _turn_data(path)
    convex_sign = 1 if counts[1] > counts[-1] else -1
    neighbors = _loop_neighbors(loop_mask)
    bends = []
    for r, c in path:
        sign = signs.get((r, c))
        if sign is None:
            continue
        dirs = [(nr - r, nc - c) for nr, nc in neighbors[(r, c)]]
        total_dr = dirs[0][0] + dirs[1][0]
        total_dc = dirs[0][1] + dirs[1][1]
        outward = (-1 if total_dr > 0 else 1, -1 if total_dc > 0 else 1)
        bends.append(
            {
                "coord": (r, c),
                "sign": sign,
                "is_convex": sign == convex_sign,
                "outward": outward,
            }
        )
    return {"path": path, "bends": bends}


def _stamp_bbox(corner: tuple[int, int], outward: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = corner
    dr, dc = outward
    h, w = shape
    anchor_r, anchor_c = r + dr, c + dc
    corner_r = 0 if dr > 0 else h - 1
    corner_c = 0 if dc > 0 else w - 1
    r0 = anchor_r - corner_r
    c0 = anchor_c - corner_c
    return int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)


def _render_output(loop_mask: np.ndarray, sprite_crop: np.ndarray) -> np.ndarray:
    h, w = loop_mask.shape
    out = np.zeros((h, w), dtype=int)
    out[loop_mask] = BLUE
    analysis = analyze_loop(loop_mask)
    for bend in analysis["bends"]:
        if not bend["is_convex"]:
            r, c = bend["coord"]
            out[r, c] = YELLOW

    reserved = np.zeros((h, w), dtype=bool)
    loop_cells = loop_mask.copy()
    sprite_crop = np.asarray(sprite_crop, dtype=int)
    for bend in analysis["bends"]:
        if not bend["is_convex"]:
            continue
        outward = bend["outward"]
        stamp = np.fliplr(sprite_crop) if outward[1] < 0 else sprite_crop
        r0, c0, r1, c1 = _stamp_bbox(bend["coord"], outward, stamp.shape)
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        if loop_cells[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[stamp == RED] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp mirrored sprite copies outside every convex loop corner while recoloring concave corners yellow.

    Input structure:
    - A black background contains one blue one-pixel-wide 4-connected closed orthogonal loop; every loop pixel has exactly two loop neighbors.
    - A small red 4-connected sprite sits strictly inside the loop, with a tight bounding box no larger than 2x3 or 3x2.
    - The loop may contain both convex and concave bends, and the red sprite is separated from the loop.

    Rule:
    - Trace the blue loop, classify bend pixels by turn sign, and treat the majority turn sign as convex while the minority sign is concave.
    - Keep straight loop pixels blue and recolor every concave bend yellow.
    - For each convex bend in clockwise loop order, compute the outward diagonal by reversing the two incident edge directions.
    - Place one red sprite copy outside that bend so the sprite bbox corner nearest the loop sits on the bend's outward diagonal neighbor; flip the sprite horizontally exactly when the outward diagonal points left.
    - If a sprite bbox would leave the grid or overlap the loop or an earlier sprite bbox, skip that bend.
    - Remove the original interior red sprite from the output.

    Key insight:
    - The same geometric attribute of each bend, its outward diagonal direction, simultaneously decides whether the bend stamps at all, where the stamp goes, and whether the stamp must be mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    loop_mask = g == BLUE
    sprite_mask = g == RED
    assert np.count_nonzero(loop_mask) > 0 and np.count_nonzero(sprite_mask) > 0
    sprite_components = components_mask(sprite_mask, connectivity=4)
    assert len(sprite_components) == 1, "expected one red sprite"
    sprite_box = bbox_coords(sprite_components[0])
    sprite_crop = crop_bbox(g, sprite_box)
    sprite_crop[sprite_crop != RED] = BLACK
    return _render_output(loop_mask, sprite_crop)


SPRITE_TEMPLATES = [
    np.array([[RED, RED, BLACK], [BLACK, RED, RED]], dtype=int),
    np.array([[RED, BLACK, BLACK], [RED, RED, RED]], dtype=int),
    np.array([[RED, RED, RED], [BLACK, RED, BLACK]], dtype=int),
    np.array([[RED, RED, BLACK], [RED, BLACK, BLACK]], dtype=int),
]
GEN_ATTEMPTS = 400


def _random_sprite(rng: np.random.Generator) -> np.ndarray:
    base = SPRITE_TEMPLATES[int(rng.integers(len(SPRITE_TEMPLATES)))]
    sprite = np.rot90(base, int(rng.integers(4)))
    if sprite.shape not in {(2, 3), (3, 2)}:
        return _random_sprite(rng)
    if np.array_equal(sprite == RED, np.fliplr(sprite == RED)):
        return _random_sprite(rng)
    coords = components_mask(sprite == RED, connectivity=4)
    if len(coords) != 1 or not (3 <= int(coords[0].shape[0]) <= 4):
        return _random_sprite(rng)
    return sprite.copy()


def _append_segment(path: list[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    r0, c0 = start
    r1, c1 = end
    assert r0 == r1 or c0 == c1
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        cells = [(r0, c) for c in range(c0, c1 + step, step)]
    else:
        step = 1 if r1 >= r0 else -1
        cells = [(r, c0) for r in range(r0, r1 + step, step)]
    if not path:
        path.extend(cells)
    else:
        assert path[-1] == cells[0]
        path.extend(cells[1:])


def _chain_from_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for a, b in zip(vertices, vertices[1:]):
        _append_segment(path, a, b)
    return path


def _random_loop(h: int, w: int, sprite_shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray | None:
    sh, sw = sprite_shape
    outer_h_min = max(10, sh + 6)
    outer_w_min = max(10, sw + 6)
    outer_h_max = h - 2 * sh
    outer_w_max = w - 2 * sw
    if outer_h_max < outer_h_min or outer_w_max < outer_w_min:
        return None

    outer_h = int(rng.integers(outer_h_min, outer_h_max + 1))
    outer_w = int(rng.integers(outer_w_min, outer_w_max + 1))
    top = int(rng.integers(sh, h - outer_h - sh + 1))
    left = int(rng.integers(sw, w - outer_w - sw + 1))
    bottom = top + outer_h - 1
    right = left + outer_w - 1

    eligible = []
    if outer_w >= 9 and outer_h >= 8:
        eligible.extend(["top", "bottom"])
    if outer_h >= 9 and outer_w >= 8:
        eligible.extend(["left", "right"])
    if not eligible:
        return None

    chosen_sides = [str(rng.choice(eligible))]
    notches: dict[str, tuple[int, int, int]] = {}
    for side in chosen_sides:
        if side in {"top", "bottom"}:
            min_span = max(3, 2 * sw + 1)
            start_max = right - min_span - 1
            if start_max < left + 2:
                return None
            start = int(rng.integers(left + 2, start_max + 1))
            end = int(rng.integers(start + min_span, right))
            depth = int(rng.integers(2, bottom - top - 2))
            notches[side] = (start, end, depth)
        else:
            min_span = max(3, 2 * sh + 1)
            start_max = bottom - min_span - 1
            if start_max < top + 2:
                return None
            start = int(rng.integers(top + 2, start_max + 1))
            end = int(rng.integers(start + min_span, bottom))
            depth = int(rng.integers(2, right - left - 2))
            notches[side] = (start, end, depth)

    top_vertices = [(top, left)]
    if "top" in notches:
        start, end, depth = notches["top"]
        top_vertices.extend([(top, start), (top + depth, start), (top + depth, end), (top, end)])
    top_vertices.append((top, right))

    right_vertices = [(top, right)]
    if "right" in notches:
        start, end, depth = notches["right"]
        right_vertices.extend([(start, right), (start, right - depth), (end, right - depth), (end, right)])
    right_vertices.append((bottom, right))

    bottom_vertices = [(bottom, right)]
    if "bottom" in notches:
        start, end, depth = notches["bottom"]
        bottom_vertices.extend([(bottom, end), (bottom - depth, end), (bottom - depth, start), (bottom, start)])
    bottom_vertices.append((bottom, left))

    left_vertices = [(bottom, left)]
    if "left" in notches:
        start, end, depth = notches["left"]
        left_vertices.extend([(end, left), (end, left + depth), (start, left + depth), (start, left)])
    left_vertices.append((top, left))

    path = _chain_from_vertices(top_vertices)
    path.extend(_chain_from_vertices(right_vertices)[1:])
    path.extend(_chain_from_vertices(bottom_vertices)[1:])
    path.extend(_chain_from_vertices(left_vertices)[1:])
    if path[-1] != path[0]:
        return None
    path = path[:-1]
    if len(set(path)) != len(path):
        return None

    loop_mask = np.zeros((h, w), dtype=bool)
    for r, c in path:
        loop_mask[r, c] = True
    return loop_mask


def _interior_from_loop(loop_mask: np.ndarray) -> np.ndarray:
    h, w = loop_mask.shape
    blocked = np.pad(loop_mask, 1, constant_values=False)
    outside = np.zeros_like(blocked, dtype=bool)
    stack = [(0, 0)]
    outside[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return ~outside[1:-1, 1:-1] & ~loop_mask


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    coords = np.argwhere(mask)
    for r, c in coords:
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _find_sprite_slot(interior: np.ndarray, loop_mask: np.ndarray, sprite: np.ndarray, rng: np.random.Generator):
    h, w = interior.shape
    sh, sw = sprite.shape
    forbidden = _dilate8(loop_mask)
    safe = interior & ~forbidden
    placements = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            patch = safe[r0 : r0 + sh, c0 : c0 + sw]
            if np.all(patch):
                placements.append((r0, c0))
    if not placements:
        return None
    return placements[int(rng.integers(len(placements)))]


def _independent_expected(loop_mask: np.ndarray, sprite: np.ndarray) -> np.ndarray:
    return _render_output(loop_mask, sprite)


def _validate_generated(grid: np.ndarray, sprite: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BLACK, BLUE, RED}:
        return False

    loop_mask = grid == BLUE
    sprite_mask = grid == RED
    if len(components_mask(loop_mask, connectivity=4)) != 1:
        return False
    if len(components_mask(sprite_mask, connectivity=4)) != 1:
        return False

    for r, c in np.argwhere(sprite_mask):
        for dr, dc in DIR8:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and loop_mask[nr, nc]:
                return False

    interior = _interior_from_loop(loop_mask)
    if len(components_mask(interior, connectivity=4)) != 1:
        return False

    loop_info = analyze_loop(loop_mask)
    convex = [b for b in loop_info["bends"] if b["is_convex"]]
    concave = [b for b in loop_info["bends"] if not b["is_convex"]]
    if len(convex) < 4 or len(concave) < 2:
        return False

    reserved = np.zeros_like(loop_mask, dtype=bool)
    for bend in convex:
        r0, c0, r1, c1 = _stamp_bbox(bend["coord"], bend["outward"], sprite.shape)
        if r0 < 0 or c0 < 0 or r1 >= grid.shape[0] or c1 >= grid.shape[1]:
            return False
        if loop_mask[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            return False
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    expected = _independent_expected(loop_mask, sprite)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False
    if np.count_nonzero(actual == YELLOW) != len(concave):
        return False
    if np.count_nonzero(actual == RED) <= np.count_nonzero(grid == RED):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        h = int(rng.integers(20, 27))
        w = int(rng.integers(20, 27))
        sprite = _random_sprite(rng)
        loop_mask = _random_loop(h, w, sprite.shape, rng)
        if loop_mask is None:
            continue
        if len(components_mask(loop_mask, connectivity=4)) != 1:
            continue
        try:
            info = analyze_loop(loop_mask)
        except AssertionError:
            continue
        convex = [b for b in info["bends"] if b["is_convex"]]
        if len(convex) < 4:
            continue
        interior = _interior_from_loop(loop_mask)
        if len(components_mask(interior, connectivity=4)) != 1:
            continue
        slot = _find_sprite_slot(interior, loop_mask, sprite, rng)
        if slot is None:
            continue
        r0, c0 = slot
        grid = np.zeros((h, w), dtype=int)
        grid[loop_mask] = BLUE
        sprite_mask = sprite == RED
        grid[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]][sprite_mask] = RED
        if _validate_generated(grid, sprite):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")


if __name__ == "__main__":
    random.seed(0)
    sample = generate_input()
    print(sample)
