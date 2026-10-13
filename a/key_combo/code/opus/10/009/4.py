import random
from collections import defaultdict, deque

import numpy as np


BG = 0
RED = 2
PALETTE = [1, 3, 4, 6, 7, 8, 9]
SIDES = ("left", "right", "up", "down")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CENTRAL_PLUS = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _component(mask: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    q = deque([start])
    seen = {start}
    out = []
    while q:
        r, c = q.popleft()
        out.append((r, c))
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return out


def _connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    q = deque([next(iter(coords))])
    seen = {q[0]}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in coords and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen == coords


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = set()
    objects = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color in (BG, RED) or (r, c) in seen:
                continue
            mask = g == color
            coords = _component(mask, (r, c))
            seen.update(coords)
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_set": set(coords),
                    "bbox": _bbox(coords),
                }
            )
    return objects


def _object_sides(obj: dict) -> dict[str, set[tuple[int, int]]]:
    r0, c0, r1, c1 = obj["bbox"]
    out = {"left": set(), "right": set(), "up": set(), "down": set()}
    for r, c in obj["coords"]:
        if c == c0:
            out["left"].add((r, c))
        if c == c1:
            out["right"].add((r, c))
        if r == r0:
            out["up"].add((r, c))
        if r == r1:
            out["down"].add((r, c))
    return out


def _associate_markers(grid: np.ndarray, objects: list[dict]) -> dict[int, tuple[int, int, str]]:
    owner = {}
    side_cells = [_object_sides(obj) for obj in objects]
    cell_to_object = {}
    for idx, obj in enumerate(objects):
        for cell in obj["coords"]:
            cell_to_object[cell] = idx

    for r, c in np.argwhere(grid == RED):
        adjacent = set()
        for nr, nc in neighbors4(int(r), int(c)):
            idx = cell_to_object.get((nr, nc))
            if idx is not None:
                adjacent.add(idx)
        assert len(adjacent) == 1
        idx = next(iter(adjacent))
        sides = []
        for side, cells in side_cells[idx].items():
            if side == "left" and any((int(r), int(c) + 1) == cell for cell in cells):
                sides.append(side)
            elif side == "right" and any((int(r), int(c) - 1) == cell for cell in cells):
                sides.append(side)
            elif side == "up" and any((int(r) + 1, int(c)) == cell for cell in cells):
                sides.append(side)
            elif side == "down" and any((int(r) - 1, int(c)) == cell for cell in cells):
                sides.append(side)
        assert len(sides) == 1
        assert idx not in owner
        owner[idx] = (int(r), int(c), sides[0])

    assert len(owner) == len(objects)
    return owner


def _mirror_coords(obj: dict, side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    out = []
    for r, c in obj["coords"]:
        if side == "left":
            out.append((r, 2 * c0 - 1 - c))
        elif side == "right":
            out.append((r, 2 * c1 + 1 - c))
        elif side == "up":
            out.append((2 * r0 - 1 - r, c))
        elif side == "down":
            out.append((2 * r1 + 1 - r, c))
        else:
            raise AssertionError(side)
    return out


def _overlap_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    markers = _associate_markers(g, objects)
    occupancy = {}
    for idx, obj in enumerate(objects):
        for cell in obj["coords"]:
            occupancy[cell] = idx

    records = []
    for idx, obj in enumerate(objects):
        side = markers[idx][2]
        count = 0
        for mr, mc in _mirror_coords(obj, side):
            if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1]:
                other = occupancy.get((mr, mc))
                if other is not None and other != idx:
                    count += 1
        records.append({"index": idx, "color": obj["color"], "count": count, "side": side})
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a right-aligned histogram ranked by how many pixels each marked object's mirror overlaps with other originals.

    Input structure:
    - A black background contains 3-5 non-red monochromatic 4-connected foreground objects, each with a unique color.
    - Every object has exactly one red marker pixel 4-adjacent to one bbox side; the touched side gives that object's mirror direction.
    - Objects are spatially separated in the input, while markers may sit in the gaps between them.

    Rule:
    - Identify every non-black, non-red object and the unique red marker associated with it.
    - Mirror each object across the exterior axis just beyond the marked bbox side, keeping the object's color.
    - For each object, count how many in-bounds pixels of its mirrored copy land on pixels belonging to other original objects.
    - Sort objects by that overlap count in descending order and output a histogram whose row color is the object's color and whose right-aligned bar length is its count.

    Key insight:
    - The bar lengths are not local shape properties: each value is a global interaction count produced only after every object's individually directed mirror is compared against the other original objects.
    """
    records = _overlap_records(np.asarray(grid, dtype=int))
    counts = [record["count"] for record in records]
    assert records
    assert len(set(counts)) == len(counts)
    width = max(counts)
    out = np.zeros((len(records), width), dtype=int)
    for row, record in enumerate(sorted(records, key=lambda item: item["count"], reverse=True)):
        out[row, width - record["count"] :] = record["color"]
    return out


def _enumerate_horizontal_templates() -> dict[int, dict[bool, list[np.ndarray]]]:
    templates = {count: {False: [], True: []} for count in range(6)}
    cells = [(r, c) for r in range(3) for c in range(5)]
    for bits in range(1, 1 << 15):
        coords = {cells[i] for i in range(15) if (bits >> i) & 1}
        area = len(coords)
        if not (4 <= area <= 8):
            continue
        if not _connected(coords):
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        if area == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1):
            continue
        if not any(c == 4 for _, c in coords):
            continue
        own = sum((r, c) in CENTRAL_PLUS for r, c in coords)
        arr = np.zeros((3, 5), dtype=bool)
        for r, c in coords:
            arr[r, c] = True
        templates[own][False].append(arr)
        if arr[1, 4]:
            templates[own][True].append(arr)
    return templates


HORIZONTAL_TEMPLATES = _enumerate_horizontal_templates()


def _orient_template(base: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return base.copy()
    if side == "right":
        return np.fliplr(base)
    if side == "up":
        return base.T.copy()
    if side == "down":
        return np.flipud(base.T)
    raise AssertionError(side)


def _source_bbox(center_r0: int, center_c0: int, side: str) -> tuple[int, int]:
    if side == "left":
        return center_r0, center_c0 - 7
    if side == "right":
        return center_r0, center_c0 + 5
    if side == "up":
        return center_r0 - 7, center_c0
    if side == "down":
        return center_r0 + 5, center_c0
    raise AssertionError(side)


def _center_marker(center_r0: int, center_c0: int, side: str) -> tuple[int, int]:
    if side == "left":
        return center_r0 + 1, center_c0 - 1
    if side == "right":
        return center_r0 + 1, center_c0 + 3
    if side == "up":
        return center_r0 - 1, center_c0 + 1
    if side == "down":
        return center_r0 + 3, center_c0 + 1
    raise AssertionError(side)


def _source_marker(mask: np.ndarray, side: str, r0: int, c0: int) -> tuple[int, int]:
    if side == "left":
        rows = [r for r in range(mask.shape[0]) if mask[r, mask.shape[1] - 1]]
        row = random.choice(rows)
        return r0 + row, c0 + mask.shape[1]
    if side == "right":
        rows = [r for r in range(mask.shape[0]) if mask[r, 0]]
        row = random.choice(rows)
        return r0 + row, c0 - 1
    if side == "up":
        cols = [c for c in range(mask.shape[1]) if mask[mask.shape[0] - 1, c]]
        col = random.choice(cols)
        return r0 + mask.shape[0], c0 + col
    if side == "down":
        cols = [c for c in range(mask.shape[1]) if mask[0, c]]
        col = random.choice(cols)
        return r0 - 1, c0 + col
    raise AssertionError(side)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                grid[r0 + r, c0 + c] = color


def _build_attempt() -> np.ndarray:
    n_objects = random.randint(3, 5)
    source_sides = random.sample(list(SIDES), n_objects - 1)
    center_side = random.choice(source_sides)
    side_counts = dict(zip(source_sides, random.sample([2, 3, 4, 5], n_objects - 1)))
    colors = random.sample(PALETTE, n_objects)
    center_color = colors[0]
    side_colors = {side: colors[i + 1] for i, side in enumerate(source_sides)}

    h = random.randint(18, 24)
    w = random.randint(18, 24)
    center_r0 = random.randint(7, h - 10)
    center_c0 = random.randint(7, w - 10)

    grid = np.zeros((h, w), dtype=int)
    center_mask = np.zeros((3, 3), dtype=bool)
    for r, c in CENTRAL_PLUS:
        center_mask[r, c] = True
    _paint_mask(grid, center_mask, center_r0, center_c0, center_color)
    mr, mc = _center_marker(center_r0, center_c0, center_side)
    grid[mr, mc] = RED

    for side in source_sides:
        count = side_counts[side]
        need_center_hit = side == center_side
        base = random.choice(HORIZONTAL_TEMPLATES[count][need_center_hit])
        mask = _orient_template(base, side)
        r0, c0 = _source_bbox(center_r0, center_c0, side)
        _paint_mask(grid, mask, r0, c0, side_colors[side])
        sr, sc = _source_marker(mask, side, r0, c0)
        grid[sr, sc] = RED

    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        records = _overlap_records(g)
        objects = _extract_objects(g)
        if not (3 <= len(objects) <= 5):
            return False
        if len(records) != len(objects):
            return False
        if len({obj["color"] for obj in objects}) != len(objects):
            return False
        for obj in objects:
            if not (4 <= len(obj["coords"]) <= 8):
                return False
        for i, obj in enumerate(objects):
            for r, c in obj["coords"]:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                            color = int(g[nr, nc])
                            if color not in (BG, RED) and color != obj["color"] and (nr, nc) not in obj["coord_set"]:
                                return False
        counts = [record["count"] for record in records]
        if len(set(counts)) != len(counts):
            return False
        if any(count <= 0 for count in counts):
            return False
        if sum(count > 0 for count in counts) < 2:
            return False
        out = main(g)
        if out.ndim != 2 or max(out.shape) > 30:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_attempt()
        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirror-overlap histogram input")
