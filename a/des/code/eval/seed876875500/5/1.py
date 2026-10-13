from __future__ import annotations

import random
from collections import deque

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> tuple[tuple[int, int], ...]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return tuple(out)


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def _mirror_rel(rel: tuple[tuple[int, int], ...], width: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r, width - 1 - c) for r, c in rel))


def _shape_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    rel, h, w = _normalize(coords)
    return h, w, rel


def _signature_from_rel(rel: tuple[tuple[int, int], ...], h: int, w: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    return h, w, tuple(sorted(rel))


def _is_local_connected(coords: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    cells = set(coords)
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _enumerate_key_shapes() -> list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]]:
    seen: set[tuple[int, int, tuple[tuple[int, int], ...]]] = set()
    pool: list[tuple[int, int, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]] = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1, 1 << (h * w)):
                coords = tuple(sorted((idx // w, idx % w) for idx in range(h * w) if (bits >> idx) & 1))
                if not (3 <= len(coords) <= 5):
                    continue
                if not _is_local_connected(coords, h, w):
                    continue
                rows = [r for r, _ in coords]
                cols = [c for _, c in coords]
                if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                    continue
                mirror = _mirror_rel(coords, w)
                if coords == mirror:
                    continue
                canon = (h, w, min(coords, mirror))
                if canon in seen:
                    continue
                seen.add(canon)
                pool.append((h, w, coords, mirror))
    return pool


KEY_SHAPE_POOL = _enumerate_key_shapes()


def _find_rooms(grid: np.ndarray, bg: int) -> list[dict]:
    rooms: list[dict] = []
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        for comp in _components_mask(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 5 or w < 5:
                continue
            if len(comp) != 2 * h + 2 * w - 4:
                continue
            frame = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                frame[r - r0, c - c0] = True
            if np.any(frame[1:-1, 1:-1]):
                continue
            if not (np.all(frame[0, :]) and np.all(frame[-1, :]) and np.all(frame[:, 0]) and np.all(frame[:, -1])):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            inner_mask = interior != bg
            if not inner_mask.any():
                continue
            inner_colors = {int(v) for v in interior[inner_mask]}
            if len(inner_colors) != 1:
                continue
            inner_color = next(iter(inner_colors))
            if inner_color == color:
                continue

            inner_comps_rel = _components_mask(inner_mask)
            if len(inner_comps_rel) != 2:
                continue

            inner_abs = [[(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in rel_comp] for rel_comp in inner_comps_rel]
            key_candidates = []
            for idx, abs_comp in enumerate(inner_abs):
                cr0, cc0, cr1, cc1 = _bbox(abs_comp)
                ch = cr1 - cr0 + 1
                cw = cc1 - cc0 + 1
                if len(abs_comp) <= 9 and ch <= 3 and cw <= 3:
                    key_candidates.append(idx)
            if len(key_candidates) != 1:
                continue

            key_idx = key_candidates[0]
            obj_idx = 1 - key_idx
            key_coords = inner_abs[key_idx]
            obj_coords = inner_abs[obj_idx]
            if len(obj_coords) <= len(key_coords) + 1:
                continue

            or0, oc0, or1, oc1 = _bbox(obj_coords)
            oh = or1 - or0 + 1
            ow = oc1 - oc0 + 1
            if len(obj_coords) <= 9 and oh <= 3 and ow <= 3:
                continue

            key_rel, kh, kw = _normalize(key_coords)
            rooms.append(
                {
                    "frame_color": color,
                    "frame_coords": tuple(sorted(comp)),
                    "bbox": (r0, c0, r1, c1),
                    "inside_color": inner_color,
                    "object_coords": tuple(sorted(obj_coords)),
                    "key_coords": tuple(sorted(key_coords)),
                    "key_rel": key_rel,
                    "key_h": kh,
                    "key_w": kw,
                    "key_box": _bbox(key_coords),
                }
            )
    return rooms


def _find_templates(grid: np.ndarray, bg: int, rooms: list[dict]) -> dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]]:
    room_borders = {frozenset(room["frame_coords"]) for room in rooms}
    room_boxes = [room["bbox"] for room in rooms]
    templates: dict[tuple[int, int, tuple[tuple[int, int], ...]], list[int]] = {}
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    for color in colors:
        for comp in _components_mask(grid == color):
            comp_key = frozenset(comp)
            if comp_key in room_borders:
                continue
            if any(all(r0 < r < r1 and c0 < c < c1 for r, c in comp) for r0, c0, r1, c1 in room_boxes):
                continue
            sig = _shape_signature(comp)
            templates.setdefault(sig, []).append(color)
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each room's small key to an exterior lockhole template by left-right reflection,
    recolor the large room object to the template color, and replace the key with its mirror.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    rooms = _find_rooms(g, bg)
    assert rooms, "expected at least one rectangular room"
    templates = _find_templates(g, bg, rooms)
    assert templates, "expected exterior templates"

    out = g.copy()
    for room in rooms:
        mirror_sig = _signature_from_rel(_mirror_rel(room["key_rel"], room["key_w"]), room["key_h"], room["key_w"])
        assert mirror_sig in templates, "every key must match one reflected template"
        colors = templates[mirror_sig]
        assert len(colors) == 1, "template match must be unique"
        target_color = int(colors[0])

        for r, c in room["object_coords"]:
            out[r, c] = target_color

        for r, c in room["key_coords"]:
            out[r, c] = bg
        kr0, kc0, _, _ = room["key_box"]
        for rr, cc in _mirror_rel(room["key_rel"], room["key_w"]):
            out[kr0 + rr, kc0 + cc] = target_color

    return out


# --------------- generator ---------------

BACKGROUND = 0
ROOM_BORDER = 5
FILL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ROOM_COUNT_CHOICES = (1, 2, 3)
GRID_BOUNDS = {
    1: ((18, 24), (18, 24)),
    2: ((22, 28), (22, 30)),
    3: ((26, 32), (26, 34)),
}
INNER_HEIGHT_RANGE = (7, 10)
INNER_WIDTH_RANGE = (7, 11)
ROOM_PLACEMENT_RETRIES = 80
CONTENT_RETRIES = 60
OBJECT_MASK_RETRIES = 40
GENERATION_RETRIES = 400


def _random_connected_mask(h: int, w: int, min_area: int) -> np.ndarray | None:
    total = h * w
    if total < min_area:
        return None
    upper = min(total, max(min_area, int(round(total * random.uniform(0.55, 0.9)))))
    target = random.randint(min_area, upper)

    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    occupied = {start}
    while len(occupied) < target:
        frontier = []
        weights = []
        for r, c in occupied:
            for nr, nc in _neighbors4(r, c, h, w):
                if mask[nr, nc]:
                    continue
                cell = (nr, nc)
                if cell not in frontier:
                    frontier.append(cell)
                    degree = sum(mask[rr, cc] for rr, cc in _neighbors4(nr, nc, h, w))
                    weights.append(max(1, int(degree)) ** 2)
        if not frontier:
            break
        pick = random.choices(frontier, weights=weights, k=1)[0]
        mask[pick] = True
        occupied.add(pick)

    extras = [(r, c) for r in range(h) for c in range(w) if not mask[r, c] and sum(mask[rr, cc] for rr, cc in _neighbors4(r, c, h, w)) >= 2]
    random.shuffle(extras)
    for r, c in extras[: random.randint(0, min(2, len(extras)))]:
        if random.random() < 0.5:
            mask[r, c] = True

    coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    if len(coords) < min_area:
        return None
    rel, bh, bw = _normalize(coords)
    if len(coords) <= 9 and bh <= 3 and bw <= 3:
        return None
    return mask


def _reserve_with_padding(reserved: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = reserved.shape
    reserved[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)] = True


def _place_rooms(grid: np.ndarray, specs: list[dict]) -> bool:
    H, W = grid.shape
    reserved = np.zeros((H, W), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        positions = [
            (r0, c0)
            for r0 in range(H - oh + 1)
            for c0 in range(W - ow + 1)
            if not reserved[max(0, r0 - 1) : min(H, r0 + oh + 1), max(0, c0 - 1) : min(W, c0 + ow + 1)].any()
        ]
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1
        grid[r0, c0 : c1 + 1] = ROOM_BORDER
        grid[r1, c0 : c1 + 1] = ROOM_BORDER
        grid[r0 : r1 + 1, c0] = ROOM_BORDER
        grid[r0 : r1 + 1, c1] = ROOM_BORDER
        _reserve_with_padding(reserved, r0, c0, oh, ow)
        spec["room_bbox"] = (r0, c0, r1, c1)
    return True


def _room_content_options(inner_h: int, inner_w: int, key_h: int, key_w: int) -> list[tuple[int, int]]:
    return [(r0, c0) for r0 in range(1, inner_h - key_h) for c0 in range(1, inner_w - key_w)]


def _object_box_options(inner_h: int, inner_w: int, blocked: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes = []
    for h in range(3, inner_h - 1):
        for w in range(3, inner_w - 1):
            if max(h, w) < 4:
                continue
            for r0 in range(1, inner_h - h):
                for c0 in range(1, inner_w - w):
                    if blocked[r0 : r0 + h, c0 : c0 + w].any():
                        continue
                    boxes.append((r0, c0, h, w))
    return boxes


def _paint_room_contents(grid: np.ndarray, spec: dict) -> bool:
    r0, c0, r1, c1 = spec["room_bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    key_h = spec["key_h"]
    key_w = spec["key_w"]
    key_rel = spec["key_rel"]
    key_positions = _room_content_options(inner_h, inner_w, key_h, key_w)
    random.shuffle(key_positions)

    for key_r0, key_c0 in key_positions[:CONTENT_RETRIES]:
        blocked = np.zeros((inner_h, inner_w), dtype=bool)
        br0 = max(0, key_r0 - 1)
        bc0 = max(0, key_c0 - 1)
        br1 = min(inner_h, key_r0 + key_h + 1)
        bc1 = min(inner_w, key_c0 + key_w + 1)
        blocked[br0:br1, bc0:bc1] = True

        boxes = _object_box_options(inner_h, inner_w, blocked)
        if not boxes:
            continue
        random.shuffle(boxes)
        min_area = max(6, len(key_rel) + 3)
        for obj_r0, obj_c0, obj_h, obj_w in boxes[:CONTENT_RETRIES]:
            for _ in range(OBJECT_MASK_RETRIES):
                obj_mask = _random_connected_mask(obj_h, obj_w, min_area)
                if obj_mask is None:
                    continue
                obj_rel = [(rr, cc) for rr in range(obj_h) for cc in range(obj_w) if obj_mask[rr, cc]]
                abs_obj = tuple(sorted((r0 + 1 + obj_r0 + rr, c0 + 1 + obj_c0 + cc) for rr, cc in obj_rel))
                abs_key = tuple(sorted((r0 + 1 + key_r0 + rr, c0 + 1 + key_c0 + cc) for rr, cc in key_rel))
                for rr, cc in abs_obj:
                    grid[rr, cc] = spec["source_color"]
                for rr, cc in abs_key:
                    grid[rr, cc] = spec["source_color"]
                spec["object_coords"] = abs_obj
                spec["key_coords"] = abs_key
                spec["key_box"] = (r0 + 1 + key_r0, c0 + 1 + key_c0, r0 + key_r0 + key_h, c0 + key_c0 + key_w)
                return True
    return False


def _place_templates(grid: np.ndarray, specs: list[dict]) -> bool:
    H, W = grid.shape
    reserved = grid != BACKGROUND
    for spec in specs:
        r0, c0, r1, c1 = spec["room_bbox"]
        _reserve_with_padding(reserved, r0, c0, r1 - r0 + 1, c1 - c0 + 1)

    order = list(range(len(specs)))
    random.shuffle(order)
    for idx in order:
        spec = specs[idx]
        rel = _mirror_rel(spec["key_rel"], spec["key_w"])
        h = spec["key_h"]
        w = spec["key_w"]
        positions = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                if reserved[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)].any():
                    continue
                if any(grid[r0 + rr, c0 + cc] != BACKGROUND for rr, cc in rel):
                    continue
                positions.append((r0, c0))
        if not positions:
            return False
        tr0, tc0 = random.choice(positions)
        for rr, cc in rel:
            grid[tr0 + rr, tc0 + cc] = spec["target_color"]
        _reserve_with_padding(reserved, tr0, tc0, h, w)
        spec["template_coords"] = tuple(sorted((tr0 + rr, tc0 + cc) for rr, cc in rel))
    return True


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["object_coords"]:
            out[r, c] = spec["target_color"]
        for r, c in spec["key_coords"]:
            out[r, c] = BACKGROUND
        kr0, kc0, _, _ = spec["key_box"]
        for rr, cc in _mirror_rel(spec["key_rel"], spec["key_w"]):
            out[kr0 + rr, kc0 + cc] = spec["target_color"]
    return out


def _background_is_dominant(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == BACKGROUND)[0][0]])
    return all(bg_count > int(count) for value, count in zip(values, counts) if int(value) != BACKGROUND)


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choice(ROOM_COUNT_CHOICES)
    (h_lo, h_hi), (w_lo, w_hi) = GRID_BOUNDS[room_count]
    H = random.randint(h_lo, h_hi)
    W = random.randint(w_lo, w_hi)
    grid = np.full((H, W), BACKGROUND, dtype=int)

    chosen_shapes = random.sample(KEY_SHAPE_POOL, room_count)
    chosen_colors = random.sample(FILL_COLORS, 2 * room_count)
    source_colors = chosen_colors[:room_count]
    target_colors = chosen_colors[room_count:]

    specs = []
    for idx in range(room_count):
        key_h, key_w, shape_a, shape_b = chosen_shapes[idx]
        specs.append(
            {
                "source_color": source_colors[idx],
                "target_color": target_colors[idx],
                "key_h": key_h,
                "key_w": key_w,
                "key_rel": random.choice([shape_a, shape_b]),
                "outer_h": random.randint(INNER_HEIGHT_RANGE[0], INNER_HEIGHT_RANGE[1]) + 2,
                "outer_w": random.randint(INNER_WIDTH_RANGE[0], INNER_WIDTH_RANGE[1]) + 2,
            }
        )

    for _ in range(ROOM_PLACEMENT_RETRIES):
        trial = np.full_like(grid, BACKGROUND)
        trial_specs = [dict(spec) for spec in specs]
        if not _place_rooms(trial, trial_specs):
            continue
        if not all(_paint_room_contents(trial, spec) for spec in trial_specs):
            continue
        if not _place_templates(trial, trial_specs):
            continue
        if not _background_is_dominant(trial):
            continue
        expected = _expected_output(trial, trial_specs)
        try:
            solved = main(trial)
        except AssertionError:
            continue
        if not np.array_equal(solved, expected):
            continue
        return trial
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
