import random
from collections import Counter

import numpy as np


TAIL_COORDS = {
    "left": [(0, 0), (1, 1), (1, 2), (2, 0)],
    "right": [(0, 2), (1, 0), (1, 1), (2, 2)],
    "top": [(0, 0), (0, 2), (1, 1), (2, 1)],
    "bottom": [(0, 1), (1, 1), (2, 0), (2, 2)],
}

GEN_ATTEMPTS = 300
GEN_MAX_OBJECTS = 5


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(),
                }
            )
    return comps


def _is_payload(block: np.ndarray, bg: int) -> bool:
    return block.shape == (3, 3) and bool(np.all(block != bg))


def _is_tail(block: np.ndarray, bg: int) -> bool:
    vals = block[block != bg]
    return 1 <= vals.size <= 8 and len(set(map(int, vals))) == 1


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _rotate_payload(patch: np.ndarray, side: str) -> np.ndarray:
    out = patch.copy()
    if side == "left":
        out[:, 3:] = np.rot90(out[:, 3:], 2)
    elif side == "right":
        out[:, :3] = np.rot90(out[:, :3], 2)
    elif side == "top":
        out[3:, :] = np.rot90(out[3:, :], 2)
    else:
        out[:3, :] = np.rot90(out[:3, :], 2)
    return out


def _paste_checked(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = canvas[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    assert not np.any((view != bg) & mask), "output objects must not overlap"
    view[mask] = patch[mask]


def _parse_object(comp: dict, bg: int, shape: tuple[int, int]) -> dict:
    patch = comp["patch"]
    r0, c0, r1, c1 = comp["bbox"]
    if patch.shape == (3, 6):
        left = patch[:, :3]
        right = patch[:, 3:]
        if _is_tail(left, bg) and _is_payload(right, bg):
            side = "left"
            direction = (0, 1)
        elif _is_payload(left, bg) and _is_tail(right, bg):
            side = "right"
            direction = (0, -1)
        else:
            raise AssertionError("expected one tail half and one 3x3 payload half")
    elif patch.shape == (6, 3):
        top = patch[:3, :]
        bottom = patch[3:, :]
        if _is_tail(top, bg) and _is_payload(bottom, bg):
            side = "top"
            direction = (1, 0)
        elif _is_payload(top, bg) and _is_tail(bottom, bg):
            side = "bottom"
            direction = (-1, 0)
        else:
            raise AssertionError("expected one tail half and one 3x3 payload half")
    else:
        raise AssertionError("each object bbox must be 3x6 or 6x3")

    dr, dc = direction
    if (dr, dc) == (0, 1):
        dist = shape[1] - 1 - c1
    elif (dr, dc) == (0, -1):
        dist = c0
    elif (dr, dc) == (1, 0):
        dist = shape[0] - 1 - r1
    else:
        dist = r0

    return {
        "patch": patch,
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "dir": direction,
        "dist": int(dist),
    }


def _stopping_position(obj: dict, stopped: list[dict], shape: tuple[int, int]) -> tuple[int, int, bool]:
    r0, c0, r1, c1 = obj["bbox"]
    h, w = obj["patch"].shape
    dr, dc = obj["dir"]
    H, W = shape

    if (dr, dc) == (0, 1):
        final_c = W - w
        collided = False
        for other in stopped:
            or0, oc0, or1, oc1 = other["bbox"]
            if oc0 <= c0 or not _intervals_overlap(r0, r1, or0, or1):
                continue
            cand = oc0 - w
            if cand < final_c:
                final_c = cand
                collided = True
        assert final_c >= c0
        return r0, int(final_c), collided

    if (dr, dc) == (0, -1):
        final_c = 0
        collided = False
        for other in stopped:
            or0, oc0, or1, oc1 = other["bbox"]
            if oc1 >= c1 or not _intervals_overlap(r0, r1, or0, or1):
                continue
            cand = oc1 + 1
            if cand > final_c:
                final_c = cand
                collided = True
        assert final_c <= c0
        return r0, int(final_c), collided

    if (dr, dc) == (1, 0):
        final_r = H - h
        collided = False
        for other in stopped:
            or0, oc0, or1, oc1 = other["bbox"]
            if or0 <= r0 or not _intervals_overlap(c0, c1, oc0, oc1):
                continue
            cand = or0 - h
            if cand < final_r:
                final_r = cand
                collided = True
        assert final_r >= r0
        return int(final_r), c0, collided

    final_r = 0
    collided = False
    for other in stopped:
        or0, oc0, or1, oc1 = other["bbox"]
        if or1 >= r1 or not _intervals_overlap(c0, c1, oc0, oc1):
            continue
        cand = or1 + 1
        if cand > final_r:
            final_r = cand
            collided = True
    assert final_r <= r0
    return int(final_r), c0, collided


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each tailed payload away from its tail, stacking nearer ones first and flipping only collision-stopped payloads.

    Input structure:
    - The background contains several 8-connected composite objects.
    - Each object is a sparse monochrome 3x3 tail attached to one side of a fully filled 3x3 payload block.
    - Every composite bbox is either 3x6 or 6x3, so the tail uniquely indicates one of the four travel directions.

    Rule:
    - For each object, move it straight away from its tail.
    - Resolve objects from smallest initial distance to the target edge to largest, and let each one slide until its bbox touches the grid edge or an already stopped object.
    - Objects that stop on the wall keep their payload unchanged; objects that stop by collision rotate only the 3x3 payload 180 degrees while keeping the tail fixed in place.

    Key insight:
    - The rotation records a collision event rather than a property of the object, so the correct output requires an ordered stacking simulation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)

    objects = [_parse_object(comp, bg, g.shape) for comp in _find_components(g, bg)]
    assert objects, "expected at least one tailed object"

    objects.sort(key=lambda obj: (obj["dist"], obj["bbox"][0], obj["bbox"][1], obj["bbox"][2], obj["bbox"][3]))

    out = np.full_like(g, bg)
    stopped = []
    for obj in objects:
        r0, c0, collided = _stopping_position(obj, stopped, g.shape)
        patch = _rotate_payload(obj["patch"], obj["side"]) if collided else obj["patch"]
        _paste_checked(out, patch, r0, c0, bg)
        h, w = patch.shape
        stopped.append({"bbox": (r0, c0, r0 + h - 1, c0 + w - 1)})

    return out


def _tail_block(side: str, marker: int, bg: int) -> np.ndarray:
    block = np.full((3, 3), bg, dtype=int)
    for r, c in TAIL_COORDS[side]:
        block[r, c] = marker
    return block


def _payload_block(colors: list[int]) -> np.ndarray:
    while True:
        block = np.array(random.choices(colors, k=9), dtype=int).reshape(3, 3)
        if not np.array_equal(block, np.rot90(block, 2)):
            return block


def _composite_patch(side: str, payload: np.ndarray, marker: int, bg: int) -> np.ndarray:
    tail = _tail_block(side, marker, bg)
    if side == "left":
        return np.concatenate([tail, payload], axis=1)
    if side == "right":
        return np.concatenate([payload, tail], axis=1)
    if side == "top":
        return np.concatenate([tail, payload], axis=0)
    return np.concatenate([payload, tail], axis=0)


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _lane_starts(count: int, extent: int) -> list[int]:
    used = 3 * count + max(0, count - 1)
    slack = extent - used
    if slack < 0:
        raise ValueError("extent too small for requested lane count")
    gaps = _scatter(slack, count + 1)
    starts = []
    pos = gaps[0]
    for i in range(count):
        starts.append(pos)
        pos += 3 + (1 if i < count - 1 else 0) + gaps[i + 1]
    return starts


def _stack_lengths() -> list[int]:
    total = random.randint(2, GEN_MAX_OBJECTS)
    first = random.randint(2, min(3, total))
    lengths = [first]
    remaining = total - first
    while remaining > 0:
        size = random.randint(1, min(2, remaining))
        lengths.append(size)
        remaining -= size
    random.shuffle(lengths)
    return lengths


def _sample_deltas(max_shift: int, count: int) -> list[int]:
    if max_shift < count:
        raise ValueError("not enough room to separate input positions")
    return sorted(random.sample(range(1, max_shift + 1), count))


def _output_anchor(direction: str, lane: int, index: int, H: int, W: int) -> tuple[int, int]:
    if direction == "right":
        return lane, W - 6 * (index + 1)
    if direction == "left":
        return lane, 6 * index
    if direction == "down":
        return H - 6 * (index + 1), lane
    return 6 * index, lane


def _input_anchor(direction: str, lane: int, index: int, delta: int, H: int, W: int) -> tuple[int, int]:
    if direction == "right":
        return lane, W - 6 * (index + 1) - delta
    if direction == "left":
        return lane, 6 * index + delta
    if direction == "down":
        return H - 6 * (index + 1) - delta, lane
    return 6 * index + delta, lane


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    lengths = _stack_lengths()
    count = len(lengths)
    horizontal = random.random() < 0.5
    max_len = max(lengths)

    if horizontal:
        H = random.randint(max(9, 4 * count - 1), min(20, 4 * count + 6))
        W = random.randint(max(14, 7 * max_len), 28)
        lanes = _lane_starts(count, H)
        directions = [random.choice(["left", "right"]) for _ in lengths]
        max_shift = W - 6 * max_len
    else:
        H = random.randint(max(14, 7 * max_len), 28)
        W = random.randint(max(9, 4 * count - 1), min(20, 4 * count + 6))
        lanes = _lane_starts(count, W)
        directions = [random.choice(["up", "down"]) for _ in lengths]
        max_shift = H - 6 * max_len

    assert max_shift >= max_len
    bg = random.choice(range(10))
    marker = random.choice([c for c in range(10) if c != bg])
    payload_colors = [c for c in range(10) if c not in {bg, marker}]
    inp = np.full((H, W), bg, dtype=int)
    expected = np.full((H, W), bg, dtype=int)

    has_collision = False
    for stack_len, lane, direction in zip(lengths, lanes, directions):
        deltas = _sample_deltas(max_shift, stack_len)
        for index, delta in enumerate(deltas):
            payload = _payload_block(payload_colors)
            side = {
                "right": "left",
                "left": "right",
                "down": "top",
                "up": "bottom",
            }[direction]
            patch = _composite_patch(side, payload, marker, bg)
            in_r, in_c = _input_anchor(direction, lane, index, delta, H, W)
            out_r, out_c = _output_anchor(direction, lane, index, H, W)
            _paste_checked(inp, patch, in_r, in_c, bg)
            placed = _rotate_payload(patch, side) if index > 0 else patch
            _paste_checked(expected, placed, out_r, out_c, bg)
            has_collision |= index > 0

    assert has_collision
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            inp, expected = _build_case()
        except (AssertionError, ValueError):
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
