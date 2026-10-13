import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACE_RETRIES = 300


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def is_connected(coords: np.ndarray) -> bool:
    if len(coords) <= 1:
        return True
    pts = {tuple(map(int, p)) for p in coords}
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS.values():
            q = (r + dr, c + dc)
            if q in pts and q not in seen:
                seen.add(q)
                stack.append(q)
    return len(seen) == len(pts)


def trim_body(body: np.ndarray, direction: str, keep_len: int) -> np.ndarray:
    r0, c0, r1, c1 = bbox(body)
    rel = body - np.array([r0, c0], dtype=int)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if direction == "right":
        keep = rel[:, 1] < keep_len
    elif direction == "left":
        keep = rel[:, 1] >= w - keep_len
    elif direction == "down":
        keep = rel[:, 0] < keep_len
    else:
        keep = rel[:, 0] >= h - keep_len
    return body[keep]


def shift_coords(coords: np.ndarray, direction: str, amount: int) -> np.ndarray:
    dr, dc = DIRS[direction]
    return coords + np.array([dr * amount, dc * amount], dtype=int)


def _tail_candidate(coords_set: set[tuple[int, int]], bb: tuple[int, int, int, int], direction: str):
    r0, c0, r1, c1 = bb
    tail: list[tuple[int, int]] = []
    anchor = None

    if direction == "right":
        fixed = None
        for c in range(c1, c0 - 1, -1):
            rows = [r for r in range(r0, r1 + 1) if (r, c) in coords_set]
            if len(rows) == 1 and (fixed is None or rows[0] == fixed):
                fixed = rows[0]
                tail.append((fixed, c))
            else:
                anchor = (fixed, c) if fixed is not None else None
                break
    elif direction == "left":
        fixed = None
        for c in range(c0, c1 + 1):
            rows = [r for r in range(r0, r1 + 1) if (r, c) in coords_set]
            if len(rows) == 1 and (fixed is None or rows[0] == fixed):
                fixed = rows[0]
                tail.append((fixed, c))
            else:
                anchor = (fixed, c) if fixed is not None else None
                break
    elif direction == "down":
        fixed = None
        for r in range(r1, r0 - 1, -1):
            cols = [c for c in range(c0, c1 + 1) if (r, c) in coords_set]
            if len(cols) == 1 and (fixed is None or cols[0] == fixed):
                fixed = cols[0]
                tail.append((r, fixed))
            else:
                anchor = (r, fixed) if fixed is not None else None
                break
    else:
        fixed = None
        for r in range(r0, r1 + 1):
            cols = [c for c in range(c0, c1 + 1) if (r, c) in coords_set]
            if len(cols) == 1 and (fixed is None or cols[0] == fixed):
                fixed = cols[0]
                tail.append((r, fixed))
            else:
                anchor = (r, fixed) if fixed is not None else None
                break

    if not (1 <= len(tail) <= 3) or anchor is None:
        return None
    if anchor not in coords_set:
        return None

    tail_arr = np.array(tail, dtype=int)
    body = np.array([p for p in coords_set if p not in set(tail)], dtype=int)
    if len(body) <= len(tail_arr) or not is_connected(body):
        return None

    br0, bc0, br1, bc1 = bbox(body)
    if direction in ("left", "right"):
        if bc1 - bc0 + 1 < 2:
            return None
    else:
        if br1 - br0 + 1 < 2:
            return None
    return direction, tail_arr, body


def split_object(coords: np.ndarray):
    coords_set = {tuple(map(int, p)) for p in coords}
    bb = bbox(coords)
    candidates = []
    for direction in ("up", "down", "left", "right"):
        cand = _tail_candidate(coords_set, bb, direction)
        if cand is not None:
            candidates.append(cand)
    assert len(candidates) == 1, "expected exactly one tail direction"
    return candidates[0]


def render_object(coords: np.ndarray, color: int, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in coords:
        out[int(r), int(c)] = color
    return out


def object_output_cells(coords: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    direction, _tail, body = split_object(coords)
    h, w = grid_shape
    r0, c0, r1, c1 = bbox(coords)
    br0, bc0, br1, bc1 = bbox(body)
    shift = {
        "right": w - 1 - c1,
        "left": c0,
        "down": h - 1 - r1,
        "up": r0,
    }[direction]
    step = (bc1 - bc0 + 1) if direction in ("left", "right") else (br1 - br0 + 1)

    parts = [shift_coords(coords, direction, shift)]
    idx = 0
    for amount in range(step, shift, step):
        keep_len = step - idx
        if keep_len <= 0:
            break
        parts.append(shift_coords(trim_body(body, direction, keep_len), direction, amount))
        idx += 1
    return np.vstack(parts)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        for comp in components_mask(g == color):
            direction, _tail, body = split_object(comp)
            r0, c0, r1, c1 = bbox(comp)
            br0, bc0, br1, bc1 = bbox(body)
            shift = {
                "right": w - 1 - c1,
                "left": c0,
                "down": h - 1 - r1,
                "up": r0,
            }[direction]
            step = (bc1 - bc0 + 1) if direction in ("left", "right") else (br1 - br0 + 1)

            idx = 0
            for amount in range(step, shift, step):
                keep_len = step - idx
                if keep_len <= 0:
                    break
                for rr, cc in shift_coords(trim_body(body, direction, keep_len), direction, amount):
                    out[int(rr), int(cc)] = color
                idx += 1

            for rr, cc in shift_coords(comp, direction, shift):
                out[int(rr), int(cc)] = color

    return out


def expand_cells(cells: np.ndarray, shape: tuple[int, int], radius: int = 1) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
                    mask[nr, nc] = True
    return mask


def make_rect_object(r0: int, c0: int, bh: int, bw: int, direction: str, tail_len: int, tail_offset: int) -> np.ndarray:
    body = [(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)]
    if direction == "right":
        tail = [(r0 + tail_offset, c0 + bw + i) for i in range(tail_len)]
    elif direction == "left":
        tail = [(r0 + tail_offset, c0 - 1 - i) for i in range(tail_len)]
    elif direction == "down":
        tail = [(r0 + bh + i, c0 + tail_offset) for i in range(tail_len)]
    else:
        tail = [(r0 - 1 - i, c0 + tail_offset) for i in range(tail_len)]
    return np.array(body + tail, dtype=int)


def sample_object(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    direction = str(rng.choice(["up", "down", "left", "right"]))
    bh = int(rng.integers(2, 5))
    bw = int(rng.integers(2, 5))
    tail_len = int(rng.integers(1, 4))
    if direction in ("left", "right"):
        tail_offset = int(rng.integers(0, bh))
        if direction == "right":
            max_c0 = w - (2 * bw + tail_len)
            if max_c0 < 1 or h - bh < 2:
                return None
            c0 = int(rng.integers(1, max_c0 + 1))
            r0 = int(rng.integers(1, h - bh))
        else:
            min_c0 = bw + tail_len
            if min_c0 > w - bw - 2 or h - bh < 2:
                return None
            c0 = int(rng.integers(min_c0, w - bw - 1))
            r0 = int(rng.integers(1, h - bh))
    else:
        tail_offset = int(rng.integers(0, bw))
        if direction == "down":
            max_r0 = h - (2 * bh + tail_len)
            if max_r0 < 1 or w - bw < 2:
                return None
            r0 = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, w - bw))
        else:
            min_r0 = bh + tail_len
            if min_r0 > h - bh - 2 or w - bw < 2:
                return None
            r0 = int(rng.integers(min_r0, h - bh - 1))
            c0 = int(rng.integers(1, w - bw))
    return make_rect_object(r0, c0, bh, bw, direction, tail_len, tail_offset)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        h = int(rng.integers(12, 21))
        w = int(rng.integers(12, 21))
        grid = np.zeros((h, w), dtype=int)
        out_mask = np.zeros((h, w), dtype=bool)
        inp_mask = np.zeros((h, w), dtype=bool)

        obj_count = int(rng.integers(2, 5))
        colors = list(rng.permutation(PALETTE))[:obj_count]
        ok = True
        for color in colors:
            placed = False
            for _ in range(PLACE_RETRIES):
                obj = sample_object(rng, (h, w))
                if obj is None:
                    continue
                obj_out = object_output_cells(obj, (h, w))
                if len({tuple(map(int, p)) for p in obj_out}) != len(obj_out):
                    continue
                if np.any(expand_cells(obj, (h, w)) & inp_mask):
                    continue
                if np.any(expand_cells(obj_out, (h, w)) & out_mask):
                    continue
                for r, c in obj:
                    grid[int(r), int(c)] = int(color)
                inp_mask |= expand_cells(obj, (h, w))
                out_mask |= expand_cells(obj_out, (h, w))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        if len(np.unique(grid)) < 3:
            continue
        transformed = main(grid)
        if np.array_equal(transformed, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
