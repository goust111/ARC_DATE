from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DICT_ROWS = 4
OPS_BY_LABEL = {RED: "hflip", BLUE: "vflip", GREEN: "rot_cw"}
LABEL_BY_OP = {v: k for k, v in OPS_BY_LABEL.items()}
OPS = ("hflip", "vflip", "rot_cw")


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    pts = np.argwhere(mask)
    assert pts.size, "cannot normalize an empty mask"
    r0, c0, r1, c1 = _bbox(pts)
    crop = np.asarray(mask[r0:r1 + 1, c0:c1 + 1], dtype=bool)
    return tuple(map(int, crop.shape)), crop.tobytes()


def _transform_mask(mask: np.ndarray, op: str) -> np.ndarray:
    if op == "hflip":
        return np.fliplr(mask)
    if op == "vflip":
        return np.flipud(mask)
    if op == "rot_cw":
        return np.rot90(mask, 3)
    raise AssertionError(f"unknown operation: {op}")


def _parse_dictionary(g: np.ndarray) -> dict[tuple[tuple[int, int], bytes], str]:
    assert g.shape[0] >= DICT_ROWS and g.shape[1] >= 3
    mapping = {}
    for c in range(1, g.shape[1] - 1):
        label = int(g[3, c])
        if label not in OPS_BY_LABEL:
            continue
        mask = g[0:3, c - 1:c + 2] == GREY
        if not bool(mask.any()):
            continue
        key = _normalize(mask)
        assert key not in mapping, "dictionary masks must be unique"
        mapping[key] = OPS_BY_LABEL[label]
    assert mapping, "expected a top dictionary"
    return mapping


def _hole_key(object_mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    h, w = object_mask.shape
    holes = []
    for coords in _components(~object_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords):
            continue
        holes.append(coords)
    assert len(holes) == 1, "each object must contain exactly one enclosed hole"

    hr0, hc0, hr1, hc1 = _bbox(holes[0])
    hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
    hole_mask[holes[0][:, 0] - hr0, holes[0][:, 1] - hc0] = True
    return _normalize(hole_mask)


def _centered_top_left(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = map(int, shape)
    nr_num = int(r0 + r1 - (h - 1))
    nc_num = int(c0 + c1 - (w - 1))
    assert nr_num % 2 == 0 and nc_num % 2 == 0, "transformed object center must land on cells"
    return nr_num // 2, nc_num // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each hollow object by looking up its hole pattern in a top dictionary.

    Input structure:
    - The background is black.
    - The top four rows contain a horizontal dictionary: each entry has a 3x3
      grey mask in rows 0-2 and one label cell below its center in row 3.
    - Red labels mean horizontal flip, blue labels mean vertical flip, and
      green labels mean clockwise rotation by 90 degrees.
    - Below the dictionary are several 4-connected monochrome foreground
      objects, each enclosing exactly one black hole whose normalized shape
      matches one dictionary mask.

    Rule:
    - Read the mask-to-operation dictionary from the top rows.
    - For every foreground object below it, extract and normalize the enclosed
      black hole shape.
    - Use that hole shape to choose the object's operation, then transform the
      whole object mask around the object's center.
    - Return the transformed objects on black background, with the dictionary
      area erased.

    Key insight:
    The hole is an instruction code: identify the enclosed black shape, look up
    its label in the dictionary, and apply that geometric operation to the
    containing object.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    dictionary = _parse_dictionary(g)

    object_pixels = g != BLACK
    object_pixels[:DICT_ROWS, :] = False
    comps = _components(object_pixels, connectivity=4)

    out = np.full((h, w), BLACK, dtype=int)
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        ph, pw = r1 - r0 + 1, c1 - c0 + 1
        mask = np.zeros((ph, pw), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        colors = {int(g[int(r), int(c)]) for r, c in coords}
        assert len(colors) == 1, "objects must be monochrome"
        color = next(iter(colors))

        key = _hole_key(mask)
        assert key in dictionary, "object hole must match a dictionary mask"
        transformed = _transform_mask(mask, dictionary[key])
        nr0, nc0 = _centered_top_left(r0, c0, r1, c1, transformed.shape)
        th, tw = transformed.shape
        assert 0 <= nr0 and 0 <= nc0 and nr0 + th <= h and nc0 + tw <= w
        target = out[nr0:nr0 + th, nc0:nc0 + tw]
        assert not bool((target != BLACK)[transformed].any()), "transformed objects must not overlap"
        target[transformed] = color

    return out


_PATTERN_POOL = [
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=bool),
]

_OBJECT_COLORS = [4, 6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 500
_OBJECT_TOP_MIN = DICT_ROWS + 1


def _valid_object_mask(mask: np.ndarray, expected_key: tuple[tuple[int, int], bytes]) -> bool:
    try:
        comps = _components(mask, connectivity=4)
        return len(comps) == 1 and _hole_key(mask) == expected_key
    except AssertionError:
        return False


def _choose_patterns(rng: random.Random) -> dict[str, np.ndarray]:
    chosen = {}
    used = set()
    op_order = list(OPS)
    rng.shuffle(op_order)
    for op in op_order:
        candidates = [p for p in _PATTERN_POOL if not np.array_equal(_transform_mask(p, op), p)]
        rng.shuffle(candidates)
        for pat in candidates:
            key = _normalize(pat)
            if key not in used and len(_components(pat, connectivity=4)) == 1:
                chosen[op] = pat.copy()
                used.add(key)
                break
        else:
            return {}
    return chosen


def _make_object_mask(hole: np.ndarray, op: str, rng: random.Random) -> np.ndarray | None:
    key = _normalize(hole)
    for _ in range(80):
        top = rng.randint(1, 2)
        bottom = rng.randint(1, 2)
        left = rng.randint(1, 2)
        right = rng.randint(1, 2)
        if rng.random() < 0.25:
            side = rng.randrange(4)
            if side == 0:
                top += 1
            elif side == 1:
                bottom += 1
            elif side == 2:
                left += 1
            else:
                right += 1
        oh, ow = top + 3 + bottom, left + 3 + right
        if op == "rot_cw" and (oh - ow) % 2 != 0:
            continue
        mask = np.ones((oh, ow), dtype=bool)
        inner = mask[top:top + 3, left:left + 3]
        inner[hole] = False
        if _valid_object_mask(mask, key):
            return mask
    return None


def _paint_dictionary(grid: np.ndarray, patterns: dict[str, np.ndarray], rng: random.Random) -> None:
    order = list(OPS)
    rng.shuffle(order)
    gap = rng.choice([1, 2])
    total_w = 3 * len(order) + gap * (len(order) - 1)
    c = rng.randint(0, grid.shape[1] - total_w)
    for op in order:
        pat = patterns[op]
        grid[0:3, c:c + 3][pat] = GREY
        grid[3, c + 1] = LABEL_BY_OP[op]
        c += 3 + gap


def _can_place(occ: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int = 1) -> bool:
    h, w = shape
    H, W = occ.shape
    if r0 < _OBJECT_TOP_MIN or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark(occ: np.ndarray, r0: int, c0: int, shape: tuple[int, int], pad: int = 1) -> None:
    h, w = shape
    H, W = occ.shape
    occ[max(0, r0 - pad):min(H, r0 + h + pad),
        max(0, c0 - pad):min(W, c0 + w + pad)] = True


def _placement_candidates(mask: np.ndarray, transformed: np.ndarray,
                          occ_in: np.ndarray, occ_out: np.ndarray) -> list[tuple[int, int, int, int]]:
    H, W = occ_in.shape
    mh, mw = mask.shape
    candidates = []
    for r0 in range(_OBJECT_TOP_MIN, H - mh + 1):
        for c0 in range(0, W - mw + 1):
            r1, c1 = r0 + mh - 1, c0 + mw - 1
            try:
                nr0, nc0 = _centered_top_left(r0, c0, r1, c1, transformed.shape)
            except AssertionError:
                continue
            if not _can_place(occ_in, r0, c0, mask.shape):
                continue
            if not _can_place(occ_out, nr0, nc0, transformed.shape):
                continue
            candidates.append((r0, c0, nr0, nc0))
    return candidates


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    patterns = _choose_patterns(rng)
    if not patterns:
        return None

    object_count = rng.randint(3, 5)
    obj_ops = list(OPS)
    while len(obj_ops) < object_count:
        obj_ops.append(rng.choice(OPS))
    rng.shuffle(obj_ops)

    gap_for_width = 2
    min_w = 3 * len(OPS) + gap_for_width * (len(OPS) - 1) + 2
    H = rng.randint(18, 30)
    W = rng.randint(max(18, min_w), 30)

    grid = np.full((H, W), BLACK, dtype=int)
    expected = np.full((H, W), BLACK, dtype=int)
    _paint_dictionary(grid, patterns, rng)

    colors = _OBJECT_COLORS[:]
    rng.shuffle(colors)
    specs = []
    for i, op in enumerate(obj_ops):
        mask = _make_object_mask(patterns[op], op, rng)
        if mask is None:
            return None
        transformed = _transform_mask(mask, op)
        specs.append({"op": op, "mask": mask, "transformed": transformed, "color": colors[i]})

    specs.sort(key=lambda d: -max(d["mask"].size, d["transformed"].size))
    occ_in = np.zeros((H, W), dtype=bool)
    occ_out = np.zeros((H, W), dtype=bool)
    occ_in[:_OBJECT_TOP_MIN, :] = True
    occ_out[:_OBJECT_TOP_MIN, :] = True

    placements = []
    for spec in specs:
        candidates = _placement_candidates(spec["mask"], spec["transformed"], occ_in, occ_out)
        if not candidates:
            return None
        r0, c0, nr0, nc0 = rng.choice(candidates)
        placements.append((spec, r0, c0, nr0, nc0))
        _mark(occ_in, r0, c0, spec["mask"].shape)
        _mark(occ_out, nr0, nc0, spec["transformed"].shape)

    for spec, r0, c0, nr0, nc0 in placements:
        mask = spec["mask"]
        transformed = spec["transformed"]
        color = int(spec["color"])
        grid[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]][mask] = color
        expected[nr0:nr0 + transformed.shape[0], nc0:nc0 + transformed.shape[1]][transformed] = color

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
