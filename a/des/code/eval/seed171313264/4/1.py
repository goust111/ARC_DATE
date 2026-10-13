import random

import numpy as np


BG = 0
COLOR_POOL = list(range(1, 10))
QUAD_OFFSETS = [(0, 0), (0, 1), (1, 1), (1, 0)]  # tl, tr, br, bl
QUAD_ROTATIONS = [0, 3, 2, 1]  # rotation from base -> quadrant
GRID_BASE_RANGE = (28, 40)
NUM_CONTAINERS_CHOICES = [2, 3, 4]
QUAD_SIZE_CHOICES = [4, 5, 6]
GEN_RETRIES = 400

_RNG = random.Random()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def components_from_mask(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def bbox_from_coords(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def trim_mask(mask: np.ndarray):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return np.zeros((0, 0), dtype=bool), (0, 0)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0))


def shape_key(mask: np.ndarray):
    trimmed, _ = trim_mask(mask)
    return (tuple(trimmed.shape), trimmed.tobytes())


def rotations_unique(mask: np.ndarray) -> bool:
    seen = {shape_key(np.rot90(mask, k)) for k in range(4)}
    return len(seen) == 4


def component_to_mask(coords):
    r0, c0, r1, c1 = bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def is_frame_component(coords) -> bool:
    mask, (r0, c0, r1, c1) = component_to_mask(coords)
    h, w = mask.shape
    if h < 5 or w < 5:
        return False
    if h != w:
        return False
    if (h - 2) % 2 != 0:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        return False
    return r1 > r0 and c1 > c0


def extract_frames(grid: np.ndarray, bg: int):
    frames = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = components_from_mask(grid == color)
        for coords in comps:
            if is_frame_component(coords):
                r0, c0, r1, c1 = bbox_from_coords(coords)
                frames.append(
                    {
                        "color": color,
                        "bbox": (r0, c0, r1, c1),
                    }
                )
    return frames


def touches_marker(marker, key_coords) -> bool:
    mr, mc = marker
    for r, c in key_coords:
        if max(abs(r - mr), abs(c - mc)) == 1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each key to the container whose marker shares the frame color, infer the
    missing quadrant of the container's rotationally symmetric inner motif, rotate
    the key to match the missing shape, then fill the gap with the key color.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    frames = extract_frames(g, bg)
    frame_by_color = {frame["color"]: frame for frame in frames}
    assert frame_by_color, "expected at least one rectangular frame"

    key_components = []
    markers = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        for coords in components_from_mask(g == color):
            if color in frame_by_color and len(coords) == 1:
                markers[color] = coords[0]
                continue
            if color in frame_by_color:
                continue
            if len(coords) <= 1:
                continue
            mask, bbox = component_to_mask(coords)
            key_components.append(
                {
                    "color": color,
                    "coords": coords,
                    "mask": mask,
                    "bbox": bbox,
                }
            )

    key_by_frame_color = {}
    used_keys = set()
    for frame_color, marker in markers.items():
        matches = []
        for idx, key in enumerate(key_components):
            if idx in used_keys:
                continue
            if touches_marker(marker, key["coords"]):
                matches.append(idx)
        assert len(matches) == 1, "each marker must touch exactly one key"
        key_by_frame_color[frame_color] = key_components[matches[0]]
        used_keys.add(matches[0])

    for frame_color, frame in frame_by_color.items():
        assert frame_color in key_by_frame_color, "every frame needs a keyed marker"
        key = key_by_frame_color[frame_color]
        marker = markers[frame_color]

        for r, c in key["coords"]:
            out[r, c] = bg
        out[marker[0], marker[1]] = bg

        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        side = inner.shape[0]
        assert inner.shape[0] == inner.shape[1] and side % 2 == 0
        q = side // 2
        color_mask = inner == frame_color

        quadrants = []
        for qr, qc in QUAD_OFFSETS:
            quadrants.append(color_mask[qr * q : (qr + 1) * q, qc * q : (qc + 1) * q].copy())

        missing = [idx for idx, quad in enumerate(quadrants) if not quad.any()]
        assert len(missing) == 1, "exactly one quadrant should be missing"
        missing_idx = missing[0]

        base_mask = None
        for idx, quad in enumerate(quadrants):
            if quad.any():
                base_mask = np.rot90(quad, QUAD_ROTATIONS[idx])
                break
        assert base_mask is not None

        expected_quads = [np.rot90(base_mask, (4 - QUAD_ROTATIONS[idx]) % 4) for idx in range(4)]
        gap_full = expected_quads[missing_idx]
        gap_trim, (gr0, gc0) = trim_mask(gap_full)
        assert gap_trim.size > 0

        chosen_rot = None
        for rot in range(4):
            if np.array_equal(np.rot90(key["mask"], rot), gap_trim):
                chosen_rot = rot
                break
        assert chosen_rot is not None, "paired key must match the gap under rotation"

        qr, qc = QUAD_OFFSETS[missing_idx]
        top = r0 + 1 + qr * q + gr0
        left = c0 + 1 + qc * q + gc0
        for rr, cc in np.argwhere(gap_trim):
            out[top + rr, left + cc] = key["color"]

    return out


def random_connected_mask(q: int, rng: random.Random):
    allowed = [(r, c) for r in range(1, q - 1) for c in range(1, q - 1)]
    max_area = len(allowed)
    for _ in range(200):
        target = rng.randint(3, max(3, max_area - 1))
        seed = rng.choice(allowed)
        cells = {seed}
        while len(cells) < target:
            frontier = []
            for r, c in list(cells):
                for nr, nc in neighbors4(r, c):
                    if (nr, nc) in cells:
                        continue
                    if 1 <= nr < q and 1 <= nc < q:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != target:
            continue
        mask = np.zeros((q, q), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not rotations_unique(mask):
            continue
        trimmed, _ = trim_mask(mask)
        if trimmed.shape[0] == 1 and trimmed.shape[1] == 1:
            continue
        return mask
    raise RuntimeError("failed to build a non-symmetric base mask")


def build_container(q: int, frame_color: int, missing_idx: int, rng: random.Random):
    base_mask = random_connected_mask(q, rng)
    side = 2 * q + 2
    patch = np.full((side, side), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    quads = []
    for idx in range(4):
        quads.append(np.rot90(base_mask, (4 - QUAD_ROTATIONS[idx]) % 4))

    for idx, quad in enumerate(quads):
        if idx == missing_idx:
            continue
        qr, qc = QUAD_OFFSETS[idx]
        sub = patch[1 + qr * q : 1 + (qr + 1) * q, 1 + qc * q : 1 + (qc + 1) * q]
        sub[quad] = frame_color

    gap_full = quads[missing_idx]
    gap_trim, (gr0, gc0) = trim_mask(gap_full)
    return {
        "patch": patch,
        "gap_trim": gap_trim,
        "gap_offset": (gr0, gc0),
        "missing_idx": missing_idx,
        "q": q,
        "frame_color": frame_color,
    }


def marker_candidates(key_mask: np.ndarray):
    h, w = key_mask.shape
    occupied = {(r, c) for r, c in np.argwhere(key_mask)}
    candidates = []
    for mr in range(-1, h + 1):
        for mc in range(-1, w + 1):
            if (mr, mc) in occupied:
                continue
            diag = False
            ortho = False
            for r, c in occupied:
                dr = abs(r - mr)
                dc = abs(c - mc)
                if dr + dc == 1:
                    ortho = True
                if dr == 1 and dc == 1:
                    diag = True
            if diag and not ortho:
                candidates.append((mr, mc))
    return candidates


def box_is_free(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> bool:
    h, w = reserved.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()


def reserve_box(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int):
    h, w = reserved.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def place_containers(grid: np.ndarray, specs, rng: random.Random):
    h, w = grid.shape
    reserved = np.zeros((h, w), dtype=bool)
    placed = []
    for spec in sorted(specs, key=lambda item: item["patch"].shape[0], reverse=True):
        ph, pw = spec["patch"].shape
        ok = False
        for _ in range(300):
            r0 = rng.randint(1, h - ph - 1)
            c0 = rng.randint(1, w - pw - 1)
            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            if not box_is_free(reserved, r0, c0, r1, c1, pad=2):
                continue
            patch = spec["patch"]
            grid[r0 : r1 + 1, c0 : c1 + 1][patch != BG] = patch[patch != BG]
            reserve_box(reserved, r0, c0, r1, c1, pad=2)
            spec = dict(spec)
            spec["bbox"] = (r0, c0, r1, c1)
            placed.append(spec)
            ok = True
            break
        if not ok:
            return None, None
    return placed, reserved


def place_keys(grid: np.ndarray, reserved: np.ndarray, specs, rng: random.Random):
    h, w = grid.shape
    for spec in specs:
        gap_trim = spec["gap_trim"]
        key_rot = rng.randrange(4)
        key_mask = np.rot90(gap_trim, key_rot)
        candidates = marker_candidates(key_mask)
        if not candidates:
            return None
        key_h, key_w = key_mask.shape
        chosen = None
        for _ in range(500):
            marker_rel = rng.choice(candidates)
            min_r = min(0, marker_rel[0])
            max_r = max(key_h - 1, marker_rel[0])
            min_c = min(0, marker_rel[1])
            max_c = max(key_w - 1, marker_rel[1])
            group_h = max_r - min_r + 1
            group_w = max_c - min_c + 1
            group_r0 = rng.randint(0, h - group_h)
            group_c0 = rng.randint(0, w - group_w)
            key_r0 = group_r0 - min_r
            key_c0 = group_c0 - min_c
            marker = (key_r0 + marker_rel[0], key_c0 + marker_rel[1])
            gr0 = min(key_r0, marker[0])
            gc0 = min(key_c0, marker[1])
            gr1 = max(key_r0 + key_h - 1, marker[0])
            gc1 = max(key_c0 + key_w - 1, marker[1])
            edge_dist = min(gr0, gc0, h - 1 - gr1, w - 1 - gc1)
            if edge_dist > 3:
                continue
            if not box_is_free(reserved, gr0, gc0, gr1, gc1, pad=1):
                continue
            chosen = (key_mask, key_r0, key_c0, marker, gr0, gc0, gr1, gc1)
            break
        if chosen is None:
            return None
        key_mask, key_r0, key_c0, marker, gr0, gc0, gr1, gc1 = chosen
        grid[key_r0 : key_r0 + key_h, key_c0 : key_c0 + key_w][key_mask] = spec["key_color"]
        grid[marker] = spec["frame_color"]
        reserve_box(reserved, gr0, gc0, gr1, gc1, pad=1)
        spec["key_mask"] = key_mask
        spec["key_bbox"] = (key_r0, key_c0, key_r0 + key_h - 1, key_c0 + key_w - 1)
        spec["marker"] = marker
    return specs


def expected_output(inp: np.ndarray, specs):
    out = inp.copy()
    for spec in specs:
        kr0, kc0, _kr1, _kc1 = spec["key_bbox"]
        for rr, cc in np.argwhere(spec["key_mask"]):
            out[kr0 + rr, kc0 + cc] = BG
        mr, mc = spec["marker"]
        out[mr, mc] = BG

        r0, c0, _r1, _c1 = spec["bbox"]
        q = spec["q"]
        qr, qc = QUAD_OFFSETS[spec["missing_idx"]]
        gr0, gc0 = spec["gap_offset"]
        top = r0 + 1 + qr * q + gr0
        left = c0 + 1 + qc * q + gc0
        for rr, cc in np.argwhere(spec["gap_trim"]):
            out[top + rr, left + cc] = spec["key_color"]
    return out


def _gen_attempt(rng: random.Random):
    count = rng.choice(NUM_CONTAINERS_CHOICES)
    min_side = GRID_BASE_RANGE[0] + 2 * count
    max_side = GRID_BASE_RANGE[1] + 2 * count
    h = rng.randint(min_side, max_side)
    w = rng.randint(min_side, max_side)
    if 2 * count > len(COLOR_POOL):
        return None

    colors = rng.sample(COLOR_POOL, 2 * count)
    frame_colors = colors[:count]
    key_colors = colors[count:]
    missing_order = rng.sample([0, 1, 2, 3], 4)

    specs = []
    for idx in range(count):
        q = rng.choice(QUAD_SIZE_CHOICES)
        spec = build_container(q, frame_colors[idx], missing_order[idx % 4], rng)
        spec["key_color"] = key_colors[idx]
        specs.append(spec)

    grid = np.full((h, w), BG, dtype=int)
    placed, reserved = place_containers(grid, specs, rng)
    if placed is None:
        return None
    placed = place_keys(grid, reserved, placed, rng)
    if placed is None:
        return None

    expected = expected_output(grid, placed)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
