import random
import numpy as np


BG = 0
FRAME_COLORS = (1, 2, 3)
MOSAIC_COLORS = (4, 6, 8)
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATE_MAX_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            pts.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _coords_key(coords: np.ndarray) -> bytes:
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order].astype(np.int16, copy=False).tobytes()


def _shape_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    mask = np.asarray(mask, dtype=bool)
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8, copy=False).tobytes()


def _frame_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] == color):
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "key": _coords_key(coords),
    }


def _find_frames(grid: np.ndarray) -> tuple[list[dict], dict[int, list[np.ndarray]]]:
    comps_by_color: dict[int, list[np.ndarray]] = {}
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) in FRAME_COLORS):
        comps = _components(grid == color)
        comps_by_color[int(color)] = comps
        for coords in comps:
            info = _frame_info(grid, int(color), coords)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames, comps_by_color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, comps_by_color = _find_frames(g)
    assert frames, "expected at least one framed pattern"

    for frame in frames:
        color = int(frame["color"])
        fr0, fc0, fr1, fc1 = frame["bbox"]
        interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        hole_comps = _components(interior == BG)
        assert len(hole_comps) == 1, "each frame interior must contain exactly one black hole"
        hole_coords = hole_comps[0]
        hole_shape = _mask_from_coords(hole_coords)

        matches = []
        for comp in comps_by_color[color]:
            if _coords_key(comp) == frame["key"]:
                continue
            if np.array_equal(_mask_from_coords(comp), hole_shape):
                matches.append(comp)
        assert len(matches) == 1, "each hole must have a unique same-color patch"
        patch_coords = matches[0]

        width = int(interior.shape[1])
        for rr, cc in hole_coords:
            mirror_c = width - 1 - int(cc)
            value = int(interior[int(rr), mirror_c])
            assert value != BG, "hole mirror counterpart must be intact"
            out[fr0 + 1 + int(rr), fc0 + 1 + int(cc)] = value

        out[patch_coords[:, 0], patch_coords[:, 1]] = BG

    return out


def _make_symmetric_mosaic(h: int, w: int, colors: tuple[int, ...]) -> np.ndarray:
    half = w // 2
    for _ in range(40):
        left = np.zeros((h, half), dtype=int)
        for r in range(h):
            for c in range(half):
                if c > 0 and random.random() < 0.45:
                    left[r, c] = left[r, c - 1]
                elif r > 0 and random.random() < 0.30:
                    left[r, c] = left[r - 1, c]
                else:
                    left[r, c] = random.choice(colors)

        mosaic = np.zeros((h, w), dtype=int)
        if half > 0:
            mosaic[:, :half] = left
            mosaic[:, w - half :] = np.fliplr(left)
        if w % 2 == 1:
            center = np.zeros(h, dtype=int)
            for r in range(h):
                if r > 0 and random.random() < 0.35:
                    center[r] = center[r - 1]
                else:
                    center[r] = random.choice(colors)
            mosaic[:, half] = center
        if len(set(int(x) for x in np.unique(mosaic))) >= 2:
            return mosaic

    mosaic = np.full((h, w), colors[0], dtype=int)
    alt = colors[1]
    for r in range(h):
        c = r % max(1, half)
        mosaic[r, c] = alt
        mosaic[r, w - 1 - c] = alt
    return mosaic


def _grow_connected(allowed: list[tuple[int, int]], size: int) -> np.ndarray | None:
    allowed_set = set(allowed)
    chosen = {random.choice(allowed)}
    while len(chosen) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in chosen:
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in allowed_set and nxt not in chosen:
                    frontier.add(nxt)
        if not frontier:
            return None
        chosen.add(random.choice(tuple(frontier)))
    return np.array(sorted(chosen), dtype=int)


def _make_hole(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    half = w // 2
    assert half >= 1
    side = random.choice(("left", "right"))
    rows = range(1, h - 1)
    if side == "left":
        cols = range(1, half)
    else:
        start = w - half
        cols = range(start, w - 1)
    allowed = [(r, c) for r in rows for c in cols]
    if not allowed:
        raise RuntimeError("failed to build a valid hole")

    max_cells = min(6, len(allowed))
    for _ in range(60):
        target = random.randint(1, max_cells)
        coords = _grow_connected(allowed, target)
        if coords is None:
            continue
        chosen = {(int(r), int(c)) for r, c in coords}
        mirrors = {(r, w - 1 - c) for r, c in chosen}
        if chosen & mirrors:
            continue
        return coords, _mask_from_coords(coords)
    raise RuntimeError("failed to build a valid hole")


def _paint(grid: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    sub = grid[top : top + h, left : left + w]
    mask = sprite != BG
    sub[mask] = sprite[mask]


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _place_bbox(
    canvas_shape: tuple[int, int],
    placed: list[dict],
    box_shape: tuple[int, int],
    pad: int,
) -> tuple[int, int] | None:
    h, w = canvas_shape
    bh, bw = box_shape
    if bh > h or bw > w:
        return None
    row_min = 1 if h - bh >= 2 else 0
    row_max = h - bh - 1 if h - bh >= 2 else h - bh
    col_min = 1 if w - bw >= 2 else 0
    col_max = w - bw - 1 if w - bw >= 2 else w - bw
    for _ in range(300):
        top = random.randint(row_min, row_max)
        left = random.randint(col_min, col_max)
        rect = (top, left, top + bh - 1, left + bw - 1)
        ok = True
        for item in placed:
            if not _separated(rect, item["bbox"], max(pad, int(item["pad"]))):
                ok = False
                break
        if ok:
            placed.append({"bbox": rect, "pad": int(pad)})
            return top, left
    return None


def _make_grey_distractor(forbidden: set[tuple[int, int, bytes]]) -> np.ndarray:
    for _ in range(80):
        bh = random.randint(1, 4)
        bw = random.randint(1, 4)
        allowed = [(r, c) for r in range(bh) for c in range(bw)]
        size = random.randint(1, min(5, bh * bw))
        coords = _grow_connected(allowed, size)
        if coords is None:
            continue
        mask = _mask_from_coords(coords)
        if _shape_signature(mask) in forbidden:
            continue
        sprite = np.zeros(mask.shape, dtype=int)
        sprite[mask] = GREY
        return sprite
    raise RuntimeError("failed to build grey distractor")


def _build_frame_spec(frame_color: int) -> dict:
    inner_h = random.randint(5, 9)
    inner_w = random.randint(5, 10)
    color_count = random.randint(2, 3)
    mosaic_colors = tuple(random.sample(MOSAIC_COLORS, color_count))
    mosaic = _make_symmetric_mosaic(inner_h, inner_w, mosaic_colors)
    hole_coords, hole_shape = _make_hole(inner_h, inner_w)

    solved = np.full((inner_h + 2, inner_w + 2), frame_color, dtype=int)
    solved[1:-1, 1:-1] = mosaic

    inp = solved.copy()
    inp[1 + hole_coords[:, 0], 1 + hole_coords[:, 1]] = BG

    patch = np.zeros(hole_shape.shape, dtype=int)
    patch[hole_shape] = frame_color

    return {"input": inp, "solved": solved, "patch": patch, "hole_sig": _shape_signature(hole_shape)}


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.randint(2, 3)
    frame_colors = random.sample(FRAME_COLORS, num_frames)
    frame_specs = [_build_frame_spec(color) for color in frame_colors]
    forbidden_holes = {spec["hole_sig"] for spec in frame_specs}

    grey_count = random.randint(1, 2)
    grey_sprites = [_make_grey_distractor(forbidden_holes) for _ in range(grey_count)]

    h = random.randint(24, 36)
    w = random.randint(24, 36)
    inp = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    placed: list[dict] = []

    frame_order = list(range(num_frames))
    frame_order.sort(key=lambda idx: frame_specs[idx]["input"].shape[0] * frame_specs[idx]["input"].shape[1], reverse=True)
    frame_positions: dict[int, tuple[int, int]] = {}
    for idx in frame_order:
        spec = frame_specs[idx]
        pos = _place_bbox((h, w), placed, spec["input"].shape, pad=2)
        if pos is None:
            return None
        frame_positions[idx] = pos
        _paint(inp, spec["input"], pos[0], pos[1])
        _paint(expected, spec["solved"], pos[0], pos[1])

    misc = [{"kind": "patch", "sprite": spec["patch"]} for spec in frame_specs]
    misc.extend({"kind": "grey", "sprite": sprite} for sprite in grey_sprites)
    random.shuffle(misc)
    for item in misc:
        pos = _place_bbox((h, w), placed, item["sprite"].shape, pad=1)
        if pos is None:
            return None
        _paint(inp, item["sprite"], pos[0], pos[1])
        if item["kind"] == "grey":
            _paint(expected, item["sprite"], pos[0], pos[1])

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
