import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_VEC = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                }
            )

    return comps


def _frame_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(comp["area"]) != int(2 * (h + w) - 4):
        return False

    local = np.zeros((h, w), dtype=bool)
    rel = comp["coords"] - np.array([[r0, c0]])
    local[rel[:, 0], rel[:, 1]] = True
    return np.array_equal(local, _frame_mask(h, w))


def _uniform_non_bg(values: np.ndarray) -> int:
    colors = {int(v) for v in values.tolist() if int(v) != BG}
    assert len(colors) == 1
    return colors.pop()


def _frame_side_info(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, str | None]:
    r0, c0, r1, c1 = bbox
    side_colors = {
        "top": _uniform_non_bg(grid[r0, c0 + 1 : c1]),
        "bottom": _uniform_non_bg(grid[r1, c0 + 1 : c1]),
        "left": _uniform_non_bg(grid[r0 + 1 : r1, c0]),
        "right": _uniform_non_bg(grid[r0 + 1 : r1, c1]),
    }
    counts: dict[int, int] = {}
    for color in side_colors.values():
        counts[color] = counts.get(color, 0) + 1

    if len(counts) == 1:
        return next(iter(counts)), None

    if len(counts) == 2:
        majority = [color for color, count in counts.items() if count == 3]
        minority = [color for color, count in counts.items() if count == 1]
        assert len(majority) == 1 and len(minority) == 1
        odd_side = next(name for name, color in side_colors.items() if color == minority[0])
        return int(majority[0]), odd_side

    raise AssertionError("frame must have either four equal sides or one odd side")


def _strictly_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and inner[2] < outer[2] and outer[1] < inner[1] and inner[3] < outer[3]


def _shift_to_boundary(
    bbox: tuple[int, int, int, int], side: str, shape: tuple[int, int]
) -> tuple[int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if side == "top":
        return -r0, 0
    if side == "bottom":
        return h - 1 - r1, 0
    if side == "left":
        return 0, -c0
    if side == "right":
        return 0, w - 1 - c1
    raise AssertionError("unknown side")


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _components(g != BG)
    frames = [comp for comp in comps if _is_frame_component(comp)]
    sprites = [comp for comp in comps if not _is_frame_component(comp)]

    frame_contents: list[list[int]] = [[] for _ in frames]
    for si, sprite in enumerate(sprites):
        owners = [fi for fi, frame in enumerate(frames) if _strictly_inside(sprite["bbox"], frame["bbox"])]
        assert len(owners) == 1, "every non-frame component must belong to exactly one frame"
        frame_contents[owners[0]].append(si)

    out = g.copy()
    for fi, frame in enumerate(frames):
        bbox = frame["bbox"]
        majority_color, odd_side = _frame_side_info(g, bbox)
        inside = frame_contents[fi]

        if odd_side is None:
            assert not inside, "closed frames must have empty interiors"
            continue

        assert len(inside) == 1, "open frames must contain exactly one sprite"
        sprite = sprites[inside[0]]
        coords = sprite["coords"]
        sprite_color = int(g[coords[0, 0], coords[0, 1]])

        out[coords[:, 0], coords[:, 1]] = BG
        _paint_frame(out, bbox, majority_color)

        dr, dc = _shift_to_boundary(sprite["bbox"], odd_side, g.shape)
        out[coords[:, 0] + dr, coords[:, 1] + dc] = sprite_color

    return out


def _normalize_cells(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _enumerate_sprite_masks() -> list[np.ndarray]:
    current = {((0, 0),)}
    states: list[tuple[tuple[int, int], ...]] = []

    for _size in range(2, 6):
        nxt: set[tuple[tuple[int, int], ...]] = set()
        for state in current:
            cells = set(state)
            frontier = set()
            for r, c in state:
                for dr, dc in DIR4:
                    nb = (r + dr, c + dc)
                    if nb not in cells:
                        frontier.add(nb)
            for nb in frontier:
                nxt.add(_normalize_cells(tuple(cells | {nb})))
        current = nxt
        states.extend(sorted(current))

    masks = []
    for state in states:
        h = max(r for r, _ in state) + 1
        w = max(c for _, c in state) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in state:
            mask[r, c] = True
        masks.append(mask)
    return masks


SPRITE_MASKS = _enumerate_sprite_masks()
GEN_GRID_H = (18, 27)
GEN_GRID_W = (18, 27)
GEN_FRAME_COUNT = (2, 4)
GEN_FRAME_SIDE = (6, 11)
GEN_PAD = 1
GEN_ATTEMPTS = 400
GEN_PLACE_ATTEMPTS = 500
COLORS = tuple(range(1, 10))


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _choice(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - pad),
        max(0, box[1] - pad),
        min(h - 1, box[2] + pad),
        min(w - 1, box[3] + pad),
    )


def _corridor_box(
    sprite_box: tuple[int, int, int, int], side: str, h: int, w: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = sprite_box
    if side == "top":
        return 0, c0, r1, c1
    if side == "bottom":
        return r0, c0, h - 1, c1
    if side == "left":
        return r0, 0, r1, c1
    if side == "right":
        return r0, c0, r1, w - 1
    raise AssertionError("unknown side")


def _shift_box(box: tuple[int, int, int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    return box[0] + dr, box[1] + dc, box[2] + dr, box[3] + dc


def _mask_to_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords[:, 0] += top
    coords[:, 1] += left
    return coords


def _draw_input_frame(grid: np.ndarray, spec: dict) -> None:
    _paint_frame(grid, spec["bbox"], spec["majority"])
    if spec["odd_side"] is not None:
        r0, c0, r1, c1 = spec["bbox"]
        odd_color = spec["odd_color"]
        if spec["odd_side"] == "top":
            grid[r0, c0 + 1 : c1] = odd_color
        elif spec["odd_side"] == "bottom":
            grid[r1, c0 + 1 : c1] = odd_color
        elif spec["odd_side"] == "left":
            grid[r0 + 1 : r1, c0] = odd_color
        else:
            grid[r0 + 1 : r1, c1] = odd_color
        coords = spec["sprite_coords"]
        grid[coords[:, 0], coords[:, 1]] = spec["sprite_color"]


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        if spec["odd_side"] is None:
            continue
        coords = spec["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = BG
        _paint_frame(out, spec["bbox"], spec["majority"])
        dr, dc = spec["shift"]
        out[coords[:, 0] + dr, coords[:, 1] + dc] = spec["sprite_color"]
    return out


def _sample_open_spec(
    rng: np.random.Generator,
    h: int,
    w: int,
    majority: int,
    reserved: list[tuple[int, int, int, int]],
) -> dict | None:
    mask = _choice(rng, SPRITE_MASKS)
    sh, sw = map(int, mask.shape)
    min_fh = max(GEN_FRAME_SIDE[0], sh + 4)
    min_fw = max(GEN_FRAME_SIDE[0], sw + 4)
    max_fh = GEN_FRAME_SIDE[1]
    max_fw = GEN_FRAME_SIDE[1]
    if min_fh > max_fh or min_fw > max_fw:
        return None

    odd_side = _choice(rng, ("top", "bottom", "left", "right"))
    fh = _randint(rng, min_fh, max_fh)
    fw = _randint(rng, min_fw, max_fw)

    if odd_side == "top":
        r0_min, r0_max = sh, h - fh
        c0_min, c0_max = 0, w - fw
    elif odd_side == "bottom":
        r0_min, r0_max = 0, h - fh - sh
        c0_min, c0_max = 0, w - fw
    elif odd_side == "left":
        r0_min, r0_max = 0, h - fh
        c0_min, c0_max = sw, w - fw
    else:
        r0_min, r0_max = 0, h - fh
        c0_min, c0_max = 0, w - fw - sw

    if r0_min > r0_max or c0_min > c0_max:
        return None

    r0 = _randint(rng, r0_min, r0_max)
    c0 = _randint(rng, c0_min, c0_max)
    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)

    inner_h = fh - 2
    inner_w = fw - 2
    sr_local = _randint(rng, 1, inner_h - sh - 1)
    sc_local = _randint(rng, 1, inner_w - sw - 1)
    sr0 = r0 + 1 + sr_local
    sc0 = c0 + 1 + sc_local
    sprite_box = (sr0, sc0, sr0 + sh - 1, sc0 + sw - 1)

    shift = _shift_to_boundary(sprite_box, odd_side, (h, w))
    final_box = _shift_box(sprite_box, shift[0], shift[1])
    corridor = _corridor_box(sprite_box, odd_side, h, w)

    odd_color = _choice(rng, [c for c in COLORS if c != majority])
    sprite_color = _choice(rng, [c for c in COLORS if c not in (majority, odd_color)])
    frame_zone = _expand_box(bbox, GEN_PAD, h, w)
    path_zone = _expand_box(corridor, GEN_PAD, h, w)

    if _boxes_overlap(frame_zone, final_box) or _boxes_overlap(bbox, final_box):
        return None
    if any(_boxes_overlap(frame_zone, box) or _boxes_overlap(path_zone, box) or _boxes_overlap(final_box, box) for box in reserved):
        return None

    return {
        "bbox": bbox,
        "majority": majority,
        "odd_side": odd_side,
        "odd_color": odd_color,
        "sprite_color": sprite_color,
        "sprite_coords": _mask_to_coords(mask, sr0, sc0),
        "shift": shift,
        "reserve": [frame_zone, path_zone],
    }


def _sample_closed_spec(
    rng: np.random.Generator,
    h: int,
    w: int,
    majority: int,
    reserved: list[tuple[int, int, int, int]],
) -> dict | None:
    fh = _randint(rng, GEN_FRAME_SIDE[0], GEN_FRAME_SIDE[1])
    fw = _randint(rng, GEN_FRAME_SIDE[0], GEN_FRAME_SIDE[1])
    r0 = _randint(rng, 0, h - fh)
    c0 = _randint(rng, 0, w - fw)
    bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
    frame_zone = _expand_box(bbox, GEN_PAD, h, w)
    if any(_boxes_overlap(frame_zone, box) for box in reserved):
        return None
    return {
        "bbox": bbox,
        "majority": majority,
        "odd_side": None,
        "odd_color": None,
        "sprite_color": None,
        "sprite_coords": None,
        "shift": None,
        "reserve": [frame_zone],
    }


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = _randint(rng, *GEN_GRID_H)
    w = _randint(rng, *GEN_GRID_W)
    total_frames = _randint(rng, *GEN_FRAME_COUNT)
    open_frames = _randint(rng, 1, total_frames - 1)
    frame_kinds = [True] * open_frames + [False] * (total_frames - open_frames)
    rng.shuffle(frame_kinds)

    majority_colors = rng.choice(np.array(COLORS, dtype=int), size=total_frames, replace=False)
    reserved: list[tuple[int, int, int, int]] = []
    specs: list[dict] = []

    for idx, is_open in enumerate(frame_kinds):
        majority = int(majority_colors[idx])
        placed = None
        for _ in range(GEN_PLACE_ATTEMPTS):
            if is_open:
                spec = _sample_open_spec(rng, h, w, majority, reserved)
            else:
                spec = _sample_closed_spec(rng, h, w, majority, reserved)
            if spec is None:
                continue
            placed = spec
            break
        if placed is None:
            return None
        specs.append(placed)
        reserved.extend(placed["reserve"])

    inp = np.zeros((h, w), dtype=int)
    for spec in specs:
        _draw_input_frame(inp, spec)

    expected = _render_expected(inp, specs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
