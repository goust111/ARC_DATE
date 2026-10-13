import numpy as np


BG = 0
GEN_ATTEMPTS = 300
RNG = np.random.default_rng()
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _normalize(coords) -> np.ndarray:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _shape_variants(base) -> list[np.ndarray]:
    base_arr = np.asarray(base, dtype=int)
    variants = {}
    for flip in (False, True):
        arr = base_arr.copy()
        if flip:
            arr[:, 1] *= -1
        for _ in range(4):
            arr = np.stack([arr[:, 1], -arr[:, 0]], axis=1)
            norm = _normalize(arr)
            variants[tuple(map(tuple, norm.tolist()))] = norm
    return list(variants.values())


_BASE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 1), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 0), (1, 1)],
]
SHAPES = []
_seen_shapes = set()
for _base in _BASE_SHAPES:
    for _variant in _shape_variants(_base):
        _key = tuple(map(tuple, _variant.tolist()))
        if _key not in _seen_shapes:
            _seen_shapes.add(_key)
            SHAPES.append(_variant)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
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
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _gap_from_side(values: np.ndarray, color: int) -> tuple[int, int] | None:
    inner = values[1:-1]
    present = inner == int(color)
    if np.all(present):
        return None
    missing = np.flatnonzero(~present)
    if len(missing) < 2:
        return (-1, -1)
    if missing[-1] - missing[0] + 1 != len(missing):
        return (-1, -1)
    if np.any(inner[missing] != BG):
        return (-1, -1)
    return (int(missing[0] + 1), int(missing[-1] + 1))


def _parse_frame(grid: np.ndarray, coords: np.ndarray, color: int):
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None
    border_mask = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    if not np.all(border_mask):
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    top = patch[0, :]
    bottom = patch[-1, :]
    left = patch[:, 0]
    right = patch[:, -1]

    if not (
        top[0] == color
        and top[-1] == color
        and bottom[0] == color
        and bottom[-1] == color
        and left[0] == color
        and left[-1] == color
        and right[0] == color
        and right[-1] == color
    ):
        return None

    gap_info = []
    for side, values in (
        ("top", top),
        ("bottom", bottom),
        ("left", left),
        ("right", right),
    ):
        gap = _gap_from_side(values, color)
        if gap == (-1, -1):
            return None
        if gap is not None:
            gap_info.append((side, gap))
        else:
            if np.any(values[1:-1] != color):
                return None

    if len(gap_info) != 1:
        return None

    gap_side, gap_span = gap_info[0]
    perimeter = 2 * h + 2 * w - 4
    gap_len = gap_span[1] - gap_span[0] + 1
    if len(coords) != perimeter - gap_len:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for coords in _components_of_color(grid, color):
            frame = _parse_frame(grid, coords, color)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1], f["color"]))
    assert frames, "expected at least one valid frame"

    used = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = np.zeros_like(grid, dtype=bool)
        interior[r0 + 1 : r1, c0 + 1 : c1] = True
        assert not np.any(used & interior), "frame interiors must not overlap"
        used |= interior
    return frames


def _find_fragments(inner: np.ndarray, frame_color: int) -> list[dict]:
    fragments = []
    for color in np.unique(inner):
        color = int(color)
        if color in (BG, int(frame_color)):
            continue
        for coords in _components_of_color(inner, color):
            assert 2 <= len(coords) <= 4, "fragments must be 2-4 pixels"
            fragments.append({"color": color, "coords": coords.copy()})
    return fragments


def _move_delta(gap_side: str) -> tuple[int, int]:
    if gap_side == "top":
        return (1, 0)
    if gap_side == "bottom":
        return (-1, 0)
    if gap_side == "left":
        return (0, 1)
    if gap_side == "right":
        return (0, -1)
    raise ValueError(f"unknown gap side: {gap_side}")


def _fragment_order_key(fragment: dict, gap_side: str):
    coords = fragment["coords"]
    if gap_side == "top":
        return (-int(coords[:, 0].max()), int(coords[:, 1].min()), int(coords[:, 0].min()))
    if gap_side == "bottom":
        return (int(coords[:, 0].min()), int(coords[:, 1].min()), int(coords[:, 0].max()))
    if gap_side == "left":
        return (-int(coords[:, 1].max()), int(coords[:, 0].min()), int(coords[:, 1].min()))
    if gap_side == "right":
        return (int(coords[:, 1].min()), int(coords[:, 0].min()), int(coords[:, 1].max()))
    raise ValueError(f"unknown gap side: {gap_side}")


def _stack_fragments(shape: tuple[int, int], fragments: list[dict], gap_side: str) -> list[np.ndarray]:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    dr, dc = _move_delta(gap_side)
    placed = []

    ordered = sorted(fragments, key=lambda frag: _fragment_order_key(frag, gap_side))
    for fragment in ordered:
        coords = fragment["coords"].copy()
        while True:
            nxt = coords + np.array([dr, dc], dtype=int)
            if (
                np.any(nxt[:, 0] < 0)
                or np.any(nxt[:, 0] >= h)
                or np.any(nxt[:, 1] < 0)
                or np.any(nxt[:, 1] >= w)
                or np.any(occupied[nxt[:, 0], nxt[:, 1]])
            ):
                break
            coords = nxt
        occupied[coords[:, 0], coords[:, 1]] = True
        placed.append(coords)
    return placed


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    out = g.copy()
    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        fragments = _find_fragments(inner, color)
        out[r0 + 1 : r1, c0 + 1 : c1] = BG
        for coords in _stack_fragments(inner.shape, fragments, frame["gap_side"]):
            out[r0 + 1 + coords[:, 0], c0 + 1 + coords[:, 1]] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each frame's inner fragments away from the border gap, then recolor the stacked result to the frame color.

    Input structure:
    - The background is black.
    - The grid contains several axis-aligned hollow rectangular frames, each drawn in one color with border thickness 1.
    - Every frame has exactly one continuous gap of black cells on one side, and its interior contains several small monochrome connected fragments.

    Rule:
    - Detect the missing side of each frame.
    - Move every interior fragment rigidly in the opposite direction, using gravity-like stacking where the fragment nearest the destination wall moves first.
    - Stop each fragment when its next step would hit the opposite inner wall or an already stacked fragment.
    - Keep the frame and its gap unchanged, but recolor all final fragment cells to the frame's own color.

    Key insight:
    - The gap is not an exit; it encodes the inverse gravity direction, and the original fragment colors are discarded after stacking.
    """
    return _solve(grid)


def _fits(shape: np.ndarray, h: int, w: int) -> bool:
    bh = int(shape[:, 0].max()) + 1
    bw = int(shape[:, 1].max()) + 1
    return bh <= h and bw <= w


def _metric_for_uniqueness(coords: np.ndarray, gap_side: str) -> int:
    if gap_side == "top":
        return int(coords[:, 0].max())
    if gap_side == "bottom":
        return int(coords[:, 0].min())
    if gap_side == "left":
        return int(coords[:, 1].max())
    if gap_side == "right":
        return int(coords[:, 1].min())
    raise ValueError(f"unknown gap side: {gap_side}")


def _can_place(coords: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = blocked.shape
    if np.any(coords[:, 0] < 0) or np.any(coords[:, 1] < 0):
        return False
    if np.any(coords[:, 0] >= h) or np.any(coords[:, 1] >= w):
        return False
    return not np.any(blocked[coords[:, 0], coords[:, 1]])


def _mark_blocked(blocked: np.ndarray, coords: np.ndarray) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for nr in range(max(0, int(r) - 1), min(h, int(r) + 2)):
            for nc in range(max(0, int(c) - 1), min(w, int(c) + 2)):
                blocked[nr, nc] = True


def _choose_fragment_count(inner_h: int, inner_w: int) -> int:
    area = inner_h * inner_w
    if area <= 28:
        return int(RNG.integers(2, 4))
    return int(RNG.integers(3, 5))


def _build_frame_patch(frame_h: int, frame_w: int, frame_color: int, gap_side: str):
    inner_h = frame_h - 2
    inner_w = frame_w - 2
    choices = [shape for shape in SHAPES if _fits(shape, inner_h, inner_w)]
    if not choices:
        return None

    patch = np.zeros((frame_h, frame_w), dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    side_len = frame_w if gap_side in ("top", "bottom") else frame_h
    gap_len = int(RNG.integers(2, side_len - 1))
    gap_start = int(RNG.integers(1, side_len - gap_len))
    gap_stop = gap_start + gap_len
    if gap_side == "top":
        patch[0, gap_start:gap_stop] = BG
    elif gap_side == "bottom":
        patch[-1, gap_start:gap_stop] = BG
    elif gap_side == "left":
        patch[gap_start:gap_stop, 0] = BG
    else:
        patch[gap_start:gap_stop, -1] = BG

    available_colors = [color for color in range(1, 10) if color != frame_color]
    frag_count = _choose_fragment_count(inner_h, inner_w)

    for _ in range(200):
        inner = np.zeros((inner_h, inner_w), dtype=int)
        blocked = np.zeros_like(inner, dtype=bool)
        metrics = set()
        fragments = []
        colors = RNG.choice(available_colors, size=frag_count, replace=False)
        success = True

        for color in colors:
            shape = choices[int(RNG.integers(0, len(choices)))]
            sh = int(shape[:, 0].max()) + 1
            sw = int(shape[:, 1].max()) + 1
            placed = False
            for _ in range(120):
                ar = int(RNG.integers(0, inner_h - sh + 1))
                ac = int(RNG.integers(0, inner_w - sw + 1))
                coords = shape + np.array([ar, ac], dtype=int)
                metric = _metric_for_uniqueness(coords, gap_side)
                if metric in metrics or not _can_place(coords, blocked):
                    continue
                inner[coords[:, 0], coords[:, 1]] = int(color)
                _mark_blocked(blocked, coords)
                metrics.add(metric)
                fragments.append({"color": int(color), "coords": coords.copy()})
                placed = True
                break
            if not placed:
                success = False
                break

        if not success:
            continue

        placed_coords = _stack_fragments(inner.shape, fragments, gap_side)
        moved = any(not np.array_equal(src["coords"], dst) for src, dst in zip(sorted(fragments, key=lambda frag: _fragment_order_key(frag, gap_side)), placed_coords))
        if not moved:
            continue

        patch[1:-1, 1:-1] = inner
        return patch

    return None


def _boxes_overlap(a, b, pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _gen_attempt():
    frame_count = int(RNG.integers(1, 4))
    canvas_h = int(RNG.integers(18, 31))
    canvas_w = int(RNG.integers(18, 31))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    directions = ["top", "bottom", "left", "right"]
    RNG.shuffle(directions)
    frame_colors = RNG.choice(np.arange(1, 10), size=frame_count, replace=False)

    patches = []
    for idx in range(frame_count):
        frame_h = int(RNG.integers(7, 12))
        frame_w = int(RNG.integers(7, 12))
        gap_side = directions[idx] if idx < len(directions) else directions[int(RNG.integers(0, 4))]
        patch = _build_frame_patch(frame_h, frame_w, int(frame_colors[idx]), gap_side)
        if patch is None:
            return None
        patches.append(patch)

    boxes = []
    for patch in patches:
        ph, pw = patch.shape
        placed = False
        for _ in range(200):
            r0 = int(RNG.integers(0, canvas_h - ph + 1))
            c0 = int(RNG.integers(0, canvas_w - pw + 1))
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_overlap(box, other, pad=2) for other in boxes):
                continue
            grid[r0 : r0 + ph, c0 : c0 + pw] = patch
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    try:
        out = _solve(grid)
    except AssertionError:
        return None

    if not np.any(out != grid):
        return None
    if len(_find_frames(grid)) != frame_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
