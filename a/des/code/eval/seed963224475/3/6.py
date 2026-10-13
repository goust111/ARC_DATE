import numpy as np


BG = 0
YELLOW = 4
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
AXES = ("row", "col")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _find_order_strip(grid: np.ndarray) -> tuple[list[int], int]:
    h, w = grid.shape
    start = h - 1
    while start >= 0 and int(grid[start, w - 1]) != BG:
        start -= 1
    start += 1
    assert start < h, "missing bottom-right order strip"
    colors = [int(grid[r, w - 1]) for r in range(start, h)]
    assert all(color != BG for color in colors)
    assert len(colors) == len(set(colors)), "strip colors must be unique"
    return colors, start


def _visible_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert len(coords) > 0, f"color {color} must remain visible"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 > r0 and c1 > c0, "frame bbox must have positive area"
    return r0, c0, r1, c1


def _marker_assignments(grid: np.ndarray, frame_colors: set[int]) -> dict[int, tuple[int, int]]:
    h, w = grid.shape
    markers = np.argwhere(grid == YELLOW)
    out: dict[int, tuple[int, int]] = {}
    for mr, mc in markers:
        neighbor_colors = {
            int(grid[nr, nc])
            for nr, nc in _neighbors4(int(mr), int(mc), h, w)
            if int(grid[nr, nc]) in frame_colors
        }
        assert len(neighbor_colors) == 1, "each yellow marker must touch exactly one frame color"
        color = next(iter(neighbor_colors))
        assert color not in out, "each frame color must have exactly one marker"
        out[color] = (int(mr), int(mc))
    assert set(out) == frame_colors, "every frame must have one visible marker"
    return out


def _infer_side(bbox: tuple[int, int, int, int], marker: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    mr, mc = marker
    if mr == r0 - 1 and c0 <= mc <= c1:
        return "top"
    if mr == r1 + 1 and c0 <= mc <= c1:
        return "bottom"
    if mc == c0 - 1 and r0 <= mr <= r1:
        return "left"
    if mc == c1 + 1 and r0 <= mr <= r1:
        return "right"
    raise AssertionError("marker must sit just outside one frame side")


def _middle_index(length: int) -> int:
    return (length - 1) // 2


def _extract_piece(
    grid: np.ndarray, bbox: tuple[int, int, int, int], side: str
) -> tuple[str, np.ndarray]:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert interior.shape[0] > 0 and interior.shape[1] > 0
    if side in ("top", "bottom"):
        mid = _middle_index(interior.shape[0])
        return "row", interior[mid : mid + 1, :].copy()
    mid = _middle_index(interior.shape[1])
    return "col", interior[:, mid : mid + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the color strip order, find each overlapping frame via its visible border,
    extract the visible middle row or column indicated by the yellow marker,
    and stack all slices into the compact output rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    order_colors, strip_start = _find_order_strip(g)
    work = g.copy()
    work[strip_start:, -1] = BG

    frame_colors = set(order_colors)
    marker_map = _marker_assignments(work, frame_colors)

    size = None
    kinds: list[str] = []
    pieces: list[np.ndarray] = []
    for color in order_colors:
        bbox = _visible_bbox(work, int(color))
        r0, c0, r1, c1 = bbox
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height == width >= 4, "every object must be a square frame"
        if size is None:
            size = height
        else:
            assert height == size, "all frames must share one size"

        side = _infer_side(bbox, marker_map[int(color)])
        kind, piece = _extract_piece(g, bbox, side)
        kinds.append(kind)
        pieces.append(piece)

    assert pieces, "must extract at least one slice"
    if all(kind == "row" for kind in kinds):
        return np.vstack(pieces)
    if all(kind == "col" for kind in kinds):
        return np.hstack(pieces)
    raise AssertionError("mixed row and column slices do not form one compact rectangle")


def _center_choices(size: int) -> list[int]:
    if size % 2 == 1:
        return [size // 2]
    return [size // 2 - 1, size // 2]


def _paint_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _render_borders(
    frames: list[dict[str, int | tuple[int, int, int, int]]],
    z_order: list[int],
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for idx in z_order:
        frame = frames[idx]
        _paint_border(grid, frame["bbox"], int(frame["color"]))  # type: ignore[index]
    return grid


def _marker_position(frame: dict[str, int | tuple[int, int, int, int]]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
    side = frame["side"]  # type: ignore[index]
    center = int(frame["center"])  # type: ignore[index]
    if side == "top":
        return r0 - 1, c0 + center
    if side == "bottom":
        return r1 + 1, c0 + center
    if side == "left":
        return r0 + center, c0 - 1
    return r0 + center, c1 + 1


def _render_scene(
    frames: list[dict[str, int | tuple[int, int, int, int]]],
    z_order: list[int],
    order_colors: list[int],
    shape: tuple[int, int],
) -> np.ndarray | None:
    h, w = shape
    grid = _render_borders(frames, z_order, shape)

    for frame in frames:
        mr, mc = _marker_position(frame)
        if not (0 <= mr < h and 0 <= mc < w):
            return None
        if int(grid[mr, mc]) != BG:
            return None
        grid[mr, mc] = YELLOW

    strip_start = h - len(order_colors)
    if np.any(grid[strip_start:, w - 1] != BG):
        return None
    if np.any(grid[:strip_start, w - 1] != BG):
        return None
    grid[strip_start:, w - 1] = np.array(order_colors, dtype=int)
    return grid


def _latent_output(
    grid: np.ndarray,
    frames_by_color: dict[int, dict[str, int | tuple[int, int, int, int]]],
    order_colors: list[int],
) -> np.ndarray:
    kinds: list[str] = []
    pieces: list[np.ndarray] = []
    for color in order_colors:
        frame = frames_by_color[color]
        kind, piece = _extract_piece(grid, frame["bbox"], str(frame["side"]))  # type: ignore[index]
        kinds.append(kind)
        pieces.append(piece)
    if all(kind == "row" for kind in kinds):
        return np.vstack(pieces)
    return np.hstack(pieces)


def _spatial_order(
    frames: list[dict[str, int | tuple[int, int, int, int]]], side: str
) -> list[int]:
    if side in ("top", "bottom"):
        return [
            int(frame["color"])
            for frame in sorted(frames, key=lambda item: int(item["bbox"][1]))  # type: ignore[index]
        ]
    return [
        int(frame["color"])
        for frame in sorted(frames, key=lambda item: int(item["bbox"][0]))  # type: ignore[index]
    ]


def _allowed_marker_options(
    frame_grid: np.ndarray,
    frame: dict[str, int | tuple[int, int, int, int]],
    side_choices: tuple[str, ...],
) -> list[tuple[str, int]]:
    h, w = frame_grid.shape
    color = int(frame["color"])
    options: list[tuple[str, int]] = []
    for side in side_choices:
        for center in _center_choices(int(frame["bbox"][2]) - int(frame["bbox"][0]) + 1):  # type: ignore[index]
            candidate = {"bbox": frame["bbox"], "side": side, "center": center}
            mr, mc = _marker_position(candidate)
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            if int(frame_grid[mr, mc]) != BG:
                continue
            neighbor_colors = {
                int(frame_grid[nr, nc])
                for nr, nc in _neighbors4(mr, mc, h, w)
                if int(frame_grid[nr, nc]) != BG
            }
            if neighbor_colors == {color}:
                options.append((side, center))
    return options


def _slice_has_occlusion(
    grid: np.ndarray, frame: dict[str, int | tuple[int, int, int, int]]
) -> bool:
    _kind, piece = _extract_piece(grid, frame["bbox"], str(frame["side"]))  # type: ignore[index]
    own = int(frame["color"])
    return bool(np.any((piece != BG) & (piece != own)))


def _validate_scene(
    grid: np.ndarray,
    frames_by_color: dict[int, dict[str, int | tuple[int, int, int, int]]],
    order_colors: list[int],
    expected: np.ndarray,
) -> bool:
    try:
        strip_colors, strip_start = _find_order_strip(grid)
        if strip_colors != order_colors:
            return False

        work = grid.copy()
        work[strip_start:, -1] = BG
        marker_map = _marker_assignments(work, set(order_colors))

        size = None
        for color in order_colors:
            bbox = _visible_bbox(work, color)
            frame = frames_by_color[color]
            if bbox != frame["bbox"]:
                return False
            if _infer_side(bbox, marker_map[color]) != frame["side"]:
                return False
            r0, c0, r1, c1 = bbox
            edge = r1 - r0 + 1
            if edge != c1 - c0 + 1:
                return False
            if size is None:
                size = edge
            elif edge != size:
                return False

        solved = main(grid)
    except Exception:
        return False
    return np.array_equal(solved, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(3, 5))
    size = int(rng.integers(6, 10))
    axis = str(rng.choice(AXES))
    colors = [int(x) for x in rng.choice(np.array(FRAME_COLORS), size=count, replace=False)]

    base_r = int(rng.integers(1, 4))
    base_c = int(rng.integers(1, 4))
    centers = _center_choices(size)
    step_lo = max(centers) + 1
    step_hi = size - 2
    step = int(rng.integers(step_lo, step_hi + 1))

    frames: list[dict[str, int | tuple[int, int, int, int]]] = []
    if axis == "row":
        lefts = [base_c + i * step for i in range(count)]
        max_right = max(lefts) + size - 1
        max_bottom = base_r + size - 1
        for color, left in zip(colors, lefts):
            bbox = (base_r, left, base_r + size - 1, left + size - 1)
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "side": str(rng.choice(("top", "bottom"))),
                    "center": int(rng.choice(centers)),
                }
            )
    else:
        tops = [base_r + i * step for i in range(count)]
        max_right = base_c + size - 1
        max_bottom = max(tops) + size - 1
        for color, top in zip(colors, tops):
            bbox = (top, base_c, top + size - 1, base_c + size - 1)
            frames.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "side": str(rng.choice(("left", "right"))),
                    "center": int(rng.choice(centers)),
                }
            )

    order_colors = [colors[i] for i in rng.permutation(count)]

    extra_bottom = int(rng.integers(0, 4))
    extra_right = int(rng.integers(0, 4))
    height = max_bottom + 2 + count + extra_bottom
    width = max_right + 3 + extra_right

    z_order = [int(i) for i in rng.permutation(count)]

    if axis == "row":
        spatial_orders = [_spatial_order(frames, "top")]
    else:
        spatial_orders = [_spatial_order(frames, "left")]
    if any(order_colors == spatial or order_colors == spatial[::-1] for spatial in spatial_orders):
        return None

    grid = _render_scene(frames, z_order, order_colors, (height, width))
    if grid is None:
        return None

    frames_by_color = {int(frame["color"]): frame for frame in frames}
    if not any(_slice_has_occlusion(grid, frame) for frame in frames):
        return None

    expected = _latent_output(grid, frames_by_color, order_colors)
    if not _validate_scene(grid, frames_by_color, order_colors, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
