import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

HFLIP_COLOR = RED
VFLIP_COLOR = BLUE
ROTATE_CW_COLOR = GREEN

FRAME_CORE_COLORS = (4, 5, 6, 7, 8, 9)
LAYER_COUNT = 4
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _read_legend_mapping(grid: np.ndarray) -> dict[int, int]:
    h, w = grid.shape
    patches = (
        grid[0:2, 0:2],
        grid[0:2, w - 2 : w],
        grid[h - 2 : h, 0:2],
        grid[h - 2 : h, w - 2 : w],
    )

    mapping: dict[int, int] = {}
    for patch in patches:
        assert patch.shape == (2, 2)
        top = int(patch[0, 0])
        bottom = int(patch[1, 0])
        assert np.all(patch[0, :] == top)
        assert np.all(patch[1, :] == bottom)
        assert bottom in (HFLIP_COLOR, VFLIP_COLOR, ROTATE_CW_COLOR)
        if top in mapping:
            assert mapping[top] == bottom
        mapping[top] = bottom
    return mapping


def _extract_central_object(grid: np.ndarray) -> np.ndarray:
    mask = grid != BLACK
    h, w = grid.shape
    mask[0:2, 0:2] = False
    mask[0:2, w - 2 : w] = False
    mask[h - 2 : h, 0:2] = False
    mask[h - 2 : h, w - 2 : w] = False

    comps = _components(mask)
    assert comps, "expected a central non-black object"
    comp = max(comps, key=len)
    r0, c0, r1, c1 = _bbox(comp)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _outer_border_color(arr: np.ndarray) -> int | None:
    h, w = arr.shape
    if h == 0 or w == 0:
        return None
    border = [
        arr[0, :].ravel(),
        arr[h - 1, :].ravel(),
    ]
    if h > 2:
        border.append(arr[1 : h - 1, 0].ravel())
        if w > 1:
            border.append(arr[1 : h - 1, w - 1].ravel())
    elif w > 1:
        border.append(arr[:, 0].ravel())
        border.append(arr[:, w - 1].ravel())
    vals = np.concatenate(border)
    color = int(vals[0])
    if np.all(vals == color):
        return color
    return None


def _apply_forward(arr: np.ndarray, op_color: int) -> np.ndarray:
    if op_color == HFLIP_COLOR:
        return np.fliplr(arr)
    if op_color == VFLIP_COLOR:
        return np.flipud(arr)
    if op_color == ROTATE_CW_COLOR:
        return np.rot90(arr, -1)
    raise AssertionError(f"unknown operation color: {op_color}")


def _apply_inverse(arr: np.ndarray, op_color: int) -> np.ndarray:
    if op_color in (HFLIP_COLOR, VFLIP_COLOR):
        return _apply_forward(arr, op_color)
    if op_color == ROTATE_CW_COLOR:
        return np.rot90(arr, 1)
    raise AssertionError(f"unknown operation color: {op_color}")


def _add_border(arr: np.ndarray, color: int) -> np.ndarray:
    h, w = arr.shape
    out = np.full((h + 2, w + 2), color, dtype=int)
    out[1:-1, 1:-1] = arr
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner legend, peel concentric rectangular frames from outside in,
    and apply the mapped transform after each peel. Return the final core only.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping = _read_legend_mapping(g)
    current = _extract_central_object(g)
    used: set[int] = set()

    while current.shape[0] > 2 and current.shape[1] > 2:
        frame_color = _outer_border_color(current)
        if frame_color is None or frame_color not in mapping or frame_color in used:
            break
        used.add(frame_color)
        current = _apply_forward(current[1:-1, 1:-1], mapping[frame_color])

    return current.copy()


def _choice(values) -> int:
    return int(values[int(_RNG.integers(0, len(values)))])


def _distinct_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts = [
        (0, min(1, w - 1)),
        (h // 2, w - 1),
        (h - 1, 0),
        (max(0, h // 3), max(0, w // 2 - 1)),
    ]
    out: list[tuple[int, int]] = []
    seen = set()
    for pt in pts:
        if pt not in seen:
            seen.add(pt)
            out.append(pt)
    return out


def _make_core(height: int, width: int, colors: tuple[int, int]) -> np.ndarray:
    c0, c1 = colors
    rr, cc = np.indices((height, width))
    style = int(_RNG.integers(0, 5))

    if style == 0:
        mask = (rr + cc) % 2 == 0
    elif style == 1:
        mask = rr % 2 == 0
    elif style == 2:
        mask = cc % 2 == 0
    elif style == 3:
        mask = (2 * rr + cc) % 3 == 0
    else:
        mask = _RNG.random((height, width)) < 0.5

    for i, (r, c) in enumerate(_distinct_positions(height, width)):
        mask[r, c] = (i % 2) == 0

    core = np.where(mask, c0, c1).astype(int)
    return core


def _is_informative(core: np.ndarray) -> bool:
    if len(np.unique(core)) != 2:
        return False
    if np.array_equal(core, np.fliplr(core)):
        return False
    if np.array_equal(core, np.flipud(core)):
        return False
    if core.shape[0] == core.shape[1] and np.array_equal(core, np.rot90(core, -1)):
        return False
    return True


def _legend_patch(frame_color: int, op_color: int) -> np.ndarray:
    return np.array(
        [[frame_color, frame_color], [op_color, op_color]],
        dtype=int,
    )


def _place_legends(grid: np.ndarray, frame_colors: list[int], op_colors: list[int]) -> None:
    corners = [
        (0, 0),
        (0, grid.shape[1] - 2),
        (grid.shape[0] - 2, 0),
        (grid.shape[0] - 2, grid.shape[1] - 2),
    ]
    for idx, corner_idx in enumerate(_RNG.permutation(len(corners))):
        r0, c0 = corners[corner_idx]
        grid[r0 : r0 + 2, c0 : c0 + 2] = _legend_patch(frame_colors[idx], op_colors[idx])


def _sample_core_shape() -> tuple[int, int]:
    if _RNG.random() < 0.7:
        h = int(_RNG.integers(3, 8))
        w = int(_RNG.integers(3, 8))
        if h == w:
            w = w + 1 if w < 7 else w - 1
    else:
        h = int(_RNG.integers(3, 7))
        w = int(_RNG.integers(3, 7))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    color_perm = list(map(int, _RNG.permutation(np.array(FRAME_CORE_COLORS))))
    frame_colors = color_perm[:LAYER_COUNT]
    core_colors = (color_perm[LAYER_COUNT], color_perm[LAYER_COUNT + 1])

    op_colors = [HFLIP_COLOR, VFLIP_COLOR, ROTATE_CW_COLOR, _choice((HFLIP_COLOR, VFLIP_COLOR, ROTATE_CW_COLOR))]
    op_colors = [int(x) for x in op_colors]
    _RNG.shuffle(op_colors)

    core_h, core_w = _sample_core_shape()
    core = _make_core(core_h, core_w, core_colors)
    if not _is_informative(core):
        return None

    current = core.copy()
    for frame_color, op_color in zip(reversed(frame_colors), reversed(op_colors)):
        current = _add_border(_apply_inverse(current, op_color), frame_color)

    obj_h, obj_w = current.shape
    if obj_h > 20 or obj_w > 20:
        return None

    total_h = obj_h + int(_RNG.integers(6, 13))
    total_w = obj_w + int(_RNG.integers(6, 13))
    if total_h < obj_h + 6 or total_w < obj_w + 6:
        return None

    grid = np.full((total_h, total_w), BLACK, dtype=int)
    r0 = (total_h - obj_h) // 2
    c0 = (total_w - obj_w) // 2
    if r0 < 2 or c0 < 2 or total_h - (r0 + obj_h) < 2 or total_w - (c0 + obj_w) < 2:
        return None
    grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = current

    _place_legends(grid, frame_colors, op_colors)

    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
