import numpy as np


_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")

# np.rot90 counts counter-clockwise quarter turns.  The visible cue is the
# missing corner: each corner asks for one fixed orientation of the piece.
_CORNER_ROTATIONS = {
    "top_left": 3,
    "top_right": 2,
    "bottom_left": 0,
    "bottom_right": 1,
}

_RNG = np.random.default_rng()


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for sr, sc in np.argwhere(grid == color):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in _DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    if int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

        arr = np.array(coords, dtype=int)
        r0, c0 = arr.min(axis=0)
        r1, c1 = arr.max(axis=0)
        comps.append(
            {
                "coords": arr,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(arr.shape[0]),
            }
        )

    return comps


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _classify_l_rect(mask: np.ndarray):
    if mask.ndim != 2 or min(mask.shape) < 2 or np.all(mask):
        return None

    h, w = mask.shape
    missing = ~mask
    if not np.any(missing):
        return None

    rows, cols = np.where(missing)
    specs = []

    if missing[0, 0]:
        mh, mw = int(rows.max()) + 1, int(cols.max()) + 1
        expected = np.zeros_like(missing)
        expected[:mh, :mw] = True
        specs.append(("top_left", mh, mw, expected))

    if missing[0, w - 1]:
        mh, mw = int(rows.max()) + 1, w - int(cols.min())
        expected = np.zeros_like(missing)
        expected[:mh, w - mw :] = True
        specs.append(("top_right", mh, mw, expected))

    if missing[h - 1, 0]:
        mh, mw = h - int(rows.min()), int(cols.max()) + 1
        expected = np.zeros_like(missing)
        expected[h - mh :, :mw] = True
        specs.append(("bottom_left", mh, mw, expected))

    if missing[h - 1, w - 1]:
        mh, mw = h - int(rows.min()), w - int(cols.min())
        expected = np.zeros_like(missing)
        expected[h - mh :, w - mw :] = True
        specs.append(("bottom_right", mh, mw, expected))

    matches = [
        (corner, mh, mw)
        for corner, mh, mw, expected in specs
        if 0 < mh < h and 0 < mw < w and np.array_equal(missing, expected)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _hole_bbox(l_bbox: tuple[int, int, int, int], corner: str, mh: int, mw: int):
    r0, c0, r1, c1 = l_bbox
    if corner == "top_left":
        return r0, c0, r0 + mh - 1, c0 + mw - 1
    if corner == "top_right":
        return r0, c1 - mw + 1, r0 + mh - 1, c1
    if corner == "bottom_left":
        return r1 - mh + 1, c0, r1, c0 + mw - 1
    return r1 - mh + 1, c1 - mw + 1, r1, c1


def _rotated_piece_mask(piece_component: dict, corner: str) -> np.ndarray:
    return np.rot90(_component_mask(piece_component), _CORNER_ROTATIONS[corner])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-colored corner scraps into their matching incomplete rectangles.

    Input structure:
    - The background is the dominant color.
    - Each foreground color appears as one L-shaped rectangle and one detached
      same-colored scrap that does not touch the rectangle.
    - The L-shaped rectangle is a solid rectangle with exactly one rectangular
      corner region missing.
    - The detached scrap has the same number of cells as that missing corner
      region, possibly with its rectangle dimensions turned sideways.

    Rule:
    - For each color, find the L-shaped component and identify which corner is
      missing: top-left, top-right, bottom-left, or bottom-right.
    - Rotate the detached same-colored scrap by the fixed amount indicated by
      that missing corner.
    - Move the rotated scrap into the missing corner region, restoring the
      complete solid rectangle.
    - Clear the scrap's original location back to the background color.

    Key insight:
    The color pairs a scrap with its rectangle, while the missing corner gives
    the scrap's unique rotation before it is moved into the rectangular gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    out = g.copy()

    for color in [int(v) for v in np.unique(g) if int(v) != bg]:
        comps = _components_of_color(g, color)
        if len(comps) < 2:
            continue

        l_infos = []
        for i, comp in enumerate(comps):
            info = _classify_l_rect(_component_mask(comp))
            if info is not None:
                l_infos.append((i, comp, info))

        assert len(l_infos) == 1, "expected one incomplete rectangle per color"
        l_index, l_comp, (corner, mh, mw) = l_infos[0]

        piece_infos = []
        for i, comp in enumerate(comps):
            if i == l_index or int(comp["area"]) != mh * mw:
                continue
            rotated = _rotated_piece_mask(comp, corner)
            if rotated.shape == (mh, mw) and int(rotated.sum()) == mh * mw:
                piece_infos.append((comp, rotated))

        assert len(piece_infos) == 1, "expected one matching scrap per color"
        piece_comp, rotated_piece = piece_infos[0]

        hr0, hc0, hr1, hc1 = _hole_bbox(l_comp["bbox"], corner, mh, mw)
        assert np.all(g[hr0 : hr1 + 1, hc0 : hc1 + 1] == bg)

        coords = piece_comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg
        out[hr0 : hr1 + 1, hc0 : hc1 + 1][rotated_piece] = color

    return out


def _l_mask(height: int, width: int, corner: str, mh: int, mw: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    if corner == "top_left":
        mask[:mh, :mw] = False
    elif corner == "top_right":
        mask[:mh, width - mw :] = False
    elif corner == "bottom_left":
        mask[height - mh :, :mw] = False
    else:
        mask[height - mh :, width - mw :] = False
    return mask


def _piece_shape_for_corner(corner: str, mh: int, mw: int) -> tuple[int, int]:
    if _CORNER_ROTATIONS[corner] % 2:
        return mw, mh
    return mh, mw


def _pair_patch(corner: str, rect_h: int, rect_w: int, mh: int, mw: int) -> np.ndarray:
    gap = 1
    piece_h, piece_w = _piece_shape_for_corner(corner, mh, mw)

    if corner == "top_left":
        patch_h, patch_w = piece_h + gap + rect_h, piece_w + gap + rect_w
        piece_r, piece_c = 0, 0
        rect_r, rect_c = piece_h + gap, piece_w + gap
    elif corner == "top_right":
        patch_h, patch_w = piece_h + gap + rect_h, rect_w + gap + piece_w
        piece_r, piece_c = 0, rect_w + gap
        rect_r, rect_c = piece_h + gap, 0
    elif corner == "bottom_left":
        patch_h, patch_w = rect_h + gap + piece_h, piece_w + gap + rect_w
        piece_r, piece_c = rect_h + gap, 0
        rect_r, rect_c = 0, piece_w + gap
    else:
        patch_h, patch_w = rect_h + gap + piece_h, rect_w + gap + piece_w
        piece_r, piece_c = rect_h + gap, rect_w + gap
        rect_r, rect_c = 0, 0

    mask = np.zeros((patch_h, patch_w), dtype=bool)
    rect = _l_mask(rect_h, rect_w, corner, mh, mw)
    mask[rect_r : rect_r + rect_h, rect_c : rect_c + rect_w] = rect
    mask[piece_r : piece_r + piece_h, piece_c : piece_c + piece_w] = True
    return mask


def _random_cut(rect_h: int, rect_w: int) -> tuple[int, int]:
    h_max = min(4, rect_h - 1)
    w_max = min(4, rect_w - 1)
    pairs = [(h, w) for h in range(1, h_max + 1) for w in range(1, w_max + 1)]
    nonsquare = [(h, w) for h, w in pairs if h != w]
    if nonsquare and float(_RNG.random()) < 0.8:
        pairs = nonsquare
    return tuple(map(int, pairs[int(_RNG.integers(0, len(pairs)))]))


def _place_patch(
    grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray, color: int
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(150):
        r0 = int(_RNG.integers(1, h - ph))
        c0 = int(_RNG.integers(1, w - pw))
        rr0, rr1 = r0 - 1, r0 + ph + 1
        cc0, cc1 = c0 - 1, c0 + pw + 1
        if np.any(reserved[rr0:rr1, cc0:cc1]):
            continue

        coords = np.argwhere(patch)
        grid[coords[:, 0] + r0, coords[:, 1] + c0] = color
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True
        return True

    return False


def _gen_attempt():
    count = int(_RNG.choice(np.array([2, 3, 4]), p=np.array([0.45, 0.4, 0.15])))
    min_side = {2: 17, 3: 22, 4: 27}[count]
    height = int(_RNG.integers(min_side, 31))
    width = int(_RNG.integers(min_side, 31))

    bg = int(_RNG.integers(0, 10))
    colors = [c for c in range(10) if c != bg]
    colors = [int(c) for c in _RNG.choice(colors, size=count, replace=False)]

    corners = list(_CORNERS)
    _RNG.shuffle(corners)

    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for color, corner in zip(colors, corners[:count]):
        rect_h = int(_RNG.integers(4, 9))
        rect_w = int(_RNG.integers(4, 9))
        mh, mw = _random_cut(rect_h, rect_w)
        patch = _pair_patch(corner, rect_h, rect_w, mh, mw)
        if not _place_patch(grid, reserved, patch, color):
            return None

    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-scrap input")
