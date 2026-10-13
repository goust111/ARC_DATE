import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
BROWN = 9

RECT_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, BROWN)
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PATTERN_LIBRARY = [
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=bool),
]

GENERATE_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_dictionary_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = np.argwhere(grid == GREY)
    assert grey.size > 0, "expected a grey texture dictionary"
    r0 = int(grey[:, 0].min())
    c0 = int(grey[:, 1].min())
    r1 = int(grey[:, 0].max())
    c1 = int(grey[:, 1].max())
    return r0, c0, r1, c1


def _parse_textures(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    row_lines = [i for i in range(sub.shape[0]) if np.all(sub[i, :] == GREY)]
    col_lines = [i for i in range(sub.shape[1]) if np.all(sub[:, i] == GREY)]

    assert row_lines[0] == 0 and row_lines[-1] == sub.shape[0] - 1
    assert col_lines[0] == 0 and col_lines[-1] == sub.shape[1] - 1
    assert all(b - a == 4 for a, b in zip(row_lines, row_lines[1:]))
    assert all(b - a == 4 for a, b in zip(col_lines, col_lines[1:]))

    textures: dict[int, np.ndarray] = {}
    for rr in range(len(row_lines) - 1):
        for cc in range(len(col_lines) - 1):
            cell = sub[row_lines[rr] + 1 : row_lines[rr + 1], col_lines[cc] + 1 : col_lines[cc + 1]]
            assert cell.shape == (3, 3)
            colors = [int(v) for v in np.unique(cell) if int(v) not in {BLACK, GREY}]
            assert len(colors) == 1, "each dictionary cell must encode one color"
            color = colors[0]
            assert color not in textures, "dictionary colors must be unique"
            texture = cell == color
            assert texture.any() and (~texture).any(), "textures must be mixed binary patterns"
            textures[color] = texture
    return textures


def _extract_rectangles(grid: np.ndarray, dict_bbox: tuple[int, int, int, int]) -> dict[int, dict]:
    r0, c0, r1, c1 = dict_bbox
    scene = np.array(grid, copy=True)
    scene[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    colors = [int(v) for v in np.unique(scene) if int(v) not in {BLACK, GREY, AZURE}]
    rectangles: dict[int, dict] = {}

    for color in colors:
        comps = _components(scene == color)
        assert len(comps) == 1, "each rectangle color must form one outside-dictionary component"
        comp = comps[0]
        br0, bc0, br1, bc1 = _bbox(comp)
        sub = grid[br0 : br1 + 1, bc0 : bc1 + 1]
        assert np.all(np.isin(sub, [color, AZURE])), "rectangle bbox may only contain its color and one azure window"
        window_pixels = np.argwhere(sub == AZURE)
        assert len(window_pixels) == 1, "each rectangle must have one azure window"
        wr = br0 + int(window_pixels[0, 0])
        wc = bc0 + int(window_pixels[0, 1])
        sides = []
        if wr == br0:
            sides.append("top")
        if wr == br1:
            sides.append("bottom")
        if wc == bc0:
            sides.append("left")
        if wc == bc1:
            sides.append("right")
        assert len(sides) == 1, "window must lie on exactly one rectangle side"
        assert np.count_nonzero(sub == color) == sub.size - 1
        rectangles[color] = {
            "color": color,
            "bbox": (br0, bc0, br1, bc1),
            "window": (wr, wc),
            "side": sides[0],
        }

    return rectangles


def _ray_target_color(grid: np.ndarray, window: tuple[int, int], side: str) -> int:
    h, w = grid.shape
    dr, dc = SIDE_TO_DIR[side]
    r, c = window
    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "ray left the grid before hitting a target"
        value = int(grid[r, c])
        if value in {BLACK, GREY, AZURE}:
            continue
        return value


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each rectangle's azure window to ray-cast toward another rectangle, then tile the looked-up texture inside the source rectangle.

    Input structure:
    - The background is black.
    - Several distinct solid-colored rectangles sit in the scene; each rectangle has exactly one azure pixel replacing one non-corner border pixel.
    - A grey framed-and-partitioned dictionary block contains several 3x3 texture cells, each drawn with one rectangle color on black.

    Rule:
    - Locate the grey dictionary and read every 3x3 cell as a binary texture keyed by its cell color.
    - For each colored rectangle, use the window side to determine an outward ray direction and march through black, azure, and grey cells until the first other colored pixel is hit.
    - Look up the hit color's dictionary texture and tile that 3x3 binary mask across the current rectangle's interior, drawing mask-on cells in the current rectangle color and leaving mask-off cells black; restore the azure window to the rectangle color.

    Key insight:
    - The transformation is a three-step chain: the window chooses a ray direction, the ray's first hit chooses a lookup color, and that lookup texture determines the interior fill of the emitting rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    dict_bbox = _find_dictionary_bbox(g)
    textures = _parse_textures(g, dict_bbox)
    rectangles = _extract_rectangles(g, dict_bbox)
    assert set(rectangles) == set(textures), "dictionary colors and rectangle colors must match"

    out = np.array(g, copy=True)
    for color in sorted(rectangles):
        rect = rectangles[color]
        r0, c0, r1, c1 = rect["bbox"]
        target_color = _ray_target_color(g, rect["window"], rect["side"])
        assert target_color in textures, "ray hit color must exist in the dictionary"
        texture = textures[target_color]

        out[r0 : r1 + 1, c0 : c1 + 1] = color
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                tr = (rr - (r0 + 1)) % 3
                tc = (cc - (c0 + 1)) % 3
                out[rr, cc] = color if texture[tr, tc] else BLACK

    return out


def _sample_dim() -> int:
    return random.choices([4, 5, 6, 7], weights=[3, 3, 2, 1], k=1)[0]


def _rect_bbox(rect: dict) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r0"] + rect["h"] - 1, rect["c0"] + rect["w"] - 1


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    row_sep = ar1 + gap < br0 or br1 + gap < ar0
    col_sep = ac1 + gap < bc0 or bc1 + gap < ac0
    return not (row_sep or col_sep)


def _make_rect(name: str, r0: int, c0: int, h: int, w: int, side: str, window: tuple[int, int], target: str) -> dict:
    return {
        "name": name,
        "r0": int(r0),
        "c0": int(c0),
        "h": int(h),
        "w": int(w),
        "side": side,
        "window": (int(window[0]), int(window[1])),
        "target": target,
    }


def _build_core_layout() -> list[dict] | None:
    for _ in range(200):
        hA, wA = _sample_dim(), _sample_dim()
        hB, wB = _sample_dim(), _sample_dim()
        hC, wC = _sample_dim(), _sample_dim()
        hD, wD = _sample_dim(), _sample_dim()

        row_ab = random.randint(1, hA - 2)
        col_da = random.randint(1, wA - 2)
        gap_ab = random.randint(2, 3)
        gap_bc = random.randint(2, 3)
        gap_cd = random.randint(2, 3)
        gap_da = random.randint(2, 3)

        rA, cA = 0, 0
        cB = wA + gap_ab + random.randint(0, 1)
        rB = row_ab - random.randint(0, hB - 1)
        off_bc_b = random.randint(1, wB - 2)
        col_bc = cB + off_bc_b

        rD = hA + gap_da + random.randint(0, 1)
        cD = col_da - random.randint(1, wD - 2)

        off_bc_c = random.randint(0, wC - 1)
        cC = col_bc - off_bc_c
        if cC < cD + wD + gap_cd:
            continue

        off_cd_c = random.randint(1, hC - 2)
        low_rC = max(rB + hB + gap_bc, rD - off_cd_c)
        high_rC = rD + hD - 1 - off_cd_c
        if low_rC > high_rC:
            continue
        rC = random.randint(low_rC, high_rC)
        row_cd = rC + off_cd_c

        rects = [
            _make_rect("A", rA, cA, hA, wA, "right", (row_ab, cA + wA - 1), "B"),
            _make_rect("B", rB, cB, hB, wB, "bottom", (rB + hB - 1, col_bc), "C"),
            _make_rect("C", rC, cC, hC, wC, "left", (row_cd, cC), "D"),
            _make_rect("D", rD, cD, hD, wD, "top", (rD, col_da), "A"),
        ]

        ok = True
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                if _boxes_conflict(_rect_bbox(rects[i]), _rect_bbox(rects[j]), gap=1):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return rects
    return None


def _build_extra(core_by_name: dict[str, dict], used_rects: list[dict]) -> dict | None:
    slots = ["top", "right", "bottom", "left"]
    random.shuffle(slots)
    for slot in slots:
        for _ in range(80):
            h = _sample_dim()
            w = _sample_dim()
            gap = random.randint(2, 3)
            if slot == "top":
                target_name = random.choice(["A", "B"])
                target = core_by_name[target_name]
                col = random.randint(target["c0"], target["c0"] + target["w"] - 1)
                off = random.randint(1, w - 2)
                c0 = col - off
                r0 = target["r0"] - gap - h
                rect = _make_rect("E", r0, c0, h, w, "bottom", (r0 + h - 1, col), target_name)
            elif slot == "right":
                target_name = random.choice(["B", "C"])
                target = core_by_name[target_name]
                row = random.randint(target["r0"], target["r0"] + target["h"] - 1)
                off = random.randint(1, h - 2)
                r0 = row - off
                c0 = target["c0"] + target["w"] + gap
                rect = _make_rect("E", r0, c0, h, w, "left", (row, c0), target_name)
            elif slot == "bottom":
                target_name = random.choice(["C", "D"])
                target = core_by_name[target_name]
                col = random.randint(target["c0"], target["c0"] + target["w"] - 1)
                off = random.randint(1, w - 2)
                c0 = col - off
                r0 = target["r0"] + target["h"] + gap
                rect = _make_rect("E", r0, c0, h, w, "top", (r0, col), target_name)
            else:
                target_name = random.choice(["A", "D"])
                target = core_by_name[target_name]
                row = random.randint(target["r0"], target["r0"] + target["h"] - 1)
                off = random.randint(1, h - 2)
                r0 = row - off
                c0 = target["c0"] - gap - w
                rect = _make_rect("E", r0, c0, h, w, "right", (row, c0 + w - 1), target_name)

            bbox = _rect_bbox(rect)
            if any(_boxes_conflict(bbox, _rect_bbox(other), gap=1) for other in used_rects):
                continue
            return rect
    return None


def _shift_rects(rects: list[dict], dr: int, dc: int) -> list[dict]:
    shifted = []
    for rect in rects:
        shifted.append(
            {
                **rect,
                "r0": rect["r0"] + dr,
                "c0": rect["c0"] + dc,
                "window": (rect["window"][0] + dr, rect["window"][1] + dc),
            }
        )
    return shifted


def _choose_dict_shape(n_rects: int) -> tuple[int, int]:
    if n_rects == 4:
        return random.choice([(1, 4), (2, 2)])
    assert n_rects == 5
    return (1, 5)


def _draw_dictionary(grid: np.ndarray, colors: list[int], textures: dict[int, np.ndarray], rows: int, cols: int) -> None:
    dict_h = 4 * rows + 1
    dict_w = 4 * cols + 1
    grid[:dict_h, :dict_w] = BLACK
    for rr in range(0, dict_h, 4):
        grid[rr, :dict_w] = GREY
    for cc in range(0, dict_w, 4):
        grid[:dict_h, cc] = GREY

    order = colors[:]
    random.shuffle(order)
    idx = 0
    for rr in range(rows):
        for cc in range(cols):
            color = order[idx]
            idx += 1
            cell_r0 = 1 + 4 * rr
            cell_c0 = 1 + 4 * cc
            cell = grid[cell_r0 : cell_r0 + 3, cell_c0 : cell_c0 + 3]
            cell[:] = BLACK
            cell[textures[color]] = color


def _draw_scene(shape: tuple[int, int], rects: list[dict], colors_by_name: dict[str, int], textures: dict[int, np.ndarray], dict_shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BLACK, dtype=int)
    colors = [colors_by_name[rect["name"]] for rect in rects]
    _draw_dictionary(grid, colors, textures, dict_shape[0], dict_shape[1])

    for rect in rects:
        color = colors_by_name[rect["name"]]
        r0, c0, r1, c1 = _rect_bbox(rect)
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    for rect in rects:
        wr, wc = rect["window"]
        grid[wr, wc] = AZURE
    return grid


def _validate_targets(grid: np.ndarray, rects: list[dict], colors_by_name: dict[str, int]) -> bool:
    for rect in rects:
        hit = _ray_target_color(grid, rect["window"], rect["side"])
        if hit != colors_by_name[rect["target"]]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    core = _build_core_layout()
    if core is None:
        return None

    extra_count = random.choice([0, 1])
    rects = list(core)
    core_by_name = {rect["name"]: rect for rect in core}
    if extra_count == 1:
        extra = _build_extra(core_by_name, rects)
        if extra is None:
            return None
        extra["name"] = "E"
        rects.append(extra)

    n_rects = len(rects)
    colors = random.sample(RECT_COLORS, n_rects)
    colors_by_name = {rect["name"]: color for rect, color in zip(rects, colors)}
    texture_masks = random.sample(PATTERN_LIBRARY, n_rects)
    textures = {color: np.array(mask, copy=True) for color, mask in zip(colors, texture_masks)}

    dict_rows, dict_cols = _choose_dict_shape(n_rects)
    dict_h = 4 * dict_rows + 1
    min_row = min(rect["r0"] for rect in rects)
    min_col = min(rect["c0"] for rect in rects)
    max_row = max(_rect_bbox(rect)[2] for rect in rects)
    max_col = max(_rect_bbox(rect)[3] for rect in rects)

    shift_r = dict_h + random.randint(2, 4) - min_row
    shift_c = random.randint(1, 4) - min_col
    shifted = _shift_rects(rects, shift_r, shift_c)

    scene_bottom = max(_rect_bbox(rect)[2] for rect in shifted)
    scene_right = max(_rect_bbox(rect)[3] for rect in shifted)
    min_width = max(4 * dict_cols + 1, scene_right + 1)
    min_height = scene_bottom + 1
    if min_width > 30 or min_height > 30:
        return None

    height = min_height + random.randint(0, 30 - min_height)
    width = min_width + random.randint(0, 30 - min_width)
    grid = _draw_scene((height, width), shifted, colors_by_name, textures, (dict_rows, dict_cols))

    if not _validate_targets(grid, shifted, colors_by_name):
        return None

    if random.random() < 0.5:
        grid = np.flipud(grid)
    if random.random() < 0.5:
        grid = np.fliplr(grid)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if np.any(out == AZURE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATE_RETRIES} attempts")
