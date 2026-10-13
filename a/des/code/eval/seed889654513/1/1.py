import numpy as np


BG = 0
FRAME_COLOR = 5
LEGEND_WIDTH = 4
TEXTURE_SIZE = 3
GAP_SIDES = ("top", "right", "bottom", "left")
ROT_K = {"top": 0, "right": 3, "bottom": 2, "left": 1}

# Generator config.
GEN_RETRIES = 400
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (22, 32)
REGION_COUNT_RANGE = (2, 4)
INNER_H_RANGE = (3, 8)
INNER_W_RANGE = (3, 8)
LEGEND_ROW_GAPS = (1, 2)
LEGEND_GAP_COLS = 2
REGION_PAD = 1
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _legend_mapping(g: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, w = g.shape
    assert w >= LEGEND_WIDTH, "grid must be wide enough for the legend"
    strip = g[:, w - LEGEND_WIDTH : w]
    full_rows = np.all(strip != bg, axis=1)

    mapping: dict[int, np.ndarray] = {}
    r = 0
    while r < h:
        if not full_rows[r]:
            r += 1
            continue
        r0 = r
        while r < h and full_rows[r]:
            r += 1
        if r - r0 != TEXTURE_SIZE:
            raise AssertionError("legend entries must be 3 rows high")
        block = g[r0:r, w - LEGEND_WIDTH : w]
        key = int(block[0, 0])
        assert key != bg and np.all(block[:, 0] == key), "legend key column must be monochrome"
        texture = block[:, 1:].copy()
        colors = set(map(int, np.unique(block)))
        assert len(colors) == 2 and key in colors, "legend block must use exactly two colors"
        assert key in set(map(int, np.unique(texture))), "texture must include the key color"
        assert key not in mapping, "legend keys must be unique"
        mapping[key] = texture

    assert mapping, "expected at least one legend entry on the right edge"
    return mapping


def _frame_info(g: np.ndarray, comp: list[tuple[int, int]], color: int, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(comp)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    border = _perimeter_cells(r0, c0, r1, c1)
    cells = set(comp)
    if any(cell not in border for cell in cells):
        return None

    missing = list(border - cells)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap[0] in (r0, r1) and gap[1] in (c0, c1):
        return None

    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    non_bg = np.argwhere(interior != bg)
    if len(non_bg) != 1:
        return None

    seed_r, seed_c = map(int, non_bg[0])
    seed_color = int(interior[seed_r, seed_c])
    if seed_color == color:
        return None

    if gap[0] == r0:
        side = "top"
    elif gap[1] == c1:
        side = "right"
    elif gap[0] == r1:
        side = "bottom"
    else:
        side = "left"

    return {
        "bbox": (r0, c0, r1, c1),
        "gap_side": side,
        "seed_color": seed_color,
    }


def _find_regions(g: np.ndarray, bg: int) -> list[dict]:
    best_color = None
    best_frames: list[dict] = []
    best_score = None

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        frames = []
        for comp in _components(g == color):
            info = _frame_info(g, comp, color, bg)
            if info is not None:
                frames.append(info)
        if not frames:
            continue
        area_sum = sum((f["bbox"][2] - f["bbox"][0] + 1) * (f["bbox"][3] - f["bbox"][1] + 1) for f in frames)
        score = (len(frames), area_sum)
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
            best_frames = frames

    assert best_color is not None and best_frames, "expected at least one gapped frame"
    return best_frames


def _tile(texture: np.ndarray, height: int, width: int) -> np.ndarray:
    rr = np.arange(height)[:, None] % TEXTURE_SIZE
    cc = np.arange(width)[None, :] % TEXTURE_SIZE
    return texture[rr, cc]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame using the legend texture keyed by the seed color.

    The seed color selects the texture. The side containing the 1-pixel border gap selects the
    texture rotation: top=0, right=90 clockwise, bottom=180, left=270 clockwise.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(np.bincount(g.ravel()).argmax())
    legend = _legend_mapping(g, bg)
    regions = _find_regions(g, bg)

    out = g.copy()
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        seed_color = region["seed_color"]
        assert seed_color in legend, "every seed color must appear in the legend"
        texture = np.rot90(legend[seed_color], ROT_K[region["gap_side"]])
        out[r0 + 1 : r1, c0 + 1 : c1] = _tile(texture, r1 - r0 - 1, c1 - c0 - 1)
    return out


def _build_texture(mask: np.ndarray, key: int, other: int) -> np.ndarray:
    return np.where(mask, key, other).astype(int)


def _texture_masks() -> list[np.ndarray]:
    masks = []
    for bits in range(1, (1 << 9) - 1):
        flat = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool)
        mask = flat.reshape(TEXTURE_SIZE, TEXTURE_SIZE)
        count = int(mask.sum())
        if count < 3 or count > 6:
            continue
        rotations = {np.rot90(mask, k).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        masks.append(mask)
    return masks


TEXTURE_MASKS = _texture_masks()


def _sample_gap_sides(rng: np.random.Generator, count: int) -> list[str]:
    start = int(rng.integers(0, len(GAP_SIDES)))
    return [GAP_SIDES[(start + i) % len(GAP_SIDES)] for i in range(count)]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _gap_cell(r0: int, c0: int, r1: int, c1: int, side: str, rng: np.random.Generator) -> tuple[int, int]:
    if side == "top":
        return r0, int(rng.integers(c0 + 1, c1))
    if side == "right":
        return int(rng.integers(r0 + 1, r1)), c1
    if side == "bottom":
        return r1, int(rng.integers(c0 + 1, c1))
    return int(rng.integers(r0 + 1, r1)), c0


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], side: str, seed_color: int, rng: np.random.Generator) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    gr, gc = _gap_cell(r0, c0, r1, c1, side, rng)
    grid[gr, gc] = BG

    sr = int(rng.integers(r0 + 1, r1))
    sc = int(rng.integers(c0 + 1, c1))
    grid[sr, sc] = seed_color


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    region_count = int(rng.integers(REGION_COUNT_RANGE[0], REGION_COUNT_RANGE[1] + 1))
    legend_gap = int(rng.choice(LEGEND_ROW_GAPS))
    legend_height = region_count * TEXTURE_SIZE + (region_count - 1) * legend_gap

    min_h = max(GRID_H_RANGE[0], legend_height + 1)
    if min_h > GRID_H_RANGE[1]:
        return None
    h = int(rng.integers(min_h, GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    if w <= LEGEND_WIDTH + LEGEND_GAP_COLS + 5:
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = list(PALETTE)
    rng.shuffle(colors)
    seed_colors = colors[:region_count]
    other_colors = colors[region_count : 2 * region_count]

    mask_ids = rng.choice(len(TEXTURE_MASKS), size=region_count, replace=False)
    textures = [
        _build_texture(TEXTURE_MASKS[int(mask_id)], seed_colors[i], other_colors[i])
        for i, mask_id in enumerate(mask_ids)
    ]

    legend_order = list(range(region_count))
    rng.shuffle(legend_order)
    legend_r0 = int(rng.integers(0, h - legend_height + 1))
    legend_c0 = w - LEGEND_WIDTH
    row = legend_r0
    for idx_i, idx in enumerate(legend_order):
        grid[row : row + TEXTURE_SIZE, legend_c0] = seed_colors[idx]
        grid[row : row + TEXTURE_SIZE, legend_c0 + 1 : w] = textures[idx]
        row += TEXTURE_SIZE
        if idx_i + 1 < region_count:
            row += legend_gap

    sides = _sample_gap_sides(rng, region_count)
    specs = []
    for i in range(region_count):
        inner_h = int(rng.integers(INNER_H_RANGE[0], INNER_H_RANGE[1] + 1))
        inner_w = int(rng.integers(INNER_W_RANGE[0], INNER_W_RANGE[1] + 1))
        specs.append(
            {
                "seed_color": seed_colors[i],
                "gap_side": sides[i],
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
            }
        )
    specs.sort(key=lambda spec: spec["outer_h"] * spec["outer_w"], reverse=True)

    max_right = legend_c0 - LEGEND_GAP_COLS - 1
    if max_right < 6:
        return None

    padded_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        max_r0 = h - oh - 1
        max_c0 = max_right - ow
        if max_r0 < 1 or max_c0 < 1:
            return None

        placed = False
        for _ in range(250):
            r0 = int(rng.integers(1, max_r0 + 1))
            c0 = int(rng.integers(1, max_c0 + 1))
            box = (r0, c0, r0 + oh - 1, c0 + ow - 1)
            padded = (r0 - REGION_PAD, c0 - REGION_PAD, r0 + oh - 1 + REGION_PAD, c0 + ow - 1 + REGION_PAD)
            if any(_boxes_overlap(padded, prev) for prev in padded_boxes):
                continue
            spec["box"] = box
            padded_boxes.append(padded)
            placed = True
            break
        if not placed:
            return None

    for spec in specs:
        _draw_frame(grid, spec["box"], spec["gap_side"], spec["seed_color"], rng)

    try:
        legend = _legend_mapping(grid, BG)
        regions = _find_regions(grid, BG)
        out = main(grid)
    except Exception:
        return None

    if len(legend) != region_count or len(regions) != region_count:
        return None
    if set(legend) != set(seed_colors):
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC sample")
