import random

import numpy as np


BG = 0
LINE_SYMBOL = 4
GREY = 5
T_SYMBOL = 7
L_SYMBOL = 9

SHAPE_TO_SYMBOL = {"L": L_SYMBOL, "T": T_SYMBOL, "line": LINE_SYMBOL}
SYMBOL_TO_SHAPE = {value: key for key, value in SHAPE_TO_SYMBOL.items()}

COLOR_POOL = (1, 2, 3, 6, 8)
SHAPES = ("L", "T", "line")
SIDES = ("top", "bottom", "left", "right")

MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = (variant.shape, variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(variant)
    return variants


def _is_line_mask(mask: np.ndarray) -> bool:
    return bool(mask.all() and (mask.shape[0] == 1 or mask.shape[1] == 1))


def _is_canonical_l(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 2:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[:, 0] = True
    return np.array_equal(mask, expected)


def _is_canonical_t(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 2 or w < 3 or w % 2 == 0:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[:, w // 2] = True
    return np.array_equal(mask, expected)


def _classify_seed(coords: np.ndarray) -> str:
    mask = _coords_to_mask(coords)
    if _is_line_mask(mask):
        return "line"
    for variant in _mask_transforms(mask):
        if _is_canonical_l(variant):
            return "L"
        if _is_canonical_t(variant):
            return "T"
    raise AssertionError("seed must be an L, T, or straight segment")


def _parse_legend_region(region: np.ndarray, axis: str) -> dict[str, int] | None:
    mapping: dict[str, int] = {}
    for i in range(3):
        tile = region[2 * i : 2 * i + 2, :] if axis == "vertical" else region[:, 2 * i : 2 * i + 2]
        if tile.shape != (2, 2) or np.any(tile == BG):
            return None
        if not np.all(tile[:, 0] == tile[0, 0]):
            return None
        if not np.all(tile[:, 1] == tile[0, 1]):
            return None
        symbol = int(tile[0, 0])
        fill = int(tile[0, 1])
        if symbol not in SYMBOL_TO_SHAPE or fill == GREY:
            return None
        shape = SYMBOL_TO_SHAPE[symbol]
        if shape in mapping:
            return None
        mapping[shape] = fill
    if set(mapping) != set(SHAPES):
        return None
    return mapping


def _find_legend_mapping(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    candidates: list[dict[str, int]] = []

    if w >= 2 and h >= 6:
        for r0 in range(0, h - 5):
            region = grid[r0 : r0 + 6, w - 2 : w]
            if np.any(grid[:r0, w - 2 : w] != BG) or np.any(grid[r0 + 6 :, w - 2 : w] != BG):
                continue
            mapping = _parse_legend_region(region, axis="vertical")
            if mapping is not None:
                candidates.append(mapping)

    if h >= 2 and w >= 6:
        for c0 in range(0, w - 5):
            region = grid[h - 2 : h, c0 : c0 + 6]
            if np.any(grid[h - 2 : h, :c0] != BG) or np.any(grid[h - 2 : h, c0 + 6 :] != BG):
                continue
            mapping = _parse_legend_region(region, axis="horizontal")
            if mapping is not None:
                candidates.append(mapping)

    assert len(candidates) == 1, "expected exactly one edge legend"
    return candidates[0]


def _frame_info(coords: list[tuple[int, int]]) -> tuple[tuple[int, int, int, int], tuple[int, int]] | None:
    pts = np.asarray(coords, dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    coord_set = {(int(r), int(c)) for r, c in pts}
    border_set: set[tuple[int, int]] = set()
    for r in range(r0, r1 + 1):
        border_set.add((r, c0))
        border_set.add((r, c1))
    for c in range(c0, c1 + 1):
        border_set.add((r0, c))
        border_set.add((r1, c))

    if not coord_set <= border_set:
        return None
    missing = list(border_set - coord_set)
    if len(missing) != 1:
        return None
    gap = missing[0]
    if gap in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None
    if len(coord_set) != len(border_set) - 1:
        return None
    return (r0, c0, r1, c1), gap


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color in (BG, GREY):
            continue
        for comp in _components(grid == color):
            info = _frame_info(comp)
            if info is None:
                continue
            bbox, gap = info
            frames.append({"color": color, "bbox": bbox, "gap": gap})
    assert frames, "expected at least one gapped rectangular frame"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _seed_coords_in_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert np.all((interior == BG) | (interior == GREY)), "frame interior must contain only bg and grey seed"
    comps = _components(interior == GREY)
    assert len(comps) == 1, "each frame must contain exactly one grey seed"
    coords = np.asarray([(r0 + 1 + r, c0 + 1 + c) for r, c in comps[0]], dtype=int)
    return coords


def _growth_direction(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    gr, gc = gap
    if gr == r0:
        return 1, 0
    if gr == r1:
        return -1, 0
    if gc == c0:
        return 0, 1
    if gc == c1:
        return 0, -1
    raise AssertionError("gap must lie on the frame border")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each grey seed across its framed region using the seed shape's legend-mapped fill color.

    Input structure:
    - The background is black. Several monochrome rectangular one-pixel-thick frames appear on the grid, and each frame has exactly one one-cell gap on one side.
    - Every frame encloses one connected grey seed whose geometry is an L, a T, or a straight line segment.
    - A small legend sits on the right edge or bottom edge as three 2x2 bars whose left column uses fixed shape-marker colors and whose right column gives the output color for that shape.

    Rule:
    - Read the edge legend to build a mapping from seed shape category to fill color.
    - For each gapped rectangular frame, the gap side determines growth direction: the seed extends away from the gap toward the opposite inner wall.
    - Replace the grey seed with its mapped fill color and continue painting from every seed cell step by step along the growth direction until the frame's opposite inner wall is reached. Keep the frame, background, and gap unchanged.

    Key insight:
    - The direction comes from the frame gap, but the fill color comes from geometric seed-shape classification via the separate edge legend, so solving requires combining two different cues.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    mapping = _find_legend_mapping(g)
    frames = _find_frames(g)
    out = g.copy()

    for frame in frames:
        bbox = frame["bbox"]
        gap = frame["gap"]
        r0, c0, r1, c1 = bbox
        seed_coords = _seed_coords_in_frame(g, bbox)
        shape = _classify_seed(seed_coords)
        fill_color = mapping[shape]
        dr, dc = _growth_direction(bbox, gap)

        for r0_seed, c0_seed in seed_coords:
            rr = int(r0_seed)
            cc = int(c0_seed)
            while r0 < rr < r1 and c0 < cc < c1:
                out[rr, cc] = fill_color
                nr = rr + dr
                nc = cc + dc
                if not (r0 < nr < r1 and c0 < nc < c1):
                    break
                rr, cc = nr, nc

    return out


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    variant = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        variant = np.fliplr(variant)
    return variant.copy()


def _make_seed_mask(shape: str, rng: random.Random) -> np.ndarray:
    if shape == "line":
        length = rng.randint(2, 4)
        mask = np.ones((1, length), dtype=bool)
        if rng.random() < 0.5:
            mask = mask.T
        return mask

    if shape == "L":
        arm_h = rng.randint(2, 4)
        arm_w = rng.randint(2, 4)
        mask = np.zeros((arm_h, arm_w), dtype=bool)
        mask[0, :] = True
        mask[:, 0] = True
        return _random_transform(mask, rng)

    assert shape == "T"
    stem_h = rng.randint(2, 4)
    bar_w = rng.choice((3, 5))
    mask = np.zeros((stem_h, bar_w), dtype=bool)
    mask[0, :] = True
    mask[:, bar_w // 2] = True
    return _random_transform(mask, rng)


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int, side: str, gap_index: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color

    if side == "top":
        grid[top, left + gap_index] = BG
    elif side == "bottom":
        grid[bottom, left + gap_index] = BG
    elif side == "left":
        grid[top + gap_index, left] = BG
    else:
        grid[top + gap_index, right] = BG


def _place_seed(grid: np.ndarray, top: int, left: int, seed_mask: np.ndarray) -> None:
    sh, sw = seed_mask.shape
    grid[top : top + sh, left : left + sw][seed_mask] = GREY


def _legend_mapping_colors(rng: random.Random) -> dict[str, int]:
    fills = rng.sample(COLOR_POOL, len(SHAPES))
    return dict(zip(rng.sample(list(SHAPES), len(SHAPES)), fills))


def _usable_limits(height: int, width: int, legend_orientation: str) -> tuple[int, int]:
    max_bottom = height - 2
    max_right = width - 2
    if legend_orientation == "bottom":
        max_bottom = height - 4
    else:
        max_right = width - 4
    return max_bottom, max_right


def _spec_for_frame(shape: str, side: str, rng: random.Random) -> dict[str, object]:
    seed_mask = _make_seed_mask(shape, rng)
    sh, sw = seed_mask.shape

    top_pad = rng.randint(0, 2)
    bottom_pad = rng.randint(0, 2)
    left_pad = rng.randint(0, 2)
    right_pad = rng.randint(0, 2)

    if side == "top":
        bottom_pad = rng.randint(1, 3)
    elif side == "bottom":
        top_pad = rng.randint(1, 3)
    elif side == "left":
        right_pad = rng.randint(1, 3)
    else:
        left_pad = rng.randint(1, 3)

    interior_h = top_pad + sh + bottom_pad
    interior_w = left_pad + sw + right_pad

    return {
        "shape": shape,
        "side": side,
        "seed_mask": seed_mask,
        "seed_offset": (top_pad, left_pad),
        "outer_h": interior_h + 2,
        "outer_w": interior_w + 2,
        "interior_h": interior_h,
        "interior_w": interior_w,
    }


def _place_frames(
    height: int,
    width: int,
    legend_orientation: str,
    specs: list[dict[str, object]],
    border_colors: list[int],
    rng: random.Random,
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    occupied = np.zeros_like(grid, dtype=bool)
    max_bottom, max_right = _usable_limits(height, width, legend_orientation)

    for spec, color in zip(specs, border_colors):
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        if outer_h > max_bottom or outer_w > max_right:
            return None

        placed = False
        max_top = max_bottom - outer_h + 1
        max_left = max_right - outer_w + 1
        if max_top < 1 or max_left < 1:
            return None

        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            bottom = top + outer_h - 1
            right = left + outer_w - 1
            patch = occupied[top - 1 : bottom + 2, left - 1 : right + 2]
            if np.any(patch):
                continue

            if spec["side"] in ("top", "bottom"):
                gap_index = rng.randint(1, outer_w - 2)
            else:
                gap_index = rng.randint(1, outer_h - 2)

            _draw_frame(grid, top, left, outer_h, outer_w, color, spec["side"], gap_index)
            seed_top = top + 1 + int(spec["seed_offset"][0])
            seed_left = left + 1 + int(spec["seed_offset"][1])
            _place_seed(grid, seed_top, seed_left, spec["seed_mask"])
            occupied[top - 1 : bottom + 2, left - 1 : right + 2] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _draw_legend(grid: np.ndarray, legend_orientation: str, mapping: dict[str, int], rng: random.Random) -> None:
    h, w = grid.shape
    order = rng.sample(list(SHAPES), len(SHAPES))
    if legend_orientation == "right":
        start_r = rng.randint(1, h - 7)
        for i, shape in enumerate(order):
            tile_r = start_r + 2 * i
            grid[tile_r : tile_r + 2, w - 2] = SHAPE_TO_SYMBOL[shape]
            grid[tile_r : tile_r + 2, w - 1] = mapping[shape]
    else:
        start_c = rng.randint(1, w - 7)
        for i, shape in enumerate(order):
            tile_c = start_c + 2 * i
            grid[h - 2 : h, tile_c] = SHAPE_TO_SYMBOL[shape]
            grid[h - 2 : h, tile_c + 1] = mapping[shape]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_orientation = rng.choice(("right", "bottom"))
    n_frames = rng.randint(1, 3)
    height = rng.randint(18, 30)
    width = rng.randint(18, 30)

    mapping = _legend_mapping_colors(rng)
    border_colors = rng.sample(COLOR_POOL, n_frames)
    specs = [_spec_for_frame(rng.choice(SHAPES), rng.choice(SIDES), rng) for _ in range(n_frames)]

    grid = _place_frames(height, width, legend_orientation, specs, border_colors, rng)
    if grid is None:
        return None

    _draw_legend(grid, legend_orientation, mapping, rng)

    out = main(grid)
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
