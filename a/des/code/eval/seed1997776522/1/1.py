from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
GREY = 5
MAROON = 9
SCENE_COLORS = (0, 1, 2, 3, 4, 6, 7)
STYLES = ("square", "diamond", "cross")
GEN_RETRIES = 400

STYLE_MASKS = {
    "square": np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    "diamond": np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    "cross": np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    ),
}

DIGIT_MASKS = {
    1: np.array([[1], [1], [1]], dtype=bool),
    2: np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    3: np.array(
        [
            [1, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    ),
    4: np.array(
        [
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
}

STYLE_KEYS = {
    tuple(tuple(int(v) for v in row) for row in mask.astype(int)): style
    for style, mask in STYLE_MASKS.items()
}
DIGIT_KEYS = {
    tuple(tuple(int(v) for v in row) for row in mask.astype(int)): digit
    for digit, mask in DIGIT_MASKS.items()
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    trimmed = _trim_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in trimmed.astype(int))


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((int(start), int(prev)))
        start = prev = idx
    runs.append((int(start), int(prev)))
    return runs


def _find_components(mask: np.ndarray) -> list[dict[str, np.ndarray | tuple[int, int, int, int] | int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, np.ndarray | tuple[int, int, int, int] | int]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            coord_arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "coords": coord_arr,
                    "bbox": _bbox_from_coords(coord_arr),
                    "area": int(coord_arr.shape[0]),
                }
            )
    return comps


def _enclosed_area(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))

    return (~blocked) & (~seen)


def _classify_style(mask: np.ndarray) -> str:
    key = _mask_key(mask)
    assert key in STYLE_KEYS
    return STYLE_KEYS[key]


def _read_digit(mask: np.ndarray) -> int:
    key = _mask_key(mask)
    assert key in DIGIT_KEYS
    return int(DIGIT_KEYS[key])


def _parse_template_strip(grid: np.ndarray, bg: int) -> tuple[dict[int, str], int]:
    grey_coords = np.argwhere(grid == GREY)
    assert grey_coords.size > 0
    strip_top = int(grey_coords[:, 0].min())
    strip_left = int(grey_coords[:, 1].min())
    strip_right = int(grey_coords[:, 1].max())
    strip = grid[strip_top:, strip_left : strip_right + 1]

    sep_cols = [c for c in range(strip.shape[1]) if np.all(strip[:, c] == GREY)]
    assert len(sep_cols) >= 5

    mapping: dict[int, str] = {}
    for left, right in zip(sep_cols, sep_cols[1:], strict=False):
        if right - left <= 1:
            continue
        cell = strip[:, left + 1 : right]
        rows = [r for r in range(cell.shape[0]) if np.any(cell[r] != bg)]
        runs = _contiguous_runs(rows)
        if len(runs) < 2:
            continue
        top0, top1 = runs[0]
        bot0, bot1 = runs[-1]
        top_part = cell[top0 : top1 + 1]
        bot_part = cell[bot0 : bot1 + 1]
        style_mask = (top_part != bg) & (top_part != GREY) & (top_part != MAROON)
        digit_mask = bot_part == MAROON
        if not style_mask.any() or not digit_mask.any():
            continue
        style = _classify_style(style_mask)
        digit = _read_digit(digit_mask)
        mapping[digit] = style

    assert set(mapping) == {1, 2, 3, 4}
    return mapping, strip_top


def _region_specs(scene: np.ndarray, bg: int) -> list[dict[str, int | tuple[int, int, int, int]]]:
    specs: list[dict[str, int | tuple[int, int, int, int]]] = []
    for comp in _find_components(scene != bg):
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        crop = scene[r0 : r1 + 1, c0 : c1 + 1]
        blocked = np.zeros(crop.shape, dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        blocked[rel[:, 0], rel[:, 1]] = True
        interior = _enclosed_area(blocked)
        if not interior.any():
            continue

        inside_colors = [int(v) for v in np.unique(crop[interior]) if int(v) != bg]
        if not inside_colors:
            continue
        assert 1 <= len(inside_colors) <= 4

        region_fg = (crop != bg) & (blocked | interior)
        vals, cnts = np.unique(crop[region_fg], return_counts=True)
        fg_pairs = [(int(v), int(n)) for v, n in zip(vals, cnts, strict=False) if int(v) != bg]
        assert fg_pairs
        top_count = max(n for _v, n in fg_pairs)
        top_colors = [v for v, n in fg_pairs if n == top_count]
        assert len(top_colors) == 1

        specs.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inside_count": int(len(inside_colors)),
                "border_color": int(top_colors[0]),
            }
        )

    specs.sort(key=lambda item: tuple(item["bbox"]))
    return specs


def _paint(out: np.ndarray, r: int, c: int, color: int) -> None:
    if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
        out[r, c] = int(color)


def _draw_shell(out: np.ndarray, bbox: tuple[int, int, int, int], style: str, color: int) -> None:
    r0, c0, r1, c1 = bbox
    top, bottom = r0 - 1, r1 + 1
    left, right = c0 - 1, c1 + 1

    if style == "square":
        for c in range(left, right + 1):
            _paint(out, top, c, color)
            _paint(out, bottom, c, color)
        for r in range(top, bottom + 1):
            _paint(out, r, left, color)
            _paint(out, r, right, color)
        return

    for c in range(c0, c1 + 1):
        _paint(out, top, c, color)
        _paint(out, bottom, c, color)
    for r in range(r0, r1 + 1):
        _paint(out, r, left, color)
        _paint(out, r, right, color)

    if style == "cross":
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2
        _paint(out, r0 - 2, mid_c, color)
        _paint(out, r1 + 2, mid_c, color)
        _paint(out, mid_r, c0 - 2, color)
        _paint(out, mid_r, c1 + 2, color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, strip_top = _parse_template_strip(g, bg)
    scene = g[:strip_top].copy()

    out = scene.copy()
    for spec in _region_specs(scene, bg):
        style = mapping[int(spec["inside_count"])]
        _draw_shell(out, spec["bbox"], style, int(spec["border_color"]))
    return out


def _sample_style_map() -> dict[int, str]:
    digits = [1, 2, 3, 4]
    style_map: dict[int, str] = {}
    cover_digits = random.sample(digits, 3)
    cover_styles = random.sample(list(STYLES), 3)
    for digit, style in zip(cover_digits, cover_styles, strict=False):
        style_map[int(digit)] = style
    remaining_digit = next(d for d in digits if d not in style_map)
    style_map[int(remaining_digit)] = random.choice(STYLES)
    return style_map


def _build_template_strip(style_map: dict[int, str]) -> np.ndarray:
    top_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 1)
    top_band_h = 5
    gap_h = 1
    digit_band_h = 3
    strip_h = top_pad + top_band_h + gap_h + digit_band_h + bottom_pad

    cell_digits = random.sample([1, 2, 3, 4], 4)
    cell_widths = [random.randint(7, 10) for _ in range(4)]
    strip_w = 5 + sum(cell_widths)
    strip = np.full((strip_h, strip_w), BG, dtype=int)

    sep_cols = [0]
    pos = 1
    for width in cell_widths:
        pos += width
        sep_cols.append(pos)
        pos += 1
    for c in sep_cols:
        strip[:, c] = GREY

    top_row = top_pad
    digit_row = top_pad + top_band_h + gap_h
    ring_colors = random.sample(SCENE_COLORS, 4)

    start = 1
    for digit, width, ring_color in zip(cell_digits, cell_widths, ring_colors, strict=False):
        style = style_map[int(digit)]
        ring = STYLE_MASKS[style]
        digit_mask = DIGIT_MASKS[int(digit)]

        ring_r = top_row + (top_band_h - ring.shape[0]) // 2
        ring_c = start + (width - ring.shape[1]) // 2
        strip[ring_r : ring_r + ring.shape[0], ring_c : ring_c + ring.shape[1]][ring] = int(ring_color)

        digit_c = start + (width - digit_mask.shape[1]) // 2
        strip[digit_row : digit_row + digit_mask.shape[0], digit_c : digit_c + digit_mask.shape[1]][digit_mask] = MAROON
        start += width + 1

    return strip


def _assign_boundary_colors(count: int, dominant: int, accents: list[int]) -> list[int]:
    colors = [dominant] * count
    if not accents:
        return colors
    cap = min(count // 3, 1 + count // 5)
    accent_total = random.randint(1, max(1, cap))
    slots = random.sample(range(count), accent_total)
    for idx in slots:
        colors[idx] = int(random.choice(accents))
    random.shuffle(colors)
    return colors


def _place_blob(
    patch: np.ndarray,
    occupied: np.ndarray,
    area_box: tuple[int, int, int, int],
    color: int,
) -> bool:
    r0, c0, r1, c1 = area_box
    candidates = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    random.shuffle(candidates)
    for sr, sc in candidates:
        if occupied[sr, sc]:
            continue
        blocked = False
        for nr, nc in _neighbors4(sr, sc):
            if 0 <= nr < occupied.shape[0] and 0 <= nc < occupied.shape[1] and occupied[nr, nc]:
                blocked = True
                break
        if blocked:
            continue
        patch[sr, sc] = int(color)
        occupied[sr, sc] = True
        return True
    return False


def _make_region(inner_count: int) -> np.ndarray | None:
    outer_h = random.randint(7, 10)
    outer_w = random.randint(7, 10)
    cells: set[tuple[int, int]] = set()

    for c in range(outer_w):
        cells.add((0, c))
        cells.add((outer_h - 1, c))
    for r in range(outer_h):
        cells.add((r, 0))
        cells.add((r, outer_w - 1))

    tab_sides = random.sample(["top", "bottom", "left", "right"], random.randint(1, 3))
    for side in tab_sides:
        depth = random.randint(1, 2)
        if side in {"top", "bottom"}:
            width = random.randint(2, min(4, outer_w - 2))
            start = random.randint(1, outer_w - width - 1)
            if side == "top":
                rows = range(-depth, 1)
            else:
                rows = range(outer_h - 1, outer_h + depth)
            for r in rows:
                for c in range(start, start + width):
                    cells.add((int(r), int(c)))
        else:
            height = random.randint(2, min(4, outer_h - 2))
            start = random.randint(1, outer_h - height - 1)
            if side == "left":
                cols = range(-depth, 1)
            else:
                cols = range(outer_w - 1, outer_w + depth)
            for r in range(start, start + height):
                for c in cols:
                    cells.add((int(r), int(c)))

    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    max_r = max(r for r, _c in norm_cells)
    max_c = max(c for _r, c in norm_cells)
    patch = np.full((max_r + 1, max_c + 1), BG, dtype=int)

    dominant = random.choice(SCENE_COLORS)
    accent_pool = [c for c in SCENE_COLORS if c != dominant]
    accent_count = random.randint(1, 2)
    accents = random.sample(accent_pool, accent_count)
    boundary_coords = list(norm_cells)
    boundary_colors = _assign_boundary_colors(len(boundary_coords), dominant, accents)
    for (r, c), color in zip(boundary_coords, boundary_colors, strict=False):
        patch[r, c] = int(color)

    inner_r0 = 1 - min_r
    inner_c0 = 1 - min_c
    inner_r1 = inner_r0 + outer_h - 3
    inner_c1 = inner_c0 + outer_w - 3
    safe_box = (inner_r0 + 1, inner_c0 + 1, inner_r1 - 1, inner_c1 - 1)
    if safe_box[0] > safe_box[2] or safe_box[1] > safe_box[3]:
        return None

    include_dom = inner_count > 1 and random.random() < 0.4
    inner_colors = [dominant] if include_dom else []
    pool = [c for c in SCENE_COLORS if c not in inner_colors]
    inner_colors.extend(random.sample(pool, inner_count - len(inner_colors)))
    random.shuffle(inner_colors)

    occupied = np.zeros_like(patch, dtype=bool)
    occupied[patch != BG] = True
    for color in inner_colors:
        if not _place_blob(patch, occupied, safe_box, int(color)):
            return None

    return patch


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_regions(scene: np.ndarray, counts: list[int]) -> list[dict[str, int | tuple[int, int, int, int]]]:
    placed: list[dict[str, int | tuple[int, int, int, int]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for count in counts:
        patch = _make_region(int(count))
        if patch is None:
            raise ValueError("failed to make region")
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(2, scene.shape[0] - ph - 1)
            for c in range(2, scene.shape[1] - pw - 1)
        ]
        random.shuffle(positions)
        chosen = None
        for r0, c0 in positions:
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_bbox_intersects(bbox, other, pad=5) for other in boxes):
                continue
            chosen = (r0, c0, bbox)
            break
        if chosen is None:
            raise ValueError("failed to place region")

        r0, c0, bbox = chosen
        patch_mask = patch != BG
        scene[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = patch[patch_mask]
        placed.append({"bbox": bbox, "inside_count": int(count)})
        boxes.append(bbox)
    return placed


def _counts_for_scene(style_map: dict[int, str]) -> list[int]:
    digits_by_style = {
        style: [digit for digit, mapped in style_map.items() if mapped == style]
        for style in STYLES
    }
    counts = [random.choice(digits_by_style[style]) for style in STYLES]
    if random.random() < 0.7:
        counts.append(random.randint(1, 4))
    random.shuffle(counts)
    return counts


def _expected_output(scene: np.ndarray, style_map: dict[int, str]) -> np.ndarray:
    out = scene.copy()
    for spec in _region_specs(scene, BG):
        _draw_shell(out, spec["bbox"], style_map[int(spec["inside_count"])], int(spec["border_color"]))
    return out


def _generate_case() -> np.ndarray | None:
    style_map = _sample_style_map()
    strip = _build_template_strip(style_map)
    counts = _counts_for_scene(style_map)
    scene_h = random.randint(22, 29) if len(counts) == 3 else random.randint(26, 33)
    scene_w = strip.shape[1] + random.randint(0, 6)
    scene = np.full((scene_h, scene_w), BG, dtype=int)
    try:
        _place_regions(scene, counts)
    except ValueError:
        return None

    expected = _expected_output(scene, style_map)
    inp = np.full((scene_h + strip.shape[0], scene_w), BG, dtype=int)
    inp[:scene_h] = scene
    strip_c0 = random.randint(0, scene_w - strip.shape[1])
    inp[scene_h:, strip_c0 : strip_c0 + strip.shape[1]] = strip
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
