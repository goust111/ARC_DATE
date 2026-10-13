from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
OBJECT_COLORS = (3, 6, 7, 8, 9)

GEOMETRIES = {
    "rot_cw": lambda patch: np.rot90(patch, -1).copy(),
    "rot_ccw": lambda patch: np.rot90(patch, 1).copy(),
    "rot_180": lambda patch: np.rot90(patch, 2).copy(),
    "flip_h": lambda patch: np.fliplr(patch).copy(),
    "flip_v": lambda patch: np.flipud(patch).copy(),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _object_mask(arr: np.ndarray) -> np.ndarray:
    return (arr != BG) & (arr != BLUE) & (arr != RED) & (arr != YELLOW) & (arr != GREY)


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _find_singleton_color(band: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(band == color)
    assert len(coords) == 1, f"expected exactly one color-{color} marker"
    return int(coords[0, 0]), int(coords[0, 1])


def _extract_side_patch(
    band: np.ndarray, marker_c: int, side: str
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if side == "left":
        sub = band[:, :marker_c]
        offset_c = 0
    else:
        sub = band[:, marker_c + 1 :]
        offset_c = marker_c + 1
    mask = _object_mask(sub)
    assert mask.any(), f"expected one object on the {side} side"
    assert _component_count(mask) == 1, f"{side} side must contain one connected object"
    r0, c0, r1, c1 = _bbox(mask)
    patch = sub[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch, (r0, c0 + offset_c, r1, c1 + offset_c)


def _infer_recolor(src: np.ndarray, dst: np.ndarray) -> tuple[int, int] | None:
    src_fg = src != BG
    if src.shape != dst.shape:
        return None
    if not np.array_equal(src_fg, dst != BG):
        return None
    if np.any(dst[~src_fg] != BG):
        return None

    mapping: dict[int, int] = {}
    for color in sorted(set(map(int, src[src_fg]))):
        targets = {int(v) for v in np.unique(dst[src == color])}
        if len(targets) != 1:
            return None
        mapping[color] = next(iter(targets))

    changed = [(src_color, dst_color) for src_color, dst_color in mapping.items() if src_color != dst_color]
    if len(changed) != 1:
        return None
    replace_from, replace_to = changed[0]
    if replace_to == BG:
        return None
    return replace_from, replace_to


def _rule_candidates(input_patch: np.ndarray, output_patch: np.ndarray) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for name, op in GEOMETRIES.items():
        transformed = op(input_patch)
        recolor = _infer_recolor(transformed, output_patch)
        if recolor is None:
            continue
        candidates.append((name, recolor[0], recolor[1]))
    return candidates


def _infer_rule(input_patch: np.ndarray, output_patch: np.ndarray) -> tuple[str, int, int]:
    candidates = _rule_candidates(input_patch, output_patch)
    assert len(candidates) == 1, "teaching example must imply a unique rule"
    return candidates[0]


def _apply_rule(patch: np.ndarray, rule: tuple[str, int, int]) -> np.ndarray:
    geom_name, replace_from, replace_to = rule
    transformed = GEOMETRIES[geom_name](patch)
    out = transformed.copy()
    out[transformed == replace_from] = replace_to
    return out


def _output_top(marker_r: int, patch_h: int) -> int:
    return int(marker_r - patch_h // 2)


def _output_left(marker_c: int) -> int:
    return int(marker_c + 2)


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left
    assert top + h <= canvas.shape[0] and left + w <= canvas.shape[1]
    mask = patch != BG
    region = canvas[top : top + h, left : left + w]
    assert np.all(region[mask] == BG), "placed object must not overlap existing pixels"
    region[mask] = patch[mask]


def _parse_band(band: np.ndarray, expect_output: bool) -> dict[str, object]:
    marker_r, marker_c = _find_singleton_color(band, YELLOW)
    input_patch, input_bbox = _extract_side_patch(band, marker_c, "left")
    info: dict[str, object] = {
        "marker_r": marker_r,
        "marker_c": marker_c,
        "input_patch": input_patch,
        "input_bbox": input_bbox,
    }
    if expect_output:
        output_patch, output_bbox = _extract_side_patch(band, marker_c, "right")
        info["output_patch"] = output_patch
        info["output_bbox"] = output_bbox
    else:
        assert not _object_mask(band[:, marker_c + 1 :]).any(), "test target area must start blank"
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer two teaching rules of the form “geometry then recolor”, select one
    via a red/blue marker, and apply it to the test object in the bottom band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]
    assert len(separator_rows) == 2, "expected exactly two grey separator rows"
    top_sep, mid_sep = separator_rows

    band_a = g[:top_sep, :].copy()
    band_b = g[top_sep + 1 : mid_sep, :].copy()
    test_band = g[mid_sep + 1 :, :].copy()
    assert band_a.size > 0 and band_b.size > 0 and test_band.size > 0

    info_a = _parse_band(band_a, expect_output=True)
    info_b = _parse_band(band_b, expect_output=True)
    rule_a = _infer_rule(info_a["input_patch"], info_a["output_patch"])
    rule_b = _infer_rule(info_b["input_patch"], info_b["output_patch"])

    for info in (info_a, info_b):
        out_patch = info["output_patch"]
        out_bbox = info["output_bbox"]
        marker_r = int(info["marker_r"])
        marker_c = int(info["marker_c"])
        assert out_bbox[1] == _output_left(marker_c), "teaching output must use the standard right-side anchor"
        assert out_bbox[0] == _output_top(marker_r, out_patch.shape[0]), "teaching output must align to the yellow anchor"

    indicator_r, indicator_c = _find_singleton_color(test_band, RED) if np.any(test_band == RED) else _find_singleton_color(test_band, BLUE)
    indicator_color = int(test_band[indicator_r, indicator_c])
    assert indicator_color in {RED, BLUE}

    test_info = _parse_band(test_band, expect_output=False)
    chosen_rule = rule_a if indicator_color == RED else rule_b
    result_patch = _apply_rule(test_info["input_patch"], chosen_rule)

    out = test_band.copy()
    top = _output_top(int(test_info["marker_r"]), result_patch.shape[0])
    left = _output_left(int(test_info["marker_c"]))
    _paste_patch(out, result_patch, top, left)
    return out


# --------------- generator ---------------

BAND_HEIGHT_RANGE = (7, 10)
GRID_WIDTH_RANGE = (18, 28)
PATCH_HW_RANGE = (2, 5)
PATCH_MIN_CELLS = 4
PATCH_ATTEMPTS = 100
GEN_RETRIES = 400


def _random_pair() -> tuple[int, int]:
    replace_from, replace_to = random.sample(OBJECT_COLORS, 2)
    return int(replace_from), int(replace_to)


def _generate_connected_mask() -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        h = random.randint(*PATCH_HW_RANGE)
        w = random.randint(*PATCH_HW_RANGE)
        area = h * w
        min_cells = max(PATCH_MIN_CELLS, h + w - 2)
        if area <= min_cells:
            continue
        target_cells = random.randint(min_cells, area - 1)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        frontier = set(_neighbors4(seed[0], seed[1], h, w))
        while len(cells) < target_cells and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nb in _neighbors4(cell[0], cell[1], h, w):
                if nb not in cells:
                    frontier.add(nb)
        if len(cells) != target_cells:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        return mask
    return None


def _colorize_mask(
    mask: np.ndarray,
    required_colors: set[int],
    preferred_colors: tuple[int, ...] = (),
) -> np.ndarray | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < len(required_colors):
        return None

    min_color_count = max(2, len(required_colors))
    color_count = random.randint(min_color_count, 3)
    palette = list(required_colors)
    for color in preferred_colors:
        if len(palette) >= color_count:
            break
        if color not in palette and random.random() < 0.5:
            palette.append(int(color))
    remaining = [c for c in OBJECT_COLORS if c not in palette]
    random.shuffle(remaining)
    while len(palette) < color_count and remaining:
        palette.append(int(remaining.pop()))

    assigned = list(palette)
    source = next(iter(required_colors))
    if len(coords) > len(assigned):
        assigned.append(int(source))
    while len(assigned) < len(coords):
        assigned.append(int(random.choice(palette)))
    random.shuffle(assigned)

    patch = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, assigned):
        patch[r, c] = int(color)
    return patch


def _generate_patch_for_rule(rule: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    geom_name, replace_from, replace_to = rule
    for _ in range(PATCH_ATTEMPTS):
        mask = _generate_connected_mask()
        if mask is None:
            continue
        patch = _colorize_mask(mask, required_colors={replace_from}, preferred_colors=(replace_to,))
        if patch is None:
            continue
        output_patch = _apply_rule(patch, rule)
        if _rule_candidates(patch, output_patch) == [rule]:
            return patch, output_patch
    return None


def _generate_test_patch(required_color: int) -> np.ndarray | None:
    for _ in range(PATCH_ATTEMPTS):
        mask = _generate_connected_mask()
        if mask is None:
            continue
        patch = _colorize_mask(mask, required_colors={required_color})
        if patch is not None:
            return patch
    return None


def _sample_band_height(*patches: np.ndarray) -> int:
    max_h = max(p.shape[0] for p in patches)
    low = max(BAND_HEIGHT_RANGE[0], max_h + 1)
    high = BAND_HEIGHT_RANGE[1]
    assert low <= high
    return random.randint(low, high)


def _sample_grid_width(min_width: int) -> int:
    low = max(GRID_WIDTH_RANGE[0], min_width)
    high = GRID_WIDTH_RANGE[1]
    assert low <= high
    return random.randint(low, high)


def _place_left_patch(band: np.ndarray, patch: np.ndarray, marker_c: int) -> None:
    h, w = patch.shape
    top = random.randint(0, band.shape[0] - h)
    left = random.randint(1, marker_c - w - 1)
    _paste_patch(band, patch, top, left)


def _marker_row_range(band_h: int, out_h: int) -> tuple[int, int]:
    low = out_h // 2
    high = band_h - (out_h - out_h // 2)
    assert low <= high
    return low, high


def _place_teaching_band(band_h: int, width: int, input_patch: np.ndarray, output_patch: np.ndarray) -> np.ndarray | None:
    in_h, in_w = input_patch.shape
    out_h, out_w = output_patch.shape
    marker_min = in_w + 2
    marker_max = width - out_w - 3
    if marker_min > marker_max:
        return None

    band = np.zeros((band_h, width), dtype=int)
    marker_c = random.randint(marker_min, marker_max)
    low_r, high_r = _marker_row_range(band_h, out_h)
    marker_r = random.randint(low_r, high_r)

    _place_left_patch(band, input_patch, marker_c)
    top = _output_top(marker_r, out_h)
    left = _output_left(marker_c)
    _paste_patch(band, output_patch, top, left)
    band[marker_r, marker_c] = YELLOW
    return band


def _place_test_band(
    band_h: int,
    width: int,
    input_patch: np.ndarray,
    result_patch: np.ndarray,
    indicator_color: int,
) -> np.ndarray | None:
    in_h, in_w = input_patch.shape
    out_h, out_w = result_patch.shape
    marker_min = in_w + 2
    marker_max = width - out_w - 3
    if marker_min > marker_max:
        return None

    band = np.zeros((band_h, width), dtype=int)
    marker_c = random.randint(marker_min, marker_max)
    low_r, high_r = _marker_row_range(band_h, out_h)
    marker_r = random.randint(low_r, high_r)

    _place_left_patch(band, input_patch, marker_c)
    band[marker_r, marker_c] = YELLOW
    band[-1, -1] = indicator_color

    top = _output_top(marker_r, out_h)
    left = _output_left(marker_c)
    if left + out_w > width - 1:
        return None
    if top <= band.shape[0] - 1 <= top + out_h - 1 and left <= width - 1 <= left + out_w - 1:
        return None
    return band


def _assemble_grid(band_a: np.ndarray, band_b: np.ndarray, test_band: np.ndarray) -> np.ndarray:
    width = band_a.shape[1]
    assert band_b.shape[1] == width and test_band.shape[1] == width
    sep1 = np.full((1, width), GREY, dtype=int)
    sep2 = np.full((1, width), GREY, dtype=int)
    return np.vstack([band_a, sep1, band_b, sep2, test_band])


def _gen_attempt() -> np.ndarray | None:
    share_kind = random.choice(("geometry", "recolor"))
    indicator_color = random.choice((RED, BLUE))

    geom_a = random.choice(tuple(GEOMETRIES))
    pair_a = _random_pair()
    rule_a = (geom_a, pair_a[0], pair_a[1])

    if share_kind == "geometry":
        geom_b = geom_a
        pair_b = _random_pair()
        while pair_b == pair_a:
            pair_b = _random_pair()
    else:
        pair_b = pair_a
        geom_b = random.choice([name for name in GEOMETRIES if name != geom_a])
    rule_b = (geom_b, pair_b[0], pair_b[1])

    ex_a = _generate_patch_for_rule(rule_a)
    ex_b = _generate_patch_for_rule(rule_b)
    if ex_a is None or ex_b is None:
        return None
    input_a, output_a = ex_a
    input_b, output_b = ex_b

    chosen_rule = rule_a if indicator_color == RED else rule_b
    test_input = _generate_test_patch(chosen_rule[1])
    if test_input is None:
        return None
    test_output = _apply_rule(test_input, chosen_rule)

    band_h_a = _sample_band_height(input_a, output_a)
    band_h_b = _sample_band_height(input_b, output_b)
    band_h_test = _sample_band_height(test_input, test_output)

    min_width = max(
        input_a.shape[1] + output_a.shape[1] + 6,
        input_b.shape[1] + output_b.shape[1] + 6,
        test_input.shape[1] + test_output.shape[1] + 6,
    )
    width = _sample_grid_width(min_width)

    band_a = _place_teaching_band(band_h_a, width, input_a, output_a)
    band_b = _place_teaching_band(band_h_b, width, input_b, output_b)
    test_band = _place_test_band(band_h_test, width, test_input, test_output, indicator_color)
    if band_a is None or band_b is None or test_band is None:
        return None

    grid = _assemble_grid(band_a, band_b, test_band)

    try:
        solved = main(grid)
    except AssertionError:
        return None
    except Exception:
        return None

    expected = test_band.copy()
    marker_r, marker_c = _find_singleton_color(test_band, YELLOW)
    _paste_patch(expected, test_output, _output_top(marker_r, test_output.shape[0]), _output_left(marker_c))
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
