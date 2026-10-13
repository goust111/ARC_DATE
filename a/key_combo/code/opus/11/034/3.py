import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
COMB_COLORS = (BLUE, RED, GREEN, YELLOW)

GEN_RETRIES = 400


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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = list(map(tuple, np.argwhere(mask)))
    if not coords:
        raise AssertionError("empty mask")
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((k, k), dtype=int)).astype(bool)


def _separator_row(grid: np.ndarray) -> int:
    rows = np.where(np.all(grid == GREY, axis=1))[0]
    assert len(rows) == 1, "expected one full-width grey separator"
    return int(rows[0])


def _extract_template(grid: np.ndarray, sep: int) -> np.ndarray:
    lower = grid[sep + 1 :, :]
    comps = _components(lower == PINK)
    assert len(comps) == 1, "expected one magenta template"
    local_mask = np.zeros_like(lower, dtype=bool)
    for r, c in comps[0]:
        local_mask[r, c] = True
    sprite = _crop_mask(local_mask)
    h, w = sprite.shape
    assert 2 <= h <= 3 and 2 <= w <= 3, "template bbox must be 2x2..3x3"
    assert int(sprite.sum()) >= 3, "template must be nontrivial"
    assert int(sprite.sum()) < h * w, "template should be irregular rather than full"
    return sprite


def _parse_comb(local_mask: np.ndarray, sep: int, color: int) -> dict:
    coords = list(map(tuple, np.argwhere(local_mask)))
    r0, c0, r1, c1 = _bbox(coords)
    assert c1 - c0 + 1 == 3, "comb must be 3 columns wide"
    assert r1 == sep - 1, "comb must be bottom-aligned to separator"

    rows: list[tuple[int, ...]] = []
    for r in range(r0, r1 + 1):
        cols = tuple(int(x) for x in np.where(local_mask[r])[0])
        rows.append(cols)

    for i, cols in enumerate(rows):
        if i % 2 == 0:
            assert cols == (c0, c1), "comb open rows must occupy both sides"
        else:
            assert cols == (c0, c0 + 1, c1), "comb solid rows must fill all 3 columns"

    k = len(rows) // 2
    assert len(rows) == 2 * k + 1 and 1 <= k <= 4, "invalid comb height"
    return {
        "color": int(color),
        "k": int(k),
        "center_c": int(c0 + 1),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
    }


def _find_combs(grid: np.ndarray, sep: int) -> list[dict]:
    upper = grid[:sep, :]
    combs: list[dict] = []
    for color in COMB_COLORS:
        mask = upper == color
        if not mask.any():
            continue
        comps = _components(mask)
        assert len(comps) == 1, "each comb color should form one component"
        full_mask = np.zeros_like(grid, dtype=bool)
        for r, c in comps[0]:
            full_mask[r, c] = True
        combs.append(_parse_comb(full_mask, sep, color))

    assert 2 <= len(combs) <= 4, "expected 2 to 4 combs"
    ks = [c["k"] for c in combs]
    assert len(set(ks)) == len(ks), "comb codes must be distinct"
    combs.sort(key=lambda d: d["center_c"])
    return combs


def _paint_stamp(out: np.ndarray, stamp_mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = stamp_mask.shape
    assert 0 <= top and top + h <= out.shape[0]
    assert 0 <= left and left + w <= out.shape[1]
    patch = out[top : top + h, left : left + w]
    patch[stamp_mask] = color


def _compose_output(shape: tuple[int, int], sep: int, template: np.ndarray, combs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    out[sep, :] = GREY
    top = sep + 1
    for comb in sorted(combs, key=lambda d: d["k"]):
        stamp = _scale_mask(template, comb["k"])
        left = comb["center_c"] - stamp.shape[1] // 2
        _paint_stamp(out, stamp, top, left, comb["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each comb's solid-row count into a local stamp scale and place recolored enlarged copies under the separator.

    Input structure:
    - A black background grid contains exactly one full-width grey separator row.
    - Above the separator are 2 to 4 distinct-color 3-column combs, each bottom-aligned to the separator and built from alternating side-pair rows and solid rows.
    - Below the separator sits one small magenta 4-connected template sprite, cropped by its own minimal bounding box.

    Rule:
    - Identify every comb and count its solid 3-pixel rows; that count is the comb's scale factor K.
    - Crop the magenta template, enlarge it by pixel replication with factor K, and recolor the enlarged copy to the comb's color.
    - Erase the original combs and the original template, keep only the grey separator, and place each enlarged copy with its top edge at the row just below the separator.
    - Horizontally anchor each enlarged copy by the comb center column using integer-grid centering `left = center_col - floor(stamp_width / 2)`; if stamps overlap, larger K is painted later and therefore overwrites smaller K.

    Key insight:
    - Different comb attributes control different parts of the output: solid-row count gives scale, horizontal position gives placement, and comb color gives recoloring.
    """
    g = np.asarray(grid, dtype=int)
    sep = _separator_row(g)
    template = _extract_template(g, sep)
    combs = _find_combs(g, sep)
    return _compose_output(g.shape, sep, template, combs)


def _random_connected_shape(rng: random.Random, h: int, w: int, area: int) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    while len(cells) < area:
        r, c = rng.choice(tuple(cells))
        nbrs = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells]
        if not nbrs:
            continue
        cells.add(rng.choice(nbrs))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_template(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.choice((2, 3))
        w = rng.choice((2, 3))
        area_min = max(3, (h * w + 1) // 2)
        area = rng.randint(area_min, h * w - 1)
        mask = _random_connected_shape(rng, h, w, area)
        if mask.sum() == h * w:
            continue
        if not np.all(mask.any(axis=0)) or not np.all(mask.any(axis=1)):
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    raise RuntimeError("template generation failed")


def _comb_rows(k: int) -> list[np.ndarray]:
    rows = []
    for i in range(2 * k + 1):
        if i % 2 == 0:
            rows.append(np.array([1, 0, 1], dtype=int))
        else:
            rows.append(np.array([1, 1, 1], dtype=int))
    return rows


def _stamp_interval(center_c: int, template_w: int, k: int) -> tuple[int, int]:
    w = template_w * k
    left = center_c - w // 2
    return left, left + w - 1


def _pairwise_overlap_pixels(template: np.ndarray, combs: list[dict], sep: int, shape: tuple[int, int]) -> list[int]:
    canvas = np.zeros(shape, dtype=int)
    overlaps: list[int] = []
    top = sep + 1
    stamps = []
    for comb in combs:
        stamp = _scale_mask(template, comb["k"])
        left = comb["center_c"] - stamp.shape[1] // 2
        stamps.append((stamp, top, left))
    for i in range(len(stamps)):
        for j in range(i + 1, len(stamps)):
            mask_i = np.zeros(shape, dtype=bool)
            mask_j = np.zeros(shape, dtype=bool)
            stamp_i, top_i, left_i = stamps[i]
            stamp_j, top_j, left_j = stamps[j]
            mask_i[top_i : top_i + stamp_i.shape[0], left_i : left_i + stamp_i.shape[1]] = stamp_i
            mask_j[top_j : top_j + stamp_j.shape[0], left_j : left_j + stamp_j.shape[1]] = stamp_j
            overlaps.append(int(np.logical_and(mask_i, mask_j).sum()))
    return overlaps


def _choose_combs(
    rng: random.Random, width: int, template: np.ndarray, count: int, ks: list[int], want_overlap: bool
) -> list[dict] | None:
    for _ in range(300):
        colors = rng.sample(list(COMB_COLORS), count)
        rng.shuffle(ks)
        pairs = list(zip(colors, ks))
        combs: list[dict] = []
        used_centers: list[int] = []
        for color, k in pairs:
            stamp_w = template.shape[1] * k
            c_min = max(1, stamp_w // 2)
            c_max = min(width - 2, width - stamp_w + stamp_w // 2)
            candidates = [c for c in range(c_min, c_max + 1) if all(abs(c - prev) >= 4 for prev in used_centers)]
            if not candidates:
                break
            if rng.random() < 0.3:
                edge_candidates = [c for c in candidates if c in (candidates[0], candidates[-1])]
                center = rng.choice(edge_candidates or candidates)
            else:
                center = rng.choice(candidates)
            used_centers.append(center)
            combs.append({"color": int(color), "k": int(k), "center_c": int(center)})
        if len(combs) != count:
            continue

        combs.sort(key=lambda d: d["center_c"])
        if count >= 3:
            gaps = [combs[i + 1]["center_c"] - combs[i]["center_c"] for i in range(count - 1)]
            if len(set(gaps)) == 1:
                continue

        overlaps = []
        for i in range(count):
            left_i, right_i = _stamp_interval(combs[i]["center_c"], template.shape[1], combs[i]["k"])
            for j in range(i + 1, count):
                left_j, right_j = _stamp_interval(combs[j]["center_c"], template.shape[1], combs[j]["k"])
                overlaps.append(max(0, min(right_i, right_j) - max(left_i, left_j) + 1))
        has_overlap = any(v > 0 for v in overlaps)
        if has_overlap != want_overlap:
            continue
        if want_overlap and overlaps and max(overlaps) > max(template.shape) * 3:
            continue
        return combs
    return None


def _build_input(
    shape: tuple[int, int], sep: int, template: np.ndarray, combs: list[dict], template_row: int, template_col: int
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    grid[sep, :] = GREY

    for comb in combs:
        k = comb["k"]
        left = comb["center_c"] - 1
        top = sep - (2 * k + 1)
        comb["bbox"] = (top, left, sep - 1, left + 2)
        for i, row in enumerate(_comb_rows(k)):
            rr = top + i
            for dc in range(3):
                if row[dc]:
                    grid[rr, left + dc] = comb["color"]

    for r in range(template.shape[0]):
        for c in range(template.shape[1]):
            if template[r, c]:
                grid[template_row + r, template_col + c] = PINK
    return grid


def _metadata_signature(combs: list[dict]) -> list[tuple[int, int, int]]:
    return sorted((int(c["color"]), int(c["k"]), int(c["center_c"])) for c in combs)


def _validate_instance(inp: np.ndarray, expected_template: np.ndarray, expected_combs: list[dict]) -> None:
    sep = _separator_row(inp)
    parsed_template = _extract_template(inp, sep)
    parsed_combs = _find_combs(inp, sep)
    assert np.array_equal(parsed_template, expected_template), "template parse mismatch"
    assert _metadata_signature(parsed_combs) == _metadata_signature(expected_combs), "comb parse mismatch"

    out = main(inp)
    intended = _compose_output(inp.shape, sep, expected_template, expected_combs)
    assert np.array_equal(out, intended), "solver output mismatch"
    assert np.all(out[:sep, :] == 0), "all combs must be erased in output"
    assert np.all(out[sep, :] == GREY), "separator must be preserved"
    assert len(parsed_combs) >= 2 and len({c["k"] for c in parsed_combs}) >= 2, "non-degenerate comb set required"
    assert int(parsed_template.sum()) >= 3 and int(parsed_template.sum()) < parsed_template.size, "template must stay irregular"

    overlaps = _pairwise_overlap_pixels(expected_template, expected_combs, sep, inp.shape)
    if any(v > 0 for v in overlaps):
        smaller_areas = []
        for i in range(len(expected_combs)):
            for j in range(i + 1, len(expected_combs)):
                a = int(_scale_mask(expected_template, expected_combs[i]["k"]).sum())
                b = int(_scale_mask(expected_template, expected_combs[j]["k"]).sum())
                smaller_areas.append(min(a, b))
        assert any(0 < ov < small for ov, small in zip(overlaps, smaller_areas)), "overlap should be visible but not total"

    for comb in expected_combs:
        stamp = _scale_mask(expected_template, comb["k"])
        left = comb["center_c"] - stamp.shape[1] // 2
        assert 0 <= left and left + stamp.shape[1] <= inp.shape[1], "stamp must fit horizontally"
        assert sep + 1 + stamp.shape[0] <= inp.shape[0], "stamp must fit vertically"


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((2, 3, 4))
    ks = rng.sample((1, 2, 3, 4), count)
    template = _random_template(rng)
    max_comb_h = max(2 * k + 1 for k in ks)
    max_stamp_h = template.shape[0] * max(ks)
    min_height = max_comb_h + 1 + max_stamp_h
    if min_height > 30:
        return None

    height = rng.randint(max(18, min_height), 30)
    sep_min = max_comb_h
    sep_max = height - max_stamp_h - 1
    if sep_min > sep_max:
        return None
    sep = rng.randint(sep_min, sep_max)

    min_width = max(14, 4 * count + 5, template.shape[1] * max(ks) + 2)
    if min_width > 30:
        return None
    width = rng.randint(min_width, 30)

    want_overlap = rng.random() < 0.45
    combs = _choose_combs(rng, width, template, count, ks[:], want_overlap)
    if combs is None:
        return None

    template_col = rng.randint(0, min(2, width - template.shape[1]))
    template_row_min = sep + 1
    template_row_max = height - template.shape[0]
    if template_row_min > template_row_max:
        return None
    low_bias_start = max(template_row_min, height - template.shape[0] - 3)
    template_row = rng.randint(low_bias_start, template_row_max)

    inp = _build_input((height, width), sep, template, combs, template_row, template_col)

    lower_non_template = inp[sep + 1 :, :] != 0
    pink_mask = inp[sep + 1 :, :] == PINK
    if int(np.logical_and(lower_non_template, ~pink_mask).sum()) != 0:
        return None

    try:
        _validate_instance(inp, template, combs)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid comb-scaling instance")
