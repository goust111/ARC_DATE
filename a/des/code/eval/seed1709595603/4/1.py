from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
MAROON = 9
FRAME_COLORS = (1, 2, 3, 4, 6, 8)

INTERIOR_SIDE_OPTIONS = (4, 5)
DICT_ENTRY_COUNT_OPTIONS = (3, 4)
FRAME_COUNT_RANGE = (2, 5)
GRID_SIDE_MIN = 20
GRID_SIDE_MAX = 30
GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if r > 0:
        out.append((int(r - 1), int(c)))
    if r + 1 < h:
        out.append((int(r + 1), int(c)))
    if c > 0:
        out.append((int(r), int(c - 1)))
    if c + 1 < w:
        out.append((int(r), int(c + 1)))
    return out


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            coords.append((int(cr), int(cc2)))
            for nr, nc in _neighbors4(int(cr), int(cc2), h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
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


def _count_components(mask: np.ndarray) -> int:
    return int(len(_components(np.asarray(mask, dtype=bool))))


def _split_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((int(start), int(prev)))
        start = prev = value
    runs.append((int(start), int(prev)))
    return runs


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    banned = {BG, GREY, ORANGE, MAROON}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in banned):
        for comp in _components(grid == int(color)):
            r0, c0, r1, c1 = comp["bbox"]
            side_h = int(r1 - r0 + 1)
            side_w = int(c1 - c0 + 1)
            if side_h != side_w or side_h < 4:
                continue
            region = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((side_h, side_w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not bool(np.all(region[border] == int(color))):
                continue
            if bool(np.any(region[1:-1, 1:-1] == int(color))):
                continue
            if int(comp["area"]) != int(border.sum()):
                continue
            frames.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": int(color),
                    "side": int(side_h),
                }
            )
    assert frames, "expected at least one colored frame"
    sides = {int(frame["side"]) for frame in frames}
    assert len(sides) == 1, "all frames must share one size"
    frames.sort(key=lambda frame: tuple(int(v) for v in frame["bbox"]))
    return frames


def _parse_dictionary(grid: np.ndarray, inner_side: int) -> dict[int, np.ndarray]:
    mask = (grid == MAROON) | (grid == ORANGE)
    b = _bbox(mask)
    assert b is not None, "dictionary area not found"
    r0, c0, r1, c1 = b
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    occ = crop != BG
    row_bands = _split_runs(np.where(occ.any(axis=1))[0].astype(int).tolist())
    col_bands = _split_runs(np.where(occ.any(axis=0))[0].astype(int).tolist())

    entries: list[tuple[int, np.ndarray]] = []
    if len(col_bands) == 1 and int(col_bands[0][1] - col_bands[0][0] + 1) == int(inner_side):
        c_start = int(c0 + col_bands[0][0])
        for band_r0, band_r1 in row_bands:
            assert int(band_r1 - band_r0 + 1) == int(inner_side + 1)
            r_start = int(r0 + band_r0)
            pattern = grid[r_start : r_start + inner_side, c_start : c_start + inner_side] == MAROON
            label = grid[r_start + inner_side, c_start : c_start + inner_side]
            assert np.all((label == BG) | (label == ORANGE))
            number = int(np.count_nonzero(label == ORANGE))
            assert 1 <= number <= 5
            assert bool(pattern.any())
            entries.append((int(number), pattern.copy()))
    elif len(row_bands) == 1 and int(row_bands[0][1] - row_bands[0][0] + 1) == int(inner_side + 1):
        r_start = int(r0 + row_bands[0][0])
        for band_c0, band_c1 in col_bands:
            assert int(band_c1 - band_c0 + 1) == int(inner_side)
            c_start = int(c0 + band_c0)
            pattern = grid[r_start : r_start + inner_side, c_start : c_start + inner_side] == MAROON
            label = grid[r_start + inner_side, c_start : c_start + inner_side]
            assert np.all((label == BG) | (label == ORANGE))
            number = int(np.count_nonzero(label == ORANGE))
            assert 1 <= number <= 5
            assert bool(pattern.any())
            entries.append((int(number), pattern.copy()))
    else:
        raise AssertionError("unsupported dictionary layout")

    stamps: dict[int, np.ndarray] = {}
    for number, pattern in entries:
        assert number not in stamps, "dictionary ids must be unique"
        assert pattern.shape == (int(inner_side), int(inner_side))
        stamps[int(number)] = pattern
    assert stamps, "expected at least one dictionary entry"
    return stamps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count grey connected components inside each colored frame, pick the stamp with
    the matching orange-numbered label, recolor the maroon pixels to the frame
    color, and clear the dictionary area.
    """
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    inner_side = int(frames[0]["side"] - 2)
    stamps = _parse_dictionary(g, inner_side)

    out = g.copy()
    out[(out == MAROON) | (out == ORANGE)] = BG

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        color = int(frame["color"])
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        count = _count_components(inner == GREY)
        assert count in stamps, "missing stamp for grey-component count"
        stamp = stamps[int(count)]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.where(stamp, int(color), BG)

    return out


def _random_stamp_mask(side: int) -> np.ndarray | None:
    area = int(side * side)
    min_fill = max(int(side + 1), int(np.ceil(area * 0.28)))
    max_fill = min(int(area - 1), int(np.floor(area * 0.58)))
    if min_fill > max_fill:
        return None
    target = random.randint(min_fill, max_fill)
    mask = np.zeros((side, side), dtype=bool)
    cells = random.sample([(r, c) for r in range(side) for c in range(side)], target)
    for r, c in cells:
        mask[int(r), int(c)] = True

    for r in range(side):
        if not bool(mask[r].any()):
            mask[int(r), random.randrange(side)] = True
    for c in range(side):
        if not bool(mask[:, c].any()):
            mask[random.randrange(side), int(c)] = True

    if bool(mask.all()):
        choices = [
            (r, c)
            for r in range(side)
            for c in range(side)
            if int(mask[r].sum()) > 1 and int(mask[:, c].sum()) > 1
        ]
        if not choices:
            return None
        rr, cc = random.choice(choices)
        mask[int(rr), int(cc)] = False

    if not bool(mask.any()) or bool(mask.all()):
        return None
    if not bool(np.all(mask.any(axis=0))) or not bool(np.all(mask.any(axis=1))):
        return None
    return mask


def _isolated_seeds(side: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(side) for c in range(side)]
    for _ in range(120):
        random.shuffle(cells)
        seeds: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(int(r) - rr) + abs(int(c) - cc) > 1 for rr, cc in seeds):
                seeds.append((int(r), int(c)))
                if len(seeds) == int(count):
                    return seeds
    return None


def _safe_to_grow(cell: tuple[int, int], index: int, comps: list[set[tuple[int, int]]], side: int) -> bool:
    r, c = cell
    if not (0 <= r < side and 0 <= c < side):
        return False
    if any((int(r), int(c)) in comp for comp in comps):
        return False
    for other_index, other in enumerate(comps):
        if other_index == int(index):
            continue
        for nr, nc in _neighbors4(int(r), int(c), side, side):
            if (int(nr), int(nc)) in other:
                return False
    return True


def _grey_mask(side: int, component_count: int) -> np.ndarray | None:
    seeds = _isolated_seeds(side, component_count)
    if seeds is None:
        return None
    comps = [{seed} for seed in seeds]
    for index, comp in enumerate(comps):
        extra = random.randint(0, max(0, side - 2))
        for _ in range(extra):
            frontier: set[tuple[int, int]] = set()
            for r, c in comp:
                for nr, nc in _neighbors4(int(r), int(c), side, side):
                    if _safe_to_grow((int(nr), int(nc)), index, comps, side):
                        frontier.add((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            comp.add((int(cell[0]), int(cell[1])))

    mask = np.zeros((side, side), dtype=bool)
    for comp in comps:
        for r, c in comp:
            mask[int(r), int(c)] = True
    if _count_components(mask) != int(component_count):
        return None
    return mask


def _frame_patch(inner_side: int, color: int, component_count: int) -> np.ndarray | None:
    grey_mask = _grey_mask(inner_side, component_count)
    if grey_mask is None:
        return None
    outer = int(inner_side + 2)
    patch = np.full((outer, outer), BG, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    patch[1:-1, 1:-1][grey_mask] = GREY
    return patch


def _dictionary_patch(
    inner_side: int,
    stamps: dict[int, np.ndarray],
    order: list[int],
    orientation: str,
    gap: int,
) -> np.ndarray:
    if orientation == "vertical":
        height = len(order) * (inner_side + 1) + (len(order) - 1) * gap
        width = inner_side
        patch = np.full((height, width), BG, dtype=int)
        for idx, number in enumerate(order):
            top = idx * (inner_side + 1 + gap)
            stamp = stamps[int(number)]
            patch[top : top + inner_side, :][stamp] = MAROON
            start = random.randint(0, inner_side - int(number))
            patch[top + inner_side, start : start + int(number)] = ORANGE
        return patch

    height = inner_side + 1
    width = len(order) * inner_side + (len(order) - 1) * gap
    patch = np.full((height, width), BG, dtype=int)
    for idx, number in enumerate(order):
        left = idx * (inner_side + gap)
        stamp = stamps[int(number)]
        patch[:inner_side, left : left + inner_side][stamp] = MAROON
        start = random.randint(0, inner_side - int(number))
        patch[inner_side, left + start : left + start + int(number)] = ORANGE
    return patch


def _place_with_padding(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    if bool(occupied[r0:r1, c0:c1].any()):
        return False
    occupied[r0:r1, c0:c1] = True
    return True


def _sample_counts(ids: list[int], frame_count: int) -> list[int]:
    counts = [int(random.choice(ids)) for _ in range(frame_count)]
    if frame_count > 1 and len(ids) > 1 and len(set(counts)) == 1:
        choices = [value for value in ids if int(value) != int(counts[0])]
        counts[random.randrange(frame_count)] = int(random.choice(choices))
    return counts


def _expected_output(
    grid: np.ndarray,
    frames: list[dict],
    inner_side: int,
    stamps: dict[int, np.ndarray],
) -> np.ndarray:
    out = grid.copy()
    out[(out == MAROON) | (out == ORANGE)] = BG
    for frame in frames:
        top = int(frame["top"])
        left = int(frame["left"])
        color = int(frame["color"])
        number = int(frame["number"])
        stamp = stamps[int(number)]
        out[top + 1 : top + 1 + inner_side, left + 1 : left + 1 + inner_side] = np.where(
            stamp,
            int(color),
            BG,
        )
    return out


def _gen_attempt() -> np.ndarray | None:
    inner_side = int(random.choice(INTERIOR_SIDE_OPTIONS))
    outer_side = int(inner_side + 2)
    entry_count = int(random.choice(DICT_ENTRY_COUNT_OPTIONS))
    available_ids = list(range(1, min(5, inner_side) + 1))
    if len(available_ids) < entry_count:
        return None
    ids = sorted(random.sample(available_ids, entry_count))
    frame_count = random.randint(FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1])
    if frame_count > len(FRAME_COLORS):
        return None

    stamps: dict[int, np.ndarray] = {}
    seen_patterns: set[bytes] = set()
    for number in ids:
        for _ in range(120):
            stamp = _random_stamp_mask(inner_side)
            if stamp is None:
                continue
            key = stamp.astype(np.uint8).tobytes()
            if key in seen_patterns:
                continue
            seen_patterns.add(key)
            stamps[int(number)] = stamp
            break
        if number not in stamps:
            return None

    orientation = random.choice(["vertical", "horizontal"])
    gap = random.choice([1, 2])
    order = ids[:]
    random.shuffle(order)
    dictionary = _dictionary_patch(inner_side, stamps, order, orientation, gap)
    dict_h, dict_w = map(int, dictionary.shape)

    required_area = int(dict_h * dict_w + frame_count * (outer_side + 2) * (outer_side + 2) + 60)
    min_h = max(GRID_SIDE_MIN, dict_h + 1, outer_side + 4)
    min_w = max(GRID_SIDE_MIN, dict_w + 1, outer_side + 4)
    if min_h > GRID_SIDE_MAX or min_w > GRID_SIDE_MAX:
        return None

    grid_h = random.randint(min_h, GRID_SIDE_MAX)
    grid_w = random.randint(min_w, GRID_SIDE_MAX)
    if int(grid_h * grid_w) < required_area:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    corner = random.choice(["tl", "tr", "bl", "br"])
    edge_margin_r = random.randint(0, 1)
    edge_margin_c = random.randint(0, 1)
    if corner == "tl":
        dict_top = edge_margin_r
        dict_left = edge_margin_c
    elif corner == "tr":
        dict_top = edge_margin_r
        dict_left = grid_w - dict_w - edge_margin_c
    elif corner == "bl":
        dict_top = grid_h - dict_h - edge_margin_r
        dict_left = edge_margin_c
    else:
        dict_top = grid_h - dict_h - edge_margin_r
        dict_left = grid_w - dict_w - edge_margin_c

    if not _place_with_padding(occupied, dict_top, dict_left, dict_h, dict_w, pad=2):
        return None
    grid[dict_top : dict_top + dict_h, dict_left : dict_left + dict_w] = dictionary

    colors = random.sample(list(FRAME_COLORS), frame_count)
    counts = _sample_counts(ids, frame_count)
    frames: list[dict] = []
    for color, number in zip(colors, counts):
        patch = _frame_patch(inner_side, int(color), int(number))
        if patch is None:
            return None
        placed = False
        for _ in range(300):
            top = random.randint(0, grid_h - outer_side)
            left = random.randint(0, grid_w - outer_side)
            if not _place_with_padding(occupied, top, left, outer_side, outer_side, pad=1):
                continue
            grid[top : top + outer_side, left : left + outer_side] = patch
            frames.append(
                {
                    "top": int(top),
                    "left": int(left),
                    "color": int(color),
                    "number": int(number),
                }
            )
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, frames, inner_side, stamps)
    try:
        produced = main(grid)
    except Exception:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATION_ATTEMPTS} attempts")
