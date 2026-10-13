import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

BODY_COLORS = (6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
TARGET_BY_COLOR = {
    RED: "top",
    BLUE: "bottom",
    GREEN: "left",
    YELLOW: "right",
}
COLOR_BY_TARGET = {side: color for color, side in TARGET_BY_COLOR.items()}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        pts: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            pts.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _filled_rect(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for comp in _components(grid == GREY):
        if not _filled_rect(comp):
            continue
        bb = _bbox(comp)
        area = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
        candidates.append((area, bb))
    assert candidates, "expected a solid grey canvas"
    return max(candidates, key=lambda item: item[0])[1]


def _parse_stamp(stamp: np.ndarray) -> tuple[str, str, np.ndarray]:
    h, w = stamp.shape
    candidates: list[tuple[str, int]] = []

    if h > 1:
        color = int(stamp[0, 0])
        if color in TARGET_BY_COLOR and np.all(stamp[0, :] == color):
            candidates.append(("top", color))
        color = int(stamp[h - 1, 0])
        if color in TARGET_BY_COLOR and np.all(stamp[h - 1, :] == color):
            candidates.append(("bottom", color))
    if w > 1:
        color = int(stamp[0, 0])
        if color in TARGET_BY_COLOR and np.all(stamp[:, 0] == color):
            candidates.append(("left", color))
        color = int(stamp[0, w - 1])
        if color in TARGET_BY_COLOR and np.all(stamp[:, w - 1] == color):
            candidates.append(("right", color))

    assert len(candidates) == 1, "each stamp must have exactly one locator edge"
    locator_side, locator_color = candidates[0]

    if locator_side == "top":
        body = np.flipud(stamp[1:, :])
    elif locator_side == "bottom":
        body = np.flipud(stamp[:-1, :])
    elif locator_side == "left":
        body = np.fliplr(stamp[:, 1:])
    else:
        body = np.fliplr(stamp[:, :-1])

    assert body.size > 0
    return locator_side, TARGET_BY_COLOR[locator_color], body


def _placement(canvas_bbox: tuple[int, int, int, int], shape: tuple[int, int], target_side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = canvas_bbox
    canvas_h = r1 - r0 + 1
    canvas_w = c1 - c0 + 1
    ph, pw = shape
    assert 1 <= ph <= canvas_h and 1 <= pw <= canvas_w

    if target_side == "top":
        return r0, c0 + (canvas_w - pw) // 2
    if target_side == "bottom":
        return r1 - ph + 1, c0 + (canvas_w - pw) // 2
    if target_side == "left":
        return r0 + (canvas_h - ph) // 2, c0
    if target_side == "right":
        return r0 + (canvas_h - ph) // 2, c1 - pw + 1
    raise AssertionError("unknown target side")


def _overlay(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    window = dst[top : top + h, left : left + w]
    mask = patch != BLACK
    window[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    canvas_bbox = _find_canvas(g)
    r0, c0, r1, c1 = canvas_bbox

    out = np.full(g.shape, BLACK, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = GREY

    for comp in _components(g != BLACK):
        bb = _bbox(comp)
        if bb == canvas_bbox and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == GREY):
            continue
        sr0, sc0, sr1, sc1 = bb
        stamp = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
        _, target_side, body = _parse_stamp(stamp)
        pr, pc = _placement(canvas_bbox, body.shape, target_side)
        _overlay(out, body, pr, pc)

    return out


GEN_ATTEMPTS = 400
CANVAS_HEIGHT_RANGE = (9, 16)
CANVAS_WIDTH_RANGE = (9, 16)
STAMP_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
PATTERN_MAX_SIDE = 5
MARGIN_EXTRA_RANGE = (0, 5)
PATTERN_RETRIES = 80
SPEC_RETRIES = 120


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _mirror_pattern(pattern: np.ndarray, locator_side: str) -> np.ndarray:
    if locator_side in ("top", "bottom"):
        return np.flipud(pattern)
    return np.fliplr(pattern)


def _attach_band(pattern: np.ndarray, locator_side: str, color: int) -> np.ndarray:
    ph, pw = pattern.shape
    if locator_side == "top":
        return np.vstack([np.full((1, pw), color, dtype=int), pattern])
    if locator_side == "bottom":
        return np.vstack([pattern, np.full((1, pw), color, dtype=int)])
    if locator_side == "left":
        return np.hstack([np.full((ph, 1), color, dtype=int), pattern])
    if locator_side == "right":
        return np.hstack([pattern, np.full((ph, 1), color, dtype=int)])
    raise AssertionError("unknown locator side")


def _pattern_symmetric(pattern: np.ndarray, flip_axis: str) -> bool:
    if flip_axis == "ud":
        return np.array_equal(pattern, np.flipud(pattern))
    return np.array_equal(pattern, np.fliplr(pattern))


def _make_pattern(max_h: int, max_w: int, flip_axis: str) -> np.ndarray:
    assert max_h >= 2 and max_w >= 2
    modes = ("noise", "row_bands", "col_bands", "checker", "stripes_plus")

    for _ in range(PATTERN_RETRIES):
        ph = random.randint(2, max_h)
        pw = random.randint(2, max_w)
        palette = random.sample(BODY_COLORS, random.randint(2, min(4, len(BODY_COLORS))))
        k = len(palette)
        mode = random.choice(modes)
        arr = np.empty((ph, pw), dtype=int)

        if mode == "noise":
            for r in range(ph):
                for c in range(pw):
                    arr[r, c] = random.choice(palette)
        elif mode == "row_bands":
            offset = random.randrange(k)
            step = random.randint(1, k - 1)
            for r in range(ph):
                arr[r, :] = palette[(offset + step * r) % k]
            for _ in range(random.randint(1, max(1, pw))):
                arr[random.randrange(ph), random.randrange(pw)] = random.choice(palette)
        elif mode == "col_bands":
            offset = random.randrange(k)
            step = random.randint(1, k - 1)
            for c in range(pw):
                arr[:, c] = palette[(offset + step * c) % k]
            for _ in range(random.randint(1, max(1, ph))):
                arr[random.randrange(ph), random.randrange(pw)] = random.choice(palette)
        elif mode == "checker":
            row_step = random.randint(1, k - 1)
            col_step = random.randint(1, k - 1)
            offset = random.randrange(k)
            for r in range(ph):
                for c in range(pw):
                    arr[r, c] = palette[(offset + row_step * r + col_step * c) % k]
        else:
            base = random.choice(palette)
            alt = random.choice([color for color in palette if color != base])
            arr[:] = base
            if random.random() < 0.5:
                cols = sorted(random.sample(range(pw), random.randint(1, max(1, pw - 1))))
                arr[:, cols] = alt
            else:
                rows = sorted(random.sample(range(ph), random.randint(1, max(1, ph - 1))))
                arr[rows, :] = alt
            arr[random.randrange(ph), random.randrange(pw)] = random.choice([color for color in palette if color != arr[0, 0]])

        if len(set(map(int, arr.flat))) < 2:
            continue
        if _pattern_symmetric(arr, flip_axis):
            continue
        return arr

    raise RuntimeError("failed to build a non-symmetric pattern")


def _local_target_rect(canvas_h: int, canvas_w: int, pattern: np.ndarray, target_side: str) -> tuple[int, int, int, int]:
    top, left = _placement((0, 0, canvas_h - 1, canvas_w - 1), pattern.shape, target_side)
    ph, pw = pattern.shape
    return top, left, top + ph - 1, left + pw - 1


def _choose_specs(canvas_h: int, canvas_w: int) -> list[dict[str, object]] | None:
    targets = random.sample(list(SIDES), random.choice(STAMP_COUNT_CHOICES))
    max_h = min(PATTERN_MAX_SIDE, canvas_h - 1)
    max_w = min(PATTERN_MAX_SIDE, canvas_w - 1)
    if max_h < 2 or max_w < 2:
        return None

    for _ in range(SPEC_RETRIES):
        specs: list[dict[str, object]] = []
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for target in targets:
            placed = False
            for _ in range(PATTERN_RETRIES):
                locator = random.choice(SIDES)
                flip_axis = "ud" if locator in ("top", "bottom") else "lr"
                pattern = _make_pattern(max_h, max_w, flip_axis)
                rect = _local_target_rect(canvas_h, canvas_w, pattern, target)
                if any(_rect_overlap(rect, other) for other in rects):
                    continue
                specs.append(
                    {
                        "target": target,
                        "locator": locator,
                        "pattern": pattern,
                        "mirrored": _mirror_pattern(pattern, locator),
                        "patch": _attach_band(pattern, locator, COLOR_BY_TARGET[target]),
                    }
                )
                rects.append(rect)
                placed = True
                break
            if not placed:
                ok = False
                break
        if ok:
            return specs
    return None


def _mark_padded(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _zone_candidates(
    occupied: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    stamp_h: int,
    stamp_w: int,
    zone: str,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    r0, c0, r1, c1 = canvas_bbox

    if zone == "top":
        row_lo, row_hi = 0, r0 - stamp_h - 1
        col_lo, col_hi = 0, w - stamp_w
    elif zone == "bottom":
        row_lo, row_hi = r1 + 2, h - stamp_h
        col_lo, col_hi = 0, w - stamp_w
    elif zone == "left":
        row_lo, row_hi = 0, h - stamp_h
        col_lo, col_hi = 0, c0 - stamp_w - 1
    else:
        row_lo, row_hi = 0, h - stamp_h
        col_lo, col_hi = c1 + 2, w - stamp_w

    if row_hi < row_lo or col_hi < col_lo:
        return []

    candidates: list[tuple[int, int]] = []
    for top in range(row_lo, row_hi + 1):
        for left in range(col_lo, col_hi + 1):
            if not occupied[top : top + stamp_h, left : left + stamp_w].any():
                candidates.append((top, left))
    return candidates


def _choose_zones(count: int) -> list[str]:
    if count == 1:
        return [random.choice(SIDES)]
    if random.random() < 0.65:
        zones = random.sample(list(SIDES), min(4, count))
        while len(zones) < count:
            zones.append(random.choice(SIDES))
        random.shuffle(zones)
        return zones
    return [random.choice(SIDES) for _ in range(count)]


def _gen_attempt() -> np.ndarray | None:
    canvas_h = random.randint(*CANVAS_HEIGHT_RANGE)
    canvas_w = random.randint(*CANVAS_WIDTH_RANGE)
    specs = _choose_specs(canvas_h, canvas_w)
    if specs is None:
        return None

    max_stamp_h = max(int(spec["patch"].shape[0]) for spec in specs)
    max_stamp_w = max(int(spec["patch"].shape[1]) for spec in specs)
    top_margin = max_stamp_h + 2 + random.randint(*MARGIN_EXTRA_RANGE)
    bottom_margin = max_stamp_h + 2 + random.randint(*MARGIN_EXTRA_RANGE)
    left_margin = max_stamp_w + 2 + random.randint(*MARGIN_EXTRA_RANGE)
    right_margin = max_stamp_w + 2 + random.randint(*MARGIN_EXTRA_RANGE)

    grid_h = top_margin + canvas_h + bottom_margin
    grid_w = left_margin + canvas_w + right_margin
    canvas_bbox = (
        top_margin,
        left_margin,
        top_margin + canvas_h - 1,
        left_margin + canvas_w - 1,
    )

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    _mark_padded(occupied, canvas_bbox[0], canvas_bbox[1], canvas_h, canvas_w, pad=1)

    placements: list[tuple[int, int] | None] = [None] * len(specs)
    zones = _choose_zones(len(specs))
    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["patch"].shape[0]) * int(specs[idx]["patch"].shape[1]),
        reverse=True,
    )

    for idx in order:
        patch = specs[idx]["patch"]
        stamp_h, stamp_w = patch.shape
        candidates = _zone_candidates(occupied, canvas_bbox, stamp_h, stamp_w, zones[idx])
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        _mark_padded(occupied, top, left, stamp_h, stamp_w, pad=1)

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    r0, c0, r1, c1 = canvas_bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    expected = np.full_like(grid, BLACK)
    expected[r0 : r1 + 1, c0 : c1 + 1] = GREY

    for spec, placement in zip(specs, placements, strict=False):
        assert placement is not None
        patch = spec["patch"]
        top, left = placement
        grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch

        mirrored = spec["mirrored"]
        pr, pc = _placement(canvas_bbox, mirrored.shape, str(spec["target"]))
        _overlay(expected, mirrored, pr, pc)

    if len(_components(grid != BLACK)) != len(specs) + 1:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-canvas input")
