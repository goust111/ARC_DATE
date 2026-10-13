from __future__ import annotations

import numpy as np


BG = 0
NONZERO_COLORS = tuple(range(1, 10))
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _extract_frame(grid: np.ndarray, color: int) -> dict | None:
    pts = np.argwhere(grid == color)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = sub == int(color)
    perimeter = np.zeros_like(mask, dtype=bool)
    perimeter[0, :] = True
    perimeter[-1, :] = True
    perimeter[:, 0] = True
    perimeter[:, -1] = True

    if bool(np.any(mask & ~perimeter)):
        return None
    if not bool(np.all(sub[1:-1, 1:-1] == BG)):
        return None
    if not bool(mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    sides = {
        "up": bool(np.all(mask[0, 1:-1])),
        "down": bool(np.all(mask[-1, 1:-1])),
        "left": bool(np.all(mask[1:-1, 0])),
        "right": bool(np.all(mask[1:-1, -1])),
    }
    if sum(int(v) for v in sides.values()) != 3:
        return None
    opening = next(side for side, present in sides.items() if not present)

    return {
        "color": int(color),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "inner_h": r1 - r0 - 1,
        "inner_w": c1 - c0 - 1,
        "opening": opening,
        "sides": sides,
    }


def _extract_frames(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BG]
    frames = []
    for color in colors:
        frame = _extract_frame(grid, color)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda f: (f["r0"], f["c0"], f["color"]))
    return frames


def _strip_info(frame: dict) -> dict:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    opening = frame["opening"]
    if opening == "up":
        return {
            "direction": "up",
            "row_lo": r0 - 1,
            "row_hi": r0 - 1,
            "col_lo": c0 + 1,
            "col_hi": c1 - 1,
        }
    if opening == "down":
        return {
            "direction": "down",
            "row_lo": r1 + 1,
            "row_hi": r1 + 1,
            "col_lo": c0 + 1,
            "col_hi": c1 - 1,
        }
    if opening == "left":
        return {
            "direction": "left",
            "row_lo": r0 + 1,
            "row_hi": r1 - 1,
            "col_lo": c0 - 1,
            "col_hi": c0 - 1,
        }
    return {
        "direction": "right",
        "row_lo": r0 + 1,
        "row_hi": r1 - 1,
        "col_lo": c1 + 1,
        "col_hi": c1 + 1,
    }


def _find_target(frame: dict, frames: list[dict], height: int, width: int) -> dict | None:
    strip = _strip_info(frame)
    row_lo, row_hi = strip["row_lo"], strip["row_hi"]
    col_lo, col_hi = strip["col_lo"], strip["col_hi"]
    direction = strip["direction"]

    if direction in ("left", "right") and not (0 <= row_lo <= row_hi < height):
        return None
    if direction in ("up", "down") and not (0 <= col_lo <= col_hi < width):
        return None
    if direction == "left" and col_lo < 0:
        return None
    if direction == "right" and col_lo >= width:
        return None
    if direction == "up" and row_lo < 0:
        return None
    if direction == "down" and row_lo >= height:
        return None

    candidates: list[tuple[int, int, dict]] = []
    for other in frames:
        if other is frame:
            continue
        if direction == "right":
            if not other["sides"]["left"]:
                continue
            if not (other["r0"] + 1 <= row_lo and row_hi <= other["r1"] - 1):
                continue
            if other["c0"] <= frame["c1"]:
                continue
            candidates.append((other["c0"], other["r0"], other))
        elif direction == "left":
            if not other["sides"]["right"]:
                continue
            if not (other["r0"] + 1 <= row_lo and row_hi <= other["r1"] - 1):
                continue
            if other["c1"] >= frame["c0"]:
                continue
            candidates.append((-other["c1"], other["r0"], other))
        elif direction == "down":
            if not other["sides"]["up"]:
                continue
            if not (other["c0"] + 1 <= col_lo and col_hi <= other["c1"] - 1):
                continue
            if other["r0"] <= frame["r1"]:
                continue
            candidates.append((other["r0"], other["c0"], other))
        else:
            if not other["sides"]["down"]:
                continue
            if not (other["c0"] + 1 <= col_lo and col_hi <= other["c1"] - 1):
                continue
            if other["r1"] >= frame["r0"]:
                continue
            candidates.append((-other["r1"], other["c0"], other))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2]["color"]))
    return candidates[0][2]


def _count_crossings(frame: dict, target: dict, frames: list[dict]) -> int:
    strip = _strip_info(frame)
    direction = strip["direction"]
    row_lo, row_hi = strip["row_lo"], strip["row_hi"]
    col_lo, col_hi = strip["col_lo"], strip["col_hi"]

    if direction == "right":
        path_col_lo, path_col_hi = frame["c1"] + 1, target["c0"] - 1
        path_row_lo, path_row_hi = row_lo, row_hi
    elif direction == "left":
        path_col_lo, path_col_hi = target["c1"] + 1, frame["c0"] - 1
        path_row_lo, path_row_hi = row_lo, row_hi
    elif direction == "down":
        path_row_lo, path_row_hi = frame["r1"] + 1, target["r0"] - 1
        path_col_lo, path_col_hi = col_lo, col_hi
    else:
        path_row_lo, path_row_hi = target["r1"] + 1, frame["r0"] - 1
        path_col_lo, path_col_hi = col_lo, col_hi

    if path_row_lo > path_row_hi or path_col_lo > path_col_hi:
        return 0

    count = 0
    for other in frames:
        if other is frame or other is target:
            continue
        r0, c0, r1, c1 = other["r0"], other["c0"], other["r1"], other["c1"]
        if other["sides"]["up"]:
            if path_row_lo <= r0 <= path_row_hi and _intervals_overlap(c0 + 1, c1 - 1, path_col_lo, path_col_hi):
                count += 1
        if other["sides"]["down"]:
            if path_row_lo <= r1 <= path_row_hi and _intervals_overlap(c0 + 1, c1 - 1, path_col_lo, path_col_hi):
                count += 1
        if other["sides"]["left"]:
            if path_col_lo <= c0 <= path_col_hi and _intervals_overlap(r0 + 1, r1 - 1, path_row_lo, path_row_hi):
                count += 1
        if other["sides"]["right"]:
            if path_col_lo <= c1 <= path_col_hi and _intervals_overlap(r0 + 1, r1 - 1, path_row_lo, path_row_hi):
                count += 1
    return count


def _paint_projection(out: np.ndarray, source: dict, target: dict, length: int) -> None:
    if source["opening"] == "right":
        span = target["inner_h"]
        if length > span:
            return
        start = target["r0"] + 1 + (span - length) // 2
        out[start : start + length, target["c0"] + 1] = source["color"]
    elif source["opening"] == "left":
        span = target["inner_h"]
        if length > span:
            return
        start = target["r0"] + 1 + (span - length) // 2
        out[start : start + length, target["c1"] - 1] = source["color"]
    elif source["opening"] == "down":
        span = target["inner_w"]
        if length > span:
            return
        start = target["c0"] + 1 + (span - length) // 2
        out[target["r0"] + 1, start : start + length] = source["color"]
    else:
        span = target["inner_w"]
        if length > span:
            return
        start = target["c0"] + 1 + (span - length) // 2
        out[target["r1"] - 1, start : start + length] = source["color"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Colored hollow frames project through their openings toward another frame.

    Each frame is a one-cell-thick rectangle with exactly one missing side.
    A ray starts from the opening, travels in the opening direction, counts how
    many solid frame edges it crosses on the way, and paints a centered segment
    of length K+1 on the hit frame's corresponding inner wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    frames = _extract_frames(g)
    out = g.copy()

    for frame in frames:
        target = _find_target(frame, frames, h, w)
        if target is None:
            continue
        crossings = _count_crossings(frame, target, frames)
        _paint_projection(out, frame, target, crossings + 1)

    return out


MAX_ATTEMPTS = 400
BAND_PITCH = 9
BAND_CENTER = 4
INNER_W_CHOICES = (1, 2, 3)
GAP_CHOICES = (2, 3, 4)


def _sample_top(rng: np.random.Generator, band_top: int, y: int, inner_h: int) -> int:
    height = inner_h + 2
    band_bottom = band_top + BAND_PITCH - 1
    low = max(band_top, y - inner_h)
    high = min(band_bottom - height + 1, y - 1)
    assert low <= high
    return int(rng.integers(low, high + 1))


def _make_frame(color: int, top: int, left: int, inner_h: int, inner_w: int, opening: str) -> dict:
    return {
        "color": int(color),
        "r0": int(top),
        "c0": int(left),
        "r1": int(top + inner_h + 1),
        "c1": int(left + inner_w + 1),
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "opening": opening,
        "sides": {
            "up": opening != "up",
            "down": opening != "down",
            "left": opening != "left",
            "right": opening != "right",
        },
    }


def _paint_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
    color = frame["color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if frame["opening"] == "up":
        grid[r0, c0 + 1 : c1] = BG
    elif frame["opening"] == "down":
        grid[r1, c0 + 1 : c1] = BG
    elif frame["opening"] == "left":
        grid[r0 + 1 : r1, c0] = BG
    else:
        grid[r0 + 1 : r1, c1] = BG


def _sample_corridor_specs(rng: np.random.Generator) -> list[dict]:
    while True:
        num_corridors = int(rng.integers(1, 4))
        specs = []
        has_hit = False
        total_boxes = 0
        for _ in range(num_corridors):
            hit = bool(rng.integers(0, 2))
            k = int(rng.integers(0, 4))
            specs.append({"hit": hit, "k": k})
            has_hit = has_hit or hit
            total_boxes += 1 + k + int(hit)
        if not has_hit:
            idx = int(rng.integers(0, num_corridors))
            specs[idx]["hit"] = True
            total_boxes += 1
        if total_boxes <= len(NONZERO_COLORS):
            return specs


def _build_base_scene(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    specs = _sample_corridor_specs(rng)
    total_boxes = sum(1 + spec["k"] + int(spec["hit"]) for spec in specs)
    colors = [int(c) for c in rng.choice(np.array(NONZERO_COLORS, dtype=int), size=total_boxes, replace=False)]
    color_idx = 0

    frames: list[dict] = []
    projections: list[tuple[dict, dict, int]] = []
    max_col = 0

    for corridor_idx, spec in enumerate(specs):
        band_top = corridor_idx * BAND_PITCH
        y = band_top + BAND_CENTER
        x = int(rng.integers(1, 4))

        source_inner_w = int(rng.choice(np.array(INNER_W_CHOICES, dtype=int)))
        source = _make_frame(colors[color_idx], y - 1, x, 1, source_inner_w, "right")
        color_idx += 1
        frames.append(source)
        x = source["c1"] + 1 + int(rng.choice(np.array(GAP_CHOICES, dtype=int)))

        if spec["k"] > 0:
            height_seq = sorted(int(v) for v in rng.choice(np.array([1, 2, 3], dtype=int), size=spec["k"], replace=False))
        else:
            height_seq = []

        crossers = []
        for inner_h in height_seq:
            inner_w = int(rng.choice(np.array(INNER_W_CHOICES, dtype=int)))
            top = _sample_top(rng, band_top, y, inner_h)
            crosser = _make_frame(colors[color_idx], top, x, inner_h, inner_w, "left")
            color_idx += 1
            crossers.append(crosser)
            frames.append(crosser)
            x = crosser["c1"] + 1 + int(rng.choice(np.array(GAP_CHOICES, dtype=int)))

        if spec["hit"]:
            proj_len = spec["k"] + 1
            extra = 0 if proj_len >= 4 else int(rng.integers(0, 2))
            target_inner_h = min(4, proj_len + extra)
            target_inner_w = int(rng.choice(np.array(INNER_W_CHOICES, dtype=int)))
            top = _sample_top(rng, band_top, y, target_inner_h)
            target = _make_frame(colors[color_idx], top, x, target_inner_h, target_inner_w, "right")
            color_idx += 1
            frames.append(target)
            projections.append((source, target, proj_len))

        for frame in frames:
            max_col = max(max_col, frame["c1"])

    rows = len(specs) * BAND_PITCH
    cols = max_col + 1 + int(rng.integers(2, 6))
    inp = np.full((rows, cols), BG, dtype=int)
    for frame in frames:
        _paint_frame(inp, frame)

    out = inp.copy()
    for source, target, length in projections:
        _paint_projection(out, source, target, length)

    return inp, out


def _pad(grid: np.ndarray, pads: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pads
    return np.pad(grid, ((top, bottom), (left, right)), mode="constant", constant_values=BG)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        base_inp, base_out = _build_base_scene(rng)
        rot_k = int(rng.integers(0, 4))
        pads = tuple(int(v) for v in rng.integers(0, 4, size=4))

        inp = _pad(np.rot90(base_inp, rot_k), pads)
        expected = _pad(np.rot90(base_out, rot_k), pads)

        if not np.any(inp != expected):
            continue
        if len(_extract_frames(inp)) != len(np.unique(inp)) - 1:
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp

    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
