from __future__ import annotations

import numpy as np


BG = 0
SIDES = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_RNG = np.random.default_rng()
_GEN_CALLS = 0

GRID_HEIGHT_RANGE = (18, 31)
GRID_WIDTH_RANGE = (18, 31)
FRAME_COUNT_RANGE = (1, 5)
INTERIOR_HEIGHT_RANGE = (4, 10)
INTERIOR_WIDTH_RANGE = (4, 10)
GAP_LENGTHS = (2, 3)
OBJECT_SPAN_MAX = 3
PLACEMENT_ATTEMPTS = 160
GENERATION_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component(grid: np.ndarray, seen: np.ndarray, start_r: int, start_c: int) -> dict:
    color = int(grid[start_r, start_c])
    stack = [(int(start_r), int(start_c))]
    seen[start_r, start_c] = True
    coords: list[tuple[int, int]] = []
    height, width = grid.shape

    while stack:
        r, c = stack.pop()
        coords.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = r + dr
            cc = c + dc
            if 0 <= rr < height and 0 <= cc < width and not seen[rr, cc] and int(grid[rr, cc]) == color:
                seen[rr, cc] = True
                stack.append((rr, cc))

    arr = np.array(coords, dtype=int)
    return {"color": color, "coords": arr, "bbox": _bbox(arr)}


def _color_components(grid: np.ndarray) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    comps: list[dict] = []
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) == BG or seen[r, c]:
                continue
            comps.append(_component(grid, seen, r, c))
    return comps


def _frame_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[comp["coords"][:, 0] - r0, comp["coords"][:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(local & ~border):
        return None

    missing = border & ~local
    if int(missing.sum()) not in GAP_LENGTHS:
        return None
    if not np.array_equal(local, border & ~missing):
        return None

    holes = np.argwhere(missing)
    if np.all(holes[:, 0] == 0):
        side = "up"
        span = np.sort(holes[:, 1])
        if int(span[0]) == 0 or int(span[-1]) == w - 1:
            return None
    elif np.all(holes[:, 0] == h - 1):
        side = "down"
        span = np.sort(holes[:, 1])
        if int(span[0]) == 0 or int(span[-1]) == w - 1:
            return None
    elif np.all(holes[:, 1] == 0):
        side = "left"
        span = np.sort(holes[:, 0])
        if int(span[0]) == 0 or int(span[-1]) == h - 1:
            return None
    elif np.all(holes[:, 1] == w - 1):
        side = "right"
        span = np.sort(holes[:, 0])
        if int(span[0]) == 0 or int(span[-1]) == h - 1:
            return None
    else:
        return None

    if not np.all(np.diff(span) == 1):
        return None

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "side": side,
        "gap_length": int(len(span)),
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _color_components(grid):
        info = _frame_info(comp)
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find open rectangular frames and eject each interior object through the gap.

    The gap side defines the translation direction and the object's pixel area
    defines the translation distance.
    """
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    out = g.copy()

    frames = _find_frames(g)
    frame_mask = np.zeros((height, width), dtype=bool)
    for frame in frames:
        frame_mask[frame["coords"][:, 0], frame["coords"][:, 1]] = True

    clear_mask = np.zeros((height, width), dtype=bool)
    moves: list[tuple[np.ndarray, int]] = []

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        obj_mask = (interior != BG) & (interior != int(frame["color"]))
        if not np.any(obj_mask):
            continue

        obj_colors = np.unique(interior[obj_mask])
        obj_color = int(obj_colors[0])
        obj_coords = np.argwhere(obj_mask) + np.array([r0 + 1, c0 + 1], dtype=int)
        clear_mask[obj_coords[:, 0], obj_coords[:, 1]] = True

        distance = int(obj_coords.shape[0])
        dr, dc = DIR_VECTORS[frame["side"]]
        shifted = obj_coords + np.array([dr * distance, dc * distance], dtype=int)
        moves.append((shifted, obj_color))

    out[clear_mask] = BG
    for shifted, color in moves:
        keep = (
            (shifted[:, 0] >= 0)
            & (shifted[:, 0] < height)
            & (shifted[:, 1] >= 0)
            & (shifted[:, 1] < width)
        )
        for r, c in shifted[keep]:
            if not frame_mask[int(r), int(c)]:
                out[int(r), int(c)] = int(color)

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(height, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(width, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _object_shapes(side: str, gap_len: int) -> list[tuple[int, int]]:
    shapes = []
    if side in ("up", "down"):
        for h in range(1, OBJECT_SPAN_MAX + 1):
            for w in range(1, gap_len + 1):
                if h * w >= 2:
                    shapes.append((h, w))
    else:
        for h in range(1, gap_len + 1):
            for w in range(1, OBJECT_SPAN_MAX + 1):
                if h * w >= 2:
                    shapes.append((h, w))
    return shapes


def _make_frame_patch(fh: int, fw: int, side: str, gap_start: int, gap_len: int) -> np.ndarray:
    patch = np.zeros((fh, fw), dtype=bool)
    patch[0, :] = True
    patch[-1, :] = True
    patch[:, 0] = True
    patch[:, -1] = True

    if side == "up":
        patch[0, gap_start + 1 : gap_start + 1 + gap_len] = False
    elif side == "down":
        patch[-1, gap_start + 1 : gap_start + 1 + gap_len] = False
    elif side == "left":
        patch[gap_start + 1 : gap_start + 1 + gap_len, 0] = False
    else:
        patch[gap_start + 1 : gap_start + 1 + gap_len, -1] = False

    return patch


def _sample_candidate(
    height: int,
    width: int,
    side: str,
    want_clip: bool,
    rng: np.random.Generator,
) -> dict | None:
    gap_len = int(rng.choice(GAP_LENGTHS))
    shapes = _object_shapes(side, gap_len)
    obj_h, obj_w = shapes[_randint(rng, 0, len(shapes) - 1)]
    area = int(obj_h * obj_w)

    ih_min = max(INTERIOR_HEIGHT_RANGE[0], obj_h + 1, gap_len if side in ("left", "right") else 0)
    ih_max = min(INTERIOR_HEIGHT_RANGE[1], height - 2)
    iw_min = max(INTERIOR_WIDTH_RANGE[0], obj_w, gap_len if side in ("up", "down") else 0)
    iw_max = min(INTERIOR_WIDTH_RANGE[1], width - 2)
    if ih_min > ih_max or iw_min > iw_max:
        return None

    ih = _randint(rng, ih_min, ih_max)
    iw = _randint(rng, iw_min, iw_max)
    fh = ih + 2
    fw = iw + 2

    if side in ("up", "down"):
        gap_start = _randint(rng, 0, iw - gap_len)
        ocol = _randint(rng, gap_start, gap_start + gap_len - obj_w)
        if side == "up":
            max_row = min(ih - obj_h, area - 2 if want_clip else area - 1)
            if max_row < 0:
                return None
            orow = _randint(rng, 0, max_row)
        else:
            min_row = max(0, ih - obj_h - (area - 1))
            if want_clip:
                min_row = max(min_row, ih - obj_h - (area - 2))
            if min_row > ih - obj_h:
                return None
            orow = _randint(rng, min_row, ih - obj_h)
    else:
        gap_start = _randint(rng, 0, ih - gap_len)
        orow = _randint(rng, gap_start, gap_start + gap_len - obj_h)
        if side == "left":
            max_col = min(iw - obj_w, area - 2 if want_clip else area - 1)
            if max_col < 0:
                return None
            ocol = _randint(rng, 0, max_col)
        else:
            min_col = max(0, iw - obj_w - (area - 1))
            if want_clip:
                min_col = max(min_col, iw - obj_w - (area - 2))
            if min_col > iw - obj_w:
                return None
            ocol = _randint(rng, min_col, iw - obj_w)

    obj_local = np.array(
        [(1 + orow + rr, 1 + ocol + cc) for rr in range(obj_h) for cc in range(obj_w)],
        dtype=int,
    )
    dr, dc = DIR_VECTORS[side]
    shifted_local = obj_local + np.array([dr * area, dc * area], dtype=int)

    fr_lo, fr_hi = 0, height - fh
    fc_lo, fc_hi = 0, width - fw

    if side == "up":
        top_row = int(shifted_local[:, 0].min())
        if want_clip:
            fr_hi = min(fr_hi, -top_row - 1)
        else:
            fr_lo = max(fr_lo, -top_row)
    elif side == "down":
        bottom_row = int(shifted_local[:, 0].max())
        if want_clip:
            fr_lo = max(fr_lo, height - bottom_row)
        else:
            fr_hi = min(fr_hi, height - bottom_row - 1)
    elif side == "left":
        left_col = int(shifted_local[:, 1].min())
        if want_clip:
            fc_hi = min(fc_hi, -left_col - 1)
        else:
            fc_lo = max(fc_lo, -left_col)
    else:
        right_col = int(shifted_local[:, 1].max())
        if want_clip:
            fc_lo = max(fc_lo, width - right_col)
        else:
            fc_hi = min(fc_hi, width - right_col - 1)

    if fr_lo > fr_hi or fc_lo > fc_hi:
        return None

    fr = _randint(rng, fr_lo, fr_hi)
    fc = _randint(rng, fc_lo, fc_hi)

    frame_color = _randint(rng, 1, 9)
    object_choices = [c for c in range(1, 10) if c != frame_color]
    object_color = int(rng.choice(object_choices))

    frame_patch = _make_frame_patch(fh, fw, side, gap_start, gap_len)
    frame_mask = np.zeros((height, width), dtype=bool)
    frame_mask[fr : fr + fh, fc : fc + fw] = frame_patch

    object_abs = obj_local + np.array([fr, fc], dtype=int)
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[object_abs[:, 0], object_abs[:, 1]] = True

    shifted_abs = shifted_local + np.array([fr, fc], dtype=int)
    keep = (
        (shifted_abs[:, 0] >= 0)
        & (shifted_abs[:, 0] < height)
        & (shifted_abs[:, 1] >= 0)
        & (shifted_abs[:, 1] < width)
    )
    if not np.any(keep):
        return None

    shifted_mask = np.zeros((height, width), dtype=bool)
    kept_shifted = shifted_abs[keep]
    shifted_mask[kept_shifted[:, 0], kept_shifted[:, 1]] = True

    return {
        "side": side,
        "want_clip": want_clip,
        "frame_color": frame_color,
        "object_color": object_color,
        "frame_mask": frame_mask,
        "object_mask": object_mask,
        "shifted_mask": shifted_mask,
    }


def _valid_generated_grid(grid: np.ndarray, expected_frames: int) -> bool:
    frames = _find_frames(grid)
    if len(frames) != expected_frames:
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        obj_colors = set(map(int, np.unique(interior))) - {BG}
        if len(obj_colors) != 1:
            return False
    return True


def _gen_attempt(call_index: int) -> np.ndarray | None:
    height = _randint(_RNG, GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1] - 1)
    width = _randint(_RNG, GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1] - 1)
    n_frames = _randint(_RNG, FRAME_COUNT_RANGE[0], FRAME_COUNT_RANGE[1] - 1)

    forced_side = SIDES[call_index % len(SIDES)]
    forced_clip = (call_index % (2 * len(SIDES))) < len(SIDES)

    side_list = [forced_side]
    for _ in range(n_frames - 1):
        side_list.append(str(_RNG.choice(SIDES)))
    _RNG.shuffle(side_list)
    side_list[0] = forced_side

    input_blocked = np.zeros((height, width), dtype=bool)
    output_blocked = np.zeros((height, width), dtype=bool)
    candidates: list[dict] = []

    for idx in range(n_frames):
        placed = None
        side = side_list[idx]
        want_clip = bool(forced_clip and idx == 0)
        for _ in range(PLACEMENT_ATTEMPTS):
            cand = _sample_candidate(height, width, side, want_clip, _RNG)
            if cand is None:
                continue
            input_mask = cand["frame_mask"] | cand["object_mask"]
            output_mask = cand["frame_mask"] | cand["shifted_mask"]
            if np.any(input_mask & input_blocked):
                continue
            if np.any(output_mask & output_blocked):
                continue

            candidates.append(cand)
            input_blocked |= _dilate(input_mask)
            output_blocked |= _dilate(output_mask)
            placed = cand
            break

        if placed is None:
            return None

    grid = np.zeros((height, width), dtype=int)
    for cand in candidates:
        grid[cand["frame_mask"]] = int(cand["frame_color"])
    for cand in candidates:
        grid[cand["object_mask"]] = int(cand["object_color"])

    if not _valid_generated_grid(grid, len(candidates)):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS

    call_index = _GEN_CALLS
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            _GEN_CALLS += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
