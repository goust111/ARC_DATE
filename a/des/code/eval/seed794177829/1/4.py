from __future__ import annotations

import random

import numpy as np


DIR_ORDER = ("top", "bottom", "left", "right")
ALL_COMBOS = [
    combo
    for mask in range(1, 1 << len(DIR_ORDER))
    for combo in [tuple(DIR_ORDER[i] for i in range(len(DIR_ORDER)) if mask & (1 << i))]
]

SYMBOL_MASKS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=int),
]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_signature(patch: np.ndarray, wall: int, bg: int) -> tuple[str, ...] | None:
    h, w = patch.shape
    if h < 3 or w < 3:
        return None
    if np.any(patch[1:-1, 1:-1] != bg):
        return None
    if not (
        int(patch[0, 0]) == wall
        and int(patch[0, -1]) == wall
        and int(patch[-1, 0]) == wall
        and int(patch[-1, -1]) == wall
    ):
        return None

    gaps: list[str] = []
    side_lines = {
        "top": patch[0, 1:-1],
        "bottom": patch[-1, 1:-1],
        "left": patch[1:-1, 0],
        "right": patch[1:-1, -1],
    }
    for name in DIR_ORDER:
        line = side_lines[name]
        if np.any((line != wall) & (line != bg)):
            return None
        missing = int(np.sum(line == bg))
        if missing > 1:
            return None
        if missing == 1:
            gaps.append(name)

    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border = patch[border_mask]
    if np.any((border != wall) & (border != bg)):
        return None
    return tuple(gaps)


def _collect_frames(grid: np.ndarray, wall: int, bg: int) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == wall):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        signature = _frame_signature(patch, wall, bg)
        if signature is None:
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gaps": signature,
                "shape": patch.shape,
                "area": int(coords.shape[0]),
            }
        )
    return frames


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    candidates: list[tuple[tuple[int, int], int]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        frames = _collect_frames(grid, color, bg)
        if not frames:
            continue
        score = (len(frames), sum(frame["area"] for frame in frames))
        candidates.append((score, color))
    assert candidates, "expected a wall color with rectangular frame components"
    candidates.sort()
    return int(candidates[-1][1])


def _is_adjacent_pair(frame_bbox: tuple[int, int, int, int], symbol_bbox: tuple[int, int, int, int]) -> bool:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = symbol_bbox
    if fr0 == sr0 and fr1 == sr1:
        if 0 <= sc0 - fc1 - 1 <= 1 or 0 <= fc0 - sc1 - 1 <= 1:
            return True
    if fc0 == sc0 and fc1 == sc1:
        if 0 <= sr0 - fr1 - 1 <= 1 or 0 <= fr0 - sr1 - 1 <= 1:
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend window-combinations from micro-frames and place the matching 3x3 symbols
    at the centers of larger rooms with the same wall-gap directions.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(np.bincount(g.ravel()).argmax())
    wall = _detect_wall_color(g, bg)
    frames = _collect_frames(g, wall, bg)

    micro_frames = [frame for frame in frames if frame["shape"] == (3, 3)]
    rooms = [frame for frame in frames if frame["shape"][0] >= 5 and frame["shape"][1] >= 5]
    assert micro_frames and rooms, "expected both legend frames and rooms"

    symbol_mask = (g != bg) & (g != wall)
    symbols: list[dict] = []
    for coords in _components(symbol_mask):
        r0, c0, r1, c1 = _bbox(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert patch.shape == (3, 3), "expected each legend symbol to occupy a 3x3 box"
        symbols.append({"bbox": (r0, c0, r1, c1), "patch": patch.copy()})

    legend_map: dict[tuple[str, ...], np.ndarray] = {}
    used_symbols: set[int] = set()
    for frame in micro_frames:
        matches = [i for i, sym in enumerate(symbols) if _is_adjacent_pair(frame["bbox"], sym["bbox"])]
        assert len(matches) == 1, "each micro-frame must pair with exactly one symbol"
        idx = int(matches[0])
        assert idx not in used_symbols, "each legend symbol should be used once"
        used_symbols.add(idx)
        combo = tuple(frame["gaps"])
        assert combo not in legend_map, "legend combinations must be unique"
        legend_map[combo] = symbols[idx]["patch"]

    out = g.copy()
    for room in rooms:
        combo = tuple(room["gaps"])
        assert combo in legend_map, "every room must have a matching legend entry"
        r0, c0, r1, c1 = room["bbox"]
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        patch = legend_map[combo]
        target = out[cr - 1 : cr + 2, cc - 1 : cc + 2]
        assert target.shape == (3, 3), "room center must fit a 3x3 symbol"
        assert r0 + 1 <= cr - 1 and cr + 1 <= r1 - 1
        assert c0 + 1 <= cc - 1 and cc + 1 <= c1 - 1
        mask = patch != bg
        target[mask] = patch[mask]
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _draw_room_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    wall: int,
    bg: int,
    combo: tuple[str, ...],
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall

    if "top" in combo:
        grid[r0, rng.randint(c0 + 1, c1 - 1)] = bg
    if "bottom" in combo:
        grid[r1, rng.randint(c0 + 1, c1 - 1)] = bg
    if "left" in combo:
        grid[rng.randint(r0 + 1, r1 - 1), c0] = bg
    if "right" in combo:
        grid[rng.randint(r0 + 1, r1 - 1), c1] = bg


def _draw_micro_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    wall: int,
    bg: int,
    combo: tuple[str, ...],
) -> None:
    grid[top : top + 3, left : left + 3] = bg
    grid[top, left : left + 3] = wall
    grid[top + 2, left : left + 3] = wall
    grid[top : top + 3, left] = wall
    grid[top : top + 3, left + 2] = wall
    if "top" in combo:
        grid[top, left + 1] = bg
    if "bottom" in combo:
        grid[top + 2, left + 1] = bg
    if "left" in combo:
        grid[top + 1, left] = bg
    if "right" in combo:
        grid[top + 1, left + 2] = bg


def _pair_patch(
    pair_mode: str,
    wall: int,
    bg: int,
    combo: tuple[str, ...],
    symbol_patch: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    if pair_mode in ("frame_left", "frame_right"):
        patch = np.full((3, 7), bg, dtype=int)
        if pair_mode == "frame_left":
            fr, fc = 0, 0
            sr, sc = 0, 4
        else:
            sr, sc = 0, 0
            fr, fc = 0, 4
    else:
        patch = np.full((7, 3), bg, dtype=int)
        if pair_mode == "frame_up":
            fr, fc = 0, 0
            sr, sc = 4, 0
        else:
            sr, sc = 0, 0
            fr, fc = 4, 0

    patch[sr : sr + 3, sc : sc + 3] = symbol_patch
    _draw_micro_frame(patch, fr, fc, wall, bg, combo)
    return patch, (fr, fc, fr + 2, fc + 2)


def _place_legend(
    grid: np.ndarray,
    wall: int,
    bg: int,
    entries: list[tuple[tuple[str, ...], np.ndarray]],
    pair_mode: str,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    spacing = 1
    pair_patches = [_pair_patch(pair_mode, wall, bg, combo, patch) for combo, patch in entries]

    if pair_mode in ("frame_left", "frame_right"):
        legend_h = len(entries) * 3 + (len(entries) - 1) * spacing
        legend_w = 7
        row_starts = [i * (3 + spacing) for i in range(len(entries))]
        col_starts = [0] * len(entries)
    else:
        legend_h = 7
        legend_w = len(entries) * 3 + (len(entries) - 1) * spacing
        row_starts = [0] * len(entries)
        col_starts = [i * (3 + spacing) for i in range(len(entries))]

    h, w = grid.shape
    corner = rng.choice(["top_left", "top_right", "bottom_left", "bottom_right"])
    base_r = 0 if "top" in corner else h - legend_h
    base_c = 0 if "left" in corner else w - legend_w

    for (pair_patch, _), r_off, c_off in zip(pair_patches, row_starts, col_starts, strict=False):
        ph, pw = pair_patch.shape
        grid[base_r + r_off : base_r + r_off + ph, base_c + c_off : base_c + c_off + pw] = pair_patch

    return (base_r, base_c, base_r + legend_h - 1, base_c + legend_w - 1)


def _symbol_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, color, bg).astype(int)


def _place_rooms(
    h: int,
    w: int,
    legend_box: tuple[int, int, int, int],
    n_rooms: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    sizes = [(rng.choice([5, 7, 9]), rng.choice([5, 7, 9])) for _ in range(n_rooms)]
    sizes.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    reserved = (max(0, legend_box[0] - 1), max(0, legend_box[1] - 1), min(h - 1, legend_box[2] + 1), min(w - 1, legend_box[3] + 1))
    for rh, rw in sizes:
        placed = False
        for _ in range(300):
            if h - rh - 1 < 1 or w - rw - 1 < 1:
                return None
            r0 = rng.randint(1, h - rh - 1)
            c0 = rng.randint(1, w - rw - 1)
            box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
            if _boxes_overlap(box, reserved, pad=0):
                continue
            if any(_boxes_overlap(box, other, pad=1) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    rng.shuffle(boxes)
    return boxes


def _build_expected_output(
    inp: np.ndarray,
    room_specs: list[tuple[tuple[int, int, int, int], tuple[str, ...]]],
    legend_lookup: dict[tuple[str, ...], np.ndarray],
    bg: int,
) -> np.ndarray:
    out = inp.copy()
    for bbox, combo in room_specs:
        r0, c0, r1, c1 = bbox
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        patch = legend_lookup[combo]
        region = out[cr - 1 : cr + 2, cc - 1 : cc + 2]
        region[patch != bg] = patch[patch != bg]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    m = rng.randint(3, 5)
    n_rooms = rng.randint(3, 5)

    wall = rng.choice([color for color in range(1, 10)])
    free_colors = [color for color in range(1, 10) if color != wall]
    rng.shuffle(free_colors)
    symbol_colors = free_colors[:m]
    symbol_masks = rng.sample(SYMBOL_MASKS, m)
    legend_combos = rng.sample(ALL_COMBOS, m)

    if n_rooms < 2:
        return None
    used_combo_count = rng.randint(2, min(m, n_rooms))
    used_combos = rng.sample(legend_combos, used_combo_count)
    room_combos = used_combos.copy()
    while len(room_combos) < n_rooms:
        room_combos.append(rng.choice(used_combos))
    rng.shuffle(room_combos)

    entries = [
        (combo, _symbol_patch(mask, color, 0))
        for combo, mask, color in zip(legend_combos, symbol_masks, symbol_colors, strict=False)
    ]
    legend_lookup = {combo: patch for combo, patch in entries}

    pair_mode = rng.choice(["frame_left", "frame_right", "frame_up", "frame_down"])
    if pair_mode in ("frame_left", "frame_right"):
        legend_h = m * 3 + (m - 1)
        legend_w = 7
    else:
        legend_h = 7
        legend_w = m * 3 + (m - 1)
    h = rng.randint(max(24, legend_h + 8), max(28, legend_h + 12))
    w = rng.randint(max(24, legend_w + 8), max(28, legend_w + 12))

    grid = np.zeros((h, w), dtype=int)
    legend_box = _place_legend(grid, wall, 0, entries, pair_mode, rng)
    room_boxes = _place_rooms(h, w, legend_box, n_rooms, rng)
    if room_boxes is None:
        return None

    room_specs = list(zip(room_boxes, room_combos, strict=False))
    for bbox, combo in room_specs:
        _draw_room_frame(grid, bbox, wall, 0, combo, rng)

    out = _build_expected_output(grid, room_specs, legend_lookup, 0)
    try:
        if np.array_equal(main(grid), out):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
