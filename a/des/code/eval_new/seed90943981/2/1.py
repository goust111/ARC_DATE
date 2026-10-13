from __future__ import annotations

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(values))


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == color, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            arr = np.array(coords, dtype=int)
            r1, c1 = arr.min(axis=0)
            r2, c2 = arr.max(axis=0)
            comps.append(
                {
                    "coords": arr,
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _mask_is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    start = tuple(int(v) for v in pts[0])
    stack = [start]
    seen[start] = True
    count = 0
    while stack:
        r, c = stack.pop()
        count += 1
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((int(nr), int(nc)))
    return count == int(mask.sum())


def _frame_from_component(grid: np.ndarray, bg: int, color: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 4 or w < 4:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))

    coords = {(int(r), int(c)) for r, c in comp["coords"]}
    if len(coords) != len(border) - 1:
        return None
    if coords - border:
        return None

    missing = list(border - coords)
    if len(missing) != 1:
        return None
    gap_r, gap_c = map(int, missing[0])
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if (gap_r, gap_c) in corners:
        return None

    if gap_r == r0:
        gap_side = "U"
    elif gap_r == r1:
        gap_side = "D"
    elif gap_c == c0:
        gap_side = "L"
    else:
        gap_side = "R"

    if r1 - r0 <= 1 or c1 - c0 <= 1:
        return None
    inner = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    if inner.size == 0 or np.any(inner == color):
        return None

    pattern_mask = inner != bg
    if not pattern_mask.any() or not _mask_is_connected(pattern_mask):
        return None
    if int(pattern_mask.sum()) < 2 or int(pattern_mask.sum()) > 4:
        return None

    pattern_colors = {int(v) for v in inner[pattern_mask]}
    if len(pattern_colors) != 1:
        return None
    pattern_color = int(next(iter(pattern_colors)))
    rel = np.argwhere(pattern_mask)
    pattern_cells = [(int(r0 + 1 + rr), int(c0 + 1 + cc)) for rr, cc in rel]

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_side": gap_side,
        "pattern_color": pattern_color,
        "pattern_cells": pattern_cells,
        "area": int(comp["area"]),
    }


def _scan_key_line(line: np.ndarray, color: int, bg: int) -> list[tuple[int, int, int]]:
    line = np.asarray(line, dtype=int)
    n = int(line.shape[0])
    all_color = {int(i) for i in np.flatnonzero(line == color)}
    candidates: list[tuple[int, int, int]] = []

    for length in range(3, 6):
        for start in range(0, n - length + 1):
            span = line[start : start + length]
            if int(np.count_nonzero(span == color)) != length - 1:
                continue
            if int(np.count_nonzero(span == bg)) != 1:
                continue
            if not np.all((span == color) | (span == bg)):
                continue
            if start > 0 and int(line[start - 1]) == color:
                continue
            if start + length < n and int(line[start + length]) == color:
                continue
            gap_idx = int(np.where(span == bg)[0][0])
            span_color = {start + i for i in range(length) if i != gap_idx}
            if span_color == all_color:
                candidates.append((int(start), int(length), int(gap_idx)))

    return candidates


def _key_direction(orientation: str, gap_idx: int, length: int) -> str:
    mid = int(length // 2)
    if gap_idx == mid:
        return "U" if orientation == "H" else "L"
    if orientation == "H":
        return "L" if gap_idx < mid else "R"
    return "U" if gap_idx < mid else "D"


def _detect_key(grid: np.ndarray, bg: int, color: int) -> dict | None:
    h, w = grid.shape
    candidates: list[dict] = []

    top = _scan_key_line(grid[0, :], color, bg)
    bottom = _scan_key_line(grid[h - 1, :], color, bg)
    left = _scan_key_line(grid[:, 0], color, bg)
    right = _scan_key_line(grid[:, w - 1], color, bg)

    for start, length, gap_idx in top:
        cells = [(0, start + i) for i in range(length)]
        candidates.append(
            {
                "orientation": "H",
                "cells": cells,
                "direction": _key_direction("H", gap_idx, length),
            }
        )
    for start, length, gap_idx in bottom:
        cells = [(h - 1, start + i) for i in range(length)]
        candidates.append(
            {
                "orientation": "H",
                "cells": cells,
                "direction": _key_direction("H", gap_idx, length),
            }
        )
    for start, length, gap_idx in left:
        cells = [(start + i, 0) for i in range(length)]
        candidates.append(
            {
                "orientation": "V",
                "cells": cells,
                "direction": _key_direction("V", gap_idx, length),
            }
        )
    for start, length, gap_idx in right:
        cells = [(start + i, w - 1) for i in range(length)]
        candidates.append(
            {
                "orientation": "V",
                "cells": cells,
                "direction": _key_direction("V", gap_idx, length),
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _shift_pattern(pattern_cells: list[tuple[int, int]], direction: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    pts = np.array(pattern_cells, dtype=int)
    rows = pts[:, 0]
    cols = pts[:, 1]
    r0, c0, r1, c1 = map(int, bbox)

    dr = 0
    dc = 0
    if direction == "L":
        dc = int(c0 - 1 - cols.max())
    elif direction == "R":
        dc = int(c1 + 1 - cols.min())
    elif direction == "U":
        dr = int(r0 - 1 - rows.max())
    else:
        dr = int(r1 + 1 - rows.min())

    return [(int(r + dr), int(c + dc)) for r, c in pts]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each edge key's notch position to unlock its same-color frame and push the interior pattern out through that side.

    Input structure:
    - bg is the most common color.
    - Several monochrome hollow rectangular frames appear away from the border; each frame has exactly one 1-cell gap on one side.
    - Each frame contains one small 2-4-cell monochrome pattern of a different color.
    - The grid border also contains thin horizontal or vertical same-color key bars; each key has exactly one bg gap and matches one frame by color.

    Rule:
    - Pair every key with the unique frame of the same color.
    - Decode the push direction from the key gap: horizontal key with gap at left/right end opens left/right, horizontal key with centered gap opens up; vertical key with gap at top/bottom end opens up/down, vertical key with centered gap opens left.
    - Erase the key, clear the pattern from inside the frame, and translate that whole pattern straight outward until it lies fully outside the frame and touches the chosen outer wall.
    - Keep the frame border unchanged, including its original gap.

    Key insight:
    - The key's own notch does not move anything directly; it encodes which side of the matching frame becomes the exit side for the interior sprite.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = _get_bg(g)

    frames: list[tuple[dict, dict | None]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        frame_infos = []
        for comp in _components_of_color(g, color):
            info = _frame_from_component(g, bg, color, comp)
            if info is not None:
                frame_infos.append(info)
        if not frame_infos:
            continue
        frame = max(frame_infos, key=lambda d: d["area"])
        key = _detect_key(g, bg, color)
        frames.append((frame, key))

    for frame, key in frames:
        direction = frame["gap_side"] if key is None else key["direction"]

        if key is not None:
            for r, c in key["cells"]:
                out[int(r), int(c)] = bg

        for r, c in frame["pattern_cells"]:
            out[int(r), int(c)] = bg

        moved = _shift_pattern(frame["pattern_cells"], direction, frame["bbox"])
        for r, c in moved:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[int(r), int(c)] = int(frame["pattern_color"])

    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 400
ALL_DIRECTIONS = ("L", "U", "R", "D")


def _grow_connected_cells(rng: np.random.Generator, h: int, w: int, area: int) -> set[tuple[int, int]]:
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return cells


def _sample_pattern_mask(rng: np.random.Generator) -> np.ndarray:
    for _ in range(100):
        area = int(rng.integers(2, 5))
        dims = [(h, w) for h in range(1, 4) for w in range(1, 4) if h * w >= area and (h, w) != (1, 1)]
        h, w = dims[int(rng.integers(0, len(dims)))]
        cells = _grow_connected_cells(rng, h, w, area)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[int(r - r0), int(c - c0)] = True
        if mask.shape[0] > 3 or mask.shape[1] > 3:
            continue
        if mask.all() and int(mask.sum()) >= 3 and rng.random() < 0.75:
            continue
        return mask
    return np.array([[True, True]], dtype=bool)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    sub = grid[top : top + h, left : left + w]
    sub[mask] = int(color)


def _paint_frame(
    grid: np.ndarray,
    *,
    top: int,
    left: int,
    h: int,
    w: int,
    color: int,
    bg: int,
    gap_side: str,
    gap_offset: int,
) -> None:
    grid[top, left : left + w] = int(color)
    grid[top + h - 1, left : left + w] = int(color)
    grid[top : top + h, left] = int(color)
    grid[top : top + h, left + w - 1] = int(color)
    if gap_side == "U":
        grid[top, left + gap_offset] = int(bg)
    elif gap_side == "D":
        grid[top + h - 1, left + gap_offset] = int(bg)
    elif gap_side == "L":
        grid[top + gap_offset, left] = int(bg)
    else:
        grid[top + gap_offset, left + w - 1] = int(bg)


def _box_clear(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 0) -> bool:
    h, w = mask.shape
    r0, c0, r1, c1 = map(int, box)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_box(mask: np.ndarray, box: tuple[int, int, int, int], pad: int = 0) -> None:
    h, w = mask.shape
    r0, c0, r1, c1 = map(int, box)
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _sample_lock_spec(rng: np.random.Generator, frame_color: int, pattern_color: int, direction: str) -> dict:
    mask = _sample_pattern_mask(rng)
    ph, pw = map(int, mask.shape)
    h = int(rng.integers(max(6, ph + 4), 9))
    w = int(rng.integers(max(6, pw + 4), 9))
    pat_r = int(rng.integers(2, h - ph - 1))
    pat_c = int(rng.integers(2, w - pw - 1))
    if direction in ("U", "D"):
        gap_offset = int(rng.integers(1, w - 1))
    else:
        gap_offset = int(rng.integers(1, h - 1))
    return {
        "frame_color": int(frame_color),
        "pattern_color": int(pattern_color),
        "direction": direction,
        "frame_h": int(h),
        "frame_w": int(w),
        "pattern_mask": mask,
        "pattern_h": ph,
        "pattern_w": pw,
        "pattern_r": int(pat_r),
        "pattern_c": int(pat_c),
        "gap_offset": int(gap_offset),
    }


def _key_options(direction: str) -> list[tuple[str, str]]:
    if direction == "L":
        return [("H", "left"), ("V", "middle")]
    if direction == "U":
        return [("V", "top"), ("H", "middle")]
    if direction == "R":
        return [("H", "right")]
    return [("V", "bottom")]


def _length_choices(gap_kind: str) -> list[int]:
    if gap_kind == "middle":
        return [3, 5]
    return [5]


def _gap_index(length: int, gap_kind: str) -> int:
    if gap_kind in ("left", "top"):
        return 0 if length == 3 else 1
    if gap_kind in ("right", "bottom"):
        return int(length - 1 if length == 3 else length - 2)
    return int(length // 2)


def _place_key(
    inp: np.ndarray,
    bg: int,
    reserved: np.ndarray,
    rng: np.random.Generator,
    color: int,
    direction: str,
) -> bool:
    h, w = inp.shape
    options = _key_options(direction)
    option_order = list(range(len(options)))
    rng.shuffle(option_order)

    for option_idx in option_order:
        orientation, gap_kind = options[option_idx]
        lengths = _length_choices(gap_kind)
        rng.shuffle(lengths)
        for length in lengths:
            gap_idx = _gap_index(length, gap_kind)
            if orientation == "H":
                rows = [0, h - 1]
                rng.shuffle(rows)
                starts = list(range(1, w - length)) if w - length >= 2 else list(range(0, w - length + 1))
                rng.shuffle(starts)
                for row in rows:
                    for start in starts:
                        cells = [(int(row), int(start + i)) for i in range(length)]
                        if any(int(inp[r, c]) != bg or reserved[r, c] for r, c in cells):
                            continue
                        adj = []
                        if start > 0:
                            adj.append((int(row), int(start - 1)))
                        if start + length < w:
                            adj.append((int(row), int(start + length)))
                        if any(int(inp[r, c]) != bg or reserved[r, c] for r, c in adj):
                            continue
                        for i, (r, c) in enumerate(cells):
                            if i != gap_idx:
                                inp[r, c] = int(color)
                            reserved[r, c] = True
                        for r, c in adj:
                            reserved[r, c] = True
                        return True
            else:
                cols = [0, w - 1]
                rng.shuffle(cols)
                starts = list(range(1, h - length)) if h - length >= 2 else list(range(0, h - length + 1))
                rng.shuffle(starts)
                for col in cols:
                    for start in starts:
                        cells = [(int(start + i), int(col)) for i in range(length)]
                        if any(int(inp[r, c]) != bg or reserved[r, c] for r, c in cells):
                            continue
                        adj = []
                        if start > 0:
                            adj.append((int(start - 1), int(col)))
                        if start + length < h:
                            adj.append((int(start + length), int(col)))
                        if any(int(inp[r, c]) != bg or reserved[r, c] for r, c in adj):
                            continue
                        for i, (r, c) in enumerate(cells):
                            if i != gap_idx:
                                inp[r, c] = int(color)
                            reserved[r, c] = True
                        for r, c in adj:
                            reserved[r, c] = True
                        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_locks = int(rng.integers(2, 5))
    h = int(rng.integers(26, 31))
    w = int(rng.integers(26, 31))

    bg = int(rng.integers(0, 10))
    palette = [c for c in range(10) if c != bg]
    if len(palette) < 2 * n_locks:
        return None
    rng.shuffle(palette)
    frame_colors = palette[:n_locks]
    pattern_colors = palette[n_locks : 2 * n_locks]

    directions = list(ALL_DIRECTIONS)
    rng.shuffle(directions)
    directions = directions[:n_locks]

    specs = [_sample_lock_spec(rng, frame_colors[i], pattern_colors[i], directions[i]) for i in range(n_locks)]

    inp = np.full((h, w), bg, dtype=int)
    expected = np.full((h, w), bg, dtype=int)
    mid_r = h // 2
    mid_c = w // 2
    slots = [
        (1, 1, mid_r - 2, mid_c - 2),
        (1, mid_c + 1, mid_r - 2, w - 2),
        (mid_r + 1, 1, h - 2, mid_c - 2),
        (mid_r + 1, mid_c + 1, h - 2, w - 2),
    ]
    slot_order = list(range(4))
    rng.shuffle(slot_order)

    for spec, slot_idx in zip(specs, slot_order[:n_locks]):
        slot_r0, slot_c0, slot_r1, slot_c1 = slots[slot_idx]
        frame_h = int(spec["frame_h"])
        frame_w = int(spec["frame_w"])
        pat_h = int(spec["pattern_h"])
        pat_w = int(spec["pattern_w"])
        direction = spec["direction"]

        if direction == "U":
            r_min = slot_r0 + pat_h
            r_max = slot_r1 - frame_h + 1
        elif direction == "D":
            r_min = slot_r0
            r_max = slot_r1 - frame_h - pat_h + 1
        else:
            r_min = slot_r0
            r_max = slot_r1 - frame_h + 1

        if direction == "L":
            c_min = slot_c0 + pat_w
            c_max = slot_c1 - frame_w + 1
        elif direction == "R":
            c_min = slot_c0
            c_max = slot_c1 - frame_w - pat_w + 1
        else:
            c_min = slot_c0
            c_max = slot_c1 - frame_w + 1

        if r_min > r_max or c_min > c_max:
            return None
        top = int(rng.integers(r_min, r_max + 1))
        left = int(rng.integers(c_min, c_max + 1))
        pat_top = int(top + spec["pattern_r"])
        pat_left = int(left + spec["pattern_c"])
        if direction == "L":
            out_top, out_left = pat_top, int(left - pat_w)
        elif direction == "R":
            out_top, out_left = pat_top, int(left + frame_w)
        elif direction == "U":
            out_top, out_left = int(top - pat_h), pat_left
        else:
            out_top, out_left = int(top + frame_h), pat_left

        _paint_frame(
            inp,
            top=int(top),
            left=int(left),
            h=frame_h,
            w=frame_w,
            color=int(spec["frame_color"]),
            bg=bg,
            gap_side=direction,
            gap_offset=int(spec["gap_offset"]),
        )
        _paint_frame(
            expected,
            top=int(top),
            left=int(left),
            h=frame_h,
            w=frame_w,
            color=int(spec["frame_color"]),
            bg=bg,
            gap_side=direction,
            gap_offset=int(spec["gap_offset"]),
        )
        _paint_mask(inp, spec["pattern_mask"], pat_top, pat_left, int(spec["pattern_color"]))
        _paint_mask(expected, spec["pattern_mask"], out_top, out_left, int(spec["pattern_color"]))

    key_reserved = np.zeros((h, w), dtype=bool)
    for spec in specs:
        if not _place_key(inp, bg, key_reserved, rng, int(spec["frame_color"]), spec["direction"]):
            return None

    if _get_bg(inp) != bg:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
