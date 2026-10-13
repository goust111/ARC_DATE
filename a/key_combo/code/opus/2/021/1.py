from __future__ import annotations

import math
import random

import numpy as np


BG = 0
GREY = 5
MASK = 6
PALETTE = (1, 2, 3, 4, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_border_mask(height: int, width: int) -> np.ndarray:
    border = np.zeros((int(height), int(width)), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    dst[int(top) : int(top) + int(h), int(left) : int(left) + int(w)] = patch


def _set_lr_pair(pattern: np.ndarray, row: int, col: int, color: int) -> None:
    row = int(row)
    col = int(col)
    color = int(color)
    pattern[row, col] = color
    pattern[row, pattern.shape[1] - 1 - col] = color


def _frame_info(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox_coords(coords)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _frame_border_mask(height, width)
    color_mask = patch == int(color)
    if np.any(color_mask & (~border)):
        return None
    if np.any(border & ~(color_mask | (patch == BG))):
        return None

    gap_mask = border & (~color_mask)
    gap_count = int(gap_mask.sum())
    if gap_count > 2:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap_count": gap_count,
        "sealed": gap_count == 0,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for color in _colors_present(grid, exclude={BG}):
        info = _frame_info(grid, int(color))
        if info is not None:
            frames.append(info)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames


def _repair_sealed_interior(interior: np.ndarray, *, lr_only: bool = False, strict: bool = True) -> np.ndarray:
    fixed = np.array(interior, copy=True)
    height, width = fixed.shape
    for row, col in zip(*np.where(interior == MASK)):
        row = int(row)
        col = int(col)
        candidates = [(row, width - 1 - col)]
        if not lr_only:
            candidates.extend(
                [
                    (height - 1 - row, col),
                    (height - 1 - row, width - 1 - col),
                ]
            )
        fill = None
        for rr, cc in candidates:
            value = int(interior[int(rr), int(cc)])
            if value != MASK:
                fill = value
                break
        if fill is None:
            if strict:
                raise AssertionError(f"masked cell {(row, col)} has no visible symmetry counterpart")
            continue
        fixed[row, col] = int(fill)
    return fixed


def _solve(
    grid: np.ndarray,
    *,
    force_status: str | None = None,
    lr_only: bool = False,
    strict: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    if strict:
        assert frames, "expected at least one rectangular frame"

    covered = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        covered[r0 + 1 : r1, c0 + 1 : c1] = True
    if strict:
        assert not np.any((g == MASK) & (~covered)), "pink masks must appear strictly inside a frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        mask = interior == MASK
        if not np.any(mask):
            continue

        sealed = frame["sealed"]
        if force_status == "sealed":
            sealed = True
        elif force_status == "broken":
            sealed = False

        if sealed:
            out[r0 + 1 : r1, c0 + 1 : c1] = _repair_sealed_interior(interior, lr_only=lr_only, strict=strict)
        else:
            sub = out[r0 + 1 : r1, c0 + 1 : c1]
            sub[mask] = GREY

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each rectangular frame's boundary integrity to choose symmetry repair or gray punishment for its masked interior cells.

    Input structure:
    - Black background surrounds several non-overlapping one-pixel rectangular frames, each drawn in its own color.
    - Some frames are complete hollow rectangles, while others have one or two background gaps on the border.
    - Each frame interior contains a left-right symmetric multicolor pattern with some cells replaced by pink masks.

    Rule:
    - Detect every frame color whose pixels lie on the perimeter of one bounding rectangle, allowing at most two missing border cells.
    - If the perimeter is complete, replace each interior pink cell with the first visible counterpart from this ordered search: left-right mirror, top-bottom mirror, then 180-degree rotation.
    - If the border has a gap, replace every interior pink cell with gray; leave the frame, non-pink interior cells, and the exterior untouched.

    Key insight:
    - The frame is a logical gate: one missing border pixel switches the whole enclosed masked pattern from symmetry completion to punishment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    return _solve(g, strict=True)


def _build_layout(frame_count: int) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    rows = 2
    min_cols = (int(frame_count) + rows - 1) // rows
    if min_cols == 2 and random.random() < 0.45:
        cols = 3
    else:
        cols = min_cols

    gutter = 1
    cell_h = random.randint(7, 10)
    cell_w = random.randint(7, 10) if cols == 2 else random.randint(6, 8)
    height = rows * cell_h + (rows + 1) * gutter
    width = cols * cell_w + (cols + 1) * gutter
    if height > 30 or width > 30:
        return None

    slots = []
    for sr in range(rows):
        for sc in range(cols):
            r0 = gutter + sr * (cell_h + gutter)
            c0 = gutter + sc * (cell_w + gutter)
            slots.append((r0, c0, r0 + cell_h - 1, c0 + cell_w - 1))
    return int(height), int(width), slots


def _random_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    half_w = (int(width) + 1) // 2
    pattern = np.empty((int(height), int(width)), dtype=int)
    row_bias = [random.choice(colors) for _ in range(height)]
    col_bias = [random.choice(colors) for _ in range(half_w)]
    phase = random.randrange(len(colors))
    for row in range(height):
        for col in range(half_w):
            roll = random.random()
            if roll < 0.35:
                color = row_bias[row]
            elif roll < 0.7:
                color = col_bias[col]
            else:
                color = colors[(row + 2 * col + phase) % len(colors)]
            _set_lr_pair(pattern, row, col, int(color))

    used = set(map(int, pattern.ravel()))
    for color in colors:
        if int(color) in used:
            continue
        row = random.randrange(height)
        col = random.randrange(half_w)
        _set_lr_pair(pattern, row, col, int(color))
        used.add(int(color))
    return pattern


def _candidate_level(pattern: np.ndarray, mask: np.ndarray, row: int, col: int) -> int:
    height, width = pattern.shape
    target = int(pattern[int(row), int(col)])
    candidates = [
        (int(row), width - 1 - int(col)),
        (height - 1 - int(row), int(col)),
        (height - 1 - int(row), width - 1 - int(col)),
    ]
    for level, (rr, cc) in enumerate(candidates, start=1):
        if not bool(mask[int(rr), int(cc)]) and int(pattern[int(rr), int(cc)]) == target:
            return int(level)
    return 0


def _max_fallback_level(pattern: np.ndarray, mask: np.ndarray) -> int:
    levels = [_candidate_level(pattern, mask, int(r), int(c)) for r, c in zip(*np.where(mask))]
    return max(levels, default=0)


def _all_masks_recoverable(pattern: np.ndarray, mask: np.ndarray) -> bool:
    return all(_candidate_level(pattern, mask, int(r), int(c)) >= 1 for r, c in zip(*np.where(mask)))


def _ensure_two_colors(pattern: np.ndarray, colors: list[int], blocked: set[tuple[int, int]]) -> bool:
    unique = set(map(int, pattern.ravel()))
    if len(unique) >= 2:
        return True

    half_w = pattern.shape[1] // 2
    options = [(row, col) for row in range(pattern.shape[0]) for col in range(half_w)]
    random.shuffle(options)
    for row, col in options:
        mirror = pattern.shape[1] - 1 - col
        if (row, col) in blocked or (row, mirror) in blocked:
            continue
        current = int(pattern[row, col])
        other_choices = [int(color) for color in colors if int(color) != current]
        if not other_choices:
            continue
        _set_lr_pair(pattern, row, col, random.choice(other_choices))
        return True
    return len(set(map(int, pattern.ravel()))) >= 2


def _plan_masks(pattern: np.ndarray, colors: list[int], *, require_triple: bool) -> np.ndarray | None:
    height, width = pattern.shape
    total = int(height * width)
    min_masks = max(1, math.ceil(total * 0.2))
    max_masks = max(min_masks, math.floor(total * 0.4))
    if require_triple:
        min_masks = max(min_masks, 3)
    if max_masks < min_masks:
        return None

    target = random.randint(min_masks, max_masks)
    mask = np.zeros((height, width), dtype=bool)
    blocked: set[tuple[int, int]] = set()

    row_pairs = [(row, height - 1 - row) for row in range(height // 2)]
    col_pairs = [(col, width - 1 - col) for col in range(width // 2)]
    quartets = [
        ((row, col), (row, mirror_c), (mirror_r, col), (mirror_r, mirror_c))
        for row, mirror_r in row_pairs
        for col, mirror_c in col_pairs
    ]
    random.shuffle(quartets)

    if require_triple:
        if not quartets:
            return None
        quartet = quartets.pop()
        fill_color = int(random.choice(colors))
        for cell in quartet:
            pattern[cell] = fill_color
            blocked.add(cell)
        for cell in quartet[:3]:
            mask[cell] = True

    remaining = target - int(mask.sum())
    for quartet in quartets:
        if remaining < 2 or random.random() >= 0.45:
            continue
        if any(cell in blocked for cell in quartet):
            continue
        fill_color = int(random.choice(colors))
        for cell in quartet:
            pattern[cell] = fill_color
            blocked.add(cell)
        pair = quartet[:2] if random.random() < 0.5 else quartet[2:]
        for cell in pair:
            mask[cell] = True
        remaining -= 2
        if remaining == 0:
            break

    pair_slots = [(row, col) for row in range(height) for col in range(width // 2)]
    random.shuffle(pair_slots)
    for row, col in pair_slots:
        if remaining == 0:
            break
        mirror = width - 1 - col
        if (row, col) in blocked or (row, mirror) in blocked:
            continue
        if mask[row, col] or mask[row, mirror]:
            continue
        chosen = (row, col) if random.random() < 0.5 else (row, mirror)
        mask[chosen] = True
        remaining -= 1

    if remaining != 0:
        return None
    if not _ensure_two_colors(pattern, colors, blocked):
        return None
    if not _all_masks_recoverable(pattern, mask):
        return None
    if require_triple and _max_fallback_level(pattern, mask) < 3:
        return None
    return mask


def _border_cells(height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    for col in range(width):
        cells.append((0, col))
        cells.append((height - 1, col))
    for row in range(1, height - 1):
        cells.append((row, 0))
        cells.append((row, width - 1))
    seen = []
    used = set()
    for cell in cells:
        if cell in used:
            continue
        used.add(cell)
        seen.append(cell)
    return seen


def _make_frame(
    slot: tuple[int, int, int, int],
    frame_color: int,
    interior_colors: list[int],
    *,
    sealed: bool,
    require_triple: bool,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    slot_r0, slot_c0, slot_r1, slot_c1 = slot
    slot_h = int(slot_r1 - slot_r0 + 1)
    slot_w = int(slot_c1 - slot_c0 + 1)
    outer_h = random.randint(5, slot_h)
    outer_w = random.randint(5, slot_w)
    inner_h = outer_h - 2
    inner_w = outer_w - 2

    pattern = _random_pattern(inner_h, inner_w, interior_colors)
    mask = _plan_masks(pattern, interior_colors, require_triple=require_triple)
    if mask is None:
        return None

    top = random.randint(slot_r0, slot_r1 - outer_h + 1)
    left = random.randint(slot_c0, slot_c1 - outer_w + 1)
    border = _frame_border_mask(outer_h, outer_w)
    input_patch = np.full((outer_h, outer_w), BG, dtype=int)
    output_patch = np.full((outer_h, outer_w), BG, dtype=int)
    input_patch[border] = int(frame_color)
    output_patch[border] = int(frame_color)

    gaps: list[tuple[int, int]] = []
    if not sealed:
        gap_count = random.randint(1, 2)
        candidates = _border_cells(outer_h, outer_w)
        random.shuffle(candidates)
        gaps = candidates[:gap_count]
        for row, col in gaps:
            input_patch[int(row), int(col)] = BG
            output_patch[int(row), int(col)] = BG

    input_inner = pattern.copy()
    input_inner[mask] = MASK
    output_inner = pattern.copy()
    if not sealed:
        output_inner[mask] = GREY

    input_patch[1:-1, 1:-1] = input_inner
    output_patch[1:-1, 1:-1] = output_inner

    return input_patch, output_patch, {
        "frame_color": int(frame_color),
        "bbox": (int(top), int(left), int(top + outer_h - 1), int(left + outer_w - 1)),
        "sealed": bool(sealed),
        "mask_count": int(mask.sum()),
        "interior_area": int(inner_h * inner_w),
        "max_fallback_level": _max_fallback_level(pattern, mask),
    }


def _validate_sample(grid: np.ndarray, expected: np.ndarray, metas: list[dict]) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    forced_sealed = _solve(grid, force_status="sealed", strict=False)
    forced_broken = _solve(grid, force_status="broken", strict=False)
    lr_only = _solve(grid, lr_only=True, strict=False)
    if np.array_equal(forced_sealed, expected):
        return False
    if np.array_equal(forced_broken, expected):
        return False
    if np.array_equal(lr_only, expected):
        return False

    frames = _find_frames(grid)
    if len(frames) != len(metas):
        return False
    meta_by_color = {meta["frame_color"]: meta for meta in metas}

    saw_sealed = False
    saw_broken = False
    saw_level3 = False
    for frame in frames:
        meta = meta_by_color.get(int(frame["color"]))
        if meta is None:
            return False
        if tuple(frame["bbox"]) != tuple(meta["bbox"]):
            return False
        if bool(frame["sealed"]) != bool(meta["sealed"]):
            return False
        ratio = float(meta["mask_count"]) / float(meta["interior_area"])
        if ratio < 0.2 or ratio > 0.4:
            return False
        saw_sealed |= bool(meta["sealed"])
        saw_broken |= not bool(meta["sealed"])
        saw_level3 |= bool(meta["sealed"] and meta["max_fallback_level"] >= 3)

    return saw_sealed and saw_broken and saw_level3


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(3, 5)
    layout = _build_layout(frame_count)
    if layout is None:
        return None
    height, width, slots = layout
    chosen_slots = random.sample(slots, frame_count)

    frame_colors = random.sample(list(PALETTE), frame_count)
    interior_palette = [color for color in PALETTE if color not in frame_colors]
    if len(interior_palette) < 2:
        return None

    broken_count = random.randint(1, frame_count - 1)
    broken_indices = set(random.sample(range(frame_count), broken_count))
    sealed_indices = [idx for idx in range(frame_count) if idx not in broken_indices]
    triple_index = random.choice(sealed_indices)

    grid = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)
    metas = []

    for idx, slot in enumerate(chosen_slots):
        sealed = idx not in broken_indices
        max_interior_colors = min(3, len(interior_palette))
        color_count = 2 if max_interior_colors == 2 else random.randint(2, max_interior_colors)
        interior_colors = random.sample(interior_palette, color_count)

        frame = None
        for _ in range(40):
            frame = _make_frame(
                slot,
                frame_colors[idx],
                interior_colors,
                sealed=sealed,
                require_triple=(idx == triple_index),
            )
            if frame is not None:
                break
        if frame is None:
            return None

        input_patch, output_patch, meta = frame
        top, left, bottom, right = meta["bbox"]
        _paste(grid, input_patch, top, left)
        _paste(expected, output_patch, top, left)
        metas.append(meta)

    if not _validate_sample(grid, expected, metas):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sealed-frame symmetry-repair input")
