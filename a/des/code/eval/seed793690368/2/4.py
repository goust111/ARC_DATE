from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

MIN_PAIRS = 2
MAX_PAIRS = 4
MIN_LENGTH = 4
MAX_LENGTH = 8
MIN_THICKNESS = 3
MAX_THICKNESS = 5
MIN_GRID = 18
MAX_GRID = 30
PLACEMENT_PAD = 1
MAX_RETRIES = 300


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in colors or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rs = [rr for rr, _ in coords]
            cs = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                    "area": len(coords),
                }
            )

    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _pattern_tuple(values: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in values.astype(int).tolist())


def _lock_candidates(comp: dict) -> list[dict]:
    assert int(comp["color"]) == GREY
    mask = _component_mask(comp)
    h, w = mask.shape
    candidates: list[tuple[str, tuple[int, ...]]] = []

    if h >= 2 and np.all(mask[1:, :]):
        missing = ~mask[0, :]
        if 0 < int(missing.sum()) < w:
            candidates.append(("top", _pattern_tuple(missing)))
    if h >= 2 and np.all(mask[:-1, :]):
        missing = ~mask[-1, :]
        if 0 < int(missing.sum()) < w:
            candidates.append(("bottom", _pattern_tuple(missing)))
    if w >= 2 and np.all(mask[:, 1:]):
        missing = ~mask[:, 0]
        if 0 < int(missing.sum()) < h:
            candidates.append(("left", _pattern_tuple(missing)))
    if w >= 2 and np.all(mask[:, :-1]):
        missing = ~mask[:, -1]
        if 0 < int(missing.sum()) < h:
            candidates.append(("right", _pattern_tuple(missing)))

    r0, c0, r1, c1 = comp["bbox"]
    parsed: list[dict] = []
    for side, pattern in candidates:
        length = len(pattern)
        thickness = (r1 - r0 + 1) if side in ("top", "bottom") else (c1 - c0 + 1)
        parsed.append(
            {
                "bbox": comp["bbox"],
                "side": side,
                "pattern": pattern,
                "length": length,
                "thickness": thickness,
            }
        )
    assert parsed, "grey component must be a lock candidate"
    return parsed


def _extract_key(comp: dict, side: str) -> dict:
    mask = _component_mask(comp)
    h, w = mask.shape

    if side == "top":
        assert h == 2 and np.all(mask[0, :])
        pattern = _pattern_tuple(mask[1, :])
        length = w
    elif side == "bottom":
        assert h == 2 and np.all(mask[1, :])
        pattern = _pattern_tuple(mask[0, :])
        length = w
    elif side == "left":
        assert w == 2 and np.all(mask[:, 0])
        pattern = _pattern_tuple(mask[:, 1])
        length = h
    else:
        assert w == 2 and np.all(mask[:, 1])
        pattern = _pattern_tuple(mask[:, 0])
        length = h

    assert 0 < sum(pattern) < length, "key must have at least one tooth and one flat segment"
    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "pattern": pattern,
        "length": length,
        "signature": (length, pattern),
    }


def _key_bbox_size(side: str, length: int) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return 2, length
    return length, 2


def _lock_cells(side: str, top: int, left: int, length: int, thickness: int, pattern: tuple[int, ...]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if side == "top":
        for r in range(thickness):
            for c in range(length):
                if r == 0 and pattern[c]:
                    continue
                cells.append((top + r, left + c))
    elif side == "bottom":
        for r in range(thickness):
            for c in range(length):
                if r == thickness - 1 and pattern[c]:
                    continue
                cells.append((top + r, left + c))
    elif side == "left":
        for r in range(length):
            for c in range(thickness):
                if c == 0 and pattern[r]:
                    continue
                cells.append((top + r, left + c))
    else:
        for r in range(length):
            for c in range(thickness):
                if c == thickness - 1 and pattern[r]:
                    continue
                cells.append((top + r, left + c))
    return cells


def _key_cells(side: str, top: int, left: int, length: int, pattern: tuple[int, ...]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if side == "top":
        for c in range(length):
            cells.append((top, left + c))
            if pattern[c]:
                cells.append((top + 1, left + c))
    elif side == "bottom":
        for c in range(length):
            if pattern[c]:
                cells.append((top, left + c))
            cells.append((top + 1, left + c))
    elif side == "left":
        for r in range(length):
            cells.append((top + r, left))
            if pattern[r]:
                cells.append((top + r, left + 1))
    else:
        for r in range(length):
            if pattern[r]:
                cells.append((top + r, left))
            cells.append((top + r, left + 1))
    return cells


def _placed_key_cells(side: str, lock_top: int, lock_left: int, length: int, thickness: int, pattern: tuple[int, ...]) -> list[tuple[int, int]]:
    if side == "top":
        return _key_cells(side, lock_top - 1, lock_left, length, pattern)
    if side == "bottom":
        return _key_cells(side, lock_top + thickness - 1, lock_left, length, pattern)
    if side == "left":
        return _key_cells(side, lock_top, lock_left - 1, length, pattern)
    return _key_cells(side, lock_top, lock_left + thickness - 1, length, pattern)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    components = _find_components(g, set(int(v) for v in np.unique(g)) - {BLACK})

    lock_candidate_groups = [_lock_candidates(comp) for comp in components if int(comp["color"]) == GREY]
    key_comps = [comp for comp in components if int(comp["color"]) not in (BLACK, GREY)]
    assert lock_candidate_groups and len(lock_candidate_groups) == len(key_comps)

    valid_configs: list[tuple[str, list[dict], list[dict]]] = []
    for side in SIDES:
        if not all(any(lock["side"] == side for lock in group) for group in lock_candidate_groups):
            continue
        locks = [next(lock for lock in group if lock["side"] == side) for group in lock_candidate_groups]
        try:
            keys = [_extract_key(comp, side) for comp in key_comps]
        except AssertionError:
            continue

        lock_by_signature: dict[tuple[int, tuple[int, ...]], dict] = {}
        duplicate_lock = False
        for lock in locks:
            sig = (int(lock["length"]), tuple(lock["pattern"]))
            if sig in lock_by_signature:
                duplicate_lock = True
                break
            lock_by_signature[sig] = lock
        if duplicate_lock:
            continue

        key_sigs = [key["signature"] for key in keys]
        if len(set(key_sigs)) != len(key_sigs):
            continue
        if set(key_sigs) != set(lock_by_signature):
            continue
        valid_configs.append((side, locks, keys))

    assert len(valid_configs) == 1, "could not infer a unique global side"
    side, locks, keys = valid_configs[0]
    lock_by_signature = {(int(lock["length"]), tuple(lock["pattern"])): lock for lock in locks}

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = BLACK

    used: set[tuple[int, tuple[int, ...]]] = set()
    for key in keys:
        sig = key["signature"]
        assert sig in lock_by_signature, "each key must match a lock"
        assert sig not in used, "each lock should receive exactly one key"
        used.add(sig)
        lock = lock_by_signature[sig]
        for r, c in _placed_key_cells(side, lock["bbox"][0], lock["bbox"][1], lock["length"], lock["thickness"], lock["pattern"]):
            out[r, c] = int(key["color"])

    assert len(used) == len(locks)
    return out


def _can_place(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = PLACEMENT_PAD) -> None:
    h, w = blocked.shape
    for r, c in cells:
        r0 = max(0, r - pad)
        r1 = min(h, r + pad + 1)
        c0 = max(0, c - pad)
        c1 = min(w, c + pad + 1)
        blocked[r0:r1, c0:c1] = True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sample_pattern(length: int, used: set[tuple[int, tuple[int, ...]]]) -> tuple[int, ...]:
    for _ in range(200):
        pattern = tuple(random.randint(0, 1) for _ in range(length))
        if 0 < sum(pattern) < length and (length, pattern) not in used:
            return pattern
    raise RuntimeError("failed to sample a unique key pattern")


def _grid_bounds(num_pairs: int) -> tuple[int, int]:
    base = max(MIN_GRID, 16 + 2 * num_pairs)
    return base, MAX_GRID


def _lock_position_ranges(side: str, h: int, w: int, length: int, thickness: int) -> tuple[range, range]:
    if side == "top":
        return range(1, h - thickness + 1), range(0, w - length + 1)
    if side == "bottom":
        return range(0, h - thickness), range(0, w - length + 1)
    if side == "left":
        return range(0, h - length + 1), range(1, w - thickness + 1)
    return range(0, h - length + 1), range(0, w - thickness)


def _candidate_lock_positions(
    side: str,
    h: int,
    w: int,
    length: int,
    thickness: int,
    pattern: tuple[int, ...],
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
) -> list[tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]]:
    positions = []
    row_range, col_range = _lock_position_ranges(side, h, w, length, thickness)
    for top in row_range:
        for left in col_range:
            lock_cells = _lock_cells(side, top, left, length, thickness, pattern)
            merged_cells = lock_cells + _placed_key_cells(side, top, left, length, thickness, pattern)
            if _can_place(input_blocked, lock_cells) and _can_place(output_blocked, merged_cells):
                positions.append((top, left, lock_cells, merged_cells))
    return positions


def _candidate_key_positions(
    side: str,
    h: int,
    w: int,
    length: int,
    pattern: tuple[int, ...],
    input_blocked: np.ndarray,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    key_h, key_w = _key_bbox_size(side, length)
    positions = []
    for top in range(0, h - key_h + 1):
        for left in range(0, w - key_w + 1):
            cells = _key_cells(side, top, left, length, pattern)
            if _can_place(input_blocked, cells):
                positions.append((top, left, cells))
    return positions


def _render_output(h: int, w: int, side: str, locks: list[dict]) -> np.ndarray:
    out = np.full((h, w), BLACK, dtype=int)
    for lock in locks:
        _paint(out, lock["lock_cells"], GREY)
        _paint(
            out,
            _placed_key_cells(side, lock["top"], lock["left"], lock["length"], lock["thickness"], lock["pattern"]),
            int(lock["color"]),
        )
    return out


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    num_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
    side = random.choice(SIDES)
    min_grid, max_grid = _grid_bounds(num_pairs)
    h = random.randint(min_grid, max_grid)
    w = random.randint(min_grid, max_grid)

    colors = random.sample(KEY_COLORS, num_pairs)
    signatures: set[tuple[int, tuple[int, ...]]] = set()
    specs = []
    for color in colors:
        length = random.randint(MIN_LENGTH, MAX_LENGTH)
        thickness = random.randint(MIN_THICKNESS, MAX_THICKNESS)
        pattern = _sample_pattern(length, signatures)
        signatures.add((length, pattern))
        specs.append({"color": color, "length": length, "thickness": thickness, "pattern": pattern})

    input_grid = np.full((h, w), BLACK, dtype=int)
    input_blocked = np.zeros((h, w), dtype=bool)
    output_blocked = np.zeros((h, w), dtype=bool)

    random.shuffle(specs)
    placed_locks = []
    for spec in specs:
        candidates = _candidate_lock_positions(
            side,
            h,
            w,
            int(spec["length"]),
            int(spec["thickness"]),
            tuple(spec["pattern"]),
            input_blocked,
            output_blocked,
        )
        if not candidates:
            return None
        top, left, lock_cells, merged_cells = random.choice(candidates)
        _paint(input_grid, lock_cells, GREY)
        _mark_blocked(input_blocked, lock_cells)
        _mark_blocked(output_blocked, merged_cells)
        placed_locks.append(
            {
                "top": top,
                "left": left,
                "length": int(spec["length"]),
                "thickness": int(spec["thickness"]),
                "pattern": tuple(spec["pattern"]),
                "color": int(spec["color"]),
                "lock_cells": lock_cells,
            }
        )

    random.shuffle(placed_locks)
    for lock in placed_locks:
        key_candidates = _candidate_key_positions(
            side,
            h,
            w,
            int(lock["length"]),
            tuple(lock["pattern"]),
            input_blocked,
        )
        if not key_candidates:
            return None
        _, _, key_cells = random.choice(key_candidates)
        _paint(input_grid, key_cells, int(lock["color"]))
        _mark_blocked(input_blocked, key_cells)

    output_grid = _render_output(h, w, side, placed_locks)
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid, output_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid key-lock puzzle")
