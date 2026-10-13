from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

LOCK_SIDES = ["top", "bottom", "left", "right"]
LOCK_TOOTH_SIDE = {"top": "down", "bottom": "up", "left": "right", "right": "left"}

GEN_RETRIES = 400
SPEC_RETRIES = 120
PLACEMENT_RETRIES = 80


def _runs(bits: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(np.asarray(bits, dtype=bool).tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(bits) - 1))
    return runs


def _signature_from_runs(runs: list[tuple[int, int]]) -> tuple[int, ...]:
    assert runs
    pieces: list[int] = []
    prev_end = None
    for start, end in runs:
        if prev_end is not None:
            pieces.append(start - prev_end - 1)
        pieces.append(end - start + 1)
        prev_end = end
    return tuple(pieces)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return comps


def _parse_lock(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.asarray(comp["mask"], dtype=bool)
    h, w = mask.shape
    if h < 4 or w < 4:
        return None
    if np.any(mask[1:-1, 1:-1]):
        return None
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return None

    top = mask[0, 1:-1]
    bottom = mask[-1, 1:-1]
    left = mask[1:-1, 0]
    right = mask[1:-1, -1]
    side_vectors = {"top": top, "bottom": bottom, "left": left, "right": right}
    damaged = [side for side, vec in side_vectors.items() if not np.all(vec)]
    if len(damaged) != 1:
        return None

    side = damaged[0]
    for other_side, vec in side_vectors.items():
        if other_side != side and not np.all(vec):
            return None

    missing_runs = _runs(~side_vectors[side])
    if not missing_runs:
        return None

    gap_cells: list[tuple[int, int]] = []
    if side == "top":
        for start, end in missing_runs:
            for pos in range(start, end + 1):
                gap_cells.append((r0, c0 + 1 + pos))
    elif side == "bottom":
        for start, end in missing_runs:
            for pos in range(start, end + 1):
                gap_cells.append((r1, c0 + 1 + pos))
    elif side == "left":
        for start, end in missing_runs:
            for pos in range(start, end + 1):
                gap_cells.append((r0 + 1 + pos, c0))
    else:
        for start, end in missing_runs:
            for pos in range(start, end + 1):
                gap_cells.append((r0 + 1 + pos, c1))

    return {
        "kind": "lock",
        "color": int(comp["color"]),
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "axis": "horizontal" if side in ("top", "bottom") else "vertical",
        "runs": missing_runs,
        "signature": _signature_from_runs(missing_runs),
        "gap_cells": gap_cells,
    }


def _parse_key(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.asarray(comp["mask"], dtype=bool)
    h, w = mask.shape

    if h == 2 and w >= 3:
        tooth_row = None
        handle_row = None
        tooth_side = None
        if np.all(mask[0]) and np.any(mask[1]):
            handle_row, tooth_row, tooth_side = 0, 1, "down"
        elif np.all(mask[1]) and np.any(mask[0]):
            handle_row, tooth_row, tooth_side = 1, 0, "up"
        if tooth_row is None:
            return None
        tooth_vec = mask[tooth_row]
        if np.all(tooth_vec):
            return None
        tooth_runs = _runs(tooth_vec)
        if not tooth_runs:
            return None
        tooth_cells = [(r0 + tooth_row, c0 + pos) for pos, present in enumerate(tooth_vec.tolist()) if present]
        handle_cells = [(r0 + handle_row, c0 + pos) for pos in range(w)]
        return {
            "kind": "key",
            "color": int(comp["color"]),
            "bbox": (r0, c0, r1, c1),
            "axis": "horizontal",
            "tooth_side": tooth_side,
            "runs": tooth_runs,
            "signature": _signature_from_runs(tooth_runs),
            "tooth_cells": tooth_cells,
            "handle_cells": handle_cells,
        }

    if w == 2 and h >= 3:
        tooth_col = None
        handle_col = None
        tooth_side = None
        if np.all(mask[:, 0]) and np.any(mask[:, 1]):
            handle_col, tooth_col, tooth_side = 0, 1, "right"
        elif np.all(mask[:, 1]) and np.any(mask[:, 0]):
            handle_col, tooth_col, tooth_side = 1, 0, "left"
        if tooth_col is None:
            return None
        tooth_vec = mask[:, tooth_col]
        if np.all(tooth_vec):
            return None
        tooth_runs = _runs(tooth_vec)
        if not tooth_runs:
            return None
        tooth_cells = [(r0 + pos, c0 + tooth_col) for pos, present in enumerate(tooth_vec.tolist()) if present]
        handle_cells = [(r0 + pos, c0 + handle_col) for pos in range(h)]
        return {
            "kind": "key",
            "color": int(comp["color"]),
            "bbox": (r0, c0, r1, c1),
            "axis": "vertical",
            "tooth_side": tooth_side,
            "runs": tooth_runs,
            "signature": _signature_from_runs(tooth_runs),
            "tooth_cells": tooth_cells,
            "handle_cells": handle_cells,
        }

    return None


def _classify_objects(grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    locks: list[dict] = []
    keys: list[dict] = []
    for comp in _find_components(grid):
        lock = _parse_lock(comp)
        if lock is not None:
            locks.append(lock)
            continue
        key = _parse_key(comp)
        if key is not None:
            keys.append(key)
            continue
        raise AssertionError("unexpected foreground object")
    assert locks and keys and len(locks) == len(keys)
    return locks, keys


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each toothed key to the frame with the same notch pattern, then use the key teeth to repair the frame.

    Input structure:
    - Background is black.
    - Several monochrome rectangular hollow frames are present; each frame has missing runs on exactly one outer side.
    - The remaining foreground objects are monochrome keys made from a straight 1-cell-thick handle plus a row or column of 1-cell teeth on one side.

    Rule:
    - For every frame, read the damaged side and extract the sequence of missing run widths and gaps between them.
    - For every key, read the tooth row or column and extract the same normalized width-gap signature.
    - Match each frame to the key with the same signature and compatible facing, slide that key along its handle axis so its teeth land exactly on the frame gaps, recolor those teeth to the frame color, and erase the whole key.

    Key insight:
    - Colors do not determine the pairing; the invariant is the spacing pattern of tooth segments versus frame-side notch segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    locks, keys = _classify_objects(g)
    key_map: dict[tuple[str, str, tuple[int, ...]], dict] = {}
    for key in keys:
        map_key = (key["axis"], key["tooth_side"], key["signature"])
        assert map_key not in key_map
        key_map[map_key] = key

    out = g.copy()
    used_keys: set[int] = set()
    for lock in locks:
        need = (lock["axis"], LOCK_TOOTH_SIDE[lock["side"]], lock["signature"])
        assert need in key_map
        key = key_map[need]
        key_id = id(key)
        assert key_id not in used_keys
        used_keys.add(key_id)

        gap_cells = set(lock["gap_cells"])
        tooth_cells = key["tooth_cells"]
        assert tooth_cells
        if lock["axis"] == "horizontal":
            gap_row = {r for r, _c in gap_cells}
            tooth_row = {r for r, _c in tooth_cells}
            assert len(gap_row) == 1 and len(tooth_row) == 1 and gap_row == tooth_row
            delta = min(c for _r, c in gap_cells) - min(c for _r, c in tooth_cells)
            shifted = {(r, c + delta) for r, c in tooth_cells}
        else:
            gap_col = {c for _r, c in gap_cells}
            tooth_col = {c for _r, c in tooth_cells}
            assert len(gap_col) == 1 and len(tooth_col) == 1 and gap_col == tooth_col
            delta = min(r for r, _c in gap_cells) - min(r for r, _c in tooth_cells)
            shifted = {(r + delta, c) for r, c in tooth_cells}
        assert shifted == gap_cells

        for r, c in key["handle_cells"] + key["tooth_cells"]:
            out[r, c] = BG
        for r, c in shifted:
            out[r, c] = int(lock["color"])

    assert len(used_keys) == len(keys)
    return out


def _split_total(total: int) -> tuple[int, int]:
    left = random.randint(0, total)
    return left, total - left


def _pattern_starts(widths: list[int], gaps: list[int]) -> list[int]:
    starts: list[int] = []
    pos = 0
    for i, width in enumerate(widths):
        starts.append(pos)
        pos += width
        if i < len(gaps):
            pos += gaps[i]
    return starts


def _signature_from_widths_and_gaps(widths: list[int], gaps: list[int]) -> tuple[int, ...]:
    pieces: list[int] = []
    for i, width in enumerate(widths):
        if i > 0:
            pieces.append(gaps[i - 1])
        pieces.append(width)
    return tuple(pieces)


def _draw_lock(
    patch: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    side: str,
    starts: list[int],
    widths: list[int],
    color: int,
    complete: bool,
) -> None:
    patch[top, left : left + width] = color
    patch[top + height - 1, left : left + width] = color
    patch[top : top + height, left] = color
    patch[top : top + height, left + width - 1] = color
    if complete:
        return

    for start, run_width in zip(starts, widths):
        if side == "top":
            patch[top, left + 1 + start : left + 1 + start + run_width] = BG
        elif side == "bottom":
            patch[top + height - 1, left + 1 + start : left + 1 + start + run_width] = BG
        elif side == "left":
            patch[top + 1 + start : top + 1 + start + run_width, left] = BG
        else:
            patch[top + 1 + start : top + 1 + start + run_width, left + width - 1] = BG


def _draw_key(
    patch: np.ndarray,
    top: int,
    left: int,
    axis: str,
    tooth_side: str,
    length: int,
    starts: list[int],
    widths: list[int],
    color: int,
) -> None:
    if axis == "horizontal":
        handle_row = top if tooth_side == "down" else top + 1
        tooth_row = top + 1 if tooth_side == "down" else top
        patch[handle_row, left : left + length] = color
        for start, run_width in zip(starts, widths):
            patch[tooth_row, left + start : left + start + run_width] = color
        return

    handle_col = left if tooth_side == "right" else left + 1
    tooth_col = left + 1 if tooth_side == "right" else left
    patch[top : top + length, handle_col] = color
    for start, run_width in zip(starts, widths):
        patch[top + start : top + start + run_width, tooth_col] = color


def _make_pair_spec(lock_color: int, key_color: int, used_signatures: set[tuple[int, ...]]) -> dict | None:
    side = random.choice(LOCK_SIDES)
    axis = "horizontal" if side in ("top", "bottom") else "vertical"
    tooth_side = LOCK_TOOTH_SIDE[side]

    for _ in range(SPEC_RETRIES):
        run_count = random.randint(1, 3)
        widths = [random.randint(1, 3) for _ in range(run_count)]
        gaps = [random.randint(1, 2) for _ in range(run_count - 1)]
        signature = _signature_from_widths_and_gaps(widths, gaps)
        if signature in used_signatures:
            continue

        span = sum(widths) + sum(gaps)
        extra_lock = random.randint(0, 2)
        min_extra_key = 1 if run_count == 1 else 0
        extra_key = random.randint(min_extra_key, 3)

        lead_lock, trail_lock = _split_total(extra_lock)
        lead_key, trail_key = _split_total(extra_key)

        lock_track = span + lead_lock + trail_lock
        key_length = span + lead_key + trail_key
        if key_length < 4:
            continue
        if axis == "horizontal":
            width = lock_track + 2
            height = random.randint(4, 6)
        else:
            height = lock_track + 2
            width = random.randint(4, 6)
        if height > 9 or width > 9:
            continue

        pattern_rel = _pattern_starts(widths, gaps)
        lock_starts = [lead_lock + start for start in pattern_rel]
        key_starts = [lead_key + start for start in pattern_rel]

        start_dir = random.choice(["before", "after"])
        sep = random.randint(1, 3)
        if axis == "horizontal":
            group_h = height + 1
            group_w = width + sep + key_length
        else:
            group_h = height + sep + key_length
            group_w = width + 1
        if group_h > 14 or group_w > 20:
            continue

        return {
            "side": side,
            "axis": axis,
            "tooth_side": tooth_side,
            "signature": signature,
            "widths": widths,
            "lock_starts": lock_starts,
            "key_starts": key_starts,
            "lock_height": height,
            "lock_width": width,
            "key_length": key_length,
            "start_dir": start_dir,
            "sep": sep,
            "lock_color": lock_color,
            "key_color": key_color,
        }
    return None


def _build_group(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    axis = spec["axis"]
    side = spec["side"]
    tooth_side = spec["tooth_side"]
    height = int(spec["lock_height"])
    width = int(spec["lock_width"])
    key_length = int(spec["key_length"])
    sep = int(spec["sep"])
    start_dir = spec["start_dir"]

    if axis == "horizontal":
        group_h = height + 1
        group_w = width + sep + key_length
        inp = np.full((group_h, group_w), BG, dtype=int)
        out = np.full((group_h, group_w), BG, dtype=int)

        lock_top = 1 if side == "top" else 0
        key_top = 0 if side == "top" else height - 1
        if start_dir == "before":
            key_left = 0
            lock_left = key_length + sep
        else:
            lock_left = 0
            key_left = width + sep
    else:
        group_h = height + sep + key_length
        group_w = width + 1
        inp = np.full((group_h, group_w), BG, dtype=int)
        out = np.full((group_h, group_w), BG, dtype=int)

        lock_left = 1 if side == "left" else 0
        key_left = 0 if side == "left" else width - 1
        if start_dir == "before":
            key_top = 0
            lock_top = key_length + sep
        else:
            lock_top = 0
            key_top = height + sep

    _draw_lock(
        inp,
        lock_top,
        lock_left,
        height,
        width,
        side,
        spec["lock_starts"],
        spec["widths"],
        int(spec["lock_color"]),
        complete=False,
    )
    _draw_key(
        inp,
        key_top,
        key_left,
        axis,
        tooth_side,
        key_length,
        spec["key_starts"],
        spec["widths"],
        int(spec["key_color"]),
    )
    _draw_lock(
        out,
        lock_top,
        lock_left,
        height,
        width,
        side,
        spec["lock_starts"],
        spec["widths"],
        int(spec["lock_color"]),
        complete=True,
    )
    return inp, out


def _fits(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not occupied[r0:r1, c0:c1].any()


def _place_groups(group_shapes: list[tuple[int, int]]) -> tuple[int, int, list[tuple[int, int]]] | None:
    max_h = max(h for h, _w in group_shapes)
    max_w = max(w for _h, w in group_shapes)
    padded_area = sum((h + 2) * (w + 2) for h, w in group_shapes)
    min_side = max(18, max_h, max_w)

    indexed = list(enumerate(group_shapes))
    indexed.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)

    for _ in range(PLACEMENT_RETRIES):
        H = random.randint(min_side, 30)
        W = random.randint(min_side, 30)
        if H * W < padded_area:
            continue

        occupied = np.zeros((H, W), dtype=bool)
        placements = [(-1, -1)] * len(group_shapes)
        ok = True
        for idx, (gh, gw) in indexed:
            candidates = [(r, c) for r in range(H - gh + 1) for c in range(W - gw + 1)]
            random.shuffle(candidates)
            placed = False
            for top, left in candidates:
                if _fits(occupied, top, left, gh, gw):
                    placements[idx] = (top, left)
                    occupied[top : top + gh, left : left + gw] = True
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return H, W, placements
    return None


def _compose_scene(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    shapes = [grp[0].shape for grp in groups]
    packed = _place_groups(shapes)
    if packed is None:
        return None
    H, W, placements = packed
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    for (group_in, group_out), (top, left) in zip(groups, placements):
        gh, gw = group_in.shape
        inp[top : top + gh, left : left + gw] = group_in
        out[top : top + gh, left : left + gw] = group_out
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)
    colors = random.sample(PALETTE, 2 * pair_count)
    lock_colors = colors[:pair_count]
    key_colors = colors[pair_count:]

    used_signatures: set[tuple[int, ...]] = set()
    groups: list[tuple[np.ndarray, np.ndarray]] = []
    for lock_color, key_color in zip(lock_colors, key_colors):
        spec = _make_pair_spec(lock_color, key_color, used_signatures)
        if spec is None:
            return None
        used_signatures.add(spec["signature"])
        groups.append(_build_group(spec))

    scene = _compose_scene(groups)
    if scene is None:
        return None
    inp, expected = scene
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid lock-and-key ARC input")
