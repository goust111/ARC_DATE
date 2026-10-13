import random
from collections import Counter

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _false_runs(line: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, filled in enumerate(line.tolist() + [True]):
        if not filled and start is None:
            start = idx
        elif filled and start is not None:
            end = idx - 1
            if start > 0 and end < len(line) - 1:
                runs.append((start, end))
            start = None
    return runs


def _extract_objects(grid: np.ndarray, strict: bool = True) -> list[dict]:
    bg = _bg_color(grid)
    objects: list[dict] = []
    for comp in _components(grid != bg):
        colors = Counter(int(grid[r, c]) for r, c in comp)
        main_color, main_count = colors.most_common(1)[0]
        marker_cells = [(r, c) for r, c in comp if int(grid[r, c]) != main_color]
        valid = (
            len(colors) == 2
            and len(marker_cells) == 1
            and main_count >= 3
            and min(colors.values()) == 1
        )
        if not valid:
            if strict:
                raise ValueError("invalid object marker structure")
            continue

        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True

        gaps: list[tuple[str, int, int]] = []
        for start, end in _false_runs(mask[0, :]):
            gaps.append(("top", start, end))
        for start, end in _false_runs(mask[-1, :]):
            gaps.append(("bottom", start, end))
        for start, end in _false_runs(mask[:, 0]):
            gaps.append(("left", start, end))
        for start, end in _false_runs(mask[:, -1]):
            gaps.append(("right", start, end))

        if len(gaps) != 1:
            if strict:
                raise ValueError("expected exactly one bbox gap")
            continue

        side, start, end = gaps[0]
        objects.append(
            {
                "cells": comp,
                "main_color": int(main_color),
                "marker_color": int(grid[marker_cells[0]]),
                "marker_cell": marker_cells[0],
                "bbox": (r0, r1, c0, c1),
                "side": side,
                "gap_start": int(start),
                "gap_end": int(end),
                "gap_width": int(end - start + 1),
            }
        )
    return objects


def _beam_segments(obj: dict, shape: tuple[int, int]):
    h, w = shape
    r0, r1, c0, c1 = obj["bbox"]
    start, end = obj["gap_start"], obj["gap_end"]
    side = obj["side"]
    if side == "top":
        cols = list(range(c0 + start, c0 + end + 1))
        for r in range(r0 - 1, -1, -1):
            yield [(r, c) for c in cols]
    elif side == "bottom":
        cols = list(range(c0 + start, c0 + end + 1))
        for r in range(r1 + 1, h):
            yield [(r, c) for c in cols]
    elif side == "left":
        rows = list(range(r0 + start, r0 + end + 1))
        for c in range(c0 - 1, -1, -1):
            yield [(r, c) for r in rows]
    else:
        rows = list(range(r0 + start, r0 + end + 1))
        for c in range(c1 + 1, w):
            yield [(r, c) for r in rows]


def _beam_status(obj: dict, occupied: np.ndarray) -> tuple[str, int]:
    painted = 0
    for segment in _beam_segments(obj, occupied.shape):
        if any(occupied[r, c] for r, c in segment):
            return "blocked", painted
        painted += 1
    return "boundary", painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit a marker-colored band from each object's unique bbox-side gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    occupied = g != bg

    for obj in _extract_objects(g, strict=True):
        for segment in _beam_segments(obj, g.shape):
            if any(occupied[r, c] for r, c in segment):
                break
            for r, c in segment:
                out[r, c] = obj["marker_color"]

    return out


# --------------- generator ---------------

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
OBJECT_COUNT_OPTIONS = (3, 4)
OBJ_H_RANGE = (4, 7)
OBJ_W_RANGE = (4, 7)
GAP_WIDTH_MAX = 3
PLACE_TRIES = 80
GEN_RETRIES = 400
MIN_OPEN_LEN = 2
MIN_BLOCKED_LEN = 2
SIDES = ("top", "bottom", "left", "right")


def _make_spec(rng: random.Random, side: str, main_color: int, marker_color: int) -> dict:
    h = rng.randint(*OBJ_H_RANGE)
    w = rng.randint(*OBJ_W_RANGE)
    edge_len = w if side in ("top", "bottom") else h
    gap_width = rng.randint(1, min(GAP_WIDTH_MAX, edge_len - 2))
    gap_start = rng.randint(1, edge_len - gap_width - 1)
    marker_r = rng.randint(1, h - 2)
    marker_c = rng.randint(1, w - 2)
    return {
        "h": h,
        "w": w,
        "side": side,
        "gap_start": gap_start,
        "gap_width": gap_width,
        "main_color": main_color,
        "marker_color": marker_color,
        "marker_pos": (marker_r, marker_c),
    }


def _bbox(spec: dict) -> tuple[int, int, int, int]:
    r0, c0 = spec["r0"], spec["c0"]
    return (r0, r0 + spec["h"] - 1, c0, c0 + spec["w"] - 1)


def _boxes_separated(spec: dict, placed: list[dict], pad: int = 1) -> bool:
    r0, r1, c0, c1 = _bbox(spec)
    for other in placed:
        or0, or1, oc0, oc1 = _bbox(other)
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _spec_patch(spec: dict, bg: int) -> np.ndarray:
    patch = np.full((spec["h"], spec["w"]), spec["main_color"], dtype=int)
    if spec["side"] == "top":
        patch[0, spec["gap_start"] : spec["gap_start"] + spec["gap_width"]] = bg
    elif spec["side"] == "bottom":
        patch[-1, spec["gap_start"] : spec["gap_start"] + spec["gap_width"]] = bg
    elif spec["side"] == "left":
        patch[spec["gap_start"] : spec["gap_start"] + spec["gap_width"], 0] = bg
    else:
        patch[spec["gap_start"] : spec["gap_start"] + spec["gap_width"], -1] = bg
    mr, mc = spec["marker_pos"]
    patch[mr, mc] = spec["marker_color"]
    return patch


def _build_grid(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        patch = _spec_patch(spec, bg)
        r0, c0 = spec["r0"], spec["c0"]
        mask = patch != bg
        view = grid[r0 : r0 + spec["h"], c0 : c0 + spec["w"]]
        view[mask] = patch[mask]
    return grid


def _object_from_spec(spec: dict) -> dict:
    r0, c0 = spec["r0"], spec["c0"]
    h, w = spec["h"], spec["w"]
    return {
        "bbox": (r0, r0 + h - 1, c0, c0 + w - 1),
        "side": spec["side"],
        "gap_start": spec["gap_start"],
        "gap_end": spec["gap_start"] + spec["gap_width"] - 1,
        "gap_width": spec["gap_width"],
        "marker_color": spec["marker_color"],
    }


def _occupied_from_specs(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    return _build_grid(h, w, bg, specs) != bg


def _candidate_positions_for_blocked(
    spec: dict, blocker: dict, grid_h: int, grid_w: int, rng: random.Random
) -> list[tuple[int, int]]:
    br0, br1, bc0, bc1 = _bbox(blocker)
    h, w = spec["h"], spec["w"]
    side = spec["side"]
    gap_start = spec["gap_start"]
    gap_width = spec["gap_width"]
    out: list[tuple[int, int]] = []

    if side == "top":
        cmin = max(0, bc0 - (gap_start + gap_width - 1))
        cmax = min(grid_w - w, bc1 - gap_start)
        if cmin <= cmax:
            for _ in range(PLACE_TRIES):
                beam_len = rng.randint(MIN_BLOCKED_LEN, max(MIN_BLOCKED_LEN, 5))
                r0 = br1 + beam_len + 1
                c0 = rng.randint(cmin, cmax)
                if 0 <= r0 <= grid_h - h:
                    out.append((r0, c0))
    elif side == "bottom":
        cmin = max(0, bc0 - (gap_start + gap_width - 1))
        cmax = min(grid_w - w, bc1 - gap_start)
        if cmin <= cmax:
            for _ in range(PLACE_TRIES):
                beam_len = rng.randint(MIN_BLOCKED_LEN, max(MIN_BLOCKED_LEN, 5))
                r0 = br0 - beam_len - h
                c0 = rng.randint(cmin, cmax)
                if 0 <= r0 <= grid_h - h:
                    out.append((r0, c0))
    elif side == "left":
        rmin = max(0, br0 - (gap_start + gap_width - 1))
        rmax = min(grid_h - h, br1 - gap_start)
        if rmin <= rmax:
            for _ in range(PLACE_TRIES):
                beam_len = rng.randint(MIN_BLOCKED_LEN, max(MIN_BLOCKED_LEN, 5))
                c0 = bc1 + beam_len + 1
                r0 = rng.randint(rmin, rmax)
                if 0 <= c0 <= grid_w - w:
                    out.append((r0, c0))
    else:
        rmin = max(0, br0 - (gap_start + gap_width - 1))
        rmax = min(grid_h - h, br1 - gap_start)
        if rmin <= rmax:
            for _ in range(PLACE_TRIES):
                beam_len = rng.randint(MIN_BLOCKED_LEN, max(MIN_BLOCKED_LEN, 5))
                c0 = bc0 - beam_len - w
                r0 = rng.randint(rmin, rmax)
                if 0 <= c0 <= grid_w - w:
                    out.append((r0, c0))

    rng.shuffle(out)
    return out


def _place_blocked(
    spec: dict, blocker: dict, placed: list[dict], grid_h: int, grid_w: int, bg: int, rng: random.Random
) -> bool:
    for r0, c0 in _candidate_positions_for_blocked(spec, blocker, grid_h, grid_w, rng):
        spec["r0"], spec["c0"] = r0, c0
        if not _boxes_separated(spec, placed):
            continue
        occupied = _occupied_from_specs(grid_h, grid_w, bg, placed + [spec])
        status, length = _beam_status(_object_from_spec(spec), occupied)
        if status == "blocked" and length >= MIN_BLOCKED_LEN:
            return True
    spec.pop("r0", None)
    spec.pop("c0", None)
    return False


def _place_with_status(
    spec: dict,
    placed: list[dict],
    grid_h: int,
    grid_w: int,
    bg: int,
    rng: random.Random,
    allowed: set[str],
    min_len: int,
) -> bool:
    for _ in range(PLACE_TRIES * 3):
        spec["r0"] = rng.randint(0, grid_h - spec["h"])
        spec["c0"] = rng.randint(0, grid_w - spec["w"])
        if not _boxes_separated(spec, placed):
            continue
        occupied = _occupied_from_specs(grid_h, grid_w, bg, placed + [spec])
        status, length = _beam_status(_object_from_spec(spec), occupied)
        if status in allowed and length >= min_len:
            return True
    spec.pop("r0", None)
    spec.pop("c0", None)
    return False


def _place_anywhere(spec: dict, placed: list[dict], grid_h: int, grid_w: int, rng: random.Random) -> bool:
    for _ in range(PLACE_TRIES * 2):
        spec["r0"] = rng.randint(0, grid_h - spec["h"])
        spec["c0"] = rng.randint(0, grid_w - spec["w"])
        if _boxes_separated(spec, placed):
            return True
    spec.pop("r0", None)
    spec.pop("c0", None)
    return False


def _validate_generated(grid: np.ndarray, expected_count: int) -> bool:
    try:
        objects = _extract_objects(grid, strict=True)
    except ValueError:
        return False
    if len(objects) != expected_count:
        return False

    bg = _bg_color(grid)
    counts = Counter(int(v) for v in grid.flat)
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False

    occupied = grid != bg
    statuses = [_beam_status(obj, occupied)[0] for obj in objects]
    if "boundary" not in statuses or "blocked" not in statuses:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out != grid)) < 8:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    bg = rng.randrange(10)
    obj_count = rng.choice(OBJECT_COUNT_OPTIONS)

    palette = [c for c in range(10) if c != bg]
    if 2 * obj_count > len(palette):
        return None
    color_pool = rng.sample(palette, 2 * obj_count)

    sides = rng.sample(list(SIDES), k=obj_count)
    specs = [
        _make_spec(rng, sides[i], color_pool[2 * i], color_pool[2 * i + 1])
        for i in range(obj_count)
    ]

    placed: list[dict] = []

    blocker = specs[0]
    if not _place_anywhere(blocker, placed, grid_h, grid_w, rng):
        return None
    placed.append(blocker)

    blocked_emitter = specs[1]
    if not _place_blocked(blocked_emitter, blocker, placed, grid_h, grid_w, bg, rng):
        return None
    placed.append(blocked_emitter)

    open_object = specs[2]
    if not _place_with_status(open_object, placed, grid_h, grid_w, bg, rng, {"boundary"}, MIN_OPEN_LEN):
        return None
    placed.append(open_object)

    for spec in specs[3:]:
        desired = {"blocked"} if rng.random() < 0.5 else {"boundary"}
        if not _place_with_status(spec, placed, grid_h, grid_w, bg, rng, desired, 1):
            if not _place_with_status(spec, placed, grid_h, grid_w, bg, rng, {"boundary", "blocked"}, 1):
                return None
        placed.append(spec)

    grid = _build_grid(grid_h, grid_w, bg, placed)
    if not _validate_generated(grid, obj_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
