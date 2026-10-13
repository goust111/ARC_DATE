from __future__ import annotations

import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = tuple(range(1, 10))
SIDES = ("up", "down", "left", "right")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if np.any(values == 0):
        return 0
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    return comps


def _is_consecutive(values: list[int]) -> bool:
    return bool(values) and values[-1] - values[0] + 1 == len(values)


def _detect_notch(mask: np.ndarray) -> tuple[str, int, int]:
    h, w = mask.shape
    missing = np.argwhere(~mask)
    assert missing.size > 0, "object must contain one notch"

    candidates: list[tuple[str, int, int]] = []

    if np.all(missing[:, 0] == 0):
        cols = sorted(int(c) for c in missing[:, 1])
        if (
            len(cols) < w
            and _is_consecutive(cols)
            and np.all(mask[1:, :])
            and np.all(mask[0, : cols[0]])
            and np.all(mask[0, cols[-1] + 1 :])
        ):
            candidates.append(("up", cols[0], cols[-1]))

    if np.all(missing[:, 0] == h - 1):
        cols = sorted(int(c) for c in missing[:, 1])
        if (
            len(cols) < w
            and _is_consecutive(cols)
            and np.all(mask[: h - 1, :])
            and np.all(mask[h - 1, : cols[0]])
            and np.all(mask[h - 1, cols[-1] + 1 :])
        ):
            candidates.append(("down", cols[0], cols[-1]))

    if np.all(missing[:, 1] == 0):
        rows = sorted(int(r) for r in missing[:, 0])
        if (
            len(rows) < h
            and _is_consecutive(rows)
            and np.all(mask[:, 1:])
            and np.all(mask[: rows[0], 0])
            and np.all(mask[rows[-1] + 1 :, 0])
        ):
            candidates.append(("left", rows[0], rows[-1]))

    if np.all(missing[:, 1] == w - 1):
        rows = sorted(int(r) for r in missing[:, 0])
        if (
            len(rows) < h
            and _is_consecutive(rows)
            and np.all(mask[:, : w - 1])
            and np.all(mask[: rows[0], w - 1])
            and np.all(mask[rows[-1] + 1 :, w - 1])
        ):
            candidates.append(("right", rows[0], rows[-1]))

    assert len(candidates) == 1, "object must be a rectangle with one unambiguous side notch"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep every notched rectangle unchanged and stamp a same-color border strip on the
    edge indicated by the notch direction. The strip length matches the notch width
    and stays aligned with the notch projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background_color(g)
    out = g.copy()

    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp["bbox"]  # type: ignore[index]
        coords = comp["coords"]  # type: ignore[index]
        color = int(comp["color"])  # type: ignore[index]
        hh = int(r1) - int(r0) + 1
        ww = int(c1) - int(c0) + 1

        mask = np.zeros((hh, ww), dtype=bool)
        for rr, cc in coords:
            mask[int(rr) - int(r0), int(cc) - int(c0)] = True

        direction, lo, hi = _detect_notch(mask)
        if direction == "up":
            out[0, int(c0) + lo : int(c0) + hi + 1] = color
        elif direction == "down":
            out[-1, int(c0) + lo : int(c0) + hi + 1] = color
        elif direction == "left":
            out[int(r0) + lo : int(r0) + hi + 1, 0] = color
        else:
            out[int(r0) + lo : int(r0) + hi + 1, -1] = color

    return out


# --------------- generator ---------------

GRID_MIN = 12
GRID_MAX = 22
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 5
OBJECT_MIN_SIDE = 3
OBJECT_MAX_HEIGHT = 7
OBJECT_MAX_WIDTH = 8
NOTCH_MAX_WIDTH = 4
OBJECT_GAP = 1
EDGE_GAP = 1
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400


def _interval_has_gap(existing: list[tuple[int, int]], lo: int, hi: int, gap: int = 0) -> bool:
    for old_lo, old_hi in existing:
        if not (hi + gap < old_lo or old_hi + gap < lo):
            return False
    return True


def _bbox_has_gap(existing: list[tuple[int, int, int, int]], bbox: tuple[int, int, int, int], gap: int) -> bool:
    r0, c0, r1, c1 = bbox
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _build_object_mask(height: int, width: int, direction: str, start: int, notch_width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    end = start + notch_width
    if direction == "up":
        mask[0, start:end] = False
    elif direction == "down":
        mask[-1, start:end] = False
    elif direction == "left":
        mask[start:end, 0] = False
    else:
        mask[start:end, -1] = False
    return mask


def _paint_object(grid: np.ndarray, spec: dict[str, int | str]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    height = int(spec["height"])
    width = int(spec["width"])
    color = int(spec["color"])
    direction = str(spec["direction"])
    start = int(spec["start"])
    notch_width = int(spec["notch_width"])

    mask = _build_object_mask(height, width, direction, start, notch_width)
    patch = np.full((height, width), color, dtype=int)
    patch[~mask] = 0
    window = grid[r0 : r0 + height, c0 : c0 + width]
    window[mask] = patch[mask]


def _paint_expected_output(grid: np.ndarray, specs: list[dict[str, int | str]]) -> np.ndarray:
    out = grid.copy()
    h, w = out.shape
    for spec in specs:
        color = int(spec["color"])
        direction = str(spec["direction"])
        start = int(spec["start"])
        notch_width = int(spec["notch_width"])
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])

        if direction == "up":
            out[0, c0 + start : c0 + start + notch_width] = color
        elif direction == "down":
            out[h - 1, c0 + start : c0 + start + notch_width] = color
        elif direction == "left":
            out[r0 + start : r0 + start + notch_width, 0] = color
        else:
            out[r0 + start : r0 + start + notch_width, w - 1] = color
    return out


def _random_directions(count: int) -> list[str]:
    base = list(SIDES)
    random.shuffle(base)
    directions = [base[i % len(base)] for i in range(count)]
    random.shuffle(directions)
    return directions


def _sample_object_spec(height: int, width: int, color: int, direction: str) -> dict[str, int | str]:
    if direction in ("up", "down"):
        obj_h = random.randint(OBJECT_MIN_SIDE, min(OBJECT_MAX_HEIGHT, height - 2))
        obj_w = random.randint(OBJECT_MIN_SIDE, min(OBJECT_MAX_WIDTH, width - 2))
        max_notch = min(NOTCH_MAX_WIDTH, obj_w - 1)
        notch_width = random.randint(1, max_notch)
        if notch_width == 1:
            start_choices = list(range(1, obj_w - 1))
        else:
            start_choices = list(range(0, obj_w - notch_width + 1))
    else:
        obj_h = random.randint(OBJECT_MIN_SIDE, min(OBJECT_MAX_HEIGHT, height - 2))
        obj_w = random.randint(OBJECT_MIN_SIDE, min(OBJECT_MAX_WIDTH, width - 2))
        max_notch = min(NOTCH_MAX_WIDTH, obj_h - 1)
        notch_width = random.randint(1, max_notch)
        if notch_width == 1:
            start_choices = list(range(1, obj_h - 1))
        else:
            start_choices = list(range(0, obj_h - notch_width + 1))

    return {
        "height": obj_h,
        "width": obj_w,
        "color": color,
        "direction": direction,
        "start": random.choice(start_choices),
        "notch_width": notch_width,
    }


def _place_scene() -> np.ndarray | None:
    count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    min_side = max(GRID_MIN, 2 * count + 6)
    grid_h = random.randint(min_side, GRID_MAX)
    grid_w = random.randint(min_side, GRID_MAX)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    colors = random.sample(PALETTE, count)
    directions = _random_directions(count)

    edge_intervals: dict[str, list[tuple[int, int]]] = {side: [] for side in SIDES}
    placed_boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict[str, int | str]] = []

    for color, direction in zip(colors, directions):
        placed = False
        for _ in range(PLACEMENT_TRIES):
            spec = _sample_object_spec(grid_h, grid_w, color, direction)
            obj_h = int(spec["height"])
            obj_w = int(spec["width"])
            r0 = random.randint(1, grid_h - obj_h - 1)
            c0 = random.randint(1, grid_w - obj_w - 1)
            bbox = (r0, c0, r0 + obj_h - 1, c0 + obj_w - 1)
            if not _bbox_has_gap(placed_boxes, bbox, OBJECT_GAP):
                continue

            start = int(spec["start"])
            notch_width = int(spec["notch_width"])
            if direction in ("up", "down"):
                lo = c0 + start
                hi = lo + notch_width - 1
            else:
                lo = r0 + start
                hi = lo + notch_width - 1
            if not _interval_has_gap(edge_intervals[direction], lo, hi, EDGE_GAP):
                continue

            spec["r0"] = r0
            spec["c0"] = c0
            _paint_object(grid, spec)

            placed_boxes.append(bbox)
            edge_intervals[direction].append((lo, hi))
            specs.append(spec)
            placed = True
            break

        if not placed:
            return None

    expected = _paint_expected_output(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    added = int(np.count_nonzero(actual != grid))
    if added != sum(int(spec["notch_width"]) for spec in specs):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _place_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notched-rectangle scene")
