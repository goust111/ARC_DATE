from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GEN_RETRIES = 500
OBJECT_RETRIES = 300
PLACEMENT_RETRIES = 120
GRID_H_RANGE = (15, 30)
GRID_W_RANGE = (15, 30)
OBJECT_COUNT_RANGE = (2, 4)
PROFILE_HEIGHT_RANGE = (2, 6)
PROFILE_SPAN_RANGE = (1, 3)
AREA_RANGE = (4, 12)
CLIP_PROB = 0.35


def _normalize_coords(
    coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = sorted({(int(r), int(c)) for r, c in coords})
    assert pts, "expected a non-empty coordinate set"
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    norm = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    return norm, int(h), int(w)


def _coords_to_mask(
    coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]
) -> np.ndarray:
    norm, h, w = _normalize_coords(coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _mask_coords(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple((int(r), int(c)) for r, c in np.argwhere(mask))


def _find_components(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            norm, ch, cw = _normalize_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": tuple(coords),
                    "local_coords": norm,
                    "mask": _coords_to_mask(coords),
                    "h": int(ch),
                    "w": int(cw),
                    "bbox": (
                        min(r for r, _ in coords),
                        min(c for _, c in coords),
                        max(r for r, _ in coords),
                        max(c for _, c in coords),
                    ),
                }
            )
    return comps


def _reflect_coords(
    coords: set[tuple[int, int]],
    orientation: str,
    axis2: int,
) -> set[tuple[int, int]]:
    if orientation == "vertical":
        return {(int(r), int(axis2 - c)) for r, c in coords}
    assert orientation == "horizontal"
    return {(int(axis2 - r), int(c)) for r, c in coords}


def _best_axis_for_orientation(mask: np.ndarray, orientation: str) -> dict[str, object]:
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    h, w = mask.shape
    limit = w - 1 if orientation == "vertical" else h - 1
    center2 = limit
    best: dict[str, object] | None = None

    for axis2 in range(0, 2 * limit + 1):
        reflected = _reflect_coords(coords, orientation, int(axis2))
        mismatch = len(coords ^ reflected)
        candidate = {
            "orientation": orientation,
            "axis2": int(axis2),
            "mismatch": int(mismatch),
            "center_gap": int(abs(axis2 - center2)),
            "reflected": reflected,
        }
        key = (candidate["mismatch"], candidate["center_gap"], candidate["axis2"])
        if best is None:
            best = candidate
            continue
        best_key = (best["mismatch"], best["center_gap"], best["axis2"])
        if key < best_key:
            best = candidate

    assert best is not None
    return best


def _direction_from_diff(
    coords: set[tuple[int, int]],
    reflected: set[tuple[int, int]],
    orientation: str,
    axis2: int,
) -> str:
    original_only = coords - reflected
    reflected_only = reflected - coords

    def side_score(points: set[tuple[int, int]]) -> int:
        score = 0
        if orientation == "vertical":
            for _r, c in points:
                delta = 2 * int(c) - int(axis2)
                score += (delta > 0) - (delta < 0)
        else:
            for r, _c in points:
                delta = 2 * int(r) - int(axis2)
                score += (delta > 0) - (delta < 0)
        return int(score)

    score = side_score(original_only)
    if score == 0:
        score = -side_score(reflected_only)
    assert score != 0, "defect side must be identifiable"

    if orientation == "vertical":
        return "right" if score > 0 else "left"
    return "down" if score > 0 else "up"


def _analyze_mask(mask: np.ndarray) -> dict[str, object]:
    assert mask.ndim == 2 and bool(mask.any()), "expected a non-empty 2D mask"
    vertical = _best_axis_for_orientation(mask, "vertical")
    horizontal = _best_axis_for_orientation(mask, "horizontal")
    candidates = [vertical, horizontal]
    candidates.sort(key=lambda item: (int(item["mismatch"]), int(item["center_gap"]), str(item["orientation"])))
    best = candidates[0]
    other = candidates[1]

    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    direction = _direction_from_diff(coords, best["reflected"], str(best["orientation"]), int(best["axis2"]))

    return {
        "orientation": str(best["orientation"]),
        "axis2": int(best["axis2"]),
        "direction": direction,
        "mismatch": int(best["mismatch"]),
        "other_mismatch": int(other["mismatch"]),
        "center_gap": int(best["center_gap"]),
        "area": int(mask.sum()),
    }


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], color: int) -> None:
    h, w = grid.shape
    for r, c in coords:
        if 0 <= int(r) < h and 0 <= int(c) < w:
            grid[int(r), int(c)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored object in the direction of its own symmetry defect by a distance equal to its area.

    Input structure:
    - The background is black.
    - The grid contains several disconnected single-color objects, each 4-connected and separated by empty space.
    - Every object is almost mirror-symmetric with respect to either a vertical or a horizontal axis.
    - Exactly one side of that near-symmetry has a one-pixel defect that breaks the symmetry.

    Rule:
    - For each object, compare vertical and horizontal mirror fits and keep the axis orientation with fewer mismatched pixels.
    - Read the side of the unmatched original pixels relative to that best axis; this side is the defect direction.
    - Translate the whole object by its own pixel count in that direction.
    - Clear the original object and redraw only the translated one, clipping any pixels that move outside the grid.

    Key insight:
    The object's geometry provides two independent quantities: near-reflection analysis gives the movement direction, while the object's area gives the movement distance.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = np.full_like(g, BG)

    for comp in _find_components(g):
        coords = tuple((int(r), int(c)) for r, c in comp["coords"])
        color = int(comp["color"])
        mask = np.asarray(comp["mask"], dtype=bool)
        info = _analyze_mask(mask)
        distance = int(info["area"])
        dr, dc = DIR_VECTORS[str(info["direction"])]
        moved = [(int(r) + dr * distance, int(c) + dc * distance) for r, c in coords]
        _paint_coords(out, moved, color)

    return out


def _random_profile(length: int, min_span: int, max_span: int, rng: random.Random) -> list[int]:
    spans = [rng.randint(min_span, max_span)]
    for _ in range(length - 1):
        lo = max(min_span, spans[-1] - 1)
        hi = min(max_span, spans[-1] + 1)
        spans.append(rng.randint(lo, hi))
    if len(set(spans)) == 1 and max_span > min_span:
        idx = rng.randrange(length)
        if spans[idx] == min_span:
            spans[idx] += 1
        elif spans[idx] == max_span:
            spans[idx] -= 1
        else:
            spans[idx] += rng.choice([-1, 1])
    return spans


def _centered_vertical_mask(height: int, even_axis: bool, spans: list[int]) -> np.ndarray:
    if even_axis:
        width = 2 * max(spans)
    else:
        width = 2 * max(spans) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, span in enumerate(spans):
        if even_axis:
            row_width = 2 * int(span)
            left = (width - row_width) // 2
        else:
            row_width = 2 * int(span) + 1
            left = (width - row_width) // 2
        mask[r, left : left + row_width] = True
    return mask


def _add_side_defect(mask: np.ndarray, side: str, rng: random.Random) -> np.ndarray:
    coords = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    rows = sorted({r for r, _ in coords})
    candidates: list[tuple[int, int]] = []
    for r in rows:
        cols = sorted(c for rr, c in coords if rr == r)
        left = cols[0]
        right = cols[-1]
        if side == "left":
            candidates.append((int(r), int(left - 1)))
        else:
            candidates.append((int(r), int(right + 1)))
    extra = rng.choice(candidates)
    coords.add((int(extra[0]), int(extra[1])))
    return _coords_to_mask(coords)


def _make_object_mask(direction: str, rng: random.Random) -> np.ndarray | None:
    for _ in range(OBJECT_RETRIES):
        height = rng.randint(*PROFILE_HEIGHT_RANGE)
        max_span = rng.randint(*PROFILE_SPAN_RANGE)
        even_axis = bool(rng.getrandbits(1))
        min_span = 1 if even_axis else 0
        spans = _random_profile(height, min_span, max_span, rng)
        base = _centered_vertical_mask(height, even_axis, spans)

        if direction in ("left", "right"):
            mask = _add_side_defect(base, direction, rng)
        else:
            transposed = _add_side_defect(base, "left" if direction == "up" else "right", rng)
            mask = transposed.T

        area = int(mask.sum())
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]):
            continue

        try:
            info = _analyze_mask(mask)
        except AssertionError:
            continue
        if str(info["direction"]) != direction:
            continue
        if direction in ("left", "right") and str(info["orientation"]) != "vertical":
            continue
        if direction in ("up", "down") and str(info["orientation"]) != "horizontal":
            continue
        if int(info["mismatch"]) <= 0:
            continue
        if int(info["mismatch"]) >= int(info["other_mismatch"]):
            continue
        return mask
    return None


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _placement_candidates(
    mask: np.ndarray,
    direction: str,
    shape: tuple[int, int],
    input_blocked: np.ndarray,
    output_blocked: np.ndarray,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = mask.shape
    grid_h, grid_w = shape
    local = _mask_coords(mask)
    area = int(mask.sum())
    dr, dc = DIR_VECTORS[direction]
    unclipped: list[tuple[int, int]] = []
    clipped: list[tuple[int, int]] = []

    for r0 in range(grid_h - h + 1):
        for c0 in range(grid_w - w + 1):
            abs_coords = [(r0 + r, c0 + c) for r, c in local]
            if any(bool(input_blocked[r, c]) for r, c in abs_coords):
                continue

            moved = [(r + dr * area, c + dc * area) for r, c in abs_coords]
            visible = [(r, c) for r, c in moved if 0 <= r < grid_h and 0 <= c < grid_w]
            if not visible:
                continue
            if any(bool(output_blocked[r, c]) for r, c in visible):
                continue

            if len(visible) == len(abs_coords):
                unclipped.append((int(r0), int(c0)))
            else:
                clipped.append((int(r0), int(c0)))

    return unclipped, clipped


def _place_objects(
    objects: list[dict[str, object]],
    shape: tuple[int, int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    grid_h, grid_w = shape
    for _ in range(PLACEMENT_RETRIES):
        input_grid = np.full((grid_h, grid_w), BG, dtype=int)
        output_grid = np.full((grid_h, grid_w), BG, dtype=int)
        input_occ = np.zeros((grid_h, grid_w), dtype=bool)
        output_occ = np.zeros((grid_h, grid_w), dtype=bool)
        input_blocked = np.zeros((grid_h, grid_w), dtype=bool)
        output_blocked = np.zeros((grid_h, grid_w), dtype=bool)

        order = list(range(len(objects)))
        rng.shuffle(order)
        order.sort(key=lambda idx: (int(objects[idx]["area"]), int(objects[idx]["mask"].shape[0] * objects[idx]["mask"].shape[1])), reverse=True)

        placed = True
        for idx in order:
            obj = objects[idx]
            want_clipped = bool(obj["want_clipped"])
            mask = np.asarray(obj["mask"], dtype=bool)
            direction = str(obj["direction"])
            color = int(obj["color"])
            area = int(obj["area"])
            local = _mask_coords(mask)
            dr, dc = DIR_VECTORS[direction]

            unclipped, clipped = _placement_candidates(mask, direction, shape, input_blocked, output_blocked)
            choices = clipped if want_clipped and clipped else unclipped if (not want_clipped and unclipped) else clipped + unclipped
            if not choices:
                placed = False
                break

            r0, c0 = rng.choice(choices)
            abs_coords = [(int(r0) + r, int(c0) + c) for r, c in local]
            moved = [(r + dr * area, c + dc * area) for r, c in abs_coords]
            visible = [(r, c) for r, c in moved if 0 <= r < grid_h and 0 <= c < grid_w]

            _paint_coords(input_grid, abs_coords, color)
            _paint_coords(output_grid, visible, color)
            for r, c in abs_coords:
                input_occ[r, c] = True
            for r, c in visible:
                output_occ[r, c] = True
            input_blocked = _dilate(input_occ)
            output_blocked = _dilate(output_occ)

        if placed:
            return input_grid, output_grid
    return None


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    object_count = rng.randint(*OBJECT_COUNT_RANGE)
    if object_count > len(PALETTE):
        return None

    colors = rng.sample(PALETTE, object_count)
    directions = [rng.choice(["up", "down", "left", "right"]) for _ in range(object_count)]
    objects: list[dict[str, object]] = []

    for color, direction in zip(colors, directions):
        mask = _make_object_mask(direction, rng)
        if mask is None:
            return None
        objects.append(
            {
                "color": int(color),
                "direction": direction,
                "mask": mask,
                "area": int(mask.sum()),
                "want_clipped": rng.random() < CLIP_PROB,
            }
        )

    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    placed = _place_objects(objects, (grid_h, grid_w), rng)
    if placed is None:
        return None

    inp, expected = placed
    return inp, expected


def generate_input() -> np.ndarray:
    for attempt in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30) + attempt * 1009)
        built = _build_case(rng)
        if built is None:
            continue
        inp, expected = built
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
