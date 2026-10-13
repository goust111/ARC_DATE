from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
ARM_COLORS = (1, 2, 3, 4)
CENTER_COLOR = 5
MARKER_COLOR = 6

PLUS_MASK = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)

ARM_DELTAS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

MARKER_OFFSETS = {
    "left": [(-1, -3), (-1, -2), (0, -3), (0, -2)],
    "right": [(-1, 2), (-1, 3), (0, 2), (0, 3)],
    "top": [(-3, -1), (-3, 0), (-2, -1), (-2, 0)],
    "bottom": [(2, -1), (2, 0), (3, -1), (3, 0)],
}

SIDE_TO_MOVEMENT = {
    "left": (0, 1),
    "right": (0, -1),
    "top": (1, 0),
    "bottom": (-1, 0),
}

ROTATE_180_ARMS = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}

GEN_HEIGHT_RANGE = (14, 18)
GEN_WIDTH_RANGE = (14, 18)
GEN_OBJECT_COUNT = (2, 3)
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(counts.most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _arm_colors_at(grid: np.ndarray, center: tuple[int, int]) -> dict[str, int]:
    r, c = center
    return {
        name: int(grid[r + dr, c + dc])
        for name, (dr, dc) in ARM_DELTAS.items()
    }


def _marker_cells(center: tuple[int, int], side: str) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in MARKER_OFFSETS[side]]


def _cross_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _composite_cells(center: tuple[int, int], side: str) -> set[tuple[int, int]]:
    return set(_cross_cells(center)) | set(_marker_cells(center, side))


def _detect_marker_side(grid: np.ndarray, center: tuple[int, int]) -> str:
    h, w = grid.shape
    sides = []
    for side, offsets in MARKER_OFFSETS.items():
        cells = []
        ok = True
        for dr, dc in offsets:
            rr = center[0] + dr
            cc = center[1] + dc
            if not _in_bounds((h, w), rr, cc):
                ok = False
                break
            cells.append((rr, cc))
        if ok and all(int(grid[r, c]) == MARKER_COLOR for r, c in cells):
            sides.append(side)
    assert len(sides) == 1, "each cross must have exactly one 2x2 marker block"
    return sides[0]


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    objects = []
    covered: set[tuple[int, int]] = set()

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            if not np.array_equal(patch != bg, PLUS_MASK):
                continue
            if np.any(patch[PLUS_MASK] == bg) or np.any(patch[PLUS_MASK] == MARKER_COLOR):
                continue
            if int(patch[1, 1]) != CENTER_COLOR:
                continue

            center = (r, c)
            side = _detect_marker_side(g, center)
            cells = _composite_cells(center, side)
            assert all(_in_bounds((h, w), rr, cc) for rr, cc in cells)
            assert all(int(g[rr, cc]) != bg for rr, cc in cells)
            assert covered.isdisjoint(cells), "objects must not overlap"
            covered |= cells
            objects.append(
                {
                    "input_center": center,
                    "side": side,
                    "center_color": CENTER_COLOR,
                    "input_arms": _arm_colors_at(g, center),
                }
            )

    assert objects, "expected at least one marked cross object"
    assert covered == {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}, "all foreground must belong to objects"
    return objects


def _final_center(shape: tuple[int, int], center: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = shape
    r, c = center
    if side == "left":
        return r, w - 2
    if side == "right":
        return r, 1
    if side == "top":
        return h - 2, c
    return 1, c


def _rotated_arms(arms: dict[str, int]) -> dict[str, int]:
    return {name: int(arms[ROTATE_180_ARMS[name]]) for name in ARM_DELTAS}


def _nearest_boundaries(shape: tuple[int, int], center: tuple[int, int]) -> tuple[str, str]:
    h, w = shape
    r, c = center
    up, down = r, h - 1 - r
    left, right = c, w - 1 - c
    assert up != down and left != right, "nearest boundaries must be unique"
    horiz = "top" if up < down else "bottom"
    vert = "left" if left < right else "right"
    return horiz, vert


def _draw_ray(
    out: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    color: int,
    bg: int,
) -> tuple[int, bool]:
    r = start[0] + step[0]
    c = start[1] + step[1]
    drawn = 0
    blocked = False
    while _in_bounds(out.shape, r, c):
        if int(out[r, c]) != bg:
            blocked = True
            break
        out[r, c] = int(color)
        drawn += 1
        r += step[0]
        c += step[1]
    return drawn, blocked


def _paint_object(out: np.ndarray, center: tuple[int, int], side: str, arms: dict[str, int]) -> None:
    r, c = center
    assert int(out[r, c]) == BACKGROUND
    out[r, c] = CENTER_COLOR
    for name, (dr, dc) in ARM_DELTAS.items():
        rr = r + dr
        cc = c + dc
        assert int(out[rr, cc]) == BACKGROUND
        out[rr, cc] = int(arms[name])
    for rr, cc in _marker_cells(center, side):
        assert int(out[rr, cc]) == BACKGROUND
        out[rr, cc] = MARKER_COLOR


def _simulate(objects: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, list[dict]]:
    out = np.full(shape, BACKGROUND, dtype=int)
    placed = []
    for obj in objects:
        rotated = _rotated_arms(obj["input_arms"])
        final_center = _final_center(shape, obj["input_center"], obj["side"])
        placed.append(
            {
                "side": obj["side"],
                "final_center": final_center,
                "rotated_arms": rotated,
                "input_arms": obj["input_arms"],
            }
        )
        _paint_object(out, final_center, obj["side"], rotated)

    placed.sort(key=lambda item: (item["final_center"][0], item["final_center"][1]))

    ray_infos = []
    for obj in placed:
        center = obj["final_center"]
        horiz, vert = _nearest_boundaries(shape, center)
        total_drawn = 0
        any_blocked = False
        for direction in (horiz, vert):
            dr, dc = ARM_DELTAS[direction]
            start = (center[0] + dr, center[1] + dc)
            drawn, blocked = _draw_ray(out, start, (dr, dc), obj["rotated_arms"][direction], BACKGROUND)
            total_drawn += drawn
            any_blocked = any_blocked or blocked
        ray_infos.append(
            {
                "center": center,
                "horiz": horiz,
                "vert": vert,
                "drawn": total_drawn,
                "blocked": any_blocked,
            }
        )
    return out, ray_infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked cross away from its marker, rotate the cross 180 degrees, and project only the arms that face the nearest horizontal and vertical boundaries.

    Input structure:
    - The grid has black background.
    - Each object is a colored 5-pixel plus with a fixed grey center and a 2x2 magenta marker block attached on exactly one side.
    - The marker side indicates the opposite slide direction, and objects are separated from each other.

    Rule:
    - For each object, move the whole composite straight away from the marker until the composite reaches the corresponding grid edge.
    - Rotate only the 3x3 plus by 180 degrees; keep the 2x2 marker block unchanged relative to the object.
    - After sliding, find the unique nearest horizontal boundary and the unique nearest vertical boundary from the moved center.
    - From the rotated arm pointing toward each of those two boundaries, extend a ray from the arm tip outward until the boundary or until an already drawn non-background pixel blocks it.
    - Place all moved objects first, then process the ray drawing in row-major order of the moved centers.

    Key insight:
    - Sliding decides which boundaries are nearest, while the 180-degree rotation swaps opposite arm colors, so the final projected colors depend on both steps together.
    """
    objects = _find_objects(np.asarray(grid, dtype=int))
    out, _ = _simulate(objects, np.asarray(grid, dtype=int).shape)
    return out


def _inner_positions(length: int) -> list[int]:
    return [p for p in range(4, length - 4) if 2 * p != length - 1]


def _half_positions(length: int, half: str) -> list[int]:
    inner = _inner_positions(length)
    if half in {"top", "left"}:
        return [p for p in inner if 2 * p < length - 1]
    return [p for p in inner if 2 * p > length - 1]


def _random_final_arms() -> dict[str, int]:
    top, bottom = random.sample(ARM_COLORS, 2)
    left, right = random.sample(ARM_COLORS, 2)
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def _render_input_object(grid: np.ndarray, center: tuple[int, int], side: str, arms: dict[str, int]) -> None:
    r, c = center
    assert int(grid[r, c]) == BACKGROUND
    grid[r, c] = CENTER_COLOR
    for name, (dr, dc) in ARM_DELTAS.items():
        rr = r + dr
        cc = c + dc
        assert int(grid[rr, cc]) == BACKGROUND
        grid[rr, cc] = int(arms[name])
    for rr, cc in _marker_cells(center, side):
        assert int(grid[rr, cc]) == BACKGROUND
        grid[rr, cc] = MARKER_COLOR


def _can_place(center: tuple[int, int], side: str, occupied: np.ndarray) -> bool:
    cells = _composite_cells(center, side)
    h, w = occupied.shape
    for r, c in cells:
        if not _in_bounds((h, w), r, c):
            return False
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if np.any(occupied[r0:r1, c0:c1]):
            return False
    return True


def _occupy(center: tuple[int, int], side: str, occupied: np.ndarray) -> None:
    for r, c in _composite_cells(center, side):
        occupied[r, c] = True


def _corner_pair_specs(h: int, w: int, corner: str) -> list[dict] | None:
    if corner == "TL":
        side_h, half_r = "right", "top"
        side_v, half_c = "bottom", "left"
    elif corner == "TR":
        side_h, half_r = "left", "top"
        side_v, half_c = "bottom", "right"
    elif corner == "BL":
        side_h, half_r = "right", "bottom"
        side_v, half_c = "top", "left"
    else:
        side_h, half_r = "left", "bottom"
        side_v, half_c = "top", "right"

    row_candidates = _half_positions(h, half_r)
    col_candidates = _half_positions(w, half_c)
    pairs = [(r, c) for r in row_candidates for c in col_candidates]
    random.shuffle(pairs)

    for row, col in pairs:
        horizontal = {"side": side_h, "final_center": (row, 1 if side_h == "right" else w - 2)}
        vertical = {"side": side_v, "final_center": (1 if side_v == "bottom" else h - 2, col)}
        occupied = np.zeros((h, w), dtype=bool)
        if not _can_place(horizontal["final_center"], horizontal["side"], occupied):
            continue
        _occupy(horizontal["final_center"], horizontal["side"], occupied)
        if not _can_place(vertical["final_center"], vertical["side"], occupied):
            continue
        return [horizontal, vertical]
    return None


def _random_extra_spec(h: int, w: int, occupied: np.ndarray) -> dict | None:
    choices = []
    for side in ("left", "right", "top", "bottom"):
        if side in {"left", "right"}:
            final_c = w - 2 if side == "left" else 1
            for row in _inner_positions(h):
                center = (row, final_c)
                if _can_place(center, side, occupied):
                    choices.append({"side": side, "final_center": center})
        else:
            final_r = h - 2 if side == "top" else 1
            for col in _inner_positions(w):
                center = (final_r, col)
                if _can_place(center, side, occupied):
                    choices.append({"side": side, "final_center": center})
    if not choices:
        return None
    return random.choice(choices)


def _input_center_from_final(final_center: tuple[int, int], side: str, h: int, w: int) -> tuple[int, int]:
    row, col = final_center
    if side == "left":
        return row, random.choice(_inner_positions(w))
    if side == "right":
        return row, random.choice(_inner_positions(w))
    if side == "top":
        return random.choice(_inner_positions(h)), col
    return random.choice(_inner_positions(h)), col


def _build_candidate() -> tuple[np.ndarray, np.ndarray, list[dict], list[dict]] | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    count = random.randint(*GEN_OBJECT_COUNT)
    corner = random.choice(("TL", "TR", "BL", "BR"))

    specs = _corner_pair_specs(h, w, corner)
    if specs is None:
        return None

    occupied_final = np.zeros((h, w), dtype=bool)
    for spec in specs:
        _occupy(spec["final_center"], spec["side"], occupied_final)

    if count == 3:
        extra = _random_extra_spec(h, w, occupied_final)
        if extra is None:
            return None
        specs.append(extra)
        _occupy(extra["final_center"], extra["side"], occupied_final)

    occupied_input = np.zeros((h, w), dtype=bool)
    objects = []
    for spec in specs:
        final_arms = _random_final_arms()
        input_arms = _rotated_arms(final_arms)
        placed = False
        for _ in range(80):
            input_center = _input_center_from_final(spec["final_center"], spec["side"], h, w)
            if _can_place(input_center, spec["side"], occupied_input):
                _occupy(input_center, spec["side"], occupied_input)
                objects.append(
                    {
                        "side": spec["side"],
                        "final_center": spec["final_center"],
                        "input_center": input_center,
                        "final_arms": final_arms,
                        "input_arms": input_arms,
                    }
                )
                placed = True
                break
        if not placed:
            return None

    inp = np.full((h, w), BACKGROUND, dtype=int)
    for obj in objects:
        _render_input_object(inp, obj["input_center"], obj["side"], obj["input_arms"])

    parsed = _find_objects(inp)
    expected, ray_infos = _simulate(parsed, inp.shape)
    meta_by_side_final = {
        (obj["side"], obj["final_center"]): obj
        for obj in objects
    }
    for obj in parsed:
        final_center = _final_center(inp.shape, obj["input_center"], obj["side"])
        built = meta_by_side_final[(obj["side"], final_center)]
        if obj["input_arms"] != built["input_arms"]:
            return None

    return inp, expected, objects, ray_infos


def _validate_candidate(inp: np.ndarray, expected: np.ndarray, objects: list[dict], ray_infos: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if len(objects) not in {2, 3}:
        return False

    parsed = _find_objects(inp)
    if len(parsed) != len(objects):
        return False

    seen_horizontal = any(obj["side"] in {"left", "right"} for obj in objects)
    seen_vertical = any(obj["side"] in {"top", "bottom"} for obj in objects)
    if not (seen_horizontal and seen_vertical):
        return False

    if not any(info["blocked"] for info in ray_infos):
        return False
    if any(info["drawn"] <= 0 for info in ray_infos):
        return False

    expected_again = main(inp)
    if not np.array_equal(expected_again, expected):
        return False

    for obj in parsed:
        final_center = _final_center(inp.shape, obj["input_center"], obj["side"])
        horiz, vert = _nearest_boundaries(inp.shape, final_center)
        rotated = _rotated_arms(obj["input_arms"])
        if rotated[horiz] == obj["input_arms"][horiz]:
            return False
        if rotated[vert] == obj["input_arms"][vert]:
            return False

    diff = int(np.count_nonzero(expected != inp))
    return diff >= 8


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        inp, expected, objects, ray_infos = built
        if _validate_candidate(inp, expected, objects, ray_infos):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
