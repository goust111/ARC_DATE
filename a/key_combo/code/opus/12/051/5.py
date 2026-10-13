from __future__ import annotations

import random
from collections import deque

import numpy as np


FRAME_TOP = 1
FRAME_RIGHT = 2
FRAME_BOTTOM = 3
FRAME_LEFT = 4
FRAME_CORNER = 5
BACKGROUND = 0

SIDE_TO_COLOR = {
    "top": FRAME_TOP,
    "right": FRAME_RIGHT,
    "bottom": FRAME_BOTTOM,
    "left": FRAME_LEFT,
}
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORTHOGONAL_DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_SIZE_RANGE = (17, 21)
OBJECT_COUNT_RANGE = (3, 5)
PATCH_SIZE = 3
MAX_GENERATION_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 250


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_bfs(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_frame(g: np.ndarray) -> tuple[int, int, int, int]:
    corners = np.argwhere(g == FRAME_CORNER)
    assert len(corners) == 4, "expected exactly four grey frame corners"
    rows = sorted(set(map(int, corners[:, 0])))
    cols = sorted(set(map(int, corners[:, 1])))
    assert len(rows) == 2 and len(cols) == 2, "grey corners must form a rectangle"
    top, bottom = rows
    left, right = cols
    expected = {(top, left), (top, right), (bottom, left), (bottom, right)}
    actual = {tuple(map(int, rc)) for rc in corners}
    assert actual == expected, "grey corners must occupy the frame rectangle corners"
    return top, left, bottom, right


def _gap_sets(g: np.ndarray, frame: tuple[int, int, int, int]) -> dict[str, set[int]]:
    top, left, bottom, right = frame
    return {
        "top": {c for c in range(left + 1, right) if int(g[top, c]) == BACKGROUND},
        "bottom": {c for c in range(left + 1, right) if int(g[bottom, c]) == BACKGROUND},
        "left": {r for r in range(top + 1, bottom) if int(g[r, left]) == BACKGROUND},
        "right": {r for r in range(top + 1, bottom) if int(g[r, right]) == BACKGROUND},
    }


def _extract_objects(g: np.ndarray, frame: tuple[int, int, int, int]) -> list[dict]:
    top, left, bottom, right = frame
    mask = (g != BACKGROUND) & (g != FRAME_CORNER)
    comps = _component_bfs(mask)
    objects: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        if r0 <= top or c0 <= left or r1 >= bottom or c1 >= right:
            continue
        if area != 9 or (r1 - r0 + 1) != PATCH_SIZE or (c1 - c0 + 1) != PATCH_SIZE:
            continue
        colors = tuple(sorted(set(map(int, g[coords[:, 0], coords[:, 1]]))))
        assert len(colors) == 2, "each interior object must use exactly two colors"
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "colors": colors,
            }
        )
    objects.sort(key=lambda obj: obj["center"])
    return objects


def _nearest_gap_side(
    center: tuple[int, int],
    frame: tuple[int, int, int, int],
    gaps: dict[str, set[int]],
) -> str:
    r, c = center
    candidates = []
    if c in gaps["top"]:
        candidates.append("top")
    if c in gaps["bottom"]:
        candidates.append("bottom")
    if r in gaps["left"]:
        candidates.append("left")
    if r in gaps["right"]:
        candidates.append("right")
    assert len(candidates) == 1, "each object must have exactly one nearest gap"
    return candidates[0]


def _analyze_scene(g: np.ndarray) -> tuple[tuple[int, int, int, int], dict[str, set[int]], list[dict]]:
    frame = _detect_frame(g)
    top, left, bottom, right = frame
    assert top + 2 < bottom and left + 2 < right, "frame interior must fit 3x3 objects"

    gaps = _gap_sets(g, frame)
    gap_total = sum(len(v) for v in gaps.values())
    assert 3 <= gap_total <= 5, "expected 3-5 border gaps"

    for c in range(left + 1, right):
        value = int(g[top, c])
        assert value in {BACKGROUND, FRAME_TOP}, "top side must contain only blue cells and gaps"
    for c in range(left + 1, right):
        value = int(g[bottom, c])
        assert value in {BACKGROUND, FRAME_BOTTOM}, "bottom side must contain only green cells and gaps"
    for r in range(top + 1, bottom):
        value = int(g[r, left])
        assert value in {BACKGROUND, FRAME_LEFT}, "left side must contain only yellow cells and gaps"
    for r in range(top + 1, bottom):
        value = int(g[r, right])
        assert value in {BACKGROUND, FRAME_RIGHT}, "right side must contain only red cells and gaps"

    objects = _extract_objects(g, frame)
    assert 3 <= len(objects) <= 5, "expected 3-5 interior objects"
    seen_rows = set()
    seen_cols = set()
    for obj in objects:
        r, c = obj["center"]
        assert r not in seen_rows and c not in seen_cols, "object centers must be row/col distinct"
        seen_rows.add(r)
        seen_cols.add(c)
        obj["side"] = _nearest_gap_side(obj["center"], frame, gaps)
    return frame, gaps, objects


def _swap_object_colors(out: np.ndarray, coords: np.ndarray, a: int, b: int) -> None:
    values = out[coords[:, 0], coords[:, 1]].copy()
    mask_a = values == a
    mask_b = values == b
    out[coords[mask_a, 0], coords[mask_a, 1]] = b
    out[coords[mask_b, 0], coords[mask_b, 1]] = a


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame gap and use its side color to gate whether the nearest 3x3 two-color object swaps colors.

    Input structure:
    - A black-background grid contains one 1-cell-thick rectangular frame with grey corner pixels.
    - The top, right, bottom, and left frame sides are blue, red, green, and yellow respectively, with 3-5 single-cell black gaps on the sides.
    - Inside the frame are 3-5 separated 3x3 foreground objects, each using exactly two colors.

    Rule:
    - Recolor every frame gap with the color of the side it lies on, restoring the border.
    - For each 3x3 object, scan from its center straight up, down, left, and right; exactly one direction reaches a frame gap first, defining the object's nearest gap side.
    - Let F be that side color. If F matches either color used by the object, leave the object unchanged; otherwise swap the object's two colors in place.
    - Preserve all other pixels, including the restored frame and grey corners.

    Key insight:
    - The filled gap is not just border repair: its side color is a gate signal, and solving requires combining direction-to-side detection with per-object color matching before deciding whether to swap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frame, gaps, objects = _analyze_scene(g)
    top, left, bottom, right = frame

    out = g.copy()
    for c in gaps["top"]:
        out[top, c] = FRAME_TOP
    for c in gaps["bottom"]:
        out[bottom, c] = FRAME_BOTTOM
    for r in gaps["left"]:
        out[r, left] = FRAME_LEFT
    for r in gaps["right"]:
        out[r, right] = FRAME_RIGHT

    for obj in objects:
        a, b = obj["colors"]
        fill_color = SIDE_TO_COLOR[obj["side"]]
        if fill_color not in {a, b}:
            _swap_object_colors(out, obj["coords"], a, b)

    return out


def _choose_centers(height: int, width: int, n_objects: int) -> list[tuple[int, int]] | None:
    row_pool = list(range(3, height - 3))
    col_pool = list(range(3, width - 3))
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        rows = random.sample(row_pool, n_objects)
        cols = random.sample(col_pool, n_objects)
        random.shuffle(cols)
        centers = list(zip(rows, cols))
        ok = True
        for i, (r1, c1) in enumerate(centers):
            for r2, c2 in centers[i + 1 :]:
                if abs(r1 - r2) <= 2 and abs(c1 - c2) <= 2:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return centers
    return None


def _random_patch(color_a: int, color_b: int) -> np.ndarray:
    minority_count = random.randint(2, 4)
    minority_color, majority_color = (color_a, color_b) if random.random() < 0.5 else (color_b, color_a)
    flat = [majority_color] * 9
    for idx in random.sample(range(9), minority_count):
        flat[idx] = minority_color
    patch = np.array(flat, dtype=int).reshape(3, 3)
    rows_mixed = sum(len(set(map(int, patch[r, :]))) == 2 for r in range(3))
    cols_mixed = sum(len(set(map(int, patch[:, c]))) == 2 for c in range(3))
    if rows_mixed + cols_mixed < 2:
        return _random_patch(color_a, color_b)
    return patch


def _swap_patch(patch: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    swapped = patch.copy()
    mask_a = patch == color_a
    mask_b = patch == color_b
    swapped[mask_a] = color_b
    swapped[mask_b] = color_a
    return swapped


def _pick_object_sides(n_objects: int) -> list[str]:
    sides = [random.choice(tuple(SIDE_TO_COLOR)) for _ in range(n_objects)]
    if len(set(sides)) < 2:
        sides[random.randrange(n_objects)] = random.choice([s for s in SIDE_TO_COLOR if s != sides[0]])
    return sides


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    n_objects = random.randint(*OBJECT_COUNT_RANGE)
    centers = _choose_centers(height, width, n_objects)
    if centers is None:
        return None

    sides = _pick_object_sides(n_objects)
    gated_flags = [True, False] + [random.random() < 0.5 for _ in range(n_objects - 2)]
    random.shuffle(gated_flags)
    tricky_swap_index = random.choice([i for i, gated in enumerate(gated_flags) if not gated])

    inp = np.full((height, width), BACKGROUND, dtype=int)
    out = np.full((height, width), BACKGROUND, dtype=int)
    top, left, bottom, right = 1, 1, height - 2, width - 2

    inp[top, left] = inp[top, right] = inp[bottom, left] = inp[bottom, right] = FRAME_CORNER
    out[top, left] = out[top, right] = out[bottom, left] = out[bottom, right] = FRAME_CORNER
    inp[top, left + 1 : right] = FRAME_TOP
    inp[bottom, left + 1 : right] = FRAME_BOTTOM
    inp[top + 1 : bottom, left] = FRAME_LEFT
    inp[top + 1 : bottom, right] = FRAME_RIGHT
    out[top, left + 1 : right] = FRAME_TOP
    out[bottom, left + 1 : right] = FRAME_BOTTOM
    out[top + 1 : bottom, left] = FRAME_LEFT
    out[top + 1 : bottom, right] = FRAME_RIGHT

    records: list[dict] = []
    for index, ((r, c), side, gated) in enumerate(zip(centers, sides, gated_flags)):
        fill_color = SIDE_TO_COLOR[side]
        if side == "top":
            inp[top, c] = BACKGROUND
        elif side == "bottom":
            inp[bottom, c] = BACKGROUND
        elif side == "left":
            inp[r, left] = BACKGROUND
        else:
            inp[r, right] = BACKGROUND

        if gated:
            other = random.choice([color for color in PALETTE if color != fill_color])
            colors = (fill_color, other)
        else:
            if index == tricky_swap_index:
                frame_other = random.choice([color for color in (1, 2, 3, 4) if color != fill_color])
                accent_other = random.choice([color for color in PALETTE if color not in {fill_color, frame_other}])
                colors = (frame_other, accent_other)
            else:
                colors = tuple(random.sample([color for color in PALETTE if color != fill_color], 2))

        patch = _random_patch(*colors)
        expected_patch = patch if gated else _swap_patch(patch, *colors)
        r0, c0 = r - 1, c - 1
        inp[r0 : r0 + 3, c0 : c0 + 3] = patch
        out[r0 : r0 + 3, c0 : c0 + 3] = expected_patch
        records.append(
            {
                "center": (r, c),
                "side": side,
                "fill_color": fill_color,
                "colors": tuple(sorted(map(int, colors))),
                "gated": bool(gated),
            }
        )

    return inp, out, records


def _validate_instance(inp: np.ndarray, expected: np.ndarray, records: list[dict]) -> bool:
    try:
        frame, gaps, objects = _analyze_scene(inp)
    except AssertionError:
        return False

    if len(objects) != len(records):
        return False
    record_by_center = {tuple(rec["center"]): rec for rec in records}
    if set(record_by_center) != {tuple(obj["center"]) for obj in objects}:
        return False

    gated_count = 0
    swapped_count = 0
    tricky_swap_seen = False
    for obj in objects:
        rec = record_by_center[tuple(obj["center"])]
        if obj["side"] != rec["side"]:
            return False
        if tuple(sorted(obj["colors"])) != rec["colors"]:
            return False
        fill_color = SIDE_TO_COLOR[obj["side"]]
        actual_gated = fill_color in obj["colors"]
        if actual_gated != rec["gated"]:
            return False
        if actual_gated:
            gated_count += 1
        else:
            swapped_count += 1
            if any(color in {1, 2, 3, 4} for color in obj["colors"]):
                tricky_swap_seen = True

    if gated_count < 1 or swapped_count < 1 or not tricky_swap_seen:
        return False
    if sum(len(v) for v in gaps.values()) != len(objects):
        return False
    if np.array_equal(inp, expected):
        return False

    try:
        actual = main(inp)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, expected, records = built
        if _validate_instance(inp, expected, records):
            return inp
    raise RuntimeError("failed to generate a valid gated color-swap frame puzzle")
