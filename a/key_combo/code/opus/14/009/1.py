from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
BODY_COLORS = (1, 3, 4)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

MIN_GRID_SIDE_BY_COUNT = {3: 18, 4: 20, 5: 22}
MAX_GRID_SIDE = 26
GEN_RETRIES = 400
OBJECT_RETRIES = 200
PLACEMENT_RETRIES = 200


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    view[mask] = patch[mask]


def _rand_lengths(span: int, max_extent: int) -> list[int]:
    for _ in range(OBJECT_RETRIES):
        lengths = [random.randint(1, max_extent)]
        for _idx in range(1, span):
            prev = lengths[-1]
            options = [v for v in (prev - 1, prev, prev + 1) if 1 <= v <= max_extent]
            lengths.append(random.choice(options))
        if max(lengths) != max_extent:
            continue
        if len(set(lengths)) == 1:
            continue
        area = sum(lengths)
        if span + 2 <= area <= 12:
            return lengths
    raise RuntimeError("failed to generate a non-degenerate major half")


def _damaged_lengths(major: list[int]) -> list[int]:
    total = sum(major)
    min_total = len(major)
    target = random.randint(max(min_total, total // 2), total - 1)
    minor = list(major)
    remove = total - target
    while remove > 0:
        choices = [idx for idx, value in enumerate(minor) if value > 1]
        if not choices:
            break
        idx = random.choice(choices)
        minor[idx] -= 1
        remove -= 1
    if sum(minor) >= total:
        idx = random.choice([i for i, value in enumerate(minor) if value > 1])
        minor[idx] -= 1
    return minor


def _make_vertical_object() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(OBJECT_RETRIES):
        height = random.randint(3, 5)
        extent = random.choice([k for k in range(2, 5) if 2 * k + 1 > height])
        major = _rand_lengths(height, extent)
        minor = _damaged_lengths(major)
        input_area = height + sum(major) + sum(minor)
        if not (10 <= input_area <= 20):
            continue

        body = random.choice(BODY_COLORS)
        side = random.choice(("left", "right"))
        axis = extent
        patch_in = np.zeros((height, 2 * extent + 1), dtype=int)
        patch_out = np.zeros_like(patch_in)
        patch_in[:, axis] = RED

        for r, (major_len, minor_len) in enumerate(zip(major, minor)):
            if side == "left":
                patch_in[r, axis - major_len : axis] = body
                patch_in[r, axis + 1 : axis + 1 + minor_len] = body
            else:
                patch_in[r, axis - minor_len : axis] = body
                patch_in[r, axis + 1 : axis + 1 + major_len] = body
            patch_out[r, axis - major_len : axis] = body
            patch_out[r, axis + 1 : axis + 1 + major_len] = body
        return patch_in, patch_out
    raise RuntimeError("failed to build a vertical fractured object")


def _make_horizontal_object() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(OBJECT_RETRIES):
        width = random.randint(3, 5)
        extent = random.choice([k for k in range(2, 5) if 2 * k + 1 > width])
        major = _rand_lengths(width, extent)
        minor = _damaged_lengths(major)
        input_area = width + sum(major) + sum(minor)
        if not (10 <= input_area <= 20):
            continue

        body = random.choice(BODY_COLORS)
        side = random.choice(("top", "bottom"))
        axis = extent
        patch_in = np.zeros((2 * extent + 1, width), dtype=int)
        patch_out = np.zeros_like(patch_in)
        patch_in[axis, :] = RED

        for c, (major_len, minor_len) in enumerate(zip(major, minor)):
            if side == "top":
                patch_in[axis - major_len : axis, c] = body
                patch_in[axis + 1 : axis + 1 + minor_len, c] = body
            else:
                patch_in[axis - minor_len : axis, c] = body
                patch_in[axis + 1 : axis + 1 + major_len, c] = body
            patch_out[axis - major_len : axis, c] = body
            patch_out[axis + 1 : axis + 1 + major_len, c] = body
        return patch_in, patch_out
    raise RuntimeError("failed to build a horizontal fractured object")


def _make_object(orientation: str) -> tuple[np.ndarray, np.ndarray]:
    if orientation == "vertical":
        return _make_vertical_object()
    if orientation == "horizontal":
        return _make_horizontal_object()
    raise ValueError(f"unknown orientation: {orientation}")


def _can_place(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = patch.shape
    if r0 < 0 or c0 < 0 or r0 + h > blocked.shape[0] or c0 + w > blocked.shape[1]:
        return False
    coords = np.argwhere(patch != BLACK)
    return all(not blocked[r0 + int(dr), c0 + int(dc)] for dr, dc in coords)


def _mark_blocked(blocked: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(patch != BLACK):
        rr = r0 + int(dr)
        cc = c0 + int(dc)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                blocked[nr, nc] = True


def _attempt_grid() -> tuple[np.ndarray, np.ndarray] | None:
    count = random.randint(3, 5)
    grid_h = random.randint(MIN_GRID_SIDE_BY_COUNT[count], MAX_GRID_SIDE)
    grid_w = random.randint(MIN_GRID_SIDE_BY_COUNT[count], MAX_GRID_SIDE)

    n_vertical = random.randint(1, count - 1)
    orientations = ["vertical"] * n_vertical + ["horizontal"] * (count - n_vertical)
    random.shuffle(orientations)

    objects = []
    for orientation in orientations:
        patch_in, patch_out = _make_object(orientation)
        objects.append((patch_in, patch_out))
    objects.sort(key=lambda item: int(np.count_nonzero(item[1])), reverse=True)

    grid_in = np.zeros((grid_h, grid_w), dtype=int)
    grid_out = np.zeros_like(grid_in)
    blocked = np.zeros_like(grid_in, dtype=bool)

    for patch_in, patch_out in objects:
        ph, pw = patch_out.shape
        positions = [(r, c) for r in range(grid_h - ph + 1) for c in range(grid_w - pw + 1)]
        random.shuffle(positions)

        placed = False
        for r0, c0 in positions[:PLACEMENT_RETRIES]:
            if not _can_place(blocked, patch_out, r0, c0):
                continue
            _paint_patch(grid_in, patch_in, r0, c0)
            _paint_patch(grid_out, patch_out, r0, c0)
            _mark_blocked(blocked, patch_out, r0, c0)
            placed = True
            break
        if not placed:
            return None

    if len(_component_coords(grid_in != BLACK)) != count:
        return None
    if not np.array_equal(main(grid_in), grid_out):
        return None
    return grid_in, grid_out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each red fracture line, then mirror the larger half of each object across that hidden axis.

    Input structure:
    - The grid is a black background containing several 4-connected multicolor objects.
    - Each object uses one body color from blue/green/yellow and exactly one internal red straight crack line.
    - Every crack is a full horizontal or vertical segment that separates the object's body into two disconnected halves once the red pixels are removed.

    Rule:
    - Turn every red crack pixel into black.
    - For each object, recover the crack's row or column, count body pixels on both sides, and keep the side with more pixels as the intact half.
    - Reflect every pixel of that larger half across the crack line and paint the reflected locations with the same body color, overwriting the smaller damaged side if needed.
    - Leave the intact half unchanged, so the final object is symmetric around the now-empty crack position.

    Key insight:
    - The red crack is both removable noise and the implicit mirror axis, and the only clue for which side to trust is the area comparison between the two halves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape

    out = g.copy()
    out[out == RED] = BLACK
    objects = _component_coords(g != BLACK)
    assert objects, "expected at least one fractured object"

    for coords in objects:
        values = {int(g[r, c]) for r, c in coords}
        assert RED in values, "each object must contain a red crack line"
        body_colors = sorted(values - {BLACK, RED})
        assert len(body_colors) == 1, "each object must have exactly one non-red body color"
        body = body_colors[0]

        red_coords = np.array([(int(r), int(c)) for r, c in coords if int(g[r, c]) == RED], dtype=int)
        body_coords = np.array([(int(r), int(c)) for r, c in coords if int(g[r, c]) == body], dtype=int)
        assert len(body_coords) > 0, "each object needs body pixels"

        red_rows = sorted({int(r) for r, _c in red_coords})
        red_cols = sorted({int(c) for _r, c in red_coords})

        if len(red_cols) == 1:
            axis_c = red_cols[0]
            assert red_rows == list(range(red_rows[0], red_rows[-1] + 1)), "vertical crack must be contiguous"
            left = body_coords[body_coords[:, 1] < axis_c]
            right = body_coords[body_coords[:, 1] > axis_c]
            assert len(left) > 0 and len(right) > 0, "crack must split the object into two halves"
            assert len(left) != len(right), "the two halves must have different areas"
            major = left if len(left) > len(right) else right
            for rr, cc in major:
                mirror_c = 2 * axis_c - int(cc)
                assert 0 <= mirror_c < w, "reflected pixel must stay in bounds"
                if int(out[int(rr), mirror_c]) not in (BLACK, body):
                    raise AssertionError("reflected repair cannot overwrite another object's color")
                out[int(rr), mirror_c] = body

        elif len(red_rows) == 1:
            axis_r = red_rows[0]
            assert red_cols == list(range(red_cols[0], red_cols[-1] + 1)), "horizontal crack must be contiguous"
            top = body_coords[body_coords[:, 0] < axis_r]
            bottom = body_coords[body_coords[:, 0] > axis_r]
            assert len(top) > 0 and len(bottom) > 0, "crack must split the object into two halves"
            assert len(top) != len(bottom), "the two halves must have different areas"
            major = top if len(top) > len(bottom) else bottom
            for rr, cc in major:
                mirror_r = 2 * axis_r - int(rr)
                assert 0 <= mirror_r < h, "reflected pixel must stay in bounds"
                if int(out[mirror_r, int(cc)]) not in (BLACK, body):
                    raise AssertionError("reflected repair cannot overwrite another object's color")
                out[mirror_r, int(cc)] = body

        else:
            raise AssertionError("crack line must be strictly horizontal or vertical")

    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _attempt_grid()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid fractured-mirror ARC input")
