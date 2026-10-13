from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _edge_contacts(coords: np.ndarray, h: int, w: int) -> list[str]:
    contacts: list[str] = []
    if np.any(coords[:, 0] == 0):
        contacts.append("top")
    if np.any(coords[:, 0] == h - 1):
        contacts.append("bottom")
    if np.any(coords[:, 1] == 0):
        contacts.append("left")
    if np.any(coords[:, 1] == w - 1):
        contacts.append("right")
    return contacts


def _find_blob(window: np.ndarray) -> tuple[int, str, int]:
    h, w = window.shape
    colors = [int(color) for color in np.unique(window)]
    assert BG not in colors, "object rectangle must not contain background holes"
    assert len(colors) == 2, "each rectangle must contain exactly two colors"

    candidates: list[tuple[int, str, int]] = []
    for color in colors:
        coords = np.argwhere(window == color)
        area = int(len(coords))
        if not (1 <= area <= 5):
            continue
        contacts = _edge_contacts(coords, h, w)
        if len(contacts) != 1:
            continue
        if len(_components(window == color)) != 1:
            continue
        candidates.append((color, contacts[0], area))

    assert len(candidates) == 1, "expected a unique small edge-attached blob"
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    objects: list[dict[str, object]] = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        window = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(window != BG), "each object component must fill its bounding rectangle"
        blob_color, side, thickness = _find_blob(window)
        objects.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "blob_color": blob_color,
                "side": side,
                "thickness": thickness,
            }
        )
    return objects


def _paint_growth(out: np.ndarray, obj: dict[str, object]) -> None:
    r0 = int(obj["r0"])
    c0 = int(obj["c0"])
    r1 = int(obj["r1"])
    c1 = int(obj["c1"])
    color = int(obj["blob_color"])
    thickness = int(obj["thickness"])
    side = str(obj["side"])

    if side == "top":
        assert r0 - thickness >= 0
        out[r0 - thickness : r0, c0 : c1 + 1] = color
    elif side == "bottom":
        assert r1 + thickness < out.shape[0]
        out[r1 + 1 : r1 + thickness + 1, c0 : c1 + 1] = color
    elif side == "left":
        assert c0 - thickness >= 0
        out[r0 : r1 + 1, c0 - thickness : c0] = color
    elif side == "right":
        assert c1 + thickness < out.shape[1]
        out[r0 : r1 + 1, c1 + 1 : c1 + thickness + 1] = color
    else:
        raise AssertionError("unknown growth side")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a colored strip outward from each rectangle based on its embedded blob.

    The blob's contacted edge gives the outward direction, its area gives the
    strip thickness, and its color gives the strip color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for obj in _extract_objects(g):
        _paint_growth(out, obj)
    return out


# --------------- generator ---------------

GRID_MIN = 18
GRID_MAX = 30
OBJECT_COUNT_MIN = 2
OBJECT_COUNT_MAX = 4
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 9
BLOB_SIZE_MIN = 1
BLOB_SIZE_MAX = 5
SPEC_ATTEMPTS = 200
SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 400


def _allowed_blob_cells(h: int, w: int, side: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if side == "top":
                ok = r <= h - 2 and 1 <= c <= w - 2
            elif side == "bottom":
                ok = 1 <= r <= h - 1 and 1 <= c <= w - 2
            elif side == "left":
                ok = 1 <= r <= h - 2 and c <= w - 2
            elif side == "right":
                ok = 1 <= r <= h - 2 and 1 <= c <= w - 1
            else:
                raise AssertionError("unknown side")
            if ok:
                cells.add((r, c))
    return cells


def _seed_cells(h: int, w: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, w - 1)]
    if side == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if side == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    raise AssertionError("unknown side")


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _generate_blob_shape(h: int, w: int, side: str, size: int) -> np.ndarray | None:
    allowed = _allowed_blob_cells(h, w, side)
    if len(allowed) < size:
        return None
    seeds = _seed_cells(h, w, side)
    if not seeds:
        return None

    for _ in range(SHAPE_ATTEMPTS):
        shape = {random.choice(seeds)}
        while len(shape) < size:
            frontier = {
                nbr
                for cell in shape
                for nbr in _neighbors(cell)
                if nbr in allowed and nbr not in shape
            }
            if not frontier:
                break
            shape.add(random.choice(tuple(frontier)))
        if len(shape) != size:
            continue
        coords = np.asarray(sorted(shape), dtype=int)
        if len(_edge_contacts(coords, h, w)) != 1:
            continue
        return coords
    return None


def _make_spec(rect_color: int, blob_color: int) -> dict[str, object] | None:
    for _ in range(SPEC_ATTEMPTS):
        side = random.choice(SIDES)
        thickness = random.randint(BLOB_SIZE_MIN, BLOB_SIZE_MAX)
        h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
        if h * w < 2 * thickness + 4:
            continue
        blob = _generate_blob_shape(h, w, side, thickness)
        if blob is None:
            continue
        return {
            "height": h,
            "width": w,
            "side": side,
            "thickness": thickness,
            "rect_color": rect_color,
            "blob_color": blob_color,
            "blob": blob,
        }
    return None


def _footprint_bounds(top: int, left: int, spec: dict[str, object]) -> tuple[int, int, int, int]:
    h = int(spec["height"])
    w = int(spec["width"])
    side = str(spec["side"])
    thickness = int(spec["thickness"])

    if side == "top":
        return top - thickness, left, top + h - 1, left + w - 1
    if side == "bottom":
        return top, left, top + h + thickness - 1, left + w - 1
    if side == "left":
        return top, left - thickness, top + h - 1, left + w - 1
    if side == "right":
        return top, left, top + h - 1, left + w + thickness - 1
    raise AssertionError("unknown side")


def _inflate(bounds: tuple[int, int, int, int], margin: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bounds
    return r0 - margin, c0 - margin, r1 + margin, c1 + margin


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _position_ranges(grid_h: int, grid_w: int, spec: dict[str, object]) -> tuple[int, int, int, int] | None:
    h = int(spec["height"])
    w = int(spec["width"])
    side = str(spec["side"])
    thickness = int(spec["thickness"])

    if side == "top":
        row_lo, row_hi = thickness + 1, grid_h - h - 1
        col_lo, col_hi = 1, grid_w - w - 1
    elif side == "bottom":
        row_lo, row_hi = 1, grid_h - h - thickness - 1
        col_lo, col_hi = 1, grid_w - w - 1
    elif side == "left":
        row_lo, row_hi = 1, grid_h - h - 1
        col_lo, col_hi = thickness + 1, grid_w - w - 1
    elif side == "right":
        row_lo, row_hi = 1, grid_h - h - 1
        col_lo, col_hi = 1, grid_w - w - thickness - 1
    else:
        raise AssertionError("unknown side")

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return row_lo, row_hi, col_lo, col_hi


def _place_specs(
    grid_h: int,
    grid_w: int,
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    placed: list[dict[str, object]] = []
    blocked: list[tuple[int, int, int, int]] = []

    specs = sorted(
        specs,
        key=lambda spec: (
            (int(spec["height"]) + (int(spec["thickness"]) if str(spec["side"]) in ("top", "bottom") else 0))
            * (int(spec["width"]) + (int(spec["thickness"]) if str(spec["side"]) in ("left", "right") else 0))
        ),
        reverse=True,
    )

    for spec in specs:
        ranges = _position_ranges(grid_h, grid_w, spec)
        if ranges is None:
            return None
        row_lo, row_hi, col_lo, col_hi = ranges

        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            box = _inflate(_footprint_bounds(top, left, spec))
            if any(_overlap(box, other) for other in blocked):
                continue
            spec_with_pos = dict(spec)
            spec_with_pos["top"] = top
            spec_with_pos["left"] = left
            placed.append(spec_with_pos)
            blocked.append(box)
            success = True
            break
        if not success:
            return None
    return placed


def _render_input(grid_h: int, grid_w: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in placed:
        top = int(spec["top"])
        left = int(spec["left"])
        h = int(spec["height"])
        w = int(spec["width"])
        rect_color = int(spec["rect_color"])
        blob_color = int(spec["blob_color"])
        blob = np.asarray(spec["blob"], dtype=int)

        grid[top : top + h, left : left + w] = rect_color
        grid[top + blob[:, 0], left + blob[:, 1]] = blob_color
    return grid


def _render_expected(grid: np.ndarray, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in placed:
        r0 = int(spec["top"])
        c0 = int(spec["left"])
        obj = {
            "r0": r0,
            "c0": c0,
            "r1": r0 + int(spec["height"]) - 1,
            "c1": c0 + int(spec["width"]) - 1,
            "blob_color": int(spec["blob_color"]),
            "side": str(spec["side"]),
            "thickness": int(spec["thickness"]),
        }
        _paint_growth(out, obj)
    return out


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    object_count = random.randint(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX)
    colors = random.sample(range(1, 10), 2 * object_count)

    specs: list[dict[str, object]] = []
    for i in range(object_count):
        spec = _make_spec(colors[i], colors[i + object_count])
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_specs(grid_h, grid_w, specs)
    if placed is None:
        return None

    grid = _render_input(grid_h, grid_w, placed)
    expected = _render_expected(grid, placed)

    try:
        parsed = _extract_objects(grid)
    except AssertionError:
        return None
    if len(parsed) != object_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
