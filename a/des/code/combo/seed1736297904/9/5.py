from __future__ import annotations

import numpy as np


SIDES = ("top", "bottom", "left", "right")
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# --------------- solver ---------------


def _most_common_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[cnts.argmax()])


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _local_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    return local, (r0, c0, r1, c1)


def _frame_info(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    local, bbox = _local_mask(coords)
    h, w = local.shape
    if h < 3 or w < 3:
        return None
    if bool(local[1:-1, 1:-1].any()):
        return None

    matches: list[str] = []
    for side in SIDES:
        expected = np.zeros((h, w), dtype=bool)
        expected[0, :] = True
        expected[-1, :] = True
        expected[:, 0] = True
        expected[:, -1] = True
        if side == "top":
            expected[0, 1:-1] = False
        elif side == "bottom":
            expected[-1, 1:-1] = False
        elif side == "left":
            expected[1:-1, 0] = False
        else:
            expected[1:-1, -1] = False
        if np.array_equal(local, expected):
            matches.append(side)

    if len(matches) != 1:
        return None

    return {
        "bbox": bbox,
        "coords": coords,
        "missing_side": matches[0],
    }


def _line_info(coords: list[tuple[int, int]]) -> dict[str, object] | None:
    local, bbox = _local_mask(coords)
    h, w = local.shape
    if h != 1 and w != 1:
        return None
    if not bool(local.all()):
        return None
    length = int(max(h, w))
    if length == 1:
        orientation = "point"
    else:
        orientation = "horizontal" if h == 1 else "vertical"
    return {
        "bbox": bbox,
        "coords": coords,
        "length": length,
        "orientation": orientation,
    }


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = _most_common_color(grid)
    objects: list[dict[str, object]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        comps = _components(grid == color)
        frame = None
        line = None
        for comp in comps:
            frame_candidate = _frame_info(comp)
            if frame_candidate is not None:
                assert frame is None
                frame = frame_candidate
                continue
            line_candidate = _line_info(comp)
            if line_candidate is not None:
                assert line is None
                line = line_candidate
                continue
            raise AssertionError(f"unexpected component for color {color}")

        assert len(comps) == 2
        assert frame is not None and line is not None

        side = str(frame["missing_side"])
        required = "horizontal" if side in ("top", "bottom") else "vertical"
        assert str(line["orientation"]) in (required, "point")
        objects.append({"color": color, "frame": frame, "line": line})

    return bg, objects


def _fill_from_missing_side(
    out: np.ndarray,
    color: int,
    bbox: tuple[int, int, int, int],
    side: str,
    depth: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        out[r0, c0 : c1 + 1] = color
        fill = min(depth, max(0, r1 - r0 - 1))
        if fill > 0:
            out[r0 + 1 : r0 + 1 + fill, c0 + 1 : c1] = color
    elif side == "bottom":
        out[r1, c0 : c1 + 1] = color
        fill = min(depth, max(0, r1 - r0 - 1))
        if fill > 0:
            out[r1 - fill : r1, c0 + 1 : c1] = color
    elif side == "left":
        out[r0 : r1 + 1, c0] = color
        fill = min(depth, max(0, c1 - c0 - 1))
        if fill > 0:
            out[r0 + 1 : r1, c0 + 1 : c0 + 1 + fill] = color
    else:
        out[r0 : r1 + 1, c1] = color
        fill = min(depth, max(0, c1 - c0 - 1))
        if fill > 0:
            out[r0 + 1 : r1, c1 - fill : c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each three-sided rectangular frame, then fill inward from the
    missing side by the matching line segment's length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, objects = _parse_objects(g)

    for obj in objects:
        color = int(obj["color"])
        frame = obj["frame"]
        line = obj["line"]
        for r, c in line["coords"]:
            out[r, c] = bg
        _fill_from_missing_side(
            out,
            color,
            tuple(int(v) for v in frame["bbox"]),
            str(frame["missing_side"]),
            int(line["length"]),
        )

    return out


# --------------- generator ---------------

BG = 0
COLOR_POOL = np.arange(1, 10, dtype=int)
NUM_OBJECTS_RANGE = (2, 6)
LINE_LENGTH_RANGE = (1, 5)
GRID_BASE_RANGE = (18, 30)
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 11
PLACEMENT_MARGIN = 1
PLACEMENT_PAD = 1
FRAME_TRIES = 150
LINE_TRIES = 200
MAX_GENERATION_ATTEMPTS = 400


def _box_clear(
    reserved: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int,
) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _reserve_box(
    reserved: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int,
) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(reserved.shape[0], r0 + h + pad)
    cc1 = min(reserved.shape[1], c0 + w + pad)
    reserved[rr0:rr1, cc0:cc1] = True


def _draw_incomplete_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    side: str,
    color: int,
) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if side == "top":
        grid[r0, c0 + 1 : c1] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 : c1] = BG
    elif side == "left":
        grid[r0 + 1 : r1, c0] = BG
    else:
        grid[r0 + 1 : r1, c1] = BG


def _draw_line(
    grid: np.ndarray,
    r0: int,
    c0: int,
    length: int,
    orientation: str,
    color: int,
) -> None:
    if orientation == "horizontal":
        grid[r0, c0 : c0 + length] = color
    else:
        grid[r0 : r0 + length, c0] = color


def _sample_sides(rng: np.random.Generator, n_objects: int) -> list[str]:
    perm = list(rng.permutation(np.array(SIDES, dtype=object)))
    sides = [str(side) for side in perm[: min(n_objects, len(SIDES))]]
    while len(sides) < n_objects:
        sides.append(str(SIDES[int(rng.integers(0, len(SIDES)))]))
    rng.shuffle(sides)
    return sides


def _sample_frame_shape(
    rng: np.random.Generator,
    side: str,
    depth: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    min_h = max(FRAME_SIZE_MIN, depth + 2 if side in ("top", "bottom") else FRAME_SIZE_MIN)
    min_w = max(FRAME_SIZE_MIN, depth + 2 if side in ("left", "right") else FRAME_SIZE_MIN)
    max_h = min(FRAME_SIZE_MAX, grid_h - 2 * PLACEMENT_MARGIN)
    max_w = min(FRAME_SIZE_MAX, grid_w - 2 * PLACEMENT_MARGIN)
    if min_h > max_h or min_w > max_w:
        return None
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    return h, w


def _place_box(
    rng: np.random.Generator,
    reserved: np.ndarray,
    h: int,
    w: int,
    tries: int,
) -> tuple[int, int] | None:
    grid_h, grid_w = reserved.shape
    if grid_h - h - PLACEMENT_MARGIN < PLACEMENT_MARGIN or grid_w - w - PLACEMENT_MARGIN < PLACEMENT_MARGIN:
        return None
    for _ in range(tries):
        r0 = int(rng.integers(PLACEMENT_MARGIN, grid_h - h - PLACEMENT_MARGIN + 1))
        c0 = int(rng.integers(PLACEMENT_MARGIN, grid_w - w - PLACEMENT_MARGIN + 1))
        if _box_clear(reserved, r0, c0, h, w, PLACEMENT_PAD):
            return r0, c0
    return None


def _render_expected(shape: tuple[int, int], specs: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        color = int(spec["color"])
        side = str(spec["side"])
        depth = int(spec["depth"])

        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        _fill_from_missing_side(out, color, (r0, c0, r1, c1), side, depth)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(*NUM_OBJECTS_RANGE))
    base = int(rng.integers(*GRID_BASE_RANGE))
    grid_h = int(base + rng.integers(0, 4 + n_objects))
    grid_w = int(base + rng.integers(0, 4 + n_objects))

    colors = [int(v) for v in rng.choice(COLOR_POOL, size=n_objects, replace=False)]
    sides = _sample_sides(rng, n_objects)
    depths = [int(v) for v in rng.integers(*LINE_LENGTH_RANGE, size=n_objects)]

    specs: list[dict[str, object]] = []
    for color, side, depth in zip(colors, sides, depths, strict=False):
        dims = _sample_frame_shape(rng, side, depth, grid_h, grid_w)
        if dims is None:
            return None
        frame_h, frame_w = dims
        orientation = "horizontal" if side in ("top", "bottom") else "vertical"
        specs.append(
            {
                "color": color,
                "side": side,
                "depth": depth,
                "orientation": orientation,
                "frame_h": frame_h,
                "frame_w": frame_w,
            }
        )

    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    frame_order = sorted(
        range(n_objects),
        key=lambda idx: int(specs[idx]["frame_h"]) * int(specs[idx]["frame_w"]),
        reverse=True,
    )
    for idx in frame_order:
        spec = specs[idx]
        pos = _place_box(
            rng,
            reserved,
            int(spec["frame_h"]),
            int(spec["frame_w"]),
            FRAME_TRIES,
        )
        if pos is None:
            return None
        r0, c0 = pos
        spec["frame_bbox"] = (
            r0,
            c0,
            r0 + int(spec["frame_h"]) - 1,
            c0 + int(spec["frame_w"]) - 1,
        )
        _reserve_box(reserved, r0, c0, int(spec["frame_h"]), int(spec["frame_w"]), PLACEMENT_PAD)

    line_order = sorted(range(n_objects), key=lambda idx: int(specs[idx]["depth"]), reverse=True)
    for idx in line_order:
        spec = specs[idx]
        length = int(spec["depth"])
        line_h = 1 if spec["orientation"] == "horizontal" else length
        line_w = length if spec["orientation"] == "horizontal" else 1
        pos = _place_box(rng, reserved, line_h, line_w, LINE_TRIES)
        if pos is None:
            return None
        r0, c0 = pos
        spec["line_bbox"] = (r0, c0, r0 + line_h - 1, c0 + line_w - 1)
        _reserve_box(reserved, r0, c0, line_h, line_w, PLACEMENT_PAD)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["frame_bbox"]
        _draw_incomplete_frame(
            inp,
            int(r0),
            int(c0),
            int(r1 - r0 + 1),
            int(c1 - c0 + 1),
            str(spec["side"]),
            int(spec["color"]),
        )
        lr0, lc0, _lr1, _lc1 = spec["line_bbox"]
        _draw_line(
            inp,
            int(lr0),
            int(lc0),
            int(spec["depth"]),
            str(spec["orientation"]),
            int(spec["color"]),
        )

    if int(np.count_nonzero(inp == BG)) <= inp.size // 2:
        return None

    expected = _render_expected(inp.shape, specs)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
