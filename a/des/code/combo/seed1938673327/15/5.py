from __future__ import annotations

import random

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = ("up", "down", "left", "right")

MAX_GENERATION_ATTEMPTS = 300
FRAME_COUNT_CHOICES = (4, 5)
FRAME_SIDE_RANGE = (7, 11)
GRID_SIDE_RANGE = {
    4: ((26, 33), (26, 33)),
    5: ((30, 37), (30, 37)),
}
FRAME_PADDING = 2
INNER_MARGIN = 1
SHAPE_ATTEMPTS = 120
FRAME_PLACEMENT_ATTEMPTS = 120


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_from_mask(g == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            if int(coords.shape[0]) != 2 * h + 2 * w - 4:
                continue
            local = np.zeros((h, w), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(local, border):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _direction_from_coords(coords: np.ndarray) -> str:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dr) > abs(dc):
        assert abs(dr) > 1e-9, "expected a non-zero vertical centroid bias"
        return "down" if dr > 0 else "up"
    assert abs(dc) > abs(dr), "expected a unique centroid-bias direction"
    return "right" if dc > 0 else "left"


def _direction_or_none(coords: np.ndarray) -> str | None:
    try:
        return _direction_from_coords(coords)
    except AssertionError:
        return None


def _carve_opening(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    obj_bbox: tuple[int, int, int, int],
    direction: str,
    bg: int,
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    or0, oc0, or1, oc1 = obj_bbox
    if direction == "up":
        out[fr0, oc0 : oc1 + 1] = bg
    elif direction == "down":
        out[fr1, oc0 : oc1 + 1] = bg
    elif direction == "left":
        out[or0 : or1 + 1, fc0] = bg
    else:
        out[or0 : or1 + 1, fc1] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each rectangular frame on the side indicated by its interior object's centroid bias.

    Input structure
    - Black background with several one-pixel-thick rectangular frames.
    - Each frame contains one 4-connected foreground object.
    - The object's shape is asymmetric: its centroid is shifted toward one side of its bbox.

    Rule
    - Detect each complete hollow rectangle.
    - Inside each frame, find the unique foreground object and compute the centroid offset
      relative to the center of the object's bbox.
    - The dominant offset direction determines which frame side to open.
    - Remove the border segment aligned with the object's projection onto that side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    frames = _frame_candidates(g, bg)
    assert frames, "expected at least one complete rectangular frame"

    out = g.copy()
    for frame in frames:
        fr0, fc0, fr1, fc1 = frame["bbox"]
        frame_color = int(frame["color"])
        assert fr1 - fr0 >= 2 and fc1 - fc0 >= 2
        interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        obj_mask = (interior != bg) & (interior != frame_color)
        comps = _components_from_mask(obj_mask)
        assert len(comps) == 1, "expected one interior foreground object per frame"
        obj_coords = comps[0] + np.array([fr0 + 1, fc0 + 1])
        obj_bbox = _bbox_from_coords(obj_coords)
        direction = _direction_from_coords(obj_coords)
        _carve_opening(out, frame["bbox"], obj_bbox, direction, bg)

    return out


def _axis_bias_margin(mask: np.ndarray, direction: str) -> float:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    dr = mean_r - center_r
    dc = mean_c - center_c
    if direction == "up":
        return -dr - abs(dc)
    if direction == "down":
        return dr - abs(dc)
    if direction == "left":
        return -dc - abs(dr)
    return dc - abs(dr)


def _make_biased_shape(direction: str, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        h = random.randint(3, max_h)
        w = random.randint(3, max_w)
        mask = np.ones((h, w), dtype=bool)
        bite_count = random.randint(1, 3)

        for _ in range(bite_count):
            if direction in ("left", "right"):
                bite_w = random.randint(1, max(1, w - 1))
                bite_h = random.randint(1, h)
                r0 = random.randint(0, h - bite_h)
                if direction == "right":
                    c0 = 0
                else:
                    c0 = w - bite_w
                mask[r0 : r0 + bite_h, c0 : c0 + bite_w] = False
            else:
                bite_h = random.randint(1, max(1, h - 1))
                bite_w = random.randint(1, w)
                c0 = random.randint(0, w - bite_w)
                if direction == "down":
                    r0 = 0
                else:
                    r0 = h - bite_h
                mask[r0 : r0 + bite_h, c0 : c0 + bite_w] = False

        if direction == "right":
            mask[:, -1] = True
        elif direction == "left":
            mask[:, 0] = True
        elif direction == "down":
            mask[-1, :] = True
        else:
            mask[0, :] = True

        if int(mask.sum()) < 4 or int(mask.sum()) >= h * w:
            continue
        if len(_components_from_mask(mask)) != 1:
            continue

        cropped = _crop_mask(mask)
        if min(cropped.shape) < 2:
            continue
        if direction in ("up", "down") and cropped.shape[1] < 2:
            continue
        if direction in ("left", "right") and cropped.shape[0] < 2:
            continue
        if _direction_or_none(np.argwhere(cropped)) != direction:
            continue
        if _axis_bias_margin(cropped, direction) <= 0.15:
            continue
        return cropped
    return None


def _place_frames(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    frames: list[tuple[int, int, int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(FRAME_PLACEMENT_ATTEMPTS):
            fh = random.randint(*FRAME_SIDE_RANGE)
            fw = random.randint(*FRAME_SIDE_RANGE)
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            ok = True
            for er0, ec0, er1, ec1 in frames:
                if not (
                    bbox[2] + FRAME_PADDING < er0
                    or er1 + FRAME_PADDING < bbox[0]
                    or bbox[3] + FRAME_PADDING < ec0
                    or ec1 + FRAME_PADDING < bbox[1]
                ):
                    ok = False
                    break
            if ok:
                frames.append(bbox)
                placed = True
                break
        if not placed:
            return None
    return frames


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _draw_object(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    direction: str,
    color: int,
) -> tuple[int, int, int, int] | None:
    fr0, fc0, fr1, fc1 = frame_bbox
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    max_h = inner_h - 2 * INNER_MARGIN
    max_w = inner_w - 2 * INNER_MARGIN
    if max_h < 3 or max_w < 3:
        return None

    shape = _make_biased_shape(direction, max_h, max_w)
    if shape is None:
        return None
    sh, sw = shape.shape
    row_lo = fr0 + 1 + INNER_MARGIN
    row_hi = fr1 - INNER_MARGIN - sh
    col_lo = fc0 + 1 + INNER_MARGIN
    col_hi = fc1 - INNER_MARGIN - sw
    if row_lo > row_hi or col_lo > col_hi:
        return None
    top = random.randint(row_lo, row_hi)
    left = random.randint(col_lo, col_hi)
    grid[top : top + sh, left : left + sw][shape] = color
    return top, left, top + sh - 1, left + sw - 1


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    (h_lo, h_hi), (w_lo, w_hi) = GRID_SIDE_RANGE[frame_count]
    h = random.randint(h_lo, h_hi - 1)
    w = random.randint(w_lo, w_hi - 1)

    frames = _place_frames(h, w, frame_count)
    if frames is None:
        return None

    directions = list(DIRS)
    if frame_count > 4:
        directions.extend(random.choices(DIRS, k=frame_count - 4))
    random.shuffle(directions)
    colors = random.sample(OBJECT_COLORS, frame_count)

    inp = np.full((h, w), BG_COLOR, dtype=int)
    expected = inp.copy()
    infos: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int], str]] = []
    for bbox, direction, color in zip(frames, directions, colors):
        _paint_frame(inp, bbox, FRAME_COLOR)
        _paint_frame(expected, bbox, FRAME_COLOR)
        obj_bbox = _draw_object(inp, bbox, direction, color)
        if obj_bbox is None:
            return None
        _drawn = inp[obj_bbox[0] : obj_bbox[2] + 1, obj_bbox[1] : obj_bbox[3] + 1] == color
        if _direction_or_none(np.argwhere(_drawn)) != direction:
            return None
        infos.append((bbox, obj_bbox, direction))

    expected[:] = inp
    for bbox, obj_bbox, direction in infos:
        _carve_opening(expected, bbox, obj_bbox, direction, BG_COLOR)

    if _dominant_color(inp) != BG_COLOR:
        return None
    frames_found = _frame_candidates(inp, BG_COLOR)
    if len(frames_found) != frame_count:
        return None
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
