from __future__ import annotations

import numpy as np


BG = 0
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
FRAME_DIRECTIONS = tuple(DIRS)
CANVAS_SIZE = 30
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if int(coords.shape[0]) != expected:
        return False
    pts = {tuple(map(int, rc)) for rc in coords}
    for rr in range(r0, r1 + 1):
        if (rr, c0) not in pts or (rr, c1) not in pts:
            return False
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in pts or (r1, cc) not in pts:
            return False
    return True


def _object_direction(frame_bbox: tuple[int, int, int, int], obj_coords: np.ndarray) -> str:
    r0, c0, r1, c1 = frame_bbox
    or0, oc0, or1, oc1 = _bbox(obj_coords)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    if oc1 < center_c:
        return "left"
    if oc0 > center_c:
        return "right"
    if or1 < center_r:
        return "up"
    if or0 > center_r:
        return "down"

    mean_r = float(obj_coords[:, 0].mean())
    mean_c = float(obj_coords[:, 1].mean())
    dr = mean_r - center_r
    dc = mean_c - center_c
    if abs(dc) >= abs(dr):
        return "right" if dc >= 0 else "left"
    return "down" if dr >= 0 else "up"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each hollow frame outward on the side indicated by its interior object, using the object area as the growth distance.

    Input structure:
    - The background is black.
    - The grid contains one or more 1-pixel-thick hollow rectangular frames, possibly sharing the same border color.
    - Each frame encloses exactly one small 4-connected object of a different color, and that object lies clearly on one side of the frame's interior center.

    Rule:
    - Detect every hollow rectangular frame and the unique object strictly inside its interior.
    - Read the object's offset direction relative to the frame center (up, down, left, or right), and count its pixels N.
    - Remove the object, erase the old frame, and redraw that frame as a larger hollow rectangle whose chosen side has moved outward by N cells while the opposite side stays fixed.

    Key insight:
    - The object's position and its area encode two separate parameters: direction selects which border moves, and pixel count gives the exact expansion distance.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    h, w = g.shape

    comps: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for coords in _components(g == color):
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})

    frames: list[dict[str, object]] = []
    others: list[dict[str, object]] = []
    for comp in comps:
        if _is_rect_frame(comp["coords"]):  # type: ignore[arg-type]
            frames.append(comp)
        else:
            others.append(comp)

    frames.sort(key=lambda comp: comp["bbox"])  # type: ignore[index]
    others.sort(key=lambda comp: comp["bbox"])  # type: ignore[index]

    transforms: list[tuple[dict[str, object], dict[str, object], tuple[int, int, int, int]]] = []
    min_r = 0
    min_c = 0
    max_r = h - 1
    max_c = w - 1

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[assignment]
        inside: list[dict[str, object]] = []
        for obj in others:
            coords = obj["coords"]  # type: ignore[assignment]
            rows = coords[:, 0]
            cols = coords[:, 1]
            if (
                int(rows.min()) >= r0 + 1
                and int(rows.max()) <= r1 - 1
                and int(cols.min()) >= c0 + 1
                and int(cols.max()) <= c1 - 1
            ):
                inside.append(obj)
        if len(inside) != 1:
            continue

        obj = inside[0]
        obj_coords = obj["coords"]  # type: ignore[assignment]
        direction = _object_direction((r0, c0, r1, c1), obj_coords)
        distance = int(obj_coords.shape[0])

        if direction == "left":
            new_bbox = (r0, c0 - distance, r1, c1)
        elif direction == "right":
            new_bbox = (r0, c0, r1, c1 + distance)
        elif direction == "up":
            new_bbox = (r0 - distance, c0, r1, c1)
        else:
            new_bbox = (r0, c0, r1 + distance, c1)

        transforms.append((frame, obj, new_bbox))
        min_r = min(min_r, new_bbox[0])
        min_c = min(min_c, new_bbox[1])
        max_r = max(max_r, new_bbox[2])
        max_c = max(max_c, new_bbox[3])

    if not transforms:
        return g.copy()

    row_off = -min_r
    col_off = -min_c
    out = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    out[row_off : row_off + h, col_off : col_off + w] = g

    for frame, obj, _new_bbox in transforms:
        for rr, cc in frame["coords"]:  # type: ignore[index]
            out[int(rr) + row_off, int(cc) + col_off] = BG
        for rr, cc in obj["coords"]:  # type: ignore[index]
            out[int(rr) + row_off, int(cc) + col_off] = BG

    for frame, _obj, new_bbox in transforms:
        nr0, nc0, nr1, nc1 = new_bbox
        _draw_border(
            out,
            (nr0 + row_off, nc0 + col_off, nr1 + row_off, nc1 + col_off),
            int(frame["color"]),
        )

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _random_object_patch(rng: np.random.Generator, area: int) -> np.ndarray:
    dims = [(ph, pw) for ph in range(1, 4) for pw in range(1, 4) if ph * pw >= area]
    ph, pw = dims[int(rng.integers(0, len(dims)))]
    cells = {(int(rng.integers(0, ph)), int(rng.integers(0, pw)))}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_object_patch(rng, area)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])

    cells = _normalize(cells)
    max_r = max(r for r, _c in cells)
    max_c = max(c for _r, c in cells)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch


def _sample_inner_dims(
    rng: np.random.Generator,
    patch_shape: tuple[int, int],
    direction: str,
) -> tuple[int, int] | None:
    ph, pw = patch_shape
    inner_h_min = max(ph + 2, 4)
    inner_w_min = max(pw + 2, 4)
    if direction in ("left", "right"):
        inner_w_min = max(inner_w_min, 2 * pw + 2)
    else:
        inner_h_min = max(inner_h_min, 2 * ph + 2)
    if inner_h_min > 9 or inner_w_min > 9:
        return None

    inner_h = int(rng.integers(inner_h_min, min(9, inner_h_min + 2) + 1))
    inner_w = int(rng.integers(inner_w_min, min(9, inner_w_min + 2) + 1))
    return inner_h, inner_w


def _axis_range(axis_len: int, patch_len: int, side: str) -> tuple[int, int]:
    lo = 1
    hi = axis_len - patch_len - 1
    if side == "neg":
        hi = min(hi, axis_len // 2 - patch_len)
    elif side == "pos":
        lo = max(lo, (axis_len - 1) // 2 + 1)
    return lo, hi


def _sample_patch_origin(
    rng: np.random.Generator,
    inner_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    direction: str,
) -> tuple[int, int] | None:
    inner_h, inner_w = inner_shape
    ph, pw = patch_shape

    row_side = "any"
    col_side = "any"
    if direction == "up":
        row_side = "neg"
    elif direction == "down":
        row_side = "pos"
    elif direction == "left":
        col_side = "neg"
    else:
        col_side = "pos"

    r_lo, r_hi = _axis_range(inner_h, ph, row_side)
    c_lo, c_hi = _axis_range(inner_w, pw, col_side)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return (
        int(rng.integers(r_lo, r_hi + 1)),
        int(rng.integers(c_lo, c_hi + 1)),
    )


def _reserve_free(reserve: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserve.shape[0] - 1, r1 + 1)
    cc1 = min(reserve.shape[1] - 1, c1 + 1)
    return not bool(reserve[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_reserved(reserve: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(reserve.shape[0] - 1, r1 + 1)
    cc1 = min(reserve.shape[1] - 1, c1 + 1)
    reserve[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _crop_pair(
    rng: np.random.Generator,
    inp: np.ndarray,
    out: np.ndarray,
    specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    min_r = min(spec["expanded_bbox"][0] for spec in specs)  # type: ignore[index]
    min_c = min(spec["expanded_bbox"][1] for spec in specs)  # type: ignore[index]
    max_r = max(spec["expanded_bbox"][2] for spec in specs)  # type: ignore[index]
    max_c = max(spec["expanded_bbox"][3] for spec in specs)  # type: ignore[index]

    top_pad = int(rng.integers(0, min(4, min_r) + 1))
    left_pad = int(rng.integers(0, min(4, min_c) + 1))
    bottom_pad = int(rng.integers(0, min(4, CANVAS_SIZE - 1 - max_r) + 1))
    right_pad = int(rng.integers(0, min(4, CANVAS_SIZE - 1 - max_c) + 1))

    r0 = min_r - top_pad
    c0 = min_c - left_pad
    r1 = max_r + bottom_pad
    c1 = max_c + right_pad
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    reserve = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)
    palette = [color for color in range(1, 10)]
    frame_color = int(rng.choice(palette))
    object_palette = [color for color in palette if color != frame_color]
    frame_count = int(rng.integers(1, 4))

    specs: list[dict[str, object]] = []
    for _ in range(frame_count):
        placed = False
        for _attempt in range(200):
            direction = FRAME_DIRECTIONS[int(rng.integers(0, len(FRAME_DIRECTIONS)))]
            area = int(rng.integers(1, 6))
            patch = _random_object_patch(rng, area)
            inner_dims = _sample_inner_dims(rng, patch.shape, direction)
            if inner_dims is None:
                continue
            inner_h, inner_w = inner_dims
            patch_origin = _sample_patch_origin(rng, inner_dims, patch.shape, direction)
            if patch_origin is None:
                continue

            frame_h = inner_h + 2
            frame_w = inner_w + 2
            grow_h = frame_h + (area if direction in ("up", "down") else 0)
            grow_w = frame_w + (area if direction in ("left", "right") else 0)
            if grow_h > CANVAS_SIZE or grow_w > CANVAS_SIZE:
                continue

            grow_r0 = int(rng.integers(0, CANVAS_SIZE - grow_h + 1))
            grow_c0 = int(rng.integers(0, CANVAS_SIZE - grow_w + 1))
            grow_bbox = (grow_r0, grow_c0, grow_r0 + grow_h - 1, grow_c0 + grow_w - 1)
            if not _reserve_free(reserve, grow_bbox):
                continue

            if direction == "right":
                frame_r0 = grow_r0
                frame_c0 = grow_c0
            elif direction == "left":
                frame_r0 = grow_r0
                frame_c0 = grow_c0 + area
            elif direction == "down":
                frame_r0 = grow_r0
                frame_c0 = grow_c0
            else:
                frame_r0 = grow_r0 + area
                frame_c0 = grow_c0
            frame_bbox = (frame_r0, frame_c0, frame_r0 + frame_h - 1, frame_c0 + frame_w - 1)

            obj_r0 = frame_r0 + 1 + patch_origin[0]
            obj_c0 = frame_c0 + 1 + patch_origin[1]
            specs.append(
                {
                    "frame_bbox": frame_bbox,
                    "expanded_bbox": grow_bbox,
                    "patch": patch,
                    "object_origin": (obj_r0, obj_c0),
                    "object_color": int(rng.choice(object_palette)),
                }
            )
            _mark_reserved(reserve, grow_bbox)
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    out = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=int)
    for spec in specs:
        _draw_border(inp, spec["frame_bbox"], frame_color)  # type: ignore[arg-type]
        _draw_border(out, spec["expanded_bbox"], frame_color)  # type: ignore[arg-type]
        patch = spec["patch"]  # type: ignore[assignment]
        obj_r0, obj_c0 = spec["object_origin"]  # type: ignore[assignment]
        patch_view = inp[obj_r0 : obj_r0 + patch.shape[0], obj_c0 : obj_c0 + patch.shape[1]]
        patch_view[patch] = int(spec["object_color"])

    cropped_inp, cropped_out = _crop_pair(rng, inp, out, specs)
    if max(cropped_inp.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if not np.array_equal(main(cropped_inp), cropped_out):
        return None
    return cropped_inp, cropped_out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        case = _build_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
