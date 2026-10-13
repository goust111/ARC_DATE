from __future__ import annotations

import numpy as np


BG = 0
FRAME_COLOR = 5
DIRECTIONS = ("top", "bottom", "left", "right")
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
TEMPLATE_H_RANGE = (3, 5)
TEMPLATE_W_RANGE = (3, 5)
INNER_PAD_RANGE = (1, 2)
LANE_GAP_RANGE = (1, 2)
OUTER_MARGIN_EXTRA = (0, 2)


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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(arr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask_1d.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask_1d)))
    return runs


def _frame_info(g: np.ndarray) -> tuple[tuple[int, int, int, int], str]:
    frame_mask = g == FRAME_COLOR
    coords = np.argwhere(frame_mask)
    assert coords.size > 0, "missing grey frame"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must be large enough for a gap"

    crop = frame_mask[r0 : r1 + 1, c0 : c1 + 1]
    assert not crop[1:-1, 1:-1].any(), "grey pixels must only form the frame border"

    edges = {
        "top": crop[0, :],
        "bottom": crop[-1, :],
        "left": crop[:, 0],
        "right": crop[:, -1],
    }
    gap_side: str | None = None
    for side, line in edges.items():
        missing = np.flatnonzero(~line)
        if len(missing) == 0:
            continue
        contiguous = missing[-1] - missing[0] + 1 == len(missing)
        valid_gap = contiguous and len(missing) >= 2 and missing[0] > 0 and missing[-1] < len(line) - 1
        assert valid_gap, "frame side gaps must be one interior contiguous run of width >= 2"
        assert gap_side is None, "expected exactly one broken frame side"
        gap_side = side

    assert gap_side is not None, "missing frame gap"
    for side, line in edges.items():
        if side == gap_side:
            continue
        assert bool(np.all(line)), "all non-gap frame sides must be complete"
    return (r0, c0, r1, c1), gap_side


def _templates_in_frame(
    g: np.ndarray, frame_box: tuple[int, int, int, int]
) -> tuple[list[dict[str, object]], tuple[float, float]]:
    r0, c0, r1, c1 = frame_box
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    colors = [int(color) for color in np.unique(interior) if int(color) not in (BG, FRAME_COLOR)]
    records: list[dict[str, object]] = []
    for color in colors:
        comps = _components(interior == color)
        assert len(comps) == 1, "each template color must appear in exactly one component"
        box = _bbox(comps[0])
        mask = _crop(interior == color, box)
        center = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        records.append({"color": color, "box": box, "mask": mask, "center": center})

    assert 2 <= len(records) <= 4, "expected 2-4 interior templates"
    shape = records[0]["mask"].shape  # type: ignore[index]
    assert all(record["mask"].shape == shape for record in records), "template bboxes must match"

    center = ((interior.shape[0] - 1) / 2.0, (interior.shape[1] - 1) / 2.0)
    return records, center


def _choose_template(
    records: list[dict[str, object]], gap_side: str, interior_center: tuple[float, float]
) -> np.ndarray:
    if gap_side == "top":
        ordered = sorted(
            records,
            key=lambda record: (
                record["center"][0],  # type: ignore[index]
                abs(record["center"][1] - interior_center[1]),  # type: ignore[index]
            ),
        )
    elif gap_side == "bottom":
        ordered = sorted(
            records,
            key=lambda record: (
                -record["center"][0],  # type: ignore[index]
                abs(record["center"][1] - interior_center[1]),  # type: ignore[index]
            ),
        )
    elif gap_side == "left":
        ordered = sorted(
            records,
            key=lambda record: (
                record["center"][1],  # type: ignore[index]
                abs(record["center"][0] - interior_center[0]),  # type: ignore[index]
            ),
        )
    else:
        ordered = sorted(
            records,
            key=lambda record: (
                -record["center"][1],  # type: ignore[index]
                abs(record["center"][0] - interior_center[0]),  # type: ignore[index]
            ),
        )
    return np.array(ordered[0]["mask"], dtype=bool)


def _palette_strip(g: np.ndarray) -> tuple[str, np.ndarray]:
    candidates: list[tuple[str, np.ndarray]] = []
    edge_lines = {
        "top": g[0, :],
        "bottom": g[-1, :],
        "left": g[:, 0],
        "right": g[:, -1],
    }
    for side, line in edge_lines.items():
        for start, end in _runs(line != BG):
            if end - start >= 2:
                candidates.append((side, np.array(line[start:end], dtype=int)))

    assert len(candidates) == 1, "expected exactly one palette strip on the grid edge"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the template indexed by the frame-gap direction, then repaint its mask using the edge palette strip.

    Input structure:
    - The background is black.
    - One grey rectangular frame contains a 1-cell-thick border with exactly one contiguous side gap of width at least 2.
    - Inside the frame there are 2-4 same-size monochrome templates placed in the top, bottom, left, and/or right slots.
    - A separate 1-cell-thick palette strip lies on one grid edge, horizontally or vertically, with length matching the template width or height.

    Rule:
    - Detect which side of the grey frame has the missing border segment.
    - Choose the interior template located in that same direction and crop its bounding box to obtain a binary mask.
    - If the palette strip is horizontal, repeat it across the template height so each output column uses the corresponding palette color.
    - If the palette strip is vertical, repeat it across the template width so each output row uses the corresponding palette color.
    - Keep only the repeated colors where the chosen template mask is filled; set all other cells in the crop to black.

    Key insight:
    The frame gap contributes only a directional index, while the selected template contributes the shape mask and the edge strip contributes the final colors.
    """
    g = np.asarray(grid, dtype=int)

    frame_box, gap_side = _frame_info(g)
    records, interior_center = _templates_in_frame(g, frame_box)
    active_mask = _choose_template(records, gap_side, interior_center)

    palette_side, strip = _palette_strip(g)
    h, w = active_mask.shape
    if palette_side in ("top", "bottom"):
        assert len(strip) == w, "horizontal strip length must equal template width"
        out = np.tile(strip, (h, 1))
    else:
        assert len(strip) == h, "vertical strip length must equal template height"
        out = np.tile(strip[:, None], (1, w))
    out[~active_mask] = BG
    return out


def _path_to(
    start: tuple[int, int], end: tuple[int, int], rng: np.random.Generator
) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        steps: list[tuple[int, int]] = []
        if r < er:
            steps.append((r + 1, c))
        elif r > er:
            steps.append((r - 1, c))
        if c < ec:
            steps.append((r, c + 1))
        elif c > ec:
            steps.append((r, c - 1))
        nr, nc = steps[int(rng.integers(0, len(steps)))]
        r, c = nr, nc
        path.append((r, c))
    return path


def _random_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray | None:
    area_cap = int(np.floor(0.78 * h * w))
    for _ in range(120):
        center = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        anchors = [
            (0, int(rng.integers(0, w))),
            (h - 1, int(rng.integers(0, w))),
            (int(rng.integers(0, h)), 0),
            (int(rng.integers(0, h)), w - 1),
        ]
        mask = np.zeros((h, w), dtype=bool)
        for anchor in anchors:
            for rr, cc in _path_to(anchor, center, rng):
                mask[rr, cc] = True

        base_area = int(mask.sum())
        area_floor = max(base_area, int(np.ceil(0.42 * h * w)))
        if area_floor > area_cap:
            continue
        target_area = int(rng.integers(area_floor, area_cap + 1))

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            for rr, cc in zip(*np.where(mask), strict=False):
                rr = int(rr)
                cc = int(cc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(0, len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue
        if bool(mask.all()):
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    return None


def _slot_positions(
    template_h: int,
    template_w: int,
    pad_r: int,
    pad_c: int,
    gap_r: int,
    gap_c: int,
) -> dict[str, tuple[int, int]]:
    middle_r = pad_r + template_h + gap_r
    middle_c = pad_c + template_w + gap_c
    return {
        "top": (pad_r, middle_c),
        "bottom": (pad_r + 2 * (template_h + gap_r), middle_c),
        "left": (middle_r, pad_c),
        "right": (middle_r, pad_c + 2 * (template_w + gap_c)),
    }


def _palette_sequence(
    rng: np.random.Generator, length: int, colors: list[int]
) -> np.ndarray:
    order = list(colors)
    rng.shuffle(order)
    seq = order[:]
    while len(seq) < length:
        options = [color for color in colors if color != seq[-1]]
        seq.append(int(rng.choice(options)))
    return np.array(seq[:length], dtype=int)


def _draw_frame_with_gap(
    grid: np.ndarray,
    frame_r0: int,
    frame_c0: int,
    frame_h: int,
    frame_w: int,
    gap_side: str,
    rng: np.random.Generator,
) -> None:
    grid[frame_r0, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    grid[frame_r0 + frame_h - 1, frame_c0 : frame_c0 + frame_w] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_h, frame_c0] = FRAME_COLOR
    grid[frame_r0 : frame_r0 + frame_h, frame_c0 + frame_w - 1] = FRAME_COLOR

    side_len = frame_w if gap_side in ("top", "bottom") else frame_h
    max_gap = min(5, side_len - 3)
    gap_w = int(rng.integers(2, max_gap + 1))
    gap_start = int(rng.integers(1, side_len - gap_w))

    if gap_side == "top":
        grid[frame_r0, frame_c0 + gap_start : frame_c0 + gap_start + gap_w] = BG
    elif gap_side == "bottom":
        row = frame_r0 + frame_h - 1
        grid[row, frame_c0 + gap_start : frame_c0 + gap_start + gap_w] = BG
    elif gap_side == "left":
        grid[frame_r0 + gap_start : frame_r0 + gap_start + gap_w, frame_c0] = BG
    else:
        col = frame_c0 + frame_w - 1
        grid[frame_r0 + gap_start : frame_r0 + gap_start + gap_w, col] = BG


def _expected_from(mask: np.ndarray, palette_side: str, strip: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    if palette_side in ("top", "bottom"):
        out = np.tile(strip, (h, 1))
    else:
        out = np.tile(strip[:, None], (1, w))
    out[~mask] = BG
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_h = int(rng.integers(TEMPLATE_H_RANGE[0], TEMPLATE_H_RANGE[1] + 1))
    template_w = int(rng.integers(TEMPLATE_W_RANGE[0], TEMPLATE_W_RANGE[1] + 1))
    pad_r = int(rng.integers(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    pad_c = int(rng.integers(INNER_PAD_RANGE[0], INNER_PAD_RANGE[1] + 1))
    gap_r = int(rng.integers(LANE_GAP_RANGE[0], LANE_GAP_RANGE[1] + 1))
    gap_c = int(rng.integers(LANE_GAP_RANGE[0], LANE_GAP_RANGE[1] + 1))

    interior_h = 3 * template_h + 2 * gap_r + 2 * pad_r
    interior_w = 3 * template_w + 2 * gap_c + 2 * pad_c
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    gap_side = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
    palette_side = DIRECTIONS[int(rng.integers(0, len(DIRECTIONS)))]
    palette_len = template_w if palette_side in ("top", "bottom") else template_h
    palette_color_count = int(rng.integers(2, min(4, palette_len) + 1))

    template_count = int(rng.integers(2, 5))
    other_dirs = [direction for direction in DIRECTIONS if direction != gap_side]
    chosen_dirs = [gap_side] + list(rng.choice(other_dirs, size=template_count - 1, replace=False))
    rng.shuffle(chosen_dirs)

    needed_colors = template_count + palette_color_count
    if needed_colors > len(COLOR_POOL):
        return None
    picked_colors = list(rng.choice(np.array(COLOR_POOL, dtype=int), size=needed_colors, replace=False))
    template_colors = picked_colors[:template_count]
    palette_colors = picked_colors[template_count:]

    shapes: list[np.ndarray] = []
    seen_masks: set[bytes] = set()
    for _ in range(template_count):
        mask = _random_mask(rng, template_h, template_w)
        if mask is None:
            return None
        key = mask.tobytes()
        retries = 0
        while key in seen_masks and retries < 40:
            mask = _random_mask(rng, template_h, template_w)
            if mask is None:
                return None
            key = mask.tobytes()
            retries += 1
        if key in seen_masks:
            return None
        seen_masks.add(key)
        shapes.append(mask)

    strip = _palette_sequence(rng, palette_len, palette_colors)
    top_space = 2 if palette_side == "top" else 1
    bottom_space = 2 if palette_side == "bottom" else 1
    left_space = 2 if palette_side == "left" else 1
    right_space = 2 if palette_side == "right" else 1
    top_space += int(rng.integers(OUTER_MARGIN_EXTRA[0], OUTER_MARGIN_EXTRA[1] + 1))
    bottom_space += int(rng.integers(OUTER_MARGIN_EXTRA[0], OUTER_MARGIN_EXTRA[1] + 1))
    left_space += int(rng.integers(OUTER_MARGIN_EXTRA[0], OUTER_MARGIN_EXTRA[1] + 1))
    right_space += int(rng.integers(OUTER_MARGIN_EXTRA[0], OUTER_MARGIN_EXTRA[1] + 1))

    grid_h = top_space + frame_h + bottom_space
    grid_w = left_space + frame_w + right_space
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    frame_r0 = top_space
    frame_c0 = left_space
    _draw_frame_with_gap(grid, frame_r0, frame_c0, frame_h, frame_w, gap_side, rng)

    slots = _slot_positions(template_h, template_w, pad_r, pad_c, gap_r, gap_c)
    interior_r0 = frame_r0 + 1
    interior_c0 = frame_c0 + 1

    active_mask: np.ndarray | None = None
    for direction, color, mask in zip(chosen_dirs, template_colors, shapes, strict=False):
        slot_r, slot_c = slots[direction]
        rr0 = interior_r0 + slot_r
        cc0 = interior_c0 + slot_c
        patch = np.zeros((template_h, template_w), dtype=int)
        patch[mask] = int(color)
        grid[rr0 : rr0 + template_h, cc0 : cc0 + template_w] = patch
        if direction == gap_side:
            active_mask = mask

    assert active_mask is not None

    if palette_side == "top":
        start = int(rng.integers(1, grid_w - palette_len))
        grid[0, start : start + palette_len] = strip
    elif palette_side == "bottom":
        start = int(rng.integers(1, grid_w - palette_len))
        grid[-1, start : start + palette_len] = strip
    elif palette_side == "left":
        start = int(rng.integers(1, grid_h - palette_len))
        grid[start : start + palette_len, 0] = strip
    else:
        start = int(rng.integers(1, grid_h - palette_len))
        grid[start : start + palette_len, -1] = strip

    expected = _expected_from(active_mask, palette_side, strip)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
