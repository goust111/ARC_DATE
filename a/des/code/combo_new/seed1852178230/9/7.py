import numpy as np


BG = 0
TARGET_FRAME = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                }
            )
    return comps


def _is_rectangle_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 <= r0 or c1 <= c0:
        return False
    coords = set(comp["coords"])
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(coords) != expected:
        return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in coords or (r1, c) not in coords:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in coords or (r, c1) not in coords:
            return False
    return True


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_structure(grid: np.ndarray) -> tuple[list[int], np.ndarray, tuple[int, int, int, int]]:
    bg = _get_bg(grid)
    frames = [comp for comp in _components_by_color(grid, bg) if _is_rectangle_frame(comp)]
    assert len(frames) >= 2

    target_candidates = []
    for comp in frames:
        r0, c0, r1, c1 = comp["bbox"]
        if np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
            target_candidates.append(comp)
    assert len(target_candidates) == 1
    target = target_candidates[0]

    source_frames = [comp for comp in frames if comp is not target]
    assert source_frames
    outer = max(source_frames, key=lambda comp: (comp["bbox"][2] - comp["bbox"][0] + 1) * (comp["bbox"][3] - comp["bbox"][1] + 1))

    bbox_map = {tuple(comp["bbox"]): comp for comp in source_frames}
    chain = [outer]
    while True:
        r0, c0, r1, c1 = chain[-1]["bbox"]
        next_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        nxt = bbox_map.get(next_bbox)
        if nxt is None:
            break
        chain.append(nxt)

    inner_r0, inner_c0, inner_r1, inner_c1 = chain[-1]["bbox"]
    core = grid[inner_r0 + 1 : inner_r1, inner_c0 + 1 : inner_c1]
    core_box = _mask_bbox(core != bg)
    assert core_box is not None
    pr0, pc0, pr1, pc1 = core_box
    pattern = core[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    frame_colors = [int(comp["color"]) for comp in chain]
    return frame_colors, pattern, tuple(target["bbox"])


def _render_reversed_into_target(
    canvas: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    frame_colors: list[int],
    pattern: np.ndarray,
    bg: int,
) -> None:
    tr0, tc0, tr1, tc1 = target_bbox
    target_h = tr1 - tr0 + 1
    target_w = tc1 - tc0 + 1
    pattern_h, pattern_w = pattern.shape

    canvas[tr0 : tr1 + 1, tc0 : tc1 + 1] = bg

    drawn = 0
    for color in reversed(frame_colors):
        remaining_h = target_h - 2 * (drawn + 1)
        remaining_w = target_w - 2 * (drawn + 1)
        if remaining_h < pattern_h or remaining_w < pattern_w:
            break
        _paint_frame(canvas, (tr0 + drawn, tc0 + drawn, tr1 - drawn, tc1 - drawn), color)
        drawn += 1

    inner_r0 = tr0 + drawn
    inner_c0 = tc0 + drawn
    inner_h = target_h - 2 * drawn
    inner_w = target_w - 2 * drawn
    start_r = inner_r0 + (inner_h - pattern_h) // 2
    start_c = inner_c0 + (inner_w - pattern_w) // 2

    for pr in range(pattern_h):
        for pc in range(pattern_w):
            color = int(pattern[pr, pc])
            if color == bg:
                continue
            gr = start_r + pr
            gc = start_c + pc
            if inner_r0 <= gr < inner_r0 + inner_h and inner_c0 <= gc < inner_c0 + inner_w:
                canvas[gr, gc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a nested frame structure inside the target rectangle with the frame-color order reversed.

    Input structure:
    - The background is black.
    - One source structure is a chain of one-cell-thick nested rectangular frames,
      with a different color on each layer.
    - The innermost frame encloses a small single-color asymmetric pattern.
    - Elsewhere, one empty target rectangle is marked by a grey one-cell-thick
      frame around a black interior.

    Rule:
    - Read the source nesting order from the outermost frame to the innermost
      frame, then crop the core pattern from inside the last frame.
    - Clear the target rectangle and redraw the nested frames there in reversed
      order, so the source innermost frame color becomes the target outermost
      frame color.
    - Each new frame is inset by one cell from the previous frame; if drawing
      another layer would leave too little room for the core pattern, stop and
      truncate the remaining inner layers.
    - Center the original core pattern inside the remaining target interior and
      leave the source structure unchanged.

    Key insight:
    - The colored frames encode a containment hierarchy, and the output keeps the
      same geometry but reverses the layer order before placing the unchanged core.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_colors, pattern, target_bbox = _extract_structure(g)
    out = g.copy()
    _render_reversed_into_target(out, target_bbox, frame_colors, pattern, bg)
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.rot90(mask, 2)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, np.rot90(mask, 1)):
            return False
        if np.array_equal(mask, np.rot90(mask, 3)):
            return False
    return True


def _random_pattern(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        ph = int(rng.integers(2, max_h + 1))
        pw = int(rng.integers(2, max_w + 1))
        area = ph * pw
        target_cells = int(rng.integers(max(3, area // 2), area))
        mask = np.zeros((ph, pw), dtype=bool)
        start = (int(rng.integers(ph)), int(rng.integers(pw)))
        mask[start] = True
        while int(mask.sum()) < target_cells:
            frontier: list[tuple[int, int]] = []
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), ph, pw):
                    if not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        bbox = _mask_bbox(mask)
        if bbox != (0, 0, ph - 1, pw - 1):
            continue
        if not _mask_connected(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to sample asymmetric pattern")


def _place_pattern_centered(
    grid: np.ndarray,
    pattern_mask: np.ndarray,
    color: int,
    top: int,
    left: int,
    height: int,
    width: int,
) -> None:
    ph, pw = pattern_mask.shape
    start_r = top + (height - ph) // 2
    start_c = left + (width - pw) // 2
    for pr, pc in np.argwhere(pattern_mask):
        grid[start_r + int(pr), start_c + int(pc)] = color


def _sample_target_dims(
    rng: np.random.Generator,
    source_h: int,
    source_w: int,
    pattern_h: int,
    pattern_w: int,
    layers: int,
) -> tuple[int, int]:
    truncate = bool(rng.integers(0, 2))
    if not truncate:
        return source_h, source_w

    fit_layers = int(rng.integers(1, layers))
    limit_height = bool(rng.integers(0, 2))
    min_h = pattern_h + 2 * fit_layers
    min_w = pattern_w + 2 * fit_layers

    if limit_height:
        max_h = min(source_h - 1, pattern_h + 2 * (fit_layers + 1) - 1)
        target_h = int(rng.integers(min_h, max_h + 1))
        target_w = int(rng.integers(min_w, source_w + 1))
    else:
        max_w = min(source_w - 1, pattern_w + 2 * (fit_layers + 1) - 1)
        target_h = int(rng.integers(min_h, source_h + 1))
        target_w = int(rng.integers(min_w, max_w + 1))

    return target_h, target_w


def _layout_bboxes(
    rng: np.random.Generator,
    source_h: int,
    source_w: int,
    target_h: int,
    target_w: int,
) -> tuple[int, int, int, int, int, int]:
    gap = int(rng.integers(2, 5))
    horizontal_base_h = max(source_h, target_h)
    horizontal_base_w = source_w + gap + target_w
    vertical_base_h = source_h + gap + target_h
    vertical_base_w = max(source_w, target_w)

    horizontal_ok = horizontal_base_h <= 28 and horizontal_base_w <= 28
    vertical_ok = vertical_base_h <= 28 and vertical_base_w <= 28
    assert horizontal_ok or vertical_ok

    if horizontal_ok and vertical_ok:
        horizontal = bool(rng.integers(0, 2))
    else:
        horizontal = horizontal_ok

    if horizontal:
        base_h = horizontal_base_h
        base_w = horizontal_base_w
        extra_h = int(rng.integers(2, min(8, 30 - base_h) + 1))
        extra_w = int(rng.integers(2, min(8, 30 - base_w) + 1))
        grid_h = base_h + extra_h
        grid_w = base_w + extra_w
        top_pad = int(rng.integers(0, extra_h + 1))
        left_pad = int(rng.integers(0, extra_w + 1))
        swap = bool(rng.integers(0, 2))
        source_r0 = top_pad + int(rng.integers(0, base_h - source_h + 1))
        target_r0 = top_pad + int(rng.integers(0, base_h - target_h + 1))
        if swap:
            target_c0 = left_pad
            source_c0 = left_pad + target_w + gap
        else:
            source_c0 = left_pad
            target_c0 = left_pad + source_w + gap
    else:
        base_h = vertical_base_h
        base_w = vertical_base_w
        extra_h = int(rng.integers(2, min(8, 30 - base_h) + 1))
        extra_w = int(rng.integers(2, min(8, 30 - base_w) + 1))
        grid_h = base_h + extra_h
        grid_w = base_w + extra_w
        top_pad = int(rng.integers(0, extra_h + 1))
        left_pad = int(rng.integers(0, extra_w + 1))
        swap = bool(rng.integers(0, 2))
        source_c0 = left_pad + int(rng.integers(0, base_w - source_w + 1))
        target_c0 = left_pad + int(rng.integers(0, base_w - target_w + 1))
        if swap:
            target_r0 = top_pad
            source_r0 = top_pad + target_h + gap
        else:
            source_r0 = top_pad
            target_r0 = top_pad + source_h + gap

    return grid_h, grid_w, source_r0, source_c0, target_r0, target_c0


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    layers = int(rng.integers(2, 5))
    max_core_h = 12 - 2 * layers
    max_core_w = 12 - 2 * layers
    if max_core_h < 3 or max_core_w < 3:
        return None

    core_h = int(rng.integers(3, max_core_h + 1))
    core_w = int(rng.integers(3, max_core_w + 1))
    pattern_mask = _random_pattern(rng, core_h, core_w)
    pattern_h, pattern_w = pattern_mask.shape

    source_h = core_h + 2 * layers
    source_w = core_w + 2 * layers
    target_h, target_w = _sample_target_dims(rng, source_h, source_w, pattern_h, pattern_w, layers)

    if target_h < 2 or target_w < 2:
        return None

    colors = list(map(int, rng.choice(GEN_COLORS, size=layers + 1, replace=False)))
    frame_colors = colors[:layers]
    pattern_color = colors[-1]

    grid_h, grid_w, source_r0, source_c0, target_r0, target_c0 = _layout_bboxes(
        rng, source_h, source_w, target_h, target_w
    )
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    for depth, color in enumerate(frame_colors):
        _paint_frame(
            grid,
            (
                source_r0 + depth,
                source_c0 + depth,
                source_r0 + source_h - 1 - depth,
                source_c0 + source_w - 1 - depth,
            ),
            color,
        )

    _place_pattern_centered(
        grid,
        pattern_mask,
        pattern_color,
        source_r0 + layers,
        source_c0 + layers,
        core_h,
        core_w,
    )
    _paint_frame(
        grid,
        (target_r0, target_c0, target_r0 + target_h - 1, target_c0 + target_w - 1),
        TARGET_FRAME,
    )

    if _get_bg(grid) != BG:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
