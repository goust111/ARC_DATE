from __future__ import annotations

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GAP_SIDES = ("top", "bottom", "left", "right")
EDGE_PAD = 1
MAX_GENERATION_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (1, 2, 3)
FRAME_SIDE_CHOICES = (6, 7, 8, 9)
GAP_WIDTH_CHOICES = (2, 3)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


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
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _components(grid == color)


def _line_gap_info(line: np.ndarray, frame_color: int, bg: int) -> tuple[int, int] | None:
    if int(line[0]) != frame_color or int(line[-1]) != frame_color:
        return None
    interior = np.asarray(line[1:-1], dtype=int)
    bad = (interior != frame_color) & (interior != bg)
    if np.any(bad):
        return None
    gap_idx = np.flatnonzero(interior == bg)
    if gap_idx.size == 0:
        return None
    if int(gap_idx.size) not in GAP_WIDTH_CHOICES:
        return None
    if int(gap_idx[-1] - gap_idx[0] + 1) != int(gap_idx.size):
        return None
    return int(gap_idx[0] + 1), int(gap_idx.size)


def _frame_gap(patch: np.ndarray, frame_color: int, bg: int) -> tuple[str, int, int] | None:
    candidates: list[tuple[str, int, int]] = []

    top = _line_gap_info(patch[0, :], frame_color, bg)
    bottom = _line_gap_info(patch[-1, :], frame_color, bg)
    left = _line_gap_info(patch[:, 0], frame_color, bg)
    right = _line_gap_info(patch[:, -1], frame_color, bg)

    if top is not None:
        candidates.append(("top", top[0], top[1]))
    if bottom is not None:
        candidates.append(("bottom", bottom[0], bottom[1]))
    if left is not None:
        candidates.append(("left", left[0], left[1]))
    if right is not None:
        candidates.append(("right", right[0], right[1]))

    if len(candidates) != 1:
        return None

    side, gap_start, gap_width = candidates[0]
    if side != "top" and not np.all(patch[0, 1:-1] == frame_color):
        return None
    if side != "bottom" and not np.all(patch[-1, 1:-1] == frame_color):
        return None
    if side != "left" and not np.all(patch[1:-1, 0] == frame_color):
        return None
    if side != "right" and not np.all(patch[1:-1, -1] == frame_color):
        return None
    return side, gap_start, gap_width


def _transform_patch(patch: np.ndarray, gap_side: str, frame_color: int, bg: int) -> np.ndarray:
    if gap_side in {"top", "bottom"}:
        transformed = np.flipud(patch)
    else:
        transformed = np.fliplr(patch)
    out = np.full(transformed.shape, bg, dtype=int)
    out[transformed != bg] = frame_color
    return out


def _pattern_anchor(frame: dict) -> tuple[int, int]:
    side = int(frame["side"])
    patch = frame["pattern_patch"]
    ph, pw = map(int, patch.shape)
    r = int(frame["r0"]) + 1 + (side - 2 - ph) // 2
    c = int(frame["c0"]) + 1 + (side - 2 - pw) // 2
    return r, c


def _projection_anchor(frame: dict) -> tuple[int, int]:
    side = int(frame["side"])
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    patch = frame["pattern_patch"]
    ph, pw = map(int, patch.shape)
    gap_side = str(frame["gap_side"])

    if gap_side == "top":
        return r0 + side, c0 + (side - pw) // 2
    if gap_side == "bottom":
        return r0 - ph, c0 + (side - pw) // 2
    if gap_side == "left":
        return r0 + (side - ph) // 2, c0 + side
    return r0 + (side - ph) // 2, c0 - pw


def _parse_frame_component(grid: np.ndarray, frame_color: int, coords: np.ndarray, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    side = r1 - r0 + 1
    if side != c1 - c0 + 1 or side < 5:
        return None

    patch = np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    gap = _frame_gap(patch, frame_color, bg)
    if gap is None:
        return None
    gap_side, gap_start, gap_width = gap

    comp_mask = np.zeros((side, side), dtype=bool)
    comp_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if np.any(comp_mask[1:-1, 1:-1]):
        return None

    expected = np.zeros((side, side), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if gap_side == "top":
        expected[0, gap_start : gap_start + gap_width] = False
    elif gap_side == "bottom":
        expected[-1, gap_start : gap_start + gap_width] = False
    elif gap_side == "left":
        expected[gap_start : gap_start + gap_width, 0] = False
    else:
        expected[gap_start : gap_start + gap_width, -1] = False

    if not np.array_equal(comp_mask, expected):
        return None
    if not np.all(patch[expected] == frame_color):
        return None

    interior = patch[1:-1, 1:-1]
    pattern_mask = interior != bg
    if not np.any(pattern_mask):
        return None

    pattern_components = _components(pattern_mask)
    if len(pattern_components) != 1:
        return None

    pattern_color_values = np.unique(interior[pattern_mask])
    if len(pattern_color_values) != 1 or int(pattern_color_values[0]) == frame_color:
        return None

    pr0, pc0, pr1, pc1 = _bbox(pattern_components[0])
    pattern_patch = np.array(interior[pr0 : pr1 + 1, pc0 : pc1 + 1], dtype=int)
    abs_pattern_coords = pattern_components[0] + np.array([r0 + 1, c0 + 1], dtype=int)

    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "side": side,
        "frame_color": int(frame_color),
        "pattern_color": int(pattern_color_values[0]),
        "gap_side": gap_side,
        "gap_start": int(gap_start),
        "gap_width": int(gap_width),
        "pattern_patch": pattern_patch,
        "pattern_coords": abs_pattern_coords,
    }


def _find_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _extract_color_components(g, color):
            info = _parse_frame_component(g, color, coords, bg)
            if info is not None:
                frames.append(info)

    frames.sort(key=lambda frame: (int(frame["r0"]), int(frame["c0"])))
    assert frames, "expected at least one gapped hollow square frame"
    return frames


def _apply_projection(out: np.ndarray, frame: dict, bg: int) -> None:
    proj = _transform_patch(frame["pattern_patch"], str(frame["gap_side"]), int(frame["frame_color"]), bg)
    pr, pc = _projection_anchor(frame)
    ph, pw = proj.shape
    h, w = out.shape
    assert 0 <= pr and 0 <= pc and pr + ph <= h and pc + pw <= w, "projection exceeds grid bounds"
    target = out[pr : pr + ph, pc : pc + pw]
    colored = proj != bg
    assert np.all(target[colored] == bg), "projection must land on background"
    target[colored] = proj[colored]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove each inner motif from its gapped square frame and project it onto the opposite outer wall in the frame's color.

    Input structure:
    - The background is the dominant color.
    - One or more non-background colors form 1-pixel-thick hollow square frames, and each such frame has exactly one contiguous gap of width 2 or 3 on one side.
    - Each frame encloses a single connected inner motif in a different color, while everything outside the frames is background.

    Rule:
    - For each frame, identify the side containing the gap and use the opposite side as the projection wall.
    - Remove the inner motif from the frame interior, mirror it along the gap-to-wall direction, and place it flush outside the opposite wall with center alignment.
    - Recolor the projected motif to the frame color, and leave the frame itself unchanged, including its gap.

    Key insight:
    - The gap indicates the illumination direction rather than the exit: the motif is mirrored onto the wall opposite the gap, and the wall color replaces the motif color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    frames = _find_frames(g)

    out = np.array(g, copy=True)
    for frame in frames:
        coords = frame["pattern_coords"]
        out[coords[:, 0], coords[:, 1]] = bg
    for frame in frames:
        _apply_projection(out, frame, bg)
    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _sample_shape(rng: np.random.Generator, side: int, gap_side: str) -> np.ndarray | None:
    max_dim = min(4, side - 4)
    parity = side % 2
    needs_vertical_flip = gap_side in {"top", "bottom"}

    dims = []
    for h in range(1, max_dim + 1):
        for w in range(1, max_dim + 1):
            if h % 2 != parity or w % 2 != parity:
                continue
            if needs_vertical_flip and h < 2:
                continue
            if not needs_vertical_flip and w < 2:
                continue
            if h * w < 2:
                continue
            dims.append((h, w))
    if not dims:
        return None

    for _ in range(300):
        h, w = dims[int(rng.integers(len(dims)))]
        area_min = max(2, max(h, w))
        area_max = h * w
        area = int(rng.integers(area_min, area_max + 1))
        chosen = rng.choice(h * w, size=area, replace=False)
        mask = np.zeros((h, w), dtype=bool)
        mask[np.unravel_index(chosen, (h, w))] = True
        if not mask.any(axis=0).all() or not mask.any(axis=1).all():
            continue
        if not _is_connected_mask(mask):
            continue
        if needs_vertical_flip and np.array_equal(mask, np.flipud(mask)):
            continue
        if not needs_vertical_flip and np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask
    return None


def _pattern_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _gap_start_options(side: int, gap_width: int) -> list[int]:
    return list(range(1, side - gap_width))


def _spec_union_bbox(spec: dict) -> tuple[int, int, int, int]:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    side = int(spec["side"])
    pr, pc = _projection_anchor(spec)
    ph, pw = map(int, spec["pattern_patch"].shape)
    return (
        min(r0, pr),
        min(c0, pc),
        max(r0 + side - 1, pr + ph - 1),
        max(c0 + side - 1, pc + pw - 1),
    )


def _reserve_clear(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1]))


def _mark_reserved(reserved: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _placement_ranges(h: int, w: int, side: int, ph: int, pw: int, gap_side: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    min_r = EDGE_PAD
    max_r = h - EDGE_PAD - side
    min_c = EDGE_PAD
    max_c = w - EDGE_PAD - side

    if gap_side == "top":
        max_r -= ph
    elif gap_side == "bottom":
        min_r += ph
    elif gap_side == "left":
        max_c -= pw
    else:
        min_c += pw

    if min_r > max_r or min_c > max_c:
        return None
    return (min_r, max_r), (min_c, max_c)


def _choose_position(
    rng: np.random.Generator,
    reserved: np.ndarray,
    side: int,
    pattern_patch: np.ndarray,
    gap_side: str,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    ph, pw = map(int, pattern_patch.shape)
    ranges = _placement_ranges(h, w, side, ph, pw, gap_side)
    if ranges is None:
        return None
    (min_r, max_r), (min_c, max_c) = ranges

    for _ in range(200):
        r0 = int(rng.integers(min_r, max_r + 1))
        c0 = int(rng.integers(min_c, max_c + 1))
        probe = {
            "r0": r0,
            "c0": c0,
            "side": side,
            "gap_side": gap_side,
            "pattern_patch": pattern_patch,
        }
        bbox = _spec_union_bbox(probe)
        if _reserve_clear(reserved, bbox, pad=1):
            return r0, c0
    return None


def _draw_frame(grid: np.ndarray, spec: dict, bg: int) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    side = int(spec["side"])
    color = int(spec["frame_color"])
    grid[r0, c0 : c0 + side] = color
    grid[r0 + side - 1, c0 : c0 + side] = color
    grid[r0 : r0 + side, c0] = color
    grid[r0 : r0 + side, c0 + side - 1] = color

    gap_start = int(spec["gap_start"])
    gap_width = int(spec["gap_width"])
    if spec["gap_side"] == "top":
        grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = bg
    elif spec["gap_side"] == "bottom":
        grid[r0 + side - 1, c0 + gap_start : c0 + gap_start + gap_width] = bg
    elif spec["gap_side"] == "left":
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = bg
    else:
        grid[r0 + gap_start : r0 + gap_start + gap_width, c0 + side - 1] = bg


def _render_input(h: int, w: int, bg: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for spec in specs:
        _draw_frame(grid, spec, bg)
        pr, pc = _pattern_anchor(spec)
        patch = spec["pattern_patch"]
        colored = patch != bg
        target = grid[pr : pr + patch.shape[0], pc : pc + patch.shape[1]]
        assert np.all(target[colored] == bg)
        target[colored] = patch[colored]
    return grid


def _render_expected(inp: np.ndarray, bg: int, specs: list[dict]) -> np.ndarray:
    out = np.array(inp, copy=True)
    for spec in specs:
        pr, pc = _pattern_anchor(spec)
        patch = spec["pattern_patch"]
        colored = patch != bg
        out[pr : pr + patch.shape[0], pc : pc + patch.shape[1]][colored] = bg
    for spec in specs:
        proj = _transform_patch(spec["pattern_patch"], str(spec["gap_side"]), int(spec["frame_color"]), bg)
        pr, pc = _projection_anchor(spec)
        colored = proj != bg
        target = out[pr : pr + proj.shape[0], pc : pc + proj.shape[1]]
        assert np.all(target[colored] == bg)
        target[colored] = proj[colored]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    frame_count = int(rng.choice(FRAME_COUNT_CHOICES))
    min_dim = 18 + 3 * frame_count
    h = int(rng.integers(min_dim, 31))
    w = int(rng.integers(min_dim, 31))
    bg = int(rng.integers(10))

    non_bg = [color for color in range(10) if color != bg]
    frame_colors = list(map(int, rng.choice(non_bg, size=frame_count, replace=False)))

    reserved = np.zeros((h, w), dtype=bool)
    specs: list[dict] = []
    for frame_color in frame_colors:
        placed = False
        for _ in range(300):
            gap_side = str(rng.choice(GAP_SIDES))
            side = int(rng.choice(FRAME_SIDE_CHOICES))
            shape_mask = _sample_shape(rng, side, gap_side)
            if shape_mask is None:
                continue

            pattern_color_choices = [color for color in range(10) if color not in {bg, frame_color}]
            pattern_color = int(rng.choice(pattern_color_choices))
            pattern_patch = _pattern_patch(shape_mask, pattern_color, bg)

            gap_width = int(rng.choice(GAP_WIDTH_CHOICES))
            gap_starts = _gap_start_options(side, gap_width)
            if not gap_starts:
                continue
            gap_start = int(rng.choice(gap_starts))

            pos = _choose_position(rng, reserved, side, pattern_patch, gap_side)
            if pos is None:
                continue
            r0, c0 = pos

            spec = {
                "r0": r0,
                "c0": c0,
                "side": side,
                "frame_color": frame_color,
                "pattern_color": pattern_color,
                "gap_side": gap_side,
                "gap_start": gap_start,
                "gap_width": gap_width,
                "pattern_patch": pattern_patch,
            }
            bbox = _spec_union_bbox(spec)
            if not _reserve_clear(reserved, bbox, pad=1):
                continue

            _mark_reserved(reserved, bbox, pad=1)
            specs.append(spec)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(h, w, bg, specs)
    if _get_bg(inp) != bg:
        return None

    expected = _render_expected(inp, bg, specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
