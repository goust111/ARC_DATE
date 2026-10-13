import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(grid: np.ndarray) -> tuple[str, int]:
    full_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    full_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    if full_rows and not full_cols:
        assert len(full_rows) == 1
        return "row", int(full_rows[0])
    if full_cols and not full_rows:
        assert len(full_cols) == 1
        return "col", int(full_cols[0])
    raise AssertionError("expected exactly one full grey separator line")


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for pts in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        pts_set = set(pts)
        if pts_set == border:
            frames.append((r0, c0, r1, c1))
    assert frames
    frames.sort()
    return frames


def _palette_band(grid: np.ndarray, axis: str, idx: int) -> np.ndarray:
    h, w = grid.shape
    if axis == "row":
        top_h = idx
        bottom_h = h - idx - 1
        if top_h < bottom_h:
            return grid[:idx, :]
        return grid[idx + 1 :, :]
    left_w = idx
    right_w = w - idx - 1
    if left_w < right_w:
        return grid[:, :idx]
    return grid[:, idx + 1 :]


def _extract_templates(grid: np.ndarray, inner_shape: tuple[int, int]) -> dict[int, np.ndarray]:
    axis, idx = _find_separator(grid)
    band = _palette_band(grid, axis, idx)
    inner_h, inner_w = inner_shape
    templates: dict[int, np.ndarray] = {}
    for color in sorted(int(c) for c in np.unique(band) if int(c) not in {BG, GREY}):
        coords = np.argwhere(band == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (inner_h, inner_w)
        block = band[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert set(int(v) for v in np.unique(block)).issubset({BG, color})
        templates[color] = block.copy()
    assert templates
    return templates


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    inner_sizes = {(r1 - r0 - 1, c1 - c0 - 1) for r0, c0, r1, c1 in frames}
    assert len(inner_sizes) == 1
    inner_shape = next(iter(inner_sizes))
    templates = _extract_templates(g, inner_shape)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        colors = sorted(int(v) for v in np.unique(inner) if int(v) not in {BG, GREY})
        assert len(colors) == 1
        query_color = colors[0]
        template = templates[query_color]
        inner[template != BG] = template[template != BG]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each grey frame interior with the palette template selected by its fragment color.

    Input structure:
    - The background is black.
    - Several equal-sized grey rectangular frames appear in the main area; each frame has a mostly black interior with a few colored fragment pixels.
    - A corner palette region is separated from the main area by one full grey line; the palette contains template blocks matching the frame interior size.

    Rule:
    - For each frame, collect the non-black, non-grey colors inside the frame interior; all fragments in that frame share one query color.
    - In the palette region, find the template block whose foreground color matches the query color.
    - Overlay that template onto the frame interior: template foreground pixels are copied in place, while template black pixels leave the existing interior unchanged.

    Key insight:
    - The fragment pixels are only color keys. Solving the task requires building a color-to-template mapping from the palette, then stamping the matched template into each frame.
    """
    return _solve(grid)


def _random_template(inner_h: int, inner_w: int) -> np.ndarray:
    area = inner_h * inner_w
    for _ in range(200):
        mask = np.zeros((inner_h, inner_w), dtype=bool)
        mask[0, np.random.randint(0, inner_w)] = True
        mask[-1, np.random.randint(0, inner_w)] = True
        mask[np.random.randint(0, inner_h), 0] = True
        mask[np.random.randint(0, inner_h), -1] = True

        extra = np.random.randint(max(1, area // 5), max(2, (3 * area) // 5) + 1)
        picks = np.random.choice(area, size=min(area, extra), replace=False)
        mask.flat[picks] = True

        if mask.all() or not mask.any():
            continue
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        if rows[0] == 0 and rows[-1] == inner_h - 1 and cols[0] == 0 and cols[-1] == inner_w - 1:
            return mask
    raise RuntimeError("failed to generate template")


def _sample_frame_layout(n_frames: int, frame_h: int, frame_w: int) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    for _ in range(200):
        rows = int(np.random.randint(1, 3))
        cols = int(np.random.randint(2, 4))
        if rows * cols < n_frames:
            continue
        gap_y = int(np.random.randint(1, 3))
        gap_x = int(np.random.randint(1, 3))
        positions = [(r, c) for r in range(rows) for c in range(cols)]
        np.random.shuffle(positions)
        chosen = positions[:n_frames]
        main_h = rows * frame_h + (rows - 1) * gap_y + 2
        main_w = cols * frame_w + (cols - 1) * gap_x + 2
        return rows, cols, gap_y, gap_x, sorted(chosen)
    raise RuntimeError("failed to sample frame layout")


def _color_usage(num_templates: int, n_frames: int) -> list[int]:
    used_count = int(np.random.randint(2, min(num_templates, n_frames) + 1))
    usage = list(np.random.choice(np.arange(num_templates), size=used_count, replace=False))
    while len(usage) < n_frames:
        usage.append(int(np.random.choice(usage)))
    np.random.shuffle(usage)
    return usage


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = int(np.random.randint(3, 7))
    inner_w = int(np.random.randint(3, 7))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    n_frames = int(np.random.randint(2, 6))
    n_templates = int(np.random.randint(2, min(5, len(COLOR_POOL)) + 1))
    palette_colors = list(np.random.choice(COLOR_POOL, size=n_templates, replace=False))

    rows, cols, gap_y, gap_x, slots = _sample_frame_layout(n_frames, frame_h, frame_w)
    main_h = rows * frame_h + (rows - 1) * gap_y + 2
    main_w = cols * frame_w + (cols - 1) * gap_x + 2

    orientation = str(np.random.choice(["horizontal", "vertical"]))
    side = int(np.random.randint(0, 2))
    corner = int(np.random.randint(0, 2))
    palette_gap = 1

    if orientation == "horizontal":
        band_h = inner_h + 2
        palette_len = n_templates * inner_w + (n_templates - 1) * palette_gap + 2
        h = band_h + 1 + main_h
        w = max(main_w, palette_len)
    else:
        band_w = inner_w + 2
        palette_len = n_templates * inner_h + (n_templates - 1) * palette_gap + 2
        h = max(main_h, palette_len)
        w = band_w + 1 + main_w

    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), BG, dtype=int)
    expected = grid.copy()

    templates: dict[int, np.ndarray] = {}
    seen_masks: set[bytes] = set()
    for color in palette_colors:
        for _ in range(200):
            mask = _random_template(inner_h, inner_w)
            sig = mask.tobytes()
            if sig not in seen_masks:
                seen_masks.add(sig)
                templates[color] = np.where(mask, color, BG)
                break
        else:
            return None

    if orientation == "horizontal":
        sep_r = inner_h + 1 if side == 0 else h - band_h - 1
        grid[sep_r, :] = GREY
        expected[sep_r, :] = GREY

        main_r0 = sep_r + 1 if side == 0 else 0
        palette_r0 = 0 if side == 0 else sep_r + 1

        x0 = 1 if corner == 0 else w - palette_len + 1
        cur_c = x0
        for color in palette_colors:
            template = templates[color]
            grid[palette_r0 + 1 : palette_r0 + 1 + inner_h, cur_c : cur_c + inner_w] = template
            expected[palette_r0 + 1 : palette_r0 + 1 + inner_h, cur_c : cur_c + inner_w] = template
            cur_c += inner_w + palette_gap

        offset_r = main_r0 + 1
        offset_c = 1
    else:
        sep_c = inner_w + 1 if side == 0 else w - band_w - 1
        grid[:, sep_c] = GREY
        expected[:, sep_c] = GREY

        main_c0 = sep_c + 1 if side == 0 else 0
        palette_c0 = 0 if side == 0 else sep_c + 1

        y0 = 1 if corner == 0 else h - palette_len + 1
        cur_r = y0
        for color in palette_colors:
            template = templates[color]
            grid[cur_r : cur_r + inner_h, palette_c0 + 1 : palette_c0 + 1 + inner_w] = template
            expected[cur_r : cur_r + inner_h, palette_c0 + 1 : palette_c0 + 1 + inner_w] = template
            cur_r += inner_h + palette_gap

        offset_r = 1
        offset_c = main_c0 + 1

    assignments = _color_usage(n_templates, n_frames)
    distinct_used = {palette_colors[idx] for idx in assignments}
    if len(distinct_used) < 2:
        return None

    for frame_idx, (slot_r, slot_c) in enumerate(slots):
        r0 = offset_r + slot_r * (frame_h + gap_y)
        c0 = offset_c + slot_c * (frame_w + gap_x)
        if r0 + frame_h > h or c0 + frame_w > w:
            return None

        grid[r0, c0 : c0 + frame_w] = GREY
        grid[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
        grid[r0 : r0 + frame_h, c0] = GREY
        grid[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY
        expected[r0, c0 : c0 + frame_w] = GREY
        expected[r0 + frame_h - 1, c0 : c0 + frame_w] = GREY
        expected[r0 : r0 + frame_h, c0] = GREY
        expected[r0 : r0 + frame_h, c0 + frame_w - 1] = GREY

        color = palette_colors[assignments[frame_idx]]
        template = templates[color]
        mask = template == color
        coords = np.argwhere(mask)
        frag_count = int(np.random.randint(1, min(4, len(coords)) + 1))
        frag_ids = np.random.choice(len(coords), size=frag_count, replace=False)
        for idx in frag_ids:
            rr, cc = coords[int(idx)]
            grid[r0 + 1 + int(rr), c0 + 1 + int(cc)] = color

        inner_expected = expected[r0 + 1 : r0 + 1 + inner_h, c0 + 1 : c0 + 1 + inner_w]
        inner_expected[mask] = color

    if np.any(grid == GREY) and (np.all(grid[0, :] == GREY) or np.all(grid[-1, :] == GREY) or np.all(grid[:, 0] == GREY) or np.all(grid[:, -1] == GREY)):
        return None

    if not np.array_equal(_solve(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid input")
