import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATION_PATTERNS = [
    ("h", "v"),
    ("h", "h"),
    ("v", "v"),
    ("h", "h", "v"),
    ("h", "v", "v"),
    ("h", "h", "v", "v"),
]
GRID_W_RANGE = (14, 31)
GRID_H_RANGE = (18, 31)
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            components.append(
                {
                    "color": color,
                    "mask": mask,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def _row_prefix_lengths(mask: np.ndarray) -> tuple[int, ...] | None:
    lengths = []
    for row in mask:
        count = int(row.sum())
        if count == 0:
            return None
        if not np.all(row[:count]) or np.any(row[count:]):
            return None
        lengths.append(count)
    return tuple(lengths)


def _row_suffix_lengths(mask: np.ndarray) -> tuple[int, ...] | None:
    widths = []
    w = mask.shape[1]
    for row in mask:
        count = int(row.sum())
        if count == 0:
            return None
        start = w - count
        if np.any(row[:start]) or not np.all(row[start:]):
            return None
        widths.append(count)
    return tuple(widths)


def _col_prefix_lengths(mask: np.ndarray) -> tuple[int, ...] | None:
    heights = []
    for col in mask.T:
        count = int(col.sum())
        if count == 0:
            return None
        if not np.all(col[:count]) or np.any(col[count:]):
            return None
        heights.append(count)
    return tuple(heights)


def _col_suffix_lengths(mask: np.ndarray) -> tuple[int, ...] | None:
    heights = []
    h = mask.shape[0]
    for col in mask.T:
        count = int(col.sum())
        if count == 0:
            return None
        start = h - count
        if np.any(col[:start]) or not np.all(col[start:]):
            return None
        heights.append(count)
    return tuple(heights)


def _piece_modes(mask: np.ndarray) -> list[dict]:
    modes = []
    profile = _row_prefix_lengths(mask)
    if profile is not None:
        modes.append({"side": "left", "profile": profile})
    profile = _row_suffix_lengths(mask)
    if profile is not None:
        modes.append({"side": "right", "profile": profile})
    profile = _col_prefix_lengths(mask)
    if profile is not None:
        modes.append({"side": "top", "profile": profile})
    profile = _col_suffix_lengths(mask)
    if profile is not None:
        modes.append({"side": "bottom", "profile": profile})
    return modes


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int, bg: int) -> bool:
    h, w = mask.shape
    patch = canvas[r0 : r0 + h, c0 : c0 + w]
    if np.any((patch != bg) & mask):
        return False
    patch[mask] = color
    return True


def _assemble_candidate(piece_a: dict, mode_a: dict, piece_b: dict, mode_b: dict, bg: int) -> np.ndarray | None:
    side_a = mode_a["side"]
    side_b = mode_b["side"]
    if {side_a, side_b} == {"left", "right"}:
        if side_a == "left":
            left_piece, left_mode = piece_a, mode_a
            right_piece, right_mode = piece_b, mode_b
        else:
            left_piece, left_mode = piece_b, mode_b
            right_piece, right_mode = piece_a, mode_a
        left_profile = left_mode["profile"]
        right_profile = right_mode["profile"]
        if len(left_profile) != len(right_profile):
            return None
        widths = [lp + rp for lp, rp in zip(left_profile, right_profile)]
        if len(set(widths)) != 1:
            return None
        rect_h = len(left_profile)
        rect_w = widths[0]
        out = np.full((rect_h, rect_w), bg, dtype=int)
        right_shift = rect_w - right_piece["mask"].shape[1]
        if not _paint_mask(out, left_piece["mask"], left_piece["color"], 0, 0, bg):
            return None
        if not _paint_mask(out, right_piece["mask"], right_piece["color"], 0, right_shift, bg):
            return None
        if np.any(out == bg):
            return None
        return out
    if {side_a, side_b} == {"top", "bottom"}:
        if side_a == "top":
            top_piece, top_mode = piece_a, mode_a
            bottom_piece, bottom_mode = piece_b, mode_b
        else:
            top_piece, top_mode = piece_b, mode_b
            bottom_piece, bottom_mode = piece_a, mode_a
        top_profile = top_mode["profile"]
        bottom_profile = bottom_mode["profile"]
        if len(top_profile) != len(bottom_profile):
            return None
        heights = [tp + bp for tp, bp in zip(top_profile, bottom_profile)]
        if len(set(heights)) != 1:
            return None
        rect_h = heights[0]
        rect_w = len(top_profile)
        out = np.full((rect_h, rect_w), bg, dtype=int)
        bottom_shift = rect_h - bottom_piece["mask"].shape[0]
        if not _paint_mask(out, top_piece["mask"], top_piece["color"], 0, 0, bg):
            return None
        if not _paint_mask(out, bottom_piece["mask"], bottom_piece["color"], bottom_shift, 0, bg):
            return None
        if np.any(out == bg):
            return None
        return out
    return None


def _rect_sort_key(rect: np.ndarray) -> tuple[int, int, int]:
    h, w = rect.shape
    return (h * w, h, w)


def _pack_layout(rectangles: list[np.ndarray], width: int) -> tuple[list[tuple[int, int]], int, bool]:
    positions: list[tuple[int, int]] = []
    row = 0
    col = 0
    row_h = 0
    wrapped = False
    for rect in rectangles:
        h, w = rect.shape
        if w > width:
            raise AssertionError("packed rectangle wider than output width")
        if col > 0 and col + w > width:
            row += row_h + 1
            col = 0
            row_h = 0
            wrapped = True
        positions.append((row, col))
        col += w + 1
        row_h = max(row_h, h)
    used_h = row + row_h
    return positions, used_h, wrapped


def _pack_rectangles(rectangles: list[np.ndarray], width: int, bg: int) -> np.ndarray:
    positions, used_h, _ = _pack_layout(rectangles, width)
    out = np.full((used_h, width), bg, dtype=int)
    for rect, (r0, c0) in zip(rectangles, positions):
        h, w = rect.shape
        out[r0 : r0 + h, c0 : c0 + w] = rect
    return out


def _solve_pairs(components: list[dict], bg: int) -> list[dict]:
    raw_candidates = []
    for i, piece_a in enumerate(components):
        for j in range(i + 1, len(components)):
            piece_b = components[j]
            for mode_a in piece_a["modes"]:
                for mode_b in piece_b["modes"]:
                    rect = _assemble_candidate(piece_a, mode_a, piece_b, mode_b, bg)
                    if rect is None:
                        continue
                    raw_candidates.append({"i": i, "j": j, "rect": rect})

    seen = set()
    candidates = []
    for cand in raw_candidates:
        key = (cand["i"], cand["j"], cand["rect"].shape, cand["rect"].tobytes())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)

    assert candidates, "expected at least one legal pairing candidate"
    by_piece = {i: [] for i in range(len(components))}
    for idx, cand in enumerate(candidates):
        by_piece[cand["i"]].append(idx)
        by_piece[cand["j"]].append(idx)

    used = [False] * len(components)
    solutions: list[list[int]] = []

    def backtrack(chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        try:
            first = next(i for i, flag in enumerate(used) if not flag)
        except StopIteration:
            solutions.append(chosen.copy())
            return
        for cand_idx in by_piece[first]:
            cand = candidates[cand_idx]
            a, b = cand["i"], cand["j"]
            if used[a] or used[b]:
                continue
            used[a] = True
            used[b] = True
            chosen.append(cand_idx)
            backtrack(chosen)
            chosen.pop()
            used[a] = False
            used[b] = False

    backtrack([])
    assert len(solutions) == 1, "pairing must be unique"
    return [candidates[idx] for idx in solutions[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments into full rectangles, then sort and pack them."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0
    for piece in components:
        piece["modes"] = _piece_modes(piece["mask"])
        assert piece["modes"], "each fragment must expose at least one flat outer edge"
    solved_pairs = _solve_pairs(components, bg)
    rectangles = [cand["rect"] for cand in solved_pairs]
    rectangles.sort(key=_rect_sort_key)
    return _pack_rectangles(rectangles, g.shape[1], bg)


def _make_profile(rng: np.random.Generator, span: int, total: int) -> tuple[int, ...] | None:
    candidates = []
    for low in range(1, total - 1):
        for high in range(low + 2, total):
            candidates.append((low, high))
    if not candidates:
        return None
    low, high = candidates[int(rng.integers(len(candidates)))]
    mids = list(range(low + 1, high))
    if not mids:
        return None
    profile = [0] * span
    profile[0] = int(mids[int(rng.integers(len(mids)))])
    profile[-1] = int(mids[int(rng.integers(len(mids)))])
    inside = [low, high]
    inside.extend(int(rng.integers(low, high + 1)) for _ in range(max(0, span - 4)))
    rng.shuffle(inside)
    profile[1:-1] = inside
    if len(set(profile)) < 2:
        return None
    return tuple(int(v) for v in profile)


def _row_prefix_mask(profile: tuple[int, ...]) -> np.ndarray:
    h = len(profile)
    w = max(profile)
    mask = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(profile):
        mask[r, :width] = True
    return mask


def _row_suffix_mask(profile: tuple[int, ...]) -> np.ndarray:
    h = len(profile)
    w = max(profile)
    mask = np.zeros((h, w), dtype=bool)
    for r, width in enumerate(profile):
        mask[r, w - width :] = True
    return mask


def _col_prefix_mask(profile: tuple[int, ...]) -> np.ndarray:
    h = max(profile)
    w = len(profile)
    mask = np.zeros((h, w), dtype=bool)
    for c, height in enumerate(profile):
        mask[:height, c] = True
    return mask


def _col_suffix_mask(profile: tuple[int, ...]) -> np.ndarray:
    h = max(profile)
    w = len(profile)
    mask = np.zeros((h, w), dtype=bool)
    for c, height in enumerate(profile):
        mask[h - height :, c] = True
    return mask


def _render_spec(mask_a: np.ndarray, side_a: str, color_a: int, mask_b: np.ndarray, side_b: str, color_b: int, bg: int) -> np.ndarray:
    piece_a = {"mask": mask_a, "color": color_a}
    piece_b = {"mask": mask_b, "color": color_b}
    mode_a = {"side": side_a, "profile": _piece_modes(mask_a)[0]["profile"]}
    mode_b = {"side": side_b, "profile": _piece_modes(mask_b)[0]["profile"]}
    rect = _assemble_candidate(piece_a, mode_a, piece_b, mode_b, bg)
    assert rect is not None
    return rect


def _make_pair_specs(rng: np.random.Generator) -> list[dict] | None:
    pattern = ORIENTATION_PATTERNS[int(rng.integers(len(ORIENTATION_PATTERNS)))]
    horizontal_count = sum(orientation == "h" for orientation in pattern)
    vertical_count = len(pattern) - horizontal_count
    horizontal_spans = list(rng.choice(np.array([4, 5, 6]), size=horizontal_count, replace=False))
    vertical_spans = list(rng.choice(np.array([4, 5, 6]), size=vertical_count, replace=False))
    areas: set[int] = set()
    specs = []
    h_idx = 0
    v_idx = 0
    for orientation in pattern:
        built = None
        for _ in range(80):
            if orientation == "h":
                rect_h = int(horizontal_spans[h_idx])
                rect_w = int(rng.integers(5, 8))
                profile = _make_profile(rng, rect_h, rect_w)
                if profile is None:
                    continue
                other = tuple(rect_w - v for v in profile)
                area = rect_h * rect_w
                if area in areas:
                    continue
                built = {
                    "orientation": "h",
                    "mask_a": _row_prefix_mask(profile),
                    "side_a": "left",
                    "mask_b": _row_suffix_mask(other),
                    "side_b": "right",
                    "rect_shape": (rect_h, rect_w),
                    "area": area,
                }
                break
            rect_w = int(vertical_spans[v_idx])
            rect_h = int(rng.integers(5, 8))
            profile = _make_profile(rng, rect_w, rect_h)
            if profile is None:
                continue
            other = tuple(rect_h - v for v in profile)
            area = rect_h * rect_w
            if area in areas:
                continue
            built = {
                "orientation": "v",
                "mask_a": _col_prefix_mask(profile),
                "side_a": "top",
                "mask_b": _col_suffix_mask(other),
                "side_b": "bottom",
                "rect_shape": (rect_h, rect_w),
                "area": area,
            }
            break
        if built is None:
            return None
        areas.add(built["area"])
        specs.append(built)
        if orientation == "h":
            h_idx += 1
        else:
            v_idx += 1
    return specs


def _scatter_pieces(rng: np.random.Generator, grid_h: int, grid_w: int, pieces: list[dict], bg: int) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(rng.permutation(len(pieces)))
    for idx in order:
        piece = pieces[idx]
        mask = piece["mask"]
        color = piece["color"]
        ph, pw = mask.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = int(rng.integers(0, grid_h - ph + 1))
            c0 = int(rng.integers(0, grid_w - pw + 1))
            r1 = r0 + ph
            c1 = c0 + pw
            rr0 = max(0, r0 - 1)
            rr1 = min(grid_h, r1 + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(grid_w, c1 + 1)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue
            grid[r0:r1, c0:c1][mask] = color
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    specs = _make_pair_specs(rng)
    if specs is None:
        return None

    bg = int(rng.integers(10))
    piece_count = 2 * len(specs)
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    colors = colors[:piece_count]

    pieces = []
    rectangles = []
    color_idx = 0
    for spec in specs:
        color_a = int(colors[color_idx])
        color_b = int(colors[color_idx + 1])
        color_idx += 2
        pieces.append({"mask": spec["mask_a"], "color": color_a})
        pieces.append({"mask": spec["mask_b"], "color": color_b})
        rectangles.append(
            _render_spec(
                spec["mask_a"],
                spec["side_a"],
                color_a,
                spec["mask_b"],
                spec["side_b"],
                color_b,
                bg,
            )
        )

    rectangles.sort(key=_rect_sort_key)
    max_rect_w = max(rect.shape[1] for rect in rectangles)
    width_options_by_wrap = {False: [], True: []}
    for grid_w in range(max(max_rect_w, GRID_W_RANGE[0]), GRID_W_RANGE[1]):
        _, used_h, wrapped = _pack_layout(rectangles, grid_w)
        width_options_by_wrap[wrapped].append((grid_w, used_h))
    available_wrap_modes = [wrapped for wrapped, options in width_options_by_wrap.items() if options]
    if not available_wrap_modes:
        return None
    if len(available_wrap_modes) == 2:
        wrap_target = bool(rng.random() < 0.89)
    else:
        wrap_target = bool(available_wrap_modes[0])
    width_options = width_options_by_wrap[wrap_target]
    width_options.sort(key=lambda item: item[0], reverse=wrap_target)
    width_options = width_options[: min(10, len(width_options))]
    rng.shuffle(width_options)

    max_piece_h = max(piece["mask"].shape[0] for piece in pieces)
    for grid_w, _used_h in width_options[:10]:
        min_h = max(GRID_H_RANGE[0], max_piece_h + 2)
        if min_h >= GRID_H_RANGE[1]:
            continue
        heights = list(range(min_h, GRID_H_RANGE[1]))
        rng.shuffle(heights)
        for grid_h in heights[:15]:
            grid = _scatter_pieces(rng, grid_h, grid_w, pieces, bg)
            if grid is None:
                continue
            if _get_bg(grid) != bg:
                continue
            expected = _pack_rectangles(rectangles, grid_w, bg)
            actual = main(grid)
            if np.array_equal(actual, expected):
                return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
