import numpy as np


WALL = 0
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_MAX_RETRIES = 400
GEN_MASK_RETRIES = 100
GEN_PLACE_RETRIES = 250


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _tight_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[str, ...]]:
    return tuple(mask.shape), tuple("".join("1" if v else "0" for v in row) for row in mask)


def _canonical_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[str, ...]]:
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _ring_is_bg(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> bool:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox
    for r in range(r0 - 1, r1 + 2):
        for c in range(c0 - 1, c1 + 2):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if int(grid[r, c]) != bg:
                return False
    return True


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[list[int], tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates = []
    for r0 in range(h - 3):
        for c0 in range(w - 1):
            colors: list[int] = []
            rr = r0
            while rr + 1 < h:
                block = grid[rr : rr + 2, c0 : c0 + 2]
                color = int(block[0, 0])
                if color in (bg, WALL) or not np.all(block == color):
                    break
                if colors and color == colors[-1]:
                    break
                colors.append(color)
                rr += 2
            if len(colors) < 2:
                continue
            bbox = (r0, c0, rr - 1, c0 + 1)
            if _ring_is_bg(grid, bbox, bg):
                candidates.append((c0, -len(colors), r0, colors, bbox))
    assert candidates, "legend not found"
    _, _, _, colors, bbox = min(candidates)
    return colors, bbox


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    comps = _components(grid == WALL)
    assert comps, "frame not found"
    frame = max(comps, key=lambda pts: int(len(pts)))
    return _bbox_coords(frame)


def _extract_slots(frame: np.ndarray, bg: int) -> list[dict]:
    h, w = frame.shape
    slots = []
    for coords in _components(frame == bg):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == h - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == w - 1):
            continue
        mask, bbox = _tight_mask(coords)
        slots.append({"coords": coords, "mask": mask, "bbox": bbox})
    slots.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return slots


def _extract_pieces(
    grid: np.ndarray,
    bg: int,
    legend_bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
) -> list[dict]:
    mask = (grid != bg) & (grid != WALL)
    lr0, lc0, lr1, lc1 = legend_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = False
    mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False

    pieces = []
    for color in sorted(set(map(int, grid[mask]))):
        for coords in _components(mask & (grid == color)):
            pmask, bbox = _tight_mask(coords)
            pieces.append({"color": int(color), "coords": coords, "mask": pmask, "bbox": bbox})
    pieces.sort(key=lambda item: (item["color"], item["bbox"][0], item["bbox"][1]))
    return pieces


def _render_frame(
    frame: np.ndarray,
    slot_assignments: list[tuple[dict, int]],
    priority: dict[int, int],
) -> np.ndarray:
    color_grid = np.full(frame.shape, -1, dtype=int)
    filled = np.zeros(frame.shape, dtype=bool)

    for slot, color in slot_assignments:
        pts = slot["coords"]
        color_grid[pts[:, 0], pts[:, 1]] = int(color)
        filled[pts[:, 0], pts[:, 1]] = True

    out = frame.copy()
    out[filled] = color_grid[filled]

    h, w = frame.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(frame[r, c]) != WALL:
                continue
            candidates: list[int] = []

            if filled[r, c - 1] and filled[r, c + 1]:
                left = int(color_grid[r, c - 1])
                right = int(color_grid[r, c + 1])
                if left != right:
                    candidates.extend((left, right))

            if filled[r - 1, c] and filled[r + 1, c]:
                up = int(color_grid[r - 1, c])
                down = int(color_grid[r + 1, c])
                if up != down:
                    candidates.extend((up, down))

            if candidates:
                out[r, c] = min(candidates, key=lambda color: priority[color])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Place scattered colored fragments into matching frame slots under rotation,
    then resolve one-pixel wall overlaps using the left legend's top-to-bottom
    priority order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    legend_colors, legend_bbox = _detect_legend(g, bg)
    priority = {int(color): idx for idx, color in enumerate(legend_colors)}

    fbbox = _frame_bbox(g)
    fr0, fc0, fr1, fc1 = fbbox
    frame = g[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()

    slots = _extract_slots(frame, bg)
    pieces = _extract_pieces(g, bg, legend_bbox, fbbox)

    assert len(slots) == len(pieces) == len(legend_colors), "slot/piece/legend count mismatch"
    assert {piece["color"] for piece in pieces} == set(legend_colors), "piece colors must match legend colors"

    slot_by_sig: dict[tuple[tuple[int, int], tuple[str, ...]], dict] = {}
    for slot in slots:
        sig = _canonical_signature(slot["mask"])
        assert sig not in slot_by_sig, "slot shapes must be unique up to rotation"
        slot_by_sig[sig] = slot

    assignments = []
    used = set()
    for piece in pieces:
        sig = _canonical_signature(piece["mask"])
        assert sig in slot_by_sig, "piece does not match any slot"
        slot = slot_by_sig[sig]
        slot_id = id(slot)
        assert slot_id not in used, "multiple pieces match the same slot"
        used.add(slot_id)
        assignments.append((slot, piece["color"]))

    return _render_frame(frame, assignments, priority)


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(_components(mask)) == 1


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask
    h, w = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack = []

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    return bool((bg & ~seen).any())


def _random_slot_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray | None:
    max_remove = min(h * w - 4, max(2, (h * w) // 3))
    if max_remove < 1:
        return None

    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(GEN_MASK_RETRIES):
        mask = np.ones((h, w), dtype=bool)
        target = int(rng.integers(1, max_remove + 1))
        rng.shuffle(cells)
        removed = 0
        for r, c in cells:
            if removed >= target:
                break
            test = mask.copy()
            test[r, c] = False
            if not _touches_all_sides(test):
                continue
            if not _connected(test):
                continue
            if _has_hole(test):
                continue
            mask = test
            removed += 1
        if removed >= 1 and _touches_all_sides(mask) and _connected(mask) and not _has_hole(mask):
            return mask
    return None


def _reserve(occ: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 1) -> None:
    h, w = occ.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_patch(
    grid: np.ndarray,
    occ: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    ph, pw = mask.shape
    if ph > h or pw > w:
        return False

    for _ in range(GEN_PLACE_RETRIES):
        r0 = int(rng.integers(0, h - ph + 1))
        c0 = int(rng.integers(0, w - pw + 1))
        if occ[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        patch = grid[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = int(color)
        _reserve(occ, (r0, c0, r0 + ph - 1, c0 + pw - 1), pad=1)
        return True
    return False


def _make_layout(rng: np.random.Generator, n_slots: int) -> tuple[np.ndarray, list[dict], str]:
    layout = "row" if rng.random() < 0.5 else "col"
    slots = []

    if layout == "row":
        common_h = int(rng.integers(4, 7))
        max_w = 6 if n_slots <= 3 else 5
        widths = [int(rng.integers(4, max_w + 1)) for _ in range(n_slots)]
        inner_h = common_h
        inner_w = sum(widths) + (n_slots - 1)
        frame = np.full((inner_h + 2, inner_w + 2), WALL, dtype=int)

        used_sigs = set()
        c = 1
        for w in widths:
            mask = _random_slot_mask(common_h, w, rng)
            if mask is None:
                return None, [], layout
            sig = _canonical_signature(mask)
            if sig in used_sigs:
                return None, [], layout
            used_sigs.add(sig)
            coords = np.argwhere(mask) + np.array([1, c])
            slots.append({"coords": coords, "mask": mask, "bbox": (1, c, common_h, c + w - 1)})
            frame[1 : 1 + common_h, c : c + w][mask] = -1
            c += w + 1
    else:
        common_w = int(rng.integers(4, 7))
        max_h = 6 if n_slots <= 3 else 5
        heights = [int(rng.integers(4, max_h + 1)) for _ in range(n_slots)]
        inner_h = sum(heights) + (n_slots - 1)
        inner_w = common_w
        frame = np.full((inner_h + 2, inner_w + 2), WALL, dtype=int)

        used_sigs = set()
        r = 1
        for h in heights:
            mask = _random_slot_mask(h, common_w, rng)
            if mask is None:
                return None, [], layout
            sig = _canonical_signature(mask)
            if sig in used_sigs:
                return None, [], layout
            used_sigs.add(sig)
            coords = np.argwhere(mask) + np.array([r, 1])
            slots.append({"coords": coords, "mask": mask, "bbox": (r, 1, r + h - 1, common_w)})
            frame[r : r + h, 1 : 1 + common_w][mask] = -1
            r += h + 1

    return frame, slots, layout


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    n_slots = int(rng.integers(2, 5))
    frame_template, slots, layout = _make_layout(rng, n_slots)
    if frame_template is None:
        return None

    palette = [c for c in range(1, 10)]
    bg = int(rng.choice(palette))
    colors = list(rng.choice([c for c in palette if c != bg], size=n_slots, replace=False))
    legend_colors = colors.copy()
    rng.shuffle(legend_colors)
    priority = {int(color): idx for idx, color in enumerate(legend_colors)}

    frame = frame_template.copy()
    frame[frame == -1] = bg
    fh, fw = frame.shape

    piece_records = []
    for slot, color in zip(slots, colors):
        rot_k = int(rng.integers(0, 4))
        piece_mask = np.rot90(slot["mask"], rot_k)
        piece_records.append({"color": int(color), "mask": piece_mask, "area": int(piece_mask.sum())})
    piece_records.sort(key=lambda item: -item["area"])

    legend_h = 2 * n_slots
    max_piece_h = max(int(item["mask"].shape[0]) for item in piece_records)
    max_piece_w = max(int(item["mask"].shape[1]) for item in piece_records)

    if layout == "row":
        grid_h = int(max(20, fh + 12, legend_h + 4, max_piece_h + 4) + rng.integers(0, 4))
        grid_w = int(max(28, fw + 8, max_piece_w + 6) + rng.integers(0, 5))
    else:
        grid_h = int(max(28, fh + 6, legend_h + 4, max_piece_h + 4) + rng.integers(0, 5))
        grid_w = int(max(20, fw + 14, max_piece_w + 6) + rng.integers(0, 4))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occ = np.zeros_like(grid, dtype=bool)

    legend = np.zeros((legend_h, 2), dtype=int)
    for i, color in enumerate(legend_colors):
        legend[2 * i : 2 * i + 2, :] = int(color)

    legend_r0 = int(rng.integers(1, grid_h - legend_h))
    legend_c0 = int(rng.integers(0, 2))
    grid[legend_r0 : legend_r0 + legend_h, legend_c0 : legend_c0 + 2] = legend
    _reserve(occ, (legend_r0, legend_c0, legend_r0 + legend_h - 1, legend_c0 + 1), pad=1)

    min_frame_c0 = max(legend_c0 + 5, 6)
    max_frame_c0 = grid_w - fw - 1
    if min_frame_c0 > max_frame_c0:
        return None
    frame_r0 = int(rng.integers(1, grid_h - fh))
    frame_c0 = int(rng.integers(min_frame_c0, max_frame_c0 + 1))
    grid[frame_r0 : frame_r0 + fh, frame_c0 : frame_c0 + fw] = frame
    _reserve(occ, (frame_r0, frame_c0, frame_r0 + fh - 1, frame_c0 + fw - 1), pad=1)

    for piece in piece_records:
        if not _place_patch(grid, occ, piece["mask"], piece["color"], rng):
            return None

    expected = _render_frame(frame, list(zip(slots, colors)), priority)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_RETRIES):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
