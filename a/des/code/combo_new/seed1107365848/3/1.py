import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 400
FRAME_SIDES = ("top", "bottom", "left", "right")
HOLE_PATTERNS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        pts: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            pts.append((rr, cc))
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _submask_from_points(pts: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(pts)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        sub[r - r0, c - c0] = True
    return sub


def _is_square_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h != w or h < 4:
        return False
    target = np.ones((h, w), dtype=bool)
    target[1:-1, 1:-1] = False
    return np.array_equal(mask, target)


def _mask_to_bits(mask: np.ndarray, top: int, left: int, side: int) -> int:
    bits = 0
    for r, c in np.argwhere(mask):
        idx = (top + int(r)) * side + left + int(c)
        bits |= 1 << idx
    return bits


def _bits_to_mask(bits: int, side: int) -> np.ndarray:
    out = np.zeros((side, side), dtype=bool)
    for idx in range(side * side):
        if (bits >> idx) & 1:
            out[idx // side, idx % side] = True
    return out


def _solve_panel_mask(
    fragment_masks: list[np.ndarray], side: int, max_unions: int = 2
) -> tuple[np.ndarray | None, int]:
    items = []
    for mask in fragment_masks:
        fh, fw = mask.shape
        if fh > side or fw > side:
            return None, 0
        placements = []
        for top in range(side - fh + 1):
            for left in range(side - fw + 1):
                placements.append(_mask_to_bits(mask, top, left, side))
        items.append(
            {
                "mask": mask,
                "area": int(mask.sum()),
                "placements": placements,
            }
        )
    items.sort(key=lambda item: (len(item["placements"]), -item["area"]))

    unions: set[int] = set()
    seen: set[tuple[int, int]] = set()

    def dfs(i: int, used: int) -> None:
        if len(unions) >= max_unions:
            return
        key = (i, used)
        if key in seen:
            return
        seen.add(key)
        if i == len(items):
            unions.add(used)
            return
        for placed in items[i]["placements"]:
            if placed & used:
                continue
            dfs(i + 1, used | placed)
            if len(unions) >= max_unions:
                return

    dfs(0, 0)
    if len(unions) != 1:
        return None, len(unions)
    return _bits_to_mask(next(iter(unions)), side), 1


def _find_frames_and_fragments(
    grid: np.ndarray, bg: int
) -> dict[int, tuple[tuple[int, int, int, int], list[np.ndarray]]]:
    out: dict[int, tuple[tuple[int, int, int, int], list[np.ndarray]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components(grid == color)
        frame_bbox = None
        fragment_masks: list[np.ndarray] = []
        for pts in comps:
            sub = _submask_from_points(pts)
            if _is_square_frame(sub):
                assert frame_bbox is None, "expected exactly one frame per color"
                frame_bbox = _bbox(pts)
            else:
                fragment_masks.append(sub)
        if frame_bbox is not None:
            assert fragment_masks, "each frame color must also have scattered fragments"
            out[color] = (frame_bbox, fragment_masks)
    assert out, "expected at least one colored frame"
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's scattered fragments into a square panel and stamp the panel's holes into the matching frame.

    Input structure:
    - Black background contains several color groups; for each non-background color, one component is a hollow square frame near the grid edge and the remaining same-color components are detached fragments.
    - Every frame has a black interior, and all frames use the same one-cell-thick border style.
    - The detached fragments of one color can be translated, without rotation or overlap, to fit inside that color's frame interior as one square panel; uncovered cells inside that panel are the holes.

    Rule:
    - For each color, identify the unique hollow square frame and use its interior size as the target panel size.
    - Translate the same-color fragments until they fit inside that square without overlap; the valid packing is unique.
    - Mark every unoccupied panel cell as a hole, and paint those hole positions with the frame color inside the matching frame.
    - Remove the original scattered fragments and keep only the frames with their stamped hole templates on a black canvas.

    Key insight:
    The fragments themselves are only a packing puzzle; the real output is the negative space left after the uniquely reconstructed panel, copied into the same-color frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    color_info = _find_frames_and_fragments(g, bg)

    out = np.full_like(g, bg)
    for color, (frame_bbox, fragment_masks) in color_info.items():
        r0, c0, r1, c1 = frame_bbox
        side = r1 - r0 - 1
        assert side == c1 - c0 - 1 and side >= 2, "frame interior must be a square"

        packed, n_unions = _solve_panel_mask(fragment_masks, side)
        assert n_unions == 1 and packed is not None, "fragments must pack into a unique square panel"

        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[~packed] = color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), size=parts - 1, replace=False))
    vals = [0] + cuts + [total]
    return [vals[i + 1] - vals[i] for i in range(parts)]


def _carve_holes(panel: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    side = panel.shape[0]
    out = panel.copy()
    hole_ops = int(rng.integers(1, 4))
    carved = 0
    for _ in range(50):
        if carved >= hole_ops:
            break
        pat = HOLE_PATTERNS[int(rng.integers(len(HOLE_PATTERNS)))]
        r0 = int(rng.integers(1, side - 1))
        c0 = int(rng.integers(1, side - 1))
        cells = [(r0 + dr, c0 + dc) for dr, dc in pat]
        if any(not (1 <= r < side - 1 and 1 <= c < side - 1) for r, c in cells):
            continue
        if any(not out[r, c] for r, c in cells):
            continue
        trial = out.copy()
        for r, c in cells:
            trial[r, c] = False
        if not _is_connected(trial):
            continue
        out = trial
        carved += 1
    if carved == 0 or out.all():
        return None
    return out


def _partition_panel(
    panel: np.ndarray, frag_count: int, rng: np.random.Generator
) -> list[np.ndarray] | None:
    area = int(panel.sum())
    if area < 2 * frag_count:
        return None

    cells = [tuple(map(int, rc)) for rc in np.argwhere(panel)]
    targets = _random_composition(area - 2 * frag_count, frag_count, rng)
    targets = [t + 2 for t in targets]
    labels = np.full(panel.shape, -1, dtype=int)
    seeds_idx = rng.choice(len(cells), size=frag_count, replace=False)
    groups: list[set[tuple[int, int]]] = []
    sizes = [0] * frag_count

    for idx, cell_idx in enumerate(seeds_idx):
        cell = cells[int(cell_idx)]
        labels[cell] = idx
        groups.append({cell})
        sizes[idx] = 1

    def frontier(label: int) -> list[tuple[int, int]]:
        pts = set()
        for r, c in groups[label]:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1] and panel[nr, nc] and labels[nr, nc] == -1:
                    pts.add((nr, nc))
        return list(pts)

    remaining = area - frag_count
    while remaining > 0:
        active: list[tuple[int, list[tuple[int, int]]]] = []
        preferred: list[tuple[int, list[tuple[int, int]]]] = []
        for label in range(frag_count):
            fr = frontier(label)
            if not fr:
                continue
            active.append((label, fr))
            if sizes[label] < targets[label]:
                preferred.append((label, fr))
        choices = preferred if preferred else active
        if not choices:
            return None
        label, fr = choices[int(rng.integers(len(choices)))]
        cell = fr[int(rng.integers(len(fr)))]
        labels[cell] = label
        groups[label].add(cell)
        sizes[label] += 1
        remaining -= 1

    fragments: list[np.ndarray] = []
    for label in range(frag_count):
        pts = list(groups[label])
        sub = _submask_from_points(pts)
        if not _is_connected(sub) or int(sub.sum()) < 2:
            return None
        fragments.append(sub)
    return fragments


def _gen_panel(
    rng: np.random.Generator, side: int
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    panel = _carve_holes(np.ones((side, side), dtype=bool), rng)
    if panel is None:
        return None

    frag_count = int(rng.integers(2, min(4, int(panel.sum()) // 2) + 1))
    fragment_masks = _partition_panel(panel, frag_count, rng)
    if fragment_masks is None:
        return None

    packed, n_unions = _solve_panel_mask(fragment_masks, side)
    if n_unions != 1 or packed is None or not np.array_equal(packed, panel):
        return None
    return panel, fragment_masks


def _rect_is_clear(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_rect(blocked: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    hh, ww = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _frame_patch(color: int, inner_side: int) -> np.ndarray:
    patch = np.zeros((inner_side + 2, inner_side + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _place_frame(
    grid: np.ndarray,
    blocked: np.ndarray,
    color: int,
    inner_side: int,
    side_name: str,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    patch = _frame_patch(color, inner_side)
    h, w = patch.shape
    gh, gw = grid.shape
    row_choices: list[int]
    col_choices: list[int]

    if side_name == "top":
        row_choices = [0] if gh == h else [0, 1]
        col_choices = list(range(0, gw - w + 1))
    elif side_name == "bottom":
        row_choices = [gh - h] if gh == h else [gh - h, gh - h - 1]
        col_choices = list(range(0, gw - w + 1))
    elif side_name == "left":
        row_choices = list(range(0, gh - h + 1))
        col_choices = [0] if gw == w else [0, 1]
    else:
        row_choices = list(range(0, gh - h + 1))
        col_choices = [gw - w] if gw == w else [gw - w, gw - w - 1]

    rng.shuffle(row_choices)
    rng.shuffle(col_choices)
    for r0 in row_choices:
        for c0 in col_choices:
            if not _rect_is_clear(blocked, r0, c0, h, w, pad=1):
                continue
            mask = patch != BG
            grid[r0 : r0 + h, c0 : c0 + w][mask] = color
            _mark_rect(blocked, r0, c0, h, w, pad=1)
            return r0, c0
    return None


def _place_fragment(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = mask.shape
    gh, gw = grid.shape
    rows = list(range(1, gh - h))
    cols = list(range(1, gw - w))
    rng.shuffle(rows)
    rng.shuffle(cols)
    for r0 in rows:
        for c0 in cols:
            if not _rect_is_clear(blocked, r0, c0, h, w, pad=1):
                continue
            patch = np.zeros((h, w), dtype=int)
            patch[mask] = color
            grid[r0 : r0 + h, c0 : c0 + w][mask] = color
            _mark_rect(blocked, r0, c0, h, w, pad=1)
            return True
    return False


def _expected_output(
    shape: tuple[int, int],
    frames: list[tuple[int, tuple[int, int], np.ndarray]],
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (r0, c0), panel in frames:
        side = panel.shape[0]
        patch = _frame_patch(color, side)
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = patch
        out[r0 + 1 : r0 + 1 + side, c0 + 1 : c0 + 1 + side][~panel] = color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    inner_side = int(rng.integers(4, 7))
    frame_outer = inner_side + 2
    color_count = int(rng.integers(2, 5))
    min_dim = max(16, 2 * frame_outer + 4)
    if min_dim > 30:
        return None
    gh = int(rng.integers(min_dim, 31))
    gw = int(rng.integers(min_dim, 31))

    grid = np.zeros((gh, gw), dtype=int)
    blocked = np.zeros((gh, gw), dtype=bool)
    colors = list(int(c) for c in rng.choice(COLOR_POOL, size=color_count, replace=False))
    sides = list(FRAME_SIDES)
    rng.shuffle(sides)
    side_choices = sides[:color_count]

    panel_specs: list[tuple[int, np.ndarray, list[np.ndarray]]] = []
    for color in colors:
        panel_pack = None
        for _ in range(50):
            panel_pack = _gen_panel(rng, inner_side)
            if panel_pack is not None:
                break
        if panel_pack is None:
            return None
        panel_specs.append((color, panel_pack[0], panel_pack[1]))

    frame_info: list[tuple[int, tuple[int, int], np.ndarray]] = []
    for (color, panel, _), side_name in zip(panel_specs, side_choices):
        placed = _place_frame(grid, blocked, color, inner_side, side_name, rng)
        if placed is None:
            return None
        frame_info.append((color, placed, panel))

    frag_order: list[tuple[int, np.ndarray]] = []
    for color, _, frags in panel_specs:
        for frag in frags:
            frag_order.append((color, frag))
    rng.shuffle(frag_order)
    for color, frag in frag_order:
        if not _place_fragment(grid, blocked, frag, color, rng):
            return None

    expected = _expected_output(grid.shape, frame_info)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the hole-stamping frame puzzle")
