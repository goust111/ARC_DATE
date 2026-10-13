import numpy as np


BLACK = 0
RED = 2
GREY = 5
PIECE_COLORS = (1, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
GEN_ATTEMPTS = 300


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _component_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (r0, c0, r1, c1)


def _monochrome_components(grid: np.ndarray, allowed: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue
            color = int(grid[r, c])
            if color not in allowed:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            mask, bbox = _component_mask_from_coords(arr)
            out.append({"color": color, "coords": arr, "mask": mask, "bbox": bbox, "area": int(mask.sum())})
    return out


def _is_connected_mask(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _template_mask(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    grey_components = _monochrome_components(grid, {GREY})
    assert grey_components, "expected at least one grey template component"
    template = max(grey_components, key=lambda comp: comp["area"])
    return template["mask"], template["bbox"]


def _piece_masks(grid: np.ndarray) -> list[np.ndarray]:
    allowed = set(range(10)) - {BLACK, GREY}
    pieces = _monochrome_components(grid, allowed)
    pieces.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    return [comp["mask"] for comp in pieces]


def _match_positions(template: np.ndarray, piece: np.ndarray) -> list[tuple[int, int]]:
    th, tw = template.shape
    ph, pw = piece.shape
    exact: list[tuple[int, int]] = []
    subset: list[tuple[int, int]] = []
    for r in range(th - ph + 1):
        for c in range(tw - pw + 1):
            patch = template[r : r + ph, c : c + pw]
            if np.all(~piece | patch):
                subset.append((r, c))
                if np.array_equal(patch, piece):
                    exact.append((r, c))
    if len(exact) == 1:
        return exact
    if len(subset) == 1:
        return subset
    return exact if exact else subset


def _render_counts(template: np.ndarray, placements: list[tuple[np.ndarray, tuple[int, int]]]) -> np.ndarray:
    counts = np.zeros(template.shape, dtype=int)
    for piece, (r, c) in placements:
        ph, pw = piece.shape
        counts[r : r + ph, c : c + pw][piece] += 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the colored shapes as puzzle pieces and embed them into the unique matching
    positions inside the grey template. Crop to the template bbox and mark coverage:
    0 -> black, 1 -> grey, 2+ -> red.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    template, _bbox = _template_mask(g)
    pieces = _piece_masks(g)
    assert pieces, "expected at least one outside piece"

    placements: list[tuple[np.ndarray, tuple[int, int]]] = []
    for piece in pieces:
        matches = _match_positions(template, piece)
        assert len(matches) == 1, "each piece must have a unique embedding"
        placements.append((piece, matches[0]))

    counts = _render_counts(template, placements)
    out = np.full(template.shape, BLACK, dtype=int)
    out[template & (counts == 1)] = GREY
    out[template & (counts >= 2)] = RED
    return out


def _boundary_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                cells.append((rr, cc))
                break
    return cells


def _random_template_mask(rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(9, 15))
        w = int(rng.integers(9, 15))
        mask = np.ones((h, w), dtype=bool)
        target = int(rng.integers(int(0.58 * h * w), int(0.8 * h * w) + 1))

        while int(mask.sum()) > target:
            removable: list[tuple[int, int]] = []
            for r, c in _boundary_cells(mask):
                test = mask.copy()
                test[r, c] = False
                if _is_connected_mask(test):
                    removable.append((r, c))
            if not removable:
                break
            rr, cc = removable[int(rng.integers(len(removable)))]
            mask[rr, cc] = False

        mask = _crop_mask(mask)
        area = int(mask.sum())
        if mask.shape[0] < 8 or mask.shape[1] < 8:
            continue
        if area < max(30, int(0.55 * mask.size)):
            continue
        if np.all(mask):
            continue
        if int((~mask).sum()) < 8:
            continue
        return mask
    return None


def _candidate_windows(template: np.ndarray) -> list[dict]:
    th, tw = template.shape
    raw: list[dict] = []
    counts: dict[tuple[int, int, bytes], int] = {}
    max_h = min(7, th + 1)
    max_w = min(7, tw + 1)

    for h in range(2, max_h):
        for w in range(2, max_w):
            for r in range(th - h + 1):
                for c in range(tw - w + 1):
                    patch = template[r : r + h, c : c + w]
                    area = int(patch.sum())
                    if area < 3 or area > 12:
                        continue
                    if area == h * w:
                        continue
                    if not patch[0, :].any() or not patch[-1, :].any():
                        continue
                    if not patch[:, 0].any() or not patch[:, -1].any():
                        continue
                    if not _is_connected_mask(patch):
                        continue
                    key = (h, w, patch.tobytes())
                    counts[key] = counts.get(key, 0) + 1
                    raw.append({"mask": patch.copy(), "pos": (r, c), "key": key, "area": area})

    return [cand for cand in raw if counts[cand["key"]] == 1]


def _coverage_for_subset(template: np.ndarray, subset: list[dict]) -> np.ndarray:
    placements = [(cand["mask"], cand["pos"]) for cand in subset]
    return _render_counts(template, placements)


def _sample_subset(
    rng: np.random.Generator,
    template: np.ndarray,
    candidates: list[dict],
    require_overlap: bool,
) -> list[dict] | None:
    area = int(template.sum())
    max_k = min(5, len(candidates), len(PIECE_COLORS))
    if max_k < 2:
        return None

    for _ in range(400):
        k = int(rng.integers(2, max_k + 1))
        idx = rng.choice(len(candidates), size=k, replace=False)
        subset = [candidates[int(i)] for i in idx]
        counts = _coverage_for_subset(template, subset)
        covered_any = int(np.sum(template & (counts > 0)))
        covered_once = int(np.sum(template & (counts == 1)))
        covered_multi = int(np.sum(template & (counts >= 2)))
        uncovered = area - covered_any

        if covered_once == 0 or uncovered == 0:
            continue
        if require_overlap and covered_multi == 0:
            continue
        if not require_overlap and covered_multi != 0:
            continue
        if covered_any < max(8, area // 5):
            continue
        if uncovered < max(5, area // 8):
            continue
        if require_overlap and covered_once < 4:
            continue
        if covered_multi > area // 2:
            continue
        return subset
    return None


def _choose_pieces(rng: np.random.Generator, template: np.ndarray, candidates: list[dict]) -> tuple[list[dict], bool] | None:
    if len(candidates) < 2:
        return None

    has_overlap = False
    for i in range(len(candidates)):
        ai = candidates[i]
        a_mask = np.zeros(template.shape, dtype=bool)
        ah, aw = ai["mask"].shape
        ar, ac = ai["pos"]
        a_mask[ar : ar + ah, ac : ac + aw] = ai["mask"]
        for j in range(i + 1, len(candidates)):
            bj = candidates[j]
            bh, bw = bj["mask"].shape
            br, bc = bj["pos"]
            b_mask = np.zeros(template.shape, dtype=bool)
            b_mask[br : br + bh, bc : bc + bw] = bj["mask"]
            if np.any(a_mask & b_mask):
                has_overlap = True
                break
        if has_overlap:
            break

    modes: list[bool]
    if has_overlap and rng.random() < 0.7:
        modes = [True, False]
    elif has_overlap:
        modes = [False, True]
    else:
        modes = [False]

    for require_overlap in modes:
        subset = _sample_subset(rng, template, candidates, require_overlap=require_overlap)
        if subset is not None:
            return subset, require_overlap
    return None


def _mark_blocked(blocked: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(blocked.shape[0], r0 + h + 1)
    cc1 = min(blocked.shape[1], c0 + w + 1)
    blocked[rr0:rr1, cc0:cc1] = True


def _place_objects(
    rng: np.random.Generator,
    template: np.ndarray,
    pieces: list[dict],
) -> np.ndarray | None:
    th, tw = template.shape
    piece_shapes = [cand["mask"].shape for cand in pieces]
    min_h = max(18, th + 6)
    min_w = max(18, tw + 6)
    max_piece_h = max(shape[0] for shape in piece_shapes)
    max_piece_w = max(shape[1] for shape in piece_shapes)
    min_h = max(min_h, th + max_piece_h + 4)
    min_w = max(min_w, tw + max_piece_w + 4)
    max_h = max(min_h + 4, 28)
    max_w = max(min_w + 4, 28)

    for _ in range(80):
        h = int(rng.integers(min_h, max_h + 1))
        w = int(rng.integers(min_w, max_w + 1))
        grid = np.full((h, w), BLACK, dtype=int)
        blocked = np.zeros((h, w), dtype=bool)

        tr = int(rng.integers(1, h - th))
        tc = int(rng.integers(1, w - tw))
        grid[tr : tr + th, tc : tc + tw][template] = GREY
        _mark_blocked(blocked, tr, tc, th, tw)

        colors = list(rng.choice(np.array(PIECE_COLORS, dtype=int), size=len(pieces), replace=False))
        rng.shuffle(colors)

        ok = True
        order = list(range(len(pieces)))
        rng.shuffle(order)
        for idx in order:
            piece = pieces[idx]["mask"]
            ph, pw = piece.shape
            candidates: list[tuple[int, int]] = []
            for pr in range(h - ph + 1):
                for pc in range(w - pw + 1):
                    if blocked[pr : pr + ph, pc : pc + pw].any():
                        continue
                    candidates.append((pr, pc))
            if not candidates:
                ok = False
                break
            pr, pc = candidates[int(rng.integers(len(candidates)))]
            grid[pr : pr + ph, pc : pc + pw][piece] = colors[idx]
            _mark_blocked(blocked, pr, pc, ph, pw)

        if ok:
            return grid
    return None


def _attempt_input(rng: np.random.Generator) -> np.ndarray | None:
    template = _random_template_mask(rng)
    if template is None:
        return None

    candidates = _candidate_windows(template)
    choice = _choose_pieces(rng, template, candidates)
    if choice is None:
        return None
    pieces, require_overlap = choice

    intended_counts = _coverage_for_subset(template, pieces)
    intended = np.full(template.shape, BLACK, dtype=int)
    intended[template & (intended_counts == 1)] = GREY
    intended[template & (intended_counts >= 2)] = RED

    if require_overlap and not np.any(intended == RED):
        return None
    if not np.any(intended == GREY):
        return None
    if not np.any(intended == BLACK):
        return None

    grid = _place_objects(rng, template, pieces)
    if grid is None:
        return None

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, intended):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_input(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-5 input")
