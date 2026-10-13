import numpy as np


BG = 0
GREY = 5
MAROON = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_ORDER = ("tl", "tr", "br", "bl")
CORNER_INDEX = {name: idx for idx, name in enumerate(CORNER_ORDER)}
RNG = np.random.default_rng()
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


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
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    if not comps:
        raise AssertionError("expected at least one component")
    largest = max(comps, key=len)
    out = np.zeros_like(mask, dtype=bool)
    out[largest[:, 0], largest[:, 1]] = True
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_object(grid: np.ndarray, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    mask = np.zeros(patch.shape, dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "patch": patch,
        "mask": mask,
        "area": int(len(coords)),
    }


def _corner_positions(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = shape
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "br": (h - 1, w - 1),
        "bl": (h - 1, 0),
    }


def _corner_name(pos: tuple[int, int], shape: tuple[int, int]) -> str:
    for name, corner_pos in _corner_positions(shape).items():
        if pos == corner_pos:
            return name
    raise AssertionError(f"position {pos} is not a bbox corner for shape {shape}")


def _rotation_k(src_corner: str, dst_corner: str) -> int:
    return (CORNER_INDEX[src_corner] - CORNER_INDEX[dst_corner]) % 4


def _boundary_reachable(open_mask: np.ndarray) -> np.ndarray:
    open_mask = np.asarray(open_mask, dtype=bool)
    h, w = open_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return seen


def _annotate_fragment(grid: np.ndarray, coords: np.ndarray) -> dict:
    obj = _extract_object(grid, coords)
    patch = obj["patch"]
    maroons = np.argwhere(patch == MAROON)
    if maroons.shape[0] != 1:
        raise AssertionError("each fragment must contain exactly one maroon marker")
    rel_marker = tuple(map(int, maroons[0]))
    color_set = {int(v) for v in patch[obj["mask"]]}
    piece_colors = color_set - {MAROON}
    if len(piece_colors) != 1:
        raise AssertionError("each fragment must have exactly one non-maroon color")
    obj["marker_corner"] = _corner_name(rel_marker, patch.shape)
    obj["piece_color"] = next(iter(piece_colors))
    return obj


def _annotate_hole(grid: np.ndarray, coords: np.ndarray) -> dict:
    obj = _extract_object(grid, coords)
    patch = obj["patch"]
    maroons = np.argwhere(patch == MAROON)
    if maroons.shape[0] != 1:
        raise AssertionError("each hole must contain exactly one maroon marker")
    rel_marker = tuple(map(int, maroons[0]))
    obj["marker_corner"] = _corner_name(rel_marker, patch.shape)
    return obj


def _match_fragments(fragments: list[dict], holes: list[dict]) -> dict[int, tuple[int, int]]:
    candidates: dict[int, list[tuple[int, int]]] = {}
    for fi, fragment in enumerate(fragments):
        options = []
        for hi, hole in enumerate(holes):
            if fragment["area"] != hole["area"]:
                continue
            k = _rotation_k(fragment["marker_corner"], hole["marker_corner"])
            rotated_mask = np.rot90(fragment["mask"], k)
            if rotated_mask.shape != hole["mask"].shape:
                continue
            if np.array_equal(rotated_mask, hole["mask"]):
                options.append((hi, k))
        if not options:
            raise AssertionError("every fragment must match at least one hole")
        candidates[fi] = options

    order = sorted(range(len(fragments)), key=lambda idx: len(candidates[idx]))
    used_holes: set[int] = set()
    assignment: dict[int, tuple[int, int]] = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        fi = order[pos]
        for hi, k in candidates[fi]:
            if hi in used_holes:
                continue
            used_holes.add(hi)
            assignment[fi] = (hi, k)
            if dfs(pos + 1):
                return True
            used_holes.remove(hi)
            del assignment[fi]
        return False

    if not dfs(0):
        raise AssertionError("fragment-hole matching should be unique and feasible")
    return assignment


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered marked fragments into the matching holes inside a grey outline.

    Input structure:
    - The background is black.
    - A large grey connected region acts as the target outline and encloses several non-grey hole regions.
    - Outside the outline are several 4-connected monochrome fragments, each with exactly one maroon marker replacing one corner pixel of its bbox.
    - Each enclosed hole region contains exactly one maroon pixel at one corner of that hole's bbox.

    Rule:
    - Pair each fragment with the hole of the same area.
    - Determine the rotation by comparing the fragment marker corner with the hole marker corner.
    - Rotate the fragment so the maroon corner aligns with the hole's maroon corner, then move it into that hole.
    - Keep the maroon marker maroon in the output and clear the fragment's original outside position back to black.

    Key insight:
    - Area identifies the matching fragment-hole pair, but the maroon corner position encodes the exact quarter-turn needed before placement.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    board_mask = _largest_component(g == GREY)

    colored_mask = (g != BG) & (g != GREY)
    colored_components = _components(colored_mask)
    fragment_mask = np.zeros_like(g, dtype=bool)
    fragments = []
    for coords in colored_components:
        values = {int(g[r, c]) for r, c in coords}
        non_maroon = values - {MAROON}
        if not non_maroon:
            continue
        fragment = _annotate_fragment(g, coords)
        fragments.append(fragment)
        fragment_mask[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = BG

    if not fragments:
        raise AssertionError("expected at least one outside fragment")

    blocked = board_mask | fragment_mask
    reachable = _boundary_reachable(~blocked)
    hole_area_mask = (~blocked) & (~reachable)

    holes = []
    for coords in _components(hole_area_mask):
        values = {int(g[r, c]) for r, c in coords}
        if MAROON not in values:
            continue
        holes.append(_annotate_hole(g, coords))

    if len(holes) != len(fragments):
        raise AssertionError("fragment and hole counts should match")

    assignment = _match_fragments(fragments, holes)
    for fi, fragment in enumerate(fragments):
        hi, k = assignment[fi]
        hole = holes[hi]
        rotated_patch = np.rot90(fragment["patch"], k)
        rotated_mask = np.rot90(fragment["mask"], k)
        hr0, hc0, _hr1, _hc1 = hole["bbox"]
        view = out[hr0 : hr0 + rotated_patch.shape[0], hc0 : hc0 + rotated_patch.shape[1]]
        view[rotated_mask] = rotated_patch[rotated_mask]

    return out


def _mask_has_holes(mask: np.ndarray) -> bool:
    bg_mask = ~np.asarray(mask, dtype=bool)
    if not bg_mask.any():
        return False
    reachable = _boundary_reachable(bg_mask)
    return bool((bg_mask & ~reachable).any())


def _random_piece_mask(area: int, corner: str) -> np.ndarray:
    feasible_dims = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if h + w - 1 <= area <= h * w
    ]
    for _ in range(200):
        h, w = feasible_dims[int(RNG.integers(len(feasible_dims)))]
        corner_pos = _corner_positions((h, w))[corner]
        cr, cc = corner_pos
        cells = set()
        row_range = range(0, h) if cr == 0 else range(h - 1, -1, -1)
        col_range = range(0, w) if cc == 0 else range(w - 1, -1, -1)
        for r in row_range:
            cells.add((r, cc))
        for c in col_range:
            cells.add((cr, c))
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c, h, w):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = tuple(frontier)[int(RNG.integers(len(frontier)))]
            cells.add((nr, nc))
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _mask_has_holes(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a valid piece mask")


def _make_piece_patch(mask: np.ndarray, color: int, corner: str) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    marker_pos = _corner_positions(mask.shape)[corner]
    patch[marker_pos] = MAROON
    return patch


def _random_row_partition(count: int) -> list[int]:
    parts = []
    remaining = count
    while remaining > 0:
        if remaining == 1:
            take = 1
        else:
            take = int(RNG.integers(1, min(2, remaining) + 1))
        parts.append(take)
        remaining -= take
    return parts


def _arrange_boxes(boxes: list[np.ndarray]) -> tuple[list[tuple[int, int]], int, int]:
    order = list(range(len(boxes)))
    RNG.shuffle(order)
    row_sizes = _random_row_partition(len(boxes))
    rows = []
    cursor = 0
    for take in row_sizes:
        rows.append(order[cursor : cursor + take])
        cursor += take

    row_data = []
    inner_w = 0
    for row in rows:
        gaps = [int(RNG.integers(1, 3)) for _ in range(max(0, len(row) - 1))]
        width = 0
        height = 0
        for idx, box_idx in enumerate(row):
            h, w = boxes[box_idx].shape
            width += w
            if idx:
                width += gaps[idx - 1]
            height = max(height, h)
        inner_w = max(inner_w, width)
        row_data.append({"row": row, "gaps": gaps, "width": width, "height": height})

    positions = [None] * len(boxes)
    y = 0
    for row_idx, data in enumerate(row_data):
        if row_idx:
            y += int(RNG.integers(1, 3))
        start_x = int(RNG.integers(0, inner_w - data["width"] + 1))
        x = start_x
        for idx, box_idx in enumerate(data["row"]):
            h, w = boxes[box_idx].shape
            y_offset = int(RNG.integers(0, data["height"] - h + 1))
            positions[box_idx] = (y + y_offset, x)
            x += w
            if idx < len(data["gaps"]):
                x += data["gaps"][idx]
        y += data["height"]

    inner_h = y
    return positions, inner_h, inner_w


def _rect_subarray(h: int, w: int, r0: int, c0: int, rh: int, rw: int, pad: int) -> tuple[int, int, int, int]:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + rh + pad)
    cc1 = min(w, c0 + rw + pad)
    return rr0, rr1, cc0, cc1


def _place_pieces(
    grid_shape: tuple[int, int], board_rect: tuple[int, int, int, int], piece_patches: list[np.ndarray]
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    br0, bc0, br1, bc1 = board_rect
    reserved = np.zeros((h, w), dtype=bool)
    reserved[br0 : br1 + 1, bc0 : bc1 + 1] = True
    placements = [None] * len(piece_patches)
    order = sorted(range(len(piece_patches)), key=lambda idx: -piece_patches[idx].size)
    for idx in order:
        patch = piece_patches[idx]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr0, rr1, cc0, cc1 = _rect_subarray(h, w, r0, c0, ph, pw, pad=1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(RNG.integers(len(candidates)))]
        placements[idx] = (r0, c0)
        reserved[r0 : r0 + ph, c0 : c0 + pw] = True
    return placements


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.integers(2, 5))
    areas = RNG.choice(np.arange(4, 11), size=count, replace=False)
    colors = RNG.choice(np.array(PIECE_COLORS, dtype=int), size=count, replace=False)

    pieces = []
    source_corners = []
    target_corners = []
    for area, color in zip(areas.tolist(), colors.tolist()):
        src_corner = CORNER_ORDER[int(RNG.integers(4))]
        dst_corner = CORNER_ORDER[int(RNG.integers(4))]
        mask = _random_piece_mask(int(area), src_corner)
        patch = _make_piece_patch(mask, int(color), src_corner)
        k = _rotation_k(src_corner, dst_corner)
        rotated_patch = np.rot90(patch, k)
        rotated_mask = np.rot90(mask, k)
        pieces.append(
            {
                "area": int(area),
                "color": int(color),
                "patch": patch,
                "mask": mask,
                "rotated_patch": rotated_patch,
                "rotated_mask": rotated_mask,
            }
        )
        source_corners.append(src_corner)
        target_corners.append(dst_corner)

    hole_boxes = [piece["rotated_patch"] for piece in pieces]
    local_positions, inner_h, inner_w = _arrange_boxes(hole_boxes)

    top_margin = int(RNG.integers(2, 4))
    bottom_margin = int(RNG.integers(2, 4))
    left_margin = int(RNG.integers(2, 4))
    right_margin = int(RNG.integers(2, 4))
    board_h = inner_h + top_margin + bottom_margin
    board_w = inner_w + left_margin + right_margin
    min_h = max(20, board_h + 4)
    min_w = max(20, board_w + 4)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(RNG.integers(min_h, 31))
    grid_w = int(RNG.integers(min_w, 31))
    board_r0 = int(RNG.integers(1, grid_h - board_h))
    board_c0 = int(RNG.integers(1, grid_w - board_w))
    board_r1 = board_r0 + board_h - 1
    board_c1 = board_c0 + board_w - 1

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    inp[board_r0 : board_r1 + 1, board_c0 : board_c1 + 1] = GREY
    out[board_r0 : board_r1 + 1, board_c0 : board_c1 + 1] = GREY

    for piece, (local_r, local_c) in zip(pieces, local_positions):
        patch = piece["rotated_patch"]
        mask = piece["rotated_mask"]
        h, w = patch.shape
        r0 = board_r0 + top_margin + local_r
        c0 = board_c0 + left_margin + local_c
        in_view = inp[r0 : r0 + h, c0 : c0 + w]
        out_view = out[r0 : r0 + h, c0 : c0 + w]
        in_view[mask] = BG
        marker = patch == MAROON
        in_view[marker] = MAROON
        out_view[mask] = patch[mask]

    placements = _place_pieces(
        inp.shape,
        (board_r0, board_c0, board_r1, board_c1),
        [piece["patch"] for piece in pieces],
    )
    if placements is None:
        return None

    for piece, (r0, c0) in zip(pieces, placements):
        patch = piece["patch"]
        mask = piece["mask"]
        view = inp[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[mask] = patch[mask]

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
