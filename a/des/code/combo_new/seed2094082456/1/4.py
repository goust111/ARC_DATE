import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GRAY = 5

PATTERN_COLORS = (1, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300
MAX_GRID_SIZE = 30

PATTERN_LIBRARY = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(np.asarray(mask, dtype=bool), k=k).copy()


def _all_pattern_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in PATTERN_LIBRARY:
        for k in range(4):
            candidate = _rotate_mask(base, k)
            key = candidate.tobytes() + bytes(candidate.shape)
            if key in seen:
                continue
            seen.add(key)
            variants.append(candidate)
    return variants


ALL_PATTERNS = _all_pattern_variants()


def _extract_islands(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    islands: list[dict] = []
    interior_mask = np.zeros_like(grid, dtype=bool)
    for coords in _components(grid == GRAY):
        r0, c0, r1, c1 = _bbox(coords)
        submask = grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY
        frame = np.zeros_like(submask, dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        assert np.array_equal(submask, frame), "gray components must be hollow rectangular frames"
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "islands need a non-empty interior"

        ir0, ic0, ir1, ic1 = r0 + 1, c0 + 1, r1 - 1, c1 - 1
        interior = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
        non_black = [int(v) for v in np.unique(interior) if int(v) != BLACK]
        assert len(non_black) == 1, "each island must contain exactly one monochrome interior pattern"
        pattern_color = non_black[0]
        pattern_cells = interior == pattern_color
        assert np.any(pattern_cells), "each island needs a visible interior pattern"
        pr0, pc0, pr1, pc1 = _bbox(np.argwhere(pattern_cells))
        pattern_mask = pattern_cells[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        interior_mask[ir0 : ir1 + 1, ic0 : ic1 + 1] = True
        islands.append(
            {
                "bbox": (r0, c0, r1, c1),
                "interior_bbox": (ir0, ic0, ir1, ic1),
                "pattern_color": pattern_color,
                "pattern_mask": pattern_mask,
            }
        )
    assert islands, "expected at least one island"
    return islands, interior_mask


def _is_contiguous_line(values: np.ndarray) -> bool:
    values = np.sort(np.asarray(values, dtype=int))
    return bool(np.all(np.diff(values) == 1))


def _find_transfers(grid: np.ndarray, islands: list[dict], interior_mask: np.ndarray) -> list[dict]:
    island_by_border: dict[tuple[int, int], int] = {}
    for idx, island in enumerate(islands):
        r0, c0, r1, c1 = island["bbox"]
        for c in range(c0, c1 + 1):
            island_by_border[(r0, c)] = idx
            island_by_border[(r1, c)] = idx
        for r in range(r0, r1 + 1):
            island_by_border[(r, c0)] = idx
            island_by_border[(r, c1)] = idx

    bridge_mask = (grid == RED) & ~interior_mask
    marker_positions = set(map(tuple, np.argwhere((grid == GREEN) & ~interior_mask)))
    used_markers: set[tuple[int, int]] = set()
    transfers: list[dict] = []
    h, w = grid.shape

    for coords in _components(bridge_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        same_row = int(rows.min()) == int(rows.max()) and _is_contiguous_line(cols)
        same_col = int(cols.min()) == int(cols.max()) and _is_contiguous_line(rows)

        horizontal = False
        vertical = False
        if same_row:
            r = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            horizontal = (
                c0 - 1 >= 0
                and c1 + 1 < w
                and (r, c0 - 1) in island_by_border
                and (r, c1 + 1) in island_by_border
            )
        if same_col:
            c = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            vertical = (
                r0 - 1 >= 0
                and r1 + 1 < h
                and (r0 - 1, c) in island_by_border
                and (r1 + 1, c) in island_by_border
            )

        assert horizontal ^ vertical, "each red bridge must resolve to a unique straight orientation"

        if horizontal:
            r = int(rows[0])
            c0 = int(cols.min())
            c1 = int(cols.max())
            span = c1 - c0 + 1
            assert span % 2 == 1, "horizontal bridges need an odd length"
            midc = c0 + span // 2
            above = (r - 1, midc)
            below = (r + 1, midc)
            above_marked = above in marker_positions
            below_marked = below in marker_positions
            assert above_marked ^ below_marked, "horizontal bridges need one green direction marker"

            left_island = island_by_border[(r, c0 - 1)]
            right_island = island_by_border[(r, c1 + 1)]
            assert left_island != right_island, "a bridge must connect two distinct islands"

            if above_marked:
                source, target, rotation, marker = left_island, right_island, 0, above
            else:
                source, target, rotation, marker = right_island, left_island, 2, below
        else:
            c = int(cols[0])
            r0 = int(rows.min())
            r1 = int(rows.max())
            span = r1 - r0 + 1
            assert span % 2 == 1, "vertical bridges need an odd length"
            midr = r0 + span // 2
            left = (midr, c - 1)
            right = (midr, c + 1)
            left_marked = left in marker_positions
            right_marked = right in marker_positions
            assert left_marked ^ right_marked, "vertical bridges need one green direction marker"

            top_island = island_by_border[(r0 - 1, c)]
            bottom_island = island_by_border[(r1 + 1, c)]
            assert top_island != bottom_island, "a bridge must connect two distinct islands"

            if left_marked:
                source, target, rotation, marker = top_island, bottom_island, 3, left
            else:
                source, target, rotation, marker = bottom_island, top_island, 1, right

        assert marker not in used_markers, "green markers must map one-to-one to bridges"
        used_markers.add(marker)
        transfers.append({"source": source, "target": target, "rotation": rotation})

    assert used_markers == marker_positions, "every visible green marker must belong to a bridge"
    targets = [transfer["target"] for transfer in transfers]
    assert len(targets) == len(set(targets)), "each target island must be addressed by at most one bridge"
    return transfers


def _stamp_center(grid: np.ndarray, interior_bbox: tuple[int, int, int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = interior_bbox
    ih = r1 - r0 + 1
    iw = c1 - c0 + 1
    mh, mw = mask.shape
    assert mh <= ih and mw <= iw, "rotated pattern must fit the target island"
    top = r0 + (ih - mh) // 2
    left = c0 + (iw - mw) // 2
    patch = grid[top : top + mh, left : left + mw]
    patch[mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transport each source island's interior sprite across its marked bridge, rotating it to match the bridge direction.

    Input structure:
    - The background is black.
    - Gray 1-pixel hollow rectangular frames act as islands, and each island contains one monochrome interior pattern.
    - Red straight bridges connect island borders, and a single green pixel beside each bridge midpoint indicates which endpoint is the source.

    Rule:
    - Identify every bridge and decode its source/target direction from the green offset marker.
    - Copy the source island's interior pattern, rotating it by 0, 180, clockwise 90, or counterclockwise 90 when the bridge leaves the source from the right, left, bottom, or top border respectively.
    - Clear the target island's old interior pattern to black, then place the rotated source pattern centered inside the target island while leaving sources, frames, bridges, and markers unchanged.

    Key insight:
    - The green pixel is an off-bridge arrow cue, and the bridge side at the source determines how the copied pattern must be rotated before it is centered in the target island.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    islands, interior_mask = _extract_islands(g)
    transfers = _find_transfers(g, islands, interior_mask)

    out = g.copy()
    for transfer in transfers:
        r0, c0, r1, c1 = islands[transfer["target"]]["interior_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    for transfer in transfers:
        source = islands[transfer["source"]]
        target = islands[transfer["target"]]
        rotated = _rotate_mask(source["pattern_mask"], transfer["rotation"])
        _stamp_center(out, target["interior_bbox"], rotated, int(source["pattern_color"]))

    return out


# --------------- generator ---------------


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    grid[top, left : left + width] = GRAY
    grid[top + height - 1, left : left + width] = GRAY
    grid[top : top + height, left] = GRAY
    grid[top : top + height, left + width - 1] = GRAY


def _choose_pattern(
    rng: random.Random,
    ih: int,
    iw: int,
    prefer_side: str | None = None,
    rotation_after_copy: int | None = None,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    candidates: list[np.ndarray] = []
    for pattern in ALL_PATTERNS:
        ph, pw = pattern.shape
        if ph > ih or pw > iw:
            continue
        if rotation_after_copy is not None and target_shape is not None:
            rotated = _rotate_mask(pattern, rotation_after_copy)
            if rotated.shape[0] > target_shape[0] or rotated.shape[1] > target_shape[1]:
                continue
        candidates.append(pattern)

    assert candidates, "pattern library must supply a fitting pattern"

    if prefer_side in {"left", "right"}:
        preferred = [p for p in candidates if p.shape[1] < iw]
        if preferred:
            candidates = preferred
    if prefer_side in {"up", "down"}:
        preferred = [p for p in candidates if p.shape[0] < ih]
        if preferred:
            candidates = preferred

    return rng.choice(candidates).copy()


def _place_pattern(
    grid: np.ndarray,
    rng: random.Random,
    top: int,
    left: int,
    height: int,
    width: int,
    mask: np.ndarray,
    color: int,
    anchor: str,
) -> None:
    ih = height - 2
    iw = width - 2
    ph, pw = mask.shape
    assert ph <= ih and pw <= iw

    if anchor == "left":
        row_off = rng.randint(0, ih - ph)
        col_off = 0
    elif anchor == "right":
        row_off = rng.randint(0, ih - ph)
        col_off = iw - pw
    elif anchor == "up":
        row_off = 0
        col_off = rng.randint(0, iw - pw)
    elif anchor == "down":
        row_off = ih - ph
        col_off = rng.randint(0, iw - pw)
    elif anchor == "center":
        row_off = (ih - ph) // 2
        col_off = (iw - pw) // 2
    else:
        row_off = rng.randint(0, ih - ph)
        col_off = rng.randint(0, iw - pw)

    patch = grid[top + 1 + row_off : top + 1 + row_off + ph, left + 1 + col_off : left + 1 + col_off + pw]
    patch[mask] = color


def _random_target_anchor(rng: random.Random) -> str:
    return rng.choice(("left", "right", "up", "down", "center", "random"))


def _make_horizontal_gadget(rng: random.Random, left_to_right: bool) -> np.ndarray:
    src_h = rng.randint(5, 6)
    src_w = rng.randint(5, 6)
    tgt_h = rng.randint(5, 6)
    tgt_w = rng.randint(5, 6)
    bridge_len = rng.choice((1, 3))

    if left_to_right:
        source_side = "right"
        rotation = 0
        marker_above = True
        left_h, left_w = src_h, src_w
        right_h, right_w = tgt_h, tgt_w
    else:
        source_side = "left"
        rotation = 2
        marker_above = False
        left_h, left_w = tgt_h, tgt_w
        right_h, right_w = src_h, src_w

    body_h = max(left_h, right_h)
    body_row = max(left_h // 2, right_h // 2)
    left_top = body_row - left_h // 2
    right_top = body_row - right_h // 2
    pad_top = 1 if marker_above else 0
    pad_bottom = 0 if marker_above else 1
    local_h = pad_top + body_h + pad_bottom
    local_w = left_w + bridge_len + right_w
    grid = np.full((local_h, local_w), BLACK, dtype=int)

    bridge_row = pad_top + body_row
    bridge_start = left_w
    grid[bridge_row, bridge_start : bridge_start + bridge_len] = RED
    marker_col = bridge_start + bridge_len // 2
    grid[bridge_row - 1 if marker_above else bridge_row + 1, marker_col] = GREEN

    left_top += pad_top
    right_top += pad_top
    _draw_frame(grid, left_top, 0, left_h, left_w)
    _draw_frame(grid, right_top, left_w + bridge_len, right_h, right_w)

    if left_to_right:
        src_top, src_left, src_height, src_width = left_top, 0, src_h, src_w
        tgt_top, tgt_left, tgt_height, tgt_width = right_top, left_w + bridge_len, tgt_h, tgt_w
    else:
        tgt_top, tgt_left, tgt_height, tgt_width = left_top, 0, tgt_h, tgt_w
        src_top, src_left, src_height, src_width = right_top, left_w + bridge_len, src_h, src_w

    src_color = rng.choice(PATTERN_COLORS)
    tgt_color = rng.choice(tuple(color for color in PATTERN_COLORS if color != src_color))

    src_mask = _choose_pattern(
        rng,
        src_height - 2,
        src_width - 2,
        prefer_side=source_side,
        rotation_after_copy=rotation,
        target_shape=(tgt_height - 2, tgt_width - 2),
    )
    tgt_mask = _choose_pattern(rng, tgt_height - 2, tgt_width - 2)
    _place_pattern(grid, rng, src_top, src_left, src_height, src_width, src_mask, src_color, source_side)
    _place_pattern(grid, rng, tgt_top, tgt_left, tgt_height, tgt_width, tgt_mask, tgt_color, _random_target_anchor(rng))
    return grid


def _make_vertical_gadget(rng: random.Random, top_to_bottom: bool) -> np.ndarray:
    src_h = rng.randint(5, 6)
    src_w = rng.randint(5, 6)
    tgt_h = rng.randint(5, 6)
    tgt_w = rng.randint(5, 6)
    bridge_len = rng.choice((1, 3))

    if top_to_bottom:
        source_side = "down"
        rotation = 3
        marker_left = True
        top_h, top_w = src_h, src_w
        bottom_h, bottom_w = tgt_h, tgt_w
    else:
        source_side = "up"
        rotation = 1
        marker_left = False
        top_h, top_w = tgt_h, tgt_w
        bottom_h, bottom_w = src_h, src_w

    body_w = max(top_w, bottom_w)
    body_col = max(top_w // 2, bottom_w // 2)
    top_left = body_col - top_w // 2
    bottom_left = body_col - bottom_w // 2
    pad_left = 1 if marker_left else 0
    pad_right = 0 if marker_left else 1
    local_h = top_h + bridge_len + bottom_h
    local_w = pad_left + body_w + pad_right
    grid = np.full((local_h, local_w), BLACK, dtype=int)

    bridge_col = pad_left + body_col
    bridge_start = top_h
    grid[bridge_start : bridge_start + bridge_len, bridge_col] = RED
    marker_row = bridge_start + bridge_len // 2
    grid[marker_row, bridge_col - 1 if marker_left else bridge_col + 1] = GREEN

    top_left += pad_left
    bottom_left += pad_left
    _draw_frame(grid, 0, top_left, top_h, top_w)
    _draw_frame(grid, top_h + bridge_len, bottom_left, bottom_h, bottom_w)

    if top_to_bottom:
        src_top, src_left, src_height, src_width = 0, top_left, src_h, src_w
        tgt_top, tgt_left, tgt_height, tgt_width = top_h + bridge_len, bottom_left, tgt_h, tgt_w
    else:
        tgt_top, tgt_left, tgt_height, tgt_width = 0, top_left, tgt_h, tgt_w
        src_top, src_left, src_height, src_width = top_h + bridge_len, bottom_left, src_h, src_w

    src_color = rng.choice(PATTERN_COLORS)
    tgt_color = rng.choice(tuple(color for color in PATTERN_COLORS if color != src_color))

    src_mask = _choose_pattern(
        rng,
        src_height - 2,
        src_width - 2,
        prefer_side=source_side,
        rotation_after_copy=rotation,
        target_shape=(tgt_height - 2, tgt_width - 2),
    )
    tgt_mask = _choose_pattern(rng, tgt_height - 2, tgt_width - 2)
    _place_pattern(grid, rng, src_top, src_left, src_height, src_width, src_mask, src_color, source_side)
    _place_pattern(grid, rng, tgt_top, tgt_left, tgt_height, tgt_width, tgt_mask, tgt_color, _random_target_anchor(rng))
    return grid


def _paste(canvas: np.ndarray, part: np.ndarray, top: int, left: int) -> None:
    h, w = part.shape
    canvas[top : top + h, left : left + w] = part


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    left_to_right = _make_horizontal_gadget(rng, True)
    right_to_left = _make_horizontal_gadget(rng, False)
    top_to_bottom = _make_vertical_gadget(rng, True)
    bottom_to_top = _make_vertical_gadget(rng, False)

    gap_top = rng.randint(1, 2)
    gap_bottom = rng.randint(1, 2)
    gap_rows = rng.randint(1, 2)

    top_row_width = left_to_right.shape[1] + gap_top + right_to_left.shape[1]
    bottom_row_width = top_to_bottom.shape[1] + gap_bottom + bottom_to_top.shape[1]
    content_w = max(top_row_width, bottom_row_width)
    top_row_h = max(left_to_right.shape[0], right_to_left.shape[0])
    bottom_row_h = max(top_to_bottom.shape[0], bottom_to_top.shape[0])
    content_h = top_row_h + gap_rows + bottom_row_h

    if content_w > MAX_GRID_SIZE or content_h > MAX_GRID_SIZE:
        return None

    extra_w = rng.randint(0, MAX_GRID_SIZE - content_w)
    extra_h = rng.randint(0, MAX_GRID_SIZE - content_h)
    left_margin = rng.randint(0, extra_w)
    right_margin = extra_w - left_margin
    top_margin = rng.randint(0, extra_h)
    bottom_margin = extra_h - top_margin

    canvas = np.full((content_h + top_margin + bottom_margin, content_w + left_margin + right_margin), BLACK, dtype=int)

    top_row_left = left_margin + rng.randint(0, content_w - top_row_width)
    bottom_row_left = left_margin + rng.randint(0, content_w - bottom_row_width)
    top_row_top = top_margin
    bottom_row_top = top_margin + top_row_h + gap_rows

    lr_top = top_row_top + rng.randint(0, top_row_h - left_to_right.shape[0])
    rl_top = top_row_top + rng.randint(0, top_row_h - right_to_left.shape[0])
    tb_top = bottom_row_top + rng.randint(0, bottom_row_h - top_to_bottom.shape[0])
    bt_top = bottom_row_top + rng.randint(0, bottom_row_h - bottom_to_top.shape[0])

    _paste(canvas, left_to_right, lr_top, top_row_left)
    _paste(canvas, right_to_left, rl_top, top_row_left + left_to_right.shape[1] + gap_top)
    _paste(canvas, top_to_bottom, tb_top, bottom_row_left)
    _paste(canvas, bottom_to_top, bt_top, bottom_row_left + top_to_bottom.shape[1] + gap_bottom)

    try:
        out = main(canvas)
    except AssertionError:
        return None

    if np.array_equal(out, canvas):
        return None
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-island puzzle")
