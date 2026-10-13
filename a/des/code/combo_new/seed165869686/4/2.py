import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()

BG_COLOR = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 8, 9)
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400
MAX_GROUP_ATTEMPTS = 200

RECT_H_VALUES = (2, 3, 4, 5)
RECT_W_VALUES = (2, 3, 4, 5, 6)
RECT_AREA_MAX = 15


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(len(coords)),
                    "min_pos": min(coords),
                }
            )
    return comps


def _normalize_piece(coords: np.ndarray | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    r0 = int(arr[:, 0].min())
    c0 = int(arr[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in arr.tolist()))


def _piece_dims(piece: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _c in piece) + 1, max(c for _r, c in piece) + 1


def _rectangle_candidates(pieces: list[tuple[tuple[int, int], ...]]) -> list[tuple[int, int]]:
    pieces = sorted(pieces, key=lambda piece: (-len(piece), _piece_dims(piece), piece))
    total_area = sum(len(piece) for piece in pieces)
    candidates: list[tuple[int, int]] = []
    for rect_h in range(1, total_area + 1):
        if total_area % rect_h != 0:
            continue
        rect_w = total_area // rect_h
        if _can_pack_rectangle(pieces, rect_h, rect_w):
            candidates.append((rect_h, rect_w))
    return candidates


def _can_pack_rectangle(
    pieces: list[tuple[tuple[int, int], ...]],
    rect_h: int,
    rect_w: int,
) -> bool:
    placements: list[list[int]] = []
    for piece in pieces:
        piece_h, piece_w = _piece_dims(piece)
        if piece_h > rect_h or piece_w > rect_w:
            return False
        piece_masks: list[int] = []
        for top in range(rect_h - piece_h + 1):
            for left in range(rect_w - piece_w + 1):
                mask = 0
                for r, c in piece:
                    idx = (top + r) * rect_w + (left + c)
                    mask |= 1 << idx
                piece_masks.append(mask)
        placements.append(piece_masks)

    full_mask = (1 << (rect_h * rect_w)) - 1
    memo: dict[tuple[int, int], bool] = {}

    def search(used_mask: int, filled_mask: int) -> bool:
        if filled_mask == full_mask:
            return True
        key = (used_mask, filled_mask)
        if key in memo:
            return memo[key]

        uncovered = full_mask & ~filled_mask
        anchor_bit = uncovered & -uncovered
        for piece_idx, piece_masks in enumerate(placements):
            if (used_mask >> piece_idx) & 1:
                continue
            for mask in piece_masks:
                if (mask & anchor_bit) and not (mask & filled_mask):
                    if search(used_mask | (1 << piece_idx), filled_mask | mask):
                        memo[key] = True
                        return True

        memo[key] = False
        return False

    return search(0, 0)


def _recover_rectangle_dims(pieces: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    candidates = _rectangle_candidates(pieces)
    assert len(candidates) == 1, f"expected one recoverable rectangle, got {candidates}"
    return candidates[0]


def _rotated_dims(rect_h: int, rect_w: int, piece_count: int) -> tuple[int, int]:
    quarter_turns = (piece_count - 1) % 4
    if quarter_turns % 2 == 1:
        return rect_w, rect_h
    return rect_h, rect_w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's complementary fragments into a rectangle, then stack the rectangles after a count-driven rotation.

    Input structure:
    - The background is black.
    - Three or four foreground colors appear.
    - Each foreground color forms one to four separate monochrome 4-connected fragments.
    - Within one color, the fragments can be translated without rotation or flip to tile one solid rectangle.
    - Different color groups have distinct topmost positions; that visible top-to-bottom order defines the output order.

    Rule:
    - For each color independently, collect its fragments and recover the unique rectangle that they tile by translation alone.
    - Let N be the number of fragments of that color.
    - Rotate that rectangle clockwise by (N-1)×90 degrees. Because the rectangle is solid, 90/270 degrees swap height and width while 0/180 degrees keep them.
    - Output only these monochrome rectangles, stacked top-to-bottom in the same visible group order, with one black row between rectangles and left aligned.

    Key insight:
    - The same-color fragments are complementary polyomino pieces of a hidden rectangle, and the fragment count encodes whether the recovered rectangle keeps or swaps its aspect.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    groups: list[dict] = []
    for color in colors:
        comps = _connected_components(g == color)
        assert comps, f"missing fragments for color {color}"
        pieces = [_normalize_piece(comp["coords"]) for comp in comps]
        rect_h, rect_w = _recover_rectangle_dims(pieces)
        out_h, out_w = _rotated_dims(rect_h, rect_w, len(pieces))
        order_key = min(comp["min_pos"] for comp in comps)
        groups.append({"color": color, "order": order_key, "shape": (out_h, out_w)})

    groups.sort(key=lambda item: item["order"])
    out_h = sum(shape_h for shape_h, _shape_w in (group["shape"] for group in groups)) + max(0, len(groups) - 1)
    out_w = max(shape_w for _shape_h, shape_w in (group["shape"] for group in groups))
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for group in groups:
        shape_h, shape_w = group["shape"]
        out[row : row + shape_h, :shape_w] = int(group["color"])
        row += shape_h + 1

    return out


def _generate_group_spec(piece_count: int) -> dict | None:
    rect_options = [
        (rect_h, rect_w)
        for rect_h in RECT_H_VALUES
        for rect_w in RECT_W_VALUES
        if rect_h * rect_w >= max(4, piece_count)
        and rect_h * rect_w <= RECT_AREA_MAX
        and not (piece_count in (2, 4) and rect_h == rect_w)
    ]

    for _ in range(MAX_GROUP_ATTEMPTS):
        rect_h, rect_w = rect_options[int(RNG.integers(len(rect_options)))]
        pieces = _partition_rectangle(rect_h, rect_w, piece_count)
        normalized = [_normalize_piece(piece) for piece in pieces]
        candidates = _rectangle_candidates(normalized)
        if candidates != [(rect_h, rect_w)]:
            continue
        return {
            "piece_count": piece_count,
            "rect_h": rect_h,
            "rect_w": rect_w,
            "pieces": normalized,
        }
    return None


def _partition_rectangle(rect_h: int, rect_w: int, piece_count: int) -> list[list[tuple[int, int]]]:
    cells = [(r, c) for r in range(rect_h) for c in range(rect_w)]
    if piece_count == 1:
        return [cells]

    owner = np.full((rect_h, rect_w), -1, dtype=int)
    seed_ids = RNG.choice(len(cells), size=piece_count, replace=False)
    for label, seed_idx in enumerate(seed_ids.tolist()):
        r, c = cells[seed_idx]
        owner[r, c] = label

    remaining = rect_h * rect_w - piece_count
    while remaining > 0:
        frontier: list[tuple[int, int, int]] = []
        for r in range(rect_h):
            for c in range(rect_w):
                label = int(owner[r, c])
                if label < 0:
                    continue
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rect_h and 0 <= nc < rect_w and owner[nr, nc] < 0:
                        frontier.append((label, nr, nc))
        label, nr, nc = frontier[int(RNG.integers(len(frontier)))]
        owner[nr, nc] = label
        remaining -= 1

    pieces: list[list[tuple[int, int]]] = []
    for label in range(piece_count):
        coords = np.argwhere(owner == label)
        pieces.append([(int(r), int(c)) for r, c in coords.tolist()])
    return pieces


def _expected_output(group_specs: list[dict]) -> np.ndarray:
    shapes = [
        (spec["color"],) + _rotated_dims(spec["rect_h"], spec["rect_w"], spec["piece_count"])
        for spec in group_specs
    ]
    out_h = sum(shape_h for _color, shape_h, _shape_w in shapes) + len(shapes) - 1
    out_w = max(shape_w for _color, _shape_h, shape_w in shapes)
    out = np.full((out_h, out_w), BG_COLOR, dtype=int)
    row = 0
    for color, shape_h, shape_w in shapes:
        out[row : row + shape_h, :shape_w] = int(color)
        row += shape_h + 1
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")


def _gen_attempt() -> np.ndarray | None:
    group_count = int(RNG.choice([3, 4]))
    piece_counts = list(map(int, RNG.permutation([1, 2, 3, 4])[:group_count].tolist()))
    colors = list(map(int, RNG.choice(COLOR_POOL, size=group_count, replace=False).tolist()))

    group_specs: list[dict] = []
    for color, piece_count in zip(colors, piece_counts, strict=False):
        spec = _generate_group_spec(piece_count)
        if spec is None:
            return None
        spec["color"] = color
        piece_dims = [_piece_dims(piece) for piece in spec["pieces"]]
        order = RNG.permutation(len(spec["pieces"])).tolist()
        gaps = [int(RNG.integers(1, 4)) for _ in range(max(0, len(order) - 1))]
        total_w = sum(piece_dims[idx][1] for idx in order) + sum(gaps)
        band_h = max(piece_h for piece_h, _piece_w in piece_dims) + int(RNG.integers(2, 5))
        spec["piece_dims"] = piece_dims
        spec["order"] = order
        spec["gaps"] = gaps
        spec["total_w"] = total_w
        spec["band_h"] = band_h
        group_specs.append(spec)

    expected = _expected_output(group_specs)
    if max(expected.shape) > ARC_MAX:
        return None

    min_grid_w = max(spec["total_w"] for spec in group_specs) + 2
    if min_grid_w > ARC_MAX:
        return None
    grid_w = min_grid_w + int(RNG.integers(0, min(ARC_MAX - min_grid_w, 5) + 1))

    top_margin = int(RNG.integers(1, 3))
    bottom_margin = int(RNG.integers(1, 3))
    band_gaps = [int(RNG.integers(2, 5)) for _ in range(group_count - 1)]
    grid_h = top_margin + bottom_margin + sum(spec["band_h"] for spec in group_specs) + sum(band_gaps)
    if grid_h > ARC_MAX:
        return None

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    row = top_margin
    for group_idx, spec in enumerate(group_specs):
        max_start_col = grid_w - spec["total_w"] - 1
        if max_start_col < 1:
            return None
        col = int(RNG.integers(1, max_start_col + 1))
        for piece_offset, piece_idx in enumerate(spec["order"]):
            piece = spec["pieces"][piece_idx]
            piece_h, piece_w = spec["piece_dims"][piece_idx]
            top = row + int(RNG.integers(0, spec["band_h"] - piece_h + 1))
            for dr, dc in piece:
                rr = top + dr
                cc = col + dc
                if grid[rr, cc] != BG_COLOR:
                    return None
                grid[rr, cc] = int(spec["color"])
            if piece_offset < len(spec["gaps"]):
                col += piece_w + spec["gaps"][piece_offset]

        row += spec["band_h"]
        if group_idx < len(band_gaps):
            row += band_gaps[group_idx]

    if get_bg(grid) != BG_COLOR:
        return None

    result = main(grid)
    if not np.array_equal(result, expected):
        return None
    if max(grid.shape) > ARC_MAX or max(result.shape) > ARC_MAX:
        return None
    return grid
