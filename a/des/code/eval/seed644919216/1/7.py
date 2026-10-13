from __future__ import annotations

import numpy as np


FOUR_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in FOUR_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
    )


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    b = mask.astype(np.uint8, copy=False)
    return int(b.shape[0]), int(b.shape[1]), b.tobytes()


def _canonical_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    return min(_mask_signature(np.rot90(mask, k)) for k in range(4))


def _edge_profiles(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(int(v) for v in mask[0, :]),
        tuple(int(v) for v in mask[:, -1]),
        tuple(int(v) for v in mask[-1, :]),
        tuple(int(v) for v in mask[:, 0]),
    )


def _uniform_separator(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape
    for r in range(h):
        row = grid[r, :]
        if int(row[0]) != bg and bool(np.all(row == row[0])):
            return "h", int(r)
    for c in range(w):
        col = grid[:, c]
        if int(col[0]) != bg and bool(np.all(col == col[0])):
            return "v", int(c)
    raise AssertionError("expected a full grey-style separator line")


def _side_slices(shape: tuple[int, int], orient: str, idx: int) -> list[tuple[slice, slice]]:
    h, w = map(int, shape)
    if orient == "h":
        return [(slice(0, idx), slice(0, w)), (slice(idx + 1, h), slice(0, w))]
    return [(slice(0, h), slice(0, idx)), (slice(0, h), slice(idx + 1, w))]


def _largest_component_in_region(
    grid: np.ndarray, region: tuple[slice, slice], bg: int
) -> tuple[int, np.ndarray]:
    rs, cs = region
    sub = grid[rs, cs]
    comps = _components(sub != bg)
    assert comps, "expected non-background content on both sides of separator"
    largest = max(comps, key=len)
    r0 = int(rs.start or 0)
    c0 = int(cs.start or 0)
    global_coords = largest + np.array([r0, c0], dtype=int)
    return int(len(largest)), global_coords


def _extract_piece_infos(grid: np.ndarray, region: tuple[slice, slice], bg: int) -> list[dict]:
    rs, cs = region
    sub = grid[rs, cs]
    infos: list[dict] = []
    for coords in _components(sub != bg):
        br0, bc0, br1, bc1 = _bbox(coords)
        patch = sub[br0 : br1 + 1, bc0 : bc1 + 1].copy()
        vals = np.unique(patch[patch != bg])
        assert len(vals) == 1, "each puzzle piece must use exactly one visible color"
        infos.append(
            {
                "bbox": (
                    int((rs.start or 0) + br0),
                    int((cs.start or 0) + bc0),
                    int((rs.start or 0) + br1),
                    int((cs.start or 0) + bc1),
                ),
                "patch": patch,
                "color": int(vals[0]),
            }
        )
    assert infos, "expected at least one loose puzzle piece"
    return infos


def _extract_hole_infos(grid: np.ndarray, board_bbox: tuple[int, int, int, int], bg: int) -> list[dict]:
    r0, c0, r1, c1 = map(int, board_bbox)
    board = grid[r0 : r1 + 1, c0 : c1 + 1]
    infos: list[dict] = []
    for coords in _components(board == bg):
        if _touches_edge(coords, board.shape):
            continue
        hr0, hc0, hr1, hc1 = _bbox(coords)
        hole_mask = board[hr0 : hr1 + 1, hc0 : hc1 + 1] == bg
        infos.append(
            {
                "bbox": (int(r0 + hr0), int(c0 + hc0), int(r0 + hr1), int(c0 + hc1)),
                "mask": hole_mask,
                "profiles": _edge_profiles(hole_mask),
            }
        )
    assert infos, "expected internal cavities inside the target board"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate scattered puzzle pieces and insert them into the matching cavities.

    The input contains a full separator line that splits the canvas into a board
    side and a loose-piece side. The board side has one large component with
    internal background cavities; the other side contains detached one-color
    pieces. A piece fits one cavity after a rotation when its occupied edge
    pixels match the cavity's open edge pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_orient, sep_idx = _uniform_separator(g, bg)
    sides = _side_slices(g.shape, sep_orient, sep_idx)

    side_infos = [_largest_component_in_region(g, region, bg) for region in sides]
    board_side_idx = 0 if side_infos[0][0] > side_infos[1][0] else 1
    board_coords = side_infos[board_side_idx][1]
    board_bbox = _bbox(board_coords)
    piece_region = sides[1 - board_side_idx]

    holes = _extract_hole_infos(g, board_bbox, bg)
    pieces = _extract_piece_infos(g, piece_region, bg)
    assert len(holes) == len(pieces), "hole and piece counts must agree"

    out = g.copy()
    out[piece_region] = bg

    used_holes: set[int] = set()
    for piece in pieces:
        candidates: list[tuple[int, int, np.ndarray]] = []
        for k in range(4):
            rotated = np.rot90(piece["patch"], k)
            rmask = rotated != bg
            rprofiles = _edge_profiles(rmask)
            for hole_idx, hole in enumerate(holes):
                if hole_idx in used_holes:
                    continue
                if rmask.shape != hole["mask"].shape:
                    continue
                if rprofiles != hole["profiles"]:
                    continue
                if not np.array_equal(rmask, hole["mask"]):
                    continue
                candidates.append((int(hole_idx), int(k), rotated))
        assert len(candidates) == 1, "each piece must match exactly one cavity"
        hole_idx, _rot, rotated = candidates[0]
        used_holes.add(hole_idx)
        r0, c0, r1, c1 = holes[hole_idx]["bbox"]
        window = out[r0 : r1 + 1, c0 : c1 + 1]
        mask = rotated != bg
        window[mask] = rotated[mask]

    assert len(used_holes) == len(holes), "all cavities must be filled"
    return out


# --------------- generator ---------------

GEN_RETRIES = 400
PIECE_COUNT_OPTIONS = (2, 3, 4)
PIECE_SIZE_RANGE = (4, 5)  # inclusive
PIECE_COLORS = np.array(list(range(10)), dtype=int)


def _random_profile(size: int) -> np.ndarray:
    prof = np.ones(size, dtype=bool)
    interior = np.arange(1, size - 1)
    max_zeroes = min(3, len(interior))
    zeroes = int(np.random.randint(1, max_zeroes + 1))
    chosen = np.random.choice(interior, size=zeroes, replace=False)
    prof[chosen] = False
    return prof


def _random_piece_mask(size: int) -> np.ndarray | None:
    mask = np.ones((size, size), dtype=bool)
    mask[0, :] = _random_profile(size)
    mask[-1, :] = _random_profile(size)
    mask[:, 0] = _random_profile(size)
    mask[:, -1] = _random_profile(size)
    if not np.all(mask[1:-1, 1:-1]):
        return None
    if len(_components(mask)) != 1:
        return None
    rot_sigs = {_mask_signature(np.rot90(mask, k)) for k in range(4)}
    if len(rot_sigs) != 4:
        return None
    removed = int((~mask).sum())
    if removed < 4:
        return None
    return mask


def _boxes_separated(
    a: tuple[int, int], b: tuple[int, int], size: int, gap: int
) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + size + gap <= br
        or br + size + gap <= ar
        or ac + size + gap <= bc
        or bc + size + gap <= ac
    )


def _place_boxes(
    height: int, width: int, size: int, count: int, border: int, gap: int
) -> list[tuple[int, int]] | None:
    if height < size + 2 * border or width < size + 2 * border:
        return None
    positions: list[tuple[int, int]] = []
    for _ in range(800):
        if len(positions) == count:
            return positions
        r = int(np.random.randint(border, height - size - border + 1))
        c = int(np.random.randint(border, width - size - border + 1))
        cand = (r, c)
        if all(_boxes_separated(cand, prev, size, gap) for prev in positions):
            positions.append(cand)
    return None


def _compact_dims(size: int, count: int, gap: int) -> tuple[int, int]:
    if count == 2:
        rows, cols = (1, 2) if np.random.rand() < 0.5 else (2, 1)
    else:
        rows, cols = 2, int(np.ceil(count / 2.0))
    height = rows * size + (rows - 1) * gap + 2
    width = cols * size + (cols - 1) * gap + 2
    height += int(np.random.randint(0, 2))
    width += int(np.random.randint(0, 2))
    return int(height), int(width)


def _sample_special_colors(count: int) -> tuple[int, int, int, np.ndarray]:
    choices = np.random.choice(PIECE_COLORS, size=count + 3, replace=False)
    bg = int(choices[0])
    sep = int(choices[1])
    board = int(choices[2])
    pieces = np.array(choices[3:], dtype=int)
    return bg, sep, board, pieces


def _build_input_output() -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = int(np.random.choice(PIECE_COUNT_OPTIONS))
    size = int(np.random.randint(PIECE_SIZE_RANGE[0], PIECE_SIZE_RANGE[1] + 1))
    bg, sep_color, board_color, piece_colors = _sample_special_colors(piece_count)

    masks: list[np.ndarray] = []
    seen: set[tuple[int, int, bytes]] = set()
    for _ in range(piece_count * 40):
        if len(masks) == piece_count:
            break
        mask = _random_piece_mask(size)
        if mask is None:
            continue
        sig = _canonical_signature(mask)
        if sig in seen:
            continue
        seen.add(sig)
        masks.append(mask)
    if len(masks) != piece_count:
        return None

    hole_rots = [int(np.random.randint(4)) for _ in range(piece_count)]
    piece_rots = [int(np.random.randint(4)) for _ in range(piece_count)]
    if all(h == p for h, p in zip(hole_rots, piece_rots, strict=False)):
        idx = int(np.random.randint(piece_count))
        piece_rots[idx] = int((piece_rots[idx] + np.random.randint(1, 4)) % 4)

    board_h, board_w = _compact_dims(size, piece_count, gap=2)
    board_positions = _place_boxes(board_h, board_w, size, piece_count, border=1, gap=1)
    if board_positions is None:
        return None

    piece_h, piece_w = _compact_dims(size, piece_count, gap=3)
    piece_positions = _place_boxes(piece_h, piece_w, size, piece_count, border=1, gap=1)
    if piece_positions is None:
        return None

    board_canvas = np.full((board_h, board_w), board_color, dtype=int)
    for idx, (r, c) in enumerate(board_positions):
        hole_mask = np.rot90(masks[idx], hole_rots[idx])
        region = board_canvas[r : r + size, c : c + size]
        region[hole_mask] = bg

    piece_canvas = np.full((piece_h, piece_w), bg, dtype=int)
    for idx, (r, c) in enumerate(piece_positions):
        rotated = np.rot90(masks[idx], piece_rots[idx])
        piece_canvas[r : r + size, c : c + size][rotated] = int(piece_colors[idx])

    sep_orient = "h" if np.random.rand() < 0.5 else "v"
    board_first = bool(np.random.rand() < 0.5)

    if sep_orient == "v":
        out_h = max(board_h, piece_h)
        out_w = board_w + piece_w + 1
        inp = np.full((out_h, out_w), bg, dtype=int)
        out = np.full((out_h, out_w), bg, dtype=int)
        left_w = board_w if board_first else piece_w
        inp[:, left_w] = sep_color
        out[:, left_w] = sep_color

        if board_first:
            board_top = int(np.random.randint(0, out_h - board_h + 1))
            piece_top = int(np.random.randint(0, out_h - piece_h + 1))
            inp[board_top : board_top + board_h, 0:board_w] = board_canvas
            inp[piece_top : piece_top + piece_h, left_w + 1 :] = piece_canvas
            out[board_top : board_top + board_h, 0:board_w] = board_canvas
            for idx, (r, c) in enumerate(board_positions):
                mask = np.rot90(masks[idx], hole_rots[idx])
                target = out[board_top + r : board_top + r + size, c : c + size]
                target[mask] = int(piece_colors[idx])
        else:
            piece_top = int(np.random.randint(0, out_h - piece_h + 1))
            board_top = int(np.random.randint(0, out_h - board_h + 1))
            inp[piece_top : piece_top + piece_h, 0:left_w] = piece_canvas
            inp[board_top : board_top + board_h, left_w + 1 :] = board_canvas
            out[board_top : board_top + board_h, left_w + 1 :] = board_canvas
            for idx, (r, c) in enumerate(board_positions):
                mask = np.rot90(masks[idx], hole_rots[idx])
                target = out[board_top + r : board_top + r + size, left_w + 1 + c : left_w + 1 + c + size]
                target[mask] = int(piece_colors[idx])
    else:
        out_h = board_h + piece_h + 1
        out_w = max(board_w, piece_w)
        inp = np.full((out_h, out_w), bg, dtype=int)
        out = np.full((out_h, out_w), bg, dtype=int)
        top_h = board_h if board_first else piece_h
        inp[top_h, :] = sep_color
        out[top_h, :] = sep_color

        if board_first:
            board_left = int(np.random.randint(0, out_w - board_w + 1))
            piece_left = int(np.random.randint(0, out_w - piece_w + 1))
            inp[0:board_h, board_left : board_left + board_w] = board_canvas
            inp[top_h + 1 :, piece_left : piece_left + piece_w] = piece_canvas
            out[0:board_h, board_left : board_left + board_w] = board_canvas
            for idx, (r, c) in enumerate(board_positions):
                mask = np.rot90(masks[idx], hole_rots[idx])
                target = out[r : r + size, board_left + c : board_left + c + size]
                target[mask] = int(piece_colors[idx])
        else:
            piece_left = int(np.random.randint(0, out_w - piece_w + 1))
            board_left = int(np.random.randint(0, out_w - board_w + 1))
            inp[0:piece_h, piece_left : piece_left + piece_w] = piece_canvas
            inp[top_h + 1 :, board_left : board_left + board_w] = board_canvas
            out[top_h + 1 :, board_left : board_left + board_w] = board_canvas
            for idx, (r, c) in enumerate(board_positions):
                mask = np.rot90(masks[idx], hole_rots[idx])
                target = out[top_h + 1 + r : top_h + 1 + r + size, board_left + c : board_left + c + size]
                target[mask] = int(piece_colors[idx])

    if np.array_equal(inp, out):
        return None
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_input_output()
    if built is None:
        return None
    inp, expected = built
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
