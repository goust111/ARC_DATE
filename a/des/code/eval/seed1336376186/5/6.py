import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6

MARKER_TO_ROT = {
    GREEN: 0,
    YELLOW: 1,
    BLUE: 2,
    MAGENTA: 3,
}

TEMPLATE_COLORS = (7, 8, 9)
MAX_GEN_ATTEMPTS = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _find_separator(grid: np.ndarray) -> int:
    h, w = grid.shape
    cands = [c for c in range(w) if 2 * c + 1 == w and np.all(grid[:, c] == GREY)]
    assert len(cands) == 1, "expected exactly one centered full-height grey separator"
    return int(cands[0])


def _find_template(left: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    comps = _components((left != BG) & (left != GREY))
    candidates: list[tuple[int, int, int, int, int]] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        area = int(comp.shape[0])
        if area == (r1 - r0 + 1) * (c1 - c0 + 1):
            candidates.append((area, r0, c0, r1, c1))
    assert candidates, "no solid colored template rectangle found"
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    area, r0, c0, r1, c1 = candidates[0]
    return (r0, c0, r1, c1), left[r0 : r1 + 1, c0 : c1 + 1].copy()


def _is_border_component(comp: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    local[comp[:, 0] - r0, comp[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(local, border))


def _find_windows(left: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_comps = _components(left == GREY)
    windows: list[tuple[int, int, int, int]] = []
    for comp in grey_comps:
        if _is_border_component(comp):
            windows.append(_bbox(comp))
    windows.sort()
    assert windows, "expected at least one grey filter window"
    return windows


def _rotated_shape(h: int, w: int, k: int) -> tuple[int, int]:
    return (h, w) if k % 2 == 0 else (w, h)


def _destination_bbox(
    sep: int, window_bbox: tuple[int, int, int, int], interior_shape: tuple[int, int], k: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = window_bbox
    h, w = interior_shape
    dh, dw = _rotated_shape(h, w, k)
    src_r0 = r0 + 1
    src_c0 = c0 + 1
    mirrored_c0 = 2 * sep - (src_c0 + w - 1)
    dst_r0 = src_r0 + (h - dh) // 2
    dst_c0 = mirrored_c0 + (w - dw) // 2
    return (dst_r0, dst_c0, dst_r0 + dh - 1, dst_c0 + dw - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each lower-left filter window into the right half after sampling the upper-left
    template through its red mask and rotating by the window's marker color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    sep = _find_separator(g)
    left = g[:, :sep]

    (_, _, _, _), template = _find_template(left)
    th, tw = template.shape
    windows = _find_windows(left)

    out = g.copy()
    out[:, sep + 1 :] = BG
    occupied = np.zeros_like(out, dtype=bool)

    for bbox in windows:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ih, iw = interior.shape
        mask = interior == RED
        assert np.any(mask), "window must contain a red mask"

        marker_coords = np.argwhere(np.isin(interior, list(MARKER_TO_ROT)))
        assert marker_coords.shape[0] == 1, "window must contain exactly one marker"
        mr, mc = map(int, marker_coords[0])
        k = MARKER_TO_ROT[int(interior[mr, mc])]

        coords = np.argwhere(mask)
        assert int(coords[:, 0].max()) < th and int(coords[:, 1].max()) < tw

        piece = np.full((ih, iw), BG, dtype=int)
        piece[mask] = template[:ih, :iw][mask]
        rotated = np.rot90(piece, -k)

        dr0, dc0, dr1, dc1 = _destination_bbox(sep, bbox, (ih, iw), k)
        assert 0 <= dr0 <= dr1 < h
        assert sep + 1 <= dc0 <= dc1 < w

        painted = rotated != BG
        if np.any(painted):
            region = occupied[dr0 : dr1 + 1, dc0 : dc1 + 1]
            assert not np.any(region & painted), "transformed pieces must not overlap"
            out[dr0 : dr1 + 1, dc0 : dc1 + 1][painted] = rotated[painted]
            region |= painted

    return out


def _sample_template(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    color_count = int(rng.integers(2, len(TEMPLATE_COLORS) + 1))
    palette = rng.choice(TEMPLATE_COLORS, size=color_count, replace=False)
    template = rng.choice(palette, size=(h, w))
    if np.unique(template).size < 2:
        rr = int(rng.integers(0, h))
        cc = int(rng.integers(0, w))
        template[rr, cc] = int(palette[0])
        template[(rr + 1) % h, cc] = int(palette[-1])
    return template.astype(int)


def _random_connected_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    total = h * w
    target = int(rng.integers(2, max(3, min(total, 1 + total // 2))))
    start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {start}
    frontier = {start}
    while len(cells) < target and frontier:
        rr, cc = list(frontier)[int(rng.integers(0, len(frontier)))]
        neigh = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                neigh.append((nr, nc))
        if not neigh:
            frontier.discard((rr, cc))
            continue
        chosen = neigh[int(rng.integers(0, len(neigh)))]
        cells.add(chosen)
        frontier.add(chosen)
        if rng.random() < 0.35:
            frontier.discard((rr, cc))

    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    if mask.sum() >= total:
        mask[int(rng.integers(0, h)), int(rng.integers(0, w))] = False
    return mask


def _window_piece(template: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    piece = np.full((h, w), BG, dtype=int)
    piece[mask] = template[:h, :w][mask]
    return piece


def _paint_window(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    mask: np.ndarray,
    marker_color: int,
    marker_pos: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = BG
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[mask] = RED
    mr, mc = marker_pos
    interior[mr, mc] = marker_color


def _sample_window_shape(rng: np.random.Generator, th: int, tw: int) -> tuple[int, int]:
    max_h = min(th, 6)
    max_w = min(tw, 6)
    parity = int(rng.integers(0, 2))
    h_opts = [x for x in range(3, max_h + 1) if x % 2 == parity]
    w_opts = [x for x in range(3, max_w + 1) if x % 2 == parity]
    if not h_opts or not w_opts:
        common = min(th, tw, 5)
        if common < 3:
            common = min(th, tw)
        return common, common
    return int(rng.choice(h_opts)), int(rng.choice(w_opts))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    half_w = int(rng.integers(18, 25))
    height = int(rng.integers(22, 31))
    width = 2 * half_w + 1
    sep = half_w

    template_h = int(rng.integers(6, min(11, height - 10)))
    template_w = int(rng.integers(6, min(11, half_w - 3)))
    template_r0 = int(rng.integers(1, 4))
    template_c0 = int(rng.integers(1, half_w - template_w))
    template_bbox = (
        template_r0,
        template_c0,
        template_r0 + template_h - 1,
        template_c0 + template_w - 1,
    )

    grid = np.full((height, width), BG, dtype=int)
    grid[:, sep] = GREY

    template = _sample_template(rng, template_h, template_w)
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template

    window_count = int(rng.integers(4, 7))
    markers = [GREEN, YELLOW, BLUE, MAGENTA]
    if window_count > 4:
        extras = rng.choice(markers, size=window_count - 4, replace=True).tolist()
        markers.extend(int(x) for x in extras)
    rng.shuffle(markers)

    src_boxes: list[tuple[int, int, int, int]] = []
    dst_boxes: list[tuple[int, int, int, int]] = []

    for marker in markers:
        rot = MARKER_TO_ROT[int(marker)]
        placed = False
        for _ in range(250):
            ih, iw = _sample_window_shape(rng, template_h, template_w)
            mask = _random_connected_mask(rng, ih, iw)
            if mask.sum() < 2 or mask.sum() >= ih * iw:
                continue

            free_cells = np.argwhere(~mask)
            if free_cells.size == 0:
                continue
            marker_idx = int(rng.integers(0, free_cells.shape[0]))
            marker_pos = tuple(map(int, free_cells[marker_idx]))

            piece = _window_piece(template, mask)
            rotated = np.rot90(piece, -rot)
            if rot != 0 and piece.shape == rotated.shape and np.array_equal(piece, rotated):
                continue

            box_h = ih + 2
            box_w = iw + 2
            row_lo = template_bbox[2] + 2
            row_hi = height - box_h - 1
            col_lo = 1
            col_hi = half_w - box_w - 1
            if row_lo > row_hi or col_lo > col_hi:
                return None

            r0 = int(rng.integers(row_lo, row_hi + 1))
            c0 = int(rng.integers(col_lo, col_hi + 1))
            src_bbox = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)

            if _rect_overlap(src_bbox, template_bbox, pad=1):
                continue
            if any(_rect_overlap(src_bbox, other, pad=1) for other in src_boxes):
                continue

            dst_bbox = _destination_bbox(sep, src_bbox, (ih, iw), rot)
            dr0, dc0, dr1, dc1 = dst_bbox
            if not (0 <= dr0 <= dr1 < height and sep + 1 <= dc0 <= dc1 < width):
                continue
            if any(_rect_overlap(dst_bbox, other, pad=1) for other in dst_boxes):
                continue

            _paint_window(grid, src_bbox, mask, int(marker), marker_pos)
            src_boxes.append(src_bbox)
            dst_boxes.append(dst_bbox)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(grid[:, sep + 1 :] != BG):
        return None
    if not np.any(out[:, sep + 1 :] != BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
