from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
CORNERS = ("TL", "TR", "BL", "BR")

_RNG = np.random.default_rng()


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(mask)
    assert pts.size, "expected a non-empty mask"
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _corner_coord(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 1
    if corner == "BL":
        return h - 1, 0
    if corner == "BR":
        return h - 1, w - 1
    raise ValueError(corner)


def _corner_name(pos: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = map(int, pos)
    h, w = map(int, shape)
    if (r, c) == (0, 0):
        return "TL"
    if (r, c) == (0, w - 1):
        return "TR"
    if (r, c) == (h - 1, 0):
        return "BL"
    if (r, c) == (h - 1, w - 1):
        return "BR"
    raise AssertionError("anchor is not on a corner")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in neighbors:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(pts, dtype=int))
    return out


def _rotated_corner(src_corner: str, shape: tuple[int, int], k: int) -> tuple[str, tuple[int, int]]:
    marker = np.zeros(shape, dtype=bool)
    marker[_corner_coord(shape, src_corner)] = True
    rotated = np.rot90(marker, int(k))
    pts = np.argwhere(rotated)
    assert len(pts) == 1
    rr, cc = map(int, pts[0])
    return _corner_name((rr, cc), rotated.shape), tuple(map(int, rotated.shape))


def _rotate_patch_to_corner(patch: np.ndarray, src_corner: str, dst_corner: str) -> np.ndarray:
    for k in range(4):
        corner, _shape = _rotated_corner(src_corner, patch.shape, k)
        if corner == dst_corner:
            return np.rot90(patch, k).copy()
    raise AssertionError("no rotation maps the source anchor corner to the target corner")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate anchored fragments into the matching slots of a gray framework.

    Input structure:
    - The background is black and one large gray rectangular framework is divided by full gray grid lines.
    - Each equal-size slot contains exactly one colored single-pixel anchor on one of its four corners.
    - Outside the framework are same-size colored fragment patches on black background.
    - Each fragment has a corner anchor whose color matches exactly one slot anchor.

    Rule:
    - Pair every external fragment with the slot that has the same anchor color.
    - Rotate the fragment by 0, 90, 180, or 270 degrees so its anchor corner matches the slot anchor corner.
    - Clear the scattered external fragments and paste each rotated patch into its slot interior.
    - Preserve all gray framework boundary and divider lines.

    Key insight:
    - Anchor color selects the destination slot, while anchor corner position independently selects the rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    gray_mask = g == GRAY
    fr0, fc0, fr1, fc1 = _bbox_from_mask(gray_mask)
    frame_gray = gray_mask[fr0 : fr1 + 1, fc0 : fc1 + 1]

    h_lines = [
        int(fr0 + r)
        for r in range(frame_gray.shape[0])
        if bool(np.all(frame_gray[r, :]))
    ]
    v_lines = [
        int(fc0 + c)
        for c in range(frame_gray.shape[1])
        if bool(np.all(frame_gray[:, c]))
    ]
    assert len(h_lines) >= 3 and len(v_lines) >= 3

    slot_heights = [h_lines[i + 1] - h_lines[i] - 1 for i in range(len(h_lines) - 1)]
    slot_widths = [v_lines[i + 1] - v_lines[i] - 1 for i in range(len(v_lines) - 1)]
    assert min(slot_heights) > 0 and min(slot_widths) > 0
    assert len(set(slot_heights)) == 1 and len(set(slot_widths)) == 1
    slot_shape = (int(slot_heights[0]), int(slot_widths[0]))

    slots: dict[int, dict] = {}
    for ri in range(len(h_lines) - 1):
        for ci in range(len(v_lines) - 1):
            sr0, sr1 = h_lines[ri] + 1, h_lines[ri + 1] - 1
            sc0, sc1 = v_lines[ci] + 1, v_lines[ci + 1] - 1
            sub = g[sr0 : sr1 + 1, sc0 : sc1 + 1]
            anchor_pts = np.argwhere((sub != BLACK) & (sub != GRAY))
            assert len(anchor_pts) == 1
            ar, ac = map(int, anchor_pts[0])
            corner = _corner_name((ar, ac), sub.shape)
            color = int(sub[ar, ac])
            assert color not in slots
            slots[color] = {
                "bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                "corner": corner,
            }

    inside_frame = np.zeros((h, w), dtype=bool)
    inside_frame[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    piece_mask = (g != BLACK) & (g != GRAY) & (~inside_frame)

    pieces: dict[int, dict] = {}
    for coords in _components(piece_mask):
        pr0 = int(coords[:, 0].min())
        pc0 = int(coords[:, 1].min())
        pr1 = int(coords[:, 0].max())
        pc1 = int(coords[:, 1].max())
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

        matching_corners = []
        for name in CORNERS:
            rr, cc = _corner_coord(patch.shape, name)
            val = int(patch[rr, cc])
            if val in slots:
                matching_corners.append((name, val))
        assert len(matching_corners) == 1
        src_corner, color = matching_corners[0]
        assert color not in pieces
        pieces[int(color)] = {"patch": patch, "corner": src_corner}

    assert set(pieces) == set(slots)

    out = np.full_like(g, BLACK)
    out[gray_mask] = GRAY
    for color, slot in slots.items():
        piece = pieces[int(color)]
        rotated = _rotate_patch_to_corner(piece["patch"], piece["corner"], slot["corner"])
        assert rotated.shape == slot_shape
        sr0, sc0, sr1, sc1 = slot["bbox"]
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = rotated

    out[gray_mask] = GRAY
    return out


def _choice(seq):
    return seq[int(_RNG.integers(0, len(seq)))]


def _paint_line_piece(size: int, anchor_color: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    patch[0, 0] = int(anchor_color)

    row = int(_RNG.integers(1, size - 1))
    col = int(_RNG.integers(1, size - 1))
    content_colors = [c for c in COLORS if c != int(anchor_color)]

    def paint(r: int, c: int) -> None:
        if (r, c) == (0, 0):
            patch[r, c] = int(anchor_color)
            return
        if (r, c) in ((0, size - 1), (size - 1, 0), (size - 1, size - 1)):
            return
        patch[r, c] = int(_choice(content_colors))

    for r in range(0, row + 1):
        paint(r, 0)
    for c in range(0, size):
        paint(row, c)
    for r in range(row, size):
        paint(r, col)

    growth_steps = int(_RNG.integers(0, size + 1))
    for _ in range(growth_steps):
        filled = np.argwhere(patch != BLACK)
        if len(filled) == 0:
            break
        r, c = map(int, filled[int(_RNG.integers(0, len(filled)))])
        nbrs = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < size and 0 <= cc < size):
                continue
            if (rr, cc) in ((0, size - 1), (size - 1, 0), (size - 1, size - 1)):
                continue
            if patch[rr, cc] == BLACK:
                nbrs.append((rr, cc))
        if nbrs:
            rr, cc = nbrs[int(_RNG.integers(0, len(nbrs)))]
            patch[rr, cc] = int(_choice(content_colors))
    return patch


def _make_piece(size: int, anchor_color: int, source_corner: str) -> np.ndarray:
    canonical = _paint_line_piece(size, anchor_color)
    k_by_corner = {"TL": 0, "BL": 1, "BR": 2, "TR": 3}
    return np.rot90(canonical, k_by_corner[source_corner]).copy()


def _draw_framework(grid: np.ndarray, r0: int, c0: int, slot_rows: int, slot_cols: int, size: int) -> None:
    frame_h = slot_rows * size + slot_rows + 1
    frame_w = slot_cols * size + slot_cols + 1
    for i in range(slot_rows + 1):
        rr = r0 + i * (size + 1)
        grid[rr, c0 : c0 + frame_w] = GRAY
    for j in range(slot_cols + 1):
        cc = c0 + j * (size + 1)
        grid[r0 : r0 + frame_h, cc] = GRAY


def _mark_rect(mask: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int) -> None:
    h, w = mask.shape
    rr0 = max(0, int(r0) - int(pad))
    cc0 = max(0, int(c0) - int(pad))
    rr1 = min(h, int(r0) + int(height) + int(pad))
    cc1 = min(w, int(c0) + int(width) + int(pad))
    mask[rr0:rr1, cc0:cc1] = True


def _place_patches(grid: np.ndarray, patches: list[np.ndarray], frame_bbox: tuple[int, int, int, int]) -> bool:
    h, w = grid.shape
    fr0, fc0, fr1, fc1 = frame_bbox
    occupied = np.zeros((h, w), dtype=bool)
    _mark_rect(occupied, fr0, fc0, fr1 - fr0 + 1, fc1 - fc0 + 1, 1)

    order = list(range(len(patches)))
    _RNG.shuffle(order)
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        placed = False
        for _ in range(800):
            r0 = int(_RNG.integers(0, h - ph + 1))
            c0 = int(_RNG.integers(0, w - pw + 1))
            if bool(occupied[r0 : r0 + ph, c0 : c0 + pw].any()):
                continue
            if not (r0 + ph - 1 < fr0 or r0 > fr1 or c0 + pw - 1 < fc0 or c0 > fc1):
                continue
            placements[idx] = (r0, c0)
            _mark_rect(occupied, r0, c0, ph, pw, 1)
            placed = True
            break
        if not placed:
            return False

    for idx, (r0, c0) in placements.items():
        patch = patches[idx]
        ph, pw = map(int, patch.shape)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return True


def _gen_attempt() -> np.ndarray | None:
    configs = (
        (2, 2, (3, 4, 5, 6)),
        (2, 3, (3, 4, 5)),
        (3, 2, (3, 4, 5)),
        (2, 4, (3, 4)),
        (4, 2, (3, 4)),
    )
    slot_rows, slot_cols, size_choices = _choice(configs)
    size = int(_choice(size_choices))
    frame_h = int(slot_rows * size + slot_rows + 1)
    frame_w = int(slot_cols * size + slot_cols + 1)
    if frame_h + 4 > 30 or frame_w + 4 > 30:
        return None

    height = int(_RNG.integers(frame_h + 4, 31))
    width = int(_RNG.integers(frame_w + 4, 31))
    fr0 = int(_RNG.integers(1, height - frame_h))
    fc0 = int(_RNG.integers(1, width - frame_w))
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros_like(grid)
    _draw_framework(grid, fr0, fc0, int(slot_rows), int(slot_cols), size)
    _draw_framework(expected, fr0, fc0, int(slot_rows), int(slot_cols), size)

    count = int(slot_rows * slot_cols)
    colors = [int(c) for c in _RNG.choice(COLORS, size=count, replace=False)]
    target_corners = [_choice(CORNERS) for _ in range(count)]
    source_corners = [_choice(CORNERS) for _ in range(count)]

    patches: list[np.ndarray] = []
    item = 0
    for sr in range(int(slot_rows)):
        for sc in range(int(slot_cols)):
            color = int(colors[item])
            slot_r0 = fr0 + sr * (size + 1) + 1
            slot_c0 = fc0 + sc * (size + 1) + 1
            target_corner = str(target_corners[item])
            ar, ac = _corner_coord((size, size), target_corner)
            grid[slot_r0 + ar, slot_c0 + ac] = color

            source_corner = str(source_corners[item])
            patch = _make_piece(size, color, source_corner)
            patches.append(patch)
            rotated = _rotate_patch_to_corner(patch, source_corner, target_corner)
            expected[slot_r0 : slot_r0 + size, slot_c0 : slot_c0 + size] = rotated
            item += 1

    if not _place_patches(grid, patches, (fr0, fc0, fr1, fc1)):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid anchored-framework input")
