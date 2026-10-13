import random
from collections import defaultdict, deque

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _bbox_coords(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = np.min(points, axis=0)
    r1, c1 = np.max(points, axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    neighbors = _NEIGHBORS4 if connectivity == 4 else ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))

    cur = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != 0:
                continue
            cur += 1
            q = deque([(int(r), int(c))])
            labels[int(r), int(c)] = int(cur)
            while q:
                rr, cc = q.popleft()
                for dr, dc in neighbors:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and labels[nr, nc] == 0:
                        labels[nr, nc] = int(cur)
                        q.append((nr, nc))

    return labels, int(cur)


def components_mask(mask: np.ndarray, connectivity: int = 4):
    labels, n = _label_mask(mask, connectivity=connectivity)
    for cid in range(1, int(n) + 1):
        pts = np.argwhere(labels == int(cid))
        if pts.size:
            yield pts


def _frame_candidates(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    h, w = g.shape
    comps: list[tuple[int, int, int, int, int]] = []

    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    for color in colors:
        labels, n = _label_mask(g == int(color), connectivity=4)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(labels == int(cid))
            if pts.size == 0:
                continue
            r0, c0, r1, c1 = _bbox_coords(pts)
            ch = int(r1 - r0 + 1)
            cw = int(c1 - c0 + 1)
            if ch < 3 or cw < 3:
                continue

            part = (labels[r0 : r1 + 1, c0 : c1 + 1] == int(cid))
            if part.shape[0] * part.shape[1] == int(part.sum()):
                # Filled block, not a hollow frame.
                continue

            expected = np.zeros_like(part, dtype=bool)
            expected[0, :] = True
            expected[-1, :] = True
            expected[:, 0] = True
            expected[:, -1] = True
            if not np.array_equal(part, expected):
                continue

            comps.append((int(color), int(r0), int(c0), int(r1), int(c1)))

    return comps


def _detect_frame(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    bg = int(_get_bg(g))
    candidates = _frame_candidates(g, bg)
    if not candidates:
        raise AssertionError("No hollow rectangular frame found")

    # Prefer the largest candidate to avoid accidental tiny inner frames.
    color, r0, c0, r1, c1 = max(candidates, key=lambda t: (int(t[3] - t[1] + 1) * int(t[4] - t[2] + 1)))
    return color, (r0, c0, r1, c1)


def _collect_anchor_cells(g: np.ndarray, bg: int, frame_bbox: tuple[int, int, int, int]) -> dict[int, list[tuple[int, int]]]:
    r0, c0, r1, c1 = frame_bbox
    anchors: dict[int, set[tuple[int, int]]] = defaultdict(set)

    # Top and bottom inner wall.
    for c in range(c0 + 1, c1):
        top = int(g[r0 + 1, c])
        if top != int(bg):
            anchors[top].add((int(r0 + 1), int(c)))

        bot = int(g[r1 - 1, c])
        if bot != int(bg):
            anchors[bot].add((int(r1 - 1), int(c)))

    # Left and right inner wall.
    for r in range(r0 + 1, r1):
        left = int(g[r, c0 + 1])
        if left != int(bg):
            anchors[left].add((int(r), int(c0 + 1)))

        right = int(g[r, c1 - 1])
        if right != int(bg):
            anchors[right].add((int(r), int(c1 - 1)))

    return {int(col): sorted(pts) for col, pts in anchors.items()}


def _boundary_colors(g: np.ndarray, coords: np.ndarray, comp_set: set[tuple[int, int]]) -> set[int]:
    h, w = g.shape
    colors: set[int] = set()
    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (int(nr), int(nc)) not in comp_set:
                colors.add(int(g[nr, nc]))
    return colors


def _crop_piece(g: np.ndarray, coords: np.ndarray, bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    patch = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for r, c in coords:
        rr = int(r) - int(r0)
        cc = int(c) - int(c0)
        patch[rr, cc] = int(g[int(r), int(c)])
    return patch


def _interior_bounds(frame_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = frame_bbox
    return int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)


def _is_inside_bounds(pts: np.ndarray, ir0: int, ic0: int, ir1: int, ic1: int) -> bool:
    if pts.size == 0:
        return False
    return bool(int(pts[:, 0].min()) >= ir0 and int(pts[:, 0].max()) <= ir1 and int(pts[:, 1].min()) >= ic0 and int(pts[:, 1].max()) <= ic1)


def _place_piece_inside(
    piece: np.ndarray,
    interface_color: int,
    wall_anchors: set[tuple[int, int]],
    ir0: int,
    ic0: int,
    ir1: int,
    ic1: int,
    bg: int,
    occupied: np.ndarray,
) -> tuple[np.ndarray, int, int] | None:
    """Try all 4 rotations and all translations in the frame interior.

    A valid placement requires:
    - the piece lies fully in the interior region,
    - it does not overlap with already painted output cells,
    - every anchor-colored boundary pixel of the piece becomes edge-adjacent to a
      same-color anchor pixel on the frame wall, and the set of touched anchors is
      complete.
    """
    target_anchor_count = len(wall_anchors)
    if target_anchor_count == 0:
        return None

    for k in range(4):
        cand = np.rot90(piece, int(k))
        anchors_local = np.argwhere(cand == int(interface_color))
        if anchors_local.shape[0] != target_anchor_count:
            continue

        fill_local = np.argwhere(cand != int(bg))
        ph, pw = cand.shape

        # Ensure piece can possibly fit in interior.
        if ph > ir1 - ir0 + 1 or pw > ic1 - ic0 + 1:
            continue

        for tr in range(ir0, ir1 - ph + 2):
            for tc in range(ic0, ic1 - pw + 2):
                final_cells = [(int(r + tr), int(c + tc)) for r, c in fill_local]
                ok = True
                for fr, fc in final_cells:
                    if occupied[fr, fc]:
                        ok = False
                        break
                if not ok:
                    continue

                touched: set[tuple[int, int]] = set()
                all_matched = True
                for r, c in anchors_local:
                    ar, ac = int(r) + tr, int(c) + tc
                    matched_here = False
                    for dr, dc in _NEIGHBORS4:
                        nr, nc = ar + dr, ac + dc
                        if (int(nr), int(nc)) in wall_anchors:
                            touched.add((int(nr), int(nc)))
                            matched_here = True
                    if not matched_here:
                        all_matched = False
                        break

                if not all_matched:
                    continue
                if touched != wall_anchors:
                    continue

                return cand, tr, tc

    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Insert irregular colored fragments into a hollow frame by anchor-matching its wall colors.

    Input structure:
    - Background is black. A large one-pixel-thick rectangular frame is drawn in a grey border color.
    - Inside the frame, the inner wall carries several colored anchor pixels on its perimeter, each anchor color appears as a small group.
    - Outside the frame are several disconnected irregular fragments of multiple colors.

    Rule:
    - Detect the frame, then detect anchor groups on its inner wall.
    - For each outside fragment, find its unique anchor-color boundary color that matches one anchor group.
    - Try 0°/90°/180°/270° rotations and translations in the interior so that all anchor-color boundary pixels of the piece become 4-neigh adjacent to same-color anchors and exactly cover the whole anchor set.
    - Build output as the frame plus all fragments placed inside it, and clear everything else to background.

    Key insight:
    - The anchor group color picks the fragment, and the geometric arrangement of anchor pixels determines a unique rotation and interior placement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_get_bg(g))

    frame_color, frame_bbox = _detect_frame(g)
    frame_mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    frame_mask[r0, c0 : c1 + 1] = True
    frame_mask[r1, c0 : c1 + 1] = True
    frame_mask[r0 : r1 + 1, c0] = True
    frame_mask[r0 : r1 + 1, c1] = True

    anchors_by_color = _collect_anchor_cells(g, bg, frame_bbox)
    if not anchors_by_color:
        raise AssertionError("No anchor pixels found on the frame inner wall")

    # Keep only likely valid anchor groups (2..3 pixels as specified).
    anchors_by_color = {
        int(col): [tuple(p) for p in pts]
        for col, pts in anchors_by_color.items()
        if 1 < len(pts) < 4
    }

    if not anchors_by_color:
        raise AssertionError("No valid anchor groups found")

    # Output starts from background, then add the fixed frame plus its inner-wall anchors.
    out = np.full((h, w), int(bg), dtype=int)
    out[frame_mask] = int(frame_color)
    for col, pts in anchors_by_color.items():
        for r, c in pts:
            out[int(r), int(c)] = int(col)

    anchor_set_by_color = {int(col): set(map(tuple, pts)) for col, pts in anchors_by_color.items()}

    # Locate all potential fragments outside the frame.
    fragment_mask = (g != int(bg)) & (~frame_mask)
    for col, pts in anchors_by_color.items():
        for r, c in pts:
            fragment_mask[int(r), int(c)] = False

    ir0, ic0, ir1, ic1 = _interior_bounds(frame_bbox)
    if ir0 > ir1 or ic0 > ic1:
        raise AssertionError("Frame interior is too small")

    occupied = np.zeros((h, w), dtype=bool)
    occupied[frame_mask] = True
    for col, pts in anchors_by_color.items():
        for r, c in pts:
            occupied[int(r), int(c)] = True

    placed = 0
    for comp in components_mask(fragment_mask, connectivity=4):
        if comp.size == 0:
            continue

        comp = np.asarray(comp, dtype=int)
        comp_set = {(int(r), int(c)) for r, c in comp.tolist()}
        colors = {int(g[r, c]) for r, c in comp}
        matches = [c for c in colors if c in anchors_by_color]

        if len(matches) != 1:
            continue

        iface_color = int(matches[0])
        if iface_color not in anchor_set_by_color:
            continue

        piece = _crop_piece(g, comp, bg)
        placement = _place_piece_inside(
            piece,
            iface_color,
            anchor_set_by_color[iface_color],
            ir0,
            ic0,
            ir1,
            ic1,
            bg,
            occupied,
        )
        if placement is None:
            continue

        placed_piece, tr, tc = placement
        filled = np.argwhere(placed_piece != int(bg))
        for r, c in filled:
            rr, cc = int(r) + tr, int(c) + tc
            out[rr, cc] = int(placed_piece[r, c])
            occupied[rr, cc] = True

        for p in anchor_set_by_color[iface_color]:
            occupied[p[0], p[1]] = True

        placed += 1

    if placed == 0:
        raise AssertionError("No pieces were placed into the frame")

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

_BG = 0
_FRAME = 1
_ANCHOR_POOL = (2, 3, 4, 5)
_ACCENT_POOL = (6, 7, 8, 9)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    grid[r0, c0 : c1 + 1] = _FRAME
    grid[r1, c0 : c1 + 1] = _FRAME
    grid[r0 : r1 + 1, c0] = _FRAME
    grid[r0 : r1 + 1, c1] = _FRAME


def _anchor_positions_for_side(
    rng: random.Random,
    side: str,
    frame_bbox: tuple[int, int, int, int],
    count: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        max_start = c1 - count
        if max_start < c0 + 1:
            return []
        max_start = min(max_start, c1 - 1)
        if max_start < c0 + 1:
            return []
        c = rng.randint(c0 + 1, max_start)
        return [(r0 + 1, int(c + i)) for i in range(count)]
    if side == "bottom":
        max_start = c1 - count
        if max_start < c0 + 1:
            return []
        max_start = min(max_start, c1 - 1)
        if max_start < c0 + 1:
            return []
        c = rng.randint(c0 + 1, max_start)
        return [(r1 - 1, int(c + i)) for i in range(count)]
    if side == "left":
        max_start = r1 - count
        if max_start < r0 + 1:
            return []
        max_start = min(max_start, r1 - 1)
        if max_start < r0 + 1:
            return []
        r = rng.randint(r0 + 1, max_start)
        return [(int(r + i), c0 + 1) for i in range(count)]
    if side == "right":
        max_start = r1 - count
        if max_start < r0 + 1:
            return []
        max_start = min(max_start, r1 - 1)
        if max_start < r0 + 1:
            return []
        r = rng.randint(r0 + 1, max_start)
        return [(int(r + i), c1 - 1) for i in range(count)]
    raise ValueError(side)


def _random_fragment_template(
    rng: random.Random,
    anchor_len: int,
    anchor_color: int,
    accent_colors: tuple[int, ...],
) -> np.ndarray:
    h = rng.randint(3, 6)
    w = rng.randint(max(3, anchor_len + 1), 7)
    w = max(anchor_len + 2, w)

    top_start = rng.randint(1, w - anchor_len - 1)
    anchor_cols = list(range(top_start, top_start + anchor_len))

    template = np.full((h, w), _BG, dtype=int)
    template[0, anchor_cols] = anchor_color

    cells: set[tuple[int, int]] = set((0, c) for c in anchor_cols)

    # Grow with 4-neighbor expansion for an irregular, connected body.
    target = rng.randint(anchor_len + 3, min(anchor_len + 8, h * w))
    frontier: list[tuple[int, int]] = []

    def _add_frontier(r: int, c: int) -> None:
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.append((nr, nc))

    for r, c in cells:
        _add_frontier(int(r), int(c))

    while len(cells) < target and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in cells:
            continue
        if len(cells) < target:
            cells.add((int(r), int(c)))
            _add_frontier(int(r), int(c))

    # Keep at least two extra colors if enough cells exist.
    for r, c in cells:
        if r == 0 and c in anchor_cols:
            continue
        choices = [c for c in accent_colors if c != anchor_color]
        if not choices:
            template[r, c] = _ACCENT_POOL[0]
        else:
            template[r, c] = int(rng.choice(choices))

    # Ensure anchor cells remain the only specified anchor color.
    for c in anchor_cols:
        template[0, int(c)] = anchor_color

    return template


def _count_color_boundary_cells(piece: np.ndarray, color: int) -> int:
    h, w = piece.shape
    cell_mask = piece == int(color)
    if not bool(np.any(cell_mask)):
        return 0

    count = 0
    cells = np.argwhere(cell_mask)
    for r, c in cells:
        r = int(r)
        c = int(c)
        border = False
        for dr, dc in _NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                border = True
                break
            if not cell_mask[int(nr), int(nc)]:
                border = True
                break
        if border:
            count += 1
    return count


def _place_fragment_outside(
    rng: random.Random,
    grid: np.ndarray,
    used: np.ndarray,
    piece: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    h, w = grid.shape
    ph, pw = piece.shape
    piece_mask = piece != _BG
    piece_cells = np.argwhere(piece_mask)
    touch_offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]

    choices: list[tuple[int, int]] = []
    for r in range(0, h - ph + 1):
        for c in range(0, w - pw + 1):
            rr = slice(r, r + ph)
            cc = slice(c, c + pw)
            if np.any(forbidden[rr, cc] & piece_mask):
                continue
            blocked = False
            for dr, dc in piece_cells:
                rr0 = int(dr) + r
                cc0 = int(dc) + c
                for off_r, off_c in touch_offsets:
                    rr = rr0 + off_r
                    cc = cc0 + off_c
                    if 0 <= rr < h and 0 <= cc < w and used[rr, cc]:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue

            choices.append((int(r), int(c)))

    if not choices:
        return False

    tr, tc = choices[rng.randrange(len(choices))]
    cells = np.argwhere(piece != _BG)
    for r, c in cells:
        rr, cc = int(r) + tr, int(c) + tc
        grid[rr, cc] = int(piece[r, c])
        used[rr, cc] = True
    return True


def _build_input_once(rng: random.Random) -> np.ndarray | None:
    H = int(rng.randint(16, 30))
    W = int(rng.randint(16, 30))

    grid = np.full((H, W), _BG, dtype=int)

    max_frame_h = min(H - 4, 24)
    max_frame_w = min(W - 4, 24)
    if max_frame_h < 9 or max_frame_w < 9:
        return None

    frame_h = int(rng.randint(9, max_frame_h))
    frame_w = int(rng.randint(9, max_frame_w))

    r0 = int(rng.randint(2, H - frame_h - 1))
    c0 = int(rng.randint(2, W - frame_w - 1))
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1

    if r1 >= H or c1 >= W:
        return None

    if frame_h < 9 or frame_w < 9:
        return None

    _paint_frame(grid, r0, c0, r1, c1)

    anchors_by_color: dict[int, list[tuple[int, int]]] = {}
    used_anchor_cells: set[tuple[int, int]] = set()
    used_anchor_colors = list(_ANCHOR_POOL)
    rng.shuffle(used_anchor_colors)

    piece_count = int(rng.choice([2, 2, 3, 3, 4]))
    piece_colors = used_anchor_colors[:piece_count]
    sides = ["top", "bottom", "left", "right"]

    for color in piece_colors:
        side = rng.choice(sides)
        count = int(rng.choice([2, 3]))
        pos = _anchor_positions_for_side(rng, side, (r0, c0, r1, c1), count)
        if len(pos) != count:
            return None
        if any((int(r), int(c)) in used_anchor_cells for r, c in pos):
            return None
        anchors_by_color[int(color)] = pos
        used_anchor_cells.update((int(r), int(c)) for r, c in pos)
        for r, c in pos:
            grid[r, c] = int(color)

    # Build forbidden zone: frame + 1-cell margin.
    forbidden = np.zeros((H, W), dtype=bool)
    fr0 = max(0, r0 - 1)
    fr1 = min(H - 1, r1 + 1)
    fc0 = max(0, c0 - 1)
    fc1 = min(W - 1, c1 + 1)
    forbidden[fr0 : fr1 + 1, fc0 : fc1 + 1] = True

    # Also block the frame itself.
    used = np.zeros((H, W), dtype=bool)
    used[0:H, 0:W] = False
    used[r0 : r1 + 1, c0 : c1 + 1] = True

    for color, pos in anchors_by_color.items():
        base = None
        for _ in range(80):
            cand = _random_fragment_template(rng, len(pos), color, _ACCENT_POOL)
            cand = np.rot90(cand, rng.randrange(4))
            if _count_color_boundary_cells(cand, int(color)) == len(pos):
                base = cand
                break
        if base is None:
            return None

        if not _place_fragment_outside(rng, grid, used, base, forbidden):
            return None

    # The task is not required to keep outer spacing strict, only disjointness.
    # Ensure at least one non-frame pixel exists outside for all pieces.
    if not np.any((grid != _BG) & (~forbidden)):
        return None

    # Validate main can parse this input and keeps all assumptions.
    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != (H, W):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(300):
        inp = _build_input_once(rng)
        if inp is None:
            continue

        try:
            # Keep only variants where the rule executes with all anchors matched.
            _ = main(inp)
            return inp
        except Exception:
            continue

    raise RuntimeError("Failed to sample a valid puzzle instance")
