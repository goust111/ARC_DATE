import random
from collections import defaultdict

import numpy as np

BLACK = 0
GREEN = 3
GREY = 5
PALETTE = [1, 2, 4, 6, 7, 8, 9]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise AssertionError("segment must be axis-aligned")


def _panel_signature(panel: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = np.where(panel == BLACK, 0, np.where(panel == GREY, 1, 2)).astype(np.uint8)
    return tuple(map(int, norm.shape)), norm.tobytes()


def _collect_panels(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    panels = [_bbox(comp) for comp in _components(grid != GREY)]
    panels.sort(key=lambda box: (box[1], box[0], box[3], box[2]))
    return panels


def _is_frame_patch(patch: np.ndarray, color: int) -> bool:
    h, w = patch.shape
    if h < 5 or w < 5:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.all((patch == color) == border))


def _find_room(panel: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    for color in sorted(int(v) for v in np.unique(panel) if int(v) not in (BLACK, GREY, GREEN)):
        for comp in _components(panel == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = panel[r0 : r1 + 1, c0 : c1 + 1]
            if _is_frame_patch(patch, color):
                return (r0, c0, r1, c1), color
    raise AssertionError("expected a unique monochrome room frame")


def _channel_component(inside: np.ndarray, block_masks: list[np.ndarray]) -> set[tuple[int, int]]:
    best_cells: set[tuple[int, int]] = set()
    best_score = (-1, -1)
    for comp in _components(inside == GREEN):
        touched = 0
        cell_set = set(comp)
        for block_mask in block_masks:
            hit = False
            for r, c in comp:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < inside.shape[0] and 0 <= nc < inside.shape[1] and block_mask[nr, nc]:
                        hit = True
                        break
                if hit:
                    break
            touched += int(hit)
        score = (touched, len(comp))
        if score > best_score:
            best_score = score
            best_cells = cell_set
    return best_cells


def _is_shielded(inside: np.ndarray, r: int, c: int) -> bool:
    up = r > 0 and inside[r - 1, c] == GREY
    down = r + 1 < inside.shape[0] and inside[r + 1, c] == GREY
    left = c > 0 and inside[r, c - 1] == GREY
    right = c + 1 < inside.shape[1] and inside[r, c + 1] == GREY
    return (up and down) or (left and right)


def _solve_panel(panel: np.ndarray) -> np.ndarray:
    out = panel.copy()
    (r0, c0, r1, c1), frame_color = _find_room(panel)
    inside = panel[r0 + 1 : r1, c0 + 1 : c1]
    out_inside = out[r0 + 1 : r1, c0 + 1 : c1]

    block_infos: list[tuple[int, int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(inside) if int(v) not in (BLACK, GREY, GREEN, frame_color)):
        for comp in _components(inside == color):
            mask = np.zeros_like(inside, dtype=bool)
            for r, c in comp:
                mask[r, c] = True
            block_infos.append((len(comp), color, mask))
    assert len(block_infos) == 2, "expected exactly two endpoint color blocks"

    block_infos.sort(key=lambda item: (item[0], item[1]), reverse=True)
    assert block_infos[0][0] != block_infos[1][0], "endpoint block areas must be distinct"
    winner_color = block_infos[0][1]
    block_masks = [info[2] for info in block_infos]
    channel = _channel_component(inside, block_masks)

    for r, c in channel:
        if not _is_shielded(inside, r, c):
            out_inside[r, c] = winner_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn the channel-painting rule from a duplicated training-room pair and apply it to the unmatched test room.

    Input structure:
    - Black rooms sit inside rectangular panel regions separated by grey divider lines.
    - Each panel contains one monochrome rectangular frame enclosing a room.
    - Inside a room, a 1-pixel green channel connects exactly two colored endpoint blocks; some channel cells may be flanked by grey wall pixels.

    Rule:
    - Find the unique pair of panels that have the same room-and-wall structure when foreground colors are ignored; they are the training input/output pair and stay unchanged.
    - Every remaining panel with a green channel is a test room.
    - In each test room, choose the larger endpoint block by visible area, use that block's color as the paint color, and recolor the room's green channel.
    - Any green channel cell with grey pixels on both opposite sides (up/down or left/right) is a shielded segment and remains green.

    Key insight:
    - The duplicated training pair tells you that propagation color comes from the larger endpoint block, while the surviving green shielded strip reveals that adjacent grey walls block recoloring without breaking the channel.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    panels = _collect_panels(g)
    grouped: dict[tuple[tuple[int, int], bytes], list[int]] = defaultdict(list)
    for idx, (r0, c0, r1, c1) in enumerate(panels):
        grouped[_panel_signature(g[r0 : r1 + 1, c0 : c1 + 1])].append(idx)

    paired_indices: set[int] = set()
    for members in grouped.values():
        if len(members) == 2:
            paired_indices.update(members)

    for idx, (r0, c0, r1, c1) in enumerate(panels):
        if idx in paired_indices:
            continue
        panel = g[r0 : r1 + 1, c0 : c1 + 1]
        if GREEN not in panel:
            continue
        out[r0 : r1 + 1, c0 : c1 + 1] = _solve_panel(panel)
    return out


def _available_length(
    ih: int,
    iw: int,
    start: tuple[int, int],
    step: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> int:
    r, c = start
    dr, dc = step
    length = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < ih and 0 <= c < iw):
            break
        if (r, c) in occupied:
            break
        length += 1
    return length


def _place_shield(
    room: np.ndarray,
    run: list[tuple[int, int]],
    orient: str,
) -> None:
    for r, c in run:
        if orient == "h":
            room[r - 1, c] = GREY
            room[r + 1, c] = GREY
        else:
            room[r, c - 1] = GREY
            room[r, c + 1] = GREY


def _build_base_room(room_h: int, room_w: int, rng: random.Random, require_shield: bool, branch_prob: float) -> np.ndarray | None:
    ih, iw = room_h - 2, room_w - 2
    if ih < 5 or iw < 7:
        return None

    color_pool = rng.sample(PALETTE, 3)
    frame_color, color_a, color_b = color_pool

    shape_options = [
        (h, w)
        for h in range(1, min(3, ih) + 1)
        for w in range(1, min(3, max(1, iw // 3)) + 1)
        if h * w <= 6
    ]
    pairs = []
    for left_shape in shape_options:
        for right_shape in shape_options:
            if left_shape[0] * left_shape[1] == right_shape[0] * right_shape[1]:
                continue
            if left_shape[1] + right_shape[1] <= iw - 4:
                pairs.append((left_shape, right_shape))
    if not pairs:
        return None

    left_shape, right_shape = rng.choice(pairs)
    if rng.random() < 0.5:
        bigger_left = left_shape[0] * left_shape[1] > right_shape[0] * right_shape[1]
    else:
        left_shape, right_shape = right_shape, left_shape
        bigger_left = left_shape[0] * left_shape[1] > right_shape[0] * right_shape[1]

    left_color = color_a if bigger_left else color_b
    right_color = color_b if bigger_left else color_a
    left_h, left_w = left_shape
    right_h, right_w = right_shape

    left_top = rng.randint(0, ih - left_h)
    right_top = rng.randint(0, ih - right_h)
    anchor_left = (left_top + left_h // 2, left_w)
    anchor_right = (right_top + right_h // 2, iw - right_w - 1)
    if anchor_right[1] - anchor_left[1] < 2:
        return None
    xmid = rng.randint(anchor_left[1] + 1, anchor_right[1] - 1)

    room = np.full((ih, iw), BLACK, dtype=int)
    room[left_top : left_top + left_h, 0:left_w] = left_color
    room[right_top : right_top + right_h, iw - right_w : iw] = right_color

    seg1 = _segment(anchor_left, (anchor_left[0], xmid))
    seg2 = _segment((anchor_left[0], xmid), (anchor_right[0], xmid))
    seg3 = _segment((anchor_right[0], xmid), anchor_right)

    channel: set[tuple[int, int]] = set(seg1) | set(seg2) | set(seg3)
    segments: list[tuple[str, list[tuple[int, int]]]] = [
        ("h", seg1),
        ("v", seg2),
        ("h", seg3),
    ]

    branch_added = False
    if rng.random() < branch_prob:
        occupied = set(channel)
        occupied.update((r, c) for r in range(left_top, left_top + left_h) for c in range(left_w))
        occupied.update((r, c) for r in range(right_top, right_top + right_h) for c in range(iw - right_w, iw))
        branch_choices: list[tuple[str, tuple[int, int], tuple[int, int], int]] = []
        for orient, cells in segments:
            for cell in cells[1:-1]:
                if orient == "h":
                    for step in [(-1, 0), (1, 0)]:
                        avail = _available_length(ih, iw, cell, step, occupied)
                        if avail >= 2:
                            branch_choices.append((orient, cell, step, avail))
                else:
                    for step in [(0, -1), (0, 1)]:
                        avail = _available_length(ih, iw, cell, step, occupied)
                        if avail >= 2:
                            branch_choices.append((orient, cell, step, avail))
        if branch_choices:
            _, base, step, avail = rng.choice(branch_choices)
            branch_len = rng.randint(2, min(4, avail))
            br, bc = base
            branch_cells = []
            for _ in range(branch_len):
                br += step[0]
                bc += step[1]
                branch_cells.append((br, bc))
            channel.update(branch_cells)
            segments.append(("v" if step[0] else "h", [base] + branch_cells))
            branch_added = True

    for r, c in channel:
        room[r, c] = GREEN

    shield_candidates: list[tuple[str, list[tuple[int, int]]]] = []
    for orient, cells in segments:
        if len(cells) < 3:
            continue
        valid: list[tuple[int, int]] = []
        for r, c in cells[1:-1]:
            if orient == "h":
                if 0 < r < ih - 1 and room[r - 1, c] == BLACK and room[r + 1, c] == BLACK:
                    valid.append((r, c))
                else:
                    valid.append(None)  # type: ignore[arg-type]
            else:
                if 0 < c < iw - 1 and room[r, c - 1] == BLACK and room[r, c + 1] == BLACK:
                    valid.append((r, c))
                else:
                    valid.append(None)  # type: ignore[arg-type]

        stretch: list[tuple[int, int]] = []
        for cell in valid + [None]:  # type: ignore[list-item]
            if cell is None:
                if stretch:
                    for length in range(1, min(3, len(stretch)) + 1):
                        for start in range(len(stretch) - length + 1):
                            shield_candidates.append((orient, stretch[start : start + length]))
                    stretch = []
            else:
                stretch.append(cell)

    want_shield = require_shield or rng.random() < 0.5
    if want_shield:
        if not shield_candidates:
            return None
        orient, run = rng.choice(shield_candidates)
        _place_shield(room, run, orient)

    wrapped = np.full((room_h, room_w), BLACK, dtype=int)
    wrapped[0, :] = frame_color
    wrapped[-1, :] = frame_color
    wrapped[:, 0] = frame_color
    wrapped[:, -1] = frame_color
    wrapped[1:-1, 1:-1] = room

    if require_shield and not np.any(wrapped == GREY):
        return None
    if not branch_added and branch_prob >= 0.5 and rng.random() < 0.4:
        return None
    return wrapped


def _build_room(room_h: int, room_w: int, rng: random.Random, require_shield: bool, branch_prob: float) -> np.ndarray | None:
    rotation = rng.randrange(4)
    base_h, base_w = (room_h, room_w) if rotation % 2 == 0 else (room_w, room_h)
    base = _build_base_room(base_h, base_w, rng, require_shield=require_shield, branch_prob=branch_prob)
    if base is None:
        return None
    return np.rot90(base, rotation).copy()


def _random_panel(
    panel_h: int,
    panel_w: int,
    rng: random.Random,
    require_shield: bool,
    branch_prob: float,
) -> np.ndarray | None:
    min_room_h = 7
    min_room_w = 7
    panel = np.full((panel_h, panel_w), BLACK, dtype=int)
    for _ in range(60):
        top = rng.randint(0, max(0, min(2, panel_h - min_room_h)))
        bottom = rng.randint(0, max(0, min(2, panel_h - min_room_h - top)))
        left = rng.randint(0, max(0, min(2, panel_w - min_room_w)))
        right = rng.randint(0, max(0, min(2, panel_w - min_room_w - left)))
        room_h = panel_h - top - bottom
        room_w = panel_w - left - right
        if room_h < min_room_h or room_w < min_room_w:
            continue
        room = _build_room(room_h, room_w, rng, require_shield=require_shield, branch_prob=branch_prob)
        if room is None:
            continue
        panel.fill(BLACK)
        panel[top : top + room_h, left : left + room_w] = room
        return panel.copy()
    return None


def _compose_grid(
    train_input: np.ndarray,
    train_output: np.ndarray,
    test_input: np.ndarray,
    swap_train_order: bool,
) -> tuple[np.ndarray, np.ndarray]:
    train_h, train_w = train_input.shape
    test_h, test_w = test_input.shape
    assert test_h == 2 * train_h + 1

    height = 2 * train_h + 3
    width = train_w + test_w + 3
    inp = np.full((height, width), GREY, dtype=int)
    out = inp.copy()

    top_rows = slice(1, 1 + train_h)
    bottom_rows = slice(train_h + 2, 2 * train_h + 2)
    left_cols = slice(1, 1 + train_w)
    right_cols = slice(train_w + 2, train_w + 2 + test_w)

    if swap_train_order:
        inp[top_rows, left_cols] = train_output
        inp[bottom_rows, left_cols] = train_input
        out[top_rows, left_cols] = train_output
        out[bottom_rows, left_cols] = train_input
    else:
        inp[top_rows, left_cols] = train_input
        inp[bottom_rows, left_cols] = train_output
        out[top_rows, left_cols] = train_input
        out[bottom_rows, left_cols] = train_output

    inp[1 : 1 + test_h, right_cols] = test_input
    out[1 : 1 + test_h, right_cols] = _solve_panel(test_input)
    return inp, out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    train_h = rng.randint(8, 11)
    train_w = rng.randint(8, 11)
    max_test_w = min(14, 27 - train_w)
    if max_test_w < 9:
        return None
    test_w = rng.randint(9, max_test_w)
    test_h = 2 * train_h + 1

    train_panel = _random_panel(
        train_h,
        train_w,
        rng,
        require_shield=True,
        branch_prob=0.5,
    )
    if train_panel is None:
        return None
    train_solved = _solve_panel(train_panel)
    if np.array_equal(train_panel, train_solved):
        return None

    test_panel = _random_panel(
        test_h,
        test_w,
        rng,
        require_shield=rng.random() < 0.7,
        branch_prob=0.5,
    )
    if test_panel is None:
        return None

    if _panel_signature(train_panel) == _panel_signature(test_panel):
        return None

    inp, expected = _compose_grid(
        train_panel,
        train_solved,
        test_panel,
        swap_train_order=rng.random() < 0.5,
    )

    grouped = defaultdict(list)
    for idx, box in enumerate(_collect_panels(inp)):
        r0, c0, r1, c1 = box
        grouped[_panel_signature(inp[r0 : r1 + 1, c0 : c1 + 1])].append(idx)
    counts = sorted(len(v) for v in grouped.values())
    if counts.count(2) != 1 or sum(counts) != 3:
        return None

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        rng = random.Random()
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input after 300 attempts")
