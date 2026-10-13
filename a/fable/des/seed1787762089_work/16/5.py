import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_SIZE = 7
INNER_SIZE = FRAME_SIZE - 2
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIRS8 = DIRS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))
TRANSFORMS = ("mirror_lr", "rot180", "swap2")


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: list[tuple[int, int]] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return int(arr[:, 0].min()), int(arr[:, 1].min()), int(arr[:, 0].max()), int(arr[:, 1].max())


def _coords_to_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, r0, c0


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    dirs = DIRS8 if connectivity == 8 else DIRS4
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in dirs:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_single_component(coords: list[tuple[int, int]], connectivity: int = 8) -> bool:
    if not coords:
        return False
    pts = set(coords)
    dirs = DIRS8 if connectivity == 8 else DIRS4
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _apply_transform(patch: np.ndarray, mode: str, bg: int) -> np.ndarray:
    if mode == "mirror_lr":
        return np.fliplr(patch)
    if mode == "rot180":
        return np.rot90(patch, 2)
    if mode == "swap2":
        colors = [int(v) for v in np.unique(patch) if int(v) != bg]
        if len(colors) != 2:
            return patch.copy()
        a, b = colors
        out = patch.copy()
        out[patch == a] = b
        out[patch == b] = a
        return out
    raise ValueError(f"unknown transform: {mode}")


def _matching_transforms(src: np.ndarray, dst: np.ndarray, bg: int) -> list[str]:
    matches: list[str] = []
    for mode in TRANSFORMS:
        if mode == "swap2" and len([int(v) for v in np.unique(src) if int(v) != bg]) != 2:
            continue
        if np.array_equal(_apply_transform(src, mode, bg), dst):
            matches.append(mode)
    return matches


def _find_training_pair(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int], int]:
    h, w = grid.shape
    border_mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    candidates: list[tuple[int, int, int]] = []
    for r in range(h - FRAME_SIZE + 1):
        for c in range(w - FRAME_SIZE + 1):
            patch = grid[r : r + FRAME_SIZE, c : c + FRAME_SIZE]
            border = patch[border_mask]
            unique = np.unique(border)
            if len(unique) != 1:
                continue
            color = int(unique[0])
            if color == bg:
                continue
            if not np.any(patch[1:-1, 1:-1] != bg):
                continue
            candidates.append((r, c, color))

    best = None
    for i in range(len(candidates)):
        r1, c1, color1 = candidates[i]
        for j in range(i + 1, len(candidates)):
            r2, c2, color2 = candidates[j]
            if color1 != color2:
                continue
            top = (r1, c1) if (r1, c1) <= (r2, c2) else (r2, c2)
            bot = (r2, c2) if top == (r1, c1) else (r1, c1)
            if abs(top[0] - bot[0]) < FRAME_SIZE:
                continue
            score = top[0] + top[1] + bot[0] + bot[1] + 5 * abs(top[1] - bot[1])
            cand = (score, top, bot, color1)
            if best is None or cand[0] < best[0]:
                best = cand
    assert best is not None, "could not find training frames"
    _score, top, bot, color = best
    return top, bot, color


def _tail_info(
    grid: np.ndarray,
    coords: list[tuple[int, int]],
    tail_color: int,
    bg: int,
) -> tuple[str, tuple[int, int], np.ndarray, int, int] | None:
    grey_cells = [(r, c) for r, c in coords if int(grid[r, c]) == tail_color]
    valid: list[tuple[str, tuple[int, int], np.ndarray, int, int]] = []
    for tr, tc in grey_cells:
        body = [(r, c) for r, c in coords if (r, c) != (tr, tc)]
        if not body or not _is_single_component(body, connectivity=8):
            continue
        body_set = set(body)
        r0, c0, r1, c1 = _bbox(body)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        side = None
        move = None
        if width % 2 == 1:
            mid_c = c0 + width // 2
            if tr == r0 - 1 and tc == mid_c and (r0, mid_c) in body_set:
                side, move = "top", (1, 0)
            if tr == r1 + 1 and tc == mid_c and (r1, mid_c) in body_set:
                side, move = "bottom", (-1, 0)
        if side is None and height % 2 == 1:
            mid_r = r0 + height // 2
            if tc == c0 - 1 and tr == mid_r and (mid_r, c0) in body_set:
                side, move = "left", (0, 1)
            if tc == c1 + 1 and tr == mid_r and (mid_r, c1) in body_set:
                side, move = "right", (0, -1)
        if side is None:
            continue
        patch, pr, pc = _coords_to_patch(grid, body, bg)
        valid.append((side, move, patch, pr, pc))
    if len(valid) != 1:
        return None
    return valid[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    h, w = g.shape

    (top_r, top_c), (bot_r, bot_c), tail_color = _find_training_pair(g, bg)
    frame_mask = np.zeros((h, w), dtype=bool)
    frame_mask[top_r : top_r + FRAME_SIZE, top_c : top_c + FRAME_SIZE] = True
    frame_mask[bot_r : bot_r + FRAME_SIZE, bot_c : bot_c + FRAME_SIZE] = True

    upper = g[top_r + 1 : top_r + 1 + INNER_SIZE, top_c + 1 : top_c + 1 + INNER_SIZE]
    lower = g[bot_r + 1 : bot_r + 1 + INNER_SIZE, bot_c + 1 : bot_c + 1 + INNER_SIZE]
    matches = _matching_transforms(upper, lower, bg)
    assert len(matches) == 1, f"ambiguous transform: {matches}"
    transform = matches[0]

    work = g.copy()
    work[frame_mask] = bg
    out = work.copy()

    moves: list[tuple[list[tuple[int, int]], np.ndarray, int, int]] = []
    for coords in _components(work != bg, connectivity=8):
        info = _tail_info(work, coords, tail_color, bg)
        if info is None:
            continue
        _side, move, patch, r0, c0 = info
        transformed = _apply_transform(patch, transform, bg)
        ph, pw = transformed.shape
        if move == (1, 0):
            nr, nc = h - ph, c0
        elif move == (-1, 0):
            nr, nc = 0, c0
        elif move == (0, 1):
            nr, nc = r0, w - pw
        else:
            nr, nc = r0, 0

        moves.append((coords, transformed, nr, nc))

    for coords, _patch, _nr, _nc in moves:
        for r, c in coords:
            out[r, c] = bg

    for _coords, transformed, nr, nc in moves:
        mask = transformed != bg
        target = out[nr : nr + transformed.shape[0], nc : nc + transformed.shape[1]]
        assert not np.any(mask & (target != bg)), "moved sprite collision"
        target[mask] = transformed[mask]

    return out


def _random_connected_mask(
    height: int,
    width: int,
    min_cells: int,
    max_cells: int,
    required: tuple[int, int] | None = None,
) -> np.ndarray:
    target = random.randint(min_cells, min(max_cells, height * width))
    start = required if required is not None else (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))
    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _mask_signature(mask: np.ndarray) -> str:
    return str(mask.astype(int).tolist())


def _shape_signature(patch: np.ndarray) -> str:
    mask = patch != BG
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return _mask_signature(cropped)


def _colorize_mask(mask: np.ndarray, color_mode: str) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if color_mode == "mono":
        color = random.choice(PALETTE)
        for r, c in coords:
            patch[r, c] = color
        return patch

    colors = random.sample(PALETTE, 2)
    random.shuffle(coords)
    split = random.randint(1, len(coords) - 1)
    first = set(coords[:split])
    for r, c in coords:
        patch[r, c] = colors[0] if (r, c) in first else colors[1]
    return patch


def _random_training_patch(mode: str) -> np.ndarray:
    for _ in range(400):
        color_mode = "bi" if mode == "swap2" else random.choice(("mono", "bi"))
        mask = _random_connected_mask(5, 5, min_cells=4, max_cells=8)
        patch = np.full((5, 5), BG, dtype=int)
        colored = _colorize_mask(mask, color_mode)
        mh, mw = colored.shape
        top = random.randint(0, 5 - mh)
        left = random.randint(0, 5 - mw)
        patch[top : top + mh, left : left + mw] = colored
        transformed = _apply_transform(patch, mode, BG)
        if np.array_equal(transformed, patch):
            continue
        if _matching_transforms(patch, transformed, BG) == [mode]:
            return patch
    raise RuntimeError("failed to generate training patch")


def _required_anchor(side: str, height: int, width: int) -> tuple[int, int]:
    if side == "top":
        return (0, width // 2)
    if side == "bottom":
        return (height - 1, width // 2)
    if side == "left":
        return (height // 2, 0)
    return (height // 2, width - 1)


def _tail_property_ok(mask: np.ndarray, side: str) -> bool:
    h, w = mask.shape
    if side in ("top", "bottom"):
        if w % 2 == 0:
            return False
        c = w // 2
        return bool(mask[0 if side == "top" else h - 1, c])
    if h % 2 == 0:
        return False
    r = h // 2
    return bool(mask[r, 0 if side == "left" else w - 1])


def _random_object_patch(mode: str, tailed: bool, side: str | None, banned_shapes: set[str]) -> np.ndarray:
    for _ in range(400):
        if tailed and side in ("top", "bottom"):
            height = random.randint(2, 5)
            width = random.choice((3, 5))
        elif tailed and side in ("left", "right"):
            height = random.choice((3, 5))
            width = random.randint(2, 5)
        else:
            height = random.randint(2, 5)
            width = random.randint(2, 5)

        required = _required_anchor(side, height, width) if tailed and side is not None else None
        mask = _random_connected_mask(height, width, min_cells=3, max_cells=min(8, height * width), required=required)
        if tailed and side is not None and not _tail_property_ok(mask, side):
            continue
        sig = _mask_signature(mask)
        if sig in banned_shapes:
            continue

        if mode == "swap2" and tailed:
            color_mode = "bi"
        else:
            color_mode = random.choice(("mono", "bi"))
        patch = _colorize_mask(mask, color_mode)
        return patch
    raise RuntimeError("failed to generate object patch")


def _cells_for_patch(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = []
    for r, c in np.argwhere(patch != BG):
        coords.append((top + int(r), left + int(c)))
    return coords


def _tail_cell(side: str, top: int, left: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return (top - 1, left + w // 2)
    if side == "bottom":
        return (top + h, left + w // 2)
    if side == "left":
        return (top + h // 2, left - 1)
    return (top + h // 2, left + w)


def _can_place(cells: list[tuple[int, int]], occupied: np.ndarray, margin: int = 1) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        r0 = max(0, r - margin)
        r1 = min(h, r + margin + 1)
        c0 = max(0, c - margin)
        c1 = min(w, c + margin + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_cells(cells: list[tuple[int, int]], occupied: np.ndarray) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _paint_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    target = grid[top : top + h, left : left + w]
    mask = patch != BG
    target[mask] = patch[mask]


def _candidate_output_positions(
    patch: np.ndarray,
    side: str,
    output_occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = output_occupied.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    if side == "top":
        row = h - ph
        for col in range(w - pw + 1):
            cells = _cells_for_patch(patch, row, col)
            if _can_place(cells, output_occupied):
                positions.append((row, col))
    elif side == "bottom":
        row = 0
        for col in range(w - pw + 1):
            cells = _cells_for_patch(patch, row, col)
            if _can_place(cells, output_occupied):
                positions.append((row, col))
    elif side == "left":
        col = w - pw
        for row in range(h - ph + 1):
            cells = _cells_for_patch(patch, row, col)
            if _can_place(cells, output_occupied):
                positions.append((row, col))
    else:
        col = 0
        for row in range(h - ph + 1):
            cells = _cells_for_patch(patch, row, col)
            if _can_place(cells, output_occupied):
                positions.append((row, col))
    random.shuffle(positions)
    return positions


def _candidate_input_positions(
    patch: np.ndarray,
    side: str,
    output_top: int,
    output_left: int,
    input_occupied: np.ndarray,
) -> list[tuple[int, int]]:
    h, w = input_occupied.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    if side == "top":
        cols = [output_left]
        rows = list(range(1, h - ph))
        rows = [row for row in rows if row < output_top]
        random.shuffle(rows)
        for row in rows:
            col = cols[0]
            cells = _cells_for_patch(patch, row, col)
            cells.append(_tail_cell(side, row, col, patch.shape))
            if _can_place(cells, input_occupied):
                positions.append((row, col))
    elif side == "bottom":
        rows = list(range(1, h - ph))
        rows = [row for row in rows if row > output_top]
        random.shuffle(rows)
        for row in rows:
            col = output_left
            cells = _cells_for_patch(patch, row, col)
            cells.append(_tail_cell(side, row, col, patch.shape))
            if _can_place(cells, input_occupied):
                positions.append((row, col))
    elif side == "left":
        cols = list(range(1, w - pw))
        cols = [col for col in cols if col < output_left]
        random.shuffle(cols)
        for col in cols:
            row = output_top
            cells = _cells_for_patch(patch, row, col)
            cells.append(_tail_cell(side, row, col, patch.shape))
            if _can_place(cells, input_occupied):
                positions.append((row, col))
    else:
        cols = list(range(1, w - pw))
        cols = [col for col in cols if col > output_left]
        random.shuffle(cols)
        for col in cols:
            row = output_top
            cells = _cells_for_patch(patch, row, col)
            cells.append(_tail_cell(side, row, col, patch.shape))
            if _can_place(cells, input_occupied):
                positions.append((row, col))
    return positions


def _gen_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(22, 28)
    w = random.randint(22, 28)
    top_frame_r = random.randint(0, 1)
    top_frame_c = random.randint(0, 1)
    frame_gap = random.randint(1, 2)
    bottom_frame_r = top_frame_r + FRAME_SIZE + frame_gap
    if bottom_frame_r + FRAME_SIZE > h:
        return None

    transform = random.choice(TRANSFORMS)
    training_top = _random_training_patch(transform)
    training_bottom = _apply_transform(training_top, transform, BG)

    frame_area = np.zeros((h, w), dtype=bool)
    frame_area[top_frame_r : top_frame_r + FRAME_SIZE, top_frame_c : top_frame_c + FRAME_SIZE] = True
    frame_area[bottom_frame_r : bottom_frame_r + FRAME_SIZE, top_frame_c : top_frame_c + FRAME_SIZE] = True

    output_occupied = np.zeros((h, w), dtype=bool)
    output_occupied[top_frame_r : bottom_frame_r + FRAME_SIZE, top_frame_c : top_frame_c + FRAME_SIZE] = True

    input_occupied = frame_area.copy()

    total_objects = random.choice((3, 4))
    tailed_count = 2 if total_objects == 3 else random.choice((2, 3))
    object_specs = [True] * tailed_count + [False] * (total_objects - tailed_count)
    random.shuffle(object_specs)

    banned_shapes = {_shape_signature(training_top)}
    objects: list[dict[str, object]] = []
    for tailed in object_specs:
        side = random.choice(("top", "bottom", "left", "right")) if tailed else None
        input_patch = _random_object_patch(transform, tailed, side, banned_shapes)
        banned_shapes.add(_shape_signature(input_patch))
        output_patch = _apply_transform(input_patch, transform, BG) if tailed else input_patch.copy()

        if tailed:
            positions = _candidate_output_positions(output_patch, side, output_occupied)
            if not positions:
                return None
            chosen_output = None
            chosen_input = None
            for out_top, out_left in positions:
                input_positions = _candidate_input_positions(input_patch, side, out_top, out_left, input_occupied)
                if input_positions:
                    chosen_output = (out_top, out_left)
                    chosen_input = random.choice(input_positions)
                    break
            if chosen_output is None or chosen_input is None:
                return None

            out_cells = _cells_for_patch(output_patch, chosen_output[0], chosen_output[1])
            in_cells = _cells_for_patch(input_patch, chosen_input[0], chosen_input[1])
            tail = _tail_cell(side, chosen_input[0], chosen_input[1], input_patch.shape)
            _mark_cells(out_cells, output_occupied)
            _mark_cells(in_cells + [tail], input_occupied)
            objects.append(
                {
                    "tailed": True,
                    "side": side,
                    "input_patch": input_patch,
                    "output_patch": output_patch,
                    "input_pos": chosen_input,
                    "output_pos": chosen_output,
                    "tail": tail,
                }
            )
        else:
            attempts = []
            for row in range(1, h - input_patch.shape[0]):
                for col in range(1, w - input_patch.shape[1]):
                    cells = _cells_for_patch(input_patch, row, col)
                    if _can_place(cells, input_occupied) and _can_place(cells, output_occupied):
                        attempts.append((row, col))
            random.shuffle(attempts)
            if not attempts:
                return None
            row, col = attempts[0]
            cells = _cells_for_patch(input_patch, row, col)
            _mark_cells(cells, input_occupied)
            _mark_cells(cells, output_occupied)
            objects.append(
                {
                    "tailed": False,
                    "input_patch": input_patch,
                    "output_patch": input_patch.copy(),
                    "input_pos": (row, col),
                    "output_pos": (row, col),
                }
            )

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    inp[top_frame_r : top_frame_r + FRAME_SIZE, top_frame_c : top_frame_c + FRAME_SIZE] = GREY
    inp[top_frame_r + 1 : top_frame_r + 1 + INNER_SIZE, top_frame_c + 1 : top_frame_c + 1 + INNER_SIZE] = training_top
    inp[bottom_frame_r : bottom_frame_r + FRAME_SIZE, top_frame_c : top_frame_c + FRAME_SIZE] = GREY
    inp[bottom_frame_r + 1 : bottom_frame_r + 1 + INNER_SIZE, top_frame_c + 1 : top_frame_c + 1 + INNER_SIZE] = training_bottom

    for obj in objects:
        in_top, in_left = obj["input_pos"]
        out_top, out_left = obj["output_pos"]
        _paint_patch(inp, obj["input_patch"], in_top, in_left)
        _paint_patch(out, obj["output_patch"], out_top, out_left)
        if obj["tailed"]:
            tr, tc = obj["tail"]
            inp[tr, tc] = GREY

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_case()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid input")
