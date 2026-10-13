import random
from collections import deque

import numpy as np


BG = 0
SEPARATOR_COLOR = 9
DIRECTIONS = ("up", "down", "left", "right")
GENERATE_RETRIES = 300
MAX_SIDE = 30


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _components(mask)
    assert comps, "expected at least one component"
    return max(comps, key=len)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _density_specs(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    h, w = mask.shape
    return {
        "up": (int(mask[0, :].sum()), int(w)),
        "down": (int(mask[-1, :].sum()), int(w)),
        "left": (int(mask[:, 0].sum()), int(h)),
        "right": (int(mask[:, -1].sum()), int(h)),
    }


def _opening_side(mask: np.ndarray) -> str:
    specs = _density_specs(mask)
    best_side = None
    best_num = 0
    best_den = 1
    ties = 0
    for side in DIRECTIONS:
        num, den = specs[side]
        if best_side is None or num * best_den < best_num * den:
            best_side = side
            best_num = num
            best_den = den
            ties = 1
        elif num * best_den == best_num * den:
            ties += 1
    assert best_side is not None and ties == 1, "opening side must be unique"
    return best_side


def _rotate_to_direction(mask: np.ndarray, direction: str) -> np.ndarray:
    for k in range(4):
        rotated = _crop(np.rot90(mask, k))
        if _opening_side(rotated) == direction:
            return rotated
    raise AssertionError("no rotation matches requested direction")


def _find_separator_band(grid: np.ndarray) -> tuple[int, int, int]:
    rows = [
        r
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != BG and bool(np.all(grid[r] == grid[r, 0]))
    ]
    assert rows, "separator row not found"
    runs = []
    start = rows[0]
    prev = rows[0]
    for row in rows[1:]:
        if row == prev + 1 and int(grid[row, 0]) == int(grid[start, 0]):
            prev = row
            continue
        runs.append((start, prev))
        start = row
        prev = row
    runs.append((start, prev))
    top, bottom = max(runs, key=lambda rc: rc[1] - rc[0])
    color = int(grid[top, 0])
    return int(top), int(bottom), color


def _extract_component_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = _largest_component(mask)
    r0, c0, r1, c1 = _bbox(coords)
    sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        sub[int(r - r0), int(c - c0)] = True
    return sub, (int(r0), int(c0), int(r1), int(c1))


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    canvas[top : top + h, left : left + w][mask] = int(color)


def _centered_top_left(
    outer_bbox: tuple[int, int, int, int], piece_shape: tuple[int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = outer_bbox
    inner_h = int(r1 - r0 - 1)
    inner_w = int(c1 - c0 - 1)
    piece_h, piece_w = piece_shape
    assert piece_h <= inner_h and piece_w <= inner_w
    dh = int(inner_h - piece_h)
    dw = int(inner_w - piece_w)
    assert dh % 2 == 0 and dw % 2 == 0, "centers must align exactly"
    return int(r0 + 1 + dh // 2), int(c0 + 1 + dw // 2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each lower-region fragment to the upper-region frame of the same color.
    Rotate the fragment so its sparsest bbox side points toward the frame gap,
    then place it centered inside the frame. Clear the lower fragments.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_top, sep_bottom, sep_color = _find_separator_band(g)

    out = np.zeros((h, w), dtype=int)
    out[:sep_top] = g[:sep_top]
    out[sep_top : sep_bottom + 1] = g[sep_top : sep_bottom + 1]

    upper = g[:sep_top]
    lower = g[sep_bottom + 1 :]
    upper_colors = {int(x) for x in np.unique(upper) if int(x) not in (BG, sep_color)}
    lower_colors = {int(x) for x in np.unique(lower) if int(x) not in (BG, sep_color)}

    for color in sorted(upper_colors & lower_colors):
        frame_mask, frame_bbox = _extract_component_mask(upper == color)
        piece_mask, _piece_bbox = _extract_component_mask(lower == color)
        gap_direction = _opening_side(frame_mask)
        rotated_piece = _rotate_to_direction(piece_mask, gap_direction)
        top, left = _centered_top_left(frame_bbox, rotated_piece.shape)
        _paste_mask(out, rotated_piece, int(top), int(left), int(color))

    return out


def _make_piece_up(rng: random.Random) -> np.ndarray:
    piece_h = rng.randint(3, 5)
    piece_w = rng.randint(3, 5)
    mask = np.ones((piece_h, piece_w), dtype=bool)

    notch_w = rng.randint(1, piece_w - 2)
    notch_depth = rng.randint(1, piece_h - 2)
    notch_c0 = rng.randint(1, piece_w - notch_w - 1)
    mask[:notch_depth, notch_c0 : notch_c0 + notch_w] = False

    return mask


def _make_frame_patch(
    color: int, inner_h: int, inner_w: int, gap_direction: str, rng: random.Random
) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    if gap_direction in ("up", "down"):
        row = 0 if gap_direction == "up" else patch.shape[0] - 1
        gap_len = rng.randint(1, patch.shape[1] - 2)
        start = rng.randint(1, patch.shape[1] - gap_len - 1)
        patch[row, start : start + gap_len] = BG
    else:
        col = 0 if gap_direction == "left" else patch.shape[1] - 1
        gap_len = rng.randint(1, patch.shape[0] - 2)
        start = rng.randint(1, patch.shape[0] - gap_len - 1)
        patch[start : start + gap_len, col] = BG

    return patch


def _arrange_patches(
    patches: list[np.ndarray], rng: random.Random
) -> tuple[dict[int, tuple[int, int]], int, int]:
    count = len(patches)
    order = list(range(count))
    rng.shuffle(order)

    if count == 1:
        rows = [order]
    elif count == 2:
        rows = [order] if rng.random() < 0.65 else [[order[0]], [order[1]]]
    else:
        if rng.random() < 0.45:
            rows = [order]
        else:
            split = rng.randint(1, count - 1)
            rows = [order[:split], order[split:]]

    layouts = []
    total_h = 0
    max_w = 0
    for row_index, row in enumerate(rows):
        row_h = max(int(patches[i].shape[0]) for i in row)
        gaps = [rng.randint(1, 3) for _ in range(max(0, len(row) - 1))]
        x = 0
        items = []
        for j, idx in enumerate(row):
            patch_h, patch_w = patches[idx].shape
            dy = 0 if patch_h == row_h else rng.randint(0, row_h - patch_h)
            items.append((idx, int(total_h + dy), int(x)))
            x += int(patch_w)
            if j < len(row) - 1:
                x += gaps[j]
        layouts.append((items, int(x)))
        total_h += int(row_h)
        if row_index < len(rows) - 1:
            total_h += rng.randint(1, 3)
        max_w = max(max_w, int(x))

    coords: dict[int, tuple[int, int]] = {}
    for items, row_w in layouts:
        shift_x = 0 if row_w == max_w else rng.randint(0, max_w - row_w)
        for idx, r, c in items:
            coords[int(idx)] = (int(r), int(c + shift_x))

    return coords, int(total_h), int(max_w)


def _build_expected(
    grid: np.ndarray,
    frame_boxes: list[tuple[int, int, int, int]],
    target_masks: list[np.ndarray],
    colors: list[int],
    sep_row: int,
) -> np.ndarray:
    out = np.zeros_like(grid)
    out[:sep_row] = grid[:sep_row]
    out[sep_row] = grid[sep_row]
    for bbox, mask, color in zip(frame_boxes, target_masks, colors):
        top, left = _centered_top_left(bbox, mask.shape)
        _paste_mask(out, mask, int(top), int(left), int(color))
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    obj_count = rng.randint(2, 4)
    colors = rng.sample([1, 2, 3, 4, 5, 6, 7, 8], obj_count)

    frame_patches: list[np.ndarray] = []
    input_piece_patches: list[np.ndarray] = []
    target_piece_masks: list[np.ndarray] = []

    for color in colors:
        piece_up = _make_piece_up(rng)
        gap_direction = rng.choice(DIRECTIONS)
        target_piece = _rotate_to_direction(piece_up, gap_direction)
        pad_r = rng.randint(0, 1)
        pad_c = rng.randint(0, 1)
        inner_h = int(target_piece.shape[0] + 2 * pad_r)
        inner_w = int(target_piece.shape[1] + 2 * pad_c)
        frame_patches.append(_make_frame_patch(color, inner_h, inner_w, gap_direction, rng))
        target_piece_masks.append(target_piece)

        input_piece = _crop(np.rot90(piece_up, rng.randrange(4)))
        input_patch = np.zeros(input_piece.shape, dtype=int)
        input_patch[input_piece] = int(color)
        input_piece_patches.append(input_patch)

    frame_coords, upper_h, upper_w = _arrange_patches(frame_patches, rng)
    piece_coords, lower_h, lower_w = _arrange_patches(input_piece_patches, rng)

    width = max(upper_w, lower_w) + rng.randint(2, 5)
    upper_x = 0 if width == upper_w else rng.randint(0, width - upper_w)
    lower_x = 0 if width == lower_w else rng.randint(0, width - lower_w)
    top_margin = rng.randint(1, 3)
    gap_before_sep = rng.randint(1, 2)
    gap_after_sep = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 3)
    sep_row = int(top_margin + upper_h + gap_before_sep)
    lower_y = int(sep_row + 1 + gap_after_sep)
    height = int(lower_y + lower_h + bottom_margin)

    if height > MAX_SIDE or width > MAX_SIDE:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[sep_row, :] = SEPARATOR_COLOR

    frame_boxes: list[tuple[int, int, int, int]] = []
    for idx, patch in enumerate(frame_patches):
        r = int(top_margin + frame_coords[idx][0])
        c = int(upper_x + frame_coords[idx][1])
        ph, pw = patch.shape
        frame_boxes.append((r, c, r + ph - 1, c + pw - 1))
        grid[r : r + ph, c : c + pw] = patch

    for idx, patch in enumerate(input_piece_patches):
        r = int(lower_y + piece_coords[idx][0])
        c = int(lower_x + piece_coords[idx][1])
        ph, pw = patch.shape
        region = grid[r : r + ph, c : c + pw]
        if region.shape != patch.shape or bool(np.any((region != BG) & (patch != BG))):
            return None
        region[patch != BG] = patch[patch != BG]

    expected = _build_expected(grid, frame_boxes, target_piece_masks, colors, sep_row)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after retries")
