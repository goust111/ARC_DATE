from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _non_separator_spans(sep_mask: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_sep in enumerate(sep_mask.tolist()):
        if not is_sep and start is None:
            start = idx
        elif is_sep and start is not None:
            spans.append((int(start), idx - 1))
            start = None
    if start is not None:
        spans.append((int(start), len(sep_mask) - 1))
    return spans


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _rotation_keys(mask: np.ndarray) -> set[tuple[str, ...]]:
    base = _tight_mask(mask)
    return {_mask_key(np.rot90(base, k)) for k in range(4)}


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells = set()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _find_layout(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    best: tuple[tuple[int, int, int], int, list[tuple[int, int]], list[tuple[int, int]]] | None = None

    for divider in map(int, np.unique(g)):
        row_sep = np.all(g == divider, axis=1)
        col_sep = np.all(g == divider, axis=0)
        if not row_sep.any() or not col_sep.any():
            continue

        row_spans = _non_separator_spans(row_sep)
        col_spans = _non_separator_spans(col_sep)
        if len(row_spans) < 2 or len(col_spans) < 1:
            continue

        row_sizes = {end - start + 1 for start, end in row_spans}
        col_sizes = {end - start + 1 for start, end in col_spans}
        if len(row_sizes) != 1 or len(col_sizes) != 1:
            continue

        room_size = next(iter(row_sizes))
        if room_size != next(iter(col_sizes)) or room_size < 2:
            continue

        score = (
            len(row_spans) * len(col_spans),
            int(row_sep.sum() + col_sep.sum()),
            room_size,
        )
        if best is None or score > best[0]:
            best = (score, divider, row_spans, col_spans)

    assert best is not None, "failed to detect the room grid"
    _, divider, row_spans, col_spans = best

    non_divider = g[g != divider]
    assert non_divider.size, "grid cannot be all divider"
    counts = np.bincount(non_divider.ravel())
    bg = int(counts.argmax())
    return bg, int(divider), row_spans, col_spans


def _foreground_mask(patch: np.ndarray, bg: int, divider: int) -> np.ndarray:
    return (patch != int(bg)) & (patch != int(divider))


def _seed_rotation(coords: tuple[int, int], room_size: int) -> int:
    corner_to_k = {
        (0, 0): 0,
        (0, room_size - 1): 3,
        (room_size - 1, room_size - 1): 2,
        (room_size - 1, 0): 1,
    }
    assert coords in corner_to_k, f"seed must be in a room corner, got {coords}"
    return corner_to_k[coords]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, divider, row_spans, col_spans = _find_layout(g)
    room_size = row_spans[0][1] - row_spans[0][0] + 1

    templates: dict[int, np.ndarray] = {}
    template_row = len(row_spans) - 1
    for ci, (c0, c1) in enumerate(col_spans):
        r0, r1 = row_spans[template_row]
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        fg = _foreground_mask(patch, bg, divider)
        assert fg.any(), f"template room {ci} is empty"
        colors = {int(v) for v in patch[fg]}
        assert len(colors) == 1, "template room must contain exactly one pattern color"
        color = colors.pop()
        assert color not in templates, "template colors must be unique"
        templates[color] = fg.copy()

    out = g.copy()
    for ri, (r0, r1) in enumerate(row_spans[:-1]):
        for ci, (c0, c1) in enumerate(col_spans):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            fg = _foreground_mask(patch, bg, divider)
            seed_coords = np.argwhere(fg)
            assert seed_coords.shape[0] == 1, "target room must contain exactly one seed"
            sr, sc = map(int, seed_coords[0])
            color = int(patch[sr, sc])
            assert color in templates, "seed color must match a template color"

            rotated = np.rot90(templates[color], _seed_rotation((sr, sc), room_size))
            room = out[r0 : r1 + 1, c0 : c1 + 1]
            room[rotated] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each non-template room with the color-matched template rotated to the seed corner.

    Input structure:
    - Black background cells and grey separator lines partition the grid into equal square rooms.
    - Every room in the bottom row is a template room containing one asymmetric monochrome pattern
      drawn near the top-left corner.
    - Every other room contains exactly one seed pixel whose color matches one template color, and
      whose position is one of the room's four corners.

    Rule:
    - Use seed color to choose the matching bottom-row template.
    - Interpret seed corner as rotation: TL=0, TR=90 clockwise, BR=180, BL=270 clockwise.
    - Paint the rotated template into the corresponding room corner, keeping the template row unchanged.
    """
    return _solve(grid)


# --------------- generator ---------------

ROOM_SIZE_RANGE = (5, 7)
ROOM_ROW_RANGE = (3, 5)
ROOM_COL_RANGE = (2, 5)
GEN_ATTEMPTS = 300
TEMPLATE_BOX_MAX = 4
CORNERS = ("tl", "tr", "br", "bl")


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                node = (nr, nc)
                if node not in seen:
                    seen.add(node)
                    stack.append(node)
    return len(seen) == int(mask.sum())


def _random_template_shape(room_size: int) -> np.ndarray | None:
    max_dim = min(TEMPLATE_BOX_MAX, room_size - 1)
    if max_dim < 2:
        return None

    for _ in range(120):
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        target_area = random.randint(4, max(4, h * w - 1))

        mask = np.zeros((h, w), dtype=bool)
        mask[0, 0] = True
        mask[0, random.randint(1, w - 1)] = True
        mask[random.randint(1, h - 1), 0] = True

        while int(mask.sum()) < target_area:
            options = _frontier(mask)
            if not options:
                break
            r, c = random.choice(options)
            mask[r, c] = True

        if int(mask.sum()) != target_area:
            continue
        if not np.any(mask[0, 1:]) or not np.any(mask[1:, 0]):
            continue
        if not _is_connected(mask):
            continue
        if len(_rotation_keys(mask)) != 4:
            continue
        return mask

    return None


def _full_template_patch(room_size: int, shape: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((room_size, room_size), BG, dtype=int)
    h, w = shape.shape
    patch[:h, :w][shape] = color
    return patch


def _balanced_sequence(items: list, count: int) -> list:
    out: list = []
    while len(out) < count:
        block = list(items)
        random.shuffle(block)
        out.extend(block)
    return out[:count]


def _room_slice(index: int, room_size: int) -> slice:
    start = 1 + index * (room_size + 1)
    return slice(start, start + room_size)


def _corner_position(corner: str, room_size: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, room_size - 1
    if corner == "br":
        return room_size - 1, room_size - 1
    if corner == "bl":
        return room_size - 1, 0
    raise AssertionError(f"unknown corner {corner}")


def _build_expected(
    grid: np.ndarray,
    room_size: int,
    room_rows: int,
    room_cols: int,
    templates: dict[int, np.ndarray],
    assignments: list[tuple[int, int, int, str]],
) -> np.ndarray:
    out = grid.copy()
    for ri, ci, color, corner in assignments:
        rotated = np.rot90(templates[color], _seed_rotation(_corner_position(corner, room_size), room_size))
        rr = _room_slice(ri, room_size)
        cc = _room_slice(ci, room_size)
        out[rr, cc][rotated] = color
    assert room_rows >= 2 and room_cols >= 1
    return out


def _gen_attempt() -> np.ndarray | None:
    room_size = random.randint(*ROOM_SIZE_RANGE)
    room_rows = random.randint(*ROOM_ROW_RANGE)
    room_cols = random.randint(*ROOM_COL_RANGE)
    if room_cols > len(PALETTE):
        return None

    colors = random.sample(PALETTE, room_cols)
    template_masks: dict[int, np.ndarray] = {}
    seen_shapes: set[tuple[str, ...]] = set()
    for color in colors:
        shape = _random_template_shape(room_size)
        if shape is None:
            return None
        key = _mask_key(shape)
        if key in seen_shapes:
            return None
        seen_shapes.add(key)
        template_masks[color] = _full_template_patch(room_size, shape, color) == color

    height = room_rows * room_size + room_rows + 1
    width = room_cols * room_size + room_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    for r in range(0, height, room_size + 1):
        grid[r, :] = DIVIDER
    for c in range(0, width, room_size + 1):
        grid[:, c] = DIVIDER

    template_row = room_rows - 1
    for ci, color in enumerate(colors):
        rr = _room_slice(template_row, room_size)
        cc = _room_slice(ci, room_size)
        patch = np.full((room_size, room_size), BG, dtype=int)
        patch[template_masks[color]] = color
        grid[rr, cc] = patch

    target_count = (room_rows - 1) * room_cols
    if target_count < 4:
        return None

    corner_seq = _balanced_sequence(list(CORNERS), target_count)
    color_seq = _balanced_sequence(colors, target_count)
    assignments: list[tuple[int, int, int, str]] = []
    idx = 0
    for ri in range(room_rows - 1):
        for ci in range(room_cols):
            color = int(color_seq[idx])
            corner = str(corner_seq[idx])
            rr = _room_slice(ri, room_size)
            cc = _room_slice(ci, room_size)
            sr, sc = _corner_position(corner, room_size)
            room = np.full((room_size, room_size), BG, dtype=int)
            room[sr, sc] = color
            grid[rr, cc] = room
            assignments.append((ri, ci, color, corner))
            idx += 1

    expected = _build_expected(
        grid=grid,
        room_size=room_size,
        room_rows=room_rows,
        room_cols=room_cols,
        templates=template_masks,
        assignments=assignments,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
