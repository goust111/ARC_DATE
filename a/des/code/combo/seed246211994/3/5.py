import random
from collections import Counter, deque

import numpy as np


ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL_STEPS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    for dr, dc in ORTHOGONAL_STEPS:
        yield r + dr, c + dc


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = ORTHOGONAL_STEPS if connectivity == 4 else ORTHOGONAL_STEPS + DIAGONAL_STEPS
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _boundary_background(grid: np.ndarray) -> int:
    h, w = grid.shape
    if h == 1:
        border = grid[0, :]
    elif w == 1:
        border = grid[:, 0]
    else:
        border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    return int(vals[counts.argmax()])


def _best_palette(grid: np.ndarray, bg: int) -> tuple[str, int, int]:
    h, w = grid.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        pal_mask = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal_mask[0:2, start:end] = True
        elif side == "bottom":
            pal_mask[h - 2 : h, start:end] = True
        elif side == "left":
            pal_mask[start:end, 0:2] = True
        else:
            pal_mask[start:end, w - 2 : w] = True

        rows, cols = np.where(pal_mask)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
            v = int(grid[r, c])
            if v == bg:
                continue
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not pal_mask[nr, nc] and int(grid[nr, nc]) == v:
                    return False
        return True

    def consider(side: str, outer: np.ndarray, inner: np.ndarray, axis_len: int):
        nonlocal best
        for start in range(axis_len):
            for end in range(start + 2, axis_len + 1):
                seg_outer = outer[start:end]
                seg_inner = inner[start:end]

                outer_colors = [int(v) for v in seg_outer.tolist() if int(v) != bg]
                if len(outer_colors) < 2 or len(outer_colors) != len(set(outer_colors)):
                    continue

                distinct_outer = set(outer_colors)
                distinct_inner = {int(v) for v in seg_inner.tolist() if int(v) != bg}
                if len(distinct_inner) < 2:
                    continue

                distinct_all = distinct_outer | distinct_inner
                if len(distinct_all) < 3:
                    continue

                if not isolated(side, start, end):
                    continue

                non_bg = int(np.count_nonzero(seg_outer != bg) + np.count_nonzero(seg_inner != bg))
                score = (len(outer_colors), len(distinct_all), -(end - start), non_bg)
                candidate = (score, side, start, end)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if h >= 2:
        consider("top", grid[0, :], grid[1, :], w)
        consider("bottom", grid[h - 1, :], grid[h - 2, :], w)
    if w >= 2:
        consider("left", grid[:, 0], grid[:, 1], h)
        consider("right", grid[:, w - 1], grid[:, w - 2], h)

    assert best is not None, "palette legend not found"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, list[int]]:
    h, w = grid.shape
    side, start, end = _best_palette(grid, bg)
    pal_mask = np.zeros((h, w), dtype=bool)

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = grid[0, start:end]
        fill_line = grid[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = grid[h - 1, start:end]
        fill_line = grid[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = grid[start:end, 0]
        fill_line = grid[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = grid[start:end, w - 1]
        fill_line = grid[start:end, w - 2]

    mapping: dict[int, int] = {}
    order: list[int] = []
    for border, fill in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border = int(border)
        fill = int(fill)
        if border == bg or border in mapping:
            continue
        mapping[border] = fill
        order.append(border)

    return mapping, pal_mask, order


def _fill_holes(grid: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray, order: list[int]) -> np.ndarray:
    out = grid.copy()
    priority = {color: idx for idx, color in enumerate(order)}
    bg_mask = (grid == bg) & (~pal_mask)
    for comp in _components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue

        border_counts: Counter[int] = Counter()
        comp_set = set(comp)
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    continue
                if (nr, nc) in comp_set or pal_mask[nr, nc]:
                    continue
                color = int(grid[nr, nc])
                if color in mapping:
                    border_counts[color] += 1

        if not border_counts:
            continue

        best_color = None
        best_count = -1
        best_rank = len(order)
        for color, count in border_counts.items():
            rank = priority.get(color, len(order))
            if count > best_count or (count == best_count and rank < best_rank):
                best_color = color
                best_count = count
                best_rank = rank

        fill_color = mapping[int(best_color)]
        for r, c in comp:
            out[r, c] = fill_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed background hole using the edge palette entry keyed by the surrounding border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _boundary_background(g)
    mapping, pal_mask, order = _palette_mapping(g, bg)
    assert mapping, "palette mapping must be non-empty"
    return _fill_holes(g, bg, mapping, pal_mask, order)


def _holes_in_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [
        comp
        for comp in _components_mask(~mask, connectivity=4)
        if not any(r == 0 or c == 0 or r == mask.shape[0] - 1 or c == mask.shape[1] - 1 for r, c in comp)
    ]


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask, connectivity=4)
    return len(comps) == 1


def _diamond_sprite() -> np.ndarray:
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 1] = True
    mask[1, 0] = True
    mask[1, 2] = True
    mask[2, 1] = True
    return mask


def _make_hollow_sprite(height: int, width: int, rng: random.Random) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    mask[1:-1, 1:-1] = False

    if height >= 5 and width >= 4 and rng.random() < 0.45:
        row = rng.randint(2, height - 2)
        span = rng.randint(1, width - 2)
        start = rng.randint(1, width - span)
        mask[row, start : start + span] = True

    if width >= 5 and height >= 4 and rng.random() < 0.45:
        col = rng.randint(2, width - 2)
        span = rng.randint(1, height - 2)
        start = rng.randint(1, height - span)
        mask[start : start + span, col] = True

    if rng.random() < 0.35:
        border_cells = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if mask[r, c] and (r in (0, height - 1) or c in (0, width - 1))
        ]
        rng.shuffle(border_cells)
        for r, c in border_cells[: rng.randint(1, min(4, len(border_cells)))]:
            test = mask.copy()
            test[r, c] = False
            if _is_connected(test) and _holes_in_mask(test):
                mask = test

    if not _is_connected(mask):
        return None

    holes = _holes_in_mask(mask)
    if not holes:
        return None

    if int(mask.sum()) > 22:
        return None

    return mask


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        if side == "top":
            pal[0, :] = border_keys
            pal[1, :] = fill_values
        else:
            pal[0, :] = fill_values
            pal[1, :] = border_keys
    else:
        pal = np.zeros((n, 2), dtype=int)
        if side == "left":
            pal[:, 0] = border_keys
            pal[:, 1] = fill_values
        else:
            pal[:, 0] = fill_values
            pal[:, 1] = border_keys
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        r0 = 0
        c0 = rng.randint(0, w - pw)
    elif side == "bottom":
        r0 = h - ph
        c0 = rng.randint(0, w - pw)
    elif side == "left":
        r0 = rng.randint(0, h - ph)
        c0 = 0
    else:
        r0 = rng.randint(0, h - ph)
        c0 = w - pw

    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _valid_positions(grid_shape: tuple[int, int], occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid_shape
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    if sh > h - 2 or sw > w - 2:
        return positions
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            r1 = r0 + sh
            c1 = c0 + sw
            if occ[max(0, r0 - 1) : min(h, r1 + 1), max(0, c0 - 1) : min(w, c1 + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    border_color: int,
    fill_color: int,
) -> list[list[tuple[int, int]]]:
    sh, sw = mask.shape
    grid[top : top + sh, left : left + sw][mask] = border_color
    holes = _holes_in_mask(mask)
    for comp in holes:
        for r, c in comp:
            grid[top + r, left + c] = fill_color
    return holes


def _choose_colors(rng: random.Random, bg: int, count: int) -> tuple[list[int], list[int]]:
    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    assert 2 * count <= len(colors)
    border_keys = colors[:count]
    fill_values = colors[count : 2 * count]
    return border_keys, fill_values


def _random_sprite(rng: random.Random, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h >= 3 and max_w >= 3 and rng.random() < 0.15:
        return _diamond_sprite()

    height = rng.randint(3, max_h)
    width = rng.randint(3, max_w)
    if height + width > 11:
        width = max(3, 11 - height)
    return _make_hollow_sprite(height, width, rng)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randint(0, 9)
    height = rng.randint(9, 16)
    width = rng.randint(9, 16)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_entries = min(4, height if side in ("left", "right") else width)
    if max_entries < 2:
        return None
    palette_size = rng.randint(2, max_entries)

    border_keys, fill_values = _choose_colors(rng, bg, palette_size)
    mapping = dict(zip(border_keys, fill_values, strict=False))

    used_key_count = rng.randint(1, palette_size)
    used_keys = rng.sample(border_keys, used_key_count)
    sprite_count = rng.randint(used_key_count, min(used_key_count + 2, 5))
    sprite_colors = used_keys[:]
    while len(sprite_colors) < sprite_count:
        sprite_colors.append(rng.choice(used_keys))
    rng.shuffle(sprite_colors)

    grid = np.full((height, width), bg, dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    palette = _build_palette(side, border_keys, fill_values)
    _place_palette(grid, occ, palette, side, rng)

    placements: list[tuple[int, int, np.ndarray, list[list[tuple[int, int]]]]] = []
    max_sprite_h = min(7, height - 2)
    max_sprite_w = min(7, width - 2)
    if max_sprite_h < 3 or max_sprite_w < 3:
        return None

    for border_color in sprite_colors:
        mask = _random_sprite(rng, max_sprite_h, max_sprite_w)
        if mask is None:
            return None
        positions = _valid_positions(grid.shape, occ, mask)
        if not positions:
            return None
        top, left = rng.choice(positions)
        holes = _paint_sprite(grid, top, left, mask, border_color, mapping[border_color])
        placements.append((top, left, mask, holes))
        sh, sw = mask.shape
        occ[max(0, top - 1) : min(height, top + sh + 1), max(0, left - 1) : min(width, left + sw + 1)] = True

    output = grid.copy()
    inp = output.copy()
    changed = 0
    for top, left, _mask, holes in placements:
        for comp in holes:
            for r, c in comp:
                inp[top + r, left + c] = bg
                changed += 1

    if changed == 0:
        return None

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, counts = np.unique(border, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
