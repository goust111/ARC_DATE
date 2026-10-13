from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
BORDER_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_PLACEMENT_ATTEMPTS = 120


def _most_common(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(values).ravel())
    assert counts
    return counts.most_common(1)[0][0]


def _components_by_color(grid: np.ndarray, bg: int, border: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (bg, border):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords_arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords_arr,
                    "bbox": (
                        int(coords_arr[:, 0].min()),
                        int(coords_arr[:, 1].min()),
                        int(coords_arr[:, 0].max()),
                        int(coords_arr[:, 1].max()),
                    ),
                }
            )
    return components


def _contact_stats(coords: np.ndarray, height: int, width: int) -> dict[str, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return {
        "top": int(np.count_nonzero(rows == 1)),
        "bottom": int(np.count_nonzero(rows == height - 2)),
        "left": int(np.count_nonzero(cols == 1)),
        "right": int(np.count_nonzero(cols == width - 2)),
    }


def _select_side(stats: dict[str, int]) -> str | None:
    positives = [(side, length) for side, length in stats.items() if length > 0]
    if not positives:
        return None
    positives.sort(key=lambda item: (-item[1], ("top", "bottom", "left", "right").index(item[0])))
    return positives[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """Flatten each border-touching object into a one-dimensional projection on its contacted frame edge.

    Input structure:
    - The background is the dominant interior color, and the whole outer ring is a one-pixel border of one other color.
    - Inside the border are several monochrome 4-connected foreground objects on the background.
    - Some objects are adjacent to one frame edge, some are free interior objects, and some corner objects can be adjacent to two frame edges.

    Rule:
    - Keep every free object unchanged.
    - For each object that touches the frame, measure how many of its cells touch each eligible edge; if two edges are touched, use the longer contact edge.
    - Erase that object's interior cells to background, then color the corresponding border positions for every occupied row or column along the chosen edge.

    Key insight:
    - Border-touching objects are not translated intact; they lose their depth and are reduced to a one-dimensional shadow on the frame, while free objects stay untouched.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    border = int(g[0, 0])
    assert np.all(g[0, :] == border)
    assert np.all(g[-1, :] == border)
    assert np.all(g[:, 0] == border)
    assert np.all(g[:, -1] == border)

    bg = _most_common(g[g != border])
    out = g.copy()

    for comp in _components_by_color(g, bg=bg, border=border):
        side = _select_side(_contact_stats(comp["coords"], h, w))
        if side is None:
            continue

        rows = comp["coords"][:, 0]
        cols = comp["coords"][:, 1]
        out[rows, cols] = bg

        color = int(comp["color"])
        if side == "top":
            out[0, np.unique(cols)] = color
        elif side == "bottom":
            out[h - 1, np.unique(cols)] = color
        elif side == "left":
            out[np.unique(rows), 0] = color
        else:
            out[np.unique(rows), w - 1] = color

    return out


def _flip_h(mask: np.ndarray) -> np.ndarray:
    return mask[:, ::-1]


def _flip_v(mask: np.ndarray) -> np.ndarray:
    return mask[::-1, :]


def _top_hist_mask(height: int, width: int) -> np.ndarray:
    depths = [random.randint(1, height) for _ in range(width)]
    depths[random.randrange(width)] = height
    mask = np.zeros((height, width), dtype=bool)
    for c, depth in enumerate(depths):
        mask[:depth, c] = True
    return mask


def _left_hist_mask(height: int, width: int) -> np.ndarray:
    widths = [random.randint(1, width) for _ in range(height)]
    widths[random.randrange(height)] = width
    mask = np.zeros((height, width), dtype=bool)
    for r, span in enumerate(widths):
        mask[r, :span] = True
    return mask


def _corner_tl_mask(top_len: int, left_len: int, height: int, width: int) -> np.ndarray:
    assert 1 <= left_len <= height
    assert 1 <= top_len <= width
    mask = np.zeros((height, width), dtype=bool)

    depths = [random.randint(1, height) for _ in range(top_len)]
    depths[0] = left_len
    for c, depth in enumerate(depths):
        mask[:depth, c] = True

    extra_growth = random.randint(0, max(1, height * width // 4))
    frontier = [(r, c) for r in range(height) for c in range(width) if mask[r, c]]
    for _ in range(extra_growth):
        candidates: list[tuple[int, int]] = []
        for r, c in frontier:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                if mask[nr, nc]:
                    continue
                if nr == 0 and nc >= top_len:
                    continue
                if nc == 0 and nr >= left_len:
                    continue
                candidates.append((nr, nc))
        if not candidates:
            break
        nr, nc = random.choice(candidates)
        mask[nr, nc] = True
        frontier.append((nr, nc))

    assert np.count_nonzero(mask[0, :]) == top_len
    assert np.count_nonzero(mask[:, 0]) == left_len
    return mask


def _touching_mask(side: str, secondary: str | None, height: int, width: int) -> np.ndarray:
    if secondary is None:
        if side == "top":
            return _top_hist_mask(height, width)
        if side == "bottom":
            return _flip_v(_top_hist_mask(height, width))
        if side == "left":
            return _left_hist_mask(height, width)
        return _flip_h(_left_hist_mask(height, width))

    if {side, secondary} not in ({"top", "left"}, {"top", "right"}, {"bottom", "left"}, {"bottom", "right"}):
        raise AssertionError("corner sides must be adjacent")

    if side in ("top", "bottom"):
        top_len = random.randint(max(2, width // 2), width)
        left_len = random.randint(1, min(height, max(1, top_len - 1)))
    else:
        left_len = random.randint(max(2, height // 2), height)
        top_len = random.randint(1, min(width, max(1, left_len - 1)))

    mask = _corner_tl_mask(top_len=top_len, left_len=left_len, height=height, width=width)
    if "right" in (side, secondary):
        mask = _flip_h(mask)
    if "bottom" in (side, secondary):
        mask = _flip_v(mask)
    return mask


def _free_mask(height: int, width: int) -> np.ndarray:
    mode = random.choice(("top", "bottom", "left", "right", "blob"))
    if mode == "top":
        return _top_hist_mask(height, width)
    if mode == "bottom":
        return _flip_v(_top_hist_mask(height, width))
    if mode == "left":
        return _left_hist_mask(height, width)
    if mode == "right":
        return _flip_h(_left_hist_mask(height, width))

    cells = {(random.randrange(height), random.randrange(width))}
    target = random.randint(max(3, (height + width) // 2), height * width)
    while len(cells) < target:
        r, c = random.choice(tuple(cells))
        dr, dc = random.choice(((-1, 0), (1, 0), (0, -1), (0, 1)))
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width:
            cells.add((nr, nc))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    mask = np.zeros((max(rows) - r0 + 1, max(cols) - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _projection_positions(mask: np.ndarray, side: str, row0: int, col0: int) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    rows = coords[:, 0] + row0
    cols = coords[:, 1] + col0
    if side == "top":
        return {(0, int(c)) for c in np.unique(cols)}
    if side == "bottom":
        return {(row0 + mask.shape[0], int(c)) for c in np.unique(cols)}
    if side == "left":
        return {(int(r), 0) for r in np.unique(rows)}
    return {(int(r), col0 + mask.shape[1]) for r in np.unique(rows)}


def _can_place(mask: np.ndarray, row0: int, col0: int, occupied: np.ndarray, padding: int = 1) -> bool:
    h, w = occupied.shape
    coords = np.argwhere(mask)
    for lr, lc in coords:
        r = row0 + int(lr)
        c = col0 + int(lc)
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        r_lo = max(1, r - padding)
        r_hi = min(h - 2, r + padding)
        c_lo = max(1, c - padding)
        c_hi = min(w - 2, c + padding)
        if occupied[r_lo : r_hi + 1, c_lo : c_hi + 1].any():
            return False
    return True


def _stamp(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, row0: int, col0: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    global_coords = np.empty_like(coords)
    for idx, (lr, lc) in enumerate(coords):
        r = row0 + int(lr)
        c = col0 + int(lc)
        grid[r, c] = color
        occupied[r, c] = True
        global_coords[idx] = (r, c)
    return global_coords


def _side_ranges(side: str, mask: np.ndarray, grid_shape: tuple[int, int]) -> tuple[list[int], list[int]]:
    h, w = grid_shape
    mh, mw = mask.shape
    if side == "top":
        return [1], list(range(2, w - mw - 1))
    if side == "bottom":
        return [h - mh - 1], list(range(2, w - mw - 1))
    if side == "left":
        return list(range(2, h - mh - 1)), [1]
    return list(range(2, h - mh - 1)), [w - mw - 1]


def _corner_anchor(side: str, secondary: str, mask: np.ndarray, grid_shape: tuple[int, int]) -> tuple[int, int]:
    h, w = grid_shape
    mh, mw = mask.shape
    row0 = 1 if "top" in (side, secondary) else h - mh - 1
    col0 = 1 if "left" in (side, secondary) else w - mw - 1
    return row0, col0


def _try_place_touching(
    grid: np.ndarray,
    occupied: np.ndarray,
    reserved_border: set[tuple[int, int]],
    color: int,
    side: str,
    secondary: str | None,
) -> tuple[np.ndarray, str] | None:
    h, w = grid.shape
    for _ in range(MAX_OBJECT_PLACEMENT_ATTEMPTS):
        if side in ("top", "bottom"):
            mh = random.randint(2, min(6, h - 4))
            mw = random.randint(2, min(7, w - 5))
        else:
            mh = random.randint(2, min(7, h - 5))
            mw = random.randint(2, min(6, w - 4))

        if secondary is not None:
            if mh < 2 or mw < 2:
                continue
        mask = _touching_mask(side=side, secondary=secondary, height=mh, width=mw)

        if secondary is None:
            row_choices, col_choices = _side_ranges(side, mask, grid.shape)
            if not row_choices or not col_choices:
                continue
            row0 = random.choice(row_choices)
            col0 = random.choice(col_choices)
        else:
            row0, col0 = _corner_anchor(side, secondary, mask, grid.shape)

        if not _can_place(mask, row0, col0, occupied):
            continue

        coords_preview = np.argwhere(mask)
        preview_global = coords_preview + np.array([[row0, col0]])
        stats = _contact_stats(preview_global, h, w)
        chosen = _select_side(stats)
        if chosen != side:
            continue

        proj = _projection_positions(mask, side, row0, col0)
        if proj & reserved_border:
            continue

        coords = _stamp(grid, occupied, mask, row0, col0, color)
        reserved_border.update(proj)
        return coords, side
    return None


def _try_place_free(grid: np.ndarray, occupied: np.ndarray, color: int) -> np.ndarray | None:
    h, w = grid.shape
    for _ in range(MAX_OBJECT_PLACEMENT_ATTEMPTS):
        mh = random.randint(2, min(6, h - 5))
        mw = random.randint(2, min(6, w - 5))
        mask = _free_mask(mh, mw)
        mh2, mw2 = mask.shape
        row_choices = list(range(2, h - mh2 - 1))
        col_choices = list(range(2, w - mw2 - 1))
        if not row_choices or not col_choices:
            continue
        row0 = random.choice(row_choices)
        col0 = random.choice(col_choices)

        if not _can_place(mask, row0, col0, occupied):
            continue

        coords_preview = np.argwhere(mask) + np.array([[row0, col0]])
        if _select_side(_contact_stats(coords_preview, h, w)) is not None:
            continue
        return _stamp(grid, occupied, mask, row0, col0, color)
    return None


def _validate_generated_grid(grid: np.ndarray) -> bool:
    border = int(grid[0, 0])
    if not (
        np.all(grid[0, :] == border)
        and np.all(grid[-1, :] == border)
        and np.all(grid[:, 0] == border)
        and np.all(grid[:, -1] == border)
    ):
        return False

    interior = grid[1:-1, 1:-1]
    bg = _most_common(interior[interior != border] if np.any(interior != border) else interior)
    counts = Counter(int(v) for v in interior.ravel())
    if counts[bg] != max(counts.values()):
        return False
    if sum(1 for value in counts.values() if value == counts[bg]) != 1:
        return False

    touching = 0
    projected: set[tuple[int, int]] = set()
    for comp in _components_by_color(grid, bg=bg, border=border):
        stats = _contact_stats(comp["coords"], grid.shape[0], grid.shape[1])
        positive_lengths = [length for length in stats.values() if length > 0]
        if not positive_lengths:
            continue
        if sum(length == max(positive_lengths) for length in positive_lengths) != 1:
            return False
        side = _select_side(stats)
        assert side is not None

        r0, c0, _r1, _c1 = comp["bbox"]
        local_mask = np.zeros((_r1 - r0 + 1, _c1 - c0 + 1), dtype=bool)
        for r, c in comp["coords"]:
            local_mask[int(r) - r0, int(c) - c0] = True
        proj = _projection_positions(local_mask, side, r0, c0)
        if proj & projected:
            return False
        projected.update(proj)
        touching += 1

    return touching >= 2


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 24)
    width = random.randint(10, 24)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[0, :] = BORDER_COLOR
    grid[-1, :] = BORDER_COLOR
    grid[:, 0] = BORDER_COLOR
    grid[:, -1] = BORDER_COLOR

    occupied = np.zeros((height, width), dtype=bool)
    reserved_border: set[tuple[int, int]] = set()

    max_objects = min(6, len(OBJECT_COLORS))
    n_touching = random.randint(2, min(4, max_objects))
    n_free = random.randint(0, min(2, max_objects - n_touching))
    colors = random.sample(OBJECT_COLORS, n_touching + n_free)

    sides_cycle = ["top", "bottom", "left", "right"]
    random.shuffle(sides_cycle)
    chosen_sides = [sides_cycle[i % 4] for i in range(n_touching)]
    random.shuffle(chosen_sides)

    touching_specs: list[tuple[str, str | None]] = []
    for side in chosen_sides:
        secondary = None
        if random.random() < 0.4:
            if side in ("top", "bottom"):
                secondary = random.choice(["left", "right"])
            else:
                secondary = random.choice(["top", "bottom"])
        touching_specs.append((side, secondary))

    placed_touching = 0
    placed_corner = 0
    for color, (side, secondary) in zip(colors[:n_touching], touching_specs):
        placed = _try_place_touching(
            grid=grid,
            occupied=occupied,
            reserved_border=reserved_border,
            color=color,
            side=side,
            secondary=secondary,
        )
        if placed is None:
            return None
        placed_touching += 1
        if secondary is not None:
            placed_corner += 1

    for color in colors[n_touching:]:
        placed = _try_place_free(grid=grid, occupied=occupied, color=color)
        if placed is None:
            return None

    if placed_touching < 2:
        return None

    # Keep total foreground small enough that background remains dominant.
    if np.count_nonzero(grid[1:-1, 1:-1] != BG_COLOR) >= ((height - 2) * (width - 2)) // 2:
        return None

    if not _validate_generated_grid(grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
