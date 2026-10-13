from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
BORDER_COLORS = (1, 2, 3, 4)
FILL_COLORS = (5, 6, 7, 8, 9)

GRID_H_RANGE = (13, 16)
GRID_W_RANGE = (14, 18)
SPRITE_COUNT_OPTIONS = (3, 4)
CONTROL_COUNT_WEIGHTS = ((1, 2), (2, 3))
SPRITE_H_RANGE = (4, 6)
SPRITE_W_RANGE = (4, 6)
MAX_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_mask(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    n = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            n += 1
            q = deque([(r, c)])
            labels[r, c] = n
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = n
                        q.append((nr, nc))
    return labels, n


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    labels, n = _label_mask(mask)
    return [np.argwhere(labels == k) for k in range(1, n + 1)]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert w >= 2, "expected palette to occupy the rightmost two columns"
    good_rows = [r for r in range(h) if int(grid[r, w - 2]) != bg and int(grid[r, w - 1]) != bg]
    assert good_rows, "expected a right-edge palette"

    runs: list[tuple[int, int]] = []
    start = good_rows[0]
    prev = good_rows[0]
    for r in good_rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = prev = r
    runs.append((start, prev))
    r0, r1 = max(runs, key=lambda item: item[1] - item[0])

    palette_mask = np.zeros_like(grid, dtype=bool)
    palette_mask[r0 : r1 + 1, w - 2 : w] = True

    mapping: dict[int, int] = {}
    for r in range(r0, r1 + 1):
        key = int(grid[r, w - 2])
        val = int(grid[r, w - 1])
        mapping[key] = val

    assert len(mapping) == (r1 - r0 + 1), "palette keys must be unique"
    return mapping, palette_mask


def _find_sprites(grid: np.ndarray, palette_mask: np.ndarray, bg: int) -> list[dict]:
    body_mask = (grid != bg) & (~palette_mask)
    body_mask[0, :] = False
    sprites: list[dict] = []
    for coords in _components_mask(body_mask):
        r0, c0, r1, c1 = _bbox(coords)
        color = int(grid[coords[0, 0], coords[0, 1]])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(patch[0, :] == color):
            continue
        if not np.all(patch[-1, :] == color):
            continue
        if not np.all(patch[:, 0] == color):
            continue
        if not np.all(patch[:, -1] == color):
            continue
        if np.any(patch[1:-1, 1:-1] != bg):
            continue
        sprites.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert 3 <= len(sprites) <= 4, "expected 3-4 hollow sprites"
    return sorted(sprites, key=lambda item: item["bbox"])


def _apply_rays(grid: np.ndarray, bg: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    control_cols = [int(c) for c in np.flatnonzero(out[0] != bg)]
    assert 1 <= len(control_cols) <= 2, "expected 1-2 top-row controls"
    for c in control_cols:
        color = int(out[0, c])
        for r in range(1, out.shape[0]):
            val = int(out[r, c])
            if val == bg:
                continue
            out[r, c] = bg if val == color else color
    return out


def _enclosed_regions_in_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    passable = crop == bg
    outside = np.zeros_like(passable, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    h, w = passable.shape
    for rr in range(h):
        for cc in (0, w - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))
    for cc in range(w):
        for rr in (0, h - 1):
            if passable[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                q.append((rr, cc))

    while q:
        rr, cc = q.popleft()
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    enclosed = passable & (~outside)
    return [coords + np.array([r0, c0]) for coords in _components_mask(enclosed)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire top-row toggle rays that may break hollow sprites, then fill only the sprites that remain enclosed.

    Input structure:
    - The background is black.
    - The middle/lower area contains 3-4 hollow single-color rectangular sprites with distinct border colors.
    - The right edge contains a 2-column palette legend whose left column gives border colors and right column gives fill colors.
    - The top row contains 1-2 control pixels whose columns define downward rays.

    Rule:
    - Read the right-edge legend as a border-color -> fill-color mapping.
    - Process the top-row controls from left to right. Along each control column, every non-background cell toggles:
      same-color cells are erased to background, while other non-background cells are recolored to the control color.
    - After all rays finish, inspect each original sprite inside its original bounding box.
      If the sprite's interior background is still enclosed by non-background border cells, fill that enclosed region using the palette color mapped from the sprite's original border color.
    - If a ray opened the border to the outside, leave that sprite's interior as background.
    - Finally clear the palette legend area to background while keeping the control pixels.

    Key insight:
    The palette alone is not enough: a sprite is fill-eligible only if the ray stage leaves its original hollow region geometrically enclosed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, palette_mask = _detect_palette(g, bg)
    sprites = _find_sprites(g, palette_mask, bg)

    out = _apply_rays(g, bg)
    for sprite in sprites:
        regions = _enclosed_regions_in_bbox(out, sprite["bbox"], bg)
        if not regions:
            continue
        fill_color = int(mapping[sprite["color"]])
        for coords in regions:
            out[coords[:, 0], coords[:, 1]] = fill_color

    out[palette_mask] = bg
    return out


def _paint_sprite(grid: np.ndarray, spec: dict) -> None:
    r0, c0, r1, c1 = spec["bbox"]
    color = int(spec["color"])
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _control_count() -> int:
    choices = [item[0] for item in CONTROL_COUNT_WEIGHTS]
    weights = [item[1] for item in CONTROL_COUNT_WEIGHTS]
    return int(random.choices(choices, weights=weights, k=1)[0])


def _find_positions(
    h: int,
    w: int,
    occupied: np.ndarray,
    *,
    include_cols: tuple[int, ...] = (),
    exclude_cols: tuple[int, ...] = (),
    row_min: int = 2,
    row_max: int | None = None,
) -> list[tuple[int, int]]:
    H, W = occupied.shape
    if row_max is None:
        row_max = H - h - 2
    candidates = []
    for r0 in range(row_min, row_max + 1):
        r1 = r0 + h - 1
        if r1 >= H - 1:
            continue
        for c0 in range(1, W - 2 - w + 1):
            c1 = c0 + w - 1
            if any(not (c0 <= col <= c1) for col in include_cols):
                continue
            if any(c0 <= col <= c1 for col in exclude_cols):
                continue
            rr0 = max(0, r0 - 1)
            rr1 = min(H, r1 + 2)
            cc0 = max(0, c0 - 1)
            cc1 = min(W, c1 + 2)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    return candidates


def _place_spec(
    occupied: np.ndarray,
    *,
    color: int,
    include_cols: tuple[int, ...] = (),
    exclude_cols: tuple[int, ...] = (),
    row_min: int = 2,
    row_max: int | None = None,
    min_width: int = SPRITE_W_RANGE[0],
) -> dict | None:
    widths = list(range(max(min_width, SPRITE_W_RANGE[0]), SPRITE_W_RANGE[1] + 1))
    heights = list(range(SPRITE_H_RANGE[0], SPRITE_H_RANGE[1] + 1))
    random.shuffle(widths)
    random.shuffle(heights)
    for w in widths:
        for h in heights:
            candidates = _find_positions(
                h,
                w,
                occupied,
                include_cols=include_cols,
                exclude_cols=exclude_cols,
                row_min=row_min,
                row_max=row_max,
            )
            if not candidates:
                continue
            r0, c0 = random.choice(candidates)
            return {
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1)),
                "fill": None,
            }
    return None


def _mark_occupied(occupied: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    occupied[max(0, r0 - 1) : min(occupied.shape[0], r1 + 2), max(0, c0 - 1) : min(occupied.shape[1], c1 + 2)] = True


def _semantic_roles(sprites: list[dict], controls: list[tuple[int, int]]) -> dict[str, list[int]]:
    broken = []
    recolored_only = []
    untouched = []
    for sprite in sprites:
        _r0, c0, _r1, c1 = sprite["bbox"]
        same_hit = any(col == sprite["color"] and c0 <= c <= c1 for c, col in controls)
        diff_hit = any(col != sprite["color"] and c0 <= c <= c1 for c, col in controls)
        if same_hit:
            broken.append(sprite["color"])
        elif diff_hit:
            recolored_only.append(sprite["color"])
        else:
            untouched.append(sprite["color"])
    return {
        "broken": broken,
        "recolored_only": recolored_only,
        "untouched": untouched,
    }


def _expected_from_construction(inp: np.ndarray, sprites: list[dict], palette_rows: tuple[int, int], controls: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    for c, color in sorted(controls):
        for r in range(1, out.shape[0]):
            val = int(out[r, c])
            if val == BG:
                continue
            out[r, c] = BG if val == color else color

    for sprite in sprites:
        r0, c0, r1, c1 = sprite["bbox"]
        border_cells = set()
        for c in range(c0, c1 + 1):
            border_cells.add((r0, c))
            border_cells.add((r1, c))
        for r in range(r0, r1 + 1):
            border_cells.add((r, c0))
            border_cells.add((r, c1))
        closed = all(int(out[r, c]) != BG for r, c in border_cells)
        if closed:
            out[r0 + 1 : r1, c0 + 1 : c1] = int(sprite["fill"])

    pr0, pr1 = palette_rows
    out[pr0 : pr1 + 1, -2:] = BG
    return out


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    sprite_count = random.choice(SPRITE_COUNT_OPTIONS)
    control_count = _control_count()
    if control_count == 2 and W < 15:
        control_count = 1

    border_colors = random.sample(BORDER_COLORS, sprite_count)
    fill_colors = random.sample(FILL_COLORS, sprite_count)
    mapping = dict(zip(border_colors, fill_colors))

    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    palette_start = random.randint(2, H - sprite_count - 1)
    palette_end = palette_start + sprite_count - 1
    for idx, border_color in enumerate(border_colors):
        row = palette_start + idx
        grid[row, W - 2] = border_color
        grid[row, W - 1] = mapping[border_color]
    occupied[max(0, palette_start - 1) : min(H, palette_end + 2), max(0, W - 3) : W] = True

    if control_count == 1:
        c1 = random.randint(2, W - 6)
        control_cols = (c1,)
    else:
        c1 = random.randint(2, W - 8)
        c2 = c1 + random.randint(2, 3)
        control_cols = (c1, c2)

    broken_a_color = border_colors[0]
    recolored_b_color = border_colors[1]
    untouched_c_color = border_colors[2]

    spec_b = _place_spec(
        occupied,
        color=recolored_b_color,
        include_cols=control_cols,
        min_width=max(SPRITE_W_RANGE[0], control_cols[-1] - control_cols[0] + 1),
        row_min=5,
    )
    if spec_b is None:
        return None
    spec_b["fill"] = mapping[recolored_b_color]
    _mark_occupied(occupied, spec_b["bbox"])

    spec_a = _place_spec(
        occupied,
        color=broken_a_color,
        include_cols=(c1,),
        exclude_cols=control_cols[1:],
        row_min=2,
        row_max=spec_b["bbox"][0] - SPRITE_H_RANGE[0] - 1,
    )
    if spec_a is None:
        return None
    spec_a["fill"] = mapping[broken_a_color]
    _mark_occupied(occupied, spec_a["bbox"])

    spec_c = _place_spec(
        occupied,
        color=untouched_c_color,
        exclude_cols=control_cols,
        row_min=2,
    )
    if spec_c is None:
        return None
    spec_c["fill"] = mapping[untouched_c_color]
    _mark_occupied(occupied, spec_c["bbox"])

    sprites = [spec_a, spec_b, spec_c]

    extra_color = None
    if sprite_count == 4:
        extra_color = border_colors[3]
        if control_count == 2:
            spec_d = _place_spec(
                occupied,
                color=extra_color,
                include_cols=(control_cols[1],),
                exclude_cols=(control_cols[0],),
                row_min=2,
            )
        else:
            mode = random.choice(("untouched", "recolored"))
            if mode == "untouched":
                spec_d = _place_spec(
                    occupied,
                    color=extra_color,
                    exclude_cols=control_cols,
                    row_min=2,
                )
            else:
                spec_d = _place_spec(
                    occupied,
                    color=extra_color,
                    include_cols=control_cols,
                    row_min=2,
                )
        if spec_d is None:
            return None
        spec_d["fill"] = mapping[extra_color]
        _mark_occupied(occupied, spec_d["bbox"])
        sprites.append(spec_d)

    controls: list[tuple[int, int]] = [(c1, broken_a_color)]
    if control_count == 2:
        second_color = extra_color if extra_color is not None else untouched_c_color
        controls.append((control_cols[1], int(second_color)))

    for c, color in controls:
        grid[0, c] = color

    for spec in sprites:
        _paint_sprite(grid, spec)

    roles = _semantic_roles(sprites, controls)
    if not roles["broken"] or not roles["recolored_only"] or not roles["untouched"]:
        return None

    expected = _expected_from_construction(grid, sprites, (palette_start, palette_end), controls)
    if np.array_equal(expected, grid):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if len(np.unique(actual)) < 3:
        return None
    if len({int(spec["fill"]) for spec in sprites}) < 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
