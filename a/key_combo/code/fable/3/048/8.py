from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

try:
    from arc_utils import bbox, crop_bbox, get_bg, paste
except Exception:  # pragma: no cover - local fallback keeps SourceFileLoader robust.
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        return int(r0), int(c0), int(r1), int(c1)

    def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = box
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def paste(
        dst: np.ndarray,
        patch: np.ndarray,
        r0: int,
        c0: int,
        *,
        transparent: int | None = None,
    ) -> np.ndarray:
        h, w = patch.shape
        view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
        mask = np.ones(patch.shape, dtype=bool)
        if transparent is not None:
            mask &= patch != int(transparent)
        view[mask] = patch[mask]
        return dst


BACKGROUND = 0
RED = 2
SPRITE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 600
PLACEMENT_GAP = 1


def _projection(crop: np.ndarray, bg: int) -> tuple[tuple[bool, ...], tuple[bool, ...]]:
    mask = crop != int(bg)
    row_pat = tuple(bool(x) for x in mask.any(axis=1))
    col_pat = tuple(bool(x) for x in mask.any(axis=0))
    return row_pat, col_pat


def _bar_pattern(line: np.ndarray, color: int) -> tuple[int, int, tuple[bool, ...]]:
    hits = np.flatnonzero(line == int(color))
    assert hits.size >= 2, "each silhouette bar must have a visible span"
    start = int(hits[0])
    end = int(hits[-1])
    segment = line[start : end + 1]
    assert int(segment[0]) == int(color) and int(segment[-1]) == int(color)
    assert np.all((segment == int(color)) | (segment == BACKGROUND)), "bar span may only contain red marks and black gaps"
    pattern = tuple(bool(x) for x in (segment == int(color)))
    assert any(not x for x in pattern[1:-1]), "silhouette pattern must contain an internal gap"
    return start, end, pattern


def _sprite_infos(g: np.ndarray, bg: int) -> list[dict]:
    infos: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg), RED}):
        mask = g == int(color)
        if not mask.any():
            continue
        box = bbox(mask)
        assert box is not None
        crop = crop_bbox(g, box)
        row_pat, col_pat = _projection(crop, bg)
        assert row_pat[0] and row_pat[-1] and col_pat[0] and col_pat[-1]
        assert any(not x for x in row_pat[1:-1]) or any(not x for x in col_pat[1:-1])
        infos.append(
            {
                "color": int(color),
                "bbox": box,
                "crop": crop,
                "row_pat": row_pat,
                "col_pat": col_pat,
            }
        )
    assert infos, "expected at least one sprite"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """Use the two red silhouette bars to select, orient, and relocate one sprite.

    Input structure:
    - A black background contains several uniquely colored sprites away from the
      bottom row and right column; their bounding boxes include empty rows or
      columns, so their row and column occupancy silhouettes have visible gaps.
    - The bottom edge holds one red gap-pattern bar whose span has the target
      sprite's column silhouette, possibly read in reverse.
    - The right edge holds one red gap-pattern bar whose span has the target
      sprite's row silhouette.

    Rule:
    - Read the right-edge red span as a row-occupancy pattern and choose the
      unique sprite with the same row silhouette.
    - Compare that sprite's column silhouette with the bottom-edge red span:
      a forward match keeps the crop unchanged, while a reversed match mirrors
      the crop left-to-right.
    - Clear all sprites and both red bars, then paste the chosen oriented crop
      into the rectangle whose rows are the right bar's span and whose columns
      are the bottom bar's span.

    Key insight:
    - The two red bars are projections of one hidden placement, not objects to
      copy; the right bar identifies the sprite and the bottom bar determines
      whether the selected sprite must be read normally or mirrored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert max(g.shape) <= 30, "ARC grids must not exceed 30 in either dimension"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected a black background"
    height, width = map(int, g.shape)
    assert height >= 5 and width >= 5

    bottom_start, bottom_end, bottom_pat = _bar_pattern(g[-1, :], RED)
    right_start, right_end, right_pat = _bar_pattern(g[:, -1], RED)
    assert bottom_end < width - 1, "bottom bar must leave the right edge column separate"
    assert right_end < height - 1, "right bar must leave the bottom edge row separate"

    red_mask = g == RED
    expected_red = np.zeros_like(red_mask)
    expected_red[-1, bottom_start : bottom_end + 1] = np.array(bottom_pat, dtype=bool)
    expected_red[right_start : right_end + 1, -1] = np.array(right_pat, dtype=bool)
    assert np.array_equal(red_mask, expected_red), "all red cells must belong to the two silhouette bars"

    body = g.copy()
    body[red_mask] = bg
    sprites = _sprite_infos(body, bg)

    row_matches = [sprite for sprite in sprites if sprite["row_pat"] == right_pat]
    assert len(row_matches) == 1, "right silhouette must identify one sprite"
    sprite = row_matches[0]

    col_pat = sprite["col_pat"]
    forward = col_pat == bottom_pat
    reversed_read = tuple(reversed(col_pat)) == bottom_pat
    assert forward ^ reversed_read, "bottom silhouette must determine exactly one orientation"

    crop = sprite["crop"]
    oriented = np.fliplr(crop) if reversed_read else crop
    assert oriented.shape == (right_end - right_start + 1, bottom_end - bottom_start + 1)

    out = np.full_like(g, bg)
    paste(out, oriented, right_start, bottom_start, transparent=bg)
    return out


# --------------- generator ---------------


def _random_pattern(length: int, *, require_nonpal: bool, avoid: set[tuple[bool, ...]]) -> tuple[bool, ...] | None:
    for _ in range(200):
        bits = [True]
        bits.extend(bool(random.getrandbits(1)) for _ in range(length - 2))
        bits.append(True)
        pat = tuple(bits)
        if pat in avoid:
            continue
        if not any(not x for x in pat[1:-1]):
            continue
        if sum(pat) < 3:
            continue
        if require_nonpal and pat == tuple(reversed(pat)):
            continue
        return pat
    return None


def _make_mask(row_pat: tuple[bool, ...], col_pat: tuple[bool, ...]) -> np.ndarray | None:
    rows = [i for i, x in enumerate(row_pat) if x]
    cols = [i for i, x in enumerate(col_pat) if x]
    allowed = {(r, c) for r in rows for c in cols}
    for _ in range(100):
        cells: set[tuple[int, int]] = set()
        for r in rows:
            cells.add((r, random.choice(cols)))
        for c in cols:
            cells.add((random.choice(rows), c))

        add_prob = random.uniform(0.25, 0.55)
        for cell in allowed:
            if random.random() < add_prob:
                cells.add(cell)

        if len(cells) >= min(len(allowed) - 1, max(len(rows), len(cols)) + 1):
            mask = np.zeros((len(row_pat), len(col_pat)), dtype=bool)
            for r, c in cells:
                mask[r, c] = True
            if tuple(mask.any(axis=1)) == row_pat and tuple(mask.any(axis=0)) == col_pat:
                if int(mask.sum()) < len(allowed):
                    return mask
    return None


def _make_sprite(row_pat: tuple[bool, ...], col_pat: tuple[bool, ...], color: int) -> np.ndarray | None:
    mask = _make_mask(row_pat, col_pat)
    if mask is None:
        return None
    sprite = np.full(mask.shape, BACKGROUND, dtype=int)
    sprite[mask] = int(color)
    return sprite


def _expanded(box: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return r0 - gap, c0 - gap, r1 + gap, c1 + gap


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_sprites(
    grid: np.ndarray,
    sprites: list[np.ndarray],
    blocked: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    height, width = map(int, grid.shape)
    placed: list[tuple[int, int, int, int]] = []
    order = list(range(len(sprites)))
    random.shuffle(order)

    locations: dict[int, tuple[int, int, int, int]] = {}
    for idx in order:
        sprite = sprites[idx]
        h, w = map(int, sprite.shape)
        candidates = [
            (r, c)
            for r in range(0, height - 1 - h + 1)
            for c in range(0, width - 1 - w + 1)
        ]
        random.shuffle(candidates)
        for r, c in candidates:
            box = (r, c, r + h - 1, c + w - 1)
            exp = _expanded(box)
            if any(_overlap(exp, _expanded(old, 0)) for old in blocked + placed):
                continue
            paste(grid, sprite, r, c, transparent=BACKGROUND)
            placed.append(box)
            locations[idx] = box
            break
        else:
            return None

    return [locations[i] for i in range(len(sprites))]


def _draw_bars(
    grid: np.ndarray,
    row_pat: tuple[bool, ...],
    col_pat: tuple[bool, ...],
    dest_r: int,
    dest_c: int,
) -> None:
    for dr, mark in enumerate(row_pat):
        if mark:
            grid[dest_r + dr, -1] = RED
    for dc, mark in enumerate(col_pat):
        if mark:
            grid[-1, dest_c + dc] = RED


def _expected_output(shape: tuple[int, int], sprite: np.ndarray, dest_r: int, dest_c: int, mirrored: bool) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    patch = np.fliplr(sprite) if mirrored else sprite
    paste(out, patch, dest_r, dest_c, transparent=BACKGROUND)
    return out


def _analyze_generated_grid(grid: np.ndarray) -> tuple[int, bool]:
    bottom_start, _bottom_end, bottom_pat = _bar_pattern(grid[-1, :], RED)
    _right_start, _right_end, right_pat = _bar_pattern(grid[:, -1], RED)
    body = grid.copy()
    body[grid == RED] = BACKGROUND
    sprites = _sprite_infos(body, BACKGROUND)
    row_matches = [i for i, sprite in enumerate(sprites) if sprite["row_pat"] == right_pat]
    if len(row_matches) != 1:
        raise AssertionError("generated right bar must select one sprite")
    idx = row_matches[0]
    col_pat = sprites[idx]["col_pat"]
    forward = col_pat == bottom_pat
    mirrored = tuple(reversed(col_pat)) == bottom_pat
    if not (forward ^ mirrored):
        raise AssertionError("generated bottom bar must select one orientation")
    if bottom_start < 0:
        raise AssertionError("unreachable")
    return int(sprites[idx]["color"]), bool(mirrored)


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(3, min(5, len(SPRITE_COLORS)))
    colors = random.sample(SPRITE_COLORS, sprite_count)

    target_h = random.randint(5, 8)
    target_w = random.randint(5, 8)
    row_patterns: set[tuple[bool, ...]] = set()
    col_patterns: set[tuple[bool, ...]] = set()

    target_row = _random_pattern(target_h, require_nonpal=False, avoid=row_patterns)
    if target_row is None:
        return None
    row_patterns.add(target_row)
    target_col = _random_pattern(target_w, require_nonpal=True, avoid=col_patterns)
    if target_col is None:
        return None
    col_patterns.add(target_col)

    specs: list[tuple[tuple[bool, ...], tuple[bool, ...]]] = [(target_row, target_col)]

    same_h_row = _random_pattern(target_h, require_nonpal=False, avoid=row_patterns)
    if same_h_row is None:
        return None
    row_patterns.add(same_h_row)
    w2 = random.choice([w for w in range(5, 9) if w != target_w] or [target_w])
    same_h_col = _random_pattern(w2, require_nonpal=False, avoid=col_patterns)
    if same_h_col is None:
        return None
    col_patterns.add(same_h_col)
    specs.append((same_h_row, same_h_col))

    h3 = random.choice([h for h in range(5, 9) if h != target_h] or [target_h])
    same_w_row = _random_pattern(h3, require_nonpal=False, avoid=row_patterns)
    if same_w_row is None:
        return None
    row_patterns.add(same_w_row)
    same_w_col = _random_pattern(target_w, require_nonpal=False, avoid=col_patterns)
    if same_w_col is None:
        return None
    col_patterns.add(same_w_col)
    specs.append((same_w_row, same_w_col))

    while len(specs) < sprite_count:
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        row_pat = _random_pattern(h, require_nonpal=False, avoid=row_patterns)
        col_pat = _random_pattern(w, require_nonpal=False, avoid=col_patterns)
        if row_pat is None or col_pat is None:
            return None
        row_patterns.add(row_pat)
        col_patterns.add(col_pat)
        specs.append((row_pat, col_pat))

    sprites: list[np.ndarray] = []
    for (row_pat, col_pat), color in zip(specs, colors):
        sprite = _make_sprite(row_pat, col_pat, color)
        if sprite is None:
            return None
        sprites.append(sprite)

    if np.array_equal(sprites[0], np.fliplr(sprites[0])):
        return None

    height = random.randint(max(22, target_h + 11), 30)
    width = random.randint(max(22, target_w + 11), 30)
    dest_r = random.randint(1, height - 2 - target_h)
    dest_c = random.randint(1, width - 2 - target_w)
    dest_box = (dest_r, dest_c, dest_r + target_h - 1, dest_c + target_w - 1)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    placed = _place_sprites(grid, sprites, [dest_box])
    if placed is None:
        return None

    mirrored = bool(random.getrandbits(1))
    bottom_pat = tuple(reversed(target_col)) if mirrored else target_col
    _draw_bars(grid, target_row, bottom_pat, dest_r, dest_c)

    expected = _expected_output(grid.shape, sprites[0], dest_r, dest_c, mirrored)
    try:
        actual = main(grid)
        selected_color, detected_mirror = _analyze_generated_grid(grid)
    except Exception:
        return None

    if selected_color != colors[0] or detected_mirror != mirrored:
        return None
    if not np.array_equal(actual, expected):
        return None
    if not any(sprites[i].shape[0] == target_h and specs[i][0] != target_row for i in range(1, len(sprites))):
        return None
    if not any(sprites[i].shape[1] == target_w and specs[i][1] != target_col for i in range(1, len(sprites))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid double-silhouette input")
