from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 9
MAX_GENERATION_ATTEMPTS = 300
SPRITE_ATTEMPTS = 120
PORTAL_ATTEMPTS = 120
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    mask = patch != int(bg)
    canvas[top : top + h, left : left + w][mask] = patch[mask]


def _find_frame_layout(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    colors = sorted(set(map(int, np.unique(g))) - {int(bg)})
    candidates: list[tuple[int, list[int], int]] = []

    for color in colors:
        cols = [c for c in range(w) if bool(np.all(g[:, c] == int(color)))]
        if not cols:
            continue
        widths = []
        prev = -1
        ok = True
        for col in cols:
            width = int(col - prev - 1)
            if width < 3:
                ok = False
                break
            widths.append(width)
            prev = int(col)
        last_width = int(w - prev - 1)
        if last_width < 3:
            ok = False
        widths.append(last_width)
        if ok and len(set(widths)) == 1:
            candidates.append((int(color), cols, int(widths[0])))

    assert len(candidates) == 1
    divider_color, divider_cols, frame_width = candidates[0]

    frames = []
    start = 0
    for col in divider_cols + [w]:
        frames.append((int(start), int(col - 1)))
        start = int(col + 1)

    assert all(int(r - l + 1) == frame_width for l, r in frames)
    assert len(frames) >= 2
    return int(divider_color), list(map(int, divider_cols)), frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Teleport each frame's sprite to the frame whose left portal matches the source right portal.
    The teleported sprite lands horizontally mirrored, preserving its row offset to the portal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, _w = g.shape
    bg = _background_color(g)
    divider_color, divider_cols, frame_bounds = _find_frame_layout(g, bg)
    frame_width = int(frame_bounds[0][1] - frame_bounds[0][0] + 1)

    frames = []
    for left_col, right_col in frame_bounds:
        frame = g[:, left_col : right_col + 1]
        left_rows = np.flatnonzero(frame[:, 0] != int(bg))
        right_rows = np.flatnonzero(frame[:, -1] != int(bg))
        assert left_rows.size == 1 and right_rows.size == 1

        left_row = int(left_rows[0])
        right_row = int(right_rows[0])
        left_color = int(frame[left_row, 0])
        right_color = int(frame[right_row, frame_width - 1])

        sprite_mask = frame != int(bg)
        sprite_mask[left_row, 0] = False
        sprite_mask[right_row, frame_width - 1] = False
        assert bool(sprite_mask.any())

        r0, c0, r1, c1 = _bbox(sprite_mask)
        assert 0 < c0 <= c1 < frame_width - 1
        patch = frame[r0 : r1 + 1, c0 : c1 + 1].copy()
        patch[~sprite_mask[r0 : r1 + 1, c0 : c1 + 1]] = int(bg)

        frames.append(
            {
                "grid_left": int(left_col),
                "grid_right": int(right_col),
                "left_row": int(left_row),
                "right_row": int(right_row),
                "left_color": int(left_color),
                "right_color": int(right_color),
                "top": int(r0),
                "left": int(c0),
                "patch": patch,
            }
        )

    left_dest = {frame["left_color"]: idx for idx, frame in enumerate(frames)}
    assert len(left_dest) == len(frames)

    out = np.full_like(g, int(bg))
    for col in divider_cols:
        out[:, int(col)] = int(divider_color)
    for frame in frames:
        out[int(frame["left_row"]), int(frame["grid_left"])] = int(frame["left_color"])
        out[int(frame["right_row"]), int(frame["grid_right"])] = int(frame["right_color"])

    for frame in frames:
        target = frames[int(left_dest[int(frame["right_color"])])]
        patch = np.fliplr(frame["patch"])
        ph, pw = patch.shape
        dest_top = int(frame["top"] + target["left_row"] - frame["right_row"])
        dest_left = int(target["grid_left"] + (frame_width - pw - frame["left"]))
        assert 0 <= dest_top and dest_top + ph <= h
        assert target["grid_left"] < dest_left
        assert dest_left + pw - 1 < target["grid_right"]
        _paste(out, patch, dest_top, dest_left, bg)

    return out


def _random_derangement(n: int) -> list[int]:
    perm = list(range(n))
    while True:
        random.shuffle(perm)
        if all(idx != perm[idx] for idx in range(n)):
            return perm[:]


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}

    while len(cells) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < height and 0 <= cc < width and (rr, cc) not in cells:
                    frontier.add((rr, cc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    for r, c in cells:
        mask[int(r), int(c)] = True
    return _crop_mask(mask)


def _make_sprite(max_height: int, max_width: int, palette: list[int]) -> np.ndarray | None:
    if max_height < 2 or max_width < 2 or len(palette) < 2:
        return None

    for _ in range(SPRITE_ATTEMPTS):
        height = random.randint(2, max_height)
        width = random.randint(2, max_width)
        area_hi = int(height * width)
        if area_hi >= 5:
            area_hi -= 1
        area_lo = min(area_hi, max(4, (height * width + 1) // 2))
        if area_lo > area_hi:
            continue

        mask = _grow_connected_mask(height, width, random.randint(area_lo, area_hi))
        if mask is None:
            continue
        sh, sw = mask.shape
        if sh < 2 or sw < 2 or sh > max_height or sw > max_width:
            continue

        patch = np.full((sh, sw), BG, dtype=int)
        color_count = random.randint(2, min(3, len(palette)))
        colors = random.sample(palette, color_count)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        random.shuffle(coords)
        for idx, (r, c) in enumerate(coords):
            color = colors[idx] if idx < color_count else random.choice(colors)
            patch[int(r), int(c)] = int(color)

        if len(set(map(int, patch[mask].tolist()))) < 2:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch

    return None


def _frame_bounds(frame_count: int, frame_width: int) -> tuple[int, list[int], list[tuple[int, int]]]:
    total_width = int(frame_count * frame_width + (frame_count - 1))
    divider_cols = [int((idx + 1) * frame_width + idx) for idx in range(frame_count - 1)]
    bounds = []
    start = 0
    for idx in range(frame_count):
        bounds.append((int(start), int(start + frame_width - 1)))
        start += int(frame_width + (1 if idx < frame_count - 1 else 0))
    return total_width, divider_cols, bounds


def _sample_portals(frame_count: int, height: int, perm: list[int]) -> tuple[list[int], list[int], list[int]] | None:
    for _ in range(PORTAL_ATTEMPTS):
        left_rows = [random.randint(1, height - 2) for _ in range(frame_count)]
        right_rows = [random.randint(1, height - 2) for _ in range(frame_count)]
        shifts = [int(left_rows[perm[idx]] - right_rows[idx]) for idx in range(frame_count)]
        if any(abs(shift) > height - 4 for shift in shifts):
            continue
        if not any(shift != 0 for shift in shifts):
            continue
        return left_rows, right_rows, shifts
    return None


def _render_case(
    height: int,
    frame_width: int,
    bounds: list[tuple[int, int]],
    divider_cols: list[int],
    left_rows: list[int],
    right_rows: list[int],
    left_colors: list[int],
    right_colors: list[int],
    specs: list[dict[str, object]],
    perm: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    total_width = int(bounds[-1][1] + 1)
    inp = np.full((height, total_width), BG, dtype=int)
    out = np.full_like(inp, BG)

    for col in divider_cols:
        inp[:, int(col)] = DIVIDER
        out[:, int(col)] = DIVIDER

    for idx, (frame_left, frame_right) in enumerate(bounds):
        inp[int(left_rows[idx]), int(frame_left)] = int(left_colors[idx])
        inp[int(right_rows[idx]), int(frame_right)] = int(right_colors[idx])
        out[int(left_rows[idx]), int(frame_left)] = int(left_colors[idx])
        out[int(right_rows[idx]), int(frame_right)] = int(right_colors[idx])

        patch = np.asarray(specs[idx]["patch"], dtype=int)
        top = int(specs[idx]["top"])
        left = int(specs[idx]["left"])
        _paste(inp, patch, top, int(frame_left + left), BG)

    for idx, spec in enumerate(specs):
        patch = np.asarray(spec["patch"], dtype=int)
        top = int(spec["top"])
        left = int(spec["left"])
        target = int(perm[idx])
        target_left = int(bounds[target][0])
        mirrored = np.fliplr(patch)
        dest_top = int(top + left_rows[target] - right_rows[idx])
        dest_left = int(target_left + (frame_width - patch.shape[1] - left))
        _paste(out, mirrored, dest_top, dest_left, BG)

    return inp, out


def _generate_attempt() -> np.ndarray | None:
    frame_count = random.choice((3, 4))
    max_frame_width = min(8, (30 - (frame_count - 1)) // frame_count)
    frame_width = random.randint(6, max_frame_width)
    height = random.randint(9, 15)

    all_colors = [color for color in range(1, 9)]
    portal_colors = random.sample(all_colors, frame_count)
    sprite_palette = [color for color in all_colors if color not in portal_colors]
    if len(sprite_palette) < 2:
        return None

    left_colors = portal_colors[:]
    random.shuffle(left_colors)
    perm = _random_derangement(frame_count)
    right_colors = [int(left_colors[perm[idx]]) for idx in range(frame_count)]

    sampled = _sample_portals(frame_count, height, perm)
    if sampled is None:
        return None
    left_rows, right_rows, shifts = sampled

    _total_width, divider_cols, bounds = _frame_bounds(frame_count, frame_width)
    specs: list[dict[str, object]] = []
    for idx in range(frame_count):
        max_height = min(5, height - 2 - abs(shifts[idx]))
        sprite = _make_sprite(max_height, frame_width - 2, sprite_palette)
        if sprite is None:
            return None

        sh, sw = sprite.shape
        top_lo = max(1, 1 - shifts[idx])
        top_hi = min(height - sh - 1, height - sh - 1 - shifts[idx])
        if top_lo > top_hi:
            return None

        left = random.randint(1, frame_width - sw - 1)
        top = random.randint(top_lo, top_hi)
        specs.append({"patch": sprite, "top": int(top), "left": int(left)})

    inp, expected = _render_case(
        height,
        frame_width,
        bounds,
        divider_cols,
        left_rows,
        right_rows,
        left_colors,
        right_colors,
        specs,
        perm,
    )

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid teleport-frame puzzle")
