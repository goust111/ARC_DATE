from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
FREE_COLORS = [1, 2, 3, 4, 6, 7, 8]
GEN_ATTEMPTS = 400


def _mode_color(values: np.ndarray) -> int:
    vals, cnts = np.unique(values.astype(int), return_counts=True)
    return int(vals[int(cnts.argmax())])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one True cell"
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _paint_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    H, W = canvas.shape
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]):
                continue
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < H and 0 <= cc < W:
                canvas[rr, cc] = int(color)


def _sprite_patch(
    shape_mask: np.ndarray,
    fill: int,
    body: int,
    accent: int,
    stripe_axis: str,
    stripe_index: int,
) -> np.ndarray:
    patch = np.full(shape_mask.shape, int(fill), dtype=int)
    patch[shape_mask] = int(body)
    if stripe_axis == "row":
        cols = np.where(shape_mask[int(stripe_index)])[0]
        patch[int(stripe_index), cols] = int(accent)
    else:
        rows = np.where(shape_mask[:, int(stripe_index)])[0]
        patch[rows, int(stripe_index)] = int(accent)
    return patch


def _paste_patch(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, fill: int) -> None:
    h, w = patch.shape
    H, W = canvas.shape
    for r in range(h):
        for c in range(w):
            rr = int(top + r)
            cc = int(left + c)
            if 0 <= rr < H and 0 <= cc < W and int(patch[r, c]) != int(fill):
                canvas[rr, cc] = int(patch[r, c])


def _extract_panels(grid: np.ndarray) -> list[np.ndarray]:
    grey_mask = grid == int(GREY)
    r0, c0, r1, c1 = _bbox(grey_mask)
    strip = grid[r0 : r1 + 1, c0 : c1 + 1]

    sep_cols = [c for c in range(strip.shape[1]) if bool(np.all(strip[:, c] == int(GREY)))]
    assert len(sep_cols) >= 5, "expected at least four input panels"

    gaps = [sep_cols[i + 1] - sep_cols[i] for i in range(len(sep_cols) - 1)]
    assert len(set(gaps)) == 1, "panel widths must match"

    return [strip[:, sep_cols[i] : sep_cols[i + 1] + 1].copy() for i in range(len(sep_cols) - 1)]


def _analyze_panel(panel: np.ndarray, fill: int) -> dict[str, object]:
    interior = panel[1:-1, 1:-1]
    sprite_mask = interior != int(fill)
    sr0, sc0, sr1, sc1 = _bbox(sprite_mask)
    patch = interior[sr0 : sr1 + 1, sc0 : sc1 + 1]
    shape_mask = patch != int(fill)

    colors = [int(c) for c in np.unique(patch[shape_mask])]
    assert len(colors) >= 2, "sprite must contain body and accent colors"

    candidates: list[tuple[int, str, int, int]] = []
    for color in colors:
        coords = np.argwhere(patch == int(color))
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            rr = int(rows[0])
            row_mask = shape_mask[rr]
            if bool(row_mask.any()) and bool(np.all(patch[rr, row_mask] == int(color))):
                candidates.append((int(coords.shape[0]), "row", rr, int(color)))
        if len(cols) == 1:
            cc = int(cols[0])
            col_mask = shape_mask[:, cc]
            if bool(col_mask.any()) and bool(np.all(patch[col_mask, cc] == int(color))):
                candidates.append((int(coords.shape[0]), "col", cc, int(color)))

    assert candidates, "expected a single moving row or column color"
    candidates.sort(key=lambda item: (item[0], 0 if item[1] == "row" else 1, item[2], item[3]))
    _count, stripe_axis, stripe_index, accent = candidates[0]

    body = _mode_color(patch[shape_mask])
    return {
        "pos": (int(sr0), int(sc0)),
        "shape_mask": shape_mask,
        "body": int(body),
        "accent": int(accent),
        "stripe_axis": stripe_axis,
        "stripe_index": int(stripe_index),
        "panel_shape": panel.shape,
    }


def _render_output_panel(
    panel_shape: tuple[int, int],
    fill: int,
    shape_mask: np.ndarray,
    body: int,
    accent: int,
    stripe_axis: str,
    stripe_index: int,
    trail_positions: list[tuple[int, int]],
    next_pos: tuple[int, int],
) -> np.ndarray:
    ph, pw = map(int, panel_shape)
    out = np.full((ph, pw), int(fill), dtype=int)
    out[0, :] = int(GREY)
    out[-1, :] = int(GREY)
    out[:, 0] = int(GREY)
    out[:, -1] = int(GREY)

    for r, c in trail_positions:
        _paint_mask(out, shape_mask, 1 + int(r), 1 + int(c), int(MAROON))

    patch = _sprite_patch(shape_mask, fill, body, accent, stripe_axis, stripe_index)
    _paste_patch(out, patch, 1 + int(next_pos[0]), 1 + int(next_pos[1]), fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Predict the next sprite frame from a row of border-shared panels.

    Each panel shows the same rectangular sprite translated by a fixed vector.
    Inside the sprite, one colored row or column shifts cyclically by one step
    between frames. The output is a single next-frame panel whose earlier sprite
    footprints are recolored maroon.
    """
    g = np.asarray(grid, dtype=int)
    panels = _extract_panels(g)
    fill = _mode_color(np.concatenate([p[p != int(GREY)] for p in panels]))

    infos = [_analyze_panel(panel, fill) for panel in panels]
    shape_mask = np.asarray(infos[0]["shape_mask"], dtype=bool)
    panel_shape = tuple(map(int, infos[0]["panel_shape"]))
    body = int(infos[0]["body"])
    accent = int(infos[0]["accent"])
    stripe_axis = str(infos[0]["stripe_axis"])

    positions = [tuple(map(int, info["pos"])) for info in infos]
    dr = int(positions[1][0] - positions[0][0])
    dc = int(positions[1][1] - positions[0][1])
    assert all(
        int(positions[i + 1][0] - positions[i][0]) == dr and int(positions[i + 1][1] - positions[i][1]) == dc
        for i in range(len(positions) - 1)
    ), "sprite translation must be constant"

    stripe_indices = [int(info["stripe_index"]) for info in infos]
    period = int(shape_mask.shape[0] if stripe_axis == "row" else shape_mask.shape[1])
    phase_step = int((stripe_indices[1] - stripe_indices[0]) % period)
    assert all(
        int((stripe_indices[i + 1] - stripe_indices[i]) % period) == phase_step
        for i in range(len(stripe_indices) - 1)
    ), "stripe motion must be constant"

    next_pos = (int(positions[-1][0] + dr), int(positions[-1][1] + dc))
    next_stripe = int((stripe_indices[-1] + phase_step) % period)

    return _render_output_panel(
        panel_shape=panel_shape,
        fill=fill,
        shape_mask=shape_mask,
        body=body,
        accent=accent,
        stripe_axis=stripe_axis,
        stripe_index=next_stripe,
        trail_positions=positions,
        next_pos=next_pos,
    )


def _position_range(length: int, sprite_length: int, delta: int, frame_count: int) -> tuple[int, int] | None:
    low = max(0, -int(delta) * int(frame_count))
    high = int(length) - int(sprite_length) - max(0, int(delta) * int(frame_count))
    if low > high:
        return None
    return int(low), int(high)


def _build_input_and_expected(
    inner_h: int,
    inner_w: int,
    frame_count: int,
    margins: tuple[int, int, int, int],
    fill: int,
    body: int,
    accent: int,
    shape_mask: np.ndarray,
    stripe_axis: str,
    stripe_start: int,
    stripe_step: int,
    motion: tuple[int, int],
    start_pos: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    panel_h = int(inner_h + 2)
    panel_w = int(inner_w + 2)
    strip_w = int(frame_count * (inner_w + 1) + 1)
    strip = np.full((panel_h, strip_w), int(GREY), dtype=int)

    dr, dc = map(int, motion)
    sr0, sc0 = map(int, start_pos)
    positions = [(int(sr0 + i * dr), int(sc0 + i * dc)) for i in range(frame_count)]

    period = int(shape_mask.shape[0] if stripe_axis == "row" else shape_mask.shape[1])
    for i in range(frame_count):
        left_sep = int(i * (inner_w + 1))
        right_sep = int(left_sep + inner_w + 1)
        strip[1:-1, left_sep + 1 : right_sep] = int(fill)
        stripe_index = int((stripe_start + i * stripe_step) % period)
        patch = _sprite_patch(shape_mask, fill, body, accent, stripe_axis, stripe_index)
        _paste_patch(strip, patch, 1 + positions[i][0], left_sep + 1 + positions[i][1], fill)

    mt, mb, ml, mr = map(int, margins)
    grid = np.full((panel_h + mt + mb, strip_w + ml + mr), int(BLACK), dtype=int)
    grid[mt : mt + panel_h, ml : ml + strip_w] = strip

    next_pos = (int(sr0 + frame_count * dr), int(sc0 + frame_count * dc))
    next_stripe = int((stripe_start + frame_count * stripe_step) % period)
    expected = _render_output_panel(
        panel_shape=(panel_h, panel_w),
        fill=fill,
        shape_mask=shape_mask,
        body=body,
        accent=accent,
        stripe_axis=stripe_axis,
        stripe_index=next_stripe,
        trail_positions=positions,
        next_pos=next_pos,
    )
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.randint(4, 6)
    stripe_axis = random.choice(["row", "col"])
    stripe_step = random.choice([-1, 1])

    unit_dirs = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    dir_r, dir_c = random.choice(unit_dirs)
    move_scale = random.choice([1, 2])
    motion = (int(dir_r * move_scale), int(dir_c * move_scale))

    inner_h = random.randint(10, 15)
    inner_w = random.randint(10, 15)

    max_sprite_h = int(inner_h - abs(motion[0]) * frame_count)
    max_sprite_w = int(inner_w - abs(motion[1]) * frame_count)
    if max_sprite_h < 3 or max_sprite_w < 3:
        return None

    sprite_h = random.randint(3, min(6, max_sprite_h))
    sprite_w = random.randint(3, min(6, max_sprite_w))
    shape_mask = np.ones((sprite_h, sprite_w), dtype=bool)

    row_range = _position_range(inner_h, sprite_h, motion[0], frame_count)
    col_range = _position_range(inner_w, sprite_w, motion[1], frame_count)
    if row_range is None or col_range is None:
        return None

    start_pos = (
        random.randint(row_range[0], row_range[1]),
        random.randint(col_range[0], col_range[1]),
    )

    period = int(sprite_h if stripe_axis == "row" else sprite_w)
    stripe_start = random.randint(0, period - 1)
    fill, body, accent = random.sample(FREE_COLORS, 3)
    margins = (
        random.randint(1, 2),
        random.randint(1, 2),
        random.randint(1, 3),
        random.randint(1, 3),
    )

    grid, expected = _build_input_and_expected(
        inner_h=inner_h,
        inner_w=inner_w,
        frame_count=frame_count,
        margins=margins,
        fill=fill,
        body=body,
        accent=accent,
        shape_mask=shape_mask,
        stripe_axis=stripe_axis,
        stripe_start=stripe_start,
        stripe_step=stripe_step,
        motion=motion,
        start_pos=start_pos,
    )

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
