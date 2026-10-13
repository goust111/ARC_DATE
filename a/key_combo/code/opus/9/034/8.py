from __future__ import annotations

import random

import numpy as np


BG = 0
SPRITE = 1
SEPARATOR = 5
INDICATOR_COLORS = (2, 3, 4, 6, 7, 8, 9)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(comp, dtype=int))
    return comps


def _flip_side(side: str) -> str:
    return "right" if side == "left" else "left"


def _split_panels(grid: np.ndarray) -> list[np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR)]
    assert len(separator_rows) == 3, "expected three separator rows"
    panel_h = separator_rows[0]
    assert panel_h > 0
    assert separator_rows == [panel_h, 2 * panel_h + 1, 3 * panel_h + 2]
    assert g.shape[0] == 4 * panel_h + 3
    panels = [
        g[0:panel_h, :],
        g[panel_h + 1 : 2 * panel_h + 1, :],
        g[2 * panel_h + 2 : 3 * panel_h + 2, :],
        g[3 * panel_h + 3 :, :],
    ]
    assert len({panel.shape for panel in panels}) == 1
    return panels


def _panel_info(panel: np.ndarray) -> dict[str, object]:
    blue_mask = panel == SPRITE
    comps = _components_mask(blue_mask)
    assert len(comps) == 1, "expected exactly one blue sprite"
    box = _bbox(blue_mask)
    assert box is not None
    r0, c0, r1, c1 = box
    shape = _crop_bbox(blue_mask.astype(int), box)
    indicator_coords = np.argwhere((panel != BG) & (panel != SPRITE) & (panel != SEPARATOR))
    assert indicator_coords.shape == (1, 2), "expected exactly one indicator pixel"
    ir, ic = map(int, indicator_coords[0])
    color = int(panel[ir, ic])
    if ic == c0 - 1:
        side = "left"
        assert shape[ir - r0, 0] == 1
    else:
        assert ic == c1 + 1, "indicator must sit on the left or right side of the sprite"
        side = "right"
        assert shape[ir - r0, -1] == 1
    return {
        "shape": shape,
        "top_left": (int(r0), int(c0)),
        "side": side,
        "row_offset": int(ir - r0),
        "indicator_color": color,
        "panel_shape": tuple(map(int, panel.shape)),
    }


def _analyze_sequence(panels: list[np.ndarray]) -> dict[str, object]:
    assert len(panels) == 4
    infos = [_panel_info(panel) for panel in panels]
    shapes = [info["shape"] for info in infos]
    positions = [info["top_left"] for info in infos]
    sides = [str(info["side"]) for info in infos]
    row_offsets = [int(info["row_offset"]) for info in infos]
    colors = [int(info["indicator_color"]) for info in infos]

    deltas = [
        (positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        for i in range(3)
    ]
    assert deltas[0] == deltas[1] == deltas[2], "sprite must move by a constant offset"
    dr, dc = deltas[0]
    assert (dr, dc) != (0, 0), "translation must be non-zero"

    assert np.array_equal(shapes[0], shapes[2]), "odd frames must share one orientation"
    assert np.array_equal(shapes[1], shapes[3]), "even frames must share the other orientation"
    assert np.array_equal(np.fliplr(shapes[0]), shapes[1]), "even frame must be the horizontal mirror of odd frame"
    assert not np.array_equal(shapes[0], np.fliplr(shapes[0])), "sprite must be horizontally asymmetric"

    assert sides[0] == sides[2]
    assert sides[1] == sides[3]
    assert sides[1] == _flip_side(sides[0]), "indicator side must alternate with the mirror state"
    assert len(set(row_offsets)) == 1, "indicator must stay attached at one sprite row"

    assert len(set(colors)) == 4, "use a visible four-color cycle"

    return {
        "infos": infos,
        "delta": (int(dr), int(dc)),
        "next_shape": shapes[0],
        "next_side": sides[0],
        "next_row_offset": row_offsets[0],
        "next_color": colors[0],
        "panel_shape": infos[0]["panel_shape"],
    }


def _render_panel(
    panel_shape: tuple[int, int],
    top_left: tuple[int, int],
    shape: np.ndarray,
    side: str,
    row_offset: int,
    indicator_color: int,
) -> np.ndarray:
    h, w = panel_shape
    out = np.zeros((h, w), dtype=int)
    r0, c0 = top_left
    sh, sw = shape.shape
    for rr, cc in np.argwhere(shape):
        out[r0 + int(rr), c0 + int(cc)] = SPRITE
    ir = r0 + int(row_offset)
    ic = c0 - 1 if side == "left" else c0 + sw
    out[ir, ic] = int(indicator_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the fifth animation panel by combining constant translation, alternating horizontal mirroring, and a four-step indicator-color cycle.

    Input structure:
    - Four equal black panels are stacked vertically and separated by full-width grey rows.
    - Each panel contains one 4-connected blue sprite and one non-blue single-pixel indicator attached on the sprite's left or right side.
    - The blue sprite is horizontally asymmetric so its mirrored and unmirrored states are visibly different.

    Rule:
    - Split the strip into its four panels and locate the blue sprite plus the single indicator in each panel.
    - Infer the constant per-frame translation from the sprite bounding-box top-left positions.
    - Verify that odd panels share one sprite orientation, even panels share the other, and the even orientation is the horizontal mirror of the odd orientation.
    - Track the indicator side alternation and the four observed indicator colors; the fifth panel repeats the odd-frame orientation/side and wraps the color cycle to the first color.

    Key insight:
    - The next frame is determined only by combining three independent periodic signals at once: motion, mirror parity, and indicator-color phase.
    """
    panels = _split_panels(np.asarray(grid, dtype=int))
    meta = _analyze_sequence(panels)
    last_top = meta["infos"][-1]["top_left"]
    dr, dc = meta["delta"]
    next_top = (int(last_top[0]) + int(dr), int(last_top[1]) + int(dc))
    return _render_panel(
        meta["panel_shape"],
        next_top,
        meta["next_shape"],
        meta["next_side"],
        meta["next_row_offset"],
        meta["next_color"],
    )


BASE_SHAPES = [
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=int),
]


def _random_shape() -> np.ndarray:
    for _ in range(50):
        shape = np.array(random.choice(BASE_SHAPES), dtype=int)
        shape = np.rot90(shape, k=random.randrange(4))
        box = _bbox(shape.astype(bool))
        assert box is not None
        shape = _crop_bbox(shape, box)
        if shape.sum() < 3 or shape.sum() > 5:
            continue
        if np.array_equal(shape, np.fliplr(shape)):
            continue
        if np.count_nonzero(shape[:, 0]) == 0 or np.count_nonzero(shape[:, -1]) == 0:
            continue
        return shape
    raise RuntimeError("failed to sample an asymmetric shape")


def _build_sequence(
    panel_shape: tuple[int, int],
    start_top: tuple[int, int],
    delta: tuple[int, int],
    base_shape: np.ndarray,
    odd_uses_base: bool,
    base_side: str,
    row_offset: int,
    colors: list[int],
) -> list[np.ndarray]:
    frames = []
    mirrored_shape = np.fliplr(base_shape)
    for t in range(5):
        use_base = (t % 2 == 0) == bool(odd_uses_base)
        shape = base_shape if use_base else mirrored_shape
        side = base_side if use_base else _flip_side(base_side)
        top = (int(start_top[0]) + t * int(delta[0]), int(start_top[1]) + t * int(delta[1]))
        frames.append(_render_panel(panel_shape, top, shape, side, row_offset, colors[t % 4]))
    return frames


def _stack_input(panels: list[np.ndarray]) -> np.ndarray:
    h, w = panels[0].shape
    out = np.full((4 * h + 3, w), BG, dtype=int)
    for i, panel in enumerate(panels):
        r0 = i * (h + 1)
        out[r0 : r0 + h, :] = panel
        if i < 3:
            out[r0 + h, :] = SEPARATOR
    return out


def _validate_example(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        panels = _split_panels(inp)
        meta = _analyze_sequence(panels)
        got = main(inp)
    except Exception:
        return False

    if not np.array_equal(got, expected):
        return False
    if any(np.array_equal(got, panel) for panel in panels):
        return False

    last_top = meta["infos"][-1]["top_left"]
    dr, dc = meta["delta"]
    next_top = (int(last_top[0]) + int(dr), int(last_top[1]) + int(dc))

    translation_only = _render_panel(
        meta["panel_shape"],
        next_top,
        meta["infos"][-1]["shape"],
        meta["infos"][-1]["side"],
        meta["infos"][-1]["row_offset"],
        meta["next_color"],
    )
    if np.array_equal(translation_only, expected):
        return False

    mirror_only = _render_panel(
        meta["panel_shape"],
        last_top,
        meta["next_shape"],
        meta["next_side"],
        meta["next_row_offset"],
        meta["next_color"],
    )
    if np.array_equal(mirror_only, expected):
        return False

    return True


def _generate_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    base_shape = _random_shape()
    sh, sw = base_shape.shape

    deltas = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
        if sh + 4 * abs(dr) <= 6 and sw + 2 + 4 * abs(dc) <= 9
    ]
    if not deltas:
        return None
    dr, dc = random.choice(deltas)

    min_h = max(5, sh + 4 * abs(dr))
    min_w = max(7, sw + 2 + 4 * abs(dc))
    panel_h = random.randint(min_h, 6)
    panel_w = random.randint(min_w, 9)

    base_side = random.choice(["left", "right"])
    candidate_rows = np.flatnonzero(base_shape[:, 0] if base_side == "left" else base_shape[:, -1])
    row_offset = int(random.choice(list(candidate_rows)))

    row_lo = 0 if dr >= 0 else -4 * dr
    row_hi = panel_h - sh if dr <= 0 else panel_h - sh - 4 * dr
    col_lo = 1 if dc >= 0 else 1 - 4 * dc
    col_hi = panel_w - sw - 1 if dc <= 0 else panel_w - sw - 1 - 4 * dc
    if row_lo > row_hi or col_lo > col_hi:
        return None

    start_top = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    odd_uses_base = bool(random.getrandbits(1))
    colors = random.sample(list(INDICATOR_COLORS), 4)

    frames = _build_sequence(
        (panel_h, panel_w),
        start_top,
        (dr, dc),
        base_shape,
        odd_uses_base,
        base_side,
        row_offset,
        colors,
    )
    inp = _stack_input(frames[:4])
    expected = frames[4]
    if not _validate_example(inp, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid mirror-translation sequence")
