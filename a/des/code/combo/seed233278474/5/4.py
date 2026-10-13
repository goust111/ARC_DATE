from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8)
GENERATION_ATTEMPTS = 400


def _candidate_panels_compatible(panels: list[np.ndarray], sep_color: int) -> bool:
    if len({panel.shape for panel in panels}) != 1:
        return False
    if any(np.any((panel == GREY) | (panel == sep_color)) for panel in panels):
        return False

    h, w = panels[0].shape
    for r in range(h):
        for c in range(w):
            seen = {
                int(panel[r, c])
                for panel in panels
                if int(panel[r, c]) not in {BLACK, MAROON}
            }
            if len(seen) > 1:
                return False
    return True


def _split_panels(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, list[np.ndarray]]] = []

    for color in (int(v) for v in np.unique(g) if int(v) != BLACK):
        cols = [c for c in range(w) if np.all(g[:, c] == color)]
        if not cols:
            continue
        if 0 in cols or (w - 1) in cols:
            continue
        if any(b == a + 1 for a, b in zip(cols, cols[1:])):
            continue

        panel_count = len(cols) + 1
        free_width = w - len(cols)
        if panel_count < 2 or free_width <= 0 or free_width % panel_count != 0:
            continue
        panel_w = free_width // panel_count
        expected_cols = [panel_w + i * (panel_w + 1) for i in range(panel_count - 1)]
        if cols != expected_cols:
            continue

        panels = [
            g[:, i * (panel_w + 1) : i * (panel_w + 1) + panel_w]
            for i in range(panel_count)
        ]
        if _candidate_panels_compatible(panels, color):
            candidates.append((color, panels))

    assert len(candidates) == 1, "expected a unique separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore one full pattern from multiple masked panel views separated by grey bars.

    Each panel shows the same underlying picture. Maroon rectangles hide different
    parts of that picture in different panels, so the output is obtained by
    fusing the visible non-background pixels at matching local coordinates.
    """
    _, panels = _split_panels(grid)
    out = np.full(panels[0].shape, BLACK, dtype=int)

    for panel in panels:
        visible = (panel != BLACK) & (panel != MAROON)
        conflict = visible & (out != BLACK) & (out != panel)
        assert not np.any(conflict), "panel views must agree where both are visible"
        out[visible] = panel[visible]
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fill_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[r0 : r0 + h, c0 : c0 + w] = color


def _draw_hollow_rect(canvas: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    canvas[r0, c0 : c0 + w] = color
    canvas[r0 + h - 1, c0 : c0 + w] = color
    canvas[r0 : r0 + h, c0] = color
    canvas[r0 : r0 + h, c0 + w - 1] = color


def _draw_plus(
    canvas: np.ndarray,
    center_r: int,
    center_c: int,
    arm_h: int,
    arm_w: int,
    thickness: int,
    color: int,
) -> None:
    h, w = canvas.shape
    vr0 = max(0, center_r - arm_h)
    vr1 = min(h, center_r + arm_h + 1)
    vc0 = max(0, center_c - thickness // 2)
    vc1 = min(w, center_c + (thickness + 1) // 2)
    canvas[vr0:vr1, vc0:vc1] = color

    hr0 = max(0, center_r - thickness // 2)
    hr1 = min(h, center_r + (thickness + 1) // 2)
    hc0 = max(0, center_c - arm_w)
    hc1 = min(w, center_c + arm_w + 1)
    canvas[hr0:hr1, hc0:hc1] = color


def _draw_el(
    canvas: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    orientation: str,
    thickness: int,
    color: int,
) -> None:
    if orientation == "tl":
        canvas[r0 : r0 + h, c0 : c0 + thickness] = color
        canvas[r0 : r0 + thickness, c0 : c0 + w] = color
    elif orientation == "tr":
        canvas[r0 : r0 + h, c0 + w - thickness : c0 + w] = color
        canvas[r0 : r0 + thickness, c0 : c0 + w] = color
    elif orientation == "bl":
        canvas[r0 : r0 + h, c0 : c0 + thickness] = color
        canvas[r0 + h - thickness : r0 + h, c0 : c0 + w] = color
    else:
        canvas[r0 : r0 + h, c0 + w - thickness : c0 + w] = color
        canvas[r0 + h - thickness : r0 + h, c0 : c0 + w] = color


def _build_mosaic(pattern: np.ndarray, palette: list[int], rng: random.Random) -> None:
    h, w = pattern.shape
    for _ in range(rng.randint(4, 7)):
        color = rng.choice(palette)
        rect_h = rng.randint(1, max(2, h // 2 + 1))
        rect_w = rng.randint(1, max(2, w // 2 + 1))
        if rng.random() < 0.4:
            if rng.random() < 0.5:
                rect_h = 1
            else:
                rect_w = 1
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        _fill_rect(pattern, r0, c0, rect_h, rect_w, color)

    if rng.random() < 0.7:
        color = rng.choice(palette)
        _draw_plus(
            pattern,
            rng.randint(1, h - 2),
            rng.randint(1, w - 2),
            rng.randint(1, max(1, h // 3)),
            rng.randint(1, max(1, w // 3)),
            1,
            color,
        )

    for _ in range(rng.randint(0, 2)):
        hole_h = rng.randint(1, max(1, h // 3))
        hole_w = rng.randint(1, max(1, w // 3))
        r0 = rng.randint(0, h - hole_h)
        c0 = rng.randint(0, w - hole_w)
        _fill_rect(pattern, r0, c0, hole_h, hole_w, BLACK)


def _build_frame(pattern: np.ndarray, palette: list[int], rng: random.Random) -> None:
    h, w = pattern.shape
    outer_h = rng.randint(max(4, h // 2), h)
    outer_w = rng.randint(max(4, w // 2), w)
    r0 = rng.randint(0, h - outer_h)
    c0 = rng.randint(0, w - outer_w)
    _draw_hollow_rect(pattern, r0, c0, outer_h, outer_w, palette[0])

    inner_h = rng.randint(2, max(2, outer_h - 1))
    inner_w = rng.randint(2, max(2, outer_w - 1))
    ir0 = rng.randint(r0, r0 + outer_h - inner_h)
    ic0 = rng.randint(c0, c0 + outer_w - inner_w)
    _fill_rect(pattern, ir0, ic0, inner_h, inner_w, palette[-1])

    for _ in range(rng.randint(1, 3)):
        color = rng.choice(palette)
        if rng.random() < 0.5:
            rect_h = rng.randint(1, 2)
            rect_w = rng.randint(max(2, w // 3), w)
        else:
            rect_h = rng.randint(max(2, h // 3), h)
            rect_w = rng.randint(1, 2)
        rect_h = min(rect_h, h)
        rect_w = min(rect_w, w)
        rr = rng.randint(0, h - rect_h)
        cc = rng.randint(0, w - rect_w)
        _fill_rect(pattern, rr, cc, rect_h, rect_w, color)

    if rng.random() < 0.6:
        hole_h = rng.randint(1, max(1, outer_h // 3))
        hole_w = rng.randint(1, max(1, outer_w // 3))
        rr = rng.randint(r0, r0 + outer_h - hole_h)
        cc = rng.randint(c0, c0 + outer_w - hole_w)
        _fill_rect(pattern, rr, cc, hole_h, hole_w, BLACK)


def _build_symbol(pattern: np.ndarray, palette: list[int], rng: random.Random) -> None:
    h, w = pattern.shape
    for _ in range(rng.randint(2, 4)):
        color = rng.choice(palette)
        if rng.random() < 0.5:
            shape_h = rng.randint(3, h)
            shape_w = rng.randint(3, w)
            r0 = rng.randint(0, h - shape_h)
            c0 = rng.randint(0, w - shape_w)
            thickness = 1 if min(shape_h, shape_w) < 5 else rng.randint(1, 2)
            _draw_el(
                pattern,
                r0,
                c0,
                shape_h,
                shape_w,
                rng.choice(("tl", "tr", "bl", "br")),
                thickness,
                color,
            )
        else:
            _draw_plus(
                pattern,
                rng.randint(1, h - 2),
                rng.randint(1, w - 2),
                rng.randint(1, max(1, h // 2 - 1)),
                rng.randint(1, max(1, w // 2 - 1)),
                rng.randint(1, 2),
                color,
            )

    for _ in range(rng.randint(1, 2)):
        color = rng.choice(palette)
        rect_h = rng.randint(1, max(2, h // 3))
        rect_w = rng.randint(1, max(2, w // 3))
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        _fill_rect(pattern, r0, c0, rect_h, rect_w, color)


def _pattern_interesting(pattern: np.ndarray) -> bool:
    mask = pattern != BLACK
    area = int(mask.sum())
    if area < max(10, pattern.size // 6):
        return False
    if area > pattern.size - 2:
        return False

    colors = {int(v) for v in pattern[mask]}
    if len(colors) < 2:
        return False

    box = _bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    if (r1 - r0 + 1) < 3 or (c1 - c0 + 1) < 3:
        return False
    return True


def _generate_pattern(panel_h: int, panel_w: int, rng: random.Random) -> np.ndarray:
    builders = (_build_mosaic, _build_frame, _build_symbol)
    for _ in range(150):
        palette = rng.sample(PATTERN_COLORS, rng.randint(2, 4))
        pattern = np.full((panel_h, panel_w), BLACK, dtype=int)
        rng.choice(builders)(pattern, palette, rng)
        if _pattern_interesting(pattern):
            return pattern
    raise RuntimeError("failed to build a non-degenerate pattern")


def _sample_mask_rect(
    shape: tuple[int, int],
    pattern_mask: np.ndarray,
    used: np.ndarray,
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = shape
    max_h = max(2, h - 2)
    max_w = max(2, w - 2)

    for _ in range(40):
        rect_h = rng.randint(2, max_h)
        rect_w = rng.randint(2, max_w)
        r0 = rng.randint(0, h - rect_h)
        c0 = rng.randint(0, w - rect_w)
        region = used[r0 : r0 + rect_h, c0 : c0 + rect_w]
        if np.any(region):
            continue
        overlap = int(pattern_mask[r0 : r0 + rect_h, c0 : c0 + rect_w].sum())
        if overlap == 0:
            continue
        return r0, c0, rect_h, rect_w
    return None


def _generate_masks(pattern: np.ndarray, panel_count: int, rng: random.Random) -> list[np.ndarray] | None:
    pattern_mask = pattern != BLACK
    colored_area = int(pattern_mask.sum())

    for _ in range(160):
        panel_masks: list[np.ndarray] = []
        for _panel in range(panel_count):
            mask = np.zeros_like(pattern_mask, dtype=bool)
            target_count = rng.randint(1, 3)
            for _ in range(target_count):
                rect = _sample_mask_rect(pattern.shape, pattern_mask, mask, rng)
                if rect is None:
                    continue
                r0, c0, rect_h, rect_w = rect
                mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True

            covered = int((mask & pattern_mask).sum())
            ratio = covered / colored_area
            if covered == 0 or covered >= colored_area or not (0.12 <= ratio <= 0.58):
                break
            panel_masks.append(mask)

        if len(panel_masks) != panel_count:
            continue

        coverage = np.sum(np.stack(panel_masks, axis=0), axis=0)
        if not np.any(coverage[pattern_mask] > 0):
            continue
        if np.any(coverage[pattern_mask] == panel_count):
            continue
        if len({mask.tobytes() for mask in panel_masks}) == 1:
            continue
        return panel_masks
    return None


def _assemble_input(pattern: np.ndarray, panel_masks: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = pattern.shape
    panel_count = len(panel_masks)
    grid_w = panel_count * panel_w + (panel_count - 1)
    grid = np.full((panel_h, grid_w), BLACK, dtype=int)

    for i, mask in enumerate(panel_masks):
        c0 = i * (panel_w + 1)
        panel = pattern.copy()
        panel[mask] = MAROON
        grid[:, c0 : c0 + panel_w] = panel
        if i < panel_count - 1:
            grid[:, c0 + panel_w] = GREY
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_count = rng.choice((3, 4))
    panel_h = rng.randint(8, 12)
    panel_w = rng.randint(7, 9) if panel_count == 3 else rng.randint(5, 6)

    pattern = _generate_pattern(panel_h, panel_w, rng)
    panel_masks = _generate_masks(pattern, panel_count, rng)
    if panel_masks is None:
        return None

    grid = _assemble_input(pattern, panel_masks)
    if not np.array_equal(main(grid), pattern):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid masked-panel input")
