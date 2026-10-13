from __future__ import annotations

from collections import Counter
import random

import numpy as np


CORNER_ORDER = ("tl", "tr", "br", "bl")
CORNER_TO_INDEX = {name: idx for idx, name in enumerate(CORNER_ORDER)}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    if w == 2 * h:
        n = h
        return grid[:, :n].copy(), grid[:, n:].copy()
    if h == 2 * w:
        n = w
        return grid[:n, :].copy(), grid[n:, :].copy()
    raise AssertionError("input must be two equal square panels placed side-by-side or stacked")


def _corner_coords(n: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, n - 1),
        "br": (n - 1, n - 1),
        "bl": (n - 1, 0),
    }


def _nearest_corner_label(r: int, c: int, n: int) -> str:
    distances = {
        name: abs(r - rr) + abs(c - cc)
        for name, (rr, cc) in _corner_coords(n).items()
    }
    best = min(distances.values())
    labels = [name for name, dist in distances.items() if dist == best]
    assert len(labels) == 1, "source pixel must have a unique nearest corner"
    return labels[0]


def _rotated_mask(mask: np.ndarray, from_corner: str, to_corner: str) -> np.ndarray:
    steps_cw = (CORNER_TO_INDEX[to_corner] - CORNER_TO_INDEX[from_corner]) % 4
    return np.rot90(mask, -steps_cw)


def _template_info(panel: np.ndarray, bg: int) -> dict:
    non_bg = panel != bg
    assert non_bg.any(), "template panel must contain a shape"

    counts = Counter(map(int, panel[non_bg].ravel()))
    assert len(counts) == 2, "template panel must use one body color and one pin color"

    pin_colors = [color for color, count in counts.items() if count == 1]
    assert len(pin_colors) == 1, "template panel must have exactly one unique pin pixel"
    pin_color = int(pin_colors[0])
    body_color = next(int(color) for color, count in counts.items() if count > 1)

    pin_coords = np.argwhere(panel == pin_color)
    assert pin_coords.shape == (1, 2), "pin color must appear exactly once"
    pin_r, pin_c = map(int, pin_coords[0])

    r0, c0, r1, c1 = _bbox(non_bg)
    bbox_corners = {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "br": (r1, c1),
        "bl": (r1, c0),
    }
    labels = [name for name, rc in bbox_corners.items() if rc == (pin_r, pin_c)]
    assert len(labels) == 1, "pin must lie on exactly one corner of the template bbox"

    assert counts[body_color] >= 2, "template body must contain at least two pixels"
    return {
        "panel": panel,
        "mask": non_bg,
        "pin_corner": labels[0],
        "pin_color": pin_color,
        "body_color": body_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each source pixel into a recolored template stamp whose rotation is chosen by that pixel's nearest panel corner.

    Input structure:
    - The input contains two equal square panels arranged side-by-side or stacked with a shared background color.
    - One panel is a sparse source field of colored pixels; each source pixel has a unique nearest corner under Manhattan distance within that panel.
    - The other panel is a template drawn in one body color plus one different pin pixel placed on a corner of the template shape's bbox.

    Rule:
    - Let each panel be N x N, and build an N^2 x N^2 background output.
    - For every non-background source pixel at (r, c), rotate the whole N x N template panel so its pin corner points to that source pixel's nearest panel corner.
    - Recolor every non-background cell of that rotated template to the source pixel's color, then stamp it into output block (r, c).

    Key insight:
    - The source panel contributes only positions and colors; the actual rotation is encoded geometrically by nearest-corner location, and the template pin is the anchor that defines which rotation corresponds to that location.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    panel_a, panel_b = _split_panels(g)
    n = panel_a.shape[0]
    assert panel_a.shape == (n, n) and panel_b.shape == (n, n)

    template = None
    source = None
    for first, second in ((panel_a, panel_b), (panel_b, panel_a)):
        try:
            template = _template_info(first, bg)
            source = second
            break
        except AssertionError:
            continue
    assert template is not None and source is not None, "could not identify template and source panels"

    out = np.full((n * n, n * n), bg, dtype=int)
    source_pixels = np.argwhere(source != bg)
    assert source_pixels.size > 0, "source panel must contain at least one colored pixel"

    for r, c in source_pixels:
        rr, cc = map(int, (r, c))
        color = int(source[rr, cc])
        target_corner = _nearest_corner_label(rr, cc, n)
        rotated = _rotated_mask(template["mask"], template["pin_corner"], target_corner)
        block = out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
        block[rotated] = color

    return out


GEN_ATTEMPTS = 400
SIZE_CHOICES = (4, 5)
PIXEL_COUNT_CHOICES = (3, 4)
COLOR_POOL = tuple(range(10))


def _build_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        options = []
        if r != er:
            options.append((r + (1 if er > r else -1), c))
        if c != ec:
            options.append((r, c + (1 if ec > c else -1)))
        r, c = random.choice(options)
        path.append((r, c))
    return path


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _shape_with_pin_bbox(h: int, w: int, pin_corner: str) -> np.ndarray:
    corners = {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "br": (h - 1, w - 1),
        "bl": (h - 1, 0),
    }
    opposite = {"tl": "br", "tr": "bl", "br": "tl", "bl": "tr"}
    start = corners[pin_corner]
    end = corners[opposite[pin_corner]]

    cells = set(_build_path(start, end))
    max_extra = max(1, min(h * w - len(cells), h + w - 2))
    target_area = len(cells) + random.randint(1, max_extra)
    while len(cells) < target_area:
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c, h, w))
        frontier -= cells
        assert frontier, "connected-growth frontier unexpectedly empty"
        cells.add(random.choice(sorted(frontier)))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    assert mask[start], "pin cell must belong to the shape"
    assert _bbox(mask) == (0, 0, h - 1, w - 1), "crop bbox must be fully used"
    return mask


def _all_rotation_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, -k) for k in range(4)]


def _distinct_rotation_count(mask: np.ndarray) -> int:
    return len({rot.tobytes() for rot in _all_rotation_masks(mask)})


def _choose_template_panel(n: int, bg: int, body_color: int, pin_color: int) -> tuple[np.ndarray, str]:
    for _ in range(200):
        h = random.randint(2, n)
        w = random.randint(2, n)
        if h == w:
            continue
        pin_corner = random.choice(CORNER_ORDER)
        crop = _shape_with_pin_bbox(h, w, pin_corner)
        panel = np.full((n, n), bg, dtype=int)
        top = random.randint(0, n - h)
        left = random.randint(0, n - w)
        panel[top : top + h, left : left + w][crop] = body_color
        pin_local = {
            "tl": (0, 0),
            "tr": (0, w - 1),
            "br": (h - 1, w - 1),
            "bl": (h - 1, 0),
        }[pin_corner]
        panel[top + pin_local[0], left + pin_local[1]] = pin_color
        mask = panel != bg
        if _distinct_rotation_count(mask) != 4:
            continue
        return panel, pin_corner
    raise RuntimeError("failed to build a non-degenerate template panel")


def _grouped_source_cells(n: int) -> dict[str, list[tuple[int, int]]]:
    groups = {corner: [] for corner in CORNER_ORDER}
    for r in range(n):
        for c in range(n):
            try:
                groups[_nearest_corner_label(r, c, n)].append((r, c))
            except AssertionError:
                pass
    return groups


def _well_spaced(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(candidate[0] - r), abs(candidate[1] - c)) > 1 for r, c in chosen)


def _choose_source_pixels(n: int) -> list[tuple[int, int, str]]:
    groups = _grouped_source_cells(n)
    count = random.choice(PIXEL_COUNT_CHOICES)
    labels = random.sample(CORNER_ORDER, count)

    for _ in range(200):
        chosen: list[tuple[int, int, str]] = []
        ok = True
        for label in labels:
            candidates = groups[label][:]
            random.shuffle(candidates)
            coord = next((rc for rc in candidates if _well_spaced(rc, [(r, c) for r, c, _ in chosen])), None)
            if coord is None:
                ok = False
                break
            chosen.append((coord[0], coord[1], label))
        if ok:
            random.shuffle(chosen)
            return chosen
    raise RuntimeError("failed to place well-spaced source pixels")


def _render_expected_output(
    n: int,
    bg: int,
    source_pixels: list[tuple[int, int, int]],
    template_mask: np.ndarray,
    pin_corner: str,
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    for r, c, color in source_pixels:
        target_corner = _nearest_corner_label(r, c, n)
        rotated = _rotated_mask(template_mask, pin_corner, target_corner)
        block = out[r * n : (r + 1) * n, c * n : (c + 1) * n]
        block[rotated] = color
    return out


def _assemble_input(
    n: int,
    bg: int,
    source_pixels: list[tuple[int, int, int]],
    template_panel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.full((n, n), bg, dtype=int)
    for r, c, color in source_pixels:
        source[r, c] = color

    if random.random() < 0.5:
        return np.concatenate([source, template_panel], axis=1), source
    return np.concatenate([source, template_panel], axis=0), source


def _validate_instance(
    inp: np.ndarray,
    bg: int,
    template_panel: np.ndarray,
    pin_corner: str,
    source_pixels: list[tuple[int, int, int]],
) -> bool:
    n = template_panel.shape[0]
    expected = _render_expected_output(n, bg, source_pixels, template_panel != bg, pin_corner)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    used_corners = {_nearest_corner_label(r, c, n) for r, c, _ in source_pixels}
    if len(used_corners) < 2:
        return False

    if _distinct_rotation_count(template_panel != bg) != 4:
        return False

    for alt_pin in CORNER_ORDER:
        if alt_pin == pin_corner:
            continue
        alt = _render_expected_output(n, bg, source_pixels, template_panel != bg, alt_pin)
        if np.array_equal(alt, expected):
            return False

    for fixed_corner in CORNER_ORDER:
        alt = np.full_like(expected, bg)
        for r, c, color in source_pixels:
            rotated = _rotated_mask(template_panel != bg, pin_corner, fixed_corner)
            block = alt[r * n : (r + 1) * n, c * n : (c + 1) * n]
            block[rotated] = color
        if np.array_equal(alt, expected):
            return False

    try:
        _template_info(template_panel, bg)
    except AssertionError:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n = random.choice(SIZE_CHOICES)
        bg = random.choice(COLOR_POOL)
        body_color, pin_color = random.sample([c for c in COLOR_POOL if c != bg], 2)
        source_positions = _choose_source_pixels(n)
        source_colors = random.sample([c for c in COLOR_POOL if c not in (bg, body_color, pin_color)], len(source_positions))
        source_pixels = [(r, c, color) for (r, c, _corner), color in zip(source_positions, source_colors)]

        template_panel, pin_corner = _choose_template_panel(n, bg, body_color, pin_color)
        inp, _source_panel = _assemble_input(n, bg, source_pixels, template_panel)

        if _validate_instance(inp, bg, template_panel, pin_corner, source_pixels):
            return inp

    raise RuntimeError("failed to generate a valid input")
