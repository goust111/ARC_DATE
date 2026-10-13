from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

PANEL_WIDTH_RANGE = (4, 6)
PANEL_HEIGHT_RANGE = (6, 14)
SQUARE_SIZE_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 300

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _panel_slices(width: int, separator_cols: list[int]) -> list[slice]:
    bounds = [-1, *separator_cols, width]
    panels = [slice(bounds[i] + 1, bounds[i + 1]) for i in range(4)]
    sizes = [s.stop - s.start for s in panels]
    assert min(sizes) > 0, "separator columns must leave non-empty panels"
    assert len(set(sizes)) == 1, "the four panels must have equal width"
    return panels


def _monochrome_components(block: np.ndarray, background: int = BLACK) -> list[dict]:
    h, w = block.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(block[r, c]) == background:
                continue

            color = int(block[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(block[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return components


def _palette_colors(panel: np.ndarray) -> tuple[int, int]:
    comps = _monochrome_components(panel, background=BLACK)
    assert len(comps) == 2, "palette panel must contain exactly two solid square swatches"

    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height == width, "palette swatches must be square"
        assert comp["area"] == height * width, "palette swatches must be solid"
        assert np.all(panel[r0 : r1 + 1, c0 : c1 + 1] == comp["color"])

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    d_color = int(comps[0]["color"])
    e_color = int(comps[1]["color"])
    assert d_color != e_color, "palette swatches must have distinct colors"
    return d_color, e_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy selected cells from a template panel into the output panel and recolor them from a palette.

    Input structure:
    - The black grid is split by three full-height grey separator columns into
      four equal-width panels.
    - Panel 1 is a two-color mask using foreground color A and background color B.
    - Panel 2 is a same-size selector mask using foreground color C and the same
      background color B.
    - Panel 3 is an empty black canvas.
    - Panel 4 contains two solid square swatches, read top-to-bottom then
      left-to-right as colors D and E.

    Rule:
    - Treat C pixels in panel 2 as active selector positions.
    - For each active position, draw D in the output when panel 1 has A there,
      and draw E when panel 1 has B there.
    - Inactive selector positions remain black.
    - Return only the content of the third panel.

    Key insight:
    The four panels are roles: template, selector, output canvas, and palette;
    the visible operation is a boolean mask intersection followed by a two-color
    recoloring.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    separator_cols = [c for c in range(w) if bool(np.all(g[:, c] == GREY))]
    assert len(separator_cols) == 3, "expected three full-height grey separators"

    panels = _panel_slices(w, separator_cols)
    p1, p2, p3, p4 = [g[:, s] for s in panels]
    assert p1.shape == p2.shape == p3.shape == p4.shape
    assert np.all(p3 == BLACK), "third panel must be an empty black canvas"

    colors1 = set(map(int, np.unique(p1)))
    colors2 = set(map(int, np.unique(p2)))
    assert len(colors1) == 2 and len(colors2) == 2, "first two panels must be binary masks"
    shared = colors1 & colors2
    assert len(shared) == 1, "the mask background color must be the unique shared color"
    b_color = shared.pop()
    a_color = next(iter(colors1 - {b_color}))
    c_color = next(iter(colors2 - {b_color}))

    d_color, e_color = _palette_colors(p4)

    selector = p2 == c_color
    out = np.full((h, p1.shape[1]), BLACK, dtype=int)
    out[selector & (p1 == a_color)] = d_color
    out[selector & (p1 == b_color)] = e_color
    return out


def _grow_blob_mask(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while int(mask.sum()) < target and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        add_frontier(r, c)
    return mask


def _rectangle_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for _ in range(random.randint(1, 3)):
        rh = random.randint(1, max(1, height // 2 + 1))
        rw = random.randint(1, max(1, width // 2 + 1))
        r0 = random.randint(0, height - rh)
        c0 = random.randint(0, width - rw)
        mask[r0 : r0 + rh, c0 : c0 + rw] = True
    return mask


def _stripe_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.choice([True, False]):
        count = random.randint(1, max(1, height - 1))
        rows = random.sample(range(height), count)
        mask[rows, :] = True
    else:
        count = random.randint(1, max(1, width - 1))
        cols = random.sample(range(width), count)
        mask[:, cols] = True
    return mask


def _checker_mask(height: int, width: int) -> np.ndarray:
    offset = random.randrange(2)
    return np.fromfunction(lambda r, c: ((r + c + offset) % 2) == 0, (height, width), dtype=int)


def _scatter_mask(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    cells = random.sample(range(height * width), target)
    for idx in cells:
        mask[idx // width, idx % width] = True
    return mask


def _make_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    min_count = max(2, area // 5)
    max_count = max(min_count, area - max(2, area // 5))

    for _ in range(80):
        mode = random.choice(["blob", "rectangles", "stripes", "checker", "scatter"])
        target = random.randint(min_count, max_count)
        if mode == "blob":
            mask = _grow_blob_mask(height, width, target)
        elif mode == "rectangles":
            mask = _rectangle_mask(height, width)
        elif mode == "stripes":
            mask = _stripe_mask(height, width)
        elif mode == "checker":
            mask = _checker_mask(height, width)
        else:
            mask = _scatter_mask(height, width, target)

        count = int(mask.sum())
        if 0 < count < area:
            return mask

    return _checker_mask(height, width)


def _make_selector(template_mask: np.ndarray) -> np.ndarray:
    height, width = template_mask.shape
    for _ in range(120):
        selector = _make_mask(height, width)
        if (
            np.any(selector & template_mask)
            and np.any(selector & ~template_mask)
            and np.any(~selector)
        ):
            return selector

    selector = np.zeros_like(template_mask, dtype=bool)
    fg_cells = np.argwhere(template_mask)
    bg_cells = np.argwhere(~template_mask)
    for cells in (fg_cells, bg_cells):
        r, c = random.choice(cells.tolist())
        selector[r, c] = True

    remaining = [
        idx
        for idx in range(height * width)
        if not selector[idx // width, idx % width]
    ]
    extra_limit = max(0, len(remaining) - 1)
    extra_count = random.randint(0, extra_limit)
    for idx in random.sample(remaining, extra_count):
        selector[idx // width, idx % width] = True
    return selector


def _separated(a: tuple[int, int], b: tuple[int, int], size: int) -> bool:
    ar, ac = a
    br, bc = b
    return (
        ar + size < br
        or br + size < ar
        or ac + size < bc
        or bc + size < ac
    )


def _place_palette(panel: np.ndarray, d_color: int, e_color: int) -> bool:
    height, width = panel.shape
    size = random.choice([s for s in SQUARE_SIZE_CHOICES if s <= height and s <= width])
    positions = [(r, c) for r in range(height - size + 1) for c in range(width - size + 1)]
    random.shuffle(positions)

    for first in positions:
        choices = [pos for pos in positions if _separated(first, pos, size)]
        if not choices:
            continue
        second = random.choice(choices)
        ordered = sorted([first, second])
        (r0, c0), (r1, c1) = ordered
        panel[r0 : r0 + size, c0 : c0 + size] = d_color
        panel[r1 : r1 + size, c1 : c1 + size] = e_color
        return True
    return False


def _expected(template_mask: np.ndarray, selector: np.ndarray, d_color: int, e_color: int) -> np.ndarray:
    out = np.full(template_mask.shape, BLACK, dtype=int)
    out[selector & template_mask] = d_color
    out[selector & ~template_mask] = e_color
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_w = random.randint(*PANEL_WIDTH_RANGE)
    height = random.randint(*PANEL_HEIGHT_RANGE)
    total_w = 4 * panel_w + 3
    if max(height, total_w) > 30:
        return None

    a_color, b_color, c_color, d_color, e_color = random.sample(AVAILABLE_COLORS, 5)
    template_mask = _make_mask(height, panel_w)
    selector = _make_selector(template_mask)
    expected = _expected(template_mask, selector, d_color, e_color)
    if not (np.any(expected == d_color) and np.any(expected == e_color) and np.any(expected == BLACK)):
        return None

    grid = np.full((height, total_w), BLACK, dtype=int)
    sep_cols = [panel_w, 2 * panel_w + 1, 3 * panel_w + 2]
    grid[:, sep_cols] = GREY
    panels = _panel_slices(total_w, sep_cols)

    first = np.full((height, panel_w), b_color, dtype=int)
    first[template_mask] = a_color
    second = np.full((height, panel_w), b_color, dtype=int)
    second[selector] = c_color
    fourth = np.full((height, panel_w), BLACK, dtype=int)
    if not _place_palette(fourth, d_color, e_color):
        return None

    grid[:, panels[0]] = first
    grid[:, panels[1]] = second
    grid[:, panels[3]] = fourth

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/4/2 input")
