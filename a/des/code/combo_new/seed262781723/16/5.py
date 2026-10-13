import random

import numpy as np


_BACKGROUND = 8
_FRAME_COLOR = 9
_SEED_COLORS = (1, 2, 3, 4, 5, 6, 7)
_CORNERS = ("tl", "tr", "bl", "br")
_MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    patch = np.zeros((height, width), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(patch, border))


def _find_frames(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    colors = sorted(set(map(int, grid.flatten())) - {int(bg)})
    candidates: list[tuple[int, list[tuple[int, int, int, int]]]] = []
    for color in colors:
        comps = _components(grid == int(color))
        if not comps:
            continue
        if all(_is_rect_frame(comp) for comp in comps):
            boxes = sorted(_bbox(comp) for comp in comps)
            candidates.append((int(color), boxes))
    assert len(candidates) == 1, "expected exactly one common frame color"
    return candidates[0]


def _seed_for_box(
    grid: np.ndarray, bg: int, frame_color: int, box: tuple[int, int, int, int]
) -> tuple[int, int, int]:
    r0, c0, r1, c1 = box
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_coords = np.argwhere((interior != int(bg)) & (interior != int(frame_color)))
    assert seed_coords.shape == (1, 2), "each frame must contain exactly one seed pixel"
    sr = int(r0 + 1 + seed_coords[0, 0])
    sc = int(c0 + 1 + seed_coords[0, 1])
    return sr, sc, int(grid[sr, sc])


def _nearest_corner(
    box: tuple[int, int, int, int], seed_r: int, seed_c: int
) -> tuple[str, int, int]:
    r0, c0, r1, c1 = box
    options = [
        ("tl", abs(seed_r - (r0 + 1)), abs(seed_c - (c0 + 1))),
        ("tr", abs(seed_r - (r0 + 1)), abs(seed_c - (c1 - 1))),
        ("bl", abs(seed_r - (r1 - 1)), abs(seed_c - (c0 + 1))),
        ("br", abs(seed_r - (r1 - 1)), abs(seed_c - (c1 - 1))),
    ]
    ranked = sorted((dr + dc, name, dr, dc) for name, dr, dc in options)
    assert len(ranked) == 1 or ranked[1][0] > ranked[0][0], "seed must have a unique nearest corner"
    _, corner, dr, dc = ranked[0]
    return str(corner), int(dr), int(dc)


def _paint_corner_rectangle(
    out: np.ndarray, box: tuple[int, int, int, int], corner: str, dr: int, dc: int, color: int
) -> None:
    r0, c0, r1, c1 = box
    height = dr + 1
    width = dc + 1
    if corner == "tl":
        rr = slice(r0 + 1, r0 + 1 + height)
        cc = slice(c0 + 1, c0 + 1 + width)
    elif corner == "tr":
        rr = slice(r0 + 1, r0 + 1 + height)
        cc = slice(c1 - width, c1)
    elif corner == "bl":
        rr = slice(r1 - height, r1)
        cc = slice(c0 + 1, c0 + 1 + width)
    elif corner == "br":
        rr = slice(r1 - height, r1)
        cc = slice(c1 - width, c1)
    else:
        raise ValueError(f"unknown corner: {corner}")
    out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed pixel into a corner-anchored solid rectangle inside its enclosing frame.

    Input structure:
    - The background is the most common color.
    - Several disjoint 1-pixel-thick rectangular frames share one common color.
    - Each frame contains exactly one differently colored seed pixel somewhere near one interior corner.

    Rule:
    - Detect the shared frame color and locate every rectangular frame component.
    - For each frame, find its lone interior seed and determine which interior corner is nearest to that seed.
    - Let the seed be offset by dr rows and dc columns from that corner; fill a solid rectangle of height dr + 1 and width dc + 1 in the seed color, anchored to that same corner.
    - Keep the original frame borders unchanged and leave everything outside the frames untouched.

    Key insight: The seed pixel encodes two things at once: its color is the fill color, and its offset from the nearest corner becomes the height and width of the rectangle to draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    frame_color, boxes = _find_frames(g, bg)

    out = g.copy()
    for box in boxes:
        seed_r, seed_c, seed_color = _seed_for_box(g, bg, frame_color, box)
        corner, dr, dc = _nearest_corner(box, seed_r, seed_c)
        _paint_corner_rectangle(out, box, corner, dr, dc, seed_color)
    return out


def _seed_position(spec: dict) -> tuple[int, int]:
    top = int(spec["top"])
    left = int(spec["left"])
    height = int(spec["height"])
    width = int(spec["width"])
    corner = str(spec["corner"])
    dr = int(spec["dr"])
    dc = int(spec["dc"])
    if corner == "tl":
        return top + 1 + dr, left + 1 + dc
    if corner == "tr":
        return top + 1 + dr, left + width - 2 - dc
    if corner == "bl":
        return top + height - 2 - dr, left + 1 + dc
    if corner == "br":
        return top + height - 2 - dr, left + width - 2 - dc
    raise ValueError(f"unknown corner: {corner}")


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, _BACKGROUND, dtype=int)
    for spec in specs:
        top = int(spec["top"])
        left = int(spec["left"])
        height = int(spec["height"])
        width = int(spec["width"])
        seed_color = int(spec["seed_color"])

        grid[top, left : left + width] = _FRAME_COLOR
        grid[top + height - 1, left : left + width] = _FRAME_COLOR
        grid[top : top + height, left] = _FRAME_COLOR
        grid[top : top + height, left + width - 1] = _FRAME_COLOR

        seed_r, seed_c = _seed_position(spec)
        grid[seed_r, seed_c] = seed_color
    return grid


def _oracle_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        box = (
            int(spec["top"]),
            int(spec["left"]),
            int(spec["top"]) + int(spec["height"]) - 1,
            int(spec["left"]) + int(spec["width"]) - 1,
        )
        _paint_corner_rectangle(
            out,
            box,
            str(spec["corner"]),
            int(spec["dr"]),
            int(spec["dc"]),
            int(spec["seed_color"]),
        )
    return out


def _valid_offsets(interior_size: int) -> list[int]:
    return [d for d in range(3) if interior_size >= 2 * d + 2]


def _free_positions(blocked: np.ndarray, height: int, width: int) -> list[tuple[int, int]]:
    h, w = blocked.shape
    positions: list[tuple[int, int]] = []
    if height > h - 2 or width > w - 2:
        return positions
    for top in range(1, h - height):
        for left in range(1, w - width):
            if blocked[top - 1 : top + height + 1, left - 1 : left + width + 1].any():
                continue
            positions.append((top, left))
    return positions


def _choose_grid_shape(rng: random.Random, n_frames: int) -> tuple[int, int]:
    if n_frames == 1:
        return rng.randint(8, 18), rng.randint(8, 18)
    if n_frames == 2:
        return rng.randint(11, 22), rng.randint(11, 22)
    if n_frames == 3:
        return rng.randint(15, 27), rng.randint(15, 27)
    return rng.randint(19, 30), rng.randint(19, 30)


def _build_specs(rng: random.Random, shape: tuple[int, int], n_frames: int) -> list[dict] | None:
    h, w = shape
    max_height = min(11, h - 2)
    max_width = min(11, w - 2)
    size_pairs = [(fh, fw) for fh in range(6, max_height + 1) for fw in range(6, max_width + 1)]
    if len(size_pairs) < n_frames:
        return None

    rng.shuffle(size_pairs)
    seed_colors = rng.sample(_SEED_COLORS, n_frames)

    specs: list[dict] = []
    blocked = np.zeros(shape, dtype=bool)
    nontrivial = False

    for idx in range(n_frames):
        placed = False
        for fh, fw in size_pairs:
            if any(int(s["height"]) == fh and int(s["width"]) == fw for s in specs):
                continue
            dr_choices = _valid_offsets(fh - 2)
            dc_choices = _valid_offsets(fw - 2)
            if not dr_choices or not dc_choices:
                continue

            corner = rng.choice(_CORNERS)
            dr = rng.choice(dr_choices)
            dc = rng.choice(dc_choices)
            if idx == n_frames - 1 and not nontrivial and dr == 0 and dc == 0:
                dr_nonzero = [value for value in dr_choices if value > 0]
                dc_nonzero = [value for value in dc_choices if value > 0]
                if dr_nonzero:
                    dr = rng.choice(dr_nonzero)
                elif dc_nonzero:
                    dc = rng.choice(dc_nonzero)
                else:
                    continue

            positions = _free_positions(blocked, fh, fw)
            if not positions:
                continue
            top, left = rng.choice(positions)
            spec = {
                "top": int(top),
                "left": int(left),
                "height": int(fh),
                "width": int(fw),
                "corner": str(corner),
                "dr": int(dr),
                "dc": int(dc),
                "seed_color": int(seed_colors[idx]),
            }
            specs.append(spec)
            blocked[top - 1 : top + fh + 1, left - 1 : left + fw + 1] = True
            nontrivial = nontrivial or dr > 0 or dc > 0
            placed = True
            break
        if not placed:
            return None

    return specs if nontrivial else None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.choice((1, 2, 3, 4))
    shape = _choose_grid_shape(rng, n_frames)
    specs = _build_specs(rng, shape, n_frames)
    if specs is None:
        return None

    inp = _render_input(shape, specs)
    expected = _oracle_output(inp, specs)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    bg = _get_bg(inp)
    frame_color, boxes = _find_frames(inp, bg)
    if frame_color != _FRAME_COLOR or len(boxes) != n_frames:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after repeated attempts")
