from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            coords: list[Coord] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((int(start), idx - 1))
            start = None
    if start is not None:
        runs.append((int(start), len(mask) - 1))
    return runs


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=int))


def _is_rect_frame(grid: np.ndarray, comp: dict, bg: int) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    patch = _crop_bbox(grid, (r0, c0, r1, c1))
    height, width = patch.shape
    color = int(comp["color"])
    if height < 3 or width < 3:
        return False
    if int(comp["area"]) != 2 * height + 2 * width - 4:
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    return bool(np.all(patch[1:-1, 1:-1] == bg))


def _block_vote(block: np.ndarray, frame_color: int, bg: int) -> tuple[int, bool]:
    counts = Counter(map(int, block.reshape(-1)))
    counts.pop(int(bg), None)
    ranked = counts.most_common()
    assert ranked, "expected a non-empty color block"
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return int(ranked[0][0]), False
    assert ranked[0][1] == ranked[1][1], "unexpected majority structure"
    assert len(ranked) == 2 or ranked[1][1] > ranked[2][1], "expected a two-way tie for first place"
    return int(frame_color), True


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    frame_candidates = [comp for comp in components if _is_rect_frame(g, comp, bg)]
    assert len(frame_candidates) == 1, "expected exactly one hollow rectangular frame"
    frame = frame_candidates[0]
    frame_color = int(frame["color"])

    lattice_mask = g != bg
    frame_coords = frame["coords"]
    lattice_mask[frame_coords[:, 0], frame_coords[:, 1]] = False
    assert int(lattice_mask.sum()) > 0, "expected a separated block lattice"
    lattice_bbox = _bbox_coords(np.argwhere(lattice_mask))
    lattice = _crop_bbox(g, lattice_bbox)

    row_runs = _runs(np.any(lattice != bg, axis=1))
    col_runs = _runs(np.any(lattice != bg, axis=0))
    assert row_runs and col_runs, "expected at least one row band and one column band"

    row_heights = {r1 - r0 + 1 for r0, r1 in row_runs}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_runs}
    assert len(row_heights) == 1, "expected equal-height row bands"
    assert len(col_widths) == 1, "expected equal-width column bands"
    block_h = next(iter(row_heights))
    block_w = next(iter(col_widths))

    covered = np.zeros_like(lattice, dtype=bool)
    summary = np.zeros((len(row_runs), len(col_runs)), dtype=int)
    tie_pos: tuple[int, int] | None = None
    tie_count = 0
    for out_r, (r0, r1) in enumerate(row_runs):
        for out_c, (c0, c1) in enumerate(col_runs):
            block = lattice[r0 : r1 + 1, c0 : c1 + 1]
            assert block.shape == (block_h, block_w)
            assert not np.any(block == bg), "blocks must not contain background cells"
            covered[r0 : r1 + 1, c0 : c1 + 1] = True
            color, tied = _block_vote(block, frame_color, bg)
            summary[out_r, out_c] = color
            if tied:
                tie_count += 1
                tie_pos = (out_r, out_c)

    assert np.all(lattice[~covered] == bg), "expected only full separator rows and columns between blocks"
    assert tie_count == 1 and tie_pos is not None, "expected exactly one tied block"

    fr0, fc0, fr1, fc1 = frame["bbox"]
    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    assert inner_h > 0 and inner_w > 0, "frame interior must be non-empty"
    assert inner_h % summary.shape[0] == 0, "frame interior height must match an integer scale"
    assert inner_w % summary.shape[1] == 0, "frame interior width must match an integer scale"
    scale = inner_h // summary.shape[0]
    assert scale == inner_w // summary.shape[1] and scale > 0, "frame interior must imply one common scale"

    output = _scale_sprite(summary, scale)
    return {
        "bg": bg,
        "frame": frame,
        "frame_color": frame_color,
        "lattice_bbox": lattice_bbox,
        "row_runs": row_runs,
        "col_runs": col_runs,
        "block_shape": (block_h, block_w),
        "summary": summary,
        "scale": int(scale),
        "tie_pos": tie_pos,
        "output": output,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the noisy block lattice by majority vote, resolve the lone tie with the frame color, and stamp the scaled summary into the frame interior.

    Input structure:
    - One part of the grid is a rectangular lattice of equal-size non-background blocks separated by full background rows and columns.
    - Exactly one block has a two-way top-count tie; all other blocks have a unique majority color.
    - Another part of the grid is a single-color hollow rectangular frame whose interior is all background.

    Rule:
    - Collapse the lattice to an m x n summary by taking each block's majority color.
    - Resolve the unique tied block by using the frame border color instead of either tied color.
    - Infer the integer scale s from the frame interior size, enlarge each summary cell to an s x s block, and return only that filled interior crop.

    Key insight:
    - The frame is both the tie-break source and the scale reference: its color decides the undecidable block, and its interior dimensions define how the compressed summary must be magnified.
    """
    parsed = _parse_scene(np.asarray(grid, dtype=int))
    return parsed["output"].copy()


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _rand_scale() -> int:
    return random.choices([1, 2, 3], weights=[2, 4, 3], k=1)[0]


def _make_majority_block(height: int, width: int, majority: int, palette: list[int], force_noise: bool) -> np.ndarray:
    area = height * width
    block = np.full((height, width), int(majority), dtype=int)
    max_noise = (area - 1) // 2
    if max_noise <= 0:
        return block
    lower = 1 if force_noise else 0
    noise_count = random.randint(lower, min(max_noise, max(1, area // 3)))
    if noise_count == 0:
        return block
    cells = random.sample(range(area), noise_count)
    noise_colors = [color for color in palette if color != majority]
    for idx in cells:
        rr, cc = divmod(idx, width)
        block[rr, cc] = random.choice(noise_colors)
    return block


def _tie_block_counts(area: int) -> tuple[str, list[int]] | None:
    options: list[tuple[str, list[int]]] = []
    if area % 2 == 0:
        options.append(("pure", [area // 2, area // 2]))
        if area >= 8:
            half = area // 2 - 1
            options.append(("extra", [half, half, 2]))
    elif area >= 5:
        half = area // 2
        options.append(("extra", [half, half, 1]))
    return random.choice(options) if options else None


def _make_tie_block(height: int, width: int, palette: list[int]) -> tuple[np.ndarray, tuple[int, ...], str]:
    area = height * width
    style_and_counts = _tie_block_counts(area)
    if style_and_counts is None:
        raise ValueError("block area is too small for a visible tie")
    style, counts = style_and_counts
    colors = random.sample(palette, len(counts))
    flat: list[int] = []
    for color, count in zip(colors, counts):
        flat.extend([int(color)] * int(count))
    random.shuffle(flat)
    block = np.array(flat, dtype=int).reshape(height, width)
    return block, tuple(colors[:2]), style


def _render_lattice(
    summary: np.ndarray,
    block_h: int,
    block_w: int,
    sep: int,
    border: int,
    frame_color: int,
    color_pool: list[int],
    tie_pos: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], str]:
    rows, cols = summary.shape
    height = rows * block_h + (rows - 1) * sep + 2 * border
    width = cols * block_w + (cols - 1) * sep + 2 * border
    grid = np.full((height, width), BG, dtype=int)

    normal_positions = [(r, c) for r in range(rows) for c in range(cols) if (r, c) != tie_pos]
    forced_noisy = set(random.sample(normal_positions, k=random.randint(1, len(normal_positions))))
    tie_style = "pure"
    for r in range(rows):
        for c in range(cols):
            r0 = border + r * (block_h + sep)
            c0 = border + c * (block_w + sep)
            if (r, c) == tie_pos:
                block, tie_colors, tie_style = _make_tie_block(block_h, block_w, color_pool)
                assert frame_color not in tie_colors
                grid[r0 : r0 + block_h, c0 : c0 + block_w] = block
            else:
                majority = int(summary[r, c])
                block = _make_majority_block(
                    block_h,
                    block_w,
                    majority,
                    color_pool,
                    force_noise=(r, c) in forced_noisy,
                )
                grid[r0 : r0 + block_h, c0 : c0 + block_w] = block
    return grid, (height, width), tie_style


def _make_frame(inner_h: int, inner_w: int, color: int) -> np.ndarray:
    frame = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    frame[0, :] = int(color)
    frame[-1, :] = int(color)
    frame[:, 0] = int(color)
    frame[:, -1] = int(color)
    return frame


def _compose_scene(lattice: np.ndarray, frame: np.ndarray) -> np.ndarray | None:
    gap = random.randint(2, 4)
    layouts: list[np.ndarray] = []
    lattice_h, lattice_w = lattice.shape
    frame_h, frame_w = frame.shape

    if lattice_w + frame_w + gap <= 28 and max(lattice_h, frame_h) <= 28:
        content_h = max(lattice_h, frame_h)
        content_w = lattice_w + frame_w + gap
        extra_h = 30 - content_h
        extra_w = 30 - content_w
        top = random.randint(1, min(3, extra_h - 1))
        left = random.randint(1, min(3, extra_w - 1))
        bottom = random.randint(1, extra_h - top)
        right = random.randint(1, extra_w - left)
        canvas = np.full((content_h + top + bottom, content_w + left + right), BG, dtype=int)
        lattice_r = top + random.randint(0, content_h - lattice_h)
        frame_r = top + random.randint(0, content_h - frame_h)
        if random.random() < 0.5:
            lattice_c = left
            frame_c = left + lattice_w + gap
        else:
            frame_c = left
            lattice_c = left + frame_w + gap
        canvas[lattice_r : lattice_r + lattice_h, lattice_c : lattice_c + lattice_w] = lattice
        canvas[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = frame
        layouts.append(canvas)

    if lattice_h + frame_h + gap <= 28 and max(lattice_w, frame_w) <= 28:
        content_h = lattice_h + frame_h + gap
        content_w = max(lattice_w, frame_w)
        extra_h = 30 - content_h
        extra_w = 30 - content_w
        top = random.randint(1, min(3, extra_h - 1))
        left = random.randint(1, min(3, extra_w - 1))
        bottom = random.randint(1, extra_h - top)
        right = random.randint(1, extra_w - left)
        canvas = np.full((content_h + top + bottom, content_w + left + right), BG, dtype=int)
        lattice_c = left + random.randint(0, content_w - lattice_w)
        frame_c = left + random.randint(0, content_w - frame_w)
        if random.random() < 0.5:
            lattice_r = top
            frame_r = top + lattice_h + gap
        else:
            frame_r = top
            lattice_r = top + frame_h + gap
        canvas[lattice_r : lattice_r + lattice_h, lattice_c : lattice_c + lattice_w] = lattice
        canvas[frame_r : frame_r + frame_h, frame_c : frame_c + frame_w] = frame
        layouts.append(canvas)

    if not layouts:
        return None
    return random.choice(layouts)


def _semantic_ok(
    grid: np.ndarray,
    expected: np.ndarray,
    summary: np.ndarray,
    scale: int,
    frame_color: int,
    tie_pos: tuple[int, int],
    tie_style: str,
) -> bool:
    try:
        parsed = _parse_scene(grid)
    except Exception:
        return False
    if parsed["scale"] != scale:
        return False
    if parsed["frame_color"] != frame_color:
        return False
    if parsed["tie_pos"] != tie_pos:
        return False
    if not np.array_equal(parsed["summary"], summary):
        return False
    if not np.array_equal(parsed["output"], expected):
        return False
    if tie_style == "pure":
        area = parsed["block_shape"][0] * parsed["block_shape"][1]
        if area % 2 != 0:
            return False
    lattice = _crop_bbox(grid, parsed["lattice_bbox"])
    if np.any(lattice == frame_color):
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    scale = _rand_scale()
    block_h = random.randint(2, 4)
    block_w = random.randint(2, 4)
    sep = random.randint(1, 2)
    border = random.randint(1, 2)

    frame_color = random.choice(PALETTE)
    usable_colors = [color for color in PALETTE if color != frame_color]
    if len(usable_colors) < 4:
        return None

    pool_size = random.randint(3, min(6, len(usable_colors)))
    color_pool = random.sample(usable_colors, pool_size)

    summary = np.array(
        [[random.choice(color_pool) for _ in range(cols)] for _ in range(rows)],
        dtype=int,
    )
    tie_pos = (random.randrange(rows), random.randrange(cols))
    summary[tie_pos] = int(frame_color)

    lattice, lattice_shape, tie_style = _render_lattice(
        summary,
        block_h,
        block_w,
        sep,
        border,
        frame_color,
        color_pool,
        tie_pos,
    )
    inner_h = rows * scale
    inner_w = cols * scale
    frame = _make_frame(inner_h, inner_w, frame_color)
    scene = _compose_scene(lattice, frame)
    if scene is None:
        return None

    expected = _scale_sprite(summary, scale)
    if not _semantic_ok(scene, expected, summary, scale, frame_color, tie_pos, tie_style):
        return None
    return scene


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
