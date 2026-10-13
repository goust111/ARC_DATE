import random

import numpy as np


BG = 0
MAX_ATTEMPTS = 400
FRAME_COUNT_CHOICES = (2, 3, 4)
PIECE_SIZE_CHOICES = (2, 3)
SIDES = ("top", "bottom", "left", "right")
GRID_MIN = 18
GRID_MAX = 30
MARGIN = 1
PADDING = 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each two-color fragment into the matching hollow frame, using the frame gap as the alignment cue.

    Input structure:
    - The background is black.
    - Several single-color rectangular hollow frames appear on the grid; each border is one pixel thick and has exactly one missing border pixel.
    - Several detached 2x2 or 3x3 solid fragments appear elsewhere; each fragment uses exactly two non-black colors, one matching a frame color and one imprint color.

    Rule:
    - Identify the frame paired with each fragment by the shared color that appears in both.
    - Keep the original frames, gaps, and scattered fragments unchanged.
    - Copy the fragment pattern into the matched frame interior: the shared color becomes transparent black, while the other color is written inside the frame.
    - The gap side sets the stamp anchor: top gaps align the fragment to the interior top edge, bottom gaps to the interior bottom edge, left gaps to the interior left edge, and right gaps to the interior right edge; the gap row or column also centers the fragment along the other axis.

    Key insight:
    - Solving the task requires three linked deductions: match each fragment to a frame by color, read the gap side and location as the placement anchor, and treat the frame-colored cells inside the fragment as a transparent mask instead of copied ink.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, fragments = _parse_scene(g)

    out = g.copy()
    for frame in frames:
        fragment = fragments[frame["color"]]
        patch = fragment["patch"]
        top, left = _stamp_origin(frame, patch.shape)
        mask = patch != frame["color"]
        target = out[top : top + patch.shape[0], left : left + patch.shape[1]]
        assert np.all(target[mask] == bg)
        target[mask] = patch[mask]

    return out


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            rows, cols = arr[:, 0], arr[:, 1]
            colors = tuple(sorted(int(x) for x in np.unique(grid[rows, cols])))
            comps.append(
                {
                    "coords": arr,
                    "bbox": _bbox_from_coords(arr),
                    "colors": colors,
                    "area": int(arr.shape[0]),
                }
            )

    return comps


def _parse_frame(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    if len(comp["colors"]) != 1:
        return None

    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    occupied = sub != bg
    boundary = np.zeros((h, w), dtype=bool)
    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    if np.any(occupied & ~boundary):
        return None
    if int(occupied.sum()) != int(boundary.sum()) - 1:
        return None

    missing = boundary & ~occupied
    if int(missing.sum()) != 1:
        return None

    mr, mc = map(int, np.argwhere(missing)[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    if mr == 0:
        side = "top"
    elif mr == h - 1:
        side = "bottom"
    elif mc == 0:
        side = "left"
    else:
        side = "right"

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "color": int(comp["colors"][0]),
        "side": side,
        "gap": (int(r0 + mr), int(c0 + mc)),
    }


def _parse_fragment(grid: np.ndarray, bg: int, comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h not in (2, 3) or w not in (2, 3) or h != w:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == bg):
        return None

    colors = tuple(sorted(int(x) for x in np.unique(sub)))
    if len(colors) != 2:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "patch": sub.copy(),
        "colors": colors,
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[int, dict]]:
    bg = _background(grid)
    comps = _components(grid, bg)
    assert comps

    frames: list[dict] = []
    fragments: list[dict] = []
    for comp in comps:
        frame = _parse_frame(grid, bg, comp)
        if frame is not None:
            frames.append(frame)
            continue

        fragment = _parse_fragment(grid, bg, comp)
        assert fragment is not None
        fragments.append(fragment)

    assert frames
    assert len(frames) == len(fragments)

    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        assert frame["color"] not in frame_by_color
        frame_by_color[frame["color"]] = frame

    fragment_by_color: dict[int, dict] = {}
    for fragment in fragments:
        matches = [color for color in fragment["colors"] if color in frame_by_color]
        assert len(matches) == 1
        match_color = int(matches[0])
        assert match_color not in fragment_by_color
        fragment_by_color[match_color] = fragment

    assert set(fragment_by_color) == set(frame_by_color)

    ordered_frames = [frame_by_color[color] for color in sorted(frame_by_color)]
    return bg, ordered_frames, fragment_by_color


def _stamp_origin(frame: dict, patch_shape: tuple[int, int]) -> tuple[int, int]:
    ph, pw = patch_shape
    r0, c0, r1, c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    side = frame["side"]

    if side == "top":
        top = r0 + 1
        left = gap_c - (pw // 2)
    elif side == "bottom":
        top = r1 - ph
        left = gap_c - (pw // 2)
    elif side == "left":
        top = gap_r - (ph // 2)
        left = c0 + 1
    else:
        top = gap_r - (ph // 2)
        left = c1 - pw

    assert r0 + 1 <= top <= r1 - ph
    assert c0 + 1 <= left <= c1 - pw
    assert top + ph - 1 <= r1 - 1
    assert left + pw - 1 <= c1 - 1
    return int(top), int(left)


def _random_piece_mask(size: int) -> np.ndarray:
    area = size * size
    count = random.randint(1, area - 1)
    chosen = set(random.sample(range(area), count))
    mask = np.zeros((size, size), dtype=bool)
    for idx in chosen:
        rr, cc = divmod(idx, size)
        mask[rr, cc] = True
    return mask


def _make_spec(frame_color: int, imprint_color: int, side: str) -> dict:
    size = random.choice(PIECE_SIZE_CHOICES)
    inner_h = random.randint(size + 1, min(size + 4, 8))
    inner_w = random.randint(size + 1, min(size + 4, 8))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    patch = np.full((size, size), frame_color, dtype=int)
    patch[_random_piece_mask(size)] = imprint_color

    if side == "top":
        stamp_top = 1
        stamp_left = random.randint(1, frame_w - size - 1)
        gap_local = (0, stamp_left + (size // 2))
    elif side == "bottom":
        stamp_top = frame_h - size - 1
        stamp_left = random.randint(1, frame_w - size - 1)
        gap_local = (frame_h - 1, stamp_left + (size // 2))
    elif side == "left":
        stamp_top = random.randint(1, frame_h - size - 1)
        stamp_left = 1
        gap_local = (stamp_top + (size // 2), 0)
    else:
        stamp_top = random.randint(1, frame_h - size - 1)
        stamp_left = frame_w - size - 1
        gap_local = (stamp_top + (size // 2), frame_w - 1)

    return {
        "frame_color": int(frame_color),
        "imprint_color": int(imprint_color),
        "side": side,
        "frame_shape": (int(frame_h), int(frame_w)),
        "piece_shape": (int(size), int(size)),
        "stamp_local": (int(stamp_top), int(stamp_left)),
        "gap_local": (int(gap_local[0]), int(gap_local[1])),
        "piece_patch": patch,
    }


def _choose_specs() -> list[dict]:
    n = random.choice(FRAME_COUNT_CHOICES)
    frame_colors = random.sample(range(1, 10), n)
    imprint_pool = [color for color in range(1, 10) if color not in frame_colors]
    imprint_colors = random.sample(imprint_pool, n)
    sides = random.sample(list(SIDES), n)
    return [
        _make_spec(frame_colors[i], imprint_colors[i], sides[i])
        for i in range(n)
    ]


def _place_rectangles(
    grid_h: int, grid_w: int, rects: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(rects)
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    order = sorted(range(len(rects)), key=lambda i: rects[i][0] * rects[i][1], reverse=True)

    for idx in order:
        rect_h, rect_w = rects[idx]
        min_r = MARGIN
        max_r = grid_h - rect_h - MARGIN
        min_c = MARGIN
        max_c = grid_w - rect_w - MARGIN
        if max_r < min_r or max_c < min_c:
            return None

        for _ in range(250):
            r0 = random.randint(min_r, max_r)
            c0 = random.randint(min_c, max_c)
            rr0 = max(0, r0 - PADDING)
            rr1 = min(grid_h, r0 + rect_h + PADDING)
            cc0 = max(0, c0 - PADDING)
            cc1 = min(grid_w, c0 + rect_w + PADDING)
            if np.any(occ[rr0:rr1, cc0:cc1]):
                continue
            occ[rr0:rr1, cc0:cc1] = True
            placements[idx] = (r0, c0)
            break

        if placements[idx] is None:
            return None

    return [(int(r), int(c)) for r, c in placements]


def _draw_frame(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    frame_h, frame_w = spec["frame_shape"]
    color = spec["frame_color"]
    grid[top, left : left + frame_w] = color
    grid[top + frame_h - 1, left : left + frame_w] = color
    grid[top : top + frame_h, left] = color
    grid[top : top + frame_h, left + frame_w - 1] = color
    gap_r, gap_c = spec["gap_local"]
    grid[top + gap_r, left + gap_c] = BG


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    specs = _choose_specs()
    total_area = sum(
        spec["frame_shape"][0] * spec["frame_shape"][1]
        + spec["piece_shape"][0] * spec["piece_shape"][1]
        for spec in specs
    )
    min_dim = max(GRID_MIN, min(GRID_MAX, int(np.ceil(np.sqrt(total_area * 1.6)))))
    grid_h = random.randint(min_dim, GRID_MAX)
    grid_w = random.randint(min_dim, GRID_MAX)

    rects: list[tuple[int, int]] = []
    for spec in specs:
        rects.append(spec["frame_shape"])
    for spec in specs:
        rects.append(spec["piece_shape"])

    placements = _place_rectangles(grid_h, grid_w, rects)
    if placements is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)

    for idx, spec in enumerate(specs):
        frame_top, frame_left = placements[idx]
        piece_top, piece_left = placements[len(specs) + idx]

        _draw_frame(grid, frame_top, frame_left, spec)
        grid[
            piece_top : piece_top + spec["piece_shape"][0],
            piece_left : piece_left + spec["piece_shape"][1],
        ] = spec["piece_patch"]

    expected[:] = grid
    for idx, spec in enumerate(specs):
        frame_top, frame_left = placements[idx]
        stamp_top, stamp_left = spec["stamp_local"]
        dest_top = frame_top + stamp_top
        dest_left = frame_left + stamp_left
        patch = spec["piece_patch"]
        mask = patch != spec["frame_color"]
        target = expected[dest_top : dest_top + patch.shape[0], dest_left : dest_left + patch.shape[1]]
        target[mask] = patch[mask]

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid input")
