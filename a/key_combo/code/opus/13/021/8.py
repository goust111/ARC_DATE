import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

FRAME_COLOR = YELLOW
HEIGHT_MARKER = GREY
DIRECTION_MARKER = PINK
MOTIF_COLORS = (BLUE, RED, GREEN)
ROTATION_BY_SIDE = {"top": 0, "right": 3, "bottom": 2, "left": 1}
ALL_SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components from a boolean mask as (N,2) coord arrays."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_motif(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Read the grey-bracketed motif at the top-left, excluding the grey column."""
    h = 0
    while h < grid.shape[0] and int(grid[h, 0]) == HEIGHT_MARKER:
        h += 1
    assert h > 0, "expected a top-left grey height marker"
    assert int(np.count_nonzero(grid == HEIGHT_MARKER)) == h, "grey pixels should only mark motif rows"

    top_patch = grid[:h, 1:]
    non_bg = np.argwhere(top_patch != BG)
    assert non_bg.size > 0, "expected motif content next to the grey marker"
    w = int(non_bg[:, 1].max()) + 1
    motif = top_patch[:, :w].copy()
    assert motif.shape[0] != motif.shape[1], "motif height and width should differ"
    return h, w, motif


def _rectangle_bbox_from_component(grid: np.ndarray, coords: np.ndarray) -> tuple[int, int, int, int]:
    """Validate that a yellow component is exactly a 1-pixel rectangular border."""
    r0, c0, r1, c1 = _bbox(coords)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a non-empty interior"

    expected = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))

    actual = {tuple(map(int, coord)) for coord in coords}
    assert actual == expected, "yellow component must be a rectangular border"
    assert np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == BG), "frame interior must be empty in the input"
    return r0, c0, r1, c1


def _marker_side(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> str:
    """Find the unique magenta marker sitting one cell outside a frame midpoint."""
    r0, c0, r1, c1 = bbox
    candidates = []
    if (c1 - c0 + 1) % 2 == 1:
        mid_c = (c0 + c1) // 2
        if r0 > 0 and int(grid[r0 - 1, mid_c]) == DIRECTION_MARKER:
            candidates.append("top")
        if r1 + 1 < grid.shape[0] and int(grid[r1 + 1, mid_c]) == DIRECTION_MARKER:
            candidates.append("bottom")
    if (r1 - r0 + 1) % 2 == 1:
        mid_r = (r0 + r1) // 2
        if c0 > 0 and int(grid[mid_r, c0 - 1]) == DIRECTION_MARKER:
            candidates.append("left")
        if c1 + 1 < grid.shape[1] and int(grid[mid_r, c1 + 1]) == DIRECTION_MARKER:
            candidates.append("right")
    assert len(candidates) == 1, "each frame must have exactly one midpoint-side marker"
    return candidates[0]


def _tile_patch(out: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray) -> None:
    """Tile a patch cyclically inside the rectangle interior starting from the top-left."""
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    rows = np.arange(inner_h)[:, None] % patch.shape[0]
    cols = np.arange(inner_w)[None, :] % patch.shape[1]
    out[r0 + 1 : r1, c0 + 1 : c1] = patch[rows, cols]


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each yellow frame with the top-left motif, rotating the motif by the side of its magenta marker.

    Input structure:
    - The background is black.
    - A contiguous grey run in the top of the leftmost column marks the motif height, and the adjacent top-left rectangle contains the motif itself.
    - Elsewhere there are several non-overlapping yellow rectangular borders with empty black interiors, each with exactly one magenta marker one cell outside a border midpoint.

    Rule:
    - Extract the motif from the marked top-left rectangle after removing the grey column.
    - For each yellow frame, read which side its magenta marker occupies: top means 0 degrees, right means 90 degrees clockwise, bottom means 180 degrees, left means 270 degrees clockwise.
    - Clear the motif source area to black, keep all yellow borders and magenta markers, and cyclically tile the rotated motif across each frame interior from its top-left corner, truncating at the interior edges.

    Key insight:
    - The top-left marked patch provides a single reusable motif, while each frame's external marker converts local border position into the rotation needed before the cyclic fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    motif_h, motif_w, motif = _extract_motif(g)
    frame_components = _component_coords(g == FRAME_COLOR)
    assert 3 <= len(frame_components) <= 4, "expected three or four yellow frames"

    out = g.copy()
    out[:motif_h, : motif_w + 1] = BG

    frame_bboxes = []
    for coords in frame_components:
        bbox = _rectangle_bbox_from_component(g, coords)
        frame_bboxes.append(bbox)

    seen_markers = 0
    for bbox in sorted(frame_bboxes):
        side = _marker_side(g, bbox)
        rotated = np.rot90(motif, k=ROTATION_BY_SIDE[side])
        _tile_patch(out, bbox, rotated)
        seen_markers += 1

    assert int(np.count_nonzero(g == DIRECTION_MARKER)) == seen_markers, "magenta pixels should only be frame markers"
    return out


def _random_motif() -> np.ndarray:
    """Generate a non-square, rotation-distinct motif using 2-3 colors."""
    for _ in range(200):
        motif_h = random.randint(2, 4)
        motif_w = random.randint(3, 5)
        if motif_h == motif_w:
            continue

        mask = np.zeros((motif_h, motif_w), dtype=bool)
        for r in range(motif_h):
            mask[r, random.randrange(motif_w)] = True
        for c in range(motif_w):
            mask[random.randrange(motif_h), c] = True
        for r in range(motif_h):
            for c in range(motif_w):
                if random.random() < 0.55:
                    mask[r, c] = True

        if not np.all(mask.any(axis=1)) or not np.all(mask.any(axis=0)):
            continue

        color_count = random.randint(2, 3)
        colors = random.sample(MOTIF_COLORS, color_count)
        motif = np.zeros((motif_h, motif_w), dtype=int)
        filled = list(zip(*np.where(mask)))
        random.shuffle(filled)
        assigned = colors[:] + [random.choice(colors) for _ in range(len(filled) - color_count)]
        random.shuffle(assigned)
        for (r, c), color in zip(filled, assigned):
            motif[r, c] = color

        if len(set(int(v) for v in motif.flatten() if v != BG)) < 2:
            continue
        if np.array_equal(motif, np.rot90(motif, 2)):
            continue
        return motif
    raise RuntimeError("failed to build a non-degenerate motif")


def _pick_frame_specs(frame_count: int) -> list[dict]:
    """Choose unique frame sizes and distinct marker directions."""
    directions = random.sample(list(ALL_SIDES), frame_count)
    specs = []
    used_sizes = set()
    for side in directions:
        for _ in range(100):
            inner_h = random.randint(3, 7)
            inner_w = random.randint(3, 7)
            if side in ("top", "bottom") and inner_w % 2 == 0:
                continue
            if side in ("left", "right") and inner_h % 2 == 0:
                continue
            if (inner_h, inner_w) in used_sizes:
                continue
            used_sizes.add((inner_h, inner_w))
            specs.append({"side": side, "inner_h": inner_h, "inner_w": inner_w})
            break
        else:
            raise RuntimeError("failed to sample frame specs")
    specs.sort(key=lambda item: item["inner_h"] * item["inner_w"], reverse=True)
    return specs


def _mark_reserved(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r1 + 2)
    cc1 = min(occupied.shape[1], c1 + 2)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_frame(occupied: np.ndarray, spec: dict) -> tuple[int, int, int, int] | None:
    """Find a non-overlapping frame placement with one-cell breathing room."""
    box_h = spec["inner_h"] + 2
    box_w = spec["inner_w"] + 2
    h, w = occupied.shape

    if h - box_h < 3 or w - box_w < 3:
        return None

    row_choices = list(range(1, h - box_h))
    col_choices = list(range(1, w - box_w))
    random.shuffle(row_choices)
    random.shuffle(col_choices)
    for r0 in row_choices:
        r1 = r0 + box_h - 1
        for c0 in col_choices:
            c1 = c0 + box_w - 1
            if occupied[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2].any():
                continue
            _mark_reserved(occupied, r0, c0, r1, c1)
            return r0, c0, r1, c1
    return None


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], side: str) -> None:
    """Render a yellow border and its matching external magenta direction marker."""
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    if side == "top":
        grid[r0 - 1, (c0 + c1) // 2] = DIRECTION_MARKER
    elif side == "bottom":
        grid[r1 + 1, (c0 + c1) // 2] = DIRECTION_MARKER
    elif side == "left":
        grid[(r0 + r1) // 2, c0 - 1] = DIRECTION_MARKER
    else:
        grid[(r0 + r1) // 2, c1 + 1] = DIRECTION_MARKER


def _expected_output(
    grid: np.ndarray,
    motif: np.ndarray,
    motif_h: int,
    motif_w: int,
    frame_records: list[dict],
) -> np.ndarray:
    out = grid.copy()
    out[:motif_h, : motif_w + 1] = BG
    for record in frame_records:
        rotated = np.rot90(motif, k=ROTATION_BY_SIDE[record["side"]])
        _tile_patch(out, record["bbox"], rotated)
    return out


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    motif = _random_motif()
    motif_h, motif_w = motif.shape
    frame_count = random.choice((3, 4))
    height = random.randint(max(16, motif_h + 10), 24)
    width = random.randint(max(16, motif_w + 10), 24)

    grid = np.zeros((height, width), dtype=int)
    grid[:motif_h, 0] = HEIGHT_MARKER
    grid[:motif_h, 1 : motif_w + 1] = motif

    occupied = np.zeros((height, width), dtype=bool)
    _mark_reserved(occupied, 0, 0, motif_h - 1, motif_w)

    frame_records = []
    try:
        specs = _pick_frame_specs(frame_count)
    except RuntimeError:
        return None

    for spec in specs:
        bbox = _place_frame(occupied, spec)
        if bbox is None:
            return None
        frame_records.append({"bbox": bbox, "side": spec["side"]})
        _draw_frame(grid, bbox, spec["side"])

    expected = _expected_output(grid, motif, motif_h, motif_w, frame_records)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_candidate()
        if built is None:
            continue
        grid, expected = built
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
