from collections import defaultdict

import numpy as np


BG_COLOR = 7
FRAME_COLOR = 5
ALL_COLORS = tuple(range(10))
PALETTE_COLORS = tuple(c for c in ALL_COLORS if c not in {BG_COLOR, FRAME_COLOR})

OUTPUT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400

FRAME_COUNT_RANGE = (3, 6)  # upper bound exclusive
FRAME_HEIGHT_RANGE = (5, 9)
FRAME_WIDTH_RANGE = (6, 11)
FRAME_GAP_RANGE = (1, 6)

TOP_MARGIN_RANGE = (0, 4)
BETWEEN_ROWS_GAP_RANGE = (1, 5)
CANVAS_EXTRA_WIDTH_RANGE = (0, 7)

PALETTE_EXTRA_COLORS_MAX = 2
PALETTE_HEIGHT_CHOICES = (1, 2)
PALETTE_SEGMENT_WIDTH_RANGE = (1, 4)


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )

    return comps


def _is_hollow_frame(comp: dict[str, object]) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _select_frame_components(comps: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [comp for comp in comps if _is_hollow_frame(comp)]
    assert candidates, "expected hollow rectangular frame components"

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for comp in candidates:
        r0, c0, r1, c1 = comp["bbox"]
        groups[(int(r1 - r0 + 1), int(c1 - c0 + 1))].append(comp)

    valid_groups = [group for group in groups.values() if len(group) >= 2]
    assert valid_groups, "expected multiple same-sized frames"
    return max(
        valid_groups,
        key=lambda group: (
            (group[0]["bbox"][2] - group[0]["bbox"][0] + 1)
            * (group[0]["bbox"][3] - group[0]["bbox"][1] + 1),
            len(group),
        ),
    )


def _read_palette_order(grid: np.ndarray, bg: int, start_row: int) -> list[int]:
    lower = grid[start_row:, :]
    mask = lower != bg
    pal_box = _bbox(mask)
    assert pal_box is not None, "expected a palette strip below the frames"
    r0, c0, r1, c1 = pal_box
    strip = lower[r0 : r1 + 1, c0 : c1 + 1]

    order: list[int] = []
    prev: int | None = None
    for col in range(strip.shape[1]):
        values = strip[:, col]
        colors = np.unique(values[values != bg])
        assert colors.size == 1, "each palette column must have one visible color"
        color = int(colors[0])
        if color != prev:
            order.append(color)
            prev = color

    assert len(order) == len(set(order)), "palette colors must appear in a unique left-to-right order"
    return order


def _frame_key(grid: np.ndarray, frame: dict[str, object], bg: int) -> tuple[int, np.ndarray]:
    r0, c0, r1, c1 = frame["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    interior = patch[1:-1, 1:-1]
    colors = np.unique(interior[(interior != bg) & (interior != int(frame["color"]))])
    assert colors.size == 1, "each frame must contain exactly one object color"
    return int(colors[0]), patch


def _render_sorted_row(patches: list[np.ndarray], bg: int) -> np.ndarray:
    assert patches, "expected at least one frame patch"
    frame_h, frame_w = patches[0].shape
    out_w = len(patches) * frame_w + max(0, len(patches) - 1) * OUTPUT_GAP
    out = np.full((frame_h, out_w), bg, dtype=int)
    x = 0
    for patch in patches:
        out[:, x : x + frame_w] = patch
        x += frame_w + OUTPUT_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Reorder framed objects by the bottom palette order and return only the sorted frame row."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    comps = _find_monochrome_components(g, bg)
    frames = _select_frame_components(comps)
    max_frame_bottom = max(int(frame["bbox"][2]) for frame in frames)

    palette_order = _read_palette_order(g, bg, max_frame_bottom + 1)
    palette_rank = {color: idx for idx, color in enumerate(palette_order)}

    keyed_patches: list[tuple[int, np.ndarray]] = []
    for frame in frames:
        color, patch = _frame_key(g, frame, bg)
        assert color in palette_rank, "frame object color must appear in the palette"
        keyed_patches.append((palette_rank[color], patch))

    keyed_patches.sort(key=lambda item: item[0])
    return _render_sorted_row([patch for _key, patch in keyed_patches], bg)


def _grow_connected_cells(
    rng: np.random.Generator, height: int, width: int, area: int
) -> list[tuple[int, int]]:
    seed = (int(rng.integers(0, height)), int(rng.integers(0, width)))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        choice = frontier[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    return list(cells)


def _random_motif_mask(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray:
    box_h = int(rng.integers(1, max_h + 1))
    box_w = int(rng.integers(1, max_w + 1))
    area = int(rng.integers(1, box_h * box_w + 1))
    cells = _grow_connected_cells(rng, box_h, box_w, area)

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    crop = _bbox(mask)
    assert crop is not None
    r0, c0, r1, c1 = crop
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _make_frame_patch(
    rng: np.random.Generator, frame_h: int, frame_w: int, object_color: int
) -> np.ndarray:
    patch = np.full((frame_h, frame_w), BG_COLOR, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[-1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, -1] = FRAME_COLOR

    max_motif_h = frame_h - 4
    max_motif_w = frame_w - 4
    motif = _random_motif_mask(rng, max_motif_h, max_motif_w)
    mh, mw = motif.shape

    top = int(rng.integers(2, frame_h - mh - 1))
    left = int(rng.integers(2, frame_w - mw - 1))
    region = patch[top : top + mh, left : left + mw]
    region[motif] = object_color
    return patch


def _expected_output(frame_colors_in_palette_order: list[int], patches: dict[int, np.ndarray]) -> np.ndarray:
    ordered = [patches[color] for color in frame_colors_in_palette_order]
    return _render_sorted_row(ordered, BG_COLOR)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))
    frame_h = int(rng.integers(*FRAME_HEIGHT_RANGE))
    frame_w = int(rng.integers(*FRAME_WIDTH_RANGE))
    palette_h = int(rng.choice(PALETTE_HEIGHT_CHOICES))

    max_extra = min(PALETTE_EXTRA_COLORS_MAX, len(PALETTE_COLORS) - n_frames)
    palette_extra = int(rng.integers(0, max_extra + 1))
    n_palette = n_frames + palette_extra

    palette_order = [
        int(x)
        for x in rng.choice(np.asarray(PALETTE_COLORS, dtype=int), size=n_palette, replace=False)
    ]
    frame_color_indices = set(
        int(x) for x in rng.choice(np.arange(n_palette), size=n_frames, replace=False)
    )
    sorted_frame_colors = [
        color for idx, color in enumerate(palette_order) if idx in frame_color_indices
    ]

    frame_order = sorted_frame_colors.copy()
    while True:
        rng.shuffle(frame_order)
        if frame_order != sorted_frame_colors:
            break

    patches = {
        color: _make_frame_patch(rng, frame_h, frame_w, color) for color in sorted_frame_colors
    }

    frame_gaps = [
        int(x) for x in rng.integers(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1], size=n_frames - 1)
    ]
    frame_span = n_frames * frame_w + sum(frame_gaps)

    palette_widths = [
        int(x)
        for x in rng.integers(
            PALETTE_SEGMENT_WIDTH_RANGE[0],
            PALETTE_SEGMENT_WIDTH_RANGE[1],
            size=n_palette,
        )
    ]
    palette_span = sum(palette_widths)

    width = max(frame_span, palette_span) + int(rng.integers(*CANVAS_EXTRA_WIDTH_RANGE))
    frame_left = int(rng.integers(0, width - frame_span + 1))
    palette_left = int(rng.integers(0, width - palette_span + 1))

    top_margin = int(rng.integers(*TOP_MARGIN_RANGE))
    between_gap = int(rng.integers(*BETWEEN_ROWS_GAP_RANGE))
    height = top_margin + frame_h + between_gap + palette_h
    frame_top = top_margin
    palette_top = height - palette_h

    grid = np.full((height, width), BG_COLOR, dtype=int)

    x = frame_left
    for idx, color in enumerate(frame_order):
        grid[frame_top : frame_top + frame_h, x : x + frame_w] = patches[color]
        x += frame_w
        if idx < len(frame_gaps):
            x += frame_gaps[idx]

    x = palette_left
    for color, seg_w in zip(palette_order, palette_widths):
        grid[palette_top : palette_top + palette_h, x : x + seg_w] = color
        x += seg_w

    if _most_common_color(grid) != BG_COLOR:
        return None

    expected = _expected_output(sorted_frame_colors, patches)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
