import numpy as np


RED = 2
BG = 0
PALETTE = (1, 3, 4, 5, 6, 7, 8, 9)
LAYOUTS = tuple(
    (n_panels, panel_w)
    for n_panels in range(3, 6)
    for panel_w in range(5, 8)
    if n_panels * panel_w + (n_panels - 1) <= 30
)
HEIGHT_RANGE = (8, 12)
MAX_GENERATION_ATTEMPTS = 500

SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
)
ASYMMETRIC_SHAPES = tuple(shape for shape in SHAPES if not np.array_equal(shape, np.fliplr(shape)))


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel())
    return int(np.argmax(counts))


def _find_separator_layout(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[int]]:
    height, width = map(int, grid.shape)
    by_color: dict[int, list[int]] = {}
    for c in range(width):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            by_color.setdefault(color, []).append(c)

    matches = []
    for color, cols in by_color.items():
        cols = sorted(cols)
        if len(cols) < 2:
            continue
        panel_w = int(cols[0])
        if panel_w <= 0:
            continue
        if int(width - cols[-1] - 1) != panel_w:
            continue
        expected = [panel_w + i * (panel_w + 1) for i in range(len(cols))]
        if cols != expected:
            continue
        n_panels = len(cols) + 1
        if n_panels < 3:
            continue
        matches.append((int(color), panel_w, n_panels, cols))

    assert len(matches) == 1, "expected one equally spaced set of full-height separator columns"
    sep_color, panel_w, n_panels, cols = matches[0]
    assert height > 0 and panel_w > 0 and n_panels * panel_w + (n_panels - 1) == width
    return sep_color, panel_w, n_panels, cols


def _split_panels(grid: np.ndarray, panel_w: int, n_panels: int) -> list[np.ndarray]:
    panels = []
    for i in range(n_panels):
        start = i * (panel_w + 1)
        panels.append(grid[:, start : start + panel_w].copy())
    return panels


def _render_folded_layers(layers: list[np.ndarray], bg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = map(int, layers[0].shape)
    out = np.full((height, width), bg, dtype=int)
    support = np.zeros((height, width), dtype=int)
    unique_counts = np.zeros((height, width), dtype=int)

    for r in range(height):
        for c in range(width):
            colors = [int(layer[r, c]) for layer in layers if int(layer[r, c]) != bg]
            if not colors:
                continue
            uniq = sorted(set(colors))
            support[r, c] = len(colors)
            unique_counts[r, c] = len(uniq)
            out[r, c] = uniq[0] if len(uniq) == 1 else RED
    return out, support, unique_counts


def _folded_layers_from_panels(panels: list[np.ndarray], mirror_odd: bool) -> list[np.ndarray]:
    layers = []
    for i, panel in enumerate(panels):
        if mirror_odd and i % 2 == 1:
            layers.append(np.fliplr(panel))
        else:
            layers.append(panel.copy())
    return layers


def _accordion_output_from_panels(panels: list[np.ndarray], bg: int, mirror_odd: bool = True) -> np.ndarray:
    layers = _folded_layers_from_panels(panels, mirror_odd=mirror_odd)
    out, _support, _unique_counts = _render_folded_layers(layers, bg)
    return out


def _last_wins_output(layers: list[np.ndarray], bg: int) -> np.ndarray:
    out = np.full_like(layers[0], bg)
    for layer in layers:
        mask = layer != bg
        out[mask] = layer[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the separator-sliced panels accordion-style onto the leftmost panel, keeping agreements and marking conflicts red.

    Input structure:
    - One full-height separator color appears in at least two vertical columns.
    - Those separator columns evenly divide the grid into three or more equal-width panels.
    - Each panel contains sparse foreground pixels on a dominant background.

    Rule:
    - Use the leftmost panel as the base output coordinate system.
    - Fold every later panel onto it from left to right: odd-indexed panels are mirrored horizontally, even-indexed panels keep their orientation.
    - At each output cell, keep the shared foreground color if all contributing layers agree, write red if different foreground colors collide, and keep background if no layer contributes.

    Key insight:
    - The separators define whole panels, and the nontrivial step is the alternating mirror needed by accordion folding rather than a simple same-direction overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _get_bg(g)
    _sep_color, panel_w, n_panels, _cols = _find_separator_layout(g, bg)
    panels = _split_panels(g, panel_w, n_panels)
    return _accordion_output_from_panels(panels, bg, mirror_odd=True)


def _region_clear(mask: np.ndarray, top: int, left: int, shape: np.ndarray, pad: int) -> bool:
    height, width = map(int, mask.shape)
    sh, sw = map(int, shape.shape)
    r0 = max(0, top - pad)
    r1 = min(height, top + sh + pad)
    c0 = max(0, left - pad)
    c1 = min(width, left + sw + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _mark_region(mask: np.ndarray, top: int, left: int, shape: np.ndarray, pad: int) -> None:
    height, width = map(int, mask.shape)
    sh, sw = map(int, shape.shape)
    r0 = max(0, top - pad)
    r1 = min(height, top + sh + pad)
    c0 = max(0, left - pad)
    c1 = min(width, left + sw + pad)
    mask[r0:r1, c0:c1] = True


def _stamp_shape(layer: np.ndarray, top: int, left: int, shape: np.ndarray, color: int) -> None:
    rr, cc = np.nonzero(shape)
    layer[top + rr, left + cc] = int(color)


def _try_place_region(
    layers: list[np.ndarray],
    panel_reserved: list[np.ndarray],
    global_reserved: np.ndarray,
    panel_ids: list[int],
    colors: list[int],
    shape_pool: tuple[np.ndarray, ...],
) -> bool:
    height, width = map(int, layers[0].shape)
    shapes = [shape for shape in shape_pool if shape.shape[0] <= height and shape.shape[1] <= width]
    if not shapes:
        return False

    order = np.random.permutation(len(shapes))
    for shape_idx in order:
        shape = shapes[int(shape_idx)]
        sh, sw = map(int, shape.shape)
        anchors = [(r, c) for r in range(height - sh + 1) for c in range(width - sw + 1)]
        np.random.shuffle(anchors)
        for top, left in anchors:
            if not _region_clear(global_reserved, top, left, shape, pad=0):
                continue
            if any(not _region_clear(panel_reserved[p], top, left, shape, pad=1) for p in panel_ids):
                continue

            for panel_id, color in zip(panel_ids, colors):
                _stamp_shape(layers[panel_id], top, left, shape, color)
                _mark_region(panel_reserved[panel_id], top, left, shape, pad=1)
            _mark_region(global_reserved, top, left, shape, pad=0)
            return True
    return False


def _sample_panel_subset(n_panels: int, size_low: int, size_high: int, require_odd: bool = False) -> list[int]:
    all_ids = list(range(n_panels))
    odd_ids = [idx for idx in all_ids if idx % 2 == 1]
    even_ids = [idx for idx in all_ids if idx % 2 == 0]

    for _ in range(100):
        size = int(np.random.randint(size_low, min(size_high, n_panels) + 1))
        subset = sorted(np.random.choice(all_ids, size=size, replace=False).tolist())
        if require_odd and not any(idx in odd_ids for idx in subset):
            continue
        if odd_ids and even_ids and not any(idx in even_ids for idx in subset):
            continue
        return subset
    raise RuntimeError("failed to sample a valid panel subset")


def _build_input_from_layers(layers: list[np.ndarray], sep_color: int, bg: int) -> np.ndarray:
    n_panels = len(layers)
    height, panel_w = map(int, layers[0].shape)
    width = n_panels * panel_w + (n_panels - 1)
    grid = np.full((height, width), bg, dtype=int)
    for i, layer in enumerate(layers):
        panel = layer if i % 2 == 0 else np.fliplr(layer)
        start = i * (panel_w + 1)
        grid[:, start : start + panel_w] = panel
        if i < n_panels - 1:
            grid[:, start + panel_w] = int(sep_color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    bg = BG
    n_panels, panel_w = LAYOUTS[int(np.random.randint(len(LAYOUTS)))]
    height = int(np.random.randint(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))

    sep_color = int(np.random.choice(PALETTE))
    fg_palette = [color for color in PALETTE if color != sep_color]
    odd_panels = [idx for idx in range(n_panels) if idx % 2 == 1]
    if not odd_panels:
        return None

    folded_layers = [np.full((height, panel_w), bg, dtype=int) for _ in range(n_panels)]
    panel_reserved = [np.zeros((height, panel_w), dtype=bool) for _ in range(n_panels)]
    global_reserved = np.zeros((height, panel_w), dtype=bool)

    agree_ids = _sample_panel_subset(n_panels, 2, 3, require_odd=True)
    agree_color = int(np.random.choice(fg_palette))
    if not _try_place_region(
        folded_layers,
        panel_reserved,
        global_reserved,
        agree_ids,
        [agree_color] * len(agree_ids),
        SHAPES[1:],
    ):
        return None

    conflict_ids = _sample_panel_subset(n_panels, 2, 3, require_odd=True)
    conflict_colors = np.random.choice(fg_palette, size=2, replace=False).tolist()
    per_panel_conflict_colors = [int(conflict_colors[0]), int(conflict_colors[1])]
    for _ in range(len(conflict_ids) - 2):
        per_panel_conflict_colors.append(int(conflict_colors[int(np.random.randint(2))]))
    if not _try_place_region(
        folded_layers,
        panel_reserved,
        global_reserved,
        conflict_ids,
        per_panel_conflict_colors,
        SHAPES[1:],
    ):
        return None

    mirror_panel = int(np.random.choice(odd_panels))
    mirror_color = int(np.random.choice(fg_palette))
    if not _try_place_region(
        folded_layers,
        panel_reserved,
        global_reserved,
        [mirror_panel],
        [mirror_color],
        ASYMMETRIC_SHAPES,
    ):
        return None

    for panel_id in range(n_panels):
        pool = ASYMMETRIC_SHAPES if panel_id % 2 == 1 else SHAPES
        color = int(np.random.choice(fg_palette))
        if not _try_place_region(
            folded_layers,
            panel_reserved,
            global_reserved,
            [panel_id],
            [color],
            pool,
        ):
            return None

    for _ in range(int(np.random.randint(0, 3))):
        if np.random.random() < 0.5:
            subset = _sample_panel_subset(n_panels, 2, 3, require_odd=False)
            color = int(np.random.choice(fg_palette))
            _try_place_region(
                folded_layers,
                panel_reserved,
                global_reserved,
                subset,
                [color] * len(subset),
                SHAPES[1:],
            )
        else:
            panel_id = int(np.random.randint(n_panels))
            pool = ASYMMETRIC_SHAPES if panel_id % 2 == 1 else SHAPES
            color = int(np.random.choice(fg_palette))
            _try_place_region(
                folded_layers,
                panel_reserved,
                global_reserved,
                [panel_id],
                [color],
                pool,
            )

    if not all(np.any(layer != bg) for layer in folded_layers):
        return None

    correct_out, support, unique_counts = _render_folded_layers(folded_layers, bg)
    has_conflict = bool(np.any(correct_out == RED))
    has_agreement = bool(np.any((support >= 2) & (unique_counts == 1)))
    if not (has_conflict and has_agreement):
        return None

    inp = _build_input_from_layers(folded_layers, sep_color=sep_color, bg=bg)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, correct_out):
        return None

    _, parsed_panel_w, parsed_n_panels, _ = _find_separator_layout(inp, bg)
    if parsed_panel_w != panel_w or parsed_n_panels != n_panels:
        return None

    panels = _split_panels(inp, panel_w, n_panels)
    wrong_no_mirror = _accordion_output_from_panels(panels, bg, mirror_odd=False)
    wrong_last_wins = _last_wins_output(folded_layers, bg)

    # Reject examples that do not force both the alternating mirror rule and the conflict-red rule.
    if np.array_equal(correct_out, wrong_no_mirror):
        return None
    if np.array_equal(correct_out, wrong_last_wins):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid accordion-fold input")
