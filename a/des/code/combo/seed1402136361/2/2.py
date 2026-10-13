import numpy as np


def _group_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            groups.append((start, idx - 1))
            start = None
    if start is not None:
        groups.append((start, len(mask) - 1))
    return groups


def _non_separator_spans(length: int, groups: list[tuple[int, int]]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for start, end in groups:
        if cursor < start:
            spans.append((cursor, start - 1))
        cursor = end + 1
    if cursor < length:
        spans.append((cursor, length - 1))
    return spans


def _find_panel_layout(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates = []

    for color in map(int, np.unique(g)):
        row_mask = np.all(g == color, axis=1)
        col_mask = np.all(g == color, axis=0)
        row_groups = _group_runs(row_mask)
        col_groups = _group_runs(col_mask)
        if not row_groups or not col_groups:
            continue

        row_spans = _non_separator_spans(h, row_groups)
        col_spans = _non_separator_spans(w, col_groups)
        if len(row_spans) < 2 or len(col_spans) < 2:
            continue

        row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
        col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
        if len(row_heights) != 1 or len(col_widths) != 1:
            continue

        line_area = int(row_mask.sum()) * w + int(col_mask.sum()) * (h - int(row_mask.sum()))
        candidates.append(
            {
                "color": color,
                "row_spans": row_spans,
                "col_spans": col_spans,
                "row_mask": row_mask,
                "col_mask": col_mask,
                "score": (line_area, len(row_groups) + len(col_groups)),
            }
        )

    assert candidates, "expected a unique separator grid"
    best = max(candidates, key=lambda item: item["score"])
    return (
        int(best["color"]),
        list(best["row_spans"]),
        list(best["col_spans"]),
        np.asarray(best["row_mask"], dtype=bool),
        np.asarray(best["col_mask"], dtype=bool),
    )


def _infer_background(grid: np.ndarray, row_mask: np.ndarray, col_mask: np.ndarray) -> int:
    mask = np.ones_like(grid, dtype=bool)
    mask[row_mask, :] = False
    mask[:, col_mask] = False
    values, counts = np.unique(grid[mask], return_counts=True)
    winners = values[counts == counts.max()]
    assert len(winners) == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _paint_block(panel: np.ndarray, row: int, col: int, color: int) -> None:
    h, w = panel.shape
    r0 = max(0, row - 1)
    r1 = min(h, row + 2)
    c0 = max(0, col - 1)
    c1 = min(w, col + 2)
    panel[r0:r1, c0:c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat the fullest panel as a blueprint. In every other non-empty panel,
    keep only seed pixels whose local coordinate and color match the blueprint,
    then expand each surviving seed into a clipped 3x3 block.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _separator, row_spans, col_spans, row_mask, col_mask = _find_panel_layout(g)
    bg = _infer_background(g, row_mask, col_mask)

    panels: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panels.append(((r0, r1, c0, c1), g[r0 : r1 + 1, c0 : c1 + 1]))

    counts = [int(np.count_nonzero(panel != bg)) for _bbox, panel in panels]
    max_count = max(counts)
    assert max_count > 0, "expected a non-empty source panel"
    source_ids = [idx for idx, count in enumerate(counts) if count == max_count]
    assert len(source_ids) == 1, "source panel must be unique"
    source_idx = source_ids[0]
    source_panel = panels[source_idx][1]

    out = g.copy()
    panel_h, panel_w = source_panel.shape
    for idx, ((r0, r1, c0, c1), panel) in enumerate(panels):
        if idx == source_idx:
            continue
        if not np.any(panel != bg):
            continue

        new_panel = np.full((panel_h, panel_w), bg, dtype=int)
        for row, col in np.argwhere(panel != bg):
            color = int(panel[row, col])
            if int(source_panel[row, col]) == color and color != bg:
                _paint_block(new_panel, int(row), int(col), color)

        out[r0 : r1 + 1, c0 : c1 + 1] = new_panel

    return out


MAX_GENERATION_ATTEMPTS = 400
PANEL_ROW_RANGE = (2, 4)
PANEL_COL_RANGE = (2, 4)
PANEL_HEIGHT_RANGE = (4, 7)
PANEL_WIDTH_RANGE = (4, 7)
SOURCE_PIXELS_RANGE = (5, 10)
PALETTE_SIZE_RANGE = (3, 5)
MAX_ACTIVE_PANELS = 6


def _randint(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _pick_one(rng: np.random.Generator, seq):
    return seq[int(rng.integers(0, len(seq)))]


def _sample_without_replacement(rng: np.random.Generator, seq, count: int):
    order = rng.permutation(len(seq))[:count]
    return [seq[int(idx)] for idx in order]


def _is_border(coord: tuple[int, int], height: int, width: int) -> bool:
    row, col = coord
    return row == 0 or row == height - 1 or col == 0 or col == width - 1


def _coord_far_enough(coord: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    return all(max(abs(coord[0] - r), abs(coord[1] - c)) > 2 for r, c in chosen)


def _sample_source_panel(
    rng: np.random.Generator, panel_h: int, panel_w: int, bg: int, palette: list[int]
) -> np.ndarray:
    panel = np.full((panel_h, panel_w), bg, dtype=int)
    all_coords = [(r, c) for r in range(panel_h) for c in range(panel_w)]
    border_coords = [coord for coord in all_coords if _is_border(coord, panel_h, panel_w)]
    interior_coords = [coord for coord in all_coords if coord not in border_coords]

    max_pixels = min(SOURCE_PIXELS_RANGE[1], panel_h * panel_w - 1)
    min_pixels = min(SOURCE_PIXELS_RANGE[0], max_pixels)
    assert min_pixels >= 2
    pixel_count = _randint(rng, min_pixels, max_pixels)

    color_count = _randint(rng, 2, min(PALETTE_SIZE_RANGE[1], len(palette)))
    colors = _sample_without_replacement(rng, palette, color_count)

    chosen = [_pick_one(rng, border_coords)]
    second_pool = interior_coords if interior_coords else [coord for coord in all_coords if coord not in chosen]
    chosen.append(_pick_one(rng, second_pool))
    remaining = [coord for coord in all_coords if coord not in chosen]
    chosen.extend(_sample_without_replacement(rng, remaining, pixel_count - len(chosen)))

    panel[chosen[0]] = colors[0]
    panel[chosen[1]] = colors[1]
    for coord in chosen[2:]:
        panel[coord] = _pick_one(rng, colors)
    return panel


def _assemble_grid(panels: list[list[np.ndarray]], bg: int, separator: int) -> np.ndarray:
    panel_rows = len(panels)
    panel_cols = len(panels[0])
    panel_h, panel_w = panels[0][0].shape
    height = panel_rows * panel_h + (panel_rows - 1)
    width = panel_cols * panel_w + (panel_cols - 1)

    grid = np.full((height, width), bg, dtype=int)
    for pr in range(1, panel_rows):
        grid[pr * panel_h + (pr - 1), :] = separator
    for pc in range(1, panel_cols):
        grid[:, pc * panel_w + (pc - 1)] = separator

    for pr in range(panel_rows):
        for pc in range(panel_cols):
            r0 = pr * (panel_h + 1)
            c0 = pc * (panel_w + 1)
            grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panels[pr][pc]
    return grid


def _seed_kind_list(rng: np.random.Generator, total: int) -> list[str]:
    kinds = [
        "valid_edge" if rng.random() < 0.5 else "valid_any",
        "invalid_bg" if rng.random() < 0.5 else "invalid_color",
    ]
    while len(kinds) < total:
        if rng.random() < 0.5:
            kinds.append("valid_edge" if rng.random() < 0.4 else "valid_any")
        else:
            kinds.append("invalid_bg" if rng.random() < 0.5 else "invalid_color")
    order = rng.permutation(len(kinds))
    return [kinds[int(idx)] for idx in order]


def _choose_seed_coord(
    rng: np.random.Generator,
    kind: str,
    source_panel: np.ndarray,
    bg: int,
    chosen: list[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    panel_h, panel_w = source_panel.shape
    occupied = [(int(r), int(c)) for r, c in np.argwhere(source_panel != bg)]
    border_occupied = [coord for coord in occupied if _is_border(coord, panel_h, panel_w)]
    empty_coords = [(int(r), int(c)) for r, c in np.argwhere(source_panel == bg)]

    if kind == "valid_edge":
        pool = border_occupied
    elif kind == "valid_any":
        pool = occupied
    elif kind == "invalid_bg":
        pool = empty_coords
    else:
        pool = occupied

    pool = [coord for coord in pool if _coord_far_enough(coord, chosen)]
    if not pool:
        raise ValueError("no valid local coordinate for seed placement")

    coord = _pick_one(rng, pool)
    if kind.startswith("valid"):
        return coord, int(source_panel[coord])

    colors = [c for c in np.unique(source_panel) if c != bg]
    if kind == "invalid_bg":
        color = int(_pick_one(rng, list(colors)))
        return coord, color

    wrong_choices = [int(c) for c in colors if int(c) != int(source_panel[coord])]
    if not wrong_choices:
        raise ValueError("need at least two source colors for invalid-color seeds")
    return coord, int(_pick_one(rng, wrong_choices))


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    panel_rows = _randint(rng, *PANEL_ROW_RANGE)
    panel_cols = _randint(rng, *PANEL_COL_RANGE)
    panel_h = _randint(rng, *PANEL_HEIGHT_RANGE)
    panel_w = _randint(rng, *PANEL_WIDTH_RANGE)

    colors = list(rng.permutation(10))
    bg = int(colors[0])
    separator = int(colors[1])
    palette_size = _randint(rng, *PALETTE_SIZE_RANGE)
    palette = [int(c) for c in colors[2 : 2 + palette_size]]

    source_panel = _sample_source_panel(rng, panel_h, panel_w, bg, palette)
    source_pos = (int(rng.integers(panel_rows)), int(rng.integers(panel_cols)))

    input_panels = [[np.full((panel_h, panel_w), bg, dtype=int) for _ in range(panel_cols)] for _ in range(panel_rows)]
    output_panels = [[np.full((panel_h, panel_w), bg, dtype=int) for _ in range(panel_cols)] for _ in range(panel_rows)]
    input_panels[source_pos[0]][source_pos[1]] = source_panel.copy()
    output_panels[source_pos[0]][source_pos[1]] = source_panel.copy()

    other_positions = [
        (pr, pc)
        for pr in range(panel_rows)
        for pc in range(panel_cols)
        if (pr, pc) != source_pos
    ]
    max_nonempty = min(MAX_ACTIVE_PANELS, len(other_positions) - 1)
    active_count = _randint(rng, 2, max_nonempty)
    active_positions = _sample_without_replacement(rng, other_positions, active_count)

    seed_counts = [1 + int(rng.random() < 0.45) for _ in active_positions]
    kinds = _seed_kind_list(rng, sum(seed_counts))
    cursor = 0

    for position, seed_count in zip(active_positions, seed_counts):
        in_panel = np.full((panel_h, panel_w), bg, dtype=int)
        out_panel = np.full((panel_h, panel_w), bg, dtype=int)
        chosen_coords: list[tuple[int, int]] = []

        for kind in kinds[cursor : cursor + seed_count]:
            coord, color = _choose_seed_coord(rng, kind, source_panel, bg, chosen_coords)
            chosen_coords.append(coord)
            in_panel[coord] = color
            if kind.startswith("valid"):
                _paint_block(out_panel, coord[0], coord[1], color)
        cursor += seed_count

        input_panels[position[0]][position[1]] = in_panel
        output_panels[position[0]][position[1]] = out_panel

    input_grid = _assemble_grid(input_panels, bg, separator)
    output_grid = _assemble_grid(output_panels, bg, separator)
    return input_grid, output_grid


def _validate_generated_case(inp: np.ndarray, out: np.ndarray) -> None:
    g = np.asarray(inp, dtype=int)
    expected = np.asarray(out, dtype=int)
    separator, row_spans, col_spans, row_mask, col_mask = _find_panel_layout(g)
    assert separator in np.unique(g)
    bg = _infer_background(g, row_mask, col_mask)

    counts = []
    nonempty_counts = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            panel = g[r0 : r1 + 1, c0 : c1 + 1]
            count = int(np.count_nonzero(panel != bg))
            counts.append(count)
            if count:
                nonempty_counts.append(count)

    assert len(row_spans) >= 2 and len(col_spans) >= 2
    assert nonempty_counts and max(nonempty_counts) >= 3
    assert counts.count(max(counts)) == 1
    assert all(count in (0, 1, 2, max(counts)) for count in counts)
    assert np.array_equal(main(g), expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, out = _build_case(rng)
            _validate_generated_case(inp, out)
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
