from __future__ import annotations

import numpy as np


BACKGROUND = 0
SEPARATOR_COLOR = 5
BLOCK_WIDTH = 2
VALID_BLOCK_HEIGHTS = (2, 3)
PANEL_WIDTH_OPTIONS = (6, 7)
PANEL_HEIGHT_OPTIONS = {
    2: (5, 6, 7),
    3: (6, 7),
}
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 600

_SEEN_INPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(panel: np.ndarray, coords: list[tuple[int, int]], bg: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    crop = panel[r0 : r1 + 1, c0 : c1 + 1].copy()
    assert crop.shape[1] == BLOCK_WIDTH, "each block must be 2 columns wide"
    assert crop.shape[0] in VALID_BLOCK_HEIGHTS, "each block must be 2x2 or 3x2"
    assert len(coords) == crop.shape[0] * crop.shape[1], "each block must fill its bbox"
    assert np.all(crop != bg), "blocks must be solid rectangles without background holes"
    return crop


def _separator_columns(grid: np.ndarray, bg: int) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    full_cols = [c for c in range(w) if int(g[0, c]) != bg and bool(np.all(g[:, c] == g[0, c]))]
    assert len(full_cols) == 2, "expected exactly two full-height separator columns"
    s0, s1 = map(int, full_cols)
    sep = int(g[0, s0])
    assert sep == SEPARATOR_COLOR, "divider columns must use grey/5"
    assert sep == int(g[0, s1]), "separator columns must share one color"
    assert int(np.count_nonzero(g == sep)) == 2 * h, "separator color must appear only in the divider columns"
    panel_w = s0
    assert s1 == panel_w + 1 + panel_w, "panels must have equal width"
    assert w == panel_w * 3 + 2, "grid width must match three panels plus two dividers"
    return s0, s1


def _split_panels(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    s0, s1 = _separator_columns(grid, bg)
    return [
        grid[:, :s0].copy(),
        grid[:, s0 + 1 : s1].copy(),
        grid[:, s1 + 1 :].copy(),
    ]


def _panel_anomaly(panel: np.ndarray, bg: int) -> np.ndarray:
    components = _find_components(panel != bg)
    assert 2 <= len(components) <= 3, "each panel must contain two or three separated blocks"

    crops = [_crop_component(panel, coords, bg) for coords in components]
    color_counts = [len(set(map(int, crop.ravel()))) for crop in crops]
    assert color_counts.count(3) == 1, "each panel must contain exactly one three-color anomaly block"
    assert sum(count == 2 for count in color_counts) == len(color_counts) - 1, "all other blocks must use two colors"
    return crops[color_counts.index(3)]


def _triangular_starts(block_w: int, count: int) -> list[int]:
    return [i * block_w + (i * (i + 1)) // 2 for i in range(count)]


def _assemble_output(crops: list[np.ndarray], bg: int) -> np.ndarray:
    assert crops, "expected at least one crop"
    block_h, block_w = map(int, crops[0].shape)
    assert all(crop.shape == (block_h, block_w) for crop in crops), "all selected crops must share one shape"

    starts = _triangular_starts(block_w, len(crops))
    width = starts[-1] + block_w
    out = np.full((block_h, width), bg, dtype=int)
    for crop, c0 in zip(crops, starts):
        out[:, c0 : c0 + block_w] = crop
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract one three-color block per divider-separated panel, then place the crops right-to-left with triangular gaps.

    Input structure:
    - The grid has black background and exactly two full-height grey divider columns that split it into three equal-width panels.
    - Each panel contains two or three separated solid 2-column rectangles, and every rectangle is either 2x2 or 3x2.
    - In each panel, exactly one rectangle uses three non-background colors, while every other rectangle uses exactly two.

    Rule:
    - Detect the divider columns and process each panel independently.
    - In every panel, find the unique solid rectangle with the largest color diversity, which here is the only three-color rectangle, and crop it tightly.
    - Output those three crops in reverse panel order (right, middle, left).
    - Treat triangular numbers as spacer offsets rather than overlapping absolute columns: if each crop has width w, their left columns are 0, w+1, and 2w+3, so the empty gaps grow by 1 then 2.

    Key insight:
    - The result depends on three independent cues: divider columns define the panel groups, color diversity selects the anomaly inside each group, and triangular spacing controls the final placement of the cropped blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    panels = _split_panels(g, bg)
    selected = [_panel_anomaly(panel, bg) for panel in panels]
    return _assemble_output(list(reversed(selected)), bg)


def _positive_compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(1, total - parts + 2):
        for tail in _positive_compositions(total - first, parts - 1):
            out.append((first,) + tail)
    return out


def _sample_colors(rng: np.random.Generator, count: int, preferred: tuple[int, ...] = ()) -> list[int]:
    chosen: list[int] = []
    shared = [int(color) for color in preferred if int(color) in COLOR_POOL]
    if shared and count >= 2 and float(rng.random()) < 0.6:
        chosen.append(int(rng.permutation(shared)[0]))
    remaining = [color for color in COLOR_POOL if color not in chosen]
    remaining = list(rng.permutation(remaining))
    chosen.extend(int(color) for color in remaining[: count - len(chosen)])
    return [int(color) for color in chosen]


def _sample_block(
    rng: np.random.Generator,
    block_h: int,
    block_w: int,
    color_count: int,
    *,
    preferred_colors: tuple[int, ...] = (),
    forbidden_signatures: set[tuple[tuple[int, int], bytes]] | None = None,
) -> np.ndarray:
    area = block_h * block_w
    compositions = _positive_compositions(area, color_count)
    for _ in range(120):
        counts = list(rng.permutation(compositions[int(rng.integers(len(compositions)))]))
        colors = _sample_colors(rng, color_count, preferred_colors)
        cells = np.repeat(np.asarray(colors, dtype=int), np.asarray(counts, dtype=int))
        rng.shuffle(cells)
        block = cells.reshape(block_h, block_w)
        if len(set(map(int, block.ravel()))) != color_count:
            continue
        sig = _signature(block)
        if forbidden_signatures is not None and sig in forbidden_signatures:
            continue
        return block
    raise RuntimeError("failed to sample a distinct filled block")


def _available_anchors(panel_h: int, panel_w: int, block_h: int, block_w: int) -> list[tuple[int, int]]:
    top_positions = list(range(0, panel_h - block_h + 1, block_h + 1))
    left_positions = list(range(0, panel_w - block_w + 1, block_w + 1))
    return [(r, c) for r in top_positions for c in left_positions]


def _place_block(canvas: np.ndarray, block: np.ndarray, top: int, left: int) -> None:
    h, w = block.shape
    canvas[top : top + h, left : left + w] = block


def _choose_block_counts(rng: np.random.Generator, anchors_per_panel: list[int]) -> list[int]:
    counts = []
    for anchor_count in anchors_per_panel:
        max_blocks = min(3, anchor_count)
        assert max_blocks >= 2, "each panel must be able to hold at least two blocks"
        if max_blocks == 2:
            counts.append(2)
        else:
            counts.append(3 if float(rng.random()) < 0.5 else 2)
    if all(count == 2 for count in counts) and any(anchor_count >= 3 for anchor_count in anchors_per_panel):
        eligible = [idx for idx, anchor_count in enumerate(anchors_per_panel) if anchor_count >= 3]
        counts[int(rng.choice(eligible))] = 3
    return counts


def _build_panel(
    rng: np.random.Generator,
    panel_h: int,
    panel_w: int,
    target_block: np.ndarray,
    block_count: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    block_h, block_w = map(int, target_block.shape)
    anchors = _available_anchors(panel_h, panel_w, block_h, block_w)
    assert len(anchors) >= block_count
    order = list(rng.permutation(len(anchors)))
    chosen_anchors = [anchors[idx] for idx in order[:block_count]]
    target_index = int(rng.integers(block_count))
    target_anchor = chosen_anchors[target_index]

    panel = np.full((panel_h, panel_w), BACKGROUND, dtype=int)
    for idx, anchor in enumerate(chosen_anchors):
        if idx == target_index:
            block = target_block
        else:
            block = _sample_block(
                rng,
                block_h,
                block_w,
                2,
                preferred_colors=tuple(int(v) for v in target_block.ravel()),
            )
        _place_block(panel, block, anchor[0], anchor[1])
    return panel, target_anchor


def _signature(grid: np.ndarray) -> tuple[tuple[int, int], bytes]:
    arr = np.asarray(grid, dtype=int)
    return tuple(map(int, arr.shape)), arr.tobytes()


def _validate_generated_example(
    grid: np.ndarray,
    left_to_right_targets: list[np.ndarray],
    target_anchors: list[tuple[int, int]],
    block_counts: list[int],
) -> np.ndarray:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert all(2 <= count <= 3 for count in block_counts)
    assert len({anchor for anchor in target_anchors}) > 1, "target positions should vary across panels to block position shortcuts"

    target_signatures = [_signature(block) for block in left_to_right_targets]
    assert len(set(target_signatures)) == 3, "right-to-left order must matter, so all selected blocks must differ"

    panels = _split_panels(grid, BACKGROUND)
    scanned = [_panel_anomaly(panel, BACKGROUND) for panel in panels]
    for actual, expected in zip(scanned, left_to_right_targets):
        assert np.array_equal(actual, expected), "panel scan must recover the constructed anomaly block"

    expected = _assemble_output(list(reversed(left_to_right_targets)), BACKGROUND)
    assert max(expected.shape) <= 30
    starts = _triangular_starts(left_to_right_targets[0].shape[1], 3)
    assert starts == [0, BLOCK_WIDTH + 1, 2 * BLOCK_WIDTH + 3]

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match the construction semantics"
    return expected


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    block_h = int(rng.choice(VALID_BLOCK_HEIGHTS))
    block_w = BLOCK_WIDTH
    panel_h = int(rng.choice(PANEL_HEIGHT_OPTIONS[block_h]))
    panel_w = int(rng.choice(PANEL_WIDTH_OPTIONS))

    anchors_per_panel = [len(_available_anchors(panel_h, panel_w, block_h, block_w))] * 3
    if min(anchors_per_panel) < 2:
        return None
    block_counts = _choose_block_counts(rng, anchors_per_panel)

    used_target_signatures: set[tuple[tuple[int, int], bytes]] = set()
    targets: list[np.ndarray] = []
    for _ in range(3):
        target = _sample_block(
            rng,
            block_h,
            block_w,
            3,
            forbidden_signatures=used_target_signatures,
        )
        used_target_signatures.add(_signature(target))
        targets.append(target)

    panels: list[np.ndarray] = []
    target_anchors: list[tuple[int, int]] = []
    for target, block_count in zip(targets, block_counts):
        panel, target_anchor = _build_panel(rng, panel_h, panel_w, target, block_count)
        panels.append(panel)
        target_anchors.append(target_anchor)

    sep_col = np.full((panel_h, 1), SEPARATOR_COLOR, dtype=int)
    grid = np.hstack([panels[0], sep_col, panels[1], sep_col, panels[2]])
    expected = _validate_generated_example(grid, targets, target_anchors, block_counts)

    input_sig = _signature(grid)
    output_sig = _signature(expected)
    if input_sig in _SEEN_INPUT_SIGNATURES or output_sig in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_INPUT_SIGNATURES.add(input_sig)
    _SEEN_OUTPUT_SIGNATURES.add(output_sig)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            attempt = _gen_attempt()
        except AssertionError:
            attempt = None
        if attempt is not None:
            grid, _expected = attempt
            return grid
    raise RuntimeError("failed to generate a valid anomaly-panel triangular-spacing input")
