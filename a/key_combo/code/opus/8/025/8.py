from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

LEGEND_COLOR_POOL = (BLUE, RED, GREEN, YELLOW)
BLOCK_SIZE = 7
PANEL_SLOT_SHAPE = (4, 4)
PANEL_SHAPE = (11, 11)
PANEL_SLOT_ORIGINS = ((1, 1), (1, 6), (6, 1), (6, 6))
SHAPE_POOL = (
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
)


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if int(connectivity) == 4:
        steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        steps = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
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
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    norm = _normalize_mask(np.asarray(mask, dtype=bool))
    return tuple(map(int, norm.shape)), norm.tobytes()


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _normalize_mask(np.asarray(mask, dtype=bool))
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for rot in range(4):
        turned = np.rot90(base, rot)
        for cand in (turned, np.fliplr(turned)):
            norm = _normalize_mask(cand)
            sig = _mask_signature(norm)
            if sig not in seen:
                seen.add(sig)
                variants.append(norm)
    return variants


def _ring_patch(palette: list[int], offset: int) -> np.ndarray:
    patch = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=int)
    for r in range(BLOCK_SIZE):
        for c in range(BLOCK_SIZE):
            depth = min(r, c, BLOCK_SIZE - 1 - r, BLOCK_SIZE - 1 - c)
            patch[r, c] = int(palette[(depth + offset) % len(palette)])
    return patch


def _find_panel_box(grid: np.ndarray) -> tuple[int, int, int, int]:
    teal_components = _components_mask(grid == TEAL, connectivity=4)
    assert teal_components, "expected a teal legend panel"
    ranked = sorted(teal_components, key=lambda coords: coords.shape[0], reverse=True)
    largest = ranked[0]
    if len(ranked) > 1:
        assert largest.shape[0] > ranked[1].shape[0], "legend panel must be uniquely largest teal component"
    box = _bbox_coords(largest)
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 7 and c1 - c0 + 1 >= 9
    return box


def _extract_legend(grid: np.ndarray, panel_box: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    panel = _crop_bbox(grid, panel_box)
    colors = sorted(int(v) for v in np.unique(panel) if int(v) not in {BLACK, TEAL})
    assert 3 <= len(colors) <= 4, "legend must contain 3 or 4 colors"

    legend: dict[int, np.ndarray] = {}
    for color in colors:
        comps = _components_mask(panel == color, connectivity=4)
        assert len(comps) == 1, "each legend color must appear once in the panel"
        mask = np.zeros(panel.shape, dtype=bool)
        coords = comps[0]
        mask[coords[:, 0], coords[:, 1]] = True
        norm = _normalize_mask(mask)
        area = int(norm.sum())
        assert 4 <= area <= 5
        assert len(_d4_variants(norm)) == 8, "legend shapes must avoid rigid symmetries"
        legend[color] = norm

    orbits = []
    for color, mask in legend.items():
        orbit = {_mask_signature(v) for v in _d4_variants(mask)}
        for prev_color, prev_orbit in orbits:
            assert orbit.isdisjoint(prev_orbit), f"legend shapes {prev_color} and {color} must not be D4-equivalent"
        orbits.append((color, orbit))
    return legend


def _find_palette_row(grid: np.ndarray, panel_box: tuple[int, int, int, int], legend_colors: set[int]) -> tuple[int, list[int]]:
    r0, _c0, r1, _c1 = panel_box
    candidates: list[tuple[int, list[int]]] = []
    legend_list = sorted(legend_colors)
    for r in range(grid.shape[0]):
        if r0 <= r <= r1:
            continue
        row = grid[r]
        cols = np.flatnonzero(np.isin(row, legend_list))
        if len(cols) != len(legend_colors):
            continue
        values = [int(row[c]) for c in cols]
        if set(values) != legend_colors or len(set(values)) != len(values):
            continue
        if any(v == GREY or v == ORANGE or v == TEAL for v in values):
            continue
        if len(cols) > 1 and not np.all(np.diff(cols) == 2):
            continue
        span = row[cols[0] : cols[-1] + 1]
        if not all(int(span[idx]) == values[idx // 2] for idx in range(0, len(span), 2)):
            continue
        if not np.all(span[1::2] == BLACK):
            continue
        candidates.append((int(r), values))
    assert len(candidates) == 1, "expected exactly one palette row"
    return candidates[0]


def _find_block_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    comps = _components_mask(grid != BLACK, connectivity=4)
    boxes: list[tuple[int, int, int, int]] = []
    for coords in comps:
        box = _bbox_coords(coords)
        patch = _crop_bbox(grid, box)
        colors = {int(v) for v in np.unique(patch)}
        if patch.shape == (BLOCK_SIZE, BLOCK_SIZE) and colors.issubset({GREY, ORANGE}) and BLACK not in colors:
            assert int(coords.shape[0]) == BLOCK_SIZE * BLOCK_SIZE
            boxes.append(box)
    assert 3 <= len(boxes) <= 4, "expected 3 or 4 encoded blocks"
    return sorted(boxes)


def _match_block_color(block: np.ndarray, legend: dict[int, np.ndarray]) -> int:
    orange = block == ORANGE
    assert orange.any() and set(np.unique(block)).issubset({GREY, ORANGE})
    marker = _normalize_mask(orange)
    marker_sig = _mask_signature(marker)
    matches = []
    for color, shape in legend.items():
        if marker_sig in {_mask_signature(v) for v in _d4_variants(shape)}:
            matches.append(int(color))
    assert len(matches) == 1, "each block marker must match exactly one legend shape"
    return matches[0]


def _parse_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    panel_box = _find_panel_box(g)
    legend = _extract_legend(g, panel_box)
    _palette_row, palette = _find_palette_row(g, panel_box, set(legend))
    block_boxes = _find_block_boxes(g)
    return {"grid": g, "panel_box": panel_box, "legend": legend, "palette": palette, "block_boxes": block_boxes}


def main(grid: np.ndarray) -> np.ndarray:
    """Use the legend shape-to-color mapping to pick a palette phase, then cyclically recolor each 7x7 encoded block by concentric layers.

    Input structure:
    - A black background contains a teal legend panel near the upper left, with one small 4-connected colored shape for each legend color.
    - Below the panel is a single palette row made of those same colors in a left-to-right order with black gaps.
    - Elsewhere are 3 or 4 separate 7x7 blocks filled with grey and carrying a centered orange marker shape.

    Rule:
    - For each encoded block, match its orange marker shape to exactly one legend shape up to rotation or reflection, and take that legend color.
    - Find that color's index in the palette row; this index is the starting phase for the block.
    - Recolor the whole 7x7 block by concentric rings, where ring depth d uses palette[(d + phase) mod palette_length], overwriting both the grey fill and the orange marker.
    - Keep the legend panel, the palette row, and all black background cells unchanged.

    Key insight:
    - The legend does not directly choose the final block color; it chooses a phase offset into the visible palette cycle, so equal ring structures can shift color-by-color between blocks.
    """
    parsed = _parse_instance(np.asarray(grid, dtype=int))
    out = parsed["grid"].copy()
    palette = parsed["palette"]
    legend = parsed["legend"]

    for box in parsed["block_boxes"]:
        r0, c0, r1, c1 = box
        block = parsed["grid"][r0 : r1 + 1, c0 : c1 + 1]
        color = _match_block_color(block, legend)
        phase = palette.index(color)
        out[r0 : r1 + 1, c0 : c1 + 1] = _ring_patch(palette, phase)

    return out


def _random_variant(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if int(rng.integers(2)):
        out = np.fliplr(out)
    return out.copy()


def _place_shape_in_slot(shape: np.ndarray, slot_shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    slot_h, slot_w = slot_shape
    h, w = shape.shape
    dr = int(rng.integers(slot_h - h + 1))
    dc = int(rng.integers(slot_w - w + 1))
    patch = np.zeros(slot_shape, dtype=bool)
    patch[dr : dr + h, dc : dc + w] = shape
    return patch


def _paint_bool(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        grid[r0 + rr, c0 + cc] = int(color)


def _make_block(marker_shape: np.ndarray) -> np.ndarray:
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), GREY, dtype=int)
    h, w = marker_shape.shape
    r0 = (BLOCK_SIZE - h) // 2
    c0 = (BLOCK_SIZE - w) // 2
    view = block[r0 : r0 + h, c0 : c0 + w]
    view[marker_shape] = ORANGE
    return block


def _expanded(mask: np.ndarray, pad: int = 1) -> np.ndarray:
    out = mask.copy()
    for _ in range(pad):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        grown[1:, 1:] |= out[:-1, :-1]
        grown[1:, :-1] |= out[:-1, 1:]
        grown[:-1, 1:] |= out[1:, :-1]
        grown[:-1, :-1] |= out[1:, 1:]
        out = grown
    return out


def _choose_block_positions(
    shape: tuple[int, int],
    occupied: np.ndarray,
    block_count: int,
    panel_box: tuple[int, int, int, int],
    palette_row: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = shape
    pr0, pc0, pr1, pc1 = panel_box
    candidates = [
        (r, c)
        for r in range(h - BLOCK_SIZE + 1)
        for c in range(w - BLOCK_SIZE + 1)
        if (r > palette_row + 1 or c > pc1 + 1)
    ]
    rng.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    current = occupied.copy()
    for r, c in candidates:
        blocked = _expanded(current, pad=1)
        if blocked[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE].any():
            continue
        chosen.append((int(r), int(c)))
        current[r : r + BLOCK_SIZE, c : c + BLOCK_SIZE] = True
        if len(chosen) == block_count:
            break

    if len(chosen) != block_count:
        return None
    if not any(c > pc1 + 1 for _r, c in chosen):
        return None
    if not any(r > palette_row + 1 for r, _c in chosen):
        return None
    return chosen


def _build_output_from_meta(grid: np.ndarray, block_specs: list[dict], palette: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in block_specs:
        r0, c0 = spec["top_left"]
        phase = palette.index(int(spec["color"]))
        out[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = _ring_patch(palette, phase)
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    legend_shapes: dict[int, np.ndarray],
    palette: list[int],
    block_specs: list[dict],
) -> None:
    parsed = _parse_instance(grid)
    assert parsed["palette"] == palette
    assert set(parsed["legend"]) == set(legend_shapes)
    assert len(parsed["block_boxes"]) == len(block_specs)
    assert set(spec["color"] for spec in block_specs) == set(legend_shapes)

    sorted_specs = sorted(block_specs, key=lambda spec: (spec["top_left"][0], spec["top_left"][1]))
    parsed_matches = []
    for box in parsed["block_boxes"]:
        r0, c0, r1, c1 = box
        parsed_matches.append(_match_block_color(grid[r0 : r1 + 1, c0 : c1 + 1], parsed["legend"]))
    assert parsed_matches == [int(spec["color"]) for spec in sorted_specs]

    ring_sigs = {_mask_signature(_ring_patch(palette, idx) != BLACK) for idx in range(len(palette))}
    assert len(ring_sigs) == 1
    color_sigs = {tuple(map(int, _ring_patch(palette, idx).ravel())) for idx in range(len(palette))}
    assert len(color_sigs) == len(palette), "different phases must yield different colored outputs"

    assert np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(300):
        height = int(rng.integers(24, 30))
        width = int(rng.integers(24, 30))
        grid = np.zeros((height, width), dtype=int)

        legend_count = 4 if int(rng.integers(2)) else 3
        block_count = legend_count if legend_count == 4 or int(rng.integers(2)) else 4

        chosen_colors = list(rng.choice(np.array(LEGEND_COLOR_POOL), size=legend_count, replace=False))
        chosen_shapes = [SHAPE_POOL[idx].copy() for idx in rng.choice(len(SHAPE_POOL), size=legend_count, replace=False)]

        panel_r0 = int(rng.integers(0, 2))
        panel_c0 = int(rng.integers(0, 2))
        panel_r1 = panel_r0 + PANEL_SHAPE[0] - 1
        panel_c1 = panel_c0 + PANEL_SHAPE[1] - 1
        if panel_r1 >= height or panel_c1 >= width:
            continue
        grid[panel_r0 : panel_r1 + 1, panel_c0 : panel_c1 + 1] = TEAL

        slot_indices = list(rng.choice(len(PANEL_SLOT_ORIGINS), size=legend_count, replace=False))
        rng.shuffle(slot_indices)
        legend_shapes: dict[int, np.ndarray] = {}
        for color, base_shape, slot_idx in zip(chosen_colors, chosen_shapes, slot_indices):
            legend_shape = _random_variant(base_shape, rng)
            legend_shapes[int(color)] = legend_shape
            slot_r, slot_c = PANEL_SLOT_ORIGINS[slot_idx]
            placed = _place_shape_in_slot(legend_shape, PANEL_SLOT_SHAPE, rng)
            _paint_bool(grid, placed, panel_r0 + slot_r, panel_c0 + slot_c, int(color))

        palette = [int(v) for v in rng.permutation(np.array(chosen_colors))]
        palette_len = 2 * legend_count - 1
        palette_r = panel_r1 + 2
        if palette_r >= height:
            continue
        palette_c0 = panel_c0 + int(rng.integers(0, PANEL_SHAPE[1] - palette_len + 1))
        for idx, color in enumerate(palette):
            grid[palette_r, palette_c0 + 2 * idx] = int(color)

        occupied = grid != BLACK
        positions = _choose_block_positions(grid.shape, occupied, block_count, (panel_r0, panel_c0, panel_r1, panel_c1), palette_r, rng)
        if positions is None:
            continue

        assignments = chosen_colors.copy()
        if block_count > legend_count:
            remaining = [int(rng.choice(np.array(chosen_colors)))]
            assignments.extend(remaining)
        rng.shuffle(assignments)

        used_variant_sigs: dict[int, set[tuple[tuple[int, int], bytes]]] = {int(color): set() for color in chosen_colors}
        block_specs: list[dict] = []
        ok = True
        for (r0, c0), color in zip(positions, assignments):
            variants = _d4_variants(legend_shapes[int(color)])
            rng.shuffle(variants)
            marker_shape = None
            for candidate in variants:
                sig = _mask_signature(candidate)
                if sig not in used_variant_sigs[int(color)]:
                    marker_shape = candidate
                    used_variant_sigs[int(color)].add(sig)
                    break
            if marker_shape is None:
                marker_shape = variants[0]
            block = _make_block(marker_shape)
            if np.any(grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] != BLACK):
                ok = False
                break
            grid[r0 : r0 + BLOCK_SIZE, c0 : c0 + BLOCK_SIZE] = block
            block_specs.append(
                {
                    "top_left": (int(r0), int(c0)),
                    "color": int(color),
                    "marker_shape": marker_shape.copy(),
                }
            )
        if not ok:
            continue

        expected = _build_output_from_meta(grid, block_specs, palette)
        _validate_generated(grid, expected, legend_shapes, palette, block_specs)
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
