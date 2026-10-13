from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
SEP = 5
RESERVED_BLUE = 1
PALETTE = (2, 3, 4, 6, 7, 8, 9)
MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1]), "consensus must be unique"
    return int(ranked[0][0])


def _solid_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    limit = grid.shape[axis]
    for idx in range(limit):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if np.all(line == color):
            lines.append((idx, color))
    return lines


def _content_bands(separators: list[int], limit: int) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for sep in [*separators, limit]:
        assert sep > start, "separator layout must leave non-empty content bands"
        bands.append((start, sep))
        start = sep + 1
    return bands


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_band(grid: np.ndarray) -> tuple[int, int, int]:
    solid_rows = _solid_lines(grid, axis=0)
    solid_map = {idx: color for idx, color in solid_rows}
    runs = []
    r = 0
    while r < grid.shape[0]:
        if r not in solid_map:
            r += 1
            continue
        color = solid_map[r]
        r1 = r
        while r1 + 1 < grid.shape[0] and solid_map.get(r1 + 1) == color:
            r1 += 1
        runs.append((r, r1, color))
        r = r1 + 1
    candidates = [(r0, r1, color) for r0, r1, color in runs if r1 - r0 + 1 == 2 and r0 > 0 and r1 < grid.shape[0] - 1]
    assert len(candidates) == 1, f"expected exactly one 2-row separator band, found {len(candidates)}"
    return candidates[0]


def _extract_tiles(upper: np.ndarray, sep_color: int) -> tuple[np.ndarray, np.ndarray]:
    solid_rows = [idx for idx, color in _solid_lines(upper, axis=0) if color == sep_color]
    solid_cols = [idx for idx, color in _solid_lines(upper, axis=1) if color == sep_color]
    assert solid_rows and solid_cols, "tile lattice must contain separator rows and columns"

    row_bands = _content_bands(solid_rows, upper.shape[0])
    col_bands = _content_bands(solid_cols, upper.shape[1])
    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "all tiles must share one size"
    tile_h = heights.pop()
    tile_w = widths.pop()
    assert len(row_bands) * len(col_bands) >= 6, "need enough copies for consensus"

    blocks = np.empty((len(row_bands), len(col_bands), tile_h, tile_w), dtype=int)
    for br, (r0, r1) in enumerate(row_bands):
        for bc, (c0, c1) in enumerate(col_bands):
            blocks[br, bc] = upper[r0:r1, c0:c1]
    return blocks, np.array([tile_h, tile_w], dtype=int)


def _consensus_and_damage(blocks: np.ndarray, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    block_rows, block_cols, tile_h, tile_w = blocks.shape
    motif = np.empty((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            motif[r, c] = _unique_mode(blocks[:, :, r, c])

    diffs = blocks != motif
    diff_counts = diffs.reshape(block_rows, block_cols, tile_h * tile_w).sum(axis=2)
    damaged = np.argwhere(diff_counts > 0)
    assert damaged.shape[0] == 1, "expected exactly one damaged tile"
    br, bc = map(int, damaged[0])
    damage_mask = diffs[br, bc]

    assert np.all(blocks[br, bc][damage_mask] == bg), "damage must overwrite to background"
    assert np.all(motif[damage_mask] != bg), "only non-background motif cells may be damaged"
    damage_coords = [tuple(map(int, rc)) for rc in np.argwhere(damage_mask)]
    return motif, damage_coords


def _find_frame(lower: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    candidates = []
    for color in sorted(set(map(int, np.unique(lower))) - {bg}):
        for comp in _components(lower == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            sub = lower[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros(sub.shape, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.all(sub[border] == color) and np.any(sub[1:-1, 1:-1] != color):
                candidates.append((color, (r0, c0, r1, c1)))
    assert len(candidates) == 1, f"expected exactly one frame, found {len(candidates)}"
    return int(candidates[0][0]), candidates[0][1]


def _parse_lower(lower: np.ndarray, tile_h: int, tile_w: int, bg: int) -> tuple[np.ndarray, dict[tuple[int, int], int], dict[int, tuple[list[tuple[int, int]], int]], dict[int, int]]:
    frame_color, (fr0, fc0, fr1, fc1) = _find_frame(lower, bg)
    crop = lower[fr0 : fr1 + 1, fc0 : fc1 + 1].copy()
    assert crop.shape == (tile_h + 2, tile_w + 2), "frame interior must match tile size"

    placeholders: dict[tuple[int, int], int] = {}
    for lr in range(tile_h):
        for lc in range(tile_w):
            value = int(crop[1 + lr, 1 + lc])
            if value not in (bg, frame_color):
                placeholders[(lr, lc)] = value
    assert placeholders, "frame must contain placeholders"
    placeholder_colors = set(placeholders.values())

    outside_mask = np.ones(lower.shape, dtype=bool)
    outside_mask[fr0 : fr1 + 1, fc0 : fc1 + 1] = False
    comps_by_color: dict[int, list[dict]] = {}
    for color in sorted(set(map(int, np.unique(lower))) - {bg, frame_color}):
        for comp in _components((lower == color) & outside_mask):
            box = _bbox(comp)
            comps_by_color.setdefault(color, []).append({"coords": comp, "bbox": box, "area": int(comp.shape[0])})

    sprites: dict[int, tuple[list[tuple[int, int]], int]] = {}
    singleton_cells: dict[tuple[int, int], int] = {}
    for color, comps in comps_by_color.items():
        for comp in comps:
            if comp["area"] == 1:
                r, c = map(int, comp["coords"][0])
                singleton_cells[(r, c)] = color

    color_map: dict[int, int] = {}
    for color in sorted(placeholder_colors):
        comps = comps_by_color.get(color, [])
        sprite_candidates = [comp for comp in comps if comp["area"] >= 3]
        assert len(sprite_candidates) == 1, f"expected one sprite for color {color}"
        sprite = sprite_candidates[0]
        r0, c0, r1, c1 = sprite["bbox"]
        assert (r1 - r0 + 1) % 2 == 1 and (c1 - c0 + 1) % 2 == 1, "sprite bbox must have an odd center"
        center_r = (r0 + r1) // 2
        center_c = (c0 + c1) // 2
        rel = sorted((int(r - center_r), int(c - center_c)) for r, c in sprite["coords"])
        sprites[color] = (rel, sprite["area"])

        legend_singletons = [tuple(map(int, comp["coords"][0])) for comp in comps if comp["area"] == 1]
        assert len(legend_singletons) == 1, f"expected one legend source cell for color {color}"
        sr, sc = legend_singletons[0]
        neighbors = []
        for nr, nc in ((sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1)):
            other = singleton_cells.get((nr, nc))
            if other is not None and other != color:
                neighbors.append(other)
        assert len(neighbors) == 1, f"legend for color {color} must touch exactly one target singleton"
        color_map[color] = int(neighbors[0])

    assert set(color_map) == placeholder_colors
    return crop, placeholders, sprites, color_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect damaged local tile coordinates, map them into the frame interior, and stamp the matched recolored sprites only at activated placeholders.

    Input structure:
    - A repeated-tile lattice fills the upper region, with equal tiles separated by one-cell solid separators and one unique damaged copy whose motif pixels were overwritten by background.
    - A two-row solid horizontal band separates the lattice from a lower scene containing one rectangular frame, sparse single-cell placeholders inside that frame, and outside the frame one sprite plus one legend pair for each placeholder color.
    - The frame interior has the same local coordinate size as one lattice tile, so damaged tile coordinates can be reused directly inside the frame.

    Rule:
    - Recover the consensus tile by per-coordinate majority vote across all lattice copies, then locate the unique damaged tile and extract the local coordinates where it differs from the consensus.
    - Interpret those damaged local coordinates as positions inside the frame interior; a coordinate activates only if a placeholder pixel sits exactly there.
    - For each activated placeholder, find the outside sprite of the same color, center-stamp it into the frame interior at that local coordinate, recolor it by the legend mapping for that placeholder color, clip any part that falls outside the frame interior, and resolve overlaps by drawing smaller-row activations first so larger-row activations overwrite later.
    - Remove every non-activated placeholder and output only the framed lower crop.

    Key insight:
    - The damaged tile is not the thing to repair: its missing local coordinates act as an instruction code that selects which frame placeholders fire, while the shared tile/frame coordinate system makes that cross-region coupling exact.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert not np.any(g == RESERVED_BLUE), "blue must be absent from the input"

    bg = _get_bg(g)
    band_r0, band_r1, sep_color = _find_band(g)
    upper = g[:band_r0, :]
    lower = g[band_r1 + 1 :, :]

    blocks, tile_shape = _extract_tiles(upper, sep_color)
    tile_h, tile_w = map(int, tile_shape)
    _motif, damage_coords = _consensus_and_damage(blocks, bg)

    crop, placeholders, sprites, color_map = _parse_lower(lower, tile_h, tile_w, bg)

    out = crop.copy()
    out[1:-1, 1:-1] = bg

    activations = []
    for lr, lc in damage_coords:
        color = placeholders.get((lr, lc))
        if color is not None:
            activations.append((lr, lc, color))

    for lr, lc, color in sorted(activations, key=lambda item: (item[0], item[1], item[2])):
        rel_coords, _area = sprites[color]
        out_color = color_map[color]
        anchor_r = 1 + lr
        anchor_c = 1 + lc
        for dr, dc in rel_coords:
            rr = anchor_r + dr
            cc = anchor_c + dc
            if 1 <= rr < out.shape[0] - 1 and 1 <= cc < out.shape[1] - 1:
                out[rr, cc] = out_color
    return out


SPRITE_SHAPES = (
    ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
    ((0, 0), (-1, 0), (0, -1), (0, 1)),
    ((0, 0), (1, 0), (0, -1), (0, 1)),
    ((0, 0), (-1, 0), (1, 0), (0, 1)),
    ((0, 0), (-1, 0), (1, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, -1), (1, 0), (1, 1)),
)


def _choose_anchor_pair(tile_h: int, tile_w: int) -> list[tuple[int, int]]:
    corners = [
        [(0, 1), (1, min(tile_w - 2, 2))],
        [(0, tile_w - 2), (1, max(1, tile_w - 3))],
        [(tile_h - 1, 1), (tile_h - 2, min(tile_w - 2, 2))],
        [(tile_h - 1, tile_w - 2), (tile_h - 2, max(1, tile_w - 3))],
    ]
    return random.choice(corners)


def _random_motif(tile_h: int, tile_w: int, required_non_bg: list[tuple[int, int]]) -> np.ndarray:
    motif = np.zeros((tile_h, tile_w), dtype=int)
    motif_colors = random.sample(PALETTE, random.randint(2, 3))
    target_count = random.randint(max(7, len(required_non_bg) + 3), min(tile_h * tile_w // 2, 10))
    all_cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    chosen = set(required_non_bg)
    remaining = [cell for cell in all_cells if cell not in chosen]
    extra = random.sample(remaining, target_count - len(chosen))
    for idx, (r, c) in enumerate(list(chosen) + extra):
        motif[r, c] = motif_colors[idx % len(motif_colors)]
    return motif


def _tile_lattice(motif: np.ndarray, block_rows: int, block_cols: int) -> np.ndarray:
    tile_h, tile_w = motif.shape
    grid = np.full((block_rows * tile_h + block_rows - 1, block_cols * tile_w + block_cols - 1), SEP, dtype=int)
    for br in range(block_rows):
        for bc in range(block_cols):
            r0 = br * (tile_h + 1)
            c0 = bc * (tile_w + 1)
            grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = motif
    return grid


def _make_sprite(shape: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    sprite = np.zeros((3, 3), dtype=int)
    for dr, dc in shape:
        sprite[1 + dr, 1 + dc] = color
    return sprite


def _stamp_expected(tile_h: int, tile_w: int, frame_color: int, placeholders: dict[tuple[int, int], int], sprites: dict[int, tuple[tuple[tuple[int, int], ...], int]], color_map: dict[int, int], damage_coords: list[tuple[int, int]]) -> np.ndarray:
    out = np.full((tile_h + 2, tile_w + 2), BG, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    for lr, lc in sorted(damage_coords):
        color = placeholders.get((lr, lc))
        if color is None:
            continue
        rel_coords, _ = sprites[color]
        for dr, dc in rel_coords:
            rr = 1 + lr + dr
            cc = 1 + lc + dc
            if 1 <= rr < tile_h + 1 and 1 <= cc < tile_w + 1:
                out[rr, cc] = color_map[color]
    return out


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, damage_coords: list[tuple[int, int]], placeholders: dict[tuple[int, int], int], activation_coords: list[tuple[int, int]], sprite_shapes: dict[int, tuple[tuple[int, int], ...]]) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(activation_coords) < 2:
        return False
    if not any(coord not in placeholders for coord in damage_coords):
        return False
    if not any(coord not in activation_coords for coord in placeholders):
        return False

    clipped = False
    painted_sets = []
    for coord in activation_coords:
        color = placeholders[coord]
        cells = set()
        lr, lc = coord
        for dr, dc in sprite_shapes[color]:
            rr = lr + dr
            cc = lc + dc
            if 0 <= rr < expected.shape[0] - 2 and 0 <= cc < expected.shape[1] - 2:
                cells.add((rr, cc))
            else:
                clipped = True
        painted_sets.append(cells)
    if not clipped:
        return False
    overlap = False
    for i in range(len(painted_sets)):
        for j in range(i + 1, len(painted_sets)):
            if painted_sets[i] & painted_sets[j]:
                overlap = True
    return overlap


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    tile_h = random.randint(4, 5)
    tile_w = random.randint(4, 5)
    block_rows = random.randint(2, 3)
    block_cols = 4
    upper_h = block_rows * tile_h + block_rows - 1
    upper_w = block_cols * tile_w + block_cols - 1
    if upper_h + 2 + 11 > 30 or upper_w > 30:
        return None

    activation_coords = _choose_anchor_pair(tile_h, tile_w)
    extra_damage_needed = random.randint(1, 2)
    extra_placeholders_needed = random.randint(1, 3)

    frame_color = random.choice([c for c in PALETTE if c != SEP])
    placeholder_color_count = random.randint(2, 3)
    placeholder_colors = random.sample([c for c in PALETTE if c not in {frame_color, SEP}], placeholder_color_count)
    target_colors = random.sample([c for c in PALETTE if c not in {frame_color, SEP, *placeholder_colors}], placeholder_color_count)
    color_map = {src: tgt for src, tgt in zip(placeholder_colors, target_colors)}

    sprite_shapes = {color: random.choice(SPRITE_SHAPES) for color in placeholder_colors}
    required_non_bg = activation_coords.copy()
    motif = _random_motif(tile_h, tile_w, required_non_bg)
    non_bg_cells = [tuple(map(int, rc)) for rc in np.argwhere(motif != BG)]
    spare_damage = [cell for cell in non_bg_cells if cell not in activation_coords]
    if len(spare_damage) < extra_damage_needed:
        return None
    extra_damage = random.sample(spare_damage, extra_damage_needed)
    damage_coords = activation_coords + extra_damage

    all_cells = [(r, c) for r in range(tile_h) for c in range(tile_w)]
    spare_placeholders = [cell for cell in all_cells if cell not in damage_coords]
    if len(spare_placeholders) < extra_placeholders_needed:
        return None
    inactive_coords = random.sample(spare_placeholders, extra_placeholders_needed)

    placeholders: dict[tuple[int, int], int] = {}
    for idx, coord in enumerate(activation_coords):
        placeholders[coord] = placeholder_colors[idx % len(placeholder_colors)]
    for idx, coord in enumerate(inactive_coords):
        placeholders[coord] = placeholder_colors[(idx + len(activation_coords)) % len(placeholder_colors)]
    for color in placeholder_colors:
        if color not in placeholders.values():
            choices = [coord for coord in inactive_coords if placeholders[coord] != color]
            if not choices:
                return None
            placeholders[random.choice(choices)] = color

    upper = _tile_lattice(motif, block_rows, block_cols)
    damaged_br = random.randrange(block_rows)
    damaged_bc = random.randrange(block_cols)
    base_r = damaged_br * (tile_h + 1)
    base_c = damaged_bc * (tile_w + 1)
    for lr, lc in damage_coords:
        upper[base_r + lr, base_c + lc] = BG

    frame_h = tile_h + 2
    frame_w = tile_w + 2
    entry_rows = 2 if len(placeholder_colors) > 2 else 1
    lower_h = max(frame_h, 5 if entry_rows == 1 else 11)
    lower = np.zeros((lower_h, upper_w), dtype=int)

    frame_r = (lower_h - frame_h) // 2
    frame_c = 0
    lower[frame_r, frame_c : frame_c + frame_w] = frame_color
    lower[frame_r + frame_h - 1, frame_c : frame_c + frame_w] = frame_color
    lower[frame_r : frame_r + frame_h, frame_c] = frame_color
    lower[frame_r : frame_r + frame_h, frame_c + frame_w - 1] = frame_color
    for (lr, lc), color in placeholders.items():
        lower[frame_r + 1 + lr, frame_c + 1 + lc] = color

    slot_offsets = [(0, 0), (0, 4), (6, 0)]
    entries_base_c = frame_w + 2
    if entries_base_c + 7 > upper_w:
        return None
    sprites_for_expected: dict[int, tuple[tuple[tuple[int, int], ...], int]] = {}
    for idx, color in enumerate(placeholder_colors):
        panel_r, panel_dc = slot_offsets[idx]
        panel_c = entries_base_c + panel_dc
        if panel_r + 4 >= lower_h:
            return None
        lower[panel_r, panel_c] = color
        lower[panel_r, panel_c + 1] = color_map[color]
        sprite = _make_sprite(sprite_shapes[color], color)
        lower[panel_r + 2 : panel_r + 5, panel_c : panel_c + 3] = np.where(
            sprite != BG, sprite, lower[panel_r + 2 : panel_r + 5, panel_c : panel_c + 3]
        )
        sprites_for_expected[color] = (sprite_shapes[color], int(np.count_nonzero(sprite)))

    full = np.zeros((upper_h + 2 + lower_h, upper_w), dtype=int)
    full[:upper_h, :] = upper
    full[upper_h : upper_h + 2, :] = SEP
    full[upper_h + 2 :, :] = lower

    expected = _stamp_expected(tile_h, tile_w, frame_color, placeholders, sprites_for_expected, color_map, damage_coords)
    if not _semantic_checks(full, expected, damage_coords, placeholders, activation_coords, sprite_shapes):
        return None
    return full, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        attempt = _gen_attempt()
        if attempt is not None:
            return attempt[0]
    raise RuntimeError(f"generate_input: failed after {MAX_RETRIES} attempts")
