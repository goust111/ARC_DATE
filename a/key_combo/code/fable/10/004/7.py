from __future__ import annotations

import random
from collections import Counter

import numpy as np


LATTICE_COLOR = 1
BACKGROUND_COLOR = 0
DATA_COLOR = 5

ARRAY_ROWS = 4
ARRAY_COLS = 4
CELL_SIZE = 6
LINE_SIZE = 1
MAX_GENERATION_ATTEMPTS = 400

LEGEND_COLOR_POOL = (2, 3, 4, 6, 7, 8, 9)
OUTER_SLOTS_BY_SIDE = {
    "top": [(0, 1), (0, 2)],
    "bottom": [(ARRAY_ROWS - 1, 1), (ARRAY_ROWS - 1, 2)],
    "left": [(1, 0), (2, 0)],
    "right": [(1, ARRAY_COLS - 1), (2, ARRAY_COLS - 1)],
}
INNER_SLOTS = [(1, 1), (1, 2), (2, 1), (2, 2)]


def _bool_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


PROTOTYPE_MASKS = [
    _bool_mask(["010", "111", "010"]),
    _bool_mask(["111", "101", "111"]),
    _bool_mask(["101", "111"]),
    _bool_mask(["101", "101", "111"]),
    _bool_mask(["010", "111", "101"]),
    _bool_mask(["010", "111", "111"]),
]
ASYMMETRIC_PROTOTYPE_IDS = {2, 3, 4, 5}


def _group_consecutive(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value == prev + 1:
            prev = value
            continue
        groups.append((start, prev))
        start = prev = value
    groups.append((start, prev))
    return groups


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated.copy())
        variants.append(np.fliplr(rotated).copy())
    return variants


def _canonical_mask(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    cropped = _tight_crop(mask.astype(bool))
    forms = []
    for variant in _d4_variants(cropped):
        v = _tight_crop(variant.astype(bool))
        forms.append(tuple(tuple(int(x) for x in row) for row in v.astype(int)))
    return min(forms)


def _complete_half(mask: np.ndarray, side: str) -> np.ndarray:
    mask = mask.astype(bool)
    if side == "left":
        assert mask[:, 0].any(), "left legend must touch the border-side column"
        full = np.concatenate([np.fliplr(mask[:, 1:]), mask], axis=1)
    elif side == "right":
        assert mask[:, -1].any(), "right legend must touch the border-side column"
        full = np.concatenate([mask, np.fliplr(mask[:, :-1])], axis=1)
    elif side == "top":
        assert mask[0, :].any(), "top legend must touch the border-side row"
        full = np.concatenate([np.flipud(mask[1:, :]), mask], axis=0)
    elif side == "bottom":
        assert mask[-1, :].any(), "bottom legend must touch the border-side row"
        full = np.concatenate([mask, np.flipud(mask[:-1, :])], axis=0)
    else:
        raise ValueError(f"unknown side {side}")
    return _tight_crop(full)


def _extract_visible_half(full_mask: np.ndarray, side: str) -> np.ndarray:
    h, w = full_mask.shape
    if side == "left":
        return full_mask[:, w // 2 :].copy()
    if side == "right":
        return full_mask[:, : w // 2 + 1].copy()
    if side == "top":
        return full_mask[h // 2 :, :].copy()
    if side == "bottom":
        return full_mask[: h // 2 + 1, :].copy()
    raise ValueError(f"unknown side {side}")


def _apply_transform(mask: np.ndarray, transform_id: int) -> np.ndarray:
    transform_id = int(transform_id)
    rotated = np.rot90(mask, transform_id % 4)
    return np.fliplr(rotated) if transform_id // 4 else rotated


def _scale2(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((2, 2), dtype=int)).astype(bool)


def _downscale2(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    assert h % 2 == 0 and w % 2 == 0, "scaled sprite bbox must have even size"
    blocks = mask.reshape(h // 2, 2, w // 2, 2)
    mins = blocks.min(axis=(1, 3))
    maxs = blocks.max(axis=(1, 3))
    assert np.array_equal(mins, maxs), "scaled sprite must be a strict 2x replication"
    return mins.astype(bool)


def _cell_spans(grid: np.ndarray, lattice_color: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == lattice_color)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == lattice_color)]
    assert row_lines and col_lines, "expected visible lattice rows and columns"
    row_groups = _group_consecutive(row_lines)
    col_groups = _group_consecutive(col_lines)
    assert row_groups[0][0] == 0 and row_groups[-1][1] == grid.shape[0] - 1
    assert col_groups[0][0] == 0 and col_groups[-1][1] == grid.shape[1] - 1

    row_spans = []
    prev_end = row_groups[0][1]
    for start, end in row_groups[1:]:
        if prev_end + 1 <= start - 1:
            row_spans.append((prev_end + 1, start - 1))
        prev_end = end

    col_spans = []
    prev_end = col_groups[0][1]
    for start, end in col_groups[1:]:
        if prev_end + 1 <= start - 1:
            col_spans.append((prev_end + 1, start - 1))
        prev_end = end

    assert len(row_spans) >= 3 and len(col_spans) >= 3, "expected an interior cell array"
    return row_spans, col_spans


def _dominant_non_lattice_color(grid: np.ndarray, lattice_color: int) -> int:
    values = [int(x) for x in grid.reshape(-1) if int(x) != lattice_color]
    assert values, "expected non-lattice pixels"
    return int(Counter(values).most_common(1)[0][0])


def _paste_mask(dst: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    h, w = mask.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey interior sprite using the color of its matching half-legend after mirror completion.

    Input structure:
    - One lattice color forms a complete cell array, including the full outer border.
    - Some non-corner perimeter cells contain a colored half-legend that touches the outer border side of that cell.
    - Interior cells are either empty or contain one grey sprite that is a 2x scaled, rotated or mirrored version of one completed legend shape.

    Rule:
    - Treat each perimeter half-legend as a generator: complete it by mirroring across the touched outer side, then use the completed shape as a codebook entry.
    - For each interior grey sprite, crop it, shrink it by an exact factor of 2, and match the result to exactly one completed legend up to rotation and mirror.
    - Recolor the whole grey sprite to the matched legend color, clear every perimeter legend cell to the background color, and leave lattice pixels and empty cells unchanged.

    Key insight:
    - The perimeter objects are not directly comparable templates: the visible half must first be completed using the outer border as the symmetry reference, and only then can the interior sprites be decoded across both scale and orientation changes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border_counts = {}
    border_values = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    for color in np.unique(border_values):
        border_counts[int(color)] = int(np.count_nonzero(border_values == color))
    full_border = 2 * g.shape[0] + 2 * g.shape[1]
    lattice_candidates = [color for color, count in border_counts.items() if count == full_border]
    assert len(lattice_candidates) == 1, "expected exactly one lattice color on the full outer border"
    lattice_color = lattice_candidates[0]

    row_spans, col_spans = _cell_spans(g, lattice_color)
    bg = _dominant_non_lattice_color(g, lattice_color)
    assert bg != DATA_COLOR, "background must differ from the grey data color"

    codebook: dict[tuple[tuple[int, ...], ...], int] = {}
    legend_cells: list[tuple[int, int, int, int]] = []
    data_cells: list[tuple[tuple[int, int, int, int], np.ndarray]] = []

    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            non_bg = {int(color) for color in np.unique(cell) if int(color) != bg}
            is_perimeter = ri in (0, len(row_spans) - 1) or ci in (0, len(col_spans) - 1)
            is_corner = ri in (0, len(row_spans) - 1) and ci in (0, len(col_spans) - 1)

            if is_perimeter:
                if not non_bg:
                    continue
                assert not is_corner, "corner legend cells are ambiguous and unsupported"
                assert DATA_COLOR not in non_bg, "grey data sprites must not appear on the perimeter"
                assert len(non_bg) == 1, "each legend cell must use one legend color"
                legend_color = next(iter(non_bg))
                legend_mask = _tight_crop(cell == legend_color)
                if ri == 0:
                    side = "top"
                elif ri == len(row_spans) - 1:
                    side = "bottom"
                elif ci == 0:
                    side = "left"
                else:
                    side = "right"

                completed = _complete_half(legend_mask, side)
                key = _canonical_mask(completed)
                assert key not in codebook, "legend shapes must be unique up to rotation/mirror"
                codebook[key] = legend_color
                legend_cells.append((r0, r1, c0, c1))
                continue

            if not non_bg:
                continue
            assert non_bg == {DATA_COLOR}, "interior non-empty cells must contain one grey sprite"
            sprite_mask = _tight_crop(cell == DATA_COLOR)
            shrunk = _downscale2(sprite_mask)
            data_cells.append(((r0, r1, c0, c1), _canonical_mask(shrunk)))

    assert codebook, "expected at least one perimeter legend"
    assert data_cells, "expected at least one grey interior sprite"

    out = g.copy()
    for r0, r1, c0, c1 in legend_cells:
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for (r0, r1, c0, c1), key in data_cells:
        assert key in codebook, "each interior sprite must match a completed legend"
        out[r0 : r1 + 1, c0 : c1 + 1][g[r0 : r1 + 1, c0 : c1 + 1] == DATA_COLOR] = codebook[key]

    return out


def _cell_origin(index: int) -> int:
    return LINE_SIZE + index * (CELL_SIZE + LINE_SIZE)


def _empty_canvas() -> np.ndarray:
    size = ARRAY_ROWS * CELL_SIZE + (ARRAY_ROWS + 1) * LINE_SIZE
    grid = np.full((size, size), LATTICE_COLOR, dtype=int)
    for ri in range(ARRAY_ROWS):
        for ci in range(ARRAY_COLS):
            r0 = _cell_origin(ri)
            c0 = _cell_origin(ci)
            grid[r0 : r0 + CELL_SIZE, c0 : c0 + CELL_SIZE] = BACKGROUND_COLOR
    return grid


def _choose_prototype_ids(count: int) -> list[int]:
    chosen = random.sample(range(len(PROTOTYPE_MASKS)), count)
    if ASYMMETRIC_PROTOTYPE_IDS.isdisjoint(chosen):
        replace_at = random.randrange(count)
        chosen[replace_at] = random.choice(tuple(ASYMMETRIC_PROTOTYPE_IDS))
        while len(set(chosen)) < count:
            chosen[replace_at] = random.choice(tuple(ASYMMETRIC_PROTOTYPE_IDS))
    return chosen


def _orient_full_mask(full_mask: np.ndarray, side: str) -> np.ndarray:
    if side in {"left", "right"}:
        return random.choice([full_mask, np.flipud(full_mask)])
    rotated = random.choice([np.rot90(full_mask, 1), np.rot90(full_mask, 3)])
    return random.choice([rotated, np.fliplr(rotated)])


def _render_legend(grid: np.ndarray, side: str, slot: tuple[int, int], full_mask: np.ndarray, color: int) -> None:
    visible = _extract_visible_half(full_mask, side)
    patch_h, patch_w = visible.shape
    r0 = _cell_origin(slot[0])
    c0 = _cell_origin(slot[1])

    if side == "left":
        rr = r0 + random.randint(1, CELL_SIZE - patch_h - 1)
        cc = c0
    elif side == "right":
        rr = r0 + random.randint(1, CELL_SIZE - patch_h - 1)
        cc = c0 + CELL_SIZE - patch_w
    elif side == "top":
        rr = r0
        cc = c0 + random.randint(1, CELL_SIZE - patch_w - 1)
    else:
        rr = r0 + CELL_SIZE - patch_h
        cc = c0 + random.randint(1, CELL_SIZE - patch_w - 1)
    _paste_mask(grid, visible, color, rr, cc)


def _render_data_sprite(
    grid: np.ndarray,
    expected: np.ndarray,
    slot: tuple[int, int],
    full_mask: np.ndarray,
    color: int,
    *,
    forced_transform: int | None = None,
) -> None:
    transform_id = random.randrange(8) if forced_transform is None else int(forced_transform)
    transformed = _apply_transform(full_mask, transform_id)
    scaled = _scale2(transformed)
    h, w = scaled.shape
    r0 = _cell_origin(slot[0]) + random.randint(0, CELL_SIZE - h)
    c0 = _cell_origin(slot[1]) + random.randint(0, CELL_SIZE - w)
    _paste_mask(grid, scaled, DATA_COLOR, r0, c0)
    _paste_mask(expected, scaled, color, r0, c0)


def _semantic_check(inp: np.ndarray, legends: list[dict], occupancy: list[int]) -> bool:
    canonical_legends = [_canonical_mask(entry["full_mask"]) for entry in legends]
    if len(set(canonical_legends)) != len(canonical_legends):
        return False
    if not any(entry["side"] in {"top", "bottom"} for entry in legends):
        return False
    if not any(entry["side"] in {"left", "right"} for entry in legends):
        return False
    if set(occupancy) - set(range(len(legends))):
        return False
    if set(occupancy) != set(range(len(legends))):
        return False
    try:
        out = main(inp)
    except AssertionError:
        return False
    unique_colors = set(int(x) for x in np.unique(out))
    return DATA_COLOR not in unique_colors


def _gen_attempt() -> np.ndarray | None:
    grid = _empty_canvas()
    expected = grid.copy()

    legend_count = random.choice([3, 4])
    selected_sides = list(OUTER_SLOTS_BY_SIDE)
    if legend_count == 3:
        selected_sides = random.sample(selected_sides, 3)
    random.shuffle(selected_sides)

    prototype_ids = _choose_prototype_ids(legend_count)
    legend_colors = random.sample(LEGEND_COLOR_POOL, legend_count)

    legends = []
    used_slots = set()
    for side, proto_id, color in zip(selected_sides, prototype_ids, legend_colors):
        slot_options = [slot for slot in OUTER_SLOTS_BY_SIDE[side] if slot not in used_slots]
        slot = random.choice(slot_options)
        used_slots.add(slot)
        full_mask = _orient_full_mask(PROTOTYPE_MASKS[proto_id], side)
        legends.append(
            {
                "side": side,
                "slot": slot,
                "proto_id": proto_id,
                "full_mask": full_mask,
                "color": color,
            }
        )
        _render_legend(grid, side, slot, full_mask, color)

    object_count = random.choice([3, 4]) if legend_count == 3 else 4
    inner_slots = random.sample(INNER_SLOTS, object_count)
    occupancy = list(range(legend_count))
    while len(occupancy) < object_count:
        occupancy.append(random.randrange(legend_count))
    random.shuffle(occupancy)

    forced_done = False
    for slot, legend_index in zip(inner_slots, occupancy):
        legend = legends[legend_index]
        forced_transform = None
        if (
            not forced_done
            and legend["proto_id"] in ASYMMETRIC_PROTOTYPE_IDS
        ):
            forced_transform = random.choice([1, 2, 3, 4, 5, 6, 7])
            forced_done = True
        _render_data_sprite(
            grid,
            expected,
            slot,
            legend["full_mask"],
            legend["color"],
            forced_transform=forced_transform,
        )

    if not forced_done:
        return None

    if not _semantic_check(grid, legends, occupancy):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 7")
