from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_HELPER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_HELPER_DIR) not in sys.path:
    sys.path.append(str(_HELPER_DIR))

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


GLYPH_SIZE = 3
LEGEND_BAND_H = GLYPH_SIZE + 2
LEGEND_ROW = 1
LEGEND_COUNT_RANGE = (3, 5)
CANVAS_COUNT_RANGE = (4, 7)
CANVAS_H_RANGE = (10, 20)
GRID_W_RANGE = (18, 30)
LEGEND_GAP_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _mask_from_bits(bits: int) -> np.ndarray:
    return np.array([(bits >> k) & 1 for k in range(GLYPH_SIZE * GLYPH_SIZE)], dtype=bool).reshape(
        GLYPH_SIZE, GLYPH_SIZE
    )


def _rotation_variants(mask: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(mask, k) for k in range(4)]


def _orbit_key(mask: np.ndarray) -> bytes:
    return min(variant.astype(np.uint8).tobytes() for variant in _rotation_variants(mask))


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _build_glyph_library() -> list[np.ndarray]:
    reps: dict[bytes, np.ndarray] = {}
    for bits in range(1, 1 << (GLYPH_SIZE * GLYPH_SIZE)):
        mask = _mask_from_bits(bits)
        area = int(mask.sum())
        if area < 4 or area > 6:
            continue
        if not _touches_all_sides(mask):
            continue
        if not _is_connected(mask):
            continue
        if len({variant.astype(np.uint8).tobytes() for variant in _rotation_variants(mask)}) != 4:
            continue
        reps.setdefault(_orbit_key(mask), mask.copy())
    library = sorted(reps.values(), key=lambda m: (int(m.sum()), m.astype(np.uint8).tobytes()))
    if len(library) < 8:
        raise RuntimeError("glyph library is unexpectedly small")
    return library


GLYPH_LIBRARY = _build_glyph_library()


def _render_mask(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, int(bg), dtype=int)
    patch[mask] = int(fg)
    return patch


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    rows = [
        int(r)
        for r in range(grid.shape[0])
        if int(grid[r, 0]) != int(bg) and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    if len(rows) != 1:
        raise ValueError("expected exactly one full-width separator row")
    return rows[0]


def _extract_sprites(region: np.ndarray, bg: int) -> list[dict]:
    sprites: list[dict] = []
    for coords in components_mask(region != int(bg), connectivity=4):
        r0, c0, r1, c1 = map(int, bbox_coords(coords))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        shifted = coords - np.array([r0, c0], dtype=int)
        mask[shifted[:, 0], shifted[:, 1]] = True
        colors = {int(region[int(r), int(c)]) for r, c in coords}
        if len(colors) != 1:
            raise ValueError("each glyph must be monochrome")
        sprites.append({"bbox": (r0, c0, r1, c1), "mask": mask, "color": colors.pop()})
    return sprites


def _match_legend(mask: np.ndarray, legend_masks: list[np.ndarray]) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for idx, legend_mask in enumerate(legend_masks):
        if legend_mask.shape != mask.shape:
            continue
        for rot, rotated in enumerate(_rotation_variants(legend_mask)):
            if np.array_equal(rotated, mask):
                matches.append((int(idx), int(rot)))
    if len(matches) != 1:
        raise ValueError(f"legend match should be unique, got {matches}")
    return matches[0]


def _apply_replacement(
    base: np.ndarray,
    legend_masks: list[np.ndarray],
    placements: list[dict],
    glyph_color: int,
    bg: int,
    *,
    keep_last: bool = False,
    ignore_rotation: bool = False,
) -> np.ndarray:
    out = np.array(base, copy=True)
    legend_count = len(legend_masks)
    for placement in placements:
        src_idx = int(placement["legend_index"])
        rot = int(placement["rotation"])
        if keep_last and src_idx == legend_count - 1:
            dst_idx = src_idx
        else:
            dst_idx = (src_idx + 1) % legend_count
        dst_mask = legend_masks[dst_idx] if ignore_rotation else np.rot90(legend_masks[dst_idx], rot)
        patch = _render_mask(dst_mask, glyph_color, bg)
        r0 = int(placement["r"])
        c0 = int(placement["c"])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = int(bg)
        paste(out, patch, r0, c0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each canvas glyph with the next legend glyph while preserving the canvas glyph's rotation.

    Input structure:
    - A single full-width separator row splits the grid into an upper canvas and a bottom legend band.
    - The legend band contains a left-to-right sequence of distinct monochrome glyphs, all with the same tight bbox size.
    - The canvas contains separated copies of those glyphs, each rotated by 0/90/180/270 degrees.

    Rule:
    - Read the legend glyphs in left-to-right order as a cyclic sequence.
    - For each canvas glyph, identify which legend glyph it is and which rotation was applied.
    - Replace that canvas glyph in place with the next legend glyph in the sequence, rotated by the same amount; keep the separator and legend unchanged.

    Key insight:
    - The legend encodes order, not just membership: the hidden variable is the rotation needed to match a canvas glyph to the legend, and that same rotation must be reapplied to the successor glyph.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    sep_row = _find_separator_row(g, bg)

    canvas = g[:sep_row, :]
    legend_band = g[sep_row + 1 :, :]
    legend_sprites = _extract_sprites(legend_band, bg)
    if len(legend_sprites) < 3:
        raise ValueError("expected at least three legend glyphs")
    legend_sprites.sort(key=lambda sprite: (int(sprite["bbox"][1]), int(sprite["bbox"][0])))
    legend_masks = [sprite["mask"] for sprite in legend_sprites]

    bbox_sizes = {mask.shape for mask in legend_masks}
    if len(bbox_sizes) != 1:
        raise ValueError("legend glyphs must share a common bbox size")

    out = g.copy()
    canvas_sprites = _extract_sprites(canvas, bg)
    canvas_sprites.sort(key=lambda sprite: (int(sprite["bbox"][0]), int(sprite["bbox"][1])))
    for sprite in canvas_sprites:
        src_idx, rot = _match_legend(sprite["mask"], legend_masks)
        dst_mask = np.rot90(legend_masks[(src_idx + 1) % len(legend_masks)], rot)
        patch = _render_mask(dst_mask, int(sprite["color"]), bg)
        r0, c0, r1, c1 = map(int, sprite["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
        paste(out, patch, r0, c0, transparent=bg)
    return out


def _build_canvas_specs(legend_count: int, canvas_count: int) -> list[dict]:
    distinct_needed = min(3, legend_count)
    other_indices = random.sample(list(range(legend_count - 1)), distinct_needed - 1)
    specs = [
        {"legend_index": legend_count - 1, "rotation": random.choice([1, 2, 3])},
        {"legend_index": other_indices[0], "rotation": 0},
    ]
    if distinct_needed == 3:
        specs.append({"legend_index": other_indices[1], "rotation": random.choice([1, 2, 3])})

    used_rotations = {int(spec["rotation"]) for spec in specs}
    for rot in range(4):
        if len(specs) >= canvas_count:
            break
        if rot not in used_rotations:
            specs.append({"legend_index": random.randrange(legend_count), "rotation": rot})
            used_rotations.add(rot)

    while len(specs) < canvas_count:
        specs.append({"legend_index": random.randrange(legend_count), "rotation": random.randrange(4)})

    random.shuffle(specs)
    if len({int(spec["legend_index"]) for spec in specs}) < distinct_needed:
        raise ValueError("canvas specs lost required source diversity")
    if len({int(spec["rotation"]) for spec in specs}) < 2:
        raise ValueError("canvas specs need multiple rotations")
    return specs


def _place_legend(
    grid: np.ndarray,
    legend_masks: list[np.ndarray],
    glyph_color: int,
    bg: int,
    *,
    left: int,
    gaps: list[int],
) -> None:
    c = int(left)
    for idx, mask in enumerate(legend_masks):
        paste(grid, _render_mask(mask, glyph_color, bg), LEGEND_ROW, c, transparent=bg)
        c += GLYPH_SIZE
        if idx < len(gaps):
            c += gaps[idx]


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    bg, glyph_color, sep_color = random.sample(PALETTE, 3)
    legend_count = random.randint(*LEGEND_COUNT_RANGE)
    legend_masks = [mask.copy() for mask in random.sample(GLYPH_LIBRARY, legend_count)]
    canvas_count = random.randint(*CANVAS_COUNT_RANGE)
    canvas_specs = _build_canvas_specs(legend_count, canvas_count)

    gaps = [random.randint(*LEGEND_GAP_RANGE) for _ in range(legend_count - 1)]
    legend_span = legend_count * GLYPH_SIZE + sum(gaps)
    width_min = max(GRID_W_RANGE[0], legend_span + 2)
    width = random.randint(width_min, GRID_W_RANGE[1])
    left_min = 1
    left_max = width - legend_span - 1
    legend_left = random.randint(left_min, left_max) if left_max >= left_min else max(0, width - legend_span)
    canvas_h = random.randint(*CANVAS_H_RANGE)
    height = canvas_h + 1 + LEGEND_BAND_H
    if max(height, width) > 30:
        raise ValueError("grid exceeds ARC size limit")

    canvas_in = np.full((canvas_h, width), bg, dtype=int)
    placements: list[dict] = []
    for spec in canvas_specs:
        src_mask = np.rot90(legend_masks[int(spec["legend_index"])], int(spec["rotation"]))
        patch = _render_mask(src_mask, glyph_color, bg)
        r0, c0 = random_free_location(canvas_in, patch, transparent=bg, border=1, padding=1, padding_connectivity=8)
        paste(canvas_in, patch, r0, c0, transparent=bg)
        placements.append(
            {
                "legend_index": int(spec["legend_index"]),
                "rotation": int(spec["rotation"]),
                "r": int(r0),
                "c": int(c0),
            }
        )

    canvas_out = _apply_replacement(
        np.full_like(canvas_in, bg),
        legend_masks,
        placements,
        glyph_color,
        bg,
    )

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)
    inp[:canvas_h, :] = canvas_in
    out[:canvas_h, :] = canvas_out
    inp[canvas_h, :] = sep_color
    out[canvas_h, :] = sep_color
    _place_legend(inp[canvas_h + 1 :, :], legend_masks, glyph_color, bg, left=legend_left, gaps=gaps)
    _place_legend(out[canvas_h + 1 :, :], legend_masks, glyph_color, bg, left=legend_left, gaps=gaps)

    if not np.array_equal(inp[canvas_h + 1 :, :], out[canvas_h + 1 :, :]):
        raise ValueError("legend band must remain unchanged")
    if np.array_equal(inp, out):
        raise ValueError("input and output should not be identical")
    if len({int(p["legend_index"]) for p in placements}) < min(3, legend_count):
        raise ValueError("canvas should cover multiple source glyphs")
    if not any(int(p["legend_index"]) == legend_count - 1 for p in placements):
        raise ValueError("canvas must include the wrap-around source glyph")
    if not any(int(p["rotation"]) != 0 for p in placements):
        raise ValueError("canvas must include a rotated glyph")
    if len({int(p["rotation"]) for p in placements}) < 2:
        raise ValueError("canvas should use multiple rotations")

    for placement in placements:
        observed = canvas_in[
            int(placement["r"]) : int(placement["r"]) + GLYPH_SIZE,
            int(placement["c"]) : int(placement["c"]) + GLYPH_SIZE,
        ] != bg
        if _match_legend(observed, legend_masks) != (
            int(placement["legend_index"]),
            int(placement["rotation"]),
        ):
            raise ValueError("generator produced a non-unique legend match")

    wrong_unrotated = _apply_replacement(
        np.full_like(canvas_in, bg),
        legend_masks,
        placements,
        glyph_color,
        bg,
        ignore_rotation=True,
    )
    if np.array_equal(wrong_unrotated, canvas_out):
        raise ValueError("rotation preservation must affect the result")

    wrong_noncyclic = _apply_replacement(
        np.full_like(canvas_in, bg),
        legend_masks,
        placements,
        glyph_color,
        bg,
        keep_last=True,
    )
    if np.array_equal(wrong_noncyclic, canvas_out):
        raise ValueError("cyclic wrap-around must affect the result")

    if _find_separator_row(inp, bg) != canvas_h:
        raise ValueError("separator row detection is ambiguous")
    if not np.array_equal(main(inp), out):
        raise ValueError("oracle mismatch")
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, _out = _generate_case()
            return inp
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid legend-successor puzzle")
