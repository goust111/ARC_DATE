from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np


BG = 0
GREY = 5
PAIR_COUNT_CHOICES = (2, 3)
PIECE_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_RETRIES = 300


@dataclass(frozen=True)
class PieceInfo:
    color: int
    side: str
    profile: tuple[int, ...]
    axis_len: int
    min_extent: int
    max_extent: int
    top_left: tuple[int, int]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _legend_rows(patch: np.ndarray, bg: int) -> list[tuple[int, int, int]] | None:
    h, w = patch.shape
    if w != 5 or h < 4:
        return None

    outer = np.concatenate(
        [patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]]
    )
    border_colors = {int(v) for v in outer}
    if len(border_colors) != 1:
        return None
    border_color = next(iter(border_colors))
    if border_color == bg:
        return None

    interior = patch[1:-1, 1:-1]
    if interior.shape[1] != 3 or interior.shape[0] < 2:
        return None
    if np.any(interior == bg) or np.any(interior == border_color):
        return None

    rows = [tuple(int(x) for x in interior[r]) for r in range(interior.shape[0])]
    if len({frozenset((a, b)) for a, b, _ in rows}) != len(rows):
        return None
    return rows


def _contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size) and int(indices[-1] - indices[0] + 1) == int(indices.size)


def _analyze_piece(mask: np.ndarray, color: int, top_left: tuple[int, int]) -> PieceInfo:
    h, w = mask.shape
    candidates: list[PieceInfo] = []

    row_runs = []
    row_ok = True
    for r in range(h):
        cols = np.flatnonzero(mask[r])
        if not _contiguous(cols):
            row_ok = False
            break
        row_runs.append((int(cols[0]), int(cols[-1])))
    if row_ok:
        starts = [s for s, _ in row_runs]
        ends = [e for _, e in row_runs]
        widths = [e - s + 1 for s, e in row_runs]
        if len(set(starts)) > 1 and all(e == w - 1 for e in ends):
            profile = tuple(s - min(starts) for s in starts)
            candidates.append(
                PieceInfo(color, "left", profile, h, min(widths), w, top_left)
            )
        if len(set(ends)) > 1 and all(s == 0 for s in starts):
            profile = tuple(e - min(ends) for e in ends)
            candidates.append(
                PieceInfo(color, "right", profile, h, min(widths), w, top_left)
            )

    col_runs = []
    col_ok = True
    for c in range(w):
        rows = np.flatnonzero(mask[:, c])
        if not _contiguous(rows):
            col_ok = False
            break
        col_runs.append((int(rows[0]), int(rows[-1])))
    if col_ok:
        tops = [s for s, _ in col_runs]
        bottoms = [e for _, e in col_runs]
        heights = [e - s + 1 for s, e in col_runs]
        if len(set(tops)) > 1 and all(b == h - 1 for b in bottoms):
            profile = tuple(t - min(tops) for t in tops)
            candidates.append(
                PieceInfo(color, "top", profile, w, min(heights), h, top_left)
            )
        if len(set(bottoms)) > 1 and all(t == 0 for t in tops):
            profile = tuple(b - min(bottoms) for b in bottoms)
            candidates.append(
                PieceInfo(color, "bottom", profile, w, min(heights), h, top_left)
            )

    assert len(candidates) == 1, f"piece must have exactly one toothed side, got {len(candidates)}"
    return candidates[0]


def _merged_shape(a: PieceInfo, b: PieceInfo) -> tuple[int, int] | None:
    if a.profile != b.profile or a.axis_len != b.axis_len:
        return None
    if a.side == "right" and b.side == "left":
        return a.axis_len, a.min_extent + b.max_extent
    if a.side == "left" and b.side == "right":
        return a.axis_len, a.max_extent + b.min_extent
    if a.side == "bottom" and b.side == "top":
        return a.min_extent + b.max_extent, a.axis_len
    if a.side == "top" and b.side == "bottom":
        return a.max_extent + b.min_extent, a.axis_len
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair toothed rectangular fragments, rebuild each full rectangle, and recolor it from the legend.

    Input structure:
    - bg = most common color. Evenly many monochrome fragments are scattered on the grid.
    - Each fragment is a connected orthogonal shape with exactly one toothed side and three straight sides.
    - One multicolor legend is enclosed by a grey rectangular border; each interior row has exactly three pixels: fragment color A, fragment color B, and output color.

    Rule:
    - Detect the legend and read every row as an unordered color-pair → output-color mapping.
    - For every non-legend fragment, identify which side is toothed and extract its normalized tooth profile along that side.
    - For each legend row, take the two fragments with those colors, verify that their toothed sides face opposite directions and share the same normalized profile, and infer the size of the full rectangle they would form when interlocked.
    - Place that reconstructed rectangle at the lexicographically earlier fragment position (row first, then column) and fill it with the legend's output color. Remove the scattered source fragments. Keep the legend unchanged.

    Key insight:
    - The toothed edge can be reduced to a one-dimensional profile of contiguous row or column extents; matching opposite-facing fragments with the same normalized profile recovers the missing rectangle dimensions, while the legend provides the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    comps = _components(g != bg)
    legend_patch = None
    legend_bbox = None
    legend_rows = None
    pieces: dict[int, PieceInfo] = {}

    for comp in comps:
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        maybe_legend = _legend_rows(patch, bg)
        if maybe_legend is not None:
            assert legend_rows is None, "expected a unique legend"
            legend_rows = maybe_legend
            legend_patch = patch.copy()
            legend_bbox = bbox
            continue

        colors = {int(v) for v in patch[patch != bg]}
        assert len(colors) == 1, "each fragment must be monochrome"
        color = next(iter(colors))
        assert color not in pieces, "fragment colors must be unique"
        mask = _component_mask(comp, bbox)
        pieces[color] = _analyze_piece(mask, color, (r0, c0))

    assert legend_rows is not None and legend_patch is not None and legend_bbox is not None

    out = np.full_like(g, bg)
    r0, c0, r1, c1 = legend_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = legend_patch

    used_colors: set[int] = set()
    for color_a, color_b, out_color in legend_rows:
        assert color_a in pieces and color_b in pieces, "legend must reference existing fragments"
        piece_a = pieces[color_a]
        piece_b = pieces[color_b]
        merged = _merged_shape(piece_a, piece_b)
        assert merged is not None, "legend pair must interlock structurally"

        anchor = min(piece_a.top_left, piece_b.top_left)
        height, width = merged
        ar, ac = anchor
        assert np.all(out[ar : ar + height, ac : ac + width] == bg), "reconstructed rectangles must not overlap"
        out[ar : ar + height, ac : ac + width] = int(out_color)
        used_colors.update((color_a, color_b))

    assert used_colors == set(pieces), "every fragment must appear in the legend"
    return out


def _random_profile(rng: random.Random, length: int, depth: int) -> tuple[int, ...]:
    for _ in range(100):
        profile = tuple(rng.randint(0, depth) for _ in range(length))
        if len(set(profile)) < 2:
            continue
        if sum(a != b for a, b in zip(profile, profile[1:])) < 2:
            continue
        return profile
    raise RuntimeError("failed to sample a non-degenerate tooth profile")


def _build_vertical_pair(profile: tuple[int, ...], left_base: int, right_base: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    h = len(profile)
    depth = max(profile)
    left = np.zeros((h, left_base + depth), dtype=bool)
    right = np.zeros((h, right_base + depth), dtype=bool)
    for r, p in enumerate(profile):
        left[r, : left_base + p] = True
        right[r, p:] = True
    return left, right, (h, left_base + right_base + depth)


def _build_horizontal_pair(profile: tuple[int, ...], top_base: int, bottom_base: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    w = len(profile)
    depth = max(profile)
    top = np.zeros((top_base + depth, w), dtype=bool)
    bottom = np.zeros((bottom_base + depth, w), dtype=bool)
    for c, p in enumerate(profile):
        top[: top_base + p, c] = True
        bottom[p:, c] = True
    return top, bottom, (top_base + bottom_base + depth, w)


def _legend_patch(rows: list[tuple[int, int, int]]) -> np.ndarray:
    patch = np.full((len(rows) + 2, 5), GREY, dtype=int)
    for i, (a, b, out_color) in enumerate(rows, start=1):
        patch[i, 1:4] = [a, b, out_color]
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    oh, ow = occupied.shape
    if r0 < 1 or c0 < 1 or r0 + h > oh - 1 or c0 + w > ow - 1:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(oh, r0 + h + pad)
    cc1 = min(ow, c0 + w + pad)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _place_box(
    occupied: np.ndarray,
    shape: tuple[int, int],
    rng: random.Random,
    row_min: int,
    row_max: int,
    col_min: int = 1,
    col_max: int | None = None,
) -> tuple[int, int] | None:
    h, w = shape
    oh, ow = occupied.shape
    if col_max is None:
        col_max = ow - w - 1
    row_max = min(row_max, oh - h - 1)
    col_max = min(col_max, ow - w - 1)
    if row_min > row_max or col_min > col_max:
        return None

    candidates = [
        (r, c)
        for r in range(row_min, row_max + 1)
        for c in range(col_min, col_max + 1)
        if _can_place(occupied, r, c, h, w)
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _reserve(occupied: np.ndarray, pos: tuple[int, int], shape: tuple[int, int], pad: int = 1) -> None:
    r0, c0 = pos
    h, w = shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, pos: tuple[int, int], color: int) -> None:
    r0, c0 = pos
    h, w = mask.shape
    grid[r0 : r0 + h, c0 : c0 + w][mask] = color


def _paint_patch(grid: np.ndarray, patch: np.ndarray, pos: tuple[int, int]) -> None:
    r0, c0 = pos
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _choose_output_colors(
    rng: random.Random,
    piece_pairs: list[tuple[int, int]],
    piece_colors: list[int],
) -> list[int]:
    unused = [c for c in PIECE_COLOR_POOL if c not in piece_colors]
    outputs: list[int] = []
    for a, b in piece_pairs:
        pool = [c for c in unused if c not in (a, b)]
        if not pool:
            pool = [c for c in PIECE_COLOR_POOL if c not in (a, b)]
        outputs.append(rng.choice(pool))
    return outputs


def _generate_pair_specs(rng: random.Random, pair_count: int) -> list[dict]:
    specs: list[dict] = []
    used_signatures: set[tuple[str, tuple[int, ...]]] = set()
    for _ in range(pair_count):
        for _ in range(100):
            orientation = rng.choice(("vertical", "horizontal"))
            if orientation == "vertical":
                height = rng.randint(4, 7)
                depth = rng.randint(1, 2)
                profile = _random_profile(rng, height, depth)
                signature = ("vertical", profile)
                if signature in used_signatures:
                    continue
                left_base = rng.randint(2, 4)
                right_base = rng.randint(2, 4)
                a_mask, b_mask, merged_shape = _build_vertical_pair(profile, left_base, right_base)
                try:
                    a_info = _analyze_piece(a_mask, 1, (0, 0))
                    b_info = _analyze_piece(b_mask, 2, (0, 0))
                except AssertionError:
                    continue
                if a_info.side != "right" or b_info.side != "left":
                    continue
                specs.append(
                    {
                        "masks": (a_mask, b_mask),
                        "merged_shape": merged_shape,
                        "anchor_idx": rng.randrange(2),
                    }
                )
                used_signatures.add(signature)
                break
            width = rng.randint(4, 7)
            depth = rng.randint(1, 2)
            profile = _random_profile(rng, width, depth)
            signature = ("horizontal", profile)
            if signature in used_signatures:
                continue
            top_base = rng.randint(2, 4)
            bottom_base = rng.randint(2, 4)
            a_mask, b_mask, merged_shape = _build_horizontal_pair(profile, top_base, bottom_base)
            try:
                a_info = _analyze_piece(a_mask, 1, (0, 0))
                b_info = _analyze_piece(b_mask, 2, (0, 0))
            except AssertionError:
                continue
            if a_info.side != "bottom" or b_info.side != "top":
                continue
            specs.append(
                {
                    "masks": (a_mask, b_mask),
                    "merged_shape": merged_shape,
                    "anchor_idx": rng.randrange(2),
                }
            )
            used_signatures.add(signature)
            break
        else:
            raise RuntimeError("failed to generate pair specification")
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    grid_h = rng.randint(24 if pair_count == 3 else 20, 30)
    grid_w = rng.randint(24 if pair_count == 3 else 20, 30)

    pair_specs = _generate_pair_specs(rng, pair_count)
    piece_colors = rng.sample(PIECE_COLOR_POOL, 2 * pair_count)
    piece_pairs = [(piece_colors[2 * i], piece_colors[2 * i + 1]) for i in range(pair_count)]
    output_colors = _choose_output_colors(rng, piece_pairs, piece_colors)

    legend_rows = [
        (piece_pairs[i][0], piece_pairs[i][1], output_colors[i])
        for i in range(pair_count)
    ]
    rng.shuffle(legend_rows)
    legend = _legend_patch(legend_rows)

    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    split_row = grid_h // 2

    legend_pos = _place_box(
        occupied,
        legend.shape,
        rng,
        row_min=split_row,
        row_max=grid_h - legend.shape[0] - 1,
    )
    if legend_pos is None:
        return None
    _reserve(occupied, legend_pos, legend.shape)

    anchor_positions: list[tuple[int, int]] = []
    for spec in pair_specs:
        slot_h, slot_w = spec["merged_shape"]
        row_max = split_row - slot_h - 1
        pos = _place_box(occupied, (slot_h, slot_w), rng, row_min=1, row_max=row_max)
        if pos is None:
            return None
        anchor_positions.append(pos)
        _reserve(occupied, pos, (slot_h, slot_w))

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    expected = np.full((grid_h, grid_w), BG, dtype=int)
    _paint_patch(inp, legend, legend_pos)
    _paint_patch(expected, legend, legend_pos)

    color_to_output = {frozenset((a, b)): out_color for a, b, out_color in legend_rows}

    for i, spec in enumerate(pair_specs):
        colors = piece_pairs[i]
        masks = spec["masks"]
        anchor_idx = spec["anchor_idx"]
        partner_idx = 1 - anchor_idx
        anchor_pos = anchor_positions[i]

        _paint_mask(inp, masks[anchor_idx], anchor_pos, colors[anchor_idx])

        partner_mask = masks[partner_idx]
        partner_pos = _place_box(
            occupied,
            partner_mask.shape,
            rng,
            row_min=split_row,
            row_max=grid_h - partner_mask.shape[0] - 1,
        )
        if partner_pos is None:
            return None
        _reserve(occupied, partner_pos, partner_mask.shape)
        _paint_mask(inp, partner_mask, partner_pos, colors[partner_idx])

        out_color = color_to_output[frozenset(colors)]
        oh, ow = spec["merged_shape"]
        ar, ac = anchor_pos
        expected[ar : ar + oh, ac : ac + ow] = out_color

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid toothed-fragment puzzle")
