from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, crop_bbox, get_bg, paste


RNG = np.random.default_rng()
BG = 0
MAX_GENERATION_ATTEMPTS = 500
OPPOSITE = {0: 3, 1: 2, 2: 1, 3: 0}
PALETTE = list(range(1, 10))
SOLUTION_ROTATIONS = (0, 1, 2, 3)


def _rotate(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, int(k)).copy()


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    row_hits = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r])
        if len(vals) == 1 and int(vals[0]) != bg:
            row_hits.append((r, int(vals[0])))
    col_hits = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if len(vals) == 1 and int(vals[0]) != bg:
            col_hits.append((c, int(vals[0])))
    assert len(row_hits) == 1 and len(col_hits) == 1
    cross_r, row_color = row_hits[0]
    cross_c, col_color = col_hits[0]
    assert row_color == col_color
    return int(cross_r), int(cross_c), int(row_color)


def _crop_fragment(region: np.ndarray, bg: int) -> np.ndarray:
    box = bbox(region != bg)
    assert box is not None
    return crop_bbox(region, box)


def _extract_fragments(grid: np.ndarray) -> tuple[int, list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    cross_r, cross_c, _separator = _find_separator_cross(g, bg)
    quadrants = [
        g[:cross_r, :cross_c],
        g[:cross_r, cross_c + 1 :],
        g[cross_r + 1 :, :cross_c],
        g[cross_r + 1 :, cross_c + 1 :],
    ]
    return bg, [_crop_fragment(region, bg) for region in quadrants]


def _compatible_overlay(a: np.ndarray, b: np.ndarray, bg: int) -> bool:
    if a.shape != b.shape:
        return False
    conflict = (a != bg) & (b != bg) & (a != b)
    return not bool(conflict.any())


def _merge_layers(layers: list[np.ndarray], bg: int) -> np.ndarray | None:
    out = np.full_like(layers[0], bg)
    for layer in layers:
        if not _compatible_overlay(out, layer, bg):
            return None
        fill = (out == bg) & (layer != bg)
        out[fill] = layer[fill]
    return out


def _all_solutions(fragments: list[np.ndarray], bg: int) -> list[dict]:
    solutions = []
    for impostor_idx in range(4):
        basis_idx = OPPOSITE[impostor_idx]
        basis = fragments[basis_idx]
        true_indices = [idx for idx in range(4) if idx != impostor_idx]
        other_indices = [idx for idx in true_indices if idx != basis_idx]
        choices = []
        for idx in other_indices:
            options = []
            for k in SOLUTION_ROTATIONS:
                rotated = _rotate(fragments[idx], k)
                if _compatible_overlay(basis, rotated, bg):
                    options.append((k, rotated))
            if not options:
                choices = []
                break
            choices.append(options)
        if not choices:
            continue
        for combo in itertools.product(*choices):
            aligned = {basis_idx: basis}
            rotations = {basis_idx: 0}
            for idx, (k, rotated) in zip(other_indices, combo):
                aligned[idx] = rotated
                rotations[idx] = int(k)
            merged = _merge_layers([aligned[idx] for idx in true_indices], bg)
            if merged is None:
                continue
            box = bbox(merged != bg)
            assert box is not None
            output = crop_bbox(merged, box)
            solutions.append(
                {
                    "impostor_idx": int(impostor_idx),
                    "basis_idx": int(basis_idx),
                    "true_indices": tuple(true_indices),
                    "rotations": rotations,
                    "aligned": aligned,
                    "merged": merged,
                    "output": output,
                }
            )
    return solutions


def _solve_details(grid: np.ndarray) -> dict:
    bg, fragments = _extract_fragments(grid)
    solutions = _all_solutions(fragments, bg)
    assert len(solutions) == 1, f"expected unique solution, found {len(solutions)}"
    details = solutions[0]
    details["bg"] = bg
    details["fragments"] = fragments
    return details


def main(grid: np.ndarray) -> np.ndarray:
    """Identify the impostor quadrant, align the three true fragments by rotation, and union-complete the sprite in the opposite quadrant's orientation.

    Input structure:
    - A non-background solid row and a non-background solid column form a full cross divider, splitting the canvas into four nonempty quadrants.
    - Three quadrants contain incomplete multicolor copies of the same sprite, each shown in a rotated orientation on the local background.
    - The remaining quadrant contains an impostor fragment that cannot be rotation-aligned with the true system without a color conflict.

    Rule:
    - Crop the single fragment from each quadrant.
    - For each possible impostor choice, use its diagonal opposite quadrant as the orientation reference and try rotating the other two candidate true fragments into that frame.
    - The unique impostor is the one for which the three remaining fragments can be overlaid without conflicts; output the tight crop of their transparent union.

    Key insight:
    - Compatibility detection is part of the task: the odd fragment must be rejected first, and its position then determines which surviving fragment supplies the final output orientation.
    """
    g = np.asarray(grid, dtype=int)
    return _solve_details(g)["output"]


def _generate_mask(h: int, w: int) -> np.ndarray | None:
    min_area = max(9, h + w)
    for _ in range(300):
        mask = np.ones((h, w), dtype=bool)
        target_area = int(RNG.integers(min_area, h * w))
        cells = list(np.ndindex(h, w))
        RNG.shuffle(cells)
        for r, c in cells:
            if int(mask.sum()) <= target_area:
                break
            mask[r, c] = False
            if not _connected(mask) or not _touches_all_sides(mask):
                mask[r, c] = True
        if int(mask.sum()) >= min_area and int(mask.sum()) < h * w and _connected(mask) and _touches_all_sides(mask):
            return mask
    return None


def _color_sprite(mask: np.ndarray, colors: list[int]) -> np.ndarray:
    h, w = mask.shape
    sprite = np.full((h, w), BG, dtype=int)
    mode = int(RNG.integers(4))
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if mode == 0:
            idx = (rr + 2 * cc) % len(colors)
        elif mode == 1:
            idx = (2 * rr + cc + (rr > h // 2)) % len(colors)
        elif mode == 2:
            idx = ((rr // 2) + 3 * cc) % len(colors)
        else:
            idx = (rr + cc + (rr * cc) % len(colors)) % len(colors)
        sprite[rr, cc] = colors[idx]
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    present = set(map(int, sprite[mask]))
    for color in colors:
        if color not in present:
            r, c = occupied[int(RNG.integers(len(occupied)))]
            sprite[r, c] = int(color)
    return sprite


def _rotations_distinct(sprite: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        rot = _rotate(sprite, k)
        seen.add((rot.shape, rot.tobytes()))
    return len(seen) == 4


def _generate_sprite() -> np.ndarray | None:
    for _ in range(300):
        h = int(RNG.integers(4, 8))
        w = int(RNG.integers(4, 8))
        if h == w and RNG.random() < 0.7:
            continue
        mask = _generate_mask(h, w)
        if mask is None:
            continue
        color_count = int(RNG.integers(2, 5))
        colors = list(map(int, RNG.choice(PALETTE, size=color_count, replace=False)))
        sprite = _color_sprite(mask, colors)
        if _rotations_distinct(sprite):
            return sprite
    return None


def _anchor_coords(sprite: np.ndarray) -> list[tuple[int, int]]:
    mask = sprite != BG
    anchors = set()
    for side_coords in (
        [(0, c) for c in np.where(mask[0])[0]],
        [(mask.shape[0] - 1, c) for c in np.where(mask[-1])[0]],
        [(r, 0) for r in np.where(mask[:, 0])[0]],
        [(r, mask.shape[1] - 1) for r in np.where(mask[:, -1])[0]],
    ):
        assert side_coords
        anchors.add(side_coords[int(RNG.integers(len(side_coords)))])
    return sorted(anchors)


def _make_true_fragments(sprite: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]] | None:
    full_coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite != BG)]
    anchors_list = _anchor_coords(sprite)
    anchors = set(anchors_list)
    core = [coord for coord in full_coords if coord not in anchors]
    if len(core) < 6:
        return None
    subset_codes = np.array([3, 5, 6, 7], dtype=int)
    subset_probs = np.array([0.28, 0.28, 0.28, 0.16], dtype=float)
    for _ in range(300):
        RNG.shuffle(core)
        bitmasks: dict[tuple[int, int], int] = {coord: 7 for coord in anchors}
        unique_coords = core[:3]
        for idx, coord in enumerate(unique_coords):
            bitmasks[coord] = 1 << idx
        for coord in core[3:]:
            bitmasks[coord] = int(RNG.choice(subset_codes, p=subset_probs))

        fragments = [np.full_like(sprite, BG) for _ in range(3)]
        for (r, c), code in bitmasks.items():
            for idx in range(3):
                if code & (1 << idx):
                    fragments[idx][r, c] = sprite[r, c]

        if any(np.array_equal(fragment, sprite) for fragment in fragments):
            continue
        if len({fragment.tobytes() for fragment in fragments}) < 3:
            continue
        merged = _merge_layers(fragments, BG)
        if merged is None or not np.array_equal(merged, sprite):
            continue
        ok = True
        for idx in range(3):
            if not np.any((sprite != BG) & (fragments[idx] == BG)):
                ok = False
                break
            others = _merge_layers([fragments[j] for j in range(3) if j != idx], BG)
            assert others is not None
            if not np.any((fragments[idx] != BG) & (others == BG)):
                ok = False
                break
        if ok:
            return fragments, anchors_list
    return None


def _rotate_coords(coords: list[tuple[int, int]], shape: tuple[int, int], k: int) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r, c in coords:
        rr, cc = int(r), int(c)
        turns = int(k) % 4
        if turns == 0:
            out.append((rr, cc))
        elif turns == 1:
            out.append((w - 1 - cc, rr))
        elif turns == 2:
            out.append((h - 1 - rr, w - 1 - cc))
        else:
            out.append((cc, h - 1 - rr))
    return out


def _make_impostor(full_oriented: np.ndarray, anchor_coords: list[tuple[int, int]], colors: list[int]) -> np.ndarray | None:
    for _ in range(200):
        impostor = full_oriented.copy()
        edits = min(len(anchor_coords), int(RNG.integers(2, 5)))
        chosen = RNG.choice(len(anchor_coords), size=edits, replace=False)
        for idx in np.atleast_1d(chosen):
            r, c = anchor_coords[int(idx)]
            current = int(impostor[r, c])
            choices = [color for color in colors if color != current]
            impostor[r, c] = int(choices[int(RNG.integers(len(choices)))])
        if RNG.random() < 0.5:
            non_anchor = [(r, c) for r, c in np.argwhere(impostor != BG) if (int(r), int(c)) not in set(anchor_coords)]
            if non_anchor:
                r, c = non_anchor[int(RNG.integers(len(non_anchor)))]
                current = int(impostor[r, c])
                choices = [color for color in colors if color != current]
                impostor[r, c] = int(choices[int(RNG.integers(len(choices)))])
        if _touches_all_sides(impostor != BG):
            return impostor
    return None


def _quadrant_origin(index: int, quadrant_h: int, quadrant_w: int) -> tuple[int, int]:
    if index == 0:
        return 0, 0
    if index == 1:
        return 0, quadrant_w + 1
    if index == 2:
        return quadrant_h + 1, 0
    return quadrant_h + 1, quadrant_w + 1


def _sample_offset(room: int) -> int:
    assert room >= 2
    if room == 2:
        return 1
    return int(RNG.integers(1, room))


def _build_grid(fragments: list[np.ndarray], separator: int) -> np.ndarray:
    max_h = max(fragment.shape[0] for fragment in fragments)
    max_w = max(fragment.shape[1] for fragment in fragments)
    quadrant_h = int(RNG.integers(max_h + 2, max_h + 8))
    quadrant_w = int(RNG.integers(max_w + 2, max_w + 8))
    assert 2 * quadrant_h + 1 <= 30 and 2 * quadrant_w + 1 <= 30
    grid = np.full((2 * quadrant_h + 1, 2 * quadrant_w + 1), BG, dtype=int)
    grid[quadrant_h, :] = separator
    grid[:, quadrant_w] = separator
    for idx, fragment in enumerate(fragments):
        origin_r, origin_c = _quadrant_origin(idx, quadrant_h, quadrant_w)
        room_r = quadrant_h - fragment.shape[0]
        room_c = quadrant_w - fragment.shape[1]
        offset_r = _sample_offset(room_r)
        offset_c = _sample_offset(room_c)
        paste(grid, fragment, origin_r + offset_r, origin_c + offset_c, transparent=BG)
    return grid


def _validate_generated_case(grid: np.ndarray, expected_impostor: int | None = None) -> None:
    details = _solve_details(grid)
    if expected_impostor is not None:
        assert details["impostor_idx"] == expected_impostor
    bg = int(details["bg"])
    true_indices = list(details["true_indices"])
    aligned = details["aligned"]
    merged = details["merged"]
    expected_output = crop_bbox(merged, bbox(merged != bg))
    assert np.array_equal(details["output"], expected_output)
    for idx in true_indices:
        others = _merge_layers([aligned[j] for j in true_indices if j != idx], bg)
        assert others is not None
        assert np.any((aligned[idx] != bg) & (others == bg)), "each true fragment must matter"
        assert np.any((merged != bg) & (aligned[idx] == bg)), "each true fragment must be incomplete"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sprite = _generate_sprite()
        if sprite is None:
            continue
        fragment_bundle = _make_true_fragments(sprite)
        if fragment_bundle is None:
            continue
        true_fragments, anchors = fragment_bundle

        sprite_colors = sorted(set(map(int, sprite[sprite != BG])))
        separator_choices = [color for color in PALETTE if color not in sprite_colors]
        separator = int(separator_choices[int(RNG.integers(len(separator_choices)))])

        orientations = list(map(int, RNG.permutation(4)))
        impostor_idx = int(RNG.integers(4))
        assigned_true = list(RNG.permutation(3))
        placed: list[np.ndarray] = [None] * 4  # type: ignore[assignment]

        true_slots = [idx for idx in range(4) if idx != impostor_idx]
        for slot, frag_idx in zip(true_slots, assigned_true):
            placed[slot] = _rotate(true_fragments[frag_idx], orientations[slot])

        full_oriented = _rotate(sprite, orientations[impostor_idx])
        anchor_oriented = _rotate_coords(anchors, sprite.shape, orientations[impostor_idx])
        impostor = _make_impostor(full_oriented, anchor_oriented, sprite_colors)
        if impostor is None:
            continue
        placed[impostor_idx] = impostor

        grid = _build_grid(placed, separator)
        try:
            _validate_generated_case(grid, expected_impostor=impostor_idx)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
