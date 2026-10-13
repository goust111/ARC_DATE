from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location
except ModuleNotFoundError:
    _ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
    if str(_ARC_UTILS_DIR) not in sys.path:
        sys.path.append(str(_ARC_UTILS_DIR))
    from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location


BG = 0
FRAME = 5
GEN_RETRIES = 400


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for coords in components_mask(grid == FRAME, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(coords)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == FRAME) and np.any(sub[1:-1, 1:-1] != FRAME):
            candidates.append((int(r0), int(c0), int(r1), int(c1)))
    assert len(candidates) == 1, f"expected one frame, found {len(candidates)}"
    return candidates[0]


def _placeholder_data(crop: np.ndarray) -> list[dict]:
    inner = crop[1:-1, 1:-1]
    colors = sorted(set(map(int, np.unique(inner))) - {BG, FRAME})
    assert 2 <= len(colors) <= 3, f"expected 2-3 placeholder colors, found {colors}"
    placeholders: list[dict] = []
    for color in colors:
        pts = np.argwhere(crop == color)
        assert pts.shape == (1, 2), f"placeholder color {color} must be a singleton"
        r, c = map(int, pts[0])
        assert 1 <= r < crop.shape[0] - 1 and 1 <= c < crop.shape[1] - 1
        placeholders.append({"color": int(color), "pos": (r, c)})
    return placeholders


def _outside_components(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], color: int) -> list[np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    return components_mask((grid == int(color)) & outside, connectivity=4)


def _legend_map(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], placeholder_colors: set[int]
) -> tuple[dict[int, int], set[tuple[int, int]]]:
    outside = np.ones(grid.shape, dtype=bool)
    r0, c0, r1, c1 = frame_bbox
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    output_colors: set[int] = set()

    for src in sorted(placeholder_colors):
        candidates: list[tuple[int, int, int]] = []
        for comp in components_mask((grid == src) & outside, connectivity=4):
            if comp.shape[0] != 1:
                continue
            r, c = map(int, comp[0])
            if r + 1 >= grid.shape[0] or not outside[r + 1, c]:
                continue
            dst = int(grid[r + 1, c])
            if dst in (BG, FRAME, src) or dst in placeholder_colors:
                continue
            below_comp = components_mask((grid == dst) & outside, connectivity=4)
            if sum(int(x.shape[0]) == 1 and int(x[0, 0]) == r + 1 and int(x[0, 1]) == c for x in below_comp) != 1:
                continue
            candidates.append((r, c, dst))
        assert len(candidates) == 1, f"legend for color {src} is ambiguous: {candidates}"
        r, c, dst = candidates[0]
        mapping[int(src)] = int(dst)
        output_colors.add(int(dst))
        legend_cells.add((int(r), int(c)))
        legend_cells.add((int(r + 1), int(c)))

    assert len(output_colors) == len(placeholder_colors), "legend outputs must be unique"
    return mapping, legend_cells


def _sprite_masks(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    placeholder_colors: set[int],
    legend_cells: set[tuple[int, int]],
) -> dict[int, np.ndarray]:
    sprites: dict[int, np.ndarray] = {}
    dims: set[tuple[int, int]] = set()

    for color in sorted(placeholder_colors):
        comps = _outside_components(grid, frame_bbox, color)
        sprite_candidates = []
        for comp in comps:
            pts = [tuple(map(int, rc)) for rc in comp]
            if any(pt in legend_cells for pt in pts):
                continue
            sprite_candidates.append(comp)
        assert len(sprite_candidates) == 1, f"expected one sprite for color {color}, found {len(sprite_candidates)}"
        comp = sprite_candidates[0]
        assert 4 <= comp.shape[0] <= 8, f"sprite area out of range for color {color}"
        r0, c0, r1, c1 = bbox_coords(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[comp[:, 0] - r0, comp[:, 1] - c0] = True
        dims.add(tuple(map(int, mask.shape)))
        sprites[int(color)] = mask

    assert len(dims) == 1, f"sprite bboxes must match, found {dims}"
    return sprites


def _stamp_origin(mask: np.ndarray, placeholder_pos: tuple[int, int]) -> tuple[int, int]:
    anchor_r = (mask.shape[0] - 1) // 2
    anchor_c = (mask.shape[1] - 1) // 2
    return int(placeholder_pos[0] - anchor_r), int(placeholder_pos[1] - anchor_c)


def _render_crop(
    crop_shape: tuple[int, int],
    placeholders: list[dict],
    sprites: dict[int, np.ndarray],
    color_map: dict[int, int],
) -> tuple[np.ndarray, dict]:
    h, w = map(int, crop_shape)
    out = np.full((h, w), BG, dtype=int)
    out[0, :] = FRAME
    out[-1, :] = FRAME
    out[:, 0] = FRAME
    out[:, -1] = FRAME

    coverage: dict[tuple[int, int], list[int]] = {}
    visible = [0] * len(placeholders)
    overlap_pixels = 0
    tie_pixels = 0
    ambiguous = 0

    for idx, item in enumerate(placeholders):
        mask = sprites[item["color"]]
        top, left = _stamp_origin(mask, item["pos"])
        for dr, dc in np.argwhere(mask):
            rr = int(top + dr)
            cc = int(left + dc)
            if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                coverage.setdefault((rr, cc), []).append(idx)

    for (rr, cc), owners in coverage.items():
        if len(owners) > 1:
            overlap_pixels += 1
        scored = []
        for idx in owners:
            pr, pc = placeholders[idx]["pos"]
            scored.append((max(abs(pr - rr), abs(pc - cc)), pr, idx))
        scored.sort()
        if len(scored) >= 2 and scored[0][:2] == scored[1][:2]:
            ambiguous += 1
        if len(scored) >= 2 and scored[0][0] == scored[1][0] and scored[0][1] != scored[1][1]:
            tie_pixels += 1
        winner = scored[0][2]
        visible[winner] += 1
        out[rr, cc] = int(color_map[placeholders[winner]["color"]])

    stats = {
        "overlap_pixels": int(overlap_pixels),
        "tie_pixels": int(tie_pixels),
        "ambiguous_pixels": int(ambiguous),
        "visible_pixels": tuple(int(v) for v in visible),
        "covered_pixels": int(len(coverage)),
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp outside sprites into the grey frame and recolor overlaps by nearest-placeholder ownership.

    Input structure:
    - Black background with one grey rectangular frame of thickness 1.
    - Inside the frame are 2-3 differently colored placeholder singleton pixels.
    - Outside the frame are one monochrome sprite per placeholder color and a legend of vertical color pairs.

    Rule:
    - Match each outside sprite to the placeholder of the same color and align the sprite's bbox-center floor to that placeholder.
    - Recolor the stamped sprite using the legend mapping from placeholder color (top pixel) to output color (bottom pixel).
    - Crop to the frame and keep the grey border; inside it, clipped stamped pixels survive, and contested pixels go to the covering stamp whose placeholder has smaller Chebyshev distance, breaking equal-distance ties by smaller placeholder row.

    Key insight:
    - The frame interior is a clipped overlap of several translated sprite masks, but color ownership is not paint order: it is a Voronoi-style partition induced by placeholder positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(get_bg(g)) == BG

    frame_bbox = _find_frame(g)
    crop = crop_bbox(g, frame_bbox)
    placeholders = _placeholder_data(crop)
    placeholder_colors = {int(x["color"]) for x in placeholders}
    color_map, legend_cells = _legend_map(g, frame_bbox, placeholder_colors)
    sprites = _sprite_masks(g, frame_bbox, placeholder_colors, legend_cells)
    out, stats = _render_crop(crop.shape, placeholders, sprites, color_map)
    assert stats["ambiguous_pixels"] == 0, "distance rule does not uniquely resolve overlaps"
    return out


def _mask_from_rows(*rows: str) -> np.ndarray:
    arr = np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)
    coords = np.argwhere(arr)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r0), int(c0)) == (0, 0)
    assert (int(r1), int(c1)) == (arr.shape[0] - 1, arr.shape[1] - 1)
    return arr


_TEMPLATES: dict[tuple[int, int], list[np.ndarray]] = {
    (3, 3): [
        _mask_from_rows("111", "010", "011"),
        _mask_from_rows("110", "011", "010"),
        _mask_from_rows("111", "001", "011"),
    ],
    (3, 4): [
        _mask_from_rows("1110", "0110", "0011"),
        _mask_from_rows("1100", "1110", "0011"),
        _mask_from_rows("1110", "0011", "0110"),
    ],
    (4, 3): [
        _mask_from_rows("110", "010", "011", "001"),
        _mask_from_rows("010", "111", "001", "001"),
        _mask_from_rows("111", "001", "011", "010"),
    ],
    (4, 4): [
        _mask_from_rows("1100", "1110", "0010", "0011"),
        _mask_from_rows("0110", "1110", "0010", "0011"),
        _mask_from_rows("1110", "0010", "0110", "0011"),
    ],
    (5, 3): [
        _mask_from_rows("010", "111", "010", "011", "010"),
        _mask_from_rows("010", "110", "010", "011", "010"),
        _mask_from_rows("010", "111", "010", "110", "010"),
    ],
    (5, 4): [
        _mask_from_rows("0100", "1100", "0100", "0110", "0011"),
        _mask_from_rows("0010", "0011", "1110", "0100", "0100"),
        _mask_from_rows("0100", "1100", "0110", "0011", "0001"),
    ],
}


def _transforms_same_shape(mask: np.ndarray) -> list[np.ndarray]:
    variants = [mask, np.fliplr(mask), np.flipud(mask), np.rot90(mask, 2)]
    if mask.shape[0] == mask.shape[1]:
        variants.extend([np.rot90(mask, 1), np.rot90(mask, 3)])
    uniq: list[np.ndarray] = []
    seen: set[bytes] = set()
    for item in variants:
        key = item.tobytes()
        if key not in seen:
            seen.add(key)
            uniq.append(item.copy())
    return uniq


def _validate_templates() -> None:
    for dims, masks in _TEMPLATES.items():
        assert len(masks) >= 2
        for mask in masks:
            assert tuple(map(int, mask.shape)) == dims
            area = int(mask.sum())
            assert 4 <= area <= 8
            comps = components_mask(mask, connectivity=4)
            assert len(comps) == 1
            assert area < mask.size


_validate_templates()


def _choose_masks(dims: tuple[int, int], n: int, force_tie: bool) -> list[np.ndarray]:
    base_masks = []
    candidates = list(_TEMPLATES[dims])
    random.shuffle(candidates)
    while candidates and len(base_masks) < n:
        base = candidates.pop()
        for variant in _transforms_same_shape(base):
            if all(not np.array_equal(variant, old) for old in base_masks):
                base_masks.append(variant)
                break
    if len(base_masks) < n:
        all_variants = []
        for base in _TEMPLATES[dims]:
            all_variants.extend(_transforms_same_shape(base))
        random.shuffle(all_variants)
        for variant in all_variants:
            if all(not np.array_equal(variant, old) for old in base_masks):
                base_masks.append(variant)
            if len(base_masks) == n:
                break
    assert len(base_masks) == n

    if force_tie:
        for idx in range(2):
            mask = base_masks[idx]
            if not (mask[0, 1] and mask[-1, 1]):
                tied = None
                for variant in _transforms_same_shape(mask):
                    if variant.shape == mask.shape and variant[0, 1] and variant[-1, 1]:
                        tied = variant
                        break
                assert tied is not None
                base_masks[idx] = tied
    return base_masks


def _relative_layouts(n: int, force_tie: bool) -> list[list[tuple[int, int]]]:
    if force_tie and n == 2:
        return [[(0, 0), (4, 0)], [(0, 0), (4, 0)]]
    if force_tie and n == 3:
        return [[(0, 0), (4, 0), (2, 4)], [(0, 0), (4, 0), (0, 4)]]
    if n == 2:
        return [[(0, 0), (0, 3)], [(0, 0), (0, 4)], [(0, 0), (2, 3)], [(0, 0), (3, 2)]]
    return [[(0, 0), (0, 4), (3, 2)], [(0, 0), (2, 3), (4, 1)], [(0, 0), (3, 1), (1, 4)]]


def _shifted_positions(
    rel_positions: list[tuple[int, int]], interior_shape: tuple[int, int]
) -> list[tuple[int, int]] | None:
    ih, iw = map(int, interior_shape)
    rows = [r for r, _ in rel_positions]
    cols = [c for _, c in rel_positions]
    max_dr = ih - 1 - max(rows)
    max_dc = iw - 1 - max(cols)
    if max_dr < 0 or max_dc < 0:
        return None
    off_r = random.randint(0, max_dr)
    off_c = random.randint(0, max_dc)
    return [(off_r + r, off_c + c) for r, c in rel_positions]


def _legend_patch(mapping: dict[int, int]) -> np.ndarray:
    keys = list(mapping)
    width = 2 * len(keys) - 1
    patch = np.zeros((2, width), dtype=int)
    for i, src in enumerate(keys):
        c = 2 * i
        patch[0, c] = int(src)
        patch[1, c] = int(mapping[src])
    return patch


def _sprite_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _render_expected_from_spec(
    interior_shape: tuple[int, int],
    placeholder_colors: list[int],
    positions: list[tuple[int, int]],
    masks: list[np.ndarray],
    mapping: dict[int, int],
) -> tuple[np.ndarray, dict]:
    crop_h = int(interior_shape[0] + 2)
    crop_w = int(interior_shape[1] + 2)
    placeholders = []
    sprites = {}
    for color, (r, c), mask in zip(placeholder_colors, positions, masks):
        placeholders.append({"color": int(color), "pos": (int(r + 1), int(c + 1))})
        sprites[int(color)] = mask
    return _render_crop((crop_h, crop_w), placeholders, sprites, mapping)


def _build_input(
    grid_shape: tuple[int, int],
    frame_top_left: tuple[int, int],
    interior_shape: tuple[int, int],
    placeholder_colors: list[int],
    positions: list[tuple[int, int]],
    masks: list[np.ndarray],
    mapping: dict[int, int],
) -> np.ndarray | None:
    h, w = map(int, grid_shape)
    ih, iw = map(int, interior_shape)
    fr0, fc0 = map(int, frame_top_left)
    fr1, fc1 = fr0 + ih + 1, fc0 + iw + 1
    if fr1 >= h or fc1 >= w:
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[fr0 : fr1 + 1, fc0] = FRAME
    grid[fr0 : fr1 + 1, fc1] = FRAME
    grid[fr0, fc0 : fc1 + 1] = FRAME
    grid[fr1, fc0 : fc1 + 1] = FRAME

    for color, (r, c) in zip(placeholder_colors, positions):
        grid[fr0 + 1 + r, fc0 + 1 + c] = int(color)

    canvas = np.zeros_like(grid)
    canvas[fr0 : fr1 + 1, fc0 : fc1 + 1] = FRAME

    legend = _legend_patch(mapping)
    try:
        lr, lc = random_free_location(canvas, legend, transparent=BG, padding=1)
    except ValueError:
        return None
    paste(grid, legend, lr, lc, transparent=BG)
    paste(canvas, legend, lr, lc, transparent=BG)

    sprite_order = list(range(len(placeholder_colors)))
    random.shuffle(sprite_order)
    for idx in sprite_order:
        patch = _sprite_patch(masks[idx], placeholder_colors[idx])
        try:
            sr, sc = random_free_location(canvas, patch, transparent=BG, padding=1)
        except ValueError:
            return None
        paste(grid, patch, sr, sc, transparent=BG)
        paste(canvas, patch, sr, sc, transparent=BG)

    return grid


def _semantic_ok(
    inp: np.ndarray,
    expected: np.ndarray,
    force_tie: bool,
    want_clip: bool,
    stats: dict,
    masks: list[np.ndarray],
    positions: list[tuple[int, int]],
    interior_shape: tuple[int, int],
) -> bool:
    if stats["ambiguous_pixels"] != 0 or stats["overlap_pixels"] <= 0:
        return False
    if force_tie and stats["tie_pixels"] <= 0:
        return False
    if not force_tie and stats["covered_pixels"] <= 6:
        return False
    if min(stats["visible_pixels"]) < 2:
        return False
    if stats["overlap_pixels"] * 3 > max(1, stats["covered_pixels"]) * 2:
        return False

    ih, iw = map(int, interior_shape)
    clipped = False
    for mask, (r, c) in zip(masks, positions):
        top = r - (mask.shape[0] - 1) // 2
        left = c - (mask.shape[1] - 1) // 2
        for dr, dc in np.argwhere(mask):
            rr = int(top + dr)
            cc = int(left + dc)
            if not (0 <= rr < ih and 0 <= cc < iw):
                clipped = True
                break
        if clipped:
            break
    if want_clip != clipped:
        return False

    if np.count_nonzero(expected[1:-1, 1:-1]) <= len(masks):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3])
    force_tie = random.random() < 0.45
    want_clip = random.random() < 0.35
    if force_tie:
        dims = (5, 3)
    else:
        dims = random.choice(sorted(_TEMPLATES))

    masks = _choose_masks(dims, n, force_tie)
    rel_positions = random.choice(_relative_layouts(n, force_tie))

    ih = random.randint(max(7, max(r for r, _ in rel_positions) + 3), 11)
    iw = random.randint(max(7, max(c for _, c in rel_positions) + 3), 11)
    positions = _shifted_positions(rel_positions, (ih, iw))
    if positions is None:
        return None

    if want_clip:
        edge_idx = random.randrange(n)
        pr, pc = positions[edge_idx]
        if random.random() < 0.5:
            pr = random.choice([0, ih - 1])
        else:
            pc = random.choice([0, iw - 1])
        positions = positions.copy()
        positions[edge_idx] = (pr, pc)

    placeholder_colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n)
    output_pool = [c for c in [1, 2, 3, 4, 6, 7, 8, 9] if c not in placeholder_colors]
    output_colors = random.sample(output_pool, n)
    mapping = {int(src): int(dst) for src, dst in zip(placeholder_colors, output_colors)}

    expected, stats = _render_expected_from_spec((ih, iw), placeholder_colors, positions, masks, mapping)

    grid_h = random.randint(max(ih + 8, 16), min(30, ih + 15))
    grid_w = random.randint(max(iw + 10, 18), min(30, iw + 16))
    frame_r = random.randint(1, grid_h - (ih + 2) - 1)
    frame_c = random.randint(1, grid_w - (iw + 2) - 1)

    inp = _build_input((grid_h, grid_w), (frame_r, frame_c), (ih, iw), placeholder_colors, positions, masks, mapping)
    if inp is None:
        return None
    if not _semantic_ok(inp, expected, force_tie, want_clip, stats, masks, positions, (ih, iw)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
