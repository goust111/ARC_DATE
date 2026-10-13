from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

try:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_shared",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    crop_bbox = _ARC_UTILS.crop_bbox
    find_components = _ARC_UTILS.find_components
    get_bg = _ARC_UTILS.get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        vals, counts = np.unique(grid, return_counts=True)
        return int(vals[np.argmax(counts)])

    def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bbox
        return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors: list[int] | None = None,
        return_format: str = "dict",
    ) -> list[dict]:
        if bg is None:
            bg = get_bg(grid)
        bg = int(bg)
        if connectivity != 4 or not monochromatic or return_format != "dict":
            raise ValueError("fallback find_components only supports 4-connected monochrome dict mode")
        if colors is None:
            colors = [int(v) for v in np.unique(grid) if int(v) != bg]
        out = []
        h, w = grid.shape
        seen = np.zeros((h, w), dtype=bool)
        for color in colors:
            for r, c in np.argwhere(grid == int(color)):
                r = int(r)
                c = int(c)
                if seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                arr = np.array(coords, dtype=int)
                r0, c0 = arr.min(axis=0)
                r1, c1 = arr.max(axis=0)
                out.append(
                    {
                        "color": int(color),
                        "coords": arr,
                        "bbox": (int(r0), int(c0), int(r1), int(c1)),
                        "area": int(len(coords)),
                    }
                )
        return out


BG = 0
PALETTE = tuple(range(1, 10))
RNG = np.random.default_rng()

GEN_RETRIES = 500
PATCH_H_RANGE = (7, 10)
PATCH_W_OPTIONS = (8, 10, 12)
GAP_RANGE = (1, 3)
NOISE_RANGE = (2, 4)


def _axis_candidates(width: int) -> list[int]:
    return list(range(1, 2 * int(width) - 2, 2))


def _mirror_col(s: int, c: int) -> int:
    return int(s - c)


def _axis_survivor_mask(patch: np.ndarray, color: int, s: int) -> np.ndarray:
    h, w = patch.shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(patch == int(color)):
        r = int(r)
        c = int(c)
        mc = _mirror_col(int(s), c)
        if 0 <= mc < w and int(patch[r, mc]) == int(color):
            mask[r, c] = True
    return mask


def _best_axis(patch: np.ndarray, color: int) -> tuple[int, np.ndarray, dict[int, int]]:
    scores: dict[int, int] = {}
    masks: dict[int, np.ndarray] = {}
    for s in _axis_candidates(patch.shape[1]):
        mask = _axis_survivor_mask(patch, color, s)
        masks[int(s)] = mask
        scores[int(s)] = int(mask.sum())
    best_score = max(scores.values())
    best_axes = [s for s, score in scores.items() if score == best_score]
    best_s = min(best_axes)
    cleaned = np.zeros_like(patch)
    cleaned[masks[best_s]] = int(color)
    return int(best_s), cleaned, scores


def _overlay(order: list[int], patches: list[np.ndarray], colors: list[int]) -> np.ndarray:
    h, w = patches[0].shape
    out = np.zeros((h, w), dtype=int)
    for idx in order:
        patch = patches[int(idx)]
        color = int(colors[int(idx)])
        out[patch == color] = color
    return out


def _layer_data(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps = [comp for comp in comps if int(comp["color"]) != bg]
    assert len(comps) == 3, "expected exactly three separated color layers"
    shapes = []
    layers = []
    for comp in comps:
        patch = crop_bbox(g, comp["bbox"])
        color = int(comp["color"])
        best_s, cleaned, scores = _best_axis(patch, color)
        layers.append(
            {
                "color": color,
                "bbox": tuple(int(v) for v in comp["bbox"]),
                "raw_patch": patch,
                "clean_patch": cleaned,
                "raw_count": int(np.count_nonzero(patch == color)),
                "clean_count": int(np.count_nonzero(cleaned == color)),
                "best_axis": int(best_s),
                "scores": scores,
            }
        )
        shapes.append(patch.shape)
    assert len(set(shapes)) == 1, "all layer bboxes must share the same size"
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each separated color layer by its own best vertical symmetry axis, then overlay the cleaned layers by surviving-pixel count.

    Input structure:
    - The background is black.
    - Exactly three non-background colors appear, each as one separated monochrome layer on the canvas.
    - The three layer bboxes have the same size, so they can be aligned by local coordinates.
    - Inside each bbox, the color forms a nearly vertically symmetric pattern plus a few unmatched noise pixels.

    Rule:
    - For each color layer independently, examine half-integer vertical axes inside that layer's own bbox and pick the axis that leaves the most same-color pixels with a mirrored partner.
    - Remove every pixel of that layer that lacks its same-color mirror across the chosen axis.
    - Count the remaining pixels in each cleaned layer.
    - Align the cleaned layers by their shared bbox size and draw them on a black output canvas from largest surviving count to smallest; later layers overwrite earlier ones at the same local cell.

    Key insight:
    - The symmetry cleaning is not cosmetic: it changes the per-layer pixel counts, and those new counts decide which cleaned layer ends up on top where local coordinates overlap.
    """
    layers = _layer_data(grid)
    counts = [int(layer["clean_count"]) for layer in layers]
    order = sorted(range(len(layers)), key=lambda idx: (-counts[idx], idx))
    patches = [layer["clean_patch"] for layer in layers]
    colors = [int(layer["color"]) for layer in layers]
    return _overlay(order, patches, colors)


def _valid_half_mask_cells(h: int, w: int, s: int) -> set[tuple[int, int]]:
    adj = (int(s) - 1) // 2
    c_low = max(0, int(s) - (int(w) - 1))
    return {(r, c) for r in range(int(h)) for c in range(int(c_low), int(adj) + 1)}


def _grow_half_mask(rng: np.random.Generator, h: int, w: int, s: int, target_pairs: int) -> np.ndarray:
    adj = (int(s) - 1) // 2
    c_low = max(0, int(s) - (int(w) - 1))
    target_pairs = int(target_pairs)
    half = np.zeros((h, w), dtype=bool)
    half[:, adj] = True
    reach_row = int(rng.integers(1, h - 1))
    half[reach_row, c_low : adj + 1] = True
    valid = _valid_half_mask_cells(h, w, s)

    if int(half.sum()) > target_pairs:
        target_pairs = int(half.sum())

    while int(half.sum()) < target_pairs:
        frontier = []
        for r, c in np.argwhere(half):
            r = int(r)
            c = int(c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                cell = (int(nr), int(nc))
                if cell in valid and not half[cell]:
                    frontier.append(cell)
        if not frontier:
            break
        nr, nc = frontier[int(rng.integers(len(frontier)))]
        half[nr, nc] = True
    return half


def _mirror_half_to_patch(half: np.ndarray, color: int, s: int) -> np.ndarray:
    h, w = half.shape
    patch = np.zeros((h, w), dtype=int)
    for r, c in np.argwhere(half):
        r = int(r)
        c = int(c)
        mc = _mirror_col(int(s), c)
        if 0 <= mc < w:
            patch[r, c] = int(color)
            patch[r, mc] = int(color)
    return patch


def _adjacent_empty_cells(patch: np.ndarray, color: int) -> list[tuple[int, int]]:
    h, w = patch.shape
    out = set()
    for r, c in np.argwhere(patch == int(color)):
        r = int(r)
        c = int(c)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and int(patch[nr, nc]) == BG:
                out.add((int(nr), int(nc)))
    return list(out)


def _add_noise_cells(
    rng: np.random.Generator,
    patch: np.ndarray,
    color: int,
    s: int,
    noise_target: int,
) -> np.ndarray:
    h, w = patch.shape
    out = patch.copy()
    base_mask = patch == int(color)

    if int(s) < w - 1:
        bridge_row = int(np.argwhere(base_mask[:, int(s)]).ravel()[0])
        for c in range(int(s) + 1, w):
            out[bridge_row, c] = int(color)
    elif int(s) > w - 1:
        c_low = max(0, int(s) - (w - 1))
        bridge_row = int(np.argwhere(base_mask[:, c_low]).ravel()[0])
        for c in range(0, c_low):
            out[bridge_row, c] = int(color)

    while int(np.count_nonzero(out == color) - np.count_nonzero(base_mask)) < int(noise_target):
        candidates = []
        occupied = out == int(color)
        for r, c in _adjacent_empty_cells(out, color):
            mc = _mirror_col(int(s), int(c))
            if 0 <= mc < w and occupied[r, mc]:
                continue
            candidates.append((int(r), int(c)))
        if not candidates:
            break
        r, c = candidates[int(rng.integers(len(candidates)))]
        out[r, c] = int(color)
    return out


def _rank_order(values: list[int]) -> tuple[int, ...]:
    return tuple(sorted(range(len(values)), key=lambda idx: (-int(values[idx]), idx)))


def _visible_overlap_count(clean_patches: list[np.ndarray], colors: list[int]) -> int:
    masks = [patch == int(color) for patch, color in zip(clean_patches, colors)]
    total = 0
    for r in range(clean_patches[0].shape[0]):
        for c in range(clean_patches[0].shape[1]):
            if sum(int(mask[r, c]) for mask in masks) >= 2:
                total += 1
    return int(total)


def _raw_overlay(raw_patches: list[np.ndarray], colors: list[int], counts: list[int]) -> np.ndarray:
    order = sorted(range(len(raw_patches)), key=lambda idx: (-int(counts[idx]), idx))
    return _overlay(order, raw_patches, colors)


def _build_layer(
    rng: np.random.Generator,
    h: int,
    w: int,
    color: int,
    s: int,
    target_pairs: int,
    noise_target: int,
) -> np.ndarray | None:
    half = _grow_half_mask(rng, h, w, s, target_pairs)
    clean_patch = _mirror_half_to_patch(half, color, s)
    raw_patch = _add_noise_cells(rng, clean_patch, color, s, noise_target)

    best_s, cleaned_again, scores = _best_axis(raw_patch, color)
    ranked_scores = sorted(scores.values(), reverse=True)
    second = ranked_scores[1] if len(ranked_scores) > 1 else -1
    if int(best_s) != int(s):
        return None
    if int(scores[best_s]) - int(second) < 2:
        return None
    if not np.array_equal(clean_patch, cleaned_again):
        return None
    if int(np.count_nonzero(raw_patch == color) - np.count_nonzero(clean_patch == color)) != int(noise_target):
        return None
    rows, cols = np.where(raw_patch == int(color))
    if rows.size == 0:
        return None
    if int(rows.min()) != 0 or int(rows.max()) != h - 1 or int(cols.min()) != 0 or int(cols.max()) != w - 1:
        return None
    return raw_patch


def _layout_positions(rng: np.random.Generator, h: int, w: int, gap: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    canvas_h = 2 * h + 3 * gap
    canvas_w = 2 * w + 3 * gap
    grid = np.zeros((canvas_h, canvas_w), dtype=int)
    slots = [
        (gap, gap),
        (gap, 2 * gap + w),
        (2 * gap + h, gap),
        (2 * gap + h, 2 * gap + w),
    ]
    chosen = rng.choice(len(slots), size=3, replace=False)
    positions = [slots[int(i)] for i in chosen]
    return grid, positions


def _validate_example(grid: np.ndarray) -> bool:
    try:
        layers = _layer_data(grid)
        output = main(grid)
    except Exception:
        return False

    colors = [int(layer["color"]) for layer in layers]
    raw_patches = [layer["raw_patch"] for layer in layers]
    clean_patches = [layer["clean_patch"] for layer in layers]
    raw_counts = [int(layer["raw_count"]) for layer in layers]
    clean_counts = [int(layer["clean_count"]) for layer in layers]
    best_axes = [int(layer["best_axis"]) for layer in layers]

    if len(set(clean_counts)) != 3:
        return False
    if len(set(best_axes)) != 3:
        return False
    if _rank_order(raw_counts) == _rank_order(clean_counts):
        return False
    if _visible_overlap_count(clean_patches, colors) < 3:
        return False

    true_order = sorted(range(3), key=lambda idx: (-clean_counts[idx], idx))
    expected = _overlay(true_order, clean_patches, colors)
    if not np.array_equal(output, expected):
        return False

    if np.array_equal(output, _raw_overlay(raw_patches, colors, raw_counts)):
        return False
    if np.array_equal(output, _raw_overlay(clean_patches, colors, raw_counts)):
        return False

    visible_colors = {int(v) for v in np.unique(output) if int(v) != BG}
    if len(visible_colors) < 2:
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(PATCH_H_RANGE[0], PATCH_H_RANGE[1] + 1))
    w = int(rng.choice(PATCH_W_OPTIONS))
    gap = int(rng.integers(GAP_RANGE[0], GAP_RANGE[1] + 1))

    if 2 * h + 3 * gap > 30 or 2 * w + 3 * gap > 30:
        return None

    center_s = w - 1
    axes = [center_s - 2, center_s, center_s + 2]
    rng.shuffle(axes)
    colors = list(rng.choice(PALETTE, size=3, replace=False))

    min_pairs = []
    max_pairs = []
    for s in axes:
        adj = (int(s) - 1) // 2
        c_low = max(0, int(s) - (w - 1))
        base_pairs = h + (adj - c_low)
        min_pairs.append(base_pairs)
        max_pairs.append(h * (adj - c_low + 1))

    target_pairs = []
    for lo, hi in zip(min_pairs, max_pairs):
        hi = min(int(hi), int(lo) + 10)
        target_pairs.append(int(rng.integers(int(lo), int(hi) + 1)))
    if len(set(2 * np.array(target_pairs))) != 3:
        return None

    noise_targets = [int(rng.integers(NOISE_RANGE[0], NOISE_RANGE[1] + 1)) for _ in range(3)]

    patches = []
    for color, s, pairs, noise in zip(colors, axes, target_pairs, noise_targets):
        patch = _build_layer(rng, h, w, int(color), int(s), int(pairs), int(noise))
        if patch is None:
            return None
        patches.append(patch)

    grid, positions = _layout_positions(rng, h, w, gap)
    for patch, (r0, c0) in zip(patches, positions):
        mask = patch != BG
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any(view[mask] != BG):
            return None
        view[mask] = patch[mask]

    if not _validate_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-clean-overlay input")
