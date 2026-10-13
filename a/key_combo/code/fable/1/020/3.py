from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if _UTILS_DIR.exists() and str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import find_components as _arc_find_components
    from arc_utils import get_bg as _arc_get_bg
    from arc_utils import paste as _arc_paste
except Exception:
    _arc_crop_bbox = None
    _arc_find_components = None
    _arc_get_bg = None
    _arc_paste = None


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_BG = 0
GEN_MAX_ATTEMPTS = 500
GEN_COLORS = tuple(range(1, 10))


def _identity(arr: np.ndarray) -> np.ndarray:
    return np.array(arr, copy=True)


OPS = {
    "id": _identity,
    "rot90": lambda arr: np.rot90(arr, 1).copy(),
    "rot180": lambda arr: np.rot90(arr, 2).copy(),
    "rot270": lambda arr: np.rot90(arr, 3).copy(),
    "flip_lr": lambda arr: np.fliplr(arr).copy(),
    "flip_ud": lambda arr: np.flipud(arr).copy(),
    "transpose": lambda arr: np.transpose(arr).copy(),
    "anti_transpose": lambda arr: np.rot90(arr, 2).T.copy(),
}


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return np.asarray(_arc_crop_bbox(grid, box), dtype=grid.dtype)
    r0, c0, r1, c1 = map(int, box)
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=grid.dtype).copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> np.ndarray:
    if _arc_paste is not None:
        return _arc_paste(dst, patch, int(r0), int(c0), transparent=transparent)
    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    dirs = DIRS4 if connectivity == 4 else tuple(
        (dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)
    )
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
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
    return_format: str = "dict",
) -> list[dict]:
    if _arc_find_components is not None:
        return _arc_find_components(
            grid,
            bg=bg,
            connectivity=connectivity,
            monochromatic=monochromatic,
            colors=colors,
            return_format=return_format,
        )

    assert monochromatic and return_format == "dict"
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _get_bg(g)
    palette = sorted(set(map(int, g.ravel())) - {int(bg)}) if colors is None else [int(c) for c in colors if int(c) != int(bg)]
    out = []
    for color in palette:
        for coords in _components_from_mask(g == color, connectivity=connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox_from_coords(coords)


def _crop_binary(mask: np.ndarray) -> np.ndarray:
    box = _mask_bbox(mask)
    assert box is not None
    return mask[box[0] : box[2] + 1, box[1] : box[3] + 1].copy()


def _colorize(mask: np.ndarray, color: int, bg: int = GEN_BG) -> np.ndarray:
    out = np.full(mask.shape, int(bg), dtype=int)
    out[mask] = int(color)
    return out


def _component_mask(grid: np.ndarray, component: dict) -> np.ndarray:
    box = tuple(component["bbox"])
    color = int(component["color"])
    crop = _crop_bbox(grid, box)
    mask = crop == color
    assert mask.any()
    return _crop_binary(mask)


def _matching_ops(src: np.ndarray, dst: np.ndarray) -> set[str]:
    return {name for name, op in OPS.items() if np.array_equal(op(src), dst)}


def _stabilizer(mask: np.ndarray) -> set[str]:
    return _matching_ops(mask, mask)


def _extract_scene(grid: np.ndarray) -> tuple[int, list[tuple[np.ndarray, np.ndarray]], tuple[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert len(components) == 5, "expected exactly five sprites"

    by_color: dict[int, list[dict]] = {}
    for component in components:
        by_color.setdefault(int(component["color"]), []).append(component)

    demo_colors = [color for color, comps in by_color.items() if len(comps) == 2]
    target_colors = [color for color, comps in by_color.items() if len(comps) == 1]
    assert len(demo_colors) == 2 and len(target_colors) == 1

    demos = []
    for color in sorted(demo_colors):
        pair = sorted(by_color[color], key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
        upper, lower = pair
        assert int(upper["bbox"][0]) < int(lower["bbox"][0]), "demo pairs must be ordered vertically"
        demos.append((_component_mask(g, upper), _component_mask(g, lower)))

    target_color = int(target_colors[0])
    target_component = by_color[target_color][0]
    return bg, demos, (target_color, _component_mask(g, target_component))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the only shared orientation change from the two demo pairs, then transform the singleton target sprite.

    Input structure:
    - A dominant background contains exactly five monochromatic 4-connected sprites.
    - Two colors appear twice each; within each same-color pair, the upper sprite is the before example and the lower sprite is the after example.
    - One remaining color appears once and marks the target sprite to transform.

    Rule:
    - Crop each sprite to its minimal bbox and compare each demo pair under the eight orientation transforms: identity, three rotations, and four reflections.
    - Each demo pair admits multiple transforms because of its own symmetry; intersect the two candidate sets and keep the unique shared transform.
    - Apply that transform to the target sprite, keep its color, and return the transformed target cropped to its minimal bbox.

    Key insight:
    - Neither demo is sufficient on its own; the answer is the single orientation that survives both same-color demonstrations simultaneously.
    """
    bg, demos, (target_color, target_mask) = _extract_scene(grid)
    candidate_sets = [_matching_ops(src, dst) for src, dst in demos]
    assert all(len(candidates) > 1 for candidates in candidate_sets), "each demo must be individually ambiguous"

    shared = set.intersection(*candidate_sets)
    assert len(shared) == 1, "demo intersection must determine a unique orientation"
    op_name = next(iter(shared))

    transformed = OPS[op_name](target_mask)
    transformed = _crop_binary(transformed)
    return _colorize(transformed, target_color, bg)


def _random_connected_mask(
    h: int,
    w: int,
    *,
    min_cells: int,
    max_cells: int,
    required_starts: list[tuple[int, int]] | None = None,
) -> np.ndarray | None:
    target = random.randint(min_cells, max_cells)
    mask = np.zeros((h, w), dtype=bool)
    if required_starts:
        start = random.choice(required_starts)
    else:
        start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    frontier = set()
    for dr, dc in DIRS4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while int(mask.sum()) < target:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        rr, cc = cell
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                frontier.add((nr, nc))
    return mask


def _mirror_lr_from_core(core: np.ndarray, full_w: int) -> np.ndarray:
    if full_w % 2 == 0:
        return np.concatenate([core, np.fliplr(core)], axis=1)
    return np.concatenate([core, np.fliplr(core[:, :-1])], axis=1)


def _mirror_ud_from_core(core: np.ndarray, full_h: int) -> np.ndarray:
    if full_h % 2 == 0:
        return np.concatenate([core, np.flipud(core)], axis=0)
    return np.concatenate([core, np.flipud(core[:-1, :])], axis=0)


def _make_symmetric_mask(kind: str) -> np.ndarray:
    target_stabilizer = {"id", kind}
    for _ in range(300):
        if kind == "flip_lr":
            h = random.randint(4, 6)
            w = random.randint(5, 6)
            core_w = (w + 1) // 2
            core = _random_connected_mask(
                h,
                core_w,
                min_cells=max(3, h // 2 + 1),
                max_cells=max(4, min(h * core_w - 1, int(round(h * core_w * 0.75)))),
                required_starts=[(r, core_w - 1) for r in range(h)],
            )
            if core is None:
                continue
            mask = _mirror_lr_from_core(core, w)
        elif kind == "flip_ud":
            h = random.randint(5, 6)
            w = random.randint(4, 6)
            core_h = (h + 1) // 2
            core = _random_connected_mask(
                core_h,
                w,
                min_cells=max(3, w // 2 + 1),
                max_cells=max(4, min(core_h * w - 1, int(round(core_h * w * 0.75)))),
                required_starts=[(core_h - 1, c) for c in range(w)],
            )
            if core is None:
                continue
            mask = _mirror_ud_from_core(core, h)
        else:
            raise ValueError(f"unsupported symmetry kind: {kind}")

        mask = _crop_binary(mask)
        if _stabilizer(mask) == target_stabilizer:
            return mask
    raise RuntimeError(f"failed to generate a mask with stabilizer {target_stabilizer}")


def _make_asymmetric_mask() -> np.ndarray:
    for _ in range(400):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        mask = _random_connected_mask(
            h,
            w,
            min_cells=max(4, min(h, w) + 1),
            max_cells=max(5, min(h * w - 1, int(round(h * w * 0.7)))),
        )
        if mask is None:
            continue
        mask = _crop_binary(mask)
        if min(mask.shape) < 2:
            continue
        if _stabilizer(mask) == {"id"}:
            return mask
    raise RuntimeError("failed to generate an asymmetric target mask")


def _build_canvas(pair_left: dict, pair_right: dict, target_patch: np.ndarray) -> np.ndarray | None:
    left_w = max(pair_left["upper"].shape[1], pair_left["lower"].shape[1]) + random.randint(0, 1)
    center_w = target_patch.shape[1] + random.randint(0, 2)
    right_w = max(pair_right["upper"].shape[1], pair_right["lower"].shape[1]) + random.randint(0, 1)

    left_gap = random.randint(2, 3)
    right_gap = random.randint(2, 3)
    margin_l = random.randint(1, 2)
    margin_r = random.randint(1, 2)

    left_pair_h = pair_left["upper"].shape[0] + pair_left["gap"] + pair_left["lower"].shape[0]
    right_pair_h = pair_right["upper"].shape[0] + pair_right["gap"] + pair_right["lower"].shape[0]

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    extra_h = random.randint(0, 4)
    h = top_margin + bottom_margin + max(left_pair_h, right_pair_h, target_patch.shape[0]) + extra_h
    w = margin_l + left_w + left_gap + center_w + right_gap + right_w + margin_r
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), GEN_BG, dtype=int)
    left_x = margin_l
    center_x = left_x + left_w + left_gap
    right_x = center_x + center_w + right_gap

    left_top = random.randint(top_margin, h - bottom_margin - left_pair_h)
    right_top = random.randint(top_margin, h - bottom_margin - right_pair_h)
    target_top = random.randint(top_margin, h - bottom_margin - target_patch.shape[0])

    left_upper_x = left_x + (left_w - pair_left["upper"].shape[1]) // 2
    left_lower_x = left_x + (left_w - pair_left["lower"].shape[1]) // 2
    right_upper_x = right_x + (right_w - pair_right["upper"].shape[1]) // 2
    right_lower_x = right_x + (right_w - pair_right["lower"].shape[1]) // 2
    target_x = center_x + random.randint(0, center_w - target_patch.shape[1])

    _paste(grid, pair_left["upper"], left_top, left_upper_x, transparent=GEN_BG)
    _paste(
        grid,
        pair_left["lower"],
        left_top + pair_left["upper"].shape[0] + pair_left["gap"],
        left_lower_x,
        transparent=GEN_BG,
    )
    _paste(grid, pair_right["upper"], right_top, right_upper_x, transparent=GEN_BG)
    _paste(
        grid,
        pair_right["lower"],
        right_top + pair_right["upper"].shape[0] + pair_right["gap"],
        right_lower_x,
        transparent=GEN_BG,
    )
    _paste(grid, target_patch, target_top, target_x, transparent=GEN_BG)
    return grid


def _attempt() -> np.ndarray | None:
    op_name = random.choice(tuple(OPS))
    demo_a_mask = _make_symmetric_mask("flip_lr")
    demo_b_mask = _make_symmetric_mask("flip_ud")
    target_mask = _make_asymmetric_mask()

    demo_a_candidates = _matching_ops(demo_a_mask, OPS[op_name](demo_a_mask))
    demo_b_candidates = _matching_ops(demo_b_mask, OPS[op_name](demo_b_mask))
    if len(demo_a_candidates) <= 1 or len(demo_b_candidates) <= 1:
        return None
    if demo_a_candidates & demo_b_candidates != {op_name}:
        return None

    colors = random.sample(GEN_COLORS, 3)
    demo_a_color, demo_b_color, target_color = colors

    pair_a = {
        "upper": _colorize(demo_a_mask, demo_a_color),
        "lower": _colorize(OPS[op_name](demo_a_mask), demo_a_color),
        "gap": random.randint(2, 3),
    }
    pair_b = {
        "upper": _colorize(demo_b_mask, demo_b_color),
        "lower": _colorize(OPS[op_name](demo_b_mask), demo_b_color),
        "gap": random.randint(2, 3),
    }
    target_patch = _colorize(target_mask, target_color)

    pair_left, pair_right = random.sample([pair_a, pair_b], 2)
    grid = _build_canvas(pair_left, pair_right, target_patch)
    if grid is None:
        return None

    expected = _colorize(OPS[op_name](target_mask), target_color)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    bg, demos, (seen_target_color, seen_target_mask) = _extract_scene(grid)
    assert bg == GEN_BG
    assert seen_target_color == target_color
    assert np.array_equal(seen_target_mask, target_mask)
    assert all(len(_matching_ops(src, dst)) > 1 for src, dst in demos)
    assert set.intersection(*[_matching_ops(src, dst) for src, dst in demos]) == {op_name}
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid dual-demo orientation puzzle")
