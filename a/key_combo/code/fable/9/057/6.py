from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _color_shape(panel: np.ndarray, color: int) -> tuple[BBox, np.ndarray]:
    coords = np.argwhere(panel == int(color))
    assert coords.size > 0
    bbox = _bbox_from_coords(coords)
    cropped = _crop(panel == int(color), bbox).astype(bool)
    return bbox, cropped


def _mask_cells(mask: np.ndarray, top: int, left: int) -> set[Coord]:
    cells = np.argwhere(mask)
    return {(int(top + r), int(left + c)) for r, c in cells}


def _mask_overlap_cells(mask_a: np.ndarray, pos_a: Coord, mask_b: np.ndarray, pos_b: Coord) -> set[Coord]:
    return _mask_cells(mask_a, *pos_a) & _mask_cells(mask_b, *pos_b)


def _fits_in_panel(mask: np.ndarray, pos: Coord, panel_shape: tuple[int, int]) -> bool:
    h, w = panel_shape
    mh, mw = mask.shape
    r, c = pos
    return 0 <= r and 0 <= c and r + mh <= h and c + mw <= w


def _place_mask(panel: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    mh, mw = mask.shape
    view = panel[top : top + mh, left : left + mw]
    view[mask] = int(color)


def _rect_overlap(pos_a: Coord, shape_a: tuple[int, int], pos_b: Coord, shape_b: tuple[int, int]) -> bool:
    ar0, ac0 = pos_a
    br0, bc0 = pos_b
    ah, aw = shape_a
    bh, bw = shape_b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _first_overlap_step(
    mask_a: np.ndarray,
    last_a: Coord,
    delta_a: Coord,
    mask_b: np.ndarray,
    last_b: Coord,
    delta_b: Coord,
    panel_shape: tuple[int, int],
    max_steps: int = 12,
) -> tuple[int, set[Coord]] | None:
    for step in range(1, max_steps + 1):
        pos_a = (last_a[0] + step * delta_a[0], last_a[1] + step * delta_a[1])
        pos_b = (last_b[0] + step * delta_b[0], last_b[1] + step * delta_b[1])
        if not _fits_in_panel(mask_a, pos_a, panel_shape) or not _fits_in_panel(mask_b, pos_b, panel_shape):
            return None
        overlap = _mask_overlap_cells(mask_a, pos_a, mask_b, pos_b)
        if overlap:
            return step, overlap
    return None


def _render_panel(
    side: int,
    fill_color: int,
    pillar_color: int,
    pillar_col: int,
    a_color: int,
    a_mask: np.ndarray,
    a_pos: Coord,
    b_color: int,
    b_mask: np.ndarray,
    b_pos: Coord,
    *,
    wrong_order: bool = False,
    rect_mode: bool = False,
) -> np.ndarray:
    panel = np.full((side, side), int(fill_color), dtype=int)

    a_draw = np.ones_like(a_mask, dtype=bool) if rect_mode else a_mask
    b_draw = np.ones_like(b_mask, dtype=bool) if rect_mode else b_mask

    if wrong_order:
        _place_mask(panel, a_draw, a_pos[0], a_pos[1], a_color)
        _place_mask(panel, b_draw, b_pos[0], b_pos[1], b_color)
        panel[1:-1, pillar_col] = int(pillar_color)
    else:
        _place_mask(panel, b_draw, b_pos[0], b_pos[1], b_color)
        panel[1:-1, pillar_col] = int(pillar_color)
        _place_mask(panel, a_draw, a_pos[0], a_pos[1], a_color)
    return panel


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(g[0, 0])

    panel_coords = _components(g != bg)
    assert len(panel_coords) >= 3, "expected a row of independent panels"
    panel_boxes = sorted((_bbox_from_coords(coords) for coords in panel_coords), key=lambda b: (b[1], b[0]))
    side_shapes = {(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in panel_boxes}
    assert len(side_shapes) == 1
    side_h, side_w = next(iter(side_shapes))
    assert side_h == side_w, "panels must be square"
    panels = [_crop(g, box) for box in panel_boxes]

    fill_color = int(Counter(map(int, panels[0].ravel())).most_common(1)[0][0])
    active_colors = sorted(set(map(int, np.unique(panels[0]))) - {fill_color})
    assert len(active_colors) == 3, "expected pillar and two sprite colors"

    a_candidates: list[dict[str, object]] = []
    for color in active_colors:
        boxes: list[BBox] = []
        masks: list[np.ndarray] = []
        for panel in panels:
            bbox, mask = _color_shape(panel, color)
            boxes.append(bbox)
            masks.append(mask)
        if any(mask.shape != masks[0].shape for mask in masks[1:]):
            continue
        if any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
            continue
        positions = [(b[0], b[1]) for b in boxes]
        deltas = [(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1]) for i in range(len(positions) - 1)]
        if len(set(deltas)) != 1 or deltas[0] == (0, 0):
            continue
        a_candidates.append(
            {
                "color": color,
                "mask": masks[0],
                "positions": positions,
                "delta": deltas[0],
            }
        )
    assert len(a_candidates) == 1, "expected a unique fully visible front sprite"
    a_info = a_candidates[0]
    a_color = int(a_info["color"])

    remaining = [color for color in active_colors if color != a_color]
    pillar_matches: list[tuple[int, int]] = []
    for color in remaining:
        cols = []
        ok = True
        for panel in panels:
            coords = np.argwhere(panel == color)
            uniq_cols = sorted(set(map(int, coords[:, 1])))
            if len(uniq_cols) != 1:
                ok = False
                break
            cols.append(uniq_cols[0])
        if ok and len(set(cols)) == 1:
            pillar_matches.append((color, cols[0]))
    assert len(pillar_matches) == 1, "expected a unique static pillar color"
    pillar_color, pillar_col = pillar_matches[0]

    b_color = next(color for color in remaining if color != pillar_color)
    b_boxes: list[BBox] = []
    b_masks_visible: list[np.ndarray] = []
    for panel in panels:
        bbox, mask = _color_shape(panel, b_color)
        b_boxes.append(bbox)
        b_masks_visible.append(mask)
    b_shapes = {(b[2] - b[0] + 1, b[3] - b[1] + 1) for b in b_boxes}
    assert len(b_shapes) == 1, "back sprite bbox must stay consistent across frames"
    b_positions = [(b[0], b[1]) for b in b_boxes]
    b_deltas = [(b_positions[i + 1][0] - b_positions[i][0], b_positions[i + 1][1] - b_positions[i][1]) for i in range(len(b_positions) - 1)]
    assert len(set(b_deltas)) == 1 and b_deltas[0] != (0, 0), "back sprite motion must be constant"

    b_mask = np.zeros_like(b_masks_visible[0], dtype=bool)
    for mask in b_masks_visible:
        assert mask.shape == b_mask.shape
        b_mask |= mask
    assert any(np.array_equal(mask, b_mask) for mask in b_masks_visible), "expected at least one unoccluded back-sprite frame"

    return {
        "bg": bg,
        "panels": panels,
        "side": side_h,
        "fill_color": fill_color,
        "pillar_color": int(pillar_color),
        "pillar_col": int(pillar_col),
        "a_color": a_color,
        "a_mask": a_info["mask"],
        "a_positions": a_info["positions"],
        "a_delta": a_info["delta"],
        "b_color": int(b_color),
        "b_mask": b_mask,
        "b_positions": b_positions,
        "b_delta": b_deltas[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Extrapolate the two moving sprites until their first overlap and draw that meeting panel with the inferred occlusion order.

    Input structure:
    - A background surrounds a horizontal row of equal square panels that do not touch each other.
    - Inside every panel, one fixed vertical pillar stands at the same local column, and two differently colored sprites translate by constant per-frame offsets.
    - One sprite is seen occluding the pillar in some input frame, while the other is partially hidden behind the pillar in some input frame; the two sprites never overlap each other in the shown panels.

    Rule:
    - Recover the common panel size, the panel fill, the pillar column, both sprite templates, and both constant motion vectors from the panel sequence.
    - Continue both sprites forward from the last shown panel and find the smallest positive step where their occupied cells first overlap.
    - Output a single standalone panel of the same size, keeping the pillar fixed and layering the sprites as: front sprite above pillar above back sprite.

    Key insight:
    - The output is not merely the next frame: you must solve a future first-meeting time, and the A-over-B order is never shown directly, only implied transitively by A being in front of the pillar while B is behind it.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    result = _first_overlap_step(
        scene["a_mask"],
        scene["a_positions"][-1],
        scene["a_delta"],
        scene["b_mask"],
        scene["b_positions"][-1],
        scene["b_delta"],
        (scene["side"], scene["side"]),
    )
    assert result is not None, "expected a future overlap within panel bounds"
    step, _overlap = result
    a_last = scene["a_positions"][-1]
    b_last = scene["b_positions"][-1]
    a_pos = (a_last[0] + step * scene["a_delta"][0], a_last[1] + step * scene["a_delta"][1])
    b_pos = (b_last[0] + step * scene["b_delta"][0], b_last[1] + step * scene["b_delta"][1])
    return _render_panel(
        scene["side"],
        scene["fill_color"],
        scene["pillar_color"],
        scene["pillar_col"],
        scene["a_color"],
        scene["a_mask"],
        a_pos,
        scene["b_color"],
        scene["b_mask"],
        b_pos,
    )


A_BASE_MASKS = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
]

B_BASE_MASKS = [
    np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=bool),
]

INTERIOR_SIZES = (9,)
SIDE_MARGINS = (1, 2)
GAP_OPTIONS = (1, 2)
T_CHOICES = (1, 2, 2, 3, 3)
COLOR_VALUES = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 500
_BLUEPRINTS: list[dict[str, object]] | None = None


def _random_mask(base_masks: list[np.ndarray]) -> np.ndarray:
    mask = random.choice(base_masks).copy()
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _possible_positions(mask: np.ndarray, side: int) -> list[Coord]:
    mh, mw = mask.shape
    return [(r, c) for r in range(0, side - mh + 1) for c in range(0, side - mw + 1)]


def _pillar_hits(mask: np.ndarray, pos: Coord, pillar_col: int) -> bool:
    return any(c == pillar_col for _r, c in _mask_cells(mask, *pos))


def _sequence_positions(last_pos: Coord, delta: Coord, frames: int) -> list[Coord]:
    return [(last_pos[0] - delta[0] * (frames - 1 - i), last_pos[1] - delta[1] * (frames - 1 - i)) for i in range(frames)]


def _first_rect_overlap_step(
    a_shape: tuple[int, int],
    last_a: Coord,
    delta_a: Coord,
    b_shape: tuple[int, int],
    last_b: Coord,
    delta_b: Coord,
    side: int,
    max_steps: int = 12,
) -> int | None:
    dummy_a = np.ones(a_shape, dtype=bool)
    dummy_b = np.ones(b_shape, dtype=bool)
    for step in range(1, max_steps + 1):
        pos_a = (last_a[0] + step * delta_a[0], last_a[1] + step * delta_a[1])
        pos_b = (last_b[0] + step * delta_b[0], last_b[1] + step * delta_b[1])
        if not _fits_in_panel(dummy_a, pos_a, (side, side)) or not _fits_in_panel(dummy_b, pos_b, (side, side)):
            return None
        if _rect_overlap(pos_a, a_shape, pos_b, b_shape):
            return step
    return None


def _build_input_and_expected(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    side = int(spec["side"])
    gap = int(spec["gap"])
    top_margin = int(spec["top_margin"])
    frames = int(spec["frames"])

    height = side + 2 * top_margin
    width = frames * side + (frames - 1) * gap
    grid = np.full((height, width), int(spec["bg"]), dtype=int)

    for i in range(frames):
        panel = _render_panel(
            side,
            int(spec["fill_color"]),
            int(spec["pillar_color"]),
            int(spec["pillar_col"]),
            int(spec["a_color"]),
            spec["a_mask"],
            spec["a_positions"][i],
            int(spec["b_color"]),
            spec["b_mask"],
            spec["b_positions"][i],
        )
        c0 = i * (side + gap)
        grid[top_margin : top_margin + side, c0 : c0 + side] = panel

    meet_step = int(spec["meet_step"])
    a_last = spec["a_positions"][-1]
    b_last = spec["b_positions"][-1]
    delta_a = spec["delta_a"]
    delta_b = spec["delta_b"]
    a_final = (a_last[0] + meet_step * delta_a[0], a_last[1] + meet_step * delta_a[1])
    b_final = (b_last[0] + meet_step * delta_b[0], b_last[1] + meet_step * delta_b[1])
    expected = _render_panel(
        side,
        int(spec["fill_color"]),
        int(spec["pillar_color"]),
        int(spec["pillar_col"]),
        int(spec["a_color"]),
        spec["a_mask"],
        a_final,
        int(spec["b_color"]),
        spec["b_mask"],
        b_final,
    )
    return grid, expected


def _valid_b_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if w != 3 or not _is_connected(mask):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != mask.shape:
        return False
    if not mask[:, 1].any():
        return False
    visible = mask.copy()
    visible[:, 1] = False
    if not visible.any():
        return False
    coords_vis = np.argwhere(visible)
    vr0, vc0, vr1, vc1 = _bbox_from_coords(coords_vis)
    return (vr1 - vr0 + 1, vc1 - vc0 + 1) == mask.shape


def _valid_a_mask(mask: np.ndarray) -> bool:
    if not _is_connected(mask):
        return False
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    if (r1 - r0 + 1, c1 - c0 + 1) != mask.shape:
        return False
    return int(mask.sum()) < mask.size


assert all(_valid_a_mask(mask) for mask in A_BASE_MASKS)
assert all(_valid_b_mask(mask) for mask in B_BASE_MASKS)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    global _BLUEPRINTS
    if _BLUEPRINTS is None:
        _BLUEPRINTS = _build_blueprints()
    if not _BLUEPRINTS:
        return None

    blueprint = random.choice(_BLUEPRINTS)
    top_margin = random.choice(SIDE_MARGINS)
    bg, fill_color, pillar_color, a_color, b_color = random.sample(COLOR_VALUES, 5)
    spec = {
        **blueprint,
        "bg": bg,
        "top_margin": top_margin,
        "fill_color": fill_color,
        "pillar_color": pillar_color,
        "a_color": a_color,
        "b_color": b_color,
    }
    grid, expected = _build_input_and_expected(spec)
    assert np.array_equal(main(grid), expected)
    return grid, expected


def _mask_variants(base_masks: list[np.ndarray]) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for base in base_masks:
        for flip_u in (False, True):
            for flip_l in (False, True):
                mask = base.copy()
                if flip_u:
                    mask = np.flipud(mask)
                if flip_l:
                    mask = np.fliplr(mask)
                if not any(np.array_equal(mask, prior) for prior in variants):
                    variants.append(mask.copy())
    return variants


def _build_blueprints(limit: int = 32) -> list[dict[str, object]]:
    side = INTERIOR_SIZES[0]
    frames = 3
    gap = 1
    pillar_col = side // 2
    side_shape = (side, side)
    a_variants = _mask_variants(A_BASE_MASKS)
    b_variants = _mask_variants(B_BASE_MASKS)
    vectors = []
    for sign in (-1, 1):
        for dra in (-1, 0, 1):
            for drb in (-1, 0, 1):
                delta_a = (dra, sign)
                delta_b = (drb, -sign)
                if delta_a[0] == 0 and delta_b[0] == 0:
                    continue
                vectors.append((delta_a, delta_b))

    fixed_colors = {
        "bg": 0,
        "fill_color": 1,
        "pillar_color": 2,
        "a_color": 3,
        "b_color": 4,
    }
    blueprints: list[dict[str, object]] = []

    for a_mask in a_variants:
        for b_mask in b_variants:
            if a_mask.shape == b_mask.shape and np.array_equal(a_mask, b_mask):
                continue
            positions_a = _possible_positions(a_mask, side)
            positions_b = _possible_positions(b_mask, side)
            for delta_a, delta_b in vectors:
                for desired_t in (1, 2, 3):
                    for last_a in positions_a:
                        a_positions = _sequence_positions(last_a, delta_a, frames)
                        if not all(_fits_in_panel(a_mask, pos, side_shape) for pos in a_positions):
                            continue
                        if not any(_pillar_hits(a_mask, pos, pillar_col) for pos in a_positions):
                            continue

                        for last_b in positions_b:
                            b_positions = _sequence_positions(last_b, delta_b, frames)
                            if not all(_fits_in_panel(b_mask, pos, side_shape) for pos in b_positions):
                                continue
                            if sum(_pillar_hits(b_mask, pos, pillar_col) for pos in b_positions) != 1:
                                continue
                            if any(_mask_overlap_cells(a_mask, a_positions[i], b_mask, b_positions[i]) for i in range(frames)):
                                continue

                            actual = _first_overlap_step(
                                a_mask,
                                last_a,
                                delta_a,
                                b_mask,
                                last_b,
                                delta_b,
                                side_shape,
                                max_steps=desired_t,
                            )
                            if actual is None or actual[0] != desired_t:
                                continue
                            overlap = actual[1]
                            if any(c == pillar_col for _r, c in overlap):
                                continue
                            if not (0 < len(overlap) < int(min(a_mask.sum(), b_mask.sum()))):
                                continue

                            spec = {
                                "side": side,
                                "gap": gap,
                                "frames": frames,
                                "pillar_col": pillar_col,
                                "a_mask": a_mask.copy(),
                                "a_positions": list(a_positions),
                                "delta_a": delta_a,
                                "b_mask": b_mask.copy(),
                                "b_positions": list(b_positions),
                                "delta_b": delta_b,
                                "meet_step": desired_t,
                                "top_margin": 1,
                                **fixed_colors,
                            }
                            grid, expected = _build_input_and_expected(spec)
                            try:
                                parsed = _extract_scene(grid)
                                solved = main(grid)
                            except AssertionError:
                                continue
                            if not np.array_equal(parsed["a_mask"], a_mask) or not np.array_equal(parsed["b_mask"], b_mask):
                                continue
                            if parsed["pillar_col"] != pillar_col:
                                continue
                            if not np.array_equal(solved, expected):
                                continue

                            a_final = (last_a[0] + desired_t * delta_a[0], last_a[1] + desired_t * delta_a[1])
                            b_final = (last_b[0] + desired_t * delta_b[0], last_b[1] + desired_t * delta_b[1])
                            wrong = _render_panel(
                                side,
                                fixed_colors["fill_color"],
                                fixed_colors["pillar_color"],
                                pillar_col,
                                fixed_colors["a_color"],
                                a_mask,
                                a_final,
                                fixed_colors["b_color"],
                                b_mask,
                                b_final,
                                wrong_order=True,
                            )
                            if np.array_equal(wrong, expected):
                                continue

                            rect_panel = _render_panel(
                                side,
                                fixed_colors["fill_color"],
                                fixed_colors["pillar_color"],
                                pillar_col,
                                fixed_colors["a_color"],
                                a_mask,
                                a_final,
                                fixed_colors["b_color"],
                                b_mask,
                                b_final,
                                rect_mode=True,
                            )
                            if np.array_equal(rect_panel, expected):
                                continue

                            blueprints.append(
                                {
                                    "side": side,
                                    "gap": gap,
                                    "frames": frames,
                                    "pillar_col": pillar_col,
                                    "a_mask": a_mask.copy(),
                                    "a_positions": list(a_positions),
                                    "delta_a": delta_a,
                                    "b_mask": b_mask.copy(),
                                    "b_positions": list(b_positions),
                                    "delta_b": delta_b,
                                    "meet_step": desired_t,
                                }
                            )
                            if len(blueprints) >= limit:
                                return blueprints
    return blueprints


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid first-overlap occlusion puzzle")
