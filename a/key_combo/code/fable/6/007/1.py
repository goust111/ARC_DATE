from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_6_007_1",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

bbox = _ARC_UTILS.bbox
components_mask = _ARC_UTILS.components_mask
is_contiguous = _ARC_UTILS.is_contiguous
paste = _ARC_UTILS.paste


GEN_ATTEMPTS = 500
PANEL_COUNTS = (3, 4)
INTERIOR_HEIGHTS = (7, 8, 9, 10, 11, 12)
INTERIOR_WIDTHS = (5, 6)
SPRITE_HEIGHTS = (3, 4)
SPRITE_WIDTHS = (3, 4)
OCCLUDER_SIZES = ((2, 2), (2, 3), (3, 2))
MOTIONS = [
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
]
ALL_COLORS = tuple(range(10))


def _unique_most_common(values) -> int:
    ranked = Counter(map(int, values)).most_common()
    assert ranked and (len(ranked) == 1 or ranked[0][1] > ranked[1][1])
    return int(ranked[0][0])


def _shift_mask(mask: np.ndarray, dr: int, dc: int, *, strict: bool = False) -> np.ndarray | None:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    if dr >= 0:
        src_r = slice(0, h - dr)
        dst_r = slice(dr, h)
    else:
        src_r = slice(-dr, h)
        dst_r = slice(0, h + dr)

    if dc >= 0:
        src_c = slice(0, w - dc)
        dst_c = slice(dc, w)
    else:
        src_c = slice(-dc, w)
        dst_c = slice(0, w + dc)

    if (h - abs(dr) <= 0) or (w - abs(dc) <= 0):
        return None if (strict and mask.any()) else out

    out[dst_r, dst_c] = mask[src_r, src_c]
    if strict and int(out.sum()) != int(mask.sum()):
        return None
    return out


def _full_rectangle(mask: np.ndarray) -> bool:
    box = bbox(mask)
    if box is None:
        return False
    r0, c0, r1, c1 = box
    return bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1]))


def _make_panel(
    visible_mask: np.ndarray,
    occluder_mask: np.ndarray,
    *,
    frame: int,
    fill: int,
    sprite_color: int,
    occluder_color: int,
) -> np.ndarray:
    ih, iw = visible_mask.shape
    panel = np.full((ih + 2, iw + 2), int(fill), dtype=int)
    panel[0, :] = int(frame)
    panel[-1, :] = int(frame)
    panel[:, 0] = int(frame)
    panel[:, -1] = int(frame)
    interior = panel[1:-1, 1:-1]
    interior[visible_mask] = int(sprite_color)
    interior[occluder_mask] = int(occluder_color)
    return panel


def _render_input(
    sprite_mask: np.ndarray,
    occluder_mask: np.ndarray,
    motion: tuple[int, int],
    panel_count: int,
    colors: tuple[int, int, int, int],
) -> np.ndarray:
    frame, fill, sprite_color, occluder_color = map(int, colors)
    ih, iw = sprite_mask.shape
    panel_w = iw + 2
    stride = panel_w - 1
    panels = []
    for i in range(panel_count):
        shifted = _shift_mask(sprite_mask, i * motion[0], i * motion[1], strict=True)
        assert shifted is not None
        panels.append(
            _make_panel(
                shifted & ~occluder_mask,
                occluder_mask,
                frame=frame,
                fill=fill,
                sprite_color=sprite_color,
                occluder_color=occluder_color,
            )
        )
    out = np.full((ih + 2, panel_count * stride + 1), int(fill), dtype=int)
    for i, panel in enumerate(panels):
        paste(out, panel, 0, i * stride)
    return out


def _render_output_panel(
    sprite_mask: np.ndarray,
    occluder_mask: np.ndarray,
    motion: tuple[int, int],
    step_index: int,
    colors: tuple[int, int, int, int],
) -> np.ndarray:
    frame, fill, sprite_color, occluder_color = map(int, colors)
    shifted = _shift_mask(sprite_mask, step_index * motion[0], step_index * motion[1], strict=True)
    assert shifted is not None
    return _make_panel(
        shifted & ~occluder_mask,
        occluder_mask,
        frame=frame,
        fill=fill,
        sprite_color=sprite_color,
        occluder_color=occluder_color,
    )


def _parse_panels(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    frame = int(g[0, 0])
    assert h >= 5 and w >= 7
    assert np.all(g[0, :] == frame) and np.all(g[-1, :] == frame)

    separators = [c for c in range(w) if np.all(g[:, c] == frame)]
    assert len(separators) >= 4
    assert separators[0] == 0 and separators[-1] == w - 1
    strides = [b - a for a, b in zip(separators, separators[1:])]
    assert len(set(strides)) == 1
    stride = strides[0]
    panel_w = stride + 1

    panels = []
    for c0, c1 in zip(separators, separators[1:]):
        panel = g[:, c0 : c1 + 1]
        assert panel.shape == (h, panel_w)
        assert np.all(panel[:, 0] == frame) and np.all(panel[:, -1] == frame)
        panels.append(panel)

    interiors = [panel[1:-1, 1:-1].copy() for panel in panels]
    fill = _unique_most_common(np.concatenate([x.ravel() for x in interiors]))

    stable_candidates = []
    palette = sorted(set(map(int, g.ravel())) - {frame, fill})
    for color in palette:
        stable = np.logical_and.reduce([interior == color for interior in interiors])
        if stable.any() and _full_rectangle(stable):
            stable_candidates.append((int(stable.sum()), color, stable))
    assert stable_candidates
    stable_candidates.sort(key=lambda item: (-item[0], item[1]))
    best_area = stable_candidates[0][0]
    best = [item for item in stable_candidates if item[0] == best_area]
    assert len(best) == 1
    _, occluder_color, occluder_mask = best[0]

    visible_masks = []
    sprite_colors = set()
    for interior in interiors:
        visible = (interior != fill) & ~occluder_mask
        assert visible.any()
        visible_masks.append(visible)
        sprite_colors.update(map(int, interior[visible]))
    assert len(sprite_colors) == 1
    sprite_color = next(iter(sprite_colors))
    assert sprite_color != occluder_color

    return {
        "frame": frame,
        "fill": fill,
        "panels": panels,
        "interiors": interiors,
        "visible_masks": visible_masks,
        "sprite_color": sprite_color,
        "occluder_color": int(occluder_color),
        "occluder_mask": occluder_mask,
        "interior_shape": interiors[0].shape,
        "panel_w": panel_w,
        "stride": stride,
    }


def _find_solutions(visible_masks: list[np.ndarray], occluder_mask: np.ndarray) -> list[dict]:
    ih, iw = occluder_mask.shape
    panel_count = len(visible_masks)
    solutions = []
    seen = set()

    for dr in range(-(ih - 1), ih):
        for dc in range(-(iw - 1), iw):
            if dr == 0 and dc == 0:
                continue
            base = np.zeros((ih, iw), dtype=bool)
            valid = True
            for i, visible in enumerate(visible_masks):
                shifted_back = _shift_mask(visible, -i * dr, -i * dc, strict=True)
                if shifted_back is None:
                    valid = False
                    break
                base |= shifted_back
            if not valid or not base.any():
                continue
            if len(components_mask(base, connectivity=4)) != 1:
                continue

            occluded_signatures = []
            for i, visible in enumerate(visible_masks):
                shifted = _shift_mask(base, i * dr, i * dc, strict=True)
                if shifted is None:
                    valid = False
                    break
                if not np.array_equal(shifted & ~occluder_mask, visible):
                    valid = False
                    break
                overlap = shifted & occluder_mask
                if not overlap.any() or int(overlap.sum()) == int(shifted.sum()):
                    valid = False
                    break
                occluded_signatures.append(overlap.tobytes())
            if not valid:
                continue
            if len(set(occluded_signatures)) != panel_count:
                continue

            key = (dr, dc, base.tobytes())
            if key in seen:
                continue
            seen.add(key)
            solutions.append({"motion": (dr, dc), "sprite_mask": base})
    return solutions


def _solve(grid: np.ndarray) -> dict:
    parsed = _parse_panels(grid)
    solutions = _find_solutions(parsed["visible_masks"], parsed["occluder_mask"])
    assert len(solutions) == 1
    solution = solutions[0]
    parsed.update(solution)
    return parsed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a moving sprite from differently occluded framed panels and extrapolate the next standalone panel.

    Input structure:
    - The grid is one horizontal strip of equal-width framed panels that share their vertical border columns.
    - Every panel has the same interior fill color and the same solid rectangular occluder at the same local position.
    - One monochrome multi-cell sprite moves by a constant local displacement from panel to panel, and any overlap with the occluder is hidden by the occluder color.
    - In every observed panel the sprite is incomplete, and the occluded subset differs across frames.

    Rule:
    - Use the relative offsets between visible sprite fragments to infer the unique constant displacement.
    - Shift the visible fragments back into one common coordinate system and take their union to recover the full sprite shape.
    - Output one standalone panel with the same frame, fill color, and occluder, placing the recovered full sprite one more step forward than the last observed panel.
    - Any cells where that advanced sprite overlaps the occluder remain covered by the occluder.

    Key insight:
    - The apparent shape changes are only occlusion artifacts: the true sprite is a rigid object that must be assembled across frames before the next frame can be drawn.
    """
    solved = _solve(grid)
    return _render_output_panel(
        solved["sprite_mask"],
        solved["occluder_mask"],
        solved["motion"],
        len(solved["visible_masks"]),
        (
            solved["frame"],
            solved["fill"],
            solved["sprite_color"],
            solved["occluder_color"],
        ),
    )


def _random_sprite_patch(h: int, w: int) -> np.ndarray | None:
    min_area = max(5, h + w - 1)
    max_area = min(h * w - 1, 8)
    if min_area > max_area:
        return None
    for _ in range(80):
        area = random.randint(min_area, max_area)
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < area:
            r, c = random.choice(tuple(cells))
            neighbors = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
            ]
            random.shuffle(neighbors)
            for rr, cc in neighbors:
                if 0 <= rr < h and 0 <= cc < w:
                    cells.add((rr, cc))
                    break
        patch = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            patch[r, c] = True
        if not is_contiguous(patch.astype(int), background=0, connectivity=4):
            continue
        box = bbox(patch)
        if box != (0, 0, h - 1, w - 1):
            continue
        if np.all(patch) or np.all(patch == patch[0, 0]):
            continue
        return patch
    return None


def _embed_patch(patch: np.ndarray, shape: tuple[int, int], top: int, left: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[top : top + patch.shape[0], left : left + patch.shape[1]] = patch
    return out


def _all_shifted_masks(sprite_mask: np.ndarray, motion: tuple[int, int], panel_count: int) -> list[np.ndarray] | None:
    shifted = []
    for i in range(panel_count + 1):
        moved = _shift_mask(sprite_mask, i * motion[0], i * motion[1], strict=True)
        if moved is None:
            return None
        shifted.append(moved)
    return shifted


def _spec_semantics_ok(sprite_mask: np.ndarray, occluder_mask: np.ndarray, motion: tuple[int, int], panel_count: int) -> bool:
    shifted = _all_shifted_masks(sprite_mask, motion, panel_count)
    if shifted is None:
        return False
    visible = [mask & ~occluder_mask for mask in shifted[:panel_count]]
    overlaps = [mask & occluder_mask for mask in shifted[:panel_count]]
    next_visible = shifted[panel_count] & ~occluder_mask
    next_overlap = shifted[panel_count] & occluder_mask

    if any((not v.any()) or (not o.any()) for v, o in zip(visible, overlaps)):
        return False
    if any(int(o.sum()) == int(s.sum()) for o, s in zip(overlaps, shifted[:panel_count])):
        return False
    if len({o.tobytes() for o in overlaps}) != panel_count:
        return False
    if not next_visible.any() or not next_overlap.any():
        return False

    aligned_union = np.zeros_like(sprite_mask, dtype=bool)
    for i, fragment in enumerate(visible):
        back = _shift_mask(fragment, -i * motion[0], -i * motion[1], strict=True)
        if back is None:
            return False
        aligned_union |= back
    if not np.array_equal(aligned_union, sprite_mask):
        return False

    raw_union = np.logical_or.reduce(visible)
    if np.array_equal(raw_union, sprite_mask):
        return False

    indispensable = 0
    for omit in range(panel_count):
        partial = np.zeros_like(sprite_mask, dtype=bool)
        for i, fragment in enumerate(visible):
            if i == omit:
                continue
            back = _shift_mask(fragment, -i * motion[0], -i * motion[1], strict=True)
            if back is None:
                return False
            partial |= back
        if not np.array_equal(partial, sprite_mask):
            indispensable += 1
    if indispensable < 2:
        return False

    last_only = _shift_mask(visible[-1], motion[0], motion[1], strict=True)
    if last_only is None:
        return False
    if np.array_equal(last_only & ~occluder_mask, next_visible):
        return False

    return True


def _choose_occluder(sprite_mask: np.ndarray, motion: tuple[int, int], panel_count: int) -> np.ndarray | None:
    ih, iw = sprite_mask.shape
    candidates = []
    for oh, ow in OCCLUDER_SIZES:
        if oh >= ih or ow >= iw:
            continue
        for top in range(1, ih - oh):
            for left in range(1, iw - ow):
                occ = np.zeros((ih, iw), dtype=bool)
                occ[top : top + oh, left : left + ow] = True
                candidates.append(occ)
    random.shuffle(candidates)
    for occ in candidates:
        if _spec_semantics_ok(sprite_mask, occ, motion, panel_count):
            return occ
    return None


def _sample_sprite(interior_shape: tuple[int, int], motion: tuple[int, int], panel_count: int) -> np.ndarray | None:
    ih, iw = interior_shape
    sh = random.choice([x for x in SPRITE_HEIGHTS if x <= ih - 2])
    sw = random.choice([x for x in SPRITE_WIDTHS if x <= iw - 2])
    patch = _random_sprite_patch(sh, sw)
    if patch is None:
        return None

    total_steps = panel_count
    min_top = max(0, -total_steps * motion[0])
    max_top = ih - sh - max(0, total_steps * motion[0])
    min_left = max(0, -total_steps * motion[1])
    max_left = iw - sw - max(0, total_steps * motion[1])
    if min_top > max_top or min_left > max_left:
        return None

    tops = list(range(min_top, max_top + 1))
    lefts = list(range(min_left, max_left + 1))
    random.shuffle(tops)
    random.shuffle(lefts)
    for top in tops:
        for left in lefts:
            sprite = _embed_patch(patch, (ih, iw), top, left)
            if _all_shifted_masks(sprite, motion, panel_count) is not None:
                return sprite
    return None


def _build_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    panel_count = random.choice(PANEL_COUNTS)
    ih = random.choice(INTERIOR_HEIGHTS)
    iw = random.choice(INTERIOR_WIDTHS)
    if panel_count * (iw + 1) + 1 > 30 or ih + 2 > 30:
        return None

    colors = tuple(random.sample(ALL_COLORS, 4))
    motion_choices = MOTIONS[:]
    random.shuffle(motion_choices)

    for motion in motion_choices:
        sprite = _sample_sprite((ih, iw), motion, panel_count)
        if sprite is None:
            continue
        occluder = _choose_occluder(sprite, motion, panel_count)
        if occluder is None:
            continue

        inp = _render_input(sprite, occluder, motion, panel_count, colors)
        out = _render_output_panel(sprite, occluder, motion, panel_count, colors)

        solved = _solve(inp)
        if solved["motion"] != motion:
            continue
        if not np.array_equal(solved["sprite_mask"], sprite):
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        pair = _build_candidate()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid occluded-frame extrapolation input")
