from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import random

import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

_MASK_BANK: list[np.ndarray] = []
for _bits in range(1, 1 << 9):
    _mask = np.array([(_bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
    if int(_mask.sum()) < 5:
        continue
    if not (_mask[0].any() and _mask[2].any() and _mask[:, 0].any() and _mask[:, 2].any()):
        continue
    coords = [tuple(map(int, rc)) for rc in np.argwhere(_mask)]
    seen = {coords[0]}
    queue = deque([coords[0]])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < 3 and 0 <= nc < 3 and _mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    if len(seen) == len(coords):
        _MASK_BANK.append(_mask.copy())


@dataclass(frozen=True)
class _Component:
    bbox: tuple[int, int, int, int]
    crop: np.ndarray
    center: tuple[int, int]
    marker_count: int


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    neighbors = _NEIGHBORS4 if connectivity == 4 else _NEIGHBORS8
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in neighbors:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) == 2, "expected exactly two foreground colors"

    counts = Counter(map(int, g[g != bg].reshape(-1)))
    least = min(counts[color] for color in colors)
    marker_candidates = [color for color in colors if counts[color] == least]
    assert len(marker_candidates) == 1, "expected a unique globally rare marker color"
    marker = int(marker_candidates[0])
    main_color = int([color for color in colors if color != marker][0])

    comps_raw = _find_components(g, bg=bg, connectivity=8)
    assert len(comps_raw) >= 2

    components: list[_Component] = []
    for coords in comps_raw:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "every sprite must have a 3x3 bbox"
        crop = _crop_bbox(g, bbox)
        marker_count = int(np.count_nonzero(crop == marker))
        components.append(
            _Component(
                bbox=bbox,
                crop=crop,
                center=(r0 + 1, c0 + 1),
                marker_count=marker_count,
            )
        )

    winner_count = max(comp.marker_count for comp in components)
    winner_indices = [i for i, comp in enumerate(components) if comp.marker_count == winner_count]
    assert len(winner_indices) == 1, "expected a unique template winner"
    winner_idx = int(winner_indices[0])

    row_span = max(comp.center[0] for comp in components) - min(comp.center[0] for comp in components)
    col_span = max(comp.center[1] for comp in components) - min(comp.center[1] for comp in components)
    horizontal = bool(col_span > row_span)

    if horizontal:
        keys = [comp.center[1] for comp in components]
        assert len(set(keys)) == len(keys), "horizontal layout requires distinct column order"
        ordered = sorted(range(len(components)), key=lambda i: components[i].center[1])
    else:
        keys = [comp.center[0] for comp in components]
        assert len(set(keys)) == len(keys), "vertical layout requires distinct row order"
        ordered = sorted(range(len(components)), key=lambda i: components[i].center[0])

    return {
        "grid": g,
        "bg": bg,
        "main_color": main_color,
        "marker": marker,
        "components": components,
        "winner_idx": winner_idx,
        "horizontal": horizontal,
        "ordered": ordered,
    }


def _render_from_analysis(
    info: dict,
    *,
    template_idx: int | None = None,
    horizontal: bool | None = None,
    alternate: bool = True,
    start_with_original: bool = True,
) -> np.ndarray:
    g = info["grid"]
    bg = int(info["bg"])
    components: list[_Component] = info["components"]
    template = components[info["winner_idx"] if template_idx is None else int(template_idx)].crop
    horizontal = info["horizontal"] if horizontal is None else bool(horizontal)
    mirrored = np.fliplr(template) if horizontal else np.flipud(template)

    out = g.copy()
    for comp in components:
        r0, c0, r1, c1 = comp.bbox
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for pos, idx in enumerate(info["ordered"]):
        comp = components[idx]
        use_original = start_with_original if pos % 2 == 0 else not start_with_original
        patch = template if (use_original or not alternate) else mirrored
        r0, c0, _r1, _c1 = comp.bbox
        out[r0 : r0 + 3, c0 : c0 + 3] = patch
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the uniquely most marker-dense 3x3 sprite and replace the whole layout with an alternating original/mirrored sequence of that template.

    Input structure:
    - A black background contains several separated 3x3 foreground sprites built from exactly two non-background colors.
    - One foreground color is the main sprite color and the other is a globally rarer marker color.
    - The sprite centers lie mostly along one axis, so their column span is larger than their row span for horizontal layouts, otherwise the layout is vertical.

    Rule:
    - Identify the marker color as the unique less frequent foreground color over the whole grid.
    - Count marker-color pixels inside each 3x3 sprite and choose the unique sprite with the largest count as the template.
    - Infer the layout axis from the centers: horizontal if column span exceeds row span, otherwise vertical.
    - Mirror the template across the axis perpendicular to that layout axis: left-right for horizontal layouts, up-down for vertical layouts.
    - Clear every original 3x3 sprite region, then refill the original sprite positions in layout order with template, mirror, template, mirror, ... starting with the unmirrored template.

    Key insight:
    - Marker density chooses which sprite becomes the template, while the global layout chooses how that template is mirrored and which ordered positions receive the original versus the mirror.
    """
    info = _analyze(grid)
    return _render_from_analysis(info)


def _sample_axis_starts(length: int, count: int) -> list[int]:
    margin = 1
    min_gap = 1
    required = 2 * margin + 3 * count + min_gap * (count - 1)
    assert length >= required
    slack = int(length - required)
    extras = _RNG.multinomial(slack, np.ones(count + 1) / (count + 1))
    starts = [margin + int(extras[0])]
    for i in range(1, count):
        starts.append(starts[-1] + 3 + min_gap + int(extras[i]))
    return starts


def _sample_cross_starts(length: int, count: int) -> list[int]:
    margin = 1
    start_min = margin
    start_max = length - margin - 3
    assert start_max >= start_min
    span = min(2, start_max - start_min)
    band = int(_RNG.integers(0, span + 1)) if span > 0 else 0
    base = int(_RNG.integers(start_min, start_max - band + 1))
    values = [base + int(_RNG.integers(0, band + 1)) for _ in range(count)]
    if band > 0 and len(set(values)) == 1:
        values[-1] = base if values[0] != base else base + band
    return values


def _make_colored_crop(
    mask: np.ndarray,
    main_color: int,
    marker_color: int,
    marker_count: int,
    *,
    require_asym_both: bool = False,
) -> np.ndarray | None:
    occupied = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if not (0 <= marker_count < len(occupied)):
        return None

    for _ in range(40):
        marker_cells = set(random.sample(occupied, marker_count))
        crop = np.zeros((3, 3), dtype=int)
        for r, c in occupied:
            crop[r, c] = marker_color if (r, c) in marker_cells else main_color
        if require_asym_both and (np.array_equal(crop, np.fliplr(crop)) or np.array_equal(crop, np.flipud(crop))):
            continue
        return crop
    return None


def _generate_components(
    count: int,
    main_color: int,
    marker_color: int,
) -> list[np.ndarray] | None:
    template_mask = _MASK_BANK[int(_RNG.integers(0, len(_MASK_BANK)))]
    winner_cap = min(4, int(template_mask.sum()) - 1)
    if winner_cap < 2:
        return None
    winner_count = int(_RNG.integers(2, winner_cap + 1))
    template = _make_colored_crop(
        template_mask,
        main_color,
        marker_color,
        winner_count,
        require_asym_both=True,
    )
    if template is None:
        return None

    crops = [template]
    positive_marker_sprites = 1
    total_area = int(template_mask.sum())
    total_markers = winner_count

    for _ in range(count - 1):
        mask = _MASK_BANK[int(_RNG.integers(0, len(_MASK_BANK)))]
        max_count = min(winner_count - 1, int(mask.sum()) - 1)
        if max_count < 0:
            return None
        choices = list(range(max_count + 1))
        weights = np.array([max_count + 2 - v for v in choices], dtype=float)
        marker_count = int(_RNG.choice(choices, p=weights / weights.sum()))
        crop = _make_colored_crop(mask, main_color, marker_color, marker_count)
        if crop is None:
            return None
        crops.append(crop)
        total_area += int(mask.sum())
        total_markers += marker_count
        positive_marker_sprites += int(marker_count > 0)

    if positive_marker_sprites < 2:
        return None
    if total_markers * 2 >= total_area:
        return None
    if all(np.array_equal(crop, template) for crop in crops[1:]):
        return None

    _RNG.shuffle(crops)
    return crops


def _place_components(
    crops: list[np.ndarray],
    *,
    horizontal: bool,
) -> np.ndarray:
    count = len(crops)
    axis_required = 2 + 3 * count + (count - 1)
    if horizontal:
        H = int(_RNG.integers(13, 19))
        W = int(_RNG.integers(axis_required, 31))
        axis_starts = _sample_axis_starts(W, count)
        cross_starts = _sample_cross_starts(H, count)
        placements = [(int(r0), int(c0)) for r0, c0 in zip(cross_starts, axis_starts)]
        shape = (H, W)
    else:
        H = int(_RNG.integers(axis_required, 31))
        W = int(_RNG.integers(13, 19))
        axis_starts = _sample_axis_starts(H, count)
        cross_starts = _sample_cross_starts(W, count)
        placements = [(int(r0), int(c0)) for r0, c0 in zip(axis_starts, cross_starts)]
        shape = (H, W)

    grid = np.zeros(shape, dtype=int)
    for crop, (r0, c0) in zip(crops, placements):
        grid[r0 : r0 + 3, c0 : c0 + 3] = crop
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except Exception:
        return False

    components: list[_Component] = info["components"]
    if len(components) not in {5, 7}:
        return False
    if np.array_equal(info["grid"], _render_from_analysis(info)):
        return False

    template = components[info["winner_idx"]].crop
    if np.array_equal(template, np.fliplr(template)) or np.array_equal(template, np.flipud(template)):
        return False

    baseline = _render_from_analysis(info)

    other_indices = [i for i in range(len(components)) if i != info["winner_idx"]]
    runner = max(other_indices, key=lambda i: components[i].marker_count)
    alternatives = [
        _render_from_analysis(info, template_idx=runner),
        _render_from_analysis(info, horizontal=not info["horizontal"]),
        _render_from_analysis(info, alternate=False),
        _render_from_analysis(info, start_with_original=False),
    ]
    if any(np.array_equal(baseline, alt) for alt in alternatives):
        return False

    changed = 0
    for comp, idx in zip((components[i] for i in info["ordered"]), range(len(info["ordered"]))):
        r0, c0, _r1, _c1 = comp.bbox
        out_patch = baseline[r0 : r0 + 3, c0 : c0 + 3]
        if not np.array_equal(out_patch, comp.crop):
            changed += 1
    return changed >= 2


def _generate_attempt() -> np.ndarray | None:
    count = int(_RNG.choice([5, 7]))
    horizontal = bool(_RNG.integers(0, 2))
    main_color, marker_color = map(int, _RNG.choice(np.arange(1, 10), size=2, replace=False))
    crops = _generate_components(count, main_color, marker_color)
    if crops is None:
        return None
    grid = _place_components(crops, horizontal=horizontal)
    if not _semantic_checks(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
