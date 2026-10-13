from __future__ import annotations

from collections import Counter, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(np.asarray(sprite, dtype=int), np.ones((int(factor), int(factor)), dtype=int))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_components(grid: np.ndarray, bg: int, colors: list[int] | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    if colors is None:
        colors = sorted(set(map(int, np.unique(g))) - {int(bg)})
    out: list[dict] = []
    for color in colors:
        for coords in _components_mask(g == int(color)):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox_coords(coords),
                    "area": int(len(coords)),
                }
            )
    return out


def _parse_legend(grid: np.ndarray, bg: int) -> list[int]:
    colors: list[int] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color == int(bg):
            break
        colors.append(color)
    assert colors, "expected a non-empty legend in column 0"
    assert len(colors) == len(set(colors)), "legend colors must be unique"
    return colors


def _paint_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    r1 = int(r0) + h
    c1 = int(c0) + w
    assert 0 <= r0 and 0 <= c0 and r1 <= dst.shape[0] and c1 <= dst.shape[1], "scaled object out of bounds"
    view = dst[r0:r1, c0:c1]
    mask = patch != int(bg)
    view[mask] = patch[mask]


def _object_sprite(canvas: np.ndarray, color: int, coords: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    sprite[coords[:, 0] - r0, coords[:, 1] - c0] = int(color)
    return sprite, (int(r0), int(c0))


def main(grid: np.ndarray) -> np.ndarray:
    """Scale each legend color's objects by the legend row number and resolve overlaps by legend priority.

    Input structure:
    - A background-colored grid contains a single-column legend in column 0, starting at row 0.
    - Each legend row holds one unique color sample; every non-background pixel outside the legend column
      belongs to a monochromatic 4-connected object whose color appears in the legend.
    - Objects may contain background gaps inside their own bounding boxes.

    Rule:
    - Clear the legend column.
    - If a color appears on legend row k (counting from 1), then every object of that color is scaled by
      factor k about its own bounding-box top-left corner: each pixel becomes a k x k block, and background
      gaps inside the object's bbox scale with the same factor.
    - Draw scaled objects from the last legend row back to the first, so earlier legend rows overwrite later
      rows wherever scaled objects overlap.

    Key insight:
    - The legend row index is the only control signal: it simultaneously names the scale factor and the
      z-order priority for every object of that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = get_bg(g)
    legend = _parse_legend(g, bg)
    scale_by_color = {color: idx + 1 for idx, color in enumerate(legend)}

    canvas = g.copy()
    canvas[:, 0] = int(bg)
    non_bg = canvas != int(bg)
    if non_bg.any():
        assert set(map(int, np.unique(canvas[non_bg]))).issubset(set(legend)), "unexpected non-legend color"

    comps = _find_components(canvas, bg=bg, colors=legend)
    assert comps, "expected at least one object outside the legend"
    by_color: dict[int, list[dict]] = {color: [] for color in legend}
    for comp in comps:
        by_color[int(comp["color"])].append(comp)
    assert all(by_color[color] for color in legend), "every legend color must appear in at least one object"

    out = np.full_like(g, int(bg))
    for color in reversed(legend):
        scale = scale_by_color[int(color)]
        for comp in by_color[int(color)]:
            sprite, (r0, c0) = _object_sprite(canvas, int(color), np.asarray(comp["coords"], dtype=int), bg)
            scaled = scale_sprite(sprite, scale)
            _paint_patch(out, scaled, r0, c0, bg)
    return out


GEN_RETRIES = 600
GRID_H_RANGE = (18, 27)
GRID_W_RANGE = (19, 31)
COLOR_COUNT_RANGE = (2, 4)
EXTRA_OBJECTS = 1
MOAT = 1


def _rect_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]


def _rect_expand(rect: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    return (
        max(0, rect[0] - int(pad)),
        max(0, rect[1] - int(pad)),
        min(shape[0] - 1, rect[2] + int(pad)),
        min(shape[1] - 1, rect[3] + int(pad)),
    )


def _rect_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


def _mask_connected(mask: np.ndarray) -> bool:
    return len(_components_mask(mask.astype(bool))) == 1


def _safe_remove_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    safe: list[tuple[int, int]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            trial = mask.copy()
            trial[r, c] = False
            if not trial.any():
                continue
            if not trial[0, :].any() or not trial[-1, :].any() or not trial[:, 0].any() or not trial[:, -1].any():
                continue
            if _mask_connected(trial):
                safe.append((r, c))
    return safe


def _make_shape(rng: np.random.Generator, scale: int, want_hole: bool = False) -> np.ndarray:
    min_side = 3 if want_hole else 2
    max_side = 5 if scale == 1 else 4
    h = int(rng.integers(min_side, max_side + 1))
    w = int(rng.integers(min_side, max_side + 1))

    if want_hole and h >= 3 and w >= 3:
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        if h >= 4 and w >= 4 and rng.random() < 0.5:
            mask[1:-1, 1:-1] = True
            hole_r = int(rng.integers(1, h - 1))
            hole_c = int(rng.integers(1, w - 1))
            mask[hole_r, hole_c] = False
        return mask

    mask = np.ones((h, w), dtype=bool)
    min_fill = max(h + w - 1, int(np.ceil(h * w * rng.uniform(0.55, 0.9))))
    while mask.sum() > min_fill:
        safe = _safe_remove_cells(mask)
        if not safe:
            break
        rr, cc = safe[int(rng.integers(len(safe)))]
        mask[rr, cc] = False
        if want_hole and h >= 3 and w >= 3 and (~mask[1:-1, 1:-1]).any():
            break
    return mask


def _scaled_rect(anchor: tuple[int, int], shape: tuple[int, int], scale: int) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    h, w = shape
    return (int(r0), int(c0), int(r0 + h * scale - 1), int(c0 + w * scale - 1))


def _input_rect(anchor: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    h, w = shape
    return (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))


def _place_mask(grid: np.ndarray, mask: np.ndarray, anchor: tuple[int, int], color: int) -> None:
    r0, c0 = anchor
    h, w = mask.shape
    view = grid[r0 : r0 + h, c0 : c0 + w]
    view[mask] = int(color)


def _scaled_patch(mask: np.ndarray, color: int, scale: int, bg: int) -> np.ndarray:
    sprite = np.where(mask, int(color), int(bg)).astype(int)
    return scale_sprite(sprite, scale)


def _can_place_input_rect(rect: tuple[int, int, int, int], occupied: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> bool:
    expanded = _rect_expand(rect, MOAT, shape)
    return all(_rect_gap(expanded, other) for other in occupied)


def _place_forced_overlap(
    rng: np.random.Generator,
    h: int,
    w: int,
    hi_scale: int,
    lo_scale: int,
    bg: int,
) -> tuple[dict, dict] | None:
    high_mask = _make_shape(rng, hi_scale, want_hole=False)
    low_mask = _make_shape(rng, lo_scale, want_hole=False)
    hh, hw = high_mask.shape
    lh, lw = low_mask.shape

    dr_lo = lh + 1
    dr_hi = lh * lo_scale - 1
    dc_lo = lw + 1
    dc_hi = lw * lo_scale - 1
    if dr_lo > dr_hi or dc_lo > dc_hi:
        return None

    for _ in range(80):
        dr = int(rng.integers(dr_lo, dr_hi + 1))
        dc = int(rng.integers(dc_lo, dc_hi + 1))
        max_br = min(h - lh * lo_scale, h - dr - hh * hi_scale)
        max_bc = min(w - lw * lo_scale, w - dc - hw * hi_scale)
        if max_br < 0 or max_bc < 2:
            continue
        br = int(rng.integers(0, max_br + 1))
        bc = int(rng.integers(2, max_bc + 1))
        ar = br + dr
        ac = bc + dc
        low = {
            "mask": low_mask,
            "anchor": (br, bc),
            "input_rect": _input_rect((br, bc), low_mask.shape),
            "scaled_rect": _scaled_rect((br, bc), low_mask.shape, lo_scale),
            "has_hole": False,
        }
        high = {
            "mask": high_mask,
            "anchor": (ar, ac),
            "input_rect": _input_rect((ar, ac), high_mask.shape),
            "scaled_rect": _scaled_rect((ar, ac), high_mask.shape, hi_scale),
            "has_hole": False,
        }
        if _rect_intersection(low["scaled_rect"], high["scaled_rect"]) is None:
            continue
        if not _rect_gap(_rect_expand(low["input_rect"], MOAT, (h, w)), high["input_rect"]):
            continue
        return high, low
    return None


def _random_anchor(
    rng: np.random.Generator,
    h: int,
    w: int,
    shape: tuple[int, int],
    scale: int,
    occupied_input: list[tuple[int, int, int, int]],
    occupied_scaled: list[tuple[int, int, int, int]],
    protect_rect: tuple[int, int, int, int] | None,
) -> tuple[int, int] | None:
    sh, sw = shape
    max_r = h - sh * scale
    max_c = w - sw * scale
    if max_r < 0 or max_c < 2:
        return None
    for _ in range(250):
        r0 = int(rng.integers(0, max_r + 1))
        c0 = int(rng.integers(2, max_c + 1))
        input_rect = _input_rect((r0, c0), shape)
        scaled_rect = _scaled_rect((r0, c0), shape, scale)
        if not _can_place_input_rect(input_rect, occupied_input, (h, w)):
            continue
        if any(not _rect_gap(_rect_expand(scaled_rect, 1, (h, w)), other) for other in occupied_scaled):
            continue
        if protect_rect is not None and not _rect_gap(_rect_expand(scaled_rect, 1, (h, w)), protect_rect):
            continue
        return (r0, c0)
    return None


def _scene_description(grid: np.ndarray) -> tuple[int, list[int], list[dict]]:
    bg = get_bg(grid)
    legend = _parse_legend(grid, bg)
    canvas = np.asarray(grid, dtype=int).copy()
    canvas[:, 0] = int(bg)
    comps = _find_components(canvas, bg=bg, colors=legend)
    scale_by_color = {color: idx + 1 for idx, color in enumerate(legend)}
    objects: list[dict] = []
    for comp in comps:
        color = int(comp["color"])
        sprite, anchor = _object_sprite(canvas, color, np.asarray(comp["coords"], dtype=int), bg)
        mask = sprite == color
        scale = scale_by_color[color]
        objects.append(
            {
                "color": color,
                "scale": scale,
                "priority": legend.index(color),
                "anchor": anchor,
                "input_rect": _input_rect(anchor, sprite.shape),
                "scaled_rect": _scaled_rect(anchor, sprite.shape, scale),
                "scaled_patch": _scaled_patch(mask, color, scale, bg),
                "has_hole": bool((sprite == bg).any()),
            }
        )
    return bg, legend, objects


def _validate_scene(inp: np.ndarray, out: np.ndarray) -> bool:
    bg, legend, objects = _scene_description(inp)
    if len(legend) < 2:
        return False
    if any(inp[r, 0] != legend[r] for r in range(len(legend))):
        return False
    if np.any(inp[len(legend) :, 0] != bg):
        return False
    counts = Counter(obj["color"] for obj in objects)
    if any(counts[color] == 0 for color in legend):
        return False
    if max(counts.values()) < 2:
        return False
    if not any(obj["has_hole"] for obj in objects):
        return False
    if np.array_equal(out, np.where(np.arange(inp.shape[1]) == 0, bg, inp)):
        return False

    top_visible = False
    lower_visible = False
    overlap_found = False
    for i, a in enumerate(objects):
        a_rect = a["scaled_rect"]
        a_patch = a["scaled_patch"]
        for j, b in enumerate(objects):
            if i == j or a["priority"] >= b["priority"]:
                continue
            inter = _rect_intersection(a_rect, b["scaled_rect"])
            if inter is None:
                continue
            overlap_found = True
            r0, c0, r1, c1 = inter
            a_view = a_patch[r0 - a_rect[0] : r1 - a_rect[0] + 1, c0 - a_rect[1] : c1 - a_rect[1] + 1]
            b_view = b["scaled_patch"][
                r0 - b["scaled_rect"][0] : r1 - b["scaled_rect"][0] + 1,
                c0 - b["scaled_rect"][1] : c1 - b["scaled_rect"][1] + 1,
            ]
            both = (a_view != bg) & (b_view != bg)
            if both.any() and np.any(out[r0 : r1 + 1, c0 : c1 + 1][both] == a["color"]):
                top_visible = True
            br0, bc0, br1, bc1 = b["scaled_rect"]
            b_only = out[br0 : br1 + 1, bc0 : bc1 + 1] == b["color"]
            if b_only.any():
                lower_visible = True
    if not overlap_found or not top_visible or not lower_visible:
        return False

    for obj in objects:
        sr0, sc0, sr1, sc1 = obj["scaled_rect"]
        if obj["has_hole"] and np.any(out[sr0 : sr1 + 1, sc0 : sc1 + 1] == bg):
            break
    else:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    bg = int(rng.integers(0, 10))
    n_colors = int(rng.integers(COLOR_COUNT_RANGE[0], COLOR_COUNT_RANGE[1] + 1))
    colors = [int(c) for c in rng.choice([c for c in range(10) if c != bg], size=n_colors, replace=False)]

    forced = _place_forced_overlap(rng, h, w, hi_scale=1, lo_scale=n_colors, bg=bg)
    if forced is None:
        return None
    high_obj, low_obj = forced
    objects_by_color: dict[int, list[dict]] = {color: [] for color in colors}
    objects_by_color[colors[0]].append(high_obj)
    objects_by_color[colors[-1]].append(low_obj)

    extra_color_index = int(rng.integers(0, n_colors))
    hole_color_index = int(rng.integers(0, n_colors))
    occupied_input = [high_obj["input_rect"], low_obj["input_rect"]]
    occupied_scaled = [high_obj["scaled_rect"], low_obj["scaled_rect"]]
    protect_rect = _rect_expand(
        (
            min(high_obj["scaled_rect"][0], low_obj["scaled_rect"][0]),
            min(high_obj["scaled_rect"][1], low_obj["scaled_rect"][1]),
            max(high_obj["scaled_rect"][2], low_obj["scaled_rect"][2]),
            max(high_obj["scaled_rect"][3], low_obj["scaled_rect"][3]),
        ),
        1,
        (h, w),
    )

    for idx, color in enumerate(colors):
        need = 1
        if idx == 0 or idx == n_colors - 1:
            need = 0
        if idx == extra_color_index:
            need += EXTRA_OBJECTS
        for copy_index in range(need):
            scale = idx + 1
            want_hole = idx == hole_color_index and copy_index == 0 and not any(
                obj["has_hole"] for objs in objects_by_color.values() for obj in objs
            )
            mask = _make_shape(rng, scale, want_hole=want_hole)
            anchor = _random_anchor(
                rng,
                h,
                w,
                mask.shape,
                scale,
                occupied_input,
                occupied_scaled,
                protect_rect,
            )
            if anchor is None:
                return None
            obj = {
                "mask": mask,
                "anchor": anchor,
                "input_rect": _input_rect(anchor, mask.shape),
                "scaled_rect": _scaled_rect(anchor, mask.shape, scale),
                "has_hole": bool((~mask).any()),
            }
            objects_by_color[color].append(obj)
            occupied_input.append(obj["input_rect"])
            occupied_scaled.append(obj["scaled_rect"])

    if not any(obj["has_hole"] for objs in objects_by_color.values() for obj in objs):
        return None

    grid = np.full((h, w), bg, dtype=int)
    for row, color in enumerate(colors):
        grid[row, 0] = int(color)
    for color, objs in objects_by_color.items():
        for obj in objs:
            _place_mask(grid, obj["mask"], obj["anchor"], color)

    try:
        out = main(grid)
    except Exception:
        return None
    if not _validate_scene(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid legend-ordered scaling example")
