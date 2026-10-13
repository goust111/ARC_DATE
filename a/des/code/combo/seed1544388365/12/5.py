from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG_COLOR = 0
YELLOW = 4
NEIGHBORS4: tuple[tuple[int, int], ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _coord_set_from_mask(mask: np.ndarray, *, origin: tuple[int, int] = (0, 0)) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(mask)
    }


def _non_bg_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid != BG_COLOR)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == coords


def _find_components_by_color(grid: np.ndarray) -> dict[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    by_color: dict[int, list[dict[str, object]]] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG_COLOR or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: set[tuple[int, int]] = set()
            while stack:
                rr, cc = stack.pop()
                coords.add((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            by_color.setdefault(color, []).append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "bbox": _bbox(coords),
                }
            )

    return by_color


def _alignment_candidates(
    grid: np.ndarray,
    reference_coords: set[tuple[int, int]],
    visible_coords: set[tuple[int, int]],
    color: int,
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    diffs = {
        (vr - rr, vc - rc)
        for vr, vc in visible_coords
        for rr, rc in reference_coords
    }

    candidates: list[dict[str, object]] = []
    for dr, dc in diffs:
        shifted = {(r + dr, c + dc) for r, c in reference_coords}
        if any(not (0 <= r < h and 0 <= c < w) for r, c in shifted):
            continue
        if not visible_coords <= shifted:
            continue

        hidden = shifted - visible_coords
        if not hidden:
            continue

        cover_colors = {int(g[r, c]) for r, c in hidden}
        if len(cover_colors) != 1:
            continue
        cover_color = next(iter(cover_colors))
        if cover_color in (BG_COLOR, color):
            continue

        candidates.append(
            {
                "shift": (dr, dc),
                "hidden": hidden,
                "cover_color": cover_color,
            }
        )

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover hidden pixels of the smaller object in each same-color pair, mark those hidden
    cells yellow, and delete the larger reference copy.

    Input structure:
    - Black background with 4-connected monochrome foreground objects.
    - Some colors appear once and stay unchanged.
    - Some colors appear exactly twice: a full reference object and a partially occluded copy.

    Rule:
    - For each duplicated color, treat the larger component as the reference and the smaller one
      as the occluded copy.
    - Translate the reference shape so that it fully contains the visible smaller component.
    - Cells present in the translated reference but missing from the visible copy are hidden by
      a single differently colored occluder; recolor those cells yellow.
    - Remove the full reference component from the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components_by_color = _find_components_by_color(g)
    out = g.copy()

    for color, components in components_by_color.items():
        if len(components) == 1:
            continue

        assert len(components) == 2, f"expected one full/partial pair for color {color}"
        ordered = sorted(components, key=lambda comp: int(comp["area"]), reverse=True)
        reference = ordered[0]
        visible = ordered[1]
        assert int(reference["area"]) > int(visible["area"])

        candidates = _alignment_candidates(
            g,
            reference["coords"],  # type: ignore[arg-type]
            visible["coords"],  # type: ignore[arg-type]
            color,
        )
        assert len(candidates) == 1, f"expected a unique alignment for color {color}"
        hidden = candidates[0]["hidden"]

        for r, c in reference["coords"]:  # type: ignore[misc]
            out[r, c] = BG_COLOR
        for r, c in hidden:  # type: ignore[misc]
            out[r, c] = YELLOW

    return out


GEN_COLORS: tuple[int, ...] = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_GRID_MIN = 14
GEN_GRID_MAX = 28
GEN_PAIR_COUNT_OPTIONS = (1, 2)
GEN_DISTRACTOR_COUNT_OPTIONS = (0, 1, 2)
GEN_MAX_ATTEMPTS = 500
GEN_PATCH_ATTEMPTS = 150


def _random_connected_cells(max_h: int, max_w: int, min_area: int, max_area: int) -> set[tuple[int, int]] | None:
    base_h = random.randint(1, max_h)
    base_w = random.randint(1, max_w)
    max_area = min(max_area, base_h * base_w)
    if min_area > max_area:
        return None

    target = random.randint(min_area, max_area)
    start = (random.randrange(base_h), random.randrange(base_w))
    cells = {start}

    while len(cells) < target:
        frontier = {
            (r + dr, c + dc)
            for r, c in cells
            for dr, dc in NEIGHBORS4
            if 0 <= r + dr < base_h and 0 <= c + dc < base_w and (r + dr, c + dc) not in cells
        }
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    r0, c0, _r1, _c1 = _bbox(cells)
    return {(r - r0, c - c0) for r, c in cells}


def _random_shape_mask() -> np.ndarray | None:
    for _ in range(80):
        cells = _random_connected_cells(max_h=6, max_w=6, min_area=5, max_area=12)
        if cells is None:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _random_hidden_subset(shape_cells: set[tuple[int, int]]) -> set[tuple[int, int]] | None:
    if len(shape_cells) < 4:
        return None

    max_r = max(r for r, _ in shape_cells)
    max_c = max(c for _, c in shape_cells)
    boundary = [
        cell
        for cell in shape_cells
        if cell[0] in (0, max_r) or cell[1] in (0, max_c)
    ]
    max_hidden = min(5, len(shape_cells) - 3)
    if max_hidden < 1:
        return None

    for _ in range(100):
        target = random.randint(1, max_hidden)
        hidden = {random.choice(boundary)}
        while len(hidden) < target:
            frontier = {
                (r + dr, c + dc)
                for r, c in hidden
                for dr, dc in NEIGHBORS4
                if (r + dr, c + dc) in shape_cells and (r + dr, c + dc) not in hidden
            }
            if not frontier:
                break
            hidden.add(random.choice(tuple(frontier)))

        if len(hidden) != target:
            continue

        visible = shape_cells - hidden
        if len(visible) < 3 or not _is_connected(visible):
            continue
        return hidden

    return None


def _grow_occluder(
    hidden_abs: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    canvas_h: int,
    canvas_w: int,
) -> set[tuple[int, int]]:
    occluder = set(hidden_abs)
    extra_target = random.randint(0, 4)

    for _ in range(extra_target):
        frontier = {
            (r + dr, c + dc)
            for r, c in occluder
            for dr, dc in NEIGHBORS4
            if 0 <= r + dr < canvas_h
            and 0 <= c + dc < canvas_w
            and (r + dr, c + dc) not in occluder
            and (r + dr, c + dc) not in forbidden
        }
        if not frontier:
            break
        occluder.add(random.choice(tuple(frontier)))

    return occluder


def _build_pair_patch(pair_color: int, occluder_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(GEN_PATCH_ATTEMPTS):
        shape_mask = _random_shape_mask()
        if shape_mask is None:
            continue

        shape_cells = _coord_set_from_mask(shape_mask)
        hidden_rel = _random_hidden_subset(shape_cells)
        if hidden_rel is None:
            continue
        visible_rel = shape_cells - hidden_rel

        h, w = shape_mask.shape
        margin = 3
        gap = random.randint(3, 5)
        slack = 2
        layout = random.choice(("LR", "RL", "TB", "BT"))

        if layout in ("LR", "RL"):
            canvas_h = h + 2 * margin + slack
            canvas_w = 2 * w + gap + 2 * margin
            ref_r = random.randint(margin - 1, margin + 1)
            vis_r = random.randint(margin - 1, margin + 1)
            left_c = margin
            right_c = margin + w + gap
            if layout == "LR":
                ref_c, vis_c = left_c, right_c
            else:
                ref_c, vis_c = right_c, left_c
        else:
            canvas_h = 2 * h + gap + 2 * margin
            canvas_w = w + 2 * margin + slack
            ref_c = random.randint(margin - 1, margin + 1)
            vis_c = random.randint(margin - 1, margin + 1)
            top_r = margin
            bottom_r = margin + h + gap
            if layout == "TB":
                ref_r, vis_r = top_r, bottom_r
            else:
                ref_r, vis_r = bottom_r, top_r

        reference_abs = _coord_set_from_mask(shape_mask, origin=(ref_r, ref_c))
        full_abs = _coord_set_from_mask(shape_mask, origin=(vis_r, vis_c))
        visible_abs = {(vis_r + r, vis_c + c) for r, c in visible_rel}
        hidden_abs = full_abs - visible_abs

        occluder_abs = _grow_occluder(
            hidden_abs,
            forbidden=reference_abs | full_abs,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )
        if occluder_abs & reference_abs:
            continue
        if occluder_abs & visible_abs:
            continue

        inp = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)
        out = np.full((canvas_h, canvas_w), BG_COLOR, dtype=int)

        for r, c in reference_abs:
            inp[r, c] = pair_color
        for r, c in visible_abs:
            inp[r, c] = pair_color
        for r, c in occluder_abs:
            inp[r, c] = occluder_color

        out[:] = inp
        for r, c in reference_abs:
            out[r, c] = BG_COLOR
        for r, c in hidden_abs:
            out[r, c] = YELLOW

        try:
            if not np.array_equal(main(inp), out):
                continue
        except Exception:
            continue

        r0, c0, r1, c1 = _non_bg_bbox(inp)
        cropped_inp = inp[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        cropped_out = out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()
        return cropped_inp, cropped_out

    return None


def _build_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(60):
        cells = _random_connected_cells(max_h=4, max_w=4, min_area=1, max_area=6)
        if cells is None:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        patch = np.full((h, w), BG_COLOR, dtype=int)
        for r, c in cells:
            patch[r, c] = color
        return patch, patch.copy()
    return None


def _rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _paste_non_bg(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.nonzero(patch != BG_COLOR)
    for r, c in zip(rows, cols, strict=False):
        canvas[top + int(r), left + int(c)] = int(patch[int(r), int(c)])


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    reserved: list[tuple[int, int, int, int]],
) -> bool:
    h, w = inp.shape
    ph, pw = patch_in.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)

    for top, left in positions:
        bbox = (top, left, top + ph - 1, left + pw - 1)
        if any(_rects_intersect(bbox, other, pad=1) for other in reserved):
            continue
        _paste_non_bg(inp, patch_in, top, left)
        _paste_non_bg(out, patch_out, top, left)
        reserved.append(bbox)
        return True

    return False


def _generate_scene() -> tuple[np.ndarray, np.ndarray] | None:
    pair_count = random.choice(GEN_PAIR_COUNT_OPTIONS)
    distractor_count = random.choice(GEN_DISTRACTOR_COUNT_OPTIONS)
    needed_colors = pair_count * 2 + distractor_count
    palette = random.sample(GEN_COLORS, needed_colors)

    pair_colors = palette[:pair_count]
    occluder_colors = palette[pair_count : 2 * pair_count]
    distractor_colors = palette[2 * pair_count :]

    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for pair_color, occluder_color in zip(pair_colors, occluder_colors, strict=False):
        patch = _build_pair_patch(pair_color, occluder_color)
        if patch is None:
            return None
        patches.append(patch)

    for color in distractor_colors:
        patch = _build_distractor_patch(color)
        if patch is None:
            return None
        patches.append(patch)

    random.shuffle(patches)
    patch_heights = [patch_in.shape[0] for patch_in, _patch_out in patches]
    patch_widths = [patch_in.shape[1] for patch_in, _patch_out in patches]
    min_area = sum((h + 2) * (w + 2) for h, w in zip(patch_heights, patch_widths, strict=False))

    for _ in range(80):
        grid_h = random.randint(max(GEN_GRID_MIN, max(patch_heights)), GEN_GRID_MAX)
        grid_w = random.randint(max(GEN_GRID_MIN, max(patch_widths)), GEN_GRID_MAX)
        if grid_h * grid_w < min_area:
            continue

        inp = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
        out = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
        reserved: list[tuple[int, int, int, int]] = []

        ok = True
        for patch_in, patch_out in patches:
            if not _place_patch(inp, out, patch_in, patch_out, reserved):
                ok = False
                break
        if not ok:
            continue

        try:
            if not np.array_equal(main(inp), out):
                continue
        except Exception:
            continue
        return inp, out

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        scene = _generate_scene()
        if scene is None:
            continue
        inp, _out = scene
        return inp
    raise RuntimeError("failed to generate valid occlusion-pair ARC input")
