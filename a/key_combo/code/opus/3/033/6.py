import random
from collections import deque

import numpy as np


MAX_GEN_ATTEMPTS = 400
BACKGROUND = 0
PATCH_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    out: list[dict] = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            out.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    out.sort(key=lambda d: tuple(d["bbox"]))
    return out


def _contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx0 in indices[1:]:
        idx = int(idx0)
        if idx == prev + 1:
            prev = idx
            continue
        runs.append((start, prev))
        start = prev = idx
    runs.append((start, prev))
    return runs


def _bottom_palette(grid: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert h >= 2, "grid must have room for a bottom palette"
    palette_rows = grid[h - 2 : h, :]
    runs = _contiguous_runs(np.flatnonzero(np.any(palette_rows != bg, axis=0)))
    best: tuple[int, int] | None = None
    for c0, c1 in runs:
        border = grid[h - 1, c0 : c1 + 1]
        fill = grid[h - 2, c0 : c1 + 1]
        if border.size < 2:
            continue
        if np.any(border == bg) or np.any(fill == bg):
            continue
        keys = [int(x) for x in border.tolist()]
        if len(keys) != len(set(keys)):
            continue
        if best is None or (c1 - c0) > (best[1] - best[0]):
            best = (c0, c1)
    assert best is not None, "bottom palette strip not found"
    c0, c1 = best
    mapping = {
        int(border): int(fill)
        for border, fill in zip(grid[h - 1, c0 : c1 + 1], grid[h - 2, c0 : c1 + 1], strict=False)
    }
    mask = np.zeros((h, w), dtype=bool)
    mask[h - 2 : h, c0 : c1 + 1] = True
    return mapping, mask


def _enclosed_holes(sprite: np.ndarray, bg: int) -> list[np.ndarray]:
    holes: list[np.ndarray] = []
    h, w = sprite.shape
    for coords in _components_mask(sprite == bg):
        r0, c0, r1, c1 = _bbox_coords(coords)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        holes.append(coords)
    return holes


def _fill_holes(sprite: np.ndarray, bg: int, fill_color: int) -> tuple[np.ndarray, list[np.ndarray]]:
    filled = sprite.copy()
    holes = _enclosed_holes(sprite, bg)
    for pts in holes:
        filled[pts[:, 0], pts[:, 1]] = int(fill_color)
    return filled, holes


def _symmetry_axes(sprite: np.ndarray) -> tuple[str, ...]:
    axes: list[str] = []
    if np.array_equal(sprite, np.fliplr(sprite)):
        axes.append("vertical")
    if np.array_equal(sprite, np.flipud(sprite)):
        axes.append("horizontal")
    return tuple(axes)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow sprite using the bottom palette, then crop the unique filled sprite that becomes color-symmetric.

    Input structure:
    - A black background contains several separated monochromatic sprites above a bottom 2xN palette strip.
    - The palette's lower row lists sprite border colors and the row above lists the fill color paired with each border.
    - Every sprite contains one or more enclosed black holes inside its own bounding box.

    Rule:
    - Read the bottom palette as a border-color -> fill-color mapping.
    - Remove the palette from consideration, crop each sprite, and fill every enclosed black component with that sprite's mapped fill color.
    - Return the unique filled sprite whose cropped color pattern is exactly symmetric under a vertical or horizontal reflection.

    Key insight:
    - Outline shape alone is insufficient: only after palette-guided hole filling does one sprite's full color pattern become symmetric.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = BACKGROUND
    mapping, palette_mask = _bottom_palette(g, bg)

    work = g.copy()
    work[palette_mask] = bg

    winners: list[np.ndarray] = []
    for comp in _find_components(work, bg):
        color = int(comp["color"])
        if color not in mapping:
            continue
        sprite = _crop_bbox(work, comp["bbox"])
        filled, holes = _fill_holes(sprite, bg, mapping[color])
        if not holes:
            continue
        if _symmetry_axes(filled):
            winners.append(filled)

    assert len(winners) == 1, "expected exactly one filled symmetric sprite"
    return winners[0]


def _safe_hole_cells(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    safe = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if (
                mask[r, c]
                and mask[r - 1, c]
                and mask[r + 1, c]
                and mask[r, c - 1]
                and mask[r, c + 1]
            ):
                safe[r, c] = True
    return safe


def _mirror_point(r: int, c: int, axis: str, h: int, w: int) -> tuple[int, int]:
    if axis == "vertical":
        return r, w - 1 - c
    if axis == "horizontal":
        return h - 1 - r, c
    raise ValueError(f"unknown axis: {axis}")


def _outer_mask(h: int, w: int, style: str, rng: random.Random) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)

    if style == "rect":
        pass
    elif style == "chamfer":
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for r, c in corners:
            mask[r, c] = False
        if min(h, w) >= 7 and rng.random() < 0.5:
            extras = [
                (0, 1),
                (1, 0),
                (0, w - 2),
                (1, w - 1),
                (h - 2, 0),
                (h - 1, 1),
                (h - 2, w - 1),
                (h - 1, w - 2),
            ]
            for r, c in extras:
                mask[r, c] = False
    elif style == "v_notch":
        side = rng.choice(["top", "bottom"])
        depth = 1 if h <= 5 else rng.choice([1, 2])
        candidates = [c for c in range(1, w - 1) if w - 1 - c != c]
        if not candidates:
            return None
        c = rng.choice(candidates)
        rows = range(depth) if side == "top" else range(h - depth, h)
        for r in rows:
            mask[r, c] = False
            mask[r, w - 1 - c] = False
    elif style == "h_notch":
        side = rng.choice(["left", "right"])
        depth = 1 if w <= 5 else rng.choice([1, 2])
        candidates = [r for r in range(1, h - 1) if h - 1 - r != r]
        if not candidates:
            return None
        r = rng.choice(candidates)
        cols = range(depth) if side == "left" else range(w - depth, w)
        for c in cols:
            mask[r, c] = False
            mask[h - 1 - r, c] = False
    else:
        raise ValueError(f"unknown style: {style}")

    if len(_components_mask(mask)) != 1:
        return None
    if int(_safe_hole_cells(mask).sum()) < 2:
        return None
    return mask


def _hole_components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    return [
        coords
        for coords in _components_mask(~mask)
        if 0 < coords[:, 0].min()
        and 0 < coords[:, 1].min()
        and coords[:, 0].max() < mask.shape[0] - 1
        and coords[:, 1].max() < mask.shape[1] - 1
    ]


def _render_sprite(mask: np.ndarray, border_color: int, fill_color: int | None) -> np.ndarray:
    sprite = np.full(mask.shape, BACKGROUND, dtype=int)
    sprite[mask] = int(border_color)
    if fill_color is not None:
        for pts in _hole_components_from_mask(mask):
            sprite[pts[:, 0], pts[:, 1]] = int(fill_color)
    return sprite


def _enumerate_patch_sets(safe: np.ndarray, blocked: np.ndarray, axis: str | None = None) -> list[tuple[tuple[int, int], ...]]:
    h, w = safe.shape
    signatures: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    for template in PATCH_TEMPLATES:
        max_r = max(r for r, _ in template)
        max_c = max(c for _, c in template)
        for r0 in range(h - max_r):
            for c0 in range(w - max_c):
                base = tuple((r0 + dr, c0 + dc) for dr, dc in template)
                if not all(safe[r, c] and not blocked[r, c] for r, c in base):
                    continue
                cells = set(base)
                if axis is not None:
                    mirrored = {_mirror_point(r, c, axis, h, w) for r, c in base}
                    cells |= mirrored
                    if not any(_mirror_point(r, c, axis, h, w) != (r, c) for r, c in cells):
                        continue
                    if not all(safe[r, c] and not blocked[r, c] for r, c in cells):
                        continue
                sig = tuple(sorted(cells))
                signatures[sig] = sig
    return list(signatures.values())


def _carve_mask(mask: np.ndarray, hole_cells: set[tuple[int, int]]) -> np.ndarray:
    carved = mask.copy()
    for r, c in hole_cells:
        carved[r, c] = False
    return carved


def _sample_hollow_mask(
    outer: np.ndarray,
    axis: str | None,
    border_color: int,
    fill_color: int,
    rng: random.Random,
) -> np.ndarray | None:
    safe = _safe_hole_cells(outer)
    for _ in range(250):
        blocked = np.zeros_like(safe, dtype=bool)
        hole_cells: set[tuple[int, int]] = set()
        patch_count = rng.randint(1, 2) if axis is not None else rng.randint(1, 3)
        for _patch_idx in range(patch_count):
            candidates = _enumerate_patch_sets(safe, blocked, axis=axis)
            if not candidates:
                break
            picked = rng.choice(candidates)
            for r, c in picked:
                blocked[r, c] = True
                hole_cells.add((r, c))
        if not hole_cells:
            continue

        carved = _carve_mask(outer, hole_cells)
        if len(_components_mask(carved)) != 1:
            continue

        holes = _hole_components_from_mask(carved)
        if not holes:
            continue

        filled = _render_sprite(carved, border_color, fill_color)
        axes = set(_symmetry_axes(filled))
        if axis is None:
            if axes:
                continue
        elif axes != {axis}:
            continue
        return carved
    return None


def _sample_object(
    rng: random.Random,
    border_color: int,
    fill_color: int,
    desired_axis: str | None,
    used_signatures: set[tuple],
) -> tuple[np.ndarray, np.ndarray] | None:
    if desired_axis == "vertical":
        styles = ["rect", "chamfer", "v_notch"]
    elif desired_axis == "horizontal":
        styles = ["rect", "chamfer", "h_notch"]
    else:
        styles = ["rect", "chamfer", "v_notch", "h_notch"]

    for _ in range(200):
        h = rng.randint(5, 7)
        w = rng.randint(5, 7)
        style = rng.choice(styles)
        outer = _outer_mask(h, w, style, rng)
        if outer is None:
            continue
        carved = _sample_hollow_mask(outer, desired_axis, border_color, fill_color, rng)
        if carved is None:
            continue
        unfilled = _render_sprite(carved, border_color, None)
        filled = _render_sprite(carved, border_color, fill_color)
        signature = (unfilled.shape, tuple(unfilled.ravel().tolist()))
        if signature in used_signatures:
            continue
        used_signatures.add(signature)
        return unfilled, filled
    return None


def _place_objects(grid: np.ndarray, patches: list[np.ndarray], rng: random.Random) -> bool:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    occ[h - 3 :, :] = True
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(0, h - 3 - ph + 1):
            for c0 in range(0, w - pw + 1):
                r1 = r0 + ph
                c1 = c0 + pw
                rr0 = max(0, r0 - 1)
                rr1 = min(h, r1 + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(w, c1 + 1)
                if occ[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = rng.choice(candidates)
        mask = patch != BACKGROUND
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True
    return True


def _build_palette_row_pairs(border_colors: list[int], fill_colors: list[int], width: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    perm = list(range(len(border_colors)))
    rng.shuffle(perm)
    top = np.full(width, BACKGROUND, dtype=int)
    bottom = np.full(width, BACKGROUND, dtype=int)
    start = rng.randint(0, width - len(perm))
    for offset, idx in enumerate(perm):
        top[start + offset] = int(fill_colors[idx])
        bottom[start + offset] = int(border_colors[idx])
    return top, bottom


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    num_objects = rng.choice([3, 4])
    winner_index = rng.randrange(num_objects)
    winner_axis = rng.choice(["vertical", "horizontal"])

    colors = list(range(1, 10))
    rng.shuffle(colors)
    palette_colors = colors[: 2 * num_objects]
    border_colors = palette_colors[:num_objects]
    fill_colors = palette_colors[num_objects:]

    used_signatures: set[tuple] = set()
    input_patches: list[np.ndarray] = []
    output_patches: list[np.ndarray] = []

    for idx in range(num_objects):
        desired_axis = winner_axis if idx == winner_index else None
        obj = _sample_object(rng, border_colors[idx], fill_colors[idx], desired_axis, used_signatures)
        if obj is None:
            return None
        unfilled, filled = obj
        input_patches.append(unfilled)
        output_patches.append(filled)

    h = rng.randint(16, 22)
    w = rng.randint(max(16, num_objects + 8), 24)
    grid = np.full((h, w), BACKGROUND, dtype=int)
    if not _place_objects(grid, input_patches, rng):
        return None

    palette_top, palette_bottom = _build_palette_row_pairs(border_colors, fill_colors, w, rng)
    grid[h - 2, :] = palette_top
    grid[h - 1, :] = palette_bottom

    try:
        output = main(grid)
    except Exception:
        return None
    expected = output_patches[winner_index]
    if not np.array_equal(output, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
