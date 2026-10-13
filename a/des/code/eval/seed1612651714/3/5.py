from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= dst.shape[0] and left + w <= dst.shape[1]
    mask = patch != bg
    view = dst[top : top + h, left : left + w]
    assert np.all(view[mask] == bg), "transformed objects must not overlap"
    view[mask] = patch[mask]


def scale_patch(patch: np.ndarray, factor: int, bg: int) -> np.ndarray:
    mask = (patch != bg).astype(int)
    scaled_mask = np.kron(mask, np.ones((factor, factor), dtype=int)).astype(bool)
    color = int(patch[patch != bg][0])
    out = np.full(scaled_mask.shape, bg, dtype=int)
    out[scaled_mask] = color
    return out


def rotate_for_direction(patch: np.ndarray, direction: str) -> np.ndarray:
    want_vertical = direction in ("left", "right")
    for k in range(4):
        rot = np.rot90(patch, k)
        if want_vertical and rot.shape[0] >= rot.shape[1]:
            return np.asarray(rot, dtype=int)
        if (not want_vertical) and rot.shape[1] >= rot.shape[0]:
            return np.asarray(rot, dtype=int)
    raise AssertionError("a rotation should always satisfy the orientation constraint")


def extract_band(grid: np.ndarray, bg: int) -> tuple[dict[int, int], list[int]]:
    cols = [c for c in range(grid.shape[1]) if int(grid[-1, c]) != bg]
    assert cols, "bottom color band is missing"
    colors = [int(grid[-1, c]) for c in cols]
    assert len(set(colors)) == len(colors), "band colors must be unique"
    rank = {color: idx + 1 for idx, color in enumerate(colors)}
    return rank, cols


def kernel_patch(component_patch: np.ndarray, kernel_color: int, bg: int) -> np.ndarray:
    coords = np.argwhere(component_patch == kernel_color)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    local = coords - np.array([r0, c0], dtype=int)
    sub[local[:, 0], local[:, 1]] = kernel_color
    return sub


def shell_and_kernel_colors(component_patch: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(x) for x in np.unique(component_patch) if int(x) != bg]
    assert len(colors) == 2, "each structure must use exactly two non-background colors"
    h, w = component_patch.shape
    border = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                v = int(component_patch[r, c])
                if v != bg:
                    border.append(v)
    border_counts = Counter(border)
    area_counts = {color: int(np.count_nonzero(component_patch == color)) for color in colors}
    shell = max(colors, key=lambda color: (border_counts[color], area_counts[color]))
    kernel = min(colors, key=lambda color: (border_counts[color], area_counts[color]))
    return int(shell), int(kernel)


def notch_direction(component_patch: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = component_patch.shape
    hits = {}
    if np.any(component_patch[0, :] == bg):
        cols = np.flatnonzero(component_patch[0, :] == bg)
        assert np.all(cols == np.arange(cols[0], cols[-1] + 1))
        hits["up"] = int(cols[0])
    if np.any(component_patch[h - 1, :] == bg):
        cols = np.flatnonzero(component_patch[h - 1, :] == bg)
        assert np.all(cols == np.arange(cols[0], cols[-1] + 1))
        hits["down"] = int(cols[0])
    if np.any(component_patch[:, 0] == bg):
        rows = np.flatnonzero(component_patch[:, 0] == bg)
        assert np.all(rows == np.arange(rows[0], rows[-1] + 1))
        hits["left"] = int(rows[0])
    if np.any(component_patch[:, w - 1] == bg):
        rows = np.flatnonzero(component_patch[:, w - 1] == bg)
        assert np.all(rows == np.arange(rows[0], rows[-1] + 1))
        hits["right"] = int(rows[0])
    assert len(hits) == 1, "concave shell must have a single side notch"
    direction, start = next(iter(hits.items()))
    return direction, int(start)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each shell-wrapped kernel structure, then either enlarge the kernel or shoot it out through the shell notch.

    Input structure:
    - The background is the dominant color.
    - Each non-band object is a two-color connected structure:
      shell color on the outside, kernel color inside.
    - Convex shells fill their bbox completely with shell+kernel.
    - Concave shells have one rectangular notch of background on exactly one bbox side.
    - The bottom row contains a color band; the left-to-right position of a kernel color
      gives the enlargement factor for convex structures.

    Rule:
    - Remove every shell from the output.
    - Convex shell: extract the kernel's own pattern, scale it by the matching band index,
      and paste it with its top-left aligned to the original structure bbox top-left.
    - Concave shell: rotate the kernel so its long bbox axis is perpendicular to the notch
      direction, then translate it straight out to the corresponding grid edge.
      Its perpendicular offset is inherited from the notch position.
    - Keep the bottom color band unchanged.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    band_rank, band_cols = extract_band(g, bg)

    work = g.copy()
    work[-1, band_cols] = bg
    components = components_mask(work != bg)
    assert components, "expected at least one shell structure"

    transforms: list[tuple[np.ndarray, int, int]] = []
    clear_coords: list[np.ndarray] = []

    for coords in sorted(components, key=lambda pts: tuple(pts.min(axis=0))):
        clear_coords.append(coords)
        bbox = bbox_coords(coords)
        r0, c0, _r1, _c1 = bbox
        patch = crop_bbox(work, bbox)
        _shell_color, kernel_color = shell_and_kernel_colors(patch, bg)
        kp = kernel_patch(patch, kernel_color, bg)

        if not np.any(patch == bg):
            factor = band_rank[kernel_color]
            scaled = scale_patch(kp, factor, bg)
            transforms.append((scaled, int(r0), int(c0)))
            continue

        direction, notch_start = notch_direction(patch, bg)
        rotated = rotate_for_direction(kp, direction)
        ph, pw = rotated.shape
        if direction == "left":
            top, left = int(r0 + notch_start), 0
        elif direction == "right":
            top, left = int(r0 + notch_start), int(w - pw)
        elif direction == "up":
            top, left = 0, int(c0 + notch_start)
        else:
            top, left = int(h - ph), int(c0 + notch_start)
        transforms.append((rotated, top, left))

    out = g.copy()
    for coords in clear_coords:
        out[coords[:, 0], coords[:, 1]] = bg
    for patch, top, left in transforms:
        paste_non_bg(out, patch, top, left, bg)
    return out


GEN_MAX_ATTEMPTS = 400
GEN_COLORS = tuple(range(10))
GEN_CONVEX_KERNEL_DIMS = ((2, 2), (2, 3), (3, 2))
GEN_CONCAVE_TARGET_DIMS = {
    "left": ((2, 1), (3, 1), (3, 2)),
    "right": ((2, 1), (3, 1), (3, 2)),
    "up": ((1, 2), (1, 3), (2, 3)),
    "down": ((1, 2), (1, 3), (2, 3)),
}


def dilated_bbox_free(mask: np.ndarray, r0: int, c0: int, hh: int, ww: int, pad: int = 1) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + hh + pad)
    cc1 = min(w, c0 + ww + pad)
    return not mask[rr0:rr1, cc0:cc1].any()


def mark_bbox(mask: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> None:
    mask[r0 : r0 + hh, c0 : c0 + ww] = True


def coords_conflict(mask: np.ndarray, coords: np.ndarray) -> bool:
    return bool(mask[coords[:, 0], coords[:, 1]].any())


def mark_coords(mask: np.ndarray, coords: np.ndarray) -> None:
    mask[coords[:, 0], coords[:, 1]] = True


def random_connected_mask(rng: random.Random, hh: int, ww: int) -> np.ndarray:
    target = rng.randint(2, hh * ww)
    cells = [(r, c) for r in range(hh) for c in range(ww)]
    start = rng.choice(cells)
    chosen = {start}
    frontier = {start}
    while len(chosen) < target:
        options = set()
        for r, c in frontier:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in chosen:
                    options.add((nr, nc))
        if not options:
            break
        nxt = rng.choice(tuple(options))
        chosen.add(nxt)
        frontier.add(nxt)
    coords = np.asarray(sorted(chosen), dtype=int)
    out = np.zeros((hh, ww), dtype=bool)
    out[coords[:, 0], coords[:, 1]] = True
    return out


def rect_touches_with_margin(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def build_convex_object(
    rng: random.Random,
    grid_shape: tuple[int, int],
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    shell_color: int,
    kernel_color: int,
    factor: int,
    bg: int,
) -> dict | None:
    h, w = grid_shape
    dims = [dim for dim in GEN_CONVEX_KERNEL_DIMS if dim[0] * factor <= 8 and dim[1] * factor <= 8]
    if not dims:
        return None
    kh, kw = rng.choice(dims)
    mask = random_connected_mask(rng, kh, kw)
    scaled_shape = (kh * factor, kw * factor)
    obj_h = rng.randint(max(kh + 2, scaled_shape[0]), max(kh + 2, scaled_shape[0]) + 2)
    obj_w = rng.randint(max(kw + 2, scaled_shape[1]), max(kw + 2, scaled_shape[1]) + 2)
    if obj_h >= h - 2 or obj_w >= w - 2:
        return None

    patch = np.full((obj_h, obj_w), shell_color, dtype=int)
    kr = rng.randint(1, obj_h - kh - 1)
    kc = rng.randint(1, obj_w - kw - 1)
    patch[kr : kr + kh, kc : kc + kw][mask] = kernel_color

    kernel = np.full((kh, kw), bg, dtype=int)
    kernel[mask] = kernel_color
    scaled = scale_patch(kernel, factor, bg)
    scaled_coords = np.argwhere(scaled != bg)

    for _ in range(80):
        r0 = rng.randint(1, h - obj_h - 2)
        c0 = rng.randint(1, w - obj_w - 1)
        if not dilated_bbox_free(input_occ, r0, c0, obj_h, obj_w, pad=1):
            continue
        out_coords = scaled_coords + np.array([r0, c0], dtype=int)
        if coords_conflict(output_occ, out_coords):
            continue
        return {
            "patch": patch,
            "bbox": (r0, c0, obj_h, obj_w),
            "output_coords": out_coords,
        }
    return None


def build_concave_object(
    rng: random.Random,
    grid_shape: tuple[int, int],
    input_occ: np.ndarray,
    output_occ: np.ndarray,
    shell_color: int,
    kernel_color: int,
    direction: str,
    bg: int,
) -> dict | None:
    h, w = grid_shape
    out_h, out_w = rng.choice(GEN_CONCAVE_TARGET_DIMS[direction])
    if rng.random() < 0.5:
        in_h, in_w = out_h, out_w
    else:
        in_h, in_w = out_w, out_h

    if direction in ("left", "right"):
        notch_h = out_h
        notch_w = rng.choice((1, 2))
        obj_h = rng.randint(max(notch_h + 2, in_h + 2), max(notch_h + 2, in_h + 2) + 3)
        obj_w = rng.randint(max(in_w + 2, notch_w + in_w + 2), max(in_w + 2, notch_w + in_w + 2) + 3)
        if obj_h >= h - 2 or obj_w >= w - 2:
            return None
        notch_r = rng.randint(1, obj_h - notch_h - 1)
        notch_c = 0 if direction == "left" else obj_w - notch_w
    else:
        notch_w = out_w
        notch_h = rng.choice((1, 2))
        obj_h = rng.randint(max(in_h + 2, notch_h + in_h + 2), max(in_h + 2, notch_h + in_h + 2) + 3)
        obj_w = rng.randint(max(notch_w + 2, in_w + 2), max(notch_w + 2, in_w + 2) + 3)
        if obj_h >= h - 2 or obj_w >= w - 2:
            return None
        notch_r = 0 if direction == "up" else obj_h - notch_h
        notch_c = rng.randint(1, obj_w - notch_w - 1)

    notch_rect = (notch_r, notch_c, notch_r + notch_h - 1, notch_c + notch_w - 1)
    kernel_positions = []
    for kr in range(1, obj_h - in_h):
        for kc in range(1, obj_w - in_w):
            ker_rect = (kr, kc, kr + in_h - 1, kc + in_w - 1)
            if rect_touches_with_margin(ker_rect, notch_rect, margin=1):
                continue
            kernel_positions.append((kr, kc))
    if not kernel_positions:
        return None

    kernel = np.full((in_h, in_w), kernel_color, dtype=int)
    rotated = rotate_for_direction(kernel, direction)
    rot_coords = np.argwhere(rotated != bg)

    for _ in range(80):
        kr, kc = rng.choice(kernel_positions)
        for _ in range(80):
            r0 = rng.randint(1, h - obj_h - 2)
            c0 = rng.randint(1, w - obj_w - 1)
            if not dilated_bbox_free(input_occ, r0, c0, obj_h, obj_w, pad=1):
                continue
            if direction == "left":
                top, left = r0 + notch_r, 0
            elif direction == "right":
                top, left = r0 + notch_r, w - out_w
            elif direction == "up":
                top, left = 0, c0 + notch_c
            else:
                top, left = h - out_h, c0 + notch_c
            out_coords = rot_coords + np.array([top, left], dtype=int)
            if np.any(out_coords[:, 0] >= h) or np.any(out_coords[:, 1] >= w):
                continue
            if coords_conflict(output_occ, out_coords):
                continue

            patch = np.full((obj_h, obj_w), shell_color, dtype=int)
            patch[notch_r : notch_r + notch_h, notch_c : notch_c + notch_w] = bg
            patch[kr : kr + in_h, kc : kc + in_w] = kernel_color
            return {
                "patch": patch,
                "bbox": (r0, c0, obj_h, obj_w),
                "output_coords": out_coords,
            }
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice((2, 4))
    h = rng.randint(18 if object_count == 4 else 16, 26)
    w = rng.randint(22 if object_count == 4 else 18, 28)
    bg = rng.choice(GEN_COLORS)
    palette = [color for color in GEN_COLORS if color != bg]
    rng.shuffle(palette)
    band_colors = palette[:4]
    shell_colors = palette[4 : 4 + object_count]
    if len(shell_colors) < object_count:
        return None

    band_order = band_colors[:]
    rng.shuffle(band_order)
    band_rank = {color: idx + 1 for idx, color in enumerate(band_order)}
    kernel_colors = rng.sample(band_order, object_count)
    structure_types = ["convex"] * (object_count // 2) + ["concave"] * (object_count // 2)
    rng.shuffle(structure_types)
    concave_dirs = rng.sample(("left", "right", "up", "down"), object_count // 2)

    grid = np.full((h, w), bg, dtype=int)
    band_start = rng.randint(0, w - len(band_order))
    for idx, color in enumerate(band_order):
        grid[h - 1, band_start + idx] = color

    input_occ = np.zeros((h, w), dtype=bool)
    output_occ = np.zeros((h, w), dtype=bool)
    output_occ[h - 1, band_start : band_start + len(band_order)] = True

    specs = []
    dir_idx = 0
    for idx in range(object_count):
        shell_color = shell_colors[idx]
        kernel_color = kernel_colors[idx]
        kind = structure_types[idx]
        spec = None
        for _ in range(120):
            if kind == "convex":
                spec = build_convex_object(
                    rng=rng,
                    grid_shape=(h, w),
                    input_occ=input_occ,
                    output_occ=output_occ,
                    shell_color=shell_color,
                    kernel_color=kernel_color,
                    factor=band_rank[kernel_color],
                    bg=bg,
                )
            else:
                spec = build_concave_object(
                    rng=rng,
                    grid_shape=(h, w),
                    input_occ=input_occ,
                    output_occ=output_occ,
                    shell_color=shell_color,
                    kernel_color=kernel_color,
                    direction=concave_dirs[dir_idx],
                    bg=bg,
                )
            if spec is None:
                continue
            r0, c0, obj_h, obj_w = spec["bbox"]
            patch = spec["patch"]
            view = grid[r0 : r0 + obj_h, c0 : c0 + obj_w]
            mask = patch != bg
            assert np.all(view[mask] == bg)
            view[mask] = patch[mask]
            mark_bbox(input_occ, r0, c0, obj_h, obj_w)
            mark_coords(output_occ, spec["output_coords"])
            specs.append(spec)
            if kind == "concave":
                dir_idx += 1
            break
        else:
            return None

    if get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
