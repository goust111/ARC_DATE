from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
GRAY = 5
BACKGROUND_CHOICES = (0, 1, 2, 3, 4, 6, 7, 8, 9)
OBJECT_COLOR_CHOICES = (1, 2, 3, 4, 6, 7, 8, 9)
INPUT_HEIGHT_RANGE = (9, 16)  # exclusive upper
INPUT_WIDTH_RANGE = (12, 27)  # exclusive upper
MAX_GENERATION_ATTEMPTS = 400
PACK_RETRIES = 40


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _fallback_neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _fallback_bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fallback_components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = [(r, c)]
            seen[r, c] = True
            coords = []
            while queue:
                cr, cc = queue.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors=None,
    min_area: int = 1,
    return_format: str = "dict",
):
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _fallback_get_bg(g)
    bg = int(bg)
    if not monochromatic:
        raise ValueError("fallback find_components only supports monochromatic=True for this task")
    if return_format != "dict":
        raise ValueError("fallback find_components only supports return_format='dict' for this task")
    if colors is None:
        colors = sorted(set(map(int, g.reshape(-1))) - {bg})

    out = []
    for color in colors:
        if int(color) == bg:
            continue
        for coords in _fallback_components_mask(g == int(color), connectivity=connectivity):
            if int(coords.shape[0]) < int(min_area):
                continue
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _fallback_bbox_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


try:
    _arc_utils = SourceFileLoader("_arc_utils_039_7", str(ARC_UTILS_PATH)).load_module()
    get_bg = _arc_utils.get_bg
    neighbors4 = _arc_utils.neighbors4
    bbox_coords = _arc_utils.bbox_coords
    components_mask = _arc_utils.components_mask
    find_components = _arc_utils.find_components
except Exception:
    get_bg = _fallback_get_bg
    neighbors4 = _fallback_neighbors4
    bbox_coords = _fallback_bbox_coords
    components_mask = _fallback_components_mask
    find_components = _fallback_find_components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != 0)
    r0, c0, r1, c1 = bbox_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_transform_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = patch.copy()
    if int(rng.integers(0, 2)):
        out = np.fliplr(out)
    return np.rot90(out, int(rng.integers(0, 4))).copy()


def _grow_from_seed(
    allowed: np.ndarray,
    seed: tuple[int, int],
    area: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    allowed = np.asarray(allowed, dtype=bool)
    if area <= 0 or not allowed[seed]:
        return None
    mask = np.zeros_like(allowed, dtype=bool)
    mask[seed] = True

    while int(mask.sum()) < int(area):
        frontier = set()
        for r, c in np.argwhere(mask):
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and allowed[nr, nc] and not mask[nr, nc]:
                    frontier.add((int(nr), int(nc)))
        if not frontier:
            return None
        choices = list(frontier)
        pick = choices[int(rng.integers(0, len(choices)))]
        mask[pick] = True
    return mask


def _random_polyomino_mask(
    rng: np.random.Generator,
    *,
    area_range: tuple[int, int] = (3, 7),
    box_h_range: tuple[int, int] = (2, 5),
    box_w_range: tuple[int, int] = (2, 5),
) -> np.ndarray:
    for _ in range(80):
        h = int(rng.integers(box_h_range[0], box_h_range[1] + 1))
        w = int(rng.integers(box_w_range[0], box_w_range[1] + 1))
        max_area = min(int(area_range[1]), h * w)
        if max_area < int(area_range[0]):
            continue
        area = int(rng.integers(area_range[0], max_area + 1))
        seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        mask = _grow_from_seed(np.ones((h, w), dtype=bool), seed, area, rng)
        if mask is None:
            continue
        return _crop_mask(mask)
    raise RuntimeError("failed to generate a connected polyomino")


def _random_object_mask(rng: np.random.Generator) -> np.ndarray:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return _random_polyomino_mask(rng)
    if mode == 1:
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        return np.ones((h, w), dtype=bool)

    h = int(rng.integers(3, 6))
    w = int(rng.integers(3, 6))
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return _random_transform_patch(mask.astype(int), rng).astype(bool)


def _make_plain_object_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    mask = _random_object_mask(rng)
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _make_special_counterexample_patch(rng: np.random.Generator, host_color: int, decoy_color: int) -> np.ndarray:
    patch = np.zeros((5, 7), dtype=int)
    patch[0, 0:4] = int(decoy_color)
    patch[4, 0:4] = int(decoy_color)
    patch[1:4, 0] = int(decoy_color)
    patch[2, 2:4] = GRAY

    host_options = (
        ((2, 4), (2, 5)),
        ((2, 4), (2, 5), (1, 5)),
        ((2, 4), (2, 5), (3, 5)),
        ((1, 4), (2, 4), (2, 5)),
        ((2, 4), (3, 4), (2, 5)),
    )
    for r, c in host_options[int(rng.integers(0, len(host_options)))]:
        patch[r, c] = int(host_color)
    return _random_transform_patch(patch, rng)


def _make_normal_blob_patch(
    rng: np.random.Generator,
    host_color: int,
    *,
    force_large_blob: bool,
) -> np.ndarray | None:
    for _ in range(80):
        host = _random_object_mask(rng)
        host_h, host_w = map(int, host.shape)
        canvas = np.zeros((host_h + 8, host_w + 8), dtype=int)
        r0 = (canvas.shape[0] - host_h) // 2
        c0 = (canvas.shape[1] - host_w) // 2
        canvas[r0 : r0 + host_h, c0 : c0 + host_w][host] = int(host_color)

        host_mask = canvas == int(host_color)
        seeds = []
        for r in range(canvas.shape[0]):
            for c in range(canvas.shape[1]):
                if canvas[r, c] != 0:
                    continue
                if any(
                    0 <= nr < canvas.shape[0] and 0 <= nc < canvas.shape[1] and host_mask[nr, nc]
                    for nr, nc in neighbors4(r, c)
                ):
                    seeds.append((r, c))
        if not seeds:
            continue

        seed = seeds[int(rng.integers(0, len(seeds)))]
        rr, cc = np.indices(canvas.shape)
        radius = 4 if force_large_blob else 3
        allowed = (canvas == 0) & ((np.abs(rr - seed[0]) + np.abs(cc - seed[1])) <= radius)

        min_area = 2 if force_large_blob else 1
        max_area = min(4, int(allowed.sum()))
        if max_area < min_area:
            continue
        blob_area = int(rng.integers(min_area, max_area + 1))
        blob = _grow_from_seed(allowed, seed, blob_area, rng)
        if blob is None:
            continue

        canvas[blob] = GRAY
        return _random_transform_patch(_crop_patch(canvas), rng)
    return None


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _center_distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "grid too small"

    bg = int(get_bg(g))
    assert bg != GRAY, "background must differ from gray cover color"
    assert np.any(g == GRAY), "expected at least one gray blob"

    top = g.copy()
    top[g == GRAY] = bg

    fg_colors = sorted(set(map(int, top.reshape(-1))) - {bg})
    assert fg_colors, "expected visible foreground objects"
    fg_components = find_components(
        top,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=fg_colors,
        return_format="dict",
    )
    assert fg_components, "expected at least one foreground component"

    coord_to_component: dict[tuple[int, int], int] = {}
    for idx, component in enumerate(fg_components):
        for r, c in component["coords"]:
            coord_to_component[(int(r), int(c))] = int(idx)

    gray_components = []
    for coords in components_mask(g == GRAY, connectivity=4):
        touching = set()
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w:
                    comp_idx = coord_to_component.get((int(nr), int(nc)))
                    if comp_idx is not None:
                        touching.add(int(comp_idx))
        assert len(touching) == 1, "every gray blob must attach to exactly one foreground object"
        host_idx = next(iter(touching))
        gray_components.append(
            {
                "coords": coords.astype(int),
                "bbox": bbox_coords(coords),
                "host_id": host_idx,
                "host_color": int(fg_components[host_idx]["color"]),
            }
        )

    return {
        "grid": g,
        "height": h,
        "width": w,
        "bg": bg,
        "top": top,
        "foreground_components": fg_components,
        "gray_components": gray_components,
    }


def _render_output(info: dict) -> np.ndarray:
    top = info["top"]
    h = int(info["height"])
    out = np.vstack([top, np.flipud(top)])
    for blob in info["gray_components"]:
        rows = h + (h - 1 - blob["coords"][:, 0])
        cols = blob["coords"][:, 1]
        out[rows, cols] = int(blob["host_color"])
    return out


def _has_counterexample_case(info: dict) -> bool:
    components = info["foreground_components"]
    for blob in info["gray_components"]:
        blob_center = _bbox_center(blob["bbox"])
        host_center = _bbox_center(components[blob["host_id"]]["bbox"])
        host_dist = _center_distance_sq(blob_center, host_center)
        for idx, component in enumerate(components):
            if idx == blob["host_id"]:
                continue
            if _center_distance_sq(blob_center, _bbox_center(component["bbox"])) < host_dist:
                return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
    except AssertionError:
        return False

    if int(info["height"]) * 2 > 30 or int(info["width"]) > 30:
        return False
    if len(info["gray_components"]) < 2:
        return False
    if len({blob["host_color"] for blob in info["gray_components"]}) < 2:
        return False
    if max(int(blob["coords"].shape[0]) for blob in info["gray_components"]) < 2:
        return False
    if not _has_counterexample_case(info):
        return False

    out = _render_output(info)
    if np.any(out[: info["height"]] == GRAY):
        return False

    mirror = np.flipud(info["top"])
    lower = out[info["height"] :]
    reveal_mask = np.zeros_like(info["top"], dtype=bool)
    for blob in info["gray_components"]:
        reveal_mask[blob["coords"][:, 0], blob["coords"][:, 1]] = True
    if not np.array_equal(lower[~np.flipud(reveal_mask)], mirror[~np.flipud(reveal_mask)]):
        return False
    return True


def _pack_patches(
    patches: list[np.ndarray],
    bg: int,
    height: int,
    width: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    grid = np.full((height, width), int(bg), dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = map(int, patch.shape)
        if ph > height or pw > width:
            return None

        positions = [(r, c) for r in range(height - ph + 1) for c in range(width - pw + 1)]
        order = rng.permutation(len(positions))
        placed = False

        for idx in order:
            r0, c0 = positions[int(idx)]
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if np.any(reserved[rr0:rr1, cc0:cc1]):
                continue

            mask = patch != 0
            view = grid[r0 : r0 + ph, c0 : c0 + pw]
            if np.any(mask & (view != int(bg))):
                continue

            view[mask] = patch[mask]
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg = int(rng.choice(BACKGROUND_CHOICES))
    available_colors = [color for color in OBJECT_COLOR_CHOICES if int(color) != bg]

    normal_blob_count = int(rng.integers(1, 4))
    plain_count = int(rng.integers(0, 3))
    total_object_colors = 2 + normal_blob_count + plain_count
    if total_object_colors > len(available_colors):
        return None

    colors = rng.choice(available_colors, size=total_object_colors, replace=False).tolist()
    patches = [_make_special_counterexample_patch(rng, colors[0], colors[1])]

    color_index = 2
    for i in range(normal_blob_count):
        patch = _make_normal_blob_patch(rng, colors[color_index], force_large_blob=(i == 0))
        if patch is None:
            return None
        patches.append(patch)
        color_index += 1

    for _ in range(plain_count):
        patches.append(_make_plain_object_patch(rng, colors[color_index]))
        color_index += 1

    padded_area = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in patches)
    max_h = max(int(patch.shape[0]) for patch in patches)
    max_w = max(int(patch.shape[1]) for patch in patches)

    for _ in range(PACK_RETRIES):
        height_low = max(INPUT_HEIGHT_RANGE[0], max_h + 1)
        if height_low >= INPUT_HEIGHT_RANGE[1]:
            return None
        height = int(rng.integers(height_low, INPUT_HEIGHT_RANGE[1]))

        width_low = max(
            INPUT_WIDTH_RANGE[0],
            max_w + 1,
            int(np.ceil(padded_area / max(height - 1, 1))),
        )
        if width_low >= INPUT_WIDTH_RANGE[1]:
            continue
        width = int(rng.integers(width_low, INPUT_WIDTH_RANGE[1]))

        grid = _pack_patches(patches, bg, height, width, rng)
        if grid is not None and _validate_generated(grid):
            return grid
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase gray cover blobs from the original scene, mirror the cleaned scene downward, and reveal the blobs only in the mirror using their host colors.

    Input structure:
    - The background contains several monochrome foreground objects plus several 4-connected gray blobs.
    - Gray cells are a cover layer, not foreground objects; each gray blob touches exactly one foreground object by 4-neighbor adjacency.
    - Different gray blobs may attach to different objects, and nearby decoy objects can sit closer in appearance than the true host.

    Rule:
    - Remove every gray blob by repainting it with the background color.
    - Build an output whose height is doubled: the upper half is the cleaned scene, and the lower half is the row-reversed mirror of that cleaned scene.
    - For each gray blob, map every blob cell to its mirrored row in the lower half and recolor that mirrored cell with the color of the unique touched foreground object.

    Key insight:
    - The gray pattern is latent information that exists only in the reflected half, so the solver must separate erase, mirror, and host-by-adjacency recoloring instead of using proximity to a visually nearest object.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    return _render_output(info)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-reveal ARC input")
