from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CORNERS = ("TL", "TR", "BR", "BL")
_NEXT_CORNER = {"TL": "TR", "TR": "BR", "BR": "BL", "BL": "TL"}


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _bbox_corner(bbox: tuple[int, int, int, int], h: int, w: int) -> str | None:
    r0, c0, r1, c1 = map(int, bbox)
    top = r0 == 0
    bottom = r1 == h - 1
    left = c0 == 0
    right = c1 == w - 1
    if top and left:
        return "TL"
    if top and right:
        return "TR"
    if bottom and right:
        return "BR"
    if bottom and left:
        return "BL"
    return None


def _corner_anchor(corner: str, h: int, w: int, obj_h: int, obj_w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - obj_w
    if corner == "BR":
        return h - obj_h, w - obj_w
    if corner == "BL":
        return h - obj_h, 0
    raise AssertionError(corner)


def _paint_component(out: np.ndarray, comp: dict) -> None:
    coords = comp["coords"]
    out[coords[:, 0], coords[:, 1]] = int(comp["color"])


def _copy_corner_objects(base: np.ndarray, corner_components: dict[str, dict]) -> np.ndarray:
    out = base.copy()
    h, w = out.shape
    planned: dict[tuple[int, int], int] = {}
    for corner, comp in corner_components.items():
        r0, c0, r1, c1 = map(int, comp["bbox"])
        obj_h = r1 - r0 + 1
        obj_w = c1 - c0 + 1
        dst_corner = _NEXT_CORNER[corner]
        dst_r0, dst_c0 = _corner_anchor(dst_corner, h, w, obj_h, obj_w)
        for rr, cc in comp["coords"]:
            tr = int(dst_r0 + int(rr) - r0)
            tc = int(dst_c0 + int(cc) - c0)
            planned[(tr, tc)] = int(comp["color"])
    for (r, c), color in planned.items():
        out[r, c] = color
    return out


def _nearest_fill_from_colors(seed_grid: np.ndarray) -> np.ndarray:
    out = seed_grid.copy()
    bg_coords = np.argwhere(out == BG)
    if len(bg_coords) == 0:
        return out

    seed_colors = sorted(int(c) for c in np.unique(out) if int(c) != BG)
    assert seed_colors
    coords_by_color = {color: np.argwhere(out == color) for color in seed_colors}

    for r, c in bg_coords:
        best_distance = None
        best_colors: list[int] = []
        rr = int(r)
        cc = int(c)
        for color, coords in coords_by_color.items():
            dists = np.abs(coords[:, 0] - rr) + np.abs(coords[:, 1] - cc)
            dist = int(dists.min())
            if best_distance is None or dist < best_distance:
                best_distance = dist
                best_colors = [color]
            elif dist == best_distance:
                best_colors.append(color)
        out[rr, cc] = best_colors[0] if len(best_colors) == 1 else SEP
    return out


def _solve(grid: np.ndarray, *, erase_distractors: bool, copy_corners: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    components = _find_components(g)
    corner_components: dict[str, dict] = {}
    distractors: list[dict] = []
    for comp in components:
        corner = _bbox_corner(comp["bbox"], h, w)
        if corner is None:
            distractors.append(comp)
        else:
            assert corner not in corner_components, "at most one corner object per corner"
            corner_components[corner] = comp

    work = np.full_like(g, BG)
    for comp in components:
        corner = _bbox_corner(comp["bbox"], h, w)
        if corner is not None or not erase_distractors:
            _paint_component(work, comp)

    if copy_corners:
        work = _copy_corner_objects(work, corner_components)
    return _nearest_fill_from_colors(work)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy corner-anchored objects clockwise, erase distractors, then Voronoi-fill from the resulting seeds.

    Input structure:
    - A black background contains 2-4 monochrome 4-connected foreground objects with distinct colors.
    - Exactly 2-3 objects are true corner objects whose bounding boxes touch two adjacent grid edges at one corner.
    - The remaining objects are distractors that lie in the interior or touch only one edge.

    Rule:
    - Keep only the corner objects and erase every distractor to black.
    - Copy each corner object's colored mask clockwise to the next corner by aligning its bounding box to that target corner, without deleting the source.
    - Treat all remaining foreground pixels after copying as seed regions and fill every black pixel by the color of the nearest seed region under Manhattan distance.
    - If the nearest distance is tied between different colors, paint that pixel gray; ties between same-color seeds stay that color.

    Key insight:
    - The answer depends on first classifying true corner objects, because the clockwise copies redefine the seed layout and any leftover distractor would corrupt the distance partition.
    """
    return _solve(grid, erase_distractors=True, copy_corners=True)


def _grow_connected_mask(
    h: int,
    w: int,
    must_have: set[tuple[int, int]],
    target_area: int,
) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in must_have:
        mask[r, c] = True

    def frontier() -> list[tuple[int, int]]:
        cells: set[tuple[int, int]] = set()
        for rr, cc in np.argwhere(mask):
            for dr, dc in _NEIGHBORS4:
                nr, nc = int(rr) + dr, int(cc) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    cells.add((nr, nc))
        return list(cells)

    while int(mask.sum()) < target_area:
        choices = frontier()
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = True

    rows, cols = np.where(mask)
    if rows.min() != 0 or rows.max() != h - 1 or cols.min() != 0 or cols.max() != w - 1:
        return None
    return mask


def _make_corner_patch(corner: str, color: int) -> np.ndarray | None:
    dims = [(hh, ww) for hh in range(2, 5) for ww in range(2, 5)]
    random.shuffle(dims)
    for h, w in dims:
        anchor = {
            "TL": (0, 0),
            "TR": (0, w - 1),
            "BR": (h - 1, w - 1),
            "BL": (h - 1, 0),
        }[corner]
        must_have = {anchor, (h - 1, random.randrange(w)), (random.randrange(h), w - 1)}
        max_area = min(10, h * w)
        min_area = max(4, len(must_have))
        if min_area > max_area:
            continue
        for _ in range(10):
            target_area = random.randint(min_area, max_area)
            mask = _grow_connected_mask(h, w, must_have, target_area)
            if mask is None:
                continue
            patch = np.zeros((h, w), dtype=int)
            patch[mask] = color
            return patch
    return None


def _make_blob_patch(color: int) -> np.ndarray | None:
    dims = [(hh, ww) for hh in range(2, 5) for ww in range(2, 5)]
    random.shuffle(dims)
    for h, w in dims:
        max_area = min(10, h * w)
        target_area = random.randint(4, max_area)
        start = {(random.randrange(h), random.randrange(w))}
        mask = _grow_connected_mask(h, w, start, target_area)
        if mask is None:
            continue
        patch = np.zeros((h, w), dtype=int)
        patch[mask] = color
        return patch
    return None


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _place_corner_patch(grid: np.ndarray, corner: str, patch: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    ph, pw = patch.shape
    r0, c0 = _corner_anchor(corner, h, w, ph, pw)
    grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
    return r0, c0, r0 + ph - 1, c0 + pw - 1


def _valid_distractor_positions(
    grid_h: int,
    grid_w: int,
    patch_h: int,
    patch_w: int,
    mode: str,
    existing_bboxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for r0 in range(grid_h - patch_h + 1):
        for c0 in range(grid_w - patch_w + 1):
            bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            touches = {
                "top": r0 == 0,
                "bottom": r0 + patch_h - 1 == grid_h - 1,
                "left": c0 == 0,
                "right": c0 + patch_w - 1 == grid_w - 1,
            }
            touched = sum(touches.values())
            if mode == "interior" and touched != 0:
                continue
            if mode == "side" and touched != 1:
                continue
            if touches["top"] and (touches["left"] or touches["right"]):
                continue
            if touches["bottom"] and (touches["left"] or touches["right"]):
                continue
            if any(_bbox_overlap(bbox, other, margin=1) for other in existing_bboxes):
                continue
            positions.append((r0, c0))
    return positions


def _structural_summary(grid: np.ndarray) -> dict:
    comps = _find_components(grid)
    h, w = grid.shape
    corner_map: dict[str, dict] = {}
    distractors: list[dict] = []
    for comp in comps:
        corner = _bbox_corner(comp["bbox"], h, w)
        if corner is None:
            distractors.append(comp)
        else:
            corner_map[corner] = comp
    return {"components": comps, "corners": corner_map, "distractors": distractors}


def _is_copy_essential(grid: np.ndarray, expected: np.ndarray) -> bool:
    simpler = _solve(grid, erase_distractors=True, copy_corners=False)
    return not np.array_equal(simpler, expected)


def _is_erase_essential(grid: np.ndarray, expected: np.ndarray) -> bool:
    polluted = _solve(grid, erase_distractors=False, copy_corners=True)
    return not np.array_equal(polluted, expected)


def _has_same_color_tie(seed_grid: np.ndarray) -> bool:
    bg_coords = np.argwhere(seed_grid == BG)
    colors = sorted(int(c) for c in np.unique(seed_grid) if int(c) != BG)
    coords_by_color = {color: np.argwhere(seed_grid == color) for color in colors}
    for r, c in bg_coords:
        rr = int(r)
        cc = int(c)
        for color, coords in coords_by_color.items():
            dists = np.abs(coords[:, 0] - rr) + np.abs(coords[:, 1] - cc)
            if int(np.sum(dists == dists.min())) >= 2:
                return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(11, 16)
    w = random.randint(11, 16)
    grid = np.zeros((h, w), dtype=int)

    corner_count = random.choice((2, 2, 3, 3, 3))
    chosen_corners = random.sample(list(_CORNERS), k=corner_count)
    distractor_count = random.choice((1, 1, 2))
    total_objects = corner_count + distractor_count

    colors = random.sample(PALETTE, k=total_objects)
    random.shuffle(colors)

    bboxes: list[tuple[int, int, int, int]] = []
    for i, corner in enumerate(chosen_corners):
        patch = _make_corner_patch(corner, colors[i])
        if patch is None:
            return None
        bbox = _place_corner_patch(grid, corner, patch)
        bboxes.append(bbox)

    remaining_colors = colors[corner_count:]
    modes_pool = ["interior", "side"]
    distractor_modes = [random.choice(modes_pool) for _ in range(distractor_count)]
    if distractor_count == 2 and distractor_modes[0] == distractor_modes[1]:
        distractor_modes[1] = "side" if distractor_modes[0] == "interior" else "interior"

    for color, mode in zip(remaining_colors, distractor_modes):
        patch = _make_blob_patch(color)
        if patch is None:
            return None
        ph, pw = patch.shape
        positions = _valid_distractor_positions(h, w, ph, pw, mode, bboxes)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG] = patch[patch != BG]
        bboxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    summary = _structural_summary(grid)
    if len(summary["components"]) != total_objects:
        return None
    if len(summary["corners"]) != corner_count or len(summary["distractors"]) != distractor_count:
        return None
    if len(summary["corners"]) >= 4:
        return None
    if any(int(comp["area"]) < 4 or int(comp["area"]) > 10 for comp in summary["components"]):
        return None
    if len({int(comp["color"]) for comp in summary["components"]}) != total_objects:
        return None
    if any(_bbox_overlap(a["bbox"], b["bbox"]) for i, a in enumerate(summary["components"]) for b in summary["components"][i + 1 :]):
        return None

    expected = main(grid)
    if expected.shape != grid.shape:
        return None
    if not np.any(expected == SEP):
        return None
    if not _is_copy_essential(grid, expected):
        return None
    if not _is_erase_essential(grid, expected):
        return None

    cleaned = np.zeros_like(grid)
    for comp in summary["corners"].values():
        _paint_component(cleaned, comp)
    copied = _copy_corner_objects(cleaned, summary["corners"])
    if not _has_same_color_tie(copied):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after retries")
