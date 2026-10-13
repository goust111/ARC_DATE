from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, crop_bbox, find_components, get_bg, in_bounds, neighbors8, paste


BG = Color.BLACK
SRC = Color.RED
MARK = Color.GREY
SYM_H = "H"
SYM_V = "V"
SYM_D = "D"
DIAG_MAIN = "main"
DIAG_ANTI = "anti"
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
GRID_MIN = 20
GRID_MAX = 24

_H_BANK = [
    np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0, 0], [0, 1, 1, 1], [1, 1, 0, 0]], dtype=bool),
]
_V_BANK = [mask.T.copy() for mask in _H_BANK]
_D_MAIN_BANK = [
    np.array([[1, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=bool),
    np.array([[0, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool),
]


def _anti_transpose(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(mask)).T


def _marker_color(marker_count: int) -> int:
    if marker_count == 0:
        return Color.BLUE
    if marker_count == 1:
        return Color.GREEN
    return Color.YELLOW


def _symmetry_flags(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    diag_main = mask.shape[0] == mask.shape[1] and np.array_equal(mask, mask.T)
    diag_anti = mask.shape[0] == mask.shape[1] and np.array_equal(mask, _anti_transpose(mask))
    return horizontal, vertical, diag_main, diag_anti


def _classify_mask(mask: np.ndarray) -> tuple[str, str | None]:
    horizontal, vertical, diag_main, diag_anti = _symmetry_flags(mask)
    diagonal = diag_main or diag_anti
    assert sum((horizontal, vertical, diagonal)) == 1, "each shape must have exactly one reflection category"
    if horizontal:
        return SYM_H, None
    if vertical:
        return SYM_V, None
    if diag_main and not diag_anti:
        return SYM_D, DIAG_MAIN
    if diag_anti and not diag_main:
        return SYM_D, DIAG_ANTI
    raise AssertionError("diagonal shapes must have exactly one diagonal axis")


def _copy_plan(mask: np.ndarray, category: str, diag_kind: str | None) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    if category == SYM_H:
        return np.flipud(mask), (h, 0)
    if category == SYM_V:
        return np.fliplr(mask), (0, w)
    assert diag_kind in {DIAG_MAIN, DIAG_ANTI}
    if diag_kind == DIAG_MAIN:
        return mask.T.copy(), (h, w)
    return _anti_transpose(mask), (h, w)


def _dest_bbox(src_bbox: tuple[int, int, int, int], mask: np.ndarray, category: str) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = src_bbox
    h, w = mask.shape
    if category == SYM_H:
        return (r0 + h, c0, r0 + 2 * h - 1, c0 + w - 1)
    if category == SYM_V:
        return (r0, c0 + w, r0 + h - 1, c0 + 2 * w - 1)
    return (r0 + h, c0 + w, r0 + 2 * h - 1, c0 + 2 * w - 1)


def _footprint_bbox(src_bbox: tuple[int, int, int, int], mask: np.ndarray, category: str) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = src_bbox
    h, w = mask.shape
    if category == SYM_H:
        return (r0, c0, r0 + 2 * h - 1, c0 + w - 1)
    if category == SYM_V:
        return (r0, c0, r0 + h - 1, c0 + 2 * w - 1)
    return (r0, c0, r0 + 2 * h - 1, c0 + 2 * w - 1)


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _min_chebyshev(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    diffs = np.abs(coords_a[:, None, :] - coords_b[None, :, :])
    return int(np.min(np.max(diffs, axis=2)))


def _shape_sets(coords: np.ndarray, shape: tuple[int, int]) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    source = {tuple(map(int, rc)) for rc in coords}
    neigh8: set[tuple[int, int]] = set()
    neigh4: set[tuple[int, int]] = set()
    for r, c in source:
        for rr, cc in neighbors8(r, c):
            if in_bounds(shape, rr, cc) and (rr, cc) not in source:
                neigh8.add((int(rr), int(cc)))
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if in_bounds(shape, rr, cc) and (rr, cc) not in source:
                neigh4.add((int(rr), int(cc)))
    return source, neigh8, neigh4


def _paint_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _analyze_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid exceeds ARC size limit"
    assert int(get_bg(g)) == BG, "expected black background"
    assert set(map(int, np.unique(g))) <= {BG, SRC, MARK}, "unexpected colors in input"

    objects = find_components(g, bg=BG, connectivity=4, monochromatic=True, colors=[SRC], return_format="dict")
    objects = sorted(objects, key=lambda obj: tuple(map(int, obj["bbox"])))
    assert 3 <= len(objects) <= 4, "expected 3-4 red shapes"
    assert sum(int(obj["area"]) for obj in objects) == int((g == SRC).sum())

    grey_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == MARK)}
    memberships = {cell: 0 for cell in grey_cells}
    infos: list[dict] = []

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        bbox = tuple(map(int, obj["bbox"]))
        mask = crop_bbox(g == SRC, bbox)
        assert 4 <= int(obj["area"]) <= 8, "shape area must stay in the puzzle range"
        category, diag_kind = _classify_mask(mask)
        copy_mask, _offset = _copy_plan(mask, category, diag_kind)
        dst_bbox = _dest_bbox(bbox, mask, category)
        footprint = _footprint_bbox(bbox, mask, category)
        assert 0 <= dst_bbox[0] <= dst_bbox[2] < g.shape[0]
        assert 0 <= dst_bbox[1] <= dst_bbox[3] < g.shape[1]
        assert np.all(crop_bbox(g, dst_bbox) == BG), "destination block must be empty in the input"

        source_set, neigh8, neigh4 = _shape_sets(coords, g.shape)
        markers = sorted(cell for cell in neigh8 if g[cell] == MARK)
        assert len(markers) <= 3, "marker count must stay within the generator contract"
        for cell in markers:
            assert cell not in neigh4, "markers must be diagonal-only neighbors"
            memberships[cell] += 1

        infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "mask": mask,
                "category": category,
                "diag_kind": diag_kind,
                "copy_mask": copy_mask,
                "dst_bbox": dst_bbox,
                "footprint": footprint,
                "markers": markers,
                "marker_count": len(markers),
                "copy_color": _marker_color(len(markers)),
                "source_set": source_set,
                "neigh8": neigh8,
            }
        )

    for i, info_a in enumerate(infos):
        for info_b in infos[i + 1 :]:
            assert _min_chebyshev(info_a["coords"], info_b["coords"]) >= 4, "source shapes must be well separated"
            assert not _boxes_intersect(info_a["dst_bbox"], info_b["dst_bbox"]), "copy blocks must not overlap"

    assert set(memberships) == grey_cells, "every grey pixel must belong to some shape neighborhood"
    assert all(count == 1 for count in memberships.values()), "grey markers must belong to exactly one shape"
    assert len({info["category"] for info in infos}) >= 2, "samples must use multiple symmetry categories"
    assert len({info["copy_color"] for info in infos}) >= 2, "samples must use multiple output colors"
    assert any(
        info_a["category"] == info_b["category"] and info_a["copy_color"] != info_b["copy_color"]
        for i, info_a in enumerate(infos)
        for info_b in infos[i + 1 :]
    ), "need same-direction objects with different marker-driven colors"
    assert any(
        info_a["category"] != info_b["category"] and info_a["copy_color"] == info_b["copy_color"]
        for i, info_a in enumerate(infos)
        for info_b in infos[i + 1 :]
    ), "need different directions that still share a marker-driven color"
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-stamp each red shape into a symmetry-selected adjacent block, with the stamp color set by nearby grey markers.

    Input structure:
    - A black background contains 3-4 separated red 4-connected shapes and a few grey marker pixels.
    - Every red shape has exactly one reflection category: horizontal-axis, vertical-axis,
      or diagonal-axis symmetry; diagonal shapes may use either diagonal axis.
    - Grey markers are diagonal-only 8-neighbors of exactly one red shape and never occupy any copy block.

    Rule:
    - Keep the original red shapes and grey markers unchanged.
    - For each red shape, use its symmetry category to choose the stamp destination:
      horizontal-symmetric shapes stamp directly below, vertical-symmetric shapes stamp directly right,
      and diagonal-symmetric shapes stamp diagonally down-right into an equally sized adjacent block.
    - The stamp is the corresponding mirror of the source shape across its own symmetry axis,
      and its color depends only on the number of adjacent grey markers: 0 -> blue, 1 -> green, 2 or 3 -> yellow.

    Key insight:
    - Symmetry and marker count are independent information channels: one decides where and how the copy is mirrored,
      while the other decides only the copy color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _analyze_grid(g):
        patch = _paint_patch(info["copy_mask"], info["copy_color"])
        paste(out, patch, info["dst_bbox"][0], info["dst_bbox"][1], transparent=BG)
    return out


def _sample_mask(category: str) -> tuple[np.ndarray, str | None]:
    if category == SYM_H:
        mask = random.choice(_H_BANK).copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask, None
    if category == SYM_V:
        mask = random.choice(_V_BANK).copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask, None
    base = random.choice(_D_MAIN_BANK).copy()
    if random.random() < 0.5:
        return base, DIAG_MAIN
    return np.fliplr(base), DIAG_ANTI


def _sample_marker_count(bucket: int) -> int:
    if bucket == 0:
        return 0
    if bucket == 1:
        return 1
    return random.choice([2, 3])


def _sample_specs() -> list[dict]:
    object_count = random.choice([3, 4])
    repeated_category = random.choice([SYM_H, SYM_V, SYM_D])
    other_categories = random.sample([cat for cat in [SYM_H, SYM_V, SYM_D] if cat != repeated_category], object_count - 2)
    categories = [repeated_category, repeated_category, *other_categories]

    first_bucket, second_bucket = random.sample([0, 1, 2], 2)
    buckets = [first_bucket, second_bucket, first_bucket]
    if object_count == 4:
        buckets.append(random.choice([0, 1, 2]))

    specs = []
    for category, bucket in zip(categories, buckets):
        mask, diag_kind = _sample_mask(category)
        specs.append(
            {
                "category": category,
                "diag_kind": diag_kind,
                "mask": mask,
                "marker_count": _sample_marker_count(bucket),
            }
        )
    random.shuffle(specs)
    return specs


def _place_sources(specs: list[dict], height: int, width: int) -> list[dict] | None:
    ordered = sorted(specs, key=lambda spec: _footprint_bbox((0, 0, 0, 0), spec["mask"], spec["category"])[2:], reverse=True)
    placed: list[dict] = []
    for spec in ordered:
        mask = spec["mask"]
        h, w = mask.shape
        if spec["category"] == SYM_H:
            foot_h, foot_w = 2 * h, w
        elif spec["category"] == SYM_V:
            foot_h, foot_w = h, 2 * w
        else:
            foot_h, foot_w = 2 * h, 2 * w
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - foot_h - 1)
            c0 = random.randint(1, width - foot_w - 1)
            src_bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            dst_bbox = _dest_bbox(src_bbox, mask, spec["category"])
            footprint = _footprint_bbox(src_bbox, mask, spec["category"])
            coords = np.argwhere(mask) + np.array([[r0, c0]])
            if any(_boxes_intersect(footprint, other["footprint"]) for other in placed):
                continue
            if any(_min_chebyshev(coords, other["coords"]) < 4 for other in placed):
                continue
            placed.append(
                {
                    **spec,
                    "src_bbox": src_bbox,
                    "dst_bbox": dst_bbox,
                    "footprint": footprint,
                    "coords": coords.astype(int),
                }
            )
            break
        else:
            return None
    return placed


def _marker_candidates(item: dict, all_items: list[dict], shape: tuple[int, int]) -> list[tuple[int, int]]:
    source_set, neigh8, neigh4 = _shape_sets(item["coords"], shape)
    destination_cells = set()
    for other in all_items:
        r0, c0, r1, c1 = other["dst_bbox"]
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                destination_cells.add((rr, cc))

    candidates = []
    for cell in sorted(neigh8 - neigh4):
        if cell in destination_cells or cell in source_set:
            continue
        if any(cell in other["neigh8"] for other in all_items if other is not item):
            continue
        candidates.append(cell)
    return candidates


def _assemble_grid(placed: list[dict], height: int, width: int) -> np.ndarray | None:
    for item in placed:
        source_set, neigh8, _neigh4 = _shape_sets(item["coords"], (height, width))
        item["source_set"] = source_set
        item["neigh8"] = neigh8

    chosen_markers: set[tuple[int, int]] = set()
    for item in sorted(placed, key=lambda info: info["marker_count"], reverse=True):
        candidates = [cell for cell in _marker_candidates(item, placed, (height, width)) if cell not in chosen_markers]
        if len(candidates) < item["marker_count"]:
            return None
        markers = random.sample(candidates, item["marker_count"])
        item["markers"] = markers
        chosen_markers.update(markers)

    grid = np.full((height, width), BG, dtype=int)
    for item in placed:
        rr, cc = item["coords"][:, 0], item["coords"][:, 1]
        grid[rr, cc] = SRC
    for item in placed:
        for r, c in item["markers"]:
            grid[r, c] = MARK
    return grid


def _validate_generated_grid(grid: np.ndarray) -> None:
    infos = _analyze_grid(grid)
    out = main(grid)
    assert out.shape == grid.shape
    assert np.array_equal(out[grid == SRC], np.full(int((grid == SRC).sum()), SRC)), "sources must stay red"
    assert np.array_equal(out[grid == MARK], np.full(int((grid == MARK).sum()), MARK)), "markers must stay grey"
    for info in infos:
        patch = crop_bbox(out, info["dst_bbox"])
        expected = _paint_patch(info["copy_mask"], info["copy_color"])
        assert np.array_equal(patch, expected), "stamped copy must match the rule"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        specs = _sample_specs()
        placed = _place_sources(specs, height, width)
        if placed is None:
            continue
        grid = _assemble_grid(placed, height, width)
        if grid is None:
            continue
        _validate_generated_grid(grid)
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")

