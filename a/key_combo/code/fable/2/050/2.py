from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
_ARC_UTILS_SPEC = importlib.util.spec_from_file_location("_arc_utils_fable_2_050_2", _ARC_UTILS_PATH)
assert _ARC_UTILS_SPEC is not None and _ARC_UTILS_SPEC.loader is not None
_ARC_UTILS = importlib.util.module_from_spec(_ARC_UTILS_SPEC)
_ARC_UTILS_SPEC.loader.exec_module(_ARC_UTILS)

bbox_coords = _ARC_UTILS.bbox_coords
colors_present = _ARC_UTILS.colors_present
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
paste = _ARC_UTILS.paste


Coord = tuple[int, int]
Shift = tuple[int, int]

BG = 0
PALETTE = tuple(range(1, 10))

TEMPLATE_H_RANGE = (5, 8)
TEMPLATE_W_RANGE = (5, 8)
FRAGMENT_COUNT_CHOICES = (3, 4)
CANVAS_MIN = 18
CANVAS_MAX = 30
PLACEMENT_PAD = 2
GEN_ATTEMPTS = 400
TEMPLATE_ATTEMPTS = 120
MISSING_ATTEMPTS = 160
PLACEMENT_ATTEMPTS = 80


def _neighbors4(cell: Coord) -> tuple[Coord, Coord, Coord, Coord]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _coord_array(coords: set[Coord] | frozenset[Coord] | list[Coord]) -> np.ndarray:
    return np.array(sorted(coords), dtype=int)


def _bbox_of_coords(coords: set[Coord] | frozenset[Coord] | list[Coord]) -> tuple[int, int, int, int]:
    arr = _coord_array(coords)
    assert arr.size > 0
    return bbox_coords(arr)


def _normalize_coords(coords: np.ndarray) -> frozenset[Coord]:
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _shift_coords(coords: frozenset[Coord], shift: Shift) -> frozenset[Coord]:
    dr, dc = shift
    return frozenset((r + dr, c + dc) for r, c in coords)


def _best_translation(a: frozenset[Coord], b: frozenset[Coord]) -> tuple[Shift, int]:
    scores: dict[Shift, int] = {}
    for ar, ac in a:
        for br, bc in b:
            shift = (ar - br, ac - bc)
            scores[shift] = scores.get(shift, 0) + 1
    assert scores, "expected non-empty fragments"
    best = max(scores.values())
    winners = [shift for shift, score in scores.items() if score == best]
    assert len(winners) == 1, "pairwise alignment must be unique"
    return winners[0], int(best)


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    colors = [int(color) for color in colors_present(grid, exclude={bg})]
    assert len(colors) >= 2, "expected multiple colored fragments"
    fragments = []
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        fragments.append(
            {
                "color": color,
                "coords": coords,
                "norm": _normalize_coords(coords),
            }
        )
    return fragments


def _solve_state(grid: np.ndarray) -> tuple[int, list[dict], list[frozenset[Coord]], frozenset[Coord]]:
    fragments = _extract_fragments(grid)
    n = len(fragments)
    anchor = max(range(n), key=lambda idx: (len(fragments[idx]["norm"]), -fragments[idx]["color"]))

    pair_shift: dict[tuple[int, int], Shift] = {}
    pair_overlap: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            shift, overlap = _best_translation(fragments[i]["norm"], fragments[j]["norm"])
            pair_shift[(i, j)] = shift
            pair_overlap[(i, j)] = overlap

    shifts: list[Shift] = [(0, 0)] * n
    for idx in range(n):
        if idx == anchor:
            continue
        shifts[idx] = pair_shift[(anchor, idx)]

    aligned = [_shift_coords(fragments[idx]["norm"], shifts[idx]) for idx in range(n)]
    template = frozenset().union(*aligned)
    assert template, "template must be non-empty"

    for idx, cells in enumerate(aligned):
        assert cells < template, f"fragment {idx} must be incomplete"

    for i in range(n):
        for j in range(i + 1, n):
            expected = (shifts[j][0] - shifts[i][0], shifts[j][1] - shifts[i][1])
            assert pair_shift[(i, j)] == expected, "pairwise translations must be globally consistent"
            assert pair_overlap[(i, j)] >= 1, "fragments must overlap after alignment"
            assert aligned[i] | aligned[j] == template, "any fragment pair must reconstruct the template"

    missing_sets = [template - cells for cells in aligned]
    for i, missing in enumerate(missing_sets):
        assert missing, f"fragment {i} must miss at least one template cell"
        for other in missing_sets[i + 1 :]:
            assert missing.isdisjoint(other), "missing regions must be pairwise disjoint"

    common = set.intersection(*map(set, aligned))
    assert common, "fragments should share a visible common core"
    return int(get_bg(grid)), fragments, aligned, template


def _negative_space_output(
    bg: int,
    fragments: list[dict],
    aligned: list[frozenset[Coord]],
    template: frozenset[Coord],
) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_coords(template)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for fragment, cells in zip(fragments, aligned):
        for r, c in template - cells:
            rr = r - r0
            cc = c - c0
            assert out[rr, cc] == bg, "each missing template cell must belong to exactly one fragment"
            out[rr, cc] = int(fragment["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align several incomplete translated copies of one hidden template, then color each copy's own missing cells.

    Input structure:
    - A black background contains several separated monochrome fragments in distinct colors.
    - Each color is one incomplete translated copy of the same unseen template; every copy is missing a different template region.
    - No fragment is rotated or flipped, and pairwise visible overlap is enough to determine a unique translation alignment.

    Rule:
    - Group cells by color and align all fragments into one shared template coordinate system using translation only.
    - Reconstruct the full template as the union of the aligned fragments, then crop the output to that template's bounding box.
    - For every template cell absent from exactly one fragment, paint that cell with the missing fragment's own color; cells shared by all fragments stay background.

    Key insight:
    - The answer is the complementary negative space, not the recovered template itself: alignment reveals which template cells each colored fragment lacks.
    """
    g = np.asarray(grid, dtype=int)
    bg, fragments, aligned, template = _solve_state(g)
    return _negative_space_output(bg, fragments, aligned, template)


def _mask_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask, connectivity=4)) == 1


def _touches_bbox_edges(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_asymmetric(mask: np.ndarray) -> bool:
    if np.array_equal(mask, np.flipud(mask)) or np.array_equal(mask, np.fliplr(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            return False
    return True


def _make_template_mask() -> np.ndarray:
    for _ in range(TEMPLATE_ATTEMPTS):
        h = random.randint(*TEMPLATE_H_RANGE)
        w = random.randint(*TEMPLATE_W_RANGE)
        area_min = max(h + w - 2, int(np.ceil(0.55 * h * w)))
        area_max = max(area_min, int(np.floor(0.82 * h * w)))
        if area_max >= h * w:
            area_max = h * w - 1
        if area_min > area_max:
            continue
        target_area = random.randint(area_min, area_max)
        mask = np.ones((h, w), dtype=bool)
        while int(mask.sum()) > target_area:
            removable = []
            for r, c in np.argwhere(mask):
                trial = mask.copy()
                trial[r, c] = False
                if _mask_connected(trial) and _touches_bbox_edges(trial):
                    removable.append((int(r), int(c)))
            if not removable:
                break
            edge_cells = [cell for cell in removable if cell[0] in (0, h - 1) or cell[1] in (0, w - 1)]
            if edge_cells and random.random() < 0.45:
                r, c = random.choice(edge_cells)
            else:
                r, c = random.choice(removable)
            mask[r, c] = False
        if int(mask.sum()) != target_area:
            continue
        if not _mask_connected(mask) or not _touches_bbox_edges(mask):
            continue
        if not _mask_asymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate template")


def _positive_partition(total: int, parts: int, max_part: int) -> list[int] | None:
    for _ in range(80):
        values = [1] * parts
        remaining = total - parts
        while remaining > 0:
            idx = random.randrange(parts)
            if values[idx] >= max_part:
                continue
            values[idx] += 1
            remaining -= 1
        if max(values) > 1:
            random.shuffle(values)
            return values
    return None


def _grow_region(
    available: set[Coord],
    size: int,
    *,
    start_pool: set[Coord],
    restrict_pool: set[Coord] | None,
) -> set[Coord] | None:
    starts = list(start_pool & available)
    random.shuffle(starts)
    for start in starts:
        region = {start}
        while len(region) < size:
            frontier = set()
            for cell in region:
                for nbr in _neighbors4(cell):
                    if nbr not in available or nbr in region:
                        continue
                    if restrict_pool is not None and nbr not in restrict_pool:
                        continue
                    frontier.add(nbr)
            if not frontier:
                break
            region.add(random.choice(tuple(frontier)))
        if len(region) == size:
            return region
    return None


def _sample_missing_regions(mask: np.ndarray, n_fragments: int) -> list[set[Coord]] | None:
    cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
    h, w = mask.shape
    boundary = {cell for cell in cells if cell[0] in (0, h - 1) or cell[1] in (0, w - 1)}
    interior = cells - boundary
    common_min = max(3, len(cells) // 3)
    total_max = min(len(cells) - common_min, 4 * n_fragments)
    if total_max < n_fragments + 1:
        return None

    for _ in range(MISSING_ATTEMPTS):
        total_missing = random.randint(n_fragments + 1, total_max)
        sizes = _positive_partition(total_missing, n_fragments, max_part=4)
        if sizes is None:
            continue

        specs: list[tuple[str, int]] = []
        size_pool = sorted(sizes)
        if interior:
            specs.append(("interior", size_pool.pop(0)))
        if boundary and size_pool:
            specs.append(("boundary", size_pool.pop(0)))
        specs.extend(("any", size) for size in size_pool)

        available = set(cells)
        regions: list[set[Coord]] = []
        ok = True
        for kind, size in specs:
            if kind == "interior":
                region = _grow_region(available, size, start_pool=interior, restrict_pool=interior)
            elif kind == "boundary":
                region = _grow_region(available, size, start_pool=boundary, restrict_pool=None)
            else:
                region = _grow_region(available, size, start_pool=available, restrict_pool=None)
            if region is None:
                ok = False
                break
            regions.append(region)
            available -= region
        if not ok:
            continue

        fragments = [cells - region for region in regions]
        if any(not fragment for fragment in fragments):
            continue
        connected_flags = [_mask_connected(_cells_to_mask(fragment)) for fragment in fragments]
        if not any(connected_flags):
            continue
        common = set.intersection(*map(set, fragments))
        if len(common) < common_min:
            continue
        template_box = (0, 0, h - 1, w - 1)
        fragment_boxes = [_bbox_of_coords(fragment) for fragment in fragments]
        if not any(box != template_box for box in fragment_boxes):
            continue
        if interior and not any(box == template_box for box in fragment_boxes):
            continue
        return regions
    return None


def _cells_to_mask(cells: set[Coord] | frozenset[Coord]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_coords(cells)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _fragment_patch(fragment_cells: set[Coord], color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_of_coords(fragment_cells)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in fragment_cells:
        patch[r - r0, c - c0] = color
    return patch


def _place_patches(patches: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(PLACEMENT_ATTEMPTS):
        max_h = max(patch.shape[0] for patch in patches)
        max_w = max(patch.shape[1] for patch in patches)
        h = random.randint(max(CANVAS_MIN, max_h + 4), CANVAS_MAX)
        w = random.randint(max(CANVAS_MIN, max_w + 4), CANVAS_MAX)
        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        positions: list[tuple[int, int] | None] = [None] * len(patches)
        order = sorted(range(len(patches)), key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)
        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = patch.shape
            candidates = []
            for r0 in range(1, h - ph):
                for c0 in range(1, w - pw):
                    rr0 = max(0, r0 - PLACEMENT_PAD)
                    cc0 = max(0, c0 - PLACEMENT_PAD)
                    rr1 = min(h, r0 + ph + PLACEMENT_PAD)
                    cc1 = min(w, c0 + pw + PLACEMENT_PAD)
                    if occupied[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            paste(grid, patch, r0, c0, transparent=BG)
            positions[idx] = (r0, c0)
            rr0 = max(0, r0 - PLACEMENT_PAD)
            cc0 = max(0, c0 - PLACEMENT_PAD)
            rr1 = min(h, r0 + ph + PLACEMENT_PAD)
            cc1 = min(w, c0 + pw + PLACEMENT_PAD)
            occupied[rr0:rr1, cc0:cc1] = True
        if ok and np.count_nonzero(grid) < (h * w) // 2:
            assert all(pos is not None for pos in positions)
            return grid, [(int(r), int(c)) for r, c in positions]  # type: ignore[arg-type]
    return None


def _expected_output(mask: np.ndarray, missing_regions: list[set[Coord]], colors: list[int]) -> np.ndarray:
    out = np.full(mask.shape, BG, dtype=int)
    for region, color in zip(missing_regions, colors):
        for r, c in region:
            out[r, c] = color
    return out


def _generate_instance() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(GEN_ATTEMPTS):
        mask = _make_template_mask()
        n_fragments = random.choice(FRAGMENT_COUNT_CHOICES)
        missing_regions = _sample_missing_regions(mask, n_fragments)
        if missing_regions is None:
            continue

        colors = random.sample(PALETTE, n_fragments)
        template_cells = {tuple(map(int, coord)) for coord in np.argwhere(mask)}
        fragments = [template_cells - region for region in missing_regions]
        patches = [_fragment_patch(fragment, color) for fragment, color in zip(fragments, colors)]
        placed = _place_patches(patches)
        if placed is None:
            continue
        grid, _positions = placed

        expected = _expected_output(mask, missing_regions, colors)
        solved = main(grid)
        if not np.array_equal(solved, expected):
            continue
        if len({int(v) for v in expected.flatten() if int(v) != BG}) != n_fragments:
            continue
        if np.count_nonzero(expected == BG) < 2:
            continue
        if np.count_nonzero(expected != BG) <= n_fragments:
            continue
        return grid, expected
    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    grid, _expected = _generate_instance()
    return grid
