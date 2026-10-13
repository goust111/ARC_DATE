from __future__ import annotations

from collections import defaultdict
from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import Color, bbox, components_mask, crop_bbox, paste
except Exception:
    _ARC_UTILS = SourceFileLoader(
        "_arc_utils_fable_1_056_2",
        "/home/lixinyu/snap/new/solvers/arc_utils.py",
    ).load_module()
    Color = _ARC_UTILS.Color
    bbox = _ARC_UTILS.bbox
    components_mask = _ARC_UTILS.components_mask
    crop_bbox = _ARC_UTILS.crop_bbox
    paste = _ARC_UTILS.paste


Coord = tuple[int, int]

BG = Color.BLACK
ANCHOR_COLOR = Color.GREY
BODY_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.ORANGE,
    Color.TEAL,
    Color.MAROON,
]

DIRS: list[Coord] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LEFT_TURN: dict[Coord, Coord] = {
    (-1, 0): (0, -1),
    (1, 0): (0, 1),
    (0, -1): (1, 0),
    (0, 1): (-1, 0),
}
RIGHT_TURN: dict[Coord, Coord] = {
    (-1, 0): (0, 1),
    (1, 0): (0, -1),
    (0, -1): (-1, 0),
    (0, 1): (1, 0),
}

MAX_SHAPE_EXTENT = 4
MIN_SCENE_SIDE = 14
MAX_SCENE_SIDE = 24
MAX_GENERATION_ATTEMPTS = 500
MAX_PLACEMENT_ATTEMPTS = 200


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIRS]


def _count_occupied_neighbors(cell: Coord, occupied: set[Coord]) -> int:
    return sum(nei in occupied for nei in _neighbors4(cell))


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        cell = stack.pop()
        for nei in _neighbors4(cell):
            if nei in cells and nei not in seen:
                seen.add(nei)
                stack.append(nei)
    return seen == cells


def _extract_copies(grid: np.ndarray) -> list[dict]:
    comps = components_mask(grid != BG, connectivity=4)
    assert len(comps) == 3, "expected exactly three incomplete copies"

    copies: list[dict] = []
    for comp in comps:
        comp_mask = np.zeros_like(grid, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        comp_bbox = bbox(comp_mask)
        assert comp_bbox is not None
        patch = crop_bbox(grid, comp_bbox)

        anchor_coords = np.argwhere(patch == ANCHOR_COLOR)
        assert anchor_coords.shape == (1, 2), "each copy must contain exactly one grey anchor"
        anchor_r, anchor_c = map(int, anchor_coords[0])

        body_colors = sorted(set(map(int, patch[patch != BG].flatten())) - {ANCHOR_COLOR})
        assert len(body_colors) == 1, "each copy must have exactly one body color"

        offsets = {
            (int(r - anchor_r), int(c - anchor_c))
            for r, c in np.argwhere(patch != BG)
        }
        copies.append(
            {
                "color": int(body_colors[0]),
                "offsets": offsets,
                "anchor": (int(comp_bbox[0] + anchor_r), int(comp_bbox[1] + anchor_c)),
            }
        )

    colors = [copy["color"] for copy in copies]
    assert len(set(colors)) == 3, "the three incomplete copies must use distinct body colors"
    return copies


def _aligned_union(copies: list[dict]) -> np.ndarray:
    coverage: defaultdict[Coord, set[int]] = defaultdict(set)
    for copy in copies:
        for cell in copy["offsets"]:
            coverage[cell].add(int(copy["color"]))

    rows = [r for r, _ in coverage]
    cols = [c for _, c in coverage]
    min_r = min(rows)
    min_c = min(cols)
    out = np.full((max(rows) - min_r + 1, max(cols) - min_c + 1), BG, dtype=int)

    for (r, c), colors in coverage.items():
        out[r - min_r, c - min_c] = (
            ANCHOR_COLOR if len(colors) >= 2 else next(iter(colors))
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align three anchor-marked incomplete copies to reconstruct the hidden shape, coloring overlaps grey.

    Input structure:
    - The background is black.
    - There are exactly three separated 4-connected objects, each mostly one body color with one grey anchor pixel inside it.
    - The three objects are incomplete copies of one hidden shape: after translation-only alignment by the grey anchors, their visible pixels become subsets of one common mask.

    Rule:
    - Treat each grey anchor as the same origin and convert every visible pixel of its object into anchor-relative coordinates.
    - Take the union of those aligned coordinates and crop tightly to that reconstructed shape's bounding box.
    - A reconstructed pixel covered by exactly one copy keeps that copy's body color; a pixel covered by two or three copies becomes grey.

    Key insight:
    - The visible objects are never complete enough to align from outline alone; the grey anchor is the only reliable shared coordinate reference, and the output color reports coverage count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _aligned_union(_extract_copies(g))


def _grow_hidden_shape() -> tuple[set[Coord], dict[Coord, Coord], list[list[Coord]], list[int], dict[Coord, int]] | None:
    branch_count = random.choice([3, 3, 4])
    branch_dirs = random.sample(DIRS, branch_count)

    occupied: set[Coord] = {(0, 0)}
    parent: dict[Coord, Coord] = {}
    owner_of: dict[Coord, int] = {(0, 0): -1}
    branches: list[list[Coord]] = []
    owners: list[int] = []

    for idx, start_dir in enumerate(branch_dirs):
        owner = idx if idx < 3 else random.randrange(3)
        current = (0, 0)
        prev_dir = start_dir
        target_len = random.randint(2, 5)
        path: list[Coord] = []

        for step in range(target_len):
            options: list[Coord] = []
            seen: set[Coord] = set()
            for move in [prev_dir, LEFT_TURN[prev_dir], RIGHT_TURN[prev_dir]] + DIRS:
                if move in seen:
                    continue
                seen.add(move)
                nxt = (current[0] + move[0], current[1] + move[1])
                if max(abs(nxt[0]), abs(nxt[1])) > MAX_SHAPE_EXTENT:
                    continue
                if nxt in occupied:
                    continue
                if _count_occupied_neighbors(nxt, occupied) != 1:
                    continue
                options.append(move)

            if not options:
                if step < 2:
                    return None
                break

            weights = [
                6 if move == prev_dir else 3 if move in (LEFT_TURN[prev_dir], RIGHT_TURN[prev_dir]) else 1
                for move in options
            ]
            move = random.choices(options, weights=weights)[0]
            nxt = (current[0] + move[0], current[1] + move[1])
            occupied.add(nxt)
            parent[nxt] = current
            owner_of[nxt] = owner
            path.append(nxt)
            current = nxt
            prev_dir = move

        if len(path) < 2:
            return None
        branches.append(path)
        owners.append(owner)

    branch_cells = [cell for path in branches for cell in path]
    for _ in range(random.randint(0, 2)):
        random.shuffle(branch_cells)
        for base in branch_cells:
            options: list[Coord] = []
            for dr, dc in DIRS:
                nxt = (base[0] + dr, base[1] + dc)
                if max(abs(nxt[0]), abs(nxt[1])) > MAX_SHAPE_EXTENT:
                    continue
                if nxt in occupied:
                    continue
                if _count_occupied_neighbors(nxt, occupied) != 1:
                    continue
                options.append(nxt)
            if options:
                nxt = random.choice(options)
                occupied.add(nxt)
                parent[nxt] = base
                owner_of[nxt] = owner_of[base]
                branch_cells.append(nxt)
                break

    return occupied, parent, branches, owners, owner_of


def _assign_coverages(
    occupied: set[Coord],
    parent: dict[Coord, Coord],
    branches: list[list[Coord]],
    owners: list[int],
    owner_of: dict[Coord, int],
) -> dict[Coord, frozenset[int]] | None:
    coverage: dict[Coord, frozenset[int]] = {(0, 0): frozenset({0, 1, 2})}

    recipes = ["all_pair_unique", "pair_unique", "all_unique"]
    random.shuffle(recipes)
    while len(recipes) < len(branches):
        recipes.append(random.choice(["all_pair_unique", "pair_unique", "all_unique"]))

    for path, owner, recipe in zip(branches, owners, recipes):
        if recipe == "all_pair_unique" and len(path) < 3:
            recipe = "pair_unique"
        partner = random.choice([idx for idx in range(3) if idx != owner])

        for step, cell in enumerate(path):
            if recipe == "all_pair_unique":
                cov = {0, 1, 2} if step == 0 else {owner, partner} if step == 1 else {owner}
            elif recipe == "pair_unique":
                cov = {owner, partner} if step == 0 else {owner}
            else:
                cov = {0, 1, 2} if step == 0 else {owner}
            coverage[cell] = frozenset(cov)

    pending = [cell for cell in occupied if cell not in coverage]
    while pending:
        progressed = False
        for cell in pending[:]:
            par = parent[cell]
            if par not in coverage:
                continue

            parent_cov = sorted(coverage[par])
            owner = owner_of[cell]
            if owner not in parent_cov:
                return None

            if len(parent_cov) == 1:
                cov = set(parent_cov)
            else:
                options: list[set[int]] = []
                for mask in range(1, 1 << len(parent_cov)):
                    subset = {
                        parent_cov[i]
                        for i in range(len(parent_cov))
                        if (mask >> i) & 1
                    }
                    if owner in subset:
                        options.append(subset)
                weights = [
                    5 if len(subset) == len(parent_cov) else 3 if len(subset) == 2 else 2
                    for subset in options
                ]
                cov = random.choices(options, weights=weights)[0]

            coverage[cell] = frozenset(cov)
            pending.remove(cell)
            progressed = True

        if not progressed:
            return None

    return coverage


def _render_copy_patch(visible_cells: set[Coord], color: int) -> tuple[np.ndarray, Coord]:
    rows = [r for r, _ in visible_cells]
    cols = [c for _, c in visible_cells]
    min_r = min(rows)
    min_c = min(cols)
    patch = np.full((max(rows) - min_r + 1, max(cols) - min_c + 1), BG, dtype=int)

    for r, c in visible_cells:
        patch[r - min_r, c - min_c] = int(color)

    anchor_local = (-min_r, -min_c)
    patch[anchor_local] = ANCHOR_COLOR
    return patch, anchor_local


def _render_expected(coverage: dict[Coord, frozenset[int]], body_colors: list[int]) -> np.ndarray:
    rows = [r for r, _ in coverage]
    cols = [c for _, c in coverage]
    min_r = min(rows)
    min_c = min(cols)
    out = np.full((max(rows) - min_r + 1, max(cols) - min_c + 1), BG, dtype=int)

    for (r, c), owners in coverage.items():
        out[r - min_r, c - min_c] = (
            ANCHOR_COLOR if len(owners) >= 2 else int(body_colors[next(iter(owners))])
        )
    return out


def _place_patches(patches: list[np.ndarray], anchors: list[Coord]) -> np.ndarray | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    min_h = max(MIN_SCENE_SIDE, max_h + 6)
    min_w = max(MIN_SCENE_SIDE, max_w + 6)

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        height = random.randint(min_h, MAX_SCENE_SIDE)
        width = random.randint(min_w, MAX_SCENE_SIDE)

        positions: list[Coord | None] = [None, None, None]
        placed: list[tuple[int, int, int, int]] = []
        used_anchor_rows: set[int] = set()
        used_anchor_cols: set[int] = set()
        order = sorted(
            range(3),
            key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1],
            reverse=True,
        )

        for idx in order:
            patch = patches[idx]
            patch_h, patch_w = patch.shape
            anchor_r, anchor_c = anchors[idx]
            candidates: list[Coord] = []

            for r0 in range(1, height - patch_h):
                for c0 in range(1, width - patch_w):
                    if (r0 + anchor_r) in used_anchor_rows or (c0 + anchor_c) in used_anchor_cols:
                        continue
                    clash = any(
                        not (
                            r0 + patch_h <= prev_r - 1
                            or prev_r + prev_h <= r0 - 1
                            or c0 + patch_w <= prev_c - 1
                            or prev_c + prev_w <= c0 - 1
                        )
                        for prev_r, prev_c, prev_h, prev_w in placed
                    )
                    if not clash:
                        candidates.append((r0, c0))

            if not candidates:
                break

            r0, c0 = random.choice(candidates)
            positions[idx] = (r0, c0)
            placed.append((r0, c0, patch_h, patch_w))
            used_anchor_rows.add(r0 + anchor_r)
            used_anchor_cols.add(c0 + anchor_c)
        else:
            scene = np.full((height, width), BG, dtype=int)
            for patch, pos in zip(patches, positions):
                assert pos is not None
                paste(scene, patch, pos[0], pos[1], transparent=BG)
            return scene

    return None


def _attempt_generation() -> np.ndarray | None:
    shape = _grow_hidden_shape()
    if shape is None:
        return None
    occupied, parent, branches, owners, owner_of = shape

    coverage = _assign_coverages(occupied, parent, branches, owners, owner_of)
    if coverage is None:
        return None

    full_cells = set(coverage)
    if len(full_cells) < 8:
        return None

    copy_cells = [
        {cell for cell, covered_by in coverage.items() if idx in covered_by}
        for idx in range(3)
    ]
    if not all(_is_connected(cells) for cells in copy_cells):
        return None
    if not all(len(cells) < len(full_cells) for cells in copy_cells):
        return None
    if not all(
        sum(1 for covered_by in coverage.values() if covered_by == {idx}) > 0
        for idx in range(3)
    ):
        return None
    if sum(
        1 for cell, covered_by in coverage.items() if cell != (0, 0) and len(covered_by) >= 2
    ) == 0:
        return None

    body_colors = random.sample(BODY_COLORS, 3)
    patches: list[np.ndarray] = []
    anchors: list[Coord] = []
    extents: list[tuple[int, int, int, int]] = []

    for idx, cells in enumerate(copy_cells):
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        extents.append((min(rows), max(rows), min(cols), max(cols)))
        patch, anchor_local = _render_copy_patch(cells, body_colors[idx])
        patches.append(patch)
        anchors.append(anchor_local)

    if len(set(extents)) == 1:
        return None
    if len(set(anchors)) == 1:
        return None

    expected = _render_expected(coverage, body_colors)
    expected_colors = set(map(int, np.unique(expected)))
    if ANCHOR_COLOR not in expected_colors or not set(body_colors).issubset(expected_colors):
        return None

    scene = _place_patches(patches, anchors)
    if scene is None:
        return None

    copies = _extract_copies(scene)
    if len({copy["anchor"][0] for copy in copies}) != 3:
        return None
    if len({copy["anchor"][1] for copy in copies}) != 3:
        return None

    actual = main(scene)
    if not np.array_equal(actual, expected):
        return None
    return scene


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generation()
        if grid is not None:
            return grid
    raise RuntimeError(
        f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts for fable/1/056/2"
    )
