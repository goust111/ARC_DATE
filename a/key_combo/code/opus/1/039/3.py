from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3

MARKER_COUNTS = (3, 4)
HEIGHT_RANGE = (10, 16)
HALF_WIDTH_RANGE = (6, 10)
OVERLAP_BRANCH_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 500

# (dest_row_offset, dest_col_offset, outward_row_step, outward_col_step)
RING_SPECS = [
    (-1, -1, -1, -1),
    (-1, 0, -1, 0),
    (-1, 1, -1, 0),
    (-1, 2, -1, 1),
    (0, -1, 0, -1),
    (1, -1, 0, -1),
    (0, 2, 0, 1),
    (1, 2, 0, 1),
    (2, -1, 1, -1),
    (2, 0, 1, 0),
    (2, 1, 1, 0),
    (2, 2, 1, 1),
]
OVERLAP_OFFSETS = [
    (0, 0),
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0),
    (1, 1),
    (-1, -1),
]


def _separator_col(grid: np.ndarray) -> int:
    separator_cols = np.flatnonzero(np.all(grid == RED, axis=0))
    assert len(separator_cols) == 1, "expected one full-height red separator column"
    col = int(separator_cols[0])
    assert 0 < col < grid.shape[1] - 1, "separator must split the grid into two nonempty halves"
    return col


def _find_block_bbox(half: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(half == BLUE)
    assert coords.shape[0] == 4, "expected exactly four blue block pixels in each half"
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "blue pixels must form a 2x2 bbox"
    assert np.all(half[r0 : r1 + 1, c0 : c1 + 1] == BLUE), "blue block bbox must be filled"
    return r0, c0, r1, c1


def _ring_projection(r: int, c: int, block_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = block_bbox

    if r < r0:
        nr = r0 - 1
    elif r > r1:
        nr = r1 + 1
    else:
        nr = r

    if c < c0:
        nc = c0 - 1
    elif c > c1:
        nc = c1 + 1
    else:
        nc = c

    return int(nr), int(nc)


def _validate_marker_alignment(
    r: int,
    c: int,
    block_bbox: tuple[int, int, int, int],
    dest: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = block_bbox
    nr, nc = dest
    row_aligned = r0 <= r <= r1
    col_aligned = c0 <= c <= c1

    assert not (r0 <= r <= r1 and c0 <= c <= c1), "marker cannot lie inside the block"
    if row_aligned:
        assert c < c0 or c > c1, "row-aligned marker must lie outside the block columns"
    if col_aligned:
        assert r < r0 or r > r1, "column-aligned marker must lie outside the block rows"
    if not row_aligned and not col_aligned:
        assert abs(r - nr) == abs(c - nc), "off-axis marker must lie on a 45-degree ray"
    assert row_aligned or col_aligned or abs(r - nr) == abs(c - nc), "marker must be row, column, or diagonal aligned"


def _slide_half(half: np.ndarray) -> np.ndarray:
    g = np.asarray(half, dtype=int)
    assert g.ndim == 2, "expected a 2D half-grid"
    assert set(np.unique(g)) <= {BG, BLUE, GREEN}, "unexpected colors inside a half"

    block_bbox = _find_block_bbox(g)
    markers = np.argwhere(g == GREEN)
    assert len(markers) in MARKER_COUNTS, "expected 3 or 4 markers in each half"

    out = g.copy()
    out[g == GREEN] = BG

    height, width = map(int, g.shape)
    destinations: set[tuple[int, int]] = set()
    for rr, cc in markers:
        r = int(rr)
        c = int(cc)
        dest = _ring_projection(r, c, block_bbox)
        _validate_marker_alignment(r, c, block_bbox, dest)

        nr, nc = dest
        assert 0 <= nr < height and 0 <= nc < width, "marker destination must stay inside the half"
        assert (nr, nc) not in destinations, "slid markers must land on distinct ring cells"
        destinations.add((nr, nc))
        out[nr, nc] = GREEN

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each half's ray-aligned markers inward, then OR-merge the slid halves into one blue mask.

    Input structure:
    - The background is black.
    - One full-height red column splits the grid into equal left and right halves.
    - Each half contains one filled blue 2x2 block and 3-4 green marker pixels, and each marker lies on a horizontal, vertical, or 45-degree diagonal ray extending away from that half's block.

    Rule:
    - In each half independently, move every green marker along its current ray toward the blue block until it reaches the matching cell in the one-cell ring around the 2x2 block.
    - Keep the blue 2x2 blocks fixed and clear the original marker cells.
    - Remove the red separator column.
    - For each aligned position of the two slid halves, output blue if either half is non-background there; otherwise output black.

    Key insight:
    - The visible merge is applied after the markers slide, so the answer is the union of two transformed half-scenes rather than the union of the original masks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(np.unique(g)) <= {BG, BLUE, RED, GREEN}, "unexpected input colors"

    sep = _separator_col(g)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape and left.size > 0, "divider must separate two equal nonempty halves"
    assert np.all(left[:, -1] == BG), "left-half input objects must not touch the divider"
    assert np.all(right[:, 0] == BG), "right-half input objects must not touch the divider"

    left_slid = _slide_half(left)
    right_slid = _slide_half(right)

    out = np.full(left.shape, BG, dtype=int)
    out[(left_slid != BG) | (right_slid != BG)] = BLUE
    return out


def _block_cells(pos: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = pos
    return {
        (r0, c0),
        (r0, c0 + 1),
        (r0 + 1, c0),
        (r0 + 1, c0 + 1),
    }


def _blocks_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not _block_cells(a).isdisjoint(_block_cells(b))


def _available_rays(
    height: int,
    width: int,
    side: str,
    block_pos: tuple[int, int],
) -> list[dict]:
    forbidden_col = width - 1 if side == "left" else 0
    r0, c0 = block_pos
    rays = []

    for ro, co, dr, dc in RING_SPECS:
        dest_r = r0 + ro
        dest_c = c0 + co
        if not (0 <= dest_r < height and 0 <= dest_c < width):
            continue

        max_dist = 0
        dist = 1
        while True:
            src_r = dest_r + dr * dist
            src_c = dest_c + dc * dist
            if not (0 <= src_r < height and 0 <= src_c < width):
                break
            if src_c == forbidden_col:
                break
            max_dist = dist
            dist += 1

        if max_dist >= 1:
            rays.append(
                {
                    "dest": (int(dest_r), int(dest_c)),
                    "dir": (int(dr), int(dc)),
                    "max_dist": int(max_dist),
                    "kind": "diag" if dr != 0 and dc != 0 else "orth",
                    "spec": (int(ro), int(co), int(dr), int(dc)),
                }
            )

    return rays


def _candidate_catalog(height: int, width: int, side: str) -> dict[tuple[int, int], list[dict]]:
    catalog: dict[tuple[int, int], list[dict]] = {}
    for r0 in range(1, height - 2):
        for c0 in range(1, width - 2):
            rays = _available_rays(height, width, side, (r0, c0))
            if len(rays) < 4:
                continue
            kinds = {ray["kind"] for ray in rays}
            if kinds != {"diag", "orth"}:
                continue
            catalog[(r0, c0)] = rays
    return catalog


def _choose_required_ray(pool: list[dict], chosen: list[dict], kind: str) -> bool:
    used_specs = {ray["spec"] for ray in chosen}
    options = [ray for ray in pool if ray["kind"] == kind and ray["spec"] not in used_specs]
    if not options:
        return False
    chosen.append(random.choice(options))
    return True


def _sample_half(
    height: int,
    width: int,
    side: str,
    block_pos: tuple[int, int],
    rays: list[dict],
    force_diag: bool,
    force_orth: bool,
) -> dict | None:
    marker_count = random.choice(MARKER_COUNTS)
    chosen: list[dict] = []
    if force_diag and not _choose_required_ray(rays, chosen, "diag"):
        return None
    if force_orth and not _choose_required_ray(rays, chosen, "orth"):
        return None

    remaining = [ray for ray in rays if ray["spec"] not in {picked["spec"] for picked in chosen}]
    need = marker_count - len(chosen)
    if len(remaining) < need:
        return None
    chosen.extend(random.sample(remaining, need))
    random.shuffle(chosen)

    long_candidates = [ray for ray in chosen if ray["max_dist"] >= 2]
    guaranteed_long = random.choice(long_candidates)["spec"] if long_candidates and random.random() < 0.8 else None

    grid = np.full((height, width), BG, dtype=int)
    final_mask = np.zeros((height, width), dtype=bool)
    block_mask = np.zeros((height, width), dtype=bool)
    r0, c0 = block_pos
    grid[r0 : r0 + 2, c0 : c0 + 2] = BLUE
    final_mask[r0 : r0 + 2, c0 : c0 + 2] = True
    block_mask[r0 : r0 + 2, c0 : c0 + 2] = True

    source_positions = []
    destination_positions = []
    distances = []
    ray_kinds = []
    used_sources: set[tuple[int, int]] = set()
    for ray in chosen:
        dest_r, dest_c = ray["dest"]
        dr, dc = ray["dir"]
        min_dist = 2 if ray["spec"] == guaranteed_long else 1
        dist = random.randint(min_dist, ray["max_dist"])
        src_r = dest_r + dr * dist
        src_c = dest_c + dc * dist
        assert 0 <= src_r < height and 0 <= src_c < width
        assert grid[src_r, src_c] == BG and (src_r, src_c) not in used_sources

        grid[src_r, src_c] = GREEN
        final_mask[dest_r, dest_c] = True
        used_sources.add((src_r, src_c))
        source_positions.append((int(src_r), int(src_c)))
        destination_positions.append((int(dest_r), int(dest_c)))
        distances.append(int(dist))
        ray_kinds.append(ray["kind"])

    return {
        "grid": grid,
        "block_pos": block_pos,
        "block_mask": block_mask,
        "final_mask": final_mask,
        "source_positions": tuple(source_positions),
        "destination_positions": tuple(destination_positions),
        "distances": tuple(distances),
        "ray_kinds": tuple(ray_kinds),
        "side": side,
    }


def _compose_input(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    separator = np.full((left.shape[0], 1), RED, dtype=int)
    return np.concatenate([left, separator, right], axis=1)


def _expected_output(left_meta: dict, right_meta: dict) -> np.ndarray:
    out = np.full(left_meta["grid"].shape, BG, dtype=int)
    out[left_meta["final_mask"] | right_meta["final_mask"]] = BLUE
    return out


def _validate_half_structure(half: np.ndarray, side: str) -> None:
    assert set(np.unique(half)) <= {BG, BLUE, GREEN}
    forbidden_col = half.shape[1] - 1 if side == "left" else 0
    assert np.all(half[:, forbidden_col] == BG), "input objects must not touch the divider"

    block_bbox = _find_block_bbox(half)
    destinations: set[tuple[int, int]] = set()
    for rr, cc in np.argwhere(half == GREEN):
        r = int(rr)
        c = int(cc)
        dest = _ring_projection(r, c, block_bbox)
        _validate_marker_alignment(r, c, block_bbox, dest)
        assert dest not in destinations, "projected ring cells must be unique"
        destinations.add(dest)

    assert len(destinations) in MARKER_COUNTS, "each half must have 3 or 4 markers"


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    left_meta: dict,
    right_meta: dict,
    overlap_branch: bool,
) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    height, half_width = expected.shape
    assert inp.shape == (height, 2 * half_width + 1)
    assert set(np.unique(inp)) <= {BG, BLUE, RED, GREEN}
    assert set(np.unique(expected)) <= {BG, BLUE}

    sep = _separator_col(inp)
    assert sep == half_width, "separator should split the input into two equal halves"
    left = inp[:, :sep]
    right = inp[:, sep + 1 :]
    _validate_half_structure(left, "left")
    _validate_half_structure(right, "right")

    naive = np.full_like(expected, BG)
    naive[(left != BG) | (right != BG)] = BLUE
    assert int(np.count_nonzero(naive != expected)) >= 2, "sliding step must visibly change the merged mask"

    blocks_only = np.full_like(expected, BG)
    blocks_only[left_meta["block_mask"] | right_meta["block_mask"]] = BLUE
    assert not np.array_equal(blocks_only, expected), "markers must affect the output"

    overlap = left_meta["final_mask"] & right_meta["final_mask"]
    assert bool(np.any(overlap)) == overlap_branch, "overlap branch coverage mismatch"
    assert np.any(left_meta["final_mask"] & ~right_meta["final_mask"]), "left half must contribute unique output cells"
    assert np.any(right_meta["final_mask"] & ~left_meta["final_mask"]), "right half must contribute unique output cells"

    all_kinds = set(left_meta["ray_kinds"]) | set(right_meta["ray_kinds"])
    assert all_kinds == {"diag", "orth"}, "samples must cover both diagonal and orthogonal rays"
    all_distances = left_meta["distances"] + right_meta["distances"]
    assert any(dist >= 2 for dist in all_distances), "at least one marker should slide more than one step"

    assert np.array_equal(main(inp), expected), "solver output must match the independently built target"


def _sample_positions(
    height: int,
    width: int,
) -> tuple[bool, tuple[int, int], list[dict], tuple[int, int], list[dict]] | None:
    left_catalog = _candidate_catalog(height, width, "left")
    right_catalog = _candidate_catalog(height, width, "right")
    if not left_catalog or not right_catalog:
        return None

    overlap_branch = random.random() < OVERLAP_BRANCH_PROB
    left_pos = random.choice(list(left_catalog))

    if overlap_branch:
        options = []
        for dr, dc in OVERLAP_OFFSETS:
            pos = (left_pos[0] + dr, left_pos[1] + dc)
            if pos in right_catalog and _blocks_overlap(left_pos, pos):
                options.append(pos)
        if not options:
            return None
        right_pos = random.choice(options)
    else:
        options = [
            pos
            for pos in right_catalog
            if pos != left_pos and not _blocks_overlap(left_pos, pos)
        ]
        if not options:
            return None
        right_pos = random.choice(options)

    return overlap_branch, left_pos, left_catalog[left_pos], right_pos, right_catalog[right_pos]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    half_width = random.randint(*HALF_WIDTH_RANGE)

    sampled = _sample_positions(height, half_width)
    if sampled is None:
        return None
    overlap_branch, left_pos, left_rays, right_pos, right_rays = sampled

    diagonal_host = random.choice(("left", "right"))
    left_meta = _sample_half(
        height,
        half_width,
        "left",
        left_pos,
        left_rays,
        force_diag=diagonal_host == "left" or random.random() < 0.35,
        force_orth=True,
    )
    if left_meta is None:
        return None
    right_meta = _sample_half(
        height,
        half_width,
        "right",
        right_pos,
        right_rays,
        force_diag=diagonal_host == "right" or random.random() < 0.35,
        force_orth=True,
    )
    if right_meta is None:
        return None

    inp = _compose_input(left_meta["grid"], right_meta["grid"])
    expected = _expected_output(left_meta, right_meta)

    try:
        _validate_instance(inp, expected, left_meta, right_meta, overlap_branch)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid task-3 input")
