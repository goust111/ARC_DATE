import random
from typing import Iterable

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
MAROON = 9

CANVAS_SIZE = 30
SAMPLE_OUTER = 5
GAP = 1
GEN_RETRIES = 300
SEED_COLOR_POOL = (1, 2, 3, 4, 6, 7)
INTERIOR_SIZES = (3, 5, 7)

PATTERN_POOL = (
    np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
)


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _partition_index(index: int, size: int) -> int:
    return min(2, (3 * index) // size)


def _partition_cells(size: int, label: int) -> list[int]:
    return [idx for idx in range(size) if _partition_index(idx, size) == label]


def _rectangular_grey_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 3 or width < 3:
            continue
        perimeter = 2 * (height + width) - 4
        if len(comp) != perimeter:
            continue
        points = set(comp)
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border != ((r, c) in points):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            frames.append((r0, c0, r1, c1))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each framed region with the centered pattern whose dictionary marker matches the seed's relative position.

    Input structure:
    - Background is black.
    - Several one-cell-thick grey rectangular frames enclose target regions; each target region contains exactly one colored seed pixel on black.
    - A dictionary in one corner contains 3 to 4 separate grey-framed 3x3 samples; each sample has a maroon pattern and one azure marker pixel.

    Rule:
    - Detect every grey rectangular frame and classify 3x3 frames with one azure marker plus a maroon shape as dictionary samples.
    - For each target region, convert the seed's row and column into a 3x3 relative-position label by partitioning the interior into thirds.
    - Look up the sample whose azure marker has the same relative-position label, clear the whole target interior to black, and center that sample's 3x3 maroon pattern inside the target.
    - Recolor the pasted maroon cells to the seed's color; leave dictionary samples and all grey borders unchanged.

    Key insight:
    - The seed does not indicate where to draw; it is only a lookup key whose coarse 3x3 position selects the dictionary pattern.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    dictionary: dict[tuple[int, int], np.ndarray] = {}
    regions: list[tuple[tuple[int, int, int, int], tuple[int, int], int]] = []

    for r0, c0, r1, c1 in _rectangular_grey_frames(g):
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ih, iw = interior.shape

        if ih == 3 and iw == 3:
            azure_positions = [tuple(map(int, pos)) for pos in np.argwhere(interior == AZURE)]
            maroon_mask = interior == MAROON
            other_colors = set(map(int, np.unique(interior))) - {BLACK, MAROON, AZURE}
            if len(azure_positions) == 1 and np.any(maroon_mask) and not other_colors:
                dictionary[azure_positions[0]] = maroon_mask.copy()
                continue

        seed_positions = [tuple(map(int, pos)) for pos in np.argwhere((interior != BLACK) & (interior != GREY))]
        assert len(seed_positions) == 1, "each non-dictionary frame must contain exactly one seed"
        sr, sc = seed_positions[0]
        seed_color = int(interior[sr, sc])
        regions.append(((r0, c0, r1, c1), (sr, sc), seed_color))

    assert 3 <= len(dictionary) <= 4, "expected 3 or 4 dictionary samples"
    assert regions, "expected at least one target region"

    for (r0, c0, r1, c1), (sr, sc), seed_color in regions:
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        key = (_partition_index(sr, interior_h), _partition_index(sc, interior_w))
        assert key in dictionary, f"missing dictionary entry for relative position {key}"

        out[r0 + 1 : r1, c0 + 1 : c1] = BLACK
        pattern = dictionary[key]
        top = r0 + 1 + (interior_h - 3) // 2
        left = c0 + 1 + (interior_w - 3) // 2
        for dr, dc in np.argwhere(pattern):
            out[top + int(dr), left + int(dc)] = seed_color

    return out


def _draw_frame(grid: np.ndarray, top: int, left: int, interior_h: int, interior_w: int) -> None:
    bottom = top + interior_h + 1
    right = left + interior_w + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY


def _mark_box(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    occupied[top : top + height, left : left + width] = True


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, padding: int = 1) -> bool:
    h, w = occupied.shape
    if top < 0 or left < 0 or top + height > h or left + width > w:
        return False
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + height + padding)
    c1 = min(w, left + width + padding)
    return not np.any(occupied[r0:r1, c0:c1])


def _dictionary_layout(num_samples: int, layout_kind: str) -> tuple[list[tuple[int, int]], int, int]:
    if layout_kind == "row":
        offsets = [(0, idx * (SAMPLE_OUTER + GAP)) for idx in range(num_samples)]
        return offsets, SAMPLE_OUTER, SAMPLE_OUTER * num_samples + GAP * (num_samples - 1)
    if layout_kind == "col":
        offsets = [(idx * (SAMPLE_OUTER + GAP), 0) for idx in range(num_samples)]
        return offsets, SAMPLE_OUTER * num_samples + GAP * (num_samples - 1), SAMPLE_OUTER
    assert num_samples == 4 and layout_kind == "block"
    offsets = [(0, 0), (0, SAMPLE_OUTER + GAP), (SAMPLE_OUTER + GAP, 0), (SAMPLE_OUTER + GAP, SAMPLE_OUTER + GAP)]
    return offsets, 2 * SAMPLE_OUTER + GAP, 2 * SAMPLE_OUTER + GAP


def _corner_anchor(corner: str, block_h: int, block_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, CANVAS_SIZE - block_w
    if corner == "bl":
        return CANVAS_SIZE - block_h, 0
    assert corner == "br"
    return CANVAS_SIZE - block_h, CANVAS_SIZE - block_w


def _choose_dictionary_entries(rng: random.Random, num_samples: int) -> list[dict[str, object]]:
    all_positions = [(r, c) for r in range(3) for c in range(3)]
    for _ in range(200):
        masks = [PATTERN_POOL[idx].copy() for idx in rng.sample(range(len(PATTERN_POOL)), num_samples)]
        order = list(range(num_samples))
        rng.shuffle(order)
        marker_positions: list[tuple[int, int] | None] = [None] * num_samples
        used: set[tuple[int, int]] = set()

        ok = True
        for idx in order:
            available = [pos for pos in all_positions if pos not in used and not bool(masks[idx][pos])]
            if not available:
                ok = False
                break
            marker = rng.choice(available)
            marker_positions[idx] = marker
            used.add(marker)

        if ok:
            return [
                {"marker": marker_positions[idx], "mask": masks[idx]}
                for idx in range(num_samples)
            ]
    raise RuntimeError("failed to assign unique dictionary markers")


def _choose_region_sizes(rng: random.Random, count: int) -> list[tuple[int, int]]:
    large_choices = [(h, w) for h in INTERIOR_SIZES for w in INTERIOR_SIZES if (h, w) != (3, 3)]
    sizes = [(3, 3), rng.choice(large_choices)]
    while len(sizes) < count:
        if rng.random() < 0.5:
            sizes.append((3, 3))
        else:
            sizes.append(rng.choice(large_choices))
    rng.shuffle(sizes)
    return sizes


def _place_regions(
    rng: random.Random,
    occupied: np.ndarray,
    specs: list[dict[str, object]],
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(
        range(len(specs)),
        key=lambda idx: int(specs[idx]["interior_h"]) * int(specs[idx]["interior_w"]),
        reverse=True,
    )

    for idx in order:
        interior_h = int(specs[idx]["interior_h"])
        interior_w = int(specs[idx]["interior_w"])
        box_h = interior_h + 2
        box_w = interior_w + 2

        candidates = [(r, c) for r in range(CANVAS_SIZE - box_h + 1) for c in range(CANVAS_SIZE - box_w + 1)]
        rng.shuffle(candidates)
        chosen = None
        for top, left in candidates:
            if _can_place(occupied, top, left, box_h, box_w, padding=1):
                chosen = (top, left)
                break

        if chosen is None:
            return None

        placements[idx] = chosen
        _mark_box(occupied, chosen[0], chosen[1], box_h, box_w)

    return [(int(top), int(left)) for top, left in placements if top is not None and left is not None]


def _crop_to_corner(grid: np.ndarray, corner: str, rng: random.Random) -> np.ndarray:
    used = np.argwhere(grid != BLACK)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)

    if corner[0] == "t":
        top = 0
        bottom = min(CANVAS_SIZE, int(r1) + 1 + rng.randint(0, 2))
    else:
        bottom = CANVAS_SIZE
        top = max(0, int(r0) - rng.randint(0, 2))

    if corner[1] == "l":
        left = 0
        right = min(CANVAS_SIZE, int(c1) + 1 + rng.randint(0, 2))
    else:
        right = CANVAS_SIZE
        left = max(0, int(c0) - rng.randint(0, 2))

    return grid[top:bottom, left:right].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.full((CANVAS_SIZE, CANVAS_SIZE), BLACK, dtype=int)
    occupied = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=bool)

    num_samples = rng.choice((3, 4))
    layout_choices = ["row", "col"]
    if num_samples == 4:
        layout_choices.append("block")
    layout_kind = rng.choice(layout_choices)
    corner = rng.choice(("tl", "tr", "bl", "br"))

    entries = _choose_dictionary_entries(rng, num_samples)
    offsets, dict_h, dict_w = _dictionary_layout(num_samples, layout_kind)
    top0, left0 = _corner_anchor(corner, dict_h, dict_w)

    if layout_kind == "block":
        rng.shuffle(offsets)

    for entry, (dr, dc) in zip(entries, offsets):
        top = top0 + dr
        left = left0 + dc
        _draw_frame(grid, top, left, 3, 3)
        interior = grid[top + 1 : top + 4, left + 1 : left + 4]
        interior[:] = BLACK
        interior[np.asarray(entry["mask"], dtype=bool)] = MAROON
        marker_r, marker_c = entry["marker"]
        interior[int(marker_r), int(marker_c)] = AZURE
        _mark_box(occupied, top, left, SAMPLE_OUTER, SAMPLE_OUTER)

    num_regions = num_samples + rng.randint(0, 1)
    sizes = _choose_region_sizes(rng, num_regions)
    region_markers = [entry["marker"] for entry in entries]
    while len(region_markers) < num_regions:
        region_markers.append(rng.choice(region_markers))
    rng.shuffle(region_markers)
    seed_colors = rng.sample(SEED_COLOR_POOL, num_regions)

    region_specs: list[dict[str, object]] = []
    for idx in range(num_regions):
        interior_h, interior_w = sizes[idx]
        part_r, part_c = region_markers[idx]
        seed_r = rng.choice(_partition_cells(interior_h, int(part_r)))
        seed_c = rng.choice(_partition_cells(interior_w, int(part_c)))
        region_specs.append(
            {
                "interior_h": interior_h,
                "interior_w": interior_w,
                "seed_pos": (seed_r, seed_c),
                "seed_color": seed_colors[idx],
            }
        )

    placements = _place_regions(rng, occupied, region_specs)
    if placements is None or len(placements) != len(region_specs):
        return None

    for spec, (top, left) in zip(region_specs, placements):
        interior_h = int(spec["interior_h"])
        interior_w = int(spec["interior_w"])
        _draw_frame(grid, top, left, interior_h, interior_w)
        interior = grid[top + 1 : top + 1 + interior_h, left + 1 : left + 1 + interior_w]
        interior[:] = BLACK
        seed_r, seed_c = spec["seed_pos"]
        interior[int(seed_r), int(seed_c)] = int(spec["seed_color"])

    cropped = _crop_to_corner(grid, corner, rng)

    try:
        out = main(cropped)
    except Exception:
        return None

    if np.array_equal(cropped, out):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
