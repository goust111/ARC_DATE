from __future__ import annotations

import itertools
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _rectangles_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _find_source_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []
    for coords in _components4(grid != bg):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(coords.shape[0])
        if area == h * w and h >= 4 and w >= 4:
            candidates.append((area, r0, c0, r1, c1))

    assert candidates, "expected a full rectangular source pattern"
    max_area = max(area for area, *_ in candidates)
    sources = [item for item in candidates if item[0] == max_area]
    assert len(sources) == 1, "source pattern must be uniquely identifiable"
    _, r0, c0, r1, c1 = sources[0]
    return r0, c0, r1, c1


def _transform_variants(source: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "flip_h": np.fliplr(source),
        "flip_v": np.flipud(source),
        "rot180": np.rot90(source, 2),
    }


def _collect_clues(
    grid: np.ndarray, bg: int, source_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = source_bbox
    clues: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(grid != bg):
        rr = int(r)
        cc = int(c)
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            continue
        clues.append((rr, cc, int(grid[rr, cc])))
    return clues


def _candidate_targets(
    grid: np.ndarray,
    clues: list[tuple[int, int, int]],
    source_bbox: tuple[int, int, int, int],
    variants: dict[str, np.ndarray],
) -> list[dict]:
    h, w = next(iter(variants.values())).shape
    grid_h, grid_w = grid.shape
    source_rect = source_bbox
    top_lefts: set[tuple[int, int]] = set()

    for r, c, _ in clues:
        for rr in range(h):
            r0 = r - rr
            if r0 < 0 or r0 + h > grid_h:
                continue
            for cc in range(w):
                c0 = c - cc
                if c0 < 0 or c0 + w > grid_w:
                    continue
                top_lefts.add((r0, c0))

    candidates: list[dict] = []
    for r0, c0 in top_lefts:
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _rectangles_overlap(rect, source_rect):
            continue

        member_ids = [
            idx
            for idx, (r, c, _) in enumerate(clues)
            if r0 <= r < r0 + h and c0 <= c < c0 + w
        ]
        if len(member_ids) not in (2, 3):
            continue

        rel_positions = [(clues[idx][0] - r0, clues[idx][1] - c0) for idx in member_ids]
        rows = [r for r, _ in rel_positions]
        cols = [c for _, c in rel_positions]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue

        matches: list[str] = []
        for name, patch in variants.items():
            if all(
                int(patch[rr, cc]) == clues[idx][2]
                for idx, (rr, cc) in zip(member_ids, rel_positions)
            ):
                matches.append(name)
        if len(matches) != 1:
            continue

        member_mask = 0
        for idx in member_ids:
            member_mask |= 1 << idx
        candidates.append(
            {
                "r0": r0,
                "c0": c0,
                "rect": rect,
                "members": member_mask,
                "transform": matches[0],
            }
        )

    return candidates


def _unique_cover(candidates: list[dict], clue_count: int) -> list[dict]:
    by_clue: list[list[int]] = [[] for _ in range(clue_count)]
    for idx, candidate in enumerate(candidates):
        for clue_id in range(clue_count):
            if (candidate["members"] >> clue_id) & 1:
                by_clue[clue_id].append(idx)

    assert all(options for options in by_clue), "every clue must belong to at least one target"
    solutions: list[tuple[int, ...]] = []

    def search(remaining: int, chosen: list[int]) -> None:
        if len(solutions) > 1:
            return
        if remaining == 0:
            solutions.append(tuple(chosen))
            return

        uncovered = [idx for idx in range(clue_count) if (remaining >> idx) & 1]
        pivot = min(
            uncovered,
            key=lambda clue_id: sum(
                1
                for cand_idx in by_clue[clue_id]
                if candidates[cand_idx]["members"] & remaining == candidates[cand_idx]["members"]
                and all(
                    not _rectangles_overlap(
                        candidates[cand_idx]["rect"], candidates[other]["rect"]
                    )
                    for other in chosen
                )
            ),
        )

        options = [
            cand_idx
            for cand_idx in by_clue[pivot]
            if candidates[cand_idx]["members"] & remaining == candidates[cand_idx]["members"]
            and all(
                not _rectangles_overlap(
                    candidates[cand_idx]["rect"], candidates[other]["rect"]
                )
                for other in chosen
            )
        ]
        options.sort(key=lambda cand_idx: candidates[cand_idx]["members"].bit_count(), reverse=True)

        for cand_idx in options:
            chosen.append(cand_idx)
            search(remaining & ~candidates[cand_idx]["members"], chosen)
            chosen.pop()

    search((1 << clue_count) - 1, [])
    assert len(solutions) == 1, "target regions must be uniquely recoverable"
    return [candidates[idx] for idx in solutions[0]]


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each sparse target region with the uniquely matched flip of the source pattern.

    Input structure:
    - The background is the most common color.
    - One compact multicolor source pattern fully occupies a small rectangle.
    - Several other same-sized regions are mostly background and only keep 2-3 clue pixels.

    Rule:
    - Extract the full rectangular source pattern.
    - Try its horizontal flip, vertical flip, and 180-degree rotation against each clue-defined target region.
    - For every region, pick the only transform whose colors match all visible clue pixels, then fill the whole region with that transformed pattern.
    - Leave the original source pattern unchanged.

    Key insight:
    - The sparse pixels are not standalone objects: they are partial observations of a hidden full pattern, so the solver must group them into same-sized regions and use the clue colors to disambiguate the correct flip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    source_bbox = _find_source_bbox(g, bg)
    sr0, sc0, sr1, sc1 = source_bbox
    source = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    variants = _transform_variants(source)

    clues = _collect_clues(g, bg, source_bbox)
    assert clues, "expected at least one target region"
    candidates = _candidate_targets(g, clues, source_bbox, variants)
    targets = _unique_cover(candidates, len(clues))

    out = g.copy()
    for target in targets:
        patch = variants[target["transform"]]
        r0 = int(target["r0"])
        c0 = int(target["c0"])
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


MIN_PATTERN_SIDE = 4
MAX_PATTERN_SIDE = 6
MIN_GRID_SIDE = 14
MAX_GRID_SIDE = 30
MIN_GAP = 2
GEN_ATTEMPTS = 400
PATTERN_ATTEMPTS = 200
PALETTE = tuple(range(10))
TRANSFORM_NAMES = ("flip_h", "flip_v", "rot180")


def _has_symmetry(pattern: np.ndarray) -> bool:
    return (
        np.array_equal(pattern, np.fliplr(pattern))
        or np.array_equal(pattern, np.flipud(pattern))
        or np.array_equal(pattern, np.rot90(pattern, 2))
    )


def _well_separated(points: tuple[tuple[int, int], ...]) -> bool:
    for (r1, c1), (r2, c2) in itertools.combinations(points, 2):
        if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
            return False
    return True


def _valid_clue_sets(variants: dict[str, np.ndarray]) -> dict[str, dict[int, list[tuple[tuple[int, int], ...]]]]:
    h, w = next(iter(variants.values())).shape
    positions = [(r, c) for r in range(h) for c in range(w)]
    valid: dict[str, dict[int, list[tuple[tuple[int, int], ...]]]] = {
        name: {2: [], 3: []} for name in variants
    }

    for name, patch in variants.items():
        for size in (2, 3):
            for cells in itertools.combinations(positions, size):
                rows = [r for r, _ in cells]
                cols = [c for _, c in cells]
                if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
                    continue
                if not _well_separated(cells):
                    continue
                matches = [
                    other
                    for other, other_patch in variants.items()
                    if all(int(other_patch[r, c]) == int(patch[r, c]) for r, c in cells)
                ]
                if matches == [name]:
                    valid[name][size].append(cells)
    return valid


def _random_pattern() -> tuple[np.ndarray, dict[str, dict[int, list[tuple[tuple[int, int], ...]]]]] | None:
    for _ in range(PATTERN_ATTEMPTS):
        h = random.randint(MIN_PATTERN_SIDE, MAX_PATTERN_SIDE)
        w = random.randint(MIN_PATTERN_SIDE, MAX_PATTERN_SIDE)
        bg = random.choice(PALETTE)
        palette_size = random.randint(3, 5)
        colors = random.sample([c for c in PALETTE if c != bg], palette_size)

        pattern = np.array(
            [[random.choice(colors) for _ in range(w)] for _ in range(h)],
            dtype=int,
        )
        for color in colors:
            rr = random.randrange(h)
            cc = random.randrange(w)
            pattern[rr, cc] = color

        if _has_symmetry(pattern):
            continue

        variants = _transform_variants(pattern)
        clue_sets = _valid_clue_sets(variants)
        if all(clue_sets[name][2] or clue_sets[name][3] for name in TRANSFORM_NAMES):
            return pattern, clue_sets
    return None


def _pack_starts(length: int, item: int, count: int) -> list[int] | None:
    slack = length - count * item - (count + 1) * MIN_GAP
    if slack < 0:
        return None
    extras = list(np.random.multinomial(slack, [1.0 / (count + 1)] * (count + 1)))
    starts: list[int] = []
    pos = MIN_GAP + int(extras[0])
    for idx in range(count):
        starts.append(pos)
        pos += item
        if idx + 1 < count:
            pos += MIN_GAP + int(extras[idx + 1])
    return starts


def _place_regions(
    box_h: int, box_w: int, total_boxes: int
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    slot_side = 2 if total_boxes <= 4 else 3
    min_h = max(MIN_GRID_SIDE, slot_side * box_h + (slot_side + 1) * MIN_GAP)
    min_w = max(MIN_GRID_SIDE, slot_side * box_w + (slot_side + 1) * MIN_GAP)
    if min_h > MAX_GRID_SIDE or min_w > MAX_GRID_SIDE:
        return None

    grid_h = random.randint(min_h, MAX_GRID_SIDE)
    grid_w = random.randint(min_w, MAX_GRID_SIDE)
    row_starts = _pack_starts(grid_h, box_h, slot_side)
    col_starts = _pack_starts(grid_w, box_w, slot_side)
    if row_starts is None or col_starts is None:
        return None

    slots = [(r, c) for r in row_starts for c in col_starts]
    chosen = random.sample(slots, total_boxes)
    return (grid_h, grid_w), chosen


def _choose_clues(
    transform_name: str,
    clue_sets: dict[str, dict[int, list[tuple[tuple[int, int], ...]]]],
) -> tuple[tuple[int, int], ...]:
    preferred = random.choice((2, 3))
    options = clue_sets[transform_name][preferred]
    if not options:
        options = clue_sets[transform_name][5 - preferred]
    return random.choice(options)


def _build_example() -> tuple[np.ndarray, np.ndarray] | None:
    pattern_info = _random_pattern()
    if pattern_info is None:
        return None
    source, clue_sets = pattern_info
    box_h, box_w = source.shape

    copies_per_transform = random.choice((1, 2))
    transforms = [name for name in TRANSFORM_NAMES for _ in range(copies_per_transform)]
    random.shuffle(transforms)
    total_boxes = 1 + len(transforms)

    placement = _place_regions(box_h, box_w, total_boxes)
    if placement is None:
        return None
    grid_shape, positions = placement
    bg = random.choice([c for c in PALETTE if c not in set(map(int, source.ravel()))])

    grid = np.full(grid_shape, bg, dtype=int)
    expected = np.full_like(grid, bg)

    source_slot = random.randrange(total_boxes)
    source_pos = positions[source_slot]
    sr0, sc0 = source_pos
    grid[sr0 : sr0 + box_h, sc0 : sc0 + box_w] = source
    expected[sr0 : sr0 + box_h, sc0 : sc0 + box_w] = source

    variants = _transform_variants(source)
    target_positions = [pos for idx, pos in enumerate(positions) if idx != source_slot]
    for transform_name, (r0, c0) in zip(transforms, target_positions):
        patch = variants[transform_name]
        clues = _choose_clues(transform_name, clue_sets)
        for rr, cc in clues:
            grid[r0 + rr, c0 + cc] = int(patch[rr, cc])
        expected[r0 : r0 + box_h, c0 : c0 + box_w] = patch

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_example()
        if built is not None:
            grid, _ = built
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle")
