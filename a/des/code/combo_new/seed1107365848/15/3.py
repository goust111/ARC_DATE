from __future__ import annotations

from collections import deque

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _all_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for turns in range(4):
        rotated = np.rot90(mask, turns)
        for variant in (rotated, np.fliplr(rotated)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(np.array(variant, copy=True))
    return variants


def _canonical_shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(map(int, variant.shape)), variant.tobytes()) for variant in _all_variants(mask))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            mask, bbox = _component_mask(coord_array)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "mask": mask,
                    "bbox": bbox,
                    "shape_key": _canonical_shape_key(mask),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every fragment to the color of the most frequent shape-equivalence class.

    Input structure:
    - The background is a dominant green field.
    - Several small monochrome fragments are scattered on the grid; each fragment is 4-connected.
    - Fragments can be partitioned into shape classes where members match after rotation and/or reflection.

    Rule:
    - Extract every non-background fragment as a separate 4-connected component.
    - Group fragments by binary shape, treating the eight dihedral transforms as equivalent.
    - Find the unique largest shape group and take its shared color.
    - Preserve every fragment's position and shape, but recolor all foreground cells to that winning color.

    Key insight:
    - Color is not the grouping signal; the hidden structure is shape equivalence under rotation/flip, and the decisive statistic is the class size.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground fragment"

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for comp in components:
        groups.setdefault(comp["shape_key"], []).append(comp)

    counts = {key: len(items) for key, items in groups.items()}
    max_count = max(counts.values())
    winners = [key for key, count in counts.items() if count == max_count]
    assert len(winners) == 1, "largest shape group must be unique"
    winner = winners[0]

    winner_colors = {int(comp["color"]) for comp in groups[winner]}
    assert len(winner_colors) == 1, "all fragments in the winning group must share one color"
    winner_color = winner_colors.pop()

    out = g.copy()
    out[g != bg] = winner_color
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG_COLOR = 3
COLOR_POOL = [0, 1, 2, 4, 5, 6, 7, 8, 9]
GROUP_COUNT_MIN, GROUP_COUNT_MAX_EXCLUSIVE = 2, 5
WINNER_COUNT_MIN, WINNER_COUNT_MAX_EXCLUSIVE = 3, 6
LOSER_COUNT_MIN = 2
TOTAL_FRAGMENTS_MAX = 12
AREA_MIN, AREA_MAX_EXCLUSIVE = 3, 7
SLOT_STEP = 7
SLOT_USABLE = 6
SLOT_DIM_CHOICES = (2, 3, 4)
MARGIN_MAX = 2
SHAPE_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400


def _normalize_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return {(r - min_r, c - min_c) for r, c in coords}


def _random_polyomino(area: int) -> np.ndarray:
    coords = {(0, 0)}
    while len(coords) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in coords:
            for dr, dc in DIRS4:
                cell = (r + dr, c + dc)
                if cell not in coords:
                    frontier.add(cell)
        coords.add(tuple(frontier)[int(_RNG.integers(len(frontier)))])

    norm = _normalize_coords(coords)
    height = max(r for r, _ in norm) + 1
    width = max(c for _, c in norm) + 1
    mask = np.zeros((height, width), dtype=bool)
    for r, c in norm:
        mask[r, c] = True
    return mask


def _sample_group_counts() -> tuple[list[int], int] | None:
    for _ in range(100):
        group_count = int(_RNG.integers(GROUP_COUNT_MIN, GROUP_COUNT_MAX_EXCLUSIVE))
        winner_index = int(_RNG.integers(group_count))
        winner_count = int(_RNG.integers(WINNER_COUNT_MIN, WINNER_COUNT_MAX_EXCLUSIVE))
        counts = []
        for group_idx in range(group_count):
            if group_idx == winner_index:
                counts.append(winner_count)
            else:
                counts.append(int(_RNG.integers(LOSER_COUNT_MIN, winner_count)))
        if sum(counts) <= TOTAL_FRAGMENTS_MAX:
            return counts, winner_index
    return None


def _sample_groups() -> tuple[list[dict], int] | None:
    sampled = _sample_group_counts()
    if sampled is None:
        return None
    counts, winner_index = sampled
    colors = list(_RNG.choice(COLOR_POOL, size=len(counts), replace=False))

    groups: list[dict] = []
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    for group_idx, count in enumerate(counts):
        for _ in range(SHAPE_ATTEMPTS):
            area = int(_RNG.integers(AREA_MIN, AREA_MAX_EXCLUSIVE))
            mask = _random_polyomino(area)
            key = _canonical_shape_key(mask)
            variants = _all_variants(mask)
            if key in used_keys:
                continue
            if group_idx == winner_index and len(variants) < 2:
                continue
            used_keys.add(key)
            groups.append(
                {
                    "count": int(count),
                    "color": int(colors[group_idx]),
                    "mask": mask,
                    "variants": variants,
                    "shape_key": key,
                }
            )
            break
        else:
            return None
    return groups, winner_index


def _colorize(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    patch[mask] = int(color)
    return patch


def _build_fragments(groups: list[dict]) -> list[dict]:
    fragments: list[dict] = []
    for group in groups:
        variants = group["variants"]
        picks = [int(_RNG.integers(len(variants))) for _ in range(group["count"])]
        if group["count"] >= 2 and len(variants) >= 2 and len(set(picks)) == 1:
            picks[1] = (picks[0] + 1) % len(variants)
        for pick in picks:
            mask = variants[pick]
            fragments.append(
                {
                    "mask": mask,
                    "patch": _colorize(mask, group["color"]),
                    "color": int(group["color"]),
                    "shape_key": group["shape_key"],
                }
            )
    _RNG.shuffle(fragments)
    return fragments


def _sample_canvas(total_fragments: int) -> tuple[int, int, int, int, int, int] | None:
    feasible = [
        (slot_rows, slot_cols)
        for slot_rows in SLOT_DIM_CHOICES
        for slot_cols in SLOT_DIM_CHOICES
        if slot_rows * slot_cols >= total_fragments
    ]
    if not feasible:
        return None

    slot_rows, slot_cols = feasible[int(_RNG.integers(len(feasible)))]
    core_h = slot_rows * SLOT_STEP - 1
    core_w = slot_cols * SLOT_STEP - 1

    max_extra_h = min(MARGIN_MAX * 2, 30 - core_h)
    max_extra_w = min(MARGIN_MAX * 2, 30 - core_w)
    if max_extra_h < 0 or max_extra_w < 0:
        return None

    top = int(_RNG.integers(0, min(MARGIN_MAX, max_extra_h) + 1))
    bottom = int(_RNG.integers(0, min(MARGIN_MAX, max_extra_h - top) + 1))
    left = int(_RNG.integers(0, min(MARGIN_MAX, max_extra_w) + 1))
    right = int(_RNG.integers(0, min(MARGIN_MAX, max_extra_w - left) + 1))

    height = core_h + top + bottom
    width = core_w + left + right
    return height, width, slot_rows, slot_cols, top, left


def _place_fragments(fragments: list[dict]) -> np.ndarray | None:
    sampled = _sample_canvas(len(fragments))
    if sampled is None:
        return None
    height, width, slot_rows, slot_cols, top, left = sampled
    grid = np.full((height, width), BG_COLOR, dtype=int)

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    _RNG.shuffle(slots)
    chosen_slots = slots[: len(fragments)]

    for fragment, (slot_r, slot_c) in zip(fragments, chosen_slots):
        patch = fragment["patch"]
        ph, pw = patch.shape
        if ph > SLOT_USABLE or pw > SLOT_USABLE:
            return None
        base_r = top + slot_r * SLOT_STEP
        base_c = left + slot_c * SLOT_STEP
        r0 = base_r + int(_RNG.integers(0, SLOT_USABLE - ph + 1))
        c0 = base_c + int(_RNG.integers(0, SLOT_USABLE - pw + 1))
        grid[r0 : r0 + ph, c0 : c0 + pw][patch != BG_COLOR] = patch[patch != BG_COLOR]

    return grid


def _expected_output(grid: np.ndarray, winner_color: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[out != BG_COLOR] = int(winner_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_groups()
    if sampled is None:
        return None
    groups, winner_index = sampled
    fragments = _build_fragments(groups)
    grid = _place_fragments(fragments)
    if grid is None:
        return None

    expected = _expected_output(grid, groups[winner_index]["color"])
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
