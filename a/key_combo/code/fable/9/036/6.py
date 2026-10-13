import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GENERATION_RETRIES = 800
MASK_RETRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _rot90_ccw_pos(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return n - 1 - c, r


def _orbit_positions(seed: tuple[int, int], n: int) -> list[tuple[int, int]]:
    orbit = []
    pos = seed
    for _ in range(4):
        orbit.append(pos)
        pos = _rot90_ccw_pos(pos, n)
    return orbit


def _all_orbits(n: int) -> list[tuple[tuple[int, int], ...]]:
    seen = set()
    out = []
    for r in range(n):
        for c in range(n):
            orbit = _orbit_positions((r, c), n)
            if len(set(orbit)) != 4:
                continue
            key = tuple(sorted(orbit))
            if key in seen:
                continue
            seen.add(key)
            out.append(tuple(_orbit_positions(min(orbit), n)))
    return out


def _rotation_turns(src: tuple[int, int], dst: tuple[int, int], n: int) -> int:
    pos = src
    for turns in range(4):
        if pos == dst:
            return turns
        pos = _rot90_ccw_pos(pos, n)
    raise AssertionError("positions are not on the same orbit")


def _separator_bands(separators: list[int], length: int) -> tuple[list[tuple[int, int]], int]:
    assert separators[0] == 0 and separators[-1] == length - 1, "expected outer separator border"
    spans = []
    for a, b in zip(separators, separators[1:]):
        assert b > a + 1, "slots must have positive size"
        spans.append((a + 1, b))
    sizes = {end - start for start, end in spans}
    assert len(sizes) == 1, "slot sizes must be uniform"
    size = sizes.pop()
    assert size >= 2
    return spans, size


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert 1 <= h <= 30 and 1 <= w <= 30, "grid must stay within ARC bounds"

    bg = _most_common_color(g)
    row_seps = [r for r in range(h) if np.all(g[r, :] == bg)]
    col_seps = [c for c in range(w) if np.all(g[:, c] == bg)]
    row_bands, slot_h = _separator_bands(row_seps, h)
    col_bands, slot_w = _separator_bands(col_seps, w)
    assert slot_h == slot_w, "slot cells must be square"
    assert len(row_bands) == len(col_bands), "table must be square"
    n = len(row_bands)
    assert n % 2 == 1 and n >= 3, "slot table must have odd size at least 3"

    slots: dict[tuple[int, int], dict] = {}
    for sr, (r0, r1) in enumerate(row_bands):
        for sc, (c0, c1) in enumerate(col_bands):
            block = g[r0:r1, c0:c1]
            non_bg = sorted(set(map(int, block.ravel())) - {bg})
            assert len(non_bg) == 1, "each slot must contain exactly one foreground color"
            color = non_bg[0]
            mask = block == color
            assert mask.any(), "slot mask cannot be empty"
            solid = bool(mask.all())
            if not solid:
                assert np.any(block == bg), "non-placeholder slots must expose slot background"
            slots[(sr, sc)] = {
                "color": color,
                "mask": mask.copy(),
                "solid": solid,
                "band": (r0, r1, c0, c1),
            }

    return {
        "grid": g,
        "bg": bg,
        "n": n,
        "slot_size": slot_h,
        "row_bands": row_bands,
        "col_bands": col_bands,
        "slots": slots,
    }


def _find_family(parsed: dict) -> dict:
    slots = parsed["slots"]
    n = parsed["n"]

    solid_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pos, info in slots.items():
        if info["solid"]:
            solid_by_color[int(info["color"])].append(pos)

    candidates = []
    for placeholder_color, positions in solid_by_color.items():
        if len(positions) != 2:
            continue
        orbit = _orbit_positions(positions[0], n)
        if positions[1] not in orbit:
            continue

        placeholders = set(positions)
        known = [pos for pos in orbit if pos not in placeholders]
        if len(known) != 2:
            continue

        info_a = slots[known[0]]
        info_b = slots[known[1]]
        if info_a["solid"] or info_b["solid"]:
            continue
        if int(info_a["color"]) != int(info_b["color"]):
            continue

        turns = _rotation_turns(known[0], known[1], n)
        if not np.array_equal(np.rot90(info_a["mask"], turns), info_b["mask"]):
            continue

        candidates.append(
            {
                "placeholder_color": int(placeholder_color),
                "family_color": int(info_a["color"]),
                "orbit": orbit,
                "placeholders": [pos for pos in orbit if pos in placeholders],
                "known": known,
            }
        )

    assert len(candidates) == 1, "expected exactly one placeholder-marked rotation family"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the two placeholder slots by rotating a known orbit member by the same angle as the slot orbit around the table center.

    Input structure:
    - The grid is an odd-by-odd table of equal square symbol slots, with a one-pixel background border and separators.
    - Each non-placeholder slot contains a monochrome symbol mask on the slot background; two slots are solid placeholder fills.
    - The two placeholder slots belong to the same four-slot 90-degree rotation orbit as two same-color symbol slots whose masks rotate with the orbit.

    Rule:
    - Parse the slot table and find the unique four-slot orbit where the two solid placeholder slots share the orbit with two same-color non-solid slots.
    - Read either known family member, rotate its mask by the number of quarter-turns needed to move its slot to each placeholder slot, and paint the rotated mask in the family color.
    - Remove the placeholder fill while leaving every other slot and every separator pixel unchanged.

    Key insight:
    - The missing masks are not determined by set difference alone: the orbit position itself encodes which rotation variant belongs in which placeholder.
    """
    parsed = _parse_grid(grid)
    family = _find_family(parsed)
    g = parsed["grid"]
    out = g.copy()
    bg = parsed["bg"]
    n = parsed["n"]
    slots = parsed["slots"]

    source_pos = family["known"][0]
    source_mask = slots[source_pos]["mask"]
    family_color = family["family_color"]

    for target_pos in family["placeholders"]:
        turns = _rotation_turns(source_pos, target_pos, n)
        target_mask = np.rot90(source_mask, turns)
        r0, r1, c0, c1 = slots[target_pos]["band"]
        block = out[r0:r1, c0:c1]
        block[:] = bg
        block[target_mask] = family_color

    return out


def _neighbors4_in_size(size: int, r: int, c: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < size and 0 <= cc < size:
            yield rr, cc


def _touches_every_row_and_col(mask: np.ndarray) -> bool:
    return bool(np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)))


def _random_connected_mask(size: int) -> np.ndarray | None:
    min_area = max(3, size + 1)
    max_area = size * size - 2
    if min_area > max_area:
        return None

    for _ in range(MASK_RETRIES):
        target_area = random.randint(min_area, max_area)
        seed = (random.randrange(size), random.randrange(size))
        cells = {seed}
        frontier = set(_neighbors4_in_size(size, *seed))
        while len(cells) < target_area and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            cells.add((rr, cc))
            for nr, nc in _neighbors4_in_size(size, rr, cc):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))

        if len(cells) != target_area:
            continue

        mask = np.zeros((size, size), dtype=bool)
        for rr, cc in cells:
            mask[rr, cc] = True
        if not _touches_every_row_and_col(mask):
            continue
        if mask.all():
            continue
        return mask
    return None


def _generate_mask(
    size: int,
    *,
    banned: set[tuple[tuple[int, ...], ...]],
    require_four_rotations: bool,
    mirror_distinct_from_rotations: bool,
) -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        mask = _random_connected_mask(size)
        if mask is None:
            return None
        rotations = [np.rot90(mask, k) for k in range(4)]
        keys = {_mask_key(rot) for rot in rotations}
        if require_four_rotations and len(keys) != 4:
            continue
        if banned & keys:
            continue
        if mirror_distinct_from_rotations:
            mirror = np.fliplr(mask)
            if any(np.array_equal(mirror, rot) for rot in rotations):
                continue
        return mask
    return None


def _build_slot_table(assignments: dict, n: int, slot_size: int, bg: int) -> np.ndarray:
    extent = n * slot_size + n + 1
    grid = np.full((extent, extent), bg, dtype=int)
    for (sr, sc), spec in assignments.items():
        r0 = 1 + sr * (slot_size + 1)
        c0 = 1 + sc * (slot_size + 1)
        block = grid[r0 : r0 + slot_size, c0 : c0 + slot_size]
        color = int(spec["color"])
        if spec["solid"]:
            block[:] = color
        else:
            block[spec["mask"]] = color
    return grid


def _wrong_unrotated_fill(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    family = _find_family(parsed)
    out = parsed["grid"].copy()
    source_mask = parsed["slots"][family["known"][0]]["mask"]
    for target_pos in family["placeholders"]:
        r0, r1, c0, c1 = parsed["slots"][target_pos]["band"]
        block = out[r0:r1, c0:c1]
        block[:] = parsed["bg"]
        block[source_mask] = family["family_color"]
    return out


def _wrong_swapped_fill(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    family = _find_family(parsed)
    out = parsed["grid"].copy()
    source_pos = family["known"][0]
    masks = []
    for target_pos in family["placeholders"]:
        turns = _rotation_turns(source_pos, target_pos, parsed["n"])
        masks.append(np.rot90(parsed["slots"][source_pos]["mask"], turns))
    masks = list(reversed(masks))
    for target_pos, target_mask in zip(family["placeholders"], masks):
        r0, r1, c0, c1 = parsed["slots"][target_pos]["band"]
        block = out[r0:r1, c0:c1]
        block[:] = parsed["bg"]
        block[target_mask] = family["family_color"]
    return out


def _validate_sample(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
        parsed = _parse_grid(grid)
        family = _find_family(parsed)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(grid, expected):
        return False

    source_pos = family["known"][0]
    target_keys = []
    for target_pos in family["placeholders"]:
        turns = _rotation_turns(source_pos, target_pos, parsed["n"])
        target_keys.append(_mask_key(np.rot90(parsed["slots"][source_pos]["mask"], turns)))
    if len(set(target_keys)) != 2:
        return False

    if np.array_equal(_wrong_unrotated_fill(grid), expected):
        return False
    if np.array_equal(_wrong_swapped_fill(grid), expected):
        return False

    row_seps = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == parsed["bg"])]
    col_seps = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == parsed["bg"])]
    if len(row_seps) != parsed["n"] + 1 or len(col_seps) != parsed["n"] + 1:
        return False

    return True


def _choice_from_masks(masks: list[np.ndarray]) -> np.ndarray:
    return masks[random.randrange(len(masks))]


def _gen_attempt() -> np.ndarray | None:
    bg = BACKGROUND
    n = random.choice((3, 5))
    slot_size = random.choice((3, 4, 5) if n == 3 else (3, 4))
    if n * slot_size + n + 1 > 30:
        return None

    family_mask = _generate_mask(
        slot_size,
        banned=set(),
        require_four_rotations=True,
        mirror_distinct_from_rotations=True,
    )
    if family_mask is None:
        return None
    family_rotations = [np.rot90(family_mask, k) for k in range(4)]
    banned = {_mask_key(mask) for mask in family_rotations}
    mirror_rotations = [np.rot90(np.fliplr(family_mask), k) for k in range(4)]
    banned |= {_mask_key(mask) for mask in mirror_rotations}

    decoy_pair_mask = _generate_mask(
        slot_size,
        banned=banned,
        require_four_rotations=True,
        mirror_distinct_from_rotations=False,
    )
    if decoy_pair_mask is None:
        return None
    banned |= {_mask_key(np.rot90(decoy_pair_mask, k)) for k in range(4)}

    extra_mask = _generate_mask(
        slot_size,
        banned=banned,
        require_four_rotations=False,
        mirror_distinct_from_rotations=False,
    )
    if extra_mask is None:
        return None
    extra_rotations = [np.rot90(extra_mask, k) for k in range(4)]

    palette = random.sample(COLOR_POOL, 6)
    placeholder_color = palette[0]
    family_color = palette[1]
    decoy_pair_color = palette[2]
    same_shape_other_color = palette[3]
    extra_color = palette[4]
    extra_color_2 = palette[5]

    orbits = _all_orbits(n)
    family_orbit = random.choice(orbits)
    other_orbits = [orbit for orbit in orbits if set(orbit) != set(family_orbit)]
    if not other_orbits:
        return None
    decoy_orbit = random.choice(other_orbits)

    adjacent_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    opposite_pairs = [(0, 2), (1, 3)]
    placeholder_indices = random.choice(adjacent_pairs if random.random() < 0.5 else opposite_pairs)
    placeholder_set = set(placeholder_indices)
    base_turn = random.randrange(4)

    assignments = {}
    for idx, pos in enumerate(family_orbit):
        if idx in placeholder_set:
            assignments[pos] = {"solid": True, "color": placeholder_color}
        else:
            assignments[pos] = {
                "solid": False,
                "color": family_color,
                "mask": family_rotations[(base_turn + idx) % 4],
            }

    decoy_pair_indices = random.choice(adjacent_pairs if random.random() < 0.5 else opposite_pairs)
    decoy_pair_set = set(decoy_pair_indices)
    decoy_base_turn = random.randrange(4)
    fillers = [pos for idx, pos in enumerate(decoy_orbit) if idx not in decoy_pair_set]
    filler_masks = [_choice_from_masks(family_rotations), _choice_from_masks(mirror_rotations)]
    filler_colors = [same_shape_other_color, family_color]
    random.shuffle(fillers)
    for idx, pos in enumerate(decoy_orbit):
        if idx in decoy_pair_set:
            assignments[pos] = {
                "solid": False,
                "color": decoy_pair_color,
                "mask": np.rot90(decoy_pair_mask, (decoy_base_turn + idx) % 4),
            }
    for pos, mask, color in zip(fillers, filler_masks, filler_colors):
        assignments[pos] = {"solid": False, "color": color, "mask": mask}

    remaining = [(r, c) for r in range(n) for c in range(n) if (r, c) not in assignments]
    for pos in remaining:
        roll = random.random()
        if roll < 0.25:
            mask = _choice_from_masks(family_rotations)
            color = random.choice([same_shape_other_color, extra_color, extra_color_2])
        elif roll < 0.50:
            mask = _choice_from_masks(mirror_rotations)
            color = family_color
        elif roll < 0.75:
            mask = _choice_from_masks(extra_rotations)
            color = random.choice([family_color, extra_color, extra_color_2])
        else:
            mask = _choice_from_masks(extra_rotations + mirror_rotations)
            color = random.choice([decoy_pair_color, same_shape_other_color, extra_color, extra_color_2])
        assignments[pos] = {"solid": False, "color": color, "mask": mask}

    grid = _build_slot_table(assignments, n, slot_size, bg)
    expected = main(grid)
    if not _validate_sample(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid orbital-synchronous-rotation input")
