from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
MAROON = 9

GRID_SIZE = 17
TILE_SIZE = 5
STRIDE = 6
SLOT_COUNT = 3
OCCUPIED_TILES = 6


def _rot90(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 1)


def _rot180(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 2)


def _rot270(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, 3)


def _flip_ud(mask: np.ndarray) -> np.ndarray:
    return np.flipud(mask)


def _flip_lr(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask)


def _diag(mask: np.ndarray) -> np.ndarray:
    return mask.T.copy()


def _anti(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(np.flipud(mask)).T.copy()


NON_ID_TRANSFORMS = (
    ("rot90", _rot90),
    ("rot180", _rot180),
    ("rot270", _rot270),
    ("flip_ud", _flip_ud),
    ("flip_lr", _flip_lr),
    ("diag", _diag),
    ("anti", _anti),
)

FAMILY_TRANSFORMS = {
    "rot90": _rot90,
    "rot180": _rot180,
    "flip_ud": _flip_ud,
    "flip_lr": _flip_lr,
    "diag": _diag,
    "anti": _anti,
}

REFLECTION_FAMILIES = ("flip_ud", "flip_lr", "diag", "anti")
FAMILY_NAMES = tuple(FAMILY_TRANSFORMS)


def _slot_origin(slot_r: int, slot_c: int) -> tuple[int, int]:
    return slot_r * STRIDE, slot_c * STRIDE


def _symmetry_closure(mask: np.ndarray, transform) -> np.ndarray:
    closed = mask.copy()
    for _ in range(4):
        nxt = closed | transform(closed)
        if np.array_equal(nxt, closed):
            break
        closed = nxt
    return closed


def _candidate_masks(tile: np.ndarray) -> list[np.ndarray]:
    visible = tile == RED
    allowed = tile != MAROON
    if not visible.any():
        return []

    unique: dict[bytes, np.ndarray] = {}
    for _name, transform in NON_ID_TRANSFORMS:
        closed = _symmetry_closure(visible, transform)
        if np.array_equal(closed, allowed):
            unique.setdefault(closed.tobytes(), closed.copy())
    return list(unique.values())


def _complete_tile(tile: np.ndarray) -> np.ndarray:
    candidates = _candidate_masks(tile)
    assert len(candidates) == 1, "fragment must determine a unique symmetric completion"
    out = np.full((TILE_SIZE, TILE_SIZE), MAROON, dtype=int)
    out[candidates[0]] = RED
    return out


def _tile_from_mask(mask: np.ndarray) -> np.ndarray:
    tile = np.full((TILE_SIZE, TILE_SIZE), MAROON, dtype=int)
    tile[mask] = RED
    return tile


def _extract_occupied_tiles(grid: np.ndarray) -> list[np.ndarray]:
    tiles = []
    for slot_r in range(SLOT_COUNT):
        for slot_c in range(SLOT_COUNT):
            r0, c0 = _slot_origin(slot_r, slot_c)
            tile = grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE]
            assert set(map(int, np.unique(tile))).issubset({BLACK, RED, MAROON})
            if np.any(tile == MAROON):
                tiles.append(tile.copy())
            else:
                assert np.all(tile == BLACK), "empty slots must be fully black"
    return tiles


def _render_sorted_tiles(tiles: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for idx, tile in enumerate(tiles):
        slot_r, slot_c = divmod(idx, SLOT_COUNT)
        r0, c0 = _slot_origin(slot_r, slot_c)
        out[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each truncated symmetric fragment, sort the completed tiles by red-pixel density, and repack them into the top two rows.

    Input structure:
    - A 17x17 grid is split by black separator rows and columns into a 3x3 array of 5x5 slots.
    - Exactly 6 slots contain maroon tiles with red marker fragments; the remaining 3 slots are fully black.
    - Within an occupied tile, red marker pixels were removed from one or two rows or one or two columns and turned black.

    Rule:
    - For each occupied tile, enumerate dihedral symmetries and find the unique symmetric completion whose orbit-closure matches the visible red pixels plus the black cut cells.
    - Restore the missing red pixels, count each completed tile's red area, and sort the 6 tiles from smallest count to largest.
    - Write the sorted tiles into the first 6 output slots in row-major order and leave the third row of slots empty.

    Key insight:
    - The black cells inside an occupied tile are not ordinary background; they are missing red pixels, so the correct symmetry is the one whose closure exactly fills every non-maroon cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 17x17 3x3 slot canvas"
    assert np.all(g[5, :] == BLACK) and np.all(g[11, :] == BLACK)
    assert np.all(g[:, 5] == BLACK) and np.all(g[:, 11] == BLACK)

    tiles = _extract_occupied_tiles(g)
    assert len(tiles) == OCCUPIED_TILES, "expected exactly 6 occupied tiles"

    completed = [_complete_tile(tile) for tile in tiles]
    completed.sort(key=lambda tile: int(np.count_nonzero(tile == RED)))
    return _render_sorted_tiles(completed)


def _coord_rot90(r: int, c: int) -> tuple[int, int]:
    return c, TILE_SIZE - 1 - r


def _coord_rot180(r: int, c: int) -> tuple[int, int]:
    return TILE_SIZE - 1 - r, TILE_SIZE - 1 - c


def _coord_flip_ud(r: int, c: int) -> tuple[int, int]:
    return TILE_SIZE - 1 - r, c


def _coord_flip_lr(r: int, c: int) -> tuple[int, int]:
    return r, TILE_SIZE - 1 - c


def _coord_diag(r: int, c: int) -> tuple[int, int]:
    return c, r


def _coord_anti(r: int, c: int) -> tuple[int, int]:
    return TILE_SIZE - 1 - c, TILE_SIZE - 1 - r


FAMILY_COORD_TRANSFORMS = {
    "rot90": _coord_rot90,
    "rot180": _coord_rot180,
    "flip_ud": _coord_flip_ud,
    "flip_lr": _coord_flip_lr,
    "diag": _coord_diag,
    "anti": _coord_anti,
}

STRIP_SELECTORS = {"rows": [], "cols": []}
for length in (1, 2):
    for start in range(TILE_SIZE - length + 1):
        row_selector = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        row_selector[start : start + length, :] = True
        STRIP_SELECTORS["rows"].append(row_selector)

        col_selector = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        col_selector[:, start : start + length] = True
        STRIP_SELECTORS["cols"].append(col_selector)


def _orbit_partition(coord_transform) -> tuple[tuple[tuple[int, int], ...], ...]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(TILE_SIZE):
        for c in range(TILE_SIZE):
            if (r, c) in seen:
                continue
            orbit = []
            cur = (r, c)
            while cur not in orbit:
                orbit.append(cur)
                cur = coord_transform(*cur)
            orbit_tuple = tuple(sorted(orbit))
            seen.update(orbit_tuple)
            orbits.append(orbit_tuple)
    return tuple(orbits)


FAMILY_ORBITS = {
    family: _orbit_partition(coord_transform)
    for family, coord_transform in FAMILY_COORD_TRANSFORMS.items()
}


def _mask_has_exact_family(mask: np.ndarray, family: str) -> bool:
    invariants = {
        name: bool(np.array_equal(mask, transform(mask)))
        for name, transform in FAMILY_TRANSFORMS.items()
    }
    reflection_hits = sum(int(invariants[name]) for name in REFLECTION_FAMILIES)
    if family == "rot90":
        return invariants["rot90"] and reflection_hits == 0
    if family == "rot180":
        return invariants["rot180"] and not invariants["rot90"] and reflection_hits == 0
    return (
        invariants[family]
        and not invariants["rot90"]
        and not invariants["rot180"]
        and reflection_hits == 1
    )


def _valid_truncated_tiles(full_mask: np.ndarray, mode: str) -> list[np.ndarray]:
    selectors = STRIP_SELECTORS[mode][:]
    options = []
    full_tile = _tile_from_mask(full_mask)
    full_count = int(full_mask.sum())
    for selector in selectors:
        removed = full_mask & selector
        removed_count = int(removed.sum())
        if removed_count == 0 or removed_count == full_count:
            continue
        tile = full_tile.copy()
        tile[removed] = BLACK
        if int(np.count_nonzero(tile == RED)) < 2:
            continue
        candidates = _candidate_masks(tile)
        if len(candidates) == 1 and np.array_equal(candidates[0], full_mask):
            options.append(tile)
    return options


def _random_family_mask(family: str) -> np.ndarray | None:
    orbits = FAMILY_ORBITS[family]
    for _ in range(400):
        mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=bool)
        chosen = [orbit for orbit in orbits if random.random() < 0.5]
        if not chosen:
            continue
        for orbit in chosen:
            for r, c in orbit:
                mask[r, c] = True
        area = int(mask.sum())
        if area < 3 or area > 16:
            continue
        if int(mask.any(axis=0).sum()) < 2 or int(mask.any(axis=1).sum()) < 2:
            continue
        if _mask_has_exact_family(mask, family):
            return mask
    return None


PIECE_CATALOG: dict[str, dict[str, dict[int, list[dict]]]] | None = None


def _build_piece_catalog() -> dict[str, dict[str, dict[int, list[dict]]]]:
    catalog = {
        family: {"rows": {}, "cols": {}}
        for family in FAMILY_NAMES
    }
    for family in FAMILY_NAMES:
        seen_masks: set[bytes] = set()
        seen_tiles: dict[tuple[str, int], set[bytes]] = {}
        for _ in range(1500):
            full_mask = _random_family_mask(family)
            if full_mask is None:
                continue
            mask_key = full_mask.tobytes()
            if mask_key in seen_masks:
                continue
            seen_masks.add(mask_key)
            red_count = int(full_mask.sum())
            output_tile = _tile_from_mask(full_mask)
            for mode in ("rows", "cols"):
                tile_keys = seen_tiles.setdefault((mode, red_count), set())
                for tile in _valid_truncated_tiles(full_mask, mode):
                    tile_key = tile.tobytes()
                    if tile_key in tile_keys:
                        continue
                    tile_keys.add(tile_key)
                    record = {
                        "count": red_count,
                        "input_tile": tile.copy(),
                        "output_tile": output_tile.copy(),
                    }
                    catalog[family][mode].setdefault(red_count, []).append(record)
            enough = True
            for mode in ("rows", "cols"):
                distinct_counts = sum(bool(records) for records in catalog[family][mode].values())
                total_records = sum(len(records) for records in catalog[family][mode].values())
                if distinct_counts < 5 or total_records < 12:
                    enough = False
                    break
            if enough:
                break
    return catalog


def _get_piece_catalog() -> dict[str, dict[str, dict[int, list[dict]]]]:
    global PIECE_CATALOG
    if PIECE_CATALOG is None:
        PIECE_CATALOG = _build_piece_catalog()
    return PIECE_CATALOG


def _make_piece(used_counts: set[int], family: str, mode: str) -> dict | None:
    catalog = _get_piece_catalog()[family][mode]
    available_counts = [count for count in catalog if count not in used_counts]
    if not available_counts:
        return None
    count = random.choice(available_counts)
    record = random.choice(catalog[count])
    return {
        "count": int(record["count"]),
        "input_tile": record["input_tile"].copy(),
        "output_tile": record["output_tile"].copy(),
    }


def _compatible_families(mode: str) -> list[str]:
    catalog = _get_piece_catalog()
    return [family for family in FAMILY_NAMES if catalog[family][mode]]


def _family_plan_for_modes(modes: list[str]) -> list[str] | None:
    families: list[str | None] = [None] * len(modes)
    distinct_goal = min(4, len(FAMILY_NAMES), len(modes))
    priority_indices = random.sample(range(len(modes)), distinct_goal)
    used: set[str] = set()

    for idx in priority_indices:
        candidates = [family for family in _compatible_families(modes[idx]) if family not in used]
        if not candidates:
            return None
        family = random.choice(candidates)
        families[idx] = family
        used.add(family)

    for idx, mode in enumerate(modes):
        if families[idx] is not None:
            continue
        candidates = _compatible_families(mode)
        if not candidates:
            return None
        families[idx] = random.choice(candidates)

    return [family for family in families if family is not None]


def _occupied_slot_indices() -> list[int]:
    slots = list(range(SLOT_COUNT * SLOT_COUNT))
    for _ in range(100):
        chosen = random.sample(slots, OCCUPIED_TILES)
        if any(idx >= 6 for idx in chosen):
            return chosen
    raise RuntimeError("failed to choose occupied slots touching the bottom row")


def _build_input_grid(pieces: list[dict], occupied_slots: list[int]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    shuffled = pieces[:]
    random.shuffle(shuffled)
    for piece, slot_idx in zip(shuffled, occupied_slots):
        slot_r, slot_c = divmod(slot_idx, SLOT_COUNT)
        r0, c0 = _slot_origin(slot_r, slot_c)
        grid[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE] = piece["input_tile"]
    return grid


def _build_output_grid(pieces: list[dict]) -> np.ndarray:
    ordered_tiles = [piece["output_tile"] for piece in sorted(pieces, key=lambda piece: piece["count"])]
    return _render_sorted_tiles(ordered_tiles)


def _gen_attempt() -> np.ndarray | None:
    modes = ["rows"] * (OCCUPIED_TILES // 2) + ["cols"] * (OCCUPIED_TILES - OCCUPIED_TILES // 2)
    random.shuffle(modes)
    families = _family_plan_for_modes(modes)
    if families is None:
        return None

    used_counts: set[int] = set()
    pieces = []
    for family, mode in zip(families, modes):
        piece = _make_piece(used_counts, family, mode)
        if piece is None:
            return None
        pieces.append(piece)
        used_counts.add(piece["count"])

    occupied_slots = _occupied_slot_indices()
    inp = _build_input_grid(pieces, occupied_slots)
    out = _build_output_grid(pieces)

    if np.array_equal(inp, out):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 10")
