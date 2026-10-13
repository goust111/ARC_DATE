from __future__ import annotations

import random

import numpy as np


BOARD_SIZE = 11
TILE_SIZE = 3
GAP = 1
TILE_STARTS = (0, 4, 8)
ALL_COLORS = tuple(range(10))
ROTATION_LABELS = ("rot90", "rot180", "rot270")
REFLECTION_LABELS = ("flip", "flip_rot90", "flip_rot180", "flip_rot270")
GENERATION_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _mask_key(mask: np.ndarray) -> bytes:
    return np.asarray(mask, dtype=np.uint8).tobytes()


def _apply_transform(arr: np.ndarray, label: str) -> np.ndarray:
    x = np.asarray(arr)
    if label == "id":
        return x.copy()
    if label == "rot90":
        return np.rot90(x, 1).copy()
    if label == "rot180":
        return np.rot90(x, 2).copy()
    if label == "rot270":
        return np.rot90(x, 3).copy()
    if label == "flip":
        return np.fliplr(x).copy()
    if label == "flip_rot90":
        return np.rot90(np.fliplr(x), 1).copy()
    if label == "flip_rot180":
        return np.rot90(np.fliplr(x), 2).copy()
    if label == "flip_rot270":
        return np.rot90(np.fliplr(x), 3).copy()
    raise ValueError(f"unknown transform: {label}")


def _named_transforms(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "id": _apply_transform(arr, "id"),
        "rot90": _apply_transform(arr, "rot90"),
        "rot180": _apply_transform(arr, "rot180"),
        "rot270": _apply_transform(arr, "rot270"),
        "flip": _apply_transform(arr, "flip"),
        "flip_rot90": _apply_transform(arr, "flip_rot90"),
        "flip_rot180": _apply_transform(arr, "flip_rot180"),
        "flip_rot270": _apply_transform(arr, "flip_rot270"),
    }


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = np.asarray(mask, dtype=bool)
    for label, variant in _named_transforms(base).items():
        if label != "id" and np.array_equal(base, variant):
            return False
    return True


def _enumerate_masks() -> tuple[list[np.ndarray], dict[int, list[np.ndarray]]]:
    occ_candidates: list[np.ndarray] = []
    by_area: dict[int, list[np.ndarray]] = {area: [] for area in range(1, 10)}
    for bits in range(1, 1 << 9):
        mask = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if (bits >> idx) & 1:
                mask[idx // 3, idx % 3] = True
        area = int(mask.sum())
        by_area[area].append(mask.copy())
        if 4 <= area <= 6 and _is_asymmetric(mask):
            occ_candidates.append(mask.copy())
    return occ_candidates, by_area


OCCUPANCY_CANDIDATES, MASKS_BY_AREA = _enumerate_masks()


def _find_board(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    candidates: list[tuple[int, int, int]] = []
    for sep in (int(x) for x in np.unique(g) if int(x) != int(bg)):
        for r0 in range(H - BOARD_SIZE + 1):
            r1 = r0 + 3
            r2 = r0 + 7
            for c0 in range(W - BOARD_SIZE + 1):
                c1 = c0 + 3
                c2 = c0 + 7
                if not np.all(g[r1, c0 : c0 + BOARD_SIZE] == sep):
                    continue
                if not np.all(g[r2, c0 : c0 + BOARD_SIZE] == sep):
                    continue
                if not np.all(g[r0 : r0 + BOARD_SIZE, c1] == sep):
                    continue
                if not np.all(g[r0 : r0 + BOARD_SIZE, c2] == sep):
                    continue
                ok = True
                for tr in TILE_STARTS:
                    for tc in TILE_STARTS:
                        if np.any(g[r0 + tr : r0 + tr + TILE_SIZE, c0 + tc : c0 + tc + TILE_SIZE] == sep):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((int(r0), int(c0), int(sep)))
    assert len(candidates) == 1, "expected exactly one 3x3-block board"
    return candidates[0]


def _extract_tiles(board: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    tiles: dict[tuple[int, int], np.ndarray] = {}
    for br, r0 in enumerate(TILE_STARTS):
        for bc, c0 in enumerate(TILE_STARTS):
            tiles[(int(br), int(bc))] = np.asarray(
                board[r0 : r0 + TILE_SIZE, c0 : c0 + TILE_SIZE], dtype=int
            ).copy()
    return tiles


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    board_r0, board_c0, sep = _find_board(g, bg)
    board = g[board_r0 : board_r0 + BOARD_SIZE, board_c0 : board_c0 + BOARD_SIZE].copy()
    tiles = _extract_tiles(board)

    occ = np.zeros((3, 3), dtype=bool)
    for pos, tile in tiles.items():
        occ[pos] = bool(np.any(tile != bg))

    orbit = _named_transforms(occ)
    exact_matches: list[tuple[int, int]] = []
    transformed_matches: list[tuple[int, int]] = []
    for pos, tile in tiles.items():
        mask = tile != bg
        if np.array_equal(mask, occ):
            exact_matches.append(pos)
        elif any(np.array_equal(mask, orbit[label]) for label in orbit if label != "id"):
            transformed_matches.append(pos)

    return {
        "bg": int(bg),
        "sep": int(sep),
        "board_origin": (int(board_r0), int(board_c0)),
        "board": board,
        "tiles": tiles,
        "occ": occ,
        "exact_matches": exact_matches,
        "transformed_matches": transformed_matches,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop out the unique 3x3 block whose internal occupancy map is the board's own block-level portrait.

    Input structure:
    - The canvas contains one 11x11 mini-board embedded in background.
    - Two full separator rows and two full separator columns split the board into a 3x3 array of 3x3 tiles.
    - Some tiles contain non-background pixels and the rest are empty.

    Rule:
    - Build a 3x3 occupancy map by marking which tiles are non-empty.
    - Exactly one tile has the same 3x3 occupied-cell pattern as that occupancy map itself, compared in the tile's native orientation.
    - Tiles whose patterns match only a rotation or reflection of the occupancy map are distractors and must be ignored.
    - Output that unique matching tile as a 3x3 crop, preserving its original colors.

    Key insight:
    - The answer tile is a self-portrait of the whole board at block scale, so the solver must first read the macro occupancy pattern and only then compare it to the micro patterns inside the tiles.
    """
    info = _analyze_grid(np.asarray(grid, dtype=int))
    assert len(info["exact_matches"]) == 1, "expected a unique self-referential tile"
    target = info["exact_matches"][0]
    return info["tiles"][target].copy()


def _choose_distractor_labels(max_count: int) -> list[str]:
    modes: list[tuple[list[str], int]] = []
    modes.append(([], 1))
    if max_count >= 1:
        modes.append((["rot"], 3))
        modes.append((["refl"], 3))
    if max_count >= 2:
        modes.append((["rot", "refl"], 5))
    labels_mode = random.choices([mode for mode, _ in modes], weights=[w for _, w in modes], k=1)[0]
    labels: list[str] = []
    for tag in labels_mode:
        if tag == "rot":
            labels.append(random.choice(ROTATION_LABELS))
        else:
            labels.append(random.choice(REFLECTION_LABELS))
    return labels


def _colorize_mask(mask: np.ndarray, bg: int, palette: list[int]) -> np.ndarray:
    tile = np.full((3, 3), int(bg), dtype=int)
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not coords:
        return tile
    colors = [int(x) for x in palette]
    assignments: list[int] = []
    if len(colors) == 1:
        assignments = [colors[0]] * len(coords)
    else:
        assignments.extend(colors[: len(coords)])
        while len(assignments) < len(coords):
            assignments.append(random.choice(colors))
        random.shuffle(assignments)
    for (r, c), col in zip(coords, assignments):
        tile[r, c] = int(col)
    return tile


def _pick_filler_mask(
    orbit_keys: set[bytes],
    used_keys: set[bytes],
    *,
    allowed_areas: list[int],
    require_self_mark: tuple[int, int] | None,
) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    for area in allowed_areas:
        for mask in MASKS_BY_AREA[area]:
            key = _mask_key(mask)
            if key in orbit_keys or key in used_keys:
                continue
            if require_self_mark is not None and not bool(mask[require_self_mark]):
                continue
            candidates.append(mask)
    if not candidates:
        return None
    return random.choice(candidates).copy()


def _paint_board(
    grid: np.ndarray,
    board_origin: tuple[int, int],
    sep: int,
    tiles: dict[tuple[int, int], np.ndarray],
) -> None:
    r0, c0 = map(int, board_origin)
    grid[r0 + 3, c0 : c0 + BOARD_SIZE] = int(sep)
    grid[r0 + 7, c0 : c0 + BOARD_SIZE] = int(sep)
    grid[r0 : r0 + BOARD_SIZE, c0 + 3] = int(sep)
    grid[r0 : r0 + BOARD_SIZE, c0 + 7] = int(sep)
    for br, rr in enumerate(TILE_STARTS):
        for bc, cc in enumerate(TILE_STARTS):
            tile = tiles[(int(br), int(bc))]
            grid[r0 + rr : r0 + rr + TILE_SIZE, c0 + cc : c0 + cc + TILE_SIZE] = tile


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        info = _analyze_grid(grid)
        occ = info["occ"]
        if not _is_asymmetric(occ):
            return False
        if len(info["exact_matches"]) != 1:
            return False
        target_pos = info["exact_matches"][0]
        if not bool(occ[target_pos]):
            return False
        same_area_decoys = 0
        self_marked_non_target = 0
        for pos, tile in info["tiles"].items():
            if not bool(occ[pos]):
                continue
            mask = tile != info["bg"]
            if pos != target_pos and int(mask.sum()) == int(occ.sum()):
                same_area_decoys += 1
            if pos != target_pos and bool(mask[pos]):
                self_marked_non_target += 1
        if same_area_decoys < 1:
            return False
        if self_marked_non_target < 1:
            return False
        out = main(grid)
        if not np.array_equal(out, info["tiles"][target_pos]):
            return False
        return True
    except Exception:
        return False


def _generate_attempt() -> np.ndarray | None:
    bg = random.choice(ALL_COLORS)
    sep = random.choice([c for c in ALL_COLORS if c != bg])
    usable_colors = [c for c in ALL_COLORS if c not in (bg, sep)]

    occ = random.choice(OCCUPANCY_CANDIDATES).copy()
    area = int(occ.sum())
    occupied_positions = [(int(r), int(c)) for r, c in np.argwhere(occ)]
    target_pos = random.choice(occupied_positions)
    remaining_positions = [pos for pos in occupied_positions if pos != target_pos]

    distractor_labels = _choose_distractor_labels(len(remaining_positions))
    random.shuffle(remaining_positions)
    distractor_positions = remaining_positions[: len(distractor_labels)]
    filler_positions = remaining_positions[len(distractor_labels) :]

    palette_size = random.choices([1, 2, 3], weights=[2, 4, 4], k=1)[0]
    target_palette = random.sample(usable_colors, palette_size)
    target_tile = _colorize_mask(occ, bg, target_palette)

    tiles: dict[tuple[int, int], np.ndarray] = {
        (r, c): np.full((3, 3), int(bg), dtype=int) for r in range(3) for c in range(3)
    }
    tiles[target_pos] = target_tile
    for pos, label in zip(distractor_positions, distractor_labels):
        tiles[pos] = _apply_transform(target_tile, label)

    orbit_keys = {_mask_key(mask) for mask in _named_transforms(occ).values()}
    used_mask_keys: set[bytes] = set()
    same_area_needed = len(distractor_positions) == 0

    for idx, pos in enumerate(filler_positions):
        require_self = pos if idx == 0 else None
        allowed_areas = [area] if same_area_needed else list(range(2, 7))
        mask = _pick_filler_mask(
            orbit_keys,
            used_mask_keys,
            allowed_areas=allowed_areas,
            require_self_mark=require_self,
        )
        if mask is None:
            return None
        used_mask_keys.add(_mask_key(mask))
        if idx == 0:
            palette = target_palette[:]
        else:
            filler_palette_size = random.choices([1, 2, 3], weights=[3, 4, 3], k=1)[0]
            palette = random.sample(usable_colors, filler_palette_size)
        tiles[pos] = _colorize_mask(mask, bg, palette)
        same_area_needed = False

    H = random.randint(11, 20)
    W = random.randint(11, 20)
    board_r0 = random.randint(0, H - BOARD_SIZE)
    board_c0 = random.randint(0, W - BOARD_SIZE)
    grid = np.full((H, W), int(bg), dtype=int)
    _paint_board(grid, (board_r0, board_c0), sep, tiles)

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-referential block puzzle")
