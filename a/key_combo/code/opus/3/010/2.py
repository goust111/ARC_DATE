from __future__ import annotations

import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
SPRITE_COLORS = (BLUE, RED, GREEN)

GRID_SIDE_RANGE = (24, 29)
RECT_SIDE_RANGE = (6, 8)
TARGET_AREA_PATTERNS = (
    (5, 5, 6),
    (5, 6, 6),
    (5, 5, 7),
    (6, 6, 7),
    (4, 5, 5),
    (4, 6, 6),
)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords) -> tuple[int, int, int, int]:
    arr = np.asarray(list(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    arr = np.asarray(list(coords), dtype=int)
    if arr.size == 0:
        return tuple()
    r0, c0 = arr.min(axis=0)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in arr)
    return tuple(rel)


def _sig_dims(sig: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in sig), 1 + max(c for _, c in sig)


def _sig_to_mask(sig: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _sig_dims(sig)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in sig:
        mask[r, c] = True
    return mask


def _transform_sig(sig: tuple[tuple[int, int], ...], op: int) -> tuple[tuple[int, int], ...]:
    mask = _sig_to_mask(sig)
    if op >= 4:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=op % 4)
    return _normalize_coords(np.argwhere(mask))


def _is_connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nb in _neighbors4(*cell):
            if nb in cells and nb not in seen:
                stack.append(nb)
    return len(seen) == len(cells)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(comp, dtype=int))
    return out


def _random_free_box(
    reserved: np.ndarray, box_h: int, box_w: int, *, border: int = 1, padding: int = 1
) -> tuple[int, int] | None:
    candidates = _free_boxes(reserved, box_h, box_w, border=border, padding=padding)
    if not candidates:
        return None
    return random.choice(candidates)


def _free_boxes(
    reserved: np.ndarray, box_h: int, box_w: int, *, border: int = 1, padding: int = 1
) -> list[tuple[int, int]]:
    h, w = reserved.shape
    candidates = []
    for r0 in range(border, h - border - box_h + 1):
        for c0 in range(border, w - border - box_w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + box_h + padding)
            cc1 = min(w, c0 + box_w + padding)
            if not reserved[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    return candidates


def _valid_target_shape(sig: tuple[tuple[int, int], ...]) -> bool:
    area = len(sig)
    if not 4 <= area <= 7:
        return False
    h, w = _sig_dims(sig)
    if min(h, w) < 2:
        return False
    if area == h * w:
        return False
    return True


def _random_connected_shape(area: int, *, max_side: int = 4) -> tuple[tuple[int, int], ...] | None:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = {
            nb
            for r, c in cells
            for nb in _neighbors4(r, c)
            if nb not in cells
        }
        frontier = list(frontier)
        random.shuffle(frontier)
        placed = False
        for cand in frontier:
            trial = set(cells)
            trial.add(cand)
            sig = _normalize_coords(trial)
            h, w = _sig_dims(sig)
            if h <= max_side and w <= max_side:
                cells = set(sig)
                placed = True
                break
        if not placed:
            return None
    sig = _normalize_coords(cells)
    if not _valid_target_shape(sig):
        return None
    return sig


def _removable_subsets(sig: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    cells = list(sig)
    cellset = set(sig)
    out = []
    for k in (1, 2):
        for subset in combinations(cells, k):
            if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in combinations(subset, 2)):
                continue
            source = cellset - set(subset)
            if not 3 <= len(source) <= 6:
                continue
            if not _is_connected(source):
                continue
            ok = True
            for cell in subset:
                if sum((nb in source) for nb in _neighbors4(*cell)) < 2:
                    ok = False
                    break
            if ok:
                out.append(tuple(sorted(subset)))
    return out


def _delete_noise_candidates(
    source_cells: tuple[tuple[int, int], ...], forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    source = set(source_cells)
    r0, c0, r1, c1 = _bbox_coords(source_cells)
    cands = []
    for r in range(r0 - 2, r1 + 3):
        for c in range(c0 - 2, c1 + 3):
            cell = (r, c)
            if cell in forbidden:
                continue
            d = min(abs(r - sr) + abs(c - sc) for sr, sc in source)
            if d not in (1, 2):
                continue
            neighbors = sum((nb in source) for nb in _neighbors4(r, c))
            if neighbors >= 2:
                continue
            cands.append(cell)
    return cands


def _pick_non_touching(
    candidates: list[tuple[int, int]], count: int, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    pool = candidates[:]
    random.shuffle(pool)
    chosen: list[tuple[int, int]] = []
    for cell in pool:
        if any(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1 for other in blocked):
            continue
        if any(abs(cell[0] - other[0]) + abs(cell[1] - other[1]) == 1 for other in chosen):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _build_shape_triplet():
    for _ in range(250):
        areas = list(random.choice(TARGET_AREA_PATTERNS))
        random.shuffle(areas)
        entries = []
        target_sigs = set()
        source_sigs = set()
        for area in areas:
            built = None
            for _ in range(250):
                base = _random_connected_shape(area)
                if base is None:
                    continue
                target = _transform_sig(base, random.randrange(8))
                if target in target_sigs:
                    continue
                subsets = _removable_subsets(target)
                if not subsets:
                    continue
                prefer_merge = 2 if random.random() < 0.45 else 1
                preferred = [s for s in subsets if len(s) == prefer_merge] or subsets
                subset = random.choice(preferred)
                target_set = set(target)
                source_cells = tuple(sorted(target_set - set(subset)))
                source_sig = _normalize_coords(source_cells)
                if source_sig in target_sigs or source_sig in source_sigs:
                    continue
                delete_cands = _delete_noise_candidates(source_cells, target_set)
                delete_min = 1
                delete_max = min(3, 4 - len(subset))
                delete_counts = list(range(delete_min, delete_max + 1))
                random.shuffle(delete_counts)
                delete_cells = None
                for delete_count in delete_counts:
                    delete_cells = _pick_non_touching(delete_cands, delete_count, set(subset))
                    if delete_cells is not None:
                        break
                if delete_cells is None:
                    continue
                built = {
                    "target_sig": target,
                    "source_cells": source_cells,
                    "source_sig": source_sig,
                    "merge_cells": tuple(sorted(subset)),
                    "delete_cells": tuple(sorted(delete_cells)),
                }
                break
            if built is None:
                break
            entries.append(built)
            target_sigs.add(built["target_sig"])
            source_sigs.add(built["source_sig"])
        if len(entries) != 3:
            continue
        all_targets = {entry["target_sig"] for entry in entries}
        if any(entry["source_sig"] in all_targets for entry in entries):
            continue
        areas_now = [len(entry["target_sig"]) for entry in entries]
        if len(set(areas_now)) == 3:
            continue
        return entries
    return None


def _make_rect_patch(
    rect_h: int, rect_w: int, hole_sig: tuple[tuple[int, int], ...], hole_offset: tuple[int, int]
) -> np.ndarray:
    patch = np.full((rect_h, rect_w), GREY, dtype=int)
    off_r, off_c = hole_offset
    for r, c in hole_sig:
        patch[off_r + r, off_c + c] = BLACK
    return patch


def _make_sprite_patch(entry: dict, color: int) -> dict:
    source = list(entry["source_cells"])
    merge = list(entry["merge_cells"])
    delete = list(entry["delete_cells"])
    all_cells = source + merge + delete
    arr = np.asarray(all_cells, dtype=int)
    min_r, min_c = arr.min(axis=0)
    max_r, max_c = arr.max(axis=0)
    patch = np.zeros((int(max_r - min_r + 1), int(max_c - min_c + 1)), dtype=int)

    def shift(cells):
        return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in cells))

    source_rel = shift(source)
    merge_rel = shift(merge)
    delete_rel = shift(delete)
    for r, c in source_rel:
        patch[r, c] = color
    for r, c in merge_rel:
        patch[r, c] = ORANGE
    for r, c in delete_rel:
        patch[r, c] = ORANGE
    return {
        "patch": patch,
        "source_rel": source_rel,
        "merge_rel": merge_rel,
        "delete_rel": delete_rel,
    }


def _render_expected(scene: dict) -> np.ndarray:
    out = np.zeros(scene["shape"], dtype=int)
    for rect in scene["rects"]:
        r0, c0 = rect["pos"]
        patch = rect["patch"]
        h, w = patch.shape
        out[r0 : r0 + h, c0 : c0 + w] = GREY
        for rr, cc in rect["hole_abs"]:
            out[rr, cc] = rect["fill_color"]
    return out


def _place_sprite_at(spec: dict, pos: tuple[int, int]) -> dict:
    placed = dict(spec)
    r0, c0 = pos
    placed["pos"] = pos
    placed["source_abs"] = tuple(sorted((r0 + r, c0 + c) for r, c in spec["source_rel"]))
    placed["merge_abs"] = tuple(sorted((r0 + r, c0 + c) for r, c in spec["merge_rel"]))
    placed["delete_abs"] = tuple(sorted((r0 + r, c0 + c) for r, c in spec["delete_rel"]))
    return placed


def _special_secondary_position(
    reserved: np.ndarray, primary: dict, secondary: dict
) -> dict | None:
    box_h, box_w = secondary["patch"].shape
    source_abs_primary = set(primary["source_abs"])
    candidates = []
    for mr, mc in primary["merge_abs"]:
        primary_support = sum((nb in source_abs_primary) for nb in _neighbors4(mr, mc))
        if primary_support < 2:
            continue
        for br, bc in secondary["source_rel"]:
            for ar, ac in _neighbors4(mr, mc):
                r0 = ar - br
                c0 = ac - bc
                h, w = reserved.shape
                if r0 < 1 or c0 < 1 or r0 + box_h > h - 1 or c0 + box_w > w - 1:
                    continue
                if reserved[r0 : r0 + box_h, c0 : c0 + box_w].any():
                    continue
                placed = _place_sprite_at(secondary, (r0, c0))
                secondary_support = sum((nb in set(placed["source_abs"])) for nb in _neighbors4(mr, mc))
                if 1 <= secondary_support < primary_support and _sprites_obey_noise_rules([primary, placed]):
                    candidates.append(placed)
    if not candidates:
        return None
    return random.choice(candidates)


def _count_sprite_neighbors(grid: np.ndarray, r: int, c: int) -> dict[int, int]:
    counts = {color: 0 for color in SPRITE_COLORS}
    h, w = grid.shape
    for nr, nc in _neighbors4(r, c):
        if 0 <= nr < h and 0 <= nc < w:
            color = int(grid[nr, nc])
            if color in counts:
                counts[color] += 1
    return counts


def _count_source_neighbors_from_sprites(sprites: list[dict], r: int, c: int) -> dict[int, int]:
    counts = {color: 0 for color in SPRITE_COLORS}
    for sprite in sprites:
        source_abs = set(sprite["source_abs"])
        counts[sprite["color"]] = sum((nb in source_abs) for nb in _neighbors4(r, c))
    return counts


def _sprites_obey_noise_rules(sprites: list[dict]) -> bool:
    for sprite in sprites:
        for r, c in sprite["merge_abs"]:
            counts = _count_source_neighbors_from_sprites(sprites, r, c)
            total = sum(counts.values())
            if total < 2:
                return False
            best = max(counts.values())
            if list(counts.values()).count(best) != 1:
                return False
            if max(counts, key=counts.get) != sprite["color"]:
                return False
        for r, c in sprite["delete_abs"]:
            counts = _count_source_neighbors_from_sprites(sprites, r, c)
            if sum(counts.values()) >= 2:
                return False
    return True


def _validate_scene(scene: dict, grid: np.ndarray) -> None:
    hole_sigs = [rect["hole_sig"] for rect in scene["rects"]]
    assert len(hole_sigs) == 3 and len(set(hole_sigs)) == 3

    source_sigs = [sprite["source_sig"] for sprite in scene["sprites"]]
    assert all(sig not in set(hole_sigs) for sig in source_sigs)

    seen_merge = False
    seen_delete = False
    seen_cross_color_merge = False
    for sprite in scene["sprites"]:
        processed = _normalize_coords(list(sprite["source_abs"]) + list(sprite["merge_abs"]))
        assert processed == sprite["target_sig"]
        assert 2 <= len(sprite["merge_abs"]) + len(sprite["delete_abs"]) <= 4
        assert sprite["merge_abs"]
        assert sprite["delete_abs"]

        for r, c in sprite["merge_abs"]:
            counts = _count_sprite_neighbors(grid, r, c)
            total = sum(counts.values())
            assert total >= 2
            best = max(counts.values())
            assert list(counts.values()).count(best) == 1
            assert max(counts, key=counts.get) == sprite["color"]
            if sum(v > 0 for v in counts.values()) > 1:
                seen_cross_color_merge = True
            seen_merge = True
        for r, c in sprite["delete_abs"]:
            counts = _count_sprite_neighbors(grid, r, c)
            assert sum(counts.values()) < 2
            seen_delete = True

    assert seen_merge and seen_delete
    if scene.get("require_cross_color_merge"):
        assert seen_cross_color_merge

    source_targets = set(hole_sigs)
    assert not all(sig in source_targets for sig in source_sigs)

    expected = _render_expected(scene)
    actual = main(grid)
    assert np.array_equal(actual, expected)


def _generate_scene() -> dict | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)

    shape_entries = _build_shape_triplet()
    if shape_entries is None:
        return None

    colors = list(SPRITE_COLORS)
    random.shuffle(colors)
    for entry, color in zip(shape_entries, colors):
        entry["color"] = color

    rects = []
    for entry in shape_entries:
        hole_h, hole_w = _sig_dims(entry["target_sig"])
        rect_h = random.randint(max(RECT_SIDE_RANGE[0], hole_h + 2), RECT_SIDE_RANGE[1])
        rect_w = random.randint(max(RECT_SIDE_RANGE[0], hole_w + 2), RECT_SIDE_RANGE[1])
        off_r = random.randint(1, rect_h - hole_h - 1)
        off_c = random.randint(1, rect_w - hole_w - 1)
        rects.append(
            {
                "patch": _make_rect_patch(rect_h, rect_w, entry["target_sig"], (off_r, off_c)),
                "hole_sig": entry["target_sig"],
                "hole_offset": (off_r, off_c),
                "fill_color": entry["color"],
            }
        )

    reserved = np.zeros((h, w), dtype=bool)
    for rect in sorted(rects, key=lambda item: item["patch"].size, reverse=True):
        rh, rw = rect["patch"].shape
        pos = _random_free_box(reserved, rh, rw, border=1, padding=1)
        if pos is None:
            return None
        rect["pos"] = pos
        r0, c0 = pos
        reserved[r0 : r0 + rh, c0 : c0 + rw] = True
        off_r, off_c = rect["hole_offset"]
        rect["hole_abs"] = tuple(
            sorted((r0 + off_r + r, c0 + off_c + c) for r, c in rect["hole_sig"])
        )

    sprite_specs = []
    for entry in shape_entries:
        spec = _make_sprite_patch(entry, entry["color"])
        spec.update(
            {
                "color": entry["color"],
                "target_sig": entry["target_sig"],
                "source_sig": entry["source_sig"],
            }
        )
        sprite_specs.append(spec)

    sprites = []
    require_cross_color_merge = False
    if len(sprite_specs) == 3 and random.random() < 0.45:
        ordered = sprite_specs[:]
        random.shuffle(ordered)
        for primary_spec in ordered:
            free_primary = _free_boxes(reserved, *primary_spec["patch"].shape, border=1, padding=1)
            random.shuffle(free_primary)
            others = [spec for spec in ordered if spec is not primary_spec]
            for pos in free_primary[:40]:
                primary = _place_sprite_at(primary_spec, pos)
                temp_reserved = reserved.copy()
                pr0, pc0 = pos
                ph, pw = primary_spec["patch"].shape
                temp_reserved[pr0 : pr0 + ph, pc0 : pc0 + pw] = True
                random.shuffle(others)
                placed_pair = None
                for secondary_spec in others:
                    secondary = _special_secondary_position(temp_reserved, primary, secondary_spec)
                    if secondary is None:
                        continue
                    placed_pair = (primary, secondary, secondary_spec)
                    break
                if placed_pair is None:
                    continue
                primary, secondary, secondary_spec = placed_pair
                sprites.extend([primary, secondary])
                sr0, sc0 = secondary["pos"]
                sh, sw = secondary["patch"].shape
                reserved = temp_reserved
                reserved[sr0 : sr0 + sh, sc0 : sc0 + sw] = True
                removed_ids = {id(primary_spec), id(secondary_spec)}
                sprite_specs = [spec for spec in sprite_specs if id(spec) not in removed_ids]
                require_cross_color_merge = True
                break
            if require_cross_color_merge:
                break

    for spec in sorted(sprite_specs, key=lambda item: item["patch"].size, reverse=True):
        ph, pw = spec["patch"].shape
        pos = _random_free_box(reserved, ph, pw, border=1, padding=1)
        if pos is None:
            return None
        reserved[pos[0] : pos[0] + ph, pos[1] : pos[1] + pw] = True
        sprites.append(_place_sprite_at(spec, pos))

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        r0, c0 = rect["pos"]
        patch = rect["patch"]
        rh, rw = patch.shape
        grid[r0 : r0 + rh, c0 : c0 + rw] = patch
    for sprite in sprites:
        r0, c0 = sprite["pos"]
        patch = sprite["patch"]
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BLACK, patch, grid[r0 : r0 + ph, c0 : c0 + pw])

    scene = {
        "shape": (h, w),
        "rects": rects,
        "sprites": sprites,
        "require_cross_color_merge": require_cross_color_merge,
    }
    _validate_scene(scene, grid)
    scene["input"] = grid
    return scene


def _extract_holes(grid: np.ndarray) -> list[dict]:
    holes = []
    for comp in _components_mask(grid == GREY):
        r0, c0, r1, c1 = _bbox_coords(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        black = patch == BLACK
        internal = []
        for hole in _components_mask(black):
            if np.any(hole[:, 0] == 0) or np.any(hole[:, 0] == patch.shape[0] - 1):
                continue
            if np.any(hole[:, 1] == 0) or np.any(hole[:, 1] == patch.shape[1] - 1):
                continue
            internal.append(hole)
        if len(internal) != 1:
            continue
        hole = internal[0]
        hole_abs = tuple(sorted((r0 + int(r), c0 + int(c)) for r, c in hole))
        holes.append({"sig": _normalize_coords(hole), "coords": hole_abs})
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sculpt each colored sprite with adjacent orange noise so it matches one grey hole, then embed it there.

    Input structure:
    - A black background contains three grey rectangular objects, each with one internal black hole.
    - Elsewhere are three colored 4-connected sprites in blue, red, and green.
    - Orange singleton noise pixels sit near the sprites; some touch a sprite strongly enough to merge, others do not.

    Rule:
    - Extract every internal hole shape from the grey rectangles, using the hole's own bounding box as the shape frame.
    - For each orange pixel, count 4-neighbors belonging to the colored sprites. Pixels with at least two sprite neighbors merge into the uniquely best-supported adjacent sprite color; pixels with fewer than two such neighbors are erased.
    - After this simultaneous noise processing, each modified sprite shape matches exactly one hole shape up to translation.
    - Clear the original sprites and orange noise, then paste each modified sprite into the matching hole in its own color.

    Key insight:
    - The orange pixels are not disposable clutter: their 4-neighbor support decides which pixels become welded onto a sprite, and only that sculpted shape fits a hole.
    """
    g = np.asarray(grid, dtype=int)
    holes = _extract_holes(g)
    hole_by_sig = {hole["sig"]: hole["coords"] for hole in holes}
    assert len(holes) == 3 and len(hole_by_sig) == 3

    sprites: dict[int, set[tuple[int, int]]] = {}
    for color in SPRITE_COLORS:
        comps = _components_mask(g == color)
        assert len(comps) == 1
        sprites[color] = {tuple(map(int, cell)) for cell in comps[0]}

    merge_for_color = {color: [] for color in SPRITE_COLORS}
    for r, c in np.argwhere(g == ORANGE):
        counts = _count_sprite_neighbors(g, int(r), int(c))
        if sum(counts.values()) < 2:
            continue
        best = max(counts.values())
        winners = [color for color, count in counts.items() if count == best]
        assert len(winners) == 1
        merge_for_color[winners[0]].append((int(r), int(c)))

    matches: dict[int, tuple[tuple[int, int], ...]] = {}
    for color, cells in sprites.items():
        sculpted = _normalize_coords(list(cells) + merge_for_color[color])
        assert sculpted in hole_by_sig
        matches[color] = hole_by_sig[sculpted]

    out = g.copy()
    out[np.isin(out, [BLUE, RED, GREEN, ORANGE])] = BLACK
    for color, coords in matches.items():
        for r, c in coords:
            out[r, c] = color
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        scene = _generate_scene()
        if scene is not None:
            return scene["input"].copy()
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
