from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_SPRITE_SIZE = 3
_STEP = 4
_PALETTE = np.arange(1, 10, dtype=int)
_GEN_ATTEMPTS = 500
_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _bool_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    seen = set()
    stack = [tuple(map(int, coords[0]))]
    while stack:
        cell = stack.pop()
        if cell in seen:
            continue
        seen.add(cell)
        for nr, nc in _neighbors8(*cell):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and bool(mask[nr, nc]) and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _d4_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in (mask, np.fliplr(mask)):
        for k in range(4):
            variant = np.rot90(base, k).copy()
            sig = variant.tobytes()
            if sig not in seen:
                seen.add(sig)
                variants.append(variant)
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return min(_mask_key(variant) for variant in _d4_variants(mask))


def _find_sprite_components(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components: list[dict] = []
    visited = np.zeros(g.shape, dtype=bool)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == bg or visited[r, c]:
                continue
            stack = [(r, c)]
            coords = []
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and not visited[nr, nc] and int(g[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            box = _bbox(coord_arr)
            r0, c0, r1, c1 = box
            assert (r1 - r0 + 1, c1 - c0 + 1) == (_SPRITE_SIZE, _SPRITE_SIZE), "every sprite must have a 3x3 bbox"
            mask = np.zeros((_SPRITE_SIZE, _SPRITE_SIZE), dtype=bool)
            rel = coord_arr - np.array([r0, c0], dtype=int)
            mask[rel[:, 0], rel[:, 1]] = True
            components.append(
                {
                    "color": color,
                    "coords": coord_arr,
                    "bbox": box,
                    "mask": mask,
                    "raw_key": _mask_key(mask),
                    "canonical_key": _canonical_key(mask),
                }
            )
    return components


def _histogram_from_groups(
    groups: dict,
    *,
    key_order_tiebreak=None,
    require_unique_majority: bool = True,
    require_unique_counts: bool = True,
) -> np.ndarray:
    items = []
    for key, entries in groups.items():
        counts = Counter(int(entry["color"]) for entry in entries)
        max_freq = max(counts.values())
        winners = [color for color, freq in counts.items() if freq == max_freq]
        if require_unique_majority:
            assert len(winners) == 1, "each class must have a unique majority color"
        items.append((key, len(entries), int(min(winners))))

    if key_order_tiebreak is None:
        ordered = sorted(items, key=lambda item: (-item[1], item[0]))
    else:
        ordered = sorted(items, key=lambda item: (-item[1], key_order_tiebreak(item[0])))
    if require_unique_counts:
        assert len({count for _key, count, _color in ordered}) == len(ordered), "class copy counts must be distinct"

    out = np.zeros((len(ordered), max(count for _key, count, _color in ordered)), dtype=int)
    for row, (_key, count, color) in enumerate(ordered):
        out[row, :count] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Group 3x3 sprites by rotation-reflection shape and draw a majority-colored frequency histogram.

    Input structure:
    - The grid has a black background with many separated monochrome 3x3 sprites.
    - Each sprite is one 8-connected component whose 3x3 binary mask may appear in different colors and different orientations.
    - Several sprites that look different by rotation or mirror are actually the same shape class.

    Rule:
    - Extract every 3x3 sprite and ignore its color while comparing shape.
    - Merge sprites whose masks become identical after any rotation or mirror transformation.
    - For each equivalence class, count its copies and also find the unique majority color among that class's instances.
    - Sort classes by copy count from largest to smallest, then draw one left-aligned horizontal bar per class with bar length equal to the class count and bar color equal to that class's majority color.

    Key insight:
    - The answer needs three dependent reductions in order: canonicalize shape under dihedral symmetry, count class frequency for row order and bar length, then count colors inside each class for bar color.
    """
    components = _find_sprite_components(np.asarray(grid, dtype=int))
    assert components, "expected at least one sprite"

    by_shape: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for component in components:
        by_shape[component["canonical_key"]].append(component)
    return _histogram_from_groups(by_shape)


def _enumerate_class_pool() -> list[dict]:
    classes = {}
    for cells in range(4, 7):
        for chosen in combinations(range(_SPRITE_SIZE * _SPRITE_SIZE), cells):
            mask = np.zeros((_SPRITE_SIZE, _SPRITE_SIZE), dtype=bool)
            for idx in chosen:
                mask[idx // _SPRITE_SIZE, idx % _SPRITE_SIZE] = True
            if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if not _bool_connected(mask):
                continue
            variants = _d4_variants(mask)
            if len(variants) < 2:
                continue
            key = _canonical_key(mask)
            if key in classes:
                continue
            classes[key] = {"canonical_key": key, "variants": variants}
    return list(classes.values())


_CLASS_POOL = _enumerate_class_pool()


def _sample_class_counts(num_classes: int) -> list[int]:
    base = np.arange(3, 8)
    picks = _RNG.choice(base, size=num_classes, replace=False)
    return sorted((int(v) for v in picks), reverse=True)


def _distribute_positive(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False).tolist())
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(int(cut - prev))
        prev = cut
    return values


def _sample_color_multiset(count: int, major_color: int, noise_color: int, force_noise: bool) -> list[int]:
    major_count = int(_RNG.integers(count // 2 + 1, count))
    remainder = count - major_count
    assert remainder >= 1

    minority_palette = [int(c) for c in _PALETTE if int(c) != int(major_color)]
    if force_noise:
        chosen_minor = [int(noise_color)]
        extra_needed = min(remainder, int(_RNG.integers(1, min(3, remainder) + 1))) - 1
        available = [c for c in minority_palette if c != int(noise_color)]
        if extra_needed > 0:
            chosen_minor.extend(int(c) for c in _RNG.choice(available, size=extra_needed, replace=False))
    else:
        n_minor_colors = int(_RNG.integers(1, min(3, remainder) + 1))
        chosen_minor = [int(c) for c in _RNG.choice(minority_palette, size=n_minor_colors, replace=False)]

    allocations = _distribute_positive(remainder, len(chosen_minor))
    colors = [major_color] * major_count
    for color, copies in zip(chosen_minor, allocations):
        colors.extend([int(color)] * int(copies))
    _RNG.shuffle(colors)
    assert len(colors) == count
    assert len(set(colors)) >= 2
    return colors


def _sample_layout(total_sprites: int) -> tuple[int, int, list[tuple[int, int]]]:
    choices = []
    for slot_rows in range(3, 8):
        for slot_cols in range(3, 8):
            if slot_rows * slot_cols < total_sprites:
                continue
            base_h = (_STEP * (slot_rows - 1)) + _SPRITE_SIZE
            base_w = (_STEP * (slot_cols - 1)) + _SPRITE_SIZE
            if base_h <= 30 and base_w <= 30:
                choices.append((slot_rows, slot_cols, base_h, base_w))
    slot_rows, slot_cols, base_h, base_w = choices[int(_RNG.integers(len(choices)))]

    extra_h = int(_RNG.integers(0, min(4, 31 - base_h)))
    extra_w = int(_RNG.integers(0, min(4, 31 - base_w)))
    top_margin = int(_RNG.integers(0, extra_h + 1))
    left_margin = int(_RNG.integers(0, extra_w + 1))
    height = base_h + extra_h
    width = base_w + extra_w

    slots = [(top_margin + _STEP * sr, left_margin + _STEP * sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    chosen = _RNG.choice(len(slots), size=total_sprites, replace=False)
    positions = [slots[int(idx)] for idx in chosen]
    return height, width, positions


def _build_grid(records: list[dict]) -> np.ndarray:
    height, width, positions = _sample_layout(len(records))
    grid = np.zeros((height, width), dtype=int)
    order = _RNG.permutation(len(records))
    for idx, pos_idx in enumerate(order):
        record = records[int(pos_idx)]
        r0, c0 = positions[idx]
        patch = np.zeros((_SPRITE_SIZE, _SPRITE_SIZE), dtype=int)
        patch[record["mask"]] = int(record["color"])
        grid[r0 : r0 + _SPRITE_SIZE, c0 : c0 + _SPRITE_SIZE] = patch
    return grid


def _wrong_orientation_output(components: list[dict]) -> np.ndarray:
    groups: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for component in components:
        groups[component["raw_key"]].append(component)
    return _histogram_from_groups(groups, require_unique_majority=False, require_unique_counts=False)


def _wrong_color_output(components: list[dict]) -> np.ndarray:
    groups: dict[int, list[dict]] = defaultdict(list)
    for component in components:
        groups[int(component["color"])].append(component)
    items = sorted(((color, len(entries)) for color, entries in groups.items()), key=lambda item: (-item[1], item[0]))
    out = np.zeros((len(items), max(count for _color, count in items)), dtype=int)
    for row, (color, count) in enumerate(items):
        out[row, :count] = int(color)
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    components = _find_sprite_components(grid)
    if len(components) < 9:
        return False

    by_shape: dict[tuple[tuple[int, ...], ...], list[dict]] = defaultdict(list)
    for component in components:
        by_shape[component["canonical_key"]].append(component)
    if not (3 <= len(by_shape) <= 5):
        return False

    shape_counts = [len(entries) for entries in by_shape.values()]
    if len(set(shape_counts)) != len(shape_counts):
        return False
    if min(shape_counts) < 3:
        return False

    multi_orientation_classes = 0
    has_three_color_class = False
    for entries in by_shape.values():
        orientations = {entry["raw_key"] for entry in entries}
        if len(orientations) >= 2:
            multi_orientation_classes += 1
        colors = Counter(int(entry["color"]) for entry in entries)
        if max(colors.values()) <= len(entries) // 2:
            return False
        if len([color for color, freq in colors.items() if freq == max(colors.values())]) != 1:
            return False
        if len(colors) >= 3:
            has_three_color_class = True
    if multi_orientation_classes < 2 or not has_three_color_class:
        return False

    correct = main(grid)
    if np.array_equal(correct, _wrong_orientation_output(components)):
        return False
    if np.array_equal(correct, _wrong_color_output(components)):
        return False
    if correct.shape[0] != len(by_shape) or correct.shape[1] != max(shape_counts):
        return False
    if len({tuple(row.tolist()) for row in correct}) != correct.shape[0]:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    num_classes = int(_RNG.integers(3, 6))
    chosen_classes = [dict(cls) for cls in _RNG.choice(_CLASS_POOL, size=num_classes, replace=False)]
    counts = _sample_class_counts(num_classes)

    major_colors = [int(c) for c in _RNG.choice(_PALETTE, size=num_classes, replace=False)]
    remaining_colors = [c for c in map(int, _PALETTE) if c not in set(major_colors)]
    noise_color = int(_RNG.choice(remaining_colors))
    noisy_class_ids = set(int(v) for v in _RNG.choice(np.arange(num_classes), size=2, replace=False))

    records: list[dict] = []
    for class_id, (class_info, count, major_color) in enumerate(zip(chosen_classes, counts, major_colors)):
        colors = _sample_color_multiset(count, major_color, noise_color, class_id in noisy_class_ids)
        variants = class_info["variants"]
        orientation_indices = list(_RNG.integers(0, len(variants), size=count))
        if count >= 2:
            orientation_indices[0] = 0
            orientation_indices[1] = 1 % len(variants)
        for color, variant_idx in zip(colors, orientation_indices):
            records.append(
                {
                    "class_id": class_id,
                    "color": int(color),
                    "mask": variants[int(variant_idx)],
                }
            )

    grid = _build_grid(records)
    if not _validate_instance(grid):
        return None

    out = main(grid)
    out_sig = (tuple(map(int, out.shape)), out.tobytes())
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rotation-reflection histogram input")
