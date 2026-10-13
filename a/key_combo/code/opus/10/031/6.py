from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)

GRID_SIDE_RANGE = (19, 20)
COLOR_COUNT_CHOICES = (3, 4)

MIN_CLEAN_H = 3
MAX_CLEAN_H = 5
MIN_CLEAN_W = 3
MAX_CLEAN_W = 5
MIN_CLEAN_COUNT = 6
MAX_CLEAN_COUNT = 14

LOCAL_PAD = 1
MAX_NOISE_PER_COLOR = 6
BEST_AXIS_GAP_MIN = 2
GEN_ATTEMPTS = 600
SHAPE_ATTEMPTS = 240
NOISE_ATTEMPTS = 120


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _axis_score(coords_set: set[tuple[int, int]], width: int, k: int) -> int:
    score = 0
    for r, c in coords_set:
        mirror_c = k - c
        if 0 <= mirror_c < width and (r, mirror_c) in coords_set:
            score += 1
    return score


def _best_vertical_axis(coords: Iterable[tuple[int, int]], width: int) -> tuple[int, int, int]:
    pts = {tuple(map(int, coord)) for coord in coords}
    best_k = 0
    best_score = -1
    second_best = -1
    for k in range(0, 2 * (width - 1) + 1):
        score = _axis_score(pts, width, k)
        if score > best_score:
            second_best = best_score
            best_score = score
            best_k = k
        elif score > second_best:
            second_best = score
    return int(best_k), int(best_score), int(second_best)


def _denoise_mask(mask: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    coords = [tuple(x) for x in np.argwhere(mask)]
    if not coords:
        return np.zeros_like(mask, dtype=bool), 0, 0, 0
    width = mask.shape[1]
    best_k, best_score, second_best = _best_vertical_axis(coords, width)
    pts = set(coords)
    clean = np.zeros_like(mask, dtype=bool)
    for r, c in coords:
        mirror_c = best_k - c
        if 0 <= mirror_c < width and (r, mirror_c) in pts:
            clean[r, c] = True
    return clean, int(best_k), int(best_score), int(second_best)


def _extract_layers(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    layers = []
    for color in colors:
        raw_mask = g == color
        clean_mask, best_k, best_score, second_best = _denoise_mask(raw_mask)
        box = _bbox(clean_mask)
        if box is None:
            continue
        patch = np.zeros((box[2] - box[0] + 1, box[3] - box[1] + 1), dtype=int)
        submask = clean_mask[box[0] : box[2] + 1, box[1] : box[3] + 1]
        patch[submask] = color
        layers.append(
            {
                "color": color,
                "raw_count": int(raw_mask.sum()),
                "clean_count": int(clean_mask.sum()),
                "best_k": int(best_k),
                "best_score": int(best_score),
                "second_best": int(second_best),
                "bbox": box,
                "patch": patch,
            }
        )
    return layers


def _pack_patches(patches: list[np.ndarray]) -> np.ndarray:
    if not patches:
        return np.zeros((1, 1), dtype=int)
    height = max(int(p.shape[0]) for p in patches)
    width = sum(int(p.shape[1]) for p in patches) + max(0, len(patches) - 1)
    out = np.zeros((height, width), dtype=int)
    c0 = 0
    for idx, patch in enumerate(patches):
        h, w = patch.shape
        out[:h, c0 : c0 + w] = patch
        c0 += w
        if idx != len(patches) - 1:
            c0 += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise each color layer by vertical symmetry, then reorder the cropped layers by their cleaned frequencies.

    Input structure:
    - The input is a black-background grid containing 3-4 non-black colors.
    - Each color occupies a separate nearly vertically symmetric layer with its own symmetry axis.
    - Every layer also includes some same-color noise pixels that lack a mirrored partner across that layer's axis.

    Rule:
    - For each non-black color independently, search all vertical axes and choose the axis that preserves the most same-color mirror pairings.
    - Delete every pixel of that color that has no same-color mirror counterpart across the chosen axis.
    - Crop each cleaned color layer to its tight bounding box.
    - Pack the cropped layers left-to-right in descending order of cleaned pixel count, inserting one black spacer column between adjacent crops and top-aligning all crops.

    Key insight:
    - The ordering is driven by the side effect of denoising: noise removal changes the per-color pixel counts, so counting must happen after symmetry cleanup rather than before it.
    """
    layers = _extract_layers(np.asarray(grid, dtype=int))
    layers.sort(key=lambda layer: (-layer["clean_count"], layer["bbox"][0], layer["bbox"][1], layer["bbox"][2], layer["bbox"][3]))
    return _pack_patches([layer["patch"] for layer in layers])


def _full_slots(height: int, width: int) -> list[tuple[int, int, int, int]]:
    mid_r = height // 2
    mid_c = width // 2
    return [
        (1, 1, mid_r - 2, mid_c - 2),
        (1, mid_c + 1, mid_r - 2, width - 2),
        (mid_r + 1, 1, height - 2, mid_c - 2),
        (mid_r + 1, mid_c + 1, height - 2, width - 2),
    ]


def _count_sequence(count: int) -> list[int]:
    counts = [count]
    while len(counts) < random.choice(COLOR_COUNT_CHOICES):
        break
    return counts


def _sample_clean_counts(n_colors: int) -> list[int]:
    for _ in range(200):
        top = random.randint(11, MAX_CLEAN_COUNT)
        counts = [top]
        for _ in range(n_colors - 1):
            counts.append(counts[-1] - random.randint(1, 3))
        if min(counts) < MIN_CLEAN_COUNT:
            continue
        if len(set(counts)) != n_colors:
            continue
        sorted_counts = sorted(counts, reverse=True)
        gaps = [sorted_counts[i] - sorted_counts[i + 1] for i in range(n_colors - 1)]
        if min(gaps) <= 3:
            random.shuffle(counts)
            return counts
    raise RuntimeError("failed to sample clean counts")


def _shape_specs(target_count: int) -> list[tuple[int, int, int]]:
    specs = []
    for h in range(MIN_CLEAN_H, MAX_CLEAN_H + 1):
        for w in range(MIN_CLEAN_W, MAX_CLEAN_W + 1):
            area = h * w
            if not (0.35 <= target_count / area <= 0.9):
                continue
            side_w = w // 2
            if w % 2 == 0:
                if target_count % 2 != 0:
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    specs.append((h, w, 0))
            else:
                for center_count in range(0, h + 1):
                    if (target_count - center_count) < 0 or (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        specs.append((h, w, center_count))
    return specs


def _left_half_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    cols = [random.randrange(side_w) for _ in range(h)]
    cols[random.randrange(h)] = 0
    if side_w > 1:
        cols[random.randrange(h)] = side_w - 1
    cells = {(r, cols[r]) for r in range(h)}
    all_cells = [(r, c) for r in range(h) for c in range(side_w)]
    while len(cells) < left_count:
        cells.add(random.choice(all_cells))
    return cells


def _build_clean_sprite(color: int, target_count: int) -> np.ndarray:
    specs = _shape_specs(target_count)
    if not specs:
        raise ValueError(f"no shape specs for count {target_count}")
    for _ in range(SHAPE_ATTEMPTS):
        h, w, center_count = random.choice(specs)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_half_cells(h, side_w, left_count)
        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            sprite[r, c] = color
            sprite[r, w - 1 - c] = color
        if center_count:
            center_rows = random.sample(range(h), center_count)
            sprite[center_rows, side_w] = color
        box = _bbox(sprite == color)
        if box != (0, 0, h - 1, w - 1):
            continue
        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        best_k, best_score, second_best = _best_vertical_axis(coords, w)
        if best_k != w - 1:
            continue
        if best_score != target_count:
            continue
        if best_score - second_best < BEST_AXIS_GAP_MIN:
            continue
        return sprite
    raise RuntimeError("failed to build clean sprite")


def _embed_sprite(sprite: np.ndarray, color: int) -> tuple[np.ndarray, int]:
    h, w = sprite.shape
    canvas = np.zeros((h + 2 * LOCAL_PAD, w + 2 * LOCAL_PAD), dtype=int)
    canvas[LOCAL_PAD : LOCAL_PAD + h, LOCAL_PAD : LOCAL_PAD + w] = sprite
    true_k = (w - 1) + 2 * LOCAL_PAD
    clean_mask = canvas == color
    clean, best_k, best_score, second_best = _denoise_mask(clean_mask)
    assert best_k == true_k
    assert best_score == int(clean_mask.sum())
    assert int(clean.sum()) == int(clean_mask.sum())
    assert best_score - second_best >= BEST_AXIS_GAP_MIN
    return canvas, int(true_k)


def _noise_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    adj = np.zeros_like(clean_mask, dtype=bool)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    width = clean_mask.shape[1]
    cols = np.arange(width)
    mirror_cols = true_k - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False
    valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
    for c in np.where(valid)[0]:
        adj[:, c] &= ~clean_mask[:, mirror_cols[c]]

    rows, cols = np.where(adj)
    return [(int(r), int(c)) for r, c in zip(rows, cols)]


def _add_noise(canvas: np.ndarray, color: int, true_k: int, noise_count: int) -> np.ndarray | None:
    clean_mask = canvas == color
    clean_coords = [tuple(x) for x in np.argwhere(clean_mask)]
    candidates = _noise_candidates(clean_mask, true_k)
    if len(candidates) < noise_count:
        return None

    width = canvas.shape[1]
    current_coords = list(clean_coords)
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()

    for _ in range(noise_count):
        viable = [cell for cell in candidates if cell not in chosen and cell not in forbidden]
        if not viable:
            return None
        scored = []
        for cell in viable:
            _, _, alt_second = _best_vertical_axis(current_coords + [cell], width)
            scored.append((alt_second, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        threshold = scored[0][0] + 1
        pool = [cell for alt_second, _, cell in scored if alt_second <= threshold]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        mirror = (cell[0], true_k - cell[1])
        forbidden.add(cell)
        if 0 <= mirror[1] < width:
            forbidden.add(mirror)

    best_k, best_score, second_best = _best_vertical_axis(current_coords, width)
    if best_k != true_k:
        return None
    if best_score != len(clean_coords):
        return None
    if best_score - second_best < BEST_AXIS_GAP_MIN:
        return None

    noisy = np.array(canvas, dtype=int)
    for r, c in chosen:
        noisy[r, c] = color
    recovered, _, _, _ = _denoise_mask(noisy == color)
    if not np.array_equal(recovered, clean_mask):
        return None
    return noisy


def _sample_noise_counts(clean_counts: list[int], capacities: list[int]) -> list[int] | None:
    if any(capacity < 1 for capacity in capacities):
        return None
    for _ in range(400):
        noise_counts = [random.randint(1, min(capacity, MAX_NOISE_PER_COLOR)) for capacity in capacities]
        noisy_counts = [clean + noise for clean, noise in zip(clean_counts, noise_counts)]
        if len(set(noisy_counts)) != len(noisy_counts):
            continue
        clean_order = sorted(range(len(clean_counts)), key=lambda i: -clean_counts[i])
        noisy_order = sorted(range(len(noisy_counts)), key=lambda i: -noisy_counts[i])
        if clean_order == noisy_order:
            continue
        if max(noise_counts) - min(noise_counts) < 2:
            continue
        reversed_pair = False
        for i in range(len(clean_counts)):
            for j in range(i + 1, len(clean_counts)):
                if (clean_counts[i] - clean_counts[j]) * (noisy_counts[i] - noisy_counts[j]) < 0:
                    reversed_pair = True
                    break
            if reversed_pair:
                break
        if reversed_pair:
            return noise_counts
    return None


def _place_object(grid: np.ndarray, patch: np.ndarray, slot: tuple[int, int, int, int]) -> tuple[int, int] | None:
    r0s, c0s, r1s, c1s = slot
    h, w = patch.shape
    if h > (r1s - r0s + 1) or w > (c1s - c0s + 1):
        return None
    r0 = random.randint(r0s, r1s - h + 1)
    c0 = random.randint(c0s, c1s - w + 1)
    target = grid[r0 : r0 + h, c0 : c0 + w]
    if np.any((target != BG) & (patch != BG)):
        return None
    target[patch != BG] = patch[patch != BG]
    return int(r0), int(c0)


def _expected_output_from_specs(specs: list[dict]) -> np.ndarray:
    ordered = sorted(specs, key=lambda spec: (-spec["clean_count"], spec["slot_index"]))
    return _pack_patches([spec["clean_patch"] for spec in ordered])


def _validate_case(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    layers = _extract_layers(inp)
    if len(layers) not in COLOR_COUNT_CHOICES:
        return False

    clean_counts = [layer["clean_count"] for layer in layers]
    raw_counts = [layer["raw_count"] for layer in layers]
    if len(set(clean_counts)) != len(clean_counts):
        return False
    if any(raw <= clean for raw, clean in zip(raw_counts, clean_counts)):
        return False
    if len({layer["best_k"] for layer in layers}) != len(layers):
        return False
    if any(layer["best_score"] - layer["second_best"] < BEST_AXIS_GAP_MIN for layer in layers):
        return False

    clean_order = sorted(range(len(layers)), key=lambda i: -clean_counts[i])
    raw_order = sorted(range(len(layers)), key=lambda i: -raw_counts[i])
    if clean_order == raw_order:
        return False

    has_reversal = False
    for i in range(len(layers)):
        for j in range(i + 1, len(layers)):
            if (clean_counts[i] - clean_counts[j]) * (raw_counts[i] - raw_counts[j]) < 0:
                has_reversal = True
                break
        if has_reversal:
            break
    if not has_reversal:
        return False

    semantic_expected = _pack_patches(
        [
            layer["patch"]
            for layer in sorted(
                layers,
                key=lambda layer: (-layer["clean_count"], layer["bbox"][0], layer["bbox"][1], layer["bbox"][2], layer["bbox"][3]),
            )
        ]
    )
    if not np.array_equal(semantic_expected, expected):
        return False
    if np.array_equal(_pack_patches([layer["patch"] for layer in sorted(layers, key=lambda layer: -layer["raw_count"])]), expected):
        return False
    return np.array_equal(main(inp), expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    n_colors = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, n_colors)
    clean_counts = _sample_clean_counts(n_colors)

    specs = []
    capacities = []
    for color, clean_count in zip(colors, clean_counts):
        try:
            sprite = _build_clean_sprite(color, clean_count)
        except RuntimeError:
            return None
        local_clean, true_k = _embed_sprite(sprite, color)
        capacity = len(_noise_candidates(local_clean == color, true_k))
        if capacity < 1:
            return None
        clean_patch = np.array(sprite, dtype=int)
        specs.append(
            {
                "color": color,
                "clean_count": clean_count,
                "clean_patch": clean_patch,
                "local_clean": local_clean,
                "true_k": true_k,
            }
        )
        capacities.append(capacity)

    noise_counts = _sample_noise_counts(clean_counts, capacities)
    if noise_counts is None:
        return None

    for spec, noise_count in zip(specs, noise_counts):
        noisy = None
        for _ in range(NOISE_ATTEMPTS):
            noisy = _add_noise(spec["local_clean"], spec["color"], spec["true_k"], noise_count)
            if noisy is not None:
                break
        if noisy is None:
            return None
        spec["local_noisy"] = noisy
        spec["noise_count"] = noise_count
        spec["raw_count"] = spec["clean_count"] + noise_count

    all_slots = _full_slots(height, width)
    chosen_slot_indices = sorted(random.sample(range(4), n_colors))
    grid = np.zeros((height, width), dtype=int)
    axes = []
    for spec, slot_index in zip(specs, chosen_slot_indices):
        placed = _place_object(grid, spec["local_noisy"], all_slots[slot_index])
        if placed is None:
            return None
        spec["slot_index"] = slot_index
        spec["grid_pos"] = placed
        spec["global_axis_k"] = spec["true_k"] + 2 * placed[1]
        axes.append(spec["global_axis_k"])

    if len(set(axes)) != len(axes):
        return None

    expected = _expected_output_from_specs(specs)
    if max(expected.shape) > 30:
        return None
    if not _validate_case(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
