import itertools
import random
from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(g[r0, c0]) == bg:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1, c1 = min(rows), min(cols)
            r2, c2 = max(rows), max(cols)
            patch = g[r1 : r2 + 1, c1 : c2 + 1].copy()

            if np.any(patch == bg):
                continue

            components.append(
                {
                    "bbox": (r1, c1, r2, c2),
                    "patch": patch,
                    "shape": tuple(map(int, patch.shape)),
                }
            )

    components.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return components


def _reconstruct_group(patches: list[np.ndarray]) -> dict | None:
    n = len(patches)
    if n < 2:
        return None

    h, w = map(int, patches[0].shape)
    palette = {int(value) for patch in patches for value in patch.ravel()}
    solutions = []

    for occluder in palette:
        occluder_masks = [patch == occluder for patch in patches]
        if not all(mask.any() and not mask.all() for mask in occluder_masks):
            continue
        if len({mask.tobytes() for mask in occluder_masks}) < 2:
            continue

        for orientation_bits in itertools.product((0, 1), repeat=n):
            oriented = [np.rot90(patch, 2) if bit else patch for patch, bit in zip(patches, orientation_bits)]
            reconstructed = np.full((h, w), -1, dtype=int)
            mixed_cells = 0
            valid = True

            for r in range(h):
                for c in range(w):
                    visible = {int(patch[r, c]) for patch in oriented if int(patch[r, c]) != occluder}
                    if len(visible) != 1:
                        valid = False
                        break
                    reconstructed[r, c] = visible.pop()
                    hidden_here = any(int(patch[r, c]) == occluder for patch in oriented)
                    shown_here = any(int(patch[r, c]) != occluder for patch in oriented)
                    if hidden_here and shown_here:
                        mixed_cells += 1
                if not valid:
                    break

            if not valid:
                continue
            if np.any(reconstructed == -1):
                continue
            if occluder in reconstructed:
                continue
            if mixed_cells == 0:
                continue
            if np.array_equal(reconstructed, np.rot90(reconstructed, 2)):
                continue

            rotated_count = int(sum(orientation_bits))
            total_visible = int(sum(np.count_nonzero(patch != occluder) for patch in oriented))
            solutions.append(
                {
                    "output": reconstructed,
                    "occluder": int(occluder),
                    "rotated_count": rotated_count,
                    "mixed_cells": mixed_cells,
                    "total_visible": total_visible,
                }
            )

    if not solutions:
        return None

    solutions.sort(
        key=lambda item: (
            item["rotated_count"],
            -item["mixed_cells"],
            -item["total_visible"],
        )
    )
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge several same-size partial copies, some rotated by 180 degrees, to recover the hidden original pattern.

    Input structure:
    - The background is dominant.
    - Several disconnected filled rectangles share one common size; these are partial copies of the same multicolor pattern.
    - Within each true copy, one shared occluder color hides different subsets of cells; some copies are rotated by 180 degrees.
    - Additional filled rectangles of other sizes may appear as distractors.

    Rule:
    - Find the repeated rectangle size and ignore components with different dimensions.
    - For that size, test each color as the possible occluder and test each copy in both its given orientation and a 180-degree rotation.
    - Keep the configuration where all non-occluded cells agree after alignment and every cell is revealed by at least one copy.
    - Reconstruct the full pattern by taking the revealed color at each aligned cell and return that recovered rectangle.

    Key insight:
    - The shared occluder color means "unknown" rather than a real pattern color, so the answer comes from aligning complementary partial views and choosing the 180-degree orientation that makes their visible cells consistent.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components, "expected at least one foreground component"

    groups: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for comp in components:
        groups[comp["shape"]].append(comp["patch"])

    candidates = []
    for shape, patches in groups.items():
        if len(patches) < 2:
            continue
        solution = _reconstruct_group(patches)
        if solution is None:
            continue
        area = int(shape[0] * shape[1])
        candidates.append(
            {
                "count": len(patches),
                "area": area,
                "solution": solution,
            }
        )

    assert candidates, "no valid repeated copy group found"
    candidates.sort(
        key=lambda item: (
            -item["count"],
            -item["area"],
            item["solution"]["rotated_count"],
            -item["solution"]["mixed_cells"],
        )
    )
    return candidates[0]["solution"]["output"]


_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 500


def _structured_pattern(height: int, width: int, colors: list[int]) -> np.ndarray:
    while True:
        base = np.full((height, width), colors[0], dtype=int)

        row_band = random.randint(0, height - 1)
        base[row_band : row_band + 1, :] = colors[1]

        col_band = random.randint(0, width - 1)
        base[:, col_band : col_band + 1] = colors[2]

        if len(colors) >= 4:
            block_h = random.randint(1, max(1, height - 1))
            block_w = random.randint(1, max(1, width - 1))
            r0 = random.randint(0, height - block_h)
            c0 = random.randint(0, width - block_w)
            base[r0 : r0 + block_h, c0 : c0 + block_w] = colors[3]

        accent_color = random.choice(colors[1:])
        accent_r = random.randint(0, height - 1)
        accent_c = random.randint(0, width - 1)
        base[accent_r, accent_c] = accent_color

        if len(set(map(int, base.ravel()))) < 3:
            continue
        if np.array_equal(base, np.rot90(base, 2)):
            continue
        return base


def _sample_occlusion_masks(n_copies: int, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    area = height * width

    while True:
        hidden = np.zeros((n_copies, height, width), dtype=bool)
        for r in range(height):
            for c in range(width):
                hidden_count = random.randint(1, n_copies - 1)
                hidden_indices = random.sample(range(n_copies), hidden_count)
                for idx in hidden_indices:
                    hidden[idx, r, c] = True

        if any(mask.all() or not mask.any() for mask in hidden):
            continue
        if len({mask.tobytes() for mask in hidden}) < n_copies:
            continue
        hidden_cells = [int(mask.sum()) for mask in hidden]
        if min(hidden_cells) < max(1, area // 5):
            continue
        if max(hidden_cells) > area - 1:
            continue
        return hidden


def _distractor_patch(height: int, width: int, colors: list[int]) -> np.ndarray:
    patch = np.full((height, width), colors[0], dtype=int)
    if height > 1:
        patch[height // 2, :] = colors[1]
    if width > 1:
        patch[:, width // 2] = colors[2]
    if len(colors) >= 4:
        patch[0, :] = colors[3]
    return patch


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int = 1) -> bool:
    gh, gw = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    if r0 < 0 or c0 < 0 or r0 + ph > gh or c0 + pw > gw:
        return False
    r1 = max(0, r0 - padding)
    c1 = max(0, c0 - padding)
    r2 = min(gh, r0 + ph + padding)
    c2 = min(gw, c0 + pw + padding)
    return bool(np.all(grid[r1:r2, c1:c2] == bg))


def _place_patch(grid: np.ndarray, patch: np.ndarray, bg: int, padding: int = 1) -> bool:
    gh, gw = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    locations = [
        (r, c)
        for r in range(gh - ph + 1)
        for c in range(gw - pw + 1)
        if _can_place(grid, patch, r, c, bg, padding=padding)
    ]
    if not locations:
        return False
    r0, c0 = random.choice(locations)
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    return True


def _build_input() -> tuple[np.ndarray, np.ndarray] | None:
    n_copies = random.choice((3, 4))
    pattern_h = random.randint(4, 6)
    pattern_w = random.randint(4, 7)

    bg, occluder = random.sample(_PALETTE, 2)
    pattern_palette = random.sample([c for c in _PALETTE if c not in {bg, occluder}], random.randint(3, 4))
    target = _structured_pattern(pattern_h, pattern_w, pattern_palette)

    hidden_masks = _sample_occlusion_masks(n_copies, target.shape)

    rotated_count = 1 if random.random() < 0.75 else 0
    rotated_indices = set(random.sample(range(n_copies), rotated_count))

    patches = []
    for idx in range(n_copies):
        patch = target.copy()
        patch[hidden_masks[idx]] = occluder
        if idx in rotated_indices:
            patch = np.rot90(patch, 2)
        patches.append(patch)

    used_shapes = {tuple(map(int, target.shape))}
    distractors = []
    n_distractors = random.randint(0, 2)
    distractor_palette = [c for c in _PALETTE if c != bg]
    for _ in range(n_distractors):
        for _attempt in range(50):
            dh = random.randint(2, 7)
            dw = random.randint(2, 7)
            shape = (dh, dw)
            if shape in used_shapes:
                continue
            used_shapes.add(shape)
            colors = random.sample(distractor_palette, min(4, len(distractor_palette)))
            distractors.append(_distractor_patch(dh, dw, colors))
            break

    all_patches = patches + distractors
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in all_patches)
    min_h = max(16, pattern_h * 2 + 4)
    min_w = max(16, pattern_w * 2 + 6)
    max_h = 30
    max_w = 30

    for _ in range(80):
        grid_h = random.randint(min_h, max_h)
        grid_w = random.randint(min_w, max_w)
        if grid_h * grid_w <= total_area * 2:
            continue
        grid = np.full((grid_h, grid_w), bg, dtype=int)

        order = all_patches[:]
        random.shuffle(order)
        placed = True
        for patch in order:
            if not _place_patch(grid, patch, bg, padding=1):
                placed = False
                break
        if not placed:
            continue

        counts = Counter(map(int, grid.ravel()))
        if counts[bg] <= max(count for color, count in counts.items() if color != bg):
            continue
        return grid, target

    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        built = _build_input()
        if built is None:
            continue
        grid, expected = built
        try:
            solved = main(grid)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError(f"generate_input failed after {_MAX_ATTEMPTS} attempts")
