from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
USABLE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
FRAME_COLOR = 5
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _orbit_groups(size: int) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for r in range(size):
        for c in range(size):
            orbit = {
                (int(r), int(c)),
                (int(c), int(size - 1 - r)),
                (int(size - 1 - r), int(size - 1 - c)),
                (int(size - 1 - c), int(r)),
            }
            key = tuple(sorted(orbit))
            if key in seen:
                continue
            seen.add(key)
            groups.append(list(key))
    return groups


def _repair_c4(core: np.ndarray) -> tuple[np.ndarray, Counter]:
    core = np.asarray(core, dtype=int)
    n, m = map(int, core.shape)
    assert n == m and n >= 1
    out = core.copy()
    defect_colors: Counter = Counter()

    for orbit in _orbit_groups(n):
        values = [int(core[r, c]) for r, c in orbit]
        counts = Counter(values).most_common()
        correct = int(counts[0][0])
        if len(counts) > 1:
            assert int(counts[0][1]) > int(counts[1][1]), "rotation orbit must have a unique majority color"
        for r, c in orbit:
            if int(core[r, c]) != correct:
                defect_colors[int(core[r, c])] += 1
                out[r, c] = correct

    return out, defect_colors


def _find_frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    candidates: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components(g == color):
            r0, c0, r1, c1 = _bbox(coords)
            side_h = int(r1 - r0 + 1)
            side_w = int(c1 - c0 + 1)
            if side_h != side_w or side_h < 5:
                continue
            mask = np.zeros((side_h, side_w), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            border = np.zeros_like(mask)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not bool(np.array_equal(mask, border)):
                continue
            candidates.append(
                {
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color": int(color),
                    "area": int(len(coords)),
                }
            )

    # Ignore accidental ring-shaped components inside a larger frame.
    outer: list[dict] = []
    for cand in sorted(candidates, key=lambda item: item["area"], reverse=True):
        r0, c0, r1, c1 = cand["bbox"]
        if any(
            rr0 <= r0 and cc0 <= c0 and rr1 >= r1 and cc1 >= c1
            for rr0, cc0, rr1, cc1 in (other["bbox"] for other in outer)
        ):
            continue
        outer.append(cand)

    return sorted(outer, key=lambda item: (item["bbox"][0], item["bbox"][1]))


def _find_palette_blocks(grid: np.ndarray, bg: int, frame_mask: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    outside = ~np.asarray(frame_mask, dtype=bool)
    for color in np.unique(g):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _components((g == color) & outside):
            coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
            blocks.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                }
            )
    return sorted(blocks, key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair C4-symmetry defects inside square frames and consume the matching palette colors.

    Input structure:
    - Black background occupies most of the canvas.
    - Small monochrome palette blocks sit on the left, outside every frame.
    - Grey hollow square frames on the right enclose color patterns that should be
      invariant under 90-degree rotation.
    - Each wrong pixel uses a palette color; after repair, that palette block loses
      one cell while keeping its top-left anchor.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flat)).most_common(1)[0][0])
    out = g.copy()

    frames = _find_frame_candidates(g, bg)
    frame_mask = np.zeros_like(g, dtype=bool)
    defect_counts: Counter = Counter()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        core = g[r0 + 1 : r1, c0 + 1 : c1]
        repaired, defects = _repair_c4(core)
        out[r0 + 1 : r1, c0 + 1 : c1] = repaired
        defect_counts.update(defects)

    palette_blocks = _find_palette_blocks(g, bg, frame_mask)
    for block in palette_blocks:
        for r, c in block["coords"]:
            out[int(r), int(c)] = bg

    remaining_need = Counter(defect_counts)
    for block in palette_blocks:
        color = int(block["color"])
        coords = block["coords"]
        need = int(remaining_need[color])
        keep = max(0, int(len(coords)) - need)
        remaining_need[color] = max(0, need - int(len(coords)))
        for r, c in coords[:keep]:
            out[int(r), int(c)] = color

    return out


def _make_symmetric_core(size: int, colors: list[int], rng: np.random.Generator) -> np.ndarray:
    orbits = _orbit_groups(size)
    for _ in range(100):
        picks = [int(colors[int(rng.integers(0, len(colors)))]) for _ in orbits]
        if len(orbits) > 1 and len(set(picks)) < 2:
            continue
        core = np.zeros((size, size), dtype=int)
        for color, orbit in zip(picks, orbits):
            for r, c in orbit:
                core[int(r), int(c)] = int(color)
        return core
    raise RuntimeError("failed to create a non-degenerate symmetric core")


def _build_frame_patch(core: np.ndarray, border_color: int) -> np.ndarray:
    size = int(core.shape[0])
    patch = np.full((size + 2, size + 2), int(border_color), dtype=int)
    patch[1:-1, 1:-1] = core
    return patch


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = map(int, patch.shape)
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _sample_palette_usage(total_defects: int, palette_size: int, rng: np.random.Generator) -> list[int]:
    slots = np.array([idx for idx in range(palette_size) for _ in range(2)], dtype=int)
    chosen = rng.choice(slots, size=total_defects, replace=False)
    return [int(x) for x in chosen]


def _generate_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(rng.integers(1, 5))
    inner_sizes = [int(rng.integers(3, 8)) for _ in range(n_frames)]
    orbit_groups = [_orbit_groups(size) for size in inner_sizes]
    eligible_orbits = [[orbit for orbit in groups if len(orbit) == 4] for groups in orbit_groups]
    if any(not orbit_list for orbit_list in eligible_orbits):
        raise RuntimeError("every generated frame must have at least one 4-cycle")

    defects_per_frame = [int(rng.integers(1, min(3, len(orbits)) + 1)) for orbits in eligible_orbits]
    total_defects = int(sum(defects_per_frame))

    min_palette = max(3, (total_defects + 1) // 2)
    max_palette = min(6, len(USABLE_COLORS) - 2)
    if min_palette > max_palette:
        raise RuntimeError("not enough visible colors for a disjoint palette/pattern split")
    palette_size = int(rng.integers(min_palette, max_palette + 1))

    palette_colors = [int(x) for x in rng.choice(np.array(USABLE_COLORS), size=palette_size, replace=False)]
    pattern_pool = [color for color in USABLE_COLORS if color not in palette_colors]
    pattern_count = int(rng.integers(2, min(4, len(pattern_pool)) + 1))
    pattern_colors = [int(x) for x in rng.choice(np.array(pattern_pool), size=pattern_count, replace=False)]

    clean_cores = [_make_symmetric_core(size, pattern_colors, rng) for size in inner_sizes]
    corrupted_cores = [core.copy() for core in clean_cores]

    defect_slots: list[tuple[int, int, int]] = []
    for frame_idx, (orbits, n_defects) in enumerate(zip(eligible_orbits, defects_per_frame)):
        chosen_orbits = rng.choice(len(orbits), size=n_defects, replace=False)
        for orbit_idx in chosen_orbits:
            orbit = orbits[int(orbit_idx)]
            pos = orbit[int(rng.integers(0, len(orbit)))]
            defect_slots.append((int(frame_idx), int(pos[0]), int(pos[1])))

    rng.shuffle(defect_slots)
    palette_indices = _sample_palette_usage(total_defects, palette_size, rng)
    defect_counts: Counter = Counter()
    for (frame_idx, r, c), palette_idx in zip(defect_slots, palette_indices):
        wrong = int(palette_colors[int(palette_idx)])
        corrupted_cores[frame_idx][int(r), int(c)] = wrong
        defect_counts[wrong] += 1

    frame_patches_in = [_build_frame_patch(core, FRAME_COLOR) for core in corrupted_cores]
    frame_patches_out = [_build_frame_patch(core, FRAME_COLOR) for core in clean_cores]

    palette_specs: list[dict] = []
    palette_top = int(rng.integers(1, 4))
    palette_anchor = int(rng.integers(1, 3))
    cursor = int(palette_top)
    palette_right = 0
    for color in palette_colors:
        horizontal = bool(rng.integers(0, 2))
        height, width = (1, 2) if horizontal else (2, 1)
        x = int(palette_anchor if horizontal else palette_anchor + int(rng.integers(0, 2)))
        coords = [(int(cursor + dr), int(x + dc)) for dr in range(height) for dc in range(width)]
        palette_right = max(palette_right, max(c for _, c in coords))
        palette_specs.append({"color": int(color), "coords": coords})
        cursor += int(height + rng.integers(1, 3))
    palette_bottom = int(cursor - 1)

    n_cols = 1 if n_frames == 1 else int(rng.integers(1, 3))
    row_gap = int(rng.integers(2, 5))
    col_gap = int(rng.integers(2, 5))
    frames_top = int(rng.integers(1, 4))
    frames_left = int(palette_right + rng.integers(3, 6))

    frame_positions: list[tuple[int, int]] = [(0, 0)] * n_frames
    y = int(frames_top)
    for start in range(0, n_frames, n_cols):
        row_ids = list(range(start, min(n_frames, start + n_cols)))
        row_height = max(int(frame_patches_in[idx].shape[0]) for idx in row_ids)
        x = int(frames_left)
        for idx in row_ids:
            patch = frame_patches_in[idx]
            frame_positions[idx] = (int(y), int(x))
            x += int(patch.shape[1] + col_gap)
        y += int(row_height + row_gap)

    frame_bottom = max(int(r + patch.shape[0] - 1) for (r, _), patch in zip(frame_positions, frame_patches_in))
    frame_right = max(int(c + patch.shape[1] - 1) for (_, c), patch in zip(frame_positions, frame_patches_in))

    height = int(max(palette_bottom, frame_bottom) + rng.integers(2, 5))
    width = int(max(palette_right, frame_right) + rng.integers(2, 5))

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    for spec in palette_specs:
        color = int(spec["color"])
        coords = sorted(spec["coords"])
        used = int(defect_counts[color])
        keep = max(0, len(coords) - used)
        for r, c in coords:
            inp[int(r), int(c)] = color
        for r, c in coords[:keep]:
            out[int(r), int(c)] = color

    for (r0, c0), patch_in, patch_out in zip(frame_positions, frame_patches_in, frame_patches_out):
        _paste(inp, patch_in, int(r0), int(c0))
        _paste(out, patch_out, int(r0), int(c0))

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        try:
            inp, expected = _generate_case(rng)
        except RuntimeError:
            continue
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
