from __future__ import annotations

from collections import Counter
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_arc_utils():
    try:
        from arc_utils import Color, find_components, label_mask

        return Color, find_components, label_mask
    except ModuleNotFoundError:
        arc_utils_path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
        spec = spec_from_file_location("arc_utils_fallback", arc_utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load arc_utils from {arc_utils_path}")
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Color, module.find_components, module.label_mask


Color, find_components, label_mask = _load_arc_utils()

BLACK = int(Color.BLACK)
GREY = int(Color.GREY)
INPUT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
ORTHO = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _input_cross_coords(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    coords = {(r, c)}
    for dr, dc in ORTHO:
        coords.add((r + dr, c + dc))
    return coords


def _bbox_coords(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _expanded_mask_coords(center: tuple[int, int], shape: tuple[int, int] | None = None) -> set[tuple[int, int]]:
    r, c = center
    coords = {(r, c)}
    for dr, dc in ORTHO:
        coords.add((r + dr, c + dc))
        coords.add((r + 2 * dr, c + 2 * dc))
    for dr, dc in DIAG:
        coords.add((r + dr, c + dc))
    if shape is None:
        return coords
    return {(rr, cc) for rr, cc in coords if _in_bounds(shape, rr, cc)}


def _touches4(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    bset = set(b)
    for r, c in a:
        if (r, c) in bset:
            return True
        for dr, dc in ORTHO:
            if (r + dr, c + dc) in bset:
                return True
    return False


def _bbox_conflict(center_a: tuple[int, int], center_b: tuple[int, int]) -> bool:
    return _touches4(_bbox_coords(center_a), _bbox_coords(center_b))


def _expanded_connected(center_a: tuple[int, int], center_b: tuple[int, int], shape: tuple[int, int] | None = None) -> bool:
    return _touches4(_expanded_mask_coords(center_a, shape), _expanded_mask_coords(center_b, shape))


def _detect_crosses(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    crosses = []
    for comp in components:
        if int(comp["area"]) != 5:
            raise AssertionError("each foreground object must be a 5-pixel cross")
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
            raise AssertionError("each cross must fit a 3x3 bbox")

        center = (r0 + 1, c0 + 1)
        coords = {tuple(map(int, coord)) for coord in comp["coords"]}
        expected = _input_cross_coords(center)
        if coords != expected:
            raise AssertionError("foreground object is not the expected plus shape")

        arm_positions = [(center[0] + dr, center[1] + dc) for dr, dc in ORTHO]
        arm_colors = {int(grid[r, c]) for r, c in arm_positions}
        if len(arm_colors) != 1:
            raise AssertionError("all four arm pixels must share one color")
        arm_color = next(iter(arm_colors))
        center_color = int(grid[center])
        if arm_color == BLACK or center_color == BLACK or arm_color == center_color:
            raise AssertionError("cross needs one non-black arm color and a different non-black center color")

        for dr, dc in DIAG:
            rr, cc = center[0] + dr, center[1] + dc
            if int(grid[rr, cc]) != BLACK:
                raise AssertionError("cross diagonals must stay black in the input")

        crosses.append(
            {
                "center": center,
                "arm_color": int(arm_color),
                "center_color": int(center_color),
            }
        )
    if not crosses:
        raise AssertionError("expected at least one cross")
    return sorted(crosses, key=lambda item: item["center"])


def _expanded_pixels(cross: dict, shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    r, c = cross["center"]
    pixels: dict[tuple[int, int], int] = {}
    for dr, dc in ORTHO:
        for scale in (1, 2):
            rr, cc = r + dr * scale, c + dc * scale
            if _in_bounds(shape, rr, cc):
                pixels[(rr, cc)] = int(cross["arm_color"])
    for dr, dc in DIAG:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            pixels[(rr, cc)] = int(cross["center_color"])
    if _in_bounds(shape, r, c):
        pixels[(r, c)] = int(cross["center_color"])
    return pixels


def _component_sizes_from_centers(centers: list[tuple[int, int]], shape: tuple[int, int]) -> list[int]:
    mask = np.zeros(shape, dtype=bool)
    for center in centers:
        for r, c in _expanded_mask_coords(center, shape):
            mask[r, c] = True
    labels, _ = label_mask(mask, connectivity=4)
    counts = Counter(int(labels[r, c]) for r, c in centers)
    return sorted(counts.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each two-color cross, then grey out the crosses that stay isolated after expansion.

    Input structure:
    - The grid has a black background with several separated 3x3 crosses.
    - Each cross is a 4-connected two-color object: four arm pixels share one color and the center pixel uses a different color.
    - Some cross centers are aligned four cells apart, so the expanded motifs can meet even though the original crosses do not touch.

    Rule:
    - Detect every 3x3 cross and expand it into a 5x5 motif: the arm color fills the orthogonal radius-2 plus, while the center color fills the center and the four radius-1 diagonals.
    - After all crosses have been expanded, build 4-connected components over every non-black pixel, ignoring color differences.
    - If an expanded component contains the centers of two or more original crosses, keep every expanded pixel in that component unchanged.
    - If an expanded component contains exactly one original cross center, recolor that cross's full expanded motif to grey.

    Key insight:
    - The connectivity test happens on the expanded motifs, not on the original 3x3 crosses. Expansion changes the topology, and only the motifs that remain singleton components are greyed out.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    crosses = _detect_crosses(g)
    expanded = np.zeros_like(g, dtype=int)
    footprints: list[dict[tuple[int, int], int]] = []

    for cross in crosses:
        pixels = _expanded_pixels(cross, g.shape)
        footprints.append(pixels)
        for (r, c), color in pixels.items():
            prev = int(expanded[r, c])
            if prev in (BLACK, int(color)):
                expanded[r, c] = int(color)

    labels, _ = label_mask(expanded != BLACK, connectivity=4)
    component_sizes = Counter(int(labels[cross["center"]]) for cross in crosses)

    out = expanded.copy()
    for cross, pixels in zip(crosses, footprints):
        component_id = int(labels[cross["center"]])
        if component_sizes[component_id] == 1:
            for r, c in pixels:
                out[r, c] = GREY
    return out


def _choose_group_counts(rng: np.random.Generator) -> tuple[int, int, int]:
    total = int(rng.integers(3, 7))
    if total == 3:
        primary_cluster = 2
    else:
        primary_cluster = int(rng.choice([2, 3]))
        if primary_cluster >= total:
            primary_cluster = 2

    remaining = total - primary_cluster
    secondary_cluster = 0
    if remaining >= 3 and rng.random() < 0.4:
        secondary_cluster = 2
    isolated = remaining - secondary_cluster
    if isolated < 1:
        secondary_cluster = 0
        isolated = remaining
    return primary_cluster, secondary_cluster, isolated


def _group_fits(
    new_centers: list[tuple[int, int]],
    placed_centers: list[tuple[int, int]],
    shape: tuple[int, int],
    *,
    require_far: bool,
) -> bool:
    H, W = shape
    for r, c in new_centers:
        if not (2 <= int(r) < H - 2 and 2 <= int(c) < W - 2):
            return False

    for i, center_a in enumerate(new_centers):
        for center_b in new_centers[i + 1 :]:
            if _bbox_conflict(center_a, center_b):
                return False
        for center_b in placed_centers:
            if _bbox_conflict(center_a, center_b):
                return False
            if _expanded_connected(center_a, center_b, shape):
                return False
            if require_far and abs(center_a[0] - center_b[0]) + abs(center_a[1] - center_b[1]) < 6:
                return False

    if len(new_centers) >= 2 and _component_sizes_from_centers(new_centers, shape) != [len(new_centers)]:
        return False
    return True


def _sample_linear_group(
    rng: np.random.Generator,
    shape: tuple[int, int],
    size: int,
    placed_centers: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    H, W = shape
    for _ in range(300):
        horizontal = bool(rng.integers(0, 2))
        if horizontal:
            if W - 2 - 4 * (size - 1) <= 2:
                continue
            row = int(rng.integers(2, H - 2))
            col0 = int(rng.integers(2, W - 2 - 4 * (size - 1)))
            centers = [(row, col0 + 4 * i) for i in range(size)]
        else:
            if H - 2 - 4 * (size - 1) <= 2:
                continue
            row0 = int(rng.integers(2, H - 2 - 4 * (size - 1)))
            col = int(rng.integers(2, W - 2))
            centers = [(row0 + 4 * i, col) for i in range(size)]

        if _group_fits(centers, placed_centers, shape, require_far=True):
            return centers
    return None


def _sample_isolated_center(
    rng: np.random.Generator,
    shape: tuple[int, int],
    placed_centers: list[tuple[int, int]],
) -> tuple[int, int] | None:
    H, W = shape
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)
    for center in candidates:
        if _group_fits([center], placed_centers, shape, require_far=True):
            return center
    return None


def _cross_spec(center: tuple[int, int], arm_color: int, center_color: int) -> dict:
    return {
        "center": tuple(map(int, center)),
        "arm_color": int(arm_color),
        "center_color": int(center_color),
    }


def _render_input(shape: tuple[int, int], crosses: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for cross in crosses:
        r, c = cross["center"]
        grid[r, c] = int(cross["center_color"])
        for dr, dc in ORTHO:
            grid[r + dr, c + dc] = int(cross["arm_color"])
    return grid


def _analyze_layout(crosses: list[dict], shape: tuple[int, int]) -> list[int]:
    return _component_sizes_from_centers([cross["center"] for cross in crosses], shape)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    primary_cluster, secondary_cluster, isolated_count = _choose_group_counts(rng)
    total = primary_cluster + secondary_cluster + isolated_count

    size_low = 15 + max(0, total - 4) * 2
    H = int(rng.integers(size_low, 23))
    W = int(rng.integers(size_low, 23))
    shape = (H, W)

    centers: list[tuple[int, int]] = []
    groups: list[list[tuple[int, int]]] = []

    main_group = _sample_linear_group(rng, shape, primary_cluster, centers)
    if main_group is None:
        return None
    centers.extend(main_group)
    groups.append(main_group)

    if secondary_cluster:
        extra_group = _sample_linear_group(rng, shape, secondary_cluster, centers)
        if extra_group is None:
            return None
        centers.extend(extra_group)
        groups.append(extra_group)

    for _ in range(isolated_count):
        center = _sample_isolated_center(rng, shape, centers)
        if center is None:
            return None
        centers.append(center)
        groups.append([center])

    palette = list(INPUT_COLORS)
    rng.shuffle(palette)
    crosses: list[dict] = []

    for group in groups:
        if len(group) >= 2:
            arm_color = int(rng.choice(INPUT_COLORS))
            center_choices = [color for color in INPUT_COLORS if color != arm_color]
            rng.shuffle(center_choices)
            for idx, center in enumerate(group):
                center_color = int(center_choices[idx % len(center_choices)])
                crosses.append(_cross_spec(center, arm_color, center_color))
        else:
            arm_color, center_color = rng.choice(INPUT_COLORS, size=2, replace=False)
            crosses.append(_cross_spec(group[0], int(arm_color), int(center_color)))

    grid = _render_input(shape, crosses)
    component_sizes = _analyze_layout(crosses, shape)
    if not any(size >= 2 for size in component_sizes):
        return None
    if not any(size == 1 for size in component_sizes):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.any(grid == GREY):
        return None
    if not np.any(out == GREY):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cross-connectivity input")
