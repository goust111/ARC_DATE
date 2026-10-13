from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC = SourceFileLoader(
    "_arc_utils_fable_1_039_3",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

bbox_coords = _ARC.bbox_coords
find_components = _ARC.find_components
get_bg = _ARC.get_bg
paste = _ARC.paste
random_free_location = _ARC.random_free_location


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_MAX_ATTEMPTS = 400
GEN_BORDER = 1
GEN_PADDING = 1


def _hole_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    occupied = np.zeros(shape, dtype=bool)
    occupied[coords[:, 0], coords[:, 1]] = True
    free = ~occupied

    height, width = map(int, shape)
    exterior = np.zeros(shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def push(r: int, c: int) -> None:
        if free[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            queue.append((r, c))

    for r in range(height):
        push(r, 0)
        push(r, width - 1)
    for c in range(width):
        push(0, c)
        push(height - 1, c)

    while queue:
        r, c = queue.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and free[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return free & ~exterior


def _normalize_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        out.append(
            {
                "coords": coords,
                "bbox": tuple(map(int, comp["bbox"])),
                "area": int(comp["area"]),
                "color": int(comp["color"]),
            }
        )
    out.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["bbox"][2], comp["bbox"][3], comp["color"]))
    return out


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    components = _normalize_components(g, bg)
    count = len(components)

    hole_masks = [_hole_mask(g.shape, comp["coords"]) for comp in components]
    contains = [[False] * count for _ in range(count)]
    for parent_idx, hole in enumerate(hole_masks):
        for child_idx, child in enumerate(components):
            if parent_idx == child_idx:
                continue
            rows = child["coords"][:, 0]
            cols = child["coords"][:, 1]
            if np.all(hole[rows, cols]):
                contains[parent_idx][child_idx] = True

    parents = [-1] * count
    for child_idx in range(count):
        candidates = [idx for idx in range(count) if contains[idx][child_idx]]
        direct = [
            idx
            for idx in candidates
            if not any(contains[idx][other] for other in candidates if other != idx)
        ]
        assert len(direct) <= 1, "nesting must form disjoint chains"
        if direct:
            parents[child_idx] = direct[0]

    children = [[] for _ in range(count)]
    for child_idx, parent_idx in enumerate(parents):
        if parent_idx != -1:
            children[parent_idx].append(child_idx)
    assert all(len(kids) <= 1 for kids in children), "nesting must not branch"

    roots = [idx for idx, parent_idx in enumerate(parents) if parent_idx == -1]
    roots.sort(key=lambda idx: components[idx]["bbox"])

    chains: list[list[int]] = []
    seen: set[int] = set()
    for root in roots:
        chain = [root]
        current = root
        while children[current]:
            current = children[current][0]
            assert current not in chain, "nesting graph must be acyclic"
            chain.append(current)
        chains.append(chain)
        seen.update(chain)
    assert len(seen) == count, "every component must belong to exactly one chain"

    return {
        "grid": g,
        "bg": bg,
        "components": components,
        "parents": parents,
        "children": children,
        "chains": chains,
        "contains": contains,
    }


def _apply_chain_reversal(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    components = analysis["components"]

    for chain in analysis["chains"]:
        if len(chain) < 2:
            continue
        colors = [components[idx]["color"] for idx in chain]
        for idx, new_color in zip(chain, reversed(colors)):
            coords = components[idx]["coords"]
            out[coords[:, 0], coords[:, 1]] = int(new_color)
    return out


def _component_output_color(grid: np.ndarray, coords: np.ndarray) -> int:
    values = {int(grid[r, c]) for r, c in coords}
    assert len(values) == 1, "component shape must stay monochrome"
    return int(next(iter(values)))


def main(grid: np.ndarray) -> np.ndarray:
    """Reverse colors along each nested containment chain while leaving every shape in place.

    Input structure:
    - A dominant background surrounds several disconnected monochrome foreground components.
    - Some components form chains where one component lies entirely inside a background hole enclosed by another component.
    - The same grid may also contain independent components that neither contain nor are contained by any other object.

    Rule:
    - Detect the direct parent-child relation induced by enclosed holes and decompose the foreground into disjoint nesting chains.
    - For every chain of length at least two, keep every component's coordinates fixed but reverse the chain's depth-ordered color sequence.
    - Leave independent components and the background unchanged.

    Key insight:
    - The transformation is driven by negative-space containment, not by size, position, or a global color permutation: colors flow only within each local nesting chain.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    return _apply_chain_reversal(np.asarray(grid, dtype=int), analysis)


def _random_connected_patch(
    color: int,
    rng: random.Random,
    *,
    h_range: tuple[int, int],
    w_range: tuple[int, int],
    fill_range: tuple[float, float],
) -> np.ndarray:
    height = rng.randint(*h_range)
    width = rng.randint(*w_range)
    total = height * width
    min_fill = max(3, int(np.ceil(total * fill_range[0])))
    max_fill = min(total, int(np.floor(total * fill_range[1])))
    target = rng.randint(min_fill, max_fill)

    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(rng.choice(tuple(frontier)))

    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in cells:
        patch[r - r0, c - c0] = int(color)

    if rng.random() < 0.5:
        patch = np.flipud(patch)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    if rng.random() < 0.35:
        patch = patch.T.copy()
    return patch.astype(int, copy=True)


def _frame_patch(height: int, width: int, thickness: int, color: int) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[:thickness, :] = int(color)
    patch[-thickness:, :] = int(color)
    patch[:, :thickness] = int(color)
    patch[:, -thickness:] = int(color)
    return patch


def _wrap_with_frame(
    child_patch: np.ndarray,
    color: int,
    thickness: int,
    rng: random.Random,
    *,
    gap_range: tuple[int, int],
    slack_range: tuple[int, int],
) -> np.ndarray:
    child_h, child_w = map(int, child_patch.shape)
    gap = rng.randint(*gap_range)
    slack_top = rng.randint(*slack_range)
    slack_bottom = rng.randint(*slack_range)
    slack_left = rng.randint(*slack_range)
    slack_right = rng.randint(*slack_range)

    hole_h = child_h + 2 * gap + slack_top + slack_bottom
    hole_w = child_w + 2 * gap + slack_left + slack_right
    outer_h = hole_h + 2 * thickness
    outer_w = hole_w + 2 * thickness

    patch = _frame_patch(outer_h, outer_w, thickness, color)
    offset_r = thickness + gap + slack_top
    offset_c = thickness + gap + slack_left
    paste(patch, child_patch, offset_r, offset_c, transparent=BG)
    return patch


def _build_chain_patch(
    colors: list[int],
    thicknesses: list[int],
    rng: random.Random,
    *,
    inner_h_range: tuple[int, int],
    inner_w_range: tuple[int, int],
    gap_ranges: list[tuple[int, int]],
    slack_ranges: list[tuple[int, int]],
) -> np.ndarray:
    assert len(colors) >= 2, "chain needs at least two components"
    assert len(thicknesses) == len(colors) - 1, "one thickness per ring layer"
    assert len(gap_ranges) == len(thicknesses), "one gap range per ring layer"
    assert len(slack_ranges) == len(thicknesses), "one slack range per ring layer"

    patch = _random_connected_patch(
        colors[-1],
        rng,
        h_range=inner_h_range,
        w_range=inner_w_range,
        fill_range=(0.55, 1.0),
    )
    for color, thickness, gap_range, slack_range in zip(
        reversed(colors[:-1]),
        reversed(thicknesses),
        reversed(gap_ranges),
        reversed(slack_ranges),
    ):
        patch = _wrap_with_frame(
            patch,
            color,
            thickness,
            rng,
            gap_range=gap_range,
            slack_range=slack_range,
        )
    return patch


def _build_independent_patch(color: int, rng: random.Random) -> np.ndarray:
    return _random_connected_patch(
        color,
        rng,
        h_range=(2, 5),
        w_range=(2, 5),
        fill_range=(0.45, 0.8),
    )


def _chain_areas(patch: np.ndarray) -> list[int]:
    analysis = _analyze_grid(patch)
    active = [chain for chain in analysis["chains"] if len(chain) >= 2]
    assert len(active) == 1, "local chain patch should contain exactly one chain"
    chain = active[0]
    return [analysis["components"][idx]["area"] for idx in chain]


def _is_monotone(values: list[int]) -> bool:
    nondecreasing = all(a <= b for a, b in zip(values, values[1:]))
    nonincreasing = all(a >= b for a, b in zip(values, values[1:]))
    return nondecreasing or nonincreasing


def _place_patch(grid: np.ndarray, reserved: np.ndarray, patch: np.ndarray) -> bool:
    blocker = np.ones(patch.shape, dtype=int)
    try:
        r0, c0 = random_free_location(
            reserved,
            blocker,
            transparent=BG,
            border=GEN_BORDER,
            padding=GEN_PADDING,
        )
    except ValueError:
        return False

    paste(grid, patch, r0, c0, transparent=BG)
    paste(reserved, blocker, r0, c0)
    return True


def _verify_semantics(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
        out = main(grid)
    except AssertionError:
        return False

    active_chains = [chain for chain in analysis["chains"] if len(chain) >= 2]
    if not active_chains:
        return False
    if max(len(chain) for chain in active_chains) < 3:
        return False
    if not any(len(chain) == 2 for chain in active_chains):
        return False

    components = analysis["components"]
    independent = [chain[0] for chain in analysis["chains"] if len(chain) == 1]
    if not independent:
        return False

    chain_colors = {components[idx]["color"] for chain in active_chains for idx in chain}
    if not any(components[idx]["color"] in chain_colors for idx in independent):
        return False

    if not any(len(chain) >= 3 and not _is_monotone([components[idx]["area"] for idx in chain]) for chain in active_chains):
        return False

    for chain in active_chains:
        in_colors = [components[idx]["color"] for idx in chain]
        out_colors = [_component_output_color(out, components[idx]["coords"]) for idx in chain]
        if out_colors != list(reversed(in_colors)):
            return False

    for idx in independent:
        if _component_output_color(out, components[idx]["coords"]) != components[idx]["color"]:
            return False

    return not np.array_equal(grid, out)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    deep_len = 3
    deep_thicknesses = [1, 2]
    short_len = 2
    short_thicknesses = [rng.choice((1, 2))]

    chain_color_count = deep_len + short_len
    all_chain_colors = rng.sample(list(range(1, 10)), chain_color_count)
    deep_colors = all_chain_colors[:deep_len]
    short_colors = all_chain_colors[deep_len:]

    deep_patch = _build_chain_patch(
        deep_colors,
        deep_thicknesses,
        rng,
        inner_h_range=(2, 3),
        inner_w_range=(2, 3),
        gap_ranges=[(1, 1), (1, 1)],
        slack_ranges=[(0, 0), (0, 1)],
    )
    short_patch = _build_chain_patch(
        short_colors,
        short_thicknesses,
        rng,
        inner_h_range=(2, 3),
        inner_w_range=(2, 3),
        gap_ranges=[(1, 1)],
        slack_ranges=[(0, 1)],
    )

    if max(deep_patch.shape) > 14 or max(short_patch.shape) > 10:
        return None
    if _is_monotone(_chain_areas(deep_patch)):
        return None

    independent_count = rng.randint(1, 3)
    independent_patches = []
    for _ in range(independent_count):
        reuse_color = rng.choice(deep_colors + short_colors)
        independent_patches.append(_build_independent_patch(reuse_color, rng))

    min_side = max(
        int(max(deep_patch.shape) + 6),
        int(max(short_patch.shape) + 6),
        20,
    )
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)

    grid = np.zeros((height, width), dtype=int)
    reserved = np.zeros_like(grid)
    patches = [deep_patch, short_patch] + independent_patches
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    for patch in patches:
        if not _place_patch(grid, reserved, patch):
            return None

    if not _verify_semantics(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-color-convection input")
