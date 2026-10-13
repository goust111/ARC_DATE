from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


GREY = Color.GREY
YELLOW = Color.YELLOW
BACKGROUND = Color.BLACK
INPUT_COLORS = (Color.BLUE, Color.RED, Color.GREEN)

BASE_SHAPES: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
)

GEN_RETRIES = 400


def _separator_cols(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]


def _regions_from_separators(width: int, separators: list[int]) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    start = 0
    for col in separators:
        if start <= col - 1:
            regions.append((start, col - 1))
        start = col + 1
    if start <= width - 1:
        regions.append((start, width - 1))
    return regions


def _sorted_components(region: np.ndarray) -> list[dict]:
    comps = find_components(region, bg=BACKGROUND, connectivity=4, monochromatic=True, return_format="dict")
    return sorted(
        comps,
        key=lambda comp: (
            int(comp["coords"][:, 0].max()),
            int(comp["coords"][:, 0].min()),
            -int(comp["coords"][:, 1].min()),
        ),
        reverse=True,
    )


def _add_component(
    active: dict[int, dict],
    owner: np.ndarray,
    color_grid: np.ndarray,
    coords: set[tuple[int, int]],
    color: int,
    next_id: int,
) -> int:
    active[int(next_id)] = {"color": int(color), "coords": set(coords)}
    for r, c in coords:
        owner[r, c] = int(next_id)
        color_grid[r, c] = int(color)
    return int(next_id + 1)


def _remove_component(active: dict[int, dict], owner: np.ndarray, color_grid: np.ndarray, cid: int) -> None:
    comp = active.pop(int(cid))
    for r, c in comp["coords"]:
        owner[r, c] = -1
        color_grid[r, c] = BACKGROUND


def _settle_region(
    region: np.ndarray,
    *,
    same_merge_mode: str = "yellow",
    diff_mode: str = "stack",
    return_events: bool = False,
) -> tuple[np.ndarray, list[dict]] | np.ndarray:
    h, w = region.shape
    comps = _sorted_components(region)
    owner = np.full((h, w), -1, dtype=int)
    color_grid = np.full((h, w), BACKGROUND, dtype=int)
    active: dict[int, dict] = {}
    next_id = 0
    events: list[dict] = []

    for comp in comps:
        color = int(comp["color"])
        coords = np.array(comp["coords"], dtype=int)
        drop = 0
        while True:
            blocked = False
            for r, c in coords:
                nr = int(r + drop + 1)
                if nr >= h or owner[nr, int(c)] != -1:
                    blocked = True
                    break
            if blocked:
                break
            drop += 1

        final_coords = {(int(r + drop), int(c)) for r, c in coords}
        contact_ids = sorted(
            {
                int(owner[r + 1, c])
                for r, c in final_coords
                if r + 1 < h and owner[r + 1, c] != -1
            }
        )
        contact_colors = {int(active[cid]["color"]) for cid in contact_ids}
        same_contact = len(contact_colors) == 1 and bool(contact_colors) and next(iter(contact_colors)) == color

        if not contact_ids:
            outcome = "bottom"
            next_id = _add_component(active, owner, color_grid, final_coords, color, next_id)
        elif same_contact:
            outcome = "same"
            merged_coords = set(final_coords)
            for cid in contact_ids:
                merged_coords |= set(active[cid]["coords"])
            for cid in contact_ids:
                _remove_component(active, owner, color_grid, cid)
            merged_color = YELLOW if same_merge_mode == "yellow" else color
            next_id = _add_component(active, owner, color_grid, merged_coords, merged_color, next_id)
        elif diff_mode == "yellow":
            outcome = "diff_merge"
            merged_coords = set(final_coords)
            for cid in contact_ids:
                merged_coords |= set(active[cid]["coords"])
            for cid in contact_ids:
                _remove_component(active, owner, color_grid, cid)
            next_id = _add_component(active, owner, color_grid, merged_coords, YELLOW, next_id)
        else:
            outcome = "diff"
            next_id = _add_component(active, owner, color_grid, final_coords, color, next_id)

        if return_events:
            events.append(
                {
                    "outcome": outcome,
                    "falling_color": color,
                    "contact_ids": contact_ids,
                    "contact_colors": sorted(contact_colors),
                    "drop": int(drop),
                    "landed_lowest_row": int(max(r for r, _ in final_coords)),
                }
            )

    if return_events:
        return color_grid.copy(), events
    return color_grid.copy()


def _solve_grid(grid: np.ndarray, *, same_merge_mode: str = "yellow", diff_mode: str = "stack") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    separators = _separator_cols(g)
    out = np.full((h, w), BACKGROUND, dtype=int)
    for col in separators:
        out[:, col] = GREY
    for c0, c1 in _regions_from_separators(w, separators):
        solved = _settle_region(g[:, c0 : c1 + 1], same_merge_mode=same_merge_mode, diff_mode=diff_mode)
        out[:, c0 : c1 + 1] = solved
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Apply independent gravity inside separator-defined column regions with same-color fusion recolored to yellow.

    Input structure:
    - A black background contains 2-3 full grey separator columns that split the grid into equal-width vertical regions.
    - Each region contains several small 4-connected monochromatic components in blue, red, or green at different heights.
    - Separator columns are not foreground pieces; they only define independent falling zones.

    Rule:
    - Keep the grey separator columns fixed and process each region independently.
    - Inside one region, find all non-black connected components and sort them by their lowest occupied row from bottommost to topmost.
    - Drop each component straight downward until it reaches the region floor or the row immediately above already settled pixels.
    - If the stopped component touches settled pixels directly below and every touched settled component has the same color as the falling one, merge them and recolor the whole merged component yellow.
    - If the touched settled color is different, keep both components unchanged and simply stack them.
    - A component that lands on the floor without touching another component keeps its color, and any previously fused yellow body participates in later color checks as yellow.

    Key insight:
    - The separator columns make gravity local to each region, and a same-color collision changes future outcomes because fusion recolors the settled body before later pieces test against it.
    """
    return _solve_grid(np.asarray(grid, dtype=int))


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in coords)


def _transform_shape(shape: tuple[tuple[int, int], ...], rot: int, flip: bool) -> list[tuple[int, int]]:
    pts = np.array(shape, dtype=int)
    if flip:
        pts[:, 1] *= -1
    for _ in range(rot % 4):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return _normalize([(int(r), int(c)) for r, c in pts])


def _random_shape(region_w: int) -> list[tuple[int, int]]:
    candidates: list[list[tuple[int, int]]] = []
    for base in BASE_SHAPES:
        for rot in range(4):
            for flip in (False, True):
                coords = _transform_shape(base, rot, flip)
                width = max(c for _, c in coords) + 1
                if width <= region_w:
                    candidates.append(coords)
    return [tuple(pt) for pt in random.choice(candidates)]


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _make_color_sequence(force_chain_same: bool, allow_four: bool) -> list[int]:
    a = random.choice(INPUT_COLORS)
    b_choices = [c for c in INPUT_COLORS if c != a]
    if force_chain_same:
        return [a, a, a, a] if allow_four and random.random() < 0.45 else [a, a, a]
    b = random.choice(b_choices)
    patterns = [
        [a, a, a],
        [a, a, b],
        [a, a, a, a],
        [a, a, b, b],
    ]
    valid = [p for p in patterns if (allow_four or len(p) == 3)]
    return list(random.choice(valid))


def _build_region(height: int, region_w: int, force_chain_same: bool) -> np.ndarray | None:
    colors_bottom_to_top = _make_color_sequence(force_chain_same, allow_four=height >= 18)
    shapes_bottom_to_top = [_random_shape(region_w - 1 if region_w > 4 else region_w) for _ in colors_bottom_to_top]
    widths = [max(c for _, c in shape) + 1 for shape in shapes_bottom_to_top]
    heights = [max(r for r, _ in shape) + 1 for shape in shapes_bottom_to_top]

    max_w = max(widths)
    left = random.randint(0, region_w - max_w)

    topdown_shapes = list(reversed(shapes_bottom_to_top))
    topdown_colors = list(reversed(colors_bottom_to_top))
    topdown_heights = list(reversed(heights))

    base_gaps = [0] + [1] * (len(topdown_shapes) - 1) + [1]
    required = sum(topdown_heights) + sum(base_gaps)
    if required > height:
        return None
    extra = _scatter(height - required, len(base_gaps))
    gaps = [base + inc for base, inc in zip(base_gaps, extra)]

    region = np.full((height, region_w), BACKGROUND, dtype=int)
    r0 = gaps[0]
    for idx, (shape, color, sh) in enumerate(zip(topdown_shapes, topdown_colors, topdown_heights)):
        for dr, dc in shape:
            region[r0 + dr, left + dc] = int(color)
        r0 += sh + gaps[idx + 1]
    return region


def _validate_region_events(region: np.ndarray) -> list[dict]:
    solved, events = _settle_region(region, return_events=True)
    assert solved.shape == region.shape
    assert any(event["outcome"] == "same" for event in events), "missing same-color collision"
    assert any(event["outcome"] == "diff" for event in events), "missing different-color collision"
    assert all(len(event["contact_ids"]) <= 1 for event in events if event["outcome"] != "bottom"), "ambiguous support"
    return events


def _validate_generated(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h <= 30 and w <= 30
    separators = _separator_cols(grid)
    assert len(separators) in (2, 3)
    regions = _regions_from_separators(w, separators)
    widths = {c1 - c0 + 1 for c0, c1 in regions}
    assert len(widths) == 1 and next(iter(widths)) in (4, 5)
    assert len(regions) in (3, 4)

    saw_chain_sensitive = False
    for c0, c1 in regions:
        region = grid[:, c0 : c1 + 1]
        comps = _sorted_components(region)
        assert 3 <= len(comps) <= 4
        events = _validate_region_events(region)
        if any(
            event["outcome"] == "diff"
            and event["falling_color"] in INPUT_COLORS
            and event["contact_colors"] == [YELLOW]
            for event in events
        ):
            saw_chain_sensitive = True
    assert saw_chain_sensitive, "expected at least one collision whose result depends on prior recoloring"

    true_out = main(grid)
    wrong_keep = _solve_grid(grid, same_merge_mode="keep", diff_mode="stack")
    wrong_all_yellow = _solve_grid(grid, same_merge_mode="yellow", diff_mode="yellow")
    assert not np.array_equal(true_out, grid), "output should differ from input"
    assert not np.array_equal(true_out, wrong_keep), "should reject same-color merge without recolor"
    assert not np.array_equal(true_out, wrong_all_yellow), "should reject yellowing every collision"


def _gen_attempt() -> np.ndarray | None:
    region_count, region_w = random.choice(((3, 4), (3, 5), (4, 4), (4, 5)))
    force_region = random.randrange(region_count)
    height = random.randint(17, 22)
    width = region_count * region_w + (region_count - 1)
    if width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for sep_idx in range(1, region_count):
        grid[:, sep_idx * region_w + (sep_idx - 1)] = GREY

    for region_idx, (c0, c1) in enumerate(_regions_from_separators(width, _separator_cols(grid))):
        region = _build_region(height, c1 - c0 + 1, force_chain_same=region_idx == force_region)
        if region is None:
            return None
        grid[:, c0 : c1 + 1] = region

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
