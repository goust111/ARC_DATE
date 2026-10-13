import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, components_mask, scale_sprite


INPUT_PALETTE = (4, 5, 6, 7, 8, 9)
FILL_BLUE = Color.BLUE
FILL_RED = Color.RED
FILL_GREEN = Color.GREEN
GEN_ATTEMPTS = 400
E_WEIGHTS = ((1, 2), (2, 3), (3, 3), (4, 2))


def _touches_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == w - 1)
    )


def _enclosed_regions(grid: np.ndarray) -> list[dict]:
    regions = []
    for coords in components_mask(grid == Color.BLACK, connectivity=4):
        if _touches_boundary(coords, grid.shape):
            continue
        b = bbox_coords(coords)
        regions.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "bbox": b,
            }
        )
    regions.sort(key=lambda region: (region["bbox"][0], region["bbox"][1]))
    return regions


def _rank_two_regions(regions: list[dict]) -> list[dict]:
    return sorted(
        regions,
        key=lambda region: (-region["area"], region["bbox"][0], region["bbox"][1]),
    )


def _filled_output(upscaled: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = upscaled.copy()
    e = len(regions)
    if e == 1:
        region = regions[0]
        out[region["coords"][:, 0], region["coords"][:, 1]] = FILL_RED
        return out
    if e == 2:
        red_region, blue_region = _rank_two_regions(regions)
        out[red_region["coords"][:, 0], red_region["coords"][:, 1]] = FILL_RED
        out[blue_region["coords"][:, 0], blue_region["coords"][:, 1]] = FILL_BLUE
        return out
    if e >= 3:
        for region in regions:
            out[region["coords"][:, 0], region["coords"][:, 1]] = FILL_GREEN
        return out
    raise AssertionError("expected at least one enclosed region")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale the pixel fence, count the enclosed black regions it creates, then fill them by the count-dependent rule.

    Input structure:
    - The input is a 4x4 or 5x5 grid with black background and 2-3 non-black foreground colors.
    - Foreground pixels act as fence cells; after 3x nearest-neighbor expansion, each becomes a solid 3x3 color block.
    - In the expanded grid, some black 4-connected regions are fully surrounded by those non-black blocks while others may stay open.

    Rule:
    - Enlarge the whole grid by factor 3 with nearest-neighbor replication.
    - In the enlarged grid, find every black 4-connected component that does not touch the grid boundary.
    - Let E be the number of those enclosed black regions.
    - If E=1, fill that region red. If E=2, fill the larger region red and the smaller blue; when their areas tie, the one with the smaller bounding-box top row is red. If E>=3, fill every enclosed region green.
    - Leave boundary-touching black regions and all original foreground blocks unchanged.

    Key insight:
    - The visible answer is determined in the enlarged geometry: first reveal the enclosed black regions after 3x scaling, then choose the fill rule from the total enclosed-region count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    upscaled = scale_sprite(g, 3)
    regions = _enclosed_regions(upscaled)
    assert 1 <= len(regions) <= 4, "valid puzzle inputs must create 1-4 enclosed regions"
    return _filled_output(upscaled, regions)


def _local_mask(frame_size: int, bits: int) -> np.ndarray:
    mask = np.zeros((frame_size, frame_size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    inner = frame_size - 2
    idx = 0
    for r in range(1, frame_size - 1):
        for c in range(1, frame_size - 1):
            if bits & (1 << idx):
                mask[r, c] = True
            idx += 1
    return mask


def _rotflip(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _embed(local_mask: np.ndarray, grid_size: int, top: int, left: int) -> np.ndarray:
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    h, w = local_mask.shape
    mask[top : top + h, left : left + w] = local_mask
    return mask


def _assign_colors(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.shape[0] >= 2
    n_colors = 2 if coords.shape[0] == 2 else random.choice((2, 3))
    colors = random.sample(INPUT_PALETTE, n_colors)
    seed_idx = random.sample(range(coords.shape[0]), n_colors)
    seeds = coords[seed_idx]

    out = np.zeros(mask.shape, dtype=int)
    for r, c in coords:
        distances = [abs(int(r) - int(sr)) + abs(int(c) - int(sc)) for sr, sc in seeds]
        color_idx = min(range(n_colors), key=lambda idx: (distances[idx], idx))
        out[int(r), int(c)] = colors[color_idx]

    used = set(map(int, out[mask]))
    assert len(used) == n_colors
    return out


def _build_catalogs() -> dict[tuple[int, int], list[np.ndarray]]:
    catalogs: dict[tuple[int, int], list[np.ndarray]] = {}
    for frame_size in (3, 4, 5):
        inner_cells = (frame_size - 2) ** 2
        for bits in range(1 << inner_cells):
            local_mask = _local_mask(frame_size, bits)
            regions = _enclosed_regions(scale_sprite(local_mask.astype(int), 3))
            e = len(regions)
            if e < 1 or e > 4:
                continue
            if e == 2:
                ranked = _rank_two_regions(regions)
                first, second = ranked
                if first["area"] == second["area"] and first["bbox"][0] == second["bbox"][0]:
                    continue
            catalogs.setdefault((frame_size, e), []).append(local_mask)
    for key, value in catalogs.items():
        assert value, f"empty catalog for {key}"
    return catalogs


CATALOGS = _build_catalogs()


def _choose_target_e() -> int:
    values = [value for value, _weight in E_WEIGHTS]
    weights = [weight for _value, weight in E_WEIGHTS]
    return random.choices(values, weights=weights, k=1)[0]


def _choose_frame_size(target_e: int) -> int:
    if target_e >= 3:
        return 5
    if target_e == 2:
        return random.choices([4, 5], weights=[3, 2], k=1)[0]
    return random.choices([3, 4, 5], weights=[3, 3, 2], k=1)[0]


def _choose_local_mask(frame_size: int, target_e: int) -> np.ndarray:
    masks = CATALOGS[(frame_size, target_e)]
    if target_e != 2:
        return random.choice(masks).copy()

    tie_masks = []
    other_masks = []
    for mask in masks:
        ranked = _rank_two_regions(_enclosed_regions(scale_sprite(mask.astype(int), 3)))
        if ranked[0]["area"] == ranked[1]["area"]:
            tie_masks.append(mask)
        else:
            other_masks.append(mask)
    if tie_masks and random.random() < 0.4:
        return random.choice(tie_masks).copy()
    return random.choice(other_masks if other_masks else tie_masks).copy()


def _choose_grid_size(frame_size: int) -> int:
    if frame_size == 5:
        return 5
    return random.choice([frame_size, 5])


def _fill_colors_for_regions(out: np.ndarray, regions: list[dict]) -> list[int]:
    colors = []
    for region in regions:
        vals = out[region["coords"][:, 0], region["coords"][:, 1]]
        colors.append(int(vals[0]))
        assert np.all(vals == vals[0])
    return colors


def _validate_generated(inp: np.ndarray, target_e: int) -> bool:
    if inp.shape not in {(4, 4), (5, 5)}:
        return False
    fg = set(map(int, inp[inp != Color.BLACK]))
    if not (2 <= len(fg) <= 3):
        return False
    if fg & {FILL_BLUE, FILL_RED, FILL_GREEN}:
        return False

    upscaled = scale_sprite(inp, 3)
    regions = _enclosed_regions(upscaled)
    if len(regions) != target_e:
        return False
    if not (1 <= len(regions) <= 4):
        return False

    out = main(inp)
    if not np.array_equal(out[upscaled != Color.BLACK], upscaled[upscaled != Color.BLACK]):
        return False

    enclosed_mask = np.zeros_like(upscaled, dtype=bool)
    for region in regions:
        enclosed_mask[region["coords"][:, 0], region["coords"][:, 1]] = True
    if not np.all(out[(upscaled == Color.BLACK) & (~enclosed_mask)] == Color.BLACK):
        return False

    region_colors = _fill_colors_for_regions(out, regions)
    if target_e == 1:
        if region_colors != [FILL_RED]:
            return False
    elif target_e == 2:
        ranked = _rank_two_regions(regions)
        expected = {tuple(map(tuple, ranked[0]["coords"])): FILL_RED, tuple(map(tuple, ranked[1]["coords"])): FILL_BLUE}
        for region, color in zip(regions, region_colors):
            if expected[tuple(map(tuple, region["coords"]))] != color:
                return False
    else:
        if set(region_colors) != {FILL_GREEN}:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target_e = _choose_target_e()
    frame_size = _choose_frame_size(target_e)
    local_mask = _rotflip(_choose_local_mask(frame_size, target_e))
    grid_size = _choose_grid_size(frame_size)

    max_offset = grid_size - frame_size
    top = random.randint(0, max_offset)
    left = random.randint(0, max_offset)
    mask = _embed(local_mask, grid_size, top, left)
    inp = _assign_colors(mask)

    if _validate_generated(inp, target_e):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 8")
