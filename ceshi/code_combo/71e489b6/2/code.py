# Combo variant of 71e489b6 (index 2)
# combo_spec: D2/2+D9/0
# Name: 泛化异常检测+灰色噪点干扰
#
# Design analysis for the fused variant
# a) Concrete trace:
#    - Original: absorb blue spurs inside black, then halo only black mini-components and leaf pixels.
#    - Here: there are no blue spurs. Instead, any non-blue 4-connected component whose outside 4-neighbors
#      are all blue is a target anomaly, even if that component mixes black/red/grey internally.
#      Separately, any grey component whose outside 4-neighbors are all black is removable noise and turns black.
#      Example: a grey cell with four blue neighbors is haloed and stays grey; a grey cell with four black
#      neighbors is erased to black. This forces domain-based reasoning rather than color-only reasoning.
# b) Complexity calibration:
#    - Keep generated anomaly sizes at 1-3 cells and use only a small color set {0,2,5} inside blue so the
#      visual rule is obvious without exploding the number of cases.
#    - Every sample includes both blue-enclosed anomalies and black-enclosed grey noise; many samples also
#      include blue-enclosed grey anomalies so the same color can lead to different outcomes by location.
# c) Implicit-assumption audit:
#    - `main()` assumes target anomalies do not touch the border and that all non-component 4-neighbors are blue.
#    - It also assumes removable grey noise is fully enclosed by black.
#    - The generator guarantees both by placing anomalies only on interior blue cells and grey noise only on
#      interior black cells, with clearance so halos do not visually collide with the removable noise.
#    - The original leaf-specific large-black logic is intentionally removed; keeping it would overlap with the
#      new "blue-enclosed anomaly" rule and make examples harder to infer from 2-4 pairs.
# d) Placement legality:
#    - Anomaly components are grown only from 4-interior blue cells and are kept separated by a Chebyshev gap.
#    - Grey noise components are grown only from 4-interior black cells and are excluded from anomaly halo zones.
# e/f/g) Observability and visual cleanliness:
#    - The output has only one marker color (orange), anomalies keep their own center color, and grey appears
#      in both domains with different outcomes. That makes the fused behavior visible but still tidy.
#    - Using location/enclosure instead of hidden ordering or numeric color rank keeps the rule human-inferable.
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask


GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 30
MIN_BLACK_REGION = 20
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
STRIPE_CUTS_RANGE = (1, 3)
CUT_MARGIN = 3
INSET_MARGIN = 2
ANOMALY_COUNT_RANGE = (3, 8)
ANOMALY_SIZE_WEIGHTS = [6, 2, 1]
ANOMALY_CLEARANCE = 2
ANOMALY_PLACE_TRIES = 60
NOISE_COUNT_RANGE = (2, 5)
NOISE_SIZE_WEIGHTS = [5, 2]
NOISE_CLEARANCE = 2
MIN_ANOMALIES = 3
MIN_NOISE = 2
MAX_GEN_ATTEMPTS = 400

BLUE = 1
BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
ANOMALY_COLORS = [BLACK, RED, GREY]
ANOMALY_COLOR_WEIGHTS = [4, 3, 3]


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    for comp in components_mask(mask, connectivity=4):
        comps.append([(int(r), int(c)) for r, c in comp])
    return comps


def _same_color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    return _mask_components(grid == color)


def _enclosed_by_color(grid: np.ndarray, comp: list[tuple[int, int]], surround_color: int) -> bool:
    H, W = grid.shape
    comp_set = set(comp)
    saw_exterior = False
    for r, c in comp:
        if r == 0 or r == H - 1 or c == 0 or c == W - 1:
            return False
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in comp_set:
                continue
            saw_exterior = True
            if grid[rr, cc] != surround_color:
                return False
    return saw_exterior


def _classify_components(grid: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    target_pixels: set[tuple[int, int]] = set()
    removable_noise: set[tuple[int, int]] = set()

    for comp in _mask_components(grid != BLUE):
        if _enclosed_by_color(grid, comp, BLUE):
            target_pixels.update(comp)

    for comp in _same_color_components(grid, GREY):
        if _enclosed_by_color(grid, comp, BLACK):
            removable_noise.update(comp)

    return target_pixels, removable_noise


def _build_halo(clean_grid: np.ndarray, target_pixels: set[tuple[int, int]], original_grid: np.ndarray) -> np.ndarray:
    H, W = clean_grid.shape
    out = clean_grid.copy()

    for r, c in target_pixels:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in target_pixels:
                    out[rr, cc] = ORANGE

    for r, c in target_pixels:
        out[r, c] = int(original_grid[r, c])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark any blue-enclosed non-blue anomaly, while erasing grey noise that sits inside black.

    Input structure:
    - Large blue regions lie on a black background, and the blue regions may contain small black, red, or grey patches.
    - Black regions may also contain tiny grey specks that are not defects and should disappear.
    - Target anomalies are fully surrounded by blue in the 4-neighborhood; removable grey noise is fully surrounded by black.

    Rule:
    - First turn every grey component enclosed by black into black.
    - Then find every non-blue component enclosed by blue and paint an orange Chebyshev-1 halo around it.
    - The anomalous pixels themselves keep their original color; all other cells stay as in the cleaned grid.

    Key insight: The same grey color can either be highlighted or erased, so enclosure by the surrounding region
    determines the action rather than the anomalous color alone.
    """
    g = np.array(grid, dtype=int)
    targets, removable_noise = _classify_components(g)

    clean = g.copy()
    for r, c in removable_noise:
        clean[r, c] = BLACK

    return _build_halo(clean, targets, g)


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a black/blue partition with large simple blue regions and no enclosed large black islands."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_cuts = random.randint(*STRIPE_CUTS_RANGE)
        cut_pool = list(range(CUT_MARGIN, H - CUT_MARGIN))
        cuts = sorted(random.sample(cut_pool, min(n_cuts, len(cut_pool))))
        boundaries = [0] + cuts + [H]
        first_color = random.randint(0, 1)
        for idx in range(len(boundaries) - 1):
            grid[boundaries[idx] : boundaries[idx + 1], :] = (first_color + idx) % 2

    elif style == "rect_inset":
        r0 = random.randint(INSET_MARGIN, H // 3)
        r1 = random.randint(2 * H // 3, H - CUT_MARGIN)
        c0 = random.randint(INSET_MARGIN, W // 3)
        c1 = random.randint(2 * W // 3, W - CUT_MARGIN)
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    else:
        vc = random.randint(W // 3, 2 * W // 3)
        hc = random.randint(H // 3, 2 * H // 3)
        if random.random() < 0.5:
            grid[:, :vc] = BLUE
            grid[:hc, vc:] = BLUE
        else:
            grid[:, vc:] = BLUE
            grid[hc:, :vc] = BLUE

    return grid


def _find_interior_cells(grid: np.ndarray, color: int, halo_safe: bool = False) -> list[tuple[int, int]]:
    H, W = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if grid[r, c] != color:
                continue
            if not all(grid[rr, cc] == color for rr, cc in _neighbors4(r, c)):
                continue
            if halo_safe:
                keep = True
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if grid[r + dr, c + dc] != color:
                            keep = False
                            break
                    if not keep:
                        break
                if not keep:
                    continue
            coords.append((r, c))
    random.shuffle(coords)
    return coords


def _grow_component(
    seed: tuple[int, int],
    desired_size: int,
    candidate_set: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    comp = [seed]
    frontier = [seed]

    while frontier and len(comp) < desired_size:
        r, c = frontier.pop(random.randrange(len(frontier)))
        nbrs = list(_neighbors4(r, c))
        random.shuffle(nbrs)
        for rr, cc in nbrs:
            if (rr, cc) not in candidate_set or (rr, cc) in occupied or (rr, cc) in comp:
                continue
            comp.append((rr, cc))
            frontier.append((rr, cc))
            if len(comp) >= desired_size:
                break

    return comp


def _valid_enclosed_component(
    grid: np.ndarray,
    comp: list[tuple[int, int]],
    carrier_color: int,
    occupied: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    comp_set = set(comp)
    for r, c in comp:
        if r == 0 or r == H - 1 or c == 0 or c == W - 1:
            return False
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in comp_set:
                continue
            if grid[rr, cc] != carrier_color or (rr, cc) in occupied:
                return False
    return True


def _choose_anomaly_plan() -> list[int]:
    n_total = random.randint(*ANOMALY_COUNT_RANGE)
    # Force one grey anomaly in blue and one differently colored anomaly for repeated support.
    colors = [GREY, random.choice([BLACK, RED])]
    while len(colors) < n_total:
        colors.append(random.choices(ANOMALY_COLORS, weights=ANOMALY_COLOR_WEIGHTS)[0])
    random.shuffle(colors)
    return colors


def _place_anomalies(grid: np.ndarray, interior_blue: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], list[int]]:
    interior_set = set(interior_blue)
    anomalies: set[tuple[int, int]] = set()
    placed_colors: list[int] = []

    for color in _choose_anomaly_plan():
        if not interior_blue:
            break
        desired_size = random.choices([1, 2, 3], weights=ANOMALY_SIZE_WEIGHTS)[0]
        placed = False

        for _ in range(ANOMALY_PLACE_TRIES):
            seed = interior_blue[random.randrange(len(interior_blue))]
            if _nearby(seed[0], seed[1], ANOMALY_CLEARANCE, anomalies):
                continue
            comp = _grow_component(seed, desired_size, interior_set, anomalies)
            if not _valid_enclosed_component(grid, comp, BLUE, anomalies):
                continue

            comp_set = set(comp)
            for r, c in comp:
                grid[r, c] = color
            anomalies.update(comp_set)
            placed_colors.append(color)
            remove = _exclusion_zone(comp_set, ANOMALY_CLEARANCE)
            interior_blue[:] = [p for p in interior_blue if p not in remove]
            interior_set -= remove
            placed = True
            break

        if not placed:
            continue

    return anomalies, placed_colors


def _noise_exclusion_from_anomalies(anomalies: set[tuple[int, int]]) -> set[tuple[int, int]]:
    # Keep grey noise away from the orange ring so its removal remains visibly separate.
    return _exclusion_zone(anomalies, NOISE_CLEARANCE)


def _place_grey_noise(
    clean_grid: np.ndarray,
    anomalies: set[tuple[int, int]],
) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    inp = clean_grid.copy()
    candidate_list = _find_interior_cells(inp, BLACK)
    candidate_set = set(candidate_list)
    occupied: set[tuple[int, int]] = set()
    noise_pixels: set[tuple[int, int]] = set()
    noise_components = 0
    forbid = _noise_exclusion_from_anomalies(anomalies)

    n_noise = random.randint(*NOISE_COUNT_RANGE)
    for _ in range(n_noise):
        if not candidate_list:
            break
        desired_size = random.choices([1, 2], weights=NOISE_SIZE_WEIGHTS)[0]
        placed = False

        for _ in range(ANOMALY_PLACE_TRIES):
            seed = candidate_list[random.randrange(len(candidate_list))]
            if seed in forbid or _nearby(seed[0], seed[1], NOISE_CLEARANCE, occupied):
                continue
            comp = _grow_component(seed, desired_size, candidate_set, occupied | forbid)
            comp_set = set(comp)
            if not comp_set or comp_set & forbid:
                continue
            if not _valid_enclosed_component(inp, comp, BLACK, occupied):
                continue

            for r, c in comp:
                inp[r, c] = GREY
            noise_pixels.update(comp_set)
            occupied.update(comp_set)
            noise_components += 1
            remove = _exclusion_zone(comp_set, NOISE_CLEARANCE)
            candidate_list[:] = [p for p in candidate_list if p not in remove and p not in forbid]
            candidate_set -= remove
            placed = True
            break

        if not placed:
            continue

    return inp, noise_pixels, noise_components


def _validate_partition(grid: np.ndarray) -> bool:
    blue_comps = _same_color_components(grid, BLUE)
    black_comps = _same_color_components(grid, BLACK)
    if not blue_comps or not black_comps:
        return False
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return False
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    clean = _build_partition(H, W)
    if not _validate_partition(clean):
        return None

    interior_blue = _find_interior_cells(clean, BLUE, halo_safe=True)
    if len(interior_blue) < 8:
        return None

    anomaly_grid = clean.copy()
    anomalies, placed_colors = _place_anomalies(anomaly_grid, interior_blue)
    if len(placed_colors) < MIN_ANOMALIES or GREY not in placed_colors:
        return None
    if not any(color in {BLACK, RED} for color in placed_colors):
        return None

    expected = _build_halo(anomaly_grid, anomalies, anomaly_grid)

    inp, noise_pixels, noise_components = _place_grey_noise(anomaly_grid, anomalies)
    if noise_components < MIN_NOISE:
        return None
    if not noise_pixels:
        return None

    # Ensure the grey-noise dimension is genuinely active in this sample.
    if not any(
        _enclosed_by_color(inp, comp, BLACK)
        for comp in _same_color_components(inp, GREY)
    ):
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
