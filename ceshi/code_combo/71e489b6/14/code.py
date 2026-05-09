# Combo variant of 71e489b6 (index 14)
# combo_spec: D2/2+D10/2
# Name: 泛化异常检测+前景背景互换

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_ROOT = ROOT / "solvers"
if str(SOLVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SOLVER_ROOT))

from arc_utils import components_mask


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5
ORANGE = 7
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Design analysis for combo D2/2 + D10/2
# a) Concrete behavior trace:
#    - Original: blue is the foreground mass, black is the background. The solver first absorbs
#      stray blue spur pixels in black territory, then marks two black defect classes:
#      small black components and leaf pixels of large black components.
#    - Fused: color roles are inverted and the defect definition is generalized at the same time.
#      Black is now the foreground mass and blue is the background. The cleanup step therefore also
#      inverts: stray black spur pixels in blue territory are absorbed back to blue. After cleanup,
#      the solver no longer asks "is this black component tiny or a leaf?"; instead it asks
#      "is this a non-black component fully enclosed by black?". Every such enclosed anomaly is
#      haloed, regardless of whether it is blue, red, or gray.
#    - Example:
#        input base            original logic on analogous scene      fused logic
#        00000                 mark only black holes/leaves           mark any non-black impurity
#        00100                 (color-specific structural test)       inside black, keep its color
#        00200                                                      => orange around blue and red
#        00000
#      The visible difference is that a red/gray impurity gets the same orange halo as a blue one.
# b) Complexity calibration:
#    - The generator uses only two anomaly colors per sample: blue plus one extra non-blue color
#      (red or gray). That is enough to make D2/2 visible without flooding the grid with too many
#      color meanings.
#    - Large black foreground objects remain simple bands / rectangles / L-shapes so that the new
#      rule is still visually readable from a few examples.
# c) Hidden assumptions audit:
#    - Original STEP 0 assumed blue was the removable spur color. Broken after D10/2; the cleanup
#      must instead examine black cells with >= 3 effective blue 4-neighbors.
#    - Original target collection iterated over black components. Broken after D2/2+D10/2 because
#      target pixels are no longer black at all. The fused solver must inspect non-black components.
#    - Original "len(comp) <= 3 or leaf degree <= 1" logic is intentionally removed. D2/2 says the
#      distinction is no longer structural-black-specific; any enclosed non-black impurity qualifies.
#    - New assumption introduced here: anomaly components should be enclosed by black and not be part
#      of the global blue background. `generate_input()` guarantees that by placing anomalies only in
#      interior black cells and by avoiding large enclosed blue islands in the base partition.
# d) Placement legality:
#    - Black foreground regions are created first on a blue canvas.
#    - Anomalies are placed only on black interior cells, with clearance to avoid overlap or merging.
#    - Input-only black spurs are injected only on blue cells that the cleanup pass provably absorbs.
# e) Visual observability:
#    - Every accepted sample contains both a blue anomaly and a non-blue anomaly inside black. This
#      makes the "any non-black impurity" generalization visible, not latent.
#    - Black spur cleanup is also visible because some input black pixels disappear in the output.
# f) Visual cleanliness:
#    - Halo footprints are spaced apart by clearance rules, and the base layouts avoid nested large
#      blue holes inside black. That keeps "background" and "anomaly" roles legible.
# g) Color-rule inferability:
#    - The output adds only one marker color (orange). The anomaly center keeps its own visible color,
#      so the human-facing rule is compact: "inside black, anything not black gets surrounded."
# h) Fusion-behavior metric used after implementation:
#    - For generated samples we can measure `nonblue_target_cells`: enclosed anomaly cells whose color
#      is not blue. If this is always zero, the dataset collapsed to pure color inversion.
#    - We can also compare the fused output against a blue-only baseline; differing outputs confirm
#      that generalized anomaly detection is actively exercised.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes enclosed non-black components are the only non-black objects fully surrounded
#      by black. The generator enforces that by constructing black foreground shapes directly on a blue
#      canvas and placing anomalies only as small interior components.
#    - `main()` assumes black spur candidates lie in blue territory and satisfy the cleanup threshold.
#      The generator validates each sample with an oracle check before returning it.


# --------------- solver ---------------

GRID_H_RANGE = (16, 19)
GRID_W_RANGE = (16, 19)
MIN_BLUE_REGION = 40
MIN_BLACK_REGION = 30
PARTITION_STYLES = ["h_stripes", "rect_inset", "l_shape"]
DEFECT_COUNT_RANGE = (3, 8)
DEFECT_SIZE_WEIGHTS = [6, 2, 1]
DEFECT_CLEARANCE = 2
DEFECT_PLACE_TRIES = 80
SPUR_COUNT_RANGE = (2, 6)
MIN_DEFECTS = 3
MIN_SPURS = 2
MAX_GEN_ATTEMPTS = 400


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


def _absorb_black_spurs(grid: np.ndarray) -> np.ndarray:
    """Turn black spur pixels back to blue when their effective 4-neighborhood is mostly blue."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLACK:
                continue
            blue_n = 0
            oob = 0
            for dr, dc in DIR4:
                rr = int(r + dr)
                cc = int(c + dc)
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLUE:
                    blue_n += 1
            if oob >= 2:
                blue_n += int(oob)
            if blue_n >= 3:
                base[r, c] = BLUE
    return base


def _collect_anomaly_components(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    """Return non-black 4-components that are completely enclosed by black."""
    H, W = grid.shape
    comps: list[list[tuple[int, int]]] = []
    for coords in components_mask(grid != BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        comp_set = set(pts)
        enclosed = True
        touches_black = False
        for r, c in pts:
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    enclosed = False
                    break
                if (rr, cc) in comp_set:
                    continue
                if int(grid[rr, cc]) != BLACK:
                    enclosed = False
                    break
                touches_black = True
            if not enclosed:
                break
        if enclosed and touches_black:
            comps.append(pts)
    return comps


def _build_halo(grid: np.ndarray, preserve: set[tuple[int, int]]) -> np.ndarray:
    """Paint orange Chebyshev-1 halos while keeping anomaly centers in their original colors."""
    H, W = grid.shape
    out = np.array(grid, dtype=int)
    for r, c in preserve:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                    out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = int(grid[r, c])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Surround every non-black impurity inside black foreground with an orange halo.

    Input structure:
    - Blue (1) is the background and black (0) forms one or more large foreground regions.
    - Black regions may contain tiny enclosed non-black impurities; these can be blue or other colors such as red (2) or gray (5).
    - Blue background areas may also contain thin black spur pixels that should disappear before anomaly detection.

    Rule:
    - First absorb any black pixel whose effective 4-neighborhood is mostly blue, using the same corner counting idea as the source task but with colors reversed.
    - Then find every 4-connected non-black component that is fully enclosed by black on all four sides of each cell.
    - Paint orange (7) on the Chebyshev-1 neighborhood of each enclosed anomaly while preserving the anomaly pixels in their original non-black colors.

    Key insight: The structural test is no longer "small black or black leaf"; after reversing figure/background roles, the marked objects are any color impurities trapped inside black foreground, while black noise in blue background is merely cleaned away.
    """
    base = _absorb_black_spurs(np.array(grid, dtype=int))
    preserve: set[tuple[int, int]] = set()
    for comp in _collect_anomaly_components(base):
        preserve.update(comp)
    return _build_halo(base, preserve)


# --------------- generator ---------------

def _validate_regions(grid: np.ndarray) -> bool:
    blue_comps = list(components_mask(grid == BLUE, connectivity=4))
    black_comps = list(components_mask(grid == BLACK, connectivity=4))
    if not blue_comps or not black_comps:
        return False
    if max(len(comp) for comp in blue_comps) < MIN_BLUE_REGION:
        return False
    if min(len(comp) for comp in black_comps) < MIN_BLACK_REGION:
        return False
    return True


def _build_partition(H: int, W: int) -> np.ndarray:
    """Construct a blue-background / black-foreground scene with simple large shapes."""
    grid = np.full((H, W), BLUE, dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "h_stripes":
        n_stripes = random.randint(1, 2)
        starts = []
        tries = 0
        while len(starts) < n_stripes and tries < 80:
            tries += 1
            height = random.randint(4, max(4, H // 2))
            r0 = random.randint(1, H - height - 1)
            r1 = r0 + height - 1
            if any(not (r1 + 2 < s0 or r0 - 2 > s1) for s0, s1 in starts):
                continue
            starts.append((r0, r1))
        if not starts:
            return grid
        for r0, r1 in starts:
            grid[r0:r1 + 1, :] = BLACK

    elif style == "rect_inset":
        r0 = random.randint(2, H // 3)
        r1 = random.randint(2 * H // 3, H - 3)
        c0 = random.randint(2, W // 3)
        c1 = random.randint(2 * W // 3, W - 3)
        grid[r0:r1 + 1, c0:c1 + 1] = BLACK
        if random.random() < 0.35:
            side = random.choice(["left", "right", "top", "bottom"])
            if side in {"left", "right"}:
                bar_w = random.randint(3, max(3, W // 5))
                if side == "left":
                    c2 = max(1, c0 - bar_w)
                    grid[r0:r1 + 1, c2:c0 + 1] = BLACK
                else:
                    c2 = min(W - 2, c1 + bar_w)
                    grid[r0:r1 + 1, c1:c2 + 1] = BLACK
            else:
                bar_h = random.randint(3, max(3, H // 5))
                if side == "top":
                    r2 = max(1, r0 - bar_h)
                    grid[r2:r0 + 1, c0:c1 + 1] = BLACK
                else:
                    r2 = min(H - 2, r1 + bar_h)
                    grid[r1:r2 + 1, c0:c1 + 1] = BLACK

    else:  # l_shape
        vert_h = random.randint(H // 2, H - 3)
        vert_w = random.randint(4, max(4, W // 3))
        hor_h = random.randint(4, max(4, H // 3))
        hor_w = random.randint(W // 2, W - 3)
        r0 = random.randint(1, H - vert_h - 1)
        c0 = random.randint(1, W - vert_w - 1)
        grid[r0:r0 + vert_h, c0:c0 + vert_w] = BLACK
        if random.random() < 0.5:
            hr0 = random.randint(r0, min(H - hor_h - 1, r0 + vert_h - hor_h))
            hc0 = c0
        else:
            hr0 = random.randint(r0, min(H - hor_h - 1, r0 + vert_h - hor_h))
            hc0 = max(1, c0 + vert_w - 1)
            hor_w = min(hor_w, W - hc0 - 1)
        grid[hr0:hr0 + hor_h, hc0:hc0 + hor_w] = BLACK

    return grid


def _find_interior_black(grid: np.ndarray) -> list[tuple[int, int]]:
    black = grid == BLACK
    eroded = black.copy()
    eroded[:-1, :] &= black[1:, :]
    eroded[1:, :] &= black[:-1, :]
    eroded[:, :-1] &= black[:, 1:]
    eroded[:, 1:] &= black[:, :-1]
    rs, cs = np.where(eroded)
    coords = list(zip(rs.tolist(), cs.tolist()))
    random.shuffle(coords)
    return coords


def _anomaly_palette(count: int) -> list[int]:
    extra = random.choice([RED, GRAY])
    colors = [BLUE, extra]
    while len(colors) < count:
        colors.append(random.choice([BLUE, extra]))
    random.shuffle(colors)
    return colors


def _place_anomalies(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    """Place small non-black anomaly components strictly inside black regions."""
    H, W = grid.shape
    interior_black = _find_interior_black(grid)
    interior_set = set(interior_black)
    anomaly_pixels: set[tuple[int, int]] = set()
    defects_placed = 0
    n_defects = random.randint(*DEFECT_COUNT_RANGE)
    colors = _anomaly_palette(n_defects)

    for idx in range(n_defects):
        if not interior_black:
            break
        defect_size = random.choices([1, 2, 3], weights=DEFECT_SIZE_WEIGHTS)[0]
        defect_color = colors[idx]

        for _ in range(DEFECT_PLACE_TRIES):
            if not interior_black:
                break
            seed_r, seed_c = interior_black[random.randint(0, len(interior_black) - 1)]
            if _nearby(seed_r, seed_c, DEFECT_CLEARANCE, anomaly_pixels):
                continue

            defect = [(seed_r, seed_c)]
            if defect_size > 1:
                dirs = list(DIR4)
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr = seed_r + dr
                    nc = seed_c + dc
                    if (nr, nc) not in interior_set or (nr, nc) in anomaly_pixels:
                        continue
                    ok = True
                    for dr2, dc2 in DIR4:
                        rr = nr + dr2
                        cc = nc + dc2
                        if (rr, cc) == (seed_r, seed_c):
                            continue
                        if 0 <= rr < H and 0 <= cc < W:
                            if grid[rr, cc] != BLACK or (rr, cc) in anomaly_pixels:
                                ok = False
                                break
                    if ok and not _nearby(nr, nc, DEFECT_CLEARANCE, anomaly_pixels):
                        defect.append((nr, nc))
                        if len(defect) >= defect_size:
                            break

            defect_set = set(defect)
            valid = True
            for r, c in defect:
                for dr, dc in DIR4:
                    rr = r + dr
                    cc = c + dc
                    if (rr, cc) in defect_set:
                        continue
                    if not (0 <= rr < H and 0 <= cc < W):
                        valid = False
                        break
                    if grid[rr, cc] != BLACK or (rr, cc) in anomaly_pixels:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                for r, c in defect:
                    grid[r, c] = defect_color
                anomaly_pixels.update(defect)
                defects_placed += 1
                remove = _exclusion_zone(defect, DEFECT_CLEARANCE)
                interior_black[:] = [p for p in interior_black if p not in remove]
                interior_set -= remove
                break

    return anomaly_pixels, defects_placed


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Blue cells that can safely become black spurs and then be absorbed back to blue."""
    H, W = grid.shape
    blue = grid == BLUE

    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += blue[1:, :]
    nb[1:, :] += blue[:-1, :]
    nb[:, :-1] += blue[:, 1:]
    nb[:, 1:] += blue[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(blue & (effective >= 3))
    candidates = [
        (int(r), int(c))
        for r, c in zip(rs, cs)
        if (int(r), int(c)) not in exclude
    ]
    random.shuffle(candidates)
    return candidates


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    grid = _build_partition(H, W)
    if not _validate_regions(grid):
        return None

    defect_pixels, defects_placed = _place_anomalies(grid)
    if defects_placed < MIN_DEFECTS:
        return None

    colors_present = {int(grid[r, c]) for r, c in defect_pixels}
    if BLUE not in colors_present or len(colors_present - {BLUE}) == 0:
        return None

    output = _build_halo(grid, defect_pixels)

    inp = grid.copy()
    spur_count = random.randint(*SPUR_COUNT_RANGE)
    candidates = _find_spur_candidates(inp, defect_pixels)
    placed_spurs = 0
    for r, c in candidates:
        if placed_spurs >= spur_count:
            break
        inp[r, c] = BLACK
        placed_spurs += 1
    if placed_spurs < MIN_SPURS:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
