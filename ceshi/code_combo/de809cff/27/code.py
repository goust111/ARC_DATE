# Combo variant of de809cff (index 27)
# combo_spec: D3/3+D10/0
# Name: 5x5邻域判定+补丁大小随距离变化

from __future__ import annotations

# Design analysis for the fused variant:
# a) Concrete behavior and trace:
#    The base task denoises two large colored regions, finds black dents by a 3x3
#    local majority, and always stamps a 3x3 opposite-color patch. The fused rule
#    changes the same trigger in two coupled ways. First, dent ownership is read
#    from a cleaned 5x5 neighborhood, not a 3x3 one. Second, patch size is not
#    fixed: after assigning each black cell to a host region, the solver refills
#    all detected dents with their host colors and measures the largest centered
#    same-color square around each dent. Radius 1 gives 3x3, radius 2 gives 5x5,
#    radius 3 gives 7x7.
#    Small trace: consider a compact 2x3 block of black cells carved inside a
#    shallow red layer. For the two middle cells, the cleaned 3x3 view loses too
#    many red neighbors to nearby dents, but the cleaned 5x5 still has a clear red
#    majority, so the whole block is restored to red conceptually before measuring
#    square depth. Those shallow dents then produce small blue patches with teal
#    centers, while a deeper single dent elsewhere in the same region might have
#    restored square radius 3 and become a 7x7 blue patch. The fused effect is:
#    broader context decides who owns the hole, and restored depth decides how big
#    its patch becomes.
# b) Complexity calibration:
#    If every dent were an isolated interior singleton, the 5x5 ownership test
#    would collapse back to the old 3x3 behavior. To keep D3/3 visible, the
#    generator always includes at least one compact multi-cell dent block whose
#    middle cells are detected by the 5x5 rule but not by the legacy 3x3 rule. To
#    keep D10 visible, every sample also contains repeated radius tiers
#    1/2/3, so the output shows multiple 3x3/5x5/7x7 patches rather than a single
#    accidental large square.
# c) Hidden assumptions audited:
#    - Base `_dent_centers` assumed dent ownership can be decided from the same
#      local 3x3 that also justifies stamping. That breaks once 5x5 context is
#      required, so host assignment and patch-radius measurement are now separate
#      steps.
#    - The D10 single-variant implementation measured radius directly on the
#      dent-blackened cleaned grid. That makes adjacent dents look shallower than
#      the underlying region really is. Under this fusion, radius must be measured
#      after refilling every detected dent with its inferred host color; otherwise
#      the new 5x5-only dent block would never support even radius 1.
#    - The base code could tolerate dense 3x3 overlaps because every patch had the
#      same size. Variable radii make overlap order matter more, so the generator
#      spaces large dents by their measured radii and reserves only the rare local
#      overlaps inside one intended dent block.
# d) Placement legality:
#    Candidate dent cells are chosen from the fully clean regions before blackening.
#    Singles are sampled from exact radius buckets 1/2/3 on the clean region, so
#    their scaled patch sizes are known in advance. A special 2x3 or 3x2 dent
#    block is sampled only when blackening the whole block makes 5x5 succeed and
#    3x3 miss at least two cells; all block cells still stay at radius 1 or 2, so
#    the output remains compact. Noise is forbidden near any dent footprint,
#    keeping cleanup stable.
# e) Visual observability:
#    A human sees two repeated cues across examples: some black holes expand into
#    larger 5x5 or 7x7 opposite-color squares, and some cells inside one compact
#    dent block still transform even though their immediate 3x3 context looks
#    ambiguous. No
#    rule depends on hidden numeric ordering of color ids.
# f) Visual cleanliness:
#    Only two foreground colors plus teal are used. Large patches are kept apart by
#    radius-aware spacing, while the 5x5-only effect is concentrated into one
#    compact local block. This keeps the output readable instead of turning it
#    into a field of heavily overlapping large squares.
# g) Inferability of the color change:
#    Color mapping stays simple and structural: every transformed dent uses the
#    opposite large-region color, and teal only marks the dent centers. The only
#    new inference burden is geometric: wider context decides region ownership, and
#    greater restored depth yields a larger odd square.
# 9) New solver assumptions checked against the generator:
#    - Every true dent has a unique host-color majority in the cleaned 5x5 window;
#      the generator validates that directly.
#    - Measuring radius after refilling detected dents is safe because the generator
#      only creates black holes that belong to the two foreground regions; no
#      isolated background black cell is allowed to satisfy the same 5x5 criterion.
#    - Large transformed squares do not need deterministic tie-breaking from overlap
#      order because the generator keeps distinct large dents beyond the sum of
#      their radii. Only cells inside a single chosen dent block may overlap, and
#      those use the same patch color.

from collections import Counter as _Counter
from pathlib import Path
import random
import sys

import numpy as np
from scipy.ndimage import label

try:
    from arc_utils import Color
except ImportError:  # pragma: no cover - fallback for direct file loading
    sys.path.append(str(Path(__file__).resolve().parents[4] / "arc2_solver_select"))
    from arc_utils import Color


def _main_colors(g: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    colors.sort(key=lambda c: int(np.sum(g == c)), reverse=True)
    assert len(colors) >= 2, "expected two foreground colors"
    return int(colors[0]), int(colors[1])


def _other_color(color: int, colors: tuple[int, int]) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


def _sub3x3(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)]


def _sub5x5(grid: np.ndarray, r: int, c: int) -> np.ndarray:
    h, w = grid.shape
    return grid[max(0, r - 2) : min(h, r + 3), max(0, c - 2) : min(w, c + 3)]


def _has_same_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = g.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                if bool(np.all(g[rr : rr + 2, cc : cc + 2] == int(color))):
                    return True
    return False


def _clean_foreground(g: np.ndarray, colors: tuple[int, int], bg: int) -> np.ndarray:
    """
    Remove one-cell specks/protrusions before reading dent ownership and depth.

    The corruption model is unchanged from the base task: one sweep clears local
    outliers, and the second lets neighbors stabilize around the corrected local
    majority. True black dents remain black through this cleanup.
    """
    h, w = g.shape
    a, b = map(int, colors)
    cur = g.copy()

    for _ in range(2):
        nxt = cur.copy()
        for r in range(h):
            for c in range(w):
                v = int(cur[r, c])
                if v not in colors:
                    nxt[r, c] = int(bg)
                    continue

                sub = _sub3x3(cur, r, c)
                own = int(np.sum(sub == int(v)))
                other = b if v == a else a
                other_count = int(np.sum(sub == int(other)))
                bg_count = int(np.sum(sub == int(bg)))

                if _has_same_2x2(cur, r, c, v):
                    nxt[r, c] = int(v)
                elif other_count > own and other_count >= 5:
                    nxt[r, c] = int(other)
                elif bg_count >= 5 and own <= 4:
                    nxt[r, c] = int(bg)
                elif own <= 2:
                    nxt[r, c] = int(bg)
                else:
                    nxt[r, c] = int(v)
        cur = nxt

    return cur


DENT_MIN_HOST_3X3 = 5
DENT_MIN_HOST_5X5 = 10


def _host_from_window(sub: np.ndarray, colors: tuple[int, int], min_host: int) -> int | None:
    counts = {int(color): int(np.sum(sub == int(color))) for color in colors}
    host = max(colors, key=lambda color: counts[int(color)])
    other = _other_color(int(host), colors)
    if counts[int(host)] < int(min_host) or counts[int(host)] <= counts[int(other)]:
        return None
    return int(host)


def _dent_candidates(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    window: int,
) -> list[tuple[int, int, int]]:
    assert window in (3, 5)
    min_host = DENT_MIN_HOST_3X3 if int(window) == 3 else DENT_MIN_HOST_5X5
    candidates: list[tuple[int, int, int]] = []
    for r, c in zip(*np.where(original == int(bg)), strict=False):
        rr = int(r)
        cc = int(c)
        sub = _sub3x3(clean, rr, cc) if int(window) == 3 else _sub5x5(clean, rr, cc)
        host = _host_from_window(sub, colors, min_host)
        if host is not None:
            candidates.append((rr, cc, int(host)))
    return candidates


def _restore_detected_dents(clean: np.ndarray, dent_hosts: list[tuple[int, int, int]]) -> np.ndarray:
    restored = clean.copy()
    for r, c, host in dent_hosts:
        restored[int(r), int(c)] = int(host)
    return restored


def _square_support_radius(restored: np.ndarray, r: int, c: int, host: int) -> int:
    """Largest centered all-host square radius after dent cells are refilled."""
    if int(restored[r, c]) != int(host):
        return 0
    h, w = restored.shape
    radius = 0
    probe = 0
    while True:
        r0 = int(r) - probe
        c0 = int(c) - probe
        r1 = int(r) + probe + 1
        c1 = int(c) + probe + 1
        if r0 < 0 or c0 < 0 or r1 > h or c1 > w:
            break
        if not bool(np.all(restored[r0:r1, c0:c1] == int(host))):
            break
        radius = probe
        probe += 1
    return int(radius)


def _dent_specs_from_window(
    original: np.ndarray,
    clean: np.ndarray,
    colors: tuple[int, int],
    bg: int,
    window: int,
) -> list[tuple[int, int, int, int]]:
    dent_hosts = _dent_candidates(original, clean, colors, bg, window)
    restored = _restore_detected_dents(clean, dent_hosts)
    specs: list[tuple[int, int, int, int]] = []
    for r, c, host in dent_hosts:
        radius = _square_support_radius(restored, int(r), int(c), int(host))
        if radius >= 1:
            specs.append((int(r), int(c), int(host), int(radius)))
    return specs


def _render_from_specs(
    clean: np.ndarray, dent_specs: list[tuple[int, int, int, int]], colors: tuple[int, int]
) -> np.ndarray:
    h, w = clean.shape
    out = clean.copy()
    for r, c, host, radius in dent_specs:
        other = _other_color(int(host), colors)
        out[max(0, r - radius) : min(h, r + radius + 1), max(0, c - radius) : min(w, c + radius + 1)] = int(other)
    for r, c, _host, _radius in dent_specs:
        out[int(r), int(c)] = int(Color.TEAL)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn black dents into teal-marked opposite-color squares with adaptive size.

    Input structure:
    - Black background with exactly two large foreground regions.
    - Inside those regions are true black dents, plus a small amount of one-cell
      foreground noise in the background or along the region boundary.

    Rule:
    - First denoise the foreground exactly as in the base task: isolated specks
      vanish to black, and one-cell cross-boundary protrusions revert to the
      locally dominant region color.
    - For each original black cell, use the cleaned 5x5 neighborhood to decide
      which foreground region owns that dent. Refill every detected dent with its
      host color conceptually, then measure the largest centered same-color square
      around the dent in that restored region.
    - Stamp the opposite foreground color on the centered square of radius d
      (3x3 for d=1, 5x5 for d=2, 7x7 for d=3, clipped at the grid boundary), and
      recolor the dent center itself teal (8).

    Key insight: The black holes are still the triggers, but ownership now comes
    from a broader 5x5 context, while patch size comes from how deeply that dent
    sits inside its restored host region.
    """
    g = np.asarray(grid, dtype=int)
    bg = Color.BLACK
    colors = _main_colors(g, bg)
    clean = _clean_foreground(g, colors, bg)
    dent_specs = _dent_specs_from_window(g, clean, colors, bg, window=5)
    return _render_from_specs(clean, dent_specs, colors)


# --------------- generator (combo) ---------------

# --- CFG: canvas & region layout ---
GRID_SIZE_MIN, GRID_SIZE_MAX = 22, 30
RECT_SIZE_FRAC_MIN, RECT_SIZE_FRAC_MAX = 0.45, 0.75
RECT_MARGIN_MAX = 3
MIN_REGION_CELLS = 100
COVERAGE_MIN, COVERAGE_MAX = 0.55, 0.85

# --- CFG: dent placement ---
TARGET_RADII = (1, 2, 3)
SINGLE_COUNT_MIN, SINGLE_COUNT_MAX = 6, 8
BLOCK_GROUP_COUNT = 1
REQUIRED_RADIUS3_COUNT = 2
EXTRA_SPACING_GAP = 1
REQUIRED_ONLY5_CELLS = 2
BLOCK_MAX_RADIUS = 2

# --- CFG: noise ---
NOISE_COUNT_MIN, NOISE_COUNT_MAX = 7, 18
BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX = 0.65, 0.85
BG_SPECK_MIN_BG_3x3 = 6
BOUNDARY_MIN_HOST_3x3 = 6

# --- CFG: generation control ---
MAX_GENERATION_ATTEMPTS = 500


def _build_regions(N: int, color_a: int, color_b: int, rng: random.Random) -> np.ndarray | None:
    """Paint 2-3 overlapping rectangles with simple occlusion."""
    n_rects = rng.choice([2, 3])
    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    chosen_corners = corners[:n_rects]

    if n_rects == 2:
        rect_colors = [color_a, color_b]
        rng.shuffle(rect_colors)
    else:
        first = rng.choice([color_a, color_b])
        mid = color_b if first == color_a else color_a
        rect_colors = [first, mid, first]

    def _make_rect(corner: str) -> tuple[int, int, int, int]:
        h = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        w = rng.randint(int(N * RECT_SIZE_FRAC_MIN), int(N * RECT_SIZE_FRAC_MAX))
        m = rng.randint(0, RECT_MARGIN_MAX)
        r0 = m if corner[0] == "T" else max(0, N - h - m)
        c0 = m if corner[1] == "L" else max(0, N - w - m)
        return r0, c0, min(N, r0 + h), min(N, c0 + w)

    grid = np.zeros((N, N), dtype=int)
    for corner, color in zip(chosen_corners, rect_colors):
        r0, c0, r1, c1 = _make_rect(corner)
        grid[r0:r1, c0:c1] = int(color)

    cnt_a = int(np.sum(grid == int(color_a)))
    cnt_b = int(np.sum(grid == int(color_b)))
    if cnt_a < MIN_REGION_CELLS or cnt_b < MIN_REGION_CELLS:
        return None
    coverage = (cnt_a + cnt_b) / (N * N)
    if not (COVERAGE_MIN <= coverage <= COVERAGE_MAX):
        return None
    if n_rects == 3:
        _, n_components = label(grid == int(rect_colors[0]))
        if int(n_components) < 2:
            return None
    return grid


def _full_square_radius(clean: np.ndarray, r: int, c: int, host: int) -> int:
    return _square_support_radius(clean, int(r), int(c), int(host))


def _extract_radius_buckets(
    clean: np.ndarray, color_a: int, color_b: int
) -> dict[int, dict[int, list[tuple[int, int]]]]:
    buckets: dict[int, dict[int, list[tuple[int, int]]]] = {
        int(color_a): {radius: [] for radius in TARGET_RADII},
        int(color_b): {radius: [] for radius in TARGET_RADII},
    }
    h, w = clean.shape
    for r in range(h):
        for c in range(w):
            v = int(clean[r, c])
            if v not in buckets:
                continue
            radius = _full_square_radius(clean, int(r), int(c), v)
            if radius in TARGET_RADII:
                buckets[v][int(radius)].append((int(r), int(c)))
    return buckets


def _far_enough(
    selected: list[tuple[int, int, int, int]],
    r: int,
    c: int,
    radius: int,
    gap: int = EXTRA_SPACING_GAP,
) -> bool:
    for rr, cc, _host, old_radius in selected:
        if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) <= int(radius) + int(old_radius) + int(gap):
            return False
    return True


def _pick_from_bucket(
    bucket: list[tuple[int, int]],
    radius: int,
    host: int,
    selected: list[tuple[int, int, int, int]],
    used: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    choices = bucket[:]
    rng.shuffle(choices)
    for r, c in choices:
        if (int(r), int(c)) in used:
            continue
        if _far_enough(selected, int(r), int(c), int(radius)):
            used.add((int(r), int(c)))
            return int(r), int(c), int(host), int(radius)
    return None


def _select_scaling_singles(
    buckets: dict[int, dict[int, list[tuple[int, int]]]],
    color_a: int,
    color_b: int,
    rng: random.Random,
    seed_specs: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]] | None:
    selected: list[tuple[int, int, int, int]] = [] if seed_specs is None else seed_specs[:]
    used: set[tuple[int, int]] = {(int(r), int(c)) for r, c, _host, _radius in selected}

    required = [
        (int(color_a), 1),
        (int(color_a), 2),
        (int(color_b), 1),
        (int(color_b), 2),
    ]
    for host, radius in required:
        spec = _pick_from_bucket(buckets[host][radius], radius, host, selected, used, rng)
        if spec is None:
            return None
        selected.append(spec)

    radius3_pool = [
        (int(host), int(r), int(c))
        for host in (int(color_a), int(color_b))
        for r, c in buckets[host][3]
        if (int(r), int(c)) not in used
    ]
    rng.shuffle(radius3_pool)
    radius3_added = 0
    for host, r, c in radius3_pool:
        if radius3_added >= REQUIRED_RADIUS3_COUNT:
            break
        if not _far_enough(selected, int(r), int(c), 3):
            continue
        used.add((int(r), int(c)))
        selected.append((int(r), int(c), int(host), 3))
        radius3_added += 1
    if radius3_added < REQUIRED_RADIUS3_COUNT:
        return None

    target_total = rng.randint(SINGLE_COUNT_MIN, SINGLE_COUNT_MAX)
    extras = [
        (int(radius), int(host), int(r), int(c))
        for host in (int(color_a), int(color_b))
        for radius in TARGET_RADII
        for r, c in buckets[host][radius]
        if (int(r), int(c)) not in used
    ]
    rng.shuffle(extras)
    for radius, host, r, c in extras:
        if len(selected) >= target_total:
            break
        if not _far_enough(selected, int(r), int(c), int(radius)):
            continue
        used.add((int(r), int(c)))
        selected.append((int(r), int(c), int(host), int(radius)))

    return selected


def _block_only5_score(clean: np.ndarray, colors: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    cell_set = {tuple(map(int, cell)) for cell in cells}
    host = int(clean[cells[0][0], cells[0][1]])
    other = _other_color(host, colors)
    score = 0

    for r, c in cells:
        sub3 = _sub3x3(clean, int(r), int(c))
        sub5 = _sub5x5(clean, int(r), int(c))
        r3, c3 = max(0, int(r) - 1), max(0, int(c) - 1)
        r5, c5 = max(0, int(r) - 2), max(0, int(c) - 2)

        removed3 = sum(1 for rr, cc in cell_set if r3 <= rr < r3 + sub3.shape[0] and c3 <= cc < c3 + sub3.shape[1])
        removed5 = sum(1 for rr, cc in cell_set if r5 <= rr < r5 + sub5.shape[0] and c5 <= cc < c5 + sub5.shape[1])
        host3 = int(np.sum(sub3 == int(host))) - removed3
        host5 = int(np.sum(sub5 == int(host))) - removed5
        other3 = int(np.sum(sub3 == int(other)))
        other5 = int(np.sum(sub5 == int(other)))
        pass3 = host3 >= DENT_MIN_HOST_3X3 and host3 > other3
        pass5 = host5 >= DENT_MIN_HOST_5X5 and host5 > other5
        if not pass5:
            return -1
        if not pass3:
            score += 1

    return int(score)


def _collect_block_candidates(
    clean: np.ndarray, colors: tuple[int, int]
) -> list[tuple[int, int, list[tuple[int, int, int, int]]]]:
    candidates: list[tuple[int, int, list[tuple[int, int, int, int]]]] = []
    h, w = clean.shape
    shapes = (
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)],
    )
    for shape in shapes:
        max_r = max(dr for dr, _dc in shape)
        max_c = max(dc for _dr, dc in shape)
        for r in range(h - max_r):
            for c in range(w - max_c):
                cells = [(int(r + dr), int(c + dc)) for dr, dc in shape]
                host = int(clean[cells[0][0], cells[0][1]])
                if host not in colors:
                    continue
                if any(int(clean[rr, cc]) != int(host) for rr, cc in cells):
                    continue
                radii = [_full_square_radius(clean, rr, cc, host) for rr, cc in cells]
                if min(radii) < 1 or max(radii) > BLOCK_MAX_RADIUS:
                    continue
                score = _block_only5_score(clean, colors, cells)
                if score >= REQUIRED_ONLY5_CELLS:
                    specs = [(int(rr), int(cc), int(host), int(radius)) for (rr, cc), radius in zip(cells, radii, strict=True)]
                    candidates.append((int(score), int(max(radii)), specs))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates


def _select_special_block(
    clean: np.ndarray,
    colors: tuple[int, int],
    selected: list[tuple[int, int, int, int]],
    used: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    candidates = _collect_block_candidates(clean, colors)
    if not candidates:
        return []
    top_score = candidates[0][0]
    choices = [item for item in candidates if item[0] == top_score]
    rng.shuffle(choices)
    for _score, _max_radius, specs in choices + candidates[len(choices) :]:
        cells = [(int(r), int(c)) for r, c, _host, _radius in specs]
        if any(cell in used for cell in cells):
            continue
        if not all(_far_enough(selected, int(r), int(c), int(radius)) for r, c, _host, radius in specs):
            continue
        for cell in cells:
            used.add(cell)
        return specs
    return []


def _build_protected_mask(shape: tuple[int, int], dent_specs: list[tuple[int, int, int, int]]) -> np.ndarray:
    h, w = shape
    protected = np.zeros((h, w), dtype=bool)
    for r, c, _host, radius in dent_specs:
        pad = int(radius) + 1
        protected[max(0, r - pad) : min(h, r + pad + 1), max(0, c - pad) : min(w, c + pad + 1)] = True
    return protected


def _would_form_2x2(grid: np.ndarray, r: int, c: int, color: int) -> bool:
    h, w = grid.shape
    for dr in (-1, 0):
        for dc in (-1, 0):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h - 1 and 0 <= cc < w - 1:
                block = grid[rr : rr + 2, cc : cc + 2].copy()
                block[r - rr, c - cc] = int(color)
                if int(np.sum(block == int(color))) == 4:
                    return True
    return False


def _neighbor_cleanup_safe(inp: np.ndarray, r: int, c: int, host: int, other: int) -> bool:
    h, w = inp.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or int(inp[nr, nc]) != int(host):
                continue
            nsub = _sub3x3(inp, nr, nc).copy()
            nr0, nc0 = max(0, nr - 1), max(0, nc - 1)
            nsub[r - nr0, c - nc0] = int(other)
            own_n = int(np.sum(nsub == int(host)))
            other_n = int(np.sum(nsub == int(other)))
            bg_n = int(np.sum(nsub == int(Color.BLACK)))
            if (other_n > own_n and other_n >= 5) or own_n <= 2 or (bg_n >= 5 and own_n <= 4):
                return False
    return True


def _add_bg_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: np.ndarray,
    n_in_bg: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    bg_candidates = list(zip(*np.where((clean == int(Color.BLACK)) & (~protected)), strict=False))
    rng.shuffle(bg_candidates)

    placed = 0
    for r_, c_ in bg_candidates:
        if placed >= n_in_bg:
            break
        r, c = int(r_), int(c_)
        sub = _sub3x3(inp, r, c)
        if int(np.sum(sub == int(Color.BLACK))) < BG_SPECK_MIN_BG_3x3:
            continue
        nearby = _sub3x3(clean, r, c)
        nearby_fg = [int(x) for x in nearby.flat if int(x) != int(Color.BLACK)]
        speck_color = _Counter(nearby_fg).most_common(1)[0][0] if nearby_fg else rng.choice([color_a, color_b])
        inp[r, c] = int(speck_color)
        placed += 1


def _add_boundary_noise(
    inp: np.ndarray,
    clean: np.ndarray,
    protected: np.ndarray,
    n_at_boundary: int,
    color_a: int,
    color_b: int,
    rng: random.Random,
) -> None:
    h, w = clean.shape
    boundary_candidates = []
    for r in range(h):
        for c in range(w):
            if protected[r, c]:
                continue
            v = int(clean[r, c])
            if v not in (int(color_a), int(color_b)):
                continue
            other = _other_color(v, (int(color_a), int(color_b)))
            if int(np.sum(_sub3x3(clean, r, c) == int(other))) >= 1:
                boundary_candidates.append((int(r), int(c), int(v), int(other)))
    rng.shuffle(boundary_candidates)

    placed = 0
    for r, c, host, other in boundary_candidates:
        if placed >= n_at_boundary:
            break
        if int(inp[r, c]) != int(host):
            continue
        sub_now = _sub3x3(inp, r, c)
        host_count = int(np.sum(sub_now == int(host)))
        other_count = int(np.sum(sub_now == int(other)))
        if host_count < BOUNDARY_MIN_HOST_3x3 or host_count - 1 <= other_count + 1:
            continue
        if _would_form_2x2(inp, r, c, other):
            continue
        if not _neighbor_cleanup_safe(inp, r, c, host, other):
            continue
        inp[r, c] = int(other)
        placed += 1


def _legacy_output_for_input(inp: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    clean = _clean_foreground(inp, colors, Color.BLACK)
    legacy_specs = _dent_specs_from_window(inp, clean, colors, Color.BLACK, window=3)
    return _render_from_specs(clean, legacy_specs, colors)


def _sample_metrics(inp: np.ndarray, colors: tuple[int, int]) -> tuple[int, dict[int, int], bool]:
    clean = _clean_foreground(inp, colors, Color.BLACK)
    specs5 = _dent_specs_from_window(inp, clean, colors, Color.BLACK, window=5)
    specs3 = _dent_specs_from_window(inp, clean, colors, Color.BLACK, window=3)
    centers5 = {(int(r), int(c)) for r, c, _host, _radius in specs5}
    centers3 = {(int(r), int(c)) for r, c, _host, _radius in specs3}
    only5 = int(len(centers5 - centers3))
    radius_hist = dict(_Counter(int(radius) for _r, _c, _host, radius in specs5))
    fused_out = _render_from_specs(clean, specs5, colors)
    legacy_out = _render_from_specs(clean, specs3, colors)
    return only5, radius_hist, bool(not np.array_equal(fused_out, legacy_out))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    N = rng.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    fg_pool = [c for c in Color.NOT_BLACK if c != Color.TEAL]
    color_a, color_b = rng.sample(fg_pool, 2)
    colors = (int(color_a), int(color_b))

    clean = _build_regions(N, color_a, color_b, rng)
    if clean is None:
        return None

    buckets = _extract_radius_buckets(clean, color_a, color_b)
    if any(not buckets[int(color_a)][radius] for radius in (1, 2)):
        return None
    if any(not buckets[int(color_b)][radius] for radius in (1, 2)):
        return None
    if len(buckets[int(color_a)][3]) + len(buckets[int(color_b)][3]) < REQUIRED_RADIUS3_COUNT:
        return None

    used: set[tuple[int, int]] = set()
    block = _select_special_block(clean, colors, [], used, rng)
    if len(block) < REQUIRED_ONLY5_CELLS:
        return None
    singles = _select_scaling_singles(buckets, color_a, color_b, rng, seed_specs=block)
    if singles is None:
        return None

    dent_specs = singles
    output = _render_from_specs(clean, dent_specs, colors)

    inp = clean.copy()
    for r, c, _host, _radius in dent_specs:
        inp[int(r), int(c)] = int(Color.BLACK)

    protected = _build_protected_mask(clean.shape, dent_specs)
    n_noise = rng.randint(NOISE_COUNT_MIN, NOISE_COUNT_MAX)
    n_in_bg = int(round(n_noise * rng.uniform(BG_NOISE_FRAC_MIN, BG_NOISE_FRAC_MAX)))
    _add_bg_noise(inp, clean, protected, n_in_bg, color_a, color_b, rng)
    _add_boundary_noise(inp, clean, protected, n_noise - n_in_bg, color_a, color_b, rng)

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, output):
        return None

    only5, radius_hist, differs_from_legacy = _sample_metrics(inp, colors)
    if only5 < REQUIRED_ONLY5_CELLS:
        return None
    if radius_hist.get(2, 0) < 2 or radius_hist.get(3, 0) < 1:
        return None
    if not differs_from_legacy:
        return None

    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after %d attempts" % MAX_GENERATION_ATTEMPTS)
