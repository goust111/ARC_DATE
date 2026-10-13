import random
from typing import Iterable

import numpy as np


QUAD_SIZE = 9
CROSS_COLOR = 6
GRID_SIZE = QUAD_SIZE * 2 + 1
OCCLUDER_COLORS = (1, 2, 3, 4)
BASE_COLORS = (5, 7, 8, 9)
QUADRANT_NAMES = ("TL", "TR", "BL", "BR")
GEN_RETRIES = 400
RECT_RETRIES = 1500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = np.where(np.all(grid == CROSS_COLOR, axis=1))[0]
    cols = np.where(np.all(grid == CROSS_COLOR, axis=0))[0]
    assert len(rows) == 1 and len(cols) == 1, "expected a unique full magenta cross"
    return int(rows[0]), int(cols[0])


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    cross_r, cross_c = _find_cross(grid)
    assert cross_r == QUAD_SIZE and cross_c == QUAD_SIZE
    quadrants = [
        grid[:cross_r, :cross_c],
        grid[:cross_r, cross_c + 1 :],
        grid[cross_r + 1 :, :cross_c],
        grid[cross_r + 1 :, cross_c + 1 :],
    ]
    assert all(q.shape == (QUAD_SIZE, QUAD_SIZE) for q in quadrants)
    return quadrants


def _find_occluder(panel: np.ndarray) -> dict:
    mask = np.isin(panel, OCCLUDER_COLORS)
    colors = sorted(int(v) for v in np.unique(panel[mask]))
    assert len(colors) == 1, "each quadrant must contain exactly one occluder color"
    comps = _connected_components(mask)
    assert len(comps) == 1, "occluder colors must form a single component"
    comp_mask = np.zeros_like(mask, dtype=bool)
    coords = comps[0]
    comp_mask[coords[:, 0], coords[:, 1]] = True
    r0, c0, r1, c1 = _bbox(comp_mask)
    rect = np.zeros_like(mask, dtype=bool)
    rect[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.array_equal(comp_mask, rect), "occluder must be a solid rectangle"
    color = colors[0]
    assert np.all(panel[rect] == color), "occluder rectangle must be monochrome"
    return {
        "color": color,
        "mask": rect,
        "bbox": (r0, c0, r1, c1),
        "area": int(rect.sum()),
    }


def _visible_density(panel: np.ndarray) -> int:
    visible = (panel != 0) & ~np.isin(panel, OCCLUDER_COLORS)
    return int(visible.sum())


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the complete base pattern from four differently occluded quadrant views.

    Input structure:
    - A 19x19 grid is split by one full magenta row and one full magenta column into four 9x9 quadrants.
    - Each quadrant shows the same underlying 9x9 multicolor pattern on black background.
    - Every quadrant contains one solid rectangular occluder, and its color is uniquely chosen from {blue, red, green, yellow}.

    Rule:
    - Remove the magenta cross and align the four quadrants by local coordinates.
    - In each quadrant, detect the unique solid rectangle whose color belongs to {1,2,3,4}; that rectangle marks the occluded cells.
    - Compute each quadrant's visible foreground density as the number of non-black, non-occluder pixels it still shows.
    - For each local coordinate, use any quadrant whose occluder does not cover that coordinate; if several are available, take the pixel from the available quadrant with the highest visible density.

    Key insight:
    - The four quadrants are complementary partial observations of one hidden pattern, so the task is quadrant-wise restoration using occluder masks plus density-based donor priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), "expected a 19x19 grid with four 9x9 quadrants"

    panels = _split_quadrants(g)
    occluders = [_find_occluder(panel) for panel in panels]
    densities = [_visible_density(panel) for panel in panels]

    out = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    for r in range(QUAD_SIZE):
        for c in range(QUAD_SIZE):
            available = [idx for idx, occ in enumerate(occluders) if not occ["mask"][r, c]]
            if not available:
                continue
            best = max(available, key=lambda idx: densities[idx])
            out[r, c] = int(panels[best][r, c])
    return out


def _paint_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0 : c0 + w] = color


def _paint_hline(grid: np.ndarray, r: int, c0: int, w: int, color: int) -> None:
    grid[r, c0 : c0 + w] = color


def _paint_vline(grid: np.ndarray, r0: int, c: int, h: int, color: int) -> None:
    grid[r0 : r0 + h, c] = color


def _paint_l(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    if random.random() < 0.5:
        _paint_vline(grid, r0, c0, h, color)
        _paint_hline(grid, r0 + h - 1, c0, w, color)
    else:
        _paint_hline(grid, r0, c0, w, color)
        _paint_vline(grid, r0, c0 + w - 1, h, color)


def _foreground_component_count(base: np.ndarray) -> int:
    return len(_connected_components(base != 0))


def _make_base_pattern() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        base = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
        palette = random.sample(list(BASE_COLORS), random.randint(2, len(BASE_COLORS)))
        steps = random.randint(5, 9)
        for _ in range(steps):
            color = random.choice(palette)
            shape_type = random.choice(("rect", "hline", "vline", "lshape"))
            if shape_type == "rect":
                h = random.randint(1, 3)
                w = random.randint(1, 4)
                r0 = random.randint(0, QUAD_SIZE - h)
                c0 = random.randint(0, QUAD_SIZE - w)
                _paint_rect(base, r0, c0, h, w, color)
            elif shape_type == "hline":
                w = random.randint(2, 5)
                r = random.randint(0, QUAD_SIZE - 1)
                c0 = random.randint(0, QUAD_SIZE - w)
                _paint_hline(base, r, c0, w, color)
            elif shape_type == "vline":
                h = random.randint(2, 5)
                r0 = random.randint(0, QUAD_SIZE - h)
                c = random.randint(0, QUAD_SIZE - 1)
                _paint_vline(base, r0, c, h, color)
            else:
                h = random.randint(2, 4)
                w = random.randint(2, 4)
                r0 = random.randint(0, QUAD_SIZE - h)
                c0 = random.randint(0, QUAD_SIZE - w)
                _paint_l(base, r0, c0, h, w, color)

        for _ in range(random.randint(2, 5)):
            base[random.randint(0, QUAD_SIZE - 1), random.randint(0, QUAD_SIZE - 1)] = random.choice(palette)

        fg = int(np.count_nonzero(base))
        used = sorted(int(v) for v in np.unique(base) if v != 0)
        rows_used = int(np.count_nonzero(np.any(base != 0, axis=1)))
        cols_used = int(np.count_nonzero(np.any(base != 0, axis=0)))
        if not (20 <= fg <= 48):
            continue
        if len(used) < 2:
            continue
        if rows_used < 5 or cols_used < 5:
            continue
        if _foreground_component_count(base) < 2:
            continue
        return base
    raise RuntimeError("failed to generate a diverse 9x9 base pattern")


def _rect_mask(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    mask = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _rect_contains(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(~b | a))


def _enumerate_candidates(base: np.ndarray) -> list[dict]:
    candidates = []
    total_fg = int(np.count_nonzero(base))
    fg_mask = base != 0
    for h in (3, 4):
        for w in (3, 4):
            for r0 in range(QUAD_SIZE - h + 1):
                for c0 in range(QUAD_SIZE - w + 1):
                    mask = _rect_mask(r0, c0, h, w)
                    covered_fg = int(np.count_nonzero(fg_mask & mask))
                    covered_bg = int(mask.sum()) - covered_fg
                    if covered_fg < 2 or covered_bg < 2:
                        continue
                    candidates.append(
                        {
                            "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                            "mask": mask,
                            "covered_fg": covered_fg,
                            "visible_density": total_fg - covered_fg,
                        }
                    )
    return candidates


def _choose_rectangles(base: np.ndarray) -> list[dict]:
    candidates = _enumerate_candidates(base)
    if len(candidates) < 4:
        raise RuntimeError("not enough occluder candidates")

    for _ in range(RECT_RETRIES):
        chosen = random.sample(candidates, 4)
        covered_counts = [cand["covered_fg"] for cand in chosen]
        if len(set(covered_counts)) != 4:
            continue

        valid_pairwise = True
        for i in range(4):
            for j in range(i + 1, 4):
                mask_i = chosen[i]["mask"]
                mask_j = chosen[j]["mask"]
                if np.array_equal(mask_i, mask_j):
                    valid_pairwise = False
                    break
                if _rect_contains(mask_i, mask_j) or _rect_contains(mask_j, mask_i):
                    valid_pairwise = False
                    break
            if not valid_pairwise:
                break
        if not valid_pairwise:
            continue

        masks = np.stack([cand["mask"] for cand in chosen], axis=0)
        coverage = masks.sum(axis=0)
        if int(coverage.max()) > 3:
            continue
        if not np.any(coverage == 3):
            continue

        densities = [cand["visible_density"] for cand in chosen]
        best_idx = int(np.argmax(densities))
        if not np.any(masks[best_idx] & (coverage <= 2)):
            continue

        provenance = np.full((QUAD_SIZE, QUAD_SIZE), -1, dtype=int)
        for r in range(QUAD_SIZE):
            for c in range(QUAD_SIZE):
                available = [idx for idx in range(4) if not masks[idx, r, c]]
                if not available:
                    break
                provenance[r, c] = max(available, key=lambda idx: densities[idx])
        if np.any(provenance < 0):
            continue

        if len(set(int(v) for v in provenance.flat)) < 2:
            continue

        naive = np.full((QUAD_SIZE, QUAD_SIZE), -1, dtype=int)
        for r in range(QUAD_SIZE):
            for c in range(QUAD_SIZE):
                for idx in range(4):
                    if not masks[idx, r, c]:
                        naive[r, c] = idx
                        break
        if not np.any((coverage <= 2) & (provenance != naive)):
            continue
        return chosen

    raise RuntimeError("failed to choose non-degenerate occluder rectangles")


def _assemble_input(base: np.ndarray, rectangles: list[dict]) -> np.ndarray:
    occ_colors = random.sample(list(OCCLUDER_COLORS), 4)
    grid = np.full((GRID_SIZE, GRID_SIZE), CROSS_COLOR, dtype=int)
    anchors = {
        "TL": (0, 0),
        "TR": (0, QUAD_SIZE + 1),
        "BL": (QUAD_SIZE + 1, 0),
        "BR": (QUAD_SIZE + 1, QUAD_SIZE + 1),
    }
    for idx, name in enumerate(QUADRANT_NAMES):
        r0, c0 = anchors[name]
        panel = base.copy()
        panel[rectangles[idx]["mask"]] = occ_colors[idx]
        grid[r0 : r0 + QUAD_SIZE, c0 : c0 + QUAD_SIZE] = panel
    return grid


def _panel_semantics(grid: np.ndarray) -> tuple[list[np.ndarray], list[dict], list[int]]:
    panels = _split_quadrants(grid)
    occluders = [_find_occluder(panel) for panel in panels]
    colors = [occ["color"] for occ in occluders]
    assert len(set(colors)) == 4
    densities = [_visible_density(panel) for panel in panels]
    assert len(set(densities)) == 4, "density order must be unambiguous"
    return panels, occluders, densities


def _validate_generation(grid: np.ndarray, expected: np.ndarray) -> None:
    panels, occluders, densities = _panel_semantics(grid)
    recovered = main(grid)
    assert np.array_equal(recovered, expected), "solver must exactly reconstruct the base pattern"

    masks = np.stack([occ["mask"] for occ in occluders], axis=0)
    coverage = masks.sum(axis=0)
    assert np.all(coverage <= 3), "every local cell must be visible in at least one quadrant"
    assert np.any(coverage == 3), "generator must include single-donor recovery cells"

    best_idx = int(np.argmax(densities))
    assert np.any(masks[best_idx] & (coverage <= 2)), "best-density donor must sometimes be unavailable"

    chosen = np.full((QUAD_SIZE, QUAD_SIZE), -1, dtype=int)
    naive = np.full((QUAD_SIZE, QUAD_SIZE), -1, dtype=int)
    for r in range(QUAD_SIZE):
        for c in range(QUAD_SIZE):
            available = [idx for idx in range(4) if not masks[idx, r, c]]
            assert available, "all-covered cells are forbidden"
            visible_values = {int(panels[idx][r, c]) for idx in available}
            assert visible_values == {int(expected[r, c])}, "all visible donors must agree on the restored value"
            chosen[r, c] = max(available, key=lambda idx: densities[idx])
            for idx in range(4):
                if idx in available:
                    naive[r, c] = idx
                    break

    assert len(set(int(v) for v in chosen.flat)) >= 2, "donor provenance should vary across cells"
    assert np.any((coverage <= 2) & (chosen != naive)), "density ranking must affect donor choice"

    output_colors = sorted(int(v) for v in np.unique(expected) if v != 0)
    assert len(output_colors) >= 2, "restored pattern must be visually non-trivial"


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        base = _make_base_pattern()
        rectangles = _choose_rectangles(base)
        grid = _assemble_input(base, rectangles)
        _validate_generation(grid, base)
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
