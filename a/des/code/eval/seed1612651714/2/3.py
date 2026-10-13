from __future__ import annotations

import random
import numpy as np


BG = 0
FRAME = 9
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].min()),
        int(coords[:, 1].max()),
    )


def _boundary_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_collection_frame(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(grid == color):
            r0, r1, c0, c1 = _bbox(comp)
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            box = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = _boundary_mask(h, w)
            if len(comp) != int(border.sum()):
                continue
            if not np.all(box[border] == color):
                continue
            if not np.all(box[~border] == BG):
                continue
            score = (r1 + c1, r0 + c0, h * w)
            if best is None or score > best[0]:
                best = (score, comp, (r0, r1, c0, c1))
    assert best is not None, "expected a framed collection box"
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each colored rectangle by removing 180-degree symmetry-breaking noise,
    then copy the repaired noise pixels into the framed box at matching relative
    coordinates.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frame_coords, (fr0, fr1, fc0, fc1) = _find_collection_frame(g)
    frame_mask = np.zeros_like(g, dtype=bool)
    frame_mask[tuple(frame_coords.T)] = True
    out[fr0 + 1 : fr1, fc0 + 1 : fc1] = BG

    collected: list[tuple[int, int, int]] = []
    for comp in _components((g != BG) & (~frame_mask)):
        r0, r1, c0, c1 = _bbox(comp)
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(block != BG):
            continue

        vals, counts = np.unique(block, return_counts=True)
        main_color = int(vals[int(np.argmax(counts))])
        h, w = block.shape

        for r in range(h):
            for c in range(w):
                rr, cc = h - 1 - r, w - 1 - c
                here = int(block[r, c])
                there = int(block[rr, cc])
                if here != there and here != main_color and there == main_color:
                    out[r0 + r, c0 + c] = main_color
                    collected.append((r, c, here))

    inner_h = fr1 - fr0 - 1
    inner_w = fc1 - fc0 - 1
    for r, c, color in collected:
        if 0 <= r < inner_h and 0 <= c < inner_w:
            tr, tc = fr0 + 1 + r, fc0 + 1 + c
            if out[tr, tc] in (BG, color):
                out[tr, tc] = color
    return out


def _orbits_180(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            rr, cc = h - 1 - r, w - 1 - c
            orbit = {(r, c), (rr, cc)}
            for cell in orbit:
                seen.add(cell)
            orbits.append(tuple(sorted(orbit)))
    return orbits


def _make_pattern_mask(
    h: int,
    w: int,
    orbits: list[tuple[tuple[int, int], ...]],
    noise_count: int,
    rng: random.Random,
) -> np.ndarray | None:
    area = h * w
    cap = min(max(2, area // 3), max(2, (area - noise_count - 1) // 2))
    pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    single_orbits = [orbit for orbit in orbits if len(orbit) == 1]
    if not pair_orbits or cap < 2:
        return None

    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        cells = 0

        first = rng.choice(pair_orbits)
        for r, c in first:
            mask[r, c] = True
        cells += len(first)

        extras = pair_orbits[:]
        rng.shuffle(extras)
        for orbit in extras:
            if orbit == first or cells + len(orbit) > cap:
                continue
            if rng.random() < 0.35:
                for r, c in orbit:
                    mask[r, c] = True
                cells += len(orbit)

        if single_orbits and cells + 1 <= cap and rng.random() < 0.5:
            orbit = rng.choice(single_orbits)
            r, c = orbit[0]
            mask[r, c] = True
            cells += 1

        if 2 <= cells <= cap:
            return mask
    return None


def _make_region(h: int, w: int, used_positions: set[tuple[int, int]], rng: random.Random) -> dict | None:
    palette = [color for color in range(1, 10) if color != FRAME]
    orbits = _orbits_180(h, w)
    pair_orbits = [orbit for orbit in orbits if len(orbit) == 2]
    if len(pair_orbits) < 1:
        return None

    for _ in range(120):
        noise_count = rng.randint(1, min(3, len(pair_orbits)))
        pattern_mask = _make_pattern_mask(h, w, orbits, noise_count, rng)
        if pattern_mask is None or not bool(pattern_mask.any()):
            continue

        candidate_orbits: list[list[tuple[int, int]]] = []
        for orbit in pair_orbits:
            if any(pattern_mask[r, c] for r, c in orbit):
                continue
            choices = [cell for cell in orbit if cell not in used_positions]
            if choices:
                candidate_orbits.append(choices)
        if len(candidate_orbits) < noise_count:
            continue

        main_color, pattern_color, noise_color = rng.sample(palette, 3)
        clean = np.full((h, w), main_color, dtype=int)
        clean[pattern_mask] = pattern_color

        noise_positions = [rng.choice(choices) for choices in rng.sample(candidate_orbits, noise_count)]
        inp = clean.copy()
        for r, c in noise_positions:
            rr, cc = h - 1 - r, w - 1 - c
            if int(clean[r, c]) != main_color or int(clean[rr, cc]) != main_color:
                break
            inp[r, c] = noise_color
        else:
            pattern_cells = int(pattern_mask.sum())
            main_cells = h * w - pattern_cells - len(noise_positions)
            if main_cells > pattern_cells and main_cells > len(noise_positions):
                return {
                    "h": h,
                    "w": w,
                    "input": inp,
                    "clean": clean,
                    "noise_color": noise_color,
                    "noise_positions": noise_positions,
                }
    return None


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    H, W = occ.shape
    occ[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)] = True


def _free_positions(occ: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    H, W = occ.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + h + 1), max(0, c0 - 1) : min(W, c0 + w + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    grid[r0 : r0 + h, c0] = FRAME
    grid[r0 : r0 + h, c0 + w - 1] = FRAME
    grid[r0, c0 : c0 + w] = FRAME
    grid[r0 + h - 1, c0 : c0 + w] = FRAME


def _render_case(
    H: int,
    W: int,
    regions: list[dict],
    placements: list[tuple[int, int]],
    frame_rect: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)

    fr0, fc0, fh, fw = frame_rect
    _draw_frame(inp, fr0, fc0, fh, fw)
    _draw_frame(out, fr0, fc0, fh, fw)

    for region, (r0, c0) in zip(regions, placements):
        h, w = region["h"], region["w"]
        inp[r0 : r0 + h, c0 : c0 + w] = region["input"]
        out[r0 : r0 + h, c0 : c0 + w] = region["clean"]
        for rr, cc in region["noise_positions"]:
            out[fr0 + 1 + rr, fc0 + 1 + cc] = region["noise_color"]
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 4)
    dims = [(rng.randint(4, 8), rng.randint(4, 8)) for _ in range(n_regions)]

    used_positions: set[tuple[int, int]] = set()
    regions: list[dict] = []
    for h, w in dims:
        region = _make_region(h, w, used_positions, rng)
        if region is None:
            return None
        regions.append(region)
        used_positions.update(region["noise_positions"])

    inner_h = max(region["h"] for region in regions) + rng.randint(0, 2)
    inner_w = max(region["w"] for region in regions) + rng.randint(0, 2)
    frame_h, frame_w = inner_h + 2, inner_w + 2

    total_area = sum(region["h"] * region["w"] for region in regions) + frame_h * frame_w
    base = int(np.ceil(np.sqrt(total_area * 2.4)))
    min_h = max(base, frame_h + 1, max(region["h"] for region in regions) + 2)
    min_w = max(base, frame_w + 1, max(region["w"] for region in regions) + 2)
    if min_h > 30 or min_w > 30:
        return None

    H = rng.randint(min_h, 30)
    W = rng.randint(min_w, 30)
    if H < frame_h or W < frame_w:
        return None

    occ = np.zeros((H, W), dtype=bool)
    bottom_margin = rng.randint(0, min(2, H - frame_h))
    right_margin = rng.randint(0, min(2, W - frame_w))
    fr0 = H - frame_h - bottom_margin
    fc0 = W - frame_w - right_margin
    _mark_occ(occ, fr0, fc0, frame_h, frame_w)

    placements: list[tuple[int, int] | None] = [None] * n_regions
    order = sorted(range(n_regions), key=lambda idx: regions[idx]["h"] * regions[idx]["w"], reverse=True)
    for idx in order:
        region = regions[idx]
        positions = _free_positions(occ, region["h"], region["w"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        placements[idx] = (r0, c0)
        _mark_occ(occ, r0, c0, region["h"], region["w"])

    inp, out = _render_case(
        H,
        W,
        regions,
        [pos for pos in placements if pos is not None],
        (fr0, fc0, frame_h, frame_w),
    )
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
