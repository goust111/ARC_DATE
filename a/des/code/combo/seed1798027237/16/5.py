import numpy as np


BG = 0
GREY = 5
RNG = np.random.default_rng()
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    if np.any(vals == BG):
        return BG
    return int(vals[np.argmax(counts)])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return comps


def _is_frame_component(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False

    mask = np.zeros((h, w), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _monochrome_components(grid, bg):
        if not _is_frame_component(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0 or inner.shape[0] != inner.shape[1]:
            continue
        inner_colors = set(map(int, np.unique(inner))) - {bg}
        if len(inner_colors) != 1:
            continue
        frames.append(comp)

    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["color"]))
    return frames


def _palette_candidates(grid: np.ndarray, bg: int, frame_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    out: list[dict] = []

    for r in range(h):
        c = 0
        while c < w:
            if int(grid[r, c]) == bg:
                c += 1
                continue
            c0 = c
            while c < w and int(grid[r, c]) != bg:
                c += 1
            c1 = c - 1

            vals = grid[r, c0 : c1 + 1]
            seg_len = int(c1 - c0 + 1)
            grey_count = int(np.sum(vals == GREY))
            if seg_len < 2 or seg_len > 4 or grey_count not in (1, 2, 3):
                continue
            if grey_count != seg_len - 1:
                continue

            other = [int(v) for v in vals if int(v) != GREY]
            if len(other) != 1:
                continue
            color = other[0]
            if color not in frame_colors:
                continue

            cells = [(r, cc) for cc in range(c0, c1 + 1)]
            out.append(
                {
                    "color": color,
                    "turns": grey_count,
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "cells": cells,
                }
            )

    return out


def _corner_distance(cand: dict, shape: tuple[int, int], corner: str) -> int:
    h, w = shape
    row = int(cand["row"])
    c0 = int(cand["c0"])
    c1 = int(cand["c1"])
    if corner == "tl":
        return row + c0
    if corner == "tr":
        return row + (w - 1 - c1)
    if corner == "bl":
        return (h - 1 - row) + c0
    if corner == "br":
        return (h - 1 - row) + (w - 1 - c1)
    raise ValueError(corner)


def _decode_palette(grid: np.ndarray, frames: list[dict], bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    frame_colors = {int(frame["color"]) for frame in frames}
    candidates = _palette_candidates(grid, bg, frame_colors)
    assert candidates, "failed to find palette rows"

    best = None
    for corner in ("tl", "tr", "bl", "br"):
        ordered = sorted(
            candidates,
            key=lambda cand: (
                _corner_distance(cand, grid.shape, corner),
                cand["row"],
                cand["c0"],
                cand["c1"],
            ),
        )
        chosen: dict[int, dict] = {}
        for cand in ordered:
            color = int(cand["color"])
            if color not in chosen:
                chosen[color] = cand
        if set(chosen) != frame_colors:
            continue

        dists = [_corner_distance(cand, grid.shape, corner) for cand in chosen.values()]
        rows = [int(cand["row"]) for cand in chosen.values()]
        cmins = [int(cand["c0"]) for cand in chosen.values()]
        cmaxs = [int(cand["c1"]) for cand in chosen.values()]
        score = (
            sum(dists),
            max(dists),
            max(rows) - min(rows),
            max(cmaxs) - min(cmins),
        )
        if best is None or score < best[0]:
            best = (score, chosen)

    assert best is not None, "failed to decode palette"
    chosen = best[1]
    mapping = {int(color): int(cand["turns"]) for color, cand in chosen.items()}
    palette_cells = {cell for cand in chosen.values() for cell in cand["cells"]}
    return mapping, palette_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the corner palette and rotate each framed pattern clockwise by the
    number of grey dots assigned to the frame's border color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "no frames detected"

    rotation_map, palette_cells = _decode_palette(g, frames, bg)
    out = g.copy()
    for r, c in palette_cells:
        out[r, c] = bg

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        turns = int(rotation_map[int(frame["color"])]) % 4
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, -turns)

    return out


COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_COUNT_CHOICES = (2, 3, 4, 5)
INNER_SIZE_RANGE = (3, 7)
PLACEMENT_PAD = 1
PALETTE_PAD = 2
GEN_RETRIES = 400


def _rect_is_free(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    hh, ww = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(mask[rr0:rr1, cc0:cc1].any())


def _occupy_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    hh, ww = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc1 = min(ww, c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _random_connected_mask(size: int) -> np.ndarray:
    area_min = max(3, size)
    area_max = max(area_min, size * size - 2)

    for _ in range(300):
        target = int(RNG.integers(area_min, area_max + 1))
        mask = np.zeros((size, size), dtype=bool)
        seed = (int(RNG.integers(size)), int(RNG.integers(size)))
        mask[seed] = True
        frontier = {seed}

        while int(mask.sum()) < target and frontier:
            rr, cc = frontier.pop()
            nbrs = []
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    nbrs.append((nr, nc))
            if not nbrs:
                continue
            nr, nc = nbrs[int(RNG.integers(len(nbrs)))]
            mask[nr, nc] = True
            frontier.add((rr, cc))
            frontier.add((nr, nc))

        if int(mask.sum()) != target:
            continue

        rs = np.where(mask.any(axis=1))[0]
        cs = np.where(mask.any(axis=0))[0]
        if len(rs) < 2 or len(cs) < 2:
            continue

        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue

        return mask

    raise RuntimeError("failed to generate asymmetric mask")


def _build_palette_rows(h: int, w: int, specs: list[dict], corner: str) -> list[dict]:
    order = list(specs)
    RNG.shuffle(order)
    margin_r = int(RNG.integers(0, 2))
    margin_c = int(RNG.integers(0, 2))
    base_r = margin_r if corner.startswith("t") else h - margin_r - len(order)

    rows: list[dict] = []
    for idx, spec in enumerate(order):
        turns = int(spec["turns"])
        row = base_r + idx
        if corner.endswith("l"):
            c0 = margin_c
        else:
            c0 = w - margin_c - (turns + 1)
        cells = [(row, c0 + offset) for offset in range(turns + 1)]
        rows.append({"color": int(spec["border"]), "turns": turns, "cells": cells})

    return rows


def _palette_bbox(rows: list[dict]) -> tuple[int, int, int, int]:
    cells = [cell for row in rows for cell in row["cells"]]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _gen_attempt() -> np.ndarray | None:
    n_frames = int(RNG.choice(FRAME_COUNT_CHOICES))
    h = int(RNG.integers(max(18, 12 + n_frames), GRID_H_RANGE[1]))
    w = int(RNG.integers(max(18, 12 + n_frames), GRID_W_RANGE[1]))
    grid = np.full((h, w), BG, dtype=int)

    if n_frames > len(COLOR_POOL):
        return None

    border_colors = [int(c) for c in RNG.choice(COLOR_POOL, size=n_frames, replace=False)]
    remaining = [int(c) for c in COLOR_POOL if int(c) not in border_colors]
    if not remaining:
        return None

    turns = [int(RNG.integers(1, 4)) for _ in range(n_frames)]
    if n_frames >= 3 and float(RNG.random()) < 0.6:
        forced = [1, 2, 3]
        RNG.shuffle(forced)
        for idx, value in enumerate(forced):
            turns[idx] = value

    frame_specs: list[dict] = []
    for idx, border in enumerate(border_colors):
        inner_size = int(RNG.integers(INNER_SIZE_RANGE[0], INNER_SIZE_RANGE[1] + 1))
        mask = _random_connected_mask(inner_size)
        mask = np.rot90(mask, -int(RNG.integers(0, 4)))
        pattern_color = int(RNG.choice([c for c in remaining if c != border]))
        frame_specs.append(
            {
                "border": border,
                "pattern": pattern_color,
                "turns": turns[idx],
                "inner_size": inner_size,
                "outer_size": inner_size + 2,
                "mask": mask,
            }
        )

    corner = str(RNG.choice(["tl", "tr", "bl", "br"]))
    palette_rows = _build_palette_rows(h, w, frame_specs, corner)
    pr0, pc0, pr1, pc1 = _palette_bbox(palette_rows)

    occupied = np.zeros((h, w), dtype=bool)
    _occupy_rect(occupied, pr0, pc0, pr1 - pr0 + 1, pc1 - pc0 + 1, PALETTE_PAD)

    placed_specs = sorted(frame_specs, key=lambda spec: -int(spec["outer_size"]))
    for spec in placed_specs:
        size = int(spec["outer_size"])
        placed = False
        for _ in range(300):
            r0 = int(RNG.integers(0, h - size + 1))
            c0 = int(RNG.integers(0, w - size + 1))
            if not _rect_is_free(occupied, r0, c0, size, size, PLACEMENT_PAD):
                continue
            spec["bbox"] = (r0, c0, r0 + size - 1, c0 + size - 1)
            _occupy_rect(occupied, r0, c0, size, size, PLACEMENT_PAD)
            placed = True
            break
        if not placed:
            return None

    for row in palette_rows:
        color = int(row["color"])
        for idx, (r, c) in enumerate(row["cells"]):
            grid[r, c] = color if idx == 0 else GREY

    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]
        border = int(spec["border"])
        pattern = int(spec["pattern"])
        grid[r0 : r1 + 1, c0] = border
        grid[r0 : r1 + 1, c1] = border
        grid[r0, c0 : c1 + 1] = border
        grid[r1, c0 : c1 + 1] = border
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner[spec["mask"]] = pattern

    expected = grid.copy()
    for row in palette_rows:
        for r, c in row["cells"]:
            expected[r, c] = BG
    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        expected[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(inner, -int(spec["turns"]))

    if len(_find_frames(grid, BG)) != n_frames:
        return None

    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
