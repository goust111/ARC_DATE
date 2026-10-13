import hashlib
import itertools
from collections import Counter

import numpy as np


_SEEN_OUTPUT_SIGS: set[str] = set()


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _random_connected_mask(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    *,
    forbid: np.ndarray | None = None,
) -> np.ndarray:
    area = rows * cols
    base_min = max(3, (area + 2) // 3)
    base_max = max(base_min, area - 1)
    for _ in range(300):
        target = int(rng.integers(base_min, base_max + 1))
        start = (int(rng.integers(rows)), int(rng.integers(cols)))
        mask = np.zeros((rows, cols), dtype=bool)
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target and frontier:
            rr, cc = frontier.pop() if rng.random() < 0.3 else list(frontier)[int(rng.integers(len(frontier)))]
            candidates = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < rows and 0 <= nc < cols and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                continue
            nr, nc = candidates[int(rng.integers(len(candidates)))]
            mask[nr, nc] = True
            frontier.add((rr, cc))
            frontier.add((nr, nc))
        if int(mask.sum()) != target:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if forbid is not None and np.array_equal(mask, forbid):
            continue
        if len(_components(mask)) != 1:
            continue
        return mask
    raise RuntimeError(f"failed to build a valid {rows}x{cols} candidate mask")


def _occupancy_code(marker_mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    h, w = marker_mask.shape
    assert h % rows == 0 and w % cols == 0
    bh = h // rows
    bw = w // cols
    out = np.zeros((rows, cols), dtype=bool)
    for rr in range(rows):
        for cc in range(cols):
            block = marker_mask[rr * bh : (rr + 1) * bh, cc * bw : (cc + 1) * bw]
            out[rr, cc] = bool(block.any())
    return out


def _parse_candidate(region: np.ndarray, coords: np.ndarray, bg: int) -> dict | None:
    bbox = bbox_coords(coords)
    patch = crop_bbox(region, bbox)
    occ = patch != bg
    if int(occ.sum()) != int(coords.shape[0]):
        return None
    colors = [int(v) for v in np.unique(patch[occ])]
    if len(colors) != 2:
        return None
    counts = {color: int(np.count_nonzero(patch == color)) for color in colors}
    accents = [color for color in colors if counts[color] == 1]
    if len(accents) != 1:
        return None
    accent_color = accents[0]
    accent_pos = tuple(map(int, np.argwhere(patch == accent_color)[0]))
    if not occ[accent_pos]:
        return None
    if len(_components(occ)) != 1:
        return None
    if not occ[0].any() or not occ[-1].any() or not occ[:, 0].any() or not occ[:, -1].any():
        return None
    return {
        "bbox": bbox,
        "patch": patch,
        "occupied": occ,
        "shape": tuple(map(int, occ.shape)),
        "accent_pos": accent_pos,
        "body_color": int([c for c in colors if c != accent_color][0]),
        "accent_color": int(accent_color),
    }


def _parse_sidebar(region: np.ndarray, bg: int) -> list[dict] | None:
    if region.size == 0 or not np.any(region == bg):
        return None
    comps = _components(region != bg)
    if len(comps) < 2:
        return None
    candidates = []
    for comp in comps:
        item = _parse_candidate(region, comp, bg)
        if item is None:
            return None
        candidates.append(item)
    candidates.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return candidates


def _parse_main_region(region: np.ndarray, bg: int) -> dict | None:
    if region.size == 0 or np.any(region == bg):
        return None
    colors = [int(v) for v in np.unique(region)]
    if len(colors) != 2:
        return None
    fill = int(Counter(map(int, region.ravel())).most_common(1)[0][0])
    marker = int([c for c in colors if c != fill][0])
    marker_count = int(np.count_nonzero(region == marker))
    if marker_count == 0 or marker_count >= region.size // 2:
        return None
    return {
        "grid": region,
        "shape": tuple(map(int, region.shape)),
        "fill": fill,
        "marker": marker,
        "marker_mask": region == marker,
    }


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    splits: list[dict] = []

    def consider(main_region: np.ndarray, sidebar: np.ndarray, divider: tuple[str, int, int]) -> None:
        sidebar_bg = get_bg(sidebar)
        main_info = _parse_main_region(main_region, sidebar_bg)
        if main_info is None:
            return
        candidates = _parse_sidebar(sidebar, sidebar_bg)
        if candidates is None:
            return
        mh, mw = main_info["shape"]
        divisible = []
        matches = []
        for cand in candidates:
            rows, cols = cand["shape"]
            if mh % rows != 0 or mw % cols != 0:
                continue
            code = _occupancy_code(main_info["marker_mask"], rows, cols)
            record = {"candidate": cand, "code": code}
            divisible.append(record)
            if np.array_equal(code, cand["occupied"]):
                matches.append(record)
        if len(divisible) < 2 or len(matches) != 1:
            return
        chosen = matches[0]["candidate"]
        rows, cols = chosen["shape"]
        bh = mh // rows
        bw = mw // cols
        ar, ac = chosen["accent_pos"]
        block = main_info["grid"][ar * bh : (ar + 1) * bh, ac * bw : (ac + 1) * bw].copy()
        splits.append(
            {
                "divider": divider,
                "main": main_info,
                "sidebar_candidates": candidates,
                "divisible": divisible,
                "selected": chosen,
                "output": block,
            }
        )

    for r in range(1, h - 1):
        if np.all(g[r, :] == g[r, 0]):
            color = int(g[r, 0])
            consider(g[:r, :], g[r + 1 :, :], ("row", r, color))
            consider(g[r + 1 :, :], g[:r, :], ("row", r, color))
    for c in range(1, w - 1):
        if np.all(g[:, c] == g[0, c]):
            color = int(g[0, c])
            consider(g[:, :c], g[:, c + 1 :], ("col", c, color))
            consider(g[:, c + 1 :], g[:, :c], ("col", c, color))

    assert len(splits) == 1, f"expected exactly one valid divider split, found {len(splits)}"
    return splits[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the hidden coarse scale from the sidebar candidate that matches the main area's occupancy code, then crop the pointed raw block.

    Input structure:
    - One full single-color divider line splits the grid into a main area and a sidebar.
    - The sidebar contains several disconnected candidate sprites on background; each sprite has one body color and exactly one accent pixel of a second color.
    - The main area is fully filled by one color except for sparse marker pixels of another color.

    Rule:
    - For every candidate whose bbox height and width both divide the main area's height and width, partition the main area into that many coarse blocks.
    - Convert that partition into a boolean occupancy code: a coarse cell is occupied iff its corresponding block contains at least one marker pixel.
    - Exactly one divisible candidate has an occupied-cell pattern identical to the occupancy code at its own scale; use that candidate's accent-cell position to select the corresponding coarse block in the main area.
    - Output the selected coarse block at full resolution, preserving its original fill and marker pixels.

    Key insight:
    - The candidate sprite does two jobs at once: its bbox defines the only self-consistent partition scale, and its single accent pixel maps one abstract occupied code cell back to a concrete raw block.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _random_marker_positions(
    rng: np.random.Generator,
    block_h: int,
    block_w: int,
    count: int,
) -> list[tuple[int, int]]:
    total = block_h * block_w
    choices = rng.choice(total, size=count, replace=False)
    return [(int(v // block_w), int(v % block_w)) for v in choices]


def _compose_slots(rng: np.random.Generator, total: int, lengths: list[int]) -> list[int] | None:
    gaps = len(lengths) + 1
    free = total - (sum(lengths) + (len(lengths) - 1))
    if free < 2:
        return None
    extras = free - 2
    buckets = [1] + [0] * (gaps - 2) + [1]
    for _ in range(extras):
        buckets[int(rng.integers(gaps))] += 1
    starts = []
    pos = buckets[0]
    for idx, length in enumerate(lengths):
        starts.append(pos)
        pos += length
        if idx < len(lengths) - 1:
            pos += 1 + buckets[idx + 1]
    return starts


def _build_candidate_patch(mask: np.ndarray, body: int, accent: int, accent_pos: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(body)
    patch[accent_pos] = int(accent)
    return patch


def _generate_main_region(
    rng: np.random.Generator,
    shape: tuple[int, int],
    candidate_mask: np.ndarray,
    accent_pos: tuple[int, int],
    fill: int,
    marker: int,
) -> np.ndarray:
    rows, cols = candidate_mask.shape
    h, w = shape
    bh = h // rows
    bw = w // cols
    out = np.full((h, w), int(fill), dtype=int)
    occupied_coords = [tuple(map(int, pt)) for pt in np.argwhere(candidate_mask)]
    for rr, cc in occupied_coords:
        if (rr, cc) == accent_pos:
            marker_count = min(3, bh * bw)
        else:
            marker_count = int(rng.integers(1, min(2, bh * bw - 1) + 1))
        for dr, dc in _random_marker_positions(rng, bh, bw, marker_count):
            out[rr * bh + dr, cc * bw + dc] = int(marker)
    return out


def _block_signature(block: np.ndarray) -> str:
    arr = np.asarray(block, dtype=int)
    return hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()


def _validate_semantics(grid: np.ndarray) -> bool:
    details = _solve_details(grid)
    main_grid = details["main"]["grid"]
    selected = details["selected"]
    rows, cols = selected["shape"]
    bh = main_grid.shape[0] // rows
    bw = main_grid.shape[1] // cols
    assert len(details["divisible"]) >= 2
    assert sum(np.array_equal(item["code"], item["candidate"]["occupied"]) for item in details["divisible"]) == 1
    assert int(selected["occupied"].sum()) >= 2
    sigs = []
    for rr in range(rows):
        for cc in range(cols):
            block = main_grid[rr * bh : (rr + 1) * bh, cc * bw : (cc + 1) * bw]
            sigs.append(_block_signature(block))
    target_sig = _block_signature(details["output"])
    assert sigs.count(target_sig) == 1
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    orientation = str(rng.choice(["vertical", "horizontal"]))
    if orientation == "vertical":
        main_h = int(rng.choice([12, 15]))
        main_w = int(rng.choice([8, 10, 12]))
    else:
        main_h = int(rng.choice([8, 10, 12]))
        main_w = int(rng.choice([12, 15]))

    row_divs = [d for d in (2, 3, 4, 5) if main_h % d == 0 and main_h // d >= 2]
    col_divs = [d for d in (2, 3, 4, 5) if main_w % d == 0 and main_w // d >= 2]
    pool = [(r, c) for r in row_divs for c in col_divs]
    rng.shuffle(pool)
    combos = list(itertools.combinations(pool, 3))
    rng.shuffle(combos)
    dims = None
    for combo in combos:
        if len(set(combo)) != 3:
            continue
        lengths = [rc[0] for rc in combo] if orientation == "vertical" else [rc[1] for rc in combo]
        if sum(lengths) + 4 <= (main_h if orientation == "vertical" else main_w):
            dims = list(combo)
            break
    if dims is None:
        return None

    target_idx = int(rng.integers(len(dims)))
    target_rows, target_cols = dims[target_idx]
    target_mask = _random_connected_mask(rng, target_rows, target_cols)
    occ_coords = [tuple(map(int, pt)) for pt in np.argwhere(target_mask)]
    accent_pos = occ_coords[int(rng.integers(len(occ_coords)))]

    colors = list(range(1, 10))
    rng.shuffle(colors)
    divider_color, fill_color, marker_color = colors[:3]
    body_colors = colors[3:6]
    accent_colors = colors[6:9]

    main_region = _generate_main_region(
        rng,
        (main_h, main_w),
        target_mask,
        accent_pos,
        fill_color,
        marker_color,
    )

    candidate_masks: list[np.ndarray] = []
    candidate_accents: list[tuple[int, int]] = []
    marker_mask = main_region == marker_color
    for idx, (rows, cols) in enumerate(dims):
        if idx == target_idx:
            candidate_masks.append(target_mask)
            candidate_accents.append(accent_pos)
            continue
        code = _occupancy_code(marker_mask, rows, cols)
        decoy = _random_connected_mask(rng, rows, cols, forbid=code)
        candidate_masks.append(decoy)
        pts = [tuple(map(int, pt)) for pt in np.argwhere(decoy)]
        candidate_accents.append(pts[int(rng.integers(len(pts)))])

    patches = [
        _build_candidate_patch(candidate_masks[i], body_colors[i], accent_colors[i], candidate_accents[i], 0)
        for i in range(3)
    ]

    if orientation == "vertical":
        sidebar_w = max(int(p.shape[1]) for p in patches) + 2
        if sidebar_w + 1 + main_w > 30:
            return None
        sidebar = np.zeros((main_h, sidebar_w), dtype=int)
        starts = _compose_slots(rng, main_h, [int(p.shape[0]) for p in patches])
        if starts is None:
            return None
        for patch, row0 in zip(patches, starts, strict=False):
            col0 = int(rng.integers(1, sidebar_w - patch.shape[1]))
            sidebar[row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]] = patch
        divider = np.full((main_h, 1), divider_color, dtype=int)
        if rng.random() < 0.5:
            grid = np.hstack([sidebar, divider, main_region])
        else:
            grid = np.hstack([main_region, divider, sidebar])
    else:
        sidebar_h = max(int(p.shape[0]) for p in patches) + 2
        if sidebar_h + 1 + main_h > 30:
            return None
        sidebar = np.zeros((sidebar_h, main_w), dtype=int)
        starts = _compose_slots(rng, main_w, [int(p.shape[1]) for p in patches])
        if starts is None:
            return None
        for patch, col0 in zip(patches, starts, strict=False):
            row0 = int(rng.integers(1, sidebar_h - patch.shape[0]))
            sidebar[row0 : row0 + patch.shape[0], col0 : col0 + patch.shape[1]] = patch
        divider = np.full((1, main_w), divider_color, dtype=int)
        if rng.random() < 0.5:
            grid = np.vstack([sidebar, divider, main_region])
        else:
            grid = np.vstack([main_region, divider, sidebar])

    details = _solve_details(grid)
    if not np.array_equal(details["output"], main(grid)):
        return None
    if not _validate_semantics(grid):
        return None

    out_sig = _block_signature(details["output"])
    if out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-scaled ARC input")
