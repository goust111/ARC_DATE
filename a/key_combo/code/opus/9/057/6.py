from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int, copy=True)


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    return _crop_bbox(grid, _bbox_coords(coords))


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _components_from_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _extract_components(grid: np.ndarray, mask: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components_from_mask(mask, connectivity=4):
        box = _bbox_coords(coords)
        r0, c0, r1, c1 = box
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for rr, cc in coords:
            patch[rr - r0, cc - c0] = int(grid[rr, cc])
        vals = sorted({int(v) for v in patch.reshape(-1) if int(v) != bg})
        comps.append(
            {
                "coords": coords,
                "bbox": box,
                "patch": patch,
                "colors": vals,
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _find_hub(grid: np.ndarray, bg: int) -> dict:
    H, W = grid.shape
    best = None
    for n in range(5, 8):
        for r0 in range(H - n + 1):
            for c0 in range(W - n + 1):
                if (
                    int(grid[r0, c0]) != bg
                    or int(grid[r0, c0 + n - 1]) != bg
                    or int(grid[r0 + n - 1, c0]) != bg
                    or int(grid[r0 + n - 1, c0 + n - 1]) != bg
                ):
                    continue
                inner = grid[r0 + 1 : r0 + n - 1, c0 + 1 : c0 + n - 1]
                if not np.all(inner == bg):
                    continue
                top = grid[r0, c0 + 1 : c0 + n - 1]
                bottom = grid[r0 + n - 1, c0 + 1 : c0 + n - 1]
                left = grid[r0 + 1 : r0 + n - 1, c0]
                right = grid[r0 + 1 : r0 + n - 1, c0 + n - 1]
                sides = [top, bottom, left, right]
                if any(np.any(side == bg) for side in sides):
                    continue
                side_colors = []
                ok = True
                for side in sides:
                    cols = {int(v) for v in side}
                    if len(cols) != 1:
                        ok = False
                        break
                    side_colors.append(int(next(iter(cols))))
                if not ok or len(set(side_colors)) != 4:
                    continue
                center_score = abs((r0 + (n - 1) / 2) - (H - 1) / 2) + abs((c0 + (n - 1) / 2) - (W - 1) / 2)
                cand = {
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(n),
                    "side_colors": {
                        "top": side_colors[0],
                        "bottom": side_colors[1],
                        "left": side_colors[2],
                        "right": side_colors[3],
                    },
                    "score": float(center_score),
                }
                if best is None or cand["score"] < best["score"]:
                    best = cand
    if best is None:
        raise AssertionError("hub not found")
    return best


def _contact_side(place_side: str) -> str:
    return {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}[place_side]


def _contiguous_run(indices: list[int]) -> bool:
    return bool(indices) and all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))


def _rotation_match(mask: np.ndarray, place_side: str, target: int) -> dict | None:
    local_side = _contact_side(place_side)
    if local_side in ("top", "bottom"):
        if not np.array_equal(mask, np.fliplr(mask)):
            return None
        border = mask[0] if local_side == "top" else mask[-1]
        cols = [int(c) for c in np.flatnonzero(border)]
        if len(cols) != target or not _contiguous_run(cols):
            return None
        return {"axis_start": cols[0]}
    if not np.array_equal(mask, np.flipud(mask)):
        return None
    border = mask[:, 0] if local_side == "left" else mask[:, -1]
    rows = [int(r) for r in np.flatnonzero(border)]
    if len(rows) != target or not _contiguous_run(rows):
        return None
    return {"axis_start": rows[0]}


def _valid_rotations(patch: np.ndarray, place_side: str, target: int, bg: int) -> list[dict]:
    out = []
    for k in range(4):
        rot = _tight_crop(np.rot90(patch, k), bg)
        mask = rot != bg
        info = _rotation_match(mask, place_side, target)
        if info is not None:
            out.append({"k": int(k), "patch": rot, "mask": mask, "info": info})
    return out


def _placement_for_side(patch: np.ndarray, place_side: str, hub: dict, axis_start: int) -> tuple[int, int]:
    r0 = int(hub["r0"])
    c0 = int(hub["c0"])
    n = int(hub["size"])
    h, w = patch.shape
    if place_side == "top":
        return r0 - h, c0 + 1 - axis_start
    if place_side == "bottom":
        return r0 + n, c0 + 1 - axis_start
    if place_side == "left":
        return r0 + 1 - axis_start, c0 - w
    return r0 + 1 - axis_start, c0 + n


def _hub_mask(shape: tuple[int, int], hub: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0 = int(hub["r0"])
    c0 = int(hub["c0"])
    n = int(hub["size"])
    mask[r0 : r0 + n, c0 : c0 + n] = True
    return mask


def _marker_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1) if int(v) != bg)
    rare = min(counts.values())
    rare_colors = [col for col, cnt in counts.items() if cnt == rare]
    if len(rare_colors) != 1:
        raise AssertionError("marker color must be uniquely rare")
    return int(rare_colors[0])


def _side_candidates(grid: np.ndarray, hub: dict, marker: int, bg: int) -> dict[str, list[dict]]:
    mask = (grid != bg) & (~_hub_mask(grid.shape, hub))
    comps = _extract_components(grid, mask, bg)
    grouped: dict[str, list[dict]] = {side: [] for side in ("top", "bottom", "left", "right")}
    for side, col in hub["side_colors"].items():
        matches = []
        for comp in comps:
            colors = set(comp["colors"])
            if colors == {int(col), int(marker)}:
                patch = comp["patch"]
                grey_count = int(np.count_nonzero(patch == marker))
                matches.append({**comp, "grey_count": grey_count, "side": side})
        grouped[side] = matches
    return grouped


def _build_output_from_selection(grid: np.ndarray, chosen: dict[str, dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    hub = _find_hub(g, bg)
    out = np.full_like(g, bg)
    r0 = int(hub["r0"])
    c0 = int(hub["c0"])
    n = int(hub["size"])
    out[r0 : r0 + n, c0 : c0 + n] = g[r0 : r0 + n, c0 : c0 + n]
    target = n - 2
    for side, comp in chosen.items():
        valids = _valid_rotations(comp["patch"], side, target, bg)
        if len(valids) != 1:
            raise AssertionError(f"expected unique rotation for {side}, got {len(valids)}")
        rot = valids[0]
        pr, pc = _placement_for_side(rot["patch"], side, hub, int(rot["info"]["axis_start"]))
        h, w = rot["patch"].shape
        if not (0 <= pr and 0 <= pc and pr + h <= out.shape[0] and pc + w <= out.shape[1]):
            raise AssertionError("attached sprite leaves grid")
        dest = out[pr : pr + h, pc : pc + w]
        overlap = (rot["patch"] != bg) & (dest != bg)
        if np.any(overlap):
            raise AssertionError("attached sprite overlaps existing object")
        _paste(out, rot["patch"], pr, pc, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the denser marked candidate for each hub-edge color, then rotate and dock it onto the central hub.

    Input structure:
    - A mostly black background contains one near-central hollow square hub whose four non-corner sides are solid and use four distinct colors.
    - Outside the hub are separated 4-connected two-color sprites; each sprite uses grey plus exactly one hub-edge color.
    - For each hub-edge color there are exactly two candidate sprites sharing that color, and grey is the unique globally rare non-background color.

    Rule:
    - Infer grey as the unique least frequent non-background color.
    - For each hub side color, collect the two sprites containing that color and choose the one with more grey pixels.
    - Rotate the chosen sprite so its contacting border has exactly N-2 contiguous cells aligned with the matching hub side, and so the sprite is mirror-symmetric along the attachment-normal axis.
    - Translate each chosen sprite to sit flush outside its matching hub side, and clear all unchosen sprites.

    Key insight:
    - Color matching only narrows each side to two candidates; the decisive step is counting the rare marker color inside each candidate before using the symmetry-constrained docking rotation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    hub = _find_hub(g, bg)
    marker = _marker_color(g, bg)
    grouped = _side_candidates(g, hub, marker, bg)
    chosen: dict[str, dict] = {}
    for side, comps in grouped.items():
        if len(comps) != 2:
            raise AssertionError(f"expected two candidates for {side}")
        grey_counts = sorted(comp["grey_count"] for comp in comps)
        if grey_counts[0] == grey_counts[1]:
            raise AssertionError(f"grey counts must differ for {side}")
        chosen[side] = max(comps, key=lambda comp: comp["grey_count"])
    return _build_output_from_selection(g, chosen)


def _row_mask(width: int, run_len: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    start = (width - run_len) // 2
    row[start : start + run_len] = 1
    return row


def _mask_from_lengths(lengths: list[int]) -> np.ndarray:
    width = max(lengths)
    rows = [_row_mask(width, run_len) for run_len in lengths]
    return np.array(rows, dtype=int)


def _base_templates(target: int) -> list[np.ndarray]:
    specs = {
        3: [[1, 3, 3], [1, 1, 3, 3]],
        4: [[2, 4], [2, 2, 4]],
        5: [[1, 5], [1, 1, 5], [3, 5]],
    }
    return [_mask_from_lengths(lengths) for lengths in specs[target]]


def _attached_templates(target: int, place_side: str) -> list[np.ndarray]:
    base = _base_templates(target)
    if place_side == "top":
        masks = [mask.copy() for mask in base]
    elif place_side == "bottom":
        masks = [np.flipud(mask) for mask in base]
    elif place_side == "left":
        masks = [mask.T for mask in base]
    else:
        masks = [np.flipud(mask).T for mask in base]
    out = []
    for mask in masks:
        patch = mask.astype(int)
        if len(_valid_rotations(patch, place_side, target, bg=0)) == 1:
            out.append(patch)
    if len(out) < 2:
        raise AssertionError("not enough valid templates")
    return out


def _colorize(mask: np.ndarray, main_color: int, marker: int, grey_count: int, rng: random.Random) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    grey_cells = set(rng.sample(coords, grey_count))
    patch = np.zeros(mask.shape, dtype=int)
    for rr, cc in coords:
        patch[rr, cc] = marker if (rr, cc) in grey_cells else main_color
    return patch


def _choose_pair_templates(target: int, place_side: str, rng: random.Random) -> list[np.ndarray]:
    templates = _attached_templates(target, place_side)
    if len(templates) == 2:
        return [templates[0].copy(), templates[1].copy()]
    picks = rng.sample(templates, 2)
    return [picks[0].copy(), picks[1].copy()]


def _mark_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> None:
    H, W = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    mask[rr0:rr1, cc0:cc1] = True


def _place_scattered(grid: np.ndarray, sprites: list[np.ndarray], hub: dict, rng: random.Random) -> bool:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)
    _mark_rect(blocked, int(hub["r0"]), int(hub["c0"]), int(hub["size"]), int(hub["size"]), pad=1)
    order = list(range(len(sprites)))
    rng.shuffle(order)
    for idx in order:
        patch = sprites[idx]
        h, w = patch.shape
        placed = False
        for _ in range(400):
            r0 = rng.randrange(0, H - h + 1)
            c0 = rng.randrange(0, W - w + 1)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(H, r0 + h + 1)
            cc1 = min(W, c0 + w + 1)
            if np.any(blocked[rr0:rr1, cc0:cc1]):
                continue
            _paste(grid, patch, r0, c0, transparent=0)
            _mark_rect(blocked, r0, c0, h, w, pad=1)
            placed = True
            break
        if not placed:
            return False
    return True


def _build_hub(H: int, W: int, n: int, side_colors: dict[str, int], rng: random.Random) -> dict:
    max_shift_r = max(1, (H - n) // 6)
    max_shift_c = max(1, (W - n) // 6)
    center_r = (H - n) // 2 + rng.randint(-max_shift_r, max_shift_r)
    center_c = (W - n) // 2 + rng.randint(-max_shift_c, max_shift_c)
    r0 = min(max(center_r, 4), H - n - 4)
    c0 = min(max(center_c, 4), W - n - 4)
    return {"r0": int(r0), "c0": int(c0), "size": int(n), "side_colors": dict(side_colors)}


def _paint_hub(grid: np.ndarray, hub: dict) -> None:
    r0 = int(hub["r0"])
    c0 = int(hub["c0"])
    n = int(hub["size"])
    grid[r0, c0 + 1 : c0 + n - 1] = int(hub["side_colors"]["top"])
    grid[r0 + n - 1, c0 + 1 : c0 + n - 1] = int(hub["side_colors"]["bottom"])
    grid[r0 + 1 : r0 + n - 1, c0] = int(hub["side_colors"]["left"])
    grid[r0 + 1 : r0 + n - 1, c0 + n - 1] = int(hub["side_colors"]["right"])


def _generate_candidate_pair(target: int, place_side: str, main_color: int, marker: int, rng: random.Random) -> tuple[dict, dict]:
    masks = _choose_pair_templates(target, place_side, rng)
    grey_counts = [2, 1]
    if rng.random() < 0.5:
        grey_counts.reverse()
    sprites = []
    for mask, grey_count in zip(masks, grey_counts):
        patch = _colorize(mask, main_color, marker, grey_count, rng)
        valids = _valid_rotations(patch, place_side, target, bg=0)
        if len(valids) != 1:
            raise AssertionError("template lost unique rotation after coloring")
        sprites.append({"patch": patch, "grey_count": grey_count, "place_side": place_side})
    if sprites[0]["grey_count"] == sprites[1]["grey_count"]:
        raise AssertionError("grey counts must differ")
    winner_idx = 0 if sprites[0]["grey_count"] > sprites[1]["grey_count"] else 1
    loser_idx = 1 - winner_idx
    return sprites[winner_idx], sprites[loser_idx]


def _scatter_version(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([1, 2, 3])
    return _tight_crop(np.rot90(patch, k), 0)


def _validate_instance(inp: np.ndarray, out: np.ndarray) -> None:
    solved = main(inp)
    if not np.array_equal(solved, out):
        raise AssertionError("main(input) != output")
    bg = _get_bg(inp)
    hub = _find_hub(inp, bg)
    marker = _marker_color(inp, bg)
    grouped = _side_candidates(inp, hub, marker, bg)
    if sum(len(v) for v in grouped.values()) != 8:
        raise AssertionError("expected eight candidate sprites")
    for side, comps in grouped.items():
        if len(comps) != 2:
            raise AssertionError(f"need exactly two components for {side}")
        if sorted(comp["grey_count"] for comp in comps) != [1, 2]:
            raise AssertionError(f"unexpected grey counts for {side}")
        for comp in comps:
            if len(_valid_rotations(comp["patch"], side, int(hub["size"]) - 2, bg)) != 1:
                raise AssertionError(f"component for {side} not uniquely dockable")
    chosen = {side: max(comps, key=lambda comp: comp["grey_count"]) for side, comps in grouped.items()}
    alt_base = _build_output_from_selection(inp, chosen)
    if not np.array_equal(alt_base, out):
        raise AssertionError("rebuilt output mismatch")
    for side, comps in grouped.items():
        swapped = dict(chosen)
        swapped[side] = min(comps, key=lambda comp: comp["grey_count"])
        alt = _build_output_from_selection(inp, swapped)
        if np.array_equal(alt, out):
            raise AssertionError(f"swapping {side} does not change output")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(21, 27)
    W = rng.randint(21, 27)
    n = rng.randint(5, 7)
    target = n - 2
    side_names = ["top", "bottom", "left", "right"]
    side_colors_list = [1, 2, 3, 4]
    rng.shuffle(side_colors_list)
    side_colors = dict(zip(side_names, side_colors_list))
    hub = _build_hub(H, W, n, side_colors, rng)

    winners: dict[str, dict] = {}
    losers: dict[str, dict] = {}
    for side in side_names:
        winner, loser = _generate_candidate_pair(target, side, int(side_colors[side]), 5, rng)
        winners[side] = winner
        losers[side] = loser

    output = np.zeros((H, W), dtype=int)
    _paint_hub(output, hub)
    for side, winner in winners.items():
        valids = _valid_rotations(winner["patch"], side, target, bg=0)
        rot = valids[0]
        pr, pc = _placement_for_side(rot["patch"], side, hub, int(rot["info"]["axis_start"]))
        h, w = rot["patch"].shape
        if pr < 0 or pc < 0 or pr + h > H or pc + w > W:
            return None
        if np.any((output[pr : pr + h, pc : pc + w] != 0) & (rot["patch"] != 0)):
            return None
        _paste(output, rot["patch"], pr, pc, transparent=0)

    input_grid = np.zeros((H, W), dtype=int)
    _paint_hub(input_grid, hub)
    scattered = []
    for side in side_names:
        scattered.append(_scatter_version(winners[side]["patch"], rng))
        scattered.append(_scatter_version(losers[side]["patch"], rng))
    if not _place_scattered(input_grid, scattered, hub, rng):
        return None

    counts = Counter(int(v) for v in input_grid.reshape(-1) if int(v) != 0)
    if 5 not in counts:
        return None
    grey_total = int(counts[5])
    if any(grey_total >= int(counts[color]) for color in side_colors_list):
        return None
    try:
        _validate_instance(input_grid, output)
    except AssertionError:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
