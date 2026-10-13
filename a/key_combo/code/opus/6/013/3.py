from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
BODY_COLORS = (1, 3, 4)
MARKER_COLORS = (7, 8, 9)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

TEMPLATES = [
    np.array([[1, 1, 0], [2, 1, 1]], dtype=int),
    np.array([[1, 2, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 2, 1]], dtype=int),
    np.array([[1, 0], [2, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 2, 0], [1, 1, 0]], dtype=int),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_from_mask(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    seen = np.zeros(mask.shape, dtype=bool)
    dirs = DIRS4 if connectivity == 4 else DIRS8
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc2 = stack.pop()
            comp.append((cr, cc2))
            for dr, dc in dirs:
                nr = cr + dr
                nc = cc2 + dc
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _multicolor_components(grid: np.ndarray, bg: int, connectivity: int) -> list[list[tuple[int, int]]]:
    return _components_from_mask(np.asarray(grid, dtype=int) != int(bg), connectivity)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_component(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int) -> None:
    h, w = patch.shape
    for r in range(h):
        for c in range(w):
            val = int(patch[r, c])
            if val == int(transparent):
                continue
            rr = r0 + r
            cc = c0 + c
            if _in_bounds(dst.shape, rr, cc):
                dst[rr, cc] = val


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(np.asarray(arr, dtype=int), -int(quarter_turns) % 4)


def _comb_info(grid: np.ndarray, coords: list[tuple[int, int]], color: int, bg: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if c1 - c0 + 1 != 3:
        return None
    local = np.zeros((r1 - r0 + 1, 3), dtype=bool)
    coord_set = set(coords)
    for r, c in coords:
        local[r - r0, c - c0] = True
    if not bool(local.any()):
        return None
    row_types: list[str] = []
    for r in range(local.shape[0]):
        row = tuple(int(x) for x in local[r])
        if row == (1, 1, 1):
            row_types.append("solid")
        elif row == (1, 0, 1):
            row_types.append("open")
        else:
            return None
    for i in range(len(row_types) - 1):
        if row_types[i] == row_types[i + 1]:
            return None
    solid_count = sum(rt == "solid" for rt in row_types)
    if not (1 <= solid_count <= 3):
        return None
    # Comb structures are standalone; reject same-colored shapes embedded in a marked sprite.
    for r, c in coords:
        for dr, dc in DIRS8:
            nr = r + dr
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and (nr, nc) not in coord_set:
                if int(grid[nr, nc]) != int(bg):
                    return None
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "solid_count": int(solid_count),
    }


def _find_combs(grid: np.ndarray, bg: int) -> list[dict]:
    combs: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        for coords in _components_from_mask(grid == int(color), connectivity=8):
            info = _comb_info(grid, coords, color, bg)
            if info is not None:
                combs.append(info)
    return combs


def _is_connected(mask: np.ndarray, connectivity: int) -> bool:
    coords = _components_from_mask(mask, connectivity)
    return len(coords) == 1


def _find_sources(grid: np.ndarray, bg: int) -> dict[int, dict]:
    sources: dict[int, dict] = {}
    for coords in _multicolor_components(grid, bg, connectivity=4):
        patch, bbox = _crop_component(grid, coords, bg)
        vals = [int(grid[r, c]) for r, c in coords]
        counts = Counter(vals)
        colors = sorted(counts)
        if len(colors) != 2:
            continue
        marker_candidates = [color for color, n in counts.items() if int(n) == 1]
        if len(marker_candidates) != 1:
            continue
        marker_color = int(marker_candidates[0])
        body_color = int(next(color for color in colors if color != marker_color))
        body_count = int(counts[body_color])
        if not (3 <= body_count <= 6):
            continue
        marker_global = next((r, c) for r, c in coords if int(grid[r, c]) == marker_color)
        body_mask = patch == int(body_color)
        if not _is_connected(body_mask, connectivity=4):
            continue
        mr, mc = np.argwhere(patch == int(marker_color))[0]
        if not any(
            _in_bounds(patch.shape, int(mr) + dr, int(mc) + dc) and int(patch[int(mr) + dr, int(mc) + dc]) == int(body_color)
            for dr, dc in DIRS4
        ):
            continue
        all_markers = [tuple(map(int, p)) for p in np.argwhere(grid == int(marker_color))]
        targets = [coord for coord in all_markers if coord != marker_global]
        if not targets or body_color in sources:
            continue
        sources[body_color] = {
            "body_color": int(body_color),
            "marker_color": int(marker_color),
            "patch": patch,
            "bbox": bbox,
            "source_marker": marker_global,
            "targets": sorted(targets),
        }
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count each comb's solid rows to decide how far its marked sprite must rotate, then stamp the rotated sprite onto matching markers.

    Input structure:
    - The background is black.
    - There are 2-3 standalone monochrome combs in body colors, each occupying a 3-column bbox whose rows alternate between solid triples and side-pairs.
    - For each comb color there is one nearby 4-connected marked sprite: a small body of the same color plus one embedded marker pixel of a different color.
    - Additional isolated target markers appear elsewhere; each target color matches exactly one embedded marker color.

    Rule:
    - For every comb, count only its solid 3-cell rows; call that count s.
    - Pair the comb with the marked sprite of the same body color.
    - Rotate that full marked sprite clockwise by (s mod 4) quarter-turns around its bbox center.
    - For every isolated target marker of the sprite's marker color, stamp the rotated sprite so the rotated marker pixel lands exactly on the target marker.
    - Clear the original combs and original source sprites; keep only the stamped copies, with later combs in row-major comb order overwriting earlier stamps on overlap.

    Key insight:
    - The comb does not encode a quantity to copy directly; its solid-row count is the rotation parameter that must be applied before anchor-based stamping.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(_get_bg(g))
    combs = sorted(_find_combs(g, bg), key=lambda item: (item["bbox"][0], item["bbox"][1]))
    sources = _find_sources(g, bg)
    out = np.full_like(g, int(bg))

    for comb in combs:
        source = sources.get(int(comb["color"]))
        if source is None:
            continue
        rotated = _rotate_cw(source["patch"], int(comb["solid_count"]) % 4)
        marker_cells = np.argwhere(rotated == int(source["marker_color"]))
        if marker_cells.shape[0] != 1:
            continue
        ar, ac = map(int, marker_cells[0])
        for tr, tc in source["targets"]:
            _paste(out, rotated, int(tr) - ar, int(tc) - ac, transparent=bg)
    return out


def _rotation_signature(patch: np.ndarray, marker_color: int, body_color: int, k: int) -> tuple[tuple[int, int], ...]:
    rotated = _rotate_cw(patch, k)
    mr, mc = np.argwhere(rotated == int(marker_color))[0]
    coords = []
    for r, c in np.argwhere(rotated == int(body_color)):
        coords.append((int(r) - int(mr), int(c) - int(mc)))
    return tuple(sorted(coords))


def _recolor_template(template: np.ndarray, body_color: int, marker_color: int) -> np.ndarray:
    patch = np.full(template.shape, BG, dtype=int)
    patch[template == 1] = int(body_color)
    patch[template == 2] = int(marker_color)
    return patch


def _random_source_patch(rng: random.Random, body_color: int, marker_color: int) -> np.ndarray:
    for _ in range(200):
        template = np.array(rng.choice(TEMPLATES), copy=True)
        if rng.random() < 0.5:
            template = np.fliplr(template)
        if rng.random() < 0.5:
            template = np.flipud(template)
        template = np.rot90(template, rng.randrange(4))
        patch = _recolor_template(template, body_color, marker_color)
        sigs = {_rotation_signature(patch, marker_color, body_color, k) for k in range(4)}
        if len(sigs) == 4:
            return patch
    raise RuntimeError("failed to generate a non-symmetric source patch")


def _comb_patch(body_color: int, solid_count: int) -> np.ndarray:
    h = 2 * solid_count - 1
    patch = np.zeros((h, 3), dtype=int)
    for r in range(h):
        if r % 2 == 0:
            patch[r, :] = int(body_color)
        else:
            patch[r, 0] = int(body_color)
            patch[r, 2] = int(body_color)
    return patch


def _compose_cluster(rng: random.Random, comb_patch: np.ndarray, source_patch: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    ch, cw = comb_patch.shape
    sh, sw = source_patch.shape
    side = rng.choice(["left", "right", "up", "down"])
    gap = rng.choice([2, 3, 4])
    delta = rng.choice([-1, 0, 1])
    if side in ("left", "right"):
        top_pad = max(0, -delta)
        bot_pad = max(0, delta)
        height = max(ch + top_pad, sh + bot_pad)
        width = cw + gap + sw
        patch = np.zeros((height, width), dtype=int)
        if side == "right":
            comb_pos = (top_pad, 0)
            source_pos = (bot_pad, cw + gap)
        else:
            source_pos = (bot_pad, 0)
            comb_pos = (top_pad, sw + gap)
    else:
        left_pad = max(0, -delta)
        right_pad = max(0, delta)
        height = ch + gap + sh
        width = max(cw + left_pad, sw + right_pad)
        patch = np.zeros((height, width), dtype=int)
        if side == "down":
            comb_pos = (0, left_pad)
            source_pos = (ch + gap, right_pad)
        else:
            source_pos = (0, right_pad)
            comb_pos = (sh + gap, left_pad)
    _paste(patch, comb_patch, comb_pos[0], comb_pos[1], transparent=BG)
    _paste(patch, source_patch, source_pos[0], source_pos[1], transparent=BG)
    return patch, comb_pos, source_pos


def _isolated(point: tuple[int, int], others: list[tuple[int, int]], min_chebyshev: int = 2) -> bool:
    r, c = point
    for orow, ocol in others:
        if max(abs(r - orow), abs(c - ocol)) < min_chebyshev:
            return False
    return True


def _stamped_cells(patch: np.ndarray, marker_color: int, target: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]]:
    mr, mc = map(int, np.argwhere(patch == int(marker_color))[0])
    r0 = int(target[0]) - mr
    c0 = int(target[1]) - mc
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(patch != BG):
        rr = r0 + int(r)
        cc = c0 + int(c)
        if _in_bounds(shape, rr, cc):
            cells.add((rr, cc))
    return cells


def _render_pair(patch: np.ndarray, marker_color: int, targets: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for target in targets:
        mr, mc = map(int, np.argwhere(patch == int(marker_color))[0])
        _paste(out, patch, int(target[0]) - mr, int(target[1]) - mc, transparent=BG)
    return out


def _choose_target_counts(rng: random.Random, n_pairs: int) -> list[int]:
    options: list[tuple[int, ...]] = []
    if n_pairs == 2:
        for a in (1, 2, 3):
            for b in (1, 2, 3):
                if 3 <= a + b <= 5:
                    options.append((a, b))
    else:
        for a in (1, 2, 3):
            for b in (1, 2, 3):
                for c in (1, 2, 3):
                    if 4 <= a + b + c <= 6:
                        options.append((a, b, c))
    return list(rng.choice(options))


def _pick_target_positions(
    rng: random.Random,
    shape: tuple[int, int],
    rotated_patch: np.ndarray,
    marker_color: int,
    count: int,
    existing_targets: list[tuple[int, int]],
    occupied_output: set[tuple[int, int]],
    need_clip: bool,
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    H, W = shape
    patch_nz = int(np.sum(rotated_patch != BG))
    candidates = [(r, c) for r in range(H) for c in range(W)]
    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    new_cells_total: set[tuple[int, int]] = set()
    clip_remaining = int(need_clip)
    for _ in range(count):
        picked = None
        picked_cells: set[tuple[int, int]] | None = None
        for tr, tc in candidates:
            target = (int(tr), int(tc))
            if target in chosen or not _isolated(target, existing_targets + chosen, min_chebyshev=2):
                continue
            cells = _stamped_cells(rotated_patch, marker_color, target, shape)
            is_clipped = len(cells) < patch_nz
            if clip_remaining and not is_clipped:
                continue
            if not clip_remaining and is_clipped and rng.random() < 0.85:
                continue
            if cells & occupied_output:
                continue
            picked = target
            picked_cells = cells
            break
        if picked is None or picked_cells is None:
            return None
        chosen.append(picked)
        new_cells_total |= picked_cells
        if clip_remaining and len(picked_cells) < patch_nz:
            clip_remaining = 0
    if clip_remaining:
        return None
    return chosen, new_cells_total


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        nxt = out.copy()
        for r, c in np.argwhere(out):
            rr = int(r)
            cc = int(c)
            for dr, dc in DIRS8:
                nr = rr + dr
                nc = cc + dc
                if _in_bounds(mask.shape, nr, nc):
                    nxt[nr, nc] = True
        out = nxt
    return out


def _random_free_location(rng: random.Random, canvas_occ: np.ndarray, patch: np.ndarray, padding: int) -> tuple[int, int] | None:
    h, w = patch.shape
    H, W = canvas_occ.shape
    if h > H or w > W:
        return None
    occ = _dilate(canvas_occ, padding) if padding > 0 else canvas_occ
    patch_mask = patch != BG
    candidates: list[tuple[int, int]] = []
    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            if not bool((occ[r0 : r0 + h, c0 : c0 + w] & patch_mask).any()):
                candidates.append((r0, c0))
    if not candidates:
        return None
    return rng.choice(candidates)


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, metadata: list[dict]) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    combs = _find_combs(inp, BG)
    sources = _find_sources(inp, BG)
    assert len(combs) == len(metadata)
    assert len(sources) == len(metadata)
    assert len({comb["solid_count"] % 4 for comb in combs}) == len(combs)
    assert np.any(expected != BG)
    for item in metadata:
        patch = item["source_patch"]
        body_color = item["body_color"]
        marker_color = item["marker_color"]
        sigs = {_rotation_signature(patch, marker_color, body_color, k) for k in range(4)}
        assert len(sigs) == 4
        correct = _render_pair(item["rotated_patch"], marker_color, item["targets"], inp.shape)
        assert np.any(correct != BG)
        for k in range(4):
            if k == item["solid_count"] % 4:
                continue
            wrong = _render_pair(_rotate_cw(patch, k), marker_color, item["targets"], inp.shape)
            assert not np.array_equal(correct, wrong)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        H = rng.randint(18, 24)
        W = rng.randint(18, 24)
        n_pairs = rng.choice([2, 3])
        body_colors = rng.sample(list(BODY_COLORS), n_pairs)
        solid_counts = rng.sample([1, 2, 3], n_pairs)
        marker_colors = rng.sample(list(MARKER_COLORS), n_pairs)
        target_counts = _choose_target_counts(rng, n_pairs)

        pair_defs: list[dict] = []
        for body_color, solid_count, marker_color in zip(body_colors, solid_counts, marker_colors):
            source_patch = _random_source_patch(rng, body_color, marker_color)
            rotated_patch = _rotate_cw(source_patch, solid_count % 4)
            pair_defs.append(
                {
                    "body_color": int(body_color),
                    "solid_count": int(solid_count),
                    "marker_color": int(marker_color),
                    "source_patch": source_patch,
                    "rotated_patch": rotated_patch,
                    "comb_patch": _comb_patch(body_color, solid_count),
                }
            )

        all_targets: list[tuple[int, int]] = []
        occupied_output: set[tuple[int, int]] = set()
        clip_index = rng.randrange(n_pairs) if rng.random() < 0.5 else -1
        ok = True
        for i, item in enumerate(pair_defs):
            placed = _pick_target_positions(
                rng,
                (H, W),
                item["rotated_patch"],
                item["marker_color"],
                target_counts[i],
                all_targets,
                occupied_output,
                need_clip=(i == clip_index),
            )
            if placed is None:
                ok = False
                break
            targets, cells = placed
            item["targets"] = targets
            all_targets.extend(targets)
            occupied_output |= cells
        if not ok:
            continue

        inp = np.zeros((H, W), dtype=int)
        for item in pair_defs:
            for tr, tc in item["targets"]:
                inp[tr, tc] = int(item["marker_color"])

        occ = inp != BG
        for item in pair_defs:
            cluster, comb_pos, source_pos = _compose_cluster(rng, item["comb_patch"], item["source_patch"])
            loc = _random_free_location(rng, occ, cluster, padding=1)
            if loc is None:
                ok = False
                break
            r0, c0 = loc
            _paste(inp, cluster, r0, c0, transparent=BG)
            occ = inp != BG
            item["comb_bbox"] = (
                r0 + comb_pos[0],
                c0 + comb_pos[1],
                r0 + comb_pos[0] + item["comb_patch"].shape[0] - 1,
                c0 + comb_pos[1] + item["comb_patch"].shape[1] - 1,
            )
        if not ok:
            continue

        expected = np.zeros((H, W), dtype=int)
        for item in sorted(pair_defs, key=lambda x: (x["comb_bbox"][0], x["comb_bbox"][1])):
            mr, mc = map(int, np.argwhere(item["rotated_patch"] == int(item["marker_color"]))[0])
            for tr, tc in item["targets"]:
                _paste(expected, item["rotated_patch"], int(tr) - mr, int(tc) - mc, transparent=BG)

        try:
            _semantic_checks(inp, expected, pair_defs)
        except AssertionError:
            continue

        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
