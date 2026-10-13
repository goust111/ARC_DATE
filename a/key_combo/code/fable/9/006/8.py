from __future__ import annotations

import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
PALETTE = list(range(10))
MAX_GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    shifted = coords - np.array([r0, c0])
    mask[shifted[:, 0], shifted[:, 1]] = True
    return mask


def _transform_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            key = (cur.shape, cur.tobytes())
            if key not in seen:
                seen.add(key)
                out.append(cur)
    return out


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((m.shape, m.tobytes()) for m in _transform_variants(mask))


def _infer_checkerboard(panel: np.ndarray) -> dict | None:
    colors = sorted(int(x) for x in np.unique(panel))
    if len(colors) != 2:
        return None

    rr, cc = np.indices(panel.shape)
    solutions = []
    for fg in colors:
        bg = colors[0] if colors[1] == fg else colors[1]
        for fg_parity in (0, 1):
            fg_cells = (rr + cc) % 2 == fg_parity
            if not np.all(panel[fg_cells] == fg):
                continue
            if not np.all((panel[~fg_cells] == fg) | (panel[~fg_cells] == bg)):
                continue
            defects = (~fg_cells) & (panel == fg)
            if not defects.any():
                continue
            if not np.any(panel[~fg_cells] == bg):
                continue
            solutions.append(
                {
                    "fg": int(fg),
                    "bg": int(bg),
                    "fg_parity": int(fg_parity),
                    "defects": defects,
                }
            )
    if len(solutions) != 1:
        return None
    return solutions[0]


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    mono_components = []
    for color in sorted(set(int(x) for x in np.unique(g)) - {bg}):
        for coords in _components(g == color, connectivity=4):
            mono_components.append({"color": int(color), "coords": coords, "bbox": _bbox_coords(coords), "area": int(coords.shape[0])})

    assert mono_components, "expected non-background objects"
    board_component = max(mono_components, key=lambda comp: comp["area"])
    board_color = int(board_component["color"])
    br0, bc0, br1, bc1 = board_component["bbox"]
    board = g[br0 : br1 + 1, bc0 : bc1 + 1]
    assert BLACK in set(int(x) for x in np.unique(board)), "board must contain black holes"
    assert set(int(x) for x in np.unique(board)) == {BLACK, board_color}, "board must be grey with black holes only"
    assert not np.any(board == bg), "board bbox must be solid"

    binary_components = []
    for coords in _components(g != bg, connectivity=8):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(sub != bg):
            binary_components.append({"coords": coords, "bbox": bbox})

    assert len(binary_components) == 2, "expected exactly two solid non-background regions"
    panel_component = [comp for comp in binary_components if comp["bbox"] != board_component["bbox"]]
    assert len(panel_component) == 1, "expected one checkerboard panel distinct from the board"
    pr0, pc0, pr1, pc1 = panel_component[0]["bbox"]
    panel = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
    panel_info = _infer_checkerboard(panel)
    assert panel_info is not None, "failed to infer checkerboard parity"

    defect_coords = np.argwhere(panel_info["defects"])
    defect_mask = _mask_from_coords(defect_coords)
    target_key = _canonical_key(defect_mask)

    hole_infos = []
    for coords in _components(board == BLACK, connectivity=8):
        mask = _mask_from_coords(coords)
        hole_infos.append(
            {
                "coords": coords,
                "mask": mask,
                "bbox": _bbox_coords(coords),
                "key": _canonical_key(mask),
                "area": int(coords.shape[0]),
            }
        )

    matches = [hole for hole in hole_infos if hole["key"] == target_key]
    assert len(matches) == 1, "defect shape must match exactly one hole up to rotation/flip"

    return {
        "canvas_bg": int(bg),
        "board_bbox": (br0, bc0, br1, bc1),
        "board_color": int(board_color),
        "board": board,
        "panel_bbox": (pr0, pc0, pr1, pc1),
        "panel": panel,
        "panel_info": panel_info,
        "defect_mask": defect_mask,
        "holes": hole_infos,
        "target_hole": matches[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn parity-violating checkerboard pixels into a shape key, repair the checkerboard, and fill the matching board hole.

    Input structure:
    - One object is a two-color checkerboard panel. One parity class is a foreground color; some cells on the opposite parity are wrongly recolored to that same foreground color.
    - Another object is a solid grey board that contains several black holes of different shapes.
    - The two objects sit on a uniform canvas background and do not touch each other.

    Rule:
    - Infer the checkerboard parity phase and locate every defect cell: a foreground-colored cell that lies on the parity that should be background.
    - Take the defects' relative layout as a shape mask. Among the black holes, find the unique hole congruent to that mask under rotation or mirror.
    - Fill only that matched hole with the checkerboard foreground color, keep the other holes black, and recolor every checkerboard defect back to its parity-correct background color.

    Key insight:
    - The useful shape is not a standalone object: it is encoded only by the checkerboard parity violations, so the same parity test both extracts the shape and tells how to repair the panel.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_input(g)
    out = g.copy()

    pr0, pc0, pr1, pc1 = info["panel_bbox"]
    panel = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
    panel[info["panel_info"]["defects"]] = int(info["panel_info"]["bg"])

    br0, bc0, _br1, _bc1 = info["board_bbox"]
    board = out[br0 : info["board_bbox"][2] + 1, bc0 : info["board_bbox"][3] + 1]
    target_coords = info["target_hole"]["coords"]
    board[target_coords[:, 0], target_coords[:, 1]] = int(info["panel_info"]["fg"])
    return out


def _connected_subset(points: tuple[tuple[int, int], ...]) -> bool:
    pts = set(points)
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _normalize_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple(sorted((r - min_r, c - min_c) for r, c in points))


def _coarse_to_mask(points: tuple[tuple[int, int], ...]) -> np.ndarray:
    mapped = [(r + c, r - c) for r, c in points]
    min_r = min(r for r, _ in mapped)
    min_c = min(c for _, c in mapped)
    shifted = [(r - min_r, c - min_c) for r, c in mapped]
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    return mask


def _build_shape_library() -> list[np.ndarray]:
    seen: set[tuple[tuple[int, int], bytes]] = set()
    library: list[np.ndarray] = []
    for h, w in ((2, 3), (3, 3), (3, 4)):
        cells = [(r, c) for r in range(h) for c in range(w)]
        for area in range(3, min(6, h * w) + 1):
            for points in itertools.combinations(cells, area):
                if not _connected_subset(points):
                    continue
                norm = _normalize_points(points)
                mask = _coarse_to_mask(norm)
                if max(mask.shape) > 7:
                    continue
                key = _canonical_key(mask)
                if key in seen:
                    continue
                seen.add(key)
                library.append(mask)
    library.sort(key=lambda m: (int(m.sum()), m.shape[0] * m.shape[1], m.shape))
    return library


SHAPE_LIBRARY = _build_shape_library()


def _pick_shape_set(target_mask: np.ndarray, total_holes: int) -> list[np.ndarray]:
    target_key = _canonical_key(target_mask)
    target_area = int(target_mask.sum())
    target_shape = target_mask.shape

    pool = [mask for mask in SHAPE_LIBRARY if _canonical_key(mask) != target_key]
    same_area = [mask for mask in pool if int(mask.sum()) == target_area]
    same_bbox = [mask for mask in pool if mask.shape == target_shape]

    chosen: list[np.ndarray] = []
    used = {target_key}

    def take(candidates: list[np.ndarray]) -> None:
        random.shuffle(candidates)
        for mask in candidates:
            key = _canonical_key(mask)
            if key not in used:
                used.add(key)
                chosen.append(mask)
                return

    take(same_area)
    take(same_bbox)

    rest = pool[:]
    random.shuffle(rest)
    for mask in rest:
        if len(chosen) >= total_holes - 1:
            break
        key = _canonical_key(mask)
        if key not in used:
            used.add(key)
            chosen.append(mask)

    if len(chosen) < total_holes - 1:
        raise RuntimeError("shape library too small for distractor selection")
    return [target_mask] + chosen[: total_holes - 1]


def _random_variant(mask: np.ndarray) -> np.ndarray:
    return random.choice(_transform_variants(mask))


def _panel_with_defects(height: int, width: int, fg: int, bg: int, fg_parity: int, defect_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    rr, cc = np.indices((height, width))
    panel = np.where((rr + cc) % 2 == fg_parity, fg, bg).astype(int)

    defect_parity = 1 - fg_parity
    q = None
    coords = np.argwhere(defect_mask)
    for r, c in coords:
        q = int((r + c) % 2)
        break
    assert q is not None

    candidates = []
    mh, mw = defect_mask.shape
    for top in range(1, height - mh):
        for left in range(1, width - mw):
            if (top + left + q) % 2 == defect_parity:
                candidates.append((top, left))
    if not candidates:
        raise RuntimeError("no valid defect placement")
    top, left = random.choice(candidates)
    local = np.argwhere(defect_mask)
    panel[top + local[:, 0], left + local[:, 1]] = fg
    return panel, (int(top), int(left))


def _can_place(mask: np.ndarray, occ: np.ndarray, top: int, left: int) -> bool:
    h, w = mask.shape
    board_h, board_w = occ.shape
    if top < 1 or left < 1 or top + h > board_h - 1 or left + w > board_w - 1:
        return False
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if occ[nr, nc]:
                    return False
    return True


def _place_mask(mask: np.ndarray, occ: np.ndarray, board: np.ndarray, top: int, left: int) -> np.ndarray:
    coords = np.argwhere(mask)
    for r, c in coords:
        rr = top + int(r)
        cc = left + int(c)
        board[rr, cc] = BLACK
        occ[rr, cc] = True
    return coords + np.array([top, left])


def _board_with_holes(height: int, width: int, hole_masks: list[np.ndarray], target_key: tuple[tuple[int, int], bytes]) -> tuple[np.ndarray, dict]:
    board = np.full((height, width), GREY, dtype=int)
    occ = np.zeros((height, width), dtype=bool)
    order = list(range(len(hole_masks)))
    random.shuffle(order)
    placed_meta = []

    for idx in order:
        oriented = _random_variant(hole_masks[idx])
        h, w = oriented.shape
        positions = [(top, left) for top in range(1, height - h) for left in range(1, width - w) if _can_place(oriented, occ, top, left)]
        if not positions:
            raise RuntimeError("failed to place hole")
        top, left = random.choice(positions)
        coords = _place_mask(oriented, occ, board, top, left)
        meta = {"coords": coords, "mask": oriented, "key": _canonical_key(oriented), "area": int(oriented.sum()), "bbox": (top, left, top + h - 1, left + w - 1)}
        placed_meta.append(meta)

    target_matches = [hole for hole in placed_meta if hole["key"] == target_key]
    if len(target_matches) != 1:
        raise RuntimeError("hole placement lost target uniqueness")
    return board, {"holes": placed_meta, "target": target_matches[0]}


def _assemble_canvas(panel: np.ndarray, board: np.ndarray, canvas_bg: int) -> np.ndarray:
    ph, pw = panel.shape
    bh, bw = board.shape
    layout = random.choice(["horizontal", "vertical"])
    swap = random.choice([False, True])
    gap = random.randint(2, 4)
    margin = random.randint(1, 3)

    if layout == "horizontal":
        height = max(ph, bh) + 2 * margin
        width = pw + bw + gap + 2 * margin
        if height > 30 or width > 30:
            raise RuntimeError("canvas too large")
        canvas = np.full((height, width), canvas_bg, dtype=int)
        top_panel = margin + random.randint(0, height - 2 * margin - ph)
        top_board = margin + random.randint(0, height - 2 * margin - bh)
        left_a = margin
        left_b = margin + pw + gap
        if swap:
            canvas[top_board : top_board + bh, left_a : left_a + bw] = board
            canvas[top_panel : top_panel + ph, left_b : left_b + pw] = panel
        else:
            canvas[top_panel : top_panel + ph, left_a : left_a + pw] = panel
            canvas[top_board : top_board + bh, left_b : left_b + bw] = board
        return canvas

    height = ph + bh + gap + 2 * margin
    width = max(pw, bw) + 2 * margin
    if height > 30 or width > 30:
        raise RuntimeError("canvas too large")
    canvas = np.full((height, width), canvas_bg, dtype=int)
    left_panel = margin + random.randint(0, width - 2 * margin - pw)
    left_board = margin + random.randint(0, width - 2 * margin - bw)
    top_a = margin
    top_b = margin + ph + gap
    if swap:
        canvas[top_a : top_a + bh, left_board : left_board + bw] = board
        canvas[top_b : top_b + ph, left_panel : left_panel + pw] = panel
    else:
        canvas[top_a : top_a + ph, left_panel : left_panel + pw] = panel
        canvas[top_b : top_b + bh, left_board : left_board + bw] = board
    return canvas


def _perfect_panel(panel: np.ndarray, fg: int, bg: int, fg_parity: int) -> np.ndarray:
    rr, cc = np.indices(panel.shape)
    return np.where((rr + cc) % 2 == fg_parity, fg, bg).astype(int)


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    info = _analyze_input(grid)
    out = np.asarray(grid, dtype=int).copy()
    pr0, pc0, pr1, pc1 = info["panel_bbox"]
    out[pr0 : pr1 + 1, pc0 : pc1 + 1] = _perfect_panel(
        info["panel"],
        info["panel_info"]["fg"],
        info["panel_info"]["bg"],
        info["panel_info"]["fg_parity"],
    )
    br0, bc0, _br1, _bc1 = info["board_bbox"]
    coords = info["target_hole"]["coords"]
    board = out[br0 : info["board_bbox"][2] + 1, bc0 : info["board_bbox"][3] + 1]
    board[coords[:, 0], coords[:, 1]] = int(info["panel_info"]["fg"])
    return out


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    info = _analyze_input(grid)
    defects = info["panel_info"]["defects"]
    assert 3 <= int(defects.sum()) <= 6, "defect shape should be nontrivial"
    assert len(info["holes"]) >= 3, "need distractor holes"

    target_area = int(info["target_hole"]["area"])
    target_shape = info["target_hole"]["mask"].shape
    distractors = [hole for hole in info["holes"] if hole["key"] != info["target_hole"]["key"]]
    assert distractors, "need nonmatching holes"
    assert any(hole["area"] == target_area or hole["mask"].shape == target_shape for hole in distractors), "need a same-area or same-bbox confuser"

    repaired = expected[
        info["panel_bbox"][0] : info["panel_bbox"][2] + 1,
        info["panel_bbox"][1] : info["panel_bbox"][3] + 1,
    ]
    ideal = _perfect_panel(repaired, info["panel_info"]["fg"], info["panel_info"]["bg"], info["panel_info"]["fg_parity"])
    assert np.array_equal(repaired, ideal), "expected panel repair must be a perfect checkerboard"

    board_expected = expected[
        info["board_bbox"][0] : info["board_bbox"][2] + 1,
        info["board_bbox"][1] : info["board_bbox"][3] + 1,
    ]
    board_input = grid[
        info["board_bbox"][0] : info["board_bbox"][2] + 1,
        info["board_bbox"][1] : info["board_bbox"][3] + 1,
    ]
    changed = board_expected != board_input
    assert int(changed.sum()) == target_area, "only the matched hole should change on the board"
    assert np.array_equal(main(grid), expected), "solver must match the independently constructed oracle"


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            canvas_bg = random.choice([c for c in PALETTE if c not in {BLACK, GREY}])
            fg, checker_bg = random.sample([c for c in PALETTE if c not in {canvas_bg, BLACK, GREY}], 2)
            target_mask = random.choice(SHAPE_LIBRARY)

            total_holes = random.randint(3, 5)
            hole_masks = _pick_shape_set(target_mask, total_holes)
            target_key = _canonical_key(target_mask)

            panel_h = random.randint(max(7, target_mask.shape[0] + 3), 12)
            panel_w = random.randint(max(7, target_mask.shape[1] + 3), 12)
            fg_parity = random.randint(0, 1)
            panel, _placement = _panel_with_defects(panel_h, panel_w, fg, checker_bg, fg_parity, target_mask)

            max_hole_h = max(mask.shape[0] for mask in hole_masks)
            max_hole_w = max(mask.shape[1] for mask in hole_masks)
            board_h = random.randint(max(8, max_hole_h + 4), 14)
            board_w = random.randint(max(8, max_hole_w + 4), 14)
            board, board_meta = _board_with_holes(board_h, board_w, hole_masks, target_key)

            grid = _assemble_canvas(panel, board, canvas_bg)
            expected = grid.copy()
            info = _analyze_input(grid)
            pr0, pc0, pr1, pc1 = info["panel_bbox"]
            expected[pr0 : pr1 + 1, pc0 : pc1 + 1] = _perfect_panel(
                info["panel"],
                info["panel_info"]["fg"],
                info["panel_info"]["bg"],
                info["panel_info"]["fg_parity"],
            )
            br0, bc0, _br1, _bc1 = info["board_bbox"]
            coords = info["target_hole"]["coords"]
            expected[br0 : info["board_bbox"][2] + 1, bc0 : info["board_bbox"][3] + 1][coords[:, 0], coords[:, 1]] = info["panel_info"]["fg"]

            _validate_generated(grid, expected)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid input")
