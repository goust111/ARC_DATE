from __future__ import annotations

import numpy as np


BG = 0
BLUE = 1
YELLOW = 4
MAROON = 9
PIECE_COLORS = [2, 3, 5, 6, 7, 8]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_RNG = np.random.default_rng()
_MAX_GEN_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(pts)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = _normalize_mask(mask)
    return m.shape, m.tobytes()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cur: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cur.append((r, c))
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cur, dtype=int))
    return comps


def _find_monochrome_components(grid: np.ndarray) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid)):
        if color == BG:
            continue
        for coords in _components(grid == color):
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return comps


def _extract_holes(grid: np.ndarray, comp: dict) -> list[dict]:
    r0, c0, r1, c1 = comp["bbox"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[dict] = []
    for rel in _components(patch == BG):
        if np.any(rel[:, 0] == 0) or np.any(rel[:, 0] == patch.shape[0] - 1):
            continue
        if np.any(rel[:, 1] == 0) or np.any(rel[:, 1] == patch.shape[1] - 1):
            continue
        coords = rel + np.array([r0, c0], dtype=int)
        holes.append({"coords": coords, "bbox": _bbox(coords)})
    return holes


def _find_template(grid: np.ndarray, comps: list[dict]) -> tuple[dict, list[dict]]:
    best_comp = None
    best_holes: list[dict] = []
    best_key = None
    for comp in comps:
        holes = _extract_holes(grid, comp)
        if not holes:
            continue
        key = (len(holes), int(comp["area"]))
        if best_key is None or key > best_key:
            best_key = key
            best_comp = comp
            best_holes = holes
    assert best_comp is not None, "expected a template component with enclosed holes"
    return best_comp, best_holes


def _count_template_contacts(coords: np.ndarray, grid: np.ndarray, template_color: int) -> int:
    h, w = grid.shape
    total = 0
    for r, c in coords:
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == template_color:
                total += 1
    return total


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for dr, dc in np.argwhere(mask):
        grid[top + int(dr), left + int(dc)] = int(color)


def _global_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    pts = np.argwhere(mask)
    return pts + np.array([top, left], dtype=int)


def _mask_perimeter(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                total += 1
    return total


def _forbidden_ok(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = blocked.shape
    for dr, dc in np.argwhere(mask):
        rr = top + int(dr)
        cc = left + int(dc)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                if blocked[nr, nc]:
                    return False
    return True


def _mark_forbidden(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = blocked.shape
    for dr, dc in np.argwhere(mask):
        rr = top + int(dr)
        cc = left + int(dc)
        for nr in range(max(0, rr - 1), min(h, rr + 2)):
            for nc in range(max(0, cc - 1), min(w, cc + 2)):
                blocked[nr, nc] = True


def _place_masks(
    masks: list[np.ndarray],
    canvas_h: int,
    canvas_w: int,
    row_low: int,
    row_high_exclusive: int,
    col_low: int,
    col_high_exclusive: int,
) -> list[tuple[int, int]] | None:
    blocked = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(masks)
    order = sorted(range(len(masks)), key=lambda i: (-int(masks[i].sum()), -_mask_perimeter(masks[i])))
    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        top_stop = row_high_exclusive - mh
        left_stop = col_high_exclusive - mw
        if row_low > top_stop or col_low > left_stop:
            return None
        candidates: list[tuple[int, int]] = []
        for top in range(row_low, top_stop + 1):
            for left in range(col_low, left_stop + 1):
                if _forbidden_ok(blocked, mask, top, left):
                    candidates.append((top, left))
        if not candidates:
            return None
        choice = candidates[int(_RNG.integers(len(candidates)))]
        placements[idx] = choice
        _mark_forbidden(blocked, mask, choice[0], choice[1])
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _random_shape() -> np.ndarray:
    for _ in range(120):
        target_area = int(_RNG.integers(3, 8))
        max_dim = int(_RNG.integers(3, 6))
        cells = {(0, 0)}
        for _step in range(target_area - 1):
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    p = (r + dr, c + dc)
                    if p not in cells:
                        frontier.add(p)
            options = []
            for nr, nc in frontier:
                trial = list(cells) + [(nr, nc)]
                rs = [p[0] for p in trial]
                cs = [p[1] for p in trial]
                if max(rs) - min(rs) + 1 <= max_dim and max(cs) - min(cs) + 1 <= max_dim:
                    options.append((nr, nc))
            if not options:
                break
            pick = options[int(_RNG.integers(len(options)))]
            cells.add(pick)
        if len(cells) != target_area:
            continue
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        h = max(rs) - min(rs) + 1
        w = max(cs) - min(cs) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r - min(rs), c - min(cs)] = True
        if _RNG.random() < 0.5:
            mask = np.fliplr(mask)
        mask = np.rot90(mask, int(_RNG.integers(4)))
        mask = _normalize_mask(mask)
        if mask.shape[0] > 5 or mask.shape[1] > 5:
            continue
        if mask.shape[0] * mask.shape[1] == int(mask.sum()) and mask.size >= 9 and _RNG.random() < 0.7:
            continue
        return mask
    raise RuntimeError("failed to sample a connected shape")


def _sample_items() -> list[dict] | None:
    n_items = int(_RNG.integers(3, 6))
    for _ in range(80):
        items: list[dict] = []
        seen = set()
        guard = 0
        while len(items) < n_items and guard < 200:
            guard += 1
            mask = _random_shape()
            key = _mask_key(mask)
            if key in seen:
                continue
            items.append({"mask": mask, "key": key, "perimeter": _mask_perimeter(mask)})
            seen.add(key)
        if len(items) != n_items:
            continue
        perimeters = [int(item["perimeter"]) for item in items]
        p_min = min(perimeters)
        p_max = max(perimeters)
        if p_min == p_max:
            continue
        if perimeters.count(p_min) != 1 or perimeters.count(p_max) != 1:
            continue
        colors = list(int(v) for v in _RNG.choice(PIECE_COLORS, size=n_items, replace=False))
        for item, color in zip(items, colors):
            item["color"] = int(color)
        _RNG.shuffle(items)
        return items
    return None


def _build_expected(grid: np.ndarray, items: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    contacts = [int(item["contact"]) for item in items]
    max_contact = max(contacts)
    min_contact = min(contacts)
    for item in items:
        piece_coords = _global_coords(item["mask"], item["piece_top"], item["piece_left"])
        out[piece_coords[:, 0], piece_coords[:, 1]] = BG
    for item in items:
        color = int(item["color"])
        contact = int(item["contact"])
        if contact == max_contact:
            fill = YELLOW
        elif contact == min_contact:
            fill = BLUE
        else:
            fill = color
        hole_coords = _global_coords(item["mask"], item["hole_top"], item["hole_left"])
        out[hole_coords[:, 0], hole_coords[:, 1]] = int(fill)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered pieces into the template holes, then recolor the
    most-contacting and least-contacting filled pieces.
    """
    g = np.array(grid, dtype=int)
    comps = _find_monochrome_components(g)
    assert comps, "expected a template and several pieces"

    template, holes = _find_template(g, comps)
    template_color = int(template["color"])
    pieces = [comp for comp in comps if comp is not template]
    assert len(pieces) == len(holes) and len(holes) >= 2, "piece/hole counts must match"

    hole_by_key: dict[tuple[tuple[int, int], bytes], dict] = {}
    for hole in holes:
        mask = _coords_to_mask(hole["coords"])
        key = _mask_key(mask)
        assert key not in hole_by_key, "hole shapes must be unique"
        hole_by_key[key] = {
            "coords": hole["coords"],
            "contact": _count_template_contacts(hole["coords"], g, template_color),
        }

    assignments = []
    for piece in pieces:
        mask = _coords_to_mask(piece["coords"])
        key = _mask_key(mask)
        assert key in hole_by_key, "each piece must match a unique hole by translation"
        assignments.append(
            {
                "piece_coords": piece["coords"],
                "hole_coords": hole_by_key[key]["coords"],
                "piece_color": int(piece["color"]),
                "contact": int(hole_by_key[key]["contact"]),
            }
        )

    out = g.copy()
    for assignment in assignments:
        coords = assignment["piece_coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    contacts = [int(a["contact"]) for a in assignments]
    max_contact = max(contacts)
    min_contact = min(contacts)
    for assignment in assignments:
        color = int(assignment["piece_color"])
        if assignment["contact"] == max_contact:
            fill = YELLOW
        elif assignment["contact"] == min_contact:
            fill = BLUE
        else:
            fill = color
        coords = assignment["hole_coords"]
        out[coords[:, 0], coords[:, 1]] = int(fill)

    return out


def _gen_attempt() -> np.ndarray | None:
    items = _sample_items()
    if items is None:
        return None

    masks = [item["mask"] for item in items]
    max_h = max(mask.shape[0] for mask in masks)
    max_w = max(mask.shape[1] for mask in masks)
    total_area = sum(int(mask.sum()) for mask in masks)

    for _ in range(40):
        template_h_min = max(8, max_h + 4)
        template_w_min = max(11, max_w + 4, total_area // 2)
        template_h = int(_RNG.integers(template_h_min, min(15, template_h_min + 4)))
        template_w = int(_RNG.integers(template_w_min, min(21, template_w_min + 7)))

        hole_positions = _place_masks(
            masks,
            template_h,
            template_w,
            1,
            template_h - 1,
            1,
            template_w - 1,
        )
        if hole_positions is None:
            continue

        min_grid_h = template_h + max_h + 8
        min_grid_w = max(18, template_w + 2, max_w + 4)
        max_grid_h = min(30, max(min_grid_h, template_h + 16))
        max_grid_w = min(30, max(min_grid_w, template_w + 12))
        if min_grid_h > max_grid_h or min_grid_w > max_grid_w:
            continue
        grid_h = int(_RNG.integers(min_grid_h, max_grid_h + 1))
        grid_w = int(_RNG.integers(min_grid_w, max_grid_w + 1))

        template_top = int(_RNG.integers(1, min(4, grid_h - template_h)))
        template_left = int(_RNG.integers(1, grid_w - template_w))

        piece_row_low = template_top + template_h + 2
        piece_row_high = grid_h - 1
        piece_positions = _place_masks(
            masks,
            grid_h,
            grid_w,
            piece_row_low,
            piece_row_high,
            1,
            grid_w - 1,
        )
        if piece_positions is None:
            continue

        grid = np.full((grid_h, grid_w), BG, dtype=int)
        grid[template_top : template_top + template_h, template_left : template_left + template_w] = MAROON

        for item, (hole_top, hole_left), (piece_top, piece_left) in zip(items, hole_positions, piece_positions):
            item["hole_top"] = template_top + int(hole_top)
            item["hole_left"] = template_left + int(hole_left)
            item["piece_top"] = int(piece_top)
            item["piece_left"] = int(piece_left)
            _paint_mask(grid, item["mask"], item["hole_top"], item["hole_left"], BG)
            _paint_mask(grid, item["mask"], item["piece_top"], item["piece_left"], int(item["color"]))

        for item in items:
            hole_coords = _global_coords(item["mask"], item["hole_top"], item["hole_left"])
            item["contact"] = _count_template_contacts(hole_coords, grid, MAROON)

        contacts = [int(item["contact"]) for item in items]
        if contacts.count(min(contacts)) != 1 or contacts.count(max(contacts)) != 1:
            continue

        expected = _build_expected(grid, items)
        try:
            solved = main(grid)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-7 input")
