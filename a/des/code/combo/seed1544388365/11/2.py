from __future__ import annotations

import random

import numpy as np


BG = 0
TEMPLATE_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 31)
TEMPLATE_H_RANGE = (9, 15)
TEMPLATE_W_RANGE = (9, 16)
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _rot180(
    norm: tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> tuple[tuple[tuple[int, int], ...], int, int]:
    rot = [(int(h - 1 - r), int(w - 1 - c)) for r, c in norm]
    return _normalize(rot)


def _shape_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    norm, h, w = _normalize(coords)
    rot, _, _ = _rot180(norm, h, w)
    return min((norm, h, w), (rot, h, w))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    for value in values:
        if int(value) == BG:
            return BG
    return int(values[int(np.argmax(counts))])


def _find_template(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int], set[tuple[int, int]]]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            area = len(comp)
            if best is None or area > best["area"]:
                best = {"color": color, "coords": comp, "area": area}

    assert best is not None, "expected a non-background template"
    template_color = int(best["color"])
    coords = {(int(r), int(c)) for r, c in best["coords"]}
    r0, c0, r1, c1 = _bbox(best["coords"])
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(map(int, np.unique(sub))) <= {bg, template_color}, "template bbox must only contain background and template color"
    assert bool((sub == bg).any()), "template must have missing edge cells"
    assert bool((sub[0, :] == template_color).any())
    assert bool((sub[-1, :] == template_color).any())
    assert bool((sub[:, 0] == template_color).any())
    assert bool((sub[:, -1] == template_color).any())
    return template_color, (int(r0), int(c0), int(r1), int(c1)), coords


def _find_markers(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> dict[str, dict[str, object]]:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox

    line_map = {
        "top": [(r0 - 1, c) for c in range(c0, c1 + 1)] if r0 > 0 else [],
        "bottom": [(r1 + 1, c) for c in range(c0, c1 + 1)] if r1 + 1 < h else [],
        "left": [(r, c0 - 1) for r in range(r0, r1 + 1)] if c0 > 0 else [],
        "right": [(r, c1 + 1) for r in range(r0, r1 + 1)] if c1 + 1 < w else [],
    }

    markers: dict[str, dict[str, object]] = {}
    colors = set()
    for side, line in line_map.items():
        candidates = [(r, c) for r, c in line if int(grid[r, c]) != bg]
        assert len(candidates) == 1, f"expected exactly one marker on {side}"
        r, c = candidates[0]
        color = int(grid[r, c])
        colors.add(color)
        markers[side] = {"pos": (int(r), int(c)), "color": color}
    assert len(colors) == 4, "all four side markers must use distinct colors"
    return markers


def _extract_notches(
    grid: np.ndarray,
    bg: int,
    template_color: int,
    bbox: tuple[int, int, int, int],
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert set(map(int, np.unique(sub))) <= {bg, template_color}

    notches: list[dict[str, object]] = []
    for comp in _components(sub == bg):
        touched = set()
        for rr, cc in comp:
            if rr == 0:
                touched.add("top")
            if rr == sub.shape[0] - 1:
                touched.add("bottom")
            if cc == 0:
                touched.add("left")
            if cc == sub.shape[1] - 1:
                touched.add("right")
        assert len(touched) == 1, "each missing component must belong to exactly one side"
        side = next(iter(touched))
        global_coords = [(int(r0 + rr), int(c0 + cc)) for rr, cc in comp]
        notches.append(
            {
                "side": side,
                "coords": global_coords,
                "signature": _shape_signature(comp),
            }
        )
    assert notches, "expected at least one notch"
    return notches


def _extract_fragments(
    grid: np.ndarray,
    bg: int,
    template_coords: set[tuple[int, int]],
    markers: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    mask = grid != bg
    for r, c in template_coords:
        mask[r, c] = False
    for info in markers.values():
        r, c = info["pos"]
        mask[int(r), int(c)] = False

    fragments = []
    for comp in _components(mask):
        fragments.append({"coords": comp, "signature": _shape_signature(comp)})
    return fragments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed scattered fragments into edge notches of the gray template and recolor them by side markers.

    Input structure:
    - Black background.
    - One large gray rectangular template with black notches carved from its edges.
    - One single-pixel marker outside each template side; marker colors define the fill color for that side.
    - Several scattered monochrome fragments whose shapes match the template notches, up to a 180° rotation.

    Rule:
    - Match each fragment to one edge notch by exact shape (allowing a 180° rotation).
    - Move the fragment into the notch.
    - Recolor the placed fragment to the marker color of that side.
    - Clear the fragment's original location to background.
    - Leave the template and side markers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    template_color, bbox, template_coords = _find_template(g, bg)
    markers = _find_markers(g, bg, bbox)
    notches = _extract_notches(g, bg, template_color, bbox)
    fragments = _extract_fragments(g, bg, template_coords, markers)

    assert len(fragments) == len(notches), "fragment and notch counts must agree"

    fragments_by_signature: dict[tuple[tuple[tuple[int, int], ...], int, int], list[dict[str, object]]] = {}
    for frag in fragments:
        fragments_by_signature.setdefault(frag["signature"], []).append(frag)

    out = g.copy()
    for notch in notches:
        bucket = fragments_by_signature.get(notch["signature"], [])
        assert bucket, "every notch must have a matching fragment"
        frag = bucket.pop()
        for r, c in frag["coords"]:
            out[int(r), int(c)] = bg
        fill_color = int(markers[str(notch["side"])]["color"])
        for r, c in notch["coords"]:
            out[int(r), int(c)] = fill_color

    assert all(not bucket for bucket in fragments_by_signature.values()), "unexpected extra fragments"
    return out


def _random_side_counts() -> dict[str, int]:
    for _ in range(100):
        counts = {side: random.choice([0, 1, 1, 2]) for side in SIDES}
        total = sum(counts.values())
        active = sum(v > 0 for v in counts.values())
        if 2 <= total <= 6 and active >= 2:
            return counts
    raise RuntimeError("failed to choose side counts")


def _make_profile(side: str) -> list[tuple[int, int]]:
    if side in ("top", "bottom"):
        span = random.randint(1, 4)
        max_depth = random.randint(1, 3)
        depths = [random.randint(1, max_depth) for _ in range(span)]
        if sum(depths) == 1:
            if max_depth >= 2:
                depths[0] = 2
            else:
                span = 2
                depths = [1, 1]
        return [(int(r), int(c)) for c, depth in enumerate(depths) for r in range(depth)]

    span = random.randint(1, 4)
    max_width = random.randint(1, 3)
    widths = [random.randint(1, max_width) for _ in range(span)]
    if sum(widths) == 1:
        if max_width >= 2:
            widths[0] = 2
        else:
            span = 2
            widths = [1, 1]
    return [(int(r), int(c)) for r, width in enumerate(widths) for c in range(width)]


def _oriented_coords(
    side: str,
    start: int,
    coords: list[tuple[int, int]],
    template_h: int,
    template_w: int,
) -> list[tuple[int, int]]:
    if side == "top":
        return [(int(r), int(start + c)) for r, c in coords]
    if side == "bottom":
        return [(int(template_h - 1 - r), int(start + c)) for r, c in coords]
    if side == "left":
        return [(int(start + r), int(c)) for r, c in coords]
    return [(int(start + r), int(template_w - 1 - c)) for r, c in coords]


def _can_add_notch(mask: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    cell_set = set(coords)
    h, w = mask.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(mask[r, c]):
            return False
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in cell_set:
                return False
    return True


def _edge_candidates(side: str, notch_mask: np.ndarray) -> list[int]:
    h, w = notch_mask.shape
    if side == "top":
        return [c for c in range(1, w - 1) if not bool(notch_mask[0, c])]
    if side == "bottom":
        return [c for c in range(1, w - 1) if not bool(notch_mask[h - 1, c])]
    if side == "left":
        return [r for r in range(1, h - 1) if not bool(notch_mask[r, 0])]
    return [r for r in range(1, h - 1) if not bool(notch_mask[r, w - 1])]


def _marker_position(
    side: str,
    coord: int,
    top: int,
    left: int,
    template_h: int,
    template_w: int,
) -> tuple[int, int]:
    if side == "top":
        return top - 1, left + coord
    if side == "bottom":
        return top + template_h, left + coord
    if side == "left":
        return top + coord, left - 1
    return top + coord, left + template_w


def _shape_mask(
    shape: tuple[tuple[int, int], ...],
    h: int,
    w: int,
) -> np.ndarray:
    patch = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        patch[int(r), int(c)] = True
    return patch


def _can_place_patch(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if top < 0 or left < 0 or top + ph > h or left + pw > w:
        return False
    for dr, dc in np.argwhere(patch):
        r = int(top + dr)
        c = int(left + dc)
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _place_patch(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    for dr, dc in np.argwhere(patch):
        occupied[int(top + dr), int(left + dc)] = True


def _build_case() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    template_h = random.randint(*TEMPLATE_H_RANGE)
    template_w = random.randint(*TEMPLATE_W_RANGE)
    if template_h + 6 > grid_h or template_w + 6 > grid_w:
        return None

    top = random.randint(2, grid_h - template_h - 3)
    left = random.randint(2, grid_w - template_w - 3)
    side_counts = _random_side_counts()

    notch_mask = np.zeros((template_h, template_w), dtype=bool)
    notches: list[dict[str, object]] = []
    used_signatures = set()

    for side in SIDES:
        for _ in range(side_counts[side]):
            placed = False
            for _ in range(PLACEMENT_ATTEMPTS):
                raw = _make_profile(side)
                if side in ("top", "bottom"):
                    span = max(c for _, c in raw) + 1
                    limit = template_w
                else:
                    span = max(r for r, _ in raw) + 1
                    limit = template_h
                if limit - span - 2 < 1:
                    continue
                start = random.randint(1, limit - span - 2)
                coords = _oriented_coords(side, start, raw, template_h, template_w)
                if not _can_add_notch(notch_mask, coords):
                    continue
                signature = _shape_signature(coords)
                if signature in used_signatures:
                    continue
                for r, c in coords:
                    notch_mask[r, c] = True
                norm, nh, nw = _normalize(coords)
                notches.append(
                    {
                        "side": side,
                        "coords": coords,
                        "shape": norm,
                        "shape_h": nh,
                        "shape_w": nw,
                        "signature": signature,
                    }
                )
                used_signatures.add(signature)
                placed = True
                break
            if not placed:
                return None

    if len(notches) < 2:
        return None

    marker_slots = {}
    for side in SIDES:
        candidates = _edge_candidates(side, notch_mask)
        if not candidates:
            return None
        marker_slots[side] = random.choice(candidates)

    marker_colors = dict(zip(SIDES, random.sample(PALETTE, 4)))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[top : top + template_h, left : left + template_w] = TEMPLATE_COLOR
    for notch in notches:
        for r, c in notch["coords"]:
            grid[top + int(r), left + int(c)] = BG

    for side in SIDES:
        mr, mc = _marker_position(side, marker_slots[side], top, left, template_h, template_w)
        grid[mr, mc] = marker_colors[side]

    occupied = grid != BG
    pieces = []
    order = sorted(range(len(notches)), key=lambda i: len(notches[i]["shape"]), reverse=True)
    for idx in order:
        notch = notches[idx]
        shape = notch["shape"]
        shape_h = int(notch["shape_h"])
        shape_w = int(notch["shape_w"])
        if random.random() < 0.5:
            shape, shape_h, shape_w = _rot180(shape, shape_h, shape_w)
        patch = _shape_mask(shape, shape_h, shape_w)
        piece_colors = [c for c in PALETTE if c != marker_colors[str(notch["side"])]]
        piece_color = random.choice(piece_colors)

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS * 2):
            pr = random.randint(0, grid_h - shape_h)
            pc = random.randint(0, grid_w - shape_w)
            if not _can_place_patch(occupied, patch, pr, pc):
                continue
            _place_patch(occupied, patch, pr, pc)
            pieces.append((patch, pr, pc, piece_color))
            placed = True
            break
        if not placed:
            return None

    for patch, pr, pc, piece_color in pieces:
        for dr, dc in np.argwhere(patch):
            grid[int(pr + dr), int(pc + dc)] = int(piece_color)

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_case()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
