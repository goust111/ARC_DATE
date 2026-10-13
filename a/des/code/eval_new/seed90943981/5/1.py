from __future__ import annotations

import random
from collections import Counter

import numpy as np


SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
PALETTE = list(range(1, 10))

GEN_RETRIES = 300
PLACE_RETRIES = 300
GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
NUM_STRUCTURES_CHOICES = [2, 3, 3]
DEPTH_MAX_CHOICES = [3, 4, 5]
GAP_SPAN_RANGE = (2, 5)
HOLE_SPAN_RANGE = (2, 5)
MARGIN_RANGE = (1, 3)
PADDING = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask_and_origin(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (int(r0), int(c0))


def _touch_sides(coords: list[tuple[int, int]], h: int, w: int) -> set[str]:
    out: set[str] = set()
    for r, c in coords:
        if r == 0:
            out.add("top")
        if r == h - 1:
            out.add("bottom")
        if c == 0:
            out.add("left")
        if c == w - 1:
            out.add("right")
    return out


def _gap_interval(edge_has_color: np.ndarray) -> tuple[int, int] | None:
    if edge_has_color.size < 3 or not bool(edge_has_color[0]) or not bool(edge_has_color[-1]):
        return None
    interior = edge_has_color[1:-1]
    missing = np.where(~interior)[0]
    if missing.size == 0:
        return None
    if not np.array_equal(missing, np.arange(missing[0], missing[-1] + 1)):
        return None
    if bool(interior[: missing[0]].all()) and bool(interior[missing[-1] + 1 :].all()):
        return int(missing[0] + 1), int(missing[-1] + 2)
    return None


def _side_profile(mask: np.ndarray, side: str) -> tuple[int, ...] | None:
    h, w = mask.shape
    if side == "top":
        if not bool(mask[0].all()):
            return None
        depths = []
        for c in range(w):
            d = 0
            while d < h and bool(mask[d, c]):
                d += 1
            if bool(mask[d:, c].any()):
                return None
            depths.append(int(d))
        return tuple(depths)
    if side == "bottom":
        if not bool(mask[-1].all()):
            return None
        depths = []
        for c in range(w):
            d = 0
            while d < h and bool(mask[h - 1 - d, c]):
                d += 1
            if bool(mask[: h - d, c].any()):
                return None
            depths.append(int(d))
        return tuple(depths)
    if side == "left":
        if not bool(mask[:, 0].all()):
            return None
        depths = []
        for r in range(h):
            d = 0
            while d < w and bool(mask[r, d]):
                d += 1
            if bool(mask[r, d:].any()):
                return None
            depths.append(int(d))
        return tuple(depths)
    if not bool(mask[:, -1].all()):
        return None
    depths = []
    for r in range(h):
        d = 0
        while d < w and bool(mask[r, w - 1 - d]):
            d += 1
        if bool(mask[r, : w - d].any()):
            return None
        depths.append(int(d))
    return tuple(depths)


def _mirror_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.flipud(mask)
    return np.fliplr(mask)


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    ph, pw = patch.shape
    window = grid[top : top + ph, left : left + pw]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def _analyze_structure(subgrid: np.ndarray, color: int, bg: int) -> dict | None:
    h, w = subgrid.shape
    if min(h, w) < 6:
        return None

    edge_masks = {
        "top": subgrid[0] == int(color),
        "bottom": subgrid[-1] == int(color),
        "left": subgrid[:, 0] == int(color),
        "right": subgrid[:, -1] == int(color),
    }
    gap_sides = []
    gap_info: dict[str, tuple[int, int]] = {}
    for side, edge in edge_masks.items():
        gap = _gap_interval(edge)
        if gap is None:
            if not bool(edge.all()):
                return None
        else:
            gap_sides.append(side)
            gap_info[side] = gap
    if len(gap_sides) != 1:
        return None
    side = gap_sides[0]

    bg_comps = _components(subgrid == int(bg))
    socket = None
    enclosed = 0
    for comp in bg_comps:
        touched = _touch_sides(comp, h, w)
        if not touched:
            enclosed += 1
            continue
        if touched == {side}:
            if socket is None or len(comp) > len(socket):
                socket = comp
            continue
        return None

    if socket is None or enclosed == 0:
        return None

    socket_mask, (sr0, sc0) = _tight_mask_and_origin(socket)
    profile = _side_profile(socket_mask, side)
    if profile is None:
        return None

    gap_start, gap_end = gap_info[side]
    if side == "top":
        if sr0 != 0 or sc0 != gap_start or socket_mask.shape[1] != gap_end - gap_start:
            return None
    elif side == "bottom":
        if sr0 + socket_mask.shape[0] != h or sc0 != gap_start or socket_mask.shape[1] != gap_end - gap_start:
            return None
    elif side == "left":
        if sc0 != 0 or sr0 != gap_start or socket_mask.shape[0] != gap_end - gap_start:
            return None
    else:
        if sc0 + socket_mask.shape[1] != w or sr0 != gap_start or socket_mask.shape[0] != gap_end - gap_start:
            return None

    return {
        "type": "structure",
        "side": side,
        "profile": profile,
        "socket_mask": socket_mask,
        "socket_origin": (int(sr0), int(sc0)),
        "socket_coords_local": [(int(r), int(c)) for r, c in socket],
        "color": int(color),
    }


def _analyze_piece(mask: np.ndarray, color: int) -> dict:
    profiles = {
        "top": _side_profile(mask, "bottom"),
        "bottom": _side_profile(mask, "top"),
        "left": _side_profile(mask, "right"),
        "right": _side_profile(mask, "left"),
    }
    mirrors = {
        "top": np.flipud(mask),
        "bottom": np.flipud(mask),
        "left": np.fliplr(mask),
        "right": np.fliplr(mask),
    }
    return {
        "type": "piece",
        "mask": mask,
        "profiles": profiles,
        "mirrors": mirrors,
        "color": int(color),
    }


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    structures: list[dict] = []
    pieces: list[dict] = []

    for color in sorted(int(x) for x in np.unique(g) if int(x) != int(bg)):
        for comp in _components(g == int(color)):
            r0, c0, r1, c1 = _bbox(comp)
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            info = _analyze_structure(sub, int(color), int(bg))
            if info is not None:
                info["bbox"] = (int(r0), int(c0), int(r1), int(c1))
                structures.append(info)
                continue
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True
            piece = _analyze_piece(mask, int(color))
            piece["bbox"] = (int(r0), int(c0), int(r1), int(c1))
            pieces.append(piece)

    structures.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["color"]))
    pieces.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["color"]))
    return structures, pieces, int(bg)


def _candidate_piece(structure: dict, piece: dict) -> bool:
    side = str(structure["side"])
    if piece["profiles"][side] != structure["profile"]:
        return False
    mirrored = piece["mirrors"][side]
    return mirrored.shape == structure["socket_mask"].shape and bool(np.array_equal(mirrored, structure["socket_mask"]))


def _match_structures(structures: list[dict], pieces: list[dict]) -> dict[int, int]:
    candidates = {si: [pi for pi, piece in enumerate(pieces) if _candidate_piece(structure, piece)] for si, structure in enumerate(structures)}
    if not structures or not pieces:
        raise AssertionError("expected at least one structure and one piece")
    if any(not cands for cands in candidates.values()):
        raise AssertionError("every structure must have a matching piece")

    order = sorted(range(len(structures)), key=lambda idx: (len(candidates[idx]), structures[idx]["bbox"][0], structures[idx]["bbox"][1]))
    used: set[int] = set()
    assign: dict[int, int] = {}

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        si = order[k]
        for pi in candidates[si]:
            if pi in used:
                continue
            used.add(pi)
            assign[si] = pi
            if dfs(k + 1):
                return True
            used.remove(pi)
            del assign[si]
        return False

    if not dfs(0):
        raise AssertionError("unable to find a one-to-one structure-piece assignment")
    return assign


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fence's side-opening notch with the mirrored outside piece that matches its jagged side profile.

    Input structure:
    - Background is black.
    - Several monochrome fence-like rectangles appear as one-pixel outer borders with exactly one continuous gap on one side.
    - Each fence also contains a same-color inner partition that turns that side gap into a single open notch, while leaving a separate enclosed interior hole.
    - Outside the fences are monochrome non-rectangular pieces of various colors.

    Rule:
    - Detect each rectangular fence and extract the background notch that touches exactly one bbox side through the border gap.
    - Read the notch's orientation (top, bottom, left, or right) and its jagged profile along the opening.
    - Among the outside pieces, find the unique piece whose profile on the opposite side matches that notch; mirror it across the notch normal and align it with the notch.
    - Repaint the mirrored inserted shape with the fence color and fill the notch, while keeping the original outside piece unchanged.

    Key insight:
    - The border gap only reveals which side is open; the real matching key is the notch's side-profile shape, and the inserted copy must be mirrored before recoloring to the fence's color.
    """
    g = np.asarray(grid, dtype=int)
    structures, pieces, bg = _extract_objects(g)
    assignment = _match_structures(structures, pieces)

    out = g.copy()
    for si, structure in enumerate(structures):
        _matched_piece = pieces[assignment[si]]
        r0, c0, _r1, _c1 = structure["bbox"]
        for rr, cc in structure["socket_coords_local"]:
            out[r0 + rr, c0 + cc] = int(structure["color"])
    return out


def _build_socket_mask(depths: tuple[int, ...], side: str) -> np.ndarray:
    max_depth = max(depths)
    if side == "top":
        mask = np.zeros((max_depth, len(depths)), dtype=bool)
        for c, d in enumerate(depths):
            mask[:d, c] = True
        return mask
    if side == "bottom":
        mask = np.zeros((max_depth, len(depths)), dtype=bool)
        for c, d in enumerate(depths):
            mask[max_depth - d :, c] = True
        return mask
    if side == "left":
        mask = np.zeros((len(depths), max_depth), dtype=bool)
        for r, d in enumerate(depths):
            mask[r, :d] = True
        return mask
    mask = np.zeros((len(depths), max_depth), dtype=bool)
    for r, d in enumerate(depths):
        mask[r, max_depth - d :] = True
    return mask


def _build_structure_spec(side: str, depths: tuple[int, ...], color: int, piece_color: int, rng: random.Random) -> dict:
    socket_mask = _build_socket_mask(depths, side)
    max_depth = max(depths)
    gap_span = len(depths)

    if side in ("top", "bottom"):
        left_margin = rng.randint(*MARGIN_RANGE)
        right_margin = rng.randint(*MARGIN_RANGE)
        hole_h = rng.randint(*HOLE_SPAN_RANGE)
        outer_h = max_depth + hole_h + 2
        outer_w = gap_span + left_margin + right_margin + 2
        patch = np.zeros((outer_h, outer_w), dtype=int)
        patch[0, :] = int(color)
        patch[-1, :] = int(color)
        patch[:, 0] = int(color)
        patch[:, -1] = int(color)
        gap_start = 1 + left_margin
        if side == "top":
            patch[1 : max_depth + 1, 1:-1] = int(color)
            origin = (0, gap_start)
        else:
            origin_row = outer_h - max_depth
            patch[origin_row - 1 : outer_h - 1, 1:-1] = int(color)
            origin = (origin_row, gap_start)
    else:
        top_margin = rng.randint(*MARGIN_RANGE)
        bottom_margin = rng.randint(*MARGIN_RANGE)
        hole_w = rng.randint(*HOLE_SPAN_RANGE)
        outer_h = gap_span + top_margin + bottom_margin + 2
        outer_w = max_depth + hole_w + 2
        patch = np.zeros((outer_h, outer_w), dtype=int)
        patch[0, :] = int(color)
        patch[-1, :] = int(color)
        patch[:, 0] = int(color)
        patch[:, -1] = int(color)
        gap_start = 1 + top_margin
        if side == "left":
            patch[1:-1, 1 : max_depth + 1] = int(color)
            origin = (gap_start, 0)
        else:
            origin_col = outer_w - max_depth
            patch[1:-1, origin_col - 1 : outer_w - 1] = int(color)
            origin = (gap_start, origin_col)

    socket_coords_local: list[tuple[int, int]] = []
    orow, ocol = origin
    for rr in range(socket_mask.shape[0]):
        for cc in range(socket_mask.shape[1]):
            if not bool(socket_mask[rr, cc]):
                continue
            patch[orow + rr, ocol + cc] = 0
            socket_coords_local.append((int(orow + rr), int(ocol + cc)))

    piece_mask = _mirror_mask(socket_mask, side)
    piece_patch = np.zeros(piece_mask.shape, dtype=int)
    piece_patch[piece_mask] = int(piece_color)

    if int(piece_mask.sum()) == int(piece_mask.size):
        raise AssertionError("piece must stay non-rectangular")

    return {
        "side": side,
        "profile": depths,
        "structure_color": int(color),
        "piece_color": int(piece_color),
        "structure_patch": patch,
        "piece_patch": piece_patch,
        "socket_coords_local": socket_coords_local,
    }


def _random_profile(side: str, rng: random.Random, used: set[tuple[str, tuple[int, ...]]]) -> tuple[int, ...]:
    for _ in range(200):
        gap_span = rng.randint(*GAP_SPAN_RANGE)
        max_depth = rng.choice(DEPTH_MAX_CHOICES)
        depths = [rng.randint(2, max_depth) for _ in range(gap_span)]
        if len(set(depths)) < 2:
            continue
        if max(depths) != max_depth:
            continue
        key = (side, tuple(depths))
        if key in used:
            continue
        used.add(key)
        return tuple(int(x) for x in depths)
    raise RuntimeError("failed to sample a unique profile")


def _random_place(occupied: np.ndarray, shape: tuple[int, int], padding: int, rng: random.Random) -> tuple[int, int] | None:
    h, w = occupied.shape
    ph, pw = shape
    if ph > h or pw > w:
        return None
    for _ in range(PLACE_RETRIES):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        r0 = max(0, top - padding)
        c0 = max(0, left - padding)
        r1 = min(h, top + ph + padding)
        c1 = min(w, left + pw + padding)
        if bool(occupied[r0:r1, c0:c1].any()):
            continue
        return int(top), int(left)
    return None


def _apply_specs_to_grid(height: int, width: int, specs: list[dict], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    inp = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    structure_order = sorted(range(len(specs)), key=lambda i: specs[i]["structure_patch"].size, reverse=True)
    structure_locs: dict[int, tuple[int, int]] = {}
    for idx in structure_order:
        patch = specs[idx]["structure_patch"]
        loc = _random_place(occupied, patch.shape, PADDING, rng)
        if loc is None:
            return None
        top, left = loc
        structure_locs[idx] = (int(top), int(left))
        _paste(inp, patch, top, left, bg)
        _paste(expected, patch, top, left, bg)
        occupied[top : top + patch.shape[0], left : left + patch.shape[1]] = True

    piece_order = sorted(range(len(specs)), key=lambda i: specs[i]["piece_patch"].size, reverse=True)
    for idx in piece_order:
        patch = specs[idx]["piece_patch"]
        loc = _random_place(occupied, patch.shape, PADDING, rng)
        if loc is None:
            return None
        top, left = loc
        _paste(inp, patch, top, left, bg)
        _paste(expected, patch, top, left, bg)
        occupied[top : top + patch.shape[0], left : left + patch.shape[1]] = True

    for idx, spec in enumerate(specs):
        top, left = structure_locs[idx]
        for rr, cc in spec["socket_coords_local"]:
            expected[top + rr, left + cc] = int(spec["structure_color"])

    return inp, expected


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    count = rng.choice(NUM_STRUCTURES_CHOICES)
    sides = rng.sample(list(SIDES), k=count)
    colors = rng.sample(PALETTE, k=2 * count)
    structure_colors = colors[:count]
    piece_colors = colors[count:]

    used_profiles: set[tuple[str, tuple[int, ...]]] = set()
    specs: list[dict] = []
    for i, side in enumerate(sides):
        depths = _random_profile(side, rng, used_profiles)
        spec = _build_structure_spec(side, depths, structure_colors[i], piece_colors[i], rng)
        specs.append(spec)

    placed = _apply_specs_to_grid(height, width, specs, rng)
    if placed is None:
        return None
    inp, expected = placed

    try:
        out = main(inp)
    except AssertionError:
        return None
    if not bool(np.array_equal(out, expected)):
        return None

    structures, pieces, _bg = _extract_objects(inp)
    if len(structures) != count or len(pieces) != count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
