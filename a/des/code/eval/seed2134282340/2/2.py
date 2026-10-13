from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _background_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: Iterable[tuple[int, int]]) -> np.ndarray:
    coords = list(coords)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = _normalize_mask(np.rot90(mask, k))
        key = (rot.shape, rot.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        variants.append(rot)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    keys = []
    for variant in _shape_variants(mask):
        keys.append(
            (int(variant.shape[0]), int(variant.shape[1]), variant.astype(np.uint8).tobytes())
        )
    return min(keys)


def _same_shape_up_to_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    for variant in _shape_variants(a):
        if variant.shape == b.shape and np.array_equal(variant, b):
            return True
    return False


def _connected(coords: set[tuple[int, int]]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            queue = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            qi = 0
            while qi < len(queue):
                cr, cc = queue[qi]
                qi += 1
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            rel = [(rr - r0, cc - c0) for rr, cc in coords]
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in rel:
                mask[rr, cc] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "rel_coords": rel,
                    "area": len(coords),
                }
            )
    return comps


def _boundary_cells(h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _frame_candidates(comp: dict, grid: np.ndarray, bg: int) -> list[dict]:
    base_r0, base_c0, base_r1, base_c1 = comp["bbox"]
    color = int(comp["color"])
    h_grid, w_grid = grid.shape
    abs_coords = set((int(r), int(c)) for r, c in comp["coords"])

    candidates: list[dict] = []
    seen = set()
    for r0 in [base_r0, base_r0 - 1]:
        for c0 in [base_c0, base_c0 - 1]:
            for r1 in [base_r1, base_r1 + 1]:
                for c1 in [base_c1, base_c1 + 1]:
                    if r0 < 0 or c0 < 0 or r1 >= h_grid or c1 >= w_grid:
                        continue
                    h = int(r1 - r0 + 1)
                    w = int(c1 - c0 + 1)
                    if h < 4 or w < 4:
                        continue

                    boundary = _boundary_cells(h, w)
                    rel_set = set((r - r0, c - c0) for r, c in abs_coords)
                    if any(cell not in boundary for cell in rel_set):
                        continue

                    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
                    valid = True
                    for rr in range(h):
                        for cc in range(w):
                            cell = (rr, cc)
                            value = int(sub[rr, cc])
                            if cell in rel_set:
                                if value != color:
                                    valid = False
                                    break
                            elif value != bg:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue

                    gap = boundary - rel_set
                    if not (2 <= len(gap) <= 5):
                        continue
                    if not _connected(gap):
                        continue

                    gap_mask = np.zeros((h, w), dtype=bool)
                    for rr, cc in gap:
                        gap_mask[rr, cc] = True
                    gap_shape = _normalize_mask(gap_mask)
                    key = ((int(r0), int(c0), int(r1), int(c1)), _shape_key(gap_shape))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "color": color,
                            "bbox": (int(r0), int(c0), int(r1), int(c1)),
                            "gap_mask": gap_mask,
                            "gap_shape": gap_shape,
                        }
                    )

    candidates.sort(
        key=lambda d: (
            -((d["bbox"][2] - d["bbox"][0] + 1) * (d["bbox"][3] - d["bbox"][1] + 1)),
            -int(d["gap_mask"].sum()),
            d["bbox"],
        )
    )
    return candidates


def _solve_assignment(pieces: list[dict], frames: list[dict]) -> list[int] | None:
    candidates: list[list[int]] = []
    for piece in pieces:
        opts = []
        for fi, frame in enumerate(frames):
            if _same_shape_up_to_rotation(piece["shape"], frame["gap_shape"]):
                opts.append(fi)
        if not opts:
            return None
        candidates.append(opts)

    order = sorted(range(len(pieces)), key=lambda i: (len(candidates[i]), pieces[i]["area"]))
    assignment = [-1] * len(pieces)
    used: set[int] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        pi = order[pos]
        for fi in candidates[pi]:
            if fi in used:
                continue
            used.add(fi)
            assignment[pi] = fi
            if dfs(pos + 1):
                return True
            assignment[pi] = -1
            used.remove(fi)
        return False

    if not dfs(0):
        return None
    return assignment


def _solve_layout(pieces: list[dict], frame_options: list[list[dict]]) -> tuple[list[dict], list[int]]:
    order = sorted(range(len(frame_options)), key=lambda i: len(frame_options[i]))
    chosen: list[dict | None] = [None] * len(frame_options)

    def dfs(pos: int) -> tuple[list[dict], list[int]] | None:
        if pos == len(order):
            frames = [frame for frame in chosen if frame is not None]
            assignment = _solve_assignment(pieces, frames)
            if assignment is None:
                return None
            return frames, assignment

        ci = order[pos]
        for option in frame_options[ci]:
            chosen[ci] = option
            result = dfs(pos + 1)
            if result is not None:
                return result
        chosen[ci] = None
        return None

    result = dfs(0)
    assert result is not None, "piece-frame matching must be unique and solvable"
    return result


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    comps = _components(g, bg)

    frame_options: list[list[dict]] = []
    pieces: list[dict] = []
    for comp in comps:
        options = _frame_candidates(comp, g, bg)
        if options:
            frame_options.append(options)
        else:
            pieces.append(
                {
                    "color": int(comp["color"]),
                    "coords": comp["coords"],
                    "shape": _normalize_mask(comp["mask"]),
                    "area": int(comp["area"]),
                }
            )

    assert frame_options, "at least one frame is required"
    assert len(frame_options) == len(pieces), "frames and pieces must pair one-to-one"

    frames, assignment = _solve_layout(pieces, frame_options)
    out = g.copy()

    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg

    for piece, frame_index in zip(pieces, assignment):
        frame = frames[frame_index]
        r0, c0, r1, c1 = frame["bbox"]
        gap_mask = frame["gap_mask"]
        for rr, cc in np.argwhere(gap_mask):
            out[r0 + int(rr), c0 + int(cc)] = int(frame["color"])
        out[r0 + 1 : r1, c0 + 1 : c1] = int(piece["color"])

    return out


SHAPE_SPECS = [
    {"name": "line2", "kind": "line", "length": 2},
    {"name": "line3", "kind": "line", "length": 3},
    {"name": "line4", "kind": "line", "length": 4},
    {"name": "line5", "kind": "line", "length": 5},
    {"name": "corner3", "kind": "corner", "arms": (2, 2)},
    {"name": "corner4", "kind": "corner", "arms": (2, 3)},
    {"name": "corner5a", "kind": "corner", "arms": (2, 4)},
    {"name": "corner5b", "kind": "corner", "arms": (3, 3)},
]

GRID_MIN = 18
GRID_MAX = 30
FRAME_SIDE_MIN = 4
FRAME_SIDE_MAX = 8
FRAME_PAD = 2
PIECE_PAD = 1
OUTER_ATTEMPTS = 400


def _full_border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _sample_gap_mask(spec: dict, h: int, w: int, rng: random.Random) -> np.ndarray | None:
    gap = np.zeros((h, w), dtype=bool)
    if spec["kind"] == "line":
        length = int(spec["length"])
        options = []
        if w >= length:
            options.extend(["top", "bottom"])
        if h >= length:
            options.extend(["left", "right"])
        if not options:
            return None
        side = rng.choice(options)
        if side == "top":
            start = rng.randint(0, w - length)
            gap[0, start : start + length] = True
        elif side == "bottom":
            start = rng.randint(0, w - length)
            gap[h - 1, start : start + length] = True
        elif side == "left":
            start = rng.randint(0, h - length)
            gap[start : start + length, 0] = True
        else:
            start = rng.randint(0, h - length)
            gap[start : start + length, w - 1] = True
        return gap

    arm_a, arm_b = spec["arms"]
    if rng.random() < 0.5:
        horiz, vert = int(arm_a), int(arm_b)
    else:
        horiz, vert = int(arm_b), int(arm_a)
    if w < horiz or h < vert:
        return None

    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        gap[0, :horiz] = True
        gap[:vert, 0] = True
    elif corner == "tr":
        gap[0, w - horiz : w] = True
        gap[:vert, w - 1] = True
    elif corner == "bl":
        gap[h - 1, :horiz] = True
        gap[h - vert : h, 0] = True
    else:
        gap[h - 1, w - horiz : w] = True
        gap[h - vert : h, w - 1] = True
    return gap


def _build_frame_and_piece(spec: dict, frame_color: int, piece_color: int, rng: random.Random) -> dict | None:
    need = int(spec.get("length", max(spec.get("arms", (0, 0)))))
    lo = max(FRAME_SIDE_MIN, need)
    hi = FRAME_SIDE_MAX
    if lo > hi:
        return None

    for _ in range(40):
        h = rng.randint(lo, hi)
        w = rng.randint(lo, hi)
        gap_mask = _sample_gap_mask(spec, h, w, rng)
        if gap_mask is None:
            continue
        border = _full_border_mask(h, w)
        frame_patch = np.zeros((h, w), dtype=int)
        frame_patch[border & (~gap_mask)] = int(frame_color)

        piece_shape = _normalize_mask(gap_mask)
        piece_shape = _normalize_mask(np.rot90(piece_shape, rng.randrange(4)))
        piece_patch = np.zeros(piece_shape.shape, dtype=int)
        piece_patch[piece_shape] = int(piece_color)

        return {
            "spec": spec,
            "frame_color": int(frame_color),
            "piece_color": int(piece_color),
            "frame_patch": frame_patch,
            "gap_mask": gap_mask,
            "piece_patch": piece_patch,
            "piece_shape": piece_shape,
        }
    return None


def _can_place_rect(
    reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int
) -> bool:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(reserved[r0:r1, c0:c1].any())


def _mark_rect(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    hh, ww = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _place_items(
    reserved: np.ndarray,
    patches: list[np.ndarray],
    pad: int,
    border: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    hh, ww = reserved.shape
    order = sorted(
        range(len(patches)),
        key=lambda i: int(patches[i].shape[0]) * int(patches[i].shape[1]),
        reverse=True,
    )
    positions: list[tuple[int, int] | None] = [None] * len(patches)

    for idx in order:
        ph, pw = map(int, patches[idx].shape)
        if hh < ph + 2 * border or ww < pw + 2 * border:
            return None
        candidates = [
            (top, left)
            for top in range(border, hh - ph - border + 1)
            for left in range(border, ww - pw - border + 1)
        ]
        rng.shuffle(candidates)
        for top, left in candidates:
            if _can_place_rect(reserved, top, left, ph, pw, pad):
                positions[idx] = (top, left)
                _mark_rect(reserved, top, left, ph, pw, pad)
                break
        if positions[idx] is None:
            return None

    return [(int(top), int(left)) for top, left in positions]  # type: ignore[misc]


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != 0
    sub = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = patch[mask]


def _build_expected(
    inp: np.ndarray, pairs: list[dict], frame_positions: list[tuple[int, int]], piece_positions: list[tuple[int, int]]
) -> np.ndarray:
    out = inp.copy()
    for pair, (ptop, pleft) in zip(pairs, piece_positions):
        piece_patch = pair["piece_patch"]
        mask = piece_patch != 0
        sub = out[ptop : ptop + piece_patch.shape[0], pleft : pleft + piece_patch.shape[1]]
        sub[mask] = 0

    for pair, (ftop, fleft) in zip(pairs, frame_positions):
        gap_mask = pair["gap_mask"]
        border_color = int(pair["frame_color"])
        piece_color = int(pair["piece_color"])
        fh, fw = pair["frame_patch"].shape
        for rr, cc in np.argwhere(gap_mask):
            out[ftop + int(rr), fleft + int(cc)] = border_color
        out[ftop + 1 : ftop + fh - 1, fleft + 1 : fleft + fw - 1] = piece_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    num_pairs = rng.choice([2, 3, 4])
    shape_specs = rng.sample(SHAPE_SPECS, num_pairs)
    colors = rng.sample(list(range(1, 10)), 2 * num_pairs)
    frame_colors = colors[:num_pairs]
    piece_colors = colors[num_pairs:]

    pairs: list[dict] = []
    for spec, frame_color, piece_color in zip(shape_specs, frame_colors, piece_colors):
        pair = _build_frame_and_piece(spec, frame_color, piece_color, rng)
        if pair is None:
            return None
        pairs.append(pair)

    shape_keys = [_shape_key(pair["piece_shape"]) for pair in pairs]
    if len(set(shape_keys)) != num_pairs:
        return None

    base = max(GRID_MIN, 12 + 4 * num_pairs)
    if base > GRID_MAX:
        return None
    height = rng.randint(base, GRID_MAX)
    width = rng.randint(base, GRID_MAX)

    reserved = np.zeros((height, width), dtype=bool)
    frame_patches = [pair["frame_patch"] for pair in pairs]
    piece_patches = [pair["piece_patch"] for pair in pairs]

    frame_positions = _place_items(reserved, frame_patches, FRAME_PAD, 1, rng)
    if frame_positions is None:
        return None
    piece_positions = _place_items(reserved, piece_patches, PIECE_PAD, 1, rng)
    if piece_positions is None:
        return None

    inp = np.zeros((height, width), dtype=int)
    for pair, (top, left) in zip(pairs, frame_positions):
        _draw_patch(inp, pair["frame_patch"], top, left)
    for pair, (top, left) in zip(pairs, piece_positions):
        _draw_patch(inp, pair["piece_patch"], top, left)

    expected = _build_expected(inp, pairs, frame_positions, piece_positions)
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(OUTER_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {OUTER_ATTEMPTS} attempts")
