from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME_MIN_SIZE = 6
FRAME_BAND = 3
GEN_RETRIES = 200
PLACEMENT_PAD = 1


SHAPE_LIBRARY: dict[str, np.ndarray] = {
    "bar2": np.array([[1, 1]], dtype=bool),
    "bar3": np.array([[1, 1, 1]], dtype=bool),
    "bar4": np.array([[1, 1, 1, 1]], dtype=bool),
    "L3": np.array([[1, 1], [1, 0]], dtype=bool),
    "L4": np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    "T4": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
}
LINE_SHAPES = ["bar2", "bar3", "bar4"]
L_SHAPES = ["L3", "L4"]
ALL_SHAPES = list(SHAPE_LIBRARY)
SIDE_TO_ROT = {"top": 0, "left": 1, "bottom": 2, "right": 3}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _tight(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _components_bool(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox(pts)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": pts,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": int(mask.sum()),
                }
            )
    return comps


def _shape_key(mask: np.ndarray) -> tuple:
    base = _tight(mask)
    keys = []
    for k in range(4):
        rot = _tight(np.rot90(base, k))
        keys.append((rot.shape[0], rot.shape[1], rot.tobytes()))
    return min(keys)


def _unique_rotations(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rot = _tight(np.rot90(mask, k))
        key = (rot.shape[0], rot.shape[1], rot.tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append(rot)
    return out


def _frame_info_from_component(comp: dict) -> dict | None:
    mask = np.array(comp["mask"], dtype=bool)
    h, w = map(int, mask.shape)
    if h < FRAME_MIN_SIZE or w < FRAME_MIN_SIZE:
        return None

    hole_infos = []
    for coords in _components_bool(~mask):
        if _touches_edge(coords, h, w):
            continue
        r0, c0, r1, c1 = _bbox(coords)
        area = int(coords.shape[0])
        if area != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        hole_infos.append({"coords": coords, "bbox": (r0, c0, r1, c1), "area": area})
    if not hole_infos:
        return None

    hole = max(hole_infos, key=lambda d: (int(d["area"]), d["bbox"]))
    hr0, hc0, hr1, hc1 = hole["bbox"]
    off_top = int(hr0)
    off_left = int(hc0)
    off_bottom = int(h - 1 - hr1)
    off_right = int(w - 1 - hc1)
    if not (off_top == off_left == off_bottom == off_right and off_top >= 1):
        return None

    thickness = int(off_top)
    if h - 2 * thickness < 2 or w - 2 * thickness < 2:
        return None

    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[thickness : h - thickness, thickness : w - thickness] = False
    if np.any(mask & ~ring_mask):
        return None

    gap_mask = ring_mask & ~mask
    gap_area = int(gap_mask.sum())
    if not (2 <= gap_area <= 4):
        return None
    if len(_components_bool(gap_mask)) != 1:
        return None

    sides = []
    if bool(gap_mask[0, :].any()):
        sides.append("top")
    if bool(gap_mask[-1, :].any()):
        sides.append("bottom")
    if bool(gap_mask[:, 0].any()):
        sides.append("left")
    if bool(gap_mask[:, -1].any()):
        sides.append("right")
    if len(sides) != 1:
        return None

    side = sides[0]
    if side == "top":
        edge_idx = np.where(gap_mask[0, :])[0]
    elif side == "bottom":
        edge_idx = np.where(gap_mask[-1, :])[0]
    elif side == "left":
        edge_idx = np.where(gap_mask[:, 0])[0]
    else:
        edge_idx = np.where(gap_mask[:, -1])[0]
    if edge_idx.size < 2 or edge_idx.size > 4:
        return None
    if int(edge_idx[-1] - edge_idx[0] + 1) != int(edge_idx.size):
        return None

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": comp["bbox"],
        "mask": mask,
        "ring_mask": ring_mask,
        "gap_mask": gap_mask,
        "gap_area": gap_area,
        "gap_key": _shape_key(gap_mask),
        "side": side,
    }


def _piece_info_from_component(comp: dict) -> dict | None:
    area = int(comp["area"])
    if not (2 <= area <= 4):
        return None
    mask = np.array(comp["mask"], dtype=bool)
    if len(_components_bool(mask)) != 1:
        return None
    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": comp["bbox"],
        "mask": mask,
        "area": area,
        "shape_key": _shape_key(mask),
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    frames: list[dict] = []
    pieces: list[dict] = []
    for comp in _find_color_components(grid, bg):
        frame = _frame_info_from_component(comp)
        if frame is not None:
            frames.append(frame)
            continue
        piece = _piece_info_from_component(comp)
        if piece is not None:
            pieces.append(piece)
            continue
        raise AssertionError(f"unrecognized component at bbox {comp['bbox']}")

    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "frame/piece count mismatch"
    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    pieces.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    return bg, frames, pieces


def _match_frames_to_pieces(frames: list[dict], pieces: list[dict]) -> tuple[list[int], int]:
    assert len(frames) == len(pieces)
    options: list[list[int]] = []
    for frame in frames:
        matches = [j for j, piece in enumerate(pieces) if frame["gap_key"] == piece["shape_key"]]
        assert matches, f"no matching piece for frame at {frame['bbox']}"
        options.append(matches)

    order = sorted(range(len(frames)), key=lambda i: (len(options[i]), frames[i]["bbox"]))
    used: set[int] = set()
    current: dict[int, int] = {}
    first_solution: dict[int, int] | None = None
    solution_count = 0

    def dfs(depth: int) -> None:
        nonlocal first_solution, solution_count
        if solution_count >= 2:
            return
        if depth == len(order):
            solution_count += 1
            if first_solution is None:
                first_solution = dict(current)
            return
        frame_idx = order[depth]
        for piece_idx in options[frame_idx]:
            if piece_idx in used:
                continue
            used.add(piece_idx)
            current[frame_idx] = piece_idx
            dfs(depth + 1)
            used.remove(piece_idx)
            current.pop(frame_idx, None)

    dfs(0)
    assert first_solution is not None, "no complete frame-piece assignment"
    return [int(first_solution[i]) for i in range(len(frames))], int(solution_count)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered polyomino to the side notch of a hollow frame, then repaint that whole frame with the piece color.

    Input structure:
    - The background is black.
    - Several thin monochrome hollow rectangular frames are scattered on the grid.
    - Each frame has exactly one small connected side notch carved out of its border band; the notch is a 2- to 4-cell bar, L, or T shape.
    - Elsewhere lie several small solid monochrome polyomino pieces, one per frame, with colors distinct from the frame colors.

    Rule:
    - For each frame, recover the missing notch shape by comparing the observed border to the full rectangular border around its hollow interior.
    - Match every piece to the unique frame whose notch has the same shape up to rotation.
    - Fill that notch and recolor the entire frame border band to the matched piece color.
    - Remove the piece from its original location; the output grid keeps the same size.

    Key insight: Shape drives the pairing, but color flows in the opposite direction: the absorbed piece does not keep its place, it transfers its color to the frame it repairs.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, pieces = _extract_scene(g)
    assignment, n_solutions = _match_frames_to_pieces(frames, pieces)
    assert n_solutions == 1, "shape matching must be unambiguous"

    out = g.copy()
    for piece in pieces:
        pts = piece["coords"]
        out[pts[:, 0], pts[:, 1]] = int(bg)

    for frame, piece_idx in zip(frames, assignment):
        piece_color = int(pieces[piece_idx]["color"])
        r0, c0, r1, c1 = frame["bbox"]
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[frame["ring_mask"]] = piece_color

    return out


def _orient_shape_to_side(shape_name: str, side: str) -> np.ndarray:
    return _tight(np.rot90(SHAPE_LIBRARY[shape_name], SIDE_TO_ROT[side]))


def _build_frame_patch(frame_color: int, shape_name: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    gap_local = _orient_shape_to_side(shape_name, side)
    gh, gw = map(int, gap_local.shape)
    min_side = max(FRAME_MIN_SIZE, 2 * FRAME_BAND + 2, gh + 2, gw + 2)
    h = random.randint(min_side, 11)
    w = random.randint(min_side, 11)

    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[FRAME_BAND : h - FRAME_BAND, FRAME_BAND : w - FRAME_BAND] = False
    gap_mask = np.zeros((h, w), dtype=bool)

    if side in ("top", "bottom"):
        start_c = random.randint(1, w - gw - 1)
        start_r = 0 if side == "top" else h - gh
    else:
        start_r = random.randint(1, h - gh - 1)
        start_c = 0 if side == "left" else w - gw

    gap_mask[start_r : start_r + gh, start_c : start_c + gw] = gap_local
    assert np.all(gap_mask <= ring_mask)

    frame_mask = ring_mask & ~gap_mask
    assert len(_components_bool(frame_mask)) == 1

    patch = np.zeros((h, w), dtype=int)
    patch[frame_mask] = int(frame_color)
    return patch, ring_mask


def _build_piece_patch(piece_color: int, shape_name: str) -> tuple[np.ndarray, np.ndarray]:
    mask = random.choice(_unique_rotations(SHAPE_LIBRARY[shape_name]))
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(piece_color)
    return patch, mask


def _reserve_box(
    reserve: np.ndarray, top: int, left: int, h: int, w: int, pad: int
) -> tuple[int, int, int, int]:
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(reserve.shape[0], top + h + pad)
    cc1 = min(reserve.shape[1], left + w + pad)
    return rr0, cc0, rr1, cc1


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, patch.shape)
    sub = grid[top : top + h, left : left + w]
    mask = patch != int(BG)
    sub[mask] = patch[mask]


def _place_patch_random(
    grid: np.ndarray, reserve: np.ndarray, patch: np.ndarray
) -> tuple[int, int] | None:
    gh, gw = map(int, grid.shape)
    ph, pw = map(int, patch.shape)
    for _ in range(500):
        top = random.randint(0, gh - ph)
        left = random.randint(0, gw - pw)
        rr0, cc0, rr1, cc1 = _reserve_box(reserve, top, left, ph, pw, PLACEMENT_PAD)
        if bool(reserve[rr0:rr1, cc0:cc1].any()):
            continue
        _paste_patch(grid, patch, top, left)
        reserve[rr0:rr1, cc0:cc1] = True
        return int(top), int(left)
    return None


def _choose_shape_names(n_pairs: int) -> list[str]:
    chosen: list[str] = []
    if n_pairs >= 3:
        chosen.append(random.choice(LINE_SHAPES))
        chosen.append(random.choice(L_SHAPES))
        chosen.append("T4")
    else:
        cats = random.sample(["line", "L", "T"], n_pairs)
        for cat in cats:
            if cat == "line":
                chosen.append(random.choice(LINE_SHAPES))
            elif cat == "L":
                chosen.append(random.choice(L_SHAPES))
            else:
                chosen.append("T4")

    remaining = [name for name in ALL_SHAPES if name not in chosen]
    while len(chosen) < n_pairs:
        pick = random.choice(remaining)
        remaining.remove(pick)
        chosen.append(pick)
    random.shuffle(chosen)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice([2, 3, 3, 4])
    shape_names = _choose_shape_names(n_pairs)
    palette = random.sample(range(1, 10), 2 * n_pairs)
    frame_colors = palette[:n_pairs]
    piece_colors = palette[n_pairs:]

    frame_records = []
    piece_records = []
    for idx, shape_name in enumerate(shape_names):
        side = random.choice(["top", "bottom", "left", "right"])
        frame_patch, ring_mask = _build_frame_patch(frame_colors[idx], shape_name, side)
        piece_patch, piece_mask = _build_piece_patch(piece_colors[idx], shape_name)
        frame_records.append(
            {
                "pair_idx": idx,
                "patch": frame_patch,
                "ring_mask": ring_mask,
                "color": frame_colors[idx],
            }
        )
        piece_records.append(
            {
                "pair_idx": idx,
                "patch": piece_patch,
                "mask": piece_mask,
                "color": piece_colors[idx],
            }
        )

    all_patches = [rec["patch"] for rec in frame_records] + [rec["patch"] for rec in piece_records]
    max_h = max(int(p.shape[0]) for p in all_patches) + 2 * PLACEMENT_PAD
    max_w = max(int(p.shape[1]) for p in all_patches) + 2 * PLACEMENT_PAD
    area_need = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in all_patches)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(area_need * 1.45))))
    base = max(base, 18 if n_pairs <= 2 else 20)
    if base > 30:
        return None

    grid = np.zeros((random.randint(base, 30), random.randint(base, 30)), dtype=int)
    reserve = np.zeros_like(grid, dtype=bool)

    objects = [("frame", i, frame_records[i]["patch"]) for i in range(n_pairs)]
    objects += [("piece", i, piece_records[i]["patch"]) for i in range(n_pairs)]
    objects.sort(key=lambda item: int(item[2].shape[0]) * int(item[2].shape[1]), reverse=True)

    for kind, idx, patch in objects:
        pos = _place_patch_random(grid, reserve, patch)
        if pos is None:
            return None
        if kind == "frame":
            frame_records[idx]["origin"] = pos
        else:
            piece_records[idx]["origin"] = pos

    expected = grid.copy()
    for rec in piece_records:
        pr, pc = rec["origin"]
        ph, pw = map(int, rec["patch"].shape)
        sub = expected[pr : pr + ph, pc : pc + pw]
        sub[rec["mask"]] = int(BG)
    for rec in frame_records:
        fr, fc = rec["origin"]
        fh, fw = map(int, rec["patch"].shape)
        sub = expected[fr : fr + fh, fc : fc + fw]
        sub[rec["ring_mask"]] = int(piece_records[rec["pair_idx"]]["color"])

    try:
        _bg, frames, pieces = _extract_scene(grid)
        _assignment, solution_count = _match_frames_to_pieces(frames, pieces)
        if len(frames) != n_pairs or len(pieces) != n_pairs or solution_count != 1:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
