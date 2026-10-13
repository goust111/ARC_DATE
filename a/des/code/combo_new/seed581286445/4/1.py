import numpy as np


_RNG = np.random.default_rng()
_SIDES = ("top", "bottom", "left", "right")
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_GEN_RETRIES = 500
_MASK_RETRIES = 120


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
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
                for dr, dc in _DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    return _mask_components(grid == int(color))


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if mask[1:-1, 1:-1].any():
        return False
    if not (mask[0, 0] and mask[0, -1] and mask[-1, 0] and mask[-1, -1]):
        return False
    if mask[0].sum() < 3 or mask[-1].sum() < 3 or mask[:, 0].sum() < 3 or mask[:, -1].sum() < 3:
        return False
    return True


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        for coords in _color_components(grid, color):
            if not _is_frame_component(coords):
                continue
            bbox = _bbox_from_coords(coords)
            r0, c0, r1, c1 = bbox
            score = ((r1 - r0 + 1) * (c1 - c0 + 1), int(coords.shape[0]))
            cand = {"color": color, "coords": coords, "bbox": bbox}
            if best is None or score > best["score"]:
                best = {"score": score, "frame": cand}
    if best is None:
        raise ValueError("frame not found")
    return best["frame"]


def _runs_of_true(vals: np.ndarray, base: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(vals.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((base + start, base + idx - 1))
            start = None
    if start is not None:
        runs.append((base + start, base + len(vals) - 1))
    return runs


def _find_gaps(grid: np.ndarray, frame: dict) -> list[dict]:
    color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    gaps: list[dict] = []
    top_missing = grid[r0, c0 + 1 : c1] != color
    bottom_missing = grid[r1, c0 + 1 : c1] != color
    left_missing = grid[r0 + 1 : r1, c0] != color
    right_missing = grid[r0 + 1 : r1, c1] != color
    for start, end in _runs_of_true(top_missing, c0 + 1):
        gaps.append({"side": "top", "start": start, "end": end, "length": end - start + 1})
    for start, end in _runs_of_true(bottom_missing, c0 + 1):
        gaps.append({"side": "bottom", "start": start, "end": end, "length": end - start + 1})
    for start, end in _runs_of_true(left_missing, r0 + 1):
        gaps.append({"side": "left", "start": start, "end": end, "length": end - start + 1})
    for start, end in _runs_of_true(right_missing, r0 + 1):
        gaps.append({"side": "right", "start": start, "end": end, "length": end - start + 1})
    if not gaps:
        raise ValueError("no gaps found")
    return gaps


def _extract_inner_pieces(grid: np.ndarray, frame: dict, bg: int) -> list[dict]:
    r0, c0, r1, c1 = frame["bbox"]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    pieces: list[dict] = []
    for rel_coords in _mask_components(inner != int(bg)):
        abs_coords = rel_coords + np.array([r0 + 1, c0 + 1], dtype=int)
        pr0, pc0, pr1, pc1 = _bbox_from_coords(abs_coords)
        patch = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), int(bg), dtype=int)
        patch[abs_coords[:, 0] - pr0, abs_coords[:, 1] - pc0] = grid[abs_coords[:, 0], abs_coords[:, 1]]
        pieces.append(
            {
                "coords": abs_coords,
                "bbox": (pr0, pc0, pr1, pc1),
                "patch": patch,
                "height": pr1 - pr0 + 1,
                "width": pc1 - pc0 + 1,
            }
        )
    return pieces


def _piece_matches_gap(piece: dict, gap: dict) -> bool:
    if gap["side"] in ("top", "bottom"):
        return int(piece["width"]) == int(gap["length"])
    return int(piece["height"]) == int(gap["length"])


def _unique_assignment(gaps: list[dict], pieces: list[dict]) -> dict[int, int]:
    candidates = {gi: [pi for pi, piece in enumerate(pieces) if _piece_matches_gap(piece, gaps[gi])] for gi in range(len(gaps))}
    if any(not opts for opts in candidates.values()):
        raise ValueError("at least one gap has no candidate piece")
    order = sorted(range(len(gaps)), key=lambda gi: (len(candidates[gi]), gi))
    used: set[int] = set()
    chosen: dict[int, int] = {}
    solutions: list[dict[int, int]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            solutions.append(chosen.copy())
            return
        gi = order[pos]
        for pi in candidates[gi]:
            if pi in used:
                continue
            used.add(pi)
            chosen[gi] = pi
            dfs(pos + 1)
            used.remove(pi)
            del chosen[gi]

    dfs(0)
    if len(solutions) != 1:
        raise ValueError("piece-gap assignment is not unique")
    return solutions[0]


def _paste(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = out[top : top + h, left : left + w]
    mask = patch != int(bg)
    sub[mask] = patch[mask]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = _get_bg(g)
    frame = _find_frame(g, bg)
    gaps = _find_gaps(g, frame)
    pieces = _extract_inner_pieces(g, frame, bg)
    assignment = _unique_assignment(gaps, pieces)
    r0, c0, r1, c1 = frame["bbox"]

    out = g.copy()
    for pi in assignment.values():
        for rr, cc in pieces[pi]["coords"]:
            out[int(rr), int(cc)] = int(bg)

    for gi, pi in assignment.items():
        gap = gaps[gi]
        piece = pieces[pi]
        patch = piece["patch"]
        if gap["side"] in ("top", "bottom"):
            moved = np.fliplr(patch)
            dest_c = int(gap["start"])
            dest_r = int(r0 - moved.shape[0]) if gap["side"] == "top" else int(r1 + 1)
        else:
            moved = np.flipud(patch)
            dest_r = int(gap["start"])
            dest_c = int(c0 - moved.shape[1]) if gap["side"] == "left" else int(c1 + 1)
        if dest_r < 0 or dest_c < 0 or dest_r + moved.shape[0] > out.shape[0] or dest_c + moved.shape[1] > out.shape[1]:
            raise ValueError("ejected piece would leave the canvas")
        _paste(out, moved, dest_r, dest_c, bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject the size-matched inner fragment through each frame gap and draw its mirrored copy outside.

    Input structure:
    - The background is the most common color.
    - A single-color one-pixel rectangular frame surrounds several 4-connected inner fragments.
    - One or more frame sides contain a continuous gap cut out of the border, while the outside remains background.

    Rule:
    - Find every continuous border gap and record which side it lies on plus the gap length.
    - Inside the frame, match each gap to the unique fragment whose span parallel to that side equals the gap length.
    - Remove the matched fragment from the interior, then translate it outward so its nearest row or column sits immediately outside the frame along the gap.
    - Draw the translated fragment as a mirror image: top/bottom gaps use a left-right flip, left/right gaps use an up-down flip.

    Key insight:
    - The pairing is controlled by geometric span along the gap direction, while the gap side independently decides both the ejection direction and the mirror axis.
    """
    return _solve(grid)


def _grow_connected_mask(h: int, w: int, target: int) -> np.ndarray | None:
    cells = {(int(_RNG.integers(0, h)), int(_RNG.integers(0, w)))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in _DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        nr, nc = frontier[int(_RNG.integers(0, len(frontier)))]
        cells.add((nr, nc))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _touches_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _make_piece_mask(h: int, w: int, mirror_axis: str | None) -> np.ndarray | None:
    area_lo = max(2, int(np.ceil(0.4 * h * w)))
    area_hi = max(area_lo, h * w - 1)
    for _ in range(_MASK_RETRIES):
        target = int(_RNG.integers(area_lo, area_hi + 1))
        mask = _grow_connected_mask(h, w, target)
        if mask is None or not _touches_bbox(mask):
            continue
        if mirror_axis == "lr" and np.array_equal(mask, np.fliplr(mask)):
            continue
        if mirror_axis == "ud" and np.array_equal(mask, np.flipud(mask)):
            continue
        return mask
    return None


def _place_pieces(inner_h: int, inner_w: int, pieces: list[dict], bg: int) -> np.ndarray | None:
    canvas = np.full((inner_h, inner_w), int(bg), dtype=int)
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    order = sorted(range(len(pieces)), key=lambda idx: pieces[idx]["h"] * pieces[idx]["w"], reverse=True)
    for idx in order:
        piece = pieces[idx]
        h = int(piece["h"])
        w = int(piece["w"])
        if inner_h - h - 1 < 1 or inner_w - w - 1 < 1:
            return None
        spots = [(r, c) for r in range(1, inner_h - h) for c in range(1, inner_w - w)]
        _RNG.shuffle(spots)
        placed = False
        for r, c in spots:
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(inner_h, r + h + 1)
            cc1 = min(inner_w, c + w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            patch = canvas[r : r + h, c : c + w]
            patch[piece["mask"]] = int(piece["color"])
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return canvas


def _choose_lengths(count: int, max_len: int) -> list[int] | None:
    if count == 0:
        return []
    pool = list(range(2, max_len + 1))
    if len(pool) < count:
        return None
    _RNG.shuffle(pool)
    return [int(v) for v in pool[:count]]


def _gen_attempt() -> np.ndarray | None:
    side_mask = int(_RNG.integers(1, 16))
    sides = [side for bit, side in enumerate(_SIDES) if (side_mask >> bit) & 1]
    horiz_sides = [side for side in sides if side in ("top", "bottom")]
    vert_sides = [side for side in sides if side in ("left", "right")]

    inner_h = int(_RNG.integers(8, 15))
    inner_w = int(_RNG.integers(8, 15))
    h_lengths = _choose_lengths(len(horiz_sides), min(5, inner_w - 2))
    v_lengths = _choose_lengths(len(vert_sides), min(5, inner_h - 2))
    if h_lengths is None or v_lengths is None:
        return None

    gap_lengths: dict[str, int] = {}
    for side, length in zip(horiz_sides, h_lengths):
        gap_lengths[side] = int(length)
    for side, length in zip(vert_sides, v_lengths):
        gap_lengths[side] = int(length)

    pieces: list[dict] = []
    side_to_piece: dict[str, dict] = {}
    used_dims: set[tuple[int, int, bytes]] = set()

    for side in sides:
        if side in ("top", "bottom"):
            h_choices = [v for v in range(2, min(5, inner_h - 2) + 1) if v not in v_lengths]
            if not h_choices:
                return None
            h = int(h_choices[int(_RNG.integers(0, len(h_choices)))])
            w = int(gap_lengths[side])
            mirror_axis = "lr"
        else:
            w_choices = [v for v in range(2, min(5, inner_w - 2) + 1) if v not in h_lengths]
            if not w_choices:
                return None
            w = int(w_choices[int(_RNG.integers(0, len(w_choices)))])
            h = int(gap_lengths[side])
            mirror_axis = "ud"
        mask = _make_piece_mask(h, w, mirror_axis)
        if mask is None:
            return None
        key = (h, w, mask.tobytes())
        if key in used_dims:
            return None
        used_dims.add(key)
        piece = {"side": side, "h": h, "w": w, "mask": mask}
        side_to_piece[side] = piece
        pieces.append(piece)

    distractor_count = int(_RNG.integers(0, 3))
    for _ in range(distractor_count):
        h_pool = [v for v in range(2, min(5, inner_h - 2) + 1) if v not in v_lengths]
        w_pool = [v for v in range(2, min(5, inner_w - 2) + 1) if v not in h_lengths]
        if not h_pool or not w_pool:
            break
        h = int(h_pool[int(_RNG.integers(0, len(h_pool)))])
        w = int(w_pool[int(_RNG.integers(0, len(w_pool)))])
        if any(int(piece["h"]) == h and int(piece["w"]) == w for piece in pieces):
            continue
        mask = _make_piece_mask(h, w, None)
        if mask is None:
            continue
        pieces.append({"side": None, "h": h, "w": w, "mask": mask})

    palette = list(range(10))
    _RNG.shuffle(palette)
    need_colors = 2 + len(pieces)
    if need_colors > len(palette):
        return None
    bg = int(palette[0])
    frame_color = int(palette[1])
    for piece, color in zip(pieces, palette[2 : 2 + len(pieces)]):
        piece["color"] = int(color)

    inner = _place_pieces(inner_h, inner_w, pieces, bg)
    if inner is None:
        return None

    gap_starts: dict[str, int] = {}
    for side in horiz_sides:
        length = int(gap_lengths[side])
        starts = list(range(2, inner_w - length + 1))
        if not starts:
            return None
        gap_starts[side] = int(starts[int(_RNG.integers(0, len(starts)))])
    for side in vert_sides:
        length = int(gap_lengths[side])
        starts = list(range(2, inner_h - length + 1))
        if not starts:
            return None
        gap_starts[side] = int(starts[int(_RNG.integers(0, len(starts)))])

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    top_margin = int(side_to_piece["top"]["h"]) if "top" in side_to_piece else 0
    bottom_margin = int(side_to_piece["bottom"]["h"]) if "bottom" in side_to_piece else 0
    left_margin = int(side_to_piece["left"]["w"]) if "left" in side_to_piece else 0
    right_margin = int(side_to_piece["right"]["w"]) if "right" in side_to_piece else 0
    slack_top = int(_RNG.integers(0, 3))
    slack_bottom = int(_RNG.integers(0, 3))
    slack_left = int(_RNG.integers(0, 3))
    slack_right = int(_RNG.integers(0, 3))
    grid_h = top_margin + slack_top + outer_h + bottom_margin + slack_bottom
    grid_w = left_margin + slack_left + outer_w + right_margin + slack_right
    if grid_h > 30 or grid_w > 30:
        return None

    frame_r0 = top_margin + slack_top
    frame_c0 = left_margin + slack_left
    frame_r1 = frame_r0 + outer_h - 1
    frame_c1 = frame_c0 + outer_w - 1

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    grid[frame_r0, frame_c0 : frame_c1 + 1] = int(frame_color)
    grid[frame_r1, frame_c0 : frame_c1 + 1] = int(frame_color)
    grid[frame_r0 : frame_r1 + 1, frame_c0] = int(frame_color)
    grid[frame_r0 : frame_r1 + 1, frame_c1] = int(frame_color)

    for side in horiz_sides:
        start = frame_c0 + int(gap_starts[side])
        end = start + int(gap_lengths[side])
        row = frame_r0 if side == "top" else frame_r1
        grid[row, start:end] = int(bg)
    for side in vert_sides:
        start = frame_r0 + int(gap_starts[side])
        end = start + int(gap_lengths[side])
        col = frame_c0 if side == "left" else frame_c1
        grid[start:end, col] = int(bg)

    grid[frame_r0 + 1 : frame_r1, frame_c0 + 1 : frame_c1] = inner

    try:
        out = _solve(grid)
    except ValueError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
