import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400
CANVAS_MIN = 22
CANVAS_MAX = 30


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _build_piece_mask(
    base_h: int,
    base_w: int,
    side: str,
    kind: str,
    start: int,
    length: int,
) -> np.ndarray:
    if kind == "indentation":
        mask = np.ones((base_h, base_w), dtype=bool)
        if side == "left":
            mask[start : start + length, 0] = False
        elif side == "right":
            mask[start : start + length, base_w - 1] = False
        elif side == "top":
            mask[0, start : start + length] = False
        elif side == "bottom":
            mask[base_h - 1, start : start + length] = False
        else:
            raise AssertionError(f"unknown side {side}")
        return mask

    if side == "left":
        mask = np.zeros((base_h, base_w + 1), dtype=bool)
        mask[:, 1:] = True
        mask[start : start + length, 0] = True
        return mask
    if side == "right":
        mask = np.zeros((base_h, base_w + 1), dtype=bool)
        mask[:, :base_w] = True
        mask[start : start + length, base_w] = True
        return mask
    if side == "top":
        mask = np.zeros((base_h + 1, base_w), dtype=bool)
        mask[1:, :] = True
        mask[0, start : start + length] = True
        return mask
    if side == "bottom":
        mask = np.zeros((base_h + 1, base_w), dtype=bool)
        mask[:base_h, :] = True
        mask[base_h, start : start + length] = True
        return mask
    raise AssertionError(f"unknown side {side}")


def _parse_piece(mask: np.ndarray) -> dict:
    h, w = mask.shape
    candidates = []
    sides = ("left", "right", "top", "bottom")
    for side in sides:
        if side in ("left", "right"):
            orth = h
            options = [("indentation", h, w)]
            if w >= 2:
                options.append(("protrusion", h, w - 1))
        else:
            orth = w
            options = [("indentation", h, w)]
            if h >= 2:
                options.append(("protrusion", h - 1, w))
        for kind, base_h, base_w in options:
            for length in range(1, orth):
                for start in range(0, orth - length + 1):
                    cand = _build_piece_mask(base_h, base_w, side, kind, start, length)
                    if cand.shape == mask.shape and np.array_equal(cand, mask):
                        candidates.append(
                            {
                                "base_h": int(base_h),
                                "base_w": int(base_w),
                                "side": side,
                                "kind": kind,
                                "start": int(start),
                                "length": int(length),
                            }
                        )
    unique = []
    seen = set()
    for cand in candidates:
        key = (
            cand["base_h"],
            cand["base_w"],
            cand["side"],
            cand["kind"],
            cand["start"],
            cand["length"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(cand)
    assert len(unique) == 1, f"expected one piece interpretation, found {len(unique)}"
    return unique[0]


def _frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        for coords in _components(grid == color):
            mask, (r0, c0, r1, c1) = _local_mask(coords)
            h, w = mask.shape
            if h < 4 or w < 4:
                continue
            border = np.ones((h, w), dtype=bool)
            border[1:-1, 1:-1] = False
            if not np.array_equal(mask, border):
                continue
            inner = grid[r0 + 1 : r1, c0 + 1 : c1].copy()
            if inner.size == 0:
                continue
            anchor_color = int(inner[0, 0])
            if anchor_color in (bg, color):
                continue
            inner[0, 0] = bg
            if np.any(inner != bg):
                continue
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "inner_r0": int(r0 + 1),
                    "inner_c0": int(c0 + 1),
                    "inner_h": int(h - 2),
                    "inner_w": int(w - 2),
                    "color": int(color),
                    "anchor_color": anchor_color,
                }
            )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    return frames


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames = _frame_candidates(grid, bg)
    assert frames, "expected at least one frame"
    reserved = np.zeros(grid.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] |= False
        reserved[r0 : r1 + 1, c0 : c1 + 1] |= (grid[r0 : r1 + 1, c0 : c1 + 1] == frame["color"])
        reserved[frame["inner_r0"], frame["inner_c0"]] = True
    return frames, reserved


def _find_pieces(grid: np.ndarray, bg: int, reserved: np.ndarray) -> list[dict]:
    pieces = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        mask = (grid == color) & (~reserved)
        for coords in _components(mask):
            local, bbox = _local_mask(coords)
            if int(local.sum()) <= 1:
                continue
            piece = {
                "bbox": bbox,
                "coords": coords,
                "mask": local,
                "color": int(color),
            }
            piece["desc"] = _parse_piece(local)
            pieces.append(piece)
    pieces.sort(key=lambda piece: (piece["bbox"][0], piece["bbox"][1], piece["color"]))
    for idx, piece in enumerate(pieces):
        piece["id"] = idx
    return pieces


def _compose_assembly(
    piece_a: dict,
    piece_b: dict,
    out_h: int,
    out_w: int,
    offset_a: tuple[int, int],
    offset_b: tuple[int, int],
    origin_piece: dict,
    bg: int,
) -> dict | None:
    patch = np.full((out_h, out_w), bg, dtype=int)
    owner = np.full((out_h, out_w), -1, dtype=int)
    for piece, (dr, dc) in ((piece_a, offset_a), (piece_b, offset_b)):
        rr, cc = np.where(piece["mask"])
        for r, c in zip(rr.tolist(), cc.tolist()):
            pr = dr + int(r)
            pc = dc + int(c)
            if not (0 <= pr < out_h and 0 <= pc < out_w):
                return None
            if owner[pr, pc] != -1:
                return None
            owner[pr, pc] = int(piece["id"])
            patch[pr, pc] = int(piece["color"])
    if np.any(owner == -1):
        return None
    if owner[0, 0] != int(origin_piece["id"]):
        return None
    return {
        "piece_ids": tuple(sorted((int(piece_a["id"]), int(piece_b["id"])))),
        "origin_piece_id": int(origin_piece["id"]),
        "origin_color": int(origin_piece["color"]),
        "patch": patch,
        "shape": (int(out_h), int(out_w)),
    }


def _assemble_lr(left_piece: dict, right_piece: dict, bg: int) -> dict | None:
    left = left_piece["desc"]
    right = right_piece["desc"]
    if left["side"] != "right" or right["side"] != "left":
        return None
    if left["kind"] == right["kind"]:
        return None
    if left["base_h"] != right["base_h"]:
        return None
    if left["start"] != right["start"] or left["length"] != right["length"]:
        return None
    out_h = int(left["base_h"])
    out_w = int(left["base_w"] + right["base_w"])
    if left["kind"] == "protrusion":
        offset_left = (0, 0)
        offset_right = (0, int(left["base_w"]))
    else:
        offset_left = (0, 0)
        offset_right = (0, int(left["base_w"] - 1))
    return _compose_assembly(
        left_piece,
        right_piece,
        out_h,
        out_w,
        offset_left,
        offset_right,
        left_piece,
        bg,
    )


def _assemble_tb(top_piece: dict, bottom_piece: dict, bg: int) -> dict | None:
    top = top_piece["desc"]
    bottom = bottom_piece["desc"]
    if top["side"] != "bottom" or bottom["side"] != "top":
        return None
    if top["kind"] == bottom["kind"]:
        return None
    if top["base_w"] != bottom["base_w"]:
        return None
    if top["start"] != bottom["start"] or top["length"] != bottom["length"]:
        return None
    out_h = int(top["base_h"] + bottom["base_h"])
    out_w = int(top["base_w"])
    if top["kind"] == "protrusion":
        offset_top = (0, 0)
        offset_bottom = (int(top["base_h"]), 0)
    else:
        offset_top = (0, 0)
        offset_bottom = (int(top["base_h"] - 1), 0)
    return _compose_assembly(
        top_piece,
        bottom_piece,
        out_h,
        out_w,
        offset_top,
        offset_bottom,
        top_piece,
        bg,
    )


def _pair_candidates(pieces: list[dict], frames: list[dict], bg: int) -> dict[tuple[int, int], list[dict]]:
    frame_lookup: dict[tuple[int, int, int], list[int]] = {}
    for frame in frames:
        key = (int(frame["inner_h"]), int(frame["inner_w"]), int(frame["anchor_color"]))
        frame_lookup.setdefault(key, []).append(int(frame["id"]))
    candidates: dict[tuple[int, int], list[dict]] = {}
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            pair_key = (i, j)
            pair_list = []
            for cand in (
                _assemble_lr(pieces[i], pieces[j], bg),
                _assemble_lr(pieces[j], pieces[i], bg),
                _assemble_tb(pieces[i], pieces[j], bg),
                _assemble_tb(pieces[j], pieces[i], bg),
            ):
                if cand is None:
                    continue
                frame_key = (
                    int(cand["shape"][0]),
                    int(cand["shape"][1]),
                    int(cand["origin_color"]),
                )
                frame_ids = sorted(frame_lookup.get(frame_key, []))
                if not frame_ids:
                    continue
                cand["frame_ids"] = frame_ids
                pair_list.append(cand)
            if pair_list:
                unique = []
                seen = set()
                for cand in pair_list:
                    key = (
                        cand["origin_piece_id"],
                        cand["shape"],
                        cand["origin_color"],
                        tuple(cand["frame_ids"]),
                        cand["patch"].tobytes(),
                    )
                    if key not in seen:
                        seen.add(key)
                        unique.append(cand)
                candidates[pair_key] = unique
    return candidates


def _solve_assignment(pieces: list[dict], frames: list[dict], candidates: dict[tuple[int, int], list[dict]]) -> list[dict]:
    assert len(pieces) == 2 * len(frames), "expected exactly two pieces per frame"
    used_pieces: set[int] = set()
    used_frames: set[int] = set()
    answer: list[dict] = []

    def dfs() -> bool:
        if len(used_pieces) == len(pieces):
            return len(answer) == len(frames)
        first = next(idx for idx in range(len(pieces)) if idx not in used_pieces)
        for second in range(first + 1, len(pieces)):
            if second in used_pieces:
                continue
            for cand in candidates.get((first, second), []):
                for frame_id in cand["frame_ids"]:
                    if frame_id in used_frames:
                        continue
                    used_pieces.add(first)
                    used_pieces.add(second)
                    used_frames.add(frame_id)
                    answer.append({"frame_id": int(frame_id), "patch": cand["patch"]})
                    if dfs():
                        return True
                    answer.pop()
                    used_frames.remove(frame_id)
                    used_pieces.remove(second)
                    used_pieces.remove(first)
        return False

    assert dfs(), "failed to find a consistent pairing"
    return answer


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair interlocking fragments, rebuild full rectangles, and place them into
    the matching framed slots so the anchor-colored piece covers the slot's
    top-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else int(Counter(g.flat.tolist()).most_common(1)[0][0])
    frames, reserved = _find_frames(g, bg)
    pieces = _find_pieces(g, bg, reserved)
    pairings = _pair_candidates(pieces, frames, bg)
    solution = _solve_assignment(pieces, frames, pairings)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[int(r), int(c)] = bg
    for placement in solution:
        frame = frames[int(placement["frame_id"])]
        r0 = int(frame["inner_r0"])
        c0 = int(frame["inner_c0"])
        patch = placement["patch"]
        out[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = patch
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, mask: np.ndarray | None = None) -> None:
    h, w = patch.shape
    if mask is None:
        grid[r0 : r0 + h, c0 : c0 + w] = patch
        return
    rr, cc = np.where(mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        grid[r0 + int(r), c0 + int(c)] = int(patch[int(r), int(c)])


def _piece_from_desc(color: int, desc: dict) -> dict:
    mask = _build_piece_mask(
        int(desc["base_h"]),
        int(desc["base_w"]),
        str(desc["side"]),
        str(desc["kind"]),
        int(desc["start"]),
        int(desc["length"]),
    )
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return {"color": int(color), "mask": mask, "patch": patch, "desc": desc, "id": -1}


def _pair_spec(size: tuple[int, int], orientation: str, anchor_color: int, other_color: int, rng: random.Random) -> dict | None:
    h, w = size
    if orientation == "lr":
        if h < 4 or w < 4:
            return None
        left_w = rng.randint(2, w - 2)
        length = rng.randint(1, h - 2)
        start = rng.randint(1, h - length - 1)
        anchor_kind = rng.choice(("protrusion", "indentation"))
        left_desc = {
            "base_h": int(h),
            "base_w": int(left_w),
            "side": "right",
            "kind": anchor_kind,
            "start": int(start),
            "length": int(length),
        }
        right_desc = {
            "base_h": int(h),
            "base_w": int(w - left_w),
            "side": "left",
            "kind": "indentation" if anchor_kind == "protrusion" else "protrusion",
            "start": int(start),
            "length": int(length),
        }
        left_piece = _piece_from_desc(anchor_color, left_desc)
        right_piece = _piece_from_desc(other_color, right_desc)
        left_piece["id"] = 0
        right_piece["id"] = 1
        assembly = _assemble_lr(left_piece, right_piece, BG)
        if assembly is None:
            return None
        profile = ("lr", int(h), int(start), int(length))
        return {
            "profile": profile,
            "anchor_piece": left_piece,
            "other_piece": right_piece,
            "patch": assembly["patch"],
            "shape": (int(h), int(w)),
        }

    if h < 4 or w < 4:
        return None
    top_h = rng.randint(2, h - 2)
    length = rng.randint(1, w - 2)
    start = rng.randint(1, w - length - 1)
    anchor_kind = rng.choice(("protrusion", "indentation"))
    top_desc = {
        "base_h": int(top_h),
        "base_w": int(w),
        "side": "bottom",
        "kind": anchor_kind,
        "start": int(start),
        "length": int(length),
    }
    bottom_desc = {
        "base_h": int(h - top_h),
        "base_w": int(w),
        "side": "top",
        "kind": "indentation" if anchor_kind == "protrusion" else "protrusion",
        "start": int(start),
        "length": int(length),
    }
    top_piece = _piece_from_desc(anchor_color, top_desc)
    bottom_piece = _piece_from_desc(other_color, bottom_desc)
    top_piece["id"] = 0
    bottom_piece["id"] = 1
    assembly = _assemble_tb(top_piece, bottom_piece, BG)
    if assembly is None:
        return None
    profile = ("tb", int(w), int(start), int(length))
    return {
        "profile": profile,
        "anchor_piece": top_piece,
        "other_piece": bottom_piece,
        "patch": assembly["patch"],
        "shape": (int(h), int(w)),
    }


def _frame_patch(inner_h: int, inner_w: int, anchor_color: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), FRAME_COLOR, dtype=int)
    patch[1:-1, 1:-1] = BG
    patch[1, 1] = int(anchor_color)
    return patch


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int, border: int = 1) -> bool:
    H, W = occ.shape
    if r0 < border or c0 < border or r0 + h > H - border or c0 + w > W - border:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(H, r0 + h + pad)
    cc1 = min(W, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _mark_occ(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occ[r0 : r0 + h, c0 : c0 + w] = True


def _random_pos(occ: np.ndarray, h: int, w: int, pad: int, rng: random.Random) -> tuple[int, int] | None:
    H, W = occ.shape
    if h + 2 > H or w + 2 > W:
        return None
    for _ in range(180):
        r0 = rng.randint(1, H - h - 1)
        c0 = rng.randint(1, W - w - 1)
        if _can_place(occ, r0, c0, h, w, pad):
            return (int(r0), int(c0))
    for r0 in range(1, H - h):
        for c0 in range(1, W - w):
            if _can_place(occ, r0, c0, h, w, pad):
                return (int(r0), int(c0))
    return None


def _expected_output(grid: np.ndarray, pair_specs: list[dict], frame_positions: list[tuple[int, int]], piece_positions: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    piece_index = 0
    for spec in pair_specs:
        for piece in (spec["anchor_piece"], spec["other_piece"]):
            pr, pc = piece_positions[piece_index]
            rr, cc = np.where(piece["mask"])
            for r, c in zip(rr.tolist(), cc.tolist()):
                out[pr + int(r), pc + int(c)] = BG
            piece_index += 1
    for spec, (fr, fc) in zip(pair_specs, frame_positions):
        patch = spec["patch"]
        out[fr + 1 : fr + 1 + patch.shape[0], fc + 1 : fc + 1 + patch.shape[1]] = patch
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice((2, 3))
    duplicate_size_mode = rng.choice((False, True))
    sizes = []
    size_pool = [(h, w) for h in range(4, 8) for w in range(4, 8)]
    if duplicate_size_mode:
        dup_size = rng.choice(size_pool)
        sizes.extend([dup_size, dup_size])
        while len(sizes) < pair_count:
            cand = rng.choice(size_pool)
            if cand != dup_size and cand not in sizes:
                sizes.append(cand)
    else:
        sizes = rng.sample(size_pool, pair_count)
    rng.shuffle(sizes)

    colors = rng.sample(PALETTE, 2 * pair_count)
    pair_specs = []
    used_profiles = set()
    for idx in range(pair_count):
        anchor_color = int(colors[2 * idx])
        other_color = int(colors[2 * idx + 1])
        made = None
        for _ in range(80):
            orientation = rng.choice(("lr", "tb"))
            spec = _pair_spec(sizes[idx], orientation, anchor_color, other_color, rng)
            if spec is None or spec["profile"] in used_profiles:
                continue
            used_profiles.add(spec["profile"])
            made = spec
            break
        if made is None:
            return None
        pair_specs.append(made)

    canvas_h = rng.randint(CANVAS_MIN, CANVAS_MAX)
    canvas_w = rng.randint(CANVAS_MIN, CANVAS_MAX)
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)
    occ = np.zeros((canvas_h, canvas_w), dtype=bool)

    frame_positions = []
    frame_order = list(range(pair_count))
    frame_order.sort(
        key=lambda idx: (_frame_patch(pair_specs[idx]["shape"][0], pair_specs[idx]["shape"][1], pair_specs[idx]["anchor_piece"]["color"]).size),
        reverse=True,
    )
    placed_frames: dict[int, tuple[int, int]] = {}
    for idx in frame_order:
        inner_h, inner_w = pair_specs[idx]["shape"]
        patch = _frame_patch(inner_h, inner_w, pair_specs[idx]["anchor_piece"]["color"])
        pos = _random_pos(occ, patch.shape[0], patch.shape[1], pad=1, rng=rng)
        if pos is None:
            return None
        _mark_occ(occ, pos[0], pos[1], patch.shape[0], patch.shape[1])
        _paint_patch(grid, patch, pos[0], pos[1])
        placed_frames[idx] = pos
    frame_positions = [placed_frames[idx] for idx in range(pair_count)]

    piece_positions = []
    piece_list = []
    for spec in pair_specs:
        piece_list.extend([spec["anchor_piece"], spec["other_piece"]])
    piece_order = list(range(len(piece_list)))
    piece_order.sort(key=lambda idx: piece_list[idx]["patch"].size, reverse=True)
    placed_pieces: dict[int, tuple[int, int]] = {}
    for idx in piece_order:
        piece = piece_list[idx]
        pos = _random_pos(occ, piece["patch"].shape[0], piece["patch"].shape[1], pad=1, rng=rng)
        if pos is None:
            return None
        _mark_occ(occ, pos[0], pos[1], piece["patch"].shape[0], piece["patch"].shape[1])
        _paint_patch(grid, piece["patch"], pos[0], pos[1], mask=piece["mask"])
        placed_pieces[idx] = pos
    piece_positions = [placed_pieces[idx] for idx in range(len(piece_list))]

    expected = _expected_output(grid, pair_specs, frame_positions, piece_positions)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for seed in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30) + seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
