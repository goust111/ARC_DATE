from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
MARK = 6
PIECE_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 500


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
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
                for dr, dc in steps:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _rotate_cw(arr: np.ndarray, deg: int) -> np.ndarray:
    turns = (int(deg) // 90) % 4
    if turns == 0:
        return arr.copy()
    return np.rot90(arr, k=(4 - turns))


def _first_fit_position(board: np.ndarray, ph: int, pw: int) -> tuple[int, int] | None:
    h, w = board.shape
    for r in range(h):
        for c in range(w):
            if board[r, c] != -1:
                continue
            if r + ph > h or c + pw > w:
                continue
            if np.all(board[r : r + ph, c : c + pw] == -1):
                return int(r), int(c)
    return None


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area**0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((int(h), int(w)))
        if h != w:
            pairs.append((int(w), int(h)))
    pairs.sort(key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))
    return pairs


def _infer_rotation(top_marks: int, right_marks: int, mark_count: int) -> int:
    if mark_count == 0:
        return 0
    if top_marks > right_marks:
        return 90
    if top_marks < right_marks:
        return 270
    return 180


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components_mask(g != BG, connectivity=4)
    sprites: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert np.all(patch != BG), "each sprite must be a solid rectangle apart from marker recolors"
        base_colors = sorted(int(c) for c in set(map(int, patch.ravel())) if int(c) not in (BG, MARK))
        assert len(base_colors) == 1, "each sprite must contain exactly one base color"
        base = int(base_colors[0])
        top_marks = int(np.count_nonzero(patch[0, :] == MARK))
        right_marks = int(np.count_nonzero(patch[:, -1] == MARK))
        mark_count = int(np.count_nonzero(patch == MARK))
        restored = patch.copy()
        restored[restored == MARK] = base
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "restored": restored,
                "base": base,
                "height": int(patch.shape[0]),
                "width": int(patch.shape[1]),
                "area": int(patch.shape[0] * patch.shape[1]),
                "mark_count": mark_count,
                "top_marks": top_marks,
                "right_marks": right_marks,
                "rotation": _infer_rotation(top_marks, right_marks, mark_count),
            }
        )
    sprites.sort(key=lambda s: (s["bbox"][0], s["bbox"][1]))
    return sprites


def _assemble_from_sprites(sprites: list[dict]) -> np.ndarray:
    solutions = _all_assemblies(sprites)
    assert len(solutions) == 1, "sprite assembly must yield a unique tiled rectangle"
    return solutions[0]


def _all_assemblies(sprites: list[dict]) -> list[np.ndarray]:
    assert len(sprites) == 4, "expected exactly four sprites"
    anchors = [s for s in sprites if s["mark_count"] == 0]
    assert len(anchors) == 1, "expected exactly one unmarked anchor sprite"
    anchor = anchors[0]
    others = [s for s in sprites if s is not anchor]
    others.sort(key=lambda s: (-s["area"], s["bbox"][0], s["bbox"][1]))

    rotated_anchor = _rotate_cw(anchor["restored"], anchor["rotation"])
    total_area = sum(int(s["area"]) for s in sprites)
    solutions: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for out_h, out_w in _factor_pairs(total_area):
        ah, aw = rotated_anchor.shape
        if ah > out_h or aw > out_w:
            continue
        board = np.full((out_h, out_w), -1, dtype=int)
        board[:ah, :aw] = rotated_anchor
        ok = True
        for sprite in others:
            rotated = _rotate_cw(sprite["restored"], sprite["rotation"])
            ph, pw = rotated.shape
            pos = _first_fit_position(board, ph, pw)
            if pos is None:
                ok = False
                break
            r0, c0 = pos
            board[r0 : r0 + ph, c0 : c0 + pw] = rotated
        if ok and np.all(board != -1):
            key = (board.shape, board.tobytes())
            if key not in seen:
                seen.add(key)
                solutions.append(board.copy())

    return solutions


def _assemble_with_custom_rotations(sprites: list[dict], rotations: dict[int, int]) -> np.ndarray | None:
    anchors = [s for s in sprites if s["mark_count"] == 0]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    others = [s for s in sprites if s is not anchor]
    others.sort(key=lambda s: (-s["area"], s["bbox"][0], s["bbox"][1]))
    total_area = sum(int(s["area"]) for s in sprites)
    anchor_rot = int(rotations.get(id(anchor), 0))
    rotated_anchor = _rotate_cw(anchor["restored"], anchor_rot)
    for out_h, out_w in _factor_pairs(total_area):
        ah, aw = rotated_anchor.shape
        if ah > out_h or aw > out_w:
            continue
        board = np.full((out_h, out_w), -1, dtype=int)
        board[:ah, :aw] = rotated_anchor
        ok = True
        for sprite in others:
            rotated = _rotate_cw(sprite["restored"], int(rotations.get(id(sprite), 0)))
            ph, pw = rotated.shape
            pos = _first_fit_position(board, ph, pw)
            if pos is None:
                ok = False
                break
            r0, c0 = pos
            board[r0 : r0 + ph, c0 : c0 + pw] = rotated
        if ok and np.all(board != -1):
            return board
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate count-coded rectangular sprites and assemble them into one anchored tiled rectangle.

    Input structure:
    - The background is black, and exactly four disconnected sprites are present.
    - Each sprite is a solid rectangle in one of four distinct colors, with some top-edge and right-edge pixels optionally replaced by magenta markers.
    - Exactly one sprite has no magenta markers at all; that unmarked rectangle is the anchor.

    Rule:
    - For each sprite, count magenta pixels on its top edge and on its right edge.
    - Restore every magenta marker back to the sprite's base color, then rotate the sprite by: 0 degrees if it has no markers, 90 degrees clockwise if top markers exceed right markers, 270 degrees clockwise if right markers exceed top markers, and 180 degrees if the counts are equal but nonzero.
    - Place the unmarked anchor sprite at the output's top-left corner.
    - Sort the remaining sprites by input bounding-box area from large to small, then place each rotated rectangle into the first row-major empty location where it fits so that the four sprites tile one gapless rectangle.

    Key insight:
    - The magenta edge counts are not defects: after restoring them, they encode which pieces keep orientation and which pieces must be transposed before the first-fit anchored rectangle assembly.
    """
    sprites = _extract_sprites(np.asarray(grid, dtype=int))
    return _assemble_from_sprites(sprites)


def _place_rect(owner: np.ndarray, rect_id: int, r0: int, c0: int, h: int, w: int) -> None:
    owner[r0 : r0 + h, c0 : c0 + w] = int(rect_id)


def _search_output_layout(
    out_h: int,
    out_w: int,
    anchor_hw: tuple[int, int],
    branches: list[int],
    rng: random.Random,
) -> list[dict] | None:
    owner = np.full((out_h, out_w), -1, dtype=int)
    ah, aw = anchor_hw
    _place_rect(owner, 0, 0, 0, ah, aw)

    def rec(idx: int, prev_area: int, rects: list[dict]) -> list[dict] | None:
        if idx == len(branches):
            return rects.copy() if np.all(owner != -1) else None

        remaining_slots = len(branches) - idx - 1
        empty = int(np.count_nonzero(owner == -1))
        branch = int(branches[idx])
        candidates: list[tuple[int, int]] = []
        for h in range(2, out_h + 1):
            for w in range(2, out_w + 1):
                area = int(h * w)
                if area >= prev_area:
                    continue
                if area > empty - 4 * remaining_slots:
                    continue
                if branch in (90, 270, 180) and h == w:
                    continue
                pos = _first_fit_position(owner, h, w)
                if pos is None:
                    continue
                candidates.append((h, w))
        rng.shuffle(candidates)
        candidates.sort(key=lambda hw: -(hw[0] * hw[1]))

        for h, w in candidates:
            pos = _first_fit_position(owner, h, w)
            if pos is None:
                continue
            r0, c0 = pos
            area = int(h * w)
            _place_rect(owner, idx + 1, r0, c0, h, w)
            rects.append({"id": idx + 1, "r0": r0, "c0": c0, "h": h, "w": w, "branch": branch, "area": area})
            solved = rec(idx + 1, area, rects)
            if solved is not None:
                return solved
            rects.pop()
            owner[r0 : r0 + h, c0 : c0 + w] = -1
        return None

    solved = rec(0, out_h * out_w + 1, [{"id": 0, "r0": 0, "c0": 0, "h": ah, "w": aw, "branch": 0, "area": ah * aw}])
    return solved


def _make_output_layout(rng: random.Random) -> list[dict]:
    for _ in range(300):
        out_h = rng.randint(6, 11)
        out_w = rng.randint(6, 11)
        ah = rng.randint(2, max(2, out_h - 2))
        aw = rng.randint(2, max(2, out_w - 2))
        if ah >= out_h or aw >= out_w:
            continue
        branches = [90, 180, 270]
        rng.shuffle(branches)
        solved = _search_output_layout(out_h, out_w, (ah, aw), branches, rng)
        if solved is not None:
            return solved
    raise RuntimeError("failed to construct a valid output rectangle layout")


def _render_output(layout: list[dict], colors: dict[int, int]) -> np.ndarray:
    out_h = max(rect["r0"] + rect["h"] for rect in layout)
    out_w = max(rect["c0"] + rect["w"] for rect in layout)
    out = np.zeros((out_h, out_w), dtype=int)
    for rect in layout:
        color = int(colors[int(rect["id"])])
        out[rect["r0"] : rect["r0"] + rect["h"], rect["c0"] : rect["c0"] + rect["w"]] = color
    return out


def _sample_marker_positions(h: int, w: int, branch: int, rng: random.Random) -> tuple[set[int], set[int]]:
    if branch == 90:
        top_need, right_need = 2, 1
    elif branch == 270:
        top_need, right_need = 1, 2
    elif branch == 180:
        top_need, right_need = 1, 1
    else:
        return set(), set()

    for _ in range(50):
        top_cols = set(rng.sample(range(w), top_need))
        right_rows = set(rng.sample(range(h), right_need))
        marker_mask = np.zeros((h, w), dtype=bool)
        for c in top_cols:
            marker_mask[0, c] = True
        for r in right_rows:
            marker_mask[r, w - 1] = True
        if int(np.count_nonzero(marker_mask)) < h * w:
            return top_cols, right_rows
    raise RuntimeError("failed to place markers without erasing the whole sprite")


def _encode_input_patch(final_patch: np.ndarray, branch: int, rng: random.Random) -> np.ndarray:
    plain = _rotate_cw(final_patch, (360 - int(branch)) % 360)
    h, w = plain.shape
    patch = plain.copy()
    if branch == 0:
        return patch
    top_cols, right_rows = _sample_marker_positions(h, w, branch, rng)
    for c in top_cols:
        patch[0, c] = MARK
    for r in right_rows:
        patch[r, w - 1] = MARK
    assert int(np.count_nonzero(patch == MARK)) < h * w
    return patch


def _scatter_patches(patches: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    heights = [int(p.shape[0]) for p in patches]
    widths = [int(p.shape[1]) for p in patches]
    min_h = max(max(heights) + 4, 16)
    min_w = max(max(widths) + 4, 16)
    canvas_h = rng.randint(min_h, min(24, max(min_h, sum(heights) + 6)))
    canvas_w = rng.randint(min_w, min(24, max(min_w, sum(widths) + 6)))
    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    order = list(range(len(patches)))
    rng.shuffle(order)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, canvas_h - ph):
            for c0 in range(1, canvas_w - pw):
                r1 = r0 + ph - 1
                c1 = c0 + pw - 1
                ok = True
                for br0, bc0, br1, bc1 in placed_boxes:
                    if not (r1 + 1 < br0 or r0 > br1 + 1 or c1 + 1 < bc0 or c0 > bc1 + 1):
                        ok = False
                        break
                if ok:
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        canvas[r0 : r0 + ph, c0 : c0 + pw] = patch
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return canvas


def _output_signature(arr: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return arr.shape, arr.tobytes()


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    sprites = _extract_sprites(inp)
    if len(sprites) != 4:
        return False
    if sum(int(s["mark_count"] == 0) for s in sprites) != 1:
        return False
    try:
        observed = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(observed, expected):
        return False

    branch_counts = sorted(int(s["rotation"]) for s in sprites if s["mark_count"] > 0)
    if branch_counts != [90, 180, 270]:
        return False

    # No-rotation shortcut must fail on at least one non-anchor piece.
    identity = _assemble_with_custom_rotations(sprites, {id(s): 0 for s in sprites})
    if identity is not None and np.array_equal(identity, expected):
        return False

    # Treating every marked piece as a transpose should also fail because the 180-piece is non-square.
    all_marked_quarter_turn = {id(s): (90 if s["mark_count"] > 0 else 0) for s in sprites}
    wrong_turn = _assemble_with_custom_rotations(sprites, all_marked_quarter_turn)
    if wrong_turn is not None and np.array_equal(wrong_turn, expected):
        return False

    # Assembly must stay unique under the actual rule.
    try:
        solutions = _all_assemblies(sprites)
    except AssertionError:
        return False
    return len(solutions) == 1 and np.array_equal(solutions[0], expected)


def _generate_case(rng: random.Random) -> np.ndarray:
    layout = _make_output_layout(rng)
    color_order = list(PIECE_COLORS)
    rng.shuffle(color_order)
    colors = {rect["id"]: color_order[i] for i, rect in enumerate(sorted(layout, key=lambda rect: rect["id"]))}
    expected = _render_output(layout, colors)

    patches: list[np.ndarray] = []
    for rect in sorted(layout, key=lambda item: item["id"]):
        final_patch = np.full((rect["h"], rect["w"]), int(colors[rect["id"]]), dtype=int)
        patches.append(_encode_input_patch(final_patch, int(rect["branch"]), rng))

    canvas = _scatter_patches(patches, rng)
    if canvas is None:
        raise RuntimeError("failed to scatter patches")
    if not _validate_semantics(canvas, expected):
        raise RuntimeError("semantic validation failed")
    return canvas


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            return _generate_case(rng)
        except RuntimeError:
            continue
    raise RuntimeError("generate_input: exhausted attempts")
