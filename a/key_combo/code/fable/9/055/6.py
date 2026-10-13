from __future__ import annotations

from collections import Counter
import random

import numpy as np


MAX_GENERATION_ATTEMPTS = 500
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _dominant_foreground_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, np.asarray(grid).ravel()))
    counts.pop(int(bg), None)
    if not counts:
        raise AssertionError("missing foreground")
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _parse_single_frame(grid: np.ndarray, frame_color: int, coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = patch.shape
    assert h >= 3 and w >= 5
    assert np.all(patch[0, :] == frame_color) and np.all(patch[-1, :] == frame_color)

    full_rows = [r for r in range(h) if np.all(patch[r, :] == frame_color)]
    assert full_rows == [0, h - 1]

    full_cols = [c for c in range(w) if np.all(patch[:, c] == frame_color)]
    assert full_cols[0] == 0 and full_cols[-1] == w - 1
    assert len(full_cols) >= 4

    full_col_set = set(full_cols)
    for c in range(w):
        if c in full_col_set:
            assert np.all(patch[:, c] == frame_color)
        else:
            assert np.all(patch[[0, h - 1], c] == frame_color)
            assert not np.any(patch[1:-1, c] == frame_color)

    slots = []
    for left, right in zip(full_cols, full_cols[1:]):
        assert right - left >= 2
        slots.append((c0 + left + 1, c0 + right - 1))

    return {
        "bbox": (r0, c0, r1, c1),
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "slots": slots,
    }


def _parse_frames(grid: np.ndarray, bg: int, frame_color: int) -> list[dict]:
    frames = [_parse_single_frame(grid, frame_color, coords) for coords in _components(grid == frame_color)]
    assert len(frames) >= 2
    frames.sort(key=lambda frame: (frame["c0"], frame["r0"]))
    assert len({frame["c0"] for frame in frames}) == len(frames)
    n_slots = len(frames[0]["slots"])
    assert all(len(frame["slots"]) == n_slots for frame in frames)
    return frames


def _slot_hint_color(grid: np.ndarray, frame: dict, slot_index: int, bg: int) -> int | None:
    c0, c1 = frame["slots"][slot_index]
    slot = grid[frame["r0"] + 1 : frame["r1"], c0 : c1 + 1]
    colors = sorted(set(map(int, np.unique(slot))) - {int(bg)})
    if not colors:
        return None
    assert len(colors) == 1
    return int(colors[0])


def _solve_with_order(grid: np.ndarray, frames: list[dict], bg: int) -> tuple[np.ndarray, list[int]]:
    n_slots = len(frames[0]["slots"])
    base: list[int | None] = [None] * n_slots

    for k, frame in enumerate(frames):
        for j in range(n_slots):
            color = _slot_hint_color(grid, frame, j, bg)
            if color is None:
                continue
            idx = (j + k) % n_slots
            if base[idx] is None:
                base[idx] = color
            else:
                assert base[idx] == color

    assert all(color is not None for color in base)
    solved_base = [int(color) for color in base]

    out = np.array(grid, dtype=int, copy=True)
    for k, frame in enumerate(frames):
        for j, (c0, c1) in enumerate(frame["slots"]):
            out[frame["r0"] + 1 : frame["r1"], c0 : c1 + 1] = solved_base[(j + k) % n_slots]
    return out, solved_base


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every empty lattice slot by undoing the rank-based cyclic phase shift between frames.

    Input structure:
    - The dominant color is the background.
    - The most frequent foreground color forms several one-pixel-thick single-row lattice frames.
    - All frames contain the same number of slots; some slot interiors are already filled with non-frame colors and the others remain background.

    Rule:
    - Order the frames from left to right as ranks 0, 1, 2, ...
    - There exists one base color sequence over the slot indices.
    - In frame k, slot j must use the base-sequence color at index (j + k) modulo the slot count.
    - Infer that base sequence from the visible filled slots, then fill every background slot interior accordingly while leaving the frames and outside background unchanged.

    Key insight:
    - Matching same-position slots across frames is wrong because each frame is phase-shifted by its left-to-right rank; the ranks must be aligned before colors can be transferred.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color = _dominant_foreground_color(g, bg)
    frames = _parse_frames(g, bg, frame_color)
    out, _ = _solve_with_order(g, frames, bg)
    return out


def _frame_separators(slot_widths: list[int]) -> list[int]:
    pos = 0
    separators = [0]
    for width in slot_widths:
        pos += width + 1
        separators.append(pos)
    return separators


def _render_frame_patch(
    slot_widths: list[int],
    inner_h: int,
    frame_color: int,
    slot_colors: list[int] | None,
    bg: int,
    reveal: list[bool] | None = None,
) -> np.ndarray:
    n_slots = len(slot_widths)
    separators = _frame_separators(slot_widths)
    height = inner_h + 2
    width = separators[-1] + 1
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, separators] = frame_color

    if slot_colors is None:
        return patch

    for j, width_j in enumerate(slot_widths):
        if reveal is not None and not reveal[j]:
            continue
        left = separators[j] + 1
        right = left + width_j
        patch[1:-1, left:right] = int(slot_colors[j])
    return patch


def _count_combo() -> tuple[int, int]:
    roll = random.random()
    if roll < 0.33:
        return 3, random.choice([3, 4])
    if roll < 0.7:
        return 4, random.choice([2, 3])
    return 5, random.choice([2, 3])


def _frame_specs(n_slots: int, n_frames: int) -> list[dict]:
    specs = []
    max_slot_w = 2 if n_frames >= 3 or n_slots >= 5 else 3
    for _ in range(n_frames):
        slot_widths = [random.randint(1, max_slot_w) for _ in range(n_slots)]
        inner_h = random.randint(1, 3 if n_frames >= 3 else 4)
        specs.append(
            {
                "slot_widths": slot_widths,
                "inner_h": inner_h,
                "height": inner_h + 2,
                "width": sum(slot_widths) + n_slots + 1,
            }
        )
    if len({tuple(spec["slot_widths"]) for spec in specs}) == 1:
        idx = random.randrange(n_frames)
        j = random.randrange(n_slots)
        specs[idx]["slot_widths"][j] = 1 if specs[idx]["slot_widths"][j] > 1 else 2
        specs[idx]["width"] = sum(specs[idx]["slot_widths"]) + n_slots + 1
    return specs


def _place_frames(specs: list[dict]) -> tuple[tuple[int, int], list[dict]] | None:
    heights = [spec["height"] for spec in specs]
    widths = [spec["width"] for spec in specs]
    min_h = max(max(heights) + 4, 12)
    min_w = max(max(widths) + 4, 14)
    max_h = 30
    max_w = 30

    for _ in range(120):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        occupied = np.zeros((h, w), dtype=bool)
        placed = []
        order = list(range(len(specs)))
        order.sort(key=lambda idx: specs[idx]["height"] * specs[idx]["width"], reverse=True)
        ok = True
        for idx in order:
            spec = specs[idx]
            candidates = []
            for r0 in range(1, h - spec["height"]):
                for c0 in range(1, w - spec["width"]):
                    r1 = r0 + spec["height"] - 1
                    c1 = c0 + spec["width"] - 1
                    rr0 = max(0, r0 - 1)
                    cc0 = max(0, c0 - 1)
                    rr1 = min(h, r1 + 2)
                    cc1 = min(w, c1 + 2)
                    if occupied[rr0:rr1, cc0:cc1].any():
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                ok = False
                break
            r0, c0 = random.choice(candidates)
            spec_copy = dict(spec)
            spec_copy["r0"] = r0
            spec_copy["c0"] = c0
            placed.append(spec_copy)
            occupied[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = True
        if not ok:
            continue
        placed.sort(key=lambda spec: (spec["c0"], spec["r0"]))
        c0s = [spec["c0"] for spec in placed]
        if len(set(c0s)) != len(c0s):
            continue
        row_order = sorted(range(len(placed)), key=lambda idx: (placed[idx]["r0"], placed[idx]["c0"]))
        if row_order == list(range(len(placed))):
            continue
        return (h, w), placed
    return None


def _build_reveal_mask(n_frames: int, n_slots: int) -> np.ndarray | None:
    for _ in range(200):
        reveal = np.zeros((n_frames, n_slots), dtype=bool)
        if n_frames > 2 and random.random() < 0.6:
            active_count = random.randint(2, n_frames - 1)
        else:
            active_count = random.randint(2, n_frames)
        active = sorted(random.sample(range(n_frames), active_count))

        special_pairs = [(a, b) for i, a in enumerate(active) for b in active[i + 1 :] if a % n_slots != b % n_slots]
        if not special_pairs:
            continue
        k1, k2 = random.choice(special_pairs)
        j_star = random.randrange(n_slots)
        reveal[k1, j_star] = True
        reveal[k2, j_star] = True
        counts = reveal.sum(axis=1)

        success = True
        for base_idx in random.sample(list(range(n_slots)), n_slots):
            if any(reveal[k, (base_idx - k) % n_slots] for k in range(n_frames)):
                continue
            options = []
            for k in active:
                j = (base_idx - k) % n_slots
                if reveal[k, j] or counts[k] >= n_slots - 1:
                    continue
                options.append((int(counts[k]), random.random(), k, j))
            if not options:
                success = False
                break
            _, _, k, j = min(options)
            reveal[k, j] = True
            counts[k] += 1
        if not success:
            continue

        extra_target = random.randint(1, max(2, n_frames))
        extra_cells = [(k, j) for k in active for j in range(n_slots) if not reveal[k, j]]
        random.shuffle(extra_cells)
        for k, j in extra_cells:
            if extra_target <= 0:
                break
            if counts[k] >= n_slots - 1:
                continue
            reveal[k, j] = True
            counts[k] += 1
            extra_target -= 1

        if any(not any(reveal[k, (base_idx - k) % n_slots] for k in range(n_frames)) for base_idx in range(n_slots)):
            continue
        if len({k for k in range(n_frames) if reveal[k].any()}) < 2:
            continue
        if not any(reveal[:, j].sum() >= 2 for j in range(n_slots)):
            continue
        if not np.any(counts <= n_slots - 2):
            continue
        return reveal
    return None


def _render_pair(shape: tuple[int, int], placed: list[dict], base_colors: list[int], reveal: np.ndarray, bg: int, frame_color: int) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, bg, dtype=int)
    out = np.full(shape, bg, dtype=int)
    for k, spec in enumerate(placed):
        slot_colors = [base_colors[(j + k) % len(base_colors)] for j in range(len(base_colors))]
        patch_out = _render_frame_patch(spec["slot_widths"], spec["inner_h"], frame_color, slot_colors, bg)
        patch_in = _render_frame_patch(
            spec["slot_widths"],
            spec["inner_h"],
            frame_color,
            slot_colors,
            bg,
            reveal=list(map(bool, reveal[k])),
        )
        r0, c0 = spec["r0"], spec["c0"]
        out[r0 : r0 + patch_out.shape[0], c0 : c0 + patch_out.shape[1]] = patch_out
        inp[r0 : r0 + patch_in.shape[0], c0 : c0 + patch_in.shape[1]] = patch_in
    return inp, out


def _same_slot_shortcut(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color = _dominant_foreground_color(g, bg)
    frames = _parse_frames(g, bg, frame_color)
    n_slots = len(frames[0]["slots"])
    by_slot: list[int | None] = [None] * n_slots
    for j in range(n_slots):
        for frame in frames:
            color = _slot_hint_color(g, frame, j, bg)
            if color is not None:
                by_slot[j] = color
                break
    out = g.copy()
    for frame in frames:
        for j, (c0, c1) in enumerate(frame["slots"]):
            if by_slot[j] is not None:
                out[frame["r0"] + 1 : frame["r1"], c0 : c1 + 1] = int(by_slot[j])
    return out


def _row_rank_shortcut_matches(grid: np.ndarray, expected: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color = _dominant_foreground_color(g, bg)
    frames = _parse_frames(g, bg, frame_color)
    left_order = [(frame["c0"], frame["r0"]) for frame in frames]
    row_frames = sorted(frames, key=lambda frame: (frame["r0"], frame["c0"]))
    row_order = [(frame["c0"], frame["r0"]) for frame in row_frames]
    if row_order == left_order:
        return True
    try:
        actual, _ = _solve_with_order(g, row_frames, bg)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _semantic_checks(inp: np.ndarray, out: np.ndarray, reveal: np.ndarray, base_colors: list[int], frame_color: int, bg: int) -> bool:
    if int(_get_bg(inp)) != int(bg):
        return False
    if int(_dominant_foreground_color(inp, bg)) != int(frame_color):
        return False

    fg_counts = Counter(map(int, inp.ravel()))
    frame_pixels = fg_counts[int(frame_color)]
    for color in base_colors:
        if fg_counts[int(color)] >= frame_pixels:
            return False

    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    if np.array_equal(_same_slot_shortcut(inp), out):
        return False
    if _row_rank_shortcut_matches(inp, out):
        return False

    if any(row.all() for row in reveal):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_slots, n_frames = _count_combo()
    palette = list(PALETTE)
    bg = random.choice(palette)
    palette.remove(bg)
    random.shuffle(palette)
    frame_color = palette[0]
    base_colors = palette[1 : 1 + n_slots]
    if len(base_colors) != n_slots:
        return None

    specs = _frame_specs(n_slots, n_frames)
    placement = _place_frames(specs)
    if placement is None:
        return None
    shape, placed = placement
    reveal = _build_reveal_mask(n_frames, n_slots)
    if reveal is None:
        return None

    inp, out = _render_pair(shape, placed, base_colors, reveal, bg, frame_color)
    if not _semantic_checks(inp, out, reveal, base_colors, frame_color, bg):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid lattice-phase input")
