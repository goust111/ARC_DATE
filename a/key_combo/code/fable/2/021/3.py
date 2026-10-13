from __future__ import annotations

import importlib.util
import random
from collections import Counter
from pathlib import Path

import numpy as np


FRAME_COLOR = 5
BACKGROUND = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
TARGET_EXTRA_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    if not ARC_UTILS_PATH.exists():
        return None
    spec = importlib.util.spec_from_file_location("_shared_arc_utils_for_fable_2_021_3", ARC_UTILS_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arc_utils = _load_arc_utils()


def _fallback_get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _fallback_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    connectivity: int = 4,
    monochromatic: bool = True,
    colors: list[int] | tuple[int, ...] | None = None,
    min_area: int = 1,
    return_format: str = "dict",
) -> list[dict] | list[np.ndarray]:
    if bg is None:
        bg = _fallback_get_bg(grid)
    bg = int(bg)
    return_format = str(return_format).lower().strip()
    assert return_format in {"dict", "objects"}
    neighbor_steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if int(connectivity) == 8:
        neighbor_steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    def components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
        seen = np.zeros_like(mask, dtype=bool)
        comps: list[np.ndarray] = []
        height, width = mask.shape
        for r in range(height):
            for c in range(width):
                if not mask[r, c] or seen[r, c]:
                    continue
                stack = [(r, c)]
                seen[r, c] = True
                coords = []
                while stack:
                    rr, cc = stack.pop()
                    coords.append((rr, cc))
                    for dr, dc in neighbor_steps:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.asarray(coords, dtype=int))
        return comps

    if colors is None:
        colors_iter = sorted(set(map(int, grid.ravel())) - {bg})
    else:
        colors_iter = sorted(set(int(color) for color in colors if int(color) != bg))

    if monochromatic:
        out_dicts: list[dict] = []
        out_objects: list[np.ndarray] = []
        for color in colors_iter:
            for coords in components_from_mask(grid == int(color)):
                if int(coords.shape[0]) < int(min_area):
                    continue
                mask = np.zeros_like(grid, dtype=bool)
                mask[coords[:, 0], coords[:, 1]] = True
                if return_format == "dict":
                    out_dicts.append(
                        {
                            "color": int(color),
                            "coords": coords,
                            "bbox": _fallback_bbox(mask),
                            "area": int(coords.shape[0]),
                        }
                    )
                else:
                    obj = np.full_like(grid, bg)
                    obj[coords[:, 0], coords[:, 1]] = int(color)
                    out_objects.append(obj)
        return out_dicts if return_format == "dict" else out_objects

    fg = (grid != bg) if colors is None else ((grid != bg) & np.isin(grid, colors_iter))
    comps = components_from_mask(fg)
    if return_format == "dict":
        out = []
        for coords in comps:
            if int(coords.shape[0]) < int(min_area):
                continue
            mask = np.zeros_like(grid, dtype=bool)
            mask[coords[:, 0], coords[:, 1]] = True
            out.append({"color": None, "coords": coords, "bbox": _fallback_bbox(mask), "area": int(coords.shape[0])})
        return out
    out = []
    for coords in comps:
        if int(coords.shape[0]) < int(min_area):
            continue
        obj = np.full_like(grid, bg)
        obj[coords[:, 0], coords[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        out.append(obj)
    return out


if _arc_utils is not None:
    bbox = _arc_utils.bbox
    get_bg = _arc_utils.get_bg
    find_components = _arc_utils.find_components
else:
    bbox = _fallback_bbox
    get_bg = _fallback_get_bg
    find_components = _fallback_find_components


def _frame_mask_from_bbox(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _find_instruction_frames(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    grey_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME_COLOR],
        return_format="dict",
    )
    assert grey_components, "expected grey instruction frames"

    frames = []
    for component in grey_components:
        box = tuple(map(int, component["bbox"]))
        r0, c0, r1, c1 = box
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        mask = np.zeros_like(grid, dtype=bool)
        coords = np.asarray(component["coords"], dtype=int)
        mask[coords[:, 0], coords[:, 1]] = True
        if not np.array_equal(mask, _frame_mask_from_bbox(grid.shape, box)):
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == FRAME_COLOR):
            continue
        colored = np.argwhere(interior != bg)
        if colored.shape[0] != 2:
            continue
        colors = [int(interior[r, c]) for r, c in colored]
        if colors[0] == colors[1]:
            continue

        frames.append(
            {
                "bbox": box,
                "swap": (colors[0], colors[1]),
                "center": ((r0 + r1) // 2, (c0 + c1) // 2),
            }
        )

    assert len(frames) == len(grey_components), "all grey components must be valid instruction frames"
    return frames


def _ordered_swaps(frames: list[dict]) -> list[tuple[int, int]]:
    assert len(frames) >= 2, "expected multiple instruction frames"
    center_rows = {frame["center"][0] for frame in frames}
    center_cols = {frame["center"][1] for frame in frames}
    same_row = len(center_rows) == 1
    same_col = len(center_cols) == 1
    assert same_row ^ same_col, "frames must align in exactly one row or one column"
    if same_row:
        ordered = sorted(frames, key=lambda frame: frame["center"][1])
    else:
        ordered = sorted(frames, key=lambda frame: frame["center"][0])
    return [tuple(map(int, frame["swap"])) for frame in ordered]


def _target_mask(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    bg = int(get_bg(grid))
    covered = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        covered[r0 : r1 + 1, c0 : c1 + 1] = True
    mask = (grid != bg) & ~covered
    components = find_components(
        np.where(mask, grid, bg),
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    assert len(components) == 1, "expected exactly one target pattern outside the frames"
    target = np.zeros_like(mask, dtype=bool)
    coords = np.asarray(components[0]["coords"], dtype=int)
    target[coords[:, 0], coords[:, 1]] = True
    return target


def _apply_swaps(values: np.ndarray, swaps: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(values, copy=True)
    for first, second in swaps:
        first = int(first)
        second = int(second)
        mask_first = out == first
        mask_second = out == second
        out[mask_first] = second
        out[mask_second] = first
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply the ordered color-swap permutation encoded by grey instruction frames to the separate target pattern.

    Input structure:
    - Grey one-cell-thick hollow rectangular frames are arranged in one row or one column on a black background.
    - Inside each frame are exactly two non-background pixels of different colors; together they encode one swap instruction.
    - A separate multicolor target pattern lies outside every frame bbox and is the only non-frame foreground object.

    Rule:
    - Interpret each frame interior as a color swap between its two colors.
    - Read the frames in spatial order: left-to-right for a row, or top-to-bottom for a column, and compose the swaps sequentially.
    - Apply the resulting permutation only to the target pattern pixels, leaving the frames, the instruction pixels, and the background unchanged.

    Key insight:
    - The frames are not editable regions; they are an ordered instruction list, so shared colors across frames make the swap order part of the meaning.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected an ARC-sized 2D grid"

    frames = _find_instruction_frames(g)
    swaps = _ordered_swaps(frames)
    target = _target_mask(g, frames)

    out = g.copy()
    out[target] = _apply_swaps(g[target], swaps)
    return out


def _make_instruction_patch(swaps: list[tuple[int, int]], horizontal: bool) -> np.ndarray:
    frame_size = 5
    gaps = [random.randint(1, 2) for _ in range(len(swaps) - 1)]
    if horizontal:
        height = frame_size
        width = len(swaps) * frame_size + sum(gaps)
    else:
        height = len(swaps) * frame_size + sum(gaps)
        width = frame_size
    patch = np.full((height, width), BACKGROUND, dtype=int)

    offset = 0
    for idx, (first, second) in enumerate(swaps):
        if horizontal:
            r0, c0 = 0, offset
        else:
            r0, c0 = offset, 0
        r1 = r0 + frame_size - 1
        c1 = c0 + frame_size - 1
        patch[r0, c0 : c1 + 1] = FRAME_COLOR
        patch[r1, c0 : c1 + 1] = FRAME_COLOR
        patch[r0 : r1 + 1, c0] = FRAME_COLOR
        patch[r0 : r1 + 1, c1] = FRAME_COLOR

        inner_slots = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        (ar, ac), (br, bc) = random.sample(inner_slots, 2)
        patch[ar, ac] = int(first)
        patch[br, bc] = int(second)

        if idx < len(gaps):
            offset += frame_size + gaps[idx]
    return patch


def _make_target_patch(colors: list[int]) -> np.ndarray:
    scale = random.choice((1, 2))
    coarse_h = random.randint(3, 5)
    coarse_w = random.randint(3, 5)
    total = coarse_h * coarse_w
    assert total >= len(colors)

    vals = list(colors)
    vals.extend(random.choices(colors, k=total - len(colors)))
    random.shuffle(vals)
    coarse = np.asarray(vals, dtype=int).reshape(coarse_h, coarse_w)
    return np.kron(coarse, np.ones((scale, scale), dtype=int))


def _place_patches(instr_patch: np.ndarray, target_patch: np.ndarray, horizontal: bool) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    instr_h, instr_w = instr_patch.shape
    target_h, target_w = target_patch.shape
    gap = random.randint(2, 4)
    margin_top = random.randint(1, 2)
    margin_bottom = random.randint(1, 2)
    margin_left = random.randint(1, 2)
    margin_right = random.randint(1, 2)

    if horizontal:
        width = max(instr_w, target_w) + margin_left + margin_right
        height = instr_h + target_h + gap + margin_top + margin_bottom
        assert width <= 30 and height <= 30
        canvas = np.full((height, width), BACKGROUND, dtype=int)
        instr_c = random.randint(margin_left, width - margin_right - instr_w)
        target_c = random.randint(margin_left, width - margin_right - target_w)
        if random.choice((False, True)):
            target_r = margin_top
            instr_r = target_r + target_h + gap
        else:
            instr_r = margin_top
            target_r = instr_r + instr_h + gap
        canvas[instr_r : instr_r + instr_h, instr_c : instr_c + instr_w] = instr_patch
        canvas[target_r : target_r + target_h, target_c : target_c + target_w] = target_patch
        target_bbox = (target_r, target_c, target_r + target_h - 1, target_c + target_w - 1)
        return canvas, target_bbox

    width = instr_w + target_w + gap + margin_left + margin_right
    height = max(instr_h, target_h) + margin_top + margin_bottom
    assert width <= 30 and height <= 30
    canvas = np.full((height, width), BACKGROUND, dtype=int)
    instr_r = random.randint(margin_top, height - margin_bottom - instr_h)
    target_r = random.randint(margin_top, height - margin_bottom - target_h)
    if random.choice((False, True)):
        target_c = margin_left
        instr_c = target_c + target_w + gap
    else:
        instr_c = margin_left
        target_c = instr_c + instr_w + gap
    canvas[instr_r : instr_r + instr_h, instr_c : instr_c + instr_w] = instr_patch
    canvas[target_r : target_r + target_h, target_c : target_c + target_w] = target_patch
    target_bbox = (target_r, target_c, target_r + target_h - 1, target_c + target_w - 1)
    return canvas, target_bbox


def _semantic_checks(target_patch: np.ndarray, swaps: list[tuple[int, int]]) -> bool:
    truth = _apply_swaps(target_patch, swaps)
    if np.array_equal(truth, target_patch):
        return False
    if np.array_equal(truth, _apply_swaps(target_patch, list(reversed(swaps)))):
        return False
    for idx in range(len(swaps)):
        partial = swaps[:idx] + swaps[idx + 1 :]
        if np.array_equal(truth, _apply_swaps(target_patch, partial)):
            return False
    return True


def _build_expected(canvas: np.ndarray, target_bbox: tuple[int, int, int, int], swaps: list[tuple[int, int]]) -> np.ndarray:
    out = canvas.copy()
    r0, c0, r1, c1 = map(int, target_bbox)
    out[r0 : r1 + 1, c0 : c1 + 1] = _apply_swaps(out[r0 : r1 + 1, c0 : c1 + 1], swaps)
    return out


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.choice((False, True))
    n_frames = random.randint(3, 4)
    chain = random.sample(PALETTE, n_frames + 1)
    swaps = [(chain[idx], chain[idx + 1]) for idx in range(n_frames)]

    remaining = [color for color in TARGET_EXTRA_COLORS if color not in chain]
    assert remaining, "target palette needs at least one unaffected color"
    target_colors = list(chain) + [random.choice(remaining)]

    target_patch = _make_target_patch(target_colors)
    if not _semantic_checks(target_patch, swaps):
        return None

    instr_patch = _make_instruction_patch(swaps, horizontal=horizontal)
    try:
        canvas, target_bbox = _place_patches(instr_patch, target_patch, horizontal=horizontal)
    except AssertionError:
        return None

    expected = _build_expected(canvas, target_bbox, swaps)
    if np.array_equal(expected, canvas):
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    detected_frames = _find_instruction_frames(canvas)
    if len(detected_frames) != n_frames:
        return None
    target_mask = _target_mask(canvas, detected_frames)
    if int(target_mask.sum()) != int(target_patch.size):
        return None
    target_box = bbox(target_mask)
    if target_box is None or tuple(map(int, target_box)) != tuple(map(int, target_bbox)):
        return None
    if int(get_bg(canvas)) != BACKGROUND:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid border-instruction permutation puzzle")
