# Variant of 2ba387bc
# Dimension: D12 — 背景色的角色 (meta)
# Change: 背景色为最常见颜色(通常为0/黑色)，框架的内部2x2区域用背景色填充... → 输出背景色与输入背景色不同

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


OUTPUT_BG = 0
INPUT_BGS = (2, 4, 8)


# Design analysis for the D12 variant
# a) Concrete behavior:
#    Input still contains scattered 4x4 objects on a uniform background. A frame is a
#    monochrome 1-cell border with its 2x2 center filled by the input background; a solid
#    is a fully filled 4x4 block. Output still sorts frames into the left column and solids
#    into the right column, each group ordered by input row. The only rule change is that
#    output blank space does not reuse the input background. Instead, every output sample
#    uses a fixed new background color (black/0), so frame interiors must also switch from
#    the input background to this new output background.
#    Small trace: if a frame is color 6 and input background is 4, the original solver would
#    paste [[6,6,6,6],[6,4,4,6],[6,4,4,6],[6,6,6,6]] into output. The variant must emit
#    [[6,6,6,6],[6,0,0,6],[6,0,0,6],[6,6,6,6]] instead. Solids are unchanged.
# b) Complexity calibration:
#    The change is isolated to background semantics, but it must be visually obvious. To make
#    that reliable, generator samples the input background from a small non-black set
#    {2,4,8}; output background is always 0. This creates repeated support across examples
#    without introducing a hard-to-infer per-sample color mapping.
# c) Original implicit assumptions that break:
#    - `bg = get_bg(g)`: still valid, but now only for interpreting the input.
#    - `sub[1:3,1:3] == bg` for frame detection: still valid because input frames still use
#      the input background inside.
#    - `out = np.full(..., bg)`: invalid in the variant because output background is no longer
#      the input background.
#    - `paste(out, sub, ...)` for frames: invalid because it would keep the old center color.
#      The solver must reconstruct frames against the new output background rather than reusing
#      the raw 4x4 input crop.
# d) Placement legality:
#    Generator still uses `random_free_location` with transparent=input_bg, so objects never
#    overlap and remain separated by padding. Each object is exactly 4x4, which preserves the
#    solver's component-and-bbox assumptions.
# e) Visual observability:
#    Across 2-4 examples, a human can see that objects are sorted exactly as before, but the
#    background and every frame interior turn black in the output even when the input background
#    is red/yellow/teal. This is a direct, visible D12 change.
# f) Visual cleanliness:
#    No extra markers are added. The only new visual effect is the uniform output background,
#    which actually makes the packed output cleaner than the input.
# g) Color-change inferability:
#    There is only one color rule: output background is always 0. The generator prevents 0 from
#    being used as an object color, so the background role stays unambiguous.


def _make_frame(col: int, bg: int) -> np.ndarray:
    tile = np.full((4, 4), int(col), dtype=int)
    tile[1:3, 1:3] = int(bg)
    return tile


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _frame_color(sub: np.ndarray, input_bg: int) -> int:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.flatten()))
    if len(bcols) != 1:
        raise ValueError("frame border is not monochrome")
    if not bool((sub[1:3, 1:3] == int(input_bg)).all()):
        raise ValueError("frame center does not match input background")
    return int(next(iter(bcols)))


def _assemble_output(
    frames: list[tuple[int, int, int]],
    solids: list[tuple[int, int, int]],
    *,
    output_bg: int = OUTPUT_BG,
) -> np.ndarray:
    out_h = 4 * max(len(frames), len(solids))
    out = np.full((out_h, 8), int(output_bg), dtype=int)

    for i, (_r0, _c0, col) in enumerate(frames):
        paste(out, _make_frame(int(col), int(output_bg)), int(i * 4), 0)
    for i, (_r0, _c0, col) in enumerate(solids):
        paste(out, _make_solid(int(col)), int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: sort the same 4x4 frames/solids as task 2ba387bc, but rebuild the output
    on a new background color instead of reusing the input background.
    """
    g = np.array(grid, dtype=int)
    input_bg = int(get_bg(g))

    comps = find_components(g, bg=input_bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, int]] = []
    solids: list[tuple[int, int, int]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        try:
            col = _frame_color(sub, input_bg)
            frames.append((int(r0), int(c0), int(col)))
        except ValueError:
            solids.append((int(r0), int(c0), int(sub[0, 0])))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))

    return _assemble_output(frames, solids, output_bg=OUTPUT_BG)


def _gen_attempt() -> np.ndarray | None:
    input_bg = int(np.random.choice(INPUT_BGS))
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    g = np.full((h, w), input_bg, dtype=int)

    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))
    n_total = n_frames + n_solids

    palette = [col for col in range(1, 10) if col not in {OUTPUT_BG, input_bg}]
    if n_total > len(palette):
        return None

    try:
        np.random.shuffle(palette)
        colors = [int(col) for col in palette[:n_total]]
        placed_frames: list[tuple[int, int, int]] = []
        placed_solids: list[tuple[int, int, int]] = []

        for col in colors[:n_frames]:
            patch = _make_frame(col, input_bg)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=input_bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=input_bg)
            placed_frames.append((int(r0), int(c0), int(col)))

        for col in colors[n_frames:]:
            patch = _make_solid(col)
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=input_bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=input_bg)
            placed_solids.append((int(r0), int(c0), int(col)))

        expected = _assemble_output(
            sorted(placed_frames, key=lambda t: (int(t[0]), int(t[1]))),
            sorted(placed_solids, key=lambda t: (int(t[0]), int(t[1]))),
            output_bg=OUTPUT_BG,
        )

        actual = main(g)
        if not np.array_equal(actual, expected):
            raise ValueError("generator produced a grid inconsistent with main()")
        if input_bg == OUTPUT_BG:
            raise ValueError("variant degenerated to original background behavior")
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
