# Combo variant of 2ba387bc (index 23)
# combo_spec: D7/0+D12/2
# Name: 框架重着色+输出背景色不同

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = PROJECT_ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


FRAME_OUT_COLOR = 5
OUTPUT_BG = 0
INPUT_BGS = (2, 4, 8)


"""
Design analysis for combo D7/0 + D12/2
======================================

a) Concrete behavior and trace
   - Input stays close to the original task: several isolated 4x4 objects lie on one
     uniform input background. A frame is a monochrome 1-cell ring with a 2x2 center
     filled by the input background; a solid is a fully filled 4x4 monochrome block.
   - Original rule:
       frame color 8 on bg 4  -> output keeps an 8-ring with center 4
       solid color 3          -> output keeps a 3-solid
       output blank space     -> also color 4
   - Combo rule:
       the same detected frame is rebuilt as a canonical 5-ring with center 0
       the same solid stays color 3
       output blank space is also 0
   - The two color changes are coupled in the same place: the solver cannot paste the
     raw frame crop anymore, because both the border color and the interior background
     must be rewritten at once.

b) Complexity calibration
   - To make the recolor obvious, every generated sample has at least 2 frames.
   - To make the background swap obvious, input backgrounds are sampled from a small
     non-zero set {2, 4, 8}, while output background is always 0.
   - To avoid ambiguous "nothing changed" cases, input object colors never use 5 and
     never use the sampled input background. This gives repeated support across a few
     examples without exploding the color vocabulary.

c) Hidden-assumption audit of the original `main()`
   - `bg = get_bg(g)`: still valid, but only for parsing the input. It no longer equals
     the output background.
   - `find_components(... monochromatic=False)`: still valid if each object remains an
     isolated 4x4 connected component. The generator keeps 1-cell padding.
   - `bbox_h(b) == 4 and bbox_w(b) == 4`: still valid because the object geometry stays
     unchanged.
   - Frame detection via "uniform border + 2x2 center == input bg" still works, because
     the input representation of frames is unchanged.
   - `out = np.full(..., bg)` is no longer valid, because output background must be 0.
   - `paste(out, sub, ...)` for frames is no longer valid, because it would preserve both
     the old frame color and the old frame interior. Frames must be reconstructed from
     structure, not copied from the input crop.
   - Pasting solids directly is still valid because solids contain no background pixels.

d) Placement legality
   - Objects are still placed with `random_free_location(... padding=1, connectivity=4)`,
     so they do not overlap, do not touch, and remain separate components.
   - Every placed object is exactly 4x4, preserving the solver's bbox assumptions.

e) Visual observability
   - Across 2-4 examples, a human can see three repeated facts:
       1. frames always go to the left column and solids to the right;
       2. all left-column frames become the same color regardless of input frame color;
       3. the output background and frame interiors switch to a new fixed color.
   - These are visible pixel-level cues, not hidden numeric ordering rules.

f) Visual cleanliness
   - No extra markers or overlays are introduced.
   - The output remains a clean 8-column packed layout, with only one canonical frame
     color and one canonical output background, so the transformed scene is easier to
     read than a noisier per-object recolor scheme would be.

g) Change inferability
   - The color mapping is low-entropy and structural:
       * if an object is a frame, its output border is always 5;
       * any output blank space, including frame interiors, is always 0.
   - Because multiple differently colored frames appear on multiple non-zero input
     backgrounds, the solver can infer both rules from a small number of examples.

New assumptions introduced by the combo solver
   - The canonical frame output color 5 must differ from output background 0.
   - Input frames must not already use color 5; otherwise recoloring could become
     invisible. The generator enforces that.
   - Input background must differ from output background 0; otherwise the D12 part would
     degenerate. The generator enforces that via `INPUT_BGS`.
"""


def _is_frame(sub: np.ndarray, input_bg: int) -> bool:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    border_colors = set(map(int, border.flatten()))
    return len(border_colors) == 1 and bool((sub[1:3, 1:3] == int(input_bg)).all())


def _frame_border_color(sub: np.ndarray, input_bg: int) -> int:
    if not _is_frame(sub, input_bg):
        raise ValueError("not a frame")
    return int(sub[0, 0])


def _make_frame(color: int, bg: int) -> np.ndarray:
    tile = np.full((4, 4), int(color), dtype=int)
    tile[1:3, 1:3] = int(bg)
    return tile


def _make_solid(color: int) -> np.ndarray:
    return np.full((4, 4), int(color), dtype=int)


def _collect_objects(g: np.ndarray, input_bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(g, bg=input_bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue

        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        if _is_frame(sub, input_bg):
            frames.append((int(r0), int(c0), sub))
        else:
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _build_output(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    *,
    input_bg: int,
    output_bg: int,
    recolor_frames: bool,
) -> np.ndarray:
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(output_bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        frame_color = int(FRAME_OUT_COLOR) if recolor_frames else _frame_border_color(sub, input_bg)
        patch = _make_frame(frame_color, int(output_bg))
        paste(out, patch, int(i * 4), 0, transparent=int(output_bg))

    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - detect the same 4x4 frame/solid objects as in the original task;
    - sort each type by input row and stack frames left, solids right;
    - rebuild every output frame with canonical border color 5;
    - rebuild the whole output on canonical background 0, so frame interiors also switch.
    """
    g = np.array(grid, dtype=int)
    input_bg = int(get_bg(g))
    frames, solids = _collect_objects(g, input_bg)
    return _build_output(
        frames,
        solids,
        input_bg=input_bg,
        output_bg=int(OUTPUT_BG),
        recolor_frames=True,
    )


def _gen_attempt() -> np.ndarray | None:
    input_bg = int(np.random.choice(np.array(INPUT_BGS, dtype=int)))
    h = int(np.random.randint(20, 31))
    w = int(np.random.randint(20, 31))
    g = np.full((h, w), input_bg, dtype=int)

    n_frames = int(np.random.randint(2, 5))
    n_solids = int(np.random.randint(1, 4))
    total = n_frames + n_solids

    palette = [col for col in range(1, 10) if col not in {int(input_bg), int(OUTPUT_BG), int(FRAME_OUT_COLOR)}]
    if total > len(palette):
        return None

    try:
        np.random.shuffle(palette)
        colors = [int(col) for col in palette[:total]]
        frame_colors = colors[:n_frames]
        solid_colors = colors[n_frames:]

        for col in frame_colors:
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

        for col in solid_colors:
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

        frames, solids = _collect_objects(g, input_bg)
        if len(frames) != n_frames or len(solids) != n_solids:
            return None

        combo_out = _build_output(
            frames,
            solids,
            input_bg=input_bg,
            output_bg=int(OUTPUT_BG),
            recolor_frames=True,
        )
        original_out = _build_output(
            frames,
            solids,
            input_bg=input_bg,
            output_bg=int(input_bg),
            recolor_frames=False,
        )
        bg_only_out = _build_output(
            frames,
            solids,
            input_bg=input_bg,
            output_bg=int(OUTPUT_BG),
            recolor_frames=False,
        )
        recolor_only_out = _build_output(
            frames,
            solids,
            input_bg=input_bg,
            output_bg=int(input_bg),
            recolor_frames=True,
        )

        if np.array_equal(combo_out, original_out):
            return None
        if np.array_equal(combo_out, bg_only_out):
            return None
        if np.array_equal(combo_out, recolor_only_out):
            return None
        if input_bg == int(OUTPUT_BG):
            return None
        if any(int(sub[0, 0]) == int(FRAME_OUT_COLOR) for _r0, _c0, sub in frames):
            return None

        actual = main(g)
        if not np.array_equal(actual, combo_out):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
